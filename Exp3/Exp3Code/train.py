import argparse
import json
import logging
import os
from time import time

import dgl
import scipy.sparse as sp

import torch
import torch.nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from network import get_sag_network
from torch.utils.data import random_split
from utils import get_stats

from dgl.data import DGLDataset
# import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score
import numpy as np
from itertools import product
import random

from module import load_data_from_pickle

def setup_logging(log_file_path):
    logging.basicConfig(filename=log_file_path,
                        filemode='a',  # 'a'表示追加模式，'w'表示覆写模式
                        level=logging.INFO,  # 可以根据需要调整日志级别
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def stratified_split(graphs, labels,random_state, test_size=0.1):
    """
    根据标签进行分层抽样，确保每个集合中都有相同比例的标签。
    """
    # 先分出测试集
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        graphs, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    # 再从剩余的训练集中分出验证集
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_graphs, train_labels, test_size=test_size / (1 - test_size), stratify=train_labels, random_state=random_state
    )
    return train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels


def create_dgl_graph(node_features, adjacency_matrix, edge_weights=None):
    """
    Create a DGL graph from a SciPy sparse matrix, ensuring the graph is on the specified device.
    """
    if not sp.isspmatrix(adjacency_matrix):
        adjacency_matrix = sp.csr_matrix(adjacency_matrix)
    g = dgl.from_scipy(adjacency_matrix, idtype=torch.int64)
    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)

    # Set all edge weights to 1 if not provided
    if edge_weights is None:
        edge_weights = np.ones(adjacency_matrix.nnz, dtype=np.float32)
    g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)

    return g

class CustomGraphDataset(DGLDataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        super().__init__(name='custom_graph_dataset')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def load_and_process_data(data_dir):
    all_graphs = []
    all_labels = []
    feature_dim = 0

    # 遍历data_dir下的所有子目录
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.pickle'):
                file_path = os.path.join(root, file)
                node_features, _, adjacency_matrix, label = load_data_from_pickle(file_path)
                if node_features is not None and len(node_features) > 0 and adjacency_matrix is not None:
                    # 修改因為bert之後錯誤append
                    # 之後可以刪掉 因為已經修改bert_embedding.py
                    # if len(node_features[0])<10:
                    #     processed_node_features = [features[2] for features in node_features]
                        
                    #     graph= create_dgl_graph(processed_node_features, adjacency_matrix)
                    # else:
                    graph= create_dgl_graph(node_features, adjacency_matrix)   
                    all_graphs.append(graph)
                    all_labels.append(label)
                    print(f"Loaded {file_path}, label: {label}")
                    if feature_dim == 0:
                        feature_dim = graph.ndata['feat'].shape[1]

    return all_graphs, all_labels, feature_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Attention Graph Pooling")
    
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay"
    )
    parser.add_argument(
        "--pool_ratio", type=float, default=0.6, help="pooling ratio"
    )
    parser.add_argument("--hid_dim", type=int, default=128, help="hidden size")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout ratio"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="max number of training epochs",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="patience for early stopping"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="DGCNDroid",
        choices=["hierarchical", "DGCNDroid"],
        help="model architecture",
    )
    parser.add_argument(
        "--conv_layers", type=int, default=2, help="number of conv layers"
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="print trainlog every k epochs, -1 for silent training",
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="number of trials"
    )
    parser.add_argument("--output_path", type=str, default="./output")

    parser.add_argument("--dir", type=str,required=True, help="Paths to Dataset folders")
    parser.add_argument("--model_dst", type=str, default="../../Model_Data/best_model.pth", help="Destination path to save the best model")

    args = parser.parse_args()

    # print every
    if args.print_every == -1:
        args.print_every = args.epochs + 1

    # paths

    return args


def train(model: torch.nn.Module, optimizer, trainloader, device):
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata['feat'])
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    all_preds = []  # 用于存储所有预测结果
    all_labels = []  # 用于存储所有真实标签，以便于后续计算其他指标
    for batch in loader:
        batch_graphs, batch_labels= batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata['feat'])
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
        all_preds.extend(pred.cpu().tolist())  # 收集预测结果
        all_labels.extend(batch_labels.cpu().tolist())  # 收集真实标签
    return correct / num_graphs, loss / num_graphs, all_preds, all_labels

def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # dataset = LegacyTUDataset(args.dataset, raw_dir=args.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # # 加载并处理数据

    # hid_dim = 128
    # pool_ratio = 0.5
    # args.hid_dim = hid_dim
    # args.pool_ratio = pool_ratio
    

    # 加载并处理数据
    dataset_name = os.path.basename(args.dir.strip('/'))
    # # 构造日志文件的名称，这次使用实际的数据大小
    # log_file_name = "DataSet={}_Hidden={}_Arch={}_Pool={}_WeightDecay={}_Lr={}.log".format(
    #     dataset_name,
    #     args.hid_dim,
    #     args.architecture,
    #     args.pool_ratio,
    #     args.weight_decay,
    #     args.lr,
    # )
    best_results = {
        'best_f1': 0,
        'best_precision': 0,
        'best_accuracy': 0,
        'best_recall': 0,
        'best_train_time': float('inf'),
        'best_parameters': None
    }

    # 检查日志文件所在的目录是否存在，如果不存在则创建
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)



    all_graphs, all_labels, feature_dim = load_and_process_data(args.dir)

    for i in range(args.num_trials):
        rs = random.randint(1,100)

        
        train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels = stratified_split(all_graphs, all_labels, rs ,test_size=0.1)

        train_dataset = CustomGraphDataset(train_graphs, train_labels)
        val_dataset = CustomGraphDataset(val_graphs, val_labels)
        test_dataset = CustomGraphDataset(test_graphs, test_labels)

        train_loader = GraphDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
        )
        val_loader = GraphDataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=2
        )
        test_loader = GraphDataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=2
        )

    
        print(f"Training with Seed={rs},hid_dim={args.hid_dim}, pool_ratio={args.pool_ratio}, lr={args.lr}")
        
        log_file_name = "DataSet={}_Hidden={}_Arch={}_Pool={}_WeightDecay={}_Lr={}.log".format(
            dataset_name,
            args.hid_dim,
            args.architecture,
            args.pool_ratio,
            args.weight_decay,
            args.lr,
        )
        log_file_path = os.path.join(args.output_path, log_file_name)
        setup_logging(log_file_path)
        logging.info(f"rs= {rs}")
        # Step 2: Create model =================================================================== #
        # num_feature, num_classes, _ = dataset.statistics()
        num_feature = feature_dim
        num_classes = 2
        model_op = get_sag_network(args.architecture)
        model = model_op(
            in_dim=num_feature,
            hid_dim=args.hid_dim,
            out_dim=num_classes,
            num_convs=args.conv_layers,
            pool_ratio=args.pool_ratio,
            dropout=args.dropout,
        ).to(device)
        args.num_feature = int(num_feature)
        args.num_classes = int(num_classes)

        # Step 3: Create training components ===================================================== #
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Step 4: training epoches =============================================================== #
        bad_count = 0
        best_val_loss = float("inf")
        best_epoch = 0
        train_times = []
        for e in range(args.epochs):
            s_time = time()
            train_loss = train(model, optimizer, train_loader, device)
            print("Epoch {} training time: {:.4f}".format(e + 1, time() - s_time))
            train_times.append(time() - s_time)
            logging.info("Epoch {} training time: {:.4f}".format(e + 1, time() - s_time))
            val_acc, val_loss, _, _ = test(model, val_loader, device)

            val_loss = round(val_loss, 4)

            # test_acc, _ , preds, labels = test(model, test_loader, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_count = 0
                best_epoch = e + 1
                # Save model if it's the best so far
                best_model_path = os.path.join(args.model_dst, f"best_model_in_{e + 1}.pth")
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Saved best model at epoch {e + 1}")
            else:
                bad_count += 1
                if bad_count >= args.patience:
                    print(f"Early stopping triggered, epoch {e + 1}")
                    logging.info(f"Early stopping triggered, epoch {e + 1}")
                    break

            if (e + 1) % args.print_every == 0:
                log_format = "Epoch {}: Train Loss={:.4f}, Val Acc={:.4f}, Val Loss={:.4f}"
                print(log_format.format(e + 1, train_loss, val_acc, val_loss))
                logging.info(log_format.format(e + 1, train_loss, val_acc, val_loss))

        model.load_state_dict(torch.load(best_model_path))
        print(f"Best model loaded from epoch {best_epoch}")
        total_train_time = sum(train_times)
        _,_, preds,labels = test(model, test_loader, device)
        f1 = f1_score(labels, preds,average='macro')
        precision = precision_score(labels, preds, average='macro')
        accuracy = accuracy_score(labels, preds)
        recall = recall_score(labels, preds, average='macro')



        print(f"Test F1-Score: {f1:.4f}")
        print(f"Test Precision : {precision:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Train Time: {total_train_time:.4f}")

        logging.info(f"Test F1-Score: {f1:.4f}")
        logging.info(f"Test Precision : {precision:.4f}")
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        logging.info(f"Test Recall: {recall:.4f}")
        logging.info(f"Train Time: {total_train_time:.4f}")

        if f1 > best_results['best_f1']:
            best_results.update({
                'best_f1': f1,
                'best_precision': precision,
                'best_accuracy': accuracy,
                'best_recall': recall,
                'best_train_time': total_train_time,
                'best_parameters': {
                    'hid_dim': args.hid_dim,
                    'pool_ratio': args.pool_ratio,
                    'lr': args.lr
                }
            })

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 所有試驗結束後打印最佳結果
    print("Best Results Achieved:")
    print(f"F1-Score: {best_results['best_f1']:.4f}")
    print(f"Precision: {best_results['best_precision']:.4f}")
    print(f"Accuracy: {best_results['best_accuracy']:.4f}")
    print(f"Recall: {best_results['best_recall']:.4f}")
    print(f"Training Time: {best_results['best_train_time']:.4f}")
    print("Best Hyperparameters:")
    print(f"Hidden Dimension: {best_results['best_parameters']['hid_dim']}")
    print(f"Pooling Ratio: {best_results['best_parameters']['pool_ratio']}")
    print(f"Learning Rate: {best_results['best_parameters']['lr']}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    #     os.makedirs(args.output_path, exist_ok=True)
    # out_dict = {
    #     "hyper-parameters": vars(args),
    #     "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
    #     "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
    # # }

    # with open(args.output_path, "w") as f:
    #     json.dump(out_dict, f, sort_keys=True, indent=4)
