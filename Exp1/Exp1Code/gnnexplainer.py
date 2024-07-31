import argparse
import torch
from dgl.nn import GNNExplainer  # 使用 GNNExplainer
import pickle
import os
import time
import dgl
import sys
from network import get_sag_network
from concurrent.futures import ThreadPoolExecutor, as_completed

from module import *

# def create_dgl_graph(node_features, adjacency_matrix,edge_weights=None):
#     """
#     从SciPy稀疏矩阵创建DGL图，并确保图在指定的设备上。
#     """
#     # 确保 adjacency_matrix 是 scipy.sparse 矩阵
#     if not sp.isspmatrix(adjacency_matrix):
#         adjacency_matrix = sp.csr_matrix(adjacency_matrix)

#     # 创建图
#     g = dgl.from_scipy(adjacency_matrix, idtype=torch.int64)

#     # 将节点特征转换为 Tensor 并赋值给图的节点数据，确保它们在 GPU 上
#     g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)

#     if edge_weights is not None:
#         # 假设edge_weights是一个一维numpy数组，每个元素对应一条边的权重
#         g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)


#     return g
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

# def main(folder,model_path, size, dst_folder):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     n_classes = 2
#     hidden_dim = 128  # 使用指令行参数指定
#     # folder_path = sample_folder
#     # file_list = sorted(os.listdir(folder_path), key=extract_number)
#     log_folder = '../log/'

#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)
#     if not os.path.exists(log_folder):
#         os.makedirs(log_folder)

#     for file in os.listdir(folder):
#         if file.endswith('.pickle'):
#             file_path = os.path.join(folder, file)
#             print(file_path)
#             try:
#                 node_features, _, adjacency_matrix, _ = load_data_from_pickle(file_path)
#                 # 修改因為bert之後錯誤append
#                 # 之後可以刪掉 因為已經修改bert_embedding.py
#                 print(len(node_features[0]))
#                 graph= create_dgl_graph(node_features, adjacency_matrix)
#                 in_dim = len(node_features[0])
#                 graph = graph.to(device)
#                 # print(in_dim)
#                 model_op = get_sag_network('hierarchical')
#                 model = model_op(
#                     in_dim=in_dim,
#                     hid_dim=hidden_dim,
#                     out_dim=n_classes,
#                     num_convs=4,
#                     pool_ratio=0.5,
#                     dropout=0.5,
#                 ).to(device)
#                 model_state = torch.load(model_path, map_location=device)
#                 model.load_state_dict(model_state)
#                 # print(model)
#                 model.eval()
#                 explainer = GNNExplainer(model, num_hops=size)
#                 start_time = time.time()
#                 feat_mask, edge_mask = explainer.explain_graph(graph, graph.ndata['feat'])
#                 base_filename = os.path.splitext(file)[0]
#                 new_filename = f"{base_filename}_explained.pickle"
#                 save_path = os.path.join(dst_folder, new_filename)
#                 with open(save_path, 'wb') as f:
#                     pickle.dump((feat_mask, edge_mask), f)
#                 end_time = time.time()
#                 elapsed_time = end_time - start_time
#                 print('Store in:', save_path)
#                 print('Time elapsed:', elapsed_time, 'seconds')
#             except IndexError as e:
#                 # 当遇到IndexError时，打印错误信息并跳过当前文件
#                 print(f"Error processing {file_path}: {e}, skipping...")
#             except Exception as e:
#                 # 可以捕获其他类型的异常，并根据需要进行处理
#                 print(f"Unexpected error processing {file_path}: {e}, skipping...")

def generate_unique_filename(base_filename, dst_folder):
    counter = 1
    new_filename = f"{base_filename}_explained.pickle"
    save_path = os.path.join(dst_folder, new_filename)
    while os.path.exists(save_path):
        new_filename = f"{base_filename}_explained_{counter}.pickle"
        save_path = os.path.join(dst_folder, new_filename)
        counter += 1
    return save_path

def process_file(file_path, model_path, size, dst_folder, device, hidden_dim, n_classes):
    try:
        node_features, _, adjacency_matrix, _ = load_data_from_pickle(file_path)
        print(len(node_features[0]))
        graph = create_dgl_graph(node_features, adjacency_matrix)
        in_dim = len(node_features[0])
        graph = graph.to(device)

        model_op = get_sag_network('hierarchical')
        model = model_op(
            in_dim=in_dim,
            hid_dim=hidden_dim,
            out_dim=n_classes,
            num_convs=4,
            pool_ratio=0.8,
            dropout=0.5,
        ).to(device)
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        model.eval()

        explainer = GNNExplainer(model, num_hops=size)
        start_time = time.time()
        feat_mask, edge_mask = explainer.explain_graph(graph, graph.ndata['feat'])
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        save_path = generate_unique_filename(base_filename, dst_folder)
        with open(save_path, 'wb') as f:
            pickle.dump((feat_mask, edge_mask), f)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Store in:', save_path)
        print('Time elapsed:', elapsed_time, 'seconds')
    except IndexError as e:
        print(f"Error processing {file_path}: {e}, skipping...")
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}, skipping...")

def main(folder, model_path, size, dst_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 2
    hidden_dim = 128
    # log_folder = '../log/'

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    # if not os.path.exists(log_folder):
    #     os.makedirs(log_folder)

    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.pickle')]
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_file, file_path, model_path, size, dst_folder, device, hidden_dim, n_classes) for file_path in files]
        for future in as_completed(futures):
            future.result()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir', type=str, help='Path to the sample folder', default=None)
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--size', type=int, required=True, help='Explain size')
    parser.add_argument('--Xai_dst', type=str, required=True, help='Destination folder for explanations')
    args = parser.parse_args()
    main(args.dir,args.model_path, args.size, args.Xai_dst)