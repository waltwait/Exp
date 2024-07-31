import os
import pickle
import dgl
import argparse
import json
import torch
import scipy.sparse as sp
from module import load_data_from_pickle, create_dgl_graph, clean_api_name
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


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

def extract_class_path(api_name):
    # 查找类路径和方法分隔符的位置
    separator_index = api_name.find('(')
    if separator_index == -1:
        print("No valid separator found.")
        return None
    
    # 提取类路径部分
    class_path = api_name[:separator_index]
    return class_path

def find_original_file(explained_file_name, malware_directory):
    base_name = os.path.basename(explained_file_name).replace('_explained', '')
    for root, _, files in os.walk(malware_directory):
        for file in files:
            if file.startswith(base_name):
                return os.path.join(root, file)
    return None

# def calculate_api_scores(apiCounts, totalNodes):
#     apiScores = {api: count / totalNodes for api, count in apiCounts.items()}
#     return apiScores

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
def fix_reflection(api_name):
    cleaned_api_name = re.sub(r"\$\d", "", api_name)
    return cleaned_api_name

def extract_class_path(api_name):
    # 查找类路径和方法分隔符的位置
    separator_index = api_name.find('(')
    if separator_index == -1:
        print("No valid separator found.")
        return None
    
    # 提取类路径部分
    class_path = api_name[:separator_index]
    return class_path

def find_original_file(explained_file_name, malware_directories):
    base_name = os.path.basename(explained_file_name).replace('_explained', '')
    for directory in malware_directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith(base_name):
                    return os.path.join(root, file)
    return None

def calculate_api_scores(apiCounts, totalNodes):
    apiScores = {api: count / totalNodes for api, count in apiCounts.items()}
    return apiScores

def process_explained_files(explained_directories, malware_directories, dst_api, percentiles):
    
    api_data = {}
    # label_data = {}

    # totalNodes = 0
    for directory in explained_directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".pickle"):
                    explained_file_path = os.path.join(root, file)
                    original_file_path = find_original_file(explained_file_path, malware_directories)
                    print(f"Processing {explained_file_path} and {original_file_path}")
                    sample_name = os.path.splitext(os.path.basename(file))[0]
                    
                    if original_file_path:
                        node_features, api_names, adjacency_matrix, label = load_data_from_pickle(original_file_path)
                        if label != 0:
                            label = 1

                        # label_data[sample_name] = label
                        
                        if node_features is not None and adjacency_matrix is not None:
                            if len(node_features[0])<10:
                                processed_node_features = [features[2] for features in node_features]
                                dgl_graph= create_dgl_graph(processed_node_features, adjacency_matrix)
                            else:
                                dgl_graph= create_dgl_graph(node_features, adjacency_matrix)
                            
                            with open(explained_file_path, 'rb') as f:
                                _, edge_mask = pickle.load(f)

                            edge_weights = edge_mask.cpu().numpy()
                            threshold = np.percentile(edge_weights, 100 - percentiles)

                            # print(edge_mask)
                            important_edges = (edge_mask > threshold).nonzero(as_tuple=True)[0]
                            subgraph = dgl.edge_subgraph(dgl_graph, important_edges.cpu().numpy())
                            original_indices = subgraph.ndata[dgl.NID].cpu().numpy()
                            explained_api_names = [api_names[idx] for idx in original_indices]

                            # Accumulate all nodes and their connections
                            for subgraph_idx, api_name in enumerate(explained_api_names):
                                api_name = fix_reflection(api_name)
                                api_class_path = extract_class_path(api_name)
                                
                                successors = len(list(subgraph.successors(subgraph_idx)))
                                predecessors = len(list(subgraph.predecessors(subgraph_idx)))
                                total_connections = successors + predecessors+1
                                if api_class_path not in api_data:
                                    api_data[api_class_path] = {}
                                api_data[api_class_path][sample_name] = api_data[api_class_path].get(sample_name, 0) + total_connections
    
    # 在所有数据处理完毕后统一计算 API 分数
    # api_scores = calculate_api_scores(apiCounts, totalNodes)
    
    # 将API数据存储到JSON文件
    with open(dst_api, 'w') as json_file:
        json.dump(api_data, json_file, indent=4)
    
    # # 将标签数据存储到另一个JSON文件
    # with open(dst_labels, 'w') as json_file:
    #     json.dump(label_data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process explained graphs and calculate API scores based on all nodes.")
    parser.add_argument("--dir", required=True, nargs='+', help="Path to the malware original files directories")
    parser.add_argument("--Xdir", required=True, nargs='+', help="Path to the explained graph files directories")
    parser.add_argument("--api_dst", required=True, help="Destination path for the output API data JSON file")
    # parser.add_argument("--label_dst", required=True, help="Destination path for the output label data JSON file")
    parser.add_argument("--percent", type=int, default=20, help="Percentiles for selecting important edges")
    
    args = parser.parse_args()
    
    process_explained_files(args.Xdir, args.dir, args.api_dst, args.percent)
