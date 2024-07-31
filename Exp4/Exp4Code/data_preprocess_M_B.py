import networkx as nx
import os
import numpy as np
import pickle
import argparse
import sys
import json

from module import clean_api_name

def extract_class_path(api_name):
    # 查找类路径和方法分隔符的位置
    separator_index = api_name.find(';')
    if separator_index == -1:
        print("No valid separator found.")
        return None
    
    # 提取类路径部分
    class_path = api_name[:separator_index]
    return class_path

def load_graphs_generator(folder):
    files = [file for file in os.listdir(folder) if file.endswith('.gexf')]
    for file in files:
        path = os.path.join(folder, file)
        graph = nx.read_gexf(path)
        yield graph,file

def process_graphs_with_features(graph, packages):
    # 筛选包含特定包的节点
    selected_nodes = []
    for node in graph.nodes():
        node_label = graph.nodes[node].get('label', "")
        class_path = extract_class_path(node_label)  # 获取类路径
        if class_path and any(pkg in class_path for pkg in packages):  # 检查类路径中是否含有任一指定包名
            selected_nodes.append(node)
    # selected_nodes = [node for node in graph.nodes() if graph.nodes[node].get('label', "") in packages]
    if len(selected_nodes) == 0:
        return None, None, None
    
    # 找到这些节点直接相连的节点
    temp_nodes = []  # 创建一个临时列表来存储邻居节点
    for node in selected_nodes:
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))
        temp_nodes.extend(predecessors)
        temp_nodes.extend(successors) 
        # print(f"Node {node} has {len(successors)+len(predecessors)} neighbors.")
    
    selected_nodes.extend(temp_nodes)  # 將鄰居節點加入選擇節點列表中

    for node in selected_nodes:
        print(node)
    # 去除重复的节点
    selected_nodes = list(set(selected_nodes))

    if not selected_nodes:
        return None, None, None

    subgraph = graph.subgraph(selected_nodes)

    node_features = []  # 不使用one-hot编码
    api_names = []
    for node, node_data in subgraph.nodes(data=True):
        features = [
            1 if node_data.get('external', False) else 0,
            1 if node_data.get('entrypoint', False) else 0
        ]
        api_name = node_data.get('label', "")
        clean_api = clean_api_name(api_name)
        api_names.append(clean_api)
        # 可以根据需要添加其他特征
        # features = []  # 目前留空，因为one-hot不再使用
        node_features.append(features)
    
    adjacency_matrix = nx.adjacency_matrix(subgraph)
    return node_features, api_names, adjacency_matrix

# def filter_apis(node_features, api_names, adjacency_matrix, label):
#     preserved_keywords = ['accounts', 'app', 'bluetooth', 'content', 'location',
#                           'media', 'net', 'nfc', 'provider', 'telecom', 'telephony']
#     to_remove = set()
#     for i, api_name in enumerate(api_names):
#         if any(api_name.startswith(prefix) for prefix in ['java.lang', 'android', 'androidx', 'junit']):
#             if not any(keyword in api_name for keyword in preserved_keywords):
#                 to_remove.add(i)
 
#     # Filter the lists and matrix by removing the indices marked for removal
#     filtered_node_features = [feat for i, feat in enumerate(node_features) if i not in to_remove]
#     filtered_api_names = [name for i, name in enumerate(api_names) if i not in to_remove]
#     filtered_adjacency_matrix = [row for i, row in enumerate(adjacency_matrix) if i not in to_remove]
#     filtered_adjacency_matrix = [[elem for i, elem in enumerate(row) if i not in to_remove] for row in filtered_adjacency_matrix]
#     # filtered_label = [l for i, l in enumerate(label) if i not in to_remove]

#     return filtered_node_features, filtered_api_names, filtered_adjacency_matrix, label
    
def save_data(node_features, api_names, adjacency_matrix, text_label,label_value, save_path, count):
    file_name = f'{text_label}_{count}.pickle'
    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump((node_features, api_names, adjacency_matrix, label_value), f)
    print(f"Saved graph {count} to {file_path}")
    return file_name

def process_and_save_graphs(folder_path, text_label, save_path, packages):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    count = 0
    file_map = {}
    for graph, gexf_file in load_graphs_generator(folder_path):
        label_mapping = {'benign': 0, 'malware': 1}
        label_value = label_mapping[text_label]
        node_features, api_names, adjacency_matrix = process_graphs_with_features(graph, packages)
        if node_features is not None:
            # node_features, api_names, adjacency_matrix = filter_apis(node_features, api_names, adjacency_matrix, label_value)
            pickle_file = save_data(node_features, api_names, adjacency_matrix, text_label,label_value, save_path, count)
            file_map[gexf_file] = pickle_file
            count += 1

    json_filename = os.path.join(save_path, f'{text_label}.json')
    with open(json_filename, 'w') as f:
        json.dump(file_map, f, indent=4)
    print(f"Mapping of GEXF to Pickle saved in {json_filename}")

def main():
    parser = argparse.ArgumentParser(description='Process and save graphs.')
    parser.add_argument('--dir', type=str, required=True, help='Sample data folder')
    parser.add_argument('--processed_dst', type=str, required=True, help='Destination folder for processed data')
    parser.add_argument('--label', type=str, required=True, choices=['malware', 'benign'], help='Label for the graphs')
    
    packages = {'/accounts/', '/app/', '/bluetooth/', '/content/', '/location/','/media/', '/net/', '/nfc/','/provider/', '/telecom/', '/telephony/'}
    
    args = parser.parse_args()

    process_and_save_graphs(args.dir, args.label, args.processed_dst, packages)

if __name__ == "__main__":
    main()