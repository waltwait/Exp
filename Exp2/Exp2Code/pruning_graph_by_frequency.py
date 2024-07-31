import networkx as nx
import json
import os
import pickle
import argparse
from module import clean_api_name, extract_class_path
import logging
import concurrent.futures
import re

def save_file_mapping(mapping, save_path):
    file_path = os.path.join(save_path, 'file_mapping.json')
    with open(file_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    print(f"Saved file mapping to {file_path}")

def fix_reflection(api_name):
    cleaned_api_name = re.sub(r"\$\d", "", api_name)
    return cleaned_api_name

def load_filtered_apis(correlation_file, threshold):
    with open(correlation_file, 'r') as file:
        correlation_data = json.load(file)
    return {fix_reflection(api): data for api, data in correlation_data.items() if data['importance'] > threshold}
    

def trim_graph(graph, top_apis, p_nodes):
    trimmed_graph = nx.DiGraph()
    visited = set()

    # 搜索图中的每个节点
    for node, node_data in graph.nodes(data=True):
        api_name = node_data.get('label', "")
        clean_api = clean_api_name(api_name)
        clean_api = fix_reflection(clean_api)
        # inser = 1 if node_data.get('external', False) else 0
        class_path = extract_class_path(clean_api)
        # 如果API在top_apis中，且未访问过，则将其加入到trimmed_graph中
        if class_path in top_apis and node not in visited:
            print(f"Adding node {clean_api} to trimmed graph")
            trimmed_graph.add_node(node, **node_data)
            visited.add(node)
            queue = [(node, 0)]
            while queue:
                current_node, current_depth = queue.pop(0)
                # 考虑当前节点的前驱和后继
                neighbors = list(graph.predecessors(current_node)) + list(graph.successors(current_node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        trimmed_graph.add_node(neighbor, **graph.nodes[neighbor])
                        # 确保在添加边时考虑图的方向性
                        if graph.has_edge(current_node, neighbor):
                            trimmed_graph.add_edge(current_node, neighbor)
                        if graph.has_edge(neighbor, current_node):
                            trimmed_graph.add_edge(neighbor, current_node)
                        visited.add(neighbor)
                        # 如果当前深度小于p_nodes，则继续BFS
                        if current_depth + 1 < p_nodes:
                            queue.append((neighbor, current_depth + 1))
                
            

    return trimmed_graph


def process_graphs_with_features(graph, top_apis, p_nodes):
    graph = trim_graph(graph, top_apis, p_nodes)
    if len(graph) == 0 or graph.number_of_edges() == 0:
        return None, None, None

    node_features = []
    api_names = []
    for node, node_data in graph.nodes(data=True):
        features = [
            1 if node_data.get('external', False) else 0,
            1 if node_data.get('entrypoint', False) else 0
        ]
        api_name = node_data.get('label', "")
        clean_api = clean_api_name(api_name)
        api_names.append(clean_api)
        node_features.append(features)
    adjacency_matrix = nx.adjacency_matrix(graph)
    return node_features, api_names, adjacency_matrix

def save_data(node_features, api_names, adjacency_matrix, text_label,label_value, save_path, count):
    file_name = f'{text_label}_{count}.pickle'
    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump((node_features, api_names, adjacency_matrix, label_value), f)
    print(f"Saved graph {count} to {file_path}")

def process_file(path, label, save_path, top_apis, p_nodes, label_value, file_index, file_mapping):
    logging.info(f"Processing file: {path}")
    graph = nx.read_gexf(path)
    node_features, api_names, adjacency_matrix = process_graphs_with_features(graph, top_apis, p_nodes)
    if node_features is not None:
        save_data(node_features, api_names, adjacency_matrix, label, label_value, save_path, file_index)
        # 更新文件映射
        file_mapping[path] = f"{label}_{file_index}.pickle"
        return f"Processed and saved graph with index {file_index}"
    else:
        return f"No data to save for graph with index {file_index}"

def find_next_index(path):
    """ This function finds the next available index by checking existing files. """
    max_index = -1
    for fname in os.listdir(path):
        if fname.endswith('.pickle'):
            parts = fname.split('_')
            index = int(parts[-1].split('.')[0])  # Assumes the format `label_index.pickle`
            max_index = max(max_index, index)
    return max_index + 1

def process_and_save_graphs(folder_path, label, save_path, top_apis, p_nodes):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logging.info(f"Created directory {save_path}")
    
    # Initialize the file index by checking existing files in save_path
    initial_index = find_next_index(save_path)
    file_mapping = {}  # 初始化文件映射

    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.gexf')]
    label_mapping = {'benign': 0, 'adware': 1, 'banking': 2, 'sms': 3, 'riskware': 4, 'malware': 1}
    label_value = label_mapping[label]
    
    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting files to the executor with a unique index
        futures = {executor.submit(process_file, file, label, save_path, top_apis, p_nodes, label_value, idx + initial_index, file_mapping): idx for idx, file in enumerate(files)}
        for future in concurrent.futures.as_completed(futures):
            logging.info(future.result())

    # 保存文件映射
    save_file_mapping(file_mapping, save_path)

def main():
    parser = argparse.ArgumentParser(description='Process and save graphs.')
    parser.add_argument('--dir', type=str, required=True, help='Sample data folder')
    parser.add_argument('--processed_dst', type=str, required=True, help='Destination folder for processed data')
    parser.add_argument('--label', type=str, required=True, choices=['adware', 'banking', 'sms', 'riskware', 'benign','malware'], help='Label for the graphs')
    parser.add_argument('--correlation_json', type=str, required=True, help='JSON file with correlation results')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for filtering APIs based on correlation')
    parser.add_argument('--Pnode', type=int, required=True, help='Number of successors to keep for each important API')
    
    args = parser.parse_args()
    top_apis = load_filtered_apis(args.correlation_json, args.threshold)
    process_and_save_graphs(args.dir, args.label, args.processed_dst, top_apis, args.Pnode)

if __name__ == "__main__":
    main()