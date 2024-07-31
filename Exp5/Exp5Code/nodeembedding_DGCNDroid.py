import os
import pickle
import networkx as nx
import numpy as np
import argparse
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor
import threading

def load_data_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            node_features, api_names, adjacency_matrix, label = pickle.load(file)
            return node_features, api_names, adjacency_matrix, label
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None, None

def calculate_structural_features(node_features, adjacency_matrix, n_hops=5):

    G = nx.from_scipy_sparse_array(csr_matrix(adjacency_matrix))
    
    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Calculate closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    
    # Calculate n-hop neighboring nodes count
    n_hop_neighbors_count = {}
    for node in G.nodes():
        neighbors = set(nx.single_source_shortest_path_length(G, node, cutoff=n_hops).keys())
        n_hop_neighbors_count[node] = len(neighbors) - 1  # Exclude the node itself
    
    # Combine the features for each node
    for i in range(len(node_features)):
        node_features[i].append(betweenness_centrality[i])
        node_features[i].append(closeness_centrality[i])
        node_features[i].append(n_hop_neighbors_count[i])
    
    return node_features

def process_file(file_path, save_path, lock, processing_files):
    save_file_path = os.path.join(save_path, os.path.basename(file_path))

    with lock:
        if file_path in processing_files:
            print(f"File is already being processed: {file_path}")
            return
        if os.path.exists(save_file_path):
            print(f"File already processed: {save_file_path}")
            return
        processing_files.add(file_path)

    try:
        node_features, api_names, adjacency_matrix, label = load_data_from_pickle(file_path)
        if node_features is not None:
            structural_features = calculate_structural_features(node_features, adjacency_matrix)
            updated_data = (structural_features, api_names, adjacency_matrix, label)
            with open(save_file_path, 'wb') as file:
                pickle.dump(updated_data, file)
            print(f"Updated graph data saved to {save_file_path}")
    finally:
        with lock:
            processing_files.remove(file_path)

def process_folder(input_folder, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    lock = threading.Lock()
    processing_files = set()
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".pickle"):
                file_path = os.path.join(input_folder, file_name)
                executor.submit(process_file, file_path, save_path, lock, processing_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pickle files in a folder and save updated data")
    parser.add_argument("--dir", help="Path to the folder containing pickle files")
    parser.add_argument("--processed_dst", help="Path to save the updated data")
    
    args = parser.parse_args()
    
    process_folder(args.dir, args.processed_dst)
