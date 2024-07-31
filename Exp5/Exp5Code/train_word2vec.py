import os
import pickle
import networkx as nx
from gensim.models import Word2Vec
import argparse
import sys


parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from module import load_data_from_pickle

def wcc_sequences(api_names, adjacency_matrix):
    G = nx.DiGraph(adjacency_matrix)
    components = list(nx.weakly_connected_components(G))
    sequences = []
    for component in components:
        sequence = [api_names[node] for node in component]
        sequences.append(sequence)
    return sequences

# 處理單個檔案
def process_file(file_path):
    _, api_names, adjacency_matrix, _ = load_data_from_pickle(file_path)
    return wcc_sequences(api_names, adjacency_matrix)

# 主函數
def main(folder_path, model_dst):
    all_sequences = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.pickle'):
                file_path = os.path.join(root, file_name)
                print(file_name)
                sequences = process_file(file_path)
                all_sequences.extend(sequences)
                # print(all_sequences)
    model = Word2Vec(sentences=all_sequences, vector_size=60, window=5, min_count=1, workers=4,sg=1)
    model.save(model_dst)
    print(f"Model saved to {model_dst}")
    return model

# 命令行參數設置
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--dir', type=str, help='The directory containing the .pkl files')
    parser.add_argument('--model_dst', type=str, help='The destination path for the processed word embedding model')

    args = parser.parse_args()
    main(args.dir, args.model_dst)