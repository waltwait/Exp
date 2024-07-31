import os
import pickle
from gensim.models import Word2Vec
import argparse

def load_model(model_path):
    # 載入已訓練的 Word2Vec 模型
    model = Word2Vec.load(model_path)
    return model

def get_word2vec_embedding(model, api_name):
    # 從 Word2Vec 模型中獲取 API 名稱的向量嵌入
    if api_name in model.wv:
        return model.wv[api_name]
    else:
        # 如果 API 名稱不在詞彙表中，返回一個零向量
        return [0] * model.vector_size

def process_and_update_graph_data(file_path, model, save_path=None):
    try:
        with open(file_path, 'rb') as file:
            node_features, api_names, adjacency_matrix, label = pickle.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    for i, api_name in enumerate(api_names):
        print(f"Processing API {i+1}/{len(api_names)}: {api_name}")
        word2vec_embedding = get_word2vec_embedding(model, api_name)
        node_features[i].extend(word2vec_embedding)

    updated_data = (node_features, api_names, adjacency_matrix, label)
    
    save_file_path = os.path.join(save_path, os.path.basename(file_path)) if save_path else file_path

    with open(save_file_path, 'wb') as file:
        pickle.dump(updated_data, file)
    print(f"Updated graph data saved to {save_file_path}")

def process_folder(data_folder, model, save_folder):
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.pickle'):
            file_path = os.path.join(data_folder, file_name)
            print(f"Processing file: {file_path}")
            process_and_update_graph_data(file_path, model, save_folder)

# 範例使用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder of .pkl files with Word2Vec embeddings.")
    parser.add_argument('--dir', type=str, required=True, help='The directory containing the .pkl files')
    parser.add_argument('--model_path', type=str, required=True, help='The path to the trained Word2Vec model')
    parser.add_argument('--processed_dst', type=str, required=True, help='The directory to save updated .pkl files')

    args = parser.parse_args()

    # Load the Word2Vec model
    model = load_model(args.model_path)

    # Process the directory of .pkl files
    process_folder(args.dir, model, args.processed_dst)
