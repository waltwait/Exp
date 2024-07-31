import pickle
import dgl
import torch
import numpy as np
import scipy.sparse as sp

def load_data_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            node_features, api_names, adjacency_matrix, label = pickle.load(file)
            return node_features, api_names, adjacency_matrix, label
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None, None
    
def create_dgl_graph(node_features, adjacency_matrix,edge_weights=None):
    """
    从SciPy稀疏矩阵创建DGL图，并确保图在指定的设备上。
    """
    # 确保 adjacency_matrix 是 scipy.sparse 矩阵
    if not sp.isspmatrix(adjacency_matrix):
        adjacency_matrix = sp.csr_matrix(adjacency_matrix)

    # 创建图
    g = dgl.from_scipy(adjacency_matrix, idtype=torch.int64)

    # 将节点特征转换为 Tensor 并赋值给图的节点数据，确保它们在 GPU 上
    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)

    if edge_weights is not None:
        # 假设edge_weights是一个一维numpy数组，每个元素对应一条边的权重
        g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)


    return g

def extract_class_path(api_name):
    # 查找类路径和方法分隔符的位置
    separator_index = api_name.find('(')
    if separator_index == -1:
        print("No valid separator found.")
        return None
    # 提取类路径部分
    class_path = api_name[:separator_index]
    return class_path

# def clean_api_name(api_name):
#     """
#     清理 API 名称：移除方法参数和返回类型，将分隔符 `;->` 替换为 `.`。
#     """
#     # 移除方法参数和返回类型
#     clean_api = api_name.split("(")[0]
#     # 替换分隔符，转换为完整类型名称
#     clean_api = clean_api.replace(";->", ".").replace("/", ".")[1:]
#     return clean_api

def clean_api_name(api_name):
    """
    清理 API 名称，保留完整的类名、方法名和方法参数。
    能够同时处理常规方法和构造函数。
    """
    if ";->" in api_name:
        # 对常规方法，从参数中分割类/方法
        class_and_method, params = api_name.split(";->")
        params = params.split(")")[0] + ")"
    else:
        # 处理构造函数或其他不包含 ";->" 的情况
        class_and_method = api_name
        params = ""

    # 替换分隔符，移除类名前的 'L'
    clean_api = class_and_method.replace("/", ".").lstrip("L")
    
    if params:
        # 如果存在方法参数，则附加到方法名后
        clean_api += "." + params
    
    return clean_api

def extract_number(filename):
    # 提取文件名中的数字部分
    number_part = filename.split('_')[1].split('.')[0]
    return int(number_part)


# def load_data_from_pickle(file_path):
#     try:
#         with open(file_path, 'rb') as file:
#             return pickle.load(file)
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None, None, None

# def create_dgl_graph(node_features, adjacency_matrix, label):
#     g = dgl.from_scipy(adjacency_matrix)
#     g.ndata['feat'] = torch.FloatTensor(node_features)
#     return g, label