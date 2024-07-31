import dgl
import torch
import torch.nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, GraphConv, MaxPooling,SAGEConv
from layer import ConvPoolBlock, SAGPool
from torch.nn import Linear, Conv1d, MaxPool1d, Module, BatchNorm1d, ReLU, Dropout, Softmax



class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_convs=2,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,  
    ):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_convs

        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(
                ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio)
            )
        self.convpools = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hid_dim * 2, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)

    def forward(self, graph,feat,eweight=None):

        if 'weight' in graph.edata:
            eweight = graph.edata['weight']
        else:
            eweight = None

        final_readout = None

        for i in range(self.num_convpools):
            #print(f"Before ConvPoolBlock {i}, eweight is: {graph.edata['weight'].shape}")  # 打印进入 ConvPoolBlock 前的边权重
            graph, feat, readout = self.convpools[i](graph, feat,eweight)
            #print(f"After ConvPoolBlock {i}, eweight is: {graph.edata['weight'].shape}")  # 打印进入 ConvPoolBlock 后的边权重
            
            if i < self.num_convpools - 1:
                eweight = graph.edata['weight']
                # print(f"eweight is: {eweight.shape}")  # 打印进入 ConvPoolBlock 后的边权重
                
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = F.log_softmax(self.lin3(feat), dim=-1)

        return feat


class DGCNDroid(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_convs, pool_ratio, dropout):
        super(DGCNDroid, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.relus = torch.nn.ModuleList()

        # 定义 num_convs 个图卷积层和批量标准化层
        for i in range(num_convs):
            input_dim = in_dim if i == 0 else hid_dim
            self.convs.append(GraphConv(input_dim, hid_dim, allow_zero_in_degree=True))
            self.batch_norms.append(BatchNorm1d(hid_dim))
            self.relus.append(ReLU(inplace=True))

        # 池化层
        self.pool = SAGPool(hid_dim, ratio=pool_ratio)
        # Readout 层
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        # 一维卷积层调整
        self.conv1d_1 = Conv1d(in_channels=hid_dim * 2, out_channels=16, kernel_size=1, stride=1)  # 修改 kernel_size 为 1
        self.conv1d_2 = Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=1)  # 修改 kernel_size 为 1

        # 最大池化层调整
        self.maxpool = MaxPool1d(kernel_size=1, stride=1)  # 修改 kernel_size 为 1

        # 全连接层的输入维度需要在前向传播中动态计算
        self.fc1 = Linear(32 * 1, 128)  # 需要调整输入维度
        self.fc2 = Linear(128, out_dim)

        # Dropout 层
        self.dropout = Dropout(p=dropout)

    def forward(self, graph, feat=None):
        if feat is None:
            feat = graph.ndata['feat']
        
        for conv, bn, relu in zip(self.convs, self.batch_norms, self.relus):
            feat = conv(graph, feat)
            feat = bn(feat)
            feat = relu(feat)
        
        graph, feat, _ = self.pool(graph, feat)
        
        avg_feats = self.avg_readout(graph, feat)
        max_feats = self.max_readout(graph, feat)
        # print("avg_feats shape:", avg_feats.shape)
        # print("max_feats shape:", max_feats.shape)
        feat = torch.cat([avg_feats, max_feats], dim=1)
        # print("Shape after concat:", feat.shape)

        # 添加一个长度维度，以适配Conv1d的输入要求
        feat = feat.unsqueeze(2)
        # print("Shape before Conv1d:", feat.shape)  # 打印添加长度维度后的形状

        feat = F.relu(self.conv1d_1(feat))
        # print("Shape after conv1d_1:", feat.shape)  # 打印第一个卷积层后的形状

        feat = F.relu(self.conv1d_2(feat))
        # print("Shape after conv1d_2:", feat.shape)  # 打印第二个卷积层后的形状

        feat = self.maxpool(feat)
        # print("Shape after maxpool:", feat.shape)  # 打印池化后的形状

        feat = feat.view(feat.size(0), -1)  # 展平特征，为全连接层准备
        feat = F.relu(self.fc1(feat))
        feat = self.dropout(feat)
        feat = self.fc2(feat)
        feat = F.log_softmax(feat, dim=1)


        return feat

    

class SAGE(torch.nn.Module):
    """Modified Self-Attention Graph Pooling Network with specified layers and structure.
    
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node features.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers (default: 3).
        pool_ratio (float, optional): The pool ratio for SAGPooling (default: 0.5).
        dropout (float, optional): The dropout ratio (default: 0).
    """
    
    def __init__(self, in_dim, hid_dim, out_dim, num_convs=4, pool_ratio=0.5, dropout=0.3):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # Adding GraphSAGE convolutional layers along with BatchNorm and ReLU
        for i in range(num_convs):
            in_features = in_dim if i == 0 else hid_dim
            self.convs.append(SAGEConv(in_features, hid_dim, 'pool'))
            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))

        self.pool = SAGPool(hid_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        # Classification layers
        self.lin1 = torch.nn.Linear(hid_dim * 2, hid_dim)  # input features are doubled due to concat of avg and max pooling
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)

        self.dropout = dropout

    def forward(self, graph, feat):
        # If no feature provided, use the default feature 'feat' in graph
        if feat is None:
            feat = graph.ndata['feat']
        
        # Apply convolutions, batch normalization and ReLU
        for conv, bn in zip(self.convs, self.batch_norms):
            feat = conv(graph, feat)
            feat = bn(feat)
            feat = F.relu(feat)
        
        # Apply pooling
        graph, feat, _ = self.pool(graph, feat)

        # Apply global readout (average and max)
        avg_feats = self.avg_readout(graph, feat)
        max_feats = self.max_readout(graph, feat)
        feat = torch.cat([avg_feats, max_feats], dim=1)

        # Classification layers
        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = F.log_softmax(self.lin3(feat), dim=-1)

        return feat


def get_sag_network(net_type: str = "hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "DGCNDroid":
        return DGCNDroid
    elif net_type == "SAGE":
        return SAGE
    else:
        raise ValueError(
            "SAGNetwork type {} is not supported.".format(net_type)
        )
