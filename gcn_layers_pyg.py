import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

torch.manual_seed(1234)

class GraphConvolution(nn.Module):
    """GCN layer using torch_geometric."""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = GCNConv(in_features, out_features, bias=bias, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features} -> {self.out_features})'