import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GraphConvolution(nn.Module):
    """Graph convolutional layer wrapper based on torch_geometric."""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # GCNConv with self-loops disabled (self-loops are already added in adjacency matrix)
        self.conv = GCNConv(in_features, out_features, bias=bias, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges] in torch_geometric format
            edge_weight: Edge weights [num_edges] (optional)
        Returns:
            Output features [num_nodes, out_features]
        """
        return self.conv(x, edge_index, edge_weight)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features} -> {self.out_features})'