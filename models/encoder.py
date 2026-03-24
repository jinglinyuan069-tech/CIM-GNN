import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphNorm


class GraphEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.2):

        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(GraphNorm(hidden_dim))

        for _ in range(num_layers - 1):

            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):

        h = x

        for conv, norm in zip(self.convs, self.norms):

            h = conv(h, edge_index, edge_weight)

            if batch is not None:
                h = norm(h, batch)

            h = F.relu(h)

            h = F.dropout(h, p=self.dropout, training=self.training)

        return h


class FCEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim):

        super().__init__()

        self.encoder = GraphEncoder(in_dim, hidden_dim)

    def forward(self, data):

        h = self.encoder(
            data.x,
            data.edge_index,
            data.edge_weight if hasattr(data, "edge_weight") else None,
            data.batch if hasattr(data, "batch") else None
        )

        return h


class SCEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim):

        super().__init__()

        self.encoder = GraphEncoder(in_dim, hidden_dim)

    def forward(self, data):

        h = self.encoder(
            data.x,
            data.edge_index,
            data.edge_weight if hasattr(data, "edge_weight") else None,
            data.batch if hasattr(data, "batch") else None
        )

        return h