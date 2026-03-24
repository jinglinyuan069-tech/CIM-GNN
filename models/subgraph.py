import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgraphSelector(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def compute_edge_features(self, h, edge_index):

        src = h[edge_index[0]]
        dst = h[edge_index[1]]

        edge_feat = torch.cat([src, dst], dim=1)

        return edge_feat


    def compute_mask(self, edge_feat):

        mask = self.edge_mlp(edge_feat)

        mask = torch.sigmoid(mask)

        return mask.squeeze()


    def forward(self, h, edge_index):

        edge_feat = self.compute_edge_features(h, edge_index)

        mask = self.compute_mask(edge_feat)

        return mask