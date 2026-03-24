import torch
import numpy as np


def adj_to_edge_index(adj):

    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj)

    edge_index = torch.nonzero(adj).t().contiguous()

    edge_weight = adj[edge_index[0], edge_index[1]]

    return edge_index, edge_weight


def threshold_graph(adj, threshold=0.0):

    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj)

    mask = adj > threshold

    adj = adj * mask

    return adj


def normalize_adj(adj):

    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj)

    deg = torch.sum(adj, dim=1)

    deg_inv_sqrt = torch.pow(deg, -0.5)

    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    D = torch.diag(deg_inv_sqrt)

    adj_norm = D @ adj @ D

    return adj_norm


def topk_graph(adj, k):

    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj)

    n = adj.shape[0]

    new_adj = torch.zeros_like(adj)

    for i in range(n):

        values, indices = torch.topk(adj[i], k)

        new_adj[i, indices] = values

    return new_adj


def symmetrize(adj):

    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj)

    adj = (adj + adj.t()) / 2

    return adj


def remove_self_loops(adj):

    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj)

    adj.fill_diagonal_(0)

    return adj


def graph_density(adj):

    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj)

    n = adj.shape[0]

    edges = torch.count_nonzero(adj)

    density = edges / (n * (n - 1))

    return density.item()