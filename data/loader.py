import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def build_graph(adj, features):

    edge_index = torch.nonzero(adj).t().contiguous()
    edge_weight = adj[edge_index[0], edge_index[1]]

    data = Data(
        x=features,
        edge_index=edge_index,
        edge_weight=edge_weight
    )

    return data


def build_dataset(fc_list, sc_list, feature_list, targets):

    fc_graphs = []
    sc_graphs = []

    for i in range(len(targets)):

        fc_adj = torch.tensor(fc_list[i]).float()
        sc_adj = torch.tensor(sc_list[i]).float()

        x = torch.tensor(feature_list[i]).float()

        fc_graph = build_graph(fc_adj, x)
        sc_graph = build_graph(sc_adj, x)

        fc_graphs.append(fc_graph)
        sc_graphs.append(sc_graph)

    y = torch.tensor(targets).float()

    return fc_graphs, sc_graphs, y


class MultimodalDataset(torch.utils.data.Dataset):

    def __init__(self, fc_graphs, sc_graphs, y):

        self.fc = fc_graphs
        self.sc = sc_graphs
        self.y = y

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return self.fc[idx], self.sc[idx], self.y[idx]


def collate_fn(batch):

    fc_list = []
    sc_list = []
    y_list = []

    for fc, sc, y in batch:

        fc_list.append(fc)
        sc_list.append(sc)
        y_list.append(y)

    return fc_list, sc_list, torch.stack(y_list)


def create_loader(fc_graphs, sc_graphs, y, batch_size, shuffle=True):

    dataset = MultimodalDataset(fc_graphs, sc_graphs, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader