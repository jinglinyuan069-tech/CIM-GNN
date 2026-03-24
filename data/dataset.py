import torch
from torch.utils.data import Dataset


class BrainDataset(Dataset):

    def __init__(self, fc_graphs, sc_graphs, targets):

        self.fc = fc_graphs
        self.sc = sc_graphs
        self.targets = targets

    def __len__(self):

        return len(self.targets)

    def __getitem__(self, idx):

        return self.fc[idx], self.sc[idx], self.targets[idx]