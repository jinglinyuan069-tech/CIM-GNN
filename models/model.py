import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool

from models.encoder import FCEncoder
from models.encoder import SCEncoder
from models.subgraph import SubgraphSelector
from models.modality import ModalityWeighting


class MultimodalBrainGNN(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.fc_encoder = FCEncoder(config.in_dim, config.hidden_dim)
        self.sc_encoder = SCEncoder(config.in_dim, config.hidden_dim)

        self.subgraph = SubgraphSelector(config.hidden_dim)

        self.modality = ModalityWeighting(config.hidden_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, 1)
        )

    def encode_modalities(self, fc_data, sc_data):

        h_fc = self.fc_encoder(fc_data)
        h_sc = self.sc_encoder(sc_data)

        return h_fc, h_sc


    def subgraph_mask(self, h, edge_index):

        mask = self.subgraph(h, edge_index)

        return mask


    def pool_embeddings(self, h_fc, h_sc, fc_data, sc_data):

        z_fc = global_mean_pool(h_fc, fc_data.batch)
        z_sc = global_mean_pool(h_sc, sc_data.batch)

        return z_fc, z_sc


    def fuse_modalities(self, z_fc, z_sc):

        z, alpha = self.modality([z_fc, z_sc])

        return z, alpha


    def predict(self, z):

        y = self.pred_head(z)

        return y


    def forward(self, fc_data, sc_data):

        h_fc, h_sc = self.encode_modalities(fc_data, sc_data)

        edge_mask = self.subgraph_mask(h_fc, fc_data.edge_index)

        z_fc, z_sc = self.pool_embeddings(h_fc, h_sc, fc_data, sc_data)

        z, alpha = self.fuse_modalities(z_fc, z_sc)

        y = self.predict(z)

        return y, h_fc, h_sc, edge_mask, alpha