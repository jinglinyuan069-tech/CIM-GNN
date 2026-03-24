import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityWeighting(nn.Module):

    def __init__(self, hidden_dim, num_modalities=2):

        super().__init__()

        self.num_modalities = num_modalities

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings):

        scores = []

        for z in embeddings:

            s = self.scorer(z)

            scores.append(s)

        scores = torch.stack(scores)

        alpha = F.softmax(scores, dim=0)

        fused = 0

        for i in range(self.num_modalities):

            fused = fused + alpha[i] * embeddings[i]

        return fused, alpha