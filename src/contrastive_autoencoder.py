"""Contrastive autoencoder + SVM model recovered from the NSL-KDD experiments."""

import torch
import torch.nn as nn
from sklearn.svm import SVC


class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def contrastive_loss(z1, z2, temperature=0.5):
    z1_norm = nn.functional.normalize(z1, dim=1)
    z2_norm = nn.functional.normalize(z2, dim=1)
    representations = torch.cat([z1_norm, z2_norm], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    pos_sim = torch.exp(torch.sum(z1_norm * z2_norm, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    mask = ~torch.eye(similarity_matrix.size(0), device=z1.device).bool()
    neg_sim = similarity_matrix[mask].view(similarity_matrix.size(0), -1)
    neg_sim_exp = torch.exp(neg_sim / temperature).sum(dim=1)
    return (-torch.log(pos_sim / neg_sim_exp)).mean()


def build_svm():
    return SVC(kernel='rbf', probability=True, random_state=42)
