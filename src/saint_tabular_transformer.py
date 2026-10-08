"""Tabular Transformer / SAINT-style architecture recovered from the NSL-KDD experiments."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NSLKDDDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, label_col='label'):
        self.cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx], self.labels[idx]


class TabTransformer(nn.Module):
    def __init__(self, cat_dims, n_classes, cont_dim, emb_dim=32, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim in cat_dims])
        self.cont_bn = nn.BatchNorm1d(cont_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(len(cat_dims) * emb_dim + cont_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_cat, x_cont):
        cat_x = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
        cat_x = torch.stack(cat_x, dim=1)
        cat_x = self.transformer(cat_x).flatten(start_dim=1)
        cont_x = self.cont_bn(x_cont)
        return self.classifier(torch.cat([cat_x, cont_x], dim=1))
