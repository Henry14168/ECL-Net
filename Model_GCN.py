import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class EGNNModel(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=256, n_layers=3, dropout_rate=0.4, edge_dim=16):
        super().__init__()

        self.embedding_in = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(512, hidden_dim)
        )

        self.layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        )

        self.con_projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                           nn.Linear(hidden_dim, hidden_dim))

    def extract_graph_features(self, x, pos, edge_index, mut_idx, env_idx, edge_attr=None, weights=None):
        h = self.embedding_in(x)

        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.silu(h)

        site = h[mut_idx]
        env = h[env_idx] if h[env_idx].dim() == 3 else h[env_idx].unsqueeze(1)

        if weights is not None:
            w = weights.unsqueeze(-1)
            env = (env * (w / (w.sum(dim=1, keepdim=True) + 1e-9))).sum(dim=1)
        else:
            env = env.mean(dim=1)

        return site, env

    def forward(self, data):
        wt_site, wt_env = self.extract_graph_features(
            data.x_s, data.pos_s, data.edge_index_s, data.mut_idx_s, data.env_idx_s,
            getattr(data, 'edge_attr_s', None), getattr(data, 'env_weight_s', None)
        )
        mut_site, mut_env = self.extract_graph_features(
            data.x_t, data.pos_t, data.edge_index_t, data.mut_idx_t, data.env_idx_t,
            getattr(data, 'edge_attr_t', None), getattr(data, 'env_weight_t', None)
        )

        logits = self.predictor(torch.cat([wt_site, wt_env, mut_site, mut_env], dim=1))

        z_wt = (self.con_projector(wt_site), self.con_projector(wt_env))
        z_mut = (self.con_projector(mut_site), self.con_projector(mut_env))

        return logits, z_wt, z_mut