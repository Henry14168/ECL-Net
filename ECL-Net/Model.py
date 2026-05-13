import torch
import torch.nn as nn
class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False):
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.edges_in_d = edges_in_d

        input_edge = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            if self.edges_in_d > 0:
                dummy_attr = torch.zeros(source.size(0), self.edges_in_d, device=source.device)
                out = torch.cat([source, target, radial, dummy_attr], dim=1)
            else:
                out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)

        out = self.edge_mlp(out)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = self.unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-10.0, max=10.0)
        agg = self.unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat)
        return h, coord, edge_attr

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        result = data.new_zeros((num_segments, data.size(1)))
        result.index_add_(0, segment_ids, data)
        return result
    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        result = data.new_zeros((num_segments, data.size(1)))
        count = data.new_zeros((num_segments, data.size(1)))
        result.index_add_(0, segment_ids, data)
        count.index_add_(0, segment_ids, torch.ones_like(data))
        return result / count.clamp(min=1)

class EGNNModel(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=256, n_layers=3, dropout_rate=0.4, edge_dim=16):
        super().__init__()
        self.embedding_in = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, hidden_dim)
        )

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(E_GCL(hidden_dim, hidden_dim, hidden_dim, edges_in_d=edge_dim))

        final_dim = hidden_dim * 4
        self.predictor = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        )

        self.cl_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward_egnn(self, x, pos, edge_index, edge_attr):
        h = self.embedding_in(x)
        for layer in self.layers:
            h, pos, _ = layer(h, edge_index, pos, edge_attr)
        return h

    def get_split_features_raw(self, x, mut_idx, env_idx, weights=None):
        feat_site = x[mut_idx]

        feat_env_raw = x[env_idx]
        if feat_env_raw.dim() == 2:
            feat_env_raw = feat_env_raw.unsqueeze(1)

        if weights is not None:
            w = weights.unsqueeze(-1)
            w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-9)
            feat_env_final = torch.sum(feat_env_raw * w_norm, dim=1)
        else:
            feat_env_final = torch.mean(feat_env_raw, dim=1)

        return feat_site, feat_env_final

    def forward(self, data):
        edge_attr_s = getattr(data, 'edge_attr_s', None)
        h_wt = self.forward_egnn(data.x_s, data.pos_s, data.edge_index_s, edge_attr_s)

        w_s = getattr(data, 'env_weight_s', None)

        wt_site, wt_env = self.get_split_features_raw(h_wt, data.mut_idx_s, data.env_idx_s, weights=w_s)

        edge_attr_t = getattr(data, 'edge_attr_t', None)
        h_mut = self.forward_egnn(data.x_t, data.pos_t, data.edge_index_t, edge_attr_t)

        w_t = getattr(data, 'env_weight_t', None)
        mut_site, mut_env = self.get_split_features_raw(h_mut, data.mut_idx_t, data.env_idx_t, weights=w_t)

        final_input = torch.cat([wt_site, wt_env, mut_site, mut_env], dim=1)
        logits = self.predictor(final_input)

        z_wt_site = self.cl_projector(wt_site)
        z_wt_env = self.cl_projector(wt_env)

        z_mut_site = self.cl_projector(mut_site)
        z_mut_env = self.cl_projector(mut_env)

        return logits, (z_wt_site, z_wt_env), (z_mut_site, z_mut_env)