"""
baseline_gru.py  –  Per-node GRU baseline (no graph structure)
==============================================================
Identical hyperparameters to STGNNFloodModel but GATConv removed.
Each node is forecast independently from its own feature history.
Purpose: isolate the contribution of graph message passing.
"""
import torch
import torch.nn as nn

class PerNodeGRU(nn.Module):
    """
    GRU applied independently per node — no spatial component.
    Equivalent to running 27 separate GRUs, one per gauge station.

    forward input:  x_seq  [B, T_in, N, F_dyn+F_static]
    forward output: pred   [B, T_out, N]
    """
    def __init__(self, f_dyn, f_static, hidden, gru_layers, t_out, dropout):
        super().__init__()

        # Same node embedding as ST-GNN
        self.node_embed = nn.Sequential(
            nn.Linear(f_dyn + f_static, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Same GRU as ST-GNN — no GAT between embedding and GRU
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Same output head as ST-GNN
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, t_out),
        )

    def forward(self, x_seq, node_attr, **kwargs):
        # kwargs absorbs edge_index/edge_attr so call signature matches ST-GNN
        B, T, N, _ = x_seq.shape

        # Concat static features — same as ST-GNN
        static_exp = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        combined   = torch.cat([x_seq, static_exp], dim=-1)  # [B, T, N, F]

        # Embed all nodes at all timesteps
        h = self.node_embed(combined.reshape(B * T * N, -1))
        h = h.view(B, T, N, -1)                              # [B, T, N, hidden]

        # GRU per node — reshape to [B*N, T, hidden]
        gru_in    = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, h_n    = self.gru(gru_in)                          # [layers, B*N, hidden]

        # Output
        pred = self.head(h_n[-1])\
                   .view(B, N, -1)\
                   .permute(0, 2, 1)                          # [B, T_out, N]
        return pred