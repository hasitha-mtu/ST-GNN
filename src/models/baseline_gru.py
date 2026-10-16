"""
baseline_gru.py  –  Per-node GRU baseline (no graph structure)
==============================================================
The ST-GNN backbone (models/st_gnn_flood.py) with the graph layers removed
(manuscript Section 3.4): the SAME input projection (Linear + LayerNorm +
ELU over [dynamic ‖ static] features), the SAME per-node GRU (shared
weights), the SAME post-GRU dropout and the SAME two-layer ReLU output
head. Because the GAT stack (hidden → hidden//2) is removed, the head reads
the GRU state directly: Linear(hidden → hidden//4) → ReLU → Dropout →
Linear(hidden//4 → T_out).

REVISION AUDIT (M8): the submitted baseline used a different embedding
(Linear + ReLU + Dropout), no post-GRU dropout and a different head
(hidden → hidden//2 → T_out), so it was not "structurally identical to the
ST-GNN with all graph message-passing layers removed".
"""
import torch
import torch.nn as nn


def recurrent_backbone_head(hidden: int, t_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden, hidden // 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 4, t_out),
    )


class PerNodeGRU(nn.Module):
    """
    forward input:  x_seq  [B, T_in, N, F_dyn], node_attr [N, F_static]
    forward output: delta  [B, T_out, N]
    """
    rnn_cls = nn.GRU

    def __init__(self, f_dyn, f_static, hidden, gru_layers, t_out, dropout):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Sequential(          # = STGNNFloodModel.input_proj
            nn.Linear(f_dyn + f_static, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
        )
        self.rnn = self.rnn_cls(
            input_size=hidden, hidden_size=hidden, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.rnn_dropout = nn.Dropout(dropout)    # = STGNNFloodModel.gru_dropout
        self.head = recurrent_backbone_head(hidden, t_out, dropout)

    def forward(self, x_seq, node_attr, **kwargs):
        # kwargs absorbs edge_index/edge_attr so the call signature matches ST-GNN
        B, T, N, _ = x_seq.shape
        static_exp = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        x = torch.cat([x_seq, static_exp], dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, _ = self.rnn(self.input_proj(x))     # [B*N, T, hidden]
        h = self.rnn_dropout(out[:, -1, :])
        return self.head(h).view(B, N, -1).permute(0, 2, 1)
