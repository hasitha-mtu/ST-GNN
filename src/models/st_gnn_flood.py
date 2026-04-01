import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class STGNNFloodModel(nn.Module):
    """
    Spatio-Temporal GNN for flood stage-anomaly forecasting.

    Forward pass:
      x_seq      [B, T_in, N, F_dyn]
      node_attr  [N, F_static]
      edge_index [2, E]
      edge_attr  [E, F_edge]

    Returns:
      pred       [B, T_out, N]   predicted stage_anomaly
    """

    def __init__(
            self,
            f_dyn: int,  # dynamic feature dim  (5)
            f_static: int,  # static  feature dim  (7)
            f_edge: int,  # edge    feature dim  (4)
            hidden: int,
            gat_heads: int,
            gru_layers: int,
            t_out: int,
            dropout: float,
    ):
        super().__init__()
        self.hidden = hidden
        self.gat_heads = gat_heads
        self.t_out = t_out

        # ── Node embedding ─────────────────────────────────────────────
        # Combines dynamic + static features into hidden_dim
        self.node_embed = nn.Sequential(
            nn.Linear(f_dyn + f_static, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Edge embedding (project edge_attr to hidden_dim for GAT) ──
        self.edge_embed = nn.Linear(f_edge, hidden)

        # ── Spatial: 2-head GAT ────────────────────────────────────────
        # concat=True → output dim = hidden * gat_heads
        # We project back to hidden after GAT
        self.gat = GATConv(
            in_channels=hidden,
            out_channels=hidden // gat_heads,
            heads=gat_heads,
            concat=True,
            edge_dim=hidden,
            dropout=dropout,
            add_self_loops=True,
        )
        self.gat_norm = nn.LayerNorm(hidden)
        self.gat_drop = nn.Dropout(dropout)

        # ── Temporal: GRU ──────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ── Output head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # ── FIX: Added to prevent overfitting to specific temporal states
            nn.Linear(hidden // 2, t_out),
        )

    def forward(self, x_seq, node_attr, edge_index, edge_attr):
        B, T, N, _ = x_seq.shape

        # ── Edge embedding (once) ────────────────────────────────────────
        edge_feat = self.edge_embed(edge_attr)  # [E, hidden]

        # ── Node embedding (all timesteps at once) ───────────────────────
        static_exp = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        combined = torch.cat([x_seq, static_exp], dim=-1)  # [B, T, N, F]
        h = self.node_embed(combined.reshape(B * T * N, -1))
        h = h.view(B * T, N, self.hidden)  # [B*T, N, hidden]

        # ── ONE batched GAT call across B*T graphs ───────────────────────
        # Build edge index once — offset by N for each of the B*T graphs
        BT = B * T
        offsets = torch.arange(BT, device=edge_index.device) * N  # [B*T]
        ei = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)  # [B*T, 2, E]
        ei = ei.permute(1, 0, 2).reshape(2, -1)  # [2, B*T*E]

        # Expanding edges isn't the most memory-efficient PyTorch operation,
        # but works fine here given the static graph topology.
        ea = edge_feat.unsqueeze(0).expand(BT, -1, -1).reshape(-1, edge_feat.shape[-1])

        h_flat = h.reshape(BT * N, self.hidden)
        gat_out = self.gat(h_flat, ei, ea)  # one call
        gat_out = self.gat_norm(gat_out + h_flat)
        gat_out = self.gat_drop(gat_out)

        # ── GRU ─────────────────────────────────────────────────────────
        # [B*T, N, hidden] → [B, N, T, hidden] → [B*N, T, hidden]
        gru_in = gat_out.view(B, T, N, self.hidden) \
            .permute(0, 2, 1, 3) \
            .reshape(B * N, T, self.hidden)
        _, h_n = self.gru(gru_in)  # [layers, B*N, hidden]

        # ── Output ──────────────────────────────────────────────────────
        pred = self.head(h_n[-1]) \
            .view(B, N, self.t_out) \
            .permute(0, 2, 1)  # [B, T_out, N]
        return pred