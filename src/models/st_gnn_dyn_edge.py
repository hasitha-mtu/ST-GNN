"""
st_gnn_dyn_edge.py  –  ST-GNN with discharge-based dynamic edge weights (Phase 1)
==================================================================================
Architecture
------------
Identical to STGNNFloodModel (static graph) except the edge attribute vector is
augmented with a fifth dynamic feature computed from observed discharge at each
forward pass.

Static edge attributes  [f_edge=4]:
    [0] river_dist_km     reach length
    [1] area_ratio        upstream_area_src / upstream_area_dst
    [2] elev_drop_m       elevation drop across reach
    [3] same_tributary    1 if both nodes on same tributary, else 0

Dynamic edge attribute  [f_edge=5]:
    [4] hydraulic_conductance   discharge-based routing weight (per batch, per timestep)

Physical basis
--------------
Manning's equation for open channel flow:

    Q = (1/n) × A × R^(2/3) × S^(1/2)

As discharge rises, cross-sectional area A and hydraulic radius R increase
nonlinearly — the reach conducts more flow per unit head difference, so
GATConv attention on that edge should increase.  The conductance feature
encodes this as a smooth, differentiable function of the discharge ratio at
the upstream node relative to a reference discharge.

    conductance(i,j,t) = sigmoid( scale × (Q_src(t) / Q_ref - 1) )

where Q_ref is a per-node reference discharge (mean over training data) and
scale is a learnable scalar initialised to 3.0.  Values approach 1 at high
flow (strong coupling) and 0.5 at mean flow (neutral).

The extra edge feature adds zero parameters to GATConv (it only affects the
attention score computation via the edge_dim projection already in place).
One learnable scalar `conductance_scale` is added for calibration (~1 param).

Backward compatibility
----------------------
f_edge=4 still accepted — conductance is then zeros (equivalent to no dynamic
weighting, identical to STGNNFloodModel).  But callers should pass f_edge=5
to use the dynamic path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class STGNNDynEdge(nn.Module):
    """
    ST-GNN with discharge-based dynamic edge weights.

    Parameters
    ----------
    f_dyn : int
        Dynamic input features per node per timestep (5 or 11).
    f_static : int
        Static node attribute count from nodes.csv.
    f_edge : int
        Edge attribute count including the dynamic conductance feature.
        Pass 5 to enable dynamic weighting (4 static + 1 dynamic).
    hidden : int
        Internal hidden dimension. Default 64.
    gat_heads : int
        Attention heads. Default 2.
    gru_layers : int
        GRU depth. Default 2.
    t_out : int
        Forecast horizons. Default 4.
    dropout : float
        Dropout rate. Default 0.1.
    sar_emb_dim : int
        SAR-FNO embedding dimension. 0 = no SAR. Default 0.
    discharge_idx : int
        Column index of discharge_m3s in the dynamic feature vector.
        Default 3 (matches build_dataset.py GAUGE_FEATURES ordering).
    discharge_ref : float
        Reference discharge for conductance normalisation (m³/s after log1p
        transform).  Default 1.0 (log1p(~1.7 m³/s); tune per catchment).
    """

    def __init__(
        self,
        f_dyn:          int,
        f_static:       int,
        f_edge:         int,
        hidden:         int   = 64,
        gat_heads:      int   = 2,
        gru_layers:     int   = 2,
        t_out:          int   = 4,
        dropout:        float = 0.1,
        sar_emb_dim:    int   = 0,
        discharge_idx:  int   = 3,
        discharge_ref:  float = 1.0,
    ):
        super().__init__()
        self.hidden         = hidden
        self.t_out          = t_out
        self.sar_emb_dim    = sar_emb_dim
        self.use_sar        = (sar_emb_dim > 0)
        self.discharge_idx  = discharge_idx
        self.discharge_ref  = discharge_ref
        self.use_dyn_edge   = (f_edge >= 5)

        # Learnable scale for sigmoid conductance — initialised to 3.0 so that
        # a 2× increase in discharge over reference gives conductance ≈ 0.95.
        if self.use_dyn_edge:
            self.conductance_scale = nn.Parameter(torch.tensor(3.0))

        # ── Input projection ───────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(f_dyn + f_static, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
        )

        # ── Temporal encoder ──────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_dropout = nn.Dropout(dropout)

        # ── SAR fusion (optional) ─────────────────────────────────────
        if self.use_sar:
            self.sar_fusion = nn.Sequential(
                nn.Linear(hidden + sar_emb_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ELU(),
            )

        # ── GATConv layers ─────────────────────────────────────────────
        self.gat1 = GATConv(hidden, hidden, heads=gat_heads, concat=True,
                            dropout=dropout, edge_dim=f_edge)
        self.norm1 = nn.LayerNorm(hidden * gat_heads)
        self.res1  = nn.Linear(hidden, hidden * gat_heads, bias=False)

        self.gat2 = GATConv(hidden * gat_heads, hidden, heads=gat_heads,
                            concat=False, dropout=dropout, edge_dim=f_edge)
        self.norm2 = nn.LayerNorm(hidden)
        self.res2  = nn.Linear(hidden * gat_heads, hidden, bias=False)

        self.gat3 = GATConv(hidden, hidden // 2, heads=2, concat=False,
                            dropout=dropout, edge_dim=f_edge)
        self.norm3 = nn.LayerNorm(hidden // 2)
        self.res3  = nn.Linear(hidden, hidden // 2, bias=False)

        # ── Output head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, t_out),
        )

    # ──────────────────────────────────────────────────────────────────
    def _dynamic_edge_attr(
        self,
        x_last:     torch.Tensor,   # [B, N, F_dyn]  last observed timestep
        edge_index: torch.Tensor,   # [2, E]
        edge_attr:  torch.Tensor,   # [E, 4]  static attributes
    ) -> torch.Tensor:
        """
        Append a discharge-based hydraulic conductance feature to edge_attr.

        conductance(i→j, t) = sigmoid( scale × (Q_i(t) / Q_ref  −  1) )

        Returns  [B, E, 5]  — first 4 dims are static, dim 4 is dynamic.
        """
        B = x_last.shape[0]
        E = edge_attr.shape[0]
        src = edge_index[0]   # [E] upstream node indices

        # Discharge at upstream node for every edge: [B, E]
        Q_src = x_last[:, src, self.discharge_idx]          # [B, E]

        # Sigmoid conductance: high when discharge >> reference
        conductance = torch.sigmoid(
            self.conductance_scale * (Q_src / self.discharge_ref - 1.0)
        )                                                     # [B, E]

        # Tile static attrs across batch, concat dynamic conductance
        static_tiled = edge_attr.unsqueeze(0).expand(B, -1, -1)  # [B, E, 4]
        dyn = conductance.unsqueeze(-1)                           # [B, E, 1]
        return torch.cat([static_tiled, dyn], dim=-1)             # [B, E, 5]

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_seq:      torch.Tensor,            # [B, T_in, N, F_dyn]
        node_attr:  torch.Tensor,            # [N, F_static]
        edge_index: torch.Tensor,            # [2, E]
        edge_attr:  torch.Tensor,            # [E, 4]  static
        sar_emb:    torch.Tensor | None = None,   # [B, N, sar_emb_dim]
    ) -> torch.Tensor:
        """Returns delta stage_anomaly [B, T_out, N]."""
        B, T_in, N, _ = x_seq.shape

        # ── Step 1: project each (timestep, node) vector ──────────────
        static_exp = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T_in, -1, -1)
        x_combined = torch.cat([x_seq, static_exp], dim=-1)
        x_reshaped = x_combined.permute(0, 2, 1, 3).reshape(B * N, T_in, -1)
        x_proj     = self.input_proj(x_reshaped)

        # ── Step 2: GRU temporal encoding ─────────────────────────────
        gru_out, _ = self.gru(x_proj)
        h          = self.gru_dropout(gru_out[:, -1, :])
        h          = h.reshape(B, N, self.hidden)

        # ── Step 3: SAR fusion ─────────────────────────────────────────
        if self.use_sar and sar_emb is not None:
            h = self.sar_fusion(torch.cat([h, sar_emb], dim=-1))

        # ── Step 4: build dynamic edge attributes ─────────────────────
        # Use last observed timestep for conductance computation
        x_last = x_seq[:, -1, :, :]                          # [B, N, F_dyn]
        if self.use_dyn_edge:
            # [B, E, 5] — batch-dependent edge features
            dyn_edge_attr = self._dynamic_edge_attr(x_last, edge_index, edge_attr)
        else:
            # Fall back: tile static attrs (no dynamic feature)
            dyn_edge_attr = edge_attr.unsqueeze(0).expand(B, -1, -1)

        # ── Step 5: GATConv batched message passing ────────────────────
        E = edge_attr.shape[0]
        h_flat  = h.reshape(B * N, self.hidden)

        offsets = torch.arange(B, device=edge_index.device) * N
        src_b   = (edge_index[0].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_b   = (edge_index[1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batched_ei   = torch.stack([src_b, dst_b], dim=0)          # [2, B×E]
        batched_ea   = dyn_edge_attr.reshape(B * E, -1)            # [B×E, f_edge]

        h1 = F.elu(self.norm1(
            self.gat1(h_flat, batched_ei, batched_ea) + self.res1(h_flat)
        ))
        h2 = F.elu(self.norm2(
            self.gat2(h1,    batched_ei, batched_ea) + self.res2(h1)
        ))
        h3 = F.elu(self.norm3(
            self.gat3(h2,    batched_ei, batched_ea) + self.res3(h2)
        ))

        h_graph = h3.reshape(B, N, self.hidden // 2)
        delta   = self.head(h_graph)
        return delta.permute(0, 2, 1)                               # [B, T_out, N]


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_in, N, T_out = 4, 32, 27, 4
    F_dyn, F_static, F_edge = 11, 7, 5
    SAR_DIM = 16
    E = 28

    edge_index = torch.randint(0, N, (2, E))
    edge_attr  = torch.randn(E, F_edge - 1)          # 4 static features
    node_attr  = torch.randn(N, F_static)
    x_seq      = torch.rand(B, T_in, N, F_dyn) * 2   # positive discharge values
    sar_emb    = torch.randn(B, N, SAR_DIM)

    model = STGNNDynEdge(
        f_dyn=F_dyn, f_static=F_static, f_edge=F_edge,
        hidden=64, gat_heads=2, gru_layers=2, t_out=T_out,
        sar_emb_dim=SAR_DIM, discharge_idx=3,
    )

    out = model(x_seq, node_attr, edge_index, edge_attr, sar_emb)
    assert out.shape == (B, T_out, N), f"Wrong shape: {out.shape}"
    print(f"Output: {tuple(out.shape)}  ✓")

    loss = out.sum(); loss.backward()
    print("Backward pass:  ✓")

    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}  (includes 1 learnable conductance_scale)")
    print("Smoke test passed.")
