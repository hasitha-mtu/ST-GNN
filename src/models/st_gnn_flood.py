"""
st_gnn_flood.py  –  ST-GNN flood model with optional SAR-FNO embedding fusion
===============================================================================
Architecture
------------
  1. Node input projection
       Linear(F_dyn + F_static) → hidden_dim, applied at every timestep

  2. Temporal encoder
       Per-node GRU (shared weights, num_layers=gru_layers)
       Input window [T_in] of projected node features → hidden state [hidden]

  3. SAR fusion (optional, quasi-static)
       If sar_emb [N, sar_emb_dim] is provided:
         Concat([gru_out, sar_emb]) → Linear(hidden + sar_emb_dim → hidden) + LayerNorm
       If absent (sar_emb is None or sar_emb_dim=0):
         GRU output passes through unchanged (compatible with baseline runs)

  4. Graph message passing
       GATConv layer 1: hidden → hidden × gat_heads  (concat=True)
       GATConv layer 2: hidden × gat_heads → hidden  (concat=False)
       GATConv layer 3: hidden → hidden // 2          (concat=False)
       Each layer has residual connection + LayerNorm + ELU

  5. Output head
       Linear(hidden // 2 → hidden // 4) + ReLU
       Linear(hidden // 4 → T_out)
       Predicts DELTA stage_anomaly (change from last observed value)

Edge attributes (river_dist_km, area_ratio, elev_drop_m, same_tributary)
are passed to all three GATConv layers via edge_dim. Attention coefficients
are therefore conditioned on the hydraulic relationship between nodes, not
just on learned node similarity.

Backward compatibility
----------------------
  sar_emb_dim=0 (default) reproduces the original no-SAR baseline exactly.
  No changes to the baseline training scripts are required — they simply
  omit the sar_emb argument and the model behaves as before.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class STGNNFloodModel(nn.Module):
    """
    Spatio-temporal GNN for multi-step water stage delta forecasting.

    Parameters
    ----------
    f_dyn : int
        Number of dynamic input features per node per timestep.
        (e.g. 3 for [stage, delta_stage, rainfall])
    f_static : int
        Number of static node attributes from nodes.csv.
        (e.g. 7 for the standard Lee graph schema)
    f_edge : int
        Number of edge attribute features from edges.csv.
        (e.g. 4 for [river_dist_km, area_ratio, elev_drop_m, same_tributary])
    hidden : int
        Internal hidden dimension. Default 64.
    gat_heads : int
        Number of attention heads for GATConv layers 1 and 3. Default 2.
        Layer 1 uses gat_heads with concat=True → hidden × gat_heads.
        Layers 2 and 3 use gat_heads=1 implicitly (concat=False averages).
    gru_layers : int
        Depth of the per-node GRU. Default 2.
    t_out : int
        Number of forecast horizons. Default 4.
    dropout : float
        Dropout probability applied after GRU and inside GATConv. Default 0.1.
    sar_emb_dim : int
        Dimension of SAR node embeddings from SARFNOEncoder. Default 0.
        Set to 16 when running with SAR. Setting to 0 disables the fusion
        layer entirely (baseline mode).
    """

    def __init__(
        self,
        f_dyn:       int,
        f_static:    int,
        f_edge:      int,
        hidden:      int  = 64,
        gat_heads:   int  = 2,
        gru_layers:  int  = 2,
        t_out:       int  = 4,
        dropout:     float = 0.1,
        sar_emb_dim: int  = 0,
    ):
        super().__init__()
        self.hidden      = hidden
        self.t_out       = t_out
        self.sar_emb_dim = sar_emb_dim
        self.use_sar     = (sar_emb_dim > 0)

        # ── 1. Node input projection (applied per-timestep) ────────────
        # Projects [dynamic features ‖ static attributes] → hidden
        self.input_proj = nn.Sequential(
            nn.Linear(f_dyn + f_static, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
        )

        # ── 2. Temporal encoder ────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=True,    # input: [B×N, T_in, hidden]
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_dropout = nn.Dropout(dropout)

        # ── 3. SAR fusion (optional) ───────────────────────────────────
        if self.use_sar:
            self.sar_fusion = nn.Sequential(
                nn.Linear(hidden + sar_emb_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ELU(),
            )

        # ── 4. GATConv layers ──────────────────────────────────────────
        # Layer 1: expand — hidden → hidden × heads (concat=True)
        self.gat1 = GATConv(
            in_channels=hidden,
            out_channels=hidden,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
            edge_dim=f_edge,
        )
        self.norm1 = nn.LayerNorm(hidden * gat_heads)
        self.res1  = nn.Linear(hidden, hidden * gat_heads, bias=False)

        # Layer 2: compress — hidden×heads → hidden (concat=False averages)
        self.gat2 = GATConv(
            in_channels=hidden * gat_heads,
            out_channels=hidden,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            edge_dim=f_edge,
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.res2  = nn.Linear(hidden * gat_heads, hidden, bias=False)

        # Layer 3: reduce — hidden → hidden//2 (2 heads, concat=False)
        self.gat3 = GATConv(
            in_channels=hidden,
            out_channels=hidden // 2,
            heads=2,
            concat=False,
            dropout=dropout,
            edge_dim=f_edge,
        )
        self.norm3 = nn.LayerNorm(hidden // 2)
        self.res3  = nn.Linear(hidden, hidden // 2, bias=False)

        # ── 5. Output head ─────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, t_out),
        )

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_seq:      torch.Tensor,           # [B, T_in, N, F_dyn]
        node_attr:  torch.Tensor,           # [N, F_static]
        edge_index: torch.Tensor,           # [2, E]
        edge_attr:  torch.Tensor,           # [E, F_edge]
        sar_emb:    torch.Tensor | None = None,  # [B, N, sar_emb_dim] or None
    ) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Delta stage_anomaly predictions, shape [B, T_out, N].
            Add last observed stage to recover absolute predictions.
        """
        B, T_in, N, F_dyn = x_seq.shape

        # ── Step 1: project each (timestep, node) feature vector ──────
        # Broadcast static node attributes across batch and time
        # node_attr: [N, F_static] → [B, T_in, N, F_static]
        node_static = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T_in, -1, -1)

        # Concat dynamic + static → [B, T_in, N, F_dyn+F_static]
        x_combined = torch.cat([x_seq, node_static], dim=-1)

        # Reshape for input_proj: [B×N, T_in, F_dyn+F_static]
        x_reshaped = x_combined.permute(0, 2, 1, 3).reshape(B * N, T_in, -1)

        # Project each timestep: [B×N, T_in, hidden]
        x_proj = self.input_proj(x_reshaped)

        # ── Step 2: GRU temporal encoding ─────────────────────────────
        # GRU output: [B×N, T_in, hidden], take final hidden state
        gru_out, _ = self.gru(x_proj)          # [B×N, T_in, hidden]
        h = gru_out[:, -1, :]                   # [B×N, hidden]
        h = self.gru_dropout(h)

        # Reshape back: [B×N, hidden] → [B, N, hidden]
        h = h.reshape(B, N, self.hidden)

        # ── Step 3: SAR fusion (quasi-static, optional) ────────────────
        if self.use_sar and sar_emb is not None:
            # sar_emb: [B, N, sar_emb_dim]
            h = self.sar_fusion(torch.cat([h, sar_emb], dim=-1))  # [B, N, hidden]

        # ── Step 4: GATConv message passing ───────────────────────────
        # GATConv operates on [N, C] node features; we loop over batch.
        # For the River Lee network (N≤28), this loop is not a bottleneck.
        # For larger graphs, consider batching with torch_geometric Batch.
        out_list = []
        for b in range(B):
            hb = h[b]   # [N, hidden]

            # Layer 1: expand (hidden → hidden × gat_heads)
            h1 = F.elu(self.norm1(
                self.gat1(hb, edge_index, edge_attr) + self.res1(hb)
            ))                                # [N, hidden × gat_heads]

            # Layer 2: compress (hidden×heads → hidden)
            h2 = F.elu(self.norm2(
                self.gat2(h1, edge_index, edge_attr) + self.res2(h1)
            ))                                # [N, hidden]

            # Layer 3: reduce (hidden → hidden // 2)
            h3 = F.elu(self.norm3(
                self.gat3(h2, edge_index, edge_attr) + self.res3(h2)
            ))                                # [N, hidden // 2]

            out_list.append(h3)

        h_graph = torch.stack(out_list, dim=0)  # [B, N, hidden // 2]

        # ── Step 5: output head ────────────────────────────────────────
        delta = self.head(h_graph)              # [B, N, T_out]
        return delta.permute(0, 2, 1)           # [B, T_out, N]


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cpu")
    B, T_in, N, T_out = 4, 32, 27, 4
    F_dyn, F_static, F_edge = 3, 7, 4
    SAR_DIM = 16

    # Random graph topology (directed river network stub)
    E = 30
    edge_index = torch.randint(0, N, (2, E))
    edge_attr  = torch.randn(E, F_edge)
    node_attr  = torch.randn(N, F_static)
    x_seq      = torch.randn(B, T_in, N, F_dyn)
    sar_emb    = torch.randn(B, N, SAR_DIM)

    # ── Baseline (no SAR) ──────────────────────────────────────────────
    model_base = STGNNFloodModel(
        f_dyn=F_dyn, f_static=F_static, f_edge=F_edge,
        hidden=64, gat_heads=2, gru_layers=2, t_out=T_out,
        sar_emb_dim=0,
    )
    out_base = model_base(x_seq, node_attr, edge_index, edge_attr)
    assert out_base.shape == (B, T_out, N), f"Baseline shape wrong: {out_base.shape}"
    print(f"Baseline output: {tuple(out_base.shape)}  ✓")

    # ── With SAR embedding ─────────────────────────────────────────────
    model_sar = STGNNFloodModel(
        f_dyn=F_dyn, f_static=F_static, f_edge=F_edge,
        hidden=64, gat_heads=2, gru_layers=2, t_out=T_out,
        sar_emb_dim=SAR_DIM,
    )
    out_sar = model_sar(x_seq, node_attr, edge_index, edge_attr, sar_emb)
    assert out_sar.shape == (B, T_out, N), f"SAR shape wrong: {out_sar.shape}"
    print(f"SAR model output: {tuple(out_sar.shape)}  ✓")

    # ── Parameter counts ──────────────────────────────────────────────
    n_base = sum(p.numel() for p in model_base.parameters())
    n_sar  = sum(p.numel() for p in model_sar.parameters())
    print(f"Parameters — baseline: {n_base:,}  SAR model: {n_sar:,}  "
          f"delta: +{n_sar - n_base:,}")

    # ── Backward pass ─────────────────────────────────────────────────
    loss = out_sar.sum()
    loss.backward()
    print("Backward pass:  ✓")
    print("All checks passed.")
