"""
st_gnn_hand_edge.py  –  ST-GNN with HAND-based dynamic topology (Phase 2)
==========================================================================
Architecture
------------
Extends STGNNDynEdge (Phase 1) by adding structurally new edges that activate
when observed stage exceeds a precomputed HAND threshold along the overland
path between non-adjacent gauge node pairs.

Two edge classes coexist in every forward pass:

  Class A  — permanent river-network edges (same as static baseline)
             [river_dist_km, area_ratio, elev_drop_m, same_tributary, conductance]
             conductance is the Phase 1 discharge-based dynamic feature.

  Class B  — HAND candidate edges (cross-tributary floodplain connections)
             [overland_dist_norm, hand_threshold_norm, 0, 0, stage_activation]
             stage_activation is the soft switch: approaches 1 when stage
             exceeds the HAND threshold, 0 when stage is well below it.

Soft activation (differentiable)
---------------------------------
Rather than a hard binary switch (which would give zero gradients when inactive),
stage-based activation uses a sigmoid:

    activation(i,j,t) = sigmoid( α × (max(h_i(t), h_j(t)) − τ_ij) )

where:
  h_i(t), h_j(t)  — normalised stage anomaly at the two endpoints  [dim 0]
  τ_ij            — HAND threshold for this pair (precomputed from DEM)
  α               — learnable sharpness parameter (init 5.0)

When stage ≪ τ_ij: activation ≈ 0 → HAND edge contributes ~nothing
When stage ≫ τ_ij: activation ≈ 1 → HAND edge contributes fully
Gradient flows through sigmoid at all times during training.

Physical basis
--------------
HAND (Height Above Nearest Drainage, Nobre et al. 2011) measures the vertical
distance from any terrain point to its nearest drainage channel following the
D8 flow direction.  The minimum HAND value along the overland path between two
non-adjacent drainage basins is the stage height at which those basins become
hydraulically connected through floodplain inundation.  The threshold distance
of 5 km for candidate pair search is consistent with Godbout et al. (2019) and
Zheng et al. (2018) recommended HAND reach discretisation scales and is
calibrated to the Lee catchment tributary spacing (~3–5 km between principal
confluences, as documented by the Irish Examiner, 2024).

Precomputed data
----------------
hand_edges.npz  — written by precompute_hand_edges.py, contains:
    src              int32  [E_hand]   source node indices
    dst              int32  [E_hand]   destination node indices
    hand_threshold   float32 [E_hand] minimum HAND (m) along overland path
    overland_dist_km float32 [E_hand] overland distance between nodes

References
----------
Nobre, A.D. et al. (2011). Height Above the Nearest Drainage – a hydrologically
    relevant new terrain model. J. Hydrol.
Aristizabal, F. et al. (2023). Extending HAND to model multiple fluvial sources.
    Water Resources Research.
Godbout, L. et al. (2019). Error assessment for HAND inundation mapping.
Zheng, X. et al. (2018). Enhancing HAND flood inundation mapping. J. Hydrol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np


class STGNNHANDEdge(nn.Module):
    """
    ST-GNN with HAND-based dynamic topology.

    Parameters
    ----------
    f_dyn : int
        Dynamic input features (5 or 11).
    f_static : int
        Static node attributes.
    f_edge : int
        Edge attribute dimension — must be 5 (4 static + 1 dynamic).
    hand_src : torch.Tensor [E_hand]
        Source node indices for HAND candidate edges.
    hand_dst : torch.Tensor [E_hand]
        Destination node indices for HAND candidate edges.
    hand_threshold : torch.Tensor [E_hand]
        Minimum HAND (m) along overland path; normalised stage threshold.
    hand_overland_dist : torch.Tensor [E_hand]
        Overland distance (km), used as first HAND edge attribute.
    hidden, gat_heads, gru_layers, t_out, dropout : same as STGNNFloodModel.
    sar_emb_dim : int
        SAR embedding dimension. 0 = no SAR.
    discharge_idx : int
        Column index of discharge in dynamic feature vector. Default 3.
    discharge_ref : float
        Reference discharge for Phase 1 conductance. Default 1.0.
    activation_sharpness : float
        Initial value of learnable sigmoid sharpness α. Default 5.0.
    """

    def __init__(
        self,
        f_dyn:               int,
        f_static:            int,
        f_edge:              int,
        hand_src:            torch.Tensor,
        hand_dst:            torch.Tensor,
        hand_threshold:      torch.Tensor,
        hand_overland_dist:  torch.Tensor,
        hidden:              int   = 64,
        gat_heads:           int   = 2,
        gru_layers:          int   = 2,
        t_out:               int   = 4,
        dropout:             float = 0.1,
        sar_emb_dim:         int   = 0,
        discharge_idx:       int   = 3,
        discharge_ref:       float = 1.0,
        activation_sharpness: float = 5.0,
    ):
        super().__init__()
        self.hidden         = hidden
        self.t_out          = t_out
        self.sar_emb_dim    = sar_emb_dim
        self.use_sar        = (sar_emb_dim > 0)
        self.discharge_idx  = discharge_idx
        self.discharge_ref  = discharge_ref

        # ── Precomputed HAND edge tensors (registered as buffers so they
        #    move to GPU with .to(device) and are saved in checkpoints) ──
        self.register_buffer("hand_src",        hand_src.long())
        self.register_buffer("hand_dst",        hand_dst.long())
        self.register_buffer("hand_threshold",  hand_threshold.float())
        self.register_buffer("hand_dist_norm",
            (hand_overland_dist / 5.0).float())   # normalise by 5 km max

        # Normalise HAND threshold for use as edge attribute (0–1 range)
        max_t = hand_threshold.max().clamp(min=1e-3)
        self.register_buffer("hand_thresh_norm",
            (hand_threshold / max_t).float())

        # ── Learnable parameters ──────────────────────────────────────
        self.conductance_scale    = nn.Parameter(torch.tensor(3.0))
        self.activation_sharpness = nn.Parameter(
            torch.tensor(float(activation_sharpness))
        )

        # ── Input projection ──────────────────────────────────────────
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

        # ── SAR fusion ────────────────────────────────────────────────
        if self.use_sar:
            self.sar_fusion = nn.Sequential(
                nn.Linear(hidden + sar_emb_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ELU(),
            )

        # ── GATConv — f_edge=5 handles both edge classes ──────────────
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
    def _river_edge_attr(
        self,
        x_last:    torch.Tensor,   # [B, N, F]
        edge_index: torch.Tensor,  # [2, E_river]
        edge_attr:  torch.Tensor,  # [E_river, 4]
    ) -> torch.Tensor:
        """
        Build [B, E_river, 5] edge attributes for permanent river edges.
        Feature 4 = discharge conductance (Phase 1 mechanism).
        """
        B = x_last.shape[0]
        src = edge_index[0]
        Q_src = x_last[:, src, self.discharge_idx]          # [B, E]
        conductance = torch.sigmoid(
            self.conductance_scale * (Q_src / self.discharge_ref - 1.0)
        )
        static = edge_attr.unsqueeze(0).expand(B, -1, -1)  # [B, E, 4]
        return torch.cat([static, conductance.unsqueeze(-1)], dim=-1)

    # ──────────────────────────────────────────────────────────────────
    def _hand_edge_attr(
        self,
        x_last: torch.Tensor,   # [B, N, F]
    ) -> torch.Tensor:
        """
        Build [B, E_hand, 5] edge attributes for HAND candidate edges.

        Feature layout:
          [0] overland_dist_norm   precomputed, static
          [1] hand_thresh_norm     precomputed, static
          [2] 0.0                  (no area_ratio for cross-tributary edges)
          [3] 0.0                  (same_tributary = 0 by construction)
          [4] stage_activation     dynamic: sigmoid( α × (max_stage − τ) )

        The sigmoid activation approaches 1 when the higher of the two
        endpoint stages exceeds the HAND threshold, and 0 when well below.
        Gradients flow through sigmoid at all values, enabling the model
        to learn the appropriate activation sharpness α.
        """
        B       = x_last.shape[0]
        E_hand  = self.hand_src.shape[0]

        h_src = x_last[:, self.hand_src, 0]   # [B, E_hand] stage at source
        h_dst = x_last[:, self.hand_dst, 0]   # [B, E_hand] stage at dest

        # Activation based on the higher of the two endpoint stages
        max_stage  = torch.maximum(h_src, h_dst)                  # [B, E_hand]
        activation = torch.sigmoid(
            self.activation_sharpness
            * (max_stage - self.hand_threshold.unsqueeze(0))
        )                                                          # [B, E_hand]

        # Static components tiled across batch
        dist_t   = self.hand_dist_norm.unsqueeze(0).expand(B, -1)    # [B, E]
        thresh_t = self.hand_thresh_norm.unsqueeze(0).expand(B, -1)  # [B, E]
        zeros    = torch.zeros(B, E_hand, device=x_last.device)      # [B, E]

        return torch.stack(
            [dist_t, thresh_t, zeros, zeros, activation], dim=-1
        )                                                            # [B, E, 5]

    # ──────────────────────────────────────────────────────────────────
    def _build_combined_graph(
        self,
        x_last:    torch.Tensor,    # [B, N, F]
        edge_index: torch.Tensor,   # [2, E_river]
        edge_attr:  torch.Tensor,   # [E_river, 4]
        B:         int,
        N:         int,
    ):
        """
        Concatenate river network edges and HAND candidate edges into a
        single batched graph for efficient GATConv.

        Returns
        -------
        batched_edge_index : [2, B × (E_river + E_hand)]
        batched_edge_attr  : [B × (E_river + E_hand), 5]
        E_total            : int — edges per graph instance
        """
        E_river = edge_index.shape[1]
        E_hand  = self.hand_src.shape[0]

        # Edge attributes
        river_ea = self._river_edge_attr(x_last, edge_index, edge_attr)
        hand_ea  = self._hand_edge_attr(x_last)

        # Combined edge index for one graph: river edges then HAND edges
        hand_ei = torch.stack(
            [self.hand_src, self.hand_dst], dim=0
        )                                                     # [2, E_hand]
        combined_ei = torch.cat([edge_index, hand_ei], dim=1) # [2, E_total]
        combined_ea = torch.cat([river_ea, hand_ea], dim=1)   # [B, E_total, 5]
        E_total = E_river + E_hand

        # Batch offsets
        offsets = torch.arange(B, device=edge_index.device) * N  # [B]
        src_b = (combined_ei[0].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_b = (combined_ei[1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batched_ei = torch.stack([src_b, dst_b], dim=0)          # [2, B×E_total]
        batched_ea = combined_ea.reshape(B * E_total, 5)          # [B×E_total, 5]

        return batched_ei, batched_ea, E_total

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_seq:      torch.Tensor,
        node_attr:  torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
        sar_emb:    torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns delta stage_anomaly [B, T_out, N]."""
        B, T_in, N, _ = x_seq.shape

        # ── Step 1: input projection ───────────────────────────────────
        static_exp = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T_in, -1, -1)
        x_combined = torch.cat([x_seq, static_exp], dim=-1)
        x_reshaped = x_combined.permute(0, 2, 1, 3).reshape(B * N, T_in, -1)
        x_proj     = self.input_proj(x_reshaped)

        # ── Step 2: GRU ───────────────────────────────────────────────
        gru_out, _ = self.gru(x_proj)
        h          = self.gru_dropout(gru_out[:, -1, :])
        h          = h.reshape(B, N, self.hidden)

        # ── Step 3: SAR fusion ─────────────────────────────────────────
        if self.use_sar and sar_emb is not None:
            h = self.sar_fusion(torch.cat([h, sar_emb], dim=-1))

        # ── Step 4: build combined dynamic graph ───────────────────────
        x_last = x_seq[:, -1, :, :]
        batched_ei, batched_ea, E_total = self._build_combined_graph(
            x_last, edge_index, edge_attr, B, N
        )

        # ── Step 5: GATConv ────────────────────────────────────────────
        h_flat = h.reshape(B * N, self.hidden)

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
        return delta.permute(0, 2, 1)                              # [B, T_out, N]


# ═══════════════════════════════════════════════════════════════════════
#  Helper: load precomputed HAND edges
# ═══════════════════════════════════════════════════════════════════════

def load_hand_edges(npz_path: str) -> dict:
    """
    Load HAND candidate edges from precompute_hand_edges.py output.

    Returns dict with keys: src, dst, hand_threshold, overland_dist_km
    (all torch.Tensor).
    """
    data = np.load(npz_path)
    return {
        "src":              torch.from_numpy(data["src"].astype(np.int64)),
        "dst":              torch.from_numpy(data["dst"].astype(np.int64)),
        "hand_threshold":   torch.from_numpy(data["hand_threshold"]),
        "overland_dist_km": torch.from_numpy(data["overland_dist_km"]),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_in, N, T_out = 4, 32, 27, 4
    F_dyn, F_static, F_edge = 11, 7, 5
    SAR_DIM = 16
    E_river, E_hand = 28, 12    # 12 synthetic HAND candidate edges

    edge_index  = torch.randint(0, N, (2, E_river))
    edge_attr   = torch.randn(E_river, F_edge - 1)
    node_attr   = torch.randn(N, F_static)
    x_seq       = torch.rand(B, T_in, N, F_dyn) * 2
    sar_emb     = torch.randn(B, N, SAR_DIM)

    # Synthetic HAND edges — all distinct pairs, threshold 1–3 m normalised
    hand_pairs = torch.randperm(N * (N - 1))[:E_hand * 2].reshape(E_hand, 2)
    hand_src   = hand_pairs[:, 0] % N
    hand_dst   = hand_pairs[:, 1] % N
    hand_thr   = torch.rand(E_hand) * 2 + 1.0   # 1–3 m
    hand_dist  = torch.rand(E_hand) * 3 + 1.0   # 1–4 km

    model = STGNNHANDEdge(
        f_dyn=F_dyn, f_static=F_static, f_edge=F_edge,
        hand_src=hand_src, hand_dst=hand_dst,
        hand_threshold=hand_thr, hand_overland_dist=hand_dist,
        hidden=64, gat_heads=2, gru_layers=2, t_out=T_out,
        sar_emb_dim=SAR_DIM, discharge_idx=3,
    )

    out = model(x_seq, node_attr, edge_index, edge_attr, sar_emb)
    assert out.shape == (B, T_out, N), f"Wrong shape: {out.shape}"
    print(f"Output: {tuple(out.shape)}  ✓")

    out.sum().backward()
    print("Backward pass:  ✓")

    # Verify activation sharpness gradient flows
    assert model.activation_sharpness.grad is not None
    print("Sharpness gradient:  ✓")

    n = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    print(f"Parameters: {n:,}   Buffers (HAND data): {n_buf:,}")
    print("Smoke test passed.")
