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

        # ── HAND message gate (Fix 2.1) ──────────────────────────────
        # Linear transform applied to source node features for HAND edges,
        # followed by per-edge multiplication with stage-based activation.
        # This ensures activation directly gates the HAND edge message,
        # rather than being one projected input to the GATConv attention.
        # hidden → hidden (same dim as GATConv layer 1 input)
        self.hand_msg_linear = nn.Linear(hidden, hidden, bias=False)
        self.hand_msg_norm   = nn.LayerNorm(hidden)
        # Learnable scalar controlling HAND message magnitude vs river msgs
        # Initialised at 0 so training begins without HAND contribution
        # and ramps up only when activation provides reliable signal.
        self.hand_scale = nn.Parameter(torch.zeros(1))

        # Projection: hand_agg comes out of hand_msg_norm at [B*N, hidden=64].
        # gat1 (concat=True, heads=gat_heads) produces h1_river at
        # [B*N, hidden*gat_heads=128].  Without this projection, adding
        # them on line 386 raises RuntimeError (128 ≠ 64).
        self.hand_proj = nn.Linear(hidden, hidden * gat_heads, bias=False)

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

        # Fix 2.1: activation is NO LONGER an edge attribute passed to GATConv.
        # It becomes an explicit multiplicative gate applied in forward().
        # Return 4-feature static attrs (compatible with river edge format)
        # PLUS activation as a separate return value.
        static_ea = torch.stack(
            [dist_t, thresh_t, zeros, zeros], dim=-1
        )                                                            # [B, E, 4]
        return static_ea, activation                                 # [B,E,4], [B,E]

    # ──────────────────────────────────────────────────────────────────
    def _build_river_graph(
        self,
        x_last:    torch.Tensor,    # [B, N, F]
        edge_index: torch.Tensor,   # [2, E_river]
        edge_attr:  torch.Tensor,   # [E_river, 4]
        B:         int,
        N:         int,
    ):
        """
        Build batched graph for RIVER edges only.

        Fix 2.1: HAND edges are processed separately in forward() with
        explicit activation gating. This method returns river edges only.

        Returns
        -------
        batched_edge_index : [2, B × E_river]
        batched_edge_attr  : [B × E_river, 5]  (4 static + conductance)
        """
        E_river = edge_index.shape[1]
        river_ea = self._river_edge_attr(x_last, edge_index, edge_attr)  # [B, E, 5]

        offsets = torch.arange(B, device=edge_index.device) * N
        src_b = (edge_index[0].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_b = (edge_index[1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batched_ei = torch.stack([src_b, dst_b], dim=0)   # [2, B×E_river]
        batched_ea = river_ea.reshape(B * E_river, 5)      # [B×E_river, 5]

        return batched_ei, batched_ea

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

        # ── Step 4: build river-only batched graph ─────────────────────
        x_last = x_seq[:, -1, :, :]
        batched_ei, batched_ea = self._build_river_graph(
            x_last, edge_index, edge_attr, B, N
        )

        # ── Step 4b: compute HAND activation and gated messages ────────
        # Fix 2.1: activation explicitly gates HAND messages.
        # This implements dynamic topology: when stage < τ, activation→0
        # and the HAND edge contributes nothing regardless of attention.
        # When stage > τ, activation→1 and the full message is aggregated.
        hand_ea_static, hand_activation = self._hand_edge_attr(x_last)
        # hand_activation: [B, E_hand] ∈ (0,1)

        # ── Step 5: GATConv on river graph ─────────────────────────────
        h_flat = h.reshape(B * N, self.hidden)

        h1_river = self.gat1(h_flat, batched_ei, batched_ea)  # [B*N, hidden*heads]

        # ── Step 5b: activation-gated HAND messages ────────────────────
        # For each HAND edge (src→dst), compute a linear transform of the
        # source node feature, then scale by activation before aggregating
        # to the destination node.  Gradient flows through both the linear
        # transform and through activation (which depends on stage).
        h_for_hand = h.reshape(B * N, self.hidden)   # same as h_flat

        # Batch-expand HAND edge indices
        offsets_h   = torch.arange(B, device=edge_index.device) * N
        hand_src_b  = (self.hand_src.unsqueeze(0) + offsets_h.unsqueeze(1)).reshape(-1)
        hand_dst_b  = (self.hand_dst.unsqueeze(0) + offsets_h.unsqueeze(1)).reshape(-1)
        E_hand      = self.hand_src.shape[0]

        # Source features for each HAND edge: [B*E_hand, hidden]
        src_feat    = h_for_hand[hand_src_b]
        hand_msg    = self.hand_msg_linear(src_feat)           # [B*E_hand, hidden]

        # Gate by activation: [B, E_hand] → [B*E_hand, 1]
        gate_flat   = hand_activation.reshape(B * E_hand, 1)  # [B*E_hand, 1]
        hand_msg_gated = hand_msg * gate_flat                  # explicit gate

        # Scatter-sum to destination nodes: [B*N, hidden]
        hand_agg = torch.zeros_like(h_for_hand)
        hand_dst_exp = hand_dst_b.unsqueeze(-1).expand_as(hand_msg_gated)
        hand_agg.scatter_add_(0, hand_dst_exp, hand_msg_gated)
        hand_agg = self.hand_msg_norm(hand_agg)               # [B*N, hidden]

        # Combine river + gated HAND contributions.
        # hand_proj lifts hand_agg from [B*N, hidden=64] to
        # [B*N, hidden*gat_heads=128] so it matches h1_river and norm1.
        # hand_scale starts at 0; grows as training demonstrates HAND utility.
        h1_combined = (h1_river
                       + torch.sigmoid(self.hand_scale)
                       * self.hand_proj(hand_agg))
        h1 = F.elu(self.norm1(h1_combined + self.res1(h_flat)))

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
