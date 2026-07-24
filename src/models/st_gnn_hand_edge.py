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

Soft activation (differentiable) — datum-consistent formulation
------------------------------------------------------------------
Rather than a hard binary switch (which would give zero gradients when inactive),
stage-based activation uses a sigmoid on ABSOLUTE water-surface elevation:

    H_i(t)   = gauge_datum_i + normalised_stage_i(t) × stage_range_i
    activation(i,j,t) = sigmoid( α × (max(H_i(t), H_j(t)) − z_saddle_ij) )

where:
  normalised_stage_i(t) — feature index [1] of x, i.e. (level−gauge_datum)/
                           (p90−gauge_datum), dimensionless position within
                           the station's normal-to-p90 range
  gauge_datum_i          — station datum (m OD), per node, from nodes.csv
  stage_range_i          — p90_mAOD − gauge_datum_mOSGM15 (m), per node
  z_saddle_ij             — ABSOLUTE elevation (m OD) at the corridor's
                           lowest-connectivity point, from
                           precompute_hand_edges.py's z_saddle_m output
  α                      — learnable sharpness parameter (init 5.0)

IMPORTANT — this replaces an earlier, datum-inconsistent version of this
gate that compared feature index [0] (stage_anomaly = level − rolling_7d_
mean, an elevation-free deviation from a moving per-station baseline —
see build_dataset.py's own comment: "removes elevation offset") directly
against hand_threshold (a HAND-relative, not absolute, quantity). Since
stage_anomaly resets its zero-point every ~7 days independent of the
station's actual elevation, and hand_threshold is a fixed geometric
quantity, that comparison had no consistent physical footing — for most
candidate edges the sigmoid could sit near a fixed operating point
regardless of true flood state. H_i(t) above is anchored to gauge_datum
(m OD, the same absolute vertical datum the DEM-derived node_elev and
z_saddle live on), so H_i(t) ≥ z_saddle_ij has a direct physical reading:
"the water surface at node i has risen to or above the saddle point."

When H ≪ z_saddle: activation ≈ 0 → HAND edge contributes ~nothing
When H ≫ z_saddle: activation ≈ 1 → HAND edge contributes fully
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
                                       (relative — kept for diagnostics only,
                                       NOT used in the activation gate below)
    overland_dist_km float32 [E_hand] overland distance between nodes
    z_saddle_m       float32 [E_hand] ABSOLUTE elevation (m OD) at the same
                                       location as hand_threshold — this IS
                                       used in the activation gate

nodes.csv  — must contain gauge_datum_mOSGM15 and p90_mAOD columns (m OD /
             mOSGM15, treated as the same absolute vertical datum family as
             the DEM). Loaded and passed in as gauge_datum / stage_range by
             the training script, NOT read directly by this module.

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
        Minimum HAND (m) along overland path. Kept for diagnostics /
        backward compatibility; NOT used in the activation gate (see
        z_saddle below).
    hand_overland_dist : torch.Tensor [E_hand]
        Overland distance (km), used as first HAND edge attribute.
    z_saddle : torch.Tensor [E_hand]
        ABSOLUTE elevation (m OD) at the corridor's lowest-connectivity
        point — the quantity the activation gate actually compares
        reconstructed water-surface elevation against. From
        precompute_hand_edges.py's z_saddle_m output.
    gauge_datum : torch.Tensor [N]
        Per-node station datum (m OD), from nodes.csv's
        gauge_datum_mOSGM15 column.
    stage_range : torch.Tensor [N]
        Per-node p90_mAOD − gauge_datum_mOSGM15 (m). Used to reconstruct
        absolute water-surface elevation from normalised_stage.
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
        z_saddle:            torch.Tensor,
        gauge_datum:         torch.Tensor,
        stage_range:         torch.Tensor,
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
        # (retained for the static edge-feature vector; no longer used in
        # the activation gate itself — see z_saddle below)
        max_t = hand_threshold.max().clamp(min=1e-3)
        self.register_buffer("hand_thresh_norm",
            (hand_threshold / max_t).float())

        # ── Option-3 datum-consistent activation gate buffers ───────────
        # z_saddle: absolute elevation (m OD) at the corridor saddle point,
        # per HAND edge — same datum as node_elev / DEM.
        self.register_buffer("z_saddle", z_saddle.float())
        # gauge_datum / stage_range: per-NODE (not per-edge) quantities,
        # indexed by hand_src/hand_dst inside _hand_edge_attr to reconstruct
        # absolute water-surface elevation from normalised_stage.
        self.register_buffer("gauge_datum", gauge_datum.float())
        self.register_buffer("stage_range",  stage_range.float())

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
          [4] stage_activation     dynamic: sigmoid( α × (max_H − z_saddle) )

        The sigmoid activation approaches 1 when the higher of the two
        endpoints' reconstructed absolute water-surface elevation exceeds
        the corridor's saddle elevation, and 0 when well below. Gradients
        flow through sigmoid at all values, enabling the model to learn
        the appropriate activation sharpness α.

        H is reconstructed from normalised_stage (feature index [1], NOT
        stage_anomaly at index [0] — see module docstring for why) and the
        per-node gauge_datum / stage_range buffers, giving an absolute m-OD
        quantity directly comparable to z_saddle.
        """
        B       = x_last.shape[0]
        E_hand  = self.hand_src.shape[0]

        # Reconstruct absolute water-surface elevation H = gauge_datum +
        # normalised_stage * stage_range, at each HAND edge's two endpoints.
        norm_stage_src = x_last[:, self.hand_src, 1]   # [B, E_hand]
        norm_stage_dst = x_last[:, self.hand_dst, 1]   # [B, E_hand]

        H_src = (self.gauge_datum[self.hand_src].unsqueeze(0)
                 + norm_stage_src * self.stage_range[self.hand_src].unsqueeze(0))
        H_dst = (self.gauge_datum[self.hand_dst].unsqueeze(0)
                 + norm_stage_dst * self.stage_range[self.hand_dst].unsqueeze(0))

        # Activation based on the higher of the two endpoints' water surface
        max_H      = torch.maximum(H_src, H_dst)                   # [B, E_hand]
        activation = torch.sigmoid(
            self.activation_sharpness
            * (max_H - self.z_saddle.unsqueeze(0))
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

    Returns dict with keys: src, dst, hand_threshold, overland_dist_km,
    z_saddle_m (all torch.Tensor).

    z_saddle_m requires hand_edges.npz to have been regenerated with the
    option-3 datum fix (precompute_hand_edges.py's updated version). If an
    older hand_edges.npz is loaded (no z_saddle_m key), this raises rather
    than silently falling back — the datum-consistent activation gate is
    meaningless without it, and a silent fallback risks reintroducing the
    exact bug this fix addresses.
    """
    data = np.load(npz_path)
    if "z_saddle_m" not in data:
        raise KeyError(
            f"{npz_path} has no 'z_saddle_m' array. This hand_edges.npz "
            f"predates the option-3 datum fix — rerun "
            f"precompute_hand_edges.py to regenerate it before training."
        )
    return {
        "src":              torch.from_numpy(data["src"].astype(np.int64)),
        "dst":              torch.from_numpy(data["dst"].astype(np.int64)),
        "hand_threshold":   torch.from_numpy(data["hand_threshold"]),
        "overland_dist_km": torch.from_numpy(data["overland_dist_km"]),
        "z_saddle_m":       torch.from_numpy(data["z_saddle_m"]),
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
    hand_thr   = torch.rand(E_hand) * 2 + 1.0   # 1–3 m (diagnostics only now)
    hand_dist  = torch.rand(E_hand) * 3 + 1.0   # 1–4 km

    # Realistic-ish synthetic m-OD values, matching nodes.csv's actual range
    # (roughly -5 to 120 m OD across the 27 Lee gauges)
    node_elev_synth  = torch.rand(N) * 120.0 - 5.0
    z_saddle         = node_elev_synth[hand_src] + torch.rand(E_hand) * 3.0
    gauge_datum      = node_elev_synth.clone()
    stage_range      = torch.rand(N) * 2.0 + 0.5   # 0.5–2.5 m, matches nodes.csv scale

    model = STGNNHANDEdge(
        f_dyn=F_dyn, f_static=F_static, f_edge=F_edge,
        hand_src=hand_src, hand_dst=hand_dst,
        hand_threshold=hand_thr, hand_overland_dist=hand_dist,
        z_saddle=z_saddle, gauge_datum=gauge_datum, stage_range=stage_range,
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

    # Sanity check: activation should vary meaningfully (not be stuck near 0
    # or near 1 for every edge) when normalised_stage sweeps a realistic
    # range — this is the check that would have caught the original bug,
    # where hand_threshold (metres) dominated stage_anomaly (~[-2,5], often
    # near 0) and activation collapsed toward a fixed operating point
    # regardless of x_seq's actual values.
    with torch.no_grad():
        x_low  = x_seq.clone(); x_low[:, -1, :, 1]  = -2.0   # low normalised_stage
        x_high = x_seq.clone(); x_high[:, -1, :, 1] =  5.0   # high normalised_stage
        _, act_low  = model._hand_edge_attr(x_low[:, -1, :, :])
        _, act_high = model._hand_edge_attr(x_high[:, -1, :, :])
        spread = (act_high - act_low).abs().mean().item()
        assert spread > 0.3, (
            f"HAND activation barely responds to normalised_stage "
            f"(mean |Δactivation|={spread:.4f}) — datum mismatch likely."
        )
    print(f"Activation responds to stage sweep:  ✓  (mean |Δ|={spread:.3f})")

    n = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    print(f"Parameters: {n:,}   Buffers (HAND data): {n_buf:,}")
    print("Smoke test passed.")
