"""
dfc_gnn_unified.py  –  Physically-Complete Dynamic Connectivity GNN (PC-DFC-GNN)
==================================================================================
Unifies the three physics mechanisms previously implemented and ablated
SEPARATELY across three different models:

  Mechanism                 Prior (separate) implementation      Status here
  ─────────────────────────  ──────────────────────────────────  ──────────────────
  Dynamic edge WEIGHT        STGNNDynEdge — Manning conductance,  Folded into
                              sigmoid(scale·(Q/Q_ref − 1)),        edge_attr, feeds
                              log-transformed, 5th edge feature    the W_e projection
                              consumed by GATConv's edge_dim       (same math)

  Dynamic TOPOLOGY            STGNNHANDEdge — HAND-triggered       Folded into the
                              cross-tributary edges, activation    attention score as
                              gate applied AFTER GATConv via a     a log-additive gate
                              separate hand_msg_linear/hand_proj/  (see design note
                              hand_scale side-path (Fix 2.1)        below) — no side-path

  Hard directional GATE       DFCGNNFlood — PhysicallyBiasedGATConv Unchanged: same
                              log-additive elevation gate           log-additive gate

No study in the reviewed 38-study corpus combines all three (verified by keyword
screen of literature_v2.xlsx — see conversation log). The closest single
comparator is HydroGAT (Sarkar et al. 2025): GAT + physically-routed edges, but
topology is fixed once built (no runtime HAND-threshold switching), no hard
directional gate, no Manning-discharge conductance term.

Design decision — one gating mechanism, not two
-------------------------------------------------
STGNNHANDEdge applied HAND activation as an EXPLICIT multiplicative gate
computed outside GATConv's attention (Fix 2.1 in that file), because folding
activation into edge_attr risked the learned W_e projection assigning it a
negative or negligible weight — exactly the failure mode STGNNDynEdge's
log_conductance fix and DFCGNNFlood's log-additive elevation gate were both
designed to avoid.

Since all three prior fixes converge on the same underlying answer — a
log-additive term added to the pre-softmax attention score guarantees a
monotonic, sign-safe gate regardless of what the learned projection does —
this version applies ONE consistent mechanism for both hard gates instead of
maintaining two different code paths:

    score_ij  = LeakyReLU(a^T [Wq·h_i ‖ Wk·h_j ‖ We·edge_attr_ij])
    score_ij += log(elevation_gate_ij + eps)      # always active, Class A + B
    score_ij += log(hand_activation_ij + eps)     # ≡ log(1) = 0 for Class A edges
    alpha     = scatter_softmax(score, dst)

This removes STGNNHANDEdge's separate hand_msg_linear / hand_msg_norm /
hand_proj / hand_scale side-path entirely (including the 64→128 dimension
mismatch fix that machinery needed). HAND edges become first-class members
of the SAME attention computation as river edges, differentiated only by
their edge_attr values and their per-edge activation gate. This is a small
but real methodological improvement over STGNNHANDEdge's original
implementation, worth stating explicitly in the methods section.

Edge set
--------
  Class A — permanent river-network edges (topology from edges.csv, the same
            28-edge set STGNNDynEdge/STGNNHANDEdge use — see module docstring
            in the training script for why this topology is kept constant
            across the ablation ladder rather than switching to DFCGNNFlood's
            702-edge dense set)
            edge_attr = [river_dist_norm, area_ratio, elev_drop_norm,
                          same_tributary, log_conductance]
            hand_activation ≡ 1 (always active)

  Class B — HAND candidate edges (from hand_edges.npz, precomputed by
            precompute_hand_edges.py)
            edge_attr = [overland_dist_norm, hand_thresh_norm, 0, 0, 0]
            hand_activation = sigmoid(α · (max(stage_i, stage_j) − τ_ij))

Both classes carry node_elev (from edge_features.npz, borrowed from the
DFC-GNN data pipeline) for the hard directional gate, and both are
concatenated into one edge_index/edge_attr pair before a single
UnifiedPhysicalGATConv layer runs — one attention computation over the
combined graph, not two separate passes.

Output head
-----------
Dual-head, unchanged from DFCGNNFlood: stage regression (primary) + flood
classification (auxiliary, BCE with bankfull-derived labels). HANDDecoder
from dfc_gnn.py is reusable as-is against this model's stage_pred output —
no changes needed there.

Usage
-----
    from models.dfc_gnn_unified import DFCGNNUnified

    model = DFCGNNUnified(
        n_nodes=27, f_dyn=11, d_model=64, n_heads=4, T_out=4,
        edge_index=edge_index,            # [2, E_river] from load_graph
        edge_attr_static=edge_attr,       # [E_river, 4] from load_graph
        node_elev=node_elev,              # [N] from load_edge_features
        hand_src=hand_data["src"], hand_dst=hand_data["dst"],
        hand_threshold=hand_data["hand_threshold"],
        hand_overland_dist=hand_data["overland_dist_km"],
        discharge_idx=3, discharge_ref=q_ref,
    )
    stage_pred, flood_logits = model(x_seq)
    # stage_pred:   [B, T_out, N]
    # flood_logits: [B, N]
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═════════════════════════════════════════════════════════════════════
# 1. Unified Physically-Biased GAT Convolution
# ═════════════════════════════════════════════════════════════════════

class UnifiedPhysicalGATConv(nn.Module):
    """
    Physically-biased GAT convolution with three simultaneous mechanisms:

      1. Continuous dynamic edge WEIGHT (Manning discharge conductance)
         — enters via the W_e edge-feature projection, same mechanism as
           STGNNDynEdge's 5th edge feature.
      2. Dynamic TOPOLOGY (HAND-triggered activation)
         — enters as a log-additive gate on the attention score, so an
           inactive HAND edge contributes exactly zero to the destination
           node's aggregation, regardless of what the learned projection
           would otherwise produce for that edge's raw score.
      3. Hard directional GATE (elevation)
         — same log-additive mechanism, always active on every edge
           (Class A and Class B alike).

    Dense edge-list implementation (no torch_geometric dependency), following
    DFCGNNFlood's PhysicallyBiasedGATConv. Efficient for graphs with a few
    hundred edges or fewer, which is the regime here (28 river + a handful
    of HAND candidate edges for the Lee catchment).
    """

    def __init__(
        self,
        d_in:            int,
        d_out:           int,
        n_heads:         int   = 4,
        n_edge_feat:     int   = 5,
        dropout:         float = 0.1,
        tau_gate:        float = 5.0,
        negative_slope:  float = 0.2,
    ):
        super().__init__()
        self.d_in      = d_in
        self.d_out     = d_out
        self.n_heads   = n_heads
        self.d_head    = d_out // n_heads
        self.tau_gate  = tau_gate

        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.W_e = nn.Linear(n_edge_feat, d_out, bias=False)

        self.attn_vec = nn.Parameter(torch.empty(n_heads, 3 * self.d_head))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout    = nn.Dropout(dropout)
        self.out_proj   = nn.Linear(d_out, d_out)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

    def forward(
        self,
        h:               Tensor,   # [B, N, d_in]   node hidden states
        edge_index:      Tensor,   # [2, E]         combined river + HAND edges
        edge_attr:       Tensor,   # [B, E, n_edge_feat]  batch-dependent
                                    # (conductance varies with discharge each
                                    #  forward pass, so unlike DFCGNNFlood's
                                    #  static [E, F] edge_attr this is batched)
        node_elev:       Tensor,   # [N]            elevation, hard gate
        hand_activation: Tensor,   # [B, E]         ∈ (0,1]; ≡1 for river edges
    ) -> Tensor:                   # [B, N, d_out]
        B, N, _ = h.shape
        E = edge_index.shape[1]
        H, D = self.n_heads, self.d_head

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # ── Linear projections ─────────────────────────────────────────
        q = self.W_q(h).view(B, N, H, D)
        k = self.W_k(h).view(B, N, H, D)
        v = self.W_v(h).view(B, N, H, D)

        # Edge feature projection — batched (edge_attr carries dynamic
        # conductance, so it differs per batch element / timestep)
        e_feat = self.W_e(edge_attr).view(B, E, H, D)   # [B, E, H, D]

        # ── Attention scores ───────────────────────────────────────────
        q_src = q[:, src_idx, :, :]                      # [B, E, H, D]
        k_dst = k[:, dst_idx, :, :]                       # [B, E, H, D]
        cat   = torch.cat([q_src, k_dst, e_feat], dim=-1)  # [B, E, H, 3D]

        score = (cat * self.attn_vec.unsqueeze(0).unsqueeze(0)).sum(-1)
        score = self.leaky_relu(score)                     # [B, E, H]

        # ── Hard elevation gate (always active — Class A + B) ──────────
        # g_ij ≈ 1 when elev_src > elev_dst (downhill, physically possible)
        # g_ij ≈ 0 when elev_src < elev_dst (uphill, gated out)
        elev_diff = node_elev[src_idx] - node_elev[dst_idx]        # [E]
        elev_gate = torch.sigmoid(elev_diff / self.tau_gate)        # [E]
        score = score + torch.log(elev_gate + 1e-6).unsqueeze(0).unsqueeze(-1)

        # ── HAND activation gate (dynamic topology) ─────────────────────
        # hand_activation ≡ 1 for river (Class A) edges → log(1)=0, no effect.
        # For HAND (Class B) edges it is the learned sigmoid switch; when it
        # collapses to 0 the edge's score → −∞ and it receives exactly zero
        # attention after softmax, regardless of the raw score computed above.
        score = score + torch.log(hand_activation + 1e-6).unsqueeze(-1)

        # ── Softmax over incoming edges per destination node ────────────
        alpha = self._scatter_softmax(score, dst_idx, N)   # [B, E, H]
        alpha = self.dropout(alpha)

        # ── Message aggregation ──────────────────────────────────────────
        v_src = v[:, src_idx, :, :]                          # [B, E, H, D]
        msg   = v_src * alpha.unsqueeze(-1)                  # [B, E, H, D]

        h_out = torch.zeros(B, N, H, D, device=h.device, dtype=h.dtype)
        dst_exp = dst_idx.view(1, E, 1, 1).expand(B, E, H, D)
        h_out.scatter_add_(1, dst_exp, msg)                  # [B, N, H, D]

        h_out = h_out.reshape(B, N, H * D)
        return self.out_proj(h_out)

    @staticmethod
    def _scatter_softmax(score: Tensor, dst_idx: Tensor, N: int) -> Tensor:
        """Numerically stable softmax over incoming edges per destination node."""
        B, E, H = score.shape

        score_max = torch.full(
            (B, N, H), float("-inf"), device=score.device, dtype=score.dtype)
        dst_exp = dst_idx.view(1, E, 1).expand(B, E, H)
        score_max.scatter_reduce_(1, dst_exp, score, reduce="amax",
                                   include_self=True)

        score_shifted = score - score_max[:, dst_idx, :]
        exp_score = score_shifted.exp()
        exp_sum   = torch.zeros(B, N, H, device=score.device, dtype=score.dtype)
        exp_sum.scatter_add_(1, dst_exp, exp_score)

        alpha = exp_score / (exp_sum[:, dst_idx, :] + 1e-9)
        return alpha


# ═════════════════════════════════════════════════════════════════════
# 2. Full Unified Model
# ═════════════════════════════════════════════════════════════════════

class DFCGNNUnified(nn.Module):
    """
    Physically-Complete Dynamic Connectivity GNN.

    GRU backbone (per-node, shared weights) → UnifiedPhysicalGATConv
    (river + HAND edges, discharge conductance, HAND activation, hard
    elevation gate, all in one attention computation) → dual head
    (stage regression + flood classification, unchanged from DFCGNNFlood).

    Note on inputs: like DFCGNNFlood (and unlike the STGNNFlood family),
    this model does NOT concatenate static node_attr at the input — all
    static/physical information lives in edge_attr (river_dist, area_ratio,
    elev_drop, same_tributary; overland_dist, hand_threshold) and in the
    node_elev buffer used by the hard gate. f_dyn should equal X.npy's
    feature count (11 for the current Lee gauge+soil-moisture feature set).

    Parameters
    ----------
    n_nodes             : int    — number of gauge nodes (27 for Lee)
    f_dyn               : int    — dynamic input features per node per timestep
    d_model             : int    — hidden dimension throughout
    n_heads             : int    — attention heads in UnifiedPhysicalGATConv
    T_out               : int    — forecast horizon (steps)
    edge_index          : Tensor — [2, E_river] river-network edges (from load_graph,
                                    same topology as STGNNDynEdge/STGNNHANDEdge)
    edge_attr_static    : Tensor — [E_river, 4] static river edge features
    node_elev           : Tensor — [N] elevation (m OD), from edge_features.npz
    hand_src/hand_dst   : Tensor — [E_hand] HAND candidate edge endpoints
    hand_threshold      : Tensor — [E_hand] minimum HAND (m), diagnostics only
                                    — NOT used in the activation gate (see
                                    z_saddle below)
    hand_overland_dist  : Tensor — [E_hand] overland distance (km)
    z_saddle            : Tensor — [E_hand] ABSOLUTE elevation (m OD) at each
                                    HAND edge's corridor saddle point, from
                                    precompute_hand_edges.py's z_saddle_m —
                                    what the activation gate actually compares
                                    reconstructed water-surface elevation
                                    against (option-3 datum fix)
    gauge_datum         : Tensor — [N] per-node station datum (m OD), from
                                    nodes.csv's gauge_datum_mOSGM15
    stage_range         : Tensor — [N] per-node p90_mAOD − gauge_datum_mOSGM15,
                                    used to reconstruct absolute water-surface
                                    elevation from normalised_stage
    n_gru_layers        : int    — GRU depth
    dropout             : float  — dropout rate
    lambda_flood        : float  — weight of auxiliary flood-flag BCE loss
    tau_gate            : float  — elevation gate softness (m)
    discharge_idx       : int    — column index of discharge in f_dyn (default 3,
                                    matches GAUGE_FEATURES ordering elsewhere)
    discharge_ref       : float or Tensor[N] — reference discharge for conductance
    activation_sharpness: float  — initial HAND-activation sigmoid sharpness α
    """

    def __init__(
        self,
        n_nodes:              int,
        f_dyn:                int,
        d_model:              int   = 64,
        n_heads:              int   = 4,
        T_out:                int   = 4,
        edge_index:           Tensor | None = None,
        edge_attr_static:     Tensor | None = None,
        node_elev:            Tensor | None = None,
        hand_src:             Tensor | None = None,
        hand_dst:             Tensor | None = None,
        hand_threshold:       Tensor | None = None,
        hand_overland_dist:   Tensor | None = None,
        z_saddle:             Tensor | None = None,
        gauge_datum:          Tensor | None = None,
        stage_range:          Tensor | None = None,
        n_gru_layers:         int   = 2,
        dropout:              float = 0.1,
        lambda_flood:         float = 0.1,
        tau_gate:             float = 5.0,
        discharge_idx:        int   = 3,
        discharge_ref               = 1.0,
        activation_sharpness: float = 5.0,
    ):
        super().__init__()
        assert edge_index is not None and edge_attr_static is not None, \
            "river edge_index/edge_attr_static are required"
        assert hand_src is not None and hand_dst is not None, \
            "HAND candidate edges are required — run precompute_hand_edges.py"
        assert node_elev is not None, \
            "node_elev is required for the hard elevation gate"
        assert z_saddle is not None, \
            "z_saddle is required (option-3 datum fix) — rerun " \
            "precompute_hand_edges.py to regenerate hand_edges.npz with " \
            "z_saddle_m before using this model"
        assert gauge_datum is not None and stage_range is not None, \
            "gauge_datum / stage_range are required (option-3 datum fix) " \
            "— load from nodes.csv's gauge_datum_mOSGM15 / p90_mAOD columns"

        self.n_nodes      = n_nodes
        self.d_model      = d_model
        self.T_out        = T_out
        self.lambda_flood = lambda_flood
        self.discharge_idx = discharge_idx

        # ── River (Class A) edge buffers ────────────────────────────────
        self.register_buffer("river_edge_index",       edge_index.long())
        self.register_buffer("river_edge_attr_static",  edge_attr_static.float())
        self.register_buffer("node_elev",               node_elev.float())

        if isinstance(discharge_ref, torch.Tensor):
            self.register_buffer("discharge_ref", discharge_ref.float())
        else:
            self.register_buffer("discharge_ref",
                                  torch.tensor([float(discharge_ref)]))

        # ── HAND (Class B) edge buffers ─────────────────────────────────
        self.register_buffer("hand_src", hand_src.long())
        self.register_buffer("hand_dst", hand_dst.long())
        self.register_buffer("hand_threshold", hand_threshold.float())
        self.register_buffer(
            "hand_dist_norm", (hand_overland_dist / 5.0).float())  # 5km max
        max_t = hand_threshold.max().clamp(min=1e-3)
        self.register_buffer("hand_thresh_norm", (hand_threshold / max_t).float())

        # ── Option-3 datum-consistent activation gate buffers ───────────
        # z_saddle: absolute elevation (m OD) at each HAND edge's corridor
        # saddle point — same datum as node_elev. gauge_datum/stage_range:
        # per-NODE quantities used to reconstruct absolute water-surface
        # elevation from normalised_stage (see _build_edge_attr_and_
        # activation). Replaces the earlier stage_anomaly-vs-hand_threshold
        # comparison, which mixed an elevation-free rolling-baseline
        # quantity with a HAND-relative one on inconsistent footing.
        self.register_buffer("z_saddle", z_saddle.float())
        self.register_buffer("gauge_datum", gauge_datum.float())
        self.register_buffer("stage_range", stage_range.float())

        # ── Combined edge_index, built once (static structure) ─────────
        self.n_river = edge_index.shape[1]
        self.n_hand  = hand_src.shape[0]
        combined_src = torch.cat([edge_index[0], hand_src])
        combined_dst = torch.cat([edge_index[1], hand_dst])
        self.register_buffer("edge_index",
                              torch.stack([combined_src, combined_dst]))

        # ── Learnable physical parameters ───────────────────────────────
        self.conductance_scale = nn.Parameter(torch.tensor(3.0))
        self.activation_sharpness = nn.Parameter(
            torch.tensor(float(activation_sharpness)))

        # ── 1. Input projection (dynamic features only, no node_attr) ──
        self.input_proj = nn.Linear(f_dyn, d_model)

        # ── 2. GRU backbone (shared across nodes) ───────────────────────
        self.gru = nn.GRU(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = n_gru_layers,
            batch_first = True,
            dropout     = dropout if n_gru_layers > 1 else 0.0,
        )

        # ── 3. Unified physically-biased GAT layer ──────────────────────
        self.gat = UnifiedPhysicalGATConv(
            d_in        = d_model,
            d_out       = d_model,
            n_heads     = n_heads,
            n_edge_feat = 5,
            dropout     = dropout,
            tau_gate    = tau_gate,
        )

        self.norm_gru = nn.LayerNorm(d_model)
        self.norm_gat = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

        # ── 4. Stage prediction head (primary task) ─────────────────────
        self.stage_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, T_out),
        )

        # ── 5. Flood flag head (auxiliary task) ──────────────────────────
        self.flood_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    # ──────────────────────────────────────────────────────────────────
    def _build_edge_attr_and_activation(self, x_last: Tensor):
        """
        x_last : [B, N, F_dyn]  last observed timestep

        Returns
        -------
        edge_attr  : [B, E_total, 5]   E_total = E_river + E_hand
        activation : [B, E_total]      ≡1 for river edges, learned sigmoid
                                        for HAND edges
        """
        B = x_last.shape[0]
        device = x_last.device

        # ── River edges: Manning discharge conductance (Class A) ───────
        src_r = self.river_edge_index[0]
        Q_src = x_last[:, src_r, self.discharge_idx]           # [B, E_river]
        Q_ref = (self.discharge_ref[src_r] if self.discharge_ref.shape[0] > 1
                 else self.discharge_ref)
        conductance = torch.sigmoid(
            self.conductance_scale * (Q_src / (Q_ref + 1e-8) - 1.0))
        # log-transform: guarantees a monotonic Q↑ → attention↑ relationship
        # regardless of the sign the learned W_e projection assigns it
        # (same reasoning as STGNNDynEdge's log_conductance fix).
        log_conductance = torch.log(conductance + 1e-6)         # [B, E_river]

        river_static = self.river_edge_attr_static.unsqueeze(0).expand(
            B, -1, -1)                                          # [B, E_river, 4]
        river_ea = torch.cat(
            [river_static, log_conductance.unsqueeze(-1)], dim=-1)  # [B, E_river, 5]
        river_activation = torch.ones(B, self.n_river, device=device)

        # ── HAND edges: datum-consistent activation (Class B) ───────────
        # Option-3 fix: reconstruct ABSOLUTE water-surface elevation
        # H = gauge_datum + normalised_stage * stage_range (feature index
        # [1], NOT stage_anomaly at index [0] — stage_anomaly is an
        # elevation-free deviation from a rolling 7-day baseline, per
        # build_dataset.py's own documentation, and has no consistent
        # physical footing against hand_threshold's DEM-relative scale).
        # H is compared against z_saddle — the absolute elevation (m OD)
        # at the same location hand_threshold was sampled from — so both
        # sides of the comparison now live on the same vertical datum.
        norm_stage_src = x_last[:, self.hand_src, 1]
        norm_stage_dst = x_last[:, self.hand_dst, 1]

        H_src = (self.gauge_datum[self.hand_src].unsqueeze(0)
                 + norm_stage_src * self.stage_range[self.hand_src].unsqueeze(0))
        H_dst = (self.gauge_datum[self.hand_dst].unsqueeze(0)
                 + norm_stage_dst * self.stage_range[self.hand_dst].unsqueeze(0))
        max_H = torch.maximum(H_src, H_dst)

        hand_activation = torch.sigmoid(
            self.activation_sharpness
            * (max_H - self.z_saddle.unsqueeze(0)))              # [B, E_hand]

        dist_t   = self.hand_dist_norm.unsqueeze(0).expand(B, -1)
        thresh_t = self.hand_thresh_norm.unsqueeze(0).expand(B, -1)
        zeros    = torch.zeros(B, self.n_hand, device=device)
        hand_ea  = torch.stack(
            [dist_t, thresh_t, zeros, zeros, zeros], dim=-1)     # [B, E_hand, 5]

        edge_attr  = torch.cat([river_ea, hand_ea], dim=1)       # [B, E_total, 5]
        activation = torch.cat([river_activation, hand_activation], dim=1)
        return edge_attr, activation

    # ──────────────────────────────────────────────────────────────────
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor [B, T_in, N, F_dyn]

        Returns
        -------
        stage_pred   : Tensor [B, T_out, N]  — predicted delta stage_anomaly
        flood_logits : Tensor [B, N]          — flood flag logits (pre-sigmoid)
        """
        B, T_in, N, F_dyn = x.shape

        # ── GRU backbone ──────────────────────────────────────────────
        h = self.input_proj(x)                                  # [B, T_in, N, d_model]
        h = h.permute(0, 2, 1, 3).reshape(B * N, T_in, self.d_model)
        h, _ = self.gru(h)
        h = h[:, -1, :].reshape(B, N, self.d_model)
        h = self.norm_gru(h)

        # ── Build combined edge_attr / activation for this batch ───────
        x_last = x[:, -1, :, :]                                  # [B, N, F_dyn]
        edge_attr, activation = self._build_edge_attr_and_activation(x_last)

        # ── Unified physically-biased attention ─────────────────────────
        h_gat = self.gat(
            h=h, edge_index=self.edge_index, edge_attr=edge_attr,
            node_elev=self.node_elev, hand_activation=activation,
        )
        h = self.norm_gat(h + self.dropout(h_gat))

        # ── Output heads ──────────────────────────────────────────────
        stage = self.stage_head(h)                               # [B, N, T_out]
        stage = stage.permute(0, 2, 1)                            # [B, T_out, N]
        flood_logits = self.flood_head(h).squeeze(-1)             # [B, N]

        return stage, flood_logits

    def compute_loss(
        self,
        stage_pred:   Tensor,
        flood_logits: Tensor,
        y_stage:      Tensor,
        y_flood:      Tensor,
        node_valid:   Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """
        Convenience method mirroring DFCGNNFlood.compute_loss. In practice
        the training script computes this inline (with pos_weight balancing,
        see train_dfc_gnn.py's make_flood_labels + train_epoch) rather than
        calling this method directly — kept for API parity / quick testing.
        """
        loss_stage = F.mse_loss(stage_pred, y_stage)
        weight = node_valid if node_valid is not None else None
        loss_flood = F.binary_cross_entropy_with_logits(
            flood_logits, y_flood.float(), weight=weight)
        loss_total = loss_stage + self.lambda_flood * loss_flood

        with torch.no_grad():
            ss_res = ((stage_pred - y_stage) ** 2).sum()
            ss_tot = ((y_stage - y_stage.mean()) ** 2).sum()
            nse    = 1.0 - ss_res / (ss_tot + 1e-8)
            flood_pred = (flood_logits.sigmoid() > 0.5).float()
            flood_acc  = (flood_pred == y_flood.float()).float().mean()

        return loss_total, {
            "loss_total": loss_total.item(),
            "loss_stage": loss_stage.item(),
            "loss_flood": loss_flood.item(),
            "nse":        nse.item(),
            "flood_acc":  flood_acc.item(),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_in, N, T_out = 4, 32, 27, 4
    F_dyn = 11
    E_river, E_hand = 28, 12

    edge_index = torch.randint(0, N, (2, E_river))
    edge_attr_static = torch.randn(E_river, 4)
    node_elev  = torch.rand(N) * 200          # 0-200 m OD
    x_seq      = torch.rand(B, T_in, N, F_dyn) * 2

    hand_pairs = torch.randperm(N * (N - 1))[:E_hand * 2].reshape(E_hand, 2)
    hand_src   = hand_pairs[:, 0] % N
    hand_dst   = hand_pairs[:, 1] % N
    hand_thr   = torch.rand(E_hand) * 2 + 1.0     # diagnostics only now
    hand_dist  = torch.rand(E_hand) * 3 + 1.0

    # Option-3 datum buffers — realistic-ish m-OD values matching the
    # actual nodes.csv range (roughly -5 to 120 m OD across the 27 gauges)
    z_saddle    = node_elev[hand_src] + torch.rand(E_hand) * 3.0
    gauge_datum = node_elev.clone()
    stage_range = torch.rand(N) * 2.0 + 0.5       # 0.5-2.5 m

    model = DFCGNNUnified(
        n_nodes=N, f_dyn=F_dyn, d_model=64, n_heads=4, T_out=T_out,
        edge_index=edge_index, edge_attr_static=edge_attr_static,
        node_elev=node_elev,
        hand_src=hand_src, hand_dst=hand_dst,
        hand_threshold=hand_thr, hand_overland_dist=hand_dist,
        z_saddle=z_saddle, gauge_datum=gauge_datum, stage_range=stage_range,
        discharge_idx=3, discharge_ref=1.0,
    )

    stage_pred, flood_logits = model(x_seq)
    assert stage_pred.shape == (B, T_out, N), f"Wrong stage shape: {stage_pred.shape}"
    assert flood_logits.shape == (B, N), f"Wrong flood shape: {flood_logits.shape}"
    print(f"stage_pred:   {tuple(stage_pred.shape)}  ✓")
    print(f"flood_logits: {tuple(flood_logits.shape)}  ✓")

    y_stage = torch.randn(B, T_out, N)
    y_flood = (torch.rand(B, N) > 0.7).float()
    loss, metrics = model.compute_loss(stage_pred, flood_logits, y_stage, y_flood)
    loss.backward()
    print("Backward pass:  ✓")
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})

    # Verify gradients flow through both learnable physical parameters
    assert model.conductance_scale.grad is not None, "conductance_scale: no grad"
    assert model.activation_sharpness.grad is not None, "activation_sharpness: no grad"
    print("conductance_scale grad:    ✓")
    print("activation_sharpness grad: ✓")

    # Sanity check: an inactive HAND edge should contribute ~zero attention.
    # Force z_saddle far above any reconstructible water-surface elevation
    # → activation≈0. (Note: z_saddle, not hand_threshold, now drives the
    # gate — hand_threshold is retained only for diagnostics/edge_attr.)
    model.z_saddle.fill_(1_000_000.0)
    edge_attr2, activation2 = model._build_edge_attr_and_activation(x_seq[:, -1])
    assert activation2[:, model.n_river:].max().item() < 1e-3, \
        "HAND edges should be ~fully deactivated when z_saddle is unreachable"
    print("HAND deactivation sanity check:  ✓  "
          f"(max activation={activation2[:, model.n_river:].max().item():.2e})")

    # Sanity check: activation should respond meaningfully to a realistic
    # normalised_stage sweep — this is the check that would have caught
    # the original datum-mismatch bug (stage_anomaly vs hand_threshold),
    # where the sigmoid could sit near a fixed operating point regardless
    # of x_seq's actual values.
    model.z_saddle.copy_(node_elev[hand_src] + torch.rand(E_hand) * 3.0)
    x_low  = x_seq.clone(); x_low[:, -1, :, 1]  = -2.0
    x_high = x_seq.clone(); x_high[:, -1, :, 1] =  5.0
    _, act_low  = model._build_edge_attr_and_activation(x_low[:, -1])
    _, act_high = model._build_edge_attr_and_activation(x_high[:, -1])
    spread = (act_high[:, model.n_river:] - act_low[:, model.n_river:]).abs().mean().item()
    assert spread > 0.3, (
        f"HAND activation barely responds to normalised_stage "
        f"(mean |Δactivation|={spread:.4f}) — datum mismatch likely."
    )
    print(f"Activation responds to stage sweep:  ✓  (mean |Δ|={spread:.3f})")

    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")
    print("Smoke test passed.")
