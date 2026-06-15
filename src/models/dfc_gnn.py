"""
models/dfc_gnn.py
═══════════════════════════════════════════════════════════════════════
Dynamic Flood Connectivity Graph Neural Network (DFC-GNN)

Replaces the static HAND threshold used in STGNNHANDEdge with a learned,
physically-constrained dynamic attention mechanism.  The attention weight
on each edge (i → j) is computed from four physical features:

    river_dist_km   — along-network distance (spatial decay prior)
    elev_diff_m     — elevation_i − elevation_j (hard uphill gate)
    travel_time_h   — kinematic wave travel time (temporal lag prior)
    hand_diff_m     — bankfull_i − bankfull_j (relative susceptibility)

plus the hidden states of both endpoint nodes from the GRU backbone.

Architecture
─────────────
                 X [B, T_in, N, F]
                        │
              ┌─────────▼──────────┐
              │   GRU backbone     │  per-node temporal encoding
              │   (shared weights) │
              └─────────┬──────────┘
                 H [B, N, d_model]
                        │
              ┌─────────▼──────────────────────────────┐
              │  PhysicallyBiasedGATConv               │
              │                                        │
              │  e_ij = MLP([h_i, h_j, edge_attr])    │
              │  g_ij = σ(-(elev_j - elev_i)/τ_gate)  │  hard gate
              │  A_ij = g_ij · softmax(e_ij)           │
              │  m_i  = Σ_j A_ij · W_v · h_j          │  message
              └─────────┬──────────────────────────────┘
                 H' [B, N, d_model]  (after residual + LayerNorm)
                        │
              ┌─────────┴──────────┐
              │                    │
    ┌─────────▼──────┐   ┌────────▼────────┐
    │  Stage head    │   │  Flood prob head │
    │  MLP → T_out   │   │  MLP → sigmoid  │
    └─────────┬──────┘   └────────┬────────┘
     ŷ [B,T_out,N]      p [B,N]  (flood flag, 0–1)

Losses
───────
  L_stage = MSE(ŷ, y)                        — primary (same as baselines)
  L_flood = BCEWithLogitsLoss(logits, flag)   — auxiliary (stage > bankfull)
  L_total = L_stage + λ_flood · L_flood       — combined

  λ_flood defaults to 0.1 — small enough not to degrade stage accuracy
  while still providing spatial flood supervision signal.

HAND decoder (HANDDecoder)
───────────────────────────
Replaces the fixed inundation formula:
    flood ← hand ≤ stage_anomaly

With a per-node learnable depth scale:
    flood ← hand ≤ τ_k · stage_anomaly      (τ_k > 0, initialised at 1.0)

τ_k is trained from SAR flood masks once available.  Until then, the
existing HAND inundation formula is equivalent to τ_k = 1 for all nodes.

Usage
──────
    from models.dfc_gnn import DFCGNNFlood, HANDDecoder

    # Initialise
    model = DFCGNNFlood(
        n_nodes    = 27,
        f_in       = 11,       # number of input features per node
        d_model    = 64,       # hidden dimension
        n_heads    = 4,        # GAT attention heads
        T_out      = 4,        # forecast horizon (steps)
        edge_index = edge_index,  # [2, E] long tensor
        edge_attr  = edge_attr,   # [E, 4] float tensor (normalised features)
        node_elev  = node_elev,   # [N] float tensor (m OD, for hard gate)
        lambda_flood = 0.1,
    )

    # Forward pass
    stage_pred, flood_logits = model(x)
    # stage_pred:   [B, T_out, N]
    # flood_logits: [B, N]  (apply sigmoid for probability)

    # Training step
    loss_stage = F.mse_loss(stage_pred, y_stage)
    loss_flood = F.binary_cross_entropy_with_logits(flood_logits, y_flood)
    loss = loss_stage + model.lambda_flood * loss_flood

    # HAND decoder (after τ_k training)
    decoder = HANDDecoder(n_nodes=27)
    flood_map = decoder(stage_pred, hand_raster, catchment_masks)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═════════════════════════════════════════════════════════════════════
# 1. Physically-Biased GAT Convolution
# ═════════════════════════════════════════════════════════════════════

class PhysicallyBiasedGATConv(nn.Module):
    """
    Graph Attention Convolution biased by four physical edge features.

    For each directed edge (i → j) the attention score is:

        raw_score = LeakyReLU(a^T [W_q·h_i ‖ W_k·h_j ‖ W_e·edge_attr])

    where ‖ is concatenation and W_e projects the 4D physical features
    into the attention space.

    Hard elevation gate:
        g_ij = σ( (elev_i − elev_j) / τ_gate )

    This gate approaches 1 when node i is above node j (water can flow
    downhill), and approaches 0 when node j is above node i.  τ_gate
    controls the softness of the gate (default 5.0 m — transitions over
    a 10m elevation band).

    Final attention weight:
        α_ij = g_ij · softmax_j(raw_score_ij)

    Message aggregation:
        m_i = Σ_j α_ij · W_v · h_j

    This implementation does NOT use torch_geometric.MessagePassing
    to keep the dependency optional.  It uses a dense edge-list approach
    which is efficient for graphs with ≤ 1000 edges.
    """

    def __init__(
        self,
        d_in:       int,    # input hidden dimension
        d_out:      int,    # output hidden dimension per head
        n_heads:    int = 4,
        n_edge_feat:int = 4,  # number of physical edge features
        dropout:    float = 0.1,
        tau_gate:   float = 5.0,   # elevation gate softness (m)
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.d_in       = d_in
        self.d_out      = d_out
        self.n_heads    = n_heads
        self.d_head     = d_out // n_heads
        self.tau_gate   = tau_gate

        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        # Query, Key, Value projections (per-head, applied to node features)
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

        # Edge feature projection into attention space
        self.W_e = nn.Linear(n_edge_feat, d_out, bias=False)

        # Attention scoring vector (one per head)
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
        h:          Tensor,   # [B, N, d_in]  node hidden states
        edge_index: Tensor,   # [2, E]  (src, dst) pairs
        edge_attr:  Tensor,   # [E, 4]  physical features (normalised)
        node_elev:  Tensor,   # [N]     elevation in m OD (for hard gate)
    ) -> Tensor:              # [B, N, d_out]
        """
        Forward pass.

        Parameters
        ----------
        h           Node hidden states from GRU backbone.
        edge_index  Directed edges as (src, dst) index pairs.
        edge_attr   Physical edge features from edge_features.npz.
        node_elev   Gauge pixel elevations for the hard elevation gate.

        Returns
        -------
        h_out   Aggregated node representations [B, N, d_out].
        """
        B, N, _ = h.shape
        E = edge_index.shape[1]
        H = self.n_heads
        D = self.d_head

        src_idx = edge_index[0]   # [E]
        dst_idx = edge_index[1]   # [E]

        # ── Linear projections ─────────────────────────────────────────
        q = self.W_q(h).view(B, N, H, D)  # [B, N, H, D]
        k = self.W_k(h).view(B, N, H, D)
        v = self.W_v(h).view(B, N, H, D)

        # Edge feature projection: [E, 4] → [E, d_out] → [E, H, D]
        e_feat = self.W_e(edge_attr).view(E, H, D)   # [E, H, D]

        # ── Attention scores ───────────────────────────────────────────
        # q_src [B, E, H, D], k_dst [B, E, H, D]
        q_src = q[:, src_idx, :, :]   # [B, E, H, D]
        k_dst = k[:, dst_idx, :, :]   # [B, E, H, D]

        # Concatenate: [q_src ‖ k_dst ‖ e_feat]  →  [B, E, H, 3D]
        e_feat_exp = e_feat.unsqueeze(0).expand(B, -1, -1, -1)  # [B, E, H, D]
        cat = torch.cat([q_src, k_dst, e_feat_exp], dim=-1)     # [B, E, H, 3D]

        # Dot with per-head attention vector  →  [B, E, H]
        score = (cat * self.attn_vec.unsqueeze(0).unsqueeze(0)).sum(-1)
        score = self.leaky_relu(score)   # [B, E, H]

        # ── Hard elevation gate ────────────────────────────────────────
        # g_ij ≈ 1 when elev_src > elev_dst (downhill — physically possible)
        # g_ij ≈ 0 when elev_src < elev_dst (uphill — gated out)
        elev_diff = node_elev[src_idx] - node_elev[dst_idx]     # [E]
        gate = torch.sigmoid(elev_diff / self.tau_gate)          # [E]
        # Broadcast: [B, E, H]
        gate = gate.unsqueeze(0).unsqueeze(-1).expand(B, E, H)
        score = score * gate

        # ── Softmax over incoming edges per destination node ───────────
        # Convert to sparse-style: for each dst, normalise over all src
        # Using a scatter-based softmax
        alpha = self._scatter_softmax(score, dst_idx, N)   # [B, E, H]
        alpha = self.dropout(alpha)

        # ── Message aggregation ────────────────────────────────────────
        # v_src [B, E, H, D], weighted by alpha [B, E, H, 1]
        v_src = v[:, src_idx, :, :]                              # [B, E, H, D]
        msg   = v_src * alpha.unsqueeze(-1)                      # [B, E, H, D]

        # Scatter sum to destination nodes
        h_out = torch.zeros(B, N, H, D, device=h.device, dtype=h.dtype)
        dst_exp = dst_idx.view(1, E, 1, 1).expand(B, E, H, D)
        h_out.scatter_add_(1, dst_exp, msg)                      # [B, N, H, D]

        # Reshape and project
        h_out = h_out.reshape(B, N, H * D)                      # [B, N, d_out]
        return self.out_proj(h_out)

    @staticmethod
    def _scatter_softmax(score: Tensor, dst_idx: Tensor, N: int) -> Tensor:
        """
        Numerically stable softmax over incoming edges per destination node.
        score:   [B, E, H]
        dst_idx: [E]
        Returns: [B, E, H]  — attention weights summing to 1 per (dst, head)
        """
        B, E, H = score.shape

        # Max per destination for numerical stability
        score_max = torch.full(
            (B, N, H), float('-inf'), device=score.device, dtype=score.dtype)
        dst_exp = dst_idx.view(1, E, 1).expand(B, E, H)
        score_max.scatter_reduce_(1, dst_exp, score, reduce='amax',
                                  include_self=True)

        # Shift by max
        score_shifted = score - score_max[:, dst_idx, :]

        # Exp and scatter sum
        exp_score = score_shifted.exp()
        exp_sum   = torch.zeros(B, N, H, device=score.device, dtype=score.dtype)
        exp_sum.scatter_add_(1, dst_exp, exp_score)

        # Normalise
        alpha = exp_score / (exp_sum[:, dst_idx, :] + 1e-9)
        return alpha


# ═════════════════════════════════════════════════════════════════════
# 2. Full DFC-GNN Model
# ═════════════════════════════════════════════════════════════════════

class DFCGNNFlood(nn.Module):
    """
    Dynamic Flood Connectivity GNN for multi-step stage forecasting.

    Extends the existing STGNNFlood variants by replacing the static HAND-
    based edge activation with a physically-biased dynamic attention layer.
    All hyperparameters are compatible with train_model.py.

    Parameters
    ----------
    n_nodes      : int    — number of gauge nodes (27 for Lee)
    f_in         : int    — input features per node per timestep
    d_model      : int    — hidden dimension throughout the model
    n_heads      : int    — number of attention heads in GAT layer
    T_out        : int    — forecast horizon (number of output timesteps)
    edge_index   : Tensor — [2, E] long  — directed edge index
    edge_attr    : Tensor — [E, 4] float — normalised physical edge features
    node_elev    : Tensor — [N] float    — gauge elevations (m OD)
    n_gru_layers : int    — number of stacked GRU layers in backbone
    dropout      : float  — dropout rate in attention and output heads
    lambda_flood : float  — weight of auxiliary flood-flag loss
    tau_gate     : float  — elevation gate softness (m)
    """

    def __init__(
        self,
        n_nodes:      int,
        f_in:         int,
        d_model:      int   = 64,
        n_heads:      int   = 4,
        T_out:        int   = 4,
        edge_index:   Tensor | None = None,
        edge_attr:    Tensor | None = None,
        node_elev:    Tensor | None = None,
        n_gru_layers: int   = 2,
        dropout:      float = 0.1,
        lambda_flood: float = 0.1,
        tau_gate:     float = 5.0,
    ):
        super().__init__()
        self.n_nodes      = n_nodes
        self.d_model      = d_model
        self.T_out        = T_out
        self.lambda_flood = lambda_flood

        # ── Register graph topology as buffers (moved with .to(device)) ─
        if edge_index is not None:
            self.register_buffer("edge_index", edge_index.long())
        if edge_attr is not None:
            self.register_buffer("edge_attr",  edge_attr.float())
        if node_elev is not None:
            self.register_buffer("node_elev",  node_elev.float())

        # ── 1. Input projection: [B, T, N, f_in] → [B, T, N, d_model] ─
        self.input_proj = nn.Linear(f_in, d_model)

        # ── 2. GRU backbone (shared across nodes) ─────────────────────
        # Processes the sequence dimension; outputs h [B, N, d_model]
        self.gru = nn.GRU(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = n_gru_layers,
            batch_first = True,
            dropout     = dropout if n_gru_layers > 1 else 0.0,
        )

        # ── 3. Physically-biased GAT layer ─────────────────────────────
        n_ef = edge_attr.shape[1] if edge_attr is not None else 4
        self.gat = PhysicallyBiasedGATConv(
            d_in        = d_model,
            d_out       = d_model,
            n_heads     = n_heads,
            n_edge_feat = n_ef,
            dropout     = dropout,
            tau_gate    = tau_gate,
        )

        # Layer norms + residual connection
        self.norm_gru = nn.LayerNorm(d_model)
        self.norm_gat = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

        # ── 4. Stage prediction head (primary task) ────────────────────
        # [B, N, d_model] → [B, N, T_out]
        self.stage_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, T_out),
        )

        # ── 5. Flood flag head (auxiliary task) ────────────────────────
        # [B, N, d_model] → [B, N, 1] logit (sigmoid → probability)
        self.flood_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for linear layers."""
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor [B, T_in, N, F]  — input sequence

        Returns
        -------
        stage_pred   : Tensor [B, T_out, N]  — predicted stage anomaly
        flood_logits : Tensor [B, N]          — flood flag logits (pre-sigmoid)
        """
        B, T_in, N, F = x.shape

        # ── Input projection ────────────────────────────────────────────
        h = self.input_proj(x)             # [B, T_in, N, d_model]

        # ── GRU backbone ────────────────────────────────────────────────
        # Reshape: process all nodes independently through shared GRU
        # [B, T_in, N, d_model] → [B*N, T_in, d_model]
        h = h.permute(0, 2, 1, 3).reshape(B * N, T_in, self.d_model)
        h, _ = self.gru(h)                 # [B*N, T_in, d_model]
        h = h[:, -1, :]                    # last timestep: [B*N, d_model]
        h = h.reshape(B, N, self.d_model)  # [B, N, d_model]
        h = self.norm_gru(h)

        # ── Physically-biased GAT ────────────────────────────────────────
        h_gat = self.gat(
            h          = h,
            edge_index = self.edge_index,
            edge_attr  = self.edge_attr,
            node_elev  = self.node_elev,
        )                                  # [B, N, d_model]
        h = self.norm_gat(h + self.dropout(h_gat))   # residual

        # ── Stage prediction head ────────────────────────────────────────
        stage = self.stage_head(h)         # [B, N, T_out]
        stage = stage.permute(0, 2, 1)    # [B, T_out, N]

        # ── Flood flag head ──────────────────────────────────────────────
        flood_logits = self.flood_head(h).squeeze(-1)  # [B, N]

        return stage, flood_logits

    def compute_loss(
        self,
        stage_pred:   Tensor,   # [B, T_out, N]
        flood_logits: Tensor,   # [B, N]
        y_stage:      Tensor,   # [B, T_out, N]
        y_flood:      Tensor,   # [B, N]  binary {0, 1}
    ) -> tuple[Tensor, dict]:
        """
        Combined loss with stage MSE (primary) + flood BCE (auxiliary).

        Returns (total_loss, metrics_dict).
        """
        loss_stage = F.mse_loss(stage_pred, y_stage)
        loss_flood = F.binary_cross_entropy_with_logits(
            flood_logits, y_flood.float())
        loss_total = loss_stage + self.lambda_flood * loss_flood

        with torch.no_grad():
            # NSE (Nash-Sutcliffe Efficiency) for monitoring
            ss_res = ((stage_pred - y_stage) ** 2).sum()
            ss_tot = ((y_stage - y_stage.mean()) ** 2).sum()
            nse    = 1.0 - ss_res / (ss_tot + 1e-8)

            # Flood detection accuracy
            flood_pred  = (flood_logits.sigmoid() > 0.5).float()
            flood_acc   = (flood_pred == y_flood.float()).float().mean()

        return loss_total, {
            "loss_total":  loss_total.item(),
            "loss_stage":  loss_stage.item(),
            "loss_flood":  loss_flood.item(),
            "nse":         nse.item(),
            "flood_acc":   flood_acc.item(),
        }


# ═════════════════════════════════════════════════════════════════════
# 3. Learnable HAND Decoder
# ═════════════════════════════════════════════════════════════════════

class HANDDecoder(nn.Module):
    """
    Learnable per-node HAND depth scale for inundation mapping.

    Replaces the fixed formula  `hand ≤ stage_anomaly`  with:
        `hand ≤ τ_k · stage_anomaly`
    where τ_k is a learnable scalar per node, initialised at 1.0.

    Physical interpretation: τ_k captures the effective depth-scaling
    relationship between the gauge stage anomaly (measured at the channel)
    and the depth of flooding on the adjacent floodplain.  In the lower
    Lee, where tidal backing-up causes water to spread farther than a
    simple stage-to-HAND relationship predicts, τ_k > 1 is expected.
    In steep upland reaches, τ_k ≈ 1 or slightly < 1.

    Training: τ_k is supervised by SAR-derived flood masks.  For each
    timestep with an available SAR observation, the predicted inundation
    `hand ≤ τ_k · stage_pred` is compared to the SAR binary mask using
    a differentiable soft IoU loss.

    During inference (no SAR available), τ_k is fixed at its trained
    value and applied to the real-time stage predictions.

    Usage (training)
    ─────────────────
        decoder = HANDDecoder(n_nodes=27)

        # During training step with SAR mask available:
        pred_flood_prob = decoder.forward_soft(
            stage_pred,      # [B, T_out, N] — GRU stage predictions
            hand_raster,     # [H, W] — HAND raster (m above drainage)
            masks,           # list[np.ndarray] — per-node catchment masks
            t_step,          # int — which forecast step to decode
        )                    # [B, H, W] — soft flood probability

        loss_sar = decoder.soft_iou_loss(pred_flood_prob, sar_mask)
        loss_sar.backward()

    Usage (inference)
    ──────────────────
        flood_map = decoder.forward_hard(stage_pred, hand_raster, masks, t_step)
        # [H, W] bool — binary inundation map
    """

    def __init__(
        self,
        n_nodes:   int,
        tau_init:  float = 1.0,
        tau_min:   float = 0.1,   # minimum scale (prevents negative depth)
        tau_max:   float = 5.0,   # maximum scale (prevents unphysical extent)
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Learnable log-scale parameter (log ensures τ > 0 without clamping)
        # τ_k = exp(log_tau_k), initialised at log(tau_init) = 0.0 for τ=1
        self.log_tau = nn.Parameter(
            torch.full((n_nodes,), math.log(tau_init)))

    @property
    def tau(self) -> Tensor:
        """Per-node depth scale τ_k, clamped to [tau_min, tau_max]."""
        return self.log_tau.exp().clamp(self.tau_min, self.tau_max)

    def _build_flood_prob(
        self,
        stage:    Tensor,           # [N] stage anomaly for one timestep
        hand_np:  "np.ndarray",     # [H, W]
        masks:    "list[np.ndarray]",# [N] boolean masks [H, W]
        soft:     bool = True,
        temp:     float = 2.0,      # temperature for sigmoid softening
    ) -> Tensor:
        """
        Build a [H, W] flood probability (soft) or binary (hard) map.
        Uses torch operations on τ for gradient flow.
        """
        import numpy as np
        H, W = hand_np.shape

        hand_t  = torch.from_numpy(hand_np.astype(np.float32)).to(stage.device)
        tau     = self.tau   # [N]

        # Accumulate flood probability across nodes
        flood   = torch.zeros(H, W, device=stage.device)
        weights = torch.zeros(H, W, device=stage.device)

        for i in range(self.n_nodes):
            s = stage[i]
            if torch.isnan(s):
                continue
            depth_threshold = tau[i] * s   # scalar
            mask_t  = torch.from_numpy(masks[i].astype(np.float32)).to(stage.device)

            if soft:
                # Soft membership: σ((threshold - HAND) / temp)
                # → 1 where HAND << threshold, 0 where HAND >> threshold
                prob = torch.sigmoid((depth_threshold - hand_t) / temp)
                flood   = flood + mask_t * prob
                weights = weights + mask_t
            else:
                flood   = flood + mask_t * (hand_t <= depth_threshold).float()
                weights = weights + mask_t

        # Normalise by number of contributing nodes per pixel
        flood = flood / (weights + 1e-8)
        return flood.clamp(0.0, 1.0)

    def forward_soft(
        self,
        stage_pred: Tensor,            # [B, T_out, N]
        hand_np:    "np.ndarray",      # [H, W]
        masks:      "list[np.ndarray]",# list of [H, W] bool
        t_step:     int = 0,           # which horizon step to decode
    ) -> Tensor:                        # [B, H, W]
        """
        Differentiable flood probability map for SAR training.
        Returns soft [0, 1] probabilities for use in IoU/BCE loss.
        """
        B = stage_pred.shape[0]
        H, W = hand_np.shape
        results = []
        for b in range(B):
            stage_b = stage_pred[b, t_step, :]    # [N]
            prob    = self._build_flood_prob(stage_b, hand_np, masks,
                                             soft=True)
            results.append(prob)
        return torch.stack(results, dim=0)         # [B, H, W]

    def forward_hard(
        self,
        stage_pred: Tensor,            # [B, T_out, N]  or [N] for single step
        hand_np:    "np.ndarray",      # [H, W]
        masks:      "list[np.ndarray]",
        t_step:     int = 0,
    ) -> "np.ndarray":                  # [H, W] bool
        """Binary inundation map for inference (no gradients needed)."""
        import numpy as np
        with torch.no_grad():
            if stage_pred.dim() == 3:
                stage = stage_pred[0, t_step, :]
            else:
                stage = stage_pred
            prob = self._build_flood_prob(stage, hand_np, masks, soft=False)
        return (prob.cpu().numpy() > 0.5)

    @staticmethod
    def soft_iou_loss(
        pred: Tensor,   # [B, H, W]  soft predictions in [0, 1]
        target: Tensor, # [B, H, W]  binary SAR mask
    ) -> Tensor:
        """
        Soft IoU loss for spatial flood extent supervision.

        Standard binary IoU = TP / (TP + FP + FN).
        Soft version: replace hard binary membership with continuous
        probabilities for gradient flow through the decoder.

        Loss = 1 - soft_IoU  (so minimising loss maximises IoU)
        """
        pred   = pred.float()
        target = target.float()

        # Flatten spatial dims
        pred_f   = pred.view(pred.shape[0], -1)
        target_f = target.view(target.shape[0], -1)

        intersection = (pred_f * target_f).sum(dim=1)
        union        = (pred_f + target_f - pred_f * target_f).sum(dim=1)

        iou  = (intersection + 1e-6) / (union + 1e-6)
        return (1.0 - iou).mean()


# ═════════════════════════════════════════════════════════════════════
# 4. Helper: load edge features from npz
# ═════════════════════════════════════════════════════════════════════

def load_edge_features(
    path: str = "dataset/graph/edge_features.npz",
    device: str = "cpu",
) -> dict:
    """
    Load pre-computed edge features and return tensors ready for DFCGNNFlood.

    Example
    -------
        ef = load_edge_features("dataset/graph/edge_features.npz")
        model = DFCGNNFlood(
            n_nodes    = 27,
            f_in       = 11,
            edge_index = ef["edge_index"],
            edge_attr  = ef["edge_attr"],
            node_elev  = ef["node_elev"],
        )
    """
    data = np.load(path, allow_pickle=True)

    edge_index = torch.tensor(
        np.stack([data["src"], data["dst"]], axis=0), dtype=torch.long
    ).to(device)

    edge_attr = torch.tensor(
        np.stack(
            [data["river_dist_norm"],
             data["elev_diff_norm"],
             data["travel_time_norm"],
             data["hand_diff_norm"]]
            + ([data["sar_wetness_norm"]] if "sar_wetness_norm" in data else []),
            axis=1),
        dtype=torch.float32,
    ).to(device)

    node_elev = torch.tensor(
        data["node_elevation_m"], dtype=torch.float32
    ).to(device)

    return {
        "edge_index": edge_index,
        "edge_attr":  edge_attr,
        "node_elev":  node_elev,
        "n_edges":    edge_index.shape[1],
        "n_nodes":    node_elev.shape[0],
    }


# ═════════════════════════════════════════════════════════════════════
# 5. Model registry (for train_model.py compatibility)
# ═════════════════════════════════════════════════════════════════════

def build_dfc_gnn(
    n_nodes:    int,
    f_in:       int,
    T_out:      int,
    ef_path:    str  = "dataset/graph/edge_features.npz",
    d_model:    int  = 64,
    n_heads:    int  = 4,
    n_layers:   int  = 2,
    dropout:    float = 0.1,
    lambda_flood:float = 0.1,
    device:     str  = "cpu",
) -> DFCGNNFlood:
    """
    Factory function called by train_model.py for the dfc_gnn variant.

    Parameters mirror the existing STGNNFlood factory signature so that
    the training script can build DFCGNNFlood with a single config change:
        --model dfc_gnn
    """
    ef = load_edge_features(ef_path, device=device)
    model = DFCGNNFlood(
        n_nodes      = n_nodes,
        f_in         = f_in,
        d_model      = d_model,
        n_heads      = n_heads,
        T_out        = T_out,
        edge_index   = ef["edge_index"],
        edge_attr    = ef["edge_attr"],
        node_elev    = ef["node_elev"],
        n_gru_layers = n_layers,
        dropout      = dropout,
        lambda_flood = lambda_flood,
    ).to(device)
    return model
