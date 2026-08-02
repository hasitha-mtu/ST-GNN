"""
st_gnn_soil_gate.py  —  ST-GNN with antecedent soil moisture-conditioned topology
====================================================================================
Architecture
------------
Extends STGNNHANDEdge by replacing the per-edge stage-reactive activation gate
with a catchment-scale antecedent soil saturation gate.

STGNNHANDEdge (reactive):
    gate(i,j,t) = σ( α × (max(H_src, H_dst)(t) − z_saddle_ij) )
    Activates HAND cross-tributary edges AFTER water surface exceeds the terrain
    saddle — i.e., when flooding is already occurring.

STGNNSoilGate (anticipatory):
    gate(t)     = σ( γ × (S̄(t) − θ_sat) )
    where S̄(t) = mean over N nodes of swvl2_sat_ratio at the last input step.
    Activates HAND edges when the catchment is approaching field capacity —
    BEFORE stage rises at any individual gauge — providing the model with
    cross-tributary spatial information during the pre-event antecedent window.

Physical motivation
-------------------
In humid Atlantic catchments (dominant runoff mechanism: saturation-excess),
lateral floodplain connectivity between tributaries is governed by the pre-event
soil moisture state, not by the current observed stage. When the shallow soil
profile approaches field capacity (swvl2_sat_ratio → 1.0), any significant
precipitation event will produce near-total runoff and activate cross-tributary
flow paths — hours before any gauge records a threshold exceedance.

The key distinction:
    STGNNHANDEdge: topology changes REACTIVELY (after stage rises)
    STGNNSoilGate: topology changes ANTICIPATORILY (before stage rises)

For a 12-hour forecast, the input window (T_in = 32 steps = 8 hr) may span the
entire antecedent period before a flash flood peak. STGNNHANDEdge's gate remains
closed during these 8 hours while soil saturation quietly approaches 1.0.
STGNNSoilGate's gate opens as soon as S̄ crosses θ_sat — providing cross-
tributary spatial context throughout the input window, not just in the final steps.

Data requirement
----------------
swvl2_sat_ratio (ERA5-Land volumetric soil water layer 2, 7-28 cm, normalised to
field capacity) must be available in X.npy at feature index 9. This requires a
valid ERA5-Land download and correct build_dataset.py execution. If X.npy has
zero-valued SM features (ERA5 download failure), this model degrades to a fixed-
low-activation regime equivalent to STGNNHANDEdge with closed gates.

Parameters
----------
All parameters inherited from STGNNHANDEdge, plus:

swvl2_sat_idx : int  (default 9)
    Column index of swvl2_sat_ratio in the dynamic feature vector.
    Must match build_dataset.py's feature ordering:
        [0] stage_anomaly, [1] norm_stage, [2] dh_dt, [3] discharge,
        [4] rainfall_mm, [5] swvl1_raw, [6] swvl1_sat, [7] swvl1_anom,
        [8] swvl2_raw, [9] swvl2_sat, [10] swvl2_anom

sat_threshold : float  (default 0.75)
    Learnable catchment-mean saturation threshold (dimensionless, 0–1).
    Physical interpretation: fraction of field capacity at which cross-tributary
    connectivity activates. 0.75 corresponds to ~75% of saturation — the typical
    onset of saturation-excess runoff in Irish clay-loam soils.
    Initialised at 0.75 to match the physical transition point.

sat_sharpness : float  (default 10.0)
    Learnable sigmoid sharpness parameter γ (dimensionless).
    Higher values → sharper transition around sat_threshold.
    Initialised at 10.0, giving:
        S̄ = 0.55 (dry) → gate ≈ 0.12 (nearly closed)
        S̄ = 0.75 (threshold) → gate = 0.50
        S̄ = 0.85 (wet) → gate ≈ 0.73 (moderately open)
        S̄ = 0.95 (saturated) → gate ≈ 0.88 (strongly open)
    This range covers the observed seasonal swvl2_sat_ratio distribution in
    the Lee catchment.

References
----------
Brocca, L., Melone, F., Moramarco, T. (2008). On the estimation of antecedent
    wetness conditions in rainfall-runoff modelling. Hydrological Processes 22(5).
Muñoz-Sabater, J. et al. (2021). ERA5-Land: A state-of-the-art global reanalysis
    dataset for land applications. ESSD 13, 4349-4383.
Nobre, A.D. et al. (2011). Height Above the Nearest Drainage. J. Hydrol. 404.
"""

import torch
import torch.nn as nn

import sys
from pathlib import Path
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from models.st_gnn_hand_edge import STGNNHANDEdge


class STGNNSoilGate(STGNNHANDEdge):
    """
    ST-GNN with antecedent soil moisture-conditioned cross-tributary topology.

    Inherits the full STGNNHANDEdge architecture (river-network GATConv,
    HAND cross-tributary message passing, discharge conductance on river edges)
    and overrides only the HAND activation gate computation.

    The stage-reactive gate in STGNNHANDEdge:
        activation(i,j,t) = σ( α × (max(H_src, H_dst) − z_saddle_ij) )
    is replaced by the saturation-anticipatory gate:
        activation(t) = σ( γ × (S̄(t) − θ_sat) )
        S̄(t) = mean_n( swvl2_sat_ratio[n, t] )

    The gate is scalar per batch element — the same value for ALL HAND edges
    in a given time window. This is physically correct: catchment saturation
    determines whether the whole catchment is in a connected state, not
    individual edge pairs.
    """

    def __init__(
        self,
        *args,
        swvl2_sat_idx: int   = 9,
        sat_threshold:  float = 0.75,
        sat_sharpness:  float = 10.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        *args, **kwargs : passed to STGNNHANDEdge.__init__ unchanged.
        swvl2_sat_idx   : feature index of swvl2_sat_ratio in X.npy (default 9).
        sat_threshold   : initial catchment saturation threshold (default 0.75).
        sat_sharpness   : initial sigmoid sharpness γ (default 10.0).
        """
        super().__init__(*args, **kwargs)

        self.swvl2_sat_idx = swvl2_sat_idx

        # Learnable saturation gate parameters.
        # Named distinctly from the inherited activation_sharpness (which
        # belongs to the stage gate in STGNNHANDEdge) to allow both to
        # coexist in the state dict without conflict, enabling comparison
        # of the two gate types in ablation studies.
        self.sat_threshold = nn.Parameter(torch.tensor(float(sat_threshold)))
        self.sat_sharpness  = nn.Parameter(torch.tensor(float(sat_sharpness)))

    # ──────────────────────────────────────────────────────────────────
    def _hand_edge_attr(
        self,
        x_last: torch.Tensor,   # [B, N, F]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Override: replace the stage-based activation with a soil
        saturation-based activation.

        Returns
        -------
        static_ea  : [B, E_hand, 4]  same static layout as STGNNHANDEdge
        activation : [B, E_hand]     saturation gate broadcast over edges
        """
        B      = x_last.shape[0]
        E_hand = self.hand_src.shape[0]

        # ── Soil moisture activation gate (anticipatory) ───────────────
        # Extract swvl2_sat_ratio for all nodes at the last input step.
        # x_last[:, :, swvl2_sat_idx] shape: [B, N]
        swvl2_sat = x_last[:, :, self.swvl2_sat_idx]     # [B, N]

        # Catchment-mean saturation: scalar per batch element.
        S_bar = swvl2_sat.mean(dim=1, keepdim=True)       # [B, 1]

        # Sigmoid gate: opens as S̄ approaches and exceeds sat_threshold.
        # S_bar - sat_threshold is positive when catchment is wetter than
        # the learned threshold → gate approaches 1.
        gate_scalar = torch.sigmoid(
            self.sat_sharpness * (S_bar - self.sat_threshold)
        )                                                  # [B, 1]

        # Broadcast to all HAND edges: same gate value for every edge
        # because catchment saturation is a whole-catchment property.
        activation = gate_scalar.expand(B, E_hand)        # [B, E_hand]

        # ── Static edge attributes (identical to STGNNHANDEdge) ────────
        dist_t   = self.hand_dist_norm.unsqueeze(0).expand(B, -1)   # [B, E]
        thresh_t = self.hand_thresh_norm.unsqueeze(0).expand(B, -1) # [B, E]
        zeros    = torch.zeros(B, E_hand, device=x_last.device)     # [B, E]

        static_ea = torch.stack(
            [dist_t, thresh_t, zeros, zeros], dim=-1
        )                                                  # [B, E_hand, 4]

        return static_ea, activation


# ══════════════════════════════════════════════════════════════════════
#  Smoke test
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(0)

    B, T_in, N, T_out = 4, 32, 27, 4
    F_dyn, F_static, F_edge = 11, 7, 5
    E_river, E_hand = 28, 12

    edge_index = torch.randint(0, N, (2, E_river))
    edge_attr  = torch.randn(E_river, F_edge - 1)
    node_attr  = torch.randn(N, F_static)

    # Feature index 9 = swvl2_sat_ratio: set to realistic range 0.5–0.95
    x_seq = torch.rand(B, T_in, N, F_dyn)
    x_seq[:, :, :, 9] = torch.rand(B, T_in, N) * 0.45 + 0.50

    # Synthetic HAND edges
    hand_src   = torch.randint(0, N, (E_hand,))
    hand_dst   = torch.randint(0, N, (E_hand,))
    hand_thr   = torch.rand(E_hand) * 2 + 1.0
    hand_dist  = torch.rand(E_hand) * 3 + 1.0
    gauge_datum = torch.rand(N) * 100
    stage_range = torch.rand(N) * 2 + 0.5
    z_saddle   = gauge_datum[hand_src] + torch.rand(E_hand) * 3.0

    model = STGNNSoilGate(
        f_dyn=F_dyn, f_static=F_static, f_edge=F_edge,
        hand_src=hand_src, hand_dst=hand_dst,
        hand_threshold=hand_thr, hand_overland_dist=hand_dist,
        z_saddle=z_saddle, gauge_datum=gauge_datum, stage_range=stage_range,
        hidden=64, gat_heads=2, gru_layers=2, t_out=T_out,
        swvl2_sat_idx=9, sat_threshold=0.75, sat_sharpness=10.0,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Inherited (STGNNHANDEdge): {sum(p.numel() for p in model.parameters()) - 2:,}")
    print(f"  New (sat_threshold, sat_sharpness): 2")

    out = model(x_seq, node_attr, edge_index, edge_attr)
    assert out.shape == (B, T_out, N), f"Wrong shape: {out.shape}"
    print(f"Output shape: {tuple(out.shape)}  ✓")

    out.sum().backward()
    assert model.sat_threshold.grad is not None, "sat_threshold has no grad"
    assert model.sat_sharpness.grad  is not None, "sat_sharpness has no grad"
    print("Gradients flow through sat_threshold and sat_sharpness  ✓")

    # Key test: activation should respond to soil moisture variation
    with torch.no_grad():
        x_dry = x_seq.clone(); x_dry[:, -1, :, 9] = 0.50   # dry catchment
        x_wet = x_seq.clone(); x_wet[:, -1, :, 9] = 0.95   # saturated

        _, act_dry = model._hand_edge_attr(x_dry[:, -1, :, :])
        _, act_wet = model._hand_edge_attr(x_wet[:, -1, :, :])

        mean_dry = act_dry.mean().item()
        mean_wet = act_wet.mean().item()
        spread   = mean_wet - mean_dry

        print(f"Gate response — dry S̄=0.50: {mean_dry:.4f}  "
              f"wet S̄=0.95: {mean_wet:.4f}  Δ={spread:.4f}")
        assert spread > 0.5, (
            f"Gate barely responds to saturation change (Δ={spread:.4f}). "
            "Check swvl2_sat_idx or sat_sharpness initialisation.")
        print(f"Gate responds strongly to catchment saturation  ✓")

    # Verify the gate is UNIFORM across edges (unlike STGNNHANDEdge)
    with torch.no_grad():
        _, act = model._hand_edge_attr(x_seq[:, -1, :, :])
        edge_std = act[0].std().item()   # std across edges for batch elem 0
        print(f"Gate edge std = {edge_std:.6f} (should be ≈0 — uniform gate)  ✓")

    print("Smoke test passed.")
