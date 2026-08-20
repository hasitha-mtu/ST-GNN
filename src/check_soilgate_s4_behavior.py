"""
check_soilgate_s4_behavior.py — investigates why STGNNSoilGate has the
WORST absolute RMSE of all 10 models on S4_SatBreakthrough at 6hr,
despite being the only model with a genuinely positive NSE at 1hr on the
same scenario.

Context (see scenario_summary.csv): implied target variance at S4/6hr is
essentially identical across all 10 models (~0.123m std), so the extreme
negative NSE values there (-19 to -27) are a metric artifact of low
target variance, not evidence any one model is uniquely catastrophic.
But SoilGate's RMSE (0.6055m) IS the largest of all 10 -- a real,
model-specific finding worth explaining, separate from the NSE framing.

Two hypotheses this script checks directly against a trained checkpoint:
  1. Gate collapse: sat_threshold/sat_sharpness have drifted to a
     degenerate value (same failure mode diagnosed and "fixed" via the
     sparsity penalty earlier -- this checks whether that fix actually
     held at longer horizons specifically, not just early epochs).
  2. Gate miscalibration under the ACTUAL S4 injection: rather than
     synthetic extreme test inputs (as used for the DFC-GNN checks),
     this loads real S4_SatBreakthrough X_synthetic.npy and evaluates
     the gate's actual activation trajectory across the real
     saturation ramp and breakthrough, checking whether it opens at a
     physically sensible point relative to the injected
     excess_threshold (0.75, from scenario_generator.py).

Usage:
    python check_soilgate_s4_behavior.py --checkpoint checkpoints/st_gnn_soil_gate/42/24/best_model.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Expects a horizon=24 (6hr) checkpoint to match the anomaly")
    p.add_argument("--scenario-dir", type=str,
                   default="dataset/scenarios/S4_SatBreakthrough",
                   help="Directory containing X_synthetic.npy/scenario_meta.json for S4")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if any("_orig_mod." in k for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    hp = ckpt.get("hparams", {})

    # Handles both the original unbounded parameterization (sat_threshold
    # saved directly -- what every existing checkpoint has) and, once
    # st_gnn_soil_gate.py's reparameterization fix is trained under, the
    # new bounded one (sat_threshold_raw, needs sigmoid to recover the
    # effective value).
    if "sat_threshold_raw" in sd:
        sat_threshold = torch.sigmoid(sd["sat_threshold_raw"])
        bounded_param = True
    elif "sat_threshold" in sd:
        sat_threshold = sd["sat_threshold"]
        bounded_param = False
    else:
        sat_threshold = None
    sat_sharpness = sd.get("sat_sharpness")
    if sat_threshold is None or sat_sharpness is None:
        print("sat_threshold/sat_sharpness not found in state_dict -- "
              "is this actually an st_gnn_soil_gate checkpoint?")
        return

    print(f"Learned sat_threshold: {sat_threshold.item():.4f}  (init was 0.75)"
          f"{'  [bounded parameterization]' if bounded_param else '  [legacy unbounded parameterization]'}")
    print(f"Learned sat_sharpness: {sat_sharpness.item():.4f}  (init was 10.0)")

    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")
    if "hparams" in ckpt and "lambda_gate_sparsity" in hp:
        print(f"lambda_gate_sparsity used in training: {hp['lambda_gate_sparsity']}")

    collapsed_low  = sat_threshold.item() < 0.30
    collapsed_high = sat_threshold.item() > 0.98
    if collapsed_low:
        print("  -> LIKELY COLLAPSED TOWARD ALWAYS-OPEN "
              "(threshold near 0 -- gate opens almost regardless of saturation)")
    elif collapsed_high:
        print("  -> LIKELY COLLAPSED TOWARD ALWAYS-CLOSED "
              "(threshold near 1 -- gate essentially never opens)")
    else:
        print("  -> threshold sits in a plausible, non-degenerate range")

    # ── Hypothesis 2: gate trajectory against the REAL S4 injection ────
    scen_dir = Path(args.scenario_dir)
    meta_path = scen_dir / "scenario_meta.json"
    x_path    = scen_dir / "X_synthetic.npy"
    if not meta_path.exists() or not x_path.exists():
        print(f"\n{scen_dir} not found -- skipping real-injection gate trajectory check. "
              f"Run scenario_generator.py --scenario S4 first if this is expected to exist.")
        return

    meta = json.load(open(meta_path))
    X = np.load(x_path)
    T_WINDOW = meta.get("T_per_window", 104)
    bt = meta["sat_breakthrough_step"]
    excess_threshold = meta.get("excess_threshold", 0.75)
    swvl2_sat_idx = 9   # matches STGNNSoilGate's default swvl2_sat_idx

    # Catchment-mean saturation trajectory for the first window, exactly
    # as STGNNSoilGate's own S_bar computation does (mean over N nodes).
    window0 = X[:T_WINDOW]
    S_bar_trajectory = window0[:, :, swvl2_sat_idx].mean(axis=1)   # [T_WINDOW]

    gate_trajectory = torch.sigmoid(
        sat_sharpness * (torch.from_numpy(S_bar_trajectory).float() - sat_threshold)
    ).numpy()

    print(f"\nReal S4 injection, window 0, breakthrough at step {bt}:")
    print(f"  S_bar at breakthrough:        {S_bar_trajectory[bt]:.4f}  "
          f"(excess_threshold={excess_threshold})")
    print(f"  Gate activation at breakthrough: {gate_trajectory[bt]:.4f}")
    print(f"  Gate activation 10 steps before: {gate_trajectory[max(0,bt-10)]:.4f}")
    print(f"  Gate activation 10 steps after:  {gate_trajectory[min(T_WINDOW-1,bt+10)]:.4f}")

    # Does the gate open BEFORE the physical excess_threshold is crossed
    # (anticipatory, as designed), AFTER it (too late, behaving reactively
    # despite being designed to anticipate), or not meaningfully at all?
    threshold_crossing_step = np.argmax(S_bar_trajectory >= excess_threshold) \
        if (S_bar_trajectory >= excess_threshold).any() else None
    gate_open_step = np.argmax(gate_trajectory >= 0.5) \
        if (gate_trajectory >= 0.5).any() else None

    print(f"\n  Physical saturation crosses excess_threshold at step: {threshold_crossing_step}")
    print(f"  Learned gate crosses 0.5 activation at step:          {gate_open_step}")
    if threshold_crossing_step is not None and gate_open_step is not None:
        lead = threshold_crossing_step - gate_open_step
        if lead > 0:
            print(f"  -> Gate opens {lead} steps BEFORE physical threshold "
                  f"(anticipatory, as designed)")
        elif lead < 0:
            print(f"  -> Gate opens {-lead} steps AFTER physical threshold "
                  f"(reactive, NOT anticipatory -- contrary to design intent)")
        else:
            print(f"  -> Gate opens exactly at the physical threshold")
    elif gate_open_step is None:
        print("  -> Gate NEVER meaningfully opens (>=0.5) across this window")


if __name__ == "__main__":
    main()
