"""
check_soilgate_threshold_sweep.py — checks sat_threshold/sat_sharpness
across ALL seeds and horizons for STGNNSoilGate, to determine whether the
out-of-range collapse found at horizon=24/seed=42 (sat_threshold=1.336,
physically unreachable since S_bar is capped at 1.0) is specific to that
one run or systemic across the model.

Usage:
    python check_soilgate_threshold_sweep.py
    python check_soilgate_threshold_sweep.py --ckpt-root checkpoints/st_gnn_soil_gate
"""
import argparse
from pathlib import Path

import torch

SEEDS    = [42, 123, 456]
HORIZONS = [4, 12, 16, 24, 48]
HZ_LABEL = {4: "1hr", 12: "3hr", 16: "4hr", 24: "6hr", 48: "12hr"}


def load_thresholds(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if any("_orig_mod." in k for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # Handles both the original unbounded parameterization (sat_threshold
    # saved directly) and, once the reparameterization fix is applied,
    # the new bounded one (sat_threshold_raw, needs sigmoid).
    if "sat_threshold_raw" in sd:
        threshold = torch.sigmoid(sd["sat_threshold_raw"]).item()
        bounded = True
    elif "sat_threshold" in sd:
        threshold = sd["sat_threshold"].item()
        bounded = False
    else:
        return None

    sharpness = sd.get("sat_sharpness", torch.tensor(float("nan"))).item()
    epoch = ckpt.get("epoch", None)
    return {"threshold": threshold, "sharpness": sharpness,
            "bounded_param": bounded, "epoch": epoch}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", type=str, default="checkpoints/st_gnn_soil_gate")
    args = p.parse_args()
    root = Path(args.ckpt_root)

    print(f"{'Seed':<8}{'Horizon':<10}{'threshold':<12}{'sharpness':<12}{'status'}")
    print("=" * 70)

    n_out_of_range = 0
    n_checked = 0

    for seed in SEEDS:
        for hz in HORIZONS:
            ckpt_path = root / str(seed) / str(hz) / "best_model.pt"
            if not ckpt_path.exists():
                print(f"{seed:<8}{HZ_LABEL.get(hz,hz):<10}{'--':<12}{'--':<12}"
                      f"checkpoint not found")
                continue

            result = load_thresholds(ckpt_path)
            if result is None:
                print(f"{seed:<8}{HZ_LABEL.get(hz,hz):<10}{'--':<12}{'--':<12}"
                      f"sat_threshold key not found -- not a soil_gate checkpoint?")
                continue

            n_checked += 1
            t, s = result["threshold"], result["sharpness"]
            out_of_range = not (0.0 <= t <= 1.0) if not result["bounded_param"] else False
            status = ("OUT OF PHYSICAL RANGE (collapsed)" if out_of_range
                      else "in range")
            if out_of_range:
                n_out_of_range += 1
            print(f"{seed:<8}{HZ_LABEL.get(hz,hz):<10}{t:<12.4f}{s:<12.4f}{status}")

    print("=" * 70)
    if n_checked > 0:
        print(f"\n{n_out_of_range}/{n_checked} checkpoints have sat_threshold "
              f"outside [0,1] (physically unreachable, S_bar is capped there by construction)")
        if n_out_of_range == 0:
            print("No collapse detected in the checkpoints checked.")
        elif n_out_of_range == n_checked:
            print("Collapse is systemic across all checked seeds/horizons -- "
                  "not specific to horizon=24/seed=42.")
        else:
            print("Collapse is present in SOME but not all seeds/horizons -- "
                  "worth checking whether it correlates with a specific horizon "
                  "(longer horizons may have less gradient signal reaching the gate) "
                  "or is seed-dependent (optimization variance).")


if __name__ == "__main__":
    main()
