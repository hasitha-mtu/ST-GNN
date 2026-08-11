"""
check_scenario_shapes.py — Quick sanity check of the saved .npy shapes for
every scenario, to confirm/deny whether S3_ChannelBlockage's synthetic
arrays are actually empty (as scenario_evaluator.py's skip messages
imply: "got 0 steps").

Usage
-----
    python src/scenarios/check_scenario_shapes.py
"""

from pathlib import Path
import json
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCEN_DIR = BASE_DIR / "dataset/scenarios"

SCENARIOS = ["S1_ConvectiveCell", "S2_GaugeFailure",
             "S3_ChannelBlockage", "S4_SatBreakthrough",
             "S5_SpatialGradient"]


def main():
    for scen_name in SCENARIOS:
        d = SCEN_DIR / scen_name
        print(f"\n{'='*60}\n  {scen_name}\n{'='*60}")

        if not d.exists():
            print(f"  [MISSING] directory does not exist: {d}")
            continue

        for fname in ["X_synthetic.npy", "y_synthetic.npy", "mask_synthetic.npy"]:
            fpath = d / fname
            if not fpath.exists():
                print(f"  [MISSING] {fname}")
                continue
            arr = np.load(fpath, mmap_mode="r")
            n_nan = None
            n_zero = None
            if arr.size > 0:
                # Only compute these on a small array / cheaply — avoid
                # loading a huge memmap fully into memory unnecessarily.
                sample = np.asarray(arr[: min(arr.shape[0], 500)])
                n_nan = int(np.isnan(sample).sum()) if np.issubdtype(sample.dtype, np.floating) else 0
                n_zero = int((sample == 0).sum())
            print(f"  {fname:<22} shape={arr.shape}  dtype={arr.dtype}"
                  + (f"  nan(sample)={n_nan}  zero(sample)={n_zero}" if arr.size else "  [EMPTY]"))

        meta_path = d / "scenario_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  scenario_meta.json keys: {list(meta.keys())}")
            # Print anything that looks like a length/slice/step field —
            # useful for spotting an off-by-one or start>=end slicing bug
            # in scenario_generator.py.
            interesting = {k: v for k, v in meta.items()
                           if any(s in k.lower() for s in
                                  ["step", "len", "start", "end", "t_", "n_"])
                           and not isinstance(v, (list, dict))}
            if interesting:
                print(f"  length/step-related meta fields: {interesting}")
        else:
            print("  [MISSING] scenario_meta.json")


if __name__ == "__main__":
    main()
