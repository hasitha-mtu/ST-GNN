"""
inspect_dataset.py — Verify the content of X.npy / y.npy / valid_mask.npy
before trusting them in a training run.

Usage
-----
    python inspect_dataset.py
    python inspect_dataset.py --proc-dir dataset/processed --graph-dir dataset/graph

Reports, for each file:
  - shape, dtype, NaN/Inf counts
  - per-feature (X.npy) summary stats, labeled using the exact
    GAUGE_FEATURES + SM_FEATURES ordering from build_dataset.py — so you
    can confirm column 3 really is discharge_m3s, not guess from position
  - per-node valid_mask coverage (fraction of timesteps with real data)
  - cross-check against nodes.csv (row count, ref alignment)
  - dataset_metadata.json contents, if present (build_dataset.py writes
    this — provenance, coverage stats, soil-moisture flag)
  - a handful of sanity flags: constant features (std≈0, likely a bug),
    features that are 100% zero (expected for discharge at level-only
    nodes, worth a second look for anything else), fraction of NaN
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Exact feature ordering from build_dataset.py — GAUGE_FEATURES + SM_FEATURES.
# If USE_SOIL_MOISTURE was False when X.npy was built, only the first 5 apply
# (F=5) — the script detects this from X's actual shape rather than assuming.
GAUGE_FEATURES = [
    "stage_anomaly", "normalized_stage", "dh_dt", "discharge_m3s", "rainfall_mm",
]
SM_FEATURES = [
    "swvl1_raw", "swvl1_sat_ratio", "swvl1_anomaly",
    "swvl2_raw", "swvl2_sat_ratio", "swvl2_anomaly",
]
ALL_FEATURES = GAUGE_FEATURES + SM_FEATURES

BASE_DIR      = Path(__file__).resolve().parent.parent
ERA5_DIR      = BASE_DIR / "dataset/era5_land_sm"
ERA5_SM_FILE  = ERA5_DIR / "era5_sm_gridded_2026-05-11_2026-05-20.nc"


def summarize_array(name: str, arr: np.ndarray):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"  shape: {arr.shape}   dtype: {arr.dtype}")
    n_total = arr.size
    n_nan = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    n_inf = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    print(f"  NaN: {n_nan:,} ({100*n_nan/n_total:.3f}%)   "
          f"Inf: {n_inf:,} ({100*n_inf/n_total:.3f}%)")


def inspect_X(X: np.ndarray, feature_names: list[str]):
    summarize_array("X.npy — dynamic features", X)
    T, N, F = X.shape
    print(f"  T (timesteps) = {T:,}   N (nodes) = {N}   F (features) = {F}")

    if F != len(feature_names):
        print(f"  WARNING: F={F} doesn't match the {len(feature_names)} known "
              f"feature names — either USE_SOIL_MOISTURE differs from what "
              f"you expect, or the feature set has changed. Falling back to "
              f"generic 'feature_{{i}}' labels.")
        feature_names = [f"feature_{i}" for i in range(F)]

    print(f"\n  Per-feature statistics (pooled over all timesteps × nodes):")
    header = f"  {'#':>2} {'name':<20} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'%zero':>7} {'%nan':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    flags = []
    for i, fname in enumerate(feature_names):
        col = X[:, :, i]
        col_flat = col.ravel()
        valid = col_flat[~np.isnan(col_flat)]
        if len(valid) == 0:
            print(f"  {i:>2} {fname:<20} {'ALL NaN':>10}")
            flags.append(f"feature[{i}] '{fname}' is entirely NaN")
            continue
        vmin, vmax = valid.min(), valid.max()
        vmean, vstd = valid.mean(), valid.std()
        pct_zero = 100 * (valid == 0).sum() / len(valid)
        pct_nan  = 100 * (len(col_flat) - len(valid)) / len(col_flat)
        print(f"  {i:>2} {fname:<20} {vmin:>10.4f} {vmax:>10.4f} "
              f"{vmean:>10.4f} {vstd:>10.4f} {pct_zero:>6.1f}% {pct_nan:>6.1f}%")

        if vstd < 1e-6:
            flags.append(f"feature[{i}] '{fname}' is constant (std={vstd:.2e}) — likely a bug")
        if pct_zero > 95 and fname != "discharge_m3s":
            flags.append(f"feature[{i}] '{fname}' is >{pct_zero:.0f}% zero — "
                         f"expected for discharge_m3s at level-only nodes, worth "
                         f"checking for anything else")
        if pct_nan > 5:
            flags.append(f"feature[{i}] '{fname}' has {pct_nan:.1f}% NaN — check gap-filling")

    if flags:
        print(f"\n  Flags to check:")
        for f in flags:
            print(f"    - {f}")
    else:
        print(f"\n  No flags raised.")

    return feature_names


def inspect_y(y: np.ndarray):
    summarize_array("y.npy — target (next-step stage_anomaly)", y)
    valid = y[~np.isnan(y)]
    print(f"  min={valid.min():.4f}  max={valid.max():.4f}  "
          f"mean={valid.mean():.4f}  std={valid.std():.4f}")


def inspect_valid_mask(mask: np.ndarray, nodes_df: pd.DataFrame | None):
    summarize_array("valid_mask.npy — per-timestep, per-node data availability", mask)
    T, N = mask.shape
    coverage = mask.astype(np.float64).mean(axis=0)  # per-node fraction valid
    print(f"\n  Per-node coverage (fraction of {T:,} timesteps with valid data):")
    order = np.argsort(coverage)
    for idx in order:
        label = f"node_{idx}"
        if nodes_df is not None and idx < len(nodes_df):
            row = nodes_df.iloc[idx]
            label = f"{row.get('ref', idx)} ({row.get('name', '?')})"
        flag = "  <-- LOW COVERAGE" if coverage[idx] < 0.5 else ""
        print(f"    [{idx:2d}] {label:<35} {coverage[idx]*100:6.2f}%{flag}")


def cross_check_nodes(X_shape: tuple, graph_dir: Path) -> pd.DataFrame | None:
    nodes_path = graph_dir / "nodes.csv"
    if not nodes_path.exists():
        print(f"\n  nodes.csv not found at {nodes_path} — skipping cross-check.")
        return None
    nodes_df = pd.read_csv(nodes_path)
    N_X = X_shape[1]
    N_nodes = len(nodes_df)
    print(f"\n  Cross-check against nodes.csv:")
    print(f"    X.npy node dimension: {N_X}")
    print(f"    nodes.csv row count:  {N_nodes}")
    if N_X != N_nodes:
        print(f"    MISMATCH — X.npy and nodes.csv disagree on node count. "
              f"Row-order-based indexing (e.g. hand_edges.npz's src/dst, "
              f"gauge_datum lookups) will silently misalign if you proceed.")
    else:
        print(f"    OK — node counts match.")
    return nodes_df


def show_metadata(proc_dir: Path):
    meta_path = proc_dir / "dataset_metadata.json"
    if not meta_path.exists():
        print(f"\n  dataset_metadata.json not found at {meta_path}.")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"\n{'='*70}\ndataset_metadata.json\n{'='*70}")
    print(json.dumps(meta, indent=2))


def cross_check_discharge(X: np.ndarray, nodes_df: pd.DataFrame | None,
                          feature_names: list[str]):
    if nodes_df is None or "discharge_m3s" not in feature_names:
        return
    if "has_discharge" not in nodes_df.columns:
        print(f"\n  nodes.csv has no 'has_discharge' column — skipping discharge cross-check.")
        return

    q_idx = feature_names.index("discharge_m3s")
    N = X.shape[1]
    print(f"\n  Per-node discharge_m3s vs. nodes.csv's has_discharge flag:")
    mismatches = []
    for i in range(min(N, len(nodes_df))):
        col = X[:, i, q_idx]
        valid = col[~np.isnan(col)]
        pct_zero = 100 * (valid == 0).sum() / len(valid) if len(valid) else 100.0
        stated = bool(nodes_df.iloc[i].get("has_discharge", False))
        actual_has_data = pct_zero < 99.0   # essentially never zero -> real gauge
        flag = ""
        if stated != actual_has_data:
            flag = "  <-- MISMATCH"
            ref = nodes_df.iloc[i].get("ref", i)
            name = nodes_df.iloc[i].get("name", "?")
            mismatches.append(f"node ref={ref} ({name}): nodes.csv says "
                              f"has_discharge={stated}, but X.npy is "
                              f"{pct_zero:.1f}% zero")
        print(f"    [{i:2d}] {nodes_df.iloc[i].get('name','?'):<28} "
              f"has_discharge={str(stated):<5}  X.npy %zero={pct_zero:6.1f}%{flag}")

    if mismatches:
        print(f"\n  {len(mismatches)} mismatch(es) found — worth checking "
              f"whether nodes.csv or X.npy's discharge sourcing is stale:")
        for m in mismatches:
            print(f"    - {m}")


def main(proc_dir: Path, graph_dir: Path):
    X = np.load(proc_dir / "X.npy")
    y = np.load(proc_dir / "y.npy")

    nodes_df = cross_check_nodes(X.shape, graph_dir)
    feature_names = inspect_X(X, ALL_FEATURES)
    cross_check_discharge(X, nodes_df, feature_names)
    inspect_y(y)

    mask_path = proc_dir / "valid_mask.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
        inspect_valid_mask(mask, nodes_df)
    else:
        print(f"\n  valid_mask.npy not found at {mask_path} — skipping.")

    ts_path = proc_dir / "timestamps.npy"
    if ts_path.exists():
        ts = np.load(ts_path, allow_pickle=True)
        print(f"\n  timestamps.npy: {len(ts)} entries, "
              f"{ts[0]} → {ts[-1]}")

    show_metadata(proc_dir)

    print(f"\n{'='*70}\nDone.\n{'='*70}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proc-dir", type=Path, default=Path("dataset/processed"))
    p.add_argument("--graph-dir", type=Path, default=Path("dataset/graph"))
    args = p.parse_args()
    main(args.proc_dir, args.graph_dir)

# import matplotlib.pyplot as plt
# import xarray as xr
#
# if __name__ == "__main__":
#     ds = xr.open_dataset(ERA5_SM_FILE)
#     print(f'variables: {ds.variables}')
#     print(f'variables keys: {ds.variables.keys()}')
#     # Select a variable and slice it (e.g., first time step)
#     # Replace 'temperature' with your actual variable name
#     ds['swvl1'].isel(time=0).plot()
#
#     # Display the map/graph
#     plt.show()