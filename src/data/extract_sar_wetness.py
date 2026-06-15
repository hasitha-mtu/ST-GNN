"""
extract_sar_wetness.py
══════════════════════════════════════════════════════════════════════
Phase 4 of DFC-GNN pipeline.

Extracts the SAR wetness similarity feature for each directed edge (i→j)
and appends it to dataset/graph/edge_features.npz as a 5th feature.

What this script does
──────────────────────
1. Identifies the processed flood sigma0 GeoTIFF for each DFC-GNN
   training event (output of build_sar_reference.py).

2. Samples the sigma0 value at each of the 27 gauge node pixel locations
   in the DEM ITM grid — one value per node per event.
   This builds the SAR wetness matrix W[E_events × N_nodes].

3. Computes the pairwise cosine similarity between node backscatter
   vectors across all flood events:
       sim(i, j) = W[:,i] · W[:,j] / (‖W[:,i]‖ ‖W[:,j]‖)
   High similarity means nodes i and j tend to be wet simultaneously —
   they are likely in the same hydraulically connected flood zone.

4. Normalises the similarity scores (z-score) and appends them to
   edge_features.npz as `sar_wetness_norm`, updating the DFC-GNN
   edge_attr from [E, 4] to [E, 5].

5. Saves the raw W matrix and per-node backscatter statistics for
   inspection and future use.

Why wetness similarity is valuable
────────────────────────────────────
The four physical features already in edge_features.npz
(river_dist, elev_diff, travel_time, hand_diff) are purely terrain-
and hydrology-derived — they encode what COULD happen based on
landscape geometry.  SAR wetness similarity encodes what ACTUALLY
DOES happen — nodes that co-flood in the observational record.

Two nodes at similar elevations and distances can behave very
differently: one may be shielded by an embankment or drain quickly,
while another accumulates water due to urban impervious surfaces.
The backscatter record captures these real-world effects invisibly
to terrain-only models.

Output
───────
  dataset/graph/edge_features.npz
      Existing arrays preserved; adds:
          sar_wetness_norm  float32[E]  — z-score normalised similarity
          sar_wetness_raw   float32[E]  — raw cosine similarity [-1, 1]
          norm_stats        float32[10] — updated (was 8, adds 2 for wetness)

  dataset/graph/sar_wetness_W.npz
      W                float32[E_events, N_nodes]  — raw sigma0 matrix
      event_names      str[E_events]               — event name per row
      node_refs        str[N_nodes]                — gauge ref per column

Usage
──────
  # Run after build_sar_reference.py has processed all events
  python src/data/extract_sar_wetness.py

  # Verbose: print per-node backscatter table
  python src/data/extract_sar_wetness.py --verbose

  # Dry run: compute W but do not update edge_features.npz
  python src/data/extract_sar_wetness.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR  = BASE_DIR / "dataset/graph"
PROC_DIR   = BASE_DIR / "dataset/validation/processed"
EF_PATH    = GRAPH_DIR / "edge_features.npz"
W_OUT_PATH = GRAPH_DIR / "sar_wetness_W.npz"

# Processed flood sigma0 filenames produced by build_sar_reference.py
# Pattern: {mission}_{role}_{date}_sigma0_itm.tif
# We look for any file matching *flood*sigma0_itm.tif in PROC_DIR.
# Can also be specified explicitly via FLOOD_IMAGE_MAP if naming varies.
FLOOD_IMAGE_MAP: dict[str, str | None] = {
    "lee_flood_oct2023":  "s1a_flood_20231020_sigma0_itm.tif",
    "lee_flood_dec2023":  "s1a_flood_20231227_sigma0_itm.tif",
    "lee_flood_jan2024":  "s1a_flood_20240108_sigma0_itm.tif",
    "lee_flood_mar2022":  "s1a_flood_20220218_sigma0_itm.tif",
    "lee_flood_nov2024":  "s1a_flood_20241127_sigma0_itm.tif",
    "validation_nov2025": "s1c_flood_20251111_sigma0_itm.tif",
}

# Minimum number of events required to compute a meaningful
# similarity. With <3 events the cosine similarity is unreliable.
MIN_EVENTS = 3


# ═════════════════════════════════════════════════════════════════════
# Step 1: Sample sigma0 at gauge node locations
# ═════════════════════════════════════════════════════════════════════

def sample_sigma0_at_nodes(
    tif_path:  Path,
    nodes_df:  pd.DataFrame,
) -> np.ndarray:
    """
    Sample the calibrated sigma0 (dB) value from a GeoTIFF at each of
    the 27 gauge node pixel locations.

    If a node pixel is NaN (e.g. tidal estuary nodes on water), the
    value is taken from the 3×3 neighbourhood median.  If the full
    neighbourhood is NaN (gauge is on open sea), the value is
    substituted with the image-wide nanmedian (a neutral reference).

    Returns [N] float32 array of sigma0 values in dB.
    """
    import rasterio

    with rasterio.open(tif_path) as src:
        sigma0 = src.read(1).astype(np.float32)
        affine = src.transform
        H, W   = src.height, src.width

    img_median = float(np.nanmedian(sigma0))
    values     = np.full(len(nodes_df), img_median, dtype=np.float32)

    for i, row in nodes_df.iterrows():
        e = float(row["easting_itm"])
        n = float(row["northing_itm"])
        col = int(np.clip(round((e - affine.c) / affine.a), 0, W - 1))
        r   = int(np.clip(round((n - affine.f) / affine.e), 0, H - 1))

        val = sigma0[r, col]
        if np.isnan(val):
            r0, r1 = max(0, r - 1), min(H, r + 2)
            c0, c1 = max(0, col - 1), min(W, col + 2)
            patch = sigma0[r0:r1, c0:c1]
            val = float(np.nanmedian(patch)) if not np.all(np.isnan(patch)) \
                  else img_median
        values[i] = float(val)

    return values


# ═════════════════════════════════════════════════════════════════════
# Step 2: Build W matrix across all events
# ═════════════════════════════════════════════════════════════════════

def build_W_matrix(
    nodes_df: pd.DataFrame,
    verbose:  bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the SAR wetness matrix W[E_events, N_nodes].

    Each row is one flood event; each column is one gauge node.
    Values are sigma0 in dB — lower values (darker SAR) indicate
    smoother surfaces (open water, saturated soil).

    Returns (W, event_names) where event_names[i] is the DFC event
    name for row i.
    """
    rows        = []
    event_names = []

    print("── Building SAR wetness matrix W ──")
    for event_name, fname in FLOOD_IMAGE_MAP.items():
        tif_path = PROC_DIR / fname
        if not tif_path.exists():
            print(f"  SKIP {event_name}: {fname} not found")
            print(f"       Run build_sar_reference.py for this event first")
            continue

        sigma0_row = sample_sigma0_at_nodes(tif_path, nodes_df)
        rows.append(sigma0_row)
        event_names.append(event_name)

        if verbose:
            print(f"  {event_name}: [{sigma0_row.min():.2f}, "
                  f"{sigma0_row.max():.2f}] dB  "
                  f"mean={sigma0_row.mean():.2f}")
        else:
            print(f"  ✓ {event_name}: {fname}")

    if not rows:
        raise FileNotFoundError(
            f"No processed flood images found in {PROC_DIR}.\n"
            "Run build_sar_reference.py first to calibrate and geocode "
            "the flood images."
        )

    W = np.stack(rows, axis=0)   # [E_events, N_nodes]
    print(f"\n  W matrix: {W.shape}  ({len(event_names)} events × {W.shape[1]} nodes)")
    return W, event_names


# ═════════════════════════════════════════════════════════════════════
# Step 3: Compute pairwise SAR wetness similarity
# ═════════════════════════════════════════════════════════════════════

def compute_wetness_similarity(W: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between node backscatter
    vectors across flood events.

    W: [E_events, N_nodes]
    Returns: [N_nodes, N_nodes] similarity matrix, values in [-1, 1].

    Interpretation:
      sim(i, j) ≈ 1.0  — nodes i and j are always wet together
      sim(i, j) ≈ 0.0  — independent flood behaviour
      sim(i, j) ≈ -1.0 — anti-correlated (rare; suggests data issue)

    The sigma0 values are mean-centred across events before computing
    cosine similarity.  This removes the DC offset from different
    land-cover types (e.g. tidal nodes are always dark regardless of
    flood state) and focuses on the co-variation signal.
    """
    # Mean-centre across events for each node
    W_centred = W - W.mean(axis=0, keepdims=True)   # [E, N]

    # L2-normalise each node's backscatter vector
    norms = np.linalg.norm(W_centred, axis=0, keepdims=True) + 1e-8  # [1, N]
    W_norm = W_centred / norms                                          # [E, N]

    # Cosine similarity matrix [N, N]
    sim = W_norm.T @ W_norm   # [N, N]

    # Clamp to [-1, 1] (numerical precision)
    sim = np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(sim, 1.0)

    return sim.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════
# Step 4: Map similarity matrix to edge array
# ═════════════════════════════════════════════════════════════════════

def similarity_to_edge_feature(
    sim:       np.ndarray,   # [N, N]
    edge_data: dict,         # loaded edge_features.npz
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract pairwise similarity for each directed edge (src, dst).
    Returns (raw_similarity [E], normalised_similarity [E]).
    """
    src = edge_data["src"]   # [E] int
    dst = edge_data["dst"]   # [E] int

    raw = sim[src, dst].astype(np.float32)   # [E]

    # Z-score normalise (robust to outliers using median/std)
    finite = raw[np.isfinite(raw)]
    mu     = float(np.median(finite))
    std    = float(finite.std()) or 1.0
    normed = ((raw - mu) / std).astype(np.float32)

    print(f"  SAR wetness similarity per edge:")
    print(f"    Raw range:   [{raw.min():.3f}, {raw.max():.3f}]  "
          f"μ={raw.mean():.3f}")
    print(f"    Norm range:  [{normed.min():.3f}, {normed.max():.3f}]")
    print(f"    High-sim pairs (>0.8): {(raw > 0.8).sum()} / {len(raw)}")
    print(f"    Low-sim pairs (<0.2):  {(raw < 0.2).sum()} / {len(raw)}")

    return raw, normed


# ═════════════════════════════════════════════════════════════════════
# Diagnostics
# ═════════════════════════════════════════════════════════════════════

def print_wetness_table(
    W:          np.ndarray,     # [E, N]
    event_names:list[str],
    nodes_df:   pd.DataFrame,
    sim:        np.ndarray,     # [N, N]
):
    """Print per-node backscatter table and most-similar node pairs."""
    refs  = nodes_df["ref"].astype(str).tolist()
    names = nodes_df["name"].tolist() if "name" in nodes_df.columns \
            else [""] * len(refs)

    print("\n  Per-node sigma0 (dB) across events:")
    print(f"  {'ref':>7}  {'name':<22}  ", end="")
    for ev in event_names:
        short = ev.replace("lee_flood_","")[:8]
        print(f"{short:>10}", end="")
    print(f"  {'mean':>8}  {'std':>6}")
    print("  " + "─" * (7 + 22 + 12 + len(event_names)*10 + 16))

    for j in range(len(refs)):
        col = W[:, j]
        print(f"  {refs[j]:>7}  {names[j][:22]:<22}  ", end="")
        for v in col:
            print(f"{v:>10.2f}", end="")
        print(f"  {col.mean():>8.2f}  {col.std():>6.2f}")

    # Top-10 most similar pairs
    N = sim.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((sim[i, j], refs[i], refs[j]))
    pairs.sort(reverse=True)

    print("\n  Top 10 most co-flooded node pairs (by SAR wetness similarity):")
    print(f"  {'ref_i':>7}  {'ref_j':>7}  {'similarity':>12}")
    print("  " + "─" * 30)
    for sim_v, ri, rj in pairs[:10]:
        print(f"  {ri:>7}  {rj:>7}  {sim_v:>12.4f}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def run(verbose: bool, dry_run: bool):
    print("═"*60)
    print("  DFC-GNN Phase 4 — SAR Wetness Feature Extraction")
    print("═"*60)

    # ── Load nodes ────────────────────────────────────────────────────
    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    nodes_df["ref"] = nodes_df["ref"].astype(str)
    N = len(nodes_df)
    print(f"\n  {N} gauge nodes")

    # ── Build W matrix ────────────────────────────────────────────────
    print()
    W, event_names = build_W_matrix(nodes_df, verbose)
    E_events = len(event_names)

    if E_events < MIN_EVENTS:
        print(f"\n  WARNING: only {E_events} events available "
              f"(minimum {MIN_EVENTS} for reliable similarity).")
        print(f"  Proceed with caution — wetness similarity may be noisy.")
        if E_events < 2:
            print("  Cannot compute cosine similarity with <2 events. Exiting.")
            return

    # ── Save W matrix ─────────────────────────────────────────────────
    if not dry_run:
        np.savez_compressed(
            W_OUT_PATH,
            W           = W,
            event_names = np.array(event_names),
            node_refs   = nodes_df["ref"].values,
        )
        print(f"\n  Saved W matrix → {W_OUT_PATH.name}")

    # ── Compute similarity ────────────────────────────────────────────
    print("\n── Computing pairwise SAR wetness similarity ──")
    sim = compute_wetness_similarity(W)
    print(f"  Similarity matrix: {sim.shape}  "
          f"range=[{sim.min():.3f}, {sim.max():.3f}]")
    print(f"  Mean off-diagonal similarity: "
          f"{sim[~np.eye(N, dtype=bool)].mean():.3f}")

    # Verbose table
    if verbose:
        print_wetness_table(W, event_names, nodes_df, sim)

    # ── Load existing edge features ───────────────────────────────────
    if not EF_PATH.exists():
        raise FileNotFoundError(
            f"edge_features.npz not found at {EF_PATH}.\n"
            "Run compute_edge_features.py first."
        )
    edge_data = dict(np.load(EF_PATH, allow_pickle=True))
    E = len(edge_data["src"])
    print(f"\n── Mapping similarity to {E} directed edges ──")

    raw_sim, normed_sim = similarity_to_edge_feature(sim, edge_data)

    # ── Update edge_features.npz ──────────────────────────────────────
    if dry_run:
        print("\n  [dry-run] edge_features.npz NOT updated")
        print("  Re-run without --dry-run to apply changes")
        return

    # Append new arrays
    edge_data["sar_wetness_raw"]  = raw_sim
    edge_data["sar_wetness_norm"] = normed_sim

    # Update norm_stats (was 8 values for 4 features, now 10 for 5)
    old_stats = edge_data.get("norm_stats", np.zeros(8, dtype=np.float32))
    finite = raw_sim[np.isfinite(raw_sim)]
    mu_w   = float(np.median(finite))
    std_w  = float(finite.std()) or 1.0
    edge_data["norm_stats"] = np.append(
        old_stats[:8], [mu_w, std_w]).astype(np.float32)

    # Keep allow_pickle=True for string array node_refs
    np.savez_compressed(EF_PATH, **edge_data)
    size_kb = EF_PATH.stat().st_size / 1024
    print(f"\n  Updated edge_features.npz → {size_kb:.1f} KB")
    print(f"  New edge_attr shape: [E={E}, 5]  (was [E, 4])")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Summary")
    print(f"{'═'*60}")
    print(f"  Events used:       {E_events}  ({', '.join(event_names)})")
    print(f"  W matrix shape:    {W.shape}")
    print(f"  Edge features:     5  (river_dist, elev_diff, travel_time,")
    print(f"                         hand_diff, sar_wetness)")
    print(f"  High-sim pairs:    {(raw_sim > 0.8).sum()} / {E} edges (>0.8)")
    print(f"\n  DFC-GNN edge_attr is now [E, 5].")
    print(f"  Update model: change n_edge_feat=4 → n_edge_feat=5 in build_dfc_gnn()")
    print(f"\n  Next: python src/train_dfc_gnn.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract SAR wetness similarity for DFC-GNN edge features")
    p.add_argument("--verbose",  action="store_true",
                   help="Print per-node backscatter table and top similar pairs")
    p.add_argument("--dry-run",  action="store_true",
                   help="Compute W and similarity but do not modify edge_features.npz")
    p.add_argument("--proc-dir", type=Path, default=PROC_DIR,
                   help=f"Directory with processed sigma0 GeoTIFFs (default: {PROC_DIR})")
    args = p.parse_args()
    PROC_DIR = args.proc_dir
    run(args.verbose, args.dry_run)
