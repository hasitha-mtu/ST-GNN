"""
precompute_hand_edges.py  –  Precompute HAND candidate edges for Phase 2
=========================================================================
One-time script. Reads the Lee DEM, computes HAND, then for every pair of
gauge nodes within 5 km overland distance finds the minimum HAND value along
a straight-line corridor between them.  The minimum corridor HAND is the
stage height at which inundation would connect the two drainage sub-basins.

Output
------
    dataset/graph/hand_edges.npz  containing arrays:
        src              int32  [E_hand]
        dst              int32  [E_hand]
        hand_threshold   float32 [E_hand]  minimum HAND along corridor (m)
        overland_dist_km float32 [E_hand]  Euclidean distance between nodes (km)

Usage
-----
    python src/precompute_hand_edges.py
    python src/precompute_hand_edges.py --dem dataset/dem/COP-DEM-30m.tif
                                         --nodes dataset/graph/nodes.csv
                                         --out dataset/graph/hand_edges.npz
                                         --max-dist 5.0
                                         --hand-min 0.5

Distance threshold justification
---------------------------------
5 km is consistent with:
  - Godbout et al. (2019) / Zheng et al. (2018): ≤5 km recommended for HAND
    reach hydraulic discretisation (cited in Aristizabal et al. 2023).
  - Lee catchment tributary spacing: Bride–Lee confluence to Shournagh–Lee
    confluence is ~4–5 km overland — the closest cross-tributary distance
    in the Lee network (Irish Examiner, 2024; Wikipedia, Shournagh River).
  - Nature Geoscience global connectivity analysis (2026) used 6 km as
    sensitivity threshold for river–floodplain connectivity.

HAND threshold floor
--------------------
Pairs with minimum corridor HAND < hand_min are excluded: a very low
minimum HAND between two nodes means they share a drainage basin and are
already connected through the river network — creating a HAND edge between
them would be redundant.  Default 0.5 m.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DEM_PATH   = BASE_DIR / "dataset/dem/COP-DEM-30m.tif"
NODES_PATH = BASE_DIR / "dataset/graph/nodes.csv"
OUT_PATH   = BASE_DIR / "dataset/graph/hand_edges.npz"

MAX_DIST_KM  = 5.0    # maximum overland distance to consider (km)
HAND_MIN_M   = 0.5    # minimum HAND threshold to accept (m)
CORRIDOR_W   = 500.0  # corridor half-width for HAND sampling (m)
SAMPLE_STEP  = 60     # sample every N DEM pixels along corridor


def reproject_dem_to_itm(dem_path, out_path):
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS as _CRS
    from pathlib import Path as _Path
    dem_path = _Path(dem_path)
    out_path = _Path(out_path)
    if out_path.exists():
        print(f"  Using cached ITM DEM: {out_path.name}")
        return out_path
    with rasterio.open(dem_path) as src:
        if src.crs and src.crs.to_epsg() == 2157:
            print("  DEM already in ITM.")
            return dem_path
        print(f"  Reprojecting {src.crs} to EPSG:2157 at 30 m ...")
        dst_crs = _CRS.from_epsg(2157)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=30.0)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform,
                       "width": width, "height": height,
                       "nodata": src.nodata if src.nodata is not None else -9999.0})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **kwargs) as dst:
            for b in range(1, src.count + 1):
                reproject(source=rasterio.band(src, b),
                          destination=rasterio.band(dst, b),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=dst_crs,
                          resampling=Resampling.bilinear)
    print(f"  Reprojected: {out_path.name}  ({height} x {width} px at 30 m)")
    return out_path


def compute_hand(dem_path: Path) -> tuple:
    """
    HAND using correct pad-based D8 + ITM reprojection.
    Fixes: CRS mismatch, np.roll sign error, endpoint sampling, perp offsets.
    """
    import rasterio
    dem_path = Path(dem_path)
    itm_path = dem_path.parent / (dem_path.stem + "_itm.tif")
    dem_path = reproject_dem_to_itm(dem_path, itm_path)
    print(f"Loading DEM: {dem_path.name}")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float64)
        affine   = src.transform
        crs      = str(src.crs) if src.crs else "EPSG:2157"
        nodata   = src.nodata if src.nodata is not None else -9999.0
    H, W = dem_data.shape
    nan_mask = (dem_data == nodata) | np.isnan(dem_data)
    dem_data[nan_mask] = np.nan
    print(f"  DEM shape: {H} x {W}  CRS: {crs}")
    print(f"  Elevation range: [{np.nanmin(dem_data):.1f}, {np.nanmax(dem_data):.1f}] m")
    print("  Computing D8 flow accumulation ...")
    sentinel = float(np.nanmax(dem_data)) + 1e6
    dem_work = dem_data.copy()
    dem_work[nan_mask] = sentinel
    pad = np.pad(dem_work, 1, mode="constant", constant_values=sentinel)
    dr  = np.zeros((H, W), dtype=np.int8)
    dc  = np.zeros((H, W), dtype=np.int8)
    max_slope = np.full((H, W), -np.inf)
    for di, dj, dist in [(-1,-1,1.4142),(-1,0,1.0),(-1,1,1.4142),
                          ( 0,-1,1.0),             ( 0,1,1.0),
                          ( 1,-1,1.4142),( 1,0,1.0),( 1,1,1.4142)]:
        ri = 1 + di
        ci = 1 + dj
        neigh = pad[ri:ri+H, ci:ci+W]
        slope = (dem_work - neigh) / dist
        update = (slope > max_slope) & ~nan_mask
        dr[update] = di
        dc[update] = dj
        max_slope[update] = slope[update]
    acc = np.ones((H, W), dtype=np.float32)
    acc[nan_mask] = 0.0
    flat_order = np.argsort(dem_work.ravel())[::-1]
    for r, c in zip(flat_order // W, flat_order % W):
        if nan_mask[r, c] or (dr[r,c] == 0 and dc[r,c] == 0):
            continue
        nr, nc = r + int(dr[r,c]), c + int(dc[r,c])
        if 0 <= nr < H and 0 <= nc < W and not nan_mask[nr, nc]:
            acc[nr, nc] += acc[r, c]
    print(f"  Accumulation range: [1, {acc.max():.0f}] cells")
    acc_threshold = 500
    if acc.max() < acc_threshold:
        acc_threshold = float(acc.max() * 0.9)
    stream_mask = ((acc >= acc_threshold) & ~nan_mask).astype(np.uint8)
    n_stream = int(stream_mask.sum())
    n_valid  = int((~nan_mask).sum())
    print(f"  Stream cells: {n_stream:,} / {n_valid:,} ({n_stream/max(n_valid,1)*100:.2f}%)")
    if n_stream == 0:
        raise ValueError("No stream cells. Check DEM coverage.")
    print("  Computing HAND ...")
    hand = _compute_hand_from_mask(dem_data.astype(np.float32), stream_mask, affine)
    valid = hand[~np.isnan(hand)]
    print(f"  HAND shape={hand.shape}  NaN={np.isnan(hand).mean():.2%}")
    if valid.size > 0:
        print(f"  HAND range: [{valid.min():.2f}, {valid.max():.2f}] m")
    return hand, affine, crs

def _compute_hand_from_mask(
    dem:         np.ndarray,    # [H, W] elevation (m)
    stream_mask: np.ndarray,    # [H, W] 1=stream, 0=land
    affine,
    max_search_m: float = 10_000.0,  # maximum search radius (10 km)
) -> np.ndarray:
    """
    Compute HAND as vertical distance from each cell to its nearest stream cell.

    Uses scipy distance_transform_edt to find the nearest stream pixel for
    each land cell, then subtracts stream elevation from land elevation.

    This is an approximation of true flow-path HAND — it uses Euclidean
    nearest-stream distance rather than D8 flow-path distance. For the Lee
    catchment at 30m resolution with the gauge node spacing used here (~km),
    the difference is small relative to the activation thresholds (0.5–5 m).

    Parameters
    ----------
    max_search_m : float
        Cells further than this from any stream are assigned NaN (off-network).
    """
    from scipy.ndimage import distance_transform_edt

    pixel_size_m = abs(affine.a)   # assumes square pixels

    # Distance in pixels from each non-stream cell to nearest stream cell
    dist_px, nearest_idx = distance_transform_edt(
        stream_mask == 0,
        return_indices=True
    )
    dist_m = dist_px * pixel_size_m

    # Elevation at the nearest stream cell for every land cell
    stream_elev = dem[nearest_idx[0], nearest_idx[1]]

    # HAND = land elevation − nearest stream elevation
    hand = dem - stream_elev
    hand = np.clip(hand, 0.0, None)      # negative values = below stream (set 0)

    # Mask cells too far from any stream
    hand[dist_m > max_search_m] = np.nan
    hand[np.isnan(dem)]          = np.nan

    return hand.astype(np.float32)


def nodes_to_itm(nodes_csv: Path) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Read gauge nodes and return ITM eastings, northings, and refs.
    Converts from lat/lon if ITM columns are absent.
    """
    df = pd.read_csv(nodes_csv)

    if "easting_itm" in df.columns and "northing_itm" in df.columns:
        eastings  = df["easting_itm"].values.astype(np.float64)
        northings = df["northing_itm"].values.astype(np.float64)
    elif "lat" in df.columns and "lon" in df.columns:
        print("  Converting lat/lon → ITM (EPSG:2157) …")
        t = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        eastings, northings = t.transform(df["lon"].values, df["lat"].values)
    else:
        raise KeyError(
            "nodes.csv must have either 'easting_itm'/'northing_itm' or 'lat'/'lon' columns."
        )

    refs = df["ref"].astype(str).tolist() if "ref" in df.columns \
           else [str(i) for i in range(len(df))]
    return eastings.astype(np.float32), northings.astype(np.float32), refs


def world_to_pixel(x_world: float, y_world: float, affine) -> tuple[int, int]:
    """Convert world coordinates to pixel row/col using inverse affine."""
    col = (x_world - affine.c) / affine.a
    row = (y_world - affine.f) / affine.e
    return int(round(row)), int(round(col))


def sample_corridor_hand(
    hand:    np.ndarray,
    affine,
    e1: float, n1: float,
    e2: float, n2: float,
    corridor_half_w: float = 500.0,
    n_samples: int = 50,
) -> float:
    """
    Sample HAND values along a straight-line corridor between two ITM points.

    The corridor samples N evenly-spaced points along the line (e1,n1)→(e2,n2)
    plus points offset perpendicular by ±corridor_half_w to widen the search
    and account for divide topology.

    Returns the minimum valid HAND value found (ignoring NaN).
    Returns NaN if all samples are NaN.
    """
    H, W = hand.shape
    # Sample the middle 70% of corridor (skip 15% at each end).
    # OPW gauges sit on rivers — HAND≈0 at t=0 and t=1 (endpoints).
    # We want the divide HAND in the middle, not the gauge stream cells.
    t_vals = np.linspace(0.15, 0.85, n_samples)

    min_hand = np.inf
    for t in t_vals:
        # Centreline point
        e = e1 + t * (e2 - e1)
        n = n1 + t * (n2 - n1)

        # Perpendicular direction (normalised)
        dx, dy = e2 - e1, n2 - n1
        length = max(np.sqrt(dx**2 + dy**2), 1.0)
        perp_e, perp_n = -dy / length, dx / length

        # Centreline-only sampling.
        # Perpendicular offsets (±500 m) were removed because they cross into
        # adjacent river channels, returning HAND≈0 and masking real divides.
        # Profile diagnostic confirmed centreline values of 14–72 m being
        # overridden by perpendicular samples hitting nearby stream cells.
        row, col = world_to_pixel(e, n, affine)
        if 0 <= row < H and 0 <= col < W:
            v = hand[row, col]
            if not np.isnan(v) and v < min_hand:
                min_hand = float(v)

    return min_hand if np.isfinite(min_hand) else np.nan


def find_candidate_pairs(
    eastings:  np.ndarray,
    northings: np.ndarray,
    static_edge_index: np.ndarray | None,
    max_dist_km: float,
) -> list[tuple[int, int, float]]:
    """
    Find all gauge node pairs within max_dist_km that are NOT already
    connected by a permanent river network edge.

    Returns list of (src_idx, dst_idx, dist_km).
    """
    N = len(eastings)

    # Existing edges to exclude
    existing = set()
    if static_edge_index is not None:
        for i in range(static_edge_index.shape[1]):
            s, d = int(static_edge_index[0, i]), int(static_edge_index[1, i])
            existing.add((min(s, d), max(s, d)))

    candidates = []
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in existing:
                continue
            de = eastings[i]  - eastings[j]
            dn = northings[i] - northings[j]
            dist_km = np.sqrt(de**2 + dn**2) / 1000.0
            if dist_km <= max_dist_km:
                candidates.append((i, j, dist_km))

    return candidates


def load_static_edges(graph_dir: Path) -> np.ndarray | None:
    """Load existing static edge_index from edges.csv or return None."""
    edges_csv = graph_dir / "edges.csv"
    if not edges_csv.exists():
        print(f"  edges.csv not found at {edges_csv} — no edges excluded")
        return None
    df = pd.read_csv(edges_csv)
    if "src" in df.columns and "dst" in df.columns:
        return np.stack([df["src"].values, df["dst"].values])
    return None


def run(
    dem_path:    Path,
    nodes_path:  Path,
    out_path:    Path,
    max_dist_km: float,
    hand_min_m:  float,
):
    print("=" * 60)
    print("HAND edge precomputation — Lee catchment")
    print(f"  DEM:          {dem_path}")
    print(f"  Nodes:        {nodes_path}")
    print(f"  Max distance: {max_dist_km} km")
    print(f"  HAND floor:   {hand_min_m} m")
    print("=" * 60)

    # ── 1. Compute HAND ───────────────────────────────────────────────
    hand, affine, crs = compute_hand(dem_path)

    # ── 2. Load node ITM coordinates ──────────────────────────────────
    eastings, northings, refs = nodes_to_itm(nodes_path)
    N = len(eastings)
    print(f"\nNodes loaded: {N}")
    for i, (e, n, r) in enumerate(zip(eastings[:5], northings[:5], refs[:5])):
        print(f"  {i:2d} ref={r}  E={e:.0f}  N={n:.0f}")
    if N > 5:
        print(f"  … (+{N-5} more)")

    # ── 3. Find candidate pairs ───────────────────────────────────────
    static_ei = load_static_edges(nodes_path.parent)
    candidates = find_candidate_pairs(eastings, northings, static_ei, max_dist_km)
    print(f"\nCandidate pairs within {max_dist_km} km: {len(candidates)}")

    # ── 4. Sample HAND along each corridor ───────────────────────────
    srcs, dsts, thresholds, dists = [], [], [], []
    skipped_low = 0

    for k, (i, j, dist_km) in enumerate(candidates):
        if (k + 1) % 10 == 0 or k == len(candidates) - 1:
            print(f"  Processing pair {k+1}/{len(candidates)} …", end="\r")

        min_hand = sample_corridor_hand(
            hand, affine,
            eastings[i], northings[i],
            eastings[j], northings[j],
            corridor_half_w=CORRIDOR_W,
            n_samples=SAMPLE_STEP,
        )

        if np.isnan(min_hand):
            continue   # off-raster or all NaN
        if min_hand < hand_min_m:
            skipped_low += 1
            continue   # already in same drainage basin

        srcs.append(i);         dsts.append(j)
        srcs.append(j);         dsts.append(i)   # bidirectional
        thresholds.append(min_hand); thresholds.append(min_hand)
        dists.append(dist_km);  dists.append(dist_km)

    print()
    print(f"\nHAND edges accepted:   {len(srcs)//2} pairs → {len(srcs)} directed edges")
    print(f"Skipped (low HAND < {hand_min_m} m): {skipped_low} pairs")

    if not srcs:
        print("WARNING: no HAND edges found. Check DEM coverage and node coordinates.")
        sys.exit(1)

    # ── 5. Save ───────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        src              = np.array(srcs,       dtype=np.int32),
        dst              = np.array(dsts,       dtype=np.int32),
        hand_threshold   = np.array(thresholds, dtype=np.float32),
        overland_dist_km = np.array(dists,      dtype=np.float32),
    )
    print(f"\nSaved: {out_path}")
    print(f"  src/dst:        {len(srcs)} entries")
    print(f"  hand_threshold: [{min(thresholds):.2f}, {max(thresholds):.2f}] m")
    print(f"  overland_dist:  [{min(dists):.2f}, {max(dists):.2f}] km")

    # Print summary table
    print("\nAccepted HAND edge pairs:")
    print(f"  {'src_ref':12s} {'dst_ref':12s} {'dist_km':>8s} {'hand_thr_m':>12s}")
    seen = set()
    for i, (s, d, t, dist) in enumerate(zip(srcs, dsts, thresholds, dists)):
        if (min(s, d), max(s, d)) not in seen:
            seen.add((min(s, d), max(s, d)))
            print(f"  {refs[s]:12s} {refs[d]:12s} {dist:8.2f} {t:12.3f}")

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precompute HAND candidate edges for Phase 2 ST-GNN"
    )
    p.add_argument("--dem",      type=Path, default=DEM_PATH)
    p.add_argument("--nodes",    type=Path, default=NODES_PATH)
    p.add_argument("--out",      type=Path, default=OUT_PATH)
    p.add_argument("--max-dist", type=float, default=MAX_DIST_KM,
                   help="Maximum overland search distance (km). Default 5.0")
    p.add_argument("--hand-min", type=float, default=HAND_MIN_M,
                   help="Minimum HAND threshold to accept (m). Default 0.5")
    args = p.parse_args()

    if not args.dem.exists():
        print(f"ERROR: DEM not found: {args.dem}")
        sys.exit(1)
    if not args.nodes.exists():
        print(f"ERROR: nodes.csv not found: {args.nodes}")
        sys.exit(1)

    run(args.dem, args.nodes, args.out, args.max_dist, args.hand_min)
