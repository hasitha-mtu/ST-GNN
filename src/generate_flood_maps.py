"""
generate_flood_maps.py  –  HAND-based flood inundation maps from ST-GNN predictions
====================================================================================
Translates point-level stage predictions from any trained ST-GNN model into spatial
flood inundation maps using HAND (Height Above Nearest Drainage, Nobre et al. 2011).

Method
------
For each timestep t and each gauge node i:
  1. Predicted stage anomaly → absolute stage (add back mean stage at node i)
  2. For every DEM pixel in node i's drainage catchment:
       pixel is INUNDATED  if  HAND[pixel] ≤ predicted_stage[i, t]
       pixel is DRY        if  HAND[pixel] >  predicted_stage[i, t]
  3. Rasterise all 27 node inundation extents into one combined map

Output
------
  figures/flood_maps/
      flood_map_YYYY-MM-DD_HH-MM.png       — one PNG per timestep
      flood_animation.gif                   — animated flood evolution
      flood_animation.mp4                   — video (if ffmpeg available)
      peak_inundation_map.png               — maximum inundation across event
      inundation_area_timeseries.png        — total inundated area vs time

References
----------
Nobre, A.D. et al. (2011). Height Above the Nearest Drainage. J. Hydrol.
Aristizabal, F. et al. (2023). Extending HAND. Water Resources Research.
Zheng, X. et al. (2018). Enhancing HAND flood inundation. J. Hydrol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.patches import Patch
import rasterio
from scipy.spatial import cKDTree

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
PROC_DIR        = BASE_DIR / "dataset/processed"
GRAPH_DIR       = BASE_DIR / "dataset/graph"
DEM_DIR         = BASE_DIR / "dataset/dem"
FIG_DIR         = BASE_DIR / "figures/flood_maps_v6"
RIVER_SHAPEFILE = BASE_DIR / "shapefiles/RiverNetwork/RiverNetwork.shp"
LAKES_SHAPEFILE = BASE_DIR / "shapefiles/LeeLakes/LeeLakes.shp"

# Use the ITM-projected DEM (produced by precompute_hand_edges.py)
DEM_PATH    = DEM_DIR / "COP-DEM-30m_itm.tif"

# Default: use Phase 2 HAND-edge model checkpoint
CKPT_ROOT    = BASE_DIR / "checkpoints"
DEFAULT_CKPT = CKPT_ROOT / "st_gnn_hand_edge"


# ══════════════════════════════════════════════════════════════════════
#  Step 1: Load DEM and precompute per-node drainage catchments
# ══════════════════════════════════════════════════════════════════════

def load_dem_itm(dem_path: Path) -> tuple:
    """Load ITM-projected DEM. Returns (dem_array, affine, H, W)."""
    with rasterio.open(dem_path) as src:
        dem  = src.read(1).astype(np.float32)
        aff  = src.transform
        nd   = src.nodata if src.nodata is not None else -9999.0
    dem[dem == nd] = np.nan
    print(f"DEM loaded: {dem.shape}  pixel={abs(aff.a):.1f} m")
    return dem, aff, dem.shape[0], dem.shape[1]


# ══════════════════════════════════════════════════════════════════════
#  D8 upstream catchment masks — replaces Voronoi nearest-neighbour
# ══════════════════════════════════════════════════════════════════════

FDIR_PATH = BASE_DIR / "dataset/graph/fdir.npz"


def build_d8_catchment_masks(
    node_itm,
    affine,
    H: int,
    W: int,
    fdir_path=None,
) -> list:
    """
    Assign each DEM pixel to its nearest DOWNSTREAM gauge using D8 flow
    direction rather than Euclidean distance (Voronoi).

    The Voronoi partition produces straight perpendicular boundaries between
    adjacent gauge catchments, creating triangular flood footprints with
    edges that cross the floodplain at right angles — physically meaningless.

    D8 catchment delineation assigns each pixel to whichever gauge its
    water flows through first, following the actual drainage network.
    Boundaries follow watershed divides (ridgelines), not straight lines.

    Algorithm: multi-source BFS on the reversed D8 graph.
      1. Snap each gauge to its raster pixel; mark as claimed.
      2. Simultaneously propagate all N BFS wavefronts upstream.
         A pixel at (nr, nc) is claimed by gauge G if:
           dr[nr, nc] == di  AND  dc[nr, nc] == dj
         where (di, dj) is the direction from (nr, nc) to the already-
         claimed pixel (r, c) that is propagating.  This ensures only
         pixels that *actually flow into* the gauge are claimed.
      3. Pixels that never reach any gauge (outside the gauged network,
         in depressions) fall back to Voronoi.

    Requires dataset/graph/fdir.npz from precompute_hand_edges.py.
    Falls back to Voronoi with a warning if the file is missing.
    """
    from collections import deque

    fp = fdir_path or FDIR_PATH
    if not Path(fp).exists():
        print(f"  WARNING: {fp} not found — falling back to Voronoi.")
        print(f"  Run precompute_hand_edges.py to generate fdir.npz.")
        return _voronoi_masks(node_itm, affine, H, W)

    data     = np.load(fp)
    dr       = data["dr"]
    dc       = data["dc"]
    nan_mask = data["nan_mask"].astype(bool)
    N        = len(node_itm)

    assignment = np.full((H, W), -1, dtype=np.int16)

    # Snap gauge ITM coordinates to raster pixels
    gauge_pixels = []
    for i in range(N):
        e, n = float(node_itm[i, 0]), float(node_itm[i, 1])
        row  = int(np.clip(round((n - affine.f) / affine.e), 0, H - 1))
        col  = int(np.clip(round((e - affine.c) / affine.a), 0, W - 1))
        if not nan_mask[row, col]:
            assignment[row, col] = i
            gauge_pixels.append((row, col, i))

    # Multi-source BFS: propagate upstream through reverse D8
    queue = deque(gauge_pixels)
    while queue:
        r, c, gid = queue.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nr, nc = r - di, c - dj
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if nan_mask[nr, nc] or assignment[nr, nc] >= 0:
                    continue
                if dr[nr, nc] == di and dc[nr, nc] == dj:
                    assignment[nr, nc] = gid
                    queue.append((nr, nc, gid))

    # Voronoi fallback for pixels unreachable by any gauge flow path
    unclaimed = (assignment < 0) & ~nan_mask
    n_unclaimed = int(unclaimed.sum())
    if n_unclaimed > 0:
        print(f"  D8: {n_unclaimed:,} unclaimed pixels "
              f"({n_unclaimed/(H*W)*100:.2f}%) → Voronoi fallback")
        fb = _voronoi_masks(node_itm, affine, H, W)
        for i in range(N):
            assignment[unclaimed & fb[i]] = i

    n_assigned = int((assignment >= 0).sum())
    n_valid    = int((~nan_mask).sum())
    print(f"  D8 catchment: {N} nodes  "
          f"{n_assigned:,}/{n_valid:,} valid pixels assigned")
    return [assignment == i for i in range(N)]


def _voronoi_masks(node_itm, affine, H, W):
    """Euclidean nearest-neighbour fallback when fdir.npz is unavailable."""
    from scipy.spatial import cKDTree
    rows, cols = np.mgrid[0:H, 0:W]
    px_e = affine.c + cols * affine.a
    px_n = affine.f + rows * affine.e
    tree = cKDTree(node_itm)
    dist, nearest = tree.query(
        np.column_stack([px_e.ravel(), px_n.ravel()]), workers=-1
    )
    nearest = nearest.reshape(H, W)
    dist    = dist.reshape(H, W) * abs(affine.a)
    return [(nearest == i) & (dist <= 8000) for i in range(len(node_itm))]

# ══════════════════════════════════════════════════════════════════════
#  Step 2: Compute HAND raster (if not already saved)
# ══════════════════════════════════════════════════════════════════════

def get_or_compute_hand(dem: np.ndarray, affine, dem_path: Path) -> np.ndarray:
    """
    Load a precomputed HAND raster or compute it on the fly.

    Priority:
      1. hand_raster.tif alongside the DEM (saved from previous run)
      2. Compute using scipy distance_transform_edt (no pysheds/Numba)
    """
    hand_path = dem_path.parent / "hand_raster.tif"

    if hand_path.exists():
        with rasterio.open(hand_path) as src:
            hand = src.read(1).astype(np.float32)
        print(f"HAND loaded from cache: {hand_path.name}")
        return hand

    print("Computing HAND raster (scipy, no Numba) ...")
    from scipy.ndimage import distance_transform_edt

    H, W     = dem.shape
    nan_mask = np.isnan(dem)

    # Simple stream network: top 0.55% of flow accumulation
    # (reuse the D8 from precompute_hand_edges.py logic)
    sentinel  = float(np.nanmax(dem)) + 1e6
    dem_work  = dem.copy()
    dem_work[nan_mask] = sentinel
    pad = np.pad(dem_work, 1, mode="constant", constant_values=sentinel)

    dr = np.zeros((H, W), dtype=np.int8)
    dc = np.zeros((H, W), dtype=np.int8)
    max_slope = np.full((H, W), -np.inf)
    for di, dj, dist in [(-1,-1,1.4142),(-1,0,1.0),(-1,1,1.4142),
                          ( 0,-1,1.0),             ( 0,1,1.0),
                          ( 1,-1,1.4142),( 1,0,1.0),( 1,1,1.4142)]:
        ri, ci = 1+di, 1+dj
        neigh  = pad[ri:ri+H, ci:ci+W]
        slope  = (dem_work - neigh) / dist
        update = (slope > max_slope) & ~nan_mask
        dr[update] = di; dc[update] = dj; max_slope[update] = slope[update]

    acc = np.ones((H, W), dtype=np.float32)
    acc[nan_mask] = 0.0
    flat = np.argsort(dem_work.ravel())[::-1]
    for r, c in zip(flat // W, flat % W):
        if nan_mask[r,c] or (dr[r,c]==0 and dc[r,c]==0): continue
        nr, nc = r+int(dr[r,c]), c+int(dc[r,c])
        if 0<=nr<H and 0<=nc<W and not nan_mask[nr,nc]:
            acc[nr,nc] += acc[r,c]

    stream_mask = (acc >= 500).astype(np.uint8)
    stream_mask[nan_mask] = 0

    # HAND = elevation - nearest stream elevation
    dist_px, idx = distance_transform_edt(stream_mask == 0, return_indices=True)
    stream_elev  = dem[idx[0], idx[1]]
    hand = np.clip(dem - stream_elev, 0, None)
    hand[nan_mask] = np.nan

    # Cache
    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()
    meta["dtype"] = "float32"
    with rasterio.open(hand_path, "w", **meta) as dst:
        dst.write(hand, 1)
    print(f"HAND computed and cached: {hand_path.name}")
    return hand


# ══════════════════════════════════════════════════════════════════════
#  Step 3: Load model predictions and node stage data
# ══════════════════════════════════════════════════════════════════════

def load_predictions_and_truth(
    ckpt_dir:   Path,
    proc_dir:   Path,
    graph_dir:  Path,
) -> tuple:
    """
    Load test-set predictions, ground truth, and node metadata.

    Returns
    -------
    pred_stage   : np.ndarray [T_test, N]  absolute predicted stage (m)
    true_stage   : np.ndarray [T_test, N]  absolute observed stage (m)
    timestamps   : pd.DatetimeIndex        test window timestamps
    node_itm     : np.ndarray [N, 2]       node ITM coordinates [E, N]
    node_refs    : list[str]               OPW gauge references
    node_mean_stage : np.ndarray [N]       per-node mean stage for denormalisation
    """
    # Load raw dataset
    X = np.load(proc_dir / "X.npy", mmap_mode="r")   # [T, N, F]
    y = np.load(proc_dir / "y.npy", mmap_mode="r")   # [T, N]

    # Load timestamps
    ts_path = proc_dir / "timestamps.csv"
    all_ts  = pd.to_datetime(pd.read_csv(ts_path)["timestamp"])

    # Load dataset metadata from the shared processed directory.
    # The file is common to all models — not stored per-checkpoint.
    meta_candidates = [
        proc_dir / "dataset_metadata.json",   # shared location (primary)
        ckpt_dir / "dataset_metadata.json",   # per-checkpoint (legacy)
    ]
    meta_path = next((p for p in meta_candidates if p.exists()), None)
    if meta_path:
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    T_in  = meta.get("t_in",  32)
    T_out = meta.get("t_out",  4)
    T     = X.shape[0]
    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)
    test_start = val_end
    test_end   = T - T_out

    # Per-node mean stage (feature [1] = normalized_stage, feature [0] = stage_anomaly)
    # Absolute stage ≈ stage_anomaly + mean_stage_at_node
    # We reconstruct from y (which is absolute stage) and X[:,0] (anomaly)
    # Mean stage per node = mean of y[:train_end] over time axis
    mean_stage = np.nanmean(y[:train_end], axis=0)   # [N]

    # Test ground truth: y at test timesteps (absolute stage)
    true_stage = y[test_start:test_end]               # [T_test, N]

    # Load saved test predictions if available
    mean_files  = sorted(ckpt_dir.glob("test_predictions_*steps_mean.npy"))
    single_file = ckpt_dir / "test_predictions.npy"
    if mean_files:
        pred_stage = np.load(mean_files[0])
        print("Ensemble mean loaded:", mean_files[0].name, pred_stage.shape)
    elif single_file.exists():
        pred_stage = np.load(single_file)
        print("Predictions loaded:", pred_stage.shape)
    else:
        print("No predictions found — using ground truth as proxy.")
        print("Run: python src/run_inference.py --model", ckpt_dir.name)
        pred_stage = true_stage.copy()
        rng = np.random.default_rng(42)
        pred_stage += rng.normal(0, 0.02, pred_stage.shape)

        # Reset to zero-based positional index so timestamps[i] means
    # "i-th timestep in the test set", not original dataset label.
    test_ts = pd.DatetimeIndex(all_ts.iloc[test_start:test_end].values)

    # Load node ITM coordinates
    nodes_df   = pd.read_csv(graph_dir / "nodes.csv")
    if "easting_itm" in nodes_df.columns:
        node_itm = nodes_df[["easting_itm", "northing_itm"]].values
    else:
        # Convert from lat/lon
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        E, N = t.transform(nodes_df["lon"].values, nodes_df["lat"].values)
        node_itm = np.column_stack([E, N])

    node_refs = nodes_df["ref"].astype(str).tolist()

    return pred_stage, true_stage, test_ts, node_itm, node_refs, mean_stage


# ══════════════════════════════════════════════════════════════════════
#  Step 4: Generate inundation map for a single timestep
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
#  Per-node bankfull stage thresholds — robust short-record method
# ══════════════════════════════════════════════════════════════════════

def compute_bankfull_thresholds(
    X:           np.ndarray,
    train_end:   int,
    stage_col:   int   = 0,
    node_refs:   list | None = None,
    json_path          = None,
) -> tuple:
    """
    Per-node bankfull stage threshold — three-tier priority method.

    Priority 1 — bankfull_thresholds.json (when available)
        Pre-derived by derive_bankfull_thresholds.py using OPW station
        data: gaugings width expansion > Gumbel annual maxima 1.5-yr
        return level > obs_max×1.3 fallback.  Most defensible for paper.

    Priority 2 — obs_max × 1.3 (training data only)
        OPW short-record approach (NERC FSR 1975).  Used when the JSON
        is absent or a node is missing from it.

    Returns
    -------
    thresholds : np.ndarray [N]   per-node stage anomaly threshold (m)
    reliable   : np.ndarray [N]   True = well-sampled, False = flagged
    """
    import json as _json

    N          = X.shape[1]
    thresholds = np.zeros(N, dtype=np.float32)
    reliable   = np.ones(N,  dtype=bool)

    # ── Priority 1: load from pre-derived JSON ─────────────────────────
    _jp = Path(json_path) if json_path else \
          BASE_DIR / "dataset/graph/bankfull_thresholds.json"
    if _jp.exists():
        with open(_jp) as fh:
            _data = _json.load(fh)
        thr_map  = _data.get("thresholds", {})
        n_loaded = 0
        for i, ref in enumerate(node_refs or []):
            if str(ref) in thr_map:
                thresholds[i] = float(thr_map[str(ref)])
                n_loaded += 1
            else:
                thresholds[i] = 0.1
                reliable[i]   = False
        thresholds = np.maximum(thresholds, 0.05)
        print(f"  Bankfull thresholds: loaded from {_jp.name} "
              f"({n_loaded}/{N} nodes from OPW station data)")
        for i, ref in enumerate(node_refs or []):
            status = "OK" if reliable[i] else "⚠ fallback"
            print(f"    {str(ref):>7}  {thresholds[i]:.3f} m  {status}")
        return thresholds, reliable

    # ── Priority 2: obs_max × 1.3 from training data ───────────────────
    print("  bankfull_thresholds.json not found — using obs_max×1.3 fallback.")
    print("  Run: python src/derive_bankfull_thresholds.py")
    stage_data  = X[:train_end, :, stage_col]
    node_maxima = np.nanmax(stage_data, axis=0)
    p10_max     = float(np.nanpercentile(node_maxima, 10))

    for i in range(N):
        s = stage_data[:, i]; s = s[~np.isnan(s)]
        if s.size == 0:
            thresholds[i] = 0.1; reliable[i] = False; continue
        obs_max = float(s.max()); p99 = float(np.percentile(s, 99))
        thresholds[i] = max(min(obs_max * 1.3, p99 * 2.0), 0.05)
        if obs_max < p10_max:
            reliable[i] = False

    TIDAL_REFS     = {"19162","19163","19161","19160"}
    RESERVOIR_REFS = {"19094","19095"}
    if node_refs is not None:
        for i, ref in enumerate(node_refs):
            if ref in TIDAL_REFS or ref in RESERVOIR_REFS:
                thresholds[i] = min(thresholds[i] * 1.5, 2.0)

    thresholds = np.maximum(thresholds, 0.05)
    years = train_end / (365.25 * 24 * 4)
    n_unreliable = int((~reliable).sum())
    print(f"  obs_max×1.3  record={years:.1f} yr  "
          f"unreliable={n_unreliable}/{N}")
    return thresholds, reliable

def stage_to_inundation(
    stage:   np.ndarray,    # [N] predicted absolute stage per node (m)
    hand:    np.ndarray,    # [H, W] HAND raster (m)
    masks:   list,          # N boolean arrays [H, W]
    thresholds = None,  # [N] per-node bankfull thresholds; None → 0.1 m uniform
) -> np.ndarray:
    """
    Convert predicted stage at N nodes to a binary inundation map.
    Pixels flood when HAND[pixel] <= stage[node] >= threshold[node].
    Pass thresholds=None to use the legacy uniform 0.1 m value.
    """
    H, W = hand.shape
    result = np.zeros((H, W), dtype=bool)
    thr = (np.full(len(stage), 0.1, dtype=np.float32)
           if thresholds is None else np.asarray(thresholds))

    for i, (s, mask, t) in enumerate(zip(stage, masks, thr)):
        if s < t or np.isnan(s):
            continue
        in_zone = mask & ~np.isnan(hand)
        result |= in_zone & (hand <= s)

    return result



# ══════════════════════════════════════════════════════════════════════
#  Shapefile loading — rivers and lakes
# ══════════════════════════════════════════════════════════════════════

def load_shapefiles(
    rivers_path: Path | None,
    lakes_path:  Path | None,
    target_crs:  str = "EPSG:2157",
) -> tuple:
    """
    Load river network and lake shapefiles, reproject to ITM (EPSG:2157)
    so coordinates align with the DEM and node positions.

    Both files are optional. Pass None to skip either layer.

    Returns
    -------
    rivers_gdf : GeoDataFrame or None
    lakes_gdf  : GeoDataFrame or None
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("  WARNING: geopandas not installed — shapefiles skipped.")
        print("  Install with: pip install geopandas")
        return None, None

    rivers_gdf = None
    lakes_gdf  = None

    if rivers_path and Path(rivers_path).exists():
        rivers_gdf = gpd.read_file(rivers_path)
        if rivers_gdf.crs is None:
            print(f"  WARNING: {rivers_path.name} has no CRS — assuming EPSG:4326")
            rivers_gdf = rivers_gdf.set_crs("EPSG:4326")
        if str(rivers_gdf.crs).upper() != target_crs.upper():
            rivers_gdf = rivers_gdf.to_crs(target_crs)
        print(f"  Rivers loaded: {len(rivers_gdf)} features  CRS→{target_crs}")
    elif rivers_path:
        print(f"  WARNING: rivers shapefile not found: {rivers_path}")

    if lakes_path and Path(lakes_path).exists():
        lakes_gdf = gpd.read_file(lakes_path)
        if lakes_gdf.crs is None:
            print(f"  WARNING: {lakes_path.name} has no CRS — assuming EPSG:4326")
            lakes_gdf = lakes_gdf.set_crs("EPSG:4326")
        if str(lakes_gdf.crs).upper() != target_crs.upper():
            lakes_gdf = lakes_gdf.to_crs(target_crs)
        print(f"  Lakes loaded:  {len(lakes_gdf)} features  CRS→{target_crs}")
    elif lakes_path:
        print(f"  WARNING: lakes shapefile not found: {lakes_path}")

    return rivers_gdf, lakes_gdf


def geometries_to_pixel(gdf, affine) -> list:
    """
    Convert all geometry coordinates in a GeoDataFrame from ITM metres
    to raster pixel (col, row) coordinates using the DEM affine transform.

    Returns a list of Shapely geometries with pixel-space coordinates,
    ready to be drawn with matplotlib descartes / direct coordinate iteration.
    """
    from shapely.affinity import affine_transform
    # Affine from ITM to pixel:
    #   col = (E - affine.c) / affine.a
    #   row = (N - affine.f) / affine.e
    # Shapely affine_transform: [a, b, d, e, xoff, yoff] → x*a + y*b + xoff
    a  = 1.0 / affine.a
    e_ = 1.0 / affine.e
    c_ = -affine.c / affine.a
    f_ = -affine.f / affine.e
    # shapely convention: new_x = a*x + b*y + xoff
    #                     new_y = d*x + e*y + yoff
    matrix = [a, 0, 0, e_, c_, f_]
    return [affine_transform(geom, matrix) for geom in gdf.geometry]


# ══════════════════════════════════════════════════════════════════════
#  Step 5: Plotting utilities
# ══════════════════════════════════════════════════════════════════════

def make_colormap():
    """Build a DEM hillshade + flood overlay colormap."""
    flood_cmap = mcolors.ListedColormap(["none", "#2166AC"])
    return flood_cmap


def _draw_polygon(ax, geom, facecolor, edgecolor, linewidth, zorder):
    """Draw a Shapely polygon or multipolygon in pixel coordinates."""
    from shapely.geometry import MultiPolygon, Polygon
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if poly.is_empty:
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, facecolor=facecolor, edgecolor=edgecolor,
                linewidth=linewidth, zorder=zorder)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, facecolor="white", edgecolor=edgecolor,
                    linewidth=linewidth * 0.6, zorder=zorder)


def _draw_linestring(ax, geom, color, linewidth, alpha, zorder):
    """Draw a Shapely linestring or multilinestring in pixel coordinates."""
    from shapely.geometry import MultiLineString, LineString
    lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
    for line in lines:
        if line.is_empty:
            continue
        xs, ys = line.xy
        ax.plot(xs, ys, color=color, linewidth=linewidth,
                alpha=alpha, zorder=zorder)


def plot_flood_frame(
    dem:         np.ndarray,
    inundation:  np.ndarray,
    node_itm:    np.ndarray,
    node_stage:  np.ndarray,
    node_refs:   list,
    timestamp:   pd.Timestamp,
    affine,
    ax=None,
    title_suffix: str = "",
    rivers_gdf   = None,
    lakes_gdf    = None,
) -> plt.Figure:
    """
    Render one flood map frame: hillshaded DEM + blue inundation + gauge markers.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.figure

    # Hillshade DEM background
    valid = dem.copy()
    valid[np.isnan(valid)] = 0
    from matplotlib.colors import LightSource
    ls   = LightSource(azdeg=315, altdeg=45)
    # rgb  = ls.shade(valid, cmap=plt.cm.terrain,
    #                 vmin=-10, vmax=700, blend_mode="soft")
    rgb = ls.shade(valid, cmap=plt.cm.Greys_r,
                   vmin=0, vmax=700, blend_mode="overlay")
    # rgb = ls.shade(valid, cmap=make_colormap(),
    #                vmin=-10, vmax=700, blend_mode="soft")
    ax.imshow(rgb, origin="upper")

    # Flood overlay (semi-transparent blue)
    flood_rgba = np.zeros((*inundation.shape, 4), dtype=np.float32)
    flood_rgba[inundation, 0] = 0.13   # R
    flood_rgba[inundation, 1] = 0.40   # G
    flood_rgba[inundation, 2] = 0.67   # B
    flood_rgba[inundation, 3] = 0.75   # alpha
    ax.imshow(flood_rgba, origin="upper")

    # ── Lakes (drawn before rivers so rivers appear on top) ──────────
    if lakes_gdf is not None and len(lakes_gdf) > 0:
        lake_geoms = geometries_to_pixel(lakes_gdf, affine)
        for geom in lake_geoms:
            if geom is None or geom.is_empty:
                continue
            _draw_polygon(ax, geom,
                          facecolor=(0.18, 0.52, 0.78, 0.55),
                          edgecolor=(0.10, 0.35, 0.60, 0.9),
                          linewidth=0.6, zorder=3)

    # ── River network ─────────────────────────────────────────────────
    if rivers_gdf is not None and len(rivers_gdf) > 0:
        river_geoms = geometries_to_pixel(rivers_gdf, affine)
        for geom in river_geoms:
            if geom is None or geom.is_empty:
                continue
            _draw_linestring(ax, geom,
                             color=(0.08, 0.30, 0.65),
                             linewidth=0.7, alpha=0.85, zorder=4)

    # Gauge node markers
    for i, (e, n) in enumerate(node_itm):
        row = (n - affine.f) / affine.e
        col = (e - affine.c) / affine.a
        s   = node_stage[i]
        clr = "red" if s > 0.3 else "orange" if s > 0.1 else "white"
        ax.plot(col, row, "o", color=clr, markersize=5,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        if s > 0.15:   # label high-stage gauges only
            ax.annotate(
                f"{node_refs[i]}\n{s:.2f}m",
                xy=(col, row), xytext=(4, 4), textcoords="offset points",
                fontsize=6, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
                zorder=6,
            )

    # Legend and title
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#2166AC", alpha=0.75, label="Predicted inundation"),
        Patch(facecolor="red",     label="High stage (>0.3 m anomaly)"),
        Patch(facecolor="orange",  label="Moderate stage (>0.1 m)"),
        Patch(facecolor="white",   label="Low stage"),
    ]
    if rivers_gdf is not None:
        legend_elements.append(
            Line2D([0],[0], color=(0.08,0.30,0.65), lw=1.2, label="River network")
        )
    if lakes_gdf is not None:
        legend_elements.append(
            Patch(facecolor=(0.18,0.52,0.78,0.55),
                  edgecolor=(0.10,0.35,0.60), label="Lakes / reservoirs")
        )
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
              framealpha=0.8)

    n_flooded = inundation.sum()
    pixel_m   = abs(affine.a)
    area_km2  = n_flooded * (pixel_m**2) / 1e6
    ax.set_title(
        f"Lee Catchment Flood Inundation  |  {timestamp.strftime('%Y-%m-%d %H:%M')}"
        f"{title_suffix}\nInundated area: {area_km2:.2f} km²",
        fontsize=10, pad=8,
    )
    ax.set_xlabel("Column (ITM)")
    ax.set_ylabel("Row (ITM)")
    ax.tick_params(labelsize=7)

    return fig


# ══════════════════════════════════════════════════════════════════════
#  Step 6: Main — find flood event, generate maps and animation
# ══════════════════════════════════════════════════════════════════════

def find_flood_events(
    true_stage:  np.ndarray,   # [T, N]
    timestamps:  pd.DatetimeIndex,
    threshold:   float = 0.3,  # stage anomaly threshold for "flood"
    min_duration_h: int = 6,   # minimum event duration
) -> list[tuple]:
    """
    Identify flood event windows where ANY node exceeds threshold.
    Returns list of (start_idx, end_idx, peak_stage, peak_ts).
    """
    max_stage = np.nanmax(true_stage, axis=1)   # [T] max across nodes
    in_flood  = max_stage > threshold

    # Find contiguous event windows
    events = []
    i = 0
    while i < len(in_flood):
        if in_flood[i]:
            j = i
            while j < len(in_flood) and in_flood[j]:
                j += 1
            duration_h = (j - i) * 15 / 60
            if duration_h >= min_duration_h:
                peak_idx = int(i + np.argmax(max_stage[i:j]))
                events.append((i, j, float(max_stage[peak_idx]),
                               timestamps[peak_idx]))
            i = j
        else:
            i += 1

    events.sort(key=lambda e: -e[2])   # sort by peak stage descending
    return events


def generate_all(
    ckpt_dir:     Path,
    proc_dir:     Path,
    graph_dir:    Path,
    dem_path:     Path,
    out_dir:      Path,
    n_events:     int  = 2,
    fps:          int  = 4,
    max_frames:   int  = 96,
    model_label:  str  = "STGNNHANDEdge",
    rivers_gdf         = None,
    lakes_gdf          = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n── Loading data ──")
    dem, affine, H, W = load_dem_itm(dem_path)
    hand              = get_or_compute_hand(dem, affine, dem_path)

    pred_stage, true_stage, timestamps, node_itm, node_refs, mean_stage = \
        load_predictions_and_truth(ckpt_dir, proc_dir, graph_dir)

    # ── Per-node bankfull thresholds (robust short-record method) ─
    X_full    = np.load(proc_dir / "X.npy", mmap_mode="r")
    train_end = int(X_full.shape[0] * 0.70)
    print("\n── Per-node bankfull thresholds ──")
    bankfull, reliable = compute_bankfull_thresholds(
        X_full, train_end, stage_col=0, node_refs=node_refs
    )
    del X_full

    N = len(node_refs)
    print(f"Test set: {len(timestamps)} timesteps  {N} nodes")

    # Build catchment masks
    print("\n── Building catchment masks ──")
    H_dem, W_dem = dem.shape
    masks = build_d8_catchment_masks(node_itm, affine, H_dem, W_dem)

    # Find flood events in ground truth
    print("\n── Finding flood events ──")
    events = find_flood_events(true_stage, timestamps)
    print(f"Found {len(events)} flood events (threshold 0.3 m, min 6 h)")
    if not events:
        print("No events found. Lowering threshold to 0.1 m ...")
        events = find_flood_events(true_stage, timestamps, threshold=0.1,
                                   min_duration_h=3)
    if not events:
        print("No events found. Generating map for peak stage timestep.")
        peak_t = int(np.nanargmax(np.nanmax(true_stage, axis=1)))
        events = [(max(0, peak_t-48), min(len(timestamps), peak_t+48),
                   float(np.nanmax(true_stage[peak_t])),
                   timestamps[peak_t])]

    print(f"Top events:")
    for i, (s, e, pk, ts) in enumerate(events[:3]):
        print(f"  {i+1}. {timestamps[s].date()} – {timestamps[e-1].date()}  "
              f"peak={pk:.3f}m  duration={int((e-s)*15/60)}h")

    inundation_areas = []
    inundation_ts    = []

    for ev_idx, (ev_start, ev_end, ev_peak, ev_ts) in enumerate(events[:n_events]):
        print(f"\n── Event {ev_idx+1}: {timestamps[ev_start].date()} "
              f"(peak {ev_peak:.3f} m at {ev_ts}) ──")

        ev_dir = out_dir / f"event_{ev_idx+1}_{timestamps[ev_start].strftime('%Y%m%d')}"
        ev_dir.mkdir(exist_ok=True)

        # Subsample frames for the animation
        ev_len   = ev_end - ev_start
        step     = max(1, ev_len // max_frames)
        frame_idxs = list(range(ev_start, ev_end, step))[:max_frames]

        print(f"  Generating {len(frame_idxs)} frames ...")

        frames_pred = []
        frames_true = []
        areas_pred  = []
        areas_true  = []
        pixel_m     = abs(affine.a)

        for t_idx in frame_idxs:
            ps = pred_stage[t_idx]
            ts_val = true_stage[t_idx]

            inn_pred = stage_to_inundation(ps,     hand, masks, bankfull)
            inn_true = stage_to_inundation(ts_val, hand, masks, bankfull)

            frames_pred.append(inn_pred)
            frames_true.append(inn_true)
            areas_pred.append(inn_pred.sum() * pixel_m**2 / 1e6)
            areas_true.append(inn_true.sum() * pixel_m**2 / 1e6)

        inundation_areas.extend(areas_pred)
        inundation_ts.extend([timestamps[i] for i in frame_idxs])

        # ── Peak inundation map ───────────────────────────────────────
        peak_local = np.argmax(areas_pred)
        peak_fig   = plot_flood_frame(
            dem, frames_pred[peak_local], node_itm,
            pred_stage[frame_idxs[peak_local]], node_refs,
            timestamps[frame_idxs[peak_local]], affine,
            title_suffix=f"  [{model_label}]",
            rivers_gdf=rivers_gdf, lakes_gdf=lakes_gdf,
        )
        peak_path = ev_dir / "peak_inundation.png"
        peak_fig.savefig(peak_path, dpi=150, bbox_inches="tight")
        plt.close(peak_fig)
        print(f"  Saved: {peak_path.name}")

        # ── Comparison: predicted vs observed at peak ─────────────────
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        plot_flood_frame(dem, frames_pred[peak_local], node_itm,
                         pred_stage[frame_idxs[peak_local]], node_refs,
                         timestamps[frame_idxs[peak_local]], affine, ax=axes[0],
                         title_suffix=f" — Predicted [{model_label}]",
                         rivers_gdf=rivers_gdf, lakes_gdf=lakes_gdf)
        plot_flood_frame(dem, frames_true[peak_local], node_itm,
                         true_stage[frame_idxs[peak_local]], node_refs,
                         timestamps[frame_idxs[peak_local]], affine, ax=axes[1],
                         title_suffix=" — Observed",
                         rivers_gdf=rivers_gdf, lakes_gdf=lakes_gdf)
        fig.suptitle(
            f"Lee Catchment Flood Inundation — Predicted vs Observed\n"
            f"Peak: {timestamps[frame_idxs[peak_local]].strftime('%Y-%m-%d %H:%M')}  "
            f"| Predicted area: {areas_pred[peak_local]:.2f} km²  "
            f"| Observed area: {areas_true[peak_local]:.2f} km²",
            fontsize=11
        )
        fig.tight_layout()
        cmp_path = ev_dir / "peak_comparison_pred_vs_observed.png"
        fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {cmp_path.name}")

        # ── Area timeseries ───────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ts_arr = [timestamps[i] for i in frame_idxs]
        ax2.fill_between(ts_arr, areas_pred, alpha=0.4, color="#2166AC",
                         label=f"Predicted ({model_label})")
        ax2.plot(ts_arr, areas_pred, color="#2166AC", lw=1.5)
        ax2.fill_between(ts_arr, areas_true, alpha=0.3, color="#D73027",
                         label="Observed")
        ax2.plot(ts_arr, areas_true, color="#D73027", lw=1.5, ls="--")
        ax2.set_ylabel("Inundated Area (km²)", fontsize=10)
        ax2.set_xlabel("Time", fontsize=10)
        ax2.set_title(
            f"Lee Catchment — Inundated Area During Flood Event "
            f"({timestamps[ev_start].strftime('%Y-%m-%d')})",
            fontsize=10
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        fig2.autofmt_xdate()
        fig2.tight_layout()
        area_path = ev_dir / "inundation_area_timeseries.png"
        fig2.savefig(area_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {area_path.name}")

        # ── Animation (GIF) ───────────────────────────────────────────
        print(f"  Building animation ({len(frames_pred)} frames) ...")
        gif_fig, gif_ax = plt.subplots(figsize=(11, 7))

        def make_frame(i):
            gif_ax.clear()
            plot_flood_frame(
                dem, frames_pred[i], node_itm,
                pred_stage[frame_idxs[i]], node_refs,
                timestamps[frame_idxs[i]], affine,
                ax=gif_ax,
                title_suffix=f"  [{model_label}]",
                rivers_gdf=rivers_gdf, lakes_gdf=lakes_gdf
            )

        ani = animation.FuncAnimation(
            gif_fig, make_frame, frames=len(frames_pred),
            interval=1000 // fps, repeat=True,
        )

        gif_path = ev_dir / "flood_animation.gif"
        writer   = animation.PillowWriter(fps=fps)
        ani.save(str(gif_path), writer=writer, dpi=100)
        plt.close(gif_fig)
        print(f"  Saved: {gif_path.name}  ({len(frames_pred)} frames at {fps} fps)")

        # Try MP4 if ffmpeg available
        try:
            mp4_fig, mp4_ax = plt.subplots(figsize=(11, 7))
            ani_mp4 = animation.FuncAnimation(
                mp4_fig, lambda i: [mp4_ax.clear(),
                    plot_flood_frame(dem, frames_pred[i], node_itm,
                                     pred_stage[frame_idxs[i]], node_refs,
                                     timestamps[frame_idxs[i]], affine,
                                     ax=mp4_ax,
                                     title_suffix=f"  [{model_label}]",
                                     rivers_gdf=rivers_gdf,
                                     lakes_gdf=lakes_gdf)],
                frames=len(frames_pred), interval=1000//fps, repeat=False,
            )
            mp4_path = ev_dir / "flood_animation.mp4"
            ani_mp4.save(str(mp4_path),
                         writer=animation.FFMpegWriter(fps=fps, bitrate=1800),
                         dpi=120)
            plt.close(mp4_fig)
            print(f"  Saved: {mp4_path.name}")
        except Exception as exc:
            print(f'Error in creating animation video: {exc}')
            pass   # ffmpeg not available — GIF is sufficient

    # ── Overall peak map across all events ───────────────────────────
    print("\n── Peak inundation across all events ──")
    peak_t    = int(np.nanargmax(np.nanmax(true_stage, axis=1)))
    inn_peak  = stage_to_inundation(true_stage[peak_t], hand, masks, bankfull)
    peak_fig  = plot_flood_frame(
        dem, inn_peak, node_itm, true_stage[peak_t], node_refs,
        timestamps[peak_t], affine,
        title_suffix=" — Maximum observed inundation",
        rivers_gdf=rivers_gdf, lakes_gdf=lakes_gdf,
    )
    overall_path = out_dir / "maximum_inundation_observed.png"
    peak_fig.savefig(overall_path, dpi=150, bbox_inches="tight")
    plt.close(peak_fig)
    print(f"Saved: {overall_path}")

    print(f"\n✓ All outputs written to: {out_dir}")
    print("  Contents:")
    for f in sorted(out_dir.rglob("*.png")) + sorted(out_dir.rglob("*.gif")):
        print(f"    {f.relative_to(out_dir)}")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

# ── Model display names ───────────────────────────────────────────────────
MODEL_LABELS = {
    "gru":              "PerNodeGRU",
    "lstm":             "PerNodeLSTM",
    "st_gnn_static":    "STGNNFlood (static)",
    "st_gnn_sar":       "STGNNFlood+SAR",
    "st_gnn_dyn_edge":  "STGNNFlood+DynEdge",
    "st_gnn_hand_edge": "STGNNFlood+HAND",
}


def run_one_model(ckpt_dir, dem_path, out_root, n_events, fps, max_frames, rivers_gdf=None, lakes_gdf=None):
    model_name  = ckpt_dir.name
    model_label = MODEL_LABELS.get(model_name, model_name)
    out_dir     = out_root / model_name
    has_mean    = any(ckpt_dir.glob("test_predictions_*steps_mean.npy"))
    has_single  = (ckpt_dir / "test_predictions.npy").exists()
    if not has_mean and not has_single:
        print("  SKIPPED " + model_name + ": no predictions found.")
        print("  Run: python src/run_inference.py --model " + model_name)
        return
    print("\n" + "="*55)
    print("  Model: " + model_label)
    print("="*55)
    try:
        generate_all(
            ckpt_dir=ckpt_dir, proc_dir=PROC_DIR, graph_dir=GRAPH_DIR,
            dem_path=dem_path, out_dir=out_dir, n_events=n_events,
            fps=fps, max_frames=max_frames, model_label=model_label,
            rivers_gdf=rivers_gdf, lakes_gdf=lakes_gdf,
        )
    except Exception as exc:
        print("  ERROR:", exc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate HAND-based flood maps from ST-GNN predictions"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--all-models", action="store_true",
                      help="All models in checkpoints/ (default when no flag given)")
    mode.add_argument("--model", type=str, default=None,
                      help="Single model name, e.g. st_gnn_hand_edge")
    mode.add_argument("--ckpt",  type=Path, default=None,
                      help="Explicit checkpoint directory path")
    p.add_argument("--dem",    type=Path, default=DEM_PATH)
    p.add_argument("--out",    type=Path, default=FIG_DIR)
    p.add_argument("--events", type=int,  default=2)
    p.add_argument("--fps",    type=int,  default=4)
    p.add_argument("--frames", type=int,  default=96)
    args = p.parse_args()

    if not args.dem.exists():
        print("DEM not found:", args.dem)
        print("Run: python src/data/precompute_hand_edges_v1.py")
        sys.exit(1)

    print("\n── Loading shapefiles ──")
    rivers_gdf, lakes_gdf = load_shapefiles(RIVER_SHAPEFILE, LAKES_SHAPEFILE)

    kw = dict(dem_path=args.dem, out_root=args.out,
              n_events=args.events, fps=args.fps, max_frames=args.frames,
              rivers_gdf=rivers_gdf, lakes_gdf=lakes_gdf)

    if args.ckpt is not None:
        run_one_model(args.ckpt, **kw)
    elif args.model is not None:
        run_one_model(CKPT_ROOT / args.model, **kw)
    else:
        # Default: all models
        dirs = [d for d in sorted(CKPT_ROOT.iterdir())
                if d.is_dir() and not d.name.startswith(".")]
        print("Generating flood maps for", len(dirs), "models ...")
        for md in dirs:
            run_one_model(md, **kw)
        print("\nDone. Outputs written to:", args.out)
