"""
precompute_hand_edges.py  –  Precompute HAND candidate edges for Phase 2
=========================================================================
One-time script. Reads the Lee DEM, computes HAND following the
algorithm of Nobre et al. (2011) exactly, then for every pair of gauge
nodes within 5 km overland distance that are not already connected by a
river-network edge, finds the minimum HAND value along the straight-line
corridor between them.  The minimum corridor HAND represents the
topographic saddle — the stage height at which inundation would connect
the two sub-catchments.

HAND algorithm (Nobre et al. 2011, Journal of Hydrology 404, 13-29)
--------------------------------------------------------------------
1. Obtain a Digital Elevation Model.
2. Fill sinks/depressions (Priority-Flood; Barnes et al. 2014).
3. Compute D8 flow direction from the pit-filled DEM
   (O'Callaghan & Mark 1984; Jenson & Domingue 1988).
4. Compute D8 flow accumulation; threshold to define the drainage
   network.
5. For each terrain cell, trace the D8 flow path downslope until
   the nearest stream cell is reached.
   HAND = elevation(cell) − elevation(stream cell at end of D8 path).

Output
------
    dataset/graph/hand_edges.npz  containing arrays:
        src              int32  [E_hand]
        dst              int32  [E_hand]
        hand_threshold   float32 [E_hand]  minimum HAND along corridor (m)
        overland_dist_km float32 [E_hand]  Euclidean distance between nodes
        z_saddle_m       float32 [E_hand]  absolute elevation (m OD) at the
                                            minimum-HAND pixel (the
                                            topographic saddle between the
                                            two sub-catchments).

Usage
-----
    python src/precompute_hand_edges.py
    python src/precompute_hand_edges.py --dem dataset/dem/COP-DEM-30m.tif
                                         --nodes dataset/graph/nodes.csv
                                         --out dataset/graph/hand_edges.npz
                                         --max-dist 5.0
                                         --hand-min 0.5

Distance threshold (5 km)
--------------------------
Consistent with reach hydraulic discretisation recommendations for
HAND-based inundation mapping (Zheng et al. 2018, Water Resources
Research) and the ~4-5 km Bride–Lee to Shournagh–Lee interfluve
distance in the Lee catchment.

HAND threshold floor (0.5 m)
-----------------------------
Pairs with minimum corridor HAND < 0.5 m are excluded because a near-
zero interfluve HAND indicates the two nodes share the same drainage
basin and are already connected through the river network.

References
----------
Nobre, A.D., et al., 2011. Height Above the Nearest Drainage.
    Journal of Hydrology 404, 13-29.
    https://doi.org/10.1016/j.jhydrol.2011.03.051

Barnes, R., Lehman, C., Mulla, D., 2014. Priority-Flood: An optimal
    depression-filling and watershed-labeling algorithm for digital
    elevation models. Computers & Geosciences 62, 117-127.
    https://doi.org/10.1016/j.cageo.2013.04.024

O'Callaghan, J.F., Mark, D.M., 1984. The extraction of drainage
    networks from digital elevation data. Computer Vision, Graphics,
    and Image Processing 28, 323-344.
    https://doi.org/10.1016/S0734-189X(84)80011-0

Jenson, S.K., Domingue, J.O., 1988. Extracting topographic structure
    from digital elevation data for GIS analysis. Photogrammetric
    Engineering and Remote Sensing 54, 1593-1600.

Zheng, X., et al., 2018. GeoFlood: Large-scale flood inundation mapping
    based on high-resolution terrain analysis. Water Resources Research
    54, 10013-10033. https://doi.org/10.1029/2018WR023457

European Space Agency, 2022. Copernicus Global Digital Elevation Model
    (GLO-30). https://doi.org/10.5069/G9028PQB
"""

from __future__ import annotations

import argparse
import heapq
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
SAMPLE_STEP  = 60     # sample points along corridor centreline
ACC_THRESH   = 500    # flow accumulation threshold for stream definition
                      # 500 cells × (30m)² = 0.45 km² drainage area


# ══════════════════════════════════════════════════════════════════════
#  Step 2 — Pit filling (Barnes et al. 2014, Priority-Flood)
# ══════════════════════════════════════════════════════════════════════

def fill_pits_priority_flood(
    dem_data: np.ndarray,
    nan_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill all sinks and depressions in the DEM using the Priority-Flood
    algorithm (Barnes et al. 2014, Computers & Geosciences 62, 117-127).

    Priority-Flood guarantees that the output surface is hydrologically
    conditioned: every cell has a downslope path to the DEM boundary,
    eliminating spurious pits that would otherwise produce undefined D8
    flow directions and break the accumulation chain.

    The algorithm initialises a min-heap with all DEM border cells and
    processes cells in order of ascending elevation.  Each unvisited
    neighbour is assigned max(its own elevation, current cell elevation),
    ensuring that inward-draining depressions are raised to the lowest
    outlet elevation that allows drainage.

    Parameters
    ----------
    dem_data : [H, W] float64  raw DEM elevations (NaN for nodata)
    nan_mask : [H, W] bool     True where cell is nodata

    Returns
    -------
    filled : [H, W] float64  pit-filled DEM (NaN preserved at nodata)
    """
    H, W    = dem_data.shape
    filled  = dem_data.copy().astype(np.float64)
    visited = np.zeros((H, W), dtype=bool)
    heap    = []

    # Seed the heap with all valid border cells
    for r in range(H):
        for c in [0, W - 1]:
            if not nan_mask[r, c]:
                heapq.heappush(heap, (filled[r, c], r, c))
                visited[r, c] = True
    for c in range(1, W - 1):
        for r in [0, H - 1]:
            if not nan_mask[r, c] and not visited[r, c]:
                heapq.heappush(heap, (filled[r, c], r, c))
                visited[r, c] = True

    neighbours = [(-1,-1),(-1, 0),(-1, 1),
                  ( 0,-1),         ( 0, 1),
                  ( 1,-1),( 1, 0),( 1, 1)]

    while heap:
        elev, r, c = heapq.heappop(heap)
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W
                    and not visited[nr, nc]
                    and not nan_mask[nr, nc]):
                visited[nr, nc] = True
                # Raise neighbour to at least the current cell's elevation
                filled[nr, nc]  = max(filled[nr, nc], elev)
                heapq.heappush(heap, (filled[nr, nc], nr, nc))

    n_raised = int(((filled > dem_data) & ~nan_mask).sum())
    print(f"  Priority-Flood: {n_raised:,} cells raised to fill depressions")
    return filled


# ══════════════════════════════════════════════════════════════════════
#  Steps 3 & 4 — D8 flow direction and flow accumulation
# ══════════════════════════════════════════════════════════════════════

def compute_d8_and_accumulation(
    filled_dem: np.ndarray,
    nan_mask:   np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute D8 flow direction and D8 flow accumulation from the
    pit-filled DEM, following O'Callaghan & Mark (1984) and
    Jenson & Domingue (1988).

    D8 assigns each cell to the steepest downslope neighbour among its
    eight cardinal and diagonal neighbours.  Diagonal slopes are
    normalised by sqrt(2) to account for the longer diagonal path
    length (O'Callaghan & Mark 1984).

    Flow accumulation is computed by processing cells in descending
    elevation order and propagating the accumulation count to the single
    D8 downstream neighbour (Jenson & Domingue 1988).

    Parameters
    ----------
    filled_dem : [H, W] float64  pit-filled DEM
    nan_mask   : [H, W] bool     True where cell is nodata

    Returns
    -------
    dr       : [H, W] int8   row offset to D8 downstream neighbour
    dc       : [H, W] int8   col offset to D8 downstream neighbour
    acc      : [H, W] float32 flow accumulation (upstream cell count)
    """
    H, W = filled_dem.shape

    # ── D8 flow direction ─────────────────────────────────────────────
    sentinel = float(np.nanmax(filled_dem)) + 1e6
    dem_work = filled_dem.copy()
    dem_work[nan_mask] = sentinel

    pad = np.pad(dem_work, 1, mode="constant", constant_values=sentinel)
    dr  = np.zeros((H, W), dtype=np.int8)
    dc  = np.zeros((H, W), dtype=np.int8)
    max_slope = np.full((H, W), -np.inf)

    for di, dj, dist in [(-1,-1,1.4142),(-1, 0,1.0),(-1, 1,1.4142),
                          ( 0,-1,1.0),              ( 0, 1,1.0),
                          ( 1,-1,1.4142),( 1, 0,1.0),( 1, 1,1.4142)]:
        ri   = 1 + di
        ci   = 1 + dj
        neigh = pad[ri:ri+H, ci:ci+W]
        slope = (dem_work - neigh) / dist
        update = (slope > max_slope) & ~nan_mask
        dr[update]        = di
        dc[update]        = dj
        max_slope[update] = slope[update]

    # ── Flow accumulation ─────────────────────────────────────────────
    acc = np.ones((H, W), dtype=np.float32)
    acc[nan_mask] = 0.0

    flat_order = np.argsort(dem_work.ravel())[::-1]   # high → low
    for idx in flat_order:
        r, c = divmod(int(idx), W)
        if nan_mask[r, c] or (dr[r, c] == 0 and dc[r, c] == 0):
            continue
        nr, nc = r + int(dr[r, c]), c + int(dc[r, c])
        if 0 <= nr < H and 0 <= nc < W and not nan_mask[nr, nc]:
            acc[nr, nc] += acc[r, c]

    print(f"  Accumulation range: [1, {acc.max():.0f}] cells")
    return dr, dc, acc


# ══════════════════════════════════════════════════════════════════════
#  Step 5 — HAND via D8 flow-path tracing (Nobre et al. 2011)
# ══════════════════════════════════════════════════════════════════════

def compute_hand_d8_path(
    dem:         np.ndarray,    # [H, W] float64  raw (unfilled) elevations
    dr:          np.ndarray,    # [H, W] int8     D8 row offset
    dc:          np.ndarray,    # [H, W] int8     D8 col offset
    stream_mask: np.ndarray,    # [H, W] uint8    1=stream
    nan_mask:    np.ndarray,    # [H, W] bool
    filled_dem:  np.ndarray,    # [H, W] float64  pit-filled (for sort order)
) -> np.ndarray:
    """
    Compute HAND following the definition of Nobre et al. (2011):
    for each terrain cell, follow the D8 flow path downslope until
    the nearest stream cell is reached; HAND equals the vertical
    distance between the terrain cell's own elevation and the
    elevation of that stream cell.

        HAND(cell) = z(cell) − z(stream cell at end of D8 path)

    Implementation: processing cells in ascending elevation order
    (lowest first) guarantees that the downstream neighbour of every
    non-stream cell has already received its stream-elevation
    assignment before the current cell is processed, because D8 flow
    is always directed to a lower (or equal) neighbour.  Stream cells
    are seeded with their own elevation, and the stream elevation
    propagates upslope through the assignment
        stream_elev[cell] = stream_elev[downstream neighbour].

    This is equivalent to tracing every D8 flow path individually but
    is O(H × W × log(H × W)) rather than O(H × W × path_length).

    Parameters
    ----------
    dem         : raw DEM — used for HAND subtraction
    dr / dc     : D8 flow direction offsets computed from pit-filled DEM
    stream_mask : 1 at stream cells, 0 elsewhere
    nan_mask    : True at nodata cells
    filled_dem  : pit-filled DEM — used only for sort order

    Returns
    -------
    hand : [H, W] float32  HAND values (m); NaN at nodata and off-network
    """
    H, W = dem.shape

    # For every cell, record the elevation of the stream cell
    # it ultimately drains to via the D8 path.
    stream_elev = np.full((H, W), np.nan, dtype=np.float64)

    # Seed: stream cells drain to themselves
    sr, sc = np.where((stream_mask == 1) & ~nan_mask)
    stream_elev[sr, sc] = dem[sr, sc]

    # Process in ascending elevation order (lowest → highest).
    # Because D8 always points downhill, the downstream neighbour
    # of any non-stream cell is always lower and therefore already
    # processed when we reach the current cell.
    sort_key = filled_dem.copy()
    sort_key[nan_mask] = np.inf
    ascending_order = np.argsort(sort_key.ravel())

    for idx in ascending_order:
        r, c = divmod(int(idx), W)
        if nan_mask[r, c] or stream_mask[r, c]:
            continue                               # nodata or already seeded
        if dr[r, c] == 0 and dc[r, c] == 0:
            continue                               # no valid D8 direction
        nr, nc = r + int(dr[r, c]), c + int(dc[r, c])
        if 0 <= nr < H and 0 <= nc < W and not nan_mask[nr, nc]:
            stream_elev[r, c] = stream_elev[nr, nc]

    # HAND = raw elevation − stream elevation along D8 path
    hand = (dem - stream_elev).astype(np.float32)
    hand = np.clip(hand, 0.0, None)               # negative → same stream level
    hand[nan_mask]                        = np.nan
    hand[np.isnan(stream_elev) & ~nan_mask] = np.nan   # off-network cells

    n_valid = int((~np.isnan(hand)).sum())
    valid   = hand[~np.isnan(hand)]
    print(f"  HAND (D8-path): {n_valid:,} valid cells  "
          f"range=[{valid.min():.2f}, {valid.max():.2f}] m")
    return hand


# ══════════════════════════════════════════════════════════════════════
#  Top-level HAND computation (Steps 1–5)
# ══════════════════════════════════════════════════════════════════════

def reproject_dem_to_itm(dem_path: Path, out_path: Path) -> Path:
    """Reproject DEM to ITM (EPSG:2157) at 30 m if not already projected."""
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS as _CRS

    if out_path.exists():
        print(f"  Using cached ITM DEM: {out_path.name}")
        return out_path

    with rasterio.open(dem_path) as src:
        if src.crs and src.crs.to_epsg() == 2157:
            print("  DEM already in ITM.")
            return dem_path

        print(f"  Reprojecting {src.crs} → EPSG:2157 at 30 m …")
        dst_crs = _CRS.from_epsg(2157)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=30.0)
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

    print(f"  Reprojected → {out_path.name}  ({height} × {width} px at 30 m)")
    return out_path


def compute_hand(
    dem_path:   Path,
    fdir_path:  Path | None = None,
    acc_thresh: int = ACC_THRESH,
) -> tuple[np.ndarray, object, str, np.ndarray]:
    """
    Execute the complete Nobre et al. (2011) HAND pipeline.

    Returns (hand, affine, crs, dem_data):
        hand     [H, W] float32  HAND values (m, D8 flow-path definition)
        affine   rasterio Affine  transform for pixel ↔ world conversion
        crs      str             coordinate reference system string
        dem_data [H, W] float64  raw DEM elevations (for z_saddle lookup)
    """
    import rasterio

    # ── Load DEM ─────────────────────────────────────────────────────
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
    print(f"  DEM shape: {H} × {W}  CRS: {crs}")
    print(f"  Elevation range: [{np.nanmin(dem_data):.1f}, "
          f"{np.nanmax(dem_data):.1f}] m")

    # ── Step 2: Priority-Flood pit filling ────────────────────────────
    print("  Step 2: Priority-Flood depression filling …")
    filled_dem = fill_pits_priority_flood(dem_data, nan_mask)

    # ── Steps 3 & 4: D8 direction + accumulation ─────────────────────
    print("  Steps 3–4: D8 flow direction and accumulation …")
    dr, dc, acc = compute_d8_and_accumulation(filled_dem, nan_mask)

    # Optionally save D8 direction arrays for downstream use
    if fdir_path is not None:
        fdir_path = Path(fdir_path)
        fdir_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fdir_path,
                            dr=dr, dc=dc,
                            nan_mask=nan_mask.astype(np.bool_))
        size_mb = fdir_path.stat().st_size / 1024**2
        print(f"  fdir.npz saved → {fdir_path.name}  {size_mb:.1f} MB")

    # Define stream network by accumulation threshold
    eff_thresh = acc_thresh if acc.max() >= acc_thresh else float(acc.max() * 0.9)
    stream_mask = ((acc >= eff_thresh) & ~nan_mask).astype(np.uint8)
    n_stream = int(stream_mask.sum())
    n_valid  = int((~nan_mask).sum())
    print(f"  Stream cells: {n_stream:,} / {n_valid:,} "
          f"({n_stream/max(n_valid,1)*100:.2f}%)  "
          f"threshold = {eff_thresh:.0f} cells "
          f"(≈ {eff_thresh * 30**2 / 1e6:.3f} km²)")
    if n_stream == 0:
        raise ValueError("No stream cells. Check DEM coverage and threshold.")

    # ── Step 5: HAND via D8 flow-path tracing (Nobre et al. 2011) ────
    print("  Step 5: HAND via D8 flow-path tracing (Nobre et al. 2011) …")
    hand = compute_hand_d8_path(
        dem_data, dr, dc, stream_mask, nan_mask, filled_dem)

    return hand, affine, crs, dem_data


# ══════════════════════════════════════════════════════════════════════
#  Spatial utilities — node coordinates and corridor sampling
# ══════════════════════════════════════════════════════════════════════

def nodes_to_itm(nodes_csv: Path) -> tuple[np.ndarray, np.ndarray, list]:
    """Return ITM eastings, northings, and refs for all gauge nodes."""
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
            "nodes.csv must have 'easting_itm'/'northing_itm' or 'lat'/'lon'.")
    refs = (df["ref"].astype(str).tolist()
            if "ref" in df.columns else [str(i) for i in range(len(df))])
    return eastings.astype(np.float32), northings.astype(np.float32), refs


def world_to_pixel(x: float, y: float, affine) -> tuple[int, int]:
    """Convert ITM world coordinates to DEM row/col."""
    col = (x - affine.c) / affine.a
    row = (y - affine.f) / affine.e
    return int(round(row)), int(round(col))


def sample_corridor_hand(
    hand:    np.ndarray,
    dem:     np.ndarray,
    affine,
    e1: float, n1: float,
    e2: float, n2: float,
    n_samples: int = SAMPLE_STEP,
) -> tuple[float, float]:
    """
    Find the minimum HAND value along the straight-line corridor
    between two ITM node coordinates.

    Samples n_samples evenly-spaced points along the centreline,
    restricted to the central 70% of the line (t ∈ [0.15, 0.85]) to
    exclude the gauge locations themselves (where HAND ≈ 0, since
    gauges sit on or adjacent to the stream network).

    The minimum HAND along the corridor approximates the topographic
    saddle between the two sub-catchments — the stage height at which
    inundation from either side would connect the two nodes' floodplains.

    Returns
    -------
    min_hand : minimum HAND value along corridor (m); NaN if all NaN
    z_saddle : absolute DEM elevation (m OD) at the minimum-HAND pixel;
               used by the activation gate in st_gnn_hand_edge.py to
               compare against reconstructed water-surface elevation.
    """
    H, W    = hand.shape
    t_vals  = np.linspace(0.15, 0.85, n_samples)
    min_hand = np.inf
    z_saddle = np.nan

    for t in t_vals:
        e = e1 + t * (e2 - e1)
        n = n1 + t * (n2 - n1)
        row, col = world_to_pixel(e, n, affine)
        if 0 <= row < H and 0 <= col < W:
            v = hand[row, col]
            if not np.isnan(v) and v < min_hand:
                min_hand = float(v)
                dv = dem[row, col]
                z_saddle = float(dv) if not np.isnan(dv) else np.nan

    if not np.isfinite(min_hand):
        return np.nan, np.nan
    return min_hand, z_saddle


def find_candidate_pairs(
    eastings:          np.ndarray,
    northings:         np.ndarray,
    static_edge_index: np.ndarray | None,
    max_dist_km:       float,
) -> list[tuple[int, int, float]]:
    """Return all gauge pairs within max_dist_km not in static edge set."""
    N        = len(eastings)
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
            de  = eastings[i]  - eastings[j]
            dn  = northings[i] - northings[j]
            dist_km = np.sqrt(de**2 + dn**2) / 1000.0
            if dist_km <= max_dist_km:
                candidates.append((i, j, dist_km))
    return candidates


def load_static_edges(graph_dir: Path) -> np.ndarray | None:
    edges_csv = graph_dir / "edges.csv"
    if not edges_csv.exists():
        print(f"  edges.csv not found — no pairs excluded")
        return None
    df = pd.read_csv(edges_csv)
    if "src" in df.columns and "dst" in df.columns:
        return np.stack([df["src"].values, df["dst"].values])
    return None


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def run(
    dem_path:    Path,
    nodes_path:  Path,
    out_path:    Path,
    max_dist_km: float,
    hand_min_m:  float,
) -> None:
    print("=" * 62)
    print("HAND edge precomputation — Lee catchment")
    print(f"  Algorithm : Nobre et al. (2011), J. Hydrology 404, 13-29")
    print(f"  Pit fill  : Priority-Flood (Barnes et al. 2014)")
    print(f"  D8        : O'Callaghan & Mark (1984)")
    print(f"  DEM       : {dem_path}")
    print(f"  Nodes     : {nodes_path}")
    print(f"  Max dist  : {max_dist_km} km")
    print(f"  HAND floor: {hand_min_m} m")
    print("=" * 62)

    # ── 1–5: Full HAND pipeline ───────────────────────────────────────
    fdir_out = out_path.parent / "fdir.npz"
    hand, affine, crs, dem_data = compute_hand(dem_path, fdir_path=fdir_out)

    # ── Node ITM coordinates ──────────────────────────────────────────
    eastings, northings, refs = nodes_to_itm(nodes_path)
    N = len(eastings)
    print(f"\nNodes loaded: {N}")
    for i, (e, n, r) in enumerate(zip(eastings[:5], northings[:5], refs[:5])):
        print(f"  {i:2d} ref={r}  E={e:.0f}  N={n:.0f}")
    if N > 5:
        print(f"  … (+{N-5} more)")

    # ── Candidate pairs ───────────────────────────────────────────────
    static_ei  = load_static_edges(nodes_path.parent)
    candidates = find_candidate_pairs(eastings, northings, static_ei, max_dist_km)
    print(f"\nCandidate pairs within {max_dist_km} km: {len(candidates)}")

    # ── Sample corridor HAND ──────────────────────────────────────────
    srcs, dsts, thresholds, dists, saddles = [], [], [], [], []
    skipped_low     = 0
    skipped_nodata  = 0

    for k, (i, j, dist_km) in enumerate(candidates):
        if (k + 1) % 10 == 0 or k == len(candidates) - 1:
            print(f"  Pair {k+1}/{len(candidates)} …", end="\r")

        min_hand, z_saddle = sample_corridor_hand(
            hand, dem_data, affine,
            eastings[i],  northings[i],
            eastings[j],  northings[j],
            n_samples=SAMPLE_STEP,
        )

        if np.isnan(min_hand):
            continue                       # corridor entirely off-raster
        if min_hand < hand_min_m:
            skipped_low += 1
            continue                       # same drainage basin
        if np.isnan(z_saddle):
            skipped_nodata += 1
            continue                       # DEM nodata at saddle pixel

        # Add both directions (inundation can spread either way)
        for src, dst in [(i, j), (j, i)]:
            srcs.append(src); dsts.append(dst)
            thresholds.append(min_hand)
            dists.append(dist_km)
            saddles.append(z_saddle)

    print()
    n_pairs = len(srcs) // 2
    print(f"\nHAND edges accepted : {n_pairs} pairs → {len(srcs)} directed")
    print(f"Skipped (HAND < {hand_min_m} m): {skipped_low} pairs "
          f"(same basin)")
    if skipped_nodata:
        print(f"Skipped (DEM nodata at saddle): {skipped_nodata} pairs")

    if not srcs:
        print("WARNING: no HAND edges. Check DEM coverage and node coords.")
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        src              = np.array(srcs,       dtype=np.int32),
        dst              = np.array(dsts,       dtype=np.int32),
        hand_threshold   = np.array(thresholds, dtype=np.float32),
        overland_dist_km = np.array(dists,      dtype=np.float32),
        z_saddle_m       = np.array(saddles,    dtype=np.float32),
    )
    print(f"\nSaved: {out_path}")
    print(f"  hand_threshold : [{min(thresholds):.2f}, {max(thresholds):.2f}] m")
    print(f"  overland_dist  : [{min(dists):.2f}, {max(dists):.2f}] km")
    print(f"  z_saddle_m     : [{min(saddles):.2f}, {max(saddles):.2f}] m OD")

    print("\nAccepted HAND edge pairs:")
    print(f"  {'src':12s} {'dst':12s} {'dist_km':>8s} "
          f"{'hand_thr_m':>12s} {'z_saddle_m':>12s}")
    seen = set()
    for s, d, t, dist, z in zip(srcs, dsts, thresholds, dists, saddles):
        key = (min(s, d), max(s, d))
        if key not in seen:
            seen.add(key)
            print(f"  {refs[s]:12s} {refs[d]:12s} {dist:8.2f} "
                  f"{t:12.3f} {z:12.2f}")
    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precompute HAND edges following Nobre et al. (2011)")
    p.add_argument("--dem",      type=Path, default=DEM_PATH)
    p.add_argument("--nodes",    type=Path, default=NODES_PATH)
    p.add_argument("--out",      type=Path, default=OUT_PATH)
    p.add_argument("--max-dist", type=float, default=MAX_DIST_KM)
    p.add_argument("--hand-min", type=float, default=HAND_MIN_M)
    args = p.parse_args()

    if not args.dem.exists():
        print(f"ERROR: DEM not found: {args.dem}"); sys.exit(1)
    if not args.nodes.exists():
        print(f"ERROR: nodes.csv not found: {args.nodes}"); sys.exit(1)

    run(args.dem, args.nodes, args.out, args.max_dist, args.hand_min)
