"""
generate_web_map.py  –  Interactive flood inundation maps on satellite imagery
==============================================================================
Converts ST-GNN inundation predictions to interactive HTML maps using Folium.
The output is a standalone HTML file that opens in any browser — no GIS
software, no API keys, no internet connection required at viewing time
(tiles are fetched live when the map is opened).

Map layers (toggleable via the layer control panel):
  • Google Maps satellite (default base layer)
  • OpenStreetMap (alternative base layer)
  • Flood inundation extent (semi-transparent cyan polygons)
  • River network (from shapefile, optional)
  • Lakes / reservoirs (from shapefile, optional)
  • OPW gauge stations (clickable markers with stage info)

Animation:
  When --animate is set the script generates a TimestampedGeoJson layer
  so you can step through the flood evolution using a time slider in the
  browser. Each frame is one 15-minute model prediction step.

Output
------
  figures/web_maps/{model_name}/
      peak_inundation.html          — static peak flood map
      flood_animation.html          — animated flood evolution (if --animate)
      inundation_peak.geojson       — peak inundation polygon (for GIS import)
      gauges.geojson                — gauge stations with stage values

Usage
-----
    # All models (uses ensemble mean predictions from run_inference.py)
    python src/generate_web_map.py

    # Single model
    python src/generate_web_map.py --model st_gnn_hand_edge

    # With shapefiles and animation
    python src/generate_web_map.py \\
        --rivers dataset/shapefiles/river_network.shp \\
        --lakes  dataset/shapefiles/lakes.shp \\
        --animate

    # Only the 4-step horizon predictions
    python src/generate_web_map.py --horizon 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
CKPT_ROOT = BASE_DIR / "checkpoints"
DEM_PATH  = BASE_DIR / "dataset/dem/COP-DEM-30m_itm.tif"
OUT_ROOT  = BASE_DIR / "figures/web_maps"

MODEL_LABELS = {
    "gru":              "PerNodeGRU",
    "lstm":             "PerNodeLSTM",
    "st_gnn_static":    "STGNNFlood (static)",
    "st_gnn_sar":       "STGNNFlood+SAR",
    "st_gnn_dyn_edge":  "STGNNFlood+DynEdge",
    "st_gnn_hand_edge": "STGNNFlood+HAND",
}

# Lee catchment approximate centre (WGS84) for map initial view
LEE_CENTRE = [51.895, -8.750]
LEE_ZOOM   = 11


# ══════════════════════════════════════════════════════════════════════
#  Step 1: Raster → GeoJSON polygon conversion
# ══════════════════════════════════════════════════════════════════════

def inundation_to_geojson(
    inundation: np.ndarray,   # [H, W] boolean, ITM pixel coords
    affine,                   # rasterio Affine (ITM)
    dem: np.ndarray,          # [H, W] elevation (m) — used for depth estimate
    simplify_tolerance: float = 60.0,   # metres — reduces polygon vertex count
) -> dict:
    """
    Convert a boolean inundation raster to a GeoJSON FeatureCollection
    in WGS84 (EPSG:4326) for web mapping.

    Uses rasterio.features.shapes() to trace polygon boundaries directly
    from the raster, then reprojects to WGS84 with pyproj.

    Each polygon feature carries:
        area_km2        total inundated area for the whole event
        depth_mean_m    mean estimated depth (stage above HAND) where available
    """
    import rasterio.features
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    import shapely
    from pyproj import Transformer

    # Extract polygon shapes from the boolean raster
    mask = inundation.astype(np.uint8)
    shapes = list(rasterio.features.shapes(mask, mask=mask, transform=affine))
    if not shapes:
        return {"type": "FeatureCollection", "features": []}

    # Merge all polygons into one (removes internal holes from raster cells)
    polys = [shape(geom) for geom, val in shapes if val == 1]
    if not polys:
        return {"type": "FeatureCollection", "features": []}

    merged = unary_union(polys)

    # Simplify to reduce file size (tolerance in ITM metres)
    if simplify_tolerance > 0:
        merged = merged.simplify(simplify_tolerance, preserve_topology=True)

    # Reproject from ITM (EPSG:2157) → WGS84 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)

    def reproject_coords(coords):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        lons, lats = transformer.transform(xs, ys)
        return list(zip(lons, lats))

    def reproject_geom(geom):
        """Recursively reproject a Shapely geometry to WGS84."""
        from shapely.geometry import Polygon, MultiPolygon
        if geom.geom_type == "Polygon":
            exterior = reproject_coords(list(geom.exterior.coords))
            interiors = [reproject_coords(list(r.coords)) for r in geom.interiors]
            return Polygon(exterior, interiors)
        elif geom.geom_type == "MultiPolygon":
            return MultiPolygon([reproject_geom(p) for p in geom.geoms])
        return geom

    merged_wgs84 = reproject_geom(merged)

    pixel_m  = abs(affine.a)
    area_km2 = inundation.sum() * pixel_m**2 / 1e6

    feature = {
        "type": "Feature",
        "geometry": mapping(merged_wgs84),
        "properties": {
            "area_km2": round(area_km2, 3),
            "layer": "inundation",
        },
    }
    return {"type": "FeatureCollection", "features": [feature]}


# ══════════════════════════════════════════════════════════════════════
#  Step 2: Gauge stations → GeoJSON
# ══════════════════════════════════════════════════════════════════════

def gauges_to_geojson(
    node_itm:   np.ndarray,   # [N, 2] ITM easting/northing
    node_refs:  list,
    node_stage: np.ndarray,   # [N] predicted stage at this timestep
    node_names: list | None = None,
) -> dict:
    """Convert gauge node positions and stage readings to GeoJSON points."""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)

    features = []
    for i, (e, n) in enumerate(node_itm):
        lon, lat = t.transform(float(e), float(n))
        stage    = float(node_stage[i]) if not np.isnan(node_stage[i]) else 0.0
        status   = ("high" if stage > 0.3 else
                    "moderate" if stage > 0.1 else "normal")
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "ref":    node_refs[i],
                "name":   node_names[i] if node_names else node_refs[i],
                "stage_m": round(stage, 3),
                "status": status,
            },
        })
    return {"type": "FeatureCollection", "features": features}


# ══════════════════════════════════════════════════════════════════════
#  Step 3: Load predictions and inundation pipeline
# ══════════════════════════════════════════════════════════════════════

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

def load_inundation_data(ckpt_dir: Path) -> tuple | None:
    """
    Load ensemble mean predictions, DEM, HAND, and gauge metadata.
    Returns (pred_stage, true_stage, timestamps, node_itm, node_refs,
             node_names, dem, hand, affine, masks) or None if missing.
    """
    # Find prediction file
    mean_files  = sorted(ckpt_dir.glob("test_predictions_*steps_mean.npy"))
    single_file = ckpt_dir / "test_predictions.npy"
    if mean_files:
        pred_stage = np.load(mean_files[0])
        print(f"  Predictions: {mean_files[0].name}  {pred_stage.shape}")
    elif single_file.exists():
        pred_stage = np.load(single_file)
        print(f"  Predictions: {single_file.name}  {pred_stage.shape}")
    else:
        print(f"  SKIPPED: no predictions in {ckpt_dir.name}")
        return None

    # Load y (ground truth) and timestamps
    y      = np.load(PROC_DIR / "y.npy", mmap_mode="r")
    T      = y.shape[0]
    T_out  = 4  # default; fine for web map
    ts_start = int(T * 0.85)
    ts_end   = T - T_out
    true_stage = y[ts_start:ts_end]

    all_ts    = pd.to_datetime(pd.read_csv(PROC_DIR / "timestamps.csv")["timestamp"])
    timestamps = pd.DatetimeIndex(all_ts.iloc[ts_start:ts_end].values)

    # Load DEM and HAND
    import rasterio
    from scipy.ndimage import distance_transform_edt

    itm_dem = DEM_PATH
    if not itm_dem.exists():
        print(f"  WARNING: ITM DEM not found at {itm_dem}")
        return None

    with rasterio.open(itm_dem) as src:
        dem    = src.read(1).astype(np.float32)
        affine = src.transform
        nodata = src.nodata or -9999.0
    dem[dem == nodata] = np.nan

    # HAND cache
    hand_path = itm_dem.parent / "hand_raster.tif"
    if hand_path.exists():
        with rasterio.open(hand_path) as src:
            hand = src.read(1).astype(np.float32)
    else:
        print("  Computing HAND on-the-fly …")
        nan_mask    = np.isnan(dem)
        stream_mask = _quick_stream_mask(dem, nan_mask)
        dist_px, idx = distance_transform_edt(stream_mask == 0, return_indices=True)
        stream_elev  = dem[idx[0], idx[1]]
        hand = np.clip(dem - stream_elev, 0, None).astype(np.float32)
        hand[nan_mask] = np.nan

    # Node metadata
    nodes_df  = pd.read_csv(GRAPH_DIR / "nodes.csv")
    if "easting_itm" in nodes_df.columns:
        node_itm = nodes_df[["easting_itm", "northing_itm"]].values
    else:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        E, N = t.transform(nodes_df["lon"].values, nodes_df["lat"].values)
        node_itm = np.column_stack([E, N])

    node_refs  = nodes_df["ref"].astype(str).tolist()
    node_names = nodes_df["name"].tolist() if "name" in nodes_df.columns else node_refs

    # D8 upstream catchment masks — watershed boundaries, not Voronoi
    H, W = dem.shape
    masks = build_d8_catchment_masks(node_itm, affine, H, W)

    # Per-node bankfull thresholds
    X_full    = np.load(PROC_DIR / "X.npy", mmap_mode="r")
    train_end = int(X_full.shape[0] * 0.70)
    bankfull, reliable = compute_bankfull_thresholds(
        X_full, train_end, stage_col=0, node_refs=node_refs
    )
    del X_full

    return (pred_stage, true_stage, timestamps,
            node_itm, node_refs, node_names,
            dem, hand, affine, masks, bankfull, reliable)


def _quick_stream_mask(dem, nan_mask):
    """Minimal D8 accumulation for HAND computation."""
    H, W = dem.shape
    sentinel = float(np.nanmax(dem)) + 1e6
    dw = dem.copy(); dw[nan_mask] = sentinel
    pad = np.pad(dw, 1, mode="constant", constant_values=sentinel)
    dr = np.zeros((H,W), dtype=np.int8)
    dc = np.zeros((H,W), dtype=np.int8)
    ms = np.full((H,W), -np.inf)
    for di,dj,d in [(-1,-1,1.414),(-1,0,1.),(-1,1,1.414),(0,-1,1.),(0,1,1.),(1,-1,1.414),(1,0,1.),(1,1,1.414)]:
        ri,ci = 1+di,1+dj
        sl = (dw - pad[ri:ri+H, ci:ci+W]) / d
        up = (sl > ms) & ~nan_mask
        dr[up]=di; dc[up]=dj; ms[up]=sl[up]
    acc = np.ones((H,W), dtype=np.float32); acc[nan_mask]=0.
    for r,c in zip(*(np.argsort(dw.ravel())[::-1].reshape(-1) // W,
                     np.argsort(dw.ravel())[::-1].reshape(-1) %  W)):
        if nan_mask[r,c] or (dr[r,c]==0 and dc[r,c]==0): continue
        nr,nc = r+int(dr[r,c]), c+int(dc[r,c])
        if 0<=nr<H and 0<=nc<W and not nan_mask[nr,nc]: acc[nr,nc]+=acc[r,c]
    return ((acc >= 500) & ~nan_mask).astype(np.uint8)



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

def stage_to_inundation(stage, hand, masks, thresholds=None):
    """Per-node bankfull thresholds. Pass thresholds=None for legacy 0.1 m."""
    H, W = hand.shape
    result = np.zeros((H, W), dtype=bool)
    thr = (np.full(len(stage), 0.1, dtype=np.float32)
           if thresholds is None else np.asarray(thresholds))
    for i, (s, mask, t) in enumerate(zip(stage, masks, thr)):
        if s < t or np.isnan(s): continue
        result |= mask & ~np.isnan(hand) & (hand <= s)
    return result


def find_peak_event(true_stage, timestamps, threshold=0.3, min_duration_h=6):
    """Return (start_idx, end_idx, peak_idx) of most severe flood event."""
    max_stage = np.nanmax(true_stage, axis=1)
    in_flood  = max_stage > threshold
    best = None

    i = 0
    while i < len(in_flood):
        if in_flood[i]:
            j = i
            while j < len(in_flood) and in_flood[j]: j += 1
            if (j - i) * 15 / 60 >= min_duration_h:
                pk = int(i + np.argmax(max_stage[i:j]))
                if best is None or max_stage[pk] > max_stage[best[2]]:
                    best = (i, j, pk)
            i = j
        else:
            i += 1

    if best is None:
        # Fall back to absolute peak
        pk = int(np.nanargmax(max_stage))
        best = (max(0, pk-48), min(len(timestamps), pk+48), pk)
    return best


# ══════════════════════════════════════════════════════════════════════
#  Step 4: Build Folium maps
# ══════════════════════════════════════════════════════════════════════

def make_base_map() -> "folium.Map":
    """Create a Folium map with Google Maps satellite + OSM tile layers."""
    import folium

    m = folium.Map(
        location=LEE_CENTRE,
        zoom_start=LEE_ZOOM,
        tiles=None,           # no default tile — we add our own below
    )

    # Google Maps satellite (no API key needed for this tile endpoint)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Maps Satellite",
        name="Google Satellite",
        overlay=False,
        control=True,
        max_zoom=20,
    ).add_to(m)

    # Google Maps hybrid (satellite + labels)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Maps Hybrid",
        name="Google Hybrid",
        overlay=False,
        control=True,
        max_zoom=20,
    ).add_to(m)

    # OpenStreetMap — useful for checking road/building names
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)

    return m


def add_inundation_layer(
    m,
    geojson: dict,
    layer_name: str = "Flood inundation",
    color: str = "#00D9F2",
    fill_opacity: float = 0.45,
) -> None:
    """Add a flood inundation GeoJSON polygon layer to the map."""
    import folium

    if not geojson["features"]:
        return

    area = geojson["features"][0]["properties"].get("area_km2", 0)

    folium.GeoJson(
        geojson,
        name=layer_name,
        style_function=lambda _: {
            "fillColor":   color,
            "color":       "#FFFFFF",      # white boundary for contrast
            "weight":      1.5,
            "fillOpacity": fill_opacity,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["area_km2"],
            aliases=["Inundated area (km²):"],
            localize=True,
        ),
    ).add_to(m)


def add_gauge_layer(m, gauges_geojson: dict, layer_name: str = "OPW Gauges") -> None:
    """Add gauge stations as circle markers with popups."""
    import folium

    fg = folium.FeatureGroup(name=layer_name)

    STATUS_COLOUR = {"high": "#E83030", "moderate": "#F5A623", "normal": "#4CAF50"}
    STATUS_LABEL  = {"high": "⚠ High", "moderate": "⚡ Moderate", "normal": "✓ Normal"}

    for feat in gauges_geojson["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        props    = feat["properties"]
        stage    = props["stage_m"]
        status   = props["status"]
        colour   = STATUS_COLOUR[status]

        popup_html = f"""
        <div style="font-family:sans-serif;font-size:13px;min-width:160px">
            <b style="font-size:14px">{props['name']}</b><br>
            <span style="color:#666">OPW ref: {props['ref']}</span><br>
            <hr style="margin:4px 0">
            Stage anomaly: <b style="color:{colour}">{stage:+.3f} m</b><br>
            Status: <b style="color:{colour}">{STATUS_LABEL[status]}</b>
        </div>"""

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=colour,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{props['name']}  {stage:+.3f} m",
        ).add_to(fg)

    fg.add_to(m)


def add_shapefile_layer(m, gdf, layer_name: str, style: dict) -> None:
    """Add a GeoDataFrame as a GeoJson overlay, reprojected to WGS84."""
    import folium

    if gdf is None or len(gdf) == 0:
        return

    gdf_wgs = gdf.to_crs("EPSG:4326") if str(gdf.crs) != "EPSG:4326" else gdf

    folium.GeoJson(
        gdf_wgs.__geo_interface__,
        name=layer_name,
        style_function=lambda _: style,
    ).add_to(m)


def generate_static_map(
    ckpt_dir:    Path,
    out_dir:     Path,
    model_label: str,
    rivers_gdf   = None,
    lakes_gdf    = None,
) -> Path | None:
    """
    Generate a static peak-flood HTML map for one model.
    Returns the output HTML path or None on failure.
    """
    import folium

    print(f"\n  Building static peak map …")
    data = load_inundation_data(ckpt_dir)
    if data is None:
        return None

    (pred_stage, true_stage, timestamps,
     node_itm, node_refs, node_names,
     dem, hand, affine, masks,
     bankfull, reliable) = data

    ev_start, ev_end, peak_idx = find_peak_event(true_stage, timestamps)
    peak_ts = timestamps[peak_idx]
    print(f"  Peak event: {peak_ts}  ({ev_end-ev_start} steps = "
          f"{int((ev_end-ev_start)*15/60)} h)")

    # Predicted and observed inundation at peak
    inn_pred = stage_to_inundation(pred_stage[peak_idx], hand, masks, bankfull)
    inn_obs  = stage_to_inundation(true_stage[peak_idx], hand, masks, bankfull)

    geojson_pred = inundation_to_geojson(inn_pred, affine, dem)
    geojson_obs  = inundation_to_geojson(inn_obs,  affine, dem)
    gauges_gj    = gauges_to_geojson(
        node_itm, node_refs, pred_stage[peak_idx], node_names
    )

    # Save GeoJSONs for GIS import
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "inundation_peak_predicted.geojson", "w") as f:
        json.dump(geojson_pred, f)
    with open(out_dir / "inundation_peak_observed.geojson", "w") as f:
        json.dump(geojson_obs, f)
    with open(out_dir / "gauges.geojson", "w") as f:
        json.dump(gauges_gj, f)

    # Build Folium map
    m = make_base_map()

    # Lakes and rivers (base context layers)
    if lakes_gdf is not None:
        add_shapefile_layer(m, lakes_gdf, "Lakes / Reservoirs", {
            "fillColor": "#4A90D9", "color": "#2266AA",
            "weight": 1.0, "fillOpacity": 0.5,
        })
    if rivers_gdf is not None:
        add_shapefile_layer(m, rivers_gdf, "River Network", {
            "color": "#1A5FA8", "weight": 1.2, "fillOpacity": 0,
        })

    # Observed inundation (grey reference)
    add_inundation_layer(m, geojson_obs,
                         layer_name="Observed inundation",
                         color="#888888", fill_opacity=0.30)

    # Predicted inundation (cyan, primary)
    add_inundation_layer(m, geojson_pred,
                         layer_name=f"Predicted inundation ({model_label})",
                         color="#00D9F2", fill_opacity=0.50)

    # Gauge markers
    add_gauge_layer(m, gauges_gj)

    # Title as a floating HTML element
    pixel_m  = abs(affine.a)
    pred_km2 = inn_pred.sum() * pixel_m**2 / 1e6
    obs_km2  = inn_obs.sum()  * pixel_m**2 / 1e6
    title_html = f"""
    <div style="position:fixed;top:12px;left:60px;z-index:1000;
                background:rgba(255,255,255,0.92);padding:10px 14px;
                border-radius:8px;font-family:sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,0.25);max-width:380px">
        <b style="font-size:14px">Lee Catchment Flood Inundation</b><br>
        <span style="font-size:12px;color:#555">
            Peak: {peak_ts.strftime('%Y-%m-%d %H:%M')}&nbsp;&nbsp;
            Model: {model_label}
        </span><br>
        <span style="font-size:12px">
            Predicted: <b style="color:#00B8CC">{pred_km2:.2f} km²</b> &nbsp;
            Observed: <b style="color:#666">{obs_km2:.2f} km²</b>
        </span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = out_dir / "peak_inundation.html"
    m.save(str(out_path))
    print(f"  Saved: {out_path}")
    return out_path


def generate_animated_map(
    ckpt_dir:    Path,
    out_dir:     Path,
    model_label: str,
    n_frames:    int  = 48,     # 48 × 15 min = 12 hours of animation
    rivers_gdf   = None,
    lakes_gdf    = None,
) -> Path | None:
    """
    Generate an animated flood map using TimestampedGeoJson.
    Each frame is one 15-minute prediction step.
    """
    try:
        from folium.plugins import TimestampedGeoJson
        import folium
    except ImportError:
        print("  folium.plugins not available — skipping animation")
        return None

    print(f"\n  Building animated map ({n_frames} frames) …")
    data = load_inundation_data(ckpt_dir)
    if data is None:
        return None

    (pred_stage, true_stage, timestamps,
     node_itm, node_refs, node_names,
     dem, hand, affine, masks,
     bankfull, reliable) = data

    ev_start, ev_end, peak_idx = find_peak_event(true_stage, timestamps)

    # Subsample frames around the event
    step     = max(1, (ev_end - ev_start) // n_frames)
    frame_ts = list(range(ev_start, ev_end, step))[:n_frames]
    print(f"  Event: {timestamps[ev_start].date()} – {timestamps[ev_end-1].date()}"
          f"  {len(frame_ts)} frames")

    pixel_m = abs(affine.a)

    # Build a TimestampedGeoJson feature collection
    # Each feature has a "time" property which the plugin uses for the slider
    timed_features = []
    for t_idx in frame_ts:
        inn = stage_to_inundation(pred_stage[t_idx], hand, masks, bankfull)
        gj  = inundation_to_geojson(inn, affine, dem)
        if not gj["features"]:
            continue
        feat = gj["features"][0].copy()
        feat["properties"]["time"]  = timestamps[t_idx].isoformat()
        feat["properties"]["style"] = {
            "color":       "#FFFFFF",
            "weight":      1.0,
            "fillColor":   "#00D9F2",
            "fillOpacity": 0.50,
        }
        feat["properties"]["area_km2"] = round(
            inn.sum() * pixel_m**2 / 1e6, 3
        )
        timed_features.append(feat)

    if not timed_features:
        print("  No inundation frames generated — skipping animation")
        return None

    m = make_base_map()

    if lakes_gdf is not None:
        add_shapefile_layer(m, lakes_gdf, "Lakes / Reservoirs", {
            "fillColor": "#4A90D9", "color": "#2266AA",
            "weight": 1.0, "fillOpacity": 0.5,
        })
    if rivers_gdf is not None:
        add_shapefile_layer(m, rivers_gdf, "River Network", {
            "color": "#1A5FA8", "weight": 1.2, "fillOpacity": 0,
        })

    # Animated inundation layer
    TimestampedGeoJson(
        data={"type": "FeatureCollection", "features": timed_features},
        period="PT15M",        # 15-minute steps
        duration="PT15M",
        auto_play=False,
        loop=True,
        max_speed=10,
        loop_button=True,
        date_options="YYYY-MM-DD HH:mm",
        time_slider_drag_update=True,
        add_last_point=True,
    ).add_to(m)

    # Static gauge layer (positions only — no time-varying stage for simplicity)
    gauges_gj = gauges_to_geojson(
        node_itm, node_refs, pred_stage[peak_idx], node_names
    )
    add_gauge_layer(m, gauges_gj, layer_name="OPW Gauges (peak stage)")

    title_html = f"""
    <div style="position:fixed;top:12px;left:60px;z-index:1000;
                background:rgba(255,255,255,0.92);padding:10px 14px;
                border-radius:8px;font-family:sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,0.25)">
        <b style="font-size:14px">Lee Catchment — Flood Evolution</b><br>
        <span style="font-size:12px;color:#555">
            {timestamps[ev_start].strftime('%Y-%m-%d')} –
            {timestamps[ev_end-1].strftime('%Y-%m-%d')} &nbsp; | &nbsp;
            Model: {model_label}
        </span><br>
        <span style="font-size:11px;color:#777">
            Use the time slider below to step through the flood
        </span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=True).add_to(m)

    out_path = out_dir / "flood_animation.html"
    m.save(str(out_path))
    print(f"  Saved: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════
#  Shapefile loader (reused from generate_flood_maps.py)
# ══════════════════════════════════════════════════════════════════════

def load_shapefiles(rivers_path, lakes_path):
    try:
        import geopandas as gpd
    except ImportError:
        print("WARNING: geopandas not installed — shapefiles skipped.")
        return None, None

    def _load(path, label):
        if path is None: return None
        path = Path(path)
        if not path.exists():
            print(f"WARNING: {label} shapefile not found: {path}")
            return None
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:2157")
        print(f"  {label}: {len(gdf)} features loaded")
        return gdf

    return _load(rivers_path, "Rivers"), _load(lakes_path, "Lakes")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate interactive flood maps on Google Maps / OSM"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--all-models", action="store_true",
                      help="All models in checkpoints/ (default)")
    mode.add_argument("--model", type=str, default=None,
                      help="Single model, e.g. st_gnn_hand_edge")
    p.add_argument("--rivers",  type=Path, default=None,
                   help="River network shapefile (.shp)")
    p.add_argument("--lakes",   type=Path, default=None,
                   help="Lakes / reservoirs shapefile (.shp)")
    p.add_argument("--animate", action="store_true",
                   help="Also generate animated flood evolution map")
    p.add_argument("--frames",  type=int, default=48,
                   help="Frames for animation (default 48 = 12 h)")
    p.add_argument("--out",     type=Path, default=OUT_ROOT)
    args = p.parse_args()

    try:
        import folium
    except ImportError:
        print("folium not installed. Run: pip install folium")
        sys.exit(1)

    print("\n── Loading shapefiles ──")
    rivers_gdf, lakes_gdf = load_shapefiles(args.rivers, args.lakes)

    shared = dict(rivers_gdf=rivers_gdf, lakes_gdf=lakes_gdf)

    if args.model:
        model_dirs = [CKPT_ROOT / args.model]
    else:
        model_dirs = [d for d in sorted(CKPT_ROOT.iterdir())
                      if d.is_dir() and not d.name.startswith(".")]

    for md in model_dirs:
        label   = MODEL_LABELS.get(md.name, md.name)
        out_dir = args.out / md.name
        print(f"\n{'='*50}\n  Model: {label}\n{'='*50}")
        try:
            generate_static_map(md, out_dir, label, **shared)
            if args.animate:
                generate_animated_map(md, out_dir, label,
                                      n_frames=args.frames, **shared)
        except Exception as exc:
            import traceback
            print("  ERROR:", exc)
            traceback.print_exc()

    print(f"\n✓ Web maps written to: {args.out}")
    print("  Open any .html file directly in your browser.")
