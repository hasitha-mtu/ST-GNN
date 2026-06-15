"""
generate_kmz.py  –  Export flood inundation to Google Earth KMZ
===============================================================
Produces self-contained KMZ files (zipped KML + embedded images) that open
directly in Google Earth Pro (desktop) or Google Earth Web (earth.google.com)
with no account or API key required.

Two output formats per model:

1. peak_inundation.kmz
   - Flood inundation polygon at event peak (vector, clean edges)
   - Observed vs predicted comparison as separate folders
   - OPW gauge placemarks with coloured icons and info balloons
   - River network and lakes from shapefiles (optional)

2. flood_animation.kmz  (with --animate)
   - One GroundOverlay PNG image per timestep, each tagged with a
     TimeSpan so Google Earth's built-in time slider animates the flood
   - Far more efficient than polygon KML for many frames
   - Play at any speed using the time animation controls in Google Earth

How to open in Google Earth
---------------------------
Google Earth Pro:  File → Open → select the .kmz file
Google Earth Web:  Drag and drop the .kmz file into the browser window
                   (earth.google.com)

Output
------
  figures/kml/{model_name}/
      peak_inundation.kmz      — static peak flood (open in Google Earth)
      flood_animation.kmz      — animated flood evolution
      peak_inundation.kml      — plain KML for ArcGIS / QGIS import

Usage
-----
    python src/generate_kmz.py
    python src/generate_kmz.py --model st_gnn_hand_edge --animate
    python src/generate_kmz.py \\
        --rivers dataset/shapefiles/river_network.shp \\
        --lakes  dataset/shapefiles/lakes.shp \\
        --animate --frames 48
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
CKPT_ROOT = BASE_DIR / "checkpoints"
DEM_PATH  = BASE_DIR / "dataset/dem/COP-DEM-30m_itm.tif"
OUT_ROOT  = BASE_DIR / "figures/kml"

MODEL_LABELS = {
    "gru":              "PerNodeGRU",
    "lstm":             "PerNodeLSTM",
    "st_gnn_static":    "STGNNFlood (static)",
    "st_gnn_sar":       "STGNNFlood+SAR",
    "st_gnn_dyn_edge":  "STGNNFlood+DynEdge",
    "st_gnn_hand_edge": "STGNNFlood+HAND",
}


# ══════════════════════════════════════════════════════════════════════
#  Coordinate helpers
# ══════════════════════════════════════════════════════════════════════

def itm_to_wgs84(eastings, northings):
    """Convert ITM (EPSG:2157) coordinates to WGS84 lon/lat."""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)
    return t.transform(eastings, northings)   # returns (lons, lats)


def get_dem_bounds_wgs84(affine, H, W):
    """
    Return the WGS84 bounding box of the DEM raster.
    Used to georeference GroundOverlay PNG images in KML.
    """
    # Corner ITM coordinates
    west_itm  = affine.c
    east_itm  = affine.c + W * affine.a
    north_itm = affine.f
    south_itm = affine.f + H * affine.e   # affine.e is negative

    lons, lats = itm_to_wgs84(
        [west_itm, east_itm, west_itm, east_itm],
        [north_itm, north_itm, south_itm, south_itm],
    )
    return min(lats), max(lats), min(lons), max(lons)  # S, N, W, E


# ══════════════════════════════════════════════════════════════════════
#  Inundation raster → PNG overlay image
# ══════════════════════════════════════════════════════════════════════

def inundation_to_png_bytes(
    inundation: np.ndarray,   # [H, W] boolean
    r: int = 0, g: int = 217, b: int = 242,  # cyan #00D9F2
    alpha: int = 160,                          # 0–255
) -> bytes:
    """
    Convert a boolean inundation mask to a PNG image in memory.
    Flooded pixels → semi-transparent cyan. Dry pixels → fully transparent.
    Google Earth displays this as a GroundOverlay.
    """
    from PIL import Image

    H, W  = inundation.shape
    rgba  = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[inundation, 0] = r
    rgba[inundation, 1] = g
    rgba[inundation, 2] = b
    rgba[inundation, 3] = alpha

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════
#  Inundation raster → KML polygon
# ══════════════════════════════════════════════════════════════════════

def inundation_to_kml_polygon(inundation, affine, simplify_m=60.0):
    """
    Convert boolean inundation raster to a list of (lon, lat) ring tuples
    suitable for KML Polygon elements.

    Returns list of exterior rings (each a list of (lon, lat) tuples).
    """
    import rasterio.features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    mask   = inundation.astype(np.uint8)
    shapes = list(rasterio.features.shapes(mask, mask=mask, transform=affine))
    if not shapes:
        return []

    polys  = [shape(g) for g, v in shapes if v == 1]
    merged = unary_union(polys).simplify(simplify_m, preserve_topology=True)

    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)

    def ring_to_lonlat(coords):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        lons, lats = t.transform(xs, ys)
        return list(zip(lons, lats))

    rings = []
    geoms = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
    for geom in geoms:
        if not geom.is_empty:
            rings.append(ring_to_lonlat(list(geom.exterior.coords)))
    return rings


# ══════════════════════════════════════════════════════════════════════
#  Data loading (reuses logic from generate_web_map.py)
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

def load_data(ckpt_dir: Path):
    """Load predictions, DEM, HAND, node metadata. Returns None if missing."""
    mean_files  = sorted(ckpt_dir.glob("test_predictions_*steps_mean.npy"))
    single_file = ckpt_dir / "test_predictions.npy"
    if mean_files:
        pred_stage = np.load(mean_files[0])
        print(f"  Predictions: {mean_files[0].name}")
    elif single_file.exists():
        pred_stage = np.load(single_file)
    else:
        print(f"  SKIPPED: no predictions in {ckpt_dir.name}")
        return None

    import rasterio
    from scipy.ndimage import distance_transform_edt
    y     = np.load(PROC_DIR / "y.npy", mmap_mode="r")
    T     = y.shape[0]
    ts_s  = int(T * 0.85)
    ts_e  = T - 4
    true_stage = y[ts_s:ts_e]
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(pd.read_csv(PROC_DIR/"timestamps.csv")["timestamp"])
        .iloc[ts_s:ts_e].values
    )

    with rasterio.open(DEM_PATH) as src:
        dem    = src.read(1).astype(np.float32)
        affine = src.transform
        nd     = src.nodata or -9999.0
    dem[dem == nd] = np.nan
    nan_mask = np.isnan(dem)
    H, W = dem.shape

    hand_path = DEM_PATH.parent / "hand_raster.tif"
    if hand_path.exists():
        with rasterio.open(hand_path) as src:
            hand = src.read(1).astype(np.float32)
    else:
        from generate_web_map import _quick_stream_mask
        stream_mask = _quick_stream_mask(dem, nan_mask)
        _, idx = distance_transform_edt(stream_mask == 0, return_indices=True)
        hand = np.clip(dem - dem[idx[0], idx[1]], 0, None).astype(np.float32)
        hand[nan_mask] = np.nan

    nodes_df = pd.read_csv(GRAPH_DIR/"nodes.csv")
    if "easting_itm" in nodes_df.columns:
        node_itm = nodes_df[["easting_itm","northing_itm"]].values
    else:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326","EPSG:2157",always_xy=True)
        E, N = tr.transform(nodes_df["lon"].values, nodes_df["lat"].values)
        node_itm = np.column_stack([E, N])
    node_refs  = nodes_df["ref"].astype(str).tolist()
    node_names = nodes_df["name"].tolist() if "name" in nodes_df.columns else node_refs

    valid = ~nan_mask
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
            dem, hand, affine, masks, H, W, bankfull, reliable)



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


def find_peak_event(true_stage, timestamps, threshold=0.3, min_h=6):
    max_s = np.nanmax(true_stage, axis=1)
    best  = None
    i = 0
    while i < len(max_s):
        if max_s[i] > threshold:
            j = i
            while j < len(max_s) and max_s[j] > threshold: j += 1
            if (j-i)*15/60 >= min_h:
                pk = int(i + np.argmax(max_s[i:j]))
                if best is None or max_s[pk] > max_s[best[2]]:
                    best = (i, j, pk)
            i = j
        else:
            i += 1
    if best is None:
        pk = int(np.nanargmax(max_s))
        best = (max(0,pk-48), min(len(timestamps),pk+48), pk)
    return best


# ══════════════════════════════════════════════════════════════════════
#  KMZ builders
# ══════════════════════════════════════════════════════════════════════

def build_peak_kmz(
    ckpt_dir:    Path,
    out_dir:     Path,
    model_label: str,
    rivers_gdf   = None,
    lakes_gdf    = None,
) -> Path | None:
    """
    Generate peak_inundation.kmz — opens directly in Google Earth.

    Structure inside the KMZ:
        doc.kml              root KML document
        images/pred.png      predicted inundation overlay image
        images/obs.png       observed inundation overlay image
    """
    data = load_data(ckpt_dir)
    if data is None: return None
    (pred_stage, true_stage, timestamps,
     node_itm, node_refs, node_names,
     dem, hand, affine, masks, H, W,
     bankfull, reliable) = data

    ev_start, ev_end, peak_idx = find_peak_event(true_stage, timestamps)
    peak_ts   = timestamps[peak_idx]
    pixel_m   = abs(affine.a)
    south, north, west, east = get_dem_bounds_wgs84(affine, H, W)

    inn_pred = stage_to_inundation(pred_stage[peak_idx], hand, masks, bankfull)
    inn_obs  = stage_to_inundation(true_stage[peak_idx], hand, masks, bankfull)

    pred_km2 = inn_pred.sum() * pixel_m**2 / 1e6
    obs_km2  = inn_obs.sum()  * pixel_m**2 / 1e6

    pred_png = inundation_to_png_bytes(inn_pred,
                                       r=0, g=217, b=242, alpha=160)
    obs_png  = inundation_to_png_bytes(inn_obs,
                                       r=120, g=120, b=120, alpha=130)

    # Build KML string
    def ground_overlay(name, href, south, north, west, east,
                       description="", draw_order=1):
        return f"""
    <GroundOverlay>
        <name>{name}</name>
        <description>{description}</description>
        <drawOrder>{draw_order}</drawOrder>
        <Icon><href>{href}</href></Icon>
        <LatLonBox>
            <north>{north:.6f}</north>
            <south>{south:.6f}</south>
            <east>{east:.6f}</east>
            <west>{west:.6f}</west>
        </LatLonBox>
    </GroundOverlay>"""

    # Gauge placemarks
    lons, lats = itm_to_wgs84(node_itm[:, 0], node_itm[:, 1])
    gauge_placemarks = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        stage  = float(pred_stage[peak_idx, i]) if not np.isnan(pred_stage[peak_idx, i]) else 0.0
        status = "high" if stage > 0.3 else "moderate" if stage > 0.1 else "normal"
        style_id = {"high": "#gauge_high", "moderate": "#gauge_mod",
                    "normal": "#gauge_ok"}[status]
        gauge_placemarks.append(f"""
    <Placemark>
        <name>{node_names[i]}</name>
        <styleUrl>{style_id}</styleUrl>
        <description><![CDATA[
            <b>{node_names[i]}</b><br/>
            OPW ref: {node_refs[i]}<br/>
            Stage anomaly: <b>{stage:+.3f} m</b><br/>
            Status: <b>{status.upper()}</b>
        ]]></description>
        <Point>
            <coordinates>{lon:.6f},{lat:.6f},0</coordinates>
        </Point>
    </Placemark>""")

    # Shapefile layers
    shapefile_kml = ""
    if rivers_gdf is not None and len(rivers_gdf) > 0:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:2157","EPSG:4326",always_xy=True)
        gdf_wgs = rivers_gdf.to_crs("EPSG:4326")
        river_lines = []
        for geom in gdf_wgs.geometry:
            if geom is None or geom.is_empty: continue
            lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                coords = " ".join(f"{x:.6f},{y:.6f},0"
                                  for x, y in zip(*line.xy))
                river_lines.append(f"""
        <Placemark>
            <styleUrl>#river_style</styleUrl>
            <LineString>
                <tessellate>1</tessellate>
                <coordinates>{coords}</coordinates>
            </LineString>
        </Placemark>""")
        shapefile_kml += f"""
    <Folder>
        <name>River Network</name>
        <visibility>1</visibility>
        {''.join(river_lines)}
    </Folder>"""

    if lakes_gdf is not None and len(lakes_gdf) > 0:
        gdf_wgs = lakes_gdf.to_crs("EPSG:4326")
        lake_polys = []
        for geom in gdf_wgs.geometry:
            if geom is None or geom.is_empty: continue
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                if poly.is_empty: continue
                ext_coords = " ".join(f"{x:.6f},{y:.6f},0"
                                      for x, y in zip(*poly.exterior.xy))
                lake_polys.append(f"""
        <Placemark>
            <styleUrl>#lake_style</styleUrl>
            <Polygon>
                <outerBoundaryIs>
                    <LinearRing>
                        <tessellate>1</tessellate>
                        <coordinates>{ext_coords}</coordinates>
                    </LinearRing>
                </outerBoundaryIs>
            </Polygon>
        </Placemark>""")
        shapefile_kml += f"""
    <Folder>
        <name>Lakes / Reservoirs</name>
        <visibility>1</visibility>
        {''.join(lake_polys)}
    </Folder>"""

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Lee Catchment Flood — {model_label}</name>
    <description>
        Peak flood inundation at {peak_ts.strftime('%Y-%m-%d %H:%M')}
        Predicted: {pred_km2:.2f} km² | Observed: {obs_km2:.2f} km²
        Model: {model_label}
    </description>

    <!-- Styles for gauge icons -->
    <Style id="gauge_high">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>0.9</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.7</scale></LabelStyle>
    </Style>
    <Style id="gauge_mod">
        <IconStyle>
            <color>ff0080ff</color>
            <scale>0.9</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.7</scale></LabelStyle>
    </Style>
    <Style id="gauge_ok">
        <IconStyle>
            <color>ff00ff00</color>
            <scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.6</scale></LabelStyle>
    </Style>
    <Style id="river_style">
        <LineStyle>
            <color>ffA85018</color>
            <width>1.5</width>
        </LineStyle>
    </Style>
    <Style id="lake_style">
        <LineStyle><color>ffAA6614</color><width>0.8</width></LineStyle>
        <PolyStyle><color>99D99040</color></PolyStyle>
    </Style>

    <!-- Observed inundation (grey, drawn first / underneath) -->
    <Folder>
        <name>Observed inundation</name>
        <visibility>1</visibility>
        {ground_overlay(
            f"Observed — {obs_km2:.2f} km²",
            "images/obs.png",
            south, north, west, east,
            description=f"Observed inundation at {peak_ts}",
            draw_order=1
        )}
    </Folder>

    <!-- Predicted inundation (cyan, on top) -->
    <Folder>
        <name>Predicted inundation ({model_label})</name>
        <visibility>1</visibility>
        {ground_overlay(
            f"Predicted — {pred_km2:.2f} km²",
            "images/pred.png",
            south, north, west, east,
            description=f"Predicted inundation at {peak_ts}",
            draw_order=2
        )}
    </Folder>

    {shapefile_kml}

    <!-- OPW Gauge stations -->
    <Folder>
        <name>OPW Gauges (stage at peak)</name>
        <visibility>1</visibility>
        {''.join(gauge_placemarks)}
    </Folder>

</Document>
</kml>"""

    # Pack into KMZ (zip)
    out_dir.mkdir(parents=True, exist_ok=True)
    kmz_path = out_dir / "peak_inundation.kmz"
    with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml",          kml.encode("utf-8"))
        zf.writestr("images/pred.png",  pred_png)
        zf.writestr("images/obs.png",   obs_png)

    # Also save plain KML for ArcGIS/QGIS import
    kml_path = out_dir / "peak_inundation.kml"
    kml_path.write_text(kml, encoding="utf-8")

    pixel_m = abs(affine.a)
    print(f"  Peak:    {peak_ts}  predicted={pred_km2:.2f} km²  observed={obs_km2:.2f} km²")
    print(f"  Saved:   {kmz_path.name}  ({kmz_path.stat().st_size/1024:.0f} KB)")
    return kmz_path


def build_animation_kmz(
    ckpt_dir:    Path,
    out_dir:     Path,
    model_label: str,
    n_frames:    int = 48,
) -> Path | None:
    """
    Generate flood_animation.kmz — animated GroundOverlay with TimeSpan.

    Each frame is a PNG image paired with a KML TimeSpan. Google Earth's
    time slider steps through frames automatically when you press Play.
    """
    data = load_data(ckpt_dir)
    if data is None: return None
    (pred_stage, true_stage, timestamps,
     node_itm, node_refs, node_names,
     dem, hand, affine, masks, H, W,
     bankfull, reliable) = data

    ev_start, ev_end, peak_idx = find_peak_event(true_stage, timestamps)
    step       = max(1, (ev_end - ev_start) // n_frames)
    frame_idxs = list(range(ev_start, ev_end, step))[:n_frames]
    south, north, west, east = get_dem_bounds_wgs84(affine, H, W)
    pixel_m = abs(affine.a)

    print(f"  Event: {timestamps[ev_start].date()} – {timestamps[ev_end-1].date()}"
          f"  {len(frame_idxs)} frames")

    overlays_kml = []
    images       = {}   # filename → PNG bytes

    for fi, t_idx in enumerate(frame_idxs):
        ts  = timestamps[t_idx]
        inn = stage_to_inundation(pred_stage[t_idx], hand, masks, bankfull)

        # TimeSpan: each frame spans one 15-minute interval
        ts_begin = ts.isoformat()
        ts_end_t = (ts + pd.Timedelta(minutes=15)).isoformat()
        area_km2 = inn.sum() * pixel_m**2 / 1e6

        img_name  = f"images/frame_{fi:04d}.png"
        images[img_name] = inundation_to_png_bytes(
            inn, r=0, g=217, b=242, alpha=155
        )

        overlays_kml.append(f"""
    <GroundOverlay>
        <name>{ts.strftime('%Y-%m-%d %H:%M')}  ({area_km2:.2f} km²)</name>
        <TimeSpan>
            <begin>{ts_begin}</begin>
            <end>{ts_end_t}</end>
        </TimeSpan>
        <drawOrder>1</drawOrder>
        <Icon><href>{img_name}</href></Icon>
        <LatLonBox>
            <north>{north:.6f}</north>
            <south>{south:.6f}</south>
            <east>{east:.6f}</east>
            <west>{west:.6f}</west>
        </LatLonBox>
    </GroundOverlay>""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Lee Flood Animation — {model_label}</name>
    <description>
        Use the time slider in Google Earth to animate the flood.
        Event: {timestamps[ev_start].strftime('%Y-%m-%d')} to
               {timestamps[ev_end-1].strftime('%Y-%m-%d')}
        Model: {model_label}  |  {len(frame_idxs)} frames × 15 min
    </description>
    <Folder>
        <name>Flood evolution ({model_label})</name>
        {''.join(overlays_kml)}
    </Folder>
</Document>
</kml>"""

    out_dir.mkdir(parents=True, exist_ok=True)
    kmz_path = out_dir / "flood_animation.kmz"
    with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml.encode("utf-8"))
        for img_name, img_bytes in images.items():
            zf.writestr(img_name, img_bytes)

    size_mb = kmz_path.stat().st_size / 1024**2
    print(f"  Saved:   {kmz_path.name}  ({size_mb:.1f} MB)  "
          f"— open in Google Earth and press Play ▶")
    return kmz_path


# ══════════════════════════════════════════════════════════════════════
#  Shapefile loading
# ══════════════════════════════════════════════════════════════════════

def load_shapefiles(rivers_path, lakes_path):
    try:
        import geopandas as gpd
    except ImportError:
        print("WARNING: geopandas not installed — shapefiles skipped.")
        return None, None

    def _load(p, label):
        if p is None: return None
        p = Path(p)
        if not p.exists():
            print(f"WARNING: {label} not found: {p}")
            return None
        gdf = gpd.read_file(p)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:2157")
        print(f"  {label}: {len(gdf)} features")
        return gdf

    return _load(rivers_path, "Rivers"), _load(lakes_path, "Lakes")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export flood inundation to Google Earth KMZ"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--all-models", action="store_true",
                      help="All models in checkpoints/ (default)")
    mode.add_argument("--model", type=str, default=None)
    p.add_argument("--rivers",  type=Path, default=None,
                   help="River network shapefile")
    p.add_argument("--lakes",   type=Path, default=None,
                   help="Lakes shapefile")
    p.add_argument("--animate", action="store_true",
                   help="Also generate animated flood_animation.kmz")
    p.add_argument("--frames",  type=int,  default=48,
                   help="Animation frames (default 48 = 12 h at 15-min steps)")
    p.add_argument("--out",     type=Path, default=OUT_ROOT)
    args = p.parse_args()

    if not DEM_PATH.exists():
        print(f"ITM DEM not found: {DEM_PATH}")
        print("Run: python src/data/precompute_hand_edges_v1.py")
        sys.exit(1)

    print("\n── Loading shapefiles ──")
    rivers_gdf, lakes_gdf = load_shapefiles(args.rivers, args.lakes)

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
            build_peak_kmz(md, out_dir, label, rivers_gdf, lakes_gdf)
            if args.animate:
                build_animation_kmz(md, out_dir, label, args.frames)
        except Exception as exc:
            import traceback
            print("  ERROR:", exc)
            traceback.print_exc()

    print(f"\n✓ KMZ files written to: {args.out}")
    print("  Open in Google Earth Pro:  File → Open → .kmz")
    print("  Open in Google Earth Web:  Drag .kmz into earth.google.com")
