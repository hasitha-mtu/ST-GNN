"""
validate_flood_maps.py
═══════════════════════════════════════════════════════════════════════
Validates HAND-based flood inundation predictions from the ST-GNN model
against Sentinel-1 SAR-derived flood masks and OPW CFRAM design extents.

Three validation tiers
───────────────────────
Tier A — Sentinel-1 SAR comparison (quantitative spatial)
  Downloads Sentinel-1 GRD products for flood events in the test period,
  derives a binary flood mask using change detection against a pre-flood
  reference, and computes standard flood mapping skill scores against
  the HAND-predicted extent at the same timestamp.

  Metrics: CSI (Critical Success Index), Hit Rate, False Alarm Ratio,
           Bias, F1-score, and a confusion map (TP/FP/FN/TN) saved as
           a GeoTIFF and PNG.

Tier B — OPW CFRAM plausibility check (qualitative spatial)
  Loads the OPW 1% AEP (100-year) CFRAM flood extent for the Lee
  catchment from floodinfo.ie (WFS service, no login required) and
  checks that the predicted inundation does not exceed these bounds
  at implausible locations.

Tier C — OPW FFWS alert cross-reference (event-level temporal)
  Reads OPW Flood Forecasting and Warning Service alert records and
  checks whether the model correctly predicts stage > bankfull at the
  nodes and times that the OPW operationally classified as flood events.

  Metrics: Precision, Recall, F1 for flood-event detection.

Sentinel-1 data access
────────────────────────
Data is freely available from the Copernicus Data Space Ecosystem:
  https://dataspace.copernicus.eu/

  Option A — Direct download (recommended for one-off validation):
    Register at dataspace.copernicus.eu → search for Lee catchment
    GRD products during known flood events → download .zip → unzip.

  Option B — sentinelhub-py API (automated):
    Requires a free Copernicus Data Space account and OAuth client.
    Set credentials in environment variables:
      SENTINELHUB_CLIENT_ID, SENTINELHUB_CLIENT_SECRET

  Option C — Google Earth Engine (if you have access):
    ee.ImageCollection('COPERNICUS/S1_GRD') — no download needed.

Key flood events in the Lee catchment test period (Jan 2025–Mar 2026):
  2025-01-12  Stage at 19114 (Carrigrohane): ~2.1 m (storm Éowyn)
  2024-11-28  Stage at 19114: ~2.26 m (pre-test, check if in val split)
  2023-10-20  Stage at 19114: ~3.055 m (exceptional — in training split)

Usage
──────
  # Tier A: SAR validation for a specific event
  python src/validate_flood_maps.py \\
      --mode sar \\
      --event-date 2025-01-12 \\
      --sar-flood   dataset/validation/s1_flood_20250112.tif \\
      --sar-ref     dataset/validation/s1_reference_2024summer.tif \\
      --model       st_gnn_hand_edge

  # Tier B: CFRAM plausibility check
  python src/validate_flood_maps.py --mode cfram --model st_gnn_hand_edge

  # Tier C: FFWS alert cross-reference
  python src/validate_flood_maps.py \\
      --mode ffws \\
      --ffws-csv dataset/validation/opw_ffws_alerts.csv

  # All tiers
  python src/validate_flood_maps.py --mode all --event-date 2025-01-12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR            = Path(__file__).resolve().parent.parent
PROC_DIR            = BASE_DIR / "dataset/processed"
GRAPH_DIR           = BASE_DIR / "dataset/graph"
CKPT_ROOT           = BASE_DIR / "checkpoints"
DEM_PATH            = BASE_DIR / "dataset/dem/COP-DEM-30m_itm.tif"
VAL_DIR             = BASE_DIR / "dataset/validation"
FIG_DIR             = BASE_DIR / "figures/validation_v3"
RIVER_SHAPEFILE     = BASE_DIR / "shapefiles/RiverNetwork/RiverNetwork.shp"
LAKES_SHAPEFILE     = BASE_DIR / "shapefiles/LeeLakes/LeeLakes.shp"
CATCHMENT_SHAPEFILE = BASE_DIR / "shapefiles/Lee-catchment/Lee-catchment.shp"

MODEL_LABELS = {
    "gru":              "PerNodeGRU",
    "lstm":             "PerNodeLSTM",
    "st_gnn_static":    "STGNNFlood (static)",
    "st_gnn_sar":       "STGNNFlood+SAR",
    "st_gnn_dyn_edge":  "STGNNFlood+DynEdge",
    "st_gnn_hand_edge": "STGNNFlood+HAND",
}


# ═════════════════════════════════════════════════════════════════════
# Shared: load predictions and build inundation at a specific timestamp
# ═════════════════════════════════════════════════════════════════════

def load_predictions_for_event(
    model_name:  str,
    event_date:  str,          # ISO date OR datetime e.g. "2025-11-11T18:21"
    horizon:     int = 4,
    max_lag_h:   float = 6.0,  # warn if nearest timestep > this many hours away
) -> tuple[np.ndarray, pd.Timestamp] | None:
    """
    Load ensemble mean predictions for a model and find the timestep
    nearest to event_date.

    Returns (pred_stage [N], matched_timestamp) or None.
    """
    ckpt_dir   = CKPT_ROOT / model_name
    mean_files = sorted(ckpt_dir.glob(
        f"test_predictions_{horizon}steps_mean.npy"))
    if not mean_files:
        mean_files = sorted(ckpt_dir.glob("test_predictions_*steps_mean.npy"))
    if not mean_files:
        single = ckpt_dir / "test_predictions.npy"
        if not single.exists():
            print(f"  No predictions found for {model_name}")
            return None
        pred_all = np.load(single)
    else:
        pred_all = np.load(mean_files[0])

    # Load timestamps
    y      = np.load(PROC_DIR / "y.npy", mmap_mode="r")
    T      = y.shape[0]
    T_out  = horizon
    ts_s   = int(T * 0.85)
    ts_e   = T - T_out
    all_ts = pd.to_datetime(
        pd.read_csv(PROC_DIR / "timestamps.csv")["timestamp"]
    )
    timestamps = pd.DatetimeIndex(all_ts.iloc[ts_s:ts_e].values)

    # Find nearest timestep to event_date
    target    = pd.Timestamp(event_date)
    idx       = int(np.argmin(np.abs(timestamps - target)))
    matched   = timestamps[idx]
    lag_h     = abs((matched - target).total_seconds()) / 3600

    if lag_h > max_lag_h:
        print(f"  WARNING: nearest test timestep is {lag_h:.1f} h from "
              f"{event_date}.")
        print(f"  Tip: pass the SAR acquisition time for a tighter match,")
        print(f"  e.g. --event-date 2025-11-11T18:21 for an 18:21 UTC pass.")

    print(f"  Event: {event_date}  →  matched: {matched}  "
          f"(lag: {lag_h:.1f} h)")
    return pred_all[idx], matched


def build_hand_inundation(
    pred_stage:           np.ndarray,
    bankfull:             np.ndarray | None = None,
    hand_flood_threshold: float | None      = None,
) -> tuple[np.ndarray, object, tuple]:
    """
    Build a boolean inundation raster [H, W] from predicted stage.
    hand_flood_threshold: if set, floods HAND <= this value in active
    catchments (calibrated); otherwise floods HAND <= stage_anomaly.
    Returns (inundation, affine, (H, W)).
    """
    import rasterio
    from scipy.spatial import cKDTree

    # Load DEM and HAND
    with rasterio.open(DEM_PATH) as src:
        dem    = src.read(1).astype(np.float32)
        affine = src.transform
        nd     = src.nodata or -9999.0
    dem[dem == nd] = np.nan
    H, W = dem.shape

    hand_path = DEM_PATH.parent / "hand_raster.tif"
    with rasterio.open(hand_path) as src:
        hand = src.read(1).astype(np.float32)

    # Load gauge ITM positions
    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    if "easting_itm" in nodes_df.columns:
        node_itm = nodes_df[["easting_itm", "northing_itm"]].values
    else:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        E, N = t.transform(nodes_df["lon"].values, nodes_df["lat"].values)
        node_itm = np.column_stack([E, N])

    # Build catchment masks using Voronoi nearest-gauge assignment.
    #
    # D8 upstream BFS was previously used here, but the raw pysheds D8
    # does not include depression filling or flat-area routing. On the
    # Lee's flat floodplain (30m DEM), D8 produces 1-pixel spaghetti
    # paths — node 19114 (Carrigrohane) was assigned only 19 pixels
    # when it should cover ~130,000. Voronoi gives each pixel to its
    # nearest gauge in ITM space, which is geographically reasonable
    # for inundation mapping even if it lacks watershed-precise boundaries.
    #
    # To restore D8 catchments, regenerate fdir.npz with depression-filled
    # D8 using pysheds.Grid.flowdir() after Grid.fill_depressions() and
    # Grid.resolve_flats(). See precompute_hand_edges.py.
    rows, cols = np.mgrid[0:H, 0:W]
    px_e = affine.c + cols * affine.a
    px_n = affine.f + rows * affine.e
    tree = cKDTree(node_itm)
    dist_m, nearest = tree.query(
        np.column_stack([px_e.ravel(), px_n.ravel()]), workers=-1)
    nearest = nearest.reshape(H, W)
    dist_m  = dist_m.reshape(H, W) * abs(affine.a)
    # Limit each node's Voronoi region to 15 km — beyond this any
    # flood prediction is not credible for that gauge.
    MAX_VORONOI_KM = 15.0
    valid_voronoi  = dist_m <= MAX_VORONOI_KM * 1000
    masks = [(nearest == i) & valid_voronoi for i in range(len(node_itm))]

    # Bankfull thresholds
    if bankfull is None:
        bf_json = GRAPH_DIR / "bankfull_thresholds.json"
        if bf_json.exists():
            thr_map = json.load(open(bf_json))["thresholds"]
            refs    = nodes_df["ref"].astype(str).tolist()
            bankfull = np.array(
                [float(thr_map.get(r, 0.1)) for r in refs],
                dtype=np.float32)
        else:
            bankfull = np.full(len(node_itm), 0.1, dtype=np.float32)

    # Build inundation
    valid_hand = ~np.isnan(hand)
    result     = np.zeros((H, W), dtype=bool)
    n_active   = 0
    for i, (s, mask, thr) in enumerate(zip(pred_stage, masks, bankfull)):
        if s < thr or np.isnan(s): continue
        n_active += 1
        if hand_flood_threshold is not None:
            result |= mask & valid_hand & (hand <= hand_flood_threshold)
        else:
            result |= mask & valid_hand & (hand <= s)
    method = (f"fixed HAND ≤ {hand_flood_threshold} m"
              if hand_flood_threshold is not None else "HAND ≤ stage_anomaly")
    area_km2 = result.sum() * abs(affine.a)**2 / 1e6
    print(f"  Inundation: {method}  active={n_active}/{len(bankfull)}"
          f"  area={area_km2:.3f} km²")

    return result, affine, (H, W)



def calibrate_hand_threshold(sar_mask, pred_stage, bankfull, affine,
                              thresholds=None):
    """Sweep HAND thresholds and find best CSI against SAR mask."""
    if thresholds is None:
        thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    print(f"\n── HAND calibration sweep ──")
    print(f"  SAR reference: {sar_mask.sum():,} px = "
          f"{sar_mask.sum()*abs(affine.a)**2/1e6:.2f} km²")
    print(f"  {'HAND(m)':>8}  {'Pred km²':>9}  {'Obs km²':>9}  "
          f"{'CSI':>7}  {'HR':>7}  {'FAR':>7}  {'Bias':>7}")
    print("  " + "─"*63)
    results = {}
    af = abs(affine.a)**2 / 1e6
    for ht in thresholds:
        p, _, _ = build_hand_inundation(pred_stage, bankfull=bankfull,
                                         hand_flood_threshold=ht)
        TP = int(( p &  sar_mask).sum())
        FP = int(( p & ~sar_mask).sum())
        FN = int((~p &  sar_mask).sum())
        d   = TP+FP+FN
        csi = TP/d       if d      > 0 else 0.0
        hr  = TP/(TP+FN) if TP+FN  > 0 else 0.0
        far = FP/(TP+FP) if TP+FP  > 0 else 0.0
        b   = (TP+FP)/(TP+FN) if TP+FN > 0 else 0.0
        results[ht] = dict(HAND_m=ht,CSI=round(csi,4),HitRate=round(hr,4),
                           FAR=round(far,4),Bias=round(b,4),
                           pred_km2=round((TP+FP)*af,3),
                           obs_km2=round((TP+FN)*af,3),TP=TP,FP=FP,FN=FN)
        flag = " ← best" if csi==max(r["CSI"] for r in results.values())                and csi>0 else ""
        print(f"  {ht:>8.1f}  {(TP+FP)*af:>9.3f}  {(TP+FN)*af:>9.3f}  "
              f"{csi:>7.4f}  {hr:>7.4f}  {far:>7.4f}  {b:>7.4f}{flag}")
    best = max(results, key=lambda k: results[k]["CSI"])
    r    = results[best]
    print(f"\n  Best: HAND ≤ {best} m  CSI={r['CSI']:.4f}  "
          f"HR={r['HitRate']:.4f}  FAR={r['FAR']:.4f}")
    print(f"  Re-run with: --hand-flood-threshold {best}")
    return {"best_threshold_m": best, "results": results}

# ═════════════════════════════════════════════════════════════════════
# Tier A — Sentinel-1 SAR validation
# ═════════════════════════════════════════════════════════════════════

def compute_sar_flood_mask(
    sar_flood_path:  Path,
    sar_ref_path:    Path,
    threshold_db:    float = -3.0,
    min_area_px:     int   = 50,
    abs_water_db:    float = -15.0,
    max_hand_m:      float = 5.0,    # remove detections above this HAND elevation
) -> np.ndarray | None:
    """
    Derive a binary flood mask from Sentinel-1 GRD imagery using
    seasonally-normalised change detection.

    The Problem with Raw Change Detection
    ──────────────────────────────────────
    A simple (flood - reference) threshold fails when the reference is
    from a different season.  Autumn/winter fields (harvested, wet soil)
    are typically 1–3 dB BRIGHTER than summer crops because rougher bare
    soil and stubble scatter more energy back to the sensor.  Using a
    summer reference with a November flood image produces a positive mean
    change (+1.47 dB in the Lee catchment Nov/Jun comparison), so no
    pixels satisfy `change < -3 dB`.

    Fix — Normalised Change Detection
    ────────────────────────────────────
    1. Compute the scene-level seasonal offset:
         offset = nanmedian(flood_db) - nanmedian(ref_db)
         This represents the mean brightness difference due to season,
         vegetation state, and soil moisture — NOT due to flooding.

    2. Compute normalised change:
         norm_change = (flood_db - ref_db) - offset
         A pixel with norm_change << -3 dB is locally DARK relative to
         its expected seasonal level → likely open water / inundation.

    3. Cross-check with absolute water threshold:
         Pixels with flood_db < abs_water_db (-15 dB) are likely open
         water regardless of season (calm VV water: -20 to -14 dB).

    4. Combined mask = (norm_change < threshold_db) OR (flood_db < abs_water_db)
       This catches both relative darkening and absolute water surfaces.
    """
    import rasterio
    from scipy.ndimage import binary_opening, label, sum as ndimage_sum

    try:
        with rasterio.open(sar_flood_path) as src:
            flood_db  = src.read(1).astype(np.float32)
            flood_aff = src.transform
    except Exception as e:
        print(f"  Could not read SAR flood image: {e}")
        return None

    try:
        with rasterio.open(sar_ref_path) as src:
            ref_db = src.read(1).astype(np.float32)
    except Exception as e:
        print(f"  Could not read SAR reference: {e}")
        return None

    # Both images are already geocoded to the same DEM grid by
    # build_sar_reference.py — shapes must match
    if flood_db.shape != ref_db.shape:
        print(f"  Shape mismatch: flood={flood_db.shape} ref={ref_db.shape}")
        print(f"  Re-run build_sar_reference.py to regenerate on the same grid.")
        return None

    valid = ~np.isnan(flood_db) & ~np.isnan(ref_db) &             (flood_db != 0) & (ref_db != 0)

    # ── Seasonal offset correction ────────────────────────────────────
    flood_median = float(np.nanmedian(flood_db[valid]))
    ref_median   = float(np.nanmedian(ref_db[valid]))
    offset       = flood_median - ref_median

    print(f"  Flood image median:    {flood_median:.2f} dB")
    print(f"  Reference median:      {ref_median:.2f} dB")
    print(f"  Seasonal offset:       {offset:+.2f} dB  "
          f"({'Nov brighter than summer — expected' if offset > 0 else 'unexpected negative offset'})")

    # ── Normalised change detection ───────────────────────────────────
    norm_change    = (flood_db - ref_db) - offset
    change_mask    = (norm_change < threshold_db) & valid
    print(f"  Normalised change < {threshold_db} dB: {change_mask.sum():,} pixels")

    # ── Absolute water threshold ──────────────────────────────────────
    water_mask = (flood_db < abs_water_db) & valid
    print(f"  Absolute water (< {abs_water_db} dB):    {water_mask.sum():,} pixels")

    # ── Combined ──────────────────────────────────────────────────────
    raw_mask = change_mask | water_mask

    # Morphological opening to remove speckle
    struct     = np.ones((3, 3), dtype=bool)
    clean_mask = binary_opening(raw_mask, structure=struct, iterations=2)

    # Remove patches smaller than min_area_px
    labelled, n_feat = label(clean_mask)
    sizes = ndimage_sum(clean_mask, labelled, range(1, n_feat + 1))
    keep  = np.zeros_like(clean_mask)
    for lbl, sz in enumerate(sizes, start=1):
        if sz >= min_area_px:
            keep[labelled == lbl] = 1
    final_mask = keep.astype(bool)

    # ── HAND floodplain mask ──────────────────────────────────────────
    # SAR detections at high HAND values (upland areas) are noise:
    # wind roughening, agricultural variability, SAR layover.
    # True fluvial flood pixels occur only at low HAND (near-channel).
    # Mask out detections where HAND > max_hand_m (default 5 m).
    hand_path = DEM_PATH.parent / "hand_raster.tif"
    if hand_path.exists():
        import rasterio as _rio
        from rasterio.warp import reproject as _repr, Resampling as _RS
        with _rio.open(hand_path) as _hs:
            _hand_dem = _hs.read(1).astype(np.float32)
        if _hand_dem.shape != final_mask.shape:
            # Reproject HAND to match flood image grid
            _hand_rs = np.full(final_mask.shape, np.nan, dtype=np.float32)
            with _rio.open(hand_path) as _hs:
                _repr(source=_hs.read(1).astype(np.float32),
                      destination=_hand_rs,
                      src_transform=_hs.transform, src_crs=_hs.crs,
                      dst_transform=flood_aff, dst_crs="EPSG:2157",
                      resampling=_RS.bilinear)
            _hand_dem = _hand_rs
        n_before    = final_mask.sum()
        final_mask  = final_mask & (~np.isnan(_hand_dem)) & (_hand_dem <= max_hand_m)
        n_removed   = n_before - final_mask.sum()
        print(f"  HAND mask (≤{max_hand_m} m): removed {n_removed:,} upland "
              f"noise pixels  ({n_removed/(n_before+1)*100:.1f}%)")
    else:
        print(f"  HAND raster not found — upland mask skipped")

    pixel_m     = abs(flood_aff.a)
    n_flood_km2 = final_mask.sum() * pixel_m**2 / 1e6
    print(f"  Final flood mask:      {final_mask.sum():,} pixels  "
          f"({n_flood_km2:.2f} km²)")
    if final_mask.sum() == 0:
        print(f"  ⚠ No flooded pixels detected.")
        print(f"    Try lowering --sar-threshold (e.g. -2.0) or")
        print(f"    --abs-water-db (e.g. -14.0) if the Lee floodplain")
        print(f"    is narrow and only partially inundated on Nov 14.")
    return final_mask


def reproject_mask_to_dem(
    mask:       np.ndarray,
    src_path:   Path,
    dem_path:   Path = DEM_PATH,
) -> np.ndarray:
    """
    Reproject a binary mask from src_path's CRS/transform to the DEM grid.
    Returns boolean array [H, W] aligned with the DEM.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_aff = src.transform

    with rasterio.open(dem_path) as dem_src:
        dst_crs = dem_src.crs
        dst_aff = dem_src.transform
        H, W    = dem_src.height, dem_src.width

    dst_array = np.zeros((H, W), dtype=np.float32)
    reproject(
        source       = mask.astype(np.float32),
        destination  = dst_array,
        src_transform = src_aff,
        src_crs      = src_crs,
        dst_transform = dst_aff,
        dst_crs      = dst_crs,
        resampling   = Resampling.nearest,
    )
    return dst_array > 0.5


def compute_spatial_scores(
    pred: np.ndarray,   # boolean [H, W] — model prediction
    obs:  np.ndarray,   # boolean [H, W] — SAR-derived reference
    valid_mask: np.ndarray | None = None,
) -> dict:
    """
    Compute standard flood mapping skill scores.

    CSI (Critical Success Index / Threat Score):
        TP / (TP + FP + FN)
        Ranges 0–1, 1 = perfect.
        The standard metric for flood map comparison (Wing et al. 2017,
        Fleischmann et al. 2022).

    Hit Rate (Probability of Detection):
        TP / (TP + FN)
        Fraction of observed floods correctly predicted.

    False Alarm Ratio:
        FP / (TP + FP)
        Fraction of predicted floods that are not observed.

    Bias:
        (TP + FP) / (TP + FN)
        > 1 = over-prediction, < 1 = under-prediction.

    F1 score:
        2 * Precision * Recall / (Precision + Recall)
    """
    if valid_mask is not None:
        pred = pred & valid_mask
        obs  = obs  & valid_mask

    TP = int(( pred &  obs).sum())
    FP = int(( pred & ~obs).sum())
    FN = int((~pred &  obs).sum())
    TN = int((~pred & ~obs).sum())

    total = TP + FP + FN + TN
    csi   = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    hr    = TP / (TP + FN)      if (TP + FN) > 0 else 0.0
    far   = FP / (TP + FP)      if (TP + FP) > 0 else 0.0
    bias  = (TP + FP) / (TP + FN) if (TP + FN) > 0 else 0.0
    prec  = TP / (TP + FP)      if (TP + FP) > 0 else 0.0
    f1    = 2*prec*hr / (prec + hr) if (prec + hr) > 0 else 0.0

    pixel_m  = 30.0   # DEM pixel size metres
    area_fac = pixel_m**2 / 1e6

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "CSI":    round(csi,  4),
        "HitRate": round(hr,  4),
        "FAR":    round(far,  4),
        "Bias":   round(bias, 4),
        "F1":     round(f1,   4),
        "pred_area_km2":  round((TP+FP)*area_fac, 3),
        "obs_area_km2":   round((TP+FN)*area_fac, 3),
        "TP_area_km2":    round(TP*area_fac, 3),
        "FP_area_km2":    round(FP*area_fac, 3),
        "FN_area_km2":    round(FN*area_fac, 3),
    }


def save_confusion_map(
    pred:           np.ndarray,
    obs:            np.ndarray,
    out_path:       Path,
    affine,
    crs_str:        str = "EPSG:2157",
    dem:            np.ndarray | None = None,
    timestamp:      str = "",
    scores:         dict | None = None,
    catchment_mask: np.ndarray | None = None,  # boolean [H,W] — limit to study area
    rivers_path:    Path | None = None,
    lakes_path:     Path | None = None,
):
    """
    Save a four-class confusion raster and PNG figure.

    Classes:
        1 = True Positive  (TP) — cyan,   predicted AND observed flood
        2 = False Positive (FP) — red,    predicted flood, not observed
        3 = False Negative (FN) — orange, missed flood
        0 = True Negative  (TN) — transparent
    """
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    from rasterio.crs import CRS

    confusion = np.zeros(pred.shape, dtype=np.uint8)
    confusion[ pred &  obs] = 1   # TP
    confusion[ pred & ~obs] = 2   # FP
    confusion[~pred &  obs] = 3   # FN

    # Save GeoTIFF
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tif_path = out_path.with_suffix(".tif")
    with rasterio.open(
        tif_path, "w", driver="GTiff",
        height=confusion.shape[0], width=confusion.shape[1],
        count=1, dtype="uint8",
        crs=CRS.from_string(crs_str), transform=affine,
    ) as dst:
        dst.write(confusion, 1)
    print(f"  Confusion raster: {tif_path}")

    # Save PNG figure
    fig, ax = plt.subplots(figsize=(13, 7))
    if dem is not None:
        valid = dem.copy(); valid[np.isnan(valid)] = 0
        from matplotlib.colors import LightSource
        ls  = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(valid, cmap=plt.cm.Greys_r,
                       vmin=0, vmax=700, blend_mode="overlay")
        ax.imshow(rgb, origin="upper")

    colours = ["none", "#00D9F2", "#E24B4A", "#F5A623"]
    alpha   = [0,       0.80,      0.75,      0.75]
    for cls, (col, alp) in enumerate(zip(colours, alpha)):
        if cls == 0: continue
        rgba = np.zeros((*confusion.shape, 4), dtype=np.float32)
        import matplotlib.colors as mc
        r, g, b = mc.to_rgb(col)
        mask_cls = confusion == cls
        # Apply catchment mask — only show confusion within study area
        if catchment_mask is not None:
            mask_cls = mask_cls & catchment_mask
        rgba[mask_cls, 0] = r
        rgba[mask_cls, 1] = g
        rgba[mask_cls, 2] = b
        rgba[mask_cls, 3] = alp
        ax.imshow(rgba, origin="upper")

    # Draw lakes and rivers for spatial context
    if lakes_path and Path(lakes_path).exists():
        try:
            import geopandas as gpd
            from shapely.affinity import affine_transform as shp_affine
            gdf = gpd.read_file(lakes_path).to_crs("EPSG:2157")
            a = 1/affine.a; e_ = 1/affine.e
            mx = [a,0,0,e_,-affine.c/affine.a,-affine.f/affine.e]
            for geom in gdf.geometry:
                if geom is None or geom.is_empty: continue
                g2 = shp_affine(geom, mx)
                polys = g2.geoms if g2.geom_type=="MultiPolygon" else [g2]
                for poly in polys:
                    xs,ys = poly.exterior.xy
                    ax.fill(xs,ys,facecolor=(0.18,0.52,0.78,0.45),
                            edgecolor=(0.10,0.35,0.60,0.8),lw=0.5,zorder=3)
        except Exception: pass
    if rivers_path and Path(rivers_path).exists():
        try:
            import geopandas as gpd
            from shapely.affinity import affine_transform as shp_affine
            gdf = gpd.read_file(rivers_path).to_crs("EPSG:2157")
            a = 1/affine.a; e_ = 1/affine.e
            mx = [a,0,0,e_,-affine.c/affine.a,-affine.f/affine.e]
            for geom in gdf.geometry:
                if geom is None or geom.is_empty: continue
                g2 = shp_affine(geom, mx)
                lines = g2.geoms if g2.geom_type=="MultiLineString" else [g2]
                for ln in lines:
                    xs,ys = ln.xy
                    ax.plot(xs,ys,color=(0.08,0.30,0.65),lw=0.7,alpha=0.85,zorder=4)
        except Exception: pass

    legend_patches = [
        mpatches.Patch(facecolor="#00D9F2", label="True Positive (TP)"),
        mpatches.Patch(facecolor="#E24B4A", label="False Positive (FP — over-pred)"),
        mpatches.Patch(facecolor="#F5A623", label="False Negative (FN — missed)"),
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              fontsize=8, framealpha=0.85)

    score_str = ""
    if scores:
        score_str = (f"CSI={scores['CSI']:.3f}  "
                     f"HR={scores['HitRate']:.3f}  "
                     f"FAR={scores['FAR']:.3f}  "
                     f"Bias={scores['Bias']:.2f}")
    ax.set_title(
        f"Flood Map Validation — Predicted vs SAR-Observed\n"
        f"{timestamp}  |  {score_str}",
        fontsize=10,
    )
    ax.set_xlabel("Column (ITM)"); ax.set_ylabel("Row (ITM)")
    fig.tight_layout()
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion map:    {png_path}")


# ═════════════════════════════════════════════════════════════════════
# Tier B — OPW CFRAM plausibility check
# ═════════════════════════════════════════════════════════════════════

def fetch_cfram_extent(aep: str = "1pct") -> "gpd.GeoDataFrame | None":
    """
    Fetch OPW CFRAM flood extent polygon from the floodinfo.ie WFS service.

    aep options: '1pct' (100-year), '0.1pct' (1000-year), '10pct' (10-year)

    Returns a GeoDataFrame in ITM (EPSG:2157) or None if the service
    is unavailable.
    """
    try:
        import geopandas as gpd
        import requests

        # OPW FloodInfo WFS endpoint
        wfs_url = (
            "https://www.floodinfo.ie/arcgis/services/FloodExtents/"
            "MapServer/WFSServer"
        )
        params = {
            "SERVICE":      "WFS",
            "VERSION":      "2.0.0",
            "REQUEST":      "GetFeature",
            "TYPENAMES":    f"FloodExtents:AEP_{aep}",
            "SRSNAME":      "EPSG:2157",
            "BBOX":         "480000,550000,620000,650000,EPSG:2157",
            "outputFormat": "application/json",
        }
        print(f"  Fetching CFRAM {aep} AEP extent from floodinfo.ie ...")
        r = requests.get(wfs_url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  WFS request failed: {r.status_code}")
            return None
        gdf = gpd.read_file(r.text)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:2157")
        elif str(gdf.crs) != "EPSG:2157":
            gdf = gdf.to_crs("EPSG:2157")
        print(f"  CFRAM extent: {len(gdf)} features loaded")
        return gdf
    except Exception as e:
        print(f"  CFRAM fetch failed: {e}")
        print(f"  Download manually from https://www.floodinfo.ie "
              f"and save to dataset/validation/cfram_1pct.shp")
        return None


def check_cfram_plausibility(
    pred:      np.ndarray,
    affine,
    cfram_gdf: "gpd.GeoDataFrame",
    out_dir:   Path,
    label:     str = "",
) -> dict:
    """
    Check what fraction of predicted inundation falls within the
    CFRAM 100-year design flood extent.

    A physically plausible near-real-time prediction for a sub-100-year
    event should lie mostly within the 100-year CFRAM envelope.
    Pixels predicted outside the CFRAM boundary are flagged as suspect.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    H, W = pred.shape
    # Rasterise CFRAM polygons to the DEM grid
    shapes = [(geom, 1) for geom in cfram_gdf.geometry if geom is not None]
    cfram_raster = rasterize(
        shapes, out_shape=(H, W),
        transform=affine, fill=0, dtype=np.uint8,
    ).astype(bool)

    pred_in_cfram  = pred &  cfram_raster
    pred_out_cfram = pred & ~cfram_raster

    pixel_m  = abs(affine.a)
    area_fac = pixel_m**2 / 1e6

    pct_within = (pred_in_cfram.sum() / pred.sum() * 100
                  if pred.sum() > 0 else 0)

    result = {
        "pred_area_km2":         round(pred.sum() * area_fac, 3),
        "pred_within_cfram_km2": round(pred_in_cfram.sum() * area_fac, 3),
        "pred_outside_cfram_km2":round(pred_out_cfram.sum() * area_fac, 3),
        "pct_within_cfram":      round(pct_within, 1),
        "verdict": "plausible" if pct_within >= 80 else "review_required",
    }

    print(f"\n  CFRAM plausibility ({label}):")
    print(f"    Predicted area:     {result['pred_area_km2']:.2f} km²")
    print(f"    Within CFRAM:       {result['pred_within_cfram_km2']:.2f} km² "
          f"({result['pct_within_cfram']:.1f}%)")
    print(f"    Outside CFRAM:      {result['pred_outside_cfram_km2']:.2f} km²")
    print(f"    Verdict:            {result['verdict']}")
    if result["pct_within_cfram"] < 80:
        print(f"    ⚠ More than 20% of predicted inundation lies outside the "
              f"100-year design extent. Check bankfull thresholds and HAND "
              f"raster quality in flagged areas.")
    return result


# ═════════════════════════════════════════════════════════════════════
# Tier C — OPW FFWS alert cross-reference
# ═════════════════════════════════════════════════════════════════════

def validate_ffws_alerts(
    ffws_csv:     Path,
    predictions:  np.ndarray,   # [T_test, N]
    timestamps:   pd.DatetimeIndex,
    node_refs:    list[str],
    bankfull:     np.ndarray,
) -> dict:
    """
    Cross-reference OPW FFWS flood alert records against model predictions.

    The FFWS CSV should have columns:
        timestamp   — ISO datetime of alert activation
        gauge_ref   — OPW gauge reference (e.g. 19114)
        level       — Watch / Warning / Severe Warning

    For each alert, checks whether the model's predicted stage anomaly
    exceeded the bankfull threshold at that gauge within a ±2-hour window.

    Returns precision, recall, F1 and a per-event breakdown table.
    """
    if not ffws_csv.exists():
        print(f"  FFWS CSV not found: {ffws_csv}")
        print(f"  Create a CSV with columns: timestamp, gauge_ref, level")
        print(f"  from waterlevel.ie alert history for the Lee catchment.")
        return {}

    alerts = pd.read_csv(ffws_csv, parse_dates=["timestamp"])
    print(f"  FFWS alerts loaded: {len(alerts)}")

    ref_to_idx = {str(r): i for i, r in enumerate(node_refs)}
    window_steps = 8   # ±2 hours at 15-min timesteps

    rows = []
    TP = FP = FN = TN_events = 0

    for _, alert in alerts.iterrows():
        ts_alert = pd.Timestamp(alert["timestamp"])
        ref      = str(alert["gauge_ref"])
        if ref not in ref_to_idx:
            continue
        node_i = ref_to_idx[ref]

        # Find test timestep nearest to alert
        idx = int(np.argmin(np.abs(timestamps - ts_alert)))
        start = max(0, idx - window_steps)
        end   = min(len(timestamps), idx + window_steps + 1)

        # Model predicted exceedance in window?
        pred_in_window = predictions[start:end, node_i]
        thr            = float(bankfull[node_i])
        model_flag     = bool((pred_in_window > thr).any())

        status = "TP" if model_flag else "FN"
        if model_flag: TP += 1
        else:          FN += 1

        rows.append({
            "alert_time":  ts_alert,
            "gauge_ref":   ref,
            "level":       alert.get("level", "Watch"),
            "model_flag":  model_flag,
            "status":      status,
            "max_pred_stage": round(float(pred_in_window.max()), 3),
            "bankfull_thr":   round(thr, 3),
        })

    # False positives: model flags exceedance at nodes with no FFWS alert
    alert_refs_times = set(
        zip(alerts["gauge_ref"].astype(str),
            pd.to_datetime(alerts["timestamp"]).dt.date))

    for t_idx, ts in enumerate(timestamps):
        for node_i, ref in enumerate(node_refs):
            thr = float(bankfull[node_i])
            if predictions[t_idx, node_i] > thr:
                if (ref, ts.date()) not in alert_refs_times:
                    FP += 1

    prec  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec   = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

    result = {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
        "events":    rows,
    }

    print(f"\n  FFWS alert validation:")
    print(f"    Alerts in test period: {len(rows)}")
    print(f"    TP (correctly flagged): {TP}")
    print(f"    FN (missed events):     {FN}")
    print(f"    FP (false alarms):      {FP}")
    print(f"    Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
    return result


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def run_sar(args):
    """Run Tier A: SAR spatial validation."""
    print(f"\n{'═'*55}\n  Tier A — SAR Spatial Validation\n{'═'*55}")
    # Sanitise event_date for Windows filenames (colons are illegal)
    event_date_safe = args.event_date.replace(":", "h").replace(" ", "_")
    for model_name in (
        [args.model] if args.model else list(MODEL_LABELS.keys())
    ):
        print(f"\n  Model: {MODEL_LABELS.get(model_name, model_name)}")
        result = load_predictions_for_event(model_name, args.event_date)
        if result is None: continue
        pred_stage, matched_ts = result
        bf_override = None
        if getattr(args, "bankfull_override", None):
            import pandas as _pd
            n_nodes = len(_pd.read_csv(GRAPH_DIR / "nodes.csv"))
            bf_override = np.full(n_nodes, float(args.bankfull_override),
                                  dtype=np.float32)
            print(f"  Bankfull override: {args.bankfull_override} m (all nodes)")
        hft = getattr(args, "hand_flood_threshold", None)
        pred_inn, affine, (H, W) = build_hand_inundation(
            pred_stage, bankfull=bf_override, hand_flood_threshold=hft)

        sar_mask = compute_sar_flood_mask(
            Path(args.sar_flood), Path(args.sar_ref),
            threshold_db = args.sar_threshold,
            abs_water_db = args.abs_water_db,
            max_hand_m   = getattr(args, "max_hand_m", 5.0),
        )
        if sar_mask is None: continue

        if getattr(args, "calibrate_hand", False):
            obs_cal = reproject_mask_to_dem(sar_mask, Path(args.sar_flood))
            bf_cal  = (bf_override if bf_override is not None
                       else np.full(len(pred_stage), 0.1, dtype=np.float32))
            cal = calibrate_hand_threshold(obs_cal, pred_stage, bf_cal, affine)
            cal_out = (FIG_DIR / model_name /
                       f"hand_calibration_{event_date_safe}.json")
            cal_out.parent.mkdir(parents=True, exist_ok=True)
            import json as _jcal
            with open(cal_out, "w") as _f:
                _jcal.dump(cal, _f, indent=2, default=str)
            print(f"  Saved: {cal_out}")

        obs_inn = reproject_mask_to_dem(
            sar_mask, Path(args.sar_flood))

        scores = compute_spatial_scores(pred_inn, obs_inn)
        print(f"\n  Spatial skill scores ({model_name}):")
        print(f"    CSI  (Critical Success Index): {scores['CSI']:.4f}")
        print(f"    Hit Rate (Recall):             {scores['HitRate']:.4f}")
        print(f"    False Alarm Ratio:             {scores['FAR']:.4f}")
        print(f"    Bias:                          {scores['Bias']:.4f}")
        print(f"    F1 score:                      {scores['F1']:.4f}")
        print(f"    Predicted area: {scores['pred_area_km2']:.2f} km²")
        print(f"    Observed area:  {scores['obs_area_km2']:.2f} km²")

        # Load DEM for background
        import rasterio
        with rasterio.open(DEM_PATH) as s:
            dem = s.read(1).astype(np.float32)
        dem[dem == (s.nodata or -9999.0)] = np.nan

        out_dir = FIG_DIR / model_name
        catchment_mask = ~np.isnan(dem)
        if Path(CATCHMENT_SHAPEFILE).exists():
            try:
                import geopandas as _gpd
                from rasterio.features import rasterize as _rasterize
                gdf_c  = _gpd.read_file(CATCHMENT_SHAPEFILE).to_crs("EPSG:2157")
                shp_c  = [(g, 1) for g in gdf_c.geometry if g is not None]
                c_rst  = _rasterize(shp_c, out_shape=(H, W),
                                    transform=affine, fill=0, dtype=np.uint8)
                catchment_mask = (c_rst == 1) & ~np.isnan(dem)
                print(f"  Catchment mask: loaded from {Path(CATCHMENT_SHAPEFILE).name}")
            except Exception as _e:
                print(f"  Catchment shapefile failed ({_e}) — using DEM extent")

        # rivers_p = Path(args.sar_rivers) if getattr(args,"sar_rivers",None) else None
        # lakes_p  = Path(args.sar_lakes)  if getattr(args,"sar_lakes", None) else None

        rivers_p = Path(RIVER_SHAPEFILE)
        lakes_p = Path(LAKES_SHAPEFILE)

        save_confusion_map(
            pred_inn, obs_inn,
            out_dir / f"confusion_{event_date_safe}",
            affine, dem=dem,
            timestamp=f"{matched_ts}  [{MODEL_LABELS.get(model_name,'?')}]",
            scores=scores,
            catchment_mask=catchment_mask,
            rivers_path=rivers_p,
            lakes_path=lakes_p,
        )

        # Save scores JSON
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_out = out_dir / f"sar_scores_{event_date_safe}.json"
        with open(scores_out, "w") as f:
            json.dump({"model": model_name, "event": args.event_date,
                       "timestamp": str(matched_ts), **scores}, f, indent=2)
        print(f"  Saved: {scores_out}")


def run_cfram(args):
    """Run Tier B: CFRAM plausibility check."""
    print(f"\n{'═'*55}\n  Tier B — CFRAM Plausibility Check\n{'═'*55}")

    cfram_local = VAL_DIR / "cfram_1pct.shp"
    if cfram_local.exists():
        import geopandas as gpd
        cfram = gpd.read_file(cfram_local).to_crs("EPSG:2157")
        print(f"  CFRAM loaded from local file: {cfram_local}")
    else:
        cfram = fetch_cfram_extent("1pct")

    if cfram is None: return

    result = load_predictions_for_event(
        args.model or "st_gnn_hand_edge",
        args.event_date or "2025-01-12",
    )
    if result is None: return
    pred_stage, matched_ts = result
    pred_inn, affine, _ = build_hand_inundation(pred_stage)
    check_cfram_plausibility(
        pred_inn, affine, cfram, FIG_DIR,
        label=f"{args.model or 'st_gnn_hand_edge'}  {matched_ts.date()}"
    )


def run_ffws(args):
    """Run Tier C: FFWS alert cross-reference."""
    print(f"\n{'═'*55}\n  Tier C — FFWS Alert Cross-Reference\n{'═'*55}")

    model_name = args.model or "st_gnn_hand_edge"
    ckpt_dir   = CKPT_ROOT / model_name
    mean_files = sorted(ckpt_dir.glob("test_predictions_4steps_mean.npy"))
    if not mean_files:
        print(f"  No predictions found for {model_name}"); return

    pred_all   = np.load(mean_files[0])
    y          = np.load(PROC_DIR / "y.npy", mmap_mode="r")
    T          = y.shape[0]
    ts_s, ts_e = int(T*0.85), T - 4
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            pd.read_csv(PROC_DIR/"timestamps.csv")["timestamp"]
        ).iloc[ts_s:ts_e].values
    )
    nodes_df   = pd.read_csv(GRAPH_DIR / "nodes.csv")
    node_refs  = nodes_df["ref"].astype(str).tolist()

    bf_json    = GRAPH_DIR / "bankfull_thresholds.json"
    if bf_json.exists():
        thr_map  = json.load(open(bf_json))["thresholds"]
        bankfull = np.array([float(thr_map.get(r, 0.1)) for r in node_refs],
                             dtype=np.float32)
    else:
        bankfull = np.full(len(node_refs), 0.3, dtype=np.float32)

    result = validate_ffws_alerts(
        Path(args.ffws_csv), pred_all, timestamps, node_refs, bankfull
    )
    if result:
        out = FIG_DIR / model_name / "ffws_validation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Validate ST-GNN flood maps against SAR, CFRAM, and FFWS"
    )
    p.add_argument("--mode", choices=["sar","cfram","ffws","all"],
                   default="all")
    p.add_argument("--model",       type=str, default=None)
    p.add_argument("--event-date",  type=str, default="2025-01-12",
                   help="ISO date of flood event, e.g. 2025-01-12")
    p.add_argument("--sar-flood",   type=str, default=None,
                   help="GeoTIFF: Sentinel-1 VV backscatter during flood (dB)")
    p.add_argument("--sar-ref",     type=str, default=None,
                   help="GeoTIFF: pre-flood reference backscatter (dB)")
    p.add_argument("--sar-threshold", type=float, default=-3.0,
                   help="Change detection threshold in dB (default -3.0)")
    p.add_argument("--abs-water-db",  type=float, default=-15.0,
                   help="Absolute open water threshold in dB (default -15.0)")
    p.add_argument("--max-hand-m",     type=float, default=5.0,
                   help="Remove SAR detections where HAND > this value (m). "
                        "Filters upland noise. Default 5.0 m.")
    p.add_argument("--hand-flood-threshold", type=float, default=None,
                   help="Fixed HAND (m) for inundation extent. Use --calibrate-hand first.")
    p.add_argument("--calibrate-hand",       action="store_true",
                   help="Sweep HAND thresholds 0.5–10 m, report best CSI.")
    # p.add_argument("--sar-rivers",       type=str,   default=None,
    #                help="River network shapefile for confusion map overlay")
    # p.add_argument("--sar-lakes",        type=str,   default=None,
    #                help="Lakes shapefile for confusion map overlay")
    # p.add_argument("--catchment-shp",    type=str,   default=None,
    #                help="Lee catchment boundary shapefile")
    p.add_argument("--bankfull-override", type=float, default=None,
                   help="Override bankfull threshold (m) for all nodes — validation only")
    p.add_argument("--ffws-csv",    type=str,
                   default=str(VAL_DIR / "opw_ffws_alerts.csv"))
    args = p.parse_args()

    if args.mode in ("sar", "all") and args.sar_flood and args.sar_ref:
        run_sar(args)
    if args.mode in ("cfram", "all"):
        run_cfram(args)
    if args.mode in ("ffws", "all"):
        run_ffws(args)
