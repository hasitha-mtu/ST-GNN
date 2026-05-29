"""
build_dataset.py  v1  –  Dynamic feature tensor construction for ST-GNN
=======================================================================
Reads the static graph produced by graph_builder.py (nodes.csv / edges.csv)
and the raw OPW time-series CSVs to build:

  X.npy               float32  [T, N, F]   dynamic node features
  y.npy               float32  [T, N]      target  (next-step stage_anomaly)
  timestamps.npy      datetime64[ns] [T]   UTC-aligned time index
  dataset_metadata.json                    provenance + coverage stats

Dynamic node features (F = 5, in column order):
  [0] stage_anomaly     water_level − rolling_7d_mean       (elevation-free)
  [1] normalized_stage  (level − gauge_datum) / (p90 − gauge_datum)
  [2] dh_dt             rate of change  m / hr
  [3] discharge_m3s     log1p-transformed; 0.0 for level-only / tidal nodes
  [4] rainfall_mm       from nearest OPW raingauge (Thiessen / nearest-neighbour)

Raw file naming conventions expected under dataset/raw/:
  water_level/   wl_{ref}.csv    columns: datetime, value
  discharge/     discharge_{ref}.csv      columns: datetime, value
  rainfall/      rain_{ref}.csv       columns: datetime, value

All raw CSVs are assumed to be at OPW native 15-min resolution.
Gaps ≤ 90 min (6 steps) are forward-filled; longer gaps are zero-filled
and flagged in dataset_metadata.json coverage stats.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths  (adjust to your local layout) ──────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
GRAPH_DIR     = BASE_DIR / "dataset/graph"           # nodes.csv, edges.csv
WL_DIR        = BASE_DIR / "dataset/raw/water_level" # water_level_{ref}.csv
DISCHARGE_DIR = BASE_DIR / "dataset/raw/discharge"   # discharge_{ref}.csv
RAINFALL_DIR  = BASE_DIR / "dataset/raw/rainfall"    # rainfall_{ref}.csv
RAIN_META     = BASE_DIR / "dataset/metadata/rainfall_stations.csv"
OUT_DIR       = BASE_DIR / "dataset/processed"
ERA5_DIR      = BASE_DIR / "dataset/era5"
ERA5_SM_FILE  = ERA5_DIR / "era5_land_sm_lee.nc"

# ── Soil moisture flag ────────────────────────────────────────────────────
# Set True once ERA5-Land data is downloaded. False = F=5 (gauge only).
USE_SOIL_MOISTURE = True
SM_LAYERS         = ["swvl1", "swvl2"]

# ── Configuration ──────────────────────────────────────────────────────────
TIMESTEP       = "15min"    # OPW native resolution
ROLLING_WINDOW = "7D"       # baseline window for stage_anomaly
GAP_FILL_STEPS = 6          # forward-fill limit (6 × 15 min = 90 min)
START_DATE     = "2023-01-01"   # ← was 2010-01-01
END_DATE       = "2026-03-25"   # ← extend to yesterday

# Dynamic feature names — must match X tensor column order exactly
GAUGE_FEATURES = [
    "stage_anomaly",       # [0]
    "normalized_stage",    # [1]
    "dh_dt",               # [2]
    "discharge_m3s",       # [3]
    "rainfall_mm",         # [4]
]
SM_FEATURES = [
    "swvl1_raw",           # [5]
    "swvl1_sat_ratio",     # [6]
    "swvl1_anomaly",       # [7]
    "swvl2_raw",           # [8]
    "swvl2_sat_ratio",     # [9]
    "swvl2_anomaly",       # [10]
]
DYNAMIC_FEATURES = GAUGE_FEATURES + (SM_FEATURES if USE_SOIL_MOISTURE else [])

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.data.soil_moisture_features import (compute_sm_features,
                                             build_node_sm_lookup,
                                             extract_node_sm_features,
                                             append_sm_to_node_features,)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _find_value_column(df: pd.DataFrame, hints: list[str]) -> str:
    """Return first column whose name contains any of the hint substrings (case-insensitive)."""
    for col in df.columns:
        for hint in hints:
            if hint in col.lower():
                return col
    # Fall back to the first non-datetime column
    return df.columns[0]


def _load_csv_series(path: Path, value_hints: list[str], resample_how: str, freq: str) -> pd.Series | None:
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=True)

    # Identify datetime column
    dt_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if not dt_candidates:
        logger.warning("  No datetime column found in %s — skipping", path.name)
        return None

    df[dt_candidates[0]] = pd.to_datetime(df[dt_candidates[0]], utc=True,    # ← utc=True handles +00:00 suffix
                                           dayfirst=False, errors="coerce")
    df[dt_candidates[0]] = df[dt_candidates[0]].dt.tz_localize(None)         # ← strip tz → naive UTC
    df = df.dropna(subset=[dt_candidates[0]])
    df = df.set_index(dt_candidates[0])
    df.index.name = "datetime"

    val_col = _find_value_column(df, value_hints)
    # ── DEBUG ─────────────────────────────────────────────────────────────
    logger.info("    DEBUG %s: total_rows=%d  val_col=%s  notna_before_filter=%d",
                path.name, len(df), val_col, df[val_col].notna().sum())
    # ── Quality filter ───────────────────────────────────────────────────
    BAD_QUALITY_CODES = {-1, 32, 101}

    if "quality_code" in df.columns:
        bad_mask = pd.to_numeric(df["quality_code"], errors="coerce").isin(BAD_QUALITY_CODES)
        df.loc[bad_mask, val_col] = np.nan
        logger.info("    DEBUG %s: masked=%d  notna_after_filter=%d",
                    path.name, bad_mask.sum(), df[val_col].notna().sum())

    # ── Check if quality_ok filter is ALSO still active (likely culprit) ──
    if "quality_ok" in df.columns:
        logger.warning("    DEBUG %s: quality_ok column present — is it still being filtered?",
                       path.name)

    s = pd.to_numeric(df[val_col], errors="coerce")
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]

    if "level" in " ".join(value_hints):
        s = s.where(s >= 0)

    if resample_how == "mean":
        return s.resample(freq).mean()
    elif resample_how == "sum":
        return s.resample(freq).sum(min_count=1)
    else:
        raise ValueError(f"Unknown resample_how: {resample_how}")


# ── Individual loaders ─────────────────────────────────────────────────────

def load_water_level(ref: str, freq: str = TIMESTEP) -> pd.Series | None:
    path = WL_DIR / f"wl_{ref}.csv"
    s = _load_csv_series(path, ["value", "level", "stage", "wl"], "mean", freq)
    if s is None:
        logger.warning("  No water level file for ref %s", ref)
        return None

    # ── DEBUG: remove after diagnosis ────────────────────────────────
    window = s["2023-01-01":"2026-03-25"]
    total = len(pd.date_range("2023-01-01", "2026-03-25", freq=freq))
    logger.info("  DEBUG ref=%s  notna=%d / %d  (%.1f%%)",
                ref, window.notna().sum(), total,
                window.notna().sum() / total * 100)
    # ─────────────────────────────────────────────────────────────────
    return s


def load_rainfall(ref: str, freq: str = TIMESTEP) -> pd.Series | None:
    path = RAINFALL_DIR / f"rain_{ref}.csv"   # ← fix prefix
    return _load_csv_series(path, ["value", "rain", "precip", "rr"], "sum", freq)


def load_discharge(ref: str, freq: str = TIMESTEP) -> pd.Series | None:
    path = DISCHARGE_DIR / f"discharge_{ref}.csv"
    return _load_csv_series(path, ["value", "discharge", "flow", "q"], "mean", freq)


# ═══════════════════════════════════════════════════════════════════════════
# Rainfall → node assignment  (nearest-neighbour)
# ═══════════════════════════════════════════════════════════════════════════

def assign_rainfall_to_nodes(nodes_df: pd.DataFrame) -> dict[str, str | None]:
    if not RAIN_META.exists():
        logger.warning("rainfall_stations.csv not found — rainfall will be 0 for all nodes")
        return {str(r): None for r in nodes_df["ref"]}

    if "lat" not in nodes_df.columns or "lon" not in nodes_df.columns:
        logger.error("nodes.csv is missing lat/lon columns — re-run graph_builder.py first")
        return {str(r): None for r in nodes_df["ref"]}

    rain_meta = pd.read_csv(RAIN_META)
    assignment = {}

    for _, row in nodes_df.iterrows():
        wl_ref = str(row["ref"])

        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            logger.warning("  Node %s has no coordinates — skipping rainfall assignment", wl_ref)
            assignment[wl_ref] = None
            continue

        dists = rain_meta.apply(
            lambda r: haversine_km(row["lat"], row["lon"], r["lat"], r["lon"]), axis=1
        )
        nearest_idx = dists.idxmin()
        rain_ref    = str(rain_meta.loc[nearest_idx, "ref"])
        dist_km     = dists[nearest_idx]
        assignment[wl_ref] = rain_ref
        logger.info("    Node %s → rain gauge %s (%.1f km)", wl_ref, rain_ref, dist_km)

    return assignment


# ═══════════════════════════════════════════════════════════════════════════
# Per-node dynamic feature computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_node_features(
    wl:           pd.Series,
    gauge_datum:  float,
    p90:          float,
    discharge:    pd.Series | None,
    rainfall:     pd.Series | None,
    common_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute all 5 dynamic features for one node aligned to common_index.

    Parameters
    ----------
    wl           Raw water level time series (mAOD, OSGM15).
    gauge_datum  Station datum from nodes.csv  (gauge_datum_mOSGM15).
    p90          90th-percentile flood threshold from nodes.csv (p90_mAOD).
    discharge    Discharge series or None for level-only / tidal nodes.
    rainfall     Rainfall series from nearest gauge or None.
    common_index Shared DatetimeIndex for the full dataset period.
    """
    # Align water level to common time axis before computing rolling stats
    # so the rolling window sees real calendar time, not just available points
    wl = wl.reindex(common_index)

    # ── [0] stage_anomaly ────────────────────────────────────────────────
    # Deviation from rolling 7-day mean — removes elevation offset so that
    # cross-station correlations reflect hydrology, not gauge position.
    # min_periods=1 avoids NaN at the very start of the record.
    rolling_mean  = wl.rolling(ROLLING_WINDOW, min_periods=1).mean()
    stage_anomaly = wl - rolling_mean

    # ── [1] normalized_stage ─────────────────────────────────────────────
    # Position within [gauge_datum, p90] range.
    # Tidal / reservoir stations may have p90 ≈ gauge_datum — protect denom.
    denom = p90 - gauge_datum if (p90 - gauge_datum) > 0.01 else 1.0
    normalized_stage = (wl - gauge_datum) / denom
    normalized_stage = normalized_stage.clip(-2.0, 5.0)  # cap sensor noise

    # ── [2] dh_dt ────────────────────────────────────────────────────────
    # Rate of change in m/hr.  15-min diff × 4 → hourly rate.
    dh_dt = wl.diff() * 4.0   # m / hr

    # ── [3] discharge ────────────────────────────────────────────────────
    # log1p-transform reduces the heavy right skew typical of streamflow.
    # Zero for nodes without a flow gauge (level-only, tidal).
    if discharge is not None:
        q = discharge.reindex(common_index).fillna(0.0)
        q_feat = np.sign(q) * np.log1p(np.abs(q))
        q_feat = pd.Series(q_feat.values, index=common_index)
    else:
        q_feat = pd.Series(0.0, index=common_index)

    # ── [4] rainfall ─────────────────────────────────────────────────────
    if rainfall is not None:
        r = rainfall.reindex(common_index).fillna(0.0)
        r = r.clip(lower=0.0)       # negative rainfall = bad sensor reading
    else:
        r = pd.Series(0.0, index=common_index)

    feat_df = pd.DataFrame({
        "stage_anomaly":    stage_anomaly,
        "normalized_stage": normalized_stage,
        "dh_dt":            dh_dt.fillna(0.0),
        "discharge_m3s":    q_feat,
        "rainfall_mm":      r,
    }, index=common_index)

    return feat_df

# ── Save CSV helper ────────────────────────────────────────────────────────
def save_dataset_csv(
    X:           np.ndarray,
    y:           np.ndarray,
    valid_mask:  np.ndarray,        # ← add parameter
    timestamps:  pd.DatetimeIndex,
    node_refs:   list[str],
    out_dir:     Path,
) -> None:
    records = []

    for i, ref in enumerate(node_refs):
        df = pd.DataFrame(X[:, i, :], columns=DYNAMIC_FEATURES, index=timestamps)
        df.index.name = "timestamp"
        df["target_y"]   = y[:, i]
        df["valid"]      = valid_mask[:, i].astype(int)   # ← 1=real, 0=imputed
        df["node_ref"]   = ref
        df = df.reset_index()[
            ["timestamp", "node_ref"] + DYNAMIC_FEATURES + ["target_y", "valid"]
        ]
        records.append(df)

    long_df = pd.concat(records, ignore_index=True)
    long_df["timestamp"] = long_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out_path = out_dir / "dataset.csv"
    long_df.to_csv(out_path, index=False, float_format="%.6f")
    logger.info("  Saved dataset.csv  (%d rows × %d cols) → %s",
                len(long_df), len(long_df.columns), out_dir)

# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def build_dataset(
    save: bool = True,
    start_date: str = START_DATE,
    end_date:   str = END_DATE,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Build the full [T, N, F] dynamic feature tensor.

    Returns
    -------
    X           float32  [T, N, F]
    y           float32  [T, N]     next-step stage_anomaly target
    timestamps  DatetimeIndex [T]
    node_refs   list[str]  length N, same order as nodes.csv
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load static graph ──────────────────────────────────────────────
    logger.info("Loading static graph …")
    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    N = len(nodes_df)
    logger.info("  %d nodes", N)

    # ── 2. Common time index ──────────────────────────────────────────────
    logger.info("Building time index  %s → %s @ %s …", start_date, end_date, TIMESTEP)
    common_index = pd.date_range(start=start_date, end=end_date, freq=TIMESTEP)
    T = len(common_index)
    logger.info("  %d timesteps  (%.1f years)", T, T * 15 / (60 * 24 * 365.25))

    # ── 3. Assign rainfall gauges to nodes ────────────────────────────────
    logger.info("Assigning rainfall gauges to nodes …")
    rain_assignment = assign_rainfall_to_nodes(nodes_df)

    # Cache so each unique rain gauge is loaded only once
    rainfall_cache: dict[str, pd.Series | None] = {}

    # ── 4. Build X [T, N, F] ─────────────────────────────────────────────
    F_gauge = len(GAUGE_FEATURES)
    F       = len(DYNAMIC_FEATURES)
    X = np.full((T, N, F_gauge), np.nan, dtype=np.float32)
    coverage_stats: list[dict] = []

    for i, row in nodes_df.iterrows():
        ref  = str(row["ref"])
        name = row.get("name", ref)
        logger.info("  [%d/%d]  %s  (%s)", i + 1, N, name, ref)

        # ── Water level ───────────────────────────────────────────────────
        wl = load_water_level(ref)
        if wl is None:
            logger.warning("    ✗ No water level data — node left as NaN")
            coverage_stats.append({"ref": ref, "name": name, "wl_coverage_pct": 0.0})
            continue

        wl = wl.loc[start_date:end_date]
        wl_coverage = float(wl.notna().sum()) / T

        # ── Discharge ─────────────────────────────────────────────────────
        discharge = load_discharge(ref)

        # ── Rainfall  (cached by rain_ref) ────────────────────────────────
        rain_ref = rain_assignment.get(ref)
        if rain_ref and rain_ref not in rainfall_cache:
            rainfall_cache[rain_ref] = load_rainfall(rain_ref)
        rainfall = rainfall_cache.get(rain_ref) if rain_ref else None

        # ── Gauge datum + p90 from static graph ───────────────────────────
        gauge_datum = float(row.get("gauge_datum_mOSGM15", 0.0))
        p90         = float(row.get("p90_mAOD", gauge_datum + 1.0))

        # ── Compute features ──────────────────────────────────────────────
        feat_df = compute_node_features(
            wl=wl,
            gauge_datum=gauge_datum,
            p90=p90,
            discharge=discharge,
            rainfall=rainfall,
            common_index=common_index,
        )

        X[:, i, :] = feat_df.values.astype(np.float32)

        logger.info("    WL coverage: %.1f%%  |  discharge: %s  |  rainfall: %s",
                    wl_coverage * 100,
                    "yes" if discharge is not None else "no",
                    f"gauge {rain_ref}" if rain_ref else "none")

        coverage_stats.append({
            "ref":              ref,
            "name":             name,
            "wl_coverage_pct":  round(wl_coverage * 100, 1),
            "has_discharge":    discharge is not None,
            "rain_gauge_ref":   rain_ref,
        })

# ── 5a. Soil moisture features ─────────────────────────────────────────
    if USE_SOIL_MOISTURE:
        logger.info("Computing ERA5-Land soil moisture features ...")
        ds_sm = compute_sm_features(str(ERA5_SM_FILE), layers=SM_LAYERS)
        try:
            if not ERA5_SM_FILE.exists():
                msg = "ERA5-Land file not found: " + str(ERA5_SM_FILE)
                msg += " -- Run download_era5_land_sm() or set USE_SOIL_MOISTURE=False."
                raise FileNotFoundError(msg)
            ds_sm = compute_sm_features(str(ERA5_SM_FILE), layers=SM_LAYERS)
            logger.info("  ERA5-Land features: %s", list(ds_sm.data_vars))
            if "lat" not in nodes_df.columns or "lon" not in nodes_df.columns:
                raise KeyError("nodes.csv needs lat/lon for ERA5 spatial matching.")
            node_lookup = build_node_sm_lookup(
                ds_sm,
                node_lons=nodes_df["lon"].values,
                node_lats=nodes_df["lat"].values,
            )
            sm_matrix, _ = extract_node_sm_features(
                ds_sm, node_lookup,
                target_timestamps=common_index,
                layers=SM_LAYERS,
            )
            logger.info("  SM matrix: %s  NaN: %.4f",
                        sm_matrix.shape, float(np.isnan(sm_matrix).mean()))
            X = append_sm_to_node_features(X, sm_matrix)
            logger.info("  Features: %d gauge + %d SM = %d total",
                        F_gauge, sm_matrix.shape[2], X.shape[2])
        except Exception as exc:
            logger.error(
                "Soil moisture integration FAILED: %s. "
                "Falling back to gauge-only (F=5). "
                "Set USE_SOIL_MOISTURE=False to suppress.",
                exc,
            )

    # ── 5b. Gap imputation (all features) ──────────────────────────────────
    logger.info("Imputing missing values (ffill <= %d steps, then zero-fill) ...", GAP_FILL_STEPS)
    F_actual = X.shape[2]
    for i in range(N):
        for f in range(F_actual):
            s = pd.Series(X[:, i, f])
            s = s.ffill(limit=GAP_FILL_STEPS)
            s = s.fillna(0.0)
            X[:, i, f] = s.values

    # ── 6. Build target y ────────────────────────────────────────────────
    # y[t] = stage_anomaly[t+1]  — 1-step ahead (15 min horizon)
    # Adjust HORIZON here if you want multi-step targets.
    HORIZON = 1
    y = np.roll(X[:, :, 0], shift=-HORIZON, axis=0).astype(np.float32)
    y[-HORIZON:, :] = np.nan  # last HORIZON steps have no future

    # ── 7. Summary stats ─────────────────────────────────────────────────
    F_final = X.shape[2]
    nan_pct = np.isnan(y).mean() * 100
    logger.info("\n✓ Dataset built:")
    logger.info("  X shape:       %s  (T=%d, N=%d, F=%d)", X.shape, T, N, F_final)
    logger.info("  y shape:       %s", y.shape)
    logger.info("  y NaN (last step masked): %.1f%%", nan_pct)
    logger.info("  Features [%d]: %s", F_final, DYNAMIC_FEATURES)

    # ── 8. Save ───────────────────────────────────────────────────────────
    node_refs = nodes_df["ref"].astype(str).tolist()

    valid_mask = np.zeros((T, N), dtype=np.float32)

    for i, row in nodes_df.iterrows():
        ref = str(row["ref"])
        wl = load_water_level(ref)
        if wl is not None:
            wl_aligned = wl.reindex(common_index)
            valid_mask[:, i] = wl_aligned.notna().astype(np.float32)

    if save:
        np.save(OUT_DIR / "X.npy",          X)
        np.save(OUT_DIR / "y.npy",          y)
        np.save(OUT_DIR / "valid_mask.npy", valid_mask)
        np.save(OUT_DIR / "timestamps.npy", np.array(common_index, dtype="datetime64[ns]"))

        # ── timestamps.csv ────────────────────────────────────────
        # Required by train_st_gnn_flood_model_sar.py to build the
        # SAR event lookup table. Column name must be "timestamp".
        pd.DataFrame({"timestamp": common_index.strftime("%Y-%m-%d %H:%M:%S")}).to_csv(
            OUT_DIR / "timestamps.csv", index=False
        )

        # ── dataset.csv  (long-format, human-readable) ────────────
        # Combines X features, y target, and valid_mask into one file
        # for inspection, plotting, and debugging. Not used by the
        # training script — use the .npy files for model training.
        save_dataset_csv(X, y, valid_mask, common_index, node_refs, OUT_DIR)

        meta = {"start_date": str(common_index[0]), "end_date": str(common_index[-1]), "timestep": TIMESTEP,
                "n_timesteps": int(T), "n_nodes": int(N), "n_features": int(X.shape[2]), "horizon_steps": HORIZON,
                "soil_moisture_active": USE_SOIL_MOISTURE and X.shape[2] > len(GAUGE_FEATURES),
                "sm_layers": SM_LAYERS if USE_SOIL_MOISTURE else [],
                "rolling_window": ROLLING_WINDOW, "gap_fill_steps": GAP_FILL_STEPS,
                "dynamic_features": DYNAMIC_FEATURES, "node_refs": node_refs, "coverage": coverage_stats,
                "rainfall_assignment_method": "nearest_neighbour", "rainfall_note": (
                "Thiessen polygon weighting not implemented — only 4 rain gauges "
                "across the catchment produce near-identical assignments to nearest-neighbour. "
                "True per-station Thiessen requires sub-catchment delineations per OPW station, "
                "available from OPW hydrology portal or EPA national catchment dataset."
            )}
        with open(OUT_DIR / "dataset_metadata.json", "w") as fp:
            json.dump(meta, fp, indent=2, default=str)

        logger.info(
            "  Saved to %s:\n"
            "    X.npy             [T=%d, N=%d, F=%d] float32\n"
            "    y.npy             [T=%d, N=%d] float32\n"
            "    valid_mask.npy    [T=%d, N=%d] float32\n"
            "    timestamps.npy    [T=%d] datetime64\n"
            "    timestamps.csv    [T=%d rows] — used by training script\n"
            "    dataset.csv       [T×N=%d rows] — long-format for inspection\n"
            "    dataset_metadata.json",
            OUT_DIR, T, N, F, T, N, T, N, T, T, T * N,
        )

    return X, y, common_index, node_refs


# ── Quick inspection helper ────────────────────────────────────────────────
def inspect_dataset(processed_dir: Path = OUT_DIR) -> None:
    """
    Print a human-readable summary of a saved dataset.
    Call this after build_dataset() to sanity-check the outputs.
    """
    X          = np.load(processed_dir / "X.npy")
    y          = np.load(processed_dir / "y.npy")

    with open(processed_dir / "dataset_metadata.json") as f:
        meta = json.load(f)

    print(f"\n{'─'*55}")
    print(f"  Dataset summary  ({processed_dir})")
    print(f"{'─'*55}")
    print(f"  Period  : {meta['start_date']}  →  {meta['end_date']}")
    print(f"  Timestep: {meta['timestep']}")
    print(f"  X shape : {X.shape}   (T × N × F)")
    print(f"  y shape : {y.shape}   (T × N)")
    print(f"\n  Dynamic features:")
    for j, name in enumerate(meta["dynamic_features"]):
        col      = X[:, :, j].ravel()
        col_vals = col[~np.isnan(col)]
        print(f"    [{j}] {name:<22}  "
              f"mean={col_vals.mean():+.4f}  "
              f"std={col_vals.std():.4f}  "
              f"[{col_vals.min():.3f}, {col_vals.max():.3f}]")

    print(f"\n  Node coverage:")
    for c in meta["coverage"]:
        flag = "✓" if c["wl_coverage_pct"] >= 70 else "✗"
        print(f"    {flag} {c['ref']}  {c.get('name',''):<30}  "
              f"WL={c['wl_coverage_pct']:5.1f}%  "
              f"Q={'yes' if c.get('has_discharge') else 'no ':3}  "
              f"rain={c.get('rain_gauge_ref','–')}")
    print(f"{'─'*55}\n")

def diagnose(ref):
    BAD_QUALITY_CODES = {-1, 32, 101}
    df = pd.read_csv(WL_DIR / f"wl_{ref}.csv")
    print(f"\n── Station {ref} ──────────────────────")
    print(f"  Total rows:       {len(df)}")
    print(f"  Columns:          {df.columns.tolist()}")
    print(f"\n  Quality codes:")
    print(df["quality_code"].value_counts().to_string())

    # Datetime parsing
    df["dt"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    n_failed = df["dt"].isna().sum()
    print(f"\n  Datetime parse failures: {n_failed}")

    # After quality masking
    bad = pd.to_numeric(df["quality_code"], errors="coerce").isin(BAD_QUALITY_CODES)
    good = df[~bad].dropna(subset=["dt"])
    good["dt"] = good["dt"].dt.tz_localize(None)
    good = good.set_index("dt")["value"]
    good = good.resample("15min").mean()

    # Coverage within 2023-2026
    window = good["2023-01-01":"2026-03-25"]
    total_steps = len(pd.date_range("2023-01-01", "2026-03-25", freq="15min"))
    print(f"\n  Good rows (2023–2026): {window.notna().sum()} / {total_steps}")
    print(f"  Coverage:              {window.notna().sum() / total_steps * 100:.1f}%")
    print(f"  First reading:         {window.first_valid_index()}")
    print(f"  Last reading:          {window.last_valid_index()}")

    # Gap analysis
    gaps = window.isna()
    gap_runs = (gaps != gaps.shift()).cumsum()[gaps]
    if len(gap_runs):
        largest_gap = gap_runs.value_counts().max()
        print(f"  Largest single gap:    {largest_gap} steps ({largest_gap * 15 / 60:.1f} hrs)")


if __name__ == "__main__":
    build_dataset(save=True)
    inspect_dataset()

# if __name__ == "__main__":
#     import xarray as xr  # reads and handles netcdf files with metadata
#     data = xr.open_dataset(ERA5_SM_FILE)
#     # print(data.variables)
#     print(data.variables.keys())


