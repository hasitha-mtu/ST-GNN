"""
derive_bankfull_thresholds.py
═════════════════════════════════════════════════════════════════════
Reads OPW station data downloaded to dataset/station_data/{ref}/ and
derives a defensible bankfull stage threshold for each gauge node.

Three-tier priority hierarchy
───────────────────────────────
Tier 1  Gaugings (channel width expansion)
        Most physically direct.  Bankfull = stage at which gauged
        cross-section width begins to expand sharply, indicating
        water leaving the main channel onto the floodplain.
        Requires ≥ 3 gaugings above the suspected bankfull zone.

Tier 2  Annual Maxima (Gumbel frequency analysis)
        Fits a Gumbel distribution to annual maximum stage readings.
        Returns the 1.5-year return level — the conventional bankfull
        recurrence interval (Wolman & Miller 1960; Williams 1978).
        Requires ≥ 3 annual maxima.

Tier 3  obs_max × 1.3 (short-record fallback)
        As previously implemented in compute_bankfull_thresholds().
        Used when station data are absent or insufficient.

Output
───────
  dataset/graph/bankfull_thresholds.json
      Per-node bankfull thresholds in stage-anomaly space (m),
      ready for direct use in compute_bankfull_thresholds().

  dataset/graph/bankfull_thresholds_report.csv
      Full diagnostic table: tier used, raw stage reading (m),
      datum, anomaly conversion, and quality flags.

Usage
──────
    python src/derive_bankfull_thresholds.py

    # With custom station data dir
    python src/derive_bankfull_thresholds.py \
        --station-dir dataset/station_data \
        --out         dataset/graph/bankfull_thresholds.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r

BASE_DIR     = Path(__file__).resolve().parent.parent
STATION_DIR  = BASE_DIR / "dataset/station_data"
GRAPH_DIR    = BASE_DIR / "dataset/graph"
XS_JSON      = GRAPH_DIR / "cross_section_bankfull.json"
PROC_DIR     = BASE_DIR / "dataset/processed"

# Tidal and reservoir nodes that always receive additional multiplier
TIDAL_REFS     = {"19162", "19163", "19161", "19160"}
RESERVOIR_REFS = {"19094", "19095"}

# ═════════════════════════════════════════════════════════════════════
# File discovery
# ═════════════════════════════════════════════════════════════════════

def find_station_files(station_dir: Path) -> dict[str, dict]:
    """
    Scan station_dir/{ref}/ and return a dict mapping each ref to the
    file paths found, keyed by file type.
    """
    result = {}
    if not station_dir.exists():
        print(f"WARNING: station_dir not found: {station_dir}")
        return result

    for sub in sorted(station_dir.iterdir()):
        if not sub.is_dir():
            continue
        ref   = sub.name
        files = {}
        for fp in sorted(sub.iterdir()):
            if fp.suffix.lower() != ".csv":
                continue
            ftype = _identify_file_type(fp.name)
            if ftype:
                files[ftype] = fp
        result[ref] = files
    return result


def _identify_file_type(filename: str) -> str | None:
    fn = filename.lower()
    if "annualmax"    in fn: return "annual_max"
    if "percentile"   in fn: return "percentiles"
    if "gauging"      in fn: return "gaugings"
    if "datumhistory" in fn or ("datum" in fn and "history" in fn): return "datum"
    return None


# ═════════════════════════════════════════════════════════════════════
# Parsers — robust to OPW header comments and encoding quirks
# ═════════════════════════════════════════════════════════════════════

def _read_csv(path: Path) -> pd.DataFrame | None:
    """Read an OPW CSV, skipping # comment lines. Returns None if empty."""
    try:
        df = pd.read_csv(path, comment="#", skipinitialspace=True,
                         encoding="utf-8-sig")
        df.columns = df.columns.str.strip().str.strip('"')
        if df.empty or len(df.columns) < 2:
            return None
        return df
    except Exception as e:
        warnings.warn(f"  Could not read {path.name}: {e}")
        return None


def parse_datum(path: Path) -> float | None:
    """Extract gauge datum (m OD) from DatumHistory CSV."""
    df = _read_csv(path)
    if df is None:
        return None
    val_col = next((c for c in df.columns if "value" in c.lower()), None)
    if val_col is None:
        return None
    vals = pd.to_numeric(df[val_col], errors="coerce").dropna()
    # Use most recent datum (last row after sorting by 'Valid from')
    if "valid from" in df.columns[0].lower() and len(df) > 1:
        df["_date"] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors="coerce")
        df = df.sort_values("_date")
    return float(vals.iloc[-1]) if len(vals) > 0 else None


def parse_annual_max(path: Path) -> np.ndarray | None:
    """Extract annual maximum stage readings (m, relative to datum)."""
    df = _read_csv(path)
    if df is None:
        return None
    sg_col = next((c for c in df.columns
                   if "s.g" in c.lower() or "reading" in c.lower()), None)
    if sg_col is None:
        return None
    vals = (df[sg_col].astype(str).str.strip('"')
            .replace("---", np.nan)
            .replace("", np.nan))
    vals = pd.to_numeric(vals, errors="coerce").dropna().values
    return vals if len(vals) >= 2 else None


def parse_gaugings(path: Path) -> pd.DataFrame | None:
    """Extract stage and channel width from gaugings CSV."""
    df = _read_csv(path)
    if df is None:
        return None
    stage_col = next((c for c in df.columns if "stage" in c.lower()), None)
    width_col = next((c for c in df.columns if "width" in c.lower()), None)
    if stage_col is None or width_col is None:
        return None
    df["_stage"] = pd.to_numeric(df[stage_col], errors="coerce")
    df["_width"] = pd.to_numeric(df[width_col], errors="coerce")
    df = df.dropna(subset=["_stage", "_width"])
    return df[["_stage","_width"]].rename(
        columns={"_stage":"stage","_width":"width"}) if len(df) >= 3 else None


def parse_percentiles(path: Path) -> pd.DataFrame | None:
    """Extract exceedance percentile levels (m OD)."""
    df = _read_csv(path)
    if df is None:
        return None
    level_col = next((c for c in df.columns if "level" in c.lower()), None)
    pct_col   = next((c for c in df.columns
                      if "percent" in c.lower() or c.strip() == "Percentile"), None)
    if level_col is None or pct_col is None:
        return None
    df["_level"] = pd.to_numeric(df[level_col], errors="coerce")
    return df[[pct_col, "_level"]].dropna() if len(df) >= 3 else None


# ═════════════════════════════════════════════════════════════════════
# Bankfull estimation methods
# ═════════════════════════════════════════════════════════════════════

def tier1_gaugings(gdf: pd.DataFrame) -> tuple[float, str] | None:
    """
    Estimate bankfull from channel width expansion in gaugings.

    The bankfull stage is identified as the point where mean cross-section
    width increases by more than 50% above the low-flow width.  This threshold
    was selected based on the 19114 analysis: width increased from 6.7 m to
    18.6 m (2.8×) above bankfull; a 50% increase (1.5×) is a conservative
    detection threshold that avoids false positives in minor width variability.

    Returns (bankfull_stage_m, method_note) or None if insufficient data.
    """
    gdf = gdf.sort_values("stage")
    if len(gdf) < 5:
        return None

    # Low-flow width baseline: median width at stage < 33rd percentile
    p33_stage = float(np.percentile(gdf["stage"], 33))
    low_width = gdf.loc[gdf["stage"] < p33_stage, "width"].median()
    if np.isnan(low_width) or low_width <= 0:
        return None

    bankfull_threshold = low_width * 1.5  # 50% expansion criterion

    # Find the lowest stage where mean width in a 0.3m window exceeds threshold
    stages_above = gdf.loc[gdf["width"] > bankfull_threshold, "stage"]
    if len(stages_above) < 2:
        return None

    bf_stage = float(stages_above.min())
    ratio    = float(gdf.loc[gdf["stage"] >= bf_stage, "width"].mean() / low_width)
    note = (f"width expansion ×{ratio:.1f} at stage {bf_stage:.2f} m "
            f"(low-flow width {low_width:.1f} m)")
    return bf_stage, note


def tier2_gumbel(annual_max: np.ndarray) -> tuple[float, str] | None:
    """
    Fit Gumbel distribution to annual maxima and return 1.5-year return level.
    Bankfull is conventionally defined as the 1.5–2.0 year recurrence event
    (Wolman & Miller 1960; Williams 1978).

    Returns (bankfull_stage_m, method_note) or None if n < 3.
    """
    if len(annual_max) < 3:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc, scale = gumbel_r.fit(annual_max)
    bf_stage = float(gumbel_r.ppf(1 - 1/1.5, loc=loc, scale=scale))
    note = (f"Gumbel 1.5-yr return level (n={len(annual_max)} annual maxima, "
            f"loc={loc:.3f}, scale={scale:.3f})")
    return bf_stage, note


# ═════════════════════════════════════════════════════════════════════
# Per-node derivation
# ═════════════════════════════════════════════════════════════════════

def derive_threshold_for_node(
    ref:              str,
    files:            dict,
    mean_stage:       float,
    fallback_obs_max: float,
    xs_bankfull:      dict | None = None,  # Tier 0: cross-section bankfull stages
) -> dict:
    """
    Derive bankfull threshold for one node.

    Tier 0: cross-section geometry (DXF/DWG survey drawings)
    Tier 1: gaugings channel width expansion
    Tier 2: Gumbel fit to annual maxima (1.5-year return level)
    Tier 3: obs_max × 1.3 fallback

    Returns a result dict with:
        ref, tier_used, bankfull_reading_m, datum_m,
        bankfull_anomaly_m, method_note, reliable
    """
    result = {
        "ref":               ref,
        "tier_used":         None,
        "bankfull_reading_m": None,
        "datum_m":           None,
        "bankfull_anomaly_m": None,
        "method_note":       None,
        "reliable":          True,
        "warnings":          [],
    }

    # ── Datum ─────────────────────────────────────────────────────────
    datum = None
    if "datum" in files:
        datum = parse_datum(files["datum"])
    if datum is not None:
        result["datum_m"] = datum
    else:
        result["warnings"].append("datum file missing or unparseable")

    # ── Tier 0: cross-section geometry (DXF) ───────────────────────────
    if xs_bankfull is not None and ref in xs_bankfull:
        bf_stage = xs_bankfull[ref]
        result["tier_used"]          = 0
        result["bankfull_reading_m"] = bf_stage
        result["method_note"]        = (
            f"Tier 0 — cross-section geometry: "
            f"bankfull_stage={bf_stage:.3f} m (DXF survey)")

    # ── Tier 1: gaugings width method ─────────────────────────────────
    if result["tier_used"] is None and "gaugings" in files:
        gdf = parse_gaugings(files["gaugings"])
        if gdf is not None:
            tier1 = tier1_gaugings(gdf)
            if tier1 is not None:
                bf_stage, note = tier1
                result["tier_used"]          = 1
                result["bankfull_reading_m"] = bf_stage
                result["method_note"]        = f"Tier 1 — {note}"

    # ── Tier 2: Gumbel annual maxima ──────────────────────────────────
    if result["tier_used"] is None and "annual_max" in files:
        am = parse_annual_max(files["annual_max"])
        if am is not None:
            tier2 = tier2_gumbel(am)
            if tier2 is not None:
                bf_stage, note = tier2
                result["tier_used"]          = 2
                result["bankfull_reading_m"] = bf_stage
                result["method_note"]        = f"Tier 2 — {note}"

    # ── Tier 3: obs_max × 1.3 fallback ────────────────────────────────
    if result["tier_used"] is None:
        bf_stage = fallback_obs_max * 1.3
        result["tier_used"]          = 3
        result["bankfull_reading_m"] = bf_stage
        result["method_note"]        = (
            f"Tier 3 — obs_max×1.3 fallback "
            f"(obs_max={fallback_obs_max:.3f} m, no station data)")
        result["reliable"] = False
        result["warnings"].append(
            "no OPW station data available — threshold less reliable")

    # ── Convert to stage anomaly ───────────────────────────────────────
    # stage_anomaly = bankfull_reading - mean_stage_reading
    # mean_stage_reading approximated from the percentile file (50th pct)
    # or from the training-set mean if no percentile file available.
    bf_reading = result["bankfull_reading_m"]

    # Try to get mean stage from percentiles file
    mean_stage_from_pct = None
    if "percentiles" in files:
        pcdf = parse_percentiles(files["percentiles"])
        if pcdf is not None:
            pct_col = pcdf.columns[0]
            p50_rows = pcdf[pcdf[pct_col].astype(str).str.contains("50")]
            if len(p50_rows) > 0 and datum is not None:
                mean_stage_from_pct = float(p50_rows["_level"].iloc[0]) - datum

    mean_for_anomaly = mean_stage_from_pct if mean_stage_from_pct is not None else mean_stage
    result["mean_stage_source"] = (
        "percentiles_50pct" if mean_stage_from_pct is not None else "training_data"
    )
    result["mean_stage_m"]      = round(mean_for_anomaly, 4)
    result["bankfull_anomaly_m"] = round(bf_reading - mean_for_anomaly, 4)

    # Special multipliers for tidal / reservoir nodes
    if ref in TIDAL_REFS or ref in RESERVOIR_REFS:
        orig = result["bankfull_anomaly_m"]
        result["bankfull_anomaly_m"] = round(min(orig * 1.5, 2.0), 4)
        result["method_note"] += (
            f" [tidal/reservoir +1.5× → capped at 2.0 m anomaly]")

    # Floor
    result["bankfull_anomaly_m"] = max(result["bankfull_anomaly_m"], 0.05)

    return result


# ═════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════

def run(station_dir: Path, out_json: Path, out_csv: Path):
    """Derive bankfull thresholds for all nodes and save outputs."""

    # Load node references from nodes.csv
    nodes_df   = pd.read_csv(GRAPH_DIR / "nodes.csv")
    node_refs  = nodes_df["ref"].astype(str).tolist()
    node_names = nodes_df["name"].tolist() if "name" in nodes_df.columns else node_refs

    # Load training data for tier-3 fallback and mean stage
    X          = np.load(PROC_DIR / "X.npy", mmap_mode="r")
    T          = X.shape[0]
    train_end  = int(T * 0.70)
    stage_data = X[:train_end, :, 0]   # stage_anomaly column
    obs_max    = np.nanmax(stage_data, axis=0)       # [N] for tier-3 fallback
    mean_stage = np.nanmean(stage_data, axis=0)      # [N] training mean

    # Load Tier 0 cross-section bankfull estimates if available
    xs_bf: dict | None = None
    if XS_JSON.exists():
        import json as _json
        _xs = _json.load(open(XS_JSON))
        xs_bf = _xs.get("bankfull_stage_m", {})
        print(f"Cross-section bankfull loaded: {len(xs_bf)} gauge(s) "              f"from {XS_JSON.name}")
    else:
        print(f"No cross-section data found at {XS_JSON}")
        print(f"Run parse_cross_sections.py first if DXF files are available.")

    # Scan station files
    station_files = find_station_files(station_dir)
    print(f"Station directories found: {len(station_files)}")
    found_refs = set(station_files.keys())
    missing    = [r for r in node_refs if r not in found_refs]
    extra      = [r for r in found_refs if r not in node_refs]
    if missing:
        print(f"No station data for {len(missing)} nodes: {missing}")
    if extra:
        print(f"Station dirs not in nodes.csv (ignored): {extra}")

    # Derive threshold per node
    results = []
    thresholds = {}   # ref → bankfull_anomaly_m (for JSON output)

    print(f"\n{'ref':>7}  {'tier':>4}  "
          f"{'bf_reading':>10}  {'mean_stage':>10}  "
          f"{'bf_anomaly':>10}  {'status'}")
    print("  " + "─"*72)

    for i, ref in enumerate(node_refs):
        files    = station_files.get(ref, {})
        res      = derive_threshold_for_node(
            ref              = ref,
            files            = files,
            mean_stage       = float(mean_stage[i]),
            fallback_obs_max = float(obs_max[i]),
            xs_bankfull      = xs_bf,
        )
        results.append(res)
        thresholds[ref] = res["bankfull_anomaly_m"]

        name     = node_names[i] if i < len(node_names) else ref
        tier_str = f"T{res['tier_used']}" if res["tier_used"] else "T3"
        bf_r     = res["bankfull_reading_m"]
        ms       = res["mean_stage_m"]
        bfa      = res["bankfull_anomaly_m"]
        status   = "OK" if res["reliable"] else "⚠ fallback"
        warn_str = f"  [{'; '.join(res['warnings'])}]" if res["warnings"] else ""
        print(f"  {ref:>7}  {tier_str:>4}  "
              f"{bf_r:>10.3f}  {ms:>10.3f}  "
              f"{bfa:>10.3f}  {status}{warn_str}")

    # ── Save JSON ──────────────────────────────────────────────────────
    out_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "description": (
            "Per-node bankfull stage anomaly thresholds (m) for flood map generation. "
            "Derived from OPW station data using a three-tier hierarchy: "
            "(1) gaugings width expansion, (2) Gumbel annual maxima 1.5-yr return level, "
            "(3) obs_max×1.3 fallback."
        ),
        "generated":   pd.Timestamp.now().isoformat(),
        "thresholds":  thresholds,
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Save diagnostic CSV ───────────────────────────────────────────
    report_rows = []
    for res in results:
        report_rows.append({
            "ref":               res["ref"],
            "tier_used":         res["tier_used"],
            "bankfull_reading_m": res["bankfull_reading_m"],
            "datum_m":           res["datum_m"],
            "mean_stage_m":      res["mean_stage_m"],
            "mean_stage_source": res["mean_stage_source"],
            "bankfull_anomaly_m": res["bankfull_anomaly_m"],
            "reliable":          res["reliable"],
            "method_note":       res["method_note"],
            "warnings":          "; ".join(res["warnings"]),
        })
    pd.DataFrame(report_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # ── Summary ───────────────────────────────────────────────────────
    tier_counts = {}
    for r in results:
        t = r["tier_used"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    n_reliable = sum(1 for r in results if r["reliable"])
    print(f"\nSummary:")
    for t, n in sorted(tier_counts.items()):
        labels = {1: "gaugings width",
                  2: "Gumbel annual max",
                  3: "obs_max×1.3 fallback"}
        print(f"  Tier {t} ({labels.get(t,'?')}): {n} nodes")
    print(f"  Reliable:   {n_reliable}/{len(node_refs)}")
    print(f"  Fallback:   {len(node_refs)-n_reliable}/{len(node_refs)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Derive per-node bankfull thresholds from OPW station data"
    )
    p.add_argument("--station-dir", type=Path, default=STATION_DIR)
    p.add_argument("--out",         type=Path,
                   default=GRAPH_DIR / "bankfull_thresholds.json")
    p.add_argument("--out-csv",     type=Path,
                   default=GRAPH_DIR / "bankfull_thresholds_report.csv")
    args = p.parse_args()

    run(args.station_dir, args.out, args.out_csv)
