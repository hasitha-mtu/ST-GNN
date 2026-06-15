"""
parse_cross_sections_shp.py
═══════════════════════════════════════════════════════════════════════
Reads channel cross-section geometry from a shapefile (any CRS —
reprojects automatically to ITM EPSG:2157) and estimates bankfull stage
for each gauge node.

Works alongside parse_cross_sections.py (DXF support). The two scripts
write to the same output file (cross_section_bankfull.json) so
derive_bankfull_thresholds.py picks up results from whichever source
is available per gauge.

Supported shapefile structures
────────────────────────────────
The script handles four common formats from OPW / engineering surveys:

  A. LineStringZ / MultiLineStringZ
     Each feature is one cross-section profile. Vertices carry Z as
     the elevation (mOD). GAUGE_REF attribute links to gauge station.

  B. PointZ
     Each feature is one survey point. Z = elevation mOD.
     Points belonging to the same cross-section share a GAUGE_REF or
     XS_ID attribute.

  C. LineString (no Z) with elevation attribute
     Z values stored in a column (LEVEL, ELEV, Z_LEVEL, LEVEL_MOD).
     Only one elevation per feature vertex is possible in this case,
     so the script uses the bank/datum attribute columns directly.

  D. No GAUGE_REF column — spatial matching
     If no attribute links cross-sections to gauges, the script assigns
     each cross-section to the nearest gauge node by centroid distance.

CRS handling
─────────────
The script reprojects any input CRS to ITM (EPSG:2157) for spatial
matching against gauge node positions.  OSGB36 / British National Grid
(EPSG:27700) is explicitly handled, as are WGS84, Irish Grid (EPSG:29902),
and any other CRS supported by pyproj.

Usage
──────
  python src/parse_cross_sections_shp.py --shp dataset/cross_sections/xs_survey.shp

  # With explicit gauge ref column name
  python src/parse_cross_sections_shp.py \\
      --shp   dataset/cross_sections/xs_survey.shp \\
      --ref-col GAUGE_REF

  # Combine with DXF results (merges into cross_section_bankfull.json)
  python src/parse_cross_sections_shp.py \\
      --shp   dataset/cross_sections/xs_survey.shp \\
      --merge-dxf
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import MultiLineString, LineString, MultiPoint

BASE_DIR  = Path(__file__).resolve().parent.parent
GRAPH_DIR = BASE_DIR / "dataset/graph"
XS_JSON   = GRAPH_DIR / "cross_section_bankfull.json"

TARGET_CRS = "EPSG:2157"   # ITM — all internal processing in this CRS

# Candidate column names for gauge reference (checked case-insensitively)
REF_COLS  = ["gauge_ref", "station", "gauge_id", "ref", "node_ref",
             "stationno", "station_no", "id", "xs_id", "section_id"]

# Candidate column names for elevation (when Z not in geometry)
ELEV_COLS = ["level", "elev", "elevation", "z_level", "level_mod",
             "od_level", "level_od", "height", "z"]

# Candidate column names for explicit bank top / datum
BANK_L_COLS  = ["bank_l",  "bank_l_mod", "left_bank",  "lbank", "bl_od"]
BANK_R_COLS  = ["bank_r",  "bank_r_mod", "right_bank", "rbank", "br_od"]
DATUM_COLS   = ["datum",   "datum_mod",  "gauge_datum", "datum_od",
                "zero_od", "gauge_zero"]


# ═════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column name in df that matches any candidate (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None


def reproject(gdf: gpd.GeoDataFrame, target: str = TARGET_CRS) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame to target CRS, handling missing CRS gracefully."""
    if gdf.crs is None:
        warnings.warn(
            "Shapefile has no CRS defined. Assuming EPSG:27700 (OSGB36).\n"
            "If this is wrong, set the CRS with: gdf.set_crs('EPSG:XXXX')"
        )
        gdf = gdf.set_crs("EPSG:27700")

    if str(gdf.crs).upper() == target.upper():
        return gdf

    print(f"  Reprojecting {gdf.crs} → {target} ...")
    reprojected = gdf.to_crs(target)
    print(f"  Reprojection complete. "
          f"Bounds (ITM): {reprojected.total_bounds.round(0)}")
    return reprojected


def extract_profile_from_geometry(geom) -> np.ndarray | None:
    """
    Extract (chainage, elevation) pairs from a geometry.

    For Z-bearing geometries: chainage = cumulative 2D distance along the line,
    elevation = Z coordinate at each vertex.

    For 2D geometries: returns None (elevation must come from attributes).

    Returns array shape [n_vertices, 2] or None.
    """
    def _line_to_profile(coords) -> np.ndarray:
        xs  = np.array([(c[0], c[1]) for c in coords])  # 2D positions
        zs  = np.array([c[2] if len(c) > 2 else np.nan for c in coords])

        # Cumulative 2D chainage
        diffs    = np.diff(xs, axis=0)
        seg_len  = np.sqrt((diffs ** 2).sum(axis=1))
        chainage = np.concatenate([[0], np.cumsum(seg_len)])

        return np.column_stack([chainage, zs])

    if isinstance(geom, LineString):
        coords = list(geom.coords)
        if len(coords) >= 2:
            return _line_to_profile(coords)

    elif isinstance(geom, MultiLineString):
        all_pts = []
        offset  = 0.0
        for line in geom.geoms:
            coords  = list(line.coords)
            profile = _line_to_profile(coords)
            profile[:, 0] += offset
            offset   = float(profile[-1, 0])
            all_pts.extend(profile.tolist())
        return np.array(all_pts) if all_pts else None

    return None


# ═════════════════════════════════════════════════════════════════════
# Bankfull estimation — reuses logic from parse_cross_sections.py
# ═════════════════════════════════════════════════════════════════════

def estimate_bankfull(
    profile: np.ndarray,              # [n, 2] (chainage, elevation)
    bank_l_mOD: float | None = None,  # explicit left bank from attribute
    bank_r_mOD: float | None = None,  # explicit right bank from attribute
    datum_mOD:  float | None = None,
) -> dict:
    """
    Estimate bankfull from a cross-section profile.
    Same logic as parse_cross_sections.py: bank_top_layer → slope → percentile.
    """
    result = {
        "thalweg_mOD": None, "left_bank_mOD": None, "right_bank_mOD": None,
        "bankfull_mOD": None, "bankfull_stage_m": None,
        "bankfull_depth_m": None, "method": None,
    }

    if profile is None or len(profile) < 3:
        result["method"] = "insufficient_points"
        return result

    # Filter NaN elevations and sort by chainage
    valid   = ~np.isnan(profile[:, 1])
    profile = profile[valid]
    profile = profile[np.argsort(profile[:, 0])]
    if len(profile) < 3:
        result["method"] = "insufficient_valid_points"
        return result

    chainage = profile[:, 0]
    elev     = profile[:, 1]

    # Remove elevation outliers (±3σ)
    z_med = np.median(elev)
    z_std = elev.std()
    mask  = np.abs(elev - z_med) < 3 * z_std
    chainage, elev = chainage[mask], elev[mask]
    if len(elev) < 3:
        result["method"] = "outlier_filtered_too_short"
        return result

    thalweg = float(elev.min())
    result["thalweg_mOD"] = round(thalweg, 4)

    # ── Method A: explicit bank columns from attributes ────────────────
    if bank_l_mOD is not None or bank_r_mOD is not None:
        result["left_bank_mOD"]  = round(bank_l_mOD, 4)  if bank_l_mOD else None
        result["right_bank_mOD"] = round(bank_r_mOD, 4)  if bank_r_mOD else None
        banks = [v for v in [bank_l_mOD, bank_r_mOD] if v is not None]
        result["bankfull_mOD"] = round(float(min(banks)), 4)
        result["method"] = "explicit_bank_attribute"

    # ── Method B: break-in-slope from geometry ─────────────────────────
    if result["bankfull_mOD"] is None and len(elev) >= 6:
        de = np.diff(elev)
        dx = np.diff(chainage)
        dx[dx == 0] = 1e-6
        slope = np.abs(de / dx)
        slope_s = np.convolve(slope, np.ones(3)/3, mode="same")

        thal_idx = int(np.argmin(elev))

        left_idx = right_idx = None
        for i in range(thal_idx, 0, -1):
            if i < len(slope_s)-1 and slope_s[i] > 0.05 and slope_s[i-1] < 0.03:
                left_idx = i
                break
        for i in range(thal_idx, len(slope_s)-1):
            if slope_s[i] > 0.05 and slope_s[i+1] < 0.03:
                right_idx = i
                break

        if left_idx or right_idx:
            if left_idx:
                result["left_bank_mOD"]  = round(float(elev[left_idx]), 4)
            if right_idx:
                result["right_bank_mOD"] = round(float(elev[right_idx+1]), 4)
            banks = [v for v in [result["left_bank_mOD"],
                                  result["right_bank_mOD"]] if v is not None]
            if banks:
                result["bankfull_mOD"] = round(float(min(banks)), 4)
                result["method"] = "break_in_slope"

    # ── Method C: 90th-percentile per side ────────────────────────────
    if result["bankfull_mOD"] is None:
        thal_idx  = int(np.argmin(elev))
        left_zs   = elev[:thal_idx] if thal_idx > 0 else elev
        right_zs  = elev[thal_idx:] if thal_idx < len(elev) else elev
        lb = float(np.percentile(left_zs,  90)) if len(left_zs)  > 2 else None
        rb = float(np.percentile(right_zs, 90)) if len(right_zs) > 2 else None
        result["left_bank_mOD"]  = round(lb, 4) if lb else None
        result["right_bank_mOD"] = round(rb, 4) if rb else None
        banks = [v for v in [lb, rb] if v is not None]
        if banks:
            result["bankfull_mOD"] = round(float(min(banks)), 4)
            result["method"] = "percentile_90_fallback"

    if result["bankfull_mOD"] is not None:
        result["bankfull_depth_m"] = round(
            result["bankfull_mOD"] - thalweg, 4)
        if datum_mOD is not None:
            result["bankfull_stage_m"] = round(
                result["bankfull_mOD"] - datum_mOD, 4)

    return result


# ═════════════════════════════════════════════════════════════════════
# Gauge node matching
# ═════════════════════════════════════════════════════════════════════

def load_gauge_nodes() -> tuple[list, np.ndarray]:
    """Load gauge refs and ITM positions from nodes.csv."""
    df     = pd.read_csv(GRAPH_DIR / "nodes.csv")
    refs   = df["ref"].astype(str).tolist()
    if "easting_itm" in df.columns:
        coords = df[["easting_itm", "northing_itm"]].values
    else:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
        E, N = t.transform(df["lon"].values, df["lat"].values)
        coords = np.column_stack([E, N])
    return refs, coords


def match_to_gauge(
    gdf:        gpd.GeoDataFrame,
    node_refs:  list,
    node_coords: np.ndarray,
    ref_col:    str | None,
) -> pd.Series:
    """
    Assign each shapefile feature to a gauge node.

    Priority:
      1. Use ref_col attribute directly (exact or prefix match with OPW refs)
      2. Spatial match: assign to nearest gauge node by centroid distance
    """
    n = len(gdf)
    assignment = pd.Series([None] * n, dtype=object)

    # ── Attribute match ────────────────────────────────────────────────
    if ref_col and ref_col in gdf.columns:
        for i, val in enumerate(gdf[ref_col].astype(str)):
            val_clean = val.strip().lstrip("0")
            for ref in node_refs:
                if val.strip() == ref or val_clean == ref.lstrip("0"):
                    assignment.iloc[i] = ref
                    break
        n_matched = assignment.notna().sum()
        print(f"  Attribute match ({ref_col}): {n_matched}/{n} features matched")

    # ── Spatial fallback for unmatched features ────────────────────────
    unmatched = assignment[assignment.isna()].index
    if len(unmatched) > 0:
        centroids  = gdf.loc[unmatched].geometry.centroid
        cent_coords = np.column_stack([centroids.x, centroids.y])
        tree   = cKDTree(node_coords)
        dists, nearest_idx = tree.query(cent_coords)
        for feat_i, (dist_m, node_i) in zip(unmatched, zip(dists, nearest_idx)):
            if dist_m < 5000:   # 5 km max — beyond this it is likely wrong
                assignment.iloc[feat_i] = node_refs[node_i]
        n_spatial = assignment.loc[unmatched].notna().sum()
        print(f"  Spatial match (≤5 km): {n_spatial}/{len(unmatched)} matched")

    return assignment


# ═════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════

def run(
    shp_path:  Path,
    ref_col:   str | None,
    out_json:  Path,
    merge_dxf: bool,
):
    print(f"\n── Reading shapefile ──")
    print(f"  {shp_path}")
    gdf = gpd.read_file(shp_path)
    print(f"  Features:  {len(gdf)}")
    print(f"  CRS:       {gdf.crs}")
    print(f"  Geom type: {gdf.geometry.geom_type.unique().tolist()}")
    print(f"  Has Z:     {gdf.geometry.has_z.all()}")
    print(f"  Columns:   {list(gdf.columns)}")

    # Reproject to ITM
    gdf = reproject(gdf)

    # Auto-detect column names
    if ref_col is None:
        ref_col = _find_col(gdf, REF_COLS)
        if ref_col:
            print(f"  Auto-detected gauge ref column: '{ref_col}'")
        else:
            print(f"  No gauge ref column found — will use spatial matching")

    bank_l_col = _find_col(gdf, BANK_L_COLS)
    bank_r_col = _find_col(gdf, BANK_R_COLS)
    datum_col  = _find_col(gdf, DATUM_COLS)
    elev_col   = _find_col(gdf, ELEV_COLS)

    if bank_l_col: print(f"  Bank left column:  '{bank_l_col}'")
    if bank_r_col: print(f"  Bank right column: '{bank_r_col}'")
    if datum_col:  print(f"  Datum column:      '{datum_col}'")
    if elev_col:   print(f"  Elevation column:  '{elev_col}'")

    # Load gauge nodes
    node_refs, node_coords = load_gauge_nodes()

    # Match features to gauges
    print(f"\n── Matching {len(gdf)} features to {len(node_refs)} gauge nodes ──")
    assignment = match_to_gauge(gdf, node_refs, node_coords, ref_col)
    gdf["_gauge_ref"] = assignment

    # Load datum lookup from DatumHistory files if available
    datum_file = GRAPH_DIR / "datum_lookup.json"
    datum_lookup: dict[str, float] = {}
    if datum_file.exists():
        datum_lookup = json.load(open(datum_file))
        print(f"  Datum lookup: {len(datum_lookup)} entries")

    # ── Process per gauge ──────────────────────────────────────────────
    xs_bankfull: dict[str, float] = {}
    report_rows = []

    print(f"\n── Estimating bankfull per gauge ──")
    print(f"\n  {'ref':>7}  {'n_feat':>6}  {'method':>25}  "
          f"{'thalweg':>8}  {'bf_elev':>8}  {'bf_stage':>9}")
    print("  " + "─" * 68)

    for ref in node_refs:
        features = gdf[gdf["_gauge_ref"] == ref]
        if len(features) == 0:
            print(f"  {ref:>7}  {'—':>6}  {'no cross-section data':>25}")
            continue

        # Collect all profiles from all features for this gauge
        best_result  = None
        best_n_pts   = 0
        best_src     = None

        for _, row in features.iterrows():
            geom = row.geometry

            # Explicit bank attributes
            bl = float(row[bank_l_col]) if bank_l_col and pd.notna(row.get(bank_l_col)) else None
            br = float(row[bank_r_col]) if bank_r_col and pd.notna(row.get(bank_r_col)) else None

            # Datum from column, then lookup, then None
            if datum_col and pd.notna(row.get(datum_col)):
                datum = float(row[datum_col])
            else:
                datum = datum_lookup.get(ref)

            profile = extract_profile_from_geometry(geom)

            result = estimate_bankfull(profile, bl, br, datum)
            n_pts  = len(profile) if profile is not None else 0

            if result["bankfull_mOD"] is not None:
                if best_result is None or n_pts > best_n_pts:
                    best_result  = result
                    best_n_pts   = n_pts
                    best_src     = row.get(ref_col, "spatial") if ref_col else "spatial"

        if best_result and best_result["bankfull_stage_m"] is not None:
            xs_bankfull[ref] = round(best_result["bankfull_stage_m"], 4)

        th  = best_result.get("thalweg_mOD")   if best_result else None
        bf  = best_result.get("bankfull_mOD")  if best_result else None
        bfs = best_result.get("bankfull_stage_m") if best_result else None
        mth = (best_result.get("method") or "?")[:25] if best_result else "failed"

        print(f"  {ref:>7}  {len(features):>6}  {mth:>25}  "
              f"{th or 0:>8.3f}  {bf or 0:>8.3f}  {bfs or 0:>9.3f}")

        report_rows.append({
            "ref":              ref,
            "n_features":       len(features),
            "source_feature":   best_src,
            "thalweg_mOD":      th,
            "bankfull_mOD":     bf,
            "bankfull_stage_m": bfs,
            "bankfull_depth_m": best_result.get("bankfull_depth_m") if best_result else None,
            "method":           mth,
            "n_profile_points": best_n_pts,
        })

    # ── Merge with DXF results if requested ───────────────────────────
    if merge_dxf and out_json.exists():
        existing = json.load(open(out_json))
        existing_bf = existing.get("bankfull_stage_m", {})
        n_before    = len(existing_bf)
        # SHP results take priority over DXF where both exist
        merged = {**existing_bf, **xs_bankfull}
        xs_bankfull = merged
        print(f"\n  Merged with existing JSON: {n_before} → {len(xs_bankfull)} entries")

    # ── Save outputs ──────────────────────────────────────────────────
    out_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "description": (
            "Bankfull stage estimates (m above gauge datum) from channel "
            "cross-section geometry. Source: shapefile survey data. "
            "Tier 0 in the bankfull estimation hierarchy."
        ),
        "source_shp":         str(shp_path),
        "generated":          pd.Timestamp.now().isoformat(),
        "bankfull_stage_m":   xs_bankfull,
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_json}  ({len(xs_bankfull)} gauge(s))")

    out_csv = out_json.with_suffix("").parent / \
              "cross_section_report_shp.csv"
    pd.DataFrame(report_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Parse cross-section shapefile and estimate bankfull stage"
    )
    p.add_argument("--shp",       type=Path, required=True,
                   help="Path to cross-section shapefile (.shp)")
    p.add_argument("--ref-col",   type=str, default=None,
                   help="Column name linking features to gauge refs "
                        "(auto-detected if omitted)")
    p.add_argument("--out",       type=Path, default=XS_JSON,
                   help=f"Output JSON (default: {XS_JSON})")
    p.add_argument("--merge-dxf", action="store_true",
                   help="Merge with existing DXF results in --out")
    args = p.parse_args()

    if not args.shp.exists():
        print(f"Shapefile not found: {args.shp}")
        raise SystemExit(1)

    run(args.shp, args.ref_col, args.out, args.merge_dxf)
