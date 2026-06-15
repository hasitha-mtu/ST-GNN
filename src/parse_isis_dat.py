"""
batch_parse_isis_dat.py
═══════════════════════════════════════════════════════════════════════
Scans a directory of ISIS / Flood Modeller .dat files, parses every
cross-section in every file, matches each section to the nearest gauge
node in nodes.csv, and writes bankfull stage estimates to
dataset/graph/cross_section_bankfull.json.

This is the primary entry point when multiple .dat files cover different
reaches of the Lee catchment (Lee mainstem, Shournagh, Dripsey, Bride,
Sullane, Glashaboy, etc.).

CRS auto-detection
───────────────────
Irish hydraulic model files are often mislabelled as OSGB36 but carry
Irish National Grid (TM65, EPSG:29902) coordinates.  The script detects
the CRS from the coordinate ranges in each file:

  E=50000–380000 AND N=10000–460000  →  Irish National Grid (EPSG:29902)
  E=480000–620000 AND N=530000–960000 →  ITM (EPSG:2157)  [pass-through]
  E=-200000–700000 AND N=0–1300000   →  OSGB36 (EPSG:27700)
  lon=-10–2 AND lat=49–61            →  WGS84 (EPSG:4326)

Bankfull extraction
────────────────────
Uses the panel boundary marker (*) in the Manning's n column — the most
physically precise bankfull indicator available (set by the hydraulic
engineer who calibrated the model).

  bankfull_mOD    = min(left_panel_boundary_mOD, right_panel_boundary_mOD)
  bankfull_stage  = bankfull_mOD − gauge_datum_mOD
  bankfull_anomaly = bankfull_stage − mean_training_stage_at_node

Gauge matching
───────────────
Each cross-section centroid is reprojected to ITM and matched to the
nearest gauge node within --max-dist metres (default 500 m — tight
enough to avoid wrong-tributary matches for the densely gauged Lee
network).

Output
───────
  dataset/graph/cross_section_bankfull.json   ← read by derive_bankfull_thresholds.py
  dataset/graph/isis_batch_report.csv         ← full diagnostic table

Usage
──────
  python src/batch_parse_isis_dat.py

  python src/batch_parse_isis_dat.py \\
      --dat-dir   dataset/cross_sections/dats \\
      --datum-csv dataset/graph/datum_lookup.csv \\
      --max-dist  500
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE_DIR  = Path(__file__).resolve().parent.parent
DAT_DIR   = BASE_DIR / "dataset/cross_sections/dats"
GRAPH_DIR = BASE_DIR / "dataset/graph"
PROC_DIR  = BASE_DIR / "dataset/processed"
XS_JSON   = GRAPH_DIR / "cross_section_bankfull.json"


# ═════════════════════════════════════════════════════════════════════
# CRS detection
# ═════════════════════════════════════════════════════════════════════

def detect_crs(easting_sample: float, northing_sample: float) -> str:
    """
    Infer the CRS from a representative coordinate pair.

    Returns an EPSG string, or raises ValueError if unrecognised.
    """
    e, n = easting_sample, northing_sample

    # WGS84 decimal degrees
    if -10 <= e <= 2 and 49 <= n <= 62:
        return "EPSG:4326"

    # ITM — already in target CRS
    if 480_000 <= e <= 620_000 and 530_000 <= n <= 960_000:
        return "EPSG:2157"

    # Irish National Grid (TM65) — mislabelled as OSGB36 in many files
    # Cork/Lee catchment: E≈140000-200000, N≈55000-110000
    if 50_000 <= e <= 380_000 and 10_000 <= n <= 460_000:
        return "EPSG:29902"

    # OSGB36 British National Grid — SE England positive, Scotland high N
    if -200_000 <= e <= 700_000 and 0 <= n <= 1_300_000:
        return "EPSG:27700"

    raise ValueError(
        f"Cannot detect CRS for coordinates ({e:.0f}, {n:.0f}). "
        f"Pass the CRS explicitly or check the data."
    )


def to_itm(eastings: np.ndarray, northings: np.ndarray,
           source_crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Reproject (eastings, northings) from source_crs to ITM EPSG:2157."""
    if source_crs == "EPSG:2157":
        return eastings, northings
    t = Transformer.from_crs(source_crs, "EPSG:2157", always_xy=True)
    return t.transform(eastings, northings)


# ═════════════════════════════════════════════════════════════════════
# ISIS .dat parser  (same logic as parse_isis_dat.py)
# ═════════════════════════════════════════════════════════════════════

def parse_dat(dat_path: Path) -> tuple[list[dict], str | None]:
    """
    Parse one ISIS .dat file.

    Returns (sections, detected_crs).
    Each section dict contains geometry, bankfull estimate, and centroid.
    """
    lines = dat_path.read_text(encoding="utf-8", errors="replace") \
                    .splitlines()
    lines = [l.rstrip('\r') for l in lines]

    sections: list[dict] = []
    detected_crs: str | None = None
    i = 0

    while i < len(lines):
        if lines[i].strip() != "SECTION":
            i += 1
            continue

        try:
            sec_name   = lines[i+1].strip()
            chainage_m = float(lines[i+2].strip())
            n_pts      = int(lines[i+3].strip())
        except (ValueError, IndexError):
            i += 1
            continue

        xs_data: list[dict] = []
        for j in range(n_pts):
            row = lines[i+4+j].split()
            if len(row) < 6:
                continue
            try:
                xs_data.append({
                    "dist_m":         float(row[0]),
                    "level_mOD":      float(row[1]),
                    "manning_n":      float(row[2].rstrip('*')),
                    "panel_boundary": '*' in row[2],
                    "coord_e":        float(row[4]),
                    "coord_n":        float(row[5]),
                })
            except (ValueError, IndexError):
                pass

        i += 4 + n_pts

        if not xs_data:
            continue

        # ── CRS detection from first valid section ─────────────────────
        if detected_crs is None:
            try:
                detected_crs = detect_crs(xs_data[0]["coord_e"],
                                           xs_data[0]["coord_n"])
            except ValueError as exc:
                warnings.warn(f"  {dat_path.name}: {exc}")

        # ── Bankfull from panel boundary markers ───────────────────────
        levels  = [d["level_mOD"] for d in xs_data]
        thal_i  = int(levels.index(min(levels)))

        lb_idx = [k for k, d in enumerate(xs_data)
                  if d["panel_boundary"] and k < thal_i]
        rb_idx = [k for k, d in enumerate(xs_data)
                  if d["panel_boundary"] and k > thal_i]

        lb_elev = xs_data[lb_idx[-1]]["level_mOD"] if lb_idx else None
        rb_elev = xs_data[rb_idx[0]]["level_mOD"]  if rb_idx else None
        bfs     = [v for v in [lb_elev, rb_elev] if v is not None]

        ce = np.mean([d["coord_e"] for d in xs_data])
        cn = np.mean([d["coord_n"] for d in xs_data])

        sections.append({
            "source_file":    dat_path.name,
            "name":           sec_name,
            "chainage_m":     chainage_m,
            "n_pts":          n_pts,
            "thalweg_mOD":    round(float(min(levels)), 4),
            "left_bank_mOD":  round(lb_elev, 4) if lb_elev else None,
            "right_bank_mOD": round(rb_elev, 4) if rb_elev else None,
            "bankfull_mOD":   round(float(min(bfs)), 4) if bfs else None,
            "bankfull_depth_m": round(float(min(bfs)) - float(min(levels)), 4)
                                if bfs else None,
            "width_m":        round(max(d["dist_m"] for d in xs_data), 2),
            "centroid_raw_e": round(ce, 2),
            "centroid_raw_n": round(cn, 2),
        })

    return sections, detected_crs


# ═════════════════════════════════════════════════════════════════════
# Gauge node utilities
# ═════════════════════════════════════════════════════════════════════

def load_gauge_nodes() -> tuple[list, list, np.ndarray]:
    df    = pd.read_csv(GRAPH_DIR / "nodes.csv")
    refs  = df["ref"].astype(str).tolist()
    names = df["name"].tolist() if "name" in df.columns else refs
    if "easting_itm" in df.columns:
        coords = df[["easting_itm", "northing_itm"]].values.astype(float)
    else:
        t = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        E, N = t.transform(df["lon"].values, df["lat"].values)
        coords = np.column_stack([E, N]).astype(float)
    return refs, names, coords


def load_datum_lookup(datum_csv: Path | None) -> dict[str, float]:
    if datum_csv is None or not datum_csv.exists():
        # Try auto-detect from station_data directories
        auto = BASE_DIR / "dataset/station_data"
        lookup: dict[str, float] = {}
        if auto.exists():
            for sub in auto.iterdir():
                if not sub.is_dir():
                    continue
                for fp in sub.glob("*Datum*"):
                    try:
                        df = pd.read_csv(fp, comment="#", skipinitialspace=True,
                                         encoding="utf-8-sig")
                        df.columns = df.columns.str.strip()
                        vcol = next((c for c in df.columns
                                     if "value" in c.lower()), None)
                        if vcol:
                            vals = pd.to_numeric(df[vcol], errors="coerce").dropna()
                            if len(vals):
                                lookup[sub.name] = float(vals.iloc[-1])
                    except Exception:
                        pass
        return lookup
    df = pd.read_csv(datum_csv)
    return {str(r["ref"]): float(r["datum_mOD"]) for _, r in df.iterrows()}


def load_mean_stage() -> dict[str, float]:
    """Training-set mean stage anomaly per node for bankfull_anomaly conversion."""
    X_path = PROC_DIR / "X.npy"
    if not X_path.exists():
        return {}
    X         = np.load(X_path, mmap_mode="r")
    train_end = int(X.shape[0] * 0.70)
    nodes_df  = pd.read_csv(GRAPH_DIR / "nodes.csv")
    refs      = nodes_df["ref"].astype(str).tolist()
    return {ref: float(np.nanmean(X[:train_end, i, 0]))
            for i, ref in enumerate(refs)}


# ═════════════════════════════════════════════════════════════════════
# Main batch runner
# ═════════════════════════════════════════════════════════════════════

def run(dat_dir: Path, datum_csv: Path | None,
        out_json: Path, out_csv: Path, max_dist_m: float):

    # ── 1. Find all .dat files ────────────────────────────────────────
    dat_files = sorted(
        list(dat_dir.glob("*.dat")) + list(dat_dir.glob("*.DAT"))
    )
    if not dat_files:
        print(f"No .dat files found in {dat_dir}")
        raise SystemExit(1)

    print(f"Found {len(dat_files)} .dat file(s) in {dat_dir}:")
    for f in dat_files:
        print(f"  {f.name}")

    # ── 2. Parse all files ────────────────────────────────────────────
    print(f"\n── Parsing ──")
    all_sections: list[dict] = []
    file_summaries: list[dict] = []

    for dat_path in dat_files:
        sections, crs = parse_dat(dat_path)

        if not sections:
            print(f"  {dat_path.name:30s}  0 sections — skipped")
            continue

        # Reproject centroids to ITM
        raw_e = np.array([s["centroid_raw_e"] for s in sections])
        raw_n = np.array([s["centroid_raw_n"] for s in sections])

        if crs is None:
            print(f"  {dat_path.name:30s}  WARNING: CRS undetected — skipping")
            continue

        itm_e, itm_n = to_itm(raw_e, raw_n, crs)
        for s, ie, in_ in zip(sections, itm_e, itm_n):
            s["itm_e"] = round(float(ie), 1)
            s["itm_n"] = round(float(in_), 1)
            s["source_crs"] = crs

        all_sections.extend(sections)
        bf_ok = sum(1 for s in sections if s["bankfull_mOD"])
        print(f"  {dat_path.name:30s}  {len(sections):3d} sections  "
              f"bankfull: {bf_ok}/{len(sections)}  CRS: {crs}")
        file_summaries.append({
            "file": dat_path.name,
            "n_sections": len(sections),
            "n_bankfull": bf_ok,
            "crs": crs,
        })

    print(f"\nTotal cross-sections loaded: {len(all_sections)}")

    if not all_sections:
        print("No sections parsed — check file formats.")
        raise SystemExit(1)

    # ── 3. Spatial match to gauge nodes ───────────────────────────────
    print(f"\n── Matching to gauge nodes (max {max_dist_m:.0f} m) ──")
    node_refs, node_names, node_coords = load_gauge_nodes()

    sec_coords = np.array([[s["itm_e"], s["itm_n"]] for s in all_sections])
    tree       = cKDTree(node_coords)
    dists, nearest_idx = tree.query(sec_coords)

    for s, dist_m, nidx in zip(all_sections, dists, nearest_idx):
        s["nearest_ref"]  = node_refs[nidx]
        s["nearest_name"] = node_names[nidx]
        s["dist_m"]       = round(float(dist_m), 1)
        s["assigned"]     = dist_m <= max_dist_m

    n_assigned = sum(1 for s in all_sections if s["assigned"])
    n_gauges   = len({s["nearest_ref"] for s in all_sections if s["assigned"]})
    print(f"  {n_assigned}/{len(all_sections)} sections assigned to "
          f"{n_gauges}/{len(node_refs)} gauges")

    unmatched_gauges = [r for r in node_refs
                        if r not in {s["nearest_ref"] for s in all_sections
                                     if s["assigned"]}]
    if unmatched_gauges:
        print(f"  Gauges with no nearby cross-section: {unmatched_gauges}")

    # ── 4. Best bankfull per gauge ─────────────────────────────────────
    print(f"\n── Deriving bankfull per gauge ──")
    datum_lookup     = load_datum_lookup(datum_csv)
    mean_stage_lookup = load_mean_stage()

    assigned = [s for s in all_sections if s["assigned"] and s["bankfull_mOD"]]

    # Group by gauge, pick section with smallest dist_m
    by_gauge: dict[str, dict] = {}
    for s in sorted(assigned, key=lambda x: x["dist_m"]):
        ref = s["nearest_ref"]
        if ref not in by_gauge:
            by_gauge[ref] = s

    xs_bankfull: dict[str, float] = {}

    print(f"\n  {'ref':>7}  {'gauge name':>25}  {'section':>18}  "
          f"{'dist(m)':>7}  {'bf_mOD':>8}  "
          f"{'datum':>7}  {'bf_stg':>7}  {'bf_anom':>8}")
    print("  " + "─" * 98)

    for ref, s in sorted(by_gauge.items()):
        bf_mOD = s["bankfull_mOD"]
        datum  = datum_lookup.get(ref)
        mean_s = mean_stage_lookup.get(ref, 0.0)

        if datum is None:
            bf_stg  = "—"
            bf_anom = "—"
            status  = " ⚠ no datum"
        else:
            bf_stage  = round(bf_mOD - datum, 4)
            bf_anomaly = round(bf_stage - mean_s, 4)
            bf_stg    = f"{bf_stage:.3f}"
            bf_anom   = f"{bf_anomaly:.3f}"
            xs_bankfull[ref] = bf_anomaly
            status = ""

        name = s["nearest_name"][:25]
        print(f"  {ref:>7}  {name:>25}  {s['name']:>18}  "
              f"{s['dist_m']:>7.1f}  {bf_mOD:>8.3f}  "
              f"{datum or 0:>7.3f}  {bf_stg:>7}  {bf_anom:>8}{status}")

    print(f"\n  Bankfull anomalies derived for {len(xs_bankfull)}/{len(by_gauge)} "
          f"matched gauges")
    if len(xs_bankfull) < len(by_gauge):
        missing = [r for r in by_gauge if r not in xs_bankfull]
        print(f"  Missing datums for: {missing}")
        print(f"  Add these to datum_lookup.csv or download DatumHistory files.")

    # ── 5. Save outputs ───────────────────────────────────────────────
    out_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "description": (
            "Bankfull stage anomaly estimates (m) from ISIS hydraulic model "
            "cross-sections. Panel boundary (*) markers set by hydraulic "
            "engineer during model calibration. Tier 0 in the bankfull hierarchy."
        ),
        "generated":          pd.Timestamp.now().isoformat(),
        "n_dat_files":        len(dat_files),
        "n_sections_total":   len(all_sections),
        "n_gauges_matched":   n_gauges,
        "file_summary":       file_summaries,
        "bankfull_stage_m":   xs_bankfull,
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Full report CSV
    report_rows = []
    for s in all_sections:
        report_rows.append({
            "source_file":    s["source_file"],
            "section":        s["name"],
            "chainage_m":     s["chainage_m"],
            "source_crs":     s.get("source_crs"),
            "itm_e":          s.get("itm_e"),
            "itm_n":          s.get("itm_n"),
            "thalweg_mOD":    s["thalweg_mOD"],
            "left_bank_mOD":  s["left_bank_mOD"],
            "right_bank_mOD": s["right_bank_mOD"],
            "bankfull_mOD":   s["bankfull_mOD"],
            "bankfull_depth_m": s["bankfull_depth_m"],
            "width_m":        s["width_m"],
            "nearest_ref":    s.get("nearest_ref"),
            "nearest_name":   s.get("nearest_name"),
            "dist_to_gauge_m": s.get("dist_m"),
            "assigned":       s.get("assigned", False),
        })
    pd.DataFrame(report_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # ── 6. Summary ────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Summary")
    print(f"{'═'*55}")
    print(f"  .dat files processed:     {len(dat_files)}")
    print(f"  Cross-sections total:     {len(all_sections)}")
    print(f"  Sections with bankfull:   "
          f"{sum(1 for s in all_sections if s['bankfull_mOD'])}")
    print(f"  Gauges matched:           {n_gauges}/{len(node_refs)}")
    print(f"  Bankfull anomalies saved: {len(xs_bankfull)}")
    print(f"\n  Next step:")
    print(f"  python src/derive_bankfull_thresholds.py")
    print(f"  (will load {out_json.name} as Tier 0 input)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Batch-parse ISIS .dat files and extract bankfull thresholds"
    )
    p.add_argument("--dat-dir",   type=Path, default=DAT_DIR,
                   help=f"Directory containing .dat files (default: {DAT_DIR})")
    p.add_argument("--datum-csv", type=Path, default=None,
                   help="CSV with [ref, datum_mOD] (auto-scanned if omitted)")
    p.add_argument("--out",       type=Path, default=XS_JSON)
    p.add_argument("--out-csv",   type=Path,
                   default=GRAPH_DIR / "isis_batch_report.csv")
    p.add_argument("--max-dist",  type=float, default=500.0,
                   help="Max distance (m) to assign section to gauge (default 500)")
    args = p.parse_args()

    if not args.dat_dir.exists():
        print(f"Directory not found: {args.dat_dir}")
        print(f"Expected: dataset/cross_sections/dats/")
        raise SystemExit(1)

    run(args.dat_dir, args.datum_csv,
        args.out, args.out_csv, args.max_dist)
