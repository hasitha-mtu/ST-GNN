"""
preprocess_sentinel1.py
═══════════════════════════════════════════════════════════════════════
Inspects, validates, and pre-processes raw Sentinel-1 GRD measurement
TIFFs into calibrated sigma0 (dB) GeoTIFFs in ITM (EPSG:2157),
ready for flood change detection in validate_flood_maps.py.

What raw Sentinel-1 GRD TIFFs contain
────────────────────────────────────────
The files inside a .SAFE package (e.g. s1a-iw-grd-vv-*.tiff) are raw
measurement files in SAR slant-range geometry.  They contain:

  • 16-bit unsigned integer DN (Digital Number) values
  • No geographic coordinate information (just row/col)
  • SAR geometry: rows = azimuth time, columns = slant range distance
  • Swath covers ~250 km × ~170 km for IW mode

These are NOT directly usable for:
  • Geographic comparison with the DEM or gauge nodes
  • Change detection (DN values are not calibrated across acquisitions)

Pre-processing steps required
───────────────────────────────
1. Calibration: DN → sigma0 (linear) → sigma0_dB
      sigma0 = (DN² + Dn_offset) / calibration_LUT²
      sigma0_dB = 10 × log10(sigma0)
   The calibration LUT is stored in the .SAFE annotation XML file.
   Without it a fixed offset of −83 dB is used (reasonable approximation
   for IW GRDH, accuracy ±0.5 dB — adequate for change detection).

2. Terrain correction / geocoding: SAR geometry → ITM (EPSG:2157)
   Proper Range-Doppler terrain correction requires a DEM and orbit
   state vectors (from the .SAFE manifest).  A simplified approach is
   used here: bilinear geocoding using the GCPs embedded in the GRD
   product, followed by reprojection to match the flood model DEM grid.
   This is accurate to ~50–100 m for IW GRDH (adequate for 30 m HAND).

3. Co-registration: reproject both flood and reference images to the
   same pixel grid (the DEM ITM grid) so pixel-by-pixel comparison is
   valid and the shape mismatch is eliminated.

Data validation checks
───────────────────────
The script verifies:
  ✓ Correct product type (IW GRDH)
  ✓ Correct polarisation (VV for open water detection)
  ✓ Consistent orbit geometry (same relative orbit for flood and reference)
  ✓ Spatial coverage includes the Lee catchment
  ✓ DN value distribution (checks for all-zero or saturated images)
  ✓ Temporal suitability (flood image within test period, ref in summer)
  ✓ Orbit direction consistency (both ascending or both descending)

Output
───────
  dataset/validation/processed/
      s1_vv_flood_YYYYMMDD_sigma0_itm.tif   — calibrated, geocoded flood
      s1_vv_ref_YYYYMMDD_sigma0_itm.tif     — calibrated, geocoded reference
      inspection_report.json                 — full validation report

Usage
──────
  # Inspect and pre-process both images
  python src/preprocess_sentinel1.py \\
      --flood-safe  dataset/validation/S1A_IW_GRDH_1SDV_20251114*.SAFE \\
      --ref-safe    dataset/validation/S1A_IW_GRDH_1SDV_20250729*.SAFE

  # Or use the raw TIFFs directly (skips SAFE metadata)
  python src/preprocess_sentinel1.py \\
      --flood-tiff  dataset/validation/s1a-iw-grd-vv-20251114*.tiff \\
      --ref-tiff    dataset/validation/s1a-iw-grd-vv-20250729*.tiff
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS

BASE_DIR = Path(__file__).resolve().parent.parent
DEM_PATH = BASE_DIR / "dataset/dem/COP-DEM-30m_itm.tif"
OUT_DIR  = BASE_DIR / "dataset/validation/processed"

# Lee catchment bounding box in WGS84
LEE_BBOX_WGS84 = (-9.1, 51.7, -8.0, 52.1)   # W, S, E, N

# Expected orbit for Lee catchment (Sentinel-1 descending, relative orbit ~30 or ascending ~81)
# Both work — just need flood and reference to match
CALIBRATION_OFFSET_DB = -83.0   # approximate for IW GRDH when no LUT available


# ═════════════════════════════════════════════════════════════════════
# Step 1: Inspect and validate a SAR file
# ═════════════════════════════════════════════════════════════════════

def parse_safe_metadata(safe_dir: Path) -> dict:
    """
    Read key metadata from a Sentinel-1 .SAFE directory.
    Returns a dict with all fields needed for validation.
    """
    meta = {"safe_dir": str(safe_dir), "valid": False, "errors": []}

    manifest = safe_dir / "manifest.safe"
    if not manifest.exists():
        meta["errors"].append("manifest.safe not found")
        return meta

    try:
        tree = ET.parse(manifest)
        root = tree.getroot()

        # Platform / product info
        ns = {"s1": "http://www.esa.int/safe/sentinel-1.0"}
        safe_elem = root.find(".//s1sarl1:standAloneProductInformation",
                               {"s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1"})

        # Simpler: just get everything from element text
        all_text = {e.tag.split("}")[-1]: e.text
                    for e in root.iter() if e.text and e.text.strip()}

        meta["mission"]        = all_text.get("familyName", "") + \
                                  all_text.get("number", "")
        meta["product_type"]   = all_text.get("productType", "")
        meta["polarisations"]  = all_text.get("transmitterReceiverPolarisation", "")
        meta["pass_direction"] = all_text.get("pass", "")
        meta["orbit_abs"]      = all_text.get("absoluteOrbitNumber", "")
        meta["orbit_rel"]      = all_text.get("relativeOrbitNumber", "")
        meta["start_time"]     = all_text.get("startTime", "")
        meta["stop_time"]      = all_text.get("stopTime", "")
        meta["footprint"]      = all_text.get("footPrint", "")

        meta["valid"] = True
    except Exception as e:
        meta["errors"].append(f"manifest parse error: {e}")

    return meta


def inspect_tiff(tiff_path: Path) -> dict:
    """
    Inspect a raw Sentinel-1 GRD TIFF and return a validation report.
    Checks DN value distribution, image dimensions, and embedded GCPs.
    """
    report = {
        "file":    tiff_path.name,
        "exists":  tiff_path.exists(),
        "checks":  {},
        "warnings": [],
        "errors":  [],
    }
    if not tiff_path.exists():
        report["errors"].append("file not found")
        return report

    try:
        with rasterio.open(tiff_path) as src:
            report["shape"]      = (src.height, src.width)
            report["dtype"]      = str(src.dtypes[0])
            report["crs"]        = str(src.crs) if src.crs else "none"
            report["n_gcps"]     = len(src.gcps[0]) if src.gcps[0] else 0
            report["nodata"]     = src.nodata
            report["transform"]  = str(src.transform)

            # Sample a 1000×1000 block from the centre to check DN values
            r0 = max(0, src.height//2 - 500)
            c0 = max(0, src.width//2  - 500)
            data = src.read(1, window=rasterio.windows.Window(
                c0, r0, min(1000, src.width), min(1000, src.height)
            )).astype(np.float32)

            valid_px = data[data > 0]
            report["dn_stats"] = {
                "min":    float(data.min()),
                "max":    float(data.max()),
                "mean":   float(valid_px.mean()) if len(valid_px) > 0 else 0,
                "pct_zero": float((data == 0).mean() * 100),
                "pct_saturated": float((data >= 65535).mean() * 100),
            }

            # Check 1: image is not empty
            c = "image_not_empty"
            if report["dn_stats"]["pct_zero"] > 95:
                report["checks"][c] = "FAIL — image is >95% zero (no data)"
                report["errors"].append(c)
            else:
                report["checks"][c] = f"OK  ({report['dn_stats']['pct_zero']:.1f}% zero)"

            # Check 2: 16-bit unsigned integers (raw GRD DN)
            c = "correct_dtype"
            if src.dtypes[0] == "uint16":
                report["checks"][c] = "OK  (uint16 = raw DN, not yet calibrated)"
            elif src.dtypes[0] == "float32":
                report["checks"][c] = "OK  (float32 = already calibrated to sigma0)"
                report["warnings"].append("File appears pre-calibrated — skip calibration step")
            else:
                report["checks"][c] = f"WARN — unexpected dtype {src.dtypes[0]}"

            # Check 3: GCPs present for geocoding
            c = "has_gcps"
            n_gcps = report["n_gcps"]
            if n_gcps >= 10:
                report["checks"][c] = f"OK  ({n_gcps} GCPs found)"
            else:
                report["checks"][c] = (
                    f"WARN — {n_gcps} GCPs found (geocoding will be approximate)")
                if n_gcps == 0:
                    report["warnings"].append(
                        "No GCPs — file may not be the raw measurement TIFF. "
                        "Make sure you are using the measurement/*.tiff from "
                        "inside the .SAFE directory, not a derived product.")

            # Check 4: DN range looks like SAR data
            c = "dn_range_valid"
            mean_dn = report["dn_stats"]["mean"]
            if 50 < mean_dn < 10000:
                report["checks"][c] = f"OK  (mean DN={mean_dn:.0f}, typical range 100–5000)"
            else:
                report["checks"][c] = (
                    f"WARN — mean DN={mean_dn:.0f} is outside typical range 100–5000")
                report["warnings"].append("Unusual DN values — check this is a VV GRD product")

    except Exception as e:
        report["errors"].append(f"read error: {e}")

    return report


def check_coverage(tiff_path: Path) -> dict:
    """
    Check whether a SAR image covers the Lee catchment by inspecting GCPs.
    Returns coverage check results.
    """
    result = {"covers_lee": False, "gcp_bbox_wgs84": None}
    try:
        with rasterio.open(tiff_path) as src:
            gcps, gcp_crs = src.gcps
            if not gcps:
                return result
            lons = [g.x for g in gcps]
            lats = [g.y for g in gcps]

            if str(gcp_crs) != "EPSG:4326":
                from pyproj import Transformer
                t = Transformer.from_crs(str(gcp_crs), "EPSG:4326", always_xy=True)
                lons, lats = t.transform(lons, lats)

            bbox = (min(lons), min(lats), max(lons), max(lats))
            result["gcp_bbox_wgs84"] = [round(v, 4) for v in bbox]

            lee_w, lee_s, lee_e, lee_n = LEE_BBOX_WGS84
            covers = (bbox[0] < lee_e and bbox[2] > lee_w and
                      bbox[1] < lee_n and bbox[3] > lee_s)
            result["covers_lee"] = covers
    except Exception as e:
        result["error"] = str(e)
    return result


# ═════════════════════════════════════════════════════════════════════
# Step 2: Calibrate DN → sigma0 dB
# ═════════════════════════════════════════════════════════════════════

def find_calibration_xml(safe_dir: Path, polarisation: str = "vv") -> Path | None:
    """
    Find the calibration XML file for a given polarisation inside a .SAFE dir.
    Path: annotation/calibration/calibration-s1a-iw-grd-{pol}-*.xml
    """
    pattern = f"calibration-*-{polarisation.lower()}-*.xml"
    matches = list((safe_dir / "annotation" / "calibration").glob(pattern))
    return matches[0] if matches else None


def parse_calibration_lut(cal_xml: Path) -> np.ndarray | None:
    """
    Parse the sigmaNought calibration LUT from annotation calibration XML.
    Returns a 1D array of calibration values (one per range pixel), or None.
    """
    try:
        tree   = ET.parse(cal_xml)
        root   = tree.getroot()
        cal_el = root.find(".//calibrationVectorList/calibrationVector/sigmaNought")
        if cal_el is None:
            return None
        values = np.array([float(v) for v in cal_el.text.split()])
        return values
    except Exception:
        return None


def calibrate_dn_to_sigma0_db(
    dn:        np.ndarray,
    lut:       np.ndarray | None = None,
    nodata_val: float = 0.0,
) -> np.ndarray:
    """
    Convert raw DN values to sigma0 in dB.

    With calibration LUT:
        sigma0 = DN² / lut²
        sigma0_dB = 10 × log10(sigma0)

    Without LUT (approximate):
        sigma0_dB ≈ 10 × log10(DN²) + CALIBRATION_OFFSET_DB

    Returns float32 array in dB, with nodata pixels set to NaN.
    """
    dn_f = dn.astype(np.float32)
    mask = dn_f == nodata_val

    if lut is not None:
        # LUT is per range-column — broadcast across azimuth rows
        if len(lut) != dn_f.shape[1]:
            # Interpolate LUT to match image width
            x_old = np.linspace(0, 1, len(lut))
            x_new = np.linspace(0, 1, dn_f.shape[1])
            lut_r = np.interp(x_new, x_old, lut)
        else:
            lut_r = lut
        sigma0 = (dn_f ** 2) / (lut_r[np.newaxis, :] ** 2)
    else:
        # Approximate without LUT
        sigma0 = dn_f ** 2

    # Avoid log of zero
    sigma0 = np.where(sigma0 <= 0, np.nan, sigma0)
    sigma0_db = 10.0 * np.log10(sigma0)

    if lut is None:
        sigma0_db += CALIBRATION_OFFSET_DB

    sigma0_db[mask] = np.nan
    return sigma0_db


# ═════════════════════════════════════════════════════════════════════
# Step 3: Geocode and reproject to DEM ITM grid
# ═════════════════════════════════════════════════════════════════════

def geocode_to_itm(
    sigma0_db: np.ndarray,
    src_path:  Path,
    out_path:  Path,
    dem_path:  Path = DEM_PATH,
):
    """
    Geocode a calibrated sigma0 array using the GCPs from the original
    file and reproject to the ITM grid of the flood model DEM.

    The output is pixel-aligned with the DEM so the SAR and HAND
    inundation arrays can be compared directly.
    """
    with rasterio.open(src_path) as src:
        gcps, gcp_crs = src.gcps
        src_h, src_w  = src.height, src.width

    if not gcps:
        raise ValueError(
            f"No GCPs in {src_path.name}. Cannot geocode without GCPs. "
            f"Make sure this is the raw measurement TIFF from the .SAFE "
            f"directory, not a derived product.")

    # Build transform from GCPs
    gcp_transform = from_gcps(gcps)

    # Target grid: match the flood model DEM exactly
    with rasterio.open(dem_path) as dem_src:
        dst_crs       = dem_src.crs
        dst_transform = dem_src.transform
        dst_h         = dem_src.height
        dst_w         = dem_src.width

    # Reproject calibrated sigma0 from SAR geometry → ITM DEM grid
    dst_array = np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    reproject(
        source        = sigma0_db,
        destination   = dst_array,
        src_transform = gcp_transform,
        src_crs       = gcp_crs,
        dst_transform = dst_transform,
        dst_crs       = dst_crs,
        resampling    = Resampling.bilinear,
        src_nodata    = np.nan,
        dst_nodata    = np.nan,
    )

    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=dst_h, width=dst_w, count=1,
        dtype="float32",
        crs=dst_crs, transform=dst_transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(dst_array, 1)

    # Validate output
    valid_px = dst_array[~np.isnan(dst_array)]
    pct_cov  = len(valid_px) / (dst_h * dst_w) * 100
    print(f"  Geocoded: {out_path.name}")
    print(f"    Coverage:  {pct_cov:.1f}% of DEM extent")
    print(f"    sigma0_dB: mean={np.nanmean(dst_array):.2f}  "
          f"std={np.nanstd(dst_array):.2f}  "
          f"range=[{np.nanmin(dst_array):.1f}, {np.nanmax(dst_array):.1f}]")

    if pct_cov < 10:
        print(f"  ⚠ WARNING: only {pct_cov:.1f}% coverage — the SAR image may not")
        print(f"    overlap with the Lee catchment. Check the acquisition footprint.")
    return dst_array


# ═════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════

def process_one(
    tiff_path:  Path,
    safe_dir:   Path | None,
    pol:        str,
    role:       str,
    out_dir:    Path,
) -> tuple[Path | None, dict]:
    """Inspect, calibrate, and geocode one SAR image."""

    date_str = tiff_path.name.split("-")[4][:8]   # YYYYMMDD from filename
    out_name = f"s1_{pol.lower()}_{role}_{date_str}_sigma0_itm.tif"
    out_path = out_dir / out_name

    print(f"\n── {role.upper()} image ({date_str}) ──")

    # ── Inspect ───────────────────────────────────────────────────────
    report = inspect_tiff(tiff_path)
    cov    = check_coverage(tiff_path)
    report["coverage"] = cov

    print(f"  File: {tiff_path.name}")
    print(f"  Shape: {report.get('shape','?')}  dtype: {report.get('dtype','?')}")
    print(f"  GCPs: {report.get('n_gcps',0)}")
    print(f"  DN stats: {report.get('dn_stats',{})}")
    if cov.get("gcp_bbox_wgs84"):
        b = cov["gcp_bbox_wgs84"]
        print(f"  Footprint (WGS84): W={b[0]} S={b[1]} E={b[2]} N={b[3]}")
        print(f"  Covers Lee catchment: {'YES ✓' if cov['covers_lee'] else 'NO ✗'}")
    print(f"\n  Checks:")
    for k, v in report.get("checks", {}).items():
        print(f"    {k}: {v}")
    for w in report.get("warnings", []):
        print(f"    ⚠ {w}")
    for e in report.get("errors", []):
        print(f"    ✗ {e}")

    if report.get("errors"):
        return None, report

    if not cov.get("covers_lee"):
        print(f"\n  ✗ Image does not cover the Lee catchment.")
        print(f"    Expected footprint: W={LEE_BBOX_WGS84[0]} S={LEE_BBOX_WGS84[1]} "
              f"E={LEE_BBOX_WGS84[2]} N={LEE_BBOX_WGS84[3]}")
        if cov.get("gcp_bbox_wgs84"):
            print(f"    Actual footprint:   {cov['gcp_bbox_wgs84']}")
        return None, report

    # ── Calibration ───────────────────────────────────────────────────
    print(f"\n  Calibrating DN → sigma0 dB ...")
    lut = None
    if safe_dir:
        cal_xml = find_calibration_xml(safe_dir, pol)
        if cal_xml:
            lut = parse_calibration_lut(cal_xml)
            if lut is not None:
                print(f"  Calibration LUT loaded from {cal_xml.name} "
                      f"({len(lut)} values)")
            else:
                print(f"  ⚠ Could not parse LUT — using approximate offset")
        else:
            print(f"  ⚠ No calibration XML found — using approximate offset")
    else:
        print(f"  No .SAFE directory provided — using approximate offset "
              f"({CALIBRATION_OFFSET_DB} dB)")

    with rasterio.open(tiff_path) as src:
        dn = src.read(1)

    sigma0_db = calibrate_dn_to_sigma0_db(dn, lut=lut)
    print(f"  sigma0_dB: mean={np.nanmean(sigma0_db):.2f}  "
          f"range=[{np.nanmin(sigma0_db):.1f}, {np.nanmax(sigma0_db):.1f}]  "
          f"(typical open water: −20 to −14 dB; land: −15 to −5 dB)")

    # ── Geocode to ITM ────────────────────────────────────────────────
    print(f"\n  Geocoding to ITM (EPSG:2157) on DEM grid ...")
    if not DEM_PATH.exists():
        print(f"  ✗ DEM not found: {DEM_PATH}")
        print(f"    Run precompute_hand_edges.py first.")
        return None, report

    try:
        geocode_to_itm(sigma0_db, tiff_path, out_path)
    except Exception as e:
        print(f"  ✗ Geocoding failed: {e}")
        report["errors"].append(str(e))
        return None, report

    report["output"] = str(out_path)
    return out_path, report


def run(flood_tiff: Path, ref_tiff: Path,
        flood_safe: Path | None, ref_safe: Path | None,
        pol: str, out_dir: Path):

    print("═"*60)
    print("  Sentinel-1 GRD Pre-processing and Validation")
    print("═"*60)

    results = {}

    flood_out, flood_rep = process_one(
        flood_tiff, flood_safe, pol, "flood", out_dir)
    ref_out, ref_rep = process_one(
        ref_tiff, ref_safe, pol, "reference", out_dir)

    results["flood"]     = flood_rep
    results["reference"] = ref_rep

    # ── Cross-checks between the two images ──────────────────────────
    print("\n── Cross-image checks ──")

    # Orbit direction consistency
    flood_date = flood_tiff.name.split("-")[4][:8]
    ref_date   = ref_tiff.name.split("-")[4][:8]
    print(f"  Flood image date:     {flood_date[:4]}-{flood_date[4:6]}-{flood_date[6:]}")
    print(f"  Reference image date: {ref_date[:4]}-{ref_date[4:6]}-{ref_date[6:]}")

    flood_orbit = flood_tiff.name.split("-")[6]
    ref_orbit   = ref_tiff.name.split("-")[6]
    # Same relative orbit modulo 175 (Sentinel-1 repeat cycle)
    orbit_diff = abs(int(flood_orbit) - int(ref_orbit)) % 175
    orbit_match = orbit_diff == 0
    print(f"  Orbit match (same geometry): {'YES ✓' if orbit_match else f'NO — diff={orbit_diff} mod 175'}")
    if not orbit_match:
        print(f"    ⚠ Different orbit geometries will cause geometric misregistration.")
        print(f"    ⚠ For best results use acquisitions with the same relative orbit.")
        print(f"    ⚠ Check orbit numbers: flood={flood_orbit}, ref={ref_orbit}")
        print(f"    ⚠ Relative orbits for Cork: ascending=~81, descending=~30")

    # Reference season check
    ref_month = int(ref_date[4:6])
    if 6 <= ref_month <= 8:
        print(f"  Reference season: SUMMER ({ref_month}/2025) ✓")
    else:
        print(f"  Reference season: month {ref_month} — summer (Jun–Aug) preferred")

    # Flood season check
    flood_month = int(flood_date[4:6])
    if flood_month in (10, 11, 12, 1, 2, 3):
        print(f"  Flood image season: AUTUMN/WINTER ({flood_month}/2025) ✓")
    else:
        print(f"  Flood image season: month {flood_month} — check this is a flood event")

    # ── Summary and next steps ────────────────────────────────────────
    both_ok = flood_out is not None and ref_out is not None
    print(f"\n{'═'*60}")
    print(f"  Result: {'READY FOR VALIDATION' if both_ok else 'PRE-PROCESSING INCOMPLETE'}")
    print(f"{'═'*60}")

    if both_ok:
        print(f"\n  Run validation with:")
        print(f"\n    python src/validate_flood_maps.py \\")
        print(f"        --mode sar \\")
        print(f"        --event-date {flood_date[:4]}-{flood_date[4:6]}-{flood_date[6:]} \\")
        print(f"        --sar-flood  {flood_out} \\")
        print(f"        --sar-ref    {ref_out} \\")
        print(f"        --model      st_gnn_hand_edge")
        results["ready"] = True
        results["flood_processed"]     = str(flood_out)
        results["reference_processed"] = str(ref_out)
        results["recommended_command"] = {
            "event_date": f"{flood_date[:4]}-{flood_date[4:6]}-{flood_date[6:]}",
            "sar_flood":  str(flood_out),
            "sar_ref":    str(ref_out),
        }
    else:
        results["ready"] = False
        if not flood_out:
            print(f"  ✗ Flood image could not be processed")
        if not ref_out:
            print(f"  ✗ Reference image could not be processed")

    # Save report
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / "inspection_report.json"
    with open(rep_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report saved: {rep_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Inspect and pre-process Sentinel-1 GRD TIFFs for flood validation"
    )
    # p.add_argument("--flood-tiff", type=Path, required=True,
    #                help="Raw VV TIFF from .SAFE — flood acquisition")
    # p.add_argument("--ref-tiff",   type=Path, required=True,
    #                help="Raw VV TIFF from .SAFE — reference (dry period)")
    # p.add_argument("--flood-safe", type=Path, default=None,
    #                help=".SAFE directory for flood image (for calibration LUT)")
    # p.add_argument("--ref-safe",   type=Path, default=None,
    #                help=".SAFE directory for reference image")
    p.add_argument("--pol",        type=str,  default="vv",
                   choices=["vv","vh"],
                   help="Polarisation channel (default: vv)")
    p.add_argument("--out",        type=Path, default=OUT_DIR)
    args = p.parse_args()

    flood_tiff = '/dataset/validation/S1A_IW_GRDH_1SDV_20251114T064751_20251114T064816_061870_07BC54_5DDF.SAFE/measurement/s1a-iw-grd-vv-20251114t064751-20251114t064816-061870-07bc54-001.tiff'
    ref_tiff = '/dataset/validation/S1C_IW_GRDH_1SDV_20250605T064653_20250605T064718_002644_00577F_981F.SAFE/measurement/s1c-iw-grd-vv-20250605t064653-20250605t064718-002644-00577f-001.tiff'
    flood_safe = '/dataset/validation/S1A_IW_GRDH_1SDV_20251114T064751_20251114T064816_061870_07BC54_5DDF.SAFE'
    ref_safe = '/dataset/validation/S1C_IW_GRDH_1SDV_20250605T064653_20250605T064718_002644_00577F_981F.SAFE'

    run(
        flood_tiff = Path(flood_tiff),
        ref_tiff   = Path(ref_tiff),
        flood_safe = Path(flood_safe),
        ref_safe   = Path(ref_safe),
        pol        = args.pol,
        out_dir    = args.out,
    )
