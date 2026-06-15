"""
build_sar_reference.py
═══════════════════════════════════════════════════════════════════════
Unzips Sentinel-1 SAFE packages, calibrates individual acquisitions,
and builds a pixel-wise median reference composite aligned to the
DEM ITM grid for use in flood change detection.

Downloaded file inventory
──────────────────────────
Flood image (recession phase, Nov 14):
  S1A  2025-11-14  orbit=23  06:47:51 UTC  ← change detection target

Reference images:
  S1A  2025-06-11  orbit=23  06:47:53 UTC  ✓ CONSISTENT GEOMETRY
  S1A  2025-06-23  orbit=23  06:47:53 UTC  ✓ CONSISTENT GEOMETRY
  S1C  2025-06-05  orbit=122 06:46:53 UTC  ⚠ different orbital slot
  S1C  2025-06-29  orbit=122 06:46:54 UTC  ⚠ different orbital slot

Processing strategy
────────────────────
1. Primary reference  — median of the two S1A orbit-23 images only.
   Identical geometry to the flood image → cleanest change detection.

2. Extended reference — median of all four images after reprojection
   to the common DEM grid.  Acceptable because after geocoding to ITM
   both S1A and S1C are on the same pixel grid.  The small geometric
   difference (~58 s orbital offset) produces sub-pixel misregistration
   (~150 m) which is acceptable at the 30 m DEM grid.

Median compositing
───────────────────
A pixel-wise median over N acquisitions suppresses:
  • Wind-roughened fields that appear temporarily bright or dark
  • Residual noise and speckle
  • Isolated rain events that lower backscatter on dry land
Two images give a simple mean; three or more give a true median.

Output
───────
  dataset/validation/processed/
      s1a_ref_primary_sigma0_itm.tif    — S1A-only median reference (orbit 23)
      s1_ref_extended_sigma0_itm.tif    — all-image median reference
      per-image calibrated GeoTIFFs aligned to the DEM grid

Usage
──────
  # Unzip, calibrate, and build both reference composites
  python src/build_sar_reference.py

  # Skip unzipping (already unzipped)
  python src/build_sar_reference.py --no-unzip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from rasterio.transform import from_gcps
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_DIR  = BASE_DIR / "dataset/validation"
OUT_DIR  = VAL_DIR / "processed"
DEM_PATH = BASE_DIR / "dataset/dem/COP-DEM-30m_itm.tif"

CALIBRATION_OFFSET_DB = -83.0   # approximate for IW GRDH when LUT unavailable

# Flood event dates — images acquired on these dates are flood images.
# All other processed images are treated as reference.
# Includes all DFC-GNN training events and the Nov 2025 validation event.
FLOOD_DATES = {
    # Validation event (Nov 2025)
    "20251111", "20251112", "20251113", "20251114",
    # DFC-GNN training events
    "20231020",   # lee_flood_oct2023  (stage 3.055 m)
    "20231227",   # lee_flood_dec2023  (stage 2.100 m)
    "20240108",   # lee_flood_jan2024  (stage 1.900 m)
    "20220218",   # lee_flood_mar2022  (stage 2.018 m)
    "20241127",   # lee_flood_nov2024  (stage 2.260 m)
}

# Minimum valid pixel coverage (%) — images below this are excluded
MIN_COVERAGE_PCT = 10.0
# Per-event reference date sets (for orbit-consistent composites)
EVENT_REF_MAP: dict[str, tuple[str, list[str]]] = {
    # flood_date: (event_name, [ref_dates])
    "20231020": ("lee_flood_oct2023",  ["20230610", "20230622"]),
    "20231227": ("lee_flood_dec2023",  ["20230606", "20230618", "20230630"]),
    "20240108": ("lee_flood_jan2024",  ["20230606", "20230618", "20230630"]),
    "20220218": ("lee_flood_mar2022",  ["20210605", "20210611", "20210617"]),
    "20241127": ("lee_flood_nov2024",  ["20240612", "20240624", "20240706"]),
    "20251111": ("validation_nov2025", ["20250602", "20250608", "20250614", "20250620"]),
}



# ═════════════════════════════════════════════════════════════════════
# Step 1: Unzip SAFE packages
# ═════════════════════════════════════════════════════════════════════

def unzip_safe(zip_path: Path, out_dir: Path) -> Path | None:
    """
    Unzip a Sentinel-1 .SAFE.zip into out_dir.
    Returns the .SAFE directory path or None if already extracted.
    """
    safe_name = zip_path.name.replace(".zip", "")
    safe_dir  = out_dir / safe_name

    if safe_dir.exists() and any(safe_dir.iterdir()):
        print(f"  Already extracted: {safe_name[:60]}")
        return safe_dir

    print(f"  Unzipping: {zip_path.name[:70]} ...")
    safe_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    if safe_dir.exists():
        print(f"  Extracted → {safe_dir.name[:60]}")
        return safe_dir

    # Some zips contain the SAFE as the top-level folder
    extracted = list(out_dir.glob("*.SAFE"))
    return extracted[0] if extracted else None


def find_measurement_tiff(safe_dir: Path, pol: str = "vv") -> Path | None:
    """Find the VV polarisation measurement TIFF inside a .SAFE directory."""
    pattern = f"*-{pol.lower()}-*.tiff"
    matches = list((safe_dir / "measurement").glob(pattern))
    if matches:
        return matches[0]
    # Fallback: search recursively
    matches = list(safe_dir.rglob(f"*{pol.lower()}*.tiff"))
    return matches[0] if matches else None


def find_calibration_xml(safe_dir: Path, pol: str = "vv") -> Path | None:
    """Find the calibration XML for a given polarisation."""
    pattern = f"calibration-*-{pol.lower()}-*.xml"
    cal_dir = safe_dir / "annotation" / "calibration"
    matches = list(cal_dir.glob(pattern)) if cal_dir.exists() else []
    return matches[0] if matches else None


# ═════════════════════════════════════════════════════════════════════
# Step 2: Calibrate and geocode one SAFE package
# ═════════════════════════════════════════════════════════════════════

def parse_lut(cal_xml: Path) -> np.ndarray | None:
    """Parse sigmaNought calibration LUT from annotation XML."""
    try:
        tree    = ET.parse(cal_xml)
        root    = tree.getroot()
        cal_el  = root.find(
            ".//calibrationVectorList/calibrationVector/sigmaNought")
        if cal_el is None:
            return None
        return np.array([float(v) for v in cal_el.text.split()])
    except Exception:
        return None


def calibrate(dn: np.ndarray, lut: np.ndarray | None) -> np.ndarray:
    """Convert raw DN to sigma0 in dB."""
    dn_f = dn.astype(np.float32)
    mask = dn_f == 0

    if lut is not None:
        if len(lut) != dn_f.shape[1]:
            x = np.linspace(0, 1, len(lut))
            lut = np.interp(np.linspace(0, 1, dn_f.shape[1]), x, lut)
        sigma0 = dn_f**2 / lut[np.newaxis, :]**2
    else:
        sigma0 = dn_f**2

    sigma0  = np.where(sigma0 <= 0, np.nan, sigma0)
    db      = 10.0 * np.log10(sigma0)
    if lut is None:
        db += CALIBRATION_OFFSET_DB
    db[mask] = np.nan
    return db


def geocode_to_dem(sigma0_db: np.ndarray, tiff_path: Path,
                   out_path: Path) -> bool:
    """
    Geocode calibrated sigma0 to the DEM ITM grid using embedded GCPs.
    Returns True on success.
    """
    with rasterio.open(tiff_path) as src:
        gcps, gcp_crs = src.gcps

    if not gcps:
        print(f"    ✗ No GCPs in {tiff_path.name}")
        return False

    gcp_transform = from_gcps(gcps)

    with rasterio.open(DEM_PATH) as dem:
        dst_crs = dem.crs
        dst_aff = dem.transform
        dst_h   = dem.height
        dst_w   = dem.width

    dst_array = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source        = sigma0_db,
        destination   = dst_array,
        src_transform = gcp_transform,
        src_crs       = gcp_crs,
        dst_transform = dst_aff,
        dst_crs       = dst_crs,
        resampling    = Resampling.bilinear,
        src_nodata    = np.nan,
        dst_nodata    = np.nan,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=dst_h, width=dst_w, count=1, dtype="float32",
        crs=dst_crs, transform=dst_aff, nodata=np.nan, compress="lzw"
    ) as dst:
        dst.write(dst_array, 1)

    valid     = ~np.isnan(dst_array)
    coverage  = valid.sum() / (dst_h * dst_w) * 100
    mean_db   = float(np.nanmean(dst_array))
    print(f"    Geocoded → {out_path.name}  "
          f"coverage={coverage:.1f}%  mean={mean_db:.2f} dB")

    if coverage < MIN_COVERAGE_PCT:
        print(f"    ⚠ Low coverage ({coverage:.1f}%) — excluding from composite")
        return False
    return True


def process_safe(safe_dir: Path, out_dir: Path,
                 role: str, pol: str = "vv") -> Path | None:
    """
    Calibrate and geocode one SAFE package.
    Returns path to the output ITM-aligned sigma0 GeoTIFF, or None.
    """
    date_str = safe_dir.name[17:25]   # YYYYMMDD from filename
    mission  = safe_dir.name[:3].lower()  # s1a or s1c
    # Determine role dynamically — no hardcoded date required
    auto_role = "flood" if date_str in FLOOD_DATES else "reference"
    out_name  = f"{mission}_{auto_role}_{date_str}_sigma0_itm.tif"
    out_path  = out_dir / out_name

    if out_path.exists():
        print(f"  Already processed: {out_name}")
        return out_path

    print(f"\n  Processing {safe_dir.name[:70]}")

    tiff = find_measurement_tiff(safe_dir, pol)
    if tiff is None:
        print(f"    ✗ No {pol.upper()} measurement TIFF found")
        return None

    print(f"    Measurement: {tiff.name}")

    cal_xml = find_calibration_xml(safe_dir, pol)
    lut     = parse_lut(cal_xml) if cal_xml else None
    if lut is not None:
        print(f"    Calibration LUT: {len(lut)} values from {cal_xml.name}")
    else:
        print(f"    Calibration LUT: not found — using offset {CALIBRATION_OFFSET_DB} dB")

    with rasterio.open(tiff) as src:
        dn = src.read(1)

    sigma0_db = calibrate(dn, lut)
    success   = geocode_to_dem(sigma0_db, tiff, out_path)
    return out_path if success else None


# ═════════════════════════════════════════════════════════════════════
# Step 3: Build median composite reference
# ═════════════════════════════════════════════════════════════════════

def build_median_composite(
    tif_paths: list[Path],
    out_path:  Path,
    label:     str,
) -> Path | None:
    """
    Pixel-wise median composite from N sigma0_dB GeoTIFFs.

    All inputs must be on the same grid (guaranteed by geocode_to_dem).
    Median suppresses wind-roughened surfaces, speckle, and local rain
    events that would bias the reference backscatter level.
    """
    if not tif_paths:
        print(f"  No images for {label} composite")
        return None

    print(f"\n  Building {label} composite from {len(tif_paths)} image(s):")
    for p in tif_paths:
        print(f"    {p.name}")

    # Load all images
    arrays = []
    ref_meta = None
    for p in tif_paths:
        with rasterio.open(p) as src:
            arrays.append(src.read(1).astype(np.float32))
            if ref_meta is None:
                ref_meta = src.meta.copy()

    stack = np.stack(arrays, axis=0)   # [N, H, W]

    if len(arrays) == 1:
        composite = arrays[0]
        method    = "single image (no compositing)"
    elif len(arrays) == 2:
        # Median of 2 = mean, but nan-safe
        composite = np.nanmean(stack, axis=0)
        method    = "mean of 2 images"
    else:
        composite = np.nanmedian(stack, axis=0)
        method    = f"pixel-wise median of {len(arrays)} images"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_meta.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(out_path, "w", **ref_meta) as dst:
        dst.write(composite, 1)

    valid    = ~np.isnan(composite)
    coverage = valid.sum() / composite.size * 100
    print(f"\n  Composite saved: {out_path.name}")
    print(f"    Method:   {method}")
    print(f"    Coverage: {coverage:.1f}%")
    print(f"    Mean dB:  {np.nanmean(composite):.2f}")
    print(f"    Std dB:   {np.nanstd(composite):.2f}")
    return out_path


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def run(no_unzip: bool = False, pol: str = "vv"):
    print("═"*60)
    print("  SAR Reference Composite Builder — Lee Catchment")
    print("═"*60)

    if not DEM_PATH.exists():
        print(f"✗ DEM not found: {DEM_PATH}")
        print("  Run precompute_hand_edges.py first.")
        return

    # ── 1. Find all zip files ────────────────────────────────────────
    zip_files = sorted(VAL_DIR.glob("*.SAFE.zip"))
    if not zip_files:
        print(f"No .SAFE.zip files found in {VAL_DIR}")
        return

    print(f"\n  Found {len(zip_files)} SAFE.zip file(s):")
    for z in zip_files:
        mb = z.stat().st_size / 1e6
        print(f"    {z.name[:70]}  ({mb:.0f} MB)")

    # ── 2. Unzip ─────────────────────────────────────────────────────
    safe_dirs: dict[str, Path] = {}   # date_str → safe_dir
    if not no_unzip:
        print(f"\n── Extracting SAFE packages ──")
        for z in zip_files:
            safe_dir = unzip_safe(z, VAL_DIR)
            if safe_dir:
                date_str = safe_dir.name[17:25]
                safe_dirs[date_str] = safe_dir
    else:
        for safe_dir in sorted(VAL_DIR.glob("*.SAFE")):
            date_str = safe_dir.name[17:25]
            safe_dirs[date_str] = safe_dir
        print(f"  Found {len(safe_dirs)} extracted SAFE dir(s)")

    if not safe_dirs:
        print("No SAFE directories found after unzipping")
        return

    # ── 3. Calibrate and geocode each image ───────────────────────────
    print(f"\n── Calibrating and geocoding ({pol.upper()} polarisation) ──")
    processed: dict[str, Path] = {}

    for date_str, safe_dir in safe_dirs.items():
        role = "flood" if date_str == FLOOD_DATES else "reference"
        out  = process_safe(safe_dir, OUT_DIR, role, pol)
        if out:
            processed[date_str] = out

    if not processed:
        print("No images successfully processed")
        return

    # ── 4. Separate flood and reference images ───────────────────────
    print(f"\n── Building per-event reference composites ──")

    flood_paths = {d: p for d, p in processed.items() if d in FLOOD_DATES}
    ref_paths   = {d: p for d, p in processed.items() if d not in FLOOD_DATES}

    print(f"  Flood images:     {len(flood_paths)}: {sorted(flood_paths.keys())}")
    print(f"  Reference images: {len(ref_paths)}: {sorted(ref_paths.keys())}")

    # ── 5. Build one reference composite per event ───────────────────
    # Uses EVENT_REF_MAP to select only orbit-consistent references.
    # This prevents cross-orbit geometry mixing in the composite.
    event_composites: dict[str, Path | None] = {}

    for flood_date, flood_path in sorted(flood_paths.items()):
        if flood_date not in EVENT_REF_MAP:
            print(f"  {flood_date}: no EVENT_REF_MAP entry — skipping composite")
            continue

        event_name, ref_dates = EVENT_REF_MAP[flood_date]
        ref_list = [ref_paths[d] for d in ref_dates if d in ref_paths]

        if not ref_list:
            print(f"  {event_name}: no reference images available in processed/")
            event_composites[event_name] = None
            continue

        composite_out = OUT_DIR / f"s1_ref_{event_name}_sigma0_itm.tif"
        composite = build_median_composite(
            ref_list, composite_out,
            f"{event_name} reference ({len(ref_list)} images)")
        event_composites[event_name] = composite

    # ── 6. Summary ────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Summary")
    print(f"{'═'*60}")
    print(f"  Total images processed: {len(processed)}")
    print(f"  Flood images:           {len(flood_paths)}")
    print(f"  Reference composites:")
    for ev, comp in event_composites.items():
        status = "OK" if comp else "FAILED — no reference images"
        print(f"    {ev:<30} {status}")
    print()
    print("  Next steps:")
    print("    python src/data/extract_sar_wetness.py --verbose")
    print("    python src/train_dfc_gnn.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build SAR median reference composite for flood validation"
    )
    p.add_argument("--no-unzip", action="store_true",
                   help="Skip unzipping (use already-extracted .SAFE dirs)")
    p.add_argument("--pol",      default="vv", choices=["vv","vh"],
                   help="Polarisation channel (default: vv)")
    args = p.parse_args()
    run(no_unzip=args.no_unzip, pol=args.pol)
