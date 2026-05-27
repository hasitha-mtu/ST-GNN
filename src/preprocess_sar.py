"""
preprocess_sar.py  –  Sentinel-1 GRD production preprocessing for River Lee PI-ST-GNN
=======================================================================================
Converts raw Sentinel-1 IW GRDH .SAFE archives into calibrated, terrain-corrected,
ITM-projected GeoTIFF rasters ready for the SARFNOEncoder.

Processing chain (via pyroSAR + ESA SNAP):
    Read → Apply-Orbit-File → ThermalNoiseRemoval → Calibration (σ⁰)
        → Terrain-Correction (Range-Doppler, external DTM)
        → LinearToFromdB → Write (GeoTIFF, ITM / EPSG:2157)

Requirements
------------
    conda install -c conda-forge gdal "numpy<2" rasterio pyproj geopandas
    pip install pyroSAR
    ESA SNAP installed (https://step.esa.int/main/download/snap-download/)

Usage
-----
    # Derive bbox from catchment shapefile (recommended)
    python preprocess_sar.py \\
        --input     /data/sentinel1/ \\
        --outdir    dataset/sar \\
        --dtm       dataset/dem/lee_dtm_30m.tif \\
        --shapefile shapefiles/Lee-catchment/Lee-catchment.shp \\
        --buffer    5000

    # Derive bbox from gauge node coordinates
    python preprocess_sar.py \\
        --input     /data/sentinel1/ \\
        --outdir    dataset/sar \\
        --dtm       dataset/dem/lee_dtm_30m.tif \\
        --nodes-csv dataset/graph/nodes.csv \\
        --buffer    5000

    # Check environment without processing
    python preprocess_sar.py --check \\
        --input /data/sentinel1/ --outdir dataset/sar --dtm dataset/dem/lee_dtm_30m.tif

Bbox resolution priority
------------------------
    1. --shapefile   catchment polygon (most accurate)
    2. --nodes-csv   gauge node coordinate envelope
    3. --bbox        explicit ITM values  (xmin ymin xmax ymax)
    4. built-in Lee default: 501400 554700 578200 593900

Output files per scene
----------------------
    <outdir>/<event_id>.tif       2-band GeoTIFF (VV dB, VH dB), float32, ITM
    <outdir>/<event_id>.npy       [2, H, W] float32 array for FNO encoder
    <outdir>/sar_events.json      Event registry (update event_start/end manually)
    <outdir>/processing_log.json  Per-scene QC record

sar_events.json schema
----------------------
    {
      "event_20241110_6ECE": {
        "sar_path":         "event_20241110_6ECE.npy",
        "acquisition_date": "2024-11-10",
        "event_start":      "2024-11-07T00:00",   ← UPDATE to match flood record
        "event_end":        "2024-11-13T23:45",   ← UPDATE to match flood record
        "bbox_itm":         [501400, 554700, 578200, 593900],
        "scene_id":         "S1A_IW_GRDH_1SDV_...",
        "vv_mean_db":       -12.3,
        "water_fraction_vv16": 0.043
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ── Optional imports (checked at runtime) ─────────────────────────────
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# ── pyroSAR: try all known import paths for geocode() ─────────────────
# The function moved between submodules across versions:
#   v0.11–0.13:  pyroSAR.snap
#   v0.14–0.18:  pyroSAR.snap.util
#   v0.19+:      pyroSAR.snap.auxil
_geocode  = None
_identify = None
HAS_PYROSAR = False
HAS_PYROSAR_ERROR = ""

try:
    from pyroSAR import identify as _identify
    for _path in ("pyroSAR.snap.auxil", "pyroSAR.snap.util", "pyroSAR.snap"):
        try:
            import importlib as _il
            _mod     = _il.import_module(_path)
            _geocode = getattr(_mod, "geocode", None)
            if _geocode is not None:
                HAS_PYROSAR = True
                break
        except (ImportError, AttributeError):
            continue
    if not HAS_PYROSAR:
        import pyroSAR as _psr
        HAS_PYROSAR_ERROR = (
            f"pyroSAR v{getattr(_psr,'__version__','?')} installed but "
            "geocode() not found in snap.auxil / snap.util / snap."
        )
except ImportError as _e:
    HAS_PYROSAR_ERROR = str(_e)

# Thin wrappers so call sites look normal
def _run_geocode(*a, **kw):
    if _geocode is None:
        raise ImportError(HAS_PYROSAR_ERROR)
    return _geocode(*a, **kw)

def _run_identify(*a, **kw):
    if _identify is None:
        raise ImportError(HAS_PYROSAR_ERROR)
    return _identify(*a, **kw)


# ── Lee catchment fallback bbox (ITM metres) ──────────────────────────
_LEE_DEFAULT_BBOX = (501_400.0, 554_700.0, 578_200.0, 593_900.0)


# ═══════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════

def get_logger(name: str = "preprocess_sar") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


# ═══════════════════════════════════════════════════════════════════════
#  Bounding box derivation
# ═══════════════════════════════════════════════════════════════════════

def bbox_from_shapefile(
    shp_path: Path,
    buffer_m: float = 5000.0,
    logger: logging.Logger = None,
) -> tuple[float, float, float, float]:
    """
    Derive an ITM bounding box from a catchment shapefile.

    Reprojects to ITM (EPSG:2157) if needed, dissolves all features into
    one polygon, adds buffer_m on all sides, and rounds outward to the
    nearest 100 m for clean pixel alignment at 20 m resolution.
    """
    if not HAS_GEOPANDAS:
        raise ImportError("pip install geopandas  (required for --shapefile)")
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    if logger:
        logger.info("  Reading shapefile: %s", shp_path)

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"Shapefile {shp_path} is empty.")

    src_crs_str = gdf.crs.to_string() if gdf.crs else "undefined"
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=29902)  # assume Irish National Grid
    if gdf.crs.to_epsg() != 2157:
        if logger:
            logger.info("  Reprojecting %s -> EPSG:2157 (ITM)", src_crs_str)
        gdf = gdf.to_crs(epsg=2157)

    b = gdf.dissolve().total_bounds  # [xmin, ymin, xmax, ymax]
    xmin = float(np.floor((b[0] - buffer_m) / 100) * 100)
    ymin = float(np.floor((b[1] - buffer_m) / 100) * 100)
    xmax = float(np.ceil( (b[2] + buffer_m) / 100) * 100)
    ymax = float(np.ceil( (b[3] + buffer_m) / 100) * 100)

    if logger:
        logger.info("  Source CRS:       %s", src_crs_str)
        logger.info("  Feature count:    %d (dissolved to single polygon)", len(gdf))
        logger.info(
            "  Raw bounds (ITM): xmin=%.0f  ymin=%.0f  xmax=%.0f  ymax=%.0f",
            b[0], b[1], b[2], b[3],
        )
        logger.info("  Buffer:           +%.0f m all sides", buffer_m)
        logger.info(
            "  Final bbox (ITM): xmin=%.0f  ymin=%.0f  xmax=%.0f  ymax=%.0f",
            xmin, ymin, xmax, ymax,
        )
        logger.info(
            "  Extent:           %.1f km W-E  x  %.1f km S-N",
            (xmax - xmin) / 1000, (ymax - ymin) / 1000,
        )
        logger.info(
            "  Raster @ 20 m:    %d x %d px",
            int((xmax - xmin) / 20), int((ymax - ymin) / 20),
        )
    return xmin, ymin, xmax, ymax


def bbox_from_nodes_csv(
    csv_path: Path,
    buffer_m: float = 5000.0,
    x_col:    str   = "easting_itm",
    y_col:    str   = "northing_itm",
    logger: logging.Logger = None,
) -> tuple[float, float, float, float]:
    """Derive ITM bbox from gauge node coordinate envelope in nodes.csv."""
    import csv as _csv
    if not csv_path.exists():
        raise FileNotFoundError(f"nodes.csv not found: {csv_path}")

    xs, ys = [], []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        fields = reader.fieldnames or []
        if x_col not in fields:
            raise KeyError(
                f"Column '{x_col}' not in {csv_path}. "
                f"Available: {fields}. Use --x-col."
            )
        if y_col not in fields:
            raise KeyError(
                f"Column '{y_col}' not in {csv_path}. "
                f"Available: {fields}. Use --y-col."
            )
        for row in reader:
            try:
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
            except (ValueError, TypeError):
                continue

    if len(xs) < 2:
        raise ValueError(f"Fewer than 2 valid coordinates in {csv_path}.")

    xmin = float(np.floor((min(xs) - buffer_m) / 100) * 100)
    ymin = float(np.floor((min(ys) - buffer_m) / 100) * 100)
    xmax = float(np.ceil( (max(xs) + buffer_m) / 100) * 100)
    ymax = float(np.ceil( (max(ys) + buffer_m) / 100) * 100)

    if logger:
        logger.info("  Nodes read:    %d from %s", len(xs), csv_path)
        logger.info(
            "  Final bbox:    xmin=%.0f  ymin=%.0f  xmax=%.0f  ymax=%.0f",
            xmin, ymin, xmax, ymax,
        )
    return xmin, ymin, xmax, ymax


def resolve_bbox(
    shapefile: Path | None,
    nodes_csv: Path | None,
    bbox_arg:  list | None,
    buffer_m:  float,
    logger:    logging.Logger,
) -> tuple[float, float, float, float]:
    """Resolve bbox with priority: shapefile > nodes_csv > bbox > default."""
    if shapefile is not None:
        logger.info("── Bbox source: catchment shapefile ────────────────")
        return bbox_from_shapefile(shapefile, buffer_m, logger)
    if nodes_csv is not None:
        logger.info("── Bbox source: nodes.csv gauge coordinates ────────")
        return bbox_from_nodes_csv(nodes_csv, buffer_m, logger=logger)
    if bbox_arg is not None:
        bbox = tuple(float(v) for v in bbox_arg)
        logger.info("── Bbox source: explicit  %.0f  %.0f  %.0f  %.0f", *bbox)
        return bbox
    logger.warning(
        "No bbox source supplied — using Lee default: "
        "%.0f  %.0f  %.0f  %.0f", *_LEE_DEFAULT_BBOX
    )
    return _LEE_DEFAULT_BBOX


def validate_nodes_in_bbox(
    nodes_csv: Path,
    bbox_itm:  tuple,
    x_col:     str = "easting_itm",
    y_col:     str = "northing_itm",
    logger:    logging.Logger = None,
) -> bool:
    """Warn about gauge nodes that fall outside the processing bbox."""
    import csv as _csv
    if not nodes_csv.exists():
        return True
    xmin, ymin, xmax, ymax = bbox_itm
    outside, total = [], 0
    with open(nodes_csv, newline="") as f:
        reader = _csv.DictReader(f)
        fields = reader.fieldnames or []
        if x_col not in fields or y_col not in fields:
            return True
        for row in reader:
            try:
                x, y = float(row[x_col]), float(row[y_col])
            except (ValueError, TypeError):
                continue
            total += 1
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                outside.append((row.get("name", row.get("ref", "?")), x, y))
    if outside and logger:
        logger.warning(
            "  %d/%d node(s) outside bbox — will receive "
            "border-clamped SAR embeddings:", len(outside), total
        )
        for name, x, y in outside:
            logger.warning("    %-30s  E=%.0f  N=%.0f", name, x, y)
    elif logger:
        logger.info("  Node coverage: all %d nodes inside bbox  OK", total)
    return len(outside) == 0


# ═══════════════════════════════════════════════════════════════════════
#  SNAP discovery
# ═══════════════════════════════════════════════════════════════════════

def find_snap_home(logger: logging.Logger = None) -> str | None:
    """
    Locate the SNAP installation directory.

    Searches in order: SNAP_HOME env var → common install paths →
    PATH scan for the gpt executable.
    Sets os.environ["SNAP_HOME"] when found via path scan.
    """
    import shutil

    snap_home = os.environ.get("SNAP_HOME")
    if snap_home:
        p = Path(snap_home)
        if (p / "bin" / "gpt").exists() or (p / "bin" / "gpt.exe").exists():
            if logger:
                logger.info("  SNAP found via SNAP_HOME: %s", snap_home)
            return snap_home
        if logger:
            logger.warning("  SNAP_HOME=%s set but bin/gpt not found there.", snap_home)

    candidates = [
        # Windows
        r"C:\Program Files\snap",
        r"C:\snap",
        os.path.expanduser(r"~\AppData\Local\Programs\esa-snap"),
        os.path.expanduser(r"~\snap"),
        # Linux
        "/opt/snap",
        "/usr/local/snap",
        os.path.expanduser("~/snap"),
        # macOS
        "/Applications/snap",
        os.path.expanduser("~/Applications/snap"),
    ]
    for c in candidates:
        p = Path(c)
        if (p / "bin" / "gpt").exists() or (p / "bin" / "gpt.exe").exists():
            if logger:
                logger.info("  SNAP found at: %s", c)
            os.environ["SNAP_HOME"] = c
            return c

    gpt = shutil.which("gpt") or shutil.which("gpt.exe")
    if gpt:
        snap_home = str(Path(gpt).parent.parent)
        if logger:
            logger.info("  SNAP found via PATH (gpt): %s", snap_home)
        os.environ["SNAP_HOME"] = snap_home
        return snap_home

    if logger:
        logger.warning("  SNAP not found in env var, %d common paths, or PATH.", len(candidates))
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Coordinate conversion
# ═══════════════════════════════════════════════════════════════════════

def itm_bbox_to_wgs84_wkt(
    bbox_itm: tuple,
    logger:   logging.Logger = None,
) -> str:
    """
    Convert an ITM bbox to a WGS84 WKT closed-ring polygon for SNAP Subset.
    Uses pyproj for accurate conversion; linear approximation fallback.
    """
    xmin, ymin, xmax, ymax = bbox_itm

    if HAS_PYPROJ:
        t = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)
        corners = [t.transform(x, y) for x, y in [
            (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
        ]]
        if logger:
            logger.info(
                "  WGS84 corners (pyproj): SW=(%.4f,%.4f) NE=(%.4f,%.4f)",
                corners[0][0], corners[0][1], corners[2][0], corners[2][1],
            )
    else:
        # Linear approximation for Cork / Lee catchment region
        def _approx(x, y):
            return -8.0 + (x - 600_000) / 68_500, 53.5 + (y - 750_000) / 111_300
        corners = [_approx(x, y) for x, y in [
            (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
        ]]
        if logger:
            logger.warning("  pyproj not installed — using WGS84 linear approximation.")

    sw, se, ne, nw = corners
    return (
        f"POLYGON(("
        f"{sw[0]:.6f} {sw[1]:.6f}, "
        f"{se[0]:.6f} {se[1]:.6f}, "
        f"{ne[0]:.6f} {ne[1]:.6f}, "
        f"{nw[0]:.6f} {nw[1]:.6f}, "
        f"{sw[0]:.6f} {sw[1]:.6f}"
        f"))"
    )


# ═══════════════════════════════════════════════════════════════════════
#  SAFE archive utilities
# ═══════════════════════════════════════════════════════════════════════

def unzip_safe(zip_path: Path, out_dir: Path, logger: logging.Logger) -> Path:
    """
    Extract a .SAFE.zip archive. Handles double-extension (.SAFE.zip.zip).
    Returns the path to the extracted .SAFE directory.
    """
    name = zip_path.name
    while name.endswith(".zip"):
        name = name[:-4]
    if not name.endswith(".SAFE"):
        name += ".SAFE"
    safe_path = out_dir / name

    if safe_path.exists():
        logger.info("  .SAFE already extracted: %s", safe_path.name)
        return safe_path

    logger.info("  Extracting %s …", zip_path.name)
    t0 = time.perf_counter()
    with zipfile.ZipFile(zip_path, "r") as zf:
        if not any(".SAFE" in n for n in zf.namelist()):
            raise ValueError(
                f"{zip_path.name} does not contain a .SAFE structure. "
                f"First entries: {zf.namelist()[:5]}"
            )
        zf.extractall(out_dir)
    logger.info("  Extracted in %.1f s → %s", time.perf_counter() - t0, safe_path.name)
    return safe_path


def parse_scene_id(safe_path: Path) -> dict:
    """
    Extract metadata from the SAFE directory name.
    Format: S1[A/B/D]_IW_GRDH_1SDV_<start>_<stop>_<orbit>_<datatake>_<hash>.SAFE
    """
    parts = safe_path.stem.split("_")
    if len(parts) < 8:
        raise ValueError(f"Unexpected SAFE name: {safe_path.stem}")
    s = parts[4]  # YYYYMMDDTHHMMSS
    return {
        "satellite":  parts[0],
        "start_time": f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}",
        "orbit":      parts[6],
        "scene_id":   safe_path.stem,
    }


def classify_input_files(zip_files: list) -> tuple[list, list, list]:
    """
    Classify input files into GRD, SLC, and COG categories.
    Returns (grd_files, slc_files, cog_files).
    """
    grd, slc, cog = [], [], []
    for f in zip_files:
        n = f.name.upper()
        if "_COG" in n:
            cog.append(f)
        elif "SLC_" in n or "SLC__" in n:
            slc.append(f)
        elif "GRDH" in n or "GRDM" in n or "GRD_" in n:
            grd.append(f)
        else:
            grd.append(f)  # unknown — attempt, fail gracefully
    return grd, slc, cog


# ═══════════════════════════════════════════════════════════════════════
#  geocode() version-adaptive caller
# ═══════════════════════════════════════════════════════════════════════

def call_geocode(
    scene,
    snap_out: Path,
    res_m:    float,
    dtm_path: Path,
    logger:   logging.Logger,
):
    """
    Call pyroSAR geocode() with parameter names matched to the installed version.

    pyroSAR renamed the pixel-spacing parameter across versions:
        tr=   (≤ v0.17)
        res=  (≥ v0.18)

    Uses inspect.signature() to detect the correct name at runtime.
    Optional parameters (cleanup, groupsize, alignToStandardGrid,
    nodataValueAtSea) are included only if present in the signature.
    """
    import inspect

    try:
        params = set(inspect.signature(_geocode).parameters)
    except Exception:
        params = None

    # Core kwargs present in all versions
    kwargs: dict = {
        "infile":               scene,
        "outdir":               str(snap_out),
        "t_srs":                2157,
        "scaling":              "db",
        "refarea":              "sigma0",
        "externalDEMFile":      str(dtm_path),
        "externalDEMNoDataValue": -9999,
    }

    # Pixel-spacing parameter (name changed across versions)
    spacing_found = False
    for name in ("res", "tr", "pixelSpacingInMeter"):
        if params is None or name in params:
            kwargs[name] = res_m
            spacing_found = True
            logger.info("  geocode() spacing param: %s=%.0f", name, res_m)
            break

    if not spacing_found:
        logger.warning(
            "  Pixel-spacing param not found. "
            "Available params: %s", sorted(params or [])
        )

    # Optional params — include only if signature supports them
    for name, val in {
        "alignToStandardGrid": True,
        "nodataValueAtSea":    True,
        "cleanup":             True,
        "groupsize":           1,
    }.items():
        if params is None or name in params:
            kwargs[name] = val

    logger.info(
        "  geocode() params: t_srs=%s res=%.0f scaling=%s refarea=%s",
        kwargs.get("t_srs"), res_m, kwargs.get("scaling"), kwargs.get("refarea"),
    )
    _run_geocode(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  QC statistics
# ═══════════════════════════════════════════════════════════════════════

def compute_qc(data: np.ndarray, nodata: float = -9999.0) -> dict:
    """
    Compute per-band dB statistics and a coarse water fraction estimate.
    data: [2, H, W] float32 (VV, VH in dB).
    Water fraction = fraction of VV pixels below -16 dB (Schumann & Moller 2015).
    """
    stats = {}
    for i, pol in enumerate(("vv", "vh")):
        v = data[i]
        valid = v[v != nodata]
        if valid.size == 0:
            stats.update({f"{pol}_mean_db": float("nan"),
                          f"{pol}_std_db":  float("nan"),
                          f"{pol}_valid_frac": 0.0})
        else:
            stats[f"{pol}_mean_db"]    = round(float(valid.mean()), 3)
            stats[f"{pol}_std_db"]     = round(float(valid.std()),  3)
            stats[f"{pol}_p05_db"]     = round(float(np.percentile(valid,  5)), 3)
            stats[f"{pol}_p95_db"]     = round(float(np.percentile(valid, 95)), 3)
            stats[f"{pol}_valid_frac"] = round(float(valid.size / v.size), 4)

    vv    = data[0]
    valid = vv[vv != nodata]
    stats["water_fraction_vv16"] = round(
        float((valid < -16.0).mean()) if valid.size > 0 else 0.0, 4
    )
    return stats


# ═══════════════════════════════════════════════════════════════════════
#  Output helpers
# ═══════════════════════════════════════════════════════════════════════

def write_geotiff(
    path: Path, data: np.ndarray, transform, crs_epsg: int = 2157,
    nodata: float = -9999.0,
):
    """Write 2-band float32 GeoTIFF with LZW compression."""
    _, H, W = data.shape
    with rasterio.open(path, "w", driver="GTiff", dtype="float32",
                       width=W, height=H, count=2,
                       crs=CRS.from_epsg(crs_epsg), transform=transform,
                       nodata=nodata, compress="lzw",
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(data[0], 1)
        dst.write(data[1], 2)
        dst.update_tags(1, name="VV_sigma0_dB")
        dst.update_tags(2, name="VH_sigma0_dB")


def make_event_id(acq_date: str, scene_id: str) -> str:
    """Generate a unique event ID: event_YYYYMMDD_<4-char orbit hash>."""
    date_clean = acq_date.replace("-", "")
    m = re.search(r"_([0-9A-F]{6})_", scene_id.upper())
    suffix = m.group(1)[-4:] if m else "0000"
    return f"event_{date_clean}_{suffix}"


# ═══════════════════════════════════════════════════════════════════════
#  sar_events.json registry
# ═══════════════════════════════════════════════════════════════════════

def load_registry(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_registry(registry: dict, path: Path):
    path.write_text(json.dumps(registry, indent=2))


def add_to_registry(
    registry:   dict,
    event_id:   str,
    npy_name:   str,
    scene_meta: dict,
    bbox_itm:   tuple,
    qc:         dict,
) -> dict:
    """
    Add an event entry.  event_start / event_end are ±3-day placeholders —
    update them to match actual OPW flood event records before training.
    """
    acq_date = scene_meta["start_time"][:10]
    acq_dt   = datetime.strptime(acq_date, "%Y-%m-%d")
    registry[event_id] = {
        "sar_path":         npy_name,
        "acquisition_date": acq_date,
        "event_start":      (acq_dt - timedelta(days=3)).strftime("%Y-%m-%dT00:00"),
        "event_end":        (acq_dt + timedelta(days=3)).strftime("%Y-%m-%dT23:45"),
        "bbox_itm":         list(bbox_itm),
        "scene_id":         scene_meta["scene_id"],
        "satellite":        scene_meta["satellite"],
        "orbit":            scene_meta["orbit"],
        **qc,
        "_note": (
            "event_start and event_end are PLACEHOLDERS (±3 days). "
            "Update them to match actual OPW flood event records."
        ),
    }
    return registry


# ═══════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_output(npy_path: Path, tif_path: Path, bbox_itm: tuple,
                    logger: logging.Logger):
    """Post-processing checks on the output files."""
    logger.info("── Validation ───────────────────────────────────────")

    arr = np.load(npy_path)
    assert arr.ndim == 3 and arr.shape[0] == 2, f"Bad shape: {arr.shape}"
    assert arr.dtype == np.float32,             f"Bad dtype: {arr.dtype}"
    assert not np.isnan(arr).any(),             "NaN values found (use -9999 for nodata)"
    logger.info("  .npy: %s  dtype=%s  OK", arr.shape, arr.dtype)

    valid = arr[arr != -9999.0]
    if valid.size:
        logger.info("  dB range: [%.1f, %.1f]", valid.min(), valid.max())

    if HAS_RASTERIO and tif_path.exists():
        with rasterio.open(tif_path) as src:
            assert src.crs.to_epsg() == 2157, f"CRS is EPSG:{src.crs.to_epsg()}"
            b = src.bounds
            xmin, ymin, xmax, ymax = bbox_itm
            assert b.left < xmax and b.right > xmin, "GeoTIFF outside bbox"
            logger.info(
                "  GeoTIFF: EPSG:2157  %.1f x %.1f m px  OK",
                abs(src.transform.a), abs(src.transform.e),
            )

    logger.info("  Validation passed  OK")


# ═══════════════════════════════════════════════════════════════════════
#  Environment check  (--check flag)
# ═══════════════════════════════════════════════════════════════════════

def run_env_check(logger: logging.Logger, zip_files: list, dtm_path: Path | None):
    """Print environment status without processing any files."""
    logger.info("")
    logger.info("══ Environment check " + "=" * 40)
    logger.info("  Python: %s", sys.executable)

    for pkg in ("rasterio", "numpy", "geopandas", "pyproj", "shapely", "pyroSAR"):
        try:
            m = __import__(pkg)
            logger.info("  %-12s  %s", pkg + ":", getattr(m, "__version__", "ok"))
        except ImportError:
            logger.warning("  %-12s  NOT INSTALLED", pkg + ":")

    logger.info("")
    if HAS_PYROSAR:
        logger.info("  pyroSAR geocode:  FOUND  OK")
    else:
        logger.warning("  pyroSAR geocode:  NOT FOUND — %s", HAS_PYROSAR_ERROR)
        logger.warning("  Fix: pip install pyroSAR  (inside this virtualenv)")

    snap = find_snap_home(logger=None)
    if snap:
        logger.info("  SNAP:  %s  OK", snap)
    else:
        logger.warning("  SNAP:  NOT FOUND")
        logger.warning("  Fix: set SNAP_HOME or install to a standard path")
        logger.warning("  Download: https://step.esa.int/main/download/snap-download/")

    logger.info("")
    if dtm_path:
        if dtm_path.exists():
            logger.info("  DTM:   %s  OK", dtm_path)
        else:
            logger.warning("  DTM:   NOT FOUND at %s", dtm_path)
    else:
        logger.warning("  DTM:   --dtm not supplied")

    logger.info("")
    logger.info("  Input scenes (%d):", len(zip_files))
    for f in zip_files:
        mb = f.stat().st_size / 1_048_576 if f.exists() else -1
        logger.info("    %-70s  %.0f MB", f.name, mb)

    ready = (HAS_PYROSAR and snap is not None
             and dtm_path is not None and dtm_path.exists())
    logger.info("")
    if ready:
        logger.info("  Status: READY — all requirements met")
    else:
        logger.warning("  Status: NOT READY — fix the issues above")
    logger.info("══" + "=" * 48)


# ═══════════════════════════════════════════════════════════════════════
#  Core processing: one SAFE scene
# ═══════════════════════════════════════════════════════════════════════

def process_scene(
    zip_path: Path,
    out_dir:  Path,
    bbox_itm: tuple,
    dtm_path: Path,
    res_m:    float,
    snap_home: str,
    logger:   logging.Logger,
) -> tuple[str, dict, dict, str] | None:
    """
    Process one GRD SAFE archive through pyroSAR + SNAP.

    Returns (event_id, scene_meta, qc, npy_filename) on success, None on failure.
    """
    tmp_dir = out_dir / "tmp_safe"
    tmp_dir.mkdir(exist_ok=True)

    try:
        # ── 1. Extract ────────────────────────────────────────────────
        safe_path = unzip_safe(zip_path, tmp_dir, logger)
        scene_meta = parse_scene_id(safe_path)
        logger.info("  Scene:    %s", scene_meta["scene_id"])
        logger.info("  Acquired: %s", scene_meta["start_time"])

        # ── 2. Reject COG format ──────────────────────────────────────
        if "_COG" in safe_path.name.upper():
            raise ValueError(
                "COG SAFE format not supported by pyroSAR. "
                "Use the standard SAFE for this acquisition "
                "(same date/orbit, no _COG suffix)."
            )

        # ── 3. Identify scene ─────────────────────────────────────────
        try:
            scene = _run_identify(str(safe_path))
        except RuntimeError as e:
            raise RuntimeError(
                f"pyroSAR could not identify {safe_path.name}: {e}. "
                "Ensure this is a standard IW GRD SAFE (not SLC, not COG)."
            ) from e
        logger.info("  pyroSAR scene identified: %s", scene)

        # ── 4. Build WGS84 subset polygon ────────────────────────────
        wgs84_poly = itm_bbox_to_wgs84_wkt(bbox_itm, logger)
        logger.info("  WGS84 subset polygon: %s", wgs84_poly)

        # ── 5. Run SNAP geocode ───────────────────────────────────────
        snap_out = out_dir / "snap_tmp" / safe_path.stem
        snap_out.mkdir(parents=True, exist_ok=True)

        logger.info("  Running SNAP geocode … (5–15 min)")
        t0 = time.perf_counter()
        call_geocode(scene, snap_out, res_m, dtm_path, logger)
        logger.info("  SNAP geocode completed in %.0f s", time.perf_counter() - t0)

        # ── 6. Stack VV + VH outputs ─────────────────────────────────
        vv_tif = next(
            (t for t in snap_out.rglob("*.tif") if "VV" in t.name.upper()), None
        )
        vh_tif = next(
            (t for t in snap_out.rglob("*.tif") if "VH" in t.name.upper()), None
        )
        if vv_tif is None or vh_tif is None:
            files = [t.name for t in snap_out.rglob("*")]
            raise FileNotFoundError(
                f"Expected VV and VH GeoTIFFs in {snap_out}. "
                f"Found: {files}"
            )

        with rasterio.open(vv_tif) as src:
            vv_data   = src.read(1).astype(np.float32)
            transform = src.transform
        with rasterio.open(vh_tif) as src:
            vh_data = src.read(1).astype(np.float32)

        data = np.stack([vv_data, vh_data], axis=0)  # [2, H, W]

        # ── 7. QC ─────────────────────────────────────────────────────
        qc = compute_qc(data)
        logger.info(
            "  QC  VV=%.1f dB  VH=%.1f dB  water=%.1f%%",
            qc["vv_mean_db"], qc["vh_mean_db"],
            qc["water_fraction_vv16"] * 100,
        )

        # ── 8. Write outputs ──────────────────────────────────────────
        event_id = make_event_id(scene_meta["start_time"][:10], scene_meta["scene_id"])
        tif_path = out_dir / f"{event_id}.tif"
        npy_path = out_dir / f"{event_id}.npy"

        write_geotiff(tif_path, data, transform)
        np.save(npy_path, data)
        logger.info("  Written: %s", tif_path.name)
        logger.info("  Written: %s  shape=%s", npy_path.name, data.shape)

        # ── 9. Validate ───────────────────────────────────────────────
        validate_output(npy_path, tif_path, bbox_itm, logger)

        return event_id, scene_meta, qc, npy_path.name

    except Exception as exc:
        logger.error("  FAILED: %s — %s", zip_path.name, exc, exc_info=True)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sentinel-1 GRD production preprocessing for River Lee PI-ST-GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Single .zip/.SAFE file or directory of zip files.")
    p.add_argument("--outdir", "-o", required=True,
                   help="Output directory for GeoTIFFs, .npy, and sar_events.json.")
    p.add_argument("--dtm", default=None,
                   help="Path to 30 m ITM DTM GeoTIFF (required for processing).")

    bbox_grp = p.add_mutually_exclusive_group()
    bbox_grp.add_argument("--shapefile", default=None, metavar="PATH",
                          help="Catchment boundary vector file (.shp, .gpkg, .geojson).")
    bbox_grp.add_argument("--nodes-csv", default=None, metavar="PATH",
                          help="nodes.csv with ITM gauge coordinates.")
    bbox_grp.add_argument("--bbox", nargs=4, type=float,
                          metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                          help="Explicit ITM bbox in metres.")

    p.add_argument("--buffer", type=float, default=5000.0, metavar="M",
                   help="Buffer in metres around catchment envelope (default 5000).")
    p.add_argument("--x-col", default="easting_itm",
                   help="ITM easting column in nodes.csv (default: easting_itm).")
    p.add_argument("--y-col", default="northing_itm",
                   help="ITM northing column in nodes.csv (default: northing_itm).")
    p.add_argument("--res", type=float, default=20.0,
                   help="Output pixel size in metres (default: 20).")
    p.add_argument("--check", action="store_true",
                   help="Print environment status without processing.")
    return p


def main():
    args   = build_parser().parse_args()
    logger = get_logger()

    input_path = Path(args.input)
    out_dir    = Path(args.outdir)
    dtm_path   = Path(args.dtm) if args.dtm else None
    res_m      = args.res
    buffer_m   = args.buffer

    # ── Resolve bbox ───────────────────────────────────────────────────
    bbox_itm = resolve_bbox(
        shapefile = Path(args.shapefile) if args.shapefile else None,
        nodes_csv = Path(args.nodes_csv) if args.nodes_csv else None,
        bbox_arg  = args.bbox,
        buffer_m  = buffer_m,
        logger    = logger,
    )

    # ── Collect GRD input files ────────────────────────────────────────
    if input_path.is_dir():
        all_files = sorted(input_path.glob("*.zip")) + \
                    sorted(input_path.glob("*.SAFE"))
        if not all_files:
            logger.error("No .zip or .SAFE files in %s", input_path)
            sys.exit(1)
    else:
        all_files = [input_path]

    grd_files, slc_files, cog_files = classify_input_files(all_files)

    if slc_files:
        logger.warning(
            "  %d SLC product(s) SKIPPED (requires GRD):", len(slc_files)
        )
        for f in slc_files:
            logger.warning("    %s", f.name)

    if cog_files:
        logger.warning(
            "  %d COG product(s) SKIPPED (pyroSAR does not support COG format):",
            len(cog_files),
        )
        for f in cog_files:
            logger.warning("    %s", f.name)

    if not grd_files:
        logger.error(
            "No GRD files to process. "
            "Download IW GRDH products from Copernicus Browser."
        )
        sys.exit(1)

    # ── --check mode ──────────────────────────────────────────────────
    if args.check:
        run_env_check(logger, grd_files, dtm_path)
        sys.exit(0)

    # ── Production mode pre-flight ────────────────────────────────────
    if not HAS_RASTERIO:
        logger.error("rasterio not installed. Run: conda install -c conda-forge rasterio")
        sys.exit(1)

    if not HAS_PYROSAR:
        logger.error("pyroSAR not available: %s", HAS_PYROSAR_ERROR)
        logger.error("Fix: pip install pyroSAR  (inside this virtualenv)")
        sys.exit(1)

    if dtm_path is None:
        logger.error("--dtm is required for production mode.")
        sys.exit(1)
    if not dtm_path.exists():
        logger.error("DTM not found: %s", dtm_path)
        sys.exit(1)

    snap_home = find_snap_home(logger)
    if snap_home is None:
        logger.error(
            "SNAP not found. Set SNAP_HOME or install SNAP: "
            "https://step.esa.int/main/download/snap-download/"
        )
        sys.exit(1)

    # ── Optional: validate nodes inside bbox ──────────────────────────
    if args.nodes_csv:
        validate_nodes_in_bbox(
            Path(args.nodes_csv), bbox_itm,
            x_col=args.x_col, y_col=args.y_col, logger=logger,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = out_dir / "sar_events.json"
    registry      = load_registry(registry_path)
    log_entries   = []

    logger.info("=" * 60)
    logger.info("SAR preprocessing — production mode")
    logger.info("  SNAP:    %s", snap_home)
    logger.info("  Inputs:  %d GRD scene(s)", len(grd_files))
    logger.info("  Output:  %s", out_dir)
    logger.info("  BBox:    %.0f  %.0f  %.0f  %.0f", *bbox_itm)
    logger.info("  Res:     %.0f m", res_m)
    logger.info("=" * 60)

    n_ok = n_fail = 0
    for zip_path in grd_files:
        logger.info("")
        logger.info("Processing: %s", zip_path.name)
        t_start = time.perf_counter()

        result = process_scene(
            zip_path  = zip_path,
            out_dir   = out_dir,
            bbox_itm  = bbox_itm,
            dtm_path  = dtm_path,
            res_m     = res_m,
            snap_home = snap_home,
            logger    = logger,
        )
        elapsed = time.perf_counter() - t_start

        if result is None:
            n_fail += 1
            log_entries.append({"file": zip_path.name, "status": "failed"})
            continue

        event_id, scene_meta, qc, npy_name = result
        registry = add_to_registry(
            registry, event_id, npy_name, scene_meta, bbox_itm, qc
        )
        save_registry(registry, registry_path)
        log_entries.append({
            "file": zip_path.name, "event_id": event_id,
            "status": "ok", "elapsed_s": round(elapsed, 1), **qc,
        })
        n_ok += 1
        logger.info("  Done in %.0f s", elapsed)

    (out_dir / "processing_log.json").write_text(json.dumps(log_entries, indent=2))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done.  %d succeeded  %d failed", n_ok, n_fail)
    logger.info("Output:   %s", out_dir)
    logger.info("Registry: %s  (%d entries)", registry_path, len(registry))
    if n_ok:
        logger.info("")
        logger.info("ACTION REQUIRED: open sar_events.json and update")
        logger.info("  event_start / event_end for each entry to match")
        logger.info("  your OPW flood event records before training.")
    logger.info("=" * 60)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
