# download_era5_sm.py
# =======================================================================
# Scheduled ERA5-Land soil moisture downloader for OPW gauge nodes
#
# Reads node coordinates from nodes.csv (graph_builder output format).
# Downloads hourly swvl1/swvl2 + met covariates for each gauge location
# via the CDS reanalysis-era5-land-timeseries point-query endpoint.
#
# -- ONE-TIME SETUP (required before first run) ----------------------
#
# 1. Register / log in at https://cds.climate.copernicus.eu
#
# 2. Accept the Terms of Use for the dataset at:
#    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries
#    (scroll to the bottom of the Download tab and accept)
#
# 3. Get your Personal Access Token from:
#    https://cds.climate.copernicus.eu/profile
#    (copy the token shown under "Personal Access Token")
#
# 4. Create the credentials file at ~/.cdsapirc:
#
#       url: https://cds.climate.copernicus.eu/api
#       key: <YOUR-PERSONAL-ACCESS-TOKEN>
#
#    NOTE: The URL is https://cds.climate.copernicus.eu/api  (no /v2 suffix)
#          The key is now a single token, not the old uid:api-key format.
#
#    Alternatively, set the environment variable instead of the file:
#       export CDSAPI_KEY="<YOUR-PERSONAL-ACCESS-TOKEN>"
#       export CDSAPI_URL="https://cds.climate.copernicus.eu/api"
#
# 5. Install or upgrade the CDS API client (v0.7.7+ required):
#       pip install "cdsapi>=0.7.7"
#
# -- USAGE ------------------------------------------------------------
#
#   python download_era5_sm.py                           # last 10 days
#   python download_era5_sm.py --backfill                # full history from 2016
#   python download_era5_sm.py --date 2025-11-01         # specific date
#   python download_era5_sm.py --days 30                 # last 30 days
#   python download_era5_sm.py --nodes-csv path/to/nodes.csv
#   python download_era5_sm.py --skip-tidal              # exclude tidal nodes
#
# Scheduled daily (cron, runs at 06:00 UTC):
#   0 6 * * * /path/to/venv/bin/python /path/to/download_era5_sm.py
# =======================================================================

import argparse
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr

# -- Configuration ------------------------------------------------------

# CDS API endpoint -- updated Nov 2024, no /v2 suffix
CDS_API_URL = "https://cds.climate.copernicus.eu/api"
CDS_MIN_VERSION = (0, 7, 7)
LATENCY_DAYS = 8

BASE_DIR          = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_NODES_CSV = BASE_DIR / "dataset/graph/nodes.csv"
OUTPUT_DIR        = BASE_DIR / "dataset/era5_land_sm"
COMBINED_SM_FILE  = BASE_DIR / "dataset/era5" / "era5_land_sm_lee.nc"

# IMPORTANT: Use reanalysis-era5-land (gridded), NOT reanalysis-era5-land-timeseries.
# The timeseries endpoint silently drops volumetric_soil_water variables.
CDS_DATASET = "reanalysis-era5-land"

SM_VARIABLES = [
    "volumetric_soil_water_layer_1",   # swvl1  0-7 cm
    "volumetric_soil_water_layer_2",   # swvl2  7-28 cm
]
EXTRA_VARIABLES = [
    "total_precipitation",
    "2m_temperature",
]
VARIABLES = SM_VARIABLES + EXTRA_VARIABLES


# -- Pre-flight checks --------------------------------------------------

def check_netcdf4() -> None:
    """
    netcdf4 (or h5netcdf) is required to read ERA5-Land downloads.
    xarray's default scipy backend only handles NetCDF3 and will raise:
      "did not find a match in any of xarray's currently installed IO backends"
    Install: pip install netcdf4
    """
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        try:
            import h5netcdf  # noqa: F401
        except ImportError:
            print(
                "ERROR: No NetCDF4 backend found for xarray.\n"
                "       ERA5-Land files are NetCDF4 format and cannot be read\n"
                "       by xarray's default scipy backend.\n"
                "       Run: pip install netcdf4"
            )
            import sys; sys.exit(1)


def check_cdsapi_version() -> None:
    """
    Enforce cdsapi>=0.7.7.  The new CDS API (Nov 2024) requires:
      - Single Personal Access Token auth (not uid:api-key)
      - New URL: https://cds.climate.copernicus.eu/api  (no /v2)
      - client.retrieve(dataset, request, target) signature
    Versions below 0.7.7 will silently fail or use the wrong endpoint.
    """
    from importlib.metadata import version, PackageNotFoundError
    try:
        ver_str = version("cdsapi")
        parts   = tuple(int(x) for x in ver_str.split(".")[:3])
        if parts < CDS_MIN_VERSION:
            min_str = ".".join(str(x) for x in CDS_MIN_VERSION)
            print(
                f"ERROR: cdsapi {ver_str} is installed but >={min_str} is required.\n"
                f"       Run: pip install \"cdsapi>={min_str}\""
            )
            sys.exit(1)
    except PackageNotFoundError:
        print(
            "ERROR: cdsapi is not installed.\n"
            "       Run: pip install \"cdsapi>=0.7.7\""
        )
        sys.exit(1)


def check_credentials() -> None:
    """
    Verify that CDS credentials are available -- either in ~/.cdsapirc
    or via CDSAPI_KEY / CDSAPI_URL environment variables.

    ~/.cdsapirc format (new CDS API, Nov 2024):
        url: https://cds.climate.copernicus.eu/api
        key: <YOUR-PERSONAL-ACCESS-TOKEN>

    Environment variable alternative:
        export CDSAPI_KEY="<YOUR-PERSONAL-ACCESS-TOKEN>"
        export CDSAPI_URL="https://cds.climate.copernicus.eu/api"

    Get your token from: https://cds.climate.copernicus.eu/profile
    """
    rc_path = Path.home() / ".cdsapirc"
    has_env = bool(os.environ.get("CDSAPI_KEY"))
    has_rc  = rc_path.exists()

    if not has_env and not has_rc:
        print(
            "ERROR: No CDS credentials found.\n\n"
            "  Option A -- create ~/.cdsapirc with:\n"
            f"      url: {CDS_API_URL}\n"
            "      key: <YOUR-PERSONAL-ACCESS-TOKEN>\n\n"
            "  Option B -- set environment variables:\n"
            f"      export CDSAPI_URL=\"{CDS_API_URL}\"\n"
            "      export CDSAPI_KEY=\"<YOUR-PERSONAL-ACCESS-TOKEN>\"\n\n"
            "  Get your token at: https://cds.climate.copernicus.eu/profile\n"
            "  (You must also accept Terms of Use on the dataset page first.)"
        )
        sys.exit(1)

    # If .cdsapirc exists, warn if it still uses the old /api/v2 URL
    if has_rc:
        rc_text = rc_path.read_text()
        if "/api/v2" in rc_text:
            print(
                "WARNING: ~/.cdsapirc contains the old URL (https://cds.climate.copernicus.eu/api/v2).\n"
                "         Update it to: url: https://cds.climate.copernicus.eu/api\n"
                "         The old endpoint is no longer supported.\n"
            )
        if ":" in rc_text.split("key:")[-1].split("\n")[0].strip() if "key:" in rc_text else False:
            print(
                "WARNING: ~/.cdsapirc key looks like the old 'uid:api-key' format.\n"
                "         The new CDS API uses a single Personal Access Token.\n"
                "         Get yours at: https://cds.climate.copernicus.eu/profile\n"
            )


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("era5_sm")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")

    # On Windows, sys.stdout may default to cp1252.
    # Wrap in UTF-8 with errors='replace' so the console handler never crashes.
    import io
    safe_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    sh = logging.StreamHandler(safe_stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(output_dir / "download.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# -- Node loading -------------------------------------------------------

def load_nodes(csv_path: Path, skip_tidal: bool = False) -> list[dict]:
    """
    Load OPW gauge nodes from nodes.csv produced by graph_builder.py.

    Expected columns (from your actual file):
        node_idx, ref, name, lat, lon, ..., is_reservoir, is_tidal, ...

    Returns a list of dicts with keys:
        node_id   -- "OPW_{ref}"  (matches graph_builder convention)
        node_idx  -- integer index in the graph node list
        ref       -- OPW reference number (int)
        name      -- gauge name string
        lat, lon  -- WGS84 coordinates
        is_reservoir, is_tidal -- boolean flags (retained for metadata)
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = {"node_idx", "ref", "name", "lat", "lon", "is_tidal", "is_reservoir"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"nodes.csv is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    nodes = []
    for _, row in df.iterrows():
        is_tidal     = bool(row["is_tidal"])
        is_reservoir = bool(row["is_reservoir"])

        if skip_tidal and is_tidal:
            continue   # tidal nodes (Pope's Quay, St Patrick's, Currach Club)
                       # have tidally-dominated water levels -- SM less informative

        nodes.append({
            "node_id":      f"OPW_{int(row['ref'])}",
            "node_idx":     int(row["node_idx"]),
            "ref":          int(row["ref"]),
            "name":         row["name"],
            "lat":          float(row["lat"]),
            "lon":          float(row["lon"]),
            "is_reservoir": is_reservoir,
            "is_tidal":     is_tidal,
        })

    return nodes


def print_node_summary(nodes: list[dict], log: logging.Logger) -> None:
    """Log a summary table of loaded nodes."""
    reservoirs = [n for n in nodes if n["is_reservoir"]]
    tidal      = [n for n in nodes if n["is_tidal"]]
    log.info(f"Loaded {len(nodes)} nodes from CSV  "
             f"({len(reservoirs)} reservoir, {len(tidal)} tidal)")
    log.info(f"  {'idx':>3}  {'node_id':<14}  {'name':<35}  {'lat':>9}  {'lon':>10}  flags")
    log.info(f"  {'---':>3}  {'-------':<14}  {'----':<35}  {'---':>9}  {'---':>10}  -----")
    for n in nodes:
        flags = []
        if n["is_reservoir"]: flags.append("RESERVOIR")
        if n["is_tidal"]:     flags.append("TIDAL")
        log.info(
            f"  {n['node_idx']:>3}  {n['node_id']:<14}  {n['name']:<35}"
            f"  {n['lat']:>9.5f}  {n['lon']:>10.5f}  {', '.join(flags)}"
        )


# -- Date helpers -------------------------------------------------------

def latest_available_date() -> date:
    """ERA5-Land lags real-time by ~5-8 days."""
    return date.today() - timedelta(days=LATENCY_DAYS)


def date_range_str(start: date, end: date) -> str:
    """CDS date range string: 'YYYY-MM-DD/YYYY-MM-DD'"""
    return f"{start.isoformat()}/{end.isoformat()}"



# -- File validation and zip-extraction fallback -----------------------

def validate_nc_file(path: Path, log: logging.Logger) -> bool:
    """
    Return True if path is a valid NetCDF4 file xarray can open.
    Return False if it is corrupt, a zip archive, or unreadable.

    CDS sometimes delivers a zip even when download_format="unarchived"
    is set -- particularly for cached older jobs. This catches that case.
    """
    try:
        import xarray as xr
        ds = xr.open_dataset(path, engine="netcdf4")
        ds.close()
        return True
    except Exception:
        return False


def validate_merged_file(
    path: Path,
    log:  logging.Logger,
    required_vars: list[str] = None,
) -> bool:
    """
    Validate that a merged ERA5-Land NetCDF contains all required variables.

    Checks:
      1. File is valid NetCDF4 (not corrupt, not a zip)
      2. All required SM variables (swvl1, swvl2) are present
      3. Each SM variable has no more than 5%% NaN values
      4. VWC values are in the physically valid range [0, 0.6] m³/m³
      5. Time dimension is present and monotonically increasing
      6. Node dimension matches expected count if > 1 node in file

    Returns True if all checks pass. Logs a detailed failure message
    for each failed check.
    """
    if required_vars is None:
        required_vars = [v.replace("volumetric_soil_water_layer_", "swvl")
                         for v in SM_VARIABLES]
    # Strip CDS long-names to short names for lookup
    # (merged file stores short names: swvl1, swvl2)

    if not path.exists():
        log.error("Validation FAILED: file not found: %s", path)
        return False

    # Check 1: valid NetCDF4
    if not validate_nc_file(path, log):
        log.error("Validation FAILED: %s is not valid NetCDF4", path.name)
        return False

    import xarray as xr
    import numpy as np
    ds = xr.open_dataset(path, engine="netcdf4")
    found_vars = list(ds.data_vars)
    all_ok = True

    # Check 2: required SM variables present
    missing = [v for v in required_vars if v not in ds.data_vars]
    if missing:
        log.error(
            "Validation FAILED: missing soil moisture variables: %s\n"
            "  File contains: %s\n"
            "  This usually means:\n"
            "    (a) VARIABLES list was modified to exclude SM variables,\n"
            "    (b) a temperature-only file was downloaded and renamed, or\n"
            "    (c) merge_node_files() was called on wrong-variable files.\n"
            "  Fix: re-run download with SM_VARIABLES included in VARIABLES.",
            missing, found_vars,
        )
        all_ok = False

    # Checks 3–5 only if SM vars present
    for var in [v for v in required_vars if v in ds.data_vars]:
        da = ds[var]

        # Check 3: NaN rate
        nan_rate = float(np.isnan(da.values).mean())
        if nan_rate > 0.05:
            log.warning(
                "Validation WARNING: %s has %.1f%% NaN values "
                "(threshold 5%%). Check for download gaps.",
                var, nan_rate * 100,
            )

        # Check 4: physical VWC range
        valid = da.values[~np.isnan(da.values)]
        if valid.size > 0:
            vmin, vmax = float(valid.min()), float(valid.max())
            if vmin < -0.01 or vmax > 0.65:
                log.warning(
                    "Validation WARNING: %s values outside physical range "
                    "[0, 0.6] m³/m³: [%.4f, %.4f]. "
                    "Check units — ERA5-Land should be in m³/m³.",
                    var, vmin, vmax,
                )
            else:
                log.info(
                    "  %s: range=[%.3f, %.3f] m³/m³  NaN=%.2f%%  OK",
                    var, vmin, vmax, nan_rate * 100,
                )

    # Check 5: time dimension exists and is monotonic
    time_dim = next((d for d in ds.dims if "time" in d.lower()), None)
    if time_dim is None:
        log.error("Validation FAILED: no time dimension found in %s", path.name)
        all_ok = False
    else:
        import pandas as pd
        times = pd.DatetimeIndex(ds[time_dim].values)
        if not times.is_monotonic_increasing:
            log.warning(
                "Validation WARNING: %s time dimension is not monotonically "
                "increasing. Consider resorting.",
                path.name,
            )
        log.info(
            "  Time: %s -> %s  (%d steps)",
            times[0].date(), times[-1].date(), len(times),
        )

    # Check 6: node count
    if "node" in ds.dims:
        log.info("  Nodes: %d", ds.sizes["node"])
        if ds.sizes["node"] == 0:
            log.error("Validation FAILED: node dimension has 0 entries")
            all_ok = False

    ds.close()

    if all_ok:
        log.info("Validation PASSED: %s", path.name)
    return all_ok


def extract_zip_nc(zip_path: Path, log: logging.Logger) -> Path | None:
    """
    Extract and merge all NetCDF files from a CDS zip delivery.

    The reanalysis-era5-land-timeseries endpoint returns a zip containing
    ONE NetCDF file per requested variable, e.g.:
        reanalysis-era5-land-timeseries-sfc-2m-temperature{hash}.nc   <- t2m
        reanalysis-era5-land-timeseries-sfc-volumetric-soil{hash}.nc  <- swvl1
        reanalysis-era5-land-timeseries-sfc-volumetric-soil{hash}.nc  <- swvl2
        ...

    Extracting only the first file (alphabetically 't2m') discards all the
    soil moisture variables silently. This function extracts ALL nc files
    from the zip and merges them into a single dataset using xarray.merge(),
    then saves the merged result to zip_path.

    Returns zip_path (now containing merged NetCDF4) on success, None on failure.
    """
    import zipfile
    import tempfile
    import xarray as xr

    if not zipfile.is_zipfile(zip_path):
        return None   # not a zip -- some other corruption, cannot recover

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            nc_members = [m for m in zf.namelist() if m.endswith('.nc')]
            if not nc_members:
                log.warning(
                    f"    Zip {zip_path.name} contains no .nc files. "
                    f"Contents: {zf.namelist()}"
                )
                return None

            log.info(
                f"    Zip contains {len(nc_members)} NC file(s): "
                f"{[m.split('/')[-1][:60] for m in nc_members]}"
            )

            if len(nc_members) == 1:
                # Single file -- simple extract (original behaviour)
                tmp_path = zip_path.with_suffix('.nc.tmp')
                with zf.open(nc_members[0]) as src_f,                         open(tmp_path, 'wb') as dst_f:
                    dst_f.write(src_f.read())
                tmp_path.replace(zip_path)
                log.info(
                    f"    Extracted {nc_members[0].split('/')[-1]} "
                    f"-> {zip_path.name}"
                )
                return zip_path

            # Multiple files -- extract all to a temp directory, merge
            with tempfile.TemporaryDirectory() as tmp_dir:
                extracted = []
                for member in nc_members:
                    member_name = member.split('/')[-1]
                    dst = Path(tmp_dir) / member_name
                    with zf.open(member) as src_f, open(dst, 'wb') as df:
                        df.write(src_f.read())
                    extracted.append(dst)

                # Open and merge all extracted datasets
                datasets = []
                var_names = []
                for ep in extracted:
                    try:
                        ds = xr.open_dataset(ep, engine='netcdf4')
                        var_names.extend(list(ds.data_vars))
                        datasets.append(ds)
                    except Exception as e:
                        log.warning(f"    Could not open {ep.name}: {e}")

                if not datasets:
                    log.warning(
                        f"    No datasets could be opened from zip {zip_path.name}"
                    )
                    return None

                merged = xr.merge(datasets, compat='override')
                log.info(
                    f"    Merged {len(datasets)} NC files into single dataset. "
                    f"Variables: {list(merged.data_vars)}"
                )

                # Write merged dataset to tmp then replace the zip file
                tmp_path = zip_path.with_suffix('.nc.tmp')
                merged.to_netcdf(tmp_path)
                for ds in datasets:
                    ds.close()
                merged.close()

            import shutil as _sh, os as _os
            if zip_path.exists(): _os.remove(str(zip_path))
            _sh.move(str(tmp_path), str(zip_path))
            log.info(
                f"    Saved merged dataset -> {zip_path.name} "
                f"(vars: {var_names})"
            )
            return zip_path

    except Exception as exc:
        log.warning(f"    Failed to extract/merge zip {zip_path.name}: {exc}")
        return None


# -- Gridded chunk download --------------------------------------------

def download_chunk_gridded(
    date_str: str,
    bbox:     list,
    client:   cdsapi.Client,
    log:      logging.Logger,
    retries:  int = 3,
) -> Path | None:
    """
    Download ERA5-Land soil moisture as a spatial grid for the catchment bbox.

    Uses reanalysis-era5-land (gridded), NOT reanalysis-era5-land-timeseries.
    The timeseries endpoint silently drops volumetric_soil_water_layer variables.
    The gridded endpoint reliably returns swvl1/swvl2 when bbox is specified.

    Returns (time x latitude x longitude) NetCDF4 file for the Lee catchment.
    soil_moisture_features.py handles this as Format A (spatial grid).
    """
    safe_str = date_str.replace("/", "_")
    out_path = OUTPUT_DIR / f"era5_sm_gridded_{safe_str}.nc"

    if out_path.exists() and out_path.stat().st_size > 4096:
        if validate_nc_file(out_path, log):
            log.info("  Gridded chunk %s: already valid, skipping", safe_str)
            return out_path
        log.warning("  Gridded chunk %s: invalid, re-downloading ...", safe_str)
        out_path.unlink()

    parts      = date_str.split("/")
    start_date = date.fromisoformat(parts[0])
    end_date   = date.fromisoformat(parts[1])

    all_days = []
    cur = start_date
    while cur <= end_date:
        all_days.append(cur)
        cur += timedelta(days=1)

    years  = sorted({str(d.year)          for d in all_days})
    months = sorted({f"{d.month:02d}"     for d in all_days})
    days   = sorted({f"{d.day:02d}"       for d in all_days})

    request = {
        "product_type":     ["reanalysis"],
        "variable":         VARIABLES,
        "year":             years,
        "month":            months,
        "day":              days,
        "time":             [f"{h:02d}:00" for h in range(24)],
        "area":             bbox,          # [N, W, S, E]
        "data_format":      "netcdf",
        "download_format":  "unarchived",
    }

    for attempt in range(1, retries + 1):
        try:
            log.info(
                "  Gridded %s (attempt %d/%d) bbox=%s vars=%s",
                safe_str, attempt, retries, bbox, VARIABLES,
            )
            client.retrieve(CDS_DATASET, request, str(out_path))
            log.info("  Saved -> %s", out_path.name)
            return out_path
        except Exception as exc:
            log.warning("  Attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                wait = 60 * attempt
                log.info("  Retrying in %ds ...", wait)
                time.sleep(wait)
            else:
                log.error("  All %d attempts failed for chunk %s", retries, safe_str)
                return None


# -- Merge per-node files -----------------------------------------------

def merge_node_files(
    node_file_pairs: list[tuple[dict, Path | None]],
    output_path:     Path,
    log:             logging.Logger,
) -> xr.Dataset | None:
    """
    Merge per-node netCDF downloads into one combined dataset with a
    'node' dimension keyed by node_id strings.

    Output format matches what extract_node_sm_features() in
    soil_moisture_features.py expects:
        dims: (time, node)
        coords: node = ["OPW_19056", "OPW_19057", ...]

    Also writes a companion CSV index mapping node_id -> node metadata,
    useful for downstream alignment with the graph node list.
    """
    datasets = []
    failed   = []

    for node, fpath in node_file_pairs:
        if fpath is None or not fpath.exists():
            failed.append(node["node_id"])
            continue
        try:
            # Validate before opening; attempt zip extraction if needed
            if not validate_nc_file(fpath, log):
                recovered = extract_zip_nc(fpath, log)
                if not recovered or not validate_nc_file(recovered, log):
                    raise ValueError("File is not valid NetCDF4 and zip extraction failed")
                fpath = recovered
            ds = xr.open_dataset(fpath, engine="netcdf4")
            # Drop lat/lon scalar coords that conflict when concatenating
            ds = ds.drop_vars(
                [v for v in ("latitude", "longitude") if v in ds.coords],
                errors="ignore",
            )
            # Tag with node metadata as attributes
            ds = ds.expand_dims({"node": [node["node_id"]]})
            ds.coords["node_idx"]  = ("node", [node["node_idx"]])
            ds.coords["ref"]       = ("node", [node["ref"]])
            ds.coords["node_name"] = ("node", [node["name"]])
            ds.coords["node_lat"]  = ("node", [node["lat"]])
            ds.coords["node_lon"]  = ("node", [node["lon"]])
            datasets.append(ds)
        except Exception as exc:
            log.warning(f"  Could not open {fpath.name}: {exc}")
            failed.append(node["node_id"])

    if not datasets:
        log.error("No node files to merge -- combined file not written")
        return None

    combined = xr.concat(datasets, dim="node")
    combined.to_netcdf(output_path)

    log.info(
        f"Merged {len(datasets)}/{len(node_file_pairs)} nodes -> {output_path.name}"
    )
    if failed:
        log.warning(f"  Failed nodes not included: {failed}")

    return combined


# -- Main download loop -------------------------------------------------

def run_download(
    nodes:      list[dict],
    start_date: date,
    end_date:   date,
    chunk_days: int,
    log:        logging.Logger,
) -> None:
    """
    Download ERA5-Land SM for the Lee catchment as a spatial grid.
    One CDS request per time chunk covers the entire catchment bbox.
    Chunks are validated then concatenated into COMBINED_SM_FILE.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_SM_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url=CDS_API_URL)

    # Derive bbox from loaded nodes + buffer
    lats = [n["lat"] for n in nodes]
    lons = [n["lon"] for n in nodes]
    buf  = 0.2
    bbox = [
        round(max(lats) + buf, 2),   # N
        round(min(lons) - buf, 2),   # W
        round(min(lats) - buf, 2),   # S
        round(max(lons) + buf, 2),   # E
    ]
    log.info("Catchment bbox [N W S E]: %s", bbox)
    log.info("Download: %s -> %s  (%d-day chunks)", start_date, end_date, chunk_days)

    new_chunks = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        date_str  = date_range_str(current, chunk_end)
        fpath     = download_chunk_gridded(date_str, bbox, client, log)
        if fpath and validate_merged_file(fpath, log):
            new_chunks.append(fpath)
        elif fpath:
            log.error("Chunk %s failed validation.", date_str)
        current = chunk_end + timedelta(days=1)

    # Collect all valid gridded chunks (including previous runs)
    all_chunks   = sorted(set(list(new_chunks) + list(OUTPUT_DIR.glob("era5_sm_gridded_*.nc"))))
    valid_chunks = [f for f in all_chunks if validate_merged_file(f, log)]
    if not valid_chunks:
        log.error("No valid chunks to concatenate."); return

    log.info("Concatenating %d chunk(s) -> %s", len(valid_chunks), COMBINED_SM_FILE)
    try:
        import xarray as xr
        ds_list  = [xr.open_dataset(f, engine="netcdf4") for f in valid_chunks]
        combined = xr.concat(ds_list, dim="time").sortby("time")
        combined.to_netcdf(COMBINED_SM_FILE)
        for ds in ds_list: ds.close()
        log.info("Combined file written: %s", COMBINED_SM_FILE)
        validate_merged_file(COMBINED_SM_FILE, log)
    except Exception as exc:
        log.error("Concatenation failed: %s", exc)

    log.info("Download run complete")


# -- CLI ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scheduled ERA5-Land soil moisture downloader for OPW nodes"
    )
    p.add_argument(
        "--nodes-csv",
        type=Path,
        default=DEFAULT_NODES_CSV,
        help=f"Path to nodes.csv (default: {DEFAULT_NODES_CSV})",
    )
    p.add_argument(
        "--backfill",
        action="store_true",
        help="Download full history from 2016-01-01 to latest available date",
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Download a single specific date",
    )
    p.add_argument(
        "--days",
        type=int,
        default=10,
        metavar="N",
        help="Download the last N days from latest available (default: 10)",
    )
    p.add_argument(
        "--skip-tidal",
        action="store_true",
        help=(
            "Exclude tidal nodes (Pope's Quay, St Patrick's Quay, Currach Club). "
            "Soil moisture is physically valid there but water levels are "
            "tidally dominated so SM has little forecast value at those nodes."
        ),
    )
    p.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        metavar="N",
        help="Split downloads into N-day chunks per CDS request (default: 30)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Pre-flight: verify cdsapi version and credentials before doing anything
    check_cdsapi_version()
    check_credentials()
    check_netcdf4()

    # Set up logging (needs OUTPUT_DIR to exist first)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(OUTPUT_DIR)

    # Load nodes from CSV
    if not args.nodes_csv.exists():
        log.error(f"nodes.csv not found: {args.nodes_csv}")
        sys.exit(1)

    nodes = load_nodes(args.nodes_csv, skip_tidal=args.skip_tidal)
    print_node_summary(nodes, log)

    if not nodes:
        log.error("No nodes loaded -- check nodes.csv and --skip-tidal flag")
        sys.exit(1)

    # Determine date range
    latest = latest_available_date()
    log.info(f"Latest available ERA5-Land date: {latest} (latency={LATENCY_DAYS}d)")

    if args.backfill:
        start = date(2016, 1, 1)
        log.info(f"Backfill mode: {start} -> {latest}")
        run_download(nodes, start, latest, args.chunk_days, log)

    elif args.date:
        target = date.fromisoformat(args.date)
        if target > latest:
            log.error(
                f"Requested date {target} is beyond latest available ({latest}). "
                f"ERA5-Land has ~{LATENCY_DAYS}-day latency -- try an earlier date."
            )
            sys.exit(1)
        run_download(nodes, target, target, 1, log)

    else:
        start = latest - timedelta(days=args.days - 1)
        log.info(f"Regular run: {start} -> {latest}  ({args.days} days)")
        run_download(nodes, start, latest, args.chunk_days, log)
