# soil_moisture_features.py
# ═══════════════════════════════════════════════════════════════════════
# ERA5-Land soil moisture → per-node ST-GNN feature pipeline
#
# Step 1: Download ERA5-Land for the Lee/Crookstown bounding box via CDS API
# Step 2: Compute saturation ratio + anomaly per layer
# Step 3: Spatially match ERA5-Land grid cells to graph nodes
# Step 4: Build final feature arrays for injection into build_dataset.py
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree

# ── Field capacity / saturation VWC for Irish soils ───────────────────
# Values for loam/clay-loam (dominant Cork soils, GSI soil map)
# Source: Clapp & Hornberger (1978) / H-TESSEL soil table
# Units: m³/m³
SOIL_SAT_VWC = {
    "swvl1": 0.472,   # 0–7 cm
    "swvl2": 0.472,   # 7–28 cm
    "swvl3": 0.452,   # 28–100 cm
    "swvl4": 0.452,   # 100–289 cm
}

# Layers to include in the feature vector (start with 1+2 for flash flood)
ACTIVE_LAYERS = ["swvl1", "swvl2"]

# ERA5-Land bounding box covering Lee + Crookstown catchments
# (51.6°N–52.2°N, 9.2°W–8.2°W) with 0.5° buffer
ERA5_BBOX = {
    "north": 52.3,
    "south": 51.5,
    "west":  -9.3,
    "east":  -8.1,
}


# ── Step 1: Download ERA5-Land from CDS ───────────────────────────────
def download_era5_land_sm(
    output_path: str,
    years: list[int],
    months: list[int] = list(range(1, 13)),
    layers: list[str] = ACTIVE_LAYERS,
) -> None:
    """
    DEPRECATED: single-block download fails silently with new CDS API.
    Use download_era5_land_sm_yearly() + merge_yearly_era5() instead.

    Left in place for backward compatibility — delegates to the
    corrected yearly downloader.
    """
    import os
    out_p   = Path(output_path)
    out_dir = out_p.parent / "yearly"
    paths   = download_era5_land_sm_yearly(
        output_dir=str(out_dir),
        years=years,
        months=months,
        layers=layers,
    )
    merge_yearly_era5(paths, output_path)


def download_era5_land_sm_yearly(
    output_dir:  str,
    years:       list[int],
    months:      list[int] = list(range(1, 13)),
    layers:      list[str] = ACTIVE_LAYERS,
) -> list[str]:
    """
    Download ERA5-Land soil moisture month-by-month (one CDS request per month).

    Why month-by-month instead of year-by-year:
        A full-year request (12 months × 31 days × 24 hr × Lee bbox × 2 vars)
        exceeds the CDS cost/size limit per request, causing a 403 error:
            "cost limits exceeded / Your request is too large".
        Monthly requests (~31 days each) stay well within the limit and match
        the approach used in the working download_era5_sm.py script
        (download_chunk_gridded, default --chunk-days 30).

    Request format mirrors download_era5_sm.py (download_chunk_gridded):
        "data_format":     "netcdf"
        "download_format": "unarchived"   ← required post-2024 CDS API
        "product_type":    ["reanalysis"]

    Parameters
    ----------
    output_dir : str
        Directory for per-month NetCDF files.
        Each file is saved as era5_land_sm_{year}_{month:02d}.nc
        Existing files larger than 10 KB are skipped (resume support).
    years : list[int]
        Calendar years, e.g. [2023, 2024, 2025, 2026].
    months : list[int]
        Months to include (default: all 12). For the current year,
        months within the NRT lag (7 days before today) are skipped.
    layers : list[str]
        Soil water layer keys, e.g. ["swvl1", "swvl2"].

    Returns
    -------
    list[str]  sorted paths to all downloaded per-month NetCDF files.
               Pass these to merge_yearly_era5() to create the combined file.

    Usage
    -----
    from src.data.soil_moisture_features import (
        download_era5_land_sm_yearly, merge_yearly_era5)

    paths = download_era5_land_sm_yearly(
        output_dir="dataset/era5/monthly",
        years=[2023, 2024, 2025, 2026],
    )
    merge_yearly_era5(paths, "dataset/era5/era5_land_sm_lee.nc")
    """
    import cdsapi
    import datetime
    import calendar
    import time as _time
    import zipfile
    import xarray as xr

    CDS_VAR_MAP = {
        "swvl1": "volumetric_soil_water_layer_1",
        "swvl2": "volumetric_soil_water_layer_2",
        "swvl3": "volumetric_soil_water_layer_3",
        "swvl4": "volumetric_soil_water_layer_4",
    }
    variables = [CDS_VAR_MAP[l] for l in layers]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    today     = datetime.date.today()
    safe_date = today - datetime.timedelta(days=7)

    # One CDS client for the entire session
    client = cdsapi.Client()

    saved: list[str] = []

    for year in sorted(years):
        for month in sorted(months):

            # Skip months beyond the NRT lag for the current year
            if year == today.year and month > safe_date.month:
                print(f"  [{year}-{month:02d}] Skipping: beyond NRT lag "
                      f"(safe up to {safe_date.month:02d})")
                continue
            if year > today.year:
                print(f"  [{year}] Skipping: future year")
                break

            out_path = out_dir / f"era5_land_sm_{year}_{month:02d}.nc"

            # Resume support: skip if already downloaded and valid
            if out_path.exists() and out_path.stat().st_size > 10_000:
                print(f"  [{year}-{month:02d}] Already exists "
                      f"({out_path.stat().st_size // 1024:,} KB) — skipping")
                saved.append(str(out_path))
                continue

            # All days in this month
            n_days = calendar.monthrange(year, month)[1]
            day_strs = [f"{d:02d}" for d in range(1, n_days + 1)]

            print(f"  [{year}-{month:02d}] Requesting {n_days} days × "
                  f"{len(variables)} variables …")

            # Request format identical to the working download_chunk_gridded()
            request = {
                "product_type":     ["reanalysis"],
                "variable":         variables,
                "year":             str(year),
                "month":            f"{month:02d}",
                "day":              day_strs,
                "time":             [f"{h:02d}:00" for h in range(24)],
                "area": [
                    ERA5_BBOX["north"],   # N
                    ERA5_BBOX["west"],    # W
                    ERA5_BBOX["south"],   # S
                    ERA5_BBOX["east"],    # E
                ],
                "data_format":      "netcdf",
                "download_format":  "unarchived",
            }

            success = False
            for attempt in range(1, 4):   # 3 attempts with backoff
                try:
                    client.retrieve("reanalysis-era5-land", request,
                                    str(out_path))

                    # Handle case where CDS returned a zip despite "unarchived"
                    if out_path.exists() and zipfile.is_zipfile(out_path):
                        print(f"  [{year}-{month:02d}] CDS returned zip — extracting …")
                        _recovered = _extract_zip_nc(out_path)
                        if _recovered:
                            out_path = _recovered
                        else:
                            raise ValueError("Zip extraction failed")

                    # Validate: catch silent NaN downloads
                    with xr.open_dataset(str(out_path), engine="netcdf4") as _ds:
                        _v = layers[0]
                        if _v in _ds:
                            _mean = float(_ds[_v].mean())
                            if not (_mean == _mean) or abs(_mean) < 1e-9:
                                raise ValueError(
                                    f"{_v}.mean()={_mean:.6f} — expected ~0.30 "
                                    f"m³/m³. CDS may have returned a zip archive "
                                    f"instead of a plain .nc despite "
                                    f"download_format='unarchived'.")
                            print(f"  [{year}-{month:02d}] OK — "
                                  f"{_v}.mean()={_mean:.4f} m³/m³  "
                                  f"→ {out_path.name}")

                    saved.append(str(out_path))
                    success = True
                    break

                except Exception as exc:
                    print(f"  [{year}-{month:02d}] Attempt {attempt}/3 failed: {exc}")
                    if out_path.exists():
                        out_path.unlink()
                    if attempt < 3:
                        wait = 60 * attempt
                        print(f"  Retrying in {wait}s …")
                        _time.sleep(wait)

            if not success:
                print(f"  [{year}-{month:02d}] ALL 3 ATTEMPTS FAILED — skipping")

    return sorted(saved)


def _extract_zip_nc(zip_path: Path) -> Path | None:
    """
    Extract NetCDF file(s) from a CDS zip delivery and merge into one file.
    Returns the path to the extracted/merged NetCDF, or None on failure.
    Internal helper for download_era5_land_sm_yearly().
    """
    import zipfile, tempfile
    import xarray as xr

    if not zipfile.is_zipfile(zip_path):
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not nc_members:
                return None
            with tempfile.TemporaryDirectory() as tmp_dir:
                extracted = []
                for member in nc_members:
                    dst = Path(tmp_dir) / Path(member).name
                    with zf.open(member) as src_f, open(dst, "wb") as df:
                        df.write(src_f.read())
                    extracted.append(dst)
                datasets = [xr.open_dataset(p, engine="netcdf4")
                            for p in extracted]
                merged = xr.merge(datasets, compat="override")
                tmp_out = zip_path.with_suffix(".nc.tmp")
                merged.to_netcdf(str(tmp_out))
                for ds in datasets: ds.close()
                merged.close()
        zip_path.unlink()
        tmp_out.rename(zip_path)
        return zip_path
    except Exception:
        return None


def merge_yearly_era5(
    yearly_paths: list[str],
    output_path:  str,
    overwrite:    bool = True,
) -> None:
    """
    Concatenate per-year ERA5-Land NetCDF files into one merged file.

    Call this after download_era5_land_sm_yearly().

    Parameters
    ----------
    yearly_paths : list[str]
        Paths returned by download_era5_land_sm_yearly(), in any order.
        Will be sorted by filename (which sorts by year).
    output_path : str
        Destination for the merged NetCDF (e.g. era5_land_sm_lee.nc).
    overwrite : bool
        If False and output_path exists, skip merging. Default True.
    """
    import xarray as xr

    out_p = Path(output_path)
    if out_p.exists() and not overwrite:
        print(f"Merge skipped — {out_p.name} exists (overwrite=False)")
        return

    if not yearly_paths:
        raise ValueError("No yearly files to merge — yearly_paths is empty")

    # Sort by path (filename encodes year)
    sorted_paths = sorted(yearly_paths)
    print(f"Merging {len(sorted_paths)} yearly files into {out_p.name} …")
    for p in sorted_paths:
        print(f"  + {Path(p).name}")

    # Detect the time dimension name from the first file
    with xr.open_dataset(sorted_paths[0]) as _probe:
        time_dim = next(
            (d for d in _probe.dims if "time" in d.lower()), "time")

    # open_mfdataset handles large files without loading into RAM
    with xr.open_mfdataset(
        sorted_paths,
        combine="nested",
        concat_dim=time_dim,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ) as merged:
        # Validate before writing
        first_var = list(merged.data_vars)[0]
        val = float(merged[first_var].mean())
        if not (val == val) or abs(val) < 1e-9:
            raise ValueError(
                f"Merged dataset has {first_var}.mean()={val:.6f} — "
                "check individual year files for NaN content")

        out_p.parent.mkdir(parents=True, exist_ok=True)
        merged.to_netcdf(str(out_p))

    size_mb = out_p.stat().st_size / 1024**2
    print(f"Merged: {out_p.name}  ({size_mb:.1f} MB)")
    print(f"  {first_var}.mean() = {val:.4f} m³/m³  "
          f"(expected ~0.28–0.38 for Irish Atlantic soils)")
    if 0.20 <= val <= 0.47:
        print("  Validation OK — values in physical range [0.20, 0.47]")
    else:
        print(f"  WARNING: {first_var}.mean()={val:.4f} is outside the "
              f"expected range [0.20, 0.47] for Irish clay-loam soils")


# ── Step 2: Feature engineering ───────────────────────────────────────
def compute_sm_features(
    era5_nc_path: str,
    layers: list[str] = ACTIVE_LAYERS,
    climatology_window_days: int = 15,
) -> xr.Dataset:
    """
    Load ERA5-Land netCDF and compute per-node or per-cell soil moisture features.

    Handles two dataset formats automatically:
      Format A — spatial grid:   dims (time, latitude, longitude)
                                 produced by reanalysis-era5-land with area bbox
      Format B — node-indexed:   dims (valid_time/time, node)
                                 pre-matched point time series

    For each layer computes three features:
      {layer}_raw         VWC (m³/m³)
      {layer}_sat_ratio   VWC / VWC_sat, clipped [0, 1]
      {layer}_anomaly     VWC − smoothed day-of-year climatology

    The anomaly uses numpy fancy-indexing to broadcast the climatology back
    to the full time axis. This avoids xarray's sel()-based broadcasting which
    can produce duplicate dimension names ('time', 'time', 'lat', 'lon') when
    the source DataArray carries 'time' as both a dimension and an auxiliary
    coordinate — a common occurrence in CF-convention ERA5-Land files.

    Parameters
    ----------
    era5_nc_path : str
        Path to the ERA5-Land NetCDF file.
    layers : list[str]
        Soil water layer identifiers, e.g. ['swvl1', 'swvl2'].
    climatology_window_days : int
        Smoothing window (days) for the DOY climatology. Default 15.
    """
    ds = xr.open_dataset(era5_nc_path, engine="netcdf4")
    print(ds.dims)

    # ── Normalise dimension names ─────────────────────────────────────
    # Squeeze out size-1 'expver' dimension (ERA5-Land NRT)
    if "expver" in ds.dims:
        ds = ds.squeeze("expver", drop=True)

    # Normalise time dimension.
    # ERA5-Land gridded files commonly have TWO time-related dimensions:
    #   'time'       size 1  — scalar forecast reference time (epoch only)
    #   'valid_time' size N  — actual hourly validity timestamps (what we need)
    # When both are present, drop the size-1 'time' and rename 'valid_time'.
    # When only 'valid_time' exists, rename it.
    # When only 'time' exists and it has N>1 steps, use it directly.
    if "valid_time" in ds.dims:
        if "time" in ds.dims:
            # Both present — drop the scalar 'time', keep 'valid_time'
            if ds.dims["time"] <= 1:
                ds = ds.squeeze("time", drop=True)
            else:
                # Unusual: both are multi-step. Keep the longer one.
                if ds.dims["valid_time"] >= ds.dims["time"]:
                    ds = ds.squeeze("time", drop=True) \
                         if ds.dims["time"] == 1 \
                         else ds.drop_dims("time")
        ds = ds.rename({"valid_time": "time"})
    elif "time" not in ds.dims:
        candidates = [d for d in ds.dims
                      if "time" in d.lower() and ds.dims[d] > 1]
        if not candidates:
            candidates = [d for d in ds.dims if "time" in d.lower()]
        if candidates:
            ds = ds.rename({candidates[0]: "time"})
        else:
            raise ValueError(
                f"No time dimension found in {era5_nc_path}. "
                f"Available dims: {list(ds.dims)}"
            )

    # Confirm 'time' now has the expected number of steps
    if ds.dims.get("time", 0) <= 1:
        raise ValueError(
            f"Time dimension has only {ds.dims.get('time',0)} step(s) after "
            f"normalisation — cannot compute day-of-year climatology.\n"
            f"Available dims: {dict(ds.dims)}\n"
            f"Check the ERA5-Land file structure with "
            f"soil_moisture_features.inspect_era5_file()"
        )

    # Detect format
    node_indexed = "node" in ds.dims

    # Format A: spatial grid — normalise lat/lon names
    if not node_indexed:
        rename_map = {}
        if "latitude"  in ds.dims: rename_map["latitude"]  = "lat"
        if "longitude" in ds.dims: rename_map["longitude"] = "lon"
        if rename_map:
            ds = ds.rename(rename_map)

    print(f"ERA5 dataset: format={'node-indexed' if node_indexed else 'spatial-grid'}")
    print(f"  dims:  {dict(ds.sizes)}")
    print(f"  vars:  {list(ds.data_vars)}")

    # Verify requested layers exist
    for layer in layers:
        if layer not in ds:
            raise KeyError(
                f"Variable '{layer}' not found in dataset. "
                f"Available variables: {list(ds.data_vars)}. "
                f"Check that --variable volumetric_soil_water_layer_1/2 "
                f"was included in the CDS request."
            )

    feature_arrays = {}

    for layer in layers:
        vwc = ds[layer]   # (time, node) or (time, lat, lon)

        # ── 1. Raw VWC ────────────────────────────────────────────────
        feature_arrays[f"{layer}_raw"] = vwc

        # ── 2. Saturation ratio ───────────────────────────────────────
        sat_ratio = (vwc / SOIL_SAT_VWC[layer]).clip(0.0, 1.0)
        sat_ratio.attrs["long_name"] = f"Saturation ratio {layer}"
        feature_arrays[f"{layer}_sat_ratio"] = sat_ratio

        # ── 3. Day-of-year climatology anomaly ────────────────────────
        # Build a smoothed DOY climatology, then subtract it from vwc.
        #
        # WHY NUMPY INDEXING (not xr.DataArray.sel):
        #   xr.DataArray.sel(dayofyear=xr.DataArray(doy_idx, dims="time"))
        #   introduces duplicate 'time' dimensions when the source DataArray
        #   carries 'time' as an auxiliary coordinate alongside the dayofyear
        #   dimension. The resulting dims become ('time', 'time', 'lat', 'lon')
        #   which makes broadcasting fail with:
        #     "broadcasting cannot handle duplicate dimensions"
        #   Using doy_smooth.values[row_indices] bypasses xarray entirely,
        #   giving a plain numpy array that is safe to wrap in a new DataArray
        #   with the original (time, ...) dims.
        doy_mean = vwc.groupby("time.dayofyear").mean("time")
        # doy_mean: (dayofyear, node) or (dayofyear, lat, lon)

        half = climatology_window_days // 2
        doy_smooth = (
            doy_mean
            .pad(dayofyear=half, mode="wrap")
            .rolling(dayofyear=climatology_window_days, center=True, min_periods=1)
            .mean()
            .isel(dayofyear=slice(half, -half if half > 0 else None))
        )

        # Map each timestep to its climatology row using numpy fancy indexing
        doy_vals   = doy_smooth.dayofyear.values          # (n_doy,)
        doy_idx    = vwc.time.dt.dayofyear.values         # (T,)
        doy_to_pos = {int(d): i for i, d in enumerate(doy_vals)}

        # Handle DOY values in vwc that might not be in the climatology
        # (can happen at dataset edges; fall back to nearest DOY in climatology)
        doy_set = set(doy_to_pos.keys())
        def nearest_doy(d):
            d = int(d)
            if d in doy_to_pos:
                return doy_to_pos[d]
            # wrap-around nearest for leap year edge cases
            nearest = min(doy_set, key=lambda x: min(abs(x-d), 366-abs(x-d)))
            return doy_to_pos[nearest]

        row_indices  = np.array([nearest_doy(d) for d in doy_idx])  # (T,)
        clim_values  = doy_smooth.values[row_indices]                # (T, ...) numpy

        # Wrap as clean DataArray with same dims/coords as vwc — no duplicates
        clim = xr.DataArray(
            clim_values.astype(np.float32),
            dims=vwc.dims,
            coords=vwc.coords,
            name=f"{layer}_climatology",
        )

        anomaly = vwc - clim
        anomaly.attrs["long_name"] = f"SM anomaly {layer}"
        feature_arrays[f"{layer}_anomaly"] = anomaly

    ds_features = xr.Dataset(feature_arrays)

    # Carry node coordinate metadata through for Format B
    if node_indexed:
        for coord in ("node", "ref", "node_name", "node_lat", "node_lon"):
            if coord in ds.coords:
                ds_features = ds_features.assign_coords({coord: ds.coords[coord]})

    return ds_features



def build_node_sm_lookup(
    ds_features: xr.Dataset,
    node_lons: np.ndarray,
    node_lats: np.ndarray,
) -> dict | None:
    """
    Build a spatial lookup from graph nodes to ERA5-Land grid cells.

    For node-indexed datasets (Format B): the data is already aligned to
    graph nodes, so this function returns None to signal that
    extract_node_sm_features should read directly by node index.

    For spatial-grid datasets (Format A): builds a cKDTree lookup mapping
    each gauge node to the nearest ERA5 grid cell.

    Returns
    -------
    dict {node_idx: (lat_idx, lon_idx)}  for Format A (spatial grid)
    None                                  for Format B (node-indexed)
    """
    # Format B: already node-indexed — no lookup needed
    if "node" in ds_features.dims:
        n_nodes = ds_features.sizes["node"]
        print(f"Node-indexed format detected ({n_nodes} nodes). "
              f"Skipping spatial lookup.")
        return None

    # Format A: spatial grid — build KDTree lookup
    era5_lats = ds_features.lat.values
    era5_lons = ds_features.lon.values

    grid_lons, grid_lats = np.meshgrid(era5_lons, era5_lats)
    grid_points  = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])
    query_points = np.column_stack([node_lats, node_lons])

    tree = cKDTree(grid_points)
    _, flat_indices = tree.query(query_points)

    lat_indices = flat_indices // len(era5_lons)
    lon_indices = flat_indices  % len(era5_lons)

    lookup = {i: (lat_indices[i], lon_indices[i]) for i in range(len(node_lats))}

    print("ERA5-Land grid -> node spatial matching:")
    for i, (li, loi) in lookup.items():
        dist_km = np.sqrt(
            ((node_lats[i] - era5_lats[li]) * 111) ** 2 +
            ((node_lons[i] - era5_lons[loi]) * 111
             * np.cos(np.radians(node_lats[i]))) ** 2
        )
        print(f"  Node {i:2d} -> ({era5_lats[li]:.2f}N, "
              f"{era5_lons[loi]:.2f}E)  dist={dist_km:.1f} km")

    return lookup



# ── Step 4: Extract time series per node and build feature matrix ──────
def extract_node_sm_features(
    ds_features: xr.Dataset,
    node_lookup: dict | None,
    target_timestamps: pd.DatetimeIndex,
    layers: list[str] = ACTIVE_LAYERS,
    feature_names: list[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract ERA5-Land SM features aligned to target_timestamps.

    Handles two cases:
      node_lookup is None  — node-indexed format: read directly by node
      node_lookup is dict  — spatial-grid format: look up by lat/lon index

    Returns
    -------
    sm_feature_matrix : np.ndarray  [T, N_nodes, n_sm_features]
    feature_names     : list[str]
    """
    if feature_names is None:
        feature_names = [
            f"{layer}_{suffix}"
            for layer in layers
            for suffix in ["raw", "sat_ratio", "anomaly"]
        ]

    n_features = len(feature_names)
    T          = len(target_timestamps)

    # ── Format B: node-indexed ────────────────────────────────────────
    if node_lookup is None:
        N_nodes = ds_features.sizes["node"]
        sm_matrix = np.full((T, N_nodes, n_features), np.nan, dtype=np.float32)

        # Normalise time coordinate name
        time_coord = "time" if "time" in ds_features.coords else "valid_time"

        era5_times = pd.DatetimeIndex(ds_features[time_coord].values)
        # Strip timezone from ERA5-Land timestamps if target is tz-naive.
        # ERA5-Land CDS files often return tz-aware UTC timestamps
        # (2023-01-01 00:00:00+00:00) while common_index / target_timestamps
        # is tz-naive (2023-01-01 00:00:00). pandas reindex() silently
        # returns all-NaN when comparing aware vs naive — no error raised.
        if era5_times.tz is not None and target_timestamps.tz is None:
            era5_times = era5_times.tz_localize(None)
        elif era5_times.tz is None and target_timestamps.tz is not None:
            era5_times = era5_times.tz_localize("UTC")

        for feat_idx, feat_name in enumerate(feature_names):
            if feat_name not in ds_features:
                continue
            feat_da = ds_features[feat_name]    # (time, node)

            for node_idx in range(N_nodes):
                node_series = feat_da.isel(node=node_idx).values
                node_ts     = pd.Series(node_series, index=era5_times)
                # ERA5-Land is hourly; align to 15-min target with nearest-hour fill
                node_aligned = node_ts.reindex(
                    target_timestamps, method="nearest", tolerance="1h"
                )
                sm_matrix[:, node_idx, feat_idx] = node_aligned.values

    # ── Format A: spatial grid with lookup ────────────────────────────
    else:
        N_nodes   = len(node_lookup)
        sm_matrix = np.full((T, N_nodes, n_features), np.nan, dtype=np.float32)

        time_coord = "time" if "time" in ds_features.coords else "valid_time"
        # Ensure era5 time index is tz-naive for reindex compatibility
        _era5_check = pd.DatetimeIndex(ds_features[time_coord].values)
        if _era5_check.tz is not None and target_timestamps.tz is None:
            # Convert era5 times to naive UTC before slicing
            ds_features = ds_features.assign_coords(
                {time_coord: _era5_check.tz_localize(None)})
        ds_sub = ds_features.sel(
            {time_coord: slice(
                str(target_timestamps[0]  - pd.Timedelta("1h")),
                str(target_timestamps[-1] + pd.Timedelta("1h")),
            )}
        )
        era5_times = pd.DatetimeIndex(ds_sub[time_coord].values)

        for feat_idx, feat_name in enumerate(feature_names):
            if feat_name not in ds_sub:
                continue
            feat_da = ds_sub[feat_name]   # (time, lat, lon)

            for node_idx, (lat_i, lon_i) in node_lookup.items():
                node_series = feat_da.isel(lat=lat_i, lon=lon_i).values
                node_ts     = pd.Series(node_series, index=era5_times)
                node_aligned = node_ts.reindex(
                    target_timestamps, method="nearest", tolerance="1h"
                )
                sm_matrix[:, node_idx, feat_idx] = node_aligned.values

    nan_rate = float(np.isnan(sm_matrix).mean())
    print(f"SM feature matrix: {sm_matrix.shape}  NaN rate: {nan_rate:.4f}")
    return sm_matrix, feature_names



# ── Step 5: Append to existing node feature tensor in build_dataset.py ─
def append_sm_to_node_features(
    existing_features: np.ndarray,  # (T, N, F_existing)
    sm_features: np.ndarray,        # (T, N, F_sm)
) -> np.ndarray:
    """
    Concatenate SM features onto the existing node feature matrix.
    Call this inside build_dataset.py after all other features are assembled.

    New feature vector per node per timestep:
      [gauge_level, discharge, precip, ..., swvl1_raw, swvl1_sat_ratio,
       swvl1_anomaly, swvl2_raw, swvl2_sat_ratio, swvl2_anomaly]
    """
    assert existing_features.shape[:2] == sm_features.shape[:2], (
        f"Shape mismatch: existing {existing_features.shape} vs "
        f"SM {sm_features.shape}"
    )
    combined = np.concatenate([existing_features, sm_features], axis=2)
    print(f"Feature vector extended: {existing_features.shape[2]} → "
          f"{combined.shape[2]} features per node")
    return combined


# ── Utility: print a quick sanity check for the Lee catchment ──────────
def sm_event_report(
    sm_matrix: np.ndarray,
    feature_names: list[str],
    target_timestamps: pd.DatetimeIndex,
    event_start: str,
    event_end: str,
    node_idx: int = 0,
) -> None:
    """
    Print layer-1 saturation ratio around a flood event for a given node.
    Useful for confirming the antecedent signal is physically sensible
    (should approach 1.0 before major Lee flood events).
    """
    mask = (target_timestamps >= event_start) & (target_timestamps <= event_end)
    sat1_idx = feature_names.index("swvl1_sat_ratio")
    anom1_idx = feature_names.index("swvl1_anomaly")

    print(f"\nSM event report — node {node_idx}, {event_start} to {event_end}")
    print(f"{'Timestamp':<22} {'swvl1_sat_ratio':>16} {'swvl1_anomaly':>14}")
    for t, ts in enumerate(target_timestamps[mask]):
        print(
            f"  {str(ts):<20}"
            f"  {sm_matrix[np.where(mask)[0][t], node_idx, sat1_idx]:>14.3f}"
            f"  {sm_matrix[np.where(mask)[0][t], node_idx, anom1_idx]:>14.4f}"
        )

if __name__ == "__main__":
    # paths = download_era5_land_sm_yearly(
    #     output_dir="dataset/era5/monthly",
    #     years=[2023, 2024, 2025, 2026],
    # )
    import glob
    paths = sorted(glob.glob("dataset/era5/monthly/era5_land_sm_*.nc"))
    merge_yearly_era5(paths, "dataset/era5/era5_land_sm_lee.nc")