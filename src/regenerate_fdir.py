"""
regenerate_fdir.py
══════════════════════════════════════════════════════════════════════
Regenerates dataset/graph/fdir.npz using pysheds with proper depression
filling and flat-area routing.

The previous fdir.npz was computed from raw D8 (steepest descent) without
depression filling. On the Lee catchment's flat floodplain, raw D8 creates
1-pixel spaghetti flow paths — node 19114 (Carrigrohane) was assigned only
19 pixels when it should cover ~130,000.

This script uses pysheds' full pre-processing pipeline:
    fill_pits → fill_depressions → resolve_flats → flowdir

This produces a proper D8 raster that propagates upstream catchments
correctly across flat terrain.

Usage
──────
    pip install pysheds
    python src/regenerate_fdir.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio

BASE_DIR  = Path(__file__).resolve().parent.parent
DEM_PATH  = BASE_DIR / "dataset/dem/COP-DEM-30m_itm.tif"
FDIR_PATH = BASE_DIR / "dataset/graph/fdir.npz"

# pysheds D8 code → (dr, dc)  [row-direction, col-direction]
PYSHEDS_TO_DRDC = {
    64: (-1, 0), 128: (-1, 1),  1: (0,  1),   2: (1,  1),
     4: ( 1, 0),   8: ( 1,-1), 16: (0, -1),  32: (-1,-1),
    -1: ( 0, 0),   0: ( 0,  0),
}


def run():
    from pysheds.grid import Grid

    print(f"DEM: {DEM_PATH}")
    print(f"Output: {FDIR_PATH}")
    print()

    if not DEM_PATH.exists():
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")

    # Load nodata mask from rasterio (pysheds handles nodata internally)
    with rasterio.open(DEM_PATH) as src:
        dem_arr  = src.read(1).astype(np.float32)
        nodata   = src.nodata if src.nodata is not None else -9999.0
        H, W     = src.height, src.width

    nan_mask = (dem_arr == nodata) | np.isnan(dem_arr)
    print(f"DEM shape: {H}×{W}  nodata pixels: {nan_mask.sum():,}")

    # ── pysheds pipeline ────────────────────────────────────────────
    print("Loading DEM into pysheds...")
    grid = Grid.from_raster(str(DEM_PATH))
    dem  = grid.read_raster(str(DEM_PATH))

    print("Filling pits...")
    pit_filled = grid.fill_pits(dem)

    print("Filling depressions...")
    flooded = grid.fill_depressions(pit_filled)

    print("Resolving flats (adds small gradient across flat areas)...")
    inflated = grid.resolve_flats(flooded)

    print("Computing D8 flow direction...")
    fdir = grid.flowdir(inflated)
    fdir_arr = np.array(fdir, dtype=np.int64)

    # ── Convert to (dr, dc) ─────────────────────────────────────────
    print("Converting to (dr, dc) arrays...")
    dr = np.zeros((H, W), dtype=np.int8)
    dc = np.zeros((H, W), dtype=np.int8)
    for code, (d_r, d_c) in PYSHEDS_TO_DRDC.items():
        mask = fdir_arr == code
        dr[mask] = d_r
        dc[mask] = d_c

    # Sanity check
    n_unresolved = int(((dr == 0) & (dc == 0) & ~nan_mask).sum())
    n_valid      = int((~nan_mask).sum())
    pct = n_unresolved / n_valid * 100 if n_valid > 0 else 0
    print(f"Unresolved flow direction: {n_unresolved:,} / {n_valid:,} ({pct:.1f}%)")
    if pct > 5:
        print("WARNING: >5% unresolved — check DEM quality")

    # ── Save ─────────────────────────────────────────────────────────
    FDIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FDIR_PATH, dr=dr, dc=dc,
                        nan_mask=nan_mask.astype(np.bool_))
    size_mb = FDIR_PATH.stat().st_size / 1024**2
    print(f"\nSaved: {FDIR_PATH}  ({size_mb:.1f} MB)")
    print("\nNext step: re-run validation or flood map generation scripts.")


if __name__ == "__main__":
    run()
