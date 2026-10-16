"""
compute_kirpich_inputs.py  —  DEM-derived hydraulic length and slope per gauge
==============================================================================
Revision audit (reviewer Priority 1, manuscript Eq. 9).

The submitted unit_hydrograph.py estimated the Kirpich slope as
    S = max(p90 - gauge_datum, max upstream-edge elev_drop) / (1500 * sqrt(A))
i.e. a BANKFULL DEPTH divided by a hydraulic length. That is not a channel
slope: with the submitted graph files, 15 of 27 nodes hit the 0.0002 slope
floor and 20 of 27 nodes then hit the 8-hour Tc ceiling (e.g. Ballincolly,
3.7 km2, Tc = 4.4 h). Tc was therefore set mostly by its clipping bounds.

This script replaces that proxy with the quantities Kirpich actually needs,
taken from the same Copernicus DEM / Priority-Flood / D8 pipeline already
used for the HAND edges (precompute_hand_edges.py):

    L_m   length of the longest D8 flow path from the gauge pixel to the
          catchment divide, following the maximum-accumulation upstream
          neighbour (main stem) at every step
    S     (z_top - z_gauge) / L_m along that path (raw DEM elevations;
          also reports the 10-85 % slope, the more robust variant)

Gauges are snapped to the highest-accumulation cell within SNAP_RADIUS_M.

Output: dataset/graph/kirpich_inputs.csv
        node_idx, ref, name, L_m, S_end_to_end, S_10_85, z_gauge_m, z_top_m,
        snap_dist_m, upstream_area_km2_dem
unit_hydrograph.py uses this file automatically when it exists.

Usage
-----
    python src/data/compute_kirpich_inputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src/data"))

DEM_PATH = BASE_DIR / "dataset/dem/COP-DEM-30m.tif"
NODES = BASE_DIR / "dataset/graph/nodes.csv"
OUT = BASE_DIR / "dataset/graph/kirpich_inputs.csv"
SNAP_RADIUS_M = 150.0          # fallback (no reference area)
AREA_SNAP_RADIUS_M = 600.0     # area-matched snapping radius

_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def snap_to_stream(r: int, c: int, acc: np.ndarray, radius_px: int,
                   target_cells: float | None = None) -> tuple[int, int]:
    """Snap a gauge to the D8 network. With a published catchment area
    (target_cells), pick the cell in the window whose upstream area best
    matches it (min |log ratio|, distance as tie-break); otherwise take the
    maximum-accumulation cell. Max-accumulation snapping alone put several
    gauges on the wrong channel (e.g. Killaclug 0.4 km2 vs 74.8 km2)."""
    H, W = acc.shape
    r0, r1 = max(0, r - radius_px), min(H, r + radius_px + 1)
    c0, c1 = max(0, c - radius_px), min(W, c + radius_px + 1)
    win = acc[r0:r1, c0:c1].astype(float)
    if target_cells is None or target_cells <= 0:
        k = int(np.nanargmax(win))
    else:
        rr, cc = np.mgrid[r0:r1, c0:c1]
        dist = np.hypot(rr - r, cc - c) / max(radius_px, 1)
        with np.errstate(divide="ignore"):
            err = np.abs(np.log(np.maximum(win, 1) / target_cells))
        k = int(np.nanargmin(err + 0.05 * dist))
    return r0 + k // win.shape[1], c0 + k % win.shape[1]


def trace_main_stem_upstream(r: int, c: int, dr: np.ndarray, dc: np.ndarray,
                             acc: np.ndarray, nan_mask: np.ndarray,
                             cell_m: float) -> list[tuple[int, int, float]]:
    """Return [(row, col, cumulative_length_m)] from gauge to divide.
    Pure function on arrays (unit-tested on a synthetic DEM)."""
    H, W = acc.shape
    path = [(r, c, 0.0)]
    seen = {(r, c)}
    L = 0.0
    while True:
        best, best_acc, best_d = None, -1.0, 0.0
        for di, dj in _NEIGH:
            nr, nc = r + di, c + dj
            if not (0 <= nr < H and 0 <= nc < W) or nan_mask[nr, nc]:
                continue
            if nr + int(dr[nr, nc]) == r and nc + int(dc[nr, nc]) == c:
                if acc[nr, nc] > best_acc:
                    best, best_acc = (nr, nc), float(acc[nr, nc])
                    best_d = cell_m * (1.41421356 if di and dj else 1.0)
        if best is None or best in seen:
            return path
        r, c = best
        seen.add(best)
        L += best_d
        path.append((r, c, L))


def path_slopes(path: list[tuple[int, int, float]], dem: np.ndarray) -> dict:
    z = np.array([dem[r, c] for r, c, _ in path], dtype=float)
    d = np.array([L for _, _, L in path], dtype=float)
    L = float(d[-1])
    if L <= 0:
        return {"L_m": np.nan, "S_end_to_end": np.nan, "S_10_85": np.nan,
                "z_gauge_m": float(z[0]), "z_top_m": float(z[0])}
    s_end = (np.nanmax(z) - z[0]) / L
    z10 = np.interp(0.10 * L, d, z)
    z85 = np.interp(0.85 * L, d, z)
    s1085 = (z85 - z10) / (0.75 * L)
    return {"L_m": L, "S_end_to_end": float(s_end), "S_10_85": float(s1085),
            "z_gauge_m": float(z[0]), "z_top_m": float(np.nanmax(z))}


def load_d8(dem_path: Path = DEM_PATH) -> dict:
    """Copernicus DEM -> ITM, Priority-Flood+eps fill, strict-downhill D8 and
    accumulation. Shared by compute_kirpich_inputs.py and
    compute_river_distances.py so both use the same flow network."""
    import rasterio
    from precompute_hand_edges import (reproject_dem_to_itm, fill_pits_priority_flood,
                                       compute_d8_and_accumulation)
    itm = reproject_dem_to_itm(dem_path, dem_path.parent / (dem_path.stem + "_itm.tif"))
    with rasterio.open(itm) as src:
        dem = src.read(1).astype(np.float64)
        aff = src.transform
        nodata = src.nodata if src.nodata is not None else -9999.0
    nan_mask = (dem == nodata) | np.isnan(dem)
    dem[nan_mask] = np.nan
    filled = fill_pits_priority_flood(dem, nan_mask)
    dr, dc, acc = compute_d8_and_accumulation(filled, nan_mask)
    return {"dem": dem, "aff": aff, "cell": abs(aff.a), "nan_mask": nan_mask,
            "dr": dr, "dc": dc, "acc": acc}


def snap_gauges(nodes_path: Path, acc: np.ndarray, aff, cell: float) -> list[dict]:
    """Snap every gauge to the D8 network (area-matched where a published
    catchment area exists). Returns one dict per node in nodes.csv order."""
    from precompute_hand_edges import nodes_to_itm, world_to_pixel
    nodes = pd.read_csv(nodes_path)
    E, Nn, _ = nodes_to_itm(nodes_path)
    out = []
    for i in range(len(nodes)):
        r, c = world_to_pixel(float(E[i]), float(Nn[i]), aff)
        area = float(np.expm1(nodes.loc[i, "log_catchment_area_km2"]))
        if area > 0.5:
            tgt = area * 1e6 / cell ** 2
            for rad in (AREA_SNAP_RADIUS_M, 1500.0):
                rs, cs = snap_to_stream(r, c, acc, int(round(rad / cell)), tgt)
                if abs(np.log(max(acc[rs, cs], 1) / tgt)) < np.log(1.5):
                    break
        else:
            rs, cs = snap_to_stream(r, c, acc, int(round(SNAP_RADIUS_M / cell)))
        a_dem = float(acc[rs, cs] * cell * cell / 1e6)
        out.append({"r0": r, "c0": c, "r": rs, "c": cs, "area_km2_opw": area,
                    "upstream_area_km2_dem": a_dem,
                    "snap_dist_m": float(np.hypot(rs - r, cs - c) * cell),
                    "area_match_ok": bool(area > 0.5 and abs(np.log(
                        max(a_dem, 1e-6) / area)) < np.log(1.5))})
    return out


def main() -> None:
    from precompute_hand_edges import nodes_to_itm
    g = load_d8()
    dem, nan_mask, cell = g["dem"], g["nan_mask"], g["cell"]
    dr, dc, acc = g["dr"], g["dc"], g["acc"]

    nodes = pd.read_csv(NODES)
    _, _, refs = nodes_to_itm(NODES)
    snaps = snap_gauges(NODES, acc, g["aff"], cell)
    rows = []
    for i in range(len(nodes)):
        sn = snaps[i]
        rs, cs = sn["r"], sn["c"]
        path = trace_main_stem_upstream(rs, cs, dr, dc, acc, nan_mask, cell)
        sl = path_slopes(path, dem)
        rows.append({"node_idx": i, "ref": refs[i], "name": nodes.loc[i, "name"],
                     **sl, "snap_dist_m": sn["snap_dist_m"],
                     "upstream_area_km2_dem": sn["upstream_area_km2_dem"],
                     "area_km2_opw": sn["area_km2_opw"],
                     # usable for Kirpich only if the snapped cell drains the
                     # published catchment (within a factor 1.5)
                     "area_match_ok": sn["area_match_ok"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
