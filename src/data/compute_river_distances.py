"""
compute_river_distances.py  —  along-channel reach lengths from the D8 network
===============================================================================
Revision audit (A20).

river_dist_km in graph_metadata.json / edges.csv is used by
  * the static ST-GNN edge attribute [0],
  * unit_hydrograph.py (reach slope = drop / L and routing lag = L / celerity),
  * precompute_backwater_edges.py (nearest upstream gauge),
  * scenario_generator.py (S6 backwater decay length),
  * compute_edge_features.py (DFC river distance and travel time).

The submitted values are inconsistent: 17 of 26 equal the straight-line
distance to within 1 % (impossible along a river), one is SHORTER than the
straight line (Fitzgerald's Park -> Pope's Quay, 1.07 vs 1.78 km) and four
are 3-6x the straight line (Cooldaniel -> Carrigadrohid 67.0 km for 11.7 km;
Carrigadrohid -> Inniscarra 54.4 km for 13.9 km).

Method
------
Each gauge is snapped to the D8 network exactly as in compute_kirpich_inputs.py
(area-matched snapping). For every river edge s -> d the D8 flow path is
followed downstream from s; the reach length is the path length at the cell
of closest approach to d. The D8 value is accepted when
    closest approach <= MAX_MISS_M,  the two gauges snap to different cells,
    and sinuosity (D8 / straight line) >= MIN_SINUOSITY.
Otherwise (the city / tidal reaches, where the Lee splits into two channels
and the tidal gauges have no catchment area to snap with) the reach length is
max(submitted value, straight-line distance), i.e. never shorter than the
straight line. Every edge's source is reported.

D8 paths cut meander corners at 30 m resolution, so the D8 lengths are lower
bounds (typically a few percent short); they are consistent across reaches.

Outputs
-------
  dataset/graph/river_distances_report.csv   per-edge audit table
  dataset/graph/graph_metadata.json          edges[].river_dist_km updated,
                                             submitted value kept as
                                             river_dist_km_submitted
  dataset/graph/edges.csv                    river_dist_km updated

Usage
-----
    python src/data/compute_river_distances.py --dry-run
    python src/data/compute_river_distances.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src/data"))

GRAPH_DIR = BASE_DIR / "dataset/graph"
MAX_MISS_M = 150.0
MIN_SINUOSITY = 0.98


def trace_downstream_length(r: int, c: int, tr: int, tc: int,
                            dr: np.ndarray, dc: np.ndarray, cell_m: float,
                            max_steps: int = 200_000) -> tuple[float, float]:
    """Follow D8 from (r, c); return (path length in m at the closest
    approach to (tr, tc), closest approach distance in m). Pure function."""
    H, W = dr.shape
    L = 0.0
    best_d, best_L = float(np.hypot(r - tr, c - tc)), 0.0
    seen = {(r, c)}
    for _ in range(max_steps):
        di, dj = int(dr[r, c]), int(dc[r, c])
        if di == 0 and dj == 0:
            break
        r, c = r + di, c + dj
        if not (0 <= r < H and 0 <= c < W) or (r, c) in seen:
            break
        seen.add((r, c))
        L += cell_m * (1.41421356 if di and dj else 1.0)
        d = float(np.hypot(r - tr, c - tc))
        if d < best_d:
            best_d, best_L = d, L
        if d == 0:
            break
    return best_L, best_d * cell_m


def choose_length(d8_km: float, miss_m: float, same_cell: bool,
                  straight_km: float, submitted_km: float) -> tuple[float, str]:
    """Acceptance rule (unit-tested)."""
    if (not same_cell and miss_m <= MAX_MISS_M
            and d8_km >= MIN_SINUOSITY * straight_km and d8_km > 0):
        return d8_km, "d8"
    return max(submitted_km, straight_km), "fallback_max(submitted,straight)"


def compute(edges: pd.DataFrame, nodes: pd.DataFrame, snaps: list[dict],
            dr: np.ndarray, dc: np.ndarray, cell: float) -> pd.DataFrame:
    xy = nodes[["easting_itm", "northing_itm"]].to_numpy(float)
    rows = []
    for e in edges.itertuples():
        s, d = int(e.src_idx), int(e.dst_idx)
        a, b = snaps[s], snaps[d]
        L_m, miss = trace_downstream_length(a["r"], a["c"], b["r"], b["c"], dr, dc, cell)
        straight = float(np.linalg.norm(xy[s] - xy[d]) / 1000.0)
        sub = float(getattr(e, "river_dist_km_submitted", e.river_dist_km))
        same = (a["r"], a["c"]) == (b["r"], b["c"])
        new, src = choose_length(L_m / 1000.0, miss, same, straight, sub)
        rows.append({"src_ref": str(e.src_ref), "dst_ref": str(e.dst_ref),
                     "src_idx": s, "dst_idx": d,
                     "src_name": nodes.loc[s, "name"], "dst_name": nodes.loc[d, "name"],
                     "river_dist_km_submitted": round(sub, 4),
                     "straight_km": round(straight, 4),
                     "d8_km": round(L_m / 1000.0, 4), "d8_miss_m": round(miss, 1),
                     "same_snap_cell": same,
                     "river_dist_km": round(new, 4), "source": src,
                     "sinuosity": round(new / straight, 3) if straight > 0 else np.nan})
    return pd.DataFrame(rows)


def apply(report: pd.DataFrame, graph_dir: Path) -> None:
    key = {(r.src_ref, r.dst_ref): r.river_dist_km for r in report.itertuples()}
    meta_p = graph_dir / "graph_metadata.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    for group in ("edges", "removed_edges_revision"):
        for e in meta.get(group, []):
            k = (str(e["src_ref"]), str(e["dst_ref"]))
            kr = (k[1], k[0])                 # removed reverse duplicates
            v = key.get(k, key.get(kr))
            if v is None:
                continue
            e.setdefault("river_dist_km_submitted", e["river_dist_km"])
            e["river_dist_km"] = float(v)
    meta["river_dist_method"] = ("D8 along-channel length (compute_river_distances.py); "
                                 "fallback max(submitted, straight line) on urban/tidal reaches")
    meta_p.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    ep = graph_dir / "edges.csv"
    ed = pd.read_csv(ep, dtype={"src_ref": str, "dst_ref": str})
    if "river_dist_km_submitted" not in ed:
        ed["river_dist_km_submitted"] = ed["river_dist_km"]
    ed["river_dist_km"] = [key.get((s, d), key.get((d, s), v)) for s, d, v in
                           zip(ed.src_ref, ed.dst_ref, ed.river_dist_km)]
    ed.to_csv(ep, index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--graph-dir", default=str(GRAPH_DIR))
    args = ap.parse_args(argv)
    gdir = Path(args.graph_dir)

    from compute_kirpich_inputs import load_d8, snap_gauges
    from utils.graph_checks import river_edges
    g = load_d8()
    nodes = pd.read_csv(gdir / "nodes.csv")
    snaps = snap_gauges(gdir / "nodes.csv", g["acc"], g["aff"], g["cell"])
    edges = river_edges(pd.read_csv(gdir / "edges.csv", dtype={"src_ref": str, "dst_ref": str}))
    rep = compute(edges, nodes, snaps, g["dr"], g["dc"], g["cell"])
    with pd.option_context("display.width", 200):
        print(rep[["src_name", "dst_name", "river_dist_km_submitted", "straight_km",
                   "d8_km", "d8_miss_m", "river_dist_km", "sinuosity", "source"]]
              .to_string(index=False))
    n_d8 = int((rep.source == "d8").sum())
    print(f"\n{n_d8}/{len(rep)} reaches from D8, {len(rep) - n_d8} fallback")
    if args.dry_run:
        print("--dry-run: nothing written")
        return
    rep.to_csv(gdir / "river_distances_report.csv", index=False)
    apply(rep, gdir)
    print(f"Wrote river_distances_report.csv, graph_metadata.json, edges.csv in {gdir}")


if __name__ == "__main__":
    main()
