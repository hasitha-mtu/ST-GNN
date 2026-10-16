"""
unit_hydrograph.py  —  Calibrate response-kernel and routing parameters
========================================================================
Writes (dataset/graph/):
    uh_params.json       per-node Tc, Tp, Tb, peak response rate + provenance
    routing_lags.json    per-edge travel time (15-min steps) + provenance

REVISION AUDIT — what changed and why
-------------------------------------
1. Kirpich units (reviewer Priority 1). Tc[min] = 0.0195 L[m]^0.77 S^-0.385
   returns MINUTES; hours = minutes / 60 (scenarios/hydro.py). The manuscript
   text "Tc in hours" is wrong for this form and must be corrected.
2. Kirpich SLOPE. The submitted code used (bankfull depth or upstream-edge
   gauge-datum drop) / hydraulic length, which is not a channel slope; most
   nodes then hit the slope floor and the 8 h Tc ceiling (see
   data/compute_kirpich_inputs.py). Priority order is now:
     (a) DEM longest-flow-path L and 10-85 % slope (kirpich_inputs.csv);
     (b) otherwise L = 1500 sqrt(A) and the median WATER-SURFACE slope of the
         node's incident, downhill, non-reservoir river reaches.
   Every node records `slope_source` and whether Tc was clipped. The run
   warns if more than 25 % of nodes sit on a clipping bound.
3. Catchment area: nodes.csv stores log(A + 1), so A = exp(.) - 1 (the
   submitted code used exp(.), adding 1 km2 to every node).
4. Triangular kernel: Tp = D/2 + 0.6 Tc and the STANDARD SCS base
   Tb = 2.67 Tp (was 2.67 Tc).
5. Calibration uses the TRAINING period only (scenario targets are later
   evaluated on test-period base windows, so the generator must not be
   fitted on them), and only real (non-gap-filled) observations.
6. Routing: water-surface slopes; destination depth from utils.datum (the
   headrace local-zero datum no longer gives a 45-61 m "bankfull depth"
   that collapsed four reservoir edges to the 1-step minimum lag); for
   edges INTO a reservoir pool the source reach depth is used. Uphill
   reverse duplicates are excluded.
7. Uncalibrated nodes inherit a rate from the GEOGRAPHICALLY nearest
   calibrated node (ITM coordinates), not the nearest node index.

Usage
-----
    python src/scenarios/unit_hydrograph.py [--validate]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate

BASE_DIR = Path(__file__).resolve().parent.parent.parent
for _p in (BASE_DIR, BASE_DIR / "src", BASE_DIR / "src/scenarios"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scenarios.hydro import (STEP_MIN, STEP_HR, kirpich_tc_minutes,
                             scs_triangle_times)
from utils.datum import bankfull_depth_proxy
from utils.graph_checks import river_edges

PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
OUT_DIR   = GRAPH_DIR

F_RAINFALL   = 4
F_STAGE_ANOM = 0
MANNING_N    = 0.035
MIN_EVENTS   = 3
TC_BOUNDS_HR = (0.25, 8.0)
SLOPE_FLOOR  = 0.0002
CLIP_WARN_FRAC = 0.25


def node_area_km2(nodes_df: pd.DataFrame) -> np.ndarray:
    """nodes.csv stores log(A + 1) (graph_builder.py)."""
    return np.clip(np.expm1(nodes_df["log_catchment_area_km2"].to_numpy(float)), 0.0, None)


def water_surface_level(nodes_df: pd.DataFrame) -> np.ndarray:
    """Typical water-surface elevation per node for reach slopes."""
    if "p50_mAOD_train" in nodes_df.columns:
        ws = nodes_df["p50_mAOD_train"].to_numpy(float)
        dat = nodes_df["gauge_datum_mOSGM15"].to_numpy(float)
        return np.where(np.isfinite(ws), ws, dat)
    return nodes_df["gauge_datum_mOSGM15"].to_numpy(float)


def reach_slopes(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.Series:
    """Water-surface slope per edge (NaN where not physically meaningful)."""
    ws = water_surface_level(nodes_df)
    is_res = nodes_df["is_reservoir"].to_numpy().astype(bool)
    is_tid = nodes_df["is_tidal"].to_numpy().astype(bool)
    from utils.datum import abs_datum_valid
    ok_datum = abs_datum_valid(nodes_df)
    out = []
    for e in edges_df.itertuples():
        s, d = int(e.src_idx), int(e.dst_idx)
        L = float(e.river_dist_km) * 1000.0
        if L < 100 or is_tid[s] or is_tid[d] or not (ok_datum[s] and ok_datum[d]):
            out.append(np.nan); continue
        drop = ws[s] - ws[d]
        out.append(drop / L if drop > 0 else np.nan)
    return pd.Series(out, index=edges_df.index)


# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Tc per node
# ══════════════════════════════════════════════════════════════════════════

def compute_tc_kirpich(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                       kirpich_inputs: pd.DataFrame | None = None) -> pd.DataFrame:
    edges_df = river_edges(edges_df)
    area = node_area_km2(nodes_df)
    slopes = reach_slopes(edges_df, nodes_df)
    is_res = nodes_df["is_reservoir"].to_numpy().astype(bool)
    median_slope = float(np.nanmedian(slopes)) if np.isfinite(slopes).any() else 0.002
    rows = []
    for i in range(len(nodes_df)):
        src = "fallback_median_reach_slope"
        L = 1500.0 * np.sqrt(max(area[i], 0.01))
        S = np.nan
        if kirpich_inputs is not None and i in set(kirpich_inputs.node_idx):
            k = kirpich_inputs.set_index("node_idx").loc[i]
            area_ok = bool(k.get("area_match_ok", True))
            if area_ok and np.isfinite(k.L_m) and k.L_m > 0:
                L = float(k.L_m)
                S = float(k.S_10_85) if np.isfinite(k.S_10_85) and k.S_10_85 > 0 \
                    else float(k.S_end_to_end)
                src = "dem_longest_flow_path"
        if not np.isfinite(S) or S <= 0:
            inc = edges_df[((edges_df.dst_idx == i) | (edges_df.src_idx == i))]
            inc = inc[~(is_res[inc.src_idx.to_numpy()] | is_res[inc.dst_idx.to_numpy()])]
            cand = slopes.loc[inc.index].dropna()
            if len(cand):
                S, src = float(cand.median()), "incident_reach_water_surface_slope"
            else:
                S = median_slope
        S_used = max(S, SLOPE_FLOOR)
        tc_min = kirpich_tc_minutes(L, S_used)
        tc_hr_raw = tc_min / 60.0
        tc_hr = float(np.clip(tc_hr_raw, *TC_BOUNDS_HR))
        rows.append({"node_idx": i, "L_m": L, "S": S, "S_used": S_used,
                     "slope_source": src, "tc_min_raw": tc_min,
                     "tc_hr": tc_hr, "tc_clipped": bool(tc_hr != tc_hr_raw),
                     "slope_floored": bool(S_used != S)})
    df = pd.DataFrame(rows).set_index("node_idx")
    tid = nodes_df["is_tidal"].to_numpy().astype(bool)
    big = node_area_km2(nodes_df) >= 300
    ordinary = ~tid & ~big
    n_clip = int(df.tc_clipped[ordinary].sum())
    print(f"  Tc clipped: {n_clip}/{int(ordinary.sum())} ordinary gauges; "
          f"{int(df.tc_clipped[big].sum())}/{int(big.sum())} large (>=300 km2) "
          f"main-stem/reservoir gauges sit on the {TC_BOUNDS_HR[1]} h cap "
          f"(Kirpich is a small-catchment formula; the cap is binding there); "
          f"tidal gauges use fixed parameters.")
    if df.tc_clipped[ordinary].mean() > CLIP_WARN_FRAC:
        print("  WARNING: Tc is bound-dominated for ordinary gauges - provide "
              "dataset/graph/kirpich_inputs.csv (compute_kirpich_inputs.py).")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — Routing lags
# ══════════════════════════════════════════════════════════════════════════

def compute_kinematic_lag(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    from utils.datum import abs_datum_valid
    edges_df = river_edges(edges_df)
    depth = bankfull_depth_proxy(nodes_df)
    # A channel-depth proxy only exists at ordinary river gauges: not at
    # tidal gauges (no p90) and not at pool gauges with a local-zero datum.
    depth_ok = abs_datum_valid(nodes_df) & ~nodes_df["is_tidal"].to_numpy().astype(bool)
    if "stage_ref_low_m" in nodes_df:
        # pool gauges (local-zero datum): drawdown range is not a channel depth
        depth_ok &= np.abs(nodes_df["stage_ref_low_m"].to_numpy(float)
                           - nodes_df["gauge_datum_mOSGM15"].to_numpy(float)) < 0.5
    med_depth = float(np.median(depth[depth_ok])) if depth_ok.any() else 1.0
    slopes = reach_slopes(edges_df, nodes_df)
    med = float(np.nanmedian(slopes)) if np.isfinite(slopes).any() else 0.001
    rows = []
    for k, e in edges_df.iterrows():
        s, d = int(e.src_idx), int(e.dst_idx)
        dist_m = float(e.river_dist_km) * 1000.0
        if dist_m < 100:
            rows.append({"src": s, "dst": d, "lag_steps": 1, "source": "short_reach"}); continue
        S = slopes.loc[k]
        s_src = "water_surface"
        if not np.isfinite(S):
            S, s_src = med, "median_reach_slope"
        S = max(S, 0.0001)
        # Depth of the conveying reach: destination gauge if it has a valid
        # channel-depth proxy, else the source gauge, else the network median.
        d_bf = depth[d] if depth_ok[d] else (depth[s] if depth_ok[s] else med_depth)
        d_bf = max(float(d_bf), 0.3)
        R = 0.6 * d_bf
        V = max((1.0 / MANNING_N) * R ** (2 / 3) * S ** 0.5, 0.1)
        c = (5.0 / 3.0) * V
        lag = max(1, int(round(dist_m / c / (STEP_MIN * 60))))
        rows.append({"src": s, "dst": d, "lag_steps": lag, "source": "kinematic",
                     "slope": S, "slope_source": s_src, "depth_m": d_bf,
                     "celerity_ms": c})
    return pd.DataFrame(rows)


def empirical_routing_lag(stage: np.ndarray, valid: np.ndarray,
                          edges_df: pd.DataFrame, max_lag_steps: int = 48) -> dict:
    """Peak normalised cross-correlation lag (training period, real obs)."""
    edges_df = river_edges(edges_df)
    out = {}
    for e in edges_df.itertuples():
        s, d = int(e.src_idx), int(e.dst_idx)
        v = valid[:, s] & valid[:, d] & np.isfinite(stage[:, s]) & np.isfinite(stage[:, d])
        if v.sum() < 500:
            continue
        a = np.where(v, stage[:, s] - np.nanmean(stage[v, s]), 0.0)
        b = np.where(v, stage[:, d] - np.nanmean(stage[v, d]), 0.0)
        xc = correlate(b, a, mode="full")
        mid = len(a) - 1
        win = xc[mid: mid + max_lag_steps + 1]          # non-negative lags only
        lag = int(np.argmax(win))
        corr = float(win[lag]) / (np.std(a[v]) * np.std(b[v]) * v.sum() + 1e-12)
        if corr > 0.50 and lag > 0:
            out[(s, d)] = (lag, corr)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Step 3 — Peak response rate
# ══════════════════════════════════════════════════════════════════════════

def calibrate_uh_peak_rate(rain: np.ndarray, stage: np.ndarray, valid: np.ndarray,
                           tc_hr: pd.Series, nodes_df: pd.DataFrame) -> pd.DataFrame:
    N = len(nodes_df)
    is_tid = nodes_df["is_tidal"].to_numpy().astype(bool)
    is_res = nodes_df["is_reservoir"].to_numpy().astype(bool)
    area = node_area_km2(nodes_df)
    rows = {}
    for n in range(N):
        if is_tid[n]:
            rows[n] = (0.003, "fixed_tidal", 0); continue
        if is_res[n]:
            rows[n] = (0.005, "fixed_reservoir", 0); continue
        tcs = max(1, int(round(tc_hr.loc[n] / STEP_HR)))
        r, h, v = rain[:, n].astype(float), stage[:, n].astype(float), valid[:, n]
        starts = np.where(r > 1.5)[0]
        if len(starts) < MIN_EVENTS:
            rows[n] = (None, "no_rain_events", 0); continue
        brk = np.where(np.diff(starts) > 8)[0]
        e0 = [starts[0]] + [starts[b + 1] for b in brk]
        rates = []
        for t0 in e0:
            t1 = min(t0 + tcs, len(r)); t3 = min(t0 + 3 * tcs, len(h))
            if t3 <= t1 or t0 < 4:
                continue
            if not (v[t0 - 4:t0].all() and v[t1:t3].mean() > 0.9):
                continue
            tot = float(r[t0:t1].sum())
            if tot < 10.0:
                continue
            rise = float(np.nanmax(h[t1:t3]) - np.nanmean(h[t0 - 4:t0]))
            if rise > 0:
                rates.append(rise / tot)
        rows[n] = ((float(np.median(rates)), "calibrated", len(rates))
                   if len(rates) >= max(1, MIN_EVENTS // 2) else (None, "too_few_events", len(rates)))
    # Inherit from geographically nearest calibrated node
    if {"easting_itm", "northing_itm"} <= set(nodes_df.columns):
        xy = nodes_df[["easting_itm", "northing_itm"]].to_numpy(float)
    else:
        xy = nodes_df[["lon", "lat"]].to_numpy(float) * np.array([69e3, 111e3])
    cal = [k for k, v in rows.items() if v[0] is not None and v[1] == "calibrated"]
    for n, (val, src, ne) in list(rows.items()):
        if val is None:
            if cal:
                dn = min(cal, key=lambda k: np.hypot(*(xy[k] - xy[n])))
                scale = np.sqrt(max(area[dn], 0.01) / max(area[n], 0.01))
                rows[n] = (float(np.clip(rows[dn][0] * scale, 0.001, 0.05)),
                           f"inherited_from_{dn}", ne)
            else:
                rows[n] = (0.005, "default", ne)
    out = []
    for n in range(N):
        val, src, ne = rows[n]
        bound = None
        if not (is_tid[n] or is_res[n]):
            floor = 0.015 if area[n] < 25 else (0.010 if area[n] < 100 else 0.005)
            clipped = float(np.clip(val, floor, 0.030))
            bound = "floor" if clipped == floor and val < floor else (
                "cap" if clipped == 0.030 and val > 0.030 else None)
            val = clipped
        out.append({"node_idx": n, "peak_rate_m_per_mm": val, "rate_source": src,
                    "n_events": ne, "rate_on_bound": bound})
    df = pd.DataFrame(out).set_index("node_idx")
    nb = df.rate_on_bound.notna().sum()
    print(f"  peak_rate on an area-tier bound: {nb}/{N} nodes "
          f"(manuscript Section 3.8.2 disclosure)")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main(validate: bool = False) -> None:
    from utils.train_utils import get_split_boundary
    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    edges_df = river_edges(pd.read_csv(GRAPH_DIR / "edges.csv"))
    X = np.load(PROC_DIR / "X.npy", mmap_mode="r")
    T = X.shape[0]
    t1, _ = get_split_boundary(T, 0.70, 0.15)
    print(f"  Calibration on training rows [0, {t1}) of {T}")
    stage = np.asarray(X[:t1, :, F_STAGE_ANOM], dtype=float)
    rain = np.asarray(X[:t1, :, F_RAINFALL], dtype=float)
    ivp = PROC_DIR / "input_valid_mask.npy"
    valid = (np.load(ivp, mmap_mode="r")[:t1].astype(bool) if ivp.exists()
             else np.ones_like(stage, dtype=bool))

    kin = None
    kp = GRAPH_DIR / "kirpich_inputs.csv"
    if kp.exists():
        kin = pd.read_csv(kp)
        print(f"  Using DEM Kirpich inputs: {kp.name}")
    else:
        print("  [warn] kirpich_inputs.csv not found - using reach-slope fallback")
    tc = compute_tc_kirpich(nodes_df, edges_df, kin)
    print(f"  Tc range [{tc.tc_hr.min():.2f}, {tc.tc_hr.max():.2f}] h; "
          f"clipped {int(tc.tc_clipped.sum())}/{len(tc)}")

    kin_lags = compute_kinematic_lag(edges_df, nodes_df)
    emp = empirical_routing_lag(stage, valid, edges_df)
    lags, lag_meta = {}, {}
    for r in kin_lags.itertuples():
        key = f"{r.src}_{r.dst}"
        if (r.src, r.dst) in emp:
            lags[key] = int(emp[(r.src, r.dst)][0])
            lag_meta[key] = {"source": "empirical_xcorr",
                             "corr": round(emp[(r.src, r.dst)][1], 3),
                             "kinematic": int(r.lag_steps)}
        else:
            lags[key] = int(r.lag_steps)
            lag_meta[key] = {"source": r.source}
        if validate:
            print(f"    {key}: {lags[key]} steps [{lag_meta[key]['source']}]")

    rates = calibrate_uh_peak_rate(rain, stage, valid, tc.tc_hr, nodes_df)
    area = node_area_km2(nodes_df)
    params = {}
    for i, row in nodes_df.iterrows():
        tp, tb = scs_triangle_times(tc.loc[i, "tc_hr"])
        params[str(int(row.ref))] = {
            "node_idx": int(i), "ref": int(row.ref), "name": row["name"],
            "tc_hr": round(float(tc.loc[i, "tc_hr"]), 4),
            "tp_hr": round(tp, 4), "base_hr": round(tb, 4),
            "tc_min_raw": round(float(tc.loc[i, "tc_min_raw"]), 2),
            "tc_clipped": bool(tc.loc[i, "tc_clipped"]),
            "L_m": round(float(tc.loc[i, "L_m"]), 1),
            "slope": float(tc.loc[i, "S_used"]),
            "slope_source": tc.loc[i, "slope_source"],
            "peak_rate_m_per_mm": round(float(rates.loc[i, "peak_rate_m_per_mm"]), 6),
            "rate_source": rates.loc[i, "rate_source"],
            "rate_on_bound": rates.loc[i, "rate_on_bound"],
            "catchment_area_km2": round(float(area[i]), 3),
            "is_reservoir": bool(row.is_reservoir), "is_tidal": bool(row.is_tidal),
        }
    with open(OUT_DIR / "uh_params.json", "w") as f:
        json.dump({"description": "Per-node response-kernel parameters (revision audit "
                                  "version; see module docstring).",
                   "units": {"tc_hr": "hours (Kirpich minutes / 60)",
                             "tp_hr": "hours, D/2 + 0.6 Tc",
                             "base_hr": "hours, 2.67 Tp (standard SCS)"},
                   "calibration_period_rows": [0, int(t1)],
                   "step_min": STEP_MIN, "params": params}, f, indent=2)
    with open(OUT_DIR / "routing_lags.json", "w") as f:
        json.dump({"description": "Per-edge routing lags in 15-min steps; key "
                                  "'src_dst'. Uphill reverse duplicates excluded.",
                   "step_min": STEP_MIN, "lags": lags, "provenance": lag_meta}, f, indent=2)
    print("  Saved uh_params.json, routing_lags.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    main(ap.parse_args().validate)
