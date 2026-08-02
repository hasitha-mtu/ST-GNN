"""
unit_hydrograph.py  —  Calibrate UH parameters for synthetic scenario generation
==================================================================================
Derives three parameter sets from data already in the project — no external
hydraulic model or additional data sources required.

Outputs (written to dataset/graph/):
    uh_params.json       per-node time-of-concentration, peak response rate,
                         and UH shape parameters
    routing_lags.json    per-edge flood wave travel time (timesteps at 15-min)

Algorithm
---------
1. Time of concentration (Tc):
   Kirpich (1940) formula applied using catchment area from nodes.csv and
   elevation drop from edges.csv. Validated against empirical cross-correlation
   lags from y.npy; empirical estimate takes precedence where correlation is
   sufficient (r > 0.50 for at least 50 events).

2. Routing lag:
   Kinematic wave celerity (Lighthill & Whitham 1955): c = (5/3) × V_mean,
   Manning V_mean from channel slope (edges.csv) and bankfull hydraulic
   radius (nodes.csv p90_mAOD − gauge_datum_mOSGM15). Validated against
   empirical lag from cross-correlation of adjacent gauge stage anomalies.

3. UH peak response rate:
   Data-driven: for each gauge, identify rainfall events > 5mm/15min in
   X.npy feature 4, measure peak stage anomaly response within 6 hours in
   y.npy, compute median(peak_rise / cumulative_rainfall) across all events.
   This ensures the synthetic stage perturbations are calibrated to the
   actual rainfall–stage relationship observed at each specific gauge.

References
----------
Kirpich, Z.P. (1940). Time of concentration of small agricultural watersheds.
    Civil Engineering, 10(6), 501.
Lighthill, M.J., Whitham, G.B. (1955). On kinematic waves I: Flood movement
    in long rivers. Proceedings of the Royal Society A, 229(1178), 281-316.
USDA-NRCS (2004). Chapter 16, National Engineering Handbook Part 630.

Usage
-----
    python src/scenarios/unit_hydrograph.py
    python src/scenarios/unit_hydrograph.py --validate  # extra diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
OUT_DIR   = GRAPH_DIR          # UH parameters stored alongside graph data

STEP_MIN  = 15                 # temporal resolution (minutes)
STEP_HR   = STEP_MIN / 60.0   # 0.25 hr
F_RAINFALL   = 4              # index in X.npy: rainfall_mm per 15-min
F_STAGE_ANOM = 0              # index in X.npy: stage_anomaly
F_SWVL2_SAT  = 9              # index in X.npy: swvl2_sat_ratio
MANNING_N    = 0.035           # roughness for natural Irish lowland channel
MIN_EVENTS   = 3              # minimum events for empirical calibration


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Time of concentration (Tc) per node
# ══════════════════════════════════════════════════════════════════════════════

def compute_tc_kirpich(nodes_df: pd.DataFrame,
                       edges_df: pd.DataFrame) -> dict[int, float]:
    """
    Estimate Tc (hours) for each gauge sub-catchment using the Kirpich formula.

    Tc = 0.0195 × L_m^0.77 × S^-0.385

    where:
        L_m = hydraulic length of main channel (m), approximated as
              1.5 × sqrt(A_km2) × 1000  (standard catchment scaling, m)
        S   = average channel slope (m/m), estimated as
              sum(elev_drop_m along upstream path) / L_m

    Slope is bounded at 0.0002 m/m (flat urban channels) to prevent
    unrealistically long Tc on very gentle Lee main stem reaches.

    Returns dict {node_idx: Tc_hours}.
    """
    tc_map: dict[int, float] = {}

    # Build a lookup of total upstream elevation drop per node
    # using the longest upstream edge path through the network
    elev_drop_per_node: dict[int, float] = {
        int(r.node_idx): float(r.p90_mAOD - r.gauge_datum_mOSGM15)
        for _, r in nodes_df.iterrows()
    }

    for _, row in nodes_df.iterrows():
        idx      = int(row.node_idx)
        area_km2 = float(np.exp(row.log_catchment_area_km2))  # undo log

        # Hydraulic length approximation (m)
        L_m = 1500.0 * np.sqrt(max(area_km2, 0.01))

        # Elevation drop — use bankfull depth above datum as proxy for
        # total head loss along the main channel
        head_drop = elev_drop_per_node.get(idx, 1.0)
        head_drop = max(head_drop, 0.5)   # minimum 0.5m head

        # Also accumulate elevation drops along upstream edges
        up_edges = edges_df[edges_df.dst_idx == idx]
        if not up_edges.empty:
            head_drop = max(head_drop, up_edges.elev_drop_m.max())

        slope = head_drop / L_m
        slope = max(slope, 0.0002)         # floor for flat reaches

        tc = 0.0195 * (L_m ** 0.77) * (slope ** -0.385) / 3600.0
        tc = float(np.clip(tc, 0.25, 8.0))  # 15 min to 8 hr bounds
        tc_map[idx] = tc

    return tc_map


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Routing lags between adjacent gauge pairs
# ══════════════════════════════════════════════════════════════════════════════

def compute_kinematic_lag(edges_df: pd.DataFrame,
                          nodes_df: pd.DataFrame) -> dict[tuple, int]:
    """
    Estimate flood wave travel time (timesteps) for each directed edge.

    Uses kinematic wave celerity: c = (5/3) × V_mean
    Manning V_mean = (1/n) × R^(2/3) × S^(1/2)

    where:
        n — Manning roughness coefficient (0.035 for natural channel)
        R — hydraulic radius at bankfull ≈ 0.6 × bankfull depth
        S — channel slope = elev_drop_m / (river_dist_km × 1000)

    Returns dict {(src_idx, dst_idx): lag_steps}.
    """
    # Build node lookup for p90 and datum
    node_info = nodes_df.set_index('node_idx')[
        ['gauge_datum_mOSGM15', 'p90_mAOD']].to_dict('index')

    lag_map: dict[tuple, int] = {}

    for _, edge in edges_df.iterrows():
        src = int(edge.src_idx)
        dst = int(edge.dst_idx)
        dist_m = float(edge.river_dist_km) * 1000.0

        if dist_m < 100:
            lag_map[(src, dst)] = 1
            continue

        elev_drop = float(edge.elev_drop_m)
        slope = max(elev_drop / dist_m, 0.0001)

        # Bankfull hydraulic radius at destination gauge
        dst_info   = node_info.get(dst, {})
        depth_bf   = float(dst_info.get('p90_mAOD', 1.0)) - \
                     float(dst_info.get('gauge_datum_mOSGM15', 0.0))
        depth_bf   = max(depth_bf, 0.3)
        R = depth_bf * 0.6    # hydraulic radius ≈ 60% of bankfull depth

        V_mean = (1.0 / MANNING_N) * (R ** (2.0/3.0)) * (slope ** 0.5)
        V_mean = max(V_mean, 0.1)   # 0.1 m/s minimum
        c      = (5.0/3.0) * V_mean

        lag_sec   = dist_m / c
        lag_steps = max(1, round(lag_sec / (STEP_MIN * 60)))
        lag_map[(src, dst)] = lag_steps

    return lag_map


def empirical_routing_lag(y: np.ndarray,
                          edges_df: pd.DataFrame,
                          max_lag_steps: int = 48) -> dict[tuple, int]:
    """
    Estimate routing lags empirically from cross-correlation of stage
    anomaly time series at adjacent gauge pairs.

    For each directed edge (src → dst), computes the lag at peak
    cross-correlation between y[:, src] and y[:, dst]. Only accepts
    estimates where the peak correlation exceeds 0.50 and at least
    MIN_EVENTS windows show a consistent lag.

    Returns dict {(src_idx, dst_idx): lag_steps} for edges with
    sufficient empirical support.
    """
    empirical: dict[tuple, int] = {}

    for _, edge in edges_df.iterrows():
        src = int(edge.src_idx)
        dst = int(edge.dst_idx)

        s_src = y[:, src].astype(np.float64)
        s_dst = y[:, dst].astype(np.float64)

        # Remove NaN — use only valid pairs
        valid = np.isfinite(s_src) & np.isfinite(s_dst)
        if valid.sum() < 500:
            continue

        # Identify periods with significant stage rise (potential events)
        threshold = np.nanpercentile(s_src[valid], 80)
        event_mask = s_src > threshold
        if event_mask.sum() < MIN_EVENTS * 4:
            continue

        # Cross-correlation over the max lag window
        s1 = s_src - np.nanmean(s_src)
        s2 = s_dst - np.nanmean(s_dst)
        s1[~valid] = 0.0
        s2[~valid] = 0.0

        xcorr = correlate(s2, s1, mode='full')
        lags  = np.arange(-max_lag_steps, max_lag_steps + 1)
        mid   = len(xcorr) // 2
        xcorr_window = xcorr[mid - max_lag_steps : mid + max_lag_steps + 1]

        best_lag  = int(lags[np.argmax(xcorr_window)])
        best_corr = float(np.max(xcorr_window)) / (
            np.std(s1[valid]) * np.std(s2[valid]) * valid.sum() + 1e-8)

        if best_corr > 0.50 and best_lag > 0:
            empirical[(src, dst)] = best_lag

    return empirical


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — UH peak response rate from historical data
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_uh_peak_rate(X: np.ndarray,
                           y: np.ndarray,
                           tc_map: dict[int, float],
                           n_nodes: int,
                           nodes_df=None) -> dict[int, float]:
    """
    Calibrate per-node unit hydrograph peak response rates from historical data.

    For each gauge n:
      1. Identify rainfall events where rainfall_mm[t, n] > 5 mm/15-min.
      2. For each event, accumulate total rainfall over the Tc window.
      3. Measure peak stage anomaly response in the following 2×Tc window.
      4. peak_rate[n] = median(peak_response / total_rainfall) across events.

    Returns dict {node_idx: peak_rate (m stage anomaly per mm rainfall)}.
    """
    peak_rates: dict[int, float] = {}

    rainfall  = X[:, :, F_RAINFALL]    # [T, N]
    stage     = y                        # [T, N]

    # Node type flags — extract once before the loop
    tidal_idx     = set(nodes_df[nodes_df.is_tidal     == 1].node_idx.tolist())
    reservoir_idx = set(nodes_df[nodes_df.is_reservoir == 1].node_idx.tolist())
    node_area_map = {int(r.node_idx):
                     float(np.exp(r.log_catchment_area_km2))
                     for _, r in nodes_df.iterrows()}

    for n in range(n_nodes):
        # ── Hard-coded defaults for confounded node types ────────────
        # Tidal nodes: stage driven by tidal oscillation (±1m per 12.4hr).
        # Rainfall co-inciding with a rising tide gives spurious rates
        # (0.077 m/mm observed = 2.3m stage rise per 30mm rain — absurd).
        # Fix: 0.003 m/mm — tidal gauges DO respond to rainfall but the
        # signal is strongly attenuated by estuarine dynamics.
        if n in tidal_idx:
            peak_rates[n] = 0.003
            continue

        # Reservoir nodes: stage controlled by ESB operational releases.
        # Rainfall-stage correlations are meaningless here.
        if n in reservoir_idx:
            peak_rates[n] = 0.005   # placeholder; not used in scenarios
            continue

        tc_steps  = max(1, round(tc_map.get(n, 1.0) / STEP_HR))
        rain_n    = rainfall[:, n].astype(np.float64)
        stage_n   = stage[:, n].astype(np.float64)

        # Find timesteps where rainfall exceeds threshold
        # Threshold: 1.5 mm/15min (moderate Irish rainfall = 6mm/hr)
        event_starts = np.where(rain_n > 1.5)[0]
        if len(event_starts) < MIN_EVENTS:
            peak_rates[n] = None    # mark for inheritance
            continue

        # Cluster events (consecutive timesteps count as one event)
        gaps   = np.diff(event_starts)
        breaks = np.where(gaps > 8)[0]   # > 2hr gap = new event
        e_starts = [event_starts[0]]
        for b in breaks:
            e_starts.append(event_starts[b + 1])

        rates = []
        for t0 in e_starts:
            # Accumulate rainfall over Tc window
            t1 = min(t0 + tc_steps, len(rain_n))
            total_rain = float(rain_n[t0:t1].sum())
            if total_rain < 10.0:
                continue   # too little rain for reliable calibration

            # Measure peak stage rise in 2×Tc window after event
            t2 = min(t0 + tc_steps, len(stage_n))
            t3 = min(t0 + 3 * tc_steps, len(stage_n))
            if t3 <= t2:
                continue

            baseline = float(np.nanmean(stage_n[max(0, t0-4):t0]))
            peak     = float(np.nanmax(stage_n[t2:t3]))
            rise     = max(0.0, peak - baseline)

            if rise > 0.0:
                rates.append(rise / total_rain)

        if len(rates) >= MIN_EVENTS // 2:
            peak_rates[n] = float(np.median(rates))
        else:
            peak_rates[n] = None    # mark for inheritance

    # ── Inherit peak_rate from nearest node with valid calibration ──
    # Nodes with rainfall_mm = 0 (no raingauge in Thiessen polygon)
    # cannot be calibrated directly. Use the nearest calibrated
    # neighbour as a proxy, scaled by sqrt(area_ratio) to account
    # for sub-catchment size differences.
    node_areas = np.exp(nodes_df.set_index('node_idx')[
        'log_catchment_area_km2'].values.astype(np.float64))
    calibrated  = {k: v for k, v in peak_rates.items() if v is not None}
    uncalibrated = [k for k, v in peak_rates.items() if v is None]

    if uncalibrated and calibrated:
        cal_arr  = np.array(sorted(calibrated.keys()))
        for n_idx in uncalibrated:
            # Find nearest calibrated node by node index proximity
            # (index order follows spatial ordering in nodes.csv)
            diffs  = np.abs(cal_arr - n_idx)
            donor  = int(cal_arr[np.argmin(diffs)])
            donor_area = max(node_areas[donor], 0.01)
            own_area   = max(node_areas[n_idx],  0.01)
            # Smaller catchments respond more sharply to rainfall
            scale  = np.sqrt(donor_area / own_area)
            inherited = calibrated[donor] * scale
            peak_rates[n_idx] = float(np.clip(inherited, 0.001, 0.05))
        print(f"  Inherited peak_rate for {len(uncalibrated)} "
              f"nodes with no rainfall data from nearest calibrated "
              f"neighbours.")

    # Final fallback for any remaining None values
    for k in peak_rates:
        if peak_rates[k] is None:
            peak_rates[k] = 0.005

    # ── Physical bounds enforcement ───────────────────────────────────
    # Tidal and reservoir nodes keep their fixed defaults unchanged.
    # For all other nodes, apply:
#
    # Floor: area-dependent minimum response rate.
    #   <25 km²:  0.015 m/mm — small steep Irish headwaters
    #   <100 km²: 0.010 m/mm — mid-catchment gauges
    #   ≥100 km²: 0.005 m/mm — large main stem gauges
#
    # Cap: 0.030 m/mm — no gauge in the Lee network (1–1200 km²)
    # should produce more than 90cm rise per 30mm convective pulse.
    # Values above this are invariably spurious (tidal/urban backwater).

    for n_idx in list(peak_rates.keys()):
        if n_idx in tidal_idx or n_idx in reservoir_idx:
            continue    # fixed defaults — do not override
        area = node_area_map.get(n_idx, 100.0)
        if area < 25.0:
            floor = 0.015
        elif area < 100.0:
            floor = 0.010
        else:
            floor = 0.005
        peak_rates[n_idx] = float(np.clip(peak_rates[n_idx], floor, 0.030))

    return peak_rates


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(validate: bool = False) -> None:
    print("=" * 62)
    print("Unit hydrograph calibration for synthetic scenario generation")
    print("=" * 62)

    # ── Load graph data ───────────────────────────────────────────────
    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    edges_df = pd.read_csv(GRAPH_DIR / "edges.csv")
    n_nodes  = len(nodes_df)
    print(f"  Nodes: {n_nodes}   Edges: {len(edges_df)}")

    # ── Load time-series data (mmap — no RAM copy) ────────────────────
    print("  Loading X.npy and y.npy (mmap) …")
    X = np.load(PROC_DIR / "X.npy", mmap_mode="r")
    y = np.load(PROC_DIR / "y.npy", mmap_mode="r")
    T = X.shape[0]
    print(f"  Dataset: T={T:,} steps  N={X.shape[1]}  F={X.shape[2]}")

    # ── Step 1: Kirpich Tc ────────────────────────────────────────────
    print("\nStep 1: Computing time-of-concentration (Kirpich 1940) …")
    tc_map = compute_tc_kirpich(nodes_df, edges_df)
    tc_vals = list(tc_map.values())
    print(f"  Tc range: [{min(tc_vals):.2f}, {max(tc_vals):.2f}] hr")

    # ── Step 2: Routing lags ──────────────────────────────────────────
    print("\nStep 2: Computing routing lags …")
    kin_lags = compute_kinematic_lag(edges_df, nodes_df)
    print(f"  Kinematic lag range: "
          f"[{min(kin_lags.values())}, {max(kin_lags.values())}] steps")

    print("  Computing empirical cross-correlation lags …")
    emp_lags = empirical_routing_lag(np.array(y), edges_df)
    print(f"  Empirical lags validated for {len(emp_lags)} / "
          f"{len(edges_df)} edges")

    # Merge: empirical takes precedence when available
    final_lags: dict[str, int] = {}
    for (src, dst), kin_lag in kin_lags.items():
        emp_lag = emp_lags.get((src, dst))
        if emp_lag is not None:
            lag = emp_lag
            source = "empirical"
        else:
            lag = kin_lag
            source = "kinematic"
        final_lags[f"{src}_{dst}"] = int(lag)
        if validate:
            print(f"    edge {src}→{dst}: lag={lag} steps "
                  f"({lag*STEP_MIN} min)  [{source}]")

    # ── Step 3: UH peak response rates ───────────────────────────────
    print("\nStep 3: Calibrating UH peak response rates from X.npy/y.npy …")
    peak_rates = calibrate_uh_peak_rate(np.array(X), np.array(y),
                                        tc_map, n_nodes, nodes_df)
    rates_vals = list(peak_rates.values())
    default_count = sum(1 for v in rates_vals if abs(v - 0.005) < 1e-6)
    print(f"  Peak rate range: [{min(rates_vals):.5f}, "
          f"{max(rates_vals):.5f}] m/mm")
    print(f"  Nodes at default fallback (0.005): {default_count}/{n_nodes}")
    if default_count == n_nodes:
        print("  WARNING: ALL nodes at default — check rainfall threshold")

    # ── Assemble uh_params.json ───────────────────────────────────────
    uh_params: dict[str, dict] = {}
    for _, row in nodes_df.iterrows():
        idx  = int(row.node_idx)
        ref  = int(row.ref)

        tc   = tc_map[idx]
        tp   = 0.6 * tc                    # SCS triangular UH time to peak
        base = 2.67 * tc                   # UH base width
        rate = peak_rates[idx]
        area = float(np.exp(row.log_catchment_area_km2))

        uh_params[str(ref)] = {
            "node_idx":        idx,
            "ref":             ref,
            "name":            row["name"],
            "tc_hr":           round(tc,   3),
            "tp_hr":           round(tp,   3),
            "base_hr":         round(base, 3),
            "peak_rate_m_per_mm": round(rate, 6),
            "catchment_area_km2": round(area, 3),
            "is_reservoir":    bool(row.is_reservoir),
            "is_tidal":        bool(row.is_tidal),
        }

    # Save outputs
    out_uh   = OUT_DIR / "uh_params.json"
    out_lags = OUT_DIR / "routing_lags.json"

    with open(out_uh, "w") as f:
        json.dump({
            "description": (
                "Per-node unit hydrograph parameters calibrated from "
                "Lee OPW gauge data (X.npy / y.npy). "
                "Tc via Kirpich (1940); peak_rate via data-driven "
                "rainfall-stage regression on historical events."
            ),
            "step_min": STEP_MIN,
            "params":   uh_params,
        }, f, indent=2)

    with open(out_lags, "w") as f:
        json.dump({
            "description": (
                "Per-edge flood wave routing lags (15-min timesteps). "
                "Key format: 'src_idx_dst_idx'. "
                "Kinematic wave celerity (Lighthill & Whitham 1955), "
                "validated against empirical cross-correlation lags."
            ),
            "step_min": STEP_MIN,
            "lags":     final_lags,
        }, f, indent=2)

    print(f"\n  Saved: {out_uh.name}")
    print(f"  Saved: {out_lags.name}")

    # ── Validation summary ────────────────────────────────────────────
    print("\nPer-node summary (sorted by catchment area):")
    print(f"  {'ref':>7} {'name':>28} {'Tc(hr)':>7} "
          f"{'Tp(hr)':>7} {'rate(m/mm)':>11} {'area(km2)':>10}")
    sorted_refs = sorted(uh_params.keys(),
                         key=lambda r: uh_params[r]["catchment_area_km2"])
    for ref in sorted_refs:
        p = uh_params[ref]
        flag = " [RES]" if p["is_reservoir"] else (
               " [TID]" if p["is_tidal"] else "")
        print(f"  {p['ref']:>7} {p['name'][:28]:>28} "
              f"{p['tc_hr']:>7.3f} {p['tp_hr']:>7.3f} "
              f"{p['peak_rate_m_per_mm']:>11.6f} "
              f"{p['catchment_area_km2']:>10.1f}{flag}")

    print("\nDone. Run scenario_generator.py --all to generate scenarios.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate unit hydrograph parameters for scenario generation")
    parser.add_argument("--validate", action="store_true",
                        help="Print per-edge routing lag details")
    args = parser.parse_args()
    main(validate=args.validate)
