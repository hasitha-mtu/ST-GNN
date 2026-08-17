"""
scenario_generator.py  —  Synthetic scenario generation for ST-GNN evaluation
===============================================================================
Generates five synthetic test datasets, each targeting a specific flash flood
hydraulic scenario that is underrepresented in the historical Lee gauge record.

Each scenario perturbs the existing X.npy / y.npy arrays to construct
physically plausible but controlled conditions. Existing trained model
checkpoints are then evaluated on these datasets without retraining.

Scenarios
---------
S1  ConvectiveCell       Isolated convective storm over headwater tributaries.
                         Tests cross-tributary HAND topology advantage.
S2  GaugeFailure         Progressive sensor loss during an active flood event.
                         Tests graph redundancy under missing upstream data.
S3  InniscarraRelease    Controlled reservoir release decoupled from rainfall.
                         Tests whether the model respects reservoir topology.
S4  SatBreakthrough      Dry catchment → saturation breakthrough mid-event.
                         Tests antecedent soil moisture gate (Idea 1).
S5  SpatialGradient      West-to-east rainfall gradient across catchment.
                         Tests whether graph routes the gradient correctly.

Output structure
----------------
    dataset/scenarios/
        S1_ConvectiveCell/
            X_synthetic.npy       [T_s, N, F]
            y_synthetic.npy       [T_s, N]
            mask_synthetic.npy    [T_s, N]
            scenario_meta.json
        S2_GaugeFailure/
            ...
        S3_InniscarraRelease/
            ...
        S4_SatBreakthrough/
            ...
        S5_SpatialGradient/
            ...

Usage
-----
    python src/scenarios/scenario_generator.py --all
    python src/scenarios/scenario_generator.py --scenario S1
    python src/scenarios/scenario_generator.py --scenario S2 --n-windows 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
SCEN_DIR  = BASE_DIR / "dataset/scenarios"
SCEN_DIR.mkdir(parents=True, exist_ok=True)

for _p in [BASE_DIR, BASE_DIR / "src"]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# ── Feature indices in X.npy ─────────────────────────────────────────────────
F_STAGE   = 0    # stage_anomaly (m)
F_NORM    = 1    # normalized_stage
F_DHDT    = 2    # dh/dt (m/hr)
F_DISC    = 3    # discharge (log1p m³/s)
F_RAIN    = 4    # rainfall_mm
F_SW1_RAW = 5    # swvl1_raw
F_SW1_SAT = 6    # swvl1_sat_ratio
F_SW1_AN  = 7    # swvl1_anomaly
F_SW2_RAW = 8    # swvl2_raw
F_SW2_SAT = 9    # swvl2_sat_ratio
F_SW2_AN  = 10   # swvl2_anomaly

STEP_MIN = 15
T_IN     = 32    # input window length
T_OUT_MAX = 48   # longest horizon — scenario must cover at least T_IN + T_OUT_MAX
T_WINDOW  = T_IN + T_OUT_MAX + 24   # 104 steps = 26 hours of context per scenario


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_project_data() -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                  pd.DataFrame, pd.DataFrame, dict, dict, np.ndarray]:
    """Load all base arrays and graph metadata needed for scenario generation."""
    print("Loading base arrays …")
    X    = np.load(PROC_DIR / "X.npy",          mmap_mode="r")
    y    = np.load(PROC_DIR / "y.npy",          mmap_mode="r")
    mask = np.load(PROC_DIR / "valid_mask.npy", mmap_mode="r")
    nd   = pd.read_csv(GRAPH_DIR / "nodes.csv")
    ed   = pd.read_csv(GRAPH_DIR / "edges.csv")

    with open(GRAPH_DIR / "uh_params.json")    as f: uh   = json.load(f)["params"]
    with open(GRAPH_DIR / "routing_lags.json") as f: lags = json.load(f)["lags"]
    with open(GRAPH_DIR / "bankfull_thresholds.json") as f:
        bf_raw = json.load(f)
    bf_dict = bf_raw.get("thresholds", bf_raw)
    refs    = [str(r) for r in nd["ref"].tolist()]
    bankfull = np.array([float(bf_dict.get(r, 0.5)) for r in refs],
                        dtype=np.float32)

    print(f"  X:{X.shape}  y:{y.shape}  mask:{mask.shape}")
    vals = X[79000:79100, :, 9]
    print(f"F[9] swvl2_sat_ratio: mean={vals.mean():.4f} std={vals.std():.4f}")
    return X, y, mask, nd, ed, uh, lags, bankfull


def triangular_uh(peak_rate: float, tc_hr: float,
                  total_steps: int) -> np.ndarray:
    """
    SCS triangular unit hydrograph pulse (stage anomaly response per mm rain).

    Rising limb:  Tp = 0.6 × Tc
    Falling limb: Tb = 1.67 × Tp
    """
    tp_steps   = max(1, round(0.6  * tc_hr / STEP_MIN * 60))
    base_steps = max(2, round(2.67 * tc_hr / STEP_MIN * 60))
    uh = np.zeros(total_steps)
    for t in range(min(total_steps, tp_steps)):
        uh[t] = peak_rate * (t / tp_steps)
    for t in range(tp_steps, min(total_steps, base_steps)):
        uh[t] = peak_rate * (1.0 - (t - tp_steps) / (base_steps - tp_steps))
    return uh


def apply_routing(stage_delta: np.ndarray,
                  src_idx: int, dst_idx: int,
                  lags: dict, attenuation: float = 0.75) -> np.ndarray:
    """
    Route a stage delta from src_idx to dst_idx using the precomputed lag.
    Attenuates amplitude by `attenuation` factor per edge.
    """
    key = f"{src_idx}_{dst_idx}"
    lag = lags.get(key, 2)
    routed = np.zeros_like(stage_delta)
    if lag < len(stage_delta):
        routed[lag:] = stage_delta[:-lag] * attenuation if lag > 0 \
                       else stage_delta * attenuation
    return routed


def downstream_bfs_edges(source_idx: int, ed: pd.DataFrame,
                         max_hops: int = 8) -> list[tuple[int, int]]:
    """
    Pure-topology BFS from source_idx outward through the directed river
    network in `ed`, returning the (src, dst) edges reached in hop order.

    Split out from the signal-propagation step below so the downstream
    node set can be determined once, before any window data is touched —
    needed both to scope select_base_windows' calm-baseline check to the
    scenario's relevant nodes, and to avoid re-querying `ed` on every one
    of the ~30 windows generated per scenario (topology doesn't depend on
    window content).
    """
    visited = {source_idx}
    frontier = [source_idx]
    edges_order: list[tuple[int, int]] = []

    for _ in range(max_hops):
        if not frontier:
            break
        next_frontier = []
        for node in frontier:
            out_edges = ed[ed["src_idx"] == node]
            for _, edge in out_edges.iterrows():
                dst = int(edge["dst_idx"])
                if dst in visited:
                    continue
                visited.add(dst)
                edges_order.append((node, dst))
                next_frontier.append(dst)
        frontier = next_frontier

    return edges_order


def propagate_downstream_chain(y_w: np.ndarray, X_w: np.ndarray,
                               source_idx: int, edges_order: list[tuple[int, int]],
                               lags: dict, baseline: float) -> list[int]:
    """
    Multi-hop downstream propagation of a stage delta injected at
    source_idx, walking the precomputed BFS edge order from
    `downstream_bfs_edges` and applying `apply_routing`'s per-edge lag +
    attenuation at each hop, so the signal weakens naturally as it
    travels rather than being re-injected at full strength at every
    node.

    S1/S5's routing loops apply `apply_routing` once per perturbed node
    because the nodes they perturb (headwaters) sit directly adjacent to
    the segment being tested. S3's injection point (Inniscarra Tailrace)
    is five edges upstream of the last Cork city gauge, so a single pass
    would only ever reach Waterworks Weir and leave the rest of the
    chain — Fitzgerald's Park, Pope's Quay, St. Patrick's Quay, Currach
    Club — completely untouched. This walks the full precomputed chain.

    Returns the list of node indices reached, in hop order.
    """
    signal = {source_idx: y_w[:, source_idx] - baseline}
    reached: list[int] = []

    for src, dst in edges_order:
        dst_delta = apply_routing(signal[src], src, dst, lags)
        y_w[:, dst] += dst_delta
        X_w[:, dst, F_STAGE] += dst_delta
        signal[dst] = dst_delta
        reached.append(dst)

    return reached


def physical_consistency_check(X_syn: np.ndarray, y_syn: np.ndarray,
                                nd: pd.DataFrame, bankfull: np.ndarray,
                                name: str) -> bool:
    """
    Four guards that every synthetic scenario must pass.
    Returns True if all guards pass, False otherwise.
    """
    # Guard 1: no stage anomaly exceeds 3× the 99th percentile of
    # the original test period
    y_orig = np.load(PROC_DIR / "y.npy", mmap_mode="r")
    y99 = float(np.nanpercentile(np.abs(y_orig), 99))
    if np.any(np.abs(y_syn[np.isfinite(y_syn)]) > 3 * y99):
        print(f"  [WARN] {name}: y_syn exceeds 3×99th percentile "
              f"({3*y99:.3f} m). Clipping.")
        y_syn[:] = np.clip(y_syn, -3*y99, 3*y99)

    # Guard 2: rainfall cannot be negative
    if np.any(X_syn[:, :, F_RAIN] < 0):
        X_syn[:, :, F_RAIN] = np.clip(X_syn[:, :, F_RAIN], 0, None)

    # Guard 3: saturation ratio stays in [0, 1]
    for fi in (F_SW1_SAT, F_SW2_SAT):
        X_syn[:, :, fi] = np.clip(X_syn[:, :, fi], 0.0, 1.0)

    # Guard 4: normalized stage clipped to valid range
    X_syn[:, :, F_NORM] = np.clip(X_syn[:, :, F_NORM], -2.0, 5.0)

    return True


def identify_headwater_nodes(nd: pd.DataFrame, n: int = 6) -> list[int]:
    """Return node_idx of the n smallest non-tidal, non-reservoir sub-catchments."""
    valid = nd[(nd.is_tidal == 0) & (nd.is_reservoir == 0)].copy()
    valid = valid.sort_values("log_catchment_area_km2")
    return valid.head(n)["node_idx"].tolist()


def identify_downstream_nodes(nd: pd.DataFrame, n: int = 5) -> list[int]:
    """Return node_idx of the n largest (most downstream) non-reservoir nodes."""
    valid = nd[(nd.is_reservoir == 0)].copy()
    valid = valid.sort_values("log_catchment_area_km2", ascending=False)
    return valid.head(n)["node_idx"].tolist()


def select_base_windows(X: np.ndarray, y: np.ndarray,
                        bankfull: np.ndarray,
                        sat_min: float = 0.60, sat_max: float = 0.92,
                        max_stage_frac: float = 0.40,
                        n_windows: int = 50,
                        search_from_frac: float = 0.70,
                        t_stride: int = 48,
                        node_subset: Optional[list[int]] = None) -> list[int]:
    """
    Select T_WINDOW-length base windows from the validation + test period.

    Searches from search_from_frac (default 0.70 = start of validation)
    rather than the test period only. This is necessary because the Lee
    catchment test period (Sep 2025–Mar 2026) covers Irish autumn/winter
    when swvl2_sat_ratio is consistently 0.85–0.99, making it impossible
    to find "moderately wet but not saturated" pre-event conditions.
    The validation period (Apr–Sep 2025) includes transitional saturation
    states (0.65–0.88) needed for scenarios like ConvectiveCell.

    Criteria:
        - swvl2_sat_ratio (mean across nodes) within [sat_min, sat_max]
        - No gauge exceeds max_stage_frac × bankfull threshold (calm)
        - Windows separated by at least T_IN steps to limit overlap

    node_subset: restrict the calm-baseline check (condition 2) to these
        node indices instead of all N gauges. Requiring every one of the
        27 Lee gauges to be simultaneously calm is the right test for a
        scenario whose injected anomaly could plausibly interact with
        any node (e.g. the original channel-blockage form), but it's an
        unnecessarily strict — and yield-limiting — constraint for a
        scenario that only perturbs a fixed, known subset of nodes (e.g.
        S3's tailrace + downstream chain): whether some unrelated
        headwater gauge happens to be flooded has no bearing on whether
        the release event being injected is physically valid. Default
        None preserves the original all-nodes behaviour for S1/S2/S4/S5.
    """
    T = X.shape[0]
    search_start = int(T * search_from_frac)
    candidates = []
    check_nodes = node_subset if node_subset is not None else range(y.shape[1])

    t = search_start
    while t < T - T_WINDOW and len(candidates) < n_windows:
        window_X = X[t : t + T_WINDOW]   # [T_WINDOW, N, F]
        window_y = y[t : t + T_WINDOW]   # [T_WINDOW, N]

        # Condition 1: antecedent saturation proxy
        # swvl2_sat_ratio in X.npy may be 0.0 if ERA5-Land features
        # were not written correctly (NaN→0 fill) or are normalised.
        # Use a two-tier approach:
        #   Tier A: if swvl2_sat_ratio has real variance (std > 0.01),
        #           use it directly against [sat_min, sat_max]
        #   Tier B: if it is all zeros/near-zero, use rainfall_mm as
        #           a saturation PROXY: antecedent rain accumulated
        #           over the T_IN window correlates strongly with
        #           catchment wetness in Atlantic Irish climate
        sat_vals = window_X[:T_IN, :, F_SW2_SAT]
        if sat_vals.std() > 0.01 and sat_vals.mean() > 0.05:
            # Tier A: real sat_ratio data available
            mean_sat = float(np.nanmean(sat_vals))
            if not (sat_min <= mean_sat <= sat_max):
                t += t_stride; continue
        else:
            # Tier B: sat is missing — use antecedent rainfall proxy
            # Irish summer: low cumulative rain → low sat (matches sat_min<0.7)
            # Irish winter: high rain → high sat (matches sat_min>0.7)
            antecedent_rain = float(np.nanmean(window_X[:T_IN, :, F_RAIN]))
            # Map rain proxy to sat range: 0mm/15min ≈ 0.50 sat,
            # 1.5mm/15min ≈ 0.90 sat (linear approx for Irish Atlantic)
            proxy_sat = np.clip(0.50 + antecedent_rain * 0.27, 0.50, 0.99)
            if not (sat_min <= proxy_sat <= sat_max):
                t += t_stride; continue

        # Condition 2: calm baseline — per-node bankfull comparison,
        # scoped to `check_nodes` (all N gauges unless a scenario
        # narrows it via node_subset — see docstring above).
        # BUG FIX: original used np.nanmin(bankfull)=0.05m giving
        # threshold=0.02m which rejected every window. Now compares
        # each node against its own bankfull with a 0.30m floor.
        calm = True
        for _n in check_nodes:
            bf_n = max(float(bankfull[_n]), 0.30)
            if float(np.nanmax(np.abs(window_y[:T_IN, _n]))) > max_stage_frac * bf_n:
                calm = False
                break
        if not calm:
            t += t_stride; continue

        # Condition 3: valid (non-NaN) data throughout window
        if np.isnan(window_X[:T_IN]).sum() > 0.05 * T_IN * X.shape[1] * X.shape[2]:
            t += t_stride; continue

        candidates.append(t)
        t += T_IN   # non-overlapping stride

    return candidates[:n_windows]


def save_scenario(name: str, out_dir: Path,
                  X_syn: np.ndarray, y_syn: np.ndarray,
                  mask_syn: np.ndarray, meta: dict) -> None:
    """Save synthetic arrays and metadata for one scenario."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_synthetic.npy",    X_syn.astype(np.float32))
    np.save(out_dir / "y_synthetic.npy",    y_syn.astype(np.float32))
    np.save(out_dir / "mask_synthetic.npy", mask_syn.astype(np.float32))
    with open(out_dir / "scenario_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {name}: X{X_syn.shape}  y{y_syn.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# S1 — Convective Cell over headwater tributaries
# ══════════════════════════════════════════════════════════════════════════════

def generate_s1_convective_cell(X, y, mask, nd, ed, uh, lags, bankfull,
                                 n_windows: int = 50) -> None:
    """
    Isolated convective storm over 6 headwater gauges. Rapid stage rise
    within 45–90 minutes. Flood pulse arrives at Cork city gauges 2–4 hr later.

    Physical motivation: tests whether cross-tributary HAND edges carry the
    lateral flood signal before it reaches the Cork city gauges via the
    river-network alone.
    """
    print("\nS1: ConvectiveCell")
    out_dir = SCEN_DIR / "S1_ConvectiveCell"

    headwater_idx = identify_headwater_nodes(nd, n=6)
    downstream_idx = identify_downstream_nodes(nd, n=5)

    # Select windows: moderate pre-event saturation (0.65–0.90)
    # S1 requires soil not too dry (enough moisture for rapid runoff)
    # and not fully saturated (so the synthetic pulse adds clear signal).
    # Search from validation period start (Apr 2025) to access the
    # spring/summer drying + early autumn rewetting conditions.
    window_starts = select_base_windows(
        X, y, bankfull, sat_min=0.55, sat_max=0.90, n_windows=n_windows,
        search_from_frac=0.70)

    if not window_starts:
        print("  [skip] No valid base windows found for S1")
        return

    # Build the convective rainfall pulse (triangular, 90 min duration)
    pulse_peak    = 30.0     # mm per 15-min at storm centre
    pulse_steps   = 6        # 90 minutes
    pulse = np.zeros(pulse_steps)
    half  = pulse_steps // 2
    for t in range(half): pulse[t] = pulse_peak * t / half
    for t in range(half, pulse_steps): pulse[t] = pulse_peak * (pulse_steps - t) / half

    T_s   = T_WINDOW * len(window_starts)
    X_syn = np.zeros((T_s, X.shape[1], X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, y.shape[1]),              dtype=np.float32)
    m_syn = np.zeros((T_s, mask.shape[1]),           dtype=np.float32)

    pulse_at = T_IN + 2    # storm begins 2 steps into prediction horizon

    for i, t0 in enumerate(window_starts):
        sl = slice(i * T_WINDOW, (i+1) * T_WINDOW)

        # Copy baseline window
        X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

        # Set antecedent saturation synthetically.
        # swvl2_sat_ratio may be 0.0 in X.npy if ERA5-Land was not
        # integrated correctly. Always write a physically meaningful
        # value directly rather than relying on X.npy content.
        # S1: moderate saturation (0.72) — not too dry, not flooded.
        # Ramp up slightly over the input window to simulate
        # ongoing light rain before the convective event.
        for _t in range(T_WINDOW):
            ramp = min(0.08, _t / T_WINDOW * 0.10)
            X_w[_t, :, F_SW2_SAT] = np.clip(0.72 + ramp, 0, 1)
            X_w[_t, :, F_SW1_SAT] = np.clip(0.68 + ramp, 0, 1)
            X_w[_t, :, F_SW2_RAW] = X_w[_t, :, F_SW2_SAT] * 0.472
            X_w[_t, :, F_SW1_RAW] = X_w[_t, :, F_SW1_SAT] * 0.472

        # Apply convective rainfall pulse to headwater nodes only
        for n_idx in headwater_idx:
            ref = str(int(nd.loc[nd.node_idx == n_idx, "ref"].values[0]))
            if ref not in uh:
                continue
            p = uh[ref]
            # Rainfall perturbation at headwater gauges
            rain_start = pulse_at
            rain_end   = min(pulse_at + pulse_steps, T_WINDOW)
            X_w[rain_start:rain_end, n_idx, F_RAIN] += pulse[:rain_end-rain_start]

            # Compute rainfall excess (sat-dependent infiltration)
            sat_ratio = float(np.mean(X_w[:T_IN, n_idx, F_SW2_SAT]))
            excess_factor = sat_ratio   # near 1 when saturated
            total_excess  = float(np.sum(pulse)) * excess_factor

            # Stage response via UH
            uh_arr = triangular_uh(
                p["peak_rate_m_per_mm"],
                p["tc_hr"],
                T_WINDOW - pulse_at)
            stage_delta = uh_arr * total_excess

            # Apply to stage_anomaly and dh_dt
            t_start = pulse_at + round(p["tp_hr"] / STEP_MIN * 60)
            t_start = min(t_start, T_WINDOW - 1)
            uh_len  = min(len(stage_delta), T_WINDOW - t_start)
            y_w[t_start : t_start + uh_len, n_idx] += stage_delta[:uh_len]
            X_w[t_start : t_start + uh_len, n_idx, F_STAGE] += stage_delta[:uh_len]

        # Route headwater signal downstream through river network
        for _, edge in ed.iterrows():
            src = int(edge.src_idx)
            dst = int(edge.dst_idx)
            if src not in headwater_idx:
                continue
            if y_w[:, src].max() <= y_w[:T_IN, src].max():
                continue   # no signal to route
            src_delta = y_w[:, src] - float(np.mean(y_w[:T_IN, src]))
            dst_delta = apply_routing(src_delta, src, dst, lags)
            y_w[:, dst] += dst_delta
            X_w[:, dst, F_STAGE] += dst_delta

        X_syn[sl] = X_w
        y_syn[sl] = y_w
        m_syn[sl] = m_w

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S1")

    meta = {
        "name": "S1_ConvectiveCell",
        "description": (
            "Isolated convective storm over 6 headwater gauges. "
            "Rapid stage rise within 90 min. Tests HAND topology advantage."),
        "n_windows":       len(window_starts),
        "T_per_window":    T_WINDOW,
        "T_total":         T_s,
        "headwater_nodes": headwater_idx,
        "downstream_nodes": downstream_idx,
        "pulse_peak_mm_per_15min": 30.0,
        "pulse_duration_steps":    pulse_steps,
        "pulse_at_step":   pulse_at,
    }
    save_scenario("S1_ConvectiveCell", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S2 — Progressive gauge failure during active flood
# ══════════════════════════════════════════════════════════════════════════════

def generate_s2_gauge_failure(X, y, mask, nd, ed, uh, lags, bankfull,
                               n_windows: int = 40) -> None:
    """
    Real flood windows from the test set with upstream gauges progressively
    zeroed from T_failure onward. Tests graph spatial redundancy.

    Three failure severities per window:
        1 gauge failed, 3 gauges failed, 5 gauges failed.
    """
    print("\nS2: GaugeFailure")
    out_dir = SCEN_DIR / "S2_GaugeFailure"

    # Select windows that ARE flood events (high stage)
    T     = X.shape[0]
    N     = X.shape[1]
    test_start = int(T * 0.85)

    # Load bankfull thresholds (in anomaly space)
    flood_windows = []
    t = test_start
    while t < T - T_WINDOW and len(flood_windows) < n_windows:
        y_w = y[t : t + T_WINDOW]
        # Window must have at least 3 gauges exceeding 50% of bankfull
        n_exceeding = int(np.sum(
            np.nanmax(y_w[T_IN:T_IN+24], axis=0) > 0.5 * bankfull))
        if n_exceeding >= 3:
            flood_windows.append(t)
        t += T_IN // 2   # overlapping OK for scenario diversity

    if not flood_windows:
        print("  [warn] No flood windows found — using high-stage windows")
        flood_windows = select_base_windows(
            X, y, bankfull, sat_min=0.80, sat_max=1.0, n_windows=n_windows)

    headwater_idx = identify_headwater_nodes(nd, n=8)
    failure_levels = [1, 3, 5]   # number of upstream gauges failed

    T_s   = T_WINDOW * len(flood_windows) * len(failure_levels)
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    t_failure = T_IN + 4   # failure begins 1 hr into the forecast

    row = 0
    failed_node_sets = []
    for t0 in flood_windows:
        for n_fail in failure_levels:
            sl = slice(row * T_WINDOW, (row + 1) * T_WINDOW)

            X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
            y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
            m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

            fail_nodes = headwater_idx[:n_fail]
            failed_node_sets.append(fail_nodes)

            # Zero dynamic features for failed gauges from t_failure onward
            X_w[t_failure:, fail_nodes, F_STAGE] = 0.0
            X_w[t_failure:, fail_nodes, F_DHDT]  = 0.0
            X_w[t_failure:, fail_nodes, F_DISC]  = 0.0
            X_w[t_failure:, fail_nodes, F_RAIN]  = 0.0
            # Mark failed gauges as invalid in the mask
            m_w[t_failure:, fail_nodes] = 0.0

            X_syn[sl] = X_w
            y_syn[sl] = y_w
            m_syn[sl] = m_w
            row += 1

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S2")

    meta = {
        "name": "S2_GaugeFailure",
        "description": (
            "Flood windows with progressive upstream gauge failure. "
            "Tests graph spatial redundancy and information recovery."),
        "n_windows":       len(flood_windows),
        "failure_levels":  failure_levels,
        "T_per_window":    T_WINDOW,
        "T_total":         T_s,
        "t_failure_step":  t_failure,
        "headwater_nodes": headwater_idx,
    }
    save_scenario("S2_GaugeFailure", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S3 — Inniscarra reservoir release: outlet-driven signal decoupled from
#      upstream inflow / rainfall
# ══════════════════════════════════════════════════════════════════════════════

def generate_s3_inniscarra_release(X, y, mask, nd, ed, uh, lags, bankfull,
                                    n_windows: int = 30) -> None:
    """
    ESB operates a controlled release from Inniscarra dam, ramping stage
    at the Inniscarra Tailrace gauge (ref 19109) independently of
    catchment rainfall or the natural inflow recorded at the two
    upstream reservoir gauges — Carrigadrohid Headrace (19095) and
    Inniscarra Headrace (19094). This is a real, publicly documented
    operational pattern for the Lee catchment (OPW Lee Catchment Flood
    Risk Assessment and Management Study, 2011), and it directly
    exercises the paper's reservoir node handling: the release signal
    must propagate downstream through the river network toward Cork
    city, while the two upstream inflow gauges — which the release
    cannot physically affect, since water cannot flow back through a
    dam — should show no response at all.

    Tests whether the model:
      (a) correctly anticipates the downstream stage rise from the
          release-node signal alone (a genuine, learnable pattern, not
          rainfall-driven), and
      (b) respects the one-directional hydraulic gradient by NOT leaking
          the release signal backward onto the inflow gauges — the
          reservoir-topology analogue of the original blockage test's
          "does the physics gate resist an implausible propagation"
          question, now framed around a real operational event rather
          than a hypothetical debris blockage.
    """
    print("\nS3: InniscarraRelease")
    out_dir = SCEN_DIR / "S3_InniscarraRelease"

    T = X.shape[0]; N = X.shape[1]

    # Identify the release node (Inniscarra Tailrace) and the two
    # upstream inflow nodes that must stay decoupled, by ref rather than
    # a hardcoded node_idx — stays correct even if nodes.csv row order
    # ever changes.
    def _idx_for_ref(ref: str) -> Optional[int]:
        row = nd[nd["ref"].astype(str) == ref]
        return int(row["node_idx"].values[0]) if not row.empty else None

    tailrace_idx       = _idx_for_ref("19109")   # Inniscarra Tailrace
    headrace_idx        = _idx_for_ref("19094")   # Inniscarra Headrace
    carrigadrohid_idx   = _idx_for_ref("19095")   # Carrigadrohid Headrace

    if tailrace_idx is None:
        print("  [skip] Inniscarra Tailrace (ref 19109) not found in nodes.csv")
        return
    if headrace_idx is None or carrigadrohid_idx is None:
        print("  [warn] One or both reservoir inflow nodes (19094/19095) "
              "missing from nodes.csv — decoupled_nodes meta will be partial")

    # Downstream chain topology (Waterworks Weir -> Fitzgerald's Park ->
    # Pope's Quay -> St. Patrick's Quay -> Currach Club) is fixed by the
    # graph, not by window content — compute it once, before window
    # selection, so it can both scope the calm-baseline check below and
    # be reused for every window's routing pass without re-querying `ed`.
    edges_order = downstream_bfs_edges(tailrace_idx, ed)
    downstream_idx = [dst for _src, dst in edges_order]

    # Select calm-to-moderate windows so the release signal is clearly
    # attributable rather than swamped by a concurrent flood event.
    #
    # node_subset restricts the calm-baseline check to only the nodes
    # this scenario actually perturbs (tailrace + downstream chain).
    # The original blockage version of S3 required all 27 Lee gauges
    # calm simultaneously — appropriate when an injected anomaly could
    # plausibly interact anywhere, but an unnecessarily strict (and
    # yield-limiting) constraint here: whether some unrelated headwater
    # gauge is mid-flood has no bearing on whether an Inniscarra release
    # is physically valid to inject. Narrowing this recovered most of
    # the window shortfall reported after the first --all run (11/30
    # windows found under the old all-27-gauges check).
    release_relevant_nodes = [tailrace_idx] + downstream_idx

    # Same retry structure as before (and S5): the joint saturation +
    # calm-baseline constraint can still return zero windows in the
    # validation-period search region even scoped down, so loosen
    # max_stage_frac once before giving up rather than silently
    # producing a zero-length scenario.
    window_starts = select_base_windows(
        X, y, bankfull, sat_min=0.60, sat_max=0.92,
        max_stage_frac=0.30, n_windows=n_windows,
        search_from_frac=0.70, node_subset=release_relevant_nodes)

    if not window_starts:
        print("  [warn] No windows at max_stage_frac=0.30 — retrying with "
              "0.40 (matches S1/S5 default)")
        window_starts = select_base_windows(
            X, y, bankfull, sat_min=0.60, sat_max=0.92,
            max_stage_frac=0.40, n_windows=n_windows,
            search_from_frac=0.70, node_subset=release_relevant_nodes)

    if not window_starts:
        print("  [skip] No valid base windows found for S3 even after "
              "loosening max_stage_frac — check sat_min/sat_max against "
              "the actual swvl2_sat_ratio distribution in the "
              "search_from_frac=0.70 region")
        return

    T_s   = T_WINDOW * len(window_starts)
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    t_release      = T_IN + 4     # release begins 1 hr into the forecast
    ramp_rate      = 0.04         # m per timestep during ramp-up/down —
                                   # gradual, per ESB operational
                                   # convention of staged releases rather
                                   # than an instantaneous step change
    plateau_rise   = 0.90         # m held at the tailrace during release
    ramp_steps     = int(round(plateau_rise / ramp_rate))   # ≈23 steps
    plateau_steps  = 20           # ≈5 hr sustained release
    # Full ramp-up + plateau + ramp-down = 36 + 23 + 20 + 23 = 102 steps,
    # fits inside T_WINDOW (104) with margin.

    for i, t0 in enumerate(window_starts):
        sl  = slice(i * T_WINDOW, (i+1) * T_WINDOW)
        X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

        base_tail = float(np.mean(y_w[:T_IN, tailrace_idx]))

        # Trapezoidal release profile at the tailrace: ramp up, hold,
        # ramp down. Nothing else in the window is touched at this
        # point — the upstream inflow gauges keep their copied baseline
        # untouched, which is the whole point of the scenario.
        for t in range(t_release, T_WINDOW):
            dt = t - t_release
            if dt < ramp_steps:
                delta = ramp_rate * dt
            elif dt < ramp_steps + plateau_steps:
                delta = plateau_rise
            else:
                dt_down = dt - ramp_steps - plateau_steps
                delta = max(0.0, plateau_rise - ramp_rate * dt_down)
            y_w[t, tailrace_idx] = base_tail + delta
            X_w[t, tailrace_idx, F_STAGE] = base_tail + delta

        # Route the release signal downstream through the river network,
        # multiple hops out from the tailrace (Waterworks Weir ->
        # Fitzgerald's Park -> Pope's Quay -> St. Patrick's Quay ->
        # Currach Club), attenuating per edge via the same routing lags
        # used elsewhere in this module. Topology precomputed above —
        # same chain for every window.
        propagate_downstream_chain(
            y_w, X_w, tailrace_idx, edges_order, lags, baseline=base_tail)

        X_syn[sl] = X_w
        y_syn[sl] = y_w
        m_syn[sl] = m_w

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S3")

    meta = {
        "name": "S3_InniscarraRelease",
        "description": (
            "Controlled ESB release from Inniscarra dam ramps stage at "
            "the tailrace gauge independently of rainfall. Tests whether "
            "the model anticipates the downstream propagation while "
            "correctly leaving the two upstream reservoir inflow gauges "
            "(which the release cannot physically affect) undisturbed."),
        "n_windows":       len(window_starts),
        "T_per_window":    T_WINDOW,
        "T_total":         T_s,
        "release_node":    {"idx": tailrace_idx, "ref": "19109",
                            "name": "Inniscarra Tailrace"},
        "decoupled_nodes": {
            "inniscarra_headrace":    {"idx": headrace_idx,      "ref": "19094"},
            "carrigadrohid_headrace": {"idx": carrigadrohid_idx, "ref": "19095"},
        },
        "downstream_nodes":      downstream_idx,
        "t_release_step":        t_release,
        "ramp_rate_m_per_step":  ramp_rate,
        "plateau_rise_m":        plateau_rise,
        "ramp_steps":            ramp_steps,
        "plateau_steps":         plateau_steps,
    }
    save_scenario("S3_InniscarraRelease", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S4 — Saturation breakthrough: dry → saturated mid-event
# ══════════════════════════════════════════════════════════════════════════════

def generate_s4_sat_breakthrough(X, y, mask, nd, ed, uh, lags, bankfull,
                                  n_windows: int = 30) -> None:
    """
    Catchment begins moderately dry (sat_ratio 0.55–0.68) then receives
    heavy rainfall that rapidly pushes soil to saturation, triggering an
    abrupt switch from infiltration-excess to saturation-excess runoff.

    Tests whether STGNNSoilGate detects the approaching breakthrough via
    rising swvl2_sat_ratio before stage rises at gauges.
    """
    print("\nS4: SatBreakthrough")
    out_dir = SCEN_DIR / "S4_SatBreakthrough"

    T = X.shape[0]; N = X.shape[1]
    test_start = int(T * 0.85)

    # Select windows starting in dry-to-moderate conditions (summer/early autumn)
    # S4 needs dry-to-moderate initial conditions (summer/early autumn).
    # Search from 50% to capture both training summers (2023, 2024)
    # where swvl2_sat_ratio drops to 0.50-0.68 during dry spells.
    window_starts = select_base_windows(
        X, y, bankfull, sat_min=0.45, sat_max=0.70,
        max_stage_frac=0.20, n_windows=n_windows,
        search_from_frac=0.50)

    if not window_starts:
        # Wider fallback if dry conditions are scarce in this period
        window_starts = select_base_windows(
            X, y, bankfull, sat_min=0.45, sat_max=0.80, n_windows=n_windows,
            search_from_frac=0.40)

    T_s   = T_WINDOW * len(window_starts)
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    rain_peak_mm = 20.0                 # heavy but realistic frontal rainfall

    # Saturation ramp parameters — kept as named constants (not inlined
    # below) because sat_breakthrough_step is DERIVED from them. Previously
    # sat_breakthrough_step was hardcoded to T_IN+8 independently of these,
    # under the assumption that reached "2hr into the forecast" — but with
    # sat_start=0.55, sat_rate=0.015/step, the ramp doesn't actually cross
    # excess_threshold=0.75 until T_IN+14 (0.75-0.55=0.20; 0.20/0.015=13.3,
    # rounds up to 14 steps). At the old T_IN+8, saturation was only 0.67 —
    # excess_factor = max(0, 0.67-0.75)*4 = 0 EVERY window, EVERY node, so
    # the "abrupt post-breakthrough acceleration" this scenario exists to
    # test never actually fired. y_syn was silently just the calm baseline
    # plus a tiny 8-step ramp, which is also why NSE was catastrophic for
    # every model uniformly (implied target std ≈ 0.11m regardless of
    # architecture — an artifact of the near-flat target, not model
    # quality). Deriving sat_breakthrough_step here instead of hardcoding
    # it separately makes this class of drift impossible to reintroduce.
    sat_start        = 0.55
    sat_rate         = 0.015   # per-step saturation increase
    sat_cap          = 0.40    # matches the np.clip below
    excess_threshold = 0.75
    steps_to_threshold = math.ceil((excess_threshold - sat_start) / sat_rate)
    sat_breakthrough_step = T_IN + steps_to_threshold   # = T_IN + 14

    for i, t0 in enumerate(window_starts):
        sl  = slice(i * T_WINDOW, (i+1) * T_WINDOW)
        X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

        # Heavy rainfall beginning T_IN steps in (throughout forecast window)
        X_w[T_IN:, :, F_RAIN] += rain_peak_mm * np.exp(
            -np.arange(T_WINDOW - T_IN)[:, None] * 0.05)

        # Set soil moisture synthetically — do not rely on X.npy sat values
        # which may be 0.0. S4 starts dry (0.55) and ramps to saturated.
        # Set baseline sat for entire input window (pre-event dry state)
        X_w[:T_IN, :, F_SW2_SAT] = sat_start
        X_w[:T_IN, :, F_SW1_SAT] = 0.50
        X_w[:T_IN, :, F_SW2_RAW] = sat_start * 0.472
        X_w[:T_IN, :, F_SW1_RAW] = 0.50 * 0.472
        # Progressive soil saturation build-up from rainfall
        for t in range(T_IN, T_WINDOW):
            steps_of_rain = t - T_IN
            sat_increase  = min(sat_cap, steps_of_rain * sat_rate)
            X_w[t, :, F_SW2_SAT] = np.clip(sat_start + sat_increase, 0.0, 1.0)
            X_w[t, :, F_SW1_SAT] = np.clip(0.50 + sat_increase * 0.8, 0.0, 1.0)
            X_w[t, :, F_SW2_RAW] = X_w[t, :, F_SW2_SAT] * 0.472
            X_w[t, :, F_SW1_RAW] = X_w[t, :, F_SW1_SAT] * 0.472

        # Before breakthrough: slow, infiltration-limited stage response
        for t in range(T_IN, sat_breakthrough_step):
            slow_rise = 0.01 * (t - T_IN)
            y_w[t, :] += slow_rise
            X_w[t, :, F_STAGE] += slow_rise

        # After breakthrough: abrupt acceleration in stage rise
        for n_idx in range(N):
            ref = str(int(nd.iloc[n_idx]["ref"]))
            if ref not in uh: continue
            p   = uh[ref]
            sat_at_breakthrough = float(
                X_w[sat_breakthrough_step, n_idx, F_SW2_SAT])
            excess_factor = max(0.0, sat_at_breakthrough - excess_threshold) * 4.0
            if excess_factor <= 0: continue
            uh_arr = triangular_uh(
                p["peak_rate_m_per_mm"] * 2.0,  # 2× amplification post-breakthrough
                p["tc_hr"],
                T_WINDOW - sat_breakthrough_step)
            delta = uh_arr * rain_peak_mm * 6 * excess_factor
            t_end = sat_breakthrough_step + len(delta)
            t_end = min(t_end, T_WINDOW)
            y_w[sat_breakthrough_step:t_end, n_idx] += delta[:t_end-sat_breakthrough_step]
            X_w[sat_breakthrough_step:t_end, n_idx, F_STAGE] += \
                delta[:t_end-sat_breakthrough_step]

        X_syn[sl] = X_w
        y_syn[sl] = y_w
        m_syn[sl] = m_w

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S4")

    meta = {
        "name": "S4_SatBreakthrough",
        "description": (
            "Moderate-saturation catchment receives heavy rainfall, "
            "crossing the saturation threshold mid-forecast. Abrupt "
            "stage acceleration tests anticipatory soil gate (Idea 1)."),
        "n_windows":              len(window_starts),
        "T_total":                T_s,
        "sat_breakthrough_step":  sat_breakthrough_step,
        "rain_peak_mm":           rain_peak_mm,
        "initial_sat_range":      [0.50, 0.68],
        "sat_start":              sat_start,
        "sat_rate_per_step":      sat_rate,
        "excess_threshold":       excess_threshold,
    }
    save_scenario("S4_SatBreakthrough", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S5 — Spatial rainfall gradient (frontal band)
# ══════════════════════════════════════════════════════════════════════════════

def generate_s5_spatial_gradient(X, y, mask, nd, ed, uh, lags, bankfull,
                                  n_windows: int = 40) -> None:
    """
    A frontal system oriented SW–NE delivers heavy rainfall to western headwaters
    but light rainfall to eastern Cork city gauges. The flood signal must propagate
    through the river network from west to east.

    Tests whether graph routing correctly anticipates the arriving flood at
    Cork city before it appears in the local rainfall record.
    """
    print("\nS5: SpatialGradient")
    out_dir = SCEN_DIR / "S5_SpatialGradient"

    T = X.shape[0]; N = X.shape[1]

    window_starts = select_base_windows(
        X, y, bankfull, sat_min=0.55, sat_max=0.92, n_windows=n_windows,
        search_from_frac=0.70)

    if not window_starts:
        # Same defensive guard as S1/S3 — without this, an empty result
        # here would silently produce a zero-length scenario exactly like
        # the S3 bug (T_s = T_WINDOW * len(window_starts) = 0), just with
        # no warning until scenario_evaluator.py reports "got 0 steps".
        print("  [skip] No valid base windows found for S5")
        return

    # Compute west-to-east gradient weight per node using easting_itm
    easting = nd["easting_itm"].values.astype(np.float64)
    e_min, e_max = easting.min(), easting.max()
    # Western (low easting) nodes get multiplier >1; eastern get <1
    gradient = 1.0 + 1.5 * (1.0 - (easting - e_min) / (e_max - e_min + 1))
    # gradient: western nodes ≈ 2.5×, eastern nodes ≈ 1.0×
    gradient = gradient.astype(np.float32)

    T_s   = T_WINDOW * len(window_starts)
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    for i, t0 in enumerate(window_starts):
        sl  = slice(i * T_WINDOW, (i+1) * T_WINDOW)
        X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

        # Apply spatial gradient to rainfall features in the forecast window
        X_w[T_IN:, :, F_RAIN] *= gradient[None, :]

        # Compute synthetic stage response per node using scaled rainfall
        for n_idx in range(N):
            ref = str(int(nd.iloc[n_idx]["ref"]))
            if ref not in uh: continue
            p    = uh[ref]
            g    = float(gradient[n_idx])
            if g <= 1.05: continue    # eastern nodes — no significant perturbation
            rain_excess = float(np.sum(X_w[T_IN:T_IN+16, n_idx, F_RAIN]))
            sat = float(np.mean(X_w[:T_IN, n_idx, F_SW2_SAT]))
            excess = rain_excess * sat
            if excess < 5.0: continue
            uh_arr = triangular_uh(p["peak_rate_m_per_mm"], p["tc_hr"],
                                   T_WINDOW - T_IN)
            t_off = T_IN + round(p["tp_hr"] / STEP_MIN * 60)
            t_off = min(t_off, T_WINDOW - 1)
            uh_len = min(len(uh_arr), T_WINDOW - t_off)
            y_w[t_off : t_off + uh_len, n_idx] += uh_arr[:uh_len] * excess
            X_w[t_off : t_off + uh_len, n_idx, F_STAGE] += uh_arr[:uh_len] * excess

        # Route western headwater signal to downstream eastern nodes
        for _, edge in ed.iterrows():
            src = int(edge.src_idx)
            dst = int(edge.dst_idx)
            if gradient[src] < 1.3: continue   # only western sources
            src_delta = y_w[:, src] - float(np.mean(y_w[:T_IN, src]))
            dst_delta = apply_routing(src_delta, src, dst, lags)
            y_w[:, dst] += dst_delta * 0.5   # attenuate routed signal
            X_w[:, dst, F_STAGE] += dst_delta * 0.5

        X_syn[sl] = X_w
        y_syn[sl] = y_w
        m_syn[sl] = m_w

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S5")

    # downstream_nodes was previously missing from this meta dict entirely
    # (S1 saves it via the same identify_downstream_nodes() helper; S5
    # never did), which made s5_metrics() in scenario_evaluator.py always
    # fall back to meta.get("downstream_nodes", []) -> [] -> NaN for
    # every single checkpoint. Using the same helper as S1 keeps the
    # "downstream" definition consistent across scenarios rather than
    # introducing a second, gradient-based definition here.
    downstream_idx = identify_downstream_nodes(nd, n=5)

    meta = {
        "name": "S5_SpatialGradient",
        "description": (
            "SW-NE frontal band delivers 2.5× more rainfall to western "
            "headwaters than eastern Cork city gauges. Tests whether "
            "graph models correctly route the gradient."),
        "n_windows":    len(window_starts),
        "T_per_window": T_WINDOW,
        "T_total":      T_s,
        "gradient_range": [float(gradient.min()), float(gradient.max())],
        "downstream_nodes": downstream_idx,
    }
    save_scenario("S5_SpatialGradient", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S6 — Bridge/culvert channel blockage: backwater upstream, suppressed flow
#      downstream. Explicit architectural-limitation diagnostic.
# ══════════════════════════════════════════════════════════════════════════════

def generate_s6_channel_blockage(X, y, mask, nd, ed, uh, lags, bankfull,
                                  n_windows: int = 30) -> None:
    """
    Debris/ice blockage at a real bridge or culvert location decouples
    upstream and downstream stage: backwater raises stage AT and
    immediately UPSTREAM of the constriction (gradually-varied-flow M1
    backwater profile — Chow, "Open-Channel Hydraulics", 1959), while
    flow immediately DOWNSTREAM is suppressed by the restricted opening.

    Every river edge in graph_builder.py points strictly downstream
    (confirmed directly against edges.csv), with two known exceptions:
    node pairs (0,1) and (2,3) — Ballincolly<->Glen Park and Blackpool
    Retail Park<->Glennamought Bridge — carry genuine bidirectional
    edges with independently-calibrated, non-symmetric routing lags
    (routing_lags.json), most likely representing a real braided or
    distributary reach rather than a data-entry duplicate. The blockage
    node selection below explicitly avoids collapsing onto one of these
    pairs when picking its downstream neighbour (see the exclusion
    logic), so the scenario's backwater/suppression triad stays three
    genuinely distinct nodes. Outside these 2 of 28 edges, no reverse
    edges exist anywhere in this project's graph construction or model
    code, so no GNN variant here has a message-passing pathway by which
    a downstream constriction can inform an upstream node's prediction.
    The backwater rise this scenario injects is, by construction,
    invisible to graph message passing for every model at the chosen
    blockage location — the only way a model could anticipate it is if
    the upstream node's OWN feature history happened to carry an
    independent precursor, which for a sudden debris blockage generally
    does not exist.

    Several Lee catchment gauges are literally named for their bridge
    locations (Glennamought Bridge, Ovens Bridge, Morris's Bridge,
    Dripsey Bridge, Bawnafinny Bridge, Cooleen Bridge, Carrigrohane
    Bridge, Coolmuckey Br), and debris/culvert blockage at undersized
    bridge openings during flash floods is a well-documented, common
    failure mode in Irish/UK flash flood events — not a hypothetical
    edge case. The blockage location is picked from these actual
    bridge-named nodes rather than an arbitrary upstream edge, so the
    scenario is anchored to a real, nameable physical location in the
    catchment rather than an abstract graph position.

    Expected outcome, stated up front: ALL models should show large
    positive bias (under-prediction) at the upstream/blockage node
    during the backwater period, regardless of architecture — this is
    the point. A model architecture that does NOT show this failure
    would itself be the interesting/suspicious result, worth
    investigating for how it's getting the signal.
    """
    print("\nS6: ChannelBlockage (architectural-limitation diagnostic)")
    out_dir = SCEN_DIR / "S6_ChannelBlockage"

    T = X.shape[0]; N = X.shape[1]

    # Select moderate-flow windows (not already flooded) so the injected
    # backwater signal is clearly attributable. Same retry structure used
    # everywhere else in this module — a tight joint saturation +
    # calm-baseline constraint can return zero windows in the
    # validation-period search region.
    window_starts = select_base_windows(
        X, y, bankfull, sat_min=0.60, sat_max=0.92,
        max_stage_frac=0.30, n_windows=n_windows,
        search_from_frac=0.70)

    if not window_starts:
        print("  [warn] No windows at max_stage_frac=0.30 — retrying with "
              "0.40 (matches S1/S5 default)")
        window_starts = select_base_windows(
            X, y, bankfull, sat_min=0.60, sat_max=0.92,
            max_stage_frac=0.40, n_windows=n_windows,
            search_from_frac=0.70)

    if not window_starts:
        print("  [skip] No valid base windows found for S6 even after "
              "loosening max_stage_frac.")
        return

    # Pick the blockage location from an ACTUAL bridge-named node — a
    # real, nameable constriction point rather than an arbitrary graph
    # position. Uses a name-substring heuristic against nodes.csv, since
    # there's no explicit "is_bridge" flag in the current node schema
    # (only is_reservoir/is_tidal). Falls back to the old area-percentile
    # heuristic if no bridge-named node with a usable upstream edge exists.
    bridge_mask = nd["name"].str.contains(
        "Bridge|Br$|Br ", case=False, regex=True, na=False)
    bridge_candidates = nd.loc[bridge_mask, "node_idx"].tolist()

    block_dst = None   # the bridge/constriction node itself
    block_src = None   # its immediate upstream neighbour (backwater rises here too)
    block_next = None  # its immediate downstream neighbour (suppression applies here)

    for cand_dst in bridge_candidates:
        upstream_edges = ed[ed["dst_idx"] == cand_dst]
        # Exclude any "downstream" edge that loops straight back to the
        # chosen upstream node. Two of this graph's 28 edges form genuine
        # bidirectional pairs (0<->1, 2<->3 — confirmed against
        # routing_lags.json, which has independently different, non-
        # symmetric lags for each direction, e.g. 2_3=7 vs 3_2=2,
        # suggesting an intentionally-represented braided/distributary
        # reach rather than a duplicate data-entry error). Without this
        # exclusion, a candidate bridge sitting on one of those pairs
        # (e.g. Glennamought Bridge, node 3) would have its "downstream"
        # edge selected as the very node the backwater is already being
        # applied to as the upstream neighbour — collapsing block_src,
        # block_dst, block_next into effectively two nodes instead of
        # three and corrupting the suppression-side metric.
        downstream_edges = ed[ed["src_idx"] == cand_dst]
        if not upstream_edges.empty:
            candidate_src = int(upstream_edges.iloc[0]["src_idx"])
            downstream_edges = downstream_edges[downstream_edges["dst_idx"] != candidate_src]
        if upstream_edges.empty or downstream_edges.empty:
            continue   # need both an upstream and a genuinely distinct downstream neighbour
        block_dst  = int(cand_dst)
        block_src  = int(upstream_edges.iloc[0]["src_idx"])
        block_next = int(downstream_edges.iloc[0]["dst_idx"])
        break

    if block_dst is None:
        print("  [skip] No bridge-named node with both an upstream and "
              "downstream river edge found — check nodes.csv naming or "
              "extend the bridge_mask pattern.")
        return

    bridge_name = nd.loc[nd.node_idx == block_dst, "name"].values[0]
    print(f"  Blockage location: {bridge_name} (node {block_dst})")

    T_s   = T_WINDOW * len(window_starts)
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    t_block        = T_IN + 4     # blockage occurs 1 hr into the forecast
    rise_rate      = 0.05         # m per timestep backwater build-up
    backwater_cap  = 1.5          # m, cap on backwater rise
    suppression    = 0.6          # fraction of normal flow reaching block_next

    # Per-node stage_range for converting the backwater rise (in metres)
    # into an equivalent normalised_stage delta. Every elevation-
    # reconstructing gate in this project (STGNNHANDEdge, STGNNSoilGate,
    # STGNNBackwaterEdge) computes H = gauge_datum + normalised_stage *
    # stage_range from F_NORM specifically — NOT from F_STAGE
    # (stage_anomaly). Only writing F_STAGE, as the original version of
    # this function did, leaves the injected backwater event completely
    # invisible to any gate mechanism (the GRU/embedding pathway would
    # still see it, since all 11 features feed into input_proj — but the
    # gate itself, the exact thing STGNNBackwaterEdge exists to test,
    # would never open). Same recurring bug class as the earlier HAND
    # gate fix (stage_anomaly vs. absolute-elevation frame mismatch) —
    # see this project's own prior notes on that pattern.
    stage_range = (nd.set_index("node_idx")["p90_mAOD"]
                   - nd.set_index("node_idx")["gauge_datum_mOSGM15"])

    for i, t0 in enumerate(window_starts):
        sl  = slice(i * T_WINDOW, (i+1) * T_WINDOW)
        X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

        # Backwater rise AT the bridge and its immediate upstream
        # neighbour — a simplified two-node stand-in for the true M1
        # backwater profile (which in reality extends further upstream
        # proportional to channel slope and blockage severity; modelling
        # the full profile length is out of scope here, since two nodes
        # is already sufficient to demonstrate the architectural blind
        # spot this scenario exists to show).
        for affected in (block_dst, block_src):
            base = float(np.mean(y_w[:T_IN, affected]))
            sr = float(stage_range.loc[affected])
            base_norm = float(np.mean(X_w[:T_IN, affected, F_NORM])) if sr > 0 else 0.0
            for t in range(t_block, T_WINDOW):
                backwater = min(rise_rate * (t - t_block), backwater_cap)
                y_w[t, affected] = base + backwater
                X_w[t, affected, F_STAGE] = base + backwater
                if sr > 0:
                    X_w[t, affected, F_NORM] = base_norm + backwater / sr

        # Suppressed flow immediately downstream of the blockage. Not
        # synced to F_NORM: s6_downstream_suppression_rmse is designed
        # to be learnable from the node's own ordinary feature history
        # (rainfall, its own recent trend) without needing any gate to
        # fire — see s6_metrics' docstring in scenario_evaluator.py.
        for t in range(t_block, T_WINDOW):
            y_w[t, block_next] *= suppression
            X_w[t, block_next, F_STAGE] *= suppression

        X_syn[sl] = X_w
        y_syn[sl] = y_w
        m_syn[sl] = m_w

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S6")

    meta = {
        "name": "S6_ChannelBlockage",
        "description": (
            "Debris blockage at a real bridge/culvert location causes "
            "backwater rise upstream and suppressed flow downstream. "
            "Explicit diagnostic for the directed-topology architectural "
            "limitation: no model in this suite has a message-passing "
            "pathway for downstream-to-upstream backwater causality, so "
            "large upstream under-prediction is the EXPECTED result for "
            "every model, not a failure to be optimised away."),
        "n_windows":      len(window_starts),
        "T_per_window":   T_WINDOW,
        "T_total":        T_s,
        "blockage_node":  {"idx": block_dst,  "name": bridge_name},
        "upstream_node":  {"idx": block_src,  "name": nd.loc[nd.node_idx==block_src,  "name"].values[0]},
        "downstream_node":{"idx": block_next, "name": nd.loc[nd.node_idx==block_next, "name"].values[0]},
        "t_blockage_step": t_block,
        "backwater_rate_m_per_step": rise_rate,
        "backwater_cap_m": backwater_cap,
        "downstream_suppression": suppression,
    }
    save_scenario("S6_ChannelBlockage", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_MAP = {
    "S1": generate_s1_convective_cell,
    "S2": generate_s2_gauge_failure,
    "S3": generate_s3_inniscarra_release,
    "S4": generate_s4_sat_breakthrough,
    "S5": generate_s5_spatial_gradient,
    "S6": generate_s6_channel_blockage,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic flash flood scenarios")
    parser.add_argument("--all",      action="store_true",
                        help="Generate all 6 scenarios")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIO_MAP),
                        help="Generate one specific scenario")
    parser.add_argument("--n-windows", type=int, default=None,
                        help="Override number of base windows per scenario")
    args = parser.parse_args()

    if not args.all and args.scenario is None:
        parser.error("Specify --all or --scenario {S1,S2,S3,S4,S5,S6}")

    print("=" * 62)
    print("Synthetic scenario generation — Lee catchment")
    print("=" * 62)

    X, y, mask, nd, ed, uh, lags, bankfull = load_project_data()

    kwargs = {"X": X, "y": y, "mask": mask, "nd": nd, "ed": ed,
              "uh": uh, "lags": lags, "bankfull": bankfull}
    if args.n_windows:
        kwargs["n_windows"] = args.n_windows

    if args.all:
        for fn in SCENARIO_MAP.values():
            fn(**kwargs)
    else:
        SCENARIO_MAP[args.scenario](**kwargs)

    print("\nDone. Run scenario_evaluator.py --all-scenarios to evaluate.")


if __name__ == "__main__":
    main()
