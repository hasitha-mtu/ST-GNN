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
                               lags: dict, baseline: float,
                               stage_range: Optional[pd.Series] = None) -> list[int]:
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

    stage_range: if given, also syncs F_NORM at each hop using the SAME
    incremental dst_delta just computed for F_STAGE — this must happen
    HERE, incrementally, rather than as a post-hoc "current minus
    original baseline" calculation by the caller. When multiple source
    chains (e.g. S1's several headwaters) converge on the same
    downstream node across separate calls, a post-hoc recomputation
    would double-count whatever an earlier call already added; doing it
    incrementally inside the one place dst_delta is actually computed
    avoids that entirely.

    Returns the list of node indices reached, in hop order.
    """
    signal = {source_idx: y_w[:, source_idx] - baseline}
    reached: list[int] = []

    for src, dst in edges_order:
        dst_delta = apply_routing(signal[src], src, dst, lags)
        y_w[:, dst] += dst_delta
        X_w[:, dst, F_STAGE] += dst_delta
        if stage_range is not None:
            sr = float(stage_range.loc[dst])
            if sr > 0:
                X_w[:, dst, F_NORM] += dst_delta / sr
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


# ══════════════════════════════════════════════════════════════════════════════
# Multi-realization parameter sampling (reviewer point 2)
# ══════════════════════════════════════════════════════════════════════════════
#
# Previously, each scenario applied ONE fixed, deterministic set of
# injection parameters (pulse magnitude, release ramp rate, blockage
# severity, etc.) across many different REAL historical base windows.
# That gives variation in antecedent conditions but not in the synthetic
# event itself -- a reviewer can reasonably ask whether conclusions
# depend on the one specific magnitude chosen. This module samples
# injection MAGNITUDE from a physically-justified range per realization,
# while deliberately keeping injection TIMING (pulse_at_step,
# t_release_step, t_blockage_step, sat_breakthrough_step) fixed across
# realizations -- every s1..s6_metrics function in scenario_evaluator.py
# reads timing as a single scalar from meta shared across all windows;
# varying timing per realization would require rewriting every one of
# those alignment computations. Varying magnitude does not, since the
# event still happens at the same relative step every time -- only how
# large it is changes. This directly addresses "do conclusions depend on
# the chosen perturbation parameters" without touching the evaluation
# side at all.
#
# Base windows (from select_base_windows) are computed ONCE per scenario
# and reused across all realizations, not redrawn per realization --
# select_base_windows has no randomness of its own, so calling it again
# with the same arguments returns the same windows, not new ones. This
# also cleanly separates two different axes of variation: which
# historical period (existing, via n_windows) vs. how severe the
# synthetic event is (new, via n_realizations) -- conflating them would
# make it harder to attribute any given result to one or the other.

# S1: pulse intensity. Marchi et al. (2010, J. Hydrology) characterise a
# genuine range of rainfall intensities across documented European flash
# flood events, not one fixed figure -- default of 30mm/15min sits near
# the middle of a plausible range for an isolated convective cell.
S1_PULSE_PEAK_RANGE_MM = (15.0, 45.0)

# S3: release severity. Real ESB-style controlled releases vary in both
# how high they ramp and how quickly -- no single fixed magnitude
# represents "a release event" any more than one fixed rainfall
# intensity represents "a storm".
S3_PLATEAU_RISE_RANGE_M   = (0.5, 1.3)
S3_RAMP_RATE_RANGE_M_STEP = (0.03, 0.08)

# S4's realization parameters (excess_threshold, sat_start, sat_rate,
# rain_peak_mm) are sampled by sample_s4_realization_params(), defined
# just above generate_s4_sat_breakthrough below -- kept there rather
# than here because sat_breakthrough_step is DERIVED from the other
# three and that derivation belongs next to the parameters it depends
# on, not separated from them.

# S5: gradient ratio. Zhu, Wright & Yu (2018, WRR) support a range of
# spatial rainfall heterogeneity intensities, not one fixed 2.5x ratio.
S5_GRADIENT_MAX_RANGE = (1.8, 3.0)

# S6: blockage severity, expressed as a single 0-1 fraction rather than
# two independently-sampled numbers -- McDermott & Quinn (2023) report
# ~50% flow-capacity reduction during a real documented Irish blockage
# event; that figure anchors the centre of this range rather than being
# hardcoded as the only value ever tested. backwater_cap and
# suppression_cap are BOTH derived from this one severity value below,
# consistent with the "single shared parameter, not two independent
# guesses" principle already discussed for this scenario.
S6_SEVERITY_RANGE = (0.30, 0.75)


def sample_s1_params(rng: np.random.Generator) -> dict:
    return {"pulse_peak_mm_per_15min": float(rng.uniform(*S1_PULSE_PEAK_RANGE_MM))}


def sample_s3_params(rng: np.random.Generator) -> dict:
    return {
        "plateau_rise_m":    float(rng.uniform(*S3_PLATEAU_RISE_RANGE_M)),
        "ramp_rate_m_per_step": float(rng.uniform(*S3_RAMP_RATE_RANGE_M_STEP)),
    }


def sample_s5_params(rng: np.random.Generator) -> dict:
    return {"gradient_max": float(rng.uniform(*S5_GRADIENT_MAX_RANGE))}


def sample_s6_params(rng: np.random.Generator) -> dict:
    severity = float(rng.uniform(*S6_SEVERITY_RANGE))
    # Derived, not independently sampled: a more severe blockage produces
    # both a larger backwater rise AND a larger downstream suppression,
    # from the same underlying physical cause -- sampling them
    # independently would let a random draw produce a large backwater
    # rise with almost no downstream suppression, which isn't a
    # physically coherent combination for a single blockage event.
    return {
        "severity":         severity,
        "backwater_cap_m":  round(0.8 + severity * 1.4, 4),    # 0.8-2.0m at severity 0-1
        "suppression_cap_m": round(0.1 + severity * 0.35, 4),  # 0.1-0.45m at severity 0-1
    }


def select_base_windows(
    X: np.ndarray, y: np.ndarray,
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
                                 n_windows: int = 50,
                                 n_realizations: int = 20,
                                 seed: int = 0) -> None:
    """
    Isolated convective storm over 6 headwater gauges. Rapid stage rise
    within 45–90 minutes. Flood pulse arrives at Cork city gauges 2–4 hr later.

    Physical motivation: tests whether cross-tributary HAND edges carry the
    lateral flood signal before it reaches the Cork city gauges via the
    river-network alone.

    n_realizations: number of independent draws of pulse_peak_mm_per_15min
    from S1_PULSE_PEAK_RANGE_MM, applied across the SAME window_starts
    (see the multi-realization module docstring above this function for
    why timing stays fixed while only magnitude varies). Total generated
    windows = n_windows * n_realizations.
    """
    print("\nS1: ConvectiveCell")
    out_dir = SCEN_DIR / "S1_ConvectiveCell"
    rng = np.random.default_rng(seed)

    headwater_idx = identify_headwater_nodes(nd, n=6)

    # BUG FIX: identify_downstream_nodes previously picked the 5 largest
    # catchments independent of whether they're actually reachable from
    # any headwater via the directed river graph. Direct BFS check found
    # ZERO 1-hop edges from headwater_idx to that node set, and 4 of the
    # 6 headwaters were entirely unreachable to it at ANY hop count —
    # meaning the "downstream (routed)" signal this scenario's own
    # stated purpose depends on ("tests whether cross-tributary HAND
    # edges carry the lateral flood signal before it reaches the Cork
    # city gauges via the river-network alone") has been exactly zero in
    # every S1 window ever generated, for every checkpoint evaluated
    # against it. Fixed by restricting downstream_idx to nodes that are
    # actually reachable from at least one headwater, ranked by
    # catchment size among that reachable set (same spirit as
    # identify_downstream_nodes, now topology-aware).
    reachable = set()
    headwater_chains: dict[int, list[tuple[int, int]]] = {}
    for hw in headwater_idx:
        edges_order = downstream_bfs_edges(hw, ed)
        headwater_chains[hw] = edges_order
        reachable.update(dst for _src, dst in edges_order)

    downstream_candidates = nd[(nd.is_reservoir == 0)
                               & (nd.node_idx.isin(reachable))
                               & (~nd.node_idx.isin(headwater_idx))].copy()
    downstream_candidates = downstream_candidates.sort_values("log_catchment_area_km2", ascending=False)
    downstream_idx = downstream_candidates.head(5)["node_idx"].tolist()

    if not downstream_idx:
        print("  [skip] No downstream nodes reachable from any headwater — "
              "check headwater_idx/graph topology before proceeding.")
        return
    if len(downstream_idx) < 5:
        print(f"  [warn] Only {len(downstream_idx)} downstream nodes reachable "
              f"from headwater_idx (wanted 5): {downstream_idx}")

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

    pulse_steps = 6        # 90 minutes, fixed across realizations
    pulse_at = T_IN + 2    # storm begins 2 steps into prediction horizon, fixed

    T_s   = T_WINDOW * len(window_starts) * n_realizations
    X_syn = np.zeros((T_s, X.shape[1], X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, y.shape[1]),              dtype=np.float32)
    m_syn = np.zeros((T_s, mask.shape[1]),           dtype=np.float32)

    # Per-node stage_range for converting injected stage deltas (in
    # metres) into equivalent normalised_stage deltas. S1's own stated
    # purpose is "tests whether cross-tributary HAND edges carry the
    # lateral flood signal" — but HAND/SoilGate/BackwaterEdge's gates all
    # reconstruct H = gauge_datum + normalised_stage * stage_range from
    # F_NORM specifically, not F_STAGE (stage_anomaly). Without also
    # writing F_NORM here, the injected convective pulse is invisible to
    # every gate mechanism this scenario exists to test — same bug
    # class, same fix, as generate_s6_channel_blockage.
    stage_range = (nd.set_index("node_idx")["p90_mAOD"]
                   - nd.set_index("node_idx")["gauge_datum_mOSGM15"])

    realizations_meta = []
    window_realization_id = []
    row = 0

    for real_id in range(n_realizations):
        params = sample_s1_params(rng)
        pulse_peak = params["pulse_peak_mm_per_15min"]
        realizations_meta.append({"id": real_id, **params})

        # Build the convective rainfall pulse (triangular, 90 min duration)
        # for THIS realization's magnitude.
        pulse = np.zeros(pulse_steps)
        half  = pulse_steps // 2
        for t in range(half): pulse[t] = pulse_peak * t / half
        for t in range(half, pulse_steps): pulse[t] = pulse_peak * (pulse_steps - t) / half

        for t0 in window_starts:
            sl = slice(row * T_WINDOW, (row+1) * T_WINDOW)
            window_realization_id.append(real_id)

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
                sr = float(stage_range.loc[n_idx])
                if sr > 0:
                    X_w[t_start : t_start + uh_len, n_idx, F_NORM] += stage_delta[:uh_len] / sr

            # Route headwater signal downstream through the river network,
            # multiple hops out from each headwater — a single-pass edge loop
            # (the previous implementation) only reaches nodes one edge away
            # from a headwater, which the BFS check above confirmed never
            # includes any of this scenario's own downstream evaluation
            # nodes. Reuses the same per-headwater topology precomputed
            # before the window loop (static, doesn't depend on window
            # content) and the same propagate_downstream_chain mechanism S3
            # uses, called once per headwater that actually received a pulse
            # this window — += accumulation in propagate_downstream_chain
            # correctly sums contributions where multiple headwaters' chains
            # converge on the same downstream node.
            for n_idx in headwater_idx:
                if y_w[:, n_idx].max() <= y_w[:T_IN, n_idx].max():
                    continue   # no signal to route from this headwater
                base = float(np.mean(y_w[:T_IN, n_idx]))
                propagate_downstream_chain(
                    y_w, X_w, n_idx, headwater_chains[n_idx], lags,
                    baseline=base, stage_range=stage_range)

            X_syn[sl] = X_w
            y_syn[sl] = y_w
            m_syn[sl] = m_w
            row += 1

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S1")

    meta = {
        "name": "S1_ConvectiveCell",
        "description": (
            "Isolated convective storm over 6 headwater gauges. "
            "Rapid stage rise within 90 min. Tests HAND topology advantage."),
        "n_windows":       row,
        "n_windows_per_realization": len(window_starts),
        "n_realizations":  n_realizations,
        "T_per_window":    T_WINDOW,
        "T_total":         T_s,
        "headwater_nodes": headwater_idx,
        "downstream_nodes": downstream_idx,
        "pulse_duration_steps": pulse_steps,
        "pulse_at_step":   pulse_at,
        "realizations":    realizations_meta,
        "window_realization_id": window_realization_id,
    }
    save_scenario("S1_ConvectiveCell", out_dir, X_syn, y_syn, m_syn, meta)


def compute_node_criticality(nd: pd.DataFrame, ed: pd.DataFrame) -> list[int]:
    """
    Ranks all N nodes by downstream-reachability count (how many other
    nodes sit downstream of, and therefore hydraulically depend on
    routing information passing through, this one) — a direct,
    topology-grounded measure of "how disruptive would losing this
    gauge be to the network's information flow", used to distinguish
    random gauge failure from failure targeted at hydrologically/
    network-critical gauges (reviewer point 1).

    Returns node indices sorted descending by reachability count (most
    critical first).
    """
    N = len(nd)
    reach_counts = []
    for n in range(N):
        chain = downstream_bfs_edges(n, ed)
        reach_counts.append((n, len(set(dst for _s, dst in chain))))
    reach_counts.sort(key=lambda x: -x[1])
    return [n for n, _ in reach_counts]


# S2: failure severity as a FRACTION of the network (27 gauges), not a
# fixed count -- a fixed [1,3,5] doesn't scale meaningfully if the gauge
# network size ever changes, and doesn't span "how bad can it get"
# (reviewer point 1 asks for 5/10/20/30/40%).
S2_FAILURE_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40]

# Failure duration: wide range from a brief outage to effectively
# permanent-within-window (T_WINDOW=104) -- a fixed "fails and never
# recovers" (the old behaviour) is one point on this range, not the
# only condition worth testing for a genuine resilience curve.
S2_DURATION_RANGE_STEPS = (8, 90)


# ══════════════════════════════════════════════════════════════════════════════
# S2 — Progressive gauge failure during active flood
# ══════════════════════════════════════════════════════════════════════════════

def generate_s2_gauge_failure(X, y, mask, nd, ed, uh, lags, bankfull,
                               n_windows: int = 40,
                               n_random_repeats: int = 4,
                               n_windows_per_realization: int = 2,
                               seed: int = 0) -> None:
    """
    Real flood windows with gauges failed from t_failure onward.

    Rebuilt from a fixed [1,3,5]-gauge-count, headwater-only, permanent-
    failure design into a genuine sensor-network resilience experiment
    (reviewer point 1), combined with point 2's multi-realization
    requirement rather than treated as separate work, since both touch
    this same function:

      - Severity: S2_FAILURE_FRACTIONS = [5,10,20,30,40]% of the 27-gauge
        network, not a fixed count.
      - Selection: "random" (uniform draw from all 27 gauges, repeated
        n_random_repeats times per fraction for robustness against one
        unlucky/lucky draw) vs "critical" (the n_fail gauges with the
        highest downstream-reachability count, via
        compute_node_criticality() — deterministic per fraction, so not
        repeated the same way, but still combined with varying duration
        across its own realizations for some variation).
      - Duration: S2_DURATION_RANGE_STEPS, sampled per realization —
        from a brief outage to effectively permanent-within-window,
        rather than only ever "fails and never recovers".

    Unlike S1/S3/S4/S5/S6, WHICH NODES fail is itself random per
    realization (not just a magnitude), so s2_metrics cannot reconstruct
    the failed-node set from a fixed formula the way it previously did
    (headwater_idx[:n_fail] from a deterministic block index) — this
    version explicitly saves fail_nodes and duration_steps per
    realization in meta, and s2_metrics has been rewritten to read them.
    """
    print("\nS2: GaugeFailure (multi-realization sensor-network resilience)")
    out_dir = SCEN_DIR / "S2_GaugeFailure"
    rng = np.random.default_rng(seed)

    T     = X.shape[0]
    N     = X.shape[1]
    test_start = int(T * 0.85)

    flood_windows = []
    t = test_start
    while t < T - T_WINDOW and len(flood_windows) < n_windows:
        y_w = y[t : t + T_WINDOW]
        n_exceeding = int(np.sum(
            np.nanmax(y_w[T_IN:T_IN+24], axis=0) > 0.5 * bankfull))
        if n_exceeding >= 3:
            flood_windows.append(t)
        t += T_IN // 2

    if not flood_windows:
        print("  [warn] No flood windows found — using high-stage windows")
        flood_windows = select_base_windows(
            X, y, bankfull, sat_min=0.80, sat_max=1.0, n_windows=n_windows)
    if not flood_windows:
        print("  [skip] No valid base windows found for S2")
        return

    all_nodes = list(range(N))
    critical_order = compute_node_criticality(nd, ed)
    t_failure = T_IN + 4   # failure onset fixed across all realizations

    # Build the full list of (n_fail, selection_mode) configurations,
    # then n_random_repeats realizations for "random" and 1 realization
    # (still with its own sampled duration) for "critical" per fraction —
    # "critical" selection is deterministic given a fraction, so
    # repeating it with the SAME node set wouldn't test anything new the
    # way repeating "random" draws does.
    configs = []
    for frac in S2_FAILURE_FRACTIONS:
        n_fail = max(1, round(frac * N))
        configs.append((n_fail, frac, "critical", 1))
        configs.append((n_fail, frac, "random",   n_random_repeats))

    realizations_meta = []
    window_realization_id = []
    all_windows: list[tuple[int, int]] = []   # (t0, realization_id)

    real_id = 0
    window_pool_idx = 0
    for n_fail, frac, mode, n_repeats in configs:
        for _ in range(n_repeats):
            if mode == "critical":
                fail_nodes = critical_order[:n_fail]
            else:
                fail_nodes = sorted(rng.choice(all_nodes, size=n_fail, replace=False).tolist())
            duration_steps = int(rng.integers(*S2_DURATION_RANGE_STEPS))

            realizations_meta.append({
                "id": real_id, "failure_fraction": frac, "n_fail": n_fail,
                "selection_mode": mode, "fail_nodes": fail_nodes,
                "duration_steps": duration_steps,
            })

            n_take = min(n_windows_per_realization, len(flood_windows))
            chosen_windows = [flood_windows[(window_pool_idx + k) % len(flood_windows)]
                              for k in range(n_take)]
            window_pool_idx += n_take
            for t0 in chosen_windows:
                all_windows.append((t0, real_id))
                window_realization_id.append(real_id)
            real_id += 1

    if not all_windows:
        print("  [skip] S2: no windows available after realization allocation")
        return

    n_total = len(all_windows)
    T_s   = T_WINDOW * n_total
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    real_by_id = {r["id"]: r for r in realizations_meta}

    for i, (t0, rid) in enumerate(all_windows):
        sl = slice(i * T_WINDOW, (i + 1) * T_WINDOW)
        X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

        r = real_by_id[rid]
        fail_nodes = r["fail_nodes"]
        duration_steps = r["duration_steps"]
        t_recover = min(t_failure + duration_steps, T_WINDOW)

        # Zero dynamic features for failed gauges from t_failure until
        # t_recover (or end of window, whichever comes first) -- unlike
        # the original always-permanent-to-end-of-window version, a
        # short duration_steps draw lets the gauge come back online
        # within the same window.
        X_w[t_failure:t_recover, fail_nodes, F_STAGE] = 0.0
        X_w[t_failure:t_recover, fail_nodes, F_DHDT]  = 0.0
        X_w[t_failure:t_recover, fail_nodes, F_DISC]  = 0.0
        X_w[t_failure:t_recover, fail_nodes, F_RAIN]  = 0.0
        m_w[t_failure:t_recover, fail_nodes] = 0.0

        X_syn[sl] = X_w
        y_syn[sl] = y_w
        m_syn[sl] = m_w

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S2")

    meta = {
        "name": "S2_GaugeFailure",
        "description": (
            "Sensor-network resilience experiment: gauges fail from "
            "t_failure for a sampled duration. Severity spans 5-40% of "
            "the 27-gauge network; selection is either uniformly random "
            "(repeated for robustness) or targeted at the most "
            "hydrologically/network-critical gauges by downstream-"
            "reachability count; duration varies from brief to "
            "effectively permanent-within-window."),
        "n_windows":       n_total,
        "n_realizations":  len(realizations_meta),
        "n_windows_per_realization": n_windows_per_realization,
        "T_per_window":    T_WINDOW,
        "T_total":         T_s,
        "t_failure_step":  t_failure,
        "failure_fractions": S2_FAILURE_FRACTIONS,
        "realizations":          realizations_meta,
        "window_realization_id": window_realization_id,
    }
    save_scenario("S2_GaugeFailure", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S3 — Inniscarra reservoir release: outlet-driven signal decoupled from
#      upstream inflow / rainfall
# ══════════════════════════════════════════════════════════════════════════════

def generate_s3_inniscarra_release(X, y, mask, nd, ed, uh, lags, bankfull,
                                    n_windows: int = 30,
                                    n_realizations: int = 20,
                                    seed: int = 0) -> None:
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

    n_realizations: independent draws of (ramp_rate_m_per_step,
    plateau_rise_m) from sample_s3_params(), applied across the SAME
    window_starts (same principle as S1: which historical period vs.
    how severe the synthetic event is are separate axes of variation).
    Release ONSET timing (t_release) stays fixed across realizations,
    so s3_metrics needs no changes -- only magnitude/duration vary.
    """
    print("\nS3: InniscarraRelease (multi-realization)")
    out_dir = SCEN_DIR / "S3_InniscarraRelease"
    rng = np.random.default_rng(seed)

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

    release_relevant_nodes = [tailrace_idx] + downstream_idx

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

    T_s   = T_WINDOW * len(window_starts) * n_realizations
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    t_release      = T_IN + 4     # release begins 1 hr into the forecast, fixed across realizations
    plateau_steps  = 20           # ≈5 hr sustained release, fixed across realizations

    realizations_meta = []
    window_realization_id = []
    row = 0

    for real_id in range(n_realizations):
        params = sample_s3_params(rng)
        ramp_rate    = params["ramp_rate_m_per_step"]
        plateau_rise = params["plateau_rise_m"]
        ramp_steps   = int(round(plateau_rise / ramp_rate))
        realizations_meta.append({"id": real_id, **params, "ramp_steps": ramp_steps})

        for t0 in window_starts:
            sl  = slice(row * T_WINDOW, (row+1) * T_WINDOW)
            window_realization_id.append(real_id)

            X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
            y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
            m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

            base_tail = float(np.mean(y_w[:T_IN, tailrace_idx]))

            # Trapezoidal release profile at the tailrace: ramp up, hold,
            # ramp down. Nothing else in the window is touched at this
            # point — the upstream inflow gauges keep their copied
            # baseline untouched, which is the whole point of the scenario.
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

            propagate_downstream_chain(
                y_w, X_w, tailrace_idx, edges_order, lags, baseline=base_tail)

            X_syn[sl] = X_w
            y_syn[sl] = y_w
            m_syn[sl] = m_w
            row += 1

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S3")

    meta = {
        "name": "S3_InniscarraRelease",
        "description": (
            "Controlled ESB release from Inniscarra dam ramps stage at "
            "the tailrace gauge independently of rainfall. Tests whether "
            "the model anticipates the downstream propagation while "
            "correctly leaving the two upstream reservoir inflow gauges "
            "(which the release cannot physically affect) undisturbed. "
            "Multi-realization: ramp_rate/plateau_rise vary per real "
            "ESB-style staged-release documentation; release onset "
            "timing stays fixed."),
        "n_windows":       row,
        "n_windows_per_realization": len(window_starts),
        "n_realizations":  n_realizations,
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
        "plateau_steps":         plateau_steps,
        "realizations":          realizations_meta,
        "window_realization_id": window_realization_id,
    }
    save_scenario("S3_InniscarraRelease", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S4 — Saturation breakthrough: dry → saturated mid-event
# ══════════════════════════════════════════════════════════════════════════════

def sample_s4_realization_params(rng: np.random.Generator) -> dict:
    """
    Physically-plausible parameter draw for one S4 realization.

    excess_threshold is grounded directly in the peer-reviewed literature
    already cited for this scenario (see the literature-backing pass
    earlier in this project): Meissl et al. (2023, Heliyon) report a
    critical saturation deficit of ~0.28 (72% of pore volume filled) in
    an Alpine catchment before exceptionally high runoff coefficients
    occur; Western & Grayson (1998) and Penna et al. (2011, HESS)
    independently document threshold-like runoff generation in the same
    general range. Sampled here as uniform(0.68, 0.80), centered on that
    ~0.72 reported value with a modest spread rather than treating it as
    a single precise constant.

    rain_peak_mm, sat_start, and sat_rate do NOT have the same direct
    literature anchor -- they are documented, physically reasonable
    ranges (heavy-but-plausible frontal/convective rainfall; dry-to-
    moderate antecedent saturation matching this scenario's own window-
    selection criteria; a soil-wetting rate consistent with the range of
    sat_rate values already validated in single-realization testing
    earlier this session) rather than values independently confirmed
    against a specific cited source. Flagged explicitly here so the
    methods section doesn't overstate which numbers are literature-
    grounded and which are reasoned defaults.
    """
    excess_threshold = float(rng.uniform(0.68, 0.80))
    sat_start         = float(rng.uniform(0.50, 0.65))
    sat_rate          = float(rng.uniform(0.010, 0.020))
    rain_peak_mm      = float(rng.uniform(12.0, 28.0))

    steps_to_threshold = math.ceil((excess_threshold - sat_start) / sat_rate)
    sat_breakthrough_step = T_IN + steps_to_threshold

    return {
        "excess_threshold":      excess_threshold,
        "sat_start":             sat_start,
        "sat_rate":              sat_rate,
        "rain_peak_mm":          rain_peak_mm,
        "sat_breakthrough_step": sat_breakthrough_step,
    }


def generate_s4_sat_breakthrough(X, y, mask, nd, ed, uh, lags, bankfull,
                                  n_realizations: int = 20,
                                  n_windows_per_realization: int = 2,
                                  seed: int = 0) -> None:
    """
    Catchment begins moderately dry then receives heavy rainfall that
    rapidly pushes soil to saturation, triggering an abrupt switch from
    infiltration-excess to saturation-excess runoff.

    Tests whether STGNNSoilGate detects the approaching breakthrough via
    rising swvl2_sat_ratio before stage rises at gauges.

    MULTI-REALIZATION DESIGN: previously used one fixed set of injection
    parameters (rain_peak_mm=20.0, sat_start=0.55, sat_rate=0.015,
    excess_threshold=0.75) replayed across many different REAL historical
    base windows -- meaning every "window" varied only in antecedent
    real conditions, never in the synthetic event's own characteristics.
    A reviewer correctly noted this leaves conclusions vulnerable to the
    argument that results depend on the specific (arbitrarily chosen)
    perturbation parameters rather than the underlying mechanism being
    tested. Now draws n_realizations independent parameter sets from
    sample_s4_realization_params(), each applied to
    n_windows_per_realization real historical baselines -- producing
    genuine event-to-event diversity, not just baseline-condition
    diversity, while keeping total data volume comparable to the
    original single-realization n_windows=30 (default here:
    20 x 2 = 40 windows).
    """
    print("\nS4: SatBreakthrough (multi-realization)")
    out_dir = SCEN_DIR / "S4_SatBreakthrough"
    rng = np.random.default_rng(seed)

    T = X.shape[0]; N = X.shape[1]

    window_starts_pool = select_base_windows(
        X, y, bankfull, sat_min=0.45, sat_max=0.70,
        max_stage_frac=0.20, n_windows=n_realizations * n_windows_per_realization,
        search_from_frac=0.50)
    if not window_starts_pool:
        window_starts_pool = select_base_windows(
            X, y, bankfull, sat_min=0.45, sat_max=0.80,
            n_windows=n_realizations * n_windows_per_realization,
            search_from_frac=0.40)
    if not window_starts_pool:
        print("  [skip] S4: no base windows found meeting antecedent-condition criteria")
        return

    stage_range = (nd.set_index("node_idx")["p90_mAOD"]
                   - nd.set_index("node_idx")["gauge_datum_mOSGM15"])

    all_windows: list[tuple[int, int, dict]] = []   # (t0, realization_id, params)
    realization_params_list: list[dict] = []
    pool_idx = 0
    for r in range(n_realizations):
        params = sample_s4_realization_params(rng)
        n_avail = len(window_starts_pool) - pool_idx
        n_take = min(n_windows_per_realization, n_avail)
        if n_take <= 0:
            break
        realization_params_list.append(params)
        for _ in range(n_take):
            all_windows.append((window_starts_pool[pool_idx], r, params))
            pool_idx += 1

    if not all_windows:
        print("  [skip] S4: no windows available after realization allocation")
        return

    n_total = len(all_windows)
    T_s   = T_WINDOW * n_total
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    realization_id_per_window: list[int] = []

    for i, (t0, real_id, params) in enumerate(all_windows):
        sl  = slice(i * T_WINDOW, (i+1) * T_WINDOW)
        X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
        m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

        rain_peak_mm     = params["rain_peak_mm"]
        sat_start        = params["sat_start"]
        sat_rate         = params["sat_rate"]
        excess_threshold = params["excess_threshold"]
        sat_breakthrough_step = params["sat_breakthrough_step"]
        sat_cap = 0.40   # matches the np.clip below; not varied per realization

        realization_id_per_window.append(real_id)

        X_w[T_IN:, :, F_RAIN] += rain_peak_mm * np.exp(
            -np.arange(T_WINDOW - T_IN)[:, None] * 0.05)

        X_w[:T_IN, :, F_SW2_SAT] = sat_start
        X_w[:T_IN, :, F_SW1_SAT] = 0.50
        X_w[:T_IN, :, F_SW2_RAW] = sat_start * 0.472
        X_w[:T_IN, :, F_SW1_RAW] = 0.50 * 0.472
        for t in range(T_IN, T_WINDOW):
            steps_of_rain = t - T_IN
            sat_increase  = min(sat_cap, steps_of_rain * sat_rate)
            X_w[t, :, F_SW2_SAT] = np.clip(sat_start + sat_increase, 0.0, 1.0)
            X_w[t, :, F_SW1_SAT] = np.clip(0.50 + sat_increase * 0.8, 0.0, 1.0)
            X_w[t, :, F_SW2_RAW] = X_w[t, :, F_SW2_SAT] * 0.472
            X_w[t, :, F_SW1_RAW] = X_w[t, :, F_SW1_SAT] * 0.472

        stage_range_arr = stage_range.reindex(range(N)).values.astype(np.float32)
        stage_range_safe = np.where(stage_range_arr > 0, stage_range_arr, np.inf)
        sat_bt_clamped = min(sat_breakthrough_step, T_WINDOW)
        for t in range(T_IN, sat_bt_clamped):
            slow_rise = 0.01 * (t - T_IN)
            y_w[t, :] += slow_rise
            X_w[t, :, F_STAGE] += slow_rise
            X_w[t, :, F_NORM] += slow_rise / stage_range_safe

        if sat_bt_clamped < T_WINDOW:
            for n_idx in range(N):
                ref = str(int(nd.iloc[n_idx]["ref"]))
                if ref not in uh: continue
                p   = uh[ref]
                sat_at_breakthrough = float(
                    X_w[sat_bt_clamped, n_idx, F_SW2_SAT])
                excess_factor = max(0.0, sat_at_breakthrough - excess_threshold) * 4.0
                if excess_factor <= 0: continue
                uh_arr = triangular_uh(
                    p["peak_rate_m_per_mm"] * 2.0,
                    p["tc_hr"],
                    T_WINDOW - sat_bt_clamped)
                delta = uh_arr * rain_peak_mm * 6 * excess_factor
                t_end = sat_bt_clamped + len(delta)
                t_end = min(t_end, T_WINDOW)
                y_w[sat_bt_clamped:t_end, n_idx] += delta[:t_end-sat_bt_clamped]
                X_w[sat_bt_clamped:t_end, n_idx, F_STAGE] += \
                    delta[:t_end-sat_bt_clamped]
                sr = float(stage_range.loc[n_idx])
                if sr > 0:
                    X_w[sat_bt_clamped:t_end, n_idx, F_NORM] += \
                        delta[:t_end-sat_bt_clamped] / sr

        X_syn[sl] = X_w
        y_syn[sl] = y_w
        m_syn[sl] = m_w

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S4")

    meta = {
        "name": "S4_SatBreakthrough",
        "description": (
            "Moderate-saturation catchment receives heavy rainfall, "
            "crossing the saturation threshold mid-forecast. Abrupt "
            "stage acceleration tests anticipatory soil gate. Multi-"
            "realization: excess_threshold ~ U(0.68,0.80) grounded in "
            "Meissl et al. (2023)/Western & Grayson (1998)/Penna et al. "
            "(2011); rain_peak_mm, sat_start, sat_rate use documented "
            "illustrative ranges (see sample_s4_realization_params). "
            "Unlike S1/S3/S5/S6, timing (sat_breakthrough_step) VARIES "
            "per realization along with magnitude -- deliberately, since "
            "a fixed, memorisable breakthrough timing would understate "
            "the anticipatory-vs-reactive gate comparison this scenario "
            "exists to test."),
        "n_windows":                 n_total,
        "n_windows_per_realization": n_windows_per_realization,
        "n_realizations":            n_realizations,
        "T_per_window":              T_WINDOW,
        "T_total":                   T_s,
        "realizations":              [dict(id=i, **p) for i, p in enumerate(realization_params_list)],
        "window_realization_id":     realization_id_per_window,
    }
    save_scenario("S4_SatBreakthrough", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S5 — Spatial rainfall gradient (frontal band)
# ══════════════════════════════════════════════════════════════════════════════

def generate_s5_spatial_gradient(X, y, mask, nd, ed, uh, lags, bankfull,
                                  n_windows: int = 40,
                                  n_realizations: int = 20,
                                  seed: int = 0) -> None:
    """
    A frontal system oriented SW–NE delivers heavy rainfall to western headwaters
    but light rainfall to eastern Cork city gauges. The flood signal must propagate
    through the river network from west to east.

    Tests whether graph routing correctly anticipates the arriving flood at
    Cork city before it appears in the local rainfall record.

    n_realizations: independent draws of gradient_max from
    sample_s5_params() -- Zhu, Wright & Yu (2018, WRR) support a range
    of spatial rainfall heterogeneity intensities, not one fixed 2.5x
    ratio. Topology (source_idx/downstream_idx/source_chains) is
    computed ONCE using a fixed reference gradient rather than
    recomputed per realization: the river network's structural
    reachability is a graph property that doesn't depend on rainfall
    intensity, only the injected magnitude should vary.
    """
    print("\nS5: SpatialGradient (multi-realization)")
    out_dir = SCEN_DIR / "S5_SpatialGradient"
    rng = np.random.default_rng(seed)

    T = X.shape[0]; N = X.shape[1]

    window_starts = select_base_windows(
        X, y, bankfull, sat_min=0.55, sat_max=0.92, n_windows=n_windows,
        search_from_frac=0.70)

    if not window_starts:
        print("  [skip] No valid base windows found for S5")
        return

    easting = nd["easting_itm"].values.astype(np.float64)
    e_min, e_max = easting.min(), easting.max()
    spatial_weight = 1.0 - (easting - e_min) / (e_max - e_min + 1)   # [0,1] per node, fixed

    def _gradient_for(gradient_max: float) -> np.ndarray:
        g = 1.0 + (gradient_max - 1.0) * spatial_weight
        return g.astype(np.float32)

    # Reference topology at gradient_max=2.5 (the original single-realization
    # value, near the middle of S5_GRADIENT_MAX_RANGE) -- fixed for all
    # realizations, see docstring above.
    ref_gradient = _gradient_for(2.5)
    source_idx = [n for n in range(N) if ref_gradient[n] >= 1.3]
    source_chains: dict[int, list[tuple[int, int]]] = {}
    reachable = set()
    for src in source_idx:
        edges_order = downstream_bfs_edges(src, ed)
        source_chains[src] = edges_order
        reachable.update(dst for _s, dst in edges_order)

    downstream_candidates = nd[(nd.is_reservoir == 0)
                               & (nd.node_idx.isin(reachable))
                               & (~nd.node_idx.isin(source_idx))].copy()
    downstream_candidates = downstream_candidates.sort_values("log_catchment_area_km2", ascending=False)
    downstream_idx = downstream_candidates.head(5)["node_idx"].tolist()
    if not downstream_idx:
        print("  [skip] No downstream nodes reachable from any high-gradient "
              "source node — check the gradient threshold or graph topology.")
        return
    if len(downstream_idx) < 5:
        print(f"  [warn] Only {len(downstream_idx)} downstream nodes reachable "
              f"from source_idx (wanted 5): {downstream_idx}")

    stage_range = (nd.set_index("node_idx")["p90_mAOD"]
                   - nd.set_index("node_idx")["gauge_datum_mOSGM15"])

    T_s   = T_WINDOW * len(window_starts) * n_realizations
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    realizations_meta = []
    window_realization_id = []
    row = 0

    for real_id in range(n_realizations):
        params = sample_s5_params(rng)
        gradient_max = params["gradient_max"]
        gradient = _gradient_for(gradient_max)
        realizations_meta.append({
            "id": real_id, **params,
            "gradient_range": [float(gradient.min()), float(gradient.max())],
        })

        for t0 in window_starts:
            sl  = slice(row * T_WINDOW, (row+1) * T_WINDOW)
            window_realization_id.append(real_id)

            X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
            y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
            m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

            X_w[T_IN:, :, F_RAIN] *= gradient[None, :]

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

            for src in source_idx:
                if y_w[:, src].max() <= y_w[:T_IN, src].max():
                    continue   # no signal to route from this source
                base = float(np.mean(y_w[:T_IN, src]))
                propagate_downstream_chain(
                    y_w, X_w, src, source_chains[src], lags,
                    baseline=base, stage_range=stage_range)

            X_syn[sl] = X_w
            y_syn[sl] = y_w
            m_syn[sl] = m_w
            row += 1

    physical_consistency_check(X_syn, y_syn, nd, bankfull, "S5")

    meta = {
        "name": "S5_SpatialGradient",
        "description": (
            "SW-NE frontal band delivers heavier rainfall to western "
            "headwaters than eastern Cork city gauges. Tests whether "
            "graph models correctly route the gradient. Multi-"
            "realization: gradient_max ~ U(1.8,3.0), Zhu, Wright & Yu "
            "(2018, WRR); source/downstream topology fixed at "
            "gradient_max=2.5 reference (structural, not intensity-"
            "dependent)."),
        "n_windows":    row,
        "n_windows_per_realization": len(window_starts),
        "n_realizations": n_realizations,
        "T_per_window": T_WINDOW,
        "T_total":      T_s,
        "downstream_nodes": downstream_idx,
        "realizations":          realizations_meta,
        "window_realization_id": window_realization_id,
    }
    save_scenario("S5_SpatialGradient", out_dir, X_syn, y_syn, m_syn, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S6 — Bridge/culvert channel blockage: backwater upstream, suppressed flow
#      downstream. Explicit architectural-limitation diagnostic.
# ══════════════════════════════════════════════════════════════════════════════

def generate_s6_channel_blockage(X, y, mask, nd, ed, uh, lags, bankfull,
                                  n_windows: int = 30,
                                  n_realizations: int = 20,
                                  seed: int = 0) -> None:
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

    n_realizations: independent draws of blockage severity (and its
    two DERIVED caps, backwater_cap_m/suppression_cap_m) from
    sample_s6_params(). Severity is anchored at McDermott & Quinn
    (2023)'s ~50% documented flow-capacity reduction; caps are derived
    from one shared severity value rather than sampled independently,
    since a real blockage's upstream rise and downstream suppression
    share the same physical cause. rise_rate/suppression_rate (the
    SPEED of approach to those caps, distinct from final magnitude)
    stay fixed across realizations, as does t_block.
    """
    print("\nS6: ChannelBlockage (multi-realization architectural-limitation diagnostic)")
    out_dir = SCEN_DIR / "S6_ChannelBlockage"
    rng = np.random.default_rng(seed)

    T = X.shape[0]; N = X.shape[1]

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

    bridge_mask = nd["name"].str.contains(
        "Bridge|Br$|Br ", case=False, regex=True, na=False)
    bridge_candidates = nd.loc[bridge_mask, "node_idx"].tolist()

    block_dst = None
    block_src = None
    block_next = None

    for cand_dst in bridge_candidates:
        upstream_edges = ed[ed["dst_idx"] == cand_dst]
        downstream_edges = ed[ed["src_idx"] == cand_dst]
        if not upstream_edges.empty:
            candidate_src = int(upstream_edges.iloc[0]["src_idx"])
            downstream_edges = downstream_edges[downstream_edges["dst_idx"] != candidate_src]
        if upstream_edges.empty or downstream_edges.empty:
            continue
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

    t_block           = T_IN + 4    # blockage occurs 1 hr into the forecast, fixed
    rise_rate         = 0.05        # m per timestep, fixed (speed, not magnitude)
    suppression_rate  = 0.02        # m per timestep, fixed (speed, not magnitude)

    stage_range = (nd.set_index("node_idx")["p90_mAOD"]
                   - nd.set_index("node_idx")["gauge_datum_mOSGM15"])

    T_s   = T_WINDOW * len(window_starts) * n_realizations
    X_syn = np.zeros((T_s, N, X.shape[2]), dtype=np.float32)
    y_syn = np.zeros((T_s, N),             dtype=np.float32)
    m_syn = np.zeros((T_s, N),             dtype=np.float32)

    realizations_meta = []
    window_realization_id = []
    row = 0

    for real_id in range(n_realizations):
        params = sample_s6_params(rng)
        backwater_cap   = params["backwater_cap_m"]
        suppression_cap = params["suppression_cap_m"]
        realizations_meta.append({"id": real_id, **params})

        for t0 in window_starts:
            sl  = slice(row * T_WINDOW, (row+1) * T_WINDOW)
            window_realization_id.append(real_id)

            X_w = np.array(X[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
            y_w = np.array(y[t0 : t0 + T_WINDOW]).copy().astype(np.float32)
            m_w = np.array(mask[t0 : t0 + T_WINDOW]).copy().astype(np.float32)

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

            for t in range(t_block, T_WINDOW):
                reduction = min(suppression_rate * (t - t_block), suppression_cap)
                y_w[t, block_next] -= reduction
                X_w[t, block_next, F_STAGE] -= reduction

            X_syn[sl] = X_w
            y_syn[sl] = y_w
            m_syn[sl] = m_w
            row += 1

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
            "every model, not a failure to be optimised away. Multi-"
            "realization: severity ~ U(0.30,0.75), anchored at McDermott "
            "& Quinn (2023)'s ~50% documented flow-capacity reduction; "
            "backwater_cap_m/suppression_cap_m derived from one shared "
            "severity value, not sampled independently."),
        "n_windows":      row,
        "n_windows_per_realization": len(window_starts),
        "n_realizations": n_realizations,
        "T_per_window":   T_WINDOW,
        "T_total":        T_s,
        "blockage_node":  {"idx": block_dst,  "name": bridge_name},
        "upstream_node":  {"idx": block_src,  "name": nd.loc[nd.node_idx==block_src,  "name"].values[0]},
        "downstream_node":{"idx": block_next, "name": nd.loc[nd.node_idx==block_next, "name"].values[0]},
        "t_blockage_step": t_block,
        "backwater_rate_m_per_step": rise_rate,
        "suppression_rate_m_per_step": suppression_rate,
        "realizations":          realizations_meta,
        "window_realization_id": window_realization_id,
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
