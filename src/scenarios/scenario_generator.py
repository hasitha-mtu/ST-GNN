"""
scenario_generator.py  —  Synthetic stress-scenario generation (revised)
=========================================================================
Generates S1-S6 by superimposing controlled perturbations on test-period
base windows of the processed dataset. Output layout and the metadata keys
read by scenario_evaluator.py are unchanged; new keys are additive.

REVISION AUDIT — implementation errors corrected in this version
---------------------------------------------------------------
G1  X/y one-step misalignment. y[t] = a(t+1) but every scenario added the
    same delta to X[t, stage] and y[t], so targets lagged inputs by one step
    inside every injected event. All stage perturbations are now built as a
    single delta series D[t] (t = 0..T_WINDOW) and applied by
    `apply_stage_delta`: X[t] += D[t], y[t] += D[t+1]. tests/test_generator.py
    asserts y[t] == X[t+1, stage] for every injected window.
G2  Routed REAL noise. Routing used (y - mean(y_input)) at the source, so the
    real historical fluctuation of the source gauge was routed and added
    downstream on top of the downstream gauge's own real record. Only the
    injected delta is routed now.
G3  Inconsistent feature updates. normalised_stage (read by every
    elevation gate) and dh/dt were not updated in S3/S5 and only partly in
    the other scenarios. `apply_stage_delta` updates stage anomaly,
    normalised stage and dh/dt together.
G4  Double time-to-peak. S1/S5 started the triangular kernel at
    onset + Tp, so the peak occurred at ~onset + 2 Tp. The response is now a
    convolution of excess rain with the kernel, starting when rain starts
    (scenarios/hydro.py), using Tb = 2.67 Tp.
G5  Uphill routing. BFS followed the two uphill reverse-duplicate edges
    (Glen Park->Ballincolly, Blackpool->Glennamought), so S1/S5 pulses were
    routed upstream into other injection nodes. River edges are filtered by
    utils/graph_checks.py and routed in topological order.
G6  S2 leakage and masking. Failed gauges still exposed their TRUE live
    stage through normalised_stage, their (unaffected) rain-gauge rainfall
    was zeroed, and their TARGETS were masked out of evaluation, although S2
    targets are real observations. Failure now freezes the last valid
    reading of every gauge-derived feature (stage anomaly, normalised stage,
    discharge; dh/dt = 0), as described in the manuscript; rainfall and soil
    moisture are untouched; the target mask is unchanged; a per-window
    failure mask and the hydraulically connected downstream set are saved.
G7  S3/S6 overwrote the base record (stage := base mean + delta) instead of
    superimposing. Both now superimpose (manuscript Section 3.8.1).
G8  S4 forcing/response inconsistency. 240-560 mm of injected rain produced
    a breakthrough response scaled by (sat - threshold) * 4 <= 0.08 at the
    breakthrough step, and a +0.01 m/step "slow rise" was REMOVED abruptly at
    breakthrough (a step drop of ~0.1 m). Rebuilt as a single rainfall series
    convolved with the response kernel, with a runoff factor that switches
    from F_PRE to 1.0 when the synthetic saturation crosses the threshold.
    The storm has a finite e-folding decay (configurable).
G9  S4 base windows were drawn from the VALIDATION period (checkpoint
    selection data), contradicting manuscript Section 3.8.1. All scenarios
    now draw base windows from the test period only. S4 overwrites soil
    moisture synthetically, so its base windows no longer need a soil-
    moisture criterion.
G10 S5 response used the TOTAL amplified rainfall (double counting the real
    rain whose response is already in the record) and produced no event at
    all in calm (dry) base windows. A frontal storm with the sampled west-
    east gradient is injected, and only the rainfall increment drives the
    response.
G11 S6 imposed the full backwater rise at the upstream gauge 3.3 km away.
    The upstream rise now decays with distance using a Samuels-type backwater
    length L = 0.7 D / S0 (option --s6-upstream-decay none restores the old
    assumption, for sensitivity).
G12 Overlapping base windows (stride T_IN=32 with 104-step windows) are
    replaced by non-overlapping windows; the data-validity check uses the
    real-observation mask (the old NaN check could never fail after
    zero-filling).
G13 Guard 1 clipped y (not X) at 3x the historical 99th percentile, which
    broke X/y consistency, clipped REAL S2 flood targets and truncated the
    designed S3/S6 magnitudes. It now only reports exceedances by default.
G14 `--n-windows` crashed S4 (unexpected keyword). Fixed.

New, for the reviewer's sensitivity experiment (Section 3.1): routing
attenuation and lag scale are explicit parameters (--attenuation,
--lag-scale) and --out-root writes the scenario set elsewhere.
Observability (reviewer, Section 3.8): every window records the first step
at which the perturbation is visible in model INPUTS and the first step at
which it changes the TARGET; scenario_evaluator.py splits metrics by
whether the forecast was issued before or after the input onset.

Usage
-----
    python src/scenarios/scenario_generator.py --all
    python src/scenarios/scenario_generator.py --scenario S3 --attenuation 0.65 \
        --lag-scale 1.25 --out-root dataset/scenarios_sens/a065_l125
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
SCEN_DIR  = BASE_DIR / "dataset/scenarios"

for _p in [BASE_DIR, BASE_DIR / "src"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scenarios.hydro import (STEP_MIN, stage_response, route_delta,
                             samuels_backwater_length_m)
from utils.graph_checks import river_edges, topological_edges_from, downstream_closure
from utils.datum import stage_reference, NORM_CLIP

GENERATOR_VERSION = "2026-10-revision-audit-1"

# ── Feature indices in X.npy ─────────────────────────────────────────────────
F_STAGE, F_NORM, F_DHDT, F_DISC, F_RAIN = 0, 1, 2, 3, 4
F_SW1_RAW, F_SW1_SAT, F_SW1_AN = 5, 6, 7
F_SW2_RAW, F_SW2_SAT, F_SW2_AN = 8, 9, 10
SOIL_SAT_VWC = 0.472

T_IN      = 32
T_OUT_MAX = 48
T_WINDOW  = T_IN + T_OUT_MAX + 24      # 104 steps

# ── Severity ranges (unchanged from the manuscript) ─────────────────────────
S1_PULSE_PEAK_RANGE_MM    = (15.0, 45.0)
S3_PLATEAU_RISE_RANGE_M   = (0.5, 1.3)
S3_RAMP_RATE_RANGE_M_STEP = (0.03, 0.08)
S5_GRADIENT_MAX_RANGE     = (1.8, 3.0)
S6_SEVERITY_RANGE         = (0.30, 0.75)
S2_FAILURE_FRACTIONS      = [0.05, 0.10, 0.20, 0.30, 0.40]
S2_DURATION_RANGE_STEPS   = (8, 90)


@dataclass
class GeneratorConfig:
    attenuation: float = 0.75          # per-edge amplitude factor (Eq. 10)
    lag_scale: float = 1.0             # multiplies every routing lag
    tc_scale: float = 1.0              # multiplies every node Tc (response timing)
    rate_scale: float = 1.0            # multiplies every peak response rate
    base_factor_on: str = "tp"         # SCS Tb = 2.67 Tp ('tc' = legacy)
    s2_failure_mode: str = "freeze"    # 'freeze' (manuscript) or 'zero'
    s4_storm_decay_per_step: float = 0.25   # e-folding 4 steps = 1 h
    s4_pre_threshold_runoff_factor: float = 0.3
    s5_storm_steps: int = 24           # 6-h frontal hyetograph
    s5_storm_depth_range_mm: tuple = (20.0, 40.0)   # at gradient factor 1
    s6_upstream_decay: str = "samuels" # or 'none' (legacy)
    window_stride: int = T_WINDOW      # non-overlapping base windows
    min_valid_frac: float = 0.95       # real-observation share in inputs
    guard_mode: str = "report"         # 'report' or 'clip' (legacy-like)
    seed: int = 0
    version: str = GENERATOR_VERSION


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectData:
    X: np.ndarray
    y: np.ndarray
    mask: np.ndarray            # target-aligned validity (valid_mask.npy)
    input_valid: np.ndarray     # observation-time validity
    nd: pd.DataFrame
    ed: pd.DataFrame            # river edges, reverse duplicates removed
    uh: dict
    lags: dict
    bankfull: np.ndarray
    test_start: int
    stage_low: np.ndarray = field(default=None)
    stage_range: np.ndarray = field(default=None)

    def __post_init__(self):
        self.stage_low, self.stage_range = stage_reference(self.nd)


def load_project_data() -> ProjectData:
    from utils.train_utils import get_split_boundary
    X    = np.load(PROC_DIR / "X.npy",          mmap_mode="r")
    y    = np.load(PROC_DIR / "y.npy",          mmap_mode="r")
    mask = np.load(PROC_DIR / "valid_mask.npy", mmap_mode="r")
    ivp  = PROC_DIR / "input_valid_mask.npy"
    if ivp.exists():
        iv = np.load(ivp, mmap_mode="r")
    else:
        print("  [warn] input_valid_mask.npy missing — rebuild the dataset; "
              "using a shifted valid_mask as approximation")
        iv = np.vstack([np.ones((1, mask.shape[1]), np.float32), np.asarray(mask[:-1])])
    nd = pd.read_csv(GRAPH_DIR / "nodes.csv")
    ed = river_edges(pd.read_csv(GRAPH_DIR / "edges.csv"))
    with open(GRAPH_DIR / "uh_params.json") as f:
        uh = json.load(f)["params"]
    with open(GRAPH_DIR / "routing_lags.json") as f:
        lags = json.load(f)["lags"]
    with open(GRAPH_DIR / "bankfull_thresholds.json") as f:
        bf = json.load(f)
    bf = bf.get("thresholds", bf)
    bankfull = np.array([float(bf.get(str(r), 0.5)) for r in nd["ref"]], np.float32)
    _, t2 = get_split_boundary(X.shape[0], 0.70, 0.15)
    print(f"  X:{X.shape}  test period starts at row {t2}")
    return ProjectData(X, y, mask, iv, nd, ed, uh, lags, bankfull, t2)


# ══════════════════════════════════════════════════════════════════════════════
# Core perturbation helpers (unit-tested)
# ══════════════════════════════════════════════════════════════════════════════

def apply_stage_delta(X_w: np.ndarray, y_w: np.ndarray, D: np.ndarray,
                      stage_range: np.ndarray) -> None:
    """Superimpose a stage perturbation consistently on inputs and targets.

    D has shape [T_WINDOW + 1, N]; D[t] is the perturbation of the stage
    at window step t (D[T_WINDOW] is needed for the last target row).
    Updates in place:
        X[t, :, stage anomaly]   += D[t]
        X[t, :, normalised stage] += D[t] / stage_range  (then clipped)
        X[t, :, dh/dt]           += 4 * (D[t] - D[t-1])   (m/h)
        y[t]                     += D[t + 1]
    """
    T = X_w.shape[0]
    assert D.shape == (T + 1, X_w.shape[1]), (D.shape, X_w.shape)
    X_w[:, :, F_STAGE] += D[:T]
    X_w[:, :, F_NORM] = np.clip(X_w[:, :, F_NORM] + D[:T] / stage_range[None, :],
                                *NORM_CLIP)
    dD = np.diff(np.vstack([np.zeros((1, D.shape[1])), D[:T]]), axis=0)
    X_w[:, :, F_DHDT] += 4.0 * dD
    y_w += D[1:T + 1]


def set_soil_saturation(X_w: np.ndarray, sat2: np.ndarray, sat1: np.ndarray) -> None:
    """Overwrite swvl1/2 saturation & raw VWC; shift anomalies by the same
    raw change so the three SM features stay mutually consistent (the
    original code left the anomaly features at their historical values)."""
    for f_raw, f_sat, f_an, sat in ((F_SW2_RAW, F_SW2_SAT, F_SW2_AN, sat2),
                                    (F_SW1_RAW, F_SW1_SAT, F_SW1_AN, sat1)):
        if X_w.shape[2] <= f_an:
            return
        new_raw = np.clip(sat, 0, 1) * SOIL_SAT_VWC
        X_w[:, :, f_an] += new_raw - X_w[:, :, f_raw]
        X_w[:, :, f_raw] = new_raw
        X_w[:, :, f_sat] = np.clip(sat, 0, 1)


def edge_lag(lags: dict, s: int, d: int, cfg: GeneratorConfig) -> int:
    base = int(lags.get(f"{s}_{d}", 2))
    return max(1, int(round(base * cfg.lag_scale)))


def route_from_sources(D: np.ndarray, sources: list[int], ed: pd.DataFrame,
                       lags: dict, cfg: GeneratorConfig,
                       chains: Optional[dict] = None) -> np.ndarray:
    """Route ONLY the injected deltas D[:, src] downstream (Eq. 10) and
    return the routed contribution (the caller adds it to D). Multiple
    sources superpose linearly; within one source's sub-network every node
    receives the sum of its incoming routed signals (topological order)."""
    routed = np.zeros_like(D)
    for src in sources:
        if not np.any(D[:, src]):
            continue
        chain = chains[src] if chains is not None else topological_edges_from(ed, src)
        sig = {src: D[:, src].copy()}
        acc = {}
        for u, v in chain:
            contrib = route_delta(sig[u], edge_lag(lags, u, v, cfg), cfg.attenuation)
            acc[v] = acc.get(v, 0.0) + contrib
            sig[v] = acc[v]            # complete once all in-edges processed
        for v, a in acc.items():
            routed[:, v] += a
    return routed


def node_response(rain_excess: np.ndarray, p: dict, cfg: GeneratorConfig) -> np.ndarray:
    return stage_response(rain_excess, float(p["peak_rate_m_per_mm"]) * cfg.rate_scale,
                          float(p["tc_hr"]) * cfg.tc_scale, cfg.base_factor_on)


def first_nonzero(a: np.ndarray, atol: float = 1e-9) -> int:
    """First row index with any |a| > atol, or -1."""
    idx = np.where(np.abs(a).reshape(a.shape[0], -1).max(axis=1) > atol)[0]
    return int(idx[0]) if len(idx) else -1


# ══════════════════════════════════════════════════════════════════════════════
# Base-window selection (test period only)
# ══════════════════════════════════════════════════════════════════════════════

def select_base_windows(data: ProjectData, cfg: GeneratorConfig,
                        sat_min: float = 0.0, sat_max: float = 1.0,
                        max_stage_frac: float = 0.40, n_windows: int = 50,
                        node_subset: Optional[list[int]] = None,
                        use_sat: bool = True) -> list[int]:
    X, y, iv, bankfull = data.X, data.y, data.input_valid, data.bankfull
    T = X.shape[0]
    check = node_subset if node_subset is not None else list(range(y.shape[1]))
    out, t = [], data.test_start
    while t <= T - T_WINDOW - 1 and len(out) < n_windows:
        ok = True
        if use_sat and X.shape[2] > F_SW2_SAT:
            sat = np.asarray(X[t:t + T_IN, :, F_SW2_SAT])
            # Real soil moisture present (the ERA5 failure mode was all-zero
            # features). The former extra `std > 0.01` condition sent windows
            # with spatially uniform saturation to the rainfall proxy.
            if sat.mean() > 0.05:
                ok = sat_min <= float(sat.mean()) <= sat_max
            else:
                proxy = np.clip(0.50 + float(np.mean(X[t:t + T_IN, :, F_RAIN])) * 0.27, 0.5, 0.99)
                ok = sat_min <= proxy <= sat_max
        if ok:
            yw = np.asarray(y[t:t + T_IN])
            for n in check:
                if np.nanmax(np.abs(yw[:, n])) > max_stage_frac * max(float(bankfull[n]), 0.30):
                    ok = False
                    break
        if ok:
            ok = float(np.asarray(iv[t:t + T_IN, check]).mean()) >= cfg.min_valid_frac
        if ok:
            out.append(t)
            t += cfg.window_stride
        else:
            t += 48
    return out


def _copy_window(data: ProjectData, t0: int):
    X_w = np.array(data.X[t0:t0 + T_WINDOW], dtype=np.float32)
    y_w = np.array(data.y[t0:t0 + T_WINDOW], dtype=np.float32)
    m_w = np.array(data.mask[t0:t0 + T_WINDOW], dtype=np.float32)
    return X_w, y_w, m_w


def physical_consistency_check(X_syn, y_syn, name: str, mode: str = "report") -> dict:
    """Plausibility guard on stage magnitudes.

    REVISION AUDIT (G13): the submitted guard silently clipped y_syn (but not
    the stage input) at 3x the historical 99th percentile. It thereby also
    clipped REAL S2 flood targets and truncated the designed S3 release /
    S6 backwater magnitudes, and broke X/y consistency. Default is now
    'report' (record the exceedance share, change nothing). 'clip' clips X
    stage and y together, for comparison only.
    """
    y_orig = np.load(PROC_DIR / "y.npy", mmap_mode="r")
    lim = 3 * float(np.nanpercentile(np.abs(np.asarray(y_orig[::7])), 99))
    frac_y = float(np.mean(np.abs(y_syn) > lim))
    frac_x = float(np.mean(np.abs(X_syn[:, :, F_STAGE]) > lim))
    if frac_y or frac_x:
        print(f"  [note] {name}: {frac_y:.3%} of targets exceed 3x the historical "
              f"99th percentile ({lim:.3f} m); guard mode = {mode}")
    if mode == "clip":
        y_syn[:] = np.clip(y_syn, -lim, lim)
        X_syn[:, :, F_STAGE] = np.clip(X_syn[:, :, F_STAGE], -lim, lim)
    X_syn[:, :, F_RAIN] = np.clip(X_syn[:, :, F_RAIN], 0, None)
    for fi in (F_SW1_SAT, F_SW2_SAT):
        if X_syn.shape[2] > fi:
            X_syn[:, :, fi] = np.clip(X_syn[:, :, fi], 0.0, 1.0)
    X_syn[:, :, F_NORM] = np.clip(X_syn[:, :, F_NORM], *NORM_CLIP)
    return {"clip_limit_m": lim, "guard_mode": mode,
            "exceed_frac_targets": frac_y, "exceed_frac_stage_inputs": frac_x,
            "clipped_frac_targets": frac_y if mode == "clip" else 0.0}


def save_scenario(out_dir: Path, X_syn, y_syn, m_syn, meta: dict,
                  extra: Optional[dict] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_synthetic.npy", X_syn.astype(np.float32))
    np.save(out_dir / "y_synthetic.npy", y_syn.astype(np.float32))
    np.save(out_dir / "mask_synthetic.npy", m_syn.astype(np.float32))
    for k, v in (extra or {}).items():
        np.save(out_dir / f"{k}.npy", v)
    with open(out_dir / "scenario_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    print(f"  Saved {meta['name']}: X{X_syn.shape}  windows={meta['n_windows']}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _alloc(n_win: int, N: int, F: int):
    T_s = T_WINDOW * n_win
    return (np.zeros((T_s, N, F), np.float32), np.zeros((T_s, N), np.float32),
            np.zeros((T_s, N), np.float32))


def _base_meta(name, desc, n_total, n_per_real, n_real, cfg, onsets_in, onsets_tg,
               window_starts, real_ids):
    return {"name": name, "description": desc, "n_windows": n_total,
            "n_windows_per_realization": n_per_real, "n_realizations": n_real,
            "T_per_window": T_WINDOW, "T_total": T_WINDOW * n_total,
            "T_in_reference": T_IN,
            "window_base_start_row": [int(t) for t in window_starts],
            "window_realization_id": [int(r) for r in real_ids],
            "window_input_onset_step": [int(o) for o in onsets_in],
            "window_target_onset_step": [int(o) for o in onsets_tg],
            "generator_config": asdict(cfg)}


def identify_headwater_nodes(nd: pd.DataFrame, n: int = 6) -> list[int]:
    v = nd[(nd.is_tidal == 0) & (nd.is_reservoir == 0)]
    return v.sort_values("log_catchment_area_km2").head(n)["node_idx"].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# S1 — Convective cell
# ══════════════════════════════════════════════════════════════════════════════

def generate_s1_convective_cell(data: ProjectData, cfg: GeneratorConfig, out_root: Path,
                                n_windows: int = 50, n_realizations: int = 20, **_):
    print("\nS1: ConvectiveCell")
    rng = np.random.default_rng(cfg.seed)
    nd, ed = data.nd, data.ed
    N, F = data.X.shape[1], data.X.shape[2]
    hw = identify_headwater_nodes(nd, 6)
    chains = {h: topological_edges_from(ed, h) for h in hw}
    reach = set().union(*[downstream_closure(ed, h) for h in hw])
    cand = nd[(nd.is_reservoir == 0) & nd.node_idx.isin(reach) & ~nd.node_idx.isin(hw)]
    downstream = cand.sort_values("log_catchment_area_km2", ascending=False).head(5)["node_idx"].tolist()
    starts = select_base_windows(data, cfg, 0.55, 0.90, n_windows=n_windows)
    if not starts:
        print("  [skip] no base windows"); return
    pulse_steps, pulse_at = 6, T_IN + 2
    X_s, y_s, m_s = _alloc(len(starts) * n_realizations, N, F)
    reals, rid, on_in, on_tg, base_rows = [], [], [], [], []
    row = 0
    for r in range(n_realizations):
        peak = float(rng.uniform(*S1_PULSE_PEAK_RANGE_MM))
        reals.append({"id": r, "pulse_peak_mm_per_15min": peak})
        half = pulse_steps // 2
        pulse = np.array([peak * t / half for t in range(half)] +
                         [peak * (pulse_steps - t) / half for t in range(half, pulse_steps)])
        for t0 in starts:
            X_w, y_w, m_w = _copy_window(data, t0)
            ramp = np.minimum(0.08, np.arange(T_WINDOW) / T_WINDOW * 0.10)[:, None]
            set_soil_saturation(X_w, np.repeat(0.72 + ramp, N, 1), np.repeat(0.68 + ramp, N, 1))
            D = np.zeros((T_WINDOW + 1, N))
            rain_add = np.zeros((T_WINDOW, N))
            for n in hw:
                p = data.uh.get(str(int(nd.loc[n, "ref"])))
                if p is None:
                    continue
                rain_add[pulse_at:pulse_at + pulse_steps, n] = pulse[:T_WINDOW - pulse_at]
                sat = float(np.mean(X_w[:T_IN, n, F_SW2_SAT])) if F > F_SW2_SAT else 1.0
                excess = np.zeros(T_WINDOW + 1)
                excess[:T_WINDOW] = rain_add[:, n] * sat
                D[:, n] += node_response(excess, p, cfg)
            D += route_from_sources(D, hw, ed, data.lags, cfg, chains)
            X_w[:, :, F_RAIN] += rain_add
            apply_stage_delta(X_w, y_w, D, data.stage_range)
            sl = slice(row * T_WINDOW, (row + 1) * T_WINDOW)
            X_s[sl], y_s[sl], m_s[sl] = X_w, y_w, m_w
            rid.append(r); base_rows.append(t0)
            on_in.append(pulse_at); on_tg.append(max(first_nonzero(D[1:]), -1))
            row += 1
    diag = physical_consistency_check(X_s, y_s, "S1", cfg.guard_mode)
    meta = _base_meta("S1_ConvectiveCell", "Convective pulse over 6 headwater gauges.",
                      row, len(starts), n_realizations, cfg, on_in, on_tg, base_rows, rid)
    meta.update({"headwater_nodes": hw, "downstream_nodes": downstream,
                 "pulse_duration_steps": pulse_steps, "pulse_at_step": pulse_at,
                 "realizations": reals, **diag})
    save_scenario(out_root / "S1_ConvectiveCell", X_s, y_s, m_s, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S2 — Gauge failure (real targets)
# ══════════════════════════════════════════════════════════════════════════════

def compute_node_criticality(nd: pd.DataFrame, ed: pd.DataFrame) -> list[int]:
    counts = [(n, len(downstream_closure(ed, n))) for n in range(len(nd))]
    counts.sort(key=lambda x: -x[1])
    return [n for n, _ in counts]


def apply_gauge_failure(X_w: np.ndarray, fail_nodes: list[int], t_fail: int,
                        t_recover: int, mode: str = "freeze") -> None:
    """Simulate telemetry loss of WATER-LEVEL gauges. Rainfall (separate rain
    gauges) and ERA5 soil moisture are not affected."""
    if not fail_nodes:
        return
    sl = slice(t_fail, t_recover)
    for f in (F_STAGE, F_NORM, F_DISC):
        if mode == "freeze":
            X_w[sl, fail_nodes, f] = X_w[t_fail - 1, fail_nodes, f][None, :]
        elif mode == "zero":
            X_w[sl, fail_nodes, f] = 0.0
        else:
            raise ValueError(mode)
    X_w[sl, fail_nodes, F_DHDT] = 0.0


def generate_s2_gauge_failure(data: ProjectData, cfg: GeneratorConfig, out_root: Path,
                              n_windows: int = 40, n_random_repeats: int = 4,
                              n_windows_per_realization: int = 2, **_):
    print("\nS2: GaugeFailure")
    rng = np.random.default_rng(cfg.seed)
    nd, ed = data.nd, data.ed
    T, N, F = data.X.shape
    flood = []
    for stride in (T_WINDOW, T_WINDOW // 2):
        t, flood = data.test_start, []
        while t <= T - T_WINDOW - 1 and len(flood) < n_windows:
            yw = np.asarray(data.y[t:t + T_WINDOW])
            if int(np.sum(np.nanmax(yw[T_IN:T_IN + 24], axis=0) > 0.5 * data.bankfull)) >= 3 \
                    and float(np.asarray(data.input_valid[t:t + T_IN]).mean()) >= cfg.min_valid_frac:
                flood.append(t); t += stride
            else:
                t += T_IN // 2
        if len(flood) >= 10:
            break
        print(f"  [warn] only {len(flood)} non-overlapping flood windows at stride {stride}")
    if not flood:
        print("  [skip] no flood windows"); return
    crit = compute_node_criticality(nd, ed)
    t_fail = T_IN + 4
    configs = []
    for frac in S2_FAILURE_FRACTIONS:
        k = max(1, round(frac * N))
        configs += [(k, frac, "critical", 1), (k, frac, "random", n_random_repeats)]
    reals, windows, pool = [], [], 0
    for k, frac, mode, rep in configs:
        for _ in range(rep):
            fn = crit[:k] if mode == "critical" else sorted(rng.choice(N, k, replace=False).tolist())
            dur = int(rng.integers(*S2_DURATION_RANGE_STEPS))
            conn = sorted(set().union(*[downstream_closure(ed, f) for f in fn]) - set(fn))
            rid = len(reals)
            reals.append({"id": rid, "failure_fraction": frac, "n_fail": k,
                          "selection_mode": mode, "fail_nodes": fn,
                          "connected_downstream_nodes": conn, "duration_steps": dur})
            for j in range(min(n_windows_per_realization, len(flood))):
                windows.append((flood[(pool + j) % len(flood)], rid))
            pool += n_windows_per_realization
    X_s, y_s, m_s = _alloc(len(windows), N, F)
    fail_mask = np.zeros((len(windows) * T_WINDOW, N), np.uint8)
    for i, (t0, rid) in enumerate(windows):
        X_w, y_w, m_w = _copy_window(data, t0)
        r = reals[rid]
        t_rec = min(t_fail + r["duration_steps"], T_WINDOW)
        apply_gauge_failure(X_w, r["fail_nodes"], t_fail, t_rec, cfg.s2_failure_mode)
        sl = slice(i * T_WINDOW, (i + 1) * T_WINDOW)
        X_s[sl], y_s[sl], m_s[sl] = X_w, y_w, m_w      # targets & mask untouched
        fail_mask[i * T_WINDOW + t_fail: i * T_WINDOW + t_rec, r["fail_nodes"]] = 1
    diag = physical_consistency_check(X_s, y_s, "S2", cfg.guard_mode)
    meta = _base_meta("S2_GaugeFailure",
                      "Telemetry loss of water-level gauges during real flood windows; "
                      f"failure mode '{cfg.s2_failure_mode}'; real targets retained.",
                      len(windows), n_windows_per_realization, len(reals), cfg,
                      [t_fail] * len(windows), [-1] * len(windows),
                      [w[0] for w in windows], [w[1] for w in windows])
    meta.update({"t_failure_step": t_fail, "failure_fractions": S2_FAILURE_FRACTIONS,
                 "failure_mode": cfg.s2_failure_mode, "realizations": reals, **diag})
    save_scenario(out_root / "S2_GaugeFailure", X_s, y_s, m_s, meta,
                  extra={"fail_mask_synthetic": fail_mask})


# ══════════════════════════════════════════════════════════════════════════════
# S3 — Inniscarra release
# ══════════════════════════════════════════════════════════════════════════════

def release_profile(T: int, t_rel: int, ramp_rate: float, plateau: float,
                    plateau_steps: int) -> np.ndarray:
    ramp_steps = int(round(plateau / ramp_rate))
    out = np.zeros(T)
    for t in range(t_rel, T):
        dt = t - t_rel
        if dt < ramp_steps:
            out[t] = ramp_rate * dt
        elif dt < ramp_steps + plateau_steps:
            out[t] = plateau
        else:
            out[t] = max(0.0, plateau - ramp_rate * (dt - ramp_steps - plateau_steps))
    return out


def generate_s3_inniscarra_release(data: ProjectData, cfg: GeneratorConfig, out_root: Path,
                                   n_windows: int = 30, n_realizations: int = 20, **_):
    print("\nS3: InniscarraRelease")
    rng = np.random.default_rng(cfg.seed)
    nd, ed = data.nd, data.ed
    N, F = data.X.shape[1], data.X.shape[2]
    idx = lambda ref: int(nd.index[nd.ref.astype(str) == ref][0]) if (nd.ref.astype(str) == ref).any() else None
    tail, head, carr = idx("19109"), idx("19094"), idx("19095")
    chain = topological_edges_from(ed, tail)
    downstream = sorted(downstream_closure(ed, tail))
    rel_nodes = [tail] + downstream
    starts = []
    for frac in (0.30, 0.40):
        starts = select_base_windows(data, cfg, 0.60, 0.92, frac, n_windows, rel_nodes)
        if starts:
            break
    if not starts:
        print("  [skip] no base windows"); return
    t_rel, plateau_steps = T_IN + 4, 20
    X_s, y_s, m_s = _alloc(len(starts) * n_realizations, N, F)
    reals, rid, on_in, on_tg, rows_ = [], [], [], [], []
    row = 0
    for r in range(n_realizations):
        rr = float(rng.uniform(*S3_RAMP_RATE_RANGE_M_STEP))
        pl = float(rng.uniform(*S3_PLATEAU_RISE_RANGE_M))
        reals.append({"id": r, "plateau_rise_m": pl, "ramp_rate_m_per_step": rr,
                      "ramp_steps": int(round(pl / rr))})
        prof = release_profile(T_WINDOW + 1, t_rel, rr, pl, plateau_steps)
        for t0 in starts:
            X_w, y_w, m_w = _copy_window(data, t0)
            D = np.zeros((T_WINDOW + 1, N))
            D[:, tail] = prof
            D += route_from_sources(D, [tail], ed, data.lags, cfg, {tail: chain})
            apply_stage_delta(X_w, y_w, D, data.stage_range)
            sl = slice(row * T_WINDOW, (row + 1) * T_WINDOW)
            X_s[sl], y_s[sl], m_s[sl] = X_w, y_w, m_w
            rid.append(r); rows_.append(t0)
            on_in.append(first_nonzero(D[:T_WINDOW])); on_tg.append(first_nonzero(D[1:]))
            row += 1
    diag = physical_consistency_check(X_s, y_s, "S3", cfg.guard_mode)
    meta = _base_meta("S3_InniscarraRelease",
                      "Release hydrograph superimposed at Inniscarra Tailrace and routed "
                      "downstream; headrace gauges untouched.", row, len(starts),
                      n_realizations, cfg, on_in, on_tg, rows_, rid)
    meta.update({"release_node": {"idx": tail, "ref": "19109", "name": "Inniscarra Tailrace"},
                 "decoupled_nodes": {"inniscarra_headrace": {"idx": head, "ref": "19094"},
                                     "carrigadrohid_headrace": {"idx": carr, "ref": "19095"}},
                 "downstream_nodes": downstream, "t_release_step": t_rel,
                 "plateau_steps": plateau_steps, "realizations": reals, **diag})
    save_scenario(out_root / "S3_InniscarraRelease", X_s, y_s, m_s, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S4 — Saturation breakthrough
# ══════════════════════════════════════════════════════════════════════════════

def sample_s4_realization_params(rng: np.random.Generator) -> dict:
    thr = float(rng.uniform(0.68, 0.80))
    s0 = float(rng.uniform(0.50, 0.65))
    rate = float(rng.uniform(0.010, 0.020))
    peak = float(rng.uniform(12.0, 28.0))
    return {"excess_threshold": thr, "sat_start": s0, "sat_rate": rate,
            "rain_peak_mm": peak,
            "sat_breakthrough_step": T_IN + math.ceil((thr - s0) / rate)}


def s4_series(params: dict, cfg: GeneratorConfig, T: int = T_WINDOW):
    """Synthetic saturation (swvl2, swvl1), rainfall and runoff factor."""
    t = np.arange(T)
    k = np.clip(t - T_IN, 0, None)
    inc = np.minimum(0.40, k * params["sat_rate"])
    sat2 = params["sat_start"] + inc
    sat1 = 0.50 + 0.8 * inc
    rain = np.where(t >= T_IN, params["rain_peak_mm"] *
                    np.exp(-cfg.s4_storm_decay_per_step * k), 0.0)
    factor = np.where(sat2 >= params["excess_threshold"], 1.0,
                      cfg.s4_pre_threshold_runoff_factor)
    return np.clip(sat2, 0, 1), np.clip(sat1, 0, 1), rain, factor


def generate_s4_sat_breakthrough(data: ProjectData, cfg: GeneratorConfig, out_root: Path,
                                 n_realizations: int = 20, n_windows_per_realization: int = 2,
                                 **_):
    print("\nS4: SatBreakthrough")
    rng = np.random.default_rng(cfg.seed)
    nd = data.nd
    N, F = data.X.shape[1], data.X.shape[2]
    pool = select_base_windows(data, cfg, max_stage_frac=0.20,
                               n_windows=n_realizations * n_windows_per_realization,
                               use_sat=False)
    if len(pool) < n_realizations * n_windows_per_realization:
        pool = select_base_windows(data, cfg, max_stage_frac=0.40,
                                   n_windows=n_realizations * n_windows_per_realization,
                                   use_sat=False)
    if not pool:
        print("  [skip] no base windows"); return
    windows, reals = [], []
    for r in range(n_realizations):
        pr = sample_s4_realization_params(rng)
        reals.append({"id": r, **pr})
        for j in range(n_windows_per_realization):
            windows.append((pool[(r * n_windows_per_realization + j) % len(pool)], r))
    X_s, y_s, m_s = _alloc(len(windows), N, F)
    on_in, on_tg = [], []
    for i, (t0, r) in enumerate(windows):
        pr = reals[r]
        X_w, y_w, m_w = _copy_window(data, t0)
        sat2, sat1, rain, factor = s4_series(pr, cfg)
        set_soil_saturation(X_w, np.repeat(sat2[:, None], N, 1), np.repeat(sat1[:, None], N, 1))
        X_w[:, :, F_RAIN] += rain[:, None]
        D = np.zeros((T_WINDOW + 1, N))
        excess = np.zeros(T_WINDOW + 1)
        excess[:T_WINDOW] = rain * factor
        for n in range(N):
            p = data.uh.get(str(int(nd.loc[n, "ref"])))
            if p is not None:
                D[:, n] = node_response(excess, p, cfg)
        apply_stage_delta(X_w, y_w, D, data.stage_range)
        sl = slice(i * T_WINDOW, (i + 1) * T_WINDOW)
        X_s[sl], y_s[sl], m_s[sl] = X_w, y_w, m_w
        on_in.append(T_IN); on_tg.append(first_nonzero(D[1:]))
    diag = physical_consistency_check(X_s, y_s, "S4", cfg.guard_mode)
    meta = _base_meta("S4_SatBreakthrough",
                      "Synthetic wetting with a runoff factor that switches at the "
                      "saturation threshold; rain and response from one series.",
                      len(windows), n_windows_per_realization, n_realizations, cfg,
                      on_in, on_tg, [w[0] for w in windows], [w[1] for w in windows])
    meta.update({"realizations": reals, **diag})
    save_scenario(out_root / "S4_SatBreakthrough", X_s, y_s, m_s, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S5 — Spatial rainfall gradient (frontal storm)
# ══════════════════════════════════════════════════════════════════════════════

def frontal_hyetograph(depth_mm: float, steps: int, T: int, start: int) -> np.ndarray:
    h = np.zeros(T)
    tri = np.concatenate([np.arange(1, steps // 2 + 1), np.arange(steps - steps // 2, 0, -1)]).astype(float)
    tri *= depth_mm / tri.sum()
    h[start:start + steps] = tri[:max(0, min(steps, T - start))]
    return h


def generate_s5_spatial_gradient(data: ProjectData, cfg: GeneratorConfig, out_root: Path,
                                 n_windows: int = 40, n_realizations: int = 20, **_):
    print("\nS5: SpatialGradient")
    rng = np.random.default_rng(cfg.seed)
    nd, ed = data.nd, data.ed
    N, F = data.X.shape[1], data.X.shape[2]
    e = nd["easting_itm"].to_numpy(float)
    w = 1.0 - (e - e.min()) / (e.max() - e.min() + 1)
    grad = lambda gmax: 1.0 + (gmax - 1.0) * w
    eligible_ever = set(np.where(grad(S5_GRADIENT_MAX_RANGE[1]) > 1.05)[0].tolist())
    src_ref = [n for n in range(N) if grad(2.5)[n] >= 1.3]
    chains = {s: topological_edges_from(ed, s) for s in eligible_ever}
    reach = set().union(*[downstream_closure(ed, s) for s in src_ref]) if src_ref else set()
    cand = nd[(nd.is_reservoir == 0) & nd.node_idx.isin(reach) & ~nd.node_idx.isin(eligible_ever)]
    downstream = cand.sort_values("log_catchment_area_km2", ascending=False).head(5)["node_idx"].tolist()
    starts = select_base_windows(data, cfg, 0.55, 0.92, n_windows=n_windows)
    if not starts:
        print("  [skip] no base windows"); return
    X_s, y_s, m_s = _alloc(len(starts) * n_realizations, N, F)
    reals, rid, on_in, on_tg, rows_ = [], [], [], [], []
    row = 0
    for r in range(n_realizations):
        gmax = float(rng.uniform(*S5_GRADIENT_MAX_RANGE))
        depth = float(rng.uniform(*cfg.s5_storm_depth_range_mm))
        g = grad(gmax)
        elig = [n for n in range(N) if g[n] > 1.05]
        reals.append({"id": r, "gradient_max": gmax, "storm_depth_mm_at_g1": depth,
                      "gradient_range": [float(g.min()), float(g.max())],
                      "injected_nodes": elig})
        storm = frontal_hyetograph(depth, cfg.s5_storm_steps, T_WINDOW, T_IN)
        for t0 in starts:
            X_w, y_w, m_w = _copy_window(data, t0)
            real_rain = X_w[:, :, F_RAIN].copy()
            rain_add = np.zeros((T_WINDOW, N))
            rain_add[T_IN:] = (g[None, :] - 1.0) * real_rain[T_IN:]
            for n in elig:
                rain_add[:, n] += storm * g[n]
            D = np.zeros((T_WINDOW + 1, N))
            for n in elig:
                p = data.uh.get(str(int(nd.loc[n, "ref"])))
                if p is None:
                    continue
                sat = float(np.mean(X_w[:T_IN, n, F_SW2_SAT])) if F > F_SW2_SAT else 1.0
                ex = np.zeros(T_WINDOW + 1)
                ex[:T_WINDOW] = rain_add[:, n] * sat
                D[:, n] = node_response(ex, p, cfg)
            D += route_from_sources(D, elig, ed, data.lags, cfg, chains)
            X_w[:, :, F_RAIN] += rain_add
            apply_stage_delta(X_w, y_w, D, data.stage_range)
            sl = slice(row * T_WINDOW, (row + 1) * T_WINDOW)
            X_s[sl], y_s[sl], m_s[sl] = X_w, y_w, m_w
            rid.append(r); rows_.append(t0)
            on_in.append(T_IN); on_tg.append(first_nonzero(D[1:]))
            row += 1
    diag = physical_consistency_check(X_s, y_s, "S5", cfg.guard_mode)
    meta = _base_meta("S5_SpatialGradient",
                      "Frontal storm with west-east gradient; response driven by the "
                      "rainfall increment only.", row, len(starts), n_realizations, cfg,
                      on_in, on_tg, rows_, rid)
    meta.update({"downstream_nodes": downstream, "realizations": reals, **diag})
    save_scenario(out_root / "S5_SpatialGradient", X_s, y_s, m_s, meta)


# ══════════════════════════════════════════════════════════════════════════════
# S6 — Channel blockage
# ══════════════════════════════════════════════════════════════════════════════

def pick_blockage_site(nd: pd.DataFrame, ed: pd.DataFrame):
    bridges = nd.loc[nd["name"].str.contains(r"Bridge|Br$|Br ", case=False, regex=True),
                     "node_idx"].tolist()
    for b in bridges:
        up = ed[ed.dst_idx == b].sort_values("river_dist_km")
        dn = ed[ed.src_idx == b]
        if up.empty or dn.empty:
            continue
        return int(b), int(up.iloc[0].src_idx), int(dn.iloc[0].dst_idx), up.iloc[0]
    return None


def generate_s6_channel_blockage(data: ProjectData, cfg: GeneratorConfig, out_root: Path,
                                 n_windows: int = 30, n_realizations: int = 20, **_):
    print("\nS6: ChannelBlockage")
    rng = np.random.default_rng(cfg.seed)
    nd, ed = data.nd, data.ed
    N, F = data.X.shape[1], data.X.shape[2]
    site = pick_blockage_site(nd, ed)
    if site is None:
        print("  [skip] no blockage site"); return
    b, up, dn, up_edge = site
    from scenarios.unit_hydrograph import water_surface_level
    ws = water_surface_level(nd)
    L_up = float(up_edge.river_dist_km) * 1000.0
    S0 = max((ws[up] - ws[b]) / L_up, 1e-5)
    depth_up = float(data.stage_range[up])
    if cfg.s6_upstream_decay == "samuels":
        Lb = samuels_backwater_length_m(depth_up, S0)
        up_factor = float(np.exp(-L_up / Lb))
    else:
        Lb, up_factor = float("inf"), 1.0
    print(f"  Blockage at {nd.loc[b, 'name']}; upstream {nd.loc[up, 'name']} "
          f"({L_up/1000:.2f} km, factor {up_factor:.3g}); downstream {nd.loc[dn, 'name']}")
    starts = []
    for frac in (0.30, 0.40):
        starts = select_base_windows(data, cfg, 0.60, 0.92, frac, n_windows)
        if starts:
            break
    if not starts:
        print("  [skip] no base windows"); return
    t_blk, rise, supp = T_IN + 4, 0.05, 0.02
    X_s, y_s, m_s = _alloc(len(starts) * n_realizations, N, F)
    reals, rid, on_in, on_tg, rows_ = [], [], [], [], []
    row = 0
    dt = np.clip(np.arange(T_WINDOW + 1) - t_blk, 0, None)
    for r in range(n_realizations):
        sev = float(rng.uniform(*S6_SEVERITY_RANGE))
        bw_cap, sup_cap = round(0.8 + sev * 1.4, 4), round(0.1 + sev * 0.35, 4)
        reals.append({"id": r, "severity": sev, "backwater_cap_m": bw_cap,
                      "suppression_cap_m": sup_cap, "upstream_factor": up_factor})
        bw = np.minimum(rise * dt, bw_cap)
        sp = -np.minimum(supp * dt, sup_cap)
        for t0 in starts:
            X_w, y_w, m_w = _copy_window(data, t0)
            D = np.zeros((T_WINDOW + 1, N))
            D[:, b] = bw
            D[:, up] = bw * up_factor
            D[:, dn] = sp
            apply_stage_delta(X_w, y_w, D, data.stage_range)
            sl = slice(row * T_WINDOW, (row + 1) * T_WINDOW)
            X_s[sl], y_s[sl], m_s[sl] = X_w, y_w, m_w
            rid.append(r); rows_.append(t0)
            on_in.append(first_nonzero(D[:T_WINDOW])); on_tg.append(first_nonzero(D[1:]))
            row += 1
    diag = physical_consistency_check(X_s, y_s, "S6", cfg.guard_mode)
    meta = _base_meta("S6_ChannelBlockage",
                      "Blockage at a bridge gauge: superimposed backwater rise (decaying "
                      "upstream) and downstream suppression.", row, len(starts),
                      n_realizations, cfg, on_in, on_tg, rows_, rid)
    meta.update({"blockage_node": {"idx": b, "name": nd.loc[b, "name"]},
                 "upstream_node": {"idx": up, "name": nd.loc[up, "name"]},
                 "downstream_node": {"idx": dn, "name": nd.loc[dn, "name"]},
                 "t_blockage_step": t_blk, "backwater_rate_m_per_step": rise,
                 "suppression_rate_m_per_step": supp,
                 "upstream_backwater_length_m": Lb, "upstream_distance_m": L_up,
                 "realizations": reals, **diag})
    save_scenario(out_root / "S6_ChannelBlockage", X_s, y_s, m_s, meta)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_MAP = {
    "S1": generate_s1_convective_cell, "S2": generate_s2_gauge_failure,
    "S3": generate_s3_inniscarra_release, "S4": generate_s4_sat_breakthrough,
    "S5": generate_s5_spatial_gradient, "S6": generate_s6_channel_blockage,
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic flash flood scenarios")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scenario", nargs="+", choices=list(SCENARIO_MAP))
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--attenuation", type=float, default=0.75)
    ap.add_argument("--lag-scale", type=float, default=1.0)
    ap.add_argument("--tc-scale", type=float, default=1.0)
    ap.add_argument("--rate-scale", type=float, default=1.0)
    ap.add_argument("--s4-runoff-factor", type=float, default=0.3)
    ap.add_argument("--s2-failure-mode", choices=["freeze", "zero"], default="freeze")
    ap.add_argument("--s6-upstream-decay", choices=["samuels", "none"], default="samuels")
    ap.add_argument("--legacy-tb", action="store_true", help="Tb = 2.67 Tc (old)")
    ap.add_argument("--guard-mode", choices=["report", "clip"], default="report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=SCEN_DIR)
    a = ap.parse_args(argv)
    if not a.all and not a.scenario:
        ap.error("Specify --all or --scenario S1 [S2 ...]")
    cfg = GeneratorConfig(attenuation=a.attenuation, lag_scale=a.lag_scale,
                          tc_scale=a.tc_scale, rate_scale=a.rate_scale,
                          s4_pre_threshold_runoff_factor=a.s4_runoff_factor,
                          base_factor_on="tc" if a.legacy_tb else "tp",
                          s2_failure_mode=a.s2_failure_mode,
                          s6_upstream_decay=a.s6_upstream_decay, seed=a.seed,
                          guard_mode=a.guard_mode)
    data = load_project_data()
    kw = {"n_windows": a.n_windows} if a.n_windows else {}
    for sid in (list(SCENARIO_MAP) if a.all else a.scenario):
        SCENARIO_MAP[sid](data, cfg, a.out_root, **kw)
    print("\nDone. Run scenario_evaluator.py next.")


if __name__ == "__main__":
    main()
