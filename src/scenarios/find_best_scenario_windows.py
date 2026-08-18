"""
find_best_scenario_windows.py — rank generated windows by illustrative clarity

For each scenario, scores every available window on two things:
  1. signal_to_noise: how large the injected event is relative to the
     window's own pre-event baseline noise (bigger = more visible)
  2. cleanliness: how little UNRELATED activity happens elsewhere in the
     window (penalises exactly the kind of confounding pre/post-event
     spikes that made some example windows confusing earlier -- activity
     the scenario generator didn't inject, just happened to be in the
     underlying real data for that particular historical window)

score = signal_to_noise * cleanliness

Prints the top 5 windows per scenario and writes a JSON file with the
single best --window-sX value for each, ready to paste directly into
plot_scenario_examples.py's CLI.

Usage:
    python find_best_scenario_windows.py
    python find_best_scenario_windows.py --top 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCEN_DIR = BASE_DIR / "dataset" / "scenarios"
OUT_DIR  = BASE_DIR / "results" / "figures"

T_WINDOW = 104
T_IN     = 32


def _load(scenario_folder: str):
    d = SCEN_DIR / scenario_folder
    meta = json.load(open(d / "scenario_meta.json"))
    X = np.load(d / "X_synthetic.npy")
    y = np.load(d / "y_synthetic.npy")
    n_windows = X.shape[0] // T_WINDOW
    return meta, X, y, n_windows


def _window(arr, idx):
    return arr[idx * T_WINDOW : (idx + 1) * T_WINDOW]


def _signal_to_noise(trace, event_start, event_len=30, pre_len=None):
    pre = trace[:event_start] if pre_len is None else trace[max(0, event_start - pre_len):event_start]
    if len(pre) < 3:
        return 0.0
    post = trace[event_start:event_start + event_len]
    if len(post) == 0:
        return 0.0
    noise = pre.std() + 1e-6
    signal = np.max(np.abs(post - pre.mean()))
    return float(signal / noise)


def _cleanliness(trace, event_start, event_len=30, pre_len=None):
    """
    Penalises activity OUTSIDE the [event_start-pre_len, event_start+event_len]
    region relative to the pre-event noise floor -- catches windows where
    the real underlying data happens to have an unrelated spike elsewhere
    (confusing in a figure meant to isolate one mechanism).
    """
    pre_start = max(0, event_start - (pre_len or event_start))
    pre = trace[:event_start]
    if len(pre) < 3:
        return 0.5
    noise = pre.std() + 1e-6
    outside = np.concatenate([
        trace[:pre_start],
        trace[event_start + event_len:],
    ])
    if len(outside) == 0:
        return 1.0
    extraneous = np.max(np.abs(outside - pre.mean())) if len(outside) else 0.0
    return float(1.0 / (1.0 + extraneous / noise))


# ─────────────────────────────────────────────────────────────────────────
# Per-scenario scoring — mirrors plot_scenario_examples.py's panel logic
# ─────────────────────────────────────────────────────────────────────────

def score_s1(meta, X, y, idx):
    hw = meta["headwater_nodes"][0]
    trace = _window(y, idx)[:, hw]
    ev = meta["pulse_at_step"]
    return _signal_to_noise(trace, ev) * _cleanliness(trace, ev)


def score_s2(meta, X, y, idx):
    hw = meta["headwater_nodes"][0]
    trace = _window(y, idx)[:, hw]
    t_fail = meta["t_failure_step"]
    pre = trace[:t_fail]
    if len(pre) < 3:
        return 0.0
    post_true = trace[t_fail:]
    frozen = np.full_like(post_true, trace[t_fail - 1])
    divergence = np.mean(np.abs(post_true - frozen))
    noise = pre.std() + 1e-6
    return float(divergence / noise)


def score_s3(meta, X, y, idx):
    rel = meta["release_node"]["idx"]
    trace = _window(y, idx)[:, rel]
    ev = meta["t_release_step"]
    sn = _signal_to_noise(trace, ev, event_len=T_WINDOW - ev)
    # Bonus: the decoupled inflow gauge should stay FLAT post-release --
    # reward windows where that's clearly true, since it's the whole point.
    inflow_names = list(meta.get("decoupled_nodes", {}).values())
    flatness_bonus = 1.0
    if inflow_names:
        inflow_idx = inflow_names[0]["idx"]
        inflow_trace = _window(y, idx)[:, inflow_idx]
        pre_std = inflow_trace[:ev].std() + 1e-6
        post_std = inflow_trace[ev:].std() + 1e-6
        flatness_bonus = float(1.0 / (1.0 + max(0.0, post_std / pre_std - 1.0)))
    return sn * flatness_bonus


def score_s4(meta, X, y, idx):
    node = 23   # Fitzgerald's Park -- same choice as plot_scenario_examples.py
    trace = _window(y, idx)[:, node]
    bt = meta["sat_breakthrough_step"]
    return _signal_to_noise(trace, bt, event_len=T_WINDOW - bt) * _cleanliness(trace, bt, event_len=T_WINDOW - bt)


def score_s5(meta, X, y, idx):
    ds = meta["downstream_nodes"][0] if meta.get("downstream_nodes") else 0
    trace = _window(y, idx)[:, ds]
    return _signal_to_noise(trace, T_IN, event_len=T_WINDOW - T_IN) * _cleanliness(trace, T_IN, event_len=T_WINDOW - T_IN)


def score_s6(meta, X, y, idx):
    bd = meta["blockage_node"]["idx"]
    up = meta["upstream_node"]["idx"]
    dn = meta["downstream_node"]["idx"]
    ev = meta["t_blockage_step"]
    yw = _window(y, idx)
    bd_score = _signal_to_noise(yw[:, bd], ev, event_len=T_WINDOW - ev)
    up_score = _signal_to_noise(yw[:, up], ev, event_len=T_WINDOW - ev)
    # Downstream suppression is multiplicative on the real baseline --
    # reward windows where the downstream node actually HAS baseline
    # activity to suppress (near-zero baseline = invisible suppression,
    # exactly the weakness found in the first example figure).
    dn_pre_activity = float(np.std(yw[:ev, dn]))
    clean = _cleanliness(yw[:, bd], ev, event_len=T_WINDOW - ev)
    return (bd_score + up_score) * (1.0 + dn_pre_activity) * clean


SCORERS = {
    "S1_ConvectiveCell":    score_s1,
    "S2_GaugeFailure":      score_s2,
    "S3_InniscarraRelease": score_s3,
    "S4_SatBreakthrough":   score_s4,
    "S5_SpatialGradient":   score_s5,
    "S6_ChannelBlockage":   score_s6,
}

CLI_KEYS = {
    "S1_ConvectiveCell": "s1", "S2_GaugeFailure": "s2",
    "S3_InniscarraRelease": "s3", "S4_SatBreakthrough": "s4",
    "S5_SpatialGradient": "s5", "S6_ChannelBlockage": "s6",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=5, help="How many top windows to print per scenario")
    args = p.parse_args()

    recommended = {}
    cli_parts = []

    for name, scorer in SCORERS.items():
        try:
            meta, X, y, n_windows = _load(name)
        except FileNotFoundError:
            print(f"[skip] {name}: not yet generated")
            continue

        scores = []
        for idx in range(n_windows):
            try:
                s = scorer(meta, X, y, idx)
            except Exception as e:
                s = float("-inf")
            scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{name}  ({n_windows} windows available)")
        for rank, (idx, s) in enumerate(scores[:args.top]):
            marker = " <-- best" if rank == 0 else ""
            print(f"  #{idx:<3d} score={s:.3f}{marker}")

        best_idx = scores[0][0] if scores else 0
        recommended[name] = best_idx
        cli_parts.append(f"--window-{CLI_KEYS[name]} {best_idx}")

    if recommended:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "recommended_scenario_windows.json"
        with open(out_path, "w") as f:
            json.dump(recommended, f, indent=2)
        print(f"\nSaved {out_path}")
        print("\nReady-to-paste command for the best window per scenario:")
        print("  python plot_scenario_examples.py --individual " + " ".join(cli_parts))


if __name__ == "__main__":
    main()
