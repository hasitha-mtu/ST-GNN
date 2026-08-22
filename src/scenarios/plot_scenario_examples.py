"""
plot_scenario_examples.py — one figure, one panel per scenario, showing an
actual injected synthetic trajectory at the node(s) each scenario targets.

Purpose: readers of the paper should be able to SEE what each scenario does
to the data, not just read a description of the injection mechanics. Each
panel plots the relevant node's stage trace over one representative
T_WINDOW-length window, with the event onset marked and annotated.

Reads directly from the already-generated scenario outputs
(X_synthetic.npy / y_synthetic.npy / scenario_meta.json in each scenario's
SCEN_DIR subfolder) — run scenario_generator.py --all first.

Usage:
    python plot_scenario_examples.py                  # combined 2x3 grid
    python plot_scenario_examples.py --individual      # + one file per scenario
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCEN_DIR = BASE_DIR / "dataset" / "scenarios"
OUT_DIR  = BASE_DIR / "results" / "figures"

T_WINDOW = 104   # steps per generation window, 15-min resolution
T_IN     = 32
STEP_MIN = 15

COLOR_PRIMARY   = "#C0392B"   # the node the event is injected AT
COLOR_UPSTREAM  = "#8E44AD"
COLOR_DOWNSTREAM = "#2471A3"
COLOR_SECONDARY = "#117864"


def _load(scenario_folder: str, window_idx: int = 0):
    d = SCEN_DIR / scenario_folder
    meta = json.load(open(d / "scenario_meta.json"))
    X_full = np.load(d / "X_synthetic.npy")
    y_full = np.load(d / "y_synthetic.npy")
    n_available = X_full.shape[0] // T_WINDOW
    if not (0 <= window_idx < n_available):
        print(f"  [warn] {scenario_folder}: window {window_idx} out of range "
              f"(0-{n_available - 1} available) -- using window 0")
        window_idx = 0
    sl = slice(window_idx * T_WINDOW, (window_idx + 1) * T_WINDOW)
    return meta, X_full[sl], y_full[sl]


def _time_axis(n_steps: int):
    """Hours relative to the start of the input window."""
    return np.arange(n_steps) * STEP_MIN / 60.0


def _mark_event(ax, t_step: int, label: str):
    t_hr = t_step * STEP_MIN / 60.0
    ax.axvline(t_hr, color="black", ls="--", lw=1, alpha=0.7)
    ax.text(t_hr, ax.get_ylim()[1], f" {label}", fontsize=7.5,
            va="top", ha="left", rotation=0)


# ─────────────────────────────────────────────────────────────────────────
# Per-scenario panel builders — each returns nothing, draws on the given ax
# ─────────────────────────────────────────────────────────────────────────

def panel_s1(ax, window_idx: int = 0):
    meta, X, y = _load("S1_ConvectiveCell", window_idx)
    hw = meta["headwater_nodes"][0]
    ds = meta["downstream_nodes"][0] if meta.get("downstream_nodes") else None
    t = _time_axis(T_WINDOW)
    ax.plot(t, y[:, hw], color=COLOR_PRIMARY, lw=1.6, label="Headwater (injected)")
    if ds is not None:
        ax.plot(t, y[:, ds], color=COLOR_DOWNSTREAM, lw=1.4,
                label="Downstream (routed)", alpha=0.85)
    _mark_event(ax, meta["pulse_at_step"], "Storm onset")
    ax.set_title("S1 — ConvectiveCell", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("Stage (m)")


def panel_s2(ax, window_idx: int = 0):
    meta, X, y = _load("S2_GaugeFailure", window_idx)

    # Multi-realization schema: fail_nodes/duration_steps now vary per
    # realization (meta["realizations"]), looked up via
    # meta["window_realization_id"][window_idx] -- no longer a fixed
    # meta["headwater_nodes"] list, since failure selection is itself
    # random (or criticality-based) per realization, not always drawn
    # from the same headwater subset. Falls back to the old field for
    # any single-realization scenario data generated before this
    # rebuild, so this panel works against both.
    realizations = meta.get("realizations")
    per_window_realization_id = meta.get("window_realization_id")
    if realizations is not None and per_window_realization_id is not None:
        rid = per_window_realization_id[window_idx]
        r = next(rr for rr in realizations if rr["id"] == rid)
        hw = r["fail_nodes"][0]
        duration = r["duration_steps"]
        mode = r["selection_mode"]
    else:
        hw = meta.get("headwater_nodes", [0])[0]
        duration = T_WINDOW
        mode = "headwater"

    t_fail = meta["t_failure_step"]
    t_recover = min(t_fail + duration, T_WINDOW)

    t = _time_axis(T_WINDOW)
    ax.plot(t, y[:, hw], color=COLOR_PRIMARY, lw=1.6, label="True stage (unobserved while failed)")
    obs = y[:, hw].copy()
    # Sensor reading is frozen only for the duration of the failure, not
    # unconditionally to end-of-window -- reflects real recovery if
    # duration_steps ends before the window does.
    obs[t_fail:t_recover] = obs[t_fail - 1]
    ax.plot(t, obs, color="gray", lw=1.4, ls=":", label="Frozen sensor reading (model input)")
    _mark_event(ax, t_fail, f"Gauge fails ({mode})")
    if t_recover < T_WINDOW:
        _mark_event(ax, t_recover, "Recovers")
    ax.set_title("S2 — GaugeFailure", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("Stage (m)")


def panel_s3(ax, window_idx: int = 0):
    meta, X, y = _load("S3_InniscarraRelease", window_idx)
    rel = meta["release_node"]["idx"]
    ds  = meta["downstream_nodes"][0] if meta.get("downstream_nodes") else None
    t = _time_axis(T_WINDOW)
    ax.plot(t, y[:, rel], color=COLOR_PRIMARY, lw=1.6,
           label=f"{meta['release_node']['name']} (release)")
    if ds is not None:
        ax.plot(t, y[:, ds], color=COLOR_DOWNSTREAM, lw=1.4,
                label="Downstream (routed)", alpha=0.85)
    inflow_names = list(meta.get("decoupled_nodes", {}).values())
    if inflow_names:
        inflow_idx = inflow_names[0]["idx"]
        ax.plot(t, y[:, inflow_idx], color=COLOR_UPSTREAM, lw=1.2, ls="--",
               label="Reservoir inflow gauge (decoupled)", alpha=0.8)
    _mark_event(ax, meta["t_release_step"], "Release begins")
    ax.set_title("S3 — InniscarraRelease", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_ylabel("Stage (m)")


def panel_s4(ax, window_idx: int = 0):
    meta, X, y = _load("S4_SatBreakthrough", window_idx)
    # Node 23 = Fitzgerald's Park, tc_hr=0.612hr -- the slowest/longest
    # unit hydrograph response in the Lee catchment (see uh_params.json).
    # Node 0 (Ballincolly, tc_hr=0.25hr = 1 timestep) was tried first but
    # its response compresses into a single spike-then-crash that reads
    # as noise rather than the sustained "abrupt acceleration" this
    # panel exists to show -- a genuinely correct injection, just a poor
    # illustrative choice for a reader unfamiliar with the mechanism.
    node = 23

    # Multi-realization schema: sat_breakthrough_step/excess_threshold
    # now vary per realization (meta["realizations"]), looked up via
    # meta["window_realization_id"][window_idx] for the SPECIFIC window
    # being plotted -- no longer one top-level scalar shared by every
    # window. Falls back to the old scalar fields for any
    # single-realization scenario data generated before this schema
    # change, so this panel works against both.
    realizations = meta.get("realizations")
    per_window_realization_id = meta.get("window_realization_id")
    if realizations is not None and per_window_realization_id is not None:
        rid = per_window_realization_id[window_idx]
        r = next(rr for rr in realizations if rr["id"] == rid)
        bt = r["sat_breakthrough_step"]
        excess_threshold = r["excess_threshold"]
    else:
        bt = meta.get("sat_breakthrough_step", 46)
        excess_threshold = meta.get("excess_threshold", 0.75)
        if "sat_start" not in meta and realizations is None:
            print("  [warn] S4 scenario_meta.json predates the multi-realization "
                  "schema -- regenerate with scenario_generator.py for correct results.")

    t = _time_axis(T_WINDOW)
    ax2 = ax.twinx()
    ax2.plot(t, X[:, node, 9], color=COLOR_SECONDARY, lw=1.3, ls="-.",
            label="Catchment saturation", alpha=0.8)
    ax2.axhline(excess_threshold, color=COLOR_SECONDARY, lw=0.8, ls=":", alpha=0.6)
    ax2.set_ylabel("swvl2_sat_ratio", color=COLOR_SECONDARY, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_SECONDARY, labelsize=7)
    ax.plot(t, y[:, node], color=COLOR_PRIMARY, lw=1.6, label="Stage response")
    _mark_event(ax, bt, "Breakthrough")
    ax.set_title("S4 — SatBreakthrough", fontsize=10, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper left")
    ax.set_ylabel("Stage (m)")


def panel_s5(ax, window_idx: int = 0):
    meta, X, y = _load("S5_SpatialGradient", window_idx)
    ds = meta["downstream_nodes"][0] if meta.get("downstream_nodes") else 0
    t = _time_axis(T_WINDOW)
    ax.plot(t, y[:, ds], color=COLOR_DOWNSTREAM, lw=1.6,
           label="Downstream (early arrival from western rainfall)")
    # T_IN (forecast start) is when the west-east rainfall gradient is
    # applied -- the definable onset, added for visual consistency with
    # every other panel's event marker.
    _mark_event(ax, T_IN, "Gradient applied")

    # Multi-realization schema: gradient_range now varies per realization
    # (meta["realizations"]), looked up via
    # meta["window_realization_id"][window_idx] -- no longer a single
    # top-level [min,max] pair shared by every window.
    realizations = meta.get("realizations")
    per_window_realization_id = meta.get("window_realization_id")
    if realizations is not None and per_window_realization_id is not None:
        rid = per_window_realization_id[window_idx]
        r = next(rr for rr in realizations if rr["id"] == rid)
        grad_range = r["gradient_range"]
    else:
        grad_range = meta.get("gradient_range", [1.0, 2.5])

    ax.set_title(
        f"S5 — SpatialGradient\ngradient range {grad_range[0]:.2f}"
        f"-{grad_range[1]:.2f}", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("Stage (m)")


def panel_s6(ax, window_idx: int = 0):
    meta, X, y = _load("S6_ChannelBlockage", window_idx)
    bd = meta["blockage_node"]["idx"]
    up = meta["upstream_node"]["idx"]
    dn = meta["downstream_node"]["idx"]
    t = _time_axis(T_WINDOW)
    ax.plot(t, y[:, bd], color=COLOR_PRIMARY, lw=1.6,
           label=f"{meta['blockage_node']['name']} (blockage)")
    ax.plot(t, y[:, up], color=COLOR_UPSTREAM, lw=1.4,
           label="Upstream (backwater rise)")
    # Downstream suppression is typically a few cm against a ~1.5m
    # backwater rise on the SAME node/edge -- sharing one y-axis crushes
    # the suppression signal to visual invisibility even when it's
    # genuinely present in the data (confirmed via
    # find_best_scenario_windows.py's dn_pre_activity scoring). Plotted
    # on its own axis, same pattern as panel_s4's dual-axis approach.
    ax2 = ax.twinx()
    ax2.plot(t, y[:, dn], color=COLOR_DOWNSTREAM, lw=1.4, ls="--",
            label="Downstream (suppressed)")
    ax2.set_ylabel("Downstream stage (m)", color=COLOR_DOWNSTREAM, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_DOWNSTREAM, labelsize=7)
    _mark_event(ax, meta["t_blockage_step"], "Blockage occurs")
    ax.set_title("S6 — ChannelBlockage", fontsize=10, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper right")
    ax.set_ylabel("Stage (m)")


PANELS = [panel_s1, panel_s2, panel_s3, panel_s4, panel_s5, panel_s6]
SCEN_KEYS = ["s1", "s2", "s3", "s4", "s5", "s6"]


def _resolve_windows(default_window: int, overrides: dict) -> list:
    """Per-panel window index: override if given, else the shared --window default."""
    return [overrides.get(k, default_window) for k in SCEN_KEYS]


def plot_combined_grid(default_window: int = 0, overrides: dict | None = None):
    overrides = overrides or {}
    windows = _resolve_windows(default_window, overrides)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor("white")
    axes = axes.reshape(-1)
    for ax, panel_fn, win in zip(axes, PANELS, windows):
        try:
            panel_fn(ax, window_idx=win)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "Scenario not yet generated\n(run scenario_generator.py first)",
                   ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Hours from window start", fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Synthetic scenario construction — example injected trajectories",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "scenario_examples_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_individual(default_window: int = 0, overrides: dict | None = None):
    overrides = overrides or {}
    windows = _resolve_windows(default_window, overrides)
    names = ["S1_ConvectiveCell", "S2_GaugeFailure", "S3_InniscarraRelease",
             "S4_SatBreakthrough", "S5_SpatialGradient", "S6_ChannelBlockage"]
    for name, panel_fn, win in zip(names, PANELS, windows):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("white")
        try:
            panel_fn(ax, window_idx=win)
        except FileNotFoundError:
            print(f"[skip] {name}: not yet generated")
            plt.close(fig)
            continue
        ax.set_xlabel("Hours from window start")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = OUT_DIR / f"scenario_example_{name}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(
        description="Plot example synthetic scenario trajectories",
        epilog="Examples:\n"
               "  python plot_scenario_examples.py --window 3\n"
               "  python plot_scenario_examples.py --window-s4 5 --window-s6 2\n"
               "  python plot_scenario_examples.py --window 0 --window-s4 7 --individual",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--individual", action="store_true",
                   help="Also save one larger figure per scenario")
    p.add_argument("--window", type=int, default=0,
                   help="Window index applied to ALL scenarios (default 0, i.e. the "
                        "first generated window). Each scenario has a different number "
                        "of windows available (check scenario_meta.json's n_windows, or "
                        "just try a few small integers -- an out-of-range value prints "
                        "the valid range and falls back to window 0 automatically).")
    for k in SCEN_KEYS:
        p.add_argument(f"--window-{k}", type=int, default=None,
                       help=f"Override window index for {k.upper()} specifically "
                            f"(takes precedence over --window)")
    args = p.parse_args()

    overrides = {k: v for k in SCEN_KEYS
                if (v := getattr(args, f"window_{k}")) is not None}

    plot_combined_grid(default_window=args.window, overrides=overrides)
    if args.individual:
        plot_individual(default_window=args.window, overrides=overrides)


if __name__ == "__main__":
    main()
