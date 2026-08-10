"""
analyse_scenarios.py  —  Scenario results analysis and paper figures
======================================================================
Reads scenario_summary.csv produced by scenario_evaluator.py and
produces four publication-quality figures plus a summary table.

Figures produced
----------------
F1  scenario_advantage_table.png
    Heatmap: ΔNSE (scenario NSE − real-data NSE) per model × scenario.
    Reveals which models benefit most from which scenario conditions.

F2  gauge_failure_degradation.png
    RMSE degradation ratio vs number of failed gauges per model.
    Key result: graph models degrade more slowly than per-node models.

F3  convective_cell_horizon.png
    Per-horizon NSE on S1 (ConvectiveCell) vs real-data NSE.
    Shows where HAND topology advantage is largest.

F4  scenario_difficulty.png
    Real-data RMSE vs scenario RMSE per model (scatter, all scenarios).
    Contextualises how much harder the synthetic dataset is.

Usage
-----
    python src/scenarios/analyse_scenarios.py
    python src/scenarios/analyse_scenarios.py --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
SCEN_RES    = RESULTS_DIR / "scenarios"
FIGS_DIR    = RESULTS_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

REAL_CSV    = RESULTS_DIR / "global_metrics_summary.csv"
SCEN_CSV    = SCEN_RES    / "scenario_summary.csv"

STEP_MIN    = 15

# ── Colour / label conventions (must match analyse_experiments.py) ────────────
MODEL_ORDER = ["gru", "lstm", "ealstm", "st_gnn",
               "st_gnn_dyn_edge", "st_gnn_hand_edge", "st_gnn_soil_gate",
               "dfc_gnn", "dfc_gnn_unified"]
MODEL_LABELS = {
    "gru":              "GRU",
    "lstm":             "LSTM",
    "ealstm":           "EA-LSTM",
    "st_gnn":           "ST-GNN (static)",
    "st_gnn_dyn_edge":  "ST-GNN DynEdge",
    "st_gnn_hand_edge": "ST-GNN HAND",
    "st_gnn_soil_gate": "ST-GNN Soil Gate",
    "dfc_gnn":          "DFC-GNN",
    "dfc_gnn_unified":  "PC-DFC-GNN",
}
MODEL_COLORS = {
    "gru":              "#1D9E75",
    "lstm":             "#D85A30",
    "ealstm":           "#E67E22",
    "st_gnn":           "#185FA5",
    "st_gnn_dyn_edge":  "#7B68EE",
    "st_gnn_hand_edge": "#9B59B6",
    "st_gnn_soil_gate": "#5B2C6F",
    "dfc_gnn":          "#B8860B",
    "dfc_gnn_unified":  "#D4A017",
}
MODEL_MARKERS = {
    "gru": "o", "lstm": "s", "ealstm": "d",
    "st_gnn": "^", "st_gnn_dyn_edge": "P",
    "st_gnn_hand_edge": "X", "st_gnn_soil_gate": "v",
    "dfc_gnn": "h", "dfc_gnn_unified": "*",
}

HZ_LABEL = {4: "1hr", 12: "3hr", 16: "4hr", 24: "6hr", 48: "12hr"}

SCEN_LABELS = {
    "S1_ConvectiveCell":   "S1\nConvective\nCell",
    "S2_GaugeFailure":     "S2\nGauge\nFailure",
    "S3_ChannelBlockage":  "S3\nChannel\nBlockage",
    "S4_SatBreakthrough":  "S4\nSat.\nBreakthrough",
    "S5_SpatialGradient":  "S5\nSpatial\nGradient",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real and scenario result CSVs."""
    if not SCEN_CSV.exists():
        print(f"Scenario summary not found: {SCEN_CSV}")
        print("Run: python src/scenarios/scenario_evaluator.py --all-scenarios")
        sys.exit(1)

    df_scen = pd.read_csv(SCEN_CSV)
    df_real = pd.read_csv(REAL_CSV) if REAL_CSV.exists() else pd.DataFrame()

    # Normalise model tag column name
    if "model_tag" in df_scen.columns and "model" not in df_scen.columns:
        df_scen = df_scen.rename(columns={"model_tag": "model"})

    print(f"Scenario rows: {len(df_scen)}")
    print(f"Real-data rows: {len(df_real)}")
    return df_scen, df_real


# ══════════════════════════════════════════════════════════════════════════════
# F1 — Scenario advantage heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_scenario_advantage_table(df_scen: pd.DataFrame,
                                   df_real: pd.DataFrame) -> None:
    """
    Heatmap showing ΔNSE = (scenario NSE) − (real-data NSE) per model × scenario.
    Positive (green) = model performs better relatively on this scenario.
    Negative (red)   = model degrades more on this scenario.

    Reference horizon: T_out=4 (1hr) — most directly comparable across both sets.
    """
    hz = 4
    scenarios = [s for s in SCEN_LABELS if s in df_scen["scenario"].unique()]
    models    = [m for m in MODEL_ORDER if m in df_scen["model"].unique()]

    # Real-data NSE per model at hz=4
    real_nse = {}
    if not df_real.empty and "nse_mean" in df_real.columns:
        for m in models:
            sub = df_real[(df_real.model == m) | (df_real.model == MODEL_LABELS.get(m, ""))]
            if not sub.empty:
                real_nse[m] = float(sub["nse_mean"].values[0])
    elif not df_real.empty and "NSE mean" in df_real.columns:
        for m in models:
            sub = df_real[
                ((df_real.model == m) | (df_real.model == MODEL_LABELS.get(m, ""))) &
                (df_real.get("horizon", pd.Series()) == HZ_LABEL.get(hz, ""))
            ]
            if not sub.empty:
                real_nse[m] = float(sub["NSE mean"].values[0])

    # Build ΔNSE matrix
    data = np.full((len(models), len(scenarios)), np.nan)
    for j, scen in enumerate(scenarios):
        for i, m in enumerate(models):
            sub = df_scen[(df_scen.scenario == scen) &
                          (df_scen.model    == m) &
                          (df_scen.horizon  == hz)]
            if sub.empty:
                continue
            scen_nse = float(sub["nse_syn"].mean())
            rn       = real_nse.get(m, np.nan)
            if np.isfinite(scen_nse) and np.isfinite(rn):
                data[i, j] = scen_nse - rn

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.8), len(models) * 0.9 + 1.5))
    fig.patch.set_facecolor("white")

    vmax = max(0.05, np.nanmax(np.abs(data)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="ΔNSE (scenario − real data)",
                 shrink=0.8)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCEN_LABELS.get(s, s) for s in scenarios],
                       fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=9)

    for i in range(len(models)):
        for j in range(len(scenarios)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.4f}", ha="center", va="center",
                        fontsize=7.5,
                        color="black" if abs(v) < 0.6 * vmax else "white")

    ax.set_title(
        f"ΔNSE: scenario performance relative to real Lee test data\n"
        f"(T_out={hz}, {HZ_LABEL[hz]}) — green = scenario reveals advantage, "
        f"red = scenario reveals weakness",
        fontsize=10)
    fig.tight_layout()
    out = FIGS_DIR / "scenario_advantage_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# F2 — Gauge failure degradation
# ══════════════════════════════════════════════════════════════════════════════

def plot_gauge_failure_degradation(df_scen: pd.DataFrame) -> None:
    """
    S2 GaugeFailure: RMSE degradation ratio vs real-data performance.
    Models with lower degradation ratio maintain accuracy under sensor loss.
    """
    df_s2 = df_scen[df_scen.scenario == "S2_GaugeFailure"].copy()
    if df_s2.empty or "s2_degradation_ratio" not in df_s2.columns:
        print("[skip] F2: S2_GaugeFailure data not available")
        return

    models  = [m for m in MODEL_ORDER if m in df_s2.model.unique()]
    hz_list = sorted(df_s2.horizon.unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(hz_list))
    w = 0.8 / max(len(models), 1)
    offsets = np.linspace(-(len(models)-1)/2*w, (len(models)-1)/2*w, len(models))

    for m, offset in zip(models, offsets):
        sub = df_s2[df_s2.model == m]
        means = [sub[sub.horizon == hz]["s2_degradation_ratio"].mean()
                 for hz in hz_list]
        stds  = [sub[sub.horizon == hz]["s2_degradation_ratio"].std(ddof=1)
                 if len(sub[sub.horizon == hz]) > 1 else 0
                 for hz in hz_list]
        ax.bar(x + offset, means, w,
               color=MODEL_COLORS.get(m, "#888888"),
               yerr=stds, capsize=3, alpha=0.85,
               label=MODEL_LABELS.get(m, m), edgecolor="white")

    ax.axhline(1.0, color="#444441", lw=1.0, ls="--",
               label="No degradation (ratio=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([HZ_LABEL.get(h, str(h)) for h in hz_list])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE ratio (scenario / real data)")
    ax.set_title(
        "S2 GaugeFailure: RMSE degradation relative to real-data baseline\n"
        "Lower ratio = more robust to upstream sensor loss",
        fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = FIGS_DIR / "gauge_failure_degradation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# F3 — ConvectiveCell horizon curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_convective_cell_horizon(df_scen: pd.DataFrame,
                                  df_real: pd.DataFrame) -> None:
    """
    S1 ConvectiveCell: NSE by horizon for graph vs no-graph models.
    Contrasted against real-data horizon curves to show where HAND
    topology advantage is amplified by the scenario conditions.
    """
    df_s1 = df_scen[df_scen.scenario == "S1_ConvectiveCell"].copy()
    if df_s1.empty:
        print("[skip] F3: S1_ConvectiveCell data not available")
        return

    models   = [m for m in MODEL_ORDER if m in df_s1.model.unique()]
    hz_list  = sorted(df_s1.horizon.unique())
    hz_ticks = [h * STEP_MIN / 60 for h in hz_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.patch.set_facecolor("white")

    for m in models:
        c   = MODEL_COLORS.get(m, "#888888")
        mk  = MODEL_MARKERS.get(m, "o")
        lbl = MODEL_LABELS.get(m, m)
        sub = df_s1[df_s1.model == m]
        means = [sub[sub.horizon == hz]["nse_syn"].mean() for hz in hz_list]
        stds  = [sub[sub.horizon == hz]["nse_syn"].std(ddof=1)
                 if len(sub[sub.horizon == hz]) > 1 else 0
                 for hz in hz_list]
        ax1.errorbar(hz_ticks, means, yerr=stds, color=c, marker=mk,
                     ms=7, lw=2, capsize=4, label=lbl)

    ax1.set_xlabel("Lead time (hr)")
    ax1.set_ylabel("NSE (synthetic ConvectiveCell scenario)")
    ax1.set_title("S1 ConvectiveCell — per-horizon NSE", fontsize=10)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: ΔNSE (scenario − real data) to isolate scenario effect
    if not df_real.empty:
        real_col = "NSE mean" if "NSE mean" in df_real.columns else "nse_mean"
        hz_col   = "horizon"
        for m in models:
            c   = MODEL_COLORS.get(m, "#888888")
            mk  = MODEL_MARKERS.get(m, "o")
            lbl = MODEL_LABELS.get(m, m)
            deltas = []
            for hz in hz_list:
                sub_s1 = df_s1[(df_s1.model == m) & (df_s1.horizon == hz)]
                hz_str = HZ_LABEL.get(hz, str(hz))
                sub_rd = df_real[
                    ((df_real.get("model", pd.Series()) == m) |
                     (df_real.get("model", pd.Series()) == MODEL_LABELS.get(m, ""))) &
                    ((df_real.get(hz_col, pd.Series()) == hz) |
                     (df_real.get(hz_col, pd.Series()) == hz_str))
                ]
                if sub_s1.empty or sub_rd.empty:
                    deltas.append(np.nan)
                    continue
                deltas.append(float(sub_s1["nse_syn"].mean()) -
                               float(sub_rd[real_col].values[0]))
            ax2.plot(hz_ticks, deltas, color=c, marker=mk, ms=7, lw=2, label=lbl)

        ax2.axhline(0, color="#444441", lw=0.8, ls="--")
        ax2.set_xlabel("Lead time (hr)")
        ax2.set_ylabel("ΔNSE (scenario − real data)")
        ax2.set_title("S1 Scenario vs real data — ΔNSE by horizon", fontsize=10)
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_title("Real-data results not available for comparison")

    fig.suptitle(
        "S1: Isolated convective storm — reveals HAND topology advantage\n"
        "Positive ΔNSE means the model performs better relatively under "
        "convective conditions than on the full historical record",
        fontsize=10)
    fig.tight_layout()
    out = FIGS_DIR / "convective_cell_horizon.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# F4 — Scenario difficulty scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_scenario_difficulty(df_scen: pd.DataFrame) -> None:
    """
    Scatter: real-data RMSE vs scenario RMSE for each model × scenario.
    Points above the diagonal (scenario RMSE > real RMSE) = harder scenario.
    """
    df = df_scen.dropna(subset=["rmse_syn", "real_rmse"]).copy()
    if df.empty:
        print("[skip] F4: no valid rmse_syn / real_rmse pairs")
        return

    scenarios = df.scenario.unique()
    n_s   = len(scenarios)
    n_col = min(3, n_s)
    n_row = int(np.ceil(n_s / n_col))

    fig, axes = plt.subplots(n_row, n_col,
                              figsize=(5 * n_col, 4.5 * n_row),
                              squeeze=False)
    fig.patch.set_facecolor("white")

    for ax_idx, scen in enumerate(scenarios):
        ax  = axes[ax_idx // n_col][ax_idx % n_col]
        sub = df[df.scenario == scen]

        ax_max = max(sub["rmse_syn"].max(), sub["real_rmse"].max()) * 1.1
        ax.plot([0, ax_max], [0, ax_max], color="#888888", lw=0.8, ls="--")

        for m in MODEL_ORDER:
            m_sub = sub[sub.model == m]
            if m_sub.empty: continue
            ax.scatter(m_sub["real_rmse"], m_sub["rmse_syn"],
                       color=MODEL_COLORS.get(m, "#888888"),
                       label=MODEL_LABELS.get(m, m),
                       s=80, zorder=3, alpha=0.85,
                       marker=MODEL_MARKERS.get(m, "o"))

        ax.set_xlabel("RMSE (real test data)", fontsize=8)
        ax.set_ylabel("RMSE (synthetic scenario)", fontsize=8)
        ax.set_title(SCEN_LABELS.get(scen, scen), fontsize=9)
        ax.tick_params(labelsize=7.5)

    # Single legend outside
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(8, len(handles)), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

    # Hide unused axes
    for ax_idx in range(n_s, n_row * n_col):
        axes[ax_idx // n_col][ax_idx % n_col].set_visible(False)

    fig.suptitle(
        "Scenario difficulty: real-data vs scenario RMSE\n"
        "Points above diagonal indicate the scenario is harder than "
        "the historical test set",
        fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIGS_DIR / "scenario_difficulty.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary table (CSV)
# ══════════════════════════════════════════════════════════════════════════════

def write_summary_table(df_scen: pd.DataFrame, df_real: pd.DataFrame) -> None:
    """
    Write one consolidated summary CSV and print key findings to console.
    """
    # Aggregate by model × scenario × horizon
    agg = (df_scen.groupby(["model", "scenario", "horizon"])
           .agg(nse_mean  = ("nse_syn",  "mean"),
                nse_std   = ("nse_syn",  "std"),
                rmse_mean = ("rmse_syn", "mean"),
                pod_mean  = ("pod_syn",  "mean"),
                far_mean  = ("far_syn",  "mean"),
                deg_ratio = ("degradation_ratio", "mean"))
           .reset_index())

    out = SCEN_RES / "scenario_summary_aggregated.csv"
    agg.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved aggregated summary: {out.name}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS (T_out=4, 1-hour horizon)")
    print("=" * 60)
    for scen in df_scen.scenario.unique():
        print(f"\n  {scen}:")
        sub = agg[(agg.scenario == scen) & (agg.horizon == 4)]
        sub = sub.set_index("model")
        best_m  = sub["nse_mean"].idxmax()
        worst_m = sub["nse_mean"].idxmin()
        if best_m and worst_m:
            best_lbl  = MODEL_LABELS.get(best_m, best_m)
            worst_lbl = MODEL_LABELS.get(worst_m, worst_m)
            print(f"    Best:  {best_lbl:<28} NSE={sub.loc[best_m,'nse_mean']:.4f}")
            print(f"    Worst: {worst_lbl:<28} NSE={sub.loc[worst_m,'nse_mean']:.4f}")
            delta = sub.loc[best_m,"nse_mean"] - sub.loc[worst_m,"nse_mean"]
            print(f"    Gap (best−worst): ΔNSE={delta:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse synthetic scenario results and produce paper figures")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display figures (save only)")
    print("=" * 60)
    print("Scenario results analysis")
    print("=" * 60)

    df_scen, df_real = load_data()

    write_summary_table(df_scen, df_real)
    plot_scenario_advantage_table(df_scen, df_real)
    plot_gauge_failure_degradation(df_scen)
    plot_convective_cell_horizon(df_scen, df_real)
    plot_scenario_difficulty(df_scen)

    print(f"\nAll figures saved to: {FIGS_DIR}")


if __name__ == "__main__":
    main()
