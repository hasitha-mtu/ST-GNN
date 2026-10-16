"""
plot_scenario_panel.py  —  Six-panel scenario ranking figure (replaces Figs 9-14)
================================================================================
Reviewer presentation request: one figure, consistent axes/typography and a
fixed colour per model. Bars = mean NSE over horizons and seeds (observable
forecasts by default), whiskers = SD over seeds of the horizon-mean.

    python src/scenarios/plot_scenario_panel.py [--scope all]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCEN_CSV = BASE_DIR / "results/scenarios/scenario_summary.csv"
OUT = BASE_DIR / "results/figures/scenario_rankings_6panel.png"
ORDER = ["S1_ConvectiveCell", "S2_GaugeFailure", "S3_InniscarraRelease",
         "S4_SatBreakthrough", "S5_SpatialGradient", "S6_ChannelBlockage"]
TITLE = {"S1_ConvectiveCell": "S1 Convective cell", "S2_GaugeFailure": "S2 Gauge failure",
         "S3_InniscarraRelease": "S3 Reservoir release", "S4_SatBreakthrough": "S4 Saturation breakthrough",
         "S5_SpatialGradient": "S5 Spatial gradient", "S6_ChannelBlockage": "S6 Channel blockage"}
LABEL = {"gru": "GRU", "lstm": "LSTM", "ealstm": "EA-LSTM", "st_gnn": "ST-GNN static",
         "st_gnn_dyn_edge": "DynEdge", "st_gnn_hand_edge": "HANDEdge", "st_gnn_soil_gate": "SoilGate",
         "st_gnn_backwater_edge": "BackwaterEdge", "dfc_gnn": "DFC-GNN", "dfc_gnn_unified": "PC-DFC-GNN"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["observable", "all"], default="observable")
    a = ap.parse_args()
    df = pd.read_csv(SCEN_CSV)
    col = "observable_nse_syn" if a.scope == "observable" and "observable_nse_syn" in df else "nse_syn"
    models = [m for m in LABEL if m in set(df.model)]
    cmap = plt.get_cmap("tab10")
    colour = {m: cmap(i % 10) for i, m in enumerate(models)}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=False)
    for ax, scen in zip(axes.ravel(), ORDER):
        d = df[df.scenario == scen]
        if d.empty:
            ax.set_visible(False); continue
        per_seed = d.groupby(["model", "seed"])[col].mean().reset_index()
        stat = per_seed.groupby("model")[col].agg(["mean", "std"]).reindex(models)
        stat = stat.sort_values("mean", ascending=True)
        ax.barh(range(len(stat)), stat["mean"], xerr=stat["std"],
                color=[colour[m] for m in stat.index], capsize=2)
        ax.set_yticks(range(len(stat))); ax.set_yticklabels([LABEL[m] for m in stat.index], fontsize=8)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(TITLE[scen], fontsize=10)
        ax.set_xlabel(f"NSE ({a.scope} forecasts)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300); plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
