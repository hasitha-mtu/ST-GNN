"""
analyse_experiments.py  –  Multi-model, multi-seed, multi-horizon evaluation
=============================================================================
Loads all results from the structured experiment directory and produces:

  1. Global metrics table  (mean ± std across 3 seeds, per model × horizon)
  2. Statistical comparison (Wilcoxon signed-rank test, GRU vs ST-GNN)
  3. Per-node NSE comparison (ST-GNN advantage map across nodes)
  4. Horizon degradation curves (NSE vs forecast horizon, all models)
  5. Skill score distribution (violin plot per model)
  6. Node-level spotlight: upstream-connected vs isolated nodes

Figures saved to results/figures/model_comparison/

Directory contract
──────────────────
checkpoints/
  {model}/{seed}/{horizon}/
    test_metrics.json     → test_loss, rmse, mae, nse, mbe
    per_node_metrics.csv  → ref, name, n_valid, rmse, mae, mbe,
                             nse, persist_nse, skill
    training_history.csv  → epoch, train_loss, val_loss, val_nse, ...

Horizon folders: 4 = T_out 4 (1 hr), 12 = T_out 12 (3 hr), 16 = T_out 16 (4 hr)
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
CKPT_DIR   = BASE_DIR / "checkpoints"
OUT_DIR    = BASE_DIR / "results/figures/model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment config ──────────────────────────────────────────────────
# ── Model registry ────────────────────────────────────────────────────
# Ordered: no-graph baselines → graph baselines → DFC-GNN (contribution)
MODELS   = [
    "gru",             # PerNodeGRU — temporal lower bound
    "lstm",            # PerNodeLSTM — temporal lower bound
    "st_gnn_static",   # STGNNFlood (static graph)
    "st_gnn_sar",      # STGNNFlood + SAR embedding
    "st_gnn_dyn_edge", # STGNNDynEdge — dynamic edge weights
    "st_gnn_hand_edge",# STGNNHANDEdge — HAND-threshold topology
    "dfc_gnn",         # DFC-GNN — physically-biased dynamic attention
]
SEEDS    = [42, 123, 456]
HORIZONS = [4, 12, 16]          # T_out steps at 15-min → 1hr, 3hr, 4hr
HZ_LABEL = {4: "1 hr", 12: "3 hr", 16: "4 hr"}

# No-graph baselines: warm colours
# Graph baselines: blue family
# DFC-GNN (primary contribution): gold — stands out visually
MODEL_COLORS = {
    "gru":             "#1D9E75",   # teal
    "lstm":            "#D85A30",   # orange-red
    "st_gnn_static":   "#185FA5",   # dark blue
    "st_gnn_sar":      "#4A90D9",   # medium blue
    "st_gnn_dyn_edge": "#7B68EE",   # slate blue
    "st_gnn_hand_edge":"#9B59B6",   # purple
    "dfc_gnn":         "#D4A017",   # gold — primary contribution
}
MODEL_LABELS = {
    "gru":             "GRU (no graph)",
    "lstm":            "LSTM (no graph)",
    "st_gnn_static":   "ST-GNN (static)",
    "st_gnn_sar":      "ST-GNN+SAR",
    "st_gnn_dyn_edge": "ST-GNN DynEdge",
    "st_gnn_hand_edge":"ST-GNN HAND",
    "dfc_gnn":         "DFC-GNN (proposed)",
}
MODEL_MARKERS = {
    "gru":             "o",
    "lstm":            "s",
    "st_gnn_static":   "^",
    "st_gnn_sar":      "D",
    "st_gnn_dyn_edge": "P",
    "st_gnn_hand_edge":"X",
    "dfc_gnn":         "*",
}
# Subset for legacy three-model comparisons (backward-compatible)
MODELS_CORE = ["gru", "lstm", "st_gnn_static"]

# Persistence baseline (constant across runs — same dataset, same mask)
PERSIST_NSE  = 0.9160
PERSIST_RMSE = 0.0732

# Nodes classified as having strong upstream connections (high accumulation)
UPSTREAM_NODES = {
    "Cooleen Bridge", "Ballyvourney", "Morris's Bridge",
    "Killaclug", "Dripsey Bridge", "Bawnafinny Bridge",
    "Coolmuckey Br", "Ovens Bridge",
}
ISOLATED_NODES = {
    "Ballincolly", "Glen Park", "Macroom Town Bridge",
    "Gothic Bridge", "Blackpool Retail Park", "Glennamought Bridge",
}
RESERVOIR_NODES = {
    "Cooldaniel", "Ovens Bridge", "Carrigadrohid Headrace",
    "Inniscarra Headrace", "Inniscarra Tailrace",
}


# ═══════════════════════════════════════════════════════════════════════
# 1.  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk the directory tree and load all test_metrics.json and
    per_node_metrics.csv files into two tidy DataFrames.

    Returns
    -------
    global_df   rows = (model, seed, horizon)
                cols = model, seed, horizon, rmse, mae, nse, mbe, test_loss
    node_df     rows = (model, seed, horizon, node)
                cols = model, seed, horizon + all per_node_metrics columns
    """
    global_rows, node_rows = [], []
    missing = []

    for model in MODELS:
        for seed in SEEDS:
            for hz in HORIZONS:
                d = CKPT_DIR / model / str(seed) / str(hz)
                tm_path = d / "test_metrics.json"
                pn_path = d / "per_node_metrics.csv"

                if not tm_path.exists() or not pn_path.exists():
                    missing.append(str(d))
                    continue

                with open(tm_path) as f:
                    tm = json.load(f)
                global_rows.append({
                    "model": model, "seed": seed, "horizon": hz,
                    "rmse":      tm.get("rmse",      np.nan),
                    "mae":       tm.get("mae",       np.nan),
                    "nse":       tm.get("nse",       np.nan),
                    "mbe":       tm.get("mbe",       np.nan),
                    "test_loss": tm.get("test_loss", np.nan),
                    # DFC-GNN extras (NaN for other models)
                    "flood_acc": tm.get("flood_acc", np.nan),
                    "n_edges":   tm.get("n_edges",   np.nan),
                })

                pn = pd.read_csv(pn_path)
                pn["model"]   = model
                pn["seed"]    = seed
                pn["horizon"] = hz
                node_rows.append(pn)

    if missing:
        print(f"[warn] {len(missing)} missing directories:")
        for m in missing[:6]:
            print(f"         {m}")

    global_df = pd.DataFrame(global_rows)
    node_df   = pd.concat(node_rows, ignore_index=True) if node_rows else pd.DataFrame()

    # ── Per-step metrics (per_step_metrics.json) ───────────────────────
    step_rows = []
    for model in MODELS:
        for seed in SEEDS:
            for hz in HORIZONS:
                d  = CKPT_DIR / model / str(seed) / str(hz)
                ps = d / "per_step_metrics.json"
                if not ps.exists():
                    continue
                with open(ps) as f:
                    steps = json.load(f)
                for s in steps:
                    step_rows.append({
                        "model": model, "seed": seed, "horizon": hz,
                        **s,
                    })
    step_df = pd.DataFrame(step_rows)
    return global_df, node_df, step_df


# ═══════════════════════════════════════════════════════════════════════
# 2.  Summary statistics
# ═══════════════════════════════════════════════════════════════════════

def summary_table(global_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± std over seeds for each (model, horizon).
    Also computes skill score vs persistence.
    """
    rows = []
    available_models = [m for m in MODELS
                        if m in global_df["model"].unique()]
    for model in available_models:
        for hz in HORIZONS:
            sub = global_df[(global_df.model == model) &
                            (global_df.horizon == hz)]
            if sub.empty:
                continue
            nse_mean  = sub.nse.mean()
            nse_std   = sub.nse.std(ddof=1)
            rmse_mean = sub.rmse.mean()
            rmse_std  = sub.rmse.std(ddof=1)
            skill     = (nse_mean - PERSIST_NSE) / (1 - PERSIST_NSE)
            row = {
                "model":      MODEL_LABELS.get(model, model),
                "horizon":    HZ_LABEL[hz],
                "NSE mean":   round(nse_mean, 4),
                "NSE std":    round(nse_std, 4),
                "RMSE mean":  round(rmse_mean, 4),
                "RMSE std":   round(rmse_std, 4),
                "Skill":      round(skill, 4),
                "MBE mean":   round(sub.mbe.mean(), 4),
            }
            if "flood_acc" in sub.columns and not sub.flood_acc.isna().all():
                row["flood_acc"] = round(sub.flood_acc.mean(), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_gru_vs_stgnn(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wilcoxon signed-rank tests across all model pairs, per horizon.

    Compares each model against GRU (no-graph baseline) and includes
    the key DFC-GNN vs ST-GNN (static) comparison.

    The Wilcoxon signed-rank test (Wilcoxon 1945, Biometrics 1(6)) is
    appropriate here because:
    - Paired observations (same node, same seed, aligned by index)
    - NSE differences are asymmetric and bounded above at 1.0
    - n=27 nodes × 3 seeds = 81 paired observations per comparison
    """
    # Comparison pairs: all baselines vs GRU + DFC-GNN vs each baseline
    COMPARE_PAIRS = [
        ("gru",   "st_gnn_static",   "GRU vs ST-GNN(static)"),
        ("gru",   "st_gnn_sar",      "GRU vs ST-GNN+SAR"),
        ("gru",   "st_gnn_dyn_edge", "GRU vs DynEdge"),
        ("gru",   "st_gnn_hand_edge","GRU vs HAND"),
        ("gru",   "dfc_gnn",         "GRU vs DFC-GNN"),
        ("st_gnn","dfc_gnn",         "ST-GNN(static) vs DFC-GNN"),
    ]
    avail = node_df["model"].unique()
    rows  = []
    for hz in HORIZONS:
        row = {"Horizon": HZ_LABEL[hz]}
        for m_a, m_b, label in COMPARE_PAIRS:
            if m_a not in avail or m_b not in avail:
                row[label] = "n/a"
                continue
            nse_a = node_df[(node_df.model==m_a) & (node_df.horizon==hz)]["nse"].values
            nse_b = node_df[(node_df.model==m_b) & (node_df.horizon==hz)]["nse"].values
            n = min(len(nse_a), len(nse_b))
            if n < 8:
                row[label] = "n/a (too few)"
                continue
            nse_a, nse_b = nse_a[:n], nse_b[:n]
            try:
                _, p = stats.wilcoxon(nse_a, nse_b, alternative="two-sided")
                winner = MODEL_LABELS.get(m_b, m_b) \
                         if nse_b.mean() > nse_a.mean() \
                         else MODEL_LABELS.get(m_a, m_a)
                row[label] = (f"p={p:.4f}{"*" if p<0.05 else ""}"
                              f" [{winner} wins ΔNSE={abs(nse_b.mean()-nse_a.mean()):.4f}]")
            except Exception:
                row[label] = "error"
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_horizon_curves(global_df: pd.DataFrame) -> None:
    """
    NSE and RMSE vs forecast horizon (1hr / 3hr / 4hr), mean ± std
    across seeds for each model.  Persistence shown as dashed baseline.

    Rationale: Lees et al. (2022) Hydrology and Earth System Sciences
    use this exact plot design to demonstrate horizon-dependent skill.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    hz_vals = sorted(HORIZONS)
    hz_ticks = [HZ_LABEL[h] for h in hz_vals]

    avail = global_df["model"].unique()
    for model in [m for m in MODELS if m in avail]:
        nse_means, nse_stds = [], []
        rmse_means, rmse_stds = [], []
        for hz in hz_vals:
            sub = global_df[(global_df.model == model) &
                            (global_df.horizon == hz)]["nse"]
            nse_means.append(sub.mean())
            nse_stds.append(sub.std(ddof=1) if len(sub) > 1 else 0)
            sub_r = global_df[(global_df.model == model) &
                              (global_df.horizon == hz)]["rmse"]
            rmse_means.append(sub_r.mean())
            rmse_stds.append(sub_r.std(ddof=1) if len(sub_r) > 1 else 0)

        c = MODEL_COLORS[model]
        mk = MODEL_MARKERS[model]
        ax1.errorbar(hz_ticks, nse_means, yerr=nse_stds,
                     color=c, marker=mk, ms=7, lw=2, capsize=4,
                     label=MODEL_LABELS[model])
        ax2.errorbar(hz_ticks, rmse_means, yerr=rmse_stds,
                     color=c, marker=mk, ms=7, lw=2, capsize=4,
                     label=MODEL_LABELS[model])

    ax1.axhline(PERSIST_NSE, color="#888780", lw=1.2, ls="--",
                label=f"Persistence ({PERSIST_NSE})")
    ax1.set_xlabel("Forecast horizon", fontsize=10)
    ax1.set_ylabel("NSE (mean ± std, 3 seeds)", fontsize=10)
    ax1.set_title("NSE vs forecast horizon", fontsize=11)
    ax1.legend(fontsize=8.5)
    ax1.tick_params(labelsize=9)

    ax2.axhline(PERSIST_RMSE, color="#888780", lw=1.2, ls="--",
                label=f"Persistence ({PERSIST_RMSE} m)")
    ax2.set_xlabel("Forecast horizon", fontsize=10)
    ax2.set_ylabel("RMSE m (mean ± std, 3 seeds)", fontsize=10)
    ax2.set_title("RMSE vs forecast horizon", fontsize=11)
    ax2.legend(fontsize=8.5)
    ax2.tick_params(labelsize=9)

    fig.suptitle("Forecast skill degradation with horizon  |  3 seeds × 3 models",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "horizon_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: horizon_curves.png")


def plot_node_advantage(node_df: pd.DataFrame, horizon: int = 4) -> None:
    """
    For each node, compute mean NSE across seeds for GRU and ST-GNN.
    Show Δ NSE (ST-GNN − GRU) as a horizontal bar chart.
    Colour positive = ST-GNN wins, negative = GRU wins.

    Nodes sorted by Δ NSE so the spatial pattern is visible.
    """
    # avail_m = node_df["model"].unique()
    # if "gru" not in avail_m or "st_gnn" not in avail_m:
    #     print(f"  [skip] node_advantage_hz{horizon}: "
    #           f"needs gru + st_gnn, missing: "
    #           f"{[m for m in ['gru','st_gnn'] if m not in avail_m]}")
    #     return
    gru = (node_df[(node_df.model == "gru") & (node_df.horizon == horizon)]
           .groupby("name")["nse"].mean().rename("gru"))
    gnn = (node_df[(node_df.model == "st_gnn") & (node_df.horizon == horizon)]
           .groupby("name")["nse"].mean().rename("stgnn"))

    df = pd.concat([gru, gnn], axis=1).dropna()
    df["delta"] = df["stgnn"] - df["gru"]
    df = df.sort_values("delta")

    colors = ["#1D9E75" if d >= 0 else "#D85A30" for d in df["delta"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.37)))
    fig.patch.set_facecolor("white")
    bars = ax.barh(df.index, df["delta"], color=colors,
                   height=0.7, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#444441", lw=0.8)
    ax.set_xlabel("ΔNSE (ST-GNN − GRU per-node)  |  positive = ST-GNN wins",
                  fontsize=9)
    ax.set_title(f"Per-node NSE advantage of ST-GNN over GRU\n"
                 f"Horizon T_out={horizon} ({HZ_LABEL[horizon]})  |  "
                 f"mean over 3 seeds", fontsize=10)
    ax.tick_params(labelsize=8.5)

    # Value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["delta"] + (0.0002 if row["delta"] >= 0 else -0.0002),
                i, f"{row['delta']:+.4f}",
                va="center", ha="left" if row["delta"] >= 0 else "right",
                fontsize=7)

    # Zone annotations
    if df.empty or df["delta"].isna().all():
        plt.close(fig)
        print(f"  [skip] node_advantage_hz{horizon}: "
              f"both models needed but one is missing")
        return
    xmax = df["delta"].abs().max() * 1.4
    if not np.isfinite(xmax) or xmax == 0:
        xmax = 0.01
    ax.set_xlim(-xmax, xmax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"node_advantage_hz{horizon}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: node_advantage_hz{horizon}.png")


def plot_skill_violin(node_df: pd.DataFrame) -> None:
    """
    Violin plot of per-node skill scores across all seeds and horizons.
    Separates upstream-connected, isolated, and reservoir node groups.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    fig.patch.set_facecolor("white")
    groups = [
        ("Upstream-connected nodes", UPSTREAM_NODES),
        ("Isolated / low-coverage nodes", ISOLATED_NODES),
        ("Reservoir nodes", RESERVOIR_NODES),
    ]

    for ax, (title, node_set) in zip(axes, groups):
        data_by_model = []
        labels = []
        avail_m = [m for m in MODELS if m in node_df["model"].unique()]
        for model in avail_m:
            sub = node_df[(node_df.model == model) &
                          (node_df["name"].isin(node_set))]["skill"].dropna()
            data_by_model.append(sub.values)
            labels.append(MODEL_LABELS.get(model, model))

        parts = ax.violinplot(data_by_model, positions=range(len(MODELS)),
                              showmedians=True, showextrema=True)
        for i, (pc, model) in enumerate(zip(parts["bodies"], MODELS)):
            pc.set_facecolor(MODEL_COLORS[model])
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("#2C2C2A")
        parts["cbars"].set_color("#888780")
        parts["cmins"].set_color("#888780")
        parts["cmaxes"].set_color("#888780")

        ax.axhline(0, color="#D85A30", lw=1.0, ls="--", alpha=0.8)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(labelsize=8.5)

    axes[0].set_ylabel("Skill score vs persistence", fontsize=9)
    fig.suptitle("Per-node skill score distribution by node type\n"
                 "All seeds × all horizons  |  dashed = 0 (persistence level)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "skill_violin.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: skill_violin.png")

def plot_seed_stability(global_df: pd.DataFrame) -> None:
    """
    NSE across seeds for each model at T_out=4 (1hr horizon).
    Shows run-to-run variance — essential for judging whether global
    metric differences are meaningful or within noise.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    hz = 4
    avail_s = global_df["model"].unique()
    for model in [m for m in MODELS if m in avail_s]:
        sub = global_df[(global_df.model == model) &
                        (global_df.horizon == hz)]
        nse_vals = [sub[sub.seed == s]["nse"].values[0]
                    if len(sub[sub.seed == s]) else np.nan
                    for s in SEEDS]
        ax.plot(range(len(SEEDS)), nse_vals,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model], ms=9, lw=1.8,
                label=MODEL_LABELS[model])
        for i, v in enumerate(nse_vals):
            if not np.isnan(v):
                ax.text(i, v + 0.0003, f"{v:.4f}",
                        ha="center", fontsize=7.5, color=MODEL_COLORS[model])

    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=9)
    ax.set_ylabel("Test NSE", fontsize=10)
    ax.set_title(f"Run-to-run NSE stability across 3 seeds  |  "
                 f"T_out={hz} ({HZ_LABEL[hz]})", fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8.5)

    # Annotate the spread
    for model in MODELS:
        sub = global_df[(global_df.model == model) & (global_df.horizon == hz)]
        std = sub["nse"].std(ddof=1)
        ax.annotate(f"σ={std:.4f}",
                    xy=(2.1, sub["nse"].mean()),
                    fontsize=7.5, color=MODEL_COLORS[model], va="center")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "seed_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: seed_stability.png")


def _find_nse_col(h: pd.DataFrame, path: Path) -> tuple:
    """
    Resolve the validation NSE column from a training_history.csv.

    Training scripts vary in what they log. Priority order:
      1. val_nse       — explicit NSE column (preferred)
      2. nse           — some scripts use this name
      3. val_loss      — fallback: convert loss to pseudo-NSE so the
                         curve shape is preserved even without NSE values.
                         val_loss is negated and shifted to [0, 1] range.

    Returns (values_array, y_label, used_fallback).
    """
    for candidate in ("val_nse", "nse", "val_metric"):
        if candidate in h.columns:
            return h[candidate].values, "Validation NSE", False

    # Fallback: normalise val_loss to a 0→1 scale (inverted, so higher=better)
    if "val_loss" in h.columns:
        vl = h["val_loss"].values.astype(float)
        lo, hi = np.nanmin(vl), np.nanmax(vl)
        if hi > lo:
            normed = 1.0 - (vl - lo) / (hi - lo)
        else:
            normed = np.zeros_like(vl)
        print(f"  [warn] no NSE column in {path.parent.name}/"
              f"{path.name} — plotting normalised val_loss")
        return normed, "Normalised val_loss (proxy, not NSE)", True

    print(f"  [warn] no usable column in {path} — skipping")
    return None, None, False


def plot_training_curves_overlay(global_df, horizon: int = 4) -> None:
    """
    Validation NSE curves for all models and seeds at a given horizon.
    Thin lines = individual seeds, thick line = mean across seeds.

    Column resolution order: val_nse → nse → normalised val_loss.
    If your training script logs a different column name, add it to
    _find_nse_col() above.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    y_label   = "Validation NSE"
    any_drawn = False

    avail_tc = global_df["model"].unique() if not global_df.empty else []
    for model in [m for m in MODELS if m in avail_tc]:
        all_vals = []
        max_ep   = 0
        for seed in SEEDS:
            path = CKPT_DIR / model / str(seed) / str(horizon) / "training_history.csv"
            if not path.exists():
                print(f"  [warn] missing: {path}")
                continue
            h = pd.read_csv(path)
            vals, col_label, fallback = _find_nse_col(h, path)
            if vals is None:
                continue
            if fallback:
                y_label = col_label
            all_vals.append(vals)
            max_ep = max(max_ep, len(vals))
            ax.plot(range(1, len(vals) + 1), vals,
                    color=MODEL_COLORS[model], lw=0.7, alpha=0.35)
            any_drawn = True

        if all_vals:
            padded = np.full((len(all_vals), max_ep), np.nan)
            for i, arr in enumerate(all_vals):
                padded[i, :len(arr)] = arr
            mean_vals = np.nanmean(padded, axis=0)
            ax.plot(range(1, max_ep + 1), mean_vals,
                    color=MODEL_COLORS[model], lw=2.5,
                    label=MODEL_LABELS[model])

    if not any_drawn:
        ax.text(0.5, 0.5,
                "No data found.\nCheck that training_history.csv contains\n"
                "a val_nse or val_loss column.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#D85A30")
    else:
        # Only draw persistence baseline when y-axis is genuine NSE
        if "NSE" in y_label:
            ax.axhline(PERSIST_NSE, color="#888780", lw=1.2, ls="--",
                       label=f"Persistence ({PERSIST_NSE})")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"Training convergence  |  T_out={horizon} ({HZ_LABEL[horizon]})\n"
                 f"Thin = individual seeds, thick = mean across 3 seeds",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"training_curves_hz{horizon}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: training_curves_hz{horizon}.png")


def plot_per_step_advantage(step_df: pd.DataFrame) -> None:
    """
    For each T_out configuration, plot NSE at every individual forecast
    step h+1 … h+T_out for all three models (mean ± std across seeds).

    This directly tests Gao et al. (2022): graph models should show a
    growing advantage over the no-graph GRU baseline as h increases,
    because upstream flood wave propagation takes time — the graph edge
    weights encode routing distance and the ST-GNN can exploit upstream
    readings more effectively at longer leads.

    A pattern where ST-GNN ≈ GRU at h+1 but ST-GNN > GRU at h+3/h+4
    is the mechanistically correct signature of spatial value in the
    graph, even if the globally-aggregated NSE shows no difference.
    """
    if step_df.empty:
        print("  [skip] per_step_metrics.json not found in any directory.")
        print("         Re-run train_model.py to generate per-step metrics.")
        return

    for hz in HORIZONS:
        sub = step_df[step_df["horizon"] == hz]
        if sub.empty:
            continue

        steps_present = sorted(sub["step"].unique())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("white")

        for model in MODELS:
            nse_means, nse_stds = [], []
            skill_means, skill_stds = [], []
            for h in steps_present:
                row = sub[(sub["model"] == model) & (sub["step"] == h)]
                nse_means.append(row["nse"].mean())
                nse_stds.append(row["nse"].std(ddof=1) if len(row) > 1 else 0)
                if "skill" in row.columns:
                    skill_means.append(row["skill"].mean())
                    skill_stds.append(row["skill"].std(ddof=1) if len(row) > 1 else 0)

            labels = [f"h+{s}" for s in steps_present]
            c  = MODEL_COLORS[model]
            mk = MODEL_MARKERS[model]

            ax1.errorbar(labels, nse_means, yerr=nse_stds,
                         color=c, marker=mk, ms=7, lw=2, capsize=4,
                         label=MODEL_LABELS[model])
            if skill_means:
                ax2.errorbar(labels, skill_means, yerr=skill_stds,
                             color=c, marker=mk, ms=7, lw=2, capsize=4,
                             label=MODEL_LABELS[model])

        ax1.axhline(PERSIST_NSE, color="#888780", lw=1.2, ls="--",
                    label=f"Persistence ({PERSIST_NSE})")
        ax1.set_xlabel("Forecast step", fontsize=10)
        ax1.set_ylabel("NSE (mean ± std, 3 seeds)", fontsize=10)
        ax1.set_title(f"NSE per forecast step  |  T_out={hz} ({HZ_LABEL[hz]})",
                      fontsize=11)
        ax1.legend(fontsize=8.5)
        ax1.tick_params(labelsize=9)

        ax2.axhline(0, color="#888780", lw=1.0, ls="--", alpha=0.7)
        ax2.set_xlabel("Forecast step", fontsize=10)
        ax2.set_ylabel("Skill score vs persistence", fontsize=10)
        ax2.set_title(f"Skill per forecast step  |  T_out={hz} ({HZ_LABEL[hz]})",
                      fontsize=11)
        ax2.legend(fontsize=8.5)
        ax2.tick_params(labelsize=9)

        fig.suptitle(
            f"Per-step horizon analysis  |  T_out={hz} ({HZ_LABEL[hz]}) | "
            f"ST-GNN advantage should grow with step index (Gao et al. 2022)",
            fontsize=10, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"per_step_hz{hz}.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: per_step_hz{hz}.png")


# ═══════════════════════════════════════════════════════════════════════
# 4.  Print results and conclusion
# ═══════════════════════════════════════════════════════════════════════

def print_summary(summ: pd.DataFrame, wilcox: pd.DataFrame) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("GLOBAL METRICS  (mean ± std across 3 seeds)")
    print(sep)
    print(summ.to_string(index=False))

    print(f"\n{sep}")
    print("WILCOXON SIGNED-RANK TEST  (per-node NSE, 27 nodes × 3 seeds)")
    print("* = p < 0.05  (statistically significant difference)")
    print(sep)
    print(wilcox.to_string(index=False))


def print_conclusion(global_df: pd.DataFrame,
                     node_df: pd.DataFrame) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("CONCLUSION")
    print(sep)

    avail_m = global_df["model"].unique()
    for hz in HORIZONS:
        gru_nse  = global_df[(global_df.model == "gru")    & (global_df.horizon == hz)]["nse"]
        gnn_nse  = global_df[(global_df.model == "st_gnn") & (global_df.horizon == hz)]["nse"]
        lstm_nse = global_df[(global_df.model == "lstm")   & (global_df.horizon == hz)]["nse"]
        dfc_nse  = global_df[(global_df.model == "dfc_gnn") & (global_df.horizon == hz)]["nse"]
        print(f"\nHorizon {HZ_LABEL[hz]} (T_out={hz}):")
        for m_tag, m_series in [
            ("gru",    gru_nse), ("lstm", lstm_nse), ("st_gnn", gnn_nse),
            ("dfc_gnn", dfc_nse),
        ]:
            if m_tag not in avail_m or m_series.empty: continue
            lbl = MODEL_LABELS.get(m_tag, m_tag)
            std = m_series.std(ddof=1) if len(m_series) > 1 else 0
            print(f"  {lbl:<26} NSE {m_series.mean():.4f} ± {std:.4f}")

        # Check if GRU-GNN difference is within variance
        gru_std  = gru_nse.std(ddof=1)
        gnn_std  = gnn_nse.std(ddof=1)
        delta    = abs(gru_nse.mean() - gnn_nse.mean())
        noise    = max(gru_std, gnn_std)
        if delta < noise:
            print(f"  → ΔNSE={delta:.4f} < max(σ)={noise:.4f}: "
                  f"GRU vs ST-GNN NOT distinguishable beyond run variance")
        else:
            winner = "GRU" if gru_nse.mean() > gnn_nse.mean() else "ST-GNN"
            print(f"  → ΔNSE={delta:.4f} > max(σ)={noise:.4f}: "
                  f"{winner} is consistently better at this horizon")

        # Node-type breakdown
        for label, node_set in [
            ("Upstream-connected", UPSTREAM_NODES),
            ("Isolated/sparse",    ISOLATED_NODES),
        ]:
            gru_n = node_df[(node_df.model=="gru") &
                            (node_df.horizon==hz) &
                            (node_df["name"].isin(node_set))]["nse"].mean()
            gnn_n = node_df[(node_df.model=="st_gnn") &
                            (node_df.horizon==hz) &
                            (node_df["name"].isin(node_set))]["nse"].mean()
            sign = ">" if gnn_n > gru_n else "<"
            print(f"  {label}: ST-GNN NSE {gnn_n:.4f} {sign} GRU {gru_n:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def plot_dfc_vs_best_baseline(node_df: pd.DataFrame, horizon: int = 4) -> None:
    """
    For each node, show ΔNSE between DFC-GNN and the best-performing
    graph baseline (whichever of ST-GNN, SAR, DynEdge, HAND has highest
    mean NSE for that node).  This isolates the contribution of the
    physically-biased dynamic attention over the previous best method.
    """
    if "dfc_gnn" not in node_df["model"].unique():
        print("  [skip] dfc_gnn not in results yet")
        return

    dfc_nse = (node_df[(node_df.model == "dfc_gnn") & (node_df.horizon == horizon)]
               .groupby("name")["nse"].mean().rename("dfc"))

    # Best among static graph baselines
    best_nse = None
    for m in ["st_gnn_hand_edge", "st_gnn_dyn_edge", "st_gnn_sar", "st_gnn"]:
        if m not in node_df["model"].unique():
            continue
        mn = (node_df[(node_df.model == m) & (node_df.horizon == horizon)]
              .groupby("name")["nse"].mean())
        best_nse = mn if best_nse is None else best_nse.combine(mn, max)

    if best_nse is None:
        print("  [skip] no graph baseline results found")
        return

    df = pd.concat([dfc_nse, best_nse.rename("best_baseline")], axis=1).dropna()
    df["delta"] = df["dfc"] - df["best_baseline"]
    df = df.sort_values("delta")

    colors = ["#D4A017" if d >= 0 else "#888780" for d in df["delta"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.37)))
    fig.patch.set_facecolor("white")
    ax.barh(df.index, df["delta"], color=colors, height=0.7,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#444441", lw=0.8)
    ax.set_xlabel("ΔNSE (DFC-GNN − best graph baseline)  |  gold = DFC-GNN wins",
                  fontsize=9)
    ax.set_title(f"DFC-GNN advantage over best graph baseline\n"
                 f"Horizon T_out={horizon} ({HZ_LABEL[horizon]})  |  "
                 f"mean over 3 seeds", fontsize=10)
    ax.tick_params(labelsize=8.5)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["delta"] + (0.0002 if row["delta"] >= 0 else -0.0002),
                i, f"{row['delta']:+.4f}",
                va="center", ha="left" if row["delta"] >= 0 else "right",
                fontsize=7)
    xmax = df["delta"].abs().max() * 1.4
    if not np.isfinite(xmax) or xmax == 0:
        xmax = 0.01
    ax.set_xlim(-xmax, xmax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"dfc_advantage_hz{horizon}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: dfc_advantage_hz{horizon}.png")


def plot_flood_acc_comparison(global_df: pd.DataFrame) -> None:
    """
    Bar chart comparing flood_acc (node-level flood classification accuracy)
    across all models that report it, at each horizon.
    Only DFC-GNN reports flood_acc; other models show NaN → grey bar at 0.
    This plot documents the contribution of the auxiliary flood head.
    """
    if "flood_acc" not in global_df.columns:
        return
    dfc = global_df[global_df.model == "dfc_gnn"]
    if dfc.empty or dfc.flood_acc.isna().all():
        print("  [skip] flood_acc not yet available")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    x   = np.arange(len(HORIZONS))
    w   = 0.5
    means = [dfc[dfc.horizon == hz]["flood_acc"].mean() for hz in HORIZONS]
    stds  = [dfc[dfc.horizon == hz]["flood_acc"].std(ddof=1)
             if len(dfc[dfc.horizon == hz]) > 1 else 0 for hz in HORIZONS]
    ax.bar(x, means, width=w, color=MODEL_COLORS["dfc_gnn"],
           yerr=stds, capsize=5, alpha=0.85,
           label="DFC-GNN flood_acc", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([HZ_LABEL[h] for h in HORIZONS])
    ax.set_ylim(0.95, 1.0)
    ax.set_ylabel("Node-level flood classification accuracy", fontsize=9)
    ax.set_title("DFC-GNN auxiliary flood head accuracy\n"
                 "Mean ± std across 3 seeds  |  "
                 "1.0 = perfect node-level flood/no-flood classification",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "flood_acc_dfc_gnn.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: flood_acc_dfc_gnn.png")


def plot_all_models_nse_table(global_df: pd.DataFrame) -> None:
    """
    Heatmap table of NSE mean across all 7 models × 3 horizons.
    Darker colour = higher NSE.  Provides a quick full-comparison view
    for the thesis figures section.
    """
    avail = [m for m in MODELS if m in global_df["model"].unique()]
    data  = np.full((len(avail), len(HORIZONS)), np.nan)
    for i, m in enumerate(avail):
        for j, hz in enumerate(HORIZONS):
            sub = global_df[(global_df.model == m) & (global_df.horizon == hz)]["nse"]
            if not sub.empty:
                data[i, j] = sub.mean()

    fig, ax = plt.subplots(figsize=(7, max(4, len(avail) * 0.6)))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0.88, vmax=1.0)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([HZ_LABEL[h] for h in HORIZONS], fontsize=9)
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in avail], fontsize=9)
    for i in range(len(avail)):
        for j in range(len(HORIZONS)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.4f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if data[i, j] > 0.97 else "black")
    plt.colorbar(im, ax=ax, label="NSE", shrink=0.8)
    ax.set_title("Test NSE — all models × horizons  (mean over 3 seeds)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nse_heatmap_all_models.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: nse_heatmap_all_models.png")


def main():
    print("Loading experiment results …")
    global_df, node_df, step_df = load_all()
    print(f"  Loaded {len(global_df)} global records, "
          f"{len(node_df)} node records")

    if global_df.empty:
        print("[ERROR] No data loaded — check CKPT_DIR path:")
        print(f"  {CKPT_DIR}")
        return

    print("\nComputing summary statistics …")
    summ   = summary_table(global_df)
    wilcox = wilcoxon_gru_vs_stgnn(node_df)
    print_summary(summ, wilcox)

    print("\nGenerating figures …")
    plot_all_models_nse_table(global_df)
    plot_horizon_curves(global_df)
    plot_seed_stability(global_df)
    plot_per_step_advantage(step_df)
    for hz in HORIZONS:
        plot_node_advantage(node_df, hz)
        plot_dfc_vs_best_baseline(node_df, hz)
    plot_skill_violin(node_df)
    plot_flood_acc_comparison(global_df)
    for hz in HORIZONS:
        plot_training_curves_overlay(global_df, hz)

    print_conclusion(global_df, node_df)

    # Save summary tables to CSV for thesis
    summ.to_csv(OUT_DIR / "global_metrics_summary.csv", index=False)
    wilcox.to_csv(OUT_DIR / "wilcoxon_results.csv", index=False)
    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
