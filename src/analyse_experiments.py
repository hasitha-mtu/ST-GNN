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
MODELS   = ["gru", "lstm", "st_gnn"]
SEEDS    = [42, 123, 456]
HORIZONS = [4, 12, 16]          # T_out steps at 15-min → 1hr, 3hr, 4hr
HZ_LABEL = {4: "1 hr", 12: "3 hr", 16: "4 hr"}

MODEL_COLORS  = {"gru": "#1D9E75", "lstm": "#D85A30", "st_gnn": "#185FA5"}
MODEL_LABELS  = {"gru": "GRU (no graph)", "lstm": "LSTM (no graph)",
                 "st_gnn": "ST-GNN"}
MODEL_MARKERS = {"gru": "o", "lstm": "s", "st_gnn": "^"}

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
                    "rmse": tm.get("rmse", np.nan),
                    "mae":  tm.get("mae",  np.nan),
                    "nse":  tm.get("nse",  np.nan),
                    "mbe":  tm.get("mbe",  np.nan),
                    "test_loss": tm.get("test_loss", np.nan),
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
    return global_df, node_df


# ═══════════════════════════════════════════════════════════════════════
# 2.  Summary statistics
# ═══════════════════════════════════════════════════════════════════════

def summary_table(global_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± std over seeds for each (model, horizon).
    Also computes skill score vs persistence.
    """
    rows = []
    for model in MODELS:
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
            rows.append({
                "model":     MODEL_LABELS[model],
                "horizon":   HZ_LABEL[hz],
                "NSE mean":  round(nse_mean, 4),
                "NSE std":   round(nse_std, 4),
                "RMSE mean": round(rmse_mean, 4),
                "RMSE std":  round(rmse_std, 4),
                "Skill":     round(skill, 4),
                "MBE mean":  round(sub.mbe.mean(), 4),
            })
    return pd.DataFrame(rows)


def wilcoxon_gru_vs_stgnn(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test comparing per-node NSE between GRU and
    ST-GNN across all seeds and nodes, per horizon.

    The Wilcoxon signed-rank test (Wilcoxon 1945, Biometrics 1(6)) is
    used here because:
    - We are comparing paired observations (same node, same seed)
    - The distribution of NSE differences is unlikely to be normal
      (asymmetric, bounded above at 1.0)
    - It is the non-parametric equivalent of the paired t-test and is
      appropriate for n=27 nodes × 3 seeds = 81 paired observations
    """
    rows = []
    for hz in HORIZONS:
        gru_nse  = node_df[(node_df.model == "gru")    & (node_df.horizon == hz)]["nse"].values
        gnn_nse  = node_df[(node_df.model == "st_gnn") & (node_df.horizon == hz)]["nse"].values
        lstm_nse = node_df[(node_df.model == "lstm")   & (node_df.horizon == hz)]["nse"].values

        n = min(len(gru_nse), len(gnn_nse), len(lstm_nse))
        if n < 8:
            continue

        gru_nse  = gru_nse[:n]
        gnn_nse  = gnn_nse[:n]
        lstm_nse = lstm_nse[:n]

        stat_gru_gnn,  p_gru_gnn  = stats.wilcoxon(gru_nse,  gnn_nse,  alternative="two-sided")
        stat_gru_lstm, p_gru_lstm = stats.wilcoxon(gru_nse,  lstm_nse, alternative="two-sided")
        stat_gnn_lstm, p_gnn_lstm = stats.wilcoxon(gnn_nse,  lstm_nse, alternative="two-sided")

        rows.append({
            "Horizon":            HZ_LABEL[hz],
            "GRU mean NSE":       round(gru_nse.mean(), 4),
            "ST-GNN mean NSE":    round(gnn_nse.mean(), 4),
            "LSTM mean NSE":      round(lstm_nse.mean(), 4),
            "p(GRU vs ST-GNN)":   f"{p_gru_gnn:.4f}{'*' if p_gru_gnn < 0.05 else ''}",
            "p(GRU vs LSTM)":     f"{p_gru_lstm:.4f}{'*' if p_gru_lstm < 0.05 else ''}",
            "p(ST-GNN vs LSTM)":  f"{p_gnn_lstm:.4f}{'*' if p_gnn_lstm < 0.05 else ''}",
        })
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

    for model in MODELS:
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
    xmax = df["delta"].abs().max() * 1.4
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
        for model in MODELS:
            sub = node_df[(node_df.model == model) &
                          (node_df["name"].isin(node_set))]["skill"].dropna()
            data_by_model.append(sub.values)
            labels.append(MODEL_LABELS[model])

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
        ax.set_xticklabels(["GRU", "ST-GNN", "LSTM"], fontsize=9)
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
    for model in MODELS:
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
    # ['epoch', 'train_loss', 'val_loss', 'val_rmse', 'val_mae', 'val_nse']
    # for candidate in ('epoch', 'train_loss', 'val_loss', 'val_rmse', 'val_mae', 'val_nse'):
    #     if candidate in h.columns:
    #         print(f'_find_nse_col|candidate: {candidate}')
    #         return h[candidate].values, "Validation NSE", False
    #
    # # Fallback: normalise val_loss to a 0→1 scale (inverted, so higher=better)
    # if "val_loss" in h.columns:
    #     vl = h["val_loss"].values.astype(float)
    #     lo, hi = np.nanmin(vl), np.nanmax(vl)
    #     if hi > lo:
    #         normed = 1.0 - (vl - lo) / (hi - lo)
    #     else:
    #         normed = np.zeros_like(vl)
    #     print(f"  [warn] no NSE column in {path.parent.name}/"
    #           f"{path.name} — plotting normalised val_loss")
    #     return normed, "Normalised val_loss (proxy, not NSE)", True
    #
    # print(f"  [warn] no usable column in {path} — skipping")
    # return None, None, False

    return h['val_nse'].values, "Validation NSE", False


def plot_training_curves_overlay(horizon: int = 4) -> None:
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

    for model in MODELS:
        all_vals = []
        max_ep   = 0
        for seed in SEEDS:
            path = CKPT_DIR / model / str(seed) / str(horizon) / "training_history.csv"
            if not path.exists():
                print(f"  [warn] missing: {path}")
                continue
            h = pd.read_csv(path)
            if len(h) <= 1:
                print(f"  [warn] {path} has only {len(h)} row(s) — "
                      f"training script wrote to root checkpoints/ not subdir. "
                      f"Re-run training with --model/--seed/--t_out args to fix.")
                continue
            vals, col_label, fallback = _find_nse_col(h, path)
            if vals is None:
                print(f"  [warn] missing: {vals}")
                continue
            if fallback:
                y_label = col_label
            all_vals.append(vals)
            print(f' all_vals : {all_vals}')
            max_ep = max(max_ep, len(vals))
            ax.plot(range(1, len(vals) + 1), vals,
                    color=MODEL_COLORS[model], lw=0.7, alpha=0.35)
            any_drawn = True

        if all_vals:
            padded = np.full((len(all_vals), max_ep), np.nan)
            for i, arr in enumerate(all_vals):
                padded[i, :len(arr)] = arr
            mean_vals = np.nanmean(padded, axis=0)
            print(f' {model} : {mean_vals}')
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

    for hz in HORIZONS:
        gru_nse  = global_df[(global_df.model == "gru")    & (global_df.horizon == hz)]["nse"]
        gnn_nse  = global_df[(global_df.model == "st_gnn") & (global_df.horizon == hz)]["nse"]
        lstm_nse = global_df[(global_df.model == "lstm")   & (global_df.horizon == hz)]["nse"]
        print(f"\nHorizon {HZ_LABEL[hz]} (T_out={hz}):")
        print(f"  GRU    NSE {gru_nse.mean():.4f} ± {gru_nse.std(ddof=1):.4f}")
        print(f"  ST-GNN NSE {gnn_nse.mean():.4f} ± {gnn_nse.std(ddof=1):.4f}")
        print(f"  LSTM   NSE {lstm_nse.mean():.4f} ± {lstm_nse.std(ddof=1):.4f}")

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

def main():
    print("Loading experiment results …")
    global_df, node_df = load_all()
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
    plot_horizon_curves(global_df)
    plot_seed_stability(global_df)
    for hz in HORIZONS:
        plot_node_advantage(node_df, hz)
    plot_skill_violin(node_df)
    for hz in HORIZONS:
        plot_training_curves_overlay(hz)

    print_conclusion(global_df, node_df)

    # Save summary tables to CSV for thesis
    summ.to_csv(OUT_DIR / "global_metrics_summary.csv", index=False)
    wilcox.to_csv(OUT_DIR / "wilcoxon_results.csv", index=False)
    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    df = pd.read_csv("./../checkpoints/gru/42/4/training_history.csv", nrows=1)
    print(df.columns.tolist())
    main()
