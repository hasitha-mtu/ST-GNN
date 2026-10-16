"""
generator_sensitivity.py  —  Are the stress rankings artefacts of the generator?
=================================================================================
Reviewer request (Section 3.1): regenerate representative scenarios under
alternative generator assumptions, re-evaluate the EXISTING checkpoints (no
retraining), and report architecture mean ranks and rank agreement.

Default design
--------------
S3 (reservoir release, routed through the network):
    attenuation  {0.65, 0.75, 0.85} x lag_scale {0.75, 1.0, 1.25}
S4 (saturation breakthrough): S4 has NO network routing — every node is
    forced locally — so attenuation/lag cannot affect it. Its generator
    assumptions are the response timing and the pre-threshold runoff:
    tc_scale     {0.75, 1.0, 1.25} x s4_runoff_factor {0.2, 0.3, 0.45}
The (0.75, 1.0) / (1.0, 0.3) cells are the baseline configuration.
These ranges are perturbations for sensitivity testing, not values validated
for the River Lee.

Outputs (results/sensitivity/)
------------------------------
    sensitivity_scores.csv     model x scenario x config x horizon x seed scores
    sensitivity_ranks.csv      mean rank per model per config (1 = best)
    sensitivity_agreement.csv  Spearman rho and Kendall tau of each config's
                               model ranking vs the baseline ranking, plus
                               top-2 / bottom-2 membership
    sensitivity_ranks.png      heatmap of mean ranks (models x configs)
    sensitivity_summary.md     narrative-ready summary

Usage
-----
    python src/scenarios/generator_sensitivity.py                 # generate + evaluate + analyse
    python src/scenarios/generator_sensitivity.py --analyse-only  # reuse evaluation results
    python src/scenarios/generator_sensitivity.py --scenarios S3 --metric observable_nse_syn
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent.parent
for _p in (BASE_DIR, BASE_DIR / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SENS_SCEN = BASE_DIR / "dataset/scenarios_sensitivity"
SENS_RES = BASE_DIR / "results/sensitivity"

GRIDS = {
    "S3": {"axes": ("attenuation", "lag_scale"),
           "values": ((0.65, 0.75, 0.85), (0.75, 1.0, 1.25)),
           "baseline": (0.75, 1.0), "full": "S3_InniscarraRelease"},
    "S4": {"axes": ("tc_scale", "s4_runoff_factor"),
           "values": ((0.75, 1.0, 1.25), (0.2, 0.3, 0.45)),
           "baseline": (1.0, 0.3), "full": "S4_SatBreakthrough"},
    "S1": {"axes": ("attenuation", "lag_scale"),
           "values": ((0.65, 0.75, 0.85), (0.75, 1.0, 1.25)),
           "baseline": (0.75, 1.0), "full": "S1_ConvectiveCell"},
    "S5": {"axes": ("attenuation", "lag_scale"),
           "values": ((0.65, 0.75, 0.85), (0.75, 1.0, 1.25)),
           "baseline": (0.75, 1.0), "full": "S5_SpatialGradient"},
}
CLI_FLAG = {"attenuation": "--attenuation", "lag_scale": "--lag-scale",
            "tc_scale": "--tc-scale", "s4_runoff_factor": "--s4-runoff-factor"}


def config_tag(axes, vals) -> str:
    return "_".join(f"{a}{v:g}" for a, v in zip(axes, vals))


def generate_and_evaluate(sid: str, seed: int, n_windows: int | None = None) -> None:
    from scenarios import scenario_generator as sg
    from scenarios import scenario_evaluator as se
    g = GRIDS[sid]
    for vals in itertools.product(*g["values"]):
        tag = config_tag(g["axes"], vals)
        scen_root = SENS_SCEN / sid / tag
        args = ["--scenario", sid, "--out-root", str(scen_root), "--seed", str(seed)]
        if n_windows:
            args += ["--n-windows", str(n_windows)]
        for a, v in zip(g["axes"], vals):
            args += [CLI_FLAG[a], str(v)]
        if not (scen_root / g["full"] / "scenario_meta.json").exists():
            print(f"\n=== generate {sid} {tag}")
            sg.main(args)
        print(f"\n=== evaluate {sid} {tag}")
        se.main(["--scenario", g["full"], "--scen-root", str(scen_root),
                 "--results-root", str(SENS_RES / sid / tag)])


def collect(sid: str) -> pd.DataFrame:
    g = GRIDS[sid]
    frames = []
    for vals in itertools.product(*g["values"]):
        tag = config_tag(g["axes"], vals)
        f = SENS_RES / sid / tag / "scenario_summary.csv"
        if not f.exists():
            print(f"  [missing] {f}")
            continue
        d = pd.read_csv(f)
        d = d[d.scenario == g["full"]].copy()
        d["config"] = tag
        d["is_baseline"] = tuple(vals) == tuple(g["baseline"])
        for a, v in zip(g["axes"], vals):
            d[a] = v
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mean_ranks(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
    """Rank models within each (config, horizon, seed) cell, then average."""
    d = df.dropna(subset=[metric]).copy()
    d["rank"] = d.groupby(["config", "horizon", "seed"])[metric].rank(
        ascending=not higher_is_better, method="average")
    return d.groupby(["config", "model"])["rank"].mean().unstack("config")


def agreement(ranks: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    b = ranks[baseline]
    top_b = set(b.nsmallest(2).index); bot_b = set(b.nlargest(2).index)
    for c in ranks.columns:
        r = ranks[c]
        ok = b.notna() & r.notna()
        rho, p_rho = spearmanr(b[ok], r[ok])
        tau, p_tau = kendalltau(b[ok], r[ok])
        rows.append({"config": c, "spearman_rho": rho, "spearman_p": p_rho,
                     "kendall_tau": tau, "kendall_p": p_tau,
                     "top2": ",".join(r.nsmallest(2).index),
                     "top2_same_as_baseline": set(r.nsmallest(2).index) == top_b,
                     "bottom2_same_as_baseline": set(r.nlargest(2).index) == bot_b,
                     "n_models": int(ok.sum())})
    return pd.DataFrame(rows)


def plot_heatmap(ranks: pd.DataFrame, title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ranks.mean(axis=1).sort_values().index
    r = ranks.loc[order]
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * r.shape[1], 0.45 * r.shape[0] + 1.5))
    im = ax.imshow(r.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(r.shape[1])); ax.set_xticklabels(r.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(r.shape[0])); ax.set_yticklabels(r.index, fontsize=9)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if np.isfinite(r.values[i, j]):
                ax.text(j, i, f"{r.values[i, j]:.1f}", ha="center", va="center",
                        color="white" if r.values[i, j] > r.values[np.isfinite(r.values)].mean() else "black",
                        fontsize=7)
    fig.colorbar(im, ax=ax, label="mean rank (1 = best)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def analyse(sids, metric: str) -> None:
    SENS_RES.mkdir(parents=True, exist_ok=True)
    md = ["# Generator-sensitivity analysis\n",
          f"Metric: `{metric}` (higher is better); ranks computed within each "
          "(configuration, horizon, seed) cell and averaged.\n"]
    all_scores, all_ranks, all_agree = [], [], []
    for sid in sids:
        df = collect(sid)
        if df.empty:
            continue
        g = GRIDS[sid]
        base_tag = config_tag(g["axes"], g["baseline"])
        ranks = mean_ranks(df, metric)
        agr = agreement(ranks, base_tag)
        all_scores.append(df); all_ranks.append(ranks.assign(scenario=sid))
        all_agree.append(agr.assign(scenario=sid))
        plot_heatmap(ranks, f"{g['full']}: mean rank under generator perturbations",
                     SENS_RES / f"sensitivity_ranks_{sid}.png")
        md += [f"\n## {g['full']}\n", f"Axes: {g['axes'][0]} x {g['axes'][1]}; "
               f"baseline {base_tag}.\n",
               f"* Spearman rho vs baseline: min {agr.spearman_rho.min():.2f}, "
               f"median {agr.spearman_rho.median():.2f}",
               f"* Kendall tau vs baseline: min {agr.kendall_tau.min():.2f}",
               f"* Top-2 models identical to baseline in "
               f"{int(agr.top2_same_as_baseline.sum())}/{len(agr)} configurations",
               f"* Baseline top-2: {agr.loc[agr.config == base_tag, 'top2'].iloc[0]}\n",
               "| config | rho | tau | top-2 |", "|---|---|---|---|"]
        md += [f"| {r.config} | {r.spearman_rho:.2f} | {r.kendall_tau:.2f} | {r.top2} |"
               for r in agr.itertuples()]
    if not all_scores:
        print("No sensitivity results found."); return
    pd.concat(all_scores).to_csv(SENS_RES / "sensitivity_scores.csv", index=False)
    pd.concat(all_ranks).to_csv(SENS_RES / "sensitivity_ranks.csv")
    pd.concat(all_agree).to_csv(SENS_RES / "sensitivity_agreement.csv", index=False)
    (SENS_RES / "sensitivity_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["S3", "S4"], choices=list(GRIDS))
    ap.add_argument("--metric", default="observable_nse_syn")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=None)
    a = ap.parse_args(argv)
    if not a.analyse_only:
        for sid in a.scenarios:
            generate_and_evaluate(sid, a.seed, a.n_windows)
    analyse(a.scenarios, a.metric)


if __name__ == "__main__":
    main()
