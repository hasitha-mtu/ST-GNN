"""
scenario_repeated_measures_inference.py — reviewer point 3: statistical
inference for the scenario experiment, using realizations as the
independent sampling unit rather than treating correlated timesteps (or,
as the earlier check_dfc_scenario_rank_significance.py did, seeds paired
with a single deterministic scenario) as independent observations.

Reads results/scenarios/scenario_summary_per_realization.csv, produced
by scenario_evaluator.py once scenarios have been regenerated with the
multi-realization design (points 1+2). Each row is one
(scenario, model, seed, horizon, realization_id) -> nse_syn_realization.

Design:
  - Independent SUBJECT = one (seed, realization_id) pair. Seeds are
    genuinely independent (different random initialisation/training
    runs); realizations are genuinely independent (different sampled
    event parameters). Pairing at this level, rather than pooling all
    seeds x realizations as if they were 3x as many independent draws
    of the SAME event (the earlier script's limitation), is what makes
    this a proper repeated-measures design.
  - REPEATED/WITHIN-SUBJECT FACTOR = model. Every model is evaluated
    against the SAME set of (seed, realization_id) subjects (subjects
    missing a value for any model under test are dropped --
    listwise deletion, reported explicitly rather than silently).
  - Run separately per horizon and per scenario -- "horizon handled as
    a repeated factor" is satisfied by never pooling across horizons
    into one test; each horizon gets its own complete analysis.

Two-stage test, standard practice for k>2 repeated-measures comparisons:
  1. Friedman test (non-parametric repeated-measures ANOVA equivalent):
     is there ANY significant difference among the k models' ranks
     across subjects? Omnibus test, controls the overall false-positive
     rate before any pairwise comparison is attempted.
  2. If significant (p<0.05): post-hoc pairwise Wilcoxon signed-rank
     tests between every model pair, Holm-Bonferroni corrected for the
     resulting multiple comparisons (implemented directly -- statsmodels
     is not assumed available).

Bootstrap 95% CI on the median paired NSE difference is also reported
for any specific pairwise comparison requested via --compare.

Usage:
    python scenario_repeated_measures_inference.py --scenario S4_SatBreakthrough --horizon 24
    python scenario_repeated_measures_inference.py --scenario S4_SatBreakthrough --horizon 24 \
        --compare dfc_gnn gru
    python scenario_repeated_measures_inference.py --all
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

PER_REALIZATION_CSV = Path("results/scenarios/scenario_summary_per_realization.csv")
STEP_TO_HZ = {4: "1hr", 12: "3hr", 16: "4hr", 24: "6hr", 48: "12hr"}


def holm_bonferroni(raw_pvalues: list[float]) -> list[float]:
    """
    Standard Holm-Bonferroni step-down correction, implemented directly
    (statsmodels not assumed available on the target machine). For m
    tests sorted ascending by raw p-value, the i-th smallest (0-indexed)
    is multiplied by (m - i), then the sequence is made non-decreasing
    (enforced monotonicity, the standard Holm step-down guarantee) and
    clipped to 1.0.
    """
    m = len(raw_pvalues)
    order = np.argsort(raw_pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = raw_pvalues[idx] * (m - rank)
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def build_subject_matrix(df: pd.DataFrame, scenario: str, horizon: int,
                         models: list[str], value_col: str = "nse_syn_realization"
                         ) -> tuple[pd.DataFrame, int, int]:
    """
    Pivots to a (seed, realization_id) x model matrix for the requested
    scenario/horizon. Drops subjects missing a value for ANY of the
    requested models (listwise deletion). Returns the complete-case
    matrix plus (n_before, n_after) subject counts for transparency.
    """
    sub = df[(df.scenario == scenario) & (df.horizon == horizon) & (df.model.isin(models))]
    pivot = sub.pivot_table(index=["seed", "realization_id"], columns="model", values=value_col)
    n_before = len(pivot)
    pivot = pivot.dropna(subset=[m for m in models if m in pivot.columns])
    missing_models = [m for m in models if m not in pivot.columns]
    if missing_models:
        print(f"  [warn] no data at all for: {missing_models}")
        return pd.DataFrame(), n_before, 0
    n_after = len(pivot)
    return pivot[models], n_before, n_after


def run_friedman_and_posthoc(pivot: pd.DataFrame, models: list[str]) -> dict:
    if len(pivot) < 8:
        return {"n_subjects": len(pivot), "note": "too few subjects (<8) for a reliable Friedman test"}

    arrays = [pivot[m].values for m in models]
    stat, p_omnibus = friedmanchisquare(*arrays)

    result = {
        "n_subjects": len(pivot),
        "friedman_stat": round(float(stat), 4),
        "friedman_p": round(float(p_omnibus), 5),
        "significant_omnibus": bool(p_omnibus < 0.05),
        "posthoc": [],
    }

    if p_omnibus < 0.05:
        pairs = list(combinations(models, 2))
        raw_p = []
        medians_diff = []
        for a, b in pairs:
            diff = pivot[a] - pivot[b]
            if (diff == 0).all():
                raw_p.append(1.0)
            else:
                _, p = wilcoxon(diff, zero_method="wilcox")
                raw_p.append(p)
            medians_diff.append(float(diff.median()))

        corrected_p = holm_bonferroni(raw_p)
        for (a, b), p_raw, p_corr, med_diff in zip(pairs, raw_p, corrected_p, medians_diff):
            result["posthoc"].append({
                "model_a": a, "model_b": b,
                "median_diff_a_minus_b": round(med_diff, 4),
                "p_raw": round(p_raw, 5),
                "p_holm": round(p_corr, 5),
                "significant_after_correction": bool(p_corr < 0.05),
            })

    return result


def bootstrap_median_diff_ci(pivot: pd.DataFrame, model_a: str, model_b: str,
                             n_boot: int = 5000, seed: int = 0) -> dict:
    """95% bootstrap CI on the median paired (model_a - model_b) NSE difference."""
    diff = (pivot[model_a] - pivot[model_b]).values
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_medians[i] = np.median(sample)
    lo, hi = np.percentile(boot_medians, [2.5, 97.5])
    return {
        "median_diff": round(float(np.median(diff)), 4),
        "ci_95_low": round(float(lo), 4),
        "ci_95_high": round(float(hi), 4),
        "n_subjects": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(PER_REALIZATION_CSV))
    p.add_argument("--scenario", type=str, help="e.g. S4_SatBreakthrough")
    p.add_argument("--horizon", type=int, help="step count, e.g. 24 for 6hr")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Subset of models to test (default: all present in the data)")
    p.add_argument("--compare", type=str, nargs=2, default=None,
                   help="Two model tags for a bootstrap CI on their paired median difference")
    p.add_argument("--all", action="store_true",
                   help="Run the omnibus test for every scenario x horizon combination present")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"{csv_path} not found. Run scenario_evaluator.py with the "
              f"multi-realization scenarios first -- this file is only "
              f"produced once real per-realization data exists.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"{csv_path} is empty -- no per-realization data yet.")
        return

    if args.all:
        combos = df[["scenario", "horizon"]].drop_duplicates().sort_values(["scenario", "horizon"])
        for _, row in combos.iterrows():
            scen, hz = row["scenario"], row["horizon"]
            models = sorted(df[(df.scenario == scen) & (df.horizon == hz)]["model"].unique())
            if len(models) < 2:
                continue
            print(f"\n{'='*70}\n{scen}  horizon={hz} ({STEP_TO_HZ.get(hz, hz)})  "
                  f"models={len(models)}\n{'='*70}")
            pivot, n_before, n_after = build_subject_matrix(df, scen, hz, models)
            if n_after < n_before:
                print(f"  [note] {n_before - n_after}/{n_before} subjects dropped "
                      f"(missing a value for at least one model)")
            if pivot.empty:
                print("  [skip] no complete-case data")
                continue
            result = run_friedman_and_posthoc(pivot, models)
            print(f"  n_subjects={result.get('n_subjects')}  "
                  f"Friedman p={result.get('friedman_p')}  "
                  f"significant={result.get('significant_omnibus')}")
            for ph in result.get("posthoc", []):
                flag = " *" if ph["significant_after_correction"] else ""
                print(f"    {ph['model_a']:<22} vs {ph['model_b']:<22} "
                      f"median_diff={ph['median_diff_a_minus_b']:+.4f}  "
                      f"p_holm={ph['p_holm']:.5f}{flag}")
        return

    if not args.scenario or args.horizon is None:
        print("Specify --scenario and --horizon (or use --all).")
        return

    models = args.models or sorted(
        df[(df.scenario == args.scenario) & (df.horizon == args.horizon)]["model"].unique())
    print(f"Scenario: {args.scenario}   Horizon: {args.horizon} ({STEP_TO_HZ.get(args.horizon, '?')})")
    print(f"Models ({len(models)}): {models}")

    pivot, n_before, n_after = build_subject_matrix(df, args.scenario, args.horizon, models)
    if n_after < n_before:
        print(f"[note] {n_before - n_after}/{n_before} subjects dropped "
              f"(missing a value for at least one requested model)")
    if pivot.empty:
        print("No complete-case data for this scenario/horizon/model set.")
        return

    result = run_friedman_and_posthoc(pivot, models)
    print(f"\nFriedman omnibus test: n_subjects={result.get('n_subjects')}  "
          f"stat={result.get('friedman_stat')}  p={result.get('friedman_p')}  "
          f"significant={result.get('significant_omnibus')}")
    if "note" in result:
        print(f"  {result['note']}")
    for ph in result.get("posthoc", []):
        flag = " *** significant after Holm correction ***" if ph["significant_after_correction"] else ""
        print(f"  {ph['model_a']:<22} vs {ph['model_b']:<22} "
              f"median_diff={ph['median_diff_a_minus_b']:+.4f}  "
              f"p_raw={ph['p_raw']:.5f}  p_holm={ph['p_holm']:.5f}{flag}")

    if args.compare:
        a, b = args.compare
        if a in pivot.columns and b in pivot.columns:
            ci = bootstrap_median_diff_ci(pivot, a, b)
            print(f"\nBootstrap 95% CI, {a} - {b}:")
            print(f"  median diff = {ci['median_diff']:+.4f}  "
                  f"95% CI [{ci['ci_95_low']:+.4f}, {ci['ci_95_high']:+.4f}]  "
                  f"(n={ci['n_subjects']} subjects)")
        else:
            print(f"\n[skip] --compare {a} {b}: one or both not present in complete-case data")


if __name__ == "__main__":
    main()
