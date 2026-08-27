"""
check_result_duplicates.py — verifies global_metrics_summary.csv,
scenario_summary.csv, and scenario_summary_per_realization.csv have
exactly one row per their expected key, before trusting
analyse_experiments.py / analyse_scenarios.py /
scenario_repeated_measures_inference.py's output.

Why this matters right now specifically: after retraining a model (e.g.
st_gnn_soil_gate, following the sat_threshold collapse fix) or
regenerating a single scenario (e.g. S5_SpatialGradient, following the
downstream-node injection-leakage fix), re-running run_inference.py /
scenario_evaluator.py should REPLACE the affected rows, not append new
ones alongside them. If the underlying pipeline appends instead of
overwrites/merges, every plot and rank table downstream silently
averages the broken (pre-fix) and fixed (post-fix) results together --
worse than not fixing it at all, since it would look like a smaller
improvement than what was actually achieved, or mask the fix entirely
depending on how badly the old rows skew the mean.

Expected uniqueness:
  global_metrics_summary.csv            : one row per (model, horizon)
  scenario_summary.csv                  : one row per (scenario, model, seed, horizon)
  scenario_summary_per_realization.csv  : one row per (scenario, model, seed, horizon, realization_id)

Usage:
    python check_result_duplicates.py
    python check_result_duplicates.py --model st_gnn_soil_gate   # focus report
"""
import argparse
import re
from pathlib import Path

import pandas as pd

GLOBAL_CSV = Path("results/figures/model_comparison/global_metrics_summary.csv")
SCENARIO_CSV = Path("results/scenarios/scenario_summary.csv")
PER_REALIZATION_CSV = Path("results/scenarios/scenario_summary_per_realization.csv")


def _normalize_model_name(name) -> str:
    """Strip parenthetical qualifiers and normalise case/whitespace, so
    --model matches regardless of whether a CSV uses raw tags
    (scenario_summary.csv: 'st_gnn_soil_gate') or display labels with
    suffixes (global_metrics_summary.csv: 'ST-GNN Soil Gate', or
    elsewhere 'GRU (no graph)') -- same normalization, same reason, as
    the fix applied earlier this session to analyse_scenarios.py's
    plot_scenario_advantage_table, which had the identical mismatch."""
    core = re.sub(r"\s*\(.*?\)\s*", "", str(name)).strip().lower()
    return core.replace("_", " ").replace("-", " ")


def check_duplicates(df: pd.DataFrame, key_cols: list[str], name: str,
                     focus_value: str | None = None, focus_col: str | None = None) -> bool:
    """Returns True if the file is clean (no duplicates on key_cols)."""
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"Total rows: {len(df)}")
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        print(f"  [error] expected key column(s) not found: {missing}")
        return False

    dupe_mask = df.duplicated(subset=key_cols, keep=False)
    n_dupe_rows = int(dupe_mask.sum())

    if n_dupe_rows == 0:
        print(f"  Clean -- exactly one row per {key_cols}, no duplicates found.")
        clean = True
    else:
        n_dupe_keys = df[dupe_mask].drop_duplicates(subset=key_cols).shape[0]
        print(f"  [WARN] {n_dupe_rows} rows across {n_dupe_keys} duplicated "
              f"{key_cols} combinations.")
        print(f"  Duplicated keys:")
        print(df[dupe_mask].sort_values(key_cols)[key_cols].drop_duplicates().to_string(index=False))
        clean = False

    if focus_value is not None and focus_col in df.columns:
        target_norm = _normalize_model_name(focus_value)
        sub = df[df[focus_col].map(_normalize_model_name) == target_norm]
        sub_dupes = sub.duplicated(subset=key_cols, keep=False)
        print(f"\n  Focus on {focus_col}='{focus_value}' (normalized match): "
              f"{len(sub)} rows, {int(sub_dupes.sum())} duplicated")
        if int(sub_dupes.sum()) > 0:
            print(sub[sub_dupes].sort_values(key_cols).to_string(index=False))

    return clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--global-csv", type=str, default=str(GLOBAL_CSV))
    p.add_argument("--scenario-csv", type=str, default=str(SCENARIO_CSV))
    p.add_argument("--per-realization-csv", type=str, default=str(PER_REALIZATION_CSV))
    p.add_argument("--model", type=str, default=None,
                   help="Model tag/label to focus-report on, e.g. st_gnn_soil_gate "
                        "or 'ST-GNN Soil Gate' (matches whichever column is present)")
    args = p.parse_args()

    all_clean = True

    gcsv = Path(args.global_csv)
    if gcsv.exists():
        gm = pd.read_csv(gcsv)
        clean = check_duplicates(
            gm, ["model", "horizon"], f"global_metrics_summary.csv ({gcsv})",
            focus_value=args.model, focus_col="model")
        all_clean &= clean
    else:
        print(f"[skip] {gcsv} not found")

    scsv = Path(args.scenario_csv)
    if scsv.exists():
        ss = pd.read_csv(scsv)
        clean = check_duplicates(
            ss, ["scenario", "model", "seed", "horizon"],
            f"scenario_summary.csv ({scsv})",
            focus_value=args.model, focus_col="model")
        all_clean &= clean
    else:
        print(f"[skip] {scsv} not found")

    prcsv = Path(args.per_realization_csv)
    if prcsv.exists():
        pr = pd.read_csv(prcsv)
        clean = check_duplicates(
            pr, ["scenario", "model", "seed", "horizon", "realization_id"],
            f"scenario_summary_per_realization.csv ({prcsv})",
            focus_value=args.model, focus_col="model")
        all_clean &= clean
    else:
        print(f"[skip] {prcsv} not found")

    print(f"\n{'='*70}")
    if all_clean:
        print("All files clean -- safe to run analyse_experiments.py / "
              "analyse_scenarios.py / scenario_repeated_measures_inference.py.")
    else:
        print("Duplicates found -- do NOT trust downstream analysis until "
              "resolved. Inspect the duplicated rows above: if they show "
              "old and new values for the same key, the evaluation "
              "pipeline appended instead of overwriting/merging.")


if __name__ == "__main__":
    main()
