"""
compute_thresholds.py
─────────────────────
Computes flood threshold statistics from the downloaded OPW 15-min water level
timeseries and writes them back into config.yaml.

Statistics computed per station
────────────────────────────────
  p75_mAOD   75th percentile of all valid readings
             → level exceeded 25% of the time; proxy for "elevated" conditions
             → used in GNN input features to contextualise current level

  p90_mAOD   90th percentile of all valid readings
             → level exceeded 10% of the time; proxy for "high" conditions
             → used as the primary flood classification threshold

  amax_med   Median of annual maxima (one peak per calendar year)
             → equivalent to the ~2-year return period flood
             → used as the flood warning reference level for loss weighting
             → requires ≥2 complete years; stations with <2 years get NaN

  amax_high  Maximum of all annual maxima (worst flood in the record)
             → upper bound for loss weighting
             → with only 4 years of dataset this is NOT a reliable extreme estimate

HOW ANNUAL MAXIMA WORK
──────────────────────
For a station with readings like:

  2022: peak = 66.8 mAOD  (Oct storm)
  2023: peak = 67.4 mAOD  (Nov storm)
  2024: peak = 65.9 mAOD  (wet winter)
  2025: peak = 66.1 mAOD  (modest season)

  amax_med  = median([66.8, 67.4, 65.9, 66.1]) = 66.45 mAOD
  amax_high = max   ([66.8, 67.4, 65.9, 66.1]) = 67.4  mAOD

A year is only included if it has ≥ MIN_YEAR_COMPLETENESS valid readings.
This prevents a year with only a few readings from producing a spurious maximum.

PARTIAL YEARS
─────────────
The first and last calendar years in the record are often partial (e.g. dataset
starting April 2022 means Jan-Mar 2022 is missing). The script flags partial
years and excludes them from amax calculations by default (safe=True).

Usage
─────
  python compute_thresholds.py                          # reads combined CSV, prints table
  python compute_thresholds.py --update-config          # also writes values into config.yaml
  python compute_thresholds.py --dataset path/to/wl.csv   # use a different input file
  python compute_thresholds.py --min-completeness 0.8  # require 80% of year present
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_PATH      = Path("/dataset\processed\combined_water_level_15min.csv")
CONFIG_PATH    = Path("C:\\Users\AdikariAdikari\PycharmProjects\ST-GNN\config\config.yaml")
MIN_YEAR_FRAC  = 0.80    # require 80% of a year's 15-min slots to include in amax
READINGS_PER_YEAR = 365.25 * 24 * 4   # 15-min slots in a full year


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Ensure UTC-aware index
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    print(f"Loaded {path.name}: {df.shape[1]} stations, "
          f"{df.index.min().date()} → {df.index.max().date()}, "
          f"{len(df):,} timesteps")
    return df


def compute_thresholds(df: pd.DataFrame, min_year_frac: float = MIN_YEAR_FRAC) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by station ref with columns:
      n_readings, pct_valid, p75, p90, n_years_used, amax_med, amax_high,
      amax_values (list as string), partial_years_dropped
    """
    rows = []

    for col in df.columns:
        s = df[col].dropna()
        n_total   = len(df[col])
        n_valid   = len(s)
        pct_valid = n_valid / n_total if n_total > 0 else 0

        if n_valid < 100:
            rows.append({"ref": col, "n_readings": n_valid, "pct_valid": pct_valid,
                         "p75": np.nan, "p90": np.nan, "n_years_used": 0,
                         "amax_med": np.nan, "amax_high": np.nan,
                         "amax_values": "", "partial_years_dropped": ""})
            continue

        # ── Percentiles (all valid readings) ─────────────────────────────────
        p75 = float(s.quantile(0.75))
        p90 = float(s.quantile(0.90))

        # ── Annual maxima ─────────────────────────────────────────────────────
        # Group by calendar year
        yearly = s.groupby(s.index.year)

        amax_rows   = []
        dropped     = []

        for year, grp in yearly:
            # Expected number of 15-min slots in this year
            expected = 366 * 24 * 4 if (year % 4 == 0 and
                        (year % 100 != 0 or year % 400 == 0)) else 365 * 24 * 4
            completeness = len(grp) / expected

            if completeness < min_year_frac:
                dropped.append(f"{year}({completeness:.0%})")
            else:
                amax_rows.append({"year": year, "amax": float(grp.max()),
                                  "completeness": completeness})

        if len(amax_rows) >= 2:
            amaxes  = [r["amax"] for r in amax_rows]
            amax_med  = float(np.median(amaxes))
            amax_high = float(np.max(amaxes))
            amax_str  = "  ".join([f"{r['year']}:{r['amax']:.3f}" for r in amax_rows])
        elif len(amax_rows) == 1:
            # Only one complete year — report amax_high but not amax_med
            amax_med  = np.nan
            amax_high = float(amax_rows[0]["amax"])
            amax_str  = f"{amax_rows[0]['year']}:{amax_rows[0]['amax']:.3f}"
        else:
            amax_med  = np.nan
            amax_high = np.nan
            amax_str  = ""

        rows.append({
            "ref":                   col,
            "n_readings":            n_valid,
            "pct_valid":             round(pct_valid, 4),
            "p75":                   round(p75, 3),
            "p90":                   round(p90, 3),
            "n_years_used":          len(amax_rows),
            "amax_med":              round(amax_med, 3) if not np.isnan(amax_med) else np.nan,
            "amax_high":             round(amax_high, 3) if not np.isnan(amax_high) else np.nan,
            "amax_values":           amax_str,
            "partial_years_dropped": ", ".join(dropped),
        })

    return pd.DataFrame(rows).set_index("ref")


def print_table(result: pd.DataFrame, station_names: dict = None):
    """Pretty-print the threshold table."""
    print()
    print("=" * 100)
    print(f"{'Ref':>6}  {'Name':<25} {'Valid%':>6}  {'p75':>8}  {'p90':>8}  "
          f"{'Yrs':>4}  {'amax_med':>9}  {'amax_high':>10}")
    print("-" * 100)
    for ref, row in result.iterrows():
        name = (station_names or {}).get(str(ref), "")[:24]
        pct  = f"{row.pct_valid:.1%}" if not np.isnan(row.pct_valid) else "N/A"
        p75  = f"{row.p75:.3f}"       if not np.isnan(row.p75)       else "N/A"
        p90  = f"{row.p90:.3f}"       if not np.isnan(row.p90)       else "N/A"
        amed = f"{row.amax_med:.3f}"  if not np.isnan(row.amax_med)  else "N/A"
        ahi  = f"{row.amax_high:.3f}" if not np.isnan(row.amax_high) else "N/A"
        yrs  = int(row.n_years_used)
        print(f"  {ref:>6}  {name:<25} {pct:>6}  {p75:>8}  {p90:>8}  "
              f"{yrs:>4}  {amed:>9}  {ahi:>10}")
        if row.partial_years_dropped:
            print(f"{'':>42}  ↳ partial years dropped: {row.partial_years_dropped}")
    print("=" * 100)
    print()

    # Warnings
    low_years = result[result.n_years_used < 3]
    if not low_years.empty:
        print(f"⚠  {len(low_years)} station(s) have <3 complete years — "
              f"amax_med estimates are unreliable:")
        for ref in low_years.index:
            print(f"   {ref}: {low_years.loc[ref, 'n_years_used']} year(s)  "
                  f"{low_years.loc[ref, 'amax_values']}")
        print()


def update_config(result: pd.DataFrame, config_path: Path):
    """Write computed thresholds back into config.yaml in-place."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    stations = cfg["subsets"]["lee_full"]["water_level_stations"]
    updated = 0
    skipped = 0

    for stn in stations:
        ref = stn["ref"]
        if ref not in result.index:
            skipped += 1
            continue

        row = result.loc[ref]
        if np.isnan(row.p75):
            skipped += 1
            continue

        stn["p75_mAOD"]  = row.p75
        stn["p90_mAOD"]  = row.p90
        stn["amax_med"]  = None if np.isnan(row.amax_med)  else row.amax_med
        stn["amax_high"] = None if np.isnan(row.amax_high) else row.amax_high
        updated += 1

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"✓ config.yaml updated: {updated} stations written, {skipped} skipped (no dataset)")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",           type=Path, default=DATA_PATH,
                    help="Path to combined_water_level_15min.csv")
    ap.add_argument("--config",         type=Path, default=CONFIG_PATH,
                    help="Path to config.yaml")
    ap.add_argument("--update-config",  action="store_true",
                    help="Write thresholds into config.yaml")
    ap.add_argument("--min-completeness", type=float, default=MIN_YEAR_FRAC,
                    help=f"Minimum fraction of a year required for amax (default {MIN_YEAR_FRAC})")
    ap.add_argument("--show-annual",    action="store_true",
                    help="Print the individual annual maxima for each station")
    args = ap.parse_args()

    # Load station names from config for display
    station_names = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for s in cfg["subsets"]["lee_full"]["water_level_stations"]:
            station_names[s["ref"]] = s["name"]

    df     = load_data(args.data)
    result = compute_thresholds(df, min_year_frac=args.min_completeness)
    print_table(result, station_names)

    if args.show_annual:
        print("\nIndividual annual maxima (level in mAOD):")
        for ref, row in result.iterrows():
            if row.amax_values:
                print(f"  {ref:>6}  {row.amax_values}")
        print()

    if args.update_config:
        if not args.config.exists():
            print(f"ERROR: config not found at {args.config}", file=sys.stderr)
            sys.exit(1)
        update_config(result, args.config)

    return result


if __name__ == "__main__":
    main()
