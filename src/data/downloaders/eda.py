"""
eda.py
------
Exploratory Data Analysis for the Lee Catchment hydrological dataset.
Reads the processed combined CSVs produced by download_data.py.

Outputs (all saved to dataset/eda/):
  01_data_availability.png     -- record length & completeness heatmap
  02_water_level_timeseries.png -- full time series, all WL stations
  03_rainfall_timeseries.png    -- cumulative & event rainfall
  04_flood_events.png           -- periods exceeding AMAX thresholds
  05_seasonal_patterns.png      -- monthly boxplots per station
  06_rainfall_runoff_lag.png    -- cross-correlation rainfall -> water level
  07_station_correlations.png   -- inter-station WL correlation matrix
  08_data_distributions.png     -- histograms + exceedance curves
  eda_summary.txt               -- key statistics for GNN design decisions

Usage:
  python eda.py                          # uses paths from config.yaml
  python eda.py --config config/config.yaml
  python eda.py --subset lee_full        # override active subset
  python eda.py --start 2022-04-01 --end 2025-12-31
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- works on all platforms
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import signal

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "legend.fontsize":  8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
PALETTE = sns.color_palette("tab10")


# =============================================================================
# Data loader
# =============================================================================

def load_data(config: dict, start: str, end: str) -> dict:
    """Load processed combined CSVs and station metadata."""
    proc = Path(config["output"]["processed_dir"])

    def read(fname):
        p = proc / fname
        if not p.exists():
            logger.warning("Not found: %s", p)
            return pd.DataFrame()
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        # Clip to analysis window
        df = df.loc[start:end]
        return df

    wl_15  = read("combined_water_level_15min.csv")
    wl_1h  = read("combined_water_level_hourly.csv")
    rain_15 = read("combined_rainfall_15min.csv")
    rain_1h = read("combined_rainfall_hourly.csv")

    logger.info("Loaded water level 15-min : %s x %s", wl_15.shape[0], wl_15.shape[1])
    logger.info("Loaded rainfall  15-min  : %s x %s", rain_15.shape[0], rain_15.shape[1])

    # Build station metadata lookup from config
    subset_name = config["active_subset"]
    subset      = config["subsets"][subset_name]
    wl_meta  = {s["ref"]: s for s in subset.get("water_level_stations", [])}
    rain_meta = {s["ref"]: s for s in subset.get("rainfall_stations", [])}

    return {
        "wl_15":    wl_15,
        "wl_1h":    wl_1h,
        "rain_15":  rain_15,
        "rain_1h":  rain_1h,
        "wl_meta":  wl_meta,
        "rain_meta": rain_meta,
    }


def station_label(ref: str, meta: dict, max_len: int = 22) -> str:
    name = meta.get(ref, {}).get("name", ref)
    label = f"{name} ({ref})"
    return label[:max_len] + ".." if len(label) > max_len else label


# =============================================================================
# Plot 01 -- Data availability
# =============================================================================

def plot_availability(data: dict, out_dir: Path):
    logger.info("Plot 01: dataset availability")
    wl   = data["wl_15"]
    rain = data["rain_15"]
    wl_meta   = data["wl_meta"]
    rain_meta = data["rain_meta"]

    # Build availability matrix (monthly % valid)
    def monthly_completeness(df, meta):
        rows = []
        for col in df.columns:
            s = df[col].resample("ME").apply(lambda x: x.notna().mean() * 100)
            label = station_label(col, meta)
            rows.append((label, s))
        return rows

    wl_rows   = monthly_completeness(wl,   wl_meta)
    rain_rows = monthly_completeness(rain, rain_meta)
    all_rows  = wl_rows + rain_rows

    if not all_rows:
        return

    labels  = [r[0] for r in all_rows]
    months  = all_rows[0][1].index
    matrix  = np.array([[r[1].reindex(months).fillna(0).values for r in all_rows]]).squeeze()
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(14, max(3, len(labels) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                   interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    # x-axis: show year labels
    yr_ticks = [i for i, m in enumerate(months) if m.month == 1]
    ax.set_xticks(yr_ticks)
    ax.set_xticklabels([months[i].year for i in yr_ticks], fontsize=8)
    ax.set_xlabel("Year")

    # Separator between WL and rain
    sep = len(wl_rows) - 0.5
    ax.axhline(sep, color="black", linewidth=1.5, linestyle="--")
    ax.text(len(months) * 1.01, len(wl_rows) / 2 - 0.5, "Water\nLevel",
            va="center", fontsize=8, rotation=90)
    ax.text(len(months) * 1.01, len(wl_rows) + len(rain_rows) / 2 - 0.5,
            "Rainfall", va="center", fontsize=8, rotation=90)

    plt.colorbar(im, ax=ax, label="% valid readings per month", shrink=0.6)
    ax.set_title("Data Availability by Station and Month")
    fig.tight_layout()
    fig.savefig(out_dir / "01_data_availability.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Plot 02 -- Water level time series
# =============================================================================

def plot_wl_timeseries(data: dict, out_dir: Path):
    logger.info("Plot 02: water level time series")
    wl     = data["wl_1h"]   # hourly is readable
    meta   = data["wl_meta"]
    thresh = {}  # ref -> {p90, amax_med}
    for ref, s in meta.items():
        thresh[ref] = {
            "p90":     s.get("p90_mAOD"),
            "amax_med": s.get("amax_med"),
        }

    cols = [c for c in wl.columns if c in meta or True]  # all cols
    n    = len(cols)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(16, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        s = wl[col].dropna()
        if s.empty:
            ax.set_visible(False)
            continue
        ax.plot(s.index, s.values, lw=0.5, color=PALETTE[0], alpha=0.8)

        t = thresh.get(col, {})
        if t.get("p90"):
            ax.axhline(t["p90"], color="orange", lw=0.8, linestyle="--",
                       label=f"p90 {t['p90']:.2f} m")
        if t.get("amax_med"):
            ax.axhline(t["amax_med"], color="red", lw=0.8, linestyle="--",
                       label=f"AMAX med {t['amax_med']:.2f} m")

        label = station_label(col, meta)
        area  = meta.get(col, {}).get("catchment_area_km2", "")
        title = f"{label}  |  catchment {area} km2" if area else label
        ax.set_ylabel("mAOD", fontsize=8)
        ax.set_title(title, loc="left", fontsize=8)
        if t.get("p90") or t.get("amax_med"):
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.suptitle("Water Level Time Series (hourly) -- Lee Catchment", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "02_water_level_timeseries.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Plot 03 -- Rainfall time series
# =============================================================================

def plot_rainfall_timeseries(data: dict, out_dir: Path):
    logger.info("Plot 03: rainfall time series")
    rain = data["rain_1h"]
    meta = data["rain_meta"]
    cols = list(rain.columns)
    n    = len(cols)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(16, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        s = rain[col].fillna(0)
        # Bar chart for hourly rainfall (readable)
        ax.bar(s.index, s.values, width=1/24, color=PALETTE[2], alpha=0.7,
               label="Hourly mm")
        # Overlay 30-day cumulative
        ax2 = ax.twinx()
        cum = s.rolling(window=24 * 30, min_periods=1).sum()
        ax2.plot(cum.index, cum.values, color="navy", lw=0.8, alpha=0.7,
                 label="30-day rolling sum")
        ax2.set_ylabel("30d sum (mm)", fontsize=7, color="navy")
        ax2.tick_params(axis="y", labelcolor="navy", labelsize=7)

        label = station_label(col, meta)
        ax.set_ylabel("mm/hr", fontsize=8)
        ax.set_title(label, loc="left", fontsize=8)
        ax.set_ylim(bottom=0)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.suptitle("Rainfall Time Series (hourly) -- Lee Catchment", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "03_rainfall_timeseries.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Plot 04 -- Flood events
# =============================================================================

def plot_flood_events(data: dict, out_dir: Path) -> pd.DataFrame:
    """
    Identify and plot periods where water level exceeds AMAX median.
    Returns a DataFrame of flood events (used in summary report).
    """
    logger.info("Plot 04: flood events")
    wl   = data["wl_1h"]
    meta = data["wl_meta"]

    all_events = []

    cols = [c for c in wl.columns if c in meta]
    if not cols:
        cols = list(wl.columns)

    n   = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        s     = wl[col]
        smeta = meta.get(col, {})
        threshold = smeta.get("amax_med")
        label = station_label(col, meta)

        ax.plot(s.index, s.values, lw=0.4, color="steelblue", alpha=0.7)

        if threshold:
            ax.axhline(threshold, color="red", lw=1.0, linestyle="--",
                       label=f"AMAX median ({threshold:.2f} m)")
            flood_mask = s > threshold
            ax.fill_between(s.index, s.values, threshold,
                            where=flood_mask, color="red", alpha=0.3,
                            label="Above AMAX median")

            # Extract discrete flood events (contiguous blocks above threshold)
            in_event  = False
            evt_start = None
            for ts, val in s.items():
                above = (not pd.isna(val)) and (val > threshold)
                if above and not in_event:
                    in_event  = True
                    evt_start = ts
                elif not above and in_event:
                    duration_h = (ts - evt_start).total_seconds() / 3600
                    peak       = s.loc[evt_start:ts].max()
                    all_events.append({
                        "station_ref":  col,
                        "station_name": smeta.get("name", col),
                        "start":        evt_start,
                        "end":          ts,
                        "duration_h":   round(duration_h, 1),
                        "peak_mAOD":    round(peak, 3),
                        "threshold_mAOD": threshold,
                        "excess_m":     round(peak - threshold, 3),
                    })
                    in_event = False

        ax.set_ylabel("mAOD", fontsize=8)
        ax.set_title(label, loc="left", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.suptitle("Flood Events (exceedance of AMAX Median threshold)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "04_flood_events.png", bbox_inches="tight")
    plt.close(fig)

    events_df = pd.DataFrame(all_events) if all_events else pd.DataFrame()
    return events_df


# =============================================================================
# Plot 05 -- Seasonal patterns
# =============================================================================

def plot_seasonal(data: dict, out_dir: Path):
    logger.info("Plot 05: seasonal patterns")
    wl   = data["wl_1h"]
    rain = data["rain_1h"]
    meta = data["wl_meta"]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Monthly mean water level (each station normalised to zero-mean)
    ax1 = fig.add_subplot(gs[0, :])
    for i, col in enumerate(wl.columns):
        s = wl[col].dropna()
        if s.empty:
            continue
        monthly_mean = s.groupby(s.index.month).mean()
        normalised   = monthly_mean - monthly_mean.mean()
        label = station_label(col, meta, max_len=18)
        ax1.plot(monthly_mean.index, normalised.values, marker="o",
                 markersize=4, lw=1.5, label=label, color=PALETTE[i % 10])
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    ax1.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax1.set_ylabel("Anomaly from mean (m)")
    ax1.set_title("Monthly Water Level Anomaly (normalised per station)")
    ax1.legend(fontsize=7, ncol=3, loc="upper right")

    # 2. Monthly mean rainfall (stacked)
    ax2 = fig.add_subplot(gs[1, 0])
    rain_monthly = rain.copy()
    rain_monthly.index = rain_monthly.index.month
    rain_monthly = rain_monthly.groupby(rain_monthly.index).mean() * 24 * 30  # mm/month approx
    for i, col in enumerate(rain_monthly.columns):
        label = station_label(col, data["rain_meta"], max_len=18)
        ax2.plot(rain_monthly.index, rain_monthly[col].values,
                 marker="o", markersize=4, lw=1.5, label=label,
                 color=PALETTE[i % 10])
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=8)
    ax2.set_ylabel("Approx mm/month")
    ax2.set_title("Mean Monthly Rainfall")
    ax2.legend(fontsize=7)

    # 3. Hour-of-day average (diurnal cycle check — should be flat for rain)
    ax3 = fig.add_subplot(gs[1, 1])
    for i, col in enumerate(wl.columns):
        s = wl[col].dropna()
        if s.empty:
            continue
        hourly = s.groupby(s.index.hour).mean()
        norm   = hourly - hourly.mean()
        label  = station_label(col, meta, max_len=18)
        ax3.plot(hourly.index, norm.values, lw=1.2, alpha=0.8,
                 label=label, color=PALETTE[i % 10])
    ax3.set_xlabel("Hour of day (UTC)")
    ax3.set_ylabel("Anomaly (m)")
    ax3.set_title("Diurnal Cycle (should be ~flat for natural rivers)")
    ax3.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax3.legend(fontsize=7, ncol=2)

    fig.suptitle("Seasonal & Temporal Patterns -- Lee Catchment", fontsize=11)
    fig.savefig(out_dir / "05_seasonal_patterns.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Plot 06 -- Rainfall-runoff lag (cross-correlation)
# =============================================================================

def plot_lag_correlation(data: dict, out_dir: Path) -> dict:
    """
    Cross-correlate each rainfall station with each WL station.
    The lag at peak correlation = approximate rainfall-to-runoff travel time.
    Critical for setting the GNN temporal window (input_seq_len).
    Returns dict of peak lags in hours.
    """
    logger.info("Plot 06: rainfall-runoff lag correlation")
    wl   = data["wl_1h"].copy()
    rain = data["rain_1h"].copy()

    # Use first-difference of WL (removes baseline drift, captures rise events)
    wl_diff = wl.diff().fillna(0)

    # Align on common index
    common = wl_diff.index.intersection(rain.index)
    wl_d   = wl_diff.loc[common]
    rain_c = rain.loc[common].fillna(0)

    MAX_LAG_H = 72   # look up to 72 hours ahead
    wl_meta   = data["wl_meta"]
    rain_meta = data["rain_meta"]

    fig, axes = plt.subplots(len(wl_d.columns), len(rain_c.columns),
                             figsize=(4 * len(rain_c.columns),
                                      3 * len(wl_d.columns)),
                             sharex=True)
    # Ensure 2D array of axes
    if len(wl_d.columns) == 1:
        axes = [axes]
    if len(rain_c.columns) == 1:
        axes = [[ax] for ax in axes]

    peak_lags = {}

    for i, wl_col in enumerate(wl_d.columns):
        for j, rain_col in enumerate(rain_c.columns):
            ax = axes[i][j]
            w  = wl_d[wl_col].values
            r  = rain_c[rain_col].values

            # Normalise
            w = (w - w.mean()) / (w.std() + 1e-9)
            r = (r - r.mean()) / (r.std() + 1e-9)

            # Cross-correlation via scipy (full, then clip to MAX_LAG_H)
            n    = len(w)
            xcorr = signal.correlate(w, r, mode="full") / n
            lags  = signal.correlation_lags(n, n, mode="full")

            # Only positive lags (rainfall BEFORE water rise)
            pos_mask = (lags >= 0) & (lags <= MAX_LAG_H)
            lags_pos  = lags[pos_mask]
            xcorr_pos = xcorr[pos_mask]

            peak_lag = int(lags_pos[np.argmax(xcorr_pos)])
            peak_lags[(wl_col, rain_col)] = peak_lag

            ax.plot(lags_pos, xcorr_pos, lw=1.2, color=PALETTE[j % 10])
            ax.axvline(peak_lag, color="red", lw=1.0, linestyle="--",
                       label=f"peak={peak_lag}h")
            ax.set_title(
                f"WL: {station_label(wl_col, wl_meta, 14)}\n"
                f"Rain: {station_label(rain_col, rain_meta, 14)}",
                fontsize=7, loc="left"
            )
            ax.legend(fontsize=7)
            if i == len(wl_d.columns) - 1:
                ax.set_xlabel("Lag (hours)")
            if j == 0:
                ax.set_ylabel("Cross-corr")

    fig.suptitle("Rainfall -> Water Level Cross-Correlation\n"
                 "(peak lag = approximate travel time -- informs GNN input window)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "06_rainfall_runoff_lag.png", bbox_inches="tight")
    plt.close(fig)
    return peak_lags


# =============================================================================
# Plot 07 -- Station correlation matrix
# =============================================================================

def plot_correlations(data: dict, out_dir: Path):
    logger.info("Plot 07: station correlations")
    wl   = data["wl_1h"]
    meta = data["wl_meta"]

    if wl.shape[1] < 2:
        logger.info("  Skipping correlation matrix -- fewer than 2 stations")
        return

    # Rename columns to readable labels
    label_map = {col: station_label(col, meta, max_len=16) for col in wl.columns}
    corr = wl.rename(columns=label_map).corr(method="pearson")

    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8),
                                    max(5, len(corr) * 0.7)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, linewidths=0.5, mask=mask,
                annot_kws={"size": 7})
    ax.set_title("Pearson Correlation Between Water Level Stations\n"
                 "(high values = strong upstream/downstream coupling)")
    fig.tight_layout()
    fig.savefig(out_dir / "07_station_correlations.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Plot 08 -- Data distributions & exceedance curves
# =============================================================================

def plot_distributions(data: dict, out_dir: Path):
    logger.info("Plot 08: distributions & exceedance curves")
    wl   = data["wl_1h"]
    meta = data["wl_meta"]
    cols = list(wl.columns)
    n    = len(cols)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 2.8 * n))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        s     = wl[col].dropna()
        smeta = meta.get(col, {})
        label = station_label(col, meta)

        # Left: histogram
        ax_h = axes[i][0]
        ax_h.hist(s.values, bins=80, color=PALETTE[0], alpha=0.7, density=True)
        for pct_key, color, ls, lbl in [
            ("p90_mAOD",  "orange", "--",  "p90"),
            ("amax_med",  "red",    "--",  "AMAX med"),
            ("amax_high", "darkred", ":", "AMAX high"),
        ]:
            val = smeta.get(pct_key)
            if val:
                ax_h.axvline(val, color=color, lw=1.0, linestyle=ls,
                             label=f"{lbl} {val:.2f}")
        ax_h.set_title(f"{label} -- distribution", fontsize=8, loc="left")
        ax_h.set_xlabel("Water level (mAOD)")
        ax_h.set_ylabel("Density")
        ax_h.legend(fontsize=7)

        # Right: flow duration / exceedance curve
        ax_e = axes[i][1]
        sorted_vals = np.sort(s.values)[::-1]
        exceedance  = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        ax_e.plot(exceedance, sorted_vals, lw=1.2, color=PALETTE[1])
        for pct_key, color, ls, lbl in [
            ("p90_mAOD",  "orange", "--",  "p90"),
            ("amax_med",  "red",    "--",  "AMAX med"),
        ]:
            val = smeta.get(pct_key)
            if val:
                ax_e.axhline(val, color=color, lw=0.8, linestyle=ls,
                             label=f"{lbl} {val:.2f}")
        ax_e.set_xlabel("% time exceeded")
        ax_e.set_ylabel("Water level (mAOD)")
        ax_e.set_title(f"{label} -- exceedance curve", fontsize=8, loc="left")
        ax_e.legend(fontsize=7)

    fig.suptitle("Water Level Distributions & Exceedance Curves", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "08_data_distributions.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Text summary
# =============================================================================

def write_summary(data: dict, events_df: pd.DataFrame,
                  peak_lags: dict, out_dir: Path, start: str, end: str):
    logger.info("Writing EDA summary")
    wl   = data["wl_1h"]
    rain = data["rain_1h"]
    meta = data["wl_meta"]
    lines = []

    lines += [
        "=" * 70,
        "  LEE CATCHMENT -- EDA SUMMARY",
        f"  Analysis window: {start} -> {end}",
        "=" * 70,
        "",
        "DATA COMPLETENESS",
        "-" * 40,
    ]
    for col in wl.columns:
        s    = wl[col]
        comp = s.notna().mean() * 100
        rng  = f"{s.min():.3f} - {s.max():.3f}" if s.notna().any() else "N/A"
        lines.append(f"  WL  {col:<8} {station_label(col, meta, 24):<26}  "
                     f"complete={comp:5.1f}%  range={rng} mAOD")
    for col in rain.columns:
        s    = rain[col]
        comp = s.notna().mean() * 100
        total = s.sum(skipna=True)
        lines.append(f"  Rain {col:<7} {station_label(col, data['rain_meta'], 24):<26}  "
                     f"complete={comp:5.1f}%  total={total:.0f} mm")

    lines += ["", "FLOOD EVENTS (exceeding AMAX Median threshold)", "-" * 40]
    if events_df.empty:
        lines.append("  No flood events detected in analysis window.")
    else:
        by_station = events_df.groupby("station_ref")
        for ref, grp in by_station:
            grp_s = grp.sort_values("duration_h", ascending=False)
            lines.append(f"\n  Station {ref} ({meta.get(ref, {}).get('name', '')})")
            lines.append(f"    Total events     : {len(grp)}")
            lines.append(f"    Total flood hours: {grp['duration_h'].sum():.0f} h")
            lines.append(f"    Longest event    : {grp_s.iloc[0]['duration_h']:.0f} h  "
                         f"({grp_s.iloc[0]['start']})")
            lines.append(f"    Max peak         : {grp['peak_mAOD'].max():.3f} mAOD  "
                         f"(threshold {grp_s.iloc[0]['threshold_mAOD']:.3f})")
            lines.append("    Top 5 events by duration:")
            for _, ev in grp_s.head(5).iterrows():
                lines.append(f"      {str(ev['start'])[:16]}  "
                              f"dur={ev['duration_h']:.0f}h  "
                              f"peak={ev['peak_mAOD']:.3f}m")

    lines += ["", "RAINFALL-RUNOFF LAG (hours)", "-" * 40,
              "  (= recommended minimum GNN temporal input window)"]
    if peak_lags:
        max_lag = max(peak_lags.values())
        lines.append(f"  Max observed lag          : {max_lag} hours")
        lines.append(f"  Recommended input_seq_len : >= {max(24, max_lag + 6)} hours")
        lines.append("")
        for (wl_col, rain_col), lag in sorted(peak_lags.items(),
                                              key=lambda x: -x[1]):
            wl_name   = meta.get(wl_col, {}).get("name", wl_col)
            rain_name = data["rain_meta"].get(rain_col, {}).get("name", rain_col)
            lines.append(f"  {rain_name:<20} -> {wl_name:<22} : {lag:3d} h")

    lines += [
        "",
        "GNN DESIGN IMPLICATIONS",
        "-" * 40,
        f"  Total WL stations         : {wl.shape[1]}",
        f"  Total rainfall stations   : {rain.shape[1]}",
        f"  15-min timesteps in window: {len(data['wl_15'])}",
        f"  Hourly timesteps in window: {len(wl)}",
    ]
    if peak_lags:
        max_lag = max(peak_lags.values())
        seq_len = max(24, max_lag + 6)
        lines += [
            f"  Suggested input_seq_len   : {seq_len} hours ({seq_len * 4} x 15-min steps)",
            f"  Suggested forecast horizon: 2, 4, 6, 12, 24 hours",
        ]
    if not events_df.empty:
        total_flood_h = events_df["duration_h"].sum()
        total_h       = len(wl)
        flood_frac    = total_flood_h / max(total_h, 1) * 100
        lines += [
            f"  Flood class imbalance     : ~{flood_frac:.1f}% of timesteps above AMAX median",
            "  => Use weighted flood loss or oversampling for rare flood events",
        ]
    lines += ["", "=" * 70]

    out_path = out_dir / "eda_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary -> %s", out_path)

    # Also print to console
    for line in lines:
        logger.info(line)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="EDA for Lee catchment OPW dataset")
    ap.add_argument("--config", default="C:\\Users\AdikariAdikari\PycharmProjects\ST-GNN\config\config.yaml")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--start",  default=None, help="Analysis start YYYY-MM-DD")
    ap.add_argument("--end",    default=None, help="Analysis end YYYY-MM-DD")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))

    if args.subset:
        config["active_subset"] = args.subset

    start = args.start or config.get("analysis_window", {}).get("start", "2022-04-01")
    end   = args.end   or config.get("analysis_window", {}).get("end",   "2025-12-31")

    out_dir = Path(config["output"]["processed_dir"]).parent / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("EDA output directory: %s", out_dir)

    data = load_data(config, start, end)

    if data["wl_15"].empty and data["rain_15"].empty:
        logger.error("No dataset loaded -- check processed_dir in config.yaml")
        sys.exit(1)

    plot_availability(data, out_dir)
    plot_wl_timeseries(data, out_dir)
    plot_rainfall_timeseries(data, out_dir)
    events_df = plot_flood_events(data, out_dir)
    plot_seasonal(data, out_dir)
    peak_lags = plot_lag_correlation(data, out_dir)
    plot_correlations(data, out_dir)
    plot_distributions(data, out_dir)
    write_summary(data, events_df, peak_lags, out_dir, start, end)

    logger.info("")
    logger.info("EDA complete. All outputs in: %s", out_dir)
    logger.info("  01_data_availability.png")
    logger.info("  02_water_level_timeseries.png")
    logger.info("  03_rainfall_timeseries.png")
    logger.info("  04_flood_events.png")
    logger.info("  05_seasonal_patterns.png")
    logger.info("  06_rainfall_runoff_lag.png")
    logger.info("  07_station_correlations.png")
    logger.info("  08_data_distributions.png")
    logger.info("  eda_summary.txt")


if __name__ == "__main__":
    main()
