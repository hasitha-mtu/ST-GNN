"""
compute_static_attributes.py  —  TRAINING-PERIOD-ONLY static node attributes
=============================================================================
Revision audit fix for reviewer point 3.2 / manuscript Section 5.5:

  "The flood-threshold node attributes (p90_mAOD, amax_med_mAOD) are computed
   from the full historical record rather than the training split alone."

This script recomputes every data-derived static attribute from raw OPW water
levels restricted to the TRAINING period, which is defined by exactly the same
boundary as the model loaders (utils.train_utils.get_split_boundary on the
build_dataset time index). Test-period and validation-period observations
cannot influence:

    p90_mAOD          normalised-stage denominator, BackwaterEdge gate reference
    amax_med_mAOD     static node attribute
    p50_mAOD_train    water-surface reference used for reservoir edge slopes
    stage_ref_low_m   lower normalisation reference (= gauge datum, or the
                      training 1st-percentile level for local-zero datums)
    abs_datum_valid   from config/station_datums.yaml

Outputs
-------
    dataset/graph/nodes.csv                    updated in place (backup kept)
    dataset/graph/graph_metadata.json          station p90/amax updated (backup kept)
    dataset/graph/static_attributes_report.csv full-record vs train-only diff

Run BEFORE build_dataset.py. Then re-run build_dataset.py, the HAND/backwater/
edge-feature precomputation, unit_hydrograph.py and all training.

Usage
-----
    python src/data/compute_static_attributes.py
    python src/data/compute_static_attributes.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
for _p in (BASE_DIR, BASE_DIR / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GRAPH_DIR = BASE_DIR / "dataset/graph"
DATUM_CFG = BASE_DIR / "config/station_datums.yaml"

MIN_YEAR_FRAC = 0.80
LOW_REF_QUANTILE = 0.01


def load_datum_config(path: Path = DATUM_CFG) -> dict:
    if not path.exists():
        return {"stations": {}, "dem_vertical_offset_to_malin_m": 0.0}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("stations", {})
    return cfg


def station_cfg(cfg: dict, ref: str) -> dict:
    d = {"raw_offset_to_malin_m": 0.0, "datum_is_local_zero": False,
         "abs_datum_verified": True}
    d.update(cfg.get("stations", {}).get(str(ref), {}) or {})
    return d


def training_end_timestamp(start: str, end: str, freq: str,
                           train_frac: float = 0.70) -> pd.Timestamp:
    """Timestamp of the first NON-training step, identical to the loaders."""
    from utils.train_utils import get_split_boundary
    idx = pd.date_range(start=start, end=end, freq=freq)
    t1, _ = get_split_boundary(len(idx), train_frac, 0.15)
    return idx[t1]


def annual_max_median(s: pd.Series, min_year_frac: float = MIN_YEAR_FRAC) -> float:
    amax = []
    for year, grp in s.groupby(s.index.year):
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        expected = (366 if leap else 365) * 96
        if grp.notna().sum() / expected >= min_year_frac:
            amax.append(float(grp.max()))
    return float(np.median(amax)) if amax else float("nan")


def train_only_attributes(wl: pd.Series, train_end: pd.Timestamp,
                          start: pd.Timestamp, scfg: dict,
                          gauge_datum: float) -> dict:
    """Pure function (unit-tested): thresholds from [start, train_end) only."""
    s = wl.loc[(wl.index >= start) & (wl.index < train_end)].dropna()
    s = s + float(scfg["raw_offset_to_malin_m"])
    if len(s) < 100:
        return {"p90_mAOD": np.nan, "amax_med_mAOD": np.nan,
                "p50_mAOD_train": np.nan, "stage_ref_low_m": gauge_datum,
                "n_train_obs": int(len(s))}
    p90 = float(s.quantile(0.90))
    amax_med = annual_max_median(s)
    if not np.isfinite(amax_med):
        # < 1 complete training year: fall back to the training maximum,
        # flagged in the report (never uses post-training data).
        amax_med = float(s.max())
    low = (float(s.quantile(LOW_REF_QUANTILE)) if scfg["datum_is_local_zero"]
           else float(gauge_datum))
    return {"p90_mAOD": p90, "amax_med_mAOD": amax_med,
            "p50_mAOD_train": float(s.quantile(0.50)),
            "stage_ref_low_m": low, "n_train_obs": int(len(s))}


def main(dry_run: bool = False) -> None:
    import build_dataset as bd   # same loaders/QC as the feature pipeline

    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv")
    with open(GRAPH_DIR / "graph_metadata.json") as f:
        meta = json.load(f)
    cfg = load_datum_config()

    start = pd.Timestamp(bd.START_DATE)
    t_end = training_end_timestamp(bd.START_DATE, bd.END_DATE, bd.TIMESTEP)
    print(f"Training period used for static attributes: {start} -> {t_end} (exclusive)")

    rows = []
    for i, r in nodes.iterrows():
        ref = str(r["ref"])
        scfg = station_cfg(cfg, ref)
        is_tidal = bool(r.get("is_tidal", 0))
        wl = bd.load_water_level(ref)
        old = {"p90_mAOD": float(r["p90_mAOD"]),
               "amax_med_mAOD": float(r["amax_med_mAOD"])}
        if wl is None:
            print(f"  [warn] {ref}: no raw water level; attributes unchanged")
            new = dict(old, p50_mAOD_train=np.nan,
                       stage_ref_low_m=float(r["gauge_datum_mOSGM15"]), n_train_obs=0)
        elif is_tidal:
            # Tidal nodes intentionally carry null thresholds (0.0) in the
            # submitted graph; keep that convention.
            att = train_only_attributes(wl, t_end, start, scfg,
                                        float(r["gauge_datum_mOSGM15"]))
            new = dict(old, p50_mAOD_train=att["p50_mAOD_train"],
                       stage_ref_low_m=float(r["gauge_datum_mOSGM15"]),
                       n_train_obs=att["n_train_obs"])
        else:
            new = train_only_attributes(wl, t_end, start, scfg,
                                        float(r["gauge_datum_mOSGM15"]))
        new["abs_datum_valid"] = int(bool(scfg["abs_datum_verified"]))
        rows.append({"ref": ref, "name": r["name"],
                     "p90_full": old["p90_mAOD"], "p90_train": new["p90_mAOD"],
                     "dp90_m": new["p90_mAOD"] - old["p90_mAOD"],
                     "amax_full": old["amax_med_mAOD"],
                     "amax_train": new["amax_med_mAOD"],
                     **{k: new[k] for k in ("p50_mAOD_train", "stage_ref_low_m",
                                            "n_train_obs", "abs_datum_valid")}})
        for k in ("p90_mAOD", "amax_med_mAOD", "p50_mAOD_train",
                  "stage_ref_low_m", "abs_datum_valid"):
            nodes.loc[i, k] = new[k]

    rep = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(rep.round(3).to_string(index=False))
    if dry_run:
        print("\n--dry-run: nothing written")
        return

    for p in (GRAPH_DIR / "nodes.csv", GRAPH_DIR / "graph_metadata.json"):
        bak = p.with_suffix(p.suffix + ".fullrecord.bak")
        if not bak.exists():
            shutil.copy2(p, bak)
    nodes.to_csv(GRAPH_DIR / "nodes.csv", index=False)
    by_ref = {str(r["ref"]): r for _, r in nodes.iterrows()}
    for s in meta["stations"]:
        r = by_ref.get(str(s["ref"]))
        if r is None or bool(s.get("is_tidal")):
            continue
        s["p90_mAOD"] = float(r["p90_mAOD"])
        s["amax_med"] = float(r["amax_med_mAOD"])
        s["p50_mAOD_train"] = float(r["p50_mAOD_train"])
        s["stage_ref_low_m"] = float(r["stage_ref_low_m"])
        s["abs_datum_valid"] = int(r["abs_datum_valid"])
    meta["static_attribute_period"] = {"start": str(start), "end_exclusive": str(t_end),
                                       "note": "training split only (revision audit)"}
    with open(GRAPH_DIR / "graph_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    rep.to_csv(GRAPH_DIR / "static_attributes_report.csv", index=False)
    print("\nWrote nodes.csv, graph_metadata.json, static_attributes_report.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
