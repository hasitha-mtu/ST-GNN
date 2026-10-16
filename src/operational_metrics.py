"""
operational_metrics.py  —  Compact operational table (reviewer Section 3.4)
===========================================================================
One table with POD, FAR, peak-timing error and RMSE at representative short
(1 h, T_out = 4) and long (12 h, T_out = 48) lead times, mean +/- SD over the
three seeds, plus the S2 gauge-failure breakdown (failed gauges, hydraulically
connected downstream gauges, rest of the network).

Inputs
    checkpoints/{model}/{seed}/{T_out}/test_metrics.json      (RMSE, NSE)
    checkpoints/{model}/{seed}/{T_out}/extended_metrics.json  (POD, FAR, peak timing)
        -> regenerate with the REVISED run_inference.py (alignment fix):
           python src/run_inference.py --all-models --force
    results/scenarios/scenario_summary.csv                    (S2 groups)

Outputs  results/operational/operational_table.csv, operational_table.md,
         s2_failure_breakdown.csv

Thresholds: per-node bankfull thresholds in stage-anomaly space
(dataset/graph/bankfull_thresholds.json), as in manuscript Section 3.9.1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT = BASE_DIR / "checkpoints"
OUT = BASE_DIR / "results/operational"
LEADS = {4: "1 h", 48: "12 h"}
LABEL = {"gru": "GRU", "lstm": "LSTM", "ealstm": "EA-LSTM", "st_gnn": "ST-GNN (static)",
         "st_gnn_dyn_edge": "ST-GNN DynEdge", "st_gnn_hand_edge": "ST-GNN HANDEdge",
         "st_gnn_soil_gate": "ST-GNN SoilGate", "st_gnn_backwater_edge": "ST-GNN BackwaterEdge",
         "dfc_gnn": "DFC-GNN", "dfc_gnn_unified": "PC-DFC-GNN"}


def collect_real() -> pd.DataFrame:
    rows = []
    for tm in CKPT.rglob("test_metrics.json"):
        model, seed, hz = tm.parent.relative_to(CKPT).parts[:3]
        if int(hz) not in LEADS:
            continue
        r = {"model": model, "seed": int(seed), "horizon": int(hz)}
        r.update({k: json.load(open(tm)).get(k) for k in ("rmse", "nse")})
        em = tm.parent / "extended_metrics.json"
        if em.exists():
            e = json.load(open(em))
            r.update({k: e.get(k) for k in ("pod", "far", "peak_timing_mean_hr",
                                            "peak_timing_median_hr", "peak_timing_n_events")})
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(m, s, nd=3):
    if not np.isfinite(m):
        return "–"
    return f"{m:.{nd}f} ± {s:.{nd}f}" if np.isfinite(s) else f"{m:.{nd}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect_real()
    if df.empty:
        print("No test_metrics.json found under checkpoints/"); return
    cols = ["rmse", "pod", "far", "peak_timing_mean_hr"]
    g = df.groupby(["model", "horizon"])[cols].agg(["mean", "std"])
    g.to_csv(OUT / "operational_table.csv")
    lines = ["| Model | Lead | RMSE (m) | POD | FAR | Peak-timing error (h) |",
             "|---|---|---|---|---|---|"]
    for (m, hz), r in g.iterrows():
        lines.append(f"| {LABEL.get(m, m)} | {LEADS[hz]} | "
                     + " | ".join(fmt(r[(c, 'mean')], r[(c, 'std')], 2 if c.startswith('peak') else 3)
                                  for c in cols) + " |")
    sc = BASE_DIR / "results/scenarios/scenario_summary.csv"
    if sc.exists():
        s = pd.read_csv(sc)
        s2 = s[s.scenario == "S2_GaugeFailure"]
        keep = [c for c in s2.columns if c.startswith(("s2_rmse_", "s2_pod_", "s2_far_", "s2_nse_"))]
        if keep:
            b = s2[s2.horizon.isin(LEADS)].groupby(["model", "horizon"])[keep].mean()
            b.to_csv(OUT / "s2_failure_breakdown.csv")
            lines += ["", "**S2 gauge failure (observed targets), mean over seeds**", "",
                      "| Model | Lead | RMSE failed | RMSE connected d/s | RMSE other | POD failed | FAR failed |",
                      "|---|---|---|---|---|---|---|"]
            for (m, hz), r in b.iterrows():
                lines.append(f"| {LABEL.get(m, m)} | {LEADS[hz]} | {r.get('s2_rmse_failed_nodes', np.nan):.3f} | "
                             f"{r.get('s2_rmse_connected_downstream', np.nan):.3f} | "
                             f"{r.get('s2_rmse_other_nodes', np.nan):.3f} | "
                             f"{r.get('s2_pod_failed_nodes', np.nan):.2f} | {r.get('s2_far_failed_nodes', np.nan):.2f} |")
    (OUT / "operational_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
