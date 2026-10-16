"""
mechanism_diagnostics.py  —  Direct evidence of gate / pathway behaviour
=========================================================================
Reviewer request (Section 3.3): one multipanel figure showing how the
mechanisms behaved, not only aggregate rankings.

Panel A — SoilGate activation under S4
    For every ST-GNN SoilGate checkpoint: effective threshold theta, sharpness
    gamma and bounds; for every S4 window, the gate value
        g(t) = sigmoid(gamma * (Sbar(t) - theta)),  Sbar = node-mean swvl2_sat_ratio
    at each forecast issue time (computed from the checkpoint's own
    parameters and the scenario inputs — identical to the model's forward
    pass, see models/st_gnn_soil_gate.py).
    GATE OPENING is defined explicitly as g >= 0.5 (i.e. Sbar >= theta).
    Activation lag = (first issue step with g >= 0.5) - (breakthrough step),
    in hours; NaN when the gate never opens within the window (reported as
    "never").
    A1: theta per checkpoint vs the sampled S4 threshold range and the
        training-period Sbar distribution.
    A2: mean gate value vs time relative to breakthrough.

Panel B — Directional response at targeted nodes (S3 and S6)
    From scenario_summary.csv (evaluator node-level columns, observable
    forecasts only):
    B1 S3: mean bias at the two reservoir HEADRACE gauges (should be ~0: the
        release cannot propagate upstream) vs RMSE at downstream nodes.
    B2 S6: bias at the blockage gauge, the upstream gauge and the downstream
        gauge.

Outputs: results/mechanism/soilgate_activation.csv, soilgate_summary.csv,
         node_level_S3_S6.csv, mechanism_figure.png

Usage
-----
    python src/scenarios/mechanism_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
for _p in (BASE_DIR, BASE_DIR / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CKPT_ROOT = BASE_DIR / "checkpoints"
SCEN_DIR = BASE_DIR / "dataset/scenarios"
RES = BASE_DIR / "results"
OUT = RES / "mechanism"
F_SW2_SAT = 9
STEP_HR = 0.25


def soilgate_params(ckpt: Path) -> dict:
    import torch
    sd = torch.load(ckpt / "best_model.pt", map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    raw = float(sd["sat_threshold_raw"])
    lo, hi = (sd["sat_bounds"].tolist() if "sat_bounds" in sd else (0.0, 1.0))
    theta = lo + (hi - lo) / (1 + np.exp(-raw))
    if "sat_sharpness_raw" in sd:                  # revision: gamma = softplus(raw)
        gamma = float(np.log1p(np.exp(float(sd["sat_sharpness_raw"]))))
    else:
        gamma = float(sd["sat_sharpness"])
    return {"theta": theta, "gamma": gamma, "lo": lo, "hi": hi}


def gate_timeline(X_syn: np.ndarray, meta: dict, theta: float, gamma: float,
                  t_in: int = 32) -> pd.DataFrame:
    """Gate value at every within-window issue time (pure, unit-testable)."""
    TW = meta["T_per_window"]
    bt = {r["id"]: r["sat_breakthrough_step"] for r in meta["realizations"]}
    thr = {r["id"]: r["excess_threshold"] for r in meta["realizations"]}
    rows = []
    for w, rid in enumerate(meta["window_realization_id"]):
        sbar = X_syn[w * TW:(w + 1) * TW, :, F_SW2_SAT].mean(axis=1)
        for issue in range(t_in - 1, TW):
            g = 1.0 / (1.0 + np.exp(-gamma * (sbar[issue] - theta)))
            rows.append({"window": w, "realization": rid, "issue_step": issue,
                         "rel_to_breakthrough_hr": (issue - bt[rid]) * STEP_HR,
                         "sbar": float(sbar[issue]), "gate": float(g),
                         "true_threshold": thr[rid]})
    return pd.DataFrame(rows)


def activation_lags(tl: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (w, rid), d in tl.groupby(["window", "realization"]):
        opened = d[d.gate >= 0.5]
        out.append({"window": w, "realization": rid,
                    "activation_lag_hr": (float(opened.rel_to_breakthrough_hr.iloc[0])
                                          if len(opened) else np.nan),
                    "opens": bool(len(opened)),
                    "max_gate": float(d.gate.max())})
    return pd.DataFrame(out)


def panel_a(scen_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sdir = scen_dir / "S4_SatBreakthrough"
    X = np.load(sdir / "X_synthetic.npy", mmap_mode="r")
    meta = json.load(open(sdir / "scenario_meta.json"))
    rows, summ = [], []
    for ck in sorted(p.parent for p in (CKPT_ROOT / "st_gnn_soil_gate").rglob("best_model.pt")):
        prm = soilgate_params(ck)
        seed, hz = ck.parts[-2], ck.parts[-1]
        tl = gate_timeline(np.asarray(X), meta, prm["theta"], prm["gamma"])
        lags = activation_lags(tl)
        tl["seed"], tl["horizon"] = seed, hz
        rows.append(tl)
        thr = [r["excess_threshold"] for r in meta["realizations"]]
        summ.append({"seed": seed, "horizon": hz, **prm,
                     "theta_above_S4_threshold_range": prm["theta"] > max(thr),
                     "share_windows_gate_opens": float(lags.opens.mean()),
                     "median_activation_lag_hr": float(lags.activation_lag_hr.median()),
                     "share_open_before_breakthrough": float(
                         (lags.activation_lag_hr < 0).mean())})
    return (pd.concat(rows) if rows else pd.DataFrame()), pd.DataFrame(summ)


def panel_b(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    keep = [c for c in df.columns if c.startswith(("s3_", "s6_"))]
    d = df[df.scenario.isin(["S3_InniscarraRelease", "S6_ChannelBlockage"])]
    return d.groupby(["scenario", "model"])[keep].mean().reset_index()


def train_sbar_distribution() -> np.ndarray | None:
    p = BASE_DIR / "dataset/processed/X.npy"
    if not p.exists():
        return None
    from utils.train_utils import get_split_boundary
    X = np.load(p, mmap_mode="r")
    t1, _ = get_split_boundary(X.shape[0])
    return np.asarray(X[:t1:4, :, F_SW2_SAT]).mean(axis=1)


def figure(tl, summ, nodes, s4_thr_range, sbar_train, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a1, a2, b1, b2 = ax.ravel()
    if not summ.empty:
        a1.axhspan(*s4_thr_range, color="#e08e45", alpha=0.25, label="S4 sampled threshold range")
        if sbar_train is not None:
            a1.violinplot([sbar_train], positions=[-1], widths=0.8, showextrema=False)
        hzs = sorted(summ.horizon.astype(int).unique())
        for k, hz in enumerate(hzs):
            d = summ[summ.horizon.astype(int) == hz]
            a1.scatter(np.full(len(d), k), d.theta, s=30, color="#1b4965")
        a1.set_xticks([-1] + list(range(len(hzs))))
        a1.set_xticklabels(["train S̄"] + [f"{h * 15 // 60} h" for h in hzs], fontsize=8)
        a1.set_xlabel("forecast horizon of checkpoint (dots = seeds)")
        a1.set_ylabel("learned threshold θ (swvl2 saturation ratio)")
        a1.set_title("A1  SoilGate threshold vs S4 range", fontsize=10)
        a1.legend(fontsize=7)
    if not tl.empty:
        g = tl.groupby("rel_to_breakthrough_hr").gate.agg(["mean", "std"]).reset_index()
        a2.plot(g.rel_to_breakthrough_hr, g["mean"], color="#1b4965")
        a2.fill_between(g.rel_to_breakthrough_hr, g["mean"] - g["std"], g["mean"] + g["std"],
                        color="#1b4965", alpha=0.2)
        a2.axvline(0, color="#d62828", ls="--", lw=1, label="imposed breakthrough")
        a2.axhline(0.5, color="grey", ls=":", lw=1, label="opening criterion g = 0.5")
        a2.set_xlabel("issue time relative to breakthrough (h)"); a2.set_ylabel("gate value")
        a2.set_title("A2  SoilGate activation (all checkpoints)", fontsize=10); a2.legend(fontsize=7)
    s3 = nodes[nodes.scenario == "S3_InniscarraRelease"]
    if not s3.empty:
        y = np.arange(len(s3))
        b1.barh(y - 0.2, s3["s3_upstream_leakage_bias"], 0.4, label="headrace bias (leakage)")
        b1.barh(y + 0.2, s3["s3_downstream_rmse_post"], 0.4, label="downstream RMSE")
        b1.set_yticks(y); b1.set_yticklabels(s3.model, fontsize=8); b1.axvline(0, color="k", lw=0.5)
        b1.set_xlabel("m"); b1.set_title("B1  S3 directional response", fontsize=10); b1.legend(fontsize=7)
    s6 = nodes[nodes.scenario == "S6_ChannelBlockage"]
    if not s6.empty and "s6_bias_blockage" in s6:
        y = np.arange(len(s6))
        for k, (c, lab) in enumerate((("s6_bias_blockage", "blockage gauge"),
                                      ("s6_bias_upstream", "upstream gauge"),
                                      ("s6_bias_downstream", "downstream gauge"))):
            b2.barh(y + (k - 1) * 0.27, s6[c], 0.27, label=lab)
        b2.set_yticks(y); b2.set_yticklabels(s6.model, fontsize=8); b2.axvline(0, color="k", lw=0.5)
        b2.set_xlabel("mean bias, prediction − target (m)")
        b2.set_title("B2  S6 node-level bias", fontsize=10); b2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=250); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tl, summ = panel_a(SCEN_DIR)
    if not tl.empty:
        tl.to_csv(OUT / "soilgate_activation.csv", index=False)
        summ.to_csv(OUT / "soilgate_summary.csv", index=False)
        print(summ.round(3).to_string(index=False))
    sc = RES / "scenarios/scenario_summary.csv"
    nodes = panel_b(sc) if sc.exists() else pd.DataFrame(columns=["scenario"])
    nodes.to_csv(OUT / "node_level_S3_S6.csv", index=False)
    meta = json.load(open(SCEN_DIR / "S4_SatBreakthrough/scenario_meta.json"))
    thr = [r["excess_threshold"] for r in meta["realizations"]]
    figure(tl, summ, nodes, (min(thr), max(thr)), train_sbar_distribution(),
           OUT / "mechanism_figure.png")
    print(f"Saved {OUT/'mechanism_figure.png'}")


if __name__ == "__main__":
    main()
