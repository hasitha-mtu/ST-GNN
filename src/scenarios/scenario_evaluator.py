"""
scenario_evaluator.py  —  Evaluate trained checkpoints on synthetic scenarios
=============================================================================
REVISION AUDIT — implementation errors corrected in this version
---------------------------------------------------------------
E1  Cross-window contamination. Windows slid with stride 1 over the
    CONCATENATED scenario series, so most evaluated forecasts had inputs and
    targets from two unrelated base windows (for T_out = 48, 79 of every 104
    positions crossed a boundary). Only windows whose inputs AND targets lie
    inside one generated block are evaluated now
    (utils.alignment.window_starts_within_blocks).
E2  Target off-by-one. The final-lead target was y[t + T_in + T_out - 1]
    (lead T_out + 1). It is now y[t + T_in + T_out - 2], the same convention
    as training (utils/alignment.py, tested against GPUSampler).
E3  Inconsistent degradation ratio. Scenario RMSE used ONLY the last lead,
    pooled over nodes, but was divided by the real-data RMSE averaged over ALL
    leads and NODES (train_utils.compute_metrics). Scenario metrics are now
    computed for all leads with the same per-node-mean definition (used for
    the degradation ratio), and also reported pooled and for the last lead.
E4  Inference used edge attributes that differed from training (see
    utils/train_utils.load_graph_arrays). Fixed upstream.
E5  POD/FAR ignored the validity mask.

NEW (reviewer requests)
-----------------------
*  Observability split (Section 3.8): every forecast is labelled by whether
   its issue time is before or after the window's input-onset step (the first
   step at which the perturbation is visible in model inputs). Metrics are
   reported for all forecasts, for `observable` forecasts (issued after the
   perturbation started) and for `pre_onset` forecasts (unforeseeable).
*  S2: failed gauges, their hydraulically connected downstream gauges and the
   rest of the network are reported separately.
*  S3/S6: node-level metrics at the nodes where the effect should occur.
*  Optional --save-predictions writes last-lead predictions for the
   mechanism-diagnostic and operational-metric scripts.
*  --scen-root/--results-root allow the generator-sensitivity experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
SCEN_DIR  = BASE_DIR / "dataset/scenarios"
CKPT_ROOT = BASE_DIR / "checkpoints"
RESULTS   = BASE_DIR / "results/scenarios"

for _p in [BASE_DIR, BASE_DIR / "src"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.alignment import window_starts_within_blocks, target_rows

EVALUATOR_VERSION = "2026-10-revision-audit-1"
F_STAGE = 0
SCENARIOS = ["S1_ConvectiveCell", "S2_GaugeFailure", "S3_InniscarraRelease",
             "S4_SatBreakthrough", "S5_SpatialGradient", "S6_ChannelBlockage"]
NO_GRAPH = ("gru", "lstm", "ealstm")
DFC = ("dfc_gnn", "dfc_gnn_unified")


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers (pure numpy, unit-tested)
# ══════════════════════════════════════════════════════════════════════════════

def nse_pooled(p, t, m) -> float:
    m = m.astype(bool) & np.isfinite(p) & np.isfinite(t)
    if m.sum() < 2:
        return float("nan")
    pv, tv = p[m], t[m]
    ss = float(((tv - tv.mean()) ** 2).sum())
    return 1.0 - float(((tv - pv) ** 2).sum()) / ss if ss > 1e-12 else float("nan")


def rmse_pooled(p, t, m) -> float:
    m = m.astype(bool) & np.isfinite(p) & np.isfinite(t)
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2))) if m.sum() else float("nan")


def node_mean_metrics(p, t, m) -> dict:
    """Same definition as utils.train_utils.compute_metrics (real-data test
    metrics): per node over all rows/leads, then averaged across nodes."""
    N = p.shape[-1]
    P, Tt, M = p.reshape(-1, N), t.reshape(-1, N), m.reshape(-1, N).astype(bool)
    r, n = [], []
    for j in range(N):
        mm = M[:, j] & np.isfinite(P[:, j]) & np.isfinite(Tt[:, j])
        if mm.sum() < 2:
            continue
        e = P[mm, j] - Tt[mm, j]
        r.append(float(np.sqrt(np.mean(e ** 2))))
        ss = float(((Tt[mm, j] - Tt[mm, j].mean()) ** 2).sum())
        n.append(1.0 - float((e ** 2).sum()) / max(ss, 1e-8))
    return {"rmse_node_mean": float(np.mean(r)) if r else float("nan"),
            "nse_node_mean": float(np.mean(n)) if n else float("nan")}


def pod_far(p, t, m, thr) -> dict:
    m = m.astype(bool) & np.isfinite(p) & np.isfinite(t)
    thr = np.broadcast_to(thr, p.shape)
    o, f = (t >= thr) & m, (p >= thr) & m
    TP, FP, FN = int((o & f).sum()), int((~o & f & m).sum()), int((o & ~f).sum())
    return {"pod": TP / (TP + FN) if TP + FN else float("nan"),
            "far": FP / (TP + FP) if TP + FP else float("nan"),
            "TP": TP, "FP": FP, "FN": FN}


def _fmt(v, nd=4):
    return round(float(v), nd) if v is not None and np.isfinite(v) else None


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def forward_delta(model, tag, x, na, ei, ea):
    if tag in DFC:
        out = model(x)
        return out[0] if isinstance(out, (tuple, list)) else out
    if tag in NO_GRAPH:
        return model(x, na)
    return model(x, na, ei, ea)


@torch.no_grad()
def predict_windows(model, tag: str, X_syn: np.ndarray, starts: np.ndarray,
                    T_in: int, T_out: int, device, batch: int = 256) -> np.ndarray:
    """Absolute stage-anomaly predictions for all leads: [n_win, T_out, N]."""
    from run_inference import load_graph
    ei, ea, na = load_graph(device)
    out = []
    idx = np.arange(T_in)
    for b0 in range(0, len(starts), batch):
        s = starts[b0:b0 + batch]
        x = torch.from_numpy(X_syn[s[:, None] + idx[None, :]]).float().to(device)
        d = forward_delta(model, tag, x, na, ei, ea)
        out.append((d + x[:, -1, :, F_STAGE].unsqueeze(1)).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, T_out, X_syn.shape[1]))


def build_eval_arrays(X_syn, y_syn, m_syn, meta, T_in, T_out, input_valid_base=None):
    """Window starts, targets, masks and per-window bookkeeping."""
    TW = int(meta.get("T_per_window", 104))
    starts = window_starts_within_blocks(len(y_syn), TW, T_in, T_out)
    rows = target_rows(starts, T_in, T_out)                  # [n, T_out]
    tgt, msk = y_syn[rows], m_syn[rows].astype(bool)
    if input_valid_base is not None and "window_base_start_row" in meta:
        base = np.asarray(meta["window_base_start_row"])
        w = starts // TW
        anchor_row = base[w] + (starts % TW) + T_in - 1
        msk &= input_valid_base[anchor_row].astype(bool)[:, None, :]
    w_idx = starts // TW
    issue_local = starts % TW + T_in - 1
    onset = np.asarray(meta.get("window_input_onset_step", [-1] * (len(y_syn) // TW)))
    on_w = onset[w_idx]
    observable = (on_w >= 0) & (issue_local >= on_w)
    pre_onset = (on_w >= 0) & (issue_local < on_w)
    return {"starts": starts, "rows": rows, "target": tgt, "mask": msk,
            "window": w_idx, "issue_local": issue_local,
            "last_local": issue_local + T_out, "observable": observable,
            "pre_onset": pre_onset}


def core_metrics(pred, ev, bankfull, prefix="") -> dict:
    t, m = ev["target"], ev["mask"]
    nm = node_mean_metrics(pred, t, m)
    pf = pod_far(pred[:, -1], t[:, -1], m[:, -1], bankfull[None, :])
    return {f"{prefix}nse_syn": _fmt(nse_pooled(pred, t, m)),
            f"{prefix}rmse_syn": _fmt(rmse_pooled(pred, t, m)),
            f"{prefix}nse_node_mean": _fmt(nm["nse_node_mean"]),
            f"{prefix}rmse_node_mean": _fmt(nm["rmse_node_mean"]),
            f"{prefix}nse_last_lead": _fmt(nse_pooled(pred[:, -1], t[:, -1], m[:, -1])),
            f"{prefix}rmse_last_lead": _fmt(rmse_pooled(pred[:, -1], t[:, -1], m[:, -1])),
            f"{prefix}pod_last_lead": _fmt(pf["pod"]), f"{prefix}far_last_lead": _fmt(pf["far"]),
            f"{prefix}n_forecasts": int(len(pred))}


def subset(ev, sel):
    return {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v)
            for k, v in ev.items()}


def node_rmse(pred, ev, rows_sel, nodes) -> float | None:
    if not nodes or rows_sel.sum() == 0:
        return None
    p = pred[rows_sel][:, :, nodes]
    t = ev["target"][rows_sel][:, :, nodes]
    m = ev["mask"][rows_sel][:, :, nodes]
    return _fmt(rmse_pooled(p, t, m), 5)


def node_bias(pred, ev, rows_sel, nodes) -> float | None:
    if not nodes or rows_sel.sum() == 0:
        return None
    p = pred[rows_sel][:, :, nodes]; t = ev["target"][rows_sel][:, :, nodes]
    m = ev["mask"][rows_sel][:, :, nodes] & np.isfinite(p) & np.isfinite(t)
    return _fmt(float(np.mean((p - t)[m])) if m.sum() else float("nan"), 5)


# ══════════════════════════════════════════════════════════════════════════════
# Scenario-specific metrics
# ══════════════════════════════════════════════════════════════════════════════

def scenario_specific(pred, ev, meta, scen_dir: Path, real_rmse: float,
                      bankfull: np.ndarray) -> dict:
    sid = meta["name"][:2]
    obs = ev["observable"]
    out = {}
    if sid in ("S1", "S5"):
        out[f"{sid.lower()}_downstream_rmse"] = node_rmse(pred, ev, obs, meta.get("downstream_nodes", []))
    elif sid == "S2":
        fm_path = scen_dir / "fail_mask_synthetic.npy"
        if fm_path.exists():
            fm = np.load(fm_path)
            # absolute synthetic row of each forecast's issue time
            issue_rows = ev["starts"] + (ev["issue_local"] - ev["starts"] % meta["T_per_window"])
            failed_now = fm[issue_rows].astype(bool)                # [n, N]
            any_failed = failed_now.any(axis=1)
            reals = {r["id"]: r for r in meta["realizations"]}
            rid = np.asarray(meta["window_realization_id"])[ev["window"]]
            conn = np.zeros_like(failed_now)
            for i, r in enumerate(rid):
                if any_failed[i]:
                    conn[i, reals[int(r)]["connected_downstream_nodes"]] = True
            conn &= ~failed_now
            other = any_failed[:, None] & ~failed_now & ~conn
            for name, nm in (("failed_nodes", failed_now), ("connected_downstream", conn),
                             ("other_nodes", other)):
                m = ev["mask"] & nm[:, None, :]
                r_ = rmse_pooled(pred, ev["target"], m)
                out[f"s2_rmse_{name}"] = _fmt(r_, 5)
                out[f"s2_nse_{name}"] = _fmt(nse_pooled(pred, ev["target"], m))
                out[f"s2_degradation_{name}"] = _fmt(r_ / real_rmse) if real_rmse > 0 else None
                pf = pod_far(pred[:, -1], ev["target"][:, -1], m[:, -1],
                             bankfull[None, :])
                out[f"s2_pod_{name}"] = _fmt(pf["pod"]); out[f"s2_far_{name}"] = _fmt(pf["far"])
            # legacy column names kept for analyse_scenarios.py
            out["s2_post_failure_rmse_failed_nodes"] = out["s2_rmse_failed_nodes"]
            out["s2_post_failure_rmse_all"] = _fmt(rmse_pooled(pred[any_failed], ev["target"][any_failed],
                                                              ev["mask"][any_failed]), 5)
            out["s2_degradation_ratio"] = (_fmt(out["s2_post_failure_rmse_all"] / real_rmse)
                                           if out["s2_post_failure_rmse_all"] and real_rmse > 0 else None)
    elif sid == "S3":
        out["s3_downstream_rmse_post"] = node_rmse(pred, ev, obs, meta.get("downstream_nodes", []))
        dec = [v["idx"] for v in meta.get("decoupled_nodes", {}).values() if v.get("idx") is not None]
        out["s3_upstream_leakage_bias"] = node_bias(pred, ev, obs, dec)
        for n in meta.get("downstream_nodes", []):
            out[f"s3_rmse_node_{n}"] = node_rmse(pred, ev, obs, [n])
            out[f"s3_bias_node_{n}"] = node_bias(pred, ev, obs, [n])
        for n in dec:
            out[f"s3_bias_node_{n}"] = node_bias(pred, ev, obs, [n])
    elif sid == "S4":
        bt = {r["id"]: r["sat_breakthrough_step"] for r in meta["realizations"]}
        rid = np.asarray(meta["window_realization_id"])[ev["window"]]
        bt_w = np.array([bt[int(r)] for r in rid])
        pre = ev["last_local"] < bt_w
        out["s4_rmse_pre_breakthrough"] = _fmt(rmse_pooled(pred[pre], ev["target"][pre], ev["mask"][pre]), 5)
        out["s4_rmse_post_breakthrough"] = _fmt(rmse_pooled(pred[~pre], ev["target"][~pre], ev["mask"][~pre]), 5)
        if out["s4_rmse_pre_breakthrough"] and out["s4_rmse_post_breakthrough"]:
            out["s4_post_pre_ratio"] = _fmt(out["s4_rmse_post_breakthrough"] /
                                            max(out["s4_rmse_pre_breakthrough"], 1e-6))
    elif sid == "S6":
        b, u, d = (meta.get(k, {}).get("idx") for k in ("blockage_node", "upstream_node", "downstream_node"))
        out["s6_upstream_backwater_bias"] = (lambda v: -v if v is not None else None)(
            node_bias(pred, ev, obs, [i for i in (b, u) if i is not None]))
        out["s6_downstream_suppression_rmse"] = node_rmse(pred, ev, obs, [d] if d is not None else [])
        for name, n in (("blockage", b), ("upstream", u), ("downstream", d)):
            if n is not None:
                out[f"s6_rmse_{name}"] = node_rmse(pred, ev, obs, [n])
                out[f"s6_bias_{name}"] = node_bias(pred, ev, obs, [n])
    return out


def per_realization_rows(pred, ev, meta, scen, tag, seed, T_out) -> list[dict]:
    if "window_realization_id" not in meta:
        return []
    rid = np.asarray(meta["window_realization_id"])[ev["window"]]
    rows = []
    for r in np.unique(rid):
        for scope, sel in (("all", rid == r), ("observable", (rid == r) & ev["observable"])):
            if sel.sum() < 5:
                continue
            rows.append({"scenario": scen, "model": tag, "seed": seed, "horizon": T_out,
                         "realization_id": int(r), "scope": scope, "n_rows": int(sel.sum()),
                         "nse_syn_realization": _fmt(nse_pooled(pred[sel], ev["target"][sel], ev["mask"][sel])),
                         "rmse_syn_realization": _fmt(rmse_pooled(pred[sel], ev["target"][sel], ev["mask"][sel])),
                         **{f"{k}_realization": v for k, v in node_mean_metrics(
                             pred[sel], ev["target"][sel], ev["mask"][sel]).items()}})
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Per-checkpoint evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(ckpt_dir: Path, scen_dir: Path, device, bankfull: np.ndarray,
                        input_valid_base=None, save_pred_dir: Path | None = None):
    from run_inference import load_model, resolve_model_tag
    tag = resolve_model_tag(ckpt_dir)
    parts = ckpt_dir.relative_to(CKPT_ROOT).parts
    try:
        seed, T_out = int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return None
    X_syn = np.load(scen_dir / "X_synthetic.npy", mmap_mode="r")
    y_syn = np.load(scen_dir / "y_synthetic.npy")
    m_syn = np.load(scen_dir / "mask_synthetic.npy")
    meta = json.load(open(scen_dir / "scenario_meta.json"))
    model, hp = load_model(ckpt_dir, device)
    T_in = int(hp.get("t_in", 32))
    ev = build_eval_arrays(X_syn, y_syn, m_syn, meta, T_in, T_out, input_valid_base)
    if len(ev["starts"]) == 0:
        print(f"    [skip] no within-window forecasts for T_in={T_in}, T_out={T_out}")
        return None
    pred = predict_windows(model, tag, np.asarray(X_syn), ev["starts"], T_in, T_out, device)

    real_rmse = float("nan")
    tm = ckpt_dir / "test_metrics.json"
    if tm.exists():
        real_rmse = float(json.load(open(tm)).get("rmse", float("nan")))

    res = {"scenario": meta["name"], "model": tag, "seed": seed, "horizon": T_out,
           **core_metrics(pred, ev, bankfull)}
    for scope in ("observable", "pre_onset"):
        sel = ev[scope]
        if sel.sum() >= 5:
            res.update(core_metrics(pred[sel], subset(ev, sel), bankfull, prefix=f"{scope}_"))
    res["real_rmse"] = _fmt(real_rmse)
    # Degradation ratio with the SAME definition as the real-data RMSE
    # (all leads, per-node mean) — see module docstring E3.
    for scope in ("", "observable_"):
        r_ = res.get(f"{scope}rmse_node_mean")
        res[f"{scope}degradation_ratio"] = (_fmt(r_ / real_rmse, 3)
                                            if r_ is not None and real_rmse > 0 else None)
    res.update(scenario_specific(pred, ev, meta, scen_dir, real_rmse, bankfull))
    res["_per_realization"] = per_realization_rows(pred, ev, meta, meta["name"], tag, seed, T_out)
    if save_pred_dir is not None:
        save_pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_pred_dir / f"{tag}_{seed}_{T_out}.npz",
                            pred_last=pred[:, -1].astype(np.float32),
                            target_last=ev["target"][:, -1].astype(np.float32),
                            mask_last=ev["mask"][:, -1], starts=ev["starts"],
                            window=ev["window"], issue_local=ev["issue_local"],
                            observable=ev["observable"])
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def load_bankfull(nd: pd.DataFrame) -> np.ndarray:
    bf = json.load(open(GRAPH_DIR / "bankfull_thresholds.json"))
    bf = bf.get("thresholds", bf)
    return np.array([float(bf.get(str(r), 0.5)) for r in nd["ref"]], np.float32)


def _merge_and_save(df_new, out_path: Path, scenarios, key="scenario"):
    if out_path.exists():
        old = pd.read_csv(out_path)
        df_new = pd.concat([old[~old[key].isin(scenarios)], df_new], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df_new)} rows)")


def main(argv=None) -> None:
    from run_inference import resolve_model_tag
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-scenarios", action="store_true")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--scenario", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--scen-root", type=Path, default=SCEN_DIR)
    ap.add_argument("--results-root", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.all_scenarios:
        names = SCENARIOS
    elif a.scenario:
        names = [next((s for s in SCENARIOS if s.startswith(x.strip())), x.strip())
                 for x in a.scenario.split(",")]
    else:
        ap.error("Specify --all-scenarios or --scenario")
    nd = pd.read_csv(GRAPH_DIR / "nodes.csv")
    bankfull = load_bankfull(nd)
    ivp = PROC_DIR / "input_valid_mask.npy"
    iv = np.load(ivp, mmap_mode="r") if ivp.exists() else None
    ckpts = sorted(p.parent for p in CKPT_ROOT.rglob("best_model.pt"))
    if a.model:
        ckpts = [c for c in ckpts if resolve_model_tag(c) == a.model]
    if a.horizon:
        ckpts = [c for c in ckpts if c.name == str(a.horizon)]
    code_sig = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    all_rows, all_real = [], []
    for scen in names:
        sdir = a.scen_root / scen
        if not (sdir / "X_synthetic.npy").exists():
            print(f"[skip] {scen}: not generated under {a.scen_root}"); continue
        rdir = a.results_root / scen
        scen_sig = hashlib.sha256((sdir / "scenario_meta.json").read_bytes()).hexdigest()[:16]
        rows = []
        for i, c in enumerate(ckpts):
            label = str(c.relative_to(CKPT_ROOT))
            cache = rdir / f"{label.replace('/', '_').replace(chr(92), '_')}.json"
            ck_sig = max((f.stat().st_mtime for f in c.glob("*.pt")), default=None)
            if cache.exists() and not a.force:
                cj = json.load(open(cache))
                if (cj.get("_scen_sig"), cj.get("_ckpt_sig"), cj.get("_code_sig")) == (scen_sig, ck_sig, code_sig):
                    rows.append({k: v for k, v in cj.items() if not k.startswith("_")})
                    all_real.extend(cj.get("_per_realization", []))
                    continue
            print(f"  [{i+1}/{len(ckpts)}] {scen} {label} … ", end="", flush=True)
            try:
                r = evaluate_checkpoint(c, sdir, device, bankfull, iv,
                                        (rdir / "predictions") if a.save_predictions else None)
            except Exception as e:      # keep the sweep alive, but say why
                print(f"FAILED ({type(e).__name__}: {e})"); r = None
            if r is None:
                continue
            per = r.pop("_per_realization", [])
            all_real.extend(per)
            rows.append(r)
            rdir.mkdir(parents=True, exist_ok=True)
            json.dump({**r, "_per_realization": per, "_scen_sig": scen_sig,
                       "_ckpt_sig": ck_sig, "_code_sig": code_sig}, open(cache, "w"))
            print(f"NSE={r.get('nse_syn')}  obsNSE={r.get('observable_nse_syn')}")
        if rows:
            pd.DataFrame(rows).to_csv(rdir / "model_metrics.csv", index=False)
            all_rows.extend(rows)
    if all_rows:
        _merge_and_save(pd.DataFrame(all_rows), a.results_root / "scenario_summary.csv", names)
    if all_real:
        _merge_and_save(pd.DataFrame(all_real), a.results_root / "scenario_summary_per_realization.csv", names)


if __name__ == "__main__":
    main()
