"""
validate_pipeline.py  —  Runtime validation of the REAL data and outputs
========================================================================
Run after each stage of the revised pipeline. Each check reports PASS / WARN /
FAIL / SKIP (SKIP = input not produced yet). Exit code 1 if any FAIL.

    python src/validate_pipeline.py                 # everything available
    python src/validate_pipeline.py --check graph dataset datum
    python src/validate_pipeline.py --scen-root dataset/scenarios_sensitivity/S3/<cfg>

Writes results/validation/validation_report.{json,md}.

Checks
------
graph      DAG, no uphill reverse duplicates, metadata edge count
static     nodes.csv carries train-only attributes; if raw water levels exist,
           p90/amax are recomputed from the training period and must match
dataset    y[t] == X[t+1, stage]; valid_mask[t] == input_valid[t+1];
           normalised stage inverts to stage_abs exactly; SM alignment and
           climatology flags in dataset_metadata.json
datum      headrace level vs DEM water surface (suggests a Poolbeg offset);
           abs_datum_valid listing
uh         uh_params units, Tb = 2.67 Tp, calibration rows inside training,
           share of Tc / peak-rate values sitting on bounds
lags       routing_lags keys are exactly the DAG edges
scenarios  per scenario: generator version, base windows in test period and
           non-overlapping, y[t] == X[t+1] inside blocks (S2 excl. failed),
           S2 targets/mask untouched, onset metadata, exceedance share
ckpts      post-revision buffers present (node_abs_valid, sat_bounds),
           SoilGate hyper-parameters equal to Table 1
results    scenario_summary produced by the revised evaluator
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
for _p in (BASE, BASE / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PROC, GRAPH = BASE / "dataset/processed", BASE / "dataset/graph"
RESULTS: list[dict] = []


def report(check, status, msg, **extra):
    RESULTS.append({"check": check, "status": status, "message": msg, **extra})
    print(f"[{status:4s}] {check}: {msg}")


def chk_graph():
    from utils.graph_checks import find_reverse_duplicates, is_dag
    ed = pd.read_csv(GRAPH / "edges.csv"); nd = pd.read_csv(GRAPH / "nodes.csv")
    rev = find_reverse_duplicates(ed)
    report("graph.reverse_duplicates", "FAIL" if rev else "PASS",
           f"{len(rev)} uphill reverse-duplicate edges")
    report("graph.dag", "PASS" if is_dag(ed, len(nd)) else "FAIL", f"{len(ed)} edges")
    # A20: a river reach can never be shorter than the straight line, and
    # sinuosity > 3 between neighbouring gauges indicates a wrong length.
    xy = nd[["easting_itm", "northing_itm"]].to_numpy(float)
    straight = np.linalg.norm(xy[ed.src_idx] - xy[ed.dst_idx], axis=1) / 1000.0
    sin = ed.river_dist_km.to_numpy(float) / np.maximum(straight, 1e-6)
    bad = ed.loc[(sin < 0.98) | (sin > 3.0), ["src_ref", "dst_ref"]].astype(str)
    report("graph.reach_lengths", "FAIL" if len(bad) else "PASS",
           f"sinuosity range [{sin.min():.2f}, {sin.max():.2f}]"
           + (f"; implausible: {[tuple(r) for r in bad.to_numpy()]} - run "
              "src/data/compute_river_distances.py" if len(bad) else ""))
    # A21: DFC pairwise distances must be along the river tree
    ef = GRAPH / "edge_features.npz"
    if ef.exists():
        import networkx as nx
        from utils.graph_checks import river_edges
        G = nx.Graph()
        for e in river_edges(ed).itertuples():
            G.add_edge(int(e.src_idx), int(e.dst_idx), w=float(e.river_dist_km))
        z = np.load(ef)
        D = dict(nx.all_pairs_dijkstra_path_length(G, weight="w"))
        tree = np.array([D[int(s)].get(int(d), np.inf) for s, d in zip(z["src"], z["dst"])])
        err = float(np.nanmax(np.abs(tree - z["river_dist_km"])))
        report("graph.dfc_river_distance", "PASS" if err < 0.01 else "FAIL",
               f"max |DFC dist - river-tree dist| = {err:.3f} km"
               + ("" if err < 0.01 else " - rerun src/data/compute_edge_features.py"))


def chk_static():
    nd = pd.read_csv(GRAPH / "nodes.csv")
    need = {"stage_ref_low_m", "abs_datum_valid", "p50_mAOD_train"}
    miss = need - set(nd.columns)
    if miss:
        report("static.train_only_columns", "FAIL",
               f"missing {sorted(miss)} - run src/data/compute_static_attributes.py"); return
    report("static.train_only_columns", "PASS", "present")
    wl_dir = BASE / "dataset/raw/water_level"
    if not wl_dir.exists():
        report("static.recompute", "SKIP", "raw water levels not available"); return
    import build_dataset as bd
    from data.compute_static_attributes import (training_end_timestamp, train_only_attributes,
                                                load_datum_config, station_cfg)
    end = training_end_timestamp(bd.START_DATE, bd.END_DATE, bd.TIMESTEP)
    cfg = load_datum_config(); bad = []
    for _, r in nd[nd.is_tidal == 0].iterrows():
        wl = bd.load_water_level(str(r.ref))
        if wl is None:
            continue
        a = train_only_attributes(wl, end, pd.Timestamp(bd.START_DATE),
                                  station_cfg(cfg, str(r.ref)), float(r.gauge_datum_mOSGM15))
        if abs(a["p90_mAOD"] - r.p90_mAOD) > 1e-3:
            bad.append((r.ref, round(a["p90_mAOD"] - r.p90_mAOD, 4)))
    report("static.recompute", "FAIL" if bad else "PASS",
           f"p90 mismatches vs train-only recomputation: {bad}" if bad else "p90 = training-period value")


def chk_dataset():
    if not (PROC / "X.npy").exists():
        report("dataset", "SKIP", "X.npy not found"); return
    X = np.load(PROC / "X.npy", mmap_mode="r"); y = np.load(PROC / "y.npy", mmap_mode="r")
    d = np.abs(np.asarray(y[:-1]) - np.asarray(X[1:, :, 0]))
    report("dataset.y_is_next_stage", "PASS" if np.nanmax(d) < 1e-5 else "FAIL",
           f"max |y[t]-X[t+1,0]| = {np.nanmax(d):.2e}")
    ivp = PROC / "input_valid_mask.npy"
    if not ivp.exists():
        report("dataset.mask_alignment", "FAIL", "input_valid_mask.npy missing - rebuild dataset")
    else:
        vm, iv = np.load(PROC / "valid_mask.npy"), np.load(ivp)
        ok = np.array_equal(vm[:-1], iv[1:])
        report("dataset.mask_alignment", "PASS" if ok else "FAIL", "valid_mask[t] == input_valid[t+1]")
        if (PROC / "stage_abs.npy").exists():
            from utils.datum import stage_reference, reconstruct_stage
            nd = pd.read_csv(GRAPH / "nodes.csv")
            low, rng = stage_reference(nd)
            h = np.load(PROC / "stage_abs.npy")
            H = reconstruct_stage(np.asarray(X[:, :, 1], float), low, rng)
            m = iv.astype(bool) & np.isfinite(h) & (np.asarray(X[:, :, 1]) < 4.999) & \
                (np.asarray(X[:, :, 1]) > -1.999)
            err = np.abs(H - h)[m]
            report("dataset.norm_stage_inverts", "PASS" if err.max() < 1e-3 else "FAIL",
                   f"max |H - h| = {err.max():.2e} m over real, unclipped samples")
    mp = PROC / "dataset_metadata.json"
    if mp.exists():
        m = json.load(open(mp))
        ok = "backward" in m.get("sm_alignment", "") and m.get("sm_climatology") == "training period only"
        report("dataset.sm_causal", "PASS" if ok else "FAIL",
               f"alignment={m.get('sm_alignment')}, climatology={m.get('sm_climatology')}")


def chk_datum():
    nd = pd.read_csv(GRAPH / "nodes.csv")
    from utils.datum import abs_datum_valid
    v = abs_datum_valid(nd)
    report("datum.abs_valid", "WARN" if (~v).any() else "PASS",
           f"nodes excluded from terrain comparisons: {nd.loc[~v, 'name'].tolist()}")
    # Headrace pools vs the DEM water surface at the gauge pixel (unsnapped).
    # (edge_features.npz now stores the pool level itself as the headrace
    # elevation, so it cannot be used for this comparison.)
    dem_p = BASE / "dataset/dem/COP-DEM-30m_itm.tif"
    if dem_p.exists() and "p50_mAOD_train" in nd:
        import rasterio
        with rasterio.open(dem_p) as src:
            for ref in ("19094", "19095"):
                i = int(nd.index[nd.ref.astype(str) == ref][0])
                z = float(next(src.sample([(float(nd.loc[i, "easting_itm"]),
                                            float(nd.loc[i, "northing_itm"]))]))[0])
                off = float(nd.loc[i, "p50_mAOD_train"] - z)
                hint = " (~2.7 m: series is probably on the Poolbeg datum)" if 2.2 < off < 3.2 else ""
                report(f"datum.headrace_{ref}", "WARN" if abs(off) > 1.0 else "PASS",
                       f"median pool level - DEM surface = {off:+.2f} m{hint}")
    else:
        report("datum.headrace", "SKIP", "DEM or train-only p50 not available")


def chk_uh():
    p = GRAPH / "uh_params.json"
    if not p.exists():
        report("uh", "SKIP", "uh_params.json not found"); return
    u = json.load(open(p))
    if "units" not in u:
        report("uh.version", "FAIL", "uh_params.json produced by the pre-revision script"); return
    P = pd.DataFrame(u["params"]).T
    tb_ok = np.allclose(P.base_hr.astype(float), 2.67 * P.tp_hr.astype(float), atol=1e-3)
    report("uh.tb_standard", "PASS" if tb_ok else "FAIL", "Tb = 2.67 Tp")
    clipped = P.tc_clipped.astype(bool).mean()
    report("uh.tc_bounds", "WARN" if clipped > 0.25 else "PASS",
           f"{clipped:.0%} of nodes on a Tc bound; slope sources {P.slope_source.value_counts().to_dict()}")
    rb = P.rate_on_bound.notna().mean()
    report("uh.rate_bounds", "WARN" if rb > 0.5 else "PASS", f"{rb:.0%} of peak rates on an area-tier bound")
    from utils.train_utils import get_split_boundary
    X = np.load(PROC / "X.npy", mmap_mode="r") if (PROC / "X.npy").exists() else None
    if X is not None:
        t1, _ = get_split_boundary(X.shape[0])
        ok = u["calibration_period_rows"][1] <= t1
        report("uh.train_only", "PASS" if ok else "FAIL", f"calibration rows {u['calibration_period_rows']}")


def chk_lags():
    p = GRAPH / "routing_lags.json"
    if not p.exists():
        report("lags", "SKIP", "routing_lags.json not found"); return
    from utils.graph_checks import river_edges
    ed = river_edges(pd.read_csv(GRAPH / "edges.csv"))
    keys = set(json.load(open(p))["lags"])
    want = {f"{r.src_idx}_{r.dst_idx}" for r in ed.itertuples()}
    report("lags.edges", "PASS" if keys == want else "FAIL",
           f"extra={sorted(keys - want)} missing={sorted(want - keys)}")


def chk_scenarios(root: Path):
    from scenarios.scenario_generator import GENERATOR_VERSION
    from utils.train_utils import get_split_boundary
    if not (PROC / "X.npy").exists():
        report("scenarios", "SKIP", "no processed data"); return
    X = np.load(PROC / "X.npy", mmap_mode="r"); y = np.load(PROC / "y.npy", mmap_mode="r")
    vm = np.load(PROC / "valid_mask.npy", mmap_mode="r")
    _, t2 = get_split_boundary(X.shape[0])
    for d in sorted(root.glob("S*_*")):
        mp = d / "scenario_meta.json"
        if not mp.exists():
            continue
        m = json.load(open(mp)); s = d.name
        ver = m.get("generator_config", {}).get("version")
        if ver != GENERATOR_VERSION:
            report(f"scen.{s}.version", "FAIL", f"generated by {ver or 'pre-revision code'}"); continue
        TW, n = m["T_per_window"], m["n_windows"]
        st = np.asarray(m["window_base_start_row"])
        report(f"scen.{s}.test_period", "PASS" if st.min() >= t2 else "FAIL",
               f"earliest base row {st.min()} (test starts {t2})")
        us = np.unique(st)
        ov = int((np.diff(us) < TW).sum())
        report(f"scen.{s}.overlap", "PASS" if ov == 0 else "WARN", f"{ov} overlapping base-window pairs")
        Xs = np.load(d / "X_synthetic.npy", mmap_mode="r"); ys = np.load(d / "y_synthetic.npy", mmap_mode="r")
        fm = np.load(d / "fail_mask_synthetic.npy") if (d / "fail_mask_synthetic.npy").exists() \
            else np.zeros(ys.shape, np.uint8)
        worst = 0.0
        for w in range(n):
            a = np.asarray(ys[w * TW:(w + 1) * TW - 1]); b = np.asarray(Xs[w * TW + 1:(w + 1) * TW, :, 0])
            ok = ~fm[w * TW + 1:(w + 1) * TW].astype(bool)
            worst = max(worst, float(np.nanmax(np.abs(a - b)[ok])))
        report(f"scen.{s}.xy_alignment", "PASS" if worst < 1e-4 else "FAIL", f"max |y[t]-X[t+1]| = {worst:.2e}")
        if s.startswith("S2"):
            ms = np.load(d / "mask_synthetic.npy", mmap_mode="r")
            okm = all(np.array_equal(np.asarray(ms[w * TW:(w + 1) * TW]),
                                     np.asarray(vm[st[w]:st[w] + TW])) for w in range(n))
            report(f"scen.{s}.targets_kept", "PASS" if okm else "FAIL", "S2 target mask equals base mask")
        report(f"scen.{s}.onset_meta", "PASS" if len(m.get("window_input_onset_step", [])) == n else "FAIL",
               "per-window input onset recorded")
        ex = m.get("exceed_frac_targets", 0)
        report(f"scen.{s}.magnitude", "WARN" if ex > 0.01 else "PASS",
               f"{ex:.2%} of targets beyond 3x historical p99 ({m.get('clip_limit_m', float('nan')):.2f} m)")


def chk_ckpts():
    import torch
    root = BASE / "checkpoints"
    if not root.exists():
        report("ckpts", "SKIP", "no checkpoints/"); return
    old = []
    for tag in ("st_gnn_hand_edge", "dfc_gnn_unified", "st_gnn_soil_gate"):
        for p in (root / tag).rglob("best_model.pt"):
            sd = torch.load(p, map_location="cpu").get("state_dict", {})
            keys = {k.replace("_orig_mod.", "") for k in sd}
            need = "sat_bounds" if tag == "st_gnn_soil_gate" else "node_abs_valid"
            if need not in keys:
                old.append(str(p.parent.relative_to(root)))
    report("ckpts.revision_buffers", "FAIL" if old else "PASS",
           f"{len(old)} pre-revision checkpoints (retrain): {old[:5]}{' ...' if len(old) > 5 else ''}")
    import train_st_gnn_soil_gate as sg
    hp = (sg.BATCH_SIZE, sg.WEIGHT_DECAY, sg.PATIENCE)
    report("ckpts.soilgate_hparams", "PASS" if hp == (32, 1e-4, 38) else "FAIL",
           f"batch/wd/patience = {hp} (Table 1: 32 / 1e-4 / 38)")


def chk_results():
    p = BASE / "results/scenarios/scenario_summary.csv"
    if not p.exists():
        report("results", "SKIP", "scenario_summary.csv not found"); return
    df = pd.read_csv(p)
    ok = "observable_nse_syn" in df.columns and "rmse_node_mean" in df.columns
    report("results.evaluator_version", "PASS" if ok else "FAIL",
           "produced by revised evaluator" if ok else "produced by pre-revision evaluator - re-run")


CHECKS = {"graph": chk_graph, "static": chk_static, "dataset": chk_dataset, "datum": chk_datum,
          "uh": chk_uh, "lags": chk_lags, "scenarios": None, "ckpts": chk_ckpts, "results": chk_results}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="+", choices=list(CHECKS), default=list(CHECKS))
    ap.add_argument("--scen-root", type=Path, default=BASE / "dataset/scenarios")
    a = ap.parse_args(argv)
    for c in a.check:
        try:
            chk_scenarios(a.scen_root) if c == "scenarios" else CHECKS[c]()
        except Exception as e:                       # a broken check is itself a finding
            report(c, "FAIL", f"check raised {type(e).__name__}: {e}")
    out = BASE / "results/validation"; out.mkdir(parents=True, exist_ok=True)
    json.dump(RESULTS, open(out / "validation_report.json", "w"), indent=2)
    md = ["| status | check | message |", "|---|---|---|"] + \
         [f"| {r['status']} | {r['check']} | {r['message']} |" for r in RESULTS]
    (out / "validation_report.md").write_text("\n".join(md) + "\n")
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    print(f"\n{n_fail} FAIL, {sum(r['status'] == 'WARN' for r in RESULTS)} WARN -> {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
