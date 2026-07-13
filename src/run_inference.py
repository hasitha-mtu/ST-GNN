"""
run_inference.py  –  Generate test predictions from any ST-GNN checkpoint
=========================================================================
Handles the three-level checkpoint structure:
    checkpoints/{model}/{seed}/{horizon}/best_model.pt

For each leaf checkpoint (one seed × one horizon), saves:
    test_predictions.npy         [T_test, N]  absolute stage (m)
    test_predictions_meta.json

After processing all seeds for a model+horizon, saves:
    checkpoints/{model}/test_predictions_{horizon}steps_mean.npy
    (ensemble mean across seeds — use this for flood maps)

Usage
-----
    # All models, all seeds, all horizons
    python src/run_inference.py --all-models

    # One model, all seeds, all horizons
    python src/run_inference.py --model st_gnn_dyn_edge

    # One specific leaf checkpoint
    python src/run_inference.py --leaf checkpoints/st_gnn_dyn_edge/42/4

    # Pick a specific horizon for flood maps (default: shortest = 4)
    python src/run_inference.py --all-models --horizon 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
CKPT_ROOT = BASE_DIR / "checkpoints"

# Add project root AND src/ to sys.path so imports work regardless of
# whether the script is run as:
#   python src/run_inference.py          (cwd = project root)
#   python run_inference.py              (cwd = src/)
for _p in [BASE_DIR, BASE_DIR / "src"]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ── Checkpoint discovery ────────────────────────────────────────────────────

def find_leaf_checkpoints(root: Path) -> list[Path]:
    """
    Recursively find every directory containing best_model.pt.
    Works regardless of nesting depth (2-level or 3-level structure).
    """
    return sorted([
        d for d in root.rglob("best_model.pt")
    ], key=lambda p: str(p))


def group_by_model_and_horizon(leaves: list[Path]) -> dict:
    """
    Given a flat list of leaf checkpoint paths, return a nested dict:
        {model_name: {horizon_str: [leaf_path, ...]}}

    Infers structure by looking at the directory depth relative to CKPT_ROOT.
    Handles both:
        2-level: checkpoints/{model}/{seed}/best_model.pt
        3-level: checkpoints/{model}/{seed}/{horizon}/best_model.pt
    """
    groups: dict[str, dict[str, list[Path]]] = {}

    for leaf in leaves:
        ckpt_dir = leaf.parent
        rel      = ckpt_dir.relative_to(CKPT_ROOT)
        parts    = rel.parts   # e.g. ("st_gnn_dyn_edge", "42", "4")

        if len(parts) == 1:
            # checkpoints/{model}/best_model.pt — flat single run
            model_name = parts[0]
            horizon    = "unknown"
        elif len(parts) == 2:
            # checkpoints/{model}/{seed}/best_model.pt
            model_name = parts[0]
            horizon    = "all"    # horizon baked into hparams
        elif len(parts) == 3:
            # checkpoints/{model}/{seed}/{horizon}/best_model.pt
            model_name = parts[0]
            horizon    = parts[2]
        else:
            # Deeper nesting — use first part as model, last as horizon
            model_name = parts[0]
            horizon    = parts[-1]

        groups.setdefault(model_name, {}).setdefault(horizon, []).append(ckpt_dir)

    return groups


# ── Model registry ──────────────────────────────────────────────────────────

# Module paths relative to src/ (which is on sys.path after the fix above).
# Do NOT include the "src." prefix — that caused "src.models is not a package"
# when running from the project root.
MODEL_REGISTRY = {
    "gru":              "models.baseline_gru.PerNodeGRU",
    "lstm":             "models.baseline_lstm.PerNodeLSTM",
    "st_gnn_static":    "models.st_gnn_flood.STGNNFloodModel",
    "st_gnn_sar":       "models.st_gnn_flood.STGNNFloodModel",
    "st_gnn_dyn_edge":  "models.st_gnn_dyn_edge.STGNNDynEdge",
    "st_gnn_hand_edge": "models.st_gnn_hand_edge.STGNNHANDEdge",
    "dfc_gnn": "models.dfc_gnn.DFCGNNFlood",
}

def resolve_model_tag(ckpt_dir: Path) -> str:
    """Infer model tag from checkpoint path (first part after CKPT_ROOT)."""
    rel   = ckpt_dir.relative_to(CKPT_ROOT)
    first = rel.parts[0]
    for key in MODEL_REGISTRY:
        if first.startswith(key) or key.startswith(first):
            return key
    return first


def import_class(dotted: str):
    mod, cls = dotted.rsplit(".", 1)
    return getattr(__import__(mod, fromlist=[cls]), cls)


def load_model(ckpt_dir: Path, device: torch.device):
    """
    Load model from a leaf checkpoint directory.

    Derives missing hparams (f_static, f_edge) from the actual data files
    rather than requiring them in the checkpoint — the training scripts
    only save f_dyn, hidden, gat_heads, gru_layers, dropout, t_in, t_out.
    The state_dict key is "state_dict" in training scripts (not "model_state_dict").
    """
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    hp   = ckpt.get("hparams", ckpt)   # some scripts embed hparams at top level
    tag  = resolve_model_tag(ckpt_dir)

    if tag not in MODEL_REGISTRY:
        raise ValueError(f"Unknown tag '{tag}'. Known: {list(MODEL_REGISTRY)}")

    cls = import_class(MODEL_REGISTRY[tag])

    # ── Derive dimensions from data files ─────────────────────────────
    # f_dyn: from X.npy feature count (saved by training script as hp["f_dyn"])
    # f_static: from nodes.csv column count (not saved in checkpoint)
    # f_edge: from edges.csv column count (not saved in checkpoint)
    # Use EXACT same columns as load_graph() in train_utils.py
    # (hardcoded 7 node features + 4 edge features).
    # Counting all non-index CSV columns gives wrong values (9 and 8).
    f_static   = 7   # log_catchment_area, gauge_datum, p90, amax, is_reservoir, is_tidal, has_discharge
    f_edge_raw = 4   # river_dist_km, area_ratio, elev_drop_m, same_tributary

    # f_dyn: prefer from checkpoint, fall back to X.npy
    if "f_dyn" in hp:
        f_dyn = hp["f_dyn"]
    else:
        X = np.load(PROC_DIR / "X.npy", mmap_mode="r")
        f_dyn = X.shape[2]

    # t_out: prefer from checkpoint directory name (most reliable),
    # then hparams, then dataset_metadata.json
    horizon_dir = ckpt_dir.name
    if horizon_dir.isdigit():
        t_out = int(horizon_dir)
    elif "t_out" in hp:
        t_out = hp["t_out"]
    else:
        t_out = 4   # default

    t_in  = hp.get("t_in", 32)

    print(f"  f_dyn={f_dyn}  f_static={f_static}  f_edge={f_edge_raw}  "
          f"t_in={t_in}  t_out={t_out}")

    common = dict(
        f_dyn      = f_dyn,
        f_static   = f_static,
        hidden     = hp.get("hidden", 64),
        t_out      = t_out,
        dropout    = hp.get("dropout", 0.1),
    )

    if tag in ("gru", "lstm"):
        n_layers = hp.get("gru_layers", hp.get("lstm_layers", 2))
        key      = "gru_layers" if tag == "gru" else "lstm_layers"
        model    = cls(f_dyn=f_dyn, f_static=f_static,
                       hidden=hp.get("hidden", 64),
                       **{key: n_layers},
                       t_out=t_out, dropout=hp.get("dropout", 0.1))
    elif tag in ("st_gnn_static", "st_gnn_sar"):
        model = cls(**common,
                    f_edge      = f_edge_raw,
                    gat_heads   = hp.get("gat_heads", 2),
                    gru_layers  = hp.get("gru_layers", 2),
                    sar_emb_dim = hp.get("sar_emb_dim", 0))
    elif tag == "st_gnn_dyn_edge":
        # Phase 1: f_edge = raw + 1 (dynamic conductance feature)
        model = cls(**common,
                    f_edge         = f_edge_raw + 1,
                    gat_heads      = hp.get("gat_heads", 2),
                    gru_layers     = hp.get("gru_layers", 2),
                    sar_emb_dim    = hp.get("sar_emb_dim", 0),
                    discharge_idx  = hp.get("discharge_idx", 3),
                    discharge_ref  = hp.get("discharge_ref", 1.0))
    elif tag == "st_gnn_hand_edge":
        # Phase 2: f_edge = raw + 1 (dynamic feature for both edge classes)
        hand  = np.load(GRAPH_DIR / "hand_edges.npz")
        model = cls(**common,
                    f_edge             = f_edge_raw + 1,
                    gat_heads          = hp.get("gat_heads", 2),
                    gru_layers         = hp.get("gru_layers", 2),
                    sar_emb_dim        = hp.get("sar_emb_dim", 0),
                    discharge_idx      = hp.get("discharge_idx", 3),
                    discharge_ref      = hp.get("discharge_ref", 1.0),
                    hand_src           = torch.from_numpy(hand["src"].astype(np.int64)),
                    hand_dst           = torch.from_numpy(hand["dst"].astype(np.int64)),
                    hand_threshold     = torch.from_numpy(hand["hand_threshold"]),
                    hand_overland_dist = torch.from_numpy(hand["overland_dist_km"]))
    elif tag == "dfc_gnn":
        # DFC-GNN: edge features are buffers inside the model;
        # use build_dfc_gnn() factory so they are loaded correctly.
        from models.dfc_gnn import build_dfc_gnn
        ef_path = GRAPH_DIR / "edge_features.npz"
        if not ef_path.exists():
            raise FileNotFoundError(
                f"edge_features.npz not found at {ef_path}. "
                f"Run compute_edge_features.py + extract_sar_wetness.py first.")
        n_nodes = np.load(PROC_DIR / "X.npy", mmap_mode="r").shape[1]
        model   = build_dfc_gnn(
            n_nodes      = n_nodes,
            f_in         = f_dyn,
            T_out        = t_out,
            ef_path      = str(ef_path),
            d_model      = hp.get("hidden", 64),
            n_heads      = hp.get("gat_heads", 4),
            n_layers     = hp.get("gru_layers", 2),
            dropout      = hp.get("dropout", 0.1),
            lambda_flood = hp.get("lambda_flood", 0.1),
            device       = "cpu",
        )
    else:
        raise ValueError(f"Unhandled tag: {tag}. Known: {list(MODEL_REGISTRY)}")

    # Load weights — training scripts save under "state_dict"
    state_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval().to(device)

    # Write back derived dimensions so test_dataloader can use them
    hp["f_dyn"]   = f_dyn
    hp["t_in"]    = t_in
    hp["t_out"]   = t_out
    hp["f_static"] = f_static
    hp["f_edge"]   = f_edge_raw

    n = sum(p.numel() for p in model.parameters())
    print(f"  {cls.__name__}  params={n:,}")
    return model, hp


# ── Data loading ────────────────────────────────────────────────────────────

def test_dataloader(hp: dict, batch_size: int = 256):
    X    = np.load(PROC_DIR / "X.npy",         mmap_mode="r")
    y    = np.load(PROC_DIR / "y.npy",          mmap_mode="r")
    mask = np.load(PROC_DIR / "valid_mask.npy", mmap_mode="r")
    T, N, F = X.shape
    T_in    = hp.get("t_in",  32)
    T_out   = hp.get("t_out",  4)
    test_start = int(T * 0.85)
    test_end   = T - T_out

    # INDEXING: must match training DataLoader (make_dataset in train_utils.py).
    # Training uses xs = X[t : t+T_in], yt = y[t+T_in : t+T_in+T_out]
    # The PREVIOUS version used X[t-T_in:t] which shifted every prediction
    # T_in = 32 steps (8 hours) early relative to its ground-truth target,
    # causing NSE ≈ -0.24 instead of the training-verified 0.993.
    n_windows = test_end - test_start - T_in + 1
    class WDS(torch.utils.data.Dataset):
        def __len__(self): return n_windows
        def __getitem__(self, i):
            t  = test_start + i          # window start (absolute timestep)
            xs = torch.from_numpy(X[t   : t+T_in       ].astype(np.float32))
            yt = torch.from_numpy(y[t+T_in : t+T_in+T_out].astype(np.float32))
            mk = torch.from_numpy(mask[t+T_in : t+T_in+T_out].astype(np.float32))
            return xs, yt, mk

    import platform
    nw = 0 if platform.system() == "Windows" else 4
    return torch.utils.data.DataLoader(
        WDS(), batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    ), T_in, T_out, test_start, test_end


def load_graph(device):
    """Mirror load_graph() in src/utils/train_utils.py exactly."""
    nd = pd.read_csv(GRAPH_DIR / "nodes.csv")
    ed = pd.read_csv(GRAPH_DIR / "edges.csv")
    node_cols = [
        "log_catchment_area_km2", "gauge_datum_mOSGM15", "p90_mAOD",
        "amax_med_mAOD", "is_reservoir", "is_tidal", "has_discharge",
    ]
    edge_cols = ["river_dist_km", "area_ratio", "elev_drop_m", "same_tributary"]
    src_col   = "src_idx" if "src_idx" in ed.columns else "src"
    dst_col   = "dst_idx" if "dst_idx" in ed.columns else "dst"
    na = torch.tensor(nd[node_cols].values, dtype=torch.float32)
    na = (na - na.mean(0)) / na.std(0).clamp(min=1e-6)  # z-score (same as train)
    na = na.to(device)
    ei = torch.tensor(ed[[src_col, dst_col]].values.T, dtype=torch.long).to(device)
    ea = torch.tensor(ed[edge_cols].values, dtype=torch.float32).to(device)
    return ei, ea, na


# ── Core inference ──────────────────────────────────────────────────────────

@torch.no_grad()
# ═══════════════════════════════════════════════════════════════════════
# Extended metric computation
# KGE (Gupta et al. 2009), POD, FAR (Moriasi et al. 2007),
# Peak timing error (operational flood warning metric)
# ═══════════════════════════════════════════════════════════════════════

def _kge_per_node(
    pred: np.ndarray,          # [T, N]
    target: np.ndarray,        # [T, N]
    mask: np.ndarray,          # [T, N] bool
) -> list[dict]:
    """
    Per-node Kling-Gupta Efficiency and its three decomposed components.

    KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
    where:
      r     = Pearson correlation coefficient
      alpha = std(pred) / std(obs)   — variability ratio
      beta  = mean(pred) / mean(obs) — bias ratio

    KGE = 1.0 is perfect; KGE = -0.41 equals mean-flow benchmark
    (Knoben et al. 2019, HESS 23(8):4323-4331).

    Returns list of dicts with keys: kge, r, alpha, beta
    """
    results = []
    N = pred.shape[1]
    for j in range(N):
        valid = mask[:, j] & np.isfinite(pred[:, j]) & np.isfinite(target[:, j])
        if valid.sum() < 10:
            results.append({"kge": np.nan, "r": np.nan,
                            "alpha": np.nan, "beta": np.nan})
            continue
        p, t = pred[valid, j].astype(float), target[valid, j].astype(float)
        r     = float(np.corrcoef(p, t)[0, 1])
        alpha = float(p.std()  / t.std())  if t.std()       > 1e-10 else np.nan
        beta  = float(p.mean() / t.mean()) if abs(t.mean()) > 1e-10 else np.nan
        if any(np.isnan(v) for v in [r, alpha, beta]):
            kge = np.nan
        else:
            kge = float(1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))
        results.append({"kge": kge, "r": r, "alpha": alpha, "beta": beta})
    return results


def _pod_far(
    pred: np.ndarray,       # [T, N]
    target: np.ndarray,     # [T, N]
    mask: np.ndarray,       # [T, N] bool
    bankfull: np.ndarray,   # [N] bankfull stage anomaly thresholds
) -> dict:
    """
    Probability of Detection (POD) and False Alarm Ratio (FAR)
    for node-level flood exceedance, aggregated over all valid (t, node) pairs.

    POD = TP / (TP + FN)  — fraction of observed floods correctly predicted
    FAR = FP / (TP + FP)  — fraction of predicted floods that did not occur

    Using per-node bankfull thresholds from bankfull_thresholds.json ensures
    the binary classification is physically grounded in the gauge network
    topology, not an arbitrary percentile.

    Reference: Gupta et al. (2009, JoH 377:80-91) — the original KGE paper.
    Knoben et al. (2019, HESS 23:4323-4331) establish KGE = -0.41 as the
    mean-flow benchmark, equivalent to NSE = 0.0.

    Reference: Moriasi et al. (2007, Trans. ASABE 50(3):885-900) recommend
    reporting POD and FAR alongside NSE for flood-event evaluation.
    """
    TP = FP = FN = TN = 0
    for j in range(pred.shape[1]):
        valid = mask[:, j] & np.isfinite(pred[:, j]) & np.isfinite(target[:, j])
        if valid.sum() < 10:
            continue
        p, t   = pred[valid, j], target[valid, j]
        thresh = float(bankfull[j])
        obs_f  = t >= thresh
        prd_f  = p >= thresh
        TP += int((prd_f &  obs_f).sum())
        FP += int((prd_f & ~obs_f).sum())
        FN += int((~prd_f & obs_f).sum())
        TN += int((~prd_f & ~obs_f).sum())
    pod = TP / (TP + FN + 1e-8)
    far = FP / (TP + FP + 1e-8)
    return {"pod": round(pod, 6), "far": round(far, 6),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN}


def _peak_timing_error(
    pred: np.ndarray,       # [T, N]
    target: np.ndarray,     # [T, N]
    mask: np.ndarray,       # [T, N] bool
    bankfull: np.ndarray,   # [N]
    dt_min: int = 15,
) -> dict:
    """
    Mean and median absolute peak timing error in hours, computed per flood
    event per gauge node.

    Algorithm
    ---------
    For each node j:
      1. Find observed peaks in target[j] above bankfull[j] with a minimum
         inter-peak separation of 4 hours (to avoid double-counting on the
         rising limb) using scipy.signal.find_peaks.
      2. For each observed peak at time t_obs, search the predicted series
         within ±48 hours and find the predicted peak closest to t_obs.
      3. Δt = |t_pred - t_obs| converted to hours.

    Aggregates all Δt values over all nodes and events.

    This metric directly measures the most operationally critical error mode:
    a model that predicts the correct flood magnitude but lags the peak by
    2 hours gives no benefit over a persistence forecast for evacuation
    decisions. No existing NSE / RMSE / KGE captures this.
    """
    MIN_PEAK_SEP  = max(2, int(4  * 60 / dt_min))   # 4-hour min separation
    SEARCH_WINDOW = int(48 * 60 / dt_min)            # ±48-hour search window

    errors = []
    N = pred.shape[1]

    for j in range(N):
        valid = mask[:, j] & np.isfinite(pred[:, j]) & np.isfinite(target[:, j])
        if valid.sum() < MIN_PEAK_SEP * 2:
            continue
        t_full = target[:, j].copy().astype(float)
        p_full = pred[:, j].copy().astype(float)
        t_full[~valid] = np.nan
        p_full[~valid] = np.nan

        thresh = float(bankfull[j])
        # Replace NaN with -inf for peak-finding (peaks must be real values)
        t_nf = np.where(np.isnan(t_full), -np.inf, t_full)
        p_nf = np.where(np.isnan(p_full), -np.inf, p_full)

        obs_peaks, _ = find_peaks(t_nf, height=thresh,
                                  distance=MIN_PEAK_SEP)
        if len(obs_peaks) == 0:
            continue

        for obs_pk in obs_peaks:
            lo = max(0, obs_pk - SEARCH_WINDOW)
            hi = min(len(p_nf), obs_pk + SEARCH_WINDOW)
            p_window = p_nf[lo:hi]
            if p_window.max() < -1e9:
                continue           # all NaN in search window

            pred_pks_local, _ = find_peaks(p_window,
                                           distance=MIN_PEAK_SEP)
            if len(pred_pks_local) == 0:
                # No local peak: use argmax in window as proxy
                pred_pk_local = int(np.argmax(p_window))
            else:
                # Closest predicted peak to the observed one
                obs_in_win = obs_pk - lo
                pred_pk_local = int(pred_pks_local[
                    np.argmin(np.abs(pred_pks_local - obs_in_win))
                ])
            pred_pk = lo + pred_pk_local
            errors.append(abs(pred_pk - obs_pk) * dt_min / 60.0)

    if not errors:
        return {"peak_timing_mean_hr":   np.nan,
                "peak_timing_median_hr": np.nan,
                "peak_timing_n_events":  0}
    return {
        "peak_timing_mean_hr":   round(float(np.mean(errors)),   4),
        "peak_timing_median_hr": round(float(np.median(errors)), 4),
        "peak_timing_n_events":  len(errors),
    }


def compute_extended_metrics(
    pred:      np.ndarray,   # [T_test, N] absolute stage predictions
    ts:        int,          # test start index in full time series
    T_in:      int,          # input window length
    ckpt_dir:  Path,
) -> None:
    """
    Compute KGE, POD, FAR, and peak timing error from saved predictions
    and write results to extended_metrics.json in ckpt_dir.

    Saves:
        ckpt_dir/extended_metrics.json
            kge_mean, kge_r_mean, kge_alpha_mean, kge_beta_mean
            pod, far, TP, FP, FN, TN
            peak_timing_mean_hr, peak_timing_median_hr, peak_timing_n_events
    """
    try:
        y_full   = np.load(PROC_DIR / "y.npy",          mmap_mode="r")
        m_full   = np.load(PROC_DIR / "valid_mask.npy", mmap_mode="r").astype(bool)

        tgt_start = ts + T_in
        n         = min(len(pred), len(y_full) - tgt_start)
        target    = y_full[tgt_start : tgt_start + n].astype(np.float32)
        mask      = m_full[tgt_start : tgt_start + n]
        p         = pred[:n]

        # ── Bankfull thresholds ───────────────────────────────────────
        bf_path = GRAPH_DIR / "bankfull_thresholds.json"
        if not bf_path.exists():
            print("  [extended_metrics] bankfull_thresholds.json not found — "
                  "POD/FAR/peak-timing skipped")
            bankfull = None
        else:
            with open(bf_path) as f:
                bf_dict = json.load(f)
            N = target.shape[1]
            # bankfull_thresholds.json is {node_ref: threshold_m} in mOD.
            # Convert to stage anomaly: threshold_anom = threshold_mOD - mean_stage_mOD
            # If the thresholds are already in anomaly space, use directly.
            bankfull = np.array([list(bf_dict.values())[j]
                                 if j < len(bf_dict) else np.inf
                                 for j in range(N)], dtype=np.float32)

        # ── KGE ──────────────────────────────────────────────────────
        kge_results = _kge_per_node(p, target, mask)
        kge_vals   = [r["kge"]   for r in kge_results if not np.isnan(r["kge"])]
        r_vals     = [r["r"]     for r in kge_results if not np.isnan(r["r"])]
        alpha_vals = [r["alpha"] for r in kge_results if not np.isnan(r["alpha"])]
        beta_vals  = [r["beta"]  for r in kge_results if not np.isnan(r["beta"])]

        ext = {
            "kge_mean":       round(float(np.mean(kge_vals)),   4) if kge_vals   else np.nan,
            "kge_r_mean":     round(float(np.mean(r_vals)),     4) if r_vals     else np.nan,
            "kge_alpha_mean": round(float(np.mean(alpha_vals)), 4) if alpha_vals else np.nan,
            "kge_beta_mean":  round(float(np.mean(beta_vals)),  4) if beta_vals  else np.nan,
        }

        # ── POD / FAR ─────────────────────────────────────────────────
        if bankfull is not None:
            ext.update(_pod_far(p, target, mask, bankfull))
        else:
            ext.update({"pod": np.nan, "far": np.nan,
                        "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN": np.nan})

        # ── Peak timing error ─────────────────────────────────────────
        if bankfull is not None:
            ext.update(_peak_timing_error(p, target, mask, bankfull))
        else:
            ext.update({"peak_timing_mean_hr": np.nan,
                        "peak_timing_median_hr": np.nan,
                        "peak_timing_n_events": 0})

        # Replace nan with null for JSON serialisation
        def _clean(v):
            if isinstance(v, float) and np.isnan(v): return None
            if isinstance(v, (np.floating, np.integer)):  return float(v)
            return v

        ext_clean = {k: _clean(v) for k, v in ext.items()}
        with open(ckpt_dir / "extended_metrics.json", "w") as f:
            json.dump(ext_clean, f, indent=2)

        print(f"  Extended metrics:  KGE={ext['kge_mean']:.4f}  "
              f"POD={ext.get('pod', float('nan')):.4f}  "
              f"FAR={ext.get('far', float('nan')):.4f}  "
              f"PeakΔt={ext.get('peak_timing_mean_hr', float('nan')):.2f}hr")

    except Exception as e:
        print(f"  [extended_metrics] FAILED: {e}")



def infer_one(ckpt_dir: Path, device: torch.device) -> np.ndarray:
    """
    Run inference for one leaf checkpoint.
    Returns absolute stage predictions [T_test, N] (1-step ahead, horizon 0).
    Saves test_predictions.npy and test_predictions_meta.json in ckpt_dir.
    """
    model, hp       = load_model(ckpt_dir, device)
    dl, T_in, T_out, ts, te = test_dataloader(hp)
    tag             = resolve_model_tag(ckpt_dir)
    # needs_graph: models that take (x_seq, na, ei, ea) as forward args
    # DFC-GNN stores edge_index/edge_attr as registered buffers internally;
    # it only takes x_seq and returns (stage_pred, flood_logits).
    needs_graph = tag not in ("gru", "lstm", "dfc_gnn")
    ei, ea, na  = load_graph(device)
    if not needs_graph:
        ei, ea = None, None
    if tag == "dfc_gnn":
        na = None   # DFC-GNN does not use static node_attr

    preds = []
    for x_seq, y_seq, mask in dl:
        x_seq = x_seq.to(device)
        if tag == "dfc_gnn":
            # DFC-GNN: forward(x) → (stage_pred [B,T_out,N], flood_logits [B,N])
            # Stage head outputs delta predictions; take step 0.
            output = model(x_seq)
            delta  = output[0] if isinstance(output, (tuple, list)) else output
        elif needs_graph:
            delta = model(x_seq, na, ei, ea)
        else:
            delta = model(x_seq, na)
        # Absolute stage = predicted delta + last observed stage anomaly.
        # x_seq[:, -1, :, 0] = stage anomaly at the last input timestep
        # (feature 0 is stage_anomaly in X.npy, same variable as y.npy).
        # This reconstruction matches the training eval in all train_*.py scripts:
        #   abs_pred = last_obs.unsqueeze(1) + delta_pred
        last_obs  = x_seq[:, -1, :, 0]                   # [B, N]
        abs_pred  = delta[:, 0, :] + last_obs             # [B, N] — 1-step ahead
        preds.append(abs_pred.cpu().numpy())

    pred = np.concatenate(preds, axis=0)   # [T_test, N]
    np.save(ckpt_dir / "test_predictions.npy", pred)

    # ── NSE sanity check ──────────────────────────────────────────────
    # Compare saved predictions against y.npy to verify scale alignment.
    # TARGET_START = TEST_START + T_IN so that pred[i] ↔ y[TARGET_START+i]
    # (both indexed from the first forecast step of the first test window).
    # If this NSE diverges > 0.05 from test_metrics.json, there is a
    # unit mismatch between predictions and targets.
    _sanity_nse = float("nan")
    try:
        y_full   = np.load(PROC_DIR / "y.npy",          mmap_mode="r")
        m_full   = np.load(PROC_DIR / "valid_mask.npy", mmap_mode="r")
        _tgt_start = ts + T_in           # first target timestep
        n_s      = min(len(pred), len(y_full) - _tgt_start)
        y_s      = y_full[_tgt_start : _tgt_start + n_s].astype(np.float32)
        m_s      = m_full[_tgt_start : _tgt_start + n_s].astype(bool)
        p_s      = pred[:n_s]
        valid    = m_s & np.isfinite(p_s) & np.isfinite(y_s)
        if valid.sum() > 0:
            p_v, t_v  = p_s[valid], y_s[valid]
            ss_res = float(np.sum((t_v - p_v) ** 2))
            ss_tot = float(np.sum((t_v - t_v.mean()) ** 2))
            _sanity_nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Load expected NSE from training evaluation (ground truth)
        _expected_nse = None
        _tm_path = ckpt_dir / "test_metrics.json"
        if _tm_path.exists():
            with open(_tm_path) as _f:
                _expected_nse = json.load(_f).get("nse")
        print(f"  NSE sanity check: {_sanity_nse:.4f}", end="")
        if _expected_nse is not None:
            gap = abs(_sanity_nse - _expected_nse)
            status = "OK" if gap < 0.05 else "WARNING: scale mismatch"
            print(f"  (training reported {_expected_nse:.4f},  gap={gap:.4f}  {status})")
        else:
            print()
    except Exception as _e:
        print(f"  NSE sanity check skipped: {_e}")

    # ── Extended metrics (KGE, POD, FAR, peak timing) ────────────────────
    compute_extended_metrics(pred, ts, T_in, ckpt_dir)

    all_ts = pd.to_datetime(pd.read_csv(PROC_DIR/"timestamps.csv")["timestamp"])
    meta   = {"ckpt": str(ckpt_dir), "model": tag,
              "shape": list(pred.shape),
              "T_out_trained": T_out, "horizon_saved": 1,
              "test_start": str(all_ts.iloc[ts]),
              "test_end":   str(all_ts.iloc[te-1])}
    with open(ckpt_dir / "test_predictions_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved test_predictions.npy  {pred.shape}")
    return pred


# ── High-level runners ──────────────────────────────────────────────────────

def run_model(model_dir: Path, device: torch.device,
              horizon_filter: str | None = None):
    """
    Process all seeds × horizons under model_dir.
    model_dir = checkpoints/st_gnn_dyn_edge

    For each horizon:
      - run inference for every seed
      - save ensemble mean as test_predictions_{H}steps_mean.npy in model_dir
    """
    leaves = find_leaf_checkpoints(model_dir)
    if not leaves:
        print(f"  No checkpoints found under {model_dir.name}")
        return

    groups = group_by_model_and_horizon(leaves)
    model_name = model_dir.name

    if model_name not in groups:
        # Single level — use the leaves directly
        groups = {model_name: {"all": [l.parent for l in leaves]}}

    for horizon, ckpt_dirs in groups.get(model_name, {}).items():
        if horizon_filter and horizon != horizon_filter:
            continue

        print(f"\n  Horizon {horizon} — {len(ckpt_dirs)} seed(s):")
        seed_preds = []
        for cd in ckpt_dirs:
            pred_path = cd / "test_predictions.npy"
            if pred_path.exists() and not getattr(args, "force", False):
                print(f"    {cd.relative_to(CKPT_ROOT)}: cached (use --force to rerun)")
                seed_preds.append(np.load(pred_path))
            else:
                print(f"    {cd.relative_to(CKPT_ROOT)}:")
                try:
                    p = infer_one(cd, device)
                    seed_preds.append(p)
                except Exception as e:
                    print(f"    FAILED: {e}")

        if len(seed_preds) > 1:
            mean_pred = np.mean(seed_preds, axis=0)
            mean_path = model_dir / f"test_predictions_{horizon}steps_mean.npy"
            np.save(mean_path, mean_pred)
            print(f"  Ensemble mean ({len(seed_preds)} seeds): {mean_path.name}")
        elif len(seed_preds) == 1:
            # Single seed — copy as mean for consistency
            mean_path = model_dir / f"test_predictions_{horizon}steps_mean.npy"
            np.save(mean_path, seed_preds[0])
            print(f"  Single seed: {mean_path.name}")


def run_all(device: torch.device, horizon_filter: str | None = None):
    model_dirs = [d for d in sorted(CKPT_ROOT.iterdir())
                  if d.is_dir() and not d.name.startswith(".")]
    print(f"Processing {len(model_dirs)} model directories ...")
    for md in model_dirs:
        print(f"\n{'='*50}\n  Model: {md.name}\n{'='*50}")
        try:
            run_model(md, device, horizon_filter)
        except Exception as e:
            print(f"  ERROR: {e}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate test_predictions.npy from ST-GNN checkpoints"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all-models", action="store_true",
                   help="Process every model in checkpoints/")
    g.add_argument("--model",  type=str,
                   help="Model name, e.g. st_gnn_dyn_edge")
    g.add_argument("--leaf",   type=Path,
                   help="Single leaf checkpoint dir, e.g. checkpoints/st_gnn_dyn_edge/42/4")
    p.add_argument("--horizon", type=str, default=None,
                   help="Process only this horizon (e.g. '4'). Default: all.")
    p.add_argument("--device",  type=str, default="cuda")
    p.add_argument("--force",   action="store_true",
                   help="Re-run even if test_predictions.npy already exists")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all_models:
        run_all(device, args.horizon)
    elif args.model:
        model_dir = CKPT_ROOT / args.model
        if not model_dir.exists():
            print(f"Model dir not found: {model_dir}"); sys.exit(1)
        run_model(model_dir, device, args.horizon)
    elif args.leaf:
        if not args.leaf.exists():
            print(f"Leaf dir not found: {args.leaf}"); sys.exit(1)
        infer_one(args.leaf, device)
