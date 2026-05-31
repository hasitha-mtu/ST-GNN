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
    else:
        raise ValueError(f"Unhandled tag: {tag}")

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

    class WDS(torch.utils.data.Dataset):
        def __len__(self): return test_end - test_start
        def __getitem__(self, i):
            t  = test_start + i
            xs = torch.from_numpy(X[t-T_in:t].astype(np.float32))
            yt = torch.from_numpy(y[t:t+T_out].astype(np.float32))
            mk = torch.from_numpy(mask[t:t+T_out].astype(np.float32))
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
def infer_one(ckpt_dir: Path, device: torch.device) -> np.ndarray:
    """
    Run inference for one leaf checkpoint.
    Returns absolute stage predictions [T_test, N] (1-step ahead, horizon 0).
    Saves test_predictions.npy and test_predictions_meta.json in ckpt_dir.
    """
    model, hp       = load_model(ckpt_dir, device)
    dl, T_in, T_out, ts, te = test_dataloader(hp)
    tag             = resolve_model_tag(ckpt_dir)
    needs_graph     = tag not in ("gru", "lstm")
    # Always load node_attr — baselines also use static node features
    # (PerNodeGRU/LSTM concatenate node_attr with x_seq before the GRU).
    # Only skip edge_index and edge_attr for non-graph models.
    ei, ea, na      = load_graph(device)
    if not needs_graph:
        ei, ea = None, None

    preds = []
    for x_seq, y_seq, mask in dl:
        x_seq = x_seq.to(device)
        if needs_graph:
            delta = model(x_seq, na, ei, ea)
        else:
            delta = model(x_seq, na)
        # Absolute stage = delta + last observed stage anomaly
        last_obs  = x_seq[:, -1, :, 0]                   # [B, N]
        abs_pred  = delta[:, 0, :] + last_obs             # [B, N] — 1-step ahead
        preds.append(abs_pred.cpu().numpy())

    pred = np.concatenate(preds, axis=0)   # [T_test, N]
    np.save(ckpt_dir / "test_predictions.npy", pred)

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
            if pred_path.exists():
                print(f"    {cd.relative_to(CKPT_ROOT)}: cached")
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
