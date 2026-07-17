"""
train_models_exp1.py  —  Experiment 1 Orchestrator  (Journal of Hydrology)
══════════════════════════════════════════════════════════════════════════════
Trains all six forecasting architectures for the Experiment 1 architectural
comparison paper targeting Journal of Hydrology.

Model roster (6 trainable + 1 computed baseline):
──────────────────────────────────────────────────
  Persistence             computed from y.npy — zero-parameter reference
  PerNodeGRU              per-node GRU, no graph (temporal lower bound)
  PerNodeLSTM             per-node LSTM, no graph (temporal lower bound)
  STGNNFloodModel         static GATConv over river network
  STGNNDynEdge            + Manning discharge conductance (5th edge feature)
  STGNNHANDEdge           + HAND-triggered cross-tributary topology
  DFCGNNFlood             redesigned PhysicallyBiasedGATConv + flood head

SAR policy for Experiment 1
──────────────────────────────
  ALL models are trained WITHOUT any Sentinel-1 SAR input.

  • PerNodeGRU / PerNodeLSTM:    no SAR by design (no sar_emb argument)
  • STGNNFloodModel:             trained via train_st_gnn_flood_model.py
                                 (no-SAR script — separate from _sar variant)
  • STGNNDynEdge:                USE_SAR = False patched at module level
  • STGNNHANDEdge:               USE_SAR = False patched at module level
  • DFCGNNFlood:                 USE_SAR_EDGE = False patched at module level
                                 → uses only 4 terrain edge features
                                 → saved to checkpoints/dfc_gnn/ (not dfc_gnn_sar)

  train_st_gnn_flood_model_sar.py and sar_fno_encoder.py are NOT imported
  here.  They belong entirely to Experiment 2 (train_models_exp2.py).

Checkpoint structure
────────────────────
  checkpoints/gru/{seed}/{t_out}/
  checkpoints/lstm/{seed}/{t_out}/
  checkpoints/st_gnn/{seed}/{t_out}/
  checkpoints/st_gnn_dyn_edge/{seed}/{t_out}/
  checkpoints/st_gnn_hand_edge/{seed}/{t_out}/
  checkpoints/dfc_gnn/{seed}/{t_out}/            ← 4-feature, no SAR

  results/baselines/persistence_{t_out}steps.csv

Horizons
────────
  T_out = 4  (1 hr)  — high-frequency response
  T_out = 12 (3 hr)  — kinematic wave propagation range
  T_out = 16 (4 hr)  — near the Lee catchment max travel time
  T_out = 24 (6 hr)  — Irish Civil Defence activation threshold
  T_out = 48 (12 hr) — Evacuation planning threshold

Usage
──────
  python src/train_models_exp1.py
  python src/train_models_exp1.py --horizons 4 12 16    # subset of horizons
  python src/train_models_exp1.py --seeds 42            # single seed (debug)
  python src/train_models_exp1.py --skip-existing       # skip trained checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR)       not in sys.path: sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR/"src") not in sys.path: sys.path.insert(0, str(BASE_DIR/"src"))

# ── Import training modules ────────────────────────────────────────────
from train_per_node_gru_model  import train as _train_gru
from train_per_node_lstm_model import train as _train_lstm
from train_st_gnn_flood_model  import train as _train_st_gnn

# DynEdge and HANDEdge: import module objects so we can patch USE_SAR
import train_st_gnn_dyn_edge  as _mod_dyn
import train_st_gnn_hand_edge as _mod_hand
import train_dfc_gnn          as _mod_dfc

from utils.common_utils import seed_everything
from utils.config       import load_config
from utils.logger       import get_logger

# ══════════════════════════════════════════════════════════════════════
# Patch SAR flags to False before any training begins
# This must happen at import time, before any train() call.
# ══════════════════════════════════════════════════════════════════════
_mod_dyn.USE_SAR      = False   # STGNNDynEdge: no SAR FNO embeddings
_mod_hand.USE_SAR     = False   # STGNNHANDEdge: no SAR FNO embeddings
_mod_dfc.USE_SAR_EDGE = False   # DFCGNNFlood: 4 terrain edge features only

# ── Experiment configuration ───────────────────────────────────────────
SEEDS     = [42, 123, 456]
T_IN      = 32                          # 8-hour input window
T_OUTS    = [4, 12, 16, 24, 48]         # 1hr, 3hr, 4hr, 6hr, 12hr
MAX_EPOCHS = 300

# SEEDS     = [42]
# T_IN      = 32                          # 8-hour input window
# T_OUTS    = [4]         # 1hr, 3hr, 4hr, 6hr, 12hr
# MAX_EPOCHS = 1

# Model registry in narrative order (baselines first, then graph models)
MODEL_REGISTRY = [
    ("gru",              _train_gru),
    ("lstm",             _train_lstm),
    ("st_gnn",           _train_st_gnn),
    ("st_gnn_dyn_edge",  _mod_dyn.train),
    ("st_gnn_hand_edge", _mod_hand.train),
    ("dfc_gnn",          _mod_dfc.train),
]


# ══════════════════════════════════════════════════════════════════════
# Persistence baseline  (computed, not trained)
# ══════════════════════════════════════════════════════════════════════

def compute_persistence_baseline(t_outs: list[int]) -> None:
    """
    Compute and save persistence baseline NSE for each T_out.

    ŷ(t + k) = y(t)  for all k in [1, T_out]

    Persistence is the zero-parameter, zero-training-cost floor.
    Including it makes the paper's Table 1 self-contained without
    requiring readers to compute it independently.

    Output: results/baselines/persistence_{t_out}steps.csv
            columns: node_ref, nse, rmse
    """
    out_dir = BASE_DIR / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = BASE_DIR / "dataset/processed"
    graph_dir = BASE_DIR / "dataset/graph"
    y_path = proc_dir / "y.npy"
    if not y_path.exists():
        print("  [skip persistence] y.npy not found")
        return

    y_full   = np.load(y_path)              # [T, N]
    mask_full= np.load(proc_dir / "valid_mask.npy").astype(bool)  # [T, N]
    T, N     = y_full.shape

    # Test split: last 15% of T (consistent with make_splits)
    test_start = int(T * 0.85)

    try:
        nodes_df = pd.read_csv(graph_dir / "nodes.csv")
    except FileNotFoundError:
        nodes_df = pd.DataFrame({"ref": range(N), "name": [f"node_{i}" for i in range(N)]})

    print(f"\n  Computing persistence baseline (ŷ = y[t], N={N} nodes) …")
    for t_out in t_outs:
        rows = []
        for node in range(N):
            # Fix 2.7b: aggregate ALL (pred, target) pairs first, then
            # compute a single NSE over the pooled predictions.
            # The original code averaged per-window NSE values, which
            # is not the same as the NSE the trained models report (the
            # trained models compute NSE over all test predictions pooled).
            all_preds, all_targets = [], []
            sq_errors = []
            for i in range(test_start, T - t_out):
                last_obs = y_full[i, node]
                targets  = y_full[i+1:i+1+t_out, node]
                masks    = mask_full[i+1:i+1+t_out, node]
                if not masks.any():
                    continue
                t_valid = targets[masks]
                p_valid = np.full_like(t_valid, last_obs)
                all_preds.extend(p_valid.tolist())
                all_targets.extend(t_valid.tolist())
                sq_errors.extend(((t_valid - p_valid) ** 2).tolist())

            if all_targets:
                p_all = np.array(all_preds)
                t_all = np.array(all_targets)
                ss_res = float(np.sum((t_all - p_all) ** 2))
                ss_tot = float(np.sum((t_all - t_all.mean()) ** 2))
                nse  = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rmse = float(np.sqrt(np.mean(sq_errors)))
            else:
                nse, rmse = float("nan"), float("nan")

            rows.append({
                "ref":  nodes_df.iloc[node]["ref"] if "ref" in nodes_df.columns else node,
                "name": nodes_df.iloc[node].get("name", f"node_{node}"),
                "nse":  round(nse, 6),
                "rmse": round(rmse, 6),
            })

        df = pd.DataFrame(rows)
        out_path = out_dir / f"persistence_{t_out}steps.csv"
        df.to_csv(out_path, index=False)
        global_nse = df["nse"].mean()
        print(f"    T_out={t_out:3d} ({t_out*15//60}hr)  "
              f"mean NSE={global_nse:.4f}  → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════

def run(seeds: list[int], t_outs: list[int], max_epochs: int,
        skip_existing: bool, logger):
    """Train all 6 Experiment 1 models across seeds and horizons."""

    total = len(seeds) * len(t_outs) * len(MODEL_REGISTRY)
    done  = 0

    print("\n" + "═"*60)
    print("  EXPERIMENT 1 ")
    print("  Six architectures, no SAR, all horizons")
    print("═"*60)
    print(f"\n  SAR flags patched:")
    print(f"    train_st_gnn_dyn_edge  .USE_SAR      = {_mod_dyn.USE_SAR}")
    print(f"    train_st_gnn_hand_edge .USE_SAR      = {_mod_hand.USE_SAR}")
    print(f"    train_dfc_gnn          .USE_SAR_EDGE = {_mod_dfc.USE_SAR_EDGE}")
    print(f"\n  Models:   {[tag for tag, _ in MODEL_REGISTRY]}")
    print(f"  Seeds:    {seeds}")
    print(f"  Horizons: {t_outs}  steps = {[t*15//60 for t in t_outs]} hr")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Total runs: {total}")
    print()

    for seed in seeds:
        seed_everything(seed)
        for t_out in t_outs:
            for tag, fn in MODEL_REGISTRY:
                done += 1
                ckpt = BASE_DIR / "checkpoints" / tag / str(seed) / str(t_out)
                exists = (ckpt / "best_model.pt").exists()

                if skip_existing and exists:
                    print(f"[{done:3d}/{total}]  SKIP  {tag:<20} "
                          f"seed={seed}  T_out={t_out}")
                    continue

                print(f"\n[{done:3d}/{total}]  {tag:<20}  seed={seed}  "
                      f"T_in={T_IN}  T_out={t_out}  ({t_out*15//60}hr)")
                fn(logger, seed, T_IN, t_out, max_epochs, None)

    print(f"\n  Experiment 1 training complete ({done} runs).")


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Experiment 1")
    p.add_argument("--seeds",    nargs="+", type=int,   default=SEEDS)
    p.add_argument("--horizons", nargs="+", type=int,   default=T_OUTS)
    p.add_argument("--epochs",   type=int,              default=MAX_EPOCHS)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip model/seed/horizon combinations already trained")
    p.add_argument("--no-persistence", action="store_true",
                   help="Skip persistence baseline computation")
    args = p.parse_args()

    config = load_config(BASE_DIR / "config" / "config.yaml")
    logger = get_logger(config["logging"]["train"])

    if not args.no_persistence:
        compute_persistence_baseline(args.horizons)

    run(args.seeds, args.horizons, args.epochs, args.skip_existing, logger)


if __name__ == "__main__":
    main()
