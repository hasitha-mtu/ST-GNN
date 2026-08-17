"""
train_st_gnn_backwater_edge.py  –  PI-ST-GNN with gated bridge/culvert backwater edges
========================================================================================
Architecture
------------
  1. Node feature projection   Linear(F_dyn + F_static) → hidden_dim
  2. Temporal encoder          Per-node GRU → hidden state [B, N, hidden]
  3. Graph message passing     GATConv ×3 layers over the COMBINED river +
                                gated backwater edge set (batched)
  4. Output head               Linear → delta stage_anomaly [B, T_out, N]

Adds a third, conditionally-gated edge class to the static river graph:
reverse pathways at bridge/culvert constriction points, from
precompute_backwater_edges.py, gated open only when the constriction
node's own reconstructed water level approaches its own p90 flood
threshold. See st_gnn_backwater_edge.py's module docstring for the full
physical motivation and scenario_generator.py's generate_s6_channel_blockage
for the diagnostic scenario this model exists to improve on.

Comparison matrix (extends the matrix in train_st_gnn_flood_model.py):
  PerNodeGRU              — no graph, no SAR  (temporal lower bound)
  PerNodeLSTM              — no graph, no SAR  (temporal lower bound)
  STGNNFlood                — static graph, no SAR
  STGNNFloodDynEdge         — dynamic edge weights     (Phase 1)
  STGNNFloodHAND            — dynamic topology         (Phase 2)
  STGNNBackwaterEdge (this) — static river graph + gated backwater edges
                              at bridge/culvert locations (S6 diagnostic)

Input window:   T_in  = 32 steps  (8 hours at 15-min resolution)
Output horizon: T_out configurable (matches train_models_exp1.py's horizon sweep)

NOTE: unlike STGNNDynEdge/STGNNHANDEdge/STGNNSoilGate, this model has NO
SAR fusion path — same "Experiment 1 scope only" precedent as
DFC-GNN Unified (see train_models_exp1.py, which does not patch a
USE_SAR flag for dfc_gnn_unified either). If SAR fusion is added later,
follow the same sar_emb_dim constructor pattern used in st_gnn_flood.py.

Gate collapse — why this training script includes a sparsity penalty
-----------------------------------------------------------------------
STGNNSoilGate's sat_threshold was diagnosed (separately, on the
historical training runs) to collapse toward the low end of its range
across every checkpoint regardless of seed or horizon — the gate ends
up essentially always-open, not because that's physically correct, but
because plain MSE gives a sigmoid gate a ONE-DIRECTIONAL incentive:
opening it further (more cross-tributary/backwater context) rarely
increases loss and often decreases it, while closing it risks losing
genuinely useful information and increasing loss. Nothing in an
unregularised MSE objective pushes the other way, so gradient descent
walks the threshold toward "always open" and stays there — the gate
becomes decorative, not load-bearing, and the physical interpretation
of gate_sharpness/gate_reference_m stops meaning anything.

This model's gate has the identical shape (sigmoid on a learnable
threshold, no built-in cost for staying open), so it's exposed to the
same risk. train_epoch below adds an explicit sparsity penalty —
LAMBDA_GATE_SPARSITY * model.last_gate_mean — to the training loss
(NOT the validation/test loss, which should reflect true predictive
performance only) so there is now a real cost to keeping the gate open,
counterbalancing MSE's implicit incentive. gate_mean_activation is
logged every epoch in training_history.csv specifically so this can be
checked rather than assumed — if it still saturates toward an extreme,
LAMBDA_GATE_SPARSITY needs tuning before the learned gate is trustworthy.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.common_utils import seed_everything
from src.utils.train_utils import load_graph
from src.utils.train_utils import compute_metrics
from src.utils.train_utils import compute_per_node_metrics
from src.utils.train_utils import compute_per_step_metrics
from src.utils.train_utils import masked_mse_horizon_weighted
from src.utils.compile_utils import compile_model
from src.utils.gpu_sampler import make_gpu_loaders

from src.models.st_gnn_backwater_edge import STGNNBackwaterEdge

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
BACKWATER_EDGES_PATH = GRAPH_DIR / "backwater_edges.npz"

# ── Hyperparameters ────────────────────────────────────────────────────
HIDDEN_DIM = 64
GAT_HEADS  = 2
GRU_LAYERS = 2
DROPOUT    = 0.1
GATE_SHARPNESS_INIT = 5.0    # beta in the backwater activation sigmoid

# Sparsity penalty weight — see the module docstring's "Gate collapse"
# section. Without this, plain MSE gives the gate a one-directional
# incentive to stay open (more context rarely increases loss, closing
# risks losing real signal, nothing pushes back the other way), which is
# the exact mechanism diagnosed behind STGNNSoilGate's sat_threshold
# collapsing toward zero across every checkpoint in that model. Start at
# this value and watch gate_mean_activation in training_history.csv —
# if it still saturates toward ~1.0 (always open) or ~0.0 (always
# closed, meaning the penalty overpowered real signal) across epochs,
# LAMBDA_GATE_SPARSITY needs tuning in the corresponding direction
# before trusting the learned gate_sharpness/gate behaviour.
LAMBDA_GATE_SPARSITY = 0.01

# Column index of normalised_stage in the dynamic feature vector.
# Confirm against build_dataset.py's GAUGE_FEATURES ordering before
# training — a wrong index here silently gates on the wrong signal
# (see st_gnn_backwater_edge.py's docstring, and the same caveat raised
# for STGNNSoilGate's swvl2_sat_idx).
STAGE_IDX = 1

BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 38

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
#  Backwater edge loading
# ═══════════════════════════════════════════════════════════════════════

def load_backwater_edges(logger, graph_dir: Path, nodes_df: pd.DataFrame,
                         device: torch.device):
    """
    Loads backwater_edges.npz (from precompute_backwater_edges.py) and
    derives the per-node gauge_datum/stage_range tensors needed for the
    model's H = datum + normalised_stage * stage_range reconstruction —
    same convention as STGNNHANDEdge's saddle-elevation gate.
    """
    path = graph_dir / "backwater_edges.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run precompute_backwater_edges.py first.")

    bw = np.load(path, allow_pickle=True)
    bw_src = torch.from_numpy(bw["src"].astype(np.int64)).to(device)
    bw_dst = torch.from_numpy(bw["dst"].astype(np.int64)).to(device)
    bw_edge_attr = torch.from_numpy(np.stack([
        bw["river_dist_km"], bw["area_ratio"],
        bw["elev_drop_m"], bw["same_tributary"],
    ], axis=1).astype(np.float32)).to(device)
    bw_gate_reference = torch.from_numpy(
        bw["gate_reference_m"].astype(np.float32)).to(device)

    gauge_datum = torch.from_numpy(
        nodes_df["gauge_datum_mOSGM15"].values.astype(np.float32)).to(device)
    stage_range = torch.from_numpy(
        (nodes_df["p90_mAOD"] - nodes_df["gauge_datum_mOSGM15"]).values
        .astype(np.float32)).to(device)

    logger.info("  Backwater edges: %d  (bridges: %s)",
               len(bw_src), list(bw["bridge_name"]))

    return bw_src, bw_dst, bw_edge_attr, bw_gate_reference, gauge_datum, stage_range


# ═══════════════════════════════════════════════════════════════════════
#  Training loop  — same structure as train_st_gnn_flood_model.py
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimiser, edge_index, edge_attr, node_attr,
                lambda_gate_sparsity: float = 0.0) -> tuple[float, float]:
    """
    Returns (mean_mse_loss, mean_gate_activation) — the sparsity penalty
    is applied to the backward pass but NOT included in the returned
    loss value, so training_history.csv's train_loss stays comparable
    across LAMBDA_GATE_SPARSITY settings and comparable to val_loss
    (which intentionally excludes the penalty — see module docstring).
    """
    model.train()
    total_mse_loss  = 0.0
    total_gate_mean = 0.0

    for x_seq, y_seq, mask in loader:
        last_obs     = x_seq[:, -1, :, 0]
        delta_target = y_seq - last_obs.unsqueeze(1)

        optimiser.zero_grad(set_to_none=True)
        delta_pred = model(x_seq, node_attr, edge_index, edge_attr)
        mse_loss = masked_mse_horizon_weighted(delta_pred, delta_target, mask)

        # last_gate_mean is set during the forward() call above (inside
        # _backwater_gate) — see st_gnn_backwater_edge.py for why this is
        # a plain attribute read rather than a second return value.
        gate_mean = getattr(model, "_orig_mod", model).last_gate_mean
        penalty = lambda_gate_sparsity * gate_mean
        loss = mse_loss + penalty

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_mse_loss  += mse_loss.item()
        total_gate_mean += gate_mean.item()

    n = len(loader)
    return total_mse_loss / n, total_gate_mean / n


@torch.no_grad()
def eval_epoch(model, loader, edge_index, edge_attr, node_attr):
    model.eval()
    total_loss = 0.0
    all_abs_pred, all_tgt, all_mask, all_persist = [], [], [], []

    for x_seq, y_seq, mask in loader:
        last_obs     = x_seq[:, -1, :, 0]
        delta_target = y_seq - last_obs.unsqueeze(1)

        delta_pred = model(x_seq, node_attr, edge_index, edge_attr)
        abs_pred   = last_obs.unsqueeze(1) + delta_pred

        total_loss += masked_mse_horizon_weighted(delta_pred, delta_target, mask).item()

        all_abs_pred.append(abs_pred.cpu())
        all_tgt.append(y_seq.cpu())
        all_mask.append(mask.cpu())
        all_persist.append(last_obs.unsqueeze(1).expand(-1, y_seq.shape[1], -1).cpu())

    cat_abs_pred = torch.cat(all_abs_pred)
    cat_tgt      = torch.cat(all_tgt)
    cat_mask     = torch.cat(all_mask)

    metrics         = compute_metrics(cat_abs_pred, cat_tgt, cat_mask)
    persist_metrics = compute_metrics(torch.cat(all_persist), cat_tgt, cat_mask)
    return total_loss / len(loader), metrics, persist_metrics


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def train(logger, seed, t_in, t_out, max_epochs, base_dir=None):
    if base_dir is None:
        base_dir = BASE_DIR
    run_tag  = "st_gnn_backwater_edge"
    ckpt_dir = base_dir / "checkpoints" / run_tag / str(seed) / str(t_out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training ST-GNN + gated backwater edges ===")
    logger.info("Device: %s", DEVICE)

    # ── Load gauge data ────────────────────────────────────────────────
    logger.info("Loading dataset …")
    X          = np.load(PROC_DIR / "X.npy")
    y          = np.load(PROC_DIR / "y.npy")
    valid_mask = np.load(PROC_DIR / "valid_mask.npy")
    T, N, F    = X.shape
    logger.info("  X: %s  y: %s  valid_mask: %s", X.shape, y.shape, valid_mask.shape)

    # ── Load river graph (existing utility) ────────────────────────────
    edge_index, edge_attr, node_attr = load_graph(logger, GRAPH_DIR, DEVICE)

    # ── Load backwater edges (new) ──────────────────────────────────────
    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    (bw_src, bw_dst, bw_edge_attr, bw_gate_reference,
     gauge_datum, stage_range) = load_backwater_edges(
        logger, GRAPH_DIR, nodes_df, DEVICE)

    # ── Dataloaders ─────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_gpu_loaders(
        X, y, valid_mask, t_in=t_in, t_out=t_out,
        batch_size=BATCH_SIZE, device=DEVICE,
    )

    # ── Model ──────────────────────────────────────────────────────────
    f_static = node_attr.shape[1]
    model = STGNNBackwaterEdge(
        f_dyn=F,
        f_static=f_static,
        edge_index=edge_index,
        edge_attr=edge_attr,
        bw_src=bw_src,
        bw_dst=bw_dst,
        bw_edge_attr=bw_edge_attr,
        bw_gate_reference=bw_gate_reference,
        gauge_datum=gauge_datum,
        stage_range=stage_range,
        stage_idx=STAGE_IDX,
        hidden=HIDDEN_DIM,
        gat_heads=GAT_HEADS,
        gru_layers=GRU_LAYERS,
        t_out=t_out,
        dropout=DROPOUT,
        gate_sharpness=GATE_SHARPNESS_INIT,
    ).to(DEVICE)

    model = compile_model(model, tag=run_tag, logger=logger)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")
    logger.info("River edges: %d   Backwater edges: %d", model.n_river, model.n_bw)

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=11, cooldown=2, min_lr=1e-6)

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = math.inf
    patience_ctr  = 0
    history       = []

    logger.info("Starting training …")
    for epoch in range(1, max_epochs + 1):
        print(f'epoch: {epoch}')

        train_loss, gate_mean_activation = train_epoch(
            model, train_loader, optimiser, edge_index, edge_attr, node_attr,
            lambda_gate_sparsity=LAMBDA_GATE_SPARSITY)
        val_loss, val_metrics, persist_metrics = eval_epoch(
            model, val_loader, edge_index, edge_attr, node_attr)
        scheduler.step(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]
        if current_lr <= 1e-6:
            logger.info("  LR floor reached — stopping")
            break

        gate_sharpness_val = model.gate_sharpness.item() if hasattr(model, "gate_sharpness") \
            else getattr(model, "_orig_mod", model).gate_sharpness.item()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "es_counter": patience_ctr,
            "gate_sharpness": round(gate_sharpness_val, 4),
            "gate_mean_activation": round(gate_mean_activation, 4),
            "lambda_gate_sparsity": LAMBDA_GATE_SPARSITY,
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        })

        logger.info(
            "Epoch %3d  train=%.6e  val=%.6e  ES=%2d/%2d  "
            "Model RMSE=%.4f NSE=%.4f  |  "
            "Persist RMSE=%.4f NSE=%.4f  gate_sharpness=%.3f  "
            "gate_mean_act=%.3f  LR=%.1e",
            epoch, train_loss, val_loss, patience_ctr, PATIENCE,
            val_metrics["rmse"], val_metrics["nse"],
            persist_metrics["rmse"], persist_metrics["nse"],
            gate_sharpness_val, gate_mean_activation, current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save({
                "epoch":       epoch,
                "state_dict":  model.state_dict(),
                "optimiser":   optimiser.state_dict(),
                "val_loss":    val_loss,
                "val_metrics": val_metrics,
                "use_sar":     False,
                "hparams": {
                    "t_in": t_in, "t_out": t_out,
                    "f_dyn": F,
                    "hidden": HIDDEN_DIM, "gat_heads": GAT_HEADS,
                    "gru_layers": GRU_LAYERS, "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE, "lr": LR,
                    "stage_idx": STAGE_IDX,
                    "n_backwater_edges": model.n_bw,
                    "gate_sharpness_init": GATE_SHARPNESS_INIT,
                    "lambda_gate_sparsity": LAMBDA_GATE_SPARSITY,
                },
            }, ckpt_dir / "best_model.pt")
            logger.info("  ✓ Saved best model (val_loss=%.4f)", val_loss)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    pd.DataFrame(history).to_csv(ckpt_dir / "training_history.csv", index=False)

    # ── Test evaluation ────────────────────────────────────────────────
    logger.info("Loading best model for test evaluation …")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_abs_pred, all_tgt, all_mask, all_persist = [], [], [], []
    with torch.no_grad():
        for x_seq, y_seq, mask in test_loader:
            x_seq    = x_seq.to(DEVICE)
            last_obs = x_seq[:, -1, :, 0]

            delta_pred = model(x_seq, node_attr, edge_index, edge_attr)
            abs_pred   = last_obs.unsqueeze(1) + delta_pred

            all_abs_pred.append(abs_pred.cpu())
            all_tgt.append(y_seq.cpu())
            all_mask.append(mask.cpu())
            all_persist.append(last_obs.unsqueeze(1).expand(-1, t_out, -1).cpu())

    cat_pred    = torch.cat(all_abs_pred).cpu()
    cat_tgt     = torch.cat(all_tgt).cpu()
    cat_mask    = torch.cat(all_mask).cpu()
    cat_persist = torch.cat(all_persist).cpu()

    test_metrics = compute_metrics(cat_pred, cat_tgt, cat_mask)
    m_all        = cat_mask.bool()
    mbe_global   = (cat_pred[m_all] - cat_tgt[m_all]).mean().item()

    logger.info(
        "\n✓ Test results:\n"
        "  RMSE: %.4f\n  MAE:  %.4f\n  NSE:  %.4f\n  MBE:  %.4f m",
        test_metrics["rmse"], test_metrics["mae"],
        test_metrics["nse"],  mbe_global,
    )

    # ── Per-node metrics ───────────────────────────────────────────────
    node_rows    = compute_per_node_metrics(cat_pred,    cat_tgt, cat_mask)
    persist_rows = compute_per_node_metrics(cat_persist, cat_tgt, cat_mask)

    pn_df = pd.DataFrame(node_rows)
    pn_df["ref"]          = nodes_df["ref"].astype(str).values
    pn_df["name"]         = nodes_df["name"].values
    pn_df["persist_nse"]  = [r["nse"] for r in persist_rows]
    pn_df["skill"]        = (
        (pn_df["nse"] - pn_df["persist_nse"])
        / (1 - pn_df["persist_nse"]).clip(lower=1e-8)
    ).round(4)
    # Flag the backwater/bridge nodes specifically — the S6 diagnostic
    # tells you this model SHOULD show its biggest improvement (if any)
    # right here, since these are the only nodes with a message-passing
    # pathway that even could improve on the directed-only baseline.
    bw_node_idx = set(np.load(BACKWATER_EDGES_PATH, allow_pickle=True)["src"].tolist()) \
        if BACKWATER_EDGES_PATH.exists() else set()
    pn_df["is_backwater_bridge_node"] = [
        i in bw_node_idx for i in range(len(pn_df))
    ]

    pn_df = pn_df[["ref", "name", "n_valid", "rmse", "mae",
                   "mbe", "nse", "persist_nse", "skill",
                   "is_backwater_bridge_node"]]
    pn_df.to_csv(ckpt_dir / "per_node_metrics.csv", index=False)
    logger.info("  Saved per_node_metrics.csv")

    # ── Aggregate + per-step metrics ───────────────────────────────────
    with open(ckpt_dir / "test_metrics.json", "w") as f:
        json.dump({
            **test_metrics,
            "mbe":     round(mbe_global, 6),
            "use_sar": False,
            "model":   "st_gnn_backwater_edge",
            "n_backwater_edges": model.n_bw,
        }, f, indent=2)

    per_step         = compute_per_step_metrics(cat_pred,    cat_tgt, cat_mask)
    per_step_persist = compute_per_step_metrics(cat_persist, cat_tgt, cat_mask)

    for h_dict, p_dict in zip(per_step, per_step_persist):
        pn = p_dict["nse"]
        mn = h_dict["nse"]
        if not (np.isnan(pn) or np.isnan(mn)) and pn < 1.0:
            h_dict["persist_nse"] = round(pn, 6)
            h_dict["skill"]       = round((mn - pn) / (1 - pn), 4)
        else:
            h_dict["persist_nse"] = float("nan")
            h_dict["skill"]       = float("nan")

    with open(ckpt_dir / "per_step_metrics.json", "w") as f:
        json.dump(per_step, f, indent=2)
    logger.info("  Saved per_step_metrics.json (%d steps)", len(per_step))

    return model, test_metrics


if __name__ == "__main__":
    seed       = 42
    t_in       = 32
    t_out      = 4
    max_epochs = 2
    seed_everything(seed)
    config = load_config(BASE_DIR / "config" / "config.yaml")
    logger = get_logger(config["logging"]["train"])
    train(logger, seed, t_in, t_out, max_epochs)
