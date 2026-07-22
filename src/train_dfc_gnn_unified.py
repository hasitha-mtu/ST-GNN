"""
train_dfc_gnn_unified.py  –  Physically-Complete Dynamic Connectivity GNN (proposed model)
==============================================================================================
Trains DFCGNNUnified — the unification of the three physics mechanisms
previously ablated separately (STGNNDynEdge's Manning conductance,
STGNNHANDEdge's HAND-triggered topology, DFCGNNFlood's hard elevation gate)
into one attention computation. See models/dfc_gnn_unified.py for the full
design rationale.

Topology note (read before changing edge sources)
----------------------------------------------------
This script deliberately loads edges via load_graph() + load_hand_edges() —
the SAME 28-edge river topology + HAND candidate set that STGNNDynEdge and
STGNNHANDEdge use — rather than DFCGNNFlood's separate 702-edge dense
edge_features.npz graph. This keeps the Experiment 1 ablation ladder
(PerNodeGRU/LSTM → STGNNFloodModel → STGNNDynEdge → STGNNHANDEdge →
DFCGNNUnified) on a constant graph topology, so that any NSE difference is
attributable to the mechanism being tested rather than a change in which
edges exist. node_elev (needed for the hard gate, not present in
nodes.csv/edges.csv) is the one thing still sourced from
edge_features.npz via load_edge_features — see load_node_elevation() below.

Loss: identical dual-head convention to train_dfc_gnn.py — horizon-weighted
masked MSE (stage, primary) + pos_weight-balanced BCE (flood flag,
auxiliary), λ_flood=0.1. Evaluation adds CSI/POD/FAR/F1 alongside RMSE/NSE.

Checkpoint directory: checkpoints/dfc_gnn_unified/
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config       import load_config
from src.utils.logger       import get_logger
from src.utils.common_utils import seed_everything
from src.utils.train_utils  import (
    load_graph, compute_metrics, compute_per_node_metrics,
    compute_per_step_metrics, masked_mse_horizon_weighted,
    get_split_boundary,
)
from src.utils.compile_utils import compile_model
from src.utils.gpu_sampler   import make_gpu_loaders

from src.models.dfc_gnn_unified import DFCGNNUnified
from src.models.st_gnn_hand_edge import load_hand_edges
from src.models.dfc_gnn import load_edge_features

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
PROC_DIR        = BASE_DIR / "dataset/processed"
GRAPH_DIR       = BASE_DIR / "dataset/graph"
HAND_EDGES_PATH = BASE_DIR / "dataset/graph/hand_edges.npz"
EDGE_FEAT_PATH  = BASE_DIR / "dataset/graph/edge_features.npz"

# ── Hyperparameters ────────────────────────────────────────────────────
HIDDEN_DIM   = 64
GAT_HEADS    = 4     # matches DFCGNNFlood (4 heads), not STGNNHANDEdge (2)
GRU_LAYERS   = 2
DROPOUT      = 0.1
LAMBDA_FLOOD = 0.1
TAU_GATE     = 5.0   # elevation gate softness (m)

BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 38

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_node_elevation(logger) -> torch.Tensor:
    """
    Source node_elev from edge_features.npz (the DFC-GNN data pipeline) —
    nodes.csv/edges.csv (used by load_graph) does not carry elevation.
    Raises a clear error rather than silently defaulting, since the hard
    elevation gate is a core mechanism of this model, not an optional extra.
    """
    if not EDGE_FEAT_PATH.exists():
        raise FileNotFoundError(
            f"edge_features.npz not found at {EDGE_FEAT_PATH}. "
            f"DFCGNNUnified needs node_elev for the hard elevation gate — "
            f"run: python src/data/compute_edge_features.py"
        )
    ef = load_edge_features(str(EDGE_FEAT_PATH), device="cpu")
    node_elev = ef["node_elev"]
    logger.info(
        "  node_elev loaded from edge_features.npz: %d nodes, range=[%.1f, %.1f] m OD",
        node_elev.shape[0], node_elev.min().item(), node_elev.max().item(),
    )
    return node_elev


def load_bankfull_thresholds(graph_dir: Path, n_nodes: int,
                             device: torch.device) -> torch.Tensor:
    """Per-node bankfull stage anomaly thresholds. Same as train_dfc_gnn.py."""
    bf_path = graph_dir / "bankfull_thresholds.json"
    if not bf_path.exists():
        print(f"  WARNING: bankfull_thresholds.json not found — using 0.5 m default")
        return torch.full((n_nodes,), 0.5, dtype=torch.float32, device=device)

    with open(bf_path) as f:
        data = json.load(f)
    thr_map = data.get("thresholds", {})

    nodes_df = pd.read_csv(graph_dir / "nodes.csv")
    thresholds = []
    for _, row in nodes_df.iterrows():
        ref = str(row["ref"])
        thresholds.append(float(thr_map.get(ref, 0.5)))

    return torch.tensor(thresholds, dtype=torch.float32, device=device)


def make_flood_labels(
    y_seq:        torch.Tensor,
    bankfull_thr: torch.Tensor,
    mask:         torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical to train_dfc_gnn.py's make_flood_labels."""
    stage_max  = y_seq.max(dim=1).values
    flood_flag = (stage_max > bankfull_thr.unsqueeze(0)).float()
    if mask is not None:
        node_valid = mask.any(dim=1).float()
    else:
        node_valid = torch.ones_like(flood_flag)
    return flood_flag, node_valid


# ═══════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimiser, bankfull_thr) -> dict:
    model.train()
    totals = {"loss": 0.0, "loss_stage": 0.0, "loss_flood": 0.0}

    for x_seq, y_seq, mask in loader:
        last_obs     = x_seq[:, -1, :, 0]
        delta_target = y_seq - last_obs.unsqueeze(1)
        y_flood, node_valid = make_flood_labels(y_seq, bankfull_thr, mask)

        optimiser.zero_grad(set_to_none=True)
        delta_pred, flood_logits = model(x_seq)

        loss_stage = masked_mse_horizon_weighted(delta_pred, delta_target, mask)

        _n_pos = (y_flood * node_valid).sum().clamp(min=1)
        _n_neg = ((1 - y_flood) * node_valid).sum().clamp(min=1)
        _pw    = (_n_neg / _n_pos).clamp(1.0, 100.0)
        loss_flood = F.binary_cross_entropy_with_logits(
            flood_logits, y_flood,
            weight=node_valid,
            pos_weight=_pw.unsqueeze(0).expand_as(flood_logits))

        loss = loss_stage + model.lambda_flood * loss_flood
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        totals["loss"]       += loss.item()
        totals["loss_stage"] += loss_stage.item()
        totals["loss_flood"] += loss_flood.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, bankfull_thr):
    model.eval()
    total_loss = 0.0
    flood_TP = flood_FP = flood_FN = 0.0
    all_abs_pred, all_tgt, all_mask, all_persist = [], [], [], []

    for x_seq, y_seq, mask in loader:
        last_obs     = x_seq[:, -1, :, 0]
        delta_target = y_seq - last_obs.unsqueeze(1)
        y_flood, node_valid = make_flood_labels(y_seq, bankfull_thr, mask)

        delta_pred, flood_logits = model(x_seq)
        abs_pred = last_obs.unsqueeze(1) + delta_pred

        loss_stage = masked_mse_horizon_weighted(delta_pred, delta_target, mask)
        _npos = (y_flood * node_valid).sum().clamp(min=1)
        _nneg = ((1 - y_flood) * node_valid).sum().clamp(min=1)
        _pw   = (_nneg / _npos).clamp(1.0, 100.0)
        loss_flood = F.binary_cross_entropy_with_logits(
            flood_logits, y_flood, weight=node_valid,
            pos_weight=_pw.unsqueeze(0).expand_as(flood_logits))
        total_loss += (loss_stage + model.lambda_flood * loss_flood).item()

        flood_pred = (flood_logits.sigmoid() > 0.5).float()
        _vm = node_valid.bool()
        flood_TP += (flood_pred[_vm] * y_flood[_vm]).sum().item()
        flood_FP += (flood_pred[_vm] * (1 - y_flood[_vm])).sum().item()
        flood_FN += ((1 - flood_pred[_vm]) * y_flood[_vm]).sum().item()

        all_abs_pred.append(abs_pred.cpu()); all_tgt.append(y_seq.cpu())
        all_mask.append(mask.cpu())
        all_persist.append(last_obs.unsqueeze(1).expand(-1, y_seq.shape[1], -1).cpu())

    cat_pred    = torch.cat(all_abs_pred)
    cat_tgt     = torch.cat(all_tgt)
    cat_mask    = torch.cat(all_mask)
    metrics         = compute_metrics(cat_pred, cat_tgt, cat_mask)
    persist_metrics = compute_metrics(torch.cat(all_persist), cat_tgt, cat_mask)

    denom = flood_TP + flood_FP + flood_FN
    metrics["csi"]       = round(flood_TP / max(denom, 1e-8), 4)
    metrics["pod"]       = round(flood_TP / max(flood_TP + flood_FN, 1e-8), 4)
    metrics["far"]       = round(flood_FP / max(flood_TP + flood_FP, 1e-8), 4)
    metrics["f1"]        = round(2*flood_TP / max(2*flood_TP + flood_FP + flood_FN, 1e-8), 4)
    metrics["flood_acc"] = metrics["pod"]

    return total_loss / len(loader), metrics, persist_metrics


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def train(logger, seed, t_in, t_out, max_epochs, base_dir=None):
    if base_dir is None:
        base_dir = BASE_DIR
    run_tag  = "dfc_gnn_unified"
    ckpt_dir = base_dir / "checkpoints" / run_tag / str(seed) / str(t_out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training DFC-GNN Unified (conductance + HAND topology + elevation gate) ===")
    logger.info("Device: %s", DEVICE)

    # ── Load gauge data ────────────────────────────────────────────────
    logger.info("Loading dataset …")
    X          = np.load(PROC_DIR / "X.npy")
    y          = np.load(PROC_DIR / "y.npy")
    valid_mask = np.load(PROC_DIR / "valid_mask.npy")
    T, N, F_dyn = X.shape
    logger.info("  X: %s  y: %s  valid_mask: %s", X.shape, y.shape, valid_mask.shape)

    # ── Load river topology (SAME as STGNNDynEdge/STGNNHANDEdge) ───────
    edge_index, edge_attr, _node_attr_unused = load_graph(logger, GRAPH_DIR, DEVICE)
    logger.info("  River edges: %d (28-edge topology, shared with ablation ladder)",
                edge_index.shape[1])

    # ── Load HAND candidate edges ───────────────────────────────────────
    if not HAND_EDGES_PATH.exists():
        raise FileNotFoundError(
            f"hand_edges.npz not found at {HAND_EDGES_PATH}. "
            f"Run: python src/precompute_hand_edges.py"
        )
    hand_data = load_hand_edges(str(HAND_EDGES_PATH))
    logger.info("  HAND candidate edges: %d", hand_data["src"].shape[0])

    # ── Load node elevation (for the hard gate) ─────────────────────────
    node_elev = load_node_elevation(logger).to(DEVICE)

    # ── Load bankfull thresholds (for flood label generation) ──────────
    bankfull_thr = load_bankfull_thresholds(GRAPH_DIR, N, DEVICE)

    # ── Discharge reference (train split only, same as STGNNDynEdge) ───
    discharge_col = 3
    t1, _ = get_split_boundary(T)
    q_ref = float(X[:t1, :, discharge_col].mean())
    q_ref = max(q_ref, 0.01)
    logger.info("  Discharge reference Q_ref=%.4f", q_ref)

    # ── Dataloaders ───────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_gpu_loaders(
        X, y, valid_mask, t_in=t_in, t_out=t_out,
        batch_size=BATCH_SIZE, device=DEVICE,
    )

    # ── Model ──────────────────────────────────────────────────────────
    model = DFCGNNUnified(
        n_nodes=N, f_dyn=F_dyn, d_model=HIDDEN_DIM, n_heads=GAT_HEADS,
        T_out=t_out,
        edge_index=edge_index, edge_attr_static=edge_attr, node_elev=node_elev,
        hand_src=hand_data["src"].to(DEVICE), hand_dst=hand_data["dst"].to(DEVICE),
        hand_threshold=hand_data["hand_threshold"].to(DEVICE),
        hand_overland_dist=hand_data["overland_dist_km"].to(DEVICE),
        n_gru_layers=GRU_LAYERS, dropout=DROPOUT, lambda_flood=LAMBDA_FLOOD,
        tau_gate=TAU_GATE, discharge_idx=discharge_col, discharge_ref=q_ref,
    ).to(DEVICE)

    model = compile_model(model, tag=run_tag, logger=logger)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")
    logger.info(
        "  River edges: %d  HAND edges: %d  λ_flood=%.3f  τ_gate=%.1f m",
        edge_index.shape[1], hand_data["src"].shape[0], LAMBDA_FLOOD, TAU_GATE,
    )

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=11, cooldown=2, min_lr=1e-6
    )

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss = math.inf
    patience_ctr  = 0
    history       = []

    logger.info("Starting training …")
    for epoch in range(1, max_epochs + 1):
        print(f"epoch: {epoch}")

        train_metrics = train_epoch(model, train_loader, optimiser, bankfull_thr)
        val_loss, val_metrics, persist_metrics = eval_epoch(model, val_loader, bankfull_thr)
        scheduler.step(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]
        if current_lr <= 1e-6:
            logger.info("  LR floor reached — stopping")
            break

        history.append({
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_loss_stage": round(train_metrics["loss_stage"], 6),
            "train_loss_flood": round(train_metrics["loss_flood"], 6),
            "val_loss": round(val_loss, 6),
            "es_counter": patience_ctr,
            **{f"val_{k}": (round(v, 4) if isinstance(v, float) else v)
               for k, v in val_metrics.items()},
        })

        logger.info(
            "Epoch %3d  train=%.6e  val=%.6e  ES=%2d/%2d  "
            "Model RMSE=%.4f NSE=%.4f CSI=%.4f  |  "
            "Persist RMSE=%.4f NSE=%.4f  LR=%.1e  "
            "conductance_scale=%.3f  activation_sharpness=%.3f",
            epoch, train_metrics["loss"], val_loss, patience_ctr, PATIENCE,
            val_metrics["rmse"], val_metrics["nse"], val_metrics["csi"],
            persist_metrics["rmse"], persist_metrics["nse"], current_lr,
            model.conductance_scale.item(), model.activation_sharpness.item(),
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
                "hparams": {
                    "t_in": t_in, "t_out": t_out, "f_dyn": F_dyn,
                    "hidden": HIDDEN_DIM, "gat_heads": GAT_HEADS,
                    "gru_layers": GRU_LAYERS, "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE, "lr": LR,
                    "lambda_flood": LAMBDA_FLOOD, "tau_gate": TAU_GATE,
                    "discharge_ref": q_ref,
                    "n_river_edges": edge_index.shape[1],
                    "n_hand_edges": hand_data["src"].shape[0],
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
    flood_TP = flood_FP = flood_FN = 0.0
    with torch.no_grad():
        for x_seq, y_seq, mask in test_loader:
            x_seq    = x_seq.to(DEVICE)
            last_obs = x_seq[:, -1, :, 0]
            delta_pred, flood_logits = model(x_seq)
            abs_pred = last_obs.unsqueeze(1) + delta_pred

            y_flood, node_valid = make_flood_labels(y_seq.to(DEVICE), bankfull_thr,
                                                     mask.to(DEVICE))
            flood_pred = (flood_logits.sigmoid() > 0.5).float()
            _vm = node_valid.bool()
            flood_TP += (flood_pred[_vm] * y_flood[_vm]).sum().item()
            flood_FP += (flood_pred[_vm] * (1 - y_flood[_vm])).sum().item()
            flood_FN += ((1 - flood_pred[_vm]) * y_flood[_vm]).sum().item()

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

    denom = flood_TP + flood_FP + flood_FN
    test_metrics["csi"] = round(flood_TP / max(denom, 1e-8), 4)
    test_metrics["pod"] = round(flood_TP / max(flood_TP + flood_FN, 1e-8), 4)
    test_metrics["far"] = round(flood_FP / max(flood_TP + flood_FP, 1e-8), 4)
    test_metrics["f1"]  = round(2*flood_TP / max(2*flood_TP + flood_FP + flood_FN, 1e-8), 4)

    logger.info(
        "\n✓ Test results:\n"
        "  RMSE: %.4f\n  MAE:  %.4f\n  NSE:  %.4f\n  MBE:  %.4f m\n"
        "  CSI:  %.4f\n  POD:  %.4f\n  FAR:  %.4f\n  F1:   %.4f",
        test_metrics["rmse"], test_metrics["mae"], test_metrics["nse"], mbe_global,
        test_metrics["csi"], test_metrics["pod"], test_metrics["far"], test_metrics["f1"],
    )
    logger.info(
        "  Learned physical parameters — conductance_scale=%.3f  activation_sharpness=%.3f",
        model.conductance_scale.item(), model.activation_sharpness.item(),
    )

    # ── Per-node metrics ───────────────────────────────────────────────
    nodes_df     = pd.read_csv(GRAPH_DIR / "nodes.csv")
    node_rows    = compute_per_node_metrics(cat_pred,    cat_tgt, cat_mask)
    persist_rows = compute_per_node_metrics(cat_persist, cat_tgt, cat_mask)

    pn_df = pd.DataFrame(node_rows)
    pn_df["ref"]         = nodes_df["ref"].astype(str).values
    pn_df["name"]        = nodes_df["name"].values
    pn_df["persist_nse"] = [r["nse"] for r in persist_rows]
    pn_df["skill"] = (
        (pn_df["nse"] - pn_df["persist_nse"])
        / (1 - pn_df["persist_nse"]).clip(lower=1e-8)
    ).round(4)

    pn_df = pn_df[["ref", "name", "n_valid", "rmse", "mae", "mbe",
                   "nse", "persist_nse", "skill"]]
    pn_df.to_csv(ckpt_dir / "per_node_metrics.csv", index=False)
    logger.info("  Saved per_node_metrics.csv")

    # ── Aggregate + per-step metrics ────────────────────────────────────
    with open(ckpt_dir / "test_metrics.json", "w") as f:
        json.dump({
            **test_metrics,
            "mbe":   round(mbe_global, 6),
            "model": "dfc_gnn_unified",
            "conductance_scale": round(model.conductance_scale.item(), 4),
            "activation_sharpness": round(model.activation_sharpness.item(), 4),
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
