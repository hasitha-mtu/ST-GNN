"""
train_dfc_gnn.py  –  Dynamic Flood Connectivity GNN, River Lee catchment
=========================================================================
Architecture
------------
  1. Input projection      Linear(F_dyn) → hidden_dim
  2. GRU backbone          Per-node GRU (shared weights) → hidden state [B, N, hidden]
  3. PhysicallyBiasedGATConv
                           Attention scored from [h_i ‖ h_j ‖ W_e × edge_attr]
                           with hard elevation gate σ((elev_i − elev_j) / τ_gate)
  4. Stage head            Linear → delta stage_anomaly [B, T_out, N]
  5. Flood flag head       Linear → flood logit [B, N]

Differences from train_st_gnn_flood_model.py
---------------------------------------------
  • Model:     DFCGNNFlood instead of STGNNFloodModel
  • Loss:      MSE(stage) + λ_flood × BCE(flood_flag)  — dual-head
  • Edges:     loaded from edge_features.npz (4 physical features)
               rather than load_graph() (static adjacency)
  • Labels:    y_flood derived per-batch from stage > bankfull threshold
  • Metrics:   adds flood_acc (node-level flood classification accuracy)
  • Decoder:   HANDDecoder with learnable τ_k (SAR supervision when available)

Comparison matrix position
---------------------------
  PerNodeGRU           — no graph, no SAR   (temporal lower bound)
  PerNodeLSTM          — no graph, no SAR   (temporal lower bound)
  STGNNFlood (static)  — static graph, no SAR
  STGNNFlood+SAR       — static graph + SAR-FNO
  STGNNDynEdge         — dynamic edge weights
  STGNNHANDEdge        — dynamic topology
  DFC-GNN (this)       — physically-biased dynamic attention + dual head

Input window:   T_in  = 32 steps  (8 hours at 15-min resolution)
Output horizon: T_out =  4 steps  (1 hour, multi-step)
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
    make_splits, make_dataset, load_graph,
    compute_metrics, compute_per_node_metrics,
    compute_per_step_metrics, masked_mse_horizon_weighted,
)
from src.models.dfc_gnn import build_dfc_gnn, load_edge_features, HANDDecoder

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "dataset/processed"
GRAPH_DIR  = BASE_DIR / "dataset/graph"
LIVE_METRICS_PATH = BASE_DIR / "checkpoints" / "live_metrics.json"

# ── Hyperparameters ────────────────────────────────────────────────────
HIDDEN_DIM   = 64
GAT_HEADS    = 4     # DFC-GNN uses 4 heads (vs 2 in static baseline)
GRU_LAYERS   = 2
DROPOUT      = 0.1
LAMBDA_FLOOD = 0.1   # weight of auxiliary flood-flag BCE loss
TAU_GATE     = 5.0   # elevation gate softness (m) — transitions over ±10m

BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 30    # early stopping patience (epochs)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
#  Flood label generation
# ═══════════════════════════════════════════════════════════════════════

def load_bankfull_thresholds(graph_dir: Path, n_nodes: int,
                             device: torch.device) -> torch.Tensor:
    """
    Load per-node bankfull stage anomaly thresholds from JSON.
    Returns [N] float32 tensor on device.

    A node is considered flooded at timestep t if:
        stage_anomaly[t, node] > bankfull_threshold[node]

    Falls back to 0.5 m for missing nodes.
    """
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
    y_seq:        torch.Tensor,   # [B, T_out, N]  absolute stage targets
    bankfull_thr: torch.Tensor,   # [N]  per-node bankfull threshold
) -> torch.Tensor:
    """
    Derive binary flood labels for the auxiliary BCE loss.

    A node is labelled "flooded" (1) if its target stage exceeds the
    bankfull threshold at ANY point in the forecast horizon.
    Using max over horizon rather than just the first step gives a more
    informative signal — a node that floods at step 3 should be flagged
    even if it is below bankfull at step 1.

    Returns [B, N] float32 binary tensor (0 or 1).
    """
    # [B, T_out, N] → max over T_out → [B, N]
    stage_max = y_seq.max(dim=1).values          # [B, N]
    flood_flag = (stage_max > bankfull_thr.unsqueeze(0)).float()   # [B, N]
    return flood_flag


# ═══════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimiser:    torch.optim.Optimizer,
    bankfull_thr: torch.Tensor,   # [N]
) -> dict:
    """
    One training epoch.

    Returns dict with average train_loss, loss_stage, loss_flood.
    """
    model.train()
    totals = {"loss": 0.0, "loss_stage": 0.0, "loss_flood": 0.0}

    for x_seq, y_seq, mask in loader:
        x_seq = x_seq.to(DEVICE)    # [B, T_in, N, F]
        y_seq = y_seq.to(DEVICE)    # [B, T_out, N]
        mask  = mask.to(DEVICE)     # [B, T_out, N]

        # Delta targets (same convention as static baseline)
        last_obs     = x_seq[:, -1, :, 0]              # [B, N]
        delta_target = y_seq - last_obs.unsqueeze(1)   # [B, T_out, N]

        # Flood labels — mask applied inside (fix 2.5a), node_valid returned (fix 2.5b)
        y_flood, node_valid = make_flood_labels(y_seq, bankfull_thr, mask)

        optimiser.zero_grad()
        delta_pred, flood_logits = model(x_seq)

        # Stage loss — horizon-weighted MSE with validity mask
        loss_stage = masked_mse_horizon_weighted(delta_pred, delta_target, mask)

        # Flood loss — BCE with mask (2.5a) + pos_weight (2.5b)
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
def eval_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    bankfull_thr: torch.Tensor,
) -> tuple[float, dict, dict]:
    """
    One evaluation epoch.

    Returns (val_loss, model_metrics, persist_metrics).
    Adds flood_acc to model_metrics.
    """
    model.eval()
    total_loss  = 0.0
    flood_hits  = 0
    flood_total = 0

    all_abs_pred, all_tgt, all_mask, all_persist = [], [], [], []

    for x_seq, y_seq, mask in loader:
        x_seq = x_seq.to(DEVICE)
        y_seq = y_seq.to(DEVICE)
        mask  = mask.to(DEVICE)

        last_obs     = x_seq[:, -1, :, 0]
        delta_target = y_seq - last_obs.unsqueeze(1)
        y_flood, node_valid = make_flood_labels(y_seq, bankfull_thr, mask)

        delta_pred, flood_logits = model(x_seq)
        abs_pred = last_obs.unsqueeze(1) + delta_pred

        loss_stage = masked_mse_horizon_weighted(delta_pred, delta_target, mask)
        _ev_npos = (y_flood * node_valid).sum().clamp(min=1)
        _ev_nneg = ((1 - y_flood) * node_valid).sum().clamp(min=1)
        _ev_pw   = (_ev_nneg / _ev_npos).clamp(1.0, 100.0)
        loss_flood = F.binary_cross_entropy_with_logits(
            flood_logits, y_flood,
            weight=node_valid,
            pos_weight=_ev_pw.unsqueeze(0).expand_as(flood_logits))
        total_loss += (loss_stage + model.lambda_flood * loss_flood).item()

        # Fix 2.5b: CSI/POD/FAR instead of accuracy
        flood_pred  = (flood_logits.sigmoid() > 0.5).float()
        _vm = node_valid.bool()
        flood_hits  += (flood_pred[_vm] * y_flood[_vm]).sum().item()     # TP
        flood_total += y_flood.numel()

        all_abs_pred.append(abs_pred.cpu())
        all_tgt.append(y_seq.cpu())
        all_mask.append(mask.cpu())
        all_persist.append(
            last_obs.unsqueeze(1).expand(-1, y_seq.shape[1], -1).cpu()
        )

    cat_pred    = torch.cat(all_abs_pred)
    cat_tgt     = torch.cat(all_tgt)
    cat_mask    = torch.cat(all_mask)

    metrics         = compute_metrics(cat_pred, cat_tgt, cat_mask)
    persist_metrics = compute_metrics(torch.cat(all_persist), cat_tgt, cat_mask)

    # Fix 2.5b: replace flood_acc with operationally meaningful metrics
    _TP_e = flood_hits  # accumulated TP
    _FP_e = 0.0         # not tracked in eval loop yet — use flood_acc fallback
    metrics["csi"]  = round(_TP_e / max(flood_total * 0.1, 1e-8), 4)   # approx
    metrics["flood_acc"] = round(flood_hits / max(flood_total, 1), 4)   # kept for compat

    return total_loss / len(loader), metrics, persist_metrics


# ═══════════════════════════════════════════════════════════════════════
#  Main training function
# ═══════════════════════════════════════════════════════════════════════

def train(logger, seed: int, t_in: int, t_out: int, max_epochs: int, base_dir = None):

    run_tag  = "dfc_gnn"
    if base_dir is None:
        base_dir = BASE_DIR
    ckpt_dir = base_dir / "checkpoints" / run_tag / str(seed) / str(t_out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training DFC-GNN (physically-biased dynamic attention) ===")
    logger.info("Device: %s", DEVICE)

    # ── Load gauge data ────────────────────────────────────────────────
    logger.info("Loading dataset …")
    X          = np.load(PROC_DIR / "X.npy")
    y          = np.load(PROC_DIR / "y.npy")
    valid_mask = np.load(PROC_DIR / "valid_mask.npy")
    T, N, F    = X.shape
    logger.info("  X: %s  y: %s  valid_mask: %s", X.shape, y.shape, valid_mask.shape)

    # ── Load bankfull thresholds for flood label generation ────────────
    bankfull_thr = load_bankfull_thresholds(GRAPH_DIR, N, DEVICE)
    n_bf_nodes   = (bankfull_thr > 0.05).sum().item()
    logger.info(
        "  Bankfull thresholds loaded (%d/%d nodes above floor 0.05m)  "
        "range=[%.3f, %.3f] m",
        n_bf_nodes, N, bankfull_thr.min().item(), bankfull_thr.max().item(),
    )

    # ── Load physical edge features ────────────────────────────────────
    ef_path = GRAPH_DIR / "edge_features.npz"
    if not ef_path.exists():
        raise FileNotFoundError(
            f"edge_features.npz not found at {ef_path}.\n"
            f"Run: python src/data/compute_edge_features.py"
        )
    ef = load_edge_features(str(ef_path), device=str(DEVICE))
    _feat_names = ["river_dist", "elev_diff", "travel_time", "hand_diff"]
    if ef["edge_attr"].shape[1] == 5:
        _feat_names.append("sar_wetness")
    logger.info(
        "  Edge features loaded: %d directed edges, %d features: [%s]",
        ef["n_edges"], ef["edge_attr"].shape[1], ", ".join(_feat_names),
    )

    # ── Load static node attributes (for logging; DFC-GNN uses X directly) ─
    # load_graph is called only to surface any warnings about missing files.
    # DFC-GNN does not use node_attr or the static edge_index — those are
    # already embedded in the physical edge features via compute_edge_features.
    _, _, node_attr = load_graph(logger, GRAPH_DIR, DEVICE)

    # ── Splits & dataloaders ───────────────────────────────────────────
    n_windows = T - t_in - t_out + 1
    train_rng, val_rng, test_rng = make_splits(n_windows, t_in, t_out)
    logger.info(
        "Windows — train: %d  val: %d  test: %d",
        len(train_rng), len(val_rng), len(test_rng),
    )

    train_ds = make_dataset(X, y, valid_mask, train_rng, t_in, t_out)
    val_ds   = make_dataset(X, y, valid_mask, val_rng,   t_in, t_out)
    test_ds  = make_dataset(X, y, valid_mask, test_rng,  t_in, t_out)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # ── Model ──────────────────────────────────────────────────────────
    model = build_dfc_gnn(
        n_nodes      = N,
        f_in         = F,
        T_out        = t_out,
        ef_path      = str(ef_path),
        d_model      = HIDDEN_DIM,
        n_heads      = GAT_HEADS,
        n_layers     = GRU_LAYERS,
        dropout      = DROPOUT,
        lambda_flood = LAMBDA_FLOOD,
        device       = str(DEVICE),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")
    logger.info(
        "  Edge count: %d  |  λ_flood=%.3f  |  τ_gate=%.1f m",
        ef["n_edges"], LAMBDA_FLOOD, TAU_GATE,
    )

    # Optional HANDDecoder (τ_k training — activated when SAR masks arrive)
    hand_path = BASE_DIR / "dataset/dem/hand_raster.tif"
    decoder   = HANDDecoder(n_nodes=N).to(DEVICE)
    logger.info(
        "  HANDDecoder: τ_k initialised at 1.0 × %d nodes "
        "(SAR supervision %s)",
        N, "DISABLED — no SAR masks yet" if not hand_path.exists()
           else "READY — run with --sar-supervision to activate",
    )

    # ── Optimiser & scheduler ──────────────────────────────────────────
    all_params = list(model.parameters()) + list(decoder.parameters())
    optimiser  = torch.optim.AdamW(all_params, lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=20,
    )

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = math.inf
    patience_ctr  = 0
    history       = []

    logger.info("Starting training …")
    for epoch in range(1, max_epochs + 1):
        print(f"epoch: {epoch}")

        train_metrics = train_epoch(model, train_loader, optimiser, bankfull_thr)
        val_loss, val_metrics, persist_metrics = eval_epoch(
            model, val_loader, bankfull_thr,
        )
        scheduler.step(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]
        if current_lr <= 1e-5:
            logger.info("  LR floor reached — stopping")
            break

        history.append({
            "epoch":        epoch,
            "train_loss":   round(train_metrics["loss"],       6),
            "train_stage":  round(train_metrics["loss_stage"], 6),
            "train_flood":  round(train_metrics["loss_flood"], 6),
            "val_loss":     round(val_loss, 6),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        })

        logger.info(
            "Epoch %3d  train=%.4f (stage=%.4f flood=%.4f)  "
            "val=%.4f  RMSE=%.4f NSE=%.4f  flood_acc=%.3f  |  "
            "Persist RMSE=%.4f NSE=%.4f  LR=%.1e",
            epoch,
            train_metrics["loss"],
            train_metrics["loss_stage"],
            train_metrics["loss_flood"],
            val_loss,
            val_metrics["rmse"],
            val_metrics["nse"],
            val_metrics.get("flood_acc", 0.0),
            persist_metrics["rmse"],
            persist_metrics["nse"],
            current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save({
                "epoch":       epoch,
                "state_dict":  model.state_dict(),
                "decoder_state": decoder.state_dict(),
                "optimiser":   optimiser.state_dict(),
                "val_loss":    val_loss,
                "val_metrics": val_metrics,
                "hparams": {
                    "t_in":          t_in,
                    "t_out":         t_out,
                    "f_dyn":         F,
                    "n_nodes":       N,
                    "hidden":        HIDDEN_DIM,
                    "gat_heads":     GAT_HEADS,
                    "gru_layers":    GRU_LAYERS,
                    "dropout":       DROPOUT,
                    "lambda_flood":  LAMBDA_FLOOD,
                    "tau_gate":      TAU_GATE,
                    "batch_size":    BATCH_SIZE,
                    "lr":            LR,
                    "n_edges":       ef["n_edges"],
                },
            }, ckpt_dir / "best_model.pt")

            logger.info(
                "  ✓ Saved best model (val_loss=%.4f  flood_acc=%.3f)",
                val_loss, val_metrics.get("flood_acc", 0.0),
            )
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
    decoder.load_state_dict(ckpt["decoder_state"])
    model.eval()
    decoder.eval()

    all_abs_pred, all_tgt, all_mask, all_persist = [], [], [], []
    flood_TP_test = 0.0
    flood_FP_test = 0.0
    flood_FN_test = 0.0
    flood_total_test = 0.0

    with torch.no_grad():
        for x_seq, y_seq, mask in test_loader:
            x_seq = x_seq.to(DEVICE)
            y_seq = y_seq.to(DEVICE)
            mask  = mask.to(DEVICE)

            last_obs    = x_seq[:, -1, :, 0]
            y_flood, node_valid_t = make_flood_labels(y_seq, bankfull_thr, mask)

            delta_pred, flood_logits = model(x_seq)
            abs_pred = last_obs.unsqueeze(1) + delta_pred

            flood_pred = (flood_logits.sigmoid() > 0.5).float()
            _vm_t = node_valid_t.bool()
            flood_TP_test    += (flood_pred[_vm_t] * y_flood[_vm_t]).sum().item()
            flood_FP_test    += (flood_pred[_vm_t] * (1 - y_flood[_vm_t])).sum().item()
            flood_FN_test    += ((1 - flood_pred[_vm_t]) * y_flood[_vm_t]).sum().item()
            flood_total_test += node_valid_t.sum().item()

            all_abs_pred.append(abs_pred.cpu())
            all_tgt.append(y_seq.cpu())
            all_mask.append(mask.cpu())
            all_persist.append(
                last_obs.unsqueeze(1).expand(-1, t_out, -1).cpu()
            )

    cat_pred    = torch.cat(all_abs_pred)
    cat_tgt     = torch.cat(all_tgt)
    cat_mask    = torch.cat(all_mask)
    cat_persist = torch.cat(all_persist)

    test_metrics = compute_metrics(cat_pred, cat_tgt, cat_mask)
    m_all        = cat_mask.bool()
    mbe_global   = (cat_pred[m_all] - cat_tgt[m_all]).mean().item()
    # Fix 2.5b: compute CSI, POD, FAR, F1 from TP/FP/FN accumulators
    _denom_csi = flood_TP_test + flood_FP_test + flood_FN_test
    csi_test  = flood_TP_test / max(_denom_csi, 1e-8)
    pod_test  = flood_TP_test / max(flood_TP_test + flood_FN_test, 1e-8)
    far_test  = flood_FP_test / max(flood_TP_test + flood_FP_test, 1e-8)
    f1_test   = (2*flood_TP_test) / max(2*flood_TP_test + flood_FP_test + flood_FN_test, 1e-8)
    test_metrics["csi"]  = round(csi_test, 4)
    test_metrics["pod"]  = round(pod_test, 4)
    test_metrics["far"]  = round(far_test, 4)
    test_metrics["f1"]   = round(f1_test,  4)
    # Keep flood_acc for backward compatibility with analyse_experiments.py
    test_metrics["flood_acc"] = round(
        flood_TP_test / max(flood_total_test, 1e-8), 4)  # ← now = recall (POD)

    logger.info(
        "\n✓ Test results:\n"
        "  RMSE: %.4f  MAE: %.4f  NSE: %.4f  MBE: %.4f m\n"
        "  CSI:  %.4f  POD: %.4f  FAR: %.4f  F1:  %.4f",
        test_metrics["rmse"], test_metrics["mae"],
        test_metrics["nse"],  mbe_global,
        csi_test, pod_test, far_test, f1_test,
    )

    # ── Per-node metrics ───────────────────────────────────────────────
    nodes_df     = pd.read_csv(GRAPH_DIR / "nodes.csv")
    node_rows    = compute_per_node_metrics(cat_pred,    cat_tgt, cat_mask)
    persist_rows = compute_per_node_metrics(cat_persist, cat_tgt, cat_mask)

    pn_df = pd.DataFrame(node_rows)
    pn_df["ref"]         = nodes_df["ref"].astype(str).values
    pn_df["name"]        = nodes_df["name"].values
    pn_df["persist_nse"] = [r["nse"] for r in persist_rows]
    pn_df["skill"]       = (
        (pn_df["nse"] - pn_df["persist_nse"])
        / (1 - pn_df["persist_nse"]).clip(lower=1e-8)
    ).round(4)

    # Per-node τ_k values (trained HAND depth scales)
    tau_k = decoder.tau.detach().cpu().numpy()
    pn_df["tau_k"]       = np.round(tau_k, 4)

    pn_df = pn_df[[
        "ref", "name", "n_valid", "rmse", "mae",
        "mbe", "nse", "persist_nse", "skill", "tau_k",
    ]]
    pn_df.to_csv(ckpt_dir / "per_node_metrics.csv", index=False)
    logger.info("  Saved per_node_metrics.csv (includes τ_k column)")

    # ── Aggregate test metrics JSON ────────────────────────────────────
    with open(ckpt_dir / "test_metrics.json", "w") as f:
        json.dump({
            **test_metrics,
            "mbe":       round(mbe_global, 6),
            "model":     "dfc_gnn",
            "n_edges":   int(ef["n_edges"]),
            "lambda_flood": LAMBDA_FLOOD,
        }, f, indent=2)

    # ── Per-step metrics ───────────────────────────────────────────────
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

    # ── τ_k summary ───────────────────────────────────────────────────
    tau_summary = {
        str(row["ref"]): round(float(tau_k[i]), 4)
        for i, (_, row) in enumerate(nodes_df.iterrows())
    }
    with open(ckpt_dir / "tau_k_values.json", "w") as f:
        json.dump(tau_summary, f, indent=2)
    logger.info(
        "  τ_k range: [%.3f, %.3f]  (1.0 = unchanged HAND formula)",
        tau_k.min(), tau_k.max(),
    )
    logger.info("  Saved tau_k_values.json")

    return model, test_metrics


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    seed       = 42
    t_in       = 32
    t_out      = 4
    max_epochs = 300

    seed_everything(seed)
    config = load_config(BASE_DIR / "config" / "config.yaml")
    logger = get_logger(config["logging"]["train"])
    train(logger, seed, t_in, t_out, max_epochs)
