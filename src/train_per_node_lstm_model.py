"""
train_per_node_lstm_model.py  –  Per-node LSTM baseline for Lee catchment flood forecasting
============================================================================================
Architecture:
  1. Node embedding   Linear([F_dyn + F_static]) → hidden_dim
  2. Temporal         LSTM (LSTM_LAYERS layers) over the T_in-step sequence
  3. Output           Linear head → T_out step forecast per node

  NOTE: This is a non-spatial baseline. No graph message passing is used.
  Each gauge node is modelled independently. It serves as Baseline B in the
  ablation study — temporal modelling only, no inter-node information.
  Compared against PerNodeGRU to isolate the effect of gating architecture.

Input window:   T_in  = 32 steps  (8 hours at 15-min resolution)
Output horizon: T_out =  4 steps  (1 hour, multi-step)

Target:         DELTA stage_anomaly — the model predicts the CHANGE from
                the last observed value rather than the absolute level.
                  delta[h] = stage_anomaly[t+h] - stage_anomaly[t]
                Absolute predictions are recovered at evaluation time:
                  abs_pred[h] = last_obs + delta_pred[h]
                An output of 0.0 is equivalent to persistence, so the model
                is explicitly penalised for lazy forecasting.

Loss:           Horizon-weighted masked MSE — later steps in the forecast
                window receive proportionally more weight.
                  h+1: 0.4x  h+2: 0.8x  h+3: 1.2x  h+4: 1.6x  (T_out=4)

Train / val / test split: chronological 70 / 15 / 15 with T_in+T_out purge gaps.
Split fractions are governed by TRAIN_FRAC / VAL_FRAC in train_utils.py.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.common_utils import seed_everything
from src.utils.train_utils import make_splits
from src.utils.train_utils import make_dataset
from src.utils.train_utils import load_graph
from src.utils.train_utils import compute_metrics
from src.utils.train_utils import compute_per_node_metrics
from src.utils.train_utils import compute_per_step_metrics
from src.utils.train_utils import masked_mse_horizon_weighted
from src.models.baseline_lstm import PerNodeLSTM

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"

# ── Hyperparameters ────────────────────────────────────────────────────
HIDDEN_DIM  = 64    # node embedding + LSTM hidden size
LSTM_LAYERS = 2     # LSTM depth
DROPOUT     = 0.1

BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 30   # early stopping patience (epochs)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimiser, node_attr) -> float:
    """
    One training epoch using delta (residual) prediction.

    Predicts delta_stage_anomaly[h] = stage_anomaly[t+h] - stage_anomaly[t].
    Horizon-weighted loss penalises later steps more heavily, preventing
    collapse to a near-zero-delta (persistence) solution.
    """
    model.train()
    total_loss = 0.0

    for x_seq, y_seq, mask in loader:
        x_seq = x_seq.to(DEVICE)
        y_seq = y_seq.to(DEVICE)
        mask  = mask.to(DEVICE)

        last_obs     = x_seq[:, -1, :, 0]               # [B, N]
        delta_target = y_seq - last_obs.unsqueeze(1)     # [B, T_out, N]

        optimiser.zero_grad()
        delta_pred = model(x_seq, node_attr)             # [B, T_out, N]
        loss = masked_mse_horizon_weighted(delta_pred, delta_target, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, node_attr):
    """
    Evaluation epoch.

    Delta predictions are reconstructed to absolute stage_anomaly space
    before computing metrics, so RMSE/MAE/NSE are directly comparable
    across all model types.
    """
    model.eval()
    total_loss = 0.0
    all_abs_pred, all_tgt, all_mask, all_persist = [], [], [], []

    for x_seq, y_seq, mask in loader:
        x_seq = x_seq.to(DEVICE)
        y_seq = y_seq.to(DEVICE)
        mask  = mask.to(DEVICE)

        last_obs     = x_seq[:, -1, :, 0]
        delta_target = y_seq - last_obs.unsqueeze(1)
        delta_pred   = model(x_seq, node_attr)
        abs_pred     = last_obs.unsqueeze(1) + delta_pred

        total_loss += masked_mse_horizon_weighted(
            delta_pred, delta_target, mask
        ).item()

        all_abs_pred.append(abs_pred.cpu())
        all_tgt.append(y_seq.cpu())
        all_mask.append(mask.cpu())
        all_persist.append(
            last_obs.unsqueeze(1).expand(-1, y_seq.shape[1], -1).cpu()
        )

    cat_abs_pred = torch.cat(all_abs_pred)
    cat_tgt      = torch.cat(all_tgt)
    cat_mask     = torch.cat(all_mask)

    metrics         = compute_metrics(cat_abs_pred, cat_tgt, cat_mask)
    persist_metrics = compute_metrics(torch.cat(all_persist), cat_tgt, cat_mask)
    return total_loss / len(loader), metrics, persist_metrics


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def train(logger, seed, t_in, t_out, max_epochs, base_dir = None):
    if base_dir is None:
        base_dir = BASE_DIR
    ckpt_dir = base_dir / "checkpoints" / "lstm" / str(seed) / str(t_out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Training Baseline LSTM ===")
    logger.info("Device: %s", DEVICE)

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    X          = np.load(PROC_DIR / "X.npy")
    y          = np.load(PROC_DIR / "y.npy")
    valid_mask = np.load(PROC_DIR / "valid_mask.npy")

    T, N, F = X.shape
    logger.info("  X: %s  y: %s  valid_mask: %s", X.shape, y.shape, valid_mask.shape)

    # ── Load graph (node_attr only — LSTM has no message passing) ─────
    _, _, node_attr = load_graph(logger, GRAPH_DIR, DEVICE)

    # ── Splits ─────────────────────────────────────────────────────────
    n_windows = T - t_in - t_out + 1
    train_rng, val_rng, test_rng = make_splits(n_windows, t_in, t_out)
    logger.info(
        "Windows — train: %d  val: %d  test: %d",
        len(train_rng), len(val_rng), len(test_rng),
    )

    train_ds = make_dataset(X, y, valid_mask, train_rng, t_in, t_out)
    val_ds   = make_dataset(X, y, valid_mask, val_rng,   t_in, t_out)
    test_ds  = make_dataset(X, y, valid_mask, test_rng,  t_in, t_out)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────
    f_static = node_attr.shape[1]
    model = PerNodeLSTM(
        f_dyn=F, f_static=f_static,
        hidden=HIDDEN_DIM,
        lstm_layers=LSTM_LAYERS, t_out=t_out, dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=20
    )

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = math.inf
    patience_ctr  = 0
    history       = []

    logger.info("Starting training …")
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, node_attr)
        val_loss, val_metrics, persist_metrics = eval_epoch(
            model, val_loader, node_attr
        )
        scheduler.step(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]
        logger.info("  LR: %.2e", current_lr)
        if current_lr <= 1e-5:
            logger.info("  LR floor reached — stopping")
            break

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        })

        logger.info(
            "Epoch %3d  train=%.4f  val=%.4f  "
            "Model  RMSE=%.4f NSE=%.4f  |  "
            "Persist RMSE=%.4f NSE=%.4f",
            epoch, train_loss, val_loss,
            val_metrics["rmse"],     val_metrics["nse"],
            persist_metrics["rmse"], persist_metrics["nse"],
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
                    "t_in": t_in, "t_out": t_out,
                    "hidden": HIDDEN_DIM, "lstm_layers": LSTM_LAYERS,
                    "dropout": DROPOUT, "batch_size": BATCH_SIZE, "lr": LR,
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
            delta_pred = model(x_seq, node_attr)
            abs_pred   = last_obs.unsqueeze(1) + delta_pred
            all_abs_pred.append(abs_pred.cpu())
            all_tgt.append(y_seq)
            all_mask.append(mask)
            all_persist.append(
                last_obs.unsqueeze(1).expand(-1, t_out, -1).cpu()
            )

    cat_pred    = torch.cat(all_abs_pred)
    cat_tgt     = torch.cat(all_tgt)
    cat_mask    = torch.cat(all_mask)
    cat_persist = torch.cat(all_persist)

    # Test loss (horizon-weighted delta space — consistent with training)
    with torch.no_grad():
        delta_losses = []
        for x_seq, y_seq, mask in test_loader:
            x_seq        = x_seq.to(DEVICE)
            last_obs     = x_seq[:, -1, :, 0]
            delta_target = y_seq.to(DEVICE) - last_obs.unsqueeze(1)
            delta_pred   = model(x_seq, node_attr)
            delta_losses.append(
                masked_mse_horizon_weighted(
                    delta_pred, delta_target, mask.to(DEVICE)
                ).item()
            )
    test_loss    = float(np.mean(delta_losses))
    test_metrics = compute_metrics(cat_pred, cat_tgt, cat_mask)

    m_all      = cat_mask.bool()
    mbe_global = (cat_pred[m_all] - cat_tgt[m_all]).mean().item()

    logger.info(
        "\n✓ Test results:\n"
        "  Loss: %.4f\n  RMSE: %.4f\n  MAE:  %.4f\n  NSE:  %.4f\n  MBE:  %.4f m",
        test_loss,
        test_metrics["rmse"], test_metrics["mae"],
        test_metrics["nse"],  mbe_global,
    )

    # ── Per-node metrics ───────────────────────────────────────────────
    nodes_df  = pd.read_csv(GRAPH_DIR / "nodes.csv")
    node_rows = compute_per_node_metrics(cat_pred, cat_tgt, cat_mask)
    pn_df = pd.DataFrame(node_rows)
    pn_df["ref"]  = nodes_df["ref"].astype(str).values
    pn_df["name"] = nodes_df["name"].values

    persist_rows = compute_per_node_metrics(cat_persist, cat_tgt, cat_mask)
    pn_df["persist_nse"] = [r["nse"] for r in persist_rows]
    pn_df["skill"] = (
        (pn_df["nse"] - pn_df["persist_nse"])
        / (1 - pn_df["persist_nse"]).clip(lower=1e-8)
    ).round(4)

    pn_df = pn_df[["ref", "name", "n_valid", "rmse", "mae", "mbe",
                   "nse", "persist_nse", "skill"]]
    pn_df.to_csv(ckpt_dir / "per_node_metrics.csv", index=False)
    logger.info("  Saved per_node_metrics.csv")

    sorted_pn = pn_df.dropna(subset=["nse"]).sort_values("nse")
    logger.info("\n  Lowest NSE nodes:\n%s",
                sorted_pn.head(5)[["name", "nse", "skill"]].to_string(index=False))
    logger.info("\n  Highest NSE nodes:\n%s",
                sorted_pn.tail(5)[["name", "nse", "skill"]].to_string(index=False))

    # ── Aggregate test metrics ─────────────────────────────────────────
    with open(ckpt_dir / "test_metrics.json", "w") as f:
        json.dump({
            "test_loss": test_loss,
            **test_metrics,
            "mbe": round(mbe_global, 6),
        }, f, indent=2)

    # ── Per-step metrics ───────────────────────────────────────────────
    per_step         = compute_per_step_metrics(cat_pred,    cat_tgt, cat_mask)
    per_step_persist = compute_per_step_metrics(cat_persist, cat_tgt, cat_mask)

    for h_dict, p_dict in zip(per_step, per_step_persist):
        persist_nse = p_dict["nse"]
        model_nse   = h_dict["nse"]
        if not (np.isnan(persist_nse) or np.isnan(model_nse)) and persist_nse < 1.0:
            h_dict["persist_nse"] = round(persist_nse, 6)
            h_dict["skill"]       = round((model_nse - persist_nse) / (1 - persist_nse), 4)
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
    max_epochs = 100
    seed_everything(seed)
    config = load_config(BASE_DIR / "config" / "config.yaml")
    logger = get_logger(config["logging"]["train"])
    train(logger, seed, t_in, t_out, max_epochs)
