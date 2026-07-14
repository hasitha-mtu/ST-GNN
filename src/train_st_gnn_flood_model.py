"""
train_st_gnn_flood_model.py  –  PI-ST-GNN static graph baseline, River Lee catchment
========================================================================================
Architecture
------------
  1. Node feature projection   Linear(F_dyn + F_static) → hidden_dim
  2. Temporal encoder          Per-node GRU → hidden state [B, N, hidden]
  3. Graph message passing     GATConv ×3 layers (batched)
  4. Output head               Linear → delta stage_anomaly [B, T_out, N]

This is the clean static-graph, no-SAR baseline for the PI-ST-GNN comparison.
It isolates the contribution of graph message passing over PerNodeGRU/LSTM.

Comparison matrix:
  PerNodeGRU           — no graph, no SAR  (temporal lower bound)
  PerNodeLSTM          — no graph, no SAR  (temporal lower bound)
  STGNNFlood (this)    — static graph, no SAR
  STGNNFlood+SAR       — static graph + SAR-FNO  (train_st_gnn_sar.py)
  STGNNFloodDynEdge    — dynamic edge weights     (Phase 1)
  STGNNFloodHAND       — dynamic topology         (Phase 2)

Input window:   T_in  = 32 steps  (8 hours at 15-min resolution)
Output horizon: T_out =  4 steps  (1 hour, multi-step)
"""


import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
from src.utils.compile_utils import compile_model
from src.utils.gpu_sampler import make_gpu_loaders

from src.models.st_gnn_flood   import STGNNFloodModel

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
LIVE_METRICS_PATH = BASE_DIR / "checkpoints" / "live_metrics.json"

# ── Hyperparameters ────────────────────────────────────────────────────
HIDDEN_DIM = 64    # node embedding + GRU hidden size
GAT_HEADS  = 2     # attention heads in GATConv
GRU_LAYERS = 2     # GRU depth
DROPOUT    = 0.1

BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 30   # early stopping patience (epochs)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
#  SAR data helpers
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(
    model, loader, optimiser,
    edge_index, edge_attr, node_attr,
) -> float:
    """One training epoch over the Lee gauge dataset."""
    model.train()
    total_loss = 0.0

    for batch_idx, (x_seq, y_seq, mask) in enumerate(loader):
        # x_seq = x_seq.to(DEVICE)   # [B, T_in, N, F]
        # y_seq = y_seq.to(DEVICE)   # [B, T_out, N]
        # mask  = mask.to(DEVICE)    # [B, T_out, N]

        last_obs     = x_seq[:, -1, :, 0]              # [B, N]
        delta_target = y_seq - last_obs.unsqueeze(1)   # [B, T_out, N]

        optimiser.zero_grad(set_to_none=True)
        delta_pred = model(x_seq, node_attr, edge_index, edge_attr)
        loss = masked_mse_horizon_weighted(delta_pred, delta_target, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model, loader,
    edge_index, edge_attr, node_attr,
):
    model.eval()
    total_loss = 0.0
    all_abs_pred, all_tgt, all_mask, all_persist = [], [], [], []

    for batch_idx, (x_seq, y_seq, mask) in enumerate(loader):
        # x_seq = x_seq.to(DEVICE)
        # y_seq = y_seq.to(DEVICE)
        # mask  = mask.to(DEVICE)

        last_obs     = x_seq[:, -1, :, 0]
        delta_target = y_seq - last_obs.unsqueeze(1)

        delta_pred = model(x_seq, node_attr, edge_index, edge_attr)
        abs_pred   = last_obs.unsqueeze(1) + delta_pred

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
    run_tag  = "st_gnn_static"
    ckpt_dir = base_dir / "checkpoints" / run_tag / str(seed) / str(t_out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training ST-GNN (static graph, no SAR) ===")
    logger.info("Device: %s", DEVICE)

    # ── Load gauge data ────────────────────────────────────────────────
    logger.info("Loading dataset …")
    X          = np.load(PROC_DIR / "X.npy")
    y          = np.load(PROC_DIR / "y.npy")
    valid_mask = np.load(PROC_DIR / "valid_mask.npy")
    T, N, F    = X.shape
    logger.info("  X: %s  y: %s  valid_mask: %s", X.shape, y.shape, valid_mask.shape)

    # ── Load graph ─────────────────────────────────────────────────────
    edge_index, edge_attr, node_attr = load_graph(logger, GRAPH_DIR, DEVICE)

    # ── Splits & dataloaders ───────────────────────────────────────────
    n_windows = T - t_in - t_out + 1
    train_rng, val_rng, test_rng = make_splits(n_windows, t_in, t_out)
    logger.info(
        "Windows — train: %d  val: %d  test: %d",
        len(train_rng), len(val_rng), len(test_rng),
    )
    #
    # train_ds = make_dataset(X, y, valid_mask, train_rng, t_in, t_out)
    # val_ds   = make_dataset(X, y, valid_mask, val_rng,   t_in, t_out)
    # test_ds  = make_dataset(X, y, valid_mask, test_rng,  t_in, t_out)

    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
    #                           shuffle=True,  num_workers=4,
    #                           pin_memory=True, persistent_workers=True)
    # val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2,
    #                           shuffle=False, num_workers=4,
    #                           pin_memory=True, persistent_workers=True)
    # test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
    #                           shuffle=False, num_workers=0)

    train_loader, val_loader, test_loader = make_gpu_loaders(
        X, y, valid_mask,
        t_in=t_in, t_out=t_out,
        batch_size=BATCH_SIZE, device=DEVICE,
    )

    # ── Model ──────────────────────────────────────────────────────────
    f_static = node_attr.shape[1]
    f_edge   = edge_attr.shape[1]
    # When SAR is active the fusion layer input is F_dyn + f_static + SAR_EMB_DIM;
    # the model handles this via the sar_emb_dim constructor argument.
    model = STGNNFloodModel(
        f_dyn=F,
        f_static=f_static,
        f_edge=f_edge,
        hidden=HIDDEN_DIM,
        gat_heads=GAT_HEADS,
        gru_layers=GRU_LAYERS,
        t_out=t_out,
        dropout=DROPOUT,
        sar_emb_dim=0,        # no SAR — static-graph baseline
    ).to(DEVICE)

    model = compile_model(model, tag=run_tag, logger=logger)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=20
    )

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = math.inf
    patience_ctr  = 0
    history       = []

    logger.info("Starting training …")
    for epoch in range(1, max_epochs + 1):
        print(f'epoch: {epoch}')

        train_loss = train_epoch(
            model, train_loader, optimiser,
            edge_index, edge_attr, node_attr,
        )
        val_loss, val_metrics, persist_metrics = eval_epoch(
            model, val_loader,
            edge_index, edge_attr, node_attr,
        )
        scheduler.step(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]
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
            "Model RMSE=%.4f NSE=%.4f  |  "
            "Persist RMSE=%.4f NSE=%.4f  LR=%.1e",
            epoch, train_loss, val_loss,
            val_metrics["rmse"],     val_metrics["nse"],
            persist_metrics["rmse"], persist_metrics["nse"],
            current_lr,
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
                    "sar_emb_dim": 0,
                    
                },
            }, ckpt_dir / "best_model.pt")
            # Log which feature set is active — helps confirm SM integration worked
            sm_active = F > 5
            logger.info(
                "  Features: F=%d (%s)",
                F,
                "5 gauge + 6 soil moisture" if sm_active else "5 gauge only — SM not integrated",
            )
            if not sm_active:
                logger.warning(
                    "  Running with F=5 (gauge only). "
                    "Set USE_SOIL_MOISTURE=True in build_dataset.py and rebuild X.npy "
                    "to include ERA5-Land soil moisture features."
                )
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
        for batch_idx, (x_seq, y_seq, mask) in enumerate(test_loader):
            x_seq    = x_seq.to(DEVICE)
            last_obs = x_seq[:, -1, :, 0]

            delta_pred = model(x_seq, node_attr, edge_index, edge_attr)
            abs_pred   = last_obs.unsqueeze(1) + delta_pred

            # all_abs_pred.append(abs_pred.cpu())
            # all_tgt.append(y_seq)
            # all_mask.append(mask)

            all_abs_pred.append(abs_pred.cpu())
            all_tgt.append(y_seq.cpu())
            all_mask.append(mask.cpu())

            all_persist.append(
                last_obs.unsqueeze(1).expand(-1, t_out, -1).cpu()
            )

    # cat_pred    = torch.cat(all_abs_pred)
    # cat_tgt     = torch.cat(all_tgt)
    # cat_mask    = torch.cat(all_mask)
    # cat_persist = torch.cat(all_persist)

    cat_pred = torch.cat(all_abs_pred).cpu()
    cat_tgt = torch.cat(all_tgt).cpu()
    cat_mask = torch.cat(all_mask).cpu()
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
    nodes_df     = pd.read_csv(GRAPH_DIR / "nodes.csv")
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

    pn_df = pn_df[["ref", "name", "n_valid", "rmse", "mae",
                   "mbe", "nse", "persist_nse", "skill"]]
    pn_df.to_csv(ckpt_dir / "per_node_metrics.csv", index=False)
    logger.info("  Saved per_node_metrics.csv")

    # ── Aggregate + per-step metrics ───────────────────────────────────
    with open(ckpt_dir / "test_metrics.json", "w") as f:
        json.dump({
            **test_metrics,
            "mbe":     round(mbe_global, 6),
            "use_sar": False,
            "model":   "stgnn_static",
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

def update_node_coordinates(nodes):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    nodes["easting_itm"], nodes["northing_itm"] = t.transform(
        nodes["lon"].values,
        nodes["lat"].values,
    )
    nodes.to_csv(GRAPH_DIR / "nodes.csv", index=False)
    return nodes


if __name__ == "__main__":
    seed       = 42
    t_in       = 32
    t_out      = 4
    max_epochs = 2
    seed_everything(seed)
    config = load_config(BASE_DIR / "config" / "config.yaml")
    logger = get_logger(config["logging"]["train"])
    train(logger, seed, t_in, t_out, max_epochs)
