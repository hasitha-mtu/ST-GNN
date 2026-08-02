"""
train_st_gnn_soil_gate.py  —  Training module for STGNNSoilGate
================================================================
Trains the antecedent soil moisture-conditioned ST-GNN (STGNNSoilGate)
for a single (seed, T_out) combination.

Called by train_models_exp1.py as:
    import train_st_gnn_soil_gate as _mod_soil
    _mod_soil.USE_SAR = False
    _mod_soil.train(logger, seed, t_in, t_out, max_epochs, base_dir)

Architecture differences from STGNNHANDEdge:
    Gate:  σ(γ × (S̄(t) − θ_sat))  ← soil moisture (this module)
    vs     σ(α × (max(H_src,H_dst) − z_saddle))  ← stage (HANDEdge)

Everything else is identical: same river-network GATConv layers,
same HAND edge structure, same loss, same optimiser, same GPU sampler.

ERA5-Land requirement
---------------------
swvl2_sat_ratio (X.npy feature index 9) must have real values.
If X.npy was built with a failed ERA5 download (all zeros at index 9),
the soil gate will operate near its baseline (sat ≈ 0 < threshold 0.75)
and behave equivalently to STGNNHANDEdge with permanently closed gates.
Run build_dataset.py after a successful ERA5-Land download before training.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
for _p in [BASE_DIR, BASE_DIR / "src"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.st_gnn_soil_gate import STGNNSoilGate
from models.st_gnn_hand_edge import load_hand_edges
from src.utils.train_utils   import (
    load_graph, compute_metrics, compute_per_node_metrics,
    compute_per_step_metrics,
)
from src.utils.gpu_sampler import make_gpu_loaders

# ── Hyperparameters ───────────────────────────────────────────────────
HIDDEN_DIM  = 64
GAT_HEADS   = 2
GRU_LAYERS  = 2
DROPOUT     = 0.10
BATCH_SIZE  = 512
LR          = 5e-4
WEIGHT_DECAY = 1e-5
PATIENCE    = 20
GRAD_CLIP   = 1.0

# Soil gate initialisation (matched to physical saturation range of the
# Lee catchment — Irish clay-loam field capacity ≈ 0.75)
SAT_THRESHOLD_INIT  = 0.75
SAT_SHARPNESS_INIT  = 10.0
SWVL2_SAT_IDX       = 9     # feature index in X.npy

# Horizon weighting: later forecast steps receive proportionally higher loss
# weight, matching the pattern used in all other training scripts.
def horizon_weights(T_out: int, device: torch.device) -> torch.Tensor:
    w = torch.arange(1, T_out + 1, dtype=torch.float32, device=device)
    return w / w.mean()

# ── SAR flag (patched to False by train_models_exp1.py for Experiment 1) ──
USE_SAR = False

# ── Paths ─────────────────────────────────────────────────────────────
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════
#  Training and evaluation epochs
# ══════════════════════════════════════════════════════════════════════

def train_epoch(
    model:     STGNNSoilGate,
    loader,
    optimiser: torch.optim.Optimizer,
    w_hz:      torch.Tensor,         # [T_out] horizon weights
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x_seq, y_seq, mask in loader:
        # All tensors already on GPU via GPUSampler
        optimiser.zero_grad(set_to_none=True)

        delta = model(
            x_seq, node_attr, edge_index, edge_attr
        )  # [B, T_out, N]

        last_obs = x_seq[:, -1, :, 0]              # [B, N]
        abs_pred = last_obs.unsqueeze(1) + delta    # [B, T_out, N]

        # Horizon-weighted masked MSE
        err2 = (abs_pred - y_seq) ** 2             # [B, T_out, N]
        wt   = w_hz.unsqueeze(0).unsqueeze(-1)     # [1, T_out, 1]
        loss = (err2 * mask * wt).sum() / (mask * wt).sum().clamp(min=1.0)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()

        total_loss += loss.item()
        n_batches  += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def eval_epoch(
    model:   STGNNSoilGate,
    loader,
    w_hz:    torch.Tensor,
) -> tuple[float, dict, dict]:
    model.eval()
    all_pred, all_tgt, all_mask, all_persist = [], [], [], []

    for x_seq, y_seq, mask in loader:
        delta    = model(x_seq, node_attr, edge_index, edge_attr)
        last_obs = x_seq[:, -1, :, 0]
        abs_pred = last_obs.unsqueeze(1) + delta
        persist  = last_obs.unsqueeze(1).expand_as(y_seq)

        all_pred.append(abs_pred.cpu())
        all_tgt.append(y_seq.cpu())
        all_mask.append(mask.cpu())
        all_persist.append(persist.cpu())

    cat_pred    = torch.cat(all_pred,    dim=0)
    cat_tgt     = torch.cat(all_tgt,     dim=0)
    cat_mask    = torch.cat(all_mask,    dim=0)
    cat_persist = torch.cat(all_persist, dim=0)

    # Weighted MSE for scheduler
    err2    = (cat_pred - cat_tgt) ** 2
    wt      = w_hz.cpu().unsqueeze(0).unsqueeze(-1)
    val_loss = ((err2 * cat_mask * wt).sum()
                / (cat_mask * wt).sum().clamp(min=1.0)).item()

    metrics         = compute_metrics(cat_pred, cat_tgt, cat_mask)
    persist_metrics = compute_metrics(cat_persist, cat_tgt, cat_mask)
    return val_loss, metrics, persist_metrics


# ══════════════════════════════════════════════════════════════════════
#  Main training function (called by train_models_exp1.py)
# ══════════════════════════════════════════════════════════════════════

# Module-level graph tensors — populated once per process in train()
node_attr   = None
edge_index  = None
edge_attr   = None


def train(
    logger,
    seed:      int,
    t_in:      int,
    t_out:     int,
    max_epochs: int,
    base_dir:  Path | None = None):
    """
    Train STGNNSoilGate for one (seed, t_out) combination.
    Checkpoint saved to checkpoints/st_gnn_soil_gate/{seed}/{t_out}/.
    """
    global node_attr, edge_index, edge_attr

    run_tag  = "st_gnn_soil_gate"
    if base_dir is None:
        base_dir = BASE_DIR

    ckpt_dir = base_dir / "checkpoints" / run_tag / str(seed) / str(t_out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("Training STGNNSoilGate (antecedent soil moisture gate)")
    logger.info("  seed=%d  T_out=%d  device=%s", seed, t_out, DEVICE)
    logger.info("=" * 60)

    # ── Load dataset ───────────────────────────────────────────────────
    logger.info("Loading dataset …")
    X          = np.load(PROC_DIR / "X.npy")
    y          = np.load(PROC_DIR / "y.npy")
    valid_mask = np.load(PROC_DIR / "valid_mask.npy")
    T, N, F    = X.shape
    logger.info("  X: %s  y: %s", X.shape, y.shape)

    # Warn if swvl2_sat_ratio looks like all-zeros (failed ERA5 download)
    swvl2_mean = float(X[:, :, SWVL2_SAT_IDX].mean())
    swvl2_std  = float(X[:, :, SWVL2_SAT_IDX].std())
    if swvl2_std < 0.01 or swvl2_mean < 0.01:
        logger.warning(
            "  WARNING: X[:,:,%d] (swvl2_sat_ratio) mean=%.4f std=%.4f — "
            "appears to be all zeros. ERA5-Land download may have failed. "
            "STGNNSoilGate will train but the soil gate will be "
            "permanently near-closed (sat ≈ 0 < threshold 0.75). "
            "Run build_dataset.py with valid ERA5-Land data first.",
            SWVL2_SAT_IDX, swvl2_mean, swvl2_std,
        )
    else:
        logger.info(
            "  swvl2_sat_ratio (F[%d]): mean=%.3f  std=%.3f  "
            "range=[%.3f, %.3f]",
            SWVL2_SAT_IDX, swvl2_mean, swvl2_std,
            float(X[:, :, SWVL2_SAT_IDX].min()),
            float(X[:, :, SWVL2_SAT_IDX].max()),
        )

    # ── Load graph ─────────────────────────────────────────────────────
    edge_index, edge_attr, node_attr = load_graph(logger, GRAPH_DIR, DEVICE)
    F_static = node_attr.shape[1]
    F_edge   = edge_attr.shape[1] + 1   # +1 for dynamic conductance

    # ── Load HAND edges ────────────────────────────────────────────────
    hand_path = GRAPH_DIR / "hand_edges.npz"
    if not hand_path.exists():
        raise FileNotFoundError(
            f"hand_edges.npz not found at {hand_path}.\n"
            "Run: python src/data/precompute_hand_edges.py"
        )
    hand = load_hand_edges(str(hand_path))
    logger.info(
        "  HAND edges: %d candidate pairs  "
        "z_saddle range=[%.1f, %.1f] m OD",
        hand["src"].shape[0],
        hand["z_saddle_m"].min().item(),
        hand["z_saddle_m"].max().item(),
    )

    # ── Per-node gauge_datum and stage_range (for inherited gate buffers) ──
    nodes_df    = pd.read_csv(GRAPH_DIR / "nodes.csv")
    gauge_datum = torch.tensor(
        nodes_df["gauge_datum_mOSGM15"].values, dtype=torch.float32)
    stage_range = torch.tensor(
        (nodes_df["p90_mAOD"] - nodes_df["gauge_datum_mOSGM15"]).clip(lower=0.5).values,
        dtype=torch.float32,
    )

    # Reservoir nodes have placeholder datum 0.0 → use p90 as fallback
    reservoir_mask = nodes_df["is_reservoir"].values.astype(bool)
    if reservoir_mask.any():
        gauge_datum[reservoir_mask] = torch.tensor(
            nodes_df.loc[reservoir_mask, "p90_mAOD"].values,
            dtype=torch.float32,
        )
        logger.info(
            "  Reservoir nodes (%d): gauge_datum set to p90_mAOD as fallback",
            int(reservoir_mask.sum()),
        )

    # ── GPU data loaders ───────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_gpu_loaders(
        X, y, valid_mask,
        t_in=t_in, t_out=t_out,
        batch_size=BATCH_SIZE, device=DEVICE,
    )

    # ── Build model ────────────────────────────────────────────────────
    model = STGNNSoilGate(
        f_dyn               = F,
        f_static            = F_static,
        f_edge              = F_edge,
        hand_src            = hand["src"],
        hand_dst            = hand["dst"],
        hand_threshold      = hand["hand_threshold"],
        hand_overland_dist  = hand["overland_dist_km"],
        z_saddle            = hand["z_saddle_m"],
        gauge_datum         = gauge_datum,
        stage_range         = stage_range,
        hidden              = HIDDEN_DIM,
        gat_heads           = GAT_HEADS,
        gru_layers          = GRU_LAYERS,
        t_out               = t_out,
        dropout             = DROPOUT,
        sar_emb_dim         = 0,       # SAR disabled for Experiment 1
        discharge_idx       = 3,
        swvl2_sat_idx       = SWVL2_SAT_IDX,
        sat_threshold       = SAT_THRESHOLD_INIT,
        sat_sharpness       = SAT_SHARPNESS_INIT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Parameters: %s", f"{n_params:,}")
    logger.info(
        "  sat_threshold=%.2f (learnable)  sat_sharpness=%.1f (learnable)  "
        "swvl2_idx=%d",
        SAT_THRESHOLD_INIT, SAT_SHARPNESS_INIT, SWVL2_SAT_IDX,
    )

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=15,
        cooldown=2, min_lr=1e-6,
    )
    w_hz = horizon_weights(t_out, DEVICE)

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = math.inf
    patience_ctr  = 0
    history       = []

    logger.info("Starting training …")
    for epoch in range(1, max_epochs + 1):

        train_m = train_epoch(model, train_loader, optimiser, w_hz)
        val_loss, val_m, persist_m = eval_epoch(model, val_loader, w_hz)
        scheduler.step(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]

        # Log learned gate parameters every 25 epochs
        sat_thr = model.sat_threshold.item()
        sat_shp = model.sat_sharpness.item()

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_m["loss"], 6),
            "val_loss":   round(val_loss, 6),
            "val_nse":    round(val_m.get("nse", float("nan")), 4),
            "val_rmse":   round(val_m.get("rmse", float("nan")), 4),
            "sat_threshold_learned": round(sat_thr, 4),
            "sat_sharpness_learned": round(sat_shp, 4),
            "lr":         current_lr,
        })

        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d  train=%.6e  val=%.6e  NSE=%.4f  RMSE=%.4f  "
                "ES=%d/%d  sat_thr=%.3f  sat_shp=%.2f  LR=%.1e",
                epoch, train_m["loss"], val_loss,
                val_m.get("nse", float("nan")),
                val_m.get("rmse", float("nan")),
                patience_ctr, PATIENCE,
                sat_thr, sat_shp, current_lr,
            )
        else:
            logger.debug(
                "Epoch %3d  train=%.6e  val=%.6e  NSE=%.4f  LR=%.1e",
                epoch, train_m["loss"], val_loss,
                val_m.get("nse", float("nan")), current_lr,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save({
                "epoch":       epoch,
                "state_dict":  model.state_dict(),
                "optimiser":   optimiser.state_dict(),
                "val_loss":    val_loss,
                "val_metrics": val_m,
                "hparams": {
                    "model":            "st_gnn_soil_gate",
                    "seed":             seed,
                    "t_in":             t_in,
                    "t_out":            t_out,
                    "f_dyn":            F,
                    "f_static":         F_static,
                    "n_nodes":          N,
                    "hidden":           HIDDEN_DIM,
                    "gat_heads":        GAT_HEADS,
                    "gru_layers":       GRU_LAYERS,
                    "dropout":          DROPOUT,
                    "batch_size":       BATCH_SIZE,
                    "lr":               LR,
                    "swvl2_sat_idx":    SWVL2_SAT_IDX,
                    "sat_threshold_init": SAT_THRESHOLD_INIT,
                    "sat_sharpness_init": SAT_SHARPNESS_INIT,
                    "sat_threshold_learned": round(sat_thr, 4),
                    "sat_sharpness_learned": round(sat_shp, 4),
                    "n_hand_edges":     int(hand["src"].shape[0]),
                },
            }, ckpt_dir / "best_model.pt")
            logger.info("  ✓ Saved best (val_loss=%.6f  NSE=%.4f)",
                        val_loss, val_m.get("nse", float("nan")))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if current_lr <= 1e-6 + 1e-9:
            logger.info("LR floor reached — stopping")
            break

    pd.DataFrame(history).to_csv(ckpt_dir / "training_history.csv",
                                  index=False)

    # ── Test evaluation ────────────────────────────────────────────────
    logger.info("Test evaluation …")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_pred, all_tgt, all_mask, all_persist = [], [], [], []
    with torch.no_grad():
        for x_seq, y_seq, mask in test_loader:
            x_seq = x_seq.to(DEVICE)
            y_seq = y_seq.to(DEVICE)
            mask  = mask.to(DEVICE)

            delta    = model(x_seq, node_attr, edge_index, edge_attr)
            last_obs = x_seq[:, -1, :, 0]
            abs_pred = last_obs.unsqueeze(1) + delta
            persist  = last_obs.unsqueeze(1).expand_as(y_seq)

            all_pred.append(abs_pred.cpu())
            all_tgt.append(y_seq.cpu())
            all_mask.append(mask.cpu())
            all_persist.append(persist.cpu())

    cat_pred    = torch.cat(all_pred,    dim=0)
    cat_tgt     = torch.cat(all_tgt,     dim=0)
    cat_mask    = torch.cat(all_mask,    dim=0)
    cat_persist = torch.cat(all_persist, dim=0)

    test_metrics    = compute_metrics(cat_pred, cat_tgt, cat_mask)
    persist_metrics = compute_metrics(cat_persist, cat_tgt, cat_mask)

    mbe_global = float(
        (cat_pred - cat_tgt)[cat_mask.bool()].mean()
        if cat_mask.bool().any() else float("nan")
    )

    logger.info(
        "Test results — NSE=%.4f  RMSE=%.4f  MAE=%.4f  MBE=%.4f  "
        "(Persist NSE=%.4f  RMSE=%.4f)",
        test_metrics["nse"], test_metrics["rmse"],
        test_metrics["mae"], mbe_global,
        persist_metrics["nse"], persist_metrics["rmse"],
    )

    # Learned gate parameters at convergence
    sat_thr_final = model.sat_threshold.item()
    sat_shp_final = model.sat_sharpness.item()
    logger.info(
        "  Learned gate — sat_threshold=%.4f  sat_sharpness=%.4f  "
        "(init: thr=%.2f  shp=%.1f)",
        sat_thr_final, sat_shp_final,
        SAT_THRESHOLD_INIT, SAT_SHARPNESS_INIT,
    )

    import json

    # ── test_metrics.json ─────────────────────────────────────────
    with open(ckpt_dir / "test_metrics.json", "w") as f:
        json.dump({
            **{k: round(v, 6) for k, v in test_metrics.items()},
            "mbe":                   round(mbe_global, 6),
            "sat_threshold_learned": round(sat_thr_final, 4),
            "sat_sharpness_learned": round(sat_shp_final, 4),
            "model":                 "st_gnn_soil_gate",
        }, f, indent=2)

    # ── per_node_metrics.csv ──────────────────────────────────────
    # compute_per_node_metrics returns a list of dicts (one per node).
    # Add ref, name, persist_nse and skill to match the format that
    # analyse_experiments.py expects.
    nodes_csv    = pd.read_csv(GRAPH_DIR / "nodes.csv")
    node_rows    = compute_per_node_metrics(cat_pred,    cat_tgt, cat_mask)
    persist_rows = compute_per_node_metrics(cat_persist, cat_tgt, cat_mask)

    pn_df = pd.DataFrame(node_rows)
    pn_df["ref"]          = nodes_csv["ref"].astype(str).values
    pn_df["name"]         = nodes_csv["name"].values
    pn_df["persist_nse"]  = [r["nse"] for r in persist_rows]
    pn_df["skill"]        = (
        (pn_df["nse"] - pn_df["persist_nse"])
        / (1 - pn_df["persist_nse"]).clip(lower=1e-8)
    ).round(4)
    pn_df = pn_df[["ref", "name", "n_valid", "rmse", "mae",
                    "mbe", "nse", "persist_nse", "skill"]]
    pn_df.to_csv(ckpt_dir / "per_node_metrics.csv", index=False)
    logger.info("  Saved per_node_metrics.csv")

    # ── per_step_metrics.json ─────────────────────────────────────
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

    logger.info("Done — %s", ckpt_dir)

    return model, test_metrics


