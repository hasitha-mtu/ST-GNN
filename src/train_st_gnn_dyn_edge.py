"""
train_st_gnn_dyn_edge.py  –  PI-ST-GNN Phase 1: discharge-weighted dynamic edges
==================================================================================
Architecture
------------
  Static river-network graph with a fifth dynamic edge feature per timestep.
  Feature 4 = hydraulic conductance derived from upstream discharge at each edge,
  following Manning's equation (conductance increases nonlinearly with stage/Q).

  All other components identical to STGNNFloodModel:
    1. Node feature projection  Linear(F_dyn + F_static) → hidden
    2. Temporal encoder         Per-node GRU
    3. SAR fusion (optional)    Concat(GRU_out, SAR_emb) → Linear + LN
    4. Graph message passing    GATConv ×3 (f_edge=5)
    5. Output head              Linear → delta stage_anomaly [B, T_out, N]

Physical basis
--------------
  Manning Q = (1/n) A R^(2/3) S^(1/2): as discharge rises, hydraulic radius
  increases, so reach conductance increases nonlinearly. Encoded as:
    conductance = sigmoid(scale × (Q_src / Q_ref − 1))
  Operationalised as a 5th edge feature computed from discharge (feature [3])
  at the upstream node of each reach at each forward pass.

Checkpoint directory: checkpoints/st_gnn_dyn_edge/
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

from src.models.st_gnn_dyn_edge import STGNNDynEdge as STGNNFloodModel
from src.models.sar_fno_encoder import SARFNOEncoder, compute_node_coords_norm

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
SAR_DIR   = BASE_DIR / "dataset/sar"
LIVE_METRICS_PATH = BASE_DIR / "checkpoints"
# ── Feature dimensions ─────────────────────────────────────────────────
SAR_EMB_DIM = 16     # FNO encoder output channels per node
# Fusion input = GRU output (HIDDEN_DIM=64) + SAR embedding (16) = 80
# Fusion output = 64 (same as GRU-only path for fair comparison)

# ── SAR on/off flag for ablation runs ──────────────────────────────────
USE_SAR = True

# ── Hyperparameters ────────────────────────────────────────────────────
HIDDEN_DIM = 64
GAT_HEADS  = 2
GRU_LAYERS = 2
DROPOUT    = 0.1

# SAR encode resolution — the FNO runs at this spatial resolution,
# not at the full preprocessed raster size (1960×3840 at 20m).
# The FNO is resolution-invariant: it encodes spatial patterns then
# bilinearly samples at 27 node positions regardless of raster size.
# Memory at full res: ~44 GB.  At 256×256: ~385 MB (fits 8 GB GPU).
# Minimum recommended: 128×128.  Diminishing returns above 512×512.
SAR_ENCODE_H = 256
SAR_ENCODE_W = 256

# FNO encoder hyperparameters
FNO_WIDTH      = 32
FNO_MODES_HIGH = 12    # blocks 1–2: fine boundary detail
FNO_MODES_LOW  = 8     # blocks 3–4: smooth spatial context

BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
#  SAR data helpers
# ═══════════════════════════════════════════════════════════════════════

def load_sar_events(logger) -> dict | None:
    """
    Load sar_events.json mapping event IDs to SAR acquisition metadata.

    Returns None if SAR data is unavailable (graceful fallback to baseline).

    Expected JSON structure:
    {
      "event_001": {
        "sar_path":         "sar/event_001.npy",   // relative to PROC_DIR
        "acquisition_date": "2019-11-26",
        "event_start":      "2019-11-26T00:00",
        "event_end":        "2019-11-28T23:45",
        "bbox_itm":         [x_min, y_min, x_max, y_max]
      },
      ...
    }
    """
    events_path = SAR_DIR / "sar_events.json"
    if not events_path.exists():
        logger.warning(
            "sar_events.json not found at %s. "
            "Running without SAR (baseline mode). "
            "Set USE_SAR=False to suppress this warning.",
            events_path,
        )
        return None

    with open(events_path) as f:
        events = json.load(f)

    logger.info("SAR events loaded: %d acquisitions", len(events))
    return events


def build_sar_embedding_cache(
    encoder:          SARFNOEncoder,
    sar_events:       dict,
    node_xy:          torch.Tensor,   # [N, 2] ITM coords
    logger,
) -> dict:
    """
    Pre-compute and cache FNO node embeddings for all SAR acquisitions.

    Running the FNO encoder once per acquisition (rather than per training
    step) is a deliberate efficiency decision:
      - The SAR raster is constant per event; re-encoding it every batch
        wastes ~5–10ms per batch with zero information gain.
      - The cache is built once at the start of training and held in CPU
        memory (~27 nodes × 16 dims × n_events × 4 bytes ≈ negligible).
      - During training, the embedding for the current event is looked up
        and moved to GPU as needed.

    Returns dict: {event_id: torch.Tensor [N, 16] on CPU}
    """
    cache = {}
    encoder.eval()

    for event_id, meta in sar_events.items():
        sar_path = SAR_DIR / meta["sar_path"]
        if not sar_path.exists():
            logger.warning("SAR file missing: %s — skipping event %s",
                           sar_path, event_id)
            continue

        sar = torch.from_numpy(
            np.load(sar_path).astype(np.float32)
        ).to(DEVICE)                                       # [2, H_orig, W_orig]

        # ── Downsample to encode resolution ──────────────────────
        # The FNO is resolution-invariant: spatial patterns are encoded
        # then bilinearly sampled at node positions. Running at full
        # raster size (1960×3840) needs ~44 GB; 256×256 needs ~385 MB.
        # node_coords_norm is computed from the encode dimensions so
        # the normalised coordinates [-1,+1] remain correct.
        H_orig, W_orig = sar.shape[1], sar.shape[2]
        if H_orig != SAR_ENCODE_H or W_orig != SAR_ENCODE_W:
            sar = F.interpolate(
                sar.unsqueeze(0),                          # [1, 2, H, W]
                size=(SAR_ENCODE_H, SAR_ENCODE_W),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)                                   # [2, H_enc, W_enc]

        node_coords_norm = compute_node_coords_norm(
            node_xy.cpu(), meta["bbox_itm"],
            SAR_ENCODE_H, SAR_ENCODE_W,
        ).to(DEVICE)                                       # [N, 2]

        emb = encoder.encode_event(sar, node_coords_norm) # [N, 16]
        cache[event_id] = emb.cpu()                        # store on CPU

    logger.info(
        "SAR encode resolution: %d × %d px  "
        "(original raster resampled from full preprocessed size)",
        SAR_ENCODE_H, SAR_ENCODE_W,
    )
    logger.info(
        "SAR embedding cache built: %d events, "
        "each [%d nodes × %d dims]",
        len(cache),
        next(iter(cache.values())).shape[0] if cache else 0,
        SAR_EMB_DIM,
    )

    # ── Save SAR diagnostics for dashboard ───────────────────────
    if cache:
        diag = {
            "events": {
                eid: {
                    "emb_mean":   emb.mean(dim=0).tolist(),
                    "emb_std":    emb.std(dim=0).tolist(),
                    "node_norms": emb.norm(dim=1).tolist(),
                    "acquisition_date": sar_events[eid].get("acquisition_date"),
                    "sar_path":         sar_events[eid].get("sar_path"),
                }
                for eid, emb in cache.items()
            },
            "n_nodes":    next(iter(cache.values())).shape[0],
            "emb_dim":    SAR_EMB_DIM,
            "encode_res": [SAR_ENCODE_H, SAR_ENCODE_W],
        }
        import ast as _a
        sar_diag_path = LIVE_METRICS_PATH.parent / "sar_diagnostics.json"
        sar_diag_path.parent.mkdir(parents=True, exist_ok=True)
        sar_diag_path.write_text(json.dumps(diag, indent=2))

    return cache


def get_sar_embedding_for_batch(
    window_start_idx: int,        # index of first timestep in this batch window
    event_lookup:     list,       # list mapping timestep index → event_id (or None)
    sar_cache:        dict,       # {event_id: [N, 16] CPU tensor}
    N:                int,        # number of nodes
) -> torch.Tensor:
    """
    Retrieve the SAR embedding for a batch window.

    The embedding is determined by the event that contains the FIRST
    timestep of the window. This is conservative — if a window straddles
    two events, we use the pre-event SAR image (the earlier acquisition),
    which is the physically correct choice for an antecedent wetness proxy.

    Returns [N, 16] on DEVICE, or a zero tensor if no SAR is available
    for this window (used as a graceful fallback for windows outside
    any defined event).
    """
    event_id = event_lookup[window_start_idx]

    if event_id is None or event_id not in sar_cache:
        return torch.zeros(N, SAR_EMB_DIM, device=DEVICE)

    return sar_cache[event_id].to(DEVICE)


def build_event_lookup(sar_events: dict, T: int,
                       timestamps: pd.DatetimeIndex) -> list:
    """
    Build a list of length T mapping each timestep index to its event_id.

    Timesteps not covered by any defined SAR event map to None.
    Used by get_sar_embedding_for_batch to resolve event IDs at batch time.

    Parameters
    ----------
    sar_events : dict
        Output of load_sar_events().
    T : int
        Total number of timesteps in the dataset.
    timestamps : pd.DatetimeIndex
        Datetime index of length T corresponding to each timestep in X.npy.
        Load from PROC_DIR / "timestamps.csv" or derive from X metadata.
    """
    lookup = [None] * T

    for event_id, meta in sar_events.items():
        start = pd.Timestamp(meta["event_start"])
        end   = pd.Timestamp(meta["event_end"])

        # Find timestep indices within this event window
        mask = (timestamps >= start) & (timestamps <= end)
        for idx in np.where(mask)[0]:
            lookup[int(idx)] = event_id

    return lookup


# ═══════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(
    model, loader, optimiser,
    edge_index, edge_attr, node_attr,
    sar_cache, event_lookup, N,
    use_sar: bool,
) -> float:
    """
    One training epoch with quasi-static SAR embeddings.

    For each batch:
      1. Retrieve the SAR embedding for this window (lookup, not recompute).
      2. Expand SAR embedding across the batch dimension B.
      3. Pass [x_seq, node_attr, edge_index, edge_attr, sar_emb] to model.

    The FNO encoder is NOT in the gradient graph during main training.
    It is pre-computed and cached. If you want to fine-tune the encoder
    end-to-end, set encoder.train() and remove the no_grad wrapping
    in build_sar_embedding_cache — but note this roughly doubles the
    memory footprint per batch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (x_seq, y_seq, mask) in enumerate(loader):
        x_seq = x_seq.to(DEVICE)   # [B, T_in, N, F]
        y_seq = y_seq.to(DEVICE)   # [B, T_out, N]
        mask  = mask.to(DEVICE)    # [B, T_out, N]

        last_obs     = x_seq[:, -1, :, 0]              # [B, N]
        delta_target = y_seq - last_obs.unsqueeze(1)   # [B, T_out, N]

        # ── SAR embedding (quasi-static per event) ────────────────────
        # batch_idx * batch_size gives a rough window start; in a full
        # implementation you would track exact window indices from the
        # DataLoader using a custom sampler. See implementation note below.
        if use_sar and sar_cache is not None:
            B = x_seq.shape[0]
            # [N, 16] → [B, N, 16]
            sar_emb = get_sar_embedding_for_batch(
                batch_idx * loader.batch_size, event_lookup, sar_cache, N
            ).unsqueeze(0).expand(B, -1, -1)
        else:
            sar_emb = None

        optimiser.zero_grad(set_to_none=True)
        delta_pred = model(x_seq, node_attr, edge_index, edge_attr, sar_emb)
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
    sar_cache, event_lookup, N,
    use_sar: bool,
):
    model.eval()
    total_loss = 0.0
    all_abs_pred, all_tgt, all_mask, all_persist = [], [], [], []

    for batch_idx, (x_seq, y_seq, mask) in enumerate(loader):
        x_seq = x_seq.to(DEVICE)
        y_seq = y_seq.to(DEVICE)
        mask  = mask.to(DEVICE)

        last_obs     = x_seq[:, -1, :, 0]
        delta_target = y_seq - last_obs.unsqueeze(1)

        if use_sar and sar_cache is not None:
            B = x_seq.shape[0]
            sar_emb = get_sar_embedding_for_batch(
                batch_idx * loader.batch_size, event_lookup, sar_cache, N
            ).unsqueeze(0).expand(B, -1, -1)
        else:
            sar_emb = None

        delta_pred = model(x_seq, node_attr, edge_index, edge_attr, sar_emb)
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
    run_tag  = "st_gnn_dyn_edge"
    ckpt_dir = base_dir / "checkpoints" / run_tag / str(seed) / str(t_out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training ST-GNN Phase 1: dynamic edge weights%s ===", " + SAR-FNO" if USE_SAR else "")
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

    # ── SAR encoder + cache ────────────────────────────────────────────
    sar_cache    = None
    event_lookup = [None] * T
    use_sar_run  = USE_SAR

    if USE_SAR:
        sar_events = load_sar_events(logger)

        if sar_events is not None:
            encoder = SARFNOEncoder(
                in_channels=2,
                width=FNO_WIDTH,
                out_channels=SAR_EMB_DIM,
                modes_high=FNO_MODES_HIGH,
                modes_low=FNO_MODES_LOW,
            ).to(DEVICE)

            n_enc_params = sum(
                p.numel() for p in encoder.parameters() if p.requires_grad
            )
            logger.info("FNO encoder parameters: %s", f"{n_enc_params:,}")

            # Node ITM coordinates from nodes.csv
            nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
            nodes_df = update_node_coordinates(nodes_df)
            node_xy  = torch.tensor(
                nodes_df[["easting_itm", "northing_itm"]].values,
                # nodes_df[["lat", "lon"]].values,
                dtype=torch.float32,
            ).to(DEVICE)

            sar_cache = build_sar_embedding_cache(
                encoder, sar_events, node_xy, logger
            )

            # Build timestep → event_id lookup
            timestamps_path = PROC_DIR / "timestamps.csv"
            print(f'timestamps_path: {timestamps_path}')
            if timestamps_path.exists():
                timestamps   = pd.to_datetime(
                    pd.read_csv(timestamps_path)["timestamp"]
                )
                event_lookup = build_event_lookup(sar_events, T, timestamps)
                n_covered    = sum(e is not None for e in event_lookup)
                logger.info(
                    "Event lookup built: %d / %d timesteps covered by SAR",
                    n_covered, T,
                )
            else:
                logger.warning(
                    "timestamps.csv not found — SAR event lookup cannot be "
                    "built. Falling back to no-SAR baseline."
                )
                sar_cache   = None
                use_sar_run = False
        else:
            use_sar_run = False

    if not use_sar_run:
        logger.info("Running without SAR (baseline GATConv path).")

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
    f_edge   = edge_attr.shape[1]
    # When SAR is active the fusion layer input is F_dyn + f_static + SAR_EMB_DIM;
    # the model handles this via the sar_emb_dim constructor argument.
    # Phase 1: f_edge=5 (4 static + 1 dynamic conductance)
    # Q_ref = mean discharge across all nodes and training timesteps
    # after log1p transform (matches build_dataset.py preprocessing).
    discharge_col = 3   # index of discharge_m3s in feature vector
    q_ref = float(X[:len(train_rng), :, discharge_col].mean())
    q_ref = max(q_ref, 0.01)   # floor to avoid division by zero
    logger.info("  Discharge reference Q_ref=%.4f (log1p-transformed)", q_ref)

    model = STGNNFloodModel(
        f_dyn=F,
        f_static=f_static,
        f_edge=f_edge + 1,     # 4 static + 1 dynamic conductance
        hidden=HIDDEN_DIM,
        gat_heads=GAT_HEADS,
        gru_layers=GRU_LAYERS,
        t_out=t_out,
        dropout=DROPOUT,
        sar_emb_dim=SAR_EMB_DIM if use_sar_run else 0,
        discharge_idx=discharge_col,
        discharge_ref=q_ref,
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
            sar_cache, event_lookup, N, use_sar_run,
        )
        val_loss, val_metrics, persist_metrics = eval_epoch(
            model, val_loader,
            edge_index, edge_attr, node_attr,
            sar_cache, event_lookup, N, use_sar_run,
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
                "use_sar":     use_sar_run,
                "hparams": {
                    "t_in": t_in, "t_out": t_out,
                    "f_dyn": F,
                    "hidden": HIDDEN_DIM, "gat_heads": GAT_HEADS,
                    "gru_layers": GRU_LAYERS, "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE, "lr": LR,
                    "sar_emb_dim": SAR_EMB_DIM if use_sar_run else 0,
                    "f_edge_dyn": f_edge + 1,
                    "discharge_ref": q_ref,
                    "fno_width": FNO_WIDTH,
                    "fno_modes_high": FNO_MODES_HIGH,
                    "fno_modes_low":  FNO_MODES_LOW,
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

            if use_sar_run and sar_cache is not None:
                B = x_seq.shape[0]
                sar_emb = get_sar_embedding_for_batch(
                    batch_idx * test_loader.batch_size,
                    event_lookup, sar_cache, N,
                ).unsqueeze(0).expand(B, -1, -1)
            else:
                sar_emb = None

            delta_pred = model(x_seq, node_attr, edge_index, edge_attr, sar_emb)
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
            "use_sar": use_sar_run,
            "model":   "stgnn_dyn_edge",
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
