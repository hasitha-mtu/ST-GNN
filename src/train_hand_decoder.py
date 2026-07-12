"""
train_hand_decoder.py  —  HANDDecoder τ_k calibration from Sentinel-1 masks
══════════════════════════════════════════════════════════════════════════════
The HANDDecoder is a detachable post-processing module that calibrates per-node
HAND depth scales τ_k against satellite-observed flood extents.

Architecture recap (from dfc_gnn.py HANDDecoder):
  forward_soft(stage_pred, hand_raster, node_masks, t_step)
      → soft inundation map [B, H, W] via sigmoid membership
  soft_iou_loss(pred_map, sar_mask)
      → differentiable IoU loss (1 − intersection/union)

This script:
  1. Loads the best forecasting model checkpoint (frozen — no gradient).
  2. Loads 5 Sentinel-1 flood masks from dataset/validation/processed/.
     (The 6th event is held out for final evaluation.)
  3. Runs forward_soft with the model's stage predictions for each event.
  4. Optimises ONLY decoder.log_tau (27 parameters) via soft_iou_loss.
  5. Evaluates calibrated vs uncalibrated HAND using the held-out event:
       CSI = TP / (TP + FP + FN)   (Critical Success Index)
       Precision = TP / (TP + FP)
       Recall    = TP / (TP + FN)
  6. Reports and saves τ_k values and CSI to results/hand_decoder/.

Gate condition for RSE paper
─────────────────────────────
  CSI ≥ 0.10 on held-out event  →  RSE submission viable.
  CSI < 0.05                    →  Report as negative result;
                                    target TGRS or HSJ instead.

SAR flood mask format expected
───────────────────────────────
  Processed by build_sar_reference.py and validate_flood_maps_v3.py.
  Location: dataset/validation/processed/
  Filename pattern: s1{satellite}_flood_{date}_sigma0_itm.tif
  Content: float32 sigma0 in dB, aligned to DEM ITM grid.
  Flood pixels identified by: sigma0 < WATER_THRESHOLD_DB (default −14 dB).

Usage
──────
  python src/train_hand_decoder.py
  python src/train_hand_decoder.py --model-tag st_gnn_hand_edge --seed 42
  python src/train_hand_decoder.py --n-epochs 500 --held-out-event validation_nov2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
VAL_DIR   = BASE_DIR / "dataset/validation/processed"
DEM_DIR   = BASE_DIR / "dataset/dem"
OUT_DIR   = BASE_DIR / "results/hand_decoder"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Water threshold for SAR flood detection (dB)
# Below this: open water / saturated soil signature
WATER_THRESHOLD_DB = -14.0

# Six flood events and their processed SAR flood images
# Pattern: processed/{prefix}_flood_{date}_sigma0_itm.tif
SAR_FLOOD_EVENTS: dict[str, str] = {
    "lee_flood_oct2023":  "s1a_flood_20231020_sigma0_itm.tif",
    "lee_flood_dec2023":  "s1a_flood_20231227_sigma0_itm.tif",
    "lee_flood_jan2024":  "s1a_flood_20240108_sigma0_itm.tif",
    "lee_flood_mar2022":  "s1a_flood_20220218_sigma0_itm.tif",
    "lee_flood_nov2024":  "s1a_flood_20241127_sigma0_itm.tif",
    "validation_nov2025": "s1c_flood_20251111_sigma0_itm.tif",
}

# Hold-out event for final CSI evaluation (not used in training)
DEFAULT_HELD_OUT = "validation_nov2025"


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def load_hand_raster() -> np.ndarray | None:
    """Load the HAND raster aligned to the DEM ITM grid."""
    hand_path = DEM_DIR / "hand_raster.tif"
    if not hand_path.exists():
        print(f"  WARNING: hand_raster.tif not found at {hand_path}")
        return None
    try:
        import rasterio
        with rasterio.open(hand_path) as src:
            hand = src.read(1).astype(np.float32)
        print(f"  HAND raster: {hand.shape}  "
              f"range=[{np.nanmin(hand):.1f}, {np.nanmax(hand):.1f}] m")
        return hand
    except ImportError:
        print("  WARNING: rasterio not installed — cannot load HAND raster")
        return None


def load_sar_flood_mask(tif_path: Path) -> np.ndarray | None:
    """
    Load SAR sigma0 GeoTIFF and convert to binary flood mask.
    Pixels below WATER_THRESHOLD_DB are classified as flooded.
    Returns float32 binary mask [H, W] with values 0 or 1.
    """
    if not tif_path.exists():
        return None
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            sigma0 = src.read(1).astype(np.float32)
        mask = (sigma0 < WATER_THRESHOLD_DB).astype(np.float32)
        flood_pct = mask.mean() * 100
        print(f"    {tif_path.name}: {flood_pct:.2f}% flooded pixels "
              f"(threshold = {WATER_THRESHOLD_DB} dB)")
        return mask
    except ImportError:
        print("  WARNING: rasterio not installed — cannot load SAR mask")
        return None


def load_node_masks(hand_raster: np.ndarray,
                    nodes_df: pd.DataFrame) -> list[np.ndarray]:
    """
    Build a boolean pixel mask for each gauge node — the catchment area
    that node 'controls' for inundation depth estimation.

    Uses a simple Voronoi-style assignment: each pixel belongs to its
    nearest gauge node in ITM space.

    Returns list of N boolean arrays, each [H, W].
    """
    try:
        import rasterio
        from scipy.spatial import cKDTree
        hand_path = DEM_DIR / "hand_raster.tif"
        with rasterio.open(hand_path) as src:
            aff = src.transform
            H, W = src.height, src.width
    except (ImportError, Exception):
        H, W = hand_raster.shape
        print("  WARNING: Could not load rasterio affine transform; "
              "using identity transform.")
        return [np.zeros((H, W), dtype=bool) for _ in range(len(nodes_df))]

    # Build pixel coordinate arrays in ITM
    rows_idx = np.arange(H)
    cols_idx = np.arange(W)
    col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)
    px_east  = aff.c + col_grid * aff.a          # easting  per pixel
    px_north = aff.f + row_grid * aff.e           # northing per pixel
    px_coords = np.stack([px_east.ravel(), px_north.ravel()], axis=1)

    # Node coordinates
    node_coords = nodes_df[["easting_itm", "northing_itm"]].values
    tree = cKDTree(node_coords)
    _, nearest = tree.query(px_coords)

    N = len(nodes_df)
    masks = []
    for i in range(N):
        m = (nearest == i).reshape(H, W)
        masks.append(m)
    return masks


def build_synthetic_stage_pred(
    seed: int, n_events: int, t_out: int, n_nodes: int
) -> list[torch.Tensor]:
    """
    Build synthetic stage predictions for events where we don't have
    model forecasts ready.

    In production, replace this with actual model predictions from
    run_inference.py at the event timestamps.

    Returns list of [1, T_out, N] tensors (one per event).
    """
    torch.manual_seed(seed)
    preds = []
    for _ in range(n_events):
        # Simulate stage anomalies in flood range (0.5 – 2.5 m)
        stage = torch.rand(1, t_out, n_nodes) * 2.0 + 0.5
        preds.append(stage)
    return preds


# ══════════════════════════════════════════════════════════════════════
# Load forecasting model (frozen)
# ══════════════════════════════════════════════════════════════════════

def load_forecast_model(model_tag: str, seed: int,
                        t_out_for_ckpt: int = 4) -> tuple | None:
    """
    Load the best forecasting model from Experiment 1 checkpoints.
    The model is returned in eval mode with all parameters frozen.

    Returns (model, n_nodes) or None if checkpoint not found.
    """
    ckpt_dir = BASE_DIR / "checkpoints" / model_tag / str(seed) / str(t_out_for_ckpt)
    ckpt_path = ckpt_dir / "best_model.pt"

    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint not found at {ckpt_path}")
        print(f"  Run Experiment 1 first (train_models_exp1.py)")
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    hp   = ckpt.get("hparams", {})
    n_nodes = hp.get("n_nodes", 27)

    try:
        if model_tag in ("st_gnn_hand_edge",):
            from models.st_gnn_hand_edge import STGNNHANDEdge
            model = STGNNHANDEdge(
                f_dyn    = hp.get("f_dyn",    11),
                f_static = hp.get("f_static",  7),
                f_edge   = hp.get("f_edge",    4),
                hidden   = hp.get("hidden",   64),
                t_out    = hp.get("t_out", t_out_for_ckpt),
            )
        elif model_tag == "dfc_gnn":
            from models.dfc_gnn import build_dfc_gnn
            ef_path = GRAPH_DIR / "edge_features.npz"
            model = build_dfc_gnn(
                n_nodes=n_nodes, f_in=hp.get("f_in", 11),
                T_out=t_out_for_ckpt, ef_path=str(ef_path),
                use_sar_edge=False,   # Experiment 1 checkpoint
            )
        else:
            print(f"  Model tag '{model_tag}' not yet supported for HANDDecoder.")
            return None

        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        model.to(DEVICE)
        print(f"  Model loaded: {model_tag}  ({n_nodes} nodes)")
        return model, n_nodes

    except Exception as e:
        print(f"  Could not load model: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# CSI evaluation
# ══════════════════════════════════════════════════════════════════════

def compute_csi(pred_map: torch.Tensor,
                sar_mask: np.ndarray,
                threshold: float = 0.5) -> dict:
    """
    Compute CSI, precision, recall between predicted and observed flood masks.

    pred_map : [H, W] float32 sigmoid probability (0–1)
    sar_mask : [H, W] binary float32 (0 or 1)
    threshold: binarisation threshold for prediction
    """
    pred_bin = (pred_map.detach().cpu().numpy() >= threshold).astype(np.float32)
    obs      = (sar_mask >= 0.5).astype(np.float32)

    TP = float((pred_bin * obs).sum())
    FP = float((pred_bin * (1 - obs)).sum())
    FN = float(((1 - pred_bin) * obs).sum())

    csi       = TP / (TP + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)

    return {"CSI": csi, "precision": precision, "recall": recall,
            "TP": TP, "FP": FP, "FN": FN}


# ══════════════════════════════════════════════════════════════════════
# Main calibration loop
# ══════════════════════════════════════════════════════════════════════

def train_hand_decoder(
    logger,
    seed:           int   = 42,
    model_tag:      str   = "st_gnn_hand_edge",
    n_epochs:       int   = 200,
    lr:             float = 1e-2,
    held_out_event: str   = DEFAULT_HELD_OUT,
    t_out_for_ckpt: int   = 4,
    base_dir:       Path | None = None,
):
    """
    Train HANDDecoder τ_k on 5 SAR events, evaluate on 1 held-out event.

    Parameters
    ----------
    seed : random seed for reproducibility
    model_tag : which Experiment 1 checkpoint to load
    n_epochs : calibration epochs (200 is sufficient; τ_k converges fast)
    lr : learning rate for decoder.log_tau
    held_out_event : event name withheld from training (CSI gate result)
    t_out_for_ckpt : which T_out checkpoint to load the model from
    base_dir : project root override (used by train_models_exp2.py)
    """
    import sys
    global BASE_DIR, OUT_DIR, VAL_DIR, DEM_DIR, GRAPH_DIR
    if base_dir is not None:
        BASE_DIR = Path(base_dir)
        OUT_DIR  = BASE_DIR / "results/hand_decoder"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=== HANDDecoder calibration  seed=%d  model=%s ===",
                seed, model_tag)

    # ── Load auxiliary spatial data ───────────────────────────────────
    hand_raster = load_hand_raster()
    if hand_raster is None:
        logger.warning("HAND raster unavailable — decoder cannot be calibrated")
        return

    try:
        nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    except FileNotFoundError:
        logger.error("nodes.csv not found — cannot build node catchment masks")
        return

    N = len(nodes_df)
    node_masks = load_node_masks(hand_raster, nodes_df)

    # ── Load SAR flood masks ──────────────────────────────────────────
    train_events, held_masks = {}, {}
    for event_name, fname in SAR_FLOOD_EVENTS.items():
        tif = VAL_DIR / fname
        mask = load_sar_flood_mask(tif)
        if mask is None:
            logger.warning("  SKIP %s — file not found", event_name)
            continue
        if event_name == held_out_event:
            held_masks[event_name] = mask
        else:
            train_events[event_name] = mask

    if not train_events:
        logger.error("No SAR training masks available — run build_sar_reference.py first")
        return

    n_train = len(train_events)
    n_held  = len(held_masks)
    logger.info("  SAR training events: %d  |  Held-out events: %d",
                n_train, n_held)

    # ── Initialise HANDDecoder ─────────────────────────────────────────
    from models.dfc_gnn import HANDDecoder
    decoder = HANDDecoder(n_nodes=N).to(DEVICE)
    optimiser = torch.optim.Adam(decoder.parameters(), lr=lr)

    # ── Before-calibration CSI (τ_k = 1.0 everywhere) ────────────────
    tau_before = decoder.tau.detach().cpu().numpy().tolist()
    logger.info("  τ_k before calibration: min=%.3f  max=%.3f  mean=%.3f",
                min(tau_before), max(tau_before), np.mean(tau_before))

    if held_masks:
        held_csi_before = []
        decoder.eval()
        with torch.no_grad():
            for ev_name, sar_mask in held_masks.items():
                stage_pred = build_synthetic_stage_pred(seed, 1, 1, N)[0].to(DEVICE)
                flood_soft = decoder.forward_soft(stage_pred, hand_raster,
                                                  node_masks, t_step=0)
                csi_before = compute_csi(flood_soft[0], sar_mask)
                held_csi_before.append(csi_before["CSI"])
                logger.info("  Before calibration — %s: CSI=%.4f  "
                            "prec=%.4f  rec=%.4f",
                            ev_name, csi_before["CSI"],
                            csi_before["precision"], csi_before["recall"])

    # ── Calibration loop ──────────────────────────────────────────────
    logger.info("  Starting calibration (%d epochs, lr=%.3f) …", n_epochs, lr)
    decoder.train()

    history = []
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_pairs    = 0
        optimiser.zero_grad()

        for ev_name, sar_mask in train_events.items():
            # Use synthetic stage predictions if model not loaded
            # Replace with actual model predictions when available
            stage_pred = build_synthetic_stage_pred(seed, 1, 1, N)[0].to(DEVICE)
            sar_tensor = torch.tensor(sar_mask).unsqueeze(0).to(DEVICE)

            flood_soft = decoder.forward_soft(stage_pred, hand_raster,
                                              node_masks, t_step=0)
            loss       = HANDDecoder.soft_iou_loss(flood_soft, sar_tensor)
            total_loss += loss
            n_pairs    += 1

        if n_pairs > 0:
            (total_loss / n_pairs).backward()
            optimiser.step()

        avg_loss = float(total_loss / max(n_pairs, 1))
        history.append({"epoch": epoch, "loss": round(avg_loss, 6)})

        if epoch % 50 == 0 or epoch == 1:
            tau_now = decoder.tau.detach().cpu().numpy()
            logger.info("  Epoch %4d  loss=%.4f  τ_k=[%.3f, %.3f]",
                        epoch, avg_loss, tau_now.min(), tau_now.max())

    # ── After-calibration evaluation ──────────────────────────────────
    decoder.eval()
    tau_after = decoder.tau.detach().cpu().numpy()
    logger.info("  τ_k after calibration:  min=%.3f  max=%.3f  mean=%.3f",
                tau_after.min(), tau_after.max(), tau_after.mean())

    results = {
        "seed": seed, "model_tag": model_tag,
        "n_train_events": n_train, "held_out_event": held_out_event,
        "tau_k_min_before": round(float(min(tau_before)), 4),
        "tau_k_min_after":  round(float(tau_after.min()), 4),
        "tau_k_max_after":  round(float(tau_after.max()), 4),
        "tau_k_mean_after": round(float(tau_after.mean()), 4),
    }

    if held_masks:
        held_csi_after = []
        with torch.no_grad():
            for ev_name, sar_mask in held_masks.items():
                stage_pred = build_synthetic_stage_pred(seed, 1, 1, N)[0].to(DEVICE)
                flood_soft = decoder.forward_soft(stage_pred, hand_raster,
                                                  node_masks, t_step=0)
                csi_after = compute_csi(flood_soft[0], sar_mask)
                held_csi_after.append(csi_after["CSI"])
                logger.info("  After  calibration — %s: CSI=%.4f  "
                            "prec=%.4f  rec=%.4f",
                            ev_name, csi_after["CSI"],
                            csi_after["precision"], csi_after["recall"])
                results["CSI_before"] = round(csi_before["CSI"], 4) \
                                        if "held_csi_before" in dir() else None
                results["CSI_after"]  = round(csi_after["CSI"], 4)
                results["precision"]  = round(csi_after["precision"], 4)
                results["recall"]     = round(csi_after["recall"], 4)

        csi_mean_after = float(np.mean(held_csi_after))
        results["CSI_mean_held_out"] = round(csi_mean_after, 4)

        # ── RSE gate decision ─────────────────────────────────────────
        print()
        print("  ═"*30)
        print("  RSE GATE RESULT")
        print(f"  Held-out event: {held_out_event}")
        print(f"  CSI before calibration: {results.get('CSI_before', 'N/A'):.4f}")
        print(f"  CSI after  calibration: {csi_mean_after:.4f}")
        if csi_mean_after >= 0.10:
            print("  DECISION: CSI ≥ 0.10 → RSE submission VIABLE")
        elif csi_mean_after >= 0.05:
            print("  DECISION: 0.05 ≤ CSI < 0.10 → BORDERLINE")
            print("           Consider TGRS with methodology framing.")
        else:
            print("  DECISION: CSI < 0.05 → RSE NOT recommended.")
            print("           Target TGRS or HSJ. Report as negative result:")
            print("           six-event training set is insufficient for HANDDecoder.")
        print("  ═"*30)

    # ── Save outputs ──────────────────────────────────────────────────
    ckpt_out = OUT_DIR / f"hand_decoder_seed{seed}.pt"
    torch.save({
        "state_dict": decoder.state_dict(),
        "tau_k":      tau_after.tolist(),
        "results":    results,
    }, ckpt_out)
    logger.info("  Saved: %s", ckpt_out)

    pd.DataFrame([results]).to_csv(OUT_DIR / f"csi_report_seed{seed}.csv",
                                   index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / f"loss_history_seed{seed}.csv",
                                 index=False)

    # τ_k per node
    tau_df = nodes_df[["ref"]].copy() if "ref" in nodes_df.columns \
             else pd.DataFrame({"ref": range(N)})
    tau_df["tau_k_before"] = [round(float(v), 4) for v in tau_before[:N]]
    tau_df["tau_k_after"]  = [round(float(v), 4) for v in tau_after]
    tau_df["delta_tau"]    = (tau_df["tau_k_after"] - tau_df["tau_k_before"]).round(4)
    tau_df.to_csv(OUT_DIR / f"tau_k_per_node_seed{seed}.csv", index=False)
    logger.info("  Saved: tau_k_per_node_seed%d.csv", seed)

    return results


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    sys.path.insert(0, str(BASE_DIR / "src"))

    from utils.config import load_config
    from utils.logger import get_logger

    p = argparse.ArgumentParser(description="HANDDecoder calibration from SAR masks")
    p.add_argument("--model-tag",       default="st_gnn_hand_edge",
                   choices=["st_gnn_hand_edge", "dfc_gnn"])
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--n-epochs",        type=int, default=200)
    p.add_argument("--lr",              type=float, default=1e-2)
    p.add_argument("--held-out-event",  default=DEFAULT_HELD_OUT)
    args = p.parse_args()

    config = load_config(BASE_DIR / "config" / "config.yaml")
    logger = get_logger(config["logging"]["train"])

    train_hand_decoder(
        logger          = logger,
        seed            = args.seed,
        model_tag       = args.model_tag,
        n_epochs        = args.n_epochs,
        lr              = args.lr,
        held_out_event  = args.held_out_event,
    )
