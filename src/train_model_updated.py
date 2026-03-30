"""
train_model.py  v2  –  ST-GNN flood forecasting for the Lee catchment
======================================================================
Architecture:
  1. Node embedding   Linear([F_dyn + F_static]) → hidden_dim
  2. Spatial          GATConv (2 heads) at every timestep in the window
  3. Temporal         GRU (2 layers) over the T_in-step sequence
  4. Output           Linear head → T_out step forecast per node

Input window:   T_in  = 32 steps  (8 hours)
Output horizon: T_out =  4 steps  (1 hour, multi-step)

Target (v2):    DELTA stage_anomaly — the model predicts the CHANGE from
                the last observed value rather than the absolute level.
                  delta[h] = stage_anomaly[t+h] - stage_anomaly[t]
                Absolute predictions are recovered at evaluation time:
                  abs_pred[h] = last_obs + delta_pred[h]
                An output of 0.0 is now equivalent to persistence, so
                the model is explicitly penalised for lazy forecasting.

Loss (v2):      Horizon-weighted masked MSE — later steps in the forecast
                window receive proportionally more weight.
                  h+1: weight 1×,  h+2: 2×,  h+3: 3×,  h+4: 4×
                This forces the model to focus on the harder long-horizon
                predictions where persistence degrades most.

Train / val / test split: chronological 70 / 15 / 15 with T_in+T_out purge gaps
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "dataset/processed"
GRAPH_DIR = BASE_DIR / "dataset/graph"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
T_IN = 32  # input window  (8 hours at 15-min)
T_OUT = 4  # output horizon (1 hour)
HIDDEN_DIM = 64  # node embedding + GRU hidden size
GAT_HEADS = 2  # attention heads in GATConv
GRU_LAYERS = 2  # GRU depth
DROPOUT = 0.1

BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 30  # early stopping patience (epochs)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
# 1. Dataset
# ═══════════════════════════════════════════════════════════════════════

class LeeFloodDataset(Dataset):
    """
    Sliding-window dataset over the [T, N, F] dynamic tensor.

    Each sample:
      x_seq   float32  [T_in,  N, F_dyn]   input window
      y_seq   float32  [T_out, N]           target stage_anomaly
      mask    float32  [T_out, N]           1 = real, 0 = imputed
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 valid_mask: np.ndarray, t_in: int, t_out: int):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.mask = torch.from_numpy(valid_mask).float()
        self.t_in = t_in
        self.t_out = t_out
        # Last t_out steps cannot form a complete target window
        self.n_samples = len(X) - t_in - t_out + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.t_in]

        # y[t] = stage_anomaly at t+1, so start from t_in-1 to get 1-step-ahead first.
        # NOTE: This explicitly assumes `y.npy` was forward-shifted by 1 index during
        # preprocessing. If `y` is just an unshifted slice (e.g., X[:,:,0]), this will
        # cause massive target leakage because y_seq[0] will equal x_seq[-1].
        start = idx + self.t_in - 1
        y_seq = self.y[start: start + self.t_out]
        mask = self.mask[start: start + self.t_out]
        return x_seq, y_seq, mask


def make_splits(n_total: int) -> tuple[range, range, range]:
    """
    Chronological train / val / test index ranges.
    Includes a 'gap' equal to T_IN + T_OUT between sets to prevent time-series leakage.
    """
    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * VAL_FRAC)
    gap = T_IN + T_OUT

    return (
        range(0, n_train),
        range(n_train + gap, n_train + gap + n_val),
        range(n_train + 2 * gap + n_val, n_total),
    )


def make_dataset(X, y, valid_mask, split_range):
    """Slice tensor to the given range before wrapping in Dataset."""
    end = split_range.stop + T_IN + T_OUT - 1  # include full window
    end = min(end, len(X))
    return LeeFloodDataset(
        X[split_range.start: end],
        y[split_range.start: end],
        valid_mask[split_range.start: end],
        T_IN, T_OUT,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. Graph construction helper
# ═══════════════════════════════════════════════════════════════════════

def load_graph(logger, graph_dir: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    """
    Load static graph from CSVs produced by graph_builder.py.
    Returns (edge_index, edge_attr, node_attr) as tensors on device.
    """
    nodes_df = pd.read_csv(graph_dir / "nodes.csv")
    edges_df = pd.read_csv(graph_dir / "edges.csv")

    # Node static features — same 7 columns built by graph_builder
    static_cols = [
        "log_catchment_area_km2",
        "gauge_datum_mOSGM15",
        "p90_mAOD",
        "amax_med_mAOD",
        "is_reservoir",
        "is_tidal",
        "has_discharge",
    ]
    node_attr = torch.tensor(
        nodes_df[static_cols].values, dtype=torch.float32
    ).to(device)

    node_mean = node_attr.mean(dim=0)
    node_std = node_attr.std(dim=0).clamp(min=1e-6)
    node_attr = (node_attr - node_mean) / node_std

    # Edge index (src, dst) — already zero-indexed from graph_builder
    edge_index = torch.tensor(
        edges_df[["src_idx", "dst_idx"]].values.T, dtype=torch.long
    ).to(device)

    # Edge attributes
    edge_cols = ["river_dist_km", "area_ratio", "elev_drop_m", "same_tributary"]
    edge_attr_np = edges_df[edge_cols].values

    # Check if a node is a reservoir (is_reservoir == 1)
    is_res = nodes_df["is_reservoir"].values
    src_is_res = is_res[edges_df["src_idx"].values]
    dst_is_res = is_res[edges_df["dst_idx"].values]

    # Mask index 2 (elev_drop_m) to 0.0 if either source or destination is a reservoir
    reservoir_edge_mask = (src_is_res == 1) | (dst_is_res == 1)
    edge_attr_np[reservoir_edge_mask, 2] = 0.0

    edge_attr = torch.tensor(
        edge_attr_np, dtype=torch.float32
    ).to(device)

    logger.info(
        "Graph loaded: %d nodes, %d edges, node_attr %s, edge_attr %s",
        node_attr.shape[0], edge_index.shape[1],
        tuple(node_attr.shape), tuple(edge_attr.shape),
    )
    return edge_index, edge_attr, node_attr


# ═══════════════════════════════════════════════════════════════════════
# 3. Model
# ═══════════════════════════════════════════════════════════════════════

class STGNNFloodModel(nn.Module):
    """
    Spatio-Temporal GNN for flood stage-anomaly forecasting.

    Forward pass:
      x_seq      [B, T_in, N, F_dyn]
      node_attr  [N, F_static]
      edge_index [2, E]
      edge_attr  [E, F_edge]

    Returns:
      pred       [B, T_out, N]   predicted stage_anomaly
    """

    def __init__(
            self,
            f_dyn: int,  # dynamic feature dim  (5)
            f_static: int,  # static  feature dim  (7)
            f_edge: int,  # edge    feature dim  (4)
            hidden: int,
            gat_heads: int,
            gru_layers: int,
            t_out: int,
            dropout: float,
    ):
        super().__init__()
        self.hidden = hidden
        self.gat_heads = gat_heads
        self.t_out = t_out

        # ── Node embedding ─────────────────────────────────────────────
        # Combines dynamic + static features into hidden_dim
        self.node_embed = nn.Sequential(
            nn.Linear(f_dyn + f_static, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Edge embedding (project edge_attr to hidden_dim for GAT) ──
        self.edge_embed = nn.Linear(f_edge, hidden)

        # ── Spatial: 2-head GAT ────────────────────────────────────────
        # concat=True → output dim = hidden * gat_heads
        # We project back to hidden after GAT
        self.gat = GATConv(
            in_channels=hidden,
            out_channels=hidden // gat_heads,
            heads=gat_heads,
            concat=True,
            edge_dim=hidden,
            dropout=dropout,
            add_self_loops=True,
        )
        self.gat_norm = nn.LayerNorm(hidden)
        self.gat_drop = nn.Dropout(dropout)

        # ── Temporal: GRU ──────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ── Output head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # ── FIX: Added to prevent overfitting to specific temporal states
            nn.Linear(hidden // 2, t_out),
        )

    def forward(self, x_seq, node_attr, edge_index, edge_attr):
        B, T, N, _ = x_seq.shape

        # ── Edge embedding (once) ────────────────────────────────────────
        edge_feat = self.edge_embed(edge_attr)  # [E, hidden]

        # ── Node embedding (all timesteps at once) ───────────────────────
        static_exp = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        combined = torch.cat([x_seq, static_exp], dim=-1)  # [B, T, N, F]
        h = self.node_embed(combined.reshape(B * T * N, -1))
        h = h.view(B * T, N, self.hidden)  # [B*T, N, hidden]

        # ── ONE batched GAT call across B*T graphs ───────────────────────
        # Build edge index once — offset by N for each of the B*T graphs
        BT = B * T
        offsets = torch.arange(BT, device=edge_index.device) * N  # [B*T]
        ei = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)  # [B*T, 2, E]
        ei = ei.permute(1, 0, 2).reshape(2, -1)  # [2, B*T*E]

        # Expanding edges isn't the most memory-efficient PyTorch operation,
        # but works fine here given the static graph topology.
        ea = edge_feat.unsqueeze(0).expand(BT, -1, -1).reshape(-1, edge_feat.shape[-1])

        h_flat = h.reshape(BT * N, self.hidden)
        gat_out = self.gat(h_flat, ei, ea)  # one call
        gat_out = self.gat_norm(gat_out + h_flat)
        gat_out = self.gat_drop(gat_out)

        # ── GRU ─────────────────────────────────────────────────────────
        # [B*T, N, hidden] → [B, N, T, hidden] → [B*N, T, hidden]
        gru_in = gat_out.view(B, T, N, self.hidden) \
            .permute(0, 2, 1, 3) \
            .reshape(B * N, T, self.hidden)
        _, h_n = self.gru(gru_in)  # [layers, B*N, hidden]

        # ── Output ──────────────────────────────────────────────────────
        pred = self.head(h_n[-1]) \
            .view(B, N, self.t_out) \
            .permute(0, 2, 1)  # [B, T_out, N]
        return pred

# ═══════════════════════════════════════════════════════════════════════
# 4. Loss
# ═══════════════════════════════════════════════════════════════════════

def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    Uniform MSE over valid positions. Kept for persistence baseline
    evaluation where horizon weighting would distort the comparison.
    pred, target, mask: [B, T_out, N]
    """
    err = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return err.sum() / denom


def masked_mse_horizon_weighted(pred: torch.Tensor, target: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
    """
    Horizon-weighted MSE over valid positions.

    Later forecast steps receive higher weight so the model is penalised
    more for errors at h+4 (60 min) than at h+1 (15 min). The weights
    increase linearly and are normalised to mean=1, keeping the overall
    loss scale comparable to plain MSE.

      h+1: 0.4x   h+2: 0.8x   h+3: 1.2x   h+4: 1.6x  (T_out=4 example)

    pred, target, mask: [B, T_out, N]
    """
    T_out = pred.shape[1]
    # Linear ramp 1, 2, ..., T_out — normalised to mean = 1
    ramp = torch.arange(1, T_out + 1, dtype=torch.float32, device=pred.device)
    ramp = ramp / ramp.mean()                       # [T_out]
    w = ramp.view(1, T_out, 1) * mask               # [B, T_out, N]
    err = (pred - target) ** 2 * w
    return err.sum() / w.sum().clamp(min=1.0)


# ═══════════════════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> dict[str, float]:
    """
    Per-node RMSE, MAE, NSE — then averaged across nodes.
    Tensors: [B, T_out, N]
    """
    N = pred.shape[2]
    rmse_list, mae_list, nse_list = [], [], []

    for n in range(N):
        m = mask[:, :, n].bool()
        p = pred[:, :, n][m].cpu().float()
        t = target[:, :, n][m].cpu().float()

        if len(t) < 2:
            continue

        rmse_list.append(((p - t) ** 2).mean().sqrt().item())
        mae_list.append((p - t).abs().mean().item())

        ss_res = ((p - t) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()  # per-node mean, not global
        nse_list.append((1 - ss_res / ss_tot.clamp(min=1e-8)).item())

    return {
        "rmse": float(np.mean(rmse_list)),
        "mae": float(np.mean(mae_list)),
        "nse": float(np.mean(nse_list)),
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimiser, edge_index, edge_attr,
                node_attr) -> float:
    """
    One training epoch using delta (residual) prediction.

    The model predicts delta_stage_anomaly[h] = stage_anomaly[t+h] - stage_anomaly[t].
    The horizon-weighted loss penalises later forecast steps more heavily,
    preventing the model from collapsing to a near-zero-delta (persistence) solution.
    """
    model.train()
    total_loss = 0.0
    for x_seq, y_seq, mask in loader:
        x_seq = x_seq.to(DEVICE)
        y_seq = y_seq.to(DEVICE)
        mask  = mask.to(DEVICE)

        # ── Compute delta targets ──────────────────────────────────────
        # last_obs: stage_anomaly at the final input timestep [B, N]
        # delta_target[h] = stage_anomaly[t+h] - stage_anomaly[t]
        last_obs     = x_seq[:, -1, :, 0]                        # [B, N]
        delta_target = y_seq - last_obs.unsqueeze(1)              # [B, T_out, N]

        optimiser.zero_grad()
        delta_pred = model(x_seq, node_attr, edge_index, edge_attr)  # [B, T_out, N]
        loss = masked_mse_horizon_weighted(delta_pred, delta_target, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, edge_index, edge_attr, node_attr):
    """
    Evaluation epoch.

    The model outputs delta predictions. Absolute predictions are
    reconstructed as abs_pred = last_obs + delta_pred before computing
    all metrics, so RMSE, MAE, and NSE are in the original stage_anomaly
    space and directly comparable across runs.

    The training loss logged here uses the same horizon-weighted delta
    loss as train_epoch for a consistent comparison. Persistence metrics
    use plain masked_mse on absolute values — no horizon weighting — to
    give an honest baseline.
    """
    model.eval()
    total_loss = 0.0
    all_abs_pred, all_tgt, all_mask = [], [], []
    all_persist = []

    for x_seq, y_seq, mask in loader:
        x_seq = x_seq.to(DEVICE)
        y_seq = y_seq.to(DEVICE)
        mask  = mask.to(DEVICE)

        # ── Delta prediction → absolute reconstruction ─────────────────
        last_obs     = x_seq[:, -1, :, 0]                        # [B, N]
        delta_target = y_seq - last_obs.unsqueeze(1)              # [B, T_out, N]
        delta_pred   = model(x_seq, node_attr, edge_index, edge_attr)
        abs_pred     = last_obs.unsqueeze(1) + delta_pred         # [B, T_out, N]

        # Training-consistent loss (horizon-weighted, delta space)
        total_loss += masked_mse_horizon_weighted(
            delta_pred, delta_target, mask
        ).item()

        all_abs_pred.append(abs_pred.cpu())
        all_tgt.append(y_seq.cpu())
        all_mask.append(mask.cpu())

        # Persistence baseline: predict last_obs at every horizon
        persist = last_obs.unsqueeze(1).expand(-1, y_seq.shape[1], -1).cpu()
        all_persist.append(persist)

    cat_abs_pred = torch.cat(all_abs_pred)
    cat_tgt      = torch.cat(all_tgt)
    cat_mask     = torch.cat(all_mask)

    # Metrics on absolute predictions — horizon-uniform so NSE is comparable
    metrics = compute_metrics(cat_abs_pred, cat_tgt, cat_mask)
    persist_metrics = compute_metrics(
        torch.cat(all_persist), cat_tgt, cat_mask
    )
    return total_loss / len(loader), metrics, persist_metrics


def compute_per_node_metrics(
        pred: torch.Tensor,  # [B, T_out, N]
        target: torch.Tensor,  # [B, T_out, N]
        mask: torch.Tensor,  # [B, T_out, N]
) -> dict[str, list]:
    """
    Compute RMSE, MAE, and NSE individually for every node.
    Returns dict of lists, one value per node, in node-index order.
    """
    N = pred.shape[2]
    rows = []
    for n in range(N):
        m = mask[:, :, n].bool()
        p = pred[:, :, n][m].cpu().float()
        t = target[:, :, n][m].cpu().float()

        if len(t) < 2:
            rows.append({"node_idx": n, "rmse": np.nan,
                         "mae": np.nan, "nse": np.nan,
                         "n_valid": 0})
            continue

        rmse = ((p - t) ** 2).mean().sqrt().item()
        mae = (p - t).abs().mean().item()
        ss_res = ((p - t) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        nse = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()
        mbe = (p - t).mean().item()
        rows.append({"node_idx": n, "rmse": round(rmse, 6),
                     "mae": round(mae, 6), "nse": round(nse, 6),
                     "mbe": round(mbe, 6),
                     "n_valid": int(m.sum())})
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════════

def train(logger):
    logger.info("Device: %s", DEVICE)

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    X = np.load(PROC_DIR / "X.npy")
    y = np.load(PROC_DIR / "y.npy")
    valid_mask = np.load(PROC_DIR / "valid_mask.npy")

    T, N, F = X.shape
    logger.info("  X: %s  y: %s  valid_mask: %s", X.shape, y.shape, valid_mask.shape)

    # ── Load graph ─────────────────────────────────────────────────────
    edge_index, edge_attr, node_attr = load_graph(logger, GRAPH_DIR, DEVICE)

    # ── Splits ─────────────────────────────────────────────────────────
    n_windows = T - T_IN - T_OUT + 1
    train_rng, val_rng, test_rng = make_splits(n_windows)
    logger.info(
        "Windows — train: %d  val: %d  test: %d",
        len(train_rng), len(val_rng), len(test_rng)
    )

    train_ds = make_dataset(X, y, valid_mask, train_rng)
    val_ds = make_dataset(X, y, valid_mask, val_rng)
    test_ds = make_dataset(X, y, valid_mask, test_rng)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4,
                              pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=4,
                            pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────
    f_static = node_attr.shape[1]
    f_edge = edge_attr.shape[1]

    model = STGNNFloodModel(
        f_dyn=F, f_static=f_static, f_edge=f_edge,
        hidden=HIDDEN_DIM, gat_heads=GAT_HEADS,
        gru_layers=GRU_LAYERS, t_out=T_OUT, dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=20
    )

    # ── Training ───────────────────────────────────────────────────────
    best_val_loss = math.inf
    patience_ctr = 0
    history = []

    logger.info("Starting training …")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimiser, edge_index, edge_attr, node_attr
        )
        val_loss, val_metrics, persist_metrics = eval_epoch(
            model, val_loader, edge_index, edge_attr, node_attr
        )
        scheduler.step(val_loss)

        current_lr = optimiser.param_groups[0]['lr']
        logger.info("  LR: %.2e", current_lr)
        if current_lr <= 1e-5:
            logger.info("  LR floor reached — stopping")
            break

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        })

        logger.info(
            "Epoch %3d  train=%.4f  val=%.4f  "
            "Model  RMSE=%.4f NSE=%.4f  |  "
            "Persist RMSE=%.4f NSE=%.4f",
            epoch, train_loss, val_loss,
            val_metrics["rmse"], val_metrics["nse"],
            persist_metrics["rmse"], persist_metrics["nse"],
        )

        # ── Checkpoint on improvement ──────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "hparams": {
                    "t_in": T_IN, "t_out": T_OUT,
                    "hidden": HIDDEN_DIM, "gat_heads": GAT_HEADS,
                    "gru_layers": GRU_LAYERS, "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE, "lr": LR,
                },
            }, CKPT_DIR / "best_model.pt")
            logger.info("  ✓ Saved best model (val_loss=%.4f)", val_loss)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ── Save training history ──────────────────────────────────────────
    pd.DataFrame(history).to_csv(CKPT_DIR / "training_history.csv", index=False)

    # ── Test evaluation ────────────────────────────────────────────────
    logger.info("Loading best model for test evaluation …")
    ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    # Collect all predictions in one pass
    model.eval()
    all_abs_pred, all_tgt, all_mask = [], [], []
    with torch.no_grad():
        for x_seq, y_seq, mask in test_loader:
            x_seq    = x_seq.to(DEVICE)
            last_obs = x_seq[:, -1, :, 0]                        # [B, N]
            delta_pred = model(x_seq, node_attr, edge_index, edge_attr)
            abs_pred   = last_obs.unsqueeze(1) + delta_pred       # [B, T_out, N]
            all_abs_pred.append(abs_pred.cpu())
            all_tgt.append(y_seq)
            all_mask.append(mask)

    cat_pred = torch.cat(all_abs_pred)
    cat_tgt  = torch.cat(all_tgt)
    cat_mask = torch.cat(all_mask)

    # Aggregate metrics
    # Test loss: horizon-weighted delta-space loss (consistent with training)
    with torch.no_grad():
        delta_losses = []
        for x_seq, y_seq, mask in test_loader:
            x_seq    = x_seq.to(DEVICE)
            y_seq_d  = y_seq.to(DEVICE)
            mask_d   = mask.to(DEVICE)
            last_obs = x_seq[:, -1, :, 0]
            delta_target = y_seq_d - last_obs.unsqueeze(1)
            delta_pred   = model(x_seq, node_attr, edge_index, edge_attr)
            delta_losses.append(
                masked_mse_horizon_weighted(delta_pred, delta_target, mask_d).item()
            )
    test_loss    = float(np.mean(delta_losses))
    test_metrics = compute_metrics(cat_pred, cat_tgt, cat_mask)

    # ── Mean Bias Error ────────────────────────────────────────────────
    # MBE = mean(pred - target) over all valid positions
    # Near-zero MBE confirms the delta formulation removed systematic bias.
    # Positive MBE = model over-predicts, negative = under-predicts.
    m_all = cat_mask.bool()
    mbe_global = (cat_pred[m_all] - cat_tgt[m_all]).mean().item()

    logger.info(
        "\n✓ Test results:\n"
        "  Loss: %.4f\n  RMSE: %.4f\n  MAE:  %.4f\n  NSE:  %.4f\n  MBE:  %.4f m",
        test_loss,
        test_metrics["rmse"], test_metrics["mae"], test_metrics["nse"],
        mbe_global,
    )

    # ── Per-node metrics ───────────────────────────────────────────────
    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    node_rows = compute_per_node_metrics(cat_pred, cat_tgt, cat_mask)

    pn_df = pd.DataFrame(node_rows)
    pn_df["ref"] = nodes_df["ref"].astype(str).values
    pn_df["name"] = nodes_df["name"].values

    # Skill score vs persistence per node
    # Persistence: last observed stage_anomaly held constant across all horizons
    all_persist = []
    with torch.no_grad():
        for x_seq, y_seq, mask in test_loader:
            last_obs = x_seq[:, -1, :, 0]                        # [B, N]
            persist  = last_obs.unsqueeze(1).expand(-1, T_OUT, -1)
            all_persist.append(persist)

    cat_persist = torch.cat(all_persist)
    persist_rows = compute_per_node_metrics(cat_persist, cat_tgt, cat_mask)
    pn_df["persist_nse"] = [r["nse"] for r in persist_rows]
    pn_df["skill"] = (
            (pn_df["nse"] - pn_df["persist_nse"])
            / (1 - pn_df["persist_nse"]).clip(lower=1e-8)
    ).round(4)

    # Reorder columns for readability
    pn_df = pn_df[["ref", "name", "n_valid", "rmse", "mae", "mbe",
                   "nse", "persist_nse", "skill"]]
    pn_df.to_csv(CKPT_DIR / "per_node_metrics.csv", index=False)
    logger.info("  Saved per_node_metrics.csv")

    # Log bottom-5 and top-5 nodes by NSE
    sorted_pn = pn_df.dropna(subset=["nse"]).sort_values("nse")
    logger.info("\n  Lowest NSE nodes:\n%s",
                sorted_pn.head(5)[["name", "nse", "skill"]].to_string(index=False))
    logger.info("\n  Highest NSE nodes:\n%s",
                sorted_pn.tail(5)[["name", "nse", "skill"]].to_string(index=False))

    # Save aggregate test metrics
    with open(CKPT_DIR / "test_metrics.json", "w") as f:
        json.dump({
            "test_loss": test_loss,
            **test_metrics,
            "mbe": round(mbe_global, 6),
        }, f, indent=2)

    return model, test_metrics


if __name__ == "__main__":
    config_path = r"C:\Users\AdikariAdikari\PycharmProjects\ST-GNN\config\config.yaml"
    config = load_config(Path(config_path))
    logger = get_logger(config["logging"]["train"])
    train(logger)