from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path
import pandas as pd


TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# ═══════════════════════════════════════════════════════════════════════
#  Dataset
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


def make_splits(n_total: int, t_in: int, t_out: int) -> tuple[range, range, range]:
    """
    Chronological train / val / test index ranges.
    Includes a 'gap' equal to T_IN + T_OUT between sets to prevent time-series leakage.
    """
    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * VAL_FRAC)
    gap = t_in + t_out

    return (
        range(0, n_train),
        range(n_train + gap, n_train + gap + n_val),
        range(n_train + 2 * gap + n_val, n_total),
    )


def make_dataset(X, y, valid_mask, split_range, t_in, t_out):
    """Slice tensor to the given range before wrapping in Dataset."""
    end = split_range.stop + t_in + t_out - 1  # include full window
    end = min(end, len(X))
    return LeeFloodDataset(
        X[split_range.start: end],
        y[split_range.start: end],
        valid_mask[split_range.start: end],
        t_in, t_out,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Graph construction helper
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
#  Loss
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
#  Metrics
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

def compute_per_step_metrics(pred: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor) -> list[dict]:
    """
    Compute RMSE, MAE, and NSE at each individual forecast step h,
    averaged across all nodes.

    Tensors shape: [B, T_out, N]

    Returns a list of T_out dicts, one per step:
        [{"step": 1, "rmse": ..., "mae": ..., "nse": ...}, ...]

    This enables the per-step horizon analysis recommended by
    Gao et al. (2022): graph models should show growing advantage
    over non-graph baselines as h increases, because upstream
    routing signals take time to propagate to downstream nodes.
    The advantage at h+1 (15 min) is expected to be small; at h+4
    (1 hr) it should be most pronounced when T_out=4.

    The mask is sliced to [:, h, :] for step h because valid_mask
    may differ by step (e.g., tidal stations flagged at certain
    steps due to data gaps).
    """
    T_out = pred.shape[1]
    N     = pred.shape[2]
    results = []

    for h in range(T_out):
        # Slice to step h: [B, N]
        pred_h = pred[:, h, :]
        tgt_h  = target[:, h, :]
        mask_h = mask[:, h, :]

        rmse_list, mae_list, nse_list = [], [], []
        for n in range(N):
            m = mask_h[:, n].bool()
            p = pred_h[:, n][m].cpu().float()
            t = tgt_h[:,  n][m].cpu().float()
            if len(t) < 2:
                continue
            rmse_list.append(((p - t) ** 2).mean().sqrt().item())
            mae_list.append((p - t).abs().mean().item())
            ss_res = ((p - t) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            nse_list.append((1 - ss_res / ss_tot.clamp(min=1e-8)).item())

        results.append({
            "step":  h + 1,                               # 1-indexed
            "lead_min": (h + 1) * 15,                     # minutes ahead
            "rmse": float(np.mean(rmse_list)) if rmse_list else float("nan"),
            "mae":  float(np.mean(mae_list))  if mae_list  else float("nan"),
            "nse":  float(np.mean(nse_list))  if nse_list  else float("nan"),
        })

    return results