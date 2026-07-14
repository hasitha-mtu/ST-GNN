from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path
import pandas as pd


TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

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
        self.X    = torch.from_numpy(X).float()
        self.y    = torch.from_numpy(y).float()
        self.mask = torch.from_numpy(valid_mask).float()
        self.t_in  = t_in
        self.t_out = t_out
        # Last t_out steps cannot form a complete target window
        self.n_samples = len(X) - t_in - t_out + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.t_in]

        # ── TARGET ALIGNMENT — READ BEFORE CHANGING ─────────────────────
        # Two possible conventions for y.npy:
#
        #   SHIFTED   y[t] = stage_anomaly at t+1  (y.npy is shifted forward)
        #             → start = idx + t_in - 1  ← CURRENT CODE
        #             y_seq[0] = stage at (idx + t_in)  ✓ correct 1-step target
#
        #   UNSHIFTED y[t] = stage_anomaly at t    (y.npy stores t directly)
        #             → start = idx + t_in          ← ALTERNATIVE
        #             If using t_in-1 with unshifted y:
        #             y_seq[0] = stage at (idx + t_in - 1) = x[-1,:,0]
        #             delta_target[0] = 0 exactly → target leakage!
#
        # VERIFY: open build_dataset.py and check whether y is saved as
        #   y = df["stage_anomaly"].values[1:]   (shifted — use t_in-1)
        #   y = df["stage_anomaly"].values        (unshifted — use t_in)
        # The NSE=0.9475 at epoch 1 suggests the current convention is
        # internally consistent, but it must match build_dataset.py exactly.
        start = idx + self.t_in - 1
        y_seq = self.y[start: start + self.t_out]
        mask  = self.mask[start: start + self.t_out]
        return x_seq, y_seq, mask


def make_splits(n_total: int, t_in: int, t_out: int) -> tuple[range, range, range]:
    """
    Chronological train / val / test index ranges with purge gaps.

    Includes a gap equal to T_IN + T_OUT between each split to prevent
    time-series leakage across boundaries.

    Fractions are controlled by the module-level TRAIN_FRAC / VAL_FRAC constants.
    """
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)
    gap     = t_in + t_out

    return (
        range(0, n_train),
        range(n_train + gap, n_train + gap + n_val),
        range(n_train + 2 * gap + n_val, n_total),
    )


def make_dataset(X, y, valid_mask, split_range, t_in, t_out):
    """Slice tensors to the given range before wrapping in LeeFloodDataset."""
    end = split_range.stop + t_in + t_out - 1   # include the full last window
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
    node_std  = node_attr.std(dim=0).clamp(min=1e-6)
    node_attr = (node_attr - node_mean) / node_std

    # Edge index (src, dst) — already zero-indexed from graph_builder
    edge_index = torch.tensor(
        edges_df[["src_idx", "dst_idx"]].values.T, dtype=torch.long
    ).to(device)

    # Edge attributes
    edge_cols = ["river_dist_km", "area_ratio", "elev_drop_m", "same_tributary"]
    edge_attr_np = edges_df[edge_cols].values

    # Mask elev_drop_m (index 2) to 0.0 where source or destination is a reservoir,
    # because impounded reaches have no meaningful hydraulic gradient.
    is_res = nodes_df["is_reservoir"].values
    src_is_res = is_res[edges_df["src_idx"].values]
    dst_is_res = is_res[edges_df["dst_idx"].values]
    reservoir_edge_mask = (src_is_res == 1) | (dst_is_res == 1)
    edge_attr_np[reservoir_edge_mask, 2] = 0.0

    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32).to(device)

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
    Uniform MSE over valid positions.

    Kept for persistence baseline evaluation where horizon weighting
    would distort the comparison.
    pred, target, mask: [B, T_out, N]
    """
    err   = (pred - target) ** 2 * mask
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
    ramp  = torch.arange(1, T_out + 1, dtype=torch.float32, device=pred.device)
    ramp  = ramp / ramp.mean()                       # normalise to mean=1  [T_out]
    w     = ramp.view(1, T_out, 1) * mask            # [B, T_out, N]
    err   = (pred - target) ** 2 * w
    return err.sum() / w.sum().clamp(min=1.0)


# ═══════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> dict[str, float]:
    """
    Aggregate RMSE, MAE, NSE — per node then averaged.

    Uses per-node mean for NSE denominator (ss_tot) rather than global
    mean, which is the hydrologically correct formulation.
    Tensors: [B, T_out, N]
    """
    # Normalise device: all three tensors must be on the same device
    # before boolean indexing.  mask.bool() inherits the mask's device;
    # if pred/target were moved to CPU but mask was not (or vice versa),
    # `pred[:, :, n][m]` raises RuntimeError on the device mismatch.
    # Moving everything to CPU here makes the function device-agnostic.
    pred   = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    mask   = mask.detach().cpu()
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
        ss_tot = ((t - t.mean()) ** 2).sum()
        nse_list.append((1 - ss_res / ss_tot.clamp(min=1e-8)).item())

    return {
        "rmse": float(np.mean(rmse_list)),
        "mae":  float(np.mean(mae_list)),
        "nse":  float(np.mean(nse_list)),
    }


def compute_per_node_metrics(
        pred:   torch.Tensor,   # [B, T_out, N]
        target: torch.Tensor,   # [B, T_out, N]
        mask:   torch.Tensor,   # [B, T_out, N]
) -> list[dict]:
    """
    Compute RMSE, MAE, NSE, and MBE individually for every node.

    Returns a list of dicts (one per node) in node-index order:
        [{"node_idx": 0, "rmse": ..., "mae": ..., "nse": ...,
          "mbe": ..., "n_valid": ...}, ...]

    Nodes with fewer than 2 valid observations receive NaN metrics.
    MBE > 0 means the model over-predicts; MBE < 0 means under-prediction.
    """
    pred   = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    mask   = mask.detach().cpu()
    N    = pred.shape[2]
    rows = []

    for n in range(N):
        m = mask[:, :, n].bool()
        p = pred[:, :, n][m].cpu().float()
        t = target[:, :, n][m].cpu().float()

        if len(t) < 2:
            rows.append({"node_idx": n, "rmse": np.nan, "mae": np.nan,
                         "nse": np.nan, "mbe": np.nan, "n_valid": 0})
            continue

        rmse   = ((p - t) ** 2).mean().sqrt().item()
        mae    = (p - t).abs().mean().item()
        mbe    = (p - t).mean().item()
        ss_res = ((p - t) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        nse    = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()

        rows.append({
            "node_idx": n,
            "rmse":    round(rmse, 6),
            "mae":     round(mae,  6),
            "nse":     round(nse,  6),
            "mbe":     round(mbe,  6),
            "n_valid": int(m.sum()),
        })

    return rows


def compute_per_step_metrics(pred: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor) -> list[dict]:
    """
    Compute RMSE, MAE, and NSE at each individual forecast step h,
    averaged across all nodes.

    Tensors shape: [B, T_out, N]

    Returns a list of T_out dicts, one per step:
        [{"step": 1, "lead_min": 15, "rmse": ..., "mae": ..., "nse": ...}, ...]

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
    pred   = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    mask   = mask.detach().cpu()
    T_out   = pred.shape[1]
    N       = pred.shape[2]
    results = []

    for h in range(T_out):
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
            "step":     h + 1,                                         # 1-indexed
            "lead_min": (h + 1) * 15,                                  # minutes ahead
            "rmse": float(np.mean(rmse_list)) if rmse_list else float("nan"),
            "mae":  float(np.mean(mae_list))  if mae_list  else float("nan"),
            "nse":  float(np.mean(nse_list))  if nse_list  else float("nan"),
        })

    return results
