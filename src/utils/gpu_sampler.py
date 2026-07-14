"""
gpu_sampler.py  —  GPU-resident window sampler for ST-GNN training
════════════════════════════════════════════════════════════════════
Replaces the CPU DataLoader + per-batch .to(DEVICE) pattern.

Why this is faster
───────────────────
The current pipeline:
  1. DataLoader spawns num_workers processes
  2. Each worker slices a numpy array on CPU
  3. Python pickles the slice and sends it over IPC
  4. Main process unpickles and calls .to(DEVICE) → PCIe transfer
  5. GPU computes loss (very fast for 27 nodes, ~700 edges)
  6. Repeat — the GPU is idle most of the time waiting for step 4

For a graph with 27 nodes the CPU-to-GPU bottleneck dominates
completely.  The GPU processes each batch in microseconds but
waits milliseconds for the next one to arrive.

GPUSampler loads X, y, mask once (~143 MB total, <2.5% of a
6 GB GPU) and serves every batch with a single vectorised
gather operation — one GPU kernel call, zero PCIe transfers
per batch, zero Python→GPU synchronisations.

The proposed GPUSimpleSampler in the review is WRONG
──────────────────────────────────────────────────────
```python
# THIS IS SLOW — proposed implementation
x_seq = torch.stack([self.X[i : i + self.t_in] for i in batch_idx])
```
`batch_idx` is a GPU tensor.  Iterating over it with `for i in batch_idx`
calls `.item()` implicitly on each element, triggering a GPU→CPU
synchronisation on every iteration.  With batch_size=32 that is 32
synchronisations per batch — worse than the DataLoader it was meant
to replace.

The correct implementation below uses advanced indexing:
```python
t_idx = batch_idx.unsqueeze(1) + self._tin_range   # [B, T_in]
x_seq = self.X[t_idx]                              # [B, T_in, N, F]
```
This is a single GPU gather kernel with no Python loop.

Usage
──────
  from gpu_sampler import GPUSampler, make_gpu_loaders

  # Build once (loads X/y/mask to GPU)
  train_ldr, val_ldr, test_ldr = make_gpu_loaders(
      X, y, valid_mask,
      t_in=32, t_out=4,
      train_frac=0.70, val_frac=0.15,
      batch_size=BATCH_SIZE,
      device=DEVICE,
  )

  # Use exactly like a DataLoader — no .to(DEVICE) needed
  for x_seq, y_seq, mask in train_ldr:
      ...  # tensors already on DEVICE
"""

from __future__ import annotations

import numpy as np
import torch


class GPUSampler:
    """
    Window sampler that serves batches directly from GPU memory.

    Parameters
    ----------
    X         : float32 array [T, N, F_dyn]  — dynamic features
    y         : float32 array [T, N]          — stage anomaly targets
    mask      : bool    array [T, N]          — validity mask
    indices   : int array [n_windows]         — window start indices
    t_in      : int  — input sequence length (steps)
    t_out     : int  — forecast horizon (steps)
    batch_size: int
    device    : torch.device
    shuffle   : bool — randomise window order each epoch
    """

    def __init__(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        mask:       np.ndarray,
        indices:    np.ndarray,
        t_in:       int,
        t_out:      int,
        batch_size: int,
        device:     torch.device,
        shuffle:    bool = True,
    ):
        # ── Pre-load entire dataset to GPU once ───────────────────────
        self.X    = torch.from_numpy(X).float().to(device)       # [T, N, F]
        self.y    = torch.from_numpy(y).float().to(device)       # [T, N]
        self.mask = torch.from_numpy(mask.astype(bool)).bool().to(device)  # [T, N]

        # Window start indices on GPU — shuffled each epoch
        self.indices    = torch.tensor(indices, dtype=torch.long, device=device)
        self.t_in       = t_in
        self.t_out      = t_out
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.device     = device

        # Pre-compute offset ranges once — reused every batch
        # Storing on GPU avoids re-allocating torch.arange every __next__
        self._tin_range  = torch.arange(t_in,  device=device)   # [T_in]
        self._tout_range = torch.arange(t_out, device=device)   # [T_out]

        self._idx = 0

    # ── Iterator protocol ─────────────────────────────────────────────

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "GPUSampler":
        if self.shuffle:
            perm          = torch.randperm(len(self.indices), device=self.device)
            self.indices  = self.indices[perm]
        self._idx = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._idx >= len(self.indices):
            raise StopIteration

        # ── Sample batch indices ──────────────────────────────────────
        batch_idx = self.indices[self._idx : self._idx + self.batch_size]
        self._idx += self.batch_size

        # ── Vectorised gather — zero Python loops, zero PCIe ─────────
        # Input:  X[t : t+T_in]  for each t in batch_idx
        # t_idx shape:  [B, T_in]  — each row is a contiguous index range
        t_idx = batch_idx.unsqueeze(1) + self._tin_range          # [B, T_in]
        x_seq = self.X[t_idx]                                     # [B, T_in, N, F]

        # Target: y[t+T_in-1 : t+T_in-1+T_out]  for each t in batch_idx
        # MUST match LeeFloodDataset.__getitem__ which uses:
        #   start = idx + t_in - 1
        # because y.npy is forward-shifted by 1 step during preprocessing
        # (y[t] = stage anomaly at t+1, per the NOTE in LeeFloodDataset).
        # Using t_in instead of t_in-1 samples a different (invalid) position
        # in valid_mask for most nodes, causing cat_mask to be all-zero
        # for those nodes and producing NaN in compute_metrics.
        tgt_idx  = batch_idx.unsqueeze(1) + self.t_in - 1 + self._tout_range  # [B, T_out]
        y_seq    = self.y[tgt_idx]                                 # [B, T_out, N]
        mask_seq = self.mask[tgt_idx].float()                     # [B, T_out, N]

        return x_seq, y_seq, mask_seq


# ── Factory function ──────────────────────────────────────────────────────────

def make_gpu_loaders(
    X:          np.ndarray,
    y:          np.ndarray,
    valid_mask: np.ndarray,
    t_in:       int,
    t_out:      int,
    batch_size: int,
    device:     torch.device,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple[GPUSampler, GPUSampler, GPUSampler]:
    """
    Build train/val/test GPU samplers with the same 70/15/15 split
    used by the existing make_dataset() / DataLoader pipeline.

    Drop-in replacement for the DataLoader construction section in
    each training script.  Tensors yielded by the returned samplers
    are already on `device`; remove all .to(DEVICE) calls from the
    train_epoch and eval_epoch loops after switching.

    Parameters
    ----------
    X, y, valid_mask : raw numpy arrays from PROC_DIR
    t_in, t_out      : sequence lengths (must match model hparams)
    batch_size       : training batch size (val/test use 2× for speed)
    device           : target GPU device
    train_frac       : fraction of T assigned to training (default 0.70)
    val_frac         : fraction of T assigned to validation (default 0.15)
                       test_frac = 1 - train_frac - val_frac (default 0.15)

    Returns
    -------
    train_loader, val_loader, test_loader : GPUSampler instances
    """
    T = X.shape[0]

    # Split boundaries (same logic as existing make_splits())
    t1 = int(T * train_frac)
    t2 = int(T * (train_frac + val_frac))

    # Window start indices for each split
    # Valid range: [0, split_end - t_in - t_out] so last target fits
    train_idx = np.arange(0,  t1 - t_in - t_out + 1, dtype=np.int64)
    val_idx   = np.arange(t1, t2 - t_in - t_out + 1, dtype=np.int64)
    test_idx  = np.arange(t2, T  - t_in - t_out + 1, dtype=np.int64)

    print(f"  GPUSampler — windows: train={len(train_idx):,}  "
          f"val={len(val_idx):,}  test={len(test_idx):,}")
    print(f"  GPU memory: X={X.nbytes/1e6:.0f} MB  "
          f"y={y.nbytes/1e6:.0f} MB  "
          f"mask={valid_mask.nbytes/1e6:.0f} MB  "
          f"total={( X.nbytes + y.nbytes + valid_mask.nbytes)/1e6:.0f} MB")

    train_loader = GPUSampler(X, y, valid_mask, train_idx,
                              t_in, t_out, batch_size,     device, shuffle=True)
    val_loader   = GPUSampler(X, y, valid_mask, val_idx,
                              t_in, t_out, batch_size * 2, device, shuffle=False)
    test_loader  = GPUSampler(X, y, valid_mask, test_idx,
                              t_in, t_out, batch_size * 2, device, shuffle=False)
    return train_loader, val_loader, test_loader
