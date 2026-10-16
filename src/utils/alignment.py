"""
alignment.py  —  Single source of truth for window / target indexing.

Data convention (build_dataset.py)
----------------------------------
    X[t, n, 0] = a(t)         stage anomaly observed at time t
    y[t, n]    = a(t + 1)     one-step-ahead stage anomaly
    target_mask[t, n] = 1     iff the observation a(t + 1) is real

For a window starting at s with T_in inputs, the forecast issue time is
t0 = s + T_in - 1 and the k-th lead (k = 1..T_out) is a(t0 + k) = y[t0 + k - 1].
Hence the target rows of the window are y[s + T_in - 1 + j], j = 0..T_out-1.

Revision audit finding: run_inference.compute_extended_metrics and
scenario_evaluator used y[s + T_in + T_out - 1] for the final lead, one
step beyond the forecast (lead T_out + 1). Every caller now uses these
helpers, and tests/test_alignment.py checks them against GPUSampler.
"""
from __future__ import annotations

import numpy as np


def issue_index(start: int | np.ndarray, t_in: int):
    """Index of the last input step (forecast issue time)."""
    return start + t_in - 1


def target_rows(start: int | np.ndarray, t_in: int, t_out: int) -> np.ndarray:
    """y-row indices for leads 1..T_out. Shape [T_out] or [B, T_out]."""
    start = np.asarray(start)
    j = np.arange(t_out)
    return (start[..., None] if start.ndim else start) + t_in - 1 + j


def last_target_row(start, t_in: int, t_out: int):
    """y-row index for the final lead T_out."""
    return np.asarray(start) + t_in + t_out - 2


def window_starts_within_blocks(n_rows: int, block_len: int,
                                t_in: int, t_out: int) -> np.ndarray:
    """Window starts that keep inputs AND targets inside one block.

    A block is one concatenated synthetic scenario window of length
    block_len. The last target row of a window starting at s is
    s + t_in + t_out - 2, which must be < block end. Windows that straddle
    two blocks (which the original evaluator included) mix two unrelated
    historical periods and are excluded here.
    """
    starts = []
    n_blocks = n_rows // block_len
    span = t_in + t_out - 1        # rows covered by inputs + targets
    for b in range(n_blocks):
        b0 = b * block_len
        last_start = b0 + block_len - span   # inclusive
        if last_start >= b0:
            starts.extend(range(b0, last_start + 1))
    return np.asarray(starts, dtype=np.int64)
