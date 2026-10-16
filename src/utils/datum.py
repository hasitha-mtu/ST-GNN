"""
datum.py  —  Single source of truth for stage normalisation and the
absolute water-surface reconstruction used by every elevation-based gate.

Why this module exists (revision audit, reviewer point 3.2)
-----------------------------------------------------------
Before this revision, four different conventions were in use:

  build_dataset.py            norm = (h - gauge_datum) / (p90 - gauge_datum)
  STGNNHANDEdge / PC-DFC-GNN  H = DEM_elev + norm * 1.5          (headraces)
  STGNNSoilGate               H = p90 + norm * (p90 - datum)     (ALL reservoir nodes)
  STGNNBackwaterEdge          H = datum + norm * (p90 - datum)

Only the last one inverts the forward normalisation. For the two headrace
gauges the HAND gate therefore saw an almost constant H, and the SoilGate
reconstruction was offset by one stage range even at the Inniscarra
Tailrace and Ovens Bridge, whose datums are valid. Every component now
calls the functions below, so the forward normalisation and the inverse
reconstruction are exact inverses by construction (tested in
tests/test_datum.py).

Conventions
-----------
stage_ref_low_m   Lower reference of the normalisation. Equal to the
                  station gauge datum for ordinary gauges. For gauges whose
                  published datum is a local gauge-zero (the two reservoir
                  headraces) it is replaced by a training-period low pool
                  level (see data/compute_static_attributes.py), so the
                  normalised-stage feature has a usable dynamic range.
stage_range_m     p90_mAOD - stage_ref_low_m (floored at MIN_STAGE_RANGE_M).
abs_datum_valid   1 if the reconstructed H is an absolute Malin-Head (OSGM15)
                  elevation that may be compared with DEM-derived terrain
                  elevations (z_saddle, node elevation); 0 otherwise.
                  Gates comparing H with terrain must be disabled for
                  edges touching a node with abs_datum_valid == 0.

Backward compatibility: if nodes.csv has not yet been regenerated with
the new columns, stage_ref_low_m falls back to gauge_datum_mOSGM15 and
abs_datum_valid falls back to 0 for the known local-zero headraces
(DEFAULT_LOCAL_ZERO_REFS) and 1 elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_STAGE_RANGE_M = 0.05
NORM_CLIP = (-2.0, 5.0)

# Gauges whose published gauge datum is a local gauge-zero rather than a
# physical channel/pool reference (graph_metadata.json datum_notes).
DEFAULT_LOCAL_ZERO_REFS = {"19094", "19095"}


def _refs(nodes_df: pd.DataFrame) -> np.ndarray:
    return nodes_df["ref"].astype(str).to_numpy()


def stage_reference(nodes_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (stage_ref_low_m [N], stage_range_m [N]) as float64 arrays."""
    if "stage_ref_low_m" in nodes_df.columns:
        low = nodes_df["stage_ref_low_m"].to_numpy(dtype=np.float64).copy()
    else:
        low = nodes_df["gauge_datum_mOSGM15"].to_numpy(dtype=np.float64).copy()
    p90 = nodes_df["p90_mAOD"].to_numpy(dtype=np.float64)
    rng = p90 - low
    # Tidal nodes carry p90 = 0 (null threshold) -> range must still be a
    # positive scale; use a unit range so the transform stays invertible.
    rng = np.where(np.isfinite(rng) & (rng > MIN_STAGE_RANGE_M), rng, 1.0)
    return low, rng


def abs_datum_valid(nodes_df: pd.DataFrame) -> np.ndarray:
    """Boolean [N]: may H be compared with absolute terrain elevations?"""
    if "abs_datum_valid" in nodes_df.columns:
        return nodes_df["abs_datum_valid"].to_numpy().astype(bool)
    refs = _refs(nodes_df)
    valid = np.array([r not in DEFAULT_LOCAL_ZERO_REFS for r in refs])
    # Tidal nodes have no p90 threshold, so the stage range is synthetic
    # and H is only approximately absolute; keep them comparable (their
    # datum is a genuine Malin-Head level).
    return valid


def normalise_stage(h: np.ndarray, low: np.ndarray, rng: np.ndarray,
                    clip: bool = True) -> np.ndarray:
    """Forward transform: absolute stage (m OD) -> normalised stage."""
    z = (h - low) / rng
    return np.clip(z, *NORM_CLIP) if clip else z


def reconstruct_stage(norm: np.ndarray, low: np.ndarray,
                      rng: np.ndarray) -> np.ndarray:
    """Inverse transform used by every gate: normalised stage -> H (m OD)."""
    return low + norm * rng


def torch_reference(nodes_df: pd.DataFrame):
    """(low, range, abs_valid) as float32 / bool torch tensors (CPU)."""
    import torch
    low, rng = stage_reference(nodes_df)
    valid = abs_datum_valid(nodes_df)
    return (torch.tensor(low, dtype=torch.float32),
            torch.tensor(rng, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.bool))


def bankfull_depth_proxy(nodes_df: pd.DataFrame) -> np.ndarray:
    """Depth proxy used for Manning routing: p90 - lower reference.

    Uses the same lower reference as the normalisation, so the two
    headrace gauges no longer produce a ~45-60 m "bankfull depth" from
    their local gauge-zero datum (manuscript Section 5.5).
    """
    low, rng = stage_reference(nodes_df)
    return rng
