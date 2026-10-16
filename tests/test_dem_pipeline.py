"""DEM conditioning / D8 used for HAND edges and Kirpich inputs."""
import numpy as np
import pandas as pd

import sys
from conftest import ROOT
sys.path.insert(0, str(ROOT / "src/data"))
import precompute_hand_edges as ph
from compute_kirpich_inputs import trace_main_stem_upstream, path_slopes


def _clipped_valley(H=40, W=60):
    """V-shaped valley draining east, CLIPPED: nodata frame around it (the
    situation of the project DEM), with a 2-m pit in the channel."""
    r, c = np.mgrid[0:H, 0:W]
    z = 100 - 1.0 * c + 2.0 * np.abs(r - H // 2)
    z[H // 2, 30] -= 2.0
    nan = np.zeros((H, W), bool); nan[:3] = nan[-3:] = True; nan[:, :3] = True; nan[:, -2:] = True
    z = z.astype(float); z[nan] = np.nan
    return z, nan


def test_fill_seeds_from_data_edge_and_drains_everything():
    z, nan = _clipped_valley()
    f = ph.fill_pits_priority_flood(z, nan)
    assert (f[~nan] > z[~nan] + 1e-9).any()          # the pit is filled (old code: nothing)
    dr, dc, acc = ph.compute_d8_and_accumulation(f, nan)
    n_valid = int((~nan).sum())
    # every valid cell drains to the outlet region: max accumulation ~ whole valley
    assert acc.max() > 0.3 * n_valid
    # strictly downhill directions only
    H, W = z.shape
    for r_, c_ in zip(*np.where(~nan & ((dr != 0) | (dc != 0)))):
        assert f[r_ + dr[r_, c_], c_ + dc[r_, c_]] < f[r_, c_]


def test_static_edges_read_from_src_idx_columns(tmp_path):
    pd.DataFrame({"src_idx": [0, 1], "dst_idx": [1, 2]}).to_csv(tmp_path / "edges.csv", index=False)
    ei = ph.load_static_edges(tmp_path)
    assert ei is not None and ei.shape == (2, 2)
    cand = ph.find_candidate_pairs(np.array([0., 100., 200.]), np.zeros(3), ei, 5.0)
    assert {(i, j) for i, j, _ in cand} == {(0, 2)}  # river-connected pairs excluded


def test_main_stem_trace_and_slope():
    z, nan = _clipped_valley()
    f = ph.fill_pits_priority_flood(z, nan)
    dr, dc, acc = ph.compute_d8_and_accumulation(f, nan)
    r0, c0 = z.shape[0] // 2, 55
    path = trace_main_stem_upstream(r0, c0, dr, dc, acc, nan, 30.0)
    s = path_slopes(path, z)
    assert s["L_m"] > 30 * 40 and 0 < s["S_10_85"] < 0.1
