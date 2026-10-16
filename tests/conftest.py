"""
Shared fixtures for the revision-audit validation suite.

The suite runs WITHOUT the real OPW / ERA5 data: it builds a small mock
project (the real 27-node graph files + a synthetic 15-minute record with
the same array conventions as build_dataset.py) and exercises the actual
pipeline code against it.

Conventions reproduced from build_dataset.py:
    X[t, n, 0] = a(t)              stage anomaly
    X[t, n, 1] = normalised stage  (utils.datum.normalise_stage)
    y[t, n]    = a(t + 1)
    valid_mask[t]       = obs at t+1 real
    input_valid_mask[t] = obs at t real
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

GRAPH_SRC = ROOT / "dataset/graph"


def make_mock_arrays(nodes: pd.DataFrame, T: int = 16000, seed: int = 1):
    from utils.datum import stage_reference, normalise_stage
    rng = np.random.default_rng(seed)
    N = len(nodes)
    low, rng_ = stage_reference(nodes)
    # absolute stage: base level + slow seasonal + events
    t = np.arange(T)
    base = (low + 0.3 * rng_)[None, :]
    seasonal = 0.03 * np.sin(2 * np.pi * t / (96 * 30))[:, None]
    noise = np.cumsum(rng.normal(0, 0.0008, (T, N)), axis=0)
    noise -= pd.DataFrame(noise).rolling(96 * 7, min_periods=1).mean().to_numpy()
    rain = np.where(rng.random((T, 1)) < 0.02, rng.gamma(1.5, 1.5, (T, 1)), 0.0) * np.ones((1, N))
    # a few flood events in the test period (needed by S2's flood-window search)
    events = np.zeros((T, N))
    for c in range(int(T * 0.86), T - 300, 700):
        events += 0.8 * np.exp(-0.5 * ((t[:, None] - c - 60) / 10.0) ** 2)
    h = base + seasonal + noise + events
    a = h - pd.DataFrame(h).rolling(96 * 7, min_periods=1).mean().to_numpy()
    X = np.zeros((T, N, 11), np.float32)
    X[:, :, 0] = a
    X[:, :, 1] = normalise_stage(h, low, rng_)
    X[1:, :, 2] = 4 * np.diff(h, axis=0)
    X[:, :, 3] = np.log1p(np.abs(rng.normal(2, 0.5, (T, N))))
    X[:, :, 4] = rain
    sat2 = np.clip(0.75 + 0.1 * np.sin(2 * np.pi * t / (96 * 60)), 0, 1)[:, None] * np.ones((1, N))
    X[:, :, 9] = sat2
    X[:, :, 8] = sat2 * 0.472
    X[:, :, 6] = np.clip(sat2 - 0.03, 0, 1)
    X[:, :, 5] = X[:, :, 6] * 0.472
    y = np.roll(a, -1, axis=0).astype(np.float32)
    y[-1] = np.nan
    iv = np.ones((T, N), np.float32)
    iv[rng.random((T, N)) < 0.002] = 0
    vm = np.zeros_like(iv)
    vm[:-1] = iv[1:]
    return X, y, vm, iv, h.astype(np.float32)


@pytest.fixture(scope="session")
def mock_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("mockproj")
    g = root / "dataset/graph"; g.mkdir(parents=True)
    p = root / "dataset/processed"; p.mkdir(parents=True)
    for f in ("nodes.csv", "edges.csv", "graph_metadata.json"):
        shutil.copy(GRAPH_SRC / f, g / f)
    nodes = pd.read_csv(g / "nodes.csv")
    X, y, vm, iv, h = make_mock_arrays(nodes)
    np.save(p / "X.npy", X); np.save(p / "y.npy", y)
    np.save(p / "valid_mask.npy", vm); np.save(p / "input_valid_mask.npy", iv)
    np.save(p / "stage_abs.npy", h)
    bf = {str(r): 0.5 for r in nodes.ref}
    json.dump({"thresholds": bf}, open(g / "bankfull_thresholds.json", "w"))
    # response-kernel parameters with the revised calibration code
    import scenarios.unit_hydrograph as uh
    old = (uh.PROC_DIR, uh.GRAPH_DIR, uh.OUT_DIR)
    uh.PROC_DIR, uh.GRAPH_DIR, uh.OUT_DIR = p, g, g
    try:
        uh.main(validate=False)
    finally:
        uh.PROC_DIR, uh.GRAPH_DIR, uh.OUT_DIR = old
    return root


@pytest.fixture(scope="session")
def generated(mock_project):
    """Run every scenario of the revised generator on the mock project."""
    import scenarios.scenario_generator as sg
    old = (sg.PROC_DIR, sg.GRAPH_DIR)
    sg.PROC_DIR = mock_project / "dataset/processed"
    sg.GRAPH_DIR = mock_project / "dataset/graph"
    out = mock_project / "dataset/scenarios"
    try:
        data = sg.load_project_data()
        cfg = sg.GeneratorConfig(seed=3)
        for sid, fn in sg.SCENARIO_MAP.items():
            fn(data, cfg, out, n_windows=6, n_realizations=3,
               n_windows_per_realization=2, n_random_repeats=1)
    finally:
        sg.PROC_DIR, sg.GRAPH_DIR = old
    return out, data
