"""Scenario evaluator: alignment, block isolation, metric definitions."""
import json

import numpy as np
import torch

from scenarios import scenario_evaluator as se


class FutureOracle(torch.nn.Module):
    """Reads the true future from extra input channels: channel k+1 holds
    a(t + k + 1). Predictions equal the targets iff alignment is right."""
    def __init__(self, t_out):
        super().__init__(); self.t_out = t_out
    def forward(self, x, na=None):
        last = x[:, -1]
        return torch.stack([last[:, :, k + 1] - last[:, :, 0] for k in range(self.t_out)], 1)


def _oracle_series(n_blocks=4, TW=104, N=3, t_out=6):
    T = n_blocks * TW
    rng = np.random.default_rng(0)
    a = np.zeros((T + t_out + 1, N), np.float32)
    for b in range(n_blocks):          # independent random walk per block
        a[b * TW:(b + 1) * TW] = np.cumsum(rng.normal(0, 1, (TW, N)), 0) + 100 * b
    X = np.zeros((T, N, t_out + 1), np.float32)
    X[:, :, 0] = a[:T]
    for b in range(n_blocks):
        for k in range(t_out):
            blk = a[b * TW:(b + 1) * TW]
            fut = np.full((TW, N), np.nan, np.float32)
            fut[:TW - k - 1] = blk[k + 1:]
            X[b * TW:(b + 1) * TW, :, k + 1] = np.nan_to_num(fut)
    y = np.zeros((T, N), np.float32)
    for b in range(n_blocks):
        blk = a[b * TW:(b + 1) * TW]
        y[b * TW:(b + 1) * TW - 1] = blk[1:]
        y[(b + 1) * TW - 1] = np.nan
    return X, y


def test_oracle_scores_perfectly_and_windows_stay_in_block():
    t_in, t_out = 32, 6
    X, y = _oracle_series(t_out=t_out)
    meta = {"T_per_window": 104, "window_input_onset_step": [40] * 4}
    ev = se.build_eval_arrays(X, y, np.ones_like(y), meta, t_in, t_out)
    pred = se.predict_windows.__wrapped__ if hasattr(se.predict_windows, "__wrapped__") else None
    m = FutureOracle(t_out)
    with torch.no_grad():
        x = torch.from_numpy(X[ev["starts"][:, None] + np.arange(t_in)[None]])
        p = (m(x) + x[:, -1, :, 0].unsqueeze(1)).numpy()
    fin = np.isfinite(ev["target"])
    assert np.allclose(p[fin], ev["target"][fin], atol=1e-4)
    assert se.nse_pooled(p, ev["target"], ev["mask"]) > 0.999999
    # every forecast's inputs and targets inside ONE block
    assert np.all(ev["starts"] // 104 == (ev["rows"][:, -1]) // 104)
    # observability split
    assert np.all(ev["observable"] == (ev["issue_local"] >= 40))


def test_old_off_by_one_would_fail():
    t_in, t_out = 32, 6
    X, y = _oracle_series(t_out=t_out)
    starts = np.arange(0, 104 - t_in - t_out + 1)
    old_rows = starts + t_in + t_out - 1                 # submitted convention
    m = FutureOracle(t_out)
    with torch.no_grad():
        x = torch.from_numpy(X[starts[:, None] + np.arange(t_in)[None]])
        p_last = (m(x)[:, -1] + x[:, -1, :, 0]).numpy()
    assert not np.allclose(p_last, y[old_rows], atol=1e-3)


def test_node_mean_metrics_match_training_definition():
    from utils.train_utils import compute_metrics
    rng = np.random.default_rng(1)
    t = rng.normal(size=(50, 4, 5)).astype(np.float32)
    p = (t + rng.normal(0, 0.3, t.shape)).astype(np.float32)
    m = (rng.random(t.shape) > 0.1)
    ref = compute_metrics(torch.from_numpy(p), torch.from_numpy(t), torch.from_numpy(m.astype(np.float32)))
    ours = se.node_mean_metrics(p, t, m)
    assert np.isclose(ours["rmse_node_mean"], ref["rmse"], rtol=1e-5)
    assert np.isclose(ours["nse_node_mean"], ref["nse"], rtol=1e-5)


def test_pod_far_respects_mask():
    p = np.array([[1.0, 0.0], [1.0, 1.0]]); t = np.array([[1.0, 1.0], [0.0, 1.0]])
    m = np.array([[1, 1], [0, 1]], bool)
    r = se.pod_far(p, t, m, np.array([0.5, 0.5])[None, :])
    assert (r["TP"], r["FP"], r["FN"]) == (2, 0, 1)


def test_s2_group_metrics_on_generated_scenario(generated):
    out, data = generated
    sdir = out / "S2_GaugeFailure"
    X = np.load(sdir / "X_synthetic.npy"); y = np.load(sdir / "y_synthetic.npy")
    m = np.load(sdir / "mask_synthetic.npy"); meta = json.load(open(sdir / "scenario_meta.json"))
    ev = se.build_eval_arrays(X, y, m, meta, 32, 4, np.asarray(data.input_valid))
    pred = ev["target"] + 0.01                     # constant-bias "model"
    res = se.scenario_specific(pred, ev, meta, sdir, 0.02, data.bankfull)
    assert np.isclose(res["s2_rmse_failed_nodes"], 0.01, atol=1e-4)
    assert res["s2_degradation_failed_nodes"] == 0.5
