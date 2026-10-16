"""Window / target alignment shared by training, inference and scenarios."""
import numpy as np
import torch

from utils.alignment import target_rows, last_target_row, window_starts_within_blocks
from utils.gpu_sampler import GPUSampler


def _toy(T=400, N=3):
    a = np.arange(T, dtype=np.float32)[:, None] * np.ones((1, N), np.float32)
    X = np.zeros((T, N, 2), np.float32); X[:, :, 0] = a
    y = np.roll(a, -1, axis=0); y[-1] = np.nan
    return X, y


def test_sampler_targets_are_leads_1_to_Tout():
    X, y = _toy()
    m = np.ones_like(y)
    s = GPUSampler(X, y, m, np.array([0, 10, 50]), 32, 6, 3, "cpu", False)
    xb, yb, _ = next(iter(s))
    issue = xb[:, -1, :, 0]                             # a(t0)
    leads = yb - issue.unsqueeze(1)
    assert torch.allclose(leads[:, :, 0], torch.arange(1, 7, dtype=torch.float32).expand(3, 6))
    rows = target_rows(np.array([0, 10, 50]), 32, 6)
    assert np.allclose(yb.numpy()[:, :, 0], y[rows][:, :, 0])
    assert np.all(last_target_row(np.array([0, 10, 50]), 32, 6) == rows[:, -1])


def test_sampler_enforces_anchor_validity():
    X, y = _toy()
    m = np.ones_like(y); iv = np.ones_like(y)
    iv[10 + 31, 1] = 0                                 # anchor of window 10, node 1
    s = GPUSampler(X, y, m, np.array([10]), 32, 4, 1, "cpu", False, input_valid=iv)
    _, _, mk = next(iter(s))
    assert mk[0, :, 1].sum() == 0 and mk[0, :, 0].sum() == 4


def test_block_windows_never_cross_blocks():
    st = window_starts_within_blocks(104 * 5, 104, 32, 48)
    assert len(st) == 5 * (104 - (32 + 48 - 2))     # last target row = s+T_in+T_out-2
    last = st + 32 + 48 - 2
    assert np.all(st // 104 == last // 104)


def test_run_inference_test_loader_matches_gpu_sampler(mock_project, monkeypatch):
    import run_inference
    from utils.gpu_sampler import make_gpu_loaders
    monkeypatch.setattr(run_inference, "PROC_DIR", mock_project / "dataset/processed")
    p = mock_project / "dataset/processed"
    X, y = np.load(p / "X.npy"), np.load(p / "y.npy")
    vm, iv = np.load(p / "valid_mask.npy"), np.load(p / "input_valid_mask.npy")
    _, _, test = make_gpu_loaders(X, y, vm, 32, 12, 64, "cpu", input_valid=iv)
    dl, T_in, T_out, ts, te = run_inference.test_dataloader({"t_in": 32, "t_out": 12}, 64)
    assert ts == int(test.indices[0]) and te - ts == len(test.indices)
    xa, ya, ma = next(iter(test))          # test sampler batch = 2 x 64
    xb, yb, mb = next(iter(dl))
    n = min(len(xa), len(xb))
    assert torch.allclose(xa[:n], xb[:n]) and torch.allclose(ya[:n], yb[:n], equal_nan=True)
    assert torch.equal(ma[:n], mb[:n])
