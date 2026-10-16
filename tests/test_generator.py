"""Integrity of the revised synthetic scenario generator (G1-G14)."""
import json

import numpy as np
import pytest

from utils.datum import reconstruct_stage
from utils.graph_checks import downstream_closure

SCEN = ["S1_ConvectiveCell", "S2_GaugeFailure", "S3_InniscarraRelease",
        "S4_SatBreakthrough", "S5_SpatialGradient", "S6_ChannelBlockage"]
TW = 104


def _load(out, s):
    d = out / s
    return (np.load(d / "X_synthetic.npy"), np.load(d / "y_synthetic.npy"),
            np.load(d / "mask_synthetic.npy"), json.load(open(d / "scenario_meta.json")))


def _base(data, meta, w):
    t0 = meta["window_base_start_row"][w]
    return (np.asarray(data.X[t0:t0 + TW]), np.asarray(data.y[t0:t0 + TW]),
            np.asarray(data.mask[t0:t0 + TW]))


@pytest.mark.parametrize("s", SCEN)
def test_targets_are_next_step_inputs(generated, s):
    """G1: y[t] == X[t+1, stage] inside every block (base data satisfies it)."""
    out, _ = generated
    X, y, _, meta = _load(out, s)
    fm = (np.load(out / s / "fail_mask_synthetic.npy").astype(bool)
          if s == "S2_GaugeFailure" else np.zeros(y.shape, bool))
    for w in range(meta["n_windows"]):
        xs, ys = X[w * TW:(w + 1) * TW, :, 0], y[w * TW:(w + 1) * TW]
        ok = ~fm[w * TW + 1:(w + 1) * TW]      # S2: failed inputs are frozen by design
        assert np.allclose(ys[:-1][ok], xs[1:][ok], atol=1e-5), (s, w)


@pytest.mark.parametrize("s", SCEN)
def test_base_windows_from_test_period_and_non_overlapping(generated, s):
    out, data = generated
    meta = _load(out, s)[3]
    st = sorted(set(meta["window_base_start_row"]))
    assert min(st) >= data.test_start
    if s != "S2_GaugeFailure":       # S2 may fall back to half-overlap with a warning
        assert all(b - a >= TW for a, b in zip(st, st[1:]))


@pytest.mark.parametrize("s", ["S1_ConvectiveCell", "S3_InniscarraRelease",
                               "S4_SatBreakthrough", "S5_SpatialGradient", "S6_ChannelBlockage"])
def test_normalised_stage_and_dhdt_consistent_with_stage(generated, s):
    """G3: the injected delta appears identically in stage anomaly,
    reconstructed absolute stage (normalised stage) and dh/dt."""
    out, data = generated
    X, y, _, meta = _load(out, s)
    for w in range(meta["n_windows"]):
        xb = _base(data, meta, w)[0]
        xs = X[w * TW:(w + 1) * TW]
        dA = xs[:, :, 0] - xb[:, :, 0]
        dH = (reconstruct_stage(xs[:, :, 1], data.stage_low, data.stage_range)
              - reconstruct_stage(xb[:, :, 1], data.stage_low, data.stage_range))
        unclipped = (xs[:, :, 1] < 4.99) & (xs[:, :, 1] > -1.99) & \
                    (xb[:, :, 1] < 4.99) & (xb[:, :, 1] > -1.99)
        assert np.allclose(dA[unclipped], dH[unclipped], atol=1e-4), (s, w)
        ddh = xs[1:, :, 2] - xb[1:, :, 2]
        ok = unclipped[1:] & unclipped[:-1]
        assert np.allclose(ddh[ok], 4 * np.diff(dA, axis=0)[ok], atol=1e-3)


def test_s2_failure_freezes_all_gauge_features_and_keeps_targets(generated):
    out, data = generated
    X, y, m, meta = _load(out, "S2_GaugeFailure")
    fm = np.load(out / "S2_GaugeFailure/fail_mask_synthetic.npy")
    reals = {r["id"]: r for r in meta["realizations"]}
    tf = meta["t_failure_step"]
    for w in range(meta["n_windows"]):
        xb, yb, mb = _base(data, meta, w)
        r = reals[meta["window_realization_id"][w]]
        fn, te = r["fail_nodes"], min(tf + r["duration_steps"], TW)
        xs = X[w * TW:(w + 1) * TW]
        for f in (0, 1, 3):          # stage anomaly, NORMALISED STAGE, discharge
            assert np.allclose(xs[tf:te, fn, f], xb[tf - 1, fn, f][None, :]), f
        assert np.all(xs[tf:te, fn, 2] == 0)
        assert np.allclose(xs[:, :, 4], xb[:, :, 4])          # rainfall untouched
        assert np.allclose(m[w * TW:(w + 1) * TW], mb)         # targets still scored
        assert np.allclose(y[w * TW:(w + 1) * TW], yb, equal_nan=True)
        assert fm[w * TW + tf:w * TW + te, fn].all()
        assert set(r["connected_downstream_nodes"]).isdisjoint(fn)


def test_s3_superimposed_and_upstream_untouched(generated):
    out, data = generated
    X, y, _, meta = _load(out, "S3_InniscarraRelease")
    tail = meta["release_node"]["idx"]
    affected = {tail} | downstream_closure(data.ed, tail)
    others = [n for n in range(X.shape[1]) if n not in affected]
    for w in range(meta["n_windows"]):
        xb = _base(data, meta, w)[0]
        xs = X[w * TW:(w + 1) * TW]
        assert np.allclose(xs[:, others], xb[:, others])        # incl. headraces
        d = xs[:, tail, 0] - xb[:, tail, 0]
        assert np.allclose(d[:meta["t_release_step"]], 0)
        assert d.max() > 0.3                                   # release present
        # superimposed: real fluctuation still there before AND after onset
        assert not np.allclose(xs[:, tail, 0], xs[0, tail, 0])


def test_s4_response_has_no_drop_at_breakthrough(generated):
    """G8: injected stage delta is non-negative and does not step down when
    the synthetic saturation crosses the threshold."""
    out, data = generated
    X, y, _, meta = _load(out, "S4_SatBreakthrough")
    reals = {r["id"]: r for r in meta["realizations"]}
    for w in range(meta["n_windows"]):
        xb = _base(data, meta, w)[0]
        d = X[w * TW:(w + 1) * TW, :, 0] - xb[:, :, 0]
        assert d.min() > -1e-5
        bt = reals[meta["window_realization_id"][w]]["sat_breakthrough_step"]
        if 1 < bt < TW - 1:
            steps = np.diff(d, axis=0)                 # [TW-1, N]
            jump = steps[bt - 1]                       # d[bt] - d[bt-1]
            # no downward discontinuity at breakthrough (the submitted code
            # removed a 0.01*k "slow rise" there): the step at breakthrough
            # must be in line with its neighbours. (Comparing it with the
            # minimum over the whole window is too strict - on a smooth
            # recession the breakthrough step can be the steepest by 1e-4 m.)
            neigh = np.minimum(np.minimum(steps[bt - 2], steps[bt]), 0)
            assert np.all(jump >= neigh - np.maximum(1e-3, 0.1 * np.abs(neigh)))


def test_s6_superimposed_with_decaying_upstream_backwater(generated):
    out, data = generated
    X, y, _, meta = _load(out, "S6_ChannelBlockage")
    b, u, dn = (meta[k]["idx"] for k in ("blockage_node", "upstream_node", "downstream_node"))
    r0 = meta["realizations"][0]
    assert 0 <= r0["upstream_factor"] < 1
    for w in range(meta["n_windows"]):
        xb = _base(data, meta, w)[0]
        xs = X[w * TW:(w + 1) * TW]
        db, du, dd = (xs[:, n, 0] - xb[:, n, 0] for n in (b, u, dn))
        assert db.max() > 0.5 and dd.min() < -0.05
        assert du.max() <= db.max() * r0["upstream_factor"] + 1e-5


def test_routing_never_flows_uphill(generated):
    """G5: in S1 the Ballincolly and Glennamought headwaters receive only
    their own local response (no routed signal from Glen Park/Blackpool)."""
    out, data = generated
    X, y, _, meta = _load(out, "S1_ConvectiveCell")
    from scenarios.scenario_generator import route_from_sources, GeneratorConfig
    hw = meta["headwater_nodes"]
    D = np.zeros((TW + 1, X.shape[1])); D[:, hw] = 1.0
    routed = route_from_sources(D, hw, data.ed, data.lags, GeneratorConfig())
    assert np.all(routed[:, 0] == 0) and np.all(routed[:, 3] == 0)


@pytest.mark.parametrize("s", SCEN)
def test_onset_metadata_present(generated, s):
    out, _ = generated
    meta = _load(out, s)[3]
    assert len(meta["window_input_onset_step"]) == meta["n_windows"]
    assert meta["generator_config"]["attenuation"] == 0.75
    assert "clipped_frac_targets" in meta


def test_sensitivity_parameters_change_routing(mock_project):
    from scenarios.scenario_generator import route_from_sources, GeneratorConfig
    import pandas as pd
    from utils.graph_checks import river_edges
    ed = river_edges(pd.read_csv(mock_project / "dataset/graph/edges.csv"))
    D = np.zeros((300, 27)); D[5, 21] = 1.0
    lags = json.load(open(mock_project / "dataset/graph/routing_lags.json"))["lags"]
    a = route_from_sources(D, [21], ed, lags, GeneratorConfig(attenuation=0.65, lag_scale=1.25))
    b = route_from_sources(D, [21], ed, lags, GeneratorConfig(attenuation=0.85, lag_scale=0.75))
    assert a[:, 22].sum() == pytest.approx(0.65) and b[:, 22].sum() == pytest.approx(0.85)
    assert np.argmax(a[:, 22]) > np.argmax(b[:, 22])
