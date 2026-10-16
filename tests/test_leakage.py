"""Temporal-leakage fixes: causal ERA5 alignment, train-only climatology,
train-only static attributes (reviewer Section 3.2)."""
import numpy as np
import pandas as pd
import pytest


def test_causal_align_never_uses_future_values():
    from data.soil_moisture_features import causal_align
    hours = pd.date_range("2024-01-01", periods=48, freq="h")
    s = pd.Series(np.arange(48, dtype=float), index=hours)      # value = hour index
    tgt = pd.date_range("2024-01-01", periods=48 * 4, freq="15min")
    a = causal_align(s, tgt)
    hour_floor = (tgt - hours[0]) // pd.Timedelta("1h")
    assert np.all(a.to_numpy() == hour_floor.to_numpy())      # never the next hour
    # the submitted nearest-hour rule DID look ahead (e.g. 00:45 -> 01:00)
    old = s.reindex(tgt, method="nearest", tolerance="1h")
    assert (old.to_numpy() > hour_floor.to_numpy()).any()


def test_climatology_uses_training_period_only(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    from data.soil_moisture_features import compute_sm_features
    t = pd.date_range("2023-01-01", "2025-12-31 23:00", freq="6h")
    v = np.full((len(t), 2, 2), 0.30, np.float32)
    end = pd.Timestamp("2025-01-01")
    v[t >= end] = 0.45                                          # "test" period is wetter
    ds = xr.Dataset({"swvl1": (("time", "latitude", "longitude"), v),
                     "swvl2": (("time", "latitude", "longitude"), v)},
                    coords={"time": t, "latitude": [51.9, 51.8], "longitude": [-9.0, -8.9]})
    f = tmp_path / "sm.nc"; ds.to_netcdf(f)
    leaky = compute_sm_features(str(f), climatology_end=None)
    clean = compute_sm_features(str(f), climatology_end=end)
    train = clean.time < np.datetime64(end)
    assert np.allclose(clean["swvl2_anomaly"].sel(time=train).values, 0.0, atol=1e-6)
    assert not np.allclose(leaky["swvl2_anomaly"].sel(time=train).values, 0.0, atol=1e-3)


def test_static_attributes_ignore_post_training_extremes():
    from data.compute_static_attributes import train_only_attributes
    idx = pd.date_range("2023-01-01", "2025-12-31", freq="15min")
    wl = pd.Series(10.0 + 0.1 * np.sin(np.arange(len(idx)) / 500), index=idx)
    end = pd.Timestamp("2025-04-05")
    base = train_only_attributes(wl, end, idx[0], {"raw_offset_to_malin_m": 0.0,
                                 "datum_is_local_zero": False}, 9.0)
    wl2 = wl.copy(); wl2[wl2.index >= end] += 5.0                # huge test-period floods
    after = train_only_attributes(wl2, end, idx[0], {"raw_offset_to_malin_m": 0.0,
                                  "datum_is_local_zero": False}, 9.0)
    for k in ("p90_mAOD", "amax_med_mAOD", "p50_mAOD_train"):
        assert after[k] == base[k]
    local = train_only_attributes(wl, end, idx[0], {"raw_offset_to_malin_m": -2.7,
                                  "datum_is_local_zero": True}, 0.0)
    assert local["stage_ref_low_m"] == pytest.approx(
        (wl[wl.index < end] - 2.7).quantile(0.01))


def test_training_boundary_matches_loaders():
    from data.compute_static_attributes import training_end_timestamp
    from utils.train_utils import get_split_boundary
    idx = pd.date_range("2023-01-01", "2026-03-25", freq="15min")
    t1, _ = get_split_boundary(len(idx))
    assert training_end_timestamp("2023-01-01", "2026-03-25", "15min") == idx[t1]
