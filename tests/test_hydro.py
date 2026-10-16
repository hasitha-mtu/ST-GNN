"""Kirpich units, SCS kernel and routing (reviewer Priority 1)."""
import numpy as np
import pandas as pd
import pytest

from scenarios.hydro import (kirpich_tc_minutes, kirpich_tc_hours, scs_triangle_times,
                             unit_triangle_kernel, stage_response, route_delta, STEP_HR)


def test_kirpich_metric_form_returns_minutes():
    # Hand calculation: 0.0195 * 1000^0.77 * 0.01^-0.385 = 23.45 min
    assert kirpich_tc_minutes(1000.0, 0.01) == pytest.approx(23.45, rel=2e-3)
    # Cross-check against the imperial form (0.0078 L_ft^0.77 S^-0.385, minutes)
    imperial = 0.0078 * (1000.0 / 0.3048) ** 0.77 * 0.01 ** -0.385
    assert kirpich_tc_minutes(1000.0, 0.01) == pytest.approx(imperial, rel=0.01)
    assert kirpich_tc_hours(1000.0, 0.01) == pytest.approx(23.45 / 60, rel=2e-3)


def test_scs_standard_base_time():
    tp, tb = scs_triangle_times(2.0)
    assert tp == pytest.approx(STEP_HR / 2 + 1.2)
    assert tb == pytest.approx(2.67 * tp)
    _, tb_legacy = scs_triangle_times(2.0, base_factor_on="tc")
    assert tb_legacy == pytest.approx(2.67 * 2.0)


def test_kernel_peak_at_tp_and_unit_height():
    k = unit_triangle_kernel(2.0, 200)
    tp, tb = scs_triangle_times(2.0)
    assert k.max() == pytest.approx(1.0)
    assert abs((np.argmax(k) + 0.5) * STEP_HR - tp) <= STEP_HR
    assert k[int(np.ceil(tb / STEP_HR)) + 1:].max() == 0.0


def test_response_starts_with_rain_and_matches_calibration_for_short_pulse():
    rain = np.zeros(200); rain[10] = 20.0         # single 15-min burst
    r = stage_response(rain, 0.01, 4.0)
    assert np.all(r[:10] == 0) and r[10] > 0      # no extra Tp delay (G4)
    assert r.max() == pytest.approx(0.01 * 20.0, rel=0.02)
    tp, _ = scs_triangle_times(4.0)
    assert abs((np.argmax(r) - 10 + 0.5) * STEP_HR - tp) <= STEP_HR


def test_route_delta_lag_and_attenuation():
    d = np.zeros(20); d[3] = 1.0
    r = route_delta(d, 4, 0.75)
    assert r[7] == pytest.approx(0.75) and r.sum() == pytest.approx(0.75)


def test_unit_hydrograph_fallback_slope_not_bankfull_depth():
    """With the submitted graph the old proxy put 20/27 nodes on the 8 h
    ceiling; the revised fallback must not be bound-dominated for the
    ordinary (non-reservoir, non-tidal, non-main-stem) gauges."""
    from scenarios.unit_hydrograph import compute_tc_kirpich, node_area_km2
    from conftest import GRAPH_SRC
    nd = pd.read_csv(GRAPH_SRC / "nodes.csv"); ed = pd.read_csv(GRAPH_SRC / "edges.csv")
    tc = compute_tc_kirpich(nd, ed)
    ordinary = (nd.is_reservoir == 0) & (nd.is_tidal == 0) & (node_area_km2(nd) < 300)
    assert tc.loc[ordinary.to_numpy(), "tc_clipped"].sum() == 0
    # small urban headwater must respond in about an hour, not 4+ hours
    assert tc.loc[0, "tc_hr"] < 1.5          # Ballincolly, 3.7 km2
    assert np.allclose(node_area_km2(nd)[0], 3.65, atol=0.01)   # log(A+1) undone
