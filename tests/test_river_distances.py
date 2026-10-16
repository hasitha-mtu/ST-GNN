"""Along-channel reach lengths (A20), DFC river-network distances (A21) and
the flood-velocity bankfull filter (A22)."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
sys.path.insert(0, str(ROOT / "src/data"))
import precompute_hand_edges as ph
import compute_river_distances as crd
import compute_edge_features as cef
from test_dem_pipeline import _clipped_valley


def test_downstream_trace_length_on_synthetic_valley():
    z, nan = _clipped_valley()
    f = ph.fill_pits_priority_flood(z, nan)
    dr, dc, _ = ph.compute_d8_and_accumulation(f, nan)
    r = z.shape[0] // 2
    L, miss = crd.trace_downstream_length(r, 10, r, 40, dr, dc, 30.0)
    assert miss == 0.0
    assert L >= 30 * 30 - 1e-6                 # never shorter than the straight line
    assert L < 30 * 30 * 1.5


def test_choose_length_rules():
    assert crd.choose_length(12.0, 0.0, False, 10.0, 40.0) == (12.0, "d8")
    # missed the downstream gauge -> never shorter than the straight line
    assert crd.choose_length(0.0, 2600.0, False, 1.78, 1.07)[0] == pytest.approx(1.78)
    # both gauges on the same cell -> fallback
    assert crd.choose_length(0.15, 0.0, True, 0.86, 0.85)[0] == pytest.approx(0.86)
    # D8 shorter than straight line -> fallback
    assert crd.choose_length(5.0, 0.0, False, 6.0, 6.1)[1].startswith("fallback")


def test_project_reach_lengths_not_shorter_than_straight_line():
    ed = pd.read_csv(ROOT / "dataset/graph/edges.csv")
    nd = pd.read_csv(ROOT / "dataset/graph/nodes.csv")
    xy = nd[["easting_itm", "northing_itm"]].to_numpy(float)
    straight = np.linalg.norm(xy[ed.src_idx] - xy[ed.dst_idx], axis=1) / 1000
    assert (ed.river_dist_km.to_numpy() >= 0.98 * straight).all()
    assert (ed.river_dist_km.to_numpy() <= 3.0 * straight + 0.5).all()


def test_dfc_distances_follow_the_river_tree(tmp_path, monkeypatch):
    # path 0 - 1 - 2 along the river, but 0 and 2 are close in a straight line
    nd = pd.DataFrame({"ref": ["a", "b", "c"], "easting_itm": [0., 5000., 100.],
                       "northing_itm": [0., 0., 0.]})
    pd.DataFrame({"src_idx": [0, 1], "dst_idx": [1, 2], "src_ref": ["a", "b"],
                  "dst_ref": ["b", "c"], "river_dist_km": [6.0, 6.5]}
                 ).to_csv(tmp_path / "edges.csv", index=False)
    monkeypatch.setattr(cef, "GRAPH_DIR", tmp_path)
    G = cef.build_network_graph(nd)
    D = cef.all_pairs_river_distance(G, 3)
    assert D[0, 2] == pytest.approx(12.5) and D[2, 0] == pytest.approx(12.5)


def test_flood_velocity_uses_bankfull_stage_not_anomaly(tmp_path, monkeypatch):
    sd = tmp_path / "station_data" / "19001"
    sd.mkdir(parents=True)
    # low stages slow, high stages fast; bankfull stage 1.5 m, anomaly 0.5 m
    pd.DataFrame({"Stage": [0.6, 0.8, 1.0, 1.2, 1.6, 1.8, 2.0],
                  "Velocity": [0.2, 0.3, 0.3, 0.4, 1.1, 1.2, 1.3]}
                 ).to_csv(sd / "19001_Gaugings.csv", index=False)
    pd.DataFrame({"ref": ["19001"], "tier_used": [2], "bankfull_reading_m": [1.5],
                  "bankfull_anomaly_m": [0.5]}
                 ).to_csv(tmp_path / "bankfull_thresholds_report.csv", index=False)
    monkeypatch.setattr(cef, "STATION_DIR", tmp_path / "station_data")
    monkeypatch.setattr(cef, "GRAPH_DIR", tmp_path)
    nd = pd.DataFrame({"ref": ["19001"]})
    v = cef.extract_node_velocities(nd, np.array([0.5]))
    assert v[0] == pytest.approx(1.2)          # median of the 3 gaugings above 1.5 m
    # the submitted filter (stage > anomaly 0.5) would have taken all 7: 0.4


def test_pool_gauge_elevation_is_median_pool_level(tmp_path, monkeypatch):
    nd = pd.DataFrame({"ref": ["1", "2"], "gauge_datum_mOSGM15": [10.0, -2.7],
                       "stage_ref_low_m": [10.0, 58.6], "p50_mAOD_train": [10.5, 60.2],
                       "abs_datum_valid": [1, 1], "is_tidal": [0, 0],
                       "easting_itm": [540524.7, 540524.7],
                       "northing_itm": [572003.3, 572003.3]})
    if not cef.DEM_PATH.exists():
        pytest.skip("project DEM not available")
    monkeypatch.setattr(cef, "GRAPH_DIR", tmp_path)
    z = cef.extract_node_elevations(nd)
    assert z[0] == pytest.approx(10.0) and z[1] == pytest.approx(60.2)


def test_validator_flags_submitted_reach_lengths_and_knn_distances(tmp_path, monkeypatch):
    import shutil, subprocess
    import validate_pipeline as vp
    g = tmp_path / "graph"; g.mkdir()
    shutil.copy(ROOT / "dataset/graph/nodes.csv", g / "nodes.csv")
    ed = pd.read_csv(ROOT / "dataset/graph/edges.csv")
    ed["river_dist_km"] = ed["river_dist_km_submitted"]          # submitted lengths
    ed.to_csv(g / "edges.csv", index=False)
    old = ROOT / "old_params/edge_features_submitted.npz"         # KNN + HAND distances
    shutil.copy(old, g / "edge_features.npz")
    monkeypatch.setattr(vp, "GRAPH", g)
    vp.RESULTS.clear(); vp.chk_graph()
    st = {r["check"]: r["status"] for r in vp.RESULTS}
    assert st["graph.reach_lengths"] == "FAIL" and st["graph.dfc_river_distance"] == "FAIL"


def test_dfc_gnn_uses_four_edge_features_even_if_sar_present(tmp_path):
    import train_dfc_gnn
    from models.dfc_gnn import load_edge_features
    z = dict(np.load(ROOT / "old_params/edge_features_submitted.npz", allow_pickle=True))
    assert "sar_wetness_norm" in z                       # the submitted file had SAR
    np.savez(tmp_path / "ef.npz", **z)
    ef = load_edge_features(str(tmp_path / "ef.npz"), use_sar_edge=train_dfc_gnn.USE_SAR_EDGE)
    assert ef["edge_attr"].shape[1] == 4


def test_velocity_at_bankfull_fallback():
    st = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    v = 0.1 + 1.0 * st                                   # v = 0.1 + h
    vb, lab = cef.velocity_at_bankfull(st, v, 1.0)       # bankfull above every gauging
    assert lab.startswith("rating-fit") and vb == pytest.approx(1.1, abs=1e-6)
    # median of all gaugings (old fallback) would give 0.5 m/s
    vb, lab = cef.velocity_at_bankfull(st, v, 5.0)       # too far beyond the data
    assert lab.startswith("gaugings-topQ") and vb == pytest.approx(np.median(v[st >= np.quantile(st, .75)]))
    vb, lab = cef.velocity_at_bankfull(st, 1.0 - st, 1.0)  # velocity falls with stage
    assert lab.startswith("gaugings-topQ")
    assert cef.velocity_at_bankfull(st[:2], v[:2], 1.0)[0] is None
