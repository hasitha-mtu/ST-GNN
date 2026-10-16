"""Single datum convention for normalised stage and every elevation gate."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import GRAPH_SRC, ROOT
from utils.datum import (stage_reference, normalise_stage, reconstruct_stage,
                         abs_datum_valid)


def test_forward_inverse_roundtrip_all_nodes():
    nd = pd.read_csv(GRAPH_SRC / "nodes.csv")
    low, rng = stage_reference(nd)
    h = low + rng * np.linspace(-1.5, 4.5, 50)[:, None]
    assert np.allclose(reconstruct_stage(normalise_stage(h, low, rng), low, rng), h)


def test_headraces_flagged_invalid_by_default():
    nd = pd.read_csv(GRAPH_SRC / "nodes.csv")
    v = abs_datum_valid(nd)
    refs = nd.ref.astype(str).to_numpy()
    assert set(refs[~v]) == {"19094", "19095"}


@pytest.mark.parametrize("script", ["train_st_gnn_hand_edge.py", "train_dfc_gnn_unified.py",
                                    "train_st_gnn_soil_gate.py", "train_st_gnn_backwater_edge.py",
                                    "build_dataset.py"])
def test_no_private_datum_conventions_left(script):
    src = (ROOT / "src" / script).read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "+ 1.5   # ~1.5 m default range" not in code
    assert 'gauge_datum[reservoir_mask]' not in code
    assert not re.search(r'p90_mAOD"\]\s*-\s*nodes_df\["gauge_datum_mOSGM15"\]', code)
    assert "utils.datum" in src


def _tiny_hand_model(valid):
    from models.st_gnn_hand_edge import STGNNHANDEdge
    N = 3
    return STGNNHANDEdge(
        f_dyn=11, f_static=7, f_edge=5,
        hand_src=torch.tensor([0, 1]), hand_dst=torch.tensor([1, 2]),
        hand_threshold=torch.tensor([1.0, 1.0]), hand_overland_dist=torch.tensor([1.0, 1.0]),
        z_saddle=torch.tensor([10.5, 10.5]), gauge_datum=torch.tensor([10.0, 10.0, 10.0]),
        stage_range=torch.tensor([1.0, 1.0, 1.0]), node_abs_valid=torch.tensor(valid),
        hidden=8, gat_heads=1, gru_layers=1, t_out=2)


def test_hand_gate_excludes_unverified_datum_endpoint():
    x_last = torch.zeros(1, 3, 11)
    x_last[0, 0, 1] = 3.0          # node 0 high: H = 13 > z_saddle
    m_ok = _tiny_hand_model([True, True, True])
    m_bad = _tiny_hand_model([False, True, True])
    _, a_ok = m_ok._hand_edge_attr(x_last)
    _, a_bad = m_bad._hand_edge_attr(x_last)
    assert a_ok[0, 0] > 0.9                       # edge 0-1 opened by node 0
    assert a_bad[0, 0] < 0.1                      # node 0 excluded -> only node 1 (H=10)
    assert torch.allclose(a_ok[0, 1], a_bad[0, 1])


def test_inference_and_training_graph_are_identical():
    from utils.train_utils import load_graph
    import run_inference
    ei1, ea1, na1 = load_graph(None, GRAPH_SRC, torch.device("cpu"))
    ei2, ea2, na2 = run_inference.load_graph(torch.device("cpu"))
    assert torch.equal(ei1, ei2) and torch.equal(ea1, ea2) and torch.allclose(na1, na2)
    nd = pd.read_csv(GRAPH_SRC / "nodes.csv")
    res = torch.tensor(nd.is_reservoir.to_numpy() == 1)
    touches = res[ei1[0]] | res[ei1[1]]
    assert torch.all(ea1[touches, 2] == 0)       # masked exactly as in training


def test_soilgate_bounded_threshold():
    from models.st_gnn_soil_gate import STGNNSoilGate
    kw = dict(f_dyn=11, f_static=7, f_edge=5, hand_src=torch.tensor([0]),
              hand_dst=torch.tensor([1]), hand_threshold=torch.tensor([1.0]),
              hand_overland_dist=torch.tensor([1.0]), z_saddle=torch.tensor([0.0]),
              gauge_datum=torch.zeros(2), stage_range=torch.ones(2), hidden=8,
              gat_heads=1, gru_layers=1, t_out=2)
    m = STGNNSoilGate(**kw, sat_threshold=0.8, sat_threshold_bounds=(0.7, 0.9))
    assert m.sat_threshold.item() == pytest.approx(0.8, abs=1e-4)
    with torch.no_grad():
        m.sat_threshold_raw.fill_(50.0)
    assert m.sat_threshold.item() <= 0.9 + 1e-6
    legacy = STGNNSoilGate(**kw, sat_threshold=0.75)
    assert legacy.sat_threshold.item() == pytest.approx(
        torch.sigmoid(legacy.sat_threshold_raw).item())
