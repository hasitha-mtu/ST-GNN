"""River-graph integrity (uphill reverse duplicates, DAG, routing order)."""
import json
import numpy as np
import pandas as pd

from conftest import GRAPH_SRC
from utils.graph_checks import (find_reverse_duplicates, river_edges, is_dag,
                                topological_edges_from)


def test_shipped_graph_is_a_dag_without_reverse_duplicates():
    ed = pd.read_csv(GRAPH_SRC / "edges.csv")
    assert len(ed) == 26
    assert find_reverse_duplicates(ed) == []
    assert is_dag(ed, 27)
    meta = json.load(open(GRAPH_SRC / "graph_metadata.json"))
    assert meta["num_edges"] == len(meta["edges"]) == 26


def test_reverse_duplicates_detected_on_submitted_graph():
    meta = json.load(open(GRAPH_SRC / "graph_metadata.json"))
    old = pd.DataFrame(meta["edges"] + meta["removed_edges_revision"])
    bad = old.loc[find_reverse_duplicates(old)]
    assert set(zip(bad.src_ref, bad.dst_ref)) == {("19057", "19056"), ("19058", "19059")}
    assert not is_dag(old, 27)
    assert is_dag(river_edges(old), 27)


def test_non_reservoir_non_tidal_edges_flow_downhill():
    nd = pd.read_csv(GRAPH_SRC / "nodes.csv"); ed = pd.read_csv(GRAPH_SRC / "edges.csv")
    special = (nd.is_reservoir == 1) | (nd.is_tidal == 1)
    ok = ~(special.to_numpy()[ed.src_idx] | special.to_numpy()[ed.dst_idx])
    assert (ed.loc[ok, "elev_drop_m"] > 0).all()


def test_topological_routing_sums_confluence_contributions():
    # 0 -> 1 -> 3 and 0 -> 2 -> 3 (diamond): node 3 must get BOTH paths
    ed = pd.DataFrame({"src_idx": [0, 0, 1, 2], "dst_idx": [1, 2, 3, 3],
                       "elev_drop_m": [1, 1, 1, 1]})
    order = topological_edges_from(ed, 0)
    assert set(order) == {(0, 1), (0, 2), (1, 3), (2, 3)}
    pos = {e: i for i, e in enumerate(order)}
    assert pos[(0, 1)] < pos[(1, 3)] and pos[(0, 2)] < pos[(2, 3)]
    from scenarios.scenario_generator import route_from_sources, GeneratorConfig
    D = np.zeros((30, 4)); D[0, 0] = 1.0
    lags = {"0_1": 1, "0_2": 1, "1_3": 1, "2_3": 1}
    out = route_from_sources(D, [0], ed, lags, GeneratorConfig(attenuation=0.5))
    assert out[2, 3] == np.float64(0.5 * 0.5 * 2)   # two paths, two hops each


def test_backwater_pairs_use_nearest_upstream_neighbour():
    import importlib.util, sys
    from conftest import ROOT
    spec = importlib.util.spec_from_file_location(
        "bw", ROOT / "src/data/precompute_backwater_edges.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    nd = pd.read_csv(GRAPH_SRC / "nodes.csv")
    meta = json.load(open(GRAPH_SRC / "graph_metadata.json"))
    old = pd.DataFrame(meta["edges"] + meta["removed_edges_revision"])
    pairs = {p["bridge_name"]: nd.loc[p["dst"], "name"] for p in m.find_bridge_backwater_pairs(nd, old)}
    assert pairs["Macroom Town Bridge"] == "Killaclug"         # not Ballymakera (41.6 km)
    assert "Glennamought Bridge" not in pairs
