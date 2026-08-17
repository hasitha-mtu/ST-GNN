"""
precompute_backwater_edges.py  —  Bridge/culvert constriction backwater edges
================================================================================
Builds a THIRD edge class (alongside the directed river edges from
graph_builder.py and the cross-tributary HAND edges from
precompute_hand_edges.py): reverse, gated pathways at bridge/culvert
locations, representing the real, well-documented gradually-varied-flow
backwater mechanism (Chow, "Open-Channel Hydraulics", 1959) — a
constriction (debris/ice blockage, undersized opening) raises water
surface elevation upstream, a genuinely reversed direction relative to
normal downstream flow.

Why this is a separate edge class, not just "make the river graph
bidirectional"
-----------------------------------------------------------------------
graph_builder.py's directed river edges are physically correct for
normal flow and should stay directed everywhere they currently are.
Backwater is a real but LOCALISED, CONDITIONAL phenomenon — it only
matters at actual constriction points (bridges, culverts), and only
when the constriction node's own stage is approaching capacity. A
blanket bidirectional river graph would inject an always-on, physically
incorrect prior everywhere for the sake of a mechanism that's only real
at a handful of specific locations. This script instead adds a small,
targeted set of reverse edges — one per bridge-named node with both an
upstream and a genuinely distinct downstream river neighbour (see the
exclusion logic below, needed because 2 of 28 edges in edges.csv already
form genuine bidirectional pairs — 0<->1, 2<->3 — most likely
representing a real braided/distributary reach; confirmed via
independently-calibrated, non-symmetric routing_lags.json entries for
each direction rather than a duplicate data-entry error) — each carrying
a per-edge activation reference stage, so a model can learn to gate this
pathway open only when constriction risk is actually plausible, the same
design pattern already used for STGNNDynEdge's discharge-based
conductance and STGNNHANDEdge's saddle-elevation activation.

Direction convention
---------------------
  src = the bridge/constriction node itself (its OWN stage indicates
        whether the opening is being pushed toward capacity)
  dst = its immediate upstream river neighbour (where backwater would
        be observed)
This matches the causal direction of the mechanism: high stage AT the
constriction is the condition that should gate the pathway open, and the
upstream node is what receives the backwater signal.

Static edge attributes (matching the existing 4-feature river/HAND
schema for edge_dim compatibility with GATConv):
  [0] river_dist_km      reused directly from the corresponding forward
                          (upstream->bridge) river edge — a symmetric
                          physical distance regardless of flow direction
  [1] area_ratio          INVERTED from the forward edge (1/area_ratio)
                          — this is the trap flagged in the bidirectional-
                          ablation discussion: naively duplicating the
                          forward edge's area_ratio would encode "bridge
                          has smaller catchment than its upstream
                          neighbour" on an edge that's supposed to
                          represent the opposite relationship
  [2] elev_drop_m          NEGATED from the forward edge — a downstream-
                          to-upstream edge is a rise, not the same drop
  [3] same_tributary       unchanged (symmetric quantity)

Dynamic activation reference (new, 5th field, analogous to HAND's
z_saddle_m): gate_reference_m — the bridge node's own p90_mAOD flood
threshold. A model consuming this edge class should gate activation on
how close the bridge node's own current stage is to this reference,
not activate unconditionally.

Output: dataset/graph/backwater_edges.npz
  src               int32   [E_bw]  bridge/constriction node index
  dst               int32   [E_bw]  upstream node index (backwater target)
  river_dist_km     float32 [E_bw]
  area_ratio        float32 [E_bw]
  elev_drop_m       float32 [E_bw]
  same_tributary    float32 [E_bw]
  gate_reference_m  float32 [E_bw]  bridge node's p90_mAOD threshold
  bridge_name       object  [E_bw]  human-readable, for auditing

Usage:
    python precompute_backwater_edges.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR = BASE_DIR / "dataset" / "graph"
NODES_CSV = GRAPH_DIR / "nodes.csv"
EDGES_CSV = GRAPH_DIR / "edges.csv"
OUT_PATH  = GRAPH_DIR / "backwater_edges.npz"

BRIDGE_NAME_PATTERN = r"Bridge|Br$|Br "


def find_bridge_backwater_pairs(nd: pd.DataFrame, ed: pd.DataFrame) -> list[dict]:
    """
    For every bridge-named node with both a genuine upstream and a
    genuinely distinct downstream river neighbour, build one reverse
    backwater edge (bridge -> upstream neighbour).

    Mirrors the exact selection logic used in
    scenario_generator.py's generate_s6_channel_blockage, so the
    diagnostic scenario and the trained model's edge set are built from
    the same physical reasoning about where blockage-driven backwater is
    plausible in this catchment.
    """
    bridge_mask = nd["name"].str.contains(
        BRIDGE_NAME_PATTERN, case=False, regex=True, na=False)
    bridge_candidates = nd.loc[bridge_mask, "node_idx"].tolist()

    pairs = []
    for cand in bridge_candidates:
        upstream_edges = ed[ed["dst_idx"] == cand]
        if upstream_edges.empty:
            continue
        up_idx = int(upstream_edges.iloc[0]["src_idx"])

        # Exclude any "downstream" edge that loops straight back to the
        # chosen upstream node — see module docstring for why (2 of 28
        # edges in this graph form genuine bidirectional pairs).
        downstream_edges = ed[(ed["src_idx"] == cand) & (ed["dst_idx"] != up_idx)]
        if downstream_edges.empty:
            continue   # e.g. Glennamought Bridge — its only outgoing
                       # edge IS the bidirectional pair back to node 2

        fwd_edge = upstream_edges.iloc[0]   # upstream_node -> bridge_node
        pairs.append({
            "src": cand,
            "dst": up_idx,
            "bridge_name": nd.loc[nd.node_idx == cand, "name"].values[0],
            "river_dist_km":  float(fwd_edge["river_dist_km"]),
            "area_ratio":     1.0 / float(fwd_edge["area_ratio"]) if fwd_edge["area_ratio"] != 0 else 1.0,
            "elev_drop_m":    -float(fwd_edge["elev_drop_m"]),
            "same_tributary": float(fwd_edge["same_tributary"]),
        })
    return pairs


def build_backwater_edges(nd: pd.DataFrame, ed: pd.DataFrame) -> dict:
    pairs = find_bridge_backwater_pairs(nd, ed)

    if not pairs:
        raise RuntimeError(
            "No valid bridge backwater pairs found — check the bridge "
            "name pattern against nodes.csv, or whether every bridge "
            "node happens to be a pure headwater (no upstream edge) or "
            "sits on one of the graph's bidirectional exception pairs.")

    src = np.array([p["src"] for p in pairs], dtype=np.int32)
    dst = np.array([p["dst"] for p in pairs], dtype=np.int32)
    river_dist_km  = np.array([p["river_dist_km"]  for p in pairs], dtype=np.float32)
    area_ratio     = np.array([p["area_ratio"]     for p in pairs], dtype=np.float32)
    elev_drop_m    = np.array([p["elev_drop_m"]    for p in pairs], dtype=np.float32)
    same_tributary = np.array([p["same_tributary"] for p in pairs], dtype=np.float32)
    bridge_name    = np.array([p["bridge_name"]    for p in pairs], dtype=object)

    p90 = nd.set_index("node_idx")["p90_mAOD"]
    gate_reference_m = np.array([float(p90.loc[s]) for s in src], dtype=np.float32)

    return {
        "src": src, "dst": dst,
        "river_dist_km": river_dist_km, "area_ratio": area_ratio,
        "elev_drop_m": elev_drop_m, "same_tributary": same_tributary,
        "gate_reference_m": gate_reference_m, "bridge_name": bridge_name,
    }


def main():
    nd = pd.read_csv(NODES_CSV)
    ed = pd.read_csv(EDGES_CSV)

    result = build_backwater_edges(nd, ed)

    print(f"Backwater edges found: {len(result['src'])}")
    for i in range(len(result["src"])):
        print(f"  {result['bridge_name'][i]} (node {result['src'][i]}) "
              f"-> node {result['dst'][i]}  "
              f"gate_reference={result['gate_reference_m'][i]:.2f} m AOD")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **result)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
