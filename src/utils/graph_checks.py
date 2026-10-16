"""
graph_checks.py  —  Topology integrity checks for the directed river graph.

Revision audit finding
----------------------
graph_metadata.json contained two uphill duplicate edges:

    19057 Glen Park           -> 19056 Ballincolly          (elev_drop -19.76 m)
    19058 Blackpool Retail Pk -> 19059 Glennamought Bridge  (elev_drop -30.14 m)

Each one reverses an existing downhill edge, which creates 2-cycles in a
graph the manuscript describes as "directed edges ... from upstream to
downstream". Consequences:
  * every graph model received uphill messages between these gauges;
  * S1/S5 routing (BFS along out-edges) propagated the synthetic headwater
    pulse from Glen Park UP to Ballincolly and from Blackpool UP to
    Glennamought (both are themselves S1 injection nodes);
  * S2 criticality counts and backwater-edge selection were affected.

`river_edges()` removes such edges. A reverse edge is identified as an
edge (j -> i) whose forward twin (i -> j) also exists and whose own
elevation drop is negative. The tidal edge 19161 -> 19160 (elev_drop
-2.6 m between two tidal gauges with no forward twin) is NOT removed;
tidal datums are not a hydraulic gradient.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


def find_reverse_duplicates(ed: pd.DataFrame) -> list[int]:
    """Row indices of edges that reverse an existing edge and point uphill."""
    pairs = {(int(r.src_idx), int(r.dst_idx)): i for i, r in ed.iterrows()}
    bad = []
    for (s, d), i in pairs.items():
        if (d, s) in pairs and float(ed.loc[i, "elev_drop_m"]) < 0:
            bad.append(i)
    return sorted(bad)


def river_edges(ed: pd.DataFrame, drop_reverse: bool = True) -> pd.DataFrame:
    """Return the edge table with uphill reverse duplicates removed."""
    if not drop_reverse:
        return ed.copy()
    bad = find_reverse_duplicates(ed)
    return ed.drop(index=bad).reset_index(drop=True)


def is_dag(ed: pd.DataFrame, n_nodes: int) -> bool:
    indeg = [0] * n_nodes
    adj = defaultdict(list)
    for r in ed.itertuples():
        adj[int(r.src_idx)].append(int(r.dst_idx))
        indeg[int(r.dst_idx)] += 1
    q = deque(i for i in range(n_nodes) if indeg[i] == 0)
    seen = 0
    while q:
        u = q.popleft(); seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == n_nodes


def downstream_closure(ed: pd.DataFrame, source: int) -> set[int]:
    adj = defaultdict(list)
    for r in ed.itertuples():
        adj[int(r.src_idx)].append(int(r.dst_idx))
    seen, stack = set(), [source]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    seen.discard(source)
    return seen


def topological_edges_from(ed: pd.DataFrame, source: int) -> list[tuple[int, int]]:
    """All edges reachable from `source`, ordered so that every edge (u,v)
    appears after all edges entering u (Kahn order on the reachable
    sub-DAG). Unlike BFS-with-visited, a node reached through two paths
    (a confluence inside one source's sub-network) receives BOTH
    contributions, and receives them only after its upstream signal is
    complete."""
    reach = downstream_closure(ed, source) | {source}
    sub = ed[ed.src_idx.isin(reach) & ed.dst_idx.isin(reach)]
    indeg = defaultdict(int)
    adj = defaultdict(list)
    for r in sub.itertuples():
        adj[int(r.src_idx)].append(int(r.dst_idx))
        indeg[int(r.dst_idx)] += 1
    order: list[tuple[int, int]] = []
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            order.append((u, v))
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order
