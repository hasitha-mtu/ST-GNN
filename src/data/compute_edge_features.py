"""
compute_edge_features.py
═══════════════════════════════════════════════════════════════════════
Computes the four physical edge features needed by the DFC-GNN attention
mechanism and saves them to dataset/graph/edge_features.npz.

Background
──────────
The DFC-GNN replaces the static HAND threshold with a learned, physically-
constrained dynamic attention layer. For each directed edge (i → j) the
attention score is computed as:

    e_ij = MLP([h_i, h_j,
                river_dist_km_ij,     ← this script
                elev_diff_m_ij,       ← this script
                travel_time_h_ij,     ← this script
                hand_diff_m_ij])      ← this script

    with hard gate:  A_ij = 0  if  elev_j > elev_i + max_flood_depth

The four features encode distinct physical flood-propagation constraints:

  river_dist_km   Along-network distance via Dijkstra on the river graph.
                  Attention decays with distance — a gauge 2 km upstream
                  has stronger influence than one 40 km away.

  elev_diff_m     Signed elevation difference (gauge_i mOD − gauge_j mOD).
                  Positive = water can flow from i toward j (downhill).
                  Used as the hard gate: if elev_j >> elev_i, flooding
                  cannot propagate from i to j regardless of stage.

  travel_time_h   Kinematic wave travel time in hours.
                  t_ij = dist_ij / c_k_i  where  c_k = 5/3 × v_flood_mean
                  v_flood_mean is the median measured velocity at node i
                  during flood events (Stage > bankfull), from the OPW
                  Gaugings CSV. Where gaugings are absent a reach-type
                  default is used (0.5 m/s for lowland, 0.8 m/s upland).

  hand_diff_m     Difference in bankfull HAND thresholds between nodes.
                  Encodes relative flood susceptibility: a node with low
                  bankfull anomaly floods more easily than one with high.

Edge set
─────────
All directed pairs (i → j) whose along-network Dijkstra distance is
below MAX_DIST_KM (default 80 km). This covers the full Lee mainstem
and all major tributaries without connecting unrelated sub-catchments.
Both directions (i→j and j→i) are included so attention can propagate
in both upstream and downstream directions.

Output:  dataset/graph/edge_features.npz
  src              int32  [E]  source node index
  dst              int32  [E]  destination node index
  river_dist_km    float32[E]  along-network distance (km)
  elev_diff_m      float32[E]  elevation_src − elevation_dst (m OD)
  travel_time_h    float32[E]  kinematic wave travel time (hours)
  hand_diff_m      float32[E]  bankfull_src − bankfull_dst (m anomaly)
  node_elevation_m float32[N]  gauge pixel elevation (m OD) — for hard gate
  node_celerity_ms float32[N]  per-node kinematic celerity c_k (m/s)
  node_velocity_ms float32[N]  per-node flood velocity (m/s, from gaugings)
  node_refs        str    [N]  OPW gauge reference strings (ordering key)

Usage
──────
  python src/data/compute_edge_features.py

  # With custom distance threshold
  python src/data/compute_edge_features.py --max-dist 50

  # Verbose: show per-node velocity and per-edge features
  python src/data/compute_edge_features.py --verbose
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
import sys as _sys
for _p in (str(BASE_DIR), str(BASE_DIR / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
GRAPH_DIR     = BASE_DIR / "dataset/graph"
STATION_DIR   = BASE_DIR / "dataset/station_data"
DEM_PATH      = BASE_DIR / "dataset/dem/COP-DEM-30m_itm.tif"
OUT_PATH      = GRAPH_DIR / "edge_features.npz"
NODE_ELEVATION_SOURCE = "channel"   # "channel" (revision) or "dem" (submitted)

MAX_DIST_KM   = 80.0   # maximum along-network distance for candidate edges
CELERITY_SCALE = 5 / 3  # kinematic wave celerity = 5/3 × mean flow velocity

# Default flood velocities by reach type (m/s) when gaugings are absent
DEFAULT_VELOCITY = {
    "lowland":  0.45,   # flat alluvial plain (lower Lee, Cork city)
    "midland":  0.65,   # valley corridor (mid Lee, Bride, Dripsey)
    "upland":   0.90,   # steep upland tributary (Sullane, upper Lee)
    "tidal":    0.25,   # tidal-influenced reach (estuary nodes)
    "reservoir":0.10,   # reservoir headrace (Inniscarra, Carrigadrohid)
}

# Reach type per node reference (assigned from knowledge of the Lee catchment)
NODE_REACH_TYPE = {
    "19056": "upland",   "19057": "upland",    "19058": "upland",
    "19059": "upland",   "19054": "upland",    "19055": "midland",
    "19101": "midland",  "19111": "midland",   "19104": "midland",
    "19107": "midland",  "19108": "lowland",   "19106": "lowland",
    "19105": "midland",  "19110": "lowland",   "19114": "lowland",
    "19113": "lowland",  "19112": "lowland",   "19045": "lowland",
    "19103": "lowland",  "19109": "reservoir", "19102": "lowland",
    "19095": "reservoir","19094": "reservoir", "19162": "tidal",
    "19163": "tidal",    "19161": "tidal",     "19160": "tidal",
}


# ═════════════════════════════════════════════════════════════════════
# Step 1: Load nodes and build river network graph
# ═════════════════════════════════════════════════════════════════════

def load_nodes() -> pd.DataFrame:
    """Load nodes.csv and return with integer index."""
    df = pd.read_csv(GRAPH_DIR / "nodes.csv")
    df["ref"] = df["ref"].astype(str)
    df = df.reset_index(drop=True)
    df["node_idx"] = df.index
    return df


NETWORK_SOURCE = "river"   # "river" (revision) or "submitted" (KNN + HAND + Euclidean)


def build_network_graph(nodes_df: pd.DataFrame) -> "nx.DiGraph":
    """
    River network for all-pairs along-channel distances.

    REVISION AUDIT (A21): the submitted version never read the river network.
    edges.csv has columns src_idx/dst_idx, but the parser looked for
    src/source and dst/target, so every row was skipped (and it would have
    used straight-line distances anyway). The "river distance" and the
    travel time between any two gauges were therefore shortest paths over
    16 HAND overland links plus 60 K-nearest-neighbour straight-line links,
    which jump across watersheds (e.g. Ballyvourney -> Waterworks Weir
    48 km instead of 66 km along the river; Coolmuckey -> Gothic Bridge
    17 km instead of 39 km). The graph is now the 26-edge river tree from
    edges.csv, undirected, weighted by the along-channel reach length
    river_dist_km (compute_river_distances.py).
    """
    import networkx as nx
    from utils.graph_checks import river_edges

    N = len(nodes_df)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    if NETWORK_SOURCE != "river":
        return _build_network_graph_submitted(nodes_df)
    ep = GRAPH_DIR / "edges.csv"
    if not ep.exists():
        raise FileNotFoundError(f"{ep} not found - run graph_builder.py first")
    ed = river_edges(pd.read_csv(ep))
    for e in ed.itertuples():
        s, d, L = int(e.src_idx), int(e.dst_idx), float(e.river_dist_km)
        G.add_edge(s, d, dist_km=L)
        G.add_edge(d, s, dist_km=L)
    comps = list(nx.connected_components(G.to_undirected()))
    print(f"  River network: {len(ed)} reaches from edges.csv, "
          f"{len(comps)} connected component(s)")
    if len(comps) != 1:
        raise RuntimeError(f"river network is not connected: {comps}")
    return G


def _build_network_graph_submitted(nodes_df: pd.DataFrame) -> "nx.DiGraph":
    """Submitted behaviour (HAND overland links + KNN straight-line links),
    kept only for the sensitivity comparison NETWORK_SOURCE = "submitted"."""
    import networkx as nx
    from scipy.spatial import cKDTree
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes_df)))
    hand_path = GRAPH_DIR / "hand_edges.npz"
    if hand_path.exists():
        data = np.load(hand_path)
        for s, d, dist in zip(data["src"], data["dst"], data["overland_dist_km"]):
            G.add_edge(int(s), int(d), dist_km=float(dist))
            G.add_edge(int(d), int(s), dist_km=float(dist))
    coords = nodes_df[["easting_itm", "northing_itm"]].values.astype(float)
    tree = cKDTree(coords)
    for i in range(len(nodes_df)):
        dists_eu, nbrs = tree.query(coords[i], k=5)
        for dist_eu, j in zip(dists_eu[1:], nbrs[1:]):
            if not G.has_edge(i, j):
                G.add_edge(i, j, dist_km=float(dist_eu) / 1000.0)
                G.add_edge(j, i, dist_km=float(dist_eu) / 1000.0)
    return G


def all_pairs_river_distance(G: "nx.DiGraph",
                              n_nodes: int) -> np.ndarray:
    """
    Compute all-pairs shortest-path distance (km) using Dijkstra on the
    river network graph. Returns [N, N] matrix with np.inf for unreachable
    pairs.
    """
    import networkx as nx

    dist_matrix = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
    np.fill_diagonal(dist_matrix, 0.0)

    paths = dict(nx.all_pairs_dijkstra_path_length(G, weight="dist_km"))
    for src, targets in paths.items():
        for dst, d in targets.items():
            dist_matrix[src, dst] = float(d)

    reachable = (dist_matrix < np.inf).sum() - n_nodes  # exclude diagonal
    print(f"  Dijkstra: {reachable} reachable directed pairs "
          f"(of {n_nodes*(n_nodes-1)} total)")
    return dist_matrix


# ═════════════════════════════════════════════════════════════════════
# Step 2: Node elevations from DEM
# ═════════════════════════════════════════════════════════════════════

def _load_gauge_datum(ref: str) -> float | None:
    """Load gauge datum (m OD) from DatumHistory CSV for a given gauge ref."""
    for station_dir in [STATION_DIR / ref, STATION_DIR]:
        if not station_dir.exists():
            continue
        for fp in station_dir.glob("*atum*"):
            try:
                df = pd.read_csv(fp, comment="#", skipinitialspace=True,
                                 encoding="utf-8-sig")
                df.columns = df.columns.str.strip()
                vcol = next((c for c in df.columns if "value" in c.lower()), None)
                if vcol:
                    vals = pd.to_numeric(df[vcol], errors="coerce").dropna()
                    if len(vals):
                        return float(vals.iloc[-1])
            except Exception:
                pass
    return None


def extract_node_elevations(nodes_df: pd.DataFrame) -> np.ndarray:
    """
    Sample the ITM DEM at each gauge node's pixel location.
    Returns [N] array of elevations in metres OD.
    """
    import rasterio

    if not DEM_PATH.exists():
        warnings.warn(f"DEM not found at {DEM_PATH} — using zero elevations")
        return np.zeros(len(nodes_df), dtype=np.float32)

    with rasterio.open(DEM_PATH) as src:
        dem    = src.read(1).astype(np.float32)
        affine = src.transform
        nodata = src.nodata if src.nodata is not None else -9999.0
        H, W   = src.height, src.width

    dem[dem == nodata] = np.nan
    elevations = np.zeros(len(nodes_df), dtype=np.float32)

    for i, row in nodes_df.iterrows():
        e = float(row["easting_itm"])
        n = float(row["northing_itm"])
        col = int(np.clip(round((e - affine.c) / affine.a), 0, W - 1))
        r   = int(np.clip(round((n - affine.f) / affine.e), 0, H - 1))
        val = dem[r, col]
        if np.isnan(val):
            # Try 3×3 neighbourhood first
            r0, r1 = max(0, r-1), min(H, r+2)
            c0, c1 = max(0, col-1), min(W, col+2)
            patch = dem[r0:r1, c0:c1]
            if not np.all(np.isnan(patch)):
                val = float(np.nanmedian(patch))
            else:
                # Fallback: load gauge datum from DatumHistory CSV
                # (more accurate than DEM for tidal/estuary nodes)
                datum_val = _load_gauge_datum(str(row["ref"]))
                val = datum_val if datum_val is not None else 0.0
        elevations[i] = float(val)

    # REVISION AUDIT: raw-pixel DEM sampling put several gauges on the valley
    # side rather than the channel (Glen Park +15 m, Glennamought +13 m,
    # Dripsey +11 m, Cooldaniel +11 m above their gauge datum) and gave
    # 0.09 / 0.28 DFC elevation-gate values on the two river edges into
    # Currach Club. The fixed elevation gate now uses the CHANNEL reference:
    # the gauge datum where the datum is a verified absolute level
    # (utils.datum.abs_datum_valid), otherwise the channel-snapped DEM
    # elevation from kirpich_inputs.csv. Set NODE_ELEVATION_SOURCE = "dem"
    # to reproduce the submitted behaviour.
    if NODE_ELEVATION_SOURCE == "channel":
        import sys as _s
        _s.path.insert(0, str(BASE_DIR / "src"))
        from utils.datum import abs_datum_valid
        ok = abs_datum_valid(nodes_df)
        kp = GRAPH_DIR / "kirpich_inputs.csv"
        snapped = pd.read_csv(kp).set_index("node_idx")["z_gauge_m"] if kp.exists() else None
        # Pool gauges (reservoir headraces, local-zero datum): the gauge zero
        # is tens of metres below the water, so the channel reference is the
        # training-period median pool level (p50_mAOD_train).
        low = nodes_df["stage_ref_low_m"].to_numpy(float) if "stage_ref_low_m" in nodes_df \
            else nodes_df["gauge_datum_mOSGM15"].to_numpy(float)
        is_pool = np.isfinite(low) & (np.abs(low - nodes_df["gauge_datum_mOSGM15"].to_numpy(float)) > 0.5) \
            & ~nodes_df.get("is_tidal", pd.Series(0, index=nodes_df.index)).to_numpy().astype(bool)
        n_pool = 0
        for i in range(len(nodes_df)):
            if ok[i] and is_pool[i] and "p50_mAOD_train" in nodes_df \
                    and np.isfinite(nodes_df.loc[i, "p50_mAOD_train"]):
                elevations[i] = float(nodes_df.loc[i, "p50_mAOD_train"]); n_pool += 1
            elif ok[i]:
                elevations[i] = float(nodes_df.loc[i, "gauge_datum_mOSGM15"])
            elif snapped is not None:
                elevations[i] = float(snapped.loc[i])
        print(f"  Node elevations: channel reference (datum for {int(ok.sum()) - n_pool} nodes, "
              f"median pool level for {n_pool}, snapped DEM for {int((~ok).sum())})")
    valid = np.isfinite(elevations) & (elevations != 0)
    print(f"  Node elevations: {valid.sum()}/{len(nodes_df)} non-zero")
    print(f"  Range: {elevations[valid].min():.1f} – {elevations[valid].max():.1f} m OD")
    return elevations


# ═════════════════════════════════════════════════════════════════════
# Step 3: Travel time — velocity from gaugings
# ═════════════════════════════════════════════════════════════════════

def _find_gaugings(ref: str) -> Path | None:
    """Find a Gaugings CSV for a given station reference."""
    station_dir = STATION_DIR / ref
    if not station_dir.exists():
        return None
    for fp in station_dir.glob("*.csv"):
        if "gauging" in fp.name.lower():
            return fp
    return None


def _load_velocity(fp: Path) -> pd.Series | None:
    """Parse the Velocity column from a Gaugings CSV."""
    try:
        df = pd.read_csv(fp, comment="#", skipinitialspace=True,
                         encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        vel_col = next((c for c in df.columns
                        if "velocity" in c.lower() or c.strip() == "Velocity"),
                       None)
        if vel_col is None:
            return None
        return pd.to_numeric(df[vel_col], errors="coerce").dropna()
    except Exception:
        return None


def load_bankfull_readings(nodes_df: pd.DataFrame) -> np.ndarray:
    """
    Bankfull STAGE (m above the gauge zero) per node, from
    bankfull_thresholds_report.csv (derive_bankfull_thresholds.py).

    This is the same reference as the "Stage" column of the OPW gaugings
    files. Only Tier 1 (gaugings width expansion) and Tier 2 (Gumbel annual
    maxima, OPW annual-maximum stage) are gauge-zero referenced; other tiers
    return NaN.
    """
    rp = GRAPH_DIR / "bankfull_thresholds_report.csv"
    out = np.full(len(nodes_df), np.nan, dtype=np.float64)
    if not rp.exists():
        warnings.warn("bankfull_thresholds_report.csv not found - run "
                      "derive_bankfull_thresholds.py; flood-velocity filter disabled")
        return out
    rep = pd.read_csv(rp, dtype={"ref": str}).set_index("ref")
    for i, ref in enumerate(nodes_df["ref"].astype(str)):
        if ref in rep.index and rep.loc[ref, "tier_used"] in (1, 2):
            out[i] = float(rep.loc[ref, "bankfull_reading_m"])
    return out


MIN_FLOOD_GAUGINGS = 2
# Fallback when fewer than MIN_FLOOD_GAUGINGS gaugings lie above bankfull:
#   "rating" (revision): station velocity-stage line evaluated at the bankfull
#            stage, bounded below by the top-quartile-stage median and above by
#            3 m/s; top-quartile median if the line is not usable
#   "all"    : median of all gaugings (low-flow biased; first revision run)
VELOCITY_FALLBACK = "rating"
RATING_MIN_N = 5
RATING_MAX_EXTRAP = 1.5      # bankfull stage may be at most 1.5x the highest gauged stage


def velocity_at_bankfull(stage: np.ndarray, vel: np.ndarray,
                         h_bf: float) -> tuple[float | None, str]:
    """Velocity at the bankfull stage from a station's gaugings when too few
    gaugings lie above bankfull. Pure function (unit-tested)."""
    ok = np.isfinite(stage) & np.isfinite(vel) & (vel > 0.01) & (vel <= 3.0)
    st, v = stage[ok], vel[ok]
    if len(v) < 3:
        return None, "too few gaugings"
    # highest-stage quarter of the gaugings, at least 2 (Ballincolly has 4
    # gaugings: a strict quartile is a single measurement)
    k = max(2, int(np.ceil(0.25 * len(v))))
    top = v[np.argsort(st)[-k:]]
    v_top = float(np.median(top))
    if (np.isfinite(h_bf) and len(v) >= RATING_MIN_N and np.ptp(st) > 0.05
            and st.max() > 0 and h_bf <= RATING_MAX_EXTRAP * st.max()):
        b, a = np.polyfit(st, v, 1)
        if b > 0:
            v_bf = float(np.clip(a + b * h_bf, v_top, 3.0))
            return v_bf, (f"rating-fit n={len(v)} v={a:.2f}+{b:.2f}h at h_bf={h_bf:.2f} m"
                          f"{' (extrap.)' if h_bf > st.max() else ''}")
    return v_top, f"gaugings-topQ n={len(top)} (stage>={np.sort(st)[-k]:.2f} m)"


def extract_node_velocities(
    nodes_df:    pd.DataFrame,
    bankfull_thr: np.ndarray,   # [N] bankfull stage anomaly (kept for the API)
    verbose:     bool = False,
    bankfull_stage: np.ndarray | None = None,   # [N] m above gauge zero
) -> np.ndarray:
    """
    Per-node flood velocity (m/s) from the OPW gaugings files.

    Flood velocity = median measured velocity over gaugings with
    Stage > bankfull stage (near or above-bank flow).

    REVISION AUDIT (A22): the submitted filter compared the gaugings Stage
    (m above the gauge zero) with the bankfull ANOMALY (bankfull stage minus
    median stage), i.e. a threshold too low by the median stage, so
    sub-bankfull gaugings were counted as flood gaugings. It now uses the
    bankfull stage itself (load_bankfull_readings). With fewer than
    MIN_FLOOD_GAUGINGS above bankfull, the station's velocity-stage line is
    evaluated at the bankfull stage (velocity_at_bankfull; VELOCITY_FALLBACK).
    The first revision run used the median of ALL gaugings there, which is
    dominated by low-flow gaugings (Gothic Bridge 0.13 m/s, i.e. a 0.22 m/s
    celerity and travel times up to 91 h). Nodes without gaugings use the
    reach-type default. Every node's source is printed.
    """
    if bankfull_stage is None:
        bankfull_stage = load_bankfull_readings(nodes_df)
    velocities = np.zeros(len(nodes_df), dtype=np.float32)
    sources = []
    for i, row in nodes_df.iterrows():
        ref = str(row["ref"])
        rtype = NODE_REACH_TYPE.get(ref, "midland")
        v_flood, label = None, None
        gp = _find_gaugings(ref)
        if gp is not None:
            try:
                df = pd.read_csv(gp, comment="#", skipinitialspace=True,
                                 encoding="utf-8-sig")
                df.columns = df.columns.str.strip()
                stage_col = next((c for c in df.columns if "stage" in c.lower()), None)
                vel_col = next((c for c in df.columns if "velocity" in c.lower()), None)
            except Exception:
                df, stage_col, vel_col = None, None, None
            if df is not None and vel_col is not None:
                v = pd.to_numeric(df[vel_col], errors="coerce")
                ok_v = v.notna() & (v > 0.01) & (v <= 3.0)     # >3 m/s non-physical
                thr = float(bankfull_stage[i])
                if stage_col is not None and np.isfinite(thr):
                    st = pd.to_numeric(df[stage_col], errors="coerce")
                    flood_v = v[ok_v & (st > thr)]
                    if len(flood_v) >= MIN_FLOOD_GAUGINGS:
                        v_flood = float(flood_v.median())
                        label = f"gaugings>bankfull n={len(flood_v)} (stage>{thr:.2f} m)"
                if v_flood is None and VELOCITY_FALLBACK == "rating" and stage_col is not None:
                    st = pd.to_numeric(df[stage_col], errors="coerce")
                    v_flood, label = velocity_at_bankfull(st.to_numpy(float),
                                                          v.to_numpy(float), thr)
                if v_flood is None and ok_v.sum() >= 3:
                    v_flood = float(v[ok_v].median())
                    label = f"gaugings-all n={int(ok_v.sum())}"
        if v_flood is not None and 0.01 < v_flood < 5.0:
            velocities[i] = v_flood
        else:
            velocities[i] = DEFAULT_VELOCITY[rtype]
            label = f"default: {rtype}"
        sources.append(f"  {ref:>7}  {velocities[i]:.3f} m/s  ({label})")

    n_flood = sum("gaugings>bankfull" in s_ for s_ in sources)
    n_fit = sum("rating-fit" in s_ or "gaugings-topQ" in s_ or "gaugings-all" in s_
                for s_ in sources)
    print(f"  Velocities: {n_flood} from above-bankfull gaugings, {n_fit} from the "
          f"station velocity-stage data at bankfull ({VELOCITY_FALLBACK}), "
          f"{len(sources) - n_flood - n_fit} reach-type defaults")
    if verbose:
        for s_ in sources:
            print(s_)
    return velocities


def compute_travel_times(
    dist_matrix:  np.ndarray,   # [N, N] along-network distance (km)
    velocities:   np.ndarray,   # [N] flood velocity at each source node (m/s)
) -> np.ndarray:
    """
    Compute kinematic wave travel time matrix [N, N] in hours.

    Travel time from i to j:
        c_k_i  = CELERITY_SCALE × velocity_i     (kinematic wave celerity, m/s)
        t_ij   = dist_ij_m / c_k_i / 3600        (hours)

    The celerity is assigned to the SOURCE node because it is the
    upstream flow velocity that determines how fast the wave propagates
    downstream. For reverse-direction edges (j → i, upstream), the
    destination node's celerity is used as an approximation.

    Returns [N, N] matrix in hours, np.inf for unreachable pairs.
    """
    N = len(velocities)
    celerities  = CELERITY_SCALE * velocities            # [N] m/s
    tt_matrix   = np.full((N, N), np.inf, dtype=np.float32)

    for i in range(N):
        c = celerities[i]
        if c <= 0:
            continue
        dist_m = dist_matrix[i, :] * 1000.0             # km → m
        reachable = np.isfinite(dist_m)
        tt_matrix[i, reachable] = (dist_m[reachable] / c / 3600.0).astype(np.float32)

    np.fill_diagonal(tt_matrix, 0.0)
    return tt_matrix


# ═════════════════════════════════════════════════════════════════════
# Step 4: Bankfull thresholds
# ═════════════════════════════════════════════════════════════════════

def load_bankfull_thresholds(nodes_df: pd.DataFrame) -> np.ndarray:
    """
    Load per-node bankfull stage anomaly thresholds from JSON.
    Returns [N] array, falling back to 0.5 m for missing nodes.
    """
    bf_path = GRAPH_DIR / "bankfull_thresholds.json"
    if not bf_path.exists():
        warnings.warn(f"bankfull_thresholds.json not found — using 0.5 m default")
        return np.full(len(nodes_df), 0.5, dtype=np.float32)

    with open(bf_path) as f:
        import json
        data = json.load(f)
    thr_map = data.get("thresholds", {})

    thresholds = np.array(
        [float(thr_map.get(str(row["ref"]), 0.5))
         for _, row in nodes_df.iterrows()],
        dtype=np.float32)

    print(f"  Bankfull thresholds: range "
          f"{thresholds.min():.3f} – {thresholds.max():.3f} m")
    return thresholds


# ═════════════════════════════════════════════════════════════════════
# Step 5: Assemble edge feature arrays
# ═════════════════════════════════════════════════════════════════════

def build_edge_arrays(
    dist_matrix:  np.ndarray,    # [N, N] river distance (km)
    tt_matrix:    np.ndarray,    # [N, N] travel time (hours)
    elevations:   np.ndarray,    # [N] elevation (m OD)
    bankfull_thr: np.ndarray,    # [N] bankfull anomaly (m)
    max_dist_km:  float,
) -> dict:
    """
    Flatten pairwise matrices into edge-index format.

    An edge (i, j) is included if:
      1. Along-network distance dist_matrix[i, j] <= max_dist_km
      2. i ≠ j

    Both directions (i→j and j→i) are included to allow the attention
    mechanism to propagate signals in both up- and downstream directions.
    """
    N    = len(elevations)
    srcs, dsts = [], []
    river_dists, elev_diffs, travel_times, hand_diffs = [], [], [], []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist_matrix[i, j]
            if not np.isfinite(d) or d > max_dist_km:
                continue
            srcs.append(i)
            dsts.append(j)
            river_dists.append(float(d))
            elev_diffs.append(float(elevations[i] - elevations[j]))
            travel_times.append(float(tt_matrix[i, j]))
            hand_diffs.append(float(bankfull_thr[i] - bankfull_thr[j]))

    print(f"  Edge arrays built: {len(srcs)} directed edges "
          f"(max_dist={max_dist_km} km)")
    return {
        "src":           np.array(srcs,         dtype=np.int32),
        "dst":           np.array(dsts,         dtype=np.int32),
        "river_dist_km": np.array(river_dists,  dtype=np.float32),
        "elev_diff_m":   np.array(elev_diffs,   dtype=np.float32),
        "travel_time_h": np.array(travel_times, dtype=np.float32),
        "hand_diff_m":   np.array(hand_diffs,   dtype=np.float32),
    }


# ═════════════════════════════════════════════════════════════════════
# Step 6: Normalise features
# ═════════════════════════════════════════════════════════════════════

def normalise(arr: np.ndarray, name: str) -> tuple[np.ndarray, float, float]:
    """
    Z-score normalise a feature array.
    Returns (normalised, mean, std) for inverse-transform if needed.
    Handles inf/nan by replacing with median before normalising.
    """
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return arr.copy(), 0.0, 1.0
    mu  = float(np.median(finite))     # robust to outliers
    std = float(finite.std()) or 1.0
    out = arr.copy()
    out[~np.isfinite(out)] = mu        # fill inf/nan with median
    out = (out - mu) / std
    print(f"  {name:>18}: μ={mu:.3f}  σ={std:.3f}  "
          f"range=[{finite.min():.2f}, {finite.max():.2f}]")
    return out.astype(np.float32), mu, std


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def run(max_dist_km: float, verbose: bool):
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("pip install networkx")

    print("═"*60)
    print("  DFC-GNN Edge Feature Computation")
    print("═"*60)

    # ── 1. Nodes ─────────────────────────────────────────────────────
    print("\n── Loading nodes ──")
    nodes_df = load_nodes()
    N        = len(nodes_df)
    print(f"  {N} gauge nodes loaded")

    # ── 2. River network graph + all-pairs distances ──────────────────
    print("\n── Building river network graph ──")
    G           = build_network_graph(nodes_df)
    dist_matrix = all_pairs_river_distance(G, N)

    # ── 3. Node elevations ────────────────────────────────────────────
    print("\n── Extracting node elevations from DEM ──")
    elevations = extract_node_elevations(nodes_df)

    # ── 4. Bankfull thresholds ────────────────────────────────────────
    print("\n── Loading bankfull thresholds ──")
    bankfull_thr = load_bankfull_thresholds(nodes_df)

    # ── 5. Velocities + travel times ─────────────────────────────────
    print("\n── Computing per-node flood velocities ──")
    velocities  = extract_node_velocities(nodes_df, bankfull_thr, verbose)
    celerities  = CELERITY_SCALE * velocities

    print("\n── Computing travel time matrix ──")
    tt_matrix   = compute_travel_times(dist_matrix, velocities)

    finite_tt = tt_matrix[np.isfinite(tt_matrix) & (tt_matrix > 0)]
    print(f"  Travel times: range [{finite_tt.min():.2f}, {finite_tt.max():.2f}] hours")

    # ── 6. Build edge arrays ──────────────────────────────────────────
    print(f"\n── Assembling edge arrays (max dist = {max_dist_km} km) ──")
    edges = build_edge_arrays(
        dist_matrix, tt_matrix, elevations, bankfull_thr, max_dist_km)
    E = len(edges["src"])

    if E == 0:
        print(f"  WARNING: no edges within {max_dist_km} km — "
              f"try increasing --max-dist")
        return

    # ── 7. Normalise features ─────────────────────────────────────────
    print("\n── Normalising features ──")
    rd_n, rd_mu, rd_std = normalise(edges["river_dist_km"], "river_dist_km")
    ed_n, ed_mu, ed_std = normalise(edges["elev_diff_m"],   "elev_diff_m")
    tt_n, tt_mu, tt_std = normalise(edges["travel_time_h"], "travel_time_h")
    hd_n, hd_mu, hd_std = normalise(edges["hand_diff_m"],  "hand_diff_m")

    # ── 8. Save ───────────────────────────────────────────────────────
    print(f"\n── Saving to {OUT_PATH.name} ──")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUT_PATH,
        # Edge index
        src             = edges["src"],
        dst             = edges["dst"],
        # Raw features (unscaled — for inspection and threshold application)
        river_dist_km   = edges["river_dist_km"],
        elev_diff_m     = edges["elev_diff_m"],
        travel_time_h   = edges["travel_time_h"],
        hand_diff_m     = edges["hand_diff_m"],
        # Normalised features (for model input)
        river_dist_norm = rd_n,
        elev_diff_norm  = ed_n,
        travel_time_norm= tt_n,
        hand_diff_norm  = hd_n,
        # Normalisation statistics (for reproducibility)
        norm_stats      = np.array([rd_mu, rd_std, ed_mu, ed_std,
                                    tt_mu, tt_std, hd_mu, hd_std],
                                   dtype=np.float32),
        # Node-level arrays
        node_elevation_m  = elevations,
        node_velocity_ms  = velocities,
        node_celerity_ms  = celerities,
        node_refs         = np.array(nodes_df["ref"].tolist()),
    )

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"  Saved: {OUT_PATH}  ({size_kb:.1f} KB)")

    # ── 9. Summary ────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Summary")
    print(f"{'═'*60}")
    print(f"  Nodes:                 {N}")
    print(f"  Directed edges:        {E}")
    print(f"  Max river distance:    {max_dist_km} km")
    print(f"  Avg travel time:       {finite_tt.mean():.2f} h")
    print(f"  Elevation range:       {elevations.min():.0f}–{elevations.max():.0f} m OD")

    if verbose:
        print(f"\n  Per-node summary:")
        print(f"  {'ref':>7}  {'elev(m)':>8}  {'v(m/s)':>7}  "
              f"{'c_k(m/s)':>9}  {'bf_thr(m)':>10}")
        print(f"  {'─'*50}")
        for i, row in nodes_df.iterrows():
            print(f"  {row['ref']:>7}  {elevations[i]:>8.1f}  "
                  f"{velocities[i]:>7.3f}  {celerities[i]:>9.3f}  "
                  f"{bankfull_thr[i]:>10.3f}")

    print(f"\n  Loaded by DFC-GNN model as:")
    print(f"    data = np.load('dataset/graph/edge_features.npz')")
    print(f"    edge_index = torch.tensor([data['src'], data['dst']])")
    print(f"    edge_attr  = torch.tensor(np.stack([")
    print(f"        data['river_dist_norm'], data['elev_diff_norm'],")
    print(f"        data['travel_time_norm'], data['hand_diff_norm']], axis=1))")
    print(f"    node_elev  = torch.tensor(data['node_elevation_m'])")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute physical edge features for DFC-GNN")
    p.add_argument("--max-dist", type=float, default=MAX_DIST_KM,
                   help=f"Maximum along-network distance (km) for edges "
                        f"(default: {MAX_DIST_KM})")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-node velocity and per-edge features")
    p.add_argument("--out", type=Path, default=OUT_PATH,
                   help=f"Output path (default: {OUT_PATH})")
    args = p.parse_args()
    OUT_PATH = args.out
    run(args.max_dist, args.verbose)
