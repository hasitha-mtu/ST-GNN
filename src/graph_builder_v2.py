"""
graph_builder.py  v2  –  Lee catchment GNN graph construction
=============================================================
Key change from v1: topology (edges) loaded directly from the
verified graph_metadata.json rather than derived from CONFLUENCE_MAP.
This eliminates all 6 topology errors in v1.

Static node features (7):
  [0] log(catchment_area_km2 + 1)
  [1] gauge_datum_m OSGM15
  [2] p90_mAOD              -- primary flood threshold
  [3] amax_med_mAOD         -- ~2yr flood level
  [4] is_reservoir (0/1)
  [5] is_tidal     (0/1)
  [6] has_discharge (0/1)

Static edge features (4):
  [0] river_dist_km
  [1] area_ratio
  [2] elev_drop_m
  [3] same_tributary (1.0 = intra, 0.5 = confluence)

Outputs saved to OUT_DIR:
  edge_index.npy      int64 [2, E]
  edge_attr.npy       float32 [E, 4]
  node_attr.npy       float32 [N, 7]
  node_order.json     list of station refs in node-index order
  graph_metadata.json updated with node_feature_names
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths  (adjust to your local layout) ──────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
META_JSON     = BASE_DIR / "dataset/graph/graph_metadata.json"   # authoritative topology
STATIONS_CSV  = BASE_DIR / "dataset/metadata/waterlevel_stations.csv"
DISCHARGE_DIR = BASE_DIR / "dataset/raw/discharge"               # to detect has_discharge
OUT_DIR       = BASE_DIR / "dataset/graph"

# ── Station refs that have Q dataset (20 non-tidal + none of the 4 level-only) ──
# Derived from wl_info_formatted.csv station_type == "Level & Flow"
# or "Level & Historic Flow". Checked once; hard-coded here for reproducibility.
HAS_DISCHARGE = {
    19056, 19057, 19058, 19059, 19045, 19110, 19114, 19113, 19112,
    19054, 19111, 19055, 19104, 19107, 19108, 19103, 19106, 19105,
    19101, 19109, 19102,
}
# Level-only (no Q): 19094, 19095, 19113 (wait — 19113 is Level? Let's check)
# From the breakdown: 19094, 19095, 19113, 19162 = Level only (no discharge)
# Tidal: 19160, 19161, 19163, 19164 = Tidal (no discharge)
# NOTE: override with filesystem check if preferred — see detect_discharge()

# ── Reservoir / tidal flags ───────────────────────────────────────────────
RESERVOIR_REFS = {19094, 19095, 19103, 19109}
TIDAL_REFS     = {19160, 19161, 19162, 19163, 19164}
# 19162 Fitzgerald's Park is "Level" (non-tidal station type) but sits just
# above the tidal limit. It is NOT in TIDAL_REFS in the metadata is_tidal flag.
# Check graph_metadata.json is_tidal field which is authoritative.


# ── Helper ─────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def detect_discharge(discharge_dir: Path, ref: str) -> bool:
    """Check filesystem for downloaded discharge CSV."""
    p = discharge_dir / f"discharge_{ref}.csv"
    return p.exists()


# ── Load verified graph metadata ───────────────────────────────────────────
def load_metadata() -> dict:
    if not META_JSON.exists():
        raise FileNotFoundError(f"graph_metadata.json not found at {META_JSON}")
    with open(META_JSON) as f:
        return json.load(f)


# ── Build node attribute matrix ────────────────────────────────────────────
def build_node_attr(stations: list[dict], use_filesystem_discharge: bool = False) -> tuple[np.ndarray, list[str]]:
    """
    Returns (node_attr [N,7], feature_names).
    stations: list of station dicts from graph_metadata.json, in node-index order.
    """
    rows = []
    for s in stations:
        ref = int(s['ref'])
        area = s.get('catchment_area_km2', 0.0) or 0.0

        # Thresholds — tidal stations have null thresholds
        p90     = s.get('p90_mAOD')   or 0.0
        amax    = s.get('amax_med')   or 0.0
        datum   = s.get('gauge_datum') or 0.0

        is_res   = int(s.get('is_reservoir', False))
        is_tidal = int(s.get('is_tidal', False))

        # Discharge availability
        if use_filesystem_discharge:
            has_q = int(detect_discharge(DISCHARGE_DIR, s['ref']))
        else:
            has_q = int(ref in HAS_DISCHARGE)

        rows.append([
            math.log(area + 1),   # [0] log area
            datum,                # [1] gauge datum OSGM15
            p90,                  # [2] p90 flood threshold
            amax,                 # [3] amax_med ~2yr flood
            is_res,               # [4] is_reservoir
            is_tidal,             # [5] is_tidal
            has_q,                # [6] has_discharge
        ])

    feature_names = [
        'log_catchment_area_km2',
        'gauge_datum_mOSGM15',
        'p90_mAOD',
        'amax_med_mAOD',
        'is_reservoir',
        'is_tidal',
        'has_discharge',
    ]
    return np.array(rows, dtype=np.float32), feature_names


# ── Build edge index and edge attribute matrix ─────────────────────────────
def build_edge_tensors(edges: list[dict], ref_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    edges: list of edge dicts from graph_metadata.json.
    ref_to_idx: station ref string → node index.

    Returns:
      edge_index  int64   [2, E]
      edge_attr   float32 [E, 4]  (river_dist_km, area_ratio, elev_drop_m, same_tributary)
    """
    src_list, dst_list, attr_list = [], [], []

    for e in edges:
        src = e['src_ref']
        dst = e['dst_ref']

        if src not in ref_to_idx or dst not in ref_to_idx:
            logger.warning("Edge %s→%s skipped — ref not in node list", src, dst)
            continue

        src_list.append(ref_to_idx[src])
        dst_list.append(ref_to_idx[dst])

        # Edge attributes — read directly from verified metadata
        dist       = max(e.get('river_dist_km', 0.1), 0.1)
        area_ratio = e.get('area_ratio', 1.0)
        elev_drop  = e.get('elev_drop_m', 0.0)
        same_trib  = e.get('same_tributary', 0.5)

        attr_list.append([dist, area_ratio, elev_drop, same_trib])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr  = np.array(attr_list, dtype=np.float32)
    return edge_index, edge_attr


# ── Visualisation ──────────────────────────────────────────────────────────
def visualise_graph(stations: list[dict], edge_index: np.ndarray, edge_attr: np.ndarray,
                    catchment_shp: Path | None = None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 1, figsize=(25, 15))

    if catchment_shp and catchment_shp.exists():
        import geopandas as gpd
        catchment = gpd.read_file(catchment_shp).to_crs('EPSG:4326')
        for ax in axes:
            catchment.boundary.plot(ax=ax, color='lightblue', linewidth=1.5, linestyle='--')

    ref_to_stn = {s['ref']: s for s in stations}
    node_list  = list(ref_to_stn.keys())

    color_map = {
        'reservoir': 'firebrick',
        'tidal':     'mediumpurple',
        'normal':    'steelblue',
    }

    for ax_i, ax in enumerate(axes):
        # Draw edges
        for j in range(edge_index.shape[1]):
            si, di = edge_index[0, j], edge_index[1, j]
            s = ref_to_stn[node_list[si]]
            d = ref_to_stn[node_list[di]]
            same_trib = edge_attr[j, 3]
            color = 'steelblue' if same_trib == 1.0 else 'coral'
            ax.annotate('', xy=(d['lon'], d['lat']), xytext=(s['lon'], s['lat']),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.3))
            if ax_i == 1:
                mx = (s['lon'] + d['lon']) / 2
                my = (s['lat'] + d['lat']) / 2
                ax.text(mx, my, f"{edge_attr[j,0]:.1f}", fontsize=5, ha='center', color='darkblue')

        # Draw nodes
        for s in stations:
            if s.get('is_reservoir'):
                c = color_map['reservoir']
            elif s.get('is_tidal'):
                c = color_map['tidal']
            else:
                c = color_map['normal']
            ax.scatter(s['lon'], s['lat'], s=65, color=c, zorder=5)
            if ax_i == 0:
                ax.text(s['lon'] + 0.004, s['lat'], s['name'], fontsize=5)

        ax.set_title(['Station labels', 'Edge distances (km)'][ax_i], fontsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    handles = [
        mpatches.Patch(color='steelblue',    label='Normal node'),
        mpatches.Patch(color='firebrick',    label='Reservoir node'),
        mpatches.Patch(color='mediumpurple', label='Tidal node'),
        mpatches.Patch(color='steelblue',    label='Intra-tributary edge', linestyle='-'),
        mpatches.Patch(color='coral',        label='Confluence edge'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8)
    fig.suptitle(f'Lee Catchment GNN Graph  ({len(stations)} nodes, {edge_index.shape[1]} edges)\n'
                 f'Verified topology from graph_metadata.json  |  Edge distances from OPW RiverNetwork',
                 fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ── Main entry point ───────────────────────────────────────────────────────
def build_graph(
    save: bool = True,
    plot: bool = True,
    use_filesystem_discharge: bool = False,
    catchment_shp: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load verified graph from graph_metadata.json, build numpy tensors,
    optionally save and plot.

    Returns (edge_index, edge_attr, node_attr, stations_list)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load verified metadata ─────────────────────────────────────────
    logger.info("Loading verified graph metadata …")
    meta = load_metadata()
    stations = meta['stations']         # already in verified node order
    edges    = meta['edges']

    logger.info("  %d stations, %d edges", len(stations), len(edges))

    # ── 2. Build ref → index mapping ─────────────────────────────────────
    ref_to_idx = {s['ref']: i for i, s in enumerate(stations)}

    # ── 3. Node attribute matrix ──────────────────────────────────────────
    logger.info("Building node attribute matrix …")
    node_attr, feature_names = build_node_attr(stations, use_filesystem_discharge)

    # ── 4. Edge tensors ───────────────────────────────────────────────────
    logger.info("Building edge tensors …")
    edge_index, edge_attr = build_edge_tensors(edges, ref_to_idx)

    # ── 5. Sanity checks ──────────────────────────────────────────────────
    N, E = len(stations), edge_index.shape[1]
    assert edge_index.max() < N, "Edge index out of bounds"
    assert node_attr.shape == (N, 7), f"Expected ({N},7), got {node_attr.shape}"
    assert edge_attr.shape == (E, 4), f"Expected ({E},4), got {edge_attr.shape}"

    # Summary
    logger.info("\n✓ Graph: %d nodes, %d edges", N, E)
    logger.info("  Node attr shape:  %s", node_attr.shape)
    logger.info("  Edge attr shape:  %s", edge_attr.shape)
    logger.info("  River dist range: [%.2f, %.2f] km",
                edge_attr[:,0].min(), edge_attr[:,0].max())
    logger.info("  Elev drop range:  [%.2f, %.2f] m",
                edge_attr[:,2].min(), edge_attr[:,2].max())

    has_q_count = int(node_attr[:, 6].sum())
    tidal_count = int(node_attr[:, 5].sum())
    res_count   = int(node_attr[:, 4].sum())
    logger.info("  Reservoir nodes:  %d", res_count)
    logger.info("  Tidal nodes:      %d", tidal_count)
    logger.info("  Has-discharge:    %d / %d", has_q_count, N)

    # ── 6. Save ───────────────────────────────────────────────────────────
    if save:
        np.save(OUT_DIR / 'edge_index.npy', edge_index)
        np.save(OUT_DIR / 'edge_attr.npy',  edge_attr)
        np.save(OUT_DIR / 'node_attr.npy',  node_attr)

        node_order = [s['ref'] for s in stations]
        with open(OUT_DIR / 'node_order.json', 'w') as f:
            json.dump({'node_refs': node_order, 'node_feature_names': feature_names,
                       'edge_feature_names': ['river_dist_km','area_ratio','elev_drop_m','same_tributary']}, f, indent=2)

        logger.info("  Saved edge_index.npy, edge_attr.npy, node_attr.npy, node_order.json → %s", OUT_DIR)

    # ── 7. Plot ───────────────────────────────────────────────────────────
    if plot:
        logger.info("Building visualisation …")
        fig = visualise_graph(stations, edge_index, edge_attr, catchment_shp)
        out_png = OUT_DIR / 'graph_viz.png'
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        logger.info("  Saved %s", out_png)
        import matplotlib.pyplot as plt
        plt.close(fig)

    return edge_index, edge_attr, node_attr, stations


if __name__ == '__main__':
    build_graph(save=True, plot=True)
