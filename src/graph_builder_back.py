"""
graph_builder.py  –  Lee catchment GNN graph construction
Uses actual river network LINE shapefiles for proper reach distances.
River segments: _010 = most upstream, altitude decreasing downstream.
"""
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import linemerge

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent.parent
STATIONS_CSV   = BASE_DIR / "ST-GNN/dataset/metadata/waterlevel_stations.csv"
RIVER_SHP      = Path("/shapefiles/network/RiverNetwork/RiverNetwork.shp")
CATCHMENT_SHP  = Path("/shapefiles/RiverLee/RiverLee.shp")
LAKE_SHP       = Path("/shapefiles/lakes/LeeLakes/LeeLakes.shp")
OUT_DIR        = BASE_DIR / "dataset/graph"

# ── Constants ──────────────────────────────────────────────────────────────
DROPPED_REFS   = [19115]          # decommissioned / <50 % complete
TIDAL_REFS     = [19163, 19164, 19160, 19161]  # estuary / zero-area stations
RESERVOIR_REFS = [19094, 19095, 19103, 19109]

# Confluence map: tributary WATER_BODY → downstream target STATION REF
# Water-body names match the 'water_body' column in wl_info_formatted.csv
CONFLUENCE_MAP = {
    # Upper Lee tributaries (above Carrigadrohid reservoir)
    "Laney":     19101,   # Morris's Bridge (Laney) → Macroom (Sullane)
    "Foherish":  19055,   # Killaclug (Foherish) → Ballymakera (Sullane)
    "TOON":      19106,   # Cooleen Bridge (Toon) → Cooldaniel (Lee main stem)
    "MARTIN":    19095,   # Muskerry-via-Martin → Carrigadrohid Headrace
    "Sullane":   19095,   # Macroom (Sullane) → Carrigadrohid Headrace
    # Between reservoirs
    "Dripsey":   19094,   # Dripsey Bridge → Inniscarra Headrace
    # Below Inniscarra
    "BLARNEY":   19105,   # Bawnafinny (Blarney) → Muskerry (Shournagh)
    "Shournagh": 19109,   # Muskerry (Shournagh) → Inniscarra Tailrace
    "Bride (Cork)": 19109,  # Ovens (county Bride) → Inniscarra Tailrace
    # Cork city
    "Bride (Cork City)": 19162,  # Blackpool (city Bride) → Fitzgerald's Park
    "Curragheen": 19162,  # County Hall (Curragheen) → Fitzgerald's Park
    "Glen Tributary": 19057,  # Glennamought → Glen Park
    "GLEN":      19162,   # Glen Park → Fitzgerald's Park
}

# Remap specific stations whose OPW water_body label is ambiguous
# 19058 Blackpool Retail Park is on the Bride (Cork City) urban stream, not the county Bride
STATION_WB_OVERRIDES = {
    19058: "Bride (Cork City)",  # Blackpool – city Bride, not county Bride
}


# ── Helper ─────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2*R*math.asin(math.sqrt(a))


# ── Station loader ─────────────────────────────────────────────────────────
def load_stations():
    df = pd.read_csv(STATIONS_CSV)
    df = df.rename(columns={
        'Station No.': 'ref', 'Station': 'name', 'Water Body': 'water_body',
        'Catchment Area': 'catchment_area', 'Latitude': 'lat', 'Longitude': 'lon',
    })
    df['catchment_area_km2'] = df['catchment_area'].str.extract(r'([\d.]+)').astype(float)
    df = df[~df['ref'].isin(DROPPED_REFS + TIDAL_REFS)]
    df = df[df['catchment_area_km2'] > 0].copy()
    df['is_reservoir'] = df['ref'].isin(RESERVOIR_REFS).astype(int)
    # Apply per-station water body corrections
    for ref, wb in STATION_WB_OVERRIDES.items():
        df.loc[df['ref'] == ref, 'water_body'] = wb
    df = df.sort_values('catchment_area_km2').reset_index(drop=True)
    return df


# ── River network loader ───────────────────────────────────────────────────
def load_river_network():
    rn = gpd.read_file(RIVER_SHP)
    rn['base_name'] = rn['NAME'].str.rsplit('_', n=1).str[0]
    rn['seg_num']   = rn['NAME'].str.rsplit('_', n=1).str[1].astype(int)
    return rn


def build_trib_cumulative(rn):
    """
    For each tributary, compute cumulative km from most upstream segment.
    Returns dict: base_name → DataFrame with columns [seg_num, cum_start_km, LENGTHKM, Altitude, Slope]
    """
    trib_cum = {}
    for base, grp in rn.groupby('base_name'):
        grp = grp.sort_values('seg_num').reset_index(drop=True)
        grp['cum_start_km'] = grp['LENGTHKM'].shift(1, fill_value=0).cumsum()
        trib_cum[base] = grp[['seg_num', 'cum_start_km', 'LENGTHKM', 'Altitude', 'Slope', 'geometry']].copy()
    return trib_cum


def station_river_dist(stn_geom, trib_df):
    """
    Given a station geometry (EPSG:29902 Point) and the sorted segment table for its tributary,
    return cumulative km from most upstream source to the station.
    """
    # Find nearest segment
    dists = trib_df.geometry.distance(stn_geom)
    nearest_pos = dists.idxmin()
    seg = trib_df.loc[nearest_pos]

    geom = seg.geometry
    if geom.geom_type == 'MultiLineString':
        geom = linemerge(geom)
        if geom.geom_type == 'MultiLineString':
            # fallback: use centroid fraction
            geom = max(geom.geoms, key=lambda g: g.length)

    seg_len_m = geom.length
    proj_m    = geom.project(stn_geom)
    frac      = (proj_m / seg_len_m) if seg_len_m > 0 else 0.5
    return seg['cum_start_km'] + frac * seg['LENGTHKM']


# ── Snap stations to river network ─────────────────────────────────────────
def snap_stations(stations, rn, trib_cum):
    """
    Returns stations enriched with:
      river_cum_km  – cumulative distance from source (within matched tributary)
      matched_trib  – base_name of matched tributary
      seg_altitude  – altitude of matched segment (proxy for elevation)
      seg_slope     – slope of matched segment
    """
    gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.lon, stations.lat),
        crs='EPSG:4326',
    ).to_crs('EPSG:29902')

    # Map OPW water_body → river network base_name
    # Prefer segments that share the water body name, else fall back to spatial minimum
    wb_to_rn = {
        'Sullane':       'SULLANE',
        'Bride (Cork)':  'BRIDE (LEE)',
        'LEE':           'LEE (CORK)',
        'Laney':         'LANEY',
        'Foherish':      'FOHERISH',
        'TOON':          'TOON',
        'Dripsey':       'DRIPSEY',
        'Shournagh':     'SHOURNAGH',
        'MARTIN':        'MARTIN',
        'Curragheen':    'CURRAGHEEN (Cork City)',
        'BLARNEY':       'BLARNEY',
        'Bride (Cork City)': 'BRIDE (Cork City)',  # Blackpool urban stream
        'Glen Tributary':'GLENNAMOUGHT TRIB BRIDE',
        'GLEN':          None,  # no dedicated segment → fall back to spatial
    }

    results = []
    for _, row in gdf.iterrows():
        wb = row.water_body
        trib_key = wb_to_rn.get(wb, None)

        if trib_key and trib_key in trib_cum:
            trib_df = trib_cum[trib_key]
            cum_km  = station_river_dist(row.geometry, trib_df)
            # get slope/altitude from nearest seg
            dists   = trib_df.geometry.distance(row.geometry)
            ni      = dists.idxmin()
            seg_alt  = trib_df.loc[ni, 'Altitude']
            seg_slp  = trib_df.loc[ni, 'Slope']
            matched  = trib_key
        else:
            # Spatial fall-back: nearest any segment
            dists_all = rn.geometry.distance(row.geometry)
            ni        = dists_all.idxmin()
            seg       = rn.loc[ni]
            trib_df   = trib_cum[seg['base_name']]
            cum_km    = station_river_dist(row.geometry, trib_df)
            seg_alt   = seg['Altitude']
            seg_slp   = seg['Slope']
            matched   = seg['base_name']

        results.append({
            'ref':          row.ref,
            'river_cum_km': round(cum_km, 3),
            'matched_trib': matched,
            'seg_altitude': round(seg_alt, 3),
            'seg_slope':    round(seg_slp, 6),
        })

    snap_df = pd.DataFrame(results)
    return stations.merge(snap_df, on='ref')


# ── Flood threshold metadata ────────────────────────────────────────────────
def load_thresholds():
    """Load p75, p90, amax, datum from graph_metadata.json (built in prior session)."""
    meta_path = OUT_DIR / "graph_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run the original graph_builder first to produce {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    rows = []
    for s in meta['stations']:
        rows.append({
            'ref':          int(s['ref']),
            'gauge_datum':  s['gauge_datum'],
            'p75_mAOD':     s['p75_mAOD'],
            'p90_mAOD':     s['p90_mAOD'],
            'amax_med':     s['amax_med'],
        })
    return pd.DataFrame(rows)


# ── Node attribute matrix ──────────────────────────────────────────────────
def build_node_attr(stations, thresholds):
    """
    6 static features per node:
      [0] log(catchment_area_km2)
      [1] gauge_datum_m
      [2] p75_mAOD
      [3] p90_mAOD
      [4] amax_med_mAOD
      [5] is_reservoir
    """
    df = stations.merge(thresholds, on='ref')
    X = np.stack([
        np.log(df['catchment_area_km2'].values + 1),
        df['gauge_datum'].values,
        df['p75_mAOD'].values,
        df['p90_mAOD'].values,
        df['amax_med'].values,
        df['is_reservoir'].values.astype(float),
    ], axis=1).astype(np.float32)
    return X


# ── Edge construction ──────────────────────────────────────────────────────
def build_edges(stations):
    """
    Returns list of dicts with keys: src_ref, dst_ref, edge_type
    """
    ref_to_wb  = dict(zip(stations.ref, stations.water_body))
    ref_to_idx = {r: i for i, r in enumerate(stations.ref)}
    wb_to_refs = {}
    for r, wb in ref_to_wb.items():
        wb_to_refs.setdefault(wb, []).append(r)

    edges = []

    # ── Intra-tributary edges (upstream → downstream by catchment area) ────
    for wb, refs in wb_to_refs.items():
        subs = stations[stations.ref.isin(refs)].sort_values('catchment_area_km2')
        for i in range(len(subs) - 1):
            edges.append({
                'src_ref': subs.iloc[i].ref,
                'dst_ref': subs.iloc[i+1].ref,
                'edge_type': 'intra',
            })

    # ── Confluence edges ──────────────────────────────────────────────────
    for wb, dst_ref in CONFLUENCE_MAP.items():
        if wb not in wb_to_refs:
            continue
        if dst_ref not in ref_to_idx:
            continue
        # most downstream station of this tributary
        subs = stations[stations.ref.isin(wb_to_refs[wb])].sort_values('catchment_area_km2')
        src_ref = subs.iloc[-1].ref
        edges.append({
            'src_ref': src_ref,
            'dst_ref': dst_ref,
            'edge_type': 'confluence',
        })

    return edges, ref_to_idx


# ── Edge attribute matrix ──────────────────────────────────────────────────
def build_edge_attr(edges, stations):
    """
    4 features per edge:
      [0] river_dist_km   – cumulative river distance between stations (intra) or Haversine (confluence)
      [1] area_ratio       – downstream_area / upstream_area
      [2] elev_drop_m      – upstream gauge_datum - downstream gauge_datum
      [3] same_tributary   – 1.0 = intra-tributary, 0.5 = confluence
    """
    ref_to_row = {r: stations[stations.ref == r].iloc[0] for r in stations.ref}

    src_idx_list, dst_idx_list, attr_rows = [], [], []
    ref_to_idx = {r: i for i, r in enumerate(stations.ref)}

    for e in edges:
        s = ref_to_row[e['src_ref']]
        d = ref_to_row[e['dst_ref']]

        # River distance
        if e['edge_type'] == 'intra':
            dist = abs(d['river_cum_km'] - s['river_cum_km'])
            # Fallback to Haversine when both stations snap to the same segment
            # (cumulative positions nearly identical → resolution lost)
            hav = haversine_km(s.lat, s.lon, d.lat, d.lon)
            if dist < hav * 0.5:   # river dist implausibly shorter than straight line
                dist = hav
        else:
            dist = haversine_km(s.lat, s.lon, d.lat, d.lon)
        dist = max(dist, 0.1)  # floor at 100 m

        area_ratio  = (d['catchment_area_km2'] + 1) / (s['catchment_area_km2'] + 1)
        elev_drop   = s['gauge_datum'] - d['gauge_datum']
        same_trib   = 1.0 if e['edge_type'] == 'intra' else 0.5

        src_idx_list.append(ref_to_idx[e['src_ref']])
        dst_idx_list.append(ref_to_idx[e['dst_ref']])
        attr_rows.append([round(dist, 3), round(area_ratio, 4),
                          round(elev_drop, 3), same_trib])

    edge_index = np.array([src_idx_list, dst_idx_list], dtype=np.int64)
    edge_attr  = np.array(attr_rows, dtype=np.float32)
    return edge_index, edge_attr


# ── Visualisation ──────────────────────────────────────────────────────────
def visualise_graph(stations, edge_index, edge_attr, catchment):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    catchment_wgs = catchment.to_crs('EPSG:4326')

    for ax_i, ax in enumerate(axes):
        catchment_wgs.boundary.plot(ax=ax, color='lightblue', linewidth=1.5, linestyle='--')

        # Draw edges
        for j in range(edge_index.shape[1]):
            si = edge_index[0, j]; di = edge_index[1, j]
            s = stations.iloc[si]; d = stations.iloc[di]
            same_trib = edge_attr[j, 3]
            ax.annotate('', xy=(d.lon, d.lat), xytext=(s.lon, s.lat),
                        arrowprops=dict(arrowstyle='->', color='steelblue' if same_trib == 1.0 else 'coral',
                                        lw=1.4 if same_trib == 1.0 else 1.0))
            if ax_i == 1:
                mx, my = (s.lon + d.lon)/2, (s.lat + d.lat)/2
                ax.text(mx, my, f'{edge_attr[j,0]:.1f}', fontsize=5.5, ha='center', color='darkblue')

        # Draw nodes
        for _, row in stations.iterrows():
            c = 'firebrick' if row.is_reservoir else 'steelblue'
            ax.scatter(row.lon, row.lat, s=60, color=c, zorder=5)
            if ax_i == 0:
                ax.text(row.lon + 0.005, row.lat, row['name'], fontsize=5.5)

        ax.set_title(['Station labels', 'Edge distances (km)'][ax_i], fontsize=10)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    blue_p  = mpatches.Patch(color='steelblue',  label='Intra-tributary edge')
    coral_p = mpatches.Patch(color='coral',       label='Confluence edge')
    red_p   = mpatches.Patch(color='firebrick',   label='Reservoir node')
    fig.legend(handles=[blue_p, coral_p, red_p], loc='lower center', ncol=3, fontsize=8)
    fig.suptitle(f'Lee Catchment GNN Graph  ({stations.shape[0]} nodes, {edge_index.shape[1]} edges)\n'
                 f'Real river distances from OPW RiverNetwork shapefile', fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ── Main ───────────────────────────────────────────────────────────────────
def build_graph(save=True, plot=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading stations …")
    stations = load_stations()

    print("Loading river network …")
    rn = load_river_network()
    trib_cum = build_trib_cumulative(rn)

    print("Snapping stations to river segments …")
    stations = snap_stations(stations, rn, trib_cum)

    print("Loading flood thresholds …")
    thresholds = load_thresholds()

    print("Building node attributes …")
    node_attr = build_node_attr(stations, thresholds)

    # Merge thresholds into stations so edge builder has gauge_datum etc.
    stations = stations.merge(thresholds, on='ref')

    print("Building edges …")
    edges, ref_to_idx = build_edges(stations)

    print("Building edge attributes …")
    edge_index, edge_attr = build_edge_attr(edges, stations)

    print(f"\n✓ Graph: {len(stations)} nodes, {edge_index.shape[1]} edges")
    print(f"  River dist range: [{edge_attr[:,0].min():.1f}, {edge_attr[:,0].max():.1f}] km")
    print(f"  Elev drop range:  [{edge_attr[:,2].min():.1f}, {edge_attr[:,2].max():.1f}] m")

    if save:
        np.save(OUT_DIR / 'edge_index.npy', edge_index)
        np.save(OUT_DIR / 'edge_attr.npy',  edge_attr)
        np.save(OUT_DIR / 'node_attr.npy',  node_attr)

        # Rebuild metadata JSON
        stations_list = []
        for _, row in stations.iterrows():
            thr = thresholds[thresholds.ref == row.ref].iloc[0]
            stations_list.append({
                'ref': str(row.ref), 'name': row['name'], 'water_body': row.water_body,
                'catchment_area_km2': row.catchment_area_km2,
                'lat': row.lat, 'lon': row.lon,
                'gauge_datum': thr.gauge_datum, 'p75_mAOD': thr.p75_mAOD,
                'p90_mAOD': thr.p90_mAOD, 'amax_med': thr.amax_med,
                'is_reservoir': int(row.is_reservoir),
                'river_cum_km': row.river_cum_km, 'matched_trib': row.matched_trib,
            })
        edge_list = []
        ref_list = list(stations.ref)
        for j in range(edge_index.shape[1]):
            si, di = edge_index[0,j], edge_index[1,j]
            edge_list.append({
                'src_ref': str(ref_list[si]), 'dst_ref': str(ref_list[di]),
                'river_dist_km': float(edge_attr[j,0]), 'area_ratio': float(edge_attr[j,1]),
                'elev_drop_m': float(edge_attr[j,2]), 'same_tributary': float(edge_attr[j,3]),
            })
        meta = {
            'num_nodes': int(len(stations)),
            'num_edges': int(edge_index.shape[1]),
            'node_feature_dim': int(node_attr.shape[1]),
            'edge_feature_dim': int(edge_attr.shape[1]),
            'reservoir_refs': RESERVOIR_REFS,
            'dropped_refs':   DROPPED_REFS,
            'confluence_map': {k: str(v) for k, v in CONFLUENCE_MAP.items()},
            'node_feature_names': ['log_catchment_area', 'gauge_datum_m', 'p75_mAOD',
                                   'p90_mAOD', 'amax_med_mAOD', 'is_reservoir'],
            'edge_feature_names': ['river_dist_km', 'area_ratio', 'elev_drop_m', 'same_tributary'],
            'stations': stations_list,
            'edges': edge_list,
        }
        with open(OUT_DIR / 'graph_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved to {OUT_DIR}")

    if plot:
        print("Building visualisation …")
        catchment = gpd.read_file(CATCHMENT_SHP).to_crs('EPSG:4326')
        fig = visualise_graph(stations, edge_index, edge_attr, catchment)
        fig.savefig(OUT_DIR / 'graph_viz.png', dpi=150, bbox_inches='tight')
        print("  Saved graph_viz.png")

    return edge_index, edge_attr, node_attr, stations


if __name__ == '__main__':
    build_graph()
