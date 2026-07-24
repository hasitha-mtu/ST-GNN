"""
plot_hand_edges.py — Visualize precomputed HAND candidate edges on the Lee
catchment gauge network (ITM coordinates), optionally with a DEM hillshade
background for terrain context.

Uses nodes.csv (uploaded) for node geography, edges.csv for the permanent
river-network backbone (directed, downstream), and the HAND edge table
transcribed directly from precompute_hand_edges.py's console output.

DEM hillshade: pass --dem pointing at the ITM-reprojected DEM
(dataset/dem/COP-DEM-30m_itm.tif — the file precompute_hand_edges.py
already produced and cached on your machine). If rasterio isn't installed,
or --dem isn't supplied / doesn't exist, the script falls back to a plain
white background rather than failing.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
import numpy as np
import pandas as pd

# ── HAND edge table, transcribed directly from the precompute_hand_edges.py
#    console output (14 undirected pairs, from the 28 directed edges saved) ──
HAND_PAIRS = [
    # (src_ref, dst_ref, dist_km, hand_thr_m, z_saddle_m)
    (19056, 19058, 1.38,  9.595, 68.10),
    (19056, 19059, 1.63, 16.267, 75.26),
    (19057, 19163, 1.98,  8.579, 58.13),
    (19057, 19161, 1.64,  3.185, 10.71),
    (19058, 19059, 1.43,  1.074, 15.81),
    (19058, 19113, 3.65,  7.821, 72.60),
    (19058, 19102, 3.56, 16.186, 80.97),
    (19059, 19161, 3.36,  2.142, 14.85),
    (19059, 19160, 3.84,  3.257, 52.81),
    (19114, 19113, 2.18,  7.724, 17.49),
    (19114, 19102, 2.18,  1.684,  6.01),
    (19113, 19102, 0.15,  0.897,  5.29),
    (19103, 19094, 2.23,  2.945, 21.45),
    (19094, 19109, 2.16,  8.206, 80.86),
]

RESERVOIR_FALLBACK_REFS = {19095, 19094}


def add_hillshade(ax, dem_path: Path):
    """
    Load an ITM-projected DEM and render it as a grayscale hillshade
    background. Returns the data extent [xmin, xmax, ymin, ymax] on
    success, or None if the DEM couldn't be loaded (caller should proceed
    without terrain context rather than fail).
    """
    try:
        import rasterio
    except ImportError:
        print("  rasterio not installed — skipping hillshade "
              "(pip install rasterio to enable it). Proceeding without terrain.")
        return None

    if not dem_path.exists():
        print(f"  DEM not found at {dem_path} — proceeding without terrain.")
        return None

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        bounds = src.bounds
        nodata = src.nodata

    if nodata is not None:
        dem[dem == nodata] = np.nan
    # COP-DEM sometimes uses large negative sentinels instead of a proper
    # nodata tag — guard against that too.
    dem[dem < -100] = np.nan

    ls = LightSource(azdeg=315, altdeg=45)
    # LightSource.hillshade doesn't accept NaN — fill with the DEM's own
    # mean for shading purposes only (doesn't affect anything else drawn).
    dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    shaded = ls.hillshade(dem_filled, vert_exag=1.5, dx=30, dy=30)
    # Re-mask true nodata areas so they render white, not a fake flat shade
    shaded = np.where(np.isnan(dem), np.nan, shaded)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    ax.imshow(shaded, cmap="gray", extent=extent, origin="upper",
              vmin=0, vmax=1, zorder=0, alpha=0.85)
    print(f"  Hillshade rendered from {dem_path.name}  "
          f"({dem.shape[1]}x{dem.shape[0]} px, extent={extent})")
    return extent


def main(nodes_path: Path, edges_path: Path | None, dem_path: Path | None,
         out_path: Path):
    nodes = pd.read_csv(nodes_path)
    nodes["ref"] = nodes["ref"].astype(int)
    pos = {int(r.ref): (r.easting_itm, r.northing_itm) for r in nodes.itertuples()}
    name = {int(r.ref): r.name for r in nodes.itertuples()}

    fig, ax = plt.subplots(figsize=(20, 18))

    dem_extent = None
    if dem_path is not None:
        dem_extent = add_hillshade(ax, dem_path)

    # ── HAND candidate edges — undirected, colour-coded by z_saddle_m ──
    z_vals = [z for _, _, _, _, z in HAND_PAIRS]
    norm = plt.Normalize(vmin=min(z_vals), vmax=max(z_vals))
    cmap = plt.cm.YlOrRd  # low saddle (easy connect) -> pale, high -> dark red

    for src, dst, dist_km, hand_thr, z_saddle in HAND_PAIRS:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        color = cmap(norm(z_saddle))
        ax.plot([x1, x2], [y1, y2], "--", color=color, linewidth=2.2,
                alpha=0.85, zorder=2,
                solid_capstyle="round")
        # Label the midpoint with the saddle elevation
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate(f"{z_saddle:.0f} m OD", (mx, my),
                    fontsize=7.5, color="#333", ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    zorder=4)

    # ── Optional: permanent river-network edges (if edges.csv supplied) ──
    if edges_path is not None and edges_path.exists():
        edges = pd.read_csv(edges_path)
        if "src_ref" in edges.columns and "dst_ref" in edges.columns:
            src_col, dst_col = "src_ref", "dst_ref"
        elif "src" in edges.columns and "dst" in edges.columns:
            src_col, dst_col = "src", "dst"
        else:
            src_col, dst_col = edges.columns[0], edges.columns[1]
        for _, row in edges.iterrows():
            s, d = int(row[src_col]), int(row[dst_col])
            if s not in pos or d not in pos:
                continue
            x1, y1 = pos[s]
            x2, y2 = pos[d]
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#1f5fa8",
                                 lw=1.8, alpha=0.85,
                                 shrinkA=8, shrinkB=8),
                zorder=3,
            )
        ax.plot([], [], color="#1f5fa8", lw=1.8,
                 label="River-network edge (Class A, directed downstream)")

    # ── Nodes ──────────────────────────────────────────────────────────
    for r in nodes.itertuples():
        x, y = r.easting_itm, r.northing_itm
        ref = int(r.ref)
        if r.is_reservoir:
            marker, fc, ec, ms = "s", "#4a7fb5", "black", 110
        elif r.is_tidal:
            marker, fc, ec, ms = "^", "#2e9e83", "black", 110
        else:
            marker, fc, ec, ms = "o", "white", "black", 85
        ax.scatter(x, y, marker=marker, s=ms, facecolor=fc, edgecolor=ec,
                   linewidth=1.1, zorder=5)
        label = r.name
        if ref in RESERVOIR_FALLBACK_REFS:
            label += "  [datum fallback]"
        ax.annotate(label, (x, y), fontsize=8, xytext=(6, 6),
                    textcoords="offset points", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # ── Colorbar for z_saddle_m ──────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("z_saddle_m — corridor saddle elevation (m OD)\n"
                   "pale = connects early (low)   dark = connects late (high)")

    # ── Legend (marker types) ────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=9, label="Gauge node"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#4a7fb5",
                   markeredgecolor="black", markersize=9, label="Reservoir node"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#2e9e83",
                   markeredgecolor="black", markersize=9, label="Tidal node"),
        plt.Line2D([0], [0], linestyle="--", color="#c44e02", linewidth=2.2,
                   label="HAND candidate edge (undirected — see note)"),
    ]
    if edges_path is not None and edges_path.exists():
        handles.append(
            plt.Line2D([0], [0], color="#1f5fa8", lw=1.8,
                       label="River-network edge (directed downstream)")
        )
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)

    ax.set_xlabel("Easting (ITM, m)")
    ax.set_ylabel("Northing (ITM, m)")
    ax.set_title(
        "Lee catchment — HAND candidate edges (14 pairs, 5 km search radius)\n"
        "Dashed lines are undirected: floodplain connectivity activates "
        "symmetrically once water rises above z_saddle_m, unlike river\n"
        "edges (solid, directed) which follow real downstream channel flow.",
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # Pin the view to the node cluster, not the full DEM extent — the DEM
    # typically covers a much larger area than the 27 gauges span, and
    # imshow's extent otherwise pulls the autoscale out to the whole raster.
    pad = 2000  # metres
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=800)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=Path, default=Path("dataset/graph/nodes.csv"))
    p.add_argument("--edges", type=Path, default=Path("dataset/graph/edges.csv"),
                   help="Optional edges.csv for the river-network backbone")
    p.add_argument("--dem", type=Path,
                   default=Path("dataset/dem/COP-DEM-30m_itm.tif"),
                   help="ITM-reprojected DEM for hillshade background "
                        "(the file precompute_hand_edges.py already cached). "
                        "Pass --dem none to disable.")
    p.add_argument("--out", type=Path, default=Path("hand_edges_map.png"))
    args = p.parse_args()

    dem_arg = None if str(args.dem).lower() == "none" else args.dem
    main(args.nodes, args.edges, dem_arg, args.out)
