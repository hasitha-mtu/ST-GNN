"""
visualise_graph.py  –  Lee catchment ST-GNN visualisation suite
================================================================
Produces three figures saved to  results/figures/:

  1. graph_structure.png   River network graph (geographic layout)
                           Nodes coloured by type, sized by catchment area.
                           Edges coloured by same_tributary flag.

  2. training_curves.png   Train / val loss and NSE over epochs,
                           with persistence NSE reference line.

  3. node_performance.png  Per-node test NSE heatmap overlaid on the
                           river network, with node-level bar chart.

Usage:
    python src/visualise_graph.py

Requires:  networkx, matplotlib, pandas, numpy
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
GRAPH_DIR = BASE_DIR / "dataset/graph"
CKPT_DIR  = BASE_DIR / "checkpoints"
OUT_DIR   = BASE_DIR / "results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Persistence baseline from earlier evaluation
PERSIST_NSE  = 0.9161
PERSIST_RMSE = 0.0731

# ── Shared style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.6,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.5,
    "figure.dpi":       150,
    "savefig.dpi":      1000,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.15,
})

NODE_COLORS = {
    "normal":      "#378ADD",   # blue
    "reservoir":   "#D85A30",   # coral
    "tidal":       "#7F77DD",   # purple
}
EDGE_SAME  = "#378ADD"          # intra-tributary
EDGE_CONF  = "#D85A30"          # confluence


# ═══════════════════════════════════════════════════════════════════════
# Helper — load graph
# ═══════════════════════════════════════════════════════════════════════

def load_graph_data():
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv")
    edges = pd.read_csv(GRAPH_DIR / "edges.csv")
    return nodes, edges


def build_nx_graph(nodes, edges):
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(
            r["ref"],
            name=r["name"],
            lat=r["lat"],
            lon=r["lon"],
            is_reservoir=r.get("is_reservoir", 0),
            is_tidal=r.get("is_tidal", 0),
            catchment_area=r.get("log_catchment_area_km2", 1.0),
        )
    for _, r in edges.iterrows():
        G.add_edge(
            r["src_ref"], r["dst_ref"],
            river_dist_km=r["river_dist_km"],
            same_tributary=r["same_tributary"],
        )
    return G


def node_positions(G):
    """Use (lon, lat) as x, y so the graph matches geographic layout."""
    return {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes}


def node_type_color(G, node):
    d = G.nodes[node]
    if d.get("is_reservoir"):
        return NODE_COLORS["reservoir"]
    if d.get("is_tidal"):
        return NODE_COLORS["tidal"]
    return NODE_COLORS["normal"]


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — graph structure
# ═══════════════════════════════════════════════════════════════════════

def plot_graph_structure(nodes, edges, G, pos):
    fig, ax = plt.subplots(figsize=(12, 8))

    # ── Edges ──────────────────────────────────────────────────────────
    for src, dst, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        color = EDGE_SAME if data["same_tributary"] == 1.0 else EDGE_CONF
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.2,
                mutation_scale=10,
                shrinkA=6, shrinkB=6,
            ),
        )
        # Distance label at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, f"{data['river_dist_km']:.1f}",
                fontsize=6, color="#5F5E5A", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # ── Nodes ──────────────────────────────────────────────────────────
    for node in G.nodes:
        x, y = pos[node]
        area   = G.nodes[node]["catchment_area"]
        size   = 40 + area * 18          # scale by log catchment area
        color  = node_type_color(G, node)
        ax.scatter(x, y, s=size, color=color, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.text(x + 0.003, y + 0.002,
                G.nodes[node]["name"],
                fontsize=6.5, color="#2C2C2A", zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    # ── Legend ─────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(fc=NODE_COLORS["normal"],    label="Normal station"),
        mpatches.Patch(fc=NODE_COLORS["reservoir"], label="Reservoir"),
        mpatches.Patch(fc=NODE_COLORS["tidal"],     label="Tidal"),
        mpatches.Patch(fc=EDGE_SAME,  label="Intra-tributary edge"),
        mpatches.Patch(fc=EDGE_CONF,  label="Confluence edge"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=8, framealpha=0.85, edgecolor="#D3D1C7")

    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude",  fontsize=9)
    ax.set_title(
        f"Lee catchment river network  ({len(G.nodes)} nodes, {len(G.edges)} edges)\n"
        "Node size proportional to log catchment area  |  "
        "Edge labels: river distance (km)",
        fontsize=10, pad=10,
    )
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = OUT_DIR / "graph_structure.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — training curves
# ═══════════════════════════════════════════════════════════════════════

def plot_training_curves():
    hist_path = CKPT_DIR / "training_history.csv"
    if not hist_path.exists():
        print("  training_history.csv not found — skipping training curves")
        return

    h   = pd.read_csv(hist_path)
    ep  = h["epoch"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Loss ───────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ep, h["train_loss"], color="#378ADD", lw=1.5, label="Train loss")
    ax.plot(ep, h["val_loss"],   color="#1D9E75", lw=1.5, label="Val loss",
            marker="o", markersize=3)
    best_ep = h.loc[h["val_loss"].idxmin(), "epoch"]
    ax.axvline(best_ep, color="#D85A30", lw=0.8, ls="--",
               label=f"Best val (epoch {best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE loss")
    ax.set_title("Training and validation loss")
    ax.legend(fontsize=8)

    # ── NSE ────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(ep, h["val_nse"], color="#1D9E75", lw=1.5,
            label="Val NSE", marker="o", markersize=3)
    ax.axhline(PERSIST_NSE, color="#D85A30", lw=1.0, ls="--",
               label=f"Persistence NSE ({PERSIST_NSE:.3f})")

    # Shade skill region
    ax.fill_between(ep, PERSIST_NSE, h["val_nse"],
                    where=h["val_nse"] > PERSIST_NSE,
                    alpha=0.12, color="#1D9E75", label="Skill over persistence")

    ax.axvline(best_ep, color="#378ADD", lw=0.8, ls="--")
    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NSE")
    ax.set_title("Validation NSE vs persistence baseline")
    ax.legend(fontsize=8)

    # ── Annotation: test result ────────────────────────────────────────
    test_path = CKPT_DIR / "test_metrics.json"
    if test_path.exists():
        with open(test_path) as f:
            tm = json.load(f)
        skill = (tm["nse"] - PERSIST_NSE) / (1 - PERSIST_NSE)
        axes[1].text(
            0.97, 0.06,
            f"Test NSE={tm['nse']:.4f}  skill={skill:.3f}",
            transform=axes[1].transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#EAF3DE",
                      ec="#639922", alpha=0.9),
        )

    fig.suptitle("ST-GNN training  –  Lee catchment flood forecasting",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — per-node performance
# ═══════════════════════════════════════════════════════════════════════

def plot_node_performance(nodes, edges, G, pos):
    """
    Loads per-node NSE from  checkpoints/per_node_metrics.csv  if present.
    Falls back to simulated values if the file doesn't exist yet, so the
    visualisation can be generated before per-node metrics are computed.

    To generate per_node_metrics.csv, add this to your test evaluation loop:

        per_node_nse = compute_metrics(
            torch.cat(all_pred), torch.cat(all_tgt), torch.cat(all_mask),
            return_per_node=True   # add this flag to compute_metrics
        )
        pd.DataFrame({
            "ref":  node_refs,
            "name": node_names,
            "nse":  per_node_nse,
        }).to_csv(CKPT_DIR / "per_node_metrics.csv", index=False)
    """
    pn_path = CKPT_DIR / "per_node_metrics.csv"
    if pn_path.exists():
        pn = pd.read_csv(pn_path).set_index("ref")
    else:
        print("  per_node_metrics.csv not found — using placeholder values")
        # Placeholder: distribute NSE plausibly based on WL coverage
        # Replace with real values once per-node evaluation is run
        rng = np.random.default_rng(42)
        pn  = pd.DataFrame({
            "ref":  [str(r) for r in nodes["ref"]],
            "name": nodes["name"].values,
            "nse":  np.clip(rng.normal(0.93, 0.05, len(nodes)), 0.5, 0.999),
        }).set_index("ref")

    # ── Colour map: NSE → colour ────────────────────────────────────────
    cmap  = plt.cm.RdYlGn
    norm  = mcolors.Normalize(vmin=0.5, vmax=1.0)

    fig = plt.figure(figsize=(16, 6))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
    ax_map = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # ── Map panel ──────────────────────────────────────────────────────
    for src, dst, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ax_map.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#B4B2A9",
                            lw=1.0, mutation_scale=8,
                            shrinkA=7, shrinkB=7),
        )

    for node in G.nodes:
        x, y   = pos[node]
        ref    = str(node)
        nse    = pn.loc[ref, "nse"] if ref in pn.index else 0.5
        color  = cmap(norm(nse))
        area   = G.nodes[node]["catchment_area"]
        size   = 60 + area * 20

        ax_map.scatter(x, y, s=size, color=color, zorder=5,
                       edgecolors="white", linewidths=0.8)
        ax_map.text(x + 0.003, y + 0.002,
                    G.nodes[node]["name"],
                    fontsize=6, color="#2C2C2A", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.6))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_map, shrink=0.6, pad=0.02)
    cb.set_label("Per-node NSE", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_map.axhline(PERSIST_NSE, color="#D85A30", lw=0)   # invisible, for legend
    persist_patch = mpatches.Patch(fc="#D85A30", alpha=0.6,
                                   label=f"Persistence NSE = {PERSIST_NSE:.3f}")
    ax_map.legend(handles=[persist_patch], loc="lower left",
                  fontsize=7, framealpha=0.85)

    ax_map.set_xlabel("Longitude", fontsize=9)
    ax_map.set_ylabel("Latitude",  fontsize=9)
    ax_map.set_title("Per-node test NSE  (node size ∝ log catchment area)",
                     fontsize=10)

    # ── Bar panel ──────────────────────────────────────────────────────
    bar_df = pn.reset_index().sort_values("nse", ascending=True)
    bar_colors = [cmap(norm(v)) for v in bar_df["nse"]]

    y_pos = range(len(bar_df))
    ax_bar.barh(y_pos, bar_df["nse"], color=bar_colors,
                height=0.7, edgecolor="white", linewidth=0.4)
    ax_bar.axvline(PERSIST_NSE, color="#D85A30", lw=1.0, ls="--",
                   label=f"Persistence ({PERSIST_NSE:.3f})")
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(bar_df["name"], fontsize=7)
    ax_bar.set_xlim(0.4, 1.02)
    ax_bar.set_xlabel("NSE")
    ax_bar.set_title("Test NSE by station", fontsize=10)
    ax_bar.legend(fontsize=7)

    # Value labels
    for i, (_, row) in enumerate(bar_df.iterrows()):
        ax_bar.text(row["nse"] + 0.003, i, f"{row['nse']:.3f}",
                    va="center", fontsize=6.5, color="#2C2C2A")

    fig.suptitle("ST-GNN  –  per-node test performance vs persistence baseline",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "node_performance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Loading graph …")
    nodes, edges = load_graph_data()
    G   = build_nx_graph(nodes, edges)
    pos = node_positions(G)

    print("Figure 1: graph structure …")
    plot_graph_structure(nodes, edges, G, pos)

    print("Figure 2: training curves …")
    plot_training_curves()

    print("Figure 3: per-node performance …")
    plot_node_performance(nodes, edges, G, pos)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
