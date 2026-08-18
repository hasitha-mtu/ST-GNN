"""
plot_model_graphs.py  —  Per-model edge/node topology visualisation
================================================================================
Produces one map per ST-GNN family model showing exactly which edge sets
that model's message passing actually uses — river edges (always), HAND
cross-tributary edges (HANDEdge/SoilGate/DFC-GNN Unified), and backwater
edges (BackwaterEdge) — plus a combined overview grid for a single
paper/thesis figure comparing all variants side by side.

Node/colour conventions match graph_builder.py's own visualise_graph()
for consistency with any existing catchment map figures:
  reservoir node : firebrick
  tidal node     : mediumpurple
  normal node    : steelblue
River edges: solid, styled by same_tributary (steelblue = intra-tributary,
coral = confluence) — same convention as graph_builder.py.
HAND edges: dashed teal.
Backwater edges: dotted, arrowed, colour-highlighted — drawn with a
slightly heavier line since there are only ever a handful of them and
they're the point of the figure when present.

MODEL_EDGE_COMPOSITION below is the one thing to check/edit before
running: it says which edge sets each of your "6 ST-GNN models" uses.
Adjust the dict (e.g. swap dfc_gnn_unified for dfc_gnn, or add/remove
entries) if your actual model roster differs — this is intentionally a
plain, editable Python dict rather than something inferred from a
checkpoint, since which edges a model uses is an architecture fact, not
a data-derived one.

Usage:
    python plot_model_graphs.py                  # individual + combined
    python plot_model_graphs.py --model st_gnn_hand_edge   # just one
    python plot_model_graphs.py --combined-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR  = Path(__file__).resolve().parent.parent
GRAPH_DIR = BASE_DIR / "dataset" / "graph"
OUT_DIR   = BASE_DIR / "results" / "figures" / "model_graphs"

NODES_CSV     = GRAPH_DIR / "nodes.csv"
EDGES_CSV     = GRAPH_DIR / "edges.csv"
HAND_NPZ      = GRAPH_DIR / "hand_edges.npz"
BACKWATER_NPZ = GRAPH_DIR / "backwater_edges.npz"

COLOR_RESERVOIR = "firebrick"
COLOR_TIDAL     = "mediumpurple"
COLOR_NORMAL    = "steelblue"
COLOR_RIVER_INTRA = "steelblue"
COLOR_RIVER_CONFL = "coral"
COLOR_HAND      = "teal"
COLOR_BACKWATER = "darkorange"

# Edit this if your actual "6 ST-GNN models" differ.
MODEL_EDGE_COMPOSITION = {
    "st_gnn": {
        "river": True, "hand": False, "backwater": False,
        "label": "ST-GNN (static)",
        "gate_desc": "No dynamic gating — static river topology only",
    },
    "st_gnn_dyn_edge": {
        "river": True, "hand": False, "backwater": False,
        "label": "ST-GNN DynEdge",
        "gate_desc": "Same river topology; discharge-based conductance\nreweights edges, does not add/remove any",
    },
    "st_gnn_hand_edge": {
        "river": True, "hand": True, "backwater": False,
        "label": "ST-GNN HAND",
        "gate_desc": "HAND edges: reactive stage-vs-saddle sigmoid gate",
    },
    "st_gnn_soil_gate": {
        "river": True, "hand": True, "backwater": False,
        "label": "ST-GNN Soil Gate",
        "gate_desc": "HAND edges: anticipatory catchment-saturation gate\n(same topology as HAND Edge, different activation)",
    },
    "st_gnn_backwater_edge": {
        "river": True, "hand": False, "backwater": True,
        "label": "ST-GNN Backwater",
        "gate_desc": "Backwater edges: bridge-stage-vs-p90 permission gate",
    },
    "dfc_gnn_unified": {
        "river": True, "hand": True, "backwater": False,
        "label": "PC-DFC-GNN",
        "gate_desc": "River: hard elevation-differential gate\nHAND: soft activation (same topology as HAND Edge)",
    },
}


def load_graph_data():
    nd = pd.read_csv(NODES_CSV)
    ed = pd.read_csv(EDGES_CSV)
    hand = np.load(HAND_NPZ, allow_pickle=True) if HAND_NPZ.exists() else None
    bw   = np.load(BACKWATER_NPZ, allow_pickle=True) if BACKWATER_NPZ.exists() else None
    return nd, ed, hand, bw


def _node_color(row) -> str:
    if row.get("is_reservoir"):
        return COLOR_RESERVOIR
    if row.get("is_tidal"):
        return COLOR_TIDAL
    return COLOR_NORMAL


def _draw_nodes(ax, nd: pd.DataFrame, label_nodes: bool = True):
    for _, s in nd.iterrows():
        ax.scatter(s["lon"], s["lat"], s=70, color=_node_color(s),
                  zorder=8, edgecolor="black", linewidth=0.4)
        if label_nodes:
            ax.text(s["lon"] + 0.004, s["lat"], s["name"], fontsize=7)


def _draw_river_edges(ax, nd: pd.DataFrame, ed: pd.DataFrame):
    node_pos = nd.set_index("node_idx")[["lon", "lat"]]
    for _, e in ed.iterrows():
        s = node_pos.loc[int(e["src_idx"])]
        d = node_pos.loc[int(e["dst_idx"])]
        color = COLOR_RIVER_INTRA if e.get("same_tributary", 1.0) == 1.0 else COLOR_RIVER_CONFL
        ax.annotate("", xy=(d["lon"], d["lat"]), xytext=(s["lon"], s["lat"]),
                   arrowprops=dict(arrowstyle="->", color=color, lw=1.1, alpha=0.85),
                   zorder=4)


def _draw_hand_edges(ax, nd: pd.DataFrame, hand):
    node_pos = nd.set_index("node_idx")[["lon", "lat"]]
    for i in range(len(hand["src"])):
        s = node_pos.loc[int(hand["src"][i])]
        d = node_pos.loc[int(hand["dst"][i])]
        ax.annotate("", xy=(d["lon"], d["lat"]), xytext=(s["lon"], s["lat"]),
                   arrowprops=dict(arrowstyle="->", color=COLOR_HAND, lw=1.3,
                                  linestyle="dashed", alpha=0.75),
                   zorder=5)


def _draw_backwater_edges(ax, nd: pd.DataFrame, bw):
    node_pos = nd.set_index("node_idx")[["lon", "lat"]]
    for i in range(len(bw["src"])):
        s = node_pos.loc[int(bw["src"][i])]
        d = node_pos.loc[int(bw["dst"][i])]
        ax.annotate("", xy=(d["lon"], d["lat"]), xytext=(s["lon"], s["lat"]),
                   arrowprops=dict(arrowstyle="->", color=COLOR_BACKWATER, lw=2.0,
                                  linestyle="dotted"),
                   zorder=6)


def plot_one_model(model_key: str, nd, ed, hand, bw, out_dir: Path,
                   label_nodes: bool = True) -> Path:
    comp = MODEL_EDGE_COMPOSITION[model_key]

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("white")

    _draw_nodes(ax, nd, label_nodes=label_nodes)

    n_river = n_hand = n_bw = 0
    if comp["river"]:
        _draw_river_edges(ax, nd, ed)
        n_river = len(ed)
    if comp["hand"] and hand is not None:
        _draw_hand_edges(ax, nd, hand)
        n_hand = len(hand["src"])
    if comp["backwater"] and bw is not None:
        _draw_backwater_edges(ax, nd, bw)
        n_bw = len(bw["src"])

    handles = [
        mpatches.Patch(color=COLOR_NORMAL,    label="Normal node"),
        mpatches.Patch(color=COLOR_RESERVOIR, label="Reservoir node"),
        mpatches.Patch(color=COLOR_TIDAL,     label="Tidal node"),
    ]
    if comp["river"]:
        handles += [
            mpatches.Patch(color=COLOR_RIVER_INTRA, label=f"River, intra-tributary"),
            mpatches.Patch(color=COLOR_RIVER_CONFL,  label=f"River, confluence"),
        ]
    if comp["hand"]:
        handles.append(mpatches.Patch(color=COLOR_HAND, label=f"HAND (n={n_hand}, gated)"))
    if comp["backwater"]:
        handles.append(mpatches.Patch(color=COLOR_BACKWATER, label=f"Backwater (n={n_bw}, gated)"))

    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    edge_summary = f"river={n_river}"
    if comp["hand"]:      edge_summary += f"  hand={n_hand}"
    if comp["backwater"]: edge_summary += f"  backwater={n_bw}"
    ax.set_title(
        f"{comp['label']}\n{edge_summary}\n{comp['gate_desc']}",
        fontsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}_graph.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_combined_grid(nd, ed, hand, bw, out_dir: Path) -> Path:
    """One compact 2x3 (or however many models) overview grid, no node
    labels — for a single comparison figure in the paper/thesis."""
    models = list(MODEL_EDGE_COMPOSITION.keys())
    ncols = 3
    nrows = -(-len(models) // ncols)   # ceil

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(-1)

    node_pos = nd.set_index("node_idx")[["lon", "lat"]]

    for ax, model_key in zip(axes, models):
        comp = MODEL_EDGE_COMPOSITION[model_key]
        for _, s in nd.iterrows():
            ax.scatter(s["lon"], s["lat"], s=25, color=_node_color(s),
                      zorder=8, edgecolor="black", linewidth=0.3)
        n_river = n_hand = n_bw = 0
        if comp["river"]:
            for _, e in ed.iterrows():
                s = node_pos.loc[int(e["src_idx"])]; d = node_pos.loc[int(e["dst_idx"])]
                color = COLOR_RIVER_INTRA if e.get("same_tributary", 1.0) == 1.0 else COLOR_RIVER_CONFL
                ax.annotate("", xy=(d["lon"], d["lat"]), xytext=(s["lon"], s["lat"]),
                           arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7))
            n_river = len(ed)
        if comp["hand"] and hand is not None:
            for i in range(len(hand["src"])):
                s = node_pos.loc[int(hand["src"][i])]; d = node_pos.loc[int(hand["dst"][i])]
                ax.annotate("", xy=(d["lon"], d["lat"]), xytext=(s["lon"], s["lat"]),
                           arrowprops=dict(arrowstyle="-", color=COLOR_HAND, lw=1.0,
                                          linestyle="dashed", alpha=0.8))
            n_hand = len(hand["src"])
        if comp["backwater"] and bw is not None:
            for i in range(len(bw["src"])):
                s = node_pos.loc[int(bw["src"][i])]; d = node_pos.loc[int(bw["dst"][i])]
                ax.annotate("", xy=(d["lon"], d["lat"]), xytext=(s["lon"], s["lat"]),
                           arrowprops=dict(arrowstyle="->", color=COLOR_BACKWATER, lw=2.2,
                                          linestyle="dotted"))
            n_bw = len(bw["src"])

        title = f"{comp['label']}\nriver={n_river}"
        if comp["hand"]:      title += f"  hand={n_hand}"
        if comp["backwater"]: title += f"  backwater={n_bw}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(models):]:
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=COLOR_NORMAL,    label="Normal node"),
        mpatches.Patch(color=COLOR_RESERVOIR, label="Reservoir node"),
        mpatches.Patch(color=COLOR_TIDAL,     label="Tidal node"),
        mpatches.Patch(color=COLOR_RIVER_INTRA, label="River edge"),
        mpatches.Patch(color=COLOR_HAND,      label="HAND edge (gated)"),
        mpatches.Patch(color=COLOR_BACKWATER, label="Backwater edge (gated)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ST-GNN family: edge topology by model", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_models_graph_grid.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Plot per-model ST-GNN graph topology")
    p.add_argument("--model", type=str, choices=list(MODEL_EDGE_COMPOSITION),
                   help="Plot just one model")
    p.add_argument("--combined-only", action="store_true",
                   help="Only produce the combined overview grid")
    p.add_argument("--no-labels", action="store_true",
                   help="Skip node name labels on individual plots")
    args = p.parse_args()

    nd, ed, hand, bw = load_graph_data()
    print(f"Loaded: {len(nd)} nodes, {len(ed)} river edges, "
          f"{len(hand['src']) if hand is not None else 0} HAND edges, "
          f"{len(bw['src']) if bw is not None else 0} backwater edges")

    if args.model:
        out = plot_one_model(args.model, nd, ed, hand, bw, OUT_DIR,
                             label_nodes=not args.no_labels)
        print(f"Saved {out}")
        return

    if not args.combined_only:
        for model_key in MODEL_EDGE_COMPOSITION:
            out = plot_one_model(model_key, nd, ed, hand, bw, OUT_DIR,
                                 label_nodes=not args.no_labels)
            print(f"Saved {out}")

    out = plot_combined_grid(nd, ed, hand, bw, OUT_DIR)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
