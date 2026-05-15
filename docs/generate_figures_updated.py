"""
generate_figures.py
===================
Generates all five data figures for the SLR paper:

    Figure 2  – Publication year trend + regional distribution
    Figure 3  – Spatial × Temporal architecture matrix
    Figure 4  – Graph construction (node type, edge type, graph size)
    Figure 5  – Physical constraint category distribution + temporal trend
    Figure 6  – Static and dynamic feature inventory

Usage
-----
    python generate_figures.py                          # uses default path
    python generate_figures.py --excel my_data.xlsx    # custom path
    python generate_figures.py --format png --dpi 300  # raster output

Output
------
All figures are saved in the same directory as the script (or the directory
specified via --outdir).  Both PDF (for LaTeX) and PNG (for preview) are
produced by default.

Requirements
------------
    pip install pandas openpyxl matplotlib numpy
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Journal-quality typography
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# Colour palette (colour-blind-friendly)
BLUE   = "#2c6fad"
ORANGE = "#e05c2a"
GREEN  = "#4caf50"
PURPLE = "#9c27b0"
AMBER  = "#ff9800"
GREY   = "#607d8b"
RED    = "#d32f2f"
TEAL   = "#009688"
PINK   = "#e91e63"

PALETTE = [BLUE, ORANGE, GREEN, PURPLE, AMBER, GREY, TEAL, PINK, RED]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading & cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_excel(path: str) -> dict:
    """
    Load all sheets from the SLR Excel workbook.

    Returns a dict keyed by sheet name, values are DataFrames with
    whitespace-normalised string columns.
    """
    path = Path(path)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    print(f"Reading workbook: {path.name}")
    sheets = pd.read_excel(path, sheet_name=None)

    # Normalise whitespace in every string cell
    for name, df in sheets.items():
        for col in df.select_dtypes(include="object").columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .replace("nan", pd.NA)
            )
        sheets[name] = df
        print(f"  Loaded sheet '{name}': {len(df)} rows × {len(df.columns)} cols")

    return sheets


def clean_type(raw: str) -> str:
    """
    Normalise the Physical Constraints 'Type' column to a short label.
    Handles embedded newlines / extra spaces from the original spreadsheet.
    """
    if pd.isna(raw):
        return "Unknown"
    s = str(raw).replace("\n", " ").strip()
    mapping = {
        "Type I":                          "Physics Loss Constraint",
        "Type II":                         "Physical Topology",
        "Type II + Learned Graph Structure": "Topology + Data-Learned (hybrid)",
        "Data-Learned (static)":          "Data-Learned (static)",
        "Dynamic graph (physics-driven)":   "Real-Time Physical Observation",
    }
    for key, label in mapping.items():
        if key.lower() in s.lower():
            return label
    return s


# def categorise_spatial(raw: str) -> str:
#     """Map raw 'Spatial' cell to a short display label."""
#     if pd.isna(raw):
#         return "Other"
#     s = str(raw)
#     if "GraphSAGE" in s and "GAT" in s:
#         return "GraphSAGE + GAT"
#     if "GCN (diffusion" in s:
#         return "GCN (diffusion)"
#     if "multi-scale" in s.lower():
#         return "Multi-scale GNN"
#     if "GN block" in s or "Graph Network" in s:
#         return "GN block"
#     if "MLP" in s and "GCN" not in s:
#         return "MLP"
#     if "SageFormer" in s:
#         return "SageFormer"
#     if "Dense graph" in s or "Gaussian" in s:
#         return "Dense / Gaussian RBF"
#     if "GAT" in s:
#         return "GAT"
#     if "GNN" in s and "GCN" not in s and "CNN" not in s:
#         return "GNN (generic)"
#     if "CNN" in s and "GCN" in s:
#         return "CNN + GCN"
#     if "GCN" in s:
#         return "GCN"
#     return "Other"

def categorise_spatial(raw: str) -> str:
    """Map raw 'Spatial' cell to a short display label."""
    if pd.isna(raw):
        print("categorise_spatial|raw: {raw}")
        return "Other"
    s = str(raw)
    if "GraphSAGE" in s:
        return "GraphSAGE"
    if "GN block" in s or "Graph Network" in s:
        return "GN block"
    if "MLP" in s and "GCN" not in s:
        return "MLP"
    if "SageFormer" in s:
        return "SageFormer"
    if "Dense graph" in s or "Gaussian" in s:
        return "Dense / Gaussian RBF"
    if "GAT" in s:
        return "GAT"
    if "GNN" in s:
        return "GNN"
    if "GCN" in s:
        return "GCN"
    if "ChebNet" in s:
        return "ChebNet"
    print(f"categorise_spatial|s: {s}")
    # return "Other"


# def categorise_temporal(raw: str) -> str:
#     """Map raw 'Temporal' cell to a short display label."""
#     if pd.isna(raw):
#         return "Other"
#     s = str(raw)
#     if "Muskingum" in s or "Cunge" in s or "δMC" in s:
#         return "Differentiable process"
#     if "AR" == s.strip() or "Autoregressive" in s or "AR rollout" in s:
#         return "AR rollout"
#     if "FEDformer" in s or "Frequency" in s:
#         return "Transformer (FED)"
#     if "TFT" in s or "Temporal Fusion" in s:
#         return "Transformer (TFT)"
#     if "Transformer" in s:
#         return "Transformer"
#     if "SageFormer" in s:
#         return "Transformer (Sage)"
#     if "TCN" in s or "Temporal Conv" in s or "T-GCN" in s:
#         return "TCN / T-GCN"
#     if "CNN" in s:
#         return "CNN"
#     if "GRU" in s:
#         return "GRU"
#     if "LSTM" in s:
#         return "LSTM"
#     if "RNN" in s:
#         return "RNN"
#     if "Pearson" in s or "ELM" in s or "SVM" in s:
#         return "Ensemble / hybrid"
#     if "Multi-step" in s:
#         return "Multi-step function"
#     return "Other"

def categorise_temporal(raw: str) -> str:
    """Map raw 'Temporal' cell to a short display label."""
    if pd.isna(raw):
        print(f'categorise_temporal|raw: {raw}')
        return "Other"
    s = str(raw)
    if "Muskingum" in s or "Cunge" in s or "δMC" in s:
        return "Muskingum-Cunge"
    if "AR" == s.strip() or "Autoregressive" in s or "AR rollout" in s:
        return "AR rollout"
    if "FEDformer" in s or "Frequency" in s:
        return "Transformer"
    if "TFT" in s or "Temporal Fusion" in s:
        return "Transformer"
    if "Transformer" in s:
        return "Transformer"
    if "SageFormer" in s:
        return "Transformer"
    if "TCN" in s or "Temporal Conv" in s or "T-GCN" in s:
        return "TCN / T-GCN"
    if "CNN" in s:
        return "CNN"
    if "GRU" in s:
        return "GRU"
    if "LSTM" in s:
        return "LSTM"
    if "RNN" in s:
        return "RNN"
    if "MLP" in s:
        return "MLP"
    if "Residual GCN" in s:
        return "Residual GCN"
    if "ELM" in s or "SVM" in s:
        return "ELM, SVM, MLP"
    if "Multi-step" in s:
        return "Multi-step function"
    print(f'categorise_temporal|s: {s}')
    return "Other"


def categorise_node(raw: str) -> str:
    """Map raw 'Node Type' cell to a tidy category."""
    if pd.isna(raw):
        return "Other"
    s = str(raw).lower()
    if "gauge" in s or "guage" in s:     # handle the typo "Guage"
        return "Gauge station"
    if "mesh" in s or "raster" in s or "cell" in s:
        return "Mesh / raster cell"
    if "reach" in s:
        return "River reach"
    if "junction" in s:
        return "River junction"
    if "drainage" in s:
        return "Drainage intersection"
    if "census" in s:
        return "Census tract"
    if "knowledge" in s:
        return "Knowledge graph node"
    if "token" in s or "series" in s:
        return "Dynamic token"
    return "Other"


def categorise_edge(raw: str) -> str:
    """Map raw 'Edge Type' cell to a broad category."""
    if pd.isna(raw):
        print(f'categorise_temporal|raw: {raw}')
        return "Other"
    s = str(raw).lower()
    if "attention" in s and "dynamic" in s:
        return "Attention-learned"
    if "attention" in s:
        return "Attention-learned"
    if "knowledge graph" in s:
        return "Knowledge graph"
    if "correlation" in s:
        return "Statistical correlation"
    if "physical distance" in s or "flow direction" in s or "physical flow" in s or "physical interaction" in s:
        return "Physical flow direction"
    if "spatial relation" in s or "adjacency" in s or "spatial dependency" in s or "adjacent" in s:
        return "Spatial adjacency"
    if "topographic" in s or "physics-based" in s:
        return "Physics-routed"
    if "adaptive" in s or "functional" in s or "weighted" in s:
        return "Adaptive / hybrid"
    if "interdepend" in s or "inter-series" in s:
        return "Inter-series dependency"
    if "connectivity" in s or "connection" in s:
        return "Physical connectivity"
    if "river reacher" in s:
        return "River reach"
    if "distance" in s:
        return "Physical flow direction"
    print(f'categorise_temporal|s: {s}')
    return "Other"


def to_numeric_graphsize(raw) -> float:
    """
    Convert the 'Graph Size' cell to a float.
    Handles strings like '11, 7' (returns the first number),
    'Dynamic', and plain integers.
    """
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    if s.lower() == "dynamic":
        return np.nan
    # Extract first number (handles '11, 7', '1,435,271', etc.)
    nums = re.findall(r"[\d]+", s.replace(",", ""))
    return float(nums[0]) if nums else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Figure 2 – Overview (year trend + regional distribution)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_overview(df_general: pd.DataFrame, outdir: Path, formats: list):
    """
    Two-panel figure:
      (a) Annual publication count bar chart (2020-2026)
      (b) Regional distribution pie chart
    """
    print("  Generating Figure 2: Overview …")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.05)
    # fig, axes = plt.subplots(1, 2)

    # ── Panel (a): year trend ─────────────────────────────────────────────
    year_counts = (
        df_general["Year"]
        .astype(int)
        .value_counts()
        .sort_index()
    )
    all_years = range(
        int(df_general["Year"].min()),
        int(df_general["Year"].max()) + 1
    )
    year_counts = year_counts.reindex(all_years, fill_value=0)

    bars = axes[0].bar(
        year_counts.index, year_counts.values,
        color=BLUE, edgecolor="white", linewidth=0.8, width=0.6,
    )
    for bar, cnt in zip(bars, year_counts.values):
        if cnt > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(int(cnt)),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
    axes[0].set_xlabel("Publication Year")
    axes[0].set_ylabel("Number of Studies")
    # axes[0].set_title("(a)  Annual Publication Trend", fontweight="bold")
    axes[0].set_title("")
    axes[0].text(
        0.5, -0.18, "(a) Annual Publication Trend",
        transform=axes[0].transAxes,
        ha="center", va="top", fontweight="bold"
    )
    axes[0].set_xticks(list(all_years))
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_ylim(0, year_counts.max() + 3)
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel (b): regional distribution ─────────────────────────────────
    region_counts = (
        df_general["Region/Country"]
        .value_counts()
        .sort_values(ascending=False)
    )
    colors_pie = PALETTE[: len(region_counts)]
    wedges, texts, autotexts = axes[1].pie(
        region_counts.values,
        labels=None,
        autopct="%1.0f%%",
        colors=colors_pie,
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
    )
    for at in autotexts:
        at.set_fontsize(8)

    legend_labels = [
        f"{region} ({cnt})"
        for region, cnt in region_counts.items()
    ]
    axes[1].legend(
        wedges, legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    # axes[1].set_title(
    #     f"(b)  Geographic Distribution  (n = {len(df_general)})",
    #     fontweight="bold",
    # )

    axes[1].set_title("")
    axes[1].text(
        0.5, -0.18,
        f"(b) Geographic Distribution (n = {len(df_general)})",
        transform=axes[1].transAxes,
        ha="center", va="top", fontweight="bold"
    )

    fig.subplots_adjust(wspace=0.05, bottom=0.2)

    fig.tight_layout()
    _save(fig, outdir, "fig2_overview", formats)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Figure 3 – Spatial × Temporal architecture matrix
# ─────────────────────────────────────────────────────────────────────────────

def fig3_arch_matrix(df_arch: pd.DataFrame, outdir: Path, formats: list):
    """
    Bubble matrix where rows = spatial component and columns = temporal
    component.  Bubble area is proportional to the number of studies in
    each cell.
    """
    print("  Generating Figure 3: Architecture matrix …")

    df = df_arch.copy()
    df["spatial_cat"]  = df["Spatial"].apply(categorise_spatial)
    df["temporal_cat"] = df["Temporal"].apply(categorise_temporal)

    # Cross-tabulation
    cross = pd.crosstab(df["spatial_cat"], df["temporal_cat"])
    print(f'cross: {cross}')
    cross.to_csv("cross.csv")

    # Order rows/cols by total frequency (most common at top / left)
    row_order = cross.sum(axis=1).sort_values(ascending=False).index.tolist()
    col_order = cross.sum(axis=0).sort_values(ascending=False).index.tolist()
    cross = cross.loc[row_order, col_order]

    fig, ax = plt.subplots(
        figsize=(max(10, len(col_order) * 1.4), max(5, len(row_order) * 0.8))
    )

    for i, row in enumerate(cross.index):
        for j, col in enumerate(cross.columns):
            cnt = cross.loc[row, col]
            if cnt > 0:
                ax.scatter(
                    j, i,
                    s=cnt * 350,
                    c=BLUE,
                    alpha=0.75,
                    edgecolors="#1a3f6e",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.text(
                    j, i, str(int(cnt)),
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white",
                    zorder=4,
                )

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=30, ha="right")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel("Temporal Component")
    ax.set_ylabel("Spatial Component")
    # ax.set_title(
    #     "Spatial × Temporal Architecture Matrix\n"
    #     "(bubble area proportional to number of studies)",
    #     fontweight="bold",
    # )

    ax.set_title("")
    # ax.text(
    #     0.5, -0.18,
    #     "Spatial × Temporal Architecture Matrix\n"
    #     "(bubble area proportional to number of studies)",
    #     transform=ax.transAxes,
    #     ha="center", va="top", fontweight="bold"
    # )

    ax.set_xlim(-0.7, len(col_order) - 0.3)
    ax.set_ylim(-0.7, len(row_order) - 0.3)
    ax.grid(True, alpha=0.25, zorder=0)

    # Bubble-size legend
    for s_val, label in [(1, "1 study"), (3, "3 studies"), (5, "5 studies")]:
        ax.scatter([], [], s=s_val * 350, c=BLUE, alpha=0.75,
                   edgecolors="#1a3f6e", linewidth=0.8, label=label)
    # ax.legend(
    #     title="Study count",
    #     loc="upper right",
    #     frameon=True,
    #     framealpha=0.9,
    #     edgecolor="lightgrey",
    # )

    fig.tight_layout()
    _save(fig, outdir, "fig3_arch_matrix", formats)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Figure 4 – Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def fig4_graph_construction(df_gc: pd.DataFrame, outdir: Path, formats: list):
    """
    Three-panel figure:
      (a) Node type horizontal bar chart
      (b) Edge type horizontal bar chart
      (c) Graph size log-scale scatter plot
    """
    print("  Generating Figure 4: Graph construction …")

    df = df_gc.copy()
    df["node_cat"] = df["Node Type (Gauge/Mesh/River reach)"].apply(categorise_node)
    df["edge_cat"] = df["Edge Type (Physical/ Mesh/  Etc)"].apply(categorise_edge)
    df["graph_size"] = df["Graph Size"].apply(to_numeric_graphsize)

    node_counts = df["node_cat"].value_counts().sort_values()
    edge_counts = df["edge_cat"].value_counts().sort_values()

    df.to_csv("construction.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # ── Panel (a): node types ─────────────────────────────────────────────
    colours_node = PALETTE[: len(node_counts)]
    bars_a = axes[0].barh(
        node_counts.index, node_counts.values,
        color=colours_node[::-1],    # invert so most-common is at top
        edgecolor="white", linewidth=0.7,
    )
    for bar, cnt in zip(bars_a, node_counts.values):
        axes[0].text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            str(int(cnt)),
            va="center", fontsize=8, fontweight="bold",
        )
    axes[0].set_xlabel("Number of Studies")
    axes[0].set_title("(a)  Node Type", fontweight="bold")
    axes[0].set_xlim(0, node_counts.max() + 3)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel (b): edge types ─────────────────────────────────────────────
    colours_edge = PALETTE[: len(edge_counts)]
    bars_b = axes[1].barh(
        edge_counts.index, edge_counts.values,
        color=colours_edge[::-1],
        edgecolor="white", linewidth=0.7,
    )
    for bar, cnt in zip(bars_b, edge_counts.values):
        axes[1].text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            str(int(cnt)),
            va="center", fontsize=8, fontweight="bold",
        )
    axes[1].set_xlabel("Number of Studies")
    axes[1].set_title("(b)  Edge Type", fontweight="bold")
    axes[1].set_xlim(0, edge_counts.max() + 3)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel (c): graph size scatter (log scale) ─────────────────────────
    df_size = df.dropna(subset=["graph_size"]).copy()
    df_size = df_size.sort_values("graph_size").reset_index(drop=True)

    # Colour by node type (gauge vs mesh)
    point_colors = [
        ORANGE if v > 1_000 else BLUE
        for v in df_size["graph_size"]
    ]

    axes[2].scatter(
        range(len(df_size)),
        df_size["graph_size"],
        c=point_colors,
        s=55, alpha=0.85,
        edgecolors="white", linewidth=0.5,
        zorder=3,
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Graph Size (nodes, log scale)")
    axes[2].set_xlabel(f"Studies with known size  (n = {len(df_size)})")
    axes[2].set_title("(c)  Graph Size Distribution", fontweight="bold")
    axes[2].set_xticks([])
    axes[2].yaxis.grid(True, alpha=0.3, zorder=0)

    # Horizontal reference lines
    for ref_val, lbl in [(10, "10"), (100, "100"), (1_000, "1 K"),
                         (10_000, "10 K"), (1_000_000, "1 M")]:
        axes[2].axhline(ref_val, color="lightgrey", linewidth=0.6,
                        linestyle="--", zorder=0)

    legend_handles = [
        mpatches.Patch(color=BLUE,   label="≤ 1,000 nodes  (gauge-based)"),
        mpatches.Patch(color=ORANGE, label="> 1,000 nodes  (mesh-based)"),
    ]
    axes[2].legend(handles=legend_handles, frameon=True,
                   framealpha=0.9, edgecolor="lightgrey", loc="upper left")

    fig.tight_layout()
    _save(fig, outdir, "fig4_graph_construction", formats)
    plt.close(fig)



def extract_components(type_str: str) -> list:
    """
    Parse a compound Type string (e.g. "Type II + Learned-Dynamic + Type I")
    into a list of individual constraint component labels.
    Handles all compound patterns present in the updated Physical Constraints sheet.
    """
    if pd.isna(type_str) or str(type_str).strip().lower() in ("nan", "none", ""):
        return ["No Constraint"]
    s = str(type_str).strip()
    comps = []
    for part in s.split("+"):
        p = part.strip()
        pl = p.lower()
        if "type iii" in pl:
            comps.append("Physics Model as Input Feature")
        elif "type ii process" in pl:
            comps.append("Governing Equation (architecture)")
        elif "type ii" in pl:
            comps.append("Physical Topology (structural prior)")
        elif "type i soft" in pl or re.fullmatch(r"type i", pl):
            comps.append("Physics Loss Constraint")
        elif "physics-emulator" in pl:
            comps.append("Physics-Emulator (training data)")
        elif "physics-driven dynamic" in pl or "physics-driven" in pl:
            comps.append("Real-Time Physical Observation")
        elif "data-driven dynamic" in pl:
            comps.append("Statistical Correlation (dynamic)")
        elif "data-driven static" in pl:
            comps.append("Statistical Correlation (static)")
        elif "knowledge-static" in pl:
            comps.append("Domain Knowledge Graph")
        elif "learned-dynamic" in pl or "learned dynamic" in pl:
            comps.append("Data-Learned (dynamic)")
        elif ("learned-static" in pl or "learned static" in pl
              or "learned graph" in pl or "learned graph structure" in pl):
            comps.append("Data-Learned (static)")
    return comps if comps else ["No Constraint"]


def fig5_physics(df_pc: pd.DataFrame, df_general: pd.DataFrame,
                 outdir: Path, formats: list):
    """
    Three-panel figure reflecting the updated two-level physics constraint taxonomy:

      (a) Component-level frequency bar chart — how many studies contain each
          constraint component (a study with a compound type contributes to
          multiple bars)
      (b) Study-level pattern chart — how many studies fall into each
          primary pattern category (purely Type II, hybrid with learned,
          with dynamic, with Type I loss constraint, fully learned)
      (c) Temporal trend — grouped bar showing study counts per broad
          category over publication years
    """
    print("  Generating Figure 5: Physical constraints (updated taxonomy) …")

    df = df_pc.copy()
    df = df.merge(df_general[["Reference", "Year"]], on="Reference", how="left")
    df["Type"] = df["Type"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    print(f'df updated: {list(df.columns.values)}')
    # ── Extract components per study ───────────────────────────────────────
    df["components"] = df["Type"].apply(extract_components)

    # ── Component frequency (panel a) ─────────────────────────────────────
    # Display order: most common → least, specific → general
    COMPONENT_ORDER = [
        "Physical Topology (structural prior)",
        "Data-Learned (dynamic)",
        "Data-Learned (static)",
        "Statistical Correlation (dynamic)",
        "Physics Loss Constraint",
        "Statistical Correlation (static)",
        "Physics-Emulator (training data)",
        "Real-Time Physical Observation",
        "Domain Knowledge Graph",
        "Governing Equation (architecture)",
        "Physics Model as Input Feature",
        "No Constraint",
    ]
    COMPONENT_COLORS = {
        "Physical Topology (structural prior)":  BLUE,
        "Data-Learned (dynamic)":            ORANGE,
        "Data-Learned (static)":             PURPLE,
        "Statistical Correlation (dynamic)":        TEAL,
        "Physics Loss Constraint":     GREEN,
        "Statistical Correlation (static)":         AMBER,
        "Physics-Emulator (training data)":           GREY,
        "Real-Time Physical Observation":     "#795548",
        "Domain Knowledge Graph":           PINK,
        "Governing Equation (architecture)":    "#0d47a1",
        "Physics Model as Input Feature":                   RED,
        "No Constraint":              "#bdbdbd",
    }

    from collections import Counter
    comp_counts = Counter()
    for comps in df["components"]:
        for c in comps:
            comp_counts[c] += 1

    # Keep only components present in data, in display order
    present_order = [c for c in COMPONENT_ORDER if comp_counts.get(c, 0) > 0]
    comp_vals = [comp_counts[c] for c in present_order]

    # ── Study-level pattern categories (panel b) ───────────────────────────
    def classify_pattern(comps):
        cs = set(comps)
        has_t1   = "Physics Loss Constraint" in cs
        has_t2   = "Physical Topology (structural prior)" in cs or "Governing Equation (architecture)" in cs
        has_ld   = "Data-Learned (dynamic)" in cs
        has_ls   = "Data-Learned (static)" in cs
        has_dd   = "Statistical Correlation (dynamic)" in cs or "Statistical Correlation (static)" in cs
        has_pd   = "Real-Time Physical Observation" in cs
        has_em   = "Physics-Emulator (training data)" in cs
        has_t3   = "Physics Model as Input Feature" in cs
        purely_learned = not has_t2 and (has_ld or has_ls) and not has_t1

        if purely_learned:
            return "Fully\nData-Learned"
        if has_t2 and has_t1:
            return "Topology\n+ Loss Penalty"
        if has_t2 and (has_ld or has_ls) and (has_dd or has_pd):
            return "Topology\n+ Learned\n+ Dynamic"
        if has_t2 and (has_ld or has_ls):
            return "Topology\n+ Learned"
        if has_t2 and (has_dd or has_pd or has_em or has_t3):
            return "Topology\n+ Dynamic"
        if has_t2:
            return "Topology\nOnly"
        return "Other"

    df["pattern"] = df["components"].apply(classify_pattern)

    PATTERN_ORDER = [
        "Topology\nOnly",
        "Topology\n+ Learned",
        "Topology\n+ Learned\n+ Dynamic",
        "Topology\n+ Dynamic",
        "Topology\n+ Loss Penalty",
        "Fully\nData-Learned",
        "Other",
    ]
    PATTERN_COLORS = {
        "Topology\nOnly":                        BLUE,
        "Topology\n+ Learned":         ORANGE,
        "Topology\n+ Learned\n+ Dynamic": TEAL,
        "Topology\n+ Dynamic":   AMBER,
        "Topology\n+ Loss Penalty":          GREEN,
        "Fully\nData-Learned":                        PURPLE,
        "Other":                               GREY,
    }

    pat_counts_raw = df["pattern"].value_counts()
    present_pats = [p for p in PATTERN_ORDER if pat_counts_raw.get(p, 0) > 0]
    pat_vals = [pat_counts_raw.get(p, 0) for p in present_pats]

    # ── Temporal trend — broad categories (panel c) ────────────────────────
    TREND_CATS = {
        "Topology\nOnly":         lambda cs: set(cs) == {"Physical Topology (structural prior)"},
        "Topology + Learned adjacency": lambda cs: (
            ("Physical Topology (structural prior)" in cs or "Governing Equation (architecture)" in cs)
            and any(c in cs for c in ["Data-Learned (dynamic)", "Data-Learned (static)",
                                       "Statistical Correlation (dynamic)", "Statistical Correlation (static)",
                                       "Real-Time Physical Observation"])
            and "Physics Loss Constraint" not in cs
        ),
        "Includes physics loss constraint": lambda cs: "Physics Loss Constraint" in cs,
        "Fully\nData-Learned":          lambda cs: (
            "Physical Topology (structural prior)" not in cs
            and "Governing Equation (architecture)" not in cs
            and ("Data-Learned (dynamic)" in cs or "Data-Learned (static)" in cs)
        ),
    }
    TREND_COLORS = {
        "Topology\nOnly":            BLUE,
        "Topology + Learned adjacency":   ORANGE,
        "Includes physics loss constraint":  GREEN,
        "Fully\nData-Learned":           PURPLE,
    }

    years_sorted = sorted(df["Year"].dropna().unique().astype(int))
    df["Year_int"] = df["Year"].astype(int)

    trend_data = {}
    for cat, fn in TREND_CATS.items():
        mask = df["components"].apply(lambda cs: fn(set(cs)))
        yearly = df[mask].groupby("Year_int").size().reindex(years_sorted, fill_value=0)
        trend_data[cat] = yearly.values

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    # gs = fig.add_gridspec(1, 3, wspace=0.38)
    gs = fig.add_gridspec(1, 3)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # Panel (a) — component frequency (horizontal bar)
    y_a = range(len(present_order))
    colors_a = [COMPONENT_COLORS.get(c, GREY) for c in present_order]
    bars_a = ax_a.barh(y_a, comp_vals, color=colors_a,
                       edgecolor="white", linewidth=0.7)
    for i, (bar, cnt) in enumerate(zip(bars_a, comp_vals)):
        ax_a.text(bar.get_width() + 0.1, i, str(cnt),
                  va="center", fontsize=8, fontweight="bold")
    ax_a.set_yticks(list(y_a))
    ax_a.set_yticklabels(present_order, fontsize=8)
    ax_a.set_xlabel("Number of Studies")
    # ax_a.set_title(
    #     "(a)  Constraint Component\nFrequency (multi-count)",
    #     fontweight="bold", fontsize=10,
    # )
    ax_a.set_title("")
    ax_a.text(
        0.5, -0.12,
        "(a) Constraint Component\nFrequency (multi-count)",
        transform=ax_a.transAxes,
        ha="center", va="top",
        fontweight="bold", fontsize=10
    )

    ax_a.set_xlim(0, max(comp_vals) + 5)
    ax_a.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # note_a = (
    #     "Note: compound types contribute\n"
    #     "to multiple component bars\n"
    #     f"(38 studies, {sum(comp_vals)} total components)"
    # )
    # ax_a.text(
    #     0.97, 0.03, note_a,
    #     transform=ax_a.transAxes, ha="right", va="bottom",
    #     fontsize=7, color="grey", style="italic",
    # )

    # Panel (b) — study pattern categories (vertical bar)
    x_b = range(len(present_pats))
    colors_b = [PATTERN_COLORS.get(p, GREY) for p in present_pats]
    bars_b = ax_b.bar(x_b, pat_vals, color=colors_b,
                      edgecolor="white", linewidth=0.7, width=0.6)
    for bar, cnt in zip(bars_b, pat_vals):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(int(cnt)),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax_b.set_xticks(list(x_b))
    ax_b.set_xticklabels(
        [p.replace("\n", "\n") for p in present_pats],
        rotation=25, ha="right", fontsize=8,
    )
    ax_b.set_ylabel("Number of Studies")
    # ax_b.set_title(
    #     f"(b)  Study-Level Pattern\n(n = {len(df)})",
    #     fontweight="bold", fontsize=10,
    # )
    ax_b.set_title("")
    ax_b.text(
        0.5, -0.12,
        f"(b) Study-Level Pattern (n = {len(df)})",
        transform=ax_b.transAxes,
        ha="center", va="top",
        fontweight="bold", fontsize=10
    )

    ax_b.set_ylim(0, max(pat_vals) + 3)
    ax_b.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Panel (c) — temporal trend (grouped bar)
    n_cats = len(TREND_CATS)
    total_w = 0.72
    bar_w = total_w / n_cats
    offsets = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_cats)
    x_c = np.arange(len(years_sorted))

    for i, (cat, offset) in enumerate(zip(TREND_CATS.keys(), offsets)):
        ax_c.bar(
            x_c + offset, trend_data[cat],
            width=bar_w,
            color=TREND_COLORS.get(cat, GREY),
            edgecolor="white", linewidth=0.5,
            label=cat,
        )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(years_sorted)
    ax_c.set_ylabel("Number of Studies")
    # ax_c.set_title(
    #     "(c)  Temporal Trend in Constraint\nPattern (2020–2026)",
    #     fontweight="bold", fontsize=10,
    # )

    ax_c.set_title("")
    ax_c.text(
        0.5, -0.12,
        "(c) Temporal Trend in Constraint\nPattern (2020–2026)",
        transform=ax_c.transAxes,
        ha="center", va="top",
        fontweight="bold", fontsize=10
    )

    ax_c.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_c.legend(
        title="Constraint pattern",
        frameon=True, framealpha=0.9, edgecolor="lightgrey",
        fontsize=7, title_fontsize=8,
        loc="upper left",
    )

    # fig.set_constrained_layout_pads(h_pad=0.05, w_pad=0.05, hspace=0.05)
    fig.subplots_adjust(bottom=0.2)

    _save(fig, outdir, "fig5_physics", formats)
    plt.close(fig)



# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figure 6 – Feature inventory (static + dynamic)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_features(df_static: pd.DataFrame, df_dynamic: pd.DataFrame,
                  outdir: Path, formats: list):
    """
    Two-panel vertical figure:
      (a) Static features (frequency > 1), sorted descending
      (b) Dynamic features (all), sorted descending, with vision-derived
          zero-count rows highlighted in red
    """
    print("  Generating Figure 6: Feature inventory …")

    # ── Static features ───────────────────────────────────────────────────
    sf = (
        df_static[["Feature", "Studies"]]
        .dropna(subset=["Feature"])
        .copy()
    )
    sf["Feature"] = sf["Feature"].astype(str).str.strip()
    sf["Studies"] = pd.to_numeric(sf["Studies"], errors="coerce").fillna(0).astype(int)
    sf = sf[sf["Studies"] > 1].sort_values("Studies", ascending=True)

    # ── Dynamic features ──────────────────────────────────────────────────
    df = (
        df_dynamic[["Feature", "Studies"]]
        .dropna(subset=["Feature"])
        .copy()
    )
    df["Feature"] = df["Feature"].astype(str).str.strip()
    df["Studies"] = pd.to_numeric(df["Studies"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("Studies", ascending=True)

    # Append explicit zero-count vision rows
    vision_rows = [
        "UAV-derived channel width",
        "Debris density / blockage index",
        "Infrastructure blockage status",
        "Satellite-derived flood extent",
        "Surface velocity (optical flow)",
    ]
    vision_df = pd.DataFrame({
        "Feature": vision_rows,
        "Studies": [0] * len(vision_rows),
    })
    df = pd.concat([vision_df, df], ignore_index=True)
    # Re-sort so vision rows appear at bottom (count = 0 < all others)
    df = df.sort_values("Studies", ascending=True).reset_index(drop=True)

    # Colours: red for vision, blue otherwise
    def dyn_color(feature, count):
        if feature in vision_rows:
            return RED
        return BLUE

    fig, axes = plt.subplots(2, 1, figsize=(12, 11))
    fig.subplots_adjust(hspace=0.35)

    # ── Panel (a) ─────────────────────────────────────────────────────────
    bar_colors_a = [BLUE] * len(sf)
    axes[0].barh(
        sf["Feature"], sf["Studies"],
        color=bar_colors_a, edgecolor="white", linewidth=0.6,
    )
    for feat, cnt in zip(sf["Feature"], sf["Studies"]):
        axes[0].text(
            cnt + 0.15, sf.index[sf["Feature"] == feat][0]
            if feat in sf["Feature"].values else 0,
            str(int(cnt)),
            va="center", fontsize=8, fontweight="bold",
        )
    # Re-draw with proper indexing
    axes[0].cla()
    y_pos = range(len(sf))
    axes[0].barh(y_pos, sf["Studies"].values,
                 color=BLUE, edgecolor="white", linewidth=0.6)
    for i, (cnt, feat) in enumerate(zip(sf["Studies"].values, sf["Feature"].values)):
        axes[0].text(cnt + 0.15, i, str(int(cnt)),
                     va="center", fontsize=8, fontweight="bold")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(sf["Feature"].values, fontsize=8)
    axes[0].set_xlabel("Number of Studies")
    axes[0].set_title(
        f"(a)  Static Input Features  (shown: frequency > 1,  n total = {len(df_static)})",
        fontweight="bold",
    )
    axes[0].set_xlim(0, sf["Studies"].max() + 3)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel (b) ─────────────────────────────────────────────────────────
    bar_colors_b = [dyn_color(f, c) for f, c in zip(df["Feature"], df["Studies"])]
    y_pos_b = range(len(df))
    axes[1].barh(y_pos_b, df["Studies"].values,
                 color=bar_colors_b, edgecolor="white", linewidth=0.6)
    for i, (cnt, feat) in enumerate(zip(df["Studies"].values, df["Feature"].values)):
        lbl  = str(int(cnt)) if cnt > 0 else "0  (none identified)"
        clr  = RED if feat in vision_rows else "black"
        x_pos = max(cnt, 0) + 0.15
        axes[1].text(x_pos, i, lbl, va="center", fontsize=8,
                     fontweight="bold", color=clr)
    axes[1].set_yticks(y_pos_b)
    axes[1].set_yticklabels(df["Feature"].values, fontsize=8)
    axes[1].set_xlabel("Number of Studies")
    axes[1].set_title(
        "(b)  Dynamic Input Features  "
        "(vision-derived rows shown in red at count = 0)",
        fontweight="bold",
    )
    axes[1].set_xlim(0, df["Studies"].max() + 5)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Vision legend patch
    p_blue = mpatches.Patch(color=BLUE, label="Sensor / reanalysis features")
    p_red  = mpatches.Patch(color=RED,  label="Vision-derived features (not used in any study)")
    axes[1].legend(handles=[p_blue, p_red], frameon=True,
                   framealpha=0.9, edgecolor="lightgrey", loc="lower right")


    _save(fig, outdir, "fig6_features", formats)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Utility – save helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, outdir: Path, stem: str, formats: list):
    for fmt in formats:
        fpath = outdir / f"{stem}.{fmt}"
        fig.savefig(fpath, format=fmt)
        print(f"    Saved → {fpath.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SLR figures from the literature Excel workbook."
    )
    parser.add_argument(
        "--excel",
        default="literature.xlsx",
        help="Path to the Excel workbook (default: literature.xlsx)",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for figures (default: same directory as script)",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg", "eps"],
        help="Output format(s) — default: pdf png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1000,
        help="DPI for raster formats (default: 300)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply DPI globally
    plt.rcParams["savefig.dpi"] = args.dpi

    excel_path = Path(args.excel)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = args.format

    print(f"\n{'='*60}")
    print("  SLR Figure Generator")
    print(f"  Excel : {excel_path}")
    print(f"  Output: {outdir}")
    print(f"  Formats: {formats}")
    print(f"{'='*60}\n")

    # ── Load all sheets ───────────────────────────────────────────────────
    sheets = load_excel(str(excel_path))

    required = {
        "General":             "General",
        "Architecture":        "Architecture",
        "Graph Construction":  "Graph Construction",
        "Physical Constraints":"Physical Constraints",
        "Static Features":     "Static Features",
        "Dynamic Features":    "Dynamic Features",
    }
    for sheet_name in required:
        if sheet_name not in sheets:
            sys.exit(
                f"[ERROR] Expected sheet '{sheet_name}' not found.\n"
                f"  Available sheets: {list(sheets.keys())}"
            )

    df_general = sheets["General"]
    df_arch    = sheets["Architecture"]
    df_gc      = sheets["Graph Construction"]
    # df_pc      = sheets["Physical Constraints"]
    df_static  = sheets["Static Features"]
    df_dynamic = sheets["Dynamic Features"]

    # ── Generate figures ──────────────────────────────────────────────────
    print("\nGenerating figures …\n")
    print(f'df_gc original: {list(df_gc.columns.values)}')
    # fig2_overview(df_general, outdir, formats)
    # fig3_arch_matrix(df_arch, outdir, formats)
    fig4_graph_construction(df_gc, outdir, formats)
    # fig5_physics(df_pc, df_general, outdir, formats)
    # fig6_features(df_static, df_dynamic, outdir, formats)

    print(f"\n{'='*60}")
    print(f"  All figures written to: {outdir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
