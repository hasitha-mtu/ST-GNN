"""
st_gnn_backwater_edge.py  –  ST-GNN with gated bridge/culvert backwater edges
================================================================================
Architecture
------------
Extends STGNNFloodModel's river-edge GATConv stack with a SECOND,
independently-gated edge set: the backwater edges from
precompute_backwater_edges.py (bridge/constriction node -> its immediate
upstream neighbour), representing the gradually-varied-flow backwater
mechanism (Chow, 1959) that the directed river topology cannot represent
on its own — see precompute_backwater_edges.py's module docstring and
scenario_generator.py's generate_s6_channel_blockage for the physical
motivation and the S6 diagnostic this model is meant to improve on.

Design follows the same "conditional, gated edge" pattern already
established by STGNNDynEdge (discharge-based conductance) and
STGNNHANDEdge/STGNNSoilGate (elevation/saturation-based activation):
the backwater pathway should stay closed under normal flow and open
only when the constriction node's own stage indicates its capacity is
being challenged. This keeps the strictly-correct directed river prior
intact everywhere else in the graph, rather than the blanket
bidirectional-everywhere alternative, which would inject an always-on,
physically incorrect prior at all 27 nodes for the sake of a mechanism
that's only real at 3 specific bridge locations in this catchment (see
precompute_backwater_edges.py's output for the Lee graph specifically).

Backwater activation gate
--------------------------
    H_bridge(t)    = gauge_datum[bridge] + normalised_stage[bridge](t) * stage_range[bridge]
    gate(edge, t)  = sigmoid( beta * (H_bridge(t) - gate_reference_m[edge]) )

where gate_reference_m is the bridge node's own p90_mAOD flood
threshold (precomputed, per edge) and beta is a learnable sharpness
parameter. This mirrors STGNNHANDEdge's stage-vs-saddle gate exactly,
except keyed on the bridge's OWN threshold (a real, node-specific
capacity indicator) rather than a cross-node saddle elevation — backwater
risk is fundamentally about whether the constriction itself is being
pushed toward capacity, not about two nodes' relative terrain.

Unlike DFC-GNN's hard elevation-differential gate (which SUPPRESSES
uphill propagation), this gate does not exist to block anything — its
entire purpose is to ALLOW propagation in a direction the plain
directed river graph structurally forbids, conditioned on it being
physically plausible. It should therefore be read as the opposite kind
of gate from DFC-GNN's: a permission gate, not a suppression gate.

Combined edge set (static structure, built once in __init__ from the
buffers passed in):
    n_river = river edges (from edges.csv)          — always-open, static
    n_bw    = backwater edges (from precompute_backwater_edges.py) — gated

Static edge attributes for GATConv's edge_dim: same 4-feature layout as
river/HAND edges (river_dist_km, area_ratio, elev_drop_m, same_tributary)
for the river edges, and a 5th "gate value" appended as the dynamic
5th feature — matching the f_edge=5 convention already used by
STGNNDynEdge for its conductance feature (see that file's
_dynamic_edge_attr for the equivalent pattern). River edges get
gate=1.0 always (unconditional); backwater edges get the sigmoid
activation above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class STGNNBackwaterEdge(nn.Module):
    """
    ST-GNN with a static directed river graph PLUS a gated, conditional
    backwater edge class at bridge/culvert constriction points.

    Parameters
    ----------
    f_dyn, f_static : int
        As in STGNNFloodModel.
    edge_index : LongTensor [2, E_river]
        River edges (from graph_builder.py / edges.csv).
    edge_attr : FloatTensor [E_river, 4]
        Static river edge attributes.
    bw_src, bw_dst : LongTensor [E_bw]
        Backwater edges (bridge node -> upstream neighbour), from
        precompute_backwater_edges.py.
    bw_edge_attr : FloatTensor [E_bw, 4]
        Static backwater edge attributes (already sign-flipped/inverted
        by precompute_backwater_edges.py — do NOT re-derive from the
        forward river edge here).
    bw_gate_reference : FloatTensor [E_bw]
        Per-edge gate_reference_m (bridge node's own p90_mAOD threshold).
    gauge_datum, stage_range : FloatTensor [N]
        Per-node absolute-elevation reconstruction terms, same
        convention as STGNNHANDEdge (H = datum + normalised_stage * range).
    stage_idx : int
        Column index of normalised_stage in the dynamic feature vector.
        Default 1 (matches build_dataset.py's [stage_anomaly, norm_stage,
        dh_dt, discharge, rainfall_mm, ...] ordering used elsewhere in
        this project — confirm against your actual feature ordering
        before training).
    hidden, gat_heads, gru_layers, t_out, dropout : as in STGNNFloodModel.
    """

    def __init__(
        self,
        f_dyn: int,
        f_static: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        bw_src: torch.Tensor,
        bw_dst: torch.Tensor,
        bw_edge_attr: torch.Tensor,
        bw_gate_reference: torch.Tensor,
        gauge_datum: torch.Tensor,
        stage_range: torch.Tensor,
        stage_idx: int = 1,
        hidden: int = 64,
        gat_heads: int = 2,
        gru_layers: int = 2,
        t_out: int = 4,
        dropout: float = 0.1,
        gate_sharpness: float = 5.0,
    ):
        super().__init__()
        self.hidden    = hidden
        self.t_out     = t_out
        self.stage_idx = stage_idx

        n_river = edge_index.shape[1]
        n_bw    = bw_src.shape[0]
        self.n_river, self.n_bw = n_river, n_bw

        # Combined static edge index — built once, matches the pattern
        # in dfc_gnn_unified.py for combining river+HAND edges.
        combined_src = torch.cat([edge_index[0], bw_src])
        combined_dst = torch.cat([edge_index[1], bw_dst])
        self.register_buffer("edge_index", torch.stack([combined_src, combined_dst]))

        # Static 4-feature attrs, concatenated (river first, then backwater)
        self.register_buffer("static_edge_attr", torch.cat([edge_attr, bw_edge_attr], dim=0))

        self.register_buffer("bw_gate_reference", bw_gate_reference.float())
        self.register_buffer("gauge_datum", gauge_datum.float())
        self.register_buffer("stage_range", stage_range.float())
        # Which node each backwater edge's gate is keyed on (the src —
        # the bridge/constriction node itself, not the upstream target).
        self.register_buffer("bw_gate_node", bw_src.long())

        self.gate_sharpness = nn.Parameter(torch.tensor(float(gate_sharpness)))
        self.last_gate_mean = torch.tensor(0.0)   # updated every forward() call

        # ── Input projection ───────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(f_dyn + f_static, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
        )

        # ── Temporal encoder ──────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=hidden, hidden_size=hidden, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_dropout = nn.Dropout(dropout)

        # ── GATConv layers — f_edge=5 (4 static + 1 gate value) ────────
        f_edge = 5
        self.gat1 = GATConv(hidden, hidden, heads=gat_heads, concat=True,
                            dropout=dropout, edge_dim=f_edge)
        self.norm1 = nn.LayerNorm(hidden * gat_heads)
        self.res1  = nn.Linear(hidden, hidden * gat_heads, bias=False)

        self.gat2 = GATConv(hidden * gat_heads, hidden, heads=gat_heads,
                            concat=False, dropout=dropout, edge_dim=f_edge)
        self.norm2 = nn.LayerNorm(hidden)
        self.res2  = nn.Linear(hidden * gat_heads, hidden, bias=False)

        self.gat3 = GATConv(hidden, hidden // 2, heads=2, concat=False,
                            dropout=dropout, edge_dim=f_edge)
        self.norm3 = nn.LayerNorm(hidden // 2)
        self.res3  = nn.Linear(hidden, hidden // 2, bias=False)

        self.head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, t_out),
        )

    # ──────────────────────────────────────────────────────────────────
    def _backwater_gate(self, x_last: torch.Tensor) -> torch.Tensor:
        """
        x_last : [B, N, F_dyn]  last observed input timestep.
        Returns gate : [B, n_bw]  sigmoid activation per backwater edge.

        Stores the batch-mean activation as self.last_gate_mean, a plain
        (non-buffer, non-parameter) tensor attribute — NOT a second
        return value. Every training script in this codebase (see
        train_st_gnn_flood_model.py, train_st_gnn_backwater_edge.py)
        calls model(...) expecting a single delta-prediction tensor back;
        changing forward()'s return signature would break that contract
        everywhere. Reading model.last_gate_mean (or
        model._orig_mod.last_gate_mean under torch.compile — see
        train_st_gnn_backwater_edge.py's gate_sharpness logging for the
        same pattern) after the forward() call is a strictly additive way
        for the training loop to add a sparsity penalty without touching
        any other call site.
        """
        B = x_last.shape[0]
        norm_stage = x_last[:, :, self.stage_idx]          # [B, N]

        H = (self.gauge_datum.unsqueeze(0)
             + norm_stage * self.stage_range.unsqueeze(0))  # [B, N]

        H_bridge = H[:, self.bw_gate_node]                  # [B, n_bw]
        gate = torch.sigmoid(
            self.gate_sharpness * (H_bridge - self.bw_gate_reference.unsqueeze(0))
        )                                                    # [B, n_bw]
        self.last_gate_mean = gate.mean()
        return gate

    def _combined_edge_attr(self, x_last: torch.Tensor) -> torch.Tensor:
        """Returns [B, n_river+n_bw, 5] — static attrs + per-edge gate value."""
        B = x_last.shape[0]
        static_tiled = self.static_edge_attr.unsqueeze(0).expand(B, -1, -1)  # [B, E, 4]

        river_gate = torch.ones(B, self.n_river, device=x_last.device)       # always open
        bw_gate    = self._backwater_gate(x_last)                            # [B, n_bw]
        gate = torch.cat([river_gate, bw_gate], dim=1).unsqueeze(-1)         # [B, E, 1]

        return torch.cat([static_tiled, gate], dim=-1)                       # [B, E, 5]

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_seq:      torch.Tensor,   # [B, T_in, N, F_dyn]
        node_attr:  torch.Tensor,   # [N, F_static]
        edge_index: torch.Tensor | None = None,  # unused — combined
        edge_attr:  torch.Tensor | None = None,  # topology is already
                                                    # registered as buffers.
                                                    # Accepted positionally
                                                    # so this matches every
                                                    # other graph model's
                                                    # call signature in
                                                    # this codebase (e.g.
                                                    # train_st_gnn_flood_model.py
                                                    # calls
                                                    # model(x_seq, node_attr,
                                                    #       edge_index, edge_attr)
                                                    # positionally — **kwargs
                                                    # alone cannot absorb
                                                    # positional arguments).
        **kwargs,
    ) -> torch.Tensor:
        B, T_in, N, _ = x_seq.shape

        static_exp = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T_in, -1, -1)
        x_combined = torch.cat([x_seq, static_exp], dim=-1)
        x_reshaped = x_combined.permute(0, 2, 1, 3).reshape(B * N, T_in, -1)
        x_proj     = self.input_proj(x_reshaped)

        gru_out, _ = self.gru(x_proj)
        h = self.gru_dropout(gru_out[:, -1, :]).reshape(B, N, self.hidden)

        x_last = x_seq[:, -1, :, :]                        # [B, N, F_dyn]
        combined_ea = self._combined_edge_attr(x_last)       # [B, E, 5]

        E = self.edge_index.shape[1]
        h_flat = h.reshape(B * N, self.hidden)
        offsets = torch.arange(B, device=x_seq.device) * N
        src_b = (self.edge_index[0].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_b = (self.edge_index[1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batched_ei = torch.stack([src_b, dst_b], dim=0)
        batched_ea = combined_ea.reshape(B * E, -1)

        h1 = F.elu(self.norm1(self.gat1(h_flat, batched_ei, batched_ea) + self.res1(h_flat)))
        h2 = F.elu(self.norm2(self.gat2(h1,     batched_ei, batched_ea) + self.res2(h1)))
        h3 = F.elu(self.norm3(self.gat3(h2,     batched_ei, batched_ea) + self.res3(h2)))

        h_graph = h3.reshape(B, N, self.hidden // 2)
        delta = self.head(h_graph)
        return delta.permute(0, 2, 1)                       # [B, T_out, N]


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_in, N, T_out = 4, 32, 27, 4
    F_dyn, F_static = 11, 7
    E_river, E_bw = 28, 3

    edge_index = torch.randint(0, N, (2, E_river))
    edge_attr  = torch.randn(E_river, 4)
    node_attr  = torch.randn(N, F_static)
    x_seq      = torch.rand(B, T_in, N, F_dyn)

    bw_src = torch.tensor([14, 15, 18])
    bw_dst = torch.tensor([4, 8, 11])
    bw_edge_attr = torch.randn(E_bw, 4)
    bw_gate_reference = torch.tensor([24.97, 21.63, 65.68])
    gauge_datum = torch.rand(N) * 50 + 10
    stage_range = torch.rand(N) * 2 + 0.5

    model = STGNNBackwaterEdge(
        f_dyn=F_dyn, f_static=F_static,
        edge_index=edge_index, edge_attr=edge_attr,
        bw_src=bw_src, bw_dst=bw_dst, bw_edge_attr=bw_edge_attr,
        bw_gate_reference=bw_gate_reference,
        gauge_datum=gauge_datum, stage_range=stage_range,
        stage_idx=1, hidden=64, gat_heads=2, gru_layers=2, t_out=T_out,
    )

    out = model(x_seq, node_attr)
    assert out.shape == (B, T_out, N), f"Wrong shape: {out.shape}"
    print(f"Output: {tuple(out.shape)}  \u2713")

    out.sum().backward()
    assert model.gate_sharpness.grad is not None, "gate_sharpness has no grad"
    print("Backward pass + gate_sharpness gradient flow:  \u2713")

    # Gate responsiveness check: bridge stage well above vs below its
    # own gate_reference should give very different gate values.
    with torch.no_grad():
        x_low = x_seq.clone()
        x_high = x_seq.clone()
        # push bridge nodes' normalised_stage low / high at the last step
        x_low[:, -1, bw_src, 1]  = -2.0
        x_high[:, -1, bw_src, 1] =  5.0
        gate_low  = model._backwater_gate(x_low[:, -1, :, :]).mean().item()
        gate_high = model._backwater_gate(x_high[:, -1, :, :]).mean().item()
        print(f"Gate at low bridge stage: {gate_low:.4f}   "
              f"high bridge stage: {gate_high:.4f}")
        assert gate_high > gate_low, "gate does not respond to bridge stage"
        print("Gate responds correctly to bridge stage magnitude:  \u2713")

    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")
    print("Smoke test passed.")
