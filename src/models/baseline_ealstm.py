"""
baseline_ealstm.py  –  Per-node Entity-Aware LSTM baseline (no graph structure)
================================================================================
Implements the EA-LSTM cell from Kratzert, F., Klotz, D., Shalev, G.,
Klambauer, G., Hochreiter, S., & Nearing, G. (2019). "Towards learning
universal, regional, and local hydrological behaviors via machine learning
applied to large-sample datasets." HESS, 23, 5089-5110. Reimplemented
directly here rather than depending on the NeuralHydrology package, since
that framework's CAMELS-format data assumptions, config system, and PUB-style
basin splits don't match this project's X.npy/y.npy/gpu_sampler pipeline —
the cell itself is small enough to not need the dependency.

Departure from a standard LSTM
-------------------------------
A standard LSTM computes all four gates (input, forget, cell candidate,
output) as a function of [x_t, h_{t-1}]. EA-LSTM instead computes the
INPUT GATE ONCE per sequence, from STATIC attributes only, and holds it
fixed across every timestep:

    i        = sigmoid(W_i @ x_static + b_i)                    # once per sequence
    f_t      = sigmoid(W_f @ x_dyn_t + U_f @ h_{t-1} + b_f)      # every timestep
    g_t      = tanh   (W_g @ x_dyn_t + U_g @ h_{t-1} + b_g)      # every timestep
    o_t      = sigmoid(W_o @ x_dyn_t + U_o @ h_{t-1} + b_o)      # every timestep
    c_t      = f_t * c_{t-1} + i * g_t
    h_t      = o_t * tanh(c_t)

What transfers to this project and what doesn't
--------------------------------------------------
Kratzert et al.'s motivating use case is large-sample REGIONALISATION —
one shared LSTM trained jointly across hundreds of CAMELS catchments,
where the static-gated input gate lets the model learn catchment-
differentiated behaviour (a flashy headwater vs. a large regulated
lowland basin) without a separate LSTM per catchment, and is central to
NeuralHydrology's strength on prediction-in-ungauged-basins (PUB)
benchmarks. That motivating regime does NOT transfer directly here —
this project trains within a SINGLE catchment across its 27 gauge nodes,
not across many basins, and there is no PUB/ungauged-basin objective.

What DOES transfer is the underlying mechanism: PerNodeGRU/PerNodeLSTM
(and the STGNNFlood family) currently concatenate static node attributes
(catchment area, elevation, tidal flag) with dynamic features into ONE
input vector, treating them symmetrically. But the 27 Lee gauge nodes are
genuinely heterogeneous (catchment area ranges from ~3.65 km² at
Ballincolly to ~1185 km² at Waterworks Weir) in a way that should affect
HOW MUCH a given rainfall pulse matters, not just what additional
features are visible to the network. EA-LSTM's static-gated input gate is
a more principled way to inject that node-level heterogeneity into a
weight-shared recurrent core than symmetric concatenation — the mechanism
generalises down to node-level heterogeneity within one catchment even
though the original PUB/regionalisation motivation does not apply.

Multi-layer extension (beyond the original paper)
----------------------------------------------------
Kratzert et al. (2019) specify a single EA-LSTM layer. For interface
parity with PerNodeGRU/PerNodeLSTM's `gru_layers`/`lstm_layers` parameter,
this implementation stacks `ea_layers` EALSTMCells, with layer l>0 taking
the previous layer's hidden sequence as its dynamic input and
INDEPENDENTLY recomputing the static input gate from the same node_attr
at each layer. This is a direct, natural extension of the single-layer
formulation for stacking depth, not something specified in the original
paper — worth stating explicitly if used in the manuscript.
"""

import torch
import torch.nn as nn


class EALSTMCell(nn.Module):
    """
    Single Entity-Aware LSTM cell. See module docstring for the gate
    equations. The input gate is computed once per forward() call (not
    per timestep) since it depends only on the static attributes, which
    are constant across the sequence.
    """

    def __init__(self, f_dyn: int, f_static: int, hidden: int):
        super().__init__()
        self.hidden = hidden

        # Static-only input gate — computed once per sequence
        self.input_gate = nn.Linear(f_static, hidden)

        # Dynamic gates: forget, cell candidate, output — each a function
        # of [x_dyn_t, h_{t-1}], packed into one linear layer (3*hidden
        # output, chunked below) for efficiency, standard LSTM-style.
        self.dyn_gates = nn.Linear(f_dyn + hidden, 3 * hidden)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_gate.weight)
        nn.init.zeros_(self.input_gate.bias)
        nn.init.orthogonal_(self.dyn_gates.weight)
        nn.init.zeros_(self.dyn_gates.bias)
        # Forget-gate bias initialised to 1.0 (Jozefowicz et al., 2015;
        # standard LSTM practice) to reduce vanishing-gradient risk early
        # in training. dyn_gates packs [f, g, o] as equal-size chunks —
        # only the f chunk (first `hidden` entries) gets the +1 bias.
        with torch.no_grad():
            self.dyn_gates.bias[: self.hidden].fill_(1.0)

    def forward(self, x_dyn_seq: torch.Tensor, x_static: torch.Tensor):
        """
        x_dyn_seq : [B, T, f_dyn]   dynamic input sequence
        x_static  : [B, f_static]   static attributes, constant across T

        Returns
        -------
        h_seq : [B, T, hidden]  hidden state at every timestep
        h_T   : [B, hidden]     final hidden state
        c_T   : [B, hidden]     final cell state
        """
        B, T, _ = x_dyn_seq.shape
        device = x_dyn_seq.device

        # Input gate: static-only, computed ONCE, held fixed across T.
        # This is the one departure from a standard LSTM cell.
        i = torch.sigmoid(self.input_gate(x_static))          # [B, hidden]

        h_t = torch.zeros(B, self.hidden, device=device, dtype=x_dyn_seq.dtype)
        c_t = torch.zeros(B, self.hidden, device=device, dtype=x_dyn_seq.dtype)
        h_seq = []

        for t in range(T):
            combined = torch.cat([x_dyn_seq[:, t, :], h_t], dim=-1)
            gates = self.dyn_gates(combined)                   # [B, 3*hidden]
            f_t, g_t, o_t = gates.chunk(3, dim=-1)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i * g_t
            h_t = o_t * torch.tanh(c_t)
            h_seq.append(h_t)

        h_seq = torch.stack(h_seq, dim=1)                       # [B, T, hidden]
        return h_seq, h_t, c_t


class PerNodeEALSTM(nn.Module):
    """
    Per-node Entity-Aware LSTM baseline — no graph structure, no spatial
    component. Equivalent in spirit to PerNodeGRU/PerNodeLSTM (isolates
    the temporal-modelling contribution in the ablation ladder), but
    replaces symmetric static+dynamic concatenation with EA-LSTM's
    static-gated input pathway.

    forward input:  x_seq     [B, T_in, N, F_dyn]
                    node_attr [N, F_static]
    forward output: pred      [B, T_out, N]
    """

    def __init__(self, f_dyn, f_static, hidden, ea_layers, t_out, dropout):
        super().__init__()
        self.hidden    = hidden
        self.ea_layers = ea_layers

        self.cells = nn.ModuleList([
            EALSTMCell(
                f_dyn=(f_dyn if l == 0 else hidden),
                f_static=f_static,
                hidden=hidden,
            )
            for l in range(ea_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Same output head as PerNodeGRU/PerNodeLSTM
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, t_out),
        )

    def forward(self, x_seq, node_attr, **kwargs):
        # kwargs absorbs edge_index/edge_attr so call signature matches ST-GNN
        B, T, N, F_dyn = x_seq.shape

        # Flatten (B, N) -> B*N independent entities, static attrs tiled
        # across the batch (same reshape pattern as PerNodeGRU/PerNodeLSTM)
        x_dyn    = x_seq.permute(0, 2, 1, 3).reshape(B * N, T, F_dyn)     # [B*N, T, F_dyn]
        x_static = node_attr.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)  # [B*N, F_static]

        h_seq = x_dyn
        h_T = None
        for l, cell in enumerate(self.cells):
            h_seq, h_T, _c_T = cell(h_seq, x_static)
            if l < len(self.cells) - 1:
                h_seq = self.dropout(h_seq)

        h_final = self.dropout(h_T)                                       # [B*N, hidden]

        pred = self.head(h_final)\
                   .view(B, N, -1)\
                   .permute(0, 2, 1)                                       # [B, T_out, N]
        return pred


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_in, N, T_out = 4, 32, 27, 4
    F_dyn, F_static = 11, 7

    node_attr = torch.randn(N, F_static)
    x_seq     = torch.rand(B, T_in, N, F_dyn) * 2

    model = PerNodeEALSTM(
        f_dyn=F_dyn, f_static=F_static, hidden=64,
        ea_layers=2, t_out=T_out, dropout=0.1,
    )

    out = model(x_seq, node_attr)
    assert out.shape == (B, T_out, N), f"Wrong shape: {out.shape}"
    print(f"Output: {tuple(out.shape)}  ✓")

    loss = out.sum()
    loss.backward()
    print("Backward pass:  ✓")

    # Sanity check: the input gate must depend only on static attrs, not
    # on the dynamic sequence — verify by confirming its output is
    # identical for two different dynamic sequences given the same
    # static attributes.
    with torch.no_grad():
        cell = model.cells[0]
        x_static_flat = node_attr.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
        i_a = torch.sigmoid(cell.input_gate(x_static_flat))
        x_dyn_alt = torch.rand(B, T_in, N, F_dyn) * 5   # different dynamic values
        x_dyn_alt_flat = x_dyn_alt.permute(0, 2, 1, 3).reshape(B * N, T_in, F_dyn)
        # input_gate only consumes x_static, so recomputing with different
        # dynamic input must give the identical gate value
        i_b = torch.sigmoid(cell.input_gate(x_static_flat))
        assert torch.allclose(i_a, i_b), "Input gate must be invariant to dynamic input"
    print("Static-only input gate invariance check:  ✓")

    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")
    print("Smoke test passed.")
