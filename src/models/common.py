"""
common.py  -  building blocks shared by the ST-GNN family (revision audit M)
============================================================================

positive_param / softplus_inv
    Learnable gate sharpness parameters (HAND alpha, backwater beta,
    SoilGate gamma, conductance lambda) were unconstrained, so training
    could flip their sign and invert the physics of the gate (e.g. a HAND
    edge opening when the water surface is BELOW the saddle, or conductance
    falling as discharge rises). They are now stored as a raw parameter and
    used through softplus, which keeps them strictly positive. The raw
    value is initialised so that softplus(raw) equals the documented
    initial value (alpha = 5.0, beta = 5.0, gamma = 10.0, lambda = 3.0).

GatedGATConv
    torch_geometric GATConv with an explicit per-edge gate g_e in [0, 1]
    added to the attention logit as log(g_e + 1e-6) before the softmax,
    the same construction as DFC-GNN / PC-DFC-GNN (Eqs. 7-8). This is the
    "independently gated edge class" of manuscript Section 3.5.5: a gate of
    0 removes the edge, regardless of what the learned attention would have
    assigned to it. Self-loops added internally by GATConv are always open.

conductance
    Discharge-conditioned conductance of manuscript Eq. 1, shared by
    DynEdge, HANDEdge, SoilGate and PC-DFC-GNN:
        c_ij(t) = sigmoid(lambda * (Q_i(t) / Q_ref,i - 1))
    with Q = sign(Q) log(1 + |Q|) (dynamic feature 3) and a PER-NODE
    reference Q_ref,i (training-period mean at node i;
    utils.train_utils.discharge_reference). The edge attribute is
    log(c_ij + 1e-6).
"""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax

GATE_EPS = 1e-6


def softplus_inv(y: float) -> float:
    """x such that softplus(x) = y (y > 0)."""
    y = float(y)
    if y <= 0:
        raise ValueError("softplus_inv needs y > 0")
    return y + math.log(-math.expm1(-y))


def positive_param(init: float) -> nn.Parameter:
    return nn.Parameter(torch.tensor(softplus_inv(init), dtype=torch.float32))


def log_conductance(Q_src: torch.Tensor, Q_ref_src: torch.Tensor,
                    scale: torch.Tensor) -> torch.Tensor:
    c = torch.sigmoid(scale * (Q_src / (Q_ref_src + 1e-8) - 1.0))
    return torch.log(c + 1e-6)


def as_node_ref(discharge_ref, n_nodes: int | None = None) -> torch.Tensor:
    """Per-node discharge reference as a float tensor [N] (or [1] scalar)."""
    if isinstance(discharge_ref, torch.Tensor):
        t = discharge_ref.detach().float().reshape(-1)
    elif isinstance(discharge_ref, (list, tuple)):
        t = torch.tensor([float(v) for v in discharge_ref], dtype=torch.float32)
    else:
        t = torch.tensor([float(discharge_ref)], dtype=torch.float32)
    if n_nodes is not None and t.numel() == 1:
        t = t.expand(n_nodes).clone()
    return t


_GAT_EDGE_UPDATE = GATConv.edge_update
_GAT_EDGE_UPDATE_SIG = inspect.signature(_GAT_EDGE_UPDATE)


def _gated_edge_update(self, *args, **kwargs):
    """GATConv.edge_update with log(gate + eps) added to the attention logit
    BEFORE the softmax over each node's incoming edges (as in DFC-GNN /
    PC-DFC-GNN, Eqs. 7-8). A multiplicative gate applied AFTER the softmax
    is not enough: the gated edge still takes part in the normalisation, so
    a closed edge's source still changes the destination's other weights.
    Works with both PyG edge_update signatures (size_i / dim_size)."""
    a = _GAT_EDGE_UPDATE_SIG.bind(self, *args, **kwargs).arguments
    alpha_j, alpha_i = a["alpha_j"], a.get("alpha_i")
    edge_attr, index, ptr = a.get("edge_attr"), a["index"], a.get("ptr")
    size = a.get("dim_size", a.get("size_i"))
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    if index.numel() == 0:
        return alpha
    if edge_attr is not None and self.lin_edge is not None:
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        alpha = alpha + (edge_attr * self.att_edge).sum(dim=-1)
    alpha = F.leaky_relu(alpha, self.negative_slope)
    g = getattr(self, "_edge_gate", None)
    if g is not None:
        g = g.reshape(-1).to(alpha.dtype)
        extra = alpha.shape[0] - g.shape[0]                 # self-loops appended
        if extra < 0:
            raise RuntimeError("edge_gate longer than the edge list")
        if extra:
            g = torch.cat([g, g.new_ones(extra)])
        alpha = alpha + torch.log(g + GATE_EPS).view(-1, 1)
    alpha = pyg_softmax(alpha, index, ptr, size)
    return F.dropout(alpha, p=self.dropout, training=self.training)


_gated_edge_update.__signature__ = _GAT_EDGE_UPDATE_SIG
_gated_edge_update.__name__ = "edge_update"      # PyG Inspector keys on __name__


class GatedGATConv(GATConv):
    """GATConv with an optional per-edge gate g_e in [0, 1] entering the
    attention logit as log(g_e + 1e-6) before the softmax. g = 0 removes the
    edge (its source no longer influences the destination at all); g = 1
    leaves GATConv unchanged. Self-loops added by GATConv are always open."""

    edge_update = _gated_edge_update

    def forward(self, x, edge_index, edge_attr=None, edge_gate=None, **kw):
        self._edge_gate = edge_gate
        try:
            return super().forward(x, edge_index, edge_attr, **kw)
        finally:
            self._edge_gate = None


def batched_graph(edge_index: torch.Tensor, B: int, N: int) -> torch.Tensor:
    """[2, E] -> [2, B*E] with per-sample node offsets (sample-major)."""
    off = torch.arange(B, device=edge_index.device) * N
    src = (edge_index[0].unsqueeze(0) + off.unsqueeze(1)).reshape(-1)
    dst = (edge_index[1].unsqueeze(0) + off.unsqueeze(1)).reshape(-1)
    return torch.stack([src, dst], dim=0)


class GATStack2(nn.Module):
    """Two-layer GAT stack with residual connections, LayerNorm and ELU
    (manuscript Section 3.5): hidden -> hidden*heads -> hidden//2."""

    def __init__(self, hidden: int, heads: int, f_edge: int, dropout: float):
        super().__init__()
        self.gat1 = GatedGATConv(hidden, hidden, heads=heads, concat=True,
                                 dropout=dropout, edge_dim=f_edge)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.res1 = nn.Linear(hidden, hidden * heads, bias=False)
        self.gat2 = GatedGATConv(hidden * heads, hidden // 2, heads=heads,
                                 concat=False, dropout=dropout, edge_dim=f_edge)
        self.norm2 = nn.LayerNorm(hidden // 2)
        self.res2 = nn.Linear(hidden * heads, hidden // 2, bias=False)

    def layer1(self, h, ei, ea, gate=None):
        return self.gat1(h, ei, ea, edge_gate=gate)

    def finish1(self, msg, h):
        return F.elu(self.norm1(msg + self.res1(h)))

    def layer2(self, h1, ei, ea, gate=None):
        return F.elu(self.norm2(self.gat2(h1, ei, ea, edge_gate=gate) + self.res2(h1)))

    def forward(self, h, ei, ea, gate=None):
        return self.layer2(self.finish1(self.layer1(h, ei, ea, gate), h), ei, ea, gate)
