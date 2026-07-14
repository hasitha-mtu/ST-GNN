"""
compile_utils.py  —  torch.compile wrapper for ST-GNN training
══════════════════════════════════════════════════════════════════
Why NOT to compile the whole graph model
─────────────────────────────────────────
All ST-GNN variants share PyG's GATConv (gat_conv.py:254) across
three layers with DIFFERENT input dimensions:

    gat1:  in=hidden(64)          → out=hidden×heads(128)  concat=True
    gat2:  in=hidden×heads(128)   → out=hidden(64)          concat=False
    gat3:  in=hidden(64)          → out=hidden//2(32)       concat=False

TorchDynamo caches compiled kernels per function signature.  Because
all three layers call the SAME GATConv forward() with different input
sizes, TorchDynamo triggers a recompile each time the size changes.
After 8 recompiles, it hits config.recompile_limit and falls back to
eager execution for ALL subsequent GATConv calls — paying compilation
overhead while getting zero benefit.

Solution: compile ONLY the GRU temporal encoder submodule (model.gru)
for graph-based models.  The GRU processes T_in × N nodes every batch
and is the dominant compute cost (~60-70% of total time); it has
perfectly static feature dimensions and benefits cleanly from compile.

mode= parameter
─────────────────
'mode' is an inductor-only parameter.  Passing it to 'aot_eager' on
Windows produces a harmless but noisy warning.  This file never passes
mode= to aot_eager; for inductor (Linux/macOS) it passes 'default'
(balanced, safe) rather than 'reduce-overhead' (which can occasionally
cause numerical issues with in-place ops like scatter_add_).
"""

from __future__ import annotations

import logging
import platform
import torch
import torch.nn as nn


def _backend() -> str:
    """aot_eager on Windows (no Triton); inductor on Linux/macOS."""
    return "aot_eager" if platform.system() == "Windows" else "inductor"


def _log(logger: logging.Logger | None, msg: str) -> None:
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ── Public API ────────────────────────────────────────────────────────────────

def compile_model(
    model:  nn.Module,
    tag:    str | None = None,
    logger: logging.Logger | None = None,
) -> nn.Module:
    """
    Apply torch.compile appropriately for the model architecture.

    No-graph models (GRU, LSTM)
        Compile the entire model.  Pure PyTorch ops, static feature
        dimensions → clean compile with no recompilation triggers.
        Expected speedup: 20-30% on GPU.

    Graph models (all ST-GNN variants, DFC-GNN)
        Compile model.gru only — the temporal encoder submodule.
        GATConv layers are left uncompiled to avoid the recompile-limit
        issue caused by different input dims across gat1/gat2/gat3.
        The GRU is the dominant compute (~60-70% of time) so this
        still gives meaningful end-to-end speedup (12-20% overall).

    Falls back silently to uncompiled if torch.compile is unavailable
    or raises (e.g. unsupported op in a custom layer).

    Parameters
    ----------
    model  : instantiated model already on the target device
    tag    : model identifier; controls compile scope
    logger : training logger; falls back to print() if None
    """
    if not hasattr(torch, "compile"):
        _log(logger, "  [compile] torch ≥ 2.0 required — skipping")
        return model

    backend = _backend()

    # No-graph baselines: compile the whole model
    if tag in ("gru", "lstm"):
        try:
            compiled = torch.compile(model, backend=backend, dynamic=False,
                                     fullgraph=False)
            _log(logger, f"  [compile] {tag}: full model compiled "
                         f"(backend={backend}, dynamic=False) — "
                         f"JIT triggers on first batch")
            return compiled
        except Exception as e:
            _log(logger, f"  [compile] WARNING: full-model compile failed "
                         f"({type(e).__name__}: {e}) — running uncompiled")
            return model

    # Graph models: compile only the GRU temporal encoder
    # GATConv layers stay uncompiled — avoids the recompile_limit(8) issue
    # where three GATConv calls with dims 64/128/64 exhaust TorchDynamo's
    # cache and fall back to eager anyway (all overhead, zero benefit).
    if not hasattr(model, "gru"):
        _log(logger, f"  [compile] {tag}: no 'gru' submodule found — skipping")
        return model

    try:
        model.gru = torch.compile(model.gru, backend=backend, dynamic=False,
                                  fullgraph=False)
        _log(logger, f"  [compile] {tag}: model.gru compiled "
                     f"(backend={backend}) — "
                     f"GATConv/custom layers left uncompiled to avoid "
                     f"recompile_limit on dim mismatch (64/128/64)")
        return model
    except Exception as e:
        _log(logger, f"  [compile] WARNING: gru submodule compile failed "
                     f"({type(e).__name__}: {e}) — running fully uncompiled")
        return model


def is_compiled(model: nn.Module) -> bool:
    """True if the whole model has been wrapped by torch.compile."""
    return hasattr(model, "_orig_mod")
