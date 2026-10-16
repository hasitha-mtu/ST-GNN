"""
baseline_lstm.py  –  Per-node LSTM baseline (no graph structure)
================================================================
Identical to PerNodeGRU (the ST-GNN backbone with the graph layers removed,
manuscript Section 3.4) with the GRU replaced by an LSTM.
"""
import torch.nn as nn

try:
    from .baseline_gru import PerNodeGRU
except ImportError:          # run as a script
    from models.baseline_gru import PerNodeGRU


class PerNodeLSTM(PerNodeGRU):
    rnn_cls = nn.LSTM

    def __init__(self, f_dyn, f_static, hidden, lstm_layers, t_out, dropout):
        super().__init__(f_dyn, f_static, hidden, lstm_layers, t_out, dropout)
