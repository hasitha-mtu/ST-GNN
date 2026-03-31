import torch
import torch.nn as nn

class PerNodeLSTM(nn.Module):
    def __init__(self, f_dyn, f_static, hidden, lstm_layers, t_out, dropout):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(f_dyn + f_static, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, t_out),
        )

    def forward(self, x_seq, node_attr, **kwargs):
        B, T, N, _ = x_seq.shape
        static_exp = node_attr.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        combined   = torch.cat([x_seq, static_exp], dim=-1)
        h = self.node_embed(combined.reshape(B * T * N, -1))
        h = h.view(B, T, N, -1)
        gru_in    = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, (h_n, _) = self.lstm(gru_in)   # LSTM returns (output, (h_n, c_n))
        pred = self.head(h_n[-1])\
                   .view(B, N, -1)\
                   .permute(0, 2, 1)
        return pred