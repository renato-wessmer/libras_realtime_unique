# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)    # (B, T, H)
        h_last = out[:, -1, :]   # (B, H)
        logits = self.fc(h_last) # (B, C)
        return logits
