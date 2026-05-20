from __future__ import annotations

import torch
from torch import nn

from stock_lstm.config import ModelConfig


class StockLSTM(nn.Module):
    def __init__(self, input_size: int, config: ModelConfig) -> None:
        super().__init__()
        dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


def build_model(input_size: int, config: ModelConfig) -> StockLSTM:
    config.validate()
    return StockLSTM(input_size=input_size, config=config)

