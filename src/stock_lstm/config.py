from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "AAPL"
    start_date: str = "2018-01-01"
    end_date: str | None = None
    target_column: str = "Close"
    lookback: int = 60
    horizon: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    artifacts_dir: Path = Path("artifacts")

    def validate(self) -> None:
        if self.lookback < 5:
            raise ValueError("lookback must be at least 5.")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1.")
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("split ratios must be positive.")


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20

    def validate(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 8
    seed: int = 42

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.patience < 1:
            raise ValueError("patience must be positive.")

