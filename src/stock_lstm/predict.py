from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load

from stock_lstm.config import ModelConfig
from stock_lstm.data import normalize_price_frame
from stock_lstm.model import build_model
from stock_lstm.utils import read_json


@dataclass(frozen=True)
class Prediction:
    date: str
    predicted_close: float


class StockPricePredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        scaler: object,
        metadata: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
        self.device = device
        self.target_column = metadata.get("target_column", "Close")
        self.lookback = int(metadata["lookback"])
        self.training_horizon = int(metadata.get("horizon", 1))

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path) -> "StockPricePredictor":
        model_path = artifacts_dir / "model.pt"
        scaler_path = artifacts_dir / "target_scaler.joblib"
        metadata_path = artifacts_dir / "metadata.json"

        missing = [str(path) for path in [model_path, scaler_path, metadata_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metadata = read_json(metadata_path)
        checkpoint = torch.load(model_path, map_location=device)
        model_config = ModelConfig(**checkpoint["model_config"])
        model = build_model(input_size=int(checkpoint["input_size"]), config=model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        scaler = load(scaler_path)

        return cls(model=model, scaler=scaler, metadata=metadata, device=device)

    def predict_future(self, prices: pd.DataFrame, horizon: int) -> list[Prediction]:
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        if self.training_horizon != 1:
            raise ValueError("Recursive API prediction requires a model trained with horizon=1.")

        normalized = normalize_price_frame(prices, target_column=self.target_column)
        if len(normalized) < self.lookback:
            raise ValueError(f"At least {self.lookback} historical rows are required.")

        window = normalized[[self.target_column]].tail(self.lookback).astype("float32")
        scaled_window = self.scaler.transform(window).astype("float32")
        rolling = scaled_window.copy()
        scaled_predictions: list[float] = []

        for _ in range(horizon):
            x_value = torch.as_tensor(rolling[-self.lookback :][None, :, :], dtype=torch.float32).to(
                self.device
            )
            with torch.no_grad():
                pred_scaled = float(self.model(x_value).detach().cpu().numpy().reshape(-1)[0])
            scaled_predictions.append(pred_scaled)
            rolling = np.vstack([rolling, np.asarray([[pred_scaled]], dtype=np.float32)])

        predictions = self.scaler.inverse_transform(
            np.asarray(scaled_predictions, dtype=np.float32).reshape(-1, 1)
        ).reshape(-1)

        last_date = normalized.index.max()
        future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)

        return [
            Prediction(date=dt.date().isoformat(), predicted_close=float(value))
            for dt, value in zip(future_dates, predictions, strict=True)
        ]
