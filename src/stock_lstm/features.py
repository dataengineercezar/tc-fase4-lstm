from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_lstm.config import DataConfig


@dataclass(frozen=True)
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_target_dates: list[str]
    scaler: object
    train_rows: int
    val_rows: int
    test_rows: int


def create_sequences(
    values: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError("values must be a 2D array shaped as [rows, features].")
    if len(values) < lookback + horizon:
        raise ValueError("Not enough rows to create at least one sequence.")

    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    target_positions: list[int] = []

    for end_idx in range(lookback, len(values) - horizon + 1):
        target_idx = end_idx + horizon - 1
        x_values.append(values[end_idx - lookback : end_idx])
        y_values.append(float(values[target_idx, 0]))
        target_positions.append(target_idx)

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32).reshape(-1, 1),
        np.asarray(target_positions, dtype=np.int64),
    )


def prepare_supervised_data(prices: pd.DataFrame, config: DataConfig) -> PreparedData:
    from sklearn.preprocessing import MinMaxScaler

    config.validate()
    if config.target_column not in prices.columns:
        raise ValueError(f"Dataframe must contain {config.target_column!r}.")

    target = prices[[config.target_column]].astype("float32")
    min_rows = config.lookback + config.horizon + 10
    if len(target) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows; got {len(target)}.")

    train_end = int(len(target) * config.train_ratio)
    val_end = int(len(target) * (config.train_ratio + config.val_ratio))

    scaler = MinMaxScaler()
    scaler.fit(target.iloc[:train_end])
    scaled_values = scaler.transform(target).astype("float32")

    x_all, y_all, target_positions = create_sequences(
        scaled_values,
        lookback=config.lookback,
        horizon=config.horizon,
    )

    train_mask = target_positions < train_end
    val_mask = (target_positions >= train_end) & (target_positions < val_end)
    test_mask = target_positions >= val_end

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Temporal split produced an empty train, validation, or test set. "
            "Increase the date range or reduce lookback/horizon."
        )

    test_dates = [prices.index[pos].date().isoformat() for pos in target_positions[test_mask]]

    return PreparedData(
        x_train=x_all[train_mask],
        y_train=y_all[train_mask],
        x_val=x_all[val_mask],
        y_val=y_all[val_mask],
        x_test=x_all[test_mask],
        y_test=y_all[test_mask],
        test_target_dates=test_dates,
        scaler=scaler,
        train_rows=int(train_mask.sum()),
        val_rows=int(val_mask.sum()),
        test_rows=int(test_mask.sum()),
    )
