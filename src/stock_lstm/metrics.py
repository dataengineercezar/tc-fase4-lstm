from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(true) == 0:
        raise ValueError("metrics require at least one observation.")

    errors = true - pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denom = np.clip(np.abs(true), 1e-8, None)
    mape = float(np.mean(np.abs(errors) / denom) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}

