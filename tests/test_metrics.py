import numpy as np

from stock_lstm.metrics import regression_metrics


def test_regression_metrics_returns_expected_values():
    metrics = regression_metrics(
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([12.0, 18.0, 33.0]),
    )

    assert round(metrics["mae"], 4) == 2.3333
    assert round(metrics["rmse"], 4) == 2.3805
    assert round(metrics["mape"], 4) == 13.3333

