import numpy as np

from stock_lstm.features import create_sequences


def test_create_sequences_respects_lookback_and_horizon():
    values = np.arange(10, dtype=np.float32).reshape(-1, 1)

    x_values, y_values, positions = create_sequences(values, lookback=3, horizon=2)

    assert x_values.shape == (6, 3, 1)
    assert y_values.shape == (6, 1)
    assert positions.tolist() == [4, 5, 6, 7, 8, 9]
    assert x_values[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert y_values[0, 0] == 4.0

