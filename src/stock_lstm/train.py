from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from joblib import dump
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stock_lstm.config import DataConfig, ModelConfig, TrainingConfig
from stock_lstm.data import download_prices, load_prices_csv
from stock_lstm.features import PreparedData, prepare_supervised_data
from stock_lstm.metrics import regression_metrics
from stock_lstm.model import build_model
from stock_lstm.utils import ensure_dir, set_seed, to_jsonable, write_json

LOGGER = logging.getLogger(__name__)


def make_loader(x_values: np.ndarray, y_values: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(x_values, dtype=torch.float32),
        torch.as_tensor(y_values, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        preds = model(batch_x)
        loss = criterion(preds, batch_y)

        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses))


def predict_scaled(model: torch.nn.Module, x_values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x_values, dtype=torch.float32).to(device)
        preds = model(tensor).detach().cpu().numpy()
    return preds


def train_model(
    prepared: PreparedData,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    set_seed(training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(input_size=prepared.x_train.shape[-1], config=model_config).to(device)

    train_loader = make_loader(
        prepared.x_train,
        prepared.y_train,
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        prepared.x_val,
        prepared.y_val,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=1e-4,
    )

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, training_config.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        LOGGER.info("epoch=%s train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        if stale_epochs >= training_config.patience:
            LOGGER.info("early stopping at epoch=%s", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(
    model: torch.nn.Module,
    prepared: PreparedData,
) -> tuple[dict[str, float], np.ndarray]:
    device = next(model.parameters()).device
    scaled_preds = predict_scaled(model, prepared.x_test, device)
    y_true = prepared.scaler.inverse_transform(prepared.y_test).reshape(-1)
    y_pred = prepared.scaler.inverse_transform(scaled_preds).reshape(-1)
    return regression_metrics(y_true, y_pred), y_pred


def save_artifacts(
    model: torch.nn.Module,
    prepared: PreparedData,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    metrics: dict[str, float],
    history: dict[str, list[float]],
    last_observed_date: str,
) -> None:
    artifacts_dir = ensure_dir(data_config.artifacts_dir)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": to_jsonable(model_config),
            "input_size": int(prepared.x_train.shape[-1]),
        },
        artifacts_dir / "model.pt",
    )
    dump(prepared.scaler, artifacts_dir / "target_scaler.joblib")

    metadata = {
        "symbol": data_config.symbol,
        "target_column": data_config.target_column,
        "lookback": data_config.lookback,
        "horizon": data_config.horizon,
        "last_observed_date": last_observed_date,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_config": to_jsonable(data_config),
        "model_config": to_jsonable(model_config),
        "training_config": to_jsonable(training_config),
        "rows": {
            "train_sequences": prepared.train_rows,
            "validation_sequences": prepared.val_rows,
            "test_sequences": prepared.test_rows,
        },
    }
    write_json(artifacts_dir / "metadata.json", metadata)
    write_json(artifacts_dir / "metrics.json", {"metrics": metrics, "history": history})


def parse_args() -> argparse.Namespace:
    defaults = DataConfig()
    parser = argparse.ArgumentParser(description="Train an LSTM for stock close forecasting.")
    parser.add_argument("--symbol", default=os.getenv("STOCK_SYMBOL", defaults.symbol))
    parser.add_argument("--start", default=defaults.start_date)
    parser.add_argument("--end", default=defaults.end_date)
    parser.add_argument("--csv", type=Path, default=None, help="Optional local CSV with Date and Close.")
    parser.add_argument("--lookback", type=int, default=defaults.lookback)
    parser.add_argument("--horizon", type=int, default=defaults.horizon)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.getenv("ARTIFACTS_DIR", "artifacts")))
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--hidden-size", type=int, default=ModelConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=ModelConfig.num_layers)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    args = parse_args()

    data_config = DataConfig(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        lookback=args.lookback,
        horizon=args.horizon,
        artifacts_dir=args.artifacts_dir,
    )
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )

    data_config.validate()
    model_config.validate()
    training_config.validate()

    if args.csv:
        prices = load_prices_csv(args.csv, target_column=data_config.target_column)
    else:
        prices = download_prices(
            symbol=data_config.symbol,
            start=data_config.start_date,
            end=data_config.end_date,
            target_column=data_config.target_column,
        )

    LOGGER.info("loaded %s rows for %s", len(prices), data_config.symbol)
    prepared = prepare_supervised_data(prices, data_config)
    model, history = train_model(prepared, model_config, training_config)
    metrics, _ = evaluate_model(model, prepared)

    LOGGER.info("test metrics: %s", metrics)
    save_artifacts(
        model=model,
        prepared=prepared,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        metrics=metrics,
        history=history,
        last_observed_date=prices.index.max().date().isoformat(),
    )
    LOGGER.info("artifacts saved to %s", data_config.artifacts_dir)


if __name__ == "__main__":
    main()

