from __future__ import annotations

import os
import time
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from stock_lstm.data import download_prices
from stock_lstm.predict import StockPricePredictor
from stock_lstm.schemas import PredictRequest, PredictResponse, YFinancePredictRequest

REQUEST_COUNT = Counter(
    "stock_lstm_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "stock_lstm_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)
MODEL_READY = Gauge("stock_lstm_model_ready", "Whether the model artifacts were loaded.")
PREDICTION_HORIZON = Histogram(
    "stock_lstm_prediction_horizon",
    "Prediction horizon requested by clients.",
    buckets=(1, 2, 3, 5, 7, 10, 15, 30),
)

app = FastAPI(
    title="Stock LSTM Forecast API",
    description="API for forecasting stock closing prices with an LSTM model.",
    version="0.1.0",
)


def artifacts_dir() -> Path:
    return Path(os.getenv("ARTIFACTS_DIR", "artifacts"))


@lru_cache(maxsize=1)
def get_predictor() -> StockPricePredictor:
    predictor = StockPricePredictor.from_artifacts(artifacts_dir())
    MODEL_READY.set(1)
    return predictor


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started_at

    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    REQUEST_COUNT.labels(request.method, path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
    return response


@app.get("/health")
def health() -> dict:
    try:
        predictor = get_predictor()
        return {
            "status": "ok",
            "model_ready": True,
            "artifacts_dir": str(artifacts_dir()),
            "symbol": predictor.metadata.get("symbol"),
            "lookback": predictor.lookback,
        }
    except FileNotFoundError as exc:
        MODEL_READY.set(0)
        return {
            "status": "degraded",
            "model_ready": False,
            "artifacts_dir": str(artifacts_dir()),
            "detail": str(exc),
        }


@app.get("/metadata")
def metadata() -> dict:
    try:
        return get_predictor().metadata
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
        frame = pd.DataFrame(
            [{"Date": item.date, "Close": item.close} for item in request.prices]
        ).set_index("Date")
        predictions = predictor.predict_future(frame, horizon=request.horizon)
        PREDICTION_HORIZON.observe(request.horizon)
        return PredictResponse(
            symbol=predictor.metadata.get("symbol"),
            horizon=request.horizon,
            predictions=[item.__dict__ for item in predictions],
            model_metadata={
                "lookback": predictor.lookback,
                "training_horizon": predictor.training_horizon,
                "target_column": predictor.target_column,
                "trained_at_utc": predictor.metadata.get("trained_at_utc"),
            },
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@app.post("/predict/yfinance", response_model=PredictResponse)
def predict_from_yfinance(request: YFinancePredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
        trained_symbol = str(predictor.metadata.get("symbol", "")).upper()
        requested_symbol = request.symbol.upper()
        if trained_symbol and requested_symbol != trained_symbol:
            raise ValueError(
                f"This model was trained for {trained_symbol}. "
                f"Train a new model before predicting {requested_symbol}."
            )

        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=request.lookback_days * 2)
        prices = download_prices(
            symbol=request.symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            target_column=predictor.target_column,
        ).tail(request.lookback_days)
        predictions = predictor.predict_future(prices, horizon=request.horizon)
        PREDICTION_HORIZON.observe(request.horizon)
        return PredictResponse(
            symbol=request.symbol,
            horizon=request.horizon,
            predictions=[item.__dict__ for item in predictions],
            model_metadata={
                "trained_symbol": predictor.metadata.get("symbol"),
                "lookback": predictor.lookback,
                "training_horizon": predictor.training_horizon,
                "target_column": predictor.target_column,
                "trained_at_utc": predictor.metadata.get("trained_at_utc"),
            },
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "stock_lstm.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
