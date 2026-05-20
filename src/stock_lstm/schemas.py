from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class PricePoint(BaseModel):
    date: date
    close: float = Field(gt=0)


class PredictRequest(BaseModel):
    prices: list[PricePoint] = Field(min_length=1)
    horizon: int = Field(default=5, ge=1, le=30)

    @field_validator("prices")
    @classmethod
    def dates_must_be_unique(cls, prices: list[PricePoint]) -> list[PricePoint]:
        dates = [item.date for item in prices]
        if len(dates) != len(set(dates)):
            raise ValueError("prices must not contain duplicated dates.")
        return prices


class YFinancePredictRequest(BaseModel):
    symbol: str = Field(default="AAPL", min_length=1, max_length=20)
    lookback_days: int = Field(default=180, ge=60, le=2000)
    horizon: int = Field(default=5, ge=1, le=30)


class PredictionItem(BaseModel):
    date: date
    predicted_close: float


class PredictResponse(BaseModel):
    symbol: str | None
    horizon: int
    predictions: list[PredictionItem]
    model_metadata: dict

