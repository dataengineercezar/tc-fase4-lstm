from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def normalize_price_frame(df: pd.DataFrame, target_column: str = "Close") -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    if target_column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(f"Column {target_column!r} not found. Available columns: {available}")

    out = df[[target_column]].copy()
    out[target_column] = pd.to_numeric(out[target_column], errors="coerce")
    out = out.dropna(subset=[target_column])
    out = out.sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "Date"

    if out.empty:
        raise ValueError("No valid price rows were found.")

    return out


def download_prices(
    symbol: str,
    start: str | date,
    end: str | date | None = None,
    target_column: str = "Close",
) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No data returned by Yahoo Finance for symbol {symbol!r}.")
    return normalize_price_frame(df, target_column=target_column)


def load_prices_csv(path: Path, target_column: str = "Close") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("CSV must include a Date column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return normalize_price_frame(df, target_column=target_column)

