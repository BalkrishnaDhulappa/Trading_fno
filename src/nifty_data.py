"""Historical NIFTY spot data helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "^NSEI"


def download_nifty_history(
    period: str = "5y",
    interval: str = "1d",
    ticker: str = DEFAULT_TICKER,
) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"No history downloaded for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    frame = raw.rename_axis("date").reset_index()
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        frame["date"] = pd.to_datetime(frame["date"])
    frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date")
    return frame.reset_index(drop=True)


def save_spot_history(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_spot_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()
    return frame.sort_values("date").reset_index(drop=True)
