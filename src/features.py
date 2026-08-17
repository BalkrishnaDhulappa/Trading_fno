"""Feature engineering for next-day NIFTY option direction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "vol_ratio_5_20",
    "rsi_14",
    "rsi_slope_3",
    "ma_ratio_5_20",
    "ma_ratio_10_50",
    "trend_20",
    "high_low_range",
    "atr_pct_14",
    "close_location",
    "volume_z_20",
    "gap_pct",
    "dow",
    "month",
    # Optional OC columns (filled with 0 when historical OC is unavailable)
    "pcr_oi",
    "pcr_vol",
    "oi_imbalance",
    "atm_iv_skew",
    "max_ce_oi_distance_pct",
    "max_pe_oi_distance_pct",
    "atm_premium_ratio",
]

OC_FEATURE_COLUMNS = [
    "pcr_oi",
    "pcr_vol",
    "oi_imbalance",
    "atm_iv_skew",
    "max_ce_oi_distance_pct",
    "max_pe_oi_distance_pct",
    "atm_premium_ratio",
]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_spot_features(spot: pd.DataFrame) -> pd.DataFrame:
    df = spot.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    df["return_1d"] = close.pct_change(1)
    df["return_2d"] = close.pct_change(2)
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["vol_ratio_5_20"] = df["volatility_5d"] / df["volatility_20d"].replace(0.0, np.nan)
    df["rsi_14"] = _rsi(close, 14)
    df["rsi_slope_3"] = df["rsi_14"].diff(3)
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["ma_ratio_5_20"] = ma5 / ma20 - 1.0
    df["ma_ratio_10_50"] = ma10 / ma50 - 1.0
    df["trend_20"] = close / ma20 - 1.0
    df["high_low_range"] = (high - low) / close.replace(0.0, np.nan)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_pct_14"] = tr.rolling(14).mean() / close.replace(0.0, np.nan)
    df["close_location"] = (close - low) / (high - low).replace(0.0, np.nan)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0.0, np.nan)
    df["volume_z_20"] = (volume - vol_mean) / vol_std
    df["gap_pct"] = open_ / close.shift(1) - 1.0
    df["dow"] = df["date"].dt.dayofweek.astype(float)
    df["month"] = df["date"].dt.month.astype(float)

    # Next-day labels for training
    df["next_return"] = close.shift(-1) / close - 1.0
    df["next_up"] = (df["next_return"] > 0).astype(int)
    df["next_option_side"] = np.where(df["next_up"] == 1, "CE", "PE")
    # Useful for options: only count days with a meaningful move
    df["next_move_gt_25bps"] = (df["next_return"].abs() >= 0.0025).astype(int)
    return df


def merge_oc_features(features: pd.DataFrame, oc_daily: pd.DataFrame | None) -> pd.DataFrame:
    df = features.copy()
    for col in OC_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    if oc_daily is None or oc_daily.empty:
        return df

    oc = oc_daily.copy()
    oc["date"] = pd.to_datetime(oc["date"]).dt.tz_localize(None).dt.normalize()
    keep = ["date"] + [c for c in OC_FEATURE_COLUMNS if c in oc.columns]
    oc = oc[keep].drop_duplicates("date", keep="last")
    df = df.drop(columns=OC_FEATURE_COLUMNS, errors="ignore")
    df = df.merge(oc, on="date", how="left")
    for col in OC_FEATURE_COLUMNS:
        df[col] = df[col].fillna(0.0)
    return df


def load_oc_feature_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()
    return frame


def make_training_frame(
    spot: pd.DataFrame,
    oc_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feats = build_spot_features(spot)
    feats = merge_oc_features(feats, oc_daily)
    usable = feats.dropna(subset=FEATURE_COLUMNS + ["next_up", "next_return"]).copy()
    # Drop final row if next-day label missing (already handled by dropna on next_*)
    x = usable[FEATURE_COLUMNS]
    y = usable["next_up"].astype(int)
    meta = usable[["date", "close", "next_return", "next_option_side"]]
    return x, y, meta
