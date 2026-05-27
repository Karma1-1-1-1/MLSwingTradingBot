"""
features.py
===========
Single source of truth for feature engineering.

Every feature here is NORMALISED (ratio, percentage, or 0/1 flag).
NO raw OHLC, NO raw EMA values, NO raw ATR.

If a feature isn't in FEATURE_COLS (config.py), the model will never see it.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from config import (
    ATR_PERIOD, EMA_LONG, EMA_MID, EMA_SHORT, FEATURE_COLS, RSI_PERIOD,
    VOL_LOOKBACK,
)

log = logging.getLogger(__name__)


# ─── Core indicators ─────────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """True ATR (Wilder), not high-low range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ─── Feature builder ─────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute every feature listed in FEATURE_COLS plus a few raw indicators
    the strategy module needs (ATR, EMAs) for stop-loss sizing.

    `spy_df` provides market_trend and rel_strength_20. If None, those become NaN.
    """
    if "Close" not in df.columns:
        raise ValueError("df must contain a 'Close' column")

    out = df.copy()
    c = out["Close"]
    h = out.get("High", c)
    l = out.get("Low", c)
    v = out.get("Volume", pd.Series(1.0, index=out.index))

    # EMAs (used by strategy.py for stops; ratios used as features)
    ema_s = c.ewm(span=EMA_SHORT, adjust=False).mean()
    ema_m = c.ewm(span=EMA_MID, adjust=False).mean()
    ema_l = c.ewm(span=EMA_LONG, adjust=False).mean()
    out["ema_short"] = ema_s
    out["ema_mid"] = ema_m
    out["ema_long"] = ema_l

    # ATR (raw — needed for stop sizing; only atr_pct is fed to the model)
    out["atr"] = _atr(h, l, c)
    out["atr_pct"] = out["atr"] / c

    # RSI
    out["rsi"] = _rsi(c)

    # EMA ratios (normalised)
    out["ema_ratio_short"] = c / ema_s
    out["ema_ratio_mid"] = c / ema_m
    out["ema_ratio_long"] = c / ema_l
    out["ema_short_vs_mid"] = ema_s / ema_m
    out["ema_mid_vs_long"] = ema_m / ema_l

    # Volume
    out["vol_avg_20"] = v.rolling(VOL_LOOKBACK).mean()
    out["vol_spike"] = v / out["vol_avg_20"]

    # Log returns at multiple horizons
    log_c = np.log(c.replace(0, np.nan))
    out["ret_1"] = log_c.diff(1)
    out["ret_3"] = log_c.diff(3)
    out["ret_5"] = log_c.diff(5)
    out["ret_10"] = log_c.diff(10)
    out["ret_20"] = log_c.diff(20)

    # Realised vol
    pct = c.pct_change()
    out["vol_20"] = pct.rolling(20).std()
    out["high_vol_flag"] = (out["vol_20"] > pct.rolling(50).std()).astype(float)

    # Setup quality
    out["pullback_5"] = (c - c.rolling(5).max()) / c.rolling(5).max()
    out["drawdown_20"] = (c - c.rolling(20).max()) / c.rolling(20).max()

    # Range compression: shrinking range often precedes a directional move
    range_5 = (h.rolling(5).max() - l.rolling(5).min()) / c
    range_20 = (h.rolling(20).max() - l.rolling(20).min()) / c
    out["range_compression"] = range_5 / (range_20 + 1e-9)

    # Market regime + relative strength via SPY
    if spy_df is not None and not spy_df.empty:
        spy_close = spy_df["Close"]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_close = spy_close.reindex(out.index).ffill()
        spy_ema50 = spy_close.ewm(span=EMA_MID, adjust=False).mean()
        out["market_trend"] = (spy_close > spy_ema50).astype(float)
        out["rel_strength_20"] = (
            np.log(c).diff(20) - np.log(spy_close).diff(20)
        )
    else:
        out["market_trend"] = np.nan
        out["rel_strength_20"] = np.nan

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        raise RuntimeError(f"Feature builder did not produce required columns: {missing}")

    return out


def features_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any model feature is NaN. Use only at training time."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df.dropna(subset=cols)
