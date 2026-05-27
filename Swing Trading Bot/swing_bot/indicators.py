"""
indicators.py — FIXED
======================
KEY FIXES:
  1. yf.download("SPY") REMOVED from build_features().
     SPY data is now passed as an optional argument — callers fetch it once.
  2. Removed redundant features (volatility_10, ma_diff, price_vs_ma, spy_return).
  3. volume_spike fillna(1) replaced with proper NaN handling.
  4. add_ml_target: target threshold lowered to >0 (was >2% — caused 75% class imbalance).
  5. All functions remain pure: input → output, no side effects.
"""

import numpy as np
import pandas as pd

from config import (
    RSI_PERIOD, MA_SHORT, MA_MID, MA_LONG,
    FEATURE_COLS, ML_FORWARD_DAYS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core indicators
# ─────────────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df       = df.copy()
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[f"ma{MA_SHORT}"] = df["Close"].rolling(MA_SHORT).mean()
    df[f"ma{MA_MID}"]   = df["Close"].rolling(MA_MID).mean()
    df[f"ma{MA_LONG}"]  = df["Close"].rolling(MA_LONG).mean()
    return df


def add_ma_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        f"ma{MA_SHORT}": "ma_ratio_short",
        f"ma{MA_MID}":   "ma_ratio_mid",
        f"ma{MA_LONG}":  "ma_ratio_long",
    }
    for col, name in mapping.items():
        df[name] = df["Close"] / df[col]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_change_pct"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    vol_ma = df["Volume"].rolling(20).mean()
    # FIXED: no fillna(1) — let NaN propagate so FEATURE_COLS dropna works properly
    df["volume_spike"] = df["Volume"] / vol_ma
    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_close = np.log(df["Close"])
    df["daily_return"] = log_close.diff(1)
    df["return_3d"]    = log_close.diff(3)
    df["return_5d"]    = log_close.diff(5)
    df["momentum_10"]  = df["Close"].pct_change(10)
    df["momentum_20"]  = df["Close"].pct_change(20)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pct = df["Close"].pct_change()
    df["volatility_20"] = pct.rolling(20).std()
    vol_long            = pct.rolling(50).std()
    df["high_vol"]      = (df["volatility_20"] > vol_long).astype(int)
    return df


def add_market_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add market-regime features using SPY.

    Parameters
    ----------
    df      : symbol DataFrame (already has Close)
    spy_df  : pre-fetched SPY OHLCV DataFrame (same or wider date range).
              If None, market_trend and relative_strength are set to NaN.

    IMPORTANT: Callers are responsible for fetching SPY ONCE and passing it in.
               Do NOT call yf.download() inside this function.
    """
    df = df.copy()

    if spy_df is None or spy_df.empty:
        df["market_trend"]     = np.nan
        df["relative_strength"] = np.nan
        return df

    spy_close = spy_df["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]

    spy_close = spy_close.reindex(df.index).ffill()
    spy_ma50  = spy_close.rolling(50).mean()

    df["market_trend"]     = (spy_close > spy_ma50).astype(int)
    df["relative_strength"] = (
        df["Close"].pct_change(20) - spy_close.pct_change(20)
    )
    return df


def add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trend_strength"] = df["Close"] / df[f"ma{MA_LONG}"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ML target
# ─────────────────────────────────────────────────────────────────────────────

def add_ml_target(
    df: pd.DataFrame,
    forward_days: int = ML_FORWARD_DAYS,
    min_return: float = 0.0,      # FIXED: was 0.02 → caused 75% class imbalance
) -> pd.DataFrame:
    """
    Binary label: 1 if Close[t+N] > Close[t] * (1 + min_return), else 0.

    Using min_return=0.0 gives ~50/50 split on trending data — much better for
    a balanced classifier. If you want to predict "meaningful" moves, use 0.005
    (0.5%) rather than 0.02 (2%) which gives only 25% positives.
    """
    df          = df.copy()
    future      = df["Close"].shift(-forward_days)
    threshold   = df["Close"] * (1 + min_return)
    df["target"] = (future > threshold).astype(float)
    df.loc[future.isna(), "target"] = np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply all indicator/feature steps in order.

    Parameters
    ----------
    df      : raw OHLCV DataFrame
    spy_df  : pre-fetched SPY DataFrame (pass once from the caller).
              If None, SPY-derived features will be NaN and dropped from clean set.
    """
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_ma_ratios(df)
    df = add_volume_features(df)
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_trend_strength(df)
    df = add_market_features(df, spy_df)
    df = add_ml_target(df)

        # =========================
    # PHASE 4.5 ADVANCED FEATURES
    # =========================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Volatility (better version)
    df["atr"] = df["High"].rolling(14).max() - df["Low"].rolling(14).min()
    df["atr_pct"] = df["atr"] / df["Close"]

    # Distance from trend
    df["ema_20"] = df["Close"].ewm(span=20).mean()
    df["ema_50"] = df["Close"].ewm(span=50).mean()

    df["dist_ema20"] = (df["Close"] - df["ema_20"]) / df["ema_20"]
    df["dist_ema50"] = (df["Close"] - df["ema_50"]) / df["ema_50"]

    # Pullback quality (VERY IMPORTANT)
    df["pullback_5"] = (df["Close"] - df["Close"].rolling(5).max()) / df["Close"].rolling(5).max()

    # Local drawdown before entry
    df["drawdown_10"] = (df["Close"] - df["Close"].rolling(10).max()) / df["Close"].rolling(10).max()

    # Volatility-adjusted momentum
    df["momentum_vol_adj"] = df["momentum_20"] / (df["atr_pct"] + 1e-6)

    # Range compression (setup detection)
    df["range_5"] = (df["High"].rolling(5).max() - df["Low"].rolling(5).min()) / df["Close"]
    df["range_10"] = (df["High"].rolling(10).max() - df["Low"].rolling(10).min()) / df["Close"]
    df["range_compression"] = df["range_5"] / (df["range_10"] + 1e-6)
    return df


def get_clean_features(df):
    df = df.copy()

    # 🚨 DROP raw price columns (VERY IMPORTANT)
    drop_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # keep only numeric data
    df = df.select_dtypes(include=["number"])

    # drop missing
    df = df.dropna()

    return df

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data import fetch_ohlcv
    raw     = fetch_ohlcv("AAPL")
    spy_raw = fetch_ohlcv("SPY")
    feat    = build_features(raw, spy_raw)
    print(feat[FEATURE_COLS + ["target"]].tail(5))
    print(f"\nTarget distribution:\n{feat['target'].value_counts(normalize=True)}")
