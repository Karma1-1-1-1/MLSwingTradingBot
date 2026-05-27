"""
strategy.py
===========
Pure-function signal generation. No I/O, no state across calls.

Entry rule (all must hold):
  • Close > EMA(TREND_FILTER_EMA)
  • EMA50 > EMA200 (if TREND_HIERARCHY)
  • Close >= max(Close, BREAKOUT_LOOKBACK days, shifted -1)  [breakout, no lookahead]
  • vol_spike >= VOLUME_SPIKE_MIN
  • ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX
  • market_trend != 0 (SPY above its EMA50) — bear regime skips entries

This module ONLY emits +1 (would-be buy) signals. Exit decisions are
made by the backtest / paper engine using each position's own stop/TP/trail —
because exits depend on entry price, peak price, and bars-held, which
are stateful and live with the position, not the bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ATR_PCT_MAX, ATR_PCT_MIN, BREAKOUT_LOOKBACK, EMA_LONG, EMA_MID,
    STOP_LOSS_ATR_MULT, STOP_LOSS_PCT_MIN, TREND_FILTER_EMA, TREND_HIERARCHY,
    VOLUME_SPIKE_MIN,
)


def compute_stop_pct(atr_pct: float) -> float:
    """Volatility-aware stop %. Lower bounded by config floor, upper at 12%."""
    if not np.isfinite(atr_pct):
        return STOP_LOSS_PCT_MIN
    return float(np.clip(max(STOP_LOSS_PCT_MIN, STOP_LOSS_ATR_MULT * atr_pct), STOP_LOSS_PCT_MIN, 0.12))


def generate_entry_signals(df: pd.DataFrame, require_market_trend: bool = True) -> pd.DataFrame:
    """
    Add a 'buy_signal' (0/1) column plus a 'skip_reason' column for traceability.

    Expects features built by features.build_features.
    """
    needed = ["Close", "ema_mid", "ema_long", "vol_spike", "atr_pct"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Run build_features first.")

    out = df.copy()
    # Pick the trend-filter EMA dynamically (matches TREND_FILTER_EMA value)
    trend_col = "ema_mid" if TREND_FILTER_EMA == EMA_MID else "ema_long"

    # Shift the rolling max so today's high doesn't include itself (no lookahead)
    breakout_level = out["Close"].rolling(BREAKOUT_LOOKBACK).max().shift(1)

    # Build masks
    cond_trend = out["Close"] > out[trend_col]
    cond_hier = (out["ema_mid"] > out["ema_long"]) if TREND_HIERARCHY else True
    cond_break = out["Close"] >= breakout_level
    cond_vol = out["vol_spike"] >= VOLUME_SPIKE_MIN
    cond_atr = (out["atr_pct"] >= ATR_PCT_MIN) & (out["atr_pct"] <= ATR_PCT_MAX)
    cond_regime = (out.get("market_trend", 1) == 1) if require_market_trend else True

    buy = cond_trend & cond_hier & cond_break & cond_vol & cond_atr & cond_regime
    out["buy_signal"] = buy.astype(int)

    # Skip reason — first failing condition wins, for log readability
    reason = pd.Series("ok", index=out.index, dtype=object)
    reason = reason.mask(~cond_trend, "trend_below_ema")
    reason = reason.mask(cond_trend & ~cond_hier, "ema_hierarchy_bad")
    reason = reason.mask(cond_trend & cond_hier & ~cond_break, "no_breakout")
    reason = reason.mask(cond_trend & cond_hier & cond_break & ~cond_vol, "weak_volume")
    reason = reason.mask(
        cond_trend & cond_hier & cond_break & cond_vol & ~cond_atr,
        "atr_out_of_band",
    )
    reason = reason.mask(
        cond_trend & cond_hier & cond_break & cond_vol & cond_atr & ~cond_regime,
        "bear_regime",
    )
    out["skip_reason"] = reason
    return out
