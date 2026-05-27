"""
trade_outcome.py — Phase 4 expected-return labeling

Instead of only labeling BUY signals, this labels every valid bar as:

    "If I entered here, what would the trade return have been?"

This gives the ML model far more training rows and lets it learn expected return,
not just win/loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRAILING_STOP_PCT


def label_expected_trade_returns(
    df: pd.DataFrame,
    max_holding_days: int = 20,
) -> pd.DataFrame:
    """
    Label every bar with hypothetical trade outcome.

    target = expected trade return from entering at that bar
    target_binary = 1 if return > 0, else 0

    This is better for Phase 4 because it trains on many rows, not only rare BUY signals.
    """
    out = df.copy()

    out["target"] = np.nan
    out["target_binary"] = np.nan
    out["trade_return"] = np.nan
    out["trade_outcome_reason"] = ""

    closes = out["Close"].values
    highs = out["High"].values if "High" in out.columns else closes
    lows = out["Low"].values if "Low" in out.columns else closes

    n = len(out)

    for i in range(n):
        if i + 1 >= n:
            continue

        entry = float(closes[i])

        if not np.isfinite(entry) or entry <= 0:
            continue

        stop_price = entry * (1 - STOP_LOSS_PCT)
        take_profit_price = entry * (1 + TAKE_PROFIT_PCT)

        highest_price = entry
        trailing_stop = entry * (1 - TRAILING_STOP_PCT)

        end_i = min(i + max_holding_days, n - 1)

        exit_price = float(closes[end_i])
        reason = "timeout"

        for j in range(i + 1, end_i + 1):
            high = float(highs[j])
            low = float(lows[j])
            close = float(closes[j])

            if not np.isfinite(high) or not np.isfinite(low) or not np.isfinite(close):
                continue

            highest_price = max(highest_price, high)
            trailing_stop = max(trailing_stop, highest_price * (1 - TRAILING_STOP_PCT))

            if low <= stop_price:
                exit_price = stop_price
                reason = "stop_loss"
                break

            if high >= take_profit_price:
                exit_price = take_profit_price
                reason = "take_profit"
                break

            if highest_price > entry and low <= trailing_stop:
                exit_price = trailing_stop
                reason = "trailing_stop"
                break

        raw_return = (exit_price - entry) / entry

        vol_adj = out["atr_pct"].iloc[i] if "atr_pct" in out.columns else 0.02

        trade_return = raw_return - (vol_adj * 0.5)

        trade_return = np.clip(trade_return, -0.10, 0.10)

        out.iloc[i, out.columns.get_loc("target")] = trade_return
        out.iloc[i, out.columns.get_loc("target_binary")] = 1 if trade_return > 0.01 else 0
        out.iloc[i, out.columns.get_loc("trade_return")] = trade_return
        out.iloc[i, out.columns.get_loc("trade_outcome_reason")] = reason

    return out