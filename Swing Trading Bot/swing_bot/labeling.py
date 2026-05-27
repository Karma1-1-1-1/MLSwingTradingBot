"""
labeling.py
===========
Generate the ML training target.

For every bar we ask: "If I had entered this bar and applied the same
stop / take-profit / trailing / time-stop rules the live strategy uses,
what would the outcome have been?"

Output columns:
    trade_return        : actual return of the simulated trade
    trade_outcome       : "stop_loss" | "take_profit" | "trailing_stop" | "timeout"
    target_binary       : 1 if trade_return >= ML_TARGET_MIN_WIN_PCT, else 0
    target_meaningful   : 1 only on real wins (excludes scratches near 0)

Important: we use the *same* SL/TP/trailing logic as the live strategy,
so the model is learning what the strategy itself would have produced —
not some idealised future-return curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ML_TARGET_LABEL_HORIZON, ML_TARGET_MIN_WIN_PCT,
    STOP_LOSS_ATR_MULT, STOP_LOSS_PCT_MIN, TAKE_PROFIT_R, TRAILING_STOP_R,
)


def label_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate forward trades and return df with target columns added."""
    if "atr_pct" not in df.columns:
        raise ValueError("df must include atr_pct (run build_features first)")

    out = df.copy()
    close = out["Close"].to_numpy(dtype=float)
    high = out["High"].to_numpy(dtype=float) if "High" in out.columns else close
    low = out["Low"].to_numpy(dtype=float) if "Low" in out.columns else close
    atr_pct = out["atr_pct"].to_numpy(dtype=float)

    n = len(out)
    horizon = ML_TARGET_LABEL_HORIZON

    trade_return = np.full(n, np.nan)
    outcome = np.array(["timeout"] * n, dtype=object)

    for i in range(n):
        if i + 1 >= n:
            continue
        entry = close[i]
        atrp = atr_pct[i]
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atrp):
            continue

        sl_pct = max(STOP_LOSS_PCT_MIN, STOP_LOSS_ATR_MULT * atrp)
        # cap at 12% so a single freak ATR doesn't allow a 30% stop
        sl_pct = min(sl_pct, 0.12)
        risk = entry * sl_pct
        stop_price = entry - risk
        tp_price = entry + TAKE_PROFIT_R * risk
        peak = entry
        trail_pct = TRAILING_STOP_R * sl_pct
        result = None

        end = min(i + horizon, n - 1)
        for j in range(i + 1, end + 1):
            hj, lj, cj = high[j], low[j], close[j]
            if not (np.isfinite(hj) and np.isfinite(lj) and np.isfinite(cj)):
                continue

            # Conservative ordering: assume the worst (stop) hits before the best (TP)
            # when both could occur intraday.
            if lj <= stop_price:
                trade_return[i] = (stop_price - entry) / entry
                outcome[i] = "stop_loss"
                result = True
                break
            if hj >= tp_price:
                trade_return[i] = (tp_price - entry) / entry
                outcome[i] = "take_profit"
                result = True
                break

            # Update trailing stop from new peak
            if hj > peak:
                peak = hj
            trail_stop = peak * (1 - trail_pct)
            if trail_stop > entry and lj <= trail_stop:
                trade_return[i] = (trail_stop - entry) / entry
                outcome[i] = "trailing_stop"
                result = True
                break

        if not result:
            # timeout
            trade_return[i] = (close[end] - entry) / entry
            outcome[i] = "timeout"

    out["trade_return"] = trade_return
    out["trade_outcome"] = outcome
    out["target_binary"] = (out["trade_return"] > 0).astype(float)
    out["target_meaningful"] = (out["trade_return"] >= ML_TARGET_MIN_WIN_PCT).astype(float)
    # Wipe labels we couldn't compute
    out.loc[~np.isfinite(out["trade_return"]), ["target_binary", "target_meaningful"]] = np.nan
    return out
