"""
walkforward.py
==============
Rolling walk-forward validation. The only honest way to estimate live
performance from historical data.

For each window:
  1. Train on [t, t + train_months)
  2. Predict P(meaningful_win) on [t + train_months, t + train_months + test_months)
  3. Run the portfolio backtest with those out-of-sample predictions
  4. Record metrics

Roll forward by WFO_STEP_MONTHS. Aggregate at the end.

Pass/fail gate (paper-trade readiness):
  - >= 3 valid windows
  - At least 60% of windows have positive return
  - Median Sharpe > 0
  - Median max drawdown >= -15%
  - Average n_trades per window >= WFO_MIN_TRADES_PER_WINDOW
  - Median profit factor (where computable) > 1.10
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from backtest import BacktestResult, run_portfolio_backtest
from config import (
    WFO_MIN_TRADES_PER_WINDOW, WFO_STEP_MONTHS, WFO_TEST_MONTHS, WFO_TRAIN_MONTHS,
)
from features import features_clean
from labeling import label_trades
from model import SwingModel

log = logging.getLogger(__name__)


@dataclass
class WindowResult:
    window: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)].copy()


def run_walkforward(
    feature_dfs: dict[str, pd.DataFrame],
    use_ml: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    feature_dfs: {symbol: featured df with buy_signal column}
    Returns (per_window_dataframe, aggregate_metrics_with_pass_fail)
    """
    # Universe-wide date span
    all_dates = sorted(set().union(*[set(df.index) for df in feature_dfs.values()]))
    if not all_dates:
        return pd.DataFrame(), {"pass": False, "reason": "no_data"}

    start, end = all_dates[0], all_dates[-1]

    records: list[WindowResult] = []
    window_id = 0
    cursor = start

    while True:
        train_end = cursor + pd.DateOffset(months=WFO_TRAIN_MONTHS)
        test_end = train_end + pd.DateOffset(months=WFO_TEST_MONTHS)
        if test_end > end:
            break

        # Build training frame from ALL symbols' bars in [cursor, train_end)
        train_frames = []
        for sym, df in feature_dfs.items():
            sub = _slice(df, cursor, train_end)
            if len(sub) < 100:
                continue
            labeled = label_trades(sub)
            cleaned = features_clean(labeled)
            cleaned = cleaned.dropna(subset=["target_meaningful"])
            if not cleaned.empty:
                cleaned["symbol"] = sym
                train_frames.append(cleaned)

        if not train_frames:
            log.warning(f"WFO window {window_id}: no training rows — skipping.")
            cursor += pd.DateOffset(months=WFO_STEP_MONTHS)
            window_id += 1
            continue

        train_df = pd.concat(train_frames).sort_index()
        try:
            model = SwingModel()
            model.train(train_df)
        except Exception as exc:
            log.warning(f"WFO window {window_id}: model training failed: {exc}")
            cursor += pd.DateOffset(months=WFO_STEP_MONTHS)
            window_id += 1
            continue

        # Test slice + ML predictions per symbol
        test_feature_dfs: dict[str, pd.DataFrame] = {}
        ml_probs: dict[str, pd.Series] = {}
        for sym, df in feature_dfs.items():
            sub = _slice(df, train_end, test_end)
            if sub.empty:
                continue
            test_feature_dfs[sym] = sub
            if use_ml and model.is_reliable:
                ml_probs[sym] = model.predict_proba(sub)

        if not test_feature_dfs:
            cursor += pd.DateOffset(months=WFO_STEP_MONTHS)
            window_id += 1
            continue

        result: BacktestResult = run_portfolio_backtest(
            test_feature_dfs,
            ml_probs=ml_probs if use_ml else None,
            use_ml=use_ml and model.is_reliable,
            label=f"wfo_{window_id}",
        )

        records.append(WindowResult(
            window=window_id,
            train_start=cursor, train_end=train_end,
            test_start=train_end, test_end=test_end,
            metrics=result.metrics,
        ))

        log.info(
            f"WFO {window_id} [{train_end.date()} → {test_end.date()}] "
            f"return={result.metrics.get('total_return_pct')}% "
            f"sharpe={result.metrics.get('sharpe')} "
            f"trades={result.metrics.get('n_trades')} "
            f"maxdd={result.metrics.get('max_drawdown_pct')}%"
        )

        cursor += pd.DateOffset(months=WFO_STEP_MONTHS)
        window_id += 1

    if not records:
        return pd.DataFrame(), {"pass": False, "reason": "no_valid_windows"}

    # Flatten to dataframe
    flat = []
    for r in records:
        row = {
            "window": r.window,
            "train_start": r.train_start.date(),
            "train_end": r.train_end.date(),
            "test_start": r.test_start.date(),
            "test_end": r.test_end.date(),
        }
        row.update({k: v for k, v in r.metrics.items() if not isinstance(v, (dict, list))})
        flat.append(row)
    df = pd.DataFrame(flat)

    # Aggregate + pass/fail
    n = len(df)
    pf_vals = df["profit_factor"].dropna()
    agg = {
        "n_windows": int(n),
        "positive_return_pct_of_windows": round(float((df["total_return_pct"] > 0).mean() * 100), 1),
        "median_return_pct": round(float(df["total_return_pct"].median()), 3),
        "median_sharpe": round(float(df["sharpe"].median()), 3),
        "median_max_drawdown_pct": round(float(df["max_drawdown_pct"].median()), 3),
        "median_profit_factor": round(float(pf_vals.median()), 3) if not pf_vals.empty else None,
        "mean_n_trades": round(float(df["n_trades"].mean()), 1),
    }

    pass_checks = {
        "windows_>=3": n >= 3,
        "pct_positive_>=60": agg["positive_return_pct_of_windows"] >= 60.0,
        "median_sharpe_>0": agg["median_sharpe"] > 0,
        "median_maxdd_>=-15": agg["median_max_drawdown_pct"] >= -15.0,
        "avg_trades_>=min": agg["mean_n_trades"] >= WFO_MIN_TRADES_PER_WINDOW,
        "median_pf_>1.10": (agg["median_profit_factor"] or 0) > 1.10,
    }
    agg["checks"] = pass_checks
    agg["pass"] = all(pass_checks.values())

    return df, agg
