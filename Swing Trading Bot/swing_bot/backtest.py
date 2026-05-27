"""
backtest.py
===========
Event-driven, portfolio-level backtest.

One cash account spans the universe. Position limits, daily-entry limits,
losing-streak pause, and drawdown kill switch are all enforced exactly as
in paper trading. This means backtest metrics actually reflect what live
execution would produce — modulo execution latency and partial fills.

Day loop (in chronological order across the union of all symbols' calendars):
  1. Mark equity using prior-close prices (mtm). Update equity peak.
  2. Tick the risk-state bar counter (for losing-streak pause).
  3. EXIT pass: every open position checks SL / TP / trailing / trend-break / time-stop.
  4. ENTRY pass: gather all symbols emitting buy_signal=1; build Candidates;
     filter by score and ML reliability; rank; take top N respecting limits.
  5. Record decisions (held / bought / sold / skipped + reason).

Costs:
  - Fees: FEE_BPS each side
  - Slippage: SLIPPAGE_BPS adverse on entry and on exit
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    COOLDOWN_BARS, FEE_BPS, INIT_CASH, MAX_HOLD_BARS, MAX_NEW_TRADES_PER_DAY,
    MAX_OPEN_POSITIONS, ML_MIN_PROB, RANKING_REGIME_MULT, SLIPPAGE_BPS,
    STOP_LOSS_ATR_MULT, STOP_LOSS_PCT_MIN, TAKE_PROFIT_R, TRAILING_STOP_R,
    TREND_BREAK_BARS, TREND_FILTER_EMA, EMA_MID,
)
from ranker import Candidate, rank_candidates, score_row
from risk import RiskState, can_enter, position_size, update_equity_peak
from strategy import compute_stop_pct

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    entry_date: pd.Timestamp
    stop_pct: float          # initial stop % (defines 1R)
    stop_price: float        # absolute stop, fixed at entry
    take_profit_price: float
    peak_price: float
    trailing_stop_price: float
    bars_held: int = 0


@dataclass
class TradeRecord:
    symbol: str
    qty: int
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    bars_held: int
    pnl: float
    return_pct: float
    exit_reason: str
    entry_reason: str = "breakout"


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    decisions: pd.DataFrame
    metrics: dict
    config_label: str = ""


# ─── core helpers ────────────────────────────────────────────────────────────
def _apply_slippage(price: float, side: str) -> float:
    """side: 'buy' -> price up by SLIPPAGE_BPS; 'sell' -> price down."""
    factor = SLIPPAGE_BPS / 10_000.0
    return price * (1 + factor) if side == "buy" else price * (1 - factor)


def _fee(notional: float) -> float:
    return abs(notional) * (FEE_BPS / 10_000.0)


def _open_position(symbol: str, dt: pd.Timestamp, price: float, qty: int, atr_pct: float) -> Position:
    stop_pct = compute_stop_pct(atr_pct)
    risk_per_share = price * stop_pct
    return Position(
        symbol=symbol,
        qty=qty,
        entry_price=price,
        entry_date=dt,
        stop_pct=stop_pct,
        stop_price=price - risk_per_share,
        take_profit_price=price + TAKE_PROFIT_R * risk_per_share,
        peak_price=price,
        trailing_stop_price=price - TRAILING_STOP_R * risk_per_share,
        bars_held=0,
    )


def _check_exit(
    pos: Position,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    trend_value: float | None,
    below_trend_run: int,
) -> tuple[Optional[float], str, int]:
    """
    Returns (exit_price, reason, new_below_trend_run). exit_price is None if no exit.
    Conservative intraday ordering: stop before TP if both hit the same bar.
    """
    if bar_low <= pos.stop_price:
        return pos.stop_price, "stop_loss", 0
    if bar_high >= pos.take_profit_price:
        return pos.take_profit_price, "take_profit", 0

    # Update peak & trailing stop using today's high
    if bar_high > pos.peak_price:
        pos.peak_price = bar_high
        pos.trailing_stop_price = max(
            pos.trailing_stop_price,
            pos.peak_price - TRAILING_STOP_R * (pos.entry_price * pos.stop_pct),
        )

    if pos.peak_price > pos.entry_price and bar_low <= pos.trailing_stop_price:
        return pos.trailing_stop_price, "trailing_stop", 0

    # Trend-break: N consecutive closes below trend EMA
    new_run = below_trend_run
    if trend_value is not None and np.isfinite(trend_value):
        if bar_close < trend_value:
            new_run += 1
            if new_run >= TREND_BREAK_BARS:
                return bar_close, "trend_break", 0
        else:
            new_run = 0

    # Time stop
    pos.bars_held += 1
    if pos.bars_held >= MAX_HOLD_BARS:
        return bar_close, "time_stop", 0

    return None, "", new_run


# ─── main backtest ───────────────────────────────────────────────────────────
def run_portfolio_backtest(
    feature_dfs: dict[str, pd.DataFrame],
    ml_probs: Optional[dict[str, pd.Series]] = None,
    use_ml: bool = True,
    label: str = "portfolio",
) -> BacktestResult:
    """
    feature_dfs: {symbol: dataframe with features, buy_signal, ema_mid, ema_long, atr_pct}
    ml_probs:    {symbol: Series of P(meaningful win) aligned to that symbol's index}, optional
    use_ml:      if False, ignore ml_probs entirely and run rules-only
    """
    if not feature_dfs:
        raise ValueError("feature_dfs is empty")

    # Build union of dates
    all_dates = sorted(set().union(*[set(df.index) for df in feature_dfs.values()]))

    cash = INIT_CASH
    positions: dict[str, Position] = {}
    state = RiskState(equity_peak=INIT_CASH)
    cooldown: dict[str, int] = {}     # symbol -> bars remaining
    below_trend_run: dict[str, int] = {s: 0 for s in feature_dfs}

    trades: list[TradeRecord] = []
    decisions: list[dict] = []
    equity_history: list[tuple[pd.Timestamp, float]] = []

    trend_col = "ema_mid" if TREND_FILTER_EMA == EMA_MID else "ema_long"

    for dt in all_dates:
        # 1) Mark-to-market and update peak
        equity_mtm = cash
        for sym, pos in positions.items():
            df = feature_dfs[sym]
            if dt in df.index:
                price = float(df.loc[dt, "Close"])
                equity_mtm += pos.qty * price
            else:
                equity_mtm += pos.qty * pos.entry_price  # carry value
        update_equity_peak(state, equity_mtm)
        equity_history.append((dt, equity_mtm))

        state.reset_daily(dt.date())
        state.tick_bar()

        # Decrement cooldowns
        for s in list(cooldown.keys()):
            cooldown[s] -= 1
            if cooldown[s] <= 0:
                del cooldown[s]

        # 2) EXIT pass
        for sym in list(positions.keys()):
            df = feature_dfs[sym]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            pos = positions[sym]
            exit_price, reason, new_run = _check_exit(
                pos,
                bar_high=float(row.get("High", row["Close"])),
                bar_low=float(row.get("Low", row["Close"])),
                bar_close=float(row["Close"]),
                trend_value=float(row[trend_col]) if trend_col in row else None,
                below_trend_run=below_trend_run.get(sym, 0),
            )
            below_trend_run[sym] = new_run
            if exit_price is not None:
                fill = _apply_slippage(exit_price, "sell")
                gross = fill * pos.qty
                fee = _fee(gross)
                cash += gross - fee
                pnl = (fill - pos.entry_price) * pos.qty - fee
                ret = pnl / (pos.entry_price * pos.qty) if pos.qty else 0.0
                trades.append(TradeRecord(
                    symbol=sym, qty=pos.qty,
                    entry_date=pos.entry_date, entry_price=pos.entry_price,
                    exit_date=dt, exit_price=fill,
                    bars_held=pos.bars_held, pnl=pnl,
                    return_pct=ret, exit_reason=reason,
                ))
                state.record_exit(pnl)
                decisions.append({
                    "date": dt, "symbol": sym, "action": "sell",
                    "reason": reason, "price": fill, "qty": pos.qty,
                    "pnl": round(pnl, 2),
                })
                cooldown[sym] = COOLDOWN_BARS
                del positions[sym]

        # 3) ENTRY pass — gather candidates
        candidates: list[Candidate] = []
        for sym, df in feature_dfs.items():
            if dt not in df.index:
                continue
            if sym in positions:
                continue
            if sym in cooldown:
                continue
            row = df.loc[dt]
            if int(row.get("buy_signal", 0)) != 1:
                continue

            # ML gate (only if ML is reliable)
            ml_prob = 0.5
            if use_ml and ml_probs is not None and sym in ml_probs:
                v = ml_probs[sym].get(dt, np.nan)
                if np.isfinite(v):
                    ml_prob = float(v)
                    if ml_prob < ML_MIN_PROB:
                        decisions.append({
                            "date": dt, "symbol": sym, "action": "skip",
                            "reason": f"ml_prob_low_{ml_prob:.2f}", "price": float(row["Close"]),
                        })
                        continue

            score, regime = score_row(row, ml_prob=ml_prob)
            candidates.append(Candidate(
                symbol=sym, date=dt, price=float(row["Close"]),
                atr_pct=float(row["atr_pct"]),
                ml_prob=ml_prob,
                rel_strength=float(row.get("rel_strength_20", 0.0) or 0.0),
                vol_spike=float(row.get("vol_spike", 1.0) or 1.0),
                momentum=float(row.get("ret_20", 0.0) or 0.0),
                trend_strength=float(row.get("ema_short_vs_mid", 1.0) or 1.0),
                regime=regime,
                score=score,
            ))

        # Top-of-day ranking
        slots = max(0, MAX_OPEN_POSITIONS - len(positions))
        daily_slots = max(0, MAX_NEW_TRADES_PER_DAY - state.trades_today(dt.date()))
        slots = min(slots, daily_slots)
        ranked = rank_candidates(candidates, top_n=slots) if slots > 0 else []

        # Log rejected candidates (didn't pass score)
        chosen_syms = {c.symbol for c in ranked}
        for c in candidates:
            if c.symbol not in chosen_syms:
                decisions.append({
                    "date": dt, "symbol": c.symbol, "action": "skip",
                    "reason": f"low_score_{c.score:.2f}", "price": c.price,
                })

        # Enter chosen
        for c in ranked:
            ok, why = can_enter(state, dt.date(), equity_mtm, len(positions))
            if not ok:
                decisions.append({
                    "date": dt, "symbol": c.symbol, "action": "skip",
                    "reason": why, "price": c.price,
                })
                continue
            stop_pct = compute_stop_pct(c.atr_pct)
            qty, _ = position_size(equity_mtm, c.price, stop_pct)
            if qty < 1:
                decisions.append({
                    "date": dt, "symbol": c.symbol, "action": "skip",
                    "reason": "qty_zero", "price": c.price,
                })
                continue
            fill = _apply_slippage(c.price, "buy")
            cost = fill * qty
            fee = _fee(cost)
            if cost + fee > cash:
                decisions.append({
                    "date": dt, "symbol": c.symbol, "action": "skip",
                    "reason": "insufficient_cash", "price": c.price,
                })
                continue
            cash -= cost + fee
            positions[c.symbol] = _open_position(c.symbol, dt, fill, qty, c.atr_pct)
            state.record_entry(dt.date())
            decisions.append({
                "date": dt, "symbol": c.symbol, "action": "buy",
                "reason": f"score_{c.score:.2f}_ml_{c.ml_prob:.2f}_regime_{c.regime}",
                "price": fill, "qty": qty,
            })

    # Final close-out: liquidate any remaining positions at last known close
    if positions:
        last_dt = all_dates[-1]
        for sym, pos in list(positions.items()):
            df = feature_dfs[sym]
            last_price = float(df["Close"].dropna().iloc[-1])
            fill = _apply_slippage(last_price, "sell")
            gross = fill * pos.qty
            fee = _fee(gross)
            cash += gross - fee
            pnl = (fill - pos.entry_price) * pos.qty - fee
            ret = pnl / (pos.entry_price * pos.qty) if pos.qty else 0.0
            trades.append(TradeRecord(
                symbol=sym, qty=pos.qty,
                entry_date=pos.entry_date, entry_price=pos.entry_price,
                exit_date=last_dt, exit_price=fill,
                bars_held=pos.bars_held, pnl=pnl, return_pct=ret,
                exit_reason="end_of_backtest",
            ))
            del positions[sym]

    equity_series = pd.Series(
        [v for _, v in equity_history],
        index=pd.DatetimeIndex([d for d, _ in equity_history], name="Date"),
        name="equity",
    )
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    decisions_df = pd.DataFrame(decisions)

    metrics = compute_metrics(equity_series, trades_df)
    return BacktestResult(
        equity_curve=equity_series,
        trades=trades_df,
        decisions=decisions_df,
        metrics=metrics,
        config_label=label,
    )


# ─── metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Compute backtest metrics with NaN-safe profit factor."""
    if equity.empty:
        return {"warning": "empty_equity_curve"}

    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    total_return = (end / start - 1.0) if start > 0 else 0.0

    # Daily returns from equity curve
    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Drawdown
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    max_dd = float(dd.min()) if len(dd) else 0.0

    # Trade-level
    n_trades = int(len(trades))
    if n_trades == 0:
        return {
            "total_return_pct": round(total_return * 100, 3),
            "max_drawdown_pct": round(max_dd * 100, 3),
            "sharpe": round(sharpe, 3),
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "expectancy_pct": None,
            "longest_losing_streak": 0,
            "warning": "no_trades",
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / n_trades

    gross_win = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    pf_reliable = n_trades >= 30
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = None  # do NOT report "inf" — it's misleading

    avg_win = float(wins["return_pct"].mean()) if not wins.empty else None
    avg_loss = float(losses["return_pct"].mean()) if not losses.empty else None
    expectancy = float(trades["return_pct"].mean())

    # Longest losing streak
    longest = cur = 0
    for pnl in trades["pnl"]:
        if pnl <= 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    out = {
        "total_return_pct": round(total_return * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "sharpe": round(sharpe, 3),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": (round(profit_factor, 3) if profit_factor is not None else None),
        "profit_factor_reliable": pf_reliable,
        "avg_win_pct": (round(avg_win * 100, 3) if avg_win is not None else None),
        "avg_loss_pct": (round(avg_loss * 100, 3) if avg_loss is not None else None),
        "expectancy_pct": round(expectancy * 100, 3),
        "longest_losing_streak": longest,
    }
    if not pf_reliable:
        out["warning"] = f"n_trades={n_trades} < 30 — profit factor unreliable"
    return out
