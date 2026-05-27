"""
risk.py
=======
Risk manager. Stateful: tracks equity peak, consecutive losers, daily entries.

Responsibilities:
- Compute share quantity from equity, stop %, and per-trade risk %
- Enforce MAX_OPEN_POSITIONS, MAX_NEW_TRADES_PER_DAY
- Enforce portfolio drawdown kill switch
- Enforce losing-streak pause
- Enforce MAX_POSITION_PCT cap

All "should we enter" decisions go through `can_enter`. Every block returns a
machine-readable reason so the decision log shows WHY we did or did not act.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from config import (
    LOSING_STREAK_PAUSE_BARS, LOSING_STREAK_PAUSE_N, MAX_NEW_TRADES_PER_DAY,
    MAX_OPEN_POSITIONS, MAX_POSITION_PCT, PORTFOLIO_DD_KILL_PCT,
    RISK_PER_TRADE_PCT,
)

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    equity_peak: float = 0.0
    consecutive_losses: int = 0
    streak_pause_until_bars_remaining: int = 0
    daily_new_trades: dict[str, int] = field(default_factory=dict)  # date_iso -> count

    def reset_daily(self, today: date):
        key = today.isoformat()
        # Drop everything except today's count
        self.daily_new_trades = {k: v for k, v in self.daily_new_trades.items() if k == key}
        self.daily_new_trades.setdefault(key, 0)

    def trades_today(self, today: date) -> int:
        return self.daily_new_trades.get(today.isoformat(), 0)

    def record_entry(self, today: date):
        key = today.isoformat()
        self.daily_new_trades[key] = self.daily_new_trades.get(key, 0) + 1

    def record_exit(self, pnl: float):
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= LOSING_STREAK_PAUSE_N:
                self.streak_pause_until_bars_remaining = LOSING_STREAK_PAUSE_BARS
                log.warning(
                    f"Losing streak: {self.consecutive_losses} losses. "
                    f"Pausing new entries for {LOSING_STREAK_PAUSE_BARS} bars."
                )
        else:
            self.consecutive_losses = 0

    def tick_bar(self):
        """Call once per simulated bar so the pause counter ticks down."""
        if self.streak_pause_until_bars_remaining > 0:
            self.streak_pause_until_bars_remaining -= 1


def position_size(equity: float, entry_price: float, stop_pct: float) -> tuple[int, float]:
    """
    Return (qty, dollar_risk) given equity and a percentage stop distance.

    qty satisfies: qty * (entry_price * stop_pct) ~= equity * RISK_PER_TRADE_PCT
    and qty * entry_price <= equity * MAX_POSITION_PCT.

    If risk math yields 0 shares, returns (0, 0). We do NOT round up to 1 share;
    that would silently breach the risk budget.
    """
    if entry_price <= 0 or stop_pct <= 0 or equity <= 0:
        return 0, 0.0
    dollar_risk_budget = equity * RISK_PER_TRADE_PCT
    risk_per_share = entry_price * stop_pct
    raw_qty = dollar_risk_budget / risk_per_share

    # Notional cap
    notional_cap_qty = (equity * MAX_POSITION_PCT) / entry_price
    qty = int(min(raw_qty, notional_cap_qty))
    if qty < 1:
        return 0, 0.0
    return qty, qty * risk_per_share


def can_enter(
    state: RiskState,
    today: date,
    equity: float,
    open_positions_count: int,
) -> tuple[bool, str]:
    """Top-level entry gate. Returns (allowed, reason)."""
    if open_positions_count >= MAX_OPEN_POSITIONS:
        return False, "max_open_positions"
    if state.trades_today(today) >= MAX_NEW_TRADES_PER_DAY:
        return False, "max_daily_trades"
    if state.streak_pause_until_bars_remaining > 0:
        return False, "losing_streak_pause"
    if state.equity_peak > 0:
        dd = 1.0 - equity / state.equity_peak
        if dd >= PORTFOLIO_DD_KILL_PCT:
            return False, "portfolio_dd_kill"
    return True, "ok"


def update_equity_peak(state: RiskState, equity: float):
    if equity > state.equity_peak:
        state.equity_peak = equity
