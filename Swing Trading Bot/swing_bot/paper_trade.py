"""
paper_trade.py
==============
Alpaca paper-trading engine. Mirrors the backtest's decision logic so that
backtest outcomes are believable estimates of live behaviour.

Safety:
  • Hard reject of live Alpaca URLs (only paper URL allowed)
  • DRY_RUN mode: do everything except submit orders
  • Open positions persisted to JSON between runs
  • Duplicate-open-position prevention
  • All decisions logged with reason to logs/decisions.csv
  • All trades logged to logs/trades.csv
  • Loss-streak pause + portfolio kill switch + max daily entries enforced

This module performs ONE iteration (call `run_once`). Schedule it externally
(cron / Task Scheduler) — see README.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ALPACA_API_KEY, ALPACA_LIVE_HOSTS, ALPACA_PAPER_URL,
    ALPACA_SECRET_KEY, COOLDOWN_BARS, DECISIONS_CSV, DRY_RUN_DEFAULT, INIT_CASH,
    MAX_HOLD_BARS, MAX_NEW_TRADES_PER_DAY, MAX_OPEN_POSITIONS, ML_MIN_PROB,
    POSITIONS_JSON, RANKING_MIN_SCORE, SYMBOLS, TAKE_PROFIT_R, TRADES_CSV,
    TRAILING_STOP_R, TREND_BREAK_BARS, TREND_FILTER_EMA, EMA_MID,
)
from features import build_features
from labeling import label_trades  # not used at runtime but imported to make API explicit
from model import SwingModel
from ranker import Candidate, rank_candidates, score_row
from risk import RiskState, can_enter, position_size, update_equity_peak
from strategy import compute_stop_pct, generate_entry_signals

log = logging.getLogger(__name__)


# ─── Safety: reject live URLs ────────────────────────────────────────────────
def _validate_paper_url(url: str) -> None:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if host in ALPACA_LIVE_HOSTS:
        raise RuntimeError(
            f"REFUSING TO START: configured Alpaca URL '{url}' resolves to a "
            f"LIVE host ('{host}'). This bot is paper-only."
        )
    if "paper" not in host:
        raise RuntimeError(
            f"REFUSING TO START: Alpaca hostname '{host}' must contain 'paper'. "
            f"This bot is paper-only."
        )


_validate_paper_url(ALPACA_PAPER_URL)


# ─── Persistence helpers ─────────────────────────────────────────────────────
TRADE_FIELDS = [
    "timestamp", "symbol", "action", "qty", "entry_price", "exit_price",
    "stop_price", "tp_price", "trailing_stop", "ml_prob", "score", "regime",
    "pnl", "return_pct", "reason", "dry_run",
]
DECISION_FIELDS = [
    "timestamp", "date", "symbol", "action", "reason", "price", "qty",
    "ml_prob", "score", "regime", "extra",
]


def _ensure_csv(path: str, fields: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _append_csv(path: str, fields: list[str], row: dict) -> None:
    _ensure_csv(path, fields)
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(
            {k: row.get(k, "") for k in fields}
        )


# ─── Position store ──────────────────────────────────────────────────────────
@dataclass
class StoredPosition:
    symbol: str
    qty: int
    entry_price: float
    entry_date: str
    stop_pct: float
    stop_price: float
    tp_price: float
    peak_price: float
    trailing_stop: float
    bars_held: int = 0
    ml_prob: float = 0.0
    score: float = 0.0


class PositionStore:
    def __init__(self, path: str = POSITIONS_JSON):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception as exc:
                log.error(f"Could not parse {self.path}: {exc}")
                self._data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def all(self) -> dict[str, dict]:
        return dict(self._data)

    def has(self, sym: str) -> bool:
        return sym in self._data

    def get(self, sym: str) -> Optional[dict]:
        return self._data.get(sym)

    def upsert(self, pos: StoredPosition) -> None:
        self._data[pos.symbol] = asdict(pos)
        self.save()

    def remove(self, sym: str) -> Optional[dict]:
        d = self._data.pop(sym, None)
        self.save()
        return d

    def count(self) -> int:
        return len(self._data)


# ─── Broker wrapper ──────────────────────────────────────────────────────────
class AlpacaPaperBroker:
    """Thin shim. In DRY_RUN mode it never imports alpaca-trade-api."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self._api = None
        self._equity_cache = INIT_CASH
        if not dry_run:
            self._connect()

    def _connect(self):
        if ALPACA_API_KEY in ("", "YOUR_ALPACA_API_KEY"):
            raise RuntimeError(
                "ALPACA_API_KEY not set. Either export it or run with --dry-run."
            )
        try:
            import alpaca_trade_api as tradeapi  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "alpaca-trade-api not installed. pip install alpaca-trade-api"
            ) from e
        self._api = tradeapi.REST(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_URL, api_version="v2"
        )
        acct = self._api.get_account()
        self._equity_cache = float(acct.equity)
        log.info(f"Connected to Alpaca paper. Equity=${self._equity_cache:,.2f}")

    def equity(self) -> float:
        if self.dry_run:
            return self._equity_cache
        try:
            return float(self._api.get_account().equity)
        except Exception as exc:
            log.warning(f"Could not refresh equity: {exc}. Using cached.")
            return self._equity_cache

    def submit_buy(self, symbol: str, qty: int) -> bool:
        if self.dry_run:
            log.info(f"[DRY] BUY {qty} {symbol}")
            return True
        try:
            self._api.submit_order(symbol=symbol, qty=qty, side="buy",
                                    type="market", time_in_force="day")
            return True
        except Exception as exc:
            log.error(f"BUY failed {symbol}: {exc}")
            return False

    def submit_sell(self, symbol: str, qty: int) -> bool:
        if self.dry_run:
            log.info(f"[DRY] SELL {qty} {symbol}")
            return True
        try:
            self._api.submit_order(symbol=symbol, qty=qty, side="sell",
                                    type="market", time_in_force="day")
            return True
        except Exception as exc:
            log.error(f"SELL failed {symbol}: {exc}")
            return False


# ─── The engine ──────────────────────────────────────────────────────────────
class PaperEngine:
    def __init__(self, dry_run: bool = DRY_RUN_DEFAULT):
        self.broker = AlpacaPaperBroker(dry_run=dry_run)
        self.positions = PositionStore()
        self.state = RiskState(equity_peak=self.broker.equity())
        self.dry_run = dry_run
        _ensure_csv(TRADES_CSV, TRADE_FIELDS)
        _ensure_csv(DECISIONS_CSV, DECISION_FIELDS)

    # — public —
    def run_once(self, feature_dfs: dict[str, pd.DataFrame],
                  ml_probs: dict[str, pd.Series] | None = None,
                  use_ml: bool = True) -> dict:
        """
        One bar's worth of decisions across the universe.
        Caller is responsible for providing feature_dfs whose LAST bar
        is the current bar to act on.
        """
        now = datetime.now(timezone.utc)
        today_str = now.date().isoformat()
        self.state.reset_daily(now.date())
        self.state.tick_bar()

        equity = self.broker.equity()
        update_equity_peak(self.state, equity)

        trend_col = "ema_mid" if TREND_FILTER_EMA == EMA_MID else "ema_long"
        # Per-symbol most-recent bar
        last_rows: dict[str, pd.Series] = {}
        for sym, df in feature_dfs.items():
            if df.empty:
                continue
            last_rows[sym] = df.iloc[-1]

        # 1) EXITS first
        exits_executed = []
        for sym in list(self.positions.all().keys()):
            row = last_rows.get(sym)
            if row is None:
                continue
            pos = self.positions.get(sym)
            if pos is None:
                continue
            high = float(row.get("High", row["Close"]))
            low = float(row.get("Low", row["Close"]))
            close = float(row["Close"])
            trend_val = float(row[trend_col]) if trend_col in row else None
            exit_price, reason = self._check_exit(sym, pos, high, low, close, trend_val)
            if exit_price is not None:
                self._execute_exit(sym, pos, exit_price, reason, now)
                exits_executed.append(sym)

        # 2) ENTRIES
        candidates: list[Candidate] = []
        for sym, row in last_rows.items():
            if self.positions.has(sym):
                continue
            if int(row.get("buy_signal", 0)) != 1:
                self._log_decision(now, sym, "hold", "no_buy_signal", row.get("Close"))
                continue

            ml_prob = 0.5
            if use_ml and ml_probs is not None and sym in ml_probs:
                v = ml_probs[sym].iloc[-1] if len(ml_probs[sym]) else np.nan
                if np.isfinite(v):
                    ml_prob = float(v)
                    if ml_prob < ML_MIN_PROB:
                        self._log_decision(now, sym, "skip", f"ml_prob_low_{ml_prob:.2f}",
                                          row["Close"], ml_prob=ml_prob)
                        continue

            score, regime = score_row(row, ml_prob=ml_prob)
            if score < RANKING_MIN_SCORE:
                self._log_decision(now, sym, "skip", f"low_score_{score:.2f}",
                                  row["Close"], ml_prob=ml_prob, score=score, regime=regime)
                continue
            candidates.append(Candidate(
                symbol=sym, date=row.name, price=float(row["Close"]),
                atr_pct=float(row["atr_pct"]), ml_prob=ml_prob,
                rel_strength=float(row.get("rel_strength_20", 0.0) or 0.0),
                vol_spike=float(row.get("vol_spike", 1.0) or 1.0),
                momentum=float(row.get("ret_20", 0.0) or 0.0),
                trend_strength=float(row.get("ema_short_vs_mid", 1.0) or 1.0),
                regime=regime, score=score,
            ))

        slots = max(0, MAX_OPEN_POSITIONS - self.positions.count())
        daily = max(0, MAX_NEW_TRADES_PER_DAY - self.state.trades_today(now.date()))
        slots = min(slots, daily)
        chosen = rank_candidates(candidates, top_n=slots) if slots else []

        # Log rejected
        chosen_syms = {c.symbol for c in chosen}
        for c in candidates:
            if c.symbol not in chosen_syms:
                self._log_decision(now, c.symbol, "skip", "ranked_below_topN",
                                  c.price, ml_prob=c.ml_prob, score=c.score, regime=c.regime)

        # Enter
        entered = []
        for c in chosen:
            ok, why = can_enter(self.state, now.date(), equity, self.positions.count())
            if not ok:
                self._log_decision(now, c.symbol, "skip", why, c.price,
                                  ml_prob=c.ml_prob, score=c.score, regime=c.regime)
                continue
            stop_pct = compute_stop_pct(c.atr_pct)
            qty, _ = position_size(equity, c.price, stop_pct)
            if qty < 1:
                self._log_decision(now, c.symbol, "skip", "qty_zero", c.price,
                                  ml_prob=c.ml_prob, score=c.score, regime=c.regime)
                continue
            self._execute_entry(c.symbol, c.price, qty, c.atr_pct, c.ml_prob, c.score, c.regime, now)
            entered.append(c.symbol)

        return {
            "timestamp": now.isoformat(),
            "equity": equity,
            "open_positions": self.positions.count(),
            "exits": exits_executed,
            "entries": entered,
            "n_candidates": len(candidates),
        }

    # — internal: exit checks (live engine) —
    def _check_exit(self, sym: str, pos: dict, high: float, low: float,
                    close: float, trend_val: float | None) -> tuple[Optional[float], str]:
        # Update peak/trailing using current high
        peak = float(pos.get("peak_price", pos["entry_price"]))
        trail = float(pos.get("trailing_stop", pos["entry_price"]))
        bars = int(pos.get("bars_held", 0)) + 1

        if low <= float(pos["stop_price"]):
            return float(pos["stop_price"]), "stop_loss"
        if high >= float(pos["tp_price"]):
            return float(pos["tp_price"]), "take_profit"

        if high > peak:
            peak = high
            new_trail = peak - TRAILING_STOP_R * (float(pos["entry_price"]) * float(pos["stop_pct"]))
            if new_trail > trail:
                trail = new_trail
        if peak > float(pos["entry_price"]) and low <= trail:
            return trail, "trailing_stop"

        # Persist updated peak/trail/bars
        p = self.positions.get(sym)
        if p:
            p["peak_price"] = peak
            p["trailing_stop"] = trail
            p["bars_held"] = bars
            self.positions._data[sym] = p
            self.positions.save()

        # Trend break (we only check single-day signal here; consecutive count
        # would require persistence we don't keep across runs — single-day break
        # below trend is conservative but acceptable for a daily-frequency engine)
        if trend_val is not None and np.isfinite(trend_val):
            if close < trend_val and bars >= TREND_BREAK_BARS:
                return close, "trend_break"

        if bars >= MAX_HOLD_BARS:
            return close, "time_stop"
        return None, ""

    def _execute_entry(self, symbol: str, price: float, qty: int,
                        atr_pct: float, ml_prob: float, score: float, regime: str,
                        now: datetime):
        ok = self.broker.submit_buy(symbol, qty)
        if not ok:
            self._log_decision(now, symbol, "skip", "broker_buy_failed", price,
                              ml_prob=ml_prob, score=score, regime=regime)
            return
        stop_pct = compute_stop_pct(atr_pct)
        risk_per_share = price * stop_pct
        pos = StoredPosition(
            symbol=symbol, qty=qty, entry_price=price,
            entry_date=now.isoformat(),
            stop_pct=stop_pct,
            stop_price=price - risk_per_share,
            tp_price=price + TAKE_PROFIT_R * risk_per_share,
            peak_price=price,
            trailing_stop=price - TRAILING_STOP_R * risk_per_share,
            bars_held=0, ml_prob=ml_prob, score=score,
        )
        self.positions.upsert(pos)
        self.state.record_entry(now.date())

        _append_csv(TRADES_CSV, TRADE_FIELDS, {
            "timestamp": now.isoformat(), "symbol": symbol, "action": "BUY",
            "qty": qty, "entry_price": round(price, 4),
            "stop_price": round(pos.stop_price, 4),
            "tp_price": round(pos.tp_price, 4),
            "trailing_stop": round(pos.trailing_stop, 4),
            "ml_prob": round(ml_prob, 4), "score": round(score, 4),
            "regime": regime, "reason": "entry", "dry_run": self.dry_run,
        })
        self._log_decision(now, symbol, "buy",
                          f"score_{score:.2f}_ml_{ml_prob:.2f}_regime_{regime}",
                          price, qty=qty, ml_prob=ml_prob, score=score, regime=regime)
        log.info(f"[{symbol}] ENTER qty={qty} @ {price:.2f} SL={pos.stop_price:.2f} TP={pos.tp_price:.2f}")

    def _execute_exit(self, symbol: str, pos: dict, exit_price: float,
                       reason: str, now: datetime):
        ok = self.broker.submit_sell(symbol, int(pos["qty"]))
        if not ok:
            log.error(f"[{symbol}] sell failed; leaving position OPEN")
            return
        qty = int(pos["qty"])
        pnl = (exit_price - float(pos["entry_price"])) * qty
        ret = pnl / (float(pos["entry_price"]) * qty) if qty else 0.0
        self.positions.remove(symbol)
        self.state.record_exit(pnl)

        _append_csv(TRADES_CSV, TRADE_FIELDS, {
            "timestamp": now.isoformat(), "symbol": symbol, "action": "SELL",
            "qty": qty, "entry_price": round(float(pos["entry_price"]), 4),
            "exit_price": round(exit_price, 4),
            "stop_price": round(float(pos["stop_price"]), 4),
            "tp_price": round(float(pos["tp_price"]), 4),
            "pnl": round(pnl, 2), "return_pct": round(ret * 100, 3),
            "reason": reason, "dry_run": self.dry_run,
        })
        self._log_decision(now, symbol, "sell", reason, exit_price,
                          qty=qty, pnl=round(pnl, 2))
        log.info(f"[{symbol}] EXIT  reason={reason} @ {exit_price:.2f} pnl={pnl:+.2f}")

    def _log_decision(self, now: datetime, symbol: str, action: str, reason: str,
                       price, qty=None, ml_prob=None, score=None, regime=None, extra=None,
                       pnl=None):
        _append_csv(DECISIONS_CSV, DECISION_FIELDS, {
            "timestamp": now.isoformat(),
            "date": now.date().isoformat(),
            "symbol": symbol, "action": action, "reason": reason,
            "price": float(price) if price is not None else "",
            "qty": qty if qty is not None else "",
            "ml_prob": round(float(ml_prob), 4) if ml_prob is not None else "",
            "score": round(float(score), 4) if score is not None else "",
            "regime": regime or "",
            "extra": json.dumps(extra) if extra else (f"pnl={pnl}" if pnl is not None else ""),
        })

    def daily_summary(self, current_prices: dict[str, float] | None = None) -> None:
        eq = self.broker.equity()
        positions = self.positions.all()
        log.info("-" * 70)
        log.info(f"DAILY SUMMARY  equity=${eq:,.2f}  open={len(positions)}  "
                 f"peak=${self.state.equity_peak:,.2f}  "
                 f"streak={self.state.consecutive_losses}")
        if positions:
            cps = current_prices or {}
            for sym, p in positions.items():
                cur = float(cps.get(sym, p["entry_price"]))
                unr = (cur - float(p["entry_price"])) * int(p["qty"])
                log.info(f"  {sym:<6} qty={int(p['qty']):>5} entry={float(p['entry_price']):>8.2f} "
                         f"now={cur:>8.2f} trail={float(p['trailing_stop']):>8.2f} unrl={unr:>+9.2f}")
        log.info("-" * 70)


# ─── CLI helper for one-shot runs ────────────────────────────────────────────
def run_paper_once(symbols: list[str] = SYMBOLS, dry_run: bool = DRY_RUN_DEFAULT,
                    use_ml: bool = True) -> dict:
    """Convenience: fetch data, build features, predict, run one engine iteration."""
    from data import fetch_ohlcv

    spy = fetch_ohlcv("SPY")
    feature_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            raw = fetch_ohlcv(sym)
            feat = build_features(raw, spy)
            feat = generate_entry_signals(feat)
            feature_dfs[sym] = feat
        except Exception as exc:
            log.error(f"[{sym}] feature build failed: {exc}")

    ml_probs: dict[str, pd.Series] = {}
    if use_ml:
        model = SwingModel()
        try:
            model.load()
            if model.is_reliable:
                for sym, df in feature_dfs.items():
                    ml_probs[sym] = model.predict_proba(df)
            else:
                log.warning("Loaded model is not reliable enough — running rules-only")
                use_ml = False
        except FileNotFoundError:
            log.warning("No saved model — running rules-only")
            use_ml = False

    engine = PaperEngine(dry_run=dry_run)
    summary = engine.run_once(feature_dfs, ml_probs=ml_probs, use_ml=use_ml)
    engine.daily_summary({s: float(df["Close"].iloc[-1]) for s, df in feature_dfs.items()})
    return summary
