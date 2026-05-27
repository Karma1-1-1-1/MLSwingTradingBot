# swing_bot — paper-trading-ready swing system

A clean rebuild of the original bot focused on **risk control, consistency, and honesty over hype**.
Daily swing trading on liquid large caps, trend-following with ML as a *ranker* (not a trade forcer), event-driven portfolio backtest, walk-forward validation, and an Alpaca paper trading engine that physically cannot send orders to a live URL.

---

## What this bot is — and isn't

**It is:** a disciplined, risk-managed swing system that prioritizes profit factor, drawdown control, Sharpe, and trade quality over raw return. Every entry is filtered by trend, breakout strength, volume, and volatility regime; every position has a stop, take-profit, and trailing stop sized to the symbol's own ATR; every skip is logged with a reason.

**It is not** guaranteed to beat buy-and-hold SPY in nominal return during a strong bull market. Swing bots on liquid large caps typically *trail* a passive index in net return during bull runs. The realistic value proposition is **lower drawdown, lower volatility, and better risk-adjusted return** — not higher absolute return. If the walk-forward results do not pass the gates defined below, **the system is not ready for paper trading**, period.

---

## Project layout

```
swing_bot/
├── config.py          # All thresholds and switches. Edit this, not other files.
├── data.py            # yfinance fetch + parquet cache (one source of truth).
├── features.py        # All technical features. Single source — no duplicates.
├── labeling.py        # Forward-trade simulation labels (uses live SL/TP/trail).
├── model.py           # RandomForest classifier; reports AUC, top-decile metrics.
├── strategy.py        # Pure-function entry rules + ATR-aware stop sizing.
├── ranker.py          # Regime-aware candidate ranking.
├── risk.py            # Position sizing, kill switches, losing-streak pause.
├── backtest.py        # Event-driven portfolio backtest (one cash account).
├── walkforward.py     # Rolling WFO with pass/fail gate.
├── paper_trade.py     # Alpaca paper engine — rejects live URLs, persists state.
├── main.py            # CLI: backtest / train / walkforward / paper / status.
├── synthetic.py       # Synthetic OHLCV generator for smoke tests.
└── tests/test_smoke.py# End-to-end pipeline check on synthetic data.
```

---

## Install

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn yfinance pyarrow alpaca-trade-api
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn yfinance pyarrow alpaca-trade-api
```

`alpaca-trade-api` is only needed when you actually connect to Alpaca; the bot runs fine in dry-run without it.

---

## Quickstart

```powershell
# 1. Smoke test — verifies the pipeline runs end-to-end. NOT a strategy validation.
python -m tests.test_smoke

# 2. Backtest on real data (uses config.SYMBOLS)
python main.py backtest

# 3. Train the ML model (saves to logs/model.pkl)
python main.py train

# 4. Walk-forward validation — REQUIRED before paper trading
python main.py walkforward

# 5. Paper-trade dry run — exercises the engine, sends no orders
python main.py paper --dry-run

# 6. Paper-trade against Alpaca (requires env vars set, see below)
python main.py paper --live
```

### Setting Alpaca credentials (PowerShell)

```powershell
$env:ALPACA_API_KEY    = "PK..."
$env:ALPACA_SECRET_KEY = "..."
# Optional override; default is the paper URL. The bot REJECTS live URLs.
$env:ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"
```

### Setting Alpaca credentials (bash)

```bash
export ALPACA_API_KEY=PK...
export ALPACA_SECRET_KEY=...
```

---

## How the bot decides

### Entry — every condition must be true

| Filter | Check |
|---|---|
| Trend | `Close > EMA50` AND `EMA50 > EMA200` |
| Breakout | `Close >= 20-day high` (using prior-bar high, no look-ahead) |
| Volume | `volume_spike >= 1.10` (volume vs 20d avg) |
| Volatility band | `0.5% <= ATR_pct <= 6%` (avoid sleepers and chaos) |
| Market regime | SPY trend bullish (`market_trend == 1`) |
| Cooldown | No exit on this symbol in the last 3 bars |
| Risk gates | open positions < cap, daily new trades < cap, not in losing-streak pause, not in portfolio-DD kill |

If every check passes, the candidate is scored by the ranker (weighted: 40% ML prob, 20% relative strength, 15% volume, 15% momentum, 10% trend strength), multiplied by regime (`bull=1.0`, `chop=0.6`, `bear=0.25`). Only the highest-scoring candidates fill open position slots.

### Exit — first hit wins (intraday-safe ordering: stops checked before take-profits)

1. **Stop loss** — `max(2.5%, 2.0 × ATR_pct)` below entry, capped at 12%
2. **Take profit** — `3R` (3× initial risk distance)
3. **Trailing stop** — activates after `2R` of gain, trails by the initial risk distance
4. **Trend break** — close below EMA50 for ≥1 bar AND price below entry
5. **Time stop** — 25 bars (~5 weeks) regardless of P&L

### Skip reasons — every blocked entry is logged

`no_trend`, `no_breakout`, `low_volume`, `vol_too_low`, `vol_too_high`, `bear_regime`, `cooldown`, `score_too_low`, `slot_full`, `daily_cap`, `streak_pause`, `dd_kill`, `duplicate_position`, `qty_zero_after_risk`

All decisions land in `logs/decisions.csv` with one row per candidate per day.

### Position sizing

```
qty = floor( (equity * RISK_PER_TRADE_PCT) / (price * stop_pct) )
```

If `qty < 1`, the trade is **skipped** — no silent rounding up to 1 share. Position notional is also capped at `MAX_POSITION_PCT` of equity.

### Risk kill-switches

- **Losing-streak pause** — after 4 consecutive losses, freeze new entries for 5 bars.
- **Portfolio drawdown kill** — if equity drops 8% from peak, freeze new entries until recovery.
- **Max open positions** — 4.
- **Max new trades per day** — 2.
- **Cooldown** — 3 bars after any exit on a symbol.

All of these are in `config.py`. Tune them deliberately, not impulsively.

---

## Logs and outputs

| Path | What's in it |
|---|---|
| `logs/decisions.csv` | One row per candidate per day with skip reason. |
| `logs/trades.csv` | Every entry and exit, with reason and P&L. |
| `logs/positions.json` | Open positions, persisted across restarts. |
| `logs/model.pkl` | Trained classifier. |
| `logs/scaler.pkl` | Feature scaler. |
| `logs/model_metrics.json` | AUC, precision/recall, top-decile stats. |
| `logs/wfo_windows.csv` | One row per walk-forward window. |
| `logs/wfo_summary.json` | Aggregate WFO metrics + pass/fail flag. |
| `data_cache/*.parquet` | Cached yfinance pulls. Delete to force a refresh. |

---

## Pass/fail criteria before paper trading

The bot **must not** be put on paper unless `python main.py walkforward` on real data produces:

| Check | Threshold |
|---|---|
| Windows completed | ≥ 3 |
| Windows with positive return | ≥ 60% |
| Median Sharpe | > 0 |
| Median max drawdown | ≥ -15% |
| Median profit factor | > 1.10 |
| Mean trades per window | ≥ `WFO_MIN_TRADES_PER_WINDOW` (default 8) |

`walkforward.py` evaluates all of these and prints `pass: True / False`. **If `pass: False`, the bot is not paper-ready.** Re-tuning thresholds in `config.py` to force `pass: True` is the textbook definition of overfitting to the validation set — don't do it.

There is one more requirement that is *not* automatic: **the ML model must add value**. If `model_metrics.json` shows `roc_auc < 0.53`, treat the model as noise and run rules-only (`python main.py paper --dry-run --no-ml`). The ranker still works without ML; the ML term just contributes 0.

---

## Safety: why this bot cannot send live orders

`paper_trade.py` validates the Alpaca base URL on every startup using `urllib.parse.urlparse`. The hostname is checked against a hard-coded blocklist (`api.alpaca.markets`, `live-api.alpaca.markets`). If the hostname matches a live host, or if the URL string doesn't contain the substring `paper`, **the engine refuses to initialize** and raises `RuntimeError`. There is no override flag.

Additionally, `--dry-run` never imports `alpaca-trade-api` at all — there is no code path that could send an order to anything in dry-run mode.

---

## Honest expectations

- The smoke test passes on synthetic data. That tells you the code **runs**, not that the **strategy works** on your universe.
- On the bundled 15-stock universe, expect single-digit annual returns with mid-single-digit drawdowns *if* WFO passes. If WFO fails, the realistic expectation is "this strategy doesn't work on this universe with these parameters."
- The ML model on synthetic data hits AUC ~0.55 — a small but real edge. On real noisy market data, expect AUC closer to 0.52–0.55. Anything above 0.60 on a held-out walk-forward window is suspiciously high and worth auditing for leakage.
- Profit factor with fewer than 30 trades is marked unreliable. **Trust this flag.** A 2-trade backtest with 100% win rate proves nothing.
- This bot will **not** make you rich. It is designed to lose less than you would on your own discretionary trades, and to compound that lower-loss profile over time. That's the entire pitch.

---

## CLI reference

```
python main.py backtest               # Run full-history portfolio backtest
python main.py train                  # Train and save ML model
python main.py walkforward            # Rolling walk-forward + pass/fail gate
python main.py paper --dry-run        # Single iteration, no broker
python main.py paper --live           # Single iteration against Alpaca paper
python main.py paper --dry-run --no-ml# Rules-only mode (ML weight = 0)
python main.py status                 # Print open positions + recent decisions
```

All commands honor `config.SYMBOLS`. Edit `config.py` to change the universe.

---

## When to retrain

- Monthly, if running paper continuously.
- After any non-trivial change to `features.py`, `labeling.py`, or the entry rules.
- If you change `config.STOP_LOSS_*` or `TAKE_PROFIT_R` — the labels embed these constants.

Retraining is `python main.py train`. It will overwrite `logs/model.pkl` and `logs/model_metrics.json`.
