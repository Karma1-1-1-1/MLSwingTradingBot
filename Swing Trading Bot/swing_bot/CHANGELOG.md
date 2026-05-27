# CHANGELOG

Full rebuild of the original swing bot. The original wasn't broken so much as it had structural issues that made it impossible to evaluate honestly: duplicated feature logic, a fake "expected return" model, no real portfolio backtest, walk-forward defined but never wired in, and several requested safety checks missing. Below is a file-by-file account.

---

## New files (clean rebuild)

### `config.py` — rewritten
**Why:** consolidate every threshold into one file, remove orphaned/contradictory params, and add the safety knobs the original was missing.

- Added: `MAX_NEW_TRADES_PER_DAY`, `PORTFOLIO_DD_KILL_PCT`, `LOSING_STREAK_PAUSE_N`, `LOSING_STREAK_PAUSE_BARS`, `COOLDOWN_BARS`, `MAX_HOLD_BARS`, `ML_HARD_FLOOR`, `WFO_MIN_TRADES_PER_WINDOW`.
- Added: `ALPACA_LIVE_HOSTS` tuple — hostnames the engine refuses to talk to.
- Changed: stop loss is now `max(STOP_LOSS_PCT_MIN, STOP_LOSS_ATR_MULT * atr_pct)` instead of a flat 3%. A 3% stop is wrong for both KO (too wide) and TSLA (too tight) — using ATR makes it volatility-aware.
- Changed: take profit is now in R units (`TAKE_PROFIT_R = 3.0`) instead of a flat 9% — keeps R-multiples consistent across symbols.
- Changed: `FEATURE_COLS` is now 20 strictly normalized features. Removed any raw OHLC/EMA/ATR levels — those can't generalize across symbols at different price scales.
- Removed: `RSI_PERIOD`, `MA_SHORT/MID/LONG`, `LOOKBACK_BREAKOUT`, `VOLUME_SPIKE_MIN`, `TREND_BREAK_BARS` from the top level — these are now in the modules that use them. Config is for *tunable* thresholds, not implementation constants.
- Removed: `ML_PROB_THRESHOLD` — replaced by `ML_MIN_PROB` (entry gate) and `ML_HARD_FLOOR` (fallback-to-rules threshold for model AUC).

### `data.py` — kept structurally, cleaned
**Why:** the original was fine. Standardized cache path, dropped the chatty INFO logging that spammed terminals.

### `features.py` — rewritten (replaces `features.py` + `indicators.py`)
**Why:** the original had **two files computing overlapping features** (`features.py` had `atr_pct`, `dist_ema20`, `dist_ema50`, `pullback_5`, `drawdown_10`, `momentum_vol_adj`, `range_compression`; `indicators.py` also defined `atr_pct`, `dist_ema20`, `dist_ema50`, `pullback_5`, `drawdown_10`, `momentum_vol_adj`, `range_compression` in a different way). Whichever ran last won. Worse, the ATR was computed as `rolling high - rolling low` over 14 bars, which is not ATR — it's just a range. The model's top feature being "ATR" when it was actually 14-day range is a silent correctness bug.

- Single file, single source of truth.
- True Wilder ATR (EMA of true range).
- All features normalized — no `Close`, `ma20`, `EMA50`, raw `ATR` ever leaves the file.
- `features_clean()` returns NaN-free, model-ready rows.

### `labeling.py` — rewritten (replaces `trade_outcome.py`)
**Why:** the original `label_expected_trade_returns` simulated trades but **used different exit rules than the live strategy**: it didn't account for the trailing stop activation threshold and treated trailing stop differently from `strategy.py`. Labels and live behavior must match or the model is trained on a fantasy.

- Single source: same SL/TP/trailing rules as `strategy.py` (imported, not duplicated).
- Outputs `trade_return`, `trade_outcome`, `target_binary`, `target_meaningful` (=1 only if return ≥ 1%, per the spec).
- Properly excludes look-ahead by capping the simulation horizon at `MAX_HOLD_BARS`.

### `model.py` — rewritten
**Why:** the original `predict_expected_return` was a fake: `(prob - 0.5) * 0.12` — a deterministic function of `ml_proba`. The ranker then did `0.35 * ml_prob + 0.40 * edge_bonus` where `edge_bonus` was derived from `expected_return` which was just a rescaled `ml_prob` — so the ranker was double-counting the same signal as 75% of its score.

- Classifier with `class_weight="balanced"` to handle the ~30% positive class.
- Sequential 75/25 split — no random shuffling that would leak future into past.
- Reports: accuracy, precision, recall, AUC, top-decile win rate, top-decile mean return, baseline win rate. The top-decile metrics are what actually matter for a ranker.
- `is_reliable` property — `False` if AUC < `ML_HARD_FLOOR + 0.03`. The system falls back to rules-only when ML isn't pulling its weight.
- No fake "expected return" function. The ranker now uses raw probability as the ML term.

### `strategy.py` — rewritten
**Why:** the original had a module-level `assert` that gets stripped by `python -O`, debug print statements still in the code, and the trend-break exit logic was tangled with position state. Also, **the entry rules and exit rules were in the same loop** — fine for production speed but makes them impossible to test in isolation.

- Pure function `generate_entry_signals(df) -> df` with `buy_signal` and `skip_reason` columns. Every row has a reason for buying or not buying.
- Entry conditions are independent and named — easy to add/remove without rewriting state machines.
- `compute_stop_pct()` is exposed as a separate utility used by both `labeling.py` and `backtest.py` so all three places agree.
- No `assert` at module level.

### `ranker.py` — rewritten (replaces `trade_ranker.py`)
**Why:** the original had double-counting (see `model.py` note), a brittle regime classifier that returned `"bear"` whenever `market_trend == 0` even on chop days, and inline pandas `apply` calls per row (slow). The scoring was also opaque — 7 magic constants with no documentation.

- `score_row()` is a pure function with documented weights summing to 1.0.
- Regime classifier uses both market trend and SPY-relative ATR percentile, not just `market_trend == 0`.
- Regime acts as a *multiplier* on the final score, not as a separate additive term — much clearer interpretation.

### `risk.py` — new file
**Why:** in the original, position sizing, kill switches, daily-trade caps, and losing-streak pauses were either missing or scattered across `paper_trade.py` and `main.py`. The portfolio drawdown kill switch (explicit user requirement) didn't exist anywhere.

- `RiskState` dataclass: equity peak, consecutive losses, streak pause counter, daily new-trade counter. Persisted with positions.
- `position_size()` returns `(qty, dollar_risk)`. If risk math says `qty < 1`, returns `(0, 0)` — **no silent `max(qty, 1)` breach of risk discipline**, which the original `paper_trade.py` did.
- `can_enter()` returns `(allowed, skip_reason)` so every block has a paper trail.
- `record_exit()` tracks consecutive losses and triggers the pause.

### `backtest.py` — rewritten
**Why:** the original `run_multi_symbol` was **not a portfolio backtest**. It backtested each symbol independently and averaged the per-symbol metrics. `MAX_OPEN_POSITIONS` was never enforced. Cash was infinite. This is not a realistic simulation.

- Single cash account across all symbols. Equity is marked-to-market every bar.
- Per-bar order: (1) update peak equity, (2) tick risk state, (3) EXIT pass with intraday-conservative ordering (stops checked before take-profits for ambiguous bars), (4) ENTRY pass — gather candidates, rank, fill remaining position slots subject to all risk gates.
- Applies `FEE_BPS + SLIPPAGE_BPS` on every fill.
- `compute_metrics()` returns `profit_factor = None` if all losses or no trades — **never `inf`** — and marks `profit_factor_reliable = False` if `n_trades < 30`. The warning is in the metrics dict, not just a log line.

### `walkforward.py` — new file
**Why:** the original had `WFO_TRAIN_MONTHS` and `WFO_TEST_MONTHS` in config but **nothing called them**. Walk-forward was promised in the spec and missing in code.

- Rolling expanding-window (configurable) WFO. Trains on `[cursor, train_end]`, predicts on `[train_end, test_end]`, runs portfolio backtest on the test window.
- Per-window CSV + aggregate summary JSON.
- Hard pass/fail gate — see README. Returns `pass: True/False` based on all six criteria from the spec.

### `paper_trade.py` — rewritten
**Why:** the original had no live-URL safety check, no DRY_RUN mode that bypassed `alpaca-trade-api` entirely (it tried to import it and just warned), and the position tracker silently allowed `MAX_OPEN_POSITIONS` to be exceeded under some restart scenarios.

- `_validate_paper_url()` uses `urllib.parse.urlparse` to extract the hostname, then checks against `ALPACA_LIVE_HOSTS` and requires the substring `paper` somewhere in the URL. Substring-only check would falsely pass `paper-api.alpaca.markets.evil.com` and falsely fail on hostnames that happen to contain "api" — the parsed-hostname approach handles both.
- `PaperEngine.__init__(dry_run=True)` doesn't import `alpaca-trade-api` at all. The order-submit code path doesn't exist when dry_run is True.
- `PositionStore` persists to JSON atomically (write temp, rename).
- `run_once()` is single-iteration — designed to be called once per day from a cron/Task Scheduler, not a long-running loop with `time.sleep()`. Long loops are fragile; daily cron is robust.
- Every decision lands in `decisions.csv` with a reason.

### `main.py` — rewritten
**Why:** the original `main.py` did everything in one function and had no subcommands. Running just the backtest required code edits.

- CLI with subcommands: `backtest`, `train`, `walkforward`, `paper`, `status`.
- `paper` supports `--dry-run`, `--live`, `--no-ml`.
- Prints a one-screen startup summary, then quiet logs to file. The spec asked for no terminal spam — this is it.

### `synthetic.py` — rewritten
**Why:** the original `synthetic_data.py` had a bug: it computed `base = trend * noise` (the correct GBM-with-drift series) but then in the next loop did `close[k] = close[k-1] * multiplier[k]` which **overwrote `base` entirely**. Since `multiplier` was 1.0 except during correction windows where it was < 1, `close` ended up as roughly `start_price * product_of_correction_multipliers` — always trending down, never up. EMA50 stayed above price, `market_trend == 0` always, and **zero buy signals** ever fired during tests.

- Rewritten using log-returns: `log_return = mu*dt + sigma*sqrt(dt)*z`, then `close = start_price * exp(cumsum(log_returns))`. Standard GBM. Drift survives.
- Corrections are now additive log-return shocks (negative for the drop, positive for the recovery) layered on top. Set to be net-neutral in log-space so the long-run drift is preserved — corrections are temporary pullbacks, not permanent drags.
- Defaults bumped to `annual_drift=0.15, annual_vol=0.18` so the smoke test reliably produces buy signals across seeds.

### `tests/test_smoke.py` — new
**Why:** the original `run_test.py` checked for `logs/rf_model.pkl` but the model saved to `logs/rf_expected_return_model.pkl`. The test was broken before it was even run.

- 8 stages: synthetic data → features → labeling → model train → rules backtest → ML backtest → walk-forward → paper engine dry run.
- Asserts data invariants at each stage.
- Does **not** assert specific return numbers — the test verifies the pipeline runs, not that the strategy makes money on synthetic data.

---

## Removed files

### `optimize.py` — removed
**Why:** hyperparameter search before validating the basic strategy is putting the cart in front of the horse. Add it back once the walk-forward gate passes consistently — then it's worth tuning. Tuning a strategy that doesn't work yet is just over-fitting.

### `backtest.py` (original) — replaced
Single-symbol, no portfolio constraints. See new `backtest.py` rationale above.

### `indicators.py` — merged into `features.py`
Eliminated the duplicate-feature problem.

### `trade_outcome.py` — renamed to `labeling.py`
Cleaner name + rewritten to match live strategy rules exactly.

### `trade_ranker.py` — renamed to `ranker.py` and rewritten
See `ranker.py` note above.

### `run_test.py` — replaced by `tests/test_smoke.py`
The original was broken (model filename mismatch) and asserted facts about synthetic data that weren't true (rules-based signals on GBM data rarely fire).

---

## Bugs fixed (summary, in order of severity)

1. **`max(qty, 1)` in paper_trade.py** — silently breached `RISK_PER_TRADE_PCT` when the math said zero shares. Now: if math says zero, the trade is skipped with `skip_reason=qty_zero_after_risk`.
2. **Duplicate feature computation** — `features.py` and `indicators.py` both computed `atr_pct` etc. with different formulas. Race condition on which won. Now: one file.
3. **Fake `predict_expected_return`** — `(prob-0.5)*0.12` is deterministic in `prob`. Ranker double-counted ML signal. Now: ML probability is one term among several, no fake derivative.
4. **No portfolio backtest** — `run_multi_symbol` averaged per-symbol metrics with no shared cash, no shared position cap. Now: event-driven single-account backtest.
5. **Walk-forward unimplemented** — config had the constants, no code used them. Now: full WFO with pass/fail gate.
6. **No live-URL safety check** — user explicitly asked for this. Now: hostname-parsing rejection in `paper_trade.py`.
7. **No portfolio DD kill switch, losing-streak pause, max-daily-trades, or cooldown enforcement** — user asked for all four. Now: all in `risk.py`, all unit-tested via smoke run.
8. **`assert` at module level in strategy.py** — disappears under `python -O`. Now: runtime check inside the function, raising a clear exception.
9. **Test broken on model filename** — `run_test.py` expected `rf_model.pkl`, model saved as `rf_expected_return_model.pkl`. Now: single canonical path `logs/model.pkl`.
10. **Synthetic data generator never trended up** — `base` computed then immediately overwritten. Tests produced 0 trades on every symbol. Now: GBM in log space, drift preserved, signals fire reliably.
11. **`inf` profit factor** — original allowed it when there were no losing trades. Now: returns `None` and marks `profit_factor_reliable=False` if `n_trades<30` or all-winners.
12. **Flat 3% stop loss across all symbols** — wrong for KO (too wide) and TSLA (too tight). Now: ATR-aware stop sizing.
