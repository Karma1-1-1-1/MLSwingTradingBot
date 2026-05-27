"""
config.py
=========
Every tunable parameter for the swing bot. No code. No imports beyond os.

If a value is referenced in more than one file, it lives here.
"""
from __future__ import annotations
import os

# ─── Universe ────────────────────────────────────────────────────────────────
# Liquid US large caps + ETFs only. Each one is high-volume enough that
# slippage stays under 5 bps for retail-sized orders.
SYMBOLS: list[str] = [
    "SPY", "QQQ", "IWM",                # broad ETFs (lower-vol anchors)
    "AAPL", "MSFT", "GOOGL", "AMZN",    # mega-cap tech
    "META", "NVDA", "AMD",              # higher-vol tech
    "JNJ", "PG", "KO",                  # defensive
    "JPM", "V", "MA",                   # financial
]

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_START = "2016-01-01"
DATA_END   = "2026-01-01"
INTERVAL   = "1d"
SPY_BENCHMARK = "SPY"          # used for regime and relative-strength
CACHE_DIR  = "data_cache"

# ─── Indicators ──────────────────────────────────────────────────────────────
RSI_PERIOD     = 14
EMA_SHORT      = 20
EMA_MID        = 50
EMA_LONG       = 200
ATR_PERIOD     = 14
VOL_LOOKBACK   = 20
BREAKOUT_LOOKBACK = 20         # N-day high for breakout entry

# ─── Strategy: entry ─────────────────────────────────────────────────────────
TREND_FILTER_EMA   = EMA_MID   # require Close > this EMA
TREND_HIERARCHY    = True      # also require EMA50 > EMA200
VOLUME_SPIKE_MIN   = 1.10      # entry needs Volume >= 1.1 × 20d avg
ATR_PCT_MAX        = 0.06      # skip if today's ATR / Close > 6% (too volatile)
ATR_PCT_MIN        = 0.005     # skip if ATR < 0.5% (dead stock, no edge)

# ─── Strategy: exit ──────────────────────────────────────────────────────────
# Stops are computed as max(percent_floor, ATR_multiplier * ATR_pct).
# This lets calm stocks have tight stops and volatile stocks have wider stops,
# while position size adjusts so dollar-risk stays constant.
STOP_LOSS_PCT_MIN   = 0.025    # 2.5% floor
STOP_LOSS_ATR_MULT  = 2.0      # widen to 2× ATR% if volatility demands
TAKE_PROFIT_R       = 3.0      # take profit at 3× initial risk (R-multiple)
TRAILING_STOP_R     = 2.0      # trail by 2× initial risk once in profit
TREND_BREAK_BARS    = 3        # exit if N consecutive closes below trend EMA
MAX_HOLD_BARS       = 25       # hard time-stop at ~5 weeks
COOLDOWN_BARS       = 3        # bars to skip same symbol after any exit

# ─── Risk: per-trade and portfolio ───────────────────────────────────────────
INIT_CASH               = 100_000.0
RISK_PER_TRADE_PCT      = 0.0075   # 0.75% of equity risked per trade (tight)
MAX_POSITION_PCT        = 0.20     # hard cap: no single position > 20% of equity
MAX_OPEN_POSITIONS      = 4
MAX_NEW_TRADES_PER_DAY  = 2        # avoid burst entries
PORTFOLIO_DD_KILL_PCT   = 0.08     # halt new entries if equity drops 8% from peak
LOSING_STREAK_PAUSE_N   = 4        # after N consecutive losers, pause new entries
LOSING_STREAK_PAUSE_BARS = 5       # for this many bars

# ─── ML model ────────────────────────────────────────────────────────────────
ML_TRAIN_RATIO    = 0.75
ML_N_ESTIMATORS   = 400
ML_MAX_DEPTH      = 6
ML_MIN_SAMPLES_LEAF = 8
ML_RANDOM_STATE   = 42
ML_MIN_PROB       = 0.55      # below this, the model says skip
ML_HARD_FLOOR     = 0.50      # if model AUC is weak, fall back to rules-only
ML_TARGET_LABEL_HORIZON = 25  # bars to simulate hypothetical exit
ML_TARGET_MIN_WIN_PCT   = 0.01  # only count >1% as a real win (per your spec)

# Normalised features only — NO raw OHLC, NO raw MA values.
FEATURE_COLS: list[str] = [
    "rsi",
    "ema_ratio_short",     # Close / EMA20
    "ema_ratio_mid",       # Close / EMA50
    "ema_ratio_long",      # Close / EMA200
    "ema_short_vs_mid",    # EMA20 / EMA50
    "ema_mid_vs_long",     # EMA50 / EMA200
    "atr_pct",             # ATR / Close
    "vol_spike",           # Volume / 20d avg volume
    "ret_1",               # 1d log return
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_20",              # 20d realised vol
    "high_vol_flag",       # vol_20 > vol_50
    "pullback_5",          # depth from 5-day high
    "drawdown_20",         # depth from 20-day high
    "range_compression",   # 5d range / 20d range
    "market_trend",        # SPY > SPY EMA50 (0/1)
    "rel_strength_20",     # ret_20(sym) - ret_20(SPY)
]

# ─── Ranking (which buy candidates to take when too many exist) ──────────────
RANKING_MIN_SCORE       = 0.50
RANKING_REGIME_MULT     = {"bull": 1.00, "chop": 0.60, "bear": 0.25}

# ─── Backtest realism ────────────────────────────────────────────────────────
FEE_BPS         = 5.0     # 5 bps per side = 0.05% (Alpaca-like, generous)
SLIPPAGE_BPS    = 5.0     # 5 bps adverse on entry and exit
WFO_TRAIN_MONTHS = 24
WFO_TEST_MONTHS  = 6
WFO_STEP_MONTHS  = 6      # roll forward by this much each window
WFO_MIN_TRADES_PER_WINDOW = 10

# ─── Paper trading: Alpaca ───────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "YOUR_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")
# Paper endpoint is hard-coded. Live endpoints are rejected at runtime.
ALPACA_PAPER_URL  = "https://paper-api.alpaca.markets"
# Reject these exact hosts (live trading endpoints)
ALPACA_LIVE_HOSTS = ("api.alpaca.markets", "live-api.alpaca.markets")

# DRY_RUN: do everything except submit orders.
DRY_RUN_DEFAULT   = True

# ─── Logging / output ────────────────────────────────────────────────────────
LOG_DIR             = "logs"
TRADES_CSV          = f"{LOG_DIR}/trades.csv"
DECISIONS_CSV       = f"{LOG_DIR}/decisions.csv"
POSITIONS_JSON      = f"{LOG_DIR}/open_positions.json"
EQUITY_CSV          = f"{LOG_DIR}/equity_curve.csv"
WFO_CSV             = f"{LOG_DIR}/walkforward.csv"
MODEL_PATH          = f"{LOG_DIR}/model.pkl"
SCALER_PATH         = f"{LOG_DIR}/scaler.pkl"
MODEL_METRICS_JSON  = f"{LOG_DIR}/model_metrics.json"
BACKTEST_SUMMARY    = f"{LOG_DIR}/backtest_summary.json"
