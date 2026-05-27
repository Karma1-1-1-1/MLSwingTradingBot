"""
data.py
=======
Fetch and cache daily OHLCV. yfinance only. No indicator logic here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import CACHE_DIR, DATA_END, DATA_START, INTERVAL

log = logging.getLogger(__name__)


def fetch_ohlcv(
    symbol: str,
    start: str = DATA_START,
    end: str = DATA_END,
    interval: str = INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a tz-naive OHLCV DataFrame. Cached to parquet."""
    cache = Path(CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"{symbol}_{start}_{end}_{interval}.parquet"

    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    # Import lazily so the module doesn't crash if yfinance is missing during tests.
    import yfinance as yf

    raw = yf.download(
        symbol, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    df = raw[keep].copy()

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index.name = "Date"

    df = df.dropna(subset=keep)
    df = df[df["Volume"] > 0].sort_index()

    df.to_parquet(cache_file)
    log.info(f"[{symbol}] {len(df)} bars cached")
    return df


def fetch_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            out[s] = fetch_ohlcv(s)
        except Exception as exc:
            log.error(f"[{s}] fetch failed: {exc}")
    return out
