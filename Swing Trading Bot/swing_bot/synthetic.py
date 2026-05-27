"""
synthetic.py
============
Generate realistic synthetic OHLCV data for tests. Deterministic trend
+ GBM noise + periodic corrections, similar to the previous synthetic_data.py
but slimmer and seed-stable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ohlcv(
    symbol: str = "SYN",
    start: str = "2018-01-01",
    end: str = "2025-01-01",
    start_price: float = 100.0,
    annual_drift: float = 0.15,
    annual_vol: float = 0.18,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    dt = 1.0 / 252.0

    # GBM in log-space: drift + volatility shocks.
    # log_return[t] = mu*dt + sigma*sqrt(dt)*z[t]
    mu = np.log(1.0 + annual_drift)
    sigma = annual_vol
    log_returns = mu * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    log_returns[0] = 0.0  # anchor first bar

    # Periodic corrections layered on top as ADDITIVE log-return shocks.
    # Drop over 5 bars, full recovery over next 7 bars — NET NEUTRAL in
    # log-space so the long-run drift is preserved. Purpose is to provide
    # realistic pullbacks for the strategy to skip, not to drag the trend down.
    i = 90
    while i + 18 < n:
        total_drop = rng.uniform(0.08, 0.15)
        drop_per_bar = np.log(1.0 - total_drop / 5.0)
        # Recovery in log-space exactly offsets the drop across 7 bars
        bounce_per_bar = -5.0 * drop_per_bar / 7.0
        for j in range(5):
            log_returns[i + j] += drop_per_bar
        for j in range(5, 12):
            log_returns[i + j] += bounce_per_bar
        i += int(90 + rng.integers(-20, 40))

    close = start_price * np.exp(np.cumsum(log_returns))

    intraday = (sigma * np.sqrt(dt)) * 0.5
    nz = rng.standard_normal((n, 3))
    op = close * (1 + intraday * nz[:, 0] * 0.3)
    hi = np.maximum(op, close) * (1 + np.abs(nz[:, 1]) * intraday * 0.6)
    lo = np.minimum(op, close) * (1 - np.abs(nz[:, 2]) * intraday * 0.6)

    base_vol = 40_000_000
    vol = (base_vol * np.exp(0.35 * rng.standard_normal(n))).astype(int)
    vol = np.clip(vol, 500_000, None)

    df = pd.DataFrame({
        "Open": np.round(op, 4),
        "High": np.round(hi, 4),
        "Low": np.round(lo, 4),
        "Close": np.round(close, 4),
        "Volume": vol,
    }, index=dates)
    df.index.name = "Date"
    return df


def generate_multi(symbols: list[str], seed_base: int = 42, **kw) -> dict[str, pd.DataFrame]:
    return {s: generate_ohlcv(s, seed=seed_base + i, **kw) for i, s in enumerate(symbols)}
