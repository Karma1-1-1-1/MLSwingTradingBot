"""
ranker.py
=========
When more candidates exist than open slots, decide which to take.

Ranking score is a weighted combination of:
  ml_prob, relative_strength, volume_spike, momentum, trend_strength.
Regime gates the score downward (chop, bear).

Caller passes a list of candidate rows for ONE date and gets back a sorted list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import RANKING_MIN_SCORE, RANKING_REGIME_MULT


@dataclass(frozen=True)
class Candidate:
    symbol: str
    date: pd.Timestamp
    price: float
    atr_pct: float
    ml_prob: float
    rel_strength: float
    vol_spike: float
    momentum: float
    trend_strength: float
    regime: str
    score: float


def classify_regime(market_trend: float, high_vol_flag: float, trend_strength: float) -> str:
    """3-state regime: bull / chop / bear."""
    if not np.isfinite(market_trend) or market_trend == 0:
        return "bear"
    if (np.isfinite(high_vol_flag) and high_vol_flag == 1) or (
        np.isfinite(trend_strength) and trend_strength < 1.01
    ):
        return "chop"
    return "bull"


def _safe(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def score_row(row: pd.Series, ml_prob: float | None = None) -> tuple[float, str]:
    """Return (score, regime) for one bar."""
    mt = _safe(row.get("market_trend", 1.0), 1.0)
    hvf = _safe(row.get("high_vol_flag", 0.0), 0.0)
    ts = _safe(row.get("ema_short_vs_mid", 1.0), 1.0)
    regime = classify_regime(mt, hvf, ts)

    p = _safe(ml_prob if ml_prob is not None else row.get("ml_proba", 0.5), 0.5)
    rs = _safe(row.get("rel_strength_20", 0.0), 0.0)
    vs = _safe(row.get("vol_spike", 1.0), 1.0)
    mom = _safe(row.get("ret_20", 0.0), 0.0)

    # Each component normalised to ~[0, 1]
    p_n = np.clip(p, 0.0, 1.0)
    rs_n = np.clip((rs + 0.05) / 0.10, 0.0, 1.0)
    vs_n = np.clip((vs - 1.0) / 1.0, 0.0, 1.0)
    mom_n = np.clip((mom + 0.05) / 0.15, 0.0, 1.0)
    ts_n = np.clip((ts - 0.98) / 0.10, 0.0, 1.0)

    # Weights sum to 1.0
    raw = 0.40 * p_n + 0.20 * rs_n + 0.15 * vs_n + 0.15 * mom_n + 0.10 * ts_n

    # Regime multiplier — bear regime should almost always block entries
    raw *= RANKING_REGIME_MULT.get(regime, 0.5)
    return float(round(raw, 4)), regime


def rank_candidates(candidates: list[Candidate], top_n: int) -> list[Candidate]:
    """Sort candidates by score descending, filter by RANKING_MIN_SCORE, take top N."""
    kept = [c for c in candidates if c.score >= RANKING_MIN_SCORE]
    kept.sort(key=lambda c: c.score, reverse=True)
    return kept[:top_n]
