"""
trade_ranker.py — Phase 4 expected-return ranking

Ranks trades using:
- expected return
- model quality score
- relative strength
- trend strength
- volume participation
- regime
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from config import (
    ML_PROB_THRESHOLD,
    RANKING_MIN_SCORE,
    RANKING_TOP_N_PER_DAY,
    REGIME_BEAR_SIZE_MULT,
    REGIME_CHOP_SIZE_MULT,
    REGIME_BULL_SIZE_MULT,
)


@dataclass(frozen=True)
class TradeCandidate:
    symbol: str
    price: float
    ml_prob: float
    expected_return: float
    score: float
    regime: str
    size_multiplier: float


def _safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def classify_regime(row: pd.Series) -> str:
    market_trend = int(_safe_float(row.get("market_trend", 0), 0))
    high_vol = int(_safe_float(row.get("high_vol", 0), 0))
    trend_strength = _safe_float(row.get("trend_strength", 1.0), 1.0)

    if market_trend == 0:
        return "bear"

    if high_vol == 1 or trend_strength < 1.01:
        return "chop"

    return "bull"


def regime_size_multiplier(regime: str) -> float:
    if regime == "bear":
        return REGIME_BEAR_SIZE_MULT
    if regime == "chop":
        return REGIME_CHOP_SIZE_MULT
    return REGIME_BULL_SIZE_MULT


def score_row(row: pd.Series) -> float:
    ml_prob = _safe_float(row.get("ml_proba", 0.0), 0.0)
    expected_return = _safe_float(row.get("expected_return", 0.0), 0.0)

    rel_strength = _safe_float(row.get("relative_strength", 0.0), 0.0)
    volume_spike = _safe_float(row.get("volume_spike", 1.0), 1.0)
    momentum_20 = _safe_float(row.get("momentum_20", 0.0), 0.0)
    trend_strength = _safe_float(row.get("trend_strength", 1.0), 1.0)

    regime = classify_regime(row)

    edge_bonus = np.clip((expected_return + 0.02) / 0.10, 0.0, 1.0)
    rs_bonus = np.clip((rel_strength + 0.04) / 0.14, 0.0, 1.0)
    vol_bonus = np.clip((volume_spike - 1.0) / 1.0, 0.0, 1.0)
    mom_bonus = np.clip((momentum_20 + 0.03) / 0.18, 0.0, 1.0)
    trend_bonus = np.clip((trend_strength - 0.98) / 0.18, 0.0, 1.0)

    regime_bonus = {
        "bull": 1.00,
        "chop": 0.55,
        "bear": 0.10,
    }.get(regime, 0.50)

    score = (
        0.35 * np.clip(ml_prob, 0.0, 1.0)
        + 0.40 * edge_bonus
        + 0.10 * rs_bonus
        + 0.05 * vol_bonus
        + 0.05 * mom_bonus
        + 0.03 * trend_bonus
        + 0.02 * regime_bonus
    )

    # stronger separation
    if expected_return < 0:
        score *= 0.6

    
    elif expected_return > 0.02:
        score *= 1.2

    if regime == "bear":
        score *= 0.60

    return float(round(score, 4))


def add_trade_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["market_regime"] = out.apply(classify_regime, axis=1)
    out["trade_score"] = out.apply(score_row, axis=1)
    out["size_multiplier"] = out["market_regime"].map(regime_size_multiplier).astype(float)

    if "ml_proba" in out.columns:
        confidence = 0.70 + 0.80 * np.clip(
            (out["ml_proba"].fillna(0.0) - ML_PROB_THRESHOLD) / 0.25,
            0,
            1,
        )
    else:
        confidence = 0.70

    if "expected_return" in out.columns:
        edge_mult = 0.70 + 0.80 * np.clip(
            (out["expected_return"].fillna(0.0) + 0.01) / 0.08,
            0,
            1,
        )
    else:
        edge_mult = 1.0

    out["ai_size_multiplier"] = np.clip(
        out["size_multiplier"] * confidence * edge_mult,
        0.20,
        1.50,
    )

    return out


def apply_ranked_trade_selection(
    signal_dfs: Dict[str, pd.DataFrame],
    top_n_per_day: int = RANKING_TOP_N_PER_DAY,
    min_score: float = RANKING_MIN_SCORE,
) -> Dict[str, pd.DataFrame]:
    ranked = {sym: add_trade_scores(df) for sym, df in signal_dfs.items()}

    for sym, df in ranked.items():
        df["signal_ranked"] = np.where(df["signal_filtered"] == -1, -1, 0)

    all_dates = sorted(set().union(*[set(df.index) for df in ranked.values()]))

    for dt in all_dates:
        candidates = []

        for sym, df in ranked.items():
            if dt not in df.index:
                continue

            row = df.loc[dt]

            if int(row.get("signal_filtered", 0)) != 1:
                continue

            ml_prob = _safe_float(row.get("ml_proba", 0.0), 0.0)

            score = _safe_float(row.get("trade_score", 0.0), 0.0)

            if score < min_score:
                continue

            candidates.append((sym, score))

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        selected = candidates[:top_n_per_day]

        for sym, _score in selected:
            ranked[sym].loc[dt, "signal_ranked"] = 1

    return ranked


def latest_candidates(signal_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for sym, df in signal_dfs.items():
        scored = add_trade_scores(df)
        latest = scored.iloc[-1]

        if int(latest.get("signal_filtered", 0)) != 1:
            continue

        ml_prob = _safe_float(latest.get("ml_proba", 0.0), 0.0)

        rows.append({
            "symbol": sym,
            "price": float(latest["Close"]),
            "ml_prob": float(latest.get("ml_proba", 0.0)),
            "expected_return": float(latest.get("expected_return", 0.0)),
            "trade_score": float(latest.get("trade_score", 0.0)),
            "market_regime": latest.get("market_regime", ""),
            "ai_size_multiplier": float(latest.get("ai_size_multiplier", 1.0)),
        })

        

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out[
        (out["trade_score"] >= RANKING_MIN_SCORE)
        & (out["expected_return"] > 0)
    ]

    out = out.sort_values("trade_score", ascending=False)
    return out.head(RANKING_TOP_N_PER_DAY)