"""
tests/test_smoke.py
===================
End-to-end smoke test on synthetic data — no network, no Alpaca needed.

Run:  python -m tests.test_smoke
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("smoke")


def run():
    from synthetic import generate_ohlcv, generate_multi
    from features import build_features, features_clean
    from labeling import label_trades
    from model import SwingModel
    from strategy import generate_entry_signals
    from backtest import run_portfolio_backtest
    from walkforward import run_walkforward
    from paper_trade import PaperEngine, PositionStore

    SEP = "─" * 60
    log.info("[1/8] Synthetic data")
    syms = ["A", "B", "C", "D"]
    raws = generate_multi(syms, start="2018-01-01", end="2024-12-31")
    spy = generate_ohlcv("SPY", start="2018-01-01", end="2024-12-31",
                         annual_drift=0.08, annual_vol=0.15, seed=99)
    assert all(len(r) > 1000 for r in raws.values())
    log.info(f"  generated {len(syms)} symbols + SPY")

    log.info(SEP)
    log.info("[2/8] Feature build")
    featured = {}
    for s, df in raws.items():
        feat = build_features(df, spy)
        feat = generate_entry_signals(feat)
        featured[s] = feat
        n_buy = int(feat["buy_signal"].sum())
        log.info(f"  {s}: {len(feat)} bars, {n_buy} buy signals")
    assert all("atr_pct" in df.columns for df in featured.values())

    log.info(SEP)
    log.info("[3/8] Labeling")
    sample = label_trades(featured["A"])
    assert "trade_return" in sample.columns
    assert "target_meaningful" in sample.columns
    labeled_valid = sample.dropna(subset=["trade_return"])
    pos_rate = float(labeled_valid["target_meaningful"].mean())
    log.info(f"  positive rate (meaningful wins): {pos_rate:.3f}")

    log.info(SEP)
    log.info("[4/8] Model training")
    frames = []
    for s, df in featured.items():
        lab = label_trades(df)
        cln = features_clean(lab).dropna(subset=["target_meaningful"])
        if not cln.empty:
            cln["symbol"] = s
            frames.append(cln)
    train_df = pd.concat(frames).sort_index()
    log.info(f"  training rows: {len(train_df):,}")
    model = SwingModel()
    metrics = model.train(train_df)
    assert metrics["roc_auc"] is None or 0.0 <= metrics["roc_auc"] <= 1.0
    log.info(f"  metrics: auc={metrics['roc_auc']} acc={metrics['accuracy']} "
             f"prec={metrics['precision']} rec={metrics['recall']}")
    log.info(f"  top-decile win rate: {metrics['top_decile_win_rate']}")

    log.info(SEP)
    log.info("[5/8] Portfolio backtest (rules only)")
    res_rules = run_portfolio_backtest(featured, ml_probs=None, use_ml=False,
                                          label="smoke_rules")
    log.info(f"  metrics: {res_rules.metrics}")
    assert "total_return_pct" in res_rules.metrics

    log.info(SEP)
    log.info("[6/8] Portfolio backtest (ML-ranked)")
    probs = {s: model.predict_proba(df) for s, df in featured.items()}
    res_ml = run_portfolio_backtest(featured, ml_probs=probs, use_ml=True,
                                      label="smoke_ml")
    log.info(f"  metrics: {res_ml.metrics}")

    log.info(SEP)
    log.info("[7/8] Walk-forward (limited windows)")
    wf_df, wf_agg = run_walkforward(featured, use_ml=True)
    log.info(f"  windows: {len(wf_df)}")
    log.info(f"  agg: {wf_agg}")

    log.info(SEP)
    log.info("[8/8] Paper engine dry-run (no Alpaca)")
    # Clear any leftover state
    for f in ("logs/open_positions.json",):
        Path(f).unlink(missing_ok=True)
    engine = PaperEngine(dry_run=True)
    summary = engine.run_once(featured, ml_probs=probs, use_ml=True)
    log.info(f"  summary: {summary}")
    assert engine.positions.count() <= 4

    log.info(SEP)
    log.info("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    run()
