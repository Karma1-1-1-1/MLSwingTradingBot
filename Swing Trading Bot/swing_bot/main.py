"""
main.py
=======
Command-line entry point.

Usage:
  python main.py backtest                 # run portfolio backtest (rules + ML)
  python main.py train                    # train and save ML model
  python main.py walkforward              # rolling WFO + pass/fail gate
  python main.py paper --dry-run          # one paper iteration, no orders
  python main.py paper                    # one paper iteration with Alpaca paper
  python main.py status                   # show open positions
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from config import (
    BACKTEST_SUMMARY, DRY_RUN_DEFAULT, EQUITY_CSV, LOG_DIR, SYMBOLS, WFO_CSV,
)


def _setup_logging():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(LOG_DIR) / "bot.log"),
        ],
    )


def _load_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    from data import fetch_ohlcv
    from features import build_features
    from strategy import generate_entry_signals

    spy = fetch_ohlcv("SPY")
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            raw = fetch_ohlcv(sym)
            feat = build_features(raw, spy)
            feat = generate_entry_signals(feat)
            out[sym] = feat
        except Exception as exc:
            logging.error(f"[{sym}] feature build failed: {exc}")
    return out


def cmd_train(args):
    from features import features_clean
    from labeling import label_trades
    from model import SwingModel

    feature_dfs = _load_universe(args.symbols)
    frames = []
    for sym, df in feature_dfs.items():
        labeled = label_trades(df)
        cleaned = features_clean(labeled)
        cleaned = cleaned.dropna(subset=["target_meaningful"])
        if not cleaned.empty:
            cleaned["symbol"] = sym
            frames.append(cleaned)
    if not frames:
        logging.error("No training rows.")
        return
    train_df = pd.concat(frames).sort_index()
    logging.info(f"Training rows: {len(train_df):,}")
    model = SwingModel()
    model.train(train_df)
    model.save()
    if not model.is_reliable:
        logging.warning("Model AUC is below the reliability floor. "
                        "Use --no-ml for the safest configuration.")


def cmd_backtest(args):
    from backtest import run_portfolio_backtest
    from model import SwingModel

    feature_dfs = _load_universe(args.symbols)

    # Rules-only
    rules_result = run_portfolio_backtest(feature_dfs, ml_probs=None,
                                            use_ml=False, label="rules_only")

    # ML-ranked (load saved model if present)
    ml_result = None
    try:
        model = SwingModel()
        model.load()
        if model.is_reliable:
            ml_probs = {s: model.predict_proba(df) for s, df in feature_dfs.items()}
            ml_result = run_portfolio_backtest(feature_dfs, ml_probs=ml_probs,
                                                 use_ml=True, label="ml_ranked")
        else:
            logging.warning("Loaded model is not reliable — skipping ML-ranked backtest")
    except FileNotFoundError:
        logging.info("No saved model. Run 'python main.py train' first to enable ML-ranked.")

    summary = {
        "rules_only": rules_result.metrics,
        "ml_ranked": ml_result.metrics if ml_result else None,
    }
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    with open(BACKTEST_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save curves
    rules_result.equity_curve.to_csv(EQUITY_CSV)
    if not rules_result.trades.empty:
        rules_result.trades.to_csv(f"{LOG_DIR}/trades_backtest_rules.csv", index=False)
    if not rules_result.decisions.empty:
        rules_result.decisions.to_csv(f"{LOG_DIR}/decisions_backtest_rules.csv", index=False)
    if ml_result is not None:
        if not ml_result.trades.empty:
            ml_result.trades.to_csv(f"{LOG_DIR}/trades_backtest_ml.csv", index=False)
        if not ml_result.decisions.empty:
            ml_result.decisions.to_csv(f"{LOG_DIR}/decisions_backtest_ml.csv", index=False)

    _print_backtest_summary(summary)


def cmd_walkforward(args):
    from walkforward import run_walkforward

    feature_dfs = _load_universe(args.symbols)
    df, agg = run_walkforward(feature_dfs, use_ml=not args.no_ml)
    if df.empty:
        logging.error(f"Walk-forward produced no windows: {agg}")
        return
    df.to_csv(WFO_CSV, index=False)
    with open(f"{LOG_DIR}/walkforward_summary.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)
    _print_wfo(agg)


def cmd_paper(args):
    from paper_trade import run_paper_once

    summary = run_paper_once(symbols=args.symbols, dry_run=args.dry_run,
                              use_ml=not args.no_ml)
    print(json.dumps(summary, indent=2, default=str))


def cmd_status(args):
    from paper_trade import PositionStore
    store = PositionStore()
    print(json.dumps(store.all(), indent=2, default=str))


def _print_backtest_summary(summary: dict):
    print("\n" + "═" * 72)
    print("BACKTEST SUMMARY")
    print("═" * 72)
    for variant, m in summary.items():
        if m is None:
            print(f"{variant:<14} : skipped")
            continue
        print(f"\n{variant.upper()}")
        for k, v in m.items():
            print(f"  {k:<28} {v}")
    print("═" * 72 + "\n")


def _print_wfo(agg: dict):
    print("\n" + "═" * 72)
    print("WALK-FORWARD SUMMARY")
    print("═" * 72)
    for k, v in agg.items():
        if k == "checks":
            print("  Pass/fail checks:")
            for ck, cv in v.items():
                mark = "PASS" if cv else "FAIL"
                print(f"    [{mark}] {ck}")
        else:
            print(f"  {k:<32} {v}")
    overall = "READY FOR PAPER" if agg.get("pass") else "NOT READY"
    print(f"\n  >>> Verdict: {overall}")
    print("═" * 72 + "\n")


def main():
    p = argparse.ArgumentParser(description="Swing Bot")
    p.add_argument("--symbols", nargs="+", default=SYMBOLS)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("backtest")
    sub.add_parser("train")

    p_wf = sub.add_parser("walkforward")
    p_wf.add_argument("--no-ml", action="store_true")

    p_pp = sub.add_parser("paper")
    p_pp.add_argument("--dry-run", action="store_true", default=DRY_RUN_DEFAULT)
    p_pp.add_argument("--live", action="store_true", help="Disable dry-run (still paper account)")
    p_pp.add_argument("--no-ml", action="store_true")

    sub.add_parser("status")

    args = p.parse_args()
    _setup_logging()

    if args.cmd == "paper" and args.live:
        args.dry_run = False

    {
        "backtest": cmd_backtest,
        "train": cmd_train,
        "walkforward": cmd_walkforward,
        "paper": cmd_paper,
        "status": cmd_status,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
