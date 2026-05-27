"""
model.py
========
A RandomForest classifier that predicts P(meaningful win) for each bar.

Design notes
------------
- Target: target_meaningful (only counts moves >= ML_TARGET_MIN_WIN_PCT as wins).
- The model is a *ranker*, not a forcer. If AUC < ML_HARD_FLOOR + epsilon, the
  caller is warned and should fall back to rules-only.
- Reports: accuracy, precision, recall, ROC AUC, positive rate, and decile lift
  (top-decile win rate / top-decile mean return).
- Sequential 75/25 split — never random — so future leakage is impossible.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLS, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, ML_N_ESTIMATORS,
    ML_RANDOM_STATE, ML_TRAIN_RATIO, MODEL_METRICS_JSON, MODEL_PATH, SCALER_PATH,
)

log = logging.getLogger(__name__)


class SwingModel:
    """Light wrapper around RandomForestClassifier."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF,
            random_state=ML_RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.feature_cols = list(FEATURE_COLS)
        self.trained = False
        self.metrics: dict = {}

    # ── Training ────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        if "target_meaningful" not in df.columns:
            raise ValueError("df must contain target_meaningful (run labeling.label_trades)")
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        data = df.dropna(subset=["target_meaningful"] + self.feature_cols).copy()
        if len(data) < 500:
            raise ValueError(f"Need >= 500 training rows, got {len(data)}")

        # Sequential split — chronological
        data = data.sort_index()
        cut = int(len(data) * ML_TRAIN_RATIO)
        train = data.iloc[:cut]
        test = data.iloc[cut:]

        X_train = train[self.feature_cols].to_numpy()
        y_train = train["target_meaningful"].astype(int).to_numpy()
        X_test = test[self.feature_cols].to_numpy()
        y_test = test["target_meaningful"].astype(int).to_numpy()

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        self.model.fit(X_train_s, y_train)

        proba = self.model.predict_proba(X_test_s)[:, 1]
        preds = (proba >= 0.5).astype(int)

        # Decile lift
        if len(test) >= 100 and "trade_return" in test.columns:
            df_eval = test.copy()
            df_eval["__proba"] = proba
            df_eval["__decile"] = pd.qcut(
                df_eval["__proba"].rank(method="first"), q=10, labels=False,
            )
            top = df_eval[df_eval["__decile"] == 9]
            top_win_rate = float((top["trade_return"] > 0).mean()) if len(top) else float("nan")
            top_mean_ret = float(top["trade_return"].mean()) if len(top) else float("nan")
            base_win_rate = float((df_eval["trade_return"] > 0).mean())
        else:
            top_win_rate = float("nan")
            top_mean_ret = float("nan")
            base_win_rate = float((y_test).mean()) if len(y_test) else float("nan")

        self.metrics = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "positive_rate_train": round(float(y_train.mean()), 4),
            "positive_rate_test": round(float(y_test.mean()), 4),
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4) if len(set(y_test)) > 1 else None,
            "top_decile_win_rate": None if np.isnan(top_win_rate) else round(top_win_rate, 4),
            "top_decile_mean_return": None if np.isnan(top_mean_ret) else round(top_mean_ret, 4),
            "baseline_win_rate": round(base_win_rate, 4),
        }
        self.trained = True

        # Honest quality warning
        if self.metrics["roc_auc"] is None:
            log.warning("ROC AUC not computable (single-class test set).")
        elif self.metrics["roc_auc"] < 0.55:
            log.warning(
                f"Model AUC = {self.metrics['roc_auc']:.3f} is weak. "
                f"Use ML as a soft tiebreaker only; do not rely on it."
            )

        log.info(f"Model metrics: {self.metrics}")
        self._log_feature_importance()
        return self.metrics

    # ── Inference ───────────────────────────────────────────────────────────
    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if not self.trained:
            raise RuntimeError("Model not trained")
        out = pd.Series(np.nan, index=df.index, name="ml_proba", dtype=float)
        mask = df[self.feature_cols].notna().all(axis=1)
        if mask.sum() == 0:
            return out
        X = self.scaler.transform(df.loc[mask, self.feature_cols].to_numpy())
        out.loc[mask] = self.model.predict_proba(X)[:, 1]
        return out.clip(0.0, 1.0)

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": self.model, "feature_cols": self.feature_cols}, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(MODEL_METRICS_JSON, "w") as f:
            json.dump(self.metrics, f, indent=2)
        log.info(f"Model saved to {model_path}")

    def load(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.feature_cols = payload.get("feature_cols", list(FEATURE_COLS))
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        try:
            with open(MODEL_METRICS_JSON) as f:
                self.metrics = json.load(f)
        except Exception:
            self.metrics = {}
        self.trained = True
        log.info(f"Model loaded from {model_path}")

    def _log_feature_importance(self):
        if not hasattr(self.model, "feature_importances_"):
            return
        pairs = sorted(
            zip(self.feature_cols, self.model.feature_importances_),
            key=lambda x: -x[1],
        )
        log.info("Top feature importances:")
        for name, val in pairs[:10]:
            log.info(f"  {name:<22} {val:.4f}")

    @property
    def is_reliable(self) -> bool:
        """True if the model is good enough to trust as a ranker."""
        from config import ML_HARD_FLOOR
        auc = self.metrics.get("roc_auc")
        return bool(auc and auc >= ML_HARD_FLOOR + 0.03)
