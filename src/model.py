"""Train and serve next-day NIFTY CE/PE direction model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass
class TrainResult:
    accuracy: float
    roc_auc: float
    brier: float
    folds: int
    n_train_rows: int
    feature_columns: list[str]


def build_model() -> Pipeline:
    # Keep the base learner relatively regularized — index next-day direction is noisy.
    base = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=0.25,
        random_state=42,
    )
    # Sigmoid calibration is stabler than isotonic on smaller financial folds.
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def _threshold_stats(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.55,
) -> dict[str, float]:
    conf = np.maximum(proba, 1.0 - proba)
    mask = conf >= threshold
    if mask.sum() == 0:
        return {"coverage": 0.0, "accuracy": float("nan"), "avg_confidence": float("nan")}
    pred = (proba[mask] >= 0.5).astype(int)
    return {
        "coverage": float(mask.mean()),
        "accuracy": float(accuracy_score(y_true[mask], pred)),
        "avg_confidence": float(conf[mask].mean()),
    }


def time_series_cv_score(
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    confidence_threshold: float = 0.55,
) -> TrainResult:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, aucs, briers = [], [], []
    thr_accs, thr_covs = [], []
    for train_idx, test_idx in tscv.split(x):
        model = build_model()
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y_test, pred))
        aucs.append(roc_auc_score(y_test, proba))
        briers.append(brier_score_loss(y_test, proba))
        thr = _threshold_stats(y_test.to_numpy(), proba, confidence_threshold)
        if thr["coverage"] > 0:
            thr_accs.append(thr["accuracy"])
            thr_covs.append(thr["coverage"])

    if not accs:
        raise RuntimeError("Time-series CV produced no valid folds")

    result = TrainResult(
        accuracy=float(np.mean(accs)),
        roc_auc=float(np.mean(aucs)),
        brier=float(np.mean(briers)),
        folds=len(accs),
        n_train_rows=len(x),
        feature_columns=list(FEATURE_COLUMNS),
    )
    # Attach thresholded stats for strategy filtering (stored via metrics.json extras).
    result_dict = asdict(result)
    result_dict["confidence_threshold"] = confidence_threshold
    result_dict["threshold_accuracy"] = float(np.mean(thr_accs)) if thr_accs else None
    result_dict["threshold_coverage"] = float(np.mean(thr_covs)) if thr_covs else None
    result._extra = result_dict  # type: ignore[attr-defined]
    return result


def train_final_model(x: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = build_model()
    model.fit(x, y)
    return model


def save_artifact(
    model: Pipeline,
    metrics: TrainResult,
    model_dir: Path,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "nifty_next_day_option.joblib"
    metrics_path = model_dir / "metrics.json"
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, model_path)
    payload = getattr(metrics, "_extra", asdict(metrics))
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return model_path


def load_artifact(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_next_day(model_bundle: dict[str, Any], feature_row: pd.DataFrame) -> dict[str, Any]:
    model: Pipeline = model_bundle["model"]
    cols = model_bundle["features"]
    x = feature_row[cols]
    proba_up = float(model.predict_proba(x)[0, 1])
    side = "CE" if proba_up >= 0.5 else "PE"
    confidence = proba_up if side == "CE" else 1.0 - proba_up
    return {
        "predicted_side": side,
        "prob_up": proba_up,
        "prob_down": 1.0 - proba_up,
        "confidence": confidence,
        "interpretation": (
            "Model leans bullish → prefer ATM/near-ATM CE"
            if side == "CE"
            else "Model leans bearish → prefer ATM/near-ATM PE"
        ),
    }
