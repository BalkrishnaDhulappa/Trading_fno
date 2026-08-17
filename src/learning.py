"""Reward / penalty learning loop for next-day NIFTY option predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

DEFAULT_LOG = Path("data/processed/prediction_log.csv")
DEFAULT_STATS = Path("data/processed/learning_stats.json")
DEFAULT_ONLINE = Path("models/online_learner.joblib")

REWARD_MOVE_REF = 0.005  # 50 bps reference move for scaling
ONLINE_BLEND = 0.20  # conservative weight of online learner in blended probability


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_prediction_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "as_of" in frame.columns:
        frame["as_of"] = pd.to_datetime(frame["as_of"]).dt.strftime("%Y-%m-%d")
    return frame


def save_prediction_log(frame: pd.DataFrame, path: Path) -> None:
    _ensure_parent(path)
    frame.to_csv(path, index=False)


def append_prediction(
    log_path: Path,
    *,
    as_of: str,
    nifty_close: float,
    predicted_side: str,
    prob_up: float,
    confidence: float,
    feature_row: pd.DataFrame,
    atm_strike: float | None = None,
    suggested_contract: str | None = None,
) -> pd.DataFrame:
    """Log today's prediction (one row per as_of date)."""
    log = load_prediction_log(log_path)
    features = {col: float(feature_row.iloc[0][col]) for col in FEATURE_COLUMNS}
    row = {
        "as_of": as_of,
        "nifty_close": float(nifty_close),
        "predicted_side": predicted_side,
        "predicted_up": int(predicted_side == "CE"),
        "prob_up": float(prob_up),
        "confidence": float(confidence),
        "atm_strike": atm_strike,
        "suggested_contract": suggested_contract,
        "features_json": json.dumps(features),
        "settled": 0,
        "actual_close": np.nan,
        "actual_return": np.nan,
        "actual_side": "",
        "correct": np.nan,
        "reward": np.nan,
        "settled_at": "",
    }
    if not log.empty:
        log = log[log["as_of"] != as_of]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    save_prediction_log(log, log_path)
    return log


def compute_reward(*, correct: bool, confidence: float, actual_return: float) -> float:
    """Positive reward for hits, negative penalty for misses.

    Larger |move| and higher confidence amplify the signal — confident wrong
    calls are penalised harder; confident right calls earn more.
    """
    conf = float(np.clip(confidence, 0.5, 1.0))
    move_scale = 1.0 + (abs(float(actual_return)) / REWARD_MOVE_REF)
    magnitude = conf * move_scale
    return float(magnitude if correct else -magnitude)


def settle_predictions(log: pd.DataFrame, spot: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Settle unsettled rows once the next trading day's close is known."""
    if log.empty:
        return log, []

    spot = spot.copy()
    spot["date"] = pd.to_datetime(spot["date"]).dt.tz_localize(None).dt.normalize()
    spot = spot.sort_values("date").drop_duplicates("date")
    closes = spot.set_index("date")["close"].astype(float)
    dates = list(closes.index)

    events: list[dict[str, Any]] = []
    out = log.copy()
    for idx, row in out.iterrows():
        if int(row.get("settled") or 0) == 1:
            continue
        as_of = pd.Timestamp(row["as_of"]).normalize()
        if as_of not in closes.index:
            continue
        pos = dates.index(as_of) if as_of in dates else None
        if pos is None or pos + 1 >= len(dates):
            continue
        next_day = dates[pos + 1]
        actual_close = float(closes.loc[next_day])
        pred_close = float(row["nifty_close"])
        actual_return = actual_close / pred_close - 1.0
        actual_side = "CE" if actual_return > 0 else "PE"
        correct = actual_side == row["predicted_side"]
        reward = compute_reward(
            correct=correct,
            confidence=float(row["confidence"]),
            actual_return=actual_return,
        )
        out.at[idx, "settled"] = 1
        out.at[idx, "actual_close"] = actual_close
        out.at[idx, "actual_return"] = actual_return
        out.at[idx, "actual_side"] = actual_side
        out.at[idx, "correct"] = int(correct)
        out.at[idx, "reward"] = reward
        out.at[idx, "settled_at"] = datetime.now(timezone.utc).isoformat()

        features = json.loads(row["features_json"])
        events.append(
            {
                "as_of": row["as_of"],
                "next_day": str(next_day.date()),
                "predicted_side": row["predicted_side"],
                "actual_side": actual_side,
                "correct": correct,
                "reward": reward,
                "actual_return": actual_return,
                "features": features,
                "label": int(actual_return > 0),
            }
        )
    return out, events


def experience_sample_weights(
    meta_dates: pd.Series,
    log: pd.DataFrame,
    *,
    base_weight: float = 1.0,
    reward_boost: float = 0.75,
    penalty_boost: float = 2.5,
) -> np.ndarray:
    """Map settled rewards into training sample weights.

    Correct predictions reinforce the day mildly; mistakes up-weight that day
    so the next fit pays more attention to the failure.
    """
    weights = np.full(len(meta_dates), base_weight, dtype=float)
    if log.empty:
        return weights

    settled = log[log["settled"].fillna(0).astype(int) == 1].copy()
    if settled.empty:
        return weights

    settled["as_of"] = pd.to_datetime(settled["as_of"]).dt.normalize()
    by_date = settled.drop_duplicates("as_of", keep="last").set_index("as_of")

    for i, dt in enumerate(pd.to_datetime(meta_dates).dt.normalize()):
        if dt not in by_date.index:
            continue
        row = by_date.loc[dt]
        correct = int(row["correct"]) == 1
        conf = float(row["confidence"])
        if correct:
            weights[i] = base_weight + reward_boost * conf
        else:
            weights[i] = base_weight + penalty_boost * conf
    return weights


def build_online_learner() -> dict[str, Any]:
    return {
        "scaler": StandardScaler(),
        "clf": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-3,
            learning_rate="optimal",
            random_state=42,
            average=True,
        ),
        "fitted": False,
        "n_updates": 0,
    }


def load_online_learner(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return joblib.load(path)


def save_online_learner(model: dict[str, Any], path: Path) -> None:
    _ensure_parent(path)
    joblib.dump(model, path)


def _event_matrix(events: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(event["features"][col]) for col in FEATURE_COLUMNS] for event in events],
        dtype=float,
    )
    y = np.asarray([int(event["label"]) for event in events], dtype=int)
    # Cap weights so penalties reinforce without collapsing the linear model.
    weights = np.asarray(
        [1.0 + min(abs(float(event["reward"])), 2.0) for event in events],
        dtype=float,
    )
    return x, y, weights


def update_online_learner(
    model: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Apply reward-aware online updates for newly settled predictions."""
    if not events:
        return model

    learner = model or build_online_learner()
    scaler: StandardScaler = learner["scaler"]
    clf: SGDClassifier = learner["clf"]
    x, y, sample_weight = _event_matrix(events)

    # Cold start / large bootstrap: full fit is stabler than many tiny partial_fits.
    if (not learner["fitted"]) or len(events) >= 10:
        scaler.fit(x)
        x_scaled = scaler.transform(x)
        clf.fit(x_scaled, y, sample_weight=sample_weight)
        learner["fitted"] = True
        learner["n_updates"] = int(learner.get("n_updates", 0)) + len(events)
        return learner

    for i in range(len(events)):
        xi = x[i : i + 1]
        yi = y[i : i + 1]
        wi = sample_weight[i : i + 1]
        scaler.partial_fit(xi)
        xi_scaled = scaler.transform(xi)
        clf.partial_fit(xi_scaled, yi, classes=np.array([0, 1]), sample_weight=wi)
        learner["n_updates"] = int(learner.get("n_updates", 0)) + 1
    learner["fitted"] = True
    return learner


def blend_probabilities(batch_prob_up: float, online_p: float | None) -> float:
    if online_p is None:
        return batch_prob_up
    online_p = float(np.clip(online_p, 0.05, 0.95))
    # Shrink online influence when it is extremely one-sided (low reliability).
    online_conf = max(online_p, 1.0 - online_p)
    blend = ONLINE_BLEND * (0.5 + (online_conf - 0.5))  # 0.10 .. 0.20
    blend = float(np.clip(blend, 0.05, ONLINE_BLEND))
    return float((1.0 - blend) * batch_prob_up + blend * online_p)


def online_prob_up(model: dict[str, Any] | None, feature_row: pd.DataFrame) -> float | None:
    if model is None or not model.get("fitted"):
        return None
    # Need enough updates before trusting the adaptive head in the blend.
    if int(model.get("n_updates", 0)) < 20:
        return None
    x = np.asarray([feature_row.iloc[0][FEATURE_COLUMNS].astype(float).to_numpy()], dtype=float)
    x_scaled = model["scaler"].transform(x)
    proba = float(model["clf"].predict_proba(x_scaled)[0, 1])
    return float(np.clip(proba, 0.05, 0.95))

def bootstrap_shadow_predictions(
    spot: pd.DataFrame,
    model_bundle: dict[str, Any],
    *,
    lookback_days: int = 30,
    oc_daily: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Create settled reward events from recent history using the current batch model."""
    from .features import build_spot_features, merge_oc_features
    from .model import predict_next_day

    feats = merge_oc_features(build_spot_features(spot), oc_daily)
    usable = feats.dropna(subset=FEATURE_COLUMNS + ["next_up", "next_return", "close"]).copy()
    if usable.empty:
        return []
    usable = usable.iloc[:-1]
    usable = usable.tail(lookback_days)

    events: list[dict[str, Any]] = []
    for _, row in usable.iterrows():
        feature_row = pd.DataFrame([row])[FEATURE_COLUMNS]
        prediction = predict_next_day(model_bundle, feature_row)
        actual_side = "CE" if int(row["next_up"]) == 1 else "PE"
        correct = prediction["predicted_side"] == actual_side
        actual_return = float(row["next_return"])
        reward = compute_reward(
            correct=correct,
            confidence=float(prediction["confidence"]),
            actual_return=actual_return,
        )
        features = {col: float(row[col]) for col in FEATURE_COLUMNS}
        events.append(
            {
                "as_of": str(pd.Timestamp(row["date"]).date()),
                "next_day": "",
                "predicted_side": prediction["predicted_side"],
                "actual_side": actual_side,
                "correct": correct,
                "reward": reward,
                "actual_return": actual_return,
                "features": features,
                "label": int(row["next_up"]),
                "nifty_close": float(row["close"]),
                "prob_up": float(prediction["prob_up"]),
                "confidence": float(prediction["confidence"]),
            }
        )
    return events


def merge_bootstrap_into_log(log: pd.DataFrame, events: list[dict[str, Any]]) -> pd.DataFrame:
    """Write bootstrap settlements into the prediction log (no duplicates by as_of)."""
    rows = []
    for event in events:
        rows.append(
            {
                "as_of": event["as_of"],
                "nifty_close": event["nifty_close"],
                "predicted_side": event["predicted_side"],
                "predicted_up": int(event["predicted_side"] == "CE"),
                "prob_up": event["prob_up"],
                "confidence": event["confidence"],
                "atm_strike": np.nan,
                "suggested_contract": "",
                "features_json": json.dumps(event["features"]),
                "settled": 1,
                "actual_close": np.nan,
                "actual_return": event["actual_return"],
                "actual_side": event["actual_side"],
                "correct": int(event["correct"]),
                "reward": event["reward"],
                "settled_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    boot = pd.DataFrame(rows)
    if log.empty:
        return boot
    existing = set(log["as_of"].astype(str))
    boot = boot[~boot["as_of"].astype(str).isin(existing)]
    return pd.concat([log, boot], ignore_index=True)


def summarize_learning(log: pd.DataFrame) -> dict[str, Any]:
    settled = log[log["settled"].fillna(0).astype(int) == 1] if not log.empty else log
    if settled is None or settled.empty:
        return {
            "settled_count": 0,
            "pending_count": int(len(log)) if not log.empty else 0,
            "hit_rate": None,
            "cumulative_reward": 0.0,
            "avg_reward": None,
            "correct": 0,
            "wrong": 0,
        }
    correct = int(settled["correct"].fillna(0).astype(int).sum())
    total = len(settled)
    return {
        "settled_count": total,
        "pending_count": int((log["settled"].fillna(0).astype(int) == 0).sum()),
        "hit_rate": correct / total if total else None,
        "cumulative_reward": float(settled["reward"].sum()),
        "avg_reward": float(settled["reward"].mean()),
        "correct": correct,
        "wrong": total - correct,
    }


def save_learning_stats(stats: dict[str, Any], path: Path) -> None:
    _ensure_parent(path)
    stats = {
        **stats,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
