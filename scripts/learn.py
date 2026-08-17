#!/usr/bin/env python3
"""Settle predictions with reward/penalty and update the learning models.

Flow:
1. Load prediction log (from scripts/predict.py)
2. Settle any rows whose next trading day close is now known
3. Reward correct calls / penalise wrong calls
4. Online-update the adaptive learner on each settlement
5. Retrain the batch model with experience sample weights

Use --bootstrap N to warm-start learning from the last N labeled days.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import load_oc_feature_history, make_training_frame
from src.learning import (
    DEFAULT_LOG,
    DEFAULT_ONLINE,
    DEFAULT_STATS,
    bootstrap_shadow_predictions,
    experience_sample_weights,
    load_online_learner,
    load_prediction_log,
    merge_bootstrap_into_log,
    save_learning_stats,
    save_online_learner,
    save_prediction_log,
    settle_predictions,
    summarize_learning,
    update_online_learner,
)
from src.model import load_artifact, save_artifact, time_series_cv_score, train_final_model
from src.nifty_data import download_nifty_history, save_spot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--log", type=Path, default=ROOT / DEFAULT_LOG)
    parser.add_argument(
        "--oc-features",
        type=Path,
        default=ROOT / "data" / "processed" / "oc_features.csv",
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--online-model", type=Path, default=ROOT / DEFAULT_ONLINE)
    parser.add_argument("--stats", type=Path, default=ROOT / DEFAULT_STATS)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Warm-start from last N historical days (shadow predictions + settle)",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Only settle + online-update; do not rebuild batch model",
    )
    parser.add_argument("--splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Downloading NIFTY history ({args.period})...")
    spot = download_nifty_history(period=args.period)
    save_spot_history(spot, ROOT / "data" / "raw" / "nifty_spot.csv")

    model_path = args.model_dir / "nifty_next_day_option.joblib"
    if not model_path.exists():
        raise SystemExit(f"Batch model missing: {model_path}. Run scripts/train.py first.")
    bundle = load_artifact(model_path)

    log = load_prediction_log(args.log)
    oc = load_oc_feature_history(args.oc_features)

    bootstrap_events: list = []
    if args.bootstrap > 0:
        print(f"Bootstrapping reward loop from last {args.bootstrap} days...")
        bootstrap_events = bootstrap_shadow_predictions(
            spot,
            bundle,
            lookback_days=args.bootstrap,
            oc_daily=oc if not oc.empty else None,
        )
        log = merge_bootstrap_into_log(log, bootstrap_events)
        save_prediction_log(log, args.log)
        correct = sum(1 for e in bootstrap_events if e["correct"])
        total_reward = sum(e["reward"] for e in bootstrap_events)
        print(
            f"Bootstrap settled={len(bootstrap_events)}  "
            f"hits={correct}  misses={len(bootstrap_events) - correct}  "
            f"reward_sum={total_reward:+.3f}"
        )

    if log.empty and not bootstrap_events:
        raise SystemExit(
            f"No predictions in {args.log}. Run scripts/predict.py first, "
            "or pass --bootstrap 30 to warm-start."
        )

    log, live_events = settle_predictions(log, spot)
    save_prediction_log(log, args.log)

    if not live_events:
        print("No new live settlements (next-day close not available yet for pending rows).")
    else:
        print(f"Settled {len(live_events)} live prediction(s):")
        for event in live_events:
            tag = "REWARD" if event["correct"] else "PENALTY"
            print(
                f"  [{tag}] as_of={event['as_of']} → {event['next_day']}  "
                f"pred={event['predicted_side']} actual={event['actual_side']}  "
                f"reward={event['reward']:+.3f}  ret={event['actual_return']:+.3%}"
            )

    events = bootstrap_events + live_events
    online = load_online_learner(args.online_model)
    # If bootstrapping onto an existing online model, only apply brand-new bootstrap
    # rows once — merge_bootstrap_into_log already skipped duplicates in the log,
    # and bootstrap_events here are only the newly generated ones.
    online = update_online_learner(online, events)
    if online is not None:
        save_online_learner(online, args.online_model)
        print(f"Online learner updated → {args.online_model}")

    stats = summarize_learning(log)
    save_learning_stats(stats, args.stats)
    hit = stats["hit_rate"]
    hit_s = f"{hit:.3f}" if hit is not None else "n/a"
    print(
        "Learning stats — "
        f"settled={stats['settled_count']}  pending={stats['pending_count']}  "
        f"hit_rate={hit_s}  "
        f"cumulative_reward={stats['cumulative_reward']:+.3f}"
    )

    if args.skip_retrain:
        print("Skipping batch retrain (--skip-retrain).")
        print(json.dumps({"events": len(events), "stats": stats}, indent=2))
        return

    x, y, meta = make_training_frame(spot, oc if not oc.empty else None)
    weights = experience_sample_weights(meta["date"], log)
    print(
        f"Retraining batch model on {len(x)} rows with experience weights "
        f"(mean={weights.mean():.3f}, max={weights.max():.3f})."
    )

    metrics = time_series_cv_score(x, y, n_splits=args.splits, confidence_threshold=0.55)
    extra = getattr(metrics, "_extra", {})
    extra["learning"] = stats
    metrics._extra = extra  # type: ignore[attr-defined]

    model = train_final_model(x, y, sample_weight=weights)
    path = save_artifact(model, metrics, args.model_dir)
    print(f"Saved batch model → {path}")
    print(f"Saved learning stats → {args.stats}")


if __name__ == "__main__":
    main()
