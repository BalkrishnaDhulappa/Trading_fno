#!/usr/bin/env python3
"""Train next-day NIFTY CE/PE direction model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import load_oc_feature_history, make_training_frame
from src.model import save_artifact, time_series_cv_score, train_final_model
from src.nifty_data import download_nifty_history, save_spot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="5y", help="yfinance history period")
    parser.add_argument(
        "--oc-features",
        type=Path,
        default=ROOT / "data" / "processed" / "oc_features.csv",
        help="Optional daily option-chain feature history CSV",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory to write model artifact",
    )
    parser.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit folds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Downloading NIFTY history ({args.period})...")
    spot = download_nifty_history(period=args.period)
    save_spot_history(spot, ROOT / "data" / "raw" / "nifty_spot.csv")
    print(f"Spot rows: {len(spot)}  ({spot['date'].min().date()} → {spot['date'].max().date()})")

    oc = load_oc_feature_history(args.oc_features)
    if oc.empty:
        print("No OC feature history found — training on spot/technical features only.")
    else:
        print(f"Loaded {len(oc)} OC feature rows from {args.oc_features}")

    x, y, meta = make_training_frame(spot, oc if not oc.empty else None)
    print(f"Training rows: {len(x)}  |  up-days: {int(y.sum())}  down-days: {int((1 - y).sum())}")

    metrics = time_series_cv_score(x, y, n_splits=args.splits, confidence_threshold=0.55)
    extra = getattr(metrics, "_extra", {})
    print(
        "CV metrics — "
        f"accuracy={metrics.accuracy:.3f}  "
        f"roc_auc={metrics.roc_auc:.3f}  "
        f"brier={metrics.brier:.3f}  "
        f"folds={metrics.folds}"
    )
    if extra.get("threshold_accuracy") is not None:
        print(
            "When confidence≥0.55 — "
            f"accuracy={extra['threshold_accuracy']:.3f}  "
            f"coverage={extra['threshold_coverage']:.3f}"
        )

    model = train_final_model(x, y)
    path = save_artifact(model, metrics, args.model_dir)
    print(f"Saved model → {path}")
    print(f"Saved metrics → {args.model_dir / 'metrics.json'}")
    print(
        "Label meaning: next_up=1 → next day NIFTY closes higher → recommend CE; "
        "else recommend PE."
    )


if __name__ == "__main__":
    main()
