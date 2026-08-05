#!/usr/bin/env python3
"""Predict next-day NIFTY option side (CE vs PE) and log it for learning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLUMNS, build_spot_features, merge_oc_features
from src.learning import (
    DEFAULT_LOG,
    DEFAULT_ONLINE,
    append_prediction,
    blend_probabilities,
    load_online_learner,
    online_prob_up,
    summarize_learning,
    load_prediction_log,
)
from src.model import load_artifact, predict_next_day
from src.nifty_data import download_nifty_history
from src.nse_option_chain import fetch_option_chain, summarize_option_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "nifty_next_day_option.joblib",
    )
    parser.add_argument(
        "--online-model",
        type=Path,
        default=ROOT / DEFAULT_ONLINE,
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / DEFAULT_LOG,
    )
    parser.add_argument("--period", default="1y", help="History window for features")
    parser.add_argument(
        "--skip-oc",
        action="store_true",
        help="Do not fetch live NSE option chain features",
    )
    parser.add_argument("--expiry", default=None, help="NSE expiry, e.g. 11-Aug-2026")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Predict without writing to the learning log",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run scripts/train.py first.")

    spot = download_nifty_history(period=args.period)
    feats = build_spot_features(spot)

    oc_row = None
    oc_summary = None
    payload = None
    if not args.skip_oc:
        try:
            payload = fetch_option_chain(symbol="NIFTY", expiry=args.expiry)
            oc_summary = summarize_option_chain(payload)
            oc_row = pd.DataFrame(
                [
                    {
                        "date": feats["date"].iloc[-1],
                        **{
                            k: oc_summary[k]
                            for k in oc_summary
                            if k
                            in {
                                "pcr_oi",
                                "pcr_vol",
                                "oi_imbalance",
                                "atm_iv_skew",
                                "max_ce_oi_distance_pct",
                                "max_pe_oi_distance_pct",
                                "atm_premium_ratio",
                            }
                        },
                    }
                ]
            )
            print(
                f"Live OC — spot={oc_summary['spot']:.2f}  "
                f"ATM={oc_summary['atm_strike']:.0f}  "
                f"PCR={oc_summary['pcr_oi']:.3f}  "
                f"expiry={payload['_meta']['expiry']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: live OC unavailable ({exc}); using spot features only.")

    feats = merge_oc_features(feats, oc_row)
    latest = feats.dropna(subset=FEATURE_COLUMNS).iloc[[-1]].copy()
    bundle = load_artifact(args.model)

    online = load_online_learner(args.online_model)
    online_p = online_prob_up(online, latest)
    batch_bundle_pred = predict_next_day(bundle, latest)
    blended = blend_probabilities(batch_bundle_pred["batch_prob_up"], online_p)
    prediction = predict_next_day(bundle, latest, blended_prob_up=blended)
    if online_p is not None:
        prediction["online_prob_up"] = online_p
        prediction["blended"] = True
    else:
        prediction["blended"] = False

    as_of = str(latest["date"].iloc[0].date())
    result = {
        "as_of": as_of,
        "nifty_close": float(latest["close"].iloc[0]),
        **prediction,
    }
    suggested = None
    atm_strike = None
    if oc_summary and payload:
        atm_strike = oc_summary["atm_strike"]
        suggested = (
            f"NIFTY {payload['_meta']['expiry']} "
            f"{int(oc_summary['atm_strike'])} {prediction['predicted_side']}"
        )
        result["atm_strike"] = atm_strike
        result["suggested_contract"] = suggested
        result["oc_features"] = {
            k: oc_summary[k]
            for k in (
                "pcr_oi",
                "pcr_vol",
                "oi_imbalance",
                "atm_iv_skew",
                "atm_ce_ltp",
                "atm_pe_ltp",
            )
        }

    if not args.no_log:
        append_prediction(
            args.log,
            as_of=as_of,
            nifty_close=float(latest["close"].iloc[0]),
            predicted_side=prediction["predicted_side"],
            prob_up=prediction["prob_up"],
            confidence=prediction["confidence"],
            feature_row=latest,
            atm_strike=atm_strike,
            suggested_contract=suggested,
        )
        stats = summarize_learning(load_prediction_log(args.log))
        result["learning"] = stats
        result["logged"] = True
        print(f"Logged prediction for {as_of} → {args.log}")
    else:
        result["logged"] = False

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
