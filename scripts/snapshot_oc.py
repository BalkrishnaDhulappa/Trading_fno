#!/usr/bin/env python3
"""Snapshot today's NIFTY option-chain features for future model training."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nse_option_chain import (
    fetch_option_chain,
    option_chain_to_frame,
    summarize_option_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expiry", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "oc_features.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "oc_snapshots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = fetch_option_chain(symbol="NIFTY", expiry=args.expiry)
    summary = summarize_option_chain(payload)
    as_of = datetime.now().date().isoformat()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    legs = option_chain_to_frame(payload)
    legs.to_csv(args.raw_dir / f"nifty_oc_{as_of}.csv", index=False)

    row = {"date": as_of, **summary, "expiry": payload["_meta"]["expiry"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        hist = pd.read_csv(args.out)
        hist = hist[hist["date"].astype(str) != as_of]
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(args.out, index=False)
    print(f"Saved OC legs → {args.raw_dir / f'nifty_oc_{as_of}.csv'}")
    print(f"Updated features → {args.out}")
    print(
        f"date={as_of} spot={summary['spot']:.2f} "
        f"ATM={summary['atm_strike']:.0f} PCR={summary['pcr_oi']:.3f}"
    )


if __name__ == "__main__":
    main()
