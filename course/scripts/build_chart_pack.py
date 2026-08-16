#!/usr/bin/env python3
"""Download NSE history and write an offline M/W/D RSI 40/60 chart-pack PDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from charts import render_symbol_charts, write_chart_pack_pdf  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", type=Path, default=ROOT / "config" / "universe.txt")
    p.add_argument("--count", type=int, default=8, help="How many symbols from the list")
    p.add_argument("--symbols", nargs="*", help="Override list, e.g. RELIANCE TCS INFY")
    p.add_argument("--name", default="Chart pack")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "pdf" / "chart-pack.pdf")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.symbols:
        symbols = args.symbols
    else:
        lines = args.universe.read_text(encoding="utf-8").splitlines()
        symbols = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        symbols = symbols[: args.count]

    png_dir = ROOT / "output" / "png"
    collected = {}
    for sym in symbols:
        try:
            print(f"charts {sym} ...")
            collected[sym] = render_symbol_charts(sym, png_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {sym}: {exc}")
    if not collected:
        raise SystemExit("No charts rendered")
    path = write_chart_pack_pdf(collected, args.out, args.name)
    print(f"wrote {path} ({len(collected)} names)")


if __name__ == "__main__":
    main()
