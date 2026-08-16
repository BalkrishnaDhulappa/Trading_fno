#!/usr/bin/env python3
"""Build lesson PDFs from course/lessons/*.md"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from pdf_lesson import build_pdf  # noqa: E402

KICKER = {
    "00-how-this-lab-works.md": "00  ·  lab rules",
    "01-candlesticks.md": "01  ·  3×3 candles",
    "02-price-structure.md": "02  ·  3×3 structure",
    "03-rsi-ranges.md": "03  ·  RSI ranges",
    "04-range-shift.md": "04  ·  range shift",
    "05-prd-nrd.md": "05  ·  PRD / NRD",
    "06-gfs.md": "06  ·  GFS",
    "07-bollinger-father-son.md": "07  ·  Bollinger",
    "08-put-together.md": "08  ·  drill",
    "annotation.md": "checklist",
}


def main() -> None:
    out = ROOT / "output" / "pdf"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "lessons").glob("*.md"))
    files.append(ROOT / "checklists" / "annotation.md")
    for path in files:
        pdf = out / f"{path.stem}.pdf"
        build_pdf(path, pdf, KICKER.get(path.name, path.stem))
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
