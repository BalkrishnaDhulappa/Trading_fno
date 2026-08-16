# 3×3 chart lab (offline PDFs)

Personal study lab for **Vishal B. Malkan–style 3×3** chart work.

**Not an iOS app** (later, maybe). **Not live.** Files you copy onto iPhone/iPad storage.

## What you get

- Lesson PDFs in **3×3 order** (candles → structure → RSI 40/60 → shift → PRD/NRD divergence → GFS → Bollinger father–son → drill)
- A **chart pack** PDF: each stock on **monthly / weekly / daily** candles with **RSI 14** and lines at **40 / 60**
- A blank **annotation** sheet after every name

## Put it on the iPad (no network after copy)

1. On a computer, build PDFs (below) or take the files in `course/output/pdf/`
2. USB / Finder / AirDrop into **Files**, or import into **Goodnotes / Notability / Apple Books**
3. Airplane mode. Markup with Pencil.

## Build on a computer (needs internet *once*, to download history)

```bash
pip install -r requirements.txt
python course/scripts/build_lessons.py
python course/scripts/build_chart_pack.py --count 8 --name "Pack A"
```

Regenerate a pack whenever you want a fresh slice of history. The iPad still only needs the new PDF copied over.

## Lesson order

| PDF | Topic |
|-----|--------|
| 00 | How this lab works |
| 01 | Candlesticks |
| 02 | Price structure |
| 03 | RSI ranges (bull / neutral / bear) |
| 04 | Range shift |
| 05 | PRD / NRD = divergence |
| 06 | GFS |
| 07 | Bollinger dual band / father–son |
| 08 | Combine + daily drill |
| annotation | Checklist |

If a 3×3 YouTube episode and a PDF disagree, **the video wins**.

## Universe

Edit `course/config/universe.txt` (NSE Yahoo symbols). Default list is large-cap cash names.
