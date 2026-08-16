"""Build monthly / weekly / daily candle + RSI(40/60) chart images and a pack PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

TIMEFRAMES = {
    "monthly": {"interval": "1mo", "period": "10y", "label": "MONTHLY  ·  grandfather"},
    "weekly": {"interval": "1wk", "period": "5y", "label": "WEEKLY  ·  father"},
    "daily": {"interval": "1d", "period": "1y", "label": "DAILY  ·  son"},
}


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def load_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    ticker = symbol if "." in symbol or symbol.startswith("^") else f"{symbol}.NS"
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"No data for {ticker} {interval}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].title() for c in raw.columns]
    else:
        raw.columns = [str(c).title() for c in raw.columns]
    need = ["Open", "High", "Low", "Close"]
    for col in need:
        if col not in raw.columns:
            raise RuntimeError(f"{ticker} missing {col}")
    frame = raw[need].dropna()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def plot_candle_rsi(frame: pd.DataFrame, title: str, out_path: Path) -> Path:
    data = frame.copy()
    data["RSI"] = rsi(data["Close"])
    data = data.dropna()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mc = mpf.make_marketcolors(up="#1F6F8B", down="#C45C26", inherit=True)
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor="#F7F5F0",
        figcolor="#F7F5F0",
        gridcolor="#E4E0D8",
        gridstyle="-",
        y_on_right=True,
        rc={
            "font.family": "DejaVu Sans",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        },
    )
    rsi_plot = mpf.make_addplot(data["RSI"], panel=1, color="#0F2744", width=1.1, ylabel="RSI")
    line40 = mpf.make_addplot([40] * len(data), panel=1, color="#C45C26", width=0.8, linestyle="--")
    line60 = mpf.make_addplot([60] * len(data), panel=1, color="#1F6F8B", width=0.8, linestyle="--")

    fig, _axes = mpf.plot(
        data,
        type="candle",
        style=style,
        addplot=[rsi_plot, line40, line60],
        volume=False,
        panel_ratios=(3, 1.15),
        figsize=(11.2, 6.4),
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=10_000,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", color="#0F2744", y=0.98)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_symbol_charts(symbol: str, png_dir: Path) -> dict[str, Path]:
    paths = {}
    for key, spec in TIMEFRAMES.items():
        frame = load_ohlc(symbol, spec["interval"], spec["period"])
        path = png_dir / f"{symbol}_{key}.png"
        plot_candle_rsi(frame, f"{symbol}   {spec['label']}", path)
        paths[key] = path
    return paths


def _draw_image_page(c: canvas.Canvas, image_path: Path, heading: str, footer: str) -> None:
    width, height = landscape(A4)
    c.setFillColorRGB(0.97, 0.96, 0.94)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColorRGB(0.06, 0.15, 0.27)
    c.rect(0, height - 14 * mm, width, 14 * mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Times-Bold", 13)
    c.drawString(12 * mm, height - 9 * mm, heading)
    img = PILImage.open(image_path)
    iw, ih = img.size
    max_w, max_h = width - 16 * mm, height - 32 * mm
    scale = min(max_w / iw, max_h / ih)
    dw, dh = iw * scale, ih * scale
    x = (width - dw) / 2
    y = 14 * mm + (max_h - dh) / 2
    c.drawImage(ImageReader(image_path), x, y, width=dw, height=dh, preserveAspectRatio=True, mask="auto")
    c.setFillColorRGB(0.36, 0.4, 0.44)
    c.setFont("Times-Italic", 8)
    c.drawString(12 * mm, 6 * mm, footer)
    c.showPage()


def write_chart_pack_pdf(
    symbol_paths: dict[str, dict[str, Path]],
    pdf_path: Path,
    pack_name: str,
) -> Path:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=landscape(A4))
    width, height = landscape(A4)
    c.setFillColorRGB(0.06, 0.15, 0.27)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Times-Bold", 28)
    c.drawString(22 * mm, height / 2 + 12 * mm, pack_name)
    c.setFont("Times-Italic", 13)
    c.drawString(22 * mm, height / 2 - 2 * mm, "Monthly / weekly / daily candles  ·  RSI 14 with 40 and 60")
    c.setFont("Times-Roman", 11)
    c.drawString(22 * mm, height / 2 - 16 * mm, "Not live. Mark ranges, shift, PRD/NRD, then GFS. Offline on iPad.")
    c.showPage()

    footer = "Mark on device  ·  40 dashed rust  ·  60 dashed teal  ·  no network required"
    for symbol, paths in symbol_paths.items():
        for key in ("monthly", "weekly", "daily"):
            label = TIMEFRAMES[key]["label"]
            _draw_image_page(c, paths[key], f"{symbol}   ·   {label}", footer)
        # annotation sheet
        c.setFillColorRGB(0.97, 0.96, 0.94)
        c.rect(0, 0, width, height, fill=1, stroke=0)
        c.setFillColorRGB(0.06, 0.15, 0.27)
        c.setFont("Times-Bold", 18)
        c.drawString(18 * mm, height - 20 * mm, f"{symbol}  ·  your marks")
        c.setFont("Times-Roman", 12)
        notes = [
            "Range (M/W/D): bullish 40–80  /  neutral 40–60  /  bearish 20–60",
            "Range shift?  where? ________________________________",
            "PRD (price LL, RSI HL) on:  M / W / D / none",
            "NRD (price HH, RSI LH) on:  M / W / D / none",
            "GFS: Monthly RSI ____   Weekly RSI ____   Daily RSI ____",
            "Alert candle / S-R: ________________________________",
            "Verdict: buy dip / sell rally / skip   because: ________________",
        ]
        y = height - 40 * mm
        for line in notes:
            c.drawString(18 * mm, y, line)
            y -= 14 * mm
        c.setFont("Times-Italic", 8)
        c.setFillColorRGB(0.36, 0.4, 0.44)
        c.drawString(18 * mm, 8 * mm, footer)
        c.showPage()
    c.save()
    return pdf_path
