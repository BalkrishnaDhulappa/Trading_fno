"""Simple Markdown → printable PDF for offline iPad study."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = HexColor("#0F2744")
TEAL = HexColor("#1F6F8B")
INK = HexColor("#1A1A1A")
MUTED = HexColor("#5C6570")
RULE = HexColor("#D7DEE6")
WASH = HexColor("#F4F7FA")
ACCENT = HexColor("#C45C26")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle(
            "kicker",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=10,
            textColor=TEAL,
            spaceAfter=4,
            tracking=0.4,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=16,
            textColor=INK,
            spaceAfter=8,
        ),
        "li": ParagraphStyle(
            "li",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            textColor=INK,
            leftIndent=4,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=11,
            leading=15,
            textColor=NAVY,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            textColor=INK,
            backColor=WASH,
        ),
    }


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier' size='9'>\1</font>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def markdown_to_flowables(md: str, styles: dict[str, ParagraphStyle]) -> list:
    flow: list = []
    lines = md.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("> "):
            buf = [line[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith("> "):
                buf.append(lines[i][2:])
                i += 1
            inner = Paragraph(_inline(" ".join(buf)), styles["callout"])
            tbl = Table([[inner]], colWidths=[170 * mm])
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), WASH),
                        ("BOX", (0, 0), (-1, -1), 0.4, TEAL),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            flow.append(tbl)
            flow.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            flow.append(Paragraph(_inline(line[2:]), styles["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            flow.append(Paragraph(_inline(line[3:]), styles["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            flow.append(Paragraph(_inline(line[4:]), styles["h3"]))
            i += 1
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(ListItem(Paragraph(_inline(lines[i][2:]), styles["li"]), leftIndent=12))
                i += 1
            flow.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=18))
            flow.append(Spacer(1, 6))
            continue
        if line.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            flow.append(Preformatted("\n".join(buf), styles["code"]))
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "-", ">", "```")):
            para.append(lines[i].strip())
            i += 1
        flow.append(Paragraph(_inline(" ".join(para)), styles["body"]))
    return flow


def build_pdf(md_path: Path, pdf_path: Path, kicker: str) -> Path:
    styles = _styles()
    md = md_path.read_text(encoding="utf-8")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    def _page(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, fill=1, stroke=0)
        canvas.setFillColor(white)
        canvas.setFont("Times-Italic", 8)
        canvas.drawString(18 * mm, A4[1] - 8 * mm, "3×3 chart lab  ·  offline  ·  Malkan RSI / BB")
        canvas.setFillColor(RULE)
        canvas.rect(0, 0, A4[0], 12 * mm, fill=1, stroke=0)
        canvas.setFillColor(MUTED)
        canvas.setFont("Times-Roman", 8)
        canvas.drawString(18 * mm, 5 * mm, kicker)
        canvas.drawRightString(A4[0] - 18 * mm, 5 * mm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=md_path.stem,
        author="Personal 3x3 chart lab",
    )
    story = [Paragraph("Personal study  ·  not live market data", styles["kicker"])]
    story.extend(markdown_to_flowables(md, styles))
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return pdf_path
