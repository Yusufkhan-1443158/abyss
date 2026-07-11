#!/usr/bin/env python3
"""Render USER_GUIDE.md into the branded USER_GUIDE.pdf (fpdf2, house style of
make_methodology_pdf.py). Re-run whenever the guide changes:

    .venv/bin/python3 scripts/make_user_guide_pdf.py
"""
import re
import subprocess
from datetime import date
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "USER_GUIDE.md"
OUT = ROOT / "USER_GUIDE.pdf"

NAVY = (15, 40, 75)
ACCENT = (0, 110, 160)
GREY = (90, 90, 90)
LIGHT = (235, 241, 247)
BODY = (30, 30, 30)


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.replace("—", "-").replace("–", "-").replace("²", "2").replace("±", "+/-")


class Guide(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, "Bathymetry from Space - User Guide", align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", align="C")


def cover(pdf: FPDF, version: str):
    pdf.add_page()
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 70, "F")
    pdf.set_y(24)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "Bathymetry from Space", align="C")
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 15)
    pdf.cell(0, 10, "User Guide", align="C")
    pdf.set_y(86)
    pdf.set_text_color(*NAVY)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prepared for AD Ports Group / Noatum Maritime - Marine Services", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6, "https://bathymetryvmarch-production.up.railway.app/", align="C")
    pdf.ln(7)
    pdf.cell(0, 6, f"{date.today().strftime('%d %B %Y')}  -  build {version}", align="C")
    pdf.set_y(130)
    pdf.set_fill_color(*LIGHT)
    pdf.set_text_color(*BODY)
    pdf.set_font("Helvetica", "", 10)
    intro = (
        "Satellite-derived bathymetry (SDB) platform for shallow coastal waters. "
        "Coordinates WGS84; vertical datum LAT (adjustable on upload); depths capped at 25 m "
        "(the physical limit of optical SDB in Gulf waters); maximum area per run 40 km2."
    )
    pdf.multi_cell(0, 6, intro, fill=True)


def h1(pdf: FPDF, t: str):
    if pdf.get_y() > pdf.h - 45:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*NAVY)
    pdf.ln(3)
    pdf.cell(0, 7, _strip_md(t))
    pdf.ln(8)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.5)
    y = pdf.get_y() - 2
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)


def para(pdf: FPDF, t: str):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*BODY)
    pdf.multi_cell(0, 5.4, _strip_md(t))
    pdf.ln(1)


def bullet(pdf: FPDF, t: str, num: str | None = None):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*BODY)
    marker = f"{num}." if num else chr(149)
    x = pdf.l_margin
    pdf.set_x(x)
    pdf.cell(7, 5.4, marker)
    pdf.multi_cell(pdf.w - pdf.r_margin - x - 7, 5.4, _strip_md(t))
    pdf.ln(0.5)


def table(pdf: FPDF, rows: list[list[str]]):
    widths = [30, pdf.w - pdf.l_margin - pdf.r_margin - 30]
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(255, 255, 255)
    for w, c in zip(widths, rows[0]):
        pdf.cell(w, 6.5, _strip_md(c), border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*BODY)
    for i, row in enumerate(rows[1:]):
        pdf.set_fill_color(*(LIGHT if i % 2 == 0 else (255, 255, 255)))
        for w, c in zip(widths, row):
            pdf.cell(w, 6, _strip_md(c), border=1, fill=True)
        pdf.ln()
    pdf.ln(2)


def build():
    version = "dev"
    try:
        version = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        pass

    lines = SRC.read_text().splitlines()
    pdf = Guide()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    cover(pdf, version)
    pdf.add_page()

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("# ") or ln == "---" or not ln.strip():
            i += 1
            continue
        # first prose block (already on cover)
        if i < 6 and not ln.startswith("#"):
            i += 1
            continue
        if ln.startswith("## "):
            h1(pdf, ln[3:])
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            table(pdf, rows)
            continue
        m = re.match(r"^(\d+)\.\s+(.*)", ln)
        if m or ln.startswith("- "):
            # gather continuation lines (indented)
            text = m.group(2) if m else ln[2:]
            num = m.group(1) if m else None
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not re.match(r"^\s*[-\d]", lines[i]):
                text += " " + lines[i].strip()
                i += 1
            bullet(pdf, text, num)
            continue
        # plain paragraph: gather until blank
        text = ln
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "-", "|")) \
                and not re.match(r"^\d+\.", lines[i]):
            text += " " + lines[i].strip()
            i += 1
        para(pdf, text)

    pdf.output(str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {pdf.page_no()} pages)")


if __name__ == "__main__":
    build()
