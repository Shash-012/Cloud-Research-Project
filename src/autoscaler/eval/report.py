"""PDF report builder (Step 5.4) with ReportLab. Every number comes from a table produced by Steps 5.1/5.2
(or the model scripts); missing inputs are reported as missing, never filled in by hand."""
import subprocess
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

STYLES = getSampleStyleSheet()
BODY = STYLES["BodyText"]
SMALL = STYLES["BodyText"].clone("small", fontSize=7.5, leading=9)


def fmt(v, digits=4):
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        return f"{v:.{digits}g}" if abs(v) >= 1e4 or (0 < abs(v) < 1e-3) else f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return str(v)


def df_table(df: pd.DataFrame, max_rows=40, digits=4, col_width=None):
    df = df.head(max_rows)
    data = [[Paragraph(f"<b>{c}</b>", SMALL) for c in df.columns]]
    data += [[Paragraph(fmt(v, digits), SMALL) for v in row] for row in df.itertuples(index=False)]
    t = Table(data, repeatRows=1, colWidths=col_width)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf2")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def image(path, width_cm=16.5):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def read(path: Path):
    return pd.read_csv(path) if path.exists() else None


def git_head(root) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def java_version() -> str:
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return (r.stderr or r.stdout).splitlines()[0]
    except (OSError, IndexError):
        return "unknown"


class Report:
    def __init__(self, title: str):
        self.story = [Paragraph(title, STYLES["Title"])]

    def h(self, text):
        self.story.append(Paragraph(text, STYLES["Heading2"]))

    def h3(self, text):
        self.story.append(Paragraph(text, STYLES["Heading3"]))

    def p(self, text):
        self.story.append(Paragraph(text, BODY))

    def table(self, df, caption=None, **kw):
        if caption:
            self.story.append(Paragraph(f"<i>{caption}</i>", SMALL))
        if df is None or len(df) == 0:
            self.p("<i>Not available: the table was not produced by this run.</i>")
            return
        self.story.append(df_table(df, **kw))
        self.story.append(Spacer(1, 6))

    def figure(self, path, caption=None, width_cm=16.5):
        path = Path(path)
        if not path.exists():
            self.p(f"<i>Figure not available: {path.name}</i>")
            return
        self.story.append(image(path, width_cm))
        if caption:
            self.story.append(Paragraph(f"<i>{caption}</i>", SMALL))
        self.story.append(Spacer(1, 6))

    def page_break(self):
        self.story.append(PageBreak())

    def build(self, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                          topMargin=1.5 * cm, bottomMargin=1.5 * cm).build(self.story)
        return out
