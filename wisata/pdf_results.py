"""Catat setiap tahap notebook ke CSV/log/gambar dan PDF yang diperbarui."""

from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import textwrap
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import pandas as pd


class Tee(StringIO):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, value):
        self.stream.write(value)
        return super().write(value)

    def flush(self):
        self.stream.flush()


class ResultRecorder:
    def __init__(self, directory: str, mock: bool):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mock = mock
        self.records = []
        self.title = "Adaptive Fusion BGE-M3 + E5 - Malik"
        self.current = None
        self.started = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat()

    @contextmanager
    def stage(self, name):
        import sys
        entry = {"stage": name, "tables": [], "figures": [], "status": "running"}
        self.records.append(entry)
        self.current = entry
        buffer = Tee(sys.stdout)
        try:
            with redirect_stdout(buffer):
                yield
            entry["status"] = "passed"
        except BaseException as exc:
            entry["status"] = "failed"
            entry["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            entry["stdout"] = buffer.getvalue()
            number = len(self.records)
            (self.directory / f"stage_{number:02d}.log").write_text(entry["stdout"], encoding="utf-8")
            self.current = None
            self.flush()

    def display(self, value):
        from IPython.display import display
        if value is None:
            return
        table = value.data if hasattr(value, "data") and isinstance(value.data, pd.DataFrame) else value
        if isinstance(table, pd.Series):
            table = table.to_frame()
        if isinstance(table, pd.DataFrame) and self.current is not None:
            name = f"table_{len(self.records):02d}_{len(self.current['tables']) + 1:02d}.csv"
            table.to_csv(self.directory / name, encoding="utf-8")
            self.current["tables"].append(name)
        display(value)

    def capture_figures(self):
        import matplotlib.pyplot as plt
        if self.current is None:
            return
        for i in plt.get_fignums():
            name = f"figure_{len(self.records):02d}_{len(self.current['figures']) + 1:02d}.png"
            plt.figure(i).savefig(self.directory / name, dpi=130, bbox_inches="tight")
            self.current["figures"].append(name)

    def flush(self):
        from matplotlib.font_manager import findfont
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        from PIL import Image as PILImage
        pdfmetrics.registerFont(TTFont("ResultFont", findfont("DejaVu Sans")))
        styles = getSampleStyleSheet()
        for style in styles.byName.values():
            style.fontName = "ResultFont"
        styles["BodyText"].fontSize = 8
        styles["BodyText"].leading = 11
        width, height = landscape(A4)
        usable = width - 64
        body = [Paragraph(self.title, styles["Title"]),
                Paragraph("UJI TEKNIS MOCK - BUKAN HASIL RISET" if self.mock else "ENCODER ASLI - periksa cakupan run dan manifest", styles["Heading2"]),
                Paragraph("Waktu mulai: " + escape(self.started), styles["BodyText"]),
                Paragraph("Tabel/gambar/console dicatat per tahap. Nilai lengkap tabel ada di CSV; PDF menampilkan maksimal 40 baris per tabel dan teks panjang diringkas. Skor mock tidak boleh dipakai memilih model penelitian atau deployment.", styles["BodyText"])]
        for entry in self.records:
            body.extend([PageBreak(), Paragraph(escape(entry["stage"] + " - " + entry["status"]), styles["Heading1"])])
            if entry.get("error"):
                body.append(Paragraph(escape(entry["error"]), styles["BodyText"]))
            for line in entry.get("stdout", "").splitlines():
                for segment in textwrap.wrap(line, 130) or [""]:
                    body.append(Paragraph(escape(segment) or "&nbsp;", styles["BodyText"]))
            for name in entry["tables"]:
                frame = pd.read_csv(self.directory / name).fillna("")
                body.extend([Spacer(1, 10), Paragraph(escape(name) + f" ({len(frame)} baris; file CSV menyimpan seluruh nilai)", styles["Heading3"])])
                for start in range(0, len(frame.columns), 6):
                    part = frame.iloc[:40, start:start + 6]
                    rows = [[Paragraph(escape(str(x)[:220]), styles["BodyText"]) for x in part.columns]]
                    rows += [[Paragraph(escape(str(x)[:220]), styles["BodyText"]) for x in row]
                             for row in part.itertuples(index=False, name=None)]
                    table = Table(rows, colWidths=[usable / len(part.columns)] * len(part.columns), repeatRows=1)
                    table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0), colors.HexColor("#dbeafe")),
                                               ("VALIGN", (0,0),(-1,-1),"TOP"),
                                               ("GRID", (0,0),(-1,-1),.25, colors.lightgrey)]))
                    body.extend([table, Spacer(1, 8)])
            for name in entry["figures"]:
                path = self.directory / name
                with PILImage.open(path) as img:
                    w, h = img.size
                ratio = min(usable / w, (height - 100) / h)
                body.extend([PageBreak(), Paragraph(escape(name), styles["Heading3"]), Image(str(path), w * ratio, h * ratio)])
        temporary = self.directory / "hasil_sementara.pdf"
        def footer(canvas, doc):
            canvas.setFont("ResultFont", 7)
            canvas.drawString(32, 18, ("MOCK - BUKAN HASIL RISET | " if self.mock else "ENCODER ASLI | ") + str(doc.page))
        SimpleDocTemplate(str(temporary), pagesize=landscape(A4), leftMargin=32, rightMargin=32,
                          topMargin=32, bottomMargin=32).build(body, onFirstPage=footer, onLaterPages=footer)
        temporary.replace(self.directory / "hasil.pdf")
        (self.directory / "catatan_tahap.json").write_text(json.dumps({"mock": self.mock, "started": self.started,
            "stages": self.records}, ensure_ascii=False, indent=2), encoding="utf-8")
