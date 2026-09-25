"""Typeset REPORT.md as REPORT.pdf in Times New Roman.

Renders the Markdown to HTML with a print stylesheet, then prints it to PDF
with headless Microsoft Edge (Windows). Run after build_report.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).parent
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CSS = """
@page { size: Letter; margin: 0.9in 0.95in; }
html { font-family: 'Times New Roman', Times, serif; font-size: 11.5pt; line-height: 1.45; color: #111; }
body { margin: 0; }
h1 { font-size: 22pt; margin: 0 0 4pt; font-weight: bold; }
h1 + p { margin-top: 0; font-style: italic; color: #333; }
h2 { font-size: 15pt; margin: 22pt 0 6pt; break-after: avoid; border-bottom: 0.6pt solid #444; padding-bottom: 2pt; }
h3 { font-size: 12.5pt; margin: 14pt 0 4pt; break-after: avoid; font-style: italic; font-weight: bold; }
p, li { text-align: justify; hyphens: auto; }
ul, ol { padding-left: 1.4em; }
li { margin-bottom: 3pt; }
img { display: block; max-width: 100%; margin: 8pt auto 10pt; break-inside: avoid; }
table { border-collapse: collapse; margin: 8pt auto 12pt; font-size: 10pt; break-inside: avoid;
        border-top: 1.2pt solid #111; border-bottom: 1.2pt solid #111; }
thead th { border-bottom: 0.6pt solid #111; }
th, td { padding: 3pt 8pt; font-variant-numeric: tabular-nums; }
th { font-weight: bold; }
code { font-family: 'Courier New', monospace; font-size: 9.5pt; }
pre { font-size: 9pt; background: #f4f4f4; padding: 8pt 10pt; white-space: pre-wrap; break-inside: avoid; }
a { color: #111; text-decoration: none; }
h2#abstract + p { font-size: 11pt; }
"""


def main() -> None:
    body = markdown.markdown((ROOT / "REPORT.md").read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "toc"])
    html_path = ROOT / "_report_print.html"
    html_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>Sportsbook Customer Economics</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    pdf_path = ROOT / "REPORT.pdf"
    subprocess.run(
        [str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    html_path.unlink()
    print(f"wrote {pdf_path.name} ({pdf_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
