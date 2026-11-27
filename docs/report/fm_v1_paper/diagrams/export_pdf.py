"""Export the SVG of a diagram-design HTML file to a vector PDF, cropped to its viewBox, with headless Chrome.

The diagram-design skill exports with Playwright, which this laptop does not have; Chrome's own print-to-PDF gives
the same kind of output (vector, fonts embedded) without installing anything. It loads the Geist fonts from Google
Fonts, so it needs network access; offline, Chrome falls back to system fonts.

    python docs/report/fm_v1_paper/diagrams/export_pdf.py docs/report/fm_v1_paper/diagrams/pipeline.html
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600'
         '&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">')


def main(src: Path) -> Path:
    html = src.read_text(encoding="utf-8")
    svg = re.search(r"<svg\b.*?</svg>", html, re.S)
    if not svg:
        raise SystemExit(f"no <svg> in {src}")
    _, _, w, h = (int(v) for v in re.search(r'viewBox="([\d ]+)"', svg.group(0)).group(1).split())
    page = (f"<!DOCTYPE html><html><head><meta charset='UTF-8'>{FONTS}<style>"
            f"@page {{ size: {w}px {h}px; margin: 0; }} html, body {{ margin: 0; padding: 0; background: #ffffff; }}"
            f"svg {{ display: block; width: {w}px; height: {h}px; }}</style></head><body>{svg.group(0)}</body></html>")
    out = src.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as tmp:
        page_file = Path(tmp) / "print.html"
        page_file.write_text(page, encoding="utf-8")
        subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=15000", f"--user-data-dir={Path(tmp) / 'profile'}",
                        f"--print-to-pdf={out}", page_file.as_uri()], check=True, capture_output=True, timeout=120)
    print("wrote", out)
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
