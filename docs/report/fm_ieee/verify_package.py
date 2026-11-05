"""Verify document provenance, historical arithmetic and delivered PDF geometry."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERIFY = HERE / "verification"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = {}
    baseline = json.loads((VERIFY / "protected_before.json").read_text(encoding="utf-8"))
    changed = [name for name, old in baseline.items() if digest(ROOT / name) != old]
    assert not changed, changed
    report["protected_files"] = {"checked": len(baseline), "changed": changed,
                                 "scope": "Tracked data, tracked results and pre-existing Report.tex"}
    assert digest(VERIFY / "protocol_v1_0.md") == "4e8df8d91c0021187c05b103a9b0a0e0319a055fd91108da1f4cf3045d00c5df"
    report["protocol_sha256"] = digest(VERIFY / "protocol_v1_0.md")
    text = (HERE / "main.tex").read_text(encoding="utf-8")
    ledger = (HERE / "CLAIM_SOURCE_LEDGER.md").read_text(encoding="utf-8")
    claims = re.findall(r"\\claim\{(C\d+)\}", text)
    assert claims == [f"C{i:02}" for i in range(1, 32)], claims
    assert all(f"| {key} |" in ledger for key in claims)
    report["claim_ledger"] = {"anchors": len(claims), "all_mapped": True}
    assert r"\documentclass[conference]{IEEEtran}" in text
    assert not re.search(r"\\(?:geometry|newgeometry|fontsize|vspace|enlargethispage|baselinestretch)\b", text)

    refs = (HERE / "references.bib").read_text(encoding="utf-8")
    keys = [key for kind, key in re.findall(r"@(\w+)\s*\{\s*([^,]+),", refs)
            if kind.lower() != "ieeetranbstctl"]
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text)
             for key in group.split(",")}
    assert len(set(keys)) == len(keys), "Duplicate bibliography keys"
    assert cited == set(keys), {"missing": sorted(cited-set(keys)), "unused": sorted(set(keys)-cited)}
    report["citations"] = {"entries": len(keys), "cited": len(cited), "unresolved": []}

    with (ROOT / "results/seed_check.csv").open(newline="", encoding="utf-8") as handle:
        seeds = list(csv.DictReader(handle))
    with (ROOT / "results/stage4/metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = {row["model"]: row for row in csv.DictReader(handle)}
    numbers = {}
    for variant in ["none", "basic", "gan"]:
        values = [float(row["acc"]) for row in seeds if row["variant"] == variant]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        acc = float(metrics[variant]["acc"])
        assert f"{acc:.2f}" in text
        assert f"{mean:.2f}\\pm{sd:.2f}" in text
        numbers[variant] = {"seed42_percent": acc, "five_seed_mean_percent": mean,
                            "five_seed_sample_sd_pp": sd, "n_seeds": len(values)}
    report["historical_table"] = numbers

    pdf_path = HERE / "main.pdf"
    pages = []
    previews = VERIFY / "pages"
    previews.mkdir(exist_ok=True)
    with pymupdf.open(pdf_path) as document:
        assert 4 <= len(document) <= 5, f"Expected 4-5 pages, got {len(document)}"
        texts = []
        for index, page in enumerate(document):
            assert abs(page.rect.width - 612) < 1 and abs(page.rect.height - 792) < 1
            page_text = page.get_text(sort=True)
            assert len(page_text.strip()) > 350, f"Near-empty page {index+1}"
            assert not any(token in page_text for token in ["[?]", "??", "\ufffd"])
            outside = []
            small = []
            span_sizes = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        rect = pymupdf.Rect(span["bbox"])
                        if rect.x0 < 0 or rect.y0 < 0 or rect.x1 > page.rect.width+0.5 or rect.y1 > page.rect.height+0.5:
                            outside.append(span["text"])
                        if span["text"].strip():
                            span_sizes.append(span["size"])
                            # IEEE's 5 TeX-pt math scripts are about 4.98 PDF pt.
                            if span["size"] < 4.5:
                                small.append(span["text"])
            assert not outside, {"page": index+1, "outside_page": outside}
            assert not small, {"page": index+1, "text_smaller_than_4_5_pdf_pt": small}
            fonts = page.get_fonts(full=True)
            assert not any(font[2] == "Type3" for font in fonts), fonts
            image_path = previews / f"page_{index+1}.png"
            page.get_pixmap(dpi=130, alpha=False).save(image_path)
            texts.append(page_text)
            pages.append({"page": index+1, "width_pt": page.rect.width,
                          "height_pt": page.rect.height, "words": len(page_text.split()),
                          "minimum_text_size_pt": min(span_sizes),
                          "outside_page_spans": len(outside), "font_names": sorted({font[3] for font in fonts})})
        (VERIFY / "pdf_text.txt").write_text("\n\n".join(f"PAGE {i+1}\n{t}" for i,t in enumerate(texts)), encoding="utf-8")
        report["pdf"] = {"pages": len(document), "sha256": digest(pdf_path), "details": pages}

    thumbnails = []
    for page in pages:
        with Image.open(previews / f"page_{page['page']}.png") as image:
            image.thumbnail((510, 660))
            thumbnails.append(image.copy())
    sheet = Image.new("RGB", (3*534, 2*700), "#151b23")
    draw = ImageDraw.Draw(sheet)
    for i, image in enumerate(thumbnails):
        x, y = (i % 3)*534+12, (i // 3)*700+27
        sheet.paste(image, (x,y))
        draw.text((x,y-19), f"PAGE {i+1}", fill="white")
    sheet.save(VERIFY / "contact_sheet.png")
    log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
    forbidden = [line for line in log.splitlines() if any(word in line for word in [
        "Overfull", "Missing character", "LaTeX Error", "Emergency stop"])
        or (line.startswith("LaTeX Warning:") and "undefined" in line)]
    assert not forbidden, forbidden
    report["latex_log"] = {"overfull_or_unresolved_or_missing_glyphs": forbidden,
                           "underfull_warnings": log.count("Underfull"),
                           "early_font_shape_fallback_warnings": log.count("LaTeX Font Warning:")}
    report["manual_review"] = "See VERIFICATION.md for visual and semantic review; automated bounds are not a layout certification."
    (VERIFY / "package_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PDF: {report['pdf']['pages']} IEEE conference pages; {len(keys)} resolved citations; {len(claims)} mapped claim anchors")
    print(f"Protected files: {len(baseline)} checked, 0 changed")
    print("LaTeX: no overfull boxes, unresolved references or missing glyphs")
    print("Package checks: PASS")


if __name__ == "__main__":
    main()
