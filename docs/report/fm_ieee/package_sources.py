"""Package authored submission sources; do not bundle retrieved paper full texts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent


def main() -> None:
    generic = json.loads((HERE / "verification/citation_validation.json").read_text(encoding="utf-8"))
    assert len(generic["errors"]) == 1
    assert generic["errors"][0]["entry"] == "RETFound2023"
    assessment = {
        "scholarly_and_tooling_references": 21,
        "resolved_unique_citations": 21,
        "duplicate_keys": 0,
        "generic_errors": generic["errors"],
        "adjudicated_errors_in_citations": 0,
        "adjudication": "RETFound's braced corporate author UK Biobank Eye & Vision Consortium contains an escaped ampersand, not an author separator. Published byline checked in retinal_refs/verification_report.md; preserve the byline rather than deleting the consortium to satisfy a heuristic.",
        "style_control": "IEEEcontrol is a nonprinting IEEEtranBSTCTL record, not a 22nd reference.",
        "recommended_field_warnings": "No invented volume/pages/DOI for records without them. Article identifiers are labeled Art. no. See VERIFICATION.md.",
        "publication_choices": {
            "DINOv2": "TMLR January 2024, verified primary PDF header and byline",
            "OCTID2020": "Verified journal version, Computers & Electrical Engineering 81, article 106532 (2020)",
            "BenchReAD2025": "Official MICCAI 2025 conference version; conflicting Springer 2026 citation metadata disclosed"
        },
        "limits": "No comprehensive correction/retraction audit. Primary reading scopes, access failures and unverified details remain explicit."
    }
    (HERE / "verification/citation_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    svg = (HERE / "methods_figure.svg").read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rare-OCT proposed-methods figure</title><style>
body{margin:0;color:#fff;font:15px/1.5 system-ui,sans-serif}main{width:100%;padding:0}
figure{margin:0;background:#151b23;padding:16px;border-radius:10px}svg{display:block;width:100%;height:auto;background:white;border-radius:4px}svg text{font-family:Arial,sans-serif!important}
figcaption{margin-top:12px;color:#fff}p{margin:0 0 12px;color:#fff}.status{font-size:13px;color:#d9e3ef}
</style></head><body><main>
<p class="status">5 compiled IEEE pages &nbsp;·&nbsp; 21 references &nbsp;·&nbsp; 31 mapped claims &nbsp;·&nbsp; proposed experiments only</p>
<figure>""" + svg + """<figcaption>Five-way recognition and nine-way NORMAL source auditing use separate heads. Model nomination uses development data only; legacy and external outcomes cannot revise it.</figcaption></figure>
</main></body></html>"""
    (HERE / "preview.html").write_text(page, encoding="utf-8")
    files = ["main.tex", "references.bib", "main.bbl", "methods_figure.pdf",
             "methods_figure.svg", "methods_figure.png", "make_figure.py",
             "CLAIM_SOURCE_LEDGER.md", "README.md", "VERIFICATION.md",
             "verification/protocol_v1_0.md", "verification/input_provenance.json",
             "verification/figure_provenance.json", "verification/package_checks.json",
             "verification/citation_assessment.json", "verification/manifest_counts.json",
             "verification/retinal_refs/verification_report.md",
             "verification/retinal_refs/field_verification.json",
             "verification/parent_refs/REFERENCE_AUDIT.md"]
    archive_path = HERE / "fm_ieee_sources.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(HERE / name, arcname="fm_ieee/" + name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name in files:
            assert archive.read("fm_ieee/" + name) == (HERE / name).read_bytes()
    manifest = {"files": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ["main.pdf"] + files},
                "source_zip_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "zip_note": "Authored submission sources, figure and concise audit records; no retrieved paper full texts, study data or results."}
    (HERE / "verification/delivery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Source package: {len(files)} files, CRC and byte equality verified; {archive_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
