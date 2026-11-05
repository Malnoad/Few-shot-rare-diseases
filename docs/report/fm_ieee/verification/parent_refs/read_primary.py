"""Archive bounded primary passages and record page locators for writing audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pymupdf
import requests

HERE = Path(__file__).resolve().parent
INDEX = HERE.parent / "literature_sources_snapshot.json"


def main() -> None:
    sources = json.loads(INDEX.read_text(encoding="utf-8"))["sources"]
    lookup = {s["id"]: s for s in sources}
    snippets = []
    for key, pages in [("F15", [0, 2]), ("F16", [0, 2]),
                       ("F21", [0]), ("F25", [0, 2])]:
        source = lookup[key]
        path = Path(source["pdf"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == source["sha256"]
        with pymupdf.open(path) as document:
            text = "\n\n".join(f"PDF PAGE {p+1}\n" + document[p].get_text(sort=True)
                                 for p in pages)
        (HERE / (key + "_primary_passages.txt")).write_text(text, encoding="utf-8")
        snippets.append({"id": key, "pdf_path": str(path), "sha256": actual,
                         "read_pdf_pages": [p+1 for p in pages]})
    with pymupdf.open(HERE / "CrossDomain.pdf") as document:
        (HERE / "CrossDomain_primary_passages.txt").write_text(
            "\n\n".join(f"PDF PAGE {p+1}\n" + document[p].get_text(sort=True)
                         for p in [0, 1, 2]), encoding="utf-8")
    response = requests.get("https://arxiv.org/pdf/2304.07193", timeout=45)
    response.raise_for_status()
    assert response.content.startswith(b"%PDF-")
    with pymupdf.open(stream=response.content, filetype="pdf") as document:
        (HERE / "DINOv2_primary_title_abstract.txt").write_text(
            document[0].get_text(sort=True), encoding="utf-8")
    snippets.append({"id": "A03", "url": "https://arxiv.org/pdf/2304.07193",
                     "sha256": hashlib.sha256(response.content).hexdigest(),
                     "read_pdf_pages": [1], "scope": "Publication header, author list, abstract only"})
    atom = ET.fromstring((HERE / "arxiv_records.txt").read_bytes())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    summaries = [{"id": e.findtext("a:id", namespaces=ns),
                  "title": " ".join(e.findtext("a:title", namespaces=ns, default="").split()),
                  "authors": [a.findtext("a:name", namespaces=ns) for a in e.findall("a:author", ns)],
                  "abstract": " ".join(e.findtext("a:summary", namespaces=ns, default="").split()),
                  "published": e.findtext("a:published", namespaces=ns),
                  "updated": e.findtext("a:updated", namespaces=ns)} for e in atom.findall("a:entry", ns)]
    (HERE / "arxiv_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (HERE / "passage_provenance.json").write_text(json.dumps(snippets, indent=2), encoding="utf-8")
    print("Read 4 hash-matched local PDFs, official CrossDomain pages 1-3, DINOv2 title/abstract, and 8 official abstracts.")


if __name__ == "__main__":
    main()
