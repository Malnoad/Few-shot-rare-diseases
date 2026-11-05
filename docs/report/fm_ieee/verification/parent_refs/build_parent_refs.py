"""Derive the parent bibliography batch from archived primary metadata."""
from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def escape(value: str) -> str:
    for char, replacement in [("&", r"\&"), ("%", r"\%"), ("_", r"\_")]:
        value = value.replace(char, replacement)
    accents = {"é": r"{\'e}", "è": r"{\`e}", "í": r"{\'i}",
               "ö": r"{\"o}", "ü": r"{\"u}", "á": r"{\'a}"}
    for char, replacement in accents.items():
        value = value.replace(char, replacement)
    return value


def entry(kind: str, key: str, fields: dict[str, str]) -> str:
    return "@" + kind + "{" + key + ",\n" + "\n".join(
        "  " + name + " = {" + value + "}," for name, value in fields.items()
    ) + "\n}\n"


def main() -> None:
    root = ET.fromstring((HERE / "arxiv_records.txt").read_bytes())
    records = {}
    for record in root.findall("a:entry", ATOM):
        identifier = record.findtext("a:id", namespaces=ATOM, default="").split("/abs/")[-1]
        records[re.sub(r"v\d+$", "", identifier)] = record
    assert len(records) == 8 and all("Error" not in r.findtext("a:title", namespaces=ATOM, default="") for r in records.values())
    output = ["% Primary records retrieved 5 November 2026. Full authors retained.\n"]
    for identifier, key, kind, year, venue in [
        ("2304.07193", "DINOv2", "article", "2024", "Transactions on Machine Learning Research"),
        ("1911.04623", "SimpleShot2019", "article", "2019", "arXiv preprint arXiv:1911.04623"),
        ("2010.07734", "STARTUP2021", "inproceedings", "2021", "International Conference on Learning Representations"),
        ("2609.00065", "Kassis2026", "article", "2026", "arXiv preprint arXiv:2609.00065"),
    ]:
        record = records[identifier]
        title = " ".join(record.findtext("a:title", namespaces=ATOM, default="").split())
        authors = [a.findtext("a:name", namespaces=ATOM, default="") for a in record.findall("a:author", ATOM)]
        fields = {"title": "{" + escape(title) + "}",
                  "author": " and ".join(escape(a) for a in authors),
                  "year": year, "journal" if kind == "article" else "booktitle": venue}
        if key == "DINOv2":
            fields["author"] = fields["author"].replace("Huy Vo", "Huy V. Vo")
            fields["url"] = "https://openreview.net/forum?id=a68SUt6zFt"
        elif key == "STARTUP2021":
            fields["url"] = "https://arxiv.org/abs/2010.07734"
        else:
            fields["doi"] = "10.48550/arXiv." + identifier
        output.append(entry(kind, key, fields))
    for file, key in [("CLAP.txt", "CLAP2024"), ("CoCoOp.txt", "CoCoOp2022")]:
        source = (HERE / file).read_text(encoding="utf-8")
        match = re.search(r'<div class="bibref pre-white-space">(.*?)</div>', source, re.S)
        assert match is not None
        bib = html.unescape(match.group(1))
        bib = re.sub(r"(@\w+\{)[^,]+", lambda m: m.group(1) + key, bib, count=1)
        bib = re.sub(r"(pages\s*=\s*\{\d+)-(\d+\})", r"\1--\2", bib)
        output.append(bib + "\n")
    bib = (HERE / "FairNLP.txt").read_text(encoding="utf-8")
    bib = re.sub(r"(@\w+\{)[^,]+", lambda m: m.group(1) + "FairNLP2023", bib, count=1)
    output.append(bib)
    official_proto = HERE / "ProtoNets.txt"
    assert official_proto.exists(), "Fetch the actual BibTeX link under /file/, not /hash/"
    bib = official_proto.read_text(encoding="utf-8")
    bib = re.sub(r"(@\w+\{)[^,]+", lambda m: m.group(1) + "ProtoNets2017", bib, count=1)
    output.append(bib)
    candidates = json.loads((HERE / "crossdomain_crossref.json").read_text(encoding="utf-8"))["message"]["items"]
    matched = [r for r in candidates if r["title"][0] == "A Broader Study of Cross-Domain Few-Shot Learning"]
    assert len(matched) == 1
    record = matched[0]
    fields = {"title": "{" + record["title"][0] + "}",
              "author": " and ".join(a["given"] + " " + a["family"] for a in record["author"]),
              "booktitle": "Computer Vision -- ECCV 2020", "year": "2020",
              "pages": record["page"].replace("-", "--"), "doi": record["DOI"]}
    output.append(entry("inproceedings", "CrossDomain2020", fields))
    (HERE / "parent_refs.bib").write_text("\n".join(output), encoding="utf-8")
    print("Parent bibliography: 9 primary-source entries")


if __name__ == "__main__":
    main()
