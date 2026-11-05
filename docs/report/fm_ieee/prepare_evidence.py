"""Capture immutable writing inputs and primary bibliographic records; no ML runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERIFY = HERE / "verification"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", required=True, type=Path)
    args = parser.parse_args()
    VERIFY.mkdir(parents=True, exist_ok=True)
    index_path = args.kb_dir / "LITERATURE_SOURCES.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    protocol = Path(index["protocol"]["path"])
    assert digest(protocol) == index["protocol"]["sha256"], "Protocol identity changed"
    copies = {
        "protocol_v1_0.md": protocol,
        "literature_kb_snapshot.md": args.kb_dir / "LITERATURE_KB.md",
        "literature_sources_snapshot.json": index_path,
    }
    provenance = {}
    for name, source in copies.items():
        target = VERIFY / name
        if target.exists():
            assert digest(source) == digest(target), f"Refuse replacement: {target}"
        else:
            shutil.copyfile(source, target)
        provenance[name] = {"original_path": str(source), "sha256": digest(source)}
    (VERIFY / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    baseline = VERIFY / "protected_before.json"
    if not baseline.exists():
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "data", "results", "docs/report/Report.tex"],
            cwd=ROOT,
        ).decode("utf-8").split("\0")
        hashes = {p: digest(ROOT / p) for p in tracked if p}
        baseline.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    with (ROOT / "data/manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    classes = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
    counts = {
        split: {c: sum(r["split"] == split and r["label"] == c for r in rows)
                for c in classes}
        for split in ["train", "val", "test"]
    }
    summary = {"rows": len(rows), "rare_counts": counts,
               "note": "Image counts only; no new patient grouping or experiment."}
    (VERIFY / "manifest_counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))

    primary = VERIFY / "parent_refs"
    primary.mkdir(exist_ok=True)
    urls = {
        "DINOv2_README": "https://raw.githubusercontent.com/facebookresearch/dinov2/main/README.md",
        "ProtoNets": "https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Bibtex.bib",
        "CoCoOp": "https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.html",
        "CLAP": "https://openaccess.thecvf.com/content/CVPR2024/html/Silva-Rodriguez_A_Closer_Look_at_the_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2024_paper.html",
        "FairNLP": "https://aclanthology.org/2023.findings-acl.779.bib",
        "IEEEtran": "https://ctan.org/pkg/ieeetran",
        "arxiv_records": "https://export.arxiv.org/api/query?id_list=2304.07193,1911.04623,2010.07734,2203.05557,2305.16938,2312.12730,2609.00065,1912.07200&max_results=8",
        "CrossDomain": "https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720120.pdf",
    }
    session = requests.Session()
    session.headers["User-Agent"] = "RareOCT-IEEE-draft-reference-verification/1.0"
    retrievals = []
    for name, url in urls.items():
        suffix = ".pdf" if url.endswith(".pdf") else ".txt"
        target = primary / (name + suffix)
        if not target.exists():
            try:
                response = session.get(url, timeout=40)
                response.raise_for_status()
                target.write_bytes(response.content)
                status = response.status_code
            except requests.RequestException as exc:
                retrievals.append({"name": name, "url": url, "error": str(exc)})
                print(name, "FAILED", str(exc))
                continue
        else:
            status = "reused local retrieval"
        retrievals.append({"name": name, "url": url, "status": status,
                           "sha256": digest(target), "local": target.name,
                           "accessed_utc": datetime.now(timezone.utc).isoformat()})
        print(name, "OK", target.stat().st_size)
        time.sleep(0.3)
    query_url = "https://api.crossref.org/works"
    query = {"query.title": "A Broader Study of Cross-Domain Few-Shot Learning", "rows": 3}
    response = session.get(query_url, params=query, timeout=40)
    response.raise_for_status()
    target = primary / "crossdomain_crossref.json"
    target.write_bytes(response.content)
    retrievals.append({"name": "crossdomain_crossref", "url": response.url,
                       "sha256": digest(target), "status": response.status_code})
    (primary / "retrievals.json").write_text(json.dumps(retrievals, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
