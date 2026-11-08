"""Prepare frozen-encoder study inputs with no model, score or GPU (FM_PROTOCOL.md v1.0).

Usage (--root is the checkout that holds data/raw; --out is any folder outside data/):
  python fm_prepare.py screen --root D:/Few-shot --out OUT
  python fm_prepare.py freeze --root D:/Few-shot --out OUT [--reviews duplicate_review.csv]

screen  fingerprints the 146 rare images and writes the candidate pairs to review (section 4).
freeze  applies exact links and Lam's review decisions, then writes groups, folds, exclusions,
        paired support rosters and the nine-way roster (sections 4, 6, 8).

--reviews is a CSV with columns a,b,decision (confirmed | unresolved | rejected), one row per
flagged dHash pair. Confirmed and unresolved pairs share a group; only rejected ones do not.

Outputs are byte-reproducible and never overwritten with different content. Nothing is written
under data/, and this script does not create the final input lock, which also needs checkpoint
identities. Rosters depend on NumPy's streams, so the saved IDs are the record (section 6).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
from itertools import combinations
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import PIL
from PIL import Image

import fm_core as fm

COMMON = fm.NINE[:4]
COMMON_CAP = 1000  # section 8: identical seed-42 sample for every encoder
Rows = list[dict[str, str]]


def text_hash(path: Path) -> str:
    """SHA256 with CRLF folded to LF, so Windows and Colab checkouts agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_manifest(root: Path) -> Rows:
    """Manifest rows in file order, with POSIX paths (the canonical image IDs)."""
    with (root / "data/manifest.csv").open(encoding="utf-8", newline="") as stream:
        return [{**row, "path": row["path"].replace("\\", "/")} for row in csv.DictReader(stream)]


def write_once(path: Path, data: bytes) -> None:
    """Atomic write that stops rather than replace an output with different bytes."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite different output: {path}")
        return
    pending = path.with_name(path.name + ".tmp")
    pending.write_bytes(data)
    pending.replace(path)


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    write_once(path, buffer.getvalue().encode("utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_once(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_csv(path: Path) -> Rows:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def versions() -> dict[str, str]:
    return dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__)


def out_dir(root: Path, out: Path) -> Path:
    resolved = out.resolve()
    if resolved.is_relative_to((root / "data").resolve()):
        raise ValueError("outputs must stay outside data/")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def screen(root: Path, out: Path) -> dict[str, Any]:
    """Fingerprint every rare benchmark image and list exact and near-duplicate pairs."""
    out = out_dir(root, out)
    rare = sorted((r for r in load_manifest(root) if r["label"] in fm.RARE), key=lambda r: r["path"])
    inventory, hashes = [], {}
    for row in rare:
        path = root / row["path"]
        with Image.open(path) as image:
            rgb, dhash, size, mode = fm.rgb_fingerprint(image), fm.dhash64(image), image.size, image.mode
        inventory.append([row["path"], row["label"], row["split"], hashlib.sha256(path.read_bytes()).hexdigest(),
                          rgb, f"{dhash:016x}", size[0], size[1], mode])
        hashes[row["path"]] = (inventory[-1][3], rgb, dhash)
    write_csv(out / "image_inventory.csv", ["path", "label", "split", "file_sha256", "rgb_sha256", "dhash",
                                            "width", "height", "mode"], inventory)
    info = {row[0]: row for row in inventory}
    pairs, nearest = [], {path: 64 for path in hashes}
    for a, b in combinations(sorted(hashes), 2):
        distance = fm.hamming(hashes[a][2], hashes[b][2])
        nearest[a], nearest[b] = min(nearest[a], distance), min(nearest[b], distance)
        kind = "file" if hashes[a][0] == hashes[b][0] else "rgb" if hashes[a][1] == hashes[b][1] else \
            "dhash" if distance <= fm.HAMMING_MAX else None
        if kind:
            pairs.append([a, b, kind, distance, info[a][1], info[b][1], info[a][2], info[b][2]])
    write_csv(out / "screen_pairs.csv", ["a", "b", "kind", "hamming", "label_a", "label_b", "split_a", "split_b"], pairs)
    histogram = {str(k): sum(v == k for v in nearest.values()) for k in sorted(set(nearest.values()))}
    summary = dict(protocol="v1.0", manifest_sha256=text_hash(root / "data/manifest.csv"), images=len(rare),
                   hamming_limit=fm.HAMMING_MAX, pairs={k: sum(p[2] == k for p in pairs) for k in ("file", "rgb", "dhash")},
                   nearest_neighbour_hamming=histogram, versions=versions())
    write_json(out / "screen_summary.json", summary)
    return summary


def review_links(pairs: Rows, reviews: Path | None) -> list[tuple[str, str]]:
    """Exact links plus reviewed dHash links. Every flagged pair needs exactly one decision."""
    flagged = {(p["a"], p["b"]) for p in pairs if p["kind"] == "dhash"}
    decisions: dict[tuple[str, str], str] = {}
    for row in read_csv(reviews) if reviews else []:
        key = (row["a"], row["b"])
        if key not in flagged or key in decisions or row["decision"] not in ("confirmed", "unresolved", "rejected"):
            raise ValueError(f"invalid review row: {row}")
        decisions[key] = row["decision"]
    if missing := flagged - decisions.keys():
        raise ValueError(f"{len(missing)} flagged pairs have no review decision; first: {sorted(missing)[0]}")
    return [(p["a"], p["b"]) for p in pairs
            if p["kind"] != "dhash" or decisions[(p["a"], p["b"])] != "rejected"]


def nine_way_roster(rows: Rows, eligible: list[str]) -> list[list[str]]:
    """Fixed seed-42 common fitting sample plus eligible rare fitting rows, then all queries."""
    from stage3_classifier import major_indices  # imported here: pulls pandas, not TensorFlow
    fitting = [r for r in rows if r["split"] == "train" and r["label"] in COMMON]
    chosen = [fitting[int(i)] for i in major_indices([r["label"] for r in fitting], COMMON_CAP, fm.SEED)]
    keep = set(eligible)
    rare = [r for r in rows if r["path"] in keep]
    tests = [r for r in rows if r["split"] == "test"]
    return [[r["path"], r["label"], "fitting"] for r in chosen + rare] + [[r["path"], r["label"], "query"] for r in tests]


def freeze(root: Path, out: Path, reviews: Path | None = None) -> dict[str, Any]:
    """Apply the screen and write every derived roster. Any failed gate stops before writing them."""
    out = out_dir(root, out)
    previous = json.loads((out / "screen_summary.json").read_text(encoding="utf-8"))
    if previous["manifest_sha256"] != text_hash(root / "data/manifest.csv"):
        raise ValueError("manifest changed since the screen")
    inventory = read_csv(out / "image_inventory.csv")
    labels = {r["path"]: r["label"] for r in inventory}
    splits = {r["path"]: r["split"] for r in inventory}
    links = review_links(read_csv(out / "screen_pairs.csv"), reviews)
    for members in fm.components(labels, links):
        if len({labels[m] for m in members}) != 1:
            raise ValueError(f"component spans disease labels; resolve before freezing: {members}")
    excluded = fm.legacy_exclusions(splits, links)
    development = {p: labels[p] for p in labels if splits[p] == "train"}
    groups = fm.build_groups(development, [(a, b) for a, b in links if a in development and b in development], excluded)
    counts = fm.check_feasible(groups)
    report_k = fm.report_k_values(groups)

    folds = {c: fm.assign_folds(groups[c], i) for i, c in enumerate(fm.RARE)}
    write_csv(out / "groups.csv", ["group_id", "label", "rank", "fold", "member"],
              [[g.id, c, g.rank, folds[c][g.id], m] for c in fm.RARE for g in groups[c] for m in g.members])
    write_csv(out / "exclusions.csv", ["path", "label", "reason", "component"],
              [[p, labels[p], "linked_to_legacy_query", excluded[p]] for p in sorted(excluded)])

    slots = [f"s{i:02d}" for i in range(1, max(fm.K_DEV) + 1)]
    write_csv(out / "episodes_development.csv", ["fold", "label", "draw", *slots],
              [[fold, c, draw, *fm.draw_supports([g for g in groups[c] if folds[c][g.id] != fold], fold, i, draw, fm.DEV_NS)[:len(slots)]]
               for fold in range(fm.N_FOLDS) for i, c in enumerate(fm.RARE) for draw in range(fm.N_DRAWS)])
    report_slots = [f"s{i:02d}" for i in range(1, max(report_k) + 1)]
    write_csv(out / "episodes_reporting.csv", ["label", "draw", *report_slots],
              [[c, draw, *fm.draw_supports(groups[c], 0, i, draw, fm.REPORT_NS)[:len(report_slots)]]
               for i, c in enumerate(fm.RARE) for draw in range(fm.N_DRAWS)])

    eligible = sorted(m for c in fm.RARE for g in groups[c] for m in g.members)
    manifest = load_manifest(root)
    roster = nine_way_roster(manifest, eligible)
    write_csv(out / "nine_way_roster.csv", ["path", "label", "role"], roster)
    needed = sorted({row[0] for row in roster} | set(labels))
    write_csv(out / "extraction_inventory.csv", ["path", "sha256"],
              [[p, hashlib.sha256((root / p).read_bytes()).hexdigest()] for p in needed])
    # Section 7: fixed in path order before any score; IDs repeat only if too few images exist.
    timing = [eligible[i % len(eligible)] for i in range(100)]
    write_csv(out / "timing_inventory.csv", ["path"], [[p] for p in timing])

    produced = ["groups.csv", "exclusions.csv", "episodes_development.csv", "episodes_reporting.csv",
                "nine_way_roster.csv", "extraction_inventory.csv", "timing_inventory.csv"]
    summary = dict(protocol="v1.0", manifest_sha256=previous["manifest_sha256"], groups_per_class=counts,
                   excluded_rows=len(excluded), report_k=list(report_k), extraction_images=len(needed),
                   fitting_rows=sum(r[2] == "fitting" for r in roster), query_rows=sum(r[2] == "query" for r in roster),
                   versions=versions(), files={name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in produced})
    write_json(out / "freeze_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("screen", "freeze"))
    parser.add_argument("--root", type=Path, required=True, help="checkout containing data/raw")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, help="freeze only: duplicate review decisions")
    args = parser.parse_args()
    result = screen(args.root, args.out) if args.step == "screen" else freeze(args.root, args.out, args.reviews)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
