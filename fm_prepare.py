"""Prepare frozen-encoder study inputs with no model, score or GPU (FM_PROTOCOL.md v1.0).

Usage (--root is the checkout that holds data/raw; --out is any folder outside data/):
  python fm_prepare.py screen --root D:/Few-shot --out OUT
  python fm_prepare.py freeze --root D:/Few-shot --out OUT [--reviews duplicate_review.csv]
  python fm_prepare.py pack --root D:/Few-shot --out OUT --archive selected-images.zip
  python fm_prepare.py lock --root D:/Few-shot --out OUT --gate model-gate.json --archive input-lock.zip

screen  fingerprints the 146 rare images and writes the candidate pairs to review (section 4).
freeze  applies exact links and Lam's review decisions, then writes groups, folds, exclusions,
        paired support rosters and the nine-way roster (sections 4, 6, 8).

--reviews is a CSV with columns a,b,decision (confirmed | unresolved | rejected), one row per
flagged dHash pair. Confirmed and unresolved pairs share a group; only rejected ones do not.

Screen/freeze outputs are byte-reproducible and never overwritten with different content.
Nothing is written under data/. The final lock needs successful checkpoint/timing pilots and
explicit source/access decisions. Saved IDs, not seeds alone, are the record (section 6).
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import StringIO
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from typing import Any
import uuid
import zipfile

import numpy as np
import PIL
from PIL import Image
from PIL import features as pillow_features
import scipy

import fm_core as fm

COMMON = fm.NINE[:4]
COMMON_CAP = 1000  # section 8: identical seed-42 sample for every encoder
Rows = list[dict[str, str]]
SCRIPT_ROOT = Path(__file__).resolve().parent
PROTOCOL = "docs/research/FM_PROTOCOL.md"
PROTOCOL_SHA256 = "4e8df8d91c0021187c05b103a9b0a0e0319a055fd91108da1f4cf3045d00c5df"
# A2 A-9: the dated amendments are hashed with the code, so any later edit invalidates the lock.
CODE_FILES = ("fm_core.py", "fm_prepare.py", "fm_extract.py", "fm_evaluate.py", "fm_external.py", "stage3_classifier.py",
              "requirements-fm.txt", "requirements.txt", "docs/research/FM_AMENDMENTS.md")
PREPARED_FILES = ("image_inventory.csv","screen_pairs.csv","screen_summary.json","groups.csv","exclusions.csv",
                 "duplicate_decisions.csv","verified_patients.csv","components.csv","fold_counts.csv",
                 "queries_development.csv","episodes_development.csv","episodes_reporting.csv",
                 "nine_way_roster.csv","extraction_inventory.csv","timing_inventory.csv")


def sha256(path: Path) -> str:
    """Streaming hash: checkpoint files need not fit in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    """Canonical relative POSIX ID, never a path outside the input root."""
    value = value.replace("\\", "/")
    if not value or value.startswith("/") or ":" in value or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError(f"invalid relative image/artifact ID: {value!r}")
    return value


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_identity() -> dict[str, str]:
    """Content identity survives an authorized commit/push and CRLF conversion."""
    return {name: text_hash(SCRIPT_ROOT / name) for name in CODE_FILES}


def preparation_code() -> dict[str,str]:
    return {name:text_hash(SCRIPT_ROOT / name) for name in ("fm_core.py","fm_prepare.py","stage3_classifier.py")}


def preparation_recipe() -> dict:
    return dict(rare_order=list(fm.RARE),nine_way_order=list(fm.NINE),seed=fm.SEED,folds=fm.N_FOLDS,
                fold_namespace=fm.FOLD_NS,development_namespace=fm.DEV_NS,reporting_namespace=fm.REPORT_NS,
                draws=fm.N_DRAWS,development_k=list(fm.K_DEV),reporting_k=list(fm.K_REPORT),
                common_cap=COMMON_CAP,hamming_max=fm.HAMMING_MAX)


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_ROOT, text=True).strip()


@contextmanager
def writer_lock(folder: Path):
    """One writer per artifact directory; stale locks need a verified owner and token."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "writer.lock"
    identity = dict(token=uuid.uuid4().hex, host=socket.gethostname(), pid=os.getpid())
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream)
    except FileExistsError:
        raise ValueError(f"writer lock exists: {path}; inspect its PID/host before token unlock") from None
    try:
        yield
    finally:
        if path.exists() and read_json(path) == identity:
            path.unlink()


def unlock(folder: Path, token: str) -> None:
    path = folder / "writer.lock"
    if read_json(path)["token"] != token:
        raise ValueError("writer token mismatch")
    path.unlink()


def verify_files(folder: Path, files: dict[str, str]) -> None:
    for name, expected in files.items():
        if sha256(folder / safe_id(name)) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")


def complete(folder: Path, files: list[str]) -> None:
    """Commit point written after every listed artifact has reached disk."""
    hashes={name:sha256(folder / name) for name in files}
    verify_child_coverage(folder,hashes)
    write_json(folder / "complete.json", {"files": hashes})


def verify_child_coverage(folder: Path,files: dict[str,str]) -> None:
    """A completed parent cannot hide completed children from resume or export."""
    for path in folder.rglob("complete.json"):
        if path.parent==folder:
            continue
        relative=path.relative_to(folder).as_posix()
        child=read_json(path).get("files")
        if not isinstance(child,dict) or not child or relative not in files:
            raise ValueError("completion marker omits child identity/artifacts")
        for name,digest in child.items():
            target=(path.parent / safe_id(name)).relative_to(folder).as_posix()
            if files.get(target) != digest:
                raise ValueError(f"completion marker omits or contradicts child artifact: {target}")


def verify_complete(folder: Path) -> dict:
    marker = read_json(folder / "complete.json")
    if not isinstance(marker.get("files"), dict) or not marker["files"]:
        raise ValueError("completion marker needs a nonempty artifact hash map")
    verify_files(folder, marker["files"])
    verify_child_coverage(folder,marker["files"])
    return marker


def experiment_out(path: Path) -> Path:
    """New experimental artifacts never share a checkout with canonical results/data."""
    resolved = path.resolve()
    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=SCRIPT_ROOT, text=True).strip()
    main = (SCRIPT_ROOT / common).resolve().parent
    if any(resolved.is_relative_to(checkout) or checkout.is_relative_to(resolved) for checkout in (SCRIPT_ROOT, main)):
        raise ValueError("experiment outputs must be outside all project checkouts")
    return resolved


def text_hash(path: Path) -> str:
    """SHA256 with CRLF folded to LF, so Windows and Colab checkouts agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_manifest(root: Path) -> Rows:
    """Manifest rows in file order, with POSIX paths (the canonical image IDs)."""
    with (root / "data/manifest.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "pid", "split"} <= set(reader.fieldnames or []):
            raise ValueError("manifest schema differs")
        rows = [{**row, "path": safe_id(row["path"])} for row in reader]
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("duplicate manifest paths")
    if any(r["label"] not in fm.NINE or r["split"] not in ("train", "val", "test") for r in rows):
        raise ValueError("unknown manifest label or split")
    return rows


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
    write_once(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def read_csv(path: Path) -> Rows:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def versions() -> dict[str, str]:
    return dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__,
                zlib=str(pillow_features.version("zlib")),
                libjpeg_turbo=str(pillow_features.version("libjpeg_turbo")))


def math_environment() -> dict[str, str]:
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def writable_path(root: Path, out: Path) -> Path:
    resolved = out.resolve()
    protected = [(root / "data").resolve(), *[(root / "data" / name).resolve()
                                             for name in ("raw", "aug_basic", "gan")]]
    if any(resolved.is_relative_to(path) for path in protected):
        raise ValueError("outputs must stay outside data/")
    return resolved


def out_dir(root: Path, out: Path) -> Path:
    resolved=writable_path(root,out)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def archive_destination(root: Path,destination: Path) -> None:
    for path in (destination,destination.with_name(destination.name+".tmp"),
                 destination.with_name(destination.name+".json"),destination.with_name(destination.name+".json.tmp")):
        writable_path(root,path)


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
    blocks=Counter(("legacy-legacy" if p[6]==p[7]=="test" else
                    "fitting-legacy" if "test" in (p[6],p[7]) else "fitting-fitting",p[2]) for p in pairs)
    summary = dict(protocol="v1.0", manifest_sha256=text_hash(root / "data/manifest.csv"), images=len(rare),
                   producer_code=preparation_code(),recipe=preparation_recipe(),
                   hamming_limit=fm.HAMMING_MAX, pairs={k: sum(p[2] == k for p in pairs) for k in ("file", "rgb", "dhash")},
                   nearest_neighbour_hamming=histogram, versions=versions(),
                   pairs_by_block={block:{kind:blocks[block,kind] for kind in ("file","rgb","dhash")}
                       for block in ("fitting-fitting","fitting-legacy","legacy-legacy")},
                   files={name: sha256(out / name) for name in ("image_inventory.csv", "screen_pairs.csv")})
    write_json(out / "screen_summary.json", summary)
    return summary


def review_links(pairs: Rows, reviews: Path | None) -> list[tuple[str, str]]:
    """Exact links plus reviewed dHash links. Every flagged pair needs exactly one decision."""
    return reviewed_links(pairs,read_csv(reviews) if reviews else [])


def reviewed_links(pairs: Rows,rows: Rows) -> list[tuple[str,str]]:
    flagged = {(p["a"], p["b"]) for p in pairs if p["kind"] == "dhash"}
    decisions: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["a"], row["b"])
        if key not in flagged or key in decisions or row["decision"] not in ("confirmed", "unresolved", "rejected"):
            raise ValueError(f"invalid review row: {row}")
        decisions[key] = row["decision"]
    if missing := flagged - decisions.keys():
        raise ValueError(f"{len(missing)} flagged pairs have no review decision; first: {sorted(missing)[0]}")
    return [(p["a"], p["b"]) for p in pairs
            if p["kind"] != "dhash" or decisions[(p["a"], p["b"])] != "rejected"]


@dataclass(frozen=True)
class Grouping:
    labels:dict[str,str]
    splits:dict[str,str]
    image_links:list[tuple[str,str]]
    excluded:dict[str,str]
    groups:dict[str,list[fm.Group]]


def derive_grouping(inventory: Rows,pairs: Rows,decisions: Rows,patients: Rows) -> Grouping:
    """Single derivation used by freeze AND verification; ledgers never overrule evidence."""
    labels={r["path"]:r["label"] for r in inventory}
    splits={r["path"]:r["split"] for r in inventory}
    if len(labels)!=len(inventory) or any(c not in fm.RARE for c in labels.values()) or \
            any(s not in ("train","test") for s in splits.values()):
        raise ValueError("rare screen has invalid labels/splits or repeated paths")
    by_path={r["path"]:r for r in inventory}
    expected=[]
    for a,b in combinations(sorted(labels),2):
        left,right=by_path[a],by_path[b]
        distance=fm.hamming(int(left["dhash"],16),int(right["dhash"],16))
        kind="file" if left["file_sha256"]==right["file_sha256"] else "rgb" if left["rgb_sha256"]==right["rgb_sha256"] else \
             "dhash" if distance<=fm.HAMMING_MAX else None
        if kind:
            expected.append(dict(a=a,b=b,kind=kind,hamming=str(distance),label_a=labels[a],label_b=labels[b],
                                 split_a=splits[a],split_b=splits[b]))
    if pairs!=expected:
        raise ValueError("saved screen pairs differ from the image fingerprint inventory")
    exact={(p["a"],p["b"]) for p in pairs if p["kind"]!="dhash"}
    saved_exact=[(r["a"],r["b"]) for r in decisions if r["decision"]=="exact"]
    if len(saved_exact)!=len(exact) or set(saved_exact)!=exact:
        raise ValueError("exact-link decisions differ from the screen")
    image_links=reviewed_links(pairs,[r for r in decisions if r["decision"]!="exact"])
    for members in fm.components(labels,image_links):
        if len({labels[m] for m in members})!=1:
            raise ValueError(f"component spans disease labels; resolve before freezing: {members}")
    links=list(image_links)
    patient_groups:dict[tuple[str,str],list[str]]={}
    seen=set()
    for row in patients:
        path=safe_id(row["path"])
        if path not in labels or path in seen or not all(row.get(k,"").strip() for k in ("dataset","patient_id","evidence")):
            raise ValueError("verified patient rows need unique rare paths, dataset, patient_id and evidence")
        seen.add(path)
        patient_groups.setdefault((row["dataset"],row["patient_id"]),[]).append(path)
    for members in patient_groups.values():
        if len({labels[p] for p in members})!=1:
            raise ValueError("verified patient spans disease labels; selector requires a dated amendment")
        links.extend((members[0],p) for p in members[1:])
    excluded=fm.legacy_exclusions(splits,links)
    development={p:labels[p] for p in labels if splits[p]=="train"}
    groups=fm.build_groups(development,[(a,b) for a,b in links if a in development and b in development],excluded)
    return Grouping(labels,splits,image_links,excluded,groups)


def nine_way_roster(rows: Rows, eligible: list[str]) -> list[list[str]]:
    """Fixed seed-42 common fitting sample plus eligible rare fitting rows, then all queries."""
    from stage3_classifier import major_indices  # imported here: pulls pandas, not TensorFlow
    fitting = [r for r in rows if r["split"] == "train" and r["label"] in COMMON]
    chosen = [fitting[int(i)] for i in major_indices([r["label"] for r in fitting], COMMON_CAP, fm.SEED)]
    keep = set(eligible)
    rare = [r for r in rows if r["path"] in keep]
    tests = [r for r in rows if r["split"] == "test"]
    return [[r["path"], r["label"], "fitting"] for r in chosen + rare] + [[r["path"], r["label"], "query"] for r in tests]


def freeze(root: Path, out: Path, reviews: Path | None = None, patients: Path | None = None) -> dict[str, Any]:
    """Apply the screen and write every derived roster. Any failed gate stops before writing them."""
    out = out_dir(root, out)
    previous = json.loads((out / "screen_summary.json").read_text(encoding="utf-8"))
    if previous["manifest_sha256"] != text_hash(root / "data/manifest.csv"):
        raise ValueError("manifest changed since the screen")
    if previous.get("producer_code") != preparation_code() or previous.get("recipe") != preparation_recipe():
        raise ValueError("preparation code/recipe changed since the screen; use a fresh preparation folder")
    verify_files(out, previous["files"])
    if previous["versions"] != versions():
        raise ValueError("screen decoder/environment changed; use the pinned environment")
    inventory = read_csv(out / "image_inventory.csv")
    manifest = load_manifest(root)
    expected = sorted((r["path"], r["label"], r["split"]) for r in manifest if r["label"] in fm.RARE)
    if [(r["path"], r["label"], r["split"]) for r in inventory] != expected:
        raise ValueError("screen inventory differs from the rare manifest rows")
    for row in inventory:
        if sha256(root / row["path"]) != row["file_sha256"]:
            raise ValueError(f"image changed since screening: {row['path']}")
    pairs = read_csv(out / "screen_pairs.csv")
    decisions=[dict(a=p["a"],b=p["b"],decision="exact") for p in pairs if p["kind"]!="dhash"]
    decisions+=read_csv(reviews) if reviews else []
    patient_rows = [{**row, "path": safe_id(row["path"])} for row in read_csv(patients)] if patients else []
    derived=derive_grouping(inventory,pairs,decisions,patient_rows)
    labels,splits,image_links,excluded,groups=derived.labels,derived.splits,derived.image_links,derived.excluded,derived.groups
    counts = fm.check_feasible(groups)
    report_k = fm.report_k_values(groups)

    folds = {c: fm.assign_folds(groups[c], i) for i, c in enumerate(fm.RARE)}
    write_csv(out / "groups.csv", ["group_id", "label", "rank", "fold", "member"],
              [[g.id, c, g.rank, folds[c][g.id], m] for c in fm.RARE for g in groups[c] for m in g.members])
    write_csv(out / "exclusions.csv", ["path", "label", "reason", "component"],
              [[p, labels[p], "linked_to_legacy_query", excluded[p]] for p in sorted(excluded)])
    write_csv(out / "duplicate_decisions.csv", ["a", "b", "decision"],
              [[r[k] for k in ("a","b","decision")] for r in decisions])
    write_csv(out / "verified_patients.csv", ["path", "dataset", "patient_id", "evidence"],
              [[r[k] for k in ("path", "dataset", "patient_id", "evidence")] for r in patient_rows])
    write_csv(out / "components.csv", ["component", "member", "size", "has_legacy_query"],
              [[members[0], member, len(members), int(any(splits[p] == "test" for p in members))]
               for members in fm.components(labels, image_links) for member in members])
    write_csv(out / "fold_counts.csv", ["label", "fold", "query_groups", "query_images", "fitting_groups", "fitting_images"],
              [[c, f, sum(folds[c][g.id] == f for g in groups[c]),
                sum(len(g.members) for g in groups[c] if folds[c][g.id] == f),
                sum(folds[c][g.id] != f for g in groups[c]),
                sum(len(g.members) for g in groups[c] if folds[c][g.id] != f)]
               for c in fm.RARE for f in range(fm.N_FOLDS)])
    write_csv(out / "queries_development.csv", ["fold", "label", "path"],
              [[f, c, m] for f in range(fm.N_FOLDS) for c in fm.RARE
               for g in groups[c] if folds[c][g.id] == f for m in g.members])

    slots = [f"s{i:02d}" for i in range(1, max(fm.K_DEV) + 1)]
    write_csv(out / "episodes_development.csv", ["fold", "label", "draw", *slots],
              [[fold, c, draw, *fm.draw_supports([g for g in groups[c] if folds[c][g.id] != fold], fold, i, draw, fm.DEV_NS)[:len(slots)]]
               for fold in range(fm.N_FOLDS) for i, c in enumerate(fm.RARE) for draw in range(fm.N_DRAWS)])
    report_slots = [f"s{i:02d}" for i in range(1, max(report_k) + 1)]
    write_csv(out / "episodes_reporting.csv", ["label", "draw", *report_slots],
              [[c, draw, *fm.draw_supports(groups[c], 0, i, draw, fm.REPORT_NS)[:len(report_slots)]]
               for i, c in enumerate(fm.RARE) for draw in range(fm.N_DRAWS)])

    eligible = sorted(m for c in fm.RARE for g in groups[c] for m in g.members)
    roster = nine_way_roster(manifest, eligible)
    write_csv(out / "nine_way_roster.csv", ["path", "label", "role"], roster)
    needed = sorted({row[0] for row in roster})
    write_csv(out / "extraction_inventory.csv", ["path", "sha256"],
              [[p, hashlib.sha256((root / p).read_bytes()).hexdigest()] for p in needed])
    # Section 7: fixed in path order before any score; IDs repeat only if too few images exist.
    timing = [eligible[i % len(eligible)] for i in range(100)]
    write_csv(out / "timing_inventory.csv", ["path"], [[p] for p in timing])

    summary = dict(protocol="v1.0", manifest_sha256=previous["manifest_sha256"], groups_per_class=counts,
                   producer_code=preparation_code(),recipe=preparation_recipe(),
                   excluded_rows=len(excluded), report_k=list(report_k), extraction_images=len(needed),
                   fitting_rows=sum(r[2] == "fitting" for r in roster), query_rows=sum(r[2] == "query" for r in roster),
                   versions=versions(), files={name: sha256(out / name) for name in PREPARED_FILES})
    write_json(out / "freeze_summary.json", summary)
    return summary


def verify_preparation(folder: Path,root: Path | None=None) -> dict:
    """Validate the producer, exact endpoints and realized recipes before locking/piloting."""
    summary=read_json(folder / "freeze_summary.json")
    if set(summary["files"]) != set(PREPARED_FILES):
        raise ValueError("preparation hash map must cover every required artifact")
    verify_files(folder,summary["files"])
    if summary.get("producer_code") != preparation_code() or summary.get("recipe") != preparation_recipe():
        raise ValueError("stale preparation code/recipe; regenerate before scores")
    rows=read_csv(folder / "groups.csv")
    grouped:dict[tuple[str,str],list[dict[str,str]]]={}
    for row in rows:
        grouped.setdefault((row["label"],row["group_id"]),[]).append(row)
    groups={c:[] for c in fm.RARE}
    for c in fm.RARE:
        keys=sorted(group for label,group in grouped if label==c)
        for rank,gid in enumerate(keys):
            members=grouped[c,gid]
            paths=tuple(r["member"] for r in members)
            if paths != tuple(sorted(set(paths))) or not paths or gid != paths[0] or \
                    any(int(r["rank"]) != rank or r["fold"] != members[0]["fold"] for r in members):
                raise ValueError("group IDs/ranks/members are inconsistent")
            groups[c].append(fm.Group(gid,c,rank,paths))
    if len({r["member"] for r in rows}) != len(rows) or any(r["label"] not in fm.RARE for r in rows):
        raise ValueError("invalid grouped inventory")
    inventory=read_csv(folder / "image_inventory.csv")
    derived=derive_grouping(inventory,read_csv(folder / "screen_pairs.csv"),
                            read_csv(folder / "duplicate_decisions.csv"),read_csv(folder / "verified_patients.csv"))
    if groups != derived.groups:
        raise ValueError("saved groups contradict duplicate/patient evidence")
    expected_exclusions=[dict(path=p,label=derived.labels[p],reason="linked_to_legacy_query",component=derived.excluded[p])
                         for p in sorted(derived.excluded)]
    if read_csv(folder / "exclusions.csv") != expected_exclusions or summary["excluded_rows"] != len(derived.excluded):
        raise ValueError("saved exclusions contradict duplicate/patient evidence")
    expected_components=[dict(component=members[0],member=m,size=str(len(members)),
                        has_legacy_query=str(int(any(derived.splits[p]=="test" for p in members))))
                        for members in fm.components(derived.labels,derived.image_links) for m in members]
    if read_csv(folder / "components.csv") != expected_components:
        raise ValueError("saved image components contradict duplicate evidence")
    counts=fm.check_feasible(groups)
    ks=fm.report_k_values(groups)
    if summary["groups_per_class"] != counts or summary["report_k"] != list(ks):
        raise ValueError("preparation endpoints/group counts differ from protocol feasibility")
    folds={c:fm.assign_folds(groups[c],ci) for ci,c in enumerate(fm.RARE)}
    if any(int(r["fold"]) != folds[r["label"]][r["group_id"]] for r in rows):
        raise ValueError("saved folds differ from the fixed random stream")
    expected_counts=[dict(label=c,fold=str(f),query_groups=str(sum(folds[c][g.id]==f for g in groups[c])),
                     query_images=str(sum(len(g.members) for g in groups[c] if folds[c][g.id]==f)),
                     fitting_groups=str(sum(folds[c][g.id]!=f for g in groups[c])),
                     fitting_images=str(sum(len(g.members) for g in groups[c] if folds[c][g.id]!=f)))
                     for c in fm.RARE for f in range(fm.N_FOLDS)]
    if read_csv(folder / "fold_counts.csv") != expected_counts:
        raise ValueError("saved fold counts differ from grouped eligibility")
    expected_queries=[dict(fold=str(f),label=c,path=m) for f in range(fm.N_FOLDS) for c in fm.RARE
                      for g in groups[c] if folds[c][g.id]==f for m in g.members]
    if read_csv(folder / "queries_development.csv") != expected_queries:
        raise ValueError("saved held-out query IDs differ from grouped folds")
    for phase,namespace,k_values in (("development",fm.DEV_NS,fm.K_DEV),("reporting",fm.REPORT_NS,ks)):
        expected=[]
        for f in range(fm.N_FOLDS if phase=="development" else 1):
            for ci,c in enumerate(fm.RARE):
                fitting=[g for g in groups[c] if phase=="reporting" or folds[c][g.id] != f]
                for draw in range(fm.N_DRAWS):
                    selected=fm.draw_supports(fitting,f,ci,draw,namespace)[:max(k_values)]
                    expected.append({**({"fold":str(f)} if phase=="development" else {}),"label":c,"draw":str(draw),
                                     **{f"s{i:02d}":p for i,p in enumerate(selected,1)}})
        if read_csv(folder / f"episodes_{phase}.csv") != expected:
            raise ValueError(f"saved {phase} support IDs differ from the fixed random streams")
    eligible=sorted(r["member"] for r in rows)
    if read_csv(folder / "timing_inventory.csv") != [{"path":eligible[i%len(eligible)]} for i in range(100)]:
        raise ValueError("timing inventory differs from the fixed eligible path order")
    screen=read_json(folder / "screen_summary.json")
    if screen["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("screen/freeze manifest identity differs")
    roster=read_csv(folder / "nine_way_roster.csv")
    extraction=read_csv(folder / "extraction_inventory.csv")
    if len({r["path"] for r in roster}) != len(roster) or \
            [r["path"] for r in extraction] != sorted(r["path"] for r in roster) or \
            summary["extraction_images"] != len(extraction) or \
            summary["fitting_rows"] != sum(r["role"]=="fitting" for r in roster) or \
            summary["query_rows"] != sum(r["role"]=="query" for r in roster):
        raise ValueError("extraction/nine-way IDs or declared counts differ")
    if root is not None:
        if screen["versions"] != versions():
            raise ValueError("source verification needs the pinned screening decoder/environment")
        if text_hash(root / "data/manifest.csv") != summary["manifest_sha256"]:
            raise ValueError("source manifest changed after preparation")
        manifest=load_manifest(root)
        canonical=nine_way_roster(manifest,eligible)
        if roster != [dict(path=p,label=c,role=role) for p,c,role in canonical]:
            raise ValueError("nine-way roster differs from the exact canonical common cap/eligible rare rows")
        inventory=read_csv(folder / "image_inventory.csv")
        if [(r["path"],r["label"],r["split"]) for r in inventory] != sorted(
                (r["path"],r["label"],r["split"]) for r in manifest if r["label"] in fm.RARE):
            raise ValueError("rare screened rows differ from source manifest")
        for row in inventory:
            if sha256(root / row["path"]) != row["file_sha256"]:
                raise ValueError("rare image bytes changed after screening")
            with Image.open(root / row["path"]) as image:
                actual=(fm.rgb_fingerprint(image),f"{fm.dhash64(image):016x}",str(image.width),str(image.height),image.mode)
            if actual != tuple(row[key] for key in ("rgb_sha256","dhash","width","height","mode")):
                raise ValueError("rare fingerprints differ from original image bytes")
        for row in extraction:
            if sha256(root / safe_id(row["path"])) != row["sha256"]:
                raise ValueError("extraction input bytes changed after preparation")
    return summary


def make_archive(destination: Path, members: list[tuple[str, Path]], expected: dict[str, str] | None = None) -> str:
    """Deterministic ZIP with explicit members, never a recursive directory scrape."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in members:
            info = zipfile.ZipInfo(safe_id(name))
            info.compress_type = zipfile.ZIP_DEFLATED
            content = source.read_bytes()
            if expected is not None and hashlib.sha256(content).hexdigest() != expected[name]:
                raise ValueError(f"source changed while archiving: {name}")
            archive.writestr(info, content)
    digest = sha256(pending)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError("refusing to replace a different archive")
        pending.unlink()
    else:
        pending.replace(destination)
    return digest


def pack(root: Path, folder: Path, destination: Path) -> dict:
    """Stage only the exact extraction inventory, with original bytes and paths."""
    summary = read_json(folder / "freeze_summary.json")
    verify_files(folder, summary["files"])
    archive_destination(root,destination)
    out_dir(root, destination.parent)
    inventory = read_csv(folder / "extraction_inventory.csv")
    for row in inventory:
        if sha256(root / safe_id(row["path"])) != row["sha256"]:
            raise ValueError(f"staging input changed: {row['path']}")
    digest = make_archive(destination, [(r["path"], root / r["path"]) for r in inventory],
                          {r["path"]: r["sha256"] for r in inventory})
    result = dict(sha256=digest, images=len(inventory), inventory_sha256=sha256(folder / "extraction_inventory.csv"))
    write_json(destination.with_name(destination.name + ".json"), result)
    return result


def lock_inputs(folder: Path, gate_path: Path, archive: Path,root: Path=SCRIPT_ROOT) -> dict:
    """Final pre-score gate; called only after Lam's review, pilots and source decisions."""
    from fm_extract import check_spec, prepared, verify_operation
    archive_destination(root,archive)
    summary = verify_preparation(folder,root)
    if text_hash(SCRIPT_ROOT / PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen protocol hash differs")
    gate = read_json(gate_path)
    rows = gate["arms"]
    names = [row["name"] for row in rows]
    first = "BioMedCLIP" if "BioMedCLIP" in names else "RETFound"
    if len(names) != 4 or set(names) != {first, "DINOv2", "MIRAGE", "Inception"}:
        raise ValueError("gate must account for every fixed arm, available or unavailable")
    if first == "BioMedCLIP" and not gate.get("retfound_failure", "").strip():
        raise ValueError("BioMedCLIP substitution requires a pre-score RETFound access/loading/terms failure")
    external = gate["external_sources"]
    if set(external) != {"OCTID", "OCTDL"}:
        raise ValueError("both external source decisions must be recorded")
    for source in external.values():
        if source["status"] == "unavailable":
            if not source.get("reason", "").strip():
                raise ValueError("unavailable external source needs a reason")
        elif source["status"] == "planned":
            if any(not source.get(key, "").strip() for key in
                   ("release", "release_sha256", "terms_evidence", "eligibility", "label_inventory")):
                raise ValueError("external release/terms/eligibility/labels must be frozen before inference")
        else:
            raise ValueError("external status must be planned or unavailable")
    arms, pilot_files = {}, []
    timings = []
    for row in rows:
        name = row["name"]
        if row["status"] == "unavailable":
            if not row.get("reason", "").strip():
                raise ValueError("unavailable arms need an explicit reason")
            arms[name] = dict(status="unavailable", reason=row["reason"])
            continue
        if row["status"] != "available":
            raise ValueError("arm status must be available or unavailable")
        pilot = Path(row["pilot"])
        marker = verify_complete(pilot)
        if not {"identity.json", "timing.json", "load_report.json", "geometry.csv"} <= marker["files"].keys():
            raise ValueError("pilot completion hashes must cover identity, timing, load report and geometry")
        verify_operation(pilot,"pilot",read_csv(folder / "timing_inventory.csv"),name)
        identity = read_json(pilot / "identity.json")
        if identity["step"] != "pilot" or identity["spec"]["arm"] != name or \
                identity["code"] != code_identity() or identity["input_files"] != summary["files"] or \
                identity.get("preparation_sha256") != sha256(folder / "freeze_summary.json"):
            raise ValueError(f"pilot identity is stale or mismatched: {name}")
        check_spec(identity["spec"])
        timing = read_json(pilot / "timing.json")
        if timing["batch_size"] != 1 or timing["images"] != 100 or timing["warm_passes"] != 1 or \
                timing["timed_passes"] != 5 or len(timing["seconds_per_image"]) != 5:
            raise ValueError("pilot does not implement the fixed timing recipe")
        seconds = np.asarray(timing["seconds_per_image"], dtype=np.float64)
        if not np.isfinite(seconds).all() or (seconds <= 0).any() or \
                timing["median_seconds_per_image"] != float(np.median(seconds)):
            raise ValueError("invalid pilot timing")
        if not identity.get("timing_session"):
            raise ValueError("pilot needs a shared timing-session identity")
        timings.append((name, identity["timing_session"], identity["hardware"], identity["environment"],timing))
        arms[name] = dict(status="available", identity={k: identity[k] for k in ("spec", "code", "environment")},
                          timing=timing, timing_session=identity["timing_session"], hardware=identity["hardware"])
        for file in ("identity.json", "timing.json", "load_report.json", "geometry.csv", "complete.json"):
            pilot_files.append((f"pilots/{name}/{file}", pilot / file))
    if arms["Inception"]["status"] != "available":
        raise ValueError("the matched Inception reference must pass its gate")
    if any(t[1:3] != timings[0][1:3] for t in timings):
        raise ValueError("runtime/hardware changed; retime all arms in one sequential session")
    fm_environments = [t[3] for t in timings if t[0] != fm.REFERENCE]
    if any(env != fm_environments[0] for env in fm_environments):
        raise ValueError("FM timing environment changed; retime all candidates before using cost")
    windows=sorted(timings,key=lambda t:t[4]["operation_started_monotonic_ns"])
    if len({t[4]["process_id"] for t in windows}) != len(windows) or any(
            left[4]["operation_finished_monotonic_ns"] > right[4]["operation_started_monotonic_ns"]
            for left,right in zip(windows,windows[1:])):
        raise ValueError("timing pilots must use distinct sequential, non-overlapping processes")
    fm_order=[t[0] for t in windows if t[0] != fm.REFERENCE]
    if fm_order != [arm for arm in (first,"DINOv2","MIRAGE") if arms[arm]["status"] == "available"]:
        raise ValueError("FM timing order must be RETFound/fallback, DINOv2, MIRAGE")
    files = {**summary["files"], "freeze_summary.json": sha256(folder / "freeze_summary.json")}
    for name, source in pilot_files:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_once(target, source.read_bytes())
        files[name] = sha256(target)
    identity = dict(schema=1, protocol_sha256=PROTOCOL_SHA256, code=code_identity(), git_revision=git_revision(),
                    evaluation_environment=math_environment(),
                    files=files, arms=arms, external_sources=external, retfound_failure=gate.get("retfound_failure"),
                    endpoints=dict(primary="five-way macro recall", development_k=list(fm.K_DEV),
                                   reporting_k=summary["report_k"], draws=fm.N_DRAWS,
                                   secondary_probe="development K=4 first25; reporting first25/K",
                                   required_companion="matched nine-way source audit; separate NORMAL FRR"))
    # Preserve the first successful timestamp on an unchanged retry.
    if (folder / "input_lock.json").exists():
        previous = read_json(folder / "input_lock.json")
        identity["created_utc"] = previous["created_utc"]
    else:
        identity["created_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(folder / "input_lock.json", identity)
    files = {**files, "input_lock.json": sha256(folder / "input_lock.json")}
    digest = make_archive(archive, [(name, folder / name) for name in sorted(files)], files)
    receipt=dict(sha256=digest,files=files,
                 note="local archive; author must preserve and timestamp it before labelled scores")
    write_json(folder / "lock_archive.json",receipt)
    write_json(archive.with_name(archive.name+".json"),receipt)
    return identity


def load_lock(folder: Path) -> dict:
    """All scoring/extraction entry points require the same archived lock and current code."""
    identity = read_json(folder / "input_lock.json")
    if identity["schema"] != 1 or identity["protocol_sha256"] != PROTOCOL_SHA256 or \
            text_hash(SCRIPT_ROOT / PROTOCOL) != PROTOCOL_SHA256 or identity["code"] != code_identity():
        raise ValueError("input lock protocol/code identity differs")
    verify_files(folder, identity["files"])
    receipt = read_json(folder / "lock_archive.json")
    if receipt["files"] != {**identity["files"], "input_lock.json": sha256(folder / "input_lock.json")}:
        raise ValueError("lock archive receipt differs from the current input lock")
    return identity


def restore_lock(archive: Path,out: Path,expected_sha256: str,root: Path=SCRIPT_ROOT) -> dict:
    """Restore a pre-score ZIP alone; regenerate its receipt from externally verified bytes."""
    if sha256(archive)!=expected_sha256:
        raise ValueError("input-lock archive differs from the saved SHA256")
    out=writable_path(root,out)
    with zipfile.ZipFile(archive) as zipped:
        names=zipped.namelist()
        if len(set(names))!=len(names) or any(safe_id(name)!=name for name in names) or "input_lock.json" not in names:
            raise ValueError("invalid input-lock archive member list")
        payload={name:zipped.read(name) for name in names}
    identity=json.loads(payload["input_lock.json"].decode("utf-8"))
    if not isinstance(identity,dict) or identity.get("schema")!=1 or identity.get("protocol_sha256")!=PROTOCOL_SHA256 or \
            identity.get("code")!=code_identity():
        raise ValueError("restored input lock protocol/code identity differs")
    expected=identity["files"]
    if set(names)!={*expected,"input_lock.json"} or any(
            hashlib.sha256(payload[name]).hexdigest()!=digest for name,digest in expected.items()):
        raise ValueError("input-lock archive members differ from the locked hashes")
    for name,content in payload.items():
        path=writable_path(root,out / name)
        path.parent.mkdir(parents=True,exist_ok=True)
        write_once(path,content)
    files={**expected,"input_lock.json":hashlib.sha256(payload["input_lock.json"]).hexdigest()}
    write_json(out / "lock_archive.json",dict(sha256=expected_sha256,files=files,
               note="local archive; author must preserve and timestamp it before labelled scores"))
    return load_lock(out)


def export(folder: Path, destination: Path,root: Path=SCRIPT_ROOT) -> dict:
    archive_destination(root,destination)
    marker = verify_complete(folder)
    names = sorted({*marker["files"], "complete.json"})
    digest = make_archive(destination, [(name, folder / name) for name in names],
                          {**marker["files"], "complete.json": sha256(folder / "complete.json")})
    result = dict(sha256=digest, files=len(names), note="verify the copied bytes at the durable destination")
    write_json(destination.with_name(destination.name + ".json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("screen", "freeze", "pack", "lock", "restore", "export", "unlock", "verify-copy"))
    parser.add_argument("--root", type=Path, required=True, help="checkout containing data/raw")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, help="freeze only: duplicate review decisions")
    parser.add_argument("--patients", type=Path, help="freeze only: verified dataset,patient_id,path,evidence CSV")
    parser.add_argument("--archive", type=Path, help="pack/lock/export ZIP destination; restore/verify-copy ZIP source")
    parser.add_argument("--gate", type=Path, help="lock only: explicit arm and external-source decisions")
    parser.add_argument("--token", help="unlock only, after verifying the old writer stopped")
    parser.add_argument("--sha256", help="restore/verify-copy: expected archive SHA256")
    args = parser.parse_args()
    if args.step == "unlock":
        writable_path(args.root,args.out / "writer.lock")
        unlock(args.out, args.token)
        result = {"unlocked": str(args.out)}
    elif args.step == "verify-copy":
        if not args.archive or sha256(args.archive) != args.sha256:
            raise ValueError("copied archive does not match the expected SHA256")
        result = {"copied_bytes_verified": True}
    elif args.step=="restore":
        if args.archive is None or not args.sha256:
            parser.error("restore requires --archive and --sha256 from the preserved archive receipt")
        out=out_dir(args.root,args.out)
        with writer_lock(out):
            result=restore_lock(args.archive,out,args.sha256,args.root)
    else:
        if args.step in ("pack", "lock", "export") and args.archive is None:
            parser.error("this step requires --archive")
        if args.step == "lock" and args.gate is None:
            parser.error("lock requires --gate")
        out = out_dir(args.root, args.out)
        with writer_lock(out):
            if args.step == "screen":
                result = screen(args.root, out)
            elif args.step == "freeze":
                result = freeze(args.root, out, args.reviews, args.patients)
            elif args.step == "pack":
                result = pack(args.root, out, args.archive)
            elif args.step == "export":
                result = export(out, args.archive,args.root)
            else:
                out_dir(args.root, args.archive.parent)
                result = lock_inputs(out, args.gate, args.archive,args.root)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
