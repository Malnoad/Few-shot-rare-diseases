"""External-source audit for the frozen-encoder study (FM_PROTOCOL.md sections 4, 6, 8, 9).

candidates  builds the fixed candidate roster from the published OCTID file list and OCTDL labels.
screen      fingerprints candidates against every locked row and each other (section 4, step 5).
freeze      applies review decisions, removes linked queries and locks final query IDs.
report      scores the locked external queries for one encoder, after the S3 nomination is sealed.
summarize   writes source-separated FRR, present-class recall, output counts and paired contrasts.

Features come from `fm_extract.py extract --external FOLDER`, in the same runtime as the main
extraction: it reads no labels, and the environment must still equal the input lock.
The rare cohorts are the constants of amendment A1 (A2 N-3); no option adds or removes one.
summarize also writes the OCTID two-way CSC-versus-MH test with its size-only reference (A2 A-1).
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

import fm_core as fm
import fm_evaluate as ev
import fm_extract as extract
import fm_prepare as prep

EXTERNAL_LABELS = ("NORMAL", "CSC", "MH")  # the only external classes the protocol names
# Borealis dataset: folder written by acquisition, canonical label. CSR maps to CSC only with evidence.
OCTID_DATASETS = {"doi:10.5683/SP/WLW4ZT": ("NORMAL", "NORMAL"), "doi:10.5683/SP/MBMQGD": ("MH", "MH"),
                  "doi:10.5683/SP/FVLZIL": ("CSR", "CSC")}
# Amendment A1 (12 Nov 2026), fixed in code by A2 N-3. Evidence: FM_EXTERNAL_MAPPING.md.
A1_COHORTS = {
    ("OCTID", "CSC"): "A1: OCTID CSR published label; OCTID paper class Central Serous Chorioretinopathy (CSC); "
                      "MeSH D056833 entry term Central Serous Retinopathy",
    ("OCTID", "MH"): "A1: OCTID MH published label, OCTID's broad definition; separate cohort, never called full-thickness",
    ("OCTDL", "MH"): "A1: OCTDL disease=VID with subcategory=MH; OCTDL full-thickness MH definition",
}
TWO_WAY = ("CSC", "MH")  # A2 A-1: the OCTID two-way test, ties to CSC (fixed class order)
CANDIDATE_HEADER = ["source", "path", "label", "patient_id", "mapping_evidence", "md5"]
INVENTORY_HEADER = [*CANDIDATE_HEADER[:5], "file_sha256", "rgb_sha256", "dhash", "width", "height", "mode"]
LOCKED_HEADER = ["path", "role", "file_sha256", "rgb_sha256", "dhash"]
PAIR_HEADER = ["a", "b", "kind", "hamming", "block"]
SCREEN_FILES = ("external_inventory.csv", "external_locked_inventory.csv", "external_screen_pairs.csv")
FREEZE_FILES = ("external_review_decisions.csv", "external_queries.csv", "external_exclusions.csv",
                "external_extraction_inventory.csv")


def candidates(octid: prep.Rows, octdl: prep.Rows) -> list[list[str]]:
    """Fixed eligibility from published metadata and the A1 cohorts. OCTDL MH uses subcategory,
    never condition alone (LH and VMT also read "MH" there)."""
    mappings = A1_COHORTS
    rows = []
    for row in octid:
        if row["doi"] in OCTID_DATASETS and row["filename"].lower().endswith(".jpeg"):
            folder, label = OCTID_DATASETS[row["doi"]]
            rows.append(("OCTID", f"data/external/OCTID/{folder}/{row['filename']}", label, "", row["md5"]))
    for row in octdl:
        label = "NORMAL" if row["disease"] == "NO" else "MH" if (row["disease"], row["subcategory"]) == ("VID", "MH") else None
        if label:
            rows.append(("OCTDL", f"data/external/OCTDL/{row['disease']}/{row['file_name']}.jpg", label, row["patient_id"], ""))
    kept = [[source, path, label, patient, mappings.get((source, label), "").strip(), md5]
            for source, path, label, patient, md5 in rows
            if label == "NORMAL" or mappings.get((source, label), "").strip()]
    return sorted(kept, key=lambda r: (r[0], r[1]))


def check_candidates(rows: prep.Rows, lock: Mapping, locked: set[str]) -> None:
    """Stop before any image is read if a row falls outside the frozen source decisions."""
    planned = {name for name, source in lock["external_sources"].items() if source["status"] == "planned"}
    seen = set()
    for row in rows:
        path = row["path"]
        if prep.safe_id(path) != path or path in seen or path in locked:
            raise ValueError(f"external path must be canonical, unique and outside the locked roster: {path}")
        seen.add(path)
        if row["source"] not in planned:
            raise ValueError(f"source was not planned at the input lock: {row['source']}")
        if row["label"] not in EXTERNAL_LABELS:
            raise ValueError(f"label is not an eligible external class: {row['label']}")
        if row["label"] != "NORMAL" and row["mapping_evidence"] != A1_COHORTS.get((row["source"], row["label"])):
            raise ValueError(f"rare external rows must be an A1 cohort with its fixed evidence: {path}")
    if missing := planned - {r["source"] for r in rows if r["label"] == "NORMAL"}:
        raise ValueError(f"planned sources have no NORMAL rows: {sorted(missing)}")


def fingerprint(path: Path) -> list:
    """The same image screen as the rare benchmark (section 4)."""
    with Image.open(path) as image:
        rgb, dhash, size, mode = fm.rgb_fingerprint(image), fm.dhash64(image), image.size, image.mode
    return [prep.sha256(path), rgb, f"{dhash:016x}", size[0], size[1], mode]


def screen_pairs(external: prep.Rows, locked: prep.Rows) -> list[dict[str, str]]:
    """Exact and near-duplicate candidates: external against locked rows and each other."""
    pairs = []

    def compare(a: Mapping[str, str], b: Mapping[str, str], block: str) -> None:
        distance = fm.hamming(int(a["dhash"], 16), int(b["dhash"], 16))
        kind = "file" if a["file_sha256"] == b["file_sha256"] else "rgb" if a["rgb_sha256"] == b["rgb_sha256"] else \
            "dhash" if distance <= fm.HAMMING_MAX else None
        if kind:
            pairs.append(dict(a=a["path"], b=b["path"], kind=kind, hamming=str(distance), block=block))

    ordered = sorted(external, key=lambda r: r["path"])
    for row in ordered:
        for other in locked:
            compare(row, other, "external-fitting" if other["role"] == "fitting" else "external-legacy")
    for a, b in combinations(ordered, 2):
        compare(a, b, "external-external")
    return pairs


def screen(root: Path, lock_folder: Path, candidate_csv: Path, out: Path) -> dict:
    """Fingerprint candidates and list every pair needing a decision; removes nothing itself."""
    out = prep.out_dir(root, out)
    lock = prep.load_lock(lock_folder)
    roster = prep.read_csv(lock_folder / "nine_way_roster.csv")
    content = {r["path"]: r["sha256"] for r in prep.read_csv(lock_folder / "extraction_inventory.csv")}
    rows = prep.read_csv(candidate_csv)
    if not rows or list(rows[0]) != CANDIDATE_HEADER:
        raise ValueError("external candidates must be nonempty with the fixed columns")
    check_candidates(rows, lock, {r["path"] for r in roster})
    inventory = []
    for row in rows:
        path = root / row["path"]
        if row["md5"] and hashlib.md5(path.read_bytes()).hexdigest() != row["md5"]:
            raise ValueError(f"external image differs from its published MD5: {row['path']}")
        inventory.append([*(row[key] for key in CANDIDATE_HEADER[:5]), *fingerprint(path)])
    locked = []
    for row in roster:
        values = fingerprint(root / row["path"])
        if values[0] != content[row["path"]]:
            raise ValueError(f"locked image changed: {row['path']}")
        locked.append([row["path"], row["role"], *values[:3]])
    prep.write_csv(out / "external_inventory.csv", INVENTORY_HEADER, inventory)
    prep.write_csv(out / "external_locked_inventory.csv", LOCKED_HEADER, locked)
    pairs = screen_pairs(prep.read_csv(out / "external_inventory.csv"), prep.read_csv(out / "external_locked_inventory.csv"))
    prep.write_csv(out / "external_screen_pairs.csv", PAIR_HEADER, [[p[key] for key in PAIR_HEADER] for p in pairs])
    blocks = Counter((p["block"], p["kind"]) for p in pairs)
    summary = dict(protocol_sha256=prep.PROTOCOL_SHA256, input_lock_sha256=prep.sha256(lock_folder / "input_lock.json"),
                   code=prep.code_identity(), versions=prep.versions(), hamming_limit=fm.HAMMING_MAX, candidates=len(inventory),
                   pairs_by_block={block: {kind: blocks[block, kind] for kind in ("file", "rgb", "dhash")}
                                   for block in ("external-fitting", "external-legacy", "external-external")},
                   files={name: prep.sha256(out / name) for name in SCREEN_FILES})
    prep.write_json(out / "external_screen_summary.json", summary)
    return summary


def derive(external: prep.Rows, locked: prep.Rows, pairs: prep.Rows, decisions: prep.Rows) -> tuple[list, list]:
    """Single derivation used by freeze and verification; a ledger can never overrule its evidence."""
    if pairs != screen_pairs(external, locked):
        raise ValueError("saved external screen pairs differ from the fingerprint inventories")
    links = prep.reviewed_links(pairs, decisions)
    by_path = {r["path"]: r for r in external}
    roles = {r["path"]: r["role"] for r in locked}
    if len(by_path) != len(external) or by_path.keys() & roles.keys():
        raise ValueError("external rows must be unique and disjoint from the locked roster")
    removed: dict[str, list[str]] = {}
    for members in fm.components([*by_path, *roles], links):
        inside = [m for m in members if m in by_path]
        touched = sorted({roles[m] for m in members if m in roles})
        if inside and touched:
            # The locked roster never changes: only the external side of the link is removed.
            reason = "linked_to_" + "_and_".join("legacy_query" if role == "query" else role for role in touched)
            removed.update({m: [reason, members[0]] for m in inside})
        elif inside and len({by_path[m]["label"] for m in inside}) != 1:
            raise ValueError(f"possible duplicates carry conflicting labels; resolve before inference: {inside}")
    keep = [r for r in external if r["path"] not in removed]
    retained = {r["path"] for r in keep}
    dependent = [(a, b) for a, b in links if a in retained and b in retained]
    patients: dict[tuple[str, str], list[str]] = {}
    for row in keep:
        if row["patient_id"].strip():  # IDs are namespaced by source; a shared patient is not a conflict
            patients.setdefault((row["source"], row["patient_id"]), []).append(row["path"])
    dependent += [(members[0], other) for members in patients.values() for other in members[1:]]
    group = {m: members[0] for members in fm.components(retained, dependent) for m in members}
    queries = sorted(([r["source"], r["path"], r["label"], r["file_sha256"], group[r["path"]]] for r in keep),
                     key=lambda r: (r[0], r[1]))
    exclusions = sorted(([by_path[p]["source"], p, by_path[p]["label"], *why] for p, why in removed.items()),
                        key=lambda r: (r[0], r[1]))
    return queries, exclusions


def freeze(root: Path, lock_folder: Path, out: Path, reviews: Path | None = None) -> dict:
    """Lock the final external query IDs. Locked fitting rosters are never changed."""
    out = prep.out_dir(root, out)
    prep.load_lock(lock_folder)
    screened = prep.read_json(out / "external_screen_summary.json")
    prep.verify_files(out, screened["files"])
    if screened["input_lock_sha256"] != prep.sha256(lock_folder / "input_lock.json") or screened["code"] != prep.code_identity():
        raise ValueError("input lock or code changed since the external screen")
    if screened["versions"] != prep.versions():
        raise ValueError("external screen decoder/environment changed; use the pinned environment")
    decisions = prep.read_csv(reviews) if reviews else []
    external = prep.read_csv(out / "external_inventory.csv")
    queries, exclusions = derive(external, prep.read_csv(out / "external_locked_inventory.csv"),
                                 prep.read_csv(out / "external_screen_pairs.csv"), decisions)
    prep.write_csv(out / "external_review_decisions.csv", ["a", "b", "decision"], [[r[k] for k in ("a", "b", "decision")] for r in decisions])
    prep.write_csv(out / "external_queries.csv", ["source", "path", "label", "sha256", "dependence_group"], queries)
    prep.write_csv(out / "external_exclusions.csv", ["source", "path", "label", "reason", "component"], exclusions)
    prep.write_csv(out / "external_extraction_inventory.csv", ["path", "sha256"], sorted([q[1], q[3]] for q in queries))
    per_source = {}
    for source in sorted({r["source"] for r in external}):
        chosen = [q for q in queries if q[0] == source]
        per_source[source] = dict(candidates=sum(r["source"] == source for r in external), queries=len(chosen),
                                  excluded=sum(e[0] == source for e in exclusions), dependence_groups=len({q[4] for q in chosen}),
                                  labels=dict(sorted(Counter(q[2] for q in chosen).items())))
    if any(value["labels"].get("NORMAL", 0) == 0 for value in per_source.values()):
        raise ValueError("a source has no NORMAL query after screening; report its source component as not run")
    identity = dict(schema=1, protocol_sha256=prep.PROTOCOL_SHA256, input_lock_sha256=screened["input_lock_sha256"],
                    code=prep.code_identity(), sources=per_source,
                    files={name: prep.sha256(out / name) for name in (*SCREEN_FILES, "external_screen_summary.json", *FREEZE_FILES)})
    previous = out / "external_lock.json"
    identity["created_utc"] = prep.read_json(previous)["created_utc"] if previous.exists() else datetime.now(timezone.utc).isoformat()
    prep.write_json(previous, identity)
    prep.complete(out, [*identity["files"], "external_lock.json"])
    return identity


def verify_external(folder: Path, lock_folder: Path) -> dict:
    """Every extraction and report rechecks the query lock against its own evidence."""
    prep.verify_complete(folder)
    identity = prep.read_json(folder / "external_lock.json")
    prep.verify_files(folder, identity["files"])
    if identity["schema"] != 1 or identity["protocol_sha256"] != prep.PROTOCOL_SHA256 or identity["code"] != prep.code_identity() or \
            identity["input_lock_sha256"] != prep.sha256(lock_folder / "input_lock.json"):
        raise ValueError("external query lock belongs to a different input lock/code")
    queries, exclusions = derive(prep.read_csv(folder / "external_inventory.csv"), prep.read_csv(folder / "external_locked_inventory.csv"),
                                 prep.read_csv(folder / "external_screen_pairs.csv"), prep.read_csv(folder / "external_review_decisions.csv"))
    header = (["source", "path", "label", "sha256", "dependence_group"], ["source", "path", "label", "reason", "component"])
    if prep.read_csv(folder / "external_queries.csv") != [dict(zip(header[0], q)) for q in queries] or \
            prep.read_csv(folder / "external_exclusions.csv") != [dict(zip(header[1], e)) for e in exclusions] or \
            prep.read_csv(folder / "external_extraction_inventory.csv") != [dict(path=q[1], sha256=q[3]) for q in sorted(queries, key=lambda q: q[1])]:
        raise ValueError("saved external ledgers contradict the screen evidence")
    return identity


def external_features(lock_folder: Path, cache: Path, external: Path, arm: str) -> ev.Features:
    """Open an external bundle only if extracted under this exact input and query lock."""
    lock = prep.load_lock(lock_folder)
    if lock["arms"].get(arm, {}).get("status") != "available":
        raise ValueError("arm unavailable at input lock")
    marker = prep.verify_complete(cache)
    if not {"features.npz", "identity.json", "geometry.csv", "load_report.json", "costs.json"} <= marker["files"].keys():
        raise ValueError("incomplete external feature bundle")
    identity = prep.read_json(cache / "identity.json")
    if identity["step"] != "extract" or identity["lock_sha256"] != prep.sha256(lock_folder / "input_lock.json") or \
            identity.get("external_lock_sha256") != prep.sha256(external / "external_lock.json") or \
            {k: identity[k] for k in ("spec", "code", "environment")} != lock["arms"][arm]["identity"]:
        raise ValueError("external feature bundle differs from the input/query locks")
    inventory = prep.read_csv(external / "external_extraction_inventory.csv")
    extract.verify_operation(cache, "extract", inventory, arm)
    values = extract.load_features(cache / "features.npz", inventory, extract.DIMENSIONS[arm])
    return ev.Features(values, {r["path"]: i for i, r in enumerate(inventory)})


def merge(main: ev.Features, other: ev.Features) -> ev.Features:
    """One lookup across fitting rows and external queries, which must stay disjoint."""
    if main.index.keys() & other.index.keys():
        raise ValueError("external queries overlap the locked feature inventory")
    offset = len(main.values)
    return ev.Features(np.concatenate((main.values, other.values)), {**main.index, **{p: i + offset for p, i in other.index.items()}})


def source_audit(lock_folder: Path, features: ev.Features, source: str, rows: prep.Rows, out: Path, identity_hash: str) -> None:
    """Section 8: the deterministic nine-way head on one source. FRR never uses a five-way head."""
    out.mkdir(parents=True, exist_ok=True)
    fitting = [r for r in prep.read_csv(lock_folder / "nine_way_roster.csv") if r["role"] == "fitting"]
    if {r["label"] for r in fitting} != set(fm.NINE):
        raise ValueError("nine-way roster needs fitting rows for every class")
    prep.write_json(out / "head.json", dict(run_identity_sha256=identity_hash, source=source, fitting=fitting, queries=rows, classes=list(fm.NINE)))
    if (out / "complete.json").exists():
        ev.verify_nine_way(out, identity_hash)
        return
    start = time.perf_counter()
    prototypes = fm.centroids([features.select([r["path"] for r in fitting if r["label"] == c]) for c in fm.NINE])
    paths, labels = [r["path"] for r in rows], [r["label"] for r in rows]
    scores = fm.unit(features.select(paths)) @ prototypes.T
    classifier_seconds = time.perf_counter() - start
    start = time.perf_counter()
    metrics = ev.write_predictions(out, paths, labels, scores, fm.NINE, "cosine")
    false_rare, normals = fm.false_rare_rate(labels, [fm.NINE[i] for i in scores.argmax(axis=1)])
    counts = {c: dict(correct=metrics["correct"][i], total=metrics["totals"][i]) for i, c in enumerate(fm.NINE) if metrics["totals"][i]}
    # A2 B-3: OCTID has no patient IDs (exact, image-level); OCTDL clusters by dependence group.
    clusters = None if source == "OCTID" else [r["dependence_group"] for r in rows]
    prep.write_json(out / "source_audit.json", dict(
        source=source, false_rare_n=false_rare, normal_n=normals, false_rare_rate=false_rare / normals if normals else None,
        class_counts=counts, dependence_groups=len({r["dependence_group"] for r in rows}),
        intervals=ev.nine_way_intervals(labels, [fm.NINE[i] for i in scores.argmax(axis=1)], clusters),
        limitation="descriptive source comparison; no causal source-shift, pretraining or patient-independence claim"))
    if not (out / "costs.json").exists():
        prep.write_json(out / "costs.json", dict(classifier_seconds=classifier_seconds, serialization_seconds=time.perf_counter() - start,
                                                 classifier_note="normalization, deterministic head fitting and query prediction"))
    ev.validate_costs(out / "costs.json")
    prep.complete(out, ["head.json", "predictions.csv", "metrics.json", "source_audit.json", "costs.json"])


def transfer_episodes(planned: Sequence[ev.Episode], rows: prep.Rows) -> list[ev.Episode]:
    """The fixed reporting supports with one source's rare queries; partial classes get no macro."""
    rare = [r for r in rows if r["label"] in fm.RARE]
    if not rare:
        return []
    return [replace(e, queries=tuple(r["path"] for r in rare), labels=tuple(r["label"] for r in rare)) for e in planned]


def run_report(lock_folder: Path, cache: Path, external_cache: Path, external: Path, arm: str, nomination: Path, out: Path,
               omission_reason: str = "") -> None:
    """One encoder's external outputs. The nominee is sealed before any external label is read."""
    out = prep.experiment_out(out)
    lock = prep.load_lock(lock_folder)
    ev.nomination_decision(lock_folder, nomination)
    if lock["evaluation_environment"] != prep.math_environment():
        raise ValueError("CPU Python/NumPy/SciPy differs from the pre-score input lock")
    verify_external(external, lock_folder)
    planned = ev.episodes(lock_folder, "reporting")
    rows = prep.read_csv(external / "external_queries.csv")
    features = merge(ev.open_features(lock_folder, cache, arm), external_features(lock_folder, external_cache, external, arm))
    identity = dict(lock_sha256=prep.sha256(lock_folder / "input_lock.json"), external_lock_sha256=prep.sha256(external / "external_lock.json"),
                    arm=arm, phase="external", code=prep.code_identity(), environment=extract.environment(), omit_logistic_reason=omission_reason,
                    cache_complete_sha256=prep.sha256(cache / "complete.json"),
                    external_cache_complete_sha256=prep.sha256(external_cache / "complete.json"),
                    nomination_sha256=prep.sha256(nomination / "nomination.json"))
    with prep.writer_lock(out):
        prep.write_json(out / "identity.json", identity)
        identity_hash = prep.sha256(out / "identity.json")
        if (out / "complete.json").exists():
            marker = prep.verify_complete(out)
            if not {"identity.json", "summary.json"} <= marker["files"].keys():
                raise ValueError("external completion marker omits identity/summary")
            return
        summary = {}
        started = time.perf_counter()
        for source in sorted({r["source"] for r in rows}):
            subset = [r for r in rows if r["source"] == source]
            source_audit(lock_folder, features, source, subset, out / source / "nine_way", identity_hash)
            transfers = transfer_episodes(planned, subset)
            counts = dict(prototype=[0, 0], logistic=[0, 0])  # complete, failed
            for episode in transfers:
                for classifier in ("prototype", "logistic"):
                    if classifier == "logistic" and (omission_reason or not ev.planned_probe(episode, "reporting")):
                        continue
                    path = out / source / classifier / episode.name
                    ev.evaluate_episode(features, episode, classifier, path, identity_hash)
                    counts[classifier][ev.read_completed_episode(path)[1]["state"] != "complete"] += 1
            summary[source] = dict(rows=len(subset), rare_queries=sum(r["label"] in fm.RARE for r in subset), episodes=counts)
        prep.write_json(out / "summary.json", dict(sources=summary, omit_logistic_reason=omission_reason or None))
        if not (out / "costs.json").exists():
            prep.write_json(out / "costs.json", dict(classifiers_and_serialization_seconds=time.perf_counter() - started, dtype="float64"))
        artifacts = ["identity.json", "summary.json", "costs.json"]
        for marker in sorted(out.rglob("complete.json")):
            if marker.parent != out:
                child = prep.verify_complete(marker.parent)
                artifacts.extend((marker.parent / name).relative_to(out).as_posix() for name in [*child["files"], "complete.json"])
        prep.complete(out, sorted(artifacts))


def read_transfer(run: Path, source: str, classifier: str, planned: Sequence[ev.Episode], root_hash: str) -> dict:
    """Present-class recall and output counts per completed K/draw; failures stay missing, not imputed."""
    cells = {}
    for episode in planned:
        path = run / source / classifier / episode.name
        if not (path / "complete.json").exists():
            continue
        record, fit = ev.read_completed_episode(path)
        if record["run_identity_sha256"] != root_hash or record["classifier"] != classifier or \
                record["supports"] != [list(p) for p in episode.supports] or \
                record["queries"] != list(episode.queries) or record["labels"] != list(episode.labels):
            raise ValueError("external episode identity differs from the locked supports/queries")
        if fit["state"] != "complete":
            continue
        metric = prep.read_json(path / "metrics.json")
        if metric["classes"] != list(fm.RARE) or metric["macro_recall"] is not None and min(metric["totals"]) == 0:
            raise ValueError("partial-class external cohort must not carry a macro recall")
        cells[episode.k, episode.draw] = {c: (metric["correct"][i] / metric["totals"][i], metric["confusion"][i])
                                          for i, c in enumerate(fm.RARE) if metric["totals"][i]}
    return cells


def kermany_rows(run: Path, arm: str, expected: Mapping[str, str]) -> list[list]:
    """Section 8 reports Kermany NORMAL beside each external source, from the sealed legacy run."""
    if not (run / "complete.json").exists():
        return []
    prep.verify_complete(run)
    identity = prep.read_json(run / "identity.json")
    if identity["arm"] != arm or identity["phase"] != "reporting" or identity["lock_sha256"] != expected["lock_sha256"] or \
            identity["nomination_sha256"] != expected["nomination_sha256"] or identity["code"] != expected["code"]:
        raise ValueError(f"legacy run belongs to a different lock/nomination/code: {arm}")
    result = prep.read_json(run / "nine_way" / "source_audit.json")
    metric = prep.read_json(run / "nine_way" / "metrics.json")
    disjoint, method = result["patient_disjoint"], result["intervals"]["method"]
    return [[arm, "Kermany", "false_rare_rate_patient_disjoint", disjoint["false_rare_n"], disjoint["normal_n"],
             disjoint["false_rare_rate"], *(disjoint["interval"] or [None, None]), method],
            *audit_rows(arm, "Kermany", result, metric, "false_rare_rate_all")]


def audit_rows(arm: str, source: str, result: dict, metric: dict, frr_name: str = "false_rare_rate") -> list[list]:
    """Nine-way FRR and per-class recall rows with their A2 B-3 intervals and method."""
    intervals = result["intervals"]
    rows = [[arm, source, frr_name, result["false_rare_n"], result["normal_n"], result["false_rare_rate"],
             *(intervals["false_rare"] or [None, None]), intervals["method"]]]
    for i, c in enumerate(metric["classes"]):
        if metric["totals"][i]:
            rows.append([arm, source, f"recall_{c}", metric["correct"][i], metric["totals"][i],
                         metric["correct"][i] / metric["totals"][i], *intervals["recall"][c], intervals["method"]])
    return rows


def two_way_counts(path: Path) -> np.ndarray:
    """A2 A-1: CSC/MH confusion from the five-way CSC and MH cosines of one episode (ties to CSC).

    Each centroid uses only its own supports, so this equals refitting with CSC and MH only.
    """
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in prep.read_csv(path / "predictions.csv"):
        if row["true_label"] in TWO_WAY:
            chosen = 0 if float(row["cosine_CSC"]) >= float(row["cosine_MH"]) else 1
            matrix[TWO_WAY.index(row["true_label"]), chosen] += 1
    return matrix


def size_two_way_counts(episode: ev.Episode, sizes: Mapping[str, tuple[int, int]]) -> np.ndarray:
    """The size-only reference (A2 N-1) on the same CSC/MH supports and OCTID queries."""
    hits = ev.size_hits(episode, dict(sizes), TWO_WAY)
    totals = np.array([episode.labels.count(c) for c in TWO_WAY])
    return np.array([[hits[0], totals[0] - hits[0]], [totals[1] - hits[1], hits[1]]], dtype=np.int64)


def two_way_rows(arm: str, role: str, matrices: Mapping[tuple[int, int], np.ndarray], ks: Sequence[int]) -> list[list]:
    """Per K: draws, denominators, macro recall summary, mean per-class recall and confusion, collapses."""
    macro = {key: float(np.mean(m.diagonal() / m.sum(axis=1))) for key, m in matrices.items()}
    rows = []
    for k in (*ks, ev.K_MEAN_LABEL):
        if k == ev.K_MEAN_LABEL:
            values, cells = list(ev.k_mean(macro, fm.N_DRAWS).values()), []
        else:
            cells = [matrices[k, r] for r in range(fm.N_DRAWS) if (k, r) in matrices]
            values = [macro[k, r] for r in range(fm.N_DRAWS) if (k, r) in matrices]
        if cells:
            mean = np.mean(cells, axis=0)
            extra = [int(cells[0][0].sum()), int(cells[0][1].sum()), *ev.draw_summary(values),
                     float(np.mean([m[0, 0] / m[0].sum() for m in cells])), float(np.mean([m[1, 1] / m[1].sum() for m in cells])),
                     *mean.ravel().tolist(), sum(int(m[:, 0].sum() == 0 or m[:, 1].sum() == 0) for m in cells)]
        else:
            extra = [None, None, *ev.draw_summary(values), *[None] * 7]
        rows.append([arm, role, k, len(values), fm.N_DRAWS, *extra])
    return rows


def summarize(lock_folder: Path, external: Path, runs: Path, nomination: Path, out: Path) -> None:
    """Source-separated tables. Partial-class cohorts never receive a five-class aggregate."""
    out = prep.experiment_out(out)
    lock = prep.load_lock(lock_folder)
    if lock["evaluation_environment"] != prep.math_environment():
        raise ValueError("summary arithmetic environment differs from the input lock")
    decision = ev.nomination_decision(lock_folder, nomination)
    verify_external(external, lock_folder)
    planned = ev.episodes(lock_folder, "reporting")
    ks = sorted({e.k for e in planned})
    queries = prep.read_csv(external / "external_queries.csv")
    sources = sorted({r["source"] for r in queries})
    order = sorted((a for a, s in lock["arms"].items() if s["status"] == "available" and (runs / a / "external" / "complete.json").exists()),
                   key=lambda arm: 3 if arm == fm.REFERENCE else fm.TIE_SLOT[arm])
    if not order:
        raise ValueError(f"no completed external runs under {runs}; check the runs path")
    expected = dict(lock_sha256=prep.sha256(lock_folder / "input_lock.json"), external_lock_sha256=prep.sha256(external / "external_lock.json"),
                    nomination_sha256=prep.sha256(nomination / "nomination.json"), code=prep.code_identity())
    audit, draws, outputs, contrasts, values = [], [], [], [], {}
    for arm in order:
        run = runs / arm / "external"
        prep.verify_complete(run)
        identity = prep.read_json(run / "identity.json")
        if identity["arm"] != arm or {k: identity[k] for k in expected} != expected:
            raise ValueError(f"external run belongs to a different lock/nomination/code: {arm}")
        audit.extend(kermany_rows(runs / arm / "reporting", arm, expected))
        root_hash = prep.sha256(run / "identity.json")
        for source in sources:
            result = prep.read_json(run / source / "nine_way" / "source_audit.json")
            audit.extend(audit_rows(arm, source, result, prep.read_json(run / source / "nine_way" / "metrics.json")))
            transfers = transfer_episodes(planned, [r for r in queries if r["source"] == source])
            for classifier, n in (("prototype", fm.N_DRAWS), ("logistic", 25)):
                cells = read_transfer(run, source, classifier, transfers, root_hash)
                values[arm, source, classifier] = cells
                for k in ks:
                    present = sorted({c for (kk, _), row in cells.items() if kk == k for c in row}, key=fm.RARE.index)
                    for label in present:
                        recalls = [cells[k, r][label][0] for r in range(n) if (k, r) in cells]
                        draws.append([arm, source, classifier, k, label, len(recalls), n, *ev.draw_summary(recalls)])
                        confusion = np.mean([cells[k, r][label][1] for r in range(n) if (k, r) in cells], axis=0)
                        outputs.extend([arm, source, classifier, k, label, c, float(confusion[i])] for i, c in enumerate(fm.RARE))
    for arm_a, arm_b in combinations(order, 2):
        for source in sources:
            for classifier, n in (("prototype", fm.N_DRAWS), ("logistic", 25)):
                left, right = values[arm_a, source, classifier], values[arm_b, source, classifier]
                for k in ks:
                    labels = sorted({c for (kk, _), row in left.items() if kk == k for c in row}, key=fm.RARE.index)
                    for label in labels:
                        differences = [left[k, r][label][0] - right[k, r][label][0] for r in range(n) if (k, r) in left and (k, r) in right]
                        primary = classifier == "prototype" and arm_a == decision["nominee"] and arm_b == fm.REFERENCE
                        contrasts.append([arm_a, arm_b, "primary" if primary else "secondary", source, classifier, k, label,
                                          len(differences), n, *ev.draw_summary(differences)])
    two_way = []
    octid = [r for r in queries if r["source"] == "OCTID"]
    if set(TWO_WAY) <= {r["label"] for r in octid}:
        transfers = transfer_episodes(planned, octid)
        for arm in order:
            done = values[arm, "OCTID", "prototype"]
            matrices = {(e.k, e.draw): two_way_counts(runs / arm / "external" / "OCTID" / "prototype" / e.name)
                        for e in transfers if (e.k, e.draw) in done}
            two_way.extend(two_way_rows(arm, "headline" if arm == decision["nominee"] else "secondary", matrices, ks))
        sizes = {**ev.image_sizes(lock_folder), **{r["path"]: (int(r["width"]), int(r["height"]))
                                                   for r in prep.read_csv(external / "external_inventory.csv")}}
        two_way.extend(two_way_rows("size_only_reference", "size_only_reference",
                                    {(e.k, e.draw): size_two_way_counts(e, sizes) for e in transfers}, ks))
    with prep.writer_lock(out):
        prep.write_csv(out / "external_source_audit.csv", ["arm", "source", "endpoint", "numerator", "denominator", "value",
                                                           "interval_low", "interval_high", "interval_method"], audit)
        prep.write_csv(out / "external_two_way.csv", ["arm", "role", "k", "available", "planned", "n_CSC", "n_MH",
                                                      "macro_mean", "macro_sd", "macro_p05", "macro_p95", "recall_CSC_mean",
                                                      "recall_MH_mean", "CSC_as_CSC", "CSC_as_MH", "MH_as_CSC", "MH_as_MH",
                                                      "collapsed_draws"], two_way)
        prep.write_csv(out / "external_transfer_summary.csv", ["arm", "source", "classifier", "k", "class", "available", "planned",
                                                               "mean", "sd", "p05", "p95"], draws)
        prep.write_csv(out / "external_transfer_outputs.csv", ["arm", "source", "classifier", "k", "true_class", "predicted_class",
                                                               "mean_count"], outputs)
        prep.write_csv(out / "external_paired_differences.csv", ["arm_a", "arm_b", "role", "source", "classifier", "k", "class", "available_pairs",
                                                                 "planned_pairs", "mean_a_minus_b", "sd", "p05", "p95"], contrasts)
        prep.write_json(out / "interpretation.json", dict(
            **expected, nominee=decision["nominee"], reporting_scope=decision["reporting_scope"],
            units="recall fractions (0..1); recall and FRR per source",
            incomplete_arms=sorted(a for a, s in lock["arms"].items() if s["status"] == "available" and a not in order),
            screened="locked rows and external candidates only; unused legacy images and private pretraining sets remain unknown",
            partial_classes="present-class recall and all five output counts only; no five-class aggregate",
            two_way="OCTID CSC vs MH from the five-way CSC/MH cosines, prototypes only; chance 0.50; the nominee row is "
                    "the headline, none if no nominee; size_only_reference is a diagnostic, never an arm (A2 A-1, N-1)",
            kermany="false_rare_rate_patient_disjoint is the headline Kermany FRR; false_rare_rate_all is secondary (A2 A-5)",
            variability="conditional on these images and support draws; not new-patient confidence intervals",
            limitation="descriptive source comparison; no causal source-shift, pretraining or patient-independence claim"))
        prep.complete(out, ["external_source_audit.csv", "external_transfer_summary.csv", "external_transfer_outputs.csv",
                            "external_paired_differences.csv", "external_two_way.csv", "interpretation.json"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    build = sub.add_parser("candidates")
    build.add_argument("--octid-files", type=Path, required=True)
    build.add_argument("--octdl-labels", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    first = sub.add_parser("screen")
    second = sub.add_parser("freeze")
    for command in (first, second):
        for name in ("root", "lock", "out"):
            command.add_argument("--" + name, type=Path, required=True)
    first.add_argument("--candidates", type=Path, required=True)
    second.add_argument("--reviews", type=Path, help="a,b,decision for every flagged dHash pair")
    report = sub.add_parser("report")
    for name in ("lock", "cache", "external-cache", "external", "nomination", "out"):
        report.add_argument("--" + name, type=Path, required=True)
    report.add_argument("--arm", choices=extract.REPOSITORIES, required=True)
    report.add_argument("--omit-logistic-reason", default="", help="dated, disclosed reduction; never outcome-based")
    tables = sub.add_parser("summarize")
    for name in ("lock", "external", "runs", "nomination", "out"):
        tables.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.step == "candidates":
        rows = candidates(prep.read_csv(args.octid_files), prep.read_csv(args.octdl_labels))
        prep.write_csv(args.out, CANDIDATE_HEADER, rows)
        print(json.dumps(dict(Counter(f"{r[0]}/{r[2]}" for r in rows)), indent=2, sort_keys=True))
    elif args.step in ("screen", "freeze"):
        out = prep.out_dir(args.root, args.out)
        with prep.writer_lock(out):
            result = screen(args.root, args.lock, args.candidates, out) if args.step == "screen" else freeze(args.root, args.lock, out, args.reviews)
        print(json.dumps({k: v for k, v in result.items() if k not in ("files", "code")}, indent=2, sort_keys=True))
    elif args.step == "report":
        run_report(args.lock, args.cache, args.external_cache, args.external, args.arm, args.nomination, args.out, args.omit_logistic_reason)
    else:
        summarize(args.lock, args.external, args.runs, args.nomination, args.out)


if __name__ == "__main__":
    main()
