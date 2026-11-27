"""CPU evaluation of locked raw-FP32 caches (FM_PROTOCOL.md sections 6-9).

develop uses held-out development queries only. nominate scores complete prototype
arms and binds one decision beside input_lock.json; later calls only copy it.
report verifies that binding before opening legacy query features. summarize adds the
amendment A2 rows (K mean, size-only reference); geometry writes the score-free geometry
and padding table. These commands never load an encoder or historical cache.
No real scores are produced during code preparation: tests use synthetic arrays.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax

import fm_core as fm
import fm_prepare as prep
import fm_extract as extract


@dataclass(frozen=True)
class Episode:
    fold: int
    k: int
    draw: int
    supports: tuple[tuple[str, ...], ...]
    queries: tuple[str, ...]
    labels: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"fold{self.fold}_k{self.k:02d}_draw{self.draw:03d}"


@dataclass
class Features:
    values: np.ndarray
    index: dict[str, int]

    def select(self, paths: Sequence[str]) -> np.ndarray:
        try:
            return self.values[[self.index[p] for p in paths]]
        except KeyError as error:
            raise ValueError(f"query/support missing from the locked feature inventory: {error}") from error


def open_features(folder: Path, cache: Path, arm: str) -> Features:
    lock = prep.load_lock(folder)
    if lock["arms"].get(arm, {}).get("status") != "available":
        raise ValueError("arm unavailable at input lock")
    marker = prep.verify_complete(cache)
    if not {"features.npz", "identity.json", "geometry.csv", "load_report.json", "costs.json"} <= marker["files"].keys():
        raise ValueError("incomplete matched feature bundle")
    identity = prep.read_json(cache / "identity.json")
    if identity["step"] != "extract" or identity["lock_sha256"] != prep.sha256(folder / "input_lock.json") or \
            {k: identity[k] for k in ("spec", "code", "environment")} != lock["arms"][arm]["identity"] or \
            identity["input_files"] != prep.read_json(folder / "freeze_summary.json")["files"] or \
            identity.get("preparation_sha256") != prep.sha256(folder / "freeze_summary.json"):
        raise ValueError("feature bundle identity differs from the input lock")
    inventory = prep.read_csv(folder / "extraction_inventory.csv")
    extract.verify_operation(cache,"extract",inventory,arm)
    values = extract.load_features(cache / "features.npz", inventory, extract.DIMENSIONS[arm])
    return Features(values, {r["path"]: i for i, r in enumerate(inventory)})


def episodes(folder: Path, phase: str) -> list[Episode]:
    """Read realized IDs, validating that no support group crosses its query fold."""
    if phase not in ("development", "reporting"):
        raise ValueError("unknown episode phase")
    groups = prep.read_csv(folder / "groups.csv")
    by_path = {r["member"]: r for r in groups}
    if len(by_path) != len(groups):
        raise ValueError("duplicate grouped member path")
    for row in groups:
        if row["label"] not in fm.RARE or int(row["fold"]) not in range(fm.N_FOLDS):
            raise ValueError("invalid grouped class/fold")
    query_rows = prep.read_csv(folder / "queries_development.csv")
    expected = {(int(r["fold"]), r["label"], r["member"]) for r in groups}
    actual = [(int(r["fold"]), r["label"], r["path"]) for r in query_rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("development query roster differs from grouped held-out rows")
    roster = prep.read_csv(folder / f"episodes_{phase}.csv")
    ks = fm.K_DEV if phase == "development" else tuple(prep.read_json(folder / "freeze_summary.json")["report_k"])
    folds = range(fm.N_FOLDS) if phase == "development" else range(1)
    lookup = {}
    for row in roster:
        key = (int(row["fold"]) if phase == "development" else 0, row["label"], int(row["draw"]))
        if key in lookup:
            raise ValueError("duplicate episode class row")
        lookup[key] = row
    keys = {(f, c, r) for f in folds for c in fm.RARE for r in range(fm.N_DRAWS)}
    if set(lookup) != keys:
        raise ValueError("required fold/class/draw roster is incomplete")
    result = []
    legacy = [r for r in prep.read_csv(folder / "nine_way_roster.csv")
              if r["role"] == "query" and r["label"] in fm.RARE]
    for fold in folds:
        queries = [r for r in query_rows if int(r["fold"]) == fold] if phase == "development" else legacy
        if {r["label"] for r in queries} != set(fm.RARE):
            raise ValueError("every required episode must contain all five query classes")
        for draw in range(fm.N_DRAWS):
            supports = []
            for label in fm.RARE:
                row = lookup[fold, label, draw]
                paths = tuple(row[f"s{i:02d}"] for i in range(1, max(ks) + 1))
                if any(p not in by_path or by_path[p]["label"] != label or
                       (phase == "development" and int(by_path[p]["fold"]) == fold) for p in paths):
                    raise ValueError("support class/fold differs from the locked episode")
                if len({by_path[p]["group_id"] for p in paths}) != len(paths):
                    raise ValueError("support roster repeats a group")
                supports.append(paths)
            for k in ks:
                result.append(Episode(fold, k, draw, tuple(p[:k] for p in supports),
                                      tuple(r["path"] for r in queries), tuple(r["label"] for r in queries)))
    return result


def classification_metrics(true: Sequence[str], predicted: Sequence[str], classes: Sequence[str]) -> dict:
    """Full allocation counts; partial-class cohorts never receive a five-class macro."""
    if not true or len(true) != len(predicted) or any(x not in classes for x in (*true, *predicted)):
        raise ValueError("query/prediction labels are invalid")
    lookup = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, chosen in zip(true, predicted, strict=True):
        matrix[lookup[actual], lookup[chosen]] += 1
    totals, hits = matrix.sum(axis=1), matrix.diagonal()
    return dict(classes=list(classes), confusion=matrix.tolist(), correct=hits.tolist(), totals=totals.tolist(),
                n_queries=len(true), accuracy=float(hits.sum() / len(true)),
                per_class_recall={c: float(hits[i]/totals[i]) if totals[i] else None for i, c in enumerate(classes)},
                macro_recall=fm.macro_recall(hits, totals) if (totals > 0).all() else None)


def predict_episode(features: Features, episode: Episode, classifier: str) -> tuple[np.ndarray | None, dict]:
    """Scores are float64. Logistic failure returns diagnostics and no query scores."""
    supports = [features.select(paths) for paths in episode.supports]
    if classifier == "prototype":
        prototypes = fm.centroids(supports)
        scores = fm.unit(features.select(episode.queries)) @ prototypes.T
        return scores, dict(state="complete", score_type="cosine", dtype="float64")
    if classifier != "logistic":
        raise ValueError("unknown classifier")
    labels = np.repeat(np.arange(len(fm.RARE)), episode.k)
    fit = fm.fit_logistic(np.concatenate(supports), labels)
    diagnostics = dict(state="complete" if fit.converged else "fit_failed", score_type="logit", dtype="float64",
                       objective=fit.objective if np.isfinite(fit.objective) else None,
                       max_abs_gradient=fit.max_abs_gradient if np.isfinite(fit.max_abs_gradient) else None,
                       converged=fit.converged, termination_reason=fit.message)
    if not fit.converged:
        return None, diagnostics
    scores = fm.unit(features.select(episode.queries)) @ fit.W.T + fit.b
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite logistic query scores")
    return scores, diagnostics


def write_predictions(out: Path, queries: Sequence[str], labels: Sequence[str], scores: np.ndarray,
                      classes: Sequence[str], score_type: str) -> dict:
    if scores.shape != (len(queries), len(classes)) or not np.isfinite(scores).all():
        raise ValueError("invalid prediction score matrix")
    chosen = scores.argmax(axis=1)
    predicted = [classes[i] for i in chosen]
    header = ["query_id", "true_label", "predicted_label", "correct", *[f"{score_type}_{c}" for c in classes]]
    values = scores
    if score_type == "logit":
        header += [f"prob_{c}" for c in classes]
        values = np.concatenate((scores, softmax(scores, axis=1)), axis=1)
    rows = [[p, y, pred, int(y == pred), *s.tolist()]
            for p, y, pred, s in zip(queries, labels, predicted, values, strict=True)]
    prep.write_csv(out / "predictions.csv", header, rows)
    metrics = classification_metrics(labels, predicted, classes)
    prep.write_json(out / "metrics.json", metrics)
    return metrics


def evaluate_episode(features: Features, episode: Episode, classifier: str, out: Path, identity_sha256: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(run_identity_sha256=identity_sha256, fold=episode.fold, k=episode.k, draw=episode.draw,
                    classifier=classifier, supports=[list(p) for p in episode.supports],
                    queries=list(episode.queries), labels=list(episode.labels))
    prep.write_json(out / "episode.json", identity)
    if (out / "complete.json").exists():
        read_completed_episode(out)
        return
    start = time.perf_counter()
    scores, diagnostics = predict_episode(features, episode, classifier)
    classifier_seconds = time.perf_counter()-start
    start = time.perf_counter()
    prep.write_json(out / "fit.json", diagnostics)
    files = ["episode.json", "fit.json"]
    if scores is not None:
        write_predictions(out, episode.queries, episode.labels, scores, fm.RARE, diagnostics["score_type"])
        files.extend(("predictions.csv", "metrics.json"))
    if not (out / "costs.json").exists():
        prep.write_json(out / "costs.json", dict(classifier_seconds=classifier_seconds,
            serialization_seconds=time.perf_counter()-start,
            classifier_note="feature selection, normalization, fitting and query prediction",
            serialization_note="fit/prediction/count files; measurement/marker write excluded"))
    validate_costs(out / "costs.json")
    prep.complete(out, [*files, "costs.json"])


def planned_probe(episode: Episode, phase: str) -> bool:
    return episode.draw < 25 and (phase == "reporting" or episode.k == 4)


def validate_costs(path: Path, keys: Sequence[str] = ("classifier_seconds", "serialization_seconds")) -> None:
    value=prep.read_json(path)
    if any(type(value.get(key)) not in (float, int) or not np.isfinite(value[key]) or value[key]<0
           for key in keys):
        raise ValueError("invalid classifier/serialization costs")


def verify_completed_run_root(out: Path, phase: str) -> None:
    """A present root marker seals metadata/costs; absent roots may expose valid child cells."""
    marker = prep.verify_complete(out)
    if not {"identity.json", "summary.json", "costs.json"} <= marker["files"].keys():
        raise ValueError(f"{phase} completion marker omits run identity/summary/costs")
    validate_costs(out / "costs.json", ("classifiers_and_serialization_seconds",))


def read_completed_episode(out: Path) -> tuple[dict, dict]:
    """A hash-valid marker must also cover the fit and every successful query output."""
    marker = prep.verify_complete(out)
    if not {"episode.json", "fit.json", "costs.json"} <= marker["files"].keys():
        raise ValueError("episode completion marker omits identity/fit/costs")
    fit = prep.read_json(out / "fit.json")
    if fit["state"] not in ("complete", "fit_failed"):
        raise ValueError("invalid completed episode fit state")
    if fit["state"] == "complete" and not {"predictions.csv", "metrics.json"} <= marker["files"].keys():
        raise ValueError("episode completion marker omits predictions/metrics")
    validate_costs(out / "costs.json")
    return prep.read_json(out / "episode.json"), fit


def read_episode_counts(out: Path, labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Validate saved counts before NumPy can broadcast or truncate them into integer arrays."""
    metric = prep.read_json(out / "metrics.json")
    totals = [labels.count(c) for c in fm.RARE]
    if metric["classes"] != list(fm.RARE) or metric["totals"] != totals:
        raise ValueError("episode query classes/denominators differ")
    correct = metric["correct"]
    if not isinstance(correct, list) or len(correct) != len(totals) or \
            any(type(value) is not int or not 0 <= value <= n for value, n in zip(correct, totals, strict=True)):
        raise ValueError("invalid integer class counts")
    return np.asarray(correct, dtype=np.int64), np.asarray(totals, dtype=np.int64)


def run_arm(folder: Path, cache: Path, arm: str, out: Path, phase: str, omission_reason: str = "",
            nomination: Path | None = None) -> None:
    out = prep.experiment_out(out)
    lock = prep.load_lock(folder)
    if phase == "reporting":
        if nomination is None:
            raise ValueError("reporting requires the prior nomination decision (including explicit no-winner cases)")
        nomination_decision(folder, nomination)
    if lock["evaluation_environment"] != prep.math_environment():
        raise ValueError("CPU Python/NumPy/SciPy differs from the pre-score input lock")
    planned = episodes(folder, phase)  # entire roster validated before any labelled query is scored
    features = open_features(folder, cache, arm)
    identity = dict(lock_sha256=prep.sha256(folder / "input_lock.json"), arm=arm, phase=phase,
                    code=prep.code_identity(), environment=extract.environment(),
                    cache_complete_sha256=prep.sha256(cache / "complete.json"),
                    omit_logistic_reason=omission_reason,
                    nomination_sha256=prep.sha256(nomination / "nomination.json") if nomination else None)
    with prep.writer_lock(out):
        prep.write_json(out / "identity.json", identity)
        identity_hash = prep.sha256(out / "identity.json")
        if (out / "complete.json").exists():
            verify_completed_run_root(out, phase)
            verify_run(out, planned, omission_reason, identity_hash)
            return
        start = time.perf_counter()
        for episode in planned:
            evaluate_episode(features, episode, "prototype", out / "prototype" / episode.name, identity_hash)
            if not omission_reason and planned_probe(episode, phase):
                evaluate_episode(features, episode, "logistic", out / "logistic" / episode.name, identity_hash)
        if phase == "reporting":
            nine_way(folder, features, out / "nine_way", identity_hash)
        summary = verify_run(out, planned, omission_reason, identity_hash)
        prep.write_json(out / "summary.json", summary)
        if not (out / "costs.json").exists():
            prep.write_json(out / "costs.json", dict(classifiers_and_serialization_seconds=time.perf_counter()-start,
                                                    dtype="float64", n_episodes=len(planned)))
        validate_costs(out / "costs.json", ("classifiers_and_serialization_seconds",))
        # The root completion/export covers predictions and every child marker too.
        artifacts = ["identity.json", "summary.json", "costs.json"]
        for classifier in ("prototype", "logistic", "nine_way"):
            for marker in sorted((out / classifier).rglob("complete.json")) if (out / classifier).exists() else []:
                child = prep.verify_complete(marker.parent)
                artifacts.extend(str((marker.parent / name).relative_to(out)).replace("\\", "/")
                                 for name in [*child["files"], "complete.json"])
        prep.complete(out, sorted(artifacts))


def verify_run(out: Path, planned: Sequence[Episode], omission_reason: str, identity_hash: str) -> dict:
    phase = prep.read_json(out / "identity.json")["phase"]
    counts = {"prototype": dict(planned=len(planned), complete=0, failed=0),
              "logistic": dict(planned=0 if omission_reason else sum(planned_probe(e, phase) for e in planned),
                               complete=0, failed=0)}
    for classifier in counts:
        selected = planned if classifier == "prototype" else [e for e in planned if planned_probe(e, phase) and not omission_reason]
        for episode in selected:
            path = out / classifier / episode.name
            saved, fit = read_completed_episode(path)
            if saved["run_identity_sha256"] != identity_hash or saved["queries"] != list(episode.queries) or \
                    saved["labels"] != list(episode.labels) or saved["classifier"] != classifier or \
                    (saved["fold"], saved["k"], saved["draw"]) != (episode.fold, episode.k, episode.draw) or \
                    saved["supports"] != [list(p) for p in episode.supports]:
                raise ValueError("completed episode identity differs")
            good = fit["state"] == "complete"
            counts[classifier]["complete" if good else "failed"] += 1
    if phase == "reporting":
        verify_nine_way(out / "nine_way",identity_hash)
    return dict(episodes=counts, omit_logistic_reason=omission_reason or None)


def collect_development(folder: Path, out: Path, arm: str) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """No partial averages: a single absent/failed required episode removes the FM score."""
    planned = episodes(folder, "development")
    query_counts = np.zeros((fm.N_FOLDS, len(fm.RARE)), dtype=np.int64)
    for fold in range(fm.N_FOLDS):
        one = next(e for e in planned if e.fold == fold)
        query_counts[fold] = [one.labels.count(c) for c in fm.RARE]
    hits = np.zeros((len(fm.K_DEV), fm.N_DRAWS, fm.N_FOLDS, len(fm.RARE)), dtype=np.int64)
    failures = []
    if (out / "complete.json").exists():
        verify_completed_run_root(out, "development")
    identity = prep.read_json(out / "identity.json") if (out / "identity.json").exists() else None
    if identity is not None and (identity["arm"] != arm or identity["phase"] != "development" or identity["code"] != prep.code_identity() or
                                identity["lock_sha256"] != prep.sha256(folder / "input_lock.json")):
        raise ValueError("development run identity differs from current lock/code")
    for episode in planned:
        path = out / "prototype" / episode.name
        if not (path / "complete.json").exists():
            failures.append(episode.name)
            continue
        saved, fit = read_completed_episode(path)
        if identity is None or \
                saved["run_identity_sha256"] != prep.sha256(out / "identity.json") or \
                saved["supports"] != [list(p) for p in episode.supports] or saved["queries"] != list(episode.queries) or \
                saved["labels"] != list(episode.labels) or saved["classifier"] != "prototype" or \
                (saved["fold"], saved["k"], saved["draw"]) != (episode.fold, episode.k, episode.draw):
            raise ValueError("development episode identity differs")
        if fit["state"] != "complete":
            failures.append(episode.name)
            continue
        correct, _ = read_episode_counts(path, episode.labels)
        hits[fm.K_DEV.index(episode.k), episode.draw, episode.fold] = correct
    status = dict(started=identity is not None, available=len(planned)-len(failures),
                  planned=len(planned), missing_or_failed=failures)
    return (None if failures else hits), query_counts, status


def nominate(folder: Path, runs: Path, out: Path) -> dict:
    """Bind the first decision beside the input lock; later outputs copy it without reselection."""
    out = prep.experiment_out(out)
    if out == folder.resolve():
        raise ValueError("nomination output must differ from the input-lock folder")
    # The study lock, rather than --out, serializes nomination in different directories.
    with prep.writer_lock(folder):
        lock = prep.load_lock(folder)
        if lock["evaluation_environment"] != prep.math_environment():
            raise ValueError("CPU Python/NumPy/SciPy differs from the pre-score input lock")
        if (folder / "nomination_binding.json").exists():
            binding = read_nomination_binding(folder)
        else:
            if any((out / name).exists() for name in ("nomination.json", "complete.json")):
                raise ValueError("missing study-level nomination binding; cannot adopt or replace an unbound decision")
            decision = development_decision(folder, runs, lock)
            binding = nomination_binding(decision, prep.sha256(folder / "input_lock.json"))
            # This is the immutable S3 commit point, before any arbitrary nomination output.
            prep.write_json(folder / "nomination_binding.json", binding)
        with prep.writer_lock(out):
            prep.write_json(out / "nomination.json", binding["decision"])
            prep.write_json(out / "nomination_binding.json", binding)
            prep.complete(out, ["nomination.json", "nomination_binding.json"])
        return binding["decision"]


def development_decision(folder: Path, runs: Path, lock: dict) -> dict:
    """Compute a decision only when this study has not yet bound any nomination."""
    scores, seconds, all_arms = {}, {}, {}
    for arm, spec in lock["arms"].items():
        if spec["status"] != "available":
            all_arms[arm] = spec
            continue
        hits, queries, status = collect_development(folder, runs / arm / "development",arm)
        record = dict(status="complete" if hits is not None else "incomplete" if status["started"] else "not_started",
                      **status)
        if hits is not None:
            score = fm.nomination_score(hits, queries)
            record.update(score=score, k_scores={str(k): float(np.mean(hits[i].sum(axis=1)/queries.sum(axis=0)))
                                               for i, k in enumerate(fm.K_DEV)},
                          per_class_queries=queries.sum(axis=0).tolist())
            scores[arm] = score
            if arm != fm.REFERENCE:
                seconds[arm] = spec["timing"]["median_seconds_per_image"]
        all_arms[arm] = record
    if not any(record.get("started", False) for record in all_arms.values()):
        raise ValueError(f"no development runs have started under {runs}; check the runs path")
    chosen = fm.nominate(scores, seconds)
    return dict(lock_sha256=prep.sha256(folder / "input_lock.json"), code=prep.code_identity(),
                nominee=chosen, primary_fm=chosen, arms=all_arms,
                interpretation="prespecified allocation rule; selected development score is optimistic",
                reporting_scope="primary nominee plus all secondary arms" if chosen else
                                "exploratory feasibility report; no comparative winner")


def nomination_binding(decision: dict, lock_hash: str) -> dict:
    """Canonical decision-plus-lock hash, with the full original decision for recovery/copies."""
    payload = dict(lock_sha256=lock_hash, decision=decision)
    return dict(**payload, binding_sha256=prep.object_hash(payload))


def read_nomination_binding(folder: Path) -> dict:
    path = folder / "nomination_binding.json"
    if not path.exists():
        raise ValueError("missing study-level nomination binding beside input_lock.json; restore the original S3 binding")
    binding = prep.read_json(path)
    decision = binding.get("decision")
    lock_hash = prep.sha256(folder / "input_lock.json")
    if not isinstance(decision, dict) or binding != nomination_binding(decision, lock_hash):
        raise ValueError("invalid or mismatched study-level nomination binding")
    if decision.get("lock_sha256") != lock_hash or decision.get("code") != prep.code_identity():
        raise ValueError("nomination decision belongs to a different input lock/code")
    return binding


def nomination_decision(folder: Path, out: Path) -> dict:
    marker = prep.verify_complete(out)
    if "nomination.json" not in marker["files"]:
        raise ValueError("nomination completion marker omits its decision")
    decision = prep.read_json(out / "nomination.json")
    if decision.get("lock_sha256") != prep.sha256(folder / "input_lock.json") or decision.get("code") != prep.code_identity():
        raise ValueError("nomination decision belongs to a different input lock/code")
    binding = read_nomination_binding(folder)
    if "nomination_binding.json" not in marker["files"]:
        raise ValueError("nomination completion marker omits its study binding")
    if binding != nomination_binding(decision, prep.sha256(folder / "input_lock.json")) or \
            prep.read_json(out / "nomination_binding.json") != binding:
        raise ValueError("nomination decision differs from the study-level nomination binding")
    return decision


CLUSTER_METHOD = "95% percentile of 5,000 whole-cluster resamples, seed 0; conditional on the fitted head"
EXACT_METHOD = "exact Clopper-Pearson 95%; image-level and conditional (no patient IDs)"


def kermany_pid(path: str) -> str:
    """Manifest pid of a Kermany image: the second field of its file name (A2 A-5)."""
    parts = Path(path).name.split("-")
    if len(parts) != 3 or not parts[1]:
        raise ValueError(f"not a Kermany file name: {path}")
    return parts[1]


def nine_way_intervals(labels: Sequence[str], predicted: Sequence[str], clusters: Sequence[str] | None) -> dict:
    """A2 B-3: FRR and per-class recall intervals; clusters=None means exact image-level intervals."""
    def interval(rows: list[int], hits: list[int]) -> list[float] | None:
        if not rows:
            return None
        if clusters is None:
            return list(fm.clopper_pearson(sum(hits), len(hits)))
        return list(fm.cluster_interval(hits, [clusters[i] for i in rows]))

    normal = [i for i, label in enumerate(labels) if label == "NORMAL"]
    recall = {}
    for c in fm.NINE:
        rows = [i for i, label in enumerate(labels) if label == c]
        if rows:
            recall[c] = interval(rows, [int(predicted[i] == c) for i in rows])
    return dict(method=EXACT_METHOD if clusters is None else CLUSTER_METHOD,
                false_rare=interval(normal, [int(predicted[i] in fm.RARE) for i in normal]), recall=recall)


def kermany_audit(fitting: Sequence[dict], queries: Sequence[dict], predicted: Sequence[str]) -> dict:
    """Kermany FRR on all NORMAL test images and on the patient-disjoint headline set (A2 A-5).

    Patient-disjoint: the pid appears in no common fitting row of any class. Intervals cluster
    by pid; rare queries have no patient IDs, so each is its own cluster.
    """
    fitted = {kermany_pid(r["path"]) for r in fitting if r["label"] in prep.COMMON}
    labels = [r["label"] for r in queries]
    clusters = [kermany_pid(r["path"]) if r["label"] in prep.COMMON else r["path"] for r in queries]
    numerator, denominator = fm.false_rare_rate(labels, predicted)
    disjoint = [i for i, r in enumerate(queries) if r["label"] == "NORMAL" and clusters[i] not in fitted]
    hits = [int(predicted[i] in fm.RARE) for i in disjoint]
    return dict(source="Kermany", false_rare_n=numerator, normal_n=denominator,
                false_rare_rate=numerator/denominator if denominator else None,
                intervals=nine_way_intervals(labels, predicted, clusters),
                patient_disjoint=dict(
                    role="headline Kermany FRR (A2 A-5); the all-image FRR above is secondary",
                    definition="NORMAL test images whose pid (file-name field 2) is in no common fitting row",
                    false_rare_n=sum(hits), normal_n=len(disjoint), patients=len({clusters[i] for i in disjoint}),
                    false_rare_rate=sum(hits)/len(disjoint) if disjoint else None,
                    interval=list(fm.cluster_interval(hits, [clusters[i] for i in disjoint])) if disjoint else None),
                external_status="not run by this legacy-only command; separate screened query lock required",
                limitation="patient overlap remains in the all-image FRR; RETFound pretraining exposure to Kermany "
                           "is not addressed; no patient independence claimed")


def nine_way(folder: Path, features: Features, out: Path, identity_hash: str) -> None:
    """One deterministic head using the complete fixed real fitting roster, never K-shot FRR."""
    out.mkdir(parents=True, exist_ok=True)
    roster = prep.read_csv(folder / "nine_way_roster.csv")
    fitting = [r for r in roster if r["role"] == "fitting"]
    queries = [r for r in roster if r["role"] == "query"]
    paths=[r["path"] for r in roster]
    if len(set(paths)) != len(paths) or len(fitting)+len(queries) != len(roster) or \
            {r["label"] for r in fitting} != set(fm.NINE):
        raise ValueError("nine-way roster needs unique disjoint fitting/query paths and every fitting class")
    identity = dict(run_identity_sha256=identity_hash, fitting=fitting, queries=queries, classes=list(fm.NINE))
    prep.write_json(out / "head.json", identity)
    if (out / "complete.json").exists():
        verify_nine_way(out,identity_hash)
        return
    start = time.perf_counter()
    supports = [features.select([r["path"] for r in fitting if r["label"] == c]) for c in fm.NINE]
    prototypes = fm.centroids(supports)
    scores = fm.unit(features.select([r["path"] for r in queries])) @ prototypes.T
    labels = [r["label"] for r in queries]
    classifier_seconds = time.perf_counter()-start
    start = time.perf_counter()
    write_predictions(out, [r["path"] for r in queries], labels, scores, fm.NINE, "cosine")
    predicted = [fm.NINE[i] for i in scores.argmax(axis=1)]
    prep.write_json(out / "source_audit.json", kermany_audit(fitting, queries, predicted))
    if not (out / "costs.json").exists():
        prep.write_json(out / "costs.json", dict(classifier_seconds=classifier_seconds,
            serialization_seconds=time.perf_counter()-start,
            classifier_note="normalization, deterministic head fitting and query prediction"))
    validate_costs(out / "costs.json")
    prep.complete(out, ["head.json", "predictions.csv", "metrics.json", "source_audit.json", "costs.json"])


def verify_nine_way(out: Path,identity_hash: str) -> None:
    marker=prep.verify_complete(out)
    required={"head.json","predictions.csv","metrics.json","source_audit.json","costs.json"}
    if not required <= marker["files"].keys() or \
            prep.read_json(out / "head.json")["run_identity_sha256"] != identity_hash:
        raise ValueError("nine-way completion/identity is incomplete or belongs to a different run")
    validate_costs(out / "costs.json")


def draw_summary(values: Sequence[float]) -> list:
    """Conditional support-draw variability, never patient CIs or episode SEM."""
    if not values:
        return [None, None, None, None]
    numbers = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numbers).all():
        raise ValueError("nonfinite draw summaries")
    low, high = np.percentile(numbers, [5, 95])
    return [float(numbers.mean()), float(numbers.std(ddof=1)) if len(numbers)>1 else None, float(low), float(high)]


def reporting_metrics(folder: Path, out: Path, planned: Sequence[Episode], arm: str, nomination_hash: str) -> dict:
    """Completed paired cells only; missing/failed cells retain their planned denominator."""
    result = {classifier: {} for classifier in ("prototype", "logistic")}
    if (out / "complete.json").exists():
        verify_completed_run_root(out, "reporting")
    if not (out / "identity.json").exists():
        return result
    identity = prep.read_json(out / "identity.json")
    if (out / "complete.json").exists():
        verify_nine_way(out / "nine_way",prep.sha256(out / "identity.json"))
    if identity["lock_sha256"] != prep.sha256(folder / "input_lock.json") or identity["code"] != prep.code_identity() or \
            identity["phase"] != "reporting" or identity["arm"] != arm or identity["nomination_sha256"] != nomination_hash:
        raise ValueError("reporting run belongs to a different lock/arm/nomination")
    root_hash = prep.sha256(out / "identity.json")
    for classifier in result:
        for episode in planned:
            if classifier == "logistic" and not planned_probe(episode,"reporting"):
                continue
            path=out / classifier / episode.name
            if not (path / "complete.json").exists():
                continue
            record, fit=read_completed_episode(path)
            if record["run_identity_sha256"] != root_hash or record["classifier"] != classifier or \
                    record["supports"] != [list(p) for p in episode.supports] or \
                    record["queries"] != list(episode.queries) or record["labels"] != list(episode.labels) or \
                    (record["fold"],record["k"],record["draw"]) != (episode.fold,episode.k,episode.draw):
                raise ValueError("reporting episode identity differs")
            if fit["state"] != "complete":
                continue
            correct, totals=read_episode_counts(path, episode.labels)
            value=fm.macro_recall(correct,totals)
            result[classifier][episode.k,episode.draw]=value
    return result


K_MEAN = (1, 2, 4, 8)  # A2 B-1: no K is primary; the single number is the mean over these K
K_MEAN_LABEL = "mean_k1_2_4_8"


def k_mean(scored: dict, n: int) -> dict:
    """Per-draw mean over K = 1, 2, 4, 8 of one arm's macro recalls; a draw missing any K is absent."""
    return {r: float(np.mean([scored[k, r] for k in K_MEAN])) for r in range(n) if all((k, r) in scored for k in K_MEAN)}


def image_sizes(folder: Path) -> dict[str, tuple[int, int]]:
    """Original (width, height) of every rare benchmark image, from the locked screen inventory."""
    return {r["path"]: (int(r["width"]), int(r["height"])) for r in prep.read_csv(folder / "image_inventory.csv")}


def size_hits(episode: Episode, sizes: dict[str, tuple[int, int]], classes: Sequence[str] = fm.RARE) -> np.ndarray:
    """Correct counts per class of the size-only reference (A2 N-1) for one locked episode."""
    picked = [episode.supports[fm.RARE.index(c)] for c in classes]
    rows = [i for i, label in enumerate(episode.labels) if label in classes]
    predicted = fm.size_predict([np.array([sizes[p] for p in paths]) for paths in picked],
                                np.array([sizes[episode.queries[i]] for i in rows]))
    hits = np.zeros(len(classes), dtype=np.int64)
    for i, chosen in zip(rows, predicted):
        hits[classes.index(episode.labels[i])] += classes[chosen] == episode.labels[i]
    return hits


def size_reference_rows(folder: Path) -> list[list]:
    """Size-only macro recall per K, development (pooled over folds, as nomination) and reporting.

    A diagnostic of the benchmark: no pixel or encoder is used, and it is never nominated.
    """
    sizes, rows = image_sizes(folder), []
    development = episodes(folder, "development")
    totals = {fold: np.array([next(e for e in development if e.fold == fold).labels.count(c) for c in fm.RARE])
              for fold in range(fm.N_FOLDS)}
    pooled: dict[tuple[int, int], np.ndarray] = {}
    for episode in development:
        pooled[episode.k, episode.draw] = pooled.get((episode.k, episode.draw), 0) + size_hits(episode, sizes)
    queries = sum(totals.values())
    dev_scores = {key: float(np.mean(value / queries)) for key, value in pooled.items()}
    reporting = {(e.k, e.draw): fm.macro_recall(size_hits(e, sizes), np.array([e.labels.count(c) for c in fm.RARE]))
                 for e in episodes(folder, "reporting")}
    for phase, scores in (("development", dev_scores), ("reporting", reporting)):
        for k in sorted({k for k, _ in scores}):
            draws = [scores[k, r] for r in range(fm.N_DRAWS) if (k, r) in scores]
            rows.append([phase, k, len(draws), fm.N_DRAWS, *draw_summary(draws)])
        means = list(k_mean(scores, fm.N_DRAWS).values())
        rows.append([phase, K_MEAN_LABEL, len(means), fm.N_DRAWS, *draw_summary(means)])
    return rows


def summarize(folder: Path, runs: Path, nomination: Path, out: Path) -> None:
    """Legacy support-draw summaries and all paired contrasts, in recall fractions.

    Adds the A2 rows: the K-mean single number per arm and contrast (B-1) and the size-only
    reference (N-1) for development and reporting episodes.
    """
    out=prep.experiment_out(out)
    lock=prep.load_lock(folder)
    if lock["evaluation_environment"] != prep.math_environment():
        raise ValueError("summary arithmetic environment differs from the input lock")
    decision=nomination_decision(folder,nomination)
    nomination_hash=prep.sha256(nomination / "nomination.json")
    planned=episodes(folder,"reporting")
    ks=sorted({e.k for e in planned})
    order=sorted(lock["arms"], key=lambda arm: 3 if arm == fm.REFERENCE else fm.TIE_SLOT[arm])
    values={arm:reporting_metrics(folder,runs/arm/"reporting",planned,arm,nomination_hash)
            for arm in order if lock["arms"][arm]["status"] == "available"}
    if not any((runs / arm / "reporting" / "identity.json").exists() for arm in values):
        raise ValueError(f"no reporting runs have started under {runs}; check the runs path")
    summaries=[]
    contrasts=[]
    for arm,cells in values.items():
        for classifier,scored in cells.items():
            n=100 if classifier == "prototype" else 25
            for k in ks:
                draws=[scored[k,r] for r in range(n) if (k,r) in scored]
                summaries.append([arm,classifier,k,len(draws),n,*draw_summary(draws)])
            means=list(k_mean(scored,n).values())
            summaries.append([arm,classifier,K_MEAN_LABEL,len(means),n,*draw_summary(means)])
    for arm_a,arm_b in combinations(values,2):
        for classifier in ("prototype","logistic"):
            n=100 if classifier == "prototype" else 25
            left,right=values[arm_a][classifier],values[arm_b][classifier]
            primary=classifier == "prototype" and arm_a == decision["nominee"] and arm_b == fm.REFERENCE
            for k in ks:
                differences=[left[k,r]-right[k,r] for r in range(n) if (k,r) in left and (k,r) in right]
                contrasts.append([arm_a,arm_b,"primary" if primary else "secondary",classifier,k,
                                  len(differences),n,*draw_summary(differences)])
            mean_left,mean_right=k_mean(left,n),k_mean(right,n)
            differences=[mean_left[r]-mean_right[r] for r in mean_left if r in mean_right]
            contrasts.append([arm_a,arm_b,"primary" if primary else "secondary",classifier,K_MEAN_LABEL,
                              len(differences),n,*draw_summary(differences)])
    size_rows=size_reference_rows(folder)
    with prep.writer_lock(out):
        prep.write_json(out / "nomination.json", decision)
        prep.write_json(out / "nomination_binding.json", read_nomination_binding(folder))
        prep.write_csv(out/"support_draw_summary.csv",["arm","classifier","k","available","planned",
                       "mean","sd","p05","p95"],summaries)
        prep.write_csv(out/"paired_differences.csv",["arm_a","arm_b","role","classifier","k","available_pairs",
                       "planned_pairs","mean_a_minus_b","sd","p05","p95"],contrasts)
        prep.write_csv(out/"size_only_reference.csv",["phase","k","available","planned","mean","sd","p05","p95"],size_rows)
        prep.write_json(out/"interpretation.json",dict(lock_sha256=prep.sha256(folder/"input_lock.json"),
            nomination_sha256=nomination_hash,nominee=decision["nominee"],units="recall fractions (0..1)",
            reporting_scope=decision["reporting_scope"],
            k=f"no K is primary (A2 B-1); the single number is {K_MEAN_LABEL}, per draw then summarized",
            size_only_reference="nearest mean support (width, height); a benchmark diagnostic, never an arm (A2 N-1)",
            variability="conditional on these images and support draws; not new-patient confidence intervals",
            paired="shared completed support/query episodes; failed/missing cells are never imputed",
            external="not run here; legacy-only summaries do not complete the required external source component",
            unavailable={arm:spec["reason"] for arm,spec in lock["arms"].items() if spec["status"] == "unavailable"}))
        prep.complete(out,["nomination.json","nomination_binding.json","support_draw_summary.csv",
                           "paired_differences.csv","size_only_reference.csv","interpretation.json"])


GEOMETRY_METRICS = ("width", "height", "retained_fraction", "border_share", "near_white_share")


def geometry_summary(rows: Sequence[dict]) -> list[list]:
    """Count, median, min and max of each geometry metric by (source, class, role) (A2, A1 rule 4)."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["source"], row["class"], row["role"]), []).append(row)
    table = []
    for key in sorted(groups):
        stats = []
        for metric in GEOMETRY_METRICS:
            values = np.array([float(r[metric]) for r in groups[key]])
            stats += [float(np.median(values)), float(values.min()), float(values.max())]
        table.append([*key, len(groups[key]), *stats])
    return table


def geometry(folder: Path, cache: Path, arm: str, out: Path, external: Path | None = None,
             external_cache: Path | None = None) -> None:
    """Score-free geometry table for one arm's verified caches; labels group rows, no output is read."""
    out = prep.experiment_out(out)
    open_features(folder, cache, arm)  # verifies the bundle against the lock before reading geometry
    roster = {r["path"]: r for r in prep.read_csv(folder / "nine_way_roster.csv")}
    rows = [dict(r, source="benchmark" if roster[r["path"]]["label"] in fm.RARE else "Kermany",
                 role=roster[r["path"]]["role"], **{"class": roster[r["path"]]["label"]})
            for r in prep.read_csv(cache / "geometry.csv")]
    if external is not None:
        from fm_external import external_features  # fm_external imports this module
        if external_cache is None:
            raise ValueError("external geometry needs the external feature cache")
        external_features(folder, external_cache, external, arm)
        queries = {r["path"]: r for r in prep.read_csv(external / "external_queries.csv")}
        rows += [dict(r, source=queries[r["path"]]["source"], role="query", **{"class": queries[r["path"]]["label"]})
                 for r in prep.read_csv(external_cache / "geometry.csv")]
    header = ["source", "class", "role", "n", *[f"{m}_{s}" for m in GEOMETRY_METRICS for s in ("median", "min", "max")]]
    with prep.writer_lock(out):
        prep.write_csv(out / "geometry_summary.csv", header, geometry_summary(rows))
        prep.write_json(out / "geometry_identity.json", dict(arm=arm, lock_sha256=prep.sha256(folder / "input_lock.json"),
            cache_complete_sha256=prep.sha256(cache / "complete.json"),
            external_cache_complete_sha256=prep.sha256(external_cache / "complete.json") if external else None,
            note="descriptive; gates nothing (A2). Retained fraction is per arm; the other metrics are image properties"))
        prep.complete(out, ["geometry_summary.csv", "geometry_identity.json"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    for phase in ("develop", "report"):
        command = sub.add_parser(phase)
        for name in ("lock", "cache", "out"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--arm", choices=extract.REPOSITORIES, required=True)
        command.add_argument("--omit-logistic-reason", default="", help="dated, disclosed reduction; never outcome-based")
        if phase == "report":
            command.add_argument("--nomination", type=Path, required=True)
    nominate_parser = sub.add_parser("nominate")
    for name in ("lock", "runs", "out"):
        nominate_parser.add_argument("--" + name, type=Path, required=True)
    summary_parser=sub.add_parser("summarize")
    for name in ("lock","runs","nomination","out"):
        summary_parser.add_argument("--"+name,type=Path,required=True)
    shape = sub.add_parser("geometry", help="score-free geometry and padding table by source, class and role")
    for name in ("lock", "cache", "out"):
        shape.add_argument("--" + name, type=Path, required=True)
    shape.add_argument("--arm", choices=extract.REPOSITORIES, required=True)
    shape.add_argument("--external", type=Path, help="external query-lock folder (fm_external.py freeze)")
    shape.add_argument("--external-cache", type=Path, help="external feature cache for the same arm")
    args = parser.parse_args()
    if args.step == "nominate":
        print(json.dumps(nominate(args.lock, args.runs, args.out), indent=2))
    elif args.step == "summarize":
        summarize(args.lock,args.runs,args.nomination,args.out)
    elif args.step == "geometry":
        geometry(args.lock, args.cache, args.arm, args.out, args.external, args.external_cache)
    else:
        run_arm(args.lock, args.cache, args.arm, args.out, "development" if args.step == "develop" else "reporting",
                args.omit_logistic_reason, getattr(args, "nomination", None))


if __name__ == "__main__":
    main()
