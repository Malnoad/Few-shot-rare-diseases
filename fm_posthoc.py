"""Post hoc analyses of the locked FM study: P0-P9, R1 and the laptop half of G1 (P9 also on G1).

The rules are protocols.md section 8 ("Plan for S5 and S6", version 2), timestamped by commit
1eb3964 on 16 November 2026 before any P1-P8 or G1 number (rule 1). Nothing here changes a
locked number, the nominee or A3's placement table (rule 2). Every output row carries a
`posthoc` label with its date; P0 and P7 say "rule fixed after first look" and P6 says
"computed before the rules were timestamped". P2 and P3 are analyses outside the locked study
(rule 4).

Layout: pure functions take arrays and return numbers or rows; `Study` owns all I/O. It checks
every locked archive against its pinned SHA256, restores the input lock and the feature caches
through the locked verifiers (fm_prepare.restore_lock, fm_evaluate.open_features,
fm_external.external_features) into a temporary folder, reads the results archive in place, and
writes only to results/fm_v1/posthoc/ (write-once: a rerun never replaces different bytes).

    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py run --attempt <attempt folder>
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py g1 --attempt <attempt folder> --padded <G1 folder>
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py g1-report     # rewrite G1_comparison.md only
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py p10-links --s1-text <text> --s1-links <links>
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py p10-screen --attempt <attempt> --data-root D:/Few-shot \
        --sheet <outside the repo>/P10_review_sheet.html

Phase-1 controls E1-E4 (protocols.md section 8, "Phase-1 controls E1-E4"; outputs never replace a locked result):

    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py e3 --attempt <attempt> --phase1 <Drive phase1/> [--padded <G1 folder>]
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py e1 --attempt <attempt> --phase1 <Drive phase1/> [--data-root D:/Few-shot]
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py e2 --attempt <attempt> --phase1 <Drive phase1/> [--sheet <Drive folder>]
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py e4 --attempt <attempt> --phase1 <Drive phase1/>
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py e-report      # E1_comparison.md from the E1 CSVs
    D:/Few-shot/.venv/Scripts/python.exe -I fm_posthoc.py e2f-marks --marks <pass1/marks.csv> [--key <key/key.csv>]

Score once, after the last session. Use `--partial` to explicitly score an incomplete delivery.
`phase1-render.zip` is cached by receipt SHA256 in the system temporary directory.
`--drop` applies the laptop drop order.

`g1 --padded` takes the Drive g1/ folder as Colab saved it (each zip checked against its receipt)
or a folder holding the unzipped MIRAGE/ and Inception/. It needs the locked P9 tables in --out.
"""
from __future__ import annotations

import sys
from pathlib import Path

# -I drops the script folder from sys.path; add back this checkout only (never the archive's folder).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

import fm_core as fm  # noqa: E402
import fm_evaluate as ev  # noqa: E402
import fm_external as fx  # noqa: E402
import fm_extract as extract  # noqa: E402
import fm_prepare as prep  # noqa: E402

ATTEMPT = "final-20261113T170804Z"
RESULTS_ZIP = "evaluation/results-20261114T093033Z.zip"
# receipt.json of the attempt (13 Nov) and the results receipt (14 Nov), as recorded in protocols.md.
PINNED = {
    RESULTS_ZIP: "c963b1d7298c28f8151769d8d29cf98538ce0897582f57ff2ee867c76351b25f",
    "input-lock.zip": "239838279b77bb37211398a016c1c766273a35f5a06649b31f5cd97f562c001c",
    "external-lock.zip": "3414889b53ee10c526c4d3dfb28b7477676f5bf449c2729008902f93447f45d0",
    "features-RETFound.zip": "d38932f6d68877fb3084be31876692f5042e2acd268836f2cb1d16198634e054",
    "features-RETFound-external.zip": "22cd66599b3f50ba91452cd8a37e769171be16703d89b3d9fc57020533e32c7c",
    "features-DINOv2.zip": "d8b674423a192df1ae5ee2de2222284038bcbe4cb2de077ec3c9c095aca0dbeb",
    "features-DINOv2-external.zip": "fc91bc30c1d94740b0a9f955005c70b2dac7aceea38e031fe4a180e51fa4884a",
    "features-MIRAGE.zip": "b435e3082e64bdcb1917ba0baa2d10e57b62123e4ded28a697098ab20890ac22",
    "features-MIRAGE-external.zip": "0c28e2e0d953ee290426e133bcefe4ccce51aa5c1a2d8b852f2f56ccd8a15512",
    "features-Inception.zip": "b3a25dcc3e40bf8a37e8cad0a5332f5fa04146b0ffb181fa99c75b6de01933ef",
    "features-Inception-external.zip": "fa464b40afc2b352a221d4cfe8725231c3a0b43784b4d11bd23220eea4544bed",
}
RULES_COMMIT = "1eb39647b861035509a0e1e2e0541939635bb69b"  # rule 1: section 8 committed and pushed, 16 Nov
ARMS = ("RETFound", "DINOv2", "MIRAGE", "Inception")  # lock order; Inception is the reference
NOMINEE = "MIRAGE"  # sealed 14 Nov (reports/nomination/nomination.json); checked in Study
COMMON = fm.NINE[:4]
KS = fm.K_DEV  # K = 1, 2, 4, 8: the K mean of A2 B-1
OUT = Path(__file__).resolve().parent / "results" / "fm_v1" / "posthoc"


def labels_for(today: str) -> dict[str, str]:
    """The `posthoc` column of every output, by analysis (protocols.md section 8, rules 1-4)."""
    timed = f"post hoc, rule timestamped by commit {RULES_COMMIT[:7]} (16 Nov 2026), computed {today}"
    return dict(
        P0=f"post hoc, rule fixed after first look (first computed 15 Nov 2026), recomputed {today}",
        P7=f"post hoc, rule fixed after first look (first computed 15 Nov 2026), recomputed {today}; "
           "supplement line only, never called a result",
        P6=f"post hoc, computed before the rules were timestamped (first computed 16 Nov 2026), recomputed {today}",
        P1=timed, P4=timed, P5=timed, P8=timed, R1=timed,
        P2=f"{timed}; outside the locked study (rule 4): trained with augmented and GAN images",
        P3=f"{timed}; outside the locked study (rule 4): OCTDL MH used as examples although A1 makes it query-only",
        G1=f"{timed}; sensitivity check outside A2 A-11, never replaces a locked result")


# =============================================================== pure functions ===
def class_recalls(correct: np.ndarray, labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """Per-class recall of one query set; every class must be present."""
    correct, labels = np.asarray(correct, dtype=bool), np.asarray(labels)
    out = []
    for c in classes:
        rows = labels == c
        if not rows.any():
            raise ValueError(f"class {c} has no query")
        out.append(correct[rows].mean())
    return np.array(out)


def macro(correct: np.ndarray, labels: Sequence[str], classes: Sequence[str]) -> float:
    return float(class_recalls(correct, labels, classes).mean())


def present_macro(correct: np.ndarray, labels: Sequence[str], classes: Sequence[str]) -> tuple[float | None, list[str]]:
    """Macro recall over the classes that have a query here (P4's empty-class rule); None if none."""
    labels = np.asarray(labels)
    present = [c for c in classes if (labels == c).any()]
    if not present:
        return None, []
    return float(class_recalls(correct, labels, present).mean()), present


def k_mean(values: Mapping[tuple[int, int], float]) -> list[float]:
    """Per-draw mean over K = 1, 2, 4, 8 (A2 B-1); a draw missing any K is left out."""
    draws = sorted({r for _, r in values})
    return [float(np.mean([values[k, r] for k in KS])) for r in draws if all((k, r) in values for k in KS)]


def by_k(values: Mapping[tuple[int, int], float], ks: Sequence[int]) -> list[tuple[int | str, list[float]]]:
    """Per-K draw lists, then the K-mean draw list; the order every K table uses."""
    rows: list[tuple[int | str, list[float]]] = [(k, [v for (kk, _), v in sorted(values.items()) if kk == k]) for k in ks]
    return [*rows, (ev.K_MEAN_LABEL, k_mean(values))]


def mean_over_k(values: np.ndarray, ks_of_rows: np.ndarray) -> float:
    """Mean over K of the per-K mean (rows are (K, draw) cells)."""
    return float(np.mean([values[ks_of_rows == k].mean() for k in np.unique(ks_of_rows)]))


def two_way_choice(scores: np.ndarray, first: int, second: int) -> np.ndarray:
    """A2 A-1: the larger of two cosine columns; exact ties go to `first` (the fixed class order)."""
    return np.where(scores[:, first] >= scores[:, second], first, second)


def zscore_fuse(score_sets: Sequence[np.ndarray]) -> np.ndarray:
    """P7: per model and query, z-score the class cosines (population SD), add with equal weight, argmax."""
    total = None
    for s in score_sets:
        z = (s - s.mean(axis=1, keepdims=True)) / s.std(axis=1, keepdims=True)
        total = z if total is None else total + z
    return np.argmax(total, axis=1)


def loo_differences(a: np.ndarray, b: np.ndarray, labels: Sequence[str], classes: Sequence[str],
                    ks_of_rows: np.ndarray) -> np.ndarray:
    """P1: arm a minus arm b, mean macro recall over K, with each query left out in turn.

    a and b are [cells, queries] correctness (cells are (K, draw) pairs on identical supports and
    queries). Returns one difference per left-out query.
    """
    labels = np.asarray(labels)
    out = np.empty(len(labels))
    for q in range(len(labels)):
        keep = np.arange(len(labels)) != q
        da = [macro(row[keep], labels[keep], classes) for row in a]
        db = [macro(row[keep], labels[keep], classes) for row in b]
        out[q] = mean_over_k(np.array(da), ks_of_rows) - mean_over_k(np.array(db), ks_of_rows)
    return out


def top_two_gap(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P5: index of the top score, of the second score, and their gap, per row."""
    order = np.argsort(-scores, axis=1, kind="stable")
    rows = np.arange(len(scores))
    return order[:, 0], order[:, 1], scores[rows, order[:, 0]] - scores[rows, order[:, 1]]


def not_sure_cutoff(gaps: Sequence[float], share: float = 0.05) -> tuple[float, int]:
    """P5 cut-off: the gap below which `share` of the calibration scans fall.

    With n scans, the cut-off is the (floor(share * n) + 1)-th smallest gap, so floor(share * n)
    scans lie strictly below it (8 of 169 Kermany normals). Returns the cut-off and that count.
    """
    values = np.sort(np.asarray(gaps, dtype=np.float64))
    index = math.floor(share * len(values))
    cut = float(values[index])
    return cut, int((values < cut).sum())


def patient_recall(correct: np.ndarray, labels: Sequence[str], patients: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """Per-class recall with each patient weighted equally (the mean of per-patient hit shares)."""
    correct, labels, patients = np.asarray(correct, dtype=bool), np.asarray(labels), np.asarray(patients)
    out = []
    for c in classes:
        ids = sorted(set(patients[labels == c]))
        out.append(np.mean([correct[(labels == c) & (patients == p)].mean() for p in ids]))
    return np.array(out)


def prototype_predict(supports: Sequence[np.ndarray], queries: np.ndarray) -> np.ndarray:
    """Section 6 cosine prototypes; argmax takes the first class on exact ties."""
    return fm.predict_prototype(fm.centroids(supports), queries)


def head_predict(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """P2: an October softmax head; argmax of the logits equals argmax of the softmax."""
    return np.argmax(np.asarray(X, np.float64) @ np.asarray(W, np.float64) + np.asarray(b, np.float64), axis=1)


def summary(values: Sequence[float]) -> list:
    """n, mean, SD, 5th and 95th percentile over draws (support-draw spread, never a CI)."""
    return [len(values), *ev.draw_summary(list(values))]


SUMMARY = ["n_draws", "mean", "sd", "p05", "p95"]


def bound_row(k: int, n: int) -> list:
    low, high = fm.clopper_pearson(k, n)
    return [k, n, k / n, low, high]


def destination(label: str) -> str:
    return "NORMAL" if label == "NORMAL" else "common" if label in COMMON else "rare"


# ===================================================================== reading ===
@dataclass(frozen=True)
class Cell:
    """One saved predictions.csv: query order, labels, predictions and the score columns."""
    queries: tuple[str, ...]
    labels: tuple[str, ...]
    predicted: tuple[str, ...]
    classes: tuple[str, ...]
    scores: np.ndarray

    @property
    def correct(self) -> np.ndarray:
        return np.array([p == t for p, t in zip(self.predicted, self.labels)])


def parse_cell(text: str) -> Cell:
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    if header[:4] != ["query_id", "true_label", "predicted_label", "correct"]:
        raise ValueError("unexpected predictions.csv header")
    prefix = header[4].split("_", 1)[0] + "_"
    columns = [i for i, h in enumerate(header) if h.startswith(prefix)]
    return Cell(tuple(r[0] for r in body), tuple(r[1] for r in body), tuple(r[2] for r in body),
                tuple(header[i][len(prefix):] for i in columns),
                np.array([[float(r[i]) for i in columns] for r in body], dtype=np.float64))


CELL = re.compile(r"runs/([^/]+)/(development|reporting|external/OCTID|external/OCTDL)/(prototype|logistic)/"
                  r"fold(\d)_k(\d+)_draw(\d+)/predictions\.csv$")


class Results:
    """The locked results archive, read in place (never extracted)."""

    def __init__(self, path: Path):
        self.zip = zipfile.ZipFile(path)
        self.index: dict[tuple[str, str, str], dict[tuple[int, int, int], str]] = {}
        for name in self.zip.namelist():
            m = CELL.match(name)
            if m:
                arm, phase, clf, f, k, d = m.groups()
                self.index.setdefault((arm, phase, clf), {})[int(f), int(k), int(d)] = name
        self._cache: dict[tuple[str, str, str], dict[tuple[int, int, int], Cell]] = {}

    def text(self, name: str) -> str:
        return self.zip.read(name).decode("utf-8")

    def cells(self, arm: str, phase: str, classifier: str = "prototype") -> dict[tuple[int, int, int], Cell]:
        key = (arm, phase, classifier)
        if key not in self._cache:
            self._cache[key] = {k: parse_cell(self.text(n)) for k, n in self.index[key].items()}
        return self._cache[key]

    def nine_way(self, arm: str, source: str) -> Cell:
        sub = "reporting" if source == "Kermany" else f"external/{source}"
        return parse_cell(self.text(f"runs/{arm}/{sub}/nine_way/predictions.csv"))

    def table(self, name: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.text(name))))


def safe_unzip(archive: Path, out: Path) -> Path:
    """Extract a locked archive whose member names are plain relative IDs (fm_prepare.safe_id)."""
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if prep.safe_id(name) != name:
                raise ValueError(f"unsafe archive member: {name}")
        z.extractall(out)
    return out


@dataclass
class Study:
    """The locked attempt, verified before use; every analysis reads through this object."""
    attempt: Path
    work: Path
    results: Results = field(init=False)
    lock: Path = field(init=False)
    external: Path = field(init=False)

    def __post_init__(self):
        receipt = prep.read_json(self.attempt / "receipt.json")
        for name, digest in PINNED.items():
            if prep.sha256(self.attempt / name) != digest:
                raise ValueError(f"{name} differs from its pinned SHA256")
            if "/" not in name and receipt["files"][name]["sha256"] != digest:
                raise ValueError(f"receipt.json disagrees with the pinned SHA256 of {name}")
        self.lock = self.work / "lock"
        prep.restore_lock(self.attempt / "input-lock.zip", self.lock, PINNED["input-lock.zip"], self.work)
        self.external = safe_unzip(self.attempt / "external-lock.zip", self.work / "external")
        fx.verify_external(self.external, self.lock)
        self.results = Results(self.attempt / RESULTS_ZIP)
        nomination = json.loads(self.results.text("reports/nomination/nomination.json"))
        if nomination["nominee"] != NOMINEE:
            raise ValueError("sealed nominee differs")
        self.sizes = ev.image_sizes(self.lock)
        self.roster = prep.read_csv(self.lock / "nine_way_roster.csv")
        self.queries = prep.read_csv(self.external / "external_queries.csv")
        inventory = {r["path"]: r for r in prep.read_csv(self.external / "external_inventory.csv")}
        self.patient = {r["path"]: inventory[r["path"]]["patient_id"] for r in self.queries if r["source"] == "OCTDL"}
        self.ext_sizes = {p: (int(r["width"]), int(r["height"])) for p, r in inventory.items()}
        self._episodes: dict[str, list[ev.Episode]] = {}
        self._features: dict[str, ev.Features] = {}

    def episodes(self, phase: str) -> list[ev.Episode]:
        if phase not in self._episodes:
            self._episodes[phase] = ev.episodes(self.lock, phase)
        return self._episodes[phase]

    def features(self, arm: str) -> ev.Features:
        """Locked study features, verified against the input lock (fm_evaluate.open_features)."""
        if arm not in self._features:
            folder = safe_unzip(self.attempt / f"features-{arm}.zip", self.work / "features" / arm)
            self._features[arm] = ev.open_features(self.lock, folder, arm)
        return self._features[arm]

    def external_features(self, arm: str) -> ev.Features:
        """Locked external features, verified against both locks (fm_external.external_features)."""
        key = f"{arm}-external"
        if key not in self._features:
            folder = safe_unzip(self.attempt / f"features-{key}.zip", self.work / "features" / key)
            self._features[key] = fx.external_features(self.lock, folder, self.external, arm)
        return self._features[key]

    def matched(self, arm: str, phase: str, classifier: str = "prototype") -> dict[tuple[int, int, int], Cell]:
        """Saved cells, each checked against the locked episode's query order and labels."""
        cells = self.results.cells(arm, phase, classifier)
        planned = {(e.fold, e.k, e.draw): e for e in self.episodes(phase)}
        for key, cell in cells.items():
            e = planned[key]
            if cell.queries != e.queries or cell.labels != e.labels or cell.classes != fm.RARE:
                raise ValueError(f"{arm} {phase} {key}: saved queries differ from the locked episode")
        if classifier == "prototype" and set(cells) != set(planned):
            raise ValueError(f"{arm} {phase}: prototype cells differ from the locked roster")
        return cells

    def kermany_normals(self) -> tuple[set[str], set[str]]:
        """All 250 Kermany NORMAL test images, and the 169 whose pid is in no common fitting row (A2 A-5)."""
        fitted = {ev.kermany_pid(r["path"]) for r in self.roster if r["role"] == "fitting" and r["label"] in COMMON}
        normals = {r["path"] for r in self.roster if r["role"] == "query" and r["label"] == "NORMAL"}
        return normals, {p for p in normals if ev.kermany_pid(p) not in fitted}


def pooled(cells: Mapping[tuple[int, int, int], Cell], order: Sequence[str],
           value: Callable[[Cell], np.ndarray] = lambda c: c.correct) -> dict[tuple[int, int], np.ndarray]:
    """Development: per (K, draw), one vector over all 124 queries in fixed order (folds pooled)."""
    position = {q: i for i, q in enumerate(order)}
    out: dict[tuple[int, int], np.ndarray] = {}
    for (fold, k, draw), cell in cells.items():
        row = out.setdefault((k, draw), np.zeros(len(order), dtype=bool))
        row[[position[q] for q in cell.queries]] = value(cell)
    return out


def dev_order(study: Study) -> tuple[list[str], list[str]]:
    rows = prep.read_csv(study.lock / "queries_development.csv")
    return [r["path"] for r in rows], [r["label"] for r in rows]


# =================================================================== analyses ===
def p0(study: Study) -> tuple[list[str], list[list]]:
    """Recall per disease: development, prototype, folds pooled, mean over draws, then over K."""
    order, labels = dev_order(study)
    rows = []
    for arm in ARMS:
        per = pooled(study.matched(arm, "development"), order)
        means = {k: np.mean([class_recalls(v, labels, fm.RARE) for (kk, _), v in per.items() if kk == k], axis=0) for k in KS}
        for i, c in enumerate([*fm.RARE, "macro"]):
            vals = [float(means[k][i]) if c != "macro" else float(means[k].mean()) for k in KS]
            rows.append([arm, c, *vals, float(np.mean(vals))])
    return ["arm", "class", *[f"k{k}" for k in KS], "mean_k1_2_4_8"], rows


def p8(study: Study) -> tuple[list[str], list[list]]:
    """CSC against MH on development images; A2 A-1 rule; folds pooled; size-only restricted the same way."""
    order, labels = dev_order(study)
    keep = np.isin(labels, ("CSC", "MH"))
    sub_labels = np.asarray(labels)[keep]
    pair = ("CSC", "MH")
    scores: dict[str, dict[tuple[int, int], float]] = {}
    per_class: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for arm in ARMS:
        hits = pooled(study.matched(arm, "development"), order,
                      lambda c: np.array(fm.RARE)[two_way_choice(c.scores, 0, 1)] == np.array(c.labels))
        per_class[arm] = {key: class_recalls(v[keep], sub_labels, pair) for key, v in hits.items()}
        scores[arm] = {key: float(r.mean()) for key, r in per_class[arm].items()}
    size: dict[tuple[int, int], np.ndarray] = {}
    totals = np.array([sum(1 for x in labels if x == c) for c in pair])
    for e in study.episodes("development"):
        size[e.k, e.draw] = size.get((e.k, e.draw), 0) + ev.size_hits(e, study.sizes, pair)
    per_class["size_only"] = {key: v / totals for key, v in size.items()}
    scores["size_only"] = {key: float(v.mean()) for key, v in per_class["size_only"].items()}
    octid = {(r["arm"], r["k"]): float(r["macro_mean"]) for r in study.results.table("reports/external/external_two_way.csv")}
    rows = []
    for arm in (*ARMS, "size_only"):
        for k, values in by_k(scores[arm], KS):
            cls = np.mean([v for (kk, _), v in per_class[arm].items() if kk == k], axis=0) if k != ev.K_MEAN_LABEL else \
                np.mean([v for v in per_class[arm].values()], axis=0)
            rows.append([arm, "two_way_macro_recall", k, *summary(values), float(cls[0]), float(cls[1]),
                         octid.get((arm, str(k)))])
    for k, values in by_k({key: scores[NOMINEE][key] - scores[fm.REFERENCE][key] for key in scores[NOMINEE]}, KS):
        rows.append([f"{NOMINEE}-{fm.REFERENCE}", "paired_difference", k, *summary(values), None, None,
                     _octid_diff(octid, k)])
    header = ["arm", "quantity", "k", *SUMMARY, "recall_CSC_mean", "recall_MH_mean", "locked_OCTID_two_way_macro_mean"]
    return header, rows


def _octid_diff(octid: Mapping[tuple[str, str], float], k) -> float | None:
    a, b = octid.get((NOMINEE, str(k))), octid.get((fm.REFERENCE, str(k)))
    return None if a is None or b is None else a - b


def p7(study: Study) -> tuple[list[str], list[list]]:
    """Z-score ensembles (E2 MIRAGE+DINOv2, E3 +Inception) on development (pooled) and the hold-out."""
    groups = {"MIRAGE": ("MIRAGE",), "DINOv2": ("DINOv2",), "Inception": ("Inception",),
              "E2 MIRAGE+DINOv2": ("MIRAGE", "DINOv2"), "E3 MIRAGE+DINOv2+Inception": ("MIRAGE", "DINOv2", "Inception")}
    order, labels = dev_order(study)
    rows = []
    for phase in ("development", "reporting"):
        cells = {arm: study.matched(arm, phase) for arm in ("MIRAGE", "DINOv2", "Inception")}
        ks = sorted({k for _, k, _ in cells["MIRAGE"]})
        values = {}
        for name, members in groups.items():
            def fused(key, members=members):
                ref = cells[members[0]][key]
                return np.array(fm.RARE)[zscore_fuse([cells[m][key].scores for m in members])] == np.array(ref.labels)
            if phase == "development":
                pos = {q: i for i, q in enumerate(order)}
                per: dict[tuple[int, int], np.ndarray] = {}
                for key in cells["MIRAGE"]:
                    row = per.setdefault((key[1], key[2]), np.zeros(len(order), dtype=bool))
                    row[[pos[q] for q in cells["MIRAGE"][key].queries]] = fused(key)
                values[name] = {key: macro(v, labels, fm.RARE) for key, v in per.items()}
            else:
                values[name] = {(k, d): macro(fused((f, k, d)), cells["MIRAGE"][f, k, d].labels, fm.RARE)
                                for (f, k, d) in cells["MIRAGE"]}
            for k, v in by_k(values[name], ks):
                rows.append([phase, name, "macro_recall", k, *summary(v)])
        for name in list(groups)[3:]:
            diff = {key: values[name][key] - values["MIRAGE"][key] for key in values["MIRAGE"]}
            for k, v in by_k(diff, ks):
                rows.append([phase, f"{name} minus MIRAGE", "paired_difference", k, *summary(v)])
    return ["phase", "model", "quantity", "k", *SUMMARY], rows


def p6(study: Study) -> tuple[list[str], list[list]]:
    """Logistic probe minus prototype on the locked run, paired on the first 25 draws."""
    order, labels = dev_order(study)
    rows = []
    for arm in ARMS:
        proto, logit = study.matched(arm, "development"), study.matched(arm, "development", "logistic")
        p = pooled({k: c for k, c in proto.items() if k[1] == 4 and k[2] < 25}, order)
        q = pooled(logit, order)
        pv, qv = [macro(p[4, d], labels, fm.RARE) for d in range(25)], [macro(q[4, d], labels, fm.RARE) for d in range(25)]
        rows.append([arm, "development", "macro_recall", 4, float(np.mean(pv)), float(np.mean(qv)),
                     *summary(list(np.subtract(qv, pv)))])
        proto, logit = study.matched(arm, "reporting"), study.matched(arm, "reporting", "logistic")
        for k in sorted({k for _, k, _ in logit}):
            pv = [macro(proto[0, k, d].correct, proto[0, k, d].labels, fm.RARE) for d in range(25)]
            qv = [macro(logit[0, k, d].correct, logit[0, k, d].labels, fm.RARE) for d in range(25)]
            rows.append([arm, "hold-out", "macro_recall", k, float(np.mean(pv)), float(np.mean(qv)),
                         *summary(list(np.subtract(qv, pv)))])
        for source, classes in (("OCTID", ("CSC", "MH")), ("OCTDL", ("MH",))):
            proto = study.results.cells(arm, f"external/{source}")
            logit = study.results.cells(arm, f"external/{source}", "logistic")
            for c in classes:
                for k in sorted({k for _, k, _ in logit}):
                    def share(cell):
                        rows_c = np.array(cell.labels) == c
                        return float(cell.correct[rows_c].mean())
                    pv = [share(proto[0, k, d]) for d in range(25)]
                    qv = [share(logit[0, k, d]) for d in range(25)]
                    rows.append([arm, f"{source} {c}", "five_way_share_correct", k, float(np.mean(pv)),
                                 float(np.mean(qv)), *summary(list(np.subtract(qv, pv)))])
    return ["arm", "set", "quantity", "k", "prototype_mean", "logistic_mean",
            "n_pairs", "diff_mean", "diff_sd", "diff_p05", "diff_p95"], rows


def nine_way_sets(study: Study) -> list[tuple[str, str, Cell, np.ndarray]]:
    """(arm, set name, cell, row mask) for every nine-way query set the audit reports."""
    normals, disjoint = study.kermany_normals()
    out = []
    for arm in ARMS:
        ker = study.results.nine_way(arm, "Kermany")
        q = np.array(ker.queries)
        out.append((arm, "Kermany patient-disjoint", ker, np.isin(q, sorted(disjoint))))
        out.append((arm, "Kermany all test", ker, np.array([lab in COMMON for lab in ker.labels])))
        out.append((arm, "Legacy 22", ker, np.array([lab in fm.RARE for lab in ker.labels])))
        for source in ("OCTID", "OCTDL"):
            cell = study.results.nine_way(arm, source)
            out.append((arm, source, cell, np.ones(len(cell.queries), dtype=bool)))
    return out


def p1_bounds(study: Study) -> tuple[list[str], list[list]]:
    """Clopper-Pearson 95% bounds for every nine-way count: per set and true class, the correct
    class, NORMAL, a common disease (CNV, DME, DRUSEN) and any rare class. OCTDL counts are
    flagged as clustered."""
    rows = []
    for arm, name, cell, mask in nine_way_sets(study):
        labels, pred, queries = np.array(cell.labels)[mask], np.array(cell.predicted)[mask], np.array(cell.queries)[mask]
        for c in [x for x in fm.NINE if x in set(labels)]:
            sel = labels == c
            n = int(sel.sum())
            if name == "OCTDL":
                patients = len({study.patient[p] for p in queries[sel]})
                flag = f"clustered: {n} images from {patients} patients; exact bound treats images as independent"
            elif name.startswith("Kermany"):
                patients, flag = len({ev.kermany_pid(p) for p in queries[sel]}), ""
            else:
                patients, flag = None, ""
            for count, hit in (("correct_class", pred[sel] == c), ("called_NORMAL", pred[sel] == "NORMAL"),
                               ("called_common_disease", np.isin(pred[sel], COMMON[1:])),
                               ("called_any_rare", np.isin(pred[sel], fm.RARE))):
                if c == "NORMAL" and count == "called_NORMAL":
                    continue  # the same count as correct_class
                rows.append([arm, name, c, count, *bound_row(int(hit.sum()), n), patients, flag])
    return ["arm", "set", "true_class", "count", "k", "n", "rate", "cp95_low", "cp95_high", "patients", "flag"], rows


def p1_loo(study: Study) -> tuple[list[str], list[list], list[str], list[list]]:
    """MIRAGE minus Inception, mean macro recall over K = 1-8, leaving out each query in turn."""
    out, summ = [], []
    order, labels = dev_order(study)
    sets = []
    a = pooled(study.matched(NOMINEE, "development"), order)
    b = pooled(study.matched(fm.REFERENCE, "development"), order)
    keys = sorted(a)
    sets.append(("development (124, folds pooled)", order, labels, keys, a, b))
    ra, rb = study.matched(NOMINEE, "reporting"), study.matched(fm.REFERENCE, "reporting")
    keys_r = sorted((k, d) for _, k, d in ra if k in KS)
    first = ra[0, 1, 0]
    sets.append(("hold-out (22)", list(first.queries), list(first.labels), keys_r,
                 {(k, d): ra[0, k, d].correct for k, d in keys_r}, {(k, d): rb[0, k, d].correct for k, d in keys_r}))
    for name, queries, labs, keys, ca, cb in sets:
        A, B = np.array([ca[x] for x in keys]), np.array([cb[x] for x in keys])
        ks_rows = np.array([k for k, _ in keys])
        full = mean_over_k(np.array([macro(r, labs, fm.RARE) for r in A]), ks_rows) - \
            mean_over_k(np.array([macro(r, labs, fm.RARE) for r in B]), ks_rows)
        diffs = loo_differences(A, B, labs, fm.RARE, ks_rows)
        for q, lab, d in zip(queries, labs, diffs):
            out.append([name, q, lab, float(d), float(d - full)])
        lo, hi = int(np.argmin(diffs)), int(np.argmax(diffs))
        summ.append([name, len(queries), float(full), float(diffs[lo]), queries[lo], float(diffs[hi]), queries[hi]])
    return (["set", "left_out", "true_class", "difference", "change_from_full"], out,
            ["set", "n_left_out", "full_difference", "smallest", "smallest_image", "largest", "largest_image"], summ)


def p4(study: Study) -> tuple[list[str], list[list], list[str], list[list]]:
    """Development queries split by whether the size-only rule names them correctly in the same episode."""
    size_right = {}
    for e in study.episodes("development"):
        pred = fm.size_predict([np.array([study.sizes[p] for p in paths]) for paths in e.supports],
                               np.array([study.sizes[q] for q in e.queries]))
        size_right[e.fold, e.k, e.draw] = np.array(fm.RARE)[pred] == np.array(e.labels)
    rows, entries = [], []
    counts: dict[tuple[int, str, str], int] = {}
    for arm in ARMS:
        cells = study.matched(arm, "development")
        values: dict[tuple[int, str], list[float]] = {}
        recalls: dict[tuple[int, str, str], list[float]] = {}
        for key, cell in cells.items():
            right = size_right[key]
            for group, mask in (("size_right", right), ("size_wrong", ~right)):
                labs = np.array(cell.labels)[mask]
                value, present = present_macro(cell.correct[mask], labs, fm.RARE)
                if value is None:
                    continue
                values.setdefault((key[1], group), []).append(value)
                for c, r in zip(present, class_recalls(cell.correct[mask], labs, present)):
                    recalls.setdefault((key[1], group, c), []).append(float(r))
                    if arm == ARMS[0]:
                        counts[key[1], group, c] = counts.get((key[1], group, c), 0) + 1
        for group in ("size_right", "size_wrong"):
            per_k = [float(np.mean(values[k, group])) for k in KS]
            for k in KS:
                rows.append([arm, group, k, len(values[k, group]), len(cells) // len(KS), *ev.draw_summary(values[k, group])])
            rows.append([arm, group, ev.K_MEAN_LABEL, None, None, float(np.mean(per_k)), None, None, None])
            for c in fm.RARE:
                for k in KS:
                    v = recalls.get((k, group, c), [])
                    entries.append([arm, group, c, k, len(v), len(cells) // len(KS), float(np.mean(v)) if v else None])
    total = sum(v.size for v in size_right.values())
    right = sum(int(v.sum()) for v in size_right.values())
    rows.append(["size_only", "share_of_query_cells_size_right", None, right, total, right / total, None, None, None])
    return (["arm", "group", "k", "episodes_used", "episodes_planned", "mean_episode_macro", "sd", "p05", "p95"], rows,
            ["arm", "group", "class", "k", "episodes_entered", "episodes_planned", "mean_recall"], entries)


def p5(study: Study, cutoff: bool = True) -> tuple[list[str], list[list], list[str], list[list], list[str], list[list]]:
    """Normal-scan audit: class received and top-two gap per normal scan; then the per-model cut-off."""
    normals, disjoint = study.kermany_normals()
    scans, summ, cuts = [], [], []
    for arm in ARMS:
        sets = []
        ker = study.results.nine_way(arm, "Kermany")
        sets.append(("Kermany patient-disjoint (calibration)", ker, np.isin(ker.queries, sorted(disjoint))))
        sets.append(("Kermany other normals", ker, np.isin(ker.queries, sorted(normals - disjoint))))
        for source in ("OCTID", "OCTDL"):
            cell = study.results.nine_way(arm, source)
            sets.append((source, cell, np.array(cell.labels) == "NORMAL"))
        gaps = {}
        for name, cell, mask in sets:
            if cell.classes != fm.NINE:
                raise ValueError("nine-way class order differs")
            top, second, gap = top_two_gap(cell.scores[mask])
            gaps[name] = (gap, np.array(fm.NINE)[top])
            for q, t, s, g in zip(np.array(cell.queries)[mask], top, second, gap):
                scans.append([arm, name, q, study.patient.get(q, ev.kermany_pid(q) if name.startswith("Kermany") else ""),
                              fm.NINE[t], destination(fm.NINE[t]), fm.NINE[s], float(g)])
            dest = np.array([destination(x) for x in np.array(fm.NINE)[top]])
            for d in ("NORMAL", "common", "rare", "all"):
                sel = np.ones(len(dest), bool) if d == "all" else dest == d
                g = gap[sel]
                summ.append([arm, name, d, int(sel.sum()), len(dest),
                             *(np.percentile(g, [5, 25, 50, 75, 95]).tolist() if len(g) else [None] * 5)])
        if cutoff:
            calib = "Kermany patient-disjoint (calibration)"
            cut, below = not_sure_cutoff(gaps[calib][0])
            for name in (calib, "OCTID", "OCTDL"):
                gap, chosen = gaps[name]
                sure = gap >= cut
                dest = np.array([destination(x) for x in chosen])
                cuts.append([arm, name, cut, below if name == calib else None, len(gap), int((~sure).sum()),
                             int((dest == "rare").sum()), int(((dest == "rare") & sure).sum()),
                             int((dest == "NORMAL").sum()), int(((dest == "NORMAL") & sure).sum()),
                             int((dest == "common").sum()), int(((dest == "common") & sure).sum()),
                             "calibration only, not a result" if name == calib else "result"])
    return (["arm", "set", "path", "patient", "received", "destination", "second", "top_two_gap"], scans,
            ["arm", "set", "destination", "n", "n_set", "gap_p05", "gap_p25", "gap_median", "gap_p75", "gap_p95"], summ,
            ["arm", "set", "cutoff_gap", "kermany_below_cutoff", "normals", "not_sure",
             "false_rare", "false_rare_sure", "called_normal", "called_normal_sure", "called_common", "called_common_sure",
             "role"], cuts)


def r1(study: Study) -> tuple[list[str], list[list]]:
    """False-rare rate with, beside it, the share of external rare images the same head calls any rare class."""
    rows = []
    _, disjoint = study.kermany_normals()
    for arm in ARMS:
        ker = study.results.nine_way(arm, "Kermany")
        sel = np.isin(ker.queries, sorted(disjoint))
        frr = int(np.isin(np.array(ker.predicted)[sel], fm.RARE).sum())
        rows.append([arm, "Kermany patient-disjoint", frr, int(sel.sum()), frr / sel.sum(), None, None, None, None, None])
        for source, classes in (("OCTID", ("CSC", "MH")), ("OCTDL", ("MH",))):
            cell = study.results.nine_way(arm, source)
            labels, pred = np.array(cell.labels), np.array(cell.predicted)
            n = int((labels == "NORMAL").sum())
            k = int(np.isin(pred[labels == "NORMAL"], fm.RARE).sum())
            for c in classes:
                sel = labels == c
                any_rare = int(np.isin(pred[sel], fm.RARE).sum())
                rows.append([arm, source, k, n, k / n, c, int(sel.sum()), any_rare, any_rare / sel.sum(),
                             int((pred[sel] == c).sum())])
    return ["arm", "source", "false_rare", "normals", "false_rare_rate", "rare_class", "rare_images",
            "rare_called_any_rare", "share_called_any_rare", "rare_called_own_class"], rows


# ======================================================================= P9 ===
P9_NS = 9101  # post hoc stream namespace for the P9 bootstrap; seed 42 as section 8 fixes
P9_BOOT = 2000
P9_FPR = 0.05
P9_ROW = "| P9 | Rare against normal, with no cut-off |"  # the protocols.md row whose commit timestamps P9


def rare_margin(scores: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """P9 score: the largest cosine to the five rare classes minus the largest to the four common classes."""
    cls = list(classes)
    rare, common = [cls.index(c) for c in fm.RARE], [cls.index(c) for c in COMMON]
    return np.asarray(scores)[:, rare].max(axis=1) - np.asarray(scores)[:, common].max(axis=1)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC of positives above negatives; ties count one half (average ranks)."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("AUROC needs at least one positive and one negative")
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def share_above_normal_cut(pos: np.ndarray, neg: np.ndarray, fpr: float = P9_FPR) -> float:
    """Share of positives strictly above the score that at most `fpr` of the negatives exceed.
    The cut is read from the same negatives, so the value is an upper bound, never a result."""
    neg = np.sort(np.asarray(neg, dtype=np.float64))[::-1]
    cut = neg[min(int(math.floor(fpr * len(neg))), len(neg) - 1)]
    return float((np.asarray(pos) > cut).mean())


def unit_index(units: Sequence[str]) -> list[np.ndarray]:
    """Item positions grouped by unit (patient or image), in first-seen order."""
    groups: dict[str, list[int]] = {}
    for i, u in enumerate(units):
        groups.setdefault(u, []).append(i)
    return [np.array(v) for v in groups.values()]


def resample_units(groups: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One cluster-bootstrap draw: units with replacement, every item of a drawn unit kept."""
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


@dataclass(frozen=True)
class P9Set:
    """One rare-against-normal comparison: per arm, scores and argmax calls in a shared item order."""
    name: str
    note: str
    pos_units: tuple[str, ...]
    neg_units: tuple[str, ...]
    pos: Mapping[str, np.ndarray]
    neg: Mapping[str, np.ndarray]
    pos_called_rare: Mapping[str, np.ndarray]
    neg_called_rare: Mapping[str, np.ndarray]


P9_HEADER = ["arm", "set", "note", "rare_scans", "rare_units", "normal_scans", "normal_units", "auroc",
             "boot95_low", "boot95_high", "argmax_rare_called_any_rare", "argmax_normals_called_rare",
             "rare_share_above_5pct_normal_cut_upper_bound"]
P9_DIFF_HEADER = ["comparison", "set", "auroc_difference", "boot95_low", "boot95_high", "boot_share_above_zero"]


def p9_table(sets: Sequence[P9Set], arms: Sequence[str], n_boot: int = P9_BOOT,
             seed: Sequence[int] = (42, P9_NS)) -> tuple[list[list], list[list]]:
    """AUROC per arm and set with unit-bootstrap intervals; MIRAGE minus each other arm on the same draws."""
    rng = np.random.default_rng(list(seed))
    rows, diffs = [], []
    for s in sets:
        pos_g, neg_g = unit_index(s.pos_units), unit_index(s.neg_units)
        draws = [(resample_units(pos_g, rng), resample_units(neg_g, rng)) for _ in range(n_boot)]
        point, boot = {}, {}
        for arm in arms:
            p, n = s.pos[arm], s.neg[arm]
            point[arm] = auroc(p, n)
            boot[arm] = np.array([auroc(p[i], n[j]) for i, j in draws])
            lo, hi = np.percentile(boot[arm], [2.5, 97.5])
            rows.append([arm, s.name, s.note, len(p), len(pos_g), len(n), len(neg_g), point[arm], lo, hi,
                         float(np.mean(s.pos_called_rare[arm])), float(np.mean(s.neg_called_rare[arm])),
                         share_above_normal_cut(p, n)])
        if NOMINEE in arms:
            for other in [a for a in arms if a != NOMINEE]:
                d = boot[NOMINEE] - boot[other]
                lo, hi = np.percentile(d, [2.5, 97.5])
                diffs.append([f"{NOMINEE} minus {other}", s.name, point[NOMINEE] - point[other], lo, hi, float((d > 0).mean())])
    return rows, diffs


def p9_sets(cells: Mapping[str, Mapping[str, Cell]], disjoint: set[str], patient: Mapping[str, str]) -> list[P9Set]:
    """The P9 sets from nine-way cells (cells[arm][source], source Kermany / OCTID / OCTDL)."""
    specs = [
        ("OCTID CSC against OCTID NORMAL", "OCTID", ("CSC",), "same clinic; image units (no patient IDs)"),
        ("OCTID MH against OCTID NORMAL", "OCTID", ("MH",), "same clinic; image units (no patient IDs)"),
        ("OCTID CSC+MH against OCTID NORMAL", "OCTID", ("CSC", "MH"), "same clinic; image units (no patient IDs)"),
        ("OCTDL MH against OCTDL NORMAL", "OCTDL", ("MH",), "same clinic; patient units"),
        ("Legacy 22 against 169 patient-disjoint Kermany normals", "Kermany", fm.RARE,
         "different sources: source and disease are mixed; rare image units, Kermany patient units"),
    ]
    arms = list(cells)
    out = []
    for name, source, positives, note in specs:
        ref = cells[arms[0]][source]
        for arm in arms[1:]:
            if cells[arm][source].queries != ref.queries or cells[arm][source].labels != ref.labels:
                raise ValueError(f"P9 {source}: {arm} has a different query order")
        q, lab = np.array(ref.queries), np.array(ref.labels)
        pos = np.isin(lab, positives)
        neg = np.isin(q, sorted(disjoint)) if source == "Kermany" else lab == "NORMAL"
        if source == "OCTDL":
            unit = lambda p: patient[p]  # noqa: E731
        elif source == "Kermany":
            unit = lambda p: p if p.startswith("data/raw/rare/") else ev.kermany_pid(p)  # noqa: E731
        else:
            unit = lambda p: p  # noqa: E731
        margin = {a: rare_margin(cells[a][source].scores, cells[a][source].classes) for a in arms}
        called = {a: np.isin(np.array(cells[a][source].predicted), fm.RARE) for a in arms}
        out.append(P9Set(name, note, tuple(unit(p) for p in q[pos]), tuple(unit(p) for p in q[neg]),
                         {a: margin[a][pos] for a in arms}, {a: margin[a][neg] for a in arms},
                         {a: called[a][pos] for a in arms}, {a: called[a][neg] for a in arms}))
    return out


def p9(study: Study) -> tuple[list[str], list[list], list[str], list[list]]:
    """P9 on the locked nine-way scores of all four arms."""
    cells = {arm: {s: study.results.nine_way(arm, s) for s in ("Kermany", "OCTID", "OCTDL")} for arm in ARMS}
    rows, diffs = p9_table(p9_sets(cells, study.kermany_normals()[1], study.patient), ARMS)
    return P9_HEADER, rows, P9_DIFF_HEADER, diffs


def rule_commit(row: str, remote: str = "origin/t3/fm-code-prep") -> str:
    """Rule 1: the commit that added `row` to protocols.md, which must also be on the pushed branch."""
    here = Path(__file__).resolve().parent
    found = subprocess.check_output(["git", "log", "--reverse", "--format=%H", "-S", row, "--", "protocols.md"],
                                    cwd=here, text=True).split()
    if not found:
        raise SystemExit(f"rule 1: '{row}' is not in any commit of protocols.md; commit and push it first")
    pushed = subprocess.run(["git", "merge-base", "--is-ancestor", found[0], remote], cwd=here).returncode == 0
    if not pushed:
        raise SystemExit(f"rule 1: commit {found[0][:7]} is not on {remote}; push it first")
    return found[0]


P3_NS = 8101  # new post hoc stream namespace (member choice uses 8102); seed 42 as section 8 fixes
P3_CLASSES = ("NORMAL", "MH")  # fixed order; exact ties go to NORMAL


def p3_groups(paths: Sequence[str], patients: Mapping[str, str] | None) -> list[fm.Group]:
    """Example units: OCTDL patients (one image drawn per patient), or single OCTID images."""
    units: dict[str, list[str]] = {}
    for p in paths:
        units.setdefault(patients[p] if patients is not None else p, []).append(p)
    members = sorted(tuple(sorted(m)) for m in units.values())
    return [fm.Group(m[0], "", rank, m) for rank, m in enumerate(members)]


def p3_supports(groups_by_class: Sequence[Sequence[fm.Group]], direction: int, draw: int) -> list[list[str]]:
    """Nested example rosters for one draw: the first K of each list are the K-shot examples."""
    return [fm.draw_supports(groups, direction, ci, draw, P3_NS)[:max(KS)] for ci, groups in enumerate(groups_by_class)]


def p3(study: Study) -> tuple[list[str], list[list], list[str], list[list]]:
    """Clinic-to-clinic macular-hole test: OCTID examples to OCTDL queries, then the reverse."""
    by: dict[tuple[str, str], list[str]] = {}
    for r in study.queries:
        by.setdefault((r["source"], r["label"]), []).append(r["path"])
    directions = (("OCTID", "OCTDL"), ("OCTDL", "OCTID"))
    rosters, out = [], []
    plans = {}
    for d, (src, dst) in enumerate(directions):
        patients = study.patient if src == "OCTDL" else None
        groups = [p3_groups(by[src, c], patients) for c in P3_CLASSES]
        plans[d] = {draw: p3_supports(groups, d, draw) for draw in range(fm.N_DRAWS)}
        for draw, lists in plans[d].items():
            for c, paths in zip(P3_CLASSES, lists):
                rosters.append([f"{src} examples -> {dst} queries", c, draw, *paths])
    for arm in ARMS:
        feats = study.external_features(arm)
        for d, (src, dst) in enumerate(directions):
            queries = [p for c in P3_CLASSES for p in by[dst, c]]
            labels = np.array([c for c in P3_CLASSES for _ in by[dst, c]])
            X = feats.select(queries)
            values: dict[str, dict[tuple[int, int], np.ndarray]] = {"image": {}, "patient": {}}
            collapsed: dict[int, int] = {}
            for draw, lists in plans[d].items():
                for k in KS:
                    pred = prototype_predict([feats.select(paths[:k]) for paths in lists], X)
                    correct = np.array(P3_CLASSES)[pred] == labels
                    values["image"][k, draw] = class_recalls(correct, labels, P3_CLASSES)
                    if dst == "OCTDL":
                        values["patient"][k, draw] = patient_recall(correct, labels, [study.patient[q] for q in queries], P3_CLASSES)
                    collapsed[k] = collapsed.get(k, 0) + int(len(set(pred)) == 1)
            for unit, cells in values.items():
                if not cells:
                    continue
                macros = {key: float(v.mean()) for key, v in cells.items()}
                for k, v in by_k(macros, KS):
                    cls = np.mean([x for (kk, _), x in cells.items() if kk == k or k == ev.K_MEAN_LABEL], axis=0)
                    n = [int((labels == c).sum()) for c in P3_CLASSES] if unit == "image" else \
                        [len({study.patient[q] for q, lab in zip(queries, labels) if lab == c}) for c in P3_CLASSES]
                    out.append([arm, f"{src} examples -> {dst} queries", unit, k, *n, *summary(v),
                                float(cls[0]), float(cls[1]), collapsed.get(k) if unit == "image" else None])
    header = ["arm", "direction", "unit", "k", "n_NORMAL", "n_MH", *SUMMARY, "recall_NORMAL_mean", "recall_MH_mean",
              "collapsed_draws"]
    return header, out, ["direction", "class", "draw", *[f"s{i:02d}" for i in range(1, max(KS) + 1)]], rosters


OCTOBER = {"none": 17, "basic": 18, "gan": 19}  # results/stage4/confusion_<variant>.csv, rare diagonal of 22


def p2(study: Study) -> tuple[list[str], list[list], list[str], list[list]]:
    """October's trained heads on OCTID and OCTDL, after two gates. A failed gate is reported, not fixed."""
    here = Path(__file__).resolve().parent
    main, external = study.features(fm.REFERENCE), study.external_features(fm.REFERENCE)
    lock = prep.load_lock(study.lock)
    spec_main = lock["arms"][fm.REFERENCE]["identity"]["spec"]
    ext_identity = prep.read_json(study.work / "features" / f"{fm.REFERENCE}-external" / "identity.json")
    gate2 = ext_identity["spec"] == spec_main and \
        spec_main["recipe"]["source"].startswith("stage3_classifier.load_image") and \
        spec_main["recipe"] == extract.recipe(fm.REFERENCE) and \
        {k: ext_identity[k] for k in ("spec", "code", "environment")} == lock["arms"][fm.REFERENCE]["identity"]
    queries = [r for r in study.roster if r["role"] == "query"]
    gates, rows = [], []
    heads = {}
    for variant, expected in OCTOBER.items():
        head = np.load(here / "results" / f"stage3_{variant}" / "head.npz", allow_pickle=False)
        order_ok = tuple(head["classes"].tolist()) == fm.NINE
        heads[variant] = head
        X = main.select([r["path"] for r in queries])
        pred = np.array(head["classes"])[head_predict(head["W"], head["b"], X)]
        labels = np.array([r["label"] for r in queries])
        rare = np.isin(labels, fm.RARE)
        hits = int((pred[rare] == labels[rare]).sum())
        saved = {r["path"]: r["pred"] for r in csv.DictReader(open(here / "results" / f"stage3_{variant}" / "test_predictions.csv",
                                                                     encoding="utf-8"))}
        agree = sum(saved[r["path"]] == p for r, p in zip(queries, pred))
        gates.append([variant, "class order", "NORMAL..RP as fm_core.NINE", "match" if order_ok else "differs", order_ok])
        gates.append([variant, "gate 1: legacy 22 named", expected, hits, order_ok and hits == expected])
        gates.append([variant, "agreement with October test predictions (1,022)", len(queries), agree, None])
        gates.append([variant, "gate 2: external Inception recipe = locked arm", "same spec and loader", gate2, gate2])
    passed = all(row[4] for row in gates if row[4] is not None)
    if passed:
        for variant, head in heads.items():
            for source in ("OCTID", "OCTDL"):
                rows_s = [r for r in study.queries if r["source"] == source]
                labels = np.array([r["label"] for r in rows_s])
                pred = np.array(head["classes"])[head_predict(head["W"], head["b"], external.select([r["path"] for r in rows_s]))]
                for c in [x for x in ("NORMAL", "CSC", "MH") if x in set(labels)]:
                    sel = labels == c
                    n = int(sel.sum())
                    counts = [int((pred[sel] == c).sum()), int((pred[sel] == "NORMAL").sum()),
                              int(np.isin(pred[sel], COMMON[1:]).sum()), int(np.isin(pred[sel], fm.RARE).sum())]
                    rows.append([variant, source, c, n, *counts, *[x / n for x in counts]])
    gate_header = ["variant", "check", "expected", "observed", "pass"]
    header = ["variant", "source", "true_class", "n", "correct_class", "called_NORMAL", "called_CNV_DME_DRUSEN",
              "called_any_rare", "share_correct", "share_NORMAL", "share_common_disease", "share_any_rare"]
    return gate_header, gates, header, rows


# ======================================================================= G1 ===
G1_ARMS = ("MIRAGE", "Inception")


def score_features(study: Study, features: ev.Features, external: ev.Features) -> dict[str, list[list]]:
    """Locked episodes and nine-way roster on one feature set: the numbers G1 reports."""
    order, labels = dev_order(study)
    pos = {q: i for i, q in enumerate(order)}
    dev: dict[tuple[int, int], np.ndarray] = {}
    for e in study.episodes("development"):
        pred = prototype_predict([features.select(p) for p in e.supports], features.select(e.queries))
        row = dev.setdefault((e.k, e.draw), np.zeros(len(order), dtype=bool))
        row[[pos[q] for q in e.queries]] = np.array(fm.RARE)[pred] == np.array(e.labels)
    out = {"development": [[k, *summary(v)] for k, v in by_k({key: macro(v, labels, fm.RARE) for key, v in dev.items()}, KS)]}
    rep, counts = {}, {}
    for e in study.episodes("reporting"):
        correct = np.array(fm.RARE)[prototype_predict([features.select(p) for p in e.supports], features.select(e.queries))] \
            == np.array(e.labels)
        rep[e.k, e.draw], counts[e.k, e.draw] = macro(correct, e.labels, fm.RARE), float(correct.sum())
    ks = sorted({k for k, _ in rep})
    out["hold-out"] = [[k, *summary(v), float(np.mean([counts[k2, d] for (k2, d) in counts if k2 == k])) if k != ev.K_MEAN_LABEL
                        else None] for k, v in by_k(rep, ks)]
    fitting = [r for r in study.roster if r["role"] == "fitting"]
    protos = fm.centroids([features.select([r["path"] for r in fitting if r["label"] == c]) for c in fm.NINE])
    nine = []
    _, disjoint = study.kermany_normals()
    query = [r for r in study.roster if r["role"] == "query"]
    sets = [("Kermany patient-disjoint", [r["path"] for r in query if r["path"] in disjoint], features),
            ("Legacy 22", [r["path"] for r in query if r["label"] in fm.RARE], features)]
    for source in ("OCTID", "OCTDL"):
        for c in ("NORMAL", "CSC", "MH"):
            paths = [r["path"] for r in study.queries if r["source"] == source and r["label"] == c]
            if paths:
                sets.append((f"{source} {c}", paths, external))
    truth = {r["path"]: r["label"] for r in [*query, *study.queries]}
    for name, paths, feats in sets:
        pred = np.array(fm.NINE)[fm.predict_prototype(protos, feats.select(paths))]
        t = np.array([truth[p] for p in paths])
        nine.append([name, len(paths), int((pred == t).sum()), int((pred == "NORMAL").sum()),
                     int(np.isin(pred, COMMON[1:]).sum()), int(np.isin(pred, fm.RARE).sum())])
    out["nine-way"] = nine
    return out


def square_paths(padding: Sequence[Mapping[str, str]]) -> list[str]:
    """G1 images that were square already: padding changes none of their pixels (padding.csv rows)."""
    return [r["path"] for r in padding if r["width"] == r["height"] == r["side"] and r["left"] == r["top"] == "0"]


def padded_root(padded: Path, work: Path) -> Path:
    """The folder holding MIRAGE/ and Inception/: given as such, or the Drive g1/ folder as Colab saved it.

    In the second case each arm must have exactly one g1-<arm>-<stamp>.zip, whose SHA256 must equal
    its .zip.json receipt (fm_prepare.export); it is then unpacked into `work`.
    """
    if all((padded / arm / "complete.json").exists() for arm in G1_ARMS):
        return padded
    out = work / "g1_padded"
    for arm in G1_ARMS:
        found = sorted(padded.glob(f"g1-{arm}-*.zip"))
        if len(found) != 1:
            raise ValueError(f"{padded}: expected one g1-{arm}-<stamp>.zip, found {[p.name for p in found]}; "
                             "pass a folder holding the unzipped MIRAGE/ and Inception/ instead")
        receipt = prep.read_json(found[0].with_name(found[0].name + ".json"))
        if prep.sha256(found[0]) != receipt["sha256"]:
            raise ValueError(f"{found[0].name} differs from the SHA256 in its .json receipt")
        safe_unzip(found[0], out / arm)
    return out


def g1(study: Study, padded: Path) -> list[list]:
    """G1 on the laptop: padded features beside the locked ones, scored the same way."""
    rows = []
    size = {(r["phase"], r["k"]): float(r["mean"]) for r in study.results.table("reports/legacy/size_only_reference.csv")}
    for arm in G1_ARMS:
        folder = padded / arm
        prep.verify_complete(folder)
        identity = prep.read_json(folder / "identity.json")
        if identity.get("arm") != arm or identity.get("lock_sha256") != prep.sha256(study.lock / "input_lock.json"):
            raise ValueError(f"{arm}: G1 output belongs to a different arm or lock")
        main_inv = prep.read_csv(study.lock / "extraction_inventory.csv")
        ext_inv = prep.read_csv(study.external / "external_extraction_inventory.csv")
        dim = extract.DIMENSIONS[arm]
        pad_main = ev.Features(extract.load_features(folder / "features.npz", main_inv, dim), {r["path"]: i for i, r in enumerate(main_inv)})
        pad_ext = ev.Features(extract.load_features(folder / "features_external.npz", ext_inv, dim), {r["path"]: i for i, r in enumerate(ext_inv)})
        locked_main, locked_ext = study.features(arm), study.external_features(arm)
        with np.load(folder / "control.npz", allow_pickle=False) as control:
            check = extract.repeat_check([locked_main.select(control["paths"].tolist())], [control["feats"]])
        # Check rows: quantity/locked hold the first measure and its value, padded_label/padded the second.
        rows.append([arm, "control: unpadded re-extraction against locked features", "", "max_difference_ratio",
                     check["max_difference_ratio"], "min_cosine", check["min_cosine"], "", ""])
        # Square images reach the model exactly as at the lock, so padded minus locked is environment noise only.
        squares = square_paths(prep.read_csv(folder / "padding.csv"))
        pairs = [(ref.select(p), got.select(p)) for ref, got, p in (
            (locked_main, pad_main, [q for q in squares if q in pad_main.index]),
            (locked_ext, pad_ext, [q for q in squares if q in pad_ext.index])) if p]
        if pairs:
            same = extract.repeat_check([a for a, _ in pairs], [b for _, b in pairs])
            rows.append([arm, f"check: {same['images']} square images (padding changes no pixel) against locked features",
                         "", "max_difference_ratio", same["max_difference_ratio"], "min_cosine", same["min_cosine"], "", ""])
        repeat = identity["repeat_check"]
        rows.append([arm, f"check: repeat on Colab ({repeat['images']} padded images encoded twice)", "",
                     "max_difference_ratio", repeat["max_difference_ratio"], "min_cosine", repeat["min_cosine"], "", ""])
        rows.append([arm, "check: Colab run against the input lock", "", "code_matches_lock", identity["code_matches_lock"],
                     "environment_matches_lock", identity["environment_matches_lock"], "", ""])
        locked, new = score_features(study, locked_main, locked_ext), score_features(study, pad_main, pad_ext)
        for table in ("development", "hold-out"):
            phase = "development" if table == "development" else "reporting"
            for a, b in zip(locked[table], new[table]):
                rows.append([arm, table, a[0], "macro_recall_mean", a[2], "padded", b[2], "size_only", size.get((phase, str(a[0])))])
                if table == "hold-out" and a[-1] is not None:
                    rows.append([arm, table, a[0], "mean_correct_of_22", a[-1], "padded", b[-1], "", ""])
        for a, b in zip(locked["nine-way"], new["nine-way"]):
            for i, what in enumerate(("correct_class", "called_NORMAL", "called_CNV_DME_DRUSEN", "called_any_rare"), start=2):
                rows.append([arm, f"nine-way {a[0]} (n={a[1]})", "", what, a[i], "padded", b[i], "", ""])
    return rows


# ====================================================================== output ===
def nine_way_cells(study: Study, features: ev.Features, external: ev.Features) -> dict[str, Cell]:
    """The locked nine-way roster scored on one feature set, as Cells (for P9 on G1's padded features)."""
    fitting = [r for r in study.roster if r["role"] == "fitting"]
    protos = fm.centroids([features.select([r["path"] for r in fitting if r["label"] == c]) for c in fm.NINE])
    query = [r for r in study.roster if r["role"] == "query"]
    out = {}
    for source, rows, feats in (("Kermany", query, features),
                                ("OCTID", [r for r in study.queries if r["source"] == "OCTID"], external),
                                ("OCTDL", [r for r in study.queries if r["source"] == "OCTDL"], external)):
        paths = [r["path"] for r in rows]
        scores = fm.unit(feats.select(paths)) @ protos.T
        out[source] = Cell(tuple(paths), tuple(r["label"] for r in rows),
                           tuple(np.array(fm.NINE)[np.argmax(scores, axis=1)]), tuple(fm.NINE), scores)
    return out


def p9_padded(study: Study, padded: Path) -> tuple[list[str], list[list], list[str], list[list]]:
    """P9 on G1's padded MIRAGE and Inception-v3 features (same sets, draws and rules as P9)."""
    cells = {}
    for arm in G1_ARMS:
        folder = padded / arm
        prep.verify_complete(folder)
        main_inv = prep.read_csv(study.lock / "extraction_inventory.csv")
        ext_inv = prep.read_csv(study.external / "external_extraction_inventory.csv")
        dim = extract.DIMENSIONS[arm]
        feats = ev.Features(extract.load_features(folder / "features.npz", main_inv, dim), {r["path"]: i for i, r in enumerate(main_inv)})
        ext = ev.Features(extract.load_features(folder / "features_external.npz", ext_inv, dim), {r["path"]: i for i, r in enumerate(ext_inv)})
        cells[arm] = nine_way_cells(study, feats, ext)
    rows, diffs = p9_table(p9_sets(cells, study.kermany_normals()[1], study.patient), list(G1_ARMS))
    return P9_HEADER, rows, P9_DIFF_HEADER, diffs


G1_FILES = ("G1_pad_to_square.csv", "P9_rare_vs_normal_auroc.csv", "P9_rare_vs_normal_differences.csv",
            "P9_G1_padded_auroc.csv", "P9_G1_padded_differences.csv")


def _num(text: str, digits: int = 3) -> str:
    return "" if text in ("", None) else f"{float(text):.{digits}f}"


def _change(locked: str, padded: str, digits: int = 3) -> str:
    return f"{float(padded) - float(locked):+.{digits}f}"


def _ci(row: Mapping[str, str], value: str, low: str, high: str) -> str:
    return f"{float(row[value]):.3f} ({float(row[low]):.3f} to {float(row[high]):.3f})"


def g1_report(g1_rows: Sequence[Mapping[str, str]], p9_locked: Sequence[Mapping[str, str]],
              p9_locked_diffs: Sequence[Mapping[str, str]], p9_padded: Sequence[Mapping[str, str]],
              p9_padded_diffs: Sequence[Mapping[str, str]]) -> str:
    """G1 locked against padded, as one Markdown page, from the five CSVs (rows as read by csv.DictReader).

    Only rearranges numbers already in those files; nothing is computed except padded minus locked.
    """
    arms = list(G1_ARMS)
    by = {(r["arm"], r["table"], r["k"], r["quantity"]): r for r in g1_rows}
    label = g1_rows[0]["posthoc"]
    out = ["# G1 pad-to-square check: locked against padded (post hoc)", "",
           "Post hoc sensitivity check (protocols.md section 8, G1), outside amendment A2 A-11. **It never replaces "
           "a locked number.** MIRAGE and Inception-v3 only; every image padded with black to a square around its "
           "centre, then each model's own locked transform; scored with the locked episodes and nine-way roster.", "",
           f"Label of every row: {label}.", "",
           f"Sources: {', '.join(f'`{n}`' for n in G1_FILES)} (results/fm_v1/posthoc/). \"Locked\" P9 rows come from "
           "the locked nine-way scores; \"padded\" rows from G1's features.", "",
           "## 1. Checks (read first)", "",
           "If the control or the square images differ from the locked features by more than 1e-4 (the A2 A-6 "
           "repeat tolerance), the Colab environment changed and the padded numbers mix padding with that change.", "",
           "| Arm | Check | Value | Value |", "|---|---|---|---|"]
    for r in g1_rows:
        if r["table"].startswith(("control", "check")):
            values = []
            for name, value in ((r["quantity"], r["locked"]), (r["padded_label"], r["padded"])):
                try:
                    values.append(f"{name} {float(value):.2e}" if "ratio" in name else f"{name} {float(value):.6f}")
                except ValueError:
                    values.append(f"{name} {value}")
            out.append(f"| {r['arm']} | {r['table']} | {values[0]} | {values[1]} |")

    def k_table(table: str, quantity: str, digits: int, size_only: bool) -> list[str]:
        ks = [k for (a, t, k, q) in by if a == arms[0] and t == table and q == quantity]
        head = "| K | " + " | ".join(f"{a} locked | {a} padded | change" for a in arms) + (" | Size only |" if size_only else " |")
        lines = [head, "|---" * (1 + 3 * len(arms) + size_only) + "|"]
        for k in ks:
            cells = []
            for a in arms:
                r = by[a, table, k, quantity]
                cells += [_num(r["locked"], digits), _num(r["padded"], digits), _change(r["locked"], r["padded"], digits)]
            extra = [_num(by[arms[0], table, k, quantity]["size_only"], digits)] if size_only else []
            lines.append(f"| {k} | " + " | ".join(cells + extra) + " |")
        return lines

    out += ["", "## 2. Development: macro recall over the five rare classes (124 queries, folds pooled; mean of 100 draws)", "",
            *k_table("development", "macro_recall_mean", 3, True),
            "", "## 3. Hold-out (legacy 22)", "", "Macro recall (mean of 100 draws):", "",
            *k_table("hold-out", "macro_recall_mean", 3, True),
            "", "Images named correctly, of 22 (mean of 100 draws):", "",
            *k_table("hold-out", "mean_correct_of_22", 2, False),
            "", "## 4. Nine-way audit: where each set goes (locked → padded)", "",
            "Same fitting roster as the locked run; in the padded column both prototypes and queries come from "
            "padded features. Counts of images, locked → padded.", "",
            "| Set | Arm | Correct class | NORMAL | CNV, DME or DRUSEN | Any rare |", "|---|---|---|---|---|---|"]
    sets = list(dict.fromkeys(t for (a, t, k, q) in by if t.startswith("nine-way")))
    for s in sets:
        for a in arms:
            cells = [f"{int(float(by[a, s, '', q]['locked']))} → {int(float(by[a, s, '', q]['padded']))}"
                     for q in ("correct_class", "called_NORMAL", "called_CNV_DME_DRUSEN", "called_any_rare")]
            out.append(f"| {s[len('nine-way '):]} | {a} | " + " | ".join(cells) + " |")
    locked = {(r["arm"], r["set"]): r for r in p9_locked if r["arm"] in arms}
    padded = {(r["arm"], r["set"]): r for r in p9_padded}
    out += ["", "## 5. P9: rare against normal AUROC, no cut-off (95% unit bootstrap, 2,000 resamples)", "",
            "| Set | " + " | ".join(f"{a} locked | {a} padded" for a in arms) + " |", "|---" * (1 + 2 * len(arms)) + "|"]
    for s in dict.fromkeys(r["set"] for r in p9_padded):
        cells = [_ci(src[a, s], "auroc", "boot95_low", "boot95_high") for a in arms for src in (locked, padded)]
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    out += ["", "Rare scans called any rare, and normals called rare, at the argmax of each column's own features "
            "(locked → padded):", "",
            "| Set | " + " | ".join(f"{a} rare called rare | {a} normals called rare" for a in arms) + " |",
            "|---" * (1 + 2 * len(arms)) + "|"]
    for s in dict.fromkeys(r["set"] for r in p9_padded):
        cells = [f"{float(locked[a, s][c]):.3f} → {float(padded[a, s][c]):.3f}" for a in arms
                 for c in ("argmax_rare_called_any_rare", "argmax_normals_called_rare")]
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    comparison = f"{NOMINEE} minus {[a for a in arms if a != NOMINEE][0]}"
    ld = {r["set"]: r for r in p9_locked_diffs if r["comparison"] == comparison}
    pd_ = {r["set"]: r for r in p9_padded_diffs if r["comparison"] == comparison}
    out += ["", f"{comparison}, AUROC difference paired on the same resamples:", "",
            "| Set | Locked | Padded |", "|---|---|---|"]
    for s in pd_:
        out.append(f"| {s} | {_ci(ld[s], 'auroc_difference', 'boot95_low', 'boot95_high')} | "
                   f"{_ci(pd_[s], 'auroc_difference', 'boot95_low', 'boot95_high')} |")
    return "\n".join(out) + "\n"


def write_g1_report(out: Path) -> Path:
    tables = []
    for name in G1_FILES:
        with open(out / name, newline="", encoding="utf-8") as stream:
            tables.append(list(csv.DictReader(stream)))
    path = out / "G1_comparison.md"
    prep.write_once(path, g1_report(tables[0], tables[1], tables[2], tables[3], tables[4]).encode("utf-8"))
    return path


# ===================================================================== P10 ===
# Same-picture audit of the 146 rare images (protocols.md section 8, P10, committed in dcbb323 and pushed).
# Choices the rule leaves open, fixed here before any P10 number:
# - S1 of Yoo et al.'s supplement lists links per disease, never per file, so step 1 records each link, its
#   disease and the kind of page its URL names; no file can be tied to a link, and the review has no
#   "shared source link" pairs.
# - Greyscale = PIL convert("RGB").convert("L") of the raw file (as the A2 border rule), after its SHA256 is
#   checked against the input lock. OpenCV SIFT with default settings; brute-force L2, two nearest neighbours,
#   Lowe ratio 0.75; fewer than 4 good matches scores 0; RANSAC homography with OpenCV's default 5-pixel
#   threshold and cv2.setRNGSeed(42) before every pair; score = inlier count, matched from the image whose
#   locked path sorts first.
# - Negatives: 2,000 (rare, Kermany) pairs drawn with replacement, uniform over the 146 rare images and the
#   Kermany images of the locked nine-way roster; stream [42, 10102]. Cut-off = their largest score.
# - Synthetic positives: stream [42, 10101] in sorted path order: keep 70-90% of each side at a random
#   offset, rescale by 0.5-1.5, JPEG quality 70; recall = share scoring at or above the cut-off.
# - Review: stream [42, 10103]. Decoys come from pairs below the cut-off whose score is at most the median of
#   all 10,585 pairs and that are not otherwise selected. Item order and left/right side are random.
P10_ROW = "| P10 | Same-picture audit of the 146 rare images |"
P10_SYNTH_NS, P10_NEG_NS, P10_REVIEW_NS = 10101, 10102, 10103
P10_RATIO, P10_RANSAC_PX, P10_NEGATIVES = 0.75, 5.0, 2000
P10_CAP, P10_TOP, P10_SYNTH, P10_DECOYS = 60, 30, 10, 10
P10_MARKS = ("same picture", "same case, other slice", "different", "unsure")
S1_HEADINGS = {"CSC": "CSC", "Macular hole": "MH", "MacTel": "MacTel", "Retinitis pigmentosa": "RP",
               "Stargardt disease": "Stargardt"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp")


def s1_links(text_lines: Sequence[str], urls: Sequence[str]) -> list[tuple[str, str]]:
    """(disease, url) for every link printed in S1, in print order.

    `text_lines` is the supplement's text; a printed link may be cut by a line break, so each printed line is
    matched to the full URL of a link annotation (`urls`) that starts with it. Links before the first disease
    heading are the two dataset links ("major").
    """
    start = [i for i, line in enumerate(text_lines) if line.strip().startswith("S1. Detailed")]
    if len(start) < 2:
        raise ValueError("S1 heading not found twice (contents line and section)")
    out, heading, pending = [], "major", list(urls)
    for line in text_lines[start[1]:]:
        s = line.strip()
        if s.startswith("S2. Detailed"):
            break
        if s in S1_HEADINGS:
            heading = S1_HEADINGS[s]
        elif s.startswith("http"):
            hit = next((u for u in pending if u.startswith(s) or s.startswith(u)), None)
            if hit is None:
                raise ValueError(f"no link annotation for printed line {s[:60]}")
            pending.remove(hit)
            out.append((heading, hit))
    return out


def link_kind(url: str) -> str:
    """What a URL names, from the URL alone (pages were not opened): image, figure, case, publication or web page."""
    u = url.lower().split("?")[0]
    last = u.rstrip("/").rsplit("/", 1)[-1]
    if u.endswith(IMAGE_SUFFIXES) or "viewimage" in u or "imagebank" in u or "/images/" in u or "photo" in last:
        return "image"
    if "/figure" in u or "figure" in url.lower():
        return "figure"
    if "case" in u:
        return "case"
    if any(t in u for t in ("doi", "/article", "journal", "pmc", "pubmed", "sciencedirect", "/books/", "/chapter/",
                            "biomedcentral", "springer", "dovepress", "hindawi", "liebertpub", "jkos", "/issues/",
                            "/abstract/", "/pdf")):
        return "publication"
    return "web page"


def p10_links(pairs: Sequence[tuple[str, str]]) -> tuple[list[str], list[list]]:
    seen: dict[str, int] = {}
    rows = []
    for disease, url in pairs:
        seen[url] = seen.get(url, 0) + 1
    for disease, url in pairs:
        rows.append([disease, url, link_kind(url), seen[url],
                     "dataset link" if disease == "major" else "no file named: S1 lists links per disease"])
    return ["disease", "url", "kind_from_url", "times_listed", "files"], rows


@dataclass
class Keys:
    """SIFT keypoint coordinates and descriptors of one image."""
    points: np.ndarray
    descriptors: np.ndarray | None


def sift_keys(gray: np.ndarray) -> Keys:
    import cv2
    kps, desc = cv2.SIFT_create().detectAndCompute(gray, None)
    return Keys(np.float32([k.pt for k in kps]).reshape(-1, 2), desc)


def sift_inliers(a: Keys, b: Keys, ratio: float = P10_RATIO) -> int:
    """RANSAC homography inliers among Lowe-ratio matches from a to b (0 if fewer than 4 good matches)."""
    import cv2
    if a.descriptors is None or b.descriptors is None or len(a.descriptors) < 2 or len(b.descriptors) < 2:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.descriptors, b.descriptors, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 4:
        return 0
    src = a.points[[m.queryIdx for m in good]]
    dst = b.points[[m.trainIdx for m in good]]
    cv2.setRNGSeed(42)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, P10_RANSAC_PX)
    return 0 if mask is None else int(mask.sum())


def synthetic_positive(gray: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A re-cropped (70-90% of each side), rescaled (0.5-1.5) JPEG-70 copy of `gray`."""
    import cv2
    h, w = gray.shape
    fw, fh = rng.uniform(0.70, 0.90, 2)
    cw, ch = max(8, round(w * fw)), max(8, round(h * fh))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    scale = float(rng.uniform(0.5, 1.5))
    crop = gray[y0:y0 + ch, x0:x0 + cw]
    size = (max(8, round(cw * scale)), max(8, round(ch * scale)))
    resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), dict(keep_w=float(fw), keep_h=float(fh), x0=x0, y0=y0, scale=scale)


def p10_review(pairs: Sequence[tuple[str, str, int]], cutoff: int, synthetic: Sequence[str],
               rng: np.random.Generator) -> tuple[list[list], int]:
    """Blind review items: (item, kind, reasons, left, right, score); and how many pairs reach the cut-off.

    kind is "pair", "synthetic" (an image against its own synthetic copy, right = "synthetic:<path>") or "decoy".
    """
    ranked = sorted(pairs, key=lambda p: (-p[2], p[0], p[1]))
    reasons: dict[tuple[str, str], list[str]] = {}
    above = [p for p in ranked if p[2] >= cutoff]
    for p in above[:P10_CAP]:
        reasons.setdefault((p[0], p[1]), []).append("at or above cut-off")
    for p in ranked[:P10_TOP]:
        reasons.setdefault((p[0], p[1]), []).append(f"top {P10_TOP}")
    score = {(p[0], p[1]): p[2] for p in ranked}
    median = float(np.median([p[2] for p in ranked]))
    pool = [p for p in ranked if p[2] < cutoff and p[2] <= median and (p[0], p[1]) not in reasons]
    items = [["pair", "; ".join(r), a, b, score[a, b]] for (a, b), r in reasons.items()]
    items += [["decoy", "random low score", pool[i][0], pool[i][1], pool[i][2]]
              for i in sorted(rng.choice(len(pool), size=min(P10_DECOYS, len(pool)), replace=False))]
    items += [["synthetic", "synthetic positive", synthetic[i], f"synthetic:{synthetic[i]}", ""]
              for i in sorted(rng.choice(len(synthetic), size=min(P10_SYNTH, len(synthetic)), replace=False))]
    order = rng.permutation(len(items))
    flip = rng.random(len(items)) < 0.5
    out = []
    for n, i in enumerate(order, start=1):
        kind, why, a, b, s = items[i]
        left, right = (b, a) if flip[n - 1] else (a, b)
        out.append([n, kind, why, left, right, s])
    return out, len(above)


def p10_gray(root: Path, path: str, digest: str) -> np.ndarray:
    from PIL import Image
    if prep.sha256(root / path) != digest:
        raise ValueError(f"{path} differs from the input lock")
    with Image.open(root / path) as im:
        return np.asarray(im.convert("RGB").convert("L"))


P10_SCREEN_FILES = ("P10_screen_pairs.csv", "P10_screen_negatives.csv", "P10_screen_synthetic.csv",
                    "P10_screen_summary.csv", "P10_review_key.csv")


def p10_screen(study: Study, root: Path) -> tuple[dict[str, tuple[list[str], list[list]]], dict[str, np.ndarray]]:
    """Steps 2 and 3 of P10: every rare pair, the negative cut-off, synthetic recall and the review list.

    Returns the tables by file name and the images the review sheet needs (synthetic copies included).
    """
    digest = {r["path"]: r["sha256"] for r in prep.read_csv(study.lock / "extraction_inventory.csv")}
    rare = sorted(r["path"] for r in study.roster if r["label"] in fm.RARE)
    role = {r["path"]: ("development" if r["role"] == "fitting" else "hold-out") for r in study.roster}
    label = {r["path"]: r["label"] for r in study.roster}
    if len(rare) != 146:
        raise ValueError(f"expected 146 rare images, found {len(rare)}")
    kermany = sorted(r["path"] for r in study.roster if r["label"] in COMMON)
    gray = {p: p10_gray(root, p, digest[p]) for p in rare}
    keys = {p: sift_keys(gray[p]) for p in rare}
    pairs = [(a, b, sift_inliers(keys[a], keys[b])) for i, a in enumerate(rare) for b in rare[i + 1:]]
    pair_rows = [[a, label[a], role[a], b, label[b], role[b], s] for a, b, s in pairs]

    rng = np.random.default_rng([42, P10_NEG_NS])
    draws = [(rare[int(i)], kermany[int(j)]) for i, j in zip(rng.integers(0, len(rare), P10_NEGATIVES),
                                                            rng.integers(0, len(kermany), P10_NEGATIVES))]
    kkeys: dict[str, Keys] = {}
    neg_rows = []
    for a, b in draws:
        if b not in kkeys:
            kkeys[b] = sift_keys(p10_gray(root, b, digest[b]))
        neg_rows.append([a, b, sift_inliers(keys[a], kkeys[b])])
    cutoff = max(r[2] for r in neg_rows)

    rng = np.random.default_rng([42, P10_SYNTH_NS])
    synth_rows, images = [], {}
    for p in rare:
        copy, params = synthetic_positive(gray[p], rng)
        images[f"synthetic:{p}"] = copy
        s = sift_inliers(keys[p], sift_keys(copy))
        synth_rows.append([p, label[p], *params.values(), s, int(s >= cutoff)])
    recall = float(np.mean([r[-1] for r in synth_rows]))

    review, n_above = p10_review(pairs, cutoff, rare, np.random.default_rng([42, P10_REVIEW_NS]))
    for row in review:
        for p in row[3:5]:
            images.setdefault(p, gray.get(p))
    scores = [s for _, _, s in pairs]
    summary_rows = [["rare pairs scored", len(pairs)], ["negative pairs (rare, Kermany)", len(neg_rows)],
                    ["Kermany images in negatives", len(kkeys)], ["cut-off (largest negative score)", cutoff],
                    ["rare pairs at or above cut-off", n_above], ["rare pairs shown for that reason", min(n_above, P10_CAP)],
                    ["synthetic positives", len(synth_rows)], ["synthetic recall at cut-off", f"{recall:.3f}"],
                    ["rare pair score median", float(np.median(scores))], ["rare pair score max", max(scores)],
                    ["pairs sharing a source link", "0 (S1 names no file)"], ["review items", len(review)]]
    tables = {
        "P10_screen_pairs.csv": (["image_a", "label_a", "set_a", "image_b", "label_b", "set_b", "inliers"], pair_rows),
        "P10_screen_negatives.csv": (["rare_image", "kermany_image", "inliers"], neg_rows),
        "P10_screen_synthetic.csv": (["image", "label", "keep_w", "keep_h", "x0", "y0", "scale", "inliers",
                                      "at_or_above_cutoff"], synth_rows),
        "P10_screen_summary.csv": (["quantity", "value"], summary_rows),
        "P10_review_key.csv": (["item", "kind", "reasons", "left", "right", "inliers"], review)}
    return tables, {p: images[p] for row in review for p in row[3:5]}


def p10_sheet(review: Sequence[Sequence], images: Mapping[str, np.ndarray], width: int = 640) -> str:
    """Self-contained HTML review sheet: no label, set, score or kind is shown; marks download as CSV."""
    import base64
    from PIL import Image

    def tag(array: np.ndarray) -> str:
        im = Image.fromarray(array)
        if im.width > width:
            im = im.resize((width, max(1, round(im.height * width / im.width))))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'

    marks = "".join(f'<label><input type="radio" name="m{{n}}" value="{m}"> {m}</label>' for m in P10_MARKS)
    blocks = []
    for row in review:
        n = row[0]
        blocks.append(f'<section id="i{n}"><h2>Pair {n} of {len(review)}</h2><div class="pair">{tag(images[row[3]])}'
                      f'{tag(images[row[4]])}</div><div class="marks">{marks.format(n=n)}'
                      f'<input class="note" name="n{n}" placeholder="note (optional)"></div></section>')
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>P10 blind review</title><style>"
            "body{font-family:sans-serif;max-width:1400px;margin:auto;padding:12px}"
            ".pair{display:flex;gap:12px;align-items:flex-start}.pair img{max-width:49%;border:1px solid #888}"
            "section{border-bottom:2px solid #ccc;padding:10px 0}.marks label{margin-right:18px}"
            ".note{width:30em}#bar{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #888}"
            "</style></head><body><div id='bar'><b>P10 blind review.</b> For each pair: same picture, same case "
            "(other slice), different, or unsure. Marks are saved in this browser as you go. "
            "<button onclick='save()'>Download marks (P10_review_marks.csv)</button> <span id='count'></span></div>"
            + "".join(blocks) +
            "<script>const N=" + str(len(review)) + ";const K='p10marks';"
            "function state(){const s={};for(let n=1;n<=N;n++){const c=document.querySelector(`input[name=m${n}]:checked`);"
            "s[n]={m:c?c.value:'',t:document.querySelector(`input[name=n${n}]`).value}};return s}"
            "function upd(){const s=state();localStorage.setItem(K,JSON.stringify(s));"
            "document.getElementById('count').textContent=Object.values(s).filter(v=>v.m).length+' of '+N+' marked'}"
            "function load(){const s=JSON.parse(localStorage.getItem(K)||'{}');for(const n in s){if(s[n].m){"
            "const r=document.querySelector(`input[name=m${n}][value=\"${s[n].m}\"]`);if(r)r.checked=true}"
            "const t=document.querySelector(`input[name=n${n}]`);if(t)t.value=s[n].t||''}upd()}"
            "function save(){const s=state();const q=v=>'\"'+String(v).replace(/\"/g,'\"\"')+'\"';"
            "let c='item,mark,note\\n';for(let n=1;n<=N;n++)c+=n+','+q(s[n].m)+','+q(s[n].t)+'\\n';"
            "const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([c],{type:'text/csv'}));"
            "a.download='P10_review_marks.csv';a.click()}"
            "document.addEventListener('change',upd);document.addEventListener('input',upd);load()</script>"
            "</body></html>")


# ====================================================================== R2 ===
R2_ROW = "| R2 | Development spread over draws |"


def r2_rows(values: Mapping[str, Mapping[tuple[int, int], float]], k_scores: Mapping[str, Mapping[int, float]],
            reference: str = "Inception") -> tuple[list[str], list[list]]:
    """Per arm and K (and the K mean): spread over draws of pooled development macro recall, then each arm
    minus the reference paired on the draw. Gate: per-K means equal the nomination's k_scores within 1e-9."""
    for arm, by_draw in values.items():
        for k, expected in k_scores[arm].items():
            got = float(np.mean([v for (kk, _), v in by_draw.items() if kk == k]))
            if abs(got - expected) > 1e-9:
                raise ValueError(f"R2 gate: {arm} K={k} mean {got} differs from nomination {expected}")
    rows = []
    for arm, by_draw in values.items():
        for k, v in by_k(by_draw, KS):
            rows.append([arm, "macro_recall", k, *summary(v)])
    for arm in values:
        if arm == reference:
            continue
        diff = {key: values[arm][key] - values[reference][key] for key in values[arm]}
        for k, v in by_k(diff, KS):
            rows.append([f"{arm} minus {reference}", "paired_difference", k, *summary(v)])
    return ["arm", "quantity", "k", *SUMMARY], rows


def r2(study: Study) -> tuple[list[str], list[list]]:
    order, labels = dev_order(study)
    nomination = json.loads(study.results.text("reports/nomination/nomination.json"))
    values, k_scores = {}, {}
    for arm in ARMS:
        values[arm] = {key: macro(v, labels, fm.RARE) for key, v in pooled(study.matched(arm, "development"), order).items()}
        k_scores[arm] = {int(k): float(s) for k, s in nomination["arms"][arm]["k_scores"].items()}
    return r2_rows(values, k_scores)


def r2_label(today: str) -> str:
    return (f"pre-specified quantity (section 5), rule timestamped by commit {rule_commit(R2_ROW)[:7]} (19 Nov 2026), "
            f"computed {today} after the results were known")


P10_MATCH = ("same picture", "same case, other slice")
P10_SETS = (("marked same picture or same case", False), ("also counting unsure", True))


def p10_matched_pairs(key_rows: Sequence[Mapping[str, str]], mark_rows: Sequence[Mapping[str, str]],
                      with_unsure: bool) -> set[frozenset]:
    """Image pairs Lam marked as a match. Synthetic copies never count; decoys count like any real pair."""
    marks = {r["item"]: r["mark"] for r in mark_rows}
    if set(marks) != {r["item"] for r in key_rows}:
        raise ValueError("P10 marks and review key name different items")
    wanted = P10_MATCH + (("unsure",) if with_unsure else ())
    return {frozenset((r["left"], r["right"])) for r in key_rows
            if r["kind"] != "synthetic" and marks[r["item"]] in wanted}


def p10_development(episodes: Sequence[ev.Episode], cells: Mapping[tuple[int, int, int], Cell],
                    order: Sequence[str], labels: Sequence[str], matched: set[frozenset]
                    ) -> tuple[dict[tuple[int, int], float], int]:
    """Per (K, draw): pooled macro recall without each query that shares a match with a support of its episode
    (classes left with no query drop out, as in P4); and the number of (query, episode) exclusions."""
    position = {q: i for i, q in enumerate(order)}
    rows: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    dropped = 0
    for e in episodes:
        cell = cells[e.fold, e.k, e.draw]
        supports = {s for group in e.supports for s in group}
        correct, keep = rows.setdefault((e.k, e.draw), (np.zeros(len(order), bool), np.ones(len(order), bool)))
        for q, ok in zip(cell.queries, cell.correct):
            correct[position[q]] = ok
            if any(frozenset((q, s)) in matched for s in supports):
                keep[position[q]] = False
                dropped += 1
    lab = np.asarray(labels)
    values = {key: present_macro(c[k], lab[k], fm.RARE)[0] for key, (c, k) in rows.items()}
    return values, dropped


def p10_holdout(cells: Mapping[tuple[int, int, int], Cell], excluded: set[str]) -> dict[tuple[int, int], float]:
    """Per (K, draw): hold-out macro recall without the excluded hold-out images."""
    out = {}
    for (fold, k, draw), cell in cells.items():
        keep = np.array([q not in excluded for q in cell.queries])
        out[k, draw] = present_macro(cell.correct[keep], np.asarray(cell.labels)[keep], fm.RARE)[0]
    return out


def p10_sensitivity(study: Study, key_rows: Sequence[Mapping[str, str]], mark_rows: Sequence[Mapping[str, str]]
                    ) -> tuple[list[str], list[list], list[str], list[list]]:
    """P10 step 3 sensitivity (rule dcbb323): development and hold-out recall without marked matches, beside
    the locked values, all four arms; each FM minus Inception-v3 paired on the draw."""
    order, labels = dev_order(study)
    dev_episodes = study.episodes("development")
    development = {r["path"] for r in study.roster if r["label"] in fm.RARE and r["role"] == "fitting"}
    rows, pairs_out = [], []
    for name, with_unsure in (("locked (no exclusion)", None), *P10_SETS):
        matched = set() if with_unsure is None else p10_matched_pairs(key_rows, mark_rows, with_unsure)
        held = {p for pair in matched for p in pair if p not in development
                and any(o in development for o in pair - {p})}
        dev_vals, hold_vals, dropped = {}, {}, 0
        for arm in ARMS:
            dev_vals[arm], dropped = p10_development(dev_episodes, study.matched(arm, "development"), order, labels,
                                                     matched)
            hold_vals[arm] = p10_holdout(study.matched(arm, "reporting"), held)
        for phase, vals, excluded in (("development", dev_vals, f"{dropped} query-episode exclusions per arm"),
                                      ("hold-out", hold_vals, f"{len(held)} hold-out images excluded")):
            ks = KS if phase == "development" else sorted({k for k, _ in vals["MIRAGE"]})
            for arm in ARMS:
                present = {key: v for key, v in vals[arm].items() if v is not None}
                for k, v in by_k(present, ks):
                    rows.append([phase, name, excluded, arm, k, *summary(v)])
            for arm in ARMS:
                if arm == "Inception":
                    continue
                diff = {key: vals[arm][key] - vals["Inception"][key] for key in vals[arm]
                        if key[0] in KS and vals[arm][key] is not None and vals["Inception"][key] is not None}
                for k, v in by_k(diff, KS):
                    rows.append([phase, name, excluded, f"{arm} minus Inception", k, *summary(v)])
        if with_unsure is not None:
            for pair in sorted(tuple(sorted(p)) for p in matched):
                sets = ["development" if p in development else "hold-out" for p in pair]
                pairs_out.append([name, pair[0], sets[0], pair[1], sets[1]])
    return (["phase", "exclusion", "excluded", "arm", "k", *SUMMARY], rows,
            ["exclusion", "image_a", "set_a", "image_b", "set_b"], pairs_out)


def p10_label(today: str) -> str:
    return f"post hoc, rule timestamped by commit {rule_commit(P10_ROW)[:7]} (18 Nov 2026), computed {today}"


# ---------------------------------------------------------------- E1-D ---
# Rule E1-D (protocols.md section 8 change log, 21 Nov; Lam's decision after the E1 verdict): P10 step 3's
# sensitivity applied unchanged to the E1-M features of MIRAGE and Inception-v3. Descriptive; it cannot change
# the E1 verdict. The same code on the locked features must reproduce the locked P10_sensitivity.csv rows.
E1D_ROW = "- 21 Nov, rule E1-D (Lam's decision after the E1 verdict; written before any E1-D number)."
E1D_SETS = ("locked", "E1-M")
E1D_TOL = 1e-12
E1D_WITH = (0.063, "0.0025")  # E1-M with the pairs (E1d_success.csv), as the fixed sentences quote them
E1D_HEADER = ["phase", "feature_set", "status", "exclusion", "excluded", "comparison", "k", *SUMMARY,
              "draws_above_0", "direction", "note"]


def e1d_label(today: str) -> str:
    return (f"post hoc, rule E1-D timestamped by commit {rule_commit(E1D_ROW)[:7]} (21 Nov 2026), written after the "
            f"E1 verdict; descriptive, cannot change the E1 verdict; outside the locked protocol; never replaces a "
            f"locked result; computed {today}")


def feature_cells(episodes: Sequence[ev.Episode], select: Callable[[Sequence[str]], np.ndarray]
                  ) -> dict[tuple[int, int, int], Cell]:
    """Per episode, the cell a predictions.csv would hold: section 6 cosine prototypes on these features."""
    rare = np.array(fm.RARE)
    out = {}
    for e in episodes:
        predicted = rare[prototype_predict([select(p) for p in e.supports], select(e.queries))]
        out[e.fold, e.k, e.draw] = Cell(tuple(e.queries), tuple(e.labels), tuple(str(p) for p in predicted),
                                        tuple(fm.RARE), np.zeros((len(e.queries), 0)))
    return out


def fmt_small(x: float) -> str:
    """Three decimals, or four when the value is within 0.01 of zero (as the E1 margin is quoted)."""
    return f"{x:.4f}" if abs(x) < 0.01 else f"{x:.3f}"


def e1d_sentence(mean: float, p05: float) -> str:
    """The two fixed sentences of rule E1-D."""
    with_mean, with_p05 = E1D_WITH
    if p05 > 0:
        return (f"Without the same-picture pairs, MIRAGE's development lead over Inception-v3 under E1-M has mean "
                f"{fmt_small(mean)} and 5th percentile {fmt_small(p05)} (with them: {with_mean:.3f} and {with_p05}).")
    return (f"Without the same-picture pairs, the 5th percentile of MIRAGE minus Inception-v3 under E1-M is "
            f"{fmt_small(p05)} (with them: {with_p05}), so the E1 pass depends on episodes that contain same-picture "
            f"pairs; the E1 verdict, fixed on the locked episodes, stands.")


def e1d_check(rows: Sequence[Sequence], p10_rows: Sequence[Mapping[str, str]]) -> float:
    """Largest difference between E1-D's locked development K-mean rows and the locked P10_sensitivity.csv rows."""
    ours = {(r[3], str(r[6])): (r[8], r[10]) for r in rows
            if r[0] == "development" and r[1] == "locked" and r[5] == "MIRAGE minus Inception"}
    theirs = {(r["exclusion"], r["k"]): (float(r["mean"]), float(r["p05"])) for r in p10_rows
              if r["phase"] == "development" and r["arm"] == "MIRAGE minus Inception"}
    keys = [key for key in ours if key in theirs]
    if not keys:
        return math.inf
    return max(abs(a - b) for key in keys for a, b in zip(ours[key], theirs[key]))


def e1d_rows(dev: Mapping[str, Mapping[tuple[int, int], float | None]],
             hold: Mapping[str, Mapping[tuple[int, int], float | None]], set_name: str, status: str, name: str,
             dropped: int, held: int) -> list[list]:
    """Development (per K and K mean, with draws above 0) and hold-out (direction) rows, MIRAGE minus Inception-v3."""
    rows = []
    for phase, vals, excluded in (("development", dev, f"{dropped} query-episode exclusions per arm"),
                                  ("hold-out", hold, f"{held} hold-out images excluded")):
        diff = {key: vals[NOMINEE][key] - vals[fm.REFERENCE][key] for key in vals[NOMINEE]
                if vals[NOMINEE][key] is not None and vals[fm.REFERENCE].get(key) is not None}
        ks = KS if phase == "development" else sorted({k for k, _ in diff})
        for k, v in by_k(diff, ks):
            stats = summary(v)
            above = sum(x > 0 for x in v)
            direction = "" if phase == "development" or k == 16 else ("positive" if stats[1] > 0 else "not positive")
            note = "shown, not read" if k == 16 else ("direction only" if phase == "hold-out" else "")
            rows.append([phase, set_name, status, name, excluded, "MIRAGE minus Inception", k, *stats, above,
                         direction, note])
    return rows


def e1d(study: Study, p1: "Phase1", key_rows: Sequence[Mapping[str, str]], mark_rows: Sequence[Mapping[str, str]],
        p10_rows: Sequence[Mapping[str, str]]) -> tuple[list[str], list[list]]:
    """Rule E1-D: P10 step 3 on the locked (check) and E1-M features of MIRAGE and Inception-v3."""
    order, labels = dev_order(study)
    dev_eps, rep_eps = study.episodes("development"), study.episodes("reporting")
    development = {r["path"] for r in study.roster if r["label"] in fm.RARE and r["role"] == "fitting"}
    rows: list[list] = []
    for set_name in E1D_SETS:
        feats = {arm: p1.features(arm, set_name) for arm in (NOMINEE, fm.REFERENCE)}
        if any(f is None for f in feats.values()):
            rows.append(["development", set_name, UNAVAILABLE, "", "", "MIRAGE minus Inception", "", *[""] * 8])
            continue
        status = (INTERPRETED if all(p1.status(arm, set_name) == INTERPRETED for arm in feats) else NOT_INTERPRETED)
        dev_cells = {arm: feature_cells(dev_eps, f[0].select) for arm, f in feats.items()}
        rep_cells = {arm: feature_cells(rep_eps, f[0].select) for arm, f in feats.items()}
        for name, with_unsure in (("locked (no exclusion)", None), *P10_SETS):
            matched = set() if with_unsure is None else p10_matched_pairs(key_rows, mark_rows, with_unsure)
            held = {p for pair in matched for p in pair if p not in development
                    and any(o in development for o in pair - {p})}
            dev, hold, dropped = {}, {}, 0
            for arm in feats:
                dev[arm], dropped = p10_development(dev_eps, dev_cells[arm], order, labels, matched)
                hold[arm] = p10_holdout(rep_cells[arm], held)
            rows += e1d_rows(dev, hold, set_name, status, name, dropped, len(held))
    gap = e1d_check(rows, p10_rows)
    passed = gap <= E1D_TOL
    rows.append(["check", "locked", "pass" if passed else NOT_INTERPRETED, "", "", "locked rows against "
                 "P10_sensitivity.csv (largest difference of mean and p05)", "", "", gap, "", "", "", "", "", ""])
    if not passed:
        for r in rows:
            if r[1] == "E1-M":
                r[2] = NOT_INTERPRETED
    main = next((r for r in rows if r[0] == "development" and r[1] == "E1-M" and r[3] == P10_SETS[0][0]
                 and r[6] == ev.K_MEAN_LABEL), None)
    if main is not None and main[2] == INTERPRETED:
        rows.append(["sentence", "E1-M", INTERPRETED, P10_SETS[0][0], "", "", "", "", "", "", "", "", "", "",
                     e1d_sentence(main[8], main[10])])
    return E1D_HEADER, rows


def write(out: Path, name: str, label: str, header: list[str], rows: list[list]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    prep.write_csv(path, ["posthoc", *header], [[label, *row] for row in rows])
    return path


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip()


def run(attempt: Path, out: Path, only: set[str] | None, today: str) -> list[Path]:
    labels = labels_for(today)
    written = []
    with tempfile.TemporaryDirectory() as tmp:
        study = Study(attempt, Path(tmp))
        want = lambda name: only is None or name in only  # noqa: E731
        if want("P0"):
            written.append(write(out, "P0_per_class_recall.csv", labels["P0"], *p0(study)))
        if want("P8"):
            written.append(write(out, "P8_csc_vs_mh_development.csv", labels["P8"], *p8(study)))
        if want("P3"):
            h, rows, rh, roster = p3(study)
            written += [write(out, "P3_clinic_to_clinic_mh.csv", labels["P3"], h, rows),
                        write(out, "P3_example_rosters.csv", labels["P3"], rh, roster)]
        if want("R1"):
            written.append(write(out, "R1_rare_beside_false_rare.csv", labels["R1"], *r1(study)))
        if want("P5"):
            h1, scans, h2, summ, h3, cuts = p5(study)
            written += [write(out, "P5_normal_scans.csv", labels["P5"], h1, scans),
                        write(out, "P5_gap_summary.csv", labels["P5"], h2, summ),
                        write(out, "P5_not_sure_cutoff.csv", labels["P5"], h3, cuts)]
        if want("P1"):
            written.append(write(out, "P1_nine_way_exact_bounds.csv", labels["P1"], *p1_bounds(study)))
            h1, rows, h2, summ = p1_loo(study)
            written += [write(out, "P1_leave_one_out.csv", labels["P1"], h1, rows),
                        write(out, "P1_leave_one_out_summary.csv", labels["P1"], h2, summ)]
        if want("P4"):
            h1, rows, h2, entries = p4(study)
            written += [write(out, "P4_size_right_vs_wrong.csv", labels["P4"], h1, rows),
                        write(out, "P4_class_entries.csv", labels["P4"], h2, entries)]
        if want("P2"):
            h1, gates, h2, rows = p2(study)
            written.append(write(out, "P2_gates.csv", labels["P2"], h1, gates))
            if rows:
                written.append(write(out, "P2_october_heads_external.csv", labels["P2"], h2, rows))
        if want("P6"):
            written.append(write(out, "P6_probe_vs_prototype.csv", labels["P6"], *p6(study)))
        if want("P7"):
            written.append(write(out, "P7_ensemble.csv", labels["P7"], *p7(study)))
        if want("R2"):
            written.append(write(out, "R2_development_spread.csv", r2_label(today), *r2(study)))
        if want("P9"):
            label = p9_label(today)
            h1, rows, h2, diffs = p9(study)
            written += [write(out, "P9_rare_vs_normal_auroc.csv", label, h1, rows),
                        write(out, "P9_rare_vs_normal_differences.csv", label, h2, diffs)]
    return written


def p9_label(today: str) -> str:
    return f"post hoc, rule timestamped by commit {rule_commit(P9_ROW)[:7]} (18 Nov 2026), computed {today}"


# ============================================================ Phase-1 E1-E4 ===
# Rules: protocols.md section 8, subsection "Phase-1 controls E1-E4", committed and pushed in 4b4e883
# (20 Nov 2026) before any E number. Choices the rules leave open, fixed here before any E number:
# - Layout read from Drive phase1/ (written by fm_control_extract.py): phase1-*.zip archives, each with a
#   .zip.json receipt {"sha256": ...}; phase1-logs-* archives are not read. Inside, every feature folder has
#   complete.json, identity.json (keys arm, set in E1-M / E1-L / E1-C / E3, variant for E3, lock_sha256,
#   repeat_check, code_matches_lock, environment_matches_lock, overrides), features.npz and
#   features_external.npz. E1 has control.npz, DINOv2 also control_external.npz, and Inception E1-M
#   also decoder_check.npz. E3 has variant_inputs.csv and the full E3-ref is its control.
#   The render folder has complete.json, render.csv, decoder.csv, one normalized [n, 256] thumbnail
#   array per set in thumbs16.npz, and wide pixel_stats.csv and edge_residual.csv tables.
# - Width, height, A2 border share and near-white share of the originals come from the locked geometry.csv
#   of the locked feature bundles (identical for every arm); grey mean and SD from pixel_stats.csv (raw).
# - Thumbnail reference: nearest mean support, squared Euclidean, first class on ties; nine-way thumbnail
#   prototypes are the class means of the fitting images' thumbnails. In E4 the thumbnails use the same rule.
# - E2b "share given the same class": over all normals of the set, the share given the same class by both arms.
# - E4 per-column AUROC: the second source of the pair is the positive class; the reading uses
#   max(AUROC, 1 - AUROC), since a column has no fitted direction.
# - E1 pixel-only reading (choice fixed in code before any E number): "is above" means p05 > 0
#   across the 100 paired MIRAGE E1-M minus E1-M thumbnail K-mean differences; z is their mean.
# - E2d bootstrap streams are keyed by arm, source, feature set and property within namespace 12101.
PHASE1_HEADING = "### Phase-1 controls E1–E4 (rules fixed 20 Nov 2026, before any E number)"
E_SETS = ("E1-M", "E1-L", "E1-C")
FEATURE_SETS = ("locked", *E_SETS)
CONTROL_TOL, EFFECT_MIN = 1e-4, 1e-3  # A2 A-6 repeat tolerance; E3 "change applied" threshold
E2_BOOT_NS, E2_SAMPLE_NS, E2_ORDER_NS = 12101, 12102, 12103
E4_FOLD_NS, E4_BOOT_NS, E4_FEWSHOT_NS = 14101, 14102, 14103
E_BOOT = 2000
NOT_INTERPRETED = "environment differs from the lock; not interpreted"
UNAVAILABLE = "unavailable"
INTERPRETED = "interpreted"
E3_VARIANTS = {"MIRAGE": ("M-crop", "M-bicubic", "M-nostretch", "M-res256"),
               "RETFound": ("R-squash", "R-pad", "R-bilinear", "R-stretch", "R-norm05", "R-res128")}
E3_REF = "E3-ref"
E3_SQUARE = ("M-crop", "R-squash", "R-pad")
E3_SCOPE = "input preprocessing only; pooling and token choice are not varied"
M_PAD, M_PAD_NOTE = "M-pad", "known before this rule (G1, 18 Nov)"
RENDER_COLUMNS = ("path", "sha256", "mode", "format", "min_alpha", "width", "height", "first_left", "first_top",
                  "first_right", "first_bottom", "box_left", "box_top", "box_right", "box_bottom", "flagged")
PIXEL_STATS = ("mean", "sd", "p01", "p50", "p99", "share_le8", "share_ge250", "entropy_bits")
GEOMETRY_G = ("width", "height", "aspect_ratio", "border_share")
E2D_PROPERTIES = ("width", "height", "aspect_ratio", "border_share", "near_white_share", "grey_mean", "grey_sd")
E1_LIMITATIONS = ("the centre square drops lateral retina unequally by source and class; MIRAGE now gets 256 px of "
                  "detail enlarged to 512; E1-M does not remove effective resolution (content short side 103-496 px), "
                  "field of view, scanner appearance, compression, intensity, black-clipped background against "
                  "speckle, or Kermany's white fill wedges")
DROPPABLE = ("E2f", "E4-probe", "E2d-intervals", "E1e", "E1h")  # the laptop drop order (rules, GPU plan)
NORMAL_SETS = ("Kermany patient-disjoint", "OCTID NORMAL", "OCTDL NORMAL")
RARE_EXTERNAL = ("OCTID CSC", "OCTID MH", "OCTDL MH")


# ------------------------------------------------------------- rule 1 of E1-E4 ---
def section_text(text: str, heading: str = PHASE1_HEADING) -> str:
    """The subsection under `heading`, up to the next heading of level 2 or 3 (line endings folded to LF)."""
    lines = text.replace("\r\n", "\n").split("\n")
    if heading not in lines:
        raise ValueError(f"heading not found: {heading}")
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(("## ", "### "))), len(lines))
    return "\n".join(lines[start:end]).strip("\n") + "\n"


def section_sha256(text: str, heading: str = PHASE1_HEADING) -> str:
    import hashlib
    try:
        return hashlib.sha256(section_text(text, heading).encode("utf-8")).hexdigest()
    except ValueError:
        return "missing"


def rule_section_hash(commit: str, heading: str = PHASE1_HEADING, working: str | None = None) -> dict:
    """Rule 1: SHA256 of the subsection at the rule commit and in the working protocols.md."""
    here = Path(__file__).resolve().parent
    committed = subprocess.check_output(["git", "show", f"{commit}:protocols.md"], cwd=here).decode("utf-8")
    now = (here / "protocols.md").read_text(encoding="utf-8") if working is None else working
    a, b = section_sha256(committed, heading), section_sha256(now, heading)
    if a == "missing":
        raise ValueError(f"the registered heading is missing at {commit}")
    return dict(commit=commit, commit_sha256=a, working_sha256=b, same=a == b)


def long_date(iso: str) -> str:
    d = datetime.date.fromisoformat(iso)
    return f"{d.day} {d:%b %Y}"


def e_label(n: int, commit: str, committed: str, today: str, section: Mapping) -> str:
    """Rule 2 label; rule 1 adds a warning whenever the working subsection differs from the rule commit."""
    label = (f"post hoc, Phase-1 rule E{n} timestamped by commit {commit[:7]} ({committed}), computed {today}; "
             "outside the locked protocol; never replaces a locked result")
    if not section["same"]:
        label += (f"; WARNING: the working Phase-1 subsection differs from commit {commit[:7]} "
                  f"(SHA256 {section['working_sha256'][:12]} against {section['commit_sha256'][:12]})")
    return label


def phase1_labels(today: str) -> tuple[dict[str, str], dict]:
    commit = rule_commit(PHASE1_HEADING)
    section = rule_section_hash(commit)
    date = subprocess.check_output(["git", "show", "-s", "--format=%cs", commit], cwd=Path(__file__).resolve().parent,
                                   text=True).strip()
    return {f"E{n}": e_label(n, commit, long_date(date), today, section) for n in range(1, 5)}, section


# -------------------------------------------------------------- pure functions ---
def nearest_mean_scores(supports: Sequence[np.ndarray], queries: np.ndarray) -> np.ndarray:
    """Minus the squared Euclidean distance to each class's mean support (argmax = nearest, first on ties)."""
    if not supports or any(len(s) == 0 for s in supports):
        raise ValueError("nearest mean needs supports for every class")
    means = np.stack([np.asarray(s, dtype=np.float64).reshape(len(s), -1).mean(axis=0) for s in supports])
    q = np.asarray(queries, dtype=np.float64).reshape(len(queries), means.shape[1])
    if not np.isfinite(means).all() or not np.isfinite(q).all():
        raise ValueError("nearest mean needs finite values and supports for every class")
    return -((q[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)


def nearest_mean_predict(supports: Sequence[np.ndarray], queries: np.ndarray) -> np.ndarray:
    """fm_core.size_predict in any dimension: nearest mean support, squared Euclidean, first class on ties."""
    return np.argmax(nearest_mean_scores(supports, queries), axis=1)


def prototype_scores(supports: Sequence[np.ndarray], queries: np.ndarray) -> np.ndarray:
    """Section 6 cosine prototypes, as the locked nine-way head computes them."""
    return fm.unit(queries) @ fm.centroids(supports).T


def success_rule(d: Sequence[float], n_required: int = fm.N_DRAWS) -> dict:
    """E1's rule: p05 = numpy.percentile(d, 5) (linear) > 0; n must be n_required, else 'not evaluable'."""
    values = np.asarray(list(d), dtype=np.float64)
    out = dict(n=len(values), mean=None, sd=None, p05=None, n_positive=int((values > 0).sum()) if len(values) else 0)
    if len(values) and np.isfinite(values).all():
        out.update(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                   p05=float(np.percentile(values, 5)))
    if len(values) != n_required or out["p05"] is None:
        out["verdict"] = "not evaluable"
    else:
        out["verdict"] = "pass" if out["p05"] > 0 else "fail"
    return out


def matched_subset(sides: Mapping[str, float], labels: Mapping[str, str]) -> tuple[float, float, list[str]]:
    """E1e: L = largest class minimum, U = smallest class maximum of the content short side; queries in [L, U]."""
    by: dict[str, list[float]] = {}
    for p, lab in labels.items():
        by.setdefault(lab, []).append(float(sides[p]))
    low, high = max(min(v) for v in by.values()), min(max(v) for v in by.values())
    return low, high, [p for p in labels if low <= float(sides[p]) <= high]


def _fmt(x: float | None, digits: int = 3, sign: bool = False) -> str:
    if x is None:
        return "not available"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _clause(variant: str, main_pass: bool, other: Mapping | None, what: str) -> str:
    """The fixed E1-L / E1-C clause for an evaluable secondary variant."""
    if other is None or other["verdict"] not in ("pass", "fail"):
        return ""
    lead = "and also" if (other["verdict"] == "pass") == main_pass else "but not"
    return f", {lead} when {what} ({variant} p05 = {_fmt(other['p05'])})"


def e1_sentences(main: Mapping, locked_mean: float | None, thumb: Mapping | None,
                 e1l: Mapping | None = None, e1c: Mapping | None = None, res256_shift: float | None = None) -> str:
    """E1's fixed sentences, filled from numbers only (protocols.md, E1 'Fixed sentences')."""
    if main["verdict"] == "not evaluable":
        return "not evaluable"
    passed = main["verdict"] == "pass"
    clauses = _clause("E1-L", passed, e1l, "effective resolution is evened out") + \
        _clause("E1-C", passed, e1c, "the image is only cropped to its content box")
    if passed:
        if thumb is None or thumb["verdict"] == "not evaluable":
            return ""  # The verdict row is published even when the sentence's companion statistic is missing.
        above = "is" if thumb["p05"] > 0 else "is not"
        return ("Under the size and framing control, MIRAGE keeps a positive development advantage over Inception-v3 "
                f"(mean {_fmt(main['mean'])}, locked {_fmt(locked_mean)}; p05 {_fmt(main['p05'])}){clauses}; "
                f"it {above} above the pixel-only reference (mean {_fmt(thumb['mean'])}).")
    if res256_shift is None:
        return ""
    return (f"MIRAGE does not keep its development advantage under this control (mean {_fmt(main['mean'])}, "
            f"p05 {_fmt(main['p05'])}){clauses}; halving MIRAGE's input resolution alone moved its K mean by "
            f"{_fmt(res256_shift, sign=True)} (E3 M-res256).")


def jaccard(a: set, b: set) -> float | None:
    """|A ∩ B| / |A ∪ B|; None ("—") when both sets are empty."""
    union = a | b
    return None if not union else len(a & b) / len(union)


def failure_overlap(failed: Mapping[str, set], items: Sequence[str]) -> list[int]:
    """Number of items failed by 0, 1, ..., len(failed) arms."""
    counts = [0] * (len(failed) + 1)
    for item in items:
        counts[sum(item in f for f in failed.values())] += 1
    return counts


def same_class_share(a: Sequence[str], b: Sequence[str]) -> float:
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def tertile_shares(values: Sequence[float], failed: Sequence[bool]) -> tuple[list[float], list[int], list[float | None]]:
    """Cut points at the 1/3 and 2/3 quantiles (numpy linear) of all values; ties go to the lower tertile.
    Returns the cuts, the count and the failure share per tertile."""
    v, f = np.asarray(values, dtype=np.float64), np.asarray(failed, dtype=bool)
    cuts = np.percentile(v, [100 / 3, 200 / 3]).tolist()
    tertile = np.where(v <= cuts[0], 0, np.where(v <= cuts[1], 1, 2))
    counts = [int((tertile == t).sum()) for t in range(3)]
    return cuts, counts, [float(f[tertile == t].mean()) if counts[t] else None for t in range(3)]


def quartiles(values: Sequence[float]) -> list[float | None]:
    if not len(values):
        return [None, None, None]
    return np.percentile(np.asarray(values, dtype=np.float64), [50, 25, 75]).tolist()


def e2d_row(values: Sequence[float], failed: Sequence[bool], units: Sequence[str], rng: np.random.Generator | None,
            n_boot: int = E_BOOT) -> list:
    """One E2d row: n, median and quartiles per group, tertile shares and the AUROC of failed against not
    failed with a unit-bootstrap 95% interval (rng None: no interval). Under 5 scans in a group: counts and
    medians only."""
    v, f, u = np.asarray(values, dtype=np.float64), np.asarray(failed, dtype=bool), np.asarray(units)
    pos, neg = v[f], v[~f]
    small = len(pos) < 5 or len(neg) < 5
    qp, qn = quartiles(pos), quartiles(neg)
    if small:
        return [len(pos), len(neg), qp[0], None, None, qn[0], None, None, *[None] * 8, None, None, None, None]
    cuts, counts, shares = tertile_shares(v, f)
    point = auroc(pos, neg)
    low = high = excludes = None
    if rng is not None:
        pg, ng = unit_index(u[f].tolist()), unit_index(u[~f].tolist())
        boot = [auroc(pos[resample_units(pg, rng)], neg[resample_units(ng, rng)]) for _ in range(n_boot)]
        low, high = np.percentile(boot, [2.5, 97.5]).tolist()
        excludes = bool(low > 0.5 or high < 0.5)
    return [len(pos), len(neg), *qp, *qn, *cuts, *counts, *shares, point, low, high, excludes]


E2D_HEADER = ["n_failed", "n_not_failed", "failed_median", "failed_q1", "failed_q3", "not_failed_median",
              "not_failed_q1", "not_failed_q3", "tertile_cut_1", "tertile_cut_2", "n_tertile_1", "n_tertile_2",
              "n_tertile_3", "failure_share_tertile_1", "failure_share_tertile_2", "failure_share_tertile_3",
              "auroc_failed_vs_not", "boot95_low", "boot95_high", "interval_excludes_0_5"]


def contact_sheet_sample(groups: Mapping[tuple[str, str], Sequence[str]], anchors: Sequence[str],
                         per_group: int = 16, n_anchors: int = 8) -> list[list]:
    """E2f: 16 per (source, group) at random (all if fewer; stream 12102), 8 Kermany anchors, one random order
    (stream 12103), neutral IDs. Returns [id, path, source, group] in display order."""
    pick = np.random.default_rng([fm.SEED, E2_SAMPLE_NS])
    chosen = []
    for (source, group) in sorted(groups):
        members = sorted(groups[source, group])
        take = members if len(members) <= per_group else [members[i] for i in sorted(pick.choice(len(members), per_group, replace=False))]
        chosen += [(p, source, group) for p in take]
    members = sorted(anchors)
    chosen += [(members[i], "Kermany", "anchor") for i in sorted(pick.choice(len(members), min(n_anchors, len(members)), replace=False))]
    order = np.random.default_rng([fm.SEED, E2_ORDER_NS]).permutation(len(chosen))
    return [[f"S{i + 1:03d}", *chosen[j]] for i, j in enumerate(order)]


E2F_MARKS = ("tilted_or_curved", "low_signal_or_heavy_speckle", "retina_cut_at_edge", "bands_or_frame",
             "overlay_text_or_markers", "visible_abnormality", "scan_width")


def mark_counts(marks: Sequence[Mapping[str, str]], key: Sequence[Mapping[str, str]]) -> list[list]:
    """E2f result: counts of each mark value per group within source; groups under 8 images are flagged."""
    where = {r["id"]: (r["source"], r["group"]) for r in key}
    size: dict[tuple[str, str], int] = {}
    for s, g in where.values():
        size[s, g] = size.get((s, g), 0) + 1
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in marks:
        s, g = where[row["id"]]
        for m in E2F_MARKS:
            value = row.get(m, "").strip().lower()
            counts[s, g, m, value] = counts.get((s, g, m, value), 0) + 1
    return [[s, g, size[s, g], m, v, n, "shown in full, no comparison" if size[s, g] < 8 else ""]
            for (s, g, m, v), n in sorted(counts.items())]


def validate_marks(marks: Sequence[Mapping[str, str]], ids: Sequence[str]) -> None:
    if sorted(r["id"] for r in marks) != sorted(ids):
        raise ValueError("marks must contain exactly one row for every neutral ID")
    for row in marks:
        for name in E2F_MARKS:
            allowed = ("narrow", "wide") if name == "scan_width" else ("yes", "no", "unsure")
            if row.get(name, "").strip().lower() not in allowed:
                raise ValueError(f"{row['id']}: fill {name} with one of {', '.join(allowed)}")


def e3_interval_check(count: int, n: int, interval: Sequence[float]) -> bool:
    """Criterion (3): the false-rare count lies inside the locked 95% interval (as a rate, bounds included)."""
    return interval[0] <= count / n <= interval[1]


def ranks_with_ties(values: Mapping[str, float]) -> dict[str, int]:
    """Rank 1 = smallest; tied values share the smallest rank (description only)."""
    return {k: 1 + sum(v2 < v for v2 in values.values()) for k, v in values.items()}


def e3_criteria(mirage: Mapping[str, Sequence[float]], retfound_vs_inception: Mapping[str, Sequence[float]],
                mirage_minus_retfound: Mapping[str, Sequence[float]]) -> list[list]:
    """Criteria (1) and (2). mirage[v]: per-draw K mean of (MIRAGE v - Inception locked);
    retfound_vs_inception[v]: (RETFound v - Inception locked); mirage_minus_retfound[v]: (MIRAGE locked - RETFound v).
    A variant absent from the mapping is 'not available'; n other than 100 is 'not evaluable'."""
    rows, failed, missing = [], [], []
    for v in E3_VARIANTS["MIRAGE"]:
        if v not in mirage:
            rows.append(["(1)", "MIRAGE", v, "p05 of K mean (variant - Inception-v3 locked)", None, None, "not available"])
            missing.append(v)
            continue
        s = success_rule(mirage[v])
        ok = s["verdict"] == "pass"
        rows.append(["(1)", "MIRAGE", v, "p05 of K mean (variant - Inception-v3 locked)", s["n"], s["p05"],
                     "met" if ok else s["verdict"] if s["verdict"] == "not evaluable" else "not met"])
        if s["verdict"] == "not evaluable":
            missing.append(v)
        elif not ok:
            failed.append(v)
    rows.append(["(1)", "MIRAGE", "all four", "result", None, None,
                 ("sensitive to " + ", ".join(failed) if failed else "kept within this grid" if not missing else
                  "not evaluable for the full grid") + (f"; unavailable or not evaluable: {', '.join(missing)}" if missing else "")])
    moved = []
    for v in E3_VARIANTS["RETFound"]:
        if v not in retfound_vs_inception or v not in mirage_minus_retfound:
            rows.append(["(2)", "RETFound", v, "both conditions", None, None, "not available"])
            moved.append(v)
            continue
        a, b = np.asarray(retfound_vs_inception[v]), np.asarray(mirage_minus_retfound[v])
        n_ok = len(a) == fm.N_DRAWS and len(b) == fm.N_DRAWS
        n_ok = n_ok and np.isfinite(a).all() and np.isfinite(b).all()
        p95 = float(np.percentile(a, 95)) if len(a) and np.isfinite(a).all() else None
        p05 = float(np.percentile(b, 5)) if len(b) and np.isfinite(b).all() else None
        rows.append(["(2)", "RETFound", v, "p95 of K mean (variant - Inception-v3 locked) < 0", len(a), p95,
                     "not evaluable" if not n_ok else "met" if p95 < 0 else "not met"])
        rows.append(["(2)", "RETFound", v, "p05 of K mean (MIRAGE locked - variant) > 0", len(b), p05,
                     "not evaluable" if not n_ok else "met" if p05 > 0 else "not met"])
        if not (n_ok and p95 < 0 and p05 > 0):
            moved.append(v)
    rows.append(["(2)", "RETFound", "all six", "result", None, None,
                 "position unchanged within this grid" if not moved else f"position changed or not shown for {', '.join(moved)}"])
    return rows


def e4_reading(value: float | None) -> str:
    if value is None:
        return "not available"
    return ("near-perfectly separable" if value >= 0.95 else "strongly separable" if value >= 0.80 else
            "partly separable" if value >= 0.60 else "hardly separable")


def e4_pair_reading(a: float, b: float, low: float, high: float, control: bool = True) -> str:
    """Probe difference b - a: ceiling first, then 'reduced' only if the paired interval lies wholly below 0."""
    if a >= 0.99 and b >= 0.99:
        return "at ceiling; the difference is not read"
    if not control:
        return "descriptive comparison"
    return "the control reduced separability" if high < 0 else "no reduction shown"


def grouped_folds(y: Sequence[int], groups: Sequence[str], pair: int) -> list[np.ndarray]:
    """E4 probe folds: StratifiedGroupKFold(5, shuffle=True) by unit; the seed comes from stream (42, 14101, pair).
    Returns the test indices of each fold."""
    from sklearn.model_selection import StratifiedGroupKFold
    seed = int(np.random.default_rng([fm.SEED, E4_FOLD_NS, pair]).integers(2 ** 31 - 1))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    y = np.asarray(y)
    return [test for _, test in splitter.split(np.zeros((len(y), 1)), y, np.asarray(groups))]


def fit_logistic_raw(X: np.ndarray, y: np.ndarray, n_classes: int = 2):
    """fm_core.fit_logistic's objective and optimiser without row normalisation; columns z-scored on the
    training rows (population SD), columns with SD 0 dropped. Returns (fit, kept columns, mean, sd)."""
    from scipy.optimize import minimize
    X = np.asarray(X, dtype=np.float64)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    keep = np.flatnonzero(np.ptp(X, axis=0) > 0)
    Z = (X[:, keep] - mean[keep]) / sd[keep]
    y = np.asarray(y)
    if not np.array_equal(np.unique(y), np.arange(n_classes)):
        raise ValueError("every class must have training rows")
    theta = np.zeros(n_classes * (Z.shape[1] + 1), dtype=np.float64)
    result = minimize(fm.logistic_objective, theta, args=(Z, y, n_classes), jac=True, method="L-BFGS-B", bounds=None,
                      options=dict(maxiter=1000, maxfun=20000, maxls=20, maxcor=10, gtol=1e-6, ftol=0.0))
    value, gradient = fm.logistic_objective(result.x, Z, y, n_classes)
    worst = float(np.abs(gradient).max())
    d = Z.shape[1]
    finite = bool(np.isfinite(result.x).all() and np.isfinite(value) and np.isfinite(gradient).all())
    fit = fm.LogisticFit(result.x[:n_classes * d].reshape(n_classes, d), result.x[n_classes * d:],
                         finite and worst <= 1e-6, value, worst, str(result.message))
    return fit, keep, mean, sd


def probe_scores(X: np.ndarray, y: np.ndarray, folds: Sequence[np.ndarray], normalise_rows: bool) -> tuple[np.ndarray, int]:
    """Out-of-fold logit differences (class 1 minus class 0), pooled; and the number of fits not converged."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    scores, failures = np.full(len(y), np.nan), 0
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        if normalise_rows:
            fit = fm.fit_logistic(X[train], y[train], n_classes=2)
            logits = fm.unit(X[test]) @ fit.W.T + fit.b
        else:
            fit, keep, mean, sd = fit_logistic_raw(X[train], y[train])
            logits = ((X[test][:, keep] - mean[keep]) / sd[keep]) @ fit.W.T + fit.b
        failures += not fit.converged
        scores[test] = logits[:, 1] - logits[:, 0]
    return scores, failures


def source_draw(units: Sequence[Sequence[str]], members: Sequence[Mapping[str, Sequence[int]]], pair: int, draw: int,
                k_max: int = max(KS)) -> list[tuple[list[str], list[int]]]:
    """E4 few-shot draw (stream 42, 14103, pair, draw): per source a random unit order and one random image of
    each of its first k_max units. The first K units are the K-shot supports (nested over K)."""
    rng = np.random.default_rng([fm.SEED, E4_FEWSHOT_NS, pair, draw])
    out = []
    for us, ms in zip(units, members):
        order = [us[i] for i in rng.permutation(len(us))]
        out.append((order, [int(ms[u][rng.integers(len(ms[u]))]) for u in order[:k_max]]))
    return out


def source_fewshot(X: np.ndarray, source: Sequence[int], unit_of: Sequence[str], pair: int,
                   predict: Callable[[Sequence[np.ndarray], np.ndarray], np.ndarray],
                   ks: Sequence[int] = KS, draws: int = fm.N_DRAWS) -> dict[tuple[int, int], float]:
    """Two-way source test: K units per source as supports (one image each), every image of the other units
    as queries; macro recall over the two sources per (K, draw). Ties go to the first source."""
    X, source, unit_of = np.asarray(X), np.asarray(source), np.asarray(unit_of)
    units = [sorted(set(unit_of[source == s].tolist())) for s in (0, 1)]
    if min(map(len, units)) <= max(ks):
        raise ValueError("each source needs at least one query unit after taking its largest support set")
    members = [{u: np.flatnonzero((unit_of == u) & (source == s)).tolist() for u in units[s]} for s in (0, 1)]
    out = {}
    for r in range(draws):
        plan = source_draw(units, members, pair, r, max(ks))
        for k in ks:
            used = [set(order[:k]) for order, _ in plan]
            supports = [X[picks[:k]] for _, picks in plan]
            query = np.array([u not in used[s] for u, s in zip(unit_of, source)])
            pred = predict(supports, X[query])
            truth = source[query]
            out[k, r] = float(np.mean([np.mean(pred[truth == s] == s) for s in (0, 1)]))
    return out


# ------------------------------------------------------------------- reading ---
def phase1_root(phase1: Path, work: Path, render_cache: Path) -> tuple[Path, list[list], Path | None]:
    """The folder holding the Phase-1 outputs: given as such, or the Drive phase1/ folder as Colab saved it.

    In the second case every phase1-*.zip except phase1-logs-* must match the SHA256 in its .zip.json receipt;
    Feature archives are unpacked in `work`. The render archive is cached in scratch by receipt SHA256
    across steps of a scoring run. Returns the worker root, verified receipts and cached render root."""
    archives = sorted(p for p in phase1.glob("phase1-*.zip") if not p.name.startswith("phase1-logs-"))
    if not archives:
        return phase1, [], None
    out, rows, rendered = work / "phase1", [], None
    for archive in archives:
        receipt = archive.with_name(archive.name + ".json")
        if not receipt.exists():
            raise ValueError(f"{archive.name} has no .json receipt")
        digest = prep.sha256(archive)
        if digest != prep.read_json(receipt)["sha256"]:
            raise ValueError(f"{archive.name} differs from the SHA256 in its .json receipt")
        if archive.name != "phase1-render.zip":
            safe_unzip(archive, out / archive.stem)
        else:
            rendered = render_cache / digest
            if not rendered.is_dir():
                render_cache.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=render_cache) as tmp:
                    pending = Path(tmp) / "unpacked"
                    safe_unzip(archive, pending)
                    try:
                        pending.rename(rendered)
                    except OSError:
                        if not rendered.is_dir():
                            raise
            with zipfile.ZipFile(archive) as source:
                if (rendered / "complete.json").read_bytes() != source.read("complete.json"):
                    raise ValueError("cached Phase-1 render marker differs from its verified archive")
        rows.append([archive.name, digest])
    return out, rows, rendered


@dataclass(frozen=True)
class Phase1Output:
    arm: str
    set: str
    variant: str
    folder: Path
    identity: Mapping


def phase1_outputs(root: Path) -> dict[tuple[str, str, str], Phase1Output]:
    """Every completed feature folder under root, keyed (set, arm, variant), from its identity.json."""
    found: dict[tuple[str, str, str], Phase1Output] = {}
    for marker in sorted(root.rglob("complete.json")):
        folder = marker.parent
        if not (folder / "features.npz").exists():
            continue
        prep.verify_complete(folder)
        identity = prep.read_json(folder / "identity.json")
        key = (identity["set"], identity["arm"], identity.get("variant") or "")
        if key[0] not in (*E_SETS, "E3") or key[1] not in ARMS:
            raise ValueError(f"{folder}: unknown set or arm {key}")
        allowed = (E3_REF, *E3_VARIANTS[key[1]]) if key[0] == "E3" and key[1] in E3_VARIANTS else ()
        if (key[0] == "E3" and key[2] not in allowed) or (key[0] in E_SETS and key[2]):
            raise ValueError(f"{folder}: unexpected variant {key}")
        if key in found:
            raise ValueError(f"two Phase-1 folders for {key}")
        found[key] = Phase1Output(key[1], key[0], key[2], folder, identity)
    return found


def render_folder(root: Path) -> Path | None:
    hits = sorted({p.parent for p in root.rglob("render.csv")})
    if len(hits) > 1:
        raise ValueError(f"more than one render folder: {hits}")
    if hits:
        prep.verify_complete(hits[0])
    return hits[0] if hits else None


@dataclass(frozen=True)
class Scored:
    """One feature set (or reference) on the locked episodes and nine-way roster."""
    dev: Mapping[tuple[int, int], np.ndarray]  # (K, draw) -> correctness over the 124 development queries
    rep: Mapping[tuple[int, int], np.ndarray]  # (K, draw) -> correctness over the legacy 22
    nine: Mapping[str, Cell] | None


def score_episodes(study: Study, select: Callable[[Sequence[str]], np.ndarray],
                   predict: Callable[[Sequence[np.ndarray], np.ndarray], np.ndarray]) -> tuple[dict, dict]:
    """Development (folds pooled, fixed query order) and hold-out correctness on the locked episodes."""
    order, _ = dev_order(study)
    pos = {q: i for i, q in enumerate(order)}
    dev: dict[tuple[int, int], np.ndarray] = {}
    for e in study.episodes("development"):
        pred = predict([select(p) for p in e.supports], select(e.queries))
        row = dev.setdefault((e.k, e.draw), np.zeros(len(order), dtype=bool))
        row[[pos[q] for q in e.queries]] = np.array(fm.RARE)[pred] == np.array(e.labels)
    rep, first = {}, None
    for e in study.episodes("reporting"):
        first = first or e.queries
        if e.queries != first:
            raise ValueError("reporting episodes differ in query order")
        rep[e.k, e.draw] = np.array(fm.RARE)[predict([select(p) for p in e.supports], select(e.queries))] == np.array(e.labels)
    return dev, rep


def nine_way_generic(study: Study, select_main: Callable, select_ext: Callable,
                     scores_of: Callable[[Sequence[np.ndarray], np.ndarray], np.ndarray]) -> dict[str, Cell]:
    """The locked nine-way roster with any class rule (nine_way_cells for features; the thumbnail rule)."""
    fitting = [r for r in study.roster if r["role"] == "fitting"]
    supports = [select_main([r["path"] for r in fitting if r["label"] == c]) for c in fm.NINE]
    query = [r for r in study.roster if r["role"] == "query"]
    out = {}
    for source, rows, select in (("Kermany", query, select_main),
                                 ("OCTID", [r for r in study.queries if r["source"] == "OCTID"], select_ext),
                                 ("OCTDL", [r for r in study.queries if r["source"] == "OCTDL"], select_ext)):
        paths = [r["path"] for r in rows]
        scores = scores_of(supports, select(paths))
        out[source] = Cell(tuple(paths), tuple(r["label"] for r in rows),
                           tuple(np.array(fm.NINE)[np.argmax(scores, axis=1)]), tuple(fm.NINE), scores)
    return out


def score_feature_set(study: Study, main: ev.Features, external: ev.Features) -> Scored:
    dev, rep = score_episodes(study, main.select, prototype_predict)
    return Scored(dev, rep, nine_way_cells(study, main, external))


def score_vectors(study: Study, vectors: Mapping[str, np.ndarray], nine: bool) -> Scored:
    """A reference without a model: nearest mean support on fixed vectors per image."""
    select = lambda paths: np.stack([np.asarray(vectors[p], dtype=np.float64).ravel() for p in paths])  # noqa: E731
    dev, rep = score_episodes(study, select, nearest_mean_predict)
    return Scored(dev, rep, nine_way_generic(study, select, select, nearest_mean_scores) if nine else None)


def dev_macro(study: Study, scored: Scored) -> dict[tuple[int, int], float]:
    _, labels = dev_order(study)
    return {key: macro(v, labels, fm.RARE) for key, v in scored.dev.items()}


def rep_macro(study: Study, scored: Scored) -> dict[tuple[int, int], float]:
    labels = [e for e in study.episodes("reporting")][0].labels
    return {key: macro(v, labels, fm.RARE) for key, v in scored.rep.items()}


def paired(a: Mapping[tuple[int, int], float], b: Mapping[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    return {key: a[key] - b[key] for key in a if key in b}


NINE_HEADER = ["query_set", "n", "correct_class", "called_NORMAL", "called_CNV_DME_DRUSEN", "called_any_rare",
               "rate", "rate_value", "ci95_low", "ci95_high", "interval_method", "NORMAL_recall",
               "NORMAL_recall_ci95_low", "NORMAL_recall_ci95_high"]


def nine_way_rows(study: Study, cells: Mapping[str, Cell]) -> list[list]:
    """Counts per query set; false-rare rate with the locked interval method on normals; R1 share on rare sets."""
    _, disjoint = study.kermany_normals()
    group = {r["path"]: r["dependence_group"] for r in study.queries}
    ker = cells["Kermany"]
    sets = [("Kermany patient-disjoint", ker, np.isin(ker.queries, sorted(disjoint))),
            ("Legacy 22", ker, np.isin(ker.labels, fm.RARE))]
    for name in ("OCTID NORMAL", "OCTID CSC", "OCTID MH", "OCTDL NORMAL", "OCTDL MH"):
        source, label = name.split()
        sets.append((name, cells[source], np.array(cells[source].labels) == label))
    rows = []
    for name, cell, mask in sets:
        q, lab, pred = np.array(cell.queries)[mask], np.array(cell.labels)[mask], np.array(cell.predicted)[mask]
        counts = [int((pred == lab).sum()), int((pred == "NORMAL").sum()), int(np.isin(pred, COMMON[1:]).sum()),
                  int(np.isin(pred, fm.RARE).sum())]
        if name in NORMAL_SETS:
            clusters = [ev.kermany_pid(p) for p in q] if name.startswith("Kermany") else \
                None if name.startswith("OCTID") else [group[p] for p in q]
            iv = ev.nine_way_intervals(lab.tolist(), pred.tolist(), clusters)
            rows.append([name, len(q), *counts, "false_rare_rate", counts[3] / len(q), *iv["false_rare"], iv["method"],
                         counts[1] / len(q), *iv["recall"]["NORMAL"]])
        elif name in RARE_EXTERNAL:
            rows.append([name, len(q), *counts, "share_called_any_rare (R1)", counts[3] / len(q), None, None, "",
                         None, None, None])
        else:
            rows.append([name, len(q), *counts, "", None, None, None, "", None, None, None])
    return rows


class Phase1:
    """The Phase-1 outputs of one attempt, verified against the locked study before use."""

    def __init__(self, study: Study, root: Path, receipts: list[list], dry_run: bool = False, partial: bool = False,
                 cached_render: Path | None = None):
        import fm_control_extract as control

        self.study, self.root, self.receipts, self.dry_run = study, root, receipts, dry_run
        self.outputs = phase1_outputs(root)
        self.render = render_folder(cached_render if cached_render is not None else root)
        if cached_render is not None and self.render is None:
            raise ValueError("render archive has no render.csv")
        missing = (["render"] if self.render is None else []) + [
            f"E1-M {arm}" for arm in ARMS if self.get("E1-M", arm) is None]
        if missing and not partial:
            raise ValueError(f"Partial Phase-1 delivery ({', '.join(missing)}). Score after the last session or use --partial.")
        if (any(name.startswith("phase1-E3-MIRAGE-M-res256") for name, _ in receipts)
                and self.get("E3", NOMINEE, "M-res256") is None):
            raise ValueError("MIRAGE M-res256 archive exists but has no complete feature output")
        lock = prep.sha256(study.lock / "input_lock.json")
        external = prep.sha256(study.external / "external_lock.json")
        expected = {"script_sha256": prep.text_hash(Path(control.__file__)), "registry_sha256": control.registry_hash()}

        def check_identity(identity: Mapping, folder: Path) -> None:
            if dry_run and identity.get("dry_run") is True:
                return
            for key, digest in expected.items():
                if identity.get(key) != digest:
                    raise ValueError(f"{folder}: {key} differs from the laptop Phase-1 runner")

        for key, out in self.outputs.items():
            if (out.identity.get("lock_sha256"), out.identity.get("external_lock_sha256")) != (lock, external):
                raise ValueError(f"{key}: Phase-1 output belongs to a different lock")
            check_identity(out.identity, out.folder)
        if self.render is not None:
            path = self.render / "identity.json"
            identity = prep.read_json(path)
            check_identity(identity, self.render)
            if (identity.get("lock_sha256"), identity.get("external_lock_sha256")) != (lock, external):
                raise ValueError("render belongs to a different lock")
            digest = prep.sha256(path)
            for out in self.outputs.values():
                if out.set in E_SETS and out.identity.get("render_identity_sha256") != digest:
                    raise ValueError(f"{out.folder}: worker used a different render")
        self.main_inv = prep.read_csv(study.lock / "extraction_inventory.csv")
        self.ext_inv = prep.read_csv(study.external / "external_extraction_inventory.csv")
        self._feats: dict = {}
        self._control: dict = {}

    def get(self, set_name: str, arm: str, variant: str = "") -> Phase1Output | None:
        return self.outputs.get((set_name, arm, variant))

    def features(self, arm: str, set_name: str, variant: str = "") -> tuple[ev.Features, ev.Features] | None:
        if set_name == "locked":
            return self.study.features(arm), self.study.external_features(arm)
        out = self.get(set_name, arm, variant)
        if out is None:
            return None
        if out.folder not in self._feats:
            dim = extract.DIMENSIONS[arm]
            self._feats[out.folder] = (
                ev.Features(extract.load_features(out.folder / "features.npz", self.main_inv, dim),
                            {r["path"]: i for i, r in enumerate(self.main_inv)}),
                ev.Features(extract.load_features(out.folder / "features_external.npz", self.ext_inv, dim),
                            {r["path"]: i for i, r in enumerate(self.ext_inv)}))
        return self._feats[out.folder]

    def locked_all(self, arm: str) -> ev.Features:
        return fx.merge(self.study.features(arm), self.study.external_features(arm))

    def control(self, out: Phase1Output) -> dict:
        """Originals through the locked path, checked on the exact registered inventory rows."""
        if out.folder not in self._control:
            arm = out.arm
            locked = self.study.features(arm)
            dim = extract.DIMENSIONS[arm]
            if out.set == "E3":
                main, ext = self.features(arm, out.set, out.variant)
                check = extract.repeat_check([locked.values, self.study.external_features(arm).values],
                                             [main.values, ext.values])
            else:
                main = self.main_inv if arm == "DINOv2" else self.main_inv[:20]
                before = [locked.select([r["path"] for r in main])]
                after = [extract.load_features(out.folder / "control.npz", main, dim)]
                if arm == "DINOv2":
                    before.append(self.study.external_features(arm).select([r["path"] for r in self.ext_inv]))
                    after.append(extract.load_features(out.folder / "control_external.npz", self.ext_inv, dim))
                check = extract.repeat_check(before, after)
            check["passed"] = check["max_difference_ratio"] <= CONTROL_TOL
            self._control[out.folder] = check
        return self._control[out.folder]

    def status(self, arm: str, set_name: str, variant: str = "") -> str:
        if set_name == "locked":
            return INTERPRETED
        out = self.get(set_name, arm, variant)
        if out is None:
            return UNAVAILABLE
        ref = self.get("E3", arm, E3_REF) if arm in E3_VARIANTS else None
        if set_name == "E3":
            return INTERPRETED if ref and self.control(ref)["passed"] else NOT_INTERPRETED
        if self.render is None and not self.dry_run:
            return NOT_INTERPRETED
        passed = self.control(out)["passed"] and (arm not in E3_VARIANTS or (ref and self.control(ref)["passed"]))
        return INTERPRETED if passed else NOT_INTERPRETED

    def render_rows(self) -> list[dict[str, str]]:
        if self.render is None:
            return []
        rows = prep.read_csv(self.render / "render.csv")
        missing = set(RENDER_COLUMNS) - set(rows[0]) if rows else set()
        if missing:
            raise ValueError(f"render.csv lacks columns {sorted(missing)}")
        if [(r["path"], r["sha256"]) for r in rows] != [
            (r["path"], r["sha256"]) for r in (*self.main_inv, *self.ext_inv)
        ]:
            raise ValueError("render.csv differs from the locked inventories")
        return rows

    def thumbs(self) -> dict[str, dict[str, np.ndarray]]:
        """Per thumbnail set (raw, E1-M, E1-L, E1-C): path -> 256 values / 255."""
        if self.render is None or not (self.render / "thumbs16.npz").exists():
            return {}
        with np.load(self.render / "thumbs16.npz", allow_pickle=False) as z:
            paths = z["paths"].tolist()
            if list(zip(paths, z["sha256"].tolist(), strict=True)) != [
                (r["path"], r["sha256"]) for r in (*self.main_inv, *self.ext_inv)
            ]:
                raise ValueError("thumbs16.npz differs from the locked inventories")
            out = {}
            for key in ("raw", *E_SETS):
                if key in z.files:
                    values = z[key]
                    if (values.shape != (len(paths), 256) or not np.isfinite(values).all()
                            or np.any((values < 0) | (values > 1))):
                        raise ValueError(f"thumbs16.npz {key}: expected [n, 256] values in [0, 1]")
                    out[key] = dict(zip(paths, values))
        return out

    def pixel_stats(self, which: str) -> dict[str, np.ndarray]:
        if self.render is None or not (self.render / "pixel_stats.csv").exists():
            return {}
        prefix = "original" if which == "raw" else which
        return {r["path"]: np.array([float(r[f"{prefix}_{c}"]) for c in PIXEL_STATS])
                for r in prep.read_csv(self.render / "pixel_stats.csv")}


def render_png(render: Path, set_name: str, path: str) -> Path:
    return render / set_name / prep.safe_id(path + ".png")


def locked_geometry(study: Study) -> dict[str, dict[str, float]]:
    """Original width, height, aspect ratio, A2 border share and near-white share (locked geometry.csv)."""
    study.features(fm.REFERENCE), study.external_features(fm.REFERENCE)  # unpack and verify both bundles
    out = {}
    for folder in (study.work / "features" / fm.REFERENCE, study.work / "features" / f"{fm.REFERENCE}-external"):
        for r in prep.read_csv(folder / "geometry.csv"):
            w, h = float(r["width"]), float(r["height"])
            out[r["path"]] = dict(width=w, height=h, aspect_ratio=w / h, border_share=float(r["border_share"]),
                                  near_white_share=float(r["near_white_share"]))
    return out


def source_group(path: str, external_source: Mapping[str, str]) -> str:
    if path in external_source:
        return external_source[path]
    if path.startswith("data/raw/rare/"):
        return path.split("/")[3]
    return "Kermany"


# ======================================================================= E1 ===
def rerender_agreement(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[int, int, int]:
    """Laptop re-render against the Colab render: images compared, images that differ, largest grey difference."""
    differ, largest = 0, 0
    for a, b in pairs:
        if a.shape != b.shape:
            differ, largest = differ + 1, 255
            continue
        d = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) if a.size else 0
        differ += d > 0
        largest = max(largest, d)
    return len(pairs), differ, largest


def laptop_rerender(p1: Phase1, data_root: Path | None) -> list[list]:
    """E1a: re-render locally and compare greys with the corresponding Colab PNGs."""
    if data_root is None or p1.render is None:
        return [["laptop re-render agreement", "", "", "status", "not run (no --data-root or no render folder)", "", ""]]
    try:
        from PIL import Image
        import fm_control_extract as control  # noqa: PLC0415  (written beside this file; not hashed)
        render = control.render_all
    except (ImportError, AttributeError) as error:
        return [["laptop re-render agreement", "", "E1-M", "status", f"not run ({error})", "", ""]]
    counts = {name: [0, 0, 0] for name in E_SETS}
    missing = unreadable = 0
    for row in p1.render_rows():
        raw = data_root / prep.safe_id(row["path"])
        if not raw.exists() or prep.sha256(raw) != row["sha256"]:
            missing += 1
            continue
        try:
            with Image.open(raw) as image:
                _, local = render(image.convert("RGB"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        for name in E_SETS:
            mine = np.asarray(local[name].convert("L"))
            with Image.open(render_png(p1.render, name, row["path"])) as image:
                theirs = np.asarray(image.convert("L"))
            n, differ, largest = rerender_agreement([(theirs, mine)])
            counts[name][0] += n
            counts[name][1] += differ
            counts[name][2] = max(counts[name][2], largest)
    metrics = ("images compared", "images differing", "largest grey difference",
               "originals missing or differing from locked SHA256", "originals not readable on this laptop")
    return [["laptop re-render agreement", "", name, metric, value, "", "a report, never a stop"]
            for name, values in counts.items() for metric, value in zip(metrics, (*values, missing, unreadable))]


def framing_residual(rows: Sequence[Mapping[str, str]], external_source: Mapping[str, str]) -> list[list]:
    """E1a framing residual per set and source / rare class, from edge_residual.csv."""
    groups: dict[str, list[Mapping[str, str]]] = {}
    for r in rows:
        groups.setdefault(source_group(r["path"], external_source), []).append(r)
    out = []
    for group, members in sorted(groups.items()):
        for name in ("original", *E_SETS):
            a2, run, white = (np.array([float(r[f"{name}_{col}"]) for r in members])
                              for col in ("border_share", "edge_run", "near_white_share"))
            for quantity, value in (("mean A2 border share", a2.mean()), ("share with A2 border share > 0", (a2 > 0).mean()),
                                    ("share with near-black edge run > 5% of side", (run > 0.05).mean()),
                                    ("share with near-black edge run > 10% of side", (run > 0.10).mean()),
                                    ("mean near-white share", white.mean())):
                out.append(["framing residual", group, name, quantity, float(value), "", f"n={len(members)}"])
    return out


def e1_checks(p1: Phase1, section: Mapping, data_root: Path | None) -> list[list]:
    rows = [["rule 1: Phase-1 subsection SHA256", "", "", "at rule commit " + section["commit"][:7], section["commit_sha256"],
             "same" if section["same"] else "DIFFERS", f"working {section['working_sha256']}"]]
    rows += [["archive receipt", "", "", name, digest, "equal", ""] for name, digest in p1.receipts]
    for (set_name, arm, variant), out in sorted(p1.outputs.items()):
        if set_name not in E_SETS:
            continue
        c = p1.control(out)
        status = "pass" if c["passed"] else NOT_INTERPRETED
        rows.append(["control: locked path against locked features", arm, set_name + (f" {variant}" if variant else ""),
                     "max_difference_ratio", c["max_difference_ratio"], status, f"{c['images']} images; min cosine {c['min_cosine']:.6f}"])
        rep = out.identity.get("repeat_check") or {}
        rows.append(["repeat (recorded, not a gate)", arm, set_name + (f" {variant}" if variant else ""), "max_difference_ratio",
                     rep.get("max_difference_ratio"), "", f"{rep.get('images')} images"])
        for key in ("code_matches_lock", "environment_matches_lock"):
            rows.append(["identity", arm, set_name + (f" {variant}" if variant else ""), key, out.identity.get(key), "", ""])
        if out.identity.get("overrides"):
            rows.append(["identity", arm, set_name + (f" {variant}" if variant else ""), "overrides",
                         json.dumps(out.identity["overrides"], sort_keys=True), "", ""])
    for arm in E3_VARIANTS:
        ref = p1.get("E3", arm, E3_REF)
        c = p1.control(ref) if ref else None
        rows.append(["E3-ref full control", arm, "", "max_difference_ratio",
                     c["max_difference_ratio"] if c else None, "pass" if c and c["passed"] else NOT_INTERPRETED,
                     f"{c['images']} images" if c else UNAVAILABLE])
    decoder = p1.get("E1-M", fm.REFERENCE)
    if decoder is not None and (decoder.folder / "decoder_check.npz").exists():
        sources = {r["path"]: r["source"] for r in p1.study.queries}
        by_source = {"Kermany": [r for r in p1.main_inv if source_group(r["path"], {}) == "Kermany"][:20],
                     **{s: [r for r in p1.ext_inv if sources[r["path"]] == s][:10] for s in ("OCTID", "OCTDL")}}
        inventory = [r for group in by_source.values() for r in group]
        values = extract.load_features(decoder.folder / "decoder_check.npz", inventory, extract.DIMENSIONS[fm.REFERENCE])
        locked = p1.locked_all(fm.REFERENCE).select([r["path"] for r in inventory])
        start = 0
        for source, group in by_source.items():
            end = start + len(group)
            check = extract.repeat_check([locked[start:end]], [values[start:end]])
            rows.append(["Inception-v3 decoder change (PIL-decoded PNG originals)", fm.REFERENCE, "E1-M", source,
                         check["max_difference_ratio"], "", f"{check['images']} images; min cosine {check['min_cosine']:.6f}"])
            start = end
        resize = decoder.identity.get("tf_resize_identity")
        rows.append(["Inception-v3 resize identity", fm.REFERENCE, "E1-M", "tf.image.resize",
                     json.dumps(resize), "failed check; not interpreted" if resize and not resize["unchanged"] else "", ""])
    render = p1.render_rows()
    if render:
        flagged = sum(r["flagged"] in ("1", "True", "true") for r in render)
        alpha = sum(int(float(r["min_alpha"])) < 255 for r in render if r["min_alpha"] not in ("", "None"))
        rows += [["render", "", "", "images rendered (twice; saved PNG pixel hashes verified)", len(render), "", ""],
                 ["render", "", "", "content box fallback to the full frame (flagged)", flagged, "", ""],
                 ["render", "", "", "files with alpha below 255 (dropped by convert('RGB'))", alpha, "", ""]]
        formats: dict[tuple[str, str], int] = {}
        for r in render:
            formats[r["mode"], r["format"]] = formats.get((r["mode"], r["format"]), 0) + 1
        rows += [["render", "", "", f"mode {m}, format {f}", n, "", ""] for (m, f), n in sorted(formats.items())]
        rare_under_128: dict[str, int] = {}
        short_sides: dict[str, list[int]] = {}
        source = {r["path"]: r["source"] for r in p1.study.queries}
        for path, size in content_sizes(p1).items():
            group = "Yoo rare" if path.startswith("data/raw/rare/") else source_group(path, source)
            short_sides.setdefault(group, []).append(min(size))
            if path.startswith("data/raw/rare/") and min(size) < 128:
                group = source_group(path, {})
                rare_under_128[group] = rare_under_128.get(group, 0) + 1
        rows += [["render", group, "", "content short-side range (px)", f"{min(sides)}–{max(sides)}", "",
                  f"{len(sides)} images"] for group, sides in sorted(short_sides.items())]
        rows += [["render", group, "E1-L", "content short side below 128 px", n, "", ""]
                 for group, n in sorted(rare_under_128.items())]
        if (p1.render / "edge_residual.csv").exists():
            rows += framing_residual(prep.read_csv(p1.render / "edge_residual.csv"), source)
    else:
        rows.append(["render", "", "", "render folder", "missing", "", ""])
    rows += laptop_rerender(p1, data_root)
    return rows


def content_sizes(p1: Phase1) -> dict[str, tuple[int, int]]:
    """Width and height of each image's final content box (render.csv)."""
    return {r["path"]: (int(r["box_right"]) - int(r["box_left"]), int(r["box_bottom"]) - int(r["box_top"]))
            for r in p1.render_rows()}


def e1(study: Study, p1: Phase1, section: Mapping, data_root: Path | None = None,
       drop: set[str] = frozenset()) -> dict[str, tuple[list[str], list[list]]]:
    """All E1 tables (E1a-E1h) as {file name: (header, rows)}."""
    tables: dict[str, tuple[list[str], list[list]]] = {}
    scored: dict[tuple[str, str], Scored] = {}
    status: dict[tuple[str, str], str] = {}
    for arm in ARMS:
        for s in FEATURE_SETS:
            feats = p1.features(arm, s)
            if feats is not None:
                scored[arm, s], status[arm, s] = score_feature_set(study, *feats), p1.status(arm, s)
    dev = {key: dev_macro(study, sc) for key, sc in scored.items()}
    raw_sizes = {p: np.array([g["width"], g["height"]]) for p, g in locked_geometry(study).items()}
    refs = {("size_only", "raw"): score_vectors(study, raw_sizes, True)}
    sizes = content_sizes(p1)
    if sizes:
        refs["content_size", "raw content box"] = score_vectors(study, {p: np.array(v, float) for p, v in sizes.items()}, True)
    for name, vectors in p1.thumbs().items():
        refs["thumbnail", name] = score_vectors(study, vectors, True)
    ref_dev = {key: dev_macro(study, sc) for key, sc in refs.items()}

    tables["E1a_checks.csv"] = (["check", "arm", "feature_set", "quantity", "value", "status", "note"],
                                e1_checks(p1, section, data_root))
    rows = [[arm, s, status[arm, s], k, *summary(v)] for (arm, s), d in dev.items() for k, v in by_k(d, KS)]
    rows += [[arm, s, UNAVAILABLE, "", *[None] * len(SUMMARY)] for arm in ARMS for s in E_SETS if (arm, s) not in dev]
    rows += [[name, s, "reference without a model", k, *summary(v)] for (name, s), d in ref_dev.items() for k, v in by_k(d, KS)]
    tables["E1b_development.csv"] = (["model", "feature_set", "status", "k", *SUMMARY], rows)

    def worst(*keys) -> str:
        values = [status.get(k, UNAVAILABLE) for k in keys]
        return UNAVAILABLE if UNAVAILABLE in values else NOT_INTERPRETED if NOT_INTERPRETED in values else INTERPRETED

    rows = []
    for s in FEATURE_SETS:
        for arm in ARMS[:-1]:
            if (arm, s) in dev and (fm.REFERENCE, s) in dev:
                rows += [[f"{arm} minus {fm.REFERENCE}", s, worst((arm, s), (fm.REFERENCE, s)), k, *summary(v)]
                         for k, v in by_k(paired(dev[arm, s], dev[fm.REFERENCE, s]), KS)]
    for s in E_SETS:
        for arm in ARMS:
            if (arm, s) in dev:
                rows += [[f"{arm} {s} minus {arm} locked", s, worst((arm, s)), k, *summary(v)]
                         for k, v in by_k(paired(dev[arm, s], dev[arm, "locked"]), KS)]
    tables["E1c_paired.csv"] = (["comparison", "feature_set", "status", "k", *SUMMARY], rows)

    # E1d: the success rule on E1-M; the same statistic for every set; two companion lines.
    stats = {}
    rows = []
    for s in FEATURE_SETS:
        if (NOMINEE, s) not in dev or (fm.REFERENCE, s) not in dev:
            rows.append(["MIRAGE minus Inception-v3, K mean", s, UNAVAILABLE, 0, None, None, None, 0,
                         "not evaluable" if s == "E1-M" else "description only", ""])
            continue
        st = success_rule(k_mean(paired(dev[NOMINEE, s], dev[fm.REFERENCE, s])))
        if s != "locked" and worst((NOMINEE, s), (fm.REFERENCE, s)) != INTERPRETED:
            st["verdict"] = "not evaluable"
        stats[s] = st
        rows.append(["MIRAGE minus Inception-v3, K mean", s, worst((NOMINEE, s), (fm.REFERENCE, s)), st["n"], st["mean"],
                     st["sd"], st["p05"], st["n_positive"], st["verdict"] if s == "E1-M" else "description only", ""])
    main = stats.get("E1-M", dict(verdict="not evaluable", n=0, mean=None, p05=None))
    reason = "" if "E1-M" in stats else "E1-M features of MIRAGE or Inception-v3 unavailable"
    if "E1-M" in stats and worst((NOMINEE, "E1-M"), (fm.REFERENCE, "E1-M")) != INTERPRETED:
        reason = "MIRAGE or Inception-v3 failed its control"
    thumb = None
    for name, s in (("content_size", "raw content box"), ("thumbnail", "E1-M")):
        if (NOMINEE, "E1-M") in dev and (name, s) in ref_dev:
            st = success_rule(k_mean(paired(dev[NOMINEE, "E1-M"], ref_dev[name, s])))
            rows.append([f"MIRAGE E1-M minus {name} reference ({s}), K mean", "E1-M", worst((NOMINEE, "E1-M")), st["n"], st["mean"],
                         None, st["p05"], st["n_positive"], "companion line", ""])
            if name == "thumbnail":
                thumb = st
    e3_res256_shift = None
    if main["verdict"] == "fail":
        feats = p1.features(NOMINEE, "E3", "M-res256")
        if feats is not None:
            resized = dev_macro(study, score_feature_set(study, *feats))
            shift = success_rule(k_mean(paired(resized, dev[NOMINEE, "locked"])))
            if shift["verdict"] == "not evaluable":
                raise ValueError("MIRAGE M-res256 does not yield 100 paired development K means")
            e3_res256_shift = shift["mean"]
            rows.append(["MIRAGE M-res256 minus MIRAGE locked, K mean", "E3 M-res256",
                         p1.status(NOMINEE, "E3", "M-res256"), shift["n"], shift["mean"], shift["sd"], shift["p05"],
                         shift["n_positive"], "description only", ""])
    sentence = e1_sentences(main, stats.get("locked", {}).get("mean"), thumb, stats.get("E1-L"),
                            stats.get("E1-C"), e3_res256_shift)
    rows.append(["fixed sentence", "E1-M", reason, None, None, None, None, None, main["verdict"], sentence])
    rows.append(["limitations (stated with every E1 result)", "", "", None, None, None, None, None, "", E1_LIMITATIONS])
    tables["E1d_success.csv"] = (["line", "feature_set", "status", "n_draws", "mean", "sd", "p05", "draws_above_0",
                                  "verdict", "text"], rows)

    order, labels = dev_order(study)
    if "E1e" not in drop and sizes:
        sides = {p: min(sizes[p]) for p in order}
        low, high, subset = matched_subset(sides, dict(zip(order, labels)))
        keep = np.isin(order, subset)
        sub_labels = np.asarray(labels)[keep]
        rows = [["matched subset", f"L={low:g} px, U={high:g} px", "", "", len(subset), "", *[None] * len(SUMMARY)]]
        values = {}
        for key, sc in scored.items():
            per = {}
            for kd, v in sc.dev.items():
                value, present = present_macro(v[keep], sub_labels, fm.RARE)
                if value is not None:
                    per[kd] = value
            values[key] = per
            rows += [[key[0], key[1], status[key], k, len(subset), " ".join(present_macro(np.ones(keep.sum(), bool), sub_labels, fm.RARE)[1]),
                      *summary(v)] for k, v in by_k(per, KS)]
        for s in FEATURE_SETS:
            if (NOMINEE, s) in values and (fm.REFERENCE, s) in values:
                rows += [[f"{NOMINEE} minus {fm.REFERENCE}", s, worst((NOMINEE, s), (fm.REFERENCE, s)), k, len(subset), "",
                          *summary(v)] for k, v in by_k(paired(values[NOMINEE, s], values[fm.REFERENCE, s]), KS)]
        tables["E1e_matched.csv"] = (["model", "feature_set", "status", "k", "n_queries", "classes_present", *SUMMARY], rows)

    rows = []
    rep = {key: rep_macro(study, sc) for key, sc in scored.items()}
    ks_rep = sorted({k for k, _ in next(iter(rep.values()))})
    for key, sc in scored.items():
        counts = {kd: float(v.sum()) for kd, v in sc.rep.items()}
        for k, v in by_k(rep[key], ks_rep):
            note = "shown, never read" if k == 16 else ""
            rows.append([key[0], key[1], status[key], k, "macro_recall", *summary(v), "", note])
            if k != ev.K_MEAN_LABEL:
                rows.append([key[0], key[1], status[key], k, "mean_correct_of_22", None,
                             float(np.mean([c for (kk, _), c in counts.items() if kk == k])), None, None, None, "", note])
    for s in FEATURE_SETS:
        if (NOMINEE, s) in rep and (fm.REFERENCE, s) in rep:
            for k, v in by_k(paired(rep[NOMINEE, s], rep[fm.REFERENCE, s]), ks_rep):
                m = float(np.mean(v))
                direction = "positive" if m > 0 else "negative" if m < 0 else "zero"
                rows.append([f"{NOMINEE} minus {fm.REFERENCE}", s, worst((NOMINEE, s), (fm.REFERENCE, s)), k, "paired_macro_recall",
                             *summary(v), direction if k != 16 and worst((NOMINEE, s), (fm.REFERENCE, s)) == INTERPRETED else "",
                             "shown, never read" if k == 16 else "direction only"])
    for (name, s), sc in refs.items():
        for k, v in by_k(rep_macro(study, sc), ks_rep):
            rows.append([name, s, "reference without a model", k, "macro_recall", *summary(v), "", "shown, never read" if k == 16 else ""])
    tables["E1f_holdout.csv"] = (["model", "feature_set", "status", "k", "quantity", *SUMMARY, "direction", "note"], rows)

    rows = [[arm, s, status[arm, s], *r] for (arm, s), sc in scored.items() for r in nine_way_rows(study, sc.nine)]
    rows += [[name, s, "reference without a model", *r] for (name, s), sc in refs.items() if sc.nine is not None
             for r in nine_way_rows(study, sc.nine)]
    tables["E1g_nine_way.csv"] = (["model", "feature_set", "status", *NINE_HEADER], rows)

    if "E1h" not in drop:
        rows = []
        for s in ("locked", "E1-M"):
            if (NOMINEE, s) not in scored or (fm.REFERENCE, s) not in scored:
                continue
            a, b = scored[NOMINEE, s].dev, scored[fm.REFERENCE, s].dev
            keys = sorted(a)
            ks_rows = np.array([k for k, _ in keys])
            A, B = np.array([a[x] for x in keys]), np.array([b[x] for x in keys])
            full = mean_over_k(np.array([macro(r, labels, fm.RARE) for r in A]), ks_rows) - \
                mean_over_k(np.array([macro(r, labels, fm.RARE) for r in B]), ks_rows)
            diffs = loo_differences(A, B, labels, fm.RARE, ks_rows)
            lo, hi = int(np.argmin(diffs)), int(np.argmax(diffs))
            rows.append([s, worst((NOMINEE, s), (fm.REFERENCE, s)), len(order), float(full), float(diffs[lo]), order[lo],
                         float(diffs[hi]), order[hi], "locked = P1, shown for reference" if s == "locked" else ""])
        tables["E1h_leave_one_out.csv"] = (["feature_set", "status", "n_left_out", "full_difference", "smallest",
                                            "smallest_image", "largest", "largest_image", "note"], rows)
    return tables


# ======================================================================= E2 ===
def e2_cells(study: Study, p1: Phase1) -> dict[tuple[str, str], tuple[str, Mapping[str, Cell]]]:
    """(arm, set) -> (status, nine-way cells by source): locked from the results archive, E1 sets rescored."""
    out = {}
    for arm in ARMS:
        out[arm, "locked"] = (INTERPRETED, {s: study.results.nine_way(arm, s) for s in ("Kermany", "OCTID", "OCTDL")})
        for s in E_SETS:
            feats = p1.features(arm, s)
            if feats is not None:
                out[arm, s] = (p1.status(arm, s), nine_way_cells(study, *feats))
    return out


def e2_sets(study: Study) -> dict[str, tuple[str, list[str]]]:
    """Query set -> (source cell, paths): the normal sets and the external rare sets."""
    _, disjoint = study.kermany_normals()
    out = {"Kermany patient-disjoint": ("Kermany", sorted(disjoint))}
    for name in (*NORMAL_SETS[1:], *RARE_EXTERNAL):
        source, label = name.split()
        out[name] = (source, [r["path"] for r in study.queries if r["source"] == source and r["label"] == label])
    return out


def e2_tables(study: Study, cells: Mapping[tuple[str, str], tuple[str, Mapping[str, Cell]]],
              properties: Mapping[str, Mapping[str, float]], categorical: Mapping[str, Mapping[str, str]],
              drop: set[str] = frozenset()) -> dict[str, tuple[list[str], list[list]]]:
    """E2a-E2e from nine-way cells (pure apart from the study's rosters). properties[path][name] holds the E2d
    numbers of each original; categorical[path] its OCTDL mode and format."""
    sets = e2_sets(study)
    pred = {key: {s: dict(zip(c.queries, c.predicted)) for s, c in cs.items()} for key, (_, cs) in cells.items()}
    status = {key: st for key, (st, _) in cells.items()}
    feature_sets = [s for s in FEATURE_SETS if any(k[1] == s for k in cells)]
    unit = {r["path"]: study.patient.get(r["path"], r["path"]) for r in study.queries}
    tables = {}
    rows = []
    for (arm, s), p in pred.items():
        for name, (source, paths) in sets.items():
            got = [p[source][q] for q in paths]
            counts = [sum(g == c for g in got) for c in fm.NINE]
            groups = [sum(destination(g) == d for g in got) for d in ("NORMAL", "common", "rare")]
            rows.append([arm, s, status[arm, s], name, len(paths), *counts, *groups, "supplement" if name in RARE_EXTERNAL else ""])
    tables["E2a_destinations.csv"] = (["arm", "feature_set", "status", "query_set", "n", *fm.NINE, "group_NORMAL",
                                       "group_CNV_DME_DRUSEN", "group_rare", "placement"], rows)
    rows = []
    for name in NORMAL_SETS:
        source, paths = sets[name]
        for s in feature_sets:
            arms = [a for a in ARMS if (a, s) in pred]
            failed = {a: {q for q in paths if pred[a, s][source][q] != "NORMAL"} for a in arms}
            rare = {a: {q for q in paths if pred[a, s][source][q] in fm.RARE} for a in arms}
            for n, count in enumerate(failure_overlap(failed, paths)):
                state = NOT_INTERPRETED if any(status[a, s] == NOT_INTERPRETED for a in arms) else INTERPRETED
                rows.append([name, s, f"normals failed by {n} of {len(arms)} arms", "", "", state, count])
            for i, a in enumerate(arms):
                for b in arms[i + 1:]:
                    state = NOT_INTERPRETED if NOT_INTERPRETED in (status[a, s], status[b, s]) else INTERPRETED
                    for what, pair_sets in (("failure Jaccard", failed), ("false-rare Jaccard", rare)):
                        j = jaccard(pair_sets[a], pair_sets[b])
                        rows.append([name, s, what, a, b, state, "—" if j is None else j])
                    rows.append([name, s, "share given the same class", a, b, state,
                                 same_class_share([pred[a, s][source][q] for q in paths], [pred[b, s][source][q] for q in paths])])
    tables["E2b_overlap.csv"] = (["query_set", "feature_set", "quantity", "arm_a", "arm_b", "status", "value"], rows)
    rows = []
    for arm in ARMS:
        if (arm, "E1-M") not in pred:
            continue
        for name in (*NORMAL_SETS, *RARE_EXTERNAL):
            source, paths = sets[name]
            for a in ("NORMAL", "common", "rare"):
                for b in ("NORMAL", "common", "rare"):
                    n = sum(destination(pred[arm, "locked"][source][q]) == a and destination(pred[arm, "E1-M"][source][q]) == b
                            for q in paths)
                    rows.append([arm, name, status[arm, "E1-M"], a, b, n])
    tables["E2c_transitions.csv"] = (["arm", "query_set", "status", "locked", "E1-M", "count"], rows)
    rows, cat_rows = [], []
    for arm_index, arm in enumerate(ARMS):
        for source_index, name in enumerate(NORMAL_SETS):
            source, paths = sets[name]
            units = [ev.kermany_pid(q) if source == "Kermany" else unit[q] for q in paths]
            for set_index, s in enumerate(("locked", "E1-M")):
                if (arm, s) not in pred:
                    continue
                failed = [pred[arm, s][source][q] != "NORMAL" for q in paths]
                for prop_index, prop in enumerate(E2D_PROPERTIES):
                    have = [i for i, q in enumerate(paths) if prop in properties.get(q, {})]
                    if not have:
                        continue
                    values = [properties[paths[i]][prop] for i in have]
                    main_text = name == "OCTDL NORMAL" and s == "locked"
                    common = sum(destination(pred[arm, s][source][paths[i]]) == "common" for i in have)
                    rare = sum(destination(pred[arm, s][source][paths[i]]) == "rare" for i in have)
                    rng = None if "E2d-intervals" in drop else np.random.default_rng(
                        [fm.SEED, E2_BOOT_NS, arm_index, source_index, set_index, prop_index])
                    rows.append([arm, name, s, status[arm, s], prop, main_text, common, rare,
                                 *e2d_row(values, [failed[i] for i in have], [units[i] for i in have], rng)])
                if source == "OCTDL":
                    for prop in ("mode", "format"):
                        values = [categorical.get(q, {}).get(prop, "") for q in paths]
                        for v in sorted(set(values) - {""}):
                            sel = [destination(pred[arm, s][source][q]) for q, x in zip(paths, values) if x == v]
                            failures = len(sel) - sel.count("NORMAL")
                            cat_rows.append([arm, name, s, status[arm, s], prop, v, len(sel), failures,
                                             sel.count("common"), sel.count("rare"), failures / len(sel)])
    tables["E2d_properties.csv"] = (["arm", "query_set", "feature_set", "status", "property", "main_text_table",
                                    "failed_common", "failed_rare", *E2D_HEADER], rows)
    tables["E2d_categorical.csv"] = (["arm", "query_set", "feature_set", "status", "property", "value", "n", "failed",
                                     "failed_common", "failed_rare", "failure_share"], cat_rows)
    rows, patient_rows = [], []
    source, paths = sets["OCTDL NORMAL"]
    for (arm, s), p in pred.items():
        by: dict[str, list[str]] = {}
        for q in paths:
            by.setdefault(study.patient[q], []).append(destination(p[source][q]))
        patient_rows.extend([arm, s, status[arm, s], f"P{i:03d}", len(v), v.count("common"), v.count("rare"),
                             (len(v) - v.count("NORMAL")) / len(v)] for i, (_, v) in enumerate(sorted(by.items()), 1))
        failures = [sum(call != "NORMAL" for call in v) for v in by.values()]
        scans = [len(v) for v in by.values()]
        rows.append([arm, s, status[arm, s], len(by), sum(n == f for n, f in zip(scans, failures)),
                     sum(0 < f < n for n, f in zip(scans, failures)), failures.count(0),
                     float(np.mean(np.divide(failures, scans))),
                     float(np.mean([v.count("common") / len(v) for v in by.values()])),
                     float(np.mean([v.count("rare") / len(v) for v in by.values()])),
                     float(np.mean(scans)), float(np.median(scans)),
                     min(scans), max(scans)])
    tables["E2e_octdl_patients.csv"] = (["arm", "feature_set", "status", "patients", "all_failed", "some_failed", "none_failed",
                                         "failure_share_mean_per_patient", "common_disease_share_mean_per_patient",
                                         "false_rare_share_mean_per_patient", "scans_per_patient_mean", "scans_per_patient_median",
                                         "scans_per_patient_min", "scans_per_patient_max"], rows)
    tables["E2e_octdl_patient_scans.csv"] = (["arm", "feature_set", "status", "patient", "scans", "called_common_disease",
                                             "called_rare", "failure_share"], patient_rows)
    return tables


def e2f_groups(study: Study, cells: Mapping[tuple[str, str], tuple[str, Mapping[str, Cell]]]) -> tuple[dict, list[str]]:
    """E2f groups from the locked nine-way: consensus failure / consensus normal (>= 3 of 4 arms) per source."""
    sets = e2_sets(study)
    groups = {}
    for name in NORMAL_SETS[1:]:
        source, paths = sets[name]
        calls = {a: dict(zip(cells[a, "locked"][1][source].queries, cells[a, "locked"][1][source].predicted)) for a in ARMS}
        groups[source, "consensus failure"] = [q for q in paths if sum(calls[a][q] != "NORMAL" for a in ARMS) >= 3]
        groups[source, "consensus normal"] = [q for q in paths if sum(calls[a][q] == "NORMAL" for a in ARMS) >= 3]
    return groups, sets["Kermany patient-disjoint"][1]


def build_contact_sheet(sample: Sequence[Sequence[str]], images: Callable[[str, int], object], sheet: Path,
                        phase: int, label: str) -> list[Path]:
    """Write one blinded pass under Drive. Pass 2 requires completed, hashed pass 1 marks."""
    sheet = prep.experiment_out(sheet)
    key = sheet / "key" / "key.csv"
    if phase == 2:
        previous = sheet / "pass1" / "marks.csv"
        if [list(r.values()) for r in prep.read_csv(key)] != list(sample):
            raise ValueError("E2f pass 2 sample differs from the saved pass 1 roster")
        validate_marks(prep.read_csv(previous), [row[0] for row in sample])
        digest = prep.sha256(previous)
        prep.write_once(previous.with_suffix(".sha256"), (digest + "\n").encode("ascii"))
        print(f"SHA256 {digest}  {previous}")
    else:
        key.parent.mkdir(parents=True, exist_ok=True)
        prep.write_csv(key, ["id", "path", "source", "group"], [list(r) for r in sample])
    folder = sheet / f"pass{phase}"
    folder.mkdir(parents=True, exist_ok=True)
    figures = []
    for neutral_id, path, *_ in sample:
        buffer = io.BytesIO()
        images(path, phase).save(buffer, format="PNG")
        prep.write_once(folder / f"{neutral_id}.png", buffer.getvalue())
        figures.append(f'<figure><img src="{neutral_id}.png" alt="{neutral_id}"><figcaption>{neutral_id}</figcaption></figure>')
    marks = folder / "marks.csv"
    if not marks.exists():
        prep.write_csv(marks, ["id", *E2F_MARKS], [[row[0], *[""] * len(E2F_MARKS)] for row in sample])
    page = folder / "sheet.html"
    from html import escape
    prep.write_once(page, ('<!doctype html><meta charset="utf-8"><title>Review sheet</title>'
                           '<style>body{background:#111;color:#fff;font:16px system-ui}'
                           'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}'
                           'img{width:100%;height:auto}figure{margin:1rem}</style>'
                           f'<p>{escape(label)}</p><main>{"".join(figures)}</main>').encode("utf-8"))
    return [page, marks]


# ======================================================================= E3 ===
def feature_difference(locked: ev.Features, new: ev.Features, paths: Sequence[str]) -> dict:
    return extract.repeat_check([locked.select(paths)], [new.select(paths)])


def e3(study: Study, p1: Phase1, padded: Path | None = None) -> dict[str, tuple[list[str], list[list]]]:
    """E3 checks, outcomes and criteria (1)-(3)."""
    geometry = locked_geometry(study)
    main_paths = [r["path"] for r in p1.main_inv]
    ext_paths = [r["path"] for r in p1.ext_inv]
    square = [p for p in main_paths if geometry[p]["width"] == geometry[p]["height"]]
    non_square = [p for p in (*main_paths, *ext_paths) if geometry[p]["width"] != geometry[p]["height"]]
    checks, scored, status = [], {}, {}
    for arm in ARMS:
        scored[arm, "locked"], status[arm, "locked"] = score_feature_set(study, study.features(arm), study.external_features(arm)), INTERPRETED
    interpreted = {}
    for arm, variants in E3_VARIANTS.items():
        ref = p1.get("E3", arm, E3_REF)
        if ref is None:
            checks.append([arm, E3_REF, "E3-ref against locked features (5,927 images)", None, UNAVAILABLE])
            interpreted[arm] = False
        else:
            c = p1.control(ref)
            interpreted[arm] = c["passed"]
            checks.append([arm, E3_REF, f"E3-ref against locked features ({c['images']} images)", c["max_difference_ratio"],
                           "pass" if interpreted[arm] else NOT_INTERPRETED])
        locked_all = p1.locked_all(arm)
        for v in (E3_REF, *variants):
            worker = p1.get("E3", arm, v)
            if worker is None:
                if v != E3_REF:
                    checks.append([arm, v, "variant", None, UNAVAILABLE])
                continue
            for field in ("overrides", "code_matches_lock", "environment_matches_lock"):
                checks.append([arm, v, field, json.dumps(worker.identity.get(field), sort_keys=True), ""])
            rep = worker.identity.get("repeat_check") or {}
            checks.append([arm, v, "repeat (recorded)", rep.get("max_difference_ratio"), f"{rep.get('images')} images"])
            inputs = prep.read_csv(worker.folder / "variant_inputs.csv")
            if [r["path"] for r in inputs] != [*main_paths, *ext_paths]:
                raise ValueError(f"{worker.folder}: variant_inputs.csv differs from the locked inventory order")
            changed = sum(r["input_changed"].lower() in ("1", "true") for r in inputs)
            checks.append([arm, v, "inputs changed (worker)", changed, ""])
            constant = sum(r.get("constant_minmax_input", "").lower() in ("1", "true") for r in inputs)
            checks.append([arm, v, "constant min-max inputs (left unchanged)", constant, ""])
            if v == E3_REF:
                continue
            feats = p1.features(arm, "E3", v)
            merged = fx.merge(*feats)
            square_ok = True
            if v in E3_SQUARE:
                sq = feature_difference(locked_all, merged, square)
                square_ok = sq["max_difference_ratio"] <= CONTROL_TOL
                checks.append([arm, v, f"square main images unchanged ({sq['images']})", sq["max_difference_ratio"],
                               "pass" if square_ok else "fail; not interpreted"])
            eff = feature_difference(locked_all, merged, non_square)
            applied = eff["worst_image_difference_ratio"] > EFFECT_MIN
            checks.append([arm, v, f"largest per-image difference on {eff['images']} non-square images",
                           eff["worst_image_difference_ratio"], "change applied" if applied else "no effect, change not applied"])
            scored[arm, v] = score_feature_set(study, *feats)
            status[arm, v] = (NOT_INTERPRETED if not interpreted[arm] else "square check failed; not interpreted"
                              if not square_ok else "no effect, change not applied; not interpreted"
                              if not applied else INTERPRETED)
    if padded is not None:
        folder = padded / NOMINEE
        prep.verify_complete(folder)
        identity = prep.read_json(folder / "identity.json")
        if (identity.get("arm"), identity.get("lock_sha256"), identity.get("external_lock_sha256")) != (
            NOMINEE, prep.sha256(study.lock / "input_lock.json"), prep.sha256(study.external / "external_lock.json")
        ):
            raise ValueError("G1 padded MIRAGE belongs to a different arm or lock")
        dim = extract.DIMENSIONS[NOMINEE]
        feats = (ev.Features(extract.load_features(folder / "features.npz", p1.main_inv, dim), {p: i for i, p in enumerate(main_paths)}),
                 ev.Features(extract.load_features(folder / "features_external.npz", p1.ext_inv, dim), {p: i for i, p in enumerate(ext_paths)}))
        scored[NOMINEE, M_PAD], status[NOMINEE, M_PAD] = score_feature_set(study, *feats), M_PAD_NOTE
    else:
        checks.append([NOMINEE, M_PAD, M_PAD_NOTE, None, UNAVAILABLE])
    dev = {key: dev_macro(study, sc) for key, sc in scored.items()}
    rep = {key: rep_macro(study, sc) for key, sc in scored.items()}
    inc = dev[fm.REFERENCE, "locked"]
    rows, holdout = [], []
    for (arm, v), d in dev.items():
        if v == "locked" and arm not in E3_VARIANTS:
            continue
        rows += [[arm, v, status[arm, v], "macro_recall", k, *summary(x)] for k, x in by_k(d, KS)]
        if v != "locked":
            rows += [[arm, v, status[arm, v], "minus Inception-v3 locked", k, *summary(x)] for k, x in by_k(paired(d, inc), KS)]
            rows += [[arm, v, status[arm, v], f"minus {arm} locked", k, *summary(x)] for k, x in by_k(paired(d, dev[arm, "locked"]), KS)]
            for k, x in by_k(paired(rep[arm, v], rep[fm.REFERENCE, "locked"]), sorted({k for k, _ in rep[arm, v]})):
                m = float(np.mean(x))
                state = status[arm, v]
                reading = ("positive" if m > 0 else "negative" if m < 0 else "zero") if state in (INTERPRETED, M_PAD_NOTE) else ""
                holdout.append([arm, v, state, k, len(x), m, "shown, never read" if k == 16 else reading])
    for arm in E3_VARIANTS:
        variants = (*E3_VARIANTS[arm], M_PAD) if arm == NOMINEE else E3_VARIANTS[arm]
        means = [float(np.mean(k_mean(dev[arm, v]))) for v in variants if (arm, v) in dev]
        if means:
            rows.append([arm, "all variants", "", "range of variant K-mean means", ev.K_MEAN_LABEL, len(means), min(means), None,
                         None, max(means)])
    tables = {"E3_checks.csv": (["arm", "variant", "check", "value", "status"], checks),
              "E3_development.csv": (["arm", "variant", "status", "quantity", "k", *SUMMARY], rows),
              "E3_holdout.csv": (["arm", "variant", "status", "k", "n_draws", "mean_difference_vs_locked_Inception",
                                  "direction"], holdout)}
    nine = []
    locked_rows = {arm: {r[0]: r for r in nine_way_rows(study, scored[arm, "locked"].nine)} for arm in ARMS}
    for (arm, v), sc in scored.items():
        if v == "locked" and arm not in E3_VARIANTS:
            continue
        for r in nine_way_rows(study, sc.nine):
            nine.append([arm, v, status[arm, v], *r])
    tables["E3_nine_way.csv"] = (["arm", "variant", "status", *NINE_HEADER], nine)
    kmean = lambda a, b: k_mean(paired(dev[a], dev[b]))  # noqa: E731
    mirage = {v: kmean((NOMINEE, v), (fm.REFERENCE, "locked")) for v in E3_VARIANTS[NOMINEE]
              if (NOMINEE, v) in dev and status[NOMINEE, v] == INTERPRETED}
    rf_inc = {v: kmean(("RETFound", v), (fm.REFERENCE, "locked")) for v in E3_VARIANTS["RETFound"]
              if ("RETFound", v) in dev and status["RETFound", v] == INTERPRETED}
    m_rf = {v: k_mean(paired(dev[NOMINEE, "locked"], dev["RETFound", v])) for v in rf_inc}
    crit = e3_criteria(mirage, rf_inc, m_rf)
    if (NOMINEE, M_PAD) in dev:
        crit.insert(len(E3_VARIANTS[NOMINEE]) + 1, ["(1) shown beside, not counted", NOMINEE, M_PAD,
                    "p05 of K mean (variant - Inception-v3 locked)", fm.N_DRAWS, success_rule(kmean((NOMINEE, M_PAD), (fm.REFERENCE, "locked")))["p05"], M_PAD_NOTE])
    if not any(("RETFound", v) in dev for v in E3_VARIANTS["RETFound"]):
        crit = [r for r in crit if r[0] != "(2)"] + [["(2)", "RETFound", "", "", None, None, "RETFound unavailable in E3"]]
    unstable, counted_variants = [], 0
    pad_outcomes: dict[str, bool] = {}
    for (arm, v), sc in scored.items():
        if v == "locked" or arm not in E3_VARIANTS:
            continue
        counted = status[arm, v] in (INTERPRETED, M_PAD_NOTE)
        counted_variants += counted
        got = {r[0]: r for r in nine_way_rows(study, sc.nine)}
        for name in NORMAL_SETS[1:]:
            lk = locked_rows[arm][name]
            inside = e3_interval_check(got[name][5], got[name][1], (lk[8], lk[9]))
            counts = {a: locked_rows[a][name][5] for a in ARMS if a != arm} | {arm: got[name][5]}
            ranks = ", ".join(f"{a} {r}" for a, r in sorted(ranks_with_ties(counts).items(), key=lambda row: row[1]))
            r1s = [f"{r[0].split()[1]} {r[5]}/{r[1]}" for r in got.values()
                   if r[0] in RARE_EXTERNAL and r[0].split()[0] == name.split()[0]]
            crit.append(["(3)", arm, v, f"{name} false-rare count inside {arm}'s locked 95% interval",
                         got[name][1], got[name][5], ("inside" if inside else "outside") +
                         f" [{lk[8]:.3f}, {lk[9]:.3f}] as a rate" +
                         f"; NORMAL recall {got[name][3] / got[name][1]:.3f}; R1 {', '.join(r1s)}"
                         f"; ranks among four arms {ranks}"
                         " (description only)" + (f"; {M_PAD_NOTE}" if v == M_PAD else
                                                  "" if counted else f"; {status[arm, v]}, not counted")])
            if v == M_PAD:
                pad_outcomes[name] = inside
            if counted and not inside:
                unstable.append(f"{arm} {v} {name}" + (f" ({M_PAD_NOTE})" if v == M_PAD else ""))
    complete_grid = counted_variants == 1 + sum(map(len, E3_VARIANTS.values()))
    stability = ("outside: " + "; ".join(unstable) if unstable else "stable" if complete_grid else
                 "not evaluable for the full grid; no evaluated variant outside")
    if pad_outcomes and all(pad_outcomes.values()):
        stability += f"; {M_PAD} inside on {', '.join(pad_outcomes)} ({M_PAD_NOTE})"
    elif not pad_outcomes:
        stability += f"; {M_PAD} {UNAVAILABLE} ({M_PAD_NOTE})"
    crit.append(["(3)", "all", "", "result", counted_variants, None, stability])
    crit.append(["scope", "", "", "", None, None, E3_SCOPE])
    tables["E3_criteria.csv"] = (["criterion", "arm", "variant", "statistic", "n", "value", "result"], crit)
    return tables


# ======================================================================= E4 ===
E4_SOURCES = ("Kermany test", "Yoo rare", "OCTID", "OCTDL")
E4_NORMAL_PAIRS = (("Kermany test NORMAL", "OCTID NORMAL"), ("Kermany test NORMAL", "OCTDL NORMAL"),
                   ("OCTID NORMAL", "OCTDL NORMAL"))


def e4_pairs() -> list[tuple[str, str]]:
    import itertools
    return [*itertools.combinations(E4_SOURCES, 2), *E4_NORMAL_PAIRS]


def e4_members(study: Study) -> dict[str, tuple[list[str], list[str]]]:
    """Source -> (paths, units): Kermany by pid, rare and OCTID by image, OCTDL by patient."""
    query = [r for r in study.roster if r["role"] == "query"]
    rare = [r["path"] for r in prep.read_csv(study.lock / "extraction_inventory.csv") if r["path"].startswith("data/raw/rare/")]
    out = {"Kermany test": [r["path"] for r in query if r["label"] in COMMON],
           "Kermany test NORMAL": [r["path"] for r in query if r["label"] == "NORMAL"],
           "Yoo rare": rare}
    for source in ("OCTID", "OCTDL"):
        out[source] = [r["path"] for r in study.queries if r["source"] == source]
        out[f"{source} NORMAL"] = [r["path"] for r in study.queries if r["source"] == source and r["label"] == "NORMAL"]

    def unit_of(p: str) -> str:
        if p.startswith("data/raw/OCT2017/"):
            return "Kermany:" + ev.kermany_pid(p)
        return "OCTDL:" + study.patient[p] if p in study.patient else p
    return {name: (paths, [unit_of(p) for p in paths]) for name, paths in out.items()}


def e4_tables(members: Mapping[str, tuple[Sequence[str], Sequence[str]]], columns: Mapping[str, Mapping[str, np.ndarray]],
              column_names: Mapping[str, Sequence[str]], arm_sets: Mapping[str, Mapping[str, np.ndarray]],
              thumbs: Mapping[str, Mapping[str, np.ndarray]], drop: set[str] = frozenset(),
              n_boot: int = E_BOOT, draws: int = fm.N_DRAWS,
              status: Mapping[str, str] | None = None) -> dict[str, tuple[list[str], list[list]]]:
    """E4 on per-image vectors. columns: G / S-raw / S-E1 -> path -> values (column_names gives their names);
    arm_sets: 'MIRAGE locked', 'MIRAGE E1-M', ... -> path -> features; thumbs: 'raw' / 'E1-M' -> path -> vector."""
    tables = {}
    status = status or {}
    pairs = e4_pairs()
    stack = lambda table, paths: np.stack([np.asarray(table[p], dtype=np.float64) for p in paths])  # noqa: E731
    # Main measure: few-shot source test on the normal-only pairs.
    rows = []
    for pair_index, (a, b) in enumerate(pairs):
        if (a, b) not in E4_NORMAL_PAIRS:
            continue
        paths = [*members[a][0], *members[b][0]]
        source = np.array([0] * len(members[a][0]) + [1] * len(members[b][0]))
        units = [*members[a][1], *members[b][1]]
        results = {}
        for name, table in arm_sets.items():
            if all(p in table for p in paths):
                results[name] = source_fewshot(stack(table, paths), source, units, pair_index, prototype_predict, draws=draws)
        for name, table in thumbs.items():
            if all(p in table for p in paths):
                results[f"thumbnail {name}"] = source_fewshot(stack(table, paths), source, units, pair_index,
                                                              nearest_mean_predict, draws=draws)
        for name, values in results.items():
            rows += [[f"{a} vs {b}", name, "two-way macro recall (chance 0.50)", k, *summary(v),
                      status.get(name, ""), ""] for k, v in by_k(values, KS)]
        controlled = {name: name.replace(" locked", " E1-M") for name in results if name.endswith(" locked")}
        controlled["thumbnail raw"] = "thumbnail E1-M"
        for name, other in controlled.items():
            if name in results and other in results:
                diff = paired(results[other], results[name])
                for k, v in by_k(diff, KS):
                    reading = ""
                    state = status.get(other, INTERPRETED)
                    if k == ev.K_MEAN_LABEL and state == INTERPRETED and v:
                        reading = "the control reduced separability" if np.percentile(v, 95) < 0 else "no reduction shown"
                    rows.append([f"{a} vs {b}", f"{other} minus {name}", "paired difference", k, *summary(v), state, reading])
        rows += [[f"{a} vs {b}", name, "", "", *[None] * len(SUMMARY), UNAVAILABLE, ""]
                 for name, state in status.items() if state == UNAVAILABLE]
    tables["E4a_fewshot_source_test.csv"] = (["pair", "feature_set", "quantity", "k", *SUMMARY, "status", "reading"], rows)
    # Secondary: per-column AUROC with no fit (second source positive).
    rows = []
    for a, b in pairs:
        for set_name, table in columns.items():
            pa, pb = members[a][0], members[b][0]
            if not pa or not pb or not all(p in table for p in (*pa, *pb)):
                continue
            A, B = stack(table, pa), stack(table, pb)
            for j, col in enumerate(column_names[set_name]):
                value = auroc(B[:, j], A[:, j])
                sep = max(value, 1 - value)
                rows.append([f"{a} vs {b}", set_name, col, len(pa), len(pb), value, sep, e4_reading(sep),
                             "different sources: source and disease are mixed" if "Yoo rare" in (a, b) else ""])
    tables["E4b_column_auroc.csv"] = (["pair", "feature_set", "column", "n_first", "n_second", "auroc_second_positive",
                                       "separability", "reading", "note"], rows)
    if "E4-probe" in drop:
        return tables
    rows, diffs = [], []
    for pair_index, (a, b) in enumerate(pairs):
        sets = {**{k: (v, False) for k, v in columns.items()}, **{k: (v, True) for k, v in arm_sets.items()}}
        paths = [*members[a][0], *members[b][0]]
        y = np.array([0] * len(members[a][0]) + [1] * len(members[b][0]))
        units = np.array([*members[a][1], *members[b][1]])
        folds = grouped_folds(y, units, pair_index)
        oof = {}
        for name, (table, rows_normal) in sets.items():
            if not all(p in table for p in paths):
                continue
            scores, failures = probe_scores(stack(table, paths), y, folds, rows_normal)
            oof[name] = (scores, failures)
        rng = np.random.default_rng([fm.SEED, E4_BOOT_NS, pair_index])
        g0, g1 = unit_index(units[y == 0].tolist()), unit_index(units[y == 1].tolist())
        i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        draws_idx = [(i0[resample_units(g0, rng)], i1[resample_units(g1, rng)]) for _ in range(n_boot)]
        boot = {}
        for name, (scores, failures) in oof.items():
            point = auroc(scores[y == 1], scores[y == 0])
            boot[name] = np.array([auroc(scores[j1], scores[j0]) for j0, j1 in draws_idx])
            lo, hi = np.percentile(boot[name], [2.5, 97.5])
            state = status.get(name, INTERPRETED)
            rows.append([f"{a} vs {b}", name, len(i0), len(i1), len(g0), len(g1), point, lo, hi, state,
                         e4_reading(point) if state == INTERPRETED else "", failures,
                         "different sources: source and disease are mixed" if "Yoo rare" in (a, b) else ""])
        point = {name: auroc(s[y == 1], s[y == 0]) for name, (s, _) in oof.items()}
        comparisons = [("S-E1", "S-raw")] + [(f"{arm} E1-M", f"{arm} locked") for arm in ARMS] + \
            [(f"{arm} locked", "S-raw") for arm in ARMS]
        for new, old in comparisons:
            if new in boot and old in boot:
                d = boot[new] - boot[old]
                lo, hi = np.percentile(d, [2.5, 97.5])
                state = status.get(new, INTERPRETED)
                diffs.append([f"{a} vs {b}", f"{new} minus {old}", point[new] - point[old], lo, hi, state,
                              e4_pair_reading(point[old], point[new], lo, hi, not new.endswith(" locked"))
                              if state == INTERPRETED else "",
                              "different sources: source and disease are mixed" if "Yoo rare" in (a, b) else ""])
    tables["E4c_probe_auroc.csv"] = (["pair", "feature_set", "n_first", "n_second", "units_first", "units_second", "auroc",
                                      "boot95_low", "boot95_high", "status", "reading", "fits_not_converged", "note"], rows)
    tables["E4d_probe_differences.csv"] = (["pair", "comparison", "auroc_difference", "boot95_low", "boot95_high",
                                           "status", "reading", "note"], diffs)
    return tables


def e4_inputs(study: Study, p1: Phase1) -> tuple[dict, dict, dict, dict, dict]:
    geometry = locked_geometry(study)
    columns = {"G": {p: np.array([g[c] for c in GEOMETRY_G]) for p, g in geometry.items()}}
    names = {"G": list(GEOMETRY_G)}
    for key, which in (("S-raw", "raw"), ("S-E1", "E1-M")):
        stats = p1.pixel_stats(which)
        if stats:
            columns[key], names[key] = stats, list(PIXEL_STATS)
    arm_sets, status = {}, {}
    for arm in ARMS:
        for s in ("locked", "E1-M"):
            feats = p1.features(arm, s)
            status[f"{arm} {s}"] = p1.status(arm, s)
            if feats is not None:
                merged = fx.merge(*feats)
                arm_sets[f"{arm} {s}"] = {p: merged.values[i] for p, i in merged.index.items()}
    thumbs = {k: v for k, v in p1.thumbs().items() if k in ("raw", "E1-M")}
    return columns, names, arm_sets, thumbs, status


# =================================================================== E report ===
E1_FILES = ("E1a_checks.csv", "E1b_development.csv", "E1c_paired.csv", "E1d_success.csv", "E1e_matched.csv",
            "E1f_holdout.csv", "E1g_nine_way.csv", "E1h_leave_one_out.csv")


def _read(out: Path, name: str) -> list[dict[str, str]]:
    path = out / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def e1_report(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    """E1_comparison.md from the E1 CSVs (rows as csv.DictReader gives them); nothing is computed."""
    success = tables["E1d_success.csv"]
    label = success[0]["posthoc"]
    out = ["# E1 size and framing control (post hoc, Phase-1)", "",
           f"Label of every row: {label}.", "",
           "Only E1-M gives the verdict on M1; E1-L, E1-C, the references and the hold-out are descriptions and cannot "
           "overturn or rescue it. The development spread is over support draws on 124 fixed queries, not a confidence interval.", "",
           'Pixel-only reference (choice fixed in code before any E number). "Is above" means p05 > 0 across the '
           "100 paired MIRAGE E1-M minus E1-M thumbnail K-mean differences. The quoted mean is the mean of those differences.", "",
           "## 1. Success table (E1d)", "", "| Line | Set | Status | n | Mean | SD | p05 | Draws above 0 | Verdict |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in success:
        if r["line"].startswith(("fixed sentence", "limitations")):
            continue
        out.append(f"| {r['line']} | {r['feature_set']} | {r['status']} | {r['n_draws']} | {_num(r['mean'])} | {_num(r['sd'])} | "
                   f"{_num(r['p05'])} | {r['draws_above_0']} | {r['verdict']} |")
    for r in success:
        if r["line"] == "fixed sentence" and r["text"]:
            out += ["", f"**Sentence.** {r['text']}"]
        if r["line"].startswith("limitations"):
            out += ["", f"**Limitations.** {r['text']}."]
    checks = tables["E1a_checks.csv"]
    out += ["", "## 2. Checks (E1a)", "", "| Check | Arm | Set | Quantity | Value | Status | Note |", "|---|---|---|---|---|---|---|"]
    out += [f"| {r['check']} | {r['arm']} | {r['feature_set']} | {r['quantity']} | {r['value']} | {r['status']} | {r['note']} |"
            for r in checks]
    out += ["", "## 3. Development, K mean (E1b; per K in the CSV)", "", "| Model | Set | Status | Mean | SD | p05 | p95 |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {r['model']} | {r['feature_set']} | {r['status']} | {_num(r['mean'])} | {_num(r['sd'])} | {_num(r['p05'])} | {_num(r['p95'])} |"
            for r in tables["E1b_development.csv"] if r["k"] == ev.K_MEAN_LABEL]
    out += ["", "## 4. Paired differences, K mean (E1c)", "", "| Comparison | Set | Status | Mean | p05 | p95 |", "|---|---|---|---|---|---|"]
    out += [f"| {r['comparison']} | {r['feature_set']} | {r['status']} | {_num(r['mean'])} | {_num(r['p05'])} | {_num(r['p95'])} |"
            for r in tables["E1c_paired.csv"] if r["k"] == ev.K_MEAN_LABEL]
    if tables.get("E1e_matched.csv"):
        out += ["", "## 5. Matched subset (E1e; supports are not matched)", "", "| Model | Set | k | n queries | Classes | Mean | p05 |",
                "|---|---|---|---|---|---|---|"]
        out += [f"| {r['model']} | {r['feature_set']} | {r['k']} | {r['n_queries']} | {r['classes_present']} | {_num(r['mean'])} | {_num(r['p05'])} |"
                for r in tables["E1e_matched.csv"] if r["k"] in (ev.K_MEAN_LABEL, "")]
    out += ["", "## 6. Hold-out, legacy 22 (E1f; direction only, K = 16 shown and never read)", "",
            "| Comparison | Set | k | Mean paired difference | Direction |", "|---|---|---|---|---|"]
    out += [f"| {r['model']} | {r['feature_set']} | {r['k']} | {_num(r['mean'])} | {r['direction']} |"
            for r in tables["E1f_holdout.csv"] if r["quantity"] == "paired_macro_recall"]
    out += ["", "## 7. Nine-way audit (E1g)", "",
            "| Model | Set | Status | Query set | n | Correct | NORMAL | CNV, DME or DRUSEN | Any rare | NORMAL recall | Rate | 95% (locked method) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tables["E1g_nine_way.csv"]:
        ci = f"{_num(r['ci95_low'])} to {_num(r['ci95_high'])}" if r["ci95_low"] else ""
        out.append(f"| {r['model']} | {r['feature_set']} | {r['status']} | {r['query_set']} | {r['n']} | "
                   f"{r['correct_class']} | {r['called_NORMAL']} | {r['called_CNV_DME_DRUSEN']} | {r['called_any_rare']} | "
                   f"{_num(r['NORMAL_recall'])} | {r['rate']} {_num(r['rate_value'])} | {ci} |")
    if tables.get("E1h_leave_one_out.csv"):
        out += ["", "## 8. MIRAGE minus Inception-v3 K mean, each development query left out (E1h)", "",
                "| Set | Full | Smallest | Image | Largest | Image |", "|---|---|---|---|---|---|"]
        out += [f"| {r['feature_set']} | {_num(r['full_difference'])} | {_num(r['smallest'])} | {r['smallest_image']} | "
                f"{_num(r['largest'])} | {r['largest_image']} |" for r in tables["E1h_leave_one_out.csv"]]
    out += ["", "Not allowed: that size or framing \"explains\" or \"does not explain\" the locked advantage (say \"survives\" "
            "or \"does not survive this control\"); \"keeps\" without the mean beside it; any reading of K = 16; E1-L or E1-C "
            "as the main result; \"robust to framing\" beyond these three variants."]
    return "\n".join(out) + "\n"


def write_e1_report(out: Path) -> Path:
    tables = {name: _read(out, name) for name in E1_FILES}
    if not tables["E1d_success.csv"]:
        raise SystemExit(f"{out} lacks E1d_success.csv: run `fm_posthoc.py e1` first")
    path = out / "E1_comparison.md"
    prep.write_once(path, e1_report(tables).encode("utf-8"))
    return path


def e2_report(rows: Sequence[Mapping[str, str]]) -> str:
    """Summarize the registered E2d table and the number of intervals excluding 0.5."""
    intervals = [r for r in rows if r["boot95_low"] and r["boot95_high"]]
    excluded = sum(r["interval_excludes_0_5"] == "True" for r in intervals)
    lines = ["# E2 external error analysis", "", rows[0]["posthoc"], "",
             f"{excluded} of {len(intervals)} reported E2d intervals exclude 0.5.", "",
             "## Locked-feature OCTDL normals", "",
             "| Arm | Property | Failures | Common disease | False rare | AUROC | 95% interval |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["main_text_table"] == "True":
            interval = (f"{_num(row['boot95_low'])} to {_num(row['boot95_high'])}" if row["boot95_low"] else "—")
            lines.append(f"| {row['arm']} | {row['property']} | {row['n_failed']} | {row['failed_common']} | "
                         f"{row['failed_rare']} | {_num(row['auroc_failed_vs_not'])} | {interval} |")
    lines += ["", "E2a records all destinations, E2b the overlap across arms, and E2c the locked-to-E1-M transitions.",
              "The intervals describe associations among these scans. E2f uses one non-clinician rater."]
    return "\n".join(lines) + "\n"


# ===================================================================== E CLI ===
def write_tables(out: Path, label: str, tables: Mapping[str, tuple[list[str], list[list]]]) -> list[Path]:
    return [write(out, name, label, header, rows) for name, (header, rows) in tables.items()]


def run_phase1(args: argparse.Namespace, out: Path) -> list[Path]:
    """Steps e1-e4: rule 1 first (rule_commit and the subsection hash), then read, score and write once."""
    labels, section = phase1_labels(args.date)
    if getattr(args, "dry_run", False):
        prep.experiment_out(out)
        labels = {k: f"{label}; dry run using locked features as stand-ins" for k, label in labels.items()}
    drop = set(args.drop.split(",")) if getattr(args, "drop", None) else set()
    if drop - set(DROPPABLE):
        raise SystemExit(f"--drop takes only {', '.join(DROPPABLE)}")
    written = []
    with tempfile.TemporaryDirectory() as tmp:
        study = Study(args.attempt, Path(tmp))
        cache = Path(tempfile.gettempdir()) / "few-shot-phase1-render"
        root, receipts, rendered = phase1_root(args.phase1, Path(tmp), cache)
        p1 = Phase1(study, root, receipts, dry_run=getattr(args, "dry_run", False), partial=args.partial,
                    cached_render=rendered)
        if args.step == "e1d":
            names = ("P10_review_key.csv", "P10_review_marks.csv", "P10_sensitivity.csv")
            key_rows, mark_rows, p10_rows = (_read(out, name) for name in names)
            if not (key_rows and mark_rows and p10_rows):
                raise SystemExit(f"{out} lacks the P10 key, marks or sensitivity CSV that rule E1-D reuses")
            label = f"{e1d_label(args.date)}; marks SHA256 {prep.sha256(out / 'P10_review_marks.csv')[:12]}"
            written.append(write(out, "E1D_duplicate_sensitivity.csv", label,
                                 *e1d(study, p1, key_rows, mark_rows, p10_rows)))
        elif args.step == "e1":
            written += write_tables(out, f"{labels['E1']}; limitations: {E1_LIMITATIONS}",
                                    e1(study, p1, section, args.data_root, drop))
        elif args.step == "e2":
            if args.sheet_pass == 2 and (args.sheet is None or "E2f" in drop):
                raise SystemExit("--sheet-pass 2 requires --sheet and cannot be combined with --drop E2f")
            cells = e2_cells(study, p1)
            geometry = locked_geometry(study)
            grey = p1.pixel_stats("raw")
            properties = {p: {**g, **({"grey_mean": grey[p][0], "grey_sd": grey[p][1]} if p in grey else {})}
                          for p, g in geometry.items()}
            fmt = {r["path"]: r["format"] for r in p1.render_rows()}
            mode = {r["path"]: r["mode"] for r in prep.read_csv(study.external / "external_inventory.csv")}
            categorical = {p: {"mode": mode[p], **({"format": fmt[p]} if p in fmt else {})} for p in mode}
            if args.sheet_pass == 1:
                written += write_tables(out, labels["E2"], e2_tables(study, cells, properties, categorical, drop))
            if "E2f" not in drop and args.sheet is not None:
                groups, anchors = e2f_groups(study, cells)
                sample = contact_sheet_sample(groups, anchors)
                written += build_contact_sheet(sample, e2f_images(p1, args.data_root), args.sheet, args.sheet_pass, labels["E2"])
        elif args.step == "e3":
            g1_drive = args.attempt / "g1"
            padded = (padded_root(args.padded or g1_drive, Path(tmp))
                      if args.padded or g1_drive.is_dir() else None)
            written += write_tables(out, f"{labels['E3']}; {E3_SCOPE}", e3(study, p1, padded))
        else:
            import sklearn
            members = e4_members(study)
            columns, names, arm_sets, thumbs, status = e4_inputs(study, p1)
            label = f"{labels['E4']}; laptop run of record, scikit-learn {sklearn.__version__}"
            written += write_tables(out, label, e4_tables(members, columns, names, arm_sets, thumbs, drop, status=status))
    return written


def e2f_images(p1: Phase1, data_root: Path | None) -> Callable[[str, int], object]:
    """Pass 1: the E1-M square from the render archive; pass 2: the original scaled to 640 px wide
    (render archive copy640 for external images, else the raw file under --data-root)."""
    from PIL import Image
    rows = {r["path"]: r for r in p1.render_rows()}
    external = {r["path"] for r in p1.ext_inv}

    def load(path: str, n: int):
        if p1.render is None:
            raise ValueError("the verified render archive is required for E2f")
        if n == 1:
            with Image.open(render_png(p1.render, "E1-M", path)) as image:
                return image.convert("RGB")
        if path in external:
            source = render_png(p1.render, "external640", path)
        else:
            source = (data_root or Path(".")) / prep.safe_id(path)
            if prep.sha256(source) != rows[path]["sha256"]:
                raise ValueError(f"review original differs from the input lock: {path}")
        with Image.open(source) as image:
            image = image.convert("RGB")
            if image.width != 640:
                image = image.resize((640, max(1, round(image.height * 640 / image.width))), Image.Resampling.BICUBIC)
            return image
    return load


def e2f_marks(marks: Path, key: Path | None, out: Path, label: str) -> list[str]:
    """Hash a marks file (the SHA256 goes into CONTEXT.md); with the key, write the unblinded counts."""
    digest = prep.sha256(marks)
    lines = [f"SHA256 {digest}  {marks}"]
    if key is not None:
        sheet = key.parent.parent
        if marks.resolve() not in {(sheet / phase / "marks.csv").resolve() for phase in ("pass1", "pass2")}:
            raise ValueError("--marks must be the completed pass1 or pass2 marks.csv from this sheet")
        roster = prep.read_csv(key)
        ids = [r["id"] for r in roster]
        for phase in ("pass1", "pass2"):
            source = sheet / phase / "marks.csv"
            validate_marks(prep.read_csv(source), ids)
            digest = prep.sha256(source)
            prep.write_once(source.with_suffix(".sha256"), (digest + "\n").encode("ascii"))
            lines.append(f"SHA256 {digest}  {source}")
        counts = mark_counts(prep.read_csv(marks), roster)
        path = write(out, f"E2f_marks_{marks.parent.name}.csv", label,
                     ["source", "group", "group_size", "mark", "value", "count", "note"], counts)
        lines.append(f"wrote {path}")
    return lines


# --- Phase-1 number freeze (protocols.md section 8) ------------------------------------------
# The freeze fingerprints every tracked file of record. After it, a frozen file changes only through an
# amendment that names the reason and keeps the old and new SHA256 (the correction of a proven error, or an
# exception the change log names); `freeze-check` (run by .agents/check) fails on any other change.
REPO = Path(__file__).resolve().parent
FREEZE_FILE = "results/fm_v1/phase1_freeze.json"
# Step 1 (22 Nov) freezes the files of record; step 2 (by 27 Nov) adds the paper's tables and figures with
# `freeze --scope docs/report/fm_v1_paper/tables docs/report/fm_v1_paper/figures --reason ...`.
FREEZE_SCOPE = ("results/fm_v1/posthoc",)
TEXT_SUFFIXES = {".csv", ".md", ".tex", ".json", ".txt", ".py"}


def fingerprint(name: str, data: bytes) -> str:
    """SHA256; text files have CRLF folded to LF so Windows and Colab checkouts agree (as build_report_evidence.py)."""
    if Path(name).suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def freeze_diff(frozen: Mapping[str, str], now: Mapping[str, str]) -> dict[str, list[str]]:
    """Files changed, added to or removed from the frozen set (each list sorted)."""
    return {"changed": sorted(p for p in frozen.keys() & now.keys() if frozen[p] != now[p]),
            "added": sorted(now.keys() - frozen.keys()), "removed": sorted(frozen.keys() - now.keys())}


def freeze_amend(manifest: Mapping, now: Mapping[str, str], date: str, reason: str,
                 scope: Sequence[str] | None = None) -> dict:
    """A new manifest whose files are `now` (and scope `scope`, if given), with one amendment recording every
    difference and its reason."""
    diff = freeze_diff(manifest["files"], now)
    if not any(diff.values()):
        raise ValueError("nothing changed since the freeze: no amendment to record")
    if not reason.strip():
        raise ValueError("an amendment needs a reason (the change-log entry that allows it)")
    changes = {p: [manifest["files"].get(p), now.get(p)] for kind in ("changed", "added", "removed") for p in diff[kind]}
    amendment = {"date": date, "reason": reason, "changes": changes}
    new = {**manifest, "files": dict(sorted(now.items()))}
    if scope is not None and list(scope) != list(manifest["scope"]):
        amendment["scope"], new["scope"] = list(scope), list(scope)
    new["amendments"] = [*manifest.get("amendments", []), amendment]
    return new


def tracked_fingerprints(scope: Sequence[str], repo: Path = REPO) -> dict[str, str]:
    """Fingerprint of every git-tracked file under `scope`, read from the working tree."""
    names = subprocess.run(["git", "ls-files", "-z", "--", *scope], cwd=repo, check=True,
                           capture_output=True).stdout.decode("utf-8").split("\0")
    return {n: fingerprint(n, (repo / n).read_bytes()) for n in sorted(filter(None, names))}


def freeze_report(diff: Mapping[str, Sequence[str]]) -> list[str]:
    return [f"{kind}: {p}" for kind in ("changed", "added", "removed") for p in diff[kind]]


def run_freeze(args: argparse.Namespace) -> list[str]:
    path = REPO / FREEZE_FILE
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if args.step == "freeze-check":
        if old is None:
            return ["Phase-1 freeze: not frozen yet"]
        now = tracked_fingerprints(old["scope"])
        problems = freeze_report(freeze_diff(old["files"], now))
        if problems:
            raise SystemExit("Phase-1 freeze broken (amend with `fm_posthoc.py freeze --reason ...`):\n  "
                             + "\n  ".join(problems))
        return [f"Phase-1 freeze: {len(now)} files unchanged"]
    if old is None:
        scope = list(dict.fromkeys([*FREEZE_SCOPE, *(args.scope or [])]))
        now = tracked_fingerprints(scope)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True)
        manifest = {"frozen": args.date, "base_commit": head.stdout.strip(), "scope": scope,
                    "rule": args.reason, "files": now, "amendments": []}
    else:
        scope = list(dict.fromkeys([*old["scope"], *(args.scope or [])]))
        now = tracked_fingerprints(scope)
        manifest = freeze_amend(old, now, args.date, args.reason, scope)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return [f"wrote {path} ({len(now)} files)"]


def add_phase1_commands(sub) -> None:
    for step in ("e1", "e2", "e3", "e4", "e1d"):
        command = sub.add_parser(step, description=f"Phase-1 {step.upper()} (protocols.md section 8, Phase-1 controls"
                                 + ("; rule E1-D, 21 Nov change log; reads the P10 CSVs in --out)." if step == "e1d" else ")."))
        command.add_argument("--out", type=Path, default=OUT)
        command.add_argument("--attempt", type=Path, required=True, help=f"the locked attempt folder ({ATTEMPT})")
        command.add_argument("--phase1", type=Path, required=True,
                             help="Drive phase1/ folder with the phase1-*.zip archives and receipts, or the unzipped root")
        command.add_argument("--date", default=datetime.date.today().isoformat())
        command.add_argument("--partial", action="store_true",
                             help="allow missing render or E1-M arms in an incomplete Phase-1 delivery")
        command.add_argument("--drop", help=f"laptop drop order, comma-separated subset of {','.join(DROPPABLE)}")
        if step == "e1":
            command.add_argument("--data-root", type=Path, help="checkout whose data/raw holds the images (laptop re-render)")
            command.add_argument("--dry-run", action="store_true", help="label stand-in output and require scratch --out")
        if step == "e2":
            command.add_argument("--data-root", type=Path, help="checkout whose data/raw holds the images (E2f pass 2)")
            command.add_argument("--sheet", type=Path, help="E2f contact-sheet folder in Drive; never inside the repository")
            command.add_argument("--sheet-pass", type=int, choices=(1, 2), default=1,
                                 help="generate E2f pass 2 after filling pass 1 marks")
        if step == "e3":
            command.add_argument("--padded", type=Path, help="G1 folder for M-pad (as for the g1 step)")
    command = sub.add_parser("e-report", description="Write E1_comparison.md from the E1 CSVs in --out (nothing is computed).")
    command.add_argument("--out", type=Path, default=OUT)
    command = sub.add_parser("e2f-marks", description="SHA256 of an E2f marks file; with --key, the unblinded counts.")
    command.add_argument("--out", type=Path, default=OUT)
    command.add_argument("--marks", type=Path, required=True)
    command.add_argument("--key", type=Path)
    command.add_argument("--date", default=datetime.date.today().isoformat())
    command = sub.add_parser("freeze", description=f"Write {FREEZE_FILE}: the SHA256 of every tracked file of record "
                             "(first call), or one amendment that records every change since (later calls).")
    command.add_argument("--reason", required=True, help="the protocols.md change-log entry that freezes or allows it")
    command.add_argument("--scope", nargs="+", help=f"folders to add to the frozen scope (always: {', '.join(FREEZE_SCOPE)})")
    command.add_argument("--date", default=datetime.date.today().isoformat())
    sub.add_parser("freeze-check", description=f"Fail if a file of record differs from {FREEZE_FILE}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    for step in ("run", "g1", "g1-report", "p10-links", "p10-screen", "p10-sensitivity"):
        command = sub.add_parser(step)
        command.add_argument("--out", type=Path, default=OUT)
        if step == "g1-report":
            command.description = "Rewrite G1_comparison.md from the five CSVs already in --out (no score is computed)."
            continue
        if step == "p10-links":
            command.description = "P10 step 1: the links of S1 in Yoo et al.'s supplement, by disease and kind."
            command.add_argument("--s1-text", type=Path, required=True, help="text of the supplement PDF, one line per line")
            command.add_argument("--s1-links", type=Path, required=True,
                                 help="link annotations of the PDF, one per line as <page><TAB><url>")
            command.add_argument("--date", default=datetime.date.today().isoformat())
            continue
        command.add_argument("--attempt", type=Path, required=True, help=f"the locked attempt folder ({ATTEMPT})")
        command.add_argument("--date", default=datetime.date.today().isoformat(), help="date in the posthoc label")
        if step == "run":
            command.add_argument("--only", help="comma-separated subset, e.g. P8,P3")
        elif step == "p10-screen":
            command.add_argument("--data-root", type=Path, required=True,
                                 help="checkout whose data/raw holds the locked images (read-only)")
            command.add_argument("--sheet", type=Path, required=True,
                                 help="where to write the blind review sheet (HTML with images); never inside the repo")
        elif step == "p10-sensitivity":
            command.description = "P10 step 3 sensitivity from P10_review_key.csv and Lam's P10_review_marks.csv in --out."
        else:
            command.add_argument("--padded", type=Path, required=True,
                                 help="folder holding MIRAGE/ and Inception/ from G1, or the Drive g1/ folder with the "
                                      "g1-<arm>-<stamp>.zip archives and their .json receipts")
    add_phase1_commands(sub)
    args = parser.parse_args()
    if args.step in ("freeze", "freeze-check"):
        for line in run_freeze(args):
            print(line)
        return
    out = prep.writable_path(Path(__file__).resolve().parent, args.out)
    if args.step in ("e1", "e2", "e3", "e4", "e1d"):
        for path in run_phase1(args, out):
            print("wrote", path)
    elif args.step == "e-report":
        print("wrote", write_e1_report(out))
        properties = _read(out, "E2d_properties.csv")
        if properties:
            page = out / "E2_comparison.md"
            prep.write_once(page, e2_report(properties).encode("utf-8"))
            print("wrote", page)
    elif args.step == "e2f-marks":
        for line in e2f_marks(args.marks, args.key, out, phase1_labels(args.date)[0]["E2"]):
            print(line)
    elif args.step == "run":
        for path in run(args.attempt, out, set(args.only.split(",")) if args.only else None, args.date):
            print("wrote", path)
    elif args.step == "g1-report":
        print("wrote", write_g1_report(out))
    elif args.step == "p10-links":
        lines = args.s1_text.read_text(encoding="utf-8").splitlines()
        urls = [line.split("\t", 1)[1].strip() for line in args.s1_links.read_text(encoding="utf-8").splitlines()
                if "\t" in line]
        print("wrote", write(out, "P10_source_links.csv", p10_label(args.date), *p10_links(s1_links(lines, urls))))
    elif args.step == "p10-screen":
        repo = Path(__file__).resolve().parent
        sheet = args.sheet.resolve()
        if repo in sheet.parents:
            raise SystemExit("the review sheet holds third-party images: write it outside the repository")
        label = p10_label(args.date)
        with tempfile.TemporaryDirectory() as tmp:
            tables, images = p10_screen(Study(args.attempt, Path(tmp)), args.data_root)
        for name in P10_SCREEN_FILES:
            print("wrote", write(out, name, label, *tables[name]))
        sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.write_text(p10_sheet(tables["P10_review_key.csv"][1], images), encoding="utf-8")
        print("wrote", sheet)
    elif args.step == "p10-sensitivity":
        key_rows, mark_rows = _read(out, "P10_review_key.csv"), _read(out, "P10_review_marks.csv")
        label = f"{p10_label(args.date)}; marks SHA256 {prep.sha256(out / 'P10_review_marks.csv')[:12]}"
        with tempfile.TemporaryDirectory() as tmp:
            h1, rows, h2, pairs = p10_sensitivity(Study(args.attempt, Path(tmp)), key_rows, mark_rows)
        print("wrote", write(out, "P10_sensitivity.csv", label, h1, rows))
        print("wrote", write(out, "P10_matched_pairs.csv", label, h2, pairs))
    else:
        if not (out / "P9_rare_vs_normal_auroc.csv").exists():
            raise SystemExit(f"{out} lacks the locked P9 tables: run `fm_posthoc.py run --only P9 --out {out}` first")
        with tempfile.TemporaryDirectory() as tmp:
            study = Study(args.attempt, Path(tmp))
            padded = padded_root(args.padded, Path(tmp))
            path = write(out, "G1_pad_to_square.csv", labels_for(args.date)["G1"],
                         ["arm", "table", "k", "quantity", "locked", "padded_label", "padded", "size_only_label", "size_only"],
                         g1(study, padded))
            print("wrote", path)
            label = f"{p9_label(args.date)}; on G1's padded features, {labels_for(args.date)['G1']}"
            h1, rows, h2, diffs = p9_padded(study, padded)
            for name, h, r in (("P9_G1_padded_auroc.csv", h1, rows), ("P9_G1_padded_differences.csv", h2, diffs)):
                print("wrote", write(out, name, label, h, r))
        print("wrote", write_g1_report(out))


if __name__ == "__main__":
    main()
