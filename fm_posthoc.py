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


def p10_label(today: str) -> str:
    return f"post hoc, rule timestamped by commit {rule_commit(P10_ROW)[:7]} (18 Nov 2026), computed {today}"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    for step in ("run", "g1", "g1-report", "p10-links", "p10-screen"):
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
        else:
            command.add_argument("--padded", type=Path, required=True,
                                 help="folder holding MIRAGE/ and Inception/ from G1, or the Drive g1/ folder with the "
                                      "g1-<arm>-<stamp>.zip archives and their .json receipts")
    args = parser.parse_args()
    out = prep.writable_path(Path(__file__).resolve().parent, args.out)
    if args.step == "run":
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
