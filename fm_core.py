"""Pure logic for the frozen-encoder rare-OCT study (docs/research/FM_PROTOCOL.md, v1.0).

Prepared for Lam's review. It holds the rules the protocol fixes
exactly: the image screen, grouping and folds (section 4), paired nested supports and both
classifiers (section 6), nomination (section 7) and false-rare rate (section 8), plus the
amendment A2 diagnostics (padding shares, size-only reference) and nine-way intervals.

No file, network, GPU or framework access: callers pass paths, arrays and link lists in, and
own all I/O, identities and persistence. Random streams are a function of NumPy's version, so
realized support IDs must be saved (section 6: "Save IDs, not seeds alone").
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

RARE = ("CSC", "MH", "MacTel", "Stargardt", "RP")  # fixed order; also resolves exact ties
NINE = ("NORMAL", "CNV", "DME", "DRUSEN", *RARE)
N_FOLDS = 5
K_DEV = (1, 2, 4, 8)
K_REPORT = (1, 2, 4, 8, 16)
N_DRAWS = 100
MIN_FITTING_GROUPS = max(K_DEV)
SEED = 42
FOLD_NS, DEV_NS, REPORT_NS = 5101, 6101, 7101  # member-choice streams use namespace + 1
HAMMING_MAX = 4
NEAR_BEST = 0.02
# RETFound/DINOv2/MIRAGE order; BioMedCLIP takes RETFound's slot if substituted (section 7).
TIE_SLOT = {"RETFound": 0, "BioMedCLIP": 0, "DINOv2": 1, "MIRAGE": 2}
REFERENCE = "Inception"


def stream(*entropy: int) -> np.random.Generator:
    """The protocol's PCG64 stream for a SeedSequence entropy list."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(entropy))))


# --------------------------------------------------------------- image screen (section 4) ---
def rgb_fingerprint(image: Image.Image) -> str:
    """Hash of decoded RGB pixels and dimensions; alpha is dropped, as the protocol requires."""
    rgb = image.convert("RGB")
    payload = f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def dhash64(image: Image.Image) -> int:
    """64-bit horizontal difference hash: RGB to L, BOX to 9x8, left>right, row-major bits."""
    small = image.convert("RGB").convert("L").resize((9, 8), Image.Resampling.BOX)
    gray = np.asarray(small, dtype=np.int16)
    bits = (gray[:, :-1] > gray[:, 1:]).ravel()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def flag_near_duplicates(hashes: Mapping[str, int], limit: int = HAMMING_MAX) -> list[tuple[str, str, int]]:
    """Candidate pairs for raw-image review. A candidate screen, not a patient detector."""
    names = sorted(hashes)
    return [(a, b, hamming(hashes[a], hashes[b]))
            for i, a in enumerate(names) for b in names[i + 1:]
            if hamming(hashes[a], hashes[b]) <= limit]


def components(nodes: Iterable[str], links: Iterable[tuple[str, str]]) -> list[tuple[str, ...]]:
    """Connected components, each member-sorted, ordered by minimum member."""
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return sorted((tuple(sorted(members)) for members in groups.values()), key=lambda m: m[0])


def legacy_exclusions(splits: Mapping[str, str], links: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Fitting rows sharing a component with a legacy query, mapped to that component's ID.

    Canonical test rows never move; only the rare fitting side of the link is dropped.
    """
    excluded: dict[str, str] = {}
    for members in components(splits, links):
        if any(splits[member] == "test" for member in members):
            excluded.update({member: members[0] for member in members if splits[member] != "test"})
    return excluded


# ------------------------------------------------------ groups and folds (section 4) ---
@dataclass(frozen=True)
class Group:
    """One development unit. id is the minimum member path; rank is the sorted class position."""
    id: str
    label: str
    rank: int
    members: tuple[str, ...]


def build_groups(labels: Mapping[str, str], links: Iterable[tuple[str, str]],
                 excluded: Iterable[str] = ()) -> dict[str, list[Group]]:
    """Eligible groups per rare class. links are confirmed/unresolved duplicates and verified patients.

    excluded removes every member of a fitting component linked to a legacy query. A component
    spanning disease labels stops here: the protocol resolves it by review or dated amendment.
    """
    drop = set(excluded)
    out: dict[str, list[tuple[str, ...]]] = {label: [] for label in RARE}
    for members in components(labels, links):
        if drop.intersection(members):
            continue
        found = {labels[member] for member in members}
        if len(found) != 1:
            raise ValueError(f"component spans labels {sorted(found)}: {members}")
        out[found.pop()].append(members)
    return {label: [Group(m[0], label, rank, m) for rank, m in enumerate(sorted(out[label], key=lambda x: x[0]))]
            for label in RARE}


def assign_folds(groups: Sequence[Group], class_index: int) -> dict[str, int]:
    """Permute the sorted group-ID array, then deal fold IDs 0..4 round-robin."""
    ids = np.array([g.id for g in sorted(groups, key=lambda g: g.id)])
    order = stream(SEED, class_index, FOLD_NS).permutation(ids)
    return {str(group_id): position % N_FOLDS for position, group_id in enumerate(order)}


def check_feasible(groups_by_class: Mapping[str, Sequence[Group]]) -> dict[str, int]:
    """Stop the whole selector if any class lacks the groups every fold needs.

    Round-robin leaves G - ceil(G/5) fitting groups at worst, so G >= 10 gives the 8 that K=8
    requires. Returns groups per class; there is no reduced-K fallback.
    """
    counts = {label: len(groups_by_class[label]) for label in RARE}
    short = {label: n for label, n in counts.items() if n - math.ceil(n / N_FOLDS) < MIN_FITTING_GROUPS}
    if short:
        raise ValueError(f"selector stops before scores; too few eligible groups: {short}")
    return counts


def report_k_values(groups_by_class: Mapping[str, Sequence[Group]]) -> tuple[int, ...]:
    """K=16 is reported only if every class has 16 fitting groups, for every encoder."""
    return tuple(k for k in K_REPORT if all(len(groups_by_class[c]) >= k for c in RARE))


# -------------------------------------------------- paired nested supports (section 6) ---
def draw_supports(groups: Sequence[Group], fold: int, class_index: int, draw: int, namespace: int) -> list[str]:
    """Ordered support images: the first K are the K-shot roster, so rosters nest across K.

    Group.rank (position in the class inventory) seeds the member choice, so a group's image
    does not depend on its permuted position or on which groups are in the fitting set. Every
    encoder and classifier shares these rosters. Reporting calls this with fold=0,
    namespace=REPORT_NS and every eligible group.
    """
    by_id = {g.id: g for g in groups}
    ids = np.array(sorted(by_id))
    order = stream(SEED, fold, class_index, draw, namespace).permutation(ids)
    supports = []
    for group_id in order:
        group = by_id[str(group_id)]
        pick = stream(SEED, fold, class_index, draw, group.rank, namespace + 1).integers(len(group.members))
        supports.append(group.members[int(pick)])
    return supports


# ------------------------------------------------------------- classifiers (section 6) ---
def unit(features: np.ndarray) -> np.ndarray:
    """L2-normalized float64 rows; nonfinite or zero-norm input stops the arm."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim not in (1, 2) or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("expected nonempty finite feature vectors")
    # Scaling before the norm avoids overflow/underflow without changing directions.
    scale = np.max(np.abs(values), axis=-1, keepdims=True)
    if (scale == 0).any():
        raise ValueError("zero-norm features")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def centroids(supports_by_class: Sequence[np.ndarray]) -> np.ndarray:
    """Normalize every support, average per class, normalize the centroid (equations 1-2)."""
    return unit(np.stack([unit(supports).mean(axis=0) for supports in supports_by_class]))


def predict_prototype(prototypes: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Class index of the largest cosine. argmax takes the first class on exact ties."""
    return np.argmax(unit(queries) @ unit(prototypes).T, axis=1)


@dataclass(frozen=True)
class LogisticFit:
    W: np.ndarray
    b: np.ndarray
    converged: bool
    objective: float
    max_abs_gradient: float
    message: str


def logistic_objective(theta: np.ndarray, X: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[float, np.ndarray]:
    """Mean cross-entropy + 0.005 * ||W||_F^2 (equation 3); the intercept is unpenalized."""
    n, d = X.shape
    W, b = theta[:n_classes * d].reshape(n_classes, d), theta[n_classes * d:]
    logits = X @ W.T + b
    loss = -log_softmax(logits, axis=1)[np.arange(n), y].mean() + 0.005 * np.sum(W ** 2)
    residual = softmax(logits, axis=1)
    residual[np.arange(n), y] -= 1.0
    grad_W = residual.T @ X / n + 0.01 * W
    return float(loss), np.concatenate([grad_W.ravel(), residual.mean(axis=0)])


def fit_logistic(supports: np.ndarray, y: np.ndarray, n_classes: int = len(RARE)) -> LogisticFit:
    """Fixed L-BFGS-B fit from zeros. A failed fit is reported, never retuned or redrawn."""
    X = unit(supports)
    y = np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("one integer class label is required per support")
    if not np.array_equal(np.unique(y), np.arange(n_classes)):
        raise ValueError("every classifier class must have supports")
    theta = np.zeros(n_classes * (X.shape[1] + 1), dtype=np.float64)
    result = minimize(logistic_objective, theta, args=(X, y, n_classes), jac=True, method="L-BFGS-B", bounds=None,
                      options=dict(maxiter=1000, maxfun=20000, maxls=20, maxcor=10, gtol=1e-6, ftol=0.0))
    value, gradient = logistic_objective(result.x, X, y, n_classes)
    worst = float(np.abs(gradient).max())
    finite = bool(np.isfinite(result.x).all() and np.isfinite(value) and np.isfinite(gradient).all())
    d = X.shape[1]
    return LogisticFit(result.x[:n_classes * d].reshape(n_classes, d), result.x[n_classes * d:],
                       finite and worst <= 1e-6, value, worst, str(result.message))


# ------------------------------------------------ endpoints and nomination (sections 7-8) ---
def macro_recall(correct: np.ndarray, totals: np.ndarray) -> float:
    """Mean of per-class recall; every class must have queries."""
    correct, totals = np.asarray(correct), np.asarray(totals)
    if correct.shape != totals.shape or not totals.size or not np.isfinite(correct).all() or \
            not np.isfinite(totals).all() or (correct < 0).any() or (totals <= 0).any() or (correct > totals).any():
        raise ValueError("invalid class counts")
    return float(np.mean(correct / totals))


def nomination_score(hits: np.ndarray, queries: np.ndarray) -> float:
    """Equation 4. hits is [K, draw, fold, class]; queries is [fold, class].

    Correctness is pooled over folds within a class before classes are averaged, so unequal
    fold class counts carry their real weight.
    """
    hits, queries = np.asarray(hits), np.asarray(queries)
    if hits.ndim != 4 or not hits.size or hits.shape[2:] != queries.shape or \
            not np.isfinite(hits).all() or not np.isfinite(queries).all() or \
            (hits < 0).any() or (queries <= 0).any() or (hits > queries).any():
        raise ValueError("invalid development counts")
    return float(np.mean(hits.sum(axis=2) / queries.sum(axis=0)))


def nominate(scores: Mapping[str, float], seconds: Mapping[str, float]) -> str | None:
    """Fastest complete FM within 0.02 of the best; None means no comparative winner.

    scores holds only FMs that finished every fold/K/draw. The Inception reference never enters.
    This allocates engineering effort; it is not an equivalence claim.
    """
    eligible = {name: score for name, score in scores.items() if name != REFERENCE}
    if any(name not in TIE_SLOT or not np.isfinite(score) or not 0 <= score <= 1
           for name, score in eligible.items()):
        raise ValueError("invalid FM name or nomination score")
    if {"RETFound", "BioMedCLIP"} <= eligible.keys():
        raise ValueError("RETFound and its fallback cannot both enter v1")
    if len(eligible) < 2:
        return None
    best = max(eligible.values())
    # Mean recalls are rounded float64 values. Admit representational roundoff at
    # the inclusive boundary, not a scientifically meaningful extra score gap.
    near = [name for name, score in eligible.items()
            if best - score <= NEAR_BEST or math.isclose(best - score, NEAR_BEST,
                                                       rel_tol=0., abs_tol=8 * math.ulp(1.))]
    if any(name not in seconds or not np.isfinite(seconds[name]) or seconds[name] <= 0 for name in near):
        raise ValueError("near-best candidates require finite positive warm timings")
    return min(near, key=lambda name: (seconds[name], TIE_SLOT[name]))


def false_rare_rate(true_labels: Sequence[str], predicted: Sequence[str]) -> tuple[int, int]:
    """Numerator and denominator of FRR: true NORMAL scans predicted as any rare class."""
    normals = [p for t, p in zip(true_labels, predicted, strict=True) if t == "NORMAL"]
    return sum(p in RARE for p in normals), len(normals)


# ------------------------------------------------- A2 diagnostics and intervals (A2, 13 Nov) ---
PAD_RANGE = 8  # A2: a border row/column is padding if its grey range is at most 8
NEAR_WHITE = 250
INTERVAL_DRAWS, INTERVAL_SEED = 5000, 0  # v1.0 section 7 and analysis.py (L9) convention


def border_share(gray: np.ndarray) -> float:
    """Share of area in complete edge rows/columns whose grey range is at most 8 (A2).

    gray is the decoded 8-bit grayscale original. Flat full-length rows are peeled from the top,
    then the bottom; flat full-length columns from the left, then the right (the review's
    `geom.py` rule). Descriptive only.
    """
    a = np.asarray(gray)
    if a.ndim != 2 or not a.size:
        raise ValueError("expected a nonempty two-dimensional grayscale image")
    h, w = a.shape
    top = bottom = left = right = 0
    while top < h and np.ptp(a[top]) <= PAD_RANGE:
        top += 1
    while bottom < h - top and np.ptp(a[h - 1 - bottom]) <= PAD_RANGE:
        bottom += 1
    while left < w and np.ptp(a[:, left]) <= PAD_RANGE:
        left += 1
    while right < w - left and np.ptp(a[:, w - 1 - right]) <= PAD_RANGE:
        right += 1
    return 1 - max(h - top - bottom, 0) * max(w - left - right, 0) / (h * w)


def near_white_share(gray: np.ndarray) -> float:
    """Share of pixels at 250 or above in the decoded 8-bit grayscale original (A2)."""
    values = np.asarray(gray)
    if values.ndim != 2 or not values.size:
        raise ValueError("expected a nonempty two-dimensional grayscale image")
    return float(np.mean(values >= NEAR_WHITE))


def size_predict(support_sizes: Sequence[np.ndarray], query_sizes: np.ndarray) -> np.ndarray:
    """Size-only reference (A2, N-1): class of the nearest mean support (width, height).

    support_sizes holds one [K, 2] array per class in fixed order; squared Euclidean distance;
    argmin takes the first class on exact ties. Uses no pixel and no encoder; never nominated.
    """
    means = np.stack([np.asarray(s, dtype=np.float64).reshape(-1, 2).mean(axis=0) for s in support_sizes])
    queries = np.asarray(query_sizes, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(means).all() or not np.isfinite(queries).all() or any(len(s) == 0 for s in support_sizes):
        raise ValueError("size reference needs finite sizes and supports for every class")
    return np.argmin(((queries[:, None, :] - means[None, :, :]) ** 2).sum(axis=2), axis=1)


def cluster_interval(successes: Sequence[int], clusters: Sequence[str],
                     draws: int = INTERVAL_DRAWS, seed: int = INTERVAL_SEED) -> tuple[float, float]:
    """95% percentile interval of a pooled proportion from whole-cluster resamples.

    Each draw samples G of the G clusters with replacement (one integers() call per draw, as in
    analysis.py) and pools every image of each chosen cluster. Conditional on the fitted head.
    """
    hits, groups = np.asarray(successes), np.asarray(clusters)
    if hits.ndim != 1 or not len(hits) or groups.shape != hits.shape or not np.isin(hits, (0, 1)).all():
        raise ValueError("expected aligned nonempty 0/1 successes and cluster IDs")
    _, index = np.unique(groups, return_inverse=True)
    g = int(index.max()) + 1
    totals = np.zeros((g, 2), dtype=np.int64)
    np.add.at(totals[:, 0], index, hits.astype(np.int64))
    np.add.at(totals[:, 1], index, 1)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        pooled = totals[rng.integers(0, g, g)].sum(axis=0)
        values[i] = pooled[0] / pooled[1]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    """Exact two-sided 95% binomial interval; image-level and conditional (A2, OCTID)."""
    from scipy.stats import beta
    if not 0 <= k <= n or n < 1:
        raise ValueError("invalid binomial counts")
    low = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return low, high
