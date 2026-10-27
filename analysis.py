"""
Training-free analyses  (GAN-FSL-OCT reproduction, Yoo et al. MBEC 2021)

Everything here reads artifacts that already exist (Stage 2 QC records, Stage 3 heads and
test predictions, the cached Inception-v3 features, results/seed_check.csv) and trains no
model, so it runs on a laptop CPU. It never writes to results/stage4/ (the run of record).

Commands (from the project root, or pass --root to the checkout that holds the data):
  python analysis.py compare        paired none/basic/gan tests on the test set + seed check  (Table 3)
  python analysis.py knn            kNN baseline sensitivity: K and Euclidean vs cosine        (Table 3 knn row)
  python analysis.py qc             CycleGAN acceptance rates from the Stage 2 QC records      (Fig. 6)
  python analysis.py roc            one-vs-rest ROC curves per variant                         (Fig. 9)
  python analysis.py tsne           t-SNE of frozen features, real-only vs + CycleGAN pool     (Fig. 7)
  python analysis.py rare           why `none` rare recall beats the paper's: prior, domain, luck (Table 4, L7)
  python analysis.py cam --n 40     CAM heatmaps from a trained head, rare test images first   (Fig. 10)
  python analysis.py rci            alternative RCI definitions + the RCI range Table 4 allows (Table 3, ~2 min)
  python analysis.py all            every command above except cam and rci

Outputs: CSVs in results/analysis/ (or --out). PNG figures as well if matplotlib is installed.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from stage4_evaluate import CLASSES, PAPER, RARE, cam, fix, load_feats, rci

VARIANTS = ["none", "basic", "gan"]

# Paper Table 4 (per-class TPR), reordered to CLASSES (the paper lists RP before Stargardt),
# and the paper's test counts (§2.2: 250 per major class, CSC 5, MH 5, MacTel 4, Stargardt 4, RP 4).
PAPER_TPR = {
    "none":  [.876, .924, .920, .880, .2, .2, 0, .25, 0],
    "basic": [.932, .940, .888, .924, .8, .8, .25, .5, .25],
    "gan":   [.912, .948, .932, .888, 1, 1, 1, .75, .75],
    "knn":   [.944, .936, .916, .904, .6, .8, 0, 0, 0],
    "proto": [.872, .916, .832, .820, 1, 1, 1, .75, .75],
}
PAPER_N = np.array([250, 250, 250, 250, 5, 5, 4, 4, 4], float)


# ---------- pure analysis (arrays in, numbers out) ----------

def mcnemar(a_ok, b_ok):
    """Exact McNemar test on paired correctness vectors.
    Returns (n_a_only, n_b_only, two-sided p): images only a got right, only b got right."""
    from scipy.stats import binomtest

    a_only, b_only = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = a_only + b_only
    return a_only, b_only, (binomtest(a_only, n, 0.5).pvalue if n else 1.0)


def paired_bootstrap(a_ok, b_ok, n_boot=1000, seed=0):
    """95% CI of accuracy(a) - accuracy(b) in points, resampling test images jointly."""
    rng = np.random.default_rng(seed)
    d = a_ok.astype(float) - b_ok.astype(float)
    boots = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(n_boot)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def seed_summary(seeds):
    """Per-pair differences over matched training seeds (columns: variant, seed, acc, rare_correct)."""
    from scipy.stats import wilcoxon

    wide = seeds.pivot(index="seed", columns="variant", values="acc")
    rows = []
    for a, b in combinations([v for v in VARIANTS if v in wide], 2):
        a, b = b, a                                       # later variant first: basic-none, gan-none, gan-basic
        d = (wide[a] - wide[b]).dropna()
        rows.append(dict(a=a, b=b, n_seeds=len(d), mean_diff=d.mean(), sd_diff=d.std(ddof=1),
                         min_diff=d.min(), max_diff=d.max(), a_wins=int((d > 0).sum()),
                         wilcoxon_p=wilcoxon(d).pvalue if len(d) >= 5 and d.any() else np.nan))
    return pd.DataFrame(rows)


def knn_scores(X, y, Xt, yt, ks, metrics):
    """Accuracy and mean rare-class recall of kNN for every (metric, K)."""
    from sklearn.neighbors import KNeighborsClassifier

    rare = [CLASSES.index(c) for c in RARE]
    rows = []
    for metric in metrics:
        for k in ks:
            pred = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute").fit(X, y).predict(Xt)
            rec = [(pred[yt == c] == c).mean() for c in rare]
            rows.append(dict(metric=metric, k=k, acc=(pred == yt).mean() * 100,
                             rare_recall_mean=np.mean(rec), rare_correct=int(sum((pred == yt)[np.isin(yt, rare)]))))
    return pd.DataFrame(rows)


def unit_rows(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def rare_shift_curve(L, y, shifts):
    """Rare test images still correct when every rare logit is lowered by d (a stronger class prior)."""
    rare = [CLASSES.index(c) for c in RARE]
    is_rare = np.isin(y, rare)
    rows = []
    for d in shifts:
        pred = (L - np.isin(np.arange(L.shape[1]), rare) * d).argmax(1)
        rows.append(dict(shift=d, rare_correct=int((pred == y)[is_rare].sum()),
                         major_acc=(pred == y)[~is_rare].mean()))
    return pd.DataFrame(rows)


def auc(pos, neg):
    """Mann-Whitney AUC: P(score of a positive > score of a negative)."""
    from scipy.stats import mannwhitneyu
    return mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def centroid_correct(X, y, test_mask, major_centroids):
    """Rare test images right under a 9-way cosine nearest-centroid rule (rare centroids from the rest)."""
    rare = [CLASSES.index(c) for c in RARE]
    C = unit_rows(np.concatenate([major_centroids, [X[(y == k) & ~test_mask].mean(0) for k in rare]]))
    return int(((unit_rows(X[test_mask]) @ C.T).argmax(1) == y[test_mask]).sum())


def qc_summary(qc):
    """One row per disease from a dict {disease: qc_<disease>.csv frame} (Stage 2 `qc` output)."""
    rows = []
    for cls, df in qc.items():
        margin = df.d_normal - df.d_rare
        rows.append(dict(disease=cls, generated=len(df), passed=int(df["pass"].sum()), kept=int(df.kept.sum()),
                         pass_rate=df["pass"].mean(), kept_margin_median=margin[df.kept].median()))
    return pd.DataFrame(rows)


def entropy(p):
    """Shannon entropy in nats of a probability array (any shape)."""
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def mutual_info(p):
    """I(true; pred) in nats from a joint probability matrix (rows true, columns predicted)."""
    return entropy(p.sum(1)) + entropy(p.sum(0)) - entropy(p)


def rci_definitions(cm):
    """Candidate RCI definitions for one confusion matrix (rows true, columns predicted).
    The first is stage4_evaluate.rci (Sindhwani 2001 / Statnikov 2005: I / H(true));
    "I (Choi 2017 Eq. 6)" is the formula as printed by the same group, without the denominator."""
    p = cm / cm.sum()
    ht, hp, i = entropy(p.sum(1)), entropy(p.sum(0)), mutual_info(p)
    balanced = cm / cm.sum(1, keepdims=True) / len(cm)          # every true class weighted 1/K
    ovr = []
    for k in range(len(cm)):                                     # 2x2 one-vs-rest table per class
        tp, fn, fp = cm[k, k], cm[k].sum() - cm[k, k], cm[:, k].sum() - cm[k, k]
        b = np.array([[tp, fn], [fp, cm.sum() - tp - fn - fp]]) / cm.sum()
        ovr.append(mutual_info(b) / entropy(b.sum(1)))
    return {"I/H(true) [stage 4]": rci(cm), "I (Choi 2017 Eq. 6)": i, "I/H(pred)": i / hp,
            "2I/(H(true)+H(pred))": 2 * i / (ht + hp), "I/sqrt(H(true)H(pred))": i / np.sqrt(ht * hp),
            "I/min(H)": i / min(ht, hp), "I/log K": i / np.log(len(cm)),
            "balanced prior I/log K": mutual_info(balanced) / np.log(len(cm)),
            "1-H(pred|true)/H(true)": 1 - (entropy(p) - ht) / ht, "macro one-vs-rest": np.mean(ovr)}


def rci_range(n, tpr):
    """Smallest and largest I/H(true) of any confusion matrix with row totals n and recalls tpr.
    I is convex in p(pred|true) for fixed p(true), so the max sits at a vertex of the feasible
    set (each row sends all its errors to one other column); at a vertex H(pred|true) is constant,
    so the max is found by enumerating vertices for the largest H(pred). The min is a convex
    problem (SLSQP from the uniform-error start)."""
    from itertools import product
    from scipy.optimize import minimize

    K, pt, r = len(n), n / n.sum(), np.asarray(tpr, float)
    err, ht = pt * (1 - r), entropy(pt)
    h_cond = sum(pt[k] * entropy(np.array([r[k], 1 - r[k]])) for k in range(K))
    rows = [k for k in range(K) if err[k] > 0]
    opts = {k: [j for j in range(K) if j != k] for k in rows}
    head, tail = rows[:2], rows[2:]
    combos = np.array(list(product(*[opts[k] for k in tail])), int).reshape(-1, len(tail))
    best = -np.inf
    for a in product(*[opts[k] for k in head]):                 # chunk the 8^9 vertices by the first two rows
        q = np.tile(pt * r, (len(combos), 1))
        for k, j in zip(head, a):
            q[:, j] += err[k]
        ar = np.arange(len(q))
        for c, k in enumerate(tail):                              # one index per row, so fancy += is safe
            q[ar, combos[:, c]] += err[k]
        best = max(best, (-(q * np.log(np.where(q > 0, q, 1)))).sum(1).max())
    off = ~np.eye(K, dtype=bool)

    def channel(x):
        c = np.diag(r); c[off] = x
        return c

    res = minimize(lambda x: mutual_info(pt[:, None] * np.clip(channel(x), 0, None)),
                   np.repeat((1 - r) / (K - 1), K - 1), bounds=[(0, 1)] * off.sum(), method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda x, k=k: channel(x)[k].sum() - 1} for k in range(K)],
                   options=dict(maxiter=500, ftol=1e-12))
    return res.fun / ht, (best - h_cond) / ht


def with_error_pattern(cm, n, tpr):
    """I/H(true) when recalls are tpr but each row's errors are spread like the rows of cm."""
    r = np.asarray(tpr, float)
    e = cm * ~np.eye(len(cm), dtype=bool)
    e = np.where(e.sum(1, keepdims=True) > 0, e / np.maximum(e.sum(1, keepdims=True), 1), ~np.eye(len(cm), dtype=bool) / (len(cm) - 1))
    p = (n / n.sum())[:, None] * (e * (1 - r)[:, None] + np.diag(r))
    return mutual_info(p) / entropy(p.sum(1))


# ---------- I/O ----------

def load_preds(root):
    out = {}
    for v in VARIANTS:
        f = root / f"results/stage3_{v}/test_predictions.csv"
        if f.exists():
            out[v] = pd.read_csv(f).assign(path=lambda t: t.path.map(fix)).set_index("path")
    return out


def save_fig(fig_fn, path):
    """Draw with matplotlib if it is installed; the CSVs are the real output either way."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not installed: skipped {path.name})")
        return
    fig = fig_fn(plt)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {path}")


def cmd_compare(root, out, a):
    preds = load_preds(root)
    rows = []
    for x, y in [("basic", "none"), ("gan", "none"), ("gan", "basic")]:
        if x not in preds or y not in preds:
            continue
        px, py = preds[x], preds[y].loc[preds[x].index]
        for subset, mask in [("all", np.ones(len(px), bool)), ("rare", px.label.isin(RARE).to_numpy())]:
            ok_x = (px.label == px.pred).to_numpy()[mask]
            ok_y = (py.label == py.pred).to_numpy()[mask]
            x_only, y_only, p = mcnemar(ok_x, ok_y)
            lo, hi = paired_bootstrap(ok_x, ok_y)
            rows.append(dict(a=x, b=y, subset=subset, n=int(mask.sum()), acc_a=ok_x.mean() * 100,
                             acc_b=ok_y.mean() * 100, diff=(ok_x.mean() - ok_y.mean()) * 100,
                             diff_ci=f"{lo:.1f}-{hi:.1f}", a_only_correct=x_only, b_only_correct=y_only,
                             mcnemar_p=p))
    t = pd.DataFrame(rows)
    t.to_csv(out / "paired_seed42.csv", index=False)
    print("=== Paired comparison, seed 42 test set (McNemar exact, paired bootstrap 95% CI) ===")
    print(t.round(4).to_string(index=False))

    f = root / "results/seed_check.csv"
    if f.exists():
        s = seed_summary(pd.read_csv(f))
        s.to_csv(out / "paired_seeds.csv", index=False)
        print("\n=== Over training seeds (results/seed_check.csv), accuracy points ===")
        print(s.round(3).to_string(index=False))
        print("With 5 seeds the smallest possible two-sided Wilcoxon p is 0.0625.")


def cmd_knn(root, out, a):
    """Same training pool as the Stage 4 knn row: major train + real rare images (no augmentation)."""
    man = pd.read_csv(root / "data/manifest.csv")
    lab = dict(zip(man.path.map(fix), man.label))
    cache = root / "results/features"
    X, paths = load_feats(cache, "major_train")
    Xs, ps = [X], list(paths)
    for c in RARE:
        Xr, pr = load_feats(cache, f"real_{c}")
        Xs.append(Xr); ps += pr
    Xt, pt = load_feats(cache, "test")
    y = np.array([CLASSES.index(lab[fix(p)]) for p in ps])
    yt = np.array([CLASSES.index(lab[fix(p)]) for p in pt])
    t = knn_scores(np.concatenate(Xs), y, Xt, yt, ks=[1, 3, 5, 10], metrics=["euclidean", "cosine"])
    t.to_csv(out / "knn_sweep.csv", index=False)
    print("=== kNN on frozen features (Stage 4 row = euclidean, K=3; paper 91.2%) ===")
    print(t.round(3).to_string(index=False))


def image_size(path):
    """(width, height) from a PNG or baseline/progressive JPEG header, without an image library."""
    import struct
    b = Path(path).read_bytes()
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", b[16:24])
    i = 2
    while i < len(b) - 9:
        if b[i] != 0xFF:
            i += 1; continue
        if b[i + 1] in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", b[i + 5:i + 9]); return w, h
        i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]


def cmd_rare(root, out, a):
    """L7: why the `none` head gets 0.75-0.80 rare recall (paper Table 4: 0.00-0.25)."""
    man = pd.read_csv(root / "data/manifest.csv")
    man["path"] = man.path.map(fix)
    split = dict(zip(man.path, man.split))
    cache = root / "results/features"
    Xm, pm = load_feats(cache, "major_train")
    ym = np.array([CLASSES.index(c) for c in man.set_index("path").label.loc[[fix(p) for p in pm]]])
    Xr, yr, pr = [], [], []
    for c in RARE:
        X, p = load_feats(cache, f"real_{c}")
        Xr.append(X); yr += [CLASSES.index(c)] * len(X); pr += [fix(q) for q in p]
    Xr, yr = np.concatenate(Xr), np.array(yr)
    Xt, pt = load_feats(cache, "test")
    pt = [fix(p) for p in pt]
    yt = np.array([CLASSES.index(c) for c in man.set_index("path").label.loc[pt]])
    is_r = np.isin(yt, [CLASSES.index(c) for c in RARE])
    head = np.load(root / "results/stage3_none/head.npz")
    L = Xt @ head["W"] + head["b"]
    rows = [("head bias range (all 9 classes)", f"{head['b'].min():.2f} to {head['b'].max():.2f}")]

    # (1) class prior: how far must rare logits drop to reach the paper's 3/22?
    curve = rare_shift_curve(L, yt, [0, 2, 4, 6, 7, 8, 10, 12, 14, 16, 20])
    curve.to_csv(out / "rare_shift.csv", index=False)
    n_tr = man[man.split == "train"].label.value_counts()
    rows.append(("log(train NORMAL / train rare), mean over rare", f"{np.log(n_tr['NORMAL'] / n_tr[RARE]).mean():.2f}"))
    rows.append(("smallest rare-logit shift giving <= 3/22 (paper none)", int(curve[curve.rare_correct <= 3]["shift"].min())))

    # (2) dataset of origin: one direction from training means separates rare from major test images
    axis = Xr.mean(0) - Xm.mean(0)
    s = Xt @ axis
    rows.append(("domain-axis AUC, rare vs major TEST images", f"{auc(s[is_r], s[~is_r]):.4f}"))
    rows.append(("rare test images above every major test image on the axis", f"{int((s[is_r] > s[~is_r].max()).sum())}/22"))
    sim_m = (unit_rows(Xt[is_r]) @ unit_rows(Xm).T).max(1)
    sim_r = unit_rows(Xt[is_r]) @ unit_rows(Xr).T
    rows.append(("rare test images whose cosine nearest train neighbour is rare", f"{int((sim_r.max(1) > sim_m).sum())}/22"))

    # (3) hold-out luck: our 22 vs random 5/5/4/4/4 hold-outs under a training-free 9-way centroid rule
    X = np.concatenate([Xr, Xt[is_r]]); y = np.concatenate([yr, yt[is_r]])
    paths = pr + [p for p, r in zip(pt, is_r) if r]
    ours = np.array([split[p] == "test" for p in paths])
    cent = np.stack([Xm[ym == k].mean(0) for k in range(4)])
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(a.n_draws):
        m = np.zeros(len(y), bool)
        for c, n in zip(RARE, [5, 5, 4, 4, 4]):
            m[rng.choice(np.flatnonzero(y == CLASSES.index(c)), n, replace=False)] = True
        sims.append(centroid_correct(X, y, m, cent))
    rows.append(("centroid rule, our hold-out", f"{centroid_correct(X, y, ours, cent)}/22"))
    rows.append((f"centroid rule, {a.n_draws} random hold-outs: median (5-95%)",
                 f"{np.median(sims):.0f} ({np.percentile(sims, 5):.0f}-{np.percentile(sims, 95):.0f})/22"))

    # (4) acquisition fingerprint in the raw files (only where data/raw exists)
    if (root / "data/raw/rare").exists():
        wh = np.array([image_size(root / p) for p in paths], float)
        med = {k: np.median(wh[(y == k) & ~ours], 0) for k in np.unique(y)}
        guess = [min(med, key=lambda k: np.abs(w - med[k]).sum()) for w in wh[ours]]
        rows.append(("rare test class from raw (width, height) alone", f"{int((np.array(guess) == y[ours]).sum())}/22"))

    per = pd.DataFrame(dict(path=np.array(pt)[is_r], label=[CLASSES[k] for k in yt[is_r]],
                            pred=[CLASSES[k] for k in L[is_r].argmax(1)], domain_score=s[is_r]))
    Lo = L[is_r].copy(); Lo[np.arange(is_r.sum()), yt[is_r]] = -np.inf
    per["logit_margin"] = L[is_r][np.arange(is_r.sum()), yt[is_r]] - Lo.max(1)
    per.to_csv(out / "rare_test_images.csv", index=False)
    t = pd.DataFrame(rows, columns=["test", "value"])
    t.to_csv(out / "rare_recall.csv", index=False)
    print("=== L7: rare recall of the `none` head (paper Table 4: none 3/22 rare correct) ===")
    print(t.to_string(index=False))
    print("\nRare-logit shift curve:\n" + curve.round(3).to_string(index=False))


def cmd_qc(root, out, a):
    qc = {c: pd.read_csv(f) for c in RARE if (f := root / f"results/stage2/qc_{c}.csv").exists()}
    if not qc:
        print("qc: no results/stage2/qc_*.csv found"); return
    t = qc_summary(qc)
    t.to_csv(out / "qc_summary.csv", index=False)
    print("=== CycleGAN acceptance (automatic QC; the paper used two ophthalmologists) ===")
    print(t.round(3).to_string(index=False))
    print("Kept sets were chosen with the original rule (sort by d_rare), see docs/research/CONTEXT.md.")


def cmd_roc(root, out, a):
    from sklearn.metrics import roc_curve

    preds = load_preds(root)
    rows = []
    for v, t in preds.items():
        for c in CLASSES:
            fpr, tpr, _ = roc_curve(t.label == c, t[f"p_{c}"])
            rows += [dict(variant=v, cls=c, fpr=f, tpr=r) for f, r in zip(fpr, tpr)]
    roc = pd.DataFrame(rows)
    roc.to_csv(out / "roc_points.csv", index=False)
    print(f"roc: {len(roc)} points for {list(preds)} -> {out / 'roc_points.csv'}")

    def draw(plt):
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
        for ax, c in zip(axes.flat, CLASSES):
            for v, g in roc[roc.cls == c].groupby("variant", sort=False):
                ax.plot(g.fpr, g.tpr, label=v)
            ax.set_title(f"{c} vs rest"); ax.plot([0, 1], [0, 1], "k:", lw=0.5)
        axes[0, 0].legend(); fig.supxlabel("false positive rate"); fig.supylabel("true positive rate")
        return fig
    save_fig(draw, out / "roc.png")


def cmd_tsne(root, out, a):
    """Frozen features, so 'none' and 'gan' differ only in which rare images are in the pool."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    man = pd.read_csv(root / "data/manifest.csv")
    lab = dict(zip(man.path.map(fix), man.label))
    cache = root / "results/features"
    rng = np.random.default_rng(a.seed)
    parts = []
    X, p = load_feats(cache, "major_train")
    y = np.array([lab[fix(q)] for q in p])
    for c in CLASSES[:4]:
        i = rng.choice(np.flatnonzero(y == c), a.per_class, replace=False)
        parts.append((X[i], c, "real"))
    for c in RARE:
        parts.append((load_feats(cache, f"real_{c}")[0], c, "real"))
        G = load_feats(cache, f"gan_{c}")[0]
        if G is not None:
            parts.append((G[rng.choice(len(G), min(a.per_class, len(G)), replace=False)], c, "gan"))
    F = np.concatenate([f for f, _, _ in parts])
    meta = pd.DataFrame([(c, s) for f, c, s in parts for _ in range(len(f))], columns=["cls", "source"])
    Z = TSNE(n_components=2, init="pca", random_state=a.seed).fit_transform(PCA(50, random_state=a.seed).fit_transform(F))
    meta[["x", "y"]] = Z
    meta.to_csv(out / "tsne.csv", index=False)
    print(f"tsne: {len(meta)} points ({a.per_class} per major class / GAN class, all real rare) -> {out / 'tsne.csv'}")

    def draw(plt):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
        for ax, title, keep in [(axes[0], "real images only (none)", meta.source == "real"),
                                (axes[1], "+ CycleGAN images (gan)", np.ones(len(meta), bool))]:
            for c in CLASSES:
                for s, mk, sz in [("gan", ".", 4), ("real", "o", 10 if c in RARE else 3)]:
                    m = keep & (meta.cls == c) & (meta.source == s)
                    if m.any():
                        ax.scatter(meta.x[m], meta.y[m], s=sz, marker=mk, color=f"C{CLASSES.index(c)}",
                                   label=c if s == "real" else None,
                                   edgecolors="k" if (s == "real" and c in RARE) else "none", linewidths=0.4)
            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        axes[1].legend(markerscale=2, fontsize=8)
        return fig
    save_fig(draw, out / "tsne.png")


def cmd_rci(root, out, a):
    """Why our RCI (0.76-0.81) is below the paper's (0.92-0.98): docs/research/REPLICATION.md L8."""
    cms = {m: pd.read_csv(f, index_col=0).loc[CLASSES, CLASSES].to_numpy(float)
           for m in PAPER if (f := root / f"results/stage4/confusion_{m}.csv").exists()}
    t = pd.DataFrame({m: rci_definitions(cm) for m, cm in cms.items()}).T
    t.insert(0, "paper", pd.Series({m: PAPER[m]["rci"] for m in cms}))
    t.index.name = "model"
    t.to_csv(out / "rci_definitions.csv")
    print("=== RCI definitions on results/stage4/confusion_*.csv (paper Table 3 first) ===")
    print(t.round(3).to_string())
    print("mean |definition - paper|:", (t.drop(columns="paper").sub(t.paper, axis=0)).abs().mean().round(3).to_dict())

    rows = []
    for m, tpr in PAPER_TPR.items():
        lo, hi = rci_range(PAPER_N, tpr)
        rows.append(dict(model=m, acc_from_tpr=(np.array(tpr) * PAPER_N).sum() / PAPER_N.sum() * 100,
                         paper_acc=PAPER[m]["acc"], rci_min=lo, rci_max=hi,
                         rci_our_error_pattern=with_error_pattern(cms[m], PAPER_N, tpr) if m in cms else np.nan,
                         paper_rci=PAPER[m]["rci"]))
    b = pd.DataFrame(rows)
    b.to_csv(out / "rci_paper_bounds.csv", index=False)
    print("\n=== I/H(true) any confusion matrix with the paper's Table 4 recalls can reach ===")
    print(b.round(3).to_string(index=False))
    # Table 4 gives 940/1022 correct for gan, Table 3 says 92.1% (941): allow one more hit anywhere.
    r = np.array(PAPER_TPR["gan"], float)
    hi = max(rci_range(PAPER_N, np.where(np.arange(len(r)) == k, r + 1 / PAPER_N, r))[1]
             for k in range(len(r)) if r[k] < 1)
    print(f"gan with one more correct image in any class: max I/H(true) = {hi:.3f} (paper {PAPER['gan']['rci']})")


def cmd_cam(root, out, a):
    cam(root, a.variant, a.n, "imagenet", out=out / f"cam_{a.variant}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="checkout that holds data/ and results/")
    ap.add_argument("--out", default=None, help="output folder (default <root>/results/analysis)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["compare", "knn", "qc", "roc", "rci", "all"]:
        sub.add_parser(name)
    for name in ["tsne", "all"]:
        p = sub.choices.get(name) or sub.add_parser(name)
        p.add_argument("--per-class", type=int, default=300, help="sampled points per major class and per GAN class")
        p.add_argument("--seed", type=int, default=42)
    for name in ["rare", "all"]:
        p = sub.choices.get(name) or sub.add_parser(name)
        p.add_argument("--n-draws", type=int, default=2000, help="random rare hold-outs for the luck check")
    p = sub.add_parser("cam")
    p.add_argument("--variant", default="gan", choices=VARIANTS)
    p.add_argument("--n", type=int, default=40, help="number of test images (rare first)")
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out) if a.out else root / "results/analysis"
    out.mkdir(parents=True, exist_ok=True)
    cmds = {"compare": cmd_compare, "knn": cmd_knn, "qc": cmd_qc, "roc": cmd_roc, "tsne": cmd_tsne,
            "rci": cmd_rci, "rare": cmd_rare, "cam": cmd_cam}
    for name in (["compare", "qc", "roc", "knn", "tsne", "rare"] if a.cmd == "all" else [a.cmd]):
        cmds[name](root, out, a)
        print()


if __name__ == "__main__":
    main()
