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
  python analysis.py cam --n 40     CAM heatmaps from a trained head, rare test images first   (Fig. 10)
  python analysis.py all            every command above except cam

Outputs: CSVs in results/analysis/ (or --out). PNG figures as well if matplotlib is installed.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from stage4_evaluate import CLASSES, RARE, cam, fix, load_feats

VARIANTS = ["none", "basic", "gan"]


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


def qc_summary(qc):
    """One row per disease from a dict {disease: qc_<disease>.csv frame} (Stage 2 `qc` output)."""
    rows = []
    for cls, df in qc.items():
        margin = df.d_normal - df.d_rare
        rows.append(dict(disease=cls, generated=len(df), passed=int(df["pass"].sum()), kept=int(df.kept.sum()),
                         pass_rate=df["pass"].mean(), kept_margin_median=margin[df.kept].median()))
    return pd.DataFrame(rows)


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


def cmd_cam(root, out, a):
    cam(root, a.variant, a.n, "imagenet", out=out / f"cam_{a.variant}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="checkout that holds data/ and results/")
    ap.add_argument("--out", default=None, help="output folder (default <root>/results/analysis)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["compare", "knn", "qc", "roc", "all"]:
        sub.add_parser(name)
    for name in ["tsne", "all"]:
        p = sub.choices.get(name) or sub.add_parser(name)
        p.add_argument("--per-class", type=int, default=300, help="sampled points per major class and per GAN class")
        p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("cam")
    p.add_argument("--variant", default="gan", choices=VARIANTS)
    p.add_argument("--n", type=int, default=40, help="number of test images (rare first)")
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out) if a.out else root / "results/analysis"
    out.mkdir(parents=True, exist_ok=True)
    cmds = {"compare": cmd_compare, "knn": cmd_knn, "qc": cmd_qc, "roc": cmd_roc, "tsne": cmd_tsne, "cam": cmd_cam}
    for name in (["compare", "qc", "roc", "knn", "tsne"] if a.cmd == "all" else [a.cmd]):
        cmds[name](root, out, a)
        print()


if __name__ == "__main__":
    main()
