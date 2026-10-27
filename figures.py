"""
Plot saved analyses (GAN-FSL-OCT reproduction, Yoo et al. MBEC 2021).

Commands (from the project root, or pass --root to another checkout):
  python figures.py roc     one-vs-rest ROC curves and saved per-class AUCs (Fig. 9)
  python figures.py tsne    two views of the recorded joint embedding     (Fig. 7)
  python figures.py all     both figures

Reads only results/analysis/{roc_points,tsne}.csv and results/stage4/auc.csv.
Writes 300-dpi PNGs and small vector PDFs to results/figures/ (or --out).
No training, feature extraction, or embedding computation takes place here.
The real-only t-SNE panel is a subset of the joint embedding, not a separate fit.
These are general research figures; journal submission dimensions are not specified.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

CLASSES = ["NORMAL", "CNV", "DME", "DRUSEN", "CSC", "MH", "MacTel", "Stargardt", "RP"]
VARIANTS = ["none", "basic", "gan"]
ROC_STYLES = {"none": ("#0072B2", "-"), "basic": ("#D55E00", "--"), "gan": ("#009E73", "-.")}
# Paul Tol's nine-color muted scheme; shape also encodes class.
# https://sronpersonalpages.nl/~pault/data/colourschemes.pdf (issue 3.2)
COLORS = ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
          "#882255", "#44AA99", "#999933", "#AA4499"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "text.color": "black",
         "figure.facecolor": "white", "axes.facecolor": "white",
         "axes.labelcolor": "black", "axes.edgecolor": "black",
         "xtick.color": "black", "ytick.color": "black", "pdf.fonttype": 42,
         "axes.spines.top": False, "axes.spines.right": False}


# ---------- pure plotting (frames in, figures out; no file I/O) ----------

def validate_roc(roc, aucs):
    """Require all 27 complete curves and agreement with the run-of-record AUCs."""
    expected = {(v, c) for v in VARIANTS for c in CLASSES}
    if not {"variant", "cls", "fpr", "tpr"} <= set(roc.columns):
        raise ValueError("ROC data needs variant, cls, fpr, tpr columns")
    if roc[["variant", "cls"]].isna().any().any():
        raise ValueError("ROC variant and class must not be missing")
    groups = roc.groupby(["variant", "cls"], sort=False)
    if set(groups.groups) != expected:
        raise ValueError("ROC data must contain every none/basic/gan and class combination")
    if not aucs.index.is_unique:
        raise ValueError("AUC table must have unique variant rows")
    values = aucs.loc[VARIANTS, CLASSES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("AUCs must be finite fractions in [0, 1]")
    for (v, c), g in groups:
        points = g[["fpr", "tpr"]].to_numpy(dtype=float)
        if not np.isfinite(points).all() or ((points < 0) | (points > 1)).any():
            raise ValueError(f"{v}/{c}: ROC rates must be finite fractions in [0, 1]")
        if (np.diff(points, axis=0) < 0).any():
            raise ValueError(f"{v}/{c}: ROC points must be monotonic in recorded order")
        if not np.array_equal(points[0], [0, 0]) or not np.array_equal(points[-1], [1, 1]):
            raise ValueError(f"{v}/{c}: ROC must start at (0, 0) and end at (1, 1)")
        area = np.sum(np.diff(points[:, 0]) * (points[1:, 1] + points[:-1, 1]) / 2)
        if abs(area - aucs.loc[v, c]) > 1e-12:
            raise ValueError(f"{v}/{c}: curve area disagrees with saved AUC")


def plot_roc(roc, aucs):
    """Nine panels; legends use saved AUCs, with color and line style for variants."""
    validate_roc(roc, aucs)
    groups = roc.groupby(["variant", "cls"], sort=False)
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(10, 10), layout="constrained")
        axes = fig.subplots(3, 3, sharex=True, sharey=True)
        for ax, c in zip(axes.flat, CLASSES):
            for v in VARIANTS:
                g = groups.get_group((v, c))
                color, linestyle = ROC_STYLES[v]
                ax.plot(g.fpr, g.tpr, color=color, linestyle=linestyle, lw=1.5,
                        label=f"{v}: AUC {aucs.loc[v, c]:.4f}")
            ax.plot([0, 1], [0, 1], color="#666666", linestyle=":", lw=.7)
            ax.set(title=f"{c} vs rest", xlim=(0, 1), ylim=(0, 1), aspect="equal")
            ax.legend(fontsize=6.5, loc="lower right", frameon=False)
        fig.supxlabel("False positive rate (fraction)")
        fig.supylabel("True positive rate (fraction)")
    return fig


def plot_tsne(tsne):
    """Real subset and full joint embedding; shared coordinates and axis limits.

    Color/shape identify class; filled/open markers identify real/GAN source.
    Separation is descriptive and does not establish classification improvement.
    """
    if not {"cls", "source", "x", "y"} <= set(tsne.columns):
        raise ValueError("t-SNE data needs cls, source, x, y columns")
    expected = {(c, "real") for c in CLASSES} | {(c, "gan") for c in CLASSES[4:]}
    if set(zip(tsne.cls, tsne.source)) != expected:
        raise ValueError("t-SNE data must contain real points for all classes and GAN points for rare classes")
    if not np.isfinite(tsne[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("t-SNE coordinates must be finite")
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(12, 6), layout="constrained")
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
        titles = ["Real subset of joint embedding", "Same embedding: real + GAN"]
        for ax, title, use_all in zip(axes, titles, [False, True]):
            # Draw GAN first so open markers do not obscure the scarce real points.
            for source in (["gan", "real"] if use_all else ["real"]):
                for c, color, marker in zip(CLASSES, COLORS, MARKERS):
                    g = tsne[(tsne.cls == c) & (tsne.source == source)]
                    ax.scatter(g.x, g.y, s=9 if c in CLASSES[:4] else 17,
                               facecolors=color if source == "real" else "none",
                               edgecolors="#333333" if source == "real" else color,
                               marker=marker, linewidths=.5)
            ax.set(title=title, xlabel="Recorded t-SNE coordinate 1 (a.u.)",
                   ylabel="Recorded t-SNE coordinate 2 (a.u.)")
        handles = [Line2D([], [], linestyle="none", marker=marker, color=color,
                          markeredgecolor="#333333", label=c)
                   for c, color, marker in zip(CLASSES, COLORS, MARKERS)]
        handles += [Line2D([], [], linestyle="none", marker="o", color="#444444",
                           markerfacecolor=fill, label=source)
                    for source, fill in [("Real (filled)", "#444444"), ("GAN (open)", "none")]]
        fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, fontsize=8)
    return fig


# ---------- I/O ----------

def save_figure(fig, path):
    """Export an opaque 300-dpi PNG and a vector PDF with embedded TrueType fonts."""
    with mpl.rc_context(STYLE):
        for fmt in ["png", "pdf"]:
            target = path.with_suffix(f".{fmt}")
            fig.savefig(target, dpi=300, facecolor="white", transparent=False)
            print(f"  figure {target}")


def cmd_roc(root, out):
    roc = pd.read_csv(root / "results/analysis/roc_points.csv")
    aucs = pd.read_csv(root / "results/stage4/auc.csv", index_col=0)
    save_figure(plot_roc(roc, aucs), out / "roc")


def cmd_tsne(root, out):
    tsne = pd.read_csv(root / "results/analysis/tsne.csv")
    save_figure(plot_tsne(tsne), out / "tsne")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="checkout that holds results/")
    ap.add_argument("--out", default=None, help="output folder (default <root>/results/figures)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["roc", "tsne", "all"]:
        sub.add_parser(name)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    out = Path(a.out).resolve() if a.out else root / "results/figures"
    # Also protect this checkout when --root points to a different input checkout.
    for checkout in {root, Path(__file__).resolve().parent}:
        for relative in ["data", "results/stage4", "results/seed_check.csv"]:
            protected = (checkout / relative).resolve()
            if out.is_relative_to(protected):
                ap.error(f"output folder must be outside {protected}")
    out.mkdir(parents=True, exist_ok=True)
    cmds = {"roc": cmd_roc, "tsne": cmd_tsne}
    for name in (["roc", "tsne"] if a.cmd == "all" else [a.cmd]):
        cmds[name](root, out)


if __name__ == "__main__":
    main()
