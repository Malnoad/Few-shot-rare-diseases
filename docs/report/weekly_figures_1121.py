"""Figures for the 21 November 2026 weekly report (frozen-encoder study).

Reads only post hoc CSVs that already exist in results/fm_v1/posthoc/ (R2, G1, P10). No image,
feature or model output is opened and no score is computed here: the script reads numbers and
plots them. Two figures are diagrams with no data (the B1 label tiers and the Phase-1 plan). No
OCT picture is ever drawn. Writes PDF + PNG into fm_weekly_figs_1121/ next to this script.

    D:/Few-shot/.venv/Scripts/python.exe -I docs/report/weekly_figures_1121.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
POSTHOC = HERE.parents[1] / "results" / "fm_v1" / "posthoc"
OUT = HERE / "fm_weekly_figs_1121"

ARMS = ("MIRAGE", "DINOv2", "Inception", "RETFound")
# Same categorical slots as weekly_figures.py (15 Nov report), so a model keeps its colour across
# reports; every series is also direct-labelled and has its own marker.
COLOR = {"MIRAGE": "#2a78d6", "DINOv2": "#eb6834", "Inception": "#1baf7a", "RETFound": "#eda100"}
MARKER = {"MIRAGE": "o", "DINOv2": "s", "Inception": "^", "RETFound": "D"}
LABEL = {"MIRAGE": "MIRAGE", "DINOv2": "DINOv2", "Inception": "Inception-v3", "RETFound": "RETFound"}
LOCKED, CHANGED = "#2a78d6", "#eb6834"  # blue = locked, orange = changed (padded, copies removed)
INK, MUTED, GRID, REF, PALE = "#1f1f1e", "#5f5e58", "#e6e5e0", "#77766f", "#f3f2ee"
KS = (1, 2, 4, 8)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})


# ---- reading (the only I/O besides saving) -------------------------------------------------------

def read(name: str) -> list[dict]:
    with open(POSTHOC / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- pure selection of the plotted numbers -------------------------------------------------------

def r2_summary(rows: list[dict]) -> dict[tuple[str, str], tuple[float, float, float]]:
    """(arm or 'X minus Inception', k) -> (mean, p05, p95) over the 100 draws."""
    return {(r["arm"], r["k"]): (float(r["mean"]), float(r["p05"]), float(r["p95"])) for r in rows}


def size_only_development(g1_rows: list[dict]) -> dict[int, float]:
    """Locked size-only reference per K on development (the G1 CSV carries it beside each arm)."""
    return {int(r["k"]): float(r["size_only"]) for r in g1_rows
            if r["arm"] == "MIRAGE" and r["table"] == "development" and r["k"].isdigit()}


def g1_destinations(g1_rows: list[dict], arm: str, source: str) -> dict[str, tuple[int, int]]:
    """Where the external normal scans of one source go: quantity -> (locked, padded) counts."""
    table = {"OCTID": "nine-way OCTID NORMAL (n=206)", "OCTDL": "nine-way OCTDL NORMAL (n=332)"}[source]
    return {r["quantity"]: (int(float(r["locked"])), int(float(r["padded"]))) for r in g1_rows
            if r["arm"] == arm and r["table"] == table}


def p10_paired(rows: list[dict]) -> dict[tuple[str, str], tuple[float, float, float]]:
    """(phase, exclusion) -> MIRAGE minus Inception-v3, mean over K = 1-8: (mean, p05, p95)."""
    return {(r["phase"], r["exclusion"]): (float(r["mean"]), float(r["p05"]), float(r["p95"])) for r in rows
            if r["arm"] == "MIRAGE minus Inception" and r["k"] == "mean_k1_2_4_8"}


def p10_review_counts(key: list[dict], marks: list[dict]) -> Counter:
    """(kind, mark) -> number of the 55 blind-review items; kind is pair, synthetic or decoy."""
    kind = {r["item"]: r["kind"] for r in key}
    return Counter((kind[r["item"]], r["mark"]) for r in marks)


# ---- figures -------------------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, fill=PALE, edge=MUTED, size=8, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fill, ec=edge, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            weight="bold" if bold else "normal", linespacing=1.3)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=0.9))


def rare_tiers() -> None:
    """Definition B1 (18 Nov) as a 2 x 2 grid: share of labelled images against size of the example pool."""
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.set_xlim(0, 100), ax.set_ylim(0, 61), ax.axis("off")
    cells = (  # x, y, fill, edge, title, body
        (20, 28, "#cde2fb", LOCKED, "A: rare and few-shot",
         "primary test classes\nYoo's five classes:\npools 19-27; 0.55-0.77%\nof the 4,146 images"),
        (60, 28, PALE, MUTED, "C: few-shot, over 5%", "small datasets;\nnever pooled with A"),
        (20, 2, PALE, MUTED, "B: rare, pool over 30", "pool capped at 30\nOCTDL MH: 1.9% of images\n(28 of 821 patients, 3.4%)"),
        (60, 2, PALE, MUTED, "D: common", "OCTID CSC and MH:\n102 each, 17.8% of 572"),
    )
    for x, y, fill, edge, title, body in cells:
        ax.add_patch(Rectangle((x, y), 39, 25, fc=fill, ec=edge, lw=1.0))
        ax.text(x + 19.5, y + 21, title, ha="center", va="center", fontsize=7.9, weight="bold", color=INK)
        ax.text(x + 19.5, y + 9.5, body, ha="center", va="center", fontsize=7.0, color=INK, linespacing=1.25)
    ax.text(39.5, 54.2, "at most 5% of the dataset's\nlabelled images (rare)", ha="center", va="bottom", fontsize=7.4, color=MUTED)
    ax.text(79.5, 54.2, "more than 5%", ha="center", va="bottom", fontsize=7.4, color=MUTED)
    ax.text(18.5, 40.5, "example pool\nat most 30\n(few-shot)", ha="right", va="center", fontsize=7.4, color=MUTED)
    ax.text(18.5, 14.5, "example pool\nmore than 30", ha="right", va="center", fontsize=7.4, color=MUTED)
    save(fig, "fig1_rare_tiers")


def _end_labels(ax, items, x, gap):
    """Direct labels at the right end of lines, spread so they never overlap."""
    last = -1e9
    for y, text, color in sorted(items):
        y = max(y, last + gap)
        ax.text(x, y, text, color=color, fontsize=8, va="center")
        last = y


def r2_spread(r2: dict, size_only: dict[int, float]) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": (1.0, 1.05)})
    xs = list(range(len(KS)))
    labels = []
    for arm in ARMS:
        mean, lo, hi = zip(*(r2[(arm, str(k))] for k in KS))
        left.fill_between(xs, lo, hi, color=COLOR[arm], alpha=0.14, lw=0)
        left.plot(xs, mean, color=COLOR[arm], lw=2, marker=MARKER[arm], ms=5, mec="white", mew=0.8)
        labels.append((mean[-1], LABEL[arm], INK))
    ref = [size_only[k] for k in KS]
    left.plot(xs, ref, color=REF, lw=1.5, ls="--")
    labels.append((ref[-1], "size only", MUTED))
    left.axhline(0.2, color=REF, lw=1, ls=":")
    labels.append((0.235, "chance", MUTED))
    _end_labels(left, labels, xs[-1] + 0.15, 0.05)
    left.set_xticks(xs, [str(k) for k in KS])
    left.set_xlim(-0.2, len(KS) - 1 + 1.3)
    left.set_ylim(0, 1)
    left.set_xlabel("K labelled examples per disease")
    left.set_ylabel("Development macro recall")
    left.set_title("Each model: mean and 5th-95th band", fontsize=9, color=INK, loc="left")
    left.grid(axis="y", color=GRID, lw=0.6)

    positions = xs + [len(KS) + 0.4]
    ticks = [str(k) for k in KS] + ["mean\nover K"]
    offsets = {"MIRAGE": -0.18, "DINOv2": 0.0, "RETFound": 0.18}
    right.axhline(0, color=INK, lw=0.9)
    for arm, dx in offsets.items():
        stats = [r2[(f"{arm} minus Inception", key)] for key in [str(k) for k in KS] + ["mean_k1_2_4_8"]]
        for x, (mean, lo, hi) in zip(positions, stats):
            right.plot([x + dx, x + dx], [lo, hi], color=COLOR[arm], lw=2.2 if arm == "MIRAGE" else 1.6, solid_capstyle="butt")
            right.plot(x + dx, mean, marker=MARKER[arm], color=COLOR[arm], ms=5.5, mec="white", mew=0.8, ls="none",
                       label=f"{LABEL[arm]} minus Inception-v3" if x == positions[0] else None)
    right.axvline(len(KS) - 0.3, color=GRID, lw=0.8)
    right.set_xticks(positions, ticks)
    right.set_xlabel("K labelled examples per disease")
    right.set_ylabel("Paired difference (same draw)")
    right.set_title("Difference from Inception-v3: mean and 5th-95th", fontsize=9, color=INK, loc="left")
    right.grid(axis="y", color=GRID, lw=0.6)
    right.legend(frameon=False, fontsize=7.6, loc="lower left", handlelength=1.2)
    right.set_ylim(-0.42, 0.3)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_r2_spread")


def g1_padding(g1_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.15), sharey=True)
    parts = (("called_NORMAL", "called normal", LOCKED), ("called_CNV_DME_DRUSEN", "called a common disease", "#c9c8c1"),
             ("called_any_rare", "called a rare disease", CHANGED))
    bars = [(arm, i, version) for arm in ("MIRAGE", "Inception") for i, version in enumerate(("locked", "padded"))]
    for ax, (source, n) in zip(axes, (("OCTID", 206), ("OCTDL", 332))):
        for row, (arm, column, version) in enumerate(bars):
            counts = g1_destinations(g1_rows, arm, source)
            left = 0.0
            for quantity, name, color in parts:
                share = counts[quantity][column] / n
                ax.barh(row, share, left=left, color=color, height=0.66, edgecolor="white", linewidth=1.0,
                        label=name if (source == "OCTID" and row == 0) else None)
                left += share
            ax.text(1.02, row, f"{counts['called_NORMAL'][column]} normal", va="center", fontsize=7.8,
                    color=INK if version == "locked" else "#b3471a")
        ax.set_yticks(range(len(bars)), [f"{LABEL[a]}, {v}" for a, _, v in bars])
        ax.set_xlim(0, 1.3)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{source} normal scans ({n})", fontsize=9, color=INK, loc="left")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()  # the y axis is shared: invert it once, or the two inversions cancel
    fig.subplots_adjust(left=0.17, right=0.99, top=0.86, bottom=0.33, wspace=0.08)
    axes[0].legend(frameon=False, fontsize=7.8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.22))
    save(fig, "fig3_g1_padding")


def p10_audit(paired: dict, review: Counter) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 2.3), gridspec_kw={"width_ratios": (1.25, 1.0)})
    rows = (("development", "locked (no exclusion)", "Development, locked", LOCKED),
            ("development", "marked same picture or same case", "Development, copies out", CHANGED),
            ("hold-out", "locked (no exclusion)", "Hold-out, locked", LOCKED),
            ("hold-out", "marked same picture or same case", "Hold-out, 4 copies out", CHANGED))
    for y, (phase, exclusion, label, color) in enumerate(rows):
        mean, lo, hi = paired[(phase, exclusion)]
        left.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="butt")
        left.plot(mean, y, "o", color=color, ms=6.5, mec="white", mew=0.9)
        left.text(hi + 0.012, y, f"{mean:+.3f}", va="center", fontsize=7.8, color=INK)
    left.axvline(0, color=INK, lw=0.9)
    left.set_yticks(range(len(rows)), [r[2] for r in rows])
    left.invert_yaxis()
    left.set_xlim(-0.05, 0.5)
    left.set_xlabel("MIRAGE minus Inception-v3, mean over K")
    left.set_title("Mean and 5th-95th over draws", fontsize=9, color=INK, loc="left")
    left.grid(axis="x", color=GRID, lw=0.6)
    left.spines["left"].set_visible(False)
    left.tick_params(axis="y", length=0)

    marks = (("same picture", "#eb6834"), ("same case, other slice", "#f4b183"), ("unsure", "#9a9993"),
             ("different", "#d9d8d2"))
    kinds = (("pair", "35 pairs"), ("synthetic", "10 synthetic\ncopies"), ("decoy", "10 decoys"))
    for y, (kind, label) in enumerate(kinds):
        left_edge = 0
        for mark, color in marks:
            count = review[(kind, mark)]
            if count:
                right.barh(y, count, left=left_edge, color=color, height=0.62, edgecolor="white", linewidth=1.0,
                           label=mark if kind == "pair" else None)
                if count >= 5:
                    right.text(left_edge + count / 2, y, str(count), ha="center", va="center", fontsize=8,
                               color="white" if mark == "same picture" else INK)
                left_edge += count
    right.set_yticks(range(len(kinds)), [k[1] for k in kinds])
    right.invert_yaxis()
    right.set_xlim(0, 36)
    right.set_xticks([0, 10, 20, 30])
    right.set_xlabel("Number of items")
    right.set_title("My blind review of 55 items", fontsize=9, color=INK, loc="left")
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.legend(frameon=False, fontsize=7.6, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.2,
                 title="my mark", title_fontsize=7.6)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig4_p10_audit")


def phase1_plan() -> None:
    """E1-E4: which request each answers, how it is read, and the dates. A diagram, no data."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_xlim(0, 100), ax.set_ylim(-4, 66), ax.axis("off")
    for x, title in ((0, "Your request"), (25.5, "Experiment"), (67.5, "How I will read it (fixed)")):
        ax.text(x, 65, title, fontsize=8, weight="bold", color=INK, va="top")
    rows = (
        ("1. Image size\nand framing\n(most critical)",
         "E1 (Colab, all four models):\ncrop the borders, take the\ncentre square, resize to 256 x 256",
         "One verdict: MIRAGE keeps its\nlead if the 5th percentile of\nthe paired difference is > 0", True),
        ("2. Errors on OCTID\nand OCTDL",
         "E2 (laptop): where external\nnormal scans go; which scans\nthe models fail together",
         "Description only, for all\nfour models; no test, no cause", False),
        ("3. Preprocessing\nsensitivity",
         "E3 (Colab): one-step changes to\nthe MIRAGE and RETFound recipes",
         "\"kept within this grid\"\nor \"sensitive to ...\"", False),
        ("My addition:\nimage source",
         "E4 (laptop): can the four\nOCT sources be told apart?",
         "Fixed words: near-perfectly,\nstrongly, partly or hardly\nseparable", False),
    )
    for i, (request, experiment, reading, main) in enumerate(rows):
        y = 50.5 - i * 11.5
        edge, fill = (LOCKED, "#cde2fb") if main else (MUTED, PALE)
        _box(ax, 0.5, y, 20, 9.5, request, size=7.2)
        _arrow(ax, 21.5, y + 4.75, 24.6, y + 4.75)
        _box(ax, 25.5, y, 38.3, 9.5, experiment, fill=fill, edge=edge, size=7.2)
        _arrow(ax, 64.8, y + 4.75, 66.4, y + 4.75)
        _box(ax, 67.5, y, 32, 9.5, reading, fill=fill, edge=edge, size=7.2)
    ax.text(0, 9.0, "Dates (latest dates)", fontsize=8, weight="bold", color=INK)
    ax.text(100, 9.0, "filled = done by 21 Nov", ha="right", fontsize=7, color=MUTED)
    steps = (("20 Nov", "rules pushed", True), ("21 Nov", "code, notebook,\nStep 0 passed", True),
             ("21-23 Nov", "Colab Sessions 1\n(running) and 2", False), ("24-26 Nov", "E1-E4 scored\nonce", False),
             ("27 Nov", "all numbers\nfrozen", False), ("3 Dec", "delivery", False))
    xs = [7 + i * 17.2 for i in range(len(steps))]
    ax.plot([xs[0], xs[-1]], [3.2, 3.2], color=MUTED, lw=1.0, zorder=1)
    for x, (date, what, done) in zip(xs, steps):
        ax.plot(x, 3.2, "o", ms=7, color=LOCKED if done else "white", mec=LOCKED if done else MUTED, mew=1.2, zorder=2)
        ax.text(x, 5.6, date, ha="center", fontsize=7.4, weight="bold", color=INK)
        ax.text(x, 0.8, what, ha="center", va="top", fontsize=7, color=MUTED, linespacing=1.15)
    save(fig, "fig5_phase1_plan")


def main() -> None:
    g1 = read("G1_pad_to_square.csv")
    rare_tiers()
    r2_spread(r2_summary(read("R2_development_spread.csv")), size_only_development(g1))
    g1_padding(g1)
    p10_audit(p10_paired(read("P10_sensitivity.csv")),
              p10_review_counts(read("P10_review_key.csv"), read("P10_review_marks.csv")))
    phase1_plan()


if __name__ == "__main__":
    main()
