"""Figures 1-4 of the v1 paper, adapted from docs/report/weekly_figures.py (15 Nov weekly report).

Reads the input lock and the locked results archive in place; plots stored values only, computes
no score. Changes from the weekly versions follow methods_results_v2.md section 8.1: the hold-out
box reads "Hold-out: 22 images (inspected before)"; panel titles say "hold-out", not "test set";
Fig. 4 reads its three destinations from the confusion cells instead of subtracting.

    D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_v1_paper/make_figures.py
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ATTEMPT = Path("D:/Few-shot/drive/fm_v1/final/final-20261113T170804Z")
LOCK = ATTEMPT / "input-lock.zip"
ARC = ATTEMPT / "evaluation" / "results-20261114T093033Z.zip"
OUT = Path(__file__).resolve().parent / "figures"
ARMS = ("MIRAGE", "DINOv2", "Inception", "RETFound")
COLOR = {"MIRAGE": "#2a78d6", "DINOv2": "#eb6834", "Inception": "#1baf7a", "RETFound": "#eda100"}
MARKER = {"MIRAGE": "o", "DINOv2": "s", "Inception": "^", "RETFound": "D"}
LABEL = {"MIRAGE": "MIRAGE", "DINOv2": "DINOv2", "Inception": "Inception-v3", "RETFound": "RETFound"}
INK, MUTED, GRID, REF = "#1f1f1e", "#5f5e58", "#e6e5e0", "#77766f"
RARE = ("CSC", "MH", "MacTel", "Stargardt", "RP")

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})


def rows(archive: zipfile.ZipFile, name: str) -> list[dict]:
    return list(csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8")))


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def episode_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    ax.set_xlim(0, 100), ax.set_ylim(0, 42), ax.axis("off")

    def box(x, y, w, h, text, fill="#f3f2ee", edge=MUTED):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.8))

    ax.text(0, 40.5, "Development (primary): 124 images, 5 classes", fontsize=8, color=INK, weight="bold")
    for i in range(5):
        held = i == 0
        ax.add_patch(FancyBboxPatch((1 + i * 3.6, 25), 3.0, 11, boxstyle="round,pad=0.2,rounding_size=0.6",
                                    fc="#cde2fb" if held else "#f3f2ee", ec=COLOR["MIRAGE"] if held else MUTED, lw=0.8))
        ax.text(2.5 + i * 3.6, 30.5, f"F{i}", ha="center", va="center", fontsize=6.8, color=INK)
    ax.text(9.6, 22.3, "5 folds per class\n(F0 = queries here)", ha="center", va="top", fontsize=6.8, color=MUTED)
    arrow(19.5, 30.5, 23.5, 30.5)
    box(24, 25, 17, 11, "Draw K per class\nfrom F1-F4\nK = 1, 2, 4, 8\n(nested), 100 draws")
    arrow(41.8, 30.5, 45.3, 30.5)
    box(46, 25, 17, 11, "Frozen encoder\n(no training):\none feature\nper image")
    arrow(63.8, 30.5, 67.3, 30.5)
    box(68, 25, 14.5, 11, "Prototype\nper class =\nnormalised mean\nof K features")
    arrow(83.3, 30.5, 86.3, 30.5)
    box(87, 25, 12.5, 11, "Query takes\nthe closest\nprototype\n(cosine)", fill="#cde2fb", edge=COLOR["MIRAGE"])
    ax.text(99, 21.6, "Repeat for F0-F4 → macro recall (mean of 5 class recalls),\naveraged over 100 draws; nominee = best mean over K",
            ha="right", va="top", fontsize=6.8, color=MUTED)
    box(1, 2, 30, 10, "Hold-out: 22 images (inspected before)\nsupports from all 124\nK = 1-16, 100 draws")
    box(35, 2, 30, 10, "Normal-scan audit: 9 prototypes\n(1,000 per common class + 124 rare)\n→ false-rare rate on 3 sources")
    box(69, 2, 30, 10, "External: OCTID CSC/MH, OCTDL MH\nwith benchmark supports\n(vs published labels)")
    ax.text(0, 14.5, "Same encoders and features, other evaluations", fontsize=8, color=INK, weight="bold")
    save(fig, "fig1_method")


def _declutter(axes) -> None:
    for ax in axes:
        texts = sorted([t for t in ax.texts], key=lambda t: t.get_position()[1])
        last = -1.0
        for t in texts:
            x, y = t.get_position()
            if y - last < 0.045:
                y = last + 0.045
                t.set_position((x, y))
            last = y


def recall_vs_k(results: zipfile.ZipFile) -> None:
    nomination = json.load(results.open("reports/nomination/nomination.json"))
    legacy = [r for r in rows(results, "reports/legacy/support_draw_summary.csv") if r["classifier"] == "prototype"]
    size = rows(results, "reports/legacy/size_only_reference.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    panels = (("development", "Development (124 images, primary)", [1, 2, 4, 8]),
              ("reporting", "Hold-out (22 images, inspected before)", [1, 2, 4, 8, 16]))
    for ax, (phase, title, ks) in zip(axes, panels):
        xs = list(range(len(ks)))
        ref = {int(r["k"]): r for r in size if r["phase"] == phase and r["k"].isdigit()}
        ax.fill_between(xs, [float(ref[k]["p05"]) for k in ks], [float(ref[k]["p95"]) for k in ks], color="#d9d8d2", lw=0)
        ax.plot(xs, [float(ref[k]["mean"]) for k in ks], color=REF, lw=1.5, ls="--")
        ax.text(xs[-1] + 0.12, float(ref[ks[-1]]["mean"]), "size only", color=MUTED, fontsize=7, va="center")
        ax.axhline(0.2, color=REF, lw=1, ls=":")
        ax.text(xs[-1] + 0.12, 0.235, "chance", color=MUTED, fontsize=7, va="center")
        for arm in ARMS:
            if phase == "development":
                ys = [nomination["arms"][arm]["k_scores"][str(k)] for k in ks]
            else:
                by_k = {int(r["k"]): float(r["mean"]) for r in legacy if r["arm"] == arm and r["k"].isdigit()}
                ys = [by_k[k] for k in ks]
            ax.plot(xs, ys, color=COLOR[arm], lw=2, marker=MARKER[arm], ms=4.5, mec="white", mew=0.8)
            ax.text(xs[-1] + 0.12, ys[-1], LABEL[arm], color=INK, fontsize=7, va="center")
        ax.set_xticks(xs, [str(k) for k in ks])
        ax.set_xlim(-0.2, len(ks) - 1 + 1.15)
        ax.set_xlabel("K labelled examples per disease")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Macro recall (5 rare classes)")
    fig.tight_layout()
    _declutter(axes)
    save(fig, "fig2_recall_vs_k")


def image_sizes(lock: zipfile.ZipFile) -> None:
    inventory = [r for r in rows(lock, "image_inventory.csv") if r["label"] in RARE]
    colors = dict(zip(RARE, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
    markers = dict(zip(RARE, ("o", "s", "^", "D", "v")))
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for label in RARE:
        for split, filled in (("train", True), ("test", False)):
            pts = [(int(r["width"]), int(r["height"])) for r in inventory if r["label"] == label and r["split"] == split]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18 if filled else 40, marker=markers[label],
                       c=colors[label] if filled else "none", edgecolors=colors[label] if not filled else "white",
                       linewidths=1.2 if not filled else 0.5, label=label if split == "train" else None, zorder=3)
    ax.set_xlabel("Image width (pixels)")
    ax.set_ylabel("Image height (pixels)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", title="filled: development\nopen: hold-out", title_fontsize=6.2)
    save(fig, "fig3_image_sizes")


def normal_routing(results: zipfile.ZipFile) -> None:
    sources = (("Kermany", "runs/{arm}/reporting/nine_way/metrics.json", "Kermany normal (all 250)"),
               ("OCTID", "runs/{arm}/external/OCTID/nine_way/metrics.json", "OCTID normal (206)"),
               ("OCTDL", "runs/{arm}/external/OCTDL/nine_way/metrics.json", "OCTDL normal (332)"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    shades = (("called normal", "#86b6ef"), ("called a common disease", "#c9c8c1"), ("called rare", "#e34948"))
    for ax, (source, path, title) in zip(axes, sources):
        for i, arm in enumerate(ARMS):
            m = json.load(results.open(path.format(arm=arm)))
            row = m["confusion"][m["classes"].index("NORMAL")]
            normal, common, rare = row[0], sum(row[1:4]), sum(row[4:9])
            n = sum(row)
            left = 0.0
            for (name, color), count in zip(shades, (normal, common, rare)):
                ax.barh(i, count / n, left=left, color=color, height=0.62, edgecolor="white", linewidth=1.2,
                        label=name if (source == "Kermany" and i == 0) else None)
                left += count / n
            ax.text(1.02, i, f"{rare}", va="center", fontsize=7, color="#b3261e" if rare else MUTED)
        ax.set_yticks(range(len(ARMS)), [LABEL[a] for a in ARMS])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.3, wspace=0.12)
    axes[0].legend(frameon=False, fontsize=6.8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.text(0.98, 0.03, "number at right = scans called rare", ha="right", fontsize=6.6, color=MUTED)
    save(fig, "fig4_normal_scans")


def main() -> None:
    with zipfile.ZipFile(LOCK) as lock, zipfile.ZipFile(ARC) as results:
        episode_schematic()
        recall_vs_k(results)
        image_sizes(lock)
        normal_routing(results)


if __name__ == "__main__":
    main()
