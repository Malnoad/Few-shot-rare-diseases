"""Figures 2-4 of the v1 paper, adapted from docs/report/weekly_figures.py (15 Nov weekly report).

Figure 1 (the weekly method schematic) was removed on 26 Nov before the step-2 freeze: the paper never
included it, and its "nominee = best mean over K" omitted the 0.02 speed tie and Inception-v3's ineligibility.

Reads the input lock and the locked results archive in place; plots stored values only, computes
no score. Changes from the weekly versions follow methods_results_v2.md section 8.1: the hold-out
box reads "Hold-out: 22 images (inspected before)"; panel titles say "hold-out", not "test set";
Fig. 4 reads its three destinations from the confusion cells instead of subtracting.
26 Nov (pre-freeze check): Fig. 4's Kermany panel uses the 169 patient-disjoint normals, as A2 A-5 requires
for any comparison with external false-rare rates (read from each arm's predictions.csv, gated against the
archive's false_rare_rate_patient_disjoint); Fig. 3's legend marks RP and Stargardt with footnote a.

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
    # No creation date in the PDF, so a rerun gives the same bytes and the step-2 freeze can fingerprint it.
    fig.savefig(OUT / f"{name}.pdf", dpi=200, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


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
                       linewidths=1.2 if not filled else 0.5, label=(label + ("$^{a}$" if label in ("RP", "Stargardt") else "")) if split == "train" else None, zorder=3)
    ax.set_xlabel("Image width (pixels)")
    ax.set_ylabel("Image height (pixels)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", title="filled: development\nopen: hold-out", title_fontsize=6.2)
    save(fig, "fig3_image_sizes")


COMMON = ("NORMAL", "CNV", "DME", "DRUSEN")


def kermany_pid(path: str) -> str:
    """Manifest pid of a Kermany image: the second field of its file name (A2 A-5; as fm_evaluate.kermany_pid)."""
    parts = Path(path).name.split("-")
    if len(parts) != 3 or not parts[1]:
        raise ValueError(f"not a Kermany file name: {path}")
    return parts[1]


def kermany_disjoint_row(lock: zipfile.ZipFile, results: zipfile.ZipFile, arm: str) -> list[int]:
    """NORMAL row of the nine-way head on the Kermany normals whose pid is in no common fitting row of the pinned
    roster (A2 A-5: 169 images), as counts in the order of metrics.json classes. Gated against the archive."""
    roster = rows(lock, "nine_way_roster.csv")
    fitted = {kermany_pid(r["path"]) for r in roster if r["role"] == "fitting" and r["label"] in COMMON}
    disjoint = {r["path"] for r in roster if r["role"] == "query" and r["label"] == "NORMAL"
                and kermany_pid(r["path"]) not in fitted}
    classes = json.load(results.open(f"runs/{arm}/reporting/nine_way/metrics.json"))["classes"]
    called = [r["predicted_label"] for r in rows(results, f"runs/{arm}/reporting/nine_way/predictions.csv")
              if r["query_id"] in disjoint]
    audit = json.load(results.open(f"runs/{arm}/reporting/nine_way/source_audit.json"))["patient_disjoint"]
    rare = sum(c in RARE for c in called)
    if len(called) != audit["normal_n"] or len(called) != 169 or rare != audit["false_rare_n"]:
        raise SystemExit(f"{arm}: patient-disjoint Kermany normals disagree with source_audit.json")
    return [called.count(c) for c in classes]


def normal_routing(lock: zipfile.ZipFile, results: zipfile.ZipFile) -> None:
    sources = (("Kermany", None, "Kermany normal (169)"),
               ("OCTID", "runs/{arm}/external/OCTID/nine_way/metrics.json", "OCTID normal (206)"),
               ("OCTDL", "runs/{arm}/external/OCTDL/nine_way/metrics.json", "OCTDL normal (332)"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    shades = (("called normal", "#86b6ef"), ("called a common disease", "#c9c8c1"), ("called rare", "#e34948"))
    for ax, (source, path, title) in zip(axes, sources):
        for i, arm in enumerate(ARMS):
            if path is None:
                row = kermany_disjoint_row(lock, results, arm)
            else:
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
        recall_vs_k(results)
        image_sizes(lock)
        normal_routing(lock, results)


if __name__ == "__main__":
    main()
