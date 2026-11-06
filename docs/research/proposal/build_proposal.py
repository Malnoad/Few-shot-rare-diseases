"""Build the proposal figure pack (HTML + one SVG per figure) from committed results.

Training-free. Reads only committed CSVs under results/, so every plotted number is
traceable to a file. Visual style follows the SC-SAM x STARCOP proposal template
(teal / blue / dashed amber / dashed gray boxes, booktabs tables).

    D:/Few-shot/.venv/Scripts/python.exe docs/research/proposal/build_proposal.py

Writes docs/research/proposal/{proposal.html, fig1_pipeline.svg, ... fig6_ladder.svg}.
"""
import csv
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
R = ROOT / "results"

# ------------------------------------------------------------------ data ----


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


SEEDS = rows(R / "seed_check.csv")
METRICS = {r["model"]: r for r in rows(R / "stage4/metrics.csv")}
RECALL = {r[""]: r for r in rows(R / "stage4/recall.csv")}
CLUSTER = rows(R / "analysis/cluster_seed42.csv")
QC = rows(R / "analysis/qc_summary.csv")
C5 = {r["disease"]: r for r in rows(R / "stage2/c5_v2_counts.csv")}
C0 = rows(R / "c0/v1_gate_summary.csv")
PAIRED = {(r["a"], r["b"]): r for r in rows(R / "analysis/paired_seeds.csv")}
# L7: rare test class named from raw (width, height) alone, current inputs ("16/22")
SIZE_ONLY_CURRENT = next(r["value"] for r in rows(R / "analysis/rare_recall.csv")
                         if r["test"].startswith("rare test class from raw")).split("/")[0]

VARIANTS = ["none", "basic", "gan"]
RARE = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
RARE_TEST = {"CSC": 5, "MH": 5, "MacTel": 4, "Stargardt": 4, "RP": 4}
PAPER_ACC = {m: float(METRICS[m]["paper_acc"]) for m in METRICS}


def seed_vals(variant, key):
    return [float(r[key]) for r in SEEDS if r["variant"] == variant]


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def rare_correct(model):
    return round(sum(float(RECALL[model][c]) * RARE_TEST[c] for c in RARE))


def t_halfwidth(sd, n):
    return stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)


def cluster_ci(variant, metric):
    for r in CLUSTER:
        if r["variant"] == variant and r["metric"] == metric:
            return float(r["ci_low"]), float(r["ci_high"])
    return None


ACC = {v: seed_vals(v, "acc") for v in VARIANTS}
RC = {v: seed_vals(v, "rare_correct") for v in VARIANTS}
DIFFS = {(a, b): [x - y for x, y in zip(ACC[a], ACC[b])]
         for a, b in [("gan", "basic"), ("gan", "none"), ("basic", "none")]}
SD_GB = float(PAIRED[("gan", "basic")]["sd_diff"])

# ------------------------------------------------------------ svg kit ----

C = {  # stroke, fill, text
    "teal": ("#1f7a7a", "#e7f3f2"), "blue": ("#2c5d9e", "#e9eff8"),
    "amber": ("#c58a12", "#fdf6e6"), "gray": ("#8b9099", "#ffffff"),
    "neutral": ("#6b7079", "#f6f6f4"), "rose": ("#b23a55", "#fbecef"),
}
FONT = "Helvetica, Arial, sans-serif"
INK = "#1b1b1b"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def defs():
    ms = "".join(
        f'<marker id="ah-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{s}"/></marker>'
        for k, (s, _) in C.items())
    hatch = ('<pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" '
             'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#8b9099" '
             'stroke-width="1.4"/></pattern>')
    return f"<defs>{ms}{hatch}</defs>"


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
            f'font-family="{FONT}" fill="{INK}" role="img">{defs()}{body}</svg>')


def text(x, y, s, size=9.5, anchor="middle", weight="normal", fill=INK, italic=False, rotate=None):
    st = ' font-style="italic"' if italic else ""
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{st}{tr}>{esc(s)}</text>')


def box(x, y, w, h, title, lines=(), kind="teal", bold=False):
    s, f = C[kind]
    dash = ' stroke-dasharray="5 3"' if kind in ("amber", "gray") else ""
    sw = 1.8 if bold else 1.1
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{f}" stroke="{s}" '
           f'stroke-width="{sw}"{dash}/>']
    n = 1 + len(lines)
    y0 = y + h / 2 - (n - 1) * 6.2 + 3.5
    out.append(text(x + w / 2, y0, title, 10, weight="bold"))
    for i, ln in enumerate(lines):
        out.append(text(x + w / 2, y0 + 12.6 * (i + 1), ln, 8.9, fill="#33373d"))
    return "".join(out)


def path(pts, kind="teal", dashed=False, head=True, width=1.1):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    da = ' stroke-dasharray="5 3"' if dashed else ""
    mk = f' marker-end="url(#ah-{kind})"' if head else ""
    return f'<path d="{d}" fill="none" stroke="{C[kind][0]}" stroke-width="{width}"{da}{mk}/>'


def badge(x, y, n):
    return (f'<circle cx="{x}" cy="{y}" r="7.5" fill="{C["rose"][0]}"/>'
            + text(x, y + 3.3, n, 9, weight="bold", fill="#fff"))


def lane_heads(xs, labels, y=24):
    out = []
    for x, lab in zip(xs, labels):
        num, rest = lab.split(" ", 1)
        out.append(f'<text x="{x}" y="{y}" font-size="9.5" letter-spacing="0.6">'
                   f'<tspan font-weight="bold">{num}</tspan> {esc(rest)}</text>')
    return "".join(out)


def legend(items, x, y):
    out, cx = [], x
    for kind, dashed, lab in items:
        s, f = C[kind]
        da = ' stroke-dasharray="4 2"' if dashed else ""
        out.append(f'<rect x="{cx}" y="{y - 8}" width="14" height="9" fill="{f}" stroke="{s}"{da}/>')
        out.append(text(cx + 19, y, lab, 8.8, anchor="start", fill="#444"))
        cx += 26 + len(lab) * 4.7
    return "".join(out)


def axis_x(x0, x1, y, lo, hi, ticks, fmt="{:g}", label=None):
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    out = [f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#555" stroke-width="0.8"/>']
    for t in ticks:
        out.append(f'<line x1="{sx(t)}" y1="{y}" x2="{sx(t)}" y2="{y + 4}" stroke="#555" stroke-width="0.8"/>')
        out.append(text(sx(t), y + 14, fmt.format(t), 8.5, fill="#444"))
    if label:
        out.append(text((x0 + x1) / 2, y + 28, label, 9, fill="#333"))
    return "".join(out), sx


def axis_y(x, y0, y1, lo, hi, ticks, label=None, grid_to=None):
    sy = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    out = [f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" stroke="#555" stroke-width="0.8"/>']
    for t in ticks:
        if grid_to:
            out.append(f'<line x1="{x}" y1="{sy(t)}" x2="{grid_to}" y2="{sy(t)}" stroke="#e6e6e6" stroke-width="0.7"/>')
        out.append(f'<line x1="{x - 4}" y1="{sy(t)}" x2="{x}" y2="{sy(t)}" stroke="#555" stroke-width="0.8"/>')
        out.append(text(x - 7, sy(t) + 3, f"{t:g}", 8.5, anchor="end", fill="#444"))
    if label:
        out.append(text(x - 32, (y0 + y1) / 2, label, 9, fill="#333", rotate=-90))
    return "".join(out), sy


def panel_tag(x, y, s):
    return text(x, y, s, 10.5, anchor="start", weight="bold")


# ---------------------------------------------------------- figure 1 ----


def fig1():
    c1, c2, c3, c4 = (22, 175), (222, 225), (472, 185), (682, 180)
    y1, y2, y3, y4, h = 60, 160, 260, 360, 60
    b = [lane_heads([c1[0], c2[0], c3[0], c4[0]],
                    ["01 DATA", "02 AUGMENTATION · STAGES 1–2", "03 CLASSIFIER · STAGE 3",
                     "04 EVALUATION · STAGE 4"])]
    # arrows first so boxes sit on top
    b += [
        path([(22, 190), (12, 190), (12, 46), (564, 46), (564, 60)], "teal"),
        path([(109, 60), (109, 46)], "teal", head=False),
        text(330, 41, "real images: all common classes · real rare images in none only", 8.6, fill="#2f6f6f"),
        path([(197, 84), (222, 84)], "teal"),
        path([(197, 108), (209, 108), (209, 178), (222, 178)], "teal"),
        path([(197, 202), (222, 202)], "teal"),
        path([(334, 220), (334, 260)], "teal"),
        path([(275, 320), (275, 360)], "amber", dashed=True),
        path([(393, 320), (393, 360)], "teal"),
        path([(447, 84), (472, 84)], "teal"),
        path([(447, 390), (459, 390), (459, 104), (472, 104)], "teal"),
        path([(564, 120), (564, 160)], "blue"),
        path([(564, 220), (564, 260)], "blue"),
        path([(657, 290), (669, 290), (669, 200), (682, 200)], "blue"),
        path([(772, 120), (772, 160)], "gray", dashed=True),
        text(778, 143, "scored against", 7.8, anchor="start", fill="#777"),
        path([(772, 220), (772, 260)], "gray"),
        path([(275, 420), (275, 450)], "amber", dashed=True),
    ]
    b += [
        box(c1[0], y1, c1[1], h, "Rare-disease OCT", ["CSC · MH · MacTel · Stargardt · RP",
                                                      "19–27 train · 4–5 test per class"], "teal"),
        box(c1[0], y2, c1[1], h, "Kermany OCT2017", ["NORMAL · CNV · DME · DRUSEN",
                                                     "73,958 fitting · 1,000 test"], "teal"),
        box(c2[0], y1, c2[1], h, "Stage 1 · basic augmentation",
            ["flips · shift · rotation · zoom", "brightness · elastic · 2,000 per class"], "teal"),
        box(c2[0], y2, c2[1], h, "Stage 2 · CycleGAN per disease",
            ["NORMAL → disease, 100 epochs", "4,000 generated per disease"], "teal"),
        box(c2[0], y3, c2[1], h, "Automatic QC gate",
            ["k = 3 cosine kNN: d_rare < d_normal", "keep ≤ 3,000 nearest real disease"], "teal"),
        box(222, y4, 107, h, "Top-up v2 (C5)", ["3,000 per class", "C2 · C3 only"], "amber"),
        box(340, y4, 107, h, "Accepted v1", ["832–3,000 per class", "C1 · runs of record"], "teal"),
        box(c3[0], y1, c3[1], h, "Training pool by variant",
            ["common: all fitting images", "rare: none | basic | gan"], "blue"),
        box(c3[0], y2, c3[1], h, "Inception-v3 (ImageNet)",
            ["frozen · avg-pool 2,048-d", "features cached once"], "blue"),
        box(c3[0], y3, c3[1], h, "Softmax head",
            ["9-way dense · Adam 1e-3 · batch 10", "the only trained weights"], "blue", bold=True),
        box(c4[0], y1, c4[1], h, "Test set (committed)",
            ["1,000 Kermany + 22 rare", "evaluation only"], "gray"),
        box(c4[0], y2, c4[1], h, "Metrics", ["accuracy · κ · MCC · recall",
                                            "RCI kept out of claims"], "neutral"),
        box(c4[0], y3, c4[1], h, "Uncertainty", ["patient-cluster bootstrap (655)",
                                                "paired by training seed"], "neutral"),
        box(c4[0], y4, c4[1], h, "Baselines, same features",
            [f"kNN K=3 {float(METRICS['knn']['acc']):.2f}%",
             f"class mean {float(METRICS['proto']['acc']):.2f}%"], "neutral"),
        box(22, 450, 175, 46, "Manifest (committed)", ["fixes every split · never re-sampled"], "neutral"),
        box(222, 450, 435, 46, "Pending Colab experiments C1–C3",
            ["C1: accepted v1 · C2 GAN amount and C3 common caps: v2 · outputs isolated on Drive"],
            "amber"),
        box(682, 450, 180, 46, "Identified risks ①–③", ["Table 3"], "rose"),
    ]
    b += [badge(440, 266, "1"), badge(650, 166, "1"), badge(190, 66, "2"), badge(855, 66, "2"),
          badge(650, 66, "3")]
    b.append(legend([("teal", False, "images (real and generated)"), ("blue", False, "classifier"),
                     ("gray", True, "evaluation only"), ("amber", True, "pending / C2–C3 only"),
                     ("rose", False, "risk")], 110, 524))
    return svg(880, 536, "".join(b))


# ---------------------------------------------------------- figure 2 ----

COLV = {"none": "#8b9099", "basic": "#2c5d9e", "gan": "#1f7a7a"}


def fig2():
    b = []
    # (a) accuracy
    ax, sy = axis_y(60, 240, 30, 86, 97, range(86, 98, 2), "test accuracy (%)", grid_to=300)
    b += [panel_tag(14, 16, "a"), text(180, 16, "Accuracy: seeds 1–5, seed 42, paper", 9.5), ax]
    jit = [-14, -7, 0, 7, 14]
    for i, v in enumerate(VARIANTS):
        cx = 100 + i * 80
        m, sd = mean_sd(ACC[v])
        b.append(f'<line x1="{cx - 22}" y1="{sy(m)}" x2="{cx + 22}" y2="{sy(m)}" stroke="{COLV[v]}" stroke-width="2.2"/>')
        b.append(f'<line x1="{cx}" y1="{sy(m - sd)}" x2="{cx}" y2="{sy(m + sd)}" stroke="{COLV[v]}" stroke-width="1"/>')
        for j, a in enumerate(ACC[v]):
            b.append(f'<circle cx="{cx + jit[j]}" cy="{sy(a)}" r="3.4" fill="{COLV[v]}" fill-opacity="0.75"/>')
        s42 = float(METRICS[v]["acc"])
        b.append(f'<path d="M{cx + 30},{sy(s42) - 5} l5,5 l-5,5 l-5,-5 z" fill="#fff" stroke="{INK}" stroke-width="1.2"/>')
        pa = PAPER_ACC[v]
        b.append(f'<line x1="{cx - 26}" y1="{sy(pa)}" x2="{cx + 36}" y2="{sy(pa)}" stroke="{C["rose"][0]}" stroke-width="1.2" stroke-dasharray="3 2"/>')
        b.append(text(cx, 256, v, 9.5, weight="bold", fill=COLV[v]))
    b.append(text(180, 276, "dots: seeds 1–5 · bar: mean ± SD · ◇ seed 42 · dashed rose: paper", 8.2, fill="#555"))

    # (b) paired differences
    x0, x1 = 420, 640
    axb, sx = axis_x(x0, x1, 205, -6, 9, range(-6, 10, 3), "{:+g}", "paired difference per seed (pp)")
    b += [panel_tag(330, 16, "b"), text(520, 16, "Paired seed differences, 95% t interval", 9.5), axb]
    b.append(f'<line x1="{sx(0)}" y1="30" x2="{sx(0)}" y2="205" stroke="#555" stroke-width="0.8"/>')
    b.append(f'<line x1="{sx(1)}" y1="30" x2="{sx(1)}" y2="205" stroke="{C["teal"][0]}" stroke-width="1" stroke-dasharray="4 3"/>')
    b.append(text(sx(1) + 3, 38, "C1 target +1 pp", 8, anchor="start", fill=C["teal"][0]))
    for i, (pair, ds) in enumerate(DIFFS.items()):
        cy = 70 + i * 50
        m, sd = mean_sd(ds)
        hw = t_halfwidth(sd, len(ds))
        b.append(text(x0 - 8, cy + 3, f"{pair[0]} − {pair[1]}", 9, anchor="end", weight="bold"))
        b.append(f'<line x1="{sx(m - hw)}" y1="{cy}" x2="{sx(m + hw)}" y2="{cy}" stroke="{INK}" stroke-width="1.6"/>')
        b.append(f'<rect x="{sx(m) - 3.5}" y="{cy - 3.5}" width="7" height="7" fill="{INK}"/>')
        for d in ds:
            b.append(f'<circle cx="{sx(d)}" cy="{cy + 12}" r="3" fill="{COLV[pair[0]]}" fill-opacity="0.7"/>')
        b.append(text(sx(m), cy - 8, f"{m:+.2f} [{m - hw:+.1f}, {m + hw:+.1f}]".replace("-", "−"), 8.2, fill="#333"))
    b.append(text(530, 276, "n = 5 seeds · intervals descriptive, uncorrected", 8.2, fill="#555"))

    # (c) rare images correct
    axc, sy2 = axis_y(720, 240, 30, 0, 22, [0, 5, 10, 15, 22], "rare test images correct (of 22)", grid_to=870)
    b += [panel_tag(665, 16, "c"), text(790, 16, "Rare classes (22 images)", 9.5), axc]
    for i, v in enumerate(VARIANTS):
        cx = 750 + i * 45
        for j, r in enumerate(RC[v]):
            b.append(f'<circle cx="{cx + jit[j] * 0.6}" cy="{sy2(r)}" r="3" fill="{COLV[v]}" fill-opacity="0.75"/>')
        s42 = rare_correct(v)
        b.append(f'<path d="M{cx + 16},{sy2(s42) - 5} l5,5 l-5,5 l-5,-5 z" fill="#fff" stroke="{INK}" stroke-width="1.2"/>')
        b.append(text(cx, 256, v, 9.5, weight="bold", fill=COLV[v]))
    b.append(f'<line x1="{742}" y1="{sy2(3)}" x2="{770}" y2="{sy2(3)}" stroke="{C["rose"][0]}" stroke-width="1.2" stroke-dasharray="3 2"/>')
    b.append(text(774, sy2(3) + 3, "paper none 3/22", 8, anchor="start", fill=C["rose"][0]))
    return svg(880, 286, "".join(b))


# ---------------------------------------------------------- figure 3 ----


def fig3():
    b = [panel_tag(14, 16, "a"), text(230, 16, "C0 gate: rare vs common separation in frozen features", 9.5)]
    axa, sx = axis_x(120, 420, 150, 0.80, 1.00, [0.80, 0.85, 0.90, 0.95, 1.00], "{:.2f}",
                     "AUC on the rare–common feature axis (22 rare vs 1,000 common tests)")
    b.append(axa)
    b.append(f'<line x1="{sx(0.90)}" y1="32" x2="{sx(0.90)}" y2="150" stroke="{C["rose"][0]}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    b.append(text(sx(0.90) - 4, 40, "train heads only if < 0.90", 8, anchor="end", fill=C["rose"][0]))
    for i, r in enumerate(C0):
        cy = 62 + i * 32
        b.append(text(112, cy + 3, r["arm"], 9.5, anchor="end", weight="bold"))
        rec, fix = float(r["recomputed_axis_auc"]), float(r["fixed_current_axis_auc"])
        b.append(f'<line x1="{sx(0.80)}" y1="{cy}" x2="{sx(rec)}" y2="{cy}" stroke="#ddd" stroke-width="1"/>')
        b.append(f'<circle cx="{sx(rec)}" cy="{cy}" r="4.2" fill="{C["blue"][0]}"/>')
        b.append(f'<circle cx="{sx(fix)}" cy="{cy + 9}" r="3.4" fill="#fff" stroke="{C["blue"][0]}" stroke-width="1.3"/>')
        b.append(text(sx(rec) - 8, cy + 3, f"{rec:.3f}", 8.2, anchor="end", fill="#333"))
    b.append(text(270, 200, "● recomputed axis · ○ fixed current axis · axis starts at 0.80", 8.2, fill="#555"))

    b += [panel_tag(470, 16, "b"), text(670, 16, "Shortcut cues among the 22 rare tests", 9.5)]
    ayb, sy = axis_y(540, 150, 30, 0, 22, [0, 5, 10, 15, 22], "rare test images", grid_to=860)
    b.append(ayb)
    for i, r in enumerate(C0):
        gx = 575 + i * 100
        size = r["size_correct"] or SIZE_ONLY_CURRENT  # current arm has no header column in C0
        ties = int(r["size_ties"] or 0)
        above = int(r["rare_above_all_common"])
        h1 = int(size)
        if ties == 22:
            b.append(f'<rect x="{gx}" y="{sy(22)}" width="26" height="{sy(0) - sy(22)}" fill="url(#hatch)" stroke="#8b9099"/>')
            b.append(text(gx + 13, sy(22) - 4, "ties 22", 8, fill="#555"))
        else:
            b.append(f'<rect x="{gx}" y="{sy(h1)}" width="26" height="{sy(0) - sy(h1)}" fill="#8b9099"/>')
            b.append(text(gx + 13, sy(h1) - 4, h1, 8.5))
        b.append(f'<rect x="{gx + 30}" y="{sy(above)}" width="26" height="{sy(0) - sy(above)}" fill="{C["rose"][0]}"/>')
        b.append(text(gx + 43, sy(above) - 4, above, 8.5))
        b.append(text(gx + 28, 164, r["arm"], 9.5, weight="bold"))
    b.append(legend([("gray", False, "class named from raw image size alone"),
                     ("rose", False, "above every common test on the axis")], 548, 186))
    return svg(880, 210, "".join(b))


# ---------------------------------------------------------- figure 4 ----


def fig4():
    b = [panel_tag(14, 16, "a"), text(230, 16, "First run: 4,000 generated per disease, automatic QC", 9.5)]
    axa, sx = axis_x(100, 420, 175, 0, 4000, [0, 1000, 2000, 3000, 4000], "{:,}", "images")
    b.append(axa)
    for i, r in enumerate(QC):
        cy = 36 + i * 27
        gen, passed, kept = int(r["generated"]), int(r["passed"]), int(r["kept"])
        b.append(text(92, cy + 12, r["disease"], 9.5, anchor="end", weight="bold"))
        b.append(f'<rect x="{sx(0)}" y="{cy}" width="{sx(gen) - sx(0)}" height="17" fill="#fff" stroke="#b9bdc4"/>')
        b.append(f'<rect x="{sx(0)}" y="{cy}" width="{sx(passed) - sx(0)}" height="17" fill="{C["teal"][1]}" stroke="{C["teal"][0]}"/>')
        b.append(f'<line x1="{sx(kept)}" y1="{cy - 2}" x2="{sx(kept)}" y2="{cy + 19}" stroke="{INK}" stroke-width="1.6"/>')
        b.append(text(sx(passed) + 5 if passed < 3300 else sx(kept) - 5, cy + 12,
                      f"{float(r['pass_rate']) * 100:.0f}% pass", 8.4,
                      anchor="start" if passed < 3300 else "end", fill="#2f6f6f"))
    b.append(text(260, 222, "outline: generated · teal: passed QC · black tick: kept (cap 3,000)", 8.2, fill="#555"))

    b += [panel_tag(470, 16, "b"), text(675, 16, "C5 top-up: final v2 pool of 3,000 per disease", 9.5)]
    axb, sx2 = axis_x(545, 855, 175, 0, 3000, [0, 1000, 2000, 3000], "{:,}", "images in v2 pool")
    b.append(axb)
    for i, d in enumerate(RARE):
        r = C5[d]
        cy = 36 + i * 27
        old, new = int(r["old_kept"]), int(r["new_accepted"])
        b.append(text(537, cy + 12, d, 9.5, anchor="end", weight="bold"))
        b.append(f'<rect x="{sx2(0)}" y="{cy}" width="{sx2(old) - sx2(0)}" height="17" fill="{C["teal"][1]}" stroke="{C["teal"][0]}"/>')
        if not new:
            b.append(text(sx2(old) - 5, cy + 12, "v1 already 3,000", 8.2, anchor="end", fill="#2f6f6f"))
        if new:
            b.append(f'<rect x="{sx2(old)}" y="{cy}" width="{sx2(old + new) - sx2(old)}" height="17" '
                     f'fill="{C["amber"][1]}" stroke="{C["amber"][0]}" stroke-dasharray="3 2"/>')
            gen = int(r["candidates_generated"])
            b.append(text(sx2(old) + 4, cy + 12, f"+{new:,} from {gen:,}", 8.2, anchor="start", fill="#7a5608"))
    b.append(legend([("teal", False, "v1 kept"), ("amber", True, "added by C5 from N new candidates")], 560, 222))
    return svg(880, 232, "".join(b))


# ---------------------------------------------------------- figure 5 ----


def fig5():
    steps = [
        ("① First stage", ["seeds 1–20 × none/basic/gan", "fixed250 final (primary)", "+ es20 (sensitivity)",
                           "120 heads"], "blue"),
        ("② Lock SD once", ["sample SD of the 20", "fixed250 gan − basic", "differences (ddof 1)",
                            "no outcome peeking"], "teal"),
        ("③ Target N", ["smallest n with paired-t", "power ≥ .80 at 1 pp, α .05", "N = max(63, n)",
                        "cap 243 pairs"], "teal"),
        ("④ Top-up", ["save topup.json first", "fixed250 seeds 21..N", "no early stopping",
                      "failed head: same seed"], "amber"),
        ("⑤ Read once", ["paired-t 95% CI [L, U]", "of gan − basic", "Wilcoxon: sensitivity",
                         "five-zone rule below"], "blue"),
    ]
    b = [lane_heads([20], ["C1 LOCKED 27 OCT 2026 · NO C1 OUTCOMES EXIST"], 18)]
    for i, (t, ls, k) in enumerate(steps):
        x = 20 + i * 171
        b.append(box(x, 30, 155, 80, t, ls, k))
        if i < 4:
            b.append(path([(x + 155, 70), (x + 171, 70)], "blue"))
    # precision strip
    b.append(text(20, 136, "Expected 95% half-width of gan − basic if the SD stays at the 5-seed value "
                  f"({SD_GB:.2f} pp):", 9, anchor="start"))
    for i, n in enumerate([5, 20, 63]):
        hw = t_halfwidth(SD_GB, n)
        x = 20 + i * 190
        b.append(f'<rect x="{x}" y="144" width="{hw * 25:.1f}" height="11" fill="{C["teal"][1]}" stroke="{C["teal"][0]}"/>')
        b.append(text(x + hw * 25 + 5, 153, f"n = {n}: ±{hw:.2f} pp", 8.6, anchor="start"))
    # five-zone rule
    axz, sx = axis_x(70, 520, 310, -3, 3, range(-3, 4), "{:+g}", "gan − basic, mean test accuracy (pp)")
    b.append(axz)
    b.append(f'<line x1="{sx(0)}" y1="160" x2="{sx(0)}" y2="310" stroke="#555" stroke-width="0.9"/>')
    b.append(f'<line x1="{sx(1)}" y1="160" x2="{sx(1)}" y2="310" stroke="{C["teal"][0]}" stroke-dasharray="4 3"/>')
    b.append(text(sx(1) + 3, 168, "1 pp", 8.4, anchor="start", fill=C["teal"][0]))
    zones = [((1.2, 2.4), "L ≥ 1: supports a gain of at least 1 pp", C["teal"][0]),
             ((0.3, 1.7), "0 < L < 1: positive gain; 1 pp not established", C["blue"][0]),
             ((-0.4, 0.8), "U < 1 (L ≤ 0): excludes a 1 pp gain; no equivalence claim", "#555"),
             ((-1.9, -0.5), "U < 0: interval favours basic", C["rose"][0]),
             ((-0.6, 1.6), "otherwise: inconclusive against 0 and 1 pp", "#8b9099")]
    for i, ((lo, hi), lab, col) in enumerate(zones):
        cy = 182 + i * 25
        b.append(f'<line x1="{sx(lo)}" y1="{cy}" x2="{sx(hi)}" y2="{cy}" stroke="{col}" stroke-width="2.4"/>')
        for v in (lo, hi):
            b.append(f'<line x1="{sx(v)}" y1="{cy - 5}" x2="{sx(v)}" y2="{cy + 5}" stroke="{col}" stroke-width="1.4"/>')
        b.append(text(540, cy + 3, lab, 9, anchor="start", fill=col, weight="bold" if i == 0 else "normal"))
    b.append(text(295, 352, "Intervals are illustrative shapes for each rule, not results.", 8.4, fill="#555",
                  italic=True))
    return svg(880, 360, "".join(b))


# ---------------------------------------------------------- figure 6 ----


def fig6():
    arms = [
        ("A0 · none", ["19–27 real rare images"], "teal", False),
        ("A1 · basic-2000 = C2 N=0", ["2,000 basic · ready"], "teal", False),
        ("A2 · basic-5000 (proposed)", ["5,000 basic · count-matched to A3"], "amber", False),
        ("A3 · gan-v2 = C2 N=3,000", ["2,000 basic + 3,000 accepted v2"], "teal", True),
        ("A4 · gan-unfiltered (proposed)", ["2,000 basic + 3,000 random generated"], "amber", False),
    ]
    contrasts = [
        ("A1 − A0: augmentation", "5 seeds so far: +2.25 ± 3.53 pp", "neutral", False),
        ("A2 − A1: image count", "3,000 more basic images", "neutral", False),
        ("A3 − A2: GAN content, equal count", "the paper's claim, isolated", "blue", True),
        ("A3 − A4: QC selection", "tests risk ① (shared feature space)", "rose", False),
    ]
    b = [lane_heads([40, 330, 640], ["01 ARM · RARE POOL PER CLASS", "02 SINGLE-CHANGE CONTRAST",
                                      "03 READOUT (EVERY ARM, PAIRED BY SEED)"], 18)]
    ymid = lambda k: 55 + 70 * k
    # C2 dose bracket (left), from A1 to A3
    b.append(path([(36, ymid(1)), (24, ymid(1)), (24, ymid(3)), (36, ymid(3))], "amber", dashed=True, head=False))
    b.append(text(15, (ymid(1) + ymid(3)) / 2, "C2 dose: N = 0 → 3,000 (ready)", 8.4, fill="#7a5608", rotate=-90))
    for k, (t, ls, kind, bold) in enumerate(arms):
        b.append(box(40, ymid(k) - 25, 250, 50, t, ls, kind, bold))
    for k, (t, sub, kind, bold) in enumerate(contrasts):
        a, c = (k, k + 1)
        xo = 302 if k % 2 == 0 else 312
        cy = (ymid(a) + ymid(c)) / 2
        b.append(path([(290, ymid(a)), (xo, ymid(a)), (xo, ymid(c)), (290, ymid(c))], "gray", head=False))
        b.append(path([(xo, cy), (330, cy)], "gray"))
        b.append(box(330, cy - 19, 270, 38, t, [sub], kind, bold))
    reads = [
        ("Primary: overall test accuracy", ["1,022 tests · paired by seed", "same five-zone rule as C1"], "blue", True),
        ("Rare-conditional accuracy (proposed)", ["argmax over the 5 rare classes", "on the 22 rare tests · training-free",
                                                  "removes the rare-vs-common cue ②"], "teal", False),
        ("Controls", ["common macro recall", "rare correct count (of 22)", "MCC"], "neutral", False),
    ]
    for i, (t, ls, kind, bold) in enumerate(reads):
        b.append(box(640, 32 + i * 100, 220, 84, t, ls, kind, bold))
    for k in range(4):
        cy = (ymid(k) + ymid(k + 1)) / 2
        b.append(path([(600, cy), (612, cy)], "blue", head=False))
    b.append(path([(612, 90), (612, 300)], "blue", head=False))
    b.append(path([(612, 195), (626, 195)], "blue", head=False))
    b.append(path([(626, 74), (626, 274)], "blue", head=False))
    for y in (74, 174, 274):
        b.append(path([(626, y), (640, y)], "blue"))
    b.append(legend([("teal", False, "existing or ready arm"), ("amber", True, "proposed arm"),
                     ("blue", False, "primary contrast / readout"), ("rose", False, "tests an identified risk")],
                    60, 384))
    return svg(880, 396, "".join(b))


# -------------------------------------------------------------- tables ----


def table(caption, head, body, num_cols=()):
    th = "".join(f'<th class="{"n" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"n" if i in num_cols else ""}">{c}</td>'
                                  for i, c in enumerate(r)) + "</tr>" for r in body)
    return f'<figure class="tbl"><figcaption>{caption}</figcaption><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></figure>'


def results_rows():
    out = []
    names = {"none": "<code>none</code>, real rare only", "basic": "<code>basic</code>, + 2,000 basic",
             "gan": "<code>gan</code>, + accepted v1 GAN", "knn": "kNN K=3, frozen features",
             "proto": "class mean, frozen features"}
    for m in ["none", "basic", "gan", "knn", "proto"]:
        acc = float(METRICS[m]["acc"])
        ci = cluster_ci(m, "acc")
        ci_s = f"{ci[0]:.2f}–{ci[1]:.2f}" if ci else METRICS[m]["acc_ci"] + " *"
        five = "–"
        if m in ACC:
            mu, sd = mean_sd(ACC[m])
            five = f"{mu:.2f} ± {sd:.2f}"
        out.append([names[m], f"{acc:.2f}", ci_s, f"{float(METRICS[m]['mcc']):.3f}",
                    f"{rare_correct(m)}/22", five, f"{PAPER_ACC[m]:.1f}"])
    return out


# ---------------------------------------------------------------- page ----

CSS = """
.paper{background:#fff;color:#1b1b1b;border-radius:var(--radius,8px);padding:28px 32px;
  font-family:'Times New Roman',Times,'Nimbus Roman',serif;font-size:14.5px;line-height:1.45;max-width:960px;margin:0 auto}
.paper .run{display:flex;justify-content:space-between;font-family:Helvetica,Arial,sans-serif;font-size:11px;
  color:#555;border-bottom:1px solid #ddd;padding-bottom:6px;margin-bottom:18px}
.paper .run b{color:#1f7a7a}
.paper h1{font-family:Helvetica,Arial,sans-serif;font-size:30px;margin:4px 0 2px;letter-spacing:-.3px}
.paper .sub{font-family:Helvetica,Arial,sans-serif;color:#1f7a7a;font-size:18px;line-height:1.3;margin:0 0 10px}
.paper .meta{font-family:Helvetica,Arial,sans-serif;font-size:11.5px;color:#444;margin-bottom:16px}
.paper h2{font-family:Helvetica,Arial,sans-serif;font-size:17px;margin:26px 0 8px}
.paper h3{font-family:Helvetica,Arial,sans-serif;font-size:13.5px;margin:16px 0 6px}
.paper p{margin:6px 0;text-align:justify}
.paper code{font-family:Consolas,monospace;font-size:12.5px}
.paper figure{margin:18px 0}
.paper figure.fig svg{display:block}
.paper figure.fig figcaption{font-size:13.5px;margin-top:6px}
.paper figure.fig figcaption b{font-weight:bold}
.paper figure.tbl figcaption{text-align:center;font-size:13.5px;margin-bottom:6px}
.paper table{width:100%;border-collapse:collapse;font-size:12.8px;border-top:1.6px solid #222;border-bottom:1.6px solid #222}
.paper thead th{border-bottom:1px solid #222;text-align:left;font-weight:normal;padding:5px 6px}
.paper td{padding:4px 6px;vertical-align:top}
.paper .n{text-align:right;white-space:nowrap}
.paper .call{background:#eaf4f4;border:1px solid #b8d6d6;border-radius:5px;padding:10px 14px;margin:14px 0;
  font-family:Helvetica,Arial,sans-serif;font-size:12.5px;line-height:1.45}
.paper .call p{text-align:left}
.paper .foot{display:flex;justify-content:space-between;font-family:Helvetica,Arial,sans-serif;font-size:11px;
  color:#555;border-top:1px solid #ddd;padding-top:6px;margin-top:24px}
.paper ol.src{font-size:12.5px;padding-left:22px}
"""


def page():
    gb = DIFFS[("gan", "basic")]
    m_gb, sd_gb = mean_sd(gb)
    sgn = lambda v: f"{v:+.2f}".replace("-", "−")
    hw5 = t_halfwidth(sd_gb, 5)
    g42, b42 = float(METRICS["gan"]["acc"]), float(METRICS["basic"]["acc"])
    gen_fit = {d: int(r["train"]) for d, r in
               ((r[""], r) for r in rows(R / "stage3_gan/data_counts.csv")) if d in RARE}
    basic_fit = {d: int(r["train"]) for d, r in
                 ((r[""], r) for r in rows(R / "stage3_basic/data_counts.csv")) if d in RARE}
    fig = lambda n, s, cap: f'<figure class="fig">{s}<figcaption><b>Figure {n}:</b> {cap}</figcaption></figure>'

    facts = table("<b>Table 1:</b> Facts the figures depend on (committed files and the paper).",
                  ["Item", "Value and consequence"], [
        ["Task", "9 classes: NORMAL, CNV, DME, DRUSEN (Kermany OCT2017) + CSC, MH, MacTel, Stargardt, RP "
                 "(rare-disease OCT, Mendeley btv6yrdbmv)."],
        ["Real rare training", "27 / 26 / 25 / 19 / 27 images (124); the paper used 97. Rare val comes only "
                               "from augmented images, so <code>none</code> has no rare validation."],
        ["Test set", "1,022 images: 250 per common class, 5/5/4/4/4 rare. 566 of 633 Kermany test patients "
                     "also appear in training; clustering cannot remove that leakage."],
        ["Classifier", "ImageNet Inception-v3, frozen, avg-pool 2,048-d, cached; only the 9-way softmax is trained."],
        ["GAN QC", "No clinician. A generated image passes if its 3-NN cosine distance to real disease beats "
                   "the distance to NORMAL, in the <i>same</i> ImageNet Inception-v3 avg-pool space "
                   "(<code>stage2_cyclegan.py:355</code>, <code>stage3_classifier.py:107</code>)."],
        ["Source", "Rare images are RGBA PNG crops; Kermany images are grayscale JPEG. Raw size alone names the "
                   "rare test class for 16/22."],
    ])
    res = table("<b>Table 2:</b> Results of record. Seed 42 with patient-cluster 95% CI (5,000 draws), "
                "and seeds 1–5. * legacy image-level CI.",
                ["Model", "Acc. %", "95% CI", "MCC", "Rare correct", "Seeds 1–5 acc. %", "Paper acc. %"],
                results_rows(), num_cols=(1, 2, 3, 4, 5, 6))
    traps = table("<b>Table 3:</b> Five traps that would make a GAN gain look larger than it is, and the response.",
                  ["Trap", "Evidence", "Response"], [
        ["① QC shares the classifier's feature space",
         "Accepted images are, by construction, the generated images nearest real disease in the frozen "
         "features the head reads.",
         "A4 gan-unfiltered arm (Fig. 6); report QC as feature-space selection, not image validity."],
        ["② Source reads as disease",
         "C0 gate: AUC 0.995 → 0.991 after format and framing normalisation (gate &lt; 0.90 failed).",
         "Rare-conditional readout; every rare-class number labelled source-confounded."],
        ["③ More images, not GAN images",
         f"gan rare fitting rows {min(gen_fit.values()):,}–{max(gen_fit.values()):,} per class vs "
         f"{basic_fit['CSC']:,} for basic.",
         "A2 basic-5000 against C2's N = 3,000 cell: equal 5,000 per class."],
        ["One seed decides",
         f"Seed 42: gan − basic {sgn(g42 - b42)} pp; seeds 1–5: {sgn(m_gb)} ± {sd_gb:.2f} pp, gan ahead on 3/5.",
         "C1: ≥ 63 paired seeds, locked stopping rule (Fig. 5)."],
        ["Test patients seen in training",
         "566 / 633 Kermany test patients overlap the training pool.",
         "State in every table; no claim of accuracy on new patients."],
    ])
    arms = table("<b>Table 4:</b> Arms. Cells name experiments, not outcomes.",
                 ["Arm", "Rare pool per class", "Protocol / seeds", "Status", "Question"], [
        ["Run of record", "none / basic / gan v1", "es (patience 20, best) · seed 42", "done", "reproduce Table 3"],
        ["Seed check", "same", "same · seeds 1–5", "done", "is the seed-42 order robust? (no)"],
        ["C0", "format / framing arms", "feature gate before heads", "stopped at gate (D20=A)", "does normalisation remove the source cue? (no)"],
        ["C5", "v2: 3,000 accepted", "–", "done", "full pools for C2/C3"],
        ["<b>C1</b>", "none / basic / gan v1", "fixed250 final · 1–20 + top-up to N ≥ 63", "ready, no outcomes", "H1: gan − basic ≥ 1 pp"],
        ["C2", "2,000 basic + N v2", "fixed250 · seeds 1–20", "ready", "GAN dose, N ∈ {0, 500, 1k, 2k, 3k}"],
        ["C3", "2,000 basic + 3,000 v2", "fixed250 · seeds 1–20", "ready", "common caps full / 10k / 5k"],
        ["A2 basic-5000", "5,000 basic", "as C2 · seeds 1–20", "proposed", "count vs GAN content"],
        ["A4 gan-unfiltered", "2,000 basic + 3,000 random generated", "as C2 · seeds 1–20", "proposed", "does QC selection carry the effect?"],
        ["Rare-conditional", "–", "all existing heads", "proposed, training-free", "rare accuracy without the rare-vs-common cue"],
    ])
    matrix = table("<b>Table 5:</b> Count-matched matrix inside C2's protocol (fixed250, v2, seeds 1–20). "
                   "Cells identify experiments, not scores.",
                   ["Extra rare images per class", "Basic only", "Basic + GAN"], [
        ["0 (2,000 total)", "A1 (C2 N = 0)", "–"],
        ["3,000 (5,000 total)", "A2 basic-5000 (proposed)", "A3 (C2 N = 3,000)"],
    ])

    body = f"""
<div class="paper">
<div class="run"><span><b>Few-shot OCT × CycleGAN</b> / Figure pack and proposal</span><span>6 November 2026</span></div>
<h1>Few-shot OCT × CycleGAN</h1>
<p class="sub">Figures, an attribution-controlled protocol and a count-matched arm design<br>for the Yoo et al. 2021 reproduction</p>
<p class="meta">Research proposal • Working figures • Version 1.0 • 6 November 2026</p>
<p><b>Abstract.</b> The reproduction of Yoo et al. (2021) trains a frozen Inception-v3 with a softmax head on
nine OCT classes, five of them rare. At seed 42 the CycleGAN variant reaches {g42:.2f}% against the paper's 92.1%,
but over five seeds it is not separated from basic augmentation ({sgn(m_gb)} ± {sd_gb:.2f} pp). Three findings
limit what any later GAN gain can mean: the GAN quality filter selects images in the classifier's own feature
space; rare and common images stay separable in that space after format and framing normalisation (AUC 0.991);
and the GAN arm adds rare images as well as GAN content. This pack draws the pipeline as built, the results of
record, the evidence behind those three limits, the locked C1 decision protocol, and a proposed ladder of arms in
which each step changes one thing. The only measured numbers are the seed-42 run, seeds 1–5, the C0 gate and the
C5 counts; no C1, C2 or C3 outcome exists.</p>

<h2>1&nbsp;&nbsp;Pipeline as reproduced</h2>
<p>Figure 1 separates images (teal) from the classifier (blue). Images reach the head only as frozen
features, so the head is the only trained part. Test labels connect only to evaluation.</p>
{fig(1, fig1(), "Reproduced pipeline. Real rare images feed basic augmentation, the per-disease CycleGAN and the QC "
     "reference; the training pool takes basic and accepted GAN images in <code>basic</code>/<code>gan</code>, real rare "
     "images only in <code>none</code>. Rose badges mark identified risks: ① QC and the head share one feature space; "
     "② rare images come from another source than Kermany; ③ the GAN arm also adds images.")}
{facts}

<h2>2&nbsp;&nbsp;Results of record</h2>
<p>The seed-42 ordering (gan &gt; none &gt; basic) does not survive five seeds. <code>basic</code> at seed 42 is its
lowest seed. Both augmentations get more rare test images right than <code>none</code> (Fig. 2c), but
<code>none</code> already gets far more than the paper's 3/22.</p>
{fig(2, fig2(), f"Results from <code>results/seed_check.csv</code> and <code>results/stage4/</code>. "
     f"(a) Test accuracy per training seed. (b) Paired differences per seed with 95% t intervals; the gan − basic "
     f"interval is ±{hw5:.2f} pp wide, so five seeds cannot test a 1 pp gain. (c) Rare test images correct.")}
{res}

<h2>3&nbsp;&nbsp;Why rare-class numbers need care</h2>
<p>C0 asked whether normalising file format and framing removes the separation between rare and common images.
It does not (Fig. 3a): the gate required AUC &lt; 0.90 before training heads, and framing still gives 0.991.
Framing removes the header-size cue (Fig. 3b) but not the separation, so disease and source stay confounded.</p>
{fig(3, fig3(), "C0 evidence (<code>results/c0/v1_gate_summary.csv</code>; size-only value for <code>current</code> "
     "from <code>results/analysis/rare_recall.csv</code>). (a) AUC of a rare-vs-common direction in frozen features. "
     "(b) Rare test images named from raw size, and rare tests ranked above every common test on the axis.")}
<p>The automatic QC kept far fewer MacTel and RP images than the paper's 3,000 (Fig. 4a). C5 filled the v2 pool to
3,000 for C2/C3, at the cost of 12,508 new RP candidates for 2,168 kept (Fig. 4b). C1 keeps v1.</p>
{fig(4, fig4(), "GAN acceptance (<code>results/analysis/qc_summary.csv</code>, <code>results/stage2/c5_v2_counts.csv</code>). "
     "Pass rates measure distance in Inception-v3 features, not clinical realism (risk ①).")}
{traps}

<h2>4&nbsp;&nbsp;Proposed protocol: the C1 decision</h2>
<p>C1 was locked on 27 October 2026, before any outcome. Figure 5 draws its stopping rule and the five-zone reading
of the final interval. The precision strip shows why 20 pairs are not enough: at the five-seed SD, their 95% interval
is still more than 1 pp wide on each side.</p>
<div class="call"><p><b>Primary hypothesis (C1).</b> Over paired classifier seeds (fixed 250 epochs, final head), the
accepted v1 GAN pool raises mean test accuracy over basic augmentation by at least 1 pp. Failure means the paper's
GAN-over-basic claim is not reproduced at 1 pp; it does not show equivalence.</p></div>
{fig(5, fig5(), "C1 schedule (<code>docs/research/C1_EXPERIMENT_PLAN.md</code>). The SD is estimated once from the "
     "first 20 fixed250 pairs; nothing after that changes N. Half-widths assume the 5-seed SD, which C1 re-estimates.")}

<h2>5&nbsp;&nbsp;Proposed architecture: a single-change ladder of arms</h2>
<p>C1 can show a GAN gain but not why. Its <code>gan</code> arm differs from <code>basic</code> in two ways at once:
GAN content and up to 2,700 more rare fitting images per class. The ladder in Figure 6 changes one thing per step.
Two arms are new; the rest already exist in C2. The rare-conditional readout costs no training: it restricts each
head's prediction to the five rare classes on the 22 rare tests, so a rare-vs-common source cue cannot help.
Source cues between the five rare classes can still help.</p>
{fig(6, fig6(), "Proposed attribution ladder. A1 and A3 are C2 cells; A2 and A4 are proposed and need new inputs: "
     "3,000 more basic images per class written outside <code>data/aug_basic</code>, and 3,000 generated images per class "
     "drawn without QC. Both feasibility points are unverified (Section 6).")}
<div class="call"><p><b>Proposed secondary hypothesis (A3 − A2).</b> At an equal 5,000 rare images per class,
2,000 basic + 3,000 accepted GAN images beat 5,000 basic images on mean test accuracy over seeds 1–20. Failure
narrows any GAN gain to image count.</p></div>
{arms}
{matrix}
<p>A1 → A3 is C2 as designed: it changes GAN content and image count together. A2 − A1 isolates count; A3 − A2
isolates GAN content at equal count, which is the paper's claim. A3 − A4 asks whether the gain depends on the QC
choosing images near real disease in the head's own feature space.</p>

<h2>6&nbsp;&nbsp;Decision rules and open feasibility</h2>
<p><b>After C1.</b> If L ≥ 1 pp, run A2 before claiming GAN content. If U &lt; 1 pp, report that the GAN-over-basic
claim is not reproduced at 1 pp, and treat C2 as a dose description. <b>After C2 + A2.</b> If accuracy rises with N
but A2 matches A3, the gain is from image count. <b>After A4.</b> If A3 beats A4, the gain depends on feature-space
selection, and the QC result must not be read as image quality. <b>Always.</b> Rare-class results stay
source-confounded, and the test set shares patients with training; no outcome supports a clinical claim.</p>
<p><b>Feasibility (unverified).</b> <code>stage1_basic_aug.py</code> has <code>--n</code> but no output-folder flag;
A2 needs one so the committed 2,000 images stay fixed, and whether a 5,000 draw keeps the current 2,000 as a
subset is unchecked. A4 needs the rejected v1/v2 candidates, and their presence on Drive is unchecked. A2 and A4
would add 40 heads to C2's 100; at the unmeasured C2 forecast of about 13.5 min per head, about 9 h.</p>

<h3>Sources</h3>
<ol class="src">
<li>Yoo, Choi, Kim. Feasibility study to improve deep learning in OCT diagnosis of rare retinal diseases with few-shot
classification. <i>MBEC</i> 59:401–415, 2021. doi:10.1007/s11517-021-02321-1.</li>
<li>Committed results: <code>results/seed_check.csv</code>, <code>results/stage4/metrics.csv</code>, <code>recall.csv</code>,
<code>results/analysis/cluster_seed42.csv</code>, <code>paired_seeds.csv</code>, <code>qc_summary.csv</code>,
<code>rare_recall.csv</code>, <code>results/c0/v1_gate_summary.csv</code>, <code>results/stage2/c5_v2_counts.csv</code>,
<code>results/stage3_*/data_counts.csv</code>.</li>
<li>Designs: <code>docs/research/C1_EXPERIMENT_PLAN.md</code>, <code>C2_C3_DESIGN.md</code>, <code>C0_REPORT.md</code>,
<code>L7_BASELINE_AUDIT.md</code>, <code>REPLICATION.md</code>.</li>
<li>Visual template: SC-SAM × STARCOP research proposal, v1.0, 14 October 2026 (layout only).</li>
</ol>
<div class="foot"><span>Seed 42, seeds 1–5, C0 gate and C5 measured • No C1–C3 outcomes claimed</span>
<span>built by <code>build_proposal.py</code></span></div>
</div>"""
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Few-shot OCT figure pack</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def main():
    figs = {"fig1_pipeline": fig1, "fig2_results": fig2, "fig3_confound": fig3,
            "fig4_qc": fig4, "fig5_c1_protocol": fig5, "fig6_ladder": fig6}
    for name, fn in figs.items():
        s = fn().replace('width="100%"', 'width="880"', 1)
        (OUT / f"{name}.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + s, encoding="utf-8")
    (OUT / "proposal.html").write_text(page(), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
