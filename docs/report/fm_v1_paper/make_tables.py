"""Write the LaTeX tables of the v1 paper from the locked archive and the post hoc CSVs.

Formatting only: every cell is read from a file named in the table's source comment; nothing is
computed on images, features or per-episode outputs. Sums appear only where the drafts already
state them (hold-out totals of nine-way counts per arm).

Rounding (agent default, open to Lam's objection, protocols.md 19 Nov "after the draft review"):
three decimals, half up (away from zero) from the shortest decimal form of the stored value,
taken as its 12-significant-digit form, so 0.7264999999999999 -> 0.7265 -> 0.727.

Phase-1 controls E1-E4 (protocols.md section 8, "Phase-1 controls E1-E4"): the e_* generators read
the CSVs that `fm_posthoc.py e1|e2|e3|e4` write (and E2f_marks_<pass>.csv from `e2f-marks`). A
generator whose CSV does not exist yet writes nothing and says nothing; main.tex and supplement.tex
show a generated table only when its .tex file exists. Nothing is computed beyond counting the E2d
rows whose interval excludes 0.5, as fm_posthoc.e2_report does for E2_comparison.md.

    D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_v1_paper/make_tables.py
    # E tables only, from another folder (e.g. a dry run), into a scratch folder:
    D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_v1_paper/make_tables.py --e-only --e-dir <dir> --out <scratch>

Stand-in or test tables never enter tables/: with --e-dir other than results/fm_v1/posthoc, or with
any E CSV labelled "dry run", writing into tables/ is refused.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PH = REPO / "results" / "fm_v1" / "posthoc"
ATTEMPT = Path("D:/Few-shot/drive/fm_v1/final/final-20261113T170804Z")
ARC = ATTEMPT / "evaluation" / "results-20261114T093033Z.zip"
TABLES = HERE / "tables"
OUT = TABLES  # --out
E_DIR = PH  # --e-dir

ARMS = ("MIRAGE", "DINOv2", "Inception", "RETFound")
NAME = {"MIRAGE": "MIRAGE", "DINOv2": "DINOv2", "Inception": "Inception-v3", "RETFound": "RETFound"}
ROLE = {"MIRAGE": "MIRAGE (nominee)", "DINOv2": "DINOv2", "Inception": "Inception-v3 (reference)",
        "RETFound": "RETFound"}
RARE = ("CSC", "MH", "MacTel", "Stargardt", "RP")
FN = r"\textsuperscript{a}"  # RP/Stargardt folder-label footnote mark
MINUS = "$-$"


def dec(x, nd=3) -> Decimal:
    return Decimal(format(float(x), ".12g")).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def f(x, nd=3, sign=False) -> str:
    """Rounded number as LaTeX text; sign=True prints + for positives."""
    if x is None or x == "":
        return "--"
    d = dec(x, nd)
    s = format(abs(d), "f")
    neg = d.is_signed() and float(x) < 0
    if neg:
        return MINUS + s
    return ("+" + s) if (sign and d != 0) else s


def rng(lo, hi, sign=False) -> str:
    """5th-95th range: en dash for unsigned quantities, 'to' for signed differences."""
    return f"{f(lo, sign=sign)} to {f(hi, sign=sign)}" if sign else f"{f(lo)}--{f(hi)}"


def two(top: str, bottom: str) -> str:
    return r"\makecell{" + top + r"\\{\scriptsize " + bottom + "}}"


def read_arc_csv(z: zipfile.ZipFile, name: str) -> list[dict]:
    return list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))


def read_ph(name: str) -> list[dict]:
    with open(PH / name, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


# ---------------------------------------------------------------- main text

def t_development(z):
    nom = json.load(z.open("reports/nomination/nomination.json"))["arms"]
    r2 = {(r["arm"], r["k"]): r for r in read_ph("R2_development_spread.csv")}
    size = {r["k"]: r for r in read_arc_csv(z, "reports/legacy/size_only_reference.csv") if r["phase"] == "development"}
    ks = ("1", "2", "4", "8")
    L = [r"% src: means ARC reports/nomination/nomination.json arms.<arm>.k_scores and .score (locked);",
         r"% src: SD and 5th-95th of arm and paired rows: PH R2_development_spread.csv (rule R2, commit 65fe98c,",
         r"%      computed 2026-11-20 after the results were known; gate: R2 means equal nomination.json within 1e-9);",
         r"%      paired means: R2 'X minus Inception' rows; size-only: ARC reports/legacy/size_only_reference.csv lines 2-6.",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Arm & $K=1$ & $K=2$ & $K=4$ & $K=8$ & Mean $K$ 1--8 \\", r"\midrule"]
    for arm in ARMS:
        cells = []
        for k in ks:
            m = nom[arm]["k_scores"][k]
            assert abs(m - float(r2[(arm, k)]["mean"])) < 1e-9
            cells.append(two(f(m), f"({f(r2[(arm, k)]['sd'])})" + r"$^\dagger$"))
        s = r2[(arm, "mean_k1_2_4_8")]
        assert abs(nom[arm]["score"] - float(s["mean"])) < 1e-9
        cells.append(two(f(nom[arm]["score"]), f"({f(s['sd'])}; {rng(s['p05'], s['p95'])})" + r"$^\dagger$"))
        L.append(ROLE[arm] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for arm in ("MIRAGE", "DINOv2", "RETFound"):
        cells = []
        for k in ks + ("mean_k1_2_4_8",):
            r = r2[(f"{arm} minus Inception", k)]
            cells.append(two(f(r["mean"], sign=True),
                             f"({f(r['sd'])}; {rng(r['p05'], r['p95'], sign=True)})" + r"$^\dagger$"))
        L.append(r"\makecell[l]{" + f"{NAME[arm]} $-$" + r"\\Inception-v3 (paired)} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    cells = [two(f(size[k]["mean"]), f"({f(size[k]['sd'])})") for k in ks]
    s = size["mean_k1_2_4_8"]
    cells.append(two(f(s["mean"]), f"({f(s['sd'])}; {rng(s['p05'], s['p95'])})"))
    L.append("Size-only reference & " + " & ".join(cells) + r" \\")
    L.append(r"Chance & 0.200 & 0.200 & 0.200 & 0.200 & 0.200 \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t1_development", L)


def t_frr(z):
    audit = {(r["arm"], r["source"], r["endpoint"]): r for r in read_arc_csv(z, "reports/external/external_source_audit.csv")}
    r1 = {}
    for r in read_ph("R1_rare_beside_false_rare.csv"):
        if r["rare_class"]:
            r1[(r["arm"], r["source"], r["rare_class"])] = r

    def cell(arm, src, ep):
        r = audit[(arm, src, ep)]
        return two(r["numerator"], f"({f(r['value'])}; {rng(r['interval_low'], r['interval_high'])})")

    L = [r"% src: ARC reports/external/external_source_audit.csv (numerator, value, interval_low, interval_high);",
         r"%      Kermany 169 = false_rare_rate_patient_disjoint, Kermany 250 = false_rare_rate_all, OCTID/OCTDL = false_rare_rate;",
         r"%      called NORMAL = recall_NORMAL numerators; R1 (post hoc) = PH R1_rare_beside_false_rare.csv rare_called_any_rare.",
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Normal scans called any rare class: count (rate; 95\% interval)} & Called NORMAL & R1$^\ddagger$: disease images called any rare \\",
         r"\cmidrule(lr){2-5}",
         r"Arm & \makecell{Kermany, of 169\\(headline)} & \makecell{Kermany, of 250\\(secondary)} & OCTID, of 206 & OCTDL, of 332 & OCTID / OCTDL & \makecell{OCTID CSC / MH (of 102);\\OCTDL MH (of 39)} \\",
         r"\midrule"]
    for arm in ARMS:
        cells = [cell(arm, "Kermany", "false_rare_rate_patient_disjoint"), cell(arm, "Kermany", "false_rare_rate_all"),
                 cell(arm, "OCTID", "false_rare_rate"), cell(arm, "OCTDL", "false_rare_rate"),
                 f"{audit[(arm, 'OCTID', 'recall_NORMAL')]['numerator']} / {audit[(arm, 'OCTDL', 'recall_NORMAL')]['numerator']}",
                 f"{r1[(arm, 'OCTID', 'CSC')]['rare_called_any_rare']} / {r1[(arm, 'OCTID', 'MH')]['rare_called_any_rare']}; "
                 f"{r1[(arm, 'OCTDL', 'MH')]['rare_called_any_rare']}"]
        L.append(NAME[arm] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t2_false_rare", L)


def t_holdout(z):
    audit = {(r["arm"], r["source"], r["endpoint"]): r for r in read_arc_csv(z, "reports/external/external_source_audit.csv")}
    sds = {(r["arm"], r["k"]): r for r in read_arc_csv(z, "reports/legacy/support_draw_summary.csv") if r["classifier"] == "prototype"}
    pd = {(r["arm_a"], r["arm_b"], r["k"]): r for r in read_arc_csv(z, "reports/legacy/paired_differences.csv") if r["classifier"] == "prototype"}
    size = {r["k"]: r for r in read_arc_csv(z, "reports/legacy/size_only_reference.csv") if r["phase"] == "reporting"}
    ks = ("1", "2", "4", "8", "16")
    L = [r"% src: nine-way counts ARC reports/external/external_source_audit.csv rows <arm>,Kermany,recall_<rare class> (numerator);",
         r"%      total = sum of the five numerators (also WR l.288-291: 20, 15, 13, 9);",
         r"%      five-way: ARC reports/legacy/support_draw_summary.csv (classifier=prototype); paired: ARC reports/legacy/paired_differences.csv",
         r"%      (MIRAGE-Inception role primary; DINOv2-, RETFound-Inception role secondary); size-only: size_only_reference.csv lines 7-12.",
         r"\begin{tabular}{@{}l*{6}{c}*{6}{c}@{}}", r"\toprule",
         r" & \multicolumn{6}{c}{\makecell{Nine-way head: images named\\correctly (exact counts)}} & \multicolumn{6}{c}{Five-way prototype: macro recall, mean of 100 draws} \\",
         r"\cmidrule(lr){2-7}\cmidrule(l){8-13}",
         r"Arm & \makecell{CSC\\(5)} & \makecell{MH\\(5)} & \makecell{MacTel\\(4)} & \makecell{Starg." + FN + r"\\(4)} & \makecell{RP" + FN + r"\\(4)} & \makecell{Total\\(22)} & $K=1$ & $K=2$ & $K=4$ & $K=8$ & $K=16$ & Mean $K$ 1--8 \\",
         r"\midrule"]
    for arm in ARMS:
        counts = [int(audit[(arm, "Kermany", f"recall_{c}")]["numerator"]) for c in RARE]
        five = [f(sds[(arm, k)]["mean"]) for k in ks]
        s = sds[(arm, "mean_k1_2_4_8")]
        five.append(two(f(s["mean"]), f"({f(s['sd'])}; {rng(s['p05'], s['p95'])})"))
        L.append(ROLE[arm] + " & " + " & ".join(str(c) for c in counts) + f" & {sum(counts)} & " + " & ".join(five) + r" \\")
    L.append(r"\midrule")
    for arm in ("MIRAGE", "DINOv2", "RETFound"):
        cells = [two(f(pd[(arm, "Inception", k)]["mean_a_minus_b"], sign=True),
                     f"({rng(pd[(arm, 'Inception', k)]['p05'], pd[(arm, 'Inception', k)]['p95'], sign=True)})") for k in ks]
        r = pd[(arm, "Inception", "mean_k1_2_4_8")]
        cells.append(two(f(r["mean_a_minus_b"], sign=True), f"({f(r['sd'])}; {rng(r['p05'], r['p95'], sign=True)})"))
        L.append(r"\makecell[l]{" + f"{NAME[arm]} $-$" + r"\\Inception-v3 (paired)} & \multicolumn{6}{c}{} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    cells = [f(size[k]["mean"]) for k in ks]
    s = size["mean_k1_2_4_8"]
    cells.append(two(f(s["mean"]), f"({f(s['sd'])}; {rng(s['p05'], s['p95'])})"))
    L.append(r"Size-only reference & \multicolumn{6}{c}{} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t3_holdout", L)


def t_octid(z):
    rows = read_arc_csv(z, "reports/external/external_two_way.csv")
    by = {(r["arm"], r["k"]): r for r in rows}
    ks = ("1", "2", "4", "8", "16")
    L = [r"% src: ARC reports/external/external_two_way.csv columns macro_mean (per k), macro_sd/p05/p95 (mean_k1_2_4_8 line),",
         r"%      collapsed_draws (k = 1, 2, 4, 8). MIRAGE K=4 stored 0.7725 -> 0.773 under the rounding rule.",
         r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r"Arm & $K=1$ & $K=2$ & $K=4$ & $K=8$ & $K=16$ & \makecell{Mean $K$ 1--8\\(SD; 5th--95th)} & \makecell{Draws collapsed\\($K$ = 1, 2, 4, 8)} \\",
         r"\midrule"]
    for arm, label in [(a, NAME[a] + (" (headline)" if a == "MIRAGE" else "")) for a in ARMS] + [("size_only_reference", "Size-only reference")]:
        cells = [f(by[(arm, k)]["macro_mean"]) for k in ks]
        m = by[(arm, "mean_k1_2_4_8")]
        cells.append(f"{f(m['macro_mean'])} ({f(m['macro_sd'])}; {rng(m['macro_p05'], m['macro_p95'])})")
        cells.append(", ".join(by[(arm, k)]["collapsed_draws"] for k in ("1", "2", "4", "8")))
        L.append(label + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t4_octid_two_way", L)


def t_p4():
    rows = read_ph("P4_size_right_vs_wrong.csv")
    by = {(r["arm"], r["group"], r["k"]): r for r in rows}
    label = rows[0]["posthoc"]
    L = [r"% src: PH P4_size_right_vs_wrong.csv column mean_episode_macro (lines 2-41); label: " + label,
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Arm & Queries & $K=1$ & $K=2$ & $K=4$ & $K=8$ & Mean \\", r"\midrule"]
    for i, arm in enumerate(ARMS):
        for g, gl in (("size_right", "size-right"), ("size_wrong", "size-wrong")):
            cells = [f(by[(arm, g, k)]["mean_episode_macro"]) for k in ("1", "2", "4", "8", "mean_k1_2_4_8")]
            L.append((NAME[arm] if g == "size_right" else "") + f" & {gl} & " + " & ".join(cells) + r" \\")
        if i < 3:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t5_p4_size_right_wrong", L)


def p4_entries():
    """P4's 'report how many episodes each class enters': one count per group, class and K. The groups come from the
    size-only rule, so the counts are the same for every arm; the table refuses to print if they are not."""
    rows = read_ph("P4_class_entries.csv")
    by: dict[tuple, set] = {}
    for r in rows:
        by.setdefault((r["group"], r["class"], r["k"]), set()).add((r["episodes_entered"], r["episodes_planned"]))
    if any(len(v) != 1 for v in by.values()):
        raise SystemExit("P4_class_entries.csv: episode counts differ between arms; print them per arm")
    planned = {p for v in by.values() for _e, p in v}
    L = [r"% src: PH P4_class_entries.csv columns episodes_entered, episodes_planned (lines 2-161; equal for all four arms);",
         r"%      label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Class & Queries & $K=1$ & $K=2$ & $K=4$ & $K=8$ \\", r"\midrule"]
    for i, c in enumerate(RARE):
        for g, gl in (("size_right", "size-right"), ("size_wrong", "size-wrong")):
            cells = [next(iter(by[(g, c, k)]))[0] for k in KS]
            L.append((c if g == "size_right" else "") + f" & {gl} & " + " & ".join(cells) + r" \\")
        if i < len(RARE) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p4_class_entries", L)
    (OUT / "p4_class_entries_planned.tex").write_text(" / ".join(sorted(planned)) + "\n", encoding="utf-8")


def t_p8():
    rows = read_ph("P8_csc_vs_mh_development.csv")
    by = {(r["arm"], r["k"]): r for r in rows}
    label = rows[0]["posthoc"]
    L = [r"% src: PH P8_csc_vs_mh_development.csv lines 2-31 (mean; sd/p05/p95 on the mean line; paired per k p05/p95);",
         r"%      last column = locked_OCTID_two_way_macro_mean on the mean line (= Table 4). Label: " + label,
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Arm & $K=1$ & $K=2$ & $K=4$ & $K=8$ & \makecell{Mean $K$ 1--8\\(SD; 5th--95th)} & \makecell{OCTID\\(locked)} \\",
         r"\midrule"]
    for arm, lab in [(a, NAME[a]) for a in ARMS] + [("size_only", "Size-only")]:
        cells = [f(by[(arm, k)]["mean"]) for k in ("1", "2", "4", "8")]
        m = by[(arm, "mean_k1_2_4_8")]
        cells.append(two(f(m["mean"]), f"({f(m['sd'])}; {rng(m['p05'], m['p95'])})"))
        lk = m["locked_OCTID_two_way_macro_mean"]
        cells.append(f(lk) if lk else "0.500")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    cells = [two(f(by[("MIRAGE-Inception", k)]["mean"], sign=True),
                 f"({rng(by[('MIRAGE-Inception', k)]['p05'], by[('MIRAGE-Inception', k)]['p95'], sign=True)})") for k in ("1", "2", "4", "8")]
    m = by[("MIRAGE-Inception", "mean_k1_2_4_8")]
    cells.append(two(f(m["mean"], sign=True), f"({f(m['sd'])}; {rng(m['p05'], m['p95'], sign=True)})"))
    cells.append(f(m["locked_OCTID_two_way_macro_mean"], sign=True))
    L.append(r"\makecell[l]{MIRAGE $-$\\Inception-v3 (paired)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t6_p8_csc_mh_development", L)


# ---------------------------------------------------------------- supplement (locked)

def s_geometry(z):
    mir = read_arc_csv(z, "reports/geometry/MIRAGE/geometry_summary.csv")
    dino = {(r["source"], r["class"], r["role"]): r for r in read_arc_csv(z, "reports/geometry/DINOv2/geometry_summary.csv")}
    order = {"benchmark": 0, "Kermany": 1, "OCTID": 2, "OCTDL": 3}
    cls = {c: i for i, c in enumerate(RARE + ("NORMAL", "CNV", "DME", "DRUSEN"))}
    mir.sort(key=lambda r: (order[r["source"]], cls[r["class"]], r["role"] != "fitting"))
    role = {"fitting": "development", "query": "hold-out"}
    L = [r"% src: ARC reports/geometry/MIRAGE/geometry_summary.csv lines 2-24 (all columns except retained fraction);",
         r"%      retained fraction: same file (MIRAGE = Inception-v3, 1.0 everywhere) and reports/geometry/DINOv2/geometry_summary.csv",
         r"%      (DINOv2 = RETFound, files identical by MD5 per the draft, methods_results_v2.md section 8, Table 6).",
         r"\begin{tabular}{@{}lllrcccccc@{}}", r"\toprule",
         r"Source & Class & Role & $n$ & Width, px & Height, px & Border share & Near-white share & \makecell{Retained:\\MIRAGE, Inc.} & \makecell{Retained:\\DINOv2, RETFound} \\",
         r"\midrule"]
    for r in mir:
        d = dino[(r["source"], r["class"], r["role"])]
        rl = role[r["role"]] if r["source"] == "benchmark" else ("fitting" if r["role"] == "fitting" else ("test" if r["source"] == "Kermany" else "query"))
        c = r["class"] + (FN if r["class"] in ("RP", "Stargardt") else "")
        L.append(" & ".join([
            r["source"], c, rl, r["n"],
            f"{int(float(r['width_median']))} ({int(float(r['width_min']))}--{int(float(r['width_max']))})" if float(r['width_median']).is_integer() else f"{r['width_median']} ({int(float(r['width_min']))}--{int(float(r['width_max']))})",
            f"{float(r['height_median']):g} ({int(float(r['height_min']))}--{int(float(r['height_max']))})",
            f"{f(r['border_share_median'])} ({f(r['border_share_min'])}--{f(r['border_share_max'])})",
            f(r["near_white_share_median"]),
            f(r["retained_fraction_median"]),
            f"{f(d['retained_fraction_median'])} ({f(d['retained_fraction_min'])}--{f(d['retained_fraction_max'])})",
        ]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s_geometry", L)


def s_logistic(z):
    sds = {(r["arm"], r["k"]): r for r in read_arc_csv(z, "reports/legacy/support_draw_summary.csv") if r["classifier"] == "logistic"}
    pd = {(r["arm_a"], r["arm_b"], r["k"]): r for r in read_arc_csv(z, "reports/legacy/paired_differences.csv") if r["classifier"] == "logistic"}
    ks = ("1", "2", "4", "8", "16")
    L = [r"% src: ARC reports/legacy/support_draw_summary.csv classifier=logistic; reports/legacy/paired_differences.csv classifier=logistic.",
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Arm & $K=1$ & $K=2$ & $K=4$ & $K=8$ & $K=16$ & \makecell{Mean $K$ 1--8\\(SD; 5th--95th)} \\", r"\midrule"]
    for arm in ARMS:
        cells = [f(sds[(arm, k)]["mean"]) for k in ks]
        s = sds[(arm, "mean_k1_2_4_8")]
        cells.append(f"{f(s['mean'])} ({f(s['sd'])}; {rng(s['p05'], s['p95'])})")
        L.append(NAME[arm] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for arm in ("MIRAGE", "DINOv2", "RETFound"):
        cells = [f(pd[(arm, "Inception", k)]["mean_a_minus_b"], sign=True) for k in ks]
        r = pd[(arm, "Inception", "mean_k1_2_4_8")]
        cells.append(f"{f(r['mean_a_minus_b'], sign=True)} ({f(r['sd'])}; {rng(r['p05'], r['p95'], sign=True)})")
        L.append(f"{NAME[arm]} $-$ Inception-v3 & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s1_logistic", L)


def s_transfer(z):
    rows = read_arc_csv(z, "reports/external/external_transfer_summary.csv")
    by = {(r["arm"], r["source"], r["classifier"], r["k"], r["class"]): r for r in rows}
    ks = ("1", "2", "4", "8", "16")
    L = [r"% src: ARC reports/external/external_transfer_summary.csv column mean (share of queries given their own class).",
         r"\begin{tabular}{@{}llcccccccccc@{}}", r"\toprule",
         r" & & \multicolumn{5}{c}{Prototype (100 draws)} & \multicolumn{5}{c}{Logistic probe (25 draws)} \\",
         r"\cmidrule(lr){3-7}\cmidrule(l){8-12}",
         r"Images & Arm & 1 & 2 & 4 & 8 & 16 & 1 & 2 & 4 & 8 & 16 \\", r"\midrule"]
    for src, c, lab in (("OCTID", "CSC", "OCTID CSC (102)"), ("OCTID", "MH", "OCTID MH (102)"), ("OCTDL", "MH", "OCTDL MH (39)")):
        for i, arm in enumerate(ARMS):
            cells = [f(by[(arm, src, cl, k, c)]["mean"]) for cl in ("prototype", "logistic") for k in ks]
            L.append((lab if i == 0 else "") + f" & {NAME[arm]} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    write("s2_transfer", L)


def s_perclass(z):
    audit = {(r["arm"], r["source"], r["endpoint"]): r for r in read_arc_csv(z, "reports/external/external_source_audit.csv")}
    sets = [("Kermany", c) for c in ("NORMAL", "CNV", "DME", "DRUSEN") + RARE] + [("OCTID", c) for c in ("NORMAL", "CSC", "MH")] + [("OCTDL", c) for c in ("NORMAL", "MH")]
    L = [r"% src: ARC reports/external/external_source_audit.csv endpoints recall_<class>: numerator/denominator (value; interval_low--interval_high).",
         r"\begin{tabular}{@{}ll" + "c" * 4 + r"@{}}", r"\toprule",
         r"Source & Class & " + " & ".join(NAME[a] for a in ARMS) + r" \\", r"\midrule"]
    for src, c in sets:
        cells = []
        for arm in ARMS:
            r = audit[(arm, src, f"recall_{c}")]
            cells.append(f"{r['numerator']}/{r['denominator']} ({f(r['value'])}; {rng(r['interval_low'], r['interval_high'])})")
        lab = ("Kermany + hold-out" if src == "Kermany" else src)
        L.append(f"{lab} & {c}{FN if c in ('RP', 'Stargardt') else ''} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s3_perclass", L)


def s_confusion(z):
    L = [r"% src: ARC runs/<arm>/reporting/nine_way/metrics.json and runs/<arm>/external/{OCTID,OCTDL}/nine_way/metrics.json,",
         r"%      keys classes and confusion (rows true, columns assigned); empty true rows dropped."]
    short = {"NORMAL": "NOR", "CNV": "CNV", "DME": "DME", "DRUSEN": "DRU", "CSC": "CSC", "MH": "MH", "MacTel": "MacT",
             "Stargardt": "Starg", "RP": "RP"}
    L += [r"\begin{tabular}{@{}lll*{9}{r}@{}}", r"\toprule"]
    for i, arm in enumerate(ARMS):
        for j, (src, path) in enumerate((("Kermany + hold-out", f"runs/{arm}/reporting/nine_way/metrics.json"),
                                         ("OCTID", f"runs/{arm}/external/OCTID/nine_way/metrics.json"),
                                         ("OCTDL", f"runs/{arm}/external/OCTDL/nine_way/metrics.json"))):
            m = json.load(z.open(path))
            if i == 0 and j == 0:
                L.append(r"Arm & Source & True & " + " & ".join(short[c] for c in m["classes"]) + r" \\")
                L.append(r"\midrule")
            first = True
            for c, row in zip(m["classes"], m["confusion"]):
                if sum(row) == 0:
                    continue
                L.append((NAME[arm] if (j == 0 and first) else "") + " & " + (src if first else "") + f" & {short[c]} & "
                         + " & ".join(str(v) for v in row) + r" \\")
                first = False
            L.append(r"\addlinespace[1pt]")
        L.append(r"\midrule" if i < 3 else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("s4_confusion", L)


def s_two_way(z):
    rows = read_arc_csv(z, "reports/external/external_two_way.csv")
    L = [r"% src: ARC reports/external/external_two_way.csv lines 2-31 (recall_CSC_mean, recall_MH_mean, mean confusion counts, collapsed_draws).",
         r"\begin{tabular}{@{}lrcccccccr@{}}", r"\toprule",
         r"Arm & $K$ & Macro & CSC recall & MH recall & CSC$\to$CSC & CSC$\to$MH & MH$\to$CSC & MH$\to$MH & Collapsed \\", r"\midrule"]
    last = None
    for r in rows:
        if r["k"] == "mean_k1_2_4_8":
            continue
        arm = "Size-only" if r["arm"] == "size_only_reference" else NAME[r["arm"]]
        if last and arm != last:
            L.append(r"\addlinespace[2pt]")
        L.append(" & ".join([arm if arm != last else "", r["k"], f(r["macro_mean"]), f(r["recall_CSC_mean"]), f(r["recall_MH_mean"]),
                             f"{float(r['CSC_as_CSC']):.2f}", f"{float(r['CSC_as_MH']):.2f}", f"{float(r['MH_as_CSC']):.2f}",
                             f"{float(r['MH_as_MH']):.2f}", r["collapsed_draws"]]) + r" \\")
        last = arm
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s7_two_way", L)


# ---------------------------------------------------------------- supplement (post hoc)

def p0():
    rows = read_ph("P0_per_class_recall.csv")
    by = {(r["arm"], r["class"]): r for r in rows}
    L = [r"% src: PH P0_per_class_recall.csv lines 2-25, column mean_k1_2_4_8; label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", "Class & " + " & ".join(NAME[a] for a in ARMS) + r" \\", r"\midrule"]
    for c in RARE + ("macro",):
        lab = {"macro": r"\midrule Macro recall"}.get(c, c + (FN if c in ("RP", "Stargardt") else ""))
        L.append(lab + " & " + " & ".join(f(by[(a, c)]["mean_k1_2_4_8"]) for a in ARMS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p0_per_class", L)


def p1():
    rows = read_ph("P1_nine_way_exact_bounds.csv")
    want = [("Kermany patient-disjoint", "NORMAL"), ("Kermany all test", "NORMAL"), ("OCTID", "NORMAL"), ("OCTDL", "NORMAL"),
            ("OCTID", "CSC"), ("OCTID", "MH"), ("OCTDL", "MH")]
    by = {(r["arm"], r["set"], r["true_class"]): r for r in rows if r["count"] == "called_any_rare"}
    L = [r"% src: PH P1_nine_way_exact_bounds.csv rows count=called_any_rare (k, n, cp95_low, cp95_high); label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}ll" + "c" * 4 + r"@{}}", r"\toprule",
         "Set & True class & " + " & ".join(NAME[a] for a in ARMS) + r" \\", r"\midrule"]
    for s, c in want:
        cells = []
        for a in ARMS:
            r = by[(a, s, c)]
            cells.append(f"{r['k']}/{r['n']} ({rng(r['cp95_low'], r['cp95_high'])})")
        L.append(f"{s} & {c} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p1_exact_bounds", L)


def p2():
    rows = read_ph("P2_october_heads_external.csv")
    L = [r"% src: PH P2_october_heads_external.csv lines 2-16; gates PH P2_gates.csv; label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}lllrrrrr@{}}", r"\toprule",
         r"Head & Source & True class & $n$ & Correct & NORMAL & \makecell{CNV, DME\\or DRUSEN} & Any rare \\", r"\midrule"]
    last = None
    for r in rows:
        if last and r["variant"] != last:
            L.append(r"\addlinespace[2pt]")
        L.append(" & ".join([r["variant"] if r["variant"] != last else "", r["source"], r["true_class"], r["n"], r["correct_class"],
                             r["called_NORMAL"], r["called_CNV_DME_DRUSEN"], r["called_any_rare"]]) + r" \\")
        last = r["variant"]
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p2_october_heads", L)


def p3():
    rows = [r for r in read_ph("P3_clinic_to_clinic_mh.csv") if r["k"] == "mean_k1_2_4_8"]
    L = [r"% src: PH P3_clinic_to_clinic_mh.csv rows k=mean_k1_2_4_8 (mean, sd, p05, p95, recall_NORMAL_mean, recall_MH_mean);",
         r"%      label: " + read_ph("P3_clinic_to_clinic_mh.csv")[0]["posthoc"],
         r"\begin{tabular}{@{}lllccc@{}}", r"\toprule",
         r"Arm & Direction & Unit & \makecell{Two-way macro recall\\(SD; 5th--95th)} & NORMAL recall & MH recall \\", r"\midrule"]
    last = None
    for r in rows:
        if last and r["arm"] != last:
            L.append(r"\addlinespace[2pt]")
        d = r["direction"].replace("->", r"$\to$")
        L.append(" & ".join([NAME[r["arm"]] if r["arm"] != last else "", d, r["unit"],
                             f"{f(r['mean'])} ({f(r['sd'])}; {rng(r['p05'], r['p95'])})", f(r["recall_NORMAL_mean"]),
                             f(r["recall_MH_mean"])]) + r" \\")
        last = r["arm"]
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p3_clinic_to_clinic", L)


def p5():
    rows = read_ph("P5_not_sure_cutoff.csv")
    L = [r"% src: PH P5_not_sure_cutoff.csv lines 2-13; label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}llrrcccc@{}}", r"\toprule",
         r"Arm & Set & Normals & Not sure & \makecell{Called rare\\(all / sure)} & \makecell{Called NORMAL\\(all / sure)} & \makecell{Called common\\(all / sure)} & Role \\",
         r"\midrule"]
    last = None
    for r in rows:
        if last and r["arm"] != last:
            L.append(r"\addlinespace[2pt]")
        st = r["set"].replace(" (calibration)", "")
        L.append(" & ".join([NAME[r["arm"]] if r["arm"] != last else "", st, r["normals"], r["not_sure"],
                             f"{r['false_rare']} / {r['false_rare_sure']}", f"{r['called_normal']} / {r['called_normal_sure']}",
                             f"{r['called_common']} / {r['called_common_sure']}",
                             "calibration" if "calibration" in r["role"] else "result"]) + r" \\")
        last = r["arm"]
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p5_not_sure", L)


def p6():
    rows = read_ph("P6_probe_vs_prototype.csv")
    by = {(r["arm"], r["set"], r["k"]): r for r in rows}
    L = [r"% src: PH P6_probe_vs_prototype.csv (diff_mean, diff_p05, diff_p95; logistic minus prototype, first 25 draws); label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & Development & \multicolumn{5}{c}{Hold-out} \\ \cmidrule(lr){2-2}\cmidrule(l){3-7}",
         r"Arm & $K=4$ & $K=1$ & $K=2$ & $K=4$ & $K=8$ & $K=16$ \\", r"\midrule"]
    for a in ARMS:
        cells = [two(f(by[(a, s, k)]["diff_mean"], sign=True), f"({rng(by[(a, s, k)]['diff_p05'], by[(a, s, k)]['diff_p95'], sign=True)})")
                 for s, k in [("development", "4")] + [("hold-out", k) for k in ("1", "2", "4", "8", "16")]]
        L.append(NAME[a] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p6_probe_vs_prototype", L)


def p9_table(source: str, name: str) -> None:
    """P9: beside each AUROC, the locked argmax point (rare scans called any rare; normals called rare, as in R1) and
    the rare share above the score that only 5% of that set's normals exceed (read from the test set: an upper bound)."""
    rows = read_ph(source)
    by = {(r["arm"], r["set"]): r for r in rows}
    sets = list(dict.fromkeys(r["set"] for r in rows))
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    L = [r"% src: PH " + source + " columns auroc, boot95_low, boot95_high, argmax_rare_called_any_rare, argmax_normals_called_rare,",
         r"%      rare_share_above_5pct_normal_cut_upper_bound (lines 2-" + str(len(rows) + 1) + "); label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Argmax point (share)} & \\ \cmidrule(lr){4-5}",
         r"Set & Arm & AUROC (95\% interval) & \makecell{Rare scans\\called rare} & \makecell{Normals\\called rare} & \makecell{Rare share above the\\5\% normal cut (upper bound)} \\",
         r"\midrule"]
    for i, s in enumerate(sets):
        lab = s.replace("Legacy 22 against 169 patient-disjoint Kermany normals", r"\makecell[l]{Hold-out 22 vs 169 Kermany\\normals (source and disease mixed)}")
        for j, a in enumerate(arms):
            r = by[(a, s)]
            L.append(" & ".join([lab if j == 0 else "", NAME[a], f"{f(r['auroc'])} ({rng(r['boot95_low'], r['boot95_high'])})",
                                 f(r["argmax_rare_called_any_rare"]), f(r["argmax_normals_called_rare"]),
                                 f(r["rare_share_above_5pct_normal_cut_upper_bound"])]) + r" \\")
        if i < len(sets) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(name, L)


def p9():
    p9_table("P9_rare_vs_normal_auroc.csv", "p9_auroc")
    p9_table("P9_G1_padded_auroc.csv", "p9_g1_padded")


def g1():
    rows = read_ph("G1_pad_to_square.csv")
    by = {(r["arm"], r["table"], r["k"], r["quantity"]): r for r in rows}
    # Every element the G1 rule names: development macro recall, hold-out counts, the size-only comparison and the
    # OCTDL normal destinations, beside the locked numbers. Size-only depends on no model (equal for both arms).
    L = [r"% src: PH G1_pad_to_square.csv columns locked, padded, size_only (development lines 6-10/54-58; hold-out 11-21/59-69,",
         r"%      mean_correct_of_22 on the even hold-out lines; nine-way lines 22-49/70-97). K = 16 omitted from any reading",
         r"%      (protocols.md 19 Nov G1 decision). Label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{MIRAGE} & \multicolumn{3}{c}{Inception-v3} & \\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Set & $K$ & Locked & Padded & Change & Locked & Padded & Change & Size-only \\", r"\midrule"]
    for t, tl in (("development", "Development"), ("hold-out", "Hold-out")):
        for k in ("1", "2", "4", "8", "mean_k1_2_4_8"):
            cells = []
            for a in ("MIRAGE", "Inception"):
                r = by[(a, t, k, "macro_recall_mean")]
                cells += [f(r["locked"]), f(r["padded"]), f(float(r["padded"]) - float(r["locked"]), sign=True)]
            so = {by[(a, t, k, "macro_recall_mean")]["size_only"] for a in ("MIRAGE", "Inception")}
            if len(so) != 1:
                raise SystemExit(f"G1_pad_to_square.csv: size_only differs between arms ({t}, K={k})")
            cells.append(f(so.pop()))
            L.append((tl if k == "1" else "") + f" & {'1--8' if k.startswith('mean') else k} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L[-1] = r"\midrule"
    L.append(r"\multicolumn{9}{@{}l}{Hold-out: mean number named correctly of 22 (100 draws)} \\")
    for k in KS:
        cells = []
        for a in ("MIRAGE", "Inception"):
            r = by[(a, "hold-out", k, "mean_correct_of_22")]
            cells += [f(r["locked"], 2), f(r["padded"], 2), f(float(r["padded"]) - float(r["locked"]), 2, sign=True)]
        L.append(f" & {k} & " + " & ".join(cells) + r" & -- \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{9}{@{}l}{Nine-way: called NORMAL / common disease (CNV, DME or DRUSEN) / any rare, locked $\to$ padded} \\")
    for s in ("Kermany patient-disjoint (n=169)", "OCTID NORMAL (n=206)", "OCTDL NORMAL (n=332)"):
        cells = []
        for a in ("MIRAGE", "Inception"):
            q = [by[(a, f"nine-way {s}", "", c)] for c in ("called_NORMAL", "called_CNV_DME_DRUSEN", "called_any_rare")]
            cells.append(r"\multicolumn{3}{c}{" + "/".join(x["locked"] for x in q) + r" $\to$ " + "/".join(x["padded"] for x in q) + "}")
        L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{{s.replace('(n=', '(')}}} & " + " & ".join(cells) + r" & -- \\")
    cells = []
    for a in ("MIRAGE", "Inception"):
        r = by[(a, "nine-way Legacy 22 (n=22)", "", "correct_class")]
        cells.append(r"\multicolumn{3}{c}{" + f"{r['locked']} $\\to$ {r['padded']}" + "}")
    L.append(r"\multicolumn{2}{@{}l}{Hold-out 22, named correctly} & " + " & ".join(cells) + r" & -- \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("g1_pad_to_square", L)


P10_EXCLUSIONS = (("locked (no exclusion)", "Locked"), ("marked same picture or same case", "\\makecell{Without same\\\\picture or case}"),
                  ("also counting unsure", "\\makecell{Also without\\\\unsure}"))


def p10_sensitivity():
    rows = read_ph("P10_sensitivity.csv")
    by = {(r["phase"], r["exclusion"], r["arm"], r["k"]): r for r in rows}
    excluded = {(r["phase"], r["exclusion"]): r["excluded"] for r in rows}
    label = rows[0]["posthoc"]
    phases = ("development", "hold-out")
    L = [r"% src: PH P10_sensitivity.csv rows k=mean_k1_2_4_8 (mean, sd, p05, p95) and column excluded; label: " + label,
         r"%      arm rows: mean (SD); paired rows ('<arm> minus Inception'): mean (p05 to p95). Locked columns equal R2 / the locked archive.",
         r"\begin{tabular}{@{}l" + "c" * 6 + r"@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Development (124 queries), mean $K$ 1--8} & \multicolumn{3}{c}{Hold-out (22 images), mean $K$ 1--8} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         "Arm & " + " & ".join(lab for _ in phases for _, lab in P10_EXCLUSIONS) + r" \\", r"\midrule"]
    for arm in ARMS:
        cells = []
        for ph in phases:
            for ex, _ in P10_EXCLUSIONS:
                r = by[(ph, ex, arm, "mean_k1_2_4_8")]
                cells.append(two(f(r["mean"]), f"({f(r['sd'])})"))
        L.append(ROLE[arm] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for arm in ("MIRAGE", "DINOv2", "RETFound"):
        cells = []
        for ph in phases:
            for ex, _ in P10_EXCLUSIONS:
                r = by[(ph, ex, f"{arm} minus Inception", "mean_k1_2_4_8")]
                cells.append(two(f(r["mean"], sign=True), f"({rng(r['p05'], r['p95'], sign=True)})"))
        L.append(r"\makecell[l]{" + f"{NAME[arm]} $-$" + r"\\Inception-v3 (paired)} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")

    def lead(text: str) -> str:
        n = int(text.split()[0])
        if n == 0:
            return "none"
        return r"\makecell{" + (f"{n:,} query--episode" + r"\\exclusions per arm" if "query" in text else f"{n} images") + "}"
    L.append("Excluded & " + " & ".join(lead(excluded[(ph, ex)]) for ph in phases for ex, _ in P10_EXCLUSIONS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p10_sensitivity", L)


P10_MARK_ORDER = ("same picture", "same case, other slice", "unsure")


def p10_pairs():
    pairs = read_ph("P10_matched_pairs.csv")
    key = read_ph("P10_review_key.csv")
    marks = {r["item"]: r for r in read_ph("P10_review_marks.csv")}
    # Same join as fm_posthoc.p10_matched_pairs: every reviewed item except the synthetic copies.
    review = {frozenset((r["left"], r["right"])): (marks[r["item"]]["mark"], r["inliers"]) for r in key if r["kind"] != "synthetic"}
    unique: dict[frozenset, dict] = {}
    for r in pairs:
        k = frozenset((r["image_a"], r["image_b"]))
        unique.setdefault(k, dict(r, sets=[]))["sets"].append(r["exclusion"])
    entries = []
    for k, r in unique.items():
        mark, inliers = review[k]
        first = "marked same picture or same case" in r["sets"]
        assert first == (mark in P10_MARK_ORDER[:2]) and (mark != "unsure" or r["sets"] == ["also counting unsure"]), (k, mark)
        entries.append((P10_MARK_ORDER.index(mark), r["image_a"], r, mark, inliers))
    counts = [sum(e[0] == i for e in entries) for i in range(3)]
    assert counts == [14, 1, 1], counts  # the 21 Nov change-log entry (protocols.md section 8)
    short = lambda p: p.replace("data/raw/rare/", "")  # noqa: E731
    L = [r"% src: PH P10_matched_pairs.csv (pairs, sets); mark and screen inliers: PH P10_review_key.csv joined with",
         r"%      PH P10_review_marks.csv on item (synthetic copies excluded, as in fm_posthoc.p10_matched_pairs);",
         r"%      label: " + pairs[0]["posthoc"],
         r"\begin{tabular}{@{}rllllr@{}}", r"\toprule",
         r" & Image & Set & Image & Set & Inliers \\", r"\midrule"]
    n = 0
    for i, mark in enumerate(P10_MARK_ORDER):
        group = sorted((e for e in entries if e[0] == i), key=lambda e: (e[1], e[2]["image_b"]))
        L.append(r"\multicolumn{6}{@{}l}{\emph{" + {"same picture": "Same picture", "same case, other slice": "Same case, other slice",
                                                    "unsure": "Unsure (counted only in the second exclusion set)"}[mark]
                 + f" ({len(group)})" + r"}} \\")
        for _, _, r, _, inliers in group:
            n += 1
            sa, sb = (r"\textbf{hold-out}" if s == "hold-out" else s for s in (r["set_a"], r["set_b"]))
            L.append(f"{n} & {short(r['image_a'])} & {sa} & {short(r['image_b'])} & {sb} & {inliers}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p10_pairs", L)


# ---------------------------------------------------------------- Phase-1 controls E1-E4 (post hoc)

E_FILES = {
    1: ("E1a_checks.csv", "E1b_development.csv", "E1c_paired.csv", "E1d_success.csv", "E1e_matched.csv",
        "E1f_holdout.csv", "E1g_nine_way.csv", "E1h_leave_one_out.csv"),
    2: ("E2a_destinations.csv", "E2b_overlap.csv", "E2c_transitions.csv", "E2d_properties.csv", "E2d_categorical.csv",
        "E2e_octdl_patients.csv", "E2e_octdl_patient_scans.csv"),
    3: ("E3_checks.csv", "E3_development.csv", "E3_holdout.csv", "E3_nine_way.csv", "E3_criteria.csv"),
    4: ("E4a_fewshot_source_test.csv", "E4b_column_auroc.csv", "E4c_probe_auroc.csv", "E4d_probe_differences.csv"),
}
FSETS = ("locked", "E1-M", "E1-L", "E1-C")
KM = "mean_k1_2_4_8"
KS = ("1", "2", "4", "8")
NORMALS = ("Kermany patient-disjoint", "OCTID NORMAL", "OCTDL NORMAL")
RARE_EXT = ("OCTID CSC", "OCTID MH", "OCTDL MH")
REFS = {"size_only": "Size-only", "content_size": "Content-size", "thumbnail": "Thumbnail"}
NINE = ("NORMAL", "CNV", "DME", "DRUSEN") + RARE
SHORT9 = ("NOR", "CNV", "DME", "DRU", "CSC", "MH", "MacT", "Starg" + FN, "RP" + FN)
PLAIN = {"interpreted", "reference without a model", ""}
ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
       "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
UNI = (("—", "---"), ("–", "--"), ("−", "$-$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("→", r"$\to$"), ("×", r"$\times$"))
HEX = re.compile(r"\b[0-9a-f]{32,64}\b")


def tex(s) -> str:
    """Plain text as LaTeX; hex digests shortened to 12 characters (full values stay in the CSV)."""
    s = HEX.sub(lambda m: m.group(0)[:12] + "\x00", str(s))
    s = "".join(ESC.get(c, c) for c in s).replace("\x00", r"\ldots{}")
    for a, b in UNI:
        s = s.replace(a, b)
    return s


def pretty(name: str) -> str:
    """Arm, reference and comparison names as printed: Inception -> Inception-v3, minus -> $-$."""
    s = re.sub(r"\bInception\b(?!-v3)", "Inception-v3", name)
    for k, v in REFS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return tex(s.replace(" minus ", " − "))


def val(x) -> str:
    """A CSV value of any kind: integers as stored, small non-zero numbers in powers of ten, else 3 decimals."""
    if x is None or str(x).strip() == "":
        return "--"
    s = str(x).strip()
    if re.fullmatch(r"-?\d+", s):
        return s
    try:
        v = float(s)
    except ValueError:
        return tex(s)
    if not math.isfinite(v):
        return tex(s)
    if v == 0:
        return "0"
    if abs(v) < 1e-3:
        m, e = f"{v:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f(v)


def has(x) -> bool:
    return x is not None and str(x).strip() != ""


def cell_sd(r, sign=False) -> str:
    """Mean over draws, SD below (support-draw spread, never a CI)."""
    if not r or not has(r.get("mean")):
        return "--"
    return two(f(r["mean"], sign=sign), f"({f(r['sd'])})") if has(r.get("sd")) else f(r["mean"], sign=sign)


def cell_p(r, sign=False) -> str:
    """Mean over draws, 5th to 95th percentile below."""
    if not r or not has(r.get("mean")):
        return "--"
    if not has(r.get("p05")):
        return f(r["mean"], sign=sign)
    return two(f(r["mean"], sign=sign), f"({rng(r['p05'], r['p95'], sign=sign)})")


def cell_all(r, sign=False) -> str:
    """Mean, then (SD; 5th to 95th percentile): the K-mean column."""
    if not r or not has(r.get("mean")):
        return "--"
    parts = [f(r["sd"])] if has(r.get("sd")) else []
    if has(r.get("p05")):
        parts.append(rng(r["p05"], r["p95"], sign=sign))
    return two(f(r["mean"], sign=sign), "(" + "; ".join(parts) + ")") if parts else f(r["mean"], sign=sign)


def model_key(m: str):
    return (0, ARMS.index(m)) if m in ARMS else (1, list(REFS).index(m)) if m in REFS else (2, m)


def set_key(s: str):
    return (FSETS.index(s), s) if s in FSETS else (len(FSETS), s)


class Marks:
    """Superscript marks for statuses other than 'interpreted' (or a reference's), explained in the note."""
    SYMBOLS = ("*", r"\dagger", r"\ddagger", r"\S", r"\P", r"\|", "**", r"\dagger\dagger")

    def __init__(self):
        self.used: dict[str, str] = {}

    def __call__(self, status: str) -> str:
        if status in PLAIN:
            return ""
        if status not in self.used:
            self.used[status] = self.SYMBOLS[len(self.used) % len(self.SYMBOLS)]
        return f"$^{{{self.used[status]}}}$"

    def note(self) -> str:
        return " ".join(f"$^{{{s}}}${tex(st)}." for st, s in self.used.items())


def read_e(name: str) -> list[dict] | None:
    path = E_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_e(name: str, source: str, lines: list[str], note: str = "") -> bool:
    """<name>.tex and <name>_note.tex (the note may be empty); the first comment names the source CSV."""
    rows = read_e(source) or [{}]
    head = [f"% src: {source} in {E_DIR.as_posix()} (generated by make_tables.py; formatting only);",
            "% label: " + rows[0].get("posthoc", "")]
    write(name, head + lines)
    (OUT / f"{name}_note.tex").write_text(note + "\n", encoding="utf-8")
    return True


def longtable(spec: str, header: list[str], body: list[str]) -> list[str]:
    head = " & ".join(header) + r" \\"
    return [r"\begin{longtable}{" + spec + "}", r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head,
            r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot", *body, r"\end{longtable}"]


def grouped(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def e_labels() -> int:
    """e<n>_label.tex: the posthoc label of each experiment's CSVs (E1 without its limitations clause, which
    e1_limitations.tex prints); all distinct labels if the files disagree."""
    written = 0
    for n, files in E_FILES.items():
        names = list(files) + ([p.name for p in sorted(E_DIR.glob("E2f_marks_*.csv"))] if n == 2 else [])
        labels = []
        for name in names:
            rows = read_e(name)
            if rows:
                lab = rows[0]["posthoc"].split("; limitations: ")[0] if n == 1 else rows[0]["posthoc"]
                if lab not in labels:
                    labels.append(lab)
        if labels:
            text = tex(labels[0]) if len(labels) == 1 else "Labels differ between files: " + " | ".join(tex(x) for x in labels)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"e{n}_label.tex").write_text(text + "\n", encoding="utf-8")
            print("wrote", f"e{n}_label")
            written += 1
    return written


# --- E1 ---------------------------------------------------------------------------------------

def e1d() -> bool:
    rows = read_e("E1d_success.csv")
    if rows is None:
        return False
    stat = [r for r in rows if r["line"] != "fixed sentence" and not r["line"].startswith("limitations")]
    L = [r"% cells: n_draws, mean, sd, p05, draws_above_0, verdict as stored; mean and p05 are paired differences (signed).",
         r"\begin{tabular}{@{}p{4.4cm}lp{2.2cm}cccccl@{}}", r"\toprule",
         r"Line & Feature set & Status & $n$ & Mean & SD & p05 & Draws $>0$ & Verdict \\", r"\midrule"]
    for r in stat:
        line = pretty(r["line"].replace(", K mean", ""))
        signed = (lambda x: f(x, sign=True) if has(x) else "--")  # noqa: E731
        L.append(" & ".join([line, tex(r["feature_set"]), tex(r["status"]) or "--", val(r["n_draws"]), signed(r["mean"]),
                             val(r["sd"]), signed(r["p05"]), val(r["draws_above_0"]), tex(r["verdict"])]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    sentence = next((r for r in rows if r["line"] == "fixed sentence"), None)
    limits = next((r for r in rows if r["line"].startswith("limitations")), None)
    note = ""
    if sentence:
        note = f"Verdict on E1-M: \\textbf{{{tex(sentence['verdict'])}}}"
        if has(sentence["status"]):
            note += f" ({tex(sentence['status'])})"
        note += "." + (f" Fixed sentence: ``{tex(sentence['text'])}''" if has(sentence["text"]) else
                       " The fixed sentence is not filled: a statistic it quotes is missing.")
    write_e("e1d_success", "E1d_success.csv", L, note)
    OUT.joinpath("e1_limitations.tex").write_text((tex(limits["text"]) + "." if limits else "") + "\n", encoding="utf-8")
    return True


def nine_cells(by: dict, count_sets=NORMALS) -> list[str]:
    """False-rare count/n (rate; interval) on the three normal sets, called NORMAL, R1 and hold-out correct."""
    cells = []
    for s in count_sets:
        r = by.get(s)
        if not r:
            cells.append("--")
            continue
        iv = f"; {rng(r['ci95_low'], r['ci95_high'])}" if has(r.get("ci95_low")) else ""
        cells.append(two(f"{r['called_any_rare']}/{r['n']}", f"({f(r['rate_value'])}{iv})"))
    # E0: beside the false-rare rate, the destinations (NORMAL; CNV, DME or DRUSEN; any rare): NORMAL/common per set.
    nor = [f"{by[s]['called_NORMAL']}/{by[s]['called_CNV_DME_DRUSEN']}" if s in by else "--" for s in NORMALS]
    cells.append("; ".join(nor))
    r1 = [by[s]["called_any_rare"] if s in by else "--" for s in RARE_EXT]
    cells.append(f"{r1[0]} / {r1[1]}; {r1[2]}")
    cells.append(by["Legacy 22"]["correct_class"] if "Legacy 22" in by else "--")
    return cells


NINE_HEAD = [r"\makecell{Kermany normals\\(patient-disjoint)}", "OCTID normals", "OCTDL normals",
             r"\makecell{Called NORMAL/common disease\\Kermany; OCTID; OCTDL}", r"\makecell{R1: OCTID CSC / MH;\\OCTDL MH}",
             r"\makecell{Hold-out\\correct}"]


def e1g() -> bool:
    rows = read_e("E1g_nine_way.csv")
    if rows is None:
        return False
    marks = Marks()
    groups = grouped(rows, "model", "feature_set")
    L = [r"% cells: called_any_rare/n (rate_value; ci95_low--ci95_high) on the normal sets; called_NORMAL/called_CNV_DME_DRUSEN; R1 = called_any_rare",
         r"%        on OCTID CSC, OCTID MH, OCTDL MH; hold-out = correct_class of 'Legacy 22'.",
         r"\begin{tabular}{@{}ll" + "c" * 6 + r"@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Normal scans called any rare class: count/$n$ (rate; 95\% interval)} & & & \\",
         r"\cmidrule(lr){3-5}", "Model & Set & " + " & ".join(NINE_HEAD) + r" \\", r"\midrule"]
    last = None
    for (model, s) in sorted(groups, key=lambda k: (model_key(k[0]), set_key(k[1]))):
        g = groups[model, s]
        if last is not None and model != last:
            L.append(r"\addlinespace[2pt]")
        by = {r["query_set"]: r for r in g}
        L.append(" & ".join([pretty(model) if model != last else "", tex(s) + marks(g[0]["status"]), *nine_cells(by)]) + r" \\")
        last = model
    L += [r"\bottomrule", r"\end{tabular}"]
    return write_e("e1g_nine_way", "E1g_nine_way.csv", L, marks.note())


def e1a() -> bool:
    rows = read_e("E1a_checks.csv")
    if rows is None:
        return False
    body = [" & ".join([tex(r["check"]), pretty(r["arm"]) if r["arm"] else "", tex(r["feature_set"]), tex(r["quantity"]),
                        val(r["value"]), tex(r["status"]), tex(r["note"])]) + r" \\" for r in rows]
    L = longtable(r"@{}p{3.3cm}lp{1.5cm}p{3.6cm}p{2.2cm}p{1.6cm}p{3.2cm}@{}",
                  ["Check", "Arm", "Set", "Quantity", "Value", "Status", "Note"], body)
    return write_e("e1a_checks", "E1a_checks.csv", L, "Hashes are shortened to 12 characters; the full values are in the CSV.")


def e_kmean_table(name: str, source: str, key: str, sign: bool) -> bool:
    """E1b (key = model) or E1c (key = comparison): per K mean with SD or percentiles, K mean in full, status."""
    rows = read_e(source)
    if rows is None:
        return False
    groups = grouped(rows, key, "feature_set")
    body, last = [], None
    order = sorted(groups, key=lambda k: (model_key(k[0].split()[0]), k[0], set_key(k[1])))
    for (m, s) in order:
        g = {r["k"]: r for r in groups[m, s]}
        status = groups[m, s][0]["status"]
        if last is not None and m != last:
            body.append(r"\addlinespace[2pt]")
        per_k = [(cell_p if sign else cell_sd)(g.get(k), sign) for k in KS]
        body.append(" & ".join([pretty(m) if m != last else "", tex(s), *per_k, cell_all(g.get(KM), sign), tex(status)]) + r" \\")
        last = m
    head = ["Model" if key == "model" else "Comparison", "Set", "$K=1$", "$K=2$", "$K=4$", "$K=8$", "Mean $K$ 1--8", "Status"]
    L = longtable(r"@{}p{3.6cm}l" + "c" * 5 + r"p{2.6cm}@{}", head, body)
    return write_e(name, source, L)


def e1e() -> bool:
    rows = read_e("E1e_matched.csv")
    if rows is None:
        return False
    head_row = next((r for r in rows if r["model"] == "matched subset"), None)
    rest = [r for r in rows if r["model"] != "matched subset"]
    groups = grouped(rest, "model", "feature_set")
    body, last = [], None
    for (m, s) in sorted(groups, key=lambda k: (model_key(k[0].split()[0]), k[0], set_key(k[1]))):
        g = {r["k"]: r for r in groups[m, s]}
        sign = " minus " in m
        if last is not None and m != last:
            body.append(r"\addlinespace[2pt]")
        first = groups[m, s][0]
        body.append(" & ".join([pretty(m) if m != last else "", tex(s), *[(cell_p if sign else cell_sd)(g.get(k), sign) for k in KS],
                                cell_all(g.get(KM), sign), tex(first["classes_present"]) or "--", tex(first["status"])]) + r" \\")
        last = m
    L = longtable(r"@{}p{3.2cm}l" + "c" * 5 + r"p{2.4cm}p{2.2cm}@{}",
                  ["Model", "Set", "$K=1$", "$K=2$", "$K=4$", "$K=8$", "Mean $K$ 1--8", "Classes present", "Status"], body)
    note = (f"Matched subset: {tex(head_row['feature_set'])}, {head_row['n_queries']} development queries; supports are not matched."
            if head_row else "")
    return write_e("e1e_matched", "E1e_matched.csv", L, note)


def e1f() -> bool:
    rows = read_e("E1f_holdout.csv")
    if rows is None:
        return False
    ks = ("1", "2", "4", "8", "16")
    body = []
    for quantity, title in (("macro_recall", "Five-way macro recall"), ("mean_correct_of_22", "Mean number named correctly (of 22)"),
                            ("paired_macro_recall", "MIRAGE minus Inception-v3, paired (direction only)")):
        sel = [r for r in rows if r["quantity"] == quantity]
        if not sel:
            continue
        body.append(r"\multicolumn{10}{@{}l}{\emph{" + title + r"}} \\")
        groups = grouped(sel, "model", "feature_set")
        last = None
        for (m, s) in sorted(groups, key=lambda k: (model_key(k[0].split()[0]), k[0], set_key(k[1]))):
            g = {r["k"]: r for r in groups[m, s]}
            sign = quantity == "paired_macro_recall"
            fmt = cell_p if sign else (lambda r, _s=False: val(r["mean"]) if r else "--")
            cells = [fmt(g.get(k), sign) if sign else fmt(g.get(k)) for k in ks]
            km = cell_all(g.get(KM), sign) if quantity != "mean_correct_of_22" else "--"
            body.append(" & ".join([pretty(m) if m != last else "", tex(s), *cells, km, tex(g[KM]["direction"]) if KM in g else "",
                                    tex(groups[m, s][0]["status"])]) + r" \\")
            last = m
    L = longtable(r"@{}p{3.0cm}l" + "c" * 6 + r"lp{2.2cm}@{}",
                  ["Model", "Set", "$K=1$", "$K=2$", "$K=4$", "$K=8$", r"\makecell{$K=16$\\(not read)}", "Mean $K$ 1--8",
                   "Direction", "Status"], body)
    return write_e("e1f_holdout", "E1f_holdout.csv", L,
                   "The hold-out is read for direction only: the sign of the mean paired MIRAGE $-$ Inception-v3 difference, "
                   "per $K$ and in the mean over $K=1$--8. $K=16$ is shown, never read.")


def e1h() -> bool:
    rows = read_e("E1h_leave_one_out.csv")
    if rows is None:
        return False
    short = lambda p: tex(p.replace("data/raw/rare/", ""))  # noqa: E731
    L = [r"\begin{tabular}{@{}llrcllll@{}}", r"\toprule",
         r"Set & Status & Left out & Full & Smallest & (image) & Largest & (image) \\", r"\midrule"]
    L += [" & ".join([tex(r["feature_set"]), tex(r["status"]), r["n_left_out"], f(r["full_difference"], sign=True),
                      f(r["smallest"], sign=True), short(r["smallest_image"]), f(r["largest"], sign=True),
                      short(r["largest_image"])]) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    notes = "; ".join(f"{tex(r['feature_set'])}: {tex(r['note'])}" for r in rows if has(r["note"]))
    return write_e("e1h_leave_one_out", "E1h_leave_one_out.csv", L, notes + ("." if notes else ""))


def e1d_exclusion(r: dict) -> tuple[str, str]:
    excluded = r["excluded"].replace(" query-episode exclusions per arm", " query-episodes").replace(
        " hold-out images excluded", " images")
    return ("none" if r["exclusion"] == "locked (no exclusion)" else r["exclusion"]), excluded


def e1d_dup() -> bool:
    """Rule E1-D (21 Nov, after the E1 verdict): MIRAGE minus Inception-v3 without P10's matches. Every element the
    rule names: development n, mean, SD, p05, p95 and draws above 0 on the K mean (e1d_duplicates); hold-out mean
    paired difference per K and on the K mean, direction only, K = 16 shown and not read (e1d_holdout); the label
    (e1d_label)."""
    rows = read_e("E1D_duplicate_sensitivity.csv")
    if rows is None:
        return False
    L = [r"\begin{tabular}{@{}llp{4.2cm}lcccccc@{}}", r"\toprule",
         r"Part & Features & Exclusion & Excluded & $n$ & Mean & SD & p05 & p95 & Draws $>0$ \\", r"\midrule"]
    for r in rows:
        if r["phase"] not in ("development", "hold-out") or r["k"] != "mean_k1_2_4_8":
            continue
        exclusion, excluded = e1d_exclusion(r)
        dev = r["phase"] == "development"  # the hold-out is read for direction only: its spread is not printed
        L.append(" & ".join([tex(r["phase"]), tex(r["feature_set"]), tex(exclusion), tex(excluded),
                             val(r["n_draws"]), f(r["mean"], sign=True), f(r["sd"]) if dev else "--",
                             f(r["p05"], 4, sign=True) if dev else "--", f(r["p95"], sign=True) if dev else "--",
                             val(r["draws_above_0"]) if dev else "--"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    hold = grouped([r for r in rows if r["phase"] == "hold-out"], "feature_set", "exclusion")
    H = [r"% cells: mean (paired MIRAGE minus Inception-v3, hold-out); direction as stored; K = 16 shown, not read.",
         r"\begin{tabular}{@{}lp{4.2cm}l" + "c" * 6 + r"l@{}}", r"\toprule",
         r"Features & Exclusion & Excluded & $K=1$ & $K=2$ & $K=4$ & $K=8$ & \makecell{$K=16$\\(not read)} & Mean $K$ 1--8 & Direction \\",
         r"\midrule"]
    for (s, _x), g in sorted(hold.items(), key=lambda kv: set_key(kv[0][0])):
        by = {r["k"]: r for r in g}
        exclusion, excluded = e1d_exclusion(g[0])
        H.append(" & ".join([tex(s), tex(exclusion), tex(excluded),
                             *[f(by[k]["mean"], sign=True) if k in by else "--" for k in (*KS, "16", KM)],
                             tex(by[KM]["direction"]) if KM in by else "--"]) + r" \\")
    H += [r"\bottomrule", r"\end{tabular}"]
    write_e("e1d_holdout", "E1D_duplicate_sensitivity.csv", H,
            "Hold-out: mean paired difference over 100 draws, read for direction only; $K=16$ is shown, never read.")
    OUT.joinpath("e1d_label.tex").write_text(tex(rows[0]["posthoc"]) + "\n", encoding="utf-8")
    check = next((r for r in rows if r["phase"] == "check"), None)
    sentence = next((r for r in rows if r["phase"] == "sentence"), None)
    note = ""
    if check:
        note += (f"Check: the same code on the locked features reproduces the locked P10 rows "
                 f"(largest difference {val(check['mean'])}; {tex(check['status'])}). ")
    if sentence:
        note += f"Fixed sentence: ``{tex(sentence['note'])}''"
    return write_e("e1d_duplicates", "E1D_duplicate_sensitivity.csv", L, note)


# --- E2 ---------------------------------------------------------------------------------------

def e2_summary() -> bool:
    """Main text: destinations of normal scans (NORMAL / common disease / rare) per arm and feature set (E2a),
    and normals failed by 0-4 arms (E2b)."""
    rows = read_e("E2a_destinations.csv")
    if rows is None:
        return False
    marks = Marks()
    by = {(r["arm"], r["feature_set"], r["query_set"]): r for r in rows}
    sets = [s for s in FSETS if any(k[1] == s for k in by)]
    L = [r"% cells: group_NORMAL / group_CNV_DME_DRUSEN / group_rare per arm, feature set and normal set.",
         r"\begin{tabular}{@{}ll" + "c" * len(sets) + r"@{}}", r"\toprule",
         r" & & \multicolumn{" + str(len(sets)) + r"}{c}{Called NORMAL / common disease / rare} \\",
         r"\cmidrule(l){3-" + str(2 + len(sets)) + "}", "Normal scans & Arm & " + " & ".join(tex(s) for s in sets) + r" \\",
         r"\midrule"]
    for i, q in enumerate(NORMALS):
        n = next((r["n"] for (a, s, qq), r in by.items() if qq == q), "?")
        label = {"Kermany patient-disjoint": "Kermany (calibration)", "OCTID NORMAL": "OCTID", "OCTDL NORMAL": "OCTDL"}[q]
        for j, arm in enumerate(ARMS):
            cells = []
            for s in sets:
                r = by.get((arm, s, q))
                cells.append("--" if r is None else f"{r['group_NORMAL']} / {r['group_CNV_DME_DRUSEN']} / {r['group_rare']}"
                             + marks(r["status"]))
            L.append((f"{label} ({n})" if j == 0 else "") + f" & {NAME[arm]} & " + " & ".join(cells) + r" \\")
        if i < len(NORMALS) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    write_e("e2_destinations_summary", "E2a_destinations.csv", L, marks.note())
    over = read_e("E2b_overlap.csv")
    if over:
        fail = [r for r in over if r["quantity"].startswith("normals failed by ")]
        parsed = {}
        for r in fail:
            m = re.fullmatch(r"normals failed by (\d+) of (\d+) arms", r["quantity"])
            parsed.setdefault((r["query_set"], r["feature_set"]), {"m": int(m.group(2)), "status": r["status"]})[int(m.group(1))] = r["value"]
        marks2 = Marks()
        L = [r"% cells: E2b_overlap.csv quantity 'normals failed by n of m arms' (value).",
             r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r" & & \multicolumn{5}{c}{Normal scans failed (any class but NORMAL) by this many arms} \\", r"\cmidrule(l){3-7}",
             r"Normal scans & Set & 0 & 1 & 2 & 3 & 4 \\", r"\midrule"]
        last = None
        for (q, s) in sorted(parsed, key=lambda k: (NORMALS.index(k[0]) if k[0] in NORMALS else 9, set_key(k[1]))):
            d = parsed[q, s]
            cells = [d.get(i, "--") for i in range(5)]
            name = {"Kermany patient-disjoint": "Kermany (calibration)"}.get(q, q.replace(" NORMAL", ""))
            L.append(" & ".join([name if q != last else "", tex(s) + marks2(d["status"]) + (f" ({d['m']} arms)" if d["m"] != 4 else ""),
                                 *cells]) + r" \\")
            last = q
        L += [r"\bottomrule", r"\end{tabular}"]
        write_e("e2_overlap_summary", "E2b_overlap.csv", L, marks2.note())
    return True


def e2a_full() -> bool:
    rows = read_e("E2a_destinations.csv")
    if rows is None:
        return False
    marks = Marks()
    body, last = [], None
    for r in sorted(rows, key=lambda r: (ARMS.index(r["arm"]), set_key(r["feature_set"]),
                                         (NORMALS + RARE_EXT).index(r["query_set"]) if r["query_set"] in NORMALS + RARE_EXT else 9)):
        key = (r["arm"], r["feature_set"])
        if last is not None and key != last:
            body.append(r"\addlinespace[1pt]")
        body.append(" & ".join([NAME[r["arm"]] if key != last else "", (tex(r["feature_set"]) + marks(r["status"])) if key != last else "",
                                tex(r["query_set"]), r["n"], *[r[c] for c in NINE],
                                f"{r['group_NORMAL']} / {r['group_CNV_DME_DRUSEN']} / {r['group_rare']}"]) + r" \\")
        last = key
    L = longtable(r"@{}lll r" + "r" * 9 + "c@{}", ["Arm", "Set", "Query set", "$n$", *SHORT9, r"\makecell{NORMAL / common\\/ rare}"], body)
    return write_e("e2a_destinations", "E2a_destinations.csv", L, marks.note())


def e2b_pairs() -> bool:
    rows = read_e("E2b_overlap.csv")
    if rows is None:
        return False
    marks = Marks()
    pairs = [r for r in rows if r["arm_a"]]
    by = grouped(pairs, "query_set", "feature_set", "arm_a", "arm_b")
    body, last = [], None
    for k in sorted(by, key=lambda k: (NORMALS.index(k[0]) if k[0] in NORMALS else 9, set_key(k[1]), ARMS.index(k[2]), ARMS.index(k[3]))):
        q = {r["quantity"]: r["value"] for r in by[k]}
        if last is not None and k[:2] != last:
            body.append(r"\addlinespace[1pt]")
        body.append(" & ".join([tex(k[0]) if k[:2] != last else "", (tex(k[1]) + marks(by[k][0]["status"])) if k[:2] != last else "",
                                f"{NAME[k[2]]} / {NAME[k[3]]}", val(q.get("failure Jaccard")), val(q.get("false-rare Jaccard")),
                                val(q.get("share given the same class"))]) + r" \\")
        last = k[:2]
    L = longtable(r"@{}lllccc@{}", ["Normal scans", "Set", "Arms", "Failure Jaccard", "False-rare Jaccard", "Same class"], body)
    return write_e("e2b_pairs", "E2b_overlap.csv", L,
                   ("--- : both sets empty (no Jaccard index). " if any(r["value"] == "—" for r in pairs) else "") + marks.note())


def e2c() -> bool:
    rows = read_e("E2c_transitions.csv")
    if rows is None:
        return False
    marks = Marks()
    groups = grouped(rows, "arm", "query_set")
    order = ("NORMAL", "common", "rare")
    body, last = [], None
    for (arm, q) in sorted(groups, key=lambda k: (ARMS.index(k[0]), (NORMALS + RARE_EXT).index(k[1]) if k[1] in NORMALS + RARE_EXT else 9)):
        c = {(r["locked"], r["E1-M"]): r["count"] for r in groups[arm, q]}
        if last is not None and arm != last:
            body.append(r"\addlinespace[1pt]")
        body.append(" & ".join([NAME[arm] if arm != last else "", tex(q) + marks(groups[arm, q][0]["status"]),
                                *[c.get((a, b), "--") for a in order for b in order]]) + r" \\")
        last = arm
    letters = {"NORMAL": "N", "common": "C", "rare": "R"}
    L = longtable(r"@{}ll" + "r" * 9 + "@{}", ["Arm", "Scans", *[f"{letters[a]}$\\to${letters[b]}" for a in order for b in order]], body)
    return write_e("e2c_transitions", "E2c_transitions.csv", L,
                   "Locked $\\to$ E1-M, per scan: N, called NORMAL; C, a common disease (CNV, DME or DRUSEN); R, any rare class. " + marks.note())


E2D_NAMES = {"width": "Width (px)", "height": "Height (px)", "aspect_ratio": "Aspect ratio", "border_share": "Border share",
             "near_white_share": "Near-white share", "grey_mean": "Grey mean", "grey_sd": "Grey SD"}


def e2d_count(rows: list[dict]) -> str:
    intervals = [r for r in rows if has(r["boot95_low"]) and has(r["boot95_high"])]
    excluded = sum(r["interval_excludes_0_5"] == "True" for r in intervals)
    return f"{excluded} of {len(intervals)} reported E2d intervals exclude 0.5 (all arms, sets, feature sets and properties)."


def e2d_main() -> bool:
    """The E2d table the E2 rule fixes as the main-text table: OCDTL normals, locked features, seven properties x four arms."""
    rows = read_e("E2d_properties.csv")
    if rows is None:
        return False
    sel = {(r["arm"], r["property"]): r for r in rows if r["main_text_table"] == "True"}
    props = [p for p in E2D_NAMES if any(k[1] == p for k in sel)]
    L = [r"% cells: auroc_failed_vs_not (boot95_low--boot95_high); below: failed_median / not_failed_median.",
         r"\begin{tabular}{@{}l" + "c" * 4 + r"@{}}", r"\toprule", "Property & " + " & ".join(NAME[a] for a in ARMS) + r" \\", r"\midrule"]
    firsts = [next((r for (a, p), r in sel.items() if a == arm), None) for arm in ARMS]
    L.append(r"\makecell[l]{Failed (common / rare);\\not failed} & " + " & ".join(
        "--" if r is None else two(f"{r['n_failed']} ({r['failed_common']} / {r['failed_rare']})", r["n_not_failed"]) for r in firsts) + r" \\")
    L.append(r"\midrule")
    for p in props:
        cells = []
        for arm in ARMS:
            r = sel.get((arm, p))
            if r is None:
                cells.append("--")
                continue
            top = f(r["auroc_failed_vs_not"]) + (f" ({rng(r['boot95_low'], r['boot95_high'])})" if has(r["boot95_low"]) else "") \
                if has(r["auroc_failed_vs_not"]) else "--"
            cells.append(two(top, f"{val(r['failed_median'])} / {val(r['not_failed_median'])}"))
        L.append(E2D_NAMES[p] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return write_e("e2d_octdl_locked", "E2d_properties.csv", L, e2d_count(rows))


def e2d_all() -> bool:
    rows = read_e("E2d_properties.csv")
    if rows is None:
        return False
    marks = Marks()
    body, last = [], None
    for r in rows:
        key = (r["arm"], r["query_set"], r["feature_set"])
        if last is not None and key != last:
            body.append(r"\addlinespace[1pt]")
        q = lambda a, b, c: f"{val(r[a])} ({val(r[b])}--{val(r[c])})" if has(r[b]) else val(r[a])  # noqa: E731
        terts = " / ".join(val(r[f"failure_share_tertile_{i}"]) for i in (1, 2, 3))
        auc = (f(r["auroc_failed_vs_not"]) + (f" ({rng(r['boot95_low'], r['boot95_high'])})" if has(r["boot95_low"]) else "")
               if has(r["auroc_failed_vs_not"]) else "--")
        body.append(" & ".join([NAME[r["arm"]] if key != last else "", tex(r["query_set"]) if key != last else "",
                                (tex(r["feature_set"]) + marks(r["status"])) if key != last else "", E2D_NAMES.get(r["property"], tex(r["property"])),
                                f"{r['n_failed']} / {r['n_not_failed']}", q("failed_median", "failed_q1", "failed_q3"),
                                q("not_failed_median", "not_failed_q1", "not_failed_q3"), terts, auc]) + r" \\")
        last = key
    L = longtable(r"@{}llllccccc@{}", ["Arm", "Normals", "Set", "Property", r"\makecell{Failed /\\not failed}",
                                                     r"\makecell{Failed: median\\(q1--q3)}", r"\makecell{Not failed: median\\(q1--q3)}",
                                                     r"\makecell{Failure share\\by tertile}", r"\makecell{AUROC failed vs not\\(95\% interval)}"], body)
    return write_e("e2d_properties", "E2d_properties.csv", L, e2d_count(rows) + " " + marks.note())


def e2d_cat() -> bool:
    rows = read_e("E2d_categorical.csv")
    if rows is None:
        return False
    body = [" & ".join([NAME[r["arm"]], tex(r["query_set"]), tex(r["feature_set"]), tex(r["property"]), tex(r["value"]), r["n"],
                        r["failed"], r["failed_common"], r["failed_rare"], val(r["failure_share"])]) + r" \\" for r in rows]
    L = longtable(r"@{}lllllrrrrc@{}", ["Arm", "Normals", "Set", "Property", "Value", "$n$", "Failed", "Common", "Rare", "Failure share"], body)
    return write_e("e2d_categorical", "E2d_categorical.csv", L)


def e2e() -> bool:
    rows = read_e("E2e_octdl_patients.csv")
    if rows is None:
        return False
    marks = Marks()
    L = [r"\begin{tabular}{@{}llrcccccc@{}}", r"\toprule",
         r"Arm & Set & Patients & \makecell{All / some / no\\scans failed} & \makecell{Failure share\\per patient} & "
         r"\makecell{Common-disease\\share per patient} & \makecell{False-rare share\\per patient} & \makecell{Scans per patient:\\mean, median (min--max)} \\",
         r"\midrule"]
    for r in sorted(rows, key=lambda r: (ARMS.index(r["arm"]), set_key(r["feature_set"]))):
        L.append(" & ".join([NAME[r["arm"]], tex(r["feature_set"]) + marks(r["status"]), r["patients"],
                             f"{r['all_failed']} / {r['some_failed']} / {r['none_failed']}", val(r["failure_share_mean_per_patient"]),
                             val(r["common_disease_share_mean_per_patient"]), val(r["false_rare_share_mean_per_patient"]),
                             f"{val(r['scans_per_patient_mean'])}, {val(r['scans_per_patient_median'])} "
                             f"({r['scans_per_patient_min']}--{r['scans_per_patient_max']})"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return write_e("e2e_patients", "E2e_octdl_patients.csv", L, marks.note())


def e2f() -> int:
    n = 0
    for path in sorted(E_DIR.glob("E2f_marks_*.csv")):
        rows = read_e(path.name)
        body = [" & ".join([tex(r["source"]), tex(r["group"]), r["group_size"], tex(r["mark"]), tex(r["value"]), r["count"],
                            tex(r["note"])]) + r" \\" for r in rows]
        L = longtable(r"@{}llrllrp{4cm}@{}", ["Source", "Group", "Size", "Mark", "Value", "Count", "Note"], body)
        n += write_e("e2f_" + path.stem.replace("E2f_marks_", "").lower(), path.name, L,
                     "One rater (Lam), not a clinician; counts of each mark per group within source.")
    return n


# --- E3 ---------------------------------------------------------------------------------------

def e3_criteria_t() -> bool:
    rows = read_e("E3_criteria.csv")
    if rows is None:
        return False
    body = [" & ".join([tex(r["criterion"]), pretty(r["arm"]), tex(r["variant"]), tex(r["statistic"]), val(r["n"]),
                        val(r["value"]), tex(r["result"])]) + r" \\" for r in rows]
    L = longtable(r"@{}p{1.6cm}lp{1.5cm}p{4.2cm}rcp{5.2cm}@{}", ["Criterion", "Arm", "Variant", "Statistic", "$n$", "Value", "Result"], body)
    return write_e("e3_criteria", "E3_criteria.csv", L)


E3_ORDER = ("locked", "E3-ref", "M-crop", "M-bicubic", "M-nostretch", "M-res256", "M-pad", "R-squash", "R-pad", "R-bilinear",
            "R-stretch", "R-norm05", "R-res128", "all variants")


def e3_key(k):
    return (model_key(k[0]), E3_ORDER.index(k[1]) if k[1] in E3_ORDER else 99, k[1])


def e3_dev() -> bool:
    rows = read_e("E3_development.csv")
    if rows is None:
        return False
    groups = grouped(rows, "arm", "variant")
    body, last = [], None
    for (arm, v) in sorted(groups, key=e3_key):
        q = {(r["quantity"], r["k"]): r for r in groups[arm, v]}
        if last is not None and arm != last:
            body.append(r"\addlinespace[2pt]")
        if v == "all variants":
            r = q[("range of variant K-mean means", KM)]
            body.append(" & ".join(["", "range over variants", f"{f(r['mean'])}--{f(r['p95'])} ({r['n_draws']} variants)", "", "", ""]) + r" \\")
            continue
        status = groups[arm, v][0]["status"]
        body.append(" & ".join([NAME[arm] if arm != last else "", tex(v), cell_all(q.get(("macro_recall", KM))),
                                cell_p(q.get(("minus Inception-v3 locked", KM)), True), cell_p(q.get((f"minus {arm} locked", KM)), True),
                                tex(status)]) + r" \\")
        last = arm
    L = longtable(r"@{}llcccp{3.6cm}@{}", ["Arm", "Variant", r"\makecell{Macro recall, mean $K$ 1--8\\(SD; 5th--95th)}",
                                            r"\makecell{$-$ Inception-v3 locked\\(paired)}", r"\makecell{$-$ same arm locked\\(paired)}",
                                            "Status"], body)
    return write_e("e3_development", "E3_development.csv", L, "Per-$K$ rows are in the CSV.")


def e3_hold() -> bool:
    rows = read_e("E3_holdout.csv")
    if rows is None:
        return False
    groups = grouped(rows, "arm", "variant")
    ks = ("1", "2", "4", "8", "16", KM)
    body, last = [], None
    for (arm, v) in sorted(groups, key=e3_key):
        g = {r["k"]: r for r in groups[arm, v]}
        if last is not None and arm != last:
            body.append(r"\addlinespace[2pt]")
        body.append(" & ".join([NAME[arm] if arm != last else "", tex(v),
                                *[f(g[k]["mean_difference_vs_locked_Inception"], sign=True) if k in g else "--" for k in ks],
                                tex(g[KM]["direction"]) if KM in g else "", tex(groups[arm, v][0]["status"])]) + r" \\")
        last = arm
    L = longtable(r"@{}ll" + "c" * 6 + r"lp{3cm}@{}", ["Arm", "Variant", "$K=1$", "$K=2$", "$K=4$", "$K=8$", r"\makecell{$K=16$\\(not read)}",
                                                       "Mean $K$ 1--8", "Direction", "Status"], body)
    return write_e("e3_holdout", "E3_holdout.csv", L,
                   "Mean paired difference from locked Inception-v3 over 100 draws; read for direction only.")


def e3_nine() -> bool:
    rows = read_e("E3_nine_way.csv")
    if rows is None:
        return False
    marks = Marks()
    groups = grouped(rows, "arm", "variant")
    body, last = [], None
    for (arm, v) in sorted(groups, key=e3_key):
        if last is not None and arm != last:
            body.append(r"\addlinespace[2pt]")
        by = {r["query_set"]: r for r in groups[arm, v]}
        body.append(" & ".join([NAME[arm] if arm != last else "", tex(v) + marks(groups[arm, v][0]["status"]), *nine_cells(by)]) + r" \\")
        last = arm
    L = longtable(r"@{}ll" + "c" * 6 + "@{}", ["Arm", "Variant", *NINE_HEAD], body)
    return write_e("e3_nine_way", "E3_nine_way.csv", L, marks.note())


def e3_checks_t() -> bool:
    rows = read_e("E3_checks.csv")
    if rows is None:
        return False
    body = [" & ".join([NAME.get(r["arm"], tex(r["arm"])), tex(r["variant"]), tex(r["check"]), val(r["value"]), tex(r["status"])]) + r" \\"
            for r in rows]
    L = longtable(r"@{}llp{5.8cm}p{3.7cm}p{3.5cm}@{}", ["Arm", "Variant", "Check", "Value", "Status"], body)
    return write_e("e3_checks", "E3_checks.csv", L, "Hashes are shortened to 12 characters; the full values are in the CSV.")


# --- E4 ---------------------------------------------------------------------------------------

E4_ABBR = {"Kermany test": "K", "Yoo rare": "Y", "OCTID": "ID", "OCTDL": "DL", "Kermany test NORMAL": "KN", "OCTID NORMAL": "IDN",
           "OCTDL NORMAL": "DLN"}
E4_NOTE = ("Sources: K, Kermany test (1,000; patient units); Y, the 146 rare images of Yoo et al.\\ (image units); ID, OCTID (410; image units); "
           "DL, OCTDL (371; patient units); KN, IDN, DLN: their normal scans only. Pairs with Y: different sources, source and disease are mixed.")


def pair_abbr(p: str) -> str:
    a, b = p.split(" vs ")
    return f"{E4_ABBR.get(a, tex(a))}--{E4_ABBR.get(b, tex(b))}"


def e4a() -> bool:
    rows = read_e("E4a_fewshot_source_test.csv")
    if rows is None:
        return False
    marks = Marks()
    groups = grouped(rows, "pair", "feature_set")
    body, last = [], None
    for (p, s) in groups:
        g = {r["k"]: r for r in groups[p, s]}
        r = g.get(KM) or groups[p, s][0]
        if last is not None and p != last:
            body.append(r"\addlinespace[2pt]")
        paired = " minus " in s
        body.append(" & ".join([pair_abbr(p) if p != last else "", pretty(s) + marks(r["status"]),
                                *[(cell_p if paired else cell_sd)(g.get(k), paired) for k in KS], cell_all(g.get(KM), paired),
                                tex(r["reading"])]) + r" \\")
        last = p
    L = longtable(r"@{}llcccccp{3cm}@{}", ["Pair", "Features", "$K=1$", "$K=2$", "$K=4$", "$K=8$", "Mean $K$ 1--8", "Reading"], body)
    return write_e("e4a_fewshot", "E4a_fewshot_source_test.csv", L,
                   "Two-way macro recall of source against source (chance 0.50), 100 draws; paired rows: E1-M minus locked on the same draw. "
                   + E4_NOTE + " " + marks.note())


def e4_matrix(name: str, source: str, rowkey, cell, title: str, note: str) -> bool:
    rows = read_e(source)
    if rows is None:
        return False
    pairs = list(dict.fromkeys(r["pair"] for r in rows))
    keys = list(dict.fromkeys(rowkey(r) for r in rows))
    by = {(rowkey(r), r["pair"]): r for r in rows}
    L = [r"\begin{tabular}{@{}l" + "c" * len(pairs) + r"@{}}", r"\toprule", title + " & " + " & ".join(pair_abbr(p) for p in pairs) + r" \\",
         r"\midrule"]
    for k in keys:
        L.append(k[1] + " & " + " & ".join(cell(by.get((k, p))) for p in pairs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return write_e(name, source, L, note)


def e4b() -> bool:
    return e4_matrix("e4b_columns", "E4b_column_auroc.csv",
                     lambda r: (r["feature_set"], f"{tex(r['feature_set'])}: {tex(r['column'])}"),
                     lambda r: "--" if r is None else f(r["separability"]),
                     "Column", "Separability of one column with no fit, max(AUROC, 1 $-$ AUROC); reading: $\\geq 0.95$ near-perfectly, "
                     "0.80--0.95 strongly, 0.60--0.80 partly, below 0.60 hardly separable. " + E4_NOTE)


def e4c() -> bool:
    def cell(r):
        if r is None:
            return "--"
        flag = "$^{*}$" if has(r["fits_not_converged"]) and r["fits_not_converged"] not in ("0", "0.0") else ""
        mark = "" if r["status"] in PLAIN else r"$^{\dagger}$"
        return two(f(r["auroc"]) + flag + mark, f"({rng(r['boot95_low'], r['boot95_high'])})")
    return e4_matrix("e4c_probe", "E4c_probe_auroc.csv", lambda r: (r["feature_set"], pretty(r["feature_set"])), cell,
                     "Features", "Logistic probe, out-of-fold AUROC pooled over five grouped folds (95\\% interval from 2,000 resamples "
                     "of units within each source, without refitting). $^{*}$Some fits did not converge (count in the CSV). "
                     "$^{\\dagger}$Status other than interpreted (in the CSV). " + E4_NOTE)


def e4d() -> bool:
    code = {"the control reduced separability": "r", "no reduction shown": "n", "at ceiling; the difference is not read": "c",
            "descriptive comparison": "d"}

    def cell(r):
        if r is None:
            return "--"
        c = code.get(r["reading"], "")
        return two(f(r["auroc_difference"], sign=True) + (f"$^{{{c}}}$" if c else ""),
                   f"({rng(r['boot95_low'], r['boot95_high'], sign=True)})")
    return e4_matrix("e4d_probe_diff", "E4d_probe_differences.csv", lambda r: (r["comparison"], pretty(r["comparison"])), cell,
                     "Comparison", "Paired AUROC differences on the same resamples. Reading: $^{r}$the control reduced separability "
                     "(interval wholly below 0); $^{n}$no reduction shown; $^{c}$at ceiling, the difference is not read; "
                     "$^{d}$descriptive comparison. " + E4_NOTE)


def tasks_summary() -> bool:
    """Every K-shot task on which all four arms were scored, side by side: mean macro recall over K = 1, 2, 4, 8, with
    the locked inputs and (where it exists) the main control E1-M, and the references that use no model. No new
    number: each cell is the K-mean of a CSV of record. No rank and no bold: the E1 rule allows no new ranking."""
    e1b, e1f = read_e("E1b_development.csv"), read_e("E1f_holdout.csv")
    if e1b is None or e1f is None:
        return False
    dev = {(r["model"], r["feature_set"]): r["mean"] for r in e1b if r["k"] == KM}
    hold = {(r["model"], r["feature_set"]): r["mean"] for r in e1f if r["k"] == KM and r["quantity"] == "macro_recall"}
    r2 = {r["arm"]: r["mean"] for r in read_ph("R2_development_spread.csv") if r["k"] == KM}
    for arm in ARMS:  # the E1 files' locked rows must be the locked development means (rule R2's gate value)
        if abs(float(dev[arm, "locked"]) - float(r2[arm])) > 1e-9:
            raise SystemExit(f"E1b_development.csv: locked {arm} differs from R2_development_spread.csv")
    p8 = {r["arm"]: r for r in read_ph("P8_csc_vs_mh_development.csv") if r["k"] == KM}

    def line(task, inputs, fixed, cells, size, thumb, chance):
        return " & ".join([task, inputs, fixed, *[f(c) for c in cells], f(size) if size else "--",
                           f(thumb) if thumb else "--", chance]) + r" \\"

    def five(src, s):
        return [src[a, s] for a in ARMS]

    L = [r"% src: PH E1b_development.csv and E1f_holdout.csv, k = mean_k1_2_4_8 (feature sets locked and E1-M; references",
         r"%      size_only raw, thumbnail raw and E1-M); the locked development cells equal R2_development_spread.csv (checked);",
         r"%      PH P8_csc_vs_mh_development.csv mean line (mean; locked_OCTID_two_way_macro_mean = Table 4);",
         r"%      OCTID size-only 0.500: ARC reports/external/external_two_way.csv size_only_reference (Table 4).",
         r"%      Generated by make_tables.py; formatting only, no new number.",
         r"\begin{tabular}{@{}lllccccccc@{}}", r"\toprule",
         r"Task & Inputs & How the row was fixed & " + " & ".join(NAME[a] for a in ARMS)
         + r" & Size-only & Thumbnail$^{*}$ & Chance \\", r"\midrule",
         line("Development, five-way (124 queries)", "locked", "registered (primary)", five(dev, "locked"),
              dev["size_only", "raw"], dev["thumbnail", "raw"], "0.20"),
         line("", "E1-M", "post hoc, E1 (rule 20 Nov)", five(dev, "E1-M"), None, dev["thumbnail", "E1-M"], "0.20"),
         r"\addlinespace[2pt]",
         line("Hold-out, five-way (22 queries)", "locked", "registered (secondary; not blind)", five(hold, "locked"),
              hold["size_only", "raw"], hold["thumbnail", "raw"], "0.20"),
         line("", "E1-M", "post hoc, E1 (rule 20 Nov)", five(hold, "E1-M"), None, hold["thumbnail", "E1-M"], "0.20"),
         r"\addlinespace[2pt]",
         line("Development, CSC against MH", "locked", "post hoc, P8 (rule 16 Nov)", [p8[a]["mean"] for a in ARMS],
              p8["size_only"]["mean"], None, "0.50"),
         line("OCTID, CSC against MH", "locked", "registered (exploratory)",
              [p8[a]["locked_OCTID_two_way_macro_mean"] for a in ARMS], "0.5", None, "0.50"),
         r"\bottomrule", r"\end{tabular}"]
    write("e1_tasks_summary", L)
    return True


E_GENERATORS = (e1d, e1g, e1a, lambda: e_kmean_table("e1b_development", "E1b_development.csv", "model", False),
                lambda: e_kmean_table("e1c_paired", "E1c_paired.csv", "comparison", True), e1e, e1f, e1h, e1d_dup,
                e2_summary, e2a_full, e2b_pairs, e2c, e2d_main, e2d_all, e2d_cat, e2e, e2f,
                e3_criteria_t, e3_dev, e3_hold, e3_nine, e3_checks_t, e4a, e4b, e4c, e4d, tasks_summary)


def e_guard() -> None:
    """Never write stand-in or test tables into tables/."""
    out, tables = OUT.resolve(), TABLES.resolve()
    if out != tables and not out.is_relative_to(tables):
        return
    if E_DIR.resolve() != PH.resolve():
        raise SystemExit("refusing to write E tables from another --e-dir into tables/; give a scratch --out")
    for name in (n for files in E_FILES.values() for n in files):
        rows = read_e(name)
        if rows and "dry run" in rows[0].get("posthoc", ""):
            raise SystemExit(f"refusing: {name} is a dry run (stand-in features); give a scratch --out")


def e_tables() -> None:
    e_guard()
    written = e_labels()
    for gen in E_GENERATORS:
        written += int(gen() or 0)
    if written:
        print(f"E1-E4: {written} generated files from {E_DIR.as_posix()}")


def main() -> None:
    global OUT, E_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=TABLES, help="folder for the .tex tables (default: tables/)")
    ap.add_argument("--e-dir", type=Path, default=PH, help="folder holding the E1-E4 CSVs (default: results/fm_v1/posthoc)")
    ap.add_argument("--e-only", action="store_true", help="write only the E1-E4 tables (no locked archive needed)")
    args = ap.parse_args()
    OUT, E_DIR = args.out.resolve(), args.e_dir.resolve()
    if not args.e_only:
        locked_and_posthoc()
    e_tables()


def locked_and_posthoc() -> None:
    with zipfile.ZipFile(ARC) as z:
        t_development(z)
        t_frr(z)
        t_holdout(z)
        t_octid(z)
        s_geometry(z)
        s_logistic(z)
        s_transfer(z)
        s_perclass(z)
        s_confusion(z)
        s_two_way(z)
    t_p4()
    p4_entries()
    t_p8()
    p0()
    p1()
    p2()
    p3()
    p5()
    p6()
    p9()
    g1()
    p10_sensitivity()
    p10_pairs()


if __name__ == "__main__":
    main()
