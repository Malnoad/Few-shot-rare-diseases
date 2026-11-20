"""Write the LaTeX tables of the v1 paper from the locked archive and the post hoc CSVs.

Formatting only: every cell is read from a file named in the table's source comment; nothing is
computed on images, features or per-episode outputs. Sums appear only where the drafts already
state them (hold-out totals of nine-way counts per arm).

Rounding (agent default, open to Lam's objection, protocols.md 19 Nov "after the draft review"):
three decimals, half up (away from zero) from the shortest decimal form of the stored value,
taken as its 12-significant-digit form, so 0.7264999999999999 -> 0.7265 -> 0.727.

    D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_v1_paper/make_tables.py
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PH = REPO / "results" / "fm_v1" / "posthoc"
ATTEMPT = Path("D:/Few-shot/drive/fm_v1/final/final-20261113T170804Z")
ARC = ATTEMPT / "evaluation" / "results-20261114T093033Z.zip"
OUT = HERE / "tables"

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
    OUT.mkdir(exist_ok=True)
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


def p9():
    rows = read_ph("P9_rare_vs_normal_auroc.csv")
    by = {(r["arm"], r["set"]): r for r in rows}
    sets = list(dict.fromkeys(r["set"] for r in rows))
    L = [r"% src: PH P9_rare_vs_normal_auroc.csv lines 2-21 (auroc, boot95_low, boot95_high); label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}l" + "c" * 4 + r"@{}}", r"\toprule", "Set & " + " & ".join(NAME[a] for a in ARMS) + r" \\", r"\midrule"]
    for s in sets:
        lab = s.replace("Legacy 22 against 169 patient-disjoint Kermany normals", r"\makecell[l]{Hold-out 22 vs 169 Kermany\\normals (source and disease mixed)}")
        L.append(lab + " & " + " & ".join(f"{f(by[(a, s)]['auroc'])} ({rng(by[(a, s)]['boot95_low'], by[(a, s)]['boot95_high'])})" for a in ARMS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("p9_auroc", L)


def g1():
    rows = read_ph("G1_pad_to_square.csv")
    by = {(r["arm"], r["table"], r["k"], r["quantity"]): r for r in rows}
    L = [r"% src: PH G1_pad_to_square.csv columns locked, padded, size_only (development lines 6-10/54-58; hold-out 11-21/59-69);",
         r"%      K = 16 omitted from any reading (protocols.md 19 Nov G1 decision). Label: " + rows[0]["posthoc"],
         r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{MIRAGE} & \multicolumn{3}{c}{Inception-v3} \\ \cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"Set & $K$ & Locked & Padded & Change & Locked & Padded & Change \\", r"\midrule"]
    for t, tl in (("development", "Development"), ("hold-out", "Hold-out")):
        for k in ("1", "2", "4", "8", "mean_k1_2_4_8"):
            cells = []
            for a in ("MIRAGE", "Inception"):
                r = by[(a, t, k, "macro_recall_mean")]
                cells += [f(r["locked"]), f(r["padded"]), f(float(r["padded"]) - float(r["locked"]), sign=True)]
            L.append((tl if k == "1" else "") + f" & {'1--8' if k.startswith('mean') else k} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L[-1] = r"\midrule"
    L.append(r" & & \multicolumn{6}{c}{Nine-way: scans called NORMAL / called any rare (locked $\to$ padded)} \\")
    for s in ("Kermany patient-disjoint (n=169)", "OCTID NORMAL (n=206)", "OCTDL NORMAL (n=332)"):
        cells = []
        for a in ("MIRAGE", "Inception"):
            n = by[(a, f"nine-way {s}", "", "called_NORMAL")]
            r = by[(a, f"nine-way {s}", "", "called_any_rare")]
            cells.append(r"\multicolumn{3}{c}{" + f"{n['locked']} $\\to$ {n['padded']} / {r['locked']} $\\to$ {r['padded']}" + "}")
        L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{{s.replace('(n=', '(').replace(')', ')')}}} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("g1_pad_to_square", L)


def main() -> None:
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
    t_p8()
    p0()
    p1()
    p2()
    p3()
    p5()
    p6()
    p9()
    g1()


if __name__ == "__main__":
    main()
