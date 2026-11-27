"""Geometry only (padding.csv, roster, file names). No model feature or prediction is read."""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

base = Path(sys.argv[1])  # scratch fable_reb
pad_m = Path(sys.argv[2])  # MIRAGE padding.csv
pad_i = Path(sys.argv[3])  # Inception padding.csv


def read(p):
    with open(p, newline="", encoding="utf-8") as s:
        return list(csv.DictReader(s))


pm, pi = read(pad_m), read(pad_i)
print("rows", len(pm), len(pi))
same_geom = all((a["path"], a["width"], a["height"], a["side"], a["left"], a["top"]) ==
                (b["path"], b["width"], b["height"], b["side"], b["left"], b["top"]) for a, b in zip(pm, pi))
print("geometry identical MIRAGE vs Inception:", same_geom)
print("modes MIRAGE", Counter(r["mode"] for r in pm), "Inception", Counter(r["mode"] for r in pi))
print("padded_sha identical rows:", sum(a["padded_sha256"] == b["padded_sha256"] for a, b in zip(pm, pi)))
g = {r["path"]: (int(r["width"]), int(r["height"]), int(r["side"])) for r in pm}
share = {p: 1 - w * h / (s * s) for p, (w, h, s) in g.items()}
roster = read(base / "lock" / "nine_way_roster.csv")
ext = read(base / "ext" / "external_queries.csv")


def pid(path):
    return Path(path).name.split("-")[1]


def size(p):
    w, h, _ = g[p]
    return f"{w}x{h}"


def summ(paths):
    v = sorted(share[p] for p in paths)
    n = len(v)
    return (f"n={n} median={median(v):.3f} min={v[0]:.3f} max={v[-1]:.3f} "
            f"<0.05:{sum(x < 0.05 for x in v)} <0.10:{sum(x < 0.10 for x in v)} >=0.25:{sum(x >= 0.25 for x in v)} "
            f">=0.30:{sum(x >= 0.30 for x in v)} >=0.60:{sum(x >= 0.60 for x in v)}")


COMMON = ("NORMAL", "CNV", "DME", "DRUSEN")
RARE = ("CSC", "MH", "MacTel", "Stargardt", "RP")
print("\n== padding share by role/label")
for role in ("fitting", "query"):
    for lab in COMMON + RARE:
        ps = [r["path"] for r in roster if r["role"] == role and r["label"] == lab]
        if ps:
            print(role, lab, summ(ps))
rare_all = [r["path"] for r in roster if r["label"] in RARE]
print("rare all", summ(rare_all))
for src in ("OCTID", "OCTDL"):
    for lab in ("NORMAL", "CSC", "MH"):
        ps = [r["path"] for r in ext if r["source"] == src and r["label"] == lab]
        if ps:
            print(src, lab, summ(ps), "sizes", Counter(size(p) for p in ps).most_common(4))

print("\n== Kermany sizes by role/label")
for role in ("fitting", "query"):
    for lab in COMMON:
        ps = [r["path"] for r in roster if r["role"] == role and r["label"] == lab]
        print(role, lab, dict(Counter(size(p) for p in ps).most_common()))

fit = [r for r in roster if r["role"] == "fitting" and r["label"] in COMMON]
fit_pids = {pid(r["path"]) for r in fit}
fit_pid_size = defaultdict(set)
fit_pid_lab_size = defaultdict(set)
for r in fit:
    fit_pid_size[pid(r["path"])].add(size(r["path"]))
    fit_pid_lab_size[pid(r["path"]), r["label"]].add(size(r["path"]))
q = [r for r in roster if r["role"] == "query" and r["label"] in COMMON]
print("\n== Kermany test: geometry group x class; patient overlap with fitting")
for sz in ("512x496", "768x496", "other"):
    for lab in COMMON:
        ps = [r["path"] for r in q if r["label"] == lab and (size(r["path"]) == sz or (sz == "other" and size(r["path"]) not in ("512x496", "768x496")))]
        if not ps:
            continue
        pids = {pid(p) for p in ps}
        ov = [p for p in ps if pid(p) in fit_pids]
        ov_same = [p for p in ps if size(p) in fit_pid_size.get(pid(p), ())]
        ov_same_lab = [p for p in ps if size(p) in fit_pid_lab_size.get((pid(p), lab), ())]
        print(f"{sz} {lab}: images={len(ps)} patients={len(pids)} pid_in_fitting={len(ov)} "
              f"pid_in_fitting_same_size={len(ov_same)} same_size_and_class={len(ov_same_lab)} "
              f"other_sizes={dict(Counter(size(p) for p in ps)) if sz == 'other' else ''}")
normals = [r["path"] for r in q if r["label"] == "NORMAL"]
disjoint = [p for p in normals if pid(p) not in fit_pids]
print("\n== 169 patient-disjoint Kermany normals:", len(disjoint), dict(Counter(size(p) for p in disjoint)),
      "patients", len({pid(p) for p in disjoint}))
print("   padding share:", summ(disjoint))
print("   wide (>=0.25) patient-disjoint normals:", sum(share[p] >= 0.25 for p in disjoint),
      "patients", len({pid(p) for p in disjoint if share[p] >= 0.25}))
# is width a patient-level property?
by_pid = defaultdict(set)
for r in roster:
    if r["label"] in COMMON:
        by_pid[pid(r["path"])].add(size(r["path"]))
print("\n== Kermany patients (roster) with one size only:", sum(len(v) == 1 for v in by_pid.values()), "of", len(by_pid))
print("\n== rare per class size summary")
for lab in RARE:
    ps = [r["path"] for r in roster if r["label"] == lab]
    ar = sorted(g[p][0] / g[p][1] for p in ps)
    print(lab, len(ps), f"w/h median {median(ar):.2f} min {ar[0]:.2f} max {ar[-1]:.2f}", summ(ps))
