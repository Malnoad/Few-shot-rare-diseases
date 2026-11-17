"""Paired prototype vs logistic macro recall from the locked results archive (post hoc, read-only)."""
import csv, io, re, sys, zipfile
from collections import defaultdict
import numpy as np

CLASSES = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
pat = re.compile(r"runs/([^/]+)/(development|reporting)/(prototype|logistic)/fold(\d)_k(\d+)_draw(\d+)/predictions\.csv$")

def macro(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["true_label"]].append(r["true_label"] == r["predicted_label"])
    return float(np.mean([np.mean(by[c]) for c in CLASSES if by[c]]))

# development: pooled out-of-fold per draw (as the nomination does); reporting: per draw
store = defaultdict(lambda: defaultdict(list))  # (arm, role, clf, k, draw) -> rows
for n in names:
    m = pat.match(n)
    if not m:
        continue
    arm, role, clf, fold, k, draw = m.groups()
    if clf == "prototype" and int(draw) >= 25:
        continue  # pair with the logistic's first 25 draws
    if role == "development" and int(k) != 4:
        continue
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8")))
    store[(arm, role, int(k))][(clf, int(draw))].extend(rows)

print("arm role K | prototype logistic diff(mean, p05, p95) [25 paired draws]")
for key in sorted(store):
    arm, role, k = key
    d = store[key]
    p = np.array([macro(d[("prototype", i)]) for i in range(25) if ("prototype", i) in d])
    l = np.array([macro(d[("logistic", i)]) for i in range(25) if ("logistic", i) in d])
    if len(p) != len(l) or not len(p):
        print(key, "unpaired", len(p), len(l)); continue
    diff = l - p
    print(f"{arm:9s} {role:11s} K={k:2d} | {p.mean():.3f} {l.mean():.3f} {diff.mean():+.3f} ({np.percentile(diff,5):+.3f}, {np.percentile(diff,95):+.3f}) SD proto {p.std(ddof=1):.3f} logit {l.std(ddof=1):.3f}")
