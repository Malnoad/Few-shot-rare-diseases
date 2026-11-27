"""Read-only: rebuild the nine-way roster in memory, check its hash, patient overlaps (A-5)."""
import sys, csv, hashlib, io
from collections import Counter, defaultdict
W = "C:/Users/ThinkBook/.t3/worktrees/Few-shot/t3-03f3a2c4"
sys.path.insert(0, W)
import fm_core as fm, fm_prepare as prep
from pathlib import Path
rows = prep.load_manifest(Path(W))
rare_dev = sorted(r["path"] for r in rows if r["label"] in fm.RARE and r["split"] == "train")
roster = prep.nine_way_roster(rows, rare_dev)  # no exclusions assumed (unverified vs. real freeze)
buf = io.StringIO(newline=""); w = csv.writer(buf, lineterminator="\n"); w.writerow(["path","label","role"]); w.writerows(roster)
print("roster rows", len(roster), "sha256", hashlib.sha256(buf.getvalue().encode()).hexdigest())
pid = {r["path"]: r["pid"] for r in rows}
fit = [r for r in roster if r[2] == "fitting" and r[1] in prep.COMMON]
q = [r for r in roster if r[2] == "query" and r[1] in prep.COMMON]
fit_pid_all = {pid[r[0]] for r in fit}
fit_pid_by = defaultdict(set)
for r in fit: fit_pid_by[r[1]].add(pid[r[0]])
for c in prep.COMMON:
    qs = [r for r in q if r[1] == c]
    same = sum(pid[r[0]] in fit_pid_by[c] for r in qs)
    anyc = sum(pid[r[0]] in fit_pid_all for r in qs)
    clean = [r for r in qs if pid[r[0]] not in fit_pid_all]
    print(c, "test n", len(qs), "patients", len({pid[r[0]] for r in qs}), "| overlap same-class fit", same, "| overlap any-class fit", anyc,
          "| disjoint imgs", len(clean), "patients", len({pid[r[0]] for r in clean}))
# overlap with the WHOLE train split (not just the 1000 cap)
train_pid = defaultdict(set)
for r in rows:
    if r["split"] in ("train",) and r["label"] in prep.COMMON: train_pid[r["label"]].add(r["pid"])
allp = set().union(*train_pid.values())
for c in prep.COMMON:
    qs = [r for r in q if r[1] == c]
    print(c, "test imgs sharing a patient with ANY manifest train row:", sum(pid[r[0]] in allp for r in qs))
# cross-class pid sharing inside test NORMAL vs fitting other classes
nq = [r for r in q if r[1] == "NORMAL"]
print("NORMAL test overlap by fitting class:", {c: sum(pid[r[0]] in fit_pid_by[c] for r in nq) for c in prep.COMMON})
print("rare pids unique per image:", len({pid[p] for p in rare_dev}), "of", len(rare_dev))
print("first five sorted rare dev:", rare_dev[:5])
print("timing inventory class mix:", Counter(p.split("/")[-2] for p in [rare_dev[i % len(rare_dev)] for i in range(100)]))
print("sorted class order:", list(dict.fromkeys(p.split("/")[-2] for p in rare_dev)))
