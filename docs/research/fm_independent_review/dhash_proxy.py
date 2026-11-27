"""Read-only proxy: how many dHash<=4 flags does the protocol screen raise between unrelated Kermany images?"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
W = "C:/Users/ThinkBook/.t3/worktrees/Few-shot/t3-03f3a2c4"; ROOT = "D:/Few-shot/"
sys.path.insert(0, W)
import fm_core as fm, fm_prepare as prep
rows = prep.load_manifest(Path(W))
pid = {r["path"]: r["pid"] for r in rows}
rare_dev = sorted(r["path"] for r in rows if r["label"] in fm.RARE and r["split"] == "train")
roster = prep.nine_way_roster(rows, rare_dev)
h = {}
for path, label, role in roster:
    with Image.open(ROOT + path) as im: h[path] = fm.dhash64(im)
def bits(paths): return np.array([[(h[p] >> i) & 1 for i in range(64)] for p in paths], dtype=np.int16)
def flags(A, B, same):
    a, b = bits(A), bits(B)
    d = a @ (1 - b).T + (1 - a) @ b.T
    n = diff = 0; rowsA = set()
    for i, j in zip(*np.nonzero(d <= 4)):
        if same and j <= i: continue
        n += 1; rowsA.add(i)
        if pid[A[i]] != pid[B[j]]: diff += 1
    return n, diff, len(rowsA)
sel = lambda lab, role: [p for p, l, r in roster if l == lab and r == role]
nq, nf = sel("NORMAL", "query"), sel("NORMAL", "fitting")
allfit = [p for p, l, r in roster if r == "fitting"]
print("NORMAL test(250) x NORMAL fitting(1000): flagged pairs, different-patient pairs, test images touched:", flags(nq, nf, False))
print("NORMAL test(250) x ALL fitting(4124):", flags(nq, allfit, False))
print("NORMAL test x NORMAL test:", flags(nq, nq, True))
for c in fm.RARE:
    print(c, "rare dev x all fitting:", flags(sel(c, "fitting"), allfit, False)[:2], "(includes self pairs)")
