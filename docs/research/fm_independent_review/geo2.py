import json, sys
from collections import Counter, defaultdict
import numpy as np
o = json.load(open(sys.argv[1]))
RARE = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
print("Kermany share width>512 by class/role:", {(c, r): round(np.mean([x["w"] > 512 for x in o if x["label"] == c and x["role"] == r]), 3)
      for c in ["NORMAL", "CNV", "DME", "DRUSEN"] for r in ["fitting", "query"]})
dev = [x for x in o if x["label"] in RARE and x["role"] == "fitting"]
X = np.array([[x["w"], x["h"], x["pad8"]] for x in dev], float); y = np.array([RARE.index(x["label"]) for x in dev])
Z = (X - X.mean(0)) / X.std(0)
for name, cols in (("width,height", [0, 1]), ("width,height,pad8", [0, 1, 2]), ("pad8 only", [2])):
    hit = np.zeros(5); tot = np.zeros(5)
    for i in range(len(Z)):
        keep = np.arange(len(Z)) != i
        cents = np.array([Z[keep & (y == c)][:, cols].mean(0) for c in range(5)])
        pred = np.argmin(((cents - Z[i, cols]) ** 2).sum(1)); tot[y[i]] += 1; hit[y[i]] += pred == y[i]
    print(f"leave-one-out nearest-centroid on {name}: per-class recall", np.round(hit / tot, 2), "macro", round(float(np.mean(hit / tot)), 3), "(chance 0.20)")
# rare vs Kermany by black-border share
r = np.array([x["pad8"] for x in o if x["label"] in RARE]); k = np.array([x["pad8"] for x in o if x["label"] not in RARE])
thr = 0.1; print("pad8>0.1: rare", round(float((r > thr).mean()), 3), "Kermany", round(float((k > thr).mean()), 3))
print("rare by class share hi<255 (dev):", {c: round(np.mean([x["hi"] < 255 for x in dev if x["label"] == c]), 2) for c in RARE})
