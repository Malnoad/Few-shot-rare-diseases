import json, sys
import numpy as np
o = json.load(open(sys.argv[1])); RARE = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
dev = [x for x in o if x["label"] in RARE and x["role"] == "fitting"]; test = [x for x in o if x["label"] in RARE and x["role"] == "query"]
X = np.array([[x["w"], x["h"]] for x in dev], float); y = np.array([RARE.index(x["label"]) for x in dev])
hit = np.zeros(5); tot = np.zeros(5)
for i in range(len(X)):
    keep = np.arange(len(X)) != i
    c = np.array([X[keep & (y == k)].mean(0) for k in range(5)]); p = np.argmin(((c - X[i]) ** 2).sum(1)); tot[y[i]] += 1; hit[y[i]] += p == y[i]
print("LOO raw-pixel (width,height) Euclidean nearest centroid, 124 dev:", hit.astype(int), "/", tot.astype(int), "macro", round(float(np.mean(hit / tot)), 3))
c = np.array([X[y == k].mean(0) for k in range(5)])
T = np.array([[x["w"], x["h"]] for x in test], float); ty = np.array([RARE.index(x["label"]) for x in test])
p = np.argmin(((c[None] - T[:, None]) ** 2).sum(2), 1)
print("all-dev centroids -> legacy 22:", [int(((p == k) & (ty == k)).sum()) for k in range(5)], "/", [int((ty == k).sum()) for k in range(5)],
      "macro", round(float(np.mean([((p == k) & (ty == k)).sum() / (ty == k).sum() for k in range(5)])), 3))
rng = np.random.default_rng(0)
for K in (1, 2, 4, 8):
    ms = []
    for _ in range(200):
        c = np.array([X[rng.choice(np.flatnonzero(y == k), K, replace=False)].mean(0) for k in range(5)])
        p = np.argmin(((c[None] - T[:, None]) ** 2).sum(2), 1)
        ms.append(np.mean([((p == k) & (ty == k)).sum() / (ty == k).sum() for k in range(5)]))
    print(f"K={K}: random-support (illustrative, not the locked episodes) macro recall on legacy 22: mean {np.mean(ms):.3f}")
