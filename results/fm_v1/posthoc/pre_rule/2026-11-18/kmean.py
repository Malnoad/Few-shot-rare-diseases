import pickle, numpy as np, pandas as pd
D = pickle.load(open("cache.pkl","rb")); RARE = ["CSC","MH","MacTel","Stargardt","RP"]
rng = np.random.default_rng(1)
def zf(*S): return sum((s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True) for s in S).argmax(1)
def mat(k, draws):
    rows = []
    for d in draws:
        for f in range(5):
            a = D[("MIRAGE","development","prototype",f,k,d)]; b = D[("DINOv2","development","prototype",f,k,d)]
            e = (np.array(RARE)[zf(a[3], b[3])] == a[1]).astype(int)
            rows += [(q, y, d, ei - ci) for q, y, ei, ci in zip(a[0], a[1], e, a[2])]
    t = pd.DataFrame(rows, columns=["q","y","draw","d"])
    return t.pivot_table(index=["y","q"], columns="draw", values="d")
def se(M, lab, ndraw, B=3000):
    idx = [np.where(lab==c)[0] for c in RARE]; out = []
    for _ in range(B):
        dd = rng.integers(0, M.shape[1], ndraw)
        out.append(np.mean([M[rng.choice(i, len(i))][:, dd].mean() for i in idx]))
    return np.std(out, ddof=1)
P = {k: mat(k, range(100)) for k in (1,2,4,8)}
lab = P[8].index.get_level_values(0).values
Mk = np.mean([P[k].values for k in (1,2,4,8)], axis=0)
print("K-mean diff", np.mean([Mk[lab==c].mean() for c in RARE]))
for n in (5, 10, 25, 100): print("K-mean 2-way SE, draws", n, round(se(Mk, lab, n), 4))
for n in (5, 10, 25, 100): print("K=8 2-way SE, draws", n, round(se(P[8].values, lab, n), 4))
# per-image SD of the mean difference at K=8, per class -> images needed
im = P[8].values.mean(1)
for c in RARE: print(c, "n", (lab==c).sum(), "per-image SD of draw-mean diff", round(im[lab==c].std(ddof=1), 3))
v = np.mean([im[lab==c].var(ddof=1) for c in RARE])
print("mean within-class var", v, "-> SE with n per class: n=25", np.sqrt(v/25/5*1), "need n per class for SE .0107:", v/5/0.0107**2)
