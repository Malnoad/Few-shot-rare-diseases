import zipfile, io, re, sys, pickle, os
import numpy as np, pandas as pd
Z = "D:/Few-shot/drive/fm_v1/final/final-20261113T170804Z/evaluation/results-20261114T093033Z.zip"
RARE = ["CSC","MH","MacTel","Stargardt","RP"]
CACHE = "cache.pkl"
def load():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE,"rb"))
    out = {}
    pat = re.compile(r"runs/(\w+)/(development|reporting)/(prototype|logistic)/fold(\d)_k(\d+)_draw(\d+)/predictions.csv")
    with zipfile.ZipFile(Z) as z:
        for n in z.namelist():
            m = pat.fullmatch(n)
            if not m or m[1] == "RETFound": continue
            df = pd.read_csv(io.BytesIO(z.read(n)))
            sc = [c for c in df.columns if c.split("_")[-1] in RARE and c not in ("true_label","predicted_label")]
            out[(m[1],m[2],m[3],int(m[4]),int(m[5]),int(m[6]))] = (df.query_id.values, df.true_label.values, df.correct.values.astype(int), df[sc].values, sc)
    pickle.dump(out, open(CACHE,"wb"))
    return out
D = load()
print("cells", len(D)); k0 = next(iter(D)); print(k0, D[k0][4])
def zf(*S):
    t = 0
    for s in S:
        t = t + (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)
    return t.argmax(1)
def macro(correct, labels):
    return np.mean([correct[labels == c].mean() for c in RARE])
def table(phase, k, fa, fb, draws):
    """per-image x draw matrix of (a_correct - b_correct); returns ids, labels, matrix [image, draw] (nan if not query)"""
    folds = sorted({key[3] for key in D if key[1]==phase})
    ids = {}; rows = []
    for d in draws:
        for f in folds:
            a = fa(phase, f, k, d); b = fb(phase, f, k, d)
            q, y = D[("MIRAGE", phase, "prototype", f, k, d)][:2]
            for qi, yi, ai, bi in zip(q, y, a, b):
                rows.append((qi, yi, d, f, ai, bi))
    return pd.DataFrame(rows, columns=["q","y","draw","fold","a","b"])
def proto(arm):
    return lambda ph,f,k,d: D[(arm,ph,"prototype",f,k,d)][2]
def logi(arm):
    return lambda ph,f,k,d: D[(arm,ph,"logistic",f,k,d)][2]
def ens(*arms):
    def g(ph,f,k,d):
        cells = [D[(a,ph,"prototype",f,k,d)] for a in arms]
        pred = zf(*[c[3] for c in cells])
        return (np.array(RARE)[pred] == cells[0][1]).astype(int)
    return g
rng = np.random.default_rng(0)
def analyse(name, phase, k, fa, fb, draws):
    t = table(phase, k, fa, fb, draws)
    # per draw: pooled over folds, macro over classes
    per_draw = t.groupby("draw").apply(lambda g: macro(g.a.values, g.y.values) - macro(g.b.values, g.y.values), include_groups=False)
    # per fold-level episode
    per_ep = t.groupby(["draw","fold"]).apply(lambda g: np.mean([g.a.values[g.y.values==c].mean() - g.b.values[g.y.values==c].mean() for c in RARE if (g.y.values==c).any()]), include_groups=False)
    # per image mean difference over draws
    im = t.assign(d=t.a - t.b).groupby(["y","q"]).d.mean().reset_index()
    est = im.groupby("y").d.mean().reindex(RARE)
    # stratified image bootstrap (images resampled within class; draws fixed)
    boots = []
    groups = [im[im.y==c].d.values for c in RARE]
    for _ in range(5000):
        boots.append(np.mean([rng.choice(g, len(g)).mean() for g in groups]))
    boots = np.array(boots)
    # two-way: resample draws and images
    piv = t.assign(d=t.a - t.b).pivot_table(index=["y","q"], columns="draw", values="d")
    lab = piv.index.get_level_values(0).values; M = piv.values
    idx = [np.where(lab==c)[0] for c in RARE]
    b2 = []
    for _ in range(3000):
        dd = rng.integers(0, M.shape[1], M.shape[1])
        b2.append(np.mean([np.nanmean(M[rng.choice(i, len(i))][:, dd]) for i in idx]))
    b2 = np.array(b2)
    nz = (im.d.abs() > 1e-12).sum()
    print(f"{name:55s} {phase[:3]} K={k:<2d} draws={len(draws):3d} mean={per_draw.mean():+.4f} SDdraw={per_draw.std(ddof=1):.4f} SDepisode={per_ep.std(ddof=1):.4f}(n={len(per_ep)}) "
          f"SE_image={boots.std(ddof=1):.4f} CI_image=[{np.quantile(boots,.025):+.3f},{np.quantile(boots,.975):+.3f}] SE_2way={b2.std(ddof=1):.4f} P(boot<=0)={np.mean(boots<=0):.3f} images_changed={nz}/{len(im)}")
    print("      per-class:", " ".join(f"{c}={v:+.3f}" for c, v in est.items()))
    return per_draw, im
A100 = range(100); A25 = range(25)
print("\n== P7 prototype ensemble MIRAGE+DINOv2 minus MIRAGE")
for k in (1,2,4,8):
    analyse("E2 proto minus MIRAGE proto", "development", k, ens("MIRAGE","DINOv2"), proto("MIRAGE"), A100)
analyse("E2 proto minus MIRAGE proto (first 25)", "development", 8, ens("MIRAGE","DINOv2"), proto("MIRAGE"), A25)
for k in (1,2,4,8,16):
    analyse("E2 proto minus MIRAGE proto", "reporting", k, ens("MIRAGE","DINOv2"), proto("MIRAGE"), A100)
print("\n== P6 logistic minus prototype (first 25)")
for arm in ("MIRAGE","DINOv2","Inception"):
    analyse(f"{arm} logistic minus prototype", "development", 4, logi(arm), proto(arm), A25)
    for k in (8,16):
        analyse(f"{arm} logistic minus prototype", "reporting", k, logi(arm), proto(arm), A25)
print("\n== single-model contrasts for scale (prototype, 100 draws)")
analyse("MIRAGE minus DINOv2", "development", 8, proto("MIRAGE"), proto("DINOv2"), A100)
analyse("MIRAGE minus DINOv2", "reporting", 8, proto("MIRAGE"), proto("DINOv2"), A100)
analyse("MIRAGE minus Inception", "development", 8, proto("MIRAGE"), proto("Inception"), A100)
