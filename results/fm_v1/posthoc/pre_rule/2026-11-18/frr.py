import zipfile, io, re
import numpy as np, pandas as pd
Z = "D:/Few-shot/drive/fm_v1/final/final-20261113T170804Z/evaluation/results-20261114T093033Z.zip"
RARE = ["CSC","MH","MacTel","Stargardt","RP"]; ALL = ["NORMAL","CNV","DME","DRUSEN"] + RARE
rng = np.random.default_rng(0)
z = zipfile.ZipFile(Z)
def nine(arm, src):
    d = pd.read_csv(io.BytesIO(z.read(f"runs/{arm}/external/{src}/nine_way/predictions.csv")))
    return d
for src in ("OCTID","OCTDL"):
    d = {a: nine(a, src) for a in ("MIRAGE","DINOv2","Inception")}
    m = d["MIRAGE"]; assert all((d[a].query_id == m.query_id).all() for a in d)
    S = {a: d[a][["cosine_"+c for c in ALL]].values for a in d}
    zs = lambda s: (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)
    pred = {a: S[a].argmax(1) for a in S}
    pred["E2"] = (zs(S["MIRAGE"]) + zs(S["DINOv2"])).argmax(1)
    norm = (m.true_label == "NORMAL").values
    fr = {a: (pred[a] >= 4)[norm].astype(int) for a in pred}
    ids = m.query_id.values[norm]
    if src == "OCTDL":
        pat = np.array([re.search(r"_(\d+)_\d+\.", q).group(1) for q in ids])
    else:
        pat = ids
    up = np.unique(pat); print(src, "normals", norm.sum(), "clusters", len(up), {a: int(fr[a].sum()) for a in fr})
    for a in ("E2","DINOv2","Inception"):
        diff = fr[a] - fr["MIRAGE"]
        b = (fr[a]==1)&(fr["MIRAGE"]==0); c = (fr[a]==0)&(fr["MIRAGE"]==1)
        per = pd.Series(diff).groupby(pat).sum(); cnt = pd.Series(diff).groupby(pat).size()
        boots = []
        for _ in range(5000):
            i = rng.integers(0, len(per), len(per))
            boots.append(per.values[i].sum() / cnt.values[i].sum())
        boots = np.array(boots)
        print(f"   {a:9s} - MIRAGE: diff={diff.mean():+.4f} new false-rare={b.sum()} removed={c.sum()} clusterSE={boots.std(ddof=1):.4f} one-sided 95% upper={np.quantile(boots,.95):+.4f}  naive image SE={np.sqrt((b.sum()+c.sum()) - (b.sum()-c.sum())**2/len(diff))/len(diff):.4f}")
    # rare images: share called any rare, own class
    for cls in sorted(set(m.true_label) - {"NORMAL"}):
        r = (m.true_label == cls).values
        print("   rare", cls, r.sum(), {a: (int((pred[a] >= 4)[r].sum()), int((np.array(ALL)[pred[a]][r] == cls).sum())) for a in pred})
    if src == "OCTDL":
        mis = fr["MIRAGE"]; per = pd.Series(mis).groupby(pat).agg(["sum","size"])
        print("   MIRAGE false-rare by patient: patients with >=1:", (per["sum"]>0).sum(), "of", len(per), "max images/patient", per["size"].max())
