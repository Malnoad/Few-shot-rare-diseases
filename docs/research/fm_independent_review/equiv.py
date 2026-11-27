import sys; sys.path.insert(0, "C:/Users/ThinkBook/.t3/worktrees/Few-shot/t3-03f3a2c4")
import numpy as np, fm_core as fm
from pathlib import Path
from PIL import Image
rng = np.random.default_rng(1); same = 0; n = 0
for _ in range(200):
    sup = [rng.normal(size=(4, 32)) for _ in range(5)]; q = rng.normal(size=(50, 32))
    five = fm.unit(q) @ fm.centroids(sup).T
    two = fm.predict_prototype(fm.centroids(sup[:2]), q)
    same += int((five[:, :2].argmax(1) == two).sum()); n += 50
print("two-way refit == restriction of five-way cosine columns:", same, "/", n)
import fm_prepare as prep
rows = prep.load_manifest(Path("C:/Users/ThinkBook/.t3/worktrees/Few-shot/t3-03f3a2c4"))
rare = sorted((r["path"], r["split"]) for r in rows if r["label"] in fm.RARE)
h = {}
for p, s in rare:
    with Image.open("D:/Few-shot/" + p) as im: h[p] = (fm.dhash64(im), fm.rgb_fingerprint(im), s)
ps = sorted(h)
for i, a in enumerate(ps):
    for b in ps[i + 1:]:
        d = fm.hamming(h[a][0], h[b][0])
        if d <= 4 or h[a][1] == h[b][1]: print("flag:", a, h[a][2], b, h[b][2], "hamming", d, "rgb-equal", h[a][1] == h[b][1])
