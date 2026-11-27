"""Rare images: where the flat black rows sit (top / bottom), alpha, and strip height in the 512 frame."""
import csv, sys
from pathlib import Path
from statistics import median
import numpy as np
from PIL import Image
base, root = Path(sys.argv[1]), Path(sys.argv[2])
rows = [r for r in csv.DictReader(open(base / "lock" / "nine_way_roster.csv", encoding="utf-8")) if "/rare/" in r["path"]]
both = top_only = bottom_only = none = 0
alpha_lt = 0
strip = []
tops, bots = [], []
for r in rows:
    with Image.open(root / r["path"]) as im:
        im = im.copy()
    if im.mode == "RGBA" and np.asarray(im)[..., 3].min() < 255:
        alpha_lt += 1
    g = np.asarray(im.convert("RGB").convert("L"))
    h, w = g.shape
    t = 0
    while t < h and np.ptp(g[t]) <= 8: t += 1
    b = 0
    while b < h - t and np.ptp(g[h - 1 - b]) <= 8: b += 1
    tf, bf = t / h, b / h
    tops.append(tf); bots.append(bf)
    big_t, big_b = tf >= 0.05, bf >= 0.05
    both += big_t and big_b; top_only += big_t and not big_b; bottom_only += big_b and not big_t; none += not (big_t or big_b)
    side = max(w, h)
    strip.append((h - t - b) / side)  # strip height as a share of the padded square = of the 512 model frame
print("rare n", len(rows), "| bars >=5% of height: both", both, "top only", top_only, "bottom only", bottom_only, "neither", none)
print("top share median %.3f, bottom share median %.3f" % (median(tops), median(bots)))
print("images with any alpha < 255:", alpha_lt)
s = sorted(strip)
print("retina strip height / frame side: median %.3f q25 %.3f q75 %.3f min %.3f max %.3f" % (median(s), s[len(s)//4], s[3*len(s)//4], s[0], s[-1]))
print("for comparison, padded externals: OCTID 500/750 = %.3f; OCTDL median 321/882 = %.3f; Kermany 768x496 = %.3f; Kermany 512x496 = %.3f" % (500/750, 321/882, 496/768, 496/512))
