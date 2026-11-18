"""Pixel layout only (flat border rows/columns, grey level, input min/max). No model is loaded.

Border rule = fm_core.border_share (A2): complete edge rows/columns with grey range <= 8, peeled
top, bottom, left, right, on image.convert("RGB").convert("L"). Raw images are opened read-only.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

base, data_root = Path(sys.argv[1]), Path(sys.argv[2])
PAD_RANGE = 8


def read(p):
    with open(p, newline="", encoding="utf-8") as s:
        return list(csv.DictReader(s))


def peel(a):
    h, w = a.shape
    top = bottom = left = right = 0
    while top < h and np.ptp(a[top]) <= PAD_RANGE:
        top += 1
    while bottom < h - top and np.ptp(a[h - 1 - bottom]) <= PAD_RANGE:
        bottom += 1
    while left < w and np.ptp(a[:, left]) <= PAD_RANGE:
        left += 1
    while right < w - left and np.ptp(a[:, w - 1 - right]) <= PAD_RANGE:
        right += 1
    return top, bottom, left, right


def pad_square(image):
    side = max(image.size)
    fill = {"L": 0, "RGB": (0, 0, 0), "RGBA": (0, 0, 0, 255)}[image.mode]
    canvas = Image.new(image.mode, (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def mirage_low_high(image):
    px = np.array(image.resize((512, 512), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255
    return float(px.min()), float(px.max())


roster = read(base / "lock" / "nine_way_roster.csv")
RARE = ("CSC", "MH", "MacTel", "Stargardt", "RP")
rows = []
for r in roster:
    if r["label"] in RARE or r["role"] == "query":
        with Image.open(data_root / r["path"]) as im:
            im = im.copy()
        gray = np.asarray(im.convert("RGB").convert("L"))
        h, w = gray.shape
        t, b, l, rt = peel(gray)
        ch, cw = max(h - t - b, 0), max(w - l - rt, 0)
        share = 1 - ch * cw / (h * w)
        rows_share = (t + b) / h  # horizontal bars (top/bottom)
        cols_share = (l + rt) / w
        border = np.ones_like(gray, dtype=bool)
        border[t:h - b, l:w - rt] = False
        bmean = float(gray[border].mean()) if border.any() else float("nan")
        dark = float(np.mean(gray <= 8))
        lo, hi = mirage_low_high(im)
        plo, phi = mirage_low_high(pad_square(im))
        rows.append(dict(path=r["path"], label=r["label"], role=r["role"], w=w, h=h, mode=im.mode, share=share,
                         rows_share=rows_share, cols_share=cols_share, border_mean=bmean, dark=dark,
                         content_aspect=(cw / ch if ch and cw else float("nan")), lo=lo, hi=hi, plo=plo, phi=phi,
                         gmin=int(gray.min())))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def show(name, rs):
    if not rs:
        return
    sh = [x["share"] for x in rs]
    bm = [x["border_mean"] for x in rs if x["share"] > 0.02]
    print(f"{name}: n={len(rs)} border_share median={median(sh):.3f} q25={q(sh, .25):.3f} q75={q(sh, .75):.3f} "
          f">=0.25:{sum(s >= 0.25 for s in sh)} >=0.10:{sum(s >= 0.10 for s in sh)} | top+bottom rows share median="
          f"{median(x['rows_share'] for x in rs):.3f} left+right cols share median={median(x['cols_share'] for x in rs):.3f} | "
          f"border mean grey (share>0.02, n={len(bm)}): median={median(bm) if bm else float('nan'):.1f} "
          f"dark(<=20):{sum(v <= 20 for v in bm)} white(>=235):{sum(v >= 235 for v in bm)} | "
          f"pixels<=8 share median={median(x['dark'] for x in rs):.3f} | content w/h median="
          f"{median(x['content_aspect'] for x in rs if x['content_aspect'] == x['content_aspect']):.2f}")
    lo = [x["lo"] for x in rs]
    print(f"    MIRAGE input before min-max: locked min==0: {sum(v == 0 for v in lo)}/{len(rs)}, locked min max={max(lo):.4f}; "
          f"padded min==0: {sum(x['plo'] == 0 for x in rs)}/{len(rs)}; images whose (min,max) change with padding: "
          f"{sum((x['lo'], x['hi']) != (x['plo'], x['phi']) for x in rs)}; largest min shift={max(x['lo'] - x['plo'] for x in rs):.4f}; "
          f"largest |max shift|={max(abs(x['hi'] - x['phi']) for x in rs):.4f}")


print("modes rare:", Counter(x["mode"] for x in rows if x["label"] in RARE))
for lab in RARE:
    show(f"rare {lab} (all)", [x for x in rows if x["label"] == lab])
show("rare all 146", [x for x in rows if x["label"] in RARE])
show("rare fitting 124", [x for x in rows if x["label"] in RARE and x["role"] == "fitting"])
for lab in ("NORMAL", "CNV", "DME", "DRUSEN"):
    for sz in (512, 768):
        show(f"Kermany test {lab} width {sz}", [x for x in rows if x["label"] == lab and x["w"] == sz])
show("Kermany test all 1000", [x for x in rows if x["label"] not in RARE])
# the layout a padded wide scan gets: share of the square that is flat, top+bottom
rare = [x for x in rows if x["label"] in RARE]
print("\nrare images with top+bottom flat rows >= 25% of height:", sum(x["rows_share"] >= 0.25 for x in rare),
      "| >= 33%:", sum(x["rows_share"] >= 0.33 for x in rare), "| left+right flat cols >= 25%:", sum(x["cols_share"] >= 0.25 for x in rare))
print("rare images, dark flat border (mean grey <= 20) and border share >= 0.25:",
      sum(x["share"] >= 0.25 and x["border_mean"] <= 20 for x in rare))
with open(base / "rare_layout.csv", "w", newline="", encoding="utf-8") as s:
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
