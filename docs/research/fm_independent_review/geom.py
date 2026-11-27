"""Read-only geometry/alpha/padding by class and source for the pinned nine-way roster."""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
W = "C:/Users/ThinkBook/.t3/worktrees/Few-shot/t3-03f3a2c4"; ROOT = "D:/Few-shot/"
sys.path.insert(0, W)
import fm_core as fm, fm_prepare as prep
rows = prep.load_manifest(Path(W))
pidm = {r["path"]: r["pid"] for r in rows}
rare_dev = sorted(r["path"] for r in rows if r["label"] in fm.RARE and r["split"] == "train")
roster = prep.nine_way_roster(rows, rare_dev)
# filename-derived pid equals manifest pid for Kermany?
bad = sum(1 for r in rows if r["label"] in prep.COMMON and r["path"].split("/")[-1].split("-")[1] != r["pid"])
print("Kermany rows where filename pid != manifest pid:", bad, "of", sum(r["label"] in prep.COMMON for r in rows))

def edge_pad(a, tol):
    """Share of area in complete border rows/cols whose (max-min) <= tol, peeled from each side."""
    h, w = a.shape
    top = bottom = left = right = 0
    while top < h and np.ptp(a[top]) <= tol: top += 1
    while bottom < h - top and np.ptp(a[h - 1 - bottom]) <= tol: bottom += 1
    while left < w and np.ptp(a[:, left]) <= tol: left += 1
    while right < w - left and np.ptp(a[:, w - 1 - right]) <= tol: right += 1
    inner = max(h - top - bottom, 0) * max(w - left - right, 0)
    return 1 - inner / (h * w)

out = []
for path, label, role in roster:
    with Image.open(ROOT + path) as im:
        fmt, mode, (w, h) = im.format, im.mode, im.size
        amin = None; rgb_under = None
        if "A" in mode:
            al = np.asarray(im.getchannel("A")); amin = int(al.min())
            if amin < 255:
                rgb = np.asarray(im.convert("RGB")); rgb_under = int(rgb[al < 255].max())
        g = np.asarray(im.convert("RGB").convert("L"))
    short = 256
    rw, rh = (short, int(short * h / w)) if w <= h else (int(short * w / h), short)
    retained = (min(224, rw) * min(224, rh)) / (rw * rh)
    out.append(dict(path=path, label=label, role=role, fmt=fmt, mode=mode, w=w, h=h, amin=amin, rgb_under=rgb_under,
                    retained=retained, aspect=w / h, pad0=edge_pad(g, 0), pad8=edge_pad(g, 8),
                    black=float((g <= 5).mean()), white=float((g >= 250).mean()), lo=int(g.min()), hi=int(g.max())))
json.dump(out, open(str(Path(__file__).with_name("geom.json")), "w"))
by = defaultdict(list)
for o in out: by[("rare" if o["label"] in fm.RARE else "Kermany", o["label"], o["role"])].append(o)
med = lambda v: float(np.median(v))
print("source label role n | fmt mode | w med[min-max] h med[min-max] aspect med | crop-retained med[min-max] | pad0 med, pad8 med[max], share pad8>0 | black<=5 med | white>=250 med | share hi==255 | share lo==0")
for k in sorted(by):
    v = by[k]
    f = lambda key: [o[key] for o in v]
    print(k, len(v), dict(Counter((o["fmt"], o["mode"]) for o in v)),
          f"w {med(f('w')):.0f}[{min(f('w'))}-{max(f('w'))}] h {med(f('h')):.0f}[{min(f('h'))}-{max(f('h'))}] asp {med(f('aspect')):.2f}",
          f"| ret {med(f('retained')):.2f}[{min(f('retained')):.2f}-{max(f('retained')):.2f}]",
          f"| pad0 {med(f('pad0')):.3f} pad8 {med(f('pad8')):.3f}[{max(f('pad8')):.2f}] >0:{np.mean(np.array(f('pad8'))>0):.2f}",
          f"| blk {med(f('black')):.3f} | wht {med(f('white')):.3f} (share>1%: {np.mean(np.array(f('white'))>0.01):.2f})",
          f"| hi255 {np.mean(np.array(f('hi'))==255):.2f} lo0 {np.mean(np.array(f('lo'))==0):.2f}")
al = [o for o in out if o["amin"] is not None]
print("images with alpha:", len(al), "alpha<255 somewhere:", sum(o["amin"] < 255 for o in al),
      "max RGB under non-opaque:", max([o["rgb_under"] for o in al if o["rgb_under"] is not None], default=None))
