import pandas as pd, sys, collections
from pathlib import Path, PureWindowsPath
from PIL import Image
root = Path(sys.argv[1])
df = pd.read_csv(root / 'data/manifest.csv')
print(df.shape, df.columns.tolist())
print(df.groupby(['label', 'split']).size().unstack(fill_value=0))
COMMON = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
k = df[df.label.isin(COMMON)].copy()
k['key'] = k.label + '-' + k.pid.astype(str)
g = k.groupby('key').size()
print('kermany patients(label-pid):', len(g), 'with2+:', int((g >= 2).sum()), 'median', g.median(), 'max', g.max())
print('unique pid', k.pid.nunique(), 'unique label-pid', len(g))
for c in COMMON:
    gc = k[k.label == c].groupby('pid').size()
    print(c, 'patients', len(gc), '2+', int((gc >= 2).sum()), 'quantiles 50/90/99', gc.quantile([.5, .9, .99]).tolist())
te = k[k.split == 'test']; tr = k[k.split != 'test']
print('test imgs', len(te), 'test patients', te.key.nunique(), 'also in train/val', len(set(te.key) & set(tr.key)))
gt = te.groupby('key').size()
print('test patients with 2+ test imgs', int((gt >= 2).sum()), 'same-patient pairs within test', int((gt * (gt - 1) // 2).sum()))
for c in COMMON:
    t = te[te.label == c]; gg = t.groupby('key').size()
    print('  test', c, 'imgs', len(t), 'patients', len(gg), 'patients2+', int((gg >= 2).sum()), 'pairs', int((gg * (gg - 1) // 2).sum()),
          'disjoint-from-train imgs', int((~t.key.isin(set(tr.key))).sum()))
k['n3'] = k.path.str.extract(r'-(\d+)\.jpeg$')[0].astype(int)
print('third-field max per patient quantiles 50/90/99/100', k.groupby('key').n3.max().quantile([.5, .9, .99, 1]).tolist())
r = df[~df.label.isin(COMMON)]
print(r.groupby(['label', 'split']).size().unstack(fill_value=0))
names = [PureWindowsPath(p).name for p in r.path]
print(names[:8], names[-8:])
print(collections.Counter(PureWindowsPath(p).suffix.lower() for p in r.path))
# image format facts only (size, mode, metadata keys); no pairwise comparison
modes = collections.Counter(); fmts = collections.Counter(); info = collections.Counter(); sizes = collections.defaultdict(collections.Counter)
for p, lab in zip(r.path, r.label):
    f = root / PureWindowsPath(p).as_posix()
    with Image.open(f) as im:
        modes[im.mode] += 1; fmts[im.format] += 1
        for key in im.info: info[key] += 1
        ex = im.getexif()
        info['EXIF_tags=%d' % len(ex)] += 1
        sizes[lab][im.size] += 1
print(modes, fmts); print(info)
for lab, c in sizes.items():
    ws = sorted(w for (w, h), n in c.items() for _ in range(n)); hs = sorted(h for (w, h), n in c.items() for _ in range(n))
    print(lab, 'n', len(ws), 'distinct sizes', len(c), 'w', ws[0], ws[len(ws)//2], ws[-1], 'h', hs[0], hs[len(hs)//2], hs[-1])
# a few kermany sizes
kc = collections.Counter()
for p in k[k.split == 'test'].path.sample(200, random_state=0):
    with Image.open(root / PureWindowsPath(p).as_posix()) as im:
        kc[(im.size, im.mode)] += 1
print('kermany test sample sizes', kc.most_common(6))
