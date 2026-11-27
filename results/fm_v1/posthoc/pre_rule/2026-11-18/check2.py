"""Feasibility only, Kermany images only. No rare-image pair is compared anywhere in this file."""
import zipfile, io, csv, sys, re, time, collections
import numpy as np
base = sys.argv[1]
with zipfile.ZipFile(base + '/input-lock.zip') as z:
    names = z.namelist()
    print([n for n in names][:20])
    inv_name = [n for n in names if n.endswith('extraction_inventory.csv')][0]
    inv = list(csv.DictReader(io.TextIOWrapper(z.open(inv_name), encoding='utf-8')))
print(len(inv), list(inv[0].keys()))
print(inv[0])
roles = collections.Counter((r.get('role', ''), r.get('label', '')) for r in inv)
print(sorted(roles.items()))
COMMON = {'CNV', 'DME', 'DRUSEN', 'NORMAL'}
pat = re.compile(r'([A-Z]+)-(\d+)-(\d+)\.jpeg$')
rng = np.random.default_rng(0)
for arm in ('MIRAGE', 'DINOv2', 'Inception', 'RETFound'):
    t0 = time.time()
    with zipfile.ZipFile(f'{base}/features-{arm}.zip') as z:
        npz = np.load(io.BytesIO(z.read('features.npz')), allow_pickle=False)
        keys = npz.files
        arrs = {k: npz[k] for k in keys}
    if arm == 'MIRAGE':
        print('npz keys', {k: (v.shape, str(v.dtype)) for k, v in arrs.items()})
    feat_key = [k for k, v in arrs.items() if v.ndim == 2][0]
    path_key = [k for k, v in arrs.items() if v.ndim == 1 and v.dtype.kind in 'US' and 'path' in k][0]
    X = arrs[feat_key].astype(np.float64); paths = arrs[path_key].tolist()
    idx, lab, pid = [], [], []
    for i, p in enumerate(paths):
        m = pat.search(p.replace('\\', '/'))
        if m and m[1] in COMMON:
            idx.append(i); lab.append(m[1]); pid.append(m[1] + '-' + m[2])
    idx = np.array(idx); lab = np.array(lab); pid = np.array(pid)
    Xk = X[idx]; Xk /= np.linalg.norm(Xk, axis=1, keepdims=True)
    print(arm, 'kermany images in locked features', len(idx), 'patients', len(set(pid)), 'load s', round(time.time() - t0, 1))
    for c in sorted(COMMON):
        m = lab == c
        S = Xk[m] @ Xk[m].T
        same = pid[m][:, None] == pid[m][None, :]
        iu = np.triu_indices(m.sum(), 1)
        s, y = S[iu], same[iu]
        n1, n0 = int(y.sum()), int((~y).sum())
        if n1 == 0:
            print('  ', c, 'no same-patient pairs'); continue
        neg = rng.choice(s[~y], size=min(n0, 200000), replace=False)
        pos = s[y]
        # AUROC by rank
        allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort() + 1
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        thr = np.quantile(neg, 0.999)
        print('  ', c, 'imgs', int(m.sum()), 'same-patient pairs', n1, 'diff pairs', n0, 'AUROC same-vs-diff %.3f' % auc,
              'share of same-patient pairs above the 99.9th pct of different-patient pairs: %.2f' % float((pos > thr).mean()))
