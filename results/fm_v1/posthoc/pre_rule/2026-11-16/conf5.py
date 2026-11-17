# Sums the pre-specified per-episode five-way confusion counts (metrics.json), prototype head only.
import zipfile, sys, json, re, collections
z = zipfile.ZipFile(sys.argv[1])
acc = collections.defaultdict(lambda: None); first = True
for n in z.namelist():
    m = re.match(r'runs/(\w+)/(development|reporting)/prototype/fold(\d)_k(\d+)_draw\d+/metrics.json', n)
    if not m: continue
    j = json.loads(z.read(n))
    if first: print({k: (v if k != 'confusion' else '...') for k, v in j.items()}); first = False
    arm, ph, f, k = m.groups(); k = int(k)
    if k == 16: continue
    key = (ph, arm)
    c = j['confusion']
    if acc[key] is None: acc[key] = [[0]*5 for _ in range(5)]
    for i in range(5):
        for jj in range(5): acc[key][i][jj] += c[i][jj]
cl = ['CSC','MH','MacTel','Stargardt','RP']
for key in sorted(acc):
    print(key, 'rows=true, cols=pred', cl, '(row shares, K=1,2,4,8 pooled)')
    for i,row in enumerate(acc[key]):
        s = sum(row); print(f'  {cl[i]:9s} n={s:7d} ', ' '.join('%.3f' % (x/s) for x in row))
