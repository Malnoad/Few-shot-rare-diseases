import json, sys, zipfile, re, collections, io, csv
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
show = True
acc = collections.defaultdict(list)   # (stage, arm, k, class) -> per-episode recall
pat = re.compile(r"runs/(\w+)/(development|reporting)/prototype/(?:fold(\d)_)?k(\d+)_draw(\d+)/metrics.json")
for n in names:
    m = pat.match(n)
    if not m: continue
    arm, stage, fold, k, draw = m.groups()
    d = json.loads(z.read(n))
    if show:
        print(n); print(json.dumps(d)[:700]); show = False
    acc[(stage, arm, int(k), "_raw")].append((fold, draw, d))
json.dump({ "|".join(map(str,k)): v for k,v in acc.items()}, open(sys.argv[2], "w"))
print(len(acc))
