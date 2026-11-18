import csv, sys, collections
def load(p): return list(csv.DictReader(open(p, newline='', encoding='utf-8')))
m = load(sys.argv[1]); i = load(sys.argv[2])
print("rows", len(m), len(i), "modes M", collections.Counter(r['mode'] for r in m), "I", collections.Counter(r['mode'] for r in i))
print("same geometry:", all((a['path'],a['width'],a['height'],a['side'],a['left'],a['top'])==(b['path'],b['width'],b['height'],b['side'],b['left'],b['top']) for a,b in zip(m,i)))
roster = {}
if len(sys.argv) > 3:
    for r in load(sys.argv[3]):
        roster[r['path']] = (r['role'], r['label'])
def grp(p):
    s = p.split('/')
    if p in roster:
        src = 'rare' if 'rare' in s else 'Kermany'
        return 'ROSTER %s %s %s' % (roster[p][0], src, roster[p][1])
    if 'rare' in s: return 'other rare/' + s[s.index('rare')+1]
    if 'OCT2017' in s: return 'other Kermany/'+s[s.index('OCT2017')+1]+'/'+s[s.index('OCT2017')+2]
    return 'EXT ' + '/'.join(s[:-1])[-45:]
g = collections.defaultdict(list)
for r in m:
    w,h,side = int(r['width']),int(r['height']),int(r['side'])
    g[grp(r['path'])].append((1-w*h/side**2, w/h, w, h))
q = lambda s,p: s[min(len(s)-1,int(p*len(s)))]
for k in sorted(g):
    v = g[k]; f = sorted(x[0] for x in v); a = sorted(x[1] for x in v)
    sizes = collections.Counter((x[2],x[3]) for x in v).most_common(3)
    print(f"{k:50s} n={len(v):5d} padfrac min/med/p90/max {f[0]:.3f}/{q(f,.5):.3f}/{q(f,.9):.3f}/{f[-1]:.3f} share>=.25: {sum(x>=.25 for x in f)/len(f):.2f} aspect med {q(a,.5):.2f} [{a[0]:.2f},{a[-1]:.2f}] sq={sum(x[2]==x[3] for x in v)} tall={sum(x[2]<x[3] for x in v)} top {sizes}")
