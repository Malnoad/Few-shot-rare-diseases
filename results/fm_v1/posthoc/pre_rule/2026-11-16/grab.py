import re, sys, html
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="ascii", errors="replace")
p, pats = sys.argv[1], sys.argv[2:]
t = open(p, encoding='utf-8', errors='replace').read()
t = re.sub(r'<xref[^>]*>(.*?)</xref>', r'[\1]', t, flags=re.S)
t = html.unescape(re.sub(r'<[^>]+>', ' ', t)); t = re.sub(r'\s+', ' ', t)
print('LEN', len(t), t[:150])
for pat in pats:
    ms = list(re.finditer(pat, t, flags=re.I))
    print(f'\n### {pat}: {len(ms)} hits')
    last = -9999
    for m in ms[:12]:
        if m.start() - last < 500: continue
        last = m.start()
        print('  ...', t[max(0,m.start()-350):m.end()+450], '...')
