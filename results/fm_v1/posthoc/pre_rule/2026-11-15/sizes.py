import csv, sys, collections
ext, inv = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(ext, newline='')))
print(rows[0].keys())
c = collections.Counter((r['source'], r['label'], r['width'], r['height']) for r in rows)
octid = {k: v for k, v in c.items() if k[0] == 'OCTID'}
for k, v in sorted(octid.items()): print(k, v)
print('OCTDL distinct sizes', len({(k[2], k[3]) for k in c if k[0]=='OCTDL'}))
rows = list(csv.DictReader(open(inv, newline='')))
print(rows[0].keys())
