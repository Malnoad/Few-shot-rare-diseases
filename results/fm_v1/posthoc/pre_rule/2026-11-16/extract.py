import zipfile, sys
z = zipfile.ZipFile(sys.argv[1]); out = sys.argv[2]
for n in z.namelist():
    if n.startswith('reports/') or (n.startswith('runs/') and 'fold' not in n and not n.endswith(('head.json','complete.json','predictions.csv'))):
        z.extract(n, out)
