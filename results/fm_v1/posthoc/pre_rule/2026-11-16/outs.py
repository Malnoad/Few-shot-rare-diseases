import csv, sys, os, collections
root = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(root,'reports/external/external_transfer_outputs.csv'))))
d = collections.defaultdict(dict)
for r in rows:
    d[(r['source'],r['true_class'],r['classifier'],r['arm'],r['k'])][r['predicted_class']] = float(r['mean_count'])
cl = ['CSC','MH','MacTel','Stargardt','RP']
for key in sorted(d):
    v = d[key]; n = sum(v.values())
    print(*key, 'n=%.0f'%n, ' '.join('%s %.2f'%(c, v.get(c,0)/n) for c in cl))
