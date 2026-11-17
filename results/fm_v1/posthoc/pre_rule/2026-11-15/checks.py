import csv, sys, collections, statistics
from scipy.stats import beta
lock, ext = sys.argv[1], sys.argv[2]
R = lambda p: list(csv.DictReader(open(p, newline='')))
roster = R(lock + '/nine_way_roster.csv')
print('nine-way roster', sorted(collections.Counter((r['role'], r['label']) for r in roster).items()))
q = R(ext + '/external_queries.csv')
print('external queries', sorted(collections.Counter((r['source'], r['label']) for r in q).items()))
print('dependence groups', {k: len({r['dependence_group'] for r in q if (r['source'], r['label']) == k}) for k in {(r['source'], r['label']) for r in q}})
pairs = R(ext + '/external_screen_pairs.csv')
print('ext pairs', collections.Counter((p['kind'], p['a'].split('/')[2], p['a'].split('/')[3] if p['a'].split('/')[2]=='OCTDL' else p['a'].split('/')[3], p['b'].split('/')[3]) for p in pairs))
inv = R(lock + '/image_inventory.csv')
for split in ('train', 'test'):
    for c in ('CSC','MH','MacTel','Stargardt','RP'):
        rows = [r for r in inv if r['label']==c and r['split']==split]
        w = [int(r['width']) for r in rows]; h = [int(r['height']) for r in rows]
        print(split, c, len(rows), 'w med', statistics.median(w), min(w), max(w), 'h med', statistics.median(h), min(h), max(h), 'modes', collections.Counter(r['mode'] for r in rows))
# fold sizes
g = R(lock + '/groups.csv')
print('fold totals', sorted(collections.Counter(r['fold'] for r in g).items()), 'groups', len({r['group_id'] for r in g}), 'members', len(g))
ep = R(lock + '/episodes_development.csv'); er = R(lock + '/episodes_reporting.csv')
print('dev rows', len(ep), 'rep rows', len(er))
# legacy queries in reporting supports?
test = {r['path'] for r in inv if r['split']=='test'}
print('reporting supports overlapping test:', sum(1 for r in er for k,v in r.items() if k.startswith('s') and v in test))
print('CP upper 0/169:', beta.ppf(0.975, 1, 169), ' 0/250:', beta.ppf(0.975, 1, 250))
