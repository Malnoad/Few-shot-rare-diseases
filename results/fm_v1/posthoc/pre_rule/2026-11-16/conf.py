import json, sys, os
root = sys.argv[1]
for src, sub in [('Kermany','reporting/nine_way'),('OCTID','external/OCTID/nine_way'),('OCTDL','external/OCTDL/nine_way')]:
    print('==', src)
    for arm in ['MIRAGE','DINOv2','RETFound','Inception']:
        m = json.load(open(os.path.join(root,'runs',arm,sub,'metrics.json')))
        c = m['classes']
        if arm=='MIRAGE': print('   pred order:', c)
        for i,row in enumerate(m['confusion']):
            if sum(row): print(f'  {arm:9s} true {c[i]:9s} n={sum(row):4d}', row)
