import json, sys
def show(o, d=0, maxd=4, key=''):
    pad = '  '*d
    if isinstance(o, dict):
        ks = list(o.keys())
        print(f"{pad}{key}: dict[{len(ks)}] {ks[:12]}")
        if d < maxd:
            for k in ks[:6]:
                show(o[k], d+1, maxd, k)
    elif isinstance(o, list):
        print(f"{pad}{key}: list[{len(o)}] {str(o[:6])[:160]}")
        if d < maxd and o and isinstance(o[0], (dict, list)):
            show(o[0], d+1, maxd, '[0]')
    else:
        print(f"{pad}{key}: {str(o)[:120]}")
for p in sys.argv[1:]:
    print("=====", p)
    show(json.load(open(p)), maxd=3)
