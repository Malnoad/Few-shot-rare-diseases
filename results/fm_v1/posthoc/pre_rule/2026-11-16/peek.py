import zipfile, sys, json
z = zipfile.ZipFile(sys.argv[1])
def show(o, d=0, k=''):
    if isinstance(o, dict):
        print(' '*d + f'{k}: dict[{len(o)}] keys={list(o)[:8]}')
        for kk in list(o)[:3]:
            if d < 6: show(o[kk], d+2, kk)
    elif isinstance(o, list):
        print(' '*d + f'{k}: list[{len(o)}]')
        if o and d < 6: show(o[0], d+2, '[0]')
    else: print(' '*d + f'{k}: {str(o)[:100]}')
for n in sys.argv[2:]:
    print('#', n); show(json.loads(z.read(n)))
