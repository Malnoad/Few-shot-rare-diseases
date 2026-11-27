import zipfile, glob, hashlib, io, os, csv, collections
from pathlib import Path
from PIL import Image

RAW = Path('D:/Few-shot/data/raw/rare')


def pix(b):
    im = Image.open(io.BytesIO(b))
    im.load()
    return hashlib.sha256(im.convert('RGBA').tobytes() + str(im.size).encode()).hexdigest()


ours_pix, ours_file = {}, {}
for p in sorted(RAW.glob('*/*')):
    b = p.read_bytes()
    rel = p.relative_to(RAW).as_posix()
    ours_pix.setdefault(pix(b), []).append(rel)
    ours_file.setdefault(hashlib.sha256(b).hexdigest(), []).append(rel)

rows = []
for z in sorted(glob.glob('web_yoo/dl_v1/*.zip')):
    zf = zipfile.ZipFile(z)
    for i in zf.infolist():
        n = i.filename
        if i.is_dir() or '__MACOSX' in n or os.path.basename(n).startswith('._'):
            continue
        b = zf.read(i)
        rows.append((Path(z).name, n, ';'.join(ours_pix.get(pix(b), [])),
                     ';'.join(ours_file.get(hashlib.sha256(b).hexdigest(), []))))

with open('web_yoo/v1_to_ours.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['v1_zip', 'v1_file', 'ours_pixel_match', 'ours_file_match'])
    w.writerows(rows)

c = collections.Counter((r[0], r[2].split('/')[0] if r[2] else 'NONE') for r in rows)
for k, v in sorted(c.items()):
    print(k, v)
print('file-identical:', sum(1 for r in rows if r[3]), 'of', len(rows))
print('same name:', sum(1 for r in rows if r[2] and r[2].split('/')[1] == r[1]), 'of', sum(1 for r in rows if r[2]))
