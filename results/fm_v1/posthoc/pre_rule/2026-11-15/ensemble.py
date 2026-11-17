"""Post hoc score-level ensemble from the stored FM results (no features, no GPU).

Rule, fixed before any ensemble number was computed: for each member model and each query image,
standardise the class cosine scores (subtract their mean, divide by their standard deviation), add
the standardised scores over members with equal weight, predict the highest sum. No tuning.
Members declared in advance: E2 = MIRAGE + DINOv2; E3 = MIRAGE + DINOv2 + Inception.
"""
import collections, csv, io, json, re, sys, zipfile
import numpy as np

ARMS = ["MIRAGE", "DINOv2", "Inception", "RETFound"]
GROUPS = {"MIRAGE": ["MIRAGE"], "DINOv2": ["DINOv2"], "Inception": ["Inception"], "RETFound": ["RETFound"],
          "E2 MIRAGE+DINOv2": ["MIRAGE", "DINOv2"], "E3 +Inception": ["MIRAGE", "DINOv2", "Inception"]}
RARE = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
z = zipfile.ZipFile(sys.argv[1])
names = set(z.namelist())


def load(path):
    rows = list(csv.DictReader(io.StringIO(z.read(path).decode("utf-8"))))
    cols = [c for c in rows[0] if c.startswith("cosine_")]
    classes = [c[len("cosine_"):] for c in cols]
    ids = [r["query_id"] for r in rows]
    true = [r["true_label"] for r in rows]
    s = np.array([[float(r[c]) for c in cols] for r in rows])
    return ids, true, classes, s


def fused(paths):
    """paths: one predictions.csv per member, same episode. Returns ids, true, classes, predicted."""
    total = None
    for p in paths:
        ids, true, classes, s = load(p)
        zs = (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)
        if total is None:
            ref, total = (ids, true, classes), zs
        else:
            assert (ids, true, classes) == ref, p
            total = total + zs
    return ref[0], ref[1], ref[2], [ref[2][i] for i in total.argmax(1)]


def episodes(prefix):
    pat = re.compile(re.escape(prefix % "MIRAGE") + r"prototype/fold(\d)_k(\d+)_draw(\d+)/predictions.csv")
    out = collections.defaultdict(list)
    for n in names:
        m = pat.match(n)
        if m:
            out[(int(m.group(2)), int(m.group(3)))].append(n.replace("runs/MIRAGE/", "runs/%s/"))
    return out


def kshot(title, prefix, classes_scored=None):
    eps = episodes(prefix)
    ks = sorted({k for k, _ in eps})
    print("\n==", title, "| draws per K:", len([1 for k, d in eps if k == ks[0]]))
    macro = {}
    per_class = {}
    for g, members in GROUPS.items():
        for k in ks:
            vals, pcs = [], []
            for (kk, draw), files in sorted(eps.items()):
                if kk != k:
                    continue
                hit, tot = collections.Counter(), collections.Counter()
                for f in files:
                    _, true, _, pred = fused([f % a for a in members])
                    for t, p in zip(true, pred):
                        tot[t] += 1; hit[t] += (t == p)
                cls = classes_scored or sorted(tot)
                rec = [hit[c] / tot[c] for c in cls]
                vals.append(sum(rec) / len(rec)); pcs.append(rec)
            macro[(g, k)] = np.array(vals); per_class[(g, k)] = (cls, np.mean(pcs, 0))
    for g in GROUPS:
        row = [macro[(g, k)].mean() for k in ks]
        print("  %-18s" % g, " ".join("K%-2d %.3f" % (k, v) for k, v in zip(ks, row)), "| mean(all K) %.3f" % np.mean(row),
              "| mean(K<=8) %.3f" % np.mean([v for k, v in zip(ks, row) if k <= 8]))
    for g in ("E2 MIRAGE+DINOv2", "E3 +Inception"):
        d = np.mean([macro[(g, k)] - macro[("MIRAGE", k)] for k in ks if k <= 8], 0)
        print("  %s minus MIRAGE, mean over K<=8: %+.3f (p05 %+.3f, p95 %+.3f)" % (g, d.mean(), np.percentile(d, 5), np.percentile(d, 95)))
        for k in ks:
            d = macro[(g, k)] - macro[("MIRAGE", k)]
            print("      K=%-2d %+.3f (p05 %+.3f, p95 %+.3f)" % (k, d.mean(), np.percentile(d, 5), np.percentile(d, 95)))
    print("  per-class recall, mean over K<=8:")
    for g in GROUPS:
        cls = per_class[(g, ks[0])][0]
        v = np.mean([per_class[(g, k)][1] for k in ks if k <= 8], 0)
        print("    %-18s" % g, "  ".join("%s %.3f" % (c, x) for c, x in zip(cls, v)))


def nine(title, path, sources=None):
    print("\n==", title)
    for g, members in GROUPS.items():
        ids, true, classes, pred = fused([path % a for a in members])
        normal = [(t, p) for t, p in zip(true, pred) if t == "NORMAL"]
        n = len(normal)
        as_normal = sum(p == "NORMAL" for _, p in normal); as_rare = sum(p in RARE for _, p in normal)
        rare = [(t, p) for t, p in zip(true, pred) if t in RARE]
        line = "  %-18s normal: %d/%d called normal, %d called rare" % (g, as_normal, n, as_rare)
        if rare:
            line += " | rare: %d/%d correct" % (sum(t == p for t, p in rare), len(rare))
            by = collections.Counter(); tot = collections.Counter()
            for t, p in rare:
                tot[t] += 1; by[t] += (t == p)
            line += " (" + ", ".join("%s %d/%d" % (c, by[c], tot[c]) for c in RARE if tot[c]) + ")"
        print(line)


kshot("Development (124 images, K-shot prototype)", "runs/%s/development/", RARE)
kshot("Hold-out (22 images)", "runs/%s/reporting/", RARE)
kshot("OCTID two-way", "runs/%s/external/OCTID/")
kshot("OCTDL MH", "runs/%s/external/OCTDL/")
nine("Nine-class audit: Kermany test + 22 hold-out", "runs/%s/reporting/nine_way/predictions.csv")
nine("Nine-class audit: OCTID", "runs/%s/external/OCTID/nine_way/predictions.csv")
nine("Nine-class audit: OCTDL", "runs/%s/external/OCTDL/nine_way/predictions.csv")
