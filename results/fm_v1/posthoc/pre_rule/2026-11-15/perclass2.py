import json, sys, collections
raw = json.load(open(sys.argv[1]))
C = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
ARMS = ["MIRAGE", "DINOv2", "Inception", "RETFound"]
tab = {}
for key, eps in raw.items():
    stage, arm, k, _ = key.split("|")
    by_draw = collections.defaultdict(lambda: [[0]*5, [0]*5])
    for fold, draw, d in eps:
        assert d["classes"] == C
        for i in range(5):
            by_draw[draw][0][i] += d["correct"][i]; by_draw[draw][1][i] += d["totals"][i]
    rec = [[c[i]/t[i] for i in range(5)] for c, t in by_draw.values()]
    tab[(stage, arm, int(k))] = ([sum(r[i] for r in rec)/len(rec) for i in range(5)], len(rec), list(by_draw.values())[0][1])
for stage in ("development", "reporting"):
    ks = sorted({k for (s, a, k) in tab if s == stage})
    print("==", stage, "totals", tab[(stage, "MIRAGE", ks[0])][2], "draws", tab[(stage, "MIRAGE", ks[0])][1])
    for k in ks + ["mean"]:
        print(" K =", k)
        for arm in ARMS:
            if k == "mean":
                v = [sum(tab[(stage, arm, kk)][0][i] for kk in ks)/len(ks) for i in range(5)]
            else:
                v = tab[(stage, arm, k)][0]
            print("   %-10s" % arm, "  ".join("%s %.3f" % (c, x) for c, x in zip(C, v)), "  macro %.3f" % (sum(v)/5))
