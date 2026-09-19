"""
Stage 4 - evaluation, metric-learning baselines and CAM  (GAN-FSL-OCT reproduction)

Metrics on the independent test set (paper Tables 3-4, Fig. 9):
  accuracy, unweighted Cohen's kappa, RCI, multiclass MCC (Gorodkin), per-class recall
  (true positive rate), one-vs-rest AUC, plus bootstrap 95% CIs for accuracy and MCC.

Models evaluated:
  none / basic / gan   every results/stage3_<variant>/test_predictions.csv found
  knn                  K=3 Euclidean nearest neighbours (Quellec et al.)      } on frozen ImageNet
  proto                nearest class-mean prototype, Euclidean                } Inception-v3 features of the
                                                                              } no-augmentation training set
  (the paper's Siamese network is not included)

Usage (from the project root, after Stage 3 has run at least once):
  python stage4_evaluate.py
  python stage4_evaluate.py --cam 40        # also CAM heatmaps for 40 test images (rare first)

Outputs in results/stage4/: metrics.csv, recall.csv, auc.csv, confusion_<model>.csv, cam_<variant>/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)

CLASSES = ["NORMAL", "CNV", "DME", "DRUSEN", "CSC", "MH", "MacTel", "Stargardt", "RP"]
RARE = CLASSES[4:]

# Paper, independent test set (Table 3 and Table 4)
PAPER = {
    "none":  dict(acc=88.4, kappa=0.847, rci=0.916, mcc=0.848),
    "basic": dict(acc=91.3, kappa=0.886, rci=0.953, mcc=0.887),
    "gan":   dict(acc=92.1, kappa=0.896, rci=0.983, mcc=0.897),
    "knn":   dict(acc=91.2, kappa=0.885, rci=0.972, mcc=0.886),
    "proto": dict(acc=80.8, kappa=0.753, rci=0.933, mcc=0.755),
}


def fix(p):
    return str(p).replace("\\", "/")


def rci(cm):
    """Relative classifier information: I(true; pred) / H(true)."""
    p = cm / cm.sum()
    pt, pp = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pt, pp)[nz])).sum()
    h = -(pt[pt > 0] * np.log(pt[pt > 0])).sum()
    return mi / h


def evaluate(y, yhat, P, n_boot=1000, seed=0):
    cm = confusion_matrix(y, yhat, labels=range(len(CLASSES)))
    m = dict(acc=accuracy_score(y, yhat) * 100, kappa=cohen_kappa_score(y, yhat),
             rci=rci(cm), mcc=matthews_corrcoef(y, yhat))
    rng = np.random.default_rng(seed)
    accs, mccs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        accs.append(accuracy_score(y[i], yhat[i]) * 100)
        mccs.append(matthews_corrcoef(y[i], yhat[i]))
    m["acc_ci"] = f"{np.percentile(accs, 2.5):.1f}-{np.percentile(accs, 97.5):.1f}"
    m["mcc_ci"] = f"{np.percentile(mccs, 2.5):.3f}-{np.percentile(mccs, 97.5):.3f}"
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    auc = [roc_auc_score(y == k, P[:, k]) if 0 < (y == k).sum() < len(y) else np.nan
           for k in range(len(CLASSES))]
    return m, recall, auc, cm


def load_feats(cache, group):
    f = cache / f"{group}.npz"
    if not f.exists():
        return None, None
    z = np.load(f, allow_pickle=False)
    return z["feats"].astype(np.float32), z["paths"].tolist()


def baselines(root, man):
    """KNN (K=3) and prototypes on cached frozen features of the no-augmentation training set."""
    from sklearn.neighbors import KNeighborsClassifier

    cache = root / "results/features"
    Xm, pm = load_feats(cache, "major_train")
    Xt, pt = load_feats(cache, "test")
    if Xm is None or Xt is None:
        print("Baselines skipped: run `python stage3_classifier.py --variant none` first (it caches the features)")
        return {}
    lab = dict(zip(man.path, man.label))
    X, y = [Xm], [lab[p] for p in pm]
    for cls in RARE:
        Xr, pr = load_feats(cache, f"real_{cls}")
        if Xr is None:
            print(f"Baselines skipped: no cached real_{cls} features (run Stage 3 --variant none)")
            return {}
        X.append(Xr); y += [lab[p] for p in pr]
    X = np.concatenate(X)
    y = np.array([CLASSES.index(c) for c in y])
    yt = np.array([CLASSES.index(lab[p]) for p in pt])

    out = {}
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    P = knn.predict_proba(Xt)
    out["knn"] = (yt, P.argmax(1), P)

    protos = np.stack([X[y == k].mean(0) for k in range(len(CLASSES))])
    d = ((Xt[:, None, :] - protos[None]) ** 2).sum(-1)
    P = np.exp(-(d - d.min(1, keepdims=True)) / d.std())
    P /= P.sum(1, keepdims=True)
    out["proto"] = (yt, d.argmin(1), P)
    return out


def cam(root, variant, n, weights):
    """Class activation maps. With a frozen backbone + GAP + one softmax layer,
    Grad-CAM is exactly CAM: ReLU(sum_k W[k, c] * A_k) over the mixed10 feature maps."""
    import cv2
    import tensorflow as tf

    head = np.load(root / f"results/stage3_{variant}/head.npz")
    W = head["W"]
    preds = pd.read_csv(root / f"results/stage3_{variant}/test_predictions.csv")
    preds = pd.concat([preds[preds.label.isin(RARE)], preds[~preds.label.isin(RARE)].sample(frac=1, random_state=0)])
    preds = preds.head(n)
    net = tf.keras.applications.InceptionV3(include_top=False, weights=None if weights == "None" else weights, input_shape=(299, 299, 3))
    out = root / f"results/stage4/cam_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    for _, r in preds.iterrows():
        raw = tf.io.decode_image(tf.io.read_file(str(root / fix(r.path))), channels=3, expand_animations=False)
        img = tf.image.resize(tf.image.resize(raw, [256, 256]), [299, 299])
        A = net(img[None] / 127.5 - 1.0, training=False).numpy()[0]           # 8x8x2048
        c = CLASSES.index(r.pred)
        m = np.maximum(A @ W[:, c], 0)
        m = cv2.resize(m / (m.max() + 1e-8), (299, 299))
        base = np.clip(img.numpy(), 0, 255).astype(np.uint8)[:, :, ::-1]
        heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
        panel = np.hstack([base, cv2.addWeighted(base, 0.6, heat, 0.4, 0)])
        name = f"{r.label}_as_{r.pred}_{Path(fix(r.path)).stem}.jpg"
        cv2.imwrite(str(out / name), panel)
    print(f"CAM: {len(preds)} panels in {out} (file name = true_as_predicted)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--cam", type=int, default=0, help="number of test images for CAM heatmaps")
    ap.add_argument("--cam-variant", default="gan", choices=["none", "basic", "gan"])
    ap.add_argument("--weights", default="imagenet", help=argparse.SUPPRESS)  # testing only
    a = ap.parse_args()

    root = Path(a.root)
    out = root / "results/stage4"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(root / "data/manifest.csv")
    man["path"] = man.path.map(fix)

    runs = {}
    for v in ["none", "basic", "gan"]:
        f = root / f"results/stage3_{v}/test_predictions.csv"
        if f.exists():
            t = pd.read_csv(f)
            y = np.array([CLASSES.index(c) for c in t.label])
            P = t[[f"p_{c}" for c in CLASSES]].to_numpy()
            runs[v] = (y, P.argmax(1), P)
    if not a.no_baselines:
        runs.update(baselines(root, man))
    if not runs:
        sys.exit("Nothing to evaluate: run stage3_classifier.py first")

    rows, recs, aucs = [], {}, {}
    for name, (y, yhat, P) in runs.items():
        m, rec, auc, cm = evaluate(y, yhat, P)
        rows.append(dict(model=name, **m, **{f"paper_{k}": v for k, v in PAPER.get(name, {}).items()}))
        recs[name], aucs[name] = rec, auc
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / f"confusion_{name}.csv")

    metrics = pd.DataFrame(rows).set_index("model")
    recall = pd.DataFrame(recs, index=CLASSES).T
    auc = pd.DataFrame(aucs, index=CLASSES).T
    metrics.to_csv(out / "metrics.csv"); recall.to_csv(out / "recall.csv"); auc.to_csv(out / "auc.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== Test set: yours vs paper (Table 3) ===")
    show = metrics[["acc", "acc_ci", "kappa", "rci", "mcc", "mcc_ci"]].round(3)
    for k in ["acc", "kappa", "rci", "mcc"]:
        if f"paper_{k}" in metrics:
            show[f"paper_{k}"] = metrics[f"paper_{k}"]
    print(show)
    print("\n=== Per-class recall (Table 4) ===")
    print(recall.round(3))
    print("\n=== One-vs-rest AUC (Fig. 9) ===")
    print(auc.round(3))
    print("\nRare test classes have only 4-5 images: one error moves recall by 0.20-0.25.")
    print(f"Saved CSVs to {out}")

    if a.cam:
        cam(root, a.cam_variant, a.cam, a.weights)


if __name__ == "__main__":
    main()
