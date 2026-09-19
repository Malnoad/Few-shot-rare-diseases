"""
Stage 3 - 9-class Inception-v3 classifier  (GAN-FSL-OCT reproduction, Yoo et al. MBEC 2021)

Paper setup: ImageNet Inception-v3 with frozen convolutional layers, only the final fully
connected (softmax) layer trained; input 299x299x3; Adam; categorical cross-entropy;
batch 10; 250 epochs; a tenth of the training data used for validation.

Because the backbone is frozen and no on-the-fly augmentation is used, backbone features
are extracted once and cached; training the softmax layer on them gives the same model as
an end-to-end run, in minutes instead of the paper's ~150 h.

All images go through the same 256x256 step before 299x299 (every rare training image
was produced at 256x256 in Stages 1-2), so image sharpness cannot reveal the class.

Variants (rows of the paper's Table 3):
  none   rare train = real images only (19-27 per class)
  basic  rare train = 2,000 basic-augmented images (Stage 1)
  gan    rare train = 2,000 basic + 3,000 accepted CycleGAN images (Stage 2)   <- proposed
Major classes are identical in all variants. Rare-class validation = 10% of the augmented
pool (basic / gan); the "none" variant has no rare validation images.

Usage (from the project root):
  python stage3_classifier.py --variant none
  python stage3_classifier.py --variant basic
  python stage3_classifier.py --variant gan
Options: --epochs 250 --batch 10 --patience 20 (0 = run all epochs, as in the paper)

Outputs:
  results/features/<group>.npz                  cached features, shared by variants and Stage 4
  results/stage3_<variant>/head.npz             trained softmax layer (W, b, classes)
  results/stage3_<variant>/history.csv, data_counts.csv
  results/stage3_<variant>/{val,test}_predictions.csv
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["NORMAL", "CNV", "DME", "DRUSEN", "CSC", "MH", "MacTel", "Stargardt", "RP"]
MAJOR, RARE = CLASSES[:4], CLASSES[4:]
tf = None


def fix(p):
    return str(p).replace("\\", "/")


def need_tf():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    return tf


def load_image(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, [256, 256])
    img = tf.image.resize(img, [299, 299])
    return img / 127.5 - 1.0  # == inception_v3.preprocess_input


def backbone(weights="imagenet"):
    need_tf()
    return tf.keras.applications.InceptionV3(include_top=False, weights=None if weights == "None" else weights,
                                             pooling="avg",
                                             input_shape=(299, 299, 3))


def features(group, paths, root, cache_dir, net_fn, bs=32):
    """Features for `paths` (relative to root), cached in results/features/<group>.npz."""
    f = cache_dir / f"{group}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if z["paths"].tolist() == list(paths):
            return z["feats"].astype(np.float32)
        print(f"  cache {f.name} is stale (image list changed) -> re-extracting")
    if not paths:
        return np.zeros((0, 2048), np.float32)

    net = net_fn()
    AT = tf.data.AUTOTUNE
    ds = (tf.data.Dataset.from_tensor_slices([str(root / p) for p in paths])
          .map(load_image, num_parallel_calls=AT).batch(bs).prefetch(2))
    out, t0, n_batches = [], time.time(), (len(paths) + bs - 1) // bs
    for i, b in enumerate(ds, 1):
        out.append(net(b, training=False).numpy().astype(np.float16))
        if i % 50 == 0 or i == n_batches:
            rate = i * bs / (time.time() - t0)
            print(f"  {group}: {min(i * bs, len(paths))}/{len(paths)} images "
                  f"({rate:.0f} img/s, ~{max(len(paths) - i * bs, 0) / rate / 60:.0f} min left)", flush=True)
    feats = np.concatenate(out)
    np.savez(f, paths=np.array(paths), feats=feats)
    return feats.astype(np.float32)


def rel_images(root, folder):
    d = root / folder
    return sorted(f"{folder}/{p.name}" for p in d.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) \
        if d.exists() else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--variant", required=True, choices=["none", "basic", "gan"])
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20, help="early stopping on val loss; 0 = off")
    ap.add_argument("--gan-per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat-batch", type=int, default=32, help="lower if feature extraction runs out of memory")
    ap.add_argument("--weights", default="imagenet", help=argparse.SUPPRESS)  # testing only
    a = ap.parse_args()

    need_tf()
    tf.keras.utils.set_random_seed(a.seed)
    root = Path(a.root)
    cache = root / "results/features"
    cache.mkdir(parents=True, exist_ok=True)
    out = root / f"results/stage3_{a.variant}"
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(root / "data/manifest.csv")
    man["path"] = man.path.map(fix)
    net_holder = {}

    def net_fn():
        if "net" not in net_holder:
            net_holder["net"] = backbone(a.weights)
        return net_holder["net"]

    def feats_for(group, frame):
        return features(group, frame.path.tolist(), root, cache, net_fn, a.feat_batch), frame.label.tolist()

    # ---- major classes (identical in every variant) and the test set
    maj = man[man.label.isin(MAJOR)]
    Xtr, ytr = feats_for("major_train", maj[maj.split == "train"])
    Xva, yva = feats_for("major_val", maj[maj.split == "val"])
    test = man[man.split == "test"]
    Xte, yte = feats_for("test", test)

    # ---- rare classes
    Xtr, ytr, Xva, yva = [Xtr], list(ytr), [Xva], list(yva)
    rng = random.Random(a.seed)
    for cls in RARE:
        if a.variant == "none":
            Xr, yr = feats_for(f"real_{cls}", man[(man.label == cls) & (man.split == "train")])
            Xtr.append(Xr); ytr += yr
            continue
        basic = rel_images(root, f"data/aug_basic/{cls}")
        gan = rel_images(root, f"data/gan/accepted/{cls}") if a.variant == "gan" else []
        if len(basic) < 2000:
            sys.exit(f"data/aug_basic/{cls} has {len(basic)} images: run Stage 1")
        if a.variant == "gan":
            if len(gan) < a.gan_per_class:
                print(f"WARNING {cls}: only {len(gan)} accepted GAN images (paper: {a.gan_per_class})")
            gan = sorted(rng.sample(gan, min(len(gan), a.gan_per_class)))
        Xb = features(f"basic_{cls}", basic, root, cache, net_fn, a.feat_batch)
        Xg = features(f"gan_{cls}", gan, root, cache, net_fn, a.feat_batch) if gan else np.zeros((0, 2048), np.float32)
        pool = np.concatenate([Xb, Xg])
        idx = np.random.default_rng(a.seed).permutation(len(pool))
        n_val = len(pool) // 10
        Xva.append(pool[idx[:n_val]]); yva += [cls] * n_val
        Xtr.append(pool[idx[n_val:]]); ytr += [cls] * (len(pool) - n_val)

    Xtr, Xva = np.concatenate(Xtr), np.concatenate(Xva)
    cidx = {c: i for i, c in enumerate(CLASSES)}
    ytr_i = np.array([cidx[c] for c in ytr]); yva_i = np.array([cidx[c] for c in yva])
    yte_i = np.array([cidx[c] for c in yte])

    counts = pd.DataFrame({s: pd.Series(y).value_counts() for s, y in
                           [("train", ytr), ("val", yva), ("test", yte)]}).reindex(CLASSES).fillna(0).astype(int)
    counts.to_csv(out / "data_counts.csv")
    print(f"\nVariant '{a.variant}' data counts:\n{counts}\n")

    # ---- the trainable part: one softmax layer on frozen 2048-d features
    model = tf.keras.Sequential([tf.keras.Input((2048,)), tf.keras.layers.Dense(len(CLASSES), activation="softmax")])
    model.compile(tf.keras.optimizers.Adam(a.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    best = out / "best.weights.h5"
    cbs = [tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_loss", save_best_only=True,
                                              save_weights_only=True),
           tf.keras.callbacks.CSVLogger(str(out / "history.csv"))]
    if a.patience > 0:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience))
    model.fit(Xtr, ytr_i, validation_data=(Xva, yva_i), epochs=a.epochs, batch_size=a.batch,
              shuffle=True, callbacks=cbs, verbose=2)
    model.load_weights(str(best))
    W, b = model.layers[-1].get_weights()
    np.savez(out / "head.npz", W=W, b=b, classes=np.array(CLASSES))

    # ---- predictions
    for name, X, y_i, paths in [("val", Xva, yva_i, None), ("test", Xte, yte_i, test.path.tolist())]:
        P = model.predict(X, batch_size=1024, verbose=0)
        d = pd.DataFrame(P, columns=[f"p_{c}" for c in CLASSES])
        d.insert(0, "pred", [CLASSES[i] for i in P.argmax(1)])
        d.insert(0, "label", [CLASSES[i] for i in y_i])
        if paths is not None:
            d.insert(0, "path", paths)
        d.to_csv(out / f"{name}_predictions.csv", index=False)

    t = pd.read_csv(out / "test_predictions.csv")
    rec = t.assign(ok=t.label == t.pred).groupby("label").ok.mean().reindex(CLASSES)
    print(f"\nTest accuracy ({a.variant}): {(t.label == t.pred).mean() * 100:.1f}%  "
          f"(paper: none 88.4, basic 91.3, gan 92.1)")
    print("Per-class recall:", "  ".join(f"{c}={v:.2f}" for c, v in rec.items()))
    print(f"Saved to {out}. Full metrics: python stage4_evaluate.py")


if __name__ == "__main__":
    main()
