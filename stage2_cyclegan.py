"""
Stage 2 of the GAN-FSL-OCT reproduction (Yoo et al., Med Biol Eng Comput 2021):
one CycleGAN per rare disease, translating NORMAL OCT -> rare-disease OCT.

Architecture and hyperparameters follow the TensorFlow CycleGAN tutorial the authors
used: U-Net generator + PatchGAN discriminator with instance norm, 256x256x3 input,
Adam(2e-4, beta1=0.5), batch 1, cycle lambda 10, identity 0.5*lambda, tutorial jitter
(resize 286 -> random crop 256 -> random horizontal flip). Self-contained: no
tensorflow_examples dependency, works with tf.keras 2 and Keras 3.

Subcommands (run from the project root, or pass --root):
  prep      LOCAL, no TensorFlow needed. Builds data/gan/{normalA, normal_src, real/<cls>}
  train     trains or resumes one disease           --disease CSC [--epochs 100]
            on Google Drive use e.g. --save-every 10 --keep 1 --light: files deleted from
            Drive go to its trash and still count toward storage until the trash is emptied.
            Speed-ups for GPU: --mixed (float16 compute) and --xla (compiled training step)
  generate  translates normal_src with trained G    --disease CSC [--n 4000]
  qc        feature-based filter + contact sheets   --disease CSC [--keep 3000]

Inputs:
  data/manifest.csv             from the manifest step
  data/aug_basic/<cls>/*.png    from Stage 1 (2,000 per class)
Outputs:
  data/gan/ckpt/<cls>/          checkpoints (resume automatically) + samples/ep_XXX.png
  data/gan/out/<cls>/           raw generated images
  data/gan/accepted/<cls>/      images that passed QC (goes to Stage 3)
"""
import argparse
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

RARE = ["CSC", "MH", "MacTel", "Stargardt", "RP"]
IMG = 256
LAMBDA = 10.0
tf = None  # imported lazily so `prep` runs without TensorFlow


def need_tf():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    return tf


def fix(p):
    """Manifest paths were written on Windows; make them portable."""
    return str(p).replace("\\", "/")


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


# --------------------------------------------------------------------------- prep
def prep(a):
    import cv2

    root = Path(a.root)
    df = pd.read_csv(root / "data/manifest.csv")
    gan = root / "data/gan"

    normals = sorted(df[(df.label == "NORMAL") & (df.split == "train")].path.map(fix))
    random.Random(a.seed).shuffle(normals)
    sets = {
        "normalA": normals[: a.n_domain],                                  # CycleGAN domain A
        "normal_src": normals[a.n_domain : a.n_domain + a.n_source],       # fresh inputs for generation
    }
    for cls in RARE:  # real rare TRAIN images only (test hold-out never touches the GAN)
        sets[f"real/{cls}"] = sorted(df[(df.label == cls) & (df.split == "train")].path.map(fix))

    for name, files in sets.items():
        out = gan / name
        out.mkdir(parents=True, exist_ok=True)
        for p in files:
            img = cv2.imread(str(root / p))
            if img is None:
                print("  unreadable, skipped:", p)
                continue
            cv2.imwrite(str(out / (Path(p).stem + ".png")), cv2.resize(img, (IMG, IMG)))
        print(f"{name:16s} {len(list_images(out)):5d} images")

    for cls in RARE:
        n = len(list_images(root / "data/aug_basic" / cls))
        print(f"aug_basic/{cls:10s} {n:5d} images" + ("" if n >= 2000 else "   <-- expected 2000, rerun Stage 1"))


# --------------------------------------------------------------------------- data
def read_img(path, size=IMG):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, [size, size])
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def to_float(img):
    return tf.cast(img, tf.float32) / 127.5 - 1.0


def jitter(img):
    img = tf.image.resize(img, [286, 286], method="nearest")
    img = tf.image.random_crop(img, [IMG, IMG, 3])
    return tf.image.random_flip_left_right(img)


def train_ds(files):
    AT = tf.data.AUTOTUNE
    return (
        tf.data.Dataset.from_tensor_slices(files)
        .map(read_img, num_parallel_calls=AT)
        .cache()  # uint8 in RAM (~400 MB per 2,000 images): disk/Drive is read only in epoch 1
        .shuffle(len(files), reshuffle_each_iteration=True)
        .map(lambda x: jitter(to_float(x)), num_parallel_calls=AT)
        .batch(1)
        .prefetch(AT)
    )


# --------------------------------------------------------------------------- models
def build_models():
    need_tf()
    L = tf.keras.layers
    init = tf.random_normal_initializer(0.0, 0.02)

    class InstanceNorm(L.Layer):
        def build(self, shape):
            self.scale = self.add_weight(name="scale", shape=(shape[-1],),
                                         initializer=tf.random_normal_initializer(1.0, 0.02))
            self.offset = self.add_weight(name="offset", shape=(shape[-1],), initializer="zeros")

        def call(self, x):
            x32 = tf.cast(x, tf.float32)
            mean, var = tf.nn.moments(x32, axes=[1, 2], keepdims=True)
            y = (tf.cast(self.scale, tf.float32) * (x32 - mean) * tf.math.rsqrt(var + 1e-5)
                 + tf.cast(self.offset, tf.float32))
            return tf.cast(y, x.dtype)

    def down(x, f, norm=True):
        x = L.Conv2D(f, 4, 2, padding="same", kernel_initializer=init, use_bias=False)(x)
        if norm:
            x = InstanceNorm()(x)
        return L.LeakyReLU()(x)

    def up(x, f, drop=False):
        x = L.Conv2DTranspose(f, 4, 2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = InstanceNorm()(x)
        if drop:
            x = L.Dropout(0.5)(x)
        return L.ReLU()(x)

    def generator(name):  # pix2pix U-Net, as in the tutorial
        inp = L.Input((IMG, IMG, 3))
        x, skips = inp, []
        for i, f in enumerate([64, 128, 256, 512, 512, 512, 512, 512]):
            x = down(x, f, norm=i > 0)
            skips.append(x)
        for i, (f, s) in enumerate(zip([512, 512, 512, 512, 256, 128, 64], reversed(skips[:-1]))):
            x = L.Concatenate()([up(x, f, drop=i < 3), s])
        out = L.Conv2DTranspose(3, 4, 2, padding="same", kernel_initializer=init, activation="tanh",
                                 dtype="float32")(x)
        return tf.keras.Model(inp, out, name=name)

    def discriminator(name):  # 70x70 PatchGAN
        inp = L.Input((IMG, IMG, 3))
        x = down(inp, 64, norm=False)
        x = down(x, 128)
        x = down(x, 256)
        x = L.ZeroPadding2D()(x)
        x = L.Conv2D(512, 4, 1, kernel_initializer=init, use_bias=False)(x)
        x = L.LeakyReLU()(InstanceNorm()(x))
        x = L.ZeroPadding2D()(x)
        out = L.Conv2D(1, 4, 1, kernel_initializer=init, dtype="float32")(x)
        return tf.keras.Model(inp, out, name=name)

    return (generator("G_normal2rare"), generator("F_rare2normal"),
            discriminator("D_normal"), discriminator("D_rare"))


def ckpt_dir(a):
    return Path(a.ckpt) if a.ckpt else Path(a.root) / "data/gan/ckpt" / a.disease


# --------------------------------------------------------------------------- train
def train(a):
    need_tf()
    root = Path(a.root)
    A = list_images(root / "data/gan/normalA")
    B = list_images(root / "data/aug_basic" / a.disease)
    assert A, "data/gan/normalA is empty: run `prep` first"
    assert B, f"data/aug_basic/{a.disease} is empty: run Stage 1 first"
    ck_dir = ckpt_dir(a)
    (ck_dir / "samples").mkdir(parents=True, exist_ok=True)

    if a.mixed:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("mixed precision: float16 compute, float32 weights", flush=True)
    G, F, DA, DB = build_models()
    opts = [tf.keras.optimizers.Adam(2e-4, beta_1=0.5) for _ in range(4)]
    if a.mixed:
        opts = [tf.keras.mixed_precision.LossScaleOptimizer(o) for o in opts]
    legacy_ls = a.mixed and not hasattr(opts[0], "scale_loss")

    def scaled(o, loss):
        if not a.mixed:
            return loss
        return o.get_scaled_loss(loss) if legacy_ls else o.scale_loss(loss)
    for o, m in zip(opts, (G, F, DA, DB)):
        if hasattr(o, "build"):
            o.build(m.trainable_variables)  # create slots now so resume restores them

    epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    if a.light:  # weights + epoch only (~0.46 GB instead of ~1.4 GB); Adam restarts on resume
        ck = tf.train.Checkpoint(G=G, F=F, DA=DA, DB=DB, epoch=epoch)
    else:
        ck = tf.train.Checkpoint(G=G, F=F, DA=DA, DB=DB, oG=opts[0], oF=opts[1], oDA=opts[2], oDB=opts[3],
                                 epoch=epoch)
    mgr = tf.train.CheckpointManager(ck, str(ck_dir), max_to_keep=a.keep)
    if mgr.latest_checkpoint:
        try:
            ck.restore(mgr.latest_checkpoint).expect_partial()
        except (tf.errors.InvalidArgumentError, ValueError):  # optimizer layout differs, e.g. --mixed toggled
            tf.train.Checkpoint(G=G, F=F, DA=DA, DB=DB, epoch=epoch).restore(mgr.latest_checkpoint).expect_partial()
            print("  optimizer state not restored (checkpoint from a different precision mode)", flush=True)
        print(f"Resumed {a.disease} from {mgr.latest_checkpoint} (epoch {int(epoch.numpy())})", flush=True)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def l1(x, y):
        return tf.reduce_mean(tf.abs(x - y))

    @tf.function(jit_compile=a.xla)
    def step(a_img, b_img):
        with tf.GradientTape(persistent=True) as tape:
            fake_b = G(a_img, training=True)
            cyc_a = F(fake_b, training=True)
            fake_a = F(b_img, training=True)
            cyc_b = G(fake_a, training=True)
            same_a = F(a_img, training=True)
            same_b = G(b_img, training=True)
            d_ra, d_fa = DA(a_img, training=True), DA(fake_a, training=True)
            d_rb, d_fb = DB(b_img, training=True), DB(fake_b, training=True)

            cyc = LAMBDA * (l1(a_img, cyc_a) + l1(b_img, cyc_b))
            loss_G = bce(tf.ones_like(d_fb), d_fb) + cyc + 0.5 * LAMBDA * l1(b_img, same_b)
            loss_F = bce(tf.ones_like(d_fa), d_fa) + cyc + 0.5 * LAMBDA * l1(a_img, same_a)
            loss_DA = 0.5 * (bce(tf.ones_like(d_ra), d_ra) + bce(tf.zeros_like(d_fa), d_fa))
            loss_DB = 0.5 * (bce(tf.ones_like(d_rb), d_rb) + bce(tf.zeros_like(d_fb), d_fb))
            scaled_losses = [scaled(o, l) for o, l in zip(opts, (loss_G, loss_F, loss_DA, loss_DB))]
        for sl, m, o in zip(scaled_losses, (G, F, DA, DB), opts):
            g = tape.gradient(sl, m.trainable_variables)
            if legacy_ls:
                g = o.get_unscaled_gradients(g)
            o.apply_gradients(zip(g, m.trainable_variables))
        return tf.stack([loss_G, loss_F, loss_DA, loss_DB])

    fixed = tf.stack([to_float(read_img(p)) for p in A[:4]])

    def save_sample(ep):  # left: real NORMAL, right: G(normal) -> rare
        out = G(fixed, training=False)
        grid = tf.concat([tf.concat([fixed[i], out[i]], axis=1) for i in range(fixed.shape[0])], axis=0)
        png = tf.cast(tf.clip_by_value((grid + 1) * 127.5, 0, 255), tf.uint8)
        tf.io.write_file(str(ck_dir / "samples" / f"ep_{ep:03d}.png"), tf.io.encode_png(png))

    dsA, dsB = train_ds(A), train_ds(B)
    steps = min(len(A), len(B))
    print(f"{a.disease}: {len(A)} normal x {len(B)} rare, {steps} steps/epoch, "
          f"epochs {int(epoch.numpy())}->{a.epochs}", flush=True)

    for ep in range(int(epoch.numpy()), a.epochs):
        t0, acc = time.time(), tf.zeros(4)
        for i, (x, y) in enumerate(tf.data.Dataset.zip((dsA, dsB)), 1):
            acc += step(x, y)
            if i % 500 == 0:
                m = (acc / i).numpy()
                print(f"  ep {ep + 1} step {i}/{steps}  G={m[0]:.3f} F={m[1]:.3f} "
                      f"D_normal={m[2]:.3f} D_rare={m[3]:.3f}", flush=True)
        epoch.assign(ep + 1)
        save_sample(ep + 1)
        saved = (ep + 1) % a.save_every == 0 or ep + 1 == a.epochs
        if saved:
            mgr.save()
        print(f"epoch {ep + 1}/{a.epochs} done in {(time.time() - t0) / 60:.1f} min"
              + (" | checkpoint saved" if saved else ""), flush=True)


# --------------------------------------------------------------------------- generate
def generate(a):
    need_tf()
    root = Path(a.root)
    latest = tf.train.latest_checkpoint(str(ckpt_dir(a)))
    assert latest, f"no checkpoint in {ckpt_dir(a)}: train first"
    G = build_models()[0]
    tf.train.Checkpoint(G=G).restore(latest).expect_partial()

    src = list_images(root / "data/gan/normal_src")
    if a.n > len(src):
        print(f"only {len(src)} source images available (asked for {a.n}); rerun prep with a larger --n-source")
    src = src[: a.n]
    out = root / "data/gan/out" / a.disease
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(src), 16):
        chunk = src[i : i + 16]
        y = G(tf.stack([to_float(read_img(p)) for p in chunk]), training=False)
        y = tf.cast(tf.clip_by_value((y + 1) * 127.5, 0, 255), tf.uint8)
        for p, img in zip(chunk, y):
            tf.io.write_file(str(out / f"{a.disease}_gan_{Path(p).stem}.png"), tf.io.encode_png(img))
    print(f"wrote {len(src)} images to {out} in {time.time() - t0:.0f}s (checkpoint {latest})")


# --------------------------------------------------------------------------- qc
def qc(a):
    """Proxy for the paper's ophthalmologist review (you must still eyeball the sheets).

    Keeps a synthetic image only if its Inception-v3 features are closer (mean cosine
    distance to 3 nearest neighbours) to REAL training images of the disease than to
    real NORMAL images, i.e. the translation actually happened, then keeps the `keep`
    closest ones.
    """
    need_tf()
    import cv2

    gan = Path(a.root) / "data/gan"
    fake = list_images(gan / "out" / a.disease)
    real = list_images(gan / "real" / a.disease)
    normal = list_images(gan / "normalA")[:500]
    assert fake, "no generated images: run generate first"
    assert real, "no real rare images in data/gan/real: run prep first"

    net = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling="avg")

    def feats(files):
        out = []
        for i in range(0, len(files), 32):
            x = tf.stack([to_float(read_img(p, 299)) for p in files[i : i + 32]])  # == Inception preprocessing
            out.append(net(x, training=False).numpy())
        f = np.concatenate(out)
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    Ff, Fr, Fn = feats(fake), feats(real), feats(normal)

    def knn_dist(S, k=3):
        return 1 - np.sort(S, axis=1)[:, -k:].mean(axis=1)

    df = pd.DataFrame({"file": fake, "d_rare": knn_dist(Ff @ Fr.T), "d_normal": knn_dist(Ff @ Fn.T)})
    df["pass"] = df.d_rare < df.d_normal
    kept = df[df["pass"]].sort_values("d_rare").head(a.keep)
    df["kept"] = df.file.isin(kept.file)
    df.to_csv(gan / f"qc_{a.disease}.csv", index=False)

    acc_dir = gan / "accepted" / a.disease
    if acc_dir.exists():
        shutil.rmtree(acc_dir)
    acc_dir.mkdir(parents=True)
    for p in kept.file:
        shutil.copy(p, acc_dir)

    def sheet(files, name):
        files = list(files)[:48]
        if not files:
            return
        tiles = [cv2.resize(cv2.imread(p), (128, 128)) for p in files]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
        cv2.imwrite(str(gan / name), np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]))

    sheet(np.random.default_rng(0).permutation(kept.file.values), f"qc_{a.disease}_kept.jpg")
    sheet(df[~df.kept].sort_values("d_rare", ascending=False).file, f"qc_{a.disease}_rejected.jpg")

    print(f"{a.disease}: {int(df['pass'].sum())}/{len(df)} closer to real {a.disease} than to NORMAL; "
          f"kept {len(kept)} -> {acc_dir}")
    print(f"  review {gan / f'qc_{a.disease}_kept.jpg'} and delete bad files from {acc_dir}")
    if len(kept) < a.keep:
        print(f"  only {len(kept)} < {a.keep}: generate more (larger --n; prep with larger --n-source if needed)")


# --------------------------------------------------------------------------- cli
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".", help="project root containing data/ (default: current folder)")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", parents=[common])
    p.add_argument("--n-domain", type=int, default=2000, help="NORMAL images for CycleGAN domain A")
    p.add_argument("--n-source", type=int, default=5000, help="separate NORMAL images to translate later")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train", parents=[common])
    p.add_argument("--disease", required=True, choices=RARE)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--xla", action="store_true", help="XLA-compile the training step (faster on GPU)")
    p.add_argument("--mixed", action="store_true", help="mixed float16 precision (faster on tensor-core GPUs such as T4)")
    p.add_argument("--ckpt", help="checkpoint folder (default data/gan/ckpt/<disease>)")
    p.add_argument("--save-every", type=int, default=1,
                   help="write a checkpoint every N epochs (and at the last epoch); samples are saved every epoch")
    p.add_argument("--keep", type=int, default=2, help="checkpoints to keep; older ones are deleted")
    p.add_argument("--light", action="store_true",
                   help="save weights + epoch only (no optimizer state): ~3x smaller checkpoints")

    p = sub.add_parser("generate", parents=[common])
    p.add_argument("--disease", required=True, choices=RARE)
    p.add_argument("--n", type=int, default=4000, help="images to generate (extra to survive QC)")
    p.add_argument("--ckpt")

    p = sub.add_parser("qc", parents=[common])
    p.add_argument("--disease", required=True, choices=RARE)
    p.add_argument("--keep", type=int, default=3000)

    a = ap.parse_args()
    {"prep": prep, "train": train, "generate": generate, "qc": qc}[a.cmd](a)


if __name__ == "__main__":
    main()
