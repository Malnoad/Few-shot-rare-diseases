"""T12 bounded cached-head equivalence/behaviour/timing experiment (no sweep)."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from fast_head import EpochLoop, make_model
from stage3_classifier import CLASSES, MAJOR, RARE, fix


def keras_shuffle_probe():
    """Record actual batch identities in a tiny compiled native fit."""
    tf.keras.utils.set_random_seed(1)
    model = make_model(tf, width=1)
    orders = []
    original = model.train_step

    def remember(x):
        orders.extend(int(v) for v in x[:, 0].numpy())

    def record(data):
        tf.py_function(remember, [data[0]], [])
        return original(data)

    model.train_step = record
    model.fit(np.arange(23, dtype=np.float32)[:, None], np.zeros(23, dtype=np.int64),
              epochs=3, batch_size=10, shuffle=True, verbose=0)
    epochs = [orders[i:i+23] for i in range(0, len(orders), 23)]
    return dict(compiled=True, orders=epochs, repeated=epochs[0] == epochs[1] == epochs[2])


def cached_data(cache, manifest, seed):
    """Reproduce Stage 3 gan's seed-specific augmented split from existing caches."""
    man = pd.read_csv(manifest)
    man["path"] = man.path.map(fix)
    def load(group):
        with np.load(cache / f"{group}.npz", allow_pickle=False) as z:
            return z["feats"].astype(np.float32), z["paths"].tolist()
    tr, tr_paths = load("major_train")
    va, va_paths = load("major_val")
    te, paths = load("test")
    major = man[man.label.isin(MAJOR)]
    if (tr_paths != major[major.split == "train"].path.tolist() or
            va_paths != major[major.split == "val"].path.tolist() or
            paths != man[man.split == "test"].path.tolist()):
        raise ValueError("Cache path order differs from the committed manifest")
    yt = major[major.split == "train"].label.tolist()
    yv = major[major.split == "val"].label.tolist()
    xs, vs = [tr], [va]
    for cls in RARE:
        pool = np.concatenate([load(f"basic_{cls}")[0], load(f"gan_{cls}")[0]])
        idx = np.random.default_rng(seed).permutation(len(pool))
        n = len(pool) // 10
        vs.append(pool[idx[:n]])
        xs.append(pool[idx[n:]])
        yv += [cls] * n
        yt += [cls] * (len(pool) - n)
    labels = man.set_index("path").label
    encode = lambda values: np.array([CLASSES.index(c) for c in values])
    return np.concatenate(xs), encode(yt), np.concatenate(vs), encode(yv), te, encode(labels.loc[paths])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    pids = Path(".agents/pids")
    pids.mkdir(parents=True, exist_ok=True)
    (pids / "t12-benchmark.pid").write_text(str(os.getpid()))
    result = dict(tensorflow=tf.__version__, keras=tf.keras.__version__, seeds=[], candidates=[])
    result["environment"] = dict(platform=platform.platform(), processor=platform.processor(),
                                 cache=str(a.cache), manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                                 batch=10, lr=.001, epochs_measured=25,
                                 timing="fit and validation, including tracing; excludes cache loading, prediction and checkpoint/Drive I/O")
    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    x, y, vx, vy, tx, ty = cached_data(a.cache, a.manifest, 1)
    # Entire training epoch plus three repeated rows forces a final partial batch.
    gx, gy = np.concatenate([x, x[:3]]), np.concatenate([y, y[:3]])
    tf.keras.utils.set_random_seed(42)
    ref, fast = make_model(tf), make_model(tf)
    fast.set_weights(ref.get_weights())
    order = np.random.default_rng(73).permutation(len(gy)).astype(np.int32)
    rh = ref.fit(gx[order], gy[order], validation_data=(vx, vy), batch_size=10,
                 shuffle=False, epochs=1, verbose=0).history
    loop = EpochLoop(tf, fast, gx, gy, vx, vy)
    fh = loop.epoch(order)
    dw = max(float(np.max(np.abs(a - b))) for a, b in zip(ref.get_weights(), fast.get_weights()))
    dv = abs(rh["val_loss"][0] - fh["val_loss"])
    slots = max(float(np.max(np.abs(a.numpy() - b.numpy())))
                for a, b in zip(ref.optimizer.variables, fast.optimizer.variables))
    result["gate"] = dict(rows=len(gy), max_weight_difference=dw, val_loss_difference=dv,
                          max_optimizer_difference=slots, keras=rh, fast=fh,
                          passed=dw < 1e-5 and dv < 1e-6)
    save()
    print("GATE", result["gate"], flush=True)
    if not result["gate"]["passed"]:
        raise SystemExit("Equivalence gate failed; do not wire fast into experiments")
    for jit in (False,):
        model = make_model(tf)
        try:
            loop = EpochLoop(tf, model, x, y, vx, vy, jit=jit)
            start = time.perf_counter()
            loop.epoch()
            cold = time.perf_counter() - start
            start = time.perf_counter()
            loop.epoch()
            result["candidates"].append(dict(jit=jit, cold_seconds=cold, warm_seconds=time.perf_counter()-start))
        except (tf.errors.OpError, ValueError) as error:
            result["candidates"].append(dict(jit=jit, unavailable=str(error)))
        save()
        print("CANDIDATE", result["candidates"][-1], flush=True)
    for seed in (1, 2, 3):
        x, y, vx, vy, tx, ty = cached_data(a.cache, a.manifest, seed)
        row = dict(seed=seed)
        for trainer in ("keras", "fast"):
            tf.keras.utils.set_random_seed(seed)
            model = make_model(tf)
            start = time.perf_counter()
            if trainer == "keras":
                history = model.fit(x, y, validation_data=(vx, vy), batch_size=10,
                                    epochs=25, shuffle=True, verbose=2).history
            else:
                loop = EpochLoop(tf, model, x, y, vx, vy)
                logs = []
                for epoch in range(25):
                    logs.append(loop.epoch())
                    print("fast", seed, epoch + 1, logs[-1], flush=True)
                history = {k: [log[k] for log in logs] for k in logs[0]}
            elapsed = time.perf_counter() - start
            accuracy = float(np.mean(model(tx, training=False).numpy().argmax(1) == ty) * 100)
            row[trainer] = dict(seconds_25=elapsed, projected_250_seconds=elapsed * 10,
                                test_accuracy_percent=accuracy, history=history)
            print("RESULT", seed, trainer, elapsed, accuracy, flush=True)
        result["seeds"].append(row)
        save()
    result["keras_shuffle_probe"] = keras_shuffle_probe()
    save()


if __name__ == "__main__":
    main()
