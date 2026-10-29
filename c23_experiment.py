"""C2 GAN amount and C3 common-class fitting caps, for manual Colab execution.

prepare extracts only the new v2 GAN features; run fits cached heads. All writes
use explicit experiment destinations outside the checkout. No canonical results
are imported as completed heads. See docs/research/C2_C3_DESIGN.md.
"""
import argparse
from contextlib import contextmanager
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import matthews_corrcoef

from c0_experiment import validate_predictions, verify_cache
from c1_experiment import ARTIFACTS, lock_json, protected_hashes, read_json
from c5_topup import checked_copy, inventory
from stage1b_normalize import atomic_json, sha256
import stage3_classifier as s3

SCRIPT_ROOT = Path(__file__).resolve().parent
major_indices = s3.major_indices
SEEDS = tuple(range(1, 21))
AMOUNTS = (0, 500, 1000, 2000, 3000)
CAPS = (None, 10000, 5000)


def schedule(experiment):
    if experiment == "c2":
        return [(seed, f"n{n:04d}", n, None) for seed in SEEDS for n in AMOUNTS]
    if experiment == "c3":
        return [(seed, "full" if cap is None else f"cap{cap}", 3000, cap)
                for seed in SEEDS for cap in CAPS]
    raise ValueError("experiment must be c2 or c3")


def base_groups(root):
    man = pd.read_csv(Path(root) / "data/manifest.csv")
    man["path"] = man.path.map(s3.fix)
    major = man[man.label.isin(s3.MAJOR)]
    groups = {f"major_{s}": major[major.split == s].path.tolist() for s in ("train", "val")}
    groups["test"] = man[man.split == "test"].path.tolist()
    groups.update({f"basic_{c}": s3.rel_images(Path(root), f"data/aug_basic/{c}") for c in s3.RARE})
    if any(len(groups[f"basic_{c}"]) != 2000 for c in s3.RARE):
        raise ValueError("Need exactly the fixed 2000 basic images per rare class")
    return groups


def full_groups(root):
    groups = base_groups(root)
    groups.update({s3.gan_cache_group(c, "v2"): s3.gan_images(Path(root), c, "v2") for c in s3.RARE})
    return groups


def selected_gan(groups, amount, seed):
    """Match Stage 3's existing sample calls and sorted subset order exactly."""
    if amount not in AMOUNTS:
        raise ValueError("unplanned GAN amount")
    rng = random.Random(seed)
    return {s3.gan_cache_group(c, "v2"): sorted(rng.sample(groups[s3.gan_cache_group(c, "v2")], amount))
            for c in s3.RARE}


def selection_record(root, groups, amount, cap, seed):
    """Exact ordered fitting/validation paths, including the pooled rare split."""
    man = pd.read_csv(Path(root) / "data/manifest.csv")
    major = man[man.label.isin(s3.MAJOR) & (man.split == "train")]
    chosen = selected_gan(groups, amount, seed)
    rare = {}
    for c in s3.RARE:
        pool = groups[f"basic_{c}"] + chosen[s3.gan_cache_group(c, "v2")]
        idx = np.random.default_rng(seed).permutation(len(pool))
        n_val = len(pool) // 10
        rare[c] = dict(train=[pool[i] for i in idx[n_val:]], val=[pool[i] for i in idx[:n_val]])
    return dict(common_train=major.iloc[major_indices(major.label, cap, seed)].path.map(s3.fix).tolist(),
                common_val=groups["major_val"], test=groups["test"], rare=rare)


def save_selection(path, value):
    pending = path.with_suffix(".tmp")
    with gzip.open(pending, "wt", encoding="utf-8") as f:
        json.dump(value, f)
    pending.replace(path)


def outside_checkout(path):
    path = Path(path).resolve()
    if path.is_relative_to(SCRIPT_ROOT) or SCRIPT_ROOT.is_relative_to(path):
        raise ValueError("Experiment destination must be outside the checkout")
    return path


@contextmanager
def writer_lock(folder):
    """Reject concurrent writers. A hard disconnect requires explicit token unlock."""
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "writer.lock"
    token = uuid.uuid4().hex
    value = dict(token=token, host=socket.gethostname(), pid=os.getpid())
    try:
        with lock.open("x", encoding="utf-8") as f:
            json.dump(value, f)
    except FileExistsError:
        raise ValueError(f"Writer lock exists: {lock}. Verify the old session stopped, then use unlock --token <saved token>") from None
    print("Writer lock:", lock, json.dumps(value), flush=True)
    try:
        yield
    finally:
        if lock.exists() and read_json(lock) == value:
            lock.unlink()


def unlock(folder, token):
    lock = outside_checkout(folder) / "writer.lock"
    if read_json(lock)["token"] != token:
        raise ValueError("Lock token mismatch; nothing removed")
    lock.unlink()


def current_manifest(root):
    if sha256(Path(root) / "data/manifest.csv") != sha256(SCRIPT_ROOT / "data/manifest.csv"):
        raise ValueError("D20=A requires the committed current manifest unchanged")


def restore_inputs(root, base_cache, pool):
    """Colab-only isolated names + read-only link to the completed C5 pool.

    No raw files, original accepted pool or canonical result folders are touched.
    Basic placeholders are allowed because their existing feature caches are used.
    """
    root, base_cache, pool = outside_checkout(root), Path(base_cache).resolve(), Path(pool).resolve()
    if root.is_relative_to(pool) or pool.is_relative_to(root) or root == pool:
        raise ValueError("Keep isolated input root separate from the C5 source pool")
    if not (pool / "complete.json").is_file():
        raise ValueError("Missing completed C5 accepted_v2 inventory")
    (root / "data").mkdir(parents=True, exist_ok=True)
    checked_copy(SCRIPT_ROOT / "data/manifest.csv", root / "data/manifest.csv")
    for c in s3.RARE:
        with np.load(base_cache / f"basic_{c}.npz", allow_pickle=False) as z:
            paths = z["paths"].tolist()
        if len(paths) != 2000 or len(set(paths)) != 2000:
            raise ValueError(f"Expected fixed2000 basic paths: {c}")
        for relative in paths:
            p = Path(relative)
            if p.parent.as_posix() != f"data/aug_basic/{c}" or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                raise ValueError(f"Unsafe/unexpected basic cache path: {relative}")
            target = (root / p).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Basic placeholder escaped isolated root")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.touch()
    target = root / "data/gan/accepted_v2"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != pool:
            raise ValueError("Existing v2 input location differs; use a fresh isolated input root")
    else:
        target.symlink_to(pool, target_is_directory=True)
    full_groups(root)
    print("Isolated current inputs ready:", root)


def prepare(root, base_cache, cache):
    """Freeze base caches + full v2 features once; no head training here.

    The input root must have actual nonempty v2 PNGs, basic names (placeholders
    allowed only for already cached basic features), and the committed manifest.
    """
    root, base_cache, cache = Path(root).resolve(), Path(base_cache).resolve(), outside_checkout(cache)
    if cache == base_cache or cache.is_relative_to(base_cache) or base_cache.is_relative_to(cache):
        raise ValueError("Use a separate C2/C3 cache bundle, not the legacy cache folder")
    if cache == root or cache.is_relative_to(root) or root.is_relative_to(cache):
        raise ValueError("Keep the cache bundle separate from the input root")
    current_manifest(root)
    groups = full_groups(root)
    base = base_groups(root)
    verify_cache(base_cache, base)
    pool = root / "data/gan/accepted_v2"
    expected = read_json(pool / "complete.json")
    actual = {c: inventory([root / p for p in groups[s3.gan_cache_group(c, "v2")]]) for c in s3.RARE}
    if actual != expected:
        raise ValueError("v2 bytes differ from C5 completed inventory")
    identity = dict(manifest_sha256=sha256(root / "data/manifest.csv"),
                    base={g: sha256(base_cache / f"{g}.npz") for g in base}, v2=actual,
                    scripts={n: sha256(SCRIPT_ROOT / n) for n in ("c23_experiment.py", "stage3_classifier.py")},
                    packages={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "tensorflow", "keras")},
                    recipe="ImageNet Inception-v3 avg; Stage3 256 then299 RGB; float16 cache",
                    legacy_history="Backbone weights/input bytes unverified for legacy base features")
    protected = protected_hashes()
    with writer_lock(cache):
        if not (cache / "identity.json").exists() and any(p.name != "writer.lock" for p in cache.iterdir()):
            raise ValueError("Unidentified cache contents; use a fresh bundle destination")
        lock_json(cache / "identity.json", identity)
        for g in base:
            checked_copy(base_cache / f"{g}.npz", cache / f"{g}.npz")
        holder = {}

        def net_fn():
            if not holder:
                holder["net"] = s3.backbone()
            return holder["net"]

        for c in s3.RARE:
            g = s3.gan_cache_group(c, "v2")
            marker = cache / f"{g}.complete.json"
            if marker.exists():
                if read_json(marker) != dict(sha256=sha256(cache / f"{g}.npz")):
                    raise ValueError(f"Changed v2 cache: {g}")
            else:
                if (cache / f"{g}.npz").exists():
                    # A disconnect after atomic cache save may leave no marker.
                    verify_cache(cache, {g: groups[g]})
                s3.need_tf()
                s3.features(g, groups[g], root, cache, net_fn)
                atomic_json(marker, dict(sha256=sha256(cache / f"{g}.npz")))
        verify_cache(cache, groups)
        # Check input bytes again before certifying extraction, including interrupted resumes.
        if actual != {c: inventory([root / p for p in groups[s3.gan_cache_group(c, "v2")]]) for c in s3.RARE}:
            raise ValueError("v2 input bytes changed during extraction")
        if protected_hashes() != protected:
            raise ValueError("Canonical inputs/results changed")
        lock_json(cache / "complete.json", dict(identity_sha256=sha256(cache / "identity.json"),
                  files={g: sha256(cache / f"{g}.npz") for g in groups}))
    print("C2/C3 cache bundle ready:", cache)


@contextmanager
def cache_view(cache, groups, amount, seed):
    """Stage3 expects subset-keyed GAN caches. Build only these in local scratch.

    Common/basic/test files are hard-linked when possible, otherwise copied.
    The immutable full bundle is never rewritten to accommodate a different N.
    """
    chosen = selected_gan(groups, amount, seed) if amount else {}
    with tempfile.TemporaryDirectory(prefix="c23-cache-") as temp:
        dest = Path(temp)
        for g, paths in groups.items():
            if g.startswith("gan_v2_"):
                if not amount:
                    continue
                with np.load(cache / f"{g}.npz", allow_pickle=False) as z:
                    index = {p: i for i, p in enumerate(paths)}
                    np.savez(dest / f"{g}.npz", paths=np.array(chosen[g]),
                             feats=z["feats"][[index[p] for p in chosen[g]]])
            else:
                try:
                    os.link(cache / f"{g}.npz", dest / f"{g}.npz")
                except OSError:
                    shutil.copyfile(cache / f"{g}.npz", dest / f"{g}.npz")
        yield dest


def metric_summary(runs, planned):
    """Descriptive seed variability, and nominal paired intervals vs fixed controls.

    Seeds reuse the same test patients; intervals do not describe patient sampling.
    No significance/equivalence decisions or selection of a best amount are made.
    """
    rows, differences, contrasts = [], [], []
    control = "n0000" if planned[0][1].startswith("n") else "full"
    conditions = list(dict.fromkeys(x[1] for x in planned))
    for metric in ("acc", "mcc", "rare_macro", "common_macro", "rare_correct"):
        wide = runs.pivot(index="seed", columns="condition", values=metric)
        for condition in conditions:
            values = wide[condition].dropna() if condition in wide else pd.Series(dtype=float)
            rows.append(dict(metric=metric, condition=condition, n=len(values),
                             mean=values.mean(), sd=values.std(ddof=1), complete=len(values) == 20))
            if condition == control or condition not in wide or control not in wide:
                continue
            pair = wide[[condition, control]].dropna()
            diff = pair[condition] - pair[control]
            for seed, value in diff.items():
                differences.append(dict(metric=metric, seed=seed, condition=condition, control=control, difference=value))
            n, sd = len(diff), diff.std(ddof=1)
            half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            contrasts.append(dict(metric=metric, condition=condition, control=control, n_pairs=n,
                                  mean_diff=diff.mean(), sd_diff=sd, ci95_lo=diff.mean() - half,
                                  ci95_hi=diff.mean() + half, complete=n == 20,
                                  inference="exploratory nominal paired-t; no multiplicity adjustment; no equivalence claim"))
    return pd.DataFrame(rows), pd.DataFrame(differences), pd.DataFrame(contrasts)


class Sweep:
    def __init__(self, root, cache, output, experiment, trainer="keras"):
        self.root, self.cache, self.output = Path(root).resolve(), Path(cache).resolve(), outside_checkout(output)
        if (self.output == self.cache or self.output.is_relative_to(self.cache) or self.cache.is_relative_to(self.output)
                or self.output == self.root or self.output.is_relative_to(self.root) or self.root.is_relative_to(self.output)):
            raise ValueError("Keep input root, cache bundle and experiment output separate")
        if trainer not in ("keras", "fast"):
            raise ValueError("Unknown trainer")
        self.plan, self.trainer = schedule(experiment), trainer
        current_manifest(self.root)
        self.groups = full_groups(self.root)
        verify_cache(self.cache, self.groups)
        marker = read_json(self.cache / "complete.json")
        if marker["identity_sha256"] != sha256(self.cache / "identity.json"):
            raise ValueError("Cache bundle identity changed")
        self.cache_hashes = {g: sha256(self.cache / f"{g}.npz") for g in self.groups}
        if marker["files"] != self.cache_hashes:
            raise ValueError("Cache bundle contents changed")
        if read_json(self.root / "data/gan/accepted_v2/complete.json") != read_json(self.cache / "identity.json")["v2"]:
            raise ValueError("C5 inventory differs from the feature bundle")
        self.protected = protected_hashes()
        man = pd.read_csv(self.root / "data/manifest.csv")
        self.test = man[man.split == "test"].copy()
        self.test["path"] = self.test.path.map(s3.fix)
        scripts = ["stage3_classifier.py", "c23_experiment.py", "c1_experiment.py", "c0_experiment.py",
                   "c5_topup.py", "stage1b_normalize.py"]
        scripts.append("fast_head.py")
        self.identity = dict(experiment=experiment, input_arm="current", gan_pool="v2", trainer=trainer,
                             manifest_sha256=sha256(self.root / "data/manifest.csv"),
                             bundle_identity_sha256=marker["identity_sha256"], caches=self.cache_hashes,
                             scripts={n: sha256(SCRIPT_ROOT / n) for n in scripts},
                             revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_ROOT,
                                                              text=True, encoding="utf-8").strip(),
                             packages={p: importlib.metadata.version(p) for p in
                                       ("numpy", "pandas", "scipy", "tensorflow", "keras", "scikit-learn")},
                             protocol=dict(epochs=250, patience=0, restore="last", batch=10, lr=.001,
                                           seeds=SEEDS, schedule=self.plan, common_sampling="fitting only; independent class streams; nested caps"),
                             shuffle_policy="fast: fresh tf.random.shuffle each epoch after set_random_seed; keras: installed array-adapter behavior; not seeded-equivalent",
                             limitation="Seeds share patients; source/disease confounding and legacy base feature history unresolved")
        self.output.mkdir(parents=True, exist_ok=True)
        if not (self.output / "identity.json").exists() and any(self.output.iterdir()):
            raise ValueError("Unidentified output contents; use a fresh experiment destination")
        lock_json(self.output / "identity.json", self.identity)
        lock_json(self.output / "schedule.json", self.plan)

    def verify_inputs(self):
        current_manifest(self.root)
        if full_groups(self.root) != self.groups or protected_hashes() != self.protected:
            raise ValueError("Input pools or canonical files changed")
        if read_json(self.root / "data/gan/accepted_v2/complete.json") != read_json(self.cache / "identity.json")["v2"]:
            raise ValueError("C5 inventory changed")
        if any(sha256(self.cache / f"{g}.npz") != h for g, h in self.cache_hashes.items()):
            raise ValueError("Immutable feature bundle changed")
        if any(sha256(SCRIPT_ROOT / n) != h for n, h in self.identity["scripts"].items()):
            raise ValueError("Training/provenance code changed during sweep")

    def completed(self):
        records = {}
        planned = {(s, c) for s, c, _, _ in self.plan}
        for marker in self.output.glob("s*/*/complete.json"):
            saved = read_json(marker)
            key = saved["seed"], saved["condition"]
            if key not in planned or marker.parent != self.output / f"s{key[0]:03d}" / key[1]:
                raise ValueError(f"Unplanned completed run: {marker}")
            if saved["identity_sha256"] != sha256(self.output / "identity.json"):
                raise ValueError(f"Run identity changed: {marker}")
            if set(saved["files"]) != set((*ARTIFACTS, "run.json", "selection.json.gz")):
                raise ValueError(f"Incomplete artifact inventory: {marker}")
            for name, digest in saved["files"].items():
                if sha256(marker.parent / name) != digest:
                    raise ValueError(f"Completed artifact changed: {marker.parent / name}")
            records[key] = saved
        return records

    def run(self):
        with writer_lock(self.output):
            self.verify_inputs()
            records = self.completed()
            try:
                for seed, condition, amount, cap in self.plan:
                    if (seed, condition) in records:
                        print("retained", seed, condition, flush=True)
                        continue
                    out = self.output / f"s{seed:03d}" / condition
                    out.mkdir(parents=True, exist_ok=True)
                    save_selection(out / "selection.json.gz", selection_record(self.root, self.groups, amount, cap, seed))
                    with cache_view(self.cache, self.groups, amount, seed) as view:
                        command = [sys.executable, str(SCRIPT_ROOT / "stage3_classifier.py"), "--root", str(self.root),
                                   "--cache-dir", str(view), "--out-dir", str(out), "--cache-only",
                                   "--variant", "gan" if amount else "basic", "--gan-pool", "v2", "--gan-per-class", str(amount),
                                   "--seed", str(seed), "--epochs", "250", "--patience", "0", "--restore", "last",
                                   "--batch", "10", "--lr", ".001", "--trainer", self.trainer]
                        if cap is not None:
                            command += ["--major-train-per-class", str(cap)]
                        atomic_json(out / "run.json", dict(command=command, identity_sha256=sha256(self.output / "identity.json")))
                        atomic_json(out / "status.json", dict(state="running"))
                        child, start = None, time.monotonic()
                        try:
                            with (out / "run.log").open("w", encoding="utf-8") as log:
                                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
                                pids = SCRIPT_ROOT / ".agents/pids"
                                pids.mkdir(parents=True, exist_ok=True)
                                (pids / "c23-stage3.pid").write_text(str(child.pid), encoding="utf-8")
                                print("running", seed, condition, "log:", out / "run.log", flush=True)
                                if child.wait():
                                    raise RuntimeError(f"Stage3 failed: {out / 'run.log'}")
                            saved = self.finish(out, seed, condition, amount, cap, start)
                            atomic_json(out / "status.json", dict(state="complete"))
                            atomic_json(out / "complete.json", saved)
                        except BaseException as error:
                            if child is not None and child.poll() is None:
                                child.terminate()
                                child.wait()
                            atomic_json(out / "status.json", dict(state="interrupted_or_failed", error=str(error)))
                            raise
            finally:
                self.verify_inputs()

    def finish(self, out, seed, condition, amount, cap, start):
        frame = pd.read_csv(out / "test_predictions.csv")
        validate_predictions(frame)
        if frame.path.map(s3.fix).tolist() != self.test.path.tolist() or frame.label.tolist() != self.test.label.tolist():
            raise ValueError("Test identities/order changed")
        probs = frame[[f"p_{c}" for c in s3.CLASSES]].to_numpy()
        if (not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
                or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)
                or frame.pred.tolist() != np.asarray(s3.CLASSES)[probs.argmax(axis=1)].tolist()):
            raise ValueError("Invalid/inconsistent test probabilities")
        history = pd.read_csv(out / "history.csv")
        if len(history) != 250 or history.epoch.tolist() != list(range(250)) or not np.isfinite(history.select_dtypes("number")).all().all():
            raise ValueError("Incomplete/nonfinite fixed250 history")
        with np.load(out / "head.npz", allow_pickle=False) as head, np.load(out / "last_head.npz", allow_pickle=False) as last:
            if (head["W"].shape != (2048, 9) or head["b"].shape != (9,)
                    or head["classes"].tolist() != s3.CLASSES or not np.isfinite(head["W"]).all()
                    or not np.isfinite(head["b"]).all()):
                raise ValueError("Invalid primary head shape/values/classes")
            if any(not np.array_equal(head[k], last[k]) for k in ("W", "b", "classes")):
                raise ValueError("Primary head must equal FINAL head")
        counts = pd.read_csv(out / "data_counts.csv", index_col=0).reindex(s3.CLASSES)
        man = pd.read_csv(self.root / "data/manifest.csv")
        expected_major = man[man.split == "train"].label.value_counts().reindex(s3.MAJOR).clip(upper=cap) if cap else man[man.split == "train"].label.value_counts().reindex(s3.MAJOR)
        if counts.loc[s3.MAJOR, "train"].tolist() != expected_major.tolist() or not (counts.loc[s3.RARE, "train"] == (2000 + amount) * 9 // 10).all():
            raise ValueError("Fitting counts differ from planned intervention")
        expected_val = man[(man.split == "val") & man.label.isin(s3.MAJOR)].label.value_counts().reindex(s3.MAJOR)
        if counts.loc[s3.MAJOR, "val"].tolist() != expected_val.tolist() or not (counts.loc[s3.RARE, "val"] == (2000 + amount) // 10).all():
            raise ValueError("Validation counts differ from fixed common/pooled rare protocol")
        if read_json(out / "trainer.json")["trainer"] != self.trainer:
            raise ValueError("Trainer differs from locked sweep")
        ok = frame.label == frame.pred
        recalls = ok.groupby(frame.label).mean()
        return dict(seed=seed, condition=condition, acc=float(ok.mean() * 100),
                    mcc=float(matthews_corrcoef(frame.label, frame.pred)),
                    rare_macro=float(recalls.reindex(s3.RARE).mean()), common_macro=float(recalls.reindex(s3.MAJOR).mean()),
                    rare_correct=int(ok[frame.label.isin(s3.RARE)].sum()),
                    minutes=(time.monotonic() - start) / 60, identity_sha256=sha256(self.output / "identity.json"),
                    files={n: sha256(out / n) for n in (*ARTIFACTS, "run.json", "selection.json.gz")})

    def summarize(self):
        with writer_lock(self.output):
            self.verify_inputs()
            records = self.completed()
            if not records:
                print("No completed heads. No results or conclusions.")
                return
            runs = pd.DataFrame(records.values()).drop(columns="files")
            summary, pairs, contrasts = metric_summary(runs, self.plan)
            for name, frame in (("runs", runs), ("summary", summary), ("paired_differences", pairs), ("contrasts", contrasts)):
                pending = self.output / f"{name}.tmp.csv"
                frame.to_csv(pending, index=False)
                pending.replace(self.output / f"{name}.csv")
            print(summary.to_string(index=False))
            print(f"Completed {len(records)}/{len(self.plan)} heads. Exploratory seed uncertainty only; nonsignificance is not equivalence.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", required=True)
    prep.add_argument("--base-cache", required=True)
    prep.add_argument("--cache", required=True)
    restore = sub.add_parser("restore")
    for name in ("root", "base-cache", "pool"):
        restore.add_argument(f"--{name}", required=True)
    for action in ("run", "summarize"):
        p = sub.add_parser(action)
        for name in ("root", "cache", "output"):
            p.add_argument(f"--{name}", required=True)
        p.add_argument("--experiment", choices=("c2", "c3"), required=True)
        p.add_argument("--trainer", choices=("keras", "fast"), default="keras")
    p = sub.add_parser("unlock", help="only after confirming the previous runtime/sweep stopped")
    p.add_argument("--folder", required=True)
    p.add_argument("--token", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.root, a.base_cache, a.cache)
    elif a.cmd == "restore":
        restore_inputs(a.root, a.base_cache, a.pool)
    elif a.cmd == "unlock":
        unlock(a.folder, a.token)
    else:
        sweep = Sweep(a.root, a.cache, a.output, a.experiment, a.trainer)
        getattr(sweep, a.cmd)()


if __name__ == "__main__":
    main()
