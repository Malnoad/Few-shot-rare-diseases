"""C5: generation-only top-up from frozen checkpoints; no classifier training.

plan is training-free. run uses a GPU on Colab and writes resumable artifacts to
--store (Drive). publish installs a complete, immutable accepted_v2 pool locally.
Original accepted images and results/stage2/qc_*.csv are read-only inputs.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import stage2_cyclegan as s2

TARGET = 3000
MARGIN = 0.20  # candidate surplus, never a QC threshold


def candidate_count(kept, passed, generated, target=TARGET, surplus=MARGIN):
    if min(kept, passed, generated, target) < 0 or surplus < 0 or passed > generated:
        raise ValueError("invalid counts or surplus")
    deficit = max(target - kept, 0)
    if not deficit:
        return 0
    if not passed or not generated:
        raise ValueError("cannot estimate candidates from a zero pass rate")
    return math.ceil(deficit * (1 + surplus) * generated / passed)


def original_qc(root, cls):
    df = pd.read_csv(root / f"results/stage2/qc_{cls}.csv")
    if not (df["pass"] == (df.d_rare < df.d_normal)).all():
        raise ValueError(f"{cls}: original rule is not zero-margin QC")
    return df


def plan(root):
    rows = []
    summary = pd.read_csv(root / "results/analysis/qc_summary.csv").set_index("disease")
    for cls in s2.RARE:
        df = original_qc(root, cls)
        generated, passed, kept = len(df), int(df["pass"].sum()), int(df.kept.sum())
        if (generated, passed, kept) != tuple(int(summary.loc[cls, k]) for k in ["generated", "passed", "kept"]):
            raise ValueError(f"{cls}: QC summary differs from original CSV")
        rows.append(dict(disease=cls, generated=generated, passed=passed, kept=kept,
                         deficit=max(TARGET - kept, 0), pass_rate=passed / generated,
                         candidate_surplus=MARGIN, candidates=candidate_count(kept, passed, generated)))
    return pd.DataFrame(rows)


def fresh_sources(man, excluded, domain_a, n):
    """Use only train NORMALs; deterministic prefixes allow larger retry budgets."""
    paths = sorted(man[(man.label == "NORMAL") & (man.split == "train")].path.map(s2.fix))
    paths = [p for p in paths if Path(p).stem not in excluded | domain_a]
    if len({Path(p).stem for p in paths}) != len(paths):
        raise ValueError("NORMAL stems are not unique")
    if n < 0 or n > len(paths):
        raise ValueError(f"need {n} fresh NORMALs, only {len(paths)} available")
    return paths[:n]


def capped_pool(old, qc, target=TARGET):
    """Keep every v1 image; add nearest passing new images, ties by filename."""
    old = sorted(old)
    if len(set(old)) != len(old) or len(old) > target:
        raise ValueError("invalid original pool")
    ranked, _ = s2.select_qc(qc, target, margin=0.0, ranking="d_rare")
    new = ranked[ranked["pass"]].sort_values(["d_rare", "file"])
    if new.file.duplicated().any() or set(old) & set(new.file):
        raise ValueError("duplicate pool entries")
    return old + new.file.tolist()[:target - len(old)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def bind(path, value):
    """A resumed run must use exactly the same immutable inputs."""
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"inputs changed: {path}; use a new version, never overwrite v1/v2")
    write_json(path, value)


def inventory(files):
    result = {}
    for file in files:
        p = Path(file)
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f"missing/empty input: {p}; restore real images, not Part 2 placeholders")
        result[p.name] = sha256(p)
    return result


def old_pool(root, cls):
    files = s2.list_images(root / "data/gan/accepted" / cls)
    expected = {Path(s2.fix(p)).name for p in original_qc(root, cls).query("kept").file}
    if {Path(p).name for p in files} != expected:
        raise ValueError(f"{cls}: restore exactly the v1 accepted set with 3.4b")
    return files, inventory(files)


def checked_copy(src, dst):
    """Resume copies without replacing different bytes at an existing v2 path."""
    if dst.exists():
        if sha256(src) != sha256(dst):
            raise ValueError(f"immutable pool file changed: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def run(a):
    import cv2

    root, store, cls = Path(a.root).resolve(), Path(a.store).resolve(), a.disease
    row = plan(root).set_index("disease").loc[cls]
    if not row.deficit:
        raise ValueError(f"{cls} is already full; no top-up")
    if a.extra < 0:
        raise ValueError("--extra must be nonnegative")
    work = store / cls
    work.mkdir(parents=True, exist_ok=True)
    old, old_hashes = old_pool(root, cls)
    normal = s2.list_images(root / "data/gan/normalA")
    real = s2.list_images(root / "data/gan/real" / cls)
    man = pd.read_csv(root / "data/manifest.csv")
    normal_paths = sorted(man[(man.label == "NORMAL") & (man.split == "train")].path.map(s2.fix))
    random.Random(42).shuffle(normal_paths)
    if {Path(p).stem for p in normal} != {Path(p).stem for p in normal_paths[:2000]}:
        raise ValueError("normalA differs from the original seed-42 prep; restore gan.zip")
    expected_real = {Path(s2.fix(p)).stem for p in man[(man.label == cls) & (man.split == "train")].path}
    if {Path(p).stem for p in real} != expected_real:
        raise ValueError("real disease references differ from manifest train; restore gan.zip")
    excluded = set()
    for disease in s2.RARE:
        excluded.update(Path(s2.fix(p)).stem.removeprefix(f"{disease}_gan_")
                        for p in original_qc(root, disease).file)
    n = int(row.candidates) + a.extra
    paths = fresh_sources(man, excluded, {Path(p).stem for p in normal}, n)
    # Pin the exact checkpoint, including shards; a later checkpoint cannot mix with resumed PNGs.
    s2.need_tf()
    latest = s2.tf.train.latest_checkpoint(str(Path(a.ckpt).resolve()))
    if not latest:
        raise ValueError(f"no checkpoint: {a.ckpt}; restore existing checkpoints, do not retrain")
    checkpoint = inventory(sorted(Path(latest).parent.glob(Path(latest).name + ".*")))
    identity = dict(version="c5_v2", disease=cls, margin=0.0, ranking="d_rare", k=3,
                    target=TARGET, old=old_hashes, normal=inventory(normal[:500]),
                    real=inventory(real), checkpoint=checkpoint,
                    manifest=sha256(root / "data/manifest.csv"),
                    original_qc={c: sha256(root / f"results/stage2/qc_{c}.csv") for c in s2.RARE})
    bind(work / "inputs.json", identity)
    sources = [dict(path=p, sha256=sha256(root / p)) for p in paths]
    previous = work / "sources.json"
    if previous.exists():
        prior = json.loads(previous.read_text(encoding="utf-8"))
        if sources[:len(prior)] != prior:
            raise ValueError("source inventory changed/shrank; only append fresh candidates")
    if (work / "complete.json").exists():
        if len(sources) != len(json.loads(previous.read_text(encoding="utf-8"))):
            raise ValueError("class already complete; its v2 pool is frozen")
        print(f"{cls}: already complete (v1 verified); publish when all classes are ready")
        return
    write_json(previous, sources)
    write_json(work / "plan.json", dict(candidates=n, extra=a.extra, overlap_with_round1=0,
                                       overlap_with_domain_a=0, observed_pass_rate=float(row.pass_rate)))
    print(f"{cls}: {n} candidates; round-1 and domain-A source overlap = 0", flush=True)
    src = root / "data/gan/topup_sources_v2" / cls
    src.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dest = src / (Path(p).stem + ".png")
        img = cv2.imread(str(root / p))
        if img is None:
            raise ValueError(f"unreadable source: {p}")
        # Same OpenCV resize as prep; generation then reads a 256px PNG as before.
        pending = dest.with_name(dest.stem + ".tmp.png")
        if not cv2.imwrite(str(pending), cv2.resize(img, (s2.IMG, s2.IMG))):
            raise OSError(f"could not write {pending}")
        pending.replace(dest)
    if len(s2.list_images(src)) != n:
        raise ValueError("unexpected files in C5 source staging directory")
    args = SimpleNamespace(root=str(root), disease=cls, ckpt=a.ckpt, n=n,
                           source_dir=str(src), output_dir=str(work / "out"), resume=True)
    s2.generate(args)
    s2.tf.keras.backend.clear_session()
    gc.collect()
    expected_outputs = {f"{cls}_gan_{Path(p).stem}.png" for p in paths}
    if {Path(p).name for p in s2.list_images(work / "out")} != expected_outputs:
        raise ValueError("candidate paths differ from source inventory")
    s2.qc(SimpleNamespace(root=str(root), disease=cls, keep=int(row.deficit), margin=0.0,
                          ranking="d_rare", input_dir=str(work / "out"),
                          accepted_dir=str(work / "accepted"), report_dir=str(work / "qc")))
    qc = pd.read_csv(work / "qc" / f"qc_{cls}.csv")
    selected = capped_pool(old, qc)
    pd.DataFrame(dict(file=selected, origin=["v1"] * len(old) + ["topup"] * (len(selected) - len(old)))).to_csv(
        work / "pool.csv", index=False)
    _, after = old_pool(root, cls)
    if after != old_hashes:
        raise ValueError("v1 changed during C5")
    if len(selected) < TARGET:
        raise ValueError(f"{cls}: only {len(selected)}/{TARGET}; increase this class's EXTRA and rerun. "
                         "QC threshold stays zero; C2 is blocked.")
    for p in selected:
        checked_copy(Path(p), store / "accepted_v2" / cls / Path(p).name)
    write_json(work / "complete.json", inventory(s2.list_images(store / "accepted_v2" / cls)))
    print(f"{cls}: complete, {TARGET} images; original v1 hashes unchanged")


def publish(root, store):
    """Install all five classes only after four top-ups succeed; CSC is copied from v1."""
    inventories = {}
    for cls in s2.RARE:
        old, hashes = old_pool(root, cls)
        if cls == "CSC":
            for p in old:
                checked_copy(Path(p), store / "accepted_v2" / cls / Path(p).name)
        else:
            if not (store / cls / "complete.json").is_file():
                raise ValueError(f"{cls}: top-up incomplete; run its 3.4c generation/QC cell first")
            expected = json.loads((store / cls / "complete.json").read_text(encoding="utf-8"))
            if hashes != json.loads((store / cls / "inputs.json").read_text(encoding="utf-8"))["old"]:
                raise ValueError(f"{cls}: v1 changed since generation")
            if inventory(s2.list_images(store / "accepted_v2" / cls)) != expected:
                raise ValueError(f"{cls}: saved pool changed/incomplete")
        files = s2.list_images(store / "accepted_v2" / cls)
        if len(files) != TARGET:
            raise ValueError(f"{cls}: need exactly {TARGET} before C2")
        inventories[cls] = inventory(files)
    for cls in s2.RARE:
        for p in s2.list_images(store / "accepted_v2" / cls):
            checked_copy(Path(p), root / "data/gan/accepted_v2" / cls / Path(p).name)
        if inventory(s2.list_images(root / "data/gan/accepted_v2" / cls)) != inventories[cls]:
            raise ValueError(f"{cls}: installed pool has unexpected files")
    reports = root / "results/stage2/c5_v2"
    for cls in s2.RARE[1:]:
        for name in ["inputs.json", "sources.json", "plan.json", "pool.csv", "complete.json"]:
            checked_copy(store / cls / name, reports / cls / name)
        for p in (store / cls / "qc").iterdir():
            checked_copy(p, reports / cls / p.name)
    bind(store / "accepted_v2/complete.json", inventories)
    bind(root / "data/gan/accepted_v2/complete.json", inventories)
    print("v2 installed: 3,000 per class. C1 still defaults to v1; C2 must pass --gan-pool v2.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--store", default="data/gan/topup/v2", help="persistent C5 folder (Drive on Colab)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("run")
    p.add_argument("--disease", choices=s2.RARE[1:], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--extra", type=int, default=0, help="append candidates after a QC shortfall")
    sub.add_parser("publish")
    a = ap.parse_args()
    if a.cmd == "plan":
        print(plan(Path(a.root)).to_string(index=False))
    elif a.cmd == "run":
        run(a)
    else:
        publish(Path(a.root).resolve(), Path(a.store).resolve())


if __name__ == "__main__":
    main()
