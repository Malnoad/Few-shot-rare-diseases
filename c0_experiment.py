"""Isolated, resumable C0 features, diagnostics and Colab head runs.

Subcommands: cue (bounded CPU check), extract (full features), train, summarize.
All artifacts go below workspace/results/c0/v1; Drive sync is in notebook Part C0.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t

import stage3_classifier as stage3
from stage1b_normalize import (ARMS, VERSION, arm_root, atomic_json, header_check,
                              normalized_path, selected_inputs, sha256, validate_source)

SEEDS = tuple(range(1, 21))
VARIANTS = ("none", "basic", "gan")
SCRIPT_ROOT = Path(__file__).resolve().parent


def output_root(workspace):
    path = (Path(workspace).resolve() / "results/c0" / VERSION).resolve()
    if not path.is_relative_to(Path(workspace).resolve() / "results/c0"):
        raise ValueError("C0 output cannot be a link outside workspace/results/c0")
    path.mkdir(parents=True, exist_ok=True)
    return path


def input_root(a, arm):
    return Path(a.source_root).resolve() if arm == "current" else arm_root(a.workspace, arm)


def code_identity():
    import cv2
    import scipy
    return dict(scripts={name: sha256(SCRIPT_ROOT / name) for name in
                         ["stage1b_normalize.py", "c0_experiment.py", "stage3_classifier.py"]},
                numpy=np.__version__, pandas=pd.__version__, opencv=cv2.__version__, scipy=scipy.__version__,
                tensorflow=stage3.need_tf().__version__)


def guard(path, identity):
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f"Provenance mismatch at {path}; use a fresh C0 version/workspace")
    atomic_json(path, identity)


def normalized_inventory(a, arm, require_full=True):
    root = input_root(a, arm)
    complete = json.loads((root / "complete.json").read_text())
    if require_full and complete["recipe"]["sample_per_class"]:
        raise ValueError("Sample normalization cannot be used for full extraction or training")
    if sha256(root / "inventory.csv") != complete["inventory_sha256"]:
        raise ValueError("Normalization inventory changed")
    if complete["recipe"]["script_sha256"] != sha256(SCRIPT_ROOT / "stage1b_normalize.py"):
        raise ValueError("Normalization recipe code changed")
    return pd.read_csv(root / "inventory.csv"), complete


def auc(positive, negative):
    ranks = rankdata(np.concatenate([positive, negative])).astype(np.float64)
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def axis_result(train, labels, test, test_labels, axis=None):
    rare_train = np.isin(labels, stage3.RARE)
    if axis is None:
        axis = train[rare_train].mean(0) - train[~rare_train].mean(0)
    scores = test @ axis
    rare_test = np.isin(test_labels, stage3.RARE)
    return dict(auc=auc(scores[rare_test], scores[~rare_test]),
                rare_above_all_common=int((scores[rare_test] > scores[~rare_test].max()).sum()),
                rare_test=int(rare_test.sum()), common_test=int((~rare_test).sum())), axis


def cue(a):
    """No fitting: identical bounded identities, fresh ImageNet features in each arm."""
    stage3.need_tf()
    out = output_root(a.workspace) / "diagnostics/sample"
    out.mkdir(parents=True, exist_ok=True)
    man, records = selected_inputs(a.source_root, a.sample_per_class)
    sample = man[man.split.isin(["train", "test"])].groupby(["label", "split"], sort=False).head(a.sample_per_class)
    # Headers use all rare raw scans (146), even though feature extraction uses 2/class/split.
    source_inventory = pd.DataFrame(records).assign(path=lambda d: d.source_path)
    results, current_axis = [], None
    net_holder = {}

    def net_fn():
        if not net_holder:
            net_holder["net"] = stage3.backbone()
        return net_holder["net"]

    for arm in ARMS:
        root = input_root(a, arm)
        if arm == "current":
            inventory = source_inventory
        else:
            inventory, _ = normalized_inventory(a, arm, require_full=False)
            if inventory.source_path.tolist() != source_inventory.source_path.tolist():
                raise ValueError("Sample inventories disagree across arms")
        header, headers = header_check(root, arm, inventory)
        headers.to_csv(out / f"{arm}_headers.csv", index=False)
        cache = out / arm / "features"
        cache.mkdir(parents=True, exist_ok=True)
        paths = [p if arm == "current" else normalized_path(p) for p in sample.path]
        hashes = [sha256(validate_source(root, p)) for p in paths]
        guard(cache.parent / "identity.json", dict(code=code_identity(), paths=paths, hashes=hashes))
        # Groups are train/test only, never head fitting or augmented validation.
        feats = {}
        for split in ["train", "test"]:
            split_paths = [p for p, keep in zip(paths, sample.split == split) if keep]
            feats[split] = stage3.features(split, split_paths, root, cache, net_fn, bs=8)
        train_labels, test_labels = sample[sample.split == "train"].label, sample[sample.split == "test"].label
        diagnostic, axis = axis_result(feats["train"], train_labels, feats["test"], test_labels)
        if arm == "current":
            current_axis = axis
        fixed, _ = axis_result(feats["train"], train_labels, feats["test"], test_labels, current_axis)
        result = dict(**header, feature_auc=diagnostic["auc"], current_axis_auc=fixed["auc"],
                      feature_train=len(train_labels), feature_test=len(test_labels),
                      interpretation="small sample only; cannot clear full-feature gate")
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    atomic_json(out / "cue_check.json", results)


def cache_groups(root, variant, seed):
    """Exact Stage 3 path lists, including its GAN sampling order and cap."""
    import random
    man = pd.read_csv(root / "data/manifest.csv")
    man["path"] = man.path.map(stage3.fix)
    major = man[man.label.isin(stage3.MAJOR)]
    groups = {f"major_{s}": major[major.split == s].path.tolist() for s in ["train", "val"]}
    groups["test"] = man[man.split == "test"].path.tolist()
    rng = random.Random(seed)
    for cls in stage3.RARE:
        if variant == "none":
            groups[f"real_{cls}"] = man[(man.label == cls) & (man.split == "train")].path.tolist()
        else:
            groups[f"basic_{cls}"] = stage3.rel_images(root, f"data/aug_basic/{cls}")
            if variant == "gan":
                gan = stage3.rel_images(root, f"data/gan/accepted/{cls}")
                groups[f"gan_{cls}"] = sorted(rng.sample(gan, min(len(gan), 3000)))
    return groups


def verify_cache(cache, groups):
    for group, paths in groups.items():
        with np.load(cache / f"{group}.npz", allow_pickle=False) as z:
            if z["paths"].tolist() != paths or z["feats"].shape != (len(paths), 2048):
                raise ValueError(f"Cache path/shape mismatch: {group}")
            if not np.isfinite(z["feats"]).all():
                raise ValueError(f"Nonfinite features: {group}")


def run_stage3(root, cache, out, variant, seed):
    command = [sys.executable, str(SCRIPT_ROOT / "stage3_classifier.py"), "--root", str(root),
               "--cache-dir", str(cache), "--out-dir", str(out), "--variant", variant, "--seed", str(seed)]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8")
        pid_dir = SCRIPT_ROOT / ".agents/pids"
        pid_dir.mkdir(parents=True, exist_ok=True)
        (pid_dir / "c0-stage3.pid").write_text(str(child.pid))
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if child.wait():
            raise RuntimeError(f"Stage 3 failed; see {out / 'run.log'}")


def extract(a):
    """Hash input content once before caches; refuse mixed recipes/code on resume."""
    stage3.need_tf()
    output = output_root(a.workspace)
    source_man, source_records = selected_inputs(a.source_root)
    source_paths = [r["source_path"] for r in source_records]
    source_hashes = [sha256(validate_source(a.source_root, p)) for p in source_paths]
    for arm in ARMS:
        root, dest = input_root(a, arm), output / arm
        if arm == "current":
            inventory = pd.DataFrame(source_records).assign(path=source_paths)
            hashes = source_hashes
            inventory["source_sha256"] = hashes
        else:
            inventory, _ = normalized_inventory(a, arm)
            if inventory.source_path.tolist() != source_paths or inventory.source_sha256.tolist() != source_hashes:
                raise ValueError("Source inventory changed between arms")
            hashes = [sha256(validate_source(root, p)) for p in inventory.path]
            if hashes != inventory.output_sha256.tolist():
                raise ValueError("Normalized pixels changed since preparation")
            expected_man = source_man.assign(path=source_man.path.map(normalized_path))
            pd.testing.assert_frame_equal(pd.read_csv(root / "data/manifest.csv"), expected_man, check_dtype=False)
        dest.mkdir(parents=True, exist_ok=True)
        identity = dict(arm=arm, version=VERSION, code=code_identity(), paths=inventory.path.tolist(), hashes=hashes,
                        input_manifest_sha256=sha256(root / "data/manifest.csv"),
                        source_manifest_sha256=sha256(Path(a.source_root) / "data/manifest.csv"))
        guard(dest / "identity.json", identity)
        header, _ = header_check(root, arm, inventory)
        atomic_json(dest / "header_check.json", header)
        cache = dest / "features"
        cache.mkdir(parents=True, exist_ok=True)
        # none supplies real rare features; gan supplies basic + all seed-specific GAN sets.
        # The accepted pool is fixed; >3000 images would give different sets per seed.
        # Cache each full pool once; train refuses a seed with a different path list.
        for cls in stage3.RARE:
            if len(stage3.rel_images(root, f"data/gan/accepted/{cls}")) > 3000:
                raise ValueError("C0 v1 requires the fixed accepted pools <=3000/class; do not change QC")
        groups = cache_groups(root, "none", 1) | cache_groups(root, "gan", 1)
        net_holder = {}

        def net_fn():
            if not net_holder:
                net_holder["net"] = stage3.backbone()
            return net_holder["net"]

        backup_files(a, [dest / "identity.json", dest / "header_check.json"])
        for group, paths in groups.items():
            stage3.features(group, paths, root, cache, net_fn)
            verify_cache(cache, {group: paths})
            backup_files(a, [cache / f"{group}.npz"])
        atomic_json(dest / "features_complete.json", dict(identity_sha256=sha256(dest / "identity.json"),
                    cache_sha256={p.name: sha256(p) for p in sorted(cache.glob("*.npz"))}))
        backup_files(a, [dest / "features_complete.json"])
    full_diagnostics(a)
    backup_files(a, [output / "full_axis.csv", output / "gate.json"])


def backup_files(a, paths):
    """Copy only completed cache groups/markers; never persist an in-flight npz."""
    if not a.backup_dir:
        return
    base = output_root(a.workspace)
    for path in paths:
        target = Path(a.backup_dir) / path.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(path, pending)
        pending.replace(target)


def full_diagnostics(a):
    output = output_root(a.workspace)
    rows, current_axis = [], None
    for arm in ARMS:
        cache, root = output / arm / "features", input_root(a, arm)
        man = pd.read_csv(root / "data/manifest.csv")
        with np.load(cache / "major_train.npz", allow_pickle=False) as z:
            train = [z["feats"].astype(np.float32)]
        labels = man[(man.split == "train") & man.label.isin(stage3.MAJOR)].label.tolist()
        for cls in stage3.RARE:
            with np.load(cache / f"real_{cls}.npz", allow_pickle=False) as z:
                train.append(z["feats"].astype(np.float32))
                labels += [cls] * len(train[-1])
        with np.load(cache / "test.npz", allow_pickle=False) as z:
            test = z["feats"].astype(np.float32)
        train = np.concatenate(train)
        result, axis = axis_result(train, labels, test, man[man.split == "test"].label)
        if arm == "current":
            current_axis = axis
        fixed, _ = axis_result(train, labels, test, man[man.split == "test"].label, current_axis)
        rows.append(dict(arm=arm, **result, current_axis_auc=fixed["auc"]))
    pd.DataFrame(rows).to_csv(output / "full_axis.csv", index=False)
    framing_auc = next(r["auc"] for r in rows if r["arm"] == "framing")
    atomic_json(output / "gate.json", dict(headers_pass=True, framing_auc=framing_auc,
                ready_for_heads=framing_auc < 0.90, rule="full framing AUC must be <0.90"))
    print(pd.DataFrame(rows).to_string(index=False))
    print("HEAD GATE:", "PASS" if framing_auc < 0.90 else "STOP: residual separation; Lam decides next step")


def train(a):
    """One seed/variant/arm per invocation lets the notebook sync after each head."""
    output = output_root(a.workspace)
    gate = json.loads((output / "gate.json").read_text())
    if not gate["ready_for_heads"]:
        raise ValueError("Full feature gate did not pass; report residual separation to Lam before training")
    root, dest = input_root(a, a.arm), output / a.arm
    identity = json.loads((dest / "identity.json").read_text())
    if identity["code"] != code_identity():
        raise ValueError("Code/environment changed after extraction")
    if sha256(Path(a.source_root) / "data/manifest.csv") != identity["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    if sha256(root / "data/manifest.csv") != identity["input_manifest_sha256"]:
        raise ValueError("Arm manifest changed")
    marker = json.loads((dest / "features_complete.json").read_text())
    if marker["identity_sha256"] != sha256(dest / "identity.json"):
        raise ValueError("Cache identity mismatch")
    cache = dest / "features"
    groups = cache_groups(root, a.variant, a.seed)
    for group in groups:
        if marker["cache_sha256"][f"{group}.npz"] != sha256(cache / f"{group}.npz"):
            raise ValueError(f"Cache content changed: {group}")
    verify_cache(cache, groups)
    out = dest / "runs" / f"{a.variant}_s{a.seed}"
    run_identity = dict(identity_sha256=marker["identity_sha256"], variant=a.variant, seed=a.seed,
                        epochs=250, patience=20, restore="best", batch=10, lr=0.001, gan_per_class=3000)
    guard(out / "run_identity.json", run_identity)
    done = out / "complete.json"
    if done.exists():
        previous = json.loads(done.read_text())
        if previous["run_identity"] != run_identity or any(sha256(out / p) != h for p, h in previous["files"].items()):
            raise ValueError("Completed head provenance/content mismatch")
        print("Completed run retained:", out)
        return
    start = time.monotonic()
    run_stage3(root, cache, out, a.variant, a.seed)
    validate_predictions(pd.read_csv(out / "test_predictions.csv"))
    atomic_json(done, dict(run_identity=run_identity, minutes=(time.monotonic() - start) / 60,
                files={name: sha256(out / name) for name in
                       ["head.npz", "history.csv", "data_counts.csv", "test_predictions.csv", "val_predictions.csv"]}))


def validate_predictions(frame):
    if len(frame) != 1022 or frame.path.duplicated().any():
        raise ValueError("Incomplete/duplicate test predictions")
    counts = frame.label.value_counts().to_dict()
    expected = {c: 250 for c in stage3.MAJOR} | dict(CSC=5, MH=5, MacTel=4, Stargardt=4, RP=4)
    if counts != expected or not frame.pred.isin(stage3.CLASSES).all():
        raise ValueError("Unexpected test labels/counts")


def recall_metrics(frame):
    recall = frame.assign(ok=frame.label == frame.pred).groupby("label").ok.mean()
    rare = frame.label.isin(stage3.RARE)
    return dict(accuracy=float((frame.label == frame.pred).mean()),
                rare_macro=float(recall[stage3.RARE].mean()), common_macro=float(recall[stage3.MAJOR].mean()),
                rare_correct=int(((frame.label == frame.pred) & rare).sum()),
                rare_to_common=int((rare & frame.pred.isin(stage3.MAJOR)).sum()),
                common_to_rare=int((~rare & frame.pred.isin(stage3.RARE)).sum()),
                **{f"recall_{c}": float(recall[c]) for c in stage3.CLASSES})


def paired_contrast(pair):
    """Apply the fixed practical rule only to the complete matched seed set."""
    loss = pair.rare_macro_left - pair.rare_macro_right
    common = pair.common_macro_left - pair.common_macro_right
    complete = len(pair) == len(SEEDS) and set(pair.seed) == set(SEEDS)
    half = float(student_t.ppf(0.975, len(loss) - 1) * loss.std(ddof=1) / np.sqrt(len(loss))) if len(loss) > 1 else None
    passed = bool(loss.mean() >= 0.10 and (loss > 0).sum() >= 16 and common.mean() <= 0.02) if complete else None
    return dict(seeds=len(pair), complete=complete, rare_macro_loss=float(loss.mean()), ci95_halfwidth=half,
                positive_seeds=int((loss > 0).sum()), common_macro_loss=float(common.mean()), rule_pass=passed)


def summarize(a):
    output = output_root(a.workspace)
    rows = []
    for marker in sorted(output.glob("*/runs/*/complete.json")):
        saved = json.loads(marker.read_text())
        for name, digest in saved["files"].items():
            if sha256(marker.parent / name) != digest:
                raise ValueError(f"Run content mismatch: {marker.parent / name}")
        frame = pd.read_csv(marker.parent / "test_predictions.csv")
        validate_predictions(frame)
        info = saved["run_identity"]
        # Compare original image identities across arms (only the appended .png differs).
        paths = frame.path.map(stage3.fix)
        if marker.parents[2].name != "current":
            paths = paths.str.removesuffix(".png")
        expected = pd.read_csv(Path(a.source_root) / "data/manifest.csv")
        expected = expected[expected.split == "test"]
        if paths.tolist() != expected.path.map(stage3.fix).tolist() or frame.label.tolist() != expected.label.tolist():
            raise ValueError("Test identities/order changed")
        rows.append(dict(arm=marker.parents[2].name, variant=info["variant"], seed=info["seed"],
                         minutes=saved["minutes"], **recall_metrics(frame)))
    if not rows:
        print("No completed heads; no conclusions.")
        return
    runs = pd.DataFrame(rows).sort_values(["variant", "seed", "arm"])
    runs.to_csv(output / "runs.csv", index=False)
    contrasts = []
    for variant in VARIANTS:
        g = runs[runs.variant == variant]
        for left, right in [("current", "framing"), ("current", "format"), ("format", "framing")]:
            pair = g[g.arm == left].merge(g[g.arm == right], on="seed", suffixes=("_left", "_right"), validate="one_to_one")
            if not len(pair):
                continue
            contrasts.append(dict(variant=variant, contrast=f"{left}-{right}", **paired_contrast(pair),
                                  primary=variant == "none" and left == "current" and right == "framing"))
    pd.DataFrame(contrasts).to_csv(output / "contrasts.csv", index=False)
    print(pd.DataFrame(contrasts).to_string(index=False))
    print("Fractions, not percentages. Partial seed blocks cannot trigger the decision rule.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", default=".")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--backup-dir", help="Colab Drive c0/v1 folder; persist each completed feature group")
    sub = ap.add_subparsers(dest="command", required=True)
    cue_parser = sub.add_parser("cue", help="small CPU feature/header check; no head training")
    cue_parser.add_argument("--sample-per-class", type=int, default=2)
    sub.add_parser("extract", help="fresh full features, full L7 axis and pre-head gate")
    head = sub.add_parser("train", help="Colab only: one isolated head")
    head.add_argument("--arm", choices=ARMS, required=True)
    head.add_argument("--variant", choices=VARIANTS, required=True)
    head.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("summarize", help="summarize completed runs without fitting")
    a = ap.parse_args()
    if a.command == "cue" and a.sample_per_class < 1:
        ap.error("sample-per-class must be positive")
    pid_dir = SCRIPT_ROOT / ".agents/pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    import os
    (pid_dir / "c0.pid").write_text(str(os.getpid()))
    dict(cue=cue, extract=extract, train=train, summarize=summarize)[a.command](a)


if __name__ == "__main__":
    main()
