"""C1/D15 cached-head sweep. Notebook Part 4 supplies Drive-only destinations.

No feature extraction or Stage 2 work. Complete heads are immutable resume units;
an interrupted head restarts. Pure scheduling/power/decision functions are below.
"""
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nct, t

from analysis import seed_summary
from c0_experiment import cache_groups, validate_predictions, verify_cache
from stage1b_normalize import atomic_json, sha256
from stage3_classifier import CLASSES, RARE, fix, rel_images

VARIANTS = ("none", "basic", "gan")
ARMS = {"fixed250": (0, "last"), "es20": (20, "best")}
PILOT = tuple(range(1, 21))
MIN_PAIRS, MAX_PAIRS = 63, 243
SCRIPT_ROOT = Path(__file__).resolve().parent
ARTIFACTS = ("head.npz", "last_head.npz", "best.weights.h5", "history.csv",
             "trainer.json", "data_counts.csv", "test_predictions.csv", "val_predictions.csv", "run.log")


def schedule(stage, target=MIN_PAIRS):
    """Seed blocks first; sensitivity is restricted to exactly seeds 1 through 20."""
    if stage not in ("pilot", "topup") or not MIN_PAIRS <= target <= MAX_PAIRS:
        raise ValueError("Invalid stage or target")
    seeds = PILOT if stage == "pilot" else range(21, target + 1)
    arms = ARMS if stage == "pilot" else ("fixed250",)
    return [(arm, seed, variant) for seed in seeds for arm in arms for variant in VARIANTS]


def require_pilot(completed):
    if not set(schedule("pilot")).issubset(completed):
        raise ValueError("Finish all fixed250 and es20 seeds 1-20 before recalculation/top-up")


def paired_power(n, sd):
    """Two-sided paired t power for a true 1 pp difference, alpha .05."""
    critical = t.ppf(.975, n - 1)
    nc = math.sqrt(n) / sd
    return float(nct.sf(critical, n - 1, nc) + nct.cdf(-critical, n - 1, nc))


def topup_rule(differences):
    """One variance-only re-estimation. Never uses a mean, CI or p-value."""
    values = np.asarray(differences, dtype=float)
    if values.shape != (20,) or not np.isfinite(values).all():
        raise ValueError("Need exactly 20 finite paired differences in percentage points")
    sd = float(values.std(ddof=1))
    # Search up to the predeclared feasibility cap; do not silently truncate power.
    required = next((n for n in range(2, MAX_PAIRS + 1)
                     if sd == 0 or paired_power(n, sd) >= .80), None)
    target = max(MIN_PAIRS, required) if required is not None else None
    return dict(sd_pp=sd, required_pairs=required, target_pairs=target,
                cap_pairs=MAX_PAIRS, feasible=target is not None,
                rule="one recalculation; alpha=.05 two-sided; power=.80 at 1 pp; floor=63; cap=243")


def decision(mean, lo, hi, complete):
    """Describe the CI against zero and D15's 1 pp practical target separately."""
    if not complete or not np.isfinite([mean, lo, hi]).all():
        return "incomplete: no final decision"
    if lo >= 1:
        return "CI supports a GAN gain of at least 1 pp"
    if lo > 0:
        return "positive GAN gain; 1 pp gain not established"
    if hi < 0:
        return "CI favors basic"
    if hi < 1:
        return "CI excludes a 1 pp GAN gain; no equivalence claim"
    return "inconclusive against zero and the 1 pp target"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lock_json(path, value):
    value = json.loads(json.dumps(value))  # compare the persisted JSON representation
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Provenance changed: {path}; retain old outputs and use a new version")
    else:
        atomic_json(path, value)


def protected_hashes():
    """Canonical small artifacts plus the source manifest; caches hashed separately."""
    paths = [SCRIPT_ROOT / "data/manifest.csv", SCRIPT_ROOT / "results/seed_check.csv"]
    paths += [p for pattern in ("stage3_*/*", "stage4/*")
              for p in (SCRIPT_ROOT / "results").glob(pattern) if p.is_file()]
    return {str(p): sha256(p) for p in paths if p.exists()}


class Sweep:
    def __init__(self, root, cache, output, input_arm, trainer="fast"):
        if trainer not in ("keras", "fast"):
            raise ValueError("Unknown trainer")
        self.trainer = trainer
        self.root, self.cache, self.output = map(Path, (root, cache, output))
        if input_arm not in ("current", "format", "framing"):
            raise ValueError("INPUT_ARM must be current, format or framing")
        if self.output.resolve().is_relative_to(SCRIPT_ROOT):
            raise ValueError("C1 outputs must be outside the checkout (Drive in Colab)")
        self.protected = protected_hashes()
        # Every variant and seed selects the same entire GAN pool below the cap.
        expected_gan = dict(CSC=3000, MH=2610, MacTel=1248, Stargardt=2199, RP=832)
        for cls in RARE:
            if len(rel_images(self.root, f"data/aug_basic/{cls}")) != 2000:
                raise ValueError(f"Fixed basic pool missing/changed: {cls}")
            if len(rel_images(self.root, f"data/gan/accepted/{cls}")) != expected_gan[cls]:
                raise ValueError(f"Fixed accepted GAN pool missing/changed: {cls}; C5 belongs to C2")
        self.groups = {}
        for variant in VARIANTS:
            self.groups.update(cache_groups(self.root, variant, 1))
        verify_cache(self.cache, self.groups)
        self.cache_hashes = {name: sha256(self.cache / f"{name}.npz") for name in self.groups}
        man = pd.read_csv(self.root / "data/manifest.csv")
        self.test = man[man.split == "test"].copy()
        self.test["path"] = self.test.path.map(fix)
        source_manifest_hash = sha256(SCRIPT_ROOT / "data/manifest.csv")
        if input_arm == "current" and sha256(self.root / "data/manifest.csv") != source_manifest_hash:
            raise ValueError("Current inputs must use the committed manifest unchanged")
        source_test = pd.read_csv(SCRIPT_ROOT / "data/manifest.csv")
        source_test = source_test[source_test.split == "test"]
        original_paths = self.test.path if input_arm == "current" else self.test.path.str.removesuffix(".png")
        if original_paths.tolist() != source_test.path.map(fix).tolist() or self.test.label.tolist() != source_test.label.tolist():
            raise ValueError("C1 test identities/labels differ from the committed split")
        provenance = {}
        if input_arm != "current":
            c0_arm = self.cache.parent
            marker = read_json(c0_arm / "features_complete.json")
            if marker["identity_sha256"] != sha256(c0_arm / "identity.json"):
                raise ValueError("C0 cache identity mismatch")
            if any(marker["cache_sha256"][f"{g}.npz"] != h for g, h in self.cache_hashes.items()):
                raise ValueError("C0 cache contents changed")
            c0_identity = read_json(c0_arm / "identity.json")
            if c0_identity["source_manifest_sha256"] != source_manifest_hash:
                raise ValueError("C0 source manifest differs from the committed split")
            if c0_identity["input_manifest_sha256"] != sha256(self.root / "data/manifest.csv"):
                raise ValueError("C0 normalized manifest changed")
            provenance = dict(c0_identity_sha256=marker["identity_sha256"])
        self.identity = dict(input_arm=input_arm, manifest_sha256=sha256(self.root / "data/manifest.csv"),
                             caches=self.cache_hashes, c0=provenance,
                             scripts={name: sha256(SCRIPT_ROOT / name) for name in
                                      ("stage3_classifier.py", "fast_head.py", "c1_experiment.py", "analysis.py", "c0_experiment.py", "stage1b_normalize.py")},
                             revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_ROOT, text=True).strip(),
                             packages={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "tensorflow", "keras")},
                             protocol=dict(trainer=trainer, epochs=250, batch=10, lr=.001, gan_per_class=3000,
                                           gan_pool="v1",
                                           arms=ARMS, first_seeds=PILOT, floor=MIN_PAIRS, cap=MAX_PAIRS),
                             cache_history="Historical backbone weights/input content unverified for legacy current caches")
        self.output.mkdir(parents=True, exist_ok=True)
        lock_json(self.output / "identity.json", self.identity)
        lock_json(self.output / "pilot_schedule.json", schedule("pilot"))

    def verify_inputs(self):
        if protected_hashes() != self.protected:
            raise ValueError("Protected canonical outputs/manifest changed")
        if sha256(self.root / "data/manifest.csv") != self.identity["manifest_sha256"]:
            raise ValueError("Input manifest changed during sweep")
        if any(sha256(self.cache / f"{g}.npz") != h for g, h in self.cache_hashes.items()):
            raise ValueError("Read-only features changed during sweep")

    def run_dir(self, arm, seed, variant):
        return self.output / arm / f"s{seed:03d}" / variant

    def completed(self):
        records = {}
        for marker in self.output.glob("*/s*/*/complete.json"):
            saved = read_json(marker)
            if saved["identity_sha256"] != sha256(self.output / "identity.json"):
                raise ValueError(f"Completed identity mismatch: {marker}")
            for name, digest in saved["files"].items():
                if sha256(marker.parent / name) != digest:
                    raise ValueError(f"Completed output changed: {marker.parent / name}")
            records[(saved["arm"], saved["seed"], saved["variant"])] = saved
        return records

    def recalculate(self):
        records = self.completed()
        require_pilot(records)
        differences = [records[("fixed250", s, "gan")]["acc"] - records[("fixed250", s, "basic")]["acc"] for s in PILOT]
        result = topup_rule(differences)
        result["pilot_completion_sha256"] = {str(p.relative_to(self.output)): sha256(p)
                                             for p in self.output.glob("*/s*/*/complete.json")
                                             if int(p.parents[1].name[1:]) <= 20}
        lock_json(self.output / "topup.json", result)
        if result["feasible"]:
            lock_json(self.output / "topup_schedule.json", schedule("topup", result["target_pairs"]))
        print(json.dumps(result, indent=2))
        return result

    def run(self, stage):
        records = self.completed()
        target = MIN_PAIRS
        if stage == "topup":
            require_pilot(records)
            result = self.recalculate()
            if not result["feasible"]:
                raise ValueError("Power target exceeds 243-pair budget; report incomplete, do not extend")
            target = result["target_pairs"]
        try:
            for arm, seed, variant in schedule(stage, target):
                if (arm, seed, variant) in records:
                    print("retained", arm, seed, variant, flush=True)
                    continue
                # Catch changed filenames before a cache-only Stage 3 process can start.
                groups = cache_groups(self.root, variant, seed)
                if groups != {g: self.groups[g] for g in groups}:
                    raise ValueError("Image pool changed")
                out = self.run_dir(arm, seed, variant)
                out.mkdir(parents=True, exist_ok=True)
                patience, restore = ARMS[arm]
                command = [sys.executable, str(SCRIPT_ROOT / "stage3_classifier.py"), "--root", str(self.root),
                           "--cache-dir", str(self.cache), "--out-dir", str(out), "--cache-only",
                           "--variant", variant, "--seed", str(seed), "--epochs", "250",
                           "--gan-pool", "v1", "--trainer", self.trainer,
                           "--patience", str(patience), "--restore", restore]
                atomic_json(out / "run.json", dict(command=command, identity=self.identity))
                atomic_json(out / "status.json", dict(state="running"))
                start = time.monotonic()
                child = None
                try:
                    with (out / "run.log").open("w", encoding="utf-8") as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8"))
                        pids = SCRIPT_ROOT / ".agents/pids"
                        pids.mkdir(parents=True, exist_ok=True)
                        (pids / "c1-stage3.pid").write_text(str(child.pid))
                        print("running", arm, seed, variant, "log:", out / "run.log", flush=True)
                        if child.wait():
                            raise RuntimeError(f"Stage 3 failed; read {out / 'run.log'}")
                    frame = pd.read_csv(out / "test_predictions.csv")
                    validate_predictions(frame)
                    if frame.path.map(fix).tolist() != self.test.path.tolist() or frame.label.tolist() != self.test.label.tolist():
                        raise ValueError("Prediction identities/order changed")
                    history = pd.read_csv(out / "history.csv")
                    if arm == "fixed250" and len(history) != 250:
                        raise ValueError("Incomplete fixed250 history")
                    ok = frame.label == frame.pred
                    saved = dict(arm=arm, seed=seed, variant=variant, acc=float(ok.mean() * 100),
                                 rare_correct=int(ok[frame.label.isin(RARE)].sum()),
                                 minutes=(time.monotonic() - start) / 60,
                                 identity_sha256=sha256(self.output / "identity.json"),
                                 files={name: sha256(out / name) for name in (*ARTIFACTS, "run.json")})
                    atomic_json(out / "status.json", dict(state="complete"))
                    atomic_json(out / "complete.json", saved)  # marker is always last
                    print("completed", arm, seed, variant, f"{saved['minutes']:.2f} min", flush=True)
                except BaseException as error:
                    if child is not None and child.poll() is None:
                        child.terminate()
                        child.wait()
                    atomic_json(out / "status.json", dict(state="interrupted_or_failed", error=str(error)))
                    raise
        finally:
            self.verify_inputs()

    def summarize(self):
        records = self.completed()
        if not records:
            print("No completed runs; no decisions")
            return
        runs = pd.DataFrame(records.values()).drop(columns=["files"])
        runs.to_csv(self.output / "runs.csv", index=False)
        summaries, pairs = [], []
        target_file = self.output / "topup.json"
        target = read_json(target_file)["target_pairs"] if target_file.exists() else None
        for arm in ARMS:
            subset = runs[runs.arm == arm]
            if subset.empty:
                continue
            summary = seed_summary(subset)
            summary.insert(0, "arm", arm)
            expected = schedule("topup", target) if arm == "fixed250" and target else []
            complete = (set(schedule("pilot")).issubset(records) and
                        (arm == "es20" or (target is not None and set(expected).issubset(records))))
            summary["complete"] = complete
            summary["decision"] = [decision(r.mean_diff, r.ci95_lo, r.ci95_hi, complete)
                                   if r.a == "gan" and r.b == "basic" else "secondary contrast"
                                   for r in summary.itertuples()]
            summaries.append(summary)
            wide = subset.pivot(index="seed", columns="variant", values="acc")
            if {"gan", "basic"}.issubset(wide.columns):
                pairs.append((wide.gan - wide.basic).rename("gan_minus_basic_pp").reset_index().assign(arm=arm))
        pd.concat(summaries).to_csv(self.output / "summary.csv", index=False)
        if pairs:
            pd.concat(pairs).to_csv(self.output / "paired_differences.csv", index=False)
        print(pd.concat(summaries).to_string(index=False))
        print("fixed250 primary; es20 sensitivity. Nominal paired t CI; Wilcoxon is sensitivity only.")
        print("Variance adaptation is not an exact sequential confidence procedure. Seeds share the test cohort.")
        self.verify_inputs()
