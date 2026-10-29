"""Audit saved results and refresh the self-contained Report.tex numeric macros.

Run with the project interpreter and either ``write`` or ``check``. This reads
saved project artifacts only; it does not train, extract features, or rerun Stage 4.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


CLASSES = ("NORMAL", "CNV", "DME", "DRUSEN", "CSC", "MH", "MacTel", "Stargardt", "RP")
VARIANTS = ("none", "basic", "gan")
BEGIN = "% BEGIN VERIFIED NUMBERS (build_report_evidence.py)"
END = "% END VERIFIED NUMBERS"


def rows(root: Path, name: str) -> list[dict[str, str]]:
    with (root / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def matrix(root: Path, variant: str) -> list[list[int]]:
    with (root / f"results/stage4/confusion_{variant}.csv").open(newline="") as stream:
        records = list(csv.reader(stream))
    assert tuple(records[0][1:]) == CLASSES
    assert tuple(row[0] for row in records[1:]) == CLASSES
    return [[int(value) for value in row[1:]] for row in records[1:]]


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), (actual, expected)


def compute(root: Path) -> tuple[dict[str, object], dict[str, str], list[str]]:
    inputs = ["data/manifest.csv", "results/seed_check.csv", "results/stage4/metrics.csv",
              "results/stage4/recall.csv", "results/analysis/cluster_seed42.csv",
              "results/analysis/paired_cluster_seed42.csv", "results/analysis/cluster_summary.csv",
              "results/analysis/paired_seeds.csv", "results/analysis/qc_summary.csv",
              "results/analysis/cam_review_seed42_Lam.csv", "results/analysis/rare_recall.csv",
              "results/analysis/rci_paper_bounds.csv", "docs/research/C1_HEAD_TIMING.json",
              "results/stage2/stage2.log", "results/analysis/knn_sweep.csv",
              "docs/research/CONTEXT.md", "docs/research/REPLICATION.md",
              "docs/research/C0_DESIGN.md", "docs/research/C0_REPORT.md",
              "docs/research/C1_EXPERIMENT_PLAN.md", "docs/research/L7_BASELINE_AUDIT.md",
              "docs/research/L8_RCI_AUDIT.md", "docs/research/PAPER_DISCREPANCIES.md",
              "docs/research/report_sources.json", "docs/research/build_report_evidence.py",
              "COLAB.md", "stage3_classifier.py", "colab_setup.ipynb"]
    man = rows(root, inputs[0])
    test = [row for row in man if row["split"] == "test"]
    normalize = lambda value: value.replace("\\", "/")
    test_by_path = {normalize(row["path"]): row for row in test}
    assert len(test_by_path) == len(test)
    counts = dict(Counter(row["split"] for row in man))
    rare_train = dict(Counter(row["label"] for row in man
                              if row["split"] == "train" and row["label"] in CLASSES[4:]))
    common_test = [row for row in test if row["label"] in CLASSES[:4]]
    patient_ids = {row["pid"] for row in common_test}
    train_ids = {row["pid"] for row in man if row["split"] == "train" and row["label"] in CLASSES[:4]}
    train_val_ids = {row["pid"] for row in man if row["split"] in ("train", "val") and row["label"] in CLASSES[:4]}
    overlap = {"test_patients": len(patient_ids), "fitting_overlap_patients": len(patient_ids & train_ids),
               "fitting_overlap_images": sum(row["pid"] in train_ids for row in common_test),
               "train_val_overlap_patients": len(patient_ids & train_val_ids),
               "train_val_overlap_images": sum(row["pid"] in train_val_ids for row in common_test)}
    saved_overlap = next(row for row in rows(root, "results/analysis/cluster_summary.csv") if row["subset"] == "common")
    assert overlap["test_patients"] == int(saved_overlap["n_clusters"])
    for source, target in (("fitting_overlap_patients", "train_overlap_clusters"),
                           ("fitting_overlap_images", "train_overlap_images"),
                           ("train_val_overlap_patients", "train_val_overlap_clusters"),
                           ("train_val_overlap_images", "train_val_overlap_images")):
        assert overlap[source] == int(float(saved_overlap[target]))
    metrics = {row["model"]: row for row in rows(root, "results/stage4/metrics.csv")}
    saved_recall = {row[""]: row for row in rows(root, "results/stage4/recall.csv")}
    cluster = rows(root, "results/analysis/cluster_seed42.csv")
    results: dict[str, object] = {}
    macros: dict[str, str] = {}
    for variant in VARIANTS:
        for suffix in ("test_predictions.csv", "data_counts.csv", "history.csv"):
            inputs.append(f"results/stage3_{variant}/{suffix}")
        inputs.append(f"results/stage4/confusion_{variant}.csv")
        pred = rows(root, f"results/stage3_{variant}/test_predictions.csv")
        assert len(pred) == len(test)
        assert {normalize(row["path"]) for row in pred} == set(test_by_path)
        cm = [[0] * len(CLASSES) for _ in CLASSES]
        for row in pred:
            assert row["label"] == test_by_path[normalize(row["path"])]["label"]
            cm[CLASSES.index(row["label"])][CLASSES.index(row["pred"])] += 1
        assert cm == matrix(root, variant)
        true_n = [sum(row) for row in cm]
        pred_n = [sum(row[j] for row in cm) for j in range(len(CLASSES))]
        n = sum(true_n)
        correct = sum(cm[i][i] for i in range(len(CLASSES)))
        recall = [cm[i][i] / true_n[i] for i in range(len(CLASSES))]
        f1 = [2 * cm[i][i] / (true_n[i] + pred_n[i]) for i in range(len(CLASSES))]
        chance = sum(a * b for a, b in zip(true_n, pred_n))
        kappa = (correct * n - chance) / (n * n - chance)
        mcc = (correct * n - chance) / math.sqrt((n*n - sum(x*x for x in pred_n)) * (n*n - sum(x*x for x in true_n)))
        entropy = -sum((value/n) * math.log(value/n) for value in true_n if value)
        mutual = sum((cm[i][j]/n) * math.log(cm[i][j]*n/(true_n[i]*pred_n[j]))
                     for i in range(len(CLASSES)) for j in range(len(CLASSES)) if cm[i][j])
        calculated = {"acc": 100*correct/n, "kappa": kappa, "mcc": mcc, "rci": mutual/entropy}
        for key, value in calculated.items():
            close(value, float(metrics[variant][key]))
        for name, value in zip(CLASSES, recall):
            close(value, float(saved_recall[variant][name]))
        ci = next(row for row in cluster if row["variant"] == variant and row["metric"] == "acc")
        close(float(ci["estimate"]), calculated["acc"])
        history = rows(root, f"results/stage3_{variant}/history.csv")
        best = min(history, key=lambda row: float(row["val_loss"]))
        results[variant] = {**calculated, "correct": correct, "common_correct": sum(cm[i][i] for i in range(4)),
                            "rare_correct": sum(cm[i][i] for i in range(4, 9)),
                            "balanced_accuracy": statistics.mean(recall), "macro_f1": statistics.mean(f1),
                            "rare_macro_recall": statistics.mean(recall[4:]), "recall": dict(zip(CLASSES, recall)),
                            "ci95_patient_cluster_acc": [float(ci["ci_low"]), float(ci["ci_high"])],
                            "epochs_run": len(history), "best_epoch_one_based": int(best["epoch"])+1,
                            "confusion": cm}
        prefix = variant.title()
        macros.update({prefix+"Acc": f"{calculated['acc']:.2f}", prefix+"Mcc": f"{mcc:.3f}",
                       prefix+"Kappa": f"{kappa:.3f}", prefix+"Rci": f"{mutual/entropy:.3f}",
                       prefix+"Ci": f"{float(ci['ci_low']):.2f}--{float(ci['ci_high']):.2f}",
                       prefix+"Rare": str(sum(cm[i][i] for i in range(4,9))),
                       prefix+"Common": str(sum(cm[i][i] for i in range(4))),
                       prefix+"Macro": f"{100*statistics.mean(recall):.2f}",
                       prefix+"Fone": f"{statistics.mean(f1):.3f}",
                       prefix+"RareMacro": f"{100*statistics.mean(recall[4:]):.1f}"})
    seeds = rows(root, "results/seed_check.csv")
    by_variant = {v: {int(row["seed"]): float(row["acc"]) for row in seeds if row["variant"] == v} for v in VARIANTS}
    assert all(set(values) == set(range(1, 6)) for values in by_variant.values())
    seed_stats = {}
    for variant, values in by_variant.items():
        samples = list(values.values())
        seed_stats[variant] = {"n": len(samples), "mean": statistics.mean(samples),
                               "sample_sd": statistics.stdev(samples), "range": [min(samples), max(samples)]}
        prefix = variant.title()
        macros[prefix+"SeedMean"] = f"{statistics.mean(samples):.2f}"
        macros[prefix+"SeedSd"] = f"{statistics.stdev(samples):.2f}"
    differences = [by_variant["gan"][seed] - by_variant["basic"][seed] for seed in range(1,6)]
    paired = next(row for row in rows(root, "results/analysis/paired_seeds.csv") if row["a"] == "gan" and row["b"] == "basic")
    close(statistics.mean(differences), float(paired["mean_diff"]))
    close(statistics.stdev(differences), float(paired["sd_diff"]))
    macros.update({"SeedDiff": f"{statistics.mean(differences):.2f}", "SeedDiffSd": f"{statistics.stdev(differences):.2f}"})
    contrast = next(row for row in rows(root, "results/analysis/paired_cluster_seed42.csv")
                    if row["a"] == "gan" and row["b"] == "basic" and row["metric"] == "acc" and row["subset"] == "all")
    macros.update({"PairedGain": f"{float(contrast['diff']):.2f}",
                   "PairedCi": f"{float(contrast['ci_low']):.2f}--{float(contrast['ci_high']):.2f}"})
    reviews = rows(root, "results/analysis/cam_review_seed42_Lam.csv")
    assert len(reviews) == 44
    assert all(row["task"] in ("cam_none", "cam_gan") and row["rater"] == "Lam" for row in reviews)
    assert all(len({row["file"] for row in reviews if row["task"] == task}) == 22 for task in ("cam_none", "cam_gan"))
    cam = {task: dict(Counter(row["answer"] for row in reviews if row["task"] == task)) for task in ("cam_none", "cam_gan")}
    qc = rows(root, "results/analysis/qc_summary.csv")
    for row in qc:
        name = row["disease"]
        inputs.append(f"results/stage2/qc_{name}.csv")
        actual = rows(root, f"results/stage2/qc_{name}.csv")
        # The saved QC CSV contains every generated image and its distance-based pass flag.
        assert len(actual) == int(row["generated"])
        passed = sum(float(item["d_rare"]) < float(item["d_normal"]) for item in actual)
        assert passed == int(row["passed"])
        assert sum(item["pass"] == "True" for item in actual) == passed
        assert sum(item["kept"] == "True" for item in actual) == int(row["kept"])
        assert int(row["kept"]) == min(3000, passed)
        close(passed/len(actual), float(row["pass_rate"]))
    timing = json.loads((root / "docs/research/C1_HEAD_TIMING.json").read_text(encoding="utf-8"))
    close(timing["projected_250_seconds"], timing["seconds"]*250/timing["epochs_measured"])
    macros["TimingSeconds"] = f"{timing['seconds']:.2f}"
    macros["HeadMinutes"] = f"{timing['projected_250_seconds']/60:.2f}"
    generated_times = [int(value) for value in re.findall(r"wrote 4000 images .*? in (\d+)s", (root / "results/stage2/stage2.log").read_text(encoding="utf-8"))]
    assert len(generated_times) == 5
    return ({"manifest_split_counts": counts, "rare_training_counts": rare_train, "patient_overlap": overlap,
             "seed42": results, "seeds_1_to_5": seed_stats,
             "gan_minus_basic_seed_differences_pp": differences,
             "gan_minus_basic_paired_seed42": contrast, "cam_author_review": cam,
             "automatic_qc": qc, "cpu_timing": timing,
             "generation_log_seconds": generated_times, "generation_log_total_seconds": sum(generated_times)}, macros, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "check"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data, macros, inputs = compute(root)
    source_hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(set(inputs))}
    report = root / "docs/report/Report.tex"
    text = report.read_text(encoding="utf-8")
    assert text.count(BEGIN) == text.count(END) == 1
    macro_text = "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in sorted(macros.items()))
    evidence = root / "docs/research/report_evidence.json"
    if args.mode == "write":
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        record = {"as_of": "2026-10-27", "base_commit": revision,
                  "scope": "Training-free audit of saved artifacts. SD uses n-1. Seed accuracies retain their saved two-decimal precision. Cluster CIs are read, not re-estimated.",
                  "script": "docs/research/build_report_evidence.py", "source_sha256": source_hashes,
                  "statistics": data, "latex_macros": macros}
        evidence.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report.write_text(text.split(BEGIN)[0] + BEGIN + "\n" + macro_text + "\n" + END + text.split(END)[1], encoding="utf-8")
    else:
        saved = json.loads(evidence.read_text(encoding="utf-8"))
        assert saved["statistics"] == data, "Saved report statistics differ from current inputs"
        assert saved["source_sha256"] == source_hashes, "Report source files changed"
        assert saved["latex_macros"] == macros
        assert text.split(BEGIN)[1].split(END)[0].strip() == macro_text
    print(f"Report evidence: PASS ({len(source_hashes)} source files; 3 x {len(rows(root, 'results/stage3_none/test_predictions.csv'))} prediction rows checked)")
    print("Saved metrics, manifest/cluster overlap, seed summaries, QC, CAM review and report numbers agree.")
    print(f"Evidence: {evidence.relative_to(root).as_posix()}")


if __name__ == "__main__":
    main()
