# Few-shot rare diseases Agent Guide

Reproduction of Yoo et al. 2021 (MBEC 59:401-415): a 9-class OCT classifier (4 Kermany
classes + 5 rare diseases with 19-27 training images each) where a per-disease CycleGAN
turns NORMAL B-scans into rare-disease images for a frozen Inception-v3 + softmax head.
Five ordered scripts (`stage0`..`stage4`), run on Colab via `colab_setup.ipynb`. Mental
model: the TensorFlow CycleGAN tutorial + Keras transfer learning on cached features.

Before any research or analysis task, read `docs/research/CONTEXT.md`.

## What We Never Compromise On
1. **Raw data is read-only.** Never modify, regenerate, or invent data in
   `data/raw/`. Every transformation is a script.
2. **Every claim has a source.** Numbers, methods and citations come from a
   file, a dataset, or a paper you actually opened.
3. **The splits never move.** `data/manifest.csv` (committed) defines every split.
   Never run `stage0_manifest.py --force`: it re-samples the rare test hold-out, and
   Linux sorts file names differently from Windows. `.agents/check` enforces this.

## Glossary
- **You:** The agent reading this file and making changes.
- **We / Us:** Lam (the author) and anyone working on the project.
- **Variant:** Stage 3 training set for rare classes: `none` (real only), `basic`
  (+2,000 augmented), `gan` (+ accepted CycleGAN images). Full glossary in
  `docs/research/CONTEXT.md`.

## Known mistakes
Rules learned from real agent mistakes (details in `.agents/mistakes.md`).
Keep this list short; when a rule becomes a check, remove it from here.
- [none yet]

## Hit Every Surface (Pre-Completion Checklist)
Before calling work done, walk this list and state which entries apply:
- [ ] **Entry points:** the stage scripts, their CLI flags, and the matching cells
  in `colab_setup.ipynb` still agree (the notebook calls the scripts by flag).
- [ ] **Reverse states:** if you added an action, its reverse exists too.
- [ ] **Data & schemas:** features caches (`results/features/*.npz`) are keyed by
  image path lists; anything that changes a list invalidates the cache.
- [ ] **Docs:** `README.md`, `COLAB.md` and `docs/research/CONTEXT.md` are updated if
  the method, data or results changed.
- [ ] **Checks:** `.agents/check` passes (paste the output).

## Ways to Hurt Yourself (Environment Rules)
1. **Commands:** Locally use `.venv/Scripts/python.exe` (Python 3.12, CPU-only
   TensorFlow), not the system `python` (3.14, no TensorFlow). Fast checks:
   `.agents/check`. GPU work (Stage 2 train/generate/qc) runs on Colab only.
2. **Processes:** Save the PID of anything you start (in
   `.agents/pids/<name>.pid`) and only stop that process.
3. **Data in worktrees:** `data/raw`, `data/aug_basic` and `data/gan` are links to
   the main checkout. Don't delete or replace them, and don't download copies.
   Large files live in Google Drive `MyDrive/octfsl`; `drive/` is a local mirror.
4. **Code reaches Colab only through GitHub.** The notebook clones the repo, so an
   unpushed change does not exist on Colab. Never push without asking.
5. **Published numbers:** `results/stage4/*.csv` and `results/seed_check.csv` are the
   run of record (seed 42, Colab). Local reruns can differ in late decimals (e.g.
   scikit-learn tie-breaking); don't overwrite them with a local rerun.
6. **Architecture & taste:** Complexity belongs at the I/O boundary; analysis
   logic stays pure and testable. If the structure fights the task, say so and
   ask before breaking it.
