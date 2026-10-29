# C2/C3: fixed Colab execution plan (29 October 2026)

Implementation ready; **no C2/C3 heads have been trained**. C2 asks how GAN
amount changes performance; C3 asks how random common-class under-sampling changes
performance after augmentation. These are exploratory sensitivity experiments,
not a new patient study or a test of clinical generalization.

## Decisions and evidence

- D18=A: C1 primary protocol, **250 epochs, final head**, seeds **1–20** for both.
- D20=A: **current inputs**; C0 stops at its gate. Disease and dataset source
  remain confounded. Neither experiment removes that limitation.
- Both use C5's completed **accepted_v2**, 3,000 GAN images/class. Earlier
  coordinator thread `522bb7c2-dad0-46ce-b514-656b06ff2c6c`, positions 628/678,
  records the recommendation and Lam's acceptance for **both** C2 and C3.
  C1 continues using v1; it is not a matched full-v2 control for these experiments.
- Lam selected C3 **A**, fitting caps 10,000/5,000, in the T9 parent thread
  `adc74c5e-ff15-43d0-9515-7fbcebc37db1` on 29 October; the parent relayed
  the explicit answer “A — Fitting caps10,000/5,000 (Recommended)”.
- [Original paper, Methods §2.3 and Results/Fig. 11](https://pmc.ncbi.nlm.nih.gov/articles/PMC7829497/),
  DOI [10.1007/s11517-021-02321-1](https://doi.org/10.1007/s11517-021-02321-1).
  Text actually read from `D:/Few-shot/.agents/reports/archive/t3-61cd05e7/l7-audit/paper.xml`.
  Fig. 11 was visually read from the embedded JPEG **PDF object 132**, extracted
  without changing the original `D:/Few-shot/Feasibility study to improve deep learning in OCT diagnosis of rare.pdf`.
  Local inspection image: `.agents/reports/t9/image-132.jpg` in the T9 worktree.
  It is a copyrighted source inspection artifact, not a new research figure.

Fig. 11(a) gives these pool sizes (paper order retained):

| Class | Proposed method | Under-sampling 1 | Under-sampling 2 |
|---|---:|---:|---:|
| NORMAL | 26,860 | 10,000 | 5,000 |
| DME | 11,348 | 10,000 | 5,000 |
| DRUSEN | 8,616 | 8,616 | 5,000 |
| CNV | 37,205 | 10,000 | 5,000 |
| Each rare class | 5,000 | 5,000 | 5,000 |

Fig. 11 also includes an original/no-augmentation arm. C3 specifically tests
the three post-augmentation balance conditions; C1 separately covers real-only.
The caption and Results specify **random selection after GAN augmentation** and
test accuracy/MCC. They do not specify sampling seed, repeats, nested subsets,
or whether the listed counts precede the internal validation split. Thus the
fitting-only design below is an **authorized adaptation**, not an exact protocol
recovery. The paper's “no significant impact” is not an equivalence criterion.

## Shared fixed settings

Frozen legacy Inception-v3 features for common/basic/test inputs; fresh full-v2
GAN features extracted with the current Stage 3 ImageNet preprocessing. Adam
0.001, batch 10, 250 epochs, patience 0, `--restore last`. The notebook explicitly
selects **`--trainer fast`**, authorized after T12's identical-order gate passed.
Direct Stage 3 and runner CLI defaults remain Keras for compatibility.

Fast uses fresh `tf.random.shuffle` every epoch. T12 found the installed Keras
adapter repeated its permutation; native-shuffle 25-epoch outcomes differed
materially. Identical supplied-order updates matched, but **seeded outcome
equivalence is not established**. Record trainer, shuffle description, packages,
`fast_head.py` and Stage 3 hashes; never resume across trainers. Fresh Colab
packages may behave differently. Do not pool these heads with historical Keras
heads or reuse historical seed outcomes as new experiment runs.

The 2,000 basic source paths per rare class are fixed. Real rare images are not
appended to augmented pools (the existing Stage 3 method). Rare validation is
the first floor(pool_size/10) rows of the existing seeded permutation of
`basic + selected_GAN`; fitting uses the remainder. Common validation and the
committed test set (1,022 images, 22 rare) stay fixed. No split is regenerated.

## C2: amount of accepted GAN images

Five conditions: **0, 500, 1,000, 2,000, 3,000** GAN images **per rare class**.
Seeds 1–20 × five conditions = **100 new heads**, ascending seed blocks.
N=0 explicitly runs `--variant basic`: the **basic augmentation baseline**,
not real-only `none`. Its rare fitting/validation counts are 1,800/200 per class.
N=3,000 has 4,500/500; intermediate counts are 2,250/250, 2,700/300, 3,600/400.

GAN selection exactly preserves Stage 3: one `random.Random(seed)` stream,
rare class order CSC/MH/MacTel/Stargardt/RP, `sample(full_v2, N)` then sort.
Conditions are **not designed to be nested**; a subset can happen to nest in
one class, but nesting is not promised across amounts/classes. Both seed and N
change selected GAN paths. The basic **source pool** stays fixed; the pooled
90/10 permutation means basic **fitting/validation membership varies with N**.
This measures the existing amount protocol, including validation composition,
sample size and number of optimizer updates, not GAN quality at fixed updates.

## C3: common fitting caps, Lam's A

Three conditions: **full / cap10000 / cap5000**, all with 2,000 basic + 3,000
v2 GAN per rare class. Seeds 1–20 × three conditions = **60 new heads**.
All are run independently in `c3/v1`; the full condition is not imported from C2.

| Common class | Full fitting | cap10000 fitting | cap5000 fitting | Fixed validation |
|---|---:|---:|---:|---:|
| NORMAL | 23,536 | 10,000 | 5,000 | 2,779 |
| DME | 10,153 | 10,000 | 5,000 | 1,195 |
| DRUSEN | 7,733 | 7,733 | 5,000 | 883 |
| CNV | 32,536 | 10,000 | 5,000 | 4,669 |

These counts come from the committed `data/manifest.csv`, not the paper's
pre-validation pool sizes. Rare fitting stays **4,500/class**, validation
500/class. cap5000 is **near-balanced**, not perfectly balanced.

For each seed/common class, generate one independent NumPy SeedSequence stream
`[seed, class_index, 2303]` (class order NORMAL/CNV/DME/DRUSEN). Take a random
permutation prefix of min(cap, available fitting rows), without replacement.
cap5000 is nested inside cap10000 inside full. Keep selected rows in their
original manifest order for fitting. This RNG never consumes the GAN-selection
or trainer stream. Rare train/val membership is identical across C3 conditions
within a seed. The seed jointly varies the common subset, rare split and head
training; it does not isolate subset variance from optimization variance.

Only common **fitting** arrays are subsetted; full source caches remain intact.
The intervention changes class distribution, training-set size and update count
per epoch together. It cannot isolate class balance from data quantity.

## Outputs, resume and analysis

Use notebook **Part 4b**. After Part 1, 4b.1 restores only an isolated input
root `/content/c23_inputs/current`: a checked manifest copy, basic cache-name
placeholders, and a link to Drive's completed C5 v2 pool. It never writes raw,
v1 accepted, original cache or recorded result files. Part 2/Stage 4 is unnecessary.

**Cold resume at the same revision:** Part1 pulls main in a fresh runtime;
4b.1 first reads/writes `D/c23/v1/revision.json` and checks out that frozen full
Git SHA **before importing any project module**. Thus later main changes cannot
silently change a resumed sweep. Keep this revision file with the feature bundle.
Run Part1 only once per fresh runtime; in the same retained kernel with setup
helpers loaded, **skip Part1** and repeat4b.1 onward. Its git pull fails on the
detached pinned checkout. If kernel/helpers are lost but the local checkout
remains, request a **fresh Colab runtime**, not merely restart the kernel;
Drive outputs are unaffected. If any project module was imported before
a required revision switch, start a fresh runtime rather than reusing stale
Python modules. Tracked working-copy edits stop checkout; they are not discarded.
Do not alter the revision file to bypass an identity mismatch.

Package versions are also locked. A changed Colab package/runtime must **stop**;
read `identity.json`'s `packages` and the cache bundle's package versions and
reproduce those versions in a compatible runtime before resuming. The notebook
does not install replacements for Colab TensorFlow or relax this check. If the
recorded environment cannot be recreated, retain all outputs and ask the parent
for a reproducible-environment plan or an explicitly separate new experiment
version; do not mix results under the old identity.

4b.2 copies the required legacy feature groups into the separate persistent
`MyDrive/octfsl/c23/v1/features/` bundle and extracts **only 15,000 full-v2 GAN
features**, never a classifier. First preparation benefits from A100/L4/T4;
after completion, switch to CPU, run Part 1 and 4b.1–4b.2 again to verify/reuse it.
No Stage 2 work is needed. Input PNG hashes must match C5's completion inventory.
Cache groups retain exact ordered path, finite-feature and (n,2048) checks.
Legacy feature history remains **unverified**; hashes freeze feature values,
not their historical backbone or raw input provenance.

4b.3–4b.4 run/summarize C2; 4b.5–4b.6 run/summarize C3. Heads write directly to
Drive **`c2/v1/sNNN/nNNNN/`** and **`c3/v1/sNNN/<full|cap10000|cap5000>/`**.
All 20 seeds are fixed: no C1-style power top-up or outcome-driven stopping.
Training uses cache-only local temporary views: full common/basic/test caches
are linked/copied, and selected GAN rows are sliced from the immutable full
bundle. Different amounts/seeds never rewrite full-v2 caches. Scratch views
are discarded automatically. A hard runtime loss may leave disposable local
scratch; a fresh runtime removes it. Never delete the Drive feature bundle.

`identity.json` locks manifest/cache/source-code hashes, Git revision, package
versions, trainer and complete schedule. Each head saves T12's `trainer.json`,
selected/final/best weights, history, counts, test/validation probabilities,
command, status/log, and **`selection.json.gz` containing exact ordered common
fitting/validation, test and rare fitting/validation paths**. `complete.json`
is written last and hashes all required artifacts. Resume checks identity and
every completed artifact before skipping it; an interrupted head restarts from
epoch 1 under the same identity. Missing/changed files stop resume, never repair
a completed result silently. Scripts/input pools/cache bytes are checked again
after success/error. Unknown pre-existing destinations are rejected.

Use **one active sweep session at a time**, including across C1/C2/C3. Writer
locks reject concurrent writes to the same bundle/sweep. Ordinary exceptions
release the lock. After a hard disconnect, first ensure the old runtime is
stopped, read `writer.lock` for its token, then explicitly run:

```bash
python c23_experiment.py unlock --folder /path/to/locked/folder --token TOKEN_FROM_WRITER_LOCK
```

Do not unlock a running session. Change protocol/inputs/trainer by keeping the
old directory and choosing a fresh version; restore full fitting by omitting
`--major-train-per-class`. Original pipelines still default to v1 and Keras.

Summaries: `runs.csv`, `summary.csv`, `paired_differences.csv`, `contrasts.csv`
within each experiment. Report overall accuracy (%), MCC, common/rare macro
recall (fractions), and rare correct count. Paired differences compare each C2
amount with N=0, and each C3 cap with full. Summaries include available pairs
only, flag incompleteness, and report mean/SD plus exploratory **nominal** paired-t
95% intervals (no multiplicity adjustment). No best-N selection, superiority,
“no impact” or equivalence rule is automated. Partial summaries do not authorize
changing the schedule. This describes training-seed variation on one fixed
cohort/GAN realization, not independent patients; more seeds cannot fix leakage,
unknown rare patient identity, or source/disease confounding.

## Execution budget and handoff

No new timing run. T12's slowest laptop fast-trainer projection is
**12.87 min/250-epoch v1 GAN head**, with91,860 fitting rows. Our largest v2
condition has96,458 fitting rows. An **unmeasured linear row-scaling assumption**
gives12.87×96458/91860≈13.5min/head, or roughly **22.5 h + 13.5 h** for all100 C2
and60 C3 heads. This is a rough forecast, **not a measured rate or upper bound**. Smaller pools
are likely faster but this is unmeasured. First-v2 extraction, Colab/Drive I/O
and serialization are **unbenchmarked**; use first-head minutes for forecasts
only. Reserve multiple sessions and about **1–2 GiB additional Drive space**
for the full feature bundle, compressed selection inventories and 160 heads;
check actual free space (estimate, not a measured export size).

Bring back each experiment's four summary CSVs plus `identity.json` and
`schedule.json`; keep per-head artifacts and feature completion metadata in
Drive for audit. Do not run Part 6 to publish these experiments. Code must first
be integrated/pushed by the parent before Lam opens the matching Colab notebook.
