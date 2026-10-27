# T6 / C0 delivery report

Prepared 2026-10-27 on `task/t6-c0-format-control`. No head or GAN training ran
locally. No push. The design was written before implementation in
[C0_DESIGN.md](C0_DESIGN.md); it remains the experiment specification.

## Design

- `current`: existing Stage 3 pipeline.
- `format`: alpha discarded, 8-bit grayscale PNG, native dimensions retained.
- `framing`: same conversion, centered black square padding, bilinear resize
  to 256×256. Retains the full scan. Visual inspection of 27 scans found variable
  rare-image black bands and tilted Kermany white borders; an unvalidated crop
  could remove retinal tissue. Padding can retain aspect-ratio/source information
  in the pixels; it is not a claim that acquisition cues disappear.
- Each arm × none/basic/gan × seeds 1–20, paired: 180 heads. These seeds match
  the available C1 implementation's seed set. Same existing augmentations and
  accepted GANs, same split and Stage 3 training settings in each arm.
- Primary: current minus framing, none, rare macro recall. Flag sensitivity if
  mean loss ≥0.10, positive in ≥16/20 seeds, and mean common macro-recall loss
  ≤0.02. Report every contrast and a paired 95% t interval across training seeds.
  A failed rule does not exclude confounding. Seeds do not add independent patients.
- Before heads, the full recomputed framing feature-axis AUC must be <0.90.
  Otherwise the notebook stops for Lam's scientific decision. Also inspect the
  fixed current-axis AUC; neither diagnostic distinguishes disease from source.

## Changed files

- `stage1b_normalize.py`: deterministic, resumable copies of manifest images,
  Stage 1 basic images and existing accepted Stage 2 GAN images. Derived manifests,
  SHA-256 inventories and header checks; no augmentation/GAN regeneration or QC.
- `c0_experiment.py`: bounded training-free cue check, full cache extraction,
  full-axis diagnostics/gate, isolated Colab head runs, paired summary.
- `stage3_classifier.py`: optional `--cache-dir` / `--out-dir`; atomic cache-group
  writes. Existing defaults and training behavior remain intact.
- `colab_setup.ipynb`: new Part C0 (five code cells) after Part 3; one index row.
  All pre-existing Parts 1–6, including Part 4, retain identical cell content.
- `.agents/check`: compiles both new scripts, runs focused C0 tests; accepts
  `PYTHON` to select the supplied main-checkout interpreter from this worktree.
- `.gitignore`: excludes normalized images and C0 artifacts.
- `tests/test_c0.py`: five focused training-free tests for pixels, split/pool
  mapping, immutable inputs, resume/content checks, placeholders, AUC/metric
  conventions and the complete-seed/common-recall decision rule.
- `.agents/mistakes.md`: logs inspection-command assumptions; no proposed
  permanent behavioral guardrail was applied.
- `docs/research/C0_DESIGN.md` and this report.

README, REPLICATION, CONTEXT and analysis.py were not edited. Proposed shared-doc
text is below and in the design; shared-document integration belongs to T1.

## Training-free evidence

Final implementation uses existing OpenCV, NumPy/pandas and SciPy dependencies;
no package was installed. Local TensorFlow was 2.21.0, reporting `GPUs: []`.
Inspected 190 inputs/arm: all 146 real rare images, two common images per class
per split (24), two basic and two accepted GAN images per rare class (20).
Verified original and copied SHA-256 hashes after feature extraction.

- Current: 122 distinct dimension/mode/encoding signatures; rare class from
  native width/height = **16/22**, no tied nearest-class medians.
- Format: all 190 are 8-bit grayscale PNG; native dimensions remain; same
  **16/22**, no ties. This arm deliberately isolates format from geometry.
- Framing: all 190 are 8-bit grayscale PNG, **256×256**, one header signature.
  The size-only classifier ties on **all 22** rare tests. Its nominal 5/22 comes
  entirely from always choosing CSC under the declared tie rule: no size signal.

ImageNet frozen features used the same first two TRAIN and two TEST images per
class in manifest order: 18 train + 18 test, 36 images/arm, 108 feature evaluations.
No test image contributes to the direction's training means. Final diagnostic:

- Current: recomputed-axis AUC **0.9375**; fixed current-axis AUC **0.9375**.
- Format: recomputed-axis AUC **0.9375**; fixed current-axis AUC **0.9375**.
- Framing: recomputed-axis AUC **0.7125**; fixed current-axis AUC **0.9250**.

The original separating direction retains substantial separation on this tiny
sample. Thus **cue removal is not established**. Only the header cue has been
removed in framing. Run the full diagnostic before deciding to spend time on
heads. The sample is neither the full L7 AUC (0.9955 in its branch report) nor an
independent estimate with patient-level uncertainty.

An earlier progress update reported framing AUC 0.8000 from a Pillow prototype.
The final OpenCV implementation avoids adding an undeclared image dependency;
the fixed arms, geometry, sample identities and thresholds did not change. The
final check was rerun in a fresh workspace; the values above supersede that
prototype and illustrate the diagnostic's sensitivity to pixel implementation.

Reproduce the final local check (PowerShell, no training):

```powershell
$py = 'D:/Few-shot/.venv/Scripts/python.exe'
$ws = '.agents/reports/c0-final'
& $py stage1b_normalize.py --root D:/Few-shot --workspace $ws --arm format --sample-per-class 2
& $py stage1b_normalize.py --root D:/Few-shot --workspace $ws --arm framing --sample-per-class 2
$env:CUDA_VISIBLE_DEVICES = '-1'
& $py c0_experiment.py --source-root D:/Few-shot --workspace $ws cue
```

Local machine-readable output:
`.agents/reports/c0-final/results/c0/v1/diagnostics/sample/cue_check.json`.
Image inventories, cached features, exact selected paths/content hashes and code
hashes are retained alongside it, ignored by Git. No results of record were used
as writable outputs.

## Verification

Command (PowerShell):

```powershell
& 'C:/Program Files/Git/bin/bash.exe' -c 'PYTHON=/d/Few-shot/.venv/Scripts/python.exe bash .agents/check'
```

Output (exit 0):

```text
.framing: verified 38/38 images
Copies: C:\Users\ThinkBook\AppData\Local\Temp\tmp69nv81b3\data\c0\v1\framing; sources untouched
framing: verified 38/38 images
....
----------------------------------------------------------------------
Ran 5 tests in 0.489s

OK
Copies: C:\Users\ThinkBook\AppData\Local\Temp\tmp69nv81b3\data\c0\v1\framing; sources untouched
```

The worktree has no raw-data link, so that check's conditional raw census was
separately run against the actual source checkout:
`D:/Few-shot/.venv/Scripts/python.exe stage0_manifest.py --root D:/Few-shot`.
It reported **84,630 images: 74,082 train / 9,526 val / 1,022 test**, ending
`VERIFIED: matches demo.ipynb`. It only verified the existing manifest.

Additional checks passed: all five new notebook code cells compile; old cells
except the index are content-identical; CLI help exposes the flags used by C0;
all 190 source and normalized hashes match for each arm after extraction;
no personal email was added to the notebook diff; `git diff --check` is clean.
Colab/Drive execution and end-to-end head training are **unverified here**.

## Exactly what Lam runs

1. First make this code available on GitHub through the authorized integration
   workflow. This branch is committed locally only. Part 1 clones/pulls the
   repository's default branch: uploading the notebook alone does not publish
   its new Python entry points. Do not run against old scripts.
2. Fresh Colab runtime, **A100 / L4 / T4**. Run Part 1, then **3.1**, **3.2** and
   **3.4b** to restore the actual raw/basic/accepted images. Skip 3.3, GAN training
   3.4, classifier training 3.5, Stage 4 3.6 and Part 4. Part 2 alone supplies
   placeholders and is unsuitable for C0.
3. Run **C0.1** (input counts/free space/C0 restore), **C0.2** (normalize both
   arms), then **C0.3** (full fresh features and L7-axis diagnostic). Completed
   cache groups are saved atomically to `MyDrive/octfsl/c0/v1/<arm>/features/`.
4. Inspect `full_axis.csv` and `gate.json`. If the full recomputed framing AUC
   is ≥0.90, stop and decide next steps; the notebook prevents C0.4 training.
   Even if it passes, inspect the fixed current-axis result before interpreting
   the intervention as removing a source cue.
5. If proceeding under the fixed design, run **C0.4** (all 180 isolated paired
   heads), then **C0.5** (partial or complete summary). Heads default to CPU.
   Outputs: Drive `c0/v1/<arm>/runs/<variant>_s<seed>/`, plus `runs.csv`,
   `contrasts.csv`, provenance, logs and diagnostics. No Part 6 is needed for C0.
6. After disconnect, repeat the same setup/real-input restore and C0.1–C0.3;
   immutable manifests/inventories and completed caches are checked/reused.
   C0.4 skips provenance-matched completed runs. Interrupted groups/heads restart.

Budget **15–60 min/arm GPU (45–180 min for three arms)** versus **2–8 h/arm CPU
(6–24 h total)** for fresh features, plus normalization/hashing/Drive transfer.
These Colab estimates are unbenchmarked; the bounded laptop run printed roughly
11–17 images/s. Reserve roughly **2 GB of additional Drive space** for caches,
inventories and all run artifacts; local PNG copies need extra runtime disk and
are rebuilt instead of uploaded. Heads: provisional **6–24 h total** at 2–8
min/head; use the saved first-head timing to plan sessions. A CPU runtime is
sufficient after feature extraction, but switching requires restoring real
inputs and C0 caches as described above. No Blackwell/RTX PRO 6000.

## Proposed shared-document text

README / CONTEXT: “C0 is a pending paired preprocessing sensitivity control:
current inputs, uniform grayscale PNG, and uniform grayscale PNG with square
padding, across none/basic/gan and seeds 1–20. Committed splits, existing basic
images and accepted GAN sets stay fixed. Standardized headers do not establish
removal of acquisition information: source and disease remain confounded. C1
input selection waits for C0.”

REPLICATION: “C0 code and isolated Colab cells are prepared. Training-free sample
checks remove header-size discrimination in the framing arm but retain residual
feature separation; the full feature check precedes head training. No C0 heads
have been trained by T6. Report the predeclared current-framing rare macro-recall
contrast and common-class control, all secondary contrasts and residual AUCs.”

COLAB: “For C0 use Part 1, 3.1, 3.2, 3.4b, then C0.1–C0.5. Use A100/L4/T4 for
fresh features; CPU suffices for cached heads. Outputs live only in
MyDrive/octfsl/c0/v1. The restore-only placeholder workflow does not supply C0
images. Fresh features are required per arm; do not reuse results/features.”

## Open questions and completion checklist

- Confirm the final C1 sensitivity arm retains the same **seeds 1–20**. The
  available C1 implementation uses them; the final sensitivity design was not
  present in the design worktree at inspection time. Reconcile before training.
- Full recomputed and fixed-axis AUCs remain unknown. If separation persists,
  Lam must decide whether to run this as a limited preprocessing sensitivity
  test or design a separately named crop/source-matched control. Do not retune
  these arms after results. No source-independent causal claim is supported.
- Image-level and patient-level source/disease confounding remains; padding
  cannot resolve it. Existing generators were trained on the original domains.
- Before using Colab, check integration/publication of these entry points,
  Drive space, actual restored pools and the full diagnostic gate.
- Inspection-command prevention proposed for Lam: “Discover actual filenames
  before reading them; for Windows Unicode notebook dumps set
  `sys.stdout.reconfigure(encoding='utf-8')`.” This is an agent workflow rule,
  not a useful project-code regression test. Logged; permanent rule not applied.

Hit Every Surface: **entry points** checked against cells/CLI; **reverse** is
selecting current inputs, with no destructive restore; **data/cache schemas**
isolated by arm, content/code/environment provenance checked; **docs** supplied
here instead of editing the shared files reserved for T1; **checks** pass above.

Harness/model: T3 Code / Codex, gpt-6.1-sol.
