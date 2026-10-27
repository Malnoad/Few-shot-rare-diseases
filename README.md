# Few-shot classification of rare retinal diseases in OCT

A reproduction of:

> Yoo TK, Choi JY, Kim HK. **Feasibility study to improve deep learning in OCT diagnosis of
> rare retinal diseases with few-shot classification.** *Medical & Biological Engineering &
> Computing* 59:401–415 (2021).
> [doi:10.1007/s11517-021-02321-1](https://doi.org/10.1007/s11517-021-02321-1) ·
> [PMC7829497](https://pmc.ncbi.nlm.nih.gov/articles/PMC7829497/)

The task is a **9-class** OCT classifier: the four common classes from the Kermany OCT2017
release (CNV, DME, DRUSEN, NORMAL, with tens of thousands of images each) plus **five rare
retinal diseases** (CSC, MH, MacTel, Stargardt, RP) with only **19–27 training images each**.
The paper's proposal is to close that gap with a **CycleGAN** trained per rare disease to
translate NORMAL OCT B-scans into that disease's appearance, then train an **Inception-v3**
classifier (ImageNet weights, frozen convolutional layers, only the final softmax layer
trained) on real + basic-augmented + GAN-generated images. This repo rebuilds that pipeline
from the raw public datasets as five ordered scripts.

---

## Status

| Stage | State |
|---|---|
| Stage 0 — manifest and splits | ✅ complete, verified |
| Stage 1 — basic augmentation | ✅ complete (2,000 images × 5 classes) |
| Stage 2 — CycleGAN | ✅ trained 100 epochs × 5 diseases, 4,000 generated each, filtered by automatic QC (no clinician review) |
| Stage 3 — classifier `none`, `basic`, `gan` | ✅ complete at seed 42, plus a 5-seed check |
| Stage 4 — metrics, baselines | ✅ complete |
| Stage 4 — CAM heatmaps | ✅ generated locally (`analysis.py cam`), not yet reviewed |
| Paper Table 2 (5-fold CV), Figs. 5 and 11, Siamese / prototypical networks | ❌ not done, see [`docs/research/REPLICATION.md`](docs/research/REPLICATION.md) |

**Short version:** at seed 42 the CycleGAN variant reaches 92.95% against the paper's 92.1%,
but over five training seeds it is not distinguishable from basic augmentation (`gan` ahead on
3 of 5 seeds). Both augmentations have higher mean accuracy than real images only (`gan`
by 1.9, `basic` by 2.2 points) and get more rare test images right, but five seeds cannot
separate the accuracy gains from training noise.
Paper item by item, with the work left: [`docs/research/REPLICATION.md`](docs/research/REPLICATION.md).
How to run everything on Colab: [`COLAB.md`](COLAB.md).

---

## Results

Numbers are taken from [`results/stage4/metrics.csv`](results/stage4/metrics.csv),
[`results/stage4/recall.csv`](results/stage4/recall.csv),
[`results/seed_check.csv`](results/seed_check.csv) and
[`results/analysis/`](results/analysis/), rounded for display only. Paper values are from
Tables 3–4 as encoded in [`stage4_evaluate.py`](stage4_evaluate.py).

Test set: **1,022 images** — 250 per common class, and 5/5/4/4/4 held-out images for
CSC/MH/MacTel/Stargardt/RP.

### Headline metrics (seed 42)

| Model | Accuracy % | 95% CI | Paper acc % | κ | Paper κ | MCC | MCC 95% CI | Paper MCC | RCI † |
|---|---:|:---:|---:|---:|---:|---:|:---:|---:|---:|
| `none` — real rare images only | **90.31** | 88.6–92.1 | 88.4 | 0.872 | 0.847 | 0.879 | 0.859–0.900 | 0.848 | 0.788 |
| `basic` — + basic augmentation | **88.85** | 86.9–90.6 | 91.3 | 0.853 | 0.886 | 0.861 | 0.838–0.882 | 0.887 | 0.760 |
| `gan` — + CycleGAN *(proposed)* | **92.95** | 91.4–94.5 | 92.1 | 0.907 | 0.896 | 0.909 | 0.890–0.929 | 0.897 | 0.814 |
| `knn` — K=3 on frozen features | 86.40 | 84.3–88.5 | 91.2 | 0.820 | 0.885 | 0.827 | 0.803–0.852 | 0.886 | 0.662 |
| `proto` — class-mean on frozen features | 68.98 | 66.0–71.9 | 80.8 ‡ | 0.600 | 0.753 ‡ | 0.604 | 0.569–0.640 | 0.755 ‡ | 0.430 |

† **RCI is not comparable to the paper** — see *Differences from the paper*. Treat accuracy,
κ and MCC as the primary comparison.
‡ The paper's row is a trained **prototypical network**; ours is a nearest-class-mean
baseline. Not a like-for-like comparison — see *Differences from the paper*.

### Five training seeds

One seed is not enough: the Stage 3 seed (head initialisation, data order, which augmented
images go to validation) alone moves accuracy by several points. Seeds 1–5 for each variant, from [`results/seed_check.csv`](results/seed_check.csv):

| Variant | Accuracy % mean ± SD | Range | Rare images correct, of 22 (range) | DRUSEN recall (mean) |
|---|---:|:---:|:---:|---:|
| `none` | 90.69 ± 1.86 | 87.87–92.86 | 14.2 (11–17) | 0.785 |
| `basic` | 92.93 ± 1.85 | 90.90–95.79 | 18.0 (17–19) | 0.877 |
| `gan` | 92.59 ± 1.29 | 91.49–94.13 | 18.6 (18–19) | 0.861 |

- **Augmentation of either kind helps the rare classes:** 17–19 of 22 correct, against 11–17
  with real images only.
- **CycleGAN and basic augmentation are not separated** (which is not the same as shown to be
  equal). Per seed, `gan` − `basic` is −0.35 ± 2.77 points (−4.30 to +2.94); `gan` is ahead on
  3 of 5 seeds ([`paired_seeds.csv`](results/analysis/paired_seeds.csv)).
- **Neither augmentation is separated from `none` on accuracy either:** `basic` − `none` is
  +2.25 ± 3.53 points (ahead on 3 of 5 seeds), `gan` − `none` +1.90 ± 2.22 (4 of 5). With five
  seeds the smallest possible two-sided Wilcoxon p is 0.0625, so no seed comparison here can
  reach p < 0.05.
- **The seed-42 ordering is not robust.** On the seed-42 test set `gan` beats `basic` by 4.1
  points (paired bootstrap CI 2.6–5.8, McNemar p < 0.001,
  [`paired_seed42.csv`](results/analysis/paired_seed42.csv)), but that test only covers
  test-image sampling, not training randomness, and `basic` at seed 42 is its lowest seed.

### Per-class recall (seed 42)

| Model | NORMAL | CNV | DME | DRUSEN | CSC | MH | MacTel | Stargardt | RP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.984 | 0.992 | 0.936 | 0.712 | 0.800 | 0.800 | 0.750 | 0.750 | 0.750 |
| `basic` | 0.992 | 0.996 | 0.852 | 0.720 | 0.800 | 1.000 | 0.500 | 1.000 | 0.750 |
| `gan` | 0.972 | 0.984 | 0.916 | 0.852 | 0.800 | 1.000 | 0.750 | 1.000 | 0.750 |
| `knn` | 0.996 | 0.988 | 0.696 | 0.832 | 0.600 | 0.400 | 0.000 | 0.000 | 0.000 |
| `proto` | 0.880 | 0.712 | 0.512 | 0.664 | 0.800 | 0.800 | 0.250 | 0.750 | 0.250 |

**Rare-class recall moves in steps of 0.20–0.25** — there are only 4–5 test images per rare
class. One image changes the number.

Our `none` already gets 0.75–0.80 on every rare class, where the paper's Table 4 reports
0.00–0.25 for its model without augmentation. So the paper's largest effect, augmentation
rescuing rare classes the baseline misses, is much smaller here. The cause has not been
investigated.

### What the confusion matrices show (seed 42)

From [`confusion_none.csv`](results/stage4/confusion_none.csv),
[`confusion_basic.csv`](results/stage4/confusion_basic.csv) and
[`confusion_gan.csv`](results/stage4/confusion_gan.csv):

| | `none` | `basic` | `gan` |
|---|---:|---:|---:|
| Rare-class images correct (of 22) | 17 | 18 | **19** |
| Common-class images correct (of 1,000) | 906 | 890 | **931** |
| Common-class images predicted as a *rare* class | 0 | 0 | 1 |
| DRUSEN predicted as CNV (of 250) | 68 | 64 | **36** |
| DME correct (of 250) | **234** | 213 | 229 |

1. **Most of `gan`'s seed-42 gain is in a common class, DRUSEN**, not in the rare classes:
   DRUSEN→CNV errors fall from 68 to 36. Over seeds, `basic` reaches a similar DRUSEN recall
   (0.877 vs 0.861), so this is not specific to CycleGAN images.
2. **Augmentation does not pull common-class images into the rare classes** (0–1 images in
   every variant; `proto` does it 66 times).
3. `basic`'s seed-42 loss against `none` is concentrated in DME (234 → 213 correct).

### One-vs-rest AUC

See [`results/stage4/auc.csv`](results/stage4/auc.csv) and the curve points in
[`results/analysis/roc_points.csv`](results/analysis/roc_points.csv). `knn` and `proto` AUCs
are computed from discretised vote/distance scores, so they are not comparable to the
softmax AUCs of `none`/`basic`/`gan`.

### Differences from the paper

| Paper | Here | Effect |
|---|---|---|
| Two ophthalmologists removed bad CycleGAN images | Automatic filter (`stage2_cyclegan.py qc`): keep images closer to real disease than to NORMAL in Inception-v3 feature space | No clinician has reviewed the accepted images |
| 3,000 accepted GAN images per disease | CSC 3,000, MH 2,610, MacTel 1,248, Stargardt 2,199, RP 832 passed QC | Four classes below target; Stage 3 warns and continues |
| 250 epochs | Early stopping on validation loss, patience 20 (`--patience 0` runs all 250) | Heads stopped after 22–27 epochs |
| Trained prototypical network, Siamese network | Nearest class-mean (`proto`); no Siamese network | `proto` row not comparable |
| kNN K=3 on "Inception-v3 trained without augmentation" | K=3 Euclidean on frozen ImageNet features | 86.4 vs 91.2%; cosine K=1 gives 91.1% ([`knn_sweep.csv`](results/analysis/knn_sweep.csv)), cause not established |
| RCI 0.92–0.98 | RCI = I(true; pred) / H(true) from the confusion matrix, 0.76–0.81 | The cited definition (Sindhwani 2001 via the paper's ref. 38) is ours, and the paper's gan 0.983 is above the 0.970 maximum that definition allows with its own Table 4 recalls; formula actually used unknown ([`rci_paper_bounds.csv`](results/analysis/rci_paper_bounds.csv), REPLICATION.md L8) |
| Table 2: 5-fold CV with GANs retrained per fold | Not done (25 CycleGANs, ~50 A100-hours) | Only the independent-test comparison (Table 3) is reproduced |
| Basic augmentation in MATLAB | albumentations, same transform ranges | |

---

## Data download

Two public datasets, neither redistributed here. Download both and arrange them exactly as
shown below.

### 1. Kermany OCT2017 — four common classes

- <https://data.mendeley.com/datasets/rscbjbr9sj/2>
- DOI: [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2)
- Kermany DS, Goldbaum M, Cai W, et al. *Identifying medical diagnoses and treatable
  diseases by image-based deep learning.* Cell 172(5):1122–1131 (2018).

### 2. Rare-disease OCT — five rare classes

- <https://data.mendeley.com/datasets/btv6yrdbmv>
- DOI: `10.17632/btv6yrdbmv` (append the version suffix Mendeley shows on the page, e.g.
  `.1`, when citing)
- This is the dataset released alongside the Yoo et al. paper.

### Target layout

```
data/
  raw/
    OCT2017/
      train/{CNV,DME,DRUSEN,NORMAL}/*.jpeg
      test/ {CNV,DME,DRUSEN,NORMAL}/*.jpeg
    rare/
      CSC/*.PNG
      MacTel/*.PNG
      MH/*.PNG
      Stargardt/*.PNG
      RP/*.PNG
```

### Cleanup pitfalls (all of these bit us)

These archives were zipped on macOS and the folder names are inconsistent. Fix all of it
**before** running Stage 0, or the manifest will silently be built from the wrong files.

1. **Delete every `__MACOSX` folder.** Both archives contain one. It mirrors the real tree
   with `._`-prefixed stub files that look like images to a glob.
2. **Delete `._*` and `.DS_Store` files** anywhere under `data/raw/`.
3. **Rename folders to exactly `MacTel` and `Stargardt`.** The archive ships them as
   `Mactel` (lowercase t) and `Stargart` (missing the `d`). The scripts match these names
   case-sensitively on Linux/Colab and will find zero images otherwise.
4. **Make sure OCT2017 is not unzipped inside `rare/`.** A double-click extract can land it
   at `data/raw/rare/OCT2017/`, where Stage 0 will not see it and `rare/` gains a bogus
   sixth class.
5. **`RP` may be a separate download** from the other four rare classes — check you have all
   five folders before starting.

On Windows PowerShell:

```powershell
Get-ChildItem data\raw -Recurse -Force -Directory -Filter __MACOSX | Remove-Item -Recurse -Force
Get-ChildItem data\raw -Recurse -Force -File | Where-Object { $_.Name -like '._*' -or $_.Name -eq '.DS_Store' } | Remove-Item -Force
```

### Not used by this pipeline

The downloads also contain `Segmentation_manual/` (72 manual segmentation PNGs) and a
`ChestXRay2017/` chest X-ray set. **Neither is used by any of the five scripts**; they are
ignored by `.gitignore` and can be deleted if you are short on disk.

Full raw data is roughly **5.6 GB**; the working folders Stage 1 and 2 add bring the project
to about **8.7 GB**.

---

## Environment

Verified on **Python 3.13.4** with **TensorFlow 2.20.0**, Windows 11, and on Colab
(Python 3.13, TensorFlow 2.20.0, October 2026). `analysis.py` was run on Python 3.12 with
TensorFlow 2.21.0 (CPU), Windows 11. The Colab workflow, including what lives in Google
Drive, is in [`COLAB.md`](COLAB.md) and [`colab_setup.ipynb`](colab_setup.ipynb).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) declares `numpy`, `pandas`, `scikit-learn`,
`opencv-python`, `albumentations>=2.0` and `tensorflow>=2.16`. The TensorFlow floor is 2.16
because Stage 3 writes Keras-3 style weights (`best.weights.h5`). pip resolves to the newest
TensorFlow with wheels for your Python version.

**TensorFlow has no GPU support on native Windows** (dropped after TF 2.10). The practical
consequence:

| Step | Where it runs |
|---|---|
| Stage 0, Stage 1, `stage2_cyclegan.py prep` | CPU, any OS. No TensorFlow needed for `prep`. |
| **Stage 2 `train` / `generate` / `qc`** | **GPU required in practice — use Colab or WSL2.** |
| Stage 3, Stage 4 | CPU is fine (slow but workable); GPU is faster. |
| `analysis.py` | CPU, seconds to a minute; trains nothing. `matplotlib` optional, for PNG figures. |

Every script takes `--root` if you are not running from the project root. All seeds are
fixed at **42**.

---

## Step-by-step run guide

Run everything from the project root, in order.

### Stage 0 — build and verify the manifest

**What and why.** Builds `data/manifest.csv` with columns `path, label, pid, split`. It uses
the Kermany official train/test split (250 test images per common class; the split is
**not** patient-disjoint, see the note below) and, for the rare classes, a **fixed seed-42
hold-out of 5/5/4/4/4 images** matching the paper's Fig. 2. A 10% validation set is carved from train **grouped by patient**
(`GroupShuffleSplit`, seed 42) so no patient spans train and val. Rare-class validation
images are then moved back into train, so that every real rare image is available to
Stage 1 and the CycleGAN — as in the paper, rare-class validation comes from *augmented*
data in Stage 3.

```bash
python stage0_manifest.py            # build if missing, otherwise verify only
python stage0_manifest.py --force    # rebuild (only if the raw data changed)
```

- **Input:** `data/raw/OCT2017/`, `data/raw/rare/`
- **Output:** `data/manifest.csv` (84,630 rows, ~5.7 MB — committed to this repo)
- **Runtime:** under a minute.
- **Check it worked:** it prints a crosstab and ends with **`VERIFIED: matches demo.ipynb`**,
  and exits 0. Compare the crosstab against *Expected counts* below. Any `MISMATCH <label>`
  line means your raw data differs — re-read the cleanup pitfalls.

The script never overwrites an existing manifest without `--force`, because Stages 1 and 2
were built from it.

> A note it may print: some patient IDs appear in both Kermany train and test (566 of the 633
> patients in our common-class test set, covering 914 of its 1,000 images, per
> `data/manifest.csv`). That comes from the published release itself and is deliberately left
> as-is, not fixed.

### Stage 1 — basic augmentation

**What and why.** The paper's "basic augmentation" baseline: expand each rare class from its
19–27 real training images to **2,000 images at 256×256**, using horizontal flip, ±5% shift,
±30° rotation, 0–20% zoom, ±10% brightness, and Gaussian elastic deformation
(σ=10, α=2). These images serve twice: as **CycleGAN domain B** in Stage 2, and as the rare
class training pool in Stage 3.

```bash
python stage1_basic_aug.py            # skip classes that already have 2,000 images
python stage1_basic_aug.py --force    # regenerate everything
```

Options: `--n 2000` (images per class), `--seed 42`, `--root`.

- **Input:** `data/manifest.csv`, `data/raw/rare/`
- **Output:** `data/aug_basic/<class>/<class>_0000.png … _1999.png` (10,000 files, ~584 MB)
  and preview sheets `data/aug_basic/_preview_<class>.jpg`
- **Runtime:** roughly 10–20 minutes on CPU for all five classes.
- **Check it worked:** ends with **`VERIFIED`**. Then **open the five preview sheets** —
  each is an 8×6 grid of 48 random samples. Retinal layers should bend and shift but stay
  continuous; if the elastic deformation has torn the layers apart, the augmentation is too
  strong.

### Stage 2 — CycleGAN per rare disease

**What and why.** The paper's core idea. One CycleGAN per rare disease learns NORMAL → that
disease, so an unlimited supply of synthetic rare-disease B-scans can be generated from the
abundant NORMAL images. Architecture and hyperparameters follow the TensorFlow CycleGAN
tutorial the authors used: U-Net generator + 70×70 PatchGAN discriminator with instance
norm, 256×256×3, Adam(2e-4, β₁=0.5), batch 1, cycle λ=10, identity 0.5λ, and the tutorial's
jitter (resize 286 → random crop 256 → random flip).
[`stage2_cyclegan.py`](stage2_cyclegan.py) is self-contained — it does **not** need
`tensorflow_examples`.

Four subcommands.

#### 2a. `prep` — local, no TensorFlow

```bash
python stage2_cyclegan.py prep
```

Options: `--n-domain 2000` (NORMAL images for CycleGAN domain A), `--n-source 5000`
(a *separate* set of NORMAL images to translate later), `--seed 42`.

- **Output:** `data/gan/normalA/` (2,000), `data/gan/normal_src/` (5,000),
  `data/gan/real/<class>/` (the real rare *training* images only — the test hold-out never
  touches the GAN)
- **Runtime:** a few minutes.
- **Check it worked:** prints a count per set, and a line per rare class confirming
  `aug_basic/<class>` has 2,000 images.

#### 2b–2d. `train`, `generate`, `qc` — GPU

```bash
python stage2_cyclegan.py train    --disease CSC --epochs 100
python stage2_cyclegan.py generate --disease CSC --n 4000
python stage2_cyclegan.py qc       --disease CSC --keep 3000   # optional: --margin 0.01
```

`--disease` is required and must be one of `CSC MH MacTel Stargardt RP`. `train` and
`generate` also accept `--ckpt` to put checkpoints somewhere other than
`data/gan/ckpt/<disease>` — **use this on Colab to write checkpoints to Drive.**

- **Outputs:** `data/gan/ckpt/<class>/` (checkpoints + `samples/ep_XXX.png`),
  `data/gan/out/<class>/` (raw generated), `data/gan/accepted/<class>/` (passed QC → Stage 3),
  `data/gan/qc_<class>.csv`, `data/gan/qc_<class>_kept.jpg`, `data/gan/qc_<class>_rejected.jpg`
- **Runtime** (from [`results/stage2/stage2.log`](results/stage2/stage2.log)): 2,000
  steps/epoch took about 3.1 min on a Colab T4 and 1.0–1.1 min on an A100, so 100 epochs is
  about 2 h per disease on an A100 and over 5 h on a T4. Lean on the resume mechanism.
- **Check it worked:**
  - `train` writes a sample grid every epoch to `data/gan/ckpt/<class>/samples/ep_XXX.png` —
    left column is the real NORMAL input, right column is `G(normal)`. Watch these across
    epochs; the right column should acquire the disease's hallmark (e.g. subretinal fluid
    for CSC, a full-thickness defect for MH) while keeping plausible retinal layers.
  - `qc` keeps images whose margin `d_normal − d_rare` (kNN distances in Inception-v3 feature
    space) exceeds `--margin` (default 0: closer to real disease than to NORMAL), most
    disease-like first, and prints `N/M above margin; kept K`. Then **open the contact
    sheets** `qc_<class>_kept.jpg` and `qc_<class>_rejected.jpg` and delete anything
    unconvincing from `data/gan/accepted/<class>/` by hand.
  - If `qc` warns `only K < 3000`, generate more (`--n` higher; rerun `prep` with a larger
    `--n-source` if you run out of source images).

#### Stage 2 on Colab

Run it through [`colab_setup.ipynb`](colab_setup.ipynb) (Part 3.4), which restores the
inputs from Drive, writes checkpoints to `MyDrive/octfsl/ckpt/<class>` and resumes after a
disconnect: rerunning the same cell prints `Resumed <class> from … (epoch N)` and continues,
with generators, discriminators, all four optimizer states and the epoch counter restored.
See [`COLAB.md`](COLAB.md) for runtimes, Drive layout and troubleshooting.

The accepted images used for the `gan` results were selected with the rule that predates
`--margin` (sort passing images by `d_rare`); their QC records are in
[`results/stage2/`](results/stage2/). Rerunning `qc` with the current code would change the
kept set for CSC (3,370 passed, 3,000 kept).

### Stage 3 — Inception-v3 classifier

**What and why.** The paper's classifier: ImageNet Inception-v3, **convolutional layers
frozen**, only the final fully-connected softmax layer trained; input 299×299×3; Adam;
categorical cross-entropy; batch 10; 250 epochs; a tenth of the training data for
validation. Because the backbone is frozen and there is no on-the-fly augmentation, the
2048-d features are **extracted once and cached**, and the softmax layer is trained on the
cache. That is the same model as an end-to-end run, in minutes rather than the paper's
~150 h. All images take the same 256→299 resize path, so sharpness cannot leak the class.

```bash
python stage3_classifier.py --variant none    # rare train = real images only (19-27/class)
python stage3_classifier.py --variant basic   # rare train = 2,000 basic-augmented
python stage3_classifier.py --variant gan     # + 3,000 accepted CycleGAN images  <- proposed
```

Options: `--epochs 250`, `--batch 10`, `--lr 1e-3`, `--patience 20` (early stopping on val
loss; `0` = run all epochs as in the paper), `--gan-per-class 3000`, `--seed 42`,
`--feat-batch 32` (lower it if feature extraction runs out of memory), `--root`.

- **Input:** `data/manifest.csv`, `data/aug_basic/`, and for `--variant gan`,
  `data/gan/accepted/`
- **Output:** `results/features/*.npz` (shared cache, ~390 MB, git-ignored),
  `results/stage3_<variant>/` containing `head.npz`, `history.csv`, `data_counts.csv`,
  `val_predictions.csv`, `test_predictions.csv`, `best.weights.h5`
- **Runtime:** the **first** run is dominated by extracting features for ~84,600 images —
  several hours on CPU. Later runs and other variants reuse the cache and take minutes. The
  script prints a live `img/s` rate and an ETA while extracting.
- **Check it worked:** it prints the per-class train/val/test counts (compare to
  `data_counts.csv` below), then the final line
  `Test accuracy (<variant>): XX.X%  (paper: none 88.4, basic 91.3, gan 92.1)` plus
  per-class recall.

Run `--variant none` **first**: it caches `major_train`, `test` and `real_<class>` features,
which Stage 4's KNN and class-mean baselines need.

`data_counts.csv` per variant ([`none`](results/stage3_none/data_counts.csv) ·
[`basic`](results/stage3_basic/data_counts.csv) ·
[`gan`](results/stage3_gan/data_counts.csv)) — common classes are identical in every
variant; only the rare rows change:

| Class | `none` train / val | `basic` train / val | `gan` train / val | test |
|---|---:|---:|---:|---:|
| NORMAL | 23,536 / 2,779 | 23,536 / 2,779 | 23,536 / 2,779 | 250 |
| CNV | 32,536 / 4,669 | 32,536 / 4,669 | 32,536 / 4,669 | 250 |
| DME | 10,153 / 1,195 | 10,153 / 1,195 | 10,153 / 1,195 | 250 |
| DRUSEN | 7,733 / 883 | 7,733 / 883 | 7,733 / 883 | 250 |
| CSC | 27 / 0 | 1,800 / 200 | 4,500 / 500 | 5 |
| MH | 26 / 0 | 1,800 / 200 | 4,149 / 461 | 5 |
| MacTel | 25 / 0 | 1,800 / 200 | 2,924 / 324 | 4 |
| Stargardt | 19 / 0 | 1,800 / 200 | 3,780 / 419 | 4 |
| RP | 27 / 0 | 1,800 / 200 | 2,549 / 283 | 4 |

The `none` variant has **no rare validation images** — there are too few real ones to spare.
The `gan` pool is the 2,000 basic images plus every accepted CycleGAN image (up to
`--gan-per-class`), so it is below 5,000 for the four classes short of 3,000 accepted images.

**Before `--variant gan`, check that `data/gan/accepted/<class>/` is populated.** If it is
empty, Stage 3 only prints a warning and trains what is effectively `basic`.

### Stage 4 — evaluation, baselines and CAM

**What and why.** Computes the paper's Table 3–4 metrics on the independent test set:
accuracy, unweighted Cohen's κ, RCI, multiclass MCC (Gorodkin), per-class recall,
one-vs-rest AUC, and bootstrap 95% CIs (1,000 resamples) for accuracy and MCC. It also adds
two few-shot baselines computed on the same frozen ImageNet features: **`knn`** (K=3
Euclidean, after Quellec et al.) and **`proto`** (nearest class-mean). It evaluates whichever
`results/stage3_<variant>/test_predictions.csv` files exist.

```bash
python stage4_evaluate.py
python stage4_evaluate.py --cam 40                      # + CAM heatmaps, 40 test images
python stage4_evaluate.py --cam 40 --cam-variant basic  # CAM for a specific variant
python stage4_evaluate.py --no-baselines                # skip knn / proto
```

Options: `--no-baselines`, `--cam N` (default 0), `--cam-variant {none,basic,gan}`
(default `gan`), `--root`.

- **Input:** `results/stage3_*/test_predictions.csv`, `results/features/*.npz`,
  `data/manifest.csv`
- **Output:** `results/stage4/` — `metrics.csv`, `recall.csv`, `auc.csv`,
  `confusion_<model>.csv`, and `cam_<variant>/` if `--cam` was used
- **Runtime:** a minute or two without `--cam`.
- **Check it worked:** prints three tables (metrics vs paper, per-class recall, AUC) and
  `Saved CSVs to results/stage4`. CAM panels are named `<true>_as_<predicted>_<stem>.jpg`,
  each a side-by-side of the input and the heatmap overlay; rare-class test images come
  first.

With a frozen backbone + global average pooling + one softmax layer, Grad-CAM reduces
exactly to CAM, so the heatmap is computed directly as `ReLU(Σₖ W[k,c]·Aₖ)` over the
`mixed10` feature maps.

> Rerunning Stage 4 rewrites `results/stage4/*.csv`. Those files are the run of record
> (seed 42, Colab); a local rerun can differ in late decimals (scikit-learn versions shift
> the `knn` AUC in the 5th decimal), so don't commit a local rerun over them. For CAM alone,
> use `analysis.py cam`, which writes elsewhere.

### Training-free analysis — `analysis.py`

Reads only what Stages 2–4 already produced (QC records, test predictions, the features
cache, `results/seed_check.csv`) and trains nothing, so it runs on a laptop CPU in about
30 s. It never writes to `results/stage4/`.

```bash
python analysis.py compare        # paired tests none/basic/gan: seed-42 test set and over seeds
python analysis.py knn            # kNN baseline sensitivity: K and Euclidean vs cosine
python analysis.py qc             # CycleGAN acceptance rates (paper Fig. 6)
python analysis.py roc            # one-vs-rest ROC curve points (Fig. 9)
python analysis.py tsne           # t-SNE of frozen features, real-only vs + CycleGAN pool (Fig. 7)
python analysis.py cam --n 40     # CAM panels from a trained head, rare test images first (Fig. 10)
python analysis.py rci            # alternative RCI definitions + RCI range the paper's Table 4 allows (~2 min)
python analysis.py all            # all of the above except cam and rci
```

- **Input:** `results/stage3_*/test_predictions.csv`, `results/features/*.npz` (`knn`,
  `tsne`, `cam`), `results/stage2/qc_*.csv`, `results/seed_check.csv`, `data/manifest.csv`
- **Output:** `results/analysis/` — `paired_seed42.csv`, `paired_seeds.csv`, `knn_sweep.csv`,
  `qc_summary.csv`, `roc_points.csv`, `tsne.csv`, `cam_<variant>/`; plus `roc.png` and
  `tsne.png` if `matplotlib` is installed
- `--root` points at a checkout that holds the data, `--out` at another output folder.

---

## Expected counts

Stage 0 hard-codes and verifies these counts. If your crosstab differs, your raw data is
not arranged correctly — go back to the cleanup pitfalls.

| Label | test | train | val |
|---|---:|---:|---:|
| CNV | 250 | 32,536 | 4,669 |
| DME | 250 | 10,153 | 1,195 |
| DRUSEN | 250 | 7,733 | 883 |
| NORMAL | 250 | 23,536 | 2,779 |
| CSC | 5 | 27 | 0 |
| MH | 5 | 26 | 0 |
| MacTel | 4 | 25 | 0 |
| RP | 4 | 27 | 0 |
| Stargardt | 4 | 19 | 0 |
| **All** | **1,022** | **74,082** | **9,526** |

Total: **84,630** images.

---

## Repository structure

Only code and small result files are tracked. Data, caches, checkpoints and third-party
material are git-ignored — see [`.gitignore`](.gitignore). Large files live in Google Drive
(`MyDrive/octfsl`, see [`COLAB.md`](COLAB.md)).

```
.
├── stage0_manifest.py          build/verify data/manifest.csv
├── stage1_basic_aug.py         basic augmentation -> data/aug_basic/
├── stage2_cyclegan.py          prep | train | generate | qc
├── stage3_classifier.py        Inception-v3, variants none|basic|gan
├── stage4_evaluate.py          metrics, knn/proto baselines, CAM
├── analysis.py                 training-free analyses (paired tests, kNN, QC, ROC, t-SNE, CAM)
├── colab_setup.ipynb           Colab: restore, rebuild, seed check, CAM, save and push
├── COLAB.md                    Colab and Drive guide
├── AGENTS.md, CLAUDE.md        instructions for coding agents (.agents/, .claude/, .codex/)
├── docs/research/              CONTEXT.md (research context), REPLICATION.md (paper vs repo, work left)
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   └── manifest.csv            tracked: 84,630 rows, defines every split
└── results/
    ├── stage2/                 qc_<class>.csv + contact sheets, stage2.log
    ├── stage3_{none,basic,gan}/ head.npz, test_predictions.csv, data_counts.csv, history.csv
    ├── stage4/                 metrics.csv, recall.csv, auc.csv, confusion_<model>.csv
    ├── analysis/               analysis.py outputs (CSVs)
    └── seed_check.csv          seeds 1-5 per variant

not tracked (regenerate locally, or restore from Drive):
    data/raw/  data/aug_basic/  data/gan/   ~8.7 GB of images
    results/features/*.npz                  cached backbone features (~390 MB)
    results/stage3_*/{best.weights.h5,val_predictions.csv}
    results/stage4/cam_*/  results/analysis/cam_*/   CAM panels
    paper/  code2017/  ChestXRay2017/       third-party material
```

A fresh clone can run `analysis.py compare|qc|roc|rci` straight away, because the test
predictions and Stage 4 confusion matrices are tracked. The `knn`/`proto` baselines,
`analysis.py knn|tsne|cam` and Stage 3 need the features cache (Stage 3 `--variant none`,
or restore it from Drive).

---

## Citation and licensing

**This code** is released under the MIT License — see [`LICENSE`](LICENSE).

**The datasets are not redistributed here** and keep their own licenses. Download them from
Mendeley Data directly and cite them:

```bibtex
@article{yoo2021feasibility,
  title   = {Feasibility study to improve deep learning in {OCT} diagnosis of
             rare retinal diseases with few-shot classification},
  author  = {Yoo, Tae Keun and Choi, Joon Yul and Kim, Hong Kyu},
  journal = {Medical \& Biological Engineering \& Computing},
  volume  = {59}, pages = {401--415}, year = {2021},
  doi     = {10.1007/s11517-021-02321-1}
}

@article{kermany2018identifying,
  title   = {Identifying medical diagnoses and treatable diseases by
             image-based deep learning},
  author  = {Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and others},
  journal = {Cell},
  volume  = {172}, number = {5}, pages = {1122--1131}, year = {2018},
  doi     = {10.1016/j.cell.2018.02.010}
}
```

The CycleGAN implementation in [`stage2_cyclegan.py`](stage2_cyclegan.py) is adapted from
the [TensorFlow CycleGAN tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan)
(Apache 2.0), reimplemented self-contained so that `tensorflow_examples` is not required.

This is an independent reproduction. It is not affiliated with or endorsed by the authors of
the paper, and it is **not a medical device** — research use only.
