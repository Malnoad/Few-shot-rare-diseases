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

## ⚠️ Status: partial reproduction, work in progress

| Stage | State |
|---|---|
| Stage 0 — manifest and splits | ✅ complete, verified |
| Stage 1 — basic augmentation | ✅ complete (2,000 images × 5 classes) |
| Stage 2 — CycleGAN | ⏳ **`prep` only. No CycleGAN has been trained yet.** |
| Stage 3 — classifier `none`, `basic` | ✅ complete |
| Stage 3 — classifier `gan` | ⏳ **pending, blocked on Stage 2** |
| Stage 4 — metrics, baselines | ✅ complete for the variants that exist |
| Stage 4 — CAM heatmaps | ⏳ not generated |

**The headline result of the paper — the `gan` variant — has not been run here.** Everything
below reports only what was actually computed. Nothing is extrapolated.

---

## Results

All numbers are taken verbatim from [`results/stage4/metrics.csv`](results/stage4/metrics.csv)
and [`results/stage4/recall.csv`](results/stage4/recall.csv), rounded for display only; the
CSVs hold full precision. Paper values are from Tables 3–4 as encoded in
[`stage4_evaluate.py`](stage4_evaluate.py).

Test set: **1,022 images** — 250 per common class, and 5/5/4/4/4 held-out images for
CSC/MH/MacTel/Stargardt/RP.

### Headline metrics

| Model | Accuracy % | 95% CI | Paper acc % | κ | Paper κ | MCC | MCC 95% CI | Paper MCC | RCI † |
|---|---:|:---:|---:|---:|---:|---:|:---:|---:|---:|
| `none` — real rare images only | **90.31** | 88.6–92.1 | 88.4 | 0.872 | 0.847 | 0.879 | 0.859–0.900 | 0.848 | 0.788 |
| `basic` — + basic augmentation | **88.85** | 86.9–90.6 | 91.3 | 0.853 | 0.886 | 0.861 | 0.838–0.882 | 0.887 | 0.760 |
| `gan` — + CycleGAN *(proposed)* | *pending, Stage 2 (CycleGAN) not yet trained* | — | 92.1 | — | 0.896 | — | — | 0.897 | — |
| `knn` — K=3 on frozen features | 86.40 | 84.3–88.5 | 91.2 | 0.820 | 0.885 | 0.827 | 0.803–0.852 | 0.886 | 0.662 |
| `proto` — class-mean on frozen features | 68.98 | 66.0–71.9 | 80.8 ‡ | 0.600 | 0.753 ‡ | 0.604 | 0.569–0.640 | 0.755 ‡ | 0.430 |

† **RCI is not comparable to the paper** — see *Differences from the paper*. Treat accuracy,
κ and MCC as the primary comparison.
‡ The paper's row is a trained **prototypical network**; ours is a nearest-class-mean
baseline. Not a like-for-like comparison — see *Differences from the paper*.

### Per-class recall

| Model | NORMAL | CNV | DME | DRUSEN | CSC | MH | MacTel | Stargardt | RP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.984 | 0.992 | 0.936 | 0.712 | 0.800 | 0.800 | 0.750 | 0.750 | 0.750 |
| `basic` | 0.992 | 0.996 | 0.852 | 0.720 | 0.800 | 1.000 | 0.500 | 1.000 | 0.750 |
| `gan` | *pending, Stage 2 (CycleGAN) not yet trained* | | | | | | | | |
| `knn` | 0.996 | 0.988 | 0.696 | 0.832 | 0.600 | 0.400 | 0.000 | 0.000 | 0.000 |
| `proto` | 0.880 | 0.712 | 0.512 | 0.664 | 0.800 | 0.800 | 0.250 | 0.750 | 0.250 |

The paper's per-class recall values are not recorded in this repo, so no paper column is
shown here rather than transcribing them by hand.

**Rare-class recall moves in steps of 0.20–0.25** — there are only 4–5 test images per rare
class. One image changes the number.

### `basic` vs `none`: what the confusion matrices actually show

`basic` scores 1.46 points lower than `none` overall, but **the two accuracy CIs overlap**
(88.6–92.1 vs 86.9–90.6), so the difference is not separated by this test set. Breaking the
1,022 test images down:

| | `none` | `basic` |
|---|---:|---:|
| Rare-class images correct (of 22) | 17 | **18** |
| Common-class images correct (of 1,000) | **906** | 890 |
| Common-class images predicted as a *rare* class | **0** | **0** |

Three factual observations, from
[`confusion_none.csv`](results/stage4/confusion_none.csv) and
[`confusion_basic.csv`](results/stage4/confusion_basic.csv):

1. **No common-class image is predicted as a rare class in either variant.** Every cell in
   the CNV/DME/DRUSEN/NORMAL rows under the five rare columns is zero. Adding 2,000
   augmented images per rare class did not pull common-class images into the rare classes.
   (For contrast, `proto` does this 66 times; `knn` does it 0 times.)
2. **The entire net change is inside the common classes, and it is concentrated in DME.**
   DME correct falls 234 → 213. DME→CNV rises 13 → 25 and DME→NORMAL rises 2 → 11. The
   other three common classes each move by ≤ 2 images.
3. **The largest single error mode in both variants is DRUSEN→CNV** — 68 images in `none`,
   64 in `basic`, out of 250 DRUSEN. This dominates both confusion matrices and is
   essentially unchanged by the augmentation.

Rare-class recall per class, `none` → `basic`: CSC 4/5 → 4/5, MH 4/5 → **5/5**,
MacTel 3/4 → **2/4**, Stargardt 3/4 → **4/4**, RP 3/4 → 3/4.

No causal conclusion is drawn here. With 22 rare test images and overlapping CIs, this run
does not establish that basic augmentation helps or hurts.

### One-vs-rest AUC

See [`results/stage4/auc.csv`](results/stage4/auc.csv). Note that `knn` and `proto` AUCs are
computed from discretised vote/distance scores, so they are not comparable to the softmax
AUCs of `none`/`basic`.

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

Verified on **Python 3.13.4** with **TensorFlow 2.20.0**, Windows 11.

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

Every script takes `--root` if you are not running from the project root. All seeds are
fixed at **42**.

---

## Step-by-step run guide

Run everything from the project root, in order.

### Stage 0 — build and verify the manifest

**What and why.** Builds `data/manifest.csv` with columns `path, label, pid, split`. It uses
the Kermany official train/test split (independent patients, 250 test images per common
class) and, for the rare classes, a **fixed seed-42 hold-out of 5/5/4/4/4 images** matching
the paper's Fig. 2. A 10% validation set is carved from train **grouped by patient**
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

> A note it may print: some patient IDs appear in both Kermany train and test. That comes
> from the published release itself and is deliberately left as-is, not fixed.

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
python stage2_cyclegan.py qc       --disease CSC --keep 3000
```

`--disease` is required and must be one of `CSC MH MacTel Stargardt RP`. `train` and
`generate` also accept `--ckpt` to put checkpoints somewhere other than
`data/gan/ckpt/<disease>` — **use this on Colab to write checkpoints to Drive.**

- **Outputs:** `data/gan/ckpt/<class>/` (checkpoints + `samples/ep_XXX.png`),
  `data/gan/out/<class>/` (raw generated), `data/gan/accepted/<class>/` (passed QC → Stage 3),
  `data/gan/qc_<class>.csv`, `data/gan/qc_<class>_kept.jpg`, `data/gan/qc_<class>_rejected.jpg`
- **Runtime:** *not yet measured here.* 2,000 steps/epoch × 100 epochs per disease, times
  five diseases, is a multi-day job on a single Colab T4. Budget accordingly and lean on the
  resume mechanism.
- **Check it worked:**
  - `train` writes a sample grid every epoch to `data/gan/ckpt/<class>/samples/ep_XXX.png` —
    left column is the real NORMAL input, right column is `G(normal)`. Watch these across
    epochs; the right column should acquire the disease's hallmark (e.g. subretinal fluid
    for CSC, a full-thickness defect for MH) while keeping plausible retinal layers.
  - `qc` prints `N/M closer to real <class> than to NORMAL; kept K`. Then **open the contact
    sheets** `qc_<class>_kept.jpg` and `qc_<class>_rejected.jpg` and delete anything
    unconvincing from `data/gan/accepted/<class>/` by hand.
  - If `qc` warns `only K < 3000`, generate more (`--n` higher; rerun `prep` with a larger
    `--n-source` if you run out of source images).

#### Colab workflow for Stage 2

**1 — package the inputs locally and upload to Drive.** You need domain A, domain B, the
generation sources and the real rare images:

```powershell
Compress-Archive -Path data\aug_basic -DestinationPath aug_basic.zip
Compress-Archive -Path data\gan       -DestinationPath gan.zip
```

Upload both to `MyDrive/octfsl/`. (~1.4 GB total.)

**2 — Colab: mount Drive and unpack to local disk.** Unzip to `/content`, *not* to Drive —
training reads these thousands of times and Drive I/O is slow.

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/project/data
!unzip -q /content/drive/MyDrive/octfsl/aug_basic.zip -d /content/project/data
!unzip -q /content/drive/MyDrive/octfsl/gan.zip       -d /content/project/data
!git clone https://github.com/Malnoad/Few-shot-rare-diseases /content/code
!cp /content/code/stage2_cyclegan.py /content/project/
!ls /content/project/data/aug_basic /content/project/data/gan
```

**3 — train one disease, with checkpoints on Drive so a disconnect costs nothing.**

```python
!mkdir -p /content/drive/MyDrive/octfsl/ckpt
%cd /content/project
!python stage2_cyclegan.py train --disease CSC --epochs 100 \
    --ckpt /content/drive/MyDrive/octfsl/ckpt/CSC
```

**4 — resume after a disconnect: rerun the exact same command.** The script uses a
`tf.train.CheckpointManager` and restores generators, discriminators, *all four optimizer
states* and the epoch counter. It prints `Resumed CSC from … (epoch N)` and continues from
there. Repeat steps 3–4 for `MH`, `MacTel`, `Stargardt`, `RP`.

**5 — generate and QC, per disease.**

```python
!python stage2_cyclegan.py generate --disease CSC --n 4000 \
    --ckpt /content/drive/MyDrive/octfsl/ckpt/CSC
!python stage2_cyclegan.py qc --disease CSC --keep 3000
```

**6 — review the contact sheets, then bring the accepted images home.**

```python
from IPython.display import Image, display
display(Image('/content/project/data/gan/qc_CSC_kept.jpg'))
display(Image('/content/project/data/gan/qc_CSC_rejected.jpg'))
```

```python
!cd /content/project/data/gan && zip -qr /content/drive/MyDrive/octfsl/accepted.zip accepted
```

Download `accepted.zip` from Drive and unpack it so that the tree reads
`data/gan/accepted/<class>/*.png` in your local project. Stage 3 `--variant gan` reads
exactly that path.

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

`data_counts.csv` for the two variants that have been run
([`none`](results/stage3_none/data_counts.csv) ·
[`basic`](results/stage3_basic/data_counts.csv)) — common classes are identical in every
variant; only the rare rows change:

| Class | `none` train / val | `basic` train / val | test |
|---|---:|---:|---:|
| NORMAL | 23,536 / 2,779 | 23,536 / 2,779 | 250 |
| CNV | 32,536 / 4,669 | 32,536 / 4,669 | 250 |
| DME | 10,153 / 1,195 | 10,153 / 1,195 | 250 |
| DRUSEN | 7,733 / 883 | 7,733 / 883 | 250 |
| CSC | 27 / 0 | 1,800 / 200 | 5 |
| MH | 26 / 0 | 1,800 / 200 | 5 |
| MacTel | 25 / 0 | 1,800 / 200 | 4 |
| Stargardt | 19 / 0 | 1,800 / 200 | 4 |
| RP | 27 / 0 | 1,800 / 200 | 4 |

The `none` variant has **no rare validation images** — there are too few real ones to spare.

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

> `--cam` defaults to `--cam-variant gan`, which does not exist in this repo yet. Pass
> `--cam-variant basic` or `--cam-variant none` until Stage 2 has been trained.

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

## Differences from the paper

1. **Stage 2 has not been trained.** The `gan` variant — the paper's actual proposal — is
   pending. Everything reported above is the `none` / `basic` / baseline subset.

2. **RCI is not comparable and should be ignored in the comparison.** This repo computes
   relative classifier information as `I(true; pred) / H(true)` — normalised mutual
   information against the label entropy. The paper's numbers cannot come from that
   formula. Compare directly: the paper reports its prototypical network at **80.8%
   accuracy with RCI 0.933**, whereas under this formula our **90.31%**-accuracy `none`
   model scores only **0.788**. A less accurate model cannot carry more normalised mutual
   information about the same labels, so the paper is using a different normalisation.
   **Use accuracy, κ and MCC as the primary comparison**; our RCI column is internally
   consistent across our own models and nothing more.

3. **`proto` is not a prototypical network.** The paper trains a prototypical network with
   an episodic few-shot objective. Ours is a **nearest-class-mean classifier on frozen
   ImageNet Inception-v3 features** — no training at all. `knn` likewise runs on the same
   frozen features rather than a learned embedding. Both are labelled accordingly in the
   tables; neither is a like-for-like reproduction of the paper's row. **The paper's Siamese
   network baseline is not implemented at all.**

4. **QC is automated, not clinical.** The paper has an ophthalmologist review every
   synthetic image. `stage2_cyclegan.py qc` substitutes a feature-distance proxy: a
   synthetic image is kept only if its Inception-v3 features are closer (mean cosine
   distance to 3 nearest neighbours) to *real* training images of that disease than to real
   NORMAL images, keeping the closest `--keep` survivors. Contact sheets are written for
   **manual review, which is still required** — this proxy detects "the translation
   happened", not "the pathology is clinically correct".

5. **The rare dataset release is larger than the paper's.** After the seeded test hold-out
   this repo has **27/26/25/19/27** training images for CSC/MH/MacTel/Stargardt/RP. The
   published version of the Mendeley dataset contains more images than the paper describes,
   so the few-shot setting here is slightly less extreme.

6. **Uniform resize path.** Every image — real, basic-augmented and GAN — goes through
   256×256 before 299×299, because every rare training image was produced at 256×256 in
   Stages 1–2. This keeps image sharpness from becoming a class cue. The paper does not
   specify this.

7. **The classifier head is trained on cached frozen features**, not end to end. With a
   frozen backbone and no on-the-fly augmentation this is mathematically the same model, but
   it does mean the backbone sees each image exactly once.

8. **Early stopping fires far short of the paper's 250 epochs.** With `--patience 20`,
   `none` ran 26 epochs (best val loss at epoch 5) and `basic` ran 27 (best at epoch 6). The
   paper trains the full 250. Pass `--patience 0` to match the paper.

9. **Seeds are fixed at 42** throughout (splits, augmentation, CycleGAN prep sampling,
   classifier init, GAN subsampling). Bootstrap CIs use seed 0. The paper does not report
   seeds, so run-to-run variance is not characterised here.

10. **Patient overlap in the Kermany release.** Some patient IDs appear in both the official
    train and test splits. Stage 0 reports this but leaves the published split untouched.

---

## Repository structure

Only code and small result files are tracked. Data, caches, checkpoints and third-party
material are git-ignored — see [`.gitignore`](.gitignore).

```
.
├── stage0_manifest.py          build/verify data/manifest.csv
├── stage1_basic_aug.py         basic augmentation -> data/aug_basic/
├── stage2_cyclegan.py          prep | train | generate | qc
├── stage3_classifier.py        Inception-v3, variants none|basic|gan
├── stage4_evaluate.py          metrics, knn/proto baselines, CAM
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   └── manifest.csv            tracked: 84,630 rows, defines every split
└── results/
    ├── stage3_none/            data_counts.csv, history.csv
    ├── stage3_basic/           data_counts.csv, history.csv
    └── stage4/                 metrics.csv, recall.csv, auc.csv,
                                confusion_{none,basic,knn,proto}.csv

not tracked (regenerate locally):
    data/raw/  data/aug_basic/  data/gan/   ~8.7 GB of images
    results/features/*.npz                  cached backbone features
    results/stage3_*/{head.npz,best.weights.h5,*_predictions.csv}
    results/stage4/cam_*/                   CAM panels
    paper/  code2017/  ChestXRay2017/       third-party material
```

A fresh clone therefore needs Stages 0–3 rerun before Stage 4 will produce anything:
`test_predictions.csv` is not tracked.

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
