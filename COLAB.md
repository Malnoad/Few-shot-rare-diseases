# Running this project on Google Colab

For the remaining replication runs (C5, C0, C1) in order, with what to bring back, see
[`guide.md`](guide.md). This file is the background reference.

Everything runs in Colab. Nothing needs to be installed on your PC except for one optional
job: preparing the raw data the first time (see *One-time preparation*).

## How the pieces fit

| Place | Holds | Lifetime |
|---|---|---|
| **GitHub** (this repo) | code, `data/manifest.csv`, small results (heads, test predictions, metrics, QC records), this guide, `colab_setup.ipynb` | permanent |
| **Google Drive** `MyDrive/octfsl/` | everything large: raw data, augmented and generated images, the Inception-v3 features cache, CycleGAN checkpoints | permanent |
| **Colab runtime** `/content/p` | a working copy rebuilt from the two above every session | wiped on disconnect |

The notebook [`colab_setup.ipynb`](colab_setup.ipynb) does the rebuilding. Any output you
want to keep must be copied back to Drive (results) or pushed to GitHub (code, small
results). Part 6 of the notebook does both.

## Requirements

**Accounts**
- A Google account with Drive and Colab. The Drive folder must be in the same account Colab
  mounts.
- A GitHub fine-grained token for pushing (*Contents: read and write*, this repository
  only), stored in Colab's **Secrets** panel as `GITHUB_TOKEN`. Not needed for reading: the
  repo is public.

**Drive space.** The project folder uses about 14.6 GB (October 2026). A free Google account
has 15 GB shared with Gmail and Photos. A full rebuild from raw data adds about 5.5 GB.
Free space first: see *Cleaning up Drive* below. Files you delete stay in Drive's trash and
keep counting until the trash is emptied.

**Runtime type** (Runtime → Change runtime type)

| Work | Runtime |
|---|---|
| Restore, Stage 3 with the features cache, Stage 4, seed check, CAM | **CPU** |
| Stage 2 train / generate / QC | **GPU: A100, L4 or T4** |
| Stage 3 from nothing (features for ~84,600 images) | GPU recommended, CPU takes hours |

Do **not** use the "RTX PRO 6000" (Blackwell) GPU: TensorFlow 2.20 has no kernels for it
and fails with `CUDA_ERROR_INVALID_PTX`. The notebook detects it and forces CPU where it can.

**Software.** Colab already provides Python, NumPy, pandas, scikit-learn, OpenCV and
TensorFlow (observed in October 2026: Python 3.13, TensorFlow 2.20.0). The notebook checks
versions and only installs what is missing or too old, in practice at most
`albumentations>=2.0`. It never reinstalls TensorFlow. [`requirements.txt`](requirements.txt)
is for local installs.

## What must be in Drive (`MyDrive/octfsl/`)

| Path | Size | Used by | Status |
|---|---:|---|---|
| `results_gan_run/results/` | 0.46 GB | Part 2 restore: seed-42 results and the full features cache | present |
| `results_latest/` | ~0.46 GB | replaces the row above once Part 6 has run | present (22 Oct 2026, owned by a second account: copy it into this one) |
| `seed_check.csv` | <1 KB | Part 2 | present |
| `stage2.log`, `work/gan/qc_*` (15 files) | ~4 MB | Stage 2 provenance, copied into `results/stage2/` | present |
| `stage2_cyclegan.py` | 19 KB | old copy with `--margin`; GitHub has it too now, so the notebook no longer uses it | present, can be deleted |
| `test.zip` | 49 MB | Part 5 CAM: OCT2017 test images | present |
| `raw/rare.zip` | small | Part 5 CAM (22 rare test images), Part 3 | present (uploaded 21 Oct 2026) |
| `raw/oct2017.zip` | ~5.5 GB | Part 3 only (full rebuild, Stage 2 `prep`, Stage 3 without cache) | present (uploaded 21 Oct 2026); not yet verified with Stage 0 on Colab |
| `aug_basic.zip` | 0.52 GB | Part 3.2, restores Stage 1 | present |
| `gan.zip` | 0.74 GB | Part 3.3, restores Stage 2 `prep` | present |
| `ckpt/<class>/` | 6.9 GB | Part 3.4, regenerating or topping up GAN images | present |
| `accepted_<class>.zip` ×5 | 0.66 GB | Part 3.4b, restores the accepted GAN images | present |
| `c5/v2/` | budget ~3 GiB extra | Part 3.4c: resumable candidates, QC/provenance, separate combined `accepted_v2/` | created only when Lam runs C5 |

Not used by the notebook: `out_all.zip`, `features.zip`, `stage3_none.zip`,
`stage3_basic.zip`, `results_gan/`, the top-level `features`, `stage3_*` and `stage4` folders
inside `results_gan_run/`, and the image folders `work/gan/{out,accepted,normalA,normal_src,real}`
(copies of the zips above).

## One-time preparation

**1. Get the notebook into Colab.** Once it is pushed, open
`https://colab.research.google.com/github/Malnoad/Few-shot-rare-diseases/blob/main/colab_setup.ipynb`.
Until then, use File → Upload notebook in Colab.

**2. Raw data (only for Part 3 and CAM).** On the PC that has the cleaned raw data (README,
*Cleanup pitfalls*: no `__MACOSX`, folders named exactly `MacTel` and `Stargardt`), make two
zips with Windows' built-in `tar`, which writes forward-slash paths:

```powershell
cd "C:\Users\Admin\.vscode\Few-shot rare diseases\data\raw"
tar.exe -a -c -f rare.zip rare
tar.exe -a -c -f oct2017.zip OCT2017     # ~5.5 GB, only if you will rebuild from raw
```

Upload them to `MyDrive/octfsl/raw/`. The notebook finds the `rare/` and `OCT2017/` folders
inside the zips wherever they sit, so an extra parent folder or backslash paths
(PowerShell's `Compress-Archive`) also work.

**3. GitHub token.** Create it at GitHub → Settings → Developer settings → Fine-grained
tokens, then add it in Colab's Secrets panel as `GITHUB_TOKEN` with notebook access on.

## Each session

1. Pick the runtime type (table above).
2. Run **Part 1** (setup, about 1–2 minutes).
3. Then run **either** Part 2 (normal work: restore results, about 2 minutes) **or** Part 3
   (rebuild stages from raw). Not both in one session: Part 2 creates empty placeholder
   images that only Stage 3 understands. Part 3 deletes them before it starts.
4. Optional: Part 4 (C1/D15; use the separate order below), Part 5 (CAM).
   For **C0** use Part 1, then 3.1, 3.2 and 3.4b (real images; Part 2's placeholders are
   not enough), then C0.1-C0.5 on an A100/L4/T4 for the fresh features. Outputs live only
   in `MyDrive/octfsl/c0/v1/`; never reuse `results/features`. If C0.3 reports a framing
   AUC >= 0.90, the notebook stops before C0.4: bring `gate.json` back for a decision.
5. Run **Part 6** to save to Drive and push to GitHub.

### Rough timings

| Step | Time |
|---|---|
| Part 1 + Part 2 | a few minutes |
| 3.1 extract raw data + Stage 0 verify | several minutes (5.5 GB from Drive) |
| 3.2 Stage 1, if not restored | 10–20 min CPU |
| 3.4 Stage 2 training | ~1.1 min/epoch in the logged run (an A100 at least in its last session), so ~2 h per disease for 100 epochs; T4 is several times slower |
| 3.5 Stage 3 with features cache | minutes per variant |
| 3.5 Stage 3 from nothing | hours on CPU |
| Part 4 C1 first 20 / provisional 63 pairs | ~37-42 h / ~112-116 h serial CPU; laptop extrapolation, es20 assumed 20-50 epochs |
| Stage 4 / CAM | 1–2 min / a few minutes |

The Stage 2 figure comes from `stage2.log`. The extraction and seed-check times are
estimates.

### C5: top up before C2; keep C1's pool fixed (D7)

On **A100 or T4**, run **Part 1 → 3.1 → 3.3 → 3.4b → 3.4c.1–3.4c.6**.
Skip Part 2 (placeholders), 3.2, the old 3.4 training cell, 3.5, 3.6 and Part 4.
Use the original `gan.zip` for QC references and existing `ckpt/<class>/` on Drive;
there is **no retraining**. Part 4 is maintained separately by T8.

The observed pass rates come from `results/stage2/qc_*.csv`, cross-checked against
`results/analysis/qc_summary.csv`. The candidate budget includes a 20% surplus:
`ceil((3000 - kept) * 1.20 / (passed / 4000))`.

| Class | Deficit | Arithmetic | Candidates | Cell | A100 / T4 compute estimate, min |
|---|---:|---|---:|---|---|
| MH | 390 | ceil(390 × 1.20 / .6525) | 718 | 3.4c.2 | 2–3 / 6–12 |
| MacTel | 1,752 | ceil(1752 × 1.20 / .312) | 6,739 | 3.4c.3 | 7–9 / 21–36 |
| Stargardt | 801 | ceil(801 × 1.20 / .54975) | 1,749 | 3.4c.4 | 3–4 / 9–16 |
| RP | 2,168 | ceil(2168 × 1.20 / .208) | 12,508 | 3.4c.5 | 13–16 / 39–64 |

CSC already has 3,000 and receives no new candidates. `c5_topup.py plan` recomputes
the table without TensorFlow. New NORMAL sources come only from manifest TRAIN,
excluding the union of **all** original QC source stems and all `normalA` stems.
The run records **zero round-1/domain-A overlap** in `plan.json` and the source
paths/hashes in `sources.json`; it stops rather than recycling sources. Different
diseases may use the same new NORMALs. Preparation uses the same OpenCV 256px resize.

Timing evidence: `results/stage2/stage2.log` records A100 generation of 4,000
images in MH 116 s, MacTel 115 s, Stargardt 126 s and RP 133 s. Generation-only
extrapolations are about 21 / 194 / 55 / 416 s for these budgets. MH's QC start
(01:29:30) to the next process (01:30:57) bounds that QC run at about 86 s;
QC was not timed directly. The table allows ~90 s QC per 4,000 plus ~1 min startup.
**T4 inference is unbenchmarked**: 3–4× is an extrapolation from the logged
T4 ~3.1 vs A100 ~1.0 min training epochs, not a measured inference ratio.
Raw extraction, checkpoint/reference hashing, Drive PNG writes and pool copies
add **20–60+ min total I/O (unbenchmarked)**. Check actual free space; allow roughly
3 GiB extra on Drive and space locally for raw data and the installed v2 pool.

QC uses the **original** rule: ImageNet Inception-v3, L2-normalized features,
mean cosine distance to 3 nearest neighbours, all restored real disease TRAIN
references and the first 500 sorted restored `normalA` images. Pass iff
`d_rare < d_normal` (threshold zero), rank passing images by ascending `d_rare`,
break ties by filename. This explicitly selects `--ranking d_rare`, because the
later standalone `qc` CLI defaults to ranking by margin. The combined pool keeps
every old accepted image, then the best new passing images up to 3,000 per class.
L4 expert ophthalmologist ratings were dropped; C5 uses automatic QC only.

Candidates are saved atomically directly in `MyDrive/octfsl/c5/v2/<class>/out/`.
QC records live in `<class>/qc/qc_<class>.csv`, with input hashes and source
inventories alongside them. Original `results/stage2/qc_*.csv` and v1 PNGs are
read-only. Restart with the same cells: completed PNGs/classes are skipped,
interrupted QC repeats. Changed checkpoints/references/v1 inputs stop resume.
If a class falls short, increase its `C5_EXTRA` in 3.4c.1 and rerun its cell;
only a deterministic prefix of additional fresh sources is appended. Keep the
threshold fixed. The candidate margin is a planning allowance, not a guarantee.

**3.4c.6** runs on CPU, requires all four top-ups complete, copies CSC's unchanged
v1 images, and installs `data/gan/accepted_v2/` with a completion inventory. It
also restores new QC/provenance to `results/stage2/c5_v2/`. Those records can be
saved via Part 6; the large images remain in Drive `c5/v2/accepted_v2/`.
Completed v2 pools are immutable. Rerun .6 after a runtime reset to reinstall them.

**C1 is unchanged:** `--gan-pool v1` is Stage 3's default and reads the original
`data/gan/accepted/` with the original `gan_<class>.npz` caches. C1 deliberately
keeps the four short pools as a documented deviation. **C2 must explicitly pass
`--gan-pool v2 --gan-per-class N`**, with separate C2 output/cache directories.
Stage 3 requires v2's full inventory and uses new `gan_v2_<class>.npz` groups;
the existing exact path-list comparison also rejects stale caches when N changes.
Its v2 default output is `results/stage3_gan_v2/`, protecting the run of record.
Switch back with `--gan-pool v1`; no deletion or mutation of either pool is needed.
## C1 / D15: manual seed sweep (wait for C0 input decision)

Use a **CPU runtime** and run **Part 1 -> 4.1 -> 4.2 -> 4.3 -> 4.4 -> 4.5**.
Leave `INPUT_ARM = "current"` in 4.1 only if Lam retains current after C0; otherwise
set `format` or `framing`. This is the single C1 input selector. Do not select
inputs from C1 accuracy. Current restoration needs only the saved features;
4.1 copies them and creates isolated placeholders, without Part 2 or Stage 4.
For normalized inputs restore real images with 3.1/3.2/3.4b, then C0.1/C0.2 to
restore completed C0 caches/metadata and recreate normalized inputs. C1 reads
`data/c0/v1/<arm>` and `results/c0/v1/<arm>/features`; it never extracts features.

- **4.1:** choose input arm after C0, preflight caches/counts/identity, open output.
- **4.2:** first 20 seeds, three variants under fixed250 primary and es20 sensitivity:
  120 heads. Fixed250 explicitly uses `--patience 0 --restore last`; es20 uses
  patience 20/best weights and the same seeds 1-20 as C0.
- **4.3:** once-only SD-based power recalculation. N is at least 63 pairs, including
  first 20, for two-sided alpha .05/power .80 at a true 1 pp difference versus zero.
  Save the target before top-up. If N exceeds the 243-pair feasibility cap, stop
  incomplete. No target changes from means/CIs/p-values or repeated recalculation.
- **4.4:** fixed250 seeds 21..N only; refuses to run until all first-20 heads finish.
- **4.5:** per-seed gan-basic differences per protocol, paired-t 95% CIs, Wilcoxon
  sensitivity and the D15 rule comparing the fixed250 CI to zero and 1 pp.
  A positive CI alone does not establish a 1 pp gain; no nonsignificance/equivalence shortcut.

Everything C1 produces is written directly to `MyDrive/octfsl/c1/v1/<INPUT_ARM>/`.
Resume after disconnect by rerunning Part 1 and 4.1-4.5; completed heads are
verified/skipped, interrupted heads restart. Run one sweep at a time. Skip 3.5,
3.6 and Part 6. No old seed-check backup/restore loop is used. Keep the existing
accepted GAN realization: MacTel 1,248 and RP 832; C5's separate pool is C2 only.

Measured local CPU head timing: **25 completed epochs in 207.58 s**, projected
**34.60 min/250 epochs**, the allowed short timing because 250 exceeded 30 min.
At that largest-variant rate, 20-seed primary = **34.6 h**; 63-pair primary =
**109.0 h**. Add **2.8-6.9 h** for 60 es20 heads if they take 20-50 epochs
(assumption). First stage total ~37-42 h; 63-pair total ~112-116 h. Worst cases
with all sensitivity heads reaching 250 are 69.2/143.6 h. Colab CPU/Drive speed is
unverified: use actual first-head timing for forecasts and resume across sessions.
GPU speedup for heads is unverified; CPU is sufficient, and paying for a GPU
solely for this tiny batch-10 Dense head needs a measured benefit first.
See the [fixed plan](docs/research/C1_EXPERIMENT_PLAN.md) and
[timing record](docs/research/C1_HEAD_TIMING.json). Patient-cluster CIs and
training-seed CIs describe different uncertainties; C1 does not resolve leakage.

## Rules that protect the results

- **Never run `stage0_manifest.py --force` on Colab.** The committed manifest defines every
  split. Rebuilding it samples the rare test images after sorting file names, and Linux
  sorts differently from Windows, so the hold-out could change without any error.
  `stage0_manifest.py` without `--force` only verifies.
- **Check the accepted-image counts before `--variant gan`.** If `data/gan/accepted/` is
  empty, Stage 3 only prints a warning and trains what is effectively `basic`.
- **C1 never writes canonical Stage 3/4 or seed_check.csv.** It uses explicit
  `--cache-dir`, `--out-dir` and `--cache-only`; completed runs are retained only
  after configuration and artifact hashes match. Partial heads restart, without
  replacing completed heads. Do not use Part 6 for C1.
- **Patient-level CIs use saved predictions, without rerunning Stage 4.** Run
  `python analysis.py cluster` (also included in `all`); the 5,000-draw percentile
  accuracy/MCC and paired CIs go to `results/analysis/`. Kermany images are clustered
  by patient across classes; the 22 rare images are assumed independent singletons.
  This adjusts test-sampling variance, leaves train/test leakage unresolved and excludes
  training-seed variability. `results/stage4/` keeps the legacy image-level CIs.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA_ERROR_INVALID_PTX`, compute capability 12.0 | Blackwell GPU | CPU runtime, or A100/L4/T4 for Stage 2 |
| Drive folder not found / empty | Colab mounted a different Google account | Remount, choose the account that owns `octfsl` |
| New Drive files not visible | stale mount | `drive.mount("/content/drive", force_remount=True)` |
| `cache ... is stale -> re-extracting` in Part 2 | placeholder names differ from the cache | stop; the cache and accepted images are out of sync |
| `MISMATCH` or missing paths from Stage 0 | raw zip layout or folder names | check `data/raw/OCT2017/{train,test}` and the exact class folder names |
| Stage 1/2 decode errors on 0-byte files | Part 2 placeholders still present | restart the runtime, or run 3.1, which removes them |
| `WARNING <class>: only N accepted GAN images` | fewer than 3,000 passed QC (MH, MacTel, Stargardt, RP) | expected; documented in the README |
| Drive storage full | checkpoints, duplicates, trash | clean up (below), then empty the trash |
| Disconnected mid-run | Colab session limit | rerun Part 1, then the same cell; Stage 2 resumes from the Drive checkpoint |

## Cleaning up Drive

Safe to delete (the notebook never reads them), about 5 GB:
`out_all.zip`, `features.zip`, `stage3_none.zip`, `stage3_basic.zip`, `results_gan/`, the
top-level `features`, `stage3_basic`, `stage4`, `stage3_gan` folders inside
`results_gan_run/` (keep `results_gan_run/results/` until `results_latest/` exists), and the
image folders `work/gan/{out,accepted,normalA,normal_src,real}` (keep the `qc_*` files in
`work/gan/`). This removes the raw generated images (`out_all.zip`, `work/gan/out`), which
are only needed to rerun QC; Part 3.4 regenerates them from the checkpoints.

`ckpt/` (6.9 GB) is needed for C5 top-ups of MH, MacTel, Stargardt and RP. Keep it
until C5 is complete and its v2 pool/provenance are verified.
