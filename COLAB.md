# Running this project on Google Colab

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
4. Optional: Part 4 (seed check), Part 5 (CAM).
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
| Part 4 seed check (15 heads) | minutes |
| Stage 4 / CAM | 1–2 min / a few minutes |

The Stage 2 figure comes from `stage2.log`. The extraction and seed-check times are
estimates.

## Rules that protect the results

- **Never run `stage0_manifest.py --force` on Colab.** The committed manifest defines every
  split. Rebuilding it samples the rare test images after sorting file names, and Linux
  sorts differently from Windows, so the hold-out could change without any error.
  `stage0_manifest.py` without `--force` only verifies.
- **Check the accepted-image counts before `--variant gan`.** If `data/gan/accepted/` is
  empty, Stage 3 only prints a warning and trains what is effectively `basic`.
- **The seed check restores the seed-42 outputs** when it finishes, so the published tables
  stay intact.
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

`ckpt/` (6.9 GB) is only needed to generate more GAN images. Delete it only if you will not
top up MacTel and RP or rerun generation.
