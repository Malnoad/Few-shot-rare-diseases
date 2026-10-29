# Colab training guide

A run sheet for the training that is left in the replication: which notebook cells to
run, in what order, on which runtime, and what to bring back. Background (how Drive,
GitHub and the runtime fit together, what is in Drive, troubleshooting) is in
[`COLAB.md`](COLAB.md). Why each run exists is in
[`docs/research/REPLICATION.md`](docs/research/REPLICATION.md).

Open the notebook from GitHub so it always matches the code:
`https://colab.research.google.com/github/Malnoad/Few-shot-rare-diseases/blob/main/colab_setup.ipynb`

## What the runs are for

| Run | Answers | Paper item | Runtime | Ready? |
|---|---|---|---|---|
| **1. C5** GAN top-up | Brings MH, MacTel, Stargardt and RP to 3,000 accepted GAN images in a separate pool (`accepted_v2`), for C2 | §2.3, Fig. 5 | GPU | yes |
| **2. C0** features and gate | Does the rare-image file format/framing act as a "which dataset" cue? | Table 4 | GPU | yes |
| **3. C0** heads | Does removing that cue change rare-class recall? | Table 4 | CPU | no: stopped at gate, D20=A |
| **4. C1** seed sweep | Does GAN beat basic augmentation under the paper's 250-epoch schedule, over enough seeds? | Tables 3, 4 | CPU | yes: current inputs (D20=A), fast (D17=B) |
| **5. C2 / C3** | Amount of GAN images; class balance | Figs. 5, 11 | CPU | **not yet**: cells come with task T9 |

Runs 1 and 2 don't depend on each other or on any open decision. Start there.

## Before the first run

- [ ] **Drive space.** The project uses about 14.6 GB. C5 needs about 3 GiB more and C0
  about 2 GB more (caches, inventories, runs). A free account has 15 GB in total, so free
  space first: [`COLAB.md`](COLAB.md), *Cleaning up Drive*, lists about 5 GB that is safe
  to delete. Keep `ckpt/` until C5 is done. Empty the Drive trash afterwards; deleted
  files keep counting until then.
- [ ] **Runtime types.** GPU means **A100, L4 or T4**. Never "RTX PRO 6000" (Blackwell):
  TensorFlow 2.20 has no kernels for it. Part 1 checks this.
- [ ] **`GITHUB_TOKEN`** in Colab's Secrets panel, only if you will run Part 6.

## Rules for every session

1. Run **Part 1** first, every session.
2. Restore real images with Part 3 cells (3.1, 3.2, 3.3, 3.4b as each run lists). Do
   **not** run Part 2 in a session for runs 1-3: Part 2 makes empty placeholder images
   that C0 and C5 reject.
3. Never run `stage0_manifest.py --force`. The committed manifest defines the splits.
4. After a disconnect, start a new runtime and rerun **the same cells in the same order**.
   Every run below is resumable: finished work in Drive is checked and skipped, and only
   the interrupted step is redone.
5. Keep one sweep process running at a time (C0.4, Part 4).
6. Don't pick inputs or settings because of accuracy you have already seen. The arms,
   seeds and stopping rules are fixed in the design files.

## Run 1: C5 GAN top-up (GPU)

**Cells:** Part 1 → 3.1 → 3.3 → 3.4b → 3.4c.1 → 3.4c.2 (MH) → 3.4c.3 (MacTel) →
3.4c.4 (Stargardt) → 3.4c.5 (RP) → 3.4c.6.

Skip 3.2, the old 3.4 training cell, 3.5, 3.6 and Part 4. There is no GAN retraining:
3.4c uses the existing checkpoints in `MyDrive/octfsl/ckpt/`.

| Class | New candidates | A100 / T4 estimate |
|---|---:|---|
| MH | 718 | 2-3 / 6-12 min |
| MacTel | 6,739 | 7-9 / 21-36 min |
| Stargardt | 1,749 | 3-4 / 9-16 min |
| RP | 12,508 | 13-16 / 39-64 min |

Add 20-60+ minutes of I/O. All times are estimates (T4 is unbenchmarked).

**Check:** 3.4c.6 prints the count per class. All five must be **3000**. If a class is
short, raise its `C5_EXTRA` in 3.4c.1 and rerun that class's cell, then 3.4c.6. Don't
change the QC rule.

**Saved to:** `MyDrive/octfsl/c5/v2/` (candidates, QC records, `accepted_v2/`).

**Bring back:** the counts printed by 3.4c.6. Optionally run Part 6 to commit the QC
records in `results/stage2/c5_v2/`.

C1 is not affected: it keeps the original pool (`--gan-pool v1`, the default).

## Run 2: C0 features and gate (GPU)

**Cells:** Part 1 → 3.1 → 3.2 → 3.4b → C0.1 → C0.2 → C0.3.

Skip 3.3, the old 3.4 training cell, 3.5, 3.6 and Part 4.

- **C0.1** checks that real images (not placeholders) are present and prints free local
  and Drive space. Make sure there is room before going on.
- **C0.2** writes grayscale and grayscale+padded copies (local only; rebuilt after a
  disconnect).
- **C0.3** extracts fresh Inception-v3 features for all three arms (about 104,500
  images each) and computes the gate. Estimate: 15-60 min per arm on GPU, 2-8 h per arm
  on CPU (unbenchmarked). Each finished feature group is saved to Drive as it completes.

**Saved to:** `MyDrive/octfsl/c0/v1/`. The gate files are `full_axis.csv` and
`gate.json`.

**Bring back:** `gate.json` and `full_axis.csv`.

**If the gate stops:** if the framing arm still separates rare from common images (AUC
≥ 0.90), C0.4 refuses to start. That is the design, not an error. Bring the two files
back and decide before any heads run; don't improvise a crop.

### Optional: runs 1 and 2 in one GPU session

C0 reads the original pool (`data/gan/accepted/`) and C5 writes a separate one
(`accepted_v2/`), so they shouldn't interfere. Order: Part 1 → 3.1 → 3.2 → 3.3 → 3.4b →
3.4c.1-3.4c.6 → C0.1-C0.3. This combined order has not been tested; if anything looks
off, use two sessions.

## Run 3: C0 closed at the gate (D20=A, 29 Oct 2026)

Do not run C0.4 heads. Normalizing format/framing did not reduce separation;
disease and dataset source remain confounded. Record this as a limitation.
C1 and later C2/C3 use current inputs. C0's retained head code is not a planned run.

## Run 4: C1 seed sweep (CPU, long)

**D20=A (29 Oct 2026):** current inputs fixed. D17=B selects fast.

**Cells:** Part 1 → 4.1 → 4.2 → 4.3 → 4.4 → 4.5.

- **4.1:** keep `INPUT_ARM = "current"` (D20=A, 29 Oct 2026). Restore saved
  features and preflight; no normalized-input/C0 head run is planned.
- **4.2:** seeds 1-20, all three variants. Primary arm: 250 epochs, final-epoch head.
  Sensitivity arm: early stopping, same seeds as C0.
- **4.3:** recalculates the number of seeds needed once, from the SD of the first 20, and
  saves it. Target = max(63, calculated); if more than 243, stop and report.
- **4.4:** the extra seeds, 21 to the saved target (primary arm only). Refuses to start
  until all 120 heads of 4.2 are done.
- **4.5:** summary; safe on partial results. No conclusion until the schedule is
  complete.

D17=B uses `--trainer fast`, with the same Keras Dense/loss/Adam steps.
[Laptop benchmark](docs/research/T12_FAST_TRAINER.md): **4.69x** paired compute
speed-up; **9.35-12.87 min/250 epochs**, projected from three 25-epoch gan fits.
C0 heads are cancelled by D20=A (historical hypothetical compute budget
3.1-7.7 h, 38.6 h ceiling). C1 first stage **13.9-15.4 h**, provisional
63 pairs **41.6-43.1 h**, including that assumed sensitivity cost; ceilings
**25.7/53.4 h** if sensitivity reaches 250. These are laptop-measured compute
projections, excluding checkpoint/Drive I/O; **Colab is unbenchmarked**.
Native-shuffle accuracy differed materially; see the report, not an equivalence claim.

**Saved to:** `MyDrive/octfsl/c1/v1/<INPUT_ARM>/` (`runs.csv`, `summary.csv`,
`paired_differences.csv`).

**Bring back:** those three files when 4.2 finishes, and again at the end.

Don't run Part 6 for C0 or C1: their results live in Drive.

## Run 5: C2 and C3

Not ready. Cells come with task T9, after decisions D17 and D18. C2 will use the
`accepted_v2` pool from run 1.

## Where everything goes

| Run | Drive folder | Files to bring back |
|---|---|---|
| C5 | `c5/v2/` | counts printed by 3.4c.6 |
| C0 gate | `c0/v1/` | `gate.json`, `full_axis.csv` |
| C0 heads | cancelled (D20=A) | no head results |
| C1 | `c1/v1/<INPUT_ARM>/` | `runs.csv`, `summary.csv`, `paired_differences.csv` |

The full designs and rules: [`C0_DESIGN.md`](docs/research/C0_DESIGN.md),
[`C1_EXPERIMENT_PLAN.md`](docs/research/C1_EXPERIMENT_PLAN.md), and the C5 section of
[`COLAB.md`](COLAB.md).
