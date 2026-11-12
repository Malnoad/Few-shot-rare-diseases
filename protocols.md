# Our study protocol, in plain words

A short guide for the team. The binding version is
[`docs/research/FM_PROTOCOL.md`](docs/research/FM_PROTOCOL.md) (v1.0, frozen 5 November 2026,
SHA256 `4e8df8d9…c5df`). If this page and that file ever disagree, **the protocol wins**.

Status on 10 November 2026: the proposal paper is final and the code is ready for its first
Colab checks. **No foundation-model (FM) result exists yet.**

## 1. The question

How well do frozen, off-the-shelf image encoders tell five rare retinal diseases apart from
OCT scans when we only have a few labelled examples per disease? And do they wrongly call
normal scans from other hospitals "rare disease"?

- **Five-way task:** CSC, MH, MacTel, Stargardt, RP.
- **Companion audit:** a nine-way classifier (4 common Kermany classes + the 5 rare ones) that
  measures the *false-rare rate* (FRR): how often a truly NORMAL scan is labelled as a rare disease.
- This is a **descriptive comparison** of checkpoints. We do not claim why one model wins, or
  anything clinical.

## 2. What we compare

| Model | Role |
|---|---|
| RETFound (Nature OCT) | FM candidate |
| DINOv2-Base | FM candidate |
| MIRAGE-Base (OCT only) | FM candidate |
| Inception-v3 (Keras, ImageNet) | Reference only, never the winner |

BioMedCLIP replaces RETFound **only** if RETFound fails on access, loading or terms before any
labelled score. All encoders stay frozen: no fine-tuning, no adapters, no augmentation, no GAN images.

## 3. Data

- `data/manifest.csv` never changes: 84,630 images (74,082 train, 9,526 validation, 1,022 test).
- Rare development images: 27/26/25/19/27 = **124** (CSC/MH/MacTel/Stargardt/RP).
- Legacy rare test images: 5/5/4/4/4 = **22**. These were inspected in earlier work, so the test is
  not "blind"; we say so in the paper.
- External sources: OCTID and OCTDL NORMAL scans for the false-rare rate. Their MH/CSR images are
  used only if a clinician confirms the label mapping (for example CSR → CSC).

## 4. How one run works

1. **Screen for duplicates.** Fingerprint the 146 rare images: identical files, identical pixels,
   and near-duplicates (a 64-bit difference hash, distance ≤ 4). A person reviews flagged pairs.
   Linked images form one group, so they never sit on both sides of a split.
2. **Make folds.** Five fixed folds per class from a fixed random stream (seed 42). Every class
   needs at least 10 groups, or the whole selection step stops.
3. **Draw supports.** For K = 1, 2, 4, 8 labelled images per class, draw 100 paired support
   sets. Every model sees exactly the same images.
4. **Classify.** Primary: cosine prototypes (normalise each support, average, normalise again,
   pick the closest class). Secondary: a fixed logistic probe, run on the first 25 draws (K = 4
   only during development, every K when reporting). It is the first thing we drop if time runs out.
5. **Nominate one FM** from development folds only. Score = per-class recall pooled over folds,
   averaged over classes, then over K and draws.
   Every FM within 0.02 of the best counts as tied; the fastest of those wins (median time per
   image on the same GPU). At least two FMs must finish, or there is no winner.
6. **Report.** Only after the nominee is locked: the 22 legacy test images (K = 1, 2, 4, 8, and 16
   if every class has 16 groups), plus the nine-way FRR for Kermany, OCTID and OCTDL **separately**.

**Spread:** we report mean, SD and 5th–95th percentiles over support draws. These describe
variability for *these* images, not confidence intervals for new patients. No p-values.

## 5. Rules everyone follows

- **No outcome-driven changes.** Any change to the method needs a dated amendment *before* scores.
  Never re-pick models, folds or settings after seeing results.
- **Lock before scoring.** Rosters, episodes, models and code are locked and hashed before the first
  labelled score. The code refuses to report before the nominee is sealed.
- **Failures are reported, never hidden.** A model that cannot load is "unavailable", not "bad".
- **Data is read-only.** Never edit `data/manifest.csv` or anything in `data/raw`, and never run
  `stage0_manifest.py --force`.
- **Nothing private in the repo.** No model weights, raw images, or Hugging Face tokens. Tokens live
  only in Colab Secrets. The repository is public.
- **Code reaches Colab only through GitHub.** Work on a branch and open a pull request; never commit
  to `main` directly.

## 6. Timeline (Bangkok time)

| Stage | Dates | Output |
|---|---|---|
| S1 | 5–11 Nov | Proposal paper. PDF freeze 10 Nov 18:00; **submit by 11 Nov 22:00** |
| S2 | 12–14 Nov | First model loads, timing pilots, duplicate screen, input lock |
| S3 | 15–19 Nov | Development runs; **nominee locked 19 Nov** |
| S4 | 20–23 Nov | Legacy and external reporting |
| S5 | 24–27 Nov | Summaries and plots; everything frozen 27 Nov |
| S6 | 28 Nov–3 Dec | Final write-up; evidence audit 1 Dec, PDF freeze 2 Dec, delivery 3 Dec |

If S2 slips, S3 starts only after a successful lock; the 27 November freeze still holds.

## 7. Where to find things

| What | File |
|---|---|
| Full protocol (binding) | `docs/research/FM_PROTOCOL.md` |
| Proposal paper | `docs/report/fm_ieee/main.pdf` |
| Progress checklist | `docs/research/FM_CHECKLIST.md` |
| How to run each step | `docs/research/FM_RUNBOOK.md` |
| Known risks and open decisions | `docs/research/FM_RISK_ACKNOWLEDGEMENT.md`, `docs/research/FM_TERMS_EVIDENCE.md` |
| Colab notebook | `colab_fm.ipynb` |
| Running the first Colab checks (for a teammate) | `docs/research/FM_FIRST_RUN_GUIDE.md` |
| Code | `fm_core.py` (rules), `fm_prepare.py` (data), `fm_extract.py` (models), `fm_evaluate.py` (scores) |
