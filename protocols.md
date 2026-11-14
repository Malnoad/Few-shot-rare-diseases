# Our study protocol, in plain words

**This is the one page to read and to keep current.** It states the study as it stands after
every amendment. The binding texts are
[`docs/research/FM_PROTOCOL.md`](docs/research/FM_PROTOCOL.md) (v1.0, frozen 5 November 2026,
SHA256 `4e8df8d9…c5df`) and its dated amendments in
[`docs/research/FM_AMENDMENTS.md`](docs/research/FM_AMENDMENTS.md) (A1, A2, A3). If this page and
those two files ever disagree, **they win** and this page gets fixed.

Status on 13 November 2026: the proposal paper is submitted, the first Colab checks passed
(11 November), the rules are final (A3) and the two Colab notebooks are built. **No
foundation-model (FM) result exists yet.** Next step: Lam's word to push, then the Colab session.

## 1. The question

How well do frozen, off-the-shelf image encoders tell five rare retinal diseases apart from
OCT scans when we only have a few labelled examples per disease? And do they wrongly call
normal scans from other hospitals "rare disease"?

- **Five-way task:** CSC, MH, MacTel, Stargardt, RP.
- **Companion audit:** a nine-way classifier (4 common Kermany classes + the 5 rare ones) that
  measures the *false-rare rate* (FRR): how often a truly NORMAL scan is labelled as a rare disease.
- This is a **descriptive comparison** of checkpoints. We do not claim why one model wins, or
  anything clinical.

**Where it sits.** Yoo et al. (2021) trained a nine-way classifier on every rare image plus
generated ones and tested it on 22 rare images. MIRAGE (2025) compared the same kinds of encoder
we use, but with classifiers trained on each full dataset. Nobody in the studies we read compares
them with 1 to 16 labelled images per disease, measures how often normal scans from another
source are called rare, or checks whether image size alone gives the answer. Those three things
are what this study adds. The full table is at the end of amendment A3.

## 2. What we compare

| Model | Role |
|---|---|
| RETFound (Nature OCT) | FM candidate, trained on retinal images |
| MIRAGE-Base (OCT only) | FM candidate, trained on retinal images |
| DINOv2-Base | FM candidate, trained on everyday images |
| Inception-v3 (Keras, ImageNet) | Reference only, never the winner: the model of Yoo et al. and of our replication |

All encoders stay frozen: no fine-tuning, no adapters, no augmentation, no GAN images. There is
no fallback model (A3): a model that fails its pilot twice is reported as unavailable.

## 3. Data

- `data/manifest.csv` never changes: 84,630 images (74,082 train, 9,526 validation, 1,022 test).
- Rare development images: 27/26/25/19/27 = **124** (CSC/MH/MacTel/Stargardt/RP).
- Legacy rare test images: 5/5/4/4/4 = **22**. These were inspected in earlier work, so the test is
  not "blind"; we say so in the paper.
- External images, used only as test queries (A1):

| Source | Images | Used for |
|---|---|---|
| OCTID NORMAL, OCTDL NORMAL | 206; 332 | False-rare rate |
| OCTID CSR (scored as CSC), OCTID MH | 102; 102 | Agreement with OCTID's published labels |
| OCTDL MH | 39 | Agreement with OCTDL's published labels |

External rare results measure agreement with those datasets' labels. They are exploratory and
are not clinically confirmed transfer.

## 4. Metrics and baselines

**Main metric: macro recall** on the five-way task. For each disease, the share of its test
images classified correctly; then the plain average of the five shares. Every disease counts
equally, however many images it has. Guessing gives 0.20. MIRAGE calls the same number
"balanced accuracy", and it is the average of the per-class rates in Yoo et al.'s Table 4.
We report it for K = 1, 2, 4, 8 and as their mean. No single K is the primary one (A2).

**Main comparison:** each model's macro recall minus Inception-v3's, on exactly the same
support images and test images.

**Other metrics.** Five-way: accuracy, per-class recall, confusion counts. Nine-way audit:
false-rare rate, NORMAL recall, per-class recall. OCTID test: two-class macro recall (guessing
gives 0.50). Cost: seconds per image.

**Baselines**, from weakest to strongest:

| Baseline | What it tells us |
|---|---|
| Chance (0.20, or 0.50 two-way) | Below this, the model is worse than guessing |
| Size-only reference | What image width and height alone achieve. A model at or below it has not shown that it reads the retina |
| Inception-v3 (ImageNet) | The previous study's model, given the same K images |
| DINOv2 | A general-purpose model, beside the two retina-specific ones |

Yoo et al.'s published accuracy is not a baseline here: their classifier saw every rare
training image plus generated ones, ours sees K per class. We do not report AUROC, kappa, MCC
or RCI; the reasons are in amendment A3.

**What goes where in the paper (A3).** Main text: the five-way development curves with the
size-only reference and the nomination; the false-rare rate by source; the legacy 22 with exact
counts; the OCTID two-way test last, as exploratory. Supplement only: the logistic probe, output
shares on external rare images, per-class nine-way intervals, full confusion counts. Everything
is released, favourable or not.

## 5. How one run works

1. **Screen for duplicates.** Fingerprint the 146 rare images: identical files, identical pixels,
   and near-duplicates (a 64-bit difference hash, distance ≤ 4). Linked images form one group, so
   they never sit on both sides of a split. No pair was flagged among the 146.
2. **Make folds.** Five fixed folds per class from a fixed random stream (seed 42). Every class
   needs at least 10 groups, or the whole selection step stops.
3. **Draw supports.** For K = 1, 2, 4, 8 labelled images per class, draw 100 paired support
   sets. Every model sees exactly the same images.
4. **Time the models and lock.** Each model is timed on 100 images and must give the same output
   twice (one retry allowed). Then rosters, episodes, models, code and these rules are hashed into
   the **input lock**, before any score.
5. **Extract features** for the study images and the external images. External images that look
   like a locked image are removed automatically; nobody reviews pairs (A3).
6. **Classify.** Primary: cosine prototypes (normalise each support, average, normalise again,
   pick the closest class). Secondary: a fixed logistic probe on the first 25 draws.
7. **Nominate one FM** from development folds only. Score = per-class recall pooled over folds,
   averaged over classes, then over K and draws. Every FM within 0.02 of the best counts as tied;
   the fastest of those wins, by the timings in the lock. At least two FMs must finish, or there
   is no winner.
8. **Report.** Only after the nominee is sealed: the 22 legacy test images (K = 1, 2, 4, 8, and 16
   if every class has 16 groups), the nine-way FRR for Kermany, OCTID and OCTDL **separately**
   (Kermany's headline uses the 169 test images whose patients are not in the fitting set), and
   the external rare images.

**Spread:** we report mean, SD and 5th–95th percentiles over support draws. These describe
variability for *these* images, not confidence intervals for new patients. Only the nine-way
rates carry 95% intervals. No p-values.

Steps 1 to 5 run in `colab_fm.ipynb` (one GPU session, no labels read). Steps 6 to 8 run in
`colab_fm_evaluate.ipynb` (CPU). The runner follows
[`docs/research/FM_FIRST_RUN_GUIDE.md`](docs/research/FM_FIRST_RUN_GUIDE.md).

## 6. Rules everyone follows

- **No outcome-driven changes.** Any change to the method needs a dated amendment *before* the
  lock. Never re-pick models, folds or settings after seeing results.
- **Lock before scoring.** The code refuses to score without the lock and refuses to report
  before the nominee is sealed.
- **Failures are reported, never hidden.** A model that cannot load is "unavailable", not "bad".
- **Data is read-only.** Never edit `data/manifest.csv` or anything in `data/raw`, and never run
  `stage0_manifest.py --force`.
- **Nothing private in the repo.** No model weights, raw images, or Hugging Face tokens. Tokens live
  only in Colab Secrets. The repository stays public until extraction has finished.
- **Code reaches Colab only through GitHub.** Work on a branch and open a pull request; never commit
  to `main` directly.

## 7. What only Lam does

| When | What |
|---|---|
| Before the Colab session | Say "push". This accepts the text of A3 and records that Lam has read this page, limits included; it stands in for the protocol walkthrough and the risk acknowledgement that were still open |
| After `colab_fm.ipynb` | Check and approve the input-lock SHA256 in writing |
| Before any score | Deposit `lock-hashes-<attempt>.txt` on OSF or Zenodo, or email it to the mentor and team first |
| In `colab_fm_evaluate.ipynb`, Step 6 | Read the nomination, then allow reporting |
| After extraction | Tighten the `octfsl` Drive sharing; make the repository private if wanted |
| Any time before the lock | Change a rule, by a dated edit of `FM_AMENDMENTS.md` |

## 8. Timeline (Bangkok time)

| Stage | Dates | Output | Status on 14 Nov |
|---|---|---|---|
| S1 | 5–11 Nov | Proposal paper | Done |
| S2 | 12–14 Nov | Final pilots, input lock, extraction (`colab_fm.ipynb`) | Done 13 Nov: attempt `final-20261113T170804Z`, run by Lam on an L4, code `2f48654`; input-lock SHA256 `239838279b77bb37211398a016c1c766273a35f5a06649b31f5cd97f562c001c`; four arms available, every repeat check exact on the first sequence; OCTID 410 and OCTDL 371 external queries, none excluded. Checked on the laptop 14 Nov. Waiting for Lam's approval and the public timestamp |
| S3 | 15–19 Nov | Development runs; **nominee sealed by 19 Nov** | Not started |
| S4 | 20–23 Nov | Legacy and external reporting | Not started |
| S5 | 24–27 Nov | Summaries and plots; everything frozen 27 Nov | Not started |
| S6 | 28 Nov–3 Dec | Final write-up; evidence audit 1 Dec, PDF freeze 2 Dec, delivery 3 Dec | Not started |

If S2 slips, S3 starts only after a successful lock; the 27 November freeze still holds.

## 9. The files we control

Keep this list short. New material goes into one of these files, not into a new one.

| File | What it is | Who changes it |
|---|---|---|
| `protocols.md` (this page) | The study in plain words, status and timeline | Kept current after every decision |
| `docs/research/FM_PROTOCOL.md` | Binding protocol v1.0 | Nobody, ever |
| `docs/research/FM_AMENDMENTS.md` | Binding amendments A1 to A3 | Lam's decisions only; frozen at the lock |
| `docs/research/FM_FIRST_RUN_GUIDE.md` | How to run the two notebooks | When a notebook step changes |
| `docs/research/CONTEXT.md` | One row per decision, with its date and reason | One new row per decision |
| `fm_core.py`, `fm_prepare.py`, `fm_extract.py`, `fm_evaluate.py`, `fm_external.py` | The rules as code; hashed into the lock | Only before the lock, with tests |
| `fm_notebook.py` | Builds `colab_fm.ipynb` and `colab_fm_evaluate.ipynb`; never edit the notebooks by hand | When a notebook step changes |

**Reference, opened only when needed:** `docs/research/FM_RUNBOOK.md` (every command and guard in
detail).

**Closed records.** These explain how a decision was reached. They are kept as evidence and are
no longer updated; nothing in them overrides the files above: `FM_RULE_TRACKER.md`,
`FM_TERMS_EVIDENCE.md`, `FM_EXTERNAL_MAPPING.md`, `FM_LABEL_RULE_PROPOSAL.md`,
`fm_independent_review/`, `FM_RISK_ACKNOWLEDGEMENT.md`, `FM_CHECKLIST.md`, `FM_CODE_PLAN.md`,
`FM_CODE_REVIEW.md`, `FM_FINAL_REBUTTAL.md`, `FM_FEWSHOT_PLAN.md`, `FM_3WEEK_SPRINT.md` and the
literature digest `LITERATURE_KB.md` (all in `docs/research/`).

**Where things live outside the repository:** code on GitHub (branch `t3/fm-code-prep`, PR #24);
lock, features and results in Drive `octfsl/fm_v1/final/<attempt>/`; the lock's hashes on OSF or
Zenodo; the proposal paper in `docs/report/fm_ieee/main.pdf`.

## 10. The sources we track

The study rests on these ten sources. A claim in the paper cites one of them or our own files;
any other paper needs a reason before it is added here.

| Source | What we use it for |
|---|---|
| Yoo, Choi, Kim 2021, MBEC 59:401 (PMC7829497) | The rare-disease images, the 22 test images, the previous study and its Inception-v3 model |
| Kermany et al. 2018, Cell 172:1122 | The four common classes and their official split |
| Zhou et al. 2023, Nature (RETFound; PMC10550819) | A model we compare; it was pretrained partly on Kermany images, which we disclose |
| Morano et al. 2025, npj Digital Medicine (MIRAGE; PMC12462498) | A model we compare; its comparison set, balanced accuracy and input scaling are our precedent |
| Oquab et al. 2024, TMLR (DINOv2) | A model we compare |
| Wang et al. 2019, arXiv:1911.04623 (SimpleShot) | Precedent for nearest-prototype classification at several K |
| Gholami et al. 2018, arXiv:1812.07056 (OCTID; Borealis DOIs in A1's evidence) | External images |
| Kulyabin et al. 2024, Scientific Data (OCTDL; PMC11009408, Zenodo 10839556) | External images, with patient IDs |
| TVST 2026 external validation on OCTID (doi:10.1167/tvst.15.1.6, PMC12786394) | Precedent for scoring against OCTID's published labels |
| Chen et al. 2026, npj Digital Medicine (EyeDiff; doi:10.1038/s41746-026-02560-2) | Same precedent: OCTID's CSR class treated as CSC |

## 11. Open items for the write-up

None of these affects the run. They are settled when the final paper is written.

- Replace the proposal's sentence that rare transfer needs "clinical confirmation" with A1's
  required wording.
- Fill the CLAIM 2024 reporting checklist if the venue asks for one.
- Add the AI-use disclosure the venue requires.
- Archive the exact locked code publicly (for example on Zenodo) for the code-availability
  statement, especially if the repository is private by then.
- Choose the venue once real results exist (deferred by Lam on 13 November).
