# Our study protocol, in plain words

**This is the one page to read and to keep current.** It states the study as it stands after
every amendment. The binding texts are
[`docs/research/FM_PROTOCOL.md`](docs/research/FM_PROTOCOL.md) (v1.0, frozen 5 November 2026,
SHA256 `4e8df8d9…c5df`) and its dated amendments in
[`docs/research/FM_AMENDMENTS.md`](docs/research/FM_AMENDMENTS.md) (A1, A2, A3). If this page and
those two files ever disagree, **they win** and this page gets fixed.

Status on 19 November 2026: the input lock was made on 13 November and its hashes deposited on
Zenodo (doi:10.5281/zenodo.24206472) before any score. **All pre-specified scores were computed
on 14 November**; the nominee is MIRAGE. The results are summarised in
`docs/report/2026-11-14_weekly_report.tex`. P0–P9 and G1 are done; P10 and the write-up remain,
on the calendar of 19 November (section 8): last analysis 20 Nov, freeze 25 Nov, **delivery
30 Nov**. Fine-tuning is not part of this study.

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
we use, but with classifiers trained on each full dataset. Sun et al. (2024) already drew
five-way K-shot tasks (K = 1, 3, 5) from the same rare images, with a trained network. What this
study adds (narrowed 19 Nov; A3's table is kept as written): a **pre-registered,
shortcut-audited evaluation of frozen foundation models** on this public rare-OCT benchmark.
Frozen encoders at 1 to 16 images per disease on identical supports, how often normal scans
from other sources are called rare, and what image size and source alone explain. Lam's
positioning, to be confirmed by the mentor at the 23 Nov review.

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
  not "blind"; we say so in the paper. They are a seed-42 random draw (`stage0_manifest.py`) that
  matches Yoo et al.'s test counts only; which 22 images Yoo et al. used is not known, so the paper
  says "a hold-out matching their class counts", never "their test set".
- Provenance (found 18 Nov, file identity only): Mendeley btv6yrdbmv version 1 (27 Mar 2020)
  holds 119 images, the paper's total; version 2 (29 Oct 2020), which we use, holds 146: the 118
  version 1 files kept unchanged plus 28 files new to version 2 (27 new file names and a different
  picture under the old name MacTel/20.PNG; count corrected 19 Nov). 21 of the 22 hold-out
  images are in version 1; RP/20.PNG is an addition. Mapping:
  `results/fm_v1/posthoc/pre_rule/2026-11-18/v1_to_ours.csv`. The paper text gives Stargardt 15+4
  and RP 12+4, the version 1 folders RP 19 and Stargardt 16 (as does Sun et al. 2024; version 2
  holds RP 31 and Stargardt 23): per-class names for
  these two carry a footnote; macro recall is unaffected.
- Layout: the locked geometry summary (`reports/geometry/MIRAGE/geometry_summary.csv` in the
  results archive; A2 border rule: flat edge rows, grey range ≤ 8) gives a median border share per
  rare class of 0.34–0.50 among development images and 0.22–0.56 among hold-out images, and 0 (at
  most 0.002) per class for Kermany, OCTID and OCTDL (corrected 19 Nov; first written 0.34–0.56). Counted per image on 18 Nov (pre
  rule, `rare_layout.csv` in the same folder): 126 of the 146 rare images have a dark flat border
  covering at least 25% of the image, mostly bands above and below the scan. Image geometry,
  no model involved.
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

| Stage | Dates | Output | Status on 15 Nov |
|---|---|---|---|
| S1 | 5–11 Nov | Proposal paper | Done |
| S2 | 12–14 Nov | Final pilots, input lock, extraction (`colab_fm.ipynb`) | Done 13 Nov: attempt `final-20261113T170804Z`, run by Lam on an L4, code `2f48654`; input-lock SHA256 `239838279b77bb37211398a016c1c766273a35f5a06649b31f5cd97f562c001c`; four arms available, every repeat check exact on the first sequence; OCTID 410 and OCTDL 371 external queries, none excluded. Checked on the laptop 14 Nov. Hashes deposited on Zenodo (doi:10.5281/zenodo.24206472, created 13 Nov 19:53 UTC); Lam's approval recorded 14 Nov 09:11 UTC, before the first score |
| S3 | 15–19 Nov | Development runs; **nominee sealed by 19 Nov** | Done 14 Nov: nominee **MIRAGE** (SHA256 of the file `nomination_binding.json`: `1738f7dec61526651511221a6d6723cd37e76e8accd0f05e09bba45569b1d82e`; the `binding_sha256` written inside it: `00891271d84e44dfd807ccd540742d046b27483337b50e83cab4dbd8a18406f7`). Evaluate notebook needed the locked Python 3.13.16 (commit `56fcb52`) |
| S4 | 20–23 Nov | Legacy and external reporting | Done 14 Nov: `evaluation/results-20261114T093033Z.zip` on Drive (SHA256 `c963b1d7298c28f8151769d8d29cf98538ce0897582f57ff2ee867c76351b25f`). Summary in the 14 Nov weekly report |
| S5 | 15–25 Nov (was 27) | Figures, tables, agreed post-hoc analyses (labelled); everything frozen 25 Nov | Started 15 Nov: first report written; P0–P9 and G1 done 18 Nov; plan below |
| S6 | 25–30 Nov (was 28 Nov–3 Dec) | Final write-up; evidence audit 27 Nov, PDF freeze 28 Nov, delivery 30 Nov | Not started |

Lam brought delivery forward to 30 November on 19 November (no GPU run is left in this study;
the work is writing and the mentor's review). The calendar below is the binding one.

**Pull-forward rule (Lam, 19 Nov).** Dates in the calendar are the latest dates. When a stage
finishes early, the next one starts at once. **Work runs in parallel (Lam, 19 Nov):** Lam does the
slow steps (Colab runs, blind reviews) at once; the agents verify, analyse and draft at the same
time; the mentor gives feedback whenever it comes and is not a gate. The gates that stay: a rule
is committed and pushed before its numbers; no new analysis rule is added after 20 Nov; mentor
feedback that arrives after the freeze changes wording only, never a number; v2 work starts as
soon as v1's numbers are frozen, and its results stay out of the v1 paper except the labelled
next-phase paragraph.

### Plan for S5 and S6 (version 2, 16 November; revised after the mentor's answers)

Version 1 (15 Nov) followed three checks: recall per disease, a two-model ensemble, and a review of
possible add-ons by three other models. Version 2 is an Opus 5.5 draft, rebutted by Fable 5.2 and
checked against `FM_AMENDMENTS.md`. Nothing below changes a locked number, the nominee or A3's
placement table. **This version may change once more after the mentor answers Q1–Q8;** each
change gets a dated line under "Changes" at the end of this subsection.

**Remaining gaps, and what we do about each.**

| Gap | Addressed by | What stays open |
|---|---|---|
| The 22 hold-out images are our own seed-42 sample, and no image has a patient ID | P1; hold-out wording (W3) | Independence cannot be tested with these images |
| Image size alone scores 0.57 (development) and 0.62 (hold-out) | P4, P3 | Every model resizes its input, so size marks where an image came from; reshaping cannot remove that |
| The comparison with Yoo et al. is between different kinds of classifier | P2, P6 | Which 22 images they tested on |
| Recipes: RETFound has no published frozen result; MIRAGE stretches each image's contrast | A GPU check on the 389 rare images, only if the mentor asks (Q2, Q4) | How the models behave when fine-tuned |
| External labels were not checked by a clinician | P3 (do the two macular-hole labels agree?) | Needs a clinician (Q5) |
| MIRAGE calls only 26 of 332 OCTDL normal scans normal | P5 | Probably stays a stated limitation |
| Spread over draws is not a confidence interval; zero counts have no bound | P1 | Patient-level uncertainty |
| MIRAGE is not the best model for macular hole | P0, P7 | No ensemble gain on the hold-out |
| Pretraining overlap with our test sources (W1, checked 16 Nov in the papers' data lists) | RETFound: Kermany yes (109,309 images, the whole release), OCTID and OCTDL no. MIRAGE: none (in-house Vienna data, Cirrus and Spectralis; includes unlabelled CSC, MacTel, MH and Stargardt scans). Inception-v3: ImageNet. DINOv2: web-scale, cannot be excluded | Whether our 169 Kermany test normals are among RETFound's 109,309 (likely, unchecked); ImageNet and LVD-142M content |
| The post hoc scripts are outside the repository and untested | W2 | — |

**Rules for every post hoc analysis.**

1. *Rules before numbers.* Before P1 runs, Lam either commits and pushes this subsection or puts
   the SHA256 of `protocols.md` in the email to the mentor. P0 and P7 were computed on 15 November,
   before this step, so they are labelled "rule fixed after first look".
2. *Placement.* A post hoc result never takes the place of a result A3 puts in the main text. All
   post hoc results sit under one dated "post hoc" heading. Where A3 says "supplement only"
   (logistic probe, full confusion counts), the post hoc version goes there too. Lam decides any
   other placement, with the mentor.
3. *Everything is released,* favourable or not (A1 rule 6), including analyses that fail a gate.
4. *Outside the protocol, stated as such.* P2 scores classifiers trained with augmented and GAN
   images, which v1 excludes; P3 uses OCTDL macular holes as examples, although A1 makes them
   query-only. Both are reported as analyses outside the locked study.

**Post hoc analyses.** Laptop only, from the locked features and results. "Always" items are
computed whatever the mentor says; "if time" items follow the drop order below.

| # | Analysis | Rule (fixed before computing, except P0 and P7) | Question | Tier |
|---|---|---|---|---|
| P0 | Recall per disease | Development, prototype, mean over K = 1–8 and draws, all four models. Computed 15 Nov: MIRAGE CSC/MH/MacTel/Stargardt/RP 0.71/0.54/0.68/0.59/0.73; MH: DINOv2 0.61, Inception-v3 0.62 | — | Always |
| P1 | Exact bounds and leave-one-image-out | Clopper–Pearson 95% bounds for every nine-way count; OCTDL counts flagged as clustered (332 images from 110 patients). MIRAGE minus Inception-v3, mean macro recall over K = 1–8, recomputed with each of the 22 hold-out images left out, and with each of the 124 development images left out of the queries; report the smallest and largest value of each | Q7, Q1 | Always |
| P2 | October's trained classifiers on OCTID and OCTDL | The three heads `results/stage3_{none,basic,gan}/head.npz`, unchanged. Gate 1: they reproduce 17, 18 and 19 of the 22 hold-out images on the locked Inception-v3 features. Gate 2: the external lock records the same Inception-v3 recipe as the locked arm (`fm_extract.py` uses the `stage3_classifier.py` loader). Check the class order in each head first. A failed gate is reported, not fixed | Q3 | If time |
| P3 | Clinic-to-clinic macular-hole test | Examples from OCTID MH and NORMAL, queries from OCTDL MH and NORMAL, then the reverse. K = 1, 2, 4, 8; 100 draws, seed 42; cosine prototypes; two-way macro recall (chance 0.50); all four models. OCTDL examples: at most one image per patient per class. OCTDL queries reported per image and per patient (MH 39 images, 28 patients). Called "agreement between two macular-hole labels", never transfer | Q1, Q5 | Always (raised 16 Nov) |
| P4 | Size-right against size-wrong | Development queries split by whether the size-only rule names them correctly in the same episode; recall per model in each group. A class with no query in a group in an episode is left out of that episode's average; report how many episodes each class enters. A description, not proof | Q2 | If time |
| P5 | Normal-scan audit | Always: for every normal scan in the nine-way audit, the class it receives and the gap between its top two scores, per model and source. If time: a "not sure" cut-off per model, set at the gap below which 5% of the 169 patient-disjoint Kermany normals fall (about 8 scans, so coarse), applied unchanged to OCTID and OCTDL. The Kermany row is calibration; only OCTID and OCTDL rows are results. Normal recall is reported beside the false-rare rate, never in its place | Q6 | Always / if time |
| P6 | Prototypes against the logistic probe | A table from the locked run; no new computation. Supplement, as A3 fixes for the logistic probe | Q3 | Always |
| P7 | MIRAGE + DINOv2 ensemble | Per model and query, z-score the class cosines, add with equal weight, take the largest (rule fixed 15 Nov, before computing, but chosen after P0 showed the MH weakness). Computed 15 Nov: development 0.695 against 0.650; hold-out 0.718 against 0.722. Three-model version beside it. One supplement line, never called a result: the protocol forbids a classifier added to improve a model | — | Always |
| P8 | CSC against MH on development images | Locked development episodes, K = 1, 2, 4, 8, 100 draws, all four models; CSC and MH queries only; the two-way prediction is the larger of the CSC and MH cosines, ties to CSC (the rule of A2 A-1). Two-class macro recall per K and mean; MIRAGE minus Inception-v3 paired over draws; the size-only rule restricted the same way beside it. Purpose: the fair comparator for the OCTID two-way test, where MIRAGE leads Inception-v3 by only +0.03 | Q2 | Always (added 16 Nov) |
| P9 | Rare against normal, with no cut-off | Locked nine-way prototype scores, all four models. Score of a scan = its largest cosine to the five rare classes minus its largest cosine to the four common classes. AUROC (Mann–Whitney, ties count one half) of rare against normal scans in three sets: OCTID CSC and MH (each, and together) against OCTID NORMAL, same clinic, image units (no patient IDs); OCTDL MH against OCTDL NORMAL, same clinic, patient units; the legacy 22 against the 169 patient-disjoint Kermany normals, labelled "different sources: source and disease are mixed". 95% interval: 2,000 bootstrap resamples, seed 42 with stream namespace 9101, units resampled within rare and within normal; MIRAGE minus each other model paired on the same resamples. Beside each AUROC: the locked argmax point (share of rare scans called any rare, share of normals called rare, as in R1) and the rare share above the score that only 5% of that set's normals exceed (a threshold read from the test set itself, so an upper bound, never a result). If G1 runs, the same table on its padded MIRAGE and Inception-v3 features. Supplement | Q6 | Always (added 18 Nov) |
| P10 | Same-picture audit of the 146 rare images | Yoo et al. took the rare images from web searches (section 2.1 of their paper), so the risk is one published picture, or slices of one case report, on both sides of a split. Three steps, laptop only, raw data read-only. (1) Provenance: map each released file to the source links in Yoo's supplementary material where possible; record per file whether a link names an image, a figure, a case or only a publication; a shared publication alone never counts as a shared patient. (2) Screen: every pair of the 146 images, across labels, scored by crop-robust keypoint matching (OpenCV SIFT, Lowe ratio 0.75, RANSAC homography; score = inlier count) on greyscale images. The cut-off is the largest inlier count seen in 2,000 random rare–Kermany pairs (different sources, certain negatives); its recall is measured on synthetic positives (each rare image re-cropped to 70–90% of each side, rescaled by 0.5–1.5 and saved as JPEG quality 70; seed 42, stream namespace 10101) and reported. (3) Review: Lam sees, in random order and blind to label, split and score, every pair at or above the cut-off (if more than 60, the 60 highest, and the total is reported), the 30 highest-scoring pairs, every pair sharing a source link, 10 synthetic positives and 10 random low-score pairs as decoys; each pair is marked "same picture", "same case, other slice", "different" or "unsure". Sensitivity, beside the locked numbers and never replacing them: development macro recall (prototype, locked episodes, all four models, mean over K = 1–8) recomputed without each query that shares a "same picture" or "same case" mark with a support of its episode, and again counting "unsure" as a match; hold-out recall recomputed without each hold-out image so marked against any development image. Wording if nothing is found: "no same picture found by this screen", never "no shared patients" | Q1 | Always (added 18 Nov) |
| G1 | Pad-to-square input check | One GPU session, MIRAGE and Inception-v3 only (the two models that squash images to a square). Each image is padded with black (0) to a square around its centre, then given to the model's own locked transform; nothing else changes. Re-extract every image those two arms used (rare 146, Kermany fitting and test, OCTID, OCTDL); score with the locked episodes and the locked nine-way roster. Report development macro recall, hold-out counts, the size-only comparison and OCTDL normal destinations, beside the locked numbers. A labelled sensitivity check outside A2 A-11; it never replaces a locked result. New code only: no hashed `fm_*.py` file is edited | Q2, Q4, Q6 | Lam approved 16 Nov; runs after rule 1 |

| R2 | Development spread over draws | Section 5 promises mean, SD and 5th–95th percentiles over support draws; the locked evaluate notebook wrote development means only. From the locked development prototype cells of all four arms: per (K, draw), macro recall over the 124 queries with the five folds pooled (the nomination's unit); per K = 1, 2, 4, 8 and the K mean (A2 B-1): n, mean, SD, 5th and 95th percentile over the 100 draws (`fm_evaluate.draw_summary`); each FM minus Inception-v3 paired on the same draw, same summaries. Gate: every arm's per-K mean equals its `nomination.json` k_score within 1e-9, otherwise nothing is reported. Label: "pre-specified quantity (section 5), computed 19 Nov after the results were known". Table 1 of the main text | Q7 | Always (added 19 Nov) |

**Reporting change R1 (16 Nov).** Wherever a false-rare rate is shown, the share of external rare images the same nine-way head calls any rare class sits beside it (OCTID CSC and MH, OCTDL MH), labelled post hoc. A low false-rare rate alone can mean the head rarely says "rare".

**Housekeeping.** W1: read the data list in the MIRAGE paper and record any overlap with our test
sources. W2: one script, `fm_posthoc.py`, with tests, holds P0 to P7; it reads the results archive
and writes only to `results/fm_v1/posthoc/`. W3: replace "their test set" in `README.md`,
`CONTEXT.md` and the paper draft with the wording of section 3. W4: section 11 items (A1 wording,
AI-use statement, code archive) and the claim ledger for the 1 December audit. W5: Lam names the
file of the final draft. `docs/report/fm_ieee/main.tex` is the delivered S1 proposal; if it is
reused, tag the proposal commit first.

**Calendar.** The laptop analyses run whatever the mentor says; the answers decide placement and
any GPU check.

| Dates | Work | Gate |
|---|---|---|
| 16 Nov | Report and email to the mentor; rule 1 (commit or hash) | Before any P1–P6 number |
| 16–17 Nov | W2 with P0, P1, P6, P8; W1 done 16 Nov; G1 code and notebook prepared | — |
| 18 Nov | Mentor's answers due; P5 (always part), P4, P3; R1 | Lam decides placement |
| 18 Nov | G1 ran on Colab (07:02 UTC) and was scored on the laptop | — |
| 19 Nov | Records and push; P10 steps 1–2 (links, SIFT screen, blind review sheet) | Push before any P10 number |
| 20 Nov | P10 step 3 (Lam's blind review) and sensitivity; W3 | **No new analysis starts after 20 Nov** |
| 21–22 Nov | Methods and results draft; final figures and tables; weekly report from the draft | 21 Nov: spare day only |
| 23–24 Nov | Draft sent to the mentor for feedback (any time, not a gate); v2 protocol drafted meanwhile, laptop only | — |
| 25 Nov | Corrections; all numbers, tables and figures frozen | — |
| 25–26 Nov | Discussion, limitations, one paragraph on the next phase; W4 | — |
| 27–30 Nov | Evidence audit (27), PDF freeze (28), spare day (29), delivery (30) | — |

Calendar changed 19 Nov at Lam's request (was: analysis to 21 Nov, review 25–26, freeze 27,
audit 1 Dec, PDF 2 Dec, delivery 3 Dec).

**If time runs short,** drop in this order: P2, the P5 cut-off, P4, G1. P0, P1, P3, P5 (destinations),
P6, P7, P8 and R1 always stay. **If the mentor has not answered by 18 Nov,** Lam sends a reminder on 19 Nov;
the post hoc results then go to the supplement, and the main text keeps only the pre-specified
results.

**How the mentor's answers may change this plan.** Options for Lam, not automatic:

| Answer | Possible change |
|---|---|
| Q1: the benchmark does not support a "follow-up to Yoo et al." claim | Retitle as a frozen-encoder benchmark on the Yoo images; the Yoo comparison moves to the discussion |
| Q2 or Q4: add an input or recipe check | One GPU session on the 389 rare images, 19–21 Nov, with a dated post hoc note and Lam's go-ahead |
| Q3: "same backbone" only | Inception-v3 relabelled; P2 stays in the supplement |
| Q5: the external mappings are doubtful | Wording of the OCTID and OCTDL results changes; their placement does not (A3) |
| Q6: normal recall matters most | Normal recall shown beside the false-rare rate in the main text, labelled post hoc |
| Q7: add Yoo's metrics | Accuracy and kappa in the supplement table only |

**Not in this study** (Lam decides after 30 November; v2 can run on Colab from 1 December):

- *Letterbox removal* (decided 19 Nov): crop the flat bands from the rare images, re-extract
  MIRAGE and Inception-v3, rescore. The first v2 experiment, under its own rule.
- *Planted-cue test* (research thread, 18 Nov; Lam's choice pending): a domain's own artefact
  added to the K rare examples only, measured as false-rare calls on clean common images.

- *Fine-tuning the encoders.* Section 2 and A3 exclude it. Of the gaps above it answers only the
  open half of the recipe gap; it does not help with the hold-out, image size, external labels or
  OCTDL. It is the first experiment of the next phase, under its own protocol, in this order:
  (1) a test where image size does not predict the disease, with patient IDs if possible;
  (2) light adaptation (LoRA or the last blocks) of MIRAGE and RETFound inside the same K-shot
  episodes, settings fixed in advance; (3) full fine-tuning only if step 2 gains clearly on
  development and the gain holds on OCTID and OCTDL. The ensemble's +0.045 is a reference point,
  not a threshold: it is itself post hoc and gave nothing on the hold-out.
- Zero-shot classification from text, generated or augmented images as examples, feature maps.
- Dropped for good: feature centring (the protocol forbids it), a linear fit that removes size
  from the features, and a new trained nine-class classifier.

**Changes**

- 15 Nov: version 1.
- 16 Nov: version 2. Added P0, rules 1 to 4, tiers, the gates in P2, patient rules in P3, the
  empty-class rule in P4, per-model cut-off in P5, W4 and W5, the answer table. Not adopted from the
  Opus draft: moving external results or the false-rare rate after the answers (outcome-driven,
  section 6), dropping P3 to make room for a GPU check, moving the answer date to 19 Nov, and
  rewriting the proposal paper in place.
- 16 Nov, later: before rule 1 was met, the agents computed part of P6 while checking a list of
  few-shot methods: logistic minus prototype per K on the hold-out and development K = 4 (paired
  first 25 draws), and the same on external rare images (five-way output shares). P6 is therefore
  labelled "computed before the rules were timestamped", like P0 and P7. No other P1–P6 number
  exists.
- 16 Nov, evening: Lam asked for P8, P3, G1, R1, P5, P1, P4 and P2. Added P8, G1 and R1; P3
  raised to always; drop order changed. Rule 1 still applies before any P1–P8 or G1 number.
- 16 Nov, 17:56 (Bangkok): rule 1 met. Lam chose "commit and push"; this file (SHA256
  `2cd37c8b…6e08`) was committed alone as `1eb3964` and pushed to PR #24 before any P1–P8 or G1
  number.
- 16 Nov, after rule 1: W2 done. `fm_posthoc.py` (with `tests/test_fm_posthoc.py`) computed P0–P8 and
  R1 from the locked archives, each checked against its receipt SHA256, into
  `results/fm_v1/posthoc/` (one CSV per table, every row labelled post hoc with its date). P0 and P7
  reproduce their 15 Nov numbers. P2: both gates passed (17/18/19 of 22; same Inception recipe).
  Choices the rules left open, fixed in the code before computing: P1 bounds cover, per set and
  true class, the correct class, NORMAL, a common disease and any rare class; P3 draws use seed 42 with
  stream namespace 8101, class order NORMAL then MH; P3 per-patient recall weights each OCTDL
  patient equally; P4 episodes are the locked fold-level episodes; P5 cut-off is the 9th-smallest gap of
  the 169 (8 lie below it). G1 code and notebook (`fm_pad_extract.py`, `fm_pad_notebook.py`,
  `colab_fm_pad.ipynb`) prepared, not run; they reach Colab only after Lam says "push".
- 18 Nov: the mentor answered (17 Nov). "Rare" is now judged by labels available: about 30 or
  fewer labelled images; Lam proposed adding a share of the dataset (3–5%). OCTID CSC and MH (102
  each, 17.8% of 572) and OCTDL MH (39, 1.9% of 2,064) are therefore uncommon-class labels, used
  only as queries; their results stay (A1 rule 6) and are worded as agreement with uncommon-class
  labels. Lam added P9 (rule above) and said "commit" for G1; this commit timestamps P9 before any P9
  number. P3's placement and a hold-out leakage check are open (Opus draft, Fable rebuttal, then Lam).
- 18 Nov, later: Lam and the mentor confirmed the definition of "rare" for the project (B1). A class
  is rare when it is at most 5% of its dataset's labelled images (also say whether at most 3%), and
  few-shot when the pool its examples are drawn from holds at most 30 labelled images; both numbers
  are reported for every class. Tiers: A rare and few-shot (primary test classes); B rare, more than
  30 (examples from a fixed pool capped at 30, labelled "capped pool"); C few-shot, more than 5%
  (small datasets, never pooled with A); D common. Under it the five Yoo classes are tier A
  (development pools 19–27 images; class totals 23–32 are 0.55–0.77% of the 4,146 images the
  study used, our own subsample of Kermany plus the 146 rare images), OCTID CSC and MH are D and OCTDL
  MH is B. Problems B2–B13 of the 18 Nov ledger go to a four-model discussion (Opus 5.5, Fable 5.2,
  Astra 6.2, Sol 6.2) before Lam decides.
- 18 Nov, evening: G1 reviewed before its Colab run; no G1 number exists. No bug was found in the code
  the notebook checks out (`39c3d73`), so the notebook link pinned at `90fd4fe` stays valid. Laptop dry
  run on real rare, Kermany, OCTID and OCTDL images, each checked against its locked SHA256: Inception-v3
  through the locked loader on CPU (unpadded control within 3e-6 of the locked features, and a square
  image, which padding leaves unchanged, equally close), MIRAGE up to its locked input transform (no
  PyTorch on the laptop). On a stand-in G1 output equal to the locked features, the scoring reproduced
  the locked numbers and P9 exactly. Added, laptop only: `fm_posthoc.py g1` reads the Drive `g1/`
  archives (each checked against its receipt), adds square-image, repeat and identity checks, and
  writes `G1_comparison.md` (locked beside padded); more padding tests.
- 18 Nov, night: four-model discussion of B2–B13 (Opus 5.5, Fable 5.2, Astra 6.2, Sol 6.2; two
  rounds). Lam's decisions: audit and preparation first, Colab later (G1 still has to run by 21 Nov
  to enter the draft). Wording adopted for the draft: external sets are named by their B1 tier in
  their own dataset (OCTID CSC and MH: D; OCTDL MH: B, 1.9% by images, 28 of 821 patients = 3.4%),
  under "agreement with published labels", with A1 rule 7's sentence kept; "rare" here means label
  scarcity within a dataset, not prevalence. P3 goes to the supplement, both directions; the main
  text cites only the locked count (MIRAGE called 26 of 332 OCTDL normals NORMAL) and points to P3,
  with no claim about the cause; no padded P3. The draft states that Yoo's rare images came from web
  searches and were labelled with knowledge of their source pages, that the release (146 images) is
  not the paper's set (119), and that web availability gives no right to redistribute; W1 adds that
  web-trained models (DINOv2) may have seen these pictures, which cannot be checked. P9 is worded
  as a per-scan AUROC of this prototype score (for OCTDL the interval resamples patients; the
  CSV's "patient units" note refers to the interval only); the 5% cut-off is retrospective;
  MIRAGE's OCTDL value (0.357) says this score ranks OCTDL scans the wrong way, not what MIRAGE's
  features contain. The hold-out shows direction only. P10 added (rule above) for B5. Next phase,
  not v2: fine-tuning (LoRA), on data where image size does not predict the class. DINOv3: Lam
  requests gated access to ViT-B/16 now; v2 or next phase decided after 3 Dec.
- 18 Nov, night: numbers computed on rare images outside any committed rule are released, as A10
  was, in `results/fm_v1/posthoc/pre_rule/` with a README: the 15 Nov per-class and ensemble
  scripts (P0 and P7 first looks), the 16 Nov probe-versus-prototype and confusion checks (before
  rule 1), and the 18 Nov reviewer scripts (power, false-rare and K-mean estimates for the v2
  design; Kermany-only checks for P10). None is a result.
- 18 Nov, mentor: our protocol and the existing results are acceptable because they are
  transparent; on comparability with Yoo et al., "even the reviewer cannot identify clearly what
  it is because no one can compare" (nobody has their test split).
- 18 Nov, G1 ran: Lam ran `colab_fm_pad.ipynb` (pinned `90fd4fe`, G1 code `39c3d73`, both pushed)
  at 07:02 UTC, before the evening review entry above was written and earlier than the calendar's
  19–21 Nov. `fm_posthoc.py g1` scored it the same day; that scoring code (`36dad48`) was committed
  but not yet pushed. Checks: both archives match their receipts; unpadded controls and 1,032
  square images within 1.43e-6 of the locked features; repeats exact; GPU L4 → A100, MIRAGE
  environment differs only in `filelock` and `regex` patch versions. Outputs committed as generated
  (`d25e152`), then two sentences of the page corrected and the page regenerated (`79f51d8`; CSV
  hashes unchanged).
- 19 Nov, Lam's decisions on G1 (Fable's package, after Opus and Fable reviews and a Fable
  rebuttal): G1 goes to the supplement under the post hoc heading; the main text keeps the locked
  26 of 332 with no claim about its cause. Allowed: development and hold-out moved by at most 0.023
  at K = 1–8 (mean of 100 draws), a near-null test because the rare inputs are already near square;
  with black padding, external normals go almost entirely to rare classes for both models
  (MIRAGE OCTID 202 → 0 and OCTDL 26 → 1 called NORMAL); the rare images are letterboxed (section 3)
  and padding gives wide external scans that layout; external nine-way results depend on input
  geometry. Not allowed: that squashing did or did not cause the OCTDL result; robustness to aspect
  ratio; padded external rare "gains" as recognition; "rare is a not-Kermany bin"; "bars caused
  it"; any reading of K = 16; any MIRAGE-versus-Inception ranking from padded scores; anything
  about what the features contain. P9 caveat: the locked OCTID AUROCs hold "under the published
  input path; with black bars added, the same score ranks OCTID the other way"; the main-text
  false-rare rate holds for frame-filling inputs. G1b dropped; letterbox removal moves to v2. One
  optional main-text sentence pointing to G1, labelled post hoc: Lam and the mentor decide.
- 19 Nov, hold-out and comparability (Lam: "wording + records"): both locked sets stay; no new
  split and no new number. Adopted for the draft: the hold-out, comparability and limitation
  wording D(1)–D(3) of `pre_rule/2026-11-18/fable_holdout_plan.md`; Inception-v3 is "the backbone
  family of Yoo et al."; Yoo's 92.1 and 93.9 are context only; the RP/Stargardt footnote and the
  provenance paragraph (section 3). Sun et al. 2024 and Tai et al. 2025 added to section 10; the
  13 Nov positioning ("nobody … compares them with 1 to 16 labelled images", A3's table, which
  cannot be edited) is narrowed in section 1. Emailing Yoo et al. is optional, Lam's choice.
- 19 Nov, positioning: Lam chose "a pre-registered, shortcut-audited evaluation of frozen
  foundation models on a public rare-OCT benchmark", wording only, confirmed with the mentor at
  the 23 Nov review. The novelty of v2 is discussed in the research thread; Lam's choice is pending.
- 19 Nov, calendar: Lam brought delivery forward to 30 Nov (section 8 tables). P10 keeps its rule
  and runs 19–20 Nov after this push. Letterbox removal is not added to this study.
- 19 Nov, released in `results/fm_v1/posthoc/pre_rule/2026-11-18/` (no rule covered them): the
  Mendeley version mapping (`map_v1.py`, `v1_to_ours.csv`, `md_versions.json`), the geometry
  scripts and per-image layout (`geom.py`, `geom_a.py`, `geom_b.py`, `geom_c.py`,
  `rare_layout.csv`), the G1 reviews (`g1_review_opus.md`, `g1_review_fable.md`,
  `g1_verdict_opus_main.md`, `g1_rebuttal_fable.md`) and `fable_holdout_plan.md`. A figure of
  "128 of 146" letterboxed images quoted in the 18 Nov discussion could not be reproduced and is
  withdrawn; the rule-based count is 126.
- 19 Nov, after the draft review (Opus drafts, Fable rebuttals, Gemini 4 Carbon review, GPT 6.2
  Sol numbers check; drafts in Drive `fm_v1/draft/2026-11-19/`), Lam's decisions: (1) R2 added
  (rule above), computed only after this entry is pushed. (2) Placement: P4 and P8 go to the main
  text in one subsection headed as post hoc, beside and never replacing the locked size-only
  reference; P0, P1, P2 and P5 go to the supplement (P3, P6, P7, P9 and G1 were already there).
  (3) Hold-out table: the locked nine-way counts of 22 per class and arm, with five-way macro
  recall per K beside them. (4) The paper is a new file, `docs/report/fm_v1_paper/main.tex`, on the
  proposal's IEEE template; the proposal commit is tagged first. Agent defaults, open to Lam's
  objection: numbers are rounded half up from their shortest decimal form; the paper quotes the
  locked geometry summary for the letterbox layout, and the pre-rule count of 126 appears only in
  the supplement with its label; "resembles" (not "follows") for Sun et al.; "every rare training
  image" for Yoo et al.'s classifier; MIRAGE's pretraining on unlabelled CSC, MacTel, MH and
  Stargardt scans is stated in Methods and Limitations; "RETFound has no published frozen result"
  is dropped (A3 records frozen linear-probe results in the MIRAGE paper).

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

**Colab notebooks: what has run and what is left.** Code reaches Colab only through GitHub; each
notebook checks out the commit it is pinned to. Steps for a runner: `FM_FIRST_RUN_GUIDE.md`.

| Notebook | What it does | Needs | Status (18 Nov) | Next action |
|---|---|---|---|---|
| `colab_setup.ipynb` | October reproduction of Yoo et al. (stages 0–4) and experiments C0–C5 | GPU for stage 2; Drive | Finished: stage 4 is the run of record (`results/stage4/`); C2/C3 closed 6 Nov; C1 left open | None for the FM study |
| `colab_fm.ipynb` | Loads the four models, times them, locks the inputs, stages OCTID and OCTDL, saves all features | GPU, Hugging Face token, about 1–1.5 h | Done: attempt `final-20261113T170804Z` (13 Nov); input lock `23983827…`, external lock `3414889b…` | None; a rerun would make a new attempt |
| `colab_fm_evaluate.ipynb` | Development, nomination, legacy, nine-way and external reports from the saved features | No GPU, no token | Done: `results-20261114T093033Z.zip` (14 Nov); nominee MIRAGE | None |
| `colab_fm_pad.ipynb` | G1: MIRAGE and Inception-v3 re-extracted with every image padded to a square | GPU, no token, about 1 h (estimate) | Done 18 Nov 07:02 UTC (A100), pinned at `90fd4fe`; archives in Drive `g1/`, receipts match | None |

Laptop only, no Colab: P0–P9 and G1 scoring (done), P10 (19–20 Nov, after its rule is pushed).

**Reference, opened only when needed:** `docs/research/FM_RUNBOOK.md` (every command and guard in
detail).

**Closed records.** These explain how a decision was reached. They are kept as evidence and are
no longer updated; nothing in them overrides the files above: `FM_RULE_TRACKER.md`,
`FM_TERMS_EVIDENCE.md`, `FM_EXTERNAL_MAPPING.md`, `FM_LABEL_RULE_PROPOSAL.md`,
`fm_independent_review/`, `FM_RISK_ACKNOWLEDGEMENT.md`, `FM_CODE_PLAN.md`,
`FM_CODE_REVIEW.md`, `FM_FINAL_REBUTTAL.md`, `FM_FEWSHOT_PLAN.md`, `FM_3WEEK_SPRINT.md` and the
literature digest `LITERATURE_KB.md` (all in `docs/research/`).

**Where things live outside the repository:** code on GitHub (branch `t3/fm-code-prep`, PR #24);
lock, features and results in Drive `octfsl/fm_v1/final/<attempt>/`; the lock's hashes on OSF or
Zenodo; the proposal paper in `docs/report/fm_ieee/main.pdf`.

## 10. The sources we track

The study rests on these twelve sources (the last two added 19 Nov). A claim in the paper cites
one of them or our own files; any other paper needs a reason before it is added here.

| Source | What we use it for |
|---|---|
| Yoo, Choi, Kim 2021, MBEC 59:401 (PMC7829497) | The rare-disease images (and test counts 5/5/4/4/4), the previous study and its Inception-v3 model |
| Kermany et al. 2018, Cell 172:1122 | The four common classes and their official split |
| Zhou et al. 2023, Nature (RETFound; PMC10550819) | A model we compare; it was pretrained partly on Kermany images, which we disclose |
| Morano et al. 2025, npj Digital Medicine (MIRAGE; PMC12462498) | A model we compare; its comparison set, balanced accuracy and input scaling are our precedent |
| Oquab et al. 2024, TMLR (DINOv2) | A model we compare |
| Wang et al. 2019, arXiv:1911.04623 (SimpleShot) | Precedent for nearest-prototype classification at several K |
| Gholami et al. 2018, arXiv:1812.07056 (OCTID; Borealis DOIs in A1's evidence) | External images |
| Kulyabin et al. 2024, Scientific Data (OCTDL; PMC11009408, Zenodo 10839556) | External images, with patient IDs |
| TVST 2026 external validation on OCTID (doi:10.1167/tvst.15.1.6, PMC12786394) | Precedent for scoring against OCTID's published labels |
| Chen et al. 2026, npj Digital Medicine (EyeDiff; doi:10.1038/s41746-026-02560-2) | Same precedent: OCTID's CSR class treated as CSC |
| Sun, Wei, Wang, Zheng 2024, Medical Image Analysis 93:103102 | Prior K-shot work on Yoo's rare images (five-way, K = 1, 3, 5, trained network, three tasks); narrows our novelty claim; not a numeric comparator (other pool and training) |
| Tai et al. 2025, arXiv:2505.20149 | Reuses Yoo's data and compares with Yoo's 93.9 without defining its test set: the loose comparison we avoid |

## 11. Open items for the write-up

None of these affects the run. They are settled when the final paper is written.

- Replace the proposal's sentence that rare transfer needs "clinical confirmation" with A1's
  required wording.
- Fill the CLAIM 2024 reporting checklist if the venue asks for one.
- Add the AI-use disclosure the venue requires.
- Archive the exact locked code publicly (for example on Zenodo) for the code-availability
  statement, especially if the repository is private by then.
- Choose the venue once real results exist (deferred by Lam on 13 November).
