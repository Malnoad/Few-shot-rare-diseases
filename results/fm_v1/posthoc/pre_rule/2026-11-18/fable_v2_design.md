# Protocol v2: review of the Opus primary (B2) and the fine-tuning conditions (B7)

Fable 5.2, 18 November 2026. Read-only: no repo file edited, nothing committed, no Colab run.
Scratch scripts and outputs: `scratchpad\fable_v2\` (`power.py`, `kmean.py`, `frr.py`, `cache.pkl`).

Source tags
- `[P7]`, `[P6]`, `[R1]` = rows of `results/fm_v1/posthoc/P7_ensemble.csv`, `P6_probe_vs_prototype.csv`, `R1_rare_beside_false_rare.csv`.
- `[zip]` = `D:/Few-shot/drive/fm_v1/final/final-20261113T170804Z/evaluation/results-20261114T093033Z.zip` (per-episode `predictions.csv`), opened read-only.
- `[mine: script]` = computed by me today from `[zip]`; nothing here is a locked or post hoc table of the study.
- `[scratch 15 Nov]` = `scratchpad\ensemble_out.txt`, an earlier scratch run that is not in the repo.
- `[notes]` = `scratchpad\mentor_addons_fable.md`, `dinov3_fable.md`, `datasets_fable.md`.
- `[pdf: id]` = text of the paper in `scratchpad\pdfs\<arXiv id>.txt`, read today.
- `[unverified]` = my estimate or judgement, no source.

What I deliberately did not compute: any probe-level fusion, any DINOv3 number, any fine-tuned score.
Those are v2 outcomes; computing them before the rules are fixed would repeat the P7 problem.

---

## Part 1. B2: the primary question

### 1.1 Verdict

**Amend, do not replace.** Keep "pre-specified ensemble against the best single model" as the
one named primary contrast. Change six things: the status of the claim, the K, the comparator,
the fusion rule, the uncertainty, and the safety endpoint.

### 1.2 What the stored results say about the ensemble (all already seen)

| Where | MIRAGE+DINOv2 minus MIRAGE (prototype cosines) | Source |
|---|---|---|
| Development, K=8 | +0.076 (SD over draws 0.030) | [P7] row development / paired_difference / 8 |
| Development, mean K=1-8 | +0.045 (SD 0.022) | [P7] row mean_k1_2_4_8 |
| Hold-out, K=8; K=16; mean K=1-8 | -0.002; -0.015; -0.003 | [P7] reporting rows |
| OCTID CSC+MH, five-way share correct, mean K=1-8 | 0.412 against 0.511 = **-0.099** (K=8 -0.125) | [scratch 15 Nov]; the block is headed "OCTID two-way" but the per-class values are five-way shares; label [unverified] |
| OCTDL MH, five-way share correct, mean K=1-8 | 0.379 against 0.225 = +0.155 | [scratch 15 Nov] |
| Nine-way, normals called rare | OCTID 1/206 against 4/206; OCTDL 33/332 against 30/332 (34 against 34 in total) | [scratch 15 Nov]; reproduced today [mine: frr.py] |
| Nine-way, external rare called own class | OCTID CSC 27 against 23, MH 51 against 10 (of 102 each); OCTDL MH 16 against 3 (of 39) | [mine: frr.py] |

So for the MIRAGE+DINOv2 pair every cell Opus names (development, hold-out, both safety counts) has
already been looked at. One clear gain (development), one null (hold-out), one clear loss (OCTID
five-way), one gain (OCTDL MH), and a safety count that is equal.

### 1.3 Is development K=8 the right primary? No, not as proposed.

1. **K=8 is the best-looking cell.** The post hoc gain rises with K: +0.018, +0.032, +0.055, +0.076
   [P7]. Naming K=8 after seeing that is a choice made on the outcome. Amendment A2 fixed "no single
   K is the primary one" (`protocols.md:72`).
2. **The development set generated the hypothesis.** It chose the nominee, then P0 showed MIRAGE's
   macular-hole weakness, then P7 was designed to fix it (`protocols.md:220`, "rule fixed after first
   look"). A test on the same 124 images cannot confirm it.
3. **The non-replication is not just noise.** Development K=8: +0.076, image-level 95% interval
   +0.043 to +0.111. Hold-out K=8: -0.002, interval -0.049 to +0.037 [mine: power.py; stratified
   bootstrap over query images]. The intervals do not overlap; difference 0.078 with SE about 0.028.
   The hold-out interval is optimistic because it resamples 4-5 images per class.
   Why: on development DINOv2 beats MIRAGE on macular hole by 0.118; on the hold-out MIRAGE beats
   DINOv2 on macular hole by 0.216, so there is nothing to complement [mine: power.py, per-class rows].
4. **K=8 is also the least precise choice.** See 1.5: the K-mean has a 30% smaller standard error.

**Decision I recommend to Lam:** primary estimand = paired difference in macro recall, **mean over
K = 1, 2, 4, 8**, development folds. K=8 is the named key secondary and the only K for fine-tuning.
The claim is worded as "pre-specified re-analysis on re-used data", never "confirmed". The word
"confirmed" needs data whose ensemble result nobody has seen: the OCTDL simulated-rare tasks of
Part 2 (non-normal, non-VID classes were never extracted) and the next datasets.

### 1.4 Ensemble against best single, or method-agnostic? Keep the ensemble.

- A method-agnostic primary ("the best method of the ladder against the probe") picks a maximum over
  many correlated methods on 124 images. That number is biased upward by construction and cannot
  be a primary [unverified: standard selection argument, no source opened].
- One named contrast is the only thing that can carry a claim, and Lam's stated aim is an ensemble.
- But three defects in the Opus wording:
  1. **"Best single FM" is not defined.** If DINOv3-L alone beats MIRAGE, "ensemble minus MIRAGE" is
     no longer "minus the best". Rule: the ensemble must beat **each of its two members** under the
     same head (both lower bounds above zero). No multiplicity correction is needed for an
     "all must hold" rule [unverified: intersection-union argument].
  2. **"z-score of class probabilities" is a new, untested rule.** Probabilities sum to one and
     saturate. Use the P7 rule on the quantity that is stored: per model and query, z-score the five
     probe **logits**, add with equal weight, largest wins, ties by the fixed class order. The
     prototype-cosine version (exactly P7) is a named secondary for continuity.
  3. **Two changes at once.** Against P7 the proposal changes the second member (DINOv3-L for DINOv2)
     and the head (probe for prototype). Named secondaries must separate them: MIRAGE+DINOv2 with
     the probe; MIRAGE+DINOv3-L with prototypes.
- The probe as the head is sound: it is at least as good as the prototype in every stored cell
  (MIRAGE +0.012 development K=4, +0.014 hold-out K=8; DINOv2 +0.029 and +0.064) [P6].

### 1.5 Power for +0.03

How I estimated: from `[zip]`, per query image, the mean over draws of (ensemble correct minus
MIRAGE correct); then a bootstrap that resamples images within each class and resamples draws
[mine: power.py, kmean.py]. The probe-level fusion is not stored on development (the probe exists
only at K=4, 125 episodes [zip]), so I assume it has the same paired spread as the prototype
fusion. P6 spreads are of the same size or smaller (SD over draws 0.019-0.029) [P6].

Two different questions give two different answers.

| Question | Spread | Result |
|---|---|---|
| "For these 124 images, over support draws" (v1's kind of spread, `protocols.md:125`) | SD over draws 0.030 at K=8 (0.027 on the first 25) [P7; mine] | 8 draws give 80% power for +0.03 (two-sided 5%); 25 draws give SE 0.006. Never the limit |
| "For new images of these diseases" | SE at K=8: 0.0192 (25 draws), 0.0177 (100 draws), 0.0174 (images only). K-mean: 0.0136 (25), 0.0122 (100) [mine] | Power for +0.03: **K=8 35-39%**, K-mean 60-69%. Smallest difference found with 80% power: K=8 0.050-0.054, K-mean 0.034-0.038 |

- More draws barely help: at K=8, 5 / 10 / 25 / 100 draws give SE 0.025 / 0.022 / 0.019 / 0.018
  [mine: kmean.py]. The 124 images are the limit.
- For 80% power at +0.03, K=8, about **70 query images per class** would be needed (we have 19-27)
  [mine: kmean.py, from the within-class variance 0.040].
- The hold-out (22 images) cannot test +0.03 at all; it is a direction check only.
- Use all 100 locked draws for frozen methods (CPU, free). "First 25" is needed only for
  per-episode GPU methods. Paired comparisons use the draws both sides have.
- The decision interval is the image-and-draw bootstrap, 95%. The SD over draws is reported beside
  it, as in v1.

### 1.6 Multiplicity over methods x models x K

1. **One primary** (1.3-1.4). No correction.
2. **One small named family:** each ladder rung against the probe on MIRAGE, K=8, development.
   About eight contrasts. Simultaneous 95% intervals from the same bootstrap resamples (the maximum
   over contrasts of the standardised difference) [unverified: max-T method, no source opened].
3. **Everything else is a grid,** reported in full with spread, no stars, no "best cell" sentence.
4. **Any method called "best" on development is re-estimated where it was not chosen:** hold-out,
   OCTDL simulated-rare tasks, external rare images.
5. "Fixed ladder order" must not mean stacking. Every rung is compared with the same baseline on the
   same model; one pre-named full stack at the end, nothing else combined.

### 1.7 The safety endpoint: right idea, four holes

1. **Which nine-way head?** v1's head is one prototype per class from **all** eligible rare images,
   not K-shot (`FM_PROTOCOL.md:263-270`). It does not sit on the same episodes as a K=8 primary.
   Fix: co-primary safety on the locked v1 roster (fused by z-scoring the nine cosines per model);
   a K-shot nine-way audit as a secondary that every rung, fine-tuned ones included, can run.
2. **A low false-rare rate can mean "never says rare".** MIRAGE calls 26 of 332 OCTDL normals normal
   and sends most to a common disease (`protocols.md:188`). Add the R1 condition: the share of
   external rare images called any rare class must not fall.
3. **OCTDL is clustered:** 332 images, 110 patients; MIRAGE's 30 false-rare images come from 23
   patients [mine: frr.py]. Use a patient bootstrap; test OCTID and OCTDL separately, both must pass.
4. **Margin.** Paired SE of a false-rare difference is 0.008-0.021 on OCTID and 0.011-0.032 on OCTDL
   [mine: frr.py]. Proposal: the one-sided 95% upper bound of (ensemble minus MIRAGE) must be below
   **+0.03** in each source (MIRAGE's rates are 0.019 and 0.090 [R1]). With a true difference of
   zero and SE 0.011 this passes about 86% of the time [unverified: normal approximation].
   **Disclosure:** for MIRAGE+DINOv2 this is already known: upper bounds -0.005 and +0.027
   [mine: frr.py]. The margin is blind only for a DINOv3 pair. Lam should fix the margin before any
   DINOv3 feature exists.

### 1.8 DINOv3-L in the primary? Only through a rule decided at the lock.

Against: not extracted; gated checkpoint; the default processor squashes while the DINOv2 arm
centre-crops [notes: dinov3]; it changes two things against P7. For: it is the only pair whose
development result is unseen, and the multi-dataset phase will want it [notes: dinov3, section 7].

Rule: **fix now** ViT-L/16, `pooler_output`, and the DINOv2 arm's input transform unchanged
(native-aspect input is a named sensitivity). At the v2 input lock, before any score: if DINOv3-L
passed its pilot and repeat check, the primary pair is MIRAGE+DINOv3-L; if not, it is
MIRAGE+DINOv2 and DINOv3 is reported "unavailable". No choice after scores.
Note: the DINOv2 arm is ViT-**Base** (`protocols.md:41`; `fm_extract.py:33`, `facebook/dinov2-base`).
The DINOv3 note calls ViT-L "the same capacity class as the DINOv2 arm"; that is wrong. A
DINOv3-L against DINOv2-B comparison mixes size with generation, so it is never read as
"DINOv3 is better"; only the ensemble contrasts use it.

### 1.9 Other corrections

- **Efficiency as a ratio breaks.** "Gain per GPU-hour" divides by about zero for CPU methods and
  flips sign for losses. Report a table instead: gain with interval, GPU-seconds per episode to
  adapt, milliseconds per query (an ensemble runs two encoders: 13.6 + 14.0 ms on the L4,
  `step11_pilots.txt`), trainable parameters, extra labelled data touched. A method is "efficient"
  if no cheaper method has a higher gain. Ratios only for methods whose interval excludes zero.
- **Centring is listed as "dropped for good"** (`protocols.md:276`). Opus's ladder brings it back.
  That needs a dated line from Lam. It also uses the Kermany mean, which carries the source
  difference between Kermany and the web-collected rare images (`CONTEXT.md:130-132`).
- **Claim wording, fixed in advance:** "supported on re-used data" = primary interval above zero
  for both members, K=8 interval above zero, safety passed, hold-out difference not negative.
  "Confirmed" = the same on unseen data. Anything else = "not supported", reported in full.

---

## Part 2. B7: fine-tuning conditions (decided)

### 2.1 What the literature allows (basis for every choice)

- Fine-tuning is sensitive to the learning rate, and a task has no validation set to pick it; the
  best rate is stable within a domain, so P>M>F searches {0.01, 0.001, 0.0001, 0} on 5 extra tasks
  per domain; a single fixed rate was "poorly tuned for some datasets"; 50 steps and augmentation
  with probability 0.9 [pdf: 2204.07305, lines 193-215, 395-398, 743-747].
- Fixed recipe without a search: SGD, rate 0.01, momentum 0.9, 100 epochs [pdf: 1912.07200,
  lines 504-507]. Meta-learning lost to plain transfer there [notes].
- Head first, then the backbone (LP-FT); its rates and stopping were chosen on in-distribution
  validation data; early stopping does not remove feature distortion [pdf: 2202.10054, lines
  552-553, 606-611].
- Medical few-shot: search "on a separate set of sampled FSL tasks"; Adam, zero-initialised head,
  rate 1e-4 with at least 120 steps "optimal or close to optimal"; fine-tuning foundation models
  was not tried as "computationally prohibitive" [pdf: 2408.08058, lines 180-186, 251-255, 432-433].
- Adapters prefer rates about ten times higher than full fine-tuning [pdf: 2305.08252, lines 751-752].
- Earlier adapters "use the large-scale test set to adjust their hyperparameters" [pdf: 2312.12730,
  line 68]. No opened source shows LoRA beating a linear probe at 1-8 shots on a medical FM [notes].

### 2.2 Variants

| # | Variant | Tuned | Models | Cost shape |
|---|---|---|---|---|
| F0 | Base-class adaptation, once per model | LoRA (rank 8, query and value projections, all blocks), cosine-prototype episodic loss, 4-way, 500 episodes, Kermany **validation-split** images only | MIRAGE, general member | One run per model; features re-cached, so all 100 draws and every frozen head are free afterwards |
| F1 | LoRA on the K supports | LoRA as above + head; head starts from the frozen probe (LP-FT order) | MIRAGE, general member, RETFound | One run per episode |
| F2 | Last block | Last transformer block + final norm + head, same start | MIRAGE | One run per episode |
| F3 | Full LP-FT | Everything, same start | MIRAGE | One run per episode; gated |

- Inception-v3 is not fine-tuned (reference only; Keras stack; `CONTEXT.md:43` puts it out of scope).
- RETFound is in F1 only: it answers the open recipe gap of `protocols.md:186` and section 8's plan.
- LoRA rank 8 and the query/value placement are my fixed defaults [unverified: LoRA paper not opened].
- F0 uses a prototype loss because the evaluated head is a prototype or probe on the features;
  caveat from BSCD-FSL above. F0 is never stacked with F1-F3.
- Not allowed: prompt tuning, text arms, anything that uses query images or unlabelled development
  images during adaptation, test-time statistics from queries, GAN or October augmented images.

### 2.3 Gates (in order; each failure is reported and the variant is not run on targets)

1. **K gate.** Per-episode variants run at K=8 on development, and K=8 and K=16 on the hold-out.
   Not at K=1-4: opened sources report gains from about 5 shots upward, and from n >= 20 against
   a strong probe [notes; pdf: 2408.08058 lines 400-404].
2. **Recipe gate.** If the search (2.5) picks rate 0, the variant is "search chose no fine-tuning".
3. **Transfer gate.** With its chosen rate, the variant must beat the frozen probe of the same model
   by at least +0.02 mean macro recall on 10 **OCTDL simulated-rare** tasks never used in the search.
4. **Ladder gate.** F3 runs only if F1 or F2 on MIRAGE has a development K=8 interval above zero
   (section 8 already says full fine-tuning only after light adaptation gains, `protocols.md:272`).
5. No gate depends on a cheaper frozen method "gaining": the frozen ladder and F1 answer different
   questions, and Lam asked to test everything.

### 2.4 Data each variant may touch

| Data | F0 | F1-F3 | Recipe search | Transfer gate |
|---|---|---|---|---|
| The episode's 5K support images (+ fixed augmentations) | no | **yes, only these** | no | no |
| Query fold, other development images, the 22 hold-out | never | never | never | never |
| Kermany validation split (CNV, DME, DRUSEN, NORMAL; patient-grouped, `stage0_manifest.py:72`) | train on it | no | 4-way K=8 tasks, one image per patient | no |
| Kermany train split (nine-way fitting roster) and test split (169 audit normals) | never | never | never | never |
| OCTDL AMD, DME, ERM, RVO (1,231 / 147 / 155 / 101 images; 421 / 107 / 71 / 50 patients [notes: datasets]) | never | never | no | 4-way K=8 tasks, split by patient |
| OCTDL NO, all VID (MH 39, LH 26, VMT 11), RAO; OCTID Normal, CSR, MH | never | never | never | never (query-only, as in A1) |
| OCTID AMD, DR | never | never | never | never (kept unseen for later) |

Overlap check: OCTID has CSR and MH, both target classes; OCTDL's `condition` says MH for all 76
VID images and only `subcategory` separates the 39 true MH (`FM_EXTERNAL_MAPPING.md:19-35`). So
the whole VID class and both OCTID classes are barred from any training or tuning. OCTDL AMD and
DME overlap Kermany's CNV/DRUSEN and DME, so for F0 the transfer gate measures a new device, not
new classes; state this. The gate reads OCTDL disease images; v1 kept OCTDL query-only (A1), so v2
must say this change in one dated line.

### 2.5 Hyperparameters

- Fixed for all: AdamW, no weight decay on LoRA or head, constant rate, **100 steps**, every step
  uses all 5K supports with fresh augmentation, float32, final-step weights.
- Augmentation (fixed, applied before each model's own transform): random crop 80-100% of the area
  resized back, horizontal flip 0.5, rotation within 5 degrees, brightness and contrast within 10%.
  The same list defines the frozen "support augmentation" rung (16 fixed views per image, cached).
- Searched: learning rate only. F1 and F0: {0, 1e-4, 1e-3, 1e-2}. F2 and F3: {0, 1e-5, 1e-4, 1e-3}.
  Per variant and model, on 10 Kermany-validation tasks (F0: on OCTDL tasks 1-10, gate on 11-20).
  Highest mean macro recall wins; within 0.01, the smaller rate wins. The full search table is
  published.
- Never tuned on Yoo images, on OCTID, or on target queries. Nothing is re-searched after a target
  score exists.

### 2.6 Early stopping, episodes, seeds

- **No early stopping and no checkpoint choice.** There is no validation data inside an episode, and
  taking images from the supports would change K. Fixed steps, last weights.
- Episodes: the first 25 locked draws x 5 folds at K=8 = 125 development episodes per variant and
  model; first 25 hold-out draws at K=8 and K=16 = 50. With 25 draws the image-and-draw SE is
  0.019 against 0.018 at 100 draws [mine], so more draws would buy nothing.
- K-shot nine-way audit: first 5 draws x 5 folds. Common prototypes from the first 100 roster images
  per class; about 1,350 images to embed, about 19 s per episode at 14 ms per image
  (`step11_pilots.txt`).
- Seed: one training seed per run, derived from (42, variant, model, fold, K, draw). The first 5
  draws are repeated with a second seed; report the seed SD beside the draw SD. F0: seeds 42 and 43.
- A run with a non-finite loss is reported as failed, never re-run with another setting; results
  are shown without it and with the frozen probe's predictions in its place.

### 2.7 Compute, pilot, stop rule

- Budget: **40 L4 GPU-hours** for all fine-tuning (search and gates 8, targets 32). The figure is
  my choice; Colab unit cost [unverified].
- Timing pilot: the first 10 recipe-search runs of each variant and model (Kermany tasks, no Yoo
  label read, no target score). Projected hours = seconds per run x 175 target episodes.
  Worst case 875 target episodes (F1 x 3 models, F2, F3); at 2 minutes each, 29 hours [unverified].
- If the projection exceeds the budget, drop before any target score, in this order: F3; RETFound
  in F1; F2; then 25 draws to 10 (SE 0.022 [mine]). Never below 10 draws.
- If a session dies mid-run: keep only complete draws (all 5 folds), report how many; a cell with
  fewer than 10 draws is labelled incomplete and enters no comparison. Nothing is dropped or added
  after looking at scores.

### 2.8 What is reported for each variant and model

Paired difference against the frozen probe of the same model (K=8 development; hold-out K=8 and
K=16 as "re-used set"), with the image-and-draw interval and the SD over draws; recall per disease;
K-shot nine-way false-rare counts by source with the share of external rare images called rare;
five-way share correct on OCTID CSC/MH and OCTDL MH; GPU-seconds per episode, milliseconds per
query, trainable parameters; final training loss; failed runs; seed SD; the search table and every
gate outcome, including variants that never ran. Favourable or not (A1 rule 6).

---

## Part 3. Decisions that stay with Lam

1. Primary K: K-mean (my recommendation) or K=8 (Opus).
2. Primary pair: the lock-time rule of 1.8, or DINOv2 only, or DINOv3 only.
3. Safety margin +0.03 per source, fixed before any DINOv3 feature.
4. Whether centring returns (`protocols.md:276` says dropped for good).
5. Whether v2 may read OCTDL AMD/DME/ERM/RVO images for the transfer gate (changes A1's query-only rule).
6. The 40 GPU-hour budget.
