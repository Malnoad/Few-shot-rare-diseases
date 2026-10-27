# Mock review of the reproduction report (2026-10-23)

**Manuscript under review:** `README.md` (Status, Results, Differences from the paper) together
with `docs/research/REPLICATION.md`, both read as a reproduction report of Yoo, Choi & Kim
2021 (MBEC 59:401-415, doi:10.1007/s11517-021-02321-1), at commit `5c77ce2`.

**Contents**
1. Corrections made to `README.md`
2. Mock peer review (3 reviewer reports, forensic audit, synthesis)
3. Statistical reporting audit
4. Draft Data and Code Availability statement
5. Proposals (not applied), in priority order
6. Sources and how every number was checked

Skills used: `nature-reviewer` (part 2), `nature-statistics` (part 3), `nature-data` (part 4),
`nature-writing` (wording of the part 4 statement only).

---

## 1. Corrections made to `README.md`

Only clear factual or statistical-reporting errors were changed. Each was checked against the
committed files named in the right column.

| # | Where | Was | Now | Evidence |
|---|---|---|---|---|
| F1 | Status, "Short version" | "over five training seeds it is tied with basic augmentation. Both beat real images only." | `gan` is "not distinguishable from basic augmentation (`gan` ahead on 3 of 5 seeds)"; both augmentations have higher *mean* accuracy than `none` (+1.9 / +2.2 points) and more rare images right, "but five seeds cannot separate the accuracy gains from training noise." | `results/analysis/paired_seeds.csv`: basic − none = +2.246 ± 3.529, ahead on 3/5, Wilcoxon p = 0.31; gan − none = +1.898 ± 2.216, ahead on 4/5, p = 0.19. `results/seed_check.csv`: rare correct means none 14.2, basic 18.0, gan 18.6. "Beat" claimed a separation that the seed data do not show. |
| F2 | Results, "Five training seeds" | "CycleGAN and basic augmentation are **tied**." | "are **not separated** (which is not the same as shown to be equal)", plus a new bullet reporting the `basic`/`gan` vs `none` seed differences and the n = 5 Wilcoxon floor (0.0625). | `paired_seeds.csv` as above. A t-based 95% CI for the mean gan − basic difference, computed from the same five pairs, is −3.8 to +3.1 points. That is far too wide to call the variants equivalent. The Wilcoxon floor 2/2⁵ = 0.0625 is also printed by `analysis.py:144`. |
| F3 | Stage 0, "What and why" + the note below it | "Kermany official train/test split (independent patients, …)" and "some patient IDs appear in both" | The split is stated to be **not** patient-disjoint, and the note gives the size: 566 of 633 test patients, 914 of 1,000 common-class test images. | `data/manifest.csv` (`pid` column): 633 unique test pids among the four common classes, 566 of them also in train/val, and their images make up 914 of the 1,000 test images. The README contradicted itself (line 260 said "independent patients", line 283 said they are not). `stage0_manifest.py:7` repeats the error; see proposal P6. |

Note on F3: it was checked against `data/manifest.csv` (the committed split of record), not
against a result CSV. If you want README fixes limited to result CSVs only, revert F3 and keep it
as a proposal.

Everything else that a reviewer would change is in Section 5 as a proposal.

---

## 2. Mock peer review

### Review setup
- **Input scope** README sections Status, Results (including Differences from the paper), the
  Stage 0 and Stage 4 method paragraphs they depend on, and `docs/research/REPLICATION.md`. The
  review was run against the text as it was before the Section 1 corrections.
- **Assessment boundary** A reproduction report, not a novel-method paper. It is judged on
  fidelity to the original, soundness of the comparisons, and honesty of the claims. The paper
  was read in its open PMC version (PMC7829497) through a web fetch that summarised the page. Its
  Table 3 and Table 4 values match `stage4_evaluate.py` PAPER and the README. Other details of
  the paper quoted below come from that summary and are marked *(PMC, unverified verbatim)*.
- **Shared manuscript claim summary** The pipeline was rebuilt from public data. At seed 42 the
  CycleGAN variant reaches 92.95% (paper 92.1%). Over five more training seeds CycleGAN and basic
  augmentation are not separated, and both are said to beat real images only. The paper's
  largest effect, rescuing rare classes the baseline misses, is not reproduced because the
  reproduction's baseline already gets most rare test images right.
- **Visible evidence base** `results/stage4/*.csv`, `results/seed_check.csv`,
  `results/analysis/*.csv`, `results/stage3_*/{data_counts,history,test_predictions}.csv`,
  `results/stage2/qc_*.csv`, `data/manifest.csv`, the stage scripts.
- **Missing materials affecting confidence** Per-seed predictions for seeds 1 to 5 (not
  retained). Patient identifiers for the rare-disease images. The paper's RCI formula. Any
  human review of the accepted CycleGAN images.
- **Independence limitation** The three reports below were written in **one shared context** by
  one agent, each with its emphasis brief fixed beforehand. They are **not mutually blind**, and
  overlap between them is not evidence of independent agreement.

Emphasis briefs, fixed before drafting:
- Reviewer 1 statistical validity and the experimental unit.
- Reviewer 2 fidelity of the reproduction to the original method.
- Reviewer 3 significance, transparency and readability for non-specialists.

### Reviewer 1

- **Overall assessment** This is a careful and unusually candid reproduction. It already flags
  that a single seed is not enough. However, its quantitative claims still rest on inference
  procedures that treat test images as independent. The test images are in fact clustered within
  patients, and most of those patients also appear in training. The headline table and the
  "both beat real images only" statement are therefore more confident than the design supports.
- **Who would be interested in the results, and why** Researchers in medical imaging who use
  GAN augmentation for rare classes. A careful negative or null replication of a frequently
  cited augmentation claim is useful to them.
- **Major strengths** The paired analyses use the right pairing (same test images, same seeds).
  The report states that the seed-42 paired test covers only test-image sampling. Rare-class
  granularity (one image moves recall by 0.20 to 0.25) is stated plainly.
- **Major Concerns**

  - **Concern ID** R1-M1
  - **Severity** Major
  - **Blocking** Yes
  - **Axis** technical soundness
  - **Claim pointer** README Results, headline table "95% CI"; "McNemar p < 0.001"; REPLICATION
    Findings "gan beats basic by 4.1 points (CI 2.6-5.8, McNemar p < 0.0001)"
  - **Evidence pointer** `stage4_evaluate.py:57-66` and `analysis.py:32-48` resample or pair
    individual images. `data/manifest.csv` gives 633 patients for the 1,000 common-class test
    images, with up to 13 images from one patient.
  - **Concern** The bootstrap CIs and the McNemar test treat the image as the independent unit.
    Images from the same patient are correlated, so these intervals are too narrow and the p
    values too small by an unknown amount.
  - **Why it matters** The only significant results in the report come from these tests.
  - **Resolution test** Repeat the bootstrap with resampling of patients (pid parsed from the
    Kermany file name; rare images as their own clusters unless patient IDs are known), and
    report how the CI widths change. This needs no training.

  - **Concern ID** R1-M2
  - **Severity** Major
  - **Blocking** Yes
  - **Axis** technical soundness
  - **Claim pointer** README Short version "Both beat real images only"; Five training seeds
    "CycleGAN and basic augmentation are tied"
  - **Evidence pointer** `results/analysis/paired_seeds.csv`
  - **Concern** With five seeds the paired differences against `none` are +2.25 ± 3.53 (basic)
    and +1.90 ± 2.22 (gan), ahead on 3 and 4 of 5 seeds. The Wilcoxon test cannot reach p < 0.05
    at n = 5. "Beat" claims a separation that is not shown. "Tied" claims an equivalence that is
    not shown either; the paired interval for gan minus basic spans several points either way.
  - **Why it matters** These two sentences are the report's main conclusions.
  - **Resolution test** Either soften the wording to "not separated", or run the planned
    20-seed sweep (REPLICATION C1). Pre-specify the comparison and an equivalence margin if
    "tied" is to be claimed.

  - **Concern ID** R1-M3
  - **Severity** Major
  - **Blocking** No
  - **Axis** technical soundness
  - **Claim pointer** README "Headline metrics (seed 42)"
  - **Evidence pointer** `results/seed_check.csv` (seeds 1 to 5 only); `colab_setup.ipynb`
    Part 4 keeps only accuracy, rare_correct and drusen_recall per seed
  - **Concern** The table labelled headline is a single seed that the report itself shows to be
    unrepresentative (basic at seed 42 is 88.85%, below all five other basic seeds). The seed
    sweep reports only accuracy, so κ and MCC have no seed variability at all. Seed 42 is not
    pooled with seeds 1 to 5.
  - **Why it matters** A reader skimming the table takes the seed-42 ordering as the result.
  - **Resolution test** Make the seed summary (all six seeds, mean and SD, plus κ and MCC) the
    primary table. Keep seed 42 as the run of record for the per-class and confusion analyses.

- **Minor Comments**
  - **Concern ID** R1-m1. **Severity** Minor. **Axis** technical soundness. **Affected element**
    REPLICATION Findings, `paired_seed42.csv`. **Evidence pointer** six McNemar tests (three
    pairs, all and rare subsets) and three Wilcoxon tests. **Issue** No multiplicity policy is
    stated. **Required correction** State that the tests are descriptive, or apply and name a
    correction.
  - **Concern ID** R1-m2. **Severity** Minor. **Axis** readability. **Affected element** headline
    table header "95% CI". **Evidence pointer** The method is given only in the Stage 4 section of
    the README. **Issue** The CI type and its scope (test images, fixed head) are not stated
    where the CI appears. **Required correction** Add a one-line footnote.
- **Technical failings that need to be addressed before the case is established** R1-M1, R1-M2.
- **Assessment against Nature-style criteria** Technical soundness is partly established. The
  descriptive results are sound. The inferential results are overstated in both directions.
- **Recommendation posture** Major revision.

### Reviewer 2

- **Overall assessment** The pipeline follows the paper closely where the paper is specific, and
  every departure is listed. Three departures are larger than the Differences table suggests:
  the classifier head stops after a handful of epochs, the validation sets differ between arms,
  and the CycleGAN images are filtered in the same feature space that the classifier uses.
  Together they may explain why the baseline behaves so differently from the paper's.
- **Who would be interested in the results, and why** Anyone trying to reuse the Yoo et al.
  rare-disease dataset or method. Readers need to know which of the paper's results reproduce
  under a faithful setup.
- **Major strengths** Data splits are committed and enforced. The rare test hold-out never
  reaches the CycleGAN or the QC reference set (`stage2_cyclegan.py:78`). Paper values are
  encoded in code, and they match the paper's Table 3.
- **Major Concerns**

  - **Concern ID** R2-M1
  - **Severity** Major
  - **Blocking** Yes
  - **Axis** technical soundness
  - **Claim pointer** README "Our none already gets 0.75-0.80 on every rare class ... The cause
    has not been investigated"; REPLICATION L7
  - **Evidence pointer** `results/stage4/recall.csv`. The paper's Table 4 gives the baseline
    0.200, 0.200, 0.000, 0.000 and 0.250 (3 of 22 rare images correct); the reproduction gets 17
    of 22 at seed 42 and 11 to 17 over seeds.
  - **Concern** The reproduction's baseline differs from the paper's on exactly the effect the
    paper is about. Until the difference is explained, the reproduction cannot say whether the
    paper's augmentation benefit was real but small, or an artefact of a weak baseline.
  - **Why it matters** This is the central replication question, not a side note.
  - **Resolution test** Run `none` with the paper's fixed 250 epochs (`--patience 0`) over
    several seeds and report rare recall. Report each rare test image's predicted class and
    softmax probability from `results/stage3_none/test_predictions.csv`.

  - **Concern ID** R2-M2
  - **Severity** Major
  - **Blocking** No
  - **Axis** technical soundness
  - **Claim pointer** README Differences "Early stopping on validation loss, patience 20 ...
    Heads stopped after 22-27 epochs"
  - **Evidence pointer** `results/stage3_*/history.csv`. The lowest validation loss is at epoch
    index 5 (none), 6 (basic) and 1 (gan). `results/stage3_*/data_counts.csv` show that `none`
    has 0 rare validation images, while `basic` and `gan` take rare validation from 10% of their
    augmented pool (`stage3_classifier.py:168-171`).
  - **Concern** The heads used are from epochs 2 to 7, against 250 in the paper. The stopping
    criterion is computed on a different validation set in each arm. For `none` it ignores the
    rare classes completely. For `basic` and `gan` the rare validation images are augmentations
    of, or are generated from, the same 19 to 27 training images, so they are not independent
    of training.
  - **Why it matters** Model selection is part of the treatment. Arm differences can come from
    when each head stopped rather than from the training data.
  - **Resolution test** Report results with a fixed epoch count shared by all arms, or select
    using common-class validation only, which is identical across arms.

  - **Concern ID** R2-M3
  - **Severity** Major
  - **Blocking** No
  - **Axis** technical soundness
  - **Claim pointer** README Differences, automatic QC replaces two ophthalmologists
  - **Evidence pointer** `stage2_cyclegan.py` `qc` docstring (kNN distance in Inception-v3
    feature space); the classifier uses the same frozen Inception-v3 features
  - **Concern** Accepted GAN images are, by construction, close to the real disease in the
    classifier's own feature space. The filter selects images that are easy for this head, which
    is a different selection from a clinician's realism check. Pass rates differ from the
    paper's ranking (lowest for RP and MacTel here, Stargardt and RP in the paper).
  - **Why it matters** The `gan` arm is not the paper's `gan` arm. Its result says little about
    whether clinically realistic synthetic images help.
  - **Resolution test** Report `gan` with unfiltered generated images as well, or with QC in a
    different feature space. Even a small blinded human rating (REPLICATION L4) would help.

- **Minor Comments**
  - **Concern ID** R2-m1. **Severity** Minor. **Axis** technical soundness. **Affected element**
    README Five training seeds, list of what the seed controls. **Evidence pointer**
    `stage3_classifier.py:164` (`rng.sample` of GAN images). **Issue** The seed also picks which
    3,000 of CSC's 3,370 accepted images are used. It does not touch Stage 1, the CycleGANs, the
    QC, or the test hold-out. **Required correction** Name both what the seed varies and what
    stays fixed.
  - **Concern ID** R2-m2. **Severity** Minor. **Axis** technical soundness. **Affected element**
    Differences table, kNN row. **Evidence pointer** `results/stage4/recall.csv` knn row
    (0.6, 0.4, 0, 0, 0) and the paper's Table 4 kNN row (0.6, 0.8, 0, 0, 0). **Issue** Rare recall
    for kNN largely matches the paper; the gap is in the common classes (DME 0.696). **Required
    correction** Say where the 4.8-point gap is.
- **Technical failings that need to be addressed before the case is established** R2-M1.
- **Assessment against Nature-style criteria** Strong on transparency. The originality claim is
  the replication result itself. That claim is weakened by the unexplained baseline and by the
  differences in selection and stopping.
- **Recommendation posture** Major revision.

### Reviewer 3

- **Overall assessment** The report is readable, honest about its limits, and keeps a clean
  ledger of paper items (REPLICATION). It reads as a lab notebook rather than a report. The
  reader meets a single-seed headline table first, then learns that the ordering in it is not
  robust. The availability of the derived data that support the results is not stated.
- **Who would be interested in the results, and why** Students and reviewers learning how
  fragile small-test-set medical AI claims are. This is a good teaching case.
- **Major strengths** Every number links to a CSV. Non-comparable metrics (RCI, `proto`, kNN
  AUCs) are flagged. Rare-class granularity is explained in plain language.
- **Major Concerns**

  - **Concern ID** R3-M1
  - **Severity** Major
  - **Blocking** No
  - **Axis** readability for nonspecialists
  - **Claim pointer** README section order (headline table before the seed table); Short version
  - **Evidence pointer** README Results structure
  - **Concern** The order of the evidence does not match the order of the conclusions. The
    conclusion is "not separated over seeds", but the first and most prominent table supports
    "gan wins".
  - **Why it matters** Non-specialists will cite the table.
  - **Resolution test** Lead with the seed summary. Present seed 42 as the run used for
    per-class and error analysis.

  - **Concern ID** R3-M2
  - **Severity** Major
  - **Blocking** No
  - **Axis** technical soundness
  - **Claim pointer** README "Test set 1,022 images"; Stage 0 paragraph
  - **Evidence pointer** `data/manifest.csv` (566 of 633 test patients overlap with train)
  - **Concern** The common-class accuracy of every arm is measured largely on patients the model
    has seen. The README mentions this only as a Stage 0 console note.
  - **Why it matters** Absolute accuracies (90 to 93%) are likely optimistic for new patients.
    Comparisons between arms are less affected, since all arms share the same leak.
  - **Resolution test** State the overlap in Results. Optionally report accuracy on the 86
    test images from patients not seen in training as a descriptive check.

  - **Concern ID** R3-M3
  - **Severity** Major
  - **Blocking** No
  - **Axis** technical soundness
  - **Claim pointer** No Data or Code Availability statement exists
  - **Evidence pointer** `COLAB.md` "What must be in Drive"; the derived images, checkpoints and
    feature cache are in a private Google Drive
  - **Concern** The accepted GAN images and the feature cache that produced every number are not
    accessible to readers. Seeds 1 to 5 kept no predictions.
  - **Why it matters** No result beyond seed 42 can be re-audited, and the `gan` arm cannot be
    rebuilt without about 50 GPU-hours.
  - **Resolution test** Deposit the derived data with a persistent identifier, and add an
    availability statement.

- **Minor Comments**
  - **Concern ID** R3-m1. **Severity** Minor. **Axis** readability. **Affected element** headline
    table RCI column. **Evidence pointer** footnote † says it is not comparable. **Issue** A
    column flagged as non-comparable still sits next to the paper columns. **Required
    correction** Move RCI to a separate table, or drop the paper RCI.
  - **Concern ID** R3-m2. **Severity** Minor. **Axis** readability. **Affected element**
    REPLICATION Table 3 row "gan ties basic"; `docs/research/CONTEXT.md` "both beat none".
    **Evidence pointer** `paired_seeds.csv`. **Issue** Same overstatement as in the README.
    **Required correction** Align the wording.
- **Technical failings that need to be addressed before the case is established** None that
  block on their own. R3-M2 and R3-M3 affect how far the numbers can be trusted.
- **Assessment against Nature-style criteria** Readability and transparency are high for a
  repository README. As a report, it needs restructuring and an availability statement.
- **Recommendation posture** Minor to major revision.

### Forensic consistency audit (editor-facing; not shown to reviewers)

| Check | Status | Detail |
|---|---|---|
| Headline accuracy, κ, MCC, RCI, CIs vs `metrics.csv` | passed | All 5 rows match to display precision. |
| Paper values vs paper Table 3 | passed | `stage4_evaluate.py` PAPER = PMC Table 3 for none, basic, gan, knn, proto. |
| Per-class recall table vs `recall.csv` | passed | All 45 cells. |
| Seed table (mean ± SD, range, rare mean and range, DRUSEN mean) vs `seed_check.csv` | passed | Recomputed with SD ddof = 1: none 90.69 ± 1.86, basic 92.93 ± 1.85, gan 92.59 ± 1.29; rare 14.2 / 18.0 / 18.6. |
| gan − basic −0.35 ± 2.77 (−4.30 to +2.94), 3 of 5 | passed | `paired_seeds.csv`. |
| Confusion-derived counts (17/18/19 rare, 906/890/931 common, DRUSEN→CNV 68/64/36, common→rare 0/0/1, proto 66, DME 234/213/229) | passed | Recomputed from `confusion_*.csv`. Accuracy identity holds: gan (19 + 931) / 1,022 = 92.95%. |
| "basic at seed 42 is its lowest seed" | passed | 88.85 is below the basic minimum over seeds 1 to 5 (90.90). |
| "Most of gan's seed-42 gain is in DRUSEN" | passed | gan − none = +27 images in total, DRUSEN +35, rare +2. |
| Early stopping 22-27 epochs | passed | history rows: 26 / 27 / 22. |
| QC counts and pass rates 84/65/31/55/21% | passed | `qc_summary.csv`. |
| kNN 86.40, cosine K=1 91.10 | passed | `knn_sweep.csv`. |
| "both beat none" (Short version) | confirmed_internal_error | Contradicted by `paired_seeds.csv` (3/5 and 4/5 wins, no test can reach p < 0.05). Fixed (F1). |
| "tied" (gan vs basic) | confirmed_internal_error | An equivalence claim from an underpowered null result. Fixed (F1, F2). |
| "independent patients" (Stage 0) | confirmed_internal_error | Contradicted by the README's own note and `data/manifest.csv`. Fixed (F3). |
| REPLICATION "McNemar p < 0.0001", "p = 0.001" | passed | 1.4e-6 and 0.0013. |
| Image as unit of analysis | aggregation_ambiguity | CIs and McNemar treat images as independent; 633 patients for 1,000 images. |
| Rare-image patient structure | unresolved_input_needed | Rare files are numbered (`01.PNG`); patient IDs are unknown. Per the paper, rare images include web-sourced cases *(PMC, unverified verbatim)*. |
| Seeds 1-5 κ/MCC/per-class values | provenance_gap | Not retained (`colab_setup.ipynb` Part 4 keeps three summary columns). |

### Cross-review synthesis (post-review; not shown to reviewers)

Reports were drafted in one context (see the limitation in Review setup). "Consensus" below
means raised in two or more reports, not independent agreement.

- **Consensus strengths** Candour about limitations, a traceable number for every claim,
  enforced splits, and paired comparisons.
- **Consensus blocking concerns**
  - Seed-level claims overstated (R1-M2, and through wording R3-m2). Fixed in the README; still
    present in REPLICATION and CONTEXT.
- **Other consensus major concerns**
  - Single-seed table presented as the headline (R1-M3, R3-M1).
  - Patient structure of the test set, through clustering (R1-M1) and train/test overlap
    (R3-M2).
- **Where emphasis differs across reviewers** Reviewer 2 alone treats the baseline discrepancy
  (R2-M1) as the central blocking issue, and alone raises selection and stopping confounds
  (R2-M2, R2-M3). Reviewer 3 alone raises data availability.
- **Forensic consistency findings** Three confirmed internal errors (all fixed in the README).
  One aggregation ambiguity (unit of analysis). One unresolved input (rare-image patient IDs).
  One provenance gap (per-seed outputs). All displayed numbers otherwise reconcile with the CSVs.
- **Minor revision checklist** Footnote defining the CIs. Multiplicity statement. Full
  description of what the seed varies. Location of the kNN gap. RCI out of the paper-comparison
  table. Wording aligned in REPLICATION and CONTEXT.
- **Broad-interest / significance readout** A transparent, partly negative replication of a
  rare-class GAN augmentation claim. Its interest rests on explaining the baseline discrepancy.
- **Most important issues to resolve before a strong case is established**
  1. Explain or bound the `none` rare-recall discrepancy (R2-M1).
  2. Replace image-level inference with patient-cluster inference and more seeds (R1-M1, R1-M2).
  3. Equalise model selection across arms (R2-M2).
  4. Make the derived data and per-seed outputs available (R3-M3).

### Risk / unsupported claims
- "Augmentation of either kind helps the rare classes" is directionally consistent (gan ahead
  of none on rare images in 5/5 seeds, basic in 4/5 with one tie), but there is no test and the
  hold-out is a single fixed draw of 22 images.
- "Over seeds, basic reaches a similar DRUSEN recall" is a comparison of means over 5 seeds,
  with no interval reported.
- Any statement about the paper beyond Tables 3-4 values rests on a summarised web fetch
  *(unverified verbatim)*.

---

## 3. Statistical reporting audit

### Statistics review scope
- **Input reviewed:** README Results, Five training seeds, Per-class recall, Confusion
  matrices, Differences; REPLICATION Findings; and the code that produced them
  (`stage4_evaluate.py:57-70`, `analysis.py:32-63`, `stage3_classifier.py:117-195`,
  `colab_setup.ipynb` Part 4).
- **Boundary / missing materials:** per-seed predictions (seeds 1-5) do not exist; patient IDs
  for rare images unknown. No new analysis was run beyond arithmetic on the committed CSVs and
  the manifest. The t-based CI in F2 is the only number computed here that is not in a CSV.
- **Study design readout:** three arms (`none`, `basic`, `gan`) that differ only in the
  rare-class training pool; frozen Inception-v3 features plus a softmax head. One fixed test
  set: 1,000 Kermany images (250 per class) plus 22 rare images (5/5/4/4/4, a single seed-42
  draw). Two sources of randomness are sampled: test images (bootstrap, seed 42 only) and
  head training (seeds 1-5). Not sampled: Stage 1 augmentation, CycleGAN training, generation
  and QC, the rare hold-out draw.
- **Independent unit and replication readout:**

  | Level | Count | Treated as independent? | Should be? |
  |---|---|---|---|
  | Common-class test image | 1,000 | yes (bootstrap, McNemar) | no: nested in 633 patients |
  | Common-class test patient | 633 (566 also in train) | no | yes, for generalisation claims |
  | Rare test image | 22 | yes | unknown: patient IDs not available |
  | Training seed (head) | 5 per arm, matched across arms | yes (Wilcoxon, SD) | yes, for training-noise claims |
  | CycleGAN / augmentation run | 1 per disease | not varied | a replicate if the claim is about the method |
  | Rare hold-out draw | 1 | not varied | a replicate if the claim is about rare-class recall |

### Major statistical issues

- **[P0] Two conclusions overstated relative to the seed evidence.**
  Evidence: `paired_seeds.csv` (see F1, F2).
  Why it matters: these are the report's main sentences.
  Fix: applied in the README. Still present in `REPLICATION.md` ("gan ties basic") and
  `docs/research/CONTEXT.md` ("both beat none"); see P1.

- **[P1] Image-level inference on patient-clustered data.**
  Evidence: 1,000 test images from 633 patients (up to 13 per patient); 914 images from patients
  also in training. `stage4_evaluate.py:63-64` and `analysis.py:47` resample images.
  Why it matters: CIs too narrow, McNemar p too small. The train/test overlap also inflates
  absolute accuracy for every arm.
  Fix: patient-cluster bootstrap (training-free, `analysis.py`). State the overlap in Results.
  Report the 86 images from unseen patients descriptively.

- **[P1] Two variance sources reported separately, never combined, and two not sampled at
  all.** The seed-42 bootstrap CI (half-width about 1.5 to 1.9 points) covers test sampling for
  one head. The seed SD (1.3 to 1.9 points) covers head training on one test set. The README
  correctly says the seed-42 paired test "only covers test-image sampling, not training
  randomness", but the headline table still shows test-only CIs. The rare hold-out draw and the
  CycleGAN/augmentation runs are fixed, so the reported variability is a lower bound.
  Fix: for the primary comparison, report the mean over seeds with a CI across seeds. Say in
  words that the test set and the generative models are fixed.

- **[P1] Model selection differs by arm.** Early stopping on validation sets that differ across
  arms (0 rare validation images in `none`; augmented copies of training images in `basic`/`gan`).
  Best epochs 6, 7 and 2 (1-based).
  Fix: fixed-epoch runs, or selection on common-class validation only (REPLICATION C1 already
  plans 250-epoch runs).

- **[P2] Multiplicity not stated.** 6 McNemar + 3 Wilcoxon tests, no correction, no statement.
  Fix: add one sentence that the tests are descriptive and uncorrected.

- **[P2] CI definition missing at point of use.** The headline "95% CI" is defined only in the
  Stage 4 method paragraph. Fix: footnote (text below).

- **[P2] Rare-subset tests are uninformative by design.** With n = 22 and discordant pairs of
  1 to 4, the McNemar p can never be small. The README already says "no comparison is
  separable"; keep the tests out of the main text or label them as such.

- **[P2] Seed sweep reports accuracy only, at two decimals, without κ/MCC.** Keep per-seed
  `test_predictions.csv` (about 180 KB each) so any metric can be recomputed.

### Ready-to-paste revision (proposals, not applied)

Footnote under the headline table:

> 95% CIs are percentile bootstraps over the 1,022 test images (1,000 resamples) for the single
> seed-42 head. They do not include training-seed variability (see *Five training seeds*), and
> they treat images as independent although the 1,000 common-class images come from 633
> patients, 566 of whom also appear in training. They are therefore likely too narrow.

Sentence for the seed section:

> Seeds vary head initialisation, batch order, the 10% validation split of the augmented pool
> and, for CSC, which 3,000 of 3,370 accepted GAN images are used. The test set, the Stage 1
> images, the CycleGANs and the QC are the same for every seed. Tests are paired by seed,
> two-sided, and not corrected for multiple comparisons.

### AUTHOR_INPUT_NEEDED
- Are any of the 22 rare test images from the same patient (or eye) as rare training images?
- Should seed 42 be pooled with seeds 1-5 (n = 6) or kept apart as the run of record?
- What is the pre-specified primary comparison for C1 (gan vs basic accuracy?) and, if
  "equivalent" is wanted, what margin?

### Reviewer-risk note
A statistical reviewer will still say that the rare-class question, the paper's actual claim,
rests on 22 fixed images and cannot be answered by more training seeds alone. Varying the rare
hold-out (repeated splits) is the only design change that addresses it, and it requires
retraining CycleGANs per split (REPLICATION C4 is the same problem).

---

## 4. Draft Data and Code Availability statement

Target journal: none specified. The statement is written for a Springer Nature-style report and
keeps data and code as separate sections. Placeholders in `[brackets]` need your input. Nothing
below is deposited yet.

### Dataset inventory and access routes

| # | Dataset | Supports | Current location | Route |
|---|---|---|---|---|
| D1 | Kermany OCT2017 v2 | common classes, all arms | Mendeley Data, doi:10.17632/rscbjbr9sj.2 | reused public source |
| D2 | Rare-disease OCT (Yoo et al.) | rare classes, Stage 1-2 | Mendeley Data, doi:10.17632/btv6yrdbmv.[version] | reused public source |
| D3 | Split manifest | every split | GitHub `data/manifest.csv` | public repository (code repo) |
| D4 | Result tables, QC records, heads, test predictions | every number in the README | GitHub `results/` | public repository (code repo) |
| D5 | Basic-augmented images (2,000 per class) | `basic`, `gan` | private Google Drive `MyDrive/octfsl` (`aug_basic.zip`, 0.52 GB) | **currently none**, needs deposit |
| D6 | Accepted CycleGAN images (9,889) and QC CSVs | `gan` | private Google Drive (`accepted_<class>.zip`, 0.66 GB); QC CSVs in GitHub | **currently none**, needs deposit |
| D7 | CycleGAN checkpoints | regeneration of D6 | private Google Drive (`ckpt/`, 6.9 GB) | **currently none**, deposit at least the final generators |
| D8 | Inception-v3 feature cache | Stage 3-4 without images | private Google Drive (inside `results_gan_run/`) | **currently none**, deposit |
| D9 | Per-seed predictions, seeds 1-5 | seed table | not retained, summary only in `results/seed_check.csv` | regenerate (`colab_setup.ipynb` Part 4) |

### Data Availability

> The OCT images analysed in this study are publicly available from two Mendeley Data
> repositories: the OCT2017 dataset of Kermany et al. (version 2,
> https://doi.org/10.17632/rscbjbr9sj.2) and the rare retinal disease OCT dataset released by
> Yoo et al. (https://doi.org/10.17632/btv6yrdbmv.[version]). These datasets are not
> redistributed by us and are subject to their original licences. The training, validation and
> test assignment of every image (`data/manifest.csv`), all result tables, quality-control
> records, trained classifier weights and test-set predictions are available in the code
> repository (see Code Availability). The images derived in this study (basic-augmented images,
> CycleGAN-generated images that passed quality control), the trained CycleGAN generators and the
> cached Inception-v3 features are deposited in [repository] at [DOI], under [licence, compatible
> with the licences of the source datasets]. Per-seed predictions for the training-seed
> sensitivity analysis were not retained; they can be regenerated with the published code and
> features.

### Code Availability

> The code used to build the splits, train the CycleGANs and classifiers, and compute every
> reported metric is available at https://github.com/Malnoad/Few-shot-rare-diseases under the MIT
> licence. The version used for this report is archived at [DOI of tagged release]. The
> CycleGAN implementation is adapted from the TensorFlow CycleGAN tutorial
> (https://www.tensorflow.org/tutorials/generative/cyclegan). Experiments ran on Google Colab
> with Python 3.13 and TensorFlow 2.20.0 (as observed in October 2026); exact package versions
> are listed in [lock file].

### Repository and citation actions
- Create one generalist-repository record (for example Zenodo, chosen because no OCT-specific
  repository was identified; unverified whether a better domain repository exists) holding D5,
  D6, D7 (final generator weights only, to limit size) and D8. Check the repository's size limits
  before uploading.
- Tag a release of the GitHub repository and archive it to get a DOI for the exact code version.
- Pin versions: `requirements.txt` has no version pins apart from `albumentations>=2.0` and
  `tensorflow>=2.16`. Export a lock file from the Colab run of record.
- Cite D1 and D2 as datasets in the reference list (author, year, title, repository, version,
  DOI), not only as URLs.
- Keep per-seed `test_predictions.csv` in future seed sweeps (closes D9).

### Missing information / risk flags
- **Licences of D1 and D2: unverified.** I did not open the licence fields on the Mendeley
  pages. Redistributing D5 (direct derivatives of D2 images) is allowed only if D2's licence
  permits it. Per the paper, some rare images came from web searches *(PMC, unverified
  verbatim)*, which may carry third-party rights. If redistribution is not allowed, deposit D7
  and D8 and state that D5/D6 can be regenerated deterministically from D2 with the code.
- **D2 version suffix:** unknown (README asks the reader to append it).
- **Google Drive is not an acceptable availability route:** it is private, has no persistent
  identifier, and part of it (`results_latest/`) is owned by a second account (`COLAB.md`).
- **"Available on request" was avoided on purpose:** there is no legal or ethical restriction
  that would justify it for derived images.
- The GitHub API reports the repository licence as "Other" although `LICENSE` is the MIT text;
  check that GitHub detects it before archiving.

中文核对 (not needed; the request is in English.)

---

## 5. Proposals (not applied)

Grouped by cost. "Training-free" items fit the laptop track.

**Text only**
- **P1** Align `docs/research/REPLICATION.md` ("Seeds 1-5 ... (gan ties basic)") and
  `docs/research/CONTEXT.md` ("both beat none") with the corrected README wording.
- **P2** Lead the Results with the seed summary; relabel "Headline metrics (seed 42)" as "Run of
  record (seed 42)" (R1-M3, R3-M1).
- **P3** Add the CI footnote and the seed-scope sentence from Section 3.
- **P4** Add a Results bullet on train/test patient overlap and image clustering (R3-M2).
- **P5** Add a "Selection and stopping" row to the Differences table: best epochs 6/7/2,
  validation sets differ by arm, QC uses the classifier's feature space (R2-M2, R2-M3).
- **P6** Fix the `stage0_manifest.py:7` docstring ("independent patients"), the same error as F3.
- **P7** Move RCI out of the paper-comparison table (R3-m1); say the kNN gap is in DME, not the
  rare classes (R2-m2).
- **P8** Add the Data and Code Availability sections from Section 4 to the README once the
  placeholders are filled.

**Training-free analysis**
- **P9** Patient-cluster bootstrap for the headline CIs and the paired differences; accuracy
  on the 86 test images from unseen patients (descriptive only).
- **P10** L7 first step: list the 22 rare test images with predicted class and probability for
  each arm, from the committed `test_predictions.csv` files.

**Colab (needs training)**
- **P11** In C1, save every seed's `test_predictions.csv`, report κ/MCC per seed, and add one
  fixed-epoch setting shared by all arms. Pre-specify the gan vs basic comparison.
- **P12** A `gan` arm without QC (or QC in another feature space) to separate generator quality
  from selection.
- **P13** Deposit D5-D8 with a DOI (Section 4).

---

## 6. Sources and how numbers were checked

- All README numbers were recomputed with `D:/Few-shot/.venv/Scripts/python.exe` (pandas,
  scipy) from `results/stage4/*.csv`, `results/seed_check.csv`, `results/analysis/*.csv`,
  `results/stage3_*/history.csv` and `data_counts.csv`, and `data/manifest.csv`. See the
  forensic audit table in Section 2. No result file was modified.
- Paper values: PMC7829497 (open access), read through a summarising web fetch; Table 3 and
  Table 4 values cross-checked against `stage4_evaluate.py` PAPER and the README. The local PDF
  could not be read (no PDF tool installed, and adding one was out of scope).
- Code behaviour: `stage0_manifest.py`, `stage2_cyclegan.py` (`prep`, `qc`),
  `stage3_classifier.py`, `stage4_evaluate.py`, `analysis.py`, `colab_setup.ipynb` Part 4.
- Repository facts: `gh repo view` (public; licence detected as "Other"), `LICENSE` (MIT text),
  `COLAB.md` (Drive contents and sizes).
