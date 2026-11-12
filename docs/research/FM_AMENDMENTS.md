# Dated amendments to protocol v1.0

`FM_PROTOCOL.md` stays byte-identical (SHA256 `4e8df8d9…c5df`, pinned in `fm_prepare.py`).
Amendments are recorded here, dated, before the input lock and before any score they affect.
Each one names what it replaces; everything else in v1.0 stands.

## A1. External rare labels (12 November 2026, before the input lock)

**Lam's decision, verbatim:** "We adopt published dataset labels for exploratory external
evaluation only. Results measure agreement with those labels, not clinically validated disease
transfer. Cohort eligibility is fixed before inference, and all planned results will be
reported."

**Replaces**, for the cohorts below only: protocol section 8, "CSR-to-CSC needs clinical
label-equivalence confirmation", and the fallback "If rare mapping fails, retain NORMAL FRR and
omit that disease-transfer claim". No clinical equivalence is claimed, so no confirmation is
required; the transfer claim is replaced by a narrower agreement claim.

**Eligible cohorts, fixed now:**

| Cohort | Images | Basis | Use |
|---|---|---|---|
| OCTDL `disease=VID` and `subcategory=MH` | 39 (28 patients) | OCTDL's MH definition ("a complete defect in the inner layers of the retina that extends to the RPE") matches the IVTS full-thickness MH definition | Query only; grouped by OCTDL patient ID |
| OCTID CSR | 102 | OCTID's paper describes the class under "Central Serous Chorioretinopathy (CSC)"; MeSH D056833 lists "Central Serous Retinopathy" as an entry term | Query only |
| OCTID MH | 102 | OCTID's own broad definition ("small hypoflective breaks to full-thickness intra-retinal spaces") | Query only; a separate cohort, never pooled with OCTDL MH, never called full-thickness MH |
| OCTID NORMAL, OCTDL NO | 206; 332 (110 patients) | Unchanged from v1.0 | Nine-way NORMAL FRR |

Excluded: OCTDL VID `subcategory` LH and VMT (their `condition` column also reads "MH"),
OCTID AMD and DR, OCTDL AMD, DME, ERM, RAO and RVO.

**Rules:**
1. External images are evaluation queries only; none enters fitting, supports or nomination.
   MacTel, Stargardt and RP receive no external claim.
2. External rare results are exploratory and descriptive. Prior studies (TVST 2026, MIRAGE
   2025, EyeDiff 2026) are cited as precedent for evaluating published categories, not as
   evidence of label equivalence. The benchmark's own CSC and MH definitions remain unverified.
3. OCTID two-way CSC-versus-MH test: benchmark-only CSC and MH supports, cosine prototypes only
   (no logistic probe). Report per-class recall, macro recall, confusion counts and denominators
   for every locked K and arm, favourable or not. OCTDL MH and the five-way head's outputs on
   external rare images are reported as output shares with denominators.
4. No claim that acquisition shortcuts are removed. Record image geometry and padding by class
   and source (benchmark crop width differs by class, e.g. CSC about 414 px, MH about 327 px),
   and report the source probe as a diagnostic, not a gate.
5. Intervals: OCTDL patient-cluster intervals (5,000 draws, seed 0, as v1.0); OCTID intervals are
   image-level and labelled conditional, because OCTID gives no patient IDs.
6. Eligibility is fixed before inference. Later criticism of a mapping (for example from the
   mentor) is reported as a limitation; scored results are never withdrawn or replaced.
7. Required wording: "We evaluated agreement with OCTID's and OCTDL's published CSC/CSR and MH
   labels using benchmark-only K-shot supports. Clinical label equivalence, patient
   independence and freedom from acquisition shortcuts were not established."

**Provenance:** draft and evidence in `FM_LABEL_RULE_PROPOSAL.md` and `FM_EXTERNAL_MAPPING.md`;
independent rebuttals by GPT-6.1-Sol (Codex) and Gemini 4 Argon High, 12 November 2026, both
"accept with changes"; revised rule above approved by Lam the same day.

## A2. Pre-lock rules (13 November 2026, before the input lock)

**Decision:** on 13 November Lam answered the 12 November triage with "You can choose", then
commissioned an independent review of every open rule (Claude Fable 5.2, report in
`fm_independent_review/FM_INDEPENDENT_REVIEW.md`) and decided: "Follow every Fable
recommendations". The rules below are those recommendations. They replace the parts of v1.0
and A1 they name; everything else stands. After the input lock this file never changes; later
notes go in `FM_DEVIATIONS.md`, which is not hashed.

**Disclosure.** Before this amendment the review computed a size-only classifier on the 124
development images and on the 22 legacy test images, from file width and height and labels
only (no encoder, no pixel). It found the size shortcut below (N-1).

### Inputs and features

**A-11, input transform.** Each encoder uses the input transform of its own published
frozen-probe code path at the pinned commit. MIRAGE: resize 512x512 bilinear, grayscale,
divide by 255, then (x - min) / (max - min) over the image (`fm_cls_config.py` `MIRAGEFM` at
`ee5eae8d`; MIRAGE paper Methods: "For MIRAGE, min-max normalization was applied to the input
images"). An image with max = min stops the arm. RETFound (`util/datasets.py` eval transform at
`ae9a9ecf`), DINOv2, Inception and BioMedCLIP are unchanged.

**A-7, final normalization.** Both frozen-feature paths keep the default LayerNorm (weight ones,
bias zeros, eps 1e-6), which the checkpoints do not contain. Upstream code read 13 November:
RETFound `models_vit.py` creates a new `fc_norm` under `global_pool=True` and deletes the trained
`norm`; `latent_feature.ipynb` loads with `strict=False` and calls `forward_features`; linear
probing (`main_finetune.py --adaptation lp`) trains only parameters named `head`. MIRAGE
`mirage_wrapper.py` `MIRAGEClsGlobal` (default `--pool global`) adds a new LayerNorm before the
patch-token mean, and its linear probe (`fm_cls_config.py`) freezes it. With A-11, MIRAGE now
follows that frozen-probe path end to end. No norm sensitivity analysis is run.

**A-6, numerical repeat (adds to the approved R3 rule).** In each timing pilot the outputs of the
first and last timed passes are compared for all 100 timing images: every image needs a cosine
similarity of at least 0.9999 and a largest element-wise difference of at most 1e-4 of its
largest absolute output. R3 stays as approved: five fixed synthetic images, encoded twice in
the same process, largest element-wise difference at most 1e-4 of the largest absolute output.
A failed check stops the pilot; the arm is then reported unavailable with the measured values.
Neither check compares with upstream outputs or catches a stable but wrong pipeline.

### Endpoints and reporting

**N-1, size-only reference.** For every locked development and reporting episode, each query is
assigned to the class whose support images have the nearest mean (width, height) in pixels of
the original file (squared Euclidean distance, ties to the fixed class order). No pixel and no
encoder is used. Its macro recall is reported per K beside every arm, and for the OCTID two-way
test. It is a diagnostic of the benchmark: it is never nominated and is not an arm.

**B-1, no primary K.** No K is primary. All locked K are shown together. Where one number is
needed it is the mean over K = 1, 2, 4, 8, the average the nomination already uses.

**A-1, OCTID two-way CSC-versus-MH test (completes A1 rule 3).** It reuses the locked reporting
episodes, keeping only their CSC and MH supports; all four arms; the reporting K set; all 100
draws; cosine prototypes only. The two-way prediction is the larger of the CSC and MH cosine
scores of the locked five-way reporting episode (ties to CSC), which equals refitting with CSC
and MH supports only. Headline: the nominee's two-class macro recall per K; chance is 0.50; a
collapse to one class is reported as such. Per-class recall and confusion counts with
denominators are reported for every arm and K. If no nominee exists, all arms are reported
without a headline. The logistic probe is never used for this test; for the five-way external
output shares it stays as in v1.0 (first 25 draws per K).

**A-5, Kermany NORMAL false-rare rate.** Headline Kermany FRR: the NORMAL test images whose
manifest `pid` (the second field of the file name) appears in no common fitting row of any class
of the pinned nine-way roster; on the current roster (`72f90c00…`) these are 169 images from 154
patients. Secondary: all 250. Comparisons between Kermany and external FRR use the
patient-disjoint images. RETFound's pretraining exposure to Kermany (B-5) is not addressed by
this.

**B-3, intervals (completes A1 rule 5).** Intervals are given only for the deterministic nine-way
outputs (false-rare rate and nine-way per-class recall): 95% percentile intervals from 5,000
cluster resamples with seed 0 (the v1.0 section 7 and L9 convention) for Kermany (cluster =
`pid`) and OCTDL (cluster = the locked dependence group, which joins each patient's images);
exact Clopper-Pearson intervals for OCTID, labelled image-level and conditional. K-shot results
carry the support-draw mean, SD and 5th-95th percentiles only.

**A1 rule 4 changes.** The source probe is withdrawn before any score. Geometry is reported as one
table by source, class and role: count, and median and range of width, height, retained field,
border share and near-white share. Border share: on the decoded 8-bit grayscale original, the
share of area in complete border rows and columns, peeled from each edge, whose grey range is at
most 8. Near-white share: the share of pixels at 250 or above. Both are descriptive and gate
nothing.

**Reporting order.** Development K-shot curves for all arms with the size-only reference; then
false-rare rate by source; then the legacy 22 with exact counts per class and arm, labelled
historically inspected; then the OCTID two-way test, still exploratory under A1, named as the
only test where image size is constant.

### External cohorts and failures

**N-3, cohorts in code.** The A1 cohorts are constants in the code. No command-line option adds or
removes a cohort.

**A-4, failed external NORMAL access (replaces section 8's "leaves a rare-only pilot").** Applied
per source: report the NORMAL FRR for each of OCTID and OCTDL that passes its access and source
audit. The study becomes a rare-only pilot only if both fail.

**A-3, OCTID icon.** No crop or mask. The orientation icon is disclosed as a known acquisition cue.

### Lock

**A-9, amendments in the lock.** `FM_AMENDMENTS.md` is one of the hashed code files, so pilots,
lock, extraction and evaluation all refuse a changed copy. It must be final before the final
pilots.

**A-8, public timestamp.** After the lock and before the first `develop`, a small text file with
the SHA256 of the lock ZIP, of the protocol and of this file, plus the code commit, is deposited
publicly (OSF or Zenodo; Lam chooses and holds the account).

**A-10, lock storage.** The lock ZIP is not committed before reporting. It is kept on Drive and on
the laptop with its hash deposited (A-8). The repository stays public until the last Colab
extraction, including the external one, has finished, because the notebook clones without
credentials.
