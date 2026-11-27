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

## A3. Final scope and simplifications (13 November 2026, before the input lock)

**Decision.** Lam's instructions, 13 November, verbatim: "finalize the protocols what we do
should be as big as what we solve so we don't reach bottleneck or unnecessarily things then I
would verify the final or bring up things that I find may not clear, follow precedents, previous
study, simplified ideas", and later the same day: "finalize the protocols, do the best as you
can to bridge the gap between fields gap, previous study and our project, I will review again
and re-ask if I can't clearly understand". This is that final text. It was written by the agent
(Claude Fable 5.2) from two independent reviews made the same day (GPT-6.1-Sol, execution path;
Claude Opus 5.5, scope against claims and precedent), each checked against the code. It binds
from the input lock, like A1 and A2. Until the lock Lam can change any rule here by a dated
edit; pushing the code to GitHub for the lock session needs Lam's word, and that word covers
this text. No score exists. A3 adds no experiment, arm, endpoint or dataset.

### What the study reports

| Level | Result | Where |
|---|---|---|
| Primary | Five-way development macro recall per K = 1, 2, 4, 8 and their mean, all four arms, paired, beside the size-only reference; the nomination | Main text |
| Secondary | Nine-way false-rare rate by source with intervals (Kermany patient-disjoint, Kermany all, OCTID, OCTDL); legacy 22 with exact counts per class and arm | Main text |
| Exploratory | OCTID two-way CSC-versus-MH test (A1, A2 A-1) | Main text, last |
| Diagnostics | Geometry table, exclusion ledgers, pilot timing and repeat checks, one compute table | Main text or supplement |
| Supplement only | Logistic probe results; five-way output shares on external rare images (prototype and logistic); nine-way per-class recall with intervals; full confusion counts | Supplement files |

Everything in the table is still computed and released, favourable or not (A1 rule 6). "Supplement
only" fixes where a result is shown, not whether it exists.

**Not done in v1:** fine-tuning, adapters, augmentation, GAN images or further checkpoints
(v1.0 section 1); a source probe, an OCTID icon mask, a LayerNorm sensitivity run (A2); a
primary K, p-values, intervals on K-shot results (v1.0 section 7, A2); external claims for
MacTel, Stargardt or RP, or any claim of clinical label equivalence (A1).

### Metrics and baselines

This section names what v1.0 sections 6-8 and A2 already define; it adds no metric.

**Main metric.** Five-way macro recall: for each class, correct queries divided by that class's
queries, then the unweighted mean over the five classes (v1.0 section 7). It is the same
quantity MIRAGE reports as balanced accuracy, and the mean of the per-class true positive
rates Yoo et al. tabulate. Chance is 0.20. It is reported per K and as the mean over
K = 1, 2, 4, 8 (A2 B-1), from cosine prototypes on the development folds.

**Main comparison.** The paired difference in macro recall between an FM and the Inception
reference on identical support draws and queries: mean, SD and 5th-95th percentiles over the
draws (v1.0 section 7). The nominee's difference is the headline; every other pair is shown.

**Metric set.**

| Task | Metrics | Unit of spread |
|---|---|---|
| Five-way K-shot (development, legacy 22) | Macro recall (main); accuracy; per-class recall; confusion counts; exact counts on the legacy 22 | Support draws |
| Nine-way source audit | False-rare rate (main); NORMAL recall; nine-way per-class recall; common-class confusions | 95% intervals (A2 B-3) |
| OCTID two-way test | Two-class macro recall (chance 0.50); per-class recall; confusion counts | Support draws |
| Cost | Median warm seconds per image from the locked pilot | None |

**Baselines.** Every arm result is read against these, all on the same episodes:

1. **Reference model: frozen ImageNet Inception-v3.** It is the backbone of Yoo et al. and of
   our replication, so it stands for the previous study. Same supports, same classifier, never
   nominated.
2. **Chance:** 0.20 five-way, 0.50 two-way.
3. **Size-only reference (A2 N-1):** the score reachable from image width and height alone. An
   arm at or below it has not shown that it reads the retina.
4. **Generalist against specialist, inside the roster:** DINOv2 (natural images) beside
   RETFound and MIRAGE (retinal images). This is the comparison set of the MIRAGE study
   itself ("we compared MIRAGE to three other selected SOTA models, all based on ViT:
   supervised ImageNet pretraining (SL-IN), DINOv2, and RETFound"), with our Inception-v3 in
   the supervised-ImageNet place. It is a description, not a causal test (v1.0 section 1).

The published numbers of Yoo et al. and of our replication are context only: they come from a
nine-way classifier trained on every rare image, with augmentation, not from K supports.

**Not used, and why.** AUROC and average precision (RETFound, MIRAGE): a prototype gives a
class decision, and with four or five legacy images per class a ranking metric is unstable;
the cosine scores are saved, so it can be computed later as a labelled post-hoc analysis.
Cohen's kappa, MCC and RCI (Yoo et al.): defined for the nine-way task, and the paper's RCI
values could not be reproduced (`L8_RCI_AUDIT.md`). p-values: v1.0 section 7.

### Simplifications

**S-1, external duplicate screen without manual review (replaces, for external images only, the
review in v1.0 section 4 step 3).** Every external pair the dHash rule flags is recorded as
unresolved and therefore linked, as section 4 step 3 already treats unresolved pairs. An
external image linked to a fitting row or a legacy query is removed; linked external images
share a dependence group. A linked group of external images carrying more than one label is
removed entirely and listed in the exclusion ledger with the reason `conflicting_labels`.
Nobody inspects pairs, so no judgement is made after the lock. Cost: a false flag removes or
groups a few queries; the ledger shows every one. The review of the 146 benchmark images is
unchanged (it flagged no pair).

**S-2, tie-break timing (replaces "If runtime/hardware changes, retime all candidates" in
v1.0 section 7).** The nomination tie-break uses the pilot timings stored in the input lock and
nothing else. There is no retiming after the lock.

**S-3, one retry of a failed repeat check (adds to A2 A-6).** If a pilot's repeat check fails,
the whole pilot sequence is run once more in a fresh timing session under unchanged code,
assets and environment, and both attempts are kept. An arm that fails again is reported
unavailable with the measured values.

**S-4, fallback closed.** RETFound passed access, terms and loading in the first checks (11
November, `first-check-20261111T175615Z`). BioMedCLIP is withdrawn as a fallback. If RETFound
fails a final pilot it is reported unavailable (v1.0 section 5); nomination still needs two
complete FMs.

**S-5, timestamp fallback (adds to A2 A-8).** If the public deposit cannot be made before the
first `develop`, Lam emails the same hashes to the mentor and the team before the first
`develop` and makes the deposit afterwards. The report states which was done and when.

**S-6, logistic probe.** It stays as in v1.0 and A2. Measured on 13 November on synthetic
features of the real dimensions, one fit takes 0.01-0.04 s on the laptop, so the roughly 2,000
planned fits are minutes of CPU (convergence on real features is unverified). It is omitted,
with the dated reason v1.0 section 9 requires, only if it is unfinished at the 27 November
freeze.

### Clarifications of A1 and A2

- **A2 B-3 replaces A1 rule 5**, it does not only complete it: intervals are given for the
  nine-way outputs alone. External rare results (the OCTID two-way test, output shares) carry
  counts, denominators and the support-draw spread, without intervals.
- **A2 B-1 replaces the "at each K" headline of v1.0 section 7.** The nominee-minus-Inception
  paired difference is shown at every locked K; where one number is needed it is the K mean.
- **A2 A-4** concerns a source that fails access or its audit before the lock: it is declared
  unavailable in the gate and is absent from the external candidates.
- **Size-only reference in the OCTID two-way test.** If every OCTID CSC and MH query has the
  same width and height, the reference assigns all of them to one class and scores 0.50 by
  construction. The report then says so instead of presenting it as a finding.

### Precedent

Sources are those opened for `FM_RULE_TRACKER.md` ("Why earlier studies did not fix these");
the metric statements were reread on 13 November in the Europe PMC full texts: MIRAGE
(PMC12462498) "including the AUROC, average precision (AP), and balanced accuracy (BAcc)
values" under linear probing with "all the parameters of the model are frozen except for a
final linear layer"; RETFound (PMC10550819) "Model performance was reported using the area
under the receiver operating curve (AUROC)", compared "against three pretrained comparison
models"; Yoo et al. (PMC7829497) "The main focus of this study was the accuracy of the
classification model", with per-class true positive rates and a 22-image rare test set.

What each side contributes, in one table:

| Question | Earlier studies | Gap, in the studies read | This study |
|---|---|---|---|
| Which encoders | MIRAGE compares SL-ImageNet, DINOv2, RETFound and MIRAGE with linear probes tuned on each dataset; its benchmark includes Kermany, OCTID and OCTDL | No comparison at 1-16 labelled images per class | The same four kinds of encoder at K = 1-16, frozen, with identical supports |
| Rare OCT diseases | Yoo et al. train a nine-way Inception-v3 head with CycleGAN augmentation and test on 22 rare images; they also try a prototypical network, a Siamese network and KNN | Every rare training image is used; no K-shot budget; one test of 22 images | Strict K-shot supports, grouped folds on 124 images, and the same 22 images as a secondary, disclosed-as-inspected test |
| Metric | MIRAGE: AUROC, AP, balanced accuracy; RETFound: AUROC, AUPR; Yoo et al.: accuracy and per-class true positive rate | No shared few-shot metric for these classes | Macro recall, which equals balanced accuracy and averages Yoo's per-class rates |
| Other sources | RETFound and a TVST 2026 study test trained models on external sets | No test of how often NORMAL scans from another source are called rare; acquisition cues are not measured | False-rare rate per source; size-only reference and geometry table; the OCTID two-way test where size is constant |
| Pre-specification | None of the four papers registers a plan | Choices can follow results | Hashed lock and nomination before any legacy or external score |

Few-shot studies report prototype results at several K with no primary K (SimpleShot). RETFound
and MIRAGE evaluate frozen encoders with a linear probe, which is why the logistic probe stays.
None of the studies read reviews duplicates by hand, retimes models, or registers its plan; the
steps kept here beyond that precedent (image screen, nomination, hashed lock, timestamp) are
the ones the pre-specification claim rests on.
