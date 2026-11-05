# Frozen-encoder rare-OCT protocol v1.0

Status: design finalized after final-round rebuttal, 5 November 2026.
Version: 1.0. Lam is the research author. Implementation and empirical feasibility
remain subject to the explicit gates below; no new FM results exist.
This consolidates the 3 November reviewed FM proposal and 5 November calendar.
It supersedes conflicting experimental settings in those planning documents.
This is a design, not a claim of implementation, data acquisition or new results.
Finalization does not itself execute training, install dependencies, download
weights/data, register a study, contact clinicians, commit, push or publish.

## 1. Question and scope

How do frozen checkpoints compare on five-way rare-OCT classification at matched
support-label budgets, and how vulnerable are their predictions to source shift?

Primary task: CSC/MH/MacTel/Stargardt/RP discrimination. Required companion:
matched nine-way real-only prototypes and external NORMAL false-rare rate (FRR).
This is a descriptive checkpoint-pipeline comparison. Architecture, pretraining,
capacity and native preprocessing differ; no causal effect of domain-specific
pretraining or identified disease-versus-source mechanism is claimed.

No new GAN training/generation, C1–C3 sweeps, adapters, fine-tuning, text-prompt
search or extra checkpoint search enters this sprint. Existing experiments and
any partial outputs are preserved separately.

## 2. Fixed decisions

| Item | Adopted design |
|---|---|
| Framing | One bounded FM adaptation/validity study; Yoo replication provides historical motivation |
| Models | RETFound Nature OCT, DINOv2-Base, MIRAGE-Base, plus matched frozen Inception-v3 reference |
| RETFound fallback | BioMedCLIP only if RETFound access/loading/terms fails, fixed before labelled scores |
| Adaptation | Original frozen checkpoint, real-only supports; cosine prototypes primary; fixed L2 logistic probe secondary |
| Development information | 124 inspected rare labels disclosed separately from each K-shot support budget |
| Common fitting cap | 1,000/class, seed 42, identical rows for every encoder |
| Development | Five fixed grouped folds, K={1,2,4,8}, 100 paired nested draws |
| Secondary development | K=4, first 25 matching draws/fold |
| Reporting | K={1,2,4,8,16} where feasible, 100 prototype draws; first 25 matching logistic draws/K |
| External | Separate OCTID/OCTDL NORMAL inventories; rare transfer conditional on verified mappings |
| Compute | Colab Pro, one GPU session at a time; frozen extraction plus CPU small classifiers |

These choices consolidate the recommended design under Lam's requested scope;
they do not assert that Lam individually selected every prior option.
The 1,000 common cap replaces the older 5,000/class proposal. No all-data ceiling
is included in this version. Secondary probes are the first experiments dropped
if time is short; the dated reduction is disclosed. No outcome-driven additions.

## 3. Data and label budgets

Canonical data/manifest.csv remains unchanged: 84,630 rows, comprising 74,082 train,
9,526 validation and 1,022 test images. Rare development counts in fixed class
order are 27/26/25/19/27 =124; rare legacy test counts are 5/5/4/4/4 =22.
No real rare validation rows exist.

Derived rosters reference canonical paths without rewriting splits.
The unique inspected target-development budget is 124 labels; each five-way
K-shot classifier uses 5K supports from that inventory, not 124+5K or folds times
draws times 5K. Disclose 22 historically inspected evaluation labels, common
fitting labels and upstream pretraining exposure separately. The legacy test
is not newly blind. Pre-score means before new-protocol scores: previous audits
already scored all 146 rare images with Inception. Those historical analyses
informed this design; they are not new validation evidence.

Existing basic/GAN pools are excluded from strict K-shot/CV: their source
images, generators and QC used the broader rare training inventory.
Subsampling accepted images cannot undo that exposure. This sprint uses real
supports only, without stochastic augmentation.

## 4. Grouping and feasibility before model scores

Keep two concepts separate: image-duplicate components and verified patient
groups. Patient IDs are namespaced by dataset. Similar filenames or a perceptual
hash do not establish patients; unknown rare patients remain unknown.

Fixed image screen for implementation:
1. For the 146 rare benchmark rows, link equal file hashes or equal decoded RGB
   pixels, including dimensions. Alpha is dropped for the RGB fingerprint; save
   original content hashes separately.
2. Flag near-duplicate candidates using a 64-bit horizontal difference hash:
   Pillow RGB-to-L conversion, resize to 9x8 with Image.Resampling.BOX, compare
   left>right for each adjacent pair, row-major bits; Hamming distance <=4.
   Pin Pillow/decoder version. BOX area averaging replaces the candidate's
   ambiguous bilinear reduction before any screening or new-protocol scores.
   This screen is not a validated duplicate or patient detector.
3. Review flagged raw-image pairs without new encoder scores. Confirmed or
   unresolved possible duplicates share connected-component groups; save the
   decisions, flagged-pair counts per block and component sizes. Conflicting
   disease labels on possible duplicate images block that component until
   resolved. A non-expert duplicate reviewer is not diagnosing disease.
4. Resolve rare-fitting versus rare-legacy-query image links by excluding the
   rare fitting component before fitting lock. Canonical test rows never move.
   Common Kermany fitting rows are not removed by this rare grouping rule:
   retain the exact seed-42 common cap roster. Common patient IDs are used for
   overlap accounting and cluster intervals, not a patient-disjoint replacement
   benchmark. Any known common image dependence is disclosed as a limitation.
5. Freeze external source/release/eligibility definitions at the data gate.
   Screen external queries against every locked fitting row, legacy queries,
   and each other, using the same image screen and any verified patient links.
   Remove fitting/legacy-linked external query components and ledger within-query
   dependence; never retrospectively mutate locked fitting rosters. Freeze final
   external query IDs before inference. Report the inventory actually screened:
   unused legacy images and private pretraining overlap remain unknown. Missing
   external data yields the reduced scope in section 8.

For rare development, union image components with verified rare patient groups,
if any. If a rare patient group spans multiple disease labels, stop the selector
and amend the fold assignment before scores; a shared patient is not itself a
contradictory image label. Within-query dependencies remain visible in reporting.

Use normalized POSIX paths in case-sensitive lexicographic order. Each group ID
is its minimum member path; sort member paths and class-specific group IDs.
group_index is the zero-based position in that sorted eligible class inventory,
not in the draw's permuted order. For each class, permute groups with NumPy PCG64
SeedSequence([42,class_index,5101]), then round-robin fold IDs 0..4.
Publish per-class/fold group and image counts and the complete exclusion ledger.

class_index is 0..4 in CSC/MH/MacTel/Stargardt/RP order; use
Generator.permutation on the sorted group-ID array. Every required fold must
contain all five query classes and eight eligible fitting groups/class.
Under round-robin allocation the minimum fitting-group count is G-ceil(G/5),
so each class requires at least 10 eligible groups for development.
Failure stops the entire selector before scores. No
automatic reduced-K selector, replacement sampling, favourable-fold omission or
threshold relaxation; a dated amendment must precede scores. Reporting K=16 is
omitted for every encoder if any class lacks 16 fitting groups, with infeasibility
reported and the prespecified smaller reporting K values retained.

## 5. Models, preprocessing and features

Fixed public identities retained from the reviewed plan:
- YukunZhou/RETFound_mae_natureOCT.
- facebook/dinov2-base.
- j-morano/MIRAGE-Base, OCT-only inference.
- Conditional fallback: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224,
  image features only, with no text-scoring arm.
- Reference: ImageNet Inception-v3, historical 256 then299 resizing and GAP.

The pre-score gate pins usable terms, weight revision/hash, input/channel handling,
deterministic native transforms, normalization, pooling, dtype and inference mode.
Do not replace a checkpoint using measured accuracy. If MIRAGE/DINOv2 or the fixed
RETFound fallback fails, report the unavailable arm and reduced roster; no
open-ended search. No availability failure is counted as a poor score.
An arm unavailable at input lock stays unavailable in v1, even if access arrives
later; it cannot enter nomination after new scores. A future separate version
may revisit access. Retrying an already locked technical operation is different
and must preserve its exact identity.

New FM extraction needs a separate Colab PyTorch loader/requirements, a later
implementation deliverable. Do not port models into the current TensorFlow stage
scripts or force all FMs through Inception preprocessing. No fitted PCA/scaler,
query adaptation or query-updated normalization.

Cache identity includes ordered image IDs/content hashes, weight revision/hash,
preprocessing/pooling, environment, dimension/dtype and code revision.
Record resize/crop geometry and retained field-of-view fraction by image/source;
use 1 for full-field transforms, excluding padding from retained-area accounting.
These describe checkpoint pipelines; do not tune cropping using outcomes.
All matched encoders, including the new Inception reference, use FP32 inference
and store raw float32 embeddings in fresh versioned caches. CPU normalization,
centroids and small-classifier arithmetic use float64. Reject nonfinite/zero-norm
features and centroids. Do not overwrite historical caches.

Legacy Inception arrays are float16-quantized and their validator checks paths,
not image/weight/preprocessing hashes. They and the saved heads are retained for
historical replay/context, not substituted into the matched FP32 comparison.
The new Inception extraction can reuse the verified image-loading/backbone code,
but must bypass the legacy float16-saving cache writer. The reference explicitly
uses stage3_classifier.load_image and backbone (historical Keras Inception weights),
in new cache groups/directories with sidecar identity. A PyTorch Inception is not
an interchangeable reference. External images always need fresh reference features.

## 6. Paired episodes and classifiers

Development: for each fold/class/draw, permute fitting group IDs with PCG64
SeedSequence([42,fold_index,class_index,draw_index,6101]), draws 0..99.
Choose one image from its sorted member list by a uniformly drawn member index,
using a separate PCG64 stream
SeedSequence([42,fold_index,class_index,draw_index,group_index,6102]).
group_index is the sorted-group rank defined in section 4.
Within-group image choices and group order remain fixed across nested K prefixes
and all encoders/classifiers. Queries are all eligible held-out fold rows.

Reporting: all eligible development groups, namespace 7101/7102 instead of 6101/6102,
fold_index=0, 100 fixed paired nested support rosters; queries are the legacy 22 or
separate external subsets. Save IDs, not seeds alone.

Primary classifier: unit-normalize embeddings, average supports per class,
unit-normalize each centroid, predict maximum cosine dot product.
Fixed rare-class order resolves exact ties. Zero/nonfinite centroids fail.

Do not substitute stage4_evaluate.baselines: its historical prototype uses
Euclidean distance on raw-feature means. analysis.centroid_correct also averages
raw supports before centroid normalization. Neither implements this protocol's
normalize-supports-first cosine rule or its full numerical guards.

Secondary classifier: five-way logistic regression on the same unit embeddings;
mean cross-entropy +0.01*||W||_F^2/2, unpenalized intercept, zero initialization.
Optimize this exact mean-loss objective and analytic gradient using
scipy.optimize.minimize(method='L-BFGS-B'), with no bounds, maxiter=1000,
maxfun=20000, maxls=20, maxcor=10, gtol=1e-6 and ftol=0. A fit is converged only
with finite parameters/objective and maximum absolute gradient <=1e-6; save the
actual termination reason. Do not replace this with a fixed sklearn C=100:
support count would change the effective penalty. No query-based tuning. Development K=4 first 25 draws/fold; reporting first 25
matching draws/K. Report convergence failures and missing pairs without changing
regularization or replacing draws after inspecting queries. This is not the
existing Adam/fixed250 softmax trainer and cannot reuse its fitted weights.

## 7. Endpoints, nomination and uncertainty

Primary: five-way macro recall = mean_c(correct_c/n_c). Secondary: ordinary
accuracy, per-class recalls and full confusion counts. Missing query classes
invalidate required development episodes.

Nomination uses prototypes only:
S_m = mean_(K,r) [mean_c(sum_f correct_m,K,r,f,c / sum_f n_query_f,c)],
where K={1,2,4,8}, r=1..100. Pool out-of-fold correctness within each class before
averaging classes. Publish every encoder's K-specific scores and denominators.
Inception is a reference, not an eligible FM winner. An FM receives a nomination
score only after every required prototype fold/K/draw is complete and valid;
partial averages cannot enter the score. Report failed/incomplete arms and all
available scores. Nominate only among at least two complete FMs; with fewer,
report a feasibility case without a comparative winner or new model search.
Uncentred cosine similarity is sensitive to checkpoint feature geometry; no
post-hoc centering or alternative classifier is introduced to improve a model.

S_max is the maximum over complete eligible FMs only, excluding Inception.
Near-best FMs have S_max-S_m <=0.02 (two percentage points).
Within this set choose the lowest median warm extraction seconds/image: one fixed 100-image
development inventory, path order before scores, same hardware, batch=1, FP32,
five timed passes after one warm pass; take the median of the five per-pass mean
seconds/image. Time all FMs sequentially in one session in RETFound/DINOv2/MIRAGE
order (fallback in RETFound's slot), one model per process. If runtime/hardware
changes, retime all candidates before using cost to break a tie.
Preprocessing included, download/disk I/O
excluded, synchronization around GPU timing. If fewer than100 eligible images,
record repeated IDs for timing only. Final tie order RETFound/DINOv2/MIRAGE;
BioMedCLIP takes RETFound's slot if substituted.
This is an engineering allocation rule, not equivalence. Publish all models;
selected development performance is optimistic after selection.

The nominee becomes the pre-designated primary FM for legacy and external
summaries, abstract/headline rows and nominee-minus-Inception paired differences
at each K. All other models and their contrasts remain fully visible as secondary.
Never re-designate the nominee after legacy/external outcomes. If nomination is
incomplete, no primary FM is declared and reporting is explicitly exploratory.

Support-draw mean/SD and 5th/95th percentiles describe variability conditional on
these images, not confidence intervals for new patients. Paired comparisons use
shared completed episodes and show available/planned counts. Large-K rosters
can overlap heavily, so narrow draw spread is not population precision. No confirmatory
p-values, automated superiority/equivalence, episode SEM or pooled-binomial CV SE.

For deterministic nine-way outputs, descriptive 95% percentile intervals may use
5,000 patient-cluster draws, seed=0, only when verified IDs exist, matching
the existing L9 resampling convention. Otherwise identify
image/duplicate-group conditional intervals explicitly. Such intervals condition
on the fitted head, excluding development-selection uncertainty. Repeated
draws do not create patients; rare n=22 limits clinical conclusions.

## 8. Matched nine-way source audit

Call stage3_classifier.major_indices(common_train_labels,1000,42) on the full
73,958-row canonical common fitting frame, preserving the exact 4,000 selected
common rows. The returned indices are positions in that frame, not manifest row
numbers; map them back to manifest paths in order. The rare exclusion ledger
applies to rare fitting rows only; append all eligible real rare fitting rows. Freeze identical rows for every
encoder/reference. Fit exactly one deterministic nine-way head per encoder using
section 6's normalized-support centroid formula, not its K-shot episodes.
Class order: NORMAL/CNV/DME/DRUSEN/CSC/MH/MacTel/Stargardt/RP.
No optimizer seeds, fixed250 training, common-validation extraction or synthetics.

Before exclusions: 4,000 common fitting +1,000 common tests +146 rare = 5,146 images
per encoder, plus eligible external inputs. Actual counts follow the ledger.

FRR = true NORMAL scans predicted as any rare class / all true NORMAL scans.
Report Kermany NORMAL and each external source separately, alongside NORMAL
recall, common confusions, nine-way rare recall and denominators. Never use a
five-way rare-only head for FRR. Low FRR alone does not imply useful diagnosis.
Kermany common patient overlap remains a legacy limitation; do not claim
patient-independent common-class performance. Upstream image exposure and
checkpoint-specific feature geometry can affect FRR differences; no causal
source-shift or pretraining-effect interpretation follows from those gaps.

Before external inference, freeze releases/hashes, terms, patient IDs, duplicates,
eligibility and label inventory. OCTID/OCTDL NORMAL enables the required source
component. External MH/CSR is conditional: retain all five rare outputs and full
allocation rows: nine prediction columns for the nine-way audit and five for
five-way K-shot transfer. Five-way macro recall is not defined for partial-class
external cohorts: report each present class's recall and all five output counts,
without a misleading five-class aggregate. Save cosine-score columns as scores, not probabilities;
logistic heads additionally save class probabilities in fixed order.
CSR-to-CSC needs clinical label-equivalence confirmation.
OCTDL condition=MH alone is unsafe: inspected metadata separates76 VID scans into
39 MH/26 LH/11 VMT via subcategory. Metadata alone is not clinical certification.

No external-outcome tuning. If NORMAL access fails, call the deliverable a
rare-only pilot and mark the required shift component not run. If rare mapping
fails, retain NORMAL FRR and omit that disease-transfer claim. No outcome-driven
replacement cohort.

## 9. Stepping stones and execution gates

| Stone / Asia-Bangkok dates | Output and acceptance |
|---|---|
| S0: available | Historical Yoo pipeline, results and audits; distinguish them from new FM evidence |
| S1: 5–11 Nov | Proposed-methods IEEE draft; full draft 8 Nov, optional-results cutoff 9 Nov 18:00, PDF freeze 10 Nov 18:00; Lam submits 11 Nov |
| S2: 12–14 Nov | Isolated loader/evaluator; access/terms, grouping and inventory pass; lock episode/input identities before labelled scores; 100-image timing pilot |
| S3: 15–19 Nov | All-model paired development; nomination 19 Nov under section 7 |
| S4: 20–23 Nov | Locked legacy and eligible external query predictions |
| S5: 24–27 Nov | Paired summaries, K-shot/source plots, compute ledger; all-arm freeze 27 Nov |
| S6: 28 Nov–3 Dec | Actual Methods/Results/Discussion; evidence audit 1 Dec, PDF/package freeze 2 Dec, delivery 3 Dec |

11 November: English IEEE conference two-column assumption; 4–5 compiled pages
including references, Problems/Related Work/Proposed Methods/Discussion, Results
optional. Full draft ready without new FM outcomes. Course-specific template and
submission hour remain administrative unknowns; freeze PDF the previous day.
Resolve checkpoint/fallback access by 6 Nov where possible; gate failure must not
delay the proposed-methods report.

Pre-score input lock: this protocol hash, exact available roster, grouping and
exclusions, all episodes, the exact nine-way roster, model/preprocessing hashes,
code/environment and
endpoints; timestamp/archive before new labelled comparisons. External final
query IDs are separately locked before their inference under the already frozen
source/release/eligibility rules and section 4's external-only exclusions. This is not an
OSF registration unless actually submitted. Disclose historical test inspection.

Identity/numerical failure stops the arm; required fold/K failure stops the whole
selector. Retry interruptions under unchanged identities. Any scientific recipe
change gets a dated amendment/new version and preserves previous outputs.
If late, drop secondary probes first; keep primary paired
prototypes and source audit or explicitly report reduced/incomplete scope.
Neither report waits for favourable outcomes.

Outputs use a new versioned runtime-local root, /content/fm_v1/{features,runs,reports}.
Save input/episode IDs, grouping/exclusion ledgers, checkpoint/environment hashes,
query-level predictions and classifier scores, per-class counts, failures,
configuration, code revision, hardware, stage wall times and peak RAM/VRAM.
Record download, staging, extraction, classifier fitting and serialization costs
separately from the warm-inference tie metric, with image counts per encoder. GPU hours/cost remain unmeasured
until the pilot; no throughput guarantee is made.
Write completion markers last with artifact hashes. Resume verifies all identities
and completed artifacts; mismatch stops instead of overwriting. Interrupted
incomplete operations retry under the same identity. Export a complete hashed
archive to a user-controlled durable destination and verify the copied bytes
before relying on it. Local Colab Files are volatile; manual download/upload is
supported. No automatic new Drive writes or result-store migration is implied.

Historical seed-42 heads are not run on new external cohorts in v1. Replay here
means checking already saved legacy-test artifacts only. New external outcomes
come from the matched prototypes; no hidden extra historical-head arm is added.
If the S2 lock slips, S3 starts after a successful lock; the 27 November freeze
and incomplete-study reporting rule still apply.

## 10. Reuse inventory

| Reuse class | Replication asset | Reuse | Boundary / new work |
|---|---|---|---|
| Direct | data/manifest.csv and Stage0 conventions | Direct IDs, labels, fixed legacy comparator | New derived folds/supports; no supplied patient independence |
| Direct | 124 real rare train +22 test; common raw links | Read-only input images | Lock content hashes, groups, exclusions; external inventory is new |
| Historical context | Stage3 heads/predictions and Stage4 CSVs | Historical results and replay context | Nine-way Inception weights cannot become five-way K-shot/FM weights |
| Historical context | seed_check.csv and paired_seeds.csv | Motivation and observed training variability | Five seeds establish neither GAN superiority nor new-protocol episodes |
| Historical context | Historical Inception caches | Historical replay/context; preserve original bytes | Fresh matched FP32 reference required; old float16 arrays cannot recover original precision/provenance |
| Direct helpers | stage3 major_indices/fix and class order | Reuse pure helpers after checks | Stage3 extraction/augmented-validation CLI is not the new evaluator |
| Adapter | c23 identity/hash/atomic-completion/resume patterns | Adapt matching I/O protections | Sweep/validation hardcodes2048x9 heads and old experiment identities |
| Not used | fast_head.py | Keep for historical softmax work | No FM loader/prototypes/fixed L2-LBFGS objective |
| Adapter | analysis.py paired/metric/cluster utilities | Adapt pure functions where contracts match | Bootstrap targets accuracy/MCC; centroid_correct averages unnormalized supports; implement section 6 normalization/zero checks and new FRR/macro-recall evaluation |
| Historical / adapter | figures.py, ROC/t-SNE/CAM review | Historical figures and plotting patterns | New embeddings/plots required; Inception CAM formula does not explain ViTs |
| Preserve; exclude K-shot | Stage1 basic pools; GAN checkpoints/QC/v1/v2 | Preserve for separate full-budget augmentation | Excluded from K-shot; Inception-based QC used all rare training inputs |
| Direct context | L7/L8/L9/C0/C5 audits | Reuse limitations/provenance evidence | C0 separation is noncausal; C5 pool completion is not model validation |
| Workflow adapter | Colab notebook / COLAB.md | Mount, input staging and manual-session pattern | New isolated PyTorch extraction/evaluation; no Run All/canonical overwrite or implicit Drive output writes |
| Direct checks | .agents/check and tests | Existing regression/data protection | New semantic loader/episode/evaluator checks during implementation |

Savings: prepared data, audited historical baselines and persistence safeguards.
New work: FM extraction adapter, episode builder and dimension-independent
classifier/evaluator. GPU-hour savings remain unmeasured.

## 11. Provenance and review

Inputs read: repository AGENTS/CONTEXT, manifest/counts, Stage3/4, fast_head,
analysis/figures/c23, seed/cluster results and prior audits; the t3-8f9fcccf
worktree's FM_FEWSHOT_PLAN.md, FM_3WEEK_SPRINT.md and fsp review records.
Those uncommitted source plans remain preserved. This protocol is self-contained.
Input hashes, final-round notes and dispositions belong to FM_FINAL_REBUTTAL.md.

Final review: parent GPT-6.1-Sol plus requested Claude Opus 5.5 and Gemini 4 Argon High.
Both independent reviewers returned READY WITH ENUMERATED CHANGES on the archived
candidate. This version applies the resolved corrections; the parent disposition
and original verdicts are recorded in FM_FINAL_REBUTTAL.md. No unresolved design
objection remains after those corrections; the future gates are not completed tests.
A frozen design does not certify access, implementation, external labels,
patient independence, registration or clinical validity.

Workflow attribution: experimental-design/scientific-critical-thinking skills.
Kassis T, Agarwal V, He Y, Patel D, Brueckner AM (2026). Scientific Agent Skills:
A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065, current
record checked 5 November 2026. https://doi.org/10.48550/arXiv.2609.00065
Tooling attribution only, not retinal evidence.

Primary method source opened during final review:
Yoo TK, Choi JY, Kim HK (2021), Feasibility study to improve deep learning in OCT
diagnosis of rare retinal diseases with few-shot classification, MBEC 59:401–415.
https://pmc.ncbi.nlm.nih.gov/articles/PMC7829497/
The independent-test frozen-backbone workflow motivates the historical pipeline;
its results and clinician review do not substitute for our new evaluation.

