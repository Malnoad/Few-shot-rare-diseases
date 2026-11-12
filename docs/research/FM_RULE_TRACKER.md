# Rule tracker: open rules and stricter options for publication

Started 12 November 2026. Covers protocol v1.0 (`FM_PROTOCOL.md`), amendment A1
(`FM_AMENDMENTS.md`) and Lam's decisions (`FM_TERMS_EVIDENCE.md` section 6). Each row says what
is weak, the stricter publication-grade option, the agent's recommendation, and the deadline.
"Before lock" items must be settled and written down before the input lock (S2, 12-14 Nov):
adding them after scores would be outcome-driven. Lam decides every row.

## A. Must be settled before the input lock

| ID | Rule | What is weak | Stricter option | Recommendation | Status |
|---|---|---|---|---|---|
| A-1 | A1 OCTID two-way CSC-vs-MH test | K values, draws, arms and headline summary are not specified | Reuse the locked reporting episodes, keeping only CSC and MH supports; all four arms; reporting K set; all 100 draws; headline = nominee's two-class macro recall per K, with per-class recall and confusion counts for every arm | Adopt the stricter option | Adopted in A2 (13 Nov); implemented: two-way result from the five-way CSC/MH cosines |
| A-2 | A1 source probe | Classifier, features, validation and metric are not defined, so it can be run many ways | Define it fully (e.g. the protocol's fixed-L2 probe settings, image-level 5-fold CV, balanced accuracy and AUC, source as target) or drop it | Drop it; keep the geometry records. Both reviewers called it a diagnostic, not a gate | Withdrawn in A2 (13 Nov); replaced by the geometry table and the size-only reference |
| A-3 | A1 optional OCTID icon-mask crop | "Optional" checks added after scores look like tuning | Decide yes or no now; if yes, one fixed crop box, reported as sensitivity only | Decide now; the agent leans "no" for simplicity | Adopted in A2 (13 Nov): no crop or mask |
| A-4 | Section 8: "Failed external NORMAL access leaves a rare-only pilot" | Does not say whether one or both sources must fail (first-reader review) | Per source: report FRR for each source that passes; "rare-only pilot" only if both OCTID and OCTDL fail | Adopt as a dated amendment | Adopted in A2 (13 Nov) |
| A-5 | Kermany NORMAL FRR | Measured 12 Nov against the pinned nine-way roster (`72f90c00…`): **81 of 250** Kermany NORMAL test images share a patient with the 1,000/class fitting roster; 169 images (154 patients) do not. CNV 193/250, DME 131/250, DRUSEN 194/250 overlap | Pre-specify Kermany FRR on the 169 patient-disjoint NORMAL images as the headline Kermany estimate, the full 250 as secondary, both with denominators | Adopt; the protocol already calls the overlap a legacy limitation, this makes it measurable | Adopted in A2 (13 Nov); implemented; 169 images / 154 patients rechecked with the code |
| A-6 | R3 numerical rule (approved) | Repeatability on synthetic images only; a stable but wrong pipeline passes; max-based tolerance can hide small-element errors | Add a check on 5 real development images (no labels used): cosine similarity of repeated outputs at least 0.9999 per image | Add the real-image repeat; upstream-output comparison stays declined | Adopted in A2 (13 Nov) as first-vs-last timed pass on all 100 pilot images; implemented |
| A-7 | RETFound and MIRAGE final norm | First checks showed both use a default ones/zeros LayerNorm (RETFound discards the checkpoint's trained `norm`) | Read the upstream feature-extraction code at the pinned commits and follow it; record the choice and the reason in the lock | Check upstream before deciding; no sensitivity analysis after scores | Decided in A2 (13 Nov): keep the defaults, they match upstream |
| A-8 | Lock timestamp | Protocol: "This is not an OSF registration unless actually submitted." A local lock is self-attested | Deposit the lock hash (not data) publicly or under embargo before scores: OSF registration or a Zenodo record | Do it; it costs minutes, works with a private repo and is the strongest pre-specification evidence | Adopted in A2 (13 Nov); Lam deposits the hash after the lock, before the first develop |
| A-9 | Amendments in the lock | `FM_AMENDMENTS.md` is not yet hashed into the lock | Add it to the hashed lock inputs with the protocol | Required (code change) | Adopted in A2 (13 Nov); implemented (`CODE_FILES`) |
| A-10 | Lock storage | Repo to become private; whether the lock may be committed | — | Lam decides (decision 6) | Decided in A2 (13 Nov): lock ZIP not committed; repo public until the last Colab extraction |
| A-11 | Input preprocessing (found 13 Nov during the A-7 read) | Upstream paths disagree. RETFound: ours = `util/datasets.py` eval transform (resize 256 bicubic, crop 224, ImageNet mean/std), used by its fine-tune and linear-probe runs; `latent_feature.ipynb` instead resizes to 224x224 and standardizes each channel per image. MIRAGE: ours = `hf/mirage_hf.py` demo (resize 512 bilinear, L, /255); its linear-probe benchmark (`fm_cls_config.py` `MIRAGEFM`) also stretches each image to [0, 1] by min-max | One principle for both: each model's own linear-probe evaluation transform. RETFound unchanged; MIRAGE adds per-image min-max (one line in `fm_extract.py` plus a test) | Adopt the one principle before the lock; effect size unverified (small when an image already spans 0-255) | Decided in A2 (13 Nov): MIRAGE per-image min-max; implemented |
| N-1 | Size shortcut (found 13 Nov by the independent review) | File width and height alone separate the rare classes: leave-one-out macro recall 0.659 on the 124 development images, 0.75 on the legacy 22 (chance 0.20) | Report a size-only reference beside every arm and in the OCTID two-way test | Adopt | Decided in A2 (13 Nov); implemented |
| N-2 | A1 rule 5 intervals | Required but neither defined nor coded | Define per B-3 | Adopt | Decided in A2 (13 Nov); implemented |
| N-3 | External cohorts not bound by the lock | Rare cohorts entered through a `--mapping` flag typed after the lock | Make the A1 cohorts code constants; delete the flag | Adopt | Decided in A2 (13 Nov); implemented |
| N-4 | Lock scope | The lock hashes the code, so every endpoint needing code is lock-blocking, and pilots must match the final code | Finish amendments and code before the final pilots | Adopt | Recorded in A2 A-9 and the runbook |
| N-5 | Field of view | Crop arms keep 0.74-0.77 of 512-wide scans but 0.25 of 1536-wide ones; the wide-scan share differs by Kermany class and split | One geometry table by source, class and role | Adopt | Implemented (`fm_evaluate.py geometry`) |

## B. Statistics and claims: decide before reporting, ideally before the lock

| ID | Rule | What is weak | Stricter option | Recommendation | Status |
|---|---|---|---|---|---|
| B-1 | Primary contrast | Nominee-minus-Inception macro recall "at each K" gives five co-primary contrasts | Name one primary K (e.g. K=4 or 8), others secondary; or state all K as descriptive with no headline K | Name one primary K before the lock | Decided in A2 (13 Nov): no primary K; K mean over 1-8 as the single number; implemented |
| B-2 | Legacy rare test | 22 queries (4-5 per class), historically inspected; one image moves recall 0.20-0.25 | Report exact counts (x/5) per class and arm; describe legacy results as historically inspected and secondary; headline external evidence = A1 OCTID test and per-source FRR | Adopt in the reporting plan | Adopted: exact counts; reporting order fixed in A2 |
| B-3 | Intervals | No confirmatory statistics by design; reviewers may ask for uncertainty on the legacy and OCTID counts | Add exact binomial (Clopper-Pearson) intervals per class, labelled image-level and conditional (no patient IDs) | Optional; decide before reporting | Decided in A2 (13 Nov): nine-way intervals only; implemented |
| B-4 | Benchmark patient independence | Rare `pid` is one per image (e.g. `CSC_01`), so patients are unknown; duplicate review is done by students | State as a limitation; record the reviewers and decisions in the lock | Already in v1.0; keep | Covered; add the Google Images origin (Yoo et al.) when writing |
| B-5 | Pretraining exposure | RETFound's Kermany-related exposure is documented; whether OCTID or OCTDL were in RETFound or MIRAGE pretraining is unverified | Check both papers' pretraining data lists before reporting external results | Check before reporting | Mostly resolved; RETFound probably saw the Kermany test images (inference, unverified): disclose |
| B-6 | Benchmark label provenance | Who labelled the Yoo/Mendeley rare images, and by which definitions, is unverified | Read Yoo et al. Methods and cite it; otherwise state as unverified | Check before the next paper draft | Resolved 13 Nov: Yoo et al. is open access (PMC7829497); disease definitions still not given |

## C. Publication writing and compliance

| ID | Item | What is weak | Stricter option | Recommendation | Status |
|---|---|---|---|---|---|
| C-1 | Reporting checklist | No checklist yet | Fill the CLAIM 2024 checklist (Radiology: AI, doi:10.1148/ryai.240300) as a supplement. TRIPOD+AI (BMJ 2024, doi:10.1136/bmj-2023-078378) targets prediction models and fits less well | Use CLAIM 2024 | Open |
| C-2 | AI-use disclosure | Removed from the 11 Nov draft at Lam's request. ICMJE: "When AI is used, users should disclose which tool was used, and for what purpose." | Restore one sentence (manuscript or cover letter), as the target venue requires | Check the venue's policy before submission | Open |
| C-3 | Paper paragraph C25 | Still says rare transfer needs "clinical confirmation", contradicting A1 | Replace with A1's required wording | Next draft | Open: next draft (`protocols.md` fixed 13 Nov) |
| C-4 | Code and data availability | Repo to go private; publication needs a code availability statement | Archive the exact locked code commit (e.g. Zenodo) at submission. Dataset licences: OCTID CC0, OCTDL CC BY 4.0 (attribution); Kermany and Yoo/Mendeley licences not checked here | Plan now, do at submission | Open |
| C-5 | Data integrity | `octfsl` Drive is "Anyone with the link: Editor" | The lock's `extraction_inventory.csv` hashes every used image, so tampering would be caught; still restrict the link after extraction | Restrict after extraction | Open: switch the Drive link to Viewer before the lock ZIP is stored there (Lam) |

## Closed or decided

- External rare labels: amendment A1 (12 Nov).
- Terms (10 Nov); external sources and R5 order, R3 rule, R2 single-runtime plan,
  walkthrough (11 Nov).
- First checks passed (11 Nov, `first-check-20261111T175615Z`): all checkpoint SHA256 values
  matched, including RETFound `e9ff7864…`.

## Why earlier studies did not fix these (12 November 2026)

Sources read in full text through Europe PMC: RETFound (Zhou et al., Nature 2023, PMC10550819),
MIRAGE (Morano et al., npj Digit Med 2025, PMC12462498), TVST 2026 OCTID external validation
(PMC12786394), EyeDiff (PMC13172480); SimpleShot (arXiv:1911.04623). Yoo et al. 2021 was
wrongly marked paywalled here on 12 Nov; its full text is open at Europe PMC (PMC7829497, read 13 Nov). "Evidence" quotes the paper;
"Inference" is the agent's reading, not a statement by the authors.

| ID | How earlier studies handled it | Evidence or inference | Why it was not their priority | Consequence for us |
|---|---|---|---|---|
| A-1 | No K-shot external test exists. TVST tested fully trained models on OCTID; MIRAGE benchmarked OCTID within the dataset | Evidence (TVST): "External testing was limited to final pathology classification to enable a direct comparison of cross-domain generalization between models." | Their question was generalization of trained models, not few-shot supports | No template; define the test ourselves (A-1 recommendation) |
| A-2 | Nobody ran a source probe. MIRAGE explains dataset-level gaps by pretraining data | Evidence (MIRAGE): "These results can be attributed to the distribution of diseases in the pretraining dataset" | They compared encoders on standard benchmarks; shortcut testing was outside scope (inference) | Dropping the probe matches field practice |
| A-3 | No icon masking reported; OCTID images were only resized | Evidence (TVST): "All OCTID scans were resized to 224 × 224 pixels." Inference for MIRAGE (not mentioned in the text searched) | Same: standard preprocessing, shortcut not examined (inference) | "No mask" matches precedent; state the icon as a known cue |
| A-4 | Not applicable: earlier studies used whatever data they had | Inference | A rule for missing data only matters under a pre-registered plan | Ours to define |
| A-5 | MIRAGE used Kermany's original split; no patient-overlap discussion was found | Evidence (MIRAGE): "The Kermany dataset with the original split, which is the same we used in this work" | Using the standard split keeps results comparable with earlier papers (inference) | Our endpoint is a false-rare rate on NORMAL, so leakage matters more for us; report the 169 patient-disjoint images as headline, all 250 as secondary for comparability |
| A-6 | No repeat-output check reported; both teams ran their own released code with five training seeds | Evidence (RETFound): "we trained the model with five different random seeds"; code released on GitHub. Absence of a repeat check is inference from the text searched | Authors run their own pipeline, so loading equivalence is true by construction (inference) | We load three upstream models in one environment, so the risk is ours; keep R3 plus the real-image repeat |
| A-7 | RETFound adapts by fine-tuning, so its `fc_norm` is trained, never left at default. MIRAGE's frozen probe uses "global average pooling in the token dimension (excluding the global token) followed by a linear layer" | Evidence (RETFound): "Models are adapted to each dataset by fine-tuning". Evidence (MIRAGE): quote at left. The MIRAGE paper text names no extra norm, while its `mirage_hf.py` wrapper builds a LayerNorm not stored in the checkpoint (our load report) | RETFound never faced a frozen-feature norm choice; MIRAGE's text and code differ (inference) | Read both upstream codes at the pinned commits before the lock; the paper text alone does not settle it |
| A-8 | None of the four papers mentions registration | Evidence: no "registration/preregistration/OSF" in MIRAGE, TVST or EyeDiff full texts; RETFound's only "registered" refers to UK Biobank participants | Model-development papers do not test pre-stated hypotheses (inference) | Our main claim is pre-specification, so a public timestamp matters more for us than for them |
| B-1 | Few-shot papers report several K side by side with no single primary K | Evidence (SimpleShot): "drawing 10,000 K-shot C-way tasks ... Following prior work, we focus on one-shot and five-shot" | Benchmarks with thousands of tasks give tight averages, so multiplicity is not a concern (inference) | Reporting all K matches convention; naming one headline K is stricter and optional |
| B-2 | Yoo et al. reported per-class rates on the same 22 rare test images | Evidence (our `REPLICATION.md` L7, from the paper's Table 4): `none` TPRs 0.20/0.20/0.00/0.00/0.25 on 5/5/4/4/4 images. Whether they discussed the small test set is unverified (full text not yet searched for it) | Their paper is a "feasibility study" of augmentation (title) | Report exact counts; keep legacy as secondary |
| B-3 | RETFound and MIRAGE give seed-replica intervals; SimpleShot averages 10,000 episodes. None gives patient-level intervals for these datasets | Evidence (RETFound): "five different random seeds ... error bars show 95% CI"; MIRAGE: five seeds with standard deviation and t-tests | Their variance is training randomness on large test sets (inference) | Our protocol rejects episode intervals as population precision; exact binomial counts per class are the honest addition |
| B-4 | MIRAGE marks datasets without patient information and drops the patient-level spread there | Evidence (MIRAGE): "The dagger symbol (†) indicates that the patient information was not available, so the standard deviation is not shown." | They accept and disclose it per dataset | Do the same: disclose per dataset that rare patients are unknown |
| B-5 | Both papers name their pretraining data | Evidence (RETFound): OCT pretraining "85.2% of them came from MEH-MIDAS and 14.8% from ref. 34" (ref. 34 = Kermany et al., Cell 2018). Evidence (MIRAGE): "pretrained on a large in-house dataset ... (VIBES) registry of the Macula Clinic at the Medical University of Vienna" | Pretraining exposure only matters for later users who test on those datasets | RETFound saw Kermany images: its Kermany FRR and common-class results carry exposure. On the papers' own descriptions, neither RETFound nor MIRAGE pretrained on OCTID or OCTDL. DINOv2 is trained on curated natural images (DINOv2 abstract); medical content is unverified. Mostly resolved |
| B-6 | RETFound states each external dataset's labelling and flags OCTID's | Evidence (RETFound): "OCTID, describes image labelling based on the diagnosis of retinal clinical experts but does not specify duplicate adjudication." | They accept public labels and disclose how they were made | Follow this: state labelling provenance per dataset; Yoo's stays unverified |
| C-1 | No CLAIM checklist found in these texts | Inference | Nature-family journals use their own reporting summary (inference, unverified) | Use CLAIM if the venue asks |
| C-4 | Both release code; MIRAGE also releases its splits | Evidence (RETFound): code "available at https://github.com/rmaphoh/RETFound_MAE"; (MIRAGE): "we also provide these datasets with the same splits used in this study ... in our GitHub repository" | Publication norm in this field | A private repo needs a public archive (code, rosters, lock hash) at submission |

A-9, A-10, C-2, C-3 and C-5 concern our own process and have no counterpart in earlier studies.

## Triage: our problems versus the field's (12 November 2026)

Agreed with Lam on 12 November as the working split. IDs refer to the tables above.

**Ours to fix before the input lock** (they change our own features, endpoints or lock, and
cannot be added after scores): A-7 final normalization; A-1 OCTID two-way test definition;
A-5 Kermany patient leakage in the NORMAL FRR; A-6 numerical check; A-4 NORMAL-access failure
rule; A-9 amendments hashed into the lock; A-8 public timestamp (not blocking, but it backs the
pre-specification claim).

**Disclose only; the field accepts these** (limitations with the precedent cited): B-4 unknown
rare patients (MIRAGE's dagger convention); B-6 unverified benchmark labelling (RETFound's
per-dataset labelling statements); B-2 small legacy test (Yoo et al. used the same 22 images);
B-5 RETFound's Kermany pretraining exposure (RETFound's own data description); A-3 OCTID
orientation icon.

**Leave or drop; the field does not do these either:** A-2 source probe; B-1 single primary K;
B-3 patient-level intervals (exact per-class counts optional); C-1, C-2, C-4 at submission per
venue rules; A-10 and C-5 housekeeping.

**Gap the field has barely tackled, a possible contribution:** source and acquisition shortcuts
in cross-dataset few-shot transfer. In the studies read, dataset-level gaps are attributed to
pretraining data, not tested for acquisition cues. This study cannot remove them, but it can
measure and report them: geometry by class and source, the source-held-constant OCTID test,
and full reporting.
