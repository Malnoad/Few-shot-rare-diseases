# Foundation-model few-shot plan (tracks the October 2026 đề cương)

> **Closed record (13 November 2026).** Kept as evidence of how decisions were reached; no longer updated. The study as it stands is in [`protocols.md`](../../protocols.md); the binding texts are `FM_PROTOCOL.md` and `FM_AMENDMENTS.md`.

Status: **Reviewed planning proposal, two GPT/Gemini rounds, 2026-11-03.** This is a planning
deliverable, not authorization to implement or train. Lam remains the author and
chooses the open decisions in §7. Inputs: `D:/Few-shot/de-cuong-nghien-cuu.pdf`
(results to 28 Oct 2026), Lam's four-step foundation-model (FM) approach, the 29 papers
in `D:/Few-shot/few-shot paper (multiple fields)/`, a web check of retinal FMs, and the
repo at `a9ea542`.

## 0. Active deadline and scope update (2026-11-05)

Lam set **11 November for a 4–5 page IEEE draft** (problems, related work, proposed
methods, optional results and discussion), and planning through **3 December**.
[The active calendar](FM_3WEEK_SPRINT.md) controls schedule and experiment size:
report-first through 11 November; bounded frozen-encoder experiments afterwards;
19 November development decision; 27 November experiment freeze; final package
by 3 December. Colab Pro and fast decisions remain the resource constraints.
No new GAN/C1–C3 sweeps enter this period; the smaller matched common-cap diagnostic
and reduced secondary-probe draws are retained.
The methodological safeguards, literature and rebuttal history below remain
relevant. The original 18-week rollout and optional expansions are background,
not the current execution schedule. No implementation was started by this update.

## 1. Tracking the đề cương

The đề cương's spine is *"Disease signal or dataset shortcut?"*: a proposed
preregistered replication (RQ1) plus tests of whether the classifier learns disease or data source
(RQ2-RQ5), in seven work packages (WP) over 18 weeks from 2 Nov 2026 with gates G1
(week 4), G2 (week 5), G3 (week 18).

| Item | Đề cương says | Repo state 2026-11-03 (`a9ea542`) | Under the merged plan |
|---|---|---|---|
| RQ1 / WP1 (C1) | ≥63 seed pairs gan−basic, fixed250; ~35 min/head, 112-116 h at N=63 | Ready (notebook 4.1-4.5), no results in repo. T12 fast trainer supersedes the estimate: 12.87 min/head conservative, first stage 13.9-15.4 h, 63 pairs 41.6-43.1 h (laptop projection; Colab unbenchmarked) ([T12](T12_FAST_TRAINER.md)) | Preserve as the Inception reference. Its power target is a true 1 pp difference against zero, not proof the gain exceeds 1 pp |
| RQ2 / WP2 | Source-swap test: false-rare rate (FRR) of external NORMAL (OCTDL, OCTID) | Not started: no `data/external/`, no `wp2_source_swap.py`, no OSF record | Keep, extend to every FM. Core shortcut test |
| RQ3 / WP3 | Does GAN teach disease or source? 4 steps | Not started | Keep training-free steps 1-3; defer step 4 (`gan_unfiltered`, 20 seeds) |
| RQ4 / WP4 | RETFound linear probe, 20 seeds | Not started; original checkpoint identity/access/pretraining exposure need a current primary-source check (§3.6) | **Expanded into the FM line (§2–§3)**, the main change |
| RQ5 / WP5 | Multi-source NORMAL to reduce shortcut | Not started | Defer until after G2 |
| WP6 | Two ophthalmologists: GAN images, CAM, labels | Not started (contact via supervisor) | Retain clinical review; prioritize external-label mapping and uncertain labels. Venue acceptance cannot be promised |
| WP7 | Patient split, repeated rare hold-out, cluster CIs | Cluster CIs done (L9); rest not started | Derived development folds/supports only; do not re-sample the committed test split or treat duplicate groups as patient IDs |
| C2 / C3 | "No cells yet" | Ready (`a9ea542`, D18=A), no results | Unchanged; priority vs the FM line is decision D-FM4 |
| Week-1 list | D11, D17; update README/CONTEXT/REPLICATION for C5/C0; OSF; external data; WP2 skeleton; contact supervisor and clinicians | D17=B done (29 Oct); D11 on hold ([discrepancies](PAPER_DISCREPANCIES.md)); README/CONTEXT/REPLICATION updated for C5/C0; OSF, external data and WP2 skeleton not started; contacts unknown | OSF registration must now also cover the FM selection rule (§3.4) |

## 2. Lam's four steps, with the information budget made explicit

1. **Start with Kermany + Yoo and three pretrained encoders (§3.6).** Freeze them
   initially. Use the same image IDs and support/query assignments for every FM.
2. **Fit classifiers on our data.** A frozen encoder plus a fitted classifier is
   meaningful training. Keep the historical 9-way comparison separate from the
   rare-only 5-way classifier. Optional common-class encoder adaptation is a later
   experiment; no representation trained on all rare labels can be reused as a
   fresh K-shot initialization.
3. **Compare rare-label development performance; nominate a model.** Use the 124
   rare training images, with support/query separation and a fixed rule (§3.4).
   This estimates transfer within the available collection. Cross-source tests
   address a different question. Carry all three FMs through cheap baselines;
   nomination gates an expensive follow-up, rather than hiding other results.
4. **Adapt to the rare labels only.** For each K, start from the original frozen
   checkpoint and fit using only that episode's support. Fit a new 5-way head;
   do not restrict or renormalize a 9-way head trained on all 124 rare labels.

Seeing a rare label is not itself incompatible with few-shot learning: its support
examples are supposed to be seen. The restriction is **how many examples and labels
influenced each stage**. Selecting the FM using all 124 rare development labels is
target-informed model development. The resulting K-shot curve measures adaptation
conditional on that development budget; it is not a claim that the whole pipeline
needed only 5K labels. A strict total-label-budget alternative selects the encoder
and recipe on common classes or independent development tasks, then locks them
before seeing rare supports. Both budgets must be reported separately.

The 22 rare tests have already been analyzed in this project. They remain a locked
legacy comparison set, not a newly blind cohort. Registration can be prospective
for new FM experiments and external tests, while disclosing the historical analyses.
The development budget is **124 unique rare labels**, not 124+5K: episode supports
are drawn from those same images. Record 22 evaluation labels separately. If
exclusions reduce eligible supports, they do not erase prior inspection/exposure.

## 3. Proposed design

### 3.1 Questions and priority

- **RQ1 / WP1:** the existing C1 gan−basic comparison, unchanged.
- **RQ-FM1:** how do the three frozen encoders compare on 5-way rare classification
  at matched support sizes, and how stable is the ranking across support draws?
- **RQ2 / WP2:** how often do 9-way heads assign external NORMAL scans to a rare
  label, compared with Kermany NORMAL?
- **RQ-FM2:** at the full available rare-training budget, does historical basic/GAN
  augmentation help each frozen FM? This is a separate augmentation study.
- **RQ-FM3:** what accuracy/cost trade-off is observed, counting development labels,
  support labels, extraction, adaptation, and generation/QC costs separately?

Minimal FM study: RQ-FM1 plus WP2. Add the large augmentation factorial only after
the access, data and extraction gates pass. No new generator or adapter is required.

### 3.2 Three complementary evaluations

| Evaluation | Training information | Endpoints | Interpretation |
|---|---|---|---|
| A: 9-way replication extension | Existing common fitting split + all rare training data; none/basic/gan kept separate | Overall accuracy, common/rare macro recall, external-NORMAL FRR | Comparison of checkpoint pipelines on the legacy task; common patient overlap remains |
| B: 5-way rare-only adaptation | Fresh classifier on K support images/class, original frozen encoder | Macro recall (balanced accuracy), ordinary accuracy, each-class recall and confusion counts | Removes the common-versus-rare decision; within-rare acquisition and patient cues remain |
| C: shift diagnostics | Apply locked heads to external data; source/disease axis fitted on development features only | FRR for A; correct mapped-class recall for B; axis AUC and framing sensitivity | Evidence about shift vulnerability; no causal identification of source versus disease from AUC alone |

FRR = fraction of true NORMAL images predicted as any of CSC/MH/MacTel/Stargardt/RP.
A closed-set 5-way rare-only head always outputs a rare class; its FRR would be 100%
by construction. Use **9-way heads only** for this endpoint. External-NORMAL FRR
cannot establish discrimination among the five rare diseases.

For A's optional historical augmentation extension, reuse fixed250, with separate
outputs and an explicit cache adapter. `fast_head.make_model(width)` alone is not an FM loader:
Stage 3 still assumes Inception extraction and cache groups. First compare real-only
heads; none/basic/gan × every seed × every FM is a gated expansion. A common cap5000
is an optional new arm, not the original full-data protocol; Inception must be rerun
on the same cap if it is used.

**Required minimal WP2 arm:** one fixed seed-42 cap5000 common fitting roster
(20,000 rows, identical across checkpoints) plus all eligible real rare fitting
rows; fit a nine-way cosine-prototype head on unit embeddings, using the same
centroid/tie rules as B. Compare the three FMs with a **matched Inception prototype
head on those same rows**, one deterministic fit/checkpoint (no optimizer seeds).
This is a new capped real-only diagnostic, not C1 or a fixed250 reproduction.
Also apply the three saved historical seed-42 nine-way heads as contextual WP2
references, clearly separated from matched comparisons. Report NORMAL recall,
all common-class confusions and nine-way rare recall alongside FRR: never predicting
rare is not evidence of useful robustness. This minimal WP2 needs common/external
feature extraction, even though the first rare-only pilot does not. If deferred,
the deliverable is only a rare-only pilot, not the complete minimal shift study.

### 3.3 Data and leakage gates

The committed manifest has **84,630 rows**: 74,082 train, 9,526 val, 1,022 test.
Rare training counts are CSC 27, MH 26, MacTel 25, Stargardt 19, RP 27; rare tests
are 5/5/4/4/4. No real rare validation rows exist. Source: `data/manifest.csv`, counted
2026-11-03. Derived episode manifests reference these rows; the original never moves.

- Screen exact/near duplicates before defining development folds. L7 found no exact
  rare duplicates; similarity groups are not verified patients. Require disjoint
  groups across support/query and record uncertain cases. If groups cannot support
  all five classes and the requested K, report infeasibility rather than relaxing
  independence silently. Image-level analysis cannot claim patient generalization.
- **Basic augmentation must descend only from the current support.** Keep its
  descendants with their parent. Existing basic pools need parent provenance before
  fold/K-specific use; generating basic views on demand is simpler.
- **Exclude existing GAN pools from K-shot/CV claims.** Their generators and QC used
  all rare training images. Filtering outputs to a support filename cannot undo
  that exposure. A genuine K-shot GAN comparison needs a generator and QC built only
  from each support, including all selection labels in its budget; defer it.
- Existing GAN pools remain usable for A's historical/full-budget experiment.
  Disclose that automatic QC used Inception features, so accepted-image selection
  favors a particular representation. New-backbone differences are not pure effects
  of pretraining, architecture or modality.
- Use each checkpoint's documented preprocessing, with a deterministic shared
  framing sensitivity if later approved. Do not force all models through the
  historical 256→299 path. Record channel handling, crop/resize, normalization,
  pooling, resolution and text prompts. Learned scaling/PCA fits on support only.
- New caches must identify image-content hashes and order, model/weight revision,
  preprocessing and pooling, feature dtype/dimension and environment. Existing
  path-only caches do not supply that provenance ([L7](L7_BASELINE_AUDIT.md)).

External tests: use OCTID NORMAL for FRR and MH / CSR as a **partial rare-label**
transfer check, with clinical confirmation of CSR→CSC and label definitions.
For the rare-only head, keep all five outputs when scoring external MH/CSC; a 2-way
renormalized score would be a different, easier task. OCTDL supplies external
NORMAL, not a verified five-rare-class cohort; do not map its broad VID label to MH.
Dataset licenses, release hashes, patient IDs, exclusions and duplicate/pretraining
overlap must be recorded before use. Do not tune on these external tests or promise
external validation of MacTel/Stargardt/RP without corresponding data.

**Metadata follow-up, not new test results:** direct inspection of the author's
[OCTDL labels, Zenodo record 10839556](https://zenodo.org/records/10839556/files/OCTDL_labels.csv?download=1)
found 2,064 rows. Of 76 VID rows, **all 76 have `condition=MH`**, but `subcategory`
is MH for 39, LH for 26 and VMT for 11. Thus `condition==MH` alone is unsafe.
Only the 39 metadata-MH rows are candidates for a clinically verified MH transfer
subset; they are not yet accepted labels or evaluated scans. NORMAL has 332 rows
and 110 unique `patient_id` values. Snapshot MD5 matches the author's listed
`2fced789038c6ca83283066c2f8ad289`; inspect the chosen image release against that
metadata before use. No external images were downloaded. A two-class external
cohort still needs full five-output confusion rows and rates of assignment to
MacTel/Stargardt/RP, not just the two correct-class recalls.

### 3.4 Development selection and final adaptation

Proposed fixed recipe, subject to Lam's approval before execution:

1. Build one five-fold stratified, duplicate-grouped development assignment using
   only the 124 rare train rows. Freeze it and a seed/episode list before FM scores.
   Check class/group feasibility first; patient IDs remain unknown. Screen links
   to legacy/external queries too. A split-crossing group triggers a documented
   derived fitting exclusion/sensitivity; it never changes the canonical manifest.
   If any required fold lacks a query class or eight eligible fitting groups/class,
   stop the **entire primary selector** and revise before model scores. Do not
   average a selectively feasible subset of folds/K values.
2. Within each fold's fitting rows, sample **100 paired support draws** per
   K ∈ {1,2,4,8}; the fold's held-out real rare rows are queries. The same draws and
   queries apply to every FM. Draw K distinct eligible groups/class, one image per
   group; use nested prefixes of the same group permutation across K. Each query
   fold must contain all five classes. Freeze the within-group image choice across
   nested prefixes. These counts are planning choices, not power claims.
3. Primary classifier: cosine nearest prototype (normalize embeddings, average per
   class, normalize each centroid). No learned temperature for argmax predictions.
   Secondary: five-way logistic regression on unit vectors, mean cross-entropy +
   `0.01 × ||W||² / 2`, unpenalized intercept, L-BFGS with at most 1,000 iterations
   and tolerance 1e-6. These are proposed fixed settings, not a tuned optimum;
   report convergence failures. No query labels for tuning, stopping, prompts,
   augmentation or fitting preprocessing.
   Encoder extraction uses inference mode; queries cannot update normalization
   statistics. Freeze class order and deterministic tie handling; fail nonfinite
   or zero-norm embeddings/centroids. Macro recall is `(1/5) × sum_c(correct_c/n_c)`;
   a missing query class fails an episode, rather than disappearing from the mean.
4. Nominate by mean query macro recall across the four K values for the primary
   classifier: pool out-of-fold correctness within each class before averaging
   classes, then average across K and support draws. Publish every FM's K-specific
   results and eligible query denominators. Specifically:
   `S_m = mean_(K,r) [(1/5) × sum_c (sum_f correct_(m,K,r,f,c) / sum_f n_query_(f,c))]`.
   Here K={1,2,4,8}, r=1..100; this weights images equally within each class,
   not folds/groups equally and not majority votes across draws. The near-best
   set is `S_max − S_m ≤ 0.02`; within it minimize median warm extraction seconds/image
   on matched hardware and inventory, with recorded checkpoint order as the final
   tie-break. This practical
   tie rule is not a statistical equivalence claim. K=all linear probing is a
   separate ceiling, not a proxy for low-shot performance.
5. Lock checkpoint, recipe, prompts and preprocessing, then apply all three models
   to the legacy 22 tests with **100 fixed paired, nested support rosters** from the
   eligible rare training split. Freeze these rosters before reporting scores.
   Report K ∈ {1,2,4,8,16}; add all=19–27/class as a distinctly unbalanced full-data
   reference, using actual post-exclusion counts. Reporting K=16 requires 16 eligible
   groups/class. K=16 is excluded from the CV selection score: a fold can leave fewer
   than 16 Stargardt fitting images even before grouping. Larger K requires an
   explicit feasibility check, never sampling with replacement to manufacture shots.
6. Evaluate external cohorts only after locking the pipeline. Positive partial
   transfer is not evidence of five-disease external generalization. If a later
   PEFT experiment is chosen, reset to an eligible initialization and count all
   adaptation/tuning labels; it needs a separate approved specification.

The selected development score is optimistic after selecting a winner. Treat it
as development, not an unbiased estimate of the selected procedure. Nested
assessment or genuinely independent data is required for that stronger claim.

### 3.5 Uncertainty and limits

Training seeds measure optimizer variability; support draws measure variability
conditional on the available images. Neither creates new patients. Report support
draw distributions separately from uncertainty about test sampling; never compute
SEM as if thousands of repeated predictions were independent test cases.

On the 22 rare tests, one error changes overall accuracy by 1/22 ≈ 4.55 pp and
class recall by 20–25 pp. Exact two-sided McNemar gives p=.03125 for 6 discordant
wins and no losses; this is an illustration for **micro accuracy**, not a power
analysis or a test of macro recall. Any confirmatory pairwise tests require a
prespecified contrast and multiplicity handling. Use confusion counts and paired
differences as primary descriptive evidence on this legacy set.

Group-aware resampling must resample independent cases and rerun data-dependent
steps for pipeline uncertainty. Where only image identities exist, label intervals
as conditional on image independence. Do not attach a binomial 124-sample SE to
overlapping CV predictions or call repeated folds independent replicates.

The source/disease axis AUC (0.995455 current; 0.991455 framing) is a separability
diagnostic, not proof that morphology was ignored. C0 remains stopped under D20=A;
new framing experiments would require a new decision, not bypassing its gate.
Kermany's train/test patient overlap remains a limitation of A. A source-shift
penalty or external-NORMAL errors can have multiple causes; test them, do not
declare an identified shortcut mechanism from a single metric.

### 3.6 Three-model shortlist

**Recommended lineup: RETFound-MAE Nature OCT + DINOv2-Base + MIRAGE-Base.**
These are three concrete pretrained representations, not three newly developed
models. They cover OCT-specialized, general visual, and paired retinal multimodal
pretraining while keeping the downstream classifiers image-only and comparable.
No theorem or result here establishes which will win.

| Candidate / exact public identifier | Rationale and required input | Access / terms checked 2026-11-03 | Exposure and feasibility limits |
|---|---|---|---|
| [RETFound-MAE Nature OCT](https://huggingface.co/YukunZhou/RETFound_mae_natureOCT), `YukunZhou/RETFound_mae_natureOCT` | Continuity with proposal WP4; OCT-specific MAE, ViT-L/16. Use its OCT checkpoint and documented preprocessing, not a fundus or downstream-finetuned weight | Official files are gated by HF acceptance/login; model card and [repository LICENSE](https://github.com/rmaphoh/RETFound/blob/main/LICENSE) specify CC BY-NC 4.0. Actual authenticated download/load is untested | [Original Methods, ref. 34](https://doi.org/10.1038/s41586-023-06555-x) include Kermany OCT images in unlabelled pretraining. This is upstream image exposure, not proof of supervised label leakage. Common-test independence from pretraining is not claimed |
| [DINOv2-Base](https://huggingface.co/facebook/dinov2-base), `facebook/dinov2-base` | General visual reference, ViT-B/14; fixed native processor and image features | Public ungated checkpoint; original DINOv2 code and weights are [Apache-2.0](https://github.com/facebookresearch/dinov2#license). Download/load and Colab timing untested | Web-scale natural-image pretraining is not an audited exclusion of published retinal images. Choose Base for the pilot; different architecture/capacity means this is a checkpoint comparison, not a causal test of domain specialization |
| [MIRAGE-Base](https://huggingface.co/j-morano/MIRAGE-Base), `j-morano/MIRAGE-Base` | OCT/SLO/layer pretraining; OCT-only inference for our B-scans. Native 512×512, patch 32; no new SLO or segmentation inputs required | Ungated official weights. HF card and live [code LICENSE](https://raw.githubusercontent.com/j-morano/MIRAGE/main/LICENSE) say CC BY-NC 4.0; a README license badge differs, so pin and record the actual terms. Download/load and pooling validation remain E0/E2 gates | [Paper](https://doi.org/10.1038/s41746-025-01852-3) describes private VIBES pretraining and evaluation on Kermany/OCTID/OCTDL. These were downstream benchmarks, not declared pretraining sets; image/patient overlap with our web-sourced Yoo cases is not auditable |

MIRAGE's missing-modality handling is supported by its paper's masking strategy
(preprint Methods §4.3) and [official wrapper](https://raw.githubusercontent.com/j-morano/MIRAGE/main/hf/mirage_hf.py).
The pilot must verify OCT-only loading, input range/channel handling and feature
pooling against the benchmark implementation before using cached vectors. Do not
run a generic setup script that downloads its benchmark datasets or alters Python.

**Alternatives, not extra required arms:**

- [EyeCLIP](https://github.com/Michi-3000/EyeCLIP) is OCT-capable and supplies a
  weight link, but explicit code/weight reuse terms were not found in the checked
  release. Its README reports repaired text weights in February 2026: any later
  use needs the corrected revision and checksum. A public link is not a license.
  It remains closest prior work; use it as a comparator only after terms and
  checkpoint semantics are resolved.
- [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
  is a practical VLM alternative if Lam explicitly wants a text-based third arm:
  ungated/MIT model card, biomedical figure-caption pretraining rather than
  OCT-specific pretraining. Published-image overlap is unknown. Its classifier
  comparison should still use the same image-only heads; zero-shot text scoring
  is secondary and requires fixed prompts.
- FLAIR and RET-CLIP are fundus-focused in their checked releases; do not call them
  OCT-specialized because an acronym sounds retinal. DINOv3 and newer RETFound
  variants exist; the proposed bounded lineup is not a latest-model leaderboard.

If RETFound access is unavailable, do not replace it after inspecting model
performance. Lam chooses a revised lineup before extraction. No checkpoints have
been downloaded and no new FM dependency has been installed.

## 4. Literature positioning

### 4.1 The 29-paper folder: what we build on
Numbers are the stable folder indices in Appendix A, not the current filesystem
sort order. Three earlier readers supplied the full-paper notes; the resumed
review re-extracted all 29 PDFs and checked the key claims against the original
texts. This is a targeted narrative synthesis, not a systematic review.

| Use in our plan | Papers | What we take (and cite them for) |
|---|---|---|
| **Selection and strong baselines** | [25] CLAP; [17] Tip-Adapter | CLAP's fixed-configuration experiments warn that adapter rankings depend on target-task tuning, and use a strong text-initialized linear probe. This does not prove every adapter requires test leakage. Tip-Adapter combines a visual cache with CLIP text logits: an image-only kNN baseline is not the full method |
| **Metric/meta-learning alternatives** | [02] MAML; [05] TADAM; [09] LEO; [10] CNAPs | Initialization, metric scaling and task conditioning are established ideas. They require different training machinery, not just a frozen-feature cache. Use a plain prototype baseline first; temperature does not change cosine-prototype argmax |
| **Cross-domain medical evidence** | [15] STARTUP / BSCD-FSL; [03] FADA | STARTUP evaluates ChestX and ISIC, improves transfer using unlabeled target-domain images, and finds its tested meta-learners weaker than naive transfer. It is not evidence that STARTUP itself loses to transfer. FADA assumes shared classes across domains; ours differ |
| **Information and tuning budgets** | [11] TIM; [14] BERT fine-tuning; [19] MetaICL; [20] T-Few; [21] FT vs ICL | Pair support draws, disclose development/tuning resources and report variability. TIM uses query-batch information: transductive methods would need a separate arm. NLP efficiency results motivate measurement, not an OCT speed-up claim |
| **VLM adaptation** | [16] CoCoOp; [18] CoOp; [23] TaskRes; [26] CLIP-Adapter; [28] 2SFS | Prompt and residual adaptation are established; base gains can hide novel-class losses. 2SFS motivates a gated PEFT follow-up, but its tested CLIP/text initialization cannot be assumed to transfer to image-only encoders |
| **Generative augmentation** | [07] FUNIT; [12] few-shot GAN adaptation; [13] cross-domain correspondence; [22] DreamBooth; [24] Textual Inversion | Memorization and incidental attribute transfer motivate lineage and diversity checks. [12] Fig. 4 shows FID can favor overfitting in a low-data setting. These observations motivate, but do not establish, a source shortcut in OCT. Defer diffusion |
| **Cross-field context** | [01] imitation; [04] voice cloning; [06] talking heads; [08] meta-RL; [27] multimodal survey; [29] test-time training | Separate expensive pretraining from adaptation and test stability across supports. [29] studies language-model reasoning on ARC/BBH, not a retinal adaptation method |

### 4.2 What the folder does not cover
The folder is a method background collection, not a representative search of
retinal research. In the reviewed notes, it does not directly evaluate the
Kermany/Yoo source-confounded benchmark or its proposed NORMAL-source substitution
test. That bounded observation cannot establish novelty against the wider field.
No cross-paper result supplies the power or patient-level uncertainty of this
22-image legacy test. Keep the common visual classifiers identical across FMs;
any text/zero-shot advantage is a separately labeled VLM-only analysis.

### 4.3 Nearest prior work outside the folder
| Prior work | Overlap with our proposal | Defensible distinction to test |
|---|---|---|
| [EyeCLIP, 2025](https://doi.org/10.1038/s41746-025-01772-2), Results/Figs. 3–4, Methods | Already evaluates few-shot OCTID/OCTDL and rare-eye-disease classification, comparing retinal and VLM baselines | Our specific Kermany/Yoo audit, explicit information budgets and paired support protocol; no claim to introduce rare-eye FM adaptation |
| [EyeDiff, 2026](https://doi.org/10.1038/s41746-026-02560-2), Results and Methods | Already augments minority/rare ophthalmic classes and evaluates RETFound, EyeFound and EyeCLIP, including OCT | A controlled CycleGAN/basic/no-augmentation comparison and shift diagnostics, with generation/QC lineage and costs; not the first synthetic-data-plus-FM study |
| [Generalist versus Specialist Vision Foundation Models, 2025 preprint](https://arxiv.org/abs/2509.03421), abstract | Already compares DINOv2/DINOv3 with RETFound variants, linear probing/fine-tuning, label and compute efficiency | Efficiency and generalist/specialist comparisons alone are not novel; our restricted rare-class and validity question needs results |
| [BenchReAD, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/1683_paper.pdf), §2/Fig. 1 | Trains with OCT2017 and evaluates OCTDL/OCTID, including unseen anomalies | Its anomaly-detection endpoint differs from our closed-set five-disease discrimination and NORMAL→rare diagnostic; cross-dataset OCT evaluation is established |
| [MIRAGE, 2025](https://doi.org/10.1038/s41746-025-01852-3), classification/cross-dataset experiments | Already compares retinal and general FMs with linear probes on OCT benchmarks, including shifted-source tests | Adding MIRAGE to three encoders is not novelty; the Kermany/Yoo information-budget and augmentation audit must supply the new evidence |

Read depth: EyeCLIP main article and official repository; EyeDiff full JATS body
retrieved via Europe PMC and DOI/title confirmed in Crossref; BenchReAD full paper;
MIRAGE paper/preprint Methods and official code; generalist/specialist preprint
**abstract only**. Do not infer absent controls from
an abstract or claim none of these studies has ever audited shortcuts.

## 5. Novelty and efficiency

### 5.1 Novelty
**Working contribution:** a reproducible evaluation of whether apparent rare-OCT
gains persist under explicit label budgets, basic-augmentation controls, repeated
paired supports and source-shift diagnostics on an audited legacy benchmark.
The contribution is an empirical validity result and reusable evaluation protocol,
not a new FM, a new adapter, or the first few-shot rare-eye application.

An application-only comparison on 22 tests would be weak novelty and weak external
evidence. A stronger outcome needs informative external NORMAL and mapped-disease
tests, a clear augmentation control and a measured cost ledger. Even then, a
negative FRR test does not prove absence of shortcuts. If external data or label
mapping fail, scope the paper to a bounded benchmark audit and conditional
adaptation results; clinical generalization remains unresolved.

Provisional title: **"Label budgets and source shift in foundation-model adaptation
for rare retinal OCT classification."** The original replication can remain the
reference arm, rather than expanding every WP into a second full research project.

### 5.2 Efficiency angle
Two meanings, both kept: the plan spends compute where information per hour is
highest, and the paper can make a measured cost claim.

**Plan efficiency: start with the 146 real rare images per FM.** Fixed-encoder
features can be reused across support draws. Encoder updates, preprocessing or
weight revisions require new cache identities; there is no universal one-pass claim.

| Step | Compute | Size | Status of estimate |
|---|---|---|---|
| WP2 with existing Inception heads | CPU extraction + saved heads | External NORMAL and mapped diseases only | Proposal's 11–17 img/s and ~2,600 scans are unverified estimates; actual subset sizes and I/O need measurement |
| B: real-only pilot | Colab GPU extraction, CPU classifiers | 124 development + 22 legacy test images per checkpoint | Time, memory and failure rate unmeasured; do this before a large extraction |
| A: capped/full extension, per FM | Colab GPU extraction; CPU heads | cap5000 manifest = 30,672 rows; +10,000 basic +9,889 GAN v1 = 50,561; full manifest plus those pools = 104,519, before external data | Counts derived from the manifest/pools; not runtime estimates. v2 is a distinct 15,000-image pool |
| A: head fitting | Laptop or Colab CPU | Separate cache adapter and width-aware head | T12's 12.87 min/head applies to its tested Inception setup only; benchmark FM widths, learning dynamics and I/O |
| C1 (Inception reference) | Colab CPU, unattended | 63+ pairs | 41.6-43.1 h projected (T12) |

For any cap arm, the fixed common subset must use identical rows for every backbone.
Available Inception features may be reused, but fitting the matched capped head
still costs compute. Original full-data results are not a cap-matched comparator.

**Paper efficiency claim: a cost ledger, not an assertion.** Per arm, record GPU type
and hours, CPU hours, real rare images used, synthetic images used, and expert time.
Separate cold start (download, extraction, generation/QC, fitting) from amortized
classifier fitting on existing caches. Published pretraining is an external resource,
not zero compute. Historical CycleGAN timing logs may be missing; a timed epoch is
an estimate, not a measured end-to-end reconstruction. Include all real rare images
and labels seen by generators, QC, model selection and tuning. Match the prediction
task and data budget before comparing costs: a 5-way head and a 9-way GAN pipeline
are not equivalent clinical tasks.

Hypothesis to test: frozen-feature adaptation offers a useful accuracy/cost trade-off
without generation. Report a frontier and uncertainty; do not claim it is cheaper
or better before measurements exist.

## 6. Rebuttal log

The requested models were used as T3 child tasks with separate note ownership:
**GPT-6.1-Sol**, max reasoning / priority, and **Gemini 4 Argon High**. Earlier
Claude paper-reader notes were retained; its unfinished FM search had failed.
Review is advisory. The parent checked claims against files/primary sources and
keeps disagreements visible; model agreement is not scientific evidence.

| Round / objection | Disposition and concrete response |
|---|---|
| GPT R1 M1–M4: budgets, low-shot selection, K=16, augmentation lineage | Accepted. Target-informed development explicitly separated from support budget; matched low-shot episodes; CV K≤8; no all-rare-trained GAN reuse |
| GPT R1 M5–M7: transforms/cache identity, pseudoreplication, patient/source uncertainty | Accepted. Inference-only raw features; support-only fitted transformations; group feasibility/exclusions; support variability separated from cohort uncertainty |
| GPT R1 M8–M10: FRR task, Inception QC, preprocessing fairness | Accepted. FRR only on nine-way heads; fresh five-way heads; historical QC-conditioned augmentation; native-checkpoint preprocessing and bounded checkpoint conclusions |
| GPT R1 M11–M12: compute scope and C1 target | Accepted. Small real-only pilot before large factorial; timings remain projections; existing C1 detects an effect against zero, not 80% power for a ≥1 pp lower bound |
| Gemini R1: EyeCLIP/MIRAGE/EyeDiff overlap, access and modality checks | Accepted where primary sources corroborate them. Narrowed novelty; gated RETFound; MIRAGE OCT-only alternative; no forced fundus-only VLM; EyeCLIP terms unresolved |
| Gemini R1: invented external FRR, pixel-stat AUC attribution, C1–C3 descriptions and original-paper identity | **Rejected.** No 99.6% external-NORMAL FRR result exists in the checked repo. AUC .995455 is the feature-axis diagnostic, not seven pixel statistics. C0 is the stopped framing gate; C2/C3 are dose/cap studies. C1 is paired cached-head fitting, not ten generator reruns. Yoo is the MBEC paper, DOI 10.1007/s11517-021-02321-1, not the Ophthalmology citation supplied in R1 |
| Gemini R1: automatic zero-overlap declarations, causal shortcut amplification, guaranteed <15 s / one-session costs, extra adapter methods | **Rejected or unverified.** Private pretraining is not an image-overlap audit; mechanisms and costs need data. No new methods or experiment replacements were adopted from these assertions |
| GPT R2: scoring/ties, feasibility stop, unique labels, minimal nine-way arm and registration | Accepted. Added an explicit OOF score, near-best-to-maximum tie set, fixed cost measure, 100 reporting rosters, whole-selector stop rule, unique-label accounting and a matched minimal nine-way prototype arm. Registration remains a proposed future action |
| Gemini R2: retractions, external confusion outputs, data gate and K fallback | R1 project errors explicitly retracted; useful source facts retained. Accepted full five-output external confusion rows and inventory before E4. Did not adopt an automatic reduced-K selector: infeasibility stops the primary design before scores, as GPT advised |
| OCTDL MH mapping disagreement | Parent checked the actual author metadata. All VID rows have condition MH, so R1's condition-only filtering is unsafe; subcategory separates 39 MH / 26 LH / 11 VMT. Clinical equivalence and image-release checks remain gates |

Local round-1 reports: `.agents/reports/fsp/sol_protocol_r1.md`,
`gemini_fm_r1.md`, and `gemini_search_provenance.md`. **The Gemini R1 report contains
the rejected assertions above and must not be used as a verified project record.**
Retain it as review history, not as source evidence for those claims.
Round-2 records: `sol_protocol_r2.md` and `gemini_fm_r2.md` in the same local folder.
Both judged the bounded plan ready for Lam's decisions; the parent applied the
remaining small clarifications above. This does not certify runtime, patient
independence, external labels or novelty before actual results.

## 7. Decisions for Lam before implementation

| ID | Decision | Options | Draft recommendation |
|---|---|---|---|
| D-FM1 | Paper framing | A: one FM adaptation/validity paper with the replication as reference. B: replication/shortcut-first, minimal FM extension. C: separate papers | A, with claims bounded by external data |
| D-FM2 | Which 3 FMs | A: RETFound Nature OCT + DINOv2-Base + MIRAGE-Base. B: replace MIRAGE with BioMedCLIP for a VLM arm. C: EyeCLIP after explicit terms/checkpoint verification | A, subject to access and inference gates |
| D-FM3 | New dependency for FM extraction | A: Colab-only `fm_extract.py` in PyTorch with its own requirements file, writing `.npz` caches that the existing heads read. B: port FMs to TensorFlow. C: precomputed embeddings from a third party | A |
| D-FM4 | C1/C2/C3 priority | A: preserve their approved protocols; schedule C1 separately if Colab resources permit, FM real-only pilot first among new tasks. B: prioritize existing C1/C2/C3 before FM work. C: change/stop C1 with a separate decision | A; no claim that simultaneous Colab sessions are available |
| D-FM5 | Adaptation scope | A: frozen features + prototypes/linear probes; real-only K-shot; historical augmentation only at full budget. B: add common-only PEFT after a separate specification. C: support-specific GAN retraining | A; B/C are not prerequisites |
| D-FM6 | External data | A: prepare OCTID/OCTDL inventory and use terms, then download after approval; mapped disease labels need clinical confirmation. B: defer and restrict claims to the available collection | A |
| D-FM7 | Label budget / selection | A: use the 124 target development labels for selection; disclose them separately from each K-shot support. B: strict total-label budget, select only on common/independent development tasks | A matches Lam's step 3; B supports a stronger total-label-efficiency claim |

No code/dependency changes, datasets/checkpoints, OSF submission, clinician contact,
training, commits or pushes are part of this planning delivery.

### Execution gates after approval

| Gate | Concrete output | Pass / stop condition |
|---|---|---|
| E0: access and provenance | Three exact checkpoints, terms, revisions, exposure table, preprocessing/pooling definitions | Do not start a checkpoint without usable weights and terms; replacements are chosen before outcome inspection |
| E1: data and registration | Immutable derived folds/episodes, class-by-fold group counts, external inventory/terms, support/query assertions, data/label budgets, prespecified endpoints | Stop the entire required selector if any fold/K is infeasible; amend before model scores. Disclose historical test inspection and prospective/exploratory components |
| E2: small pilot | Frozen rare embeddings, paired prototype/linear-probe outputs; GPU/CPU time and peak memory | Confirm load/extraction semantics and budget before extracting common/synthetic pools; accuracy is not an access gate |
| E3: locked evaluation | All three models, legacy five-way comparison, external mapped-class check and nine-way FRR | No external-test selection. If mapping/access fails, narrow the claim rather than substituting a new test post hoc |
| E4: optional extension | Full-budget none/basic/gan, or a separately specified PEFT study | Proceed only if the preceding evidence and Lam's priorities justify the extra compute |

The proposal's 18-week schedule and G1/G2/G3 remain a planning reference, not an
assurance. Week 1 should prioritize E0/E1, registration and external inventory;
weeks 2–4 the pilot and WP2. Re-estimate the remaining timeline using actual
extraction/I/O results before allocating the large head factorial.

## Appendix A. Traceable index of all 29 folder papers

Source directory: `D:/Few-shot/few-shot paper (multiple fields)/`. Year, venue and
title below identify the supplied PDFs; these are not independent publication
status verifications. Full filenames and reader notes are also preserved locally
in `.agents/reports/fsp/paper_index.tsv` and `group_{V,M,G}.md` (gitignored).

| ID | Supplied paper | How it informs this plan |
|---|---|---|
| 01 | One-Shot Visual Imitation Learning via Meta-Learning (CoRL 2017) | Cross-field context; expensive task-distribution learning precedes adaptation |
| 02 | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (ICML 2017) | Historical optimization-based baseline; not required for five rare classes |
| 03 | Few-Shot Adversarial Domain Adaptation (NeurIPS 2017) | Domain shift with shared labels, unlike the common/rare split |
| 04 | Neural Voice Cloning with a Few Samples (NeurIPS 2018) | Separate pretraining and adaptation cost |
| 05 | TADAM: Task dependent adaptive metric for improved few-shot learning (NeurIPS 2018) | Distinguish learned metric scaling from fixed-prototype inference |
| 06 | Few-Shot Adversarial Learning of Realistic Neural Talking Head Models (ICCV 2019) | Generator adaptation and realism evaluation are separate from diagnosis |
| 07 | Few-Shot Unsupervised Image-to-Image Translation / FUNIT (ICCV 2019) | Generative few-shot transfer and downstream classification precedent |
| 08 | Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning (ICLR 2019) | Adaptation under shift; no OCT accuracy evidence |
| 09 | Meta-Learning with Latent Embedding Optimization (ICLR 2019) | Learned adaptation in parameter space; exceeds the minimal cache baseline |
| 10 | Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes (NeurIPS 2019) | Task-conditioned features; different training requirements |
| 11 | Transductive Information Maximization For Few-Shot Learning (NeurIPS 2020) | Query-batch use must be declared as transduction |
| 12 | Few-Shot Adaptation of Generative Adversarial Networks (arXiv 2020) | Memorization-aware evaluation; FID alone is insufficient |
| 13 | Few-shot Image Generation via Cross-domain Correspondence (CVPR 2021) | Transfer regularization and incidental attribute leakage |
| 14 | Revisiting Few-sample BERT Fine-tuning (ICLR 2021) | Stability and model-selection budgets |
| 15 | Self-training For Few-shot Transfer Across Extreme Task Differences / STARTUP (ICLR 2021) | Medical cross-domain evidence; unlabeled target-data budget matters |
| 16 | Conditional Prompt Learning for Vision-Language Models / CoCoOp (CVPR 2022) | Base/novel trade-off; needs a text encoder |
| 17 | Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification (ECCV 2022) | Cache-plus-text precedent and cheap frozen-feature baseline |
| 18 | Learning to Prompt for Vision-Language Models / CoOp (IJCV 2022) | Prompt adaptation is established, not our contribution |
| 19 | MetaICL: Learning to Learn In Context (NAACL 2022) | Sensitivity to support draws; cross-field context |
| 20 | Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning / T-Few (NeurIPS 2022) | Count adapted parameters and measured cost, without importing NLP speedups |
| 21 | Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation (ACL Findings 2023) | Matched tuning/model/data budgets |
| 22 | DreamBooth (CVPR 2023) | Later diffusion context; not a required new arm |
| 23 | Task Residual for Tuning Vision-Language Models (CVPR 2023) | Residual classifier adaptation; text-derived initialization |
| 24 | An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion (ICLR 2023) | Concept personalization is not disease-valid augmentation evidence |
| 25 | A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models / CLAP (CVPR 2024) | Fixed-recipe comparisons and strong linear-probe baseline |
| 26 | CLIP-Adapter: Better Vision-Language Models with Feature Adapters (IJCV 2024) | Lightweight feature adapters; not automatically superior to probing |
| 27 | Few-shot Adaptation of Multi-modal Foundation Models: A Survey (arXiv 2024) | Taxonomy and broader positioning, not proof of novelty |
| 28 | Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages (CVPR 2025) | Optional PEFT stopping/adaptation design |
| 29 | The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (ICML 2025) | Language-model ARC/BBH evidence; do not conflate with retinal query adaptation |

## Appendix B. Search scope, attribution and verification

Search/access date: **2026-11-03**. Targeted web searches included `EyeCLIP few-shot`,
`rare OCT foundation few-shot`, `Boosting foundation models for rare eye disease
diagnosis`, and `BenchReAD retinal anomaly detection`; followed author/publisher
links and official code/model cards. Scope is bounded, with no exhaustive or
"first-ever" claim. Exact-title Crossref query (3 returned records; correct DOI
identified) verified EyeDiff's title/DOI. Europe PMC supplied its full JATS body.
A broader Europe PMC query combining OCT, few-shot and foundation-model terms
failed HTTP 503 on both attempts: this is unavailable coverage, not zero hits.
Publisher HTML redirects/CAPTCHAs required primary-source mirrors/full-text XML.

Local evidence: `.agents/reports/fsp/` contains the freshly extracted proposal,
29 paper texts, original reader reports, API payloads and GPT/Gemini reports.
Those files are scratch; this plan retains the substantive protocol, source links,
paper index and review dispositions so it does not depend solely on ignored notes.

Tooling: literature-review and paper-lookup procedures supported the synthesis;
their library is described by Kassis, Agarwal, He, Patel & Brueckner (2026),
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)
(current record v2 checked). This is tooling attribution, not scientific evidence
for a retinal-model claim. AI review is not clinical validation or peer review.

Repository verification on 2026-11-03 (existing harness, main checkout Python 3.12):

```text
Identity check: PASS (0 outgoing commits; author and committer match local noreply)
Agent checks: PASS
```

Completion surfaces: entry points/notebook flags, reverse actions and feature-cache
schemas are unchanged; no data/results changed. CONTEXT links this proposal as
pending planning. README/COLAB remain execution instructions for the approved
pipeline and need no new runnable commands for an unimplemented proposal.
