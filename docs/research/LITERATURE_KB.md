# Retinal imaging and few-shot learning: project knowledge base

Version 1.0 · reviewed 5 November 2026 · research author: Lam.

## Start here

A compact evidence map covering **all 49 supplied PDFs** (20 retinal/medical,
29 cross-field few-shot) and **17 targeted additions**. This is a single-agent
narrative synthesis, not an exhaustive review of either field. All supplied PDFs
were text-extracted and inspected at abstract/available-conclusion depth; selected
Methods passages were also read for STARTUP and CLAP. **This is not a claim of
full-paper review of all 49.** Added sources have individual reading scopes below.

Read sections 1–4 for the project argument. Use the catalogue to find a paper.
[The source index](LITERATURE_SOURCES.json) provides exact PDF paths/hashes,
primary URLs, reading depth and search/access provenance. Before adding a number,
detailed-method claim or claim about absent controls, open the relevant primary
passage and record its location. Do not reread the entire corpus for routine tasks.

Protocol v1.0 governs the proposed experiment; data/results and primary sources
support observations. This digest supplies no new experiment or protocol change.
The exact protocol path and SHA256 are recorded in the source index. The protocol
is committed as [FM_PROTOCOL.md](FM_PROTOCOL.md); the source index keeps the
original worktree paths as recorded on 5 November.

## 1. The two domains and the argument connecting them

**Retinal imaging:** Yoo supplies the rare-OCT augmentation benchmark; Quellec
supplies earlier rare-fundus few-shot evidence; RETFound, EyeCLIP and MIRAGE
establish retinal pretrained representation/adaptation precedents. Fundus
photographs, OCT B-scans, OCT volumes and OCT angiography are different inputs.
Classification, segmentation, anomaly detection and referral are different tasks.

**Few-shot learning across fields:** metric/meta-learning, frozen-feature
classification, prompts, parameter-efficient tuning, generation and in-context
learning reduce different task-specific burdens. Their pretraining, development
labels, unlabelled-target data, supports and query access must be distinguished.

**Our intersection:** compare fixed pretrained checkpoint pipelines on five-way
rare-OCT classification with identical support-label budgets, then probe errors
on shifted-source NORMAL images with separate nine-way classifiers. The proposed
contribution is an auditable evaluation in this collection, not a new encoder or
adaptation algorithm, first rare-eye FM application, or demonstrated clinical value.

| Evidence | Consequence for this project |
|---|---|
| R01 Yoo; R03 Quellec | Acknowledge existing rare-retinal few-shot work. Legacy augmentation differs from strict K-shot adaptation. |
| A02/A04/A05 retinal FMs; A07 generalist/specialist study | Neither comparing DINO with retinal FMs nor label-efficiency analysis is new by itself. Dataset-specific validity evidence must carry our argument. |
| F25 CLAP; F21 fair NLP comparison | Use strong simple baselines and explicit tuning budgets; no universal linear-probe winner follows. |
| A11 ProtoNets; A12 SimpleShot; F05 TADAM | Geometry and normalisation matter. Our frozen cosine centroids are not a trained ProtoNet or an exact SimpleShot/TADAM implementation. |
| F15 STARTUP; F11 TIM | Unlabelled target images and query-batch information are extra resources, excluded from our inductive protocol. |
| F16/F18 prompts; F23/F26 adapters; F28 2SFS | Text-dependent adaptation is not a drop-in image-only method. Small trainable parameter count is not measured total cost. |
| Generative papers; A06 EyeDiff | Plausibility, diversity and diagnostic utility are separate. Subsampling generated images cannot undo broader training/QC exposure. |
| A15 PathPT | Relevant medical precedent, but prompts, spatial aggregation and tile supervision are combined; this is not OCT evidence or an isolated prompt effect. |
| A08 BenchReAD; R15 MRI anomaly study | Shift evaluation is established; anomaly detection differs from naming five diseases. |

There is **no universal shot-count ladder** establishing prototypes best at 1–5,
PEFT appropriate only at tens, or full fine-tuning requiring hundreds. Data,
features, optimisation and selection matter. Measure our actual runtime/memory.

## 2. Active method: compressed protocol v1.0 snapshot

This snapshot is for reasoning/writing. Open the complete protocol before
implementation. Its SHA256 is
`4e8df8d91c0021187c05b103a9b0a0e0319a055fd91108da1f4cf3045d00c5df`.

- **Roster:** RETFound Nature OCT, DINOv2-Base, MIRAGE-Base; matched Keras
  Inception-v3 reference. BioMedCLIP replaces RETFound only for the predefined
  access/loading/terms failure, before new labelled comparisons.
- **Data:** CSC/MH/MacTel/Stargardt/RP development counts 27/26/25/19/27 = 124;
  legacy queries 5/5/4/4/4 = 22. Counts rechecked from the manifest on this date.
  These are images, not verified independent patients.
- **Information budget:** 5K real supports/head; 124 unique inspected development
  labels disclosed separately, with supports drawn from that same inventory.
  Disclose the historically inspected 22 evaluation labels, common labels and
  upstream pretraining exposure separately.
- **Primary prototype:** normalise each support embedding, average by class,
  normalise the centroid, predict maximum cosine similarity. Fixed class-order
  ties; reject nonfinite/zero-norm embeddings and centroids.
- **Secondary logistic probe:** unit features; mean cross-entropy
  + 0.005*||W||_F^2, unpenalised intercept, zero initialisation, fixed L-BFGS-B
  settings/convergence rules from the full protocol. Balanced K/class supports;
  no added imbalance weights. Neither CLAP nor the historical Adam head.
- **Development:** five fixed grouped folds; K={1,2,4,8}; 100 paired nested
  support draws. Logistic sensitivity: K=4, first 25 matching draws/fold.
  Require at least ten eligible groups/class and all specified fold conditions;
  infeasibility stops the selector. Image-similarity groups are not patient IDs.
- **Nomination:** complete prototype results only; pool out-of-fold correctness
  within each class, then average classes/K/draws. At least two complete FMs.
  Among FMs within two percentage points of the best, use the protocol's measured
  warm-extraction timing rule. Inception is reference only. Lock the primary
  nominee before legacy/external outcomes; publish every model.
- **Reporting:** K={1,2,4,8,16}, 100 prototype draws where feasible; first 25
  matching logistic draws/K. Omit K=16 for every encoder if any class lacks
  sufficient groups. Draw spread is conditional on these images, not a patient CI.
- **Nine-way source audit:** identical seed-42 common fitting cap of 1,000/class
  plus all eligible real rare fitting images; one deterministic prototype head
  per encoder. FRR = true NORMAL scans predicted as any rare class / all true
  NORMAL scans. Report per source, with NORMAL recall, common confusions and rare
  recall. A five-way rare-only head cannot supply a meaningful NORMAL FRR.
- **External:** OCTID/OCTDL NORMAL inventories; rare transfer requires verified
  mappings. OCTDL's broad VID/condition labels are not automatically MH;
  CSR-to-CSC requires confirmation. No five-class aggregate on partial-class
  cohorts. Freeze eligibility before outcomes, final query IDs before inference.
- **Features/compute:** fresh FP32 caches with image/checkpoint/preprocessing
  identity; CPU classifiers; one Colab Pro GPU session at a time. Legacy float16
  caches and nine-way fitted weights are historical context, not matched features.
- **Outside this sprint:** synthetics in K-shot/CV, zero-shot text scoring,
  prompt search, PEFT, LP-FT, backbone fine-tuning, new generation, C1–C3 sweeps.
  Drop secondary probes first if late.

Do not restore older 5,000/class caps, all-data ceilings or historical-head
external arms. The active cap is 1,000. Runtime, access, grouping, implementation
and clinical label mapping remain gates; review agreement did not pass them.

## 3. Claims that must retain their limitations

| Claim or observation | Required boundary |
|---|---|
| Retinal and cross-field papers motivate our design. | They do not predict the winner or establish clinical value. |
| Matched supports/classifiers improve comparability. | Architecture, capacity, preprocessing and upstream exposure differ: no causal pretraining-effect conclusion. |
| Five-way rare classification removes the common/rare output decision. | Within-rare acquisition and patient cues can remain. |
| External NORMAL FRR can expose a practical source-dependent failure. | It does not identify its cause; low FRR without useful rare recall is not robustness. |
| The historical source/disease-axis AUC is high. | Source and disease are confounded; C0 did not prove morphology was ignored. |
| Duplicate screening/grouped folds help limit dependence. | Unknown rare patients remain unknown; clustering cannot repair common-patient split overlap. |
| Repeated support draws quantify example-selection variability. | Draws are not new patients and do not justify population confidence intervals. |
| The legacy split is fixed. | Its 22 rare labels were historically inspected; it is not newly blind. |
| Protocol v1.0 is design-finalized. | Not implemented, publicly preregistered, clinically validated or demonstrated to fit GPU resources. |

Historical numerical evidence lives in `data/manifest.csv`,
`results/stage4/*.csv`, `results/seed_check.csv`,
`results/analysis/{paired_seeds,cluster_summary}.csv`,
`results/c0/v1_gate_summary.csv`, L7/L8 audits and `REPLICATION.md`.
For example, the saved five-seed mean GAN–basic accuracy difference is **-0.348
percentage points**, descriptive historical evidence that does not establish
GAN superiority. No new FM scores are supplied here. Preserve all run-of-record
outputs and the committed splits.

## 4. Drafting use

Synthesize four themes: (1) retinal need and the Kermany/Yoo lineage, with Quellec
where useful; (2) RETFound/DINOv2/MIRAGE and nearest EyeCLIP precedent;
(3) simple adaptation and information-budget lessons from prototype lineage,
CLAP, STARTUP/BSCD-FSL; (4) the bounded validity question and limitations.
EyeDiff/Tai/Iwanicka belong where augmentation is discussed; PathPT is a
cross-medical adaptation contrast, not an extra experiment arm.

Start with roughly 8–12 essential primary references, adding citations where
claims require them rather than enforcing a quota. The complete catalogue is
not a demand to cite all 66 papers in a 4–5 page draft. Robotics, speech, NLP and
personalisation mostly supply background reasoning. Surveys organise the field;
original studies support empirical claims. Mark preprints explicitly.

The 11 November report is a complete **proposed-methods draft**, with new results
optional. No invented accuracy table, empty Results heading or claimed clinical
benefit. Sparse dataset labels do not establish epidemiological disease rarity.
Use [the Task 1 starter prompt](TASK1_DRAFT_PROMPT.md) for acceptance criteria.

## 5. Supplied retinal / medical collection: all 20 PDFs

T = abstract and available-conclusion primary-text inspection, not a full audit.
Secondary reviews/perspectives are labelled by scope. Exact local paths and hashes
are in the source index.

| ID | Paper / scope | Takeaway | Boundary |
|---|---|---|---|
| R01 | [Feasibility study to improve deep learning in OCT diagnosis of rare retinal diseases with few-shot classification](https://doi.org/10.1007/s11517-021-02321-1) — Direct OCT; historical motivation | Per-disease CycleGAN augmentation feeds a frozen-backbone nine-way OCT classifier. | Full-budget augmentation differs from strict support-only K-shot adaptation; our automatic QC is not its clinician review. |
| R02 | ['Squeeze & Excite' Guided Few-Shot Segmentation of Volumetric Images](https://arxiv.org/abs/1902.01314) — Medical CT; method context | A support-conditioned two-arm network segments volumes using few annotated slices. | Segmentation and prior task training differ from image-level OCT classification. |
| R03 | [Automatic detection of rare pathologies in fundus photographs using few-shot learning](https://doi.org/10.1016/j.media.2020.101660) — Direct retinal; different modality | Features learned for frequent conditions support a probabilistic rare-condition detector. | Fundus, multiple conditions and t-SNE-based modelling differ from our five-way OCT task. |
| R04 | [Anomaly Detection-Inspired Few-Shot Medical Image Segmentation Through Self-Supervision With Supervoxels](https://doi.org/10.1016/j.media.2022.102385) — Medical segmentation; context | A foreground prototype and anomaly scoring avoid explicit heterogeneous-background modelling. | Pixel-level anomaly segmentation is not closed-set disease classification or NORMAL false-rare rate. |
| R05 | [Discriminative ensemble learning for few-shot chest x-ray diagnosis](https://doi.org/10.1016/j.media.2020.101911) — Medical few-shot classification | Separate broad feature learning from a task-specific saliency/autoencoder ensemble. | Chest-X-ray evidence supports modular adaptation, not a numerical OCT performance expectation. |
| R06 | [TOP-GAN: Stain-free cancer cell classification using deep learning with a small training set](https://doi.org/10.1016/j.media.2019.06.014) — Cell imaging; transfer context | Transfer a discriminator learned from unlabelled cell images to small labelled cancer-cell tasks. | This is representation transfer, not evidence that synthetic OCT images are diagnostic. |
| R07 | [Generative Adversarial Network in Medical Imaging: A Review](https://doi.org/10.1016/j.media.2019.101552) — Secondary overview | Maps synthesis, translation and augmentation applications of adversarial learning. | Use for taxonomy; cite original experiments for performance claims. |
| R08 | [Semi-supervised Task-driven Data Augmentation for Medical Image Segmentation](https://arxiv.org/abs/2007.05363) — Medical segmentation | Optimise intensity/shape synthesis for a downstream segmentation task using labelled and unlabelled data. | Count target-image access and training costs; no guaranteed OCT classification gain. |
| R09 | [Diffusion Models for Medical Image Analysis: A Comprehensive Survey](https://arxiv.org/abs/2211.07804) — Secondary overview | Organises diffusion formulations and medical synthesis/analysis applications. | A survey does not validate a generator or justify adding diffusion to this sprint. |
| R10 | [Synthesizing retinal and neuronal images with generative adversarial nets](https://doi.org/10.1016/j.media.2018.07.001) — Retinal fundus synthesis | Generate diverse images conditional on vessel/neuronal structure; evaluate segmentation use. | Structure-conditioned fundus synthesis does not establish rare-OCT lesion fidelity. |
| R11 | [Not-so-supervised: a survey of semi-supervised, multi-instance, and transfer learning in medical image analysis](https://arxiv.org/abs/1804.06353) — Secondary overview | Distinguishes transfer, semi-supervised and multi-instance learning information sources. | Weak or unlabelled supervision is still information; it is not a zero-data budget. |
| R12 | [Embracing Imperfect Datasets: A Review of Deep Learning Solutions for Medical Image Segmentation](https://arxiv.org/abs/1908.10454) — Secondary overview | Separates scarce labels from sparse/noisy/weak annotation problems. | Few labels and unreliable labels need different remedies; do not treat them as interchangeable. |
| R13 | [On the Challenges and Perspectives of Foundation Models for Medical Image Analysis](https://arxiv.org/abs/2306.05705) — Secondary perspective | Distinguishes general, modality-specific and organ/task-specific pretrained models. | Domain specialisation is motivation to test, not an established winner for our dataset. |
| R14 | [Models Genesis](https://doi.org/10.1016/j.media.2020.101840) — Medical 3D self-supervision | Restoration-based self-supervision creates transferable representations from unlabelled 3D scans. | Not an OCT K-shot experiment; the supplied filename's 2022 date is not the publication year (2021). |
| R15 | [Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study](https://arxiv.org/abs/2004.03271) — Medical anomaly detection | Compares reconstruction-based anomaly methods under controlled data/architecture choices and domain shift. | Anomaly detection differs from naming five diseases; matched comparisons remain important. |
| R16 | [REFUGE Challenge: A Unified Framework for Evaluating Automated Methods for Glaucoma Assessment from Fundus Photographs](https://doi.org/10.1016/j.media.2019.101570) — Retinal fundus benchmark | Standardises glaucoma classification and disc/cup segmentation evaluation. | Clinical task, modality and endpoint differ; use its benchmarking lesson, not its accuracy. |
| R17 | [OCTA-500: A retinal dataset for optical coherence tomography angiography study](https://doi.org/10.1016/j.media.2024.103092) — Retinal OCT/OCTA dataset | Provides paired modalities, volumetric/projection data and structural segmentation labels. | OCTA/segmentation is not our B-scan classification; the supplied 2020 filename masks a 2024 journal report. |
| R18 | [Applications of Deep Learning in Fundus Images: A Review](https://arxiv.org/abs/2101.09864) — Secondary retinal overview | Surveys fundus disease classification, lesion/biomarker segmentation and synthesis. | Fundus findings cannot be silently relabelled as OCT evidence. |
| R19 | [GAMMA Challenge:Glaucoma grAding from Multi-Modality imAges](https://arxiv.org/abs/2202.06511) — Multimodal retinal benchmark | Evaluates glaucoma grading with fundus images and OCT volumes. | Paired multimodal inference is different from our OCT-only downstream inputs. |
| R20 | [Structured layer surface segmentation for retina OCT using fully convolutional regression networks](https://doi.org/10.1016/j.media.2020.101856) — Direct OCT; segmentation | Predicts ordered continuous retinal-layer surfaces in a single network. | Anatomical segmentation can motivate morphology-aware work; no segmentation arm is planned here. |

## 6. Supplied cross-field collection: all 29 PDFs

F IDs retain the previous plan's stable index, not a new filesystem sort.
All are T; F15/F25 additionally have selected Methods inspected. The index
separates arXiv submission year from unverified venue dates.

| ID | Paper / scope | Takeaway | Boundary |
|---|---|---|---|
| F01 | [One-Shot Visual Imitation Learning via Meta-Learning](https://arxiv.org/abs/1709.04905) — Robotics | Meta-learns from prior tasks so a robot can adapt from a demonstration. | One-shot refers to demonstrations after extensive training, not total study data. |
| F02 | [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) — Meta-learning | Learns an initialization that adapts with a few gradient updates. | Requires a task distribution and meta-training; not our frozen-feature procedure. |
| F03 | [Few-Shot Adversarial Domain Adaptation](https://arxiv.org/abs/1711.02536) — Domain adaptation | Adversarially aligns source/target examples using a few labelled target cases. | Assumes shared label semantics; common Kermany and five rare classes do not supply that match. |
| F04 | [Neural Voice Cloning with a Few Samples](https://arxiv.org/abs/1802.06006) — Speech | Contrasts speaker fine-tuning with inferring a new speaker embedding. | Separate initial training, adaptation and per-query cost; audio results do not predict OCT gains. |
| F05 | [TADAM: Task dependent adaptive metric for improved few-shot learning](https://arxiv.org/abs/1805.10123) — Metric meta-learning | Learns task-conditioned metrics and studies metric scaling. | Training-time metric scaling differs from multiplying fixed cosine logits, which leaves argmax unchanged. |
| F06 | [Few-Shot Adversarial Learning of Realistic Neural Talking Head Models](https://arxiv.org/abs/1905.08233) — Generative vision | Extensive meta-training enables personalised generators from few images. | Visual realism is not clinical fidelity; count pretraining separately. |
| F07 | [Few-Shot Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1905.01723) — Few-shot image translation | Uses many source classes to translate to a new class specified by few reference images. | Not equivalent to a per-disease CycleGAN; source-task supervision and synthesis costs matter. |
| F08 | [Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347) — Robotics/control | Learns a dynamics prior for online adaptation to changing conditions. | Its feedback and test-time update setting are outside this static OCT protocol. |
| F09 | [Meta-Learning with Latent Embedding Optimization](https://arxiv.org/abs/1807.05960) — Meta-learning | Adapts a learned low-dimensional generative representation of classifier parameters. | Requires meta-training; small latent updates do not mean a training-free method. |
| F10 | [Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes](https://arxiv.org/abs/1906.07697) — Task-conditioned classification | Uses an adaptation network to modulate classifier/features for new tasks. | Fast test-time adaptation follows trained task-conditioning machinery. |
| F11 | [Transductive Information Maximization For Few-Shot Learning](https://arxiv.org/abs/2008.11297) — Transductive classification | Uses support supervision plus mutual information over unlabelled queries. | Query-batch access changes the information budget; excluded from our inductive comparison. |
| F12 | [Few-Shot Adaptation of Generative Adversarial Networks](https://arxiv.org/abs/2010.11943) — Few-shot generator adaptation | Adapts pretrained GAN singular values while retaining singular vectors. | Generation metrics and visual quality do not establish downstream diagnostic utility. |
| F13 | [Few-shot Image Generation via Cross-domain Correspondence](https://arxiv.org/abs/2104.06820) — Few-shot generation | Regularises transfer to preserve source diversity with few target images. | Transferred incidental source attributes can persist; this motivates audits, not proof of OCT shortcuts. |
| F14 | [Revisiting Few-sample BERT Fine-tuning](https://arxiv.org/abs/2006.05987) — NLP optimisation | Shows optimisation choices strongly affect small-sample stability and baseline rankings. | No universal shot threshold makes full fine-tuning safe or unsafe across models. |
| F15 | [Self-training for Few-shot Transfer Across Extreme Task Differences](https://arxiv.org/abs/2010.07734) — Cross-domain few-shot; includes medicine | Uses unlabelled target-domain images for representation self-training before support-only classification. | An additional unlabelled-data budget; this does not establish our original frozen checkpoints will generalise. |
| F16 | [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557) — Vision-language prompts | Conditions learned prompts on each image to improve transfer to unseen classes. | Needs a compatible text encoder; base/new class transfer is not our same-five-class support split. |
| F17 | [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification](https://arxiv.org/abs/2207.09519) — Vision-language cache adaptation | Combines a labelled visual cache with CLIP's text-derived classifier; optional cache fine-tuning. | Our image-only cosine centroids are not the complete Tip-Adapter method. |
| F18 | [Learning to Prompt for Vision-Language Models](https://doi.org/10.1007/s11263-022-01653-1) — Vision-language prompts | Learns continuous textual context while pretrained encoders remain fixed. | Trainable prompt parameters still require optimisation; image-only encoders cannot use this text recipe. |
| F19 | [MetaICL: Learning to Learn In Context](https://arxiv.org/abs/2110.15943) — NLP in-context learning | Meta-trains a language model to condition on examples across tasks. | No update at deployment does not erase the earlier task-training budget. |
| F20 | [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638) — NLP parameter efficiency | Learns activation rescaling with a fixed few-shot adaptation recipe. | NLP parameter/runtime gains are not measurements on our retinal encoders. |
| F21 | [Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation](https://arxiv.org/abs/2305.16938) — NLP fair comparison | Matches model sizes and example budgets when comparing adaptation strategies. | A lesson in fair controls, not proof of a universal fine-tuning or prompting winner. |
| F22 | [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — Text-to-image personalisation | Fine-tunes a diffusion model to preserve a subject from few reference images. | Subject identity fidelity is not disease-labelled synthetic-data validation. |
| F23 | [Task Residual for Tuning Vision-Language Models](https://arxiv.org/abs/2211.10277) — Vision-language residual tuning | Learns a residual to a frozen text-derived classifier. | Requires VLM text priors; not ordinary zero-initialised image-only logistic regression. |
| F24 | [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618) — Text-to-image personalisation | Learns new token embeddings for concepts while the generator stays frozen. | Concept personalisation is not evidence of diagnostic correctness. |
| F25 | [A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models](https://arxiv.org/abs/2312.12730) — Few-shot evaluation | Tests target-specific tuning and shift; motivates strong, explicitly configured linear baselines. | Its CLIP text-initialised/constrained probe is not our fixed-L2 logistic objective. |
| F26 | [CLIP-Adapter: Better Vision-Language Models with Feature Adapters](https://doi.org/10.1007/s11263-023-01891-x) — Vision-language feature adaptation | Fits a bottleneck adapter and blends adapted and pretrained features. | Small adapter size alone does not guarantee lower total GPU cost or better OCT accuracy. |
| F27 | [Few-shot Adaptation of Multi-modal Foundation Models: A Survey](https://arxiv.org/abs/2401.01736) — Secondary overview | Organises prompt, adapter and external-knowledge adaptation families. | Use as taxonomy; retrieve primary studies for effect or superiority claims. |
| F28 | [Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](https://arxiv.org/abs/2503.11609) — Vision-language PEFT | Learns PEFT features first, then trains a linear classifier. | This order differs from LP-FT, which fits a head before unfreezing the backbone. |
| F29 | [The Surprising Effectiveness of Test-Time Training for Few-Shot Learning](https://arxiv.org/abs/2411.07279) — Language reasoning | Updates language-model parameters from in-context examples on ARC/BBH. | Not retinal evidence, and not an implementation of our frozen-encoder episodes. |

## 7. Targeted additions: 17 sources

A = official abstract/metadata only; T = named primary-text passages. Do not use
an A entry for uninspected experimental details or absence-of-control claims.

| ID | Source | Read scope | Use and boundary |
|---|---|---|---|
| A01 | [Kermany et al. (2018), Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://doi.org/10.1016/j.cell.2018.02.010) | T: publisher summary, transfer-learning figure and evaluation description | Common-class OCT data and frozen-backbone transfer-learning precedent. The paper's test description does not erase patient overlap found in our actual manifest. |
| A02 | [Zhou et al. (2023), A foundation model for generalizable disease detection from retinal images (RETFound)](https://doi.org/10.1038/s41586-023-06555-x) | T: publisher text and archived primary-text passages; prior source ledger for ref.34 | Retinal self-supervision and label-efficient downstream adaptation motivate an OCT-specific checkpoint. Its downstream fine-tuning results do not predict our frozen-prototype result; Kermany pretraining exposure needs disclosure. |
| A03 | [Oquab et al. (2023; arXiv revision 2024), DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | A: official arXiv abstract/record | General-purpose self-supervised features motivate the general visual comparator. Not direct rare-OCT evidence; architecture/capacity differ from the retinal checkpoints. |
| A04 | [Morano et al. (2025), Multimodal foundation model and benchmark for comprehensive retinal OCT image analysis (MIRAGE)](https://doi.org/10.1038/s41746-025-01852-3) | T: publisher abstract/introduction and indexed architecture/pretraining passages | OCT/SLO multimodal pretraining and a classification/segmentation benchmark are close prior work. OCT-only loading and pooling still need verification; our Base arm is not its headline Large configuration. |
| A05 | [Shi et al. (2025), A multimodal visual-language foundation model for computational ophthalmology (EyeCLIP)](https://doi.org/10.1038/s41746-025-01772-2) | T: publisher abstract/introduction and initial Results; prior plan records deeper review | Ophthalmic VLM with few-shot/zero-shot evaluations is an essential nearest-prior citation. This review did not re-audit every task-specific result; do not claim the first rare-eye FM application. |
| A06 | [Chen et al. (2026), Boosting foundation models for rare eye disease diagnosis via a multimodal text-to-image generative framework (EyeDiff)](https://doi.org/10.1038/s41746-026-02560-2) | T: archived full-text Introduction, Results and targeted Methods; live publisher redirect failed | Combines synthetic ophthalmic augmentation with foundation models, including OCT and rare-disease tasks. No first-ever synthetic-plus-retinal-FM claim; its protocols and results are not ours. |
| A07 | [Zhou et al. (2025), Generalist versus Specialist Vision Foundation Models for Ocular Disease and Oculomics](https://arxiv.org/abs/2509.03421) | A: official arXiv abstract/record; preprint | Already compares generalist DINO variants with retinal specialists and examines adaptation/label efficiency. No inference about uninspected controls or universal specialist superiority; not a causal explanation of our future gaps. |
| A08 | [Lian et al. (2025), BenchReAD: A systematic benchmark for retinal anomaly detection](https://papers.miccai.org/miccai-2025/0092-Paper1683.html) | T: official conference abstract/metadata and paper section 2 excerpts | Retinal cross-dataset seen/unseen anomaly evaluation is established. Anomaly detection is different from our closed-set five-way classification and nine-way false-rare diagnostic. |
| A09 | [Kulyabin et al. (2024), OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods](https://doi.org/10.1038/s41597-024-03182-7) | T: publisher abstract/dataset description and official Zenodo release/metadata fields | External OCT source with disease/subcategory and patient fields; supports a source-shift inventory. Broad VID/condition labels are not automatically MH; local clinical label equivalence remains gated. |
| A10 | [Gholami et al. (2018; revised 2019), OCTID: Optical Coherence Tomography Image Database](https://arxiv.org/abs/1812.07056) | A: official arXiv abstract/record; cited as this preprint version | Independent OCT source lists NORMAL, MH, AMD, CSR and DR. CSR-to-CSC equivalence and actual release inventory require checks; no all-five-rare external validation. |
| A11 | [Snell, Swersky and Zemel (2017), Prototypical Networks for Few-shot Learning](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html) | A: official proceedings abstract/metadata | Class means in a learned embedding motivate prototype-based classification. Our frozen pretrained embedding plus cosine centroids is not a newly meta-trained Prototypical Network. |
| A12 | [Wang et al. (2019), SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning](https://arxiv.org/abs/1911.04623) | A: official arXiv abstract/metadata | Shows that simple feature transformations and nearest-neighbour baselines deserve serious comparison. Its centering/normalisation recipe is not identical to ours; do not add centering after seeing results. |
| A13 | [Guo et al. (2020), A Broader Study of Cross-Domain Few-Shot Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720120.pdf) | T: official paper benchmark/abstract excerpts and author project record | Includes satellite, dermatology and X-ray targets; highlights limits of within-domain few-shot rankings. Its tested fine-tuning/meta-learning ranking cannot establish a winner for our modern frozen encoders. |
| A14 | [Kumar et al. (2022), Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://openreview.net/forum?id=UYneFzXSJWh) | A: paper abstract, including official ICLR PDF excerpt | LP-FT addresses feature distortion in their transfer-learning settings. No universal sample threshold or guarantee for OCT; full-backbone adaptation stays outside this sprint. |
| A15 | [He et al. (2026), Boosting pathology foundation models via few-shot prompt-tuning for rare cancer subtyping (PathPT)](https://doi.org/10.1038/s41467-026-71715-2) | T: publisher abstract/introduction; indexed primary Methods/training/data-curation passages | Uses learned textual prompts, spatial aggregation and tile supervision for rare-cancer whole-slide classification. Not prompt tuning alone, not OCT, and not proof of scanner/staining invariance or feasibility on our Colab GPU. |
| A16 | [Tai et al. (2025), Improvement Strategies for Few-Shot Learning in OCT Image Classification of Rare Retinal Diseases](https://arxiv.org/abs/2505.20149) | A: official arXiv abstract/record; preprint | A nearby augmentation/balancing/attention extension of the rare-OCT classification problem. Headline accuracy is not comparable without a matched data/split/protocol audit; no new arm implied. |
| A17 | [Iwanicka and Lu (2026 proceedings; MIUA 2025), Diffusion with Adversarial Fine-Tuning for Improving Rare Retinal Disease Diagnosis](https://eprints.whiterose.ac.uk/id/eprint/228674/) | A: author-repository abstract and bibliographic record | Reports class-aware diffusion, attention/adversarial fine-tuning and downstream rare-OCT classification. Full experimental methods not audited here; do not turn its FID/accuracy into clinical or comparative proof. |

## 8. Provenance, citation traps and maintenance

- Coverage reconciles at 20/20 + 29/29 supplied PDFs. These 66 catalogue entries
  are not 66 independent patient cohorts or a PRISMA included-study count.
- Official arXiv API: all 43/43 requested identifiers returned. This checks record
  identity, not local-file equality to the latest revision or peer-review status.
- Crossref: seven direct DOI records retrieved; an OCTA-500 title query returned
  three candidates and the exact-title/DOI match was identified.
- Two Europe PMC queries returned HTTP 503, without result counts. Nature repeat
  opens sometimes failed on cookie redirects; PMC live opens showed browser
  checks. The index records initial publisher/indexed passages and the previously
  archived primary text used instead. Access failures are not negative findings.
- No complete correction/retraction audit was performed. CLIP-Adapter's publisher
  page marks an update whose contents were not audited; check before using
  detailed potentially affected results.
- **Filename traps:** Models Genesis is a **2021 journal article**, despite the
  supplied 2022 label; OCTA-500's matched journal article is **2024**, despite its
  2020 filename. TOP-GAN's journal title uses **Stain-free** (2019), replacing the
  supplied older Label-free wording. CLIP-Adapter was online in 2023, issue 2024.
- Iwanicka/Lu MIUA 2025 and proceedings 2026 identify one paper, not two studies.
  Link preprints and journal versions instead of counting both as independent.
- To add evidence: assign a stable ID, primary URL, date, reading scope, supported
  statement and limitation in the source index; update only the affected
  synthesis. A new paper does not automatically authorise a new experiment arm.
- Preserve supplied PDFs. Absolute local paths are machine-specific; primary URLs
  keep the index useful elsewhere. Use current primary metadata for final BibTeX.

Workflow attribution: literature-review and paper-lookup skills, described by
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner
(2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents](https://arxiv.org/abs/2609.00065), current record checked 5 November 2026.
Tooling acknowledgement only, separate from retinal scientific evidence.

