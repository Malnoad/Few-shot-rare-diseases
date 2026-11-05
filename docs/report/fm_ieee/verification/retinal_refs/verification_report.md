# Retinal bibliography batch: primary-source verification

Verified 5 November 2026 for Lam's proposed-methods draft. Scope: R01, R03, A01, A02, A04–A10 and A15 only. No experiments, manuscript edits, canonical data/results changes, commits or pushes.

All 12 identities are verified. The selected bibliography contains nine journal articles, one conference paper and two explicitly identified arXiv preprints. Complete author names **as published**, their order, exact titles and field-by-field source files are in [field_verification.json](field_verification.json); unexpanded middle initials are retained. [retinal_refs.bib](retinal_refs.bib) contains the 12 requested keys. Article identifiers are identified as such, not invented page ranges.

The main bibliographic qualifications are RETFound's consortium representation, BenchReAD's conflicting citation years, and OCTID's later journal publication. The main claim boundaries are adaptation scope, imaging modality, earlier synthetic-data work and external label equivalence.

## R01 — Yoo2021

Tae Keun Yoo, Joon Yul Choi, Hong Kyu Kim. *Feasibility study to improve deep learning in OCT diagnosis of rare retinal diseases with few-shot classification*. **Medical & Biological Engineering & Computing 59(2), 401–415 (2021)**. [DOI: 10.1007/s11517-021-02321-1](https://doi.org/10.1007/s11517-021-02321-1). Crossref and publisher agree; online 25 January, February issue.

**Passages read:** local publisher PDF, printed pp. 402–405, Methods §§2.1–2.4, with targeted text decoding of the PDF content streams. In particular, p. 404, §2.3 describes separate unpaired CycleGAN translations from NORMAL to each rare disease, basic augmentation and ophthalmologist exclusion of severe artifacts. On p. 405, §2.4: “The backbone convolutional layers of Inception-v3 were left frozen”. Thus the paper's use of *fine-tuning* does not mean backbone updating in this classifier. It is augmentation-assisted nine-class OCT classification, not our proposed strict support-only FM comparison. Its clinician review is different from the reproduction's automatic QC.

**Exact local source:** `D:/Few-shot/retinal imaging/[MBEC 2021] GAN-FSL-OCT - Feasibility Study to Improve Deep Learning in OCT Diagnosis of Rare Retinal Diseases with Few-Shot Classification.pdf`, SHA256 `4611767a28ac3743a147673665ab11b86ecbbf276f9ae963f39fc1f4f1ac7f11`. PDF page objects 40/68, content objects 63/76, contain pp. 404/405. Online publisher access exposed abstract/metadata only. Raw metadata: `raw/R01_crossref.json`, `raw/R01_publisher.html`.

## R03 — Quellec2020

Gwenolé Quellec, Mathieu Lamard, Pierre-Henri Conze, Pascale Massin, Béatrice Cochener. *Automatic detection of rare pathologies in fundus photographs using few-shot learning*. **Medical Image Analysis 61, article 101660 (April 2020)**. [DOI: 10.1016/j.media.2020.101660](https://doi.org/10.1016/j.media.2020.101660).

**Passages read:** official [arXiv:1907.09449v3](https://arxiv.org/abs/1907.09449v3), PDF pp. 1–5, abstract, §3/Fig. 1 and §§3.2, 3.3, 3.7; p. 7, §4.6 evaluation. It trains a CNN on frequent **fundus** conditions and derives rare-condition probabilistic detectors from its representations, incorporating t-SNE and neighbour regression. This is a retinal few-shot precedent, but not an OCT FM or five-way cosine-prototype experiment. Its patient-grouped cross-testing builds reference sets partly from other test folds; do not describe it as our fixed, support-only inductive protocol.

**Version:** first submitted 22 July 2019; v3 revised 10 February 2020 and linked by the official arXiv record to this journal DOI. Bibliography cites the 2020 journal article; methods inspection used the explicitly identified author preprint. Raw records: `raw/R03_crossref.json`, `raw/arxiv_batch.xml`, `raw/R03_arxiv.html`; passage locators in `raw/web_quellec_methods_find.json`: PDF lines 125–209, 418–466, 606–620. Existing local PDF listed by the supplied index has SHA256 `7bf828a706c112dad660e4c324bb756c32d05408e38461d5d49c4e95409bcccb`; these detailed passages were read from official arXiv, not inferred from its filename.

## A01 — Kermany2018

Daniel S. Kermany and 38 coauthors, full order in the JSON/BibTeX. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. **Cell 172(5), 1122–1131.e9 (22 February 2018)**. [DOI: 10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010). Crossref, NCBI PubMed author list and Elsevier's official API record agree.

**Passages read:** publisher-indexed full-text excerpts at [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0092867418301545): Fig. 1 caption; Fig. 2 and Patient and Image Characteristics; STAR Methods → Transfer Learning Methods and Expert Comparisons. They describe CNV, DME, drusen and NORMAL, frozen ImageNet Inception-v3 convolutional features, stored bottlenecks and retraining of the final softmax layer. This supports the dataset/method lineage. The paper's statement of patient-independent testing does not establish independence of the project's actual downloaded split.

**Access limit:** direct Cell/ScienceDirect opens returned HTTP 403; the primary publisher's indexed passages were readable. This is not a claim of a fresh full-paper/PDF audit. Raw passage response: `raw/web_kermany_indexed_primary.json`, use only its ScienceDirect result. Raw metadata: `raw/A01_crossref.json`, `raw/A01_pubmed.xml`, `raw/A01_elsevier.xml` (metadata only).

## A02 — RETFound2023

Yukun Zhou et al. *A foundation model for generalizable disease detection from retinal images*. **Nature 622(7981), 156–163 (2023)**. [DOI: 10.1038/s41586-023-06555-x](https://doi.org/10.1038/s41586-023-06555-x). Online 13 September; October issue.

**Passages read:** live publisher Main (`#Sec1`), Ocular disease diagnosis (`#Sec2`), Methods → Datasets for developing RETFound (`#Sec11`), Adaptation to downstream tasks (`#Sec16`), and Discussion (`#Sec9`); reference 34. The work develops **separate CFP and OCT models**, uses masked-autoencoder retinal self-supervision, and reports downstream fine-tuning and label efficiency. It explicitly says CFP–OCT fusion was not investigated. Methods identifies OCT pretraining data from ref. 34, which is Kermany 2018: disclose that exposure rather than calling the encoder independent of the common-data source. Its downstream results do not establish how our frozen prototypes will perform.

**Author correction:** the rendered primary byline has 17 individuals **plus UK Biobank Eye & Vision Consortium**, between Yuka Kihara and Andre Altmann. HTML citation-author tags omit the group; Crossref expands its 77 members; Springer RIS expands them and places the group last. The provided entry preserves the rendered byline and explicitly includes the consortium. Do not silently treat the expanded membership as the main author sequence.

Raw sources: `raw/A02_publisher.html`, `raw/A02_crossref_retry.json`, `raw/A02_citation.ris`.

## A04 — MIRAGE2025

José Morano et al. *Multimodal foundation model and benchmark for comprehensive retinal OCT image analysis*. **npj Digital Medicine 8(1), article 576 (25 September 2025)**. [DOI: 10.1038/s41746-025-01852-3](https://doi.org/10.1038/s41746-025-01852-3). All ten author names/order agree between publisher and Crossref.

**Passages read:** live publisher Methods → Foundation model architecture and its model/head/training descriptions; Fig. 8; Discussion limitations. Useful exact paragraph starts in `raw/A04_publisher.html`: `The pretraining pretext task consists`; `For the classification experiments, we used`; `For the classification tasks, modality-specific decoders`. Pretraining uses paired OCT/SLO/retinal-layer pseudo-labels. Modality sampling permits **any subset** of inputs downstream, supporting OCT-only inference in principle. Classification uses a **frozen encoder**, averages patch-token features **excluding the global token**, and trains a linear softmax head. Segmentation has separate head/tuning variants, including a full-tuning comparison; do not generalize that to the classification protocol.

ViT-Base and ViT-Large are both specified; these are different configurations. Paper-level OCT-only support does **not** verify our actual Base checkpoint loader, preprocessing, pooling implementation or resource feasibility. Raw metadata: `raw/A04_crossref_retry.json`.

## A05 — EyeCLIP2025

Danli Shi et al. *A multimodal visual–language foundation model for computational ophthalmology*. **npj Digital Medicine 8(1), article 381 (21 June 2025)**. [DOI: 10.1038/s41746-025-01772-2](https://doi.org/10.1038/s41746-025-01772-2). All 15 authors agree between publisher and Crossref.

**Passages read:** live publisher abstract, Introduction, Results → ocular classification (`#Sec4`, Figs. 3–4 and rare-disease paragraph), and Methods → Downstream Validation Details (`#Sec16`). Multimodal images and clinical text are aligned by reconstruction and contrastive objectives. The study already tests zero-shot and few-shot ophthalmic classification, including OCT datasets and a separate rare-disease Retina Image Bank subset. The latter spans modalities; it is not this project's same five-class OCT experiment.

Methods defines few-shot classification via fine-tuning; full-data classification freezes the encoder for five epochs and then unfreezes it. This is different from an entirely frozen prototype evaluation. Methods/Fig. 3 list shots 1/2/4/8/16; nearby Results prose instead says 6, an internal reporting inconsistency if an exact shot grid is quoted. Avoid “first rare-eye FM application.” Sources: `raw/A05_publisher.html`, `raw/A05_crossref.json`.

## A06 — EyeDiff2026

Ruoyu Chen et al. *Boosting foundation models for rare eye disease diagnosis via a multimodal text-to-image generative framework*. **npj Digital Medicine 9(1), article 371 (24 March 2026)**. [DOI: 10.1038/s41746-026-02560-2](https://doi.org/10.1038/s41746-026-02560-2). All nine authors agree.

**Passages read:** live publisher Results → multiclass/rare disease diagnosis (`#Sec8`, `#Sec10`); Methods → training/validation data (`#Sec14`, `#Sec15`), architecture/development (`#Sec17`, `#Sec18`) and downstream applicability (`#Sec21`). EyeDiff is a multimodal text-conditioned diffusion framework. Generated images supplement real training images for RETFound, EyeFound and EyeCLIP, including OCTID/OCTDL and rare-disease tasks. This is direct prior synthetic-plus-retinal-FM work; no “first-ever” claim is supportable for that combination.

The paper's generator data, augmentation budgets and downstream tasks differ from the proposed support-only comparison. Its rare-disease collections are multimodal; do not label every rare-disease result an OCT result. The inspected downstream passage does not specify a complete encoder freeze/unfreeze schedule, so that detail remains **unverified** here. Sources: `raw/A06_publisher.html`, `raw/A06_crossref.json`. Live full text was accessible this time; reliance on an older archive was unnecessary.

## A07 — Generalist2025

Yukun Zhou and 22 coauthors. *Generalist versus Specialist Vision Foundation Models for Ocular Disease and Oculomics*. **arXiv:2509.03421v1, submitted 3 September 2025**. [Official version](https://arxiv.org/abs/2509.03421v1), [arXiv DOI](https://doi.org/10.48550/arXiv.2509.03421).

**Read scope:** official abstract, complete author list and submission/version history; PDF title/byline page only. The abstract explicitly compares DINOv2/DINOv3 with RETFound-MAE/RETFound-DINOv2 using **fine-tuning and linear probing**, including data/adaptation efficiency. Therefore a generalist-versus-retinal-specialist comparison or label-efficiency analysis is not new by itself. This abstract-level inspection does not establish their complete controls, OCT-specific task coverage, or universal specialist superiority.

The API currently returns v1, with no journal reference/linked journal DOI. That absence is **not** proof that no publication exists. Cite the verified preprint version, with no invented venue, volume or pages. Methods PDF follow-up searches timed out; HTML conversion returned 404. Sources: `raw/arxiv_batch.xml`, `raw/A07_arxiv.html`, `raw/A07_datacite.json`, `raw/web_generalist_pdf_and_discovery.json` (primary arXiv results only).

## A08 — BenchReAD2025

Chenyu Lian, Hong-Yu Zhou, Zhanli Hu, Jing Qin. *BenchReAD: A systematic benchmark for retinal anomaly detection*. **MICCAI 2025, Lecture Notes in Computer Science 15961, 35–45**. [Official conference record](https://papers.miccai.org/miccai-2025/0092-Paper1683.html); [DOI: 10.1007/978-3-032-04937-7_4](https://doi.org/10.1007/978-3-032-04937-7_4). Author order, volume and page range verified.

**Year qualification:** conference-provided BibTeX and Crossref give **2025**, and Springer says first online/published 20 September 2025. Springer also currently gives **2026** in its citation-publication metadata, copyright and suggested citation. The supplied BibTeX follows the **2025 official conference version**, not a silently reconciled 2026 chapter citation. Parent should retain/resolve this explicit discrepancy if choosing Springer's recommended chapter citation.

**Passages read:** official conference abstract and linked open-access PDF, pp. 1–4, particularly §2.1 on p. 3. It trains the OCT benchmark on OCT2017 and evaluates OCT2017, OCTDL and OCTID, with seen/unseen anomalies. This is a direct retinal cross-source evaluation precedent. Its anomaly-detection objective is different from naming five rare classes or computing nine-way NORMAL false-rare rate. Do not equate its “seen” disease mappings with clinically verified label equivalence for our task.

Sources: `raw/A08_miccai.html`, `raw/A08_springer.html`, `raw/A08_crossref.json`, `raw/A08_paper.pdf`; PDF text locator: `raw/web_primary_pdf_open.json`, lines 51–106.

## A09 — OCTDL2024

Mikhail Kulyabin et al. *OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods*. **Scientific Data 11(1), article 365 (11 April 2024)**. [DOI: 10.1038/s41597-024-03182-7](https://doi.org/10.1038/s41597-024-03182-7). All 11 journal-author names/order agree between publisher and Crossref. Use the **journal** author sequence: the Zenodo depositor sequence differs.

**Passages read:** live publisher abstract; Background; Methods → VID (`#Sec9`); Data Records (`#Sec11`); Technical Validation (`#Sec12`). These support a separate OCT acquisition source and distinguish disease groups from pathological conditions. Technical Validation explicitly describes using OCTDL to test a Kermany-trained model: an additional source-shift precedent, not validation of this project's classifiers.

**Release verification:** [Zenodo record 10839556](https://doi.org/10.5281/zenodo.10839556), metadata and **label CSV only**, no image archive downloaded. The seven disease labels include NO and VID. Among 76 VID rows, `condition` is MH throughout, but `subcategory` is MH 39 / VMT 11 / LH 26. Neither the broad VID group nor the broad MH condition establishes 76 equivalent target-class MH cases. Clinical mapping/image eligibility remains gated. This source cannot supply all five rare categories.

Sources: `raw/A09_publisher.html`, `raw/A09_crossref.json`, `raw/A09_zenodo_10839556.json`, `raw/A09_labels.csv`, `raw/A09_repo_README.md`. CSV SHA256 `b7db437cfe67741623f26e8bb2cef0f4d6828d6d86150f752f412da1c4ee8b3a`; actual header spells `image_hight`, unlike the article's `image_height`.

## A10 — OCTID2018

Peyman Gholami, Priyanka Roy, Mohana Kuppuswamy Parthasarathy, Vasudevan Lakshminarayanan. *OCTID: Optical Coherence Tomography Image Database*. **arXiv:1812.07056**, first submitted 17 December 2018; **v2 revised 27 May 2019**. [Official v2](https://arxiv.org/abs/1812.07056v2), [arXiv DOI](https://doi.org/10.48550/arXiv.1812.07056).

**Version correction:** a journal version exists: *OCTID: Optical coherence tomography image database*, **Computers & Electrical Engineering 81, article 106532 (January 2020)**, [DOI: 10.1016/j.compeleceng.2019.106532](https://doi.org/10.1016/j.compeleceng.2019.106532). Its DOI's 2019 component is not the volume year. The requested `OCTID2018` entry intentionally retains the preprint, with its revision noted. For a final journal-preferred bibliography, parent may switch this work to the 2020 record; do not mix 2018 with journal volume 81.

**Read scope:** official arXiv abstract/version record, Crossref journal metadata, and official OCTID dataverse/citation record. The source identifies NO, MH, AMD, CSR and DR. It does not offer MacTel, Stargardt or RP classes; CSR terminology alone does not settle local CSR→CSC eligibility. Image inventory and clinical equivalence were not audited. Sources: `raw/A10_arxiv.html`, `raw/arxiv_batch.xml`, `raw/A10_datacite.json`, `raw/A10_journal_crossref.json`, `raw/A10_dataset_W43PFI.json` (citation dataset only).

## A15 — PathPT2026

Dexuan He et al. *Boosting pathology foundation models via few-shot prompt-tuning for rare cancer subtyping*. **Nature Communications 17(1), article 5058 (11 April 2026)**. [DOI: 10.1038/s41467-026-71715-2](https://doi.org/10.1038/s41467-026-71715-2). All 15 authors agree between publisher and Crossref.

**Passages read:** live publisher abstract; Methods → few-shot definition (`#Sec12`), architecture (`#Sec13`), tile supervision (`#Sec14`), loss (`#Sec15`) and training (`#Sec17`). Vision and text encoders remain frozen, but **learned text prompts and trainable spatial visual aggregation** are combined. Local convolutional interactions and a global transformer operate on spatially ordered tiles. Training also uses tile pseudo-labels conditioned on slide labels, a manually constructed/selected prompt pool, and several losses. Thus the reported gain is not evidence for prompt tuning alone. Its support unit is a whole slide containing many tiles, not an OCT B-scan.

Use as a cross-medical adaptation precedent, not retinal validation, evidence of scanner/stain invariance, an isolated module effect, or proof of Colab feasibility. Source: `raw/A15_publisher.html`, `raw/A15_crossref.json`.

## Reuse, limits and checks

- `retrieval_manifest.json` records exact requested/final URLs, status, retrieval time and raw-response SHA256. Failed/rate-limited responses are retained; two Crossref retries succeeded. `raw/*.json`, XML, RIS and HTML are retrieved evidence; web-tool JSON is a raw tool response, not a new publisher API.
- Publisher metadata and named passages were checked, not all full papers, supplements, code, correction notices or retraction history. No unsupported absence-of-control, performance-superiority or clinical-benefit claims follow from this batch.
- The citation-management validator parsed all 12 entries with no duplicates. Its single reported error is a **reviewed false positive**: it treats the correctly escaped ampersand inside the protected corporate author `{UK Biobank Eye \& Vision Consortium}` as an author separator. Its six warnings request page ranges for records that use article identifiers, already rendered explicitly as `Art. no.` notes. The raw validator result is retained as `bibtex_validation.json`; no claim that this generic validator returned an unqualified pass is made.
- A guessed Zenodo ID returned unrelated **cuNMF software**. It is retained as `raw/excluded_unrelated_zenodo_10844444.json`, explicitly excluded in the manifest, and never used for OCTDL claims. The replacement release was selected from the existing source archive and its title/DOI checked. Proposed prevention: require record-title/DOI identity checks before labeling dataset evidence; no repository guard change applied.
- An inspection command repeated the repository's known Windows stdout-encoding failure; it was rerun with UTF-8. The existing proposed UTF-8 wrapper/round-trip check remains the practical fix. No `.agents/mistakes.md` edit or guard installation was made because this task explicitly owns only this directory; these operational notes preserve the facts for the parent.
- Pre-completion surfaces: stage/notebook entry points, reverse actions, feature-cache schemas and project method/data/results docs are **not applicable**: none changed. Bibliography/evidence artifacts are the only deliverable. Repository `.agents/check` was not run: it explicitly compiles/tests project files and can write outside the sole authorized directory. Artifact-only identity, key, hash and scope checks are recorded in `verification_checks.txt`; this is not a claim that repository checks passed.

Parent decisions remaining: preserve or change the OCTID preprint citation; choose the explicit BenchReAD conference/chapter citation convention; validate actual FM checkpoint loading and external clinical label mappings in their own implementation/protocol work. No experiment is proposed or authorized by this report.
