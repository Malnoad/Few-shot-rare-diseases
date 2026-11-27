# Proposal for rebuttal: adopt prior studies' label practice instead of an expert check

> **Closed record (13 November 2026).** Kept as evidence of how decisions were reached; no longer updated. The study as it stands is in [`protocols.md`](../../protocols.md); the binding texts are `FM_PROTOCOL.md` and `FM_AMENDMENTS.md`.

Draft by the agent, 12 November 2026, for adversarial review. **Adopted in revised form as amendment A1 (`FM_AMENDMENTS.md`), 12 November 2026.** Lam's position:
"either stick to specific details that need expert or use a reliable study that help that
simplified our work"; focus on the technical side; no clinician is available before the lock.

## Background (verified facts)

- Study: frozen-encoder few-shot classification of five rare OCT diseases (CSC, MH, MacTel,
  Stargardt, RP) from a benchmark released with Yoo et al. 2021 (Mendeley btv6yrdbmv), plus a
  NORMAL false-rare audit. Protocol v1.0 section 8 currently says: "CSR-to-CSC needs clinical
  label-equivalence confirmation" and "Metadata alone is not clinical certification".
  Without it: keep NORMAL FRR and omit the rare-transfer claim.
- External candidates and counts: OCTID CSR 102, OCTID MH 102, OCTDL VID with subcategory MH
  39 images from 28 patients. OCTDL also has LH 26 and VMT 11, which its `condition` column also
  labels "MH". Benchmark rare counts: CSC 32, MH 31, MacTel 29, Stargardt 23, RP 31.
- How prior peer-reviewed studies matched labels across datasets, with no new clinical
  adjudication:
  - TVST 2026 (doi:10.1167/tvst.15.1.6, PMC12786394) used OCTID for external validation and
    states: "OCTID includes four diagnostic categories: AMD, CSC, FTMH, and Healthy. To ensure
    compatibility with our seven-class internal model, we considered only dAMD predictions as a
    single AMD category ... following the dataset's labeling structure."
  - MIRAGE (Morano et al., npj Digit Med 2025, PMC12462498) ran cross-dataset classification,
    tuning on Noor Eye Hospital (AMD, DME) and evaluating on UMN (AMD, DME) plus Duke Srinivasan
    (DME, iAMD), with linear probing on frozen encoders; it also benchmarks OCTID (Control, MH,
    AMD, CSR, DR) and OCTDL with the datasets' own labels.
  - EyeDiff (doi:10.1038/s41746-026-02560-2) lists OCTID's classes as "MH, CSCR, AMD DR, Normal".
- Dataset documentation: OCTID's paper describes its CSR class under the heading "Central Serous
  Chorioretinopathy (CSC)"; labels "based on the diagnosis of retinal clinical experts". It
  describes MH as "small hypoflective breaks to full-thickness intra-retinal spaces". OCTDL
  defines MH as "a complete defect in the inner layers of the retina that extends to the RPE";
  two specialists labelled with consensus and the clinic head confirmed every diagnosis. MeSH
  D056833 lists "Central Serous Retinopathy" as an entry term of "Central Serous
  Chorioretinopathy". IVTS (Duker et al. 2013) defines FTMH as interruption of all layers from
  ILM to RPE.
- Technical confounds: benchmark crop size differs by rare class (median width CSC 414, MH 327,
  MacTel 482, Stargardt 385, RP 393 px; all RGBA crops with black padding); OCTID images are
  500x750 with a scan-orientation icon; OCTDL images are wide (about 740-1100 x 265-410).
  An earlier audit found frozen features separate rare from common images with AUC about 0.995,
  unchanged by format/framing normalization (0.991).

## The proposal (a dated amendment before the input lock)

1. **Label rule: dataset labels as published, harmonized by name as prior studies did.**
   OCTID CSR -> CSC; OCTDL `disease=VID` and `subcategory=MH` -> MH; OCTID MH -> MH, reported as
   a separate cohort "MH (OCTID definition)". LH, VMT, AMD, DR, DME, ERM, RAO, RVO are not mapped.
   Claims are worded "dataset-label transfer, following the label practice of [TVST 2026,
   MIRAGE 2025]", never "clinically confirmed". Whole classes only; no per-image relabelling.
2. **Technical design that holds source constant.** Primary external rare test: two-way CSC
   versus MH on OCTID queries (102 + 102, one device, one size), with K-shot supports from the
   benchmark's CSC and MH only, using the protocol's prototype and probe heads. Because both OCTID
   classes share one source, the source cannot separate them. Secondary: MH-only output shares
   on OCTDL MH (grouped by patient); the five-way head's outputs on all external rare images
   reported as output shares, not recall claims.
3. **Leakage and dependence:** the existing duplicate/near-duplicate screen and patient
   grouping (OCTDL patient IDs) apply; OCTID has no patient IDs, so intervals are image-level and
   labelled conditional.
4. **Diagnostics, no tuning:** a source probe (how well features predict source) is reported
   beside the transfer results; nothing is adjusted after seeing them.
5. **Fallback:** if the mentor or a clinician later rejects a mapping, that cohort's transfer
   claim is withdrawn in a dated note; its results stay archived and are reported as withdrawn.

## What the reviewers are asked

Attack this proposal. Is name-harmonization following prior studies a defensible substitute for
an expert check for a student proposal paper? What breaks? Then give the best "way out":
the simplest rule that keeps the work honest and technically sound without a clinician.

## Revised after the rebuttal (12 November 2026), for Lam's approval

Reviewers: GPT-6.1-Sol (Codex) and Gemini 4 Argon High, independently. Both: "accept with
changes"; both reject using prior studies as proof of clinical equivalence. Not adopted until
Lam approves.

**Rule: published dataset labels, used as evaluation labels only.**

| Cohort | Images | Tier | Use |
|---|---|---|---|
| OCTDL `disease=VID`, `subcategory=MH` | 39 (28 patients) | Definition-matched: OCTDL's definition matches IVTS full-thickness MH | Query-only; grouped by patient |
| OCTID CSR | 102 | Synonym-matched (OCTID's own CSC heading, MeSH D056833); spectrum not verified | Query-only |
| OCTID MH | 102 | Broad: OCTID's own definition covers partial to full-thickness defects | Query-only; separate cohort, never pooled with OCTDL MH, never called FTMH |
| OCTID and OCTDL NORMAL | 206 and 332 (110 patients) | Unchanged | Nine-way NORMAL FRR, as in protocol v1.0 |
| OCTDL LH, VMT; OCTID AMD, DR; OCTDL AMD, DME, ERM, RAO, RVO | — | — | Excluded |

Changes from the draft:
1. **Prior studies are cited as precedent for evaluating published categories, not as evidence
   of equivalence.** The benchmark's own MH and CSC definitions stay "unverified" (both reviewers).
2. **External rare results are exploratory and descriptive.** No external image enters fitting.
   MacTel, Stargardt and RP get no external claim.
3. **OCTID CSC-versus-MH test: cosine prototypes only.** The draft's logistic probe is dropped
   for this test (Gemini: intercept and support-geometry asymmetry; GPT: a binary head would
   have to be specified separately). Report per-class recall, macro recall, confusion counts and
   denominators for every locked K and arm, including unfavourable ones.
4. **No claim that source shortcuts are removed.** The query source is constant, but benchmark
   supports keep class-linked crop geometry (CSC about 414 px, MH about 327 px wide) that can
   pull OCTID queries toward one centroid. Record geometry and padding by class and source, and
   state the residual risk. The source probe is a diagnostic, not a gate.
5. **The post-score fallback is dropped.** Eligibility is locked before inference. If the
   mentor or a clinician later questions a mapping, that becomes a stated limitation; scored
   results are never withdrawn (Gemini: withdrawal after scores enables selective reporting;
   GPT: it leaves equivalence implicitly accepted until challenged).
6. **Intervals:** OCTDL patient-cluster intervals (5,000 draws, seed 0, as protocol v1.0);
   OCTID intervals are image-level and labelled conditional.
7. Leave-one-source-out is dropped (GPT: not needed). Optional: one fixed crop that masks the
   OCTID orientation icon as a sensitivity check.

**Claim wording:** "We evaluated agreement with OCTID's and OCTDL's published CSC/CSR and MH
labels using benchmark-only K-shot supports. Clinical label equivalence, patient independence
and freedom from acquisition shortcuts were not established."

Agent's corrections to the reviews: OCTID selects one fovea-centred B-scan per volumetric scan
(OCTID paper), so "repeated B-scans per eye" is unverified; patient repetition across scans is
unknown because OCTID gives no patient IDs. K supports are balanced per class, so the probe
concern is support geometry, not support count.

This replaces protocol section 8's "clinical label-equivalence confirmation" for these cohorts
only, as a dated amendment before the input lock.
