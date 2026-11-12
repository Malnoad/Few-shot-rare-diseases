# Research calendar: 5 November–3 December 2026

> **Closed record (13 November 2026).** Kept as evidence of how decisions were reached; no longer updated. The study as it stands is in [`protocols.md`](../../protocols.md); the binding texts are `FM_PROTOCOL.md` and `FM_AMENDMENTS.md`.

**Active plan, updated 2026-11-05.** Two deliverables govern this calendar:
**an IEEE-style draft report, 4–5 pages, due 11 November**, followed by the
**revised report and reproducible experiment package by 3 December**.
All dates/times below use Asia/Bangkok. Submission times have not been supplied;
finish each PDF the previous day. The period is 29 calendar days, with roughly
three weeks for implementation/experiments after the first report.

This replaces the relative 14-day schedule previously in this file. The stable
filename is retained for existing links. [The full research plan](FM_FEWSHOT_PLAN.md)
retains source checks, the 29-paper index, protocol details and rebuttals.
This request is planning: no IEEE manuscript, implementation or new results are
claimed complete. Existing stage/C1–C3 protocols and results are preserved.

## 1. Deliverables and priority

| Deadline | Required deliverable | What counts as complete |
|---|---|---|
| **11 Nov, Wednesday** | **4–5 page IEEE draft** | Problems, related work, proposed methods and discussion; results optional. Compiled/readable PDF plus editable source and checked citations |
| **19 Nov** | Development decision | All available encoders compared under the frozen development protocol; one working model nominated, all scores retained |
| **27 Nov** | Experiment freeze | Query-level predictions, metrics, figures and measured compute captured; incomplete arms explicitly identified |
| **3 Dec, Thursday** | Revised report + evidence package | Findings integrated into the report; source/PDF, prediction provenance, plots, configuration and limitations checked |

The 11 November report is viable without new FM results. Put the research question,
nearest prior work, executable proposed protocol and limitations first. New results
must not sit on the critical path to that deadline.

Working assumption for 3 December: revise the same report and deliver its evidence
package; retain the 4–5 page target unless the course specifies another final length.
This calendar reserves the final six days for analysis/writing/verification.

## 2. IEEE draft blueprint: target 4.5 pages, up to 4.8 with results

Use the instructor's IEEE template if specified. Otherwise assume an English,
two-column IEEE conference template, conservatively counting title, figures,
tables and references within the 4–5 pages. The 4–5 page limit comes from Lam,
not a universal IEEE rule. IEEE supplies official Word/LaTeX templates:
[IEEE Author Center](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/)
(checked 2026-11-05).

**Working title:** “Evaluating Few-Shot Foundation Models for Rare Retinal Disease
Classification in OCT.”

**Argument to develop:** scarce rare-class labels motivate pretrained features;
the current collection also confounds disease with acquisition source. Compare
three frozen encoders with matched simple classifiers and explicit information
budgets, then test transport on verified external subsets. At the draft stage,
this is a proposed evaluation, not demonstrated superiority.

| Report component | Page budget | Content and completion criterion |
|---|---:|---|
| Title, abstract, keywords | 0.35 | Problem, research question, proposed comparison and scope. State whether new results exist; write the abstract last |
| I. Introduction / Problem Statement | 0.75 | Scarce labels; Kermany/Yoo setting; acquisition/source concern; one precise question and bounded objectives |
| II. Related Work | 0.80 | Three thematic paragraphs: generative augmentation; retinal/general FMs; realistic few-shot evaluation. End each with its relevance to this study |
| III. Proposed Methods | 1.45 | Data/splits, encoder choices, support/query design, prototype and L2 heads, selection rule, metrics, external checks and compute plan. Include one compact pipeline figure within this budget |
| IV. Results / Preliminary Observations, optional | +0.30 | Only verified observations available by the cutoff; identify model, data, K, protocol and limitations. Omit the section if none qualify |
| Discussion | 0.60 | What the design can establish; nearest-prior overlap; unknown patients, small test, source confounding and partial external label coverage; feasible next steps |
| Conclusion | 0.10 | Research question and bounded contribution/status |
| References | 0.45 | Approximately 8–12 relevant primary sources; sufficient citations take priority over an arbitrary reference count |
| **Total** | **4.50 / 4.80** | Actual layout must be checked by compiling; page budgets are targets, not measured page counts |

If results are omitted, Discussion becomes Section IV and Conclusion Section V.
If included, renumber them automatically. Do not leave an empty Results section
or fill it with expected accuracies. Reserve about 0.2 page below the maximum for
layout variation. Keep the daily schedule and detailed run inventory in this
repository, not in the 4–5 page paper.

### Paragraph and evidence map

- **Problems:** two or three paragraphs narrowing from scarce OCT labels to the
  specific disease/source ambiguity, then the question addressed by this design.
  Cite Yoo and Kermany; distinguish scarce dataset labels from epidemiological rarity.
- **Related work:** augmentation and Yoo's benchmark; RETFound/DINOv2/MIRAGE;
  EyeCLIP/EyeDiff and CLAP's evaluation lesson. Explain overlap rather than claiming
  a first rare-eye FM application. Use the full plan's verified source links.
- **Proposed methods:** define five-way classification first; show one pipeline:
  fixed images → native preprocessing → frozen encoder → support-only classifier
  → held-out queries. Then specify paired draws, development nomination, nine-way
  external-NORMAL FRR and measured cost.
- **Optional results:** one small table or figure, with traceable input/prediction
  files. Historical Inception results are background evidence and must be labelled
  separately from any new FM experiment.
- **Discussion:** interpret the design or actual evidence, connect it to prior
  work, explain which claims remain unsupported, and identify the next resolving
  test. Do not interpret a source-axis AUC as causal proof of a shortcut.
- **Terminology:** OCT = optical coherence tomography; FM = foundation model;
  macro recall = balanced accuracy here; FRR = false-rare rate. A fitted cosine
  prototype classifier is not a trained prototypical network. Use these names
  consistently with CONTEXT and the full plan.

| Claim in draft | Evidence location | Allowed status on 11 Nov |
|---|---|---|
| Current rare development/test counts and split | data/manifest.csv; CONTEXT | Observed dataset description: 124 development, 22 legacy tests |
| Existing replication motivates further evaluation | results/seed_check.csv; L7/REPLICATION | Historical results only; no assertion that five seeds establish GAN superiority |
| Proposed FMs/classifiers and resource bounds | Full plan §3 and the fixed roster below | Proposed until implemented/verified |
| A particular FM improves performance or generalizes externally | New prediction files under the locked protocol | Needs evidence; omit if unavailable |
| A patient-independent or all-five-disease clinical conclusion | Not established by the current data | Unsupported; do not claim |

## 3. Daily plan before the 11 November deadline

| Date | Main work | Concrete output / decision |
|---|---|---|
| **Thu 5 Nov** | Fix report question and section outline; select IEEE template; assemble existing evidence and citation shortlist | Title + paragraph outline + empty compiling report structure; list course-format/time assumptions |
| **Fri 6 Nov** | Write Problems and Related Work; check checkpoint terms/access without delaying writing | Complete first two sections; three-theme literature matrix; settle inaccessible-checkpoint fallback before outcomes |
| **Sat 7 Nov** | Write Proposed Methods and draw the pipeline; specify data and evaluation boundaries | Methods text + figure + compact protocol table; each implementation-dependent item marked planned |
| **Sun 8 Nov** | Write Discussion/Conclusion; assemble and compile the entire draft | Draft v1 already within 4–5 pages, with all mandatory sections present; no dependence on new results |
| **Mon 9 Nov** | Check claims/citations and obtain early author/supervisor feedback if available; finish abstract | Draft v2. **18:00 cutoff for optional results**; otherwise finalize the report without new FM results |
| **Tue 10 Nov** | Resolve feedback, compress text, check figures/references and PDF layout | **18:00 PDF/content freeze**; editable source and frozen PDF saved; all placeholders resolved or clearly stated as proposed work |
| **Wed 11 Nov** | Final page/readability/file checks and submission by Lam | Deliver IEEE draft before the actual course cutoff; preserve exactly the submitted PDF and source snapshot |

**Effort rule for this week:** complete the report first. Model access and a tiny
technical pilot may run in parallel only if they leave the full draft ready on
8 November. Do not spend the last two days debugging a backbone for optional results.
If feedback does not arrive in time, complete the internal evidence/layout review
and keep the submission date.

Lam remains author and submits the report. The agent helps with outlines, prose,
citations, figures and verification within the requested task; no supervisor
messages or external submission are authorized by this plan.

## 4. Implementation, experiments and final report after the draft

| Dates | Work | Exit criterion |
|---|---|---|
| **12–14 Nov** | Minimal extractor/evaluator and Colab setup; verify weights/preprocessing/pooling, duplicates/groups, external inventory and label definitions; freeze protocol and episode rosters before labelled pilot scores | Working encoders, valid group counts, versioned cache identity, and representative 100-image timing pilot per FM. Record unresolved items honestly |
| **15–18 Nov** | Extract only required real-image features; run paired development episodes; record hardware, memory and time | All-model prototype development table, conditional support variability, secondary K=4 probe where completed |
| **19 Nov** | Apply the predeclared score/tie rule and lock the reporting recipe | One documented working-model nomination; all-model scores retained, or an explicit tie/failure. No new checkpoint search |
| **20–23 Nov** | Apply locked methods to the legacy test and eligible external sets; run matched nine-way source diagnostic | Saved query predictions, per-class recall/confusions and per-source FRR with NORMAL and rare recall |
| **24–26 Nov** | Audit outputs and analyse paired differences; make K-shot curves and external-source plots; finish compute ledger | Traceable tables/plots with correct denominators and uncertainty; no episodes treated as new patients |
| **27 Nov** | Review missing arms and errors; freeze experiments | Final result manifest and a list of completed/failed/not-run conditions; no selective removal of poor models |
| **28–30 Nov** | Integrate actual findings into the IEEE report; update Methods to what ran and revise Discussion/abstract | Complete revised report v1; proposed, completed and deferred work clearly distinguished |
| **1 Dec** | Scientific/evidence audit: numbers, citations, class order, provenance, fair comparisons and limitations | Every main empirical claim traceable to saved artifacts; contradictions resolved or disclosed |
| **2 Dec** | Final proofreading/layout and reproducibility check; save delivery copy | Final source/PDF and experiment bundle frozen; no new experiments |
| **3 Dec** | Deliver report and evidence package; archive versions and remaining questions | Final handoff; exact submission/sharing route and time follow course instructions |

No new GAN training/generation, C1–C3 sweeps, adapters or full backbone fine-tuning
enter this calendar. Extra elapsed time protects implementation and analysis, not
an expanding experiment matrix.

## 5. Fixed experiment roster and stop rules

The full plan's label budgets, grouping, preprocessing, cache provenance and
uncertainty rules apply. Retain the bounded roster established on 4 November:

- **Encoders:** RETFound Nature OCT, DINOv2-Base and MIRAGE-Base, plus frozen
  Inception-v3 reference features. Use BioMedCLIP as the proposed technical
  replacement for inaccessible RETFound, before outcome inspection. Record the
  exact final lineup; unresolved loading is not a fourth-model search.
- **Development:** five fixed grouped folds; K={1,2,4,8}; 100 paired nested draws
  for prototypes. L2 logistic probe: K=4 and the first 25 matching draws/fold.
  Use prototypes only for the fixed development nomination score.
- **Reporting:** prototypes at K={1,2,4,8,16}, 100 paired nested draws where group
  feasibility permits; L2 probes use the first 25 matching draws/K. Compare
  classifier differences on shared draws. All-data ceilings are optional.
- **Nine-way audit:** one seed-42 common roster capped at 1,000 fitting images/class,
  plus eligible real rare fitting images. Same cosine-prototype rule and rows for
  every encoder; report NORMAL recall/common confusions and rare recall with FRR.
- **External data:** separate NORMAL sources; verified MH labels with all five
  outputs retained; CSR→CSC only after label-equivalence confirmation. In OCTDL,
  condition=MH selects all 76 VID scans; subcategory distinguishes 39 MH / 26 LH /
  11 VMT. This metadata observation does not establish clinical equivalence.
- **Data size:** before exclusions, 4,000 common fitting + 1,000 common tests + 146
  rare = 5,146 images/encoder, plus the locked eligible external subset.
  No full common validation or synthetic pool extraction is needed.
- **Counting:** four encoders give 8,000 prototype / 500 logistic development fits;
  five feasible reporting K values give 2,000 prototype / 500 logistic fits.
  These small-classifier counts are not encoder training runs or independent patients.

**Decisions with deadlines:**

1. **6 Nov:** resolve checkpoint access or record a fixed technical fallback.
   Resolve exact IEEE template/page counting/submission time when available; use
   the conservative assumptions above meanwhile.
2. **9 Nov 18:00:** optional-results cutoff. A report with no new results remains
   complete when its proposed design and discussion are coherent and honest.
3. **14 Nov:** group, source and loading feasibility gate. If any required fold
   lacks all five query classes or eight fitting groups/class, stop and amend the
   whole selector before scores. Do not salvage a favourable subset of folds.
4. **19 Nov:** nomination and reporting lock. The selected score is development
   evidence, not unbiased confirmation of a winning model's generalization.
5. **23 Nov:** if time is slipping, drop optional all-data ceilings and secondary
   probes first. Preserve paired primary prototypes, provenance and the external
   audit. Any reduced roster is disclosed, with the reason and timing.
6. **27 Nov:** experiment freeze. Failed access/ambiguous labels remain limitations.
   Missing external data means a collection-specific pilot, not evidence of transport.
   Do not hold either report deadline waiting for a positive result.

## 6. Compute, resources and completion checks

Colab Pro remains the GPU ceiling; use one available GPU session at a time, with
no dependency on a specific accelerator. GPU work is frozen embedding extraction;
small classifiers/plots run on CPU. Record actual throughput and memory in the
12–14 November pilot, then allocate resumable sessions. GPU hours and compute-unit
cost remain unmeasured; hardware availability/runtime dynamics follow the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html).
Keep source data read-only, copy working inputs to runtime-local storage, and
verify persisted output copies.

**11 November acceptance:** 4–5 compiled pages; all requested sections; evidence
for every cited factual claim; results omitted or clearly sourced; legible pipeline
figure; consistent terms; checked references; editable source + submitted PDF.
Check the rendered PDF, not a word-count estimate. The existing
[Report.tex](../report/Report.tex) is an 11pt article, not the new IEEE draft.

**3 December acceptance:** updated report/source/PDF, frozen configurations and
episode/image IDs, checkpoint/preprocessing identities, query predictions,
K-shot and external-source tables/plots, measured compute, and explicit failures/
limitations. Keep unknown rare patients, historical test inspection, common-patient
overlap, and partial external disease coverage visible.

Writing assistance follows the generic scientific-writing workflow; it does not
impose a Nature format on IEEE. Tooling attribution and research source provenance
remain in the [full plan](FM_FEWSHOT_PLAN.md#appendix-b-search-scope-attribution-and-verification).
No new results were generated in making this calendar.

Planning verification, 2026-11-05: date/weekday and page-budget arithmetic, local
document links, manifest hash and data/results/report preservation passed.
Entry points, reverse actions, feature schemas and executable README/COLAB
instructions are unchanged; CONTEXT/full-plan pointers are updated. Existing
repository harness output:

```text
Identity check: PASS (0 outgoing commits; author and committer match local noreply)
Agent checks: PASS
```
