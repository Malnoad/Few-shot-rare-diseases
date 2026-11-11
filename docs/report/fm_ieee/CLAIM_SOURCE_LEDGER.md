# Claim-to-source ledger

Manuscript: `main.tex`; invisible `\claim{C01}`–`\claim{C31}` anchors identify paragraphs. Reviewed 5 November 2026. The abstract and Fig. 1 summarize C04 and C11–C31; they assert no new outcomes. This is a targeted narrative synthesis, not an exhaustive/systematic literature review.

**Evidence states:** H = archived historical observation; L = primary literature; P = proposed operation/decision; B = explicit interpretation boundary. P never means implemented, registered, measured or validated.

**Controlling method source:** [protocol v1.0](verification/protocol_v1_0.md), SHA256 `4e8df8d91c0021187c05b103a9b0a0e0319a055fd91108da1f4cf3045d00c5df`. Original absolute path and source-index provenance are in [input_provenance.json](verification/input_provenance.json). The user-specified knowledge base and index are preserved as snapshots under `verification/`. Protocol sections, rather than older planning documents or the digest, control the proposed settings.

| Anchor | Manuscript claim / decision | State | Source actually opened and locator | Boundary retained |
|---|---|---|---|---|
| C01 | Common OCT transfer-learning precedent; five rare categories motivate the question | L/P | Kermany2018, publisher summary/transfer-learning description; Yoo2021, abstract and methods; protocol §1 | No clinical effectiveness or current-model performance inferred |
| C02 | Historical nine-way Inception pipeline and Table I accuracy | H/B | `stage3_classifier.py` loading/backbone/head; `results/stage4/metrics.csv` rows none/basic/gan; `results/seed_check.csv` all 15 rows; protocol §10 | Nine-way full-budget history differs from five-way K-shot; five seeds do not establish superiority/equivalence |
| C03 | Historical rare/common axis AUC ~0.995→0.991; legacy common patient overlap | H/B | `results/c0/v1_gate_summary.csv`, current/framing `recomputed_axis_auc` 0.995455/0.991455; `docs/research/CONTEXT.md`, Data/Known pitfalls; protocol §§4,8 | Disease/source confounded; normalization test does not identify what the model used; no patient-independent claim |
| C04 | Descriptive matched-checkpoint study; finalized design, not new algorithm/results | P/B | Protocol §§1–2,9,11 | No first-ever FM/rare-eye claim; gates remain unpassed |
| C05 | Yoo rare OCT augmentation; Quellec rare fundus few-shot precedent | L/B | R01 and R03 primary passages in retinal audit; Yoo dataset/training methods; Quellec abstract/method overview | Different modalities, tasks and information exposure |
| C06 | RETFound, MIRAGE, EyeCLIP, DINOv2 and existing generalist/specialist comparisons motivate roster | L/B | A02/A04/A05/A07 retinal audit; A03 official DINOv2 PDF p.1 (TMLR header/abstract), `parent_refs/DINOv2_primary_title_abstract.txt`; protocol §5 | Frozen-cosine result not inferred from fine-tuning/other cohorts; native transforms/capacity/upstream exposure differ |
| C07 | EyeDiff precedes synthetic+retinal-FM idea; existing project pools excluded from K-shot | L/P/B | A06 primary Introduction/Results/Methods in retinal audit; `stage2_cyclegan.py` QC; CONTEXT Decisions; protocol §3 | Plausibility, diversity and diagnostic utility are distinct; subsampling cannot undo generator/QC exposure |
| C08 | Learned prototypes and simple normalized-feature baselines; cross-domain ranking caution | L/B | A11 official NeurIPS abstract/BibTeX; A12 arXiv abstract; A13 official ECCV PDF pp.1–3 in `parent_refs/CrossDomain_primary_passages.txt` | Proposed normalize-supports-first cosine classifier is not meta-trained ProtoNets/exact SimpleShot; no OCT winner imported |
| C09 | STARTUP extra unlabelled-target access; NLP budget control; CLAP tuning/shift lesson | L/B | F15 local PDF pp.1,3; F21 p.1; F25 pp.1,3, all SHA-matched against source index; `parent_refs/F*_primary_passages.txt` | No target self-training/query adaptation here; fixed-L2 probe is not CLAP; NLP evidence is methodological analogy |
| C10 | CoCoOp image-conditioned text prompts; PathPT combined modules; BenchReAD anomaly precedent | L/B | F16 local PDF pp.1,3 and official CVPR abstract/BibTeX; A15 primary Methods/abstract; A08 official MICCAI abstract and PDF §2 (retinal audit) | No prompt-only effect, OCT transfer guarantee or low-cost claim; anomaly detection differs from five-way/FRR tasks |
| C11 | Canonical counts: 84,630 total; 124 rare development and 22 rare legacy queries | H/P/B | Read-only `data/manifest.csv` (`label,split` aggregation), `verification/manifest_counts.json`; protocol §3 | Image counts; no new patients/groups asserted; canonical split unchanged |
| C12 | 5K supports from 124 inspected labels; 22 historically inspected evaluation labels; excluded resources | P/H/B | Protocol §§3,5–6 | Not 124+5K; labels not multiplied by folds/draws; legacy is not newly blind |
| C13 | Exact/RGB identity plus fixed dHash candidate screen; manual grouping | P/B | Protocol §4 steps 1–3 | BOX/9×8/left>right/64 bit/≤4 exactly retained; not a validated duplicate/patient detector |
| C14 | Exclude rare fitting links to legacy; grouped folds and ≥10 groups/class gate | P/B | Protocol §4 steps 4–5 and fold rules | Verified patients distinct from image groups; common cap not altered; all selector folds/classes required |
| C15 | RETFound/DINOv2/MIRAGE + Keras reference; conditional BioMedCLIP fallback | P | Protocol §§2,5; Table II copies exact public identities | Terms/loading/revisions not claimed passed; availability failure is not poor accuracy; no model search |
| C16 | Native deterministic transforms, crop/FOV logging, fresh FP32 caches and float64 classifiers | P/H/B | Protocol §5; historical `stage3_classifier.py` load/backbone/cache code | Old float16 arrays are preserved and not precision-restored; no causal pretraining inference |
| C17 | Unit support → mean → unit centroid → max cosine; numerical guards/ties | P | Protocol §6, primary classifier; contrast with §5/§10 reuse inventory | Existing raw-feature/mean-first helpers are not interchangeable; no outcome-driven centering |
| C18 | Mean CE+0.005||W||²; unpenalized b; exact L-BFGS-B settings and gradient rule | P | Protocol §6, secondary classifier (0.01||W||²/2 = 0.005||W||²) | No sklearn-C substitution, Adam-head reuse, imbalance weights or query-driven retries |
| C19 | Five folds, nested paired draws; development/reporting K and secondary schedule | P | Protocol §§4,6 | Shared within-group choices/IDs; K16 omitted for all if any class infeasible; no replacement sampling |
| C20 | Pooled out-of-fold class correctness; complete prototype-only FM nomination | P | Protocol §7 formula/eligibility | At least two complete FMs; Inception excluded from candidates; no partial averages |
| C21 | Near-best ≤0.02; measured timing tie-break and prespecified final order | P/B | Protocol §7, 100-image/five-pass timing paragraph | Timing not yet measured; not equivalence; pre-reporting nominee lock; all arms shown |
| C22 | Paired differences, conditional draw mean/SD/percentiles and no confirmatory inference | P/B | Protocol §7, reporting/uncertainty paragraphs | Draws not patients; development-selection optimism disclosed |
| C23 | Nine-way roster: seed42 1,000/common class + all eligible real rare fits | P | Protocol §8 first two paragraphs; `stage3_classifier.major_indices`; canonical common-fitting count | Exact 4,000 common rows preserved; 5,146 is pre-exclusion extraction count, not a new realized inventory |
| C24 | FRR numerator/denominator; by-source reporting with other errors | P/L/B | Protocol §8; A09 OCTDL publisher/release and A10 OCTID primary record (retinal audit) | NORMAL inventories gated; rare-only head cannot provide meaningful FRR; low FRR insufficient |
| C25 | Freeze external eligibility/query IDs; exclude fitting/legacy links; conditional rare mapping | P/L/B | Protocol §§4,8; A09 release labels and publisher; A10 record | Broad OCTDL condition not MH; CSR-to-CSC requires clinical confirmation; no five-class aggregate on partial cohorts; scores ≠ probabilities |
| C26 | Optional deterministic-head intervals: 5,000 cluster draws seed0 only with verified IDs | P/B | Protocol §7; CONTEXT Units & conventions | Otherwise explicit image/group-conditional intervals; no selection uncertainty or leakage repair |
| C27 | Joint recognition/source audit constrains interpretation but does not isolate mechanism | P/B | Protocol §§1,7–8; synthesis of C03/C06/C24 | Descriptive checkpoint-pipeline comparison, no causal morphology/source/pretraining claim |
| C28 | Four/five queries imply 0.25/0.20 recall steps; draw overlap limits precision | H/B | Manifest counts C11; arithmetic 1/4 and 1/5; protocol §7 | No new accuracy estimate, patient CI or clinical transportability assertion |
| C29 | Isolated Colab extractor/CPU heads; future costs and preserved historical artifacts | P | Protocol §§2,5,9–10 | Not implemented or benchmarked; no guaranteed speed/VRAM; no new experiments in this drafting task |
| C30 | Prescore identities/local lock, failure/reduction rules and preserved versioned artifacts | P/B | Protocol §9 pre-score/output rules; §§4,8 external lock | Not OSF registration; NORMAL failure explicitly removes required shift component |
| C31 | Nov11 proposed-methods deliverable; Nov12–27 execution milestones | P | Protocol §9 stepping-stone table and failure policy | Dates Asia/Bangkok, 2026; not completed gates/results; secondary probes dropped first |

## Historical table arithmetic

Source CSVs retain their original bytes. `verify_package.py` recomputes summaries only in memory to validate the manuscript. Across seeds 1–5, means (accuracy %) and sample SD (percentage points, denominator n−1) are:

| Pool | Mean | Sample SD | Manuscript rounding |
|---|---:|---:|---|
| none | 90.688 | 1.857032579 | 90.69 ± 1.86 |
| basic | 92.934 | 1.852411941 | 92.93 ± 1.85 |
| gan | 92.586 | 1.289197425 | 92.59 ± 1.29 |

These describe optimizer-seed variation on the same fixed legacy images, not independent cohorts. No historical confidence interval, RCI re-interpretation or new model output is introduced.

## Citation reading and verification scope

The 12-source retinal batch records per-field metadata, exact primary passage locators and access failures in `verification/retinal_refs/`. The nine-source parent batch is documented in `verification/parent_refs/REFERENCE_AUDIT.md`. Full author lists are retained in BibTeX; IEEE's rendered bibliography may abbreviate long lists. Internal design decisions are attributed to protocol v1.0, not misattributed to external literature. Bibliography identity checks do not constitute a comprehensive retraction/correction audit.

Only short, relevant claims from inspected passages enter the manuscript. Full-paper reading of the entire 49-PDF seed corpus is not asserted. Preprints, conference papers and journal versions are distinguished in the audits. The tooling paper was cited only by the Writing and Reproducibility Note, which the authors removed on 11 November.

## Terminology and notation

| Canonical form | Meaning / decision |
|---|---|
| OCT | Optical coherence tomography; B-scan classification here, not OCTA/volumetric segmentation |
| CSC / MH / MacTel / Stargardt / RP | Fixed rare-class order throughout text, equations and intended artifacts |
| FM / frozen encoder | Pretrained checkpoint with weights unchanged during this proposed study |
| K-shot / support | K real labelled images per rare class; 5K per head |
| Development labels | 124 unique inspected rare labels; supports come from this inventory |
| Legacy queries | Fixed 22 rare test images with historical label/outcome inspection |
| Cosine prototype | Normalize each support, average by class, normalize centroid; no meta-training |
| Macro recall | Equal mean of class recall; development nomination pools correctness within class across folds |
| FRR | True NORMAL assigned any rare class / all true NORMAL, under a separate nine-way head |
| Image component / patient group | Distinct concepts; similarity is not verified patient identity |
| Conditional draw spread | Variability over support choices on these images; not population confidence |
| FP32 / float64 | Fresh inference/raw storage / CPU normalization and classifier arithmetic |

## Remaining author or administrative inputs

Only “Lam” is known; no full name, institution, coauthor, funding or ethics approval is invented. No course-specific template or submission hour was supplied: unmodified IEEEtran conference, US Letter, two columns is the protocol's stated assumption. The author should supply the required author block and confirm the course format/hour before submission. Model access, exact loading revisions/transforms, eligible image groups, external inventories/terms and clinical label mappings are future gates, not missing new results for this proposed-methods draft.
