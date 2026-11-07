# Frozen-encoder FM study: checklist

Step-by-step tracker for the proposal in `main.pdf` ("Frozen Encoders for Few-Shot
Rare Retinal OCT Classification: A Proposed Evaluation Protocol"). Steps follow the
stepping stones S0-S6 of protocol v1.0 §9; `§n` points to that protocol. The
protocol is the authority: if this list and the protocol disagree, the protocol
wins and this list gets fixed. Dates are Asia/Bangkok, 2026.

Tick a box only with evidence (file, hash or command output) written next to it.

**Now:** S1, the 11 Nov submission. The 6 Nov target for checkpoint access has passed with access still unchecked; it must not delay the report.

## Roadmap documents, in time order

How the project got from the Yoo reproduction to this study. Dates are when each
document was written (first commit if older than 4 Nov). Paths are relative to the repo root.

| Date | Document | Role |
|---|---|---|
| 22 Oct | `docs/research/REPLICATION.md`, `docs/research/CONTEXT.md` | Replication tracker and research context (both kept up to date) |
| 23 Oct | `docs/reviews/2026-10-23-mock-review.md` | Mock peer review of the reproduction |
| 26 Oct | `docs/report/2026-10-26_status_report.tex` | Project status snapshot |
| 27 Oct | `docs/research/L7_BASELINE_AUDIT.md`, `L8_RCI_AUDIT.md`, `PAPER_DISCREPANCIES.md`, `lit-scan-gan-vs-basic.md` | Audits of where our numbers differ from the paper |
| 27 Oct | `docs/research/C0_DESIGN.md`, `C0_REPORT.md`, `C1_EXPERIMENT_PLAN.md` | C0 source-confound control and the C1 seed-sweep plan (D15) |
| 27 Oct | `docs/report/Report.tex` | Results, open gaps and research plan (committed 29 Oct) |
| 29 Oct | `docs/research/C2_C3_DESIGN.md`, `T12_FAST_TRAINER.md` | C2/C3 sensitivity design and fast-trainer gate |
| 3-5 Nov | `docs/research/FM_FEWSHOT_PLAN.md`, `FM_3WEEK_SPRINT.md` | Frozen-FM plan with rebuttals; 5 Nov-3 Dec calendar |
| 5 Nov | `docs/research/FM_PROTOCOL.md`, `FM_FINAL_REBUTTAL.md` | **Protocol v1.0** (SHA256 `4e8df8d9…c5df`) and final review |
| 5 Nov | `docs/research/LITERATURE_KB.md`, `LITERATURE_SOURCES.json`, `TASK1_DRAFT_PROMPT.md` | Literature digest, source index, drafting prompt |
| 5 Nov | `docs/report/fm_ieee/` (`main.tex`, `main.pdf`, `CLAIM_SOURCE_LEDGER.md`, `VERIFICATION.md`) | IEEE proposed-methods draft (= `main.pdf`) |
| 5 Nov | `docs/research/C2_C3_RUN_REPORT.md` | Audit of the C2/C3 Colab runs |
| 6 Nov | `docs/research/proposal/` | Proposal figures and architecture page |
| 6 Nov | `docs/research/REPLICATION.md` (closeout section) | C2 100/100 and C3 60/60 runs closed |
| 6 Nov | `docs/research/FM_CHECKLIST.md` | This checklist |

## Standing rules (every step)

- No outcome-driven changes: any recipe change gets a dated amendment and keeps previous outputs (§9).
- Identity or numerical failure stops that arm; a required fold/K failure stops the selector (§9).
- Unavailable arms are reported as unavailable, never scored as poor models (§5).
- If late, drop secondary logistic probes first and disclose the reduced scope (§2, §9).
- Canonical data, `data/manifest.csv`, historical heads, caches and results are never overwritten.

## S0: Historical baseline (available)

- [x] Yoo pipeline Stages 0-4 and seed-42 results (`results/stage4/metrics.csv`)
- [x] Seed check 1-5 (`results/seed_check.csv`); audits L7, L8, L9; C0 gate; C5 v2 pool (`docs/research/CONTEXT.md`)
- Done when: these are cited only as motivation, never as FM evidence (§10).

## S1: Proposed-methods IEEE draft (5-11 Nov)

- [x] Protocol v1.0 finalized after final-round review (5 Nov; hash above)
- [x] Five-page IEEEtran draft with 21 references; `verify_package.py` PASS on 5 Nov (`VERIFICATION.md`)
- [x] Preserve the protocol and draft sources in git: committed 7 Nov on `t3/proposal-project-checklist` (protocol blob hash checked against `4e8df8d9…c5df`); reaches `main` only after Lam's merge
- [ ] Bring the C2/C3 aggregate CSVs from Drive into `results/` (the closeout cites Drive files owned by a second account)
- [ ] Checkpoint and fallback access attempted by 6 Nov: RETFound terms, DINOv2, MIRAGE, BiomedCLIP (§9). Must not delay the report.
- [ ] Replace "Lam" with the full author/affiliation block
- [ ] Confirm the course template and submission hour
- [ ] Full draft 8 Nov · optional-results cutoff 9 Nov 18:00 · PDF freeze 10 Nov 18:00
- [ ] Lam submits 11 Nov
- Done when: the submitted PDF reports proposed methods only, with no new FM outcomes.

## S2: Implementation and pre-score input lock (12-14 Nov)

Nothing in S2 may compute a labelled score.

Loader and features (§5)
- [ ] Isolated Colab PyTorch loader with its own requirements; stage scripts stay untouched
- [ ] Per arm, pin: terms, weight revision/hash, channel handling, native transforms, normalization, pooling, dtype, inference mode
- [ ] Fallback decision: BiomedCLIP replaces RETFound only for an access, loading or terms failure
- [ ] Keras Inception reference via `stage3_classifier.load_image` and its backbone; FP32; new cache directory with sidecar identity (bypass the float16 writer)
- [ ] Cache identity: ordered image IDs/content hashes, weights, transforms/pooling, environment, dim/dtype, code revision; crop geometry and field-of-view fraction logged

Dependence screen and grouping (§4)
- [ ] 146 rare images: file hashes + RGB pixel fingerprints; dHash 9x8 BOX, Hamming ≤4; Pillow version pinned
- [ ] Raw-image review of flagged pairs → connected components; label conflicts block the component
- [ ] Exclude rare fitting components linked to legacy queries (test rows never move)
- [ ] Group IDs and five folds: PCG64 `SeedSequence([42,class_index,5101])`, round-robin
- [ ] Feasibility: ≥10 eligible groups/class and 8 fitting groups/class/fold, else the selector stops
- [ ] K=16 reporting feasible only if every class has ≥16 fitting groups; otherwise omit for all encoders
- [ ] Publish per-class/fold counts and the full exclusion ledger

Episodes and classifiers (§6)
- [ ] Episode builder: development namespaces 6101/6102, reporting 7101/7102; realized support/query IDs saved
- [ ] Cosine prototype: normalize supports, then average, then normalize the centroid; fixed rare-order ties; zero/nonfinite guards
- [ ] Logistic probe: mean CE + 0.005‖W‖², L-BFGS-B with the exact §6 settings; convergence = finite and max |grad| ≤1e-6
- [ ] Focused tests for the prototype rule, episode determinism and the probe objective

Nine-way roster and external data (§8)
- [ ] `stage3_classifier.major_indices(common_train_labels, 1000, 42)` on the 73,958-row common frame, mapped back to paths in order, plus eligible rare fitting rows (5,146 images before exclusions)
- [ ] OCTID/OCTDL: release/hash, terms, label inventory and eligibility frozen
- [ ] Rare mappings: CSR→CSC needs clinical confirmation; OCTDL MH uses the subcategory, not the condition label alone

Lock (§7, §9)
- [ ] 100-image warm timing pilot (batch 1, FP32, one warm pass + five timed)
- [ ] Input lock archived: protocol hash, roster, grouping/exclusions, all episodes, nine-way roster, model/preprocessing hashes, code/env, endpoints
- [ ] `.agents/check` passes
- Done when: the lock archive exists and its timestamp precedes every labelled score.

## S3: Paired development and nomination (15-19 Nov)

- [ ] Features extracted for every available arm, identity-checked
- [ ] Prototype development: 5 folds × K={1,2,4,8} × 100 draws, every arm
- [ ] Secondary probe: K=4, first 25 draws/fold (first to drop if late)
- [ ] Completeness per FM; at least two complete FMs, else a feasibility report with no winner
- [ ] S_m per FM (eq. 4); near-best set within 0.02 of S_max; tie broken by median warm s/image
- [ ] Nominee locked 19 Nov, before any legacy or external outcome
- Done when: the nominee and every arm's K-specific scores and denominators are archived.

## S4: Locked query reporting (20-23 Nov)

- [ ] Reporting episodes on the 22 legacy queries, K={1,2,4,8,(16)}
- [ ] External queries screened against fitting rows, legacy queries and each other; final IDs frozen before inference
- [ ] One deterministic nine-way prototype head per encoder
- [ ] FRR on NORMAL for Kermany, OCTID and OCTDL separately, with NORMAL recall, common confusions, rare recall and denominators
- [ ] Partial-class external cohorts: present-class recalls and all output counts, no five-class macro
- Done when: query-level predictions and scores are saved for every arm.

## S5: Summaries and all-arm freeze (24-27 Nov)

- [ ] Paired nominee-minus-Inception macro recall by K, plus all secondary contrasts
- [ ] Support-draw mean, SD, 5th/95th percentiles (labelled as draw variability, not patient CIs)
- [ ] K-shot and source plots
- [ ] Compute ledger: download, staging, extraction, fitting, serialization, peak RAM/VRAM
- [ ] Hashed archive copied to a durable location and bytes verified
- [ ] All-arm freeze 27 Nov
- Done when: the archive is frozen, failures included.

## S6: Final write-up (28 Nov-3 Dec)

- [ ] Actual Methods / Results / Discussion
- [ ] Evidence audit 1 Dec
- [ ] PDF and package freeze 2 Dec
- [ ] `README.md`, `COLAB.md`, `docs/research/CONTEXT.md` updated
- [ ] Delivery 3 Dec
