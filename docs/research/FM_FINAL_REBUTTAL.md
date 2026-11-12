# Final FM protocol rebuttal and resolution record

> **Closed record (13 November 2026).** Kept as evidence of how decisions were reached; no longer updated. The study as it stands is in [`protocols.md`](../../protocols.md); the binding texts are `FM_PROTOCOL.md` and `FM_AMENDMENTS.md`.

Date: 5 November 2026. Owner: Lam. Parent/consolidator: GPT-6.1-Sol / Codex.
Status: design finalized; all required R3 design findings resolved in protocol v1.0.
Execution remains gated; no new experimental or clinical validation is claimed.
No study data, splits, model outputs or executable pipeline are changed here.

## 1. Scope and versions

Lam requested one further rebuttal round by the parent, Claude Opus 5.5 and
Gemini 4 Argon High, followed by a finalized protocol, project stepping stones and
a reuse inventory for the Yoo 2021 reproduction.

Reviewed candidate: docs/research/FM_PROTOCOL.md, SHA256
A8049A8153159BADEEDB2B7888512754DA144949D9E0B3045D876FB182808996.
The candidate was held unchanged while the two children reviewed it.
All final changes are parent consolidation after that review, not a claim that
the reviewers independently reread the final hash.

Final protocol SHA256: 4e8df8d91c0021187c05b103a9b0a0e0319a055fd91108da1f4cf3045d00c5df.

Input planning snapshots and their SHA256 records are preserved under
.agents/reports/fm-final/prior/ and planning_input_hashes.json. These include the
3 November two-round proposal/reviews and the 5 November calendar from the
t3-8f9fcccf worktree. The original uncommitted source files are not edited.
FM_PROTOCOL.md is the self-contained source of truth for this sprint.

## 2. Participants and ownership

| Reviewer | Actual selected provider/model | Scope and original verdict |
|---|---|---|
| Parent | Codex / gpt-6.1-sol, max reasoning | Implementation/reuse, inference scope, overlap, persistence; ready with enumerated changes |
| Claude | claudeAgent / claude-opus-5-5, effort=max | Scientific/statistical design and feasibility; Ready with enumerated changes |
| Gemini | antigravity_antigravity2 / gemini-4-argon-high | Independent coherence, exposure and reuse; ready with enumerated changes |

Children shared the checkout and owned only claude-r3.md and gemini-r3.md in
.agents/reports/fm-final/. The parent owned the protocol, this record and the
presentation. No uncommitted work was copied into a new worktree; no other models
were substituted or additional reviewers spawned.

The raw reports retain their original wording and candidate-specific references.
The following dispositions use code/CSV evidence independently rechecked here;
some reviewer source line numbers refer to other offsets and are not canonical.

## 3. Gemini findings and disposition

| ID | Disposition | Concrete correction |
|---|---|---|
| G3-01: overlap scope/order | Accepted | Keep the canonical common cap roster; rare image/patient grouping affects rare fitting only; later external overlap excludes external queries and cannot mutate fitting rosters |
| G3-02: mean-loss L2 scaling | Accepted | Specify the exact mean cross-entropy +0.005 Frobenius-squared objective, optimized directly with SciPy L-BFGS-B; no fixed sklearn C surrogate |
| G3-03: group identity/order | Accepted | Minimum POSIX member path as group ID; sorted members and groups; group_index is sorted inventory rank |
| G3-04: deterministic nine-way head/outputs | Accepted | One full-roster prototype head per encoder; five- versus nine-column prediction allocation; cosine scores are not probabilities |
| G3-05: cache/centroid reuse contract | Accepted | Fresh matched FP32 embeddings/reference; legacy float16 caches are historical; do not reuse raw-mean centroid_correct as the unit-normalize-first prototype |

Reviewer assertions not adopted:
- The calendar is not certified to fit Colab Pro. Throughput, memory, I/O and
  access are unmeasured gates; no GPU-hour guarantee follows from small row counts.
- 506 patients /804 images concern overlap with our fitting split. Official
  train-plus-validation overlap is566 /914, from cluster_summary.csv.
  These test-patient counts do not determine the fraction of all fitting rows
  removed by exclusions; an unmeasured majority-removal inference is rejected.
- fast_head.py is excluded by this fixed prototype/L2-LBFGS recipe. Adam heads
  are not intrinsically invalid for few-shot learning; fitted all-label weights
  cannot be reused as fresh support-only classifiers.

## 4. Parent findings and disposition

| ID | Disposition | Correction / retained boundary |
|---|---|---|
| P1: external-query overlap | Accepted | Explicit rare-only fitting screen, external-query-only exclusion direction and lock timing, actual screened inventory and unknown remaining overlap |
| P2: stable identities | Accepted | Dataset-namespaced patients; canonical group/member ordering and saved actual rosters |
| P3: nomination completeness | Accepted | No score for a partial/failed prototype arm; no winner from fewer than two complete FMs |
| P4: precision/cost fairness | Accepted | Fresh FP32 extraction/cache for all matched encoders; float64 small-classifier arithmetic; fixed batch1 FP32 timing |
| P5: durable outputs | Accepted | New versioned local root, completion hashes, identity-checked resume and verified user-controlled archive; no implicit Drive migration |
| P6: final design versus execution | Retained limitation | Final protocol fixes rules; access, grouping and clinical mappings remain empirical gates, not alleged completed validation |

## 5. Claude findings and disposition

| ID | Disposition | Concrete correction |
|---|---|---|
| R3-1: exclusion/roster contradiction | Accepted, option A | Rare grouping/exclusions do not purge common Kermany rows; keep the exact canonical 4,000-row cap roster and disclose patient overlap. A new patient-disjoint comparator is outside this version |
| R3-2: late external links | Accepted | Remove linked external queries, not locked fitting rows; include the exact nine-way roster in input lock |
| R3-3: nomination role | Accepted | Nominee is the pre-designated primary FM for legacy/external headline summaries and nominee-minus-Inception contrasts; no re-designation after those results |
| R3-4: rule precision | Accepted | S_max over complete FMs only; minimum median of five per-pass means; one-session fixed-order FP32 timing and retime after runtime change; fixed seed indices; no late arm entry; undefined optional ceilings removed |
| R3-5: reference/reuse precision | Accepted with fresh-reference resolution | Historical Keras loader/backbone, fresh FP32 caches, reuse classes and both raw-mean hazards explicit; no historical-head external arm; cluster draws aligned to5,000/seed0. No legacy cache is admitted to the matched comparison, so a tolerance-based acceptance test is unnecessary |
| R3-6: partial-class metrics/ledger | Accepted disclosure | No five-class aggregate on partial external cohorts; per-present-class recall and complete allocation rows; per-stage timing/memory/image counts |
| R3-7: prior inspection/claim bounds | Accepted disclosure | Pre-score means new-protocol scores, not never-inspected images; prior Inception analyses of all146 and checkpoint/source confounding are disclosed |

Additional gate clarifications from Claude: fixed BOX downsampling and image
fingerprint details, flagged-pair/component counts before adjudication, minimum
10 groups/class for the grouped development design, recorded native geometry,
and a fixed27 Nov freeze if the S2 lock slips. These create no new model arms.

The raw review's suggestion that common-overlap effects might be small is not
an empirical finding; the effect remains unmeasured. Its runtime assurance is
also not adopted. A bounded implementation is planned, not guaranteed by review.


## 6. Evidence behind reuse decisions

| Checked source | Verified contract |
|---|---|
| stage3_classifier.py:59,112,132 | major_indices is a pure helper; features validates ordered paths and stores float16 |
| Saved Stage3 heads | All three W kernels are(2048,9), biases(9,), with the recorded nine-class order |
| Main-checkout caches | test1022x2048; rare real caches27/26/25/19/27x2048; this is availability/alignment evidence, not historical input provenance |
| c23_experiment.py:430 | Sweep validation enforces2048x9, fixed250 histories and old augmented-pool contracts |
| stage4_evaluate.py:57,83,118 | Evaluation assumes9 classes; historical prototypes use Euclidean raw means; CAM is Inception-specific |
| analysis.py:68,79,173,195 | Unknown rare patient IDs become singleton paths; bootstrap targets accuracy/MCC; raw-mean centroids differ from the final recipe |
| figures.py:41 | ROC validation requires the27 historical variant/class curves |
| results/analysis/cluster_summary.csv |633 common test patients;506 fitting overlap,566 train-plus-validation overlap |
| results/analysis/paired_seeds.csv | Five-seed GAN-basic mean−0.348 pp, SD2.7697508913; no established GAN superiority |
| results/c0/v1_gate_summary.csv | Current/framing axisAUC0.995455/0.991455; ready_for_heads=False |
| results/stage2/c5_v2_counts.csv | Separate v2 accepted pools total3000 per disease; no proof of downstream model validity |

Direct reuse: immutable IDs/labels/raw-input references, historical results/audits,
small matching helpers, and protection/plotting patterns. New implementation:
FM adapters, grouped episodes, support-only classifiers, dimension-independent
metrics/prediction alignment, and external-source evaluation. Existing GAN pools,
nine-way weights and Inception CAMs do not become new K-shot/FM evidence.

## 7. Operational gates retained after design freeze

- Access/terms and exact checkpoint revisions/preprocessing/pooling.
- Pre-score group/exclusion/episode feasibility; stop the whole selector if the
  required five-fold K<=8 design is impossible.
- External source/release eligibility, duplicate screening and clinical mappings.
  Missing mapped rare data narrows disease-transfer claims; missing NORMAL
  marks the required source component incomplete.
- Measured runtime/memory, versioned outputs and successful implementation checks.
- Administrative IEEE template/submission hour; report remains viable without
  new results. No public preregistration or external submission is claimed.

These gates have predetermined outcomes/stop rules. They are not permission to
change model recipes or test cohorts after seeing results.

## 8. Verification and completion surfaces

Repository harness completed successfully (parent run; the reviewers did not run
research fitting). Its TensorFlow fits are small temporary unit-test fixtures,
not new research experiments. Exact terminal lines:

    Identity check: PASS (0 outgoing commits; author and committer match local noreply)
    Agent checks: PASS

Six main-checkout real/test feature groups match canonical manifest paths/order;
all are float16 and2048-dimensional. The16 protected manifest/result hashes match
the pre-review record. Pillow BOX enum and SciPy minimize availability were checked
without fitting. Final document/presentation checks are captured in the local
verification record.

Pre-completion surfaces:
- Entry points: no scripts, flags or notebook cells changed.
- Reverse states: no runtime action added; not applicable.
- Data/schemas: no input lists/caches/results changed or new episode data generated.
- Docs: protocol/rebuttal added; CONTEXT, README and COLAB link the planned study
  and distinguish it from executable reproduction instructions.
- Checks: existing .agents/check passed; scoped source/hash/link/table and visual
  checks verify the documentation/presentation. No study run or public registration.

Only these docs, the ignored presentation/review artifacts and the existing
mistake log changed. Tooling parsing/anchor errors were caught before mutations
and logged; no permanent guard/workflow fix was installed.

Technical objective reference opened:
[SciPy L-BFGS-B documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
This supports solver-option meanings, not empirical model performance.

Research-workflow attribution is recorded in FM_PROTOCOL.md, citing the current
[Scientific Agent Skills record](https://arxiv.org/abs/2609.00065) checked 5 November 2026.
Review agreement is advisory and is not evidence of clinical accuracy or novelty.

Pillow filter semantics opened during resolution:
https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
The code API was also checked locally; this does not validate dHash on OCT.

