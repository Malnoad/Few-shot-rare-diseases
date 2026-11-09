# FM pre-run risks and acknowledgement

9 November 2026. Scope: the transition from code preparation to real checkpoint
staging/loading, score-free Colab pilots, the S2 input lock and external queries.
Lam remains the research author and decision maker.

**Status: agent assessment recorded; Lam's acknowledgement and real-run approvals
are pending. Claude R12 returned READY WITH CHANGES for this record as a pre-run
aid; its findings and the parent's documentation corrections are recorded below.**

This is a review record for existing code gates and manual obligations. It adds no
runtime approval flag or scientific recipe. Acknowledging a risk does not make an
unverified result verified or permit a failed gate to be bypassed. The frozen
[protocol](FM_PROTOCOL.md), especially sections 4, 5, 7–9, remains authoritative.

## Snapshot and evidence limits

Branch `t3/fm-code-prep`. This review ran against HEAD `80aa652` with the
implementation uncommitted; Lam then asked on 9 November to finalize the code
state, and it was committed on this branch with these same bytes. That request
is the commit authorization only. Normalized source SHA256 prefixes (CRLF folded
to LF), rechecked immediately before that commit:

- `fm_core.py`: `fea7a5eb0c5a8a93`
- `fm_prepare.py`: `d8c83038ccbdc653`
- `fm_extract.py`: `8db7e10596a105cf`
- `fm_evaluate.py`: `b32477d2f573a801`
- Protocol SHA256: `4e8df8d91c0021187c05b103a9b0a0e0319a055fd91108da1f4cf3045d00c5df`

The [review record](FM_CODE_REVIEW.md) documents 149 passing FM tests, the full
project check and synthetic CLI/recovery counterexamples. Windows CPU package,
preprocessing and synthetic checkpoint-metadata checks are bounded evidence.
They do not verify published checkpoint contents, successful encoder loading,
Colab/GPU behavior, real-feature probe convergence or external cohorts.

Two read-only Codex reviews of this snapshot found no new confirmed code defect.
The risks below include integration blockers, known limitations and manual
obligations; they are not all bugs and are not assigned numerical probabilities.

## Risks and the protection actually present

### R1. Actual assets and executable source — loading gate

Pinned revisions/digests and [asset verification](../../fm_extract.py#L144)
protect identity. Restricted `.pth` loading and the scoped Namespace allowance
protect serialization; they do not sandbox the pinned Python source executed by
[import_source](../../fm_extract.py#L249). Actual checkpoint layouts and allowed
default norm parameters remain unverified.

Before acquisition/loading, record per-arm terms evidence (source, date and Lam's
decision) and review the pinned source/imports. `--terms-evidence` accepts nonempty
free text; code cannot verify that terms were reviewed or accepted. Match acquired
bytes to the pinned digests. Preserve actual missing/unexpected keys and layout,
and the loader's declared default-norm policy; its norm-description string is not
an independent measurement of parameter values. **Code stop:** digest mismatch,
unsupported metadata, unexpected encoder keys, incompatible layout or wrong
shape. Do not switch to
unrestricted pickle loading, widen allowlists blindly or silently accept random
encoder parameters. RETFound fallback remains conditional on a recorded
access/loading/terms failure before labelled scores.

**Manual stop:** missing usable terms or unreviewed executable source. Separate
acquisition from execution. Before running upstream source, use a worker without
credential environment variables or access to the Hugging Face login cache, and
leave Drive unmounted unless needed. The offline flags do not clear credentials;
these precautions reduce exposure but do not sandbox the source. No credential
cleanup or terms acceptance has been performed by this review.

### R2. Colab dependencies and evaluation environment — integration gate

The direct [package pins](../../requirements-fm.txt) are not a complete Colab or
TensorFlow environment lock. [Extraction identity](../../fm_extract.py#L104)
records actual package versions, and the gate compares FM environments. These
checks do not themselves establish that the intended pins were installed.
[CPU arithmetic identity](../../fm_prepare.py#L236) checks Python/NumPy/SciPy
versions; it does not capture every BLAS build, OS or CPU detail.

At full extraction, the entire recorded environment must equal the pilot identity
in the lock: Python, `platform.platform()` and every installed distribution.
Even a transitive package or kernel-string change can stop a fresh extraction.
This also applies to the separate TensorFlow reference environment. **Code stop:**
any such mismatch; no supported command relaxes this equality.

Finish setup before final pilots. Compare direct versions with the pin file,
check dependency consistency and ordinary imports, and retain package/OS/CUDA/
driver evidence. Record how the exact environment will survive or be rebuilt;
a full package freeze alone cannot reproduce a base image or kernel. Keeping the
runtime alive through pilot, author-approved lock transfer and extraction is an
operational option, not a persistence guarantee. Use the intended CPU evaluation
machine. **Manual stop:** resolver/import failures or an unplanned environment
change. Do not migrate evaluation solely because three version strings match.
Normal OpenCLIP package initialization still needs a real environment check;
prior numeric transform tests isolated its module from tokenizer initialization.

As an operational safeguard, extract and durably export every locked available
arm's current inventory before the first `develop`. Before labelled scores,
environment loss can be handled with fresh pilots and a new author-approved lock,
preserving the earlier attempt. After scores, stop and refer a changed identity
to Lam for a dated amendment/new version; never rewrite the old lock. Later fresh
external features still need the environment/versioning plan in R5 and R9.

### R3. FP32, GPU numerics and capacity — pilot gate

The [loader](../../fm_extract.py#L273) selects FP32, disables TF32/autocast and
checks output dtype; [feature validation](../../fm_extract.py#L171) rejects
nonfinite/zero vectors. These controls do not guarantee bitwise GPU repeatability
or sufficient RAM/VRAM. Actual published models have not been executed here.

**Code stop:** unintended lower precision, invalid shape, nonfinite or zero
embeddings. A timing pilot validates these outputs but discards them; it does not
perform or save a repeat-output comparison. A pilot pass establishes loadability
and valid output, not equivalence to the intended upstream embedding.

The proposed repeat-numerics check is manual: write its acceptance rule before
execution, use synthetic inputs with each actual encoder in a separate process,
and wait for that process to exit before the fixed timing sequence. Preserve
numerical differences and backend settings separately, alongside pilot geometry
and memory evidence. Lam's decision on an additional upstream-output comparison
remains pending. **Manual stop:** a failed predeclared check or unexplained
instability; no numerical repeatability threshold has been adopted here.
Retrying under unchanged identity
is different from changing the scientific recipe. Cross-device bitwise equality
and throughput remain unclaimed until measured.

### R4. Comparable timing needs process supervision — manual operation gate

[Pilot verification](../../fm_prepare.py#L675) checks one session/hardware,
common FM environments, distinct PIDs and ordered nonoverlapping recorded windows.
These are code gates.
A recorded operation window ends before final writes/process exit; it cannot
prove the old process exited or that another workload was absent.

Run synchronous commands in RETFound/fallback → DINOv2 → MIRAGE order, then the
separate TensorFlow reference process, waiting for each exit. Retain launcher/
worker PID and exit receipts. **Manual stop and retime the comparable sequence**
after competing GPU work or relevant runtime/hardware/environment changes. Use
fresh `--out` folders and a new shared `--timing-session`; rerunning a completed
pilot verifies and returns its old receipt instead of retiming. Keep every
attempt and record why the final comparable sequence was used. Warm timing
and separately measured download/staging/fitting/serialization costs must not be
conflated. Download time is not measured by the extractor: the acquisition
operator must record each asset's source, byte size, elapsed seconds and date
separately under protocol section 9.

### R5. External implementation must precede the final code freeze — blocker

Current external source fields are planning declarations, not an acquisition,
screening, query-lock or inference implementation. [CODE_FILES](../../fm_prepare.py#L55)
includes all four FM scripts, `stage3_classifier.py` and both requirements files;
[load_lock](../../fm_prepare.py#L717) rejects changes to them. Edits to core,
preparation or stage 3 also stale the preparation producer identity, requiring
fresh `screen`/`freeze` outputs as well as pilots. Adding external support after
the real lock would invalidate the locked execution snapshot.

Before final pilots/lock, settle the implementation/versioning plan. Prefer
implementing and synthetically verifying the required external interfaces first.
An independent external implementation would instead need its own reviewed code,
environment and query identities explicitly bound to S2 and the original S3
nominee; an unhashed extra script is insufficient. No such alternative has been
implemented or approved here. **Manual stop:** do not create the final S2 lock
while the intended external scope and its implementation/versioning plan remain
unresolved. `unavailable` must record an actual access or terms failure and the
protocol section 8 reduced scope; unfinished implementation is not such evidence.
A genuine access failure may lead to a disclosed rare-only pilot, with the
required shift component marked not run. A `planned` source needs its actual
release digest and terms/source/eligibility evidence before lock; nonempty
planning strings cannot certify these facts. Do not edit lock bytes or weaken
hash checks later.
Protocol section 9 already permits final external query IDs to be locked later,
before inference, under frozen source/release/eligibility rules. This is distinct
from changing the locked code or scientific recipe.

Once Lam has approved the relevant source/access route, asset staging and initial
score-free loading checks can precede this decision. They do not become final
pilot/lock evidence if later changes alter their recorded code identity;
regenerate affected preparation and pilot artifacts. If S2 slips, S3 starts only
after a successful lock. These gates must not delay the 11 November report.

### R6. Screening is weaker than patient identification — known limitation

[Grouping verification](../../fm_prepare.py#L462) rebuilds duplicate/patient
evidence, exclusions, folds and supports. It enforces complete review decisions,
cross-label stops and group feasibility. The prescribed dHash screen is not a
validated duplicate/patient detector; zero flagged pairs do not prove independence.
Rare manifest `pid` values are not verified patient links. An evidence string
cannot certify that its underlying patient linkage is true.

Review actual image decisions, patient-source evidence and saved per-fold counts
before pilots and lock; the counts are produced by `freeze`. **Code stop:**
missing required review decisions, conflicting image labels, cross-disease
verified patients or infeasibility under the protocol. A recorded `unresolved`
decision is a valid conservative link; it does not itself stop the selector
unless the resulting component conflicts or becomes infeasible. **Manual stop:**
unsupported evidence presented as verified. Unknown
patients remain a disclosed limitation; do not infer them from filenames or
change the fixed Hamming threshold to obtain a preferred grouping.

`verified_patients.csv` is a hashed lock member. If genuine patient links become
available, decide how that lock will travel before any public push; a valid hash
does not make patient evidence suitable for a public repository.

### R7. First nomination is sealed within one lock folder — operational precaution

[nominate](../../fm_evaluate.py#L360) seals the first decision beside the lock,
including a no-winner decision. Entirely unstarted runs are rejected, but a first
call can still bind an unintended incomplete subset once development has begun.
Later completion or a different output directory cannot replace that decision
when using the same lock folder. This code guarantee is local to that folder:
a restored/copied S2 lock without the original S3 binding can compute another
decision. It is not a global registry of every copy of the study.

Before the first call, inspect every intended arm's available/planned prototype
cells and terminal failure/availability status. Continue recoverable development
until the intended stopping point; explicitly distinguish deliberate feasibility
closure from an accidental early call. Use one designated study lock folder.
**Manual stop:** deleting, losing or bypassing the binding through a different
copy to recompute selection after outcomes. After first nomination and before
the first legacy `report`, copy the nomination export to durable storage, verify
its bytes and record the binding hash outside the runtime. Recovery requires the
original binding. **Code stop:** reporting with a missing or mismatched binding
in the chosen lock folder.

### R8. Colab storage and strict resume — operational precaution

Completion receipts, child coverage, immutable writes and
[restore](../../fm_prepare.py#L730) protect completed artifacts. They do not
create a durable off-runtime copy. Extraction saves the feature array after its
full loop, so an early interruption may require redoing inference. Hardware or
environment changes can prevent reuse of an incomplete operation or fresh
extraction under the lock, even if earlier operations completed successfully.

Verify copied bytes at the durable destination for the S2 ZIP/receipt, later S3
decision/binding, completed caches and run exports. The S2 ZIP predates S3 and
cannot recover the later nomination by itself. **Manual stop:** reporting if the
original S3 state is missing; never regenerate selection as a backup substitute. Inspect
the real owner before token-unlocking a stale writer. Preserve previous versions.

Lock artifacts use raw-byte SHA256. This checkout has `core.autocrlf=true`, and
`results/fm_v1/**` currently has no no-conversion attribute. Git may therefore
change CSV/JSON bytes on checkout. **Code stop:** the resulting hash mismatch.
Before any public lock commit, add a reviewed `results/fm_v1/** -text` attribute
and verify stored/fresh-checkout bytes, or transport the private lock through its
verified ZIP/restore path. This configuration change has not been applied here.
Neither path substitutes for preserving the later S3 binding. Review patient
evidence and terms before deciding whether any lock member can be published.

### R9. External labels, bytes and preprocessing — external-inference gate

[External declarations](../../fm_prepare.py#L623) require statuses and nonempty
fields; they do not verify acquired cohort bytes, clinical equivalence or final
screened query IDs. The protocol requires external queries to be screened against
locked fitting rows, legacy queries and each other. Exclusions apply to external
queries; the locked fitting roster must not be retrospectively changed.

For a planned source, actual release digests, terms and source/eligibility
definitions precede S2; final screened query IDs may follow under those frozen
rules. Before inference, reconcile those definitions, verified patient evidence
and the final exclusion/query ledger with the acquired artifacts. Check
new dimensions/modes against native processors without outcome-driven tuning:
DINO's one-pixel-height rejection was outside the current selected image domain,
but that says nothing about future external images. **Manual stop:** absent
implemented query-lock/inference support, unsupported input
handling or unresolved mappings. CSR-to-CSC and OCTDL MH subcategories require
clinical evidence; omit unsupported rare-transfer claims. If NORMAL access fails,
report the required source component as not run, not a completed shift study.

### R10. Successful software does not establish clinical validity — claim limit

Fresh FP32 caches, paired episodes and [full reporting counts](../../fm_evaluate.py#L451)
do not remove historical label inspection, unknown pretraining exposure,
common-patient overlap or disease/source confounding. Synthetic tests do not
predict real-feature logistic convergence. Failed probes retain diagnostics and
missingness; they must not be retuned or replaced using outcomes.

Retain source-specific NORMAL FRR from the nine-way head, all class outputs and
actual denominators. Partial-class external cohorts do not receive a five-class
macro recall. [Draw summaries](../../fm_evaluate.py#L543) are conditional support-
draw variability, not new-patient confidence intervals. Continue the descriptive
checkpoint-pipeline comparison within these limits; do not convert it into a
causal pretraining-effect or clinical-generalization claim. These interpretation
limits are manual reporting obligations, not an automated claim validator.

## Transition checks and acknowledgement record

These are evidence slots, not assertions that the next stages have happened.

- [x] Agent risk assessment recorded against the source snapshot above.
- [x] Claude R12 rebuttal completed; original verdict and parent dispositions below.
- [ ] Lam acknowledges the residual risks/claim limits; date/message reference recorded.
- [ ] Lam's requested paced protocol sections 4/6 walkthrough disposition recorded.
  Lam's decision on 9 November: do this and the risk acknowledgement after the first
  score-free Colab checks and before final pilots/S2 lock. Both remain open.
- [ ] Before real setup: Lam's code/source/access review, per-arm usable terms
  (source/date/decision) and any commit/push authorization recorded separately.
  Tokens and weights remain outside the public repo; source workers' exposure reviewed.
- [ ] Before final pilots/S2 lock: R5 external implementation/versioning plan
  resolved and required interfaces tested, or actual access failure/reduced scope
  documented; real release digests and source/terms/eligibility decisions recorded.
- [ ] Exact-environment preservation/rebuild plan recorded for both frameworks;
  manual numerical acceptance rule and Lam's upstream-comparison decision recorded.
- [ ] Actual pinned assets, isolated environments, native load reports and fixed
  sequential GPU pilots pass, with process receipts and all attempts preserved.
- [ ] Lock transport/publication and byte-preservation plan resolved, including
  patient-evidence sensitivity and the conditional Git-attribute change in R8.
- [ ] Lam approves the exact real S2 inventory/roster, code/environment, available
  arms/endpoints and archive SHA256; ZIP/receipt copied, bytes verified and an
  timestamp/receipt outside the runtime recorded before labelled comparisons
  (for example, Lam's archive-approval message). `created_utc` alone is only a
  local clock record, not independent timestamp evidence.
- [ ] Every locked available arm's current inventory extracted/exported and copied
  bytes verified before first `develop` (operational safeguard in R2).
- [ ] Before first S3 nomination: completion/failure inventory deliberately reviewed.
- [ ] After first nomination, before first legacy `report`: nomination export copied
  and bytes verified; original binding hash recorded outside the runtime.
- [ ] Before external inference: separate external query lock, clinical mapping
  dispositions and original S3 binding verified; required source outputs implemented.

Lam acknowledgement: **pending**. Code push authorization: **given 9 November for
the branch and its PR only; merging to `main` waits for Lam's sections 4/6 read**.
Real S2 lock approval/hash: **pending; no real lock exists**.
These fields may be completed only from Lam's actual decisions and recorded evidence.

## Independent rebuttal and disposition

Two bounded Codex source reviews found no new confirmed defect and supplied the
runtime/integrity risks above. Neither ran models or tests. Earlier passing test
evidence is in [FM_CODE_REVIEW.md](FM_CODE_REVIEW.md), not a fresh empirical result.

Claude Opus 5.5, attempt R11: **failed; no verdict or objections delivered**.
The orchestration result was: "Claude gave up after repeated API errors."
No conclusion is inferred from this unavailable review.

Claude Opus 5.5, R12 (standard mode, max effort): **READY WITH CHANGES for this
record as a pre-run aid only**. The reviewer found no confirmed code defect in
bounded static inspection and ran no tests, models or downloads. Their original
verdict is retained; Claude has not re-reviewed the corrections below.

Parent dispositions after checking the cited source:

1. **Post-lock environment equality — accepted.** R2/R8 now describe full
   environment matching, exact-runtime recovery limits, pre-score re-lock versus
   post-score amendment, and extracting/exporting available arms before development.
   A package freeze is evidence, not a guarantee of platform reconstruction.
2. **External unavailable bypass — accepted with the protocol qualification.**
   R5 requires actual access/terms evidence; unfinished code cannot be disguised
   as unavailability. Genuine access failure still permits section 8's reduced
   scope. Planned source digests precede S2. The implementation decision is open.
3. **Manual checks overstated — accepted.** Code/manual stops are identified.
   R1 distinguishes declared norm policy from measured values; R3 states that
   repeat/upstream comparisons are not performed by the pilot. Acceptance rules
   and Lam's comparison decision remain pending.
4. **Consent and terms — accepted.** Digests are called pinned, terms text is
   explicitly unverified, and per-arm decisions plus the outstanding walkthrough
   have evidence slots. The runbook example no longer asserts Lam's review.
5. **Nomination scope — accepted.** R7 and the runbook limit sealing to one lock
   folder and require the original binding backup/hash before legacy reporting.
   A restored S2 copy without that binding cannot recover the S3 decision.
6. **Retiming versus resume — accepted.** R4/runbook require fresh output folders
   and a new timing session for retiming; every attempt/reason is retained.
7. **Download costs — accepted.** The acquisition operator owns the separate
   source/size/seconds/date ledger; extractor download cost remains unmeasured.
8. **Credential reachability — accepted as a manual precaution.** R1/runbook
   separate acquisition from source execution and minimize worker credential and
   Drive access. Offline flags and these precautions are not a sandbox.
9. **Unresolved links — accepted.** R6 distinguishes a missing decision from the
   valid conservative `unresolved` link; conflict/feasibility checks still apply.
10. **Git byte conversion — accepted, conditional configuration action pending.**
    R8/runbook explain raw-byte hashes, the proposed no-conversion attribute and
    a verified private ZIP alternative. No `.gitattributes` edit or lock commit
    was made during this documentation review.

Smaller corrections accepted: distinguish local draft commits from uncommitted
changes; name every hashed code/requirements file and preparation regeneration;
review fold counts after freeze; preserve a timestamp outside the runtime;
resolve patient-evidence publication before pushing; retain the S2 slip rule and
11 November report priority. These corrections add no scientific recipe change.

Remaining operational blockers are the unchecked transition items above. No
author decision or execution result is completed merely by this disposition.

The next operational instructions remain in [FM_RUNBOOK.md](FM_RUNBOOK.md).
This review has not downloaded assets/data, accepted terms, launched Colab,
changed the protocol or created an input lock.
