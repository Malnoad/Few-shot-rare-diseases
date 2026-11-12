# FM code preparation: review and verification record

> **Closed record (13 November 2026).** Kept as evidence of how decisions were reached; no longer updated. The study as it stands is in [`protocols.md`](../../protocols.md); the binding texts are `FM_PROTOCOL.md` and `FM_AMENDMENTS.md`.

9 November 2026. Lam authorized autonomous code preparation until 05:00
Asia/Bangkok and requested subagents plus repeated rebuttal rounds. This record
concerns draft code, not scientific results, an input lock or model feasibility.
The frozen [protocol](FM_PROTOCOL.md) remains unchanged. Operational limits and
commands are in [the runbook](FM_RUNBOOK.md).

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| Exact PCG64 folds/member streams and nested supports are implemented | Literal independently realized stream fixtures in `tests/test_fm_core.py`; saved full-roster tests in `tests/test_fm_prepare.py` | Verified on synthetic inputs |
| Prototype supports normalize before averaging; arithmetic is float64 | Hand-built unequal-magnitude supports, zero/cancelling/nonfinite guards and evaluator fixtures | Verified on synthetic inputs |
| Logistic mean loss, weight-only penalty, optimizer settings and gradient convergence gate match §6 | Objective/count-invariance/finite-difference tests and captured optimizer arguments in `tests/test_fm_core.py` | Verified on synthetic inputs |
| Stale preparation cannot silently change endpoints/streams at final lock | Producer code/recipe, deterministic semantic revalidation, source manifest/bytes, full-summary pilot SHA; explicit stale-summary/namespace/manifest tests | Verified on synthetic inputs |
| Pilots use the fixed IDs, process order and measured recipe | Realized IDs, operation receipts, distinct PIDs, ordered nonoverlapping windows, same FM environment/runtime/hardware tests | Verified structurally with mocks; GPU timing unverified |
| Typed revisions alone prove staged asset identity | Counterexample: arbitrary same-architecture weights could be hashed under a typed revision | False in initial draft; now rejected using official weight/config/source digests |
| Caches retain raw FP32 and reject wrong identities | Cache path/content/order/dtype/numerical tests; full receipt validation at evaluator boundary | Verified with synthetic arrays |
| Incomplete arms cannot receive partial nomination scores | Unequal-fold pooled counts, one-missing-cell rejection, fixed eligible FM set, reference exclusion and no-winner fixtures | Verified on synthetic inputs |
| Legacy reporting requires the prior nominee decision | Gate-before-feature-access tests; nomination hash in reporting identities | Verified structurally; no actual nominee exists |
| Nine-way FRR uses all nine outputs and true NORMAL denominator | Hand-counted confusion/FRR fixture and nine-way completion/export tests | Verified on synthetic inputs |
| Support-draw summaries imply patient confidence intervals | Protocol §7 and explicit summary interpretation metadata | False; mean/SD/percentiles describe conditional draw variability only |
| The real model stack and checkpoints load correctly on Colab | No real weights loaded and no new encoder run performed | Unverified; run the S2 pilot |
| Legacy-only reporting completes the external source component | External acquisition/screening/final query lock/inference are not wired into these commands | False; reduced scope remains explicit |

## Rebuttal rounds and dispositions

Round identifiers below refer to this task's app-owned delegated jobs. Reviewers
received the original brief plus earlier findings/dispositions on subsequent
rounds. They used separate file ownership; read-only reviewers did not edit code.

1. **Initial implementation/test round — Codex / gpt-6.1-sol.** Extraction owner:
   `fm_extract.py`, `requirements-fm.txt`, `tests/test_fm_extract.py`. Preparation
   test owner: `tests/test_fm_prepare.py`. Parent owned core/preparation/evaluation
   integration and docs. Preparation tests exposed mismatched FM timing environments
   and incomplete pilot completion hash maps. Both were accepted and fixed.
   Extraction source checks corrected native channel/resize order, DINO crop
   rounding, timing-validation overhead and cost/resume receipts.
2. **Planned independent protocol round — Claude / claude-opus-5-5.** Provider
   returned `ENETUNREACH` before findings. This is **not** a completed review or
   evidence of correctness. A fresh Codex task replaced that pass.
3. **Evaluator rebuttal — Gemini / gemini-4-argon-high.** Accepted cross-arm run
   identity, nine-way completion and partial-cost validation findings. Added
   corresponding guards/tests. Requiring every arm to finish before any incomplete
   study summary was declined: §7 explicitly requires failed/incomplete arms and
   available/planned paired counts. Completely unstarted/wrong run roots now stop.
   Root completion hashes are verified when a root marker exists; completed
   prototype cells remain usable without waiting for secondary probes.
4. **Protocol/persistence rebuttal — Codex / gpt-6.1-sol.** Accepted stale
   preparation semantics, missing realized timing/order validation, raw export
   protection bypass and missing split-block counts. Final lock now validates
   producer code/recipe and exact IDs/endpoints/source inputs, pilots bind the
   entire preparation summary, operation receipts/windows are checked, every
   archive/temporary/sidecar destination is protected and block counts include
   zeros. Excluded rare fitting rows no longer enter the extraction cache.
5. **Evaluator regression round — Codex / gpt-6.1-sol.** Added end-to-end synthetic
   control-flow and artifact tests for complete-only nomination, unequal-fold
   pooling, missing pairs, exploratory no-winner labels, contrast signs and export.
   Fixed malformed count broadcasting/truncation and unsealed/malformed completion
   and cost records. No scientific recipe changed.
6. **Extraction rebuttal — Gemini / gemini-4-argon-high.** Accepted the staged
   asset provenance gap: weights, configurations and source files now match
   verified public hashes, including the historical Keras cache verified against
   official MD5. Accepted full preparation-SHA validation at the feature boundary
   and deterministic clocks in mocked timing tests. Strict same-hardware/full-env
   extraction resume remains intentional fail-closed behavior; a portable completed
   cache can be evaluated on the locked CPU environment without loading weights.
   The runbook now explains this limit. A missing-runbook observation became stale
   when the parent's documentation work landed.
7. **Integrated round 4 — Codex and Gemini.** Gemini returned READY for code prep,
   with no P0/P1 findings and two nonblocking suggestions. Codex independently
   returned NOT READY: four P1 reproductions remained. All four were accepted:
   a fresh nomination output directory could re-designate the winner after
   reporting; patient evidence was not reconciled back into saved grouping;
   a bare input-lock ZIP lacked the required regenerated receipt; a thin root
   completion marker could omit all child predictions from export. The passing
   135-test gate did not cover those cross-command cases.
8. **Repair round 5 — parent plus Codex.** Added one study-level immutable
   nomination binding, including no-winner decisions and portable backup copies.
   A later nomination output copies the original without recomputation. Parent shared grouping derivation
   between freeze and verification, revalidated source fingerprints/ledgers,
   added hash-verified ZIP restoration and enforced completed-child coverage in
   all parent completion readers/writers/exports. Completed run roots also require
   sealed summary/cost metadata. Preparation55 and evaluator39 targeted tests passed.
9. **Counterexample rebuttal round 6 — independent Codex.** Returned **READY for
   code preparation only**, with no remaining P0/P1 findings within the stated
   trust assumptions. Independently executed all four round 4 counterexamples:
   late MIRAGE completion could not change the nominee; rehashed contradictory
   patient links stopped before pilots/lock; ZIP+saved-digest restoration recovered
   byte-identical members with original folders moved away; thin parent markers
   failed at writer, reader and exporter while a genuine report exported. Five
   additional focused lifecycle/serialization/binding/cost tests passed (9.112s).
   No actual encoder, model score or study gate was certified by this verdict.

**Numerical rebuttal detail.** The initial inclusive two-point comparison failed
on some representable values. The suggested `.85/.83` example did not reproduce
in the specified Python. A separate sweep found genuine examples (`.14/.12`,
`.20/.18`); a protocol-shaped count-array counterexample confirmed the latter.
The final comparison admits only float64 representation roundoff (eight machine
epsilons) at exactly0.02, with all hundredths boundary cases and a just-outside
case tested. No data-dependent threshold tuning occurred.

**Norm-order correction.** The 8 November coding plan reversed MIRAGE's norm/mean
order. The pinned upstream wrapper applies LayerNorm per token before patch-token
averaging. The plan and code now use that order. No old feature extraction used
the mistaken description; the protocol was not edited.

## Verification checkpoint

Specified local interpreter: `D:/Few-shot/.venv/Scripts/python.exe`.
The project environment is unchanged. The continuation audit installed planned
CPU dependencies and CPU-only PyTorch into a temporary environment. No pretrained weights,
real model execution, raw-data transformation, published-result overwrite, real
input lock, commit, push or PR occurred.

Combined focused run, 9 November:

```text
python -X utf8 -m unittest discover -s tests -p test_fm_*.py
Ran 135 tests in 111.486s
OK
```

The task runner saves its child PID under `.agents/pids/`, captures output under
ignored `.agents/reports/fm-code-prep/` and propagates the child exit status. A
deliberate exit7 probe returned7 at the tool boundary; it was a runner check, not
a failed project check. The wrapper is temporary task tooling, not an installed
permanent workflow fix.

The first full gate passed on the pre-round5 snapshot:

```text
Ran 29 tests in 0.029s     # core
Ran 52 tests in 97.849s    # preparation
Ran 25 tests in 2.860s     # extraction
Ran 29 tests in 13.351s    # evaluation
Identity check: PASS (2 outgoing commits; author and committer match local noreply)
Agent checks: PASS
Child exit: 0
```

The round 5 gate passed 148 FM tests before the extractor metadata change. The
latest full gate includes that change and its additional regression, captured in
`.agents/reports/fm-code-prep/check-034150.log`:

```text
Ran 29 tests in 0.031s
OK
Ran 55 tests in 111.848s
OK
Ran 26 tests in 3.946s
OK
Ran 39 tests in 11.891s
OK
Identity check: PASS (2 outgoing commits; author and committer match local noreply)
Agent checks: PASS
Child exit: 0
```

Those are **149 FM tests** (core29, preparation55, extraction26, evaluation39).
The same gate also passed the existing reproduction/guard tests and manifest and
notebook checks. `git diff --check` passed. The protocol SHA256 still equals
`4e8df8d91c0021187c05b103a9b0a0e0319a055fd91108da1f4cf3045d00c5df`.
Stage scripts, notebook, canonical manifest/results and report sources/PDF were
checked unchanged against HEAD.

Normalized source SHA256 prefixes at the native/CLI rehearsal snapshot (CRLF→LF):

| File | SHA256 prefix |
|---|---|
| `fm_core.py` | `fea7a5eb0c5a8a93` |
| `fm_prepare.py` | `8ddae5f9164e811d` |
| `fm_extract.py` | `c5f021f4a84e9e7e` |
| `fm_evaluate.py` | `b32477d2f573a801` |

No branch commit/push or actual lock was created. Model-cache, pilot, input-lock
and evaluation fixtures were explicitly synthetic. The separate score-free
staging preview below preserved actual selected image bytes in a temporary ZIP.

## Continuation audit before the 05:00 cutoff

The active timed goal continued after the first handoff; these are additional
checks, not a claim that Colab/model gates have passed.

- **Final preparation against actual inputs:** two score-free screen/freeze runs
  with source verification wrote only to Temp and were byte-identical. Observed
  groups27/26/25/19/27, zero file/RGB/dHash flags, 4,124 fitting +1,022 query rows,
  5,146 extraction images and reporting K={1,2,4,8,16}. The source manifest stayed
  unchanged. Evidence: ignored `final-real-preview.json` under the task reports.
- **Selected-image staging preview:** the actual 5,146 selected originals were
  packed into a 334,952,098-byte temporary ZIP and copied. Every member matched
  the frozen inventory hash and the copied ZIP matched SHA256
  `e9e2108f55c03c49660380d81e6bef3416c34b255e965e671a8bb850abd48c67`.
  This is a pre-lock staging check, not an author-approved input lock or upload.
  Evidence: `.agents/reports/fm-code-prep/final-stage-preview.json`.
- **Round7 CPU dependency verification:** temporary Python3.12.10 with exact
  NumPy2.2.6/SciPy1.15.3/Pillow11.2.1/pandas2.2.3/scikit-learn1.6.1 passed all148 FM
  tests on that snapshot, with no failures/errors/skips, as did the laptop stack.
  `pip check` passed; 14 synthetic frozen CSVs were byte-identical, including
  2,500 development and500 reporting rows. Environment guards rejected mixed
  reuse. This was Windows CPU evidence; full framework/Colab compatibility remains
  unverified. Evidence: `.agents/reports/fm-cpu-subset-20261109/evidence.json`.
- **Round7 primary-source loader audit:** exact pinned upstream sources and SDK
  signatures revealed one RETFound writer-compatibility gap. The authors' writer
  saves parsed `argparse.Namespace`; PyTorch2.7.1 does not permit that class by
  default in a restricted load. This does **not** establish the exact published
  OCT checkpoint's serialized types. The helper now temporarily permits only
  Namespace while retaining `weights_only=True`, preserves preexisting permission,
  and removes newly added permission on success/failure. The26 mocked extractor
  tests pass. Actual pretrained checkpoint loading remains unverified.
- **Round8 native restricted-load verification:** CPU-only PyTorch2.7.1+cpu in
  the temporary environment passed seven checks using the production helper and
  tiny synthetic tensor/Namespace checkpoints. Success/failure restored the
  allowlist exactly; existing permissions survived, unrelated metadata failed
  closed and nonmapping roots were rejected. `pip check` and the26 extractor
  tests passed. No encoder was constructed. Evidence:
  `.agents/reports/fm-round8-native-20261109/evidence.json`.
- **Round8 independent metadata review — Gemini / gemini-4-argon-high:** accepted
  the scoped fix after checking pinned PyTorch allowlist semantics and RETFound's
  writer. The published OCT asset's metadata types remain unverified; another
  unsupported type would still stop the restricted loader explicitly.
- **Native reference preprocessing:** ten temporary grayscale/RGB/RGBA PNG/JPEG
  fixtures with odd rectangular dimensions produced bit-identical float32
  299×299×3 tensors through the actual timing-preprocessing closure and historical
  `stage3_classifier.load_image`, using TensorFlow2.21.0 on CPU. Only the closure's
  preprocessing was executed with an identity callable; no encoder was constructed.
  Evidence: `.agents/reports/fm-code-prep/inception-preprocess-evidence.json`.
- **Round9 process-level rehearsal — Codex / gpt-6.1-sol:** all three synthetic
  arms completed7,500 prototype episodes,750 logistic fits and three nine-way
  audits. All111,402 prediction rows, pooled scores, denominators, NORMAL FRR and
  nomination matched expectations recorded before scoring. Controlled interruption
  preserved completed artifacts; completed CLI reruns were idempotent. Exact
  common cap1,000, K=16 eligibility and ZIP-only pre-S3 restore passed. The bounded
  agent verified copied DINOv2 feature/development exports, then stopped. Its
  partial status is retained in `ROUND9_PARTIAL_EVIDENCE.json` beneath the TEMP
  `SYNTHETIC-round9-cli-6osq2evr/ignored-process-reports` directory.
- **Parent continuation of the completed synthetic run:** nomination/summary
  exports included the binding and passed copied-byte verification. A fresh
  post-report nomination directory retained the original decision despite a new
  empty runs path; a second summary retained identical completion bytes. A late
  unavailable-arm gate was rejected. Restoring S2 alone rejected reporting until
  the preserved original S3 binding was restored, after which summary bytes
  matched. Evidence: `PARENT_STUDY_STATE_EVIDENCE.json` beside the partial report.
  Other per-arm export permutations were not repeated in this process rehearsal.
- **Round9 installed CPU libraries and parent native continuation:** requested
  direct pins installed only in TEMP, preserving the earlier torch 2.7.1+cpu and
  adding torchvision 0.22.1+cpu. Additional hash-verified wheels totaled21.3MB;
  `pip check` passed. The agent's own
  time guard stopped its probe before execution. The parent then ran it: the
  first probe found DINO's native one-pixel-height channel-inference rejection.
  Follow-up passed seven checks and189 transform comparisons, explicitly asserting
  those three native rejections. Real selected image headers showed minimum
  width290 and height258, with none at ambiguous heights1/3. No production
  preprocessing override was added. Evidence: ignored
  `fm-round9-transforms-20261109/native-transforms-followup.json` and
  `fm-code-prep/real-selected-geometry.json`; the initial failed probe is retained.
  OpenCLIP normal import attempted its local SimpleTokenizer constructor and was
  stopped before the body; its unmodified installed image-transform module was
  compared separately. This does not certify full package initialization or any
  pretrained encoder. No encoder, text input, model download or network call ran.

After these rehearsals, the only source edits corrected CLI help for archive
restoration and an OpenCLIP tokenizer comment. Scientific and persistence logic
did not change. The last full gate covers these help/comment corrections too.
Its log is `.agents/reports/fm-code-prep/check-044818.log`: core29,
preparation55, extraction26 and evaluation39 all passed, followed by the existing
project/guard checks, `Identity check: PASS`, `Agent checks: PASS`, and child exit0.
The gate finished just after05:00; no further implementation work was started.
Final normalized source SHA256 prefixes are `fea7a5eb0c5a8a93` (core),
`d8c83038ccbdc653` (prepare), `8db7e10596a105cf` (extract) and
`b32477d2f573a801` (evaluate). `git diff --check` passed after the source changes.

The TensorFlow synchronization API used by the reference is present at the
declared main requirement floor: [TensorFlow2.16.1 source](https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/framework/test_util.py#L3896).
That is API availability evidence, not GPU timing verification.

RETFound source evidence: [parsed args caller](https://github.com/rmaphoh/RETFound_MAE/blob/ae9a9ecf37857cf47b8aa9f87cd6f710d75db287/main_finetune.py#L403),
[checkpoint writer](https://github.com/rmaphoh/RETFound_MAE/blob/ae9a9ecf37857cf47b8aa9f87cd6f710d75db287/util/misc.py#L290),
[PyTorch restricted unpickler](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_weights_only_unpickler.py#L173).
Other examined timm/MIRAGE/DINO/OpenCLIP signatures showed no confirmed additional
defect; actual public checkpoint key layouts and native loading remain unverified.

## Changed files

| Area | Files | Purpose |
|---|---|---|
| Implementation | `fm_core.py`, `fm_prepare.py`, new `fm_extract.py`, new `fm_evaluate.py` | Exact protocol rules and guarded artifact lifecycle |
| Verification | `tests/test_fm_core.py`, `tests/test_fm_prepare.py`, new `tests/test_fm_extract.py`, new `tests/test_fm_evaluate.py`, `.agents/check` | Focused scientific and persistence regressions |
| Environment/housekeeping | new `requirements-fm.txt`, `.gitignore`, `.agents/mistakes.md` | Proposed isolated runtime pins, weight exclusions and behavioral notes |
| Documentation | `README.md`, `COLAB.md`, `CONTEXT.md`, `FM_CODE_PLAN.md`, `FM_CHECKLIST.md`, new `FM_RUNBOOK.md`, this file | Draft status, corrected MIRAGE description, evidence, commands and remaining gates |

## Pre-completion surfaces

- **Entry points:** new flags are documented in `FM_RUNBOOK.md`. Existing stage
  scripts and reproduction notebook cells are untouched; no FM cell was added.
- **Reverse states:** writer ownership has an explicit token unlock; mismatched
  artifacts stop without overwriting. A new approved version preserves the old
  lock and results; no destructive reset command was introduced.
- **Data/schemas:** only fresh FM caches use the new ordered path/content/asset/
  environment/code identity. The canonical manifest, historical arrays/heads and
  recorded results stay unchanged.
- **Docs:** README, COLAB, CONTEXT, coding plan and checklist distinguish draft
  code from real gates/results. Author affiliations, course template and submission
  hour remain Lam's administrative decisions; no LaTeX/PDF edit was made.
- **Checks:** final 149 FM tests and full `.agents/check` pass; round 6 re-tested the
  four prior P1 cases independently. This remains code-prep verification only.

## Pre-commit recheck (9 November)

Rerun on the exact bytes committed: 149 FM tests pass (29 core, 55 preparation,
26 extraction, 39 evaluation), as do the guard and outgoing-identity checks. The
four source hashes match the risk record; the protocol hash and manifest are
unchanged; the changed files contain no weights, tokens, images or archives.
`tests/test_fast_head_tf.py` failed twice with "An Application Control policy has
blocked this file", naming a different unsigned TensorFlow DLL each time, then
passed. Windows Smart App Control is on and that code is identical to `main`, so
this is a laptop policy interruption of `.agents/check`, not an FM code failure.

## Still needed from Lam / Colab

Lam's paced §4/§6 walkthrough is not marked confirmed by autonomous coding. The
earlier Rule 1 proposed confirmation remains an agent assessment, not Lam's sign-off.
Review the code and rebuttal record before the final input lock. Lam asked to
finalize the code state on 9 November, which covers the local commit only.

S2 still needs actual access/download/load/terms evidence, correct checkpoint key
reports, sequential GPU timing and a preserved timestamped real input archive.
The external definitions/query screen, final external IDs and required external
NORMAL source component are not completed by this prep. Affiliation/template/
submission-hour confirmation and the S1 PDF freeze remain separate pending work.
