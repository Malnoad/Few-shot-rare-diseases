# C2/C3 Colab output audit — 5 November 2026

**Historical snapshot:** C2 and C3 subsequently completed. The
[6 November stage closeout](REPLICATION.md) supersedes this report's pending
status; the original errors, recovery and observation cutoff remain preserved below.

C2 already has substantial training output. Its latest observed recovery cell
reports **96/100 completion markers and a successful unlock**. C3 and the C2
aggregate summary have no completed output in the inspected notebook. The saved
failures concern restarting the experiment, rather than a demonstrated failure
of model evaluation.

Observation cutoff: **2026-11-05 06:34:49 UTC / 13:34:49 Bangkok**. The notebook
changed during inspection; earlier failures and the later recovery are recorded
separately below. This is a snapshot, not a continuing watcher.

Sources: [CPU notebook](https://colab.research.google.com/drive/1Od2i179FZfL2tPsL06Rw7ITDSFarTNe5#scrollTo=c2-run),
[C2 output folder](https://drive.google.com/drive/folders/1eiknB_zZ5-N58egodczXimjxQ6d-lVbN),
and [fixed design](C2_C3_DESIGN.md). Read-only browser inspection captured the
notebook's source, outputs and execution metadata, plus selected Drive previews.
The detailed local capture is `.agents/reports/c23-2026-11-05/evidence.json`
(gitignored). No cell was executed, runtime connected, lock removed, or remote
notebook edited by this audit. Canonical results and data were untouched.

## What each 4b section actually shows

Notebook metadata timestamps are reported in UTC; they are saved execution
timestamps, not observations of a currently running process.

| Section / cell ID | Saved execution | Output and interpretation |
|---|---|---|
| 4b.1 `c23-restore` | 2 Nov, 13:19:11.765; `ok` | Inputs restored; revision pinned to `a9ea5429ebac7f1c034a9e706f0e3b1721b7dbd0`; input root `/content/c23_inputs/current`; Drive output folders `c2/v1`, `c3/v1`. |
| 4b.2 `c23-prepare` | 2 Nov, 15:42:20.999; `ok` | `C2/C3 cache bundle ready` and `Bundle ready. Cached heads below run on CPU, explicitly trainer=fast.` Saved elapsed time 1,368.146 s (~22.8 min). This is preparation/verification, not classifier training. |
| Extra recovery cell `a8vPgtHGRxbA`, earlier capture | 5 Nov, 06:28:01.049; `error` | `AssertionError` at `assert Path("/content/p/c23_experiment.py").is_file()`. Unlock was never reached in that attempt. |
| Same recovery cell, latest capture | 5 Nov, 06:33:09.021; `ok` | Temporary runner `/content/c23_unlock_recovery/c23_experiment.py`; output reports 96 markers, `Unlock completed`, `Lock still exists: False`. |
| 4b.3 `c2-run` | 2 Nov, 15:42:26.770; `error` | `ValueError: Writer lock exists: …/octfsl/c2/v1/writer.lock. Verify the old session stopped, then use unlock --token <saved token>`. Failure at entry to `writer_lock(self.output)`. This red traceback predates the successful unlock. |
| 4b.4 `c2-summary` | 2 Nov, 15:42:26.778; `aborted` | No output; summary did not run in that execution sequence. |
| 4b.5 `c3-run` | 2 Nov, 15:42:26.795; `aborted` | No output; no C3 head launched by this attempt. |
| 4b.6 `c3-summary` | 2 Nov, 15:42:26.802; `aborted` | No output; no C3 summary/evaluation result. |

The latest recovery stdout is:

```text
Latest file: s020/n0500/best.weights.h5
Latest update: 2026-10-30 08:31:01+00:00
Completion markers: 96
Nếu không còn writer C2 hoạt động, gõ UNLOCK C2: UNLOCK C2
Unlock completed
Lock still exists: False
```

The final Drive listing also omitted the old `writer.lock`; its former file
preview reported it was in the owner's trash. This audit did not remove it.
The output confirms a recovery action, not a subsequent successful training run.

## Causes and consequences

**Existing writer lock.** `writer_lock()` exclusively creates `writer.lock` and
refuses entry if one exists ([runner](../../c23_experiment.py#L106), lines
106–130). Ordinary exceptions remove a lock owned by that invocation; a hard
runtime loss can bypass cleanup. The recovery cell identifies the old owner as
host `a45ec20359cd`, PID `4036`, and reports no result update after 30 October.
A stale lock following interruption is consistent with this evidence, but the
original termination reason (quota, disconnection, manual stop, etc.) is
**unverified**. A local PID check in another Colab host cannot establish whether
the old writer is alive.

The 4b.3 traceback stops before `verify_inputs()`, `completed()`, or any new Stage
3 subprocess starts ([runner](../../c23_experiment.py#L371), lines 371–406).
Consequently this attempt neither invalidates earlier results nor revalidates
all their artifact hashes. `summarize()` uses the same writer lock, so rerunning
4b.4 while that lock existed would also have failed (lines 455–469).

**Missing recovery runner.** The earlier assertion proves that
`/content/p/c23_experiment.py` was absent in that recovery runtime. Part 1 normally
clones the repo there; Colab's `/content` storage is separate from persistent
Drive outputs. A fresh runtime without completed setup, or a different checkout
location, can produce this error. Which occurred is **unverified**. The failure
does not demonstrate that Drive results disappeared. The later temporary runner
successfully reached the official `unlock` command; that does not restore the
normal notebook helpers and training inputs by itself.

**Aborted downstream cells.** The execution metadata places the three aborts
immediately after C2's lock error. No numerical evaluation exception is recorded
there. No run-level accuracy comparison can be inferred from their empty output.

## Prior training and evaluation evidence

Drive contains seed directories `s001` through `s020`. In
[seed 20](https://drive.google.com/drive/folders/100uIJ01zmlTxvH4pvq0azL_Qq6bksTH5),
only `n0000` and `n0500` were listed:

- **`s020/n0000` completed.** Its folder contains `complete.json`, heads,
  predictions, history, counts, trainer metadata and `run.log`.
- **`s020/n0500` is incomplete.** Its
  [status](https://drive.google.com/file/d/1LYp6SkkPXreGitQuNYq0maDqHsRAttJl/view)
  says `{"state": "running"}`, but the listing contains only six files:
  `best.weights.h5`, `data_counts.csv`, `trainer.json`, `status.json`, `run.json`,
  `selection.json.gz`. No completion marker, final heads, predictions,
  `history.csv` or `run.log` were listed. A persisted `running` value is not a
  process heartbeat. The abrupt loss/synchronization explanation is plausible;
  the precise reason those files are absent is unverified.
- `s020/n1000`, `n2000`, `n3000` were not listed. Combining this with the reported
  96 markers and the fixed ascending schedule suggests **four heads remain**:
  restart `s020/n0500`, then run the three absent conditions. This is conditional
  on the runner validating all prior completed artifacts; the audit did not
  independently enumerate and rehash all 96 markers.

One directly inspected result is
[seed 20's basic-only completion marker](https://drive.google.com/file/d/1mzmYWhIA4ez21fg6x5uCtUmrab4sA9rV/view):

| Metric | Recorded value |
|---|---:|
| Test accuracy | 91.97651663405088% = 940/1,022 correct |
| MCC | 0.8983943613310622 |
| Rare correct | 18/22 |
| Rare macro recall | 0.81 |
| Common macro recall | 0.922 |
| Recorded head duration | 14.859710349883487 min |

`n0000` means **basic augmentation with zero GAN images**, not real-only training.
This is one seed/condition, not a C2 mean or evidence of an optimal GAN amount.

The corresponding [run log](https://drive.google.com/file/d/1YEnxP2YArl03msTAqcjcBSAJOOTafb5_/view)
displays all 250 epochs, followed by test accuracy `92.0%` and recalls
NORMAL 0.98, CNV 1.00, DME 0.91, DRUSEN 0.80, CSC 0.80, MH 1.00,
MacTel 0.50, Stargardt 1.00, RP 0.75. Rounded recalls agree with the marker.
The CUDA-driver and `cuInit ... UNKNOWN ERROR (303)` messages precede successful
CPU training and predictions; they are not fatal in this inspected run.

Validation loss is lowest at epoch 6 (0.2602923512458801), versus
0.35294246673583984 at epoch 250; training loss decreases from 0.2813205122947693
at epoch 6 to 0.24442628026008606 at epoch 250. This is consistent with poorer
late-epoch validation generalization. Keeping the final head is the approved
fixed250 protocol, not a checkpoint-selection bug. This single trace does not
justify changing the protocol after seeing results or choosing a best test run.
The marker reports identical hashes for `head.npz` and `last_head.npz`; the
underlying NPZ files were not independently replayed here.

No `runs.csv`, `summary.csv`, `paired_differences.csv` or `contrasts.csv` appeared
in the inspected C2 root. C3 has no completed output in this notebook and no C3
folder was listed in the inspected project Drive root. Other notebooks/versions
were not exhaustively searched. The GAN-amount and imbalance conclusions remain
unavailable. Their design limitations (fixed patients, source/disease confounding,
changes to fitting size/update count) remain as documented in the fixed plan.

## Next actions, without changing the experiment

1. Do not repeat the old unlock merely because 4b.3 still displays its November 2
   error. The latest observed recovery removed that lock. If a new lock appears,
   establish its owner/session state before any further recovery.
2. Restore a compatible CPU environment with the pinned revision and the recorded
   package versions: NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, TensorFlow 2.21.0,
   Keras 3.13.2, scikit-learn 1.6.1. These are from
   [C2's identity](https://drive.google.com/file/d/135aGlLtambP88VYkYoWPT10s9-7S8lGq/view),
   not suggested upgrades. Do not bypass identity checks to resume.
3. In a fresh runtime, run only Part 1 → 4b.1 → 4b.2 → 4b.3. In an intact,
   already-pinned kernel, follow the plan's retained-kernel instructions instead.
   The runner validates each completed artifact before skipping it, then restarts
   an incomplete head from epoch 1. Do not restart or delete the whole sweep.
4. Run 4b.4 to create C2's four summary CSVs. A validated partial summary is also
   allowed by the existing code, but must be labelled incomplete. Collect
   `identity.json` and `schedule.json` with the CSVs. Then run 4b.5 → 4b.6 for C3
   if continuing that approved experiment, with one active sweep session at a time.

These actions are recommendations; none were executed by this audit.

## Verification and scope

All seven source-script SHA-256 values in the captured C2 identity match their
Git blobs at `a9ea5429ebac7f1c034a9e706f0e3b1721b7dbd0`. The local checkout is
at that revision. Local verification parsed the captured JSON, extracted all
250 logged epochs and checked the displayed accuracy/count consistency. This
does not certify all Drive artifacts, cache contents, legacy provenance or
training reproducibility.

Completion checklist: entry points and reverse actions unchanged; data/schemas
and caches unchanged; documentation updated through this report and the tracker.
No method or canonical result changed, so README/COLAB instructions were retained.
Verification command: `bash .agents/check` through Windows Git Bash, using the
main checkout's Python environment. All required suites passed. Final output
excerpt:

```text
Ran 7 tests in 18.324s

OK
Identity check: PASS (0 outgoing commits; author and committer match local noreply)
Agent checks: PASS
```

`git diff --check` passed. The self-contained local status view is
`.agents/reports/c23-2026-11-05/summary.html`; its 728px preview had no console
messages and was visually inspected. Inspection-command mistakes are logged in
`.agents/mistakes.md` entries 27–28; no proposed guard was installed.
