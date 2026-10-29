# T12 equivalence gate (declared before speed measurements)

Gate: identical float32 weights and explicit per-epoch permutations, batch 10,
at least one complete epoch including a partial batch. Require max absolute
kernel/bias difference < 1e-5 and absolute val_loss difference < 1e-6.
These bounds allow float32 reduction rounding, but are much smaller than an Adam
step (learning rate .001). Compare optimizer slots/iterations too for diagnosis.
No speed or accuracy outcome can relax this gate.

Behaviour: seeds 1, 2, 3, 25 epochs each, native independent epoch shuffles,
full cached gan training/validation/test sets with Stage 3's seed-specific split.
Report curves and final test accuracy; no statistical equivalence claim from 3 seeds.
Speed: compare compiled epoch variants on this CPU only after exactness passes;
report compilation/startup separately and project 25-epoch times to 250 epochs.
Keep batch 10, Adam .001 and 250-epoch production ceiling unchanged.

## Implementation and evidence

`fast_head.EpochLoop` retains float32 features and labels as tensors and compiles
one training epoch and one validation epoch with `tf.function` (no XLA). Each
batch calls the existing model's `train_step` / `test_step`. Production Stage 3
constructs the same Dense(9, softmax), with its default GlorotUniform kernel and
zero bias, before selecting a trainer. No optimizer equation is reimplemented.

Inspected installed Keras 3.15.1 sources:
`keras/src/backend/tensorflow/trainer.py` train_step weights the loss tracker by
the actual batch size, including the partial last batch. Both paths therefore
report the sample-weighted epoch mean, not an unweighted mean of batch means.
`keras/src/optimizers/adam.py` defaults are beta1=.9, beta2=.999, epsilon=1e-7,
amsgrad=False. Its update uses m += (g-m)(1-beta1), v += (g*g-v)(1-beta2),
alpha = lr*sqrt(1-beta2**t)/(1-beta1**t), w -= alpha*m/(sqrt(v)+epsilon).
Both paths invoke that same implementation and the same sparse cross-entropy
(including Keras's handling of softmax logits).

Epoch indices use a new `tf.random.shuffle` permutation of every training row;
there is no replacement or dropped remainder. Keras's array adapter also uses
TensorFlow permutations, but its seeded epoch iterator can repeat them (see the
measured probe below). Native trajectories need not be identical for the same seed.
The explicit-order gate isolates numerical updates from this RNG difference.

The same ModelCheckpoint, CSVLogger and EarlyStopping callbacks receive each
epoch's metrics. Early stopping is off at patience=0. The common Stage 3 tail
saves last_head before optionally loading best.weights.h5 into head.npz.

C0/C1 select fast and lock its name plus training script hashes into immutable
run identity. C0's already-extracted caches retain their original identity:
only the exact inspected pre-T12 c0_experiment/stage3 hashes (LF and CRLF) are
compatible; normalization recipe, environment, image lists and cache hashes
remain checked. No blanket acceptance of arbitrary older code was added.

Stage 3's default remains keras. To reverse the selection use keras in a new
experiment directory; a trainer/configuration mismatch stops resume. Future
C2/C3 runners must explicitly select fast and lock it in their run identities;
neither runner currently exists. No C0/C1 outcomes or canonical results are
created by this benchmark. The original pools and feature lists do not change.

The sole hard-coded Drive-prefix assertion was notebook 4.1. It now checks
`C1_DRIVE.resolve().is_relative_to(Path(D).resolve() / "c1")`. C0/C5 cells and
scripts were inspected and do not contain the same MyDrive-prefix assumption.
Colab execution through an actual shared-folder shortcut remains unverified.

## Laptop benchmark (2026-10-29)

Ryzen AI 9 365, Windows CPU, TensorFlow 2.21.0 / Keras 3.15.1. Full cached gan:
91,860 train, 11,513 validation, 1,022 test. No full sweep or 250-epoch fit locally.
[Raw curves, times, cache hashes and shuffle probe](T12_HEAD_BENCHMARK.json).
Reproduce: `D:/Few-shot/.venv/Scripts/python.exe benchmark_head.py --cache D:/Few-shot/results/features --output <new-report.json>`.

| Claim | Evidence | Status |
|---|---|---|
| Identical-order numerical updates | 91,863-row epoch (full train + 3 repeats), max kernel/bias and optimizer-state difference **0.0**, val_loss difference **0.0**, both val_loss **0.3023718894** | verified: gate passes |
| Matching native seeded outcomes | All three fast final test accuracies lower; native curves differ | false for these runs |
| Faster compute | Sum of paired 25-epoch times: Keras 941.46 s, fast 200.54 s, **4.69x** | verified on laptop only |
| Colab/Drive and full-250 wall time | Not measured | unverified |

| Seed | Keras / fast 25-epoch seconds | Keras / fast final val_loss | Keras / fast test accuracy % | Curve mean / max absolute loss difference |
|---|---|---|---|---|
| 1 | 342.38 / 67.24 | .269198 / .292293 | 93.64 / 88.94 | .015544 / .044323 |
| 2 | 327.51 / 77.20 | .272521 / .330929 | 93.05 / 87.18 | .021711 / .058407 |
| 3 | 271.57 / 56.10 | .307653 / .308687 | 91.59 / 90.31 | .018245 / .062219 |

**Behavioural limitation:** compiled native Keras fit in the retained 23-row,
three-epoch probe repeated the same permutation each epoch despite shuffle=True
and set_random_seed(1). Fast freshly reshuffles each epoch, as requested. Both
use uniform permutations, but their epoch-order behaviour is not identical on
this installed version. This is a plausible explanation for different curves,
not a controlled attribution of the accuracy gaps. We do not claim behavioural
or statistical equivalence. Do not pool old Keras results with new fast results.
The user-declared pass/fail gate was identical-order exactness, which passed;
C1 therefore selects fast. C0 support is retained but its heads are cancelled by
D20=A (29 Oct 2026). Historical Stage 3 default remains keras.

Ordinary tf.function probe: 3.18 s cold / 2.85 s warm epoch. XLA compiled but
its first epoch still had not finished after >90 s; that process was stopped.
No XLA speedup is claimed. The chosen path is the ordinary compiled epoch.

Per-250 projections from each 25-epoch fast fit: **11.21 / 12.87 / 9.35 min**.
Use the slowest, **12.87 min/head**, for conservative largest-variant budgets.
The paired current Keras projections are 57.06 / 54.59 / 45.26 min; the older
34.60-min record is historical, not this session's comparison baseline.
Compute timing includes tracing/validation, excludes cache loading, predictions,
checkpoint writes and Drive I/O. These costs must be added to session budgets.
Colab CPU/GPU is unbenchmarked; revise time forecasts from the first actual head.

| Serial compute budget | Fast, laptop projection |
|---|---|
| C0 historical hypothetical 180 es20 heads (cancelled by D20=A) | **3.1-7.7 h**; 250-epoch ceiling **38.6 h** |
| C1 first 20 primary (60 fixed250 heads) | **12.9 h** |
| C1 60 es20 sensitivity heads, assumed 20-50 epochs | **1.0-2.6 h** |
| C1 first stage total | **13.9-15.4 h**; all-250 ceiling **25.7 h** |
| C1 provisional 63-pair primary + sensitivity | **41.6-43.1 h**; all-250 ceiling **53.4 h** |
| C1 top-up alone (129 heads) | **27.7 h** |
| C1 243-pair cap primary alone (729 heads) | **156.3 h**, plus sensitivity |

D20=A (29 Oct 2026) closes C0 at its gate as a limitation and fixes current
inputs for C1 and later C2/C3. No C0 heads are planned. The 20-50-epoch ranges are assumptions,
not observed es20 stopping times. Counts, D15 stopping/top-up and D7 pools stay fixed.

## Verification surfaces

Entry points: Stage 3 flag, C0 train flag and notebook C0.4/4.1 agree.
Reverse: select keras in a fresh output workspace/version. Data/cache lists:
unchanged, no invalidation. Docs: README, COLAB, guide, CONTEXT and C1 plan updated.
Focused checks cover provenance, shortcut containment, explicit-order gradients,
partial batches, loss metrics, optimizer slots and best/last/early stopping.
`.agents/check` must pass again after the final rebase.

Windows Code Integrity transiently blocked TensorFlow/wrapt .pyd imports. A normal
retry succeeded without changing policy or files; the full exactness gate and
all six short fits then completed locally. No policy workaround was used.
