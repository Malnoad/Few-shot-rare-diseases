# C1: D15 fixed 250 epochs, final head, 1 pp target

Execution specification fixed 2026-10-27, before any C1 seed-pair result.
Code and notebook Part 4 are ready for Lam to run by hand on Colab. The only
local fitting was the explicitly authorized, isolated cached-head timing below;
no C1 pilot/sensitivity/top-up runs or feature/GAN training were performed locally.

## Question, inputs and protocol

On the committed split, does the accepted GAN pool improve mean test accuracy
over basic augmentation across classifier seeds? Primary contrast: **gan minus
basic in percentage points**, fixed250 final heads. A meaningful gain is **1 pp**.
None is a secondary baseline; rare/class-specific metrics are exploratory.

D6: **input choice waits for C0 and Lam's decision**. Notebook 4.1 has one input
parameter, `INPUT_ARM = "current"`. Leave it there only if Lam retains current
inputs; otherwise set `format` or `framing` after C0 review. Normalized inputs use
Part C0's `data/c0/v1/<arm>/` and `results/c0/v1/<arm>/features/` with its completion
and identity records. Do not choose inputs using C1 outcomes. No C1 feature extraction.

- Primary fixed250: every variant reaches 250 epochs, `--patience 0 --restore last`.
  `head.npz` is the final-epoch head; `last_head.npz` also retains the last head;
  `best.weights.h5` separately retains the lowest-validation-loss checkpoint.
- Sensitivity es20: 250-epoch ceiling, `--patience 20 --restore best`, exactly
  seeds 1-20, matching C0's seeds and default checkpoint protocol. No es20 top-up.
- Stage 3's default stays `--restore best`: **patience 0 alone still restores best**.
- Keep Adam 0.001, batch 10, existing backbone caches and pool construction.
  The repository replaces rare real training with basic augmentations in basic/gan;
  none has no rare validation. The seed also changes the augmented validation draw.
- D7: keep the **current accepted GAN realization**, cap 3,000 per disease:
  CSC 3,000; MH 2,610; MacTel 1,248; Stargardt 2,199; RP 832. This deviates from
  the paper's 3,000/class, especially MacTel/RP. No count matching or Stage 2 changes.
  C5 top-up lives in a separate pool **for C2 only**, excluded from C1.
  Every C1 command explicitly pins `--gan-pool v1`; v2 routing from C5 is preserved.

The [original methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7829497/) specify
250 epochs, batch 10 and frozen convolutional layers; they do not establish the
checkpoint choice or the repository's undocumented learning rate. Final-head
selection is Lam's operational choice, not a verified literal paper implementation.

## Stopping and top-up rule (locked before outcomes)

1. Retrain **seeds 1-20**, all three variants, under fixed250 and es20: 120 heads.
   Run ascending seed blocks, fixed250 then es20, none/basic/gan within each arm.
   Both protocols for all 20 seeds must complete before any top-up. Historical
   seeds 1-5 overlap in numeric labels only: their inspected outputs are never
   imported or counted as new runs. C0 shares the es20 seeds for comparison.
2. Once only, compute the sample SD (ddof 1) of the 20 **fixed250 gan-basic**
   accuracy differences. Retain full-precision accuracies; units are pp.
3. Calculate the smallest integer n with two-sided paired-t power >=0.80 for a
   true mean difference of **1 pp versus zero**, alpha 0.05. Use Student-t critical
   value t(.975,n-1), noncentrality sqrt(n)/SD and both tails of the noncentral t.
   **Target N = max(63, calculated n)**, including the first 20 seeds. Zero sample
   SD keeps the floor of 63. The historical es20 pilot SD 2.7697508913 gives 63;
   exactly 2.8 gives 64. Thus 63 is provisional, not a fixed significance threshold.
4. Predeclared feasible cap: **243 total pairs** (the earlier 2x-pilot-SD scenario).
   If power requires more than 243, stop after the first stage and report the
   target as beyond budget/incomplete. Do not silently truncate the target,
   claim adequate power, or increase the cap after seeing outcomes in this sweep.
5. Save immutable `topup.json` and the schedule **before** training any seed >20.
   Run fixed250 only for consecutive seeds **21..N**, three variants each.
   Finish that schedule and stop. No second variance recalculation, no early
   efficacy/futility stopping, no extension based on means, CI, p-value or signs.
   A failed head is retried under the same identity/seed; no replacement seed.
   An interrupted schedule is reported incomplete, without a final decision.

The first 20 pairs are included in the final estimate (internal pilot); they
estimate the final-head variance, not the already-inspected es20 variance.
Power is conditional on the estimated SD and paired-difference model, not guaranteed.
The ordinary final paired-t CI is **nominal** under this variance-adaptive sample
size; it is not an exact sequential confidence procedure. No claim of exact
adaptive type-I error control is made. Persisting the target prevents repeated looks
from changing the schedule. Do not interpret the pilot summary as a stopping test.

Calculation implemented in pure `c1_experiment.topup_rule` using
[SciPy noncentral t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html).
No new dependency is added. The schedule is deterministic to make ordering and
resume auditable; the older candidate seeds 101-163/randomized schedule are superseded.

## Summary and D15 interpretation

Notebook 4.5 writes the per-seed gan-basic differences for each protocol,
all variant-pair means/SDs, paired-t 95% intervals and two-sided Wilcoxon p-values.
Primary analysis uses only fixed250. Es20 is sensitivity, never a substitute
primary because its result is more favorable. Other contrasts are secondary.

For a complete schedule, interpret gan-basic's interval [L,U] as follows:

- L >= 1 pp: interval supports a gain of at least 1 pp.
- 0 < L < 1 pp: positive gain, but a 1 pp gain is not established.
- U < 0: interval favors basic.
- Otherwise, U < 1: interval excludes a 1 pp GAN gain; no equivalence claim.
- Otherwise: inconclusive against zero and the 1 pp target.

Always report the point estimate and interval relative to **both** zero and 1 pp.
Power to reject zero at a true 1 pp gain is not power to prove gain >=1 pp.
Missing pairs are not imputed; partial runs cannot trigger the final rule.
Inspect the saved differences for skew, outliers and symmetry without changing
the primary test to chase significance. Wilcoxon is a sensitivity test requiring
symmetric paired differences; ties/zeros are reported and SciPy method auto is
version recorded ([API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)).

The seed is the unit for training-randomness inference. Every head shares the
same 1,022 test images (22 rare), augmentation pool and GAN realization. More
seeds do not add patients, address patient leakage or establish clinical validity.
Patient-cluster uncertainty from L9 remains separate from seed uncertainty.

## Output protection and manual Colab order

CPU runtime; Part 1 then **4.1 -> 4.2 -> 4.3 -> 4.4 -> 4.5**. Wait for C0 input
selection before 4.2. Current needs the original Drive features only; 4.1 builds
isolated placeholders without Part 2/Stage 4. For a normalized arm, restore real
inputs with 3.1/3.2/3.4b, then C0.1/C0.2 to restore metadata/caches and recreate
normalized data; no C0 feature extraction or heads need repeating for C1.

All C1 artifacts go directly to **MyDrive/octfsl/c1/v1/<INPUT_ARM>/**, under
`<fixed250|es20>/sNNN/<variant>/`. Never run 3.5, 3.6 or Part 6 for this sweep.
No temporary writes to canonical Stage 3 folders, Stage 4 or seed_check.csv.

Preflight verifies all required cache path lists, shapes and finite features,
fixed GAN/basic counts and test identities. `--cache-only` refuses missing/stale
caches before extraction. Save manifest/cache hashes, script hashes, code revision,
package versions and configuration in `identity.json`. Normalized arms additionally
verify C0's cache identity and manifest. Legacy caches cannot establish historical
backbone weights or input-byte provenance; that remains **unverified**, not
certified by a new file hash. Cache SHA256 locks the actual feature values used.

Each head saves last/selected/best checkpoints, all test/validation probabilities,
counts, history, command, run log and status. `complete.json` is written last and
contains artifact hashes. Completed runs are skipped only after hashes and identity
match; mismatch stops without overwriting. Interrupted heads restart from epoch 1
in their same incomplete directory. Disconnects retain completed Drive heads.
Source/canonical hashes and cache hashes are checked again after success/error/
interruption. Use one active notebook sweep at a time. To change inputs/protocol
or reverse a selection, retain the old version and select another arm/new version;
never delete a completed experiment to make resume pass.

## Measured CPU head budget

[Timing record](C1_HEAD_TIMING.json), 2026-10-27: cached gan seed 1, batch 10,
Adam 0.001, no early stopping, Ryzen AI 9 365, Windows CPU, TensorFlow 2.21.0.
The largest variant has 91,860 training rows and 11,513 validation rows (data_counts
from the isolated run), 9,186 batches/epoch. At about 8 s/epoch, 250 would exceed
30 minutes, so timing stopped after **25 completed epochs: 207.58 s (3.46 min)**,
including startup/cache loading. The next epoch had just started at termination;
no completed C1 head or seed pair was counted. Linear 10x extrapolation is
**2,075.82 s = 34.60 min per 250-epoch head**; conservative because it repeats
startup overhead. Timing output stayed in ignored .agents/reports/c1-timing.

Using that largest-variant rate for every variant, serial CPU estimates are:

- First 20 primary seed blocks: 60 fixed250 heads, **34.6 h**.
- Es20 sensitivity: 60 more heads. If 20-50 epochs/head, **2.8-6.9 h** (assumption,
  not measured); first stage total approximately **37-42 h**. Worst case es20 also
  reaches 250: total **69.2 h** for all 120 heads.
- Provisional 63-pair primary stage: 189 fixed250 heads, **109.0 h**, plus the same
  sensitivity cost: approximately **112-116 h**, ceiling **143.6 h** if all es20
  runs reach 250. Top-up alone (seeds 21-63): 129 heads, **74.4 h**.
- At the 243-pair cap: 729 primary heads, **420.4 h**, plus sensitivity. This is a
  feasibility ceiling, not a recommendation to spend that time without review.

Drive checkpoint/log I/O, CPU allocation, Colab TensorFlow version and runtime
limits can change these estimates. Print actual first-head minutes and revise
**time forecasts only**, never the stopping rule based on outcomes. Resume across
sessions. GPU head timing is **unverified**: this tiny Dense(9), batch-10 workload
has many small steps and a GPU is not required. Paying for GPU solely for heads
is not justified by this CPU measurement; use CPU initially and consider a
supported Colab GPU only if a measured head speedup offsets its cost. C0 fresh
feature extraction still benefits from A100/L4/T4; no Blackwell runtime.

Open before execution: Lam's C0 input choice and willingness to allocate the
resulting multi-session runtime. No C1 efficacy conclusion exists yet.
