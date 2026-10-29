# C1: D15 fixed 250 epochs, final head, 1 pp target

Execution specification fixed 2026-10-27, before any C1 seed-pair result.
Code and notebook Part 4 are ready for Lam to run by hand on Colab. Local fitting includes the explicitly authorized T12 isolated cached-head
exactness gate and three seeds x 25 epochs per trainer;
no C1 pilot/sensitivity/top-up runs or feature/GAN training were performed locally.

## Question, inputs and protocol

On the committed split, does the accepted GAN pool improve mean test accuracy
over basic augmentation across classifier seeds? Primary contrast: **gan minus
basic in percentage points**, fixed250 final heads. A meaningful gain is **1 pp**.
None is a secondary baseline; rare/class-specific metrics are exploratory.

**D20=A (29 Oct 2026)** resolves D6: `INPUT_ARM = "current"` is fixed for
C1 and later C2/C3. C0 stops at its gate as a limitation. Format/framing
normalization did not reduce separation; disease and dataset source remain
confounded. No C0 heads are planned. Keep the harmless format/framing code
branch for future explicitly authorized protocols. No C1 feature extraction.

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
  C5 top-up lives in a separate pool **for C2/C3**, excluded from C1.
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
   imported or counted as new runs. C0 originally shared the es20 seeds; its heads are now cancelled by D20=A.
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

CPU runtime; Part 1 then **4.1 -> 4.2 -> 4.3 -> 4.4 -> 4.5**.
D20=A fixes current inputs: 4.1 restores the original Drive features and builds
isolated placeholders without Part 2/Stage 4. No normalized-input or C0 head run
is planned.

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

## Measured CPU head budget (D17=B)

D17=B uses `--trainer fast`, with the same Keras Dense/loss/Adam steps.
[Laptop benchmark](T12_FAST_TRAINER.md): **4.69x** paired compute
speed-up; **9.35-12.87 min/250 epochs**, projected from three 25-epoch gan fits.
C0 heads are cancelled by D20=A (historical hypothetical compute budget
3.1-7.7 h, 38.6 h ceiling). C1 first stage **13.9-15.4 h**, provisional
63 pairs **41.6-43.1 h**, including that assumed sensitivity cost; ceilings
**25.7/53.4 h** if sensitivity reaches 250. These are laptop-measured compute
projections, excluding checkpoint/Drive I/O; **Colab is unbenchmarked**.
Native-shuffle accuracy differed materially; see the report, not an equivalence claim.

First-20 primary alone is 12.9 h; 63-pair primary alone 40.5 h; top-up alone
27.7 h. At the 243-pair cap primary alone is 156.3 h plus sensitivity.
The old [34.60-min Keras timing](C1_HEAD_TIMING.json) is retained as historical.
Fast retains fresh per-epoch shuffles; native Keras 3.15.1 repeated its seeded
permutation in the diagnostic. The numerical gate passes, but native final
accuracies differ. Trainer and training-code identities prevent mixed resumes.
D20=A fixes current inputs; no C1 efficacy outcomes exist.
