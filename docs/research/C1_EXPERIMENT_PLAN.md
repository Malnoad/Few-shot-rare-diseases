# C1 preparation: fixed 250 epochs, 1 percentage point target

Status: training-free plan, updated 2026-10-27 (D15). **Lam selected option A (250 epochs,
final head primary) and a 1 percentage point meaningful GAN-over-basic gain.**
No training, new packages, classifier changes or notebook changes were made by
this preparation. This plan does not authorize training; D1 separately authorizes integrating the plan.

## Question and endpoint

On the existing fixed data split, feature caches and accepted GAN realization,
does adding the accepted GAN pool improve overall independent-test accuracy
compared with basic augmentation, averaged over new classifier seeds?

- Primary contrast: `gan - basic` final-head accuracy, in percentage points.
- Protocol: all classifier runs reach 250 epochs; final-epoch head is primary.
  Save the best-validation head separately. Run `es20` (early stopping with
  patience 20 and best-validation head) on the first 20 seeds as a sensitivity check.
- `none` is a secondary baseline, trained under the same fixed-epoch protocol.
- Keep the current train/validation pools, preprocessing, backbone features,
  optimizer settings and batch size fixed and fingerprinted. The current code
  defaults to Adam, learning rate 0.001 and batch size 10; those are repository
  settings, not proof of the paper's undocumented learning-rate value.
- Do not silently count-match the rare training pool, top up GAN images, change
  the test split or alter validation composition. Those are separate experiments.

The [original methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7829497/) specify
250 epochs, batch 10 and a frozen convolutional backbone. They do not establish
the checkpoint-selection details needed to prove exact identity with this plan.
Final-head primary is Lam's explicit operational choice for the replication.

**Important:** `--patience 0` in the current Stage 3 code disables early stopping
but still restores the best-validation checkpoint. That flag alone does not
implement the selected endpoint (`stage3_classifier.py:187-194`). Do not run
the old sweep cell as though it did.

## Precision and candidate run count

Existing seeds 1-5 are an already-inspected exploratory pilot. Their paired
overall-accuracy differences are 2.94, -4.30, 0.10, 1.18 and -1.66 points, with
SD 2.7697508913. That SD describes the earlier early-stopped/best-head protocol.

Conditional two-sided paired-t planning, alpha 0.05 and power 0.80, gives:

| Assumed SD of paired differences | True difference used for power | Candidate new seed pairs |
|---|---:|---:|
| Pilot SD, 2.77 points | 1 point | 63 |
| 1.5 × pilot SD, 4.15 points | 1 point | 138 |
| 2 × pilot SD, 5.54 points | 1 point | 243 |

These are sensitivity scenarios, not confidence bounds or verified power for
250-epoch final heads. Only four degrees of freedom estimate the pilot SD.
**63 is the provisional power-based target for a 1-point difference at an SD of
about 2.8 points, not an arbitrary 20-seed endpoint.** Lam's 27 Oct decision is to
run 20 fixed-250/final-head seed pairs first if that SD is uncertain, re-estimate
the paired-difference SD, then top up to the power-based count. Record the
variance-only re-estimation rule and feasible budget before execution.
No significance-driven extension or stopping is allowed by this plan.

Power here concerns rejecting a zero-difference null when the true mean difference
is 1 point. It does not provide 80% power to prove that the gain exceeds 1 point.
Report the paired mean and its 95% interval relative to both zero and the 1-point
practical target; do not turn a nonsignificant result into an equivalence claim.

Calculation: `.agents/reports/c1-plan/plan_c1.py`, using the
[SciPy noncentral-t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html).
The script, software version and full sensitivity rows are saved locally in
`design-options.json`. No simulation or classifier training is involved.

## Candidate allocation and analysis

For the provisional 63-pair scenario, candidate seeds are 101-163 inclusive;
seeds 1-5 and 42 remain historical. Each candidate seed runs all three variants
(189 primary-protocol classifier runs), plus es20 for the first 20 seeds
(60 sensitivity runs). Randomize seed-block order and variant order within each
block using a recorded allocation seed, 24102026. Save the resulting schedule
before execution. The candidate schedule is local at
`.agents/reports/c1-plan/candidate-schedule.csv`; it is not a run authorization.

Use one primary GAN-versus-basic contrast. Summarize the per-seed paired
differences and a two-sided 95% paired-t interval under the declared model.
Inspect paired differences without selecting a new primary endpoint/test to chase
significance. Report per-seed values and any failures; complete the fixed schedule
or document an interrupted/incomplete experiment. Best-head, real-only,
rare-only and class-specific findings are explicitly secondary/exploratory.

The seed pair is the unit for training-randomness inference. All runs share the
same 1,022 test images, including only 22 rare images, and the same GAN realization.
More seeds do not create new patients or establish new-cohort/clinical validity.
Test-set uncertainty and training-seed uncertainty must remain separate.

## Required output protection before any execution

The current notebook Part 4 (code cell 20) writes canonical variant folders and
`results/seed_check.csv`; its restoration runs only after the loop succeeds.
It must not be executed unchanged for this experiment.

The execution implementation should:

1. Write to a new experiment directory, with a separate seed/variant subdirectory.
   Never use canonical `results/stage3_*` or overwrite the run-of-record CSVs.
2. Fail before training when a required cache is missing, paths differ, or input
   fingerprints change. This experiment may read caches but must not regenerate them.
3. Retain final and best heads, full probabilities, histories, data counts,
   command/configuration, run status, code revision and package versions.
4. Record hashes for manifest, caches, accepted GAN inventory and backbone weights.
   Compare protected canonical hashes after success, error and interruption.
5. Publish a complete experiment summary separately, identifying missing/failed
   runs. Resume only matching unfinished run IDs without replacing completed outputs.

Those are requirements, not existing CLI flags. No new flags or runnable training
command are claimed here. Before a later implementation, review the concrete diff
and test missing-cache/error/interruption paths without fitting a classifier.

## Decisions still needed before training

- Design decision settled (Lam, 2026-10-27): fixed 250 epochs/final head is
  primary; `es20` on the first 20 seeds is sensitivity only. Plan for about 63
  pairs for a 1-point difference if paired SD is about 2.8 points. If SD is
  uncertain, run 20 seeds first to re-estimate it, then top up by the power calculation.
- Still needed: specify the variance-only recalculation rule, calibration seed
  inclusion and execution budget before training; do not top up based on significance.
- Review and authorize the isolated-output/checkpoint implementation, then decide
  when to push and execute it on Colab. Neither is required for this training-free plan.

Source inputs: `results/seed_check.csv`, `stage3_classifier.py`, notebook cell 20,
[L7](L7_BASELINE_AUDIT.md), [L8](L8_RCI_AUDIT.md), and the original methods XML
archived by L7. Workflow assistance used the experimental-design and
statistical-analysis skills; numerical assumptions remain explicit.
