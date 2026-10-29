# Research context

What every agent should know about this research. Keep it short (one or two
screens). Agents read this before any research or analysis task.

## Question
Does adding CycleGAN-generated images of rare retinal diseases to the training set
improve a few-shot 9-class OCT classifier, compared with real images only and with
basic augmentation? (Reproduction of Yoo et al. 2021.)

## Hypotheses
- H1 (paper): `gan` > `basic` > `none` on the independent test set
  (paper accuracy 92.1 / 91.3 / 88.4).
- Current evidence: seed 42 gives gan 92.95, none 90.31, basic 88.85; over seeds 1-5
  the means are gan ~92.6, basic ~92.9, none ~90.7 (`results/seed_check.csv`).
  The augmented variants have higher descriptive mean accuracy than none;
  these five seeds do not establish GAN superiority over basic or equivalence.
  L7 confirms protocol/data-count differences from the paper without a unique
  cause for the strong real-only baseline ([audit](L7_BASELINE_AUDIT.md)).
  L8 independently verifies our RCI calculation; the paper's implementation
  remains unresolved ([audit](L8_RCI_AUDIT.md)).

## Scope
In: the paper's pipeline (Stages 0-4), its metrics (Tables 3-4), knn/prototype
baselines on frozen features, CAM. Out: the paper's Siamese network, a trained
prototypical network, end-to-end fine-tuning of Inception-v3.

## Glossary
- **Common classes:** NORMAL, CNV, DME, DRUSEN (Kermany OCT2017).
- **Rare classes:** CSC, MH, MacTel, Stargardt, RP; 5/5/4/4/4 test images each.
- **Variant:** `none` / `basic` / `gan` rare-class training pool in Stage 3.
- **Accepted images:** CycleGAN outputs that passed `stage2_cyclegan.py qc`
  (Inception-v3 kNN distance closer to real disease than to NORMAL).
- **Head:** the single trained softmax layer (`results/stage3_<variant>/head.npz`).

## Units & conventions
- Accuracy in %, kappa/MCC/RCI/AUC as fractions. Seed-42 none/basic/gan 95% CIs:
  5,000 patient-cluster percentile draws (RNG seed 0), 633 Kermany patients + 22 rare
  image singletons (patient IDs unknown; shared rare patients could leave CIs too narrow).
  `analysis.py cluster` writes `results/analysis/`; Stage 4 retains legacy image CIs.
  Gan-basic accuracy: +4.11 points, CI 2.47–5.76; single-run conclusion unchanged (L9).
- Images resized to 256x256 everywhere, then 299x299 for Inception-v3.
- All seeds 42 unless a seed sweep says otherwise.

## Data
| Dataset | Source / version | Units | Resolution | Known issues |
|---|---|---|---|---|
| Kermany OCT2017 | Mendeley rscbjbr9sj v2, doi:10.17632/rscbjbr9sj.2 | JPEG B-scans | varies -> 256 | 633 test patients; 566 overlap official train (our train + val), 506 overlap fitting split. Clustering cannot fix leakage; splits fixed (L9) |
| Rare-disease OCT | Mendeley btv6yrdbmv (released with the paper) | PNG | varies -> 256 | macOS junk (`__MACOSX`, `._*`); folders shipped as `Mactel`, `Stargart` |
| Derived (aug_basic, gan, ckpt, features) | Google Drive `MyDrive/octfsl` | | 256 | see `COLAB.md` |

## Key sources
| Citation | DOI / link | What it actually says | How we use it |
|---|---|---|---|
| Yoo, Choi, Kim 2021, MBEC 59:401 | 10.1007/s11517-021-02321-1 | CycleGAN augmentation for few-shot rare OCT classes | method and target numbers (`stage4_evaluate.py` PAPER) |
| Kermany et al. 2018, Cell 172:1122 | 10.1016/j.cell.2018.02.010 | OCT2017 dataset and split | common classes, official test split |
| TensorFlow CycleGAN tutorial | tensorflow.org/tutorials/generative/cyclegan | U-Net G, PatchGAN D, lambda 10 | Stage 2 architecture |

## Decisions
| Date | Decision | Why | Evidence |
|---|---|---|---|
| (pre-2026-10) | Cache frozen Inception-v3 features, train only the softmax head | Same model as end-to-end with a frozen backbone; minutes not ~150 h | `stage3_classifier.py` docstring |
| (pre-2026-10) | Rare-class val images moved back to train | Every real rare image feeds Stage 1/2; rare val comes from augmented data, as in the paper | `stage0_manifest.py:75` |
| (pre-2026-10) | Feature-distance QC replaces the paper's ophthalmologist review | No clinician available | `stage2_cyclegan.py` `qc` docstring |
| 2026-10-22 | Keep the accepted GAN set from the original QC rule (sort by d_rare) | It is what the gan result was trained on; `--margin` would change CSC's kept set | QC csvs == accepted folders == `gan_*.npz` |
| 2026-10-22 | Commit `head.npz` and `test_predictions.csv` for each variant | Small (~76 / ~180 KB); `COLAB.md` and `analysis.py` need them on GitHub | `.gitignore` |
| 2026-10-22 | Local work on the laptop is training-free; training runs on Colab | No dedicated GPU locally | `docs/research/REPLICATION.md` |
| 2026-10-27 | D15: fixed 250 epochs/final head primary; es20 on first 20 seeds sensitivity; about 63 pairs for 1 pp at SD about 2.8 pp, or 20 first to re-estimate SD then top up | Lam selected protocol and power-based planning; T8 must align the excluded sweep implementation | [C1 plan](C1_EXPERIMENT_PLAN.md) |
| 2026-10-27 | D7/C5: C1 keeps v1; run C5 before C2, topping up MH/MacTel/Stargardt/RP using existing checkpoints and original automatic QC only | L4 expert ratings were dropped; C2 needs full 3,000-image pools. Completed on Colab; all five v2 classes now have 3,000 images (v1 preserved) | [Colab 3.4c](../../COLAB.md#c5-top-up-before-c2-keep-c1s-pool-fixed-d7), `c5_topup.py`; explicit Stage 3 `--gan-pool v2`, separate caches and outputs |
| 2026-10-29 | Record first Colab outcomes: C5 done; C0 gate stopped, no heads. D20=A (29 Oct 2026): stop C0 at the gate as a limitation; C1 and later C2/C3 use current inputs | Normalizing format and framing did not reduce strong rare-versus-common separation (AUC 0.995 to 0.991). The separation does not come from the tested file-format/framing differences; disease and dataset source remain confounded and this cannot tell them apart | b109d0d notebook cells 19-24, 29-32; [C5 counts](../../results/stage2/c5_v2_counts.csv), [C0 gate](../../results/c0/v1_gate_summary.csv) |
| 2026-10-29 | D17=B: accelerate cached Stage 3 heads without changing the maths; C1 selects fast; C0 trainer support retained but heads stopped under D20=A; historical Stage 3 default stays keras | Compiled epoch calls the same Keras steps; identical-order full-epoch gate passes with zero weight/val-loss difference on laptop | [T12 gate and benchmark](T12_FAST_TRAINER.md) |
| 2026-10-29 | C2/C3 ready for manual Colab: current inputs, v2, fixed250 FINAL, seeds1-20 (D18=A). Lam chose C3 A: common fitting caps10000/5000, nested per class/seed, common val/test fixed | Fig11 counts visually checked; fitting-only adaptation, cap5000 vs4500 rare fitting is near-balanced. C2 N=0 is basic; original non-nested-by-design sampling and pooled rare90/10 preserved | [Design and source/decision provenance](C2_C3_DESIGN.md); notebook Part4b; `c23_experiment.py` |

## Known pitfalls
- Rare test classes have 4-5 images: one image moves recall by 0.20-0.25.
- Seed variance is large (DRUSEN recall 0.62-0.94 across seeds); compare variants over seeds.
- Rare images come from another source than Kermany (RGBA PNG crops vs grayscale JPEG; raw size
  alone names the rare test class 16/22) and are almost linearly separable from major images in
  frozen features. Rare recall may partly measure the source (`REPLICATION.md` L7, R5).
- Original v1 accepted GAN images are fewer than 3,000 for MH, MacTel, Stargardt, RP (2,610 / 1,248 /
  2,199 / 832); Stage 3 warns but continues. C5 completed a separate 3,000/class v2 pool for C2/C3; C1 keeps v1.

## C1 execution (T8, 2026-10-27)

Notebook 4.1-4.5 now implements D15: fixed250 final heads primary; es20 on
C0-matched seeds 1-20 sensitivity. First 20 complete before a one-time SD-only
power recalculation (1 pp, alpha .05, power .80, floor 63/cap 243); top up fixed250
seeds 21..N. D20=A (29 Oct 2026) fixes INPUT_ARM=current; C0 stops at its gate as a limitation. Original accepted GAN
pool only; C5 is for C2/C3. Drive c1/v1/<input_arm> outputs preserve canonical results.
D17=B selects the fast trainer after an identical-order gate with zero weight,
optimizer-state and val_loss differences. Three native-shuffle 25-epoch pairs
show 4.69x compute speed-up but lower fast test accuracies; no behavioural
equivalence claim. Conservative laptop projection: 12.87 min/250 epochs,
C1 first stage 13.9-15.4 h / 63 pairs 41.6-43.1 h (es20 20-50 epochs assumed).
Colab/Drive is unbenchmarked. [Gate and limitations](T12_FAST_TRAINER.md).
