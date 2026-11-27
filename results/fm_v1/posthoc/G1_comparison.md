# G1 pad-to-square check: locked against padded (post hoc)

Post hoc sensitivity check (protocols.md section 8, G1), outside amendment A2 A-11. **It never replaces a locked number.** MIRAGE and Inception-v3 only; every image padded with black to a square around its centre, then each model's own locked transform; scored with the locked episodes and nine-way roster.

Label of every row: post hoc, rule timestamped by commit 1eb3964 (16 Nov 2026), computed 2026-11-18; sensitivity check outside A2 A-11, never replaces a locked result.

Sources: `G1_pad_to_square.csv`, `P9_rare_vs_normal_auroc.csv`, `P9_rare_vs_normal_differences.csv`, `P9_G1_padded_auroc.csv`, `P9_G1_padded_differences.csv` (results/fm_v1/posthoc/). "Locked" P9 rows come from the locked nine-way scores; "padded" rows from G1's features.

## 1. Checks (read first)

If the control or the square images differ from the locked features by more than 1e-4 (the A2 A-6 repeat tolerance), the Colab environment changed and the padded numbers mix padding with that change.

| Arm | Check | Value | Value |
|---|---|---|---|
| MIRAGE | control: unpadded re-extraction against locked features | max_difference_ratio 1.38e-07 | min_cosine 1.000000 |
| MIRAGE | check: 1032 square images (padding changes no pixel) against locked features | max_difference_ratio 1.95e-07 | min_cosine 1.000000 |
| MIRAGE | check: repeat on Colab (5 padded images encoded twice) | max_difference_ratio 0.00e+00 | min_cosine 1.000000 |
| MIRAGE | check: Colab run against the input lock | code_matches_lock True | environment_matches_lock False |
| Inception | control: unpadded re-extraction against locked features | max_difference_ratio 1.43e-06 | min_cosine 1.000000 |
| Inception | check: 1032 square images (padding changes no pixel) against locked features | max_difference_ratio 1.07e-06 | min_cosine 1.000000 |
| Inception | check: repeat on Colab (5 padded images encoded twice) | max_difference_ratio 0.00e+00 | min_cosine 1.000000 |
| Inception | check: Colab run against the input lock | code_matches_lock True | environment_matches_lock True |

## 2. Development: macro recall over the five rare classes (124 queries, folds pooled; mean of 100 draws)

| K | MIRAGE locked | MIRAGE padded | change | Inception locked | Inception padded | change | Size only |
|---|---|---|---|---|---|---|---|
| 1 | 0.495 | 0.497 | +0.002 | 0.409 | 0.399 | -0.010 | 0.578 |
| 2 | 0.615 | 0.611 | -0.004 | 0.499 | 0.484 | -0.015 | 0.550 |
| 4 | 0.712 | 0.703 | -0.010 | 0.581 | 0.574 | -0.007 | 0.560 |
| 8 | 0.777 | 0.774 | -0.003 | 0.647 | 0.642 | -0.005 | 0.589 |
| mean_k1_2_4_8 | 0.650 | 0.646 | -0.004 | 0.534 | 0.525 | -0.009 | 0.569 |

## 3. Hold-out (legacy 22)

Macro recall (mean of 100 draws):

| K | MIRAGE locked | MIRAGE padded | change | Inception locked | Inception padded | change | Size only |
|---|---|---|---|---|---|---|---|
| 1 | 0.536 | 0.518 | -0.018 | 0.430 | 0.427 | -0.003 | 0.645 |
| 2 | 0.685 | 0.662 | -0.023 | 0.497 | 0.500 | +0.003 | 0.590 |
| 4 | 0.799 | 0.788 | -0.011 | 0.556 | 0.559 | +0.003 | 0.592 |
| 8 | 0.867 | 0.861 | -0.005 | 0.614 | 0.618 | +0.004 | 0.645 |
| 16 | 0.894 | 0.930 | +0.036 | 0.621 | 0.636 | +0.016 | 0.726 |
| mean_k1_2_4_8 | 0.722 | 0.707 | -0.014 | 0.524 | 0.526 | +0.002 | 0.618 |

Images named correctly, of 22 (mean of 100 draws):

| K | MIRAGE locked | MIRAGE padded | change | Inception locked | Inception padded | change |
|---|---|---|---|---|---|---|
| 1 | 11.88 | 11.40 | -0.48 | 9.69 | 9.58 | -0.11 |
| 2 | 15.09 | 14.54 | -0.55 | 11.21 | 11.29 | +0.08 |
| 4 | 17.66 | 17.35 | -0.31 | 12.67 | 12.71 | +0.04 |
| 8 | 19.23 | 19.01 | -0.22 | 14.00 | 14.05 | +0.05 |
| 16 | 19.87 | 20.54 | +0.67 | 14.22 | 14.46 | +0.24 |

## 4. Nine-way audit: where each set goes (locked → padded)

Same fitting roster as the locked run; in the padded column both prototypes and queries come from padded features. Counts of images, locked → padded.

| Set | Arm | Correct class | NORMAL | CNV, DME or DRUSEN | Any rare |
|---|---|---|---|---|---|
| Kermany patient-disjoint (n=169) | MIRAGE | 169 → 166 | 169 → 166 | 0 → 1 | 0 → 2 |
| Kermany patient-disjoint (n=169) | Inception | 150 → 157 | 150 → 157 | 15 → 8 | 4 → 4 |
| Legacy 22 (n=22) | MIRAGE | 20 → 21 | 0 → 0 | 0 → 0 | 22 → 22 |
| Legacy 22 (n=22) | Inception | 13 → 14 | 0 → 0 | 2 → 2 | 20 → 20 |
| OCTID NORMAL (n=206) | MIRAGE | 202 → 0 | 202 → 0 | 0 → 0 | 4 → 206 |
| OCTID NORMAL (n=206) | Inception | 182 → 8 | 182 → 8 | 9 → 0 | 15 → 198 |
| OCTID CSC (n=102) | MIRAGE | 23 → 19 | 8 → 0 | 53 → 4 | 41 → 98 |
| OCTID CSC (n=102) | Inception | 22 → 22 | 3 → 1 | 53 → 18 | 46 → 83 |
| OCTID MH (n=102) | MIRAGE | 10 → 54 | 8 → 0 | 83 → 21 | 11 → 81 |
| OCTID MH (n=102) | Inception | 27 → 19 | 6 → 2 | 52 → 12 | 44 → 88 |
| OCTDL NORMAL (n=332) | MIRAGE | 26 → 1 | 26 → 1 | 276 → 25 | 30 → 306 |
| OCTDL NORMAL (n=332) | Inception | 66 → 13 | 66 → 13 | 255 → 17 | 11 → 302 |
| OCTDL MH (n=39) | MIRAGE | 3 → 2 | 0 → 0 | 36 → 6 | 3 → 33 |
| OCTDL MH (n=39) | Inception | 0 → 15 | 2 → 0 | 36 → 1 | 1 → 38 |

## 5. P9: rare against normal AUROC, no cut-off (95% unit bootstrap, 2,000 resamples)

| Set | MIRAGE locked | MIRAGE padded | Inception locked | Inception padded |
|---|---|---|---|---|
| OCTID CSC against OCTID NORMAL | 0.878 (0.832 to 0.919) | 0.305 (0.241 to 0.375) | 0.805 (0.749 to 0.854) | 0.501 (0.424 to 0.575) |
| OCTID MH against OCTID NORMAL | 0.684 (0.618 to 0.748) | 0.090 (0.059 to 0.131) | 0.784 (0.724 to 0.838) | 0.525 (0.450 to 0.602) |
| OCTID CSC+MH against OCTID NORMAL | 0.781 (0.733 to 0.824) | 0.198 (0.154 to 0.240) | 0.795 (0.749 to 0.835) | 0.513 (0.454 to 0.570) |
| OCTDL MH against OCTDL NORMAL | 0.357 (0.261 to 0.468) | 0.320 (0.241 to 0.406) | 0.634 (0.543 to 0.717) | 0.656 (0.550 to 0.746) |
| Legacy 22 against 169 patient-disjoint Kermany normals | 1.000 (1.000 to 1.000) | 1.000 (0.998 to 1.000) | 0.989 (0.967 to 1.000) | 0.989 (0.973 to 0.999) |

Rare scans called any rare, and normals called rare, at the argmax of each column's own features (locked → padded):

| Set | MIRAGE rare called rare | MIRAGE normals called rare | Inception rare called rare | Inception normals called rare |
|---|---|---|---|---|
| OCTID CSC against OCTID NORMAL | 0.402 → 0.961 | 0.019 → 1.000 | 0.451 → 0.814 | 0.073 → 0.961 |
| OCTID MH against OCTID NORMAL | 0.108 → 0.794 | 0.019 → 1.000 | 0.431 → 0.863 | 0.073 → 0.961 |
| OCTID CSC+MH against OCTID NORMAL | 0.255 → 0.877 | 0.019 → 1.000 | 0.441 → 0.838 | 0.073 → 0.961 |
| OCTDL MH against OCTDL NORMAL | 0.077 → 0.846 | 0.090 → 0.922 | 0.026 → 0.974 | 0.033 → 0.910 |
| Legacy 22 against 169 patient-disjoint Kermany normals | 1.000 → 1.000 | 0.000 → 0.012 | 0.909 → 0.909 | 0.024 → 0.024 |

MIRAGE minus Inception, AUROC difference paired on the same resamples:

| Set | Locked | Padded |
|---|---|---|
| OCTID CSC against OCTID NORMAL | 0.072 (0.014 to 0.131) | -0.196 (-0.282 to -0.108) |
| OCTID MH against OCTID NORMAL | -0.100 (-0.171 to -0.032) | -0.435 (-0.511 to -0.354) |
| OCTID CSC+MH against OCTID NORMAL | -0.014 (-0.068 to 0.039) | -0.316 (-0.377 to -0.257) |
| OCTDL MH against OCTDL NORMAL | -0.277 (-0.379 to -0.158) | -0.335 (-0.461 to -0.194) |
| Legacy 22 against 169 patient-disjoint Kermany normals | 0.011 (0.000 to 0.033) | 0.011 (0.001 to 0.026) |
