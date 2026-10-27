# Replication tracker: Yoo et al. 2021 (MBEC 59:401-415)

What the paper reports, what this repo reproduces, and the prioritized work left.
Status as of 2026-10-22. Update the tables when an item changes state.

Sources: the paper PDF (main checkout, not in git), `results/stage4/*.csv`,
`results/seed_check.csv`, `results/stage2/qc_*.csv`, the 22 Oct status report.

## Paper vs repo

| Paper item | What the paper did | Repo state |
|---|---|---|
| Basic augmentation (§2.3) | flips, ±5% shift, ±30° rotation, 0-20% zoom, ±10% brightness, elastic (MATLAB) | Done: Stage 1, 2,000/class (albumentations) |
| CycleGAN per disease (§2.3, §3.1) | TF tutorial, 100 epochs, ophthalmologists remove artifacts, 3,000 kept/class | Done, 100 epochs. Automatic kNN QC instead of clinicians; kept 3,000 / 2,610 / 1,248 / 2,199 / 832 (CSC/MH/MacTel/Stargardt/RP) |
| Fig. 5: amount of augmentation | best at 2,000 basic + 3,000 GAN | Not done |
| Fig. 6: acceptance rate | 100 samples/class rated by 2 ophthalmologists; Stargardt and RP rejected most | Automatic only: QC pass rate 84 / 65 / 31 / 55 / 21% (RP and MacTel lowest). No clinician |
| Fig. 7: t-SNE | Inception-v3 features, none vs gan | Coordinates done (`analysis.py tsne`); figure needs matplotlib |
| Table 3: independent test | none 88.4, basic 91.3, gan 92.1 % | Done, seed 42: 90.31 / 88.85 / 92.95 %. Seeds 1-5: 90.7 / 92.9 / 92.6 (gan ties basic) |
| Table 4: per-class TPR | none gets rare TPR 0.00-0.25, gan 0.75-1.00 | Done. **Our `none` already gets 0.75-0.80 on rare classes**, so the paper's rare-class rescue is not reproduced at the baseline |
| Fig. 9: one-vs-rest ROC | per-class curves | AUC table done; curve points done (`analysis.py roc`), figure needs matplotlib |
| Fig. 10: Grad-CAM | saliency maps | 40 panels for gan (`analysis.py cam`), not yet reviewed |
| Table 2: 5-fold CV on whole data | GANs retrained per fold, gan 93.9 ± 4.5 % | Not done |
| Fig. 11: imbalance by under-sampling | "no significant impact" | Not done |
| kNN K=3 baseline | 91.2 % | Done: 86.40 % |
| Prototypical network | trained, 80.8 % | Class-mean baseline only: 68.98 % (out of scope) |
| Siamese network | 80.9 % | Out of scope |
| Human expert | 97.5 % | Not reproducible |
| Fig. 12: segmentation CycleGAN | 72 clinician-labelled masks, Dice 0.784 | Not reproducible (no masks) |
| Supplement: MobileNet-v2, ResNet | "similar" to Inception-v3 | Not done |
| Training schedule | 250 epochs, batch 10, Adam | Batch 10, Adam, but early stopping (patience 20) |
| RCI | 0.916-0.983 | 0.76-0.81 with the cited definition I(true; pred)/H(true). The paper's values do not fit that definition (gan 0.983 is above the 0.970 maximum its own Table 4 allows); no tested alternative reproduces them (L8) |

## Prioritized work

The laptop has no dedicated GPU, so local work is **training-free**: it only reads artifacts that
already exist (QC records, heads, test predictions, the features cache). Anything that trains a
model is on the Colab track. Tool: `analysis.py` (CPU, about 30 s for `all`); outputs in
`results/analysis/`. To run from a worktree without data:
`D:/Few-shot/.venv/Scripts/python.exe analysis.py --root D:/Few-shot --out results/analysis all`.

### Laptop track (training-free), in priority order

| # | Work | Paper | Status | Tool / output |
|---|---|---|---|---|
| L1 | Get the October work onto GitHub (branch + PR), rewrite README results | | Done in the PR from branch `t3/28d8122c` | git |
| L2 | Paired tests none/basic/gan: test set (seed 42) and training seeds | Table 3 | Done | `compare` -> `paired_seed42.csv`, `paired_seeds.csv` |
| L3 | CAM heatmaps, all 22 rare test images, for gan and none | Fig. 10 | 40 gan panels done | `cam --variant gan|none --n 40` |
| L4 | Look at GAN images: rate 100 random accepted per class (Fig. 6 protocol) | Fig. 6 | Automatic rates done; human look not done | `qc` -> `qc_summary.csv`; `results/stage2/qc_*_kept.jpg` |
| L5 | t-SNE and ROC figures | Figs. 7, 9 | Coordinates done; PNGs need matplotlib | `tsne`, `roc` |
| L6 | kNN gap (ours 86.4, paper 91.2) | Table 3 | Sweep done | `knn` -> `knn_sweep.csv` |
| L7 | Why our `none` rare recall (0.75-0.80) is far above the paper's (0.00-0.25) | Table 4 | Not started | inspect `results/stage3_none/test_predictions.csv` |
| L8 | RCI gap (ours 0.76-0.81, paper 0.92-0.98) | Table 3 | Done: our RCI follows the cited definition; the paper's values don't fit it. Open: the paper's actual formula (ask the authors) | `rci` -> `rci_definitions.csv`, `rci_paper_bounds.csv` |

### Findings so far (2026-10-22, `results/analysis/`)
- **Seed 42 test set:** gan beats basic by 4.1 points (CI 2.6-5.8, McNemar p < 0.0001) and none
  by 2.6 (p = 0.001). This only covers test-set sampling, not training randomness.
- **Over 5 training seeds:** gan - basic = -0.35 ± 2.77 points, gan ahead on 3 of 5 seeds. The
  seed-42 gap is within training noise, so the seed sweep (C1) decides the claim, not more tests
  on one run.
- **Rare classes (22 images):** no comparison is separable (gan vs basic differs on 1 image).
- **QC pass rate:** CSC 84%, MH 65%, Stargardt 55%, MacTel 31%, RP 21%. The paper's clinicians
  rejected Stargardt and RP most; our automatic filter rejects RP and MacTel most.
- **kNN:** Euclidean K=3 reproduces Stage 4 (86.40%). Cosine K=1 gives 91.10%, close to the
  paper's 91.2%, but the paper states K=3 and Euclidean, so this is a hint, not an explanation.
  K>1 drops rare recall sharply (19-27 rare vs ~74,000 major neighbours).
- **RCI (L8, 2026-10-23):** our `stage4_evaluate.rci()` follows the definition the paper cites, and
  the paper's values cannot come from that definition. Sources are in `references.bib`.
  - *What is cited.* The paper cites refs 38-39 for kappa/RCI/MCC. Ref 39 (Gorodkin 2004) is the
    multiclass MCC. Ref 38 (Yoo 2019) is not open access; its reference list includes Sindhwani 2001
    and Statnikov 2005. Statnikov §2.8 describes RCI as how much a classifier reduces "the uncertainty of
    a decision problem ... relative to classifying using only the priors", i.e. I(true; pred) / H(true).
    That is our formula. The same group's 2017 paper (Choi et al., Eq. 6) prints the same thing
    without the division (I in nats).
  - *Alternatives on our confusion matrices.* We tried 10 definitions: ours, I in nats, I/H(pred),
    three symmetric normalisations, I/log K, balanced class prior, H(pred|true) in place of
    H(true|pred), and macro one-vs-rest. None comes within 0.19 of the paper on average
    (`rci_definitions.csv`). None gets near the paper's proto (0.933 at 80.8%); our proto is
    0.29-0.64 at 69%.
  - *The paper's own Table 4.* With the paper's per-class recalls and test counts, I/H(true) can only
    lie in 0.64-0.97 for any confusion matrix (exact range; the maximum is found by enumerating
    vertices, since I is convex in p(pred|true)).
    - gan: the paper reports 0.983, above the maximum of 0.970. Even allowing the one extra correct
      image that Table 3's 92.1% implies, the maximum is 0.971. So under the cited definition the
      paper's gan RCI is impossible.
    - none / basic / knn: these sit near the maximum (knn 0.972 vs 0.974). Reaching it needs every
      class's errors to fall in a single other class.
    - With our error pattern, the paper's recalls give 0.70-0.78, our range
      (`rci_paper_bounds.csv`).
  - *Bottom line.* The gap is not our bug. What the paper computed is unverified; its code is not
    public. Next step: ask the authors for the RCI code or the confusion matrices. Keep RCI out of
    claims, as the README already does.
  - *Side finding.* Table 4's proto row implies 86.1% accuracy, but Table 3 says 80.8%.
    Table 4's gan row implies 940/1022 correct (92.0%), but Table 3 says 92.1%.


### Disagreement ? L8 Claude analysis and Codex audit (D16, 2026-10-27)

The Claude branch is the primary analysis and code above: its reference-chain check
and Table 4 bounds imply that the paper's GAN RCI exceeds the maximum under
`I(true; pred)/H(true)`. The independent [Codex audit](L8_RCI_AUDIT.md) verifies our
five saved RCI values within 1e-12, but did not inspect the cited 2019 full text or
verify the primary 2001 equation directly. It therefore leaves the paper's actual
normalization, aggregation and implementation unresolved and does not certify a
paper error. These conclusions differ in the strength of the paper-inconsistency
claim; both are retained, without selecting one as a new scientific finding.

Codex adds an independent conditional-entropy computation, axis-reversal results,
log-base/count-scaling invariance, and perfect/independent/constant/permuted-label/
empty-class fixtures. Axis reversal does not reproduce the paper's targets; RCI
can equal 1 for a bijective label permutation, so accuracy and label alignment are
still needed. The one-class zero-entropy case was not certified. A blocked sklearn
DLL required an AST-only function audit, rather than a full Stage 4 evaluation.
Published Stage 4 values remain unchanged. See the audit for its source boundaries.

### Colab track (needs training)
| # | Work | Paper | Cost |
|---|---|---|---|
| C1 | Seed sweep (~20 seeds) none/basic/gan, plus fixed 250 epochs without early stopping | Table 3 | head only, minutes per run (Stage 3 already ran on Colab CPU on 4 Oct) |
| C2 | Amount of GAN images 0 / 500 / 1,000 / 2,000 / 3,000 (`--gan-per-class`), over seeds | Fig. 5 | as C1 |
| C3 | Under-sampling to control class balance | Fig. 11 | as C1, needs a Stage 3 option |
| C4 | 5-fold CV with per-fold Stage 1 + CycleGANs (25 GANs, ~50 A100-hours by `stage2.log`); without per-fold GANs the test fold leaks | Table 2 | GPU, large |
| C5 | Top up MacTel and RP toward 3,000 GAN images | §2.3 | GPU |

### Not reproducible / out of scope
Ophthalmologist review and human-expert baseline (no clinician), segmentation CycleGAN (no
labelled masks), Siamese and trained prototypical networks (scope decision in `CONTEXT.md`).

### Housekeeping (from the 22 Oct report)
Move `results_latest/` off the second Google account that wrote it, verify the raw upload with Stage 0 on Colab,
delete about 5 GB of Drive duplicates.
