# Few-shot classification of rare retinal diseases in OCT

In the public benchmark of Yoo et al. 2021, five rare retinal diseases (CSC, MH, MacTel,
Stargardt, RP) have only 19–27 labelled OCT images each. The four common Kermany OCT2017 classes
(CNV, DME, DRUSEN, NORMAL) have tens of thousands. This repository holds two studies on that
benchmark:

1. **Replication of Yoo et al. 2021** (closed 6 Nov 2026). A CycleGAN per rare disease adds
   synthetic images for an Inception-v3 classifier.
2. **Frozen foundation models, few-shot** (Phase 1 complete 27 Nov 2026). Four frozen encoders
   classify the five rare diseases from K = 1–8 examples each, with cosine prototypes: MIRAGE,
   DINOv2, RETFound and ImageNet Inception-v3. The protocol and inputs were hashed before any
   score.

## Results in brief

**Frozen-model study.** The paper is *Image Size Alone Beats Frozen Foundation Models at One
Shot*, in [`docs/report/fm_v1_paper/`](docs/report/fm_v1_paper/).
- A rule that sees only image width and height scores 0.578 at K = 1. That is above all four
  models; the best is MIRAGE at 0.495.
- MIRAGE has the highest development macro recall averaged over K (0.650; Inception-v3 0.534).
  It was the pre-registered nominee.
- The results below are post hoc, each under a rule committed before its numbers:
  - Under a size and framing control, MIRAGE keeps a positive lead over Inception-v3, but the
    lead shrinks from 0.116 to 0.063.
  - How many outside normal scans are called rare depends on the input recipe.
  - A few-shot test tells normal scans from three sources apart, above chance.
  - 4 of the 22 hold-out images have a copy among the development images.
- The paper therefore reports rankings beside size, framing and source controls.

**Replication.** Details are in [`docs/research/REPLICATION.md`](docs/research/REPLICATION.md).
- At seed 42 the CycleGAN variant reaches 92.95% accuracy; the paper reports 92.1%. Real images
  only reach 90.31%, and basic augmentation 88.85%.
- Over five training seeds, CycleGAN (92.59 ± 1.29%) and basic augmentation (92.93 ± 1.85%)
  are not separated; CycleGAN is ahead on 3 of 5 seeds. Both get 17–19 of the 22 rare test images right, against 11–17 with
  real images only.
- Caveats:
  - The Kermany test split shares patients with training: 566 of its 633 patients.
  - No clinician reviewed the generated images.

## Where to start

| You want | Read |
|---|---|
| The frozen-model protocol, its decisions and change log | [`protocols.md`](protocols.md) |
| The v1 paper and how to build it | [`docs/report/fm_v1_paper/README.md`](docs/report/fm_v1_paper/README.md) |
| To run the frozen-model notebooks | [`FM_FIRST_RUN_GUIDE.md`](docs/research/FM_FIRST_RUN_GUIDE.md), [`FM_RUNBOOK.md`](docs/research/FM_RUNBOOK.md) |
| The replication, item by item | [`docs/research/REPLICATION.md`](docs/research/REPLICATION.md) |
| To run the replication on Colab | [`COLAB.md`](COLAB.md), [`guide.md`](guide.md) |
| Data download and cleanup, local setup, the stage-by-stage run guide, full result tables | the [previous README](https://github.com/Malnoad/Few-shot-rare-diseases/blob/6116f93d7788238b8a306be428e04af1d301d5c8/README.md), unchanged at commit `6116f93` (`git show 6116f93:README.md`) |
| Project context and glossary | [`docs/research/CONTEXT.md`](docs/research/CONTEXT.md) |

## Setup

- **Python environment:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate        # Windows; Linux/macOS: source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Requirements:** `requirements.txt` covers the replication (TensorFlow). `requirements-fm.txt`
  is the pinned PyTorch environment for the frozen-model notebooks on Colab.
- **GPU steps** run on Colab.
- **Data is not redistributed:**
  - Kermany OCT2017: [Mendeley rscbjbr9sj v2](https://data.mendeley.com/datasets/rscbjbr9sj/2).
  - The rare-disease set: [Mendeley btv6yrdbmv](https://data.mendeley.com/datasets/btv6yrdbmv).
  - Both go under `data/raw/`, cleaned as the previous README describes. Large files live in
    Google Drive (`MyDrive/octfsl`).

## Rules that never move

1. `data/raw/` is read-only; every transformation is a script.
2. `data/manifest.csv` (committed) defines every split. Never run `stage0_manifest.py --force`:
   it re-samples the rare hold-out.
3. Files marked `[lock]` below are never edited.
4. The frozen results in `results/fm_v1/phase1_freeze.json` change only through a recorded
   amendment. `python fm_posthoc.py freeze-check` verifies them.

## Repository structure

Only code and small result files are tracked. Data, caches and checkpoints are git-ignored.

The code files sit in the repository root, or beside the document they build. They are not
split into folders, because the notebooks and the FM input lock refer to these exact paths.
Files marked `[lock]` are hashed into the FM input lock (`CODE_FILES` in `fm_prepare.py`).
`stage3_classifier.py` belongs to Group 1, but the locked `fm_prepare.py` and `fm_extract.py`
import it as the Inception-v3 input path.

**Group 1. Replication of Yoo et al. 2021 (closed, 6 November)**

```
├── stage0_manifest.py          build/verify data/manifest.csv
├── stage1_basic_aug.py         basic augmentation -> data/aug_basic/
├── stage1b_normalize.py        C0 classifier-input copies
├── stage2_cyclegan.py          prep | train | generate | qc
├── stage3_classifier.py        Inception-v3, variants none|basic|gan (also used by the FM study) [lock]
├── stage4_evaluate.py          metrics, knn/proto baselines, CAM
├── analysis.py                 training-free analyses (paired tests, kNN, QC, ROC, t-SNE, CAM)
├── figures.py                  plots of saved analyses
├── review_sheet.py             review sheet for looking at rare-disease images by eye
├── c0_experiment.py            C0 source-confound control
├── c1_experiment.py            C1 cached-head seed sweep (planned)
├── c23_experiment.py           C2 GAN amount and C3 common-class caps
├── c5_topup.py                 C5 generation-only top-up
├── fast_head.py                fast cached-head trainer
├── benchmark_head.py           T12 fast-trainer benchmark
├── test_analysis_cluster.py    cluster-bootstrap check
├── colab_setup.ipynb           Colab: restore, rebuild, seed check, CAM, save and push
├── requirements.txt            [lock]
├── docs/research/build_report_evidence.py   audit saved results; refresh Report.tex macros
└── tests/test_c0.py, test_c1.py, test_c23.py, test_c5.py, test_fast_head.py, test_fast_head_tf.py
```

**Group 2. Frozen-encoder (FM) study, Phase 1** (numbers frozen 22 Nov, paper tables and
figures 27 Nov 2026)

*2a. Locked study* (input lock 13 Nov, code of record `2f48654`; evaluate notebook `56fcb52`):

```
├── fm_core.py                  protocol rules (screen, folds, supports, classifiers, nomination) [lock]
├── fm_prepare.py               data preparation and the input lock [lock]
├── fm_extract.py               frozen model loading, timing pilots, feature extraction [lock]
├── fm_evaluate.py              development, nomination, reporting and geometry tables (CPU) [lock]
├── fm_external.py              OCTID/OCTDL screen, query lock and external reporting [lock]
├── colab_fm.ipynb              Colab, GPU: final pilots, input lock, feature extraction (no scores)
├── colab_fm_evaluate.ipynb     Colab, CPU: development, nomination and reports
├── fm_notebook.py              builds both notebooks
├── requirements-fm.txt         pinned PyTorch environment for Colab [lock]
└── docs/research/FM_PROTOCOL.md, FM_AMENDMENTS.md   the binding texts (hashed into the lock)
```

*2b. Post hoc analyses and Phase-1 controls* (each rule committed before its numbers;
protocols.md section 8):

```
├── fm_posthoc.py               P0-P10, R1, R2, G1 (laptop half), E1-E4 and E1-D, E1 report,
│                               E2f marks hash, the Phase-1 freeze (freeze, freeze-check)
├── fm_control_extract.py       E1 and E3: render the controlled inputs, re-extract features (GPU)
├── fm_control_notebook.py      builds colab_fm_control.ipynb (the E1/E3 Colab session)
├── fm_pad_extract.py           G1: pad-to-square re-extraction, MIRAGE and Inception-v3 (GPU)
├── fm_pad_notebook.py          builds colab_fm_pad.ipynb (the G1 Colab session)
├── colab_fm_control.ipynb, colab_fm_pad.ipynb   generated; never edited by hand
├── results/fm_v1/posthoc/      files of record (frozen)
└── results/fm_v1/posthoc/pre_rule/   25 scripts run before any rule covered them; released as
                                evidence, never results (see its README.md)
```

*2c. Papers, reports and independent checks:*

```
├── docs/report/fm_v1_paper/make_tables.py, make_figures.py   the v1 paper's tables and figures
├── docs/report/fm_ieee/        the proposal paper and its 8 build, evidence and verification scripts
├── docs/report/weekly_figures_1121.py   figures of the 21 Nov weekly report
├── docs/research/proposal/build_proposal.py   the proposal figure pack (HTML and SVG)
└── docs/research/fm_independent_review/   6 read-only scripts of the independent code review
```

*2d. Tests:* `tests/test_fm_*.py` (8 files). Run one file at a time, for example
`python -m unittest discover -s tests -p test_fm_core.py`.

## Citation and licensing

- **The code** is MIT-licensed; see [`LICENSE`](LICENSE).
- **The datasets are not redistributed** and keep their own licences. Cite them:
  - Yoo TK, Choi JY, Kim HK. *Feasibility study to improve deep learning in OCT diagnosis of rare
    retinal diseases with few-shot classification.* Med Biol Eng Comput 59:401–415 (2021).
    [doi:10.1007/s11517-021-02321-1](https://doi.org/10.1007/s11517-021-02321-1)
  - Kermany DS et al. *Identifying medical diagnoses and treatable diseases by image-based deep
    learning.* Cell 172(5):1122–1131.e9 (2018).
    [doi:10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
- `stage2_cyclegan.py` adapts the
  [TensorFlow CycleGAN tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan)
  (Apache 2.0).
- This work is independent: it is not affiliated with or endorsed by the paper's authors.
- It is **not a medical device**; research use only.
