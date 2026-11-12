# FM study: coding plan (8 November 2026)

> **Closed record (13 November 2026).** Kept as evidence of how decisions were reached; no longer updated. The study as it stands is in [`protocols.md`](../../protocols.md); the binding texts are `FM_PROTOCOL.md` and `FM_AMENDMENTS.md`.

How we turn [protocol v1.0](FM_PROTOCOL.md) into code for stone S2 of the
[checklist](FM_CHECKLIST.md). The protocol stays the authority; `§n` points to it.
Nothing here is an input lock. No pretrained weights have been downloaded and no
actual encoder has been executed.

9 November update: all four files under layout A are now drafted. Synthetic CPU
contracts, repeated rebuttal dispositions and remaining Colab gates are tracked
in [the review record](FM_CODE_REVIEW.md) and [runbook](FM_RUNBOOK.md). This does not
complete S2 or Lam's paced protocol walkthrough; no real lock or score exists.

## 1. What we already know (dry runs, no model scores)

Read-only checks on the real manifest and images, using the draft `fm_core.py`.
They preview the gates; the locked versions run at S2 with pinned versions.

| Check | Result | Consequence |
|---|---|---|
| Manifest vs §3 | 84,630 rows; rare development 27/26/25/19/27; legacy 5/5/4/4/4; common fitting 73,958 | Protocol counts hold |
| §8 inventory | `major_indices(…, 1000, 42)` gives 1,000 per class; 4,000 + 1,000 + 146 = 5,146 | Existing helper is reusable as written |
| §4 exact screen | 0 equal file hashes, 0 equal RGB fingerprints among 146 images | Nothing to exclude |
| §4 dHash ≤4 | 0 flagged pairs, so no raw-image review is triggered | Groups are 124 singletons, no legacy exclusions |
| Is the hash working? | 146 distinct hashes; nearest-neighbour distance min 5, median 10, max 21; only 17.7 of 64 bits set on average | Not blind, but weak (see below) |
| §4 feasibility | 27/26/25/19/27 groups, all ≥10; worst fold still has 15 fitting groups (Stargardt) | Selector can run; no amendment needed |
| K=16 reporting | Every class has ≥16 groups | Feasible; Stargardt uses 16 of 19, so rosters overlap heavily |
| §6 rosters | 2,500 development rosters in about 1 s | Deterministic and cheap |
| §6 L-BFGS-B | Converged 40/40 synthetic fits at d = 512–2048, K = 4 and 16; 8–43 ms each | Fixed settings look workable; real features still untested |

**Caveats to carry forward.**
- Ten images sit at distance 5, one bit above the fixed limit, including cross-disease
  pairs (CSC/21 ~ RP/14, CSC/30 ~ MH/24). Dark OCT backgrounds make this hash coarse.
  The ≤4 limit was fixed before this run and must not move. Report the distribution:
  zero flags does not show the images come from different patients.
- The manifest's rare `pid` values (`CSC_01`, …) are one per image. They are not
  verified patient links and must never feed grouping.

## 2. Checkpoint gate: public metadata only (§5)

The metadata probe downloaded no weights and accepted no terms. Sizes and revisions
come from the Hugging Face API on 8 November; Lam's later access/terms decisions
are recorded in section 6 below.

| Arm | Revision | Access | Licence tag | Weights | Native input | Still to pin at the gate |
|---|---|---|---|---|---|---|
| RETFound Nature OCT | `8e551e2f` | Gated (`auto`): HF account, access form, token; Lam reported access 9 Nov | None on the API; code repo is CC BY-NC 4.0 | 3.95 GB `.pth` (larger than a ViT-L encoder alone; contents unverified, plan the download) | 224; its train script uses `--global_pool` | Actual weight loading and key layout |
| DINOv2-Base | `f9e44c81` | Open | Apache-2.0 | 346 MB | Shortest edge 256, centre crop 224 | Confirm embedding in the pilot |
| MIRAGE-Base | `473aadad` | Open | CC-BY-NC-4.0 | 348 MB | 512, patch 32, `bscan-slo` config | Actual load/pilot; code revision and asset hashes are pinned in the draft |
| BioMedCLIP (fallback) | `9f341de2` | Open | MIT | 784 MB | 224 | Used only if RETFound fails |
| Keras Inception-v3 | Historical | Built in | n/a | n/a | 256 then 299, GAP | Bypass the float16 cache writer |

The metadata check left MIRAGE's non-commercial terms for Lam's judgement; Lam's
subsequent acceptance is recorded in section 6.
DINOv2's centre crop drops field of view; §5 keeps native transforms and logs it.

## 3. Code layout: options

| | Layout | For | Against |
|---|---|---|---|
| **A (chosen)** | `fm_core.py` (pure) + `fm_prepare.py` (screen, rosters, lock; no GPU) + `fm_extract.py` (one encoder per process; Colab) + `fm_evaluate.py` (CPU heads, nomination, reports) | Matches the flat repo and the "complexity at the I/O boundary" rule; PyTorch and Keras never share a process; lock step runs on the laptop | Four top-level files |
| B | A `fm/` package with one module per concern | Tidier as it grows | Fights the flat layout; more ceremony for a 3-week study |
| C | One `fm_experiment.py` with subcommands, like `c23_experiment.py` | One file | Mixes framework imports with pure logic; hard to test; the file that needs to be stable also changes most |

Under A, `requirements-fm.txt` pins the Colab PyTorch stack and Pillow; TensorFlow
stage scripts stay untouched (§5). `tests/test_fm_core.py` joins `.agents/check`.

## 4. Data flow under A

```text
data/manifest.csv + raw images
  -> fm_prepare screen    candidates, duplicate decisions (Lam reviews any flags)
  -> fm_prepare freeze    groups, folds, exclusions, episodes, nine-way roster
  -> fm_prepare lock      input_lock.json: protocol hash, roster, models, code, endpoints
  -> fm_extract <arm>     raw float32 embeddings + identity sidecar + timing
  -> fm_evaluate develop  query-level predictions, S_m, nomination (19 Nov)
  -> fm_evaluate report   legacy + external queries, nine-way FRR, summaries
```

- Every stage reads the lock and stops on any identity mismatch (§9). Completion
  markers are written last with artefact hashes, adapting the `c23_experiment.py`
  `writer_lock` and `atomic_json` patterns.
- Cache identity holds ordered image IDs and content hashes, weight revision/hash,
  transforms and pooling, environment, dim/dtype and code revision (§5).
- Do not reuse `stage4_evaluate.baselines` or `analysis.centroid_correct`: neither
  normalizes supports first (§6).

## 5. Build order

| Step | Deliverable | Done when | Needs |
|---|---|---|---|
| 0 | Merge PR #23; branch from `main` | Protocol and results on `main` | Lam |
| 1 | Review and land `fm_core.py` and its tests (drafted, on `t3/fm-code-prep`) | Lam has read §4/§6 code against the protocol | Lam |
| 2 | `fm_prepare.py` screen + freeze: **drafted**; final `lock` waits for checkpoint hashes | Done on real data 8 Nov: two runs byte-identical, 5,146 images, 4,124 fitting + 1,022 query rows (preview, not the lock) | Laptop |
| 3 | `fm_extract.py`: drafted for fixed arms, native pipelines and identity/resume/timing guards | CPU mocks pass; actual 100-image pilot still required | Colab GPU |
| 4 | Keras Inception reference path: drafted | Separate process and FP32 cache; actual load/extraction unverified | Colab |
| 5 | `fm_evaluate.py`: synthetic development, nomination, legacy reporting and paired summaries drafted | Hand-built counts and persistence tests; real features still pending | CPU |
| 6 | External inventories (OCTID, OCTDL) | Terms, labels, eligibility frozen before inference | Lam, clinical mapping |
| 7 | Final input lock, then S3 | Archive timestamp precedes every labelled score | All above |

Steps 1 and 2 need no GPU or checkpoint access. S1 (the 11 Nov report) keeps priority:
none of this may delay it.

## 6. Decisions (8 November 2026)

Lam chose layout **A** and delegated items 2-5 to the agent; acceptance of licence terms
stays with Lam. Every choice below is made before any encoder has been run or scored,
and none may be revisited because of an outcome.

**Lock location.** Rosters, episodes and `input_lock.json` are committed under
`results/fm_v1/lock/`. Code only reaches Colab through GitHub, and the commit timestamps
the lock before any score exists. Features and runs stay in `/content/fm_v1` (§9). This is
not a public preregistration and must not be described as one.

**Saving features.** Keep §9 unchanged: no automatic Drive writes, no amendment. Download
a hashed archive after each encoder. 5,146 images x dimension x 4 bytes is about 21 MB
(RETFound, 1,024-d), 16 MB (DINOv2, MIRAGE, 768-d) and 42 MB (Inception, 2,048-d), so
manual export is cheap. To cut staging, `fm_prepare` packs only the 5,146 images into one
hashed zip for Lam to upload, instead of the 5.5 GB `oct2017.zip`.

**One embedding per arm**, each the authors' own documented path. No alternative pooling
is tried.

| Arm | Input (native) | Embedding | Source |
|---|---|---|---|
| RETFound | 256 bicubic, centre crop 224, ImageNet mean/std | `RETFound_mae(global_pool=True)`: mean of patch tokens, then `fc_norm`; 1,024-d | `models_vit.py`, `main_finetune.py` default, `util/datasets.py` eval (code `ae9a9ecf`) |
| DINOv2 | Its `preprocessor_config.json` | `pooler_output` (class token after layer norm); 768-d | HF model files; confirm in the pilot |
| MIRAGE | Native PIL 512 bilinear, then grayscale, [0,1]; pass `{"bscan": x}` only, so OCT-only | `--pool global`, the authors' default: their `LayerNorm` on each token, then patch-token mean with global token excluded | `hf/mirage_hf.py`, `mirage_wrapper.py:MIRAGEClsGlobal.forward/pool`, `run_cls_tuning.py` (code `ee5eae8d`) |
| BioMedCLIP (fallback only) | Its open_clip transform | Projected image feature; 512-d | HF `open_clip_config.json` |
| Inception-v3 | `stage3_classifier.load_image` | Global average pool; 2,048-d | `stage3_classifier.py` |

The MIRAGE ordering above corrects the 8 November plan's reversed norm/mean order,
based on the pinned upstream `forward` and `pool` functions opened on 9 November.
No previous feature run used that description; the protocol is unchanged.

The pilot must log what each load reports. In RETFound and MIRAGE the wrapper norm layer
comes from the classifier wrapper and is expected to be absent from the checkpoint
(unverified until the weights load). It then holds default values, so it is deterministic
but untrained. Record this as a property of the pipeline; do not swap it out.

**Licence terms: accepted by Lam on 8 November for both, CC BY-NC 4.0** (MIRAGE weights and the
RETFound code repository): non-commercial use with attribution. Cite both papers and never
commit weights. MIRAGE needs no form and its files downloaded openly (HTTP 200) in
the metadata check. RETFound returned HTTP 401 without an approved account then;
its form and account-specific terms remain Lam's responsibility. The agent has
no account there. The later access result is recorded below.

9 November access update, reported by Lam: the Colab access test printed
`RETFound access OK` for account `Malnoad`. Weights have not been downloaded.

If RETFound access is not working by the input lock, BioMedCLIP replaces it (§5) and the
arm cannot return in v1.

`scipy` and `pillow` are now listed in `requirements.txt`; versions used are saved in each
summary file.
