# FM terms, source and external-cohort evidence

Checked by the agent on 10 November 2026 from public pages and metadata. **Lam's decisions
are pending.** This file collects evidence for those decisions and records none of them.
No checkpoint or image was downloaded and no upstream code was run for this check.

## 1. Model terms (for `DECISIONS` in `colab_fm.ipynb`)

| Arm | Pinned source | Licence seen | Access | Limits of this check |
|---|---|---|---|---|
| RETFound | [weights](https://huggingface.co/YukunZhou/RETFound_mae_natureOCT) `8e551e2f`; [code](https://github.com/rmaphoh/RETFound_MAE/tree/ae9a9ecf37857cf47b8aa9f87cd6f710d75db287) `ae9a9ecf` | Code: CC BY-NC 4.0 (`LICENSE`). Weights: no licence in the public metadata | Gated form (name, affiliation, job title). Lam's access confirmed 9 Nov | The model card needs a signed-in view, so the agent has not read it. Lam should read it |
| DINOv2-Base | [weights and config](https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415) `f9e44c81` | Apache-2.0 (model card) | Public | None found |
| MIRAGE-Base | [weights](https://huggingface.co/j-morano/MIRAGE-Base/tree/473aadadae9940fcbeccd197db1bc3fac6b376fb) `473aadad`; [code](https://github.com/j-morano/MIRAGE/tree/ee5eae8d3524ec791e719ced1f3cca27eae4c1b0) `ee5eae8d` | CC BY-NC 4.0 for weights and code: "non-commercial, academic research purposes with proper attribution" | Public | None found |
| Inception-v3 (reference) | Keras ImageNet no-top `.h5`, SHA256 pinned in `fm_extract.py` | Keras code: Apache-2.0. Weight licence: **unverified**, not stated in `keras/src/applications/inception_v3.py` | Public | The same weights already underlie the Yoo reproduction |

The two CC BY-NC arms restrict use to non-commercial work and require attribution. Weights
stay out of this public repository.

## 2. Pinned upstream source read (risk R1)

Two files execute as upstream Python. Both were read as text at the pinned commits, and
their SHA256 values equal `fm_extract.EXPECTED_ASSET_SHA256`.

| File | Imports | Reached by the loader | Not reached |
|---|---|---|---|
| `models_vit.py` | `functools`, `timm`, `torch` | `RETFound_mae`, `forward_features` | `Dinov3` calls `torch.hub.load(..., trust_repo=True)` and the other builders use `pretrained=True`; none is called |
| `mirage_hf.py` | stdlib, `requests`, `PIL`, `numpy`, `torch`, `einops` | `MIRAGEWrapper`, `MIRAGELight` | `load_process_image` calls `requests.get`; the `__main__` block calls `torch.load(weights_only=False)`. `import_source` does not run `__main__` |

Neither file does anything at import beyond imports and definitions. This is a static read
of two files, not a sandbox. It does not cover `timm`, `transformers` or other packages.
`colab_fm.ipynb` Step 8 still removes credentials from the worker and requires Drive off.

## 3. External cohorts (protocol section 8, risk R5)

Metadata only; no image was downloaded.

| Source | Release | Licence | Candidates in the release | Digest available |
|---|---|---|---|---|
| OCTID | [Borealis dataverse](https://borealisdata.ca/dataverse/OCTID), per-class datasets, each v1.0, published 19 Dec 2018 | CC0 1.0 | NORMAL 206 ([SP/WLW4ZT](https://doi.org/10.5683/SP/WLW4ZT)); MH 102 ([SP/MBMQGD](https://doi.org/10.5683/SP/MBMQGD)); CSR 102 ([SP/FVLZIL](https://doi.org/10.5683/SP/FVLZIL)) | Per-file MD5 |
| OCTDL | [Zenodo 10839556](https://zenodo.org/records/10839556), published 19 Mar 2024 | CC BY 4.0 | NORMAL 332 from 110 patients; VID 76, split 39 MH / 26 LH / 11 VMT by `subcategory` | `OCTDL.zip` MD5 `dec9766fc916202c45a333c4f5a09576`; `OCTDL_labels.csv` MD5 `2fced789038c6ca83283066c2f8ad289` (SHA256 `b7db437c…8b3a`, fetched and matched) |

The lock needs SHA256 of the actual release bytes, which only exists after an approved
download. OCTID lists no patient IDs in this metadata; patient independence is unknown.

## 4. Decisions still needed from Lam

Before the first pilots:

1. **Terms for each of the four arms**, written into `DECISIONS`.

Before the final pilots and S2 lock:

2. **External sources:** approve downloading OCTID and OCTDL, or record a real access or
   terms failure and the protocol's reduced scope.
3. **Rare mappings:** CSR-to-CSC and OCTDL MH need clinical confirmation. Without it, keep
   NORMAL FRR and omit those transfer claims (protocol section 8).
4. **Numerical acceptance rule** (R3). Agent's proposal, not adopted: run each encoder twice
   in one process on five fixed synthetic images and require a largest element-wise
   difference of at most `1e-4` of the largest absolute output value. Whether to also
   compare against upstream's own outputs is Lam's call.
5. **Environment plan** (R2). Agent's proposal, not adopted: run the final pilots, lock
   approval and full extraction in one continuous Colab runtime. If that runtime is lost
   before any score, keep the attempt and redo pilots under a new approved lock.
6. **Lock transport** (R8). Agent's recommendation, not adopted: carry the lock privately
   as its verified ZIP instead of committing it to this public repository.
7. **Protocol sections 4 and 6 walkthrough and the risk acknowledgement**, already scheduled
   for after the first score-free checks.
