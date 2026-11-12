# FM terms, source and external-cohort evidence

> **Closed record (13 November 2026).** Kept as evidence of how decisions were reached; no longer updated. The study as it stands is in [`protocols.md`](../../protocols.md); the binding texts are `FM_PROTOCOL.md` and `FM_AMENDMENTS.md`. The decisions recorded here stand and are copied into the notebooks.

Checked by the agent on 10 November 2026 from public pages and metadata. Sections 1–4 collect
evidence and open decisions. **Lam's terms decisions for the four arms are recorded in section 5**
(10 November); the other decisions in section 4 remain pending.
No checkpoint or image was downloaded and no upstream code was run for this check.

## 1. Model terms (for `DECISIONS` in `colab_fm.ipynb`)

| Arm | Pinned source | Licence seen | Access | Limits of this check |
|---|---|---|---|---|
| RETFound | [weights](https://huggingface.co/YukunZhou/RETFound_mae_natureOCT) `8e551e2f`; [code](https://github.com/rmaphoh/RETFound_MAE/tree/ae9a9ecf37857cf47b8aa9f87cd6f710d75db287) `ae9a9ecf` | Code: CC BY-NC 4.0 (`LICENSE`). Weights: no licence in the public metadata; the signed-in model card states "License: Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)" (read by Lam on 10 Nov at huggingface.co/YukunZhou/RETFound_mae_natureOCT, which also showed "You have been granted access to this model". The card's heading reads "Model Card for RETFound_MAE_MEH", a leftover from the separate `RETFound_mae_meh` repo; its "fill the token in the code" note refers to upstream's training scripts, which this study does not run) | Gated form (name, affiliation, job title). Lam's access confirmed 9 Nov. Approval is automatic: `"gated": "auto"`, no gate prompt or licence field ([API at the pinned revision](https://huggingface.co/api/models/YukunZhou/RETFound_mae_natureOCT/revision/8e551e2f547d3e27aedf43c9cf1c5bcea1c17171), 10 Nov) | The model card needs a signed-in view (README returns HTTP 401); the agent has not read it, and the card text in the Licence column is as Lam reported it. The pinned GitHub README states no weight terms |
| DINOv2-Base | [weights and config](https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415) `f9e44c81` | Apache-2.0 (model card) | Public | None found |
| MIRAGE-Base | [weights](https://huggingface.co/j-morano/MIRAGE-Base/tree/473aadadae9940fcbeccd197db1bc3fac6b376fb) `473aadad`; [code](https://github.com/j-morano/MIRAGE/tree/ee5eae8d3524ec791e719ced1f3cca27eae4c1b0) `ee5eae8d` | CC BY-NC 4.0 for weights and code: "non-commercial, academic research purposes with proper attribution" | Public | None found |
| Inception-v3 (reference) | Keras ImageNet no-top `.h5`, SHA256 pinned in `fm_extract.py` | Keras code: Apache-2.0. Weights: "released under the Apache License" ([Keras 2.2.4 docs](https://raw.githubusercontent.com/keras-team/keras/2.2.4/docs/templates/applications.md), InceptionV3 section, lines 464-468); the same no-top MD5 `bcbd6486…` appears in [Keras 2.0.6 `inception_v3.py`](https://raw.githubusercontent.com/keras-team/keras/2.0.6/keras/applications/inception_v3.py) and `fm_extract.py:61`. Current `keras/src/applications/inception_v3.py` states no weight licence | Public | Corrected 10 Nov after the terms rebuttal (GPT-6.1-Sol); earlier rows said "unverified" |

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

1. ~~**Terms for each of the four arms**, written into `DECISIONS`.~~ Decided 10 November; see
   section 5.

Before the final pilots and S2 lock (Lam's answers of 11 November are in section 6; items
2, 3, 4, 5 and 7 and the R5 code and order are decided; 6 is still open):

2. **External sources:** approve downloading OCTID and OCTDL, or record a real access or
   terms failure and the protocol's reduced scope.
3. **Rare mappings:** CSR-to-CSC and OCTDL MH need clinical confirmation. Without it, keep
   NORMAL FRR and omit those transfer claims (protocol section 8).
   **External code and order** (R5): review the uncommitted `fm_external.py`, and decide
   whether external features may be extracted before S3 (runbook, "External source audit").
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

## 5. Lam's terms decisions (10 November 2026)

Lam's instruction, 10 November 2026: "Use the drafts, date 2026-11-10". The drafts came from
an agent draft revised after an adversarial rebuttal (GPT-6.1-Sol and Gemini 4 Argon High,
10 November); Lam takes responsibility for them as final. The text below is verbatim and is
pre-filled in `colab_fm.ipynb` (first cell); the notebook harness checks that the two agree.

**RETFound:**

> Lam, 2026-11-10. I approve the pinned RETFound_mae_natureOCT weights and models_vit.py as a frozen encoder for feature extraction and timing in our non-commercial student research, run by me or a teammate. Licence: CC BY-NC 4.0 for the code and, per the model card I read on 2026-11-10, for the weights. Each runner uses their own approved Hugging Face account. Tokens and weights are never shared, saved to Drive, redistributed or committed. We will cite Zhou et al. 2023 (Nature).

**DINOv2:**

> Lam, 2026-11-10. I approve the pinned facebook/dinov2-base as a frozen encoder for feature extraction and timing in our non-commercial student research, run by me or a teammate. Licence: Apache-2.0. The weights are not redistributed or committed. We will cite Oquab et al. 2024 (TMLR).

**MIRAGE:**

> Lam, 2026-11-10. I approve the pinned MIRAGE-Base weights and mirage_hf.py as a frozen, OCT-only encoder for feature extraction and timing in our non-commercial student research, run by me or a teammate. Licence: CC BY-NC 4.0 (non-commercial academic research with attribution). Weights and code are not redistributed or committed. We will cite Morano et al. 2025 (npj Digital Medicine).

**Inception:**

> Lam, 2026-11-10. I approve the pinned Keras ImageNet no-top Inception-v3 weights as the reference arm only, never nominated, for feature extraction and timing in our non-commercial student research, run by me or a teammate. Licence: Apache-2.0 for the Keras code and, per the Keras 2.2.4 documentation, for these weights. The weights are not redistributed or committed.

## 6. Lam's lock decisions (11 November 2026)

Lam answered the section 4 list on 11 November. Lam's words are quoted; the agent's notes
follow each.

| Item | Lam's answer | Status and note |
|---|---|---|
| 2. External sources, and R5 external code and order | "Approve" | Approved: download OCTID and OCTDL; review and use the drafted `fm_external.py`; external features may be extracted before S3 (runbook, "External source audit"). The code is still uncommitted and the notebook download step still lives outside the repo. |
| 3. Rare mappings | "How to confirm without expert? Any previous paper help that process"; then, 12 Nov: "We adopt published dataset labels for exploratory external evaluation only. Results measure agreement with those labels, not clinically validated disease transfer. Cohort eligibility is fixed before inference, and all planned results will be reported." | **Decided 12 Nov:** amendment A1 in `FM_AMENDMENTS.md` (cohorts, rules and wording). Options B and C are no longer needed. |
| 4. Numerical acceptance rule (R3) | "Approve but explain pros and cons" | Approved as proposed: run each encoder twice in one process on five fixed synthetic images and require a largest element-wise difference of at most `1e-4` of the largest absolute output value. Comparing against upstream's own outputs was not chosen, so it is not part of the rule. The check is not implemented yet. |
| 5. Environment plan (R2) | "Approve" | Approved: final pilots, lock approval and full extraction run in one continuous Colab runtime; if that runtime is lost before any score, keep the attempt and redo pilots under a new approved lock. |
| 6. Lock transport (R8) | "I only make the Drive public for my team work, the repo I will put it back to private" | **Needs one confirmation:** whether the lock may be committed once the repository is private. On 11 November the repository was still public (GitHub API `visibility: public`). |
| 7. Protocol sections 4 and 6 walkthrough and risk acknowledgement | "Approve" | Approved; scheduled after the first score-free checks. |

**Rare-mapping evidence (item 3), read 11 November.** The protocol requires "clinical
label-equivalence confirmation" for CSR-to-CSC and says OCTDL "metadata alone is not clinical
certification" (protocol section 8). What is available without a clinician:

- Naming: MeSH descriptor D056833 "Central Serous Chorioretinopathy" lists "Central Serous
  Retinopathy" as an entry term (NLM MeSH lookup API). OCTID's dataset is titled "Central serous
  retinopathy retinal OCT images" (Borealis, doi:10.5683/SP/FVLZIL).
- Earlier papers treat OCTID's CSR class as CSC: EyeDiff (Chen et al., npj Digital Medicine
  2026, doi:10.1038/s41746-026-02560-2, Fig. 1 table) lists OCTID as "MH, CSCR, AMD DR, Normal";
  a 2026 TVST external validation (doi:10.1167/tvst.15.1.6, PMC12786394) describes OCTID's
  categories as "AMD, CSC, FTMH, and Healthy"; a 2025 BMC Ophthalmology review of CSC deep
  learning (doi:10.1186/s12886-025-04372-6, PMC12502260) calls OCTID's class CSCR.
- OCTDL labels (Kulyabin et al., Scientific Data 2024, PMC11009408): seven trained medical
  students labelled first, two clinical specialists labelled independently with consensus, and
  the head of the clinic confirmed every diagnosis. The paper defines macular hole as "a
  complete defect in the inner layers of the retina that extends to the RPE" and treats the
  lamellar tear as a separate variant, which matches the `subcategory` split 39 MH / 26 LH /
  11 VMT.
- Unverified: how Yoo et al.'s MH and CSC images were defined, so equivalence with the
  benchmark's own labels rests on the disease names.

These sources support the names and show the datasets were labelled by clinicians. They are
not a clinician confirming that these external labels match the benchmark's labels, which is
what the protocol asks for.
