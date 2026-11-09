# Frozen-encoder code preparation and runbook

Status: 9 November 2026, **implementation under review; no new FM results**.
[Protocol v1.0](FM_PROTOCOL.md) is authoritative. This guide adds no scientific
recipe and is not the input lock. The November 8 image previews are not a lock.
The new scripts do not participate in `colab_setup.ipynb`'s reproduction cells.

Before real execution, review the [risk and acknowledgement record](FM_RISK_ACKNOWLEDGEMENT.md).
It distinguishes existing automatic checks from manual obligations and records
the unresolved external implementation/versioning decision before the real code lock.

## What is ready to review

- `fm_core.py`: image screen, groups/folds, paired supports, float64 classifiers,
  endpoint arithmetic and nomination. Exact random-stream/objective tests run on CPU.
- `fm_prepare.py`: screen/review/freeze, verified-patient links, explicit support
  and query IDs, exclusion/component/fold ledgers, selected-image ZIP, final lock,
  hashed export and token unlock. No encoder or labelled score.
- `fm_extract.py`: staged-asset specifications, a score-free timing pilot and
  raw float32 caches with identities and completion receipts. GPU commands reject
  execution outside Colab. One encoder and one framework per process.
- `fm_evaluate.py`: paired development, nomination, legacy five-/nine-way
  reporting and conditional support-draw summaries. CPU only; cannot substitute
  historical quantized arrays or fitted softmax heads.

**Still unverified:** real checkpoint loading, package compatibility in Colab,
GPU numerical behavior, throughput and real-feature logistic convergence. No
pretrained weights were downloaded. The planned CPU math/image subset was tested
in a temporary environment without modifying the project venv; this does not
verify the full Colab framework stack. RETFound
access was reported by Lam on 9 November (`RETFound access OK`, account `Malnoad`);
this is access evidence, not a successful model load.

External acquisition, external duplicate/patient screening, the separate final
external query lock and external inference are **not implemented by these legacy
commands**. They remain an S2/S4 deliverable. A legacy report is not completion of
the required external source component. Optional cluster intervals and figures
also remain future work. No manuscript outcome can be inferred from these tests.

## Preparation on the laptop

Use `D:/Few-shot/.venv/Scripts/python.exe` from this worktree. The following are
**future execution recipes**, not a record that these actions have occurred.
Keep preview outputs in a separate folder from the eventual author-approved lock.

```powershell
& D:/Few-shot/.venv/Scripts/python.exe fm_prepare.py screen --root D:/Few-shot --out PREVIEW
& D:/Few-shot/.venv/Scripts/python.exe fm_prepare.py freeze --root D:/Few-shot --out PREVIEW
& D:/Few-shot/.venv/Scripts/python.exe fm_prepare.py pack --root D:/Few-shot --out PREVIEW --archive selected-images.zip
```

For flagged dHash pairs, supply `--reviews decisions.csv` to `freeze`: columns
`a,b,decision`, with `confirmed`, `unresolved` or `rejected`. Exact file/RGB links
are automatic. Every flagged dHash pair requires exactly one decision. Review
raw images without new encoder scores; the fixed Hamming limit stays **4**.

If genuine rare patient links become available, additionally supply `--patients
verified-patients.csv`. Columns are `path,dataset,patient_id,evidence`. The dataset
namespaces patient IDs; the manifest's rare `pid` values are not patient evidence.
Cross-disease verified patients stop the selector and require Lam's amendment.
Image duplicates with conflicting labels stop for resolution.

`freeze` saves split-block candidate counts, review decisions, image components,
verified patient evidence, legacy-linked fitting exclusions, fold counts,
development query IDs, all nested support IDs, the exact nine-way roster and the
100 timing IDs. Excluded fitting images do not enter the extraction inventory.
Canonical test rows and the exact common cap stay fixed.

Screen/freeze output bytes are immutable. Changed source bytes, manifests,
decoders, producer code or recipes stop reuse. A code fix before any scores needs
fresh preparation/pilots under the fixed protocol; do not overwrite prior outputs.
`selected-images.zip` preserves the original bytes and POSIX IDs. Check its JSON
SHA256 receipt after copying. Upload/extract it only into the intended Colab
input root; no automatic Drive output write is made.

## Model assets and isolated Colab processes

Code reaches Colab through GitHub, so Lam must authorize a push before using the
new scripts there. Do not use the reproduction notebook's Run All for this study.
`colab_fm.ipynb` stages the code, separate environment, images from `octfsl/raw`,
rosters and pinned assets, checking each by hash. Steps 7-10 then run score-free first
checks of each encoder, after Lam fills in the terms decisions ([evidence](FM_TERMS_EVIDENCE.md)).
It creates no lock, and these are not the final pilots: its rosters and pilots are a
preview to regenerate if the locked code later changes.
`requirements-fm.txt` pins the proposed PyTorch stack. Its Windows CPU variant
resolved in an isolated temporary environment and passed `pip check`; actual
Colab dependency and encoder compatibility remains unverified. Install the study
runtime in its dedicated Colab environment. The Keras reference
runs in its own process with TensorFlow available, preserving the legacy loader.

Before final pilots, plan to preserve or demonstrably rebuild each full recorded
environment, including the separate TensorFlow reference. Extraction requires
exact equality of Python, platform string and every installed distribution with
the pilot/lock identity. Direct pins or a full package freeze alone cannot
reproduce a Colab base image/kernel. Keeping one runtime alive through lock
approval/transfer and all extraction is an operational option, not a guarantee.
An environment change stops extraction; before scores, preserve the attempt and
make fresh pilots plus an author-approved new lock. After scores, refer changed
identities to Lam for a dated amendment/new version rather than relaxing guards.

The temporary Windows Python 3.12.10 check used NumPy 2.2.6, SciPy 1.15.3,
Pillow 11.2.1, pandas 2.2.3 and scikit-learn 1.6.1. All 148 FM tests on the pre-RETFound-
metadata-fix snapshot passed in both that environment and the laptop environment;
`pip check` passed, and all 14 synthetic frozen CSVs matched byte-for-byte.
This verifies the CPU subset and RNG/roster portability for those versions, not
Linux/Colab, CUDA or actual model loading. The later extractor-only metadata fix
passed seven native checks with CPU-only PyTorch 2.7.1+cpu and synthetic checkpoints
in that temporary environment. The recorded full project gate passed 149 FM tests.

RETFound's official writer can save `argparse.Namespace` metadata. Its loader
temporarily permits only that class while retaining `weights_only=True` and
restoring prior permissions on exit. The exact published OCT checkpoint's
metadata remains unverified; any other unsupported type still stops loading.
This CPU serialization check did not construct an encoder or load public weights.

The temporary CPU stack also passed native import/signature checks and 189 image
transform comparisons. OpenCLIP's transform was tested separately from package
initialization: the normal import attempts an unused local tokenizer, which the
probe blocked before its constructor body. No text encoder or text scoring is
used. DINO's pinned native processor rejects one-pixel-high images through channel
inference; those three synthetic rejections were asserted separately. The 5,146
selected originals have minimum width 290 and height 258, so none hit that boundary.
These checks do not verify pretrained models or GPU execution.

Acquire assets at these pinned identities through Lam's approved access route.
Keep tokens private and weights outside the repository. `spec` hashes existing
local files; it neither downloads nor accepts any terms.

For every arm, record the actual terms source, review date and Lam's decision.
`--terms-evidence` checks only nonempty text; it cannot verify consent or terms.
Record each acquisition's source, byte size, elapsed seconds and date in the
manual download-cost ledger. Download separately from executing upstream source.
Before execution, review worker access to credential environment variables and
the Hugging Face login cache; use a worker without those credentials and leave
Drive unmounted unless needed. Offline flags do not clear credentials or sandbox
Python source. These are manual precautions and remain pending.

| Arm | Checkpoint revision | Required local asset names |
|---|---|---|
| RETFound | `8e551e2f547d3e27aedf43c9cf1c5bcea1c17171` | `RETFound_mae_natureOCT.pth`, `models_vit.py` |
| DINOv2 | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` | `model.safetensors`, `config.json`, `preprocessor_config.json` |
| MIRAGE | `473aadadae9940fcbeccd197db1bc3fac6b376fb` | `model.safetensors`, `config.json`, `mirage_hf.py` |
| BioMedCLIP, fallback only | `9f341de24bfb00180f1b847274256e9b65a3a32e` | `open_clip_pytorch_model.bin`, `open_clip_config.json` |
| Inception | `keras-imagenet-notop` (local identity label; verified weight SHA256 pinned) | `inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5` |

RETFound source pin: `ae9a9ecf37857cf47b8aa9f87cd6f710d75db287`.
MIRAGE source pin: `ee5eae8d3524ec791e719ced1f3cca27eae4c1b0`.
The asset folder is per encoder. Weight/config/source bytes must match the public
digests in `fm_extract.EXPECTED_ASSET_SHA256`; a typed revision string alone is
insufficient. The Inception SHA256 was computed from the existing historical
local cache; its MD5 matches the official Keras no-top hash. No weights were downloaded.
Do not copy unrelated model files into the asset folder.
Do not replace RETFound with BioMedCLIP for an accuracy outcome. A substitution
requires a recorded access/loading/terms failure before scores and stays fixed.

Example DINOv2 asset spec, in the dedicated Colab environment:

```sh
python fm_extract.py describe --arm DINOv2
python fm_extract.py spec --arm DINOv2 --assets /content/assets/DINOv2 --revision f9e44c814b77203eaa57a6bdbbd535f21ede1415 --terms-evidence "REPLACE_WITH_TERMS_SOURCE_DATE_AND_LAMS_ACTUAL_DECISION" --out /content/fm_v1/specs/DINOv2.json
```

For RETFound/MIRAGE add `--source-revision` with the pin above. Use
`--revision keras-imagenet-notop` for Inception; its actual weight SHA256 and
Keras package environment are locked. Preserve attribution/licence evidence.

The pipelines follow their native order, including alpha/channel handling:

- RETFound: native RGB, bicubic short edge 256, center crop 224, ImageNet
  normalization, patch-token mean followed by `fc_norm`, 1,024 dimensions.
- DINOv2: its slow native image processor/configuration and normalized class-token
  `pooler_output`, 768 dimensions. Crop coordinates use the processor's integer
  convention, rather than assuming torchvision rounding.
- MIRAGE: native PIL bilinear resize 512, then grayscale/[0,1]; pass only `bscan`.
  The authors' classification-wrapper LayerNorm acts **on tokens before patch
  averaging**, excluding the final global token; 768 dimensions. The earlier
  coding plan had that order reversed; no previous feature run used it.
- BioMedCLIP: native OpenCLIP evaluation transform, projected image tower only,
  512 dimensions. OpenCLIP imports include an unused local tokenizer; no text
  encoder is constructed and no text inputs or text scores are used.
- Inception: exactly `stage3_classifier.load_image` (TF decode, 256 then 299,
  historical scaling) and Keras no-top GAP backbone, 2,048 dimensions. Fresh FP32
  inference/cache; the old float16 writer is bypassed.

The pilot load report records actual missing/unexpected checkpoint keys and the
declared policy for deterministic default norm parameters. Its `default_norm`
description is not an independent measurement of parameter values. An unexpected
missing encoder parameter stops the arm. A successful pilot checks loadability,
shape, FP32 and finite/nonzero outputs; it does not establish upstream embedding
equivalence or perform a repeat-output comparison. Write a manual numerical
acceptance rule and record Lam's decision on any additional upstream-output check
before final pilots. Run such checks in a separate process and wait for it to
exit before starting the timing sequence.

## Score-free timing pilot, then final lock

Resolve the external implementation/versioning plan before final pilots and S2.
The four FM scripts, stage 3 and both requirements files are hashed in the lock;
later edits invalidate it. Core/preparation/stage-3 changes also require fresh
screen/freeze artifacts. Follow R5 in the risk record: implement/test the intended
external workflow, or document genuine access/terms failure and the protocol's
reduced scope. Unfinished code is not external-data unavailability. Planned
sources need real release digests and source/terms/eligibility evidence at S2;
the separate final screened query IDs may follow before external inference.
If S2 slips, S3 starts after the successful lock; the 11 November report proceeds.

Run each pilot as a separate synchronous command and **wait for its process to
exit**. One shared `--timing-session` string identifies one uninterrupted Colab
runtime. Time FMs in RETFound/fallback → DINOv2 → MIRAGE order, skipping arms
already documented unavailable; then run the reference. Do not run pilots in
parallel. Changes to runtime/hardware require retiming all candidates before cost
can break a nomination tie.

Retiming requires fresh `--out` folders and a new shared `--timing-session`.
A completed pilot rerun returns its existing verified receipt. Keep all attempts,
PID/exit evidence and the reason the final comparable sequence was selected.

```sh
python fm_extract.py pilot --root /content/Few-shot --prepared /content/prepared --spec /content/fm_v1/specs/DINOv2.json --assets /content/assets/DINOv2 --out /content/fm_v1/pilots/ATTEMPT_ID/DINOv2 --timing-session SESSION_ID
```

The realized 100 IDs come from `timing_inventory.csv`; repeated IDs are timing-only
when there are fewer than 100 eligible images. Disk I/O/decoding occur before the
timed passes. Timing includes preprocessing/inference/output transfer, batch1,
FP32, one warm pass then five timed passes with GPU synchronization. Validation
is outside the intervals. Use the median of the five per-pass means.

The final gate JSON has these fields; replace examples with actual evidence:

```json
{
  "arms": [
    {"name": "RETFound", "status": "available", "pilot": "LOCAL_PILOT_DIRECTORY"},
    {"name": "DINOv2", "status": "available", "pilot": "LOCAL_PILOT_DIRECTORY"},
    {"name": "MIRAGE", "status": "available", "pilot": "LOCAL_PILOT_DIRECTORY"},
    {"name": "Inception", "status": "available", "pilot": "LOCAL_PILOT_DIRECTORY"}
  ],
  "external_sources": {
    "OCTID": {"status": "REPLACE_WITH_ACTUAL_STATUS"},
    "OCTDL": {"status": "REPLACE_WITH_ACTUAL_STATUS"}
  }
}
```

The external placeholder statuses are intentionally invalid; replace them with
real reviewed decisions and required fields. `unavailable` needs evidence of an
actual access/terms failure, never the example text or unfinished implementation.
The code verifies field presence, not the truth of these declarations.

An unavailable arm uses `status: unavailable` and a nonempty `reason`, not a pilot
directory. Account for every fixed slot. A fallback roster replaces RETFound's
entry with BioMedCLIP and includes `retfound_failure`. A planned external source
needs `status: planned`, `release`, `release_sha256`, `terms_evidence`, `eligibility`
and `label_inventory`. These definitions do not authorize unimplemented external
inference. Do not infer CSR↔CSC or MH subcategories without clinical confirmation.

After Lam reviews the code, pilots and decisions, and **before any labelled
scores**, assemble the final prepared folder and pilot files on the locking
machine. Use the Python/NumPy/SciPy environment intended for CPU evaluation:

```powershell
& D:/Few-shot/.venv/Scripts/python.exe fm_prepare.py lock --root D:/Few-shot --out results/fm_v1/lock --gate model-gate.json --archive input-lock.zip
```

`lock` revalidates preparation producer/code/recipe, exact folds/support/query
rosters, feasible K values, source manifest/bytes and canonical nine-way rows.
Pilots must bind the entire preparation summary, all four hashed pilot artifacts,
fixed timing IDs, operation receipts, common FM environment/hardware/session and
ordered nonoverlapping process windows. Available/unavailable arms stay fixed.

The portable lock includes code hashes, actual environments, endpoint definitions,
copied pilot evidence and original timestamps. Its ZIP receipt hashes all members.
Lam must approve the exact archive SHA256; preserve its copied bytes and a
timestamp/receipt outside the runtime before labelled scores. The lock's local
`created_utc` alone is not independent timestamp evidence. Code reaches Colab
through an authorized GitHub push; the lock may instead travel privately as its
verified ZIP. If Lam chooses to commit/push a lock, first review its terms and
patient evidence and add a reviewed `results/fm_v1/** -text` attribute, then check
Git-stored and fresh-checkout bytes against the raw hashes. The current checkout
has `core.autocrlf=true` without that protection; the attribute change remains
pending. Public publication is not implied by lock creation. This is not public
preregistration. A matching code hash
is authoritative across CRLF/LF and authorized commits; the creation Git revision
is retained as provenance. Locking does not itself commit or push anything.

Keep the input-lock ZIP and its adjacent `.zip.json` receipt together. Restore
with the saved SHA256 rather than unpacking the ZIP by hand: `restore` verifies
the archive and member hashes, preserves the original lock bytes/timestamps and
recreates `lock_archive.json` without needing the original source folder.

```powershell
& D:/Few-shot/.venv/Scripts/python.exe fm_prepare.py restore --root D:/Few-shot --out RESTORED_LOCK --archive input-lock.zip --sha256 SAVED_ARCHIVE_SHA256
```

This restores the S2 input snapshot. Later S3 nomination records/bindings and run
archives must also be preserved; restoring an old S2 ZIP does not restore later
study state. Never use a restore to re-designate a nominee after reporting.

## Extraction, nomination and legacy reporting

Only after that gate, run each arm separately. Outputs stay under
`/content/fm_v1/{features,runs,reports}`; no automatic Drive result writes.
As an operational safeguard, finish extraction and verified durable feature
exports for every locked available arm before the first `develop`. That allows
environment loss to be handled before labelled comparisons. Later external
features still require the environment/versioning plan above.

```sh
python fm_extract.py extract --root /content/Few-shot --prepared results/fm_v1/lock --spec /content/fm_v1/specs/DINOv2.json --assets /content/assets/DINOv2 --out /content/fm_v1/features/DINOv2
python fm_evaluate.py develop --lock results/fm_v1/lock --cache /content/fm_v1/features/DINOv2 --arm DINOv2 --out /content/fm_v1/runs/DINOv2/development
python fm_evaluate.py nominate --lock results/fm_v1/lock --runs /content/fm_v1/runs --out /content/fm_v1/reports/nomination
python fm_evaluate.py report --lock results/fm_v1/lock --cache /content/fm_v1/features/DINOv2 --arm DINOv2 --out /content/fm_v1/runs/DINOv2/reporting --nomination /content/fm_v1/reports/nomination
python fm_evaluate.py summarize --lock results/fm_v1/lock --runs /content/fm_v1/runs --nomination /content/fm_v1/reports/nomination --out /content/fm_v1/reports/legacy
```

The CPU examples require the lock's exact Python/NumPy/SciPy environment; the
proposed Colab dependency pins differ from the current laptop math versions.
For the laptop-lock command shown above, run the CPU commands on that laptop
after manual cache export, using `D:/Few-shot/.venv/Scripts/python.exe`, local
lock/cache paths and an output root outside the checkout. Do not silently change
the evaluation environment to make the examples run.
The examples show individual lifecycle commands, not a script to run unchanged
from top to bottom. Complete all available-arm extraction/exports before any
development, and all intended development before deliberate nomination.
Do not execute score-producing commands as part of code prep or before S2.
Run `develop` for all available arms before making the deliberate nomination
decision. Never change the nominee after legacy/external outcomes. Inception is
not eligible; fewer than two complete FMs gives no comparative winner.

Use one designated study lock folder. The first nomination creates
`nomination_binding.json` **beside the input lock**,
under one study writer lock. It seals the full original decision and its canonical
hash against that input-lock SHA256. Calling `nominate` with a different output
folder later copies the same decision; it does not recompute a winner when a
late arm finishes. A no-winner feasibility decision is immutable too. Reporting
and summaries verify this binding before opening query features or metrics.
A restored/copied S2 lock lacking the original S3 binding can compute a new
decision: the seal is local to the chosen lock folder, not a registry of copies.
Do not use that capability to reselect after outcomes.

After first nomination and before the first legacy `report`, export the nomination
folder, verify the copied archive bytes and record the original binding hash
outside the runtime. Nomination and summary exports include copies of both
`nomination.json` and `nomination_binding.json`. Preserve the original S3 binding beside a restored S2
lock before reporting. Missing bindings stop; the code never automatically adopts
an unbound decision during reporting. An explicit `nominate` on an unbound copy
can recompute selection, which is forbidden for recovery after outcomes. This
protects the recorded study lifecycle, not intentional manual deletion/forgery of study
state. A pre-S3 backup is not a complete backup of later work.

All required prototype cells must be valid for an arm's nomination score. Pool
out-of-fold correctness within each class first. Failed logistic fits save the
actual termination reason and no query predictions; do not retune or replace
draws. `--omit-logistic-reason` records a disclosed reduction if time is short.
Paired summaries use shared completed cells with available/planned counts.
Means/SD/5th–95th percentiles are support-draw variability, not patient confidence
intervals. Cosine columns are scores; logistic probability columns are separate.

Kermany nine-way reporting uses one full-roster deterministic prototype head and
all nine output columns. NORMAL FRR, NORMAL recall, common confusions and rare
recall retain their actual denominators. The historical common-patient overlap
remains a limitation. No new patient-independence or causal source claim follows.

## Resume, export and reversal

Outputs are immutable for a fixed identity. Completion markers are written last
and must cover every completed child operation, its predictions and its own
marker, as well as root summaries/costs. A thin parent marker cannot authorize
an incomplete export. Interrupted operations can
reuse only verified receipts under unchanged inputs/code/environment/hardware. A changed
identity stops rather than overwriting; retain the prior version and use a fresh
output folder only for an author-approved change. Do not use directory deletion
to bypass a mismatch. Unavailable-at-lock arms cannot enter v1 later.

`writer.lock` contains host/PID/token. After verifying its owner process has stopped,
remove only that lock with `fm_prepare.py unlock --root ROOT --out FOLDER --token
SAVED_TOKEN`. A wrong token cannot remove a replacement writer's lock. Do not kill
unrelated processes. All raw-data paths, including linked raw folders and export
temporary/sidecar paths, remain protected.

```sh
python fm_prepare.py export --root /content/Few-shot --out /content/fm_v1/features/DINOv2 --archive /content/DINOv2-features.zip
python fm_prepare.py verify-copy --root /content/Few-shot --out /content/fm_v1/features/DINOv2 --archive COPIED_ARCHIVE.zip --sha256 EXPECTED_SHA256
```

Download the ZIP and receipt manually to durable storage. Verify the copied bytes
before trusting the backup; Colab Files are volatile. Weights, tokens, raw images
and publisher PDFs do not enter repository commits or generic result exports.

## Primary implementation sources opened during prep

- [RETFound source and global pooling](https://github.com/rmaphoh/RETFound_MAE/blob/ae9a9ecf37857cf47b8aa9f87cd6f710d75db287/models_vit.py)
  and [native evaluation transforms](https://github.com/rmaphoh/RETFound_MAE/blob/ae9a9ecf37857cf47b8aa9f87cd6f710d75db287/util/datasets.py).
- [DINOv2 processor configuration](https://huggingface.co/facebook/dinov2-base/blob/f9e44c814b77203eaa57a6bdbbd535f21ede1415/preprocessor_config.json).
- [MIRAGE lightweight HF encoder](https://github.com/j-morano/MIRAGE/blob/ee5eae8d3524ec791e719ced1f3cca27eae4c1b0/hf/mirage_hf.py)
  and [classification wrapper normalization/pooling](https://github.com/j-morano/MIRAGE/blob/ee5eae8d3524ec791e719ced1f3cca27eae4c1b0/mirage_wrapper.py).
- [BioMedCLIP configuration](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/9f341de24bfb00180f1b847274256e9b65a3a32e/open_clip_config.json)
  and [OpenCLIP 2.32.0 factory](https://github.com/mlfoundations/open_clip/blob/v2.32.0/src/open_clip/factory.py).
- [PyPI version metadata](https://pypi.org/pypi/torchvision/0.22.1/json) verifies
  the declared torch 2.7.1/torchvision0.22.1 pair; published metadata does not prove
  that every encoder loads in the intended runtime.
