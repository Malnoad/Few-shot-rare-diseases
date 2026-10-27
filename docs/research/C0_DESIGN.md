# C0 preprocessing control — fixed before implementation

Design fixed on 2026-10-27, before C0 feature extraction or head training.
No arms, thresholds, images, seeds or endpoints may change after C0 outcomes.
Lam runs training on Colab; laptop verification is training-free.

## Question and arms

Does standardizing file representation and framing reduce rare-class recall in
the frozen-Inception classifier? Source dataset and disease are confounded here:
this is a sensitivity control, not identification of an acquisition effect.

1. `current`: unchanged Stage 3 decoding, 256 then 299 resize.
2. `format`: decode RGB while discarding alpha (no compositing), convert to
   8-bit grayscale, encode every image as PNG. Preserve native dimensions.
3. `framing`: the same grayscale conversion; center on a black square with side
   max(width, height), then bilinear resize to 256 square, encode as grayscale PNG.
   Stage 3 still performs its usual 256 then 299 resize and three-channel decode.

Visual basis: inspected the first training scan and first two test scans per
class in manifest order (27 images). Rare scans include variable black bands;
Kermany scans include tilted white wedges, and retinal layers extend toward the
edges in both datasets. Pad rather than crop: there is no validated scan/lesion
segmentation, and deleting edge tissue would confound this control further.
Padding preserves the full field of view. It does NOT erase the original aspect
ratio from the pixels or remove existing black bands, white wedges, JPEG
artifacts, device texture, or disease. Fixed file dimensions alone do not prove
that all source information is removed.

`format` intentionally retains the width/height cue; only `framing` removes
native dimensions from the normalized headers. PNG re-encoding cannot reverse
past JPEG compression. The current TensorFlow loader already drops alpha and
decodes three channels, so a small format-only effect is plausible.

Normalize the existing raw, Stage 1 basic and accepted Stage 2 GAN pools equally.
Do not regenerate augmentations, retrain GANs, rerun QC, change accepted sets, or
move splits. This estimates a classifier-input intervention conditional on the
existing generators and QC. It does not test GAN training on normalized domains.
Stage 3 `basic` replaces rare real training images (not adds to them); preserve
that behavior and the existing rare-validation construction.

## Paired runs and endpoints

Full crossing: 3 arms × 3 variants (`none/basic/gan`) × seeds 1–20 = 180 heads.
Seeds 1–20 are shared with the C1 sweep in branch `task/c1-seed-sweep`, notebook
Part 4. C1's final sensitivity specification must retain these same seeds or
explicitly reconcile the designs before training; do not silently choose a subset.
Seed 42 published outputs are provenance only, never the paired comparator.
All runs use existing Stage 3 defaults: 250-epoch ceiling, patience 20, best
validation-loss weights, Adam 0.001, batch 10, GAN cap 3,000 per disease.
Use the same image order, GAN selections, seed, validation assignment and software
environment in each arm. Process seeds outside variants outside arms, so an
interruption leaves matched blocks. Resume only completed, provenance-matched runs.

Primary contrast: `current - framing`, `none`, mean paired change in rare macro
recall (five class recalls, equally weighted). Secondary: current-format,
format-framing, the same contrasts for basic/gan; report all without selecting
the most favorable result. Also report per-class recalls, correct /22, common
macro recall, overall accuracy and rare-vs-common errors for each seed.

Predeclared practical rule: flag preprocessing sensitivity (“the cue may matter”)
if primary mean recall loss is at least 0.10 and positive in at least 16/20 seeds,
while mean common macro recall loses at most 0.02. Report a paired Student-t
95% interval across seeds as training-variation context, not patient uncertainty.
The 0.10/0.02/16 thresholds are design tolerances, not paper-derived constants.
If both common and rare performance fall, interpret as general degradation.
If the rule fails, report no detected sensitivity to these interventions, not
absence of source confounding. There are only 22 rare test images (4–5/class);
seeds and augmented descendants are not independent patients. Do not tune on them.

## Training-free checks, fixed before outcomes

Header census of all selected images: encoding/mode and native dimensions;
repeat L7's rare-class nearest-training-median width/height rule (L1 distance,
ties choose the first class in Stage 3 order). Report ties. Constant dimensions
must give tied scores; any apparent correct labels then reflect only tie choice.
Require all framing copies to be 8-bit L-mode PNG, exactly 256×256. Require all
format copies to be 8-bit L-mode PNG, dimensions equal to their source.

Feature diagnostic: L7 axis = mean(real rare TRAIN features) minus mean(common
TRAIN features), score untouched TEST images, report rare-vs-common rank AUC.
For the bounded CPU check use the first 2 train and 2 test images per class in
manifest order (36 images/arm, three arms), identical identities between arms.
This small deterministic sample is a smoke diagnostic, not the full L7 estimate.
On Colab repeat the exact axis using all cached real train and test features.
No head fitting is involved. Recompute the axis in each arm; also report scores
against the current-arm axis to expose rotation of the separating direction.

Gate: headers must pass before extraction/training. Residual feature AUC ≥0.90
in framing means the tested direction still strongly separates sources/diseases;
report “residual separation”, never “cue removed”. Small-sample AUC cannot clear
this gate: use the full-cache diagnostic on Colab before heads. If full framing
AUC ≥0.90, the notebook stops before training pending Lam's scientific decision.
Do not invent a new crop or tune preprocessing after seeing that result.

## Isolation, resume, costs

Copies: `data/c0/v1/{format,framing}/data/...`, with derived manifests and an
immutable content inventory. Originals stay read-only. Current uses original
paths. Arm-specific caches and runs live only under `results/c0/v1/`, mirrored
to `MyDrive/octfsl/c0/v1/`. Never reuse the run-of-record feature caches: their
path-only validation cannot certify preprocessing or historical weights.
Normalized images stay on Colab local disk and can be deterministically rebuilt
after disconnect; do not duplicate tens of thousands of PNGs into Drive. Save
their manifests/inventories plus features, diagnostics, logs and heads to Drive.
Completed cache groups and heads are resume units; an interrupted group/head is
recomputed. Each saved cache must match input content and code provenance.
Reverse: select `current`; no destructive undo or raw-data restoration is needed.

Recommend A100/L4/T4 for fresh features, then CPU for cached heads if preferred.
There are 84,630 manifest images, 10,000 basic and 9,889 accepted GAN images in
the locally inspected pools (104,519 total). Three arm caches require about
1.3 GB for float16 2048-vectors plus paths; normalized PNGs need additional local
disk. Budget, unbenchmarked on Colab: feature extraction roughly 15–60 min/arm
on supported GPU versus 2–8 h/arm on CPU; conversion/Drive transfer adds time.
180 early-stopped heads can take many hours (roughly 6–24 h at 2–8 min/head,
not measured here). Print actual extraction throughput and first-head timing.
Do not use Blackwell/RTX PRO 6000 with the documented TensorFlow 2.20 setup.

## Proposed text for shared docs (not edited by T6)

README / CONTEXT: “C0 is a pending paired preprocessing sensitivity control:
current inputs, uniform grayscale PNG, and uniform grayscale PNG with square
padding, across none/basic/gan and seeds 1–20. It preserves committed splits,
existing augmentations and accepted GANs. Source and disease remain confounded;
standard headers do not establish removal of acquisition information. C1 input
selection waits for C0.”

REPLICATION: “C0: code and isolated Colab workflow prepared; no head training
performed locally. Full feature-axis diagnostics precede training. Report the
predeclared current-versus-framing rare macro-recall contrast, common-class
control, all secondary contrasts and residual source separation before choosing
C1 inputs.”
