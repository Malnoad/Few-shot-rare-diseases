# L7: strong real-only rare-class baseline

Audit date: 2026-10-23. **Bounded audit complete; a unique cause is unresolved.**
This is exploratory artifact analysis: the aggregate outcomes were already known.
Lam remains the research author. No training, feature extraction, split changes,
accepted-image changes or published-result overwrites were performed.

## Observation and primary-source verification

The original paper's Table 4 implies **3/22** correct rare test images for its
real-only model; our saved seed-42 predictions give **17/22**. Class order matters:
the paper places RP before Stargardt; this repo uses the reverse order.

| Quantity | CSC | MH | MacTel | Stargardt | RP | Total |
|---|---:|---:|---:|---:|---:|---:|
| Paper initial rare training images | 25 | 25 | 20 | 15 | 12 | 97 |
| Our real training images | 27 | 26 | 25 | 19 | 27 | 124 |
| Paper real-only correct / test images | 1/5 | 1/5 | 0/4 | 1/4 | 0/4 | 3/22 |
| Our seed-42 real-only correct / test images | 4/5 | 4/5 | 3/4 | 3/4 | 3/4 | 17/22 |

Source: Yoo, Choi and Kim (2021), [dataset methods and Table 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC7829497/),
DOI [10.1007/s11517-021-02321-1](https://doi.org/10.1007/s11517-021-02321-1).
The paper specifies a frozen convolutional backbone, 250 classifier epochs and a
one-tenth validation fraction. Its augmented rare pool is 2,000 basic plus 3,000
GAN images; it does not require a separate additional real-image pool there.

The primary JATS XML was retrieved from
<https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7829497/fullTextXML> after the PMC
HTML endpoint presented a CAPTCHA. SHA-256:
`4b4a2e5833422421c503b2bb4dc22fe8ad86b360ff7de71600172a7953d6a883`.
Local copy: `.agents/reports/l7-audit/paper.xml` (gitignored). This was a targeted
check of the original methods and Tables 3-4, not a systematic literature review.

## Rival explanations and discriminating checks

Each row is a candidate explanation, not an established cause. Observing a
protocol difference does not measure its effect on the result.

| Candidate | Check and observation | What the evidence permits |
|---|---|---|
| Wrong labels, class order or exported head | Cache paths and CSV labels match the manifest. Every saved head reproduces all 1,022 predicted labels; maximum probability error is below 0.000002. All three dense kernels/biases exactly match their local H5 checkpoints. | Contradicted for these saved artifacts. Historical training provenance remains a separate question. |
| Exact rare train/test image duplication | SHA-256 of files and decoded RGB arrays finds no duplicate groups among all 146 rare images. No exact quantized feature matches connect any of the 22 rare test images to the 73,958 common or 124 rare real training rows. | No evidence of exact duplication in these checks. This does not exclude crops, near duplicates, common-source images or shared patients. Raw common images were not exhaustively pixel-hashed. |
| Same training/test conditions as the paper | Our rare training total is 124 versus 97. Original test-image identities are not established by matching test counts. | A confirmed comparability difference. More examples or a different/easier holdout could contribute; the effect cannot be isolated from existing results. Do not alter the committed split to chase the paper's number. |
| Checkpoint/training schedule differences | Histories contain 26/27/22 epochs for none/basic/gan, with minimum validation loss at epochs 6/7/2. Stage 3 restores that checkpoint. None has zero rare validation images, versus 1,000/1,987 for basic/gan. | Confirmed differences in training duration and validation composition. We lack final-epoch/250-epoch predictions, so their effects are unresolved. |
| Backbone fine-tuning explains the discrepancy | Original methods and current Stage 3 both freeze the convolutional backbone and train the final dense layer. | Full-backbone fine-tuning is not a supported explanation of this paper/repo difference. Numerical equivalence of historical feature extraction is still unverified. |
| Stale or differently extracted cached features | Caches contain only paths and float16 features. The validity check compares paths, without input-image hashes, backbone/weight fingerprints, preprocessing identity or extraction environment. The current loader uses 256→299 resizing and [-1,1] scaling. | Path/order consistency passes. Historical extraction provenance cannot be reconstructed from these caches alone; no claim that they are wrong is justified. No features were regenerated. |
| Acquisition/source cues or correlated cases | Raw rare images have class-associated dimension patterns and are all four-channel PNGs. Resized-thumbnail similarity does not establish patient identity. Rare `pid` values are class plus filename, not verified patient IDs. | Source/case dependence remains plausible and unresolved. Resizing alone cannot guarantee removal of all source cues. Patient/source metadata or controlled image interventions would be needed to distinguish these from disease morphology. |
| One fortunate classifier seed | Existing none seed results are 15, 16, 17, 12 and 11 rare images correct (50.0–77.3%). | The strong baseline persists descriptively across these five runs on the same holdout, although its size varies. Seeds are not five independent test cohorts. |

Code pointers: `stage0_manifest.py:67` creates rare image IDs;
`stage3_classifier.py:61` loads/resizes images; `:75` caches features; `:153`
builds the real-only pool; `:167` constructs augmented pools; `:187` selects
the best validation checkpoint. These refer to the audited checkout at
`5c77ce22c5b7eda743321e610acad0ebf6690c1a`.

## Similarity-screen limits

The diagnostic script compares existing features and grayscale 64×64 thumbnails;
it fits no model and extracts no new backbone features. Pixel Pearson nearest
matches range from 0.644 to 0.923. These values have no validated duplicate
threshold here. The four highest-correlation pairs were visually inspected;
that screening does not establish patient independence or exclude transformed
duplicates. Pair records remain available for review.

For the 22 rare test images, nearest-neighbor cosine matching is correct for
18 when restricted to the rare training cache, and 13 when all real training
examples compete. Pixel-thumbnail correlation matching is correct for 11.
These exploratory screens show structure in the data; they cannot attribute it
to disease, source or patient identity, and they are not replacement baselines.

## Conclusion and smallest follow-up

The saved baseline is internally consistent; a simple export/label bug or exact
rare-image duplication does not explain the gap. We have confirmed differences
in training counts, validation composition and checkpoint schedule, while original
split identity, feature provenance and patient/source dependence remain unresolved.
There is no defensible single-cause finding from the existing artifacts.

**Next training-free task: L8, verify the RCI definition and implementation.**
Before future classifier training, C1 should propose a protocol that distinguishes
the current early-stopped result from a fixed-250-epoch sensitivity condition,
using isolated outputs and preserving both best and final heads. This should be
reviewed with Lam; no new experiment was run or preregistered by this audit.

Keep the present manifest intact. Any later paper-count-matched training subset,
source-held-out split or new raw-image preprocessing is a separately named
experiment requiring a research-design decision. Do not silently replace the
existing run of record.

## Reproduction and verification

Local diagnostic command (no training):

```powershell
& D:/Few-shot/.venv/Scripts/python.exe -X utf8 -B .agents/reports/l7-audit/audit_l7.py
```

The script and its machine-readable evidence are local/gitignored. Outputs:
`evidence.json`, `rare-image-metadata.csv`, `rare-nearest-train.csv`,
`rare-test-predictions.csv`, and `similarity-screen.html` in that same folder.
Input hashes cover 186 files, including every rare raw image, every existing
feature cache, committed predictions/heads/histories, the manifest and protected
evaluation CSVs. All hashes match after the diagnostics.

Pre-completion surfaces: no CLI/notebook, action/reverse, data/schema/cache-list
or research-method changes. Documentation records audit findings only. Run
`.agents/check` before handing back; its raw-data branch is unavailable in this
worktree, while raw rare images were read from the main checkout for this audit.

Workflow assistance: the hypothesis-generation skill organized the rival checks;
it did not establish the findings.
