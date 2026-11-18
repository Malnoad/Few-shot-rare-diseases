# G1 adversarial review (Fable 5.2, 18 Nov 2026, read-only)

Paths: `fm_posthoc.py`, `fm_pad_extract.py`, `fm_extract.py`, `fm_core.py`, `protocols.md` are in the worktree root. `padding.csv` is `D:/Few-shot/drive/fm_v1/final/final-20261113T170804Z/g1/unzipped/MIRAGE/padding.csv` (geometry identical in the Inception copy; I compared all 5,927 rows). No number below comes from model features.

## 1. Integrity: OK

- **No mixing.** `score_features` builds prototypes and queries from the same feature object (fm_posthoc.py:959-960, 964-973). `g1` calls it once on locked and once on padded features (fm_posthoc.py:1042). `nine_way_cells` does the same for P9 (fm_posthoc.py:1058-1066, 1081-1083). Padded column = padded prototypes and padded queries.
- **Alignment.** `load_features` rejects any path or hash order that differs from the locked inventory (fm_extract.py:201-211). Lock and arm identity are checked (fm_posthoc.py:1015).
- **Code identity.** `g1_script_sha256` in both identity.json files is `496cc1a8…`, equal to `git show 39c3d73:fm_pad_extract.py`.
- **Cross-check.** The two code paths agree: 306/332 = 0.922 and 198/206 = 0.961 (G1_comparison.md:73, 68, 95, 92).
- **Label.** The G1 row is unchanged since `1eb3964` (empty diff on that row), so the G1 label is right. P9-on-padded is ruled by `39c3d73` (protocols.md:222), which is on origin.
- **Wording to fix.** "Prototypes from the locked fitting roster" (G1_comparison.md:59) is true of the roster only. Say: "padded column: prototypes and queries both from padded features". "At the locked argmax" (G1_comparison.md:88) is also misleading for the padded column.
- **Control gap.** The control is the first 20 inventory rows, all Kermany test CNV (fm_pad_extract.py:131; padding.csv). The 1,032 squares are 1,018 Kermany and 14 rare. No external image has a control: none is square. For Inception the padded path (`keras_decode` + `keras_batch`, fm_extract.py:312-324) differs from the locked path (`keras_path`, fm_extract.py:318-320). It is verified on Kermany L-mode files only, not on OCTID/OCTDL RGB files. Risk is low: both arms, with different frameworks, flip the same way.

## 2. Is the external flip real? Yes as a computation; the proposed mechanism is wrong

Padding fraction per set (share of the square that is black), from padding.csv:

| Set | n | median | range |
|---|---|---|---|
| Rare, all 146 | 146 | under 0.02 | 0 to 0.352; 95 under 0.05; 6 at 0.25 or more |
| Kermany fitting NORMAL | 1,000 | 0.031 | 33% at 0.25 or more |
| Kermany fitting CNV / DME / DRUSEN | 1,000 each | 0.031 / 0 / 0.031 | 46% / 10% / 37% at 0.25 or more |
| Kermany query NORMAL | 250 | 0.031 | 6% at 0.25 or more |
| OCTID (all three) | 410 | 0.333 | all exactly 0.333 |
| OCTDL NORMAL / MH | 332 / 39 | 0.637 / 0.635 | 0.475 to 0.686 |

- **Rare images get almost no bars.** They are near-square web crops. The brief's mechanism ("rare also get bars, so bars point to rare") is contradicted.
- **Kermany gets more bars than rare.** CNV fitting has 128 images at 1536x496 (0.677, OCTDL-like). Yet padded externals go to rare, not CNV.
- **Bar size cannot explain P9 inside OCTID.** All 410 OCTID images have identical padding. MIRAGE MH against NORMAL is 0.090 (G1_comparison.md:83): an inverted ranking, not a random one. Hypothesis [unverified]: padded normals lose the NORMAL prototype, while diseased scans keep some CNV/DME similarity, which lowers their rare-minus-common score.
- **Why rare?** [unverified]. Candidates: in-image dark margins or a small retina in the web crops; rare prototypes acting as "not Kermany" attractors. 
- **Diagnostics that need a committed, pushed rule first** (model outputs): nine-way destinations of Kermany test images split by padding fraction (768x496 gives 0.354, close to OCTID, within one source); an unpadded external control through the padded code path.
- **Diagnostic with no model output:** share of near-black rows and columns inside the 146 rare images.

## 3. What G1 supports

- **Development and hold-out are a near-null test.** 122 of 146 rare images get under 10% padding. "Barely moves" (G1_comparison.md:32, 45) is expected by construction. It is not evidence that squashing is harmless, and it says little on Q2.
- **"Squashing caused MIRAGE's OCTDL collapse": not supported.** Padded NORMAL recall fell, 26 to 1 of 332 (G1_comparison.md:73).
- **Not refuted either.** Padding swaps a 2.75x squash for about 64% black area. Neither matches the fitting images. Source and bar size are fully confounded: every external image has at least 0.333, rare prototypes almost none.
- **Placement.** Supplement, under the dated post hoc heading (protocols.md:199-202). The main text keeps the locked 26 of 332 with no cause, as decided (protocols.md:339-341). No padded number replaces a locked one (protocols.md:224).
- **Rule coverage.** The G1 rule lists "OCTDL normal destinations" only (protocols.md:224). Kermany, legacy 22, OCTID and OCTDL MH destination rows go beyond that list. They use the ruled roster and must be released (protocols.md:203), but say so.

## 4. Other problems

- **K=16 hold-out** (+0.036, +0.67 image; G1_comparison.md:44, 55). Opposite sign to K=1-8, under one image of 22, and outside the K mean (fm_core.py:27-28). No change has an interval. Do not read it.
- **Page lacks the geometry table.** Without it, section 4 reads as a model failure, not a confounded input change.
- **Environment.** The page shows `False` for MIRAGE (G1_comparison.md:18) without naming the two packages or the GPU change. I did not verify the filelock/regex diff or the L4 claim [unverified]; identity.json shows A100. The controls (1.4e-7) make it harmless, but state it.
- **Unreleased.** The four outputs are untracked. Local commits 36dad48, dcbb323 and 3b0212b are not on origin (origin head is 90fd4fe), so the scoring code is unpushed.
- **Chronology.** The change log says "18 Nov, evening: no G1 number exists" (protocols.md:325), committed 03:13-03:59. The Colab stamp is 07:02Z (14:02 Bangkok), ahead of the 19-21 Nov calendar (protocols.md:244). Add a dated entry.
- **Write-once** holds (fm_prepare.py:203-211). The date is in the label, so a rerun on another day stops rather than overwrites.

## Verdict

**Integrity: OK.** No mixing, no alignment bug, labels correct. Fix two phrases, add the geometry table, note the control gap.

**Allowed claims**
1. With black padding to a square, development and hold-out macro recall changed by at most 0.023 at K = 1-8 for both models.
2. The rare images are near square (122 of 146 under 10% padding), so this check barely changes their input.
3. Padding did not restore normal recall on OCTDL or OCTID; it lowered it (MIRAGE 26 to 1 of 332; 202 to 0 of 206).
4. Every external image received at least 33% black area and the rare examples almost none, so source and padding are confounded in the nine-way audit.
5. Both nine-way results depend strongly on input geometry.

**Forbidden claims**
1. "Squashing caused (or did not cause) MIRAGE's OCTDL collapse."
2. "The models are robust to aspect ratio" from the development or hold-out rows.
3. Any padded number in place of a locked one, any reading of the K=16 row, or any MIRAGE-against-Inception ranking from padded P9.

**Proposed supplement wording**
"As a post hoc sensitivity check (rule fixed 16 Nov 2026, before any number), we padded every image with black to a square before each model's own transform, for MIRAGE and Inception-v3. Five-way results on the rare images changed little (mean macro recall over K = 1-8: development 0.650 to 0.646 and 0.534 to 0.525), which is expected because these images are already near square. In the nine-way audit, padded OCTID and OCTDL normals, which all received 33-69% black area, were almost all assigned to rare classes (MIRAGE: 0 of 206 and 1 of 332 called normal); because padding amount and image source are confounded, this shows that the audit depends on input geometry and does not identify the cause of the locked external errors."
