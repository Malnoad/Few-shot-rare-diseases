# G1 review (Opus 5.5, read-only, 18 Nov)

Abbreviations: G1c = `results/fm_v1/posthoc/G1_comparison.md`; prot = `protocols.md`; AM = `docs/research/FM_AMENDMENTS.md`; pad = `g1/unzipped/MIRAGE/padding.csv` (my tally of width/height only; no model output used).

## 0. Run validity
- Control, square-image and repeat checks all pass (<=1.43e-6, cosine 1.000) (G1c:15-21).
- MIRAGE `environment_matches_lock False` (G1c:18) is harmless: unpadded control matches at 1.38e-7 (G1c:15).
- Prototypes and queries are both padded. `score_features` and `nine_way_cells` build centroids from the padded fitting roster and score padded queries (fm_posthoc.py:959-973, 1058-1066; called with padded features at 1042, 1081-1083). No mixed-pipeline bug.

## 1. What G1 answered
**(a) OCTDL normal collapse.** Padding did not fix it. It got worse: 26 -> 1 NORMAL, 276 -> 25 common disease, 30 -> 306 rare (G1c:73). G1 does not show that squashing caused it. It also does not show squashing is innocent: OCTDL is about 2.75:1 (pad: median w/h 2.75, range 1.90-3.18), and both pipelines are far from the training geometry. Keep "no claim about the cause" (prot:340-341).

**(b) Development and hold-out.** Insensitive. MIRAGE mean K1-8 0.650 -> 0.646; hold-out 0.722 -> 0.707; Inception 0.534 -> 0.525 and 0.524 -> 0.526 (G1c:32, 45). Arm order and the size-only reference (0.569/0.618) are unchanged. This is expected from geometry: Yoo rare images are near-square (median w/h 1.00-1.11 per class) and Kermany is mostly 512x496 (1.03). So padding barely changes these inputs. It is not evidence against the size shortcut. Padding removes distortion, not size (prot:184).

**(c) P9.** Legacy 22 vs Kermany stays ~1.000 (G1c:86). OCTID collapses: MIRAGE CSC 0.878 -> 0.305, MH 0.684 -> 0.090, both 0.781 -> 0.198; Inception to ~0.50-0.53 (G1c:82-84). OCTDL barely moves (MIRAGE 0.357 -> 0.320; Inception 0.634 -> 0.656) (G1c:85). The locked OCTID AUROC is therefore not robust to the input pipeline.

## 2. Why the external flip
Geometry (pad; fm_pad_extract.py:40-44):

| Set | w/h | padded share black |
|---|---|---|
| Yoo rare, 5 classes | median 1.00-1.11 | median 0.5-10% |
| Kermany test (1000) | 1.03 (235/250 normals) | median 3% |
| OCTID NORMAL/CSC/MH | 1.50, all | 33% |
| OCTDL NO/VID | ~2.75 | ~64% |

- The simple story "external scans get bars, prototypes don't" is only partly right. The padded NORMAL prototype includes 229 Kermany images at 768x496 (1.55, ~35% black) and 98 at >=2 aspect, out of 1,000 (input-lock `nine_way_roster.csv`, my tally). So padded OCTID (33% black) has the same bar share as about a quarter of the NORMAL fitting images. Yet 206/206 go rare (G1c:67).
- A more likely reading [interpretation, unverified]: the four common centroids are tight, Kermany-specific directions (1,000 images each). The rare centroids average 19-27 heterogeneous web crops. A query that moves off the Kermany manifold lands nearer the diffuse rare centroids. "Rare" then works as a "not Kermany-like" bin. This fits C0 (rare-vs-common AUC ~0.995; `docs/report/fm_ieee/CLAIM_SOURCE_LEDGER.md:13`) and the size-only rule.
- Two pipeline details may add to it [unverified]. MIRAGE min-max normalises per image (fm_extract.py:223-228), so a black bar can reset the minimum. Inception maps black to -1 (fm_extract.py:100).
- Within OCTID all three classes share 1.50 aspect (pad). So the bars cannot by themselves create a class difference. The P9 inversion means the margin then tracks something other than disease.
- The external "gains" (MIRAGE OCTID MH correct 10 -> 54; Inception OCTDL MH 0 -> 15; G1c:71, 76) come from everything shifting to rare. They are not better recognition. Rare-called rates for normals reach 0.92-1.00 (G1c:92-95).
- Note: the locked squash is MIRAGE's own published input path (AM:71, A-11). Padding is off-recipe. G1 measures sensitivity, not a "correct" pipeline.

## 3. Meaning for the paper and project
- It strengthens the source/geometry story. A preprocessing change that leaves Kermany and Yoo results within 0.04 (G1c:28-45) flips external normal recall from 98% to 0% (G1c:67). The nine-way head's external behaviour is governed by input geometry and source, not by the disease the head was built for.
- Wording on OCTDL: keep the locked count and "no claim about the cause" (prot:340-341). Add one fact only: padding did not restore it (1/332).
- The P9 wording for OCTDL (prot:347-348) holds under both pipelines. The OCTID P9 values need a caveat: they depend on the resize pipeline.
- A1 rule 4 already says shortcuts are not removed (AM:41). G1 is a concrete case.
- B11 (aspect ~1-1.2 datasets) [B11 text not found in repo; unverified]: G1 supports it. In this study, wide external sets cannot be fed to square-input models without a choice that changes the answer. For v2: match aspect and scale between supports and queries. Or report both pipelines as a pre-specified sensitivity. The next-phase plan already asks for data where size does not predict the class (prot:271-272, 349).

## 4. Placement and draft wording
G1 is post hoc (prot:224). Rule 2: it never replaces an A3 main-text result; it goes under the dated post hoc heading; supplement unless Lam and the mentor decide otherwise (prot:199-202). Release regardless (prot:203; AM:46).

**Supplement paragraph.** "Post hoc pad-to-square check (G1; rule fixed 16 Nov before any G1 number). MIRAGE and Inception-v3 resize every image to a square, which stretches wide scans (OCTID width/height 1.50, OCTDL about 2.75; Kermany and the Yoo images mostly 1.0-1.1). We re-extracted both arms with every image padded with black to a square before the locked transform and scored the locked episodes and nine-way roster. Unpadded controls matched the locked features. Development and hold-out recall changed by at most 0.04 (MIRAGE mean over K 0.650 to 0.646; hold-out 0.722 to 0.707), and the order of arms and the size-only reference were unchanged. External results changed sharply: MIRAGE called 0 of 206 OCTID and 1 of 332 OCTDL normal scans NORMAL (202 and 26 locked), assigning almost all to rare classes; Inception-v3 behaved alike (182 to 8; 66 to 13). The OCTID rare-against-normal AUROC fell from 0.878 to 0.305 (MIRAGE, CSC). Higher external rare counts under padding reflect this shift, not better recognition. The external nine-way results therefore depend on how images are fitted to a square. Padding did not restore OCTDL normals; we make no claim about the cause. This sensitivity check lies outside A2 A-11 and replaces no locked result."

**Main-text sentence** (justified, because it qualifies a main-text secondary result; Lam decides, labelled post hoc): "A post hoc check that padded images to a square instead of resizing them left development and hold-out recall nearly unchanged but sent almost all OCTID and OCTDL normal scans to rare classes (Supplement), so the external nine-way results depend on input geometry."

## Key findings
1. G1 is technically clean; prototypes and queries are both padded.
2. Development and hold-out do not depend on squashing, because those images are near-square.
3. Padding does not rescue OCTDL normals; it sends 92-100% of external normals to rare.
4. The locked OCTID P9 AUROCs are pipeline-dependent; OCTDL and legacy are not.
5. The result supports "rare = not-Kermany-like" as a shortcut reading.

**Claim:** sensitivity of external results to input geometry; stability of development and hold-out.
**Do not claim:** squashing caused (or did not cause) the OCTDL collapse; padding is better or worse; padded external "gains"; any statement about what MIRAGE's features encode.

## Proposed follow-up (candidate rule P11; needs a committed and pushed rule first, prot:196; laptop only, under an hour, fits before 21 Nov)
"Same-source bar test. Kermany test images (1,000; locked roster), split by geometry from `padding.csv`: 512x496 versus 768x496 (CNV 68, DME 109, DRUSEN 38, NORMAL 15) versus other. For MIRAGE and Inception-v3, locked and padded: nine-way destination counts (correct, NORMAL, common disease, any rare) per class and geometry group, with Clopper-Pearson bounds. Patient overlap with fitting is labelled as in 'Kermany all test'. Descriptive only." It asks whether ~35% black bars alone push same-source scans to rare. If the 768x496 scans stay correct when padded, the flip is about source, not bars.
