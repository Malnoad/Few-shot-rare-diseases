# G1 rebuttal (Fable 5.2, high effort, 18 Nov 2026, read-only)

Abbreviations: V = `g1_verdict_opus_main.md`; G1c = `results/fm_v1/posthoc/G1_comparison.md`; prot = `protocols.md`; AM = `docs/research/FM_AMENDMENTS.md`; pad = `g1/unzipped/MIRAGE/padding.csv`; LG = the locked `reports/geometry/MIRAGE/geometry_summary.csv` inside `evaluation/results-20261114T093033Z.zip`; S-a, S-b, S-c = my scripts `geom_a.py`, `geom_b.py`, `geom_c.py` in `C:\Users\THINKB~1\AppData\Local\Temp\claude\c38f92eaa1d8ed7a\34630c3a-5fe8-47a5-a7e3-d2b9d1ac6536\scratchpad\fable_reb\scripts\`. No model feature or prediction was read. Pixel tallies read the raw files only.

## Headline

**The rare images are already letterboxed in black. G1's black padding gives wide external scans the rare images' own layout.** V point 2 dropped the right mechanism for the wrong reason.

- All 146 rare images are opaque RGBA. 122 have black rows both above and below the scan. 126 have at least 25% of their height in flat rows. The flat border is grey 0 in all 141 images that have one (S-b, S-c, on `data/raw/rare`).
- Locked, pre-specified confirmation: border share median 0.34 to 0.56 per rare class; median 0 for Kermany, OCTID and OCTDL (LG; rule at AM:130-135).
- Recorded before the lock: "every rare image is an RGBA PNG crop with black border bands" (`docs/research/fm_independent_review/FM_INDEPENDENT_REVIEW.md:88-92`).
- Retina strip height as a share of the model frame: rare median 0.58 (quartiles 0.47 to 0.66); padded OCTID 0.67; padded OCTDL about 0.36; Kermany 512x496 0.97 (S-c; pad).
- pad records only what G1 added. "122 of 146 under 10% padding" (V:6) does not mean "no bars".

## A. Point by point

1. **Integrity: agree.** Re-checked: script hash `496cc1a8…` equals `39c3d73:fm_pad_extract.py`; environment gap is filelock 4.2.10 to 4.2.12 and regex 2026.11.9 to 2026.11.18 (`identity.json` against `input_lock.json`).
2. **Mechanism: disagree.**
   - "Not Kermany-like bin" contradicts the locked column. Unpadded OCTID and OCTDL normals are not Kermany, yet few go rare: 4 of 206 and 30 of 332 (MIRAGE), 15 and 11 (Inception) (G1c:67-68, 73-74). Rare wins only once bars are added.
   - "Tight against spread prototypes" has no measurement in any file [unverified]. Prototypes are unit-normalised (fm_core.py:194-196).
   - Leading reading [unverified at model level]: black letterbox layout plus non-Kermany appearance. Bars alone are not enough: exactly 3 of 169 patient-disjoint Kermany normals leave NORMAL when padded (MIRAGE, G1c:63); 9 of those 169 are 768x496 (S-a); so at least 6 of 9 stay. This is a bound from released counts, not a new score.
   - "About a quarter" is a third: 327 of 1,000 NORMAL fitting images get 30% or more (S-a).
3. **Near-null: agree.** Amend the reason: the files are near square because they are letterboxed; the retina inside is a strip of median width/height 1.74 (S-b).
4. **Squashing: agree.** Add: G1 cannot isolate squashing, because its fill colour is the rare images' canvas colour. The earlier line "black bars are also a new input the models never saw" (`results/fm_v1/posthoc/pre_rule/2026-11-18/fable_a2_a4.md:60`) is wrong.
5. **P9 caveat: amend; it overreaches.** The locked values use each model's published input (AM:71-76), and no OCTID file has bars (LG). G1 shows the padded variant enters the letterbox regime (206 of 206 normals rare, G1c:67). It does not show 0.878 is an artefact. Say: "under the published input path; with black bars added, the same score ranks OCTID the other way". "OCTDL does not depend" is too strong: its AUROC holds, but normals called rare go 0.090 to 0.922 (G1c:95). The caveat that matters sits on the main-text false-rare rate (AM:187): it holds for frame-filling inputs.
6. **Placement: agree on supplement; amend the main-text sentence.** Rule 2 keeps post hoc results under one dated heading; other placement is Lam's, with the mentor (prot:199-202). The 18 Nov decision allows only the locked count in the main text (prot:339-341). So a G1 sentence there needs a new decision. Better: cite the locked geometry table, which A3 already allows in the main text (AM:189), and point to G1.
7. **Claims: mostly agree.** "At most 0.023 at K = 1-8" is correct for both arms and both phases: largest is -0.0228 (MIRAGE hold-out, K = 2), next -0.018 (`G1_pad_to_square.csv`; G1c:41). Add "mean of 100 draws". Show K = 16 (+0.036, G1c:44). Add allowed: the letterbox facts. Add forbidden: "rare is a not-Kermany bin"; "source, not bars"; "bars caused it".
8. **G1b: disagree** (B3, B4).
9. **Records: amend.**
   - `g1-report` cannot rewrite the page: `write_once` refuses different bytes (fm_posthoc.py:1194; fm_prepare.py:203-207). The page must be deleted, and it is untracked. Commit the four outputs as generated first; change wording and regenerate in a second commit.
   - Numbers cannot change: `g1_report` reads only the CSVs, label included (fm_posthoc.py:1109-1113). Check the hashes stay `008e5f56…`, `f641b601…`, `ea7d7127…`. Keep the geometry table out of `g1_report`; it would need pad from Drive. `tests/test_fm_posthoc.py:241` must follow.
   - Rule 1 covers the rule text (prot:196-198). The G1 row (`1eb3964`) and padded P9 (`39c3d73`) are on origin (`90fd4fe`); `rule_commit` enforces it (fm_posthoc.py:808-818). Scoring code `36dad48` was committed 03:13, 11 hours before the 14:02 run and 14:33 scoring, but is unpushed. No breach. Disclose that the choices beyond the rule (extra destination rows) carry a local timestamp only.
   - Also record: the "evening" and "night" entries (prot:325, 334) were committed 03:13 to 03:59; the run preceded the audit, against "audit and preparation first, Colab later" (prot:335).

## B. What both reviews and the verdict missed

1. The letterbox (headline).
2. The locked column refutes the "not-Kermany bin" (A2).
3. G1 is already the within-source bar test for external scans. Same OCTID files: no bars, 202 of 206 NORMAL; bars, 0 of 206 (G1c:67). "Source, not bars" (V:18) is a false choice: source is constant across G1's two columns.
4. G1b confounds (S-a; the counts CNV 68, DME 109, DRUSEN 38, NORMAL 15 are right):
   - Same patient and same size in fitting: CNV 37 of 68, DRUSEN 25 of 38, DME 15 of 109, NORMAL 1 of 15. Patient-disjoint wide images: 17, 8, 51, 9.
   - Padded common prototypes hold barred images in unequal shares: CNV 46%, DRUSEN 37%, NORMAL 33%, DME 10%. "Stays correct" then follows prototype bar share, not disease.
   - Pad then resize couples bars with scale (strip share = 1 minus bar share). No laptop test separates them.
   - The NORMAL answer is already bounded (A2).
5. The min-max rival is mostly closed by pixels: MIRAGE's minimum before rescaling is already 0 in 1,000 of 1,000 Kermany test and 142 of 146 rare images (S-b). External files are not on the laptop [unverified]. Inception has no min-max and flips alike (G1c:68, 74).
6. Separable by geometry alone: G1-added against in-file bars; min-max. Needing a rule: prototype spread; bars against appearance. Needing a GPU: bars against scale; fill colour.

## C. Corrected verdict

- **Integrity:** OK.
- **Mechanism:** state geometry only. "The rare examples carry black bars (median 39% of area); padding gives wide external scans that layout; we did not test whether the models use it" (A1 rule 4, AM:41).
- **Claims:** V's list with the changes in A5 and A7.
- **Placement:** supplement. Main text may cite the locked geometry table.
- **Follow-up: no G1b before 21 Nov.** Its question is wrongly posed, its strongest subset is 85 images, and it is a third outcome-chosen layer (OCTDL, then G1, then G1b; compare prot:285-288; cut-off prot:244). The decisive test is the mirror: remove the black rows from the rare images and re-extract. That needs a GPU and changes five-way inputs, so it belongs to the next phase (prot:272).

## D. Choices for Lam

1. **Mechanism wording.** (a) Letterbox geometry, no cause — recommended. (b) "Not-Kermany bin". (c) No mechanism.
2. **Follow-up.** (a) None now; letterbox removal in the next phase — recommended. (b) Reduced G1b: patient-disjoint Kermany only, "any rare" share only, labelled "chosen after seeing G1". (c) V's G1b.
3. **Main text.** (a) One sentence on the locked geometry table, G1 in the supplement — recommended. (b) A G1 sentence in the main text. (c) Supplement only.
4. **Page fix.** (a) Two commits, as generated then reworded — recommended. (b) Commit as generated; fix wording in the supplement only.
