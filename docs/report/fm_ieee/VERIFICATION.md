# Verification record — 5 November 2026

## Delivered draft

`main.pdf` is **five compiled pages including all 21 references**. It contains a complete abstract, introduction/problem statement, synthesized related work, proposed methods with five numbered equations, discussion/execution boundaries, conclusion and tooling disclosure. Table I is explicitly historical; Fig. 1 depicts planned operations only. There is no new FM Results section or invented outcome.

Build: **Tectonic 0.17.0**, IEEEtran class 1.8b / IEEEtran BibTeX style 1.14; standard conference option, 10 pt, 612×792 PDF points (US Letter), two columns. No reduced font, geometry override, negative spacing or margin adjustment was used to achieve the page count. The bibliography's documented `IEEEtranBSTCTL` interface abbreviates author lists longer than six; the BibTeX keeps all published authors.

## Mechanical checks and layout review

Command: `D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_ieee/verify_package.py` (with the temporary PyMuPDF directory on PYTHONPATH).

```text
PDF: 5 IEEE conference pages; 21 resolved citations; 31 mapped claim anchors
Protected files: 62 checked, 0 changed
LaTeX: no overfull boxes, unresolved references or missing glyphs
Package checks: PASS
```

`verification/package_checks.json` records the final PDF SHA256, each page's dimensions/fonts/word count and the preservation results. `verification/pdf_text.txt` is extracted text, not a substitute for the PDF's reading order. All pages were rasterized at 130 dpi and visually reviewed individually, including page 3's schematic/table/equations and page 5's complete bibliography. No cropped text, colliding equation numbers, missing citations or blank final page was found. `verification/pages/` and `contact_sheet.png` retain the reviewed renderings. The standalone figure is 7.16×2.38 inches; PNG 2148×714 pixels at 300 dpi, with PDF TrueType fonts embedded.

The TeX log retains ordinary underfull-box notices and early XeTeX/IEEEtran font-encoding fallback warnings during class initialization. Actual rendered body/title fonts were inspected as embedded Times-compatible Nimbus Roman, with Computer Modern mathematics and an embedded DejaVu Sans figure. No Type 3 bitmap fonts or missing-glyph messages remain. Standard 5-TeX-point math scripts are approximately 4.98 PDF points; the bounds checker accounts for that unit difference. These checks are not a publisher-compliance certification.

## Protocol consistency

The controlling copy is byte-identical to the source-index target: SHA256 `4e8df8d91c0021187c05b103a9b0a0e0319a055fd91108da1f4cf3045d00c5df`.

| Protocol requirement | Manuscript location / check |
|---|---|
| Descriptive fixed-checkpoint study; no new outcomes | Abstract, I, IV–V; historical Table I explicitly labelled |
| RETFound Nature OCT / DINOv2-Base / MIRAGE-Base; Keras reference | III-C, Table II; exact public identities; prescore-only RETFound fallback |
| 124 inspected labels distinct from 5K supports; 22 inspected legacy labels | III-A; manifest counts rechecked read-only |
| Real supports only; no synthetics/query adaptation/scaler | II-A and III-A/C; all broader-exposure augmentation remains historical |
| Exact/RGB/dHash dependence screen; verified patients kept distinct | III-B; BOX 9×8, 64-bit horizontal left>right, Hamming ≤4; no new groups asserted |
| Five grouped folds; ≥10 groups/class and eight fitting groups/class/fold | III-B; whole-selector stop before scores on failure |
| Unit supports → mean → normalized centroid; numerical failures stop | III-C/D, (1)–(2); fixed rare-order ties |
| Mean CE + 0.005||W||², zero init, exact L-BFGS-B settings | III-D, (3); coefficient equals protocol's 0.01||W||²/2; no fixed C or historical Adam head |
| Development K1/2/4/8, 100 paired nested draws; reporting adds K16 only if feasible for all | III-E; secondary development K4/first25, reporting first25; full RNG recipe delegated explicitly to accompanying protocol §§4,6 |
| Pooled within-class out-of-fold nomination; complete FMs only, ≥2 | III-E, (4); Inception excluded from candidacy |
| Near-best ≤0.02; 100-image, one-warm/five-pass, batch1 FP32 timing | III-E; prescore order, same session/hardware, median per-pass means, input preprocessing included, I/O excluded, final fixed tie order |
| Nominee locked before legacy/external; all arms reported | III-E; no partial-score nomination or outcome-driven re-designation |
| Support-draw spread conditional; not patient CI or confirmatory inference | III-E, IV; paired counts and development-selection optimism |
| Separate deterministic nine-way real-only prototypes; 1,000/common class, seed42 | III-F; exact common roster retained; new FP32 Keras features, no old quantized cache substitution |
| NORMAL FRR separately by Kermany/OCTID/OCTDL plus other error endpoints | III-F, (5); five-way head not used for FRR |
| External release/eligibility gate, screen/link exclusions, pre-inference query lock | III-F; no retroactive fitting changes; mappings/terms/access not asserted passed |
| Partial external rare cohorts retain full outputs; no five-class aggregate | III-F; CSR/CSC and OCTDL MH restrictions; cosine scores not probabilities |
| 5,000 patient-cluster draws seed0 only with verified IDs | III-F; otherwise explicitly conditional image/group intervals |
| Isolated versioned outputs, identity lock and failures; secondary probes dropped first | IV; November milestones and no registration claim |

The main paper compresses implementation-level RNG namespaces, external-output persistence details and durable-archive mechanics by referencing the complete protocol; it does not alter them. No checkpoint revisions, exact native transforms, realized eligible groups, external cohort counts or measured compute values are fabricated. Those remain future gates.

## Citation verification and qualifications

The twelve-source retinal batch was checked against primary publisher/conference/arXiv records and named passages; the nine-source parent batch used official proceedings, arXiv records and hash-matched local papers. See both audit reports under `verification/`. The reading scope is targeted and explicitly recorded, not claimed full-paper review of the entire supplied corpus.

- **DINOv2:** the primary PDF's TMLR January 2024 header and author list control over the older 2023 submission year; Huy V. Vo is retained from the PDF.
- **OCTID:** final bibliography uses the verified **2020 Computers & Electrical Engineering 81, article 106532** journal version, DOI `10.1016/j.compeleceng.2019.106532`. The audit's original `OCTID2018` preprint is retained as provenance, not cited as a second study. The DOI's 2019 component is not its issue year.
- **BenchReAD:** explicitly the **MICCAI 2025 conference version**, supported by the official conference record and Crossref. Springer's conflicting 2026 citation metadata is documented rather than silently merged.
- **RETFound:** complete published byline includes the UK Biobank Eye & Vision Consortium in its rendered position; some exports omit/reorder/expand it. The generic citation validator flags the escaped ampersand inside this valid braced corporate author as if it separated authors. This is a documented **false positive**, checked against the primary byline; the author separators are `and`.
- Generic missing-page warnings are not repaired with fabricated fields: article identifiers are labelled “Art. no.”; TMLR/preprints and ICLR have no invented volume/page/DOI fields; the official ProtoNets BibTeX leaves pages empty. The nonprinting `IEEEcontrol` entry is a style control, not an unused scientific reference. `verification/citation_validation.json` retains the generic report; the resolved assessment is in `citation_assessment.json`.
- All 21 scientific/tooling references resolve to manuscript keys, with no duplicate keys or unreferenced scholarly entry. There was no comprehensive correction/retraction audit. Detailed A07 controls, EyeDiff's encoder-freezing schedule and clinical external-label equivalence are not claimed verified.

## Historical arithmetic and preservation

Seed-42 accuracy comes directly from the unchanged `results/stage4/metrics.csv`. Means and sample SDs were checked in memory from all 15 `results/seed_check.csv` rows; the final table is 90.69±1.86, 92.93±1.85 and 92.59±1.29 across five training seeds. This check does not rerun the classifier or overwrite historical summaries. C0 AUC quantities are archived observations with a noncausal interpretation.

All 62 tracked files under `data/`, `results/` and the pre-existing report have their original SHA256 values. No stage training/generation/QC, feature extraction, FM inference, new grouping screen, download of study weights/images or evaluation experiment was run. Only primary literature/metadata and temporary document tools were retrieved. No project dependency, split, raw image, result or cache was changed; no commit/push was made.

## Repository pre-completion surfaces

- **Entry points:** unchanged; stage scripts, flags and notebook were not edited.
- **Reverse states:** not applicable to static document artifacts; no product action was added.
- **Data & schemas:** unchanged; no image list/cache/schema mutation.
- **Docs:** new draft and provenance only. README/COLAB/CONTEXT need no method/data/result amendment because this task reports the already finalized proposal without changing or executing it.
- **Checks:** repository `.agents/check` passed via Git Bash using the main Python 3.12 environment. It exercised the existing 5/6/4/10/5/2/7-test suites, manifest/guard checks and outgoing identity check. The TensorFlow tests use their existing temporary synthetic fixtures; they are not study experiments. Relevant terminal output:

```text
Ran 5 tests in 0.917s
OK
Ran 6 tests in 2.703s
OK
Ran 4 tests in 0.064s
OK
Ran 10 tests in 1.414s
OK
Ran 5 tests in 0.019s
OK
Ran 2 tests in 4.310s
OK
Ran 7 tests in 20.899s
OK
Identity check: PASS (0 outgoing commits; author and committer match local noreply)
Agent checks: PASS
```

Operational lookup mistakes were recorded in `.agents/mistakes.md` as required. The existing proposed path/UTF-8 wrapper and a dataset-title/DOI identity guard remain proposals; no permanent guardrail change was installed without Lam's approval.

## Author check before submission

Confirm the required full name/affiliation block, any course-specific template and the submission hour. Review the prose, historical framing and AI/tooling disclosure. Access/loading, duplicate/group feasibility, external terms/inventories and clinical mappings are explicitly future gates and do not block this completed proposed-methods report.

## Author update, 8 November 2026

Only the author block and PDF author metadata in `main.tex` changed, then the PDF was rebuilt with the same Tectonic 0.17.0 engine: five pages, no overfull boxes or undefined references, title block inspected. Current `main.pdf` SHA256: `7e30690e0148c17d2db2deb6bc9404576e22d1cb17555d3be9ab7950b95978d4`, after the closing note was also changed to "The authors remain responsible for this work." (four-author group). `verification/package_checks.json` and the other records above describe the 5 November build; `verify_package.py` was not rerun.

## Affiliation update, 10 November 2026

Lam supplied the affiliation, National Economics University, for all four authors; a
mentor may be added later. Lam confirmed the course template is IEEE format and the
submission deadline is 11 November, 22:00. Only the author block in `main.tex` changed
(`\IEEEauthorblockA{National Economics University}`); the PDF was rebuilt with the same
Tectonic 0.17.0 engine. `verify_package.py` was rerun and passed:

```text
PDF: 5 IEEE conference pages; 21 resolved citations; 31 mapped claim anchors
Protected files: 62 checked, 0 changed
LaTeX: no overfull boxes, unresolved references or missing glyphs
Package checks: PASS
```

The log keeps 9 ordinary underfull-box notices. All five rendered pages were reviewed by eye;
the title block shows the four names above the affiliation. Format: `IEEEtran` conference
class, 10 pt, two columns, US Letter (612×792 pt), with no geometry, font-size or spacing
overrides. Current `main.pdf` SHA256:
`0748d23c93251513adbd70bce75b56db46088a5e568a3d56f40685213ca85dc8`.

## Figure and closing-note update, 11 November 2026

At the authors' request, the Writing and Reproducibility Note was removed from `main.tex`.
It was the only citation of `Kassis2026`, so that entry left `references.bib` (and
`assemble_bibliography.py` now drops it); the bibliography has 20 references. Fig. 1 was
redrawn by `make_figure.py` in the figures4papers house style
(github.com/ChenLiu-1996/figures4papers, `scientific-figure-making/references/design-theory.md`:
its blue/green/red/neutral palette, Arial, rounded pastel boxes, block arrows, 300 dpi PNG).
The code is new; none of that CC BY-NC repository's code or images were copied. The figure
keeps its 7.16x2.38 inch size and its content, adds the roster as four labels and draws the
"no outcome feedback" rule as a dashed line; the caption is unchanged. Figure fonts are
embedded TrueType Arial. Rebuilt with the same Tectonic 0.17.0 engine; `verify_package.py`:

```text
PDF: 5 IEEE conference pages; 20 resolved citations; 31 mapped claim anchors
Protected files: 62 checked, 0 changed
LaTeX: no overfull boxes, unresolved references or missing glyphs
Package checks: PASS
```

9 ordinary underfull-box notices remain. Pages 3 (figure) and 5 (references) were reviewed by
eye. That build's `main.pdf` SHA256 was
`b9bcb9a9cfa59dbed112b0c7e2d856344c0cb1d93ec16f5b574807b5f4dfab3e`.

## Plain-language rewrite, 11 November 2026

At the authors' request, the prose of `main.tex` was rewritten in plainer language after the
style of Cao and Pham, arXiv:2608.01602 (problem first; "rather than proposing ..., we
test ..."; a numbered contribution list; bold run-in paragraph leads; an explicit
"Limitations." paragraph). All 31 claim anchors, numbers, equations, tables, the figure and
the 20 citations are unchanged in substance; no claim or source was added. The first draft ran
to six pages, so the abstract and the longest paragraphs (C01-C03, C21, C25, conclusion) were
shortened. `verify_package.py`: PASS, 5 pages, 20 citations, 31 anchors, 62 protected files
unchanged, no overfull boxes; 7 underfull notices. All five pages were reviewed by eye.
That build's `main.pdf` SHA256 was `99ecfcb457f3fe190c239f6abfb36e5844d2b8e18beeb463dd38a5ba7e84846a`.

## First-reader review fixes, 11 November 2026

A read-only clarity review (GPT-6.1-Sol, Codex, acting as a project verifier) found no count
errors; it rechecked 124/22/146, 84,630, 73,958, 1,000 common test images and 5,146. Its
wording fixes were applied after checking each against the protocol:

- Abstract defines frozen weights, supports and queries; "logistic probe" reads "logistic
  regression"; nomination happens "before any new test result" (also "new" in C21).
- C03 names the AUC as rare-versus-common and says what was normalized (file format and
  framing; `docs/research/CONTEXT.md`, decision of 29 Oct).
- C04 defines "arm"; C11 defines "legacy queries" and replaces "no real rare validation images"
  with the manifest-split versus development-fold distinction.
- C13 explains components; C14 says which folds hold out and which fit.
- C19 states the 100 reporting prototype draws (protocol section 1 table); C20 defines macro
  recall; C21's heading and first sentence say the nominee is the fastest FM within 0.02,
  not necessarily the top scorer.
- C23 replaces K by each class's support count for the nine-way head and names the 1,000 common
  test images (protocol line "4,000 common fitting +1,000 common tests +146 rare = 5,146").
- C24 says the prediction is nine-way and that the five-way head cannot give a meaningful FRR.
- C25 expands CSR as OCTID's "central serous retinopathy" (Borealis dataset
  doi:10.5683/SP/FVLZIL title, read 11 Nov).
- Table II row "RETFound fallback"; Fig. 1 reads "100 prototype draws; 25 probe draws".

To stay within five pages, sentences repeating points made elsewhere were cut: the historical
prototype/Adam-head remarks (C17, C18), the synthetic-plausibility remark (C07), the
parameter-count remark (C10), the common-patient-overlap repeat (C26), the failures repeat
(C28), the versioned-outputs repeat (C29) and one conclusion sentence. Not changed, author
decision: the reviewer noted that "failed external NORMAL access" does not say whether one or
both external sources must fail; the protocol has the same wording.

`verify_package.py`: PASS, 5 pages, 20 citations, 31 anchors, 62 protected files unchanged, no
overfull boxes; 8 underfull notices. All five pages were reviewed by eye. Current `main.pdf`
SHA256: `bdce1e31df5c99c93e887700529614e4fa28168c98fc1ab548e218d992abadba`.

## Sent to the mentor, 11 November 2026

Lam sent this build to the mentor as `Draft.pdf` and chose it as the version of record (12 Nov).
Its SHA256 equals the current `main.pdf`:
`bdce1e31df5c99c93e887700529614e4fa28168c98fc1ab548e218d992abadba`.
