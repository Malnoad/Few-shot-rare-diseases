# v1 paper: frozen foundation models on the Yoo et al. rare-OCT benchmark

Assembled on 20 Nov 2026 from the reviewed drafts in Drive `fm_v1/draft/2026-11-19/` (`intro.md`,
`methods_results_v2.md`, `discussion.md`, `claim_ledger.md`, rebuttal logs) and the R2 development
spread (`results/fm_v1/posthoc/R2_development_spread.csv`). Template: `docs/report/fm_ieee/`
(tag `fm-proposal-s1`), IEEEtran conference. Delivery: 3 Dec 2026.

## Files

- `main.tex`: paper (title, abstract, Sections I-VIII, declarations); `supplement.tex`: supplement
  (locked supplement-only results S1-S7, post hoc P0-P3, P5-P7, P9, G1, P10 placeholder, pre-rule files).
- `references.bib`: the 21 proposal entries plus Sun 2024, Tai 2025, TVST 2026 and Zech 2018 (the
  one shortcut-learning source, added 24 Nov). Sun 2024 and TVST 2026 were completed on 24 Nov from
  their Crossref and Europe PMC records.
- `make_tables.py`: writes `tables/*.tex` from the locked results archive and the post hoc CSVs
  (formatting only; no computation on images or model outputs).
- `make_figures.py`: writes `figures/fig2-4` (`.pdf` for the paper, `.png` for viewing) from the
  input lock and the results archive. Fig. 1, the weekly method schematic, was removed on 26 Nov.
- Tables and figures are frozen (step 2, 27 Nov; `results/fm_v1/phase1_freeze.json`). The figure
  PDFs are tracked despite `.gitignore`; they are written without a creation date, so a rerun
  reproduces every file byte for byte.
- Every numeric sentence in the `.tex` files has a `% src:` comment.

## Build

```text
D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_v1_paper/make_tables.py
D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_v1_paper/make_figures.py
cd docs/report/fm_v1_paper && tectonic --keep-logs main.tex
```

Both scripts read `D:/Few-shot/drive/fm_v1/final/final-20261113T170804Z/` in place.

## Decisions applied

Lam's (protocols.md, 19 Nov "after the draft review"): R2 spread in Table I with its label; P4 and P8
in one main-text post hoc subsection; hold-out table = nine-way counts with five-way recall per K;
P0, P1, P2, P3, P5, P6, P7, P9, G1 in the supplement.

Agent defaults recorded there, applied: rounding half up from the shortest decimal form (the
12-significant-digit form of the stored value, so 0.7264999999999999 -> 0.727); letterbox from the
locked geometry summary, 126/146 only in the supplement with its pre-rule label; "resembles" for Sun
et al.; "every rare training image"; MIRAGE pretraining stated in Methods and Limitations;
"RETFound has no published frozen result" dropped; OCTID CSC/MH named by B1 tier; frame-filling
qualifier wherever false-rare rates are generalised; one optional G1 sentence (Results IV-C).

Further defaults taken from the drafts, open to Lam's objection: 4,146 denominator with its
qualifier (C9); called-NORMAL counts without intervals in Table II (C5); R1 column beside the FRR
(C6); Fig. 3 in the main text, geometry table in the supplement; Fig. 4 on all 250 Kermany normals
(replaced on 26 Nov by the 169 patient-disjoint normals, as A2 A-5 requires); "changed after
submission" restored (C11, source: Yoo et al. article history).

Phase-1 E1-E4 (21 Nov): Results (Section IV-F, part "Phase-1 controls") and Discussion
("Phase-1 controls") written from the verdicts Lam approved (protocols.md change log, 21 Nov
"Phase-1 verdicts"); default placement (E1d, E1g, the E2 summary and the OCTDL E2d table in the
main text, the rest in supplement S-D); a "Phase-1 controls" paragraph in Limitations; the RETFound
sentence in "What the results do not show" now points to E3.

## Open (visible TODO markers in the PDF)

Only Lam can supply: Zenodo publication time; data and code archive locations; AI-use wording;
ethics, funding, competing interests, acknowledgements; venue.

Settled by Lam on 26 Nov, with no second mentor review (protocols.md change log, 26–27 Nov): the
side-by-side task table (`tab:tasks`) stays at the head of the post hoc subsection; G1 keeps one
main-text sentence, now with its two plain counts; "Next phase" states the one-line plan (fixed
audit, planted cue, same-patient leakage, second source) with its rules marked not yet approved.

Settled on 24 Nov (protocols.md change log, "24 Nov, remaining Phase-1 choices"): title, abstract,
framing sentences and conclusion kept as drafted; the audit-first order; the task table; the
provenance and P10 paragraph; the pre-freeze wording list; one shortcut-learning citation; the
bibliography fields.

The C2 deviation (OCTID size not constant) is recorded as D1 in `docs/research/FM_DEVIATIONS.md`.
`pre_rule/README.md` carries a 19 Nov exception note for the 126/146 count in the supplement.
