# v1 paper: frozen foundation models on the Yoo et al. rare-OCT benchmark

Assembled on 20 Nov 2026 from the reviewed drafts in Drive `fm_v1/draft/2026-11-19/` (`intro.md`,
`methods_results_v2.md`, `discussion.md`, `claim_ledger.md`, rebuttal logs) and the R2 development
spread (`results/fm_v1/posthoc/R2_development_spread.csv`). Template: `docs/report/fm_ieee/`
(tag `fm-proposal-s1`), IEEEtran conference. Delivery: 3 Dec 2026.

## Files

- `main.tex`: paper (title, abstract, Sections I-VIII, declarations); `supplement.tex`: supplement
  (locked supplement-only results S1-S7, post hoc P0-P3, P5-P7, P9, G1, P10 placeholder, pre-rule files).
- `references.bib`: the 21 proposal entries plus Sun 2024, Tai 2025 and TVST 2026 (fields no
  repository file holds are marked TODO).
- `make_tables.py`: writes `tables/*.tex` from the locked results archive and the post hoc CSVs
  (formatting only; no computation on images or model outputs).
- `make_figures.py`: writes `figures/fig1-4` from the input lock and the results archive.
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
(C6); Fig. 3 in the main text, geometry table in the supplement; Fig. 4 on all 250 Kermany normals;
"changed after submission" restored (C11, source: Yoo et al. article history).

## Open (visible TODO markers in the PDF)

Title, abstract, conclusion and the reliability reframing (Lam approves); Phase-1 controls E1-E4
(Results and Discussion placeholders); Zenodo publication time;
shortcut-learning citation; planted-cue wording; data and code archive locations; AI-use wording;
ethics, funding, competing interests, acknowledgements; venue; missing bib fields (Sun 2024 title
and DOI, TVST 2026 authors and title).

Not done here: where the C2 deviation (OCTID size not constant) is recorded (`FM_DEVIATIONS.md`
does not exist); `pre_rule/README.md` still says its files may not be quoted, while the 19 Nov
default puts the 126/146 count in the supplement.
