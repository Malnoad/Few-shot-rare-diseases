# Rare-OCT IEEE proposed-methods draft

Prepared 5 November 2026 for the 11 November submission. The complete English manuscript is **main.pdf**, five pages **including references**, in unmodified IEEEtran conference format (10 pt, US Letter, two columns). No course-specific template was supplied.

- `main.tex`, `references.bib`: editable manuscript and 21 scholarly/tooling references; full author lists retained. The IEEE bibliography uses its documented control entry to abbreviate lists longer than six authors.
- `methods_figure.pdf` / `.svg` / `.png`: original vector workflow schematic and 300 dpi preview; no patient images, synthetic clinical images or measured outcomes. The PDF is the manuscript's figure source. Alt text and provenance: `verification/figure_provenance.json`.
- `CLAIM_SOURCE_LEDGER.md`: 31 paragraph anchors, evidence states, exact sources/locators, terminology and remaining administrative inputs.
- `VERIFICATION.md`: compilation, page/layout review, protocol audit, citation qualifications and protected-file checks.
- `verification/protocol_v1_0.md`: byte-identical finalized protocol; its original source-index path and SHA256 are recorded in `verification/input_provenance.json`.
- `verification/retinal_refs/` and `verification/parent_refs/`: citation audit reports, primary records and bounded reading scopes. These are research provenance, not an asserted exhaustive review.
- `verification/pages/`, `verification/contact_sheet.png`: rendered PDF review images.

The paper is a **proposal**. Only Table I and the explicitly identified prior audit describe historical outputs. No FM scores, new checkpoint access, clinical label confirmation, public registration or completed implementation is claimed. Existing data, results, feature caches and `docs/report/Report.tex` are preserved.

## Compile the supplied sources

From this directory, with an existing Tectonic installation:

```text
tectonic --keep-logs --keep-intermediates main.tex
```

Or use a normal TeX installation with IEEEtran and BibTeX:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The prebuilt vector figure is sufficient; no study dependencies or data are needed to compile. `main.bbl` is also supplied. Do not treat a failed/partial compile as the deliverable PDF: inspect the regenerated output and page count.

## Rebuild and verify within this repository

The authoring scripts are document tools, not a study implementation. They install nothing automatically. Use `D:/Few-shot/.venv/Scripts/python.exe` in this worktree.

```text
D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_ieee/build.py --engine PATH_TO_TECTONIC
D:/Few-shot/.venv/Scripts/python.exe docs/report/fm_ieee/verify_package.py
```

`build.py` uses the existing Matplotlib installation to regenerate the schematic. `verify_package.py` additionally requires PyMuPDF and Pillow to inspect/render the PDF, verifies the frozen input hash and historical arithmetic, and compares 62 protected tracked-file hashes. During preparation, Tectonic 0.17.0 and PyMuPDF 2.1 were placed in a temporary tools directory; no project dependency or main virtual environment was changed. Exact versions are in `verification/build_environment.json` and `verification/figure_provenance.json`.

`assemble_bibliography.py` regenerates `references.bib` from the preserved two audit batches, preferring the verified OCTID 2020 journal version. `prepare_evidence.py` and `verification/parent_refs/read_primary.py` are optional provenance-retrieval utilities, not part of normal compilation. They access only specified writing inputs/primary records; the actual study loader, grouping and classifiers remain future work.

## Before submission

Replace “Lam” with the required full author/affiliation block and confirm the course's template and submission hour. The methods, proposal status and literature are complete; new experimental results are not required. The source package includes an explicit AI/tooling disclosure for author review.

## Not committed

The repository is public, so third-party material retrieved during citation
checks is kept only in the local archive
`D:/Few-shot/.agents/reports/archive/t3-a5214cb1/` (gitignored): publisher pages and
PDFs (`verification/retinal_refs/raw/`, `verification/parent_refs/*.pdf`), extracted
paper text (`parent_refs/*_passages.txt`, `CLAP.txt`, `CoCoOp.txt`, `DINOv2_*.txt`,
`FairNLP.txt`, `ProtoNets.txt`), and regenerable build output (page renders,
`contact_sheet.png`, `preview.html`, `main.aux/.log/.out/.blg`, `fm_ieee_sources.zip`).
`verify_package.py` regenerates the renders; the audits that cite those sources are committed.
