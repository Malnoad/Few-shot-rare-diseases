# L8: RCI definition and numerical audit

2026-10-23. **Bounded audit complete; paper comparability unresolved.**
No training, package changes or published-result overwrites.

## Supported conclusion

The repository computes relative classifier information as
`I(true; predicted) / H(true)`, equivalently
`[H(true) - H(true | predicted)] / H(true)`.
This agrees with the formula in the
[official PyCM RCI documentation](https://www.pycm.io/doc/#RCI-(Relative-classifier-information)),
opened on the audit date. No arithmetic error was found for the saved
five nondegenerate confusion matrices. This does not establish that the original
OCT paper used the same implementation, normalization, class aggregation or inputs.

| Model | Saved/recomputed RCI | Independent conditional-entropy calculation | With true/predicted axes swapped | Paper Table 3 |
|---|---:|---:|---:|---:|
| none | 0.787521 | 0.787521 | 0.808235 | 0.916 |
| basic | 0.760097 | 0.760097 | 0.778680 | 0.953 |
| gan | 0.814041 | 0.814041 | 0.819256 | 0.983 |
| knn | 0.661864 | 0.661864 | 0.709959 | 0.972 |
| proto | 0.429951 | 0.429951 | 0.387658 | 0.933 |

Local inputs: `results/stage4/metrics.csv` and `confusion_<model>.csv`.
Paper inputs: original Table 3, directly checked in the primary XML archived
with [L7](L7_BASELINE_AUDIT.md). Recomputations agree within 1e-12 before rounding.
Changing logarithm base cancels in the ratio. Axis reversal changes the denominator
and these numerical values, but does not reproduce the paper's targets.

## Source boundary

The original [Yoo et al. 2021 main text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7829497/)
names RCI without giving its equation. Its statistical-analysis paragraph cites
reference 38, an earlier OCT/fundus study (DOI
[10.1007/s11517-018-1915-z](https://doi.org/10.1007/s11517-018-1915-z)),
and reference 39, Gorodkin's multiclass correlation paper. The earlier study was
identified through the Europe PMC API, which reports no open full text there;
its formula was not inspected. No author code/full confusion matrices were found
within this bounded check; supplementary files were not exhaustively audited.

The primary RCI paper was located as Sindhwani, Bhattacharya and Rakshit (2001),
[Information Theoretic Feature Crediting in Multiclass Support Vector Machines](https://epubs.siam.org/doi/abs/10.1137/1.9781611972719.16).
Its author-hosted PDF was opened, but text extraction was garbled and screenshot
retrieval timed out. Therefore this audit uses the explicit formula from the
official implementation documentation; it does not claim direct verification of
the original 2001 equation or the OCT authors' implementation.

Do not reverse-engineer a new formula merely to match the target numbers, label
the OCT paper erroneous, or replace existing RCI values without new evidence.

## Verification

An independent column-conditional entropy calculation was checked against the
unchanged `stage4_evaluate.py:rci` function for all five matrices. Five additional
fixtures pass: perfect classification, independence, constant prediction,
bijective label permutation, and an empty class. Count scaling also leaves the
ratio unchanged. A label permutation can retain RCI=1 with incorrect class names:
RCI measures information, so interpretation still needs accuracy/class alignment.

The one-class zero-entropy domain is outside these saved inputs and was not
certified by this audit. No code change is proposed for it here.

Windows Application Control blocked scikit-learn's `_isfinite` DLL. The audit
therefore compiled only the unchanged RCI function AST and evaluated it with
NumPy; the independent computation uses SciPy entropy. The full Stage 4 module
was not imported or executed. This is a focused function audit, not a successful
full evaluation rerun or a claim to have used PyCM software.

```powershell
& D:/Few-shot/.venv/Scripts/python.exe -X utf8 -B .agents/reports/l8-audit/audit_rci.py
```

Local/gitignored outputs: `evidence.json`, `rci-comparison.csv` and the audit
script in `.agents/reports/l8-audit/`. All Stage 4 CSV hashes remain unchanged.
No entry point, notebook, schema, cache path list or method was changed.
Next: prepare C1 design choices with Lam. Retain RCI as explicitly
defined/descriptive, with the paper comparison marked unresolved.
