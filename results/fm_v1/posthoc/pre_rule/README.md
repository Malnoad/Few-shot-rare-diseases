# Numbers computed outside a committed rule (released 18 Nov 2026)

`protocols.md` section 8, rule 1, says no post hoc number exists before its rule is committed.
The scripts and outputs here broke that rule or were never covered by a rule. They are released
under rule 3 ("everything is released"), as the P6 slip (A10) was. **None of them is a result**;
none may be quoted in the paper except to say it exists.

They were written by the AI agents working with Lam, in a temporary scratch folder, while reading
the locked archives (`results-20261114T093033Z.zip`, the input lock and the feature caches). Paths
inside the scripts point to that scratch folder and to Lam's laptop; they are kept as written.

| Folder | What | Rule status |
|---|---|---|
| `2026-11-15/` | First looks at the locked results: per-class recall (`perclass*.py`, later P0), the MIRAGE + DINOv2 z-score ensemble on development, hold-out and the external sets (`ensemble.py`, `ensemble_out.txt`, later P7), archive structure, image sizes and exact bounds (`dump.py`, `struct.py`, `sizes.py`, `checks.py`) | Before any section 8 rule existed (P0 and P7 are already labelled "rule fixed after first look") |
| `2026-11-16/` | Probe against prototype per K (`probe_vs_proto*`, later P6), standard errors and confusion checks for the 16 Nov interpretation (`chk.py`, `conf*.py`, `extract.py`, `grab.py`, `outs.py`, `peek.py`, `fable_findings.md`) | Before rule 1 was met (commit `1eb3964`, 16 Nov 17:56 Bangkok) |
| `2026-11-18/` | Reviewer estimates for the v2 design: power, false-rare bounds and K-mean differences (`power.py`, `frr.py`, `kmean.py`, written up in `fable_v2_design.md`); Kermany-only feasibility checks for the leakage audit (`check1.py`, `check2.py`, written up in `fable_a2_a4.md`). A 19 MB feature cache the power script made was not kept | No rule covers them; `check1.py` and `check2.py` compare no pair of rare images |
| `2026-11-18/` (added 19 Nov) | Mendeley provenance: `map_v1.py` maps version 1 (119 files, not stored here) to our 146 by file and pixel hash, giving `v1_to_ours.csv`; `md_versions.json` is the dataset's version list. Image geometry: `geom.py` (padding share per set, from G1's `padding.csv`), `geom_a.py`, `geom_b.py` (A2 border rule per image, writes `rare_layout.csv`), `geom_c.py` (top and bottom bands). G1 reviews: `g1_review_opus.md`, `g1_review_fable.md`, `g1_verdict_opus_main.md` (its point 2, "rare images get no bars", is wrong and was withdrawn after `g1_rebuttal_fable.md`), `g1_rebuttal_fable.md`. Hold-out wording plan: `fable_holdout_plan.md` (its section D wording is adopted, protocols.md 19 Nov) | File identity and pixel layout only, no model output, except the G1 reviews, which read the G1 outputs after scoring |

The ensemble's external shares in `ensemble_out.txt` (OCTID -0.099, OCTDL MH +0.155 against
MIRAGE) and the v2 power figures were quoted in the 18 Nov discussion; they stay development
notes, never results.
