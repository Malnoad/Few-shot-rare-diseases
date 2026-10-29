# Agent mistakes log

Every real agent mistake in this project, with the enforcement outcome or an
explicit Note when no practical check exists. Agents add entries with the `log-mistake` skill; Lam approves the fix.

**Fix levels, strongest first:**
1. **Check**: a command in `.agents/check`, a test, a permission rule, or a
   hook. The mistake is caught automatically. Prefer this whenever possible.
2. **Proposed check**: exact runnable code below; not installed because it needs
   a broader workflow or another runtime. No vague advice is retained as a fix.
3. **Note**: logged here only. Use for one-offs; if it happens again, escalate.

| # | Date | Agent / model | What went wrong | Root cause | Fix level | Fix (where) | Seen again? |
|---|---|---|---|---|---|---|---|
| 1 | YYYY-MM-DD | e.g. Codex · gpt-6.1-sol | e.g. Ran the full test suite (8 min) for a one-line change | AGENTS.md didn't name a fast check | Note | Template example, not a real incident; no check claimed. | no |
| 2 | 2026-10-26 | Claude Code · claude-opus-5-5 | Lam's review-sheet verdicts were lost: I showed my private (incognito) test tab in the preview, kept automating it and cleared its storage | Assumed a test tab stays mine once visible, and that browser storage is a safe place for answers | Check | Existing `review_sheet.py serve` persists answers to disk, merges by timestamp and reports save failures. Retain this script-enforced fix; reject the unenforced hidden-tab advice. | no |
| 3 | 2026-10-23 | Codex · gpt-6.1-sol | Skill inventory stopped when Chinese text reached a cp1252 stdout | Assumed Windows Python stdout supported Unicode | Note | External skill-inventory audit has no maintained runner here. Reject a generic UTF-8 reminder as a fix; the new project guard runner uses UTF-8 explicitly but does not police arbitrary audits. | no |
| 4 | 2026-10-23 | Codex · gpt-6.1-sol | Unit-audit fixture used a positional filename instead of required --input | Applied another helper's CLI convention before reading this helper's --help | Note | External unit-audit fixture/parser is not maintained here. No fixture to attach an argv check to; reject the generic CLI reminder. | no |
| 5 | 2026-10-23 | Codex · gpt-6.1-sol | Git Bash path probe used backslash-space escaping in PowerShell; a later nested bash check also lost argument quotes | Mixed shell quoting conventions; corrected by Python subprocess argv | Note | No maintained path-probe wrapper here; arbitrary shell quoting cannot be checked safely by a substring rule. Reject the generic subprocess reminder. | yes, during this audit; persistent fix pending |
| 6 | 2026-10-23 | Codex · gpt-6.1-sol | git status on agent-config failed because it has no Git metadata | Assumed the shared configuration directory was already a repository | Note | External agent-config directory, not a project repository command. A .git existence rule is also wrong for worktrees; no project check selected. | no |
| 7 | 2026-10-23 | Codex · gpt-6.1-sol | Tried reading nonexistent results/stage4/comparison.csv | Guessed the result filename before listing artifacts | Note | One-off lookup; no maintained consumer of comparison.csv. Reject a generic list-files reminder. | no |
| 8 | 2026-10-23 | Codex · gpt-6.1-sol | Inline C1 planning snippet failed Python syntax parsing | Mixed dict keyword arguments with colon-style entries | Note | Temporary snippet is not retained; nothing for py_compile to check. Existing maintained scripts already compile in `.agents/check`. | no |
| 9 | 2026-10-24 | Codex · gpt-6.1-sol | GAN review sample had 10 repeated paths; CSV export also used literal line-break escapes | Assumed PNG glob patterns differed by case on Windows; over-escaped embedded JavaScript | Proposed check | Exact uniqueness and JavaScript CSV round-trip check below. Not added to `.agents/check`: Node is not a project Python/Colab dependency. | no |
| 10 | 2026-10-27 | Codex · gpt-6.1-sol | T1 identity lookup lost jq quotes in PowerShell; a Unicode read failed on cp1252 stdout; a later smart apostrophe broke shell parsing | Repeated shell-quoting and stdout assumptions (entries 3 and 5) | Note | Repeats 3/5, but no maintained T1 integration runner exists here. Reject an unspecified preflight helper; the new guard scripts use argv and UTF-8 at their own I/O boundary. | yes |
| 11 | 2026-10-27 | Codex · gpt-6.1-sol | GitHub server-side merge of PR 2 used the account Gmail author despite noreply branch metadata | Assumed local identity controls the merge API author | Check | `.agents/check_identity.py` checks both outgoing author and committer against repo-local noreply; `.agents/hooks/pre-push` enforces it locally (install documented in README). Both tool hooks deny gh pr merge; Claude also has a deny rule. Integrate by local rebase and non-force fast-forward push only; old history stays unchanged. | yes: PR #10 rebase-merge by Claude Code · claude-opus-5-5 set the Gmail as committer |
| 12 | 2026-10-27 | Codex · gpt-6.1-sol | Temporary T1 verification falsely flagged the authorized README edit, then looked for subcommands in argparse choices | Assumed path separators and parser structure before checking the actual representations | Note | Temporary T1 verification is not maintained here; no stable parser/schema contract to test. Reject path/parser heuristics as a fix. | no |
| 13 | 2026-10-27 | Codex · gpt-6.1-sol | Inline Python environment probe failed with SyntaxError; first patch failed its context check | PowerShell stripped nested quotes; default text decoding misread a Unicode character | Note | Temporary T5 probe/patch, no retained code to check. Reject here-string/encoding advice as a persistent fix; project check compiles the maintained scripts. | no |
| 14 | 2026-10-27 | Codex · gpt-6.1-sol | Notebook dump failed on Unicode; guessed C1 filename and a literal Windows rg glob also failed | Assumed console encoding and paths rather than verifying them | Note | One-off notebook/path inspection, no maintained caller here; reject generic encoding and discovery reminders. | no |
| 15 | 2026-10-27 | Codex · gpt-6.1-sol | First L9 check invocation lost quoted Python -c arguments inside a Bash function passed through PowerShell | Repeated nested native-shell quoting assumption (5, 10) | Note | No maintained L9 inspection launcher here; reject an unspecified quoting preflight as a fix. The project check runner is exercised directly through Git Bash. | yes |
| 16 | 2026-10-22 | Claude Code · claude-opus-5-5 | ln -s of the main .venv in a worktree made a 2 GB copy | Git Bash without Developer Mode copies instead of creating this symlink | Check | `.claude/settings.json` denies Bash(ln -s *) and Bash(ln -sf *); both PreToolUse guards also block symbolic-link options. AGENTS.md names the main venv path. | no |
| 17 | 2026-10-23 | Claude Code · claude-opus-5-5 (lit-scan thread) | paper-lookup NCBI email parameter sent Lam's email to NCBI | Passed personal identity to an external API without an input guard | Check | Both PreToolUse hooks use `.agents/tool_guard.py`: regex candidates, lowercase SHA-256, stored digest only; fake-address tests in `.agents/test_guards.py`. | no |
| 18 | 2026-10-23 | Claude Code · claude-opus-5-5 (review-sheet thread) | Called a partial HTML preview full | Inferred completeness from a partial rendering | Note | No reliable automatic check that a narrative completeness claim matches the view inspected; reject a vague preview reminder. | no |
| 19 | 2026-10-27 | Codex · gpt-6.1-sol | A PowerShell-to-Python pipe replaced middle dots in new log text with question marks | Set Python UTF-8 output, but overlooked PowerShell's input pipe encoding (repeat of 3/10) | Note | Repaired text through apply_patch. No maintained shell-input wrapper exists here; reject another encoding reminder as a fix. | yes: repeats 3/10 |
| 20 | 2026-10-27 | Codex · gpt-6.1-sol | codex --strict-config features list was rejected | Assumed an option in top-level help was supported by the features subcommand | Note | No unsupported invocation retained. Settings verified from official docs and configured command probes; strict CLI validation remains unverified. No maintained caller to check. | no |
| 21 | 2026-10-27 | Codex · gpt-6.1-sol | C5 notebook dump failed on cp1252 stdout, repeating entries 3/10/14; a temporary preservation check counted 8 added cells instead of 7 | Forgot UTF-8 mode; guessed the cell offset | Note; check proposed | Recovered using `-X utf8` and the actual cell-count delta; propose a shared inspection launcher that sets `PYTHONUTF8=1` and asserts stdout.encoding is utf-8. No persistent fix applied pending Lam | yes |

| 22 | 2026-10-27 | Codex · gpt-6.1-sol | T8 notebook inspection again failed on cp1252 Unicode output | Repeated entries 3/10/14 by assuming stdout encoding | Note; stronger check proposed | Corrected session stdout and used ASCII cell instructions. Propose a persistent PowerShell subprocess wrapper setting `$OutputEncoding = [System.Text.UTF8Encoding]::new()` and Python UTF-8 mode, with a Unicode stdin/stdout round-trip check; awaits Lam approval | yes |

| 23 | 2026-10-29 | Codex · gpt-6.1-sol | T10a inspection repeated cp1252 output failure; extraction probes also assumed uniform source types and no leading blank/header lines | Repeated encoding issue (21/22) and assumptions about notebook serialization | Proposed check; note | Recovered with UTF-8 and parsed actual text. Escalation proposed: route notebook inspection through the persistent UTF-8 wrapper/round-trip check proposed in 22; compare joined source text separately from source JSON representation. No permanent fix applied pending Lam | yes |

| 23 | 2026-10-27 | Codex · gpt-6.1-sol | Report inspection probed absent archive paths, guessed helper names, lost native-shell quotes, and assumed a PowerShell redirected UTF-16 log was UTF-8 | Repeated path/CLI and encoding assumptions from 5/7/14/22 | Proposed check | Extend entry 22's proposed wrapper with required-path validation, Python through UTF-8 stdin and an explicitly encoded log boundary; candidate and verification below. Not installed pending Lam's approval. | yes |
| 24 | 2026-10-27 | Codex · gpt-6.1-sol | Initial report used C0/C1/C5 task codes without an upfront explanation; Lam requested clarification | Carried tracker vocabulary into a reader-facing report without defining it | Note | Added the explicitly requested C0--C5 purpose/activity/status table and dependency explanation in Report.tex. A label-presence test would not verify explanatory clarity; no permanent guard proposed. White/black formatting is Lam's new explicit preference, not an error in following the earlier dark-mode default. | no |
| 25 | 2026-10-27 | Codex · gpt-6.1-sol | Report referred to Lam in the third person and mixed in plural author pronouns; Lam requested first-person singular | Wrote as an external adviser instead of consistently using the research author's perspective | Proposed check | Revised prose, planning tables and assistance disclosure to I/my; proposed narrative-pronoun screen below is not installed pending approval. | no |
| 26 | 2026-10-27 | Codex · gpt-6.1-sol | A temporary prose-preservation screen falsely flagged changed repetitions of C1/v1 as numerical result changes | An unrestricted digit regex conflated task/pool identifiers with quantities | Check (existing) | The maintained report-evidence checker verifies actual result macros; the corrected temporary comparison separately confirms standalone quantities and bibliography. No new permanent guard installed. | yes: same temporary-verification overreach as entry 12 |

## Proposed report-voice check for entry 25 (not installed)

This bounded check catches plural author pronouns and third-person references
to Lam in the narrative. It does not prove that every sentence uses the right
voice; planning tense and attribution still require editorial review.

```python
import re
from pathlib import Path
text = Path("Report.tex").read_text(encoding="utf-8")
body = text.split(r"\begin{document}", 1)[1].split(r"\begin{thebibliography}", 1)[0]
assert not re.search(r"\b(?:we|our|us|Lam(?:'s)?)\b", body, re.IGNORECASE)
```

No permanent check or hook was added. The requested voice edit is applied;
the proposed automation remains for Lam's approval.

## Proposed inspection wrapper for entries 22--23 (not installed)

Candidate `.agents/inspect-python.ps1`; a caller supplies paths discovered in the
current checkout and Python source as a PowerShell here-string. Missing paths
fail explicitly; Python source is not passed through native `-c` quoting.
Approval is required before adding this wrapper or wiring a permanent check.

```powershell
param([string[]]$RequiredPath = @(), [Parameter(Mandatory)][string]$Code)
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
foreach ($inspectionPath in $RequiredPath) {
    if (-not (Test-Path -LiteralPath $inspectionPath)) { throw "Missing inspection input: $inspectionPath" }
}
$Code | & 'D:/Few-shot/.venv/Scripts/python.exe' -X utf8 -
exit $LASTEXITCODE
```

Proposed verification: supply a temporary path and Python source that reads and
prints Unicode, quotes and dollar signs literally; then supply an absent path
and assert Python did not start. This would check the wrapper's I/O boundary,
not attempt to police arbitrary terminal commands.
If persistent log capture is added, write its bytes explicitly as UTF-8 and
assert a non-ASCII log round-trip. Existing PowerShell redirection logs require
BOM inspection before decoding; the report's verification record records its
actual UTF-16 encoding. No log-capture guard was installed.

## Proposed check for entry 9 (not installed)

This runs against the maintained sampler and the actual JavaScript exporter,
without generating images or changing data. Node is an additional runtime, so
it is proposed rather than required by the Python/Colab fast check. Save as
`.agents/check_review_export.py` if approved, and add
`"$PY" .agents/check_review_export.py` to `.agents/check`:

```python
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path.cwd()))  # .agents/check runs from the project root
import review_sheet as sheet

for disease in sheet.RARE:
    folder = Path("data/gan/accepted") / disease
    names = [p.name for p in folder.glob("*.png")]
    sample = sheet.sample_files(names, 100, 42)
    assert len(sample) == len(set(sample)) == min(100, len(names))
    assert set(sample) <= set(names)
fixture = {field: "value" for field in sheet.ANSWER_FIELDS}
fixture["rater"] = 'comma, quote" and newline\nnext line'
js = sheet.TEMPLATE.split("const csvCell = ", 1)[1].split("const ganBody = ", 1)[0]
prefix = "const D = " + json.dumps({"cams": [], "gan": {"X": [fixture]}}) + ";"
prefix += 'const RARE=["X"]; const ganRow=r=>r; const camRow=r=>r;'
result = subprocess.run(["node", "-e", prefix + "const csvCell = " + js + "process.stdout.write(csv());"],
                        capture_output=True, text=True, encoding="utf-8", check=True)
assert list(csv.DictReader(io.StringIO(result.stdout))) == [fixture]
print("Review sample uniqueness and CSV round-trip: PASS")
```
