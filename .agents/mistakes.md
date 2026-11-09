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
| T12 | 2026-10-29 | Codex ? gpt-6.1-sol | Benchmark test-label lookup failed on Windows manifest paths | Reimplemented cache loading without Stage 3 fix() normalization | Proposed check | Pending Lam approval: assert fix(r"data\raw\x.PNG") == fix("data/raw/x.PNG") in tests/test_fast_head.py; loader corrected within T12 | Related to #12 |
| T12b | 2026-10-29 | Codex ? gpt-6.1-sol | Report redirection failed before running checks | Assumed ignored .agents/reports already existed in a new worktree | Note | Created the directory, reran both commands; no maintained report launcher to attach a check to | no |

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

| 27 | 2026-10-29 | Codex · gpt-6.1-sol | T9 paper-text inspection again failed on Unicode stdout | Repeated omitted UTF-8 boundary (3/10/14/21–23) | Proposed check | Corrected session encoding; persistent UTF-8 wrapper/Unicode round-trip from entry22 remains proposed, not installed pending Lam | yes |
| 28 | 2026-10-29 | Codex · gpt-6.1-sol | T9 sampler fixture incorrectly expected CSC500 not to nest in1000; several atomic patches guessed nonexistent final context | Assumed individual sample behavior and patch context before inspecting them | Note | Corrected fixture uses exact Stage3 sampler calls and later-class non-nesting; corrected patches use opened context. No extra guard installed | yes: context repeats13 |
| 29 | 2026-10-30 | Codex · gpt-6.1-sol | A direct PowerShell curl.exe literature query lost embedded quotes, returned unrelated records and reported a host-parsing error | Repeated native-command argument quoting assumption (5/10/23) | Proposed check | Extend the pending inspection wrapper with a curl argument boundary test and assert the API echoes the exact requested query before screening results; corrected lookup used Python subprocess argv and verified request.queryString. No permanent fix applied pending Lam | yes |
| 30 | 2026-10-30 | Codex · gpt-6.1-sol | Repeated the tracker's no-masks segmentation exclusion without checking the source release, and omitted the explicit C6 drop from the planning summary | Treated local/tracker absence and scope decisions as verified feasibility evidence | Proposed rule | Require each exclusion to cite either Lam's actual decision or a dated source-release/resource check; absent local files and cancelled threads alone do not establish impossibility or experiment cancellation. Mendeley v2 lists Segmentation_manual.zip; range-read directory confirms 72 PNGs, contents/mapping still unverified. No permanent rule/check applied pending Lam | no |
| 31 | 2026-10-30 | Codex · gpt-6.1-sol | A read-only NPZ inventory failed because numpy.lib.format._read_array_header is absent | Assumed a private NumPy helper was available in the installed version | Note | Corrected the temporary probe using introspected public versioned header readers; completed the inventory without loading feature values. No maintained caller or permanent prevention change added | no |
| 32 | 2026-10-30 | Codex · gpt-6.1-sol | An HTML patch failed because its paragraph was embedded in a longer line | Guessed patch line boundaries instead of deriving them from current file text (repeat of 28) | Proposed check | Corrected the temporary edit with exact current lines and assert text.count(anchor) == 1 for each replacement. A retained report-edit helper could enforce that check if approved; no permanent helper/check installed | yes |
| 33 | 2026-10-30 | Codex / gpt-6.1-sol | Colab pilot cell lookup raised TypeError before execution because a text cell lacked getCellId; later a file-provider probe treated its files property as an array | Assumed heterogeneous Colab browser objects shared a collection shape | Note | Notebook cells are resolved through the observed model getId method. The retained execution monitor reads files through Python and explicit JSON rather than the raw file-provider object. No maintained browser adapter exists here for a permanent check; no new rule or dependency installed. | yes, during full-run inspection |
| 34 | 2026-10-30 | Codex / gpt-6.1-sol | A local tracker replacement patch combined Delete File and Add File for the same path; apply_patch rejected it before mutation | Assumed replacement operations for one path could share a patch | Note | Corrected with Update File. Retained Drive tracker writes already use temporary-file replacement; there is no maintained local patch wrapper to attach a meaningful check. No permanent rule or dependency installed. | no |
| 35 | 2026-10-30 | Codex / gpt-6.1-sol | Preview wait timed out at15s during a read-only audit; orchestration continued into a cell-source update before execution settled | Treated a returned MCP isError result as if the preceding notebook action had completed | Proposed check | Abort dependent mutations on result.isError and require an idle kernel before setText/manualExecute; exact proposed guard below, not installed pending Lam. Existing audit was polled and proved exit0; training writer/data/lock untouched. | no |
| 36 | 2026-10-30 | Codex / gpt-6.1-sol | Local capture transfer exceeded Windows process command length; CreateProcess rejected it before execution | Put a large base64 artifact payload in the shell command rather than a data file | Note | Staged payload with apply_patch and decoded through a short file-based command. This is a one-off tool transport boundary, with no maintained project launcher to attach a meaningful check; no permanent rule/dependency installed. | no |
| 37 | 2026-10-30 | Codex / gpt-6.1-sol | Portable Colab timing setup raised FileNotFoundError when nvidia-smi was absent | Made optional NVIDIA metadata collection unconditional before checking TensorFlow GPU visibility | Proposed check | Exact code/test patch: .agents/reports/c23-highend-pilot/nvidia-smi-preflight.patch. Require a TensorFlow GPU in an isolated probe; collect nvidia-smi only when available. Mocked CPU, missing-tool GPU and normal GPU cases pass; git apply --check passes. Not applied pending Lam approval; no pilot launched. | no |
| 38 | 2026-11-02 | Codex / gpt-6.1-sol | One-off guide verification falsely failed a Unicode-arrow match; a later successful git diff command masked that Python failure in the shell exit code | Repeated native PowerShell stdin encoding assumption from22 and failed to propagate the first command's status | Note; existing wrapper proposal retained | Task verification was rerun successfully from the UTF-8 file .agents/reports/c23-guide/verify_guide.py using ASCII escapes. Entry22's persistent UTF-8 wrapper remains uninstalled pending Lam; proposed wrapper must also exit on nonzero child status. No AGENTS or harness change applied. | yes:22 |
| 39 | 2026-11-02 | Codex / gpt-6.1-sol | A temporary JavaScript wrapper for a read-only Git audit failed syntax parsing before any command executed | Unbalanced function-call parentheses in ad hoc orchestration | Note | functions.exec rejected the script before operations; corrected the result binding and reran the remote-SHA/bundle checks successfully. No maintained wrapper exists here for a project regression test; no permanent rule or dependency added. | no |
| 40 | 2026-11-03 | Codex · gpt-6.1-sol | FM-plan inspection passed a literal wildcard to rg, guessed an L9 filename and a manifest field, lost embedded curl quotes, and submitted two patches with stray unmatched context | Repeated shell/path/schema/context assumptions (5/7/14/22/28) | Proposed check | Extend the proposed inspection wrapper below with schema assertions and a native-argument round-trip; corrected reads use discovered paths, manifest label, rg -g, subprocess argv and opened patch context. No permanent fix installed | yes |
| 41 | 2026-11-03 | T3 Gemini · gemini-4-argon-high, audited by Codex · gpt-6.1-sol | Rebuttal R1 invented external FRR and project methods, misattributed AUC and cited the wrong Yoo paper | Reconstructed project facts despite failed reads instead of limiting conclusions to verified inputs | Proposed check | Reject unsupported claims, retain explicit dispositions in FM_FEWSHOT_PLAN.md and request corrected R2; proposed artifact-backed quantitative claim check below, not installed pending Lam | no |
| 42 | 2026-11-05 | Codex · gpt-6.1-sol | Tried reading nature-writing/core files before resolving the manifest's static/core paths | Repeated guessed-path inspection from 14/22/40 | Proposed check (existing) | Corrected reads use exact manifest paths. Entry 40's required-path inspection check remains proposed; no permanent wrapper or rule installed without Lam's approval | yes |
| 43 | 2026-11-05 | Codex · gpt-6.1-sol | Presentation verification failed before execution with a PowerShell missing-string-terminator error | Nested Python regex quotes inside a native -c argument repeated entries 5/10/15/23 | Proposed check | Rerun verification through a UTF-8 here-string on stdin. Proposed persistent change: adopt the exact .agents/inspect-python.ps1 candidate and missing-path/Unicode/quote round-trip verification above; not installed pending Lam approval. | yes |
| 44 | 2026-11-05 | Codex · gpt-6.1-sol | Initial protocol-writing orchestration cell failed JavaScript parsing before any write | Embedded Markdown backticks terminated the JavaScript template literal | Note | Reissued the temporary cell with literal text safely delimited. No maintained call site exists for a persistent check; no permanent guard or workflow change installed. | related quoting failures:5/43 |
| 45 | 2026-11-05 | Codex · gpt-6.1-sol | Two temporary document-replacement anchor checks failed before any disk mutation | Used a leftover placeholder anchor and then assumed separate instead of the actual separately | Existing check | The orchestration helper requires the old passage before replacement and checkpoints staged text; inspected actual passages and resumed. No production helper or permanent workflow fix was added. | related context assumptions:13/28 |
| 46 | 2026-11-05 | Codex · gpt-6.1-sol | Presentation replacement stopped because a previous-turn store value was unavailable | Assumed transient tool storage retained the earlier HTML instead of reading its saved file | Note | Read the existing UTF-8 file explicitly and reran the replacement; no research files or results were affected. No persistent guard installed. | no |
| 47 | 2026-11-05 | Codex / gpt-6.1-sol | Assumed Lam wanted me to choose clarification topics and offered a topic menu | Treated a request to read context for discussion as permission to set the teaching agenda | Note; rule proposed | Proposed wording: When Lam asks me to review context for questions, read it and let Lam choose what needs explanation. No permanent rule applied; conversational intent has no reliable automated check. | no |
| 48 | 2026-11-05 | Codex / gpt-6.1-sol | Literature inspection repeated PowerShell native -c quote loss and a child parser's cp1252 Unicode failure | Parent UTF-8 mode was assumed to cover a separately spawned Python process; inline source still crossed native argument parsing | Proposed check, recurrence | Corrected session commands use stdin and explicit child -X utf8. Extend the existing entry22 inspect-python wrapper proposal with a child-process Unicode round-trip; no permanent wrapper or check installed pending Lam | yes: 3/10/14/22/23/27 |
| 49 | 2026-11-05 | Codex / gpt-6.1-sol | Scratch literature builders hit truncated JSON, tuple-key JSON serialization and a variable missing from a fresh exec isolate | Assumed output capacity and transient state instead of bounding batches and explicitly loading stored values | Note | Split extraction into bounded batches, used JSON-safe keys and loaded cross-call state explicitly; artifacts are checked against the original 49-file inventory. No permanent guard applied to one-off scratch code | no |
| 50 | 2026-11-05 | Codex / gpt-6.1-sol | Starter prompt repeated the full protocol and grew to about 1,300 words; Lam asked for a shorter prompt | Confused a reusable task entry point with a self-contained protocol document | Note | Shortened the requested prompt to objective, knowledge-base pointer, outputs and completion checks. Detailed settings remain in the authoritative documents; no permanent rule/check added | no |
| 51 | 2026-11-05 | Codex · gpt-6.1-sol | IEEE drafting probed a starter file in the wrong worktree and guessed a ProtoNets BibTeX URL under /hash/ rather than following its /file/ link | Repeated cross-worktree path/lookup assumptions from 7/14/23 | Proposed check | Extend the already proposed inspection wrapper's RequiredPaths validation (entry 23); resolve local inputs from the supplied source index and remote assets from actual primary-page links. Existing candidate remains uninstalled pending Lam's approval. | yes |
| 52 | 2026-11-05 | Codex · gpt-6.1-sol, delegated citation audit | A guessed Zenodo ID returned unrelated software; the audit excluded it and recovered OCTDL's actual release. An inspection also repeated the stdout-encoding failure | Guessed record identity and repeated Windows encoding assumptions | Proposed check | Proposed metadata guard: assert the expected title and DOI before recording a dataset response as evidence; retain excluded responses separately. Reuse entry 22's proposed UTF-8 round-trip wrapper. No permanent guard installed. | yes: encoding repeats 22/23 |
| 53 | 2026-11-05 | Codex / gpt-6.1-sol | Browser click rejected role selector with unsupported exact=true | Assumed locator syntax accepted an attribute not supported by the live browser tool | Note | Recovered with observed page-model/DOM reads. No project browser-selector wrapper exists to validate; no repository guard changed. | no |
| 54 | 2026-11-05 | Codex / gpt-6.1-sol | Optional D:/Few-shot/drive mirror listing failed because the directory is absent | Assumed a documented optional mirror existed; repeats path-assumption entries 7/14/23 | Proposed check | Extend the pending inspection-wrapper proposal (22/23) with Test-Path -LiteralPath $taskMirror -PathType Container before optional mirror reads; report unavailable and use the authorized browser source. No persistent fix applied pending Lam. | yes |
| 55 | 2026-11-08 | Claude Code · claude-opus-5-5 | `gh pr merge 23 --rebase` was denied by the project tool guard | Did not read .agents/README.md first; assumed a GitHub-side merge was allowed, repeating the entry 11 risk | Note (existing check) | The guard worked. Integrated by local rebase, identity check and fast-forward push to main; no new change needed | yes: 11 |
| 56 | 2026-11-08 | Claude Code · claude-opus-5-5 | A sed edit of the LaTeX author line silently did nothing, so the first PDF rebuild still said Lam; a heredoc diagnostic had earlier turned a LaTeX macro's backslash into a newline | Double backslashes were collapsed passing through the shell tool; assumed they arrive intact | Rule | AGENTS.md Known mistakes: edit backslash text with the Edit tool and read the line back. A check cannot see every ad hoc shell edit | yes: 5/10/15/23 |
| 57 | 2026-11-09 | Codex · gpt-6.1-sol | A read-only Python version probe lost string quotes through PowerShell and raised NameError | Repeated native inline-source quoting assumption from 5/13/43/48 | Proposed check (existing) | Recovered using stdin; the exact inspection-wrapper and argument round-trip proposals in entries 22/40 remain pending Lam's approval. No permanent workflow fix installed. | yes: 5/13/43/48 |
| 58 | 2026-11-09 | Codex · gpt-6.1-sol, preparation-test subagent | A synthetic dHash boundary fixture repeated four changed comparisons over eight rows, producing 32 changed bits | Fixture geometry was assumed instead of counted | Note | Corrected the fixture to one modified row; the focused boundary assertion exposed it. No permanent harness rule/check proposed beyond the task's authorized tests. | no |
| 59 | 2026-11-09 | Codex · gpt-6.1-sol | Start-Process/WaitForExit exposed no usable ExitCode, so a failing focused test printed FAILED while the temporary PowerShell wrapper returned zero | Repeated child-exit propagation assumption from 38 | Proposed check | Use the task's temporary Python subprocess runner with explicit wait/exit status; proposed persistent inspection-wrapper verification must assert a deliberate child exit 7 is propagated as 7. Not installed pending Lam's approval. | yes: 38 |
| 60 | 2026-11-09 | Codex · gpt-6.1-sol, extraction subagent | A guessed RETFound models_mae.py source URL returned 404 | Assumed an upstream filename instead of reading the pinned repository tree | Proposed check (existing) | Resolve source paths from the pinned tree; the pending required-path/source-identity inspection proposal remains uninstalled. No conclusion used the missing source. | yes: 7/14/42 |
| 61 | 2026-11-09 | Codex · gpt-6.1-sol | A count-validation patch used stale context; a later document patch ended with a stray deletion hunk. Both were rejected before writes | Reconstructed patch context instead of checking the exact current passage | Note (existing guard) | Read the actual passages and split the corrected document patch; retained apply_patch's context check. No permanent workflow guard added. | yes: 28/45; twice this turn |
| 62 | 2026-11-09 | Codex · gpt-6.1-sol, nomination-binding subagent | A literal Windows rg glob and a resolved-path fixture assumption caused preliminary inspection/test failures | Repeated shell/path assumptions | Proposed check (existing) | Agent corrected the search and fixture paths; the pending required-path/argument round-trip inspection proposal remains uninstalled. No permanent workflow rule changed. | yes: 14/40 |
| 63 | 2026-11-09 | Codex · gpt-6.1-sol, CLI rehearsal subagent | Synthetic pilot timestamps were equal after short waits, so the production timing guard rejected the fixture | Assumed Windows clock advancement instead of asserting it | Note | The scratch constructor now waits for an observed increase and asserts a positive window before writing. No production timing rule or permanent harness change was applied. | no |
| 64 | 2026-11-09 | Codex · gpt-6.1-sol, native-transform subagent | Literal PowerShell rg globs failed again during read-only inspection | Repeated path-expansion assumption from14/62 | Proposed check (existing) | Corrected searches use directory roots and rg filters; the pending required-path/argument-boundary inspection-wrapper check remains the proposed prevention. No permanent fix installed. | yes: 14/62 |
| 65 | 2026-11-09 | Codex · gpt-6.1-sol | A header probe guessed rare_inventory.csv instead of the saved image_inventory.csv | Assumed the artifact filename before listing the preview | Proposed check (existing) | Listed the actual frozen files and used extraction_inventory.csv; the pending required-path inspection check remains uninstalled. No scientific conclusion used the missing file. | yes: 7/14/42 |

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

## Proposed preview sequencing check for entry 35 (not installed)

Add these checks to the maintained browser orchestration wrapper if approved.
They stop follow-up mutations after a tool error and while a cell is executing;
the existing process is inspected rather than restarted after a timeout.

```javascript
if (result.isError) {
  throw new Error("Preview action failed; inspect pending execution before further notebook mutations");
}
if (colab.global.notebook.isExecuting()) {
  throw new Error("Notebook execution pending; poll the existing process before editing a cell");
}
```

## Proposed quantitative-claim check for entry 41 (not installed)

Require an evidence row for every observed numerical project result, with
`source_csv`, `row_match`, `field` and `value`; planned values carry a separate
status and cannot appear as observed. A validator should perform the equivalent
of this check before accepting a reviewer result into a maintained report:

```python
records = list(csv.DictReader((root / claim["source_csv"]).open(encoding="utf-8-sig")))
matched = [r for r in records if all(r[k] == v for k, v in claim["row_match"].items())]
assert len(matched) == 1
assert math.isclose(float(matched[0][claim["field"]]), claim["value"], abs_tol=1e-10, rel_tol=0)
```

This extends the artifact-verification approach already used by
`docs/research/build_report_evidence.py`; it does not validate causal prose,
paper identity or clinical claims. The proposed fixture should accept C0's
`current` / `recomputed_axis_auc` value 0.995455 and fail a missing source,
missing column or different value. No new checker or hook was installed.

## Proposed extension for entry 40 (not installed)

Extend entry 22's inspection-wrapper verification with these exact assertions:
`assert {'path', 'label', 'pid', 'split'} <= set(csv.DictReader(manifest).fieldnames)`
for manifest consumers, and a subprocess argument round-trip that verifies
`['optical coherence tomography', '"few-shot"']` arrives unchanged. Exercise
required-path validation with a missing file and require a nonzero exit before
inspection starts. These are proposed checks of a maintained inspection wrapper,
not a new restriction on research code; Lam's approval is still required.
