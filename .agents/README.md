# .agents

Agent tooling for this repo, created from `~/agent-config/project-template` on
2026-10-22. The full guide (design principles, mistake loop, skills, install) is
`~/agent-config/README.md`; this file only covers what lives here.

| File | What it does | When it runs |
|---|---|---|
| `check` | Fast checks every agent must pass (~15 s) | Claude's Stop hook (`.claude/hooks/check-done.sh`) when there are uncommitted changes; T3 "Checks" button; run by hand: `bash .agents/check` |
| `setup-worktree` | Links `data/raw`, `data/aug_basic`, `data/gan` from the main checkout (junctions on Windows); no install step, reuse the main `.venv` | T3 Code, when it creates a worktree (`t3.json`) |
| `mistakes.md` | Log of real agent mistakes and their fixes | Agents append via the `log-mistake` skill; Lam approves fixes |
| `tool_guard.py`, `run-python.sh` | Shared privacy and command guard, using the main venv without a link/copy | Claude and Codex `PreToolUse` hooks |
| `test_guards.py` | Fake-email, path-boundary, hook-protocol and outgoing-identity tests; scans project files for the protected email digest | `.agents/check` |
| `check_identity.py`, `hooks/pre-push` | Reject outgoing commits unless both author and committer equal repo-local GitHub noreply | `.agents/check`; every local push after hook installation |
| `reports/`, `pids/` | Agent scratch output and PIDs of started processes | Created on demand; git-ignored |

## What `check` verifies

1. `data/manifest.csv` is unchanged vs. `HEAD` (the splits never move; catches `stage0_manifest.py --force`).
2. Nothing under `data/` is committed except the manifest.
3. The five stage scripts, `analysis.py`, `review_sheet.py`, `figures.py`, `stage1b_normalize.py` and `c0_experiment.py` compile; `colab_setup.ipynb` is valid JSON; the existing `test_c0.py` suite passes.
4. Stage 0 verification passes, when `data/raw/OCT2017` is present.
5. Agent guard tests pass, including a fake email/hash, scratch-path escapes and separate author/committer violations. Project files contain no protected email literal.
6. `origin/main..HEAD` has only repo-local noreply authors and committers. Fetch first; historical commits already on main are not rewritten.

It uses this checkout's `.venv`, then the main checkout's `.venv` (Windows or
Linux layout), else `PYTHON` or `python3` (e.g. Colab), with Python UTF-8 mode enabled.
In a Windows worktree, call `D:/Few-shot/.venv/Scripts/python.exe` directly.

## Permissions and privacy (D3/D4)

Claude's three broad recursive-rm ask rules moved to `tool_guard.py` because
[Bash rules match text, not resolved paths](https://code.claude.com/docs/en/permissions#bash).
An allow wildcard cannot exclude a second operand or `..`, and an ask rule
still wins over a hook's allow. The hook approves only a single literal `rm`
command whose operands resolve below this checkout's `.agents/reports/`,
`.agents/pids/` or `/tmp/`. It checks every operand and existing glob match;
only terminal `/*` and `/**` globs are accepted. Scratch roots themselves,
parent traversal, symlink/junction escapes, shell expansions, compound commands,
unsupported flags and other paths require approval. On Windows the launcher
maps Git Bash's `/tmp` with `cygpath`; it does not confuse it with `C:/tmp`.
This is a guard for direct shell commands, not a sandbox for arbitrary scripts.

The same hook scans every tool's input strings (including nested MCP values
and keys) for email-like literals, lowercases and SHA-256 hashes candidates,
and denies a matching digest without printing the address or input. Only the
digest is stored. It was computed locally from
`git log --author=Malnoad --format=%ae 5c77ce2^`, selecting the unique historical
Gmail identity entirely in memory. Tests inject a fake address and its hash;
the production digest cannot be overridden through environment variables.
The guard also denies symbolic-link shell options and all `gh pr merge` modes.
Claude additionally denies `Bash(ln -s *)`, `Bash(ln -sf *)` and
`Bash(gh pr merge *)`; the existing Stop hook and other permission rules remain.

Verified Claude fields: `permissions.deny`, `permissions.ask`, `hooks.PreToolUse`,
`hooks`, `type: command`, `command`, `tool_name`, `tool_input`, `cwd`, and
`hookSpecificOutput` with `hookEventName`, `permissionDecision` and
`permissionDecisionReason`. Sources:
[permissions](https://code.claude.com/docs/en/permissions),
[hook reference](https://code.claude.com/docs/en/hooks).

Codex uses `.codex/hooks.json` with the shared guard. `commandWindows` selects
the main Windows venv directly; the ordinary `command` uses Git Bash/POSIX.
Verified against the [official OpenAI hook docs](https://learn.chatgpt.com/docs/hooks)
and [config reference](https://learn.chatgpt.com/docs/config-file/config-reference):
`hooks`, `PreToolUse`, handler `hooks`, `type: command`, `command`,
`commandWindows`, and the same input/deny-output fields. Installed CLI 0.164.0
reports the `hooks` feature as enabled. No invented approval configuration is
added: **Codex cannot reproduce D3's scratch allow / otherwise ask behavior**.
Its PreToolUse `ask` response is explicitly unsupported; its prefix rules
cannot express resolved-path exceptions and apply to sandbox escalation.
Codex rm therefore retains its existing approval/sandbox policy.

Review/trust the new Codex definitions in `/hooks` before relying on them;
untrusted hooks are skipped. Reload Claude's settings/hooks for existing
sessions. End-to-end blocking in a live model session is **unverified**; the
actual command handlers and documented JSON protocol are tested locally.
`codex --strict-config features list` is unsupported in this CLI, so strict
CLI validation is also unverified; no success is claimed for that probe.
Hosted tools and ongoing `write_stdin` sessions can bypass Codex PreToolUse;
encoded/constructed emails and arbitrary programs are outside this literal
input check. Hook launch failures/timeouts are not a complete security boundary.

## Outgoing commit identity

Install once per clone, after checking that there is no existing custom hook
to preserve (the setting is shared by this clone's worktrees):

```sh
git config --local core.hooksPath .agents/hooks
```

This checkout had no custom hooks before installation. Git's documented
[pre-push stdin](https://git-scm.com/docs/githooks#_pre_push) supplies each
outgoing ref range. The hook checks both identities on every outgoing commit,
including new branches, against `git config --local user.email`, which must
end in `@users.noreply.github.com`. Unknown remote objects fail closed: fetch
before retrying. Deleting a ref sends no commits. Failures never echo email
metadata. Other clones need their own installation; GitHub-side merges cannot
run local hooks, so both agent tool hooks also deny `gh pr merge`.

Integrate with a local `git fetch origin`, `git rebase origin/main`, run
`.agents/check`, verify the outgoing identities, then
`git push origin HEAD:refs/heads/main` (fast-forward only). If main moves, repeat.
GitHub rebase, squash and merge commits can introduce the account identity;
none is permitted here. Existing history is left intact.

To reverse the local installation, `git config --local --unset core.hooksPath`
restores the default hook directory. Remove a harness's PreToolUse registration
to disable that harness's guard; its independent permission rules and Git hook
still apply. [Git documents `core.hooksPath`](https://git-scm.com/docs/git-config#Documentation/git-config.txt-corehooksPath).

## Adding a check

When a mistake can be caught by a command, add one line between "Edit below" and
"Edit above" in `check`, then drop the matching rule from "Known mistakes" in
`AGENTS.md`. Keep it fast: seconds, not minutes.

## Differences from the template

- No `/data` line in `.gitignore`: `data/manifest.csv` is tracked. The large data
  folders are already ignored individually.
- `setup-worktree` links data subfolders, because `data/` always exists here.
- The Stop hook is called through `bash` so it runs on Windows.
- `.gitattributes` keeps agent shell entry points at LF even with Windows `core.autocrlf`.
