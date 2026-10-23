# .agents

Agent tooling for this repo, created from `~/agent-config/project-template` on
2026-10-22. The full guide (design principles, mistake loop, skills, install) is
`~/agent-config/README.md`; this file only covers what lives here.

| File | What it does | When it runs |
|---|---|---|
| `check` | Fast checks every agent must pass (~15 s) | Claude's Stop hook (`.claude/hooks/check-done.sh`) when there are uncommitted changes; T3 "Checks" button; run by hand: `bash .agents/check` |
| `setup-worktree` | Links `data/raw`, `data/aug_basic`, `data/gan` from the main checkout (junctions on Windows); no install step, reuse the main `.venv` | T3 Code, when it creates a worktree (`t3.json`) |
| `mistakes.md` | Log of real agent mistakes and their fixes | Agents append via the `log-mistake` skill; Lam approves fixes |
| `reports/`, `pids/` | Agent scratch output and PIDs of started processes | Created on demand; git-ignored |

## What `check` verifies

1. `data/manifest.csv` is unchanged vs. `HEAD` (the splits never move; catches `stage0_manifest.py --force`).
2. Nothing under `data/` is committed except the manifest.
3. The five stage scripts compile, and `colab_setup.ipynb` is valid JSON.
4. Stage 0 verification passes, when `data/raw/OCT2017` is present.

It uses `.venv` if it exists (Windows or Linux layout), else `python3` (e.g. Colab).

## Adding a check

When a mistake can be caught by a command, add one line between "Edit below" and
"Edit above" in `check`, then drop the matching rule from "Known mistakes" in
`AGENTS.md`. Keep it fast: seconds, not minutes.

## Differences from the template

- No `/data` line in `.gitignore`: `data/manifest.csv` is tracked. The large data
  folders are already ignored individually.
- `setup-worktree` links data subfolders, because `data/` always exists here.
- The Stop hook is called through `bash` so it runs on Windows.
