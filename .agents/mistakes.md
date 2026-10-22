# Agent mistakes log

Every real agent mistake in this project, and how we made sure it can't happen
again. Agents add entries with the `log-mistake` skill; Lam approves the fix.

**Fix levels, strongest first:**
1. **Check**: a command in `.agents/check`, a test, a permission rule, or a
   hook. The mistake is caught automatically. Prefer this whenever possible.
2. **Rule**: one line in "Known mistakes" in `AGENTS.md`. Agents read it, but
   can still slip.
3. **Note**: logged here only. Use for one-offs; if it happens again, escalate.

| # | Date | Agent / model | What went wrong | Root cause | Fix level | Fix (where) | Seen again? |
|---|---|---|---|---|---|---|---|
| 1 | YYYY-MM-DD | e.g. Codex · gpt-6.1-sol | e.g. Ran the full test suite (8 min) for a one-line change | AGENTS.md didn't name a fast check | Check | Added fast tests to `.agents/check` | no |
