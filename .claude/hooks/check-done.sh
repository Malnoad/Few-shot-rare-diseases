#!/usr/bin/env bash
# Stop hook: before Claude finishes a turn while there are uncommitted changes,
# run the project's fast checks. If they fail, Claude gets the errors and keeps
# going (once).
input="$(cat)"

# Avoid an endless loop: if Claude is already continuing because of this hook,
# let it stop this time.
if printf '%s' "$input" | grep -q '"stop_hook_active": *true'; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# Only check when there are uncommitted changes (skip plain Q&A turns).
[ -n "$(git status --porcelain 2>/dev/null)" ] || exit 0

[ -x .agents/check ] || exit 0

if ! output="$(.agents/check 2>&1)"; then
  {
    echo "Checks in .agents/check failed:"
    printf '%s\n' "$output" | tail -n 40
    echo
    echo "Fix only failures caused by your own changes in this task."
    echo "If a failure is in code you didn't touch (e.g. the user's unfinished"
    echo "work), don't edit it: report it to the user and stop."
  } >&2
  exit 2
fi
exit 0
