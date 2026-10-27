#!/usr/bin/env bash
# Reuse the main checkout's interpreter without copying/linking its venv.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
common="$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)"
main="$(cd "$common/.." && pwd)"
PY=python3
for p in "$root/.venv/Scripts/python.exe" "$root/.venv/bin/python" "$main/.venv/Scripts/python.exe" "$main/.venv/bin/python"; do
  if [ -x "$p" ]; then PY="$p"; break; fi
done
# Windows Python and Git Bash resolve /tmp differently.
if command -v cygpath > /dev/null; then export AGENT_TMP_ROOT="$(cygpath -m /tmp)"; fi
exec "$PY" -X utf8 "$root/$1" "${@:2}"
