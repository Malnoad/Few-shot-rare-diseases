"""PreToolUse guard shared by Claude and Codex; never echo tool input."""

import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys


EMAIL = re.compile(r"[a-z0-9._%+\-]{1,64}@[a-z0-9.\-]{1,253}\.[a-z]{2,63}", re.I)
PROTECTED_HASH = "75b1792eb5d4bd5f0bf2c7d19e737f35e8335bee44312d3dc1be5e9e22f8916e"
ROOT = Path(__file__).resolve().parent.parent
RM = re.compile(r"(?<![\w.-])(?:[^\s'\";|&()]*/)?rm(?:\.exe)?(?=[\s'\";|&()]|$)")
LINK = re.compile(r"(?<![\w.-])(?:[^\s'\";|&()]*/)?ln(?:\.exe)?(?=\s)")
MERGE = re.compile(r"\bgh\s+pr\s+merge\b")


def shell_words(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return []


def strings(value):
    """Visit every string, including nested MCP arguments and object keys."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def contains_protected_email(value, protected_hash=PROTECTED_HASH):
    return any(
        hashlib.sha256(match.group().lower().encode("utf-8")).hexdigest() == protected_hash
        for text in strings(value)
        if "@" in text
        for match in EMAIL.finditer(text)
    )


def scratch_operand(operand, cwd, root=ROOT):
    """Resolve operands and glob matches; never approve a scratch root or escape."""
    root = root.resolve()
    cwd = cwd.resolve()
    if "\\" in operand or ".." in operand.split("/"):
        return False
    # Only terminal /* and /** globs: no wildcard may select a parent directory.
    base = operand.removesuffix("/**").removesuffix("/*")
    if any(c in base for c in "*?[]"):
        return False
    tmp = Path(os.environ.get("AGENT_TMP_ROOT", "/tmp"))
    if operand.startswith("/tmp/"):
        operand = str(tmp / operand[5:])
        base = str(tmp / base[5:]) if base.startswith("/tmp/") else str(tmp)
    roots = [root / ".agents/reports", root / ".agents/pids", tmp]
    # A replaced reports/pids directory must not expand the allowed boundary.
    if any(p.resolve() != p.absolute() for p in roots[:2]):
        return False
    candidate = Path(base)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    candidate = candidate.resolve()
    allowed = [p.resolve() for p in roots]
    is_glob = base != operand
    if not any(candidate.is_relative_to(p) and (candidate != p or is_glob) for p in allowed):
        return False
    pattern = Path(operand)
    if not pattern.is_absolute():
        pattern = cwd / pattern
    return all(
        any(Path(p).resolve().is_relative_to(boundary) for boundary in allowed)
        for p in glob.glob(str(pattern), recursive=False)
    )


def rm_decision(command, cwd, root=ROOT):
    """Auto-approve only one literal rm invocation; complex forms require review."""
    if not RM.search(command) and not any(Path(w).name in ("rm", "rm.exe") for w in shell_words(command)):
        return None
    # Reject expansion, chaining, redirection, comments and shell control syntax.
    if any(c in command for c in "\n\r;&|<>()$`\\{}~!#"):
        return "ask"
    try:
        argv = shlex.split(command)
    except ValueError:
        return "ask"
    if not argv or argv[0] not in ("rm", "/bin/rm", "/usr/bin/rm"):
        return "ask"
    operands = []
    options = True
    for arg in argv[1:]:
        if options and arg == "--":
            options = False
        elif options and arg.startswith("-"):
            if arg not in ("--recursive", "--force", "--verbose", "--dir") and not re.fullmatch(r"-[rRfvd]+", arg):
                return "ask"
        else:
            operands.append(arg)
    return "allow" if operands and all(scratch_operand(p, cwd, root) for p in operands) else "ask"


def decision(event, harness, protected_hash=PROTECTED_HASH):
    tool_input = event["tool_input"]
    if contains_protected_email(tool_input, protected_hash):
        return "deny", "Protected email detected in tool input (address redacted)."
    if event["tool_name"] not in ("Bash", "exec_command", "shell_command"):
        return None
    command = tool_input.get("command", tool_input.get("cmd", ""))
    words = shell_words(command)
    # All GitHub-side merge modes rewrite identity; local FF push is required.
    if MERGE.search(command) or any(
        Path(words[i]).name in ("gh", "gh.exe") and words[i + 1:i + 3] == ["pr", "merge"]
        for i in range(len(words))
    ):
        return "deny", "Use a local rebase, verify outgoing identities, then fast-forward push main."
    # Claude has explicit deny rules too. Cover combined/long options here.
    if (LINK.search(command) or any(Path(w).name in ("ln", "ln.exe") for w in words)) and (
        any(w == "--symbolic" or re.fullmatch(r"-[a-zA-Z]*s[a-zA-Z]*", w) for w in words)
        or re.search(r"(?:\s-[a-zA-Z]*s[a-zA-Z]*\b|\s--symbolic\b)", command)
    ):
        return "deny", "Call the main venv by path; symbolic-link commands can copy it on Windows."
    if harness == "claude":
        result = rm_decision(command, Path(event["cwd"]))
        if result:
            return result, "Only simple rm commands confined to this worktree's scratch paths or /tmp are preapproved."
    return None


def main():
    try:
        event = json.load(sys.stdin)
        result = decision(event, sys.argv[1])
    except (AttributeError, KeyError, TypeError, ValueError, OSError, RuntimeError):
        # A malformed event must not silently disable protection or expose input.
        result = "deny", "Agent guard could not validate tool input."
    if result:
        permission, reason = result
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": permission,
            "permissionDecisionReason": reason,
        }}))


if __name__ == "__main__":
    main()
