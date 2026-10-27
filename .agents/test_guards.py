"""Focused checks for the permission boundary, privacy and outgoing metadata."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import tool_guard as guard


ROOT = Path(__file__).resolve().parent.parent
FAKE = "private.person@example.invalid"
FAKE_HASH = hashlib.sha256(FAKE.encode()).hexdigest()


class GuardTests(unittest.TestCase):
    def event(self, command):
        return {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(ROOT)}

    def test_fake_email(self):
        for data in (FAKE, f"EMAIL={FAKE.upper()}", {"nested": [f"https://example.invalid/?email={FAKE}"]}):
            with self.subTest(data=data):
                event = {"tool_name": "mcp__fake__lookup", "tool_input": data}
                result = guard.decision(event, "claude", FAKE_HASH)
                self.assertEqual(result[0], "deny")
                self.assertNotIn(FAKE, result[1].lower())
        self.assertIsNone(guard.decision(self.event("git status --short"), "claude", FAKE_HASH))

    def test_rm_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("reports", "pids"):
                (root / ".agents" / name).mkdir(parents=True)
            with patch.dict(os.environ, {"AGENT_TMP_ROOT": str(root / "tmp")}):
                for command in (
                    "rm -rf .agents/reports/run", "rm -fr .agents/pids/job.pid",
                    "rm -r -- .agents/reports/run .agents/pids/job.pid",
                    "rm -rf /tmp/agent-test", "rm -rf .agents/reports/*",
                    'rm -rf ".agents/reports/path with spaces"',
                ):
                    self.assertEqual(guard.rm_decision(command, root, root), "allow", command)
                for command in (
                    "rm -rf data/raw", "rm -rf .agents/reports/x data/raw",
                    "rm -rf .agents/reports/../../data/raw", "rm -rf /tmp/../home",
                    "rm -rf .agents/reports", "rm -rf /tmp", "rm -rf .agents/reports-other/x",
                    "rm -rf .agents/reports/*/data", "rm -rf .agents/reports/$TARGET",
                    "rm -rf .agents/reports/x && git status", "cd /other && rm -rf .agents/reports/x",
                    "env rm -rf .agents/reports/x", "rm -rf $(pwd)/.agents/reports/x",
                    "r''m -rf data/raw", "command r''m -rf .agents/reports/x",
                    "rm -rf '.agents/reports/x' > data/raw/x", "rm -rf .agents/reports/x\nrm -rf data",
                    "rm --no-preserve-root -rf .agents/reports/x", "rm -rf '.agents/reports/x",
                ):
                    self.assertEqual(guard.rm_decision(command, root, root), "ask", command)
                self.assertEqual(guard.rm_decision("rm -rf x", root / ".agents/reports", root), "allow")
                self.assertEqual(guard.rm_decision("rm -rf .agents/reports/x", root / "elsewhere", root), "ask")

    def test_resolved_escape(self):
        # No Windows Developer Mode needed: mock a symlink/junction resolution.
        original = Path.resolve
        target = ROOT / ".agents/reports/escape"

        def resolve(path, *args, **kwargs):
            return ROOT / "data/raw" if path == target else original(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve):
            self.assertFalse(guard.scratch_operand(str(target), ROOT))

    def test_link_and_merge_denied(self):
        for harness in ("claude", "codex"):
            for command in (
                "ln -s main/.venv .venv", "ln -sf main/.venv .venv", "ln -fs main/.venv .venv",
                "ln --symbolic main/.venv .venv", "ln '-s' main/.venv .venv",
                "gh pr merge 1 --rebase", "gh pr merge 1 --merge", "gh pr merge 1 --squash",
                "gh 'pr' 'merge' 1", "git status && gh pr merge 1",
            ):
                self.assertEqual(guard.decision(self.event(command), harness)[0], "deny", command)
        self.assertIsNone(guard.decision(self.event("git status"), "codex"))
        # Codex cannot implement ask for rm: leave its normal approval policy intact.
        self.assertIsNone(guard.decision(self.event("rm -rf data/raw"), "codex"))

    def test_hook_protocol(self):
        for harness in ("claude", "codex"):
            result = subprocess.run(
                [sys.executable, str(ROOT / ".agents/tool_guard.py"), harness],
                input=json.dumps(self.event("git status --short")), capture_output=True, text=True,
            )
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""))
            result = subprocess.run(
                [sys.executable, str(ROOT / ".agents/tool_guard.py"), harness],
                input="{invalid", capture_output=True, text=True,
            )
            self.assertEqual(json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_settings_and_no_protected_literal(self):
        settings = json.loads((ROOT / ".claude/settings.json").read_text(encoding="utf-8"))
        for rule in ("Bash(ln -s *)", "Bash(ln -sf *)", "Bash(gh pr merge *)"):
            self.assertIn(rule, settings["permissions"]["deny"])
        self.assertFalse(any(rule.startswith("Bash(rm ") for rule in settings["permissions"]["ask"]))
        for filename in (".claude/settings.json", ".codex/hooks.json"):
            groups = json.loads((ROOT / filename).read_text(encoding="utf-8"))["hooks"]["PreToolUse"]
            self.assertEqual(len(groups), 1)
            self.assertNotIn("matcher", groups[0])  # all tool calls, including MCP
            self.assertIn(".agents/tool_guard.py", groups[0]["hooks"][0]["command"])
        paths = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).decode("utf-8").split("\0")
        for name in filter(None, paths):
            path = ROOT / name
            if path.is_file():
                self.assertFalse(guard.contains_protected_email(path.read_bytes().decode("utf-8", errors="ignore")), "Protected literal in project file (path redacted)")

    def test_outgoing_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args, env=None):
                return subprocess.check_output(["git", *args], cwd=repo, env=env, stderr=subprocess.PIPE, text=True).strip()

            git("init", "--initial-branch=main")
            git("config", "user.name", "Guard fixture")
            git("config", "user.email", "fixture@users.noreply.github.com")
            git("-c", "core.hooksPath=/dev/null", "commit", "--allow-empty", "-m", "base")
            base = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/main", base)
            git("-c", "core.hooksPath=/dev/null", "commit", "--allow-empty", "-m", "safe")
            script = [sys.executable, str(ROOT / ".agents/check_identity.py")]
            self.assertEqual(subprocess.run(script, cwd=repo, capture_output=True).returncode, 0)
            safe = git("rev-parse", "HEAD")
            for variable in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
                previous = git("rev-parse", "HEAD")
                env = dict(os.environ, **{variable: "wrong@example.invalid"})
                git("-c", "core.hooksPath=/dev/null", "commit", "--allow-empty", "-m", "bad", env=env)
                bad = git("rev-parse", "HEAD")
                result = subprocess.run(script + ["--pre-push", "origin"], cwd=repo, input=f"refs/heads/main {bad} refs/heads/main {previous}\n", capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("wrong@example.invalid", result.stderr)
                # New branch uploads are checked too, excluding existing origin history.
                result = subprocess.run(script + ["--pre-push", "origin"], cwd=repo, input=f"refs/heads/test {bad} refs/heads/test {'0' * 40}\n", capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
            result = subprocess.run(script + ["--pre-push", "origin"], cwd=repo, input=f"refs/heads/main {safe} refs/heads/main {base}\n", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
