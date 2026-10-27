"""Reject outgoing author/committer identities that differ from local noreply."""

import subprocess
import sys


def git(*args):
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8", stderr=subprocess.PIPE).strip()


def verify(revisions):
    expected = git("config", "--local", "user.email").lower()
    if not expected.endswith("@users.noreply.github.com"):
        raise ValueError("Repo-local user.email must be a GitHub noreply identity.")
    rows = git("log", "--format=%H%x00%ae%x00%ce", *revisions).splitlines()
    for row in rows:
        oid, author, committer = row.split("\0")
        if author.lower() != expected or committer.lower() != expected:
            raise ValueError(f"Outgoing commit {oid[:12]} has a non-repository author or committer (redacted).")
    return len(rows)


def main():
    try:
        count = 0
        if len(sys.argv) > 1 and sys.argv[1] == "--pre-push":
            remote = sys.argv[2]
            for line in sys.stdin:
                _, local_oid, _, remote_oid = line.split()
                if set(local_oid) == {"0"}:  # deleting a ref sends no commits
                    continue
                revisions = [f"{remote_oid}..{local_oid}"] if set(remote_oid) != {"0"} else [local_oid, "--not", f"--remotes={remote}"]
                count += verify(revisions)
        else:
            count = verify(["origin/main..HEAD"])
        print(f"Identity check: PASS ({count} outgoing commits; author and committer match local noreply)")
    except (ValueError, subprocess.CalledProcessError):
        print("Identity check: FAIL; verify local noreply identity and fetch the remote before retrying. Outgoing identities redacted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
