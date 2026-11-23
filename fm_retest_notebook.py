"""Build colab_fm_retest.ipynb for the re-tests RT1 and RT2 (protocols.md section 13). Never edit the notebook by hand.

It is made the way fm_control_notebook.py makes the Phase-1 notebook: the locked notebook's verified cells
(settings, checkout and environment, image unpacking, pinned downloads, external downloads, credential guard)
and the Phase-1 cells for the preflight, the lock restore, the archives with receipts and the stamp, reused by
import. New cells only check the rule, render the RT2 inputs and run fm_retest_extract.py.

While section 13 is a DRAFT the committed notebook pins the all-zero commit and cannot run. A runnable build
needs the pushed code commit and the commit that fixed the approved rule, and is refused otherwise.

    python fm_retest_notebook.py --draft                                  # the not-runnable draft notebook
    python fm_retest_notebook.py --commit <40 hex> --rule-commit <40 hex>   # after approval and push
    python fm_retest_notebook.py --check                                  # the notebook is this builder's output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pformat
import re
import subprocess

import fm_control_notebook as cn
import fm_notebook as nb
import fm_pad_notebook as pad
import fm_retest_extract as retest

HERE = Path(__file__).resolve().parent
NAME = "colab_fm_retest.ipynb"
DRAFT_PIN = "0" * 40
RULE_HEADING = "## 13. Re-tests RT1"  # as fm_posthoc.RT_HEADING: the heading's start; the rest of the line is its state
WORKER_FILES = ("fm_retest_extract.py", "fm_control_extract.py", "fm_pad_extract.py")


def swap(text: str, *pairs: tuple[str, str]) -> str:
    """Replace each old text exactly once, so a change in a reused cell stops the build instead of passing silently."""
    for old, new in pairs:
        if text.count(old) != 1:
            raise ValueError(f"reused cell changed: expected exactly one {old!r}")
        text = text.replace(old, new)
    return text


INTRO = """# FM study · re-tests RT1 and RT2 (post hoc)

**This notebook cannot run while it pins commit `000…0`.** Section 13 of `protocols.md` is a draft rule.
After Lam approves it, the rule and this code are pushed and an agent rebuilds the notebook with both
commits; Step 2a then checks them before anything is downloaded.

One session on an **A100 or L4**, about 2.5 h (an estimate from the Phase-1 timings, not measured): setup up
to 1 h; RT2 about 70 min (Inception-v3 needs about 34 min for all 5,927 images, and there are two sets); RT1
about 20 min. It reads no label and computes no score. Scoring runs once, on the laptop.

Fill in **RUN_BY**. Add `HF_TOKEN` to Colab Secrets and grant this notebook access (RETFound is gated), keep
the `octfsl` shortcut in My Drive, then **Runtime → Run all**.

| Step | What it does | What proves it worked |
|---|---|---|
| 0 | Checks the token, the RETFound page and Drive | The `RETEST PREFLIGHT` line |
| 1–2 | Checks the GPU, clones the pinned code, builds the locked Python environment | Commit matches; `pip check` passes |
| 2a | Rules before numbers: the pinned commit is on `origin` and holds the approved rule | `rule ok` |
| 3–6 | Unpacks the images, restores both locks, downloads RETFound, Inception-v3, OCTID and OCTDL | Every SHA256 and MD5 matches |
| 6c | Repeats the checks in one line | The `RETEST CHECKS` line |
| 7–8 | Verifies earlier saves; decodes every image with PIL and saves it as PNG (RT2 inputs) | `render` saved |
| 9 | RT2 first (I-pil, then the control I-ref), then RT1 (R-ref, the upstream check, three read-outs) | Each worker saved before the next starts |
| 10 | Saves the logs and prints the stamp | The `RETEST STAMP` line |

Keep `DROP_LEVEL = 0`. With `DROP_LEVEL = 1`, RT2 leaves out OCTID and OCTDL (about 10 min less) and its
nine-way table is not scored. Choose before the first run: a second run does not add them.

Every completed worker is archived with a SHA256 receipt as soon as it exits. Run a failed step once more
with the same settings. If it fails again, run **Save logs and stamp** and report the failure. The
destination is `octfsl/fm_v1/final/final-20261113T170804Z/retest/`. Send the **RETEST STAMP** line, any
retries and their reasons, and confirmation of the Drive files.
"""

FIRST = swap(pad.FIRST, ("# Fill in this cell before Run all (Step 1 stops until it is complete)",
                         "# Fill in RUN_BY before Run all")) + '''
DROP_LEVEL = 0  # 1: RT2 without OCTID and OCTDL (its nine-way table is then not scored)
RULE_COMMIT = "@RULE@"  # the commit that fixed the approved re-test rule (protocols.md section 13)


def job_id(name, arm, variant):
    return "-".join(filter(None, (name, arm, variant)))
'''

# RT2 first, then RT1; every job is required. (re-test, arm, variant, drop tier as the reused Phase-1 cells expect)
JOBS = [("RT2", retest.RT2_ARM, "I-pil", 0), ("RT2", retest.RT2_ARM, retest.RT2_REF, 0),
        ("RT1", retest.RT1_ARM, retest.RT1_REF, 0), ("RT1", retest.RT1_ARM, retest.UPSTREAM, 0),
        *[("RT1", retest.RT1_ARM, variant, 0) for variant in ("R-meanraw", "R-clsnorm", "R-meannorm")]]
if {job[2] for job in JOBS} != {*retest.RT1_VARIANTS, *retest.RT2_VARIANTS} or len(JOBS) != len(retest.VARIANTS):
    raise ValueError("the notebook jobs differ from the re-test registry")

PINS = {retest.RT1_ARM: cn.PINS[retest.RT1_ARM]}  # the only gated model here
PREFLIGHT = swap(cn.PREFLIGHT, ("PHASE1 PREFLIGHT", "RETEST PREFLIGHT"), (".phase1-write-test-", ".retest-write-test-"))

START = r'''# Step 1. Select the plan and verify the GPU.
assert DROP_LEVEL in (0, 1), "Use DROP_LEVEL 0 or 1."
assert "A100" in RUNTIME or "L4" in RUNTIME, "Use an A100 or L4 GPU."
PHASE = pathlib.Path("/content/retest")
LOCK, EXTERNAL_LOCK = PHASE / "lock", PHASE / "external"
EXPORT, RENDER, PNG = PHASE / "exports", PHASE / "render", PHASE / "png"
PNG_READY = PHASE / "png.ok"
DRIVE_FOLDER = f"final/{ATTEMPT}/retest"
SCRIPT = f"{ROOT}/fm_retest_extract.py"
RT_ARMS = ("Inception", "RETFound")
'''

CHECKOUT = swap(cn.CHECKOUT, ("This code pin predates the Phase-1 runner.", "This code pin predates the re-test runner."))

RULE_CELL = r'''# Step 2a. Rules before numbers: the pinned commit is on origin and holds the approved, pushed re-test rule.
contains = subprocess.check_output(["git", "-C", ROOT, "branch", "-r", "--contains", COMMIT], text=True)
branches = [line.strip() for line in contains.splitlines() if line.strip() and "->" not in line]
assert branches, "The pinned commit is on no origin branch. Stop and tell Lam."
headings = [line for line in pathlib.Path(ROOT, "protocols.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("@HEADING@")]
assert len(headings) == 1 and "DRAFT" not in headings[0].upper(), (
    "The re-test rule (protocols.md section 13) is missing or still a DRAFT at this commit. Stop: no worker may run.")
added = subprocess.check_output(["git", "-C", ROOT, "log", "--reverse", "--format=%H", "-S", headings[0], "--",
                                 "protocols.md"], text=True).split()
assert added and added[0] == RULE_COMMIT, "RULE_COMMIT is not the commit that fixed the approved rule. Stop and tell Lam."
assert subprocess.run(["git", "-C", ROOT, "merge-base", "--is-ancestor", RULE_COMMIT, COMMIT]).returncode == 0, (
    "The rule commit is not part of the pinned code. Stop and tell Lam.")
RULE_LINE = f"commit {COMMIT[:7]} on {branches[0]} | rule {RULE_COMMIT[:7]}: {headings[0]}"
print("rule ok |", RULE_LINE)
'''.replace("@HEADING@", RULE_HEADING)

ASSETS_CELL = swap(cn.ASSETS_CELL, ("asset_issues = {}\n", "asset_issues = {}\n"
                                    "FILES = [row for row in FILES if row[0] in RT_ARMS]  # RETFound and Inception-v3 only\n"))

CHECKS = r'''# Step 6c. The checks in one line, before any worker: code and rule, both locks, the weights.
for name, expected in (("input-lock.zip", LOCK_SHA256), ("external-lock.zip", EXTERNAL_SHA256)):
    assert sha256(f"{PHASE}/archives/{name}") == expected, f"{name} differs from the locked SHA256: stop and tell Lam."
for arm, name, _, _ in FILES:
    if arm not in asset_issues:
        assert sha256(f"{ASSETS}/{arm}/{name}") == pins["digests"][arm][name], f"{arm}/{name} differs from its pinned SHA256."
print(f"RETEST CHECKS | {RULE_LINE} | locks {LOCK_SHA256[:12]} {EXTERNAL_SHA256[:12]} | weights "
      + ", ".join(f"{arm}={asset_issues.get(arm, 'verified')}" for arm in RT_ARMS))
'''

ARCHIVES = swap(cn.ARCHIVES, ('return f"phase1-{key}.zip"', 'return f"retest-{key}.zip"'))

RUN_STEP = swap(cn.RENDER_CELL.split("\n\nif not (RENDER", 1)[0],
                ("# Step 8. Session 1 renders twice; Session 2 uses the verified render of record.",
                 "# Step 8. RT2 inputs (CPU): every image decoded by PIL and saved as PNG; only the receipts are archived."))
RENDER_CELL = RUN_STEP + r'''


if not PNG_READY.is_file():  # also rebuilds the PNGs when the receipts were restored from Drive, and compares them
    assert run_step("render", [PY, "-E", "-s", SCRIPT, "render", "--root", ROOT, "--lock", str(LOCK),
                               "--external", str(EXTERNAL_LOCK), "--out", str(RENDER), "--png", str(PNG)]), "See render logs."
    PNG_READY.write_text("ok\n", encoding="utf-8")
if "render" not in saved:
    try_save("render", RENDER)
'''

EXTRACT_CELL = r'''# Step 9. One set or read-out per process: RT2 first, then RT1. Save each result before continuing.
assert PNG_READY.is_file(), "Finish the render step before extraction."
for name, arm, variant, _ in selected:
    key = job_id(name, arm, variant)
    if key in saved:
        print(f"{key}: verified on Drive")
        continue
    if arm in asset_issues:
        print(f"{key}: {asset_issues[arm]}")
        continue
    output = PHASE / "out" / name / variant
    if not (output / "complete.json").is_file():
        command = [INTERPRETER[arm], "-E", "-s", SCRIPT, "extract", "--root", ROOT, "--lock", str(LOCK),
                   "--external", str(EXTERNAL_LOCK), "--variant", variant, "--assets", f"{ASSETS}/{arm}",
                   "--out", str(output)]
        if variant == "I-pil":
            command += ["--render", str(RENDER), "--png", str(PNG)]
        if name == "RT2" and DROP_LEVEL:
            command.append("--main-only")
        if not run_step(key, command):
            tried_twice = (PHASE / "logs" / f"{key}-try2.log").is_file()
            print(f"{key}: worker failed twice; report the logs." if tried_twice else
                  f"{key}: worker failed. Rerun this cell unchanged to use the single retry.")
            continue
    try_save(key, output)
'''

STAMP = swap(cn.STAMP,
             ("PHASE1 STAMP | {stamp} | session={SESSION} | by=", "RETEST STAMP | {stamp} | drop_level={DROP_LEVEL} | by="),
             ("commit={COMMIT} | ", "commit={COMMIT} | rule={RULE_COMMIT} | "),
             ('f"phase1-logs-s{SESSION}-{stamp}.zip"', 'f"retest-logs-{stamp}.zip"'),
             ('print("Score once, after the last session, on the laptop.")',
              'print("Send this line to the agents. Scoring runs once, on the laptop (fm_posthoc.py rt).")'))


def build(commit: str, rule: str) -> str:
    settings = nb.SETTINGS + "\n# One session; the reused Phase-1 cells read these two names.\nSESSION, SESSION_JOBS = 1, {1: " \
        + pformat(JOBS) + "}\n"
    preflight = f"MODEL_HEADS = {PINS!r}\n" + PREFLIGHT
    steps = (swap(FIRST, ("@RULE@", rule)), settings, preflight, START, CHECKOUT, RULE_CELL, pad.STEP2_GPU, nb.STEP3,
             cn.RESTORE_LOCKS, nb.STEP5A, ASSETS_CELL, cn.STAGE_EXTERNAL, cn.CREDENTIAL_GUARD, CHECKS, ARCHIVES,
             RENDER_CELL, EXTRACT_CELL, STAMP)
    return nb.notebook(NAME, INTRO, steps, commit, True)


def pinned(path: Path = HERE / NAME) -> tuple[str, str]:
    """The code commit and the rule commit the notebook pins."""
    source = "".join(line for cell in json.loads(path.read_text(encoding="utf-8"))["cells"] for line in cell["source"])
    found = [re.search(rf'^{name} = "([0-9a-f]{{40}})"', source, re.MULTILINE) for name in ("COMMIT", "RULE_COMMIT")]
    if not all(found):
        raise ValueError(f"{path.name} has no pinned commit or rule commit")
    return found[0][1], found[1][1]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True)


def check_pin(commit: str, rule: str) -> None:
    """A runnable pin must hold these exact worker bytes and the approved rule, fixed by `rule`. The draft pins
    (all zeros) are reported as not runnable; one zero pin beside a real one is an error."""
    if DRAFT_PIN in (commit, rule):
        if commit != rule:
            raise SystemExit("pin both the code commit and the rule commit, or neither")
        print("draft pins: the notebook cannot run until the rule is approved and it is rebuilt with --commit and --rule-commit")
        return
    stale = []
    for name in WORKER_FILES:
        shown = _git("show", f"{commit}:{name}")
        if shown.returncode or shown.stdout.replace(b"\r\n", b"\n") != (HERE / name).read_bytes().replace(b"\r\n", b"\n"):
            stale.append(name)
    if stale:
        raise SystemExit(f"re-test notebook pin {commit} is missing or differs in: {', '.join(stale)}")
    protocol = _git("show", f"{commit}:protocols.md")
    headings = [line for line in protocol.stdout.decode("utf-8").splitlines() if line.startswith(RULE_HEADING)]
    if protocol.returncode or len(headings) != 1 or "DRAFT" in headings[0].upper():
        raise SystemExit(f"at {commit} the re-test rule (protocols.md section 13) is missing or still a DRAFT: "
                         "no runnable notebook is built")
    added = _git("log", "--reverse", "--format=%H", "-S", headings[0], commit, "--", "protocols.md").stdout.decode().split()
    if not added or added[0] != rule:
        raise SystemExit(f"--rule-commit {rule} is not the commit that fixed the approved rule heading")
    print(f"re-test pin ok: {commit}, rule {rule}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true", help="write the not-runnable draft notebook (all-zero pins)")
    mode.add_argument("--commit", help="pushed commit with the re-test runner and the approved rule")
    mode.add_argument("--check", action="store_true", help="verify the generated notebook and its pins")
    parser.add_argument("--rule-commit", help="the commit that fixed the approved rule heading (with --commit)")
    args = parser.parse_args()
    path = HERE / NAME
    if args.check:
        commit, rule = pinned(path)
    elif args.draft:
        commit = rule = DRAFT_PIN
    else:
        commit, rule = args.commit, args.rule_commit or ""
    if bool(args.rule_commit) != bool(args.commit) or not all(re.fullmatch(r"[0-9a-f]{40}", pin) for pin in (commit, rule)):
        parser.error("--commit and --rule-commit go together, each a full 40-character lowercase commit hash")
    contents = build(commit, rule)
    if args.check:
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") != contents:
            raise SystemExit(f"{NAME} differs from its builder. Rebuild it.")
        check_pin(commit, rule)
        print(f"{NAME} matches its builder, pinned to {commit} (rule {rule})")
    else:
        check_pin(commit, rule)
        path.write_text(contents, encoding="utf-8", newline="\n")
        print(f"Wrote {NAME}, pinned to {commit} (rule {rule})")


if __name__ == "__main__":
    main()
