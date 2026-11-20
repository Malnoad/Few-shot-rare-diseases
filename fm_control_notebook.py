"""Build colab_fm_control.ipynb for the Phase-1 E1 and E3 controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pformat
import re
import subprocess

import fm_control_extract as control
import fm_extract as extract
import fm_notebook as nb
import fm_pad_notebook as pad

HERE = Path(__file__).resolve().parent
NAME = "colab_fm_control.ipynb"

INTRO = """# FM study Phase 1 · E1 and E3

Fill in **RUN_BY**. On a **CPU** runtime run the first three code cells through **Step 0**.
Add `HF_TOKEN` to Colab Secrets, grant this notebook access, and add the `octfsl` shortcut to
My Drive. If RETFound access fails, confirm access on Hugging Face and repeat Step 0. Send the
preflight line to the agents.

For **Session 1** (~3 h), switch to an **A100 or L4** and run all with `SESSION = 1`. It renders
E1-M, E1-L and E1-C, runs E1-M for the four arms, E1-L for MIRAGE and Inception-v3,
and both E3 references and the full E3 grid. M-pad comes from the existing G1 run.

For **Session 2** (~1.5 h, optional), set `SESSION = 2` and run all on an A100 or L4.
It restores the Session 1 render archive from Drive and completes E1-C for all four arms
and E1-L for RETFound and DINOv2.

Keep `DROP_LEVEL = 0` for the full session. If GPU time runs short, drop these groups in order
by raising the level in the first cell and running from Step 7 again.

1. E1-C and E1-L for RETFound and DINOv2.
2. E1-C for MIRAGE and Inception-v3.
3. E3 R-res128 and R-bilinear.
4. E3 R-stretch and R-norm05.

Every completed worker is archived with a SHA256 receipt as soon as it exits.
Run a failed step once more with the same settings. If the worker still fails, run
**Save logs and stamp** and report the failure. Already saved outputs are reused.
The destination is `octfsl/fm_v1/final/final-20261113T170804Z/phase1/`. Send the final
**PHASE1 STAMP** line, any retries and their reasons, and confirmation of the Drive files.
Score once, after the last session, on the laptop.
"""

FIRST = pad.FIRST.replace("# Fill in this cell before Run all (Step 1 stops until it is complete)",
                          "# Fill in RUN_BY and select a session before Run all") + '''
SESSION = 1  # 1: required; 2: optional
DROP_LEVEL = 0  # drop optional groups cumulatively, in the order above


def job_id(name, arm, variant):
    return "-".join(filter(None, (name, arm, variant)))
'''

PINS = {arm: f"https://huggingface.co/{extract.REPOSITORIES[arm]}/resolve/{extract.REVISIONS[arm]}/"
        f"{extract.REQUIRED_ASSETS[arm][0]}" for arm in ("MIRAGE", "RETFound", "DINOv2")}

PREFLIGHT = r'''# Step 0. CPU is enough. Check access to the locked models and Drive; do no extraction.
from google.colab import drive, userdata
import requests


def colab_token():
    try:
        return userdata.get("HF_TOKEN")
    except Exception as exc:
        print(f"HF_TOKEN secret could not be read ({type(exc).__name__}). "
              "Check Colab Secrets access and retry if the request timed out.")
        return None


def access_check(url, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with requests.head(url, headers=headers, timeout=30, allow_redirects=False) as response:
            return ("ok" if response.status_code in (200, 206) or
                    (response.is_redirect and response.headers.get("Location")) else f"HTTP {response.status_code}")
    except requests.RequestException as exc:
        return type(exc).__name__


def preflight():
    assert RUN_BY.strip() and not any(mark in RUN_BY for mark in ("<", "[", "\n")), "Fill in RUN_BY."
    imports = subprocess.run([sys.executable, "-E", "-s", "-c", "import tensorflow, scipy, PIL"],
                             capture_output=True, text=True)
    print("Step 0 imports (tensorflow, scipy, PIL):",
          "ok" if imports.returncode == 0 else f"failed (exit {imports.returncode})\n{imports.stderr.strip()}")
    token = colab_token()
    models = {
        arm: access_check(url, token if arm == "RETFound" else None)
        if arm != "RETFound" or token else "HF_TOKEN missing or notebook access off"
        for arm, url in MODEL_HEADS.items()
    }
    gpu = shutil.which("nvidia-smi")
    result = subprocess.run([gpu, "--query-gpu=name,compute_cap,memory.total,driver_version",
                             "--format=csv,noheader"], capture_output=True, text=True) if gpu else None
    runtime = result.stdout.strip() if result and result.returncode == 0 else "CPU"
    drive.mount("/content/drive", force_remount=True)
    try:
        assert pathlib.Path(RAW).is_dir() and pathlib.Path(OUT).is_dir(), "Add the octfsl My Drive shortcut."
        free = shutil.disk_usage(OUT).free
        assert free >= 6_000_000_000, "Drive requires at least 6 GB free."
        probe = pathlib.Path(OUT, f".phase1-write-test-{time.time_ns()}")
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    finally:
        drive.flush_and_unmount()
    print(f"PHASE1 PREFLIGHT | by={RUN_BY} | commit={COMMIT} | runtime={runtime} | "
          f"Drive free={free / 1e9:.1f} GB | imports={'ok' if imports.returncode == 0 else 'failed'} | "
          + " | ".join(f"{arm}={state}" for arm, state in models.items()))
    return runtime


RUNTIME = preflight()
'''

START = r'''# Step 1. Select the plan and verify the GPU.
assert SESSION in (1, 2) and DROP_LEVEL in range(5), "Use SESSION 1 or 2 and DROP_LEVEL 0–4."
assert "A100" in RUNTIME or "L4" in RUNTIME, "Use an A100 or L4 GPU for the sessions."
PHASE = pathlib.Path("/content/phase1")
LOCK, EXTERNAL_LOCK = PHASE / "lock", PHASE / "external"
EXPORT, RENDER = PHASE / "exports", PHASE / "render"
DRIVE_FOLDER = f"final/{ATTEMPT}/phase1"
SCRIPT = f"{ROOT}/fm_control_extract.py"
'''

RESTORE_LOCKS = pad.STEP4.replace("{G1}", "{PHASE}").replace(
    "{PY} {ROOT}/fm_prepare.py restore", "{PY} -E -s {ROOT}/fm_prepare.py restore")
CHECKOUT = nb.E_STEP2.replace('print("code:", head)', '''print("code:", head)
assert pathlib.Path(SCRIPT).is_file(), "This code pin predates the Phase-1 runner. Open the rebuilt notebook."''')
STAGE_EXTERNAL = nb.STEP8.split('sh(f"{PY} {ROOT}/fm_external.py candidates', 1)[0].replace(
    "# Step 8.", "# Step 6a.", 1)
CREDENTIAL_GUARD = nb.STEP9.replace("# Step 9.", "# Step 6b.", 1)

ASSETS_CELL = r'''# Step 5b. Stage pinned weights. A download failure in one arm leaves the others runnable.
from collections import defaultdict

asset_issues = {}
assets_by_arm = defaultdict(list)
for arm, name, url, gated in FILES:
    assets_by_arm[arm].append((name, url, gated))
for arm, assets in assets_by_arm.items():
    try:
        for name, url, gated in assets:
            token = colab_token() if gated else None
            if gated and not token:
                raise ValueError("HF_TOKEN missing or notebook access off")
            fetch(arm, name, url, token)
    except (OSError, ValueError, requests.RequestException) as exc:
        asset_issues[arm] = f"weights could not be staged ({type(exc).__name__})"
        print(f"{arm}: {asset_issues[arm]}. Rerun this cell once unchanged if needed.")
    finally:
        token = None
'''

# Tiers 3 and 4 are the least essential E3 variants. Keep all reference and check workers at tier 0.
E3_DROP = {"R-res128": 3, "R-bilinear": 3, "R-stretch": 4, "R-norm05": 4}
REQUIRED = [("E1-M", arm, "", 0) for arm in control.ARMS]
REQUIRED += [("E1-L", arm, "", 0) for arm in ("MIRAGE", "Inception")]
REQUIRED += [("E3", arm, variant, E3_DROP.get(variant, 0))
             for arm, variants in control.E3_VARIANTS.items() for variant in variants]
REQUIRED.sort(key=lambda row: (row[3] != 0, -row[3]))
OPTIONAL = [("E1-C", arm, "", 2) for arm in ("MIRAGE", "Inception")]
OPTIONAL += [(name, arm, "", 1) for name in ("E1-C", "E1-L") for arm in ("RETFound", "DINOv2")]

ARCHIVES = r'''# Step 7. Verify previous exports and prepare immediate saves.
selected = [row for row in SESSION_JOBS[SESSION] if not row[3] or DROP_LEVEL < row[3]]


def archive_name(key):
    return f"phase1-{key}.zip"


def receipt_path(archive):
    return archive.with_suffix(".zip.json")


saved = {}
pending_saves = globals().get("pending_saves", {})
EXPORT.mkdir(parents=True, exist_ok=True)
drive.mount("/content/drive", force_remount=True)
try:
    folder = pathlib.Path(OUT, DRIVE_FOLDER)
    keys = ["render", *(job_id(name, arm, variant) for name, arm, variant, _ in selected)]
    for key in keys:
        remote = folder / archive_name(key)
        receipt = receipt_path(remote)
        if remote.is_file() != receipt.is_file():
            orphan = remote if remote.is_file() else receipt
            orphan.unlink()
            print(f"{key}: removed unpaired Drive file {orphan.name}")
        elif remote.is_file():
            expected = json.loads(receipt.read_text(encoding="utf-8"))["sha256"]
            assert sha256(remote) == expected, f"{remote.name} differs from its Drive receipt."
            saved[key] = expected
            if key == "render" and not (RENDER / "complete.json").is_file():
                local = EXPORT / remote.name
                shutil.copyfile(remote, local)
                assert sha256(local) == expected, "Local render archive differs from its Drive receipt."
                pending = PHASE / "render-restore"
                pending.mkdir(exist_ok=True)
                with zipfile.ZipFile(local) as archive:
                    members = archive.namelist()
                    assert len(members) == len(set(members)) and "complete.json" in members
                    assert all(not pathlib.PurePosixPath(name).is_absolute() and
                               ".." not in pathlib.PurePosixPath(name).parts for name in members)
                    archive.extractall(pending)
                assert not RENDER.exists(), "The local render folder is incomplete. Use a fresh runtime."
                pending.rename(RENDER)
finally:
    drive.flush_and_unmount()


def save_output(key, output):
    archive = EXPORT / archive_name(key)
    receipt = receipt_path(archive)
    if not archive.is_file() or not receipt.is_file():
        sh(shlex.join([PY, "-E", "-s", f"{ROOT}/fm_prepare.py", "export", "--root", ROOT,
                       "--out", str(output), "--archive", str(archive)]))
    expected = json.loads(receipt.read_text(encoding="utf-8"))["sha256"]
    assert sha256(archive) == expected, f"{archive.name} differs from its local receipt."
    save_to_drive(DRIVE_FOLDER, [receipt, archive])
    saved[key] = expected


def try_save(key, output):
    try:
        save_output(key, output)
    except Exception as exc:
        pending_saves[key] = output
        message = f"{key}: Drive save failed ({type(exc).__name__}: {exc})"
        print(message, flush=True)
        logs = PHASE / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / f"{key}-save.log").open("a", encoding="utf-8") as log:
            log.write(message + "\n")
    else:
        pending_saves.pop(key, None)
'''

RENDER_CELL = r'''# Step 8. Session 1 renders twice; Session 2 uses the verified render of record.
def run_step(key, command):
    """Run at most twice unchanged in this runtime and keep both logs."""
    assert not os.path.ismount("/content/drive"), "Unmount Drive before running workers."
    logs = PHASE / "logs"
    for attempt in (1, 2):
        log = logs / f"{key}-try{attempt}.log"
        if not log.exists():
            print(f"{key}: attempt {attempt}", flush=True)
            _, code = worker(command, str(log))
            return code == 0
    print(f"{key}: two attempts recorded; see the logs.")
    return False


if not (RENDER / "complete.json").is_file():
    assert SESSION == 1, "Session 2 requires the Session 1 render archive and receipt on Drive."
    assert run_step("render", [PY, "-E", "-s", SCRIPT, "render", "--root", ROOT, "--lock", str(LOCK),
                               "--external", str(EXTERNAL_LOCK), "--out", str(RENDER)]), "See render logs."
if "render" not in saved:
    try_save("render", RENDER)
'''

EXTRACT_CELL = r'''# Step 9. Run one arm and feature set per process. Save each result before continuing.
assert (RENDER / "complete.json").is_file(), "Finish the render step before extraction."
for name, arm, variant, _ in selected:
    key = job_id(name, arm, variant)
    if key in saved:
        print(f"{key}: verified on Drive")
        continue
    if arm in asset_issues:
        print(f"{key}: {asset_issues[arm]}")
        continue
    output = PHASE / "out" / name / arm
    if variant:
        output /= variant
    if not (output / "complete.json").is_file():
        command = [INTERPRETER[arm], "-E", "-s", SCRIPT, "extract", "--root", ROOT, "--lock", str(LOCK),
                   "--external", str(EXTERNAL_LOCK), "--arm", arm, "--set", name, "--assets", f"{ASSETS}/{arm}",
                   "--out", str(output)]
        command += ["--variant", variant] if variant else ["--render", str(RENDER)]
        if not run_step(key, command):
            tried_twice = (PHASE / "logs" / f"{key}-try2.log").is_file()
            print(f"{key}: worker failed twice; report the logs." if tried_twice else
                  f"{key}: worker failed. Rerun this cell unchanged to use the single retry.")
            continue
    try_save(key, output)
'''

STAMP = r'''# Step 10. Save logs and stamp, including on a partial run.
for key, output in list(globals().get("pending_saves", {}).items()):
    try_save(key, output)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
jobs = SESSION_JOBS[SESSION]
keys = [job_id(name, arm, variant) for name, arm, variant, tier in jobs if not tier or DROP_LEVEL < tier]
saved = globals().get("saved", {})
missing = (["render"] if "render" not in saved else []) + [key for key in keys if key not in saved]
dropped = [job_id(name, arm, variant) for name, arm, variant, tier in jobs if tier and tier <= DROP_LEVEL]
completed = ", ".join(saved) or "none"
line = (f"PHASE1 STAMP | {stamp} | session={SESSION} | by={RUN_BY} | GPU={RUNTIME} | commit={COMMIT} | "
        f"completed={completed} | missing={', '.join(missing) or 'none'} | dropped={', '.join(dropped) or 'none'}")
logs = EXPORT / f"phase1-logs-s{SESSION}-{stamp}.zip"
EXPORT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(logs, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((PHASE / "logs").glob("*.log")):
        archive.write(path, path.name)
    if pathlib.Path(LEDGER).is_file():
        archive.write(LEDGER, "download_ledger.csv")
    archive.writestr("stamp.txt", line + "\n")
    archive.writestr("asset_issues.json", json.dumps(globals().get("asset_issues", {}), indent=2) + "\n")
receipt = logs.with_suffix(".zip.json")
receipt.write_text(json.dumps({"sha256": sha256(logs)}, indent=2) + "\n", encoding="utf-8")
print(line)
print("Score once, after the last session, on the laptop.")
save_to_drive(DRIVE_FOLDER, [receipt, logs])
'''


def build(commit: str) -> str:
    preflight = f"MODEL_HEADS = {PINS!r}\n" + PREFLIGHT
    settings = nb.SETTINGS + "\nSESSION_JOBS = " + pformat({1: REQUIRED, 2: OPTIONAL}) + "\n"
    steps = (FIRST, settings, preflight, START, CHECKOUT, pad.STEP2_GPU, nb.STEP3, RESTORE_LOCKS,
             nb.STEP5A, ASSETS_CELL, STAGE_EXTERNAL, CREDENTIAL_GUARD, ARCHIVES, RENDER_CELL, EXTRACT_CELL, STAMP)
    return nb.notebook(NAME, INTRO, steps, commit, True)


def pinned_commit(path: Path = HERE / NAME) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        match = re.search(r'^COMMIT = "([0-9a-f]{40})"', "".join(cell["source"]), re.MULTILINE)
        if match:
            return match[1]
    raise ValueError(f"{path.name} has no pinned commit")


def check_pin(commit: str) -> None:
    """Require the pinned checkout to contain these exact worker and helper bytes."""
    stale = []
    for name in ("fm_control_extract.py", "fm_pad_extract.py", "fm_notebook.py"):
        shown = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=HERE, capture_output=True)
        if shown.returncode or shown.stdout != (HERE / name).read_bytes():
            stale.append(name)
    if stale:
        raise SystemExit(f"Phase-1 notebook pin {commit} is missing or differs in: {', '.join(stale)}. "
                         "Rebuild with --commit after the worker is pushed.")
    print(f"Phase-1 pin ok: {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--commit", help="commit with the Phase-1 runner on GitHub")
    mode.add_argument("--check", action="store_true", help="verify the generated notebook and its code pin")
    args = parser.parse_args()
    path = HERE / NAME
    commit = pinned_commit(path) if args.check else args.commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        parser.error("--commit needs a full 40-character lowercase commit hash")
    contents = build(commit)
    if args.check:
        if path.read_text(encoding="utf-8") != contents:
            raise SystemExit(f"{NAME} differs from its builder. Rebuild it with --commit {commit}.")
        check_pin(commit)
        print(f"{NAME} matches its builder, pinned to {commit}")
    else:
        path.write_text(contents, encoding="utf-8", newline="\n")
        print(f"Wrote {NAME}, pinned to {commit}")


if __name__ == "__main__":
    main()
