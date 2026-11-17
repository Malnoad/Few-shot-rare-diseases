"""Builds colab_fm_pad.ipynb, the G1 GPU session (protocols.md section 8). Never edit the notebook by hand.

It is made the way fm_notebook.py makes the locked notebooks: the same cell() and notebook()
helpers, and the locked notebook's own verified cells for settings, code checkout and
environment, image unpacking, pinned checkpoint downloads, external downloads (published MD5)
and the credential guard, imported unchanged. New cells only restore the locked input and
external query locks from Drive, download the MIRAGE and Inception files (neither is gated, so no
token is read), run fm_pad_extract.py for the two arms and save the outputs to Drive.

    python fm_pad_notebook.py --commit <40-char commit>   # write colab_fm_pad.ipynb
    python fm_pad_notebook.py --check                      # the committed notebook is this builder's output
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import fm_notebook as nb

HERE = Path(__file__).resolve().parent
NAME = "colab_fm_pad.ipynb"
ATTEMPT = "final-20261113T170804Z"
LOCK_SHA256 = "239838279b77bb37211398a016c1c766273a35f5a06649b31f5cd97f562c001c"
EXTERNAL_SHA256 = "3414889b53ee10c526c4d3dfb28b7477676f5bf449c2729008902f93447f45d0"

INTRO = r'''# FM study, post hoc check G1: pad-to-square re-extraction (MIRAGE and Inception-v3)

One GPU session, about 1 hour (estimate, not measured). It re-extracts features for **MIRAGE and
Inception-v3 only**, with every image first padded with black to a square around its centre and then
given to the model's own locked input transform. Everything else is the locked study: the same images,
the same weights (checked against their pinned SHA256), the same code for every hashed file. It reads
**no labels and produces no scores**; scoring runs on the laptop (`fm_posthoc.py g1`).

This is a labelled post hoc sensitivity check (protocols.md section 8, G1), outside amendment A2 A-11.
It never replaces a locked result. Guide: `docs/research/FM_FIRST_RUN_GUIDE.md`, Part G1.

**Runtime:** GPU **A100, L4 or T4** (not Blackwell). No Hugging Face token is needed: MIRAGE and the
Inception weights are public downloads. Your My Drive needs the shortcut to Lam's `octfsl` folder.

| Step | What it does | What proves it worked |
|---|---|---|
| 1 | Checks the first cell and the GPU | `RUN_BY` filled in; a supported GPU |
| 2 | Clones the exact code; installs the locked Python 3.13.16 and the study environment (as the evaluate notebook) | Commit matches; `pip check` passes; PyTorch sees the GPU |
| 3 | Unpacks the images from `octfsl/raw`, then disconnects Drive | All 84,630 manifest images exist |
| 4 | Restores the locked input lock and external query lock from `octfsl/fm_v1/final/` | Both SHA256 match; lock restored |
| 5 | Downloads MIRAGE and the Inception weights | Every file matches its pinned SHA256 |
| 6 | Downloads OCTID and OCTDL (as the locked notebook) | Every file matches its published MD5 |
| 7 | Keeps Drive and credentials away from the model code | Drive is off and no login is cached |
| 8 | Pads and extracts, one process per model | `exit code 0` twice |
| 9 | Saves both feature folders to `octfsl/fm_v1/final/<attempt>/g1/`, then disconnects Drive | Every Drive copy's SHA256 matches |

**Fill in `RUN_BY` in the next cell, then Runtime → Run all.** Keep the runtime alive until Step 9 ends.
'''

FIRST = f'''# Fill in this cell before Run all (Step 1 stops until it is complete)
RUN_BY = ""  # your name
ATTEMPT = "{ATTEMPT}"  # the locked attempt; do not change
LOCK_SHA256 = "{LOCK_SHA256}"
EXTERNAL_SHA256 = "{EXTERNAL_SHA256}"
# Used by the reused external-download cell only; copied verbatim from the locked notebook's first cell.
{re.search(r"EXTERNAL_DECISION = \(.*?\)\n", nb.DECISIONS, re.S).group(0)}'''

STEP1 = r'''# Step 1. Check the first cell and the GPU
assert RUN_BY.strip() and "[" not in RUN_BY and "<" not in RUN_BY, "Stopped on purpose: write your name in RUN_BY, run that cell, then Run all."
G1_ARMS = ("MIRAGE", "Inception")  # G1 re-extracts these two arms only (ARMS stays the locked four)
G1 = "/content/g1"
query = "nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader"
gpu = subprocess.run(query, shell=True, capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU: Runtime > Change runtime type > GPU (A100, L4 or T4)"
print(gpu.stdout.strip())
assert float(gpu.stdout.split(",")[1]) < 12, "Blackwell GPU: switch to A100, L4 or T4"
'''

STEP2_GPU = r'''# Step 2b. PyTorch in the study environment sees the GPU (MIRAGE); Inception uses Colab's own TensorFlow, as at the lock
sh(f"""{PY} -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot see the GPU"; print("torch", torch.__version__, "cuda", torch.version.cuda)'""")
sh("""python3 -c 'import sys, tensorflow as tf; print("Colab python", sys.version.split()[0], "tensorflow", tf.__version__, "(the lock had 3.13.16 and 2.21.0; differences are recorded and measured by the control images)")'""")
'''

STEP4 = r'''# Step 4. Restore the locked input lock and external query lock from Drive (read-only copies)
from google.colab import drive
os.makedirs(f"{G1}/archives", exist_ok=True)
drive.mount("/content/drive", force_remount=True)
try:
    for name in ("input-lock.zip", "external-lock.zip"):
        shutil.copyfile(f"{OUT}/final/{ATTEMPT}/{name}", f"{G1}/archives/{name}")
finally:
    drive.flush_and_unmount()
assert sha256(f"{G1}/archives/input-lock.zip") == LOCK_SHA256, "input-lock.zip differs from the locked SHA256: stop and tell Lam."
assert sha256(f"{G1}/archives/external-lock.zip") == EXTERNAL_SHA256, "external-lock.zip differs from the locked SHA256: stop and tell Lam."
sh(f"{PY} {ROOT}/fm_prepare.py restore --root {ROOT} --out {G1}/lock --archive {G1}/archives/input-lock.zip --sha256 {LOCK_SHA256}")
with zipfile.ZipFile(f"{G1}/archives/external-lock.zip") as archive:
    archive.extractall(f"{G1}/external")
print("Locks restored:", LOCK_SHA256[:12], EXTERNAL_SHA256[:12])
'''

STEP5B = r'''# Step 5b. Download MIRAGE and the Inception weights (public; no token is read)
for arm, name, url, gated in FILES:
    if arm in G1_ARMS:
        assert not gated, f"{arm}/{name} needs a token; G1 must not read one"
        fetch(arm, name, url)
'''

STEP8 = r'''# Step 8. Pad and extract, one process per model, no credentials in its environment (no labels, no scores)
assert not os.path.ismount("/content/drive"), "Drive is mounted: run drive.flush_and_unmount() first."
for arm in G1_ARMS:
    out = f"{G1}/out/{arm}"
    if not os.path.exists(f"{out}/complete.json"):
        print(f"\n=== {arm} ===", flush=True)
        started = time.time()
        pid, code = worker([INTERPRETER[arm], f"{ROOT}/fm_pad_extract.py", "extract", "--root", ROOT, "--lock", f"{G1}/lock",
                            "--external", f"{G1}/external", "--arm", arm, "--assets", f"{ASSETS}/{arm}", "--out", out],
                           f"{G1}/logs/{arm}.log")
        print(f"{arm}: exit code {code} after {round(time.time() - started, 1)} s", flush=True)
        assert code == 0 and os.path.exists(f"{out}/complete.json"), (
            f"G1 extraction failed for {arm}. The log is {G1}/logs/{arm}.log. Run this step once more, changing nothing; "
            "if it fails again, run Step 9 to save the logs and tell Lam.")
'''

STEP9 = r'''# Step 9. Save both feature folders and the logs to octfsl/fm_v1/final/<attempt>/g1/, then disconnect Drive
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
archives = []
for arm in G1_ARMS:
    if os.path.exists(f"{G1}/out/{arm}/complete.json"):
        archive = f"{G1}/exports/g1-{arm}-{stamp}.zip"
        os.makedirs(os.path.dirname(archive), exist_ok=True)
        sh(f"{PY} {ROOT}/fm_prepare.py export --root {ROOT} --out {G1}/out/{arm} --archive {archive}")
        archives += [archive, archive + ".json"]
logs = f"{G1}/exports/g1-logs-{stamp}.zip"
with zipfile.ZipFile(logs, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(pathlib.Path(G1, "logs").glob("*.log")):
        archive.write(path, path.name)
save_to_drive(f"final/{ATTEMPT}/g1", [*archives, logs])
print(f"All done. Tell Lam: G1 saved in octfsl/fm_v1/final/{ATTEMPT}/g1/ with stamp {stamp}; no score exists yet.")
'''


def build(commit: str) -> str:
    # E_STEP2 (not STEP2) installs the locked Python 3.13.16, so MIRAGE runs in the environment of the lock.
    code = (FIRST, nb.SETTINGS, STEP1, nb.E_STEP2, STEP2_GPU, nb.STEP3, STEP4, nb.STEP5A, STEP5B, nb.STEP8, nb.STEP9, STEP8, STEP9)
    return nb.notebook(NAME, INTRO, code, commit, True)


def pinned_commit() -> str:
    path = HERE / NAME
    found = re.search(r'COMMIT = \\"([0-9a-f]{40})\\"', path.read_text(encoding="utf-8")) if path.exists() else None
    return found.group(1) if found else "0" * 40


def main() -> None:
    commit = sys.argv[sys.argv.index("--commit") + 1] if "--commit" in sys.argv else pinned_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("--commit needs a full 40-character commit hash")
    text, path = build(commit), HERE / NAME
    if "--check" in sys.argv:
        if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != text:
            raise SystemExit(f"{NAME} differs from fm_pad_notebook.py: rebuild it")
        print(f"{NAME} matches its builder, pinned to {commit}")
        return
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", NAME, "pinned to", commit)


if __name__ == "__main__":
    main()
