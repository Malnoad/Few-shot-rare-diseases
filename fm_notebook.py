"""Generate the two Colab notebooks of the frozen-encoder study.

    python fm_notebook.py            write colab_fm.ipynb and colab_fm_evaluate.ipynb
    python fm_notebook.py --check    fail if the committed notebooks differ from this file
    python fm_notebook.py --commit SHA   also pin the code commit both notebooks clone
    python fm_notebook.py --check-pin    fail if that commit's hashed files differ from this checkout

colab_fm.ipynb           one GPU session: final pilots, input lock, external screen, extraction.
colab_fm_evaluate.ipynb  CPU only: development, nomination, reports (the first labelled scores).

Cell text lives here so backslashes never pass through a shell. This file and the notebooks
are not hashed into the input lock; the scripts they call are. Tested by tests/test_fm_notebook.py.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = r'''# FM study: final pilots, input lock and feature extraction

One GPU session, about 1 to 1.5 hours (estimate, not yet measured). It loads the four image models,
times them, **locks the study's inputs**, downloads the two external datasets, and saves every
model's features to Lam's Drive. It reads **no labels and produces no scores**; scores come later
from `colab_fm_evaluate.ipynb`. The rules are in `docs/research/FM_RUNBOOK.md`.

**Step-by-step guide for whoever runs this:** `docs/research/FM_FIRST_RUN_GUIDE.md`.

**Runtime:** Runtime → Change runtime type → **GPU: A100, L4 or T4**, not RTX PRO 6000 / Blackwell.
**Before you start:** `HF_TOKEN` (a Hugging Face account with RETFound access) must be in Colab Secrets
(key icon) with notebook access on, and your My Drive must have a shortcut to Lam's `octfsl` folder
(guide, Part 1). Lam does not need to run it.

| Step | What it does | What proves it worked |
|---|---|---|
| 1 | Checks the first cell and the GPU | First cell complete; a supported GPU is attached |
| 2 | Clones the exact code and builds a separate Python environment | Commit matches; `pip check` passes; PyTorch sees the GPU |
| 3 | Checks Drive, unpacks the images from `octfsl/raw`, then disconnects Drive | `fm_v1` is writable; all 84,630 manifest images exist |
| 4 | Rebuilds the groups, episodes and rosters | Six files match the laptop's byte for byte |
| 5 | Downloads RETFound, DINOv2, MIRAGE and the Inception reference | Every file matches its pinned SHA256 |
| 6 | Prints the model inventory | Every file verified |
| 7 | Records each model's identity and Lam's terms decision (pre-filled in the next cell) | One spec file per model |
| 8 | Downloads OCTID and OCTDL (external images) | Every file matches its published MD5 |
| 9 | Keeps Drive and credentials away from the model code | Drive is off and no login is cached |
| 10 | Final pilots: loads each model and times it on 100 images, one process at a time | Every process exits cleanly |
| 11 | Prints the pilot report, with the repeat checks | Four model lines |
| 12 | Makes the **input lock** | One SHA256, printed in a box |
| 13 | Saves the lock and its hash file to `octfsl/fm_v1/final/` | The Drive copy's SHA256 matches |
| 14 | Screens the external images against the locked ones and fixes the external query list | Counts per source |
| 15 | Extracts features for every model, study images and external images | `exit code 0` eight times |
| 16 | Saves features and evidence to `octfsl/fm_v1/final/`, then disconnects Drive | Every Drive copy's SHA256 matches |

**Fill in `RUN_BY` and `RETFOUND_ACCESS` in the next cell, then Runtime → Run all.** Any failed check stops
the notebook; Step 1 stops on purpose while the first cell is incomplete or still holds a placeholder.
Files live in `/content` and vanish when the runtime ends, so **keep this runtime alive until Step 16
has finished**. Rerunning a finished step is safe: it is skipped or repeats the same check.

After Step 16, send Lam the attempt name and the input-lock SHA256. Lam approves that exact SHA256
and records the public timestamp before anyone opens `colab_fm_evaluate.ipynb`.
'''

DECISIONS = r'''# Fill in this cell before Run all (Step 1 stops until it is complete)
# DECISIONS: Lam's terms decision for each model, as Lam finalized it on 2026-11-10 (recorded in
# docs/research/FM_TERMS_EVIDENCE.md, section 4). Do not edit: only Lam changes these.
DECISIONS = {
    "RETFound": "Lam, 2026-11-10. I approve the pinned RETFound_mae_natureOCT weights and models_vit.py as a "
                "frozen encoder for feature extraction and timing in our non-commercial student research, "
                "run by me or a teammate. Licence: CC BY-NC 4.0 for the code and, per the model card I read "
                "on 2026-11-10, for the weights. Each runner uses their own approved Hugging Face account. "
                "Tokens and weights are never shared, saved to Drive, redistributed or committed. "
                "We will cite Zhou et al. 2023 (Nature).",
    "DINOv2": "Lam, 2026-11-10. I approve the pinned facebook/dinov2-base as a frozen encoder for feature "
              "extraction and timing in our non-commercial student research, run by me or a teammate. "
              "Licence: Apache-2.0. The weights are not redistributed or committed. We will cite "
              "Oquab et al. 2024 (TMLR).",
    "MIRAGE": "Lam, 2026-11-10. I approve the pinned MIRAGE-Base weights and mirage_hf.py as a frozen, OCT-only "
              "encoder for feature extraction and timing in our non-commercial student research, run by me "
              "or a teammate. Licence: CC BY-NC 4.0 (non-commercial academic research with attribution). "
              "Weights and code are not redistributed or committed. We will cite Morano et al. 2025 "
              "(npj Digital Medicine).",
    "Inception": "Lam, 2026-11-10. I approve the pinned Keras ImageNet no-top Inception-v3 weights as the "
                 "reference arm only, never nominated, for feature extraction and timing in our "
                 "non-commercial student research, run by me or a teammate. Licence: Apache-2.0 for the "
                 "Keras code and, per the Keras 2.2.4 documentation, for these weights. The weights are "
                 "not redistributed or committed.",
}
# Lam's decision on the external datasets (FM_TERMS_EVIDENCE.md, section 6, item 2). Do not edit.
EXTERNAL_DECISION = ('Lam, 2026-11-11: "Approve" (download OCTID and OCTDL; cohorts fixed by amendment A1, '
                     "12 November 2026).")
# Only Lam fills this, and only after two failed pilot sequences (amendment A3, S-3):
# {"MIRAGE": "reason, with the measured values"}. An arm listed here is locked as unavailable.
UNAVAILABLE = {}
# Your name (the person running this notebook).
RUN_BY = ""
# Whose Hugging Face account the HF_TOKEN belongs to, and the date its RETFound access was approved.
RETFOUND_ACCESS = ""
'''

SETTINGS = r'''# Settings and helpers
import csv, datetime, hashlib, json, os, pathlib, shlex, shutil, subprocess, sys, time, zipfile

COMMIT = "@COMMIT@"  # the locked code; change only on purpose
REPO = "https://github.com/Malnoad/Few-shot-rare-diseases"
ROOT, ENV = "/content/Few-shot", "/content/fm-env"
PY = f"{ENV}/bin/python"
PREPARED, ASSETS = "/content/prepared", "/content/assets"  # PREPARED becomes the input-lock folder at Step 12
WORK = "/content/fm_v1"  # specs, ledger, pilots, logs, reports, external query lock and features
LEDGER, REPORTS = f"{WORK}/download_ledger.csv", f"{WORK}/reports"
FEATURES, EXTERNAL_LOCK, META = f"{WORK}/features", f"{WORK}/external", f"{WORK}/external_meta"
DOWNLOADS, EXPORTS = "/content/external_downloads", "/content/fm_exports"  # big downloads; ZIPs made for Drive
LOCK_ZIP = f"{EXPORTS}/input-lock.zip"
RAW = "/content/drive/MyDrive/octfsl/raw"  # Lam's folder, through the runner's octfsl shortcut (guide, Part 1)
OUT = "/content/drive/MyDrive/octfsl/fm_v1"  # Lam's output folder, through the same shortcut
IMAGES = 84630
ARMS = ("RETFound", "DINOv2", "MIRAGE", "Inception")  # three FMs and the reference (A3 S-4: no fallback)

# Produced on the laptop from the same code and manifest (9 Nov).
EXPECTED_PREPARED = {
    "groups.csv": "6991d8229149aadf4af5ee53f1e2eb3991805d3bf56ebb5b367337a30224c5d0",
    "nine_way_roster.csv": "72f90c00b98810af46d257fe53d047d93f5312552ebaa478aef72d58da81b586",
    "episodes_development.csv": "cd060bddb434b5de476f4fd3550b8b21481ebaf3eb597298f727517968544695",
    "episodes_reporting.csv": "f9e1e3c6a4f2aacd405171837d472190271da2b03dbce79b619a03900b202b8f",
    "timing_inventory.csv": "a6be7a5ec3ba1bbe220674faaef8e51105bc7e4da59e9672d20e4e4562be97ed",
    "extraction_inventory.csv": "5ca665e4828b2d45c7d90cc6363708d9548793de7019d417e57222ee2b76b44d",
}


def sh(command):
    """Run a shell command, streaming its output; any failure stops the notebook."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    if process.wait():
        raise RuntimeError(f"FAILED: {command}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report(name, lines):
    """Print a report and keep a copy in REPORTS, so the save step can put it on Drive."""
    text = "\n".join(lines)
    print(text)
    path = pathlib.Path(REPORTS, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def save_to_drive(folder, paths, replace=False):
    """Copy files into octfsl/fm_v1/<folder>, verify every copy, then disconnect Drive.
    A file already there with the same bytes is kept; different bytes are never overwritten,
    except a receipt saved with replace=True."""
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    try:
        target = pathlib.Path(OUT, folder)
        target.mkdir(parents=True, exist_ok=True)
        for path in map(pathlib.Path, paths):
            digest, copy = sha256(path), target / path.name
            if copy.exists() and not replace:
                assert sha256(copy) == digest, f"{copy} already exists with different bytes; nothing was overwritten"
            else:  # copy under a temporary name first, so an interrupted copy never blocks a retry
                pending = target / (path.name + ".part")
                shutil.copyfile(path, pending)
                assert sha256(pending) == digest, (f"The Drive copy of {path.name} differs: run this step again; if it fails "
                                                   f"again, download {path} from the Files panel and send it to Lam.")
                os.replace(pending, copy)
            print(f"saved {path.name}  {digest}")
    finally:
        drive.flush_and_unmount()
'''

STEP1 = r'''# Step 1. Check the first cell and the GPU
first_cell = {**DECISIONS, "RUN_BY": RUN_BY, "RETFOUND_ACCESS": RETFOUND_ACCESS}
blank = [name for name, value in first_cell.items() if not value.strip()]
assert not blank, f"Stopped on purpose: fill in {blank} in the first cell, run that cell, then Run all again."
template = [name for name, value in first_cell.items() if "[" in value or "<" in value]
assert not template, f"Stopped on purpose: {template} still contain a [ or < placeholder; replace it with the real text."
assert EXTERNAL_DECISION.strip(), "Stopped on purpose: Lam's external-data decision is missing from the first cell."
wrong = [arm for arm, reason in UNAVAILABLE.items() if arm not in ARMS[:3] or not str(reason).strip()]
assert not wrong, f"Stopped on purpose: UNAVAILABLE may only name RETFound, DINOv2 or MIRAGE, each with Lam's reason: {wrong}"
RUN_ARMS = [arm for arm in ARMS if arm not in UNAVAILABLE]
query = "nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader"
gpu = subprocess.run(query, shell=True, capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU: Runtime > Change runtime type > GPU (A100, L4 or T4)"
print(gpu.stdout.strip())
assert float(gpu.stdout.split(",")[1]) < 12, "Blackwell GPU: switch to A100, L4 or T4"
'''

STEP2 = r'''# Step 2. Get the exact code and build the study environment
if not os.path.isdir(ROOT):
    sh(f"git clone -q {REPO} {ROOT}")
sh(f"git -C {ROOT} checkout -q {COMMIT}")
head = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
assert head == COMMIT, head
print("code:", head)

# Separate from Colab's own Python, so its packages cannot conflict. --without-pip needs no ensurepip.
if not os.path.exists(PY):
    sh(f"python3 -m venv --without-pip {ENV}")
sh(f"python3 -m pip --python {PY} install -q -r {ROOT}/requirements-fm.txt")
sh(f"python3 -m pip --python {PY} check")
sh(f"""{PY} -c 'import sys, torch, numpy, scipy, PIL, transformers; assert torch.cuda.is_available(), "PyTorch cannot see the GPU"; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "numpy", numpy.__version__)'""")
'''

STEP3 = r'''# Step 3. Check Drive, unpack the images, then disconnect Drive
from google.colab import drive


def drive_ready():
    """Check both folders now, so a Drive problem cannot surface only at Step 11."""
    for path in (RAW, OUT):
        assert os.path.isdir(path), f"{path} not found: add the octfsl shortcut to your My Drive (guide, Part 1)"
    probe = pathlib.Path(OUT, f".write-test-{time.time_ns()}")
    probe.write_text("ok", encoding="utf-8")
    assert probe.read_text(encoding="utf-8") == "ok", "octfsl/fm_v1 is not writable: ask Lam to check Editor access"
    probe.unlink()
    print("Drive ok: octfsl/raw is readable and octfsl/fm_v1 is writable")


def extract(zip_path, anchor):
    """Copy every file below the folder named `anchor` into data/raw, skipping macOS junk."""
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = [p for p in info.filename.replace("\\", "/").split("/") if p]
            if not parts or info.filename.endswith(("/", "\\")) or "__MACOSX" in parts or parts[-1].startswith("._"):
                continue
            if anchor not in parts[:-1]:
                continue
            target = pathlib.Path(ROOT, "data/raw", *parts[parts.index(anchor):])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            count += 1
    print(f"{pathlib.Path(zip_path).name}: {count} files")


with open(f"{ROOT}/data/manifest.csv", newline="", encoding="utf-8") as stream:
    MANIFEST = [row["path"].replace("\\", "/") for row in csv.DictReader(stream)]


def absent():
    return [p for p in MANIFEST if not os.path.isfile(f"{ROOT}/{p}")]


assert len(MANIFEST) == IMAGES, len(MANIFEST)
if absent() or not globals().get("DRIVE_OK"):
    drive.mount("/content/drive", force_remount=True)
    drive_ready()
    if absent():
        extract(f"{RAW}/oct2017.zip", "OCT2017")
        extract(f"{RAW}/rare.zip", "rare")
    drive.flush_and_unmount()
    DRIVE_OK = True
missing_images = absent()
assert not missing_images, f"{len(missing_images)} manifest images missing; first: {missing_images[0]}"
print(len(MANIFEST), "manifest images present")
'''

STEP4 = r'''# Step 4. Rebuild the rosters and compare them with the laptop's
sh(f"{PY} {ROOT}/fm_prepare.py screen --root {ROOT} --out {PREPARED}")
sh(f"{PY} {ROOT}/fm_prepare.py freeze --root {ROOT} --out {PREPARED}")
different = [name for name, digest in EXPECTED_PREPARED.items() if sha256(f"{PREPARED}/{name}") != digest]
assert not different, f"Differs from the laptop, stop and report: {different}"
print("All", len(EXPECTED_PREPARED), "files match the laptop byte for byte, so the 5,146 images and rosters agree.")
'''

STEP5A = r'''# Step 5a. Define the checkpoint downloads (revisions and digests come from fm_extract.py)
import requests

pins = json.loads(subprocess.check_output([PY, "-c", (
    "import json, fm_extract as x; print(json.dumps(dict(repositories=x.REPOSITORIES, revisions=x.REVISIONS, "
    "sources=x.SOURCE_REVISIONS, digests=x.EXPECTED_ASSET_SHA256)))")], cwd=ROOT, text=True))


def hf(arm, name):
    return f"https://huggingface.co/{pins['repositories'][arm]}/resolve/{pins['revisions'][arm]}/{name}"


def github(repo, arm, path):
    return f"https://raw.githubusercontent.com/{repo}/{pins['sources'][arm]}/{path}"


KERAS = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
FILES = [  # arm, file name, source, needs the Hugging Face token
    ("RETFound", "RETFound_mae_natureOCT.pth", hf("RETFound", "RETFound_mae_natureOCT.pth"), True),
    ("RETFound", "models_vit.py", github("rmaphoh/RETFound_MAE", "RETFound", "models_vit.py"), False),
    ("DINOv2", "model.safetensors", hf("DINOv2", "model.safetensors"), False),
    ("DINOv2", "config.json", hf("DINOv2", "config.json"), False),
    ("DINOv2", "preprocessor_config.json", hf("DINOv2", "preprocessor_config.json"), False),
    ("MIRAGE", "model.safetensors", hf("MIRAGE", "model.safetensors"), False),
    ("MIRAGE", "config.json", hf("MIRAGE", "config.json"), False),
    ("MIRAGE", "mirage_hf.py", github("j-morano/MIRAGE", "MIRAGE", "hf/mirage_hf.py"), False),
    ("Inception", KERAS, f"https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/{KERAS}", False),
]
assert {(arm, name) for arm, name, _, _ in FILES} == {(arm, name) for arm in ARMS for name in pins["digests"][arm]}


def record(row):
    """Append to the download-cost ledger the protocol asks for (runbook, R4)."""
    path = pathlib.Path(LEDGER)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(["arm", "file", "url", "bytes", "seconds", "utc"])
        writer.writerow(row)


def fetch(arm, name, url, token=None):
    """Download one pinned file. Bytes that fail the digest are deleted, never kept."""
    target = pathlib.Path(ASSETS, arm, name)
    expected = pins["digests"][arm][name]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == expected:
        print(f"already verified  {arm}/{name}")
        return
    partial = target.with_name(name + ".part")
    digest, size, start = hashlib.sha256(), 0, time.time()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1 << 20):
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    seconds = round(time.time() - start, 1)
    if digest.hexdigest() != expected:
        partial.unlink()
        raise ValueError(f"{arm}/{name} does not match its pinned SHA256; nothing was kept")
    partial.replace(target)
    record([arm, name, url, size, seconds, datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")])
    print(f"downloaded {size / 1e6:9.1f} MB in {seconds:7.1f} s  {arm}/{name}")
'''

STEP5B = r'''# Step 5b. Download. The token is read only for the RETFound weights and is never stored or printed.
from google.colab import userdata

for arm, name, url, gated in FILES:
    fetch(arm, name, url, userdata.get("HF_TOKEN") if gated else None)
'''

STEP6 = r'''# Step 6. Model inventory (Step 16 saves it to Drive as step6_inventory.txt)
lines = []
for arm in ARMS:
    for name, expected in pins["digests"][arm].items():
        path = pathlib.Path(ASSETS, arm, name)
        assert sha256(path) == expected, f"{arm}/{name}"
        lines.append(f"OK {path.stat().st_size / 1e6:9.1f} MB  {arm}/{name}")
lines += ["", open(LEDGER, encoding="utf-8").read() if os.path.exists(LEDGER) else "No new downloads this session."]
lines.append("code: " + subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip())
lines.append("Setup complete. No encoder has been run and no score exists.")
report("step6_inventory.txt", lines)
'''

STEP7 = r'''# Step 7. Record each model's identity and terms (hashes the staged files; downloads nothing)
TERMS = {
    "RETFound": f"Weights: huggingface.co/YukunZhou/RETFound_mae_natureOCT at the pinned revision; gated, downloaded with: {RETFOUND_ACCESS.strip()}. Code: github.com/rmaphoh/RETFound_MAE at the pinned commit, CC BY-NC 4.0.",
    "DINOv2": "huggingface.co/facebook/dinov2-base at the pinned revision, Apache-2.0.",
    "MIRAGE": "huggingface.co/j-morano/MIRAGE-Base at the pinned revision and github.com/j-morano/MIRAGE at the pinned commit, CC BY-NC 4.0 (non-commercial academic research with attribution).",
    "Inception": "Keras ImageNet no-top weights from storage.googleapis.com/tensorflow/keras-applications (no-top MD5 bcbd6486424b2319ff4ef7d526e38f63, as in keras 2.0.6 keras/applications/inception_v3.py). Keras code is Apache-2.0; keras 2.2.4 docs/templates/applications.md (InceptionV3, License) states the weights are released under the Apache License.",
}
CHECKED = "Public pages checked by the agent on 2026-11-10."

undecided = [arm for arm in ARMS if not DECISIONS[arm].strip()]
assert not undecided, f"Stopped on purpose: write your terms decision for {undecided} in the cell at the top, run that cell, then run from here."
SPECS = "/content/fm_v1/specs"
os.makedirs(SPECS, exist_ok=True)
for arm in ARMS:
    evidence = f"{TERMS[arm]} {CHECKED} Lam's decision: {DECISIONS[arm].strip()}"
    source = f"--source-revision {pins['sources'][arm]} " if arm in pins["sources"] else ""
    sh(f"{PY} {ROOT}/fm_extract.py spec --arm {arm} --assets {ASSETS}/{arm} --revision {pins['revisions'][arm]} "
       f"{source}--terms-evidence {shlex.quote(evidence)} --out {SPECS}/{arm}.json")
    print("spec written:", arm)
'''

STEP8 = r'''# Step 8. Download the external datasets (images only: no encoder, no label used, no score)
BOREALIS = "https://borealisdata.ca/api"
OCTID = {"doi:10.5683/SP/WLW4ZT": "NORMAL", "doi:10.5683/SP/MBMQGD": "MH", "doi:10.5683/SP/FVLZIL": "CSR"}
OCTID_LISTING_SHA256 = "984fde2b5222c44a3c2ed7bf34b5d513a921b7d88d3a12e4518cbb28f52b6f21"  # checked 10 Nov
OCTID_IMAGES = {"NORMAL": 206, "MH": 102, "CSR": 102}
ZENODO = "https://zenodo.org/records/10839556/files"
OCTDL_MD5 = {"OCTDL.zip": "dec9766fc916202c45a333c4f5a09576", "OCTDL_labels.csv": "2fced789038c6ca83283066c2f8ad289"}
OCTDL_IMAGES = {"NO": 332, "VID": 76}
A1_COUNTS = {"OCTDL/MH": 39, "OCTDL/NORMAL": 332, "OCTID/CSC": 102, "OCTID/MH": 102, "OCTID/NORMAL": 206}  # amendment A1
EXTERNAL = pathlib.Path(ROOT, "data/external")
for folder in (META, DOWNLOADS):
    os.makedirs(folder, exist_ok=True)


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_md5(source, url, target, expected):
    """Download one public file. Bytes that fail the published MD5 are deleted, never kept."""
    target = pathlib.Path(target)
    if target.exists() and md5(target) == expected:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial, size, start = target.with_name(target.name + ".part"), 0, time.time()
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1 << 20):
                stream.write(chunk)
                size += len(chunk)
    if md5(partial) != expected:
        partial.unlink()
        raise ValueError(f"{target.name} does not match its published MD5; nothing was kept")
    partial.replace(target)
    record([source, target.name, url, size, round(time.time() - start, 1), datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")])


listing = []
for doi in OCTID:
    response = requests.get(f"{BOREALIS}/datasets/:persistentId/", params={"persistentId": doi}, timeout=120)
    response.raise_for_status()
    version = response.json()["data"]["latestVersion"]
    title = next(field["value"] for field in version["metadataBlocks"]["citation"]["fields"] if field["typeName"] == "title")
    listing += [[doi, title, f["dataFile"]["filename"], f["dataFile"]["filesize"], f["dataFile"]["md5"], f["dataFile"]["id"]] for f in version["files"]]
listing.sort(key=lambda row: (row[0], row[2]))
with open(f"{META}/octid_files.csv", "w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["doi", "title", "filename", "bytes", "md5", "file_id"])
    writer.writerows(listing)
assert sha256(f"{META}/octid_files.csv") == OCTID_LISTING_SHA256, "The OCTID release changed since 10 Nov: stop and report."
for doi, _, name, _, digest, file_id in listing:
    if name.lower().endswith(".jpeg"):
        fetch_md5("OCTID", f"{BOREALIS}/access/datafile/{file_id}", EXTERNAL / "OCTID" / OCTID[doi] / name, digest)
found = {folder: len(list((EXTERNAL / "OCTID" / folder).glob("*.jpeg"))) for folder in OCTID_IMAGES}
assert found == OCTID_IMAGES, found

fetch_md5("OCTDL", f"{ZENODO}/OCTDL.zip?download=1", f"{DOWNLOADS}/OCTDL.zip", OCTDL_MD5["OCTDL.zip"])
fetch_md5("OCTDL", f"{ZENODO}/OCTDL_labels.csv?download=1", f"{META}/OCTDL_labels.csv", OCTDL_MD5["OCTDL_labels.csv"])
with zipfile.ZipFile(f"{DOWNLOADS}/OCTDL.zip") as archive:
    for info in archive.infolist():
        parts = info.filename.split("/")
        if not info.is_dir() and len(parts) == 3 and parts[0] == "OCTDL" and parts[1] in OCTDL_IMAGES:
            target = EXTERNAL.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
found = {folder: len(list((EXTERNAL / "OCTDL" / folder).glob("*.jpg"))) for folder in OCTDL_IMAGES}
assert found == OCTDL_IMAGES, found

sh(f"{PY} {ROOT}/fm_external.py candidates --octid-files {META}/octid_files.csv --octdl-labels {META}/OCTDL_labels.csv "
   f"--out {META}/external_candidates.csv")
with open(f"{META}/external_candidates.csv", newline="", encoding="utf-8") as stream:
    counts = {}
    for row in csv.DictReader(stream):
        counts[f"{row['source']}/{row['label']}"] = counts.get(f"{row['source']}/{row['label']}", 0) + 1
assert counts == A1_COUNTS, f"External cohorts differ from amendment A1, stop and report: {counts}"
ELIGIBILITY = ("Amendment A1 (12 November 2026), fixed in fm_external.py: NORMAL queries for the false-rare rate; "
               "OCTID CSR as CSC and OCTID MH; OCTDL disease=VID with subcategory=MH. Queries only.")
EXTERNAL_GATE = {
    "OCTID": dict(status="planned",
                  release="borealisdata.ca/dataverse/OCTID: doi:10.5683/SP/WLW4ZT, SP/MBMQGD, SP/FVLZIL, each v1.0. "
                          "release_sha256 is the SHA256 of the file listing (name, size and MD5 of every file); "
                          "every image was checked against its MD5.",
                  release_sha256=sha256(f"{META}/octid_files.csv"),
                  terms_evidence=f"CC0 1.0 (Borealis dataset pages). {EXTERNAL_DECISION}",
                  eligibility=ELIGIBILITY,
                  label_inventory=f"external_candidates.csv SHA256 {sha256(META + '/external_candidates.csv')}: "
                                  "NORMAL 206, CSC 102, MH 102"),
    "OCTDL": dict(status="planned",
                  release="zenodo.org/records/10839556 (OCTDL.zip and OCTDL_labels.csv, published MD5 verified). "
                          "release_sha256 is the SHA256 of OCTDL.zip.",
                  release_sha256=sha256(f"{DOWNLOADS}/OCTDL.zip"),
                  terms_evidence=f"CC BY 4.0 (Zenodo record). {EXTERNAL_DECISION}",
                  eligibility=ELIGIBILITY,
                  label_inventory=f"OCTDL_labels.csv SHA256 {sha256(META + '/OCTDL_labels.csv')}: NORMAL 332, MH 39"),
}
print(json.dumps(counts, indent=2, sort_keys=True))
print("External images are staged. No encoder has run on them and no external label has been used.")
'''

STEP9 = r'''# Step 9. Keep Drive and credentials away from the model code (risk record R1)
TOKEN_CACHES = ("/root/.cache/huggingface/token", "/root/.cache/huggingface/stored_tokens", "/root/.huggingface/token")
assert not os.path.ismount("/content/drive"), "Drive is mounted: run drive.flush_and_unmount() first."
cached = [path for path in TOKEN_CACHES if os.path.exists(path)]
assert not cached, f"A Hugging Face login is cached at {cached}. Use a fresh runtime, so the model code cannot read it."
SAFE_ENV = {name: value for name, value in os.environ.items()
            if not any(word in name.upper() for word in ("TOKEN", "SECRET", "KEY", "PASSW", "CREDENTIAL", "HF_", "HUGGING"))}
print("Removed from the worker environment (names only):", sorted(set(os.environ) - set(SAFE_ENV)) or "nothing")
INTERPRETER = {**{arm: PY for arm in ARMS}, "Inception": sys.executable}  # the reference uses Colab's own TensorFlow


def worker(command, log):
    """Run one process with no credentials in its environment, keep its output in `log`, wait until it exits."""
    process = subprocess.Popen(command, env=SAFE_ENV, cwd="/content", stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as stream:
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
    return process.pid, process.wait()


def gpu_users():
    return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
'''

STEP10 = r'''# Step 10. Final pilots: load each model and time it on 100 images, one process at a time (no labels, no scores)
def pilots_done():
    return bool(globals().get("PILOTS")) and all(os.path.exists(f"{PILOTS}/{arm}/complete.json") for arm in RUN_ARMS)


if pilots_done():
    print("Pilots already complete for", RUN_ARMS, "in", ATTEMPT)
else:
    assert not os.path.exists(f"{PREPARED}/input_lock.json"), "The input lock exists: pilots cannot be rerun under it."
    ATTEMPTS = globals().get("ATTEMPTS", [])
    assert len(ATTEMPTS) < 2, ("Two pilot sequences have failed (amendment A3, S-3). Stop and tell Lam: "
                               "only Lam can list a model in UNAVAILABLE in the first cell.")
    ATTEMPT = datetime.datetime.now(datetime.timezone.utc).strftime("final-%Y%m%dT%H%M%SZ")
    ATTEMPTS.append(ATTEMPT)
    PILOTS = f"{WORK}/pilots/{ATTEMPT}"
    os.makedirs(PILOTS, exist_ok=True)
    RESULTS = {}
    for arm in RUN_ARMS:  # fixed protocol order; one failed model does not hide the others
        assert not gpu_users(), f"Another process is using the GPU ({gpu_users()}): restart the runtime so timings stay comparable."
        print(f"\n=== {arm} ===", flush=True)
        started = time.time()
        pid, code = worker([INTERPRETER[arm], f"{ROOT}/fm_extract.py", "pilot", "--root", ROOT, "--prepared", PREPARED,
                            "--spec", f"{SPECS}/{arm}.json", "--assets", f"{ASSETS}/{arm}", "--out", f"{PILOTS}/{arm}",
                            "--timing-session", ATTEMPT], f"{PILOTS}/{arm}.log")
        RESULTS[arm] = dict(pid=pid, exit_code=code, seconds=round(time.time() - started, 1))
        print(f"{arm}: exit code {code} after {RESULTS[arm]['seconds']} s", flush=True)
    pathlib.Path(PILOTS, "processes.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
'''

STEP11 = r'''# Step 11. Pilot report (Step 16 saves it to Drive as step11_pilots.txt)
failed, lines = [], []
for arm in ARMS:
    folder = pathlib.Path(PILOTS, arm)
    if arm in UNAVAILABLE:
        lines.append(f"{arm:10s} UNAVAILABLE (Lam): {UNAVAILABLE[arm]}")
        continue
    if not (folder / "complete.json").exists():
        failed.append(arm)
        lines.append(f"{arm:10s} FAILED: the error is in the Step 10 output and {arm}.log")
        continue
    timing = json.loads((folder / "timing.json").read_text(encoding="utf-8"))
    load = json.loads((folder / "load_report.json").read_text(encoding="utf-8"))
    costs, numerical = timing["costs"], timing["numerical"]
    unexpected = load.get("unexpected_keys", [])
    notes = {key: load[key] for key in ("checkpoint_layout", "default_norm", "text_tower") if key in load}
    lines.append(f"{arm:10s} {timing['median_seconds_per_image'] * 1000:8.2f} ms/image   load {costs['model_loading_seconds']:6.1f} s   "
                 f"peak GPU memory {costs['peak_vram_bytes'] / 1e9:5.2f} GB")
    lines.append(f"           repeat check passed: real images worst difference {numerical['real']['worst_image_difference_ratio']:.2e}, "
                 f"lowest cosine {numerical['real']['min_cosine']:.6f}; synthetic {numerical['synthetic']['max_difference_ratio']:.2e}")
    lines.append(f"           missing keys: {load.get('missing_keys', [])}   unexpected keys: {len(unexpected)} {unexpected[:3]}   {notes}")
lines += ["", f"attempt: {ATTEMPT}   pilot sequences this session: {len(ATTEMPTS)}", "gpu: " + gpu.stdout.strip(),
          "code: " + subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip(),
          "No labels were used and no score exists."]
report("step11_pilots.txt", lines)
if failed and len(ATTEMPTS) < 2:
    raise RuntimeError(f"Pilot failed for {failed}. Amendment A3 allows one retry: run Step 10 and Step 11 once more, "
                       "changing nothing. Tell Lam it happened.")
if failed:
    raise RuntimeError(f"Pilot failed twice for {failed}. Stop and tell Lam. Do not change anything in the notebook.")
'''

STEP12 = r'''# Step 12. Make the input lock (fixes rosters, episodes, models, code and endpoints before any score)
GATE = f"{WORK}/model-gate.json"
gate = dict(arms=[dict(name=arm, status="unavailable", reason=str(UNAVAILABLE[arm]).strip()) if arm in UNAVAILABLE
                  else dict(name=arm, status="available", pilot=f"{PILOTS}/{arm}") for arm in ARMS],
            external_sources=EXTERNAL_GATE)
if not os.path.exists(GATE):
    pathlib.Path(GATE).write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
assert json.loads(pathlib.Path(GATE).read_text(encoding="utf-8")) == gate, "The gate changed after it was written: stop and tell Lam."
os.makedirs(EXPORTS, exist_ok=True)
if not (os.path.exists(LOCK_ZIP) and os.path.exists(LOCK_ZIP + ".json") and os.path.exists(f"{PREPARED}/input_lock.json")):
    sh(f"{PY} {ROOT}/fm_prepare.py lock --root {ROOT} --out {PREPARED} --gate {GATE} --archive {LOCK_ZIP}")
LOCK_SHA256 = json.loads(pathlib.Path(LOCK_ZIP + ".json").read_text(encoding="utf-8"))["sha256"]
assert sha256(LOCK_ZIP) == LOCK_SHA256, "The lock archive differs from its receipt: stop and tell Lam."
LOCK = json.loads(pathlib.Path(PREPARED, "input_lock.json").read_text(encoding="utf-8"))
print("+" + "-" * 78 + "+")
print(f"| INPUT LOCK SHA256  {LOCK_SHA256} |")
print("+" + "-" * 78 + "+")
print("created:", LOCK["created_utc"], "  arms:", {arm: value["status"] for arm, value in LOCK["arms"].items()})
print("evaluation environment:", LOCK["evaluation_environment"], "  reporting K:", LOCK["endpoints"]["reporting_k"])
# The text Lam deposits or emails as the public timestamp (amendments A2 A-8 and A3 S-5).
HASHES = f"{EXPORTS}/lock-hashes-{ATTEMPT}.txt"
pathlib.Path(HASHES).write_text("\n".join([
    "Frozen-encoder rare-OCT study, protocol v1.0: input lock, made before any labelled score.",
    f"attempt: {ATTEMPT}",
    f"input-lock ZIP SHA256: {LOCK_SHA256}",
    f"protocol FM_PROTOCOL.md SHA256: {LOCK['protocol_sha256']}",
    f"amendments FM_AMENDMENTS.md SHA256 (LF line endings): {LOCK['code']['docs/research/FM_AMENDMENTS.md']}",
    f"code commit: {COMMIT}",
    f"lock created (UTC): {LOCK['created_utc']}"]) + "\n", encoding="utf-8")
print(pathlib.Path(HASHES).read_text(encoding="utf-8"))
'''

STEP13 = r'''# Step 13. Save the lock to octfsl/fm_v1/final/<attempt>/ now, before the long extraction
save_to_drive(f"final/{ATTEMPT}", [LOCK_ZIP, LOCK_ZIP + ".json", HASHES])
print(f"Send Lam now: attempt {ATTEMPT}, input-lock SHA256 {LOCK_SHA256}")
'''

STEP14 = r'''# Step 14. Screen the external images against the locked ones and fix the external query list
# Amendment A3, S-1: every flagged near-duplicate counts as a link; nobody reviews pairs.
if not os.path.exists(f"{EXTERNAL_LOCK}/complete.json"):  # written last, so an interrupted freeze is finished on a rerun
    if not os.path.exists(f"{EXTERNAL_LOCK}/external_screen_summary.json"):
        sh(f"{PY} {ROOT}/fm_external.py screen --root {ROOT} --lock {PREPARED} --candidates {META}/external_candidates.csv --out {EXTERNAL_LOCK}")
    sh(f"{PY} {ROOT}/fm_external.py freeze --root {ROOT} --lock {PREPARED} --out {EXTERNAL_LOCK}")
assert os.path.exists(f"{EXTERNAL_LOCK}/complete.json"), "The external query lock did not finish: run this step again."
external_lock = json.loads(pathlib.Path(EXTERNAL_LOCK, "external_lock.json").read_text(encoding="utf-8"))
screened = json.loads(pathlib.Path(EXTERNAL_LOCK, "external_screen_summary.json").read_text(encoding="utf-8"))
lines = [f"{source}: {value['queries']} queries from {value['candidates']} candidates, {value['excluded']} excluded, "
         f"{value['dependence_groups']} dependence groups, labels {value['labels']}" for source, value in external_lock["sources"].items()]
lines += ["flagged pairs by block: " + json.dumps(screened["pairs_by_block"]),
          "Excluded images are listed in external_exclusions.csv. No label was scored."]
report("step14_external.txt", lines)
sh(f"{PY} {ROOT}/fm_prepare.py export --root {ROOT} --out {EXTERNAL_LOCK} --archive {EXPORTS}/external-lock.zip")
'''

STEP15 = r'''# Step 15. Extract features: every available model, study images then external images (no labels, no scores)
assert not os.path.ismount("/content/drive"), "Drive is mounted: run drive.flush_and_unmount() first."
EXTRACTED = []
for arm in RUN_ARMS:
    for kind, out, extra in (("main", f"{FEATURES}/{arm}", []),
                             ("external", f"{FEATURES}/{arm}-external", ["--external", EXTERNAL_LOCK])):
        if not os.path.exists(f"{out}/complete.json"):
            print(f"\n=== {arm} {kind} ===", flush=True)
            started = time.time()
            pid, code = worker([INTERPRETER[arm], f"{ROOT}/fm_extract.py", "extract", "--root", ROOT, "--prepared", PREPARED,
                                "--spec", f"{SPECS}/{arm}.json", "--assets", f"{ASSETS}/{arm}", "--out", out, *extra],
                               f"{WORK}/logs/extract-{arm}-{kind}.log")
            print(f"{arm} {kind}: exit code {code} after {round(time.time() - started, 1)} s", flush=True)
            assert code == 0 and os.path.exists(f"{out}/complete.json"), (
                f"Extraction failed for {arm} ({kind}). The log is {WORK}/logs/extract-{arm}-{kind}.log. "
                "Run this step again once, changing nothing; if it fails again, run Step 16 to save the evidence and tell Lam.")
        EXTRACTED.append(out)
print("\nExtracted:", [os.path.basename(path) for path in EXTRACTED])
'''

STEP16 = r'''# Step 16. Save features and evidence to octfsl/fm_v1/final/<attempt>/, then disconnect Drive
# Saved: input lock, external query lock, one feature archive per model and cohort, reports, specs, ledger,
# pilot receipts and logs. No weights and no images.
archives = [LOCK_ZIP, f"{EXPORTS}/external-lock.zip"]
for folder in globals().get("EXTRACTED", []):
    archive = f"{EXPORTS}/features-{os.path.basename(folder)}.zip"
    sh(f"{PY} {ROOT}/fm_prepare.py export --root {ROOT} --out {folder} --archive {archive}")
    archives.append(archive)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
evidence = f"{EXPORTS}/evidence-{stamp}.zip"  # a new name each time, so a second save never collides with the first
with zipfile.ZipFile(evidence, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(pathlib.Path(WORK).rglob("*")):
        relative = path.relative_to(WORK)
        if path.is_file() and relative.parts[0] not in ("features", "external"):
            archive.write(path, relative.as_posix())
archives.append(evidence)
expected = 2 + 2 * len(RUN_ARMS)
receipt = dict(attempt=ATTEMPT, run_by=RUN_BY.strip(), code=COMMIT, gpu=gpu.stdout.strip(), lock_sha256=LOCK_SHA256,
               arms={arm: LOCK["arms"][arm]["status"] for arm in ARMS}, complete=len(archives) == expected + 1,
               files={os.path.basename(path): dict(sha256=sha256(path), bytes=os.path.getsize(path)) for path in archives},
               saved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
pathlib.Path(EXPORTS, "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
save_to_drive(f"final/{ATTEMPT}", [*archives, *[path + ".json" for path in archives if os.path.exists(path + ".json")],
                                    f"{REPORTS}/step6_inventory.txt", f"{REPORTS}/step11_pilots.txt",
                                    f"{REPORTS}/step14_external.txt"])
save_to_drive(f"final/{ATTEMPT}", [f"{EXPORTS}/receipt.json"], replace=True)  # last, so a receipt on Drive means the rest is there
assert receipt["complete"], "Some features are missing: the evidence is saved; tell Lam and keep the runtime open."
print(f"\nAll done. Send Lam:\n  attempt          {ATTEMPT}\n  input-lock SHA256 {LOCK_SHA256}")
print("No score exists yet. Lam approves this SHA256 and records the public timestamp before colab_fm_evaluate.ipynb is opened.")
'''


# ---------------------------------------------------------------------------------------------
# colab_fm_evaluate.ipynb: CPU only. The first labelled scores of the study are computed here.
# ---------------------------------------------------------------------------------------------
E_INTRO = r'''# FM study: development, nomination and reports

This notebook computes the study's **first labelled scores**. Run it only after Lam has approved the
input-lock SHA256 and recorded the public timestamp (amendments A2 A-8 and A3 S-5). It needs no GPU and
no Hugging Face token: it reads the lock and the features that `colab_fm.ipynb` saved to
`octfsl/fm_v1/final/<attempt>/`.

**Step-by-step guide:** `docs/research/FM_FIRST_RUN_GUIDE.md`, Part 4.

| Step | What it does | What proves it worked |
|---|---|---|
| 1 | Checks the first cell | Every field filled; the SHA256 has 64 characters |
| 2 | Clones the exact code and builds the same Python environment | Commit matches; `pip check` passes |
| 3 | Copies the lock and features from Drive, then disconnects Drive | Every file matches the saved receipt and Lam's SHA256 |
| 4 | Restores the lock and unpacks the features | Lock restored; same Python, NumPy and SciPy as at the lock |
| 5 | Development: K-shot episodes on the five folds, every model | One `development` folder per model |
| 6 | Nominates one model, saves the decision to Drive, **then stops** | The nomination is printed |
| 7 | Reports: legacy 22 images, nine-way audit, external images | One `reporting` and one `external` folder per model |
| 8 | Summary tables and the geometry table | Files listed |
| 9 | Saves everything to `octfsl/fm_v1/final/<attempt>/evaluation/` | The Drive copy's SHA256 matches |

Step 6 stops on purpose. Lam reads the nomination, sets `REPORT_AFTER_NOMINATION = True` in the first
cell, and the remaining steps are then run. The nominee can never change after that: the code seals it.
Nothing here may be edited to change a result.
'''

E_FIRST = r'''# Fill in this cell before Run all (Step 1 stops until it is complete)
ATTEMPT = ""            # printed at the end of colab_fm.ipynb, for example final-20261114T030000Z
LOCK_SHA256 = ""        # the input-lock SHA256 Lam approved (64 characters)
LOCK_APPROVAL = ""      # Lam's approval: name, date and words
TIMESTAMP_RECORD = ""   # where and when the hashes were deposited, or emailed to the mentor and team
RUN_BY = ""             # your name
# Lam sets this to True after reading the nomination that Step 6 prints.
REPORT_AFTER_NOMINATION = False
# Leave empty. Only Lam fills it, for a dated and disclosed reduction (protocol section 9).
OMIT_LOGISTIC_REASON = ""
'''

E_SETTINGS = r'''# Settings and helpers
import datetime, hashlib, json, os, pathlib, shlex, shutil, subprocess, sys, zipfile

COMMIT = "@COMMIT@"  # the locked code; change only on purpose
REPO = "https://github.com/Malnoad/Few-shot-rare-diseases"
ROOT, ENV = "/content/Few-shot", "/content/fm-env"
PY = f"{ENV}/bin/python"
EVAL = "/content/fm_eval"  # outside the code checkout, as the scripts require
ARCHIVES, LOCK, FEATURES, EXTERNAL = f"{EVAL}/archives", f"{EVAL}/lock", f"{EVAL}/features", f"{EVAL}/external"
RUNS, TABLES, NOMINATION = f"{EVAL}/runs", f"{EVAL}/reports", f"{EVAL}/reports/nomination"
OUT = "/content/drive/MyDrive/octfsl/fm_v1"  # Lam's output folder, through the runner's octfsl shortcut
ARMS = ("RETFound", "DINOv2", "MIRAGE", "Inception")


def sh(command):
    """Run a shell command, streaming its output; any failure stops the notebook."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    if process.wait():
        raise RuntimeError(f"FAILED: {command}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_to_drive(folder, paths, replace=False):
    """Copy files into octfsl/fm_v1/<folder>, verify every copy, then disconnect Drive.
    A file already there with the same bytes is kept; different bytes are never overwritten,
    except a receipt saved with replace=True."""
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    try:
        target = pathlib.Path(OUT, folder)
        target.mkdir(parents=True, exist_ok=True)
        for path in map(pathlib.Path, paths):
            digest, copy = sha256(path), target / path.name
            if copy.exists() and not replace:
                assert sha256(copy) == digest, f"{copy} already exists with different bytes; nothing was overwritten"
            else:  # copy under a temporary name first, so an interrupted copy never blocks a retry
                pending = target / (path.name + ".part")
                shutil.copyfile(path, pending)
                assert sha256(pending) == digest, (f"The Drive copy of {path.name} differs: run this step again; if it fails "
                                                   f"again, download {path} from the Files panel and send it to Lam.")
                os.replace(pending, copy)
            print(f"saved {path.name}  {digest}")
    finally:
        drive.flush_and_unmount()
'''

E_STEP1 = r'''# Step 1. Check the first cell
first_cell = dict(ATTEMPT=ATTEMPT, LOCK_SHA256=LOCK_SHA256, LOCK_APPROVAL=LOCK_APPROVAL,
                  TIMESTAMP_RECORD=TIMESTAMP_RECORD, RUN_BY=RUN_BY)
blank = [name for name, value in first_cell.items() if not value.strip()]
assert not blank, f"Stopped on purpose: fill in {blank} in the first cell, run that cell, then Run all again."
template = [name for name, value in first_cell.items() if "[" in value or "<" in value]
assert not template, f"Stopped on purpose: {template} still contain a [ or < placeholder; replace it with the real text."
LOCK_SHA256 = LOCK_SHA256.strip().lower()
assert len(LOCK_SHA256) == 64 and set(LOCK_SHA256) <= set("0123456789abcdef"), "LOCK_SHA256 must be the 64-character SHA256 Lam approved."
assert ATTEMPT.strip().startswith("final-"), "ATTEMPT must be the name printed by colab_fm.ipynb, starting with final-"
ATTEMPT = ATTEMPT.strip()
assert isinstance(REPORT_AFTER_NOMINATION, bool), "REPORT_AFTER_NOMINATION must be True or False."
print("First cell complete. Attempt:", ATTEMPT)
'''

E_STEP2 = r'''# Step 2. Get the exact code and build the same Python environment as at the lock
if not os.path.isdir(ROOT):
    sh(f"git clone -q {REPO} {ROOT}")
sh(f"git -C {ROOT} checkout -q {COMMIT}")
head = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
assert head == COMMIT, head
print("code:", head)
if not os.path.exists(PY):
    sh(f"python3 -m venv --without-pip {ENV}")
sh(f"python3 -m pip --python {PY} install -q -r {ROOT}/requirements-fm.txt")
sh(f"python3 -m pip --python {PY} check")
sh(f"""{PY} -c 'import sys, numpy, scipy; print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__)'""")
'''

E_STEP3 = r'''# Step 3. Copy the lock and features from Drive, check them, then disconnect Drive
from google.colab import drive

SOURCE = f"{OUT}/final/{ATTEMPT}"
os.makedirs(ARCHIVES, exist_ok=True)
if not os.path.exists(f"{ARCHIVES}/receipt.json"):
    drive.mount("/content/drive", force_remount=True)
    try:
        assert os.path.isfile(f"{SOURCE}/receipt.json"), f"{SOURCE}/receipt.json not found: check ATTEMPT and the octfsl shortcut."
        for name in sorted(os.listdir(SOURCE)):
            if os.path.isfile(f"{SOURCE}/{name}") and name != "receipt.json":
                shutil.copyfile(f"{SOURCE}/{name}", f"{ARCHIVES}/{name}")
        saved = f"{SOURCE}/evaluation/nomination_binding.json"  # present only if a nomination was already made
        if os.path.isfile(saved):
            shutil.copyfile(saved, f"{ARCHIVES}/nomination_binding.json")
        shutil.copyfile(f"{SOURCE}/receipt.json", f"{ARCHIVES}/receipt.json")
    finally:
        drive.flush_and_unmount()
RECEIPT = json.loads(pathlib.Path(ARCHIVES, "receipt.json").read_text(encoding="utf-8"))
assert RECEIPT["attempt"] == ATTEMPT and RECEIPT["complete"], "The saved receipt is for another attempt or is incomplete: tell Lam."
assert RECEIPT["code"] == COMMIT, f"The features were made with code {RECEIPT['code']}, not this notebook's {COMMIT}: tell Lam."
assert RECEIPT["lock_sha256"] == LOCK_SHA256, "LOCK_SHA256 in the first cell is not the lock of this attempt: check with Lam."
for name, entry in RECEIPT["files"].items():
    assert sha256(f"{ARCHIVES}/{name}") == entry["sha256"], f"{name} differs from the saved receipt: tell Lam."
assert sha256(f"{ARCHIVES}/input-lock.zip") == LOCK_SHA256
print(len(RECEIPT["files"]), "archives match the receipt; the lock is the one Lam approved.")
'''

E_STEP4 = r'''# Step 4. Restore the lock and unpack the features
# restore is safe to repeat: it writes each file once and finishes with the receipt
sh(f"{PY} {ROOT}/fm_prepare.py restore --root {ROOT} --out {LOCK} --archive {ARCHIVES}/input-lock.zip --sha256 {LOCK_SHA256}")
if os.path.exists(f"{ARCHIVES}/nomination_binding.json") and not os.path.exists(f"{LOCK}/nomination_binding.json"):
    shutil.copyfile(f"{ARCHIVES}/nomination_binding.json", f"{LOCK}/nomination_binding.json")  # the original decision
    print("Restored the nomination already made for this lock; it will not be recomputed.")


def unpack(archive, folder):
    """Unpack beside the target and rename at the end, so a folder that exists is always complete."""
    if not os.path.isdir(folder):
        shutil.rmtree(folder + ".part", ignore_errors=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(folder + ".part")
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        os.replace(folder + ".part", folder)


LOCKED = json.loads(pathlib.Path(LOCK, "input_lock.json").read_text(encoding="utf-8"))
AVAILABLE = [arm for arm in ARMS if LOCKED["arms"][arm]["status"] == "available"]
unpack(f"{ARCHIVES}/external-lock.zip", EXTERNAL)
for arm in AVAILABLE:
    unpack(f"{ARCHIVES}/features-{arm}.zip", f"{FEATURES}/{arm}")
    unpack(f"{ARCHIVES}/features-{arm}-external.zip", f"{FEATURES}/{arm}-external")
current = json.loads(subprocess.check_output([PY, "-c", "import json, fm_prepare; print(json.dumps(fm_prepare.math_environment()))"], cwd=ROOT, text=True))
assert current == LOCKED["evaluation_environment"], (
    f"This runtime has {current}, the lock has {LOCKED['evaluation_environment']}. Scores must use the locked versions: stop and tell Lam.")
APPROVALS = f"{EVAL}/approvals.json"
if not os.path.exists(APPROVALS):
    pathlib.Path(APPROVALS).write_text(json.dumps(dict(
        attempt=ATTEMPT, lock_sha256=LOCK_SHA256, lock_approval=LOCK_APPROVAL.strip(), timestamp_record=TIMESTAMP_RECORD.strip(),
        run_by=RUN_BY.strip(), code=COMMIT, recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
        indent=2) + "\n", encoding="utf-8")
OMIT = f" --omit-logistic-reason {shlex.quote(OMIT_LOGISTIC_REASON.strip())}" if OMIT_LOGISTIC_REASON.strip() else ""
print("Lock restored. Available models:", AVAILABLE, " environment:", current)
'''

E_STEP5 = r'''# Step 5. Development: paired K-shot episodes on the five folds, every available model (first labelled scores)
for arm in AVAILABLE:
    out = f"{RUNS}/{arm}/development"
    if not os.path.exists(f"{out}/complete.json"):
        sh(f"{PY} {ROOT}/fm_evaluate.py develop --lock {LOCK} --cache {FEATURES}/{arm} --arm {arm} --out {out}{OMIT}")
    print("development complete:", arm)
'''

E_STEP6 = r'''# Step 6. Nominate one model from the development folds, save the decision to Drive, then stop for Lam
if not os.path.exists(f"{NOMINATION}/complete.json"):
    sh(f"{PY} {ROOT}/fm_evaluate.py nominate --lock {LOCK} --runs {RUNS} --out {NOMINATION}")
os.makedirs(f"{EVAL}/exports", exist_ok=True)
sh(f"{PY} {ROOT}/fm_prepare.py export --root {ROOT} --out {NOMINATION} --archive {EVAL}/exports/nomination.zip")
BINDING_SHA256 = sha256(f"{LOCK}/nomination_binding.json")
save_to_drive(f"final/{ATTEMPT}/evaluation", [f"{LOCK}/nomination_binding.json", f"{EVAL}/exports/nomination.zip",
                                               f"{EVAL}/exports/nomination.zip.json"])
print(pathlib.Path(NOMINATION, "nomination.json").read_text(encoding="utf-8"))
print("Nomination sealed. Binding SHA256:", BINDING_SHA256)
assert REPORT_AFTER_NOMINATION, ("Stopped on purpose. Send Lam the nomination printed above. When Lam has read it, set "
                                 "REPORT_AFTER_NOMINATION = True in the first cell, run that cell, then run Step 6 onwards.")
'''

E_STEP7 = r'''# Step 7. Reports: legacy 22 images and nine-way audit, then external images, every available model
for arm in AVAILABLE:
    out = f"{RUNS}/{arm}/reporting"
    if not os.path.exists(f"{out}/complete.json"):
        sh(f"{PY} {ROOT}/fm_evaluate.py report --lock {LOCK} --cache {FEATURES}/{arm} --arm {arm} --out {out} "
           f"--nomination {NOMINATION}{OMIT}")
    out = f"{RUNS}/{arm}/external"
    if not os.path.exists(f"{out}/complete.json"):
        sh(f"{PY} {ROOT}/fm_external.py report --lock {LOCK} --cache {FEATURES}/{arm} --external-cache {FEATURES}/{arm}-external "
           f"--external {EXTERNAL} --arm {arm} --nomination {NOMINATION} --out {out}{OMIT}")
    print("reports complete:", arm)
'''

E_STEP8 = r'''# Step 8. Summary tables and the geometry table
if not os.path.exists(f"{TABLES}/legacy/complete.json"):
    sh(f"{PY} {ROOT}/fm_evaluate.py summarize --lock {LOCK} --runs {RUNS} --nomination {NOMINATION} --out {TABLES}/legacy")
if not os.path.exists(f"{TABLES}/external/complete.json"):
    sh(f"{PY} {ROOT}/fm_external.py summarize --lock {LOCK} --external {EXTERNAL} --runs {RUNS} --nomination {NOMINATION} "
       f"--out {TABLES}/external")
for arm in AVAILABLE:
    out = f"{TABLES}/geometry/{arm}"
    if not os.path.exists(f"{out}/complete.json"):
        sh(f"{PY} {ROOT}/fm_evaluate.py geometry --lock {LOCK} --cache {FEATURES}/{arm} --arm {arm} --out {out} "
           f"--external {EXTERNAL} --external-cache {FEATURES}/{arm}-external")
for path in sorted(pathlib.Path(TABLES).rglob("*.csv")):
    print(path.relative_to(TABLES).as_posix())
'''

E_STEP9 = r'''# Step 9. Save runs and tables to octfsl/fm_v1/final/<attempt>/evaluation/, then disconnect Drive
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
results = f"{EVAL}/exports/results-{stamp}.zip"
with zipfile.ZipFile(results, "w", zipfile.ZIP_DEFLATED) as archive:
    for folder in (RUNS, TABLES):
        for path in sorted(pathlib.Path(folder).rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(EVAL).as_posix())
    archive.write(APPROVALS, "approvals.json")
receipt = dict(attempt=ATTEMPT, run_by=RUN_BY.strip(), code=COMMIT, lock_sha256=LOCK_SHA256, binding_sha256=BINDING_SHA256,
               zip=os.path.basename(results), zip_sha256=sha256(results), zip_bytes=os.path.getsize(results), saved_utc=stamp)
receipt_path = f"{EVAL}/exports/results-{stamp}.json"
pathlib.Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
save_to_drive(f"final/{ATTEMPT}/evaluation", [results, receipt_path])  # new names each run, so nothing is replaced
print(f"All done. Tell Lam: results saved as {os.path.basename(results)}, SHA256 {receipt['zip_sha256']}")
'''


def cell(kind, text):
    lines = text.strip("\n").splitlines(keepends=True)
    base = dict(cell_type=kind, metadata={}, source=lines)
    return {**base, "execution_count": None, "outputs": []} if kind == "code" else base


def notebook(name, intro, code, commit, gpu):
    metadata = dict(colab=dict(provenance=[], name=name), kernelspec=dict(name="python3", display_name="Python 3"),
                    language_info=dict(name="python"))
    if gpu:
        metadata = dict(accelerator="GPU", **metadata)
    cells = [cell("markdown", intro)] + [cell("code", text.replace("@COMMIT@", commit)) for text in code]
    return json.dumps(dict(nbformat=4, nbformat_minor=0, metadata=metadata, cells=cells), indent=1, ensure_ascii=False) + "\n"


def build(commit):
    return {
        "colab_fm.ipynb": notebook("colab_fm.ipynb", INTRO, (
            DECISIONS, SETTINGS, STEP1, STEP2, STEP3, STEP4, STEP5A, STEP5B, STEP6, STEP7, STEP8, STEP9, STEP10, STEP11,
            STEP12, STEP13, STEP14, STEP15, STEP16), commit, True),
        "colab_fm_evaluate.ipynb": notebook("colab_fm_evaluate.ipynb", E_INTRO, (
            E_FIRST, E_SETTINGS, E_STEP1, E_STEP2, E_STEP3, E_STEP4, E_STEP5, E_STEP6, E_STEP7, E_STEP8, E_STEP9),
            commit, False),
    }


def pinned_commit():
    """The commit the committed notebook pins, so a rebuild keeps it unless --commit is given."""
    path = HERE / "colab_fm.ipynb"
    found = re.search(r'COMMIT = \\"([0-9a-f]{40})\\"', path.read_text(encoding="utf-8")) if path.exists() else None
    return found.group(1) if found else "0" * 40


def check_pin(commit):
    """The pinned commit must hold exactly the hashed files of this checkout, or Colab would lock other code."""
    import hashlib
    import subprocess
    import fm_prepare as prep
    stale = []
    for name in (*prep.CODE_FILES, prep.PROTOCOL):
        shown = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=HERE, capture_output=True)
        if shown.returncode or hashlib.sha256(shown.stdout.replace(b"\r\n", b"\n")).hexdigest() != prep.text_hash(HERE / name):
            stale.append(name)
    if stale:
        raise SystemExit(f"the notebooks pin {commit}, where these hashed files differ from this checkout: {stale}")
    print("pin ok:", commit, "holds the", len(prep.CODE_FILES) + 1, "hashed files of this checkout")


def main():
    commit = sys.argv[sys.argv.index("--commit") + 1] if "--commit" in sys.argv else pinned_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("--commit needs a full 40-character commit hash")
    if "--check-pin" in sys.argv:
        return check_pin(commit)
    different = []
    for name, text in build(commit).items():
        path = HERE / name
        if "--check" in sys.argv:
            if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != text:
                different.append(name)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print("wrote", name, "pinned to", commit)
    if different:
        raise SystemExit(f"notebooks differ from fm_notebook.py, rebuild them: {different}")


if __name__ == "__main__":
    main()
