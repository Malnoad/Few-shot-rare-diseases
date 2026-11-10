# First Colab run: step-by-step guide

For a teammate running `colab_fm.ipynb` for Lam. You don't need to own the `octfsl` Drive folder or
know the code. The notebook loads four image models and times them on 100 images. It uses **no
labels and produces no scores**: these are first checks, not results. The evidence ends up in Lam's
Drive folder `octfsl/fm_v1/first-checks/`.

Who does what:

| Part | Who | When |
|---|---|---|
| 0. Share the folder and send the links | Lam | Once, before Part 1 |
| 1. Accounts, token and Drive shortcut | Runner | Once |
| 2. Run the notebook | Runner | Each attempt |
| 3. Check the result | Runner, then Lam | After each attempt |

The golden rule: **if a check fails, stop and tell Lam. Never edit a hash, a commit, or any cell
except the first one to make a check pass.** A failed check is useful evidence; a "fixed" one is not.

## Part 0 (Lam): before the runner starts

1. In Drive, inside `octfsl`, create a folder named exactly `fm_v1` (skip if it exists).
2. Sharing: `octfsl` is shared as **"Anyone with the link: Editor"** (Lam's choice, 10 Nov, made
   after weighing per-person Viewer/Editor sharing). Anyone holding that link can change or delete
   the raw data, checkpoints and earlier results, so send it privately and never put it in the
   repository, the PR or any public place.
3. Send the runner, privately, the `octfsl` folder link, plus the Colab link below and this guide.
   Your four terms decisions (10 Nov) are already filled in the notebook's first cell, so there is
   nothing to paste.

Colab link (pinned notebook version): <https://colab.research.google.com/github/Malnoad/Few-shot-rare-diseases/blob/8ee693670587ec47b8f48b8b65b5dd64bff80810/colab_fm.ipynb>

## Part 1 (runner): one-time setup

**1.1 RETFound access.** RETFound is gated: each Hugging Face account needs its own approval.
Sign in at huggingface.co (create a free account if needed), open
<https://huggingface.co/YukunZhou/RETFound_mae_natureOCT>, and fill in the access form (it asks
for name, affiliation and job title). Approval is automatic, according to the repository's public
metadata on 10 Nov; check that the page then shows the files before Part 2. Read the model card while
you are there and tell Lam if it states terms Lam has not seen.

**1.2 Token.** On Hugging Face: Settings → Access Tokens → create a token of type **Read**. Treat it
like a password: never paste it into a notebook cell, a chat or a screenshot.

**1.3 Colab secret.** Open the Colab link. In the left sidebar click the key icon (Secrets) → Add new
secret. Name: `HF_TOKEN`. Value: the token. Turn **Notebook access** on.

**1.4 Drive shortcut.** The notebook expects these two paths in *your* Drive:
`My Drive/octfsl/raw` and `My Drive/octfsl/fm_v1`. One shortcut to Lam's `octfsl` folder gives both.

1. Signed in with the Google account you will use in Colab, open the `octfsl` link Lam sent.
2. Click the folder name at the top → Organize → Add shortcut → **My Drive** (the top level, not a
   subfolder). Menu labels can differ slightly between Drive versions.
3. Check: My Drive now shows a shortcut named exactly `octfsl`, and inside it `raw` and `fm_v1`.
4. Make sure nothing else in My Drive is named `octfsl` (rename any folder of your own), so the
   notebook's path can only mean Lam's folder.

Step 3 of the notebook checks both paths, and that `fm_v1` is writable, within the first minutes. If
the shortcut doesn't work, you find out then, not at the end. Only write inside `fm_v1`; the notebook
never writes anywhere else, and nothing else in `octfsl` should be touched.

**1.5 GPU.** The notebook needs an **A100, L4 or T4** GPU. A100 and L4 usually need Colab Pro or
compute units; a free account usually gets a T4. Any of the three is fine for this run.

## Part 2 (runner): run it

1. Open the Colab link Lam sent. If Colab warns that the notebook is from GitHub, choose Run anyway.
   You don't need to save a copy; never save a copy back to GitHub.
2. Runtime → Change runtime type → GPU: **A100, L4 or T4** (not RTX PRO 6000 / Blackwell) → Save.
3. In the **first code cell only**, fill in two lines (leave `DECISIONS` exactly as it is; those are
   Lam's terms decisions):
   - `RUN_BY`: your name.
   - `RETFOUND_ACCESS`: your Hugging Face username and the date RETFound approved you, written out
     in full, for example `"Hugging Face account jdoe, access approved 2026-11-11"`. Step 1 stops if
     a `[` or `<` placeholder is left in.
4. Runtime → **Run all**.
5. When Step 3 asks to connect Google Drive, choose **your own** Google account and allow access.
   When Step 5b asks for access to the `HF_TOKEN` secret, allow it.
6. Keep the tab open and the computer awake. Don't run other GPU work in this runtime, and don't
   click individual cells while it runs. Download, unpacking and loading times are not yet measured;
   the first run will tell us.

What each step does, so you know where you are:

| Step | What it does | Healthy output ends with |
|---|---|---|
| 1 | Checks the first cell and the GPU | One line naming the GPU |
| 2 | Downloads the exact code and builds a separate Python | `code: 41782db…` and a `python … torch …` line |
| 3 | Checks Drive, copies 84,630 images from `raw`, disconnects Drive | `84630 manifest images present` |
| 4 | Rebuilds the image groups and rosters | `All 6 files match the laptop byte for byte` |
| 5 | Downloads the four models | One `downloaded … MB` line per file |
| 6 | Checks every model file | `Setup complete. No encoder has been run and no score exists.` |
| 7 | Writes one identity file per model | `spec written:` four times |
| 8 | Hides credentials and Drive from the model code | `Removed from the worker environment …` |
| 9 | Loads and times each model, one at a time | `exit code 0` four times |
| 10 | Prints the pilot report | Four model lines, then `No labels were used …` |
| 11 | Saves the evidence to Lam's `fm_v1` folder | `All four models loaded and timed.` |

## Part 3: check the result

**Runner.** Success is the last line of Step 11: `All four models loaded and timed. Tell Lam the
attempt name above.` Send Lam the attempt name (`first-check-…`). The Step 6 and Step 10 output are
saved in Drive, so you don't need to copy them, but pasting them in the message does no harm.

If Step 11 says `Pilot failed for [...]`, the evidence is still saved. Tell Lam which model and stop.

**Lam.** In `octfsl/fm_v1/first-checks/<attempt>/` you will find `step6_inventory.txt`,
`step10_pilots.txt`, `receipt.json` and `<attempt>.zip` (specs, download ledger, pilot receipts and
one log per model; no weights, no images). Download the zip and compare its SHA256 with
`zip_sha256` in the receipt, for example with
`D:/Few-shot/.venv/Scripts/python.exe -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <zip>`.
Step 11's own check reads the copy through the Colab Drive mount, so this download check is the
independent one. Then paste `step6_inventory.txt` and `step10_pilots.txt` into the project thread.

## When something goes wrong

| Where | Message | What to do |
|---|---|---|
| Step 1 | `Stopped on purpose: fill in [...]` | Fill in the named fields in the first cell, then Run all |
| Step 1 | `… still contain a [ or < placeholder` | Replace the example text with your real name, username or date |
| Step 1 | `No GPU` or `Blackwell GPU` | Change the runtime type (Part 2, step 2), then Run all |
| Step 2 | Any `FAILED:` line | Stop. Send Lam the whole Step 2 output |
| Step 3 | `… not found: add the octfsl shortcut` | Redo Part 1.4; ask Lam whether `octfsl/fm_v1` exists |
| Step 3 | `octfsl/fm_v1 is not writable` or a permission error | Ask Lam to check the link gives Editor access |
| Step 3 | `0 files` for a zip, or images missing | Stop. Send Lam the Step 3 output |
| Step 4 | `Differs from the laptop, stop and report` | **Stop.** Send Lam the output. Do not rerun with changes |
| Step 5 | `401` or `403` for the RETFound file | Access not approved yet, wrong token, or Notebook access off (Part 1.1–1.3) |
| Step 5 | `does not match its pinned SHA256` | **Stop.** Send Lam the output. Never edit the hash |
| Step 8 | `Drive is mounted` or `login is cached` | Runtime → Disconnect and delete runtime, then start again from Part 2 |
| Step 9 | `Another process is using the GPU` | Runtime → Restart session, then Run all |
| Step 9–11 | `exit code` other than 0, or `Pilot failed` | Let Step 11 finish (it saves the evidence), then tell Lam |
| Step 11 | `The Drive copy differs` or Drive won't connect | In the Files panel, download `/content/<attempt>.zip` and send it to Lam |
| Any | The runtime disconnects | Everything in `/content` is gone. Start again from Part 2 and tell Lam it happened |

## Rules

- Edit only `RUN_BY` and `RETFOUND_ACCESS` in the first cell. Don't change `DECISIONS`, `COMMIT`, any
  hash, or any other cell.
- Keep the token in Colab Secrets only. Don't share the notebook with outputs that show the secret
  panel, and don't download or share the model files.
- Nothing this notebook produces is a result or a score. Don't report numbers from it as findings.
- Don't connect Drive by hand during Steps 7–10; the notebook keeps Drive off while model code runs.

The rules behind these checks are in `docs/research/FM_RUNBOOK.md` and
`docs/research/FM_RISK_ACKNOWLEDGEMENT.md`; the evidence for the terms decisions is in
`docs/research/FM_TERMS_EVIDENCE.md`.
