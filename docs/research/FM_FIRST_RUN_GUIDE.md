# Colab runs: step-by-step guide

For a teammate running the study's two notebooks for Lam. You don't need to own the `octfsl` Drive
folder or know the code.

| Notebook | What it does | Needs | Produces scores? |
|---|---|---|---|
| `colab_fm.ipynb` | Loads four image models, times them, locks the study's inputs, saves every model's features | GPU, Hugging Face token, about 1 to 1.5 hours (estimate) | **No** |
| `colab_fm_evaluate.ipynb` | Development, nomination and reports, from the saved features | No GPU, no token | **Yes: the first ones** |

Everything is saved to Lam's Drive folder `octfsl/fm_v1/final/<attempt>/`.

Who does what:

| Part | Who | When |
|---|---|---|
| 0. Share the folder and send the links | Lam | Once, before Part 1 |
| 1. Accounts, token and Drive shortcut | Runner | Once |
| 2. Run `colab_fm.ipynb` | Runner | One session |
| 3. Check the result, approve the lock, record the timestamp | Runner, then Lam | After Part 2 |
| 4. Run `colab_fm_evaluate.ipynb` | Runner, with Lam at Step 6 | After Lam's approval |

The golden rule: **if a check fails, stop and tell Lam. Never edit a hash, a commit, or any cell
except the first one to make a check pass.** A failed check is useful evidence; a "fixed" one is not.

## Part 0 (Lam): before the runner starts

1. In Drive, inside `octfsl`, create a folder named exactly `fm_v1` (skip if it exists).
2. Sharing: `octfsl` is shared as **"Anyone with the link: Editor"** (Lam's choice, 10 Nov). Anyone
   holding that link can change or delete the raw data, checkpoints and results, so send it privately
   and never put it in the repository, the PR or any public place.
3. Send the runner, privately, the `octfsl` folder link, the two Colab links and this guide. Your
   terms decisions and your external-data decision are already filled in the first cell.
4. Keep the repository public until Part 2 has finished: the notebooks clone it without a password.

Colab links (pinned notebook versions; they work once Lam has had the branch pushed to GitHub):

- Part 2: <https://colab.research.google.com/github/Malnoad/Few-shot-rare-diseases/blob/c0a53adc8070d9ed0dc0fffea99ae5421c813753/colab_fm.ipynb>
- Part 4: <https://colab.research.google.com/github/Malnoad/Few-shot-rare-diseases/blob/c0a53adc8070d9ed0dc0fffea99ae5421c813753/colab_fm_evaluate.ipynb>

## Part 1 (runner): one-time setup

**1.1 RETFound access.** RETFound is gated: each Hugging Face account needs its own approval.
Sign in at huggingface.co (create a free account if needed), open
<https://huggingface.co/YukunZhou/RETFound_mae_natureOCT>, and fill in the access form (it asks
for name, affiliation and job title). Approval is automatic, according to the repository's public
metadata on 10 Nov; check that the page then shows the files before Part 2. Read the model card while
you are there and tell Lam if it states terms Lam has not seen.

**1.2 Token.** On Hugging Face: Settings → Access Tokens → create a token of type **Read**. Treat it
like a password: never paste it into a notebook cell, a chat or a screenshot.

**1.3 Colab secret.** Open the first Colab link. In the left sidebar click the key icon (Secrets) →
Add new secret. Name: `HF_TOKEN`. Value: the token. Turn **Notebook access** on.

**1.4 Drive shortcut.** The notebooks expect these two paths in *your* Drive:
`My Drive/octfsl/raw` and `My Drive/octfsl/fm_v1`. One shortcut to Lam's `octfsl` folder gives both.

1. Signed in with the Google account you will use in Colab, open the `octfsl` link Lam sent.
2. Click the folder name at the top → Organize → Add shortcut → **My Drive** (the top level, not a
   subfolder). Menu labels can differ slightly between Drive versions.
3. Check: My Drive now shows a shortcut named exactly `octfsl`, and inside it `raw` and `fm_v1`.
4. Make sure nothing else in My Drive is named `octfsl` (rename any folder of your own), so the
   notebook's path can only mean Lam's folder.

Only write inside `fm_v1`; the notebooks never write anywhere else.

**1.5 GPU.** `colab_fm.ipynb` needs an **A100, L4 or T4** GPU. A100 and L4 usually need Colab Pro or
compute units; a free account usually gets a T4. Any of the three works, but a T4 is slower and a
free session can disconnect; an L4 is the safer choice (the first checks ran on one).

## Part 2 (runner): run `colab_fm.ipynb`

Pick a time when you can keep the tab open for about two hours. **The runtime must stay alive from
Step 10 to Step 16**: features can only be extracted in the same runtime that made the lock.

1. Open the Colab link. If Colab warns that the notebook is from GitHub, choose Run anyway.
   You don't need to save a copy; never save a copy back to GitHub.
2. Runtime → Change runtime type → GPU: **A100, L4 or T4** (not RTX PRO 6000 / Blackwell) → Save.
3. In the **first code cell only**, fill in two lines (leave everything else exactly as it is):
   - `RUN_BY`: your name.
   - `RETFOUND_ACCESS`: your Hugging Face username and the date RETFound approved you, written out
     in full, for example `"Hugging Face account jdoe, access approved 2026-11-11"`.
4. Runtime → **Run all**.
5. When a step asks to connect Google Drive, choose **your own** Google account and allow access.
   This happens at Steps 3, 13 and 16. When Step 5b asks for the `HF_TOKEN` secret, allow it.
6. Keep the tab open and the computer awake. Don't run other GPU work in this runtime, and don't
   click individual cells while it runs.

What each step does, so you know where you are:

| Step | What it does | Healthy output ends with |
|---|---|---|
| 1 | Checks the first cell and the GPU | One line naming the GPU |
| 2 | Downloads the exact code and builds a separate Python | `code: …` and a `python … torch …` line |
| 3 | Checks Drive, copies 84,630 images from `raw`, disconnects Drive | `84630 manifest images present` |
| 4 | Rebuilds the image groups and rosters | `All 6 files match the laptop byte for byte` |
| 5 | Downloads the four models | One `downloaded … MB` line per file |
| 6 | Checks every model file | `Setup complete. No encoder has been run and no score exists.` |
| 7 | Writes one identity file per model | `spec written:` four times |
| 8 | Downloads the external datasets OCTID and OCTDL | Five counts, then `External images are staged.` |
| 9 | Hides credentials and Drive from the model code | `Removed from the worker environment …` |
| 10 | Final pilots: loads and times each model, one at a time | `exit code 0` four times |
| 11 | Prints the pilot report | Four model lines, each with `repeat check passed` |
| 12 | Makes the input lock | A box with `INPUT LOCK SHA256` |
| 13 | Saves the lock to Drive | `Send Lam now: attempt …` |
| 14 | Compares external images with the locked ones | One line per source with its query count |
| 15 | Extracts features, model by model (Inception is the slow one, about half an hour) | `exit code 0` eight times |
| 16 | Saves features and evidence to Drive | `All done. Send Lam:` with the attempt and the SHA256 |

At Step 13 you can already send Lam the attempt name and the SHA256, while Step 15 runs.

## Part 3: check the result, approve, timestamp

**Runner.** Success is the end of Step 16: `All done. Send Lam:` followed by the attempt name
(`final-…`) and the input-lock SHA256. Send both to Lam.

**Lam.**

1. In `octfsl/fm_v1/final/<attempt>/` check that `receipt.json` says `"complete": true` and read
   `step11_pilots.txt` and `step14_external.txt`. Download `input-lock.zip` and compare its SHA256
   with the one the runner sent, for example with
   `D:/Few-shot/.venv/Scripts/python.exe -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <zip>`.
   This download check is the independent one.
2. Approve that exact SHA256 in writing (a dated message in the project thread).
3. Record the public timestamp (amendment A2, A-8): deposit `lock-hashes-<attempt>.txt` from the
   same Drive folder on OSF or Zenodo. The notebook wrote it; it holds the lock SHA256, the protocol
   SHA256, the SHA256 of `FM_AMENDMENTS.md` and the code commit, and no data. If the deposit can't
   be made yet, email that file to the mentor and the team first and deposit afterwards
   (amendment A3, S-5).
4. Tighten the `octfsl` sharing: keep Editor only for the account that will run Part 4, by address,
   and remove the open Editor link (tracker C-5).
5. Only now send the runner the go-ahead for Part 4, with the three texts for its first cell.

## Part 4: run `colab_fm_evaluate.ipynb`

This notebook produces the study's first scores. It needs no GPU and no token.

1. Open the second Colab link. A CPU runtime is enough.
2. Fill in the first cell with what Lam sent:
   - `ATTEMPT`: the attempt name from Part 2.
   - `LOCK_SHA256`: the SHA256 Lam approved.
   - `LOCK_APPROVAL`: Lam's approval, with name and date.
   - `TIMESTAMP_RECORD`: where and when the hashes were deposited or emailed.
   - `RUN_BY`: your name.
   Leave `REPORT_AFTER_NOMINATION = False` and `OMIT_LOGISTIC_REASON = ""`.
3. Runtime → **Run all**. Allow Drive access when asked (Steps 3 and 6).
4. **Step 6 stops on purpose** after printing the nomination: which model the development folds
   selected, with every model's scores. Send that output to Lam. Do not close the runtime.
5. When Lam has read it, set `REPORT_AFTER_NOMINATION = True` in the first cell, run that cell, then
   click Step 6 and choose Runtime → **Run cell and below** (called "Run after" in some Colab versions).
6. Step 9 ends with `All done. Tell Lam: results saved as results-….zip`.

If the runtime was lost after Step 6, start Part 4 again with `REPORT_AFTER_NOMINATION = True`. The
notebook finds the saved nomination on Drive and reuses it; it never selects a model twice.

| Step | What it does | Healthy output ends with |
|---|---|---|
| 1 | Checks the first cell | `First cell complete.` |
| 2 | Downloads the exact code and builds the same Python | A `python … numpy … scipy …` line |
| 3 | Copies the lock and features from Drive and checks them | `… archives match the receipt; the lock is the one Lam approved.` |
| 4 | Restores the lock, unpacks the features | `Lock restored. Available models: …` |
| 5 | Development episodes for every model | `development complete:` once per model |
| 6 | Nominates one model, saves the decision, stops | The nomination, then `Stopped on purpose` |
| 7 | Legacy, nine-way and external reports | `reports complete:` once per model |
| 8 | Summary tables | A list of CSV files |
| 9 | Saves everything to Drive | `All done. Tell Lam: …` |

## When something goes wrong

| Where | Message | What to do |
|---|---|---|
| Step 1 | `Stopped on purpose: fill in [...]` | Fill in the named fields in the first cell, then Run all |
| Step 1 | `… still contain a [ or < placeholder` | Replace the example text with your real name, username or date |
| Step 1 | `No GPU` or `Blackwell GPU` | Change the runtime type (Part 2, step 2), then Run all |
| Step 2 | Any `FAILED:` line | Stop. Send Lam the whole Step 2 output |
| Step 3 | `… not found: add the octfsl shortcut` | Redo Part 1.4; ask Lam whether `octfsl/fm_v1` exists |
| Step 3 | `octfsl/fm_v1 is not writable` or a permission error | Ask Lam to check the link gives Editor access |
| Step 4 | `Differs from the laptop, stop and report` | **Stop.** Send Lam the output. Do not rerun with changes |
| Step 5 | `401` or `403` for the RETFound file | Access not approved yet, wrong token, or Notebook access off (Part 1.1–1.3) |
| Step 5 | `does not match its pinned SHA256` | **Stop.** Send Lam the output. Never edit the hash |
| Step 8 | `The OCTID release changed` or `External cohorts differ` | **Stop.** Send Lam the output |
| Step 8 | A download error or timeout | Run Step 8 again; files already verified are skipped |
| Step 9 | `Drive is mounted` or `login is cached` | Runtime → Disconnect and delete runtime, then start again from Part 2 |
| Step 10 | `Another process is using the GPU` | Runtime → Restart session, then Run all |
| Step 11 | `Pilot failed for [...]. … one retry` | Run Step 10, then Step 11, once more, changing nothing. Tell Lam it happened |
| Step 11 | `Pilot failed twice` | **Stop** and tell Lam. Keep the runtime open. Only Lam decides what happens next |
| Step 12 | Any `FAILED:` line | **Stop.** Send Lam the whole Step 12 output. Keep the runtime open |
| Step 13, 16 | `The Drive copy … differs` or Drive won't connect | Run the step again. If it still fails, download the named file from the Files panel and send it to Lam |
| Step 15 | `Extraction failed for …` | Run Step 15 again once, changing nothing. If it fails again, run Step 16 and tell Lam |
| Part 2, any | The runtime disconnects before Step 16 has finished | Everything in `/content` is gone. Tell Lam; the session starts again from Part 2 and makes a new lock |
| Part 4, Step 3 | `… is not the lock of this attempt` or `differs from the saved receipt` | **Stop.** Check `ATTEMPT` and `LOCK_SHA256` with Lam |
| Part 4, Step 4 | `This runtime has … the lock has …` | **Stop.** Colab's Python changed since the lock. Tell Lam |
| Part 4, any | The runtime disconnects | Start Part 4 again; nothing on Drive is lost |

## Rules

- In `colab_fm.ipynb` edit only `RUN_BY` and `RETFOUND_ACCESS`. In `colab_fm_evaluate.ipynb` edit only
  the first cell. Don't change `DECISIONS`, `COMMIT`, any hash, or any other cell.
- `UNAVAILABLE`, `REPORT_AFTER_NOMINATION` and `OMIT_LOGISTIC_REASON` are Lam's to decide. Change
  them only on Lam's written word.
- Keep the token in Colab Secrets only. Don't share the notebook with outputs that show the secret
  panel, and don't download or share the model files.
- Nothing `colab_fm.ipynb` prints is a result. What `colab_fm_evaluate.ipynb` prints is a result:
  send it to Lam and nobody else, and never rerun it to "get a better number".
- Don't connect Drive by hand; the notebooks connect and disconnect it themselves, and keep it off
  while model code runs.

The study in plain words is in `protocols.md` at the top of the repository; every command and
guard is described in `docs/research/FM_RUNBOOK.md`.
