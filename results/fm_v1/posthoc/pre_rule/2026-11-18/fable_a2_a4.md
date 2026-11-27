# Fable rebuttal of the Opus positions on A2 and A4, 18 Nov 2026

Read-only on the repository. Feasibility scripts and extracted lock files are in
`C:\Users\THINKB~1\AppData\Local\Temp\claude\c38f92eaa1d8ed7a\34630c3a-5fe8-47a5-a7e3-d2b9d1ac6536\scratchpad\fable_a4\`
(`check1.py`, `check2.py`, `lock\`). No pair of rare images was compared by me.

Tags: [file:line], [opened: URL], [run: script] = output of a feasibility script, [unverified].

One disclosure first. To test whether the Kermany calibration can work at all, I computed
same-patient against different-patient cosine separability on the 5,000 Kermany images in the
locked features (`check2.py`). These are Kermany-only numbers, not a leakage result on rare
images, but they exist before any P10 rule is committed. If any of them is ever reported, it
must be labelled "feasibility pilot, computed 18 Nov before the rule".

---

## A2: where P3 goes

### What Opus got wrong or left out

1. **"(d) is not allowed: outside-protocol analyses go to the supplement (A3, rule 2)" is a
   misquote.** Rule 2 says a post hoc result never replaces a main-text result, all post hoc
   results sit under one dated "post hoc" heading, and "Lam decides any other placement, with the
   mentor" [protocols.md:199-202]. Rule 4 only requires the label "outside the locked study"
   [protocols.md:204-206]. A3's table does not mention P3 at all [FM_AMENDMENTS.md:184-190]. The
   automatic "post hoc goes to the supplement" fallback applied only if the mentor had not
   answered [protocols.md:250-252]; the mentor answered on 17 Nov (uncommitted change note in
   protocols.md, 18 Nov entry). So a main-text figure is a bad idea, but it is Lam's and the
   mentor's call, not a rule violation.
2. **The proposed sentence sits in the wrong place.** Opus puts it in discussion/limitations.
   Rule 2 wants every post hoc result under the one post hoc heading [protocols.md:200]. A
   discussion sentence may point to it; it should not carry the P3 number itself.
3. **The main text does not need P3 to say this.** The locked nine-way result, which A3 places
   in the main text, already shows MIRAGE calling only 26 of 332 OCTDL normals normal
   [protocols.md:188; FM_AMENDMENTS.md:187]. P3 adds one thing: the collapse also happens when the
   examples are OCTID normals instead of Kermany normals (MIRAGE NORMAL recall 0.022 at K=8)
   [results/fm_v1/posthoc/P3_clinic_to_clinic_mh.csv, MIRAGE OCTID->OCTDL image K=8]. So the
   main-text sentence can rest on the locked number and cite P3 as supplementary support.
4. **"Their collapse happens exactly when queries are OCTDL" is true for MIRAGE only.**
   Inception-v3 is at chance in both directions (0.499 and 0.526 at K=8; reverse-direction
   recalls 0.465/0.586), so it shows no direction effect, just no signal
   [P3 csv, Inception rows]. RETFound also loses OCTDL NORMAL recall as K grows (0.61 at K=1 to
   0.42 at K=8) although it crops rather than squashes [P3 csv; fm_extract.py:85,386-387]. A
   pattern that gets worse with more examples (MIRAGE 0.605 at K=1 to 0.511 at K=8) looks like a
   fixed offset between the two clinics' images, which squashing is only one candidate cause of.
   The sentence "the two models that squash ... G1 tests whether squashing explains this"
   overstates what the data separates.
5. **G1 already covers most of the question.** G1's fixed rule reports "OCTDL normal
   destinations" on padded features [protocols.md:222 (G1 row); fm_posthoc.py:816-829]. Padded
   P3 is a second view of the same thing, not the only test.
6. **Not said: the padded-P3 rule is written after the unpadded P3 was seen.** It is fixed
   before any padded number, but it is chosen because of a seen result. Label it like P7:
   "rule fixed before computing, chosen after seeing P3". Precedent for adding a padded table to
   G1 by rule: P9, "If G1 runs, the same table on its padded MIRAGE and Inception-v3 features"
   [protocols.md, P9 row, uncommitted 18 Nov].
7. **Not said: G1 can be dropped.** Drop order is P2, P5 cut-off, P4, G1 [protocols.md:249]. A
   main-text sentence promising that "G1 tests this" must be written only after G1 has run.
8. **What padded P3 can and cannot show.** Padding changes examples and queries together. If
   the collapse stays: squashing is not the main cause. If it goes: consistent with geometry,
   but black bars are also a new input the models never saw. Either way it describes; it does
   not prove a cause.
9. "Ratio 2.75" for OCTDL is [unverified]; I did not check it. The "tier B/D" wording is not in
   the repository; the mentor's rule as recorded is "about 30 or fewer labelled images"
   (protocols.md, 18 Nov change note, uncommitted).

### Feasibility of (c), checked

- G1's output holds `features_external.npz` for all 781 OCTID/OCTDL images, so OCTID and OCTDL
  examples and queries are both there [fm_pad_extract.py:10,119,135]. G1 loads them already as
  `pad_ext` [fm_posthoc.py:847].
- `p3()` reads features inside the function (`study.external_features(arm)`)
  [fm_posthoc.py:707]; it needs a small change to accept a feature set, plus one test. No hashed
  `fm_*.py` file is touched. Laptop only, seconds. No extra GPU time.
- G1 has not run: no padded folder on the Drive mirror, no G1 file in
  `results/fm_v1/posthoc/` [run: ls]. So the rule can still be fixed first.
- Honest, if labelled as in point 6 and reported whichever way it comes out (rule 3).

### "Move P3 to the next phase" and "per patient only"

- P3 is computed; rule 3 and A1 rule 6 say it is released and never withdrawn
  [protocols.md:203; FM_AMENDMENTS.md:46-47]. It cannot move out. What can move is a proper
  version (patient IDs on both sides, labels checked): next-phase item 1 already asks for that
  [protocols.md:270].
- Per-patient and per-image numbers are almost the same (MIRAGE 0.519 against 0.511; DINOv2
  0.915 against 0.917) [P3 csv]. Reporting only one hides nothing and gains nothing. Keep both in
  the supplement table.

### Choices for Lam (A2)

| Choice | Cost | What it can claim |
|---|---|---|
| A. Supplement table only | none | Nothing in the main text; P3 is released as required |
| B. A + one main-text sentence built on the locked 26-of-332 number, pointing to P3 in the supplement | 15 min of writing | "MIRAGE's OCTDL normal failure does not depend on which clinic supplied the normal examples (supplement, outside the locked study)" |
| C. B + padded P3 added to G1's scoring, rule committed before G1 runs, labelled "chosen after seeing P3" | 1-2 h of code and test, no extra GPU; only if G1 runs | "With padding the collapse stayed / shrank"; a description, not a cause |
| D. A or B, and a properly designed two-clinic test goes into the next-phase paragraph | one sentence | Nothing now; honest pointer forward |

**Recommendation: C, with B as the fallback if G1 is dropped; add D's sentence in any case.**
Wording: "agreement between two clinics' uncommon-class macular-hole labels"; never "transfer",
never "rare"; and A1's required sentence stays [FM_AMENDMENTS.md:48-50].

---

## A4: leakage between development and hold-out without contacting Yoo

### Three facts Opus did not have

1. **The pixel-duplicate screen is already in the lock.** Before any score, all 146 rare images
   were screened for identical files, identical RGB pixels and near-duplicates (64-bit
   difference hash, distance <= 4) [FM_PROTOCOL.md:76-85; protocols.md:102-104]. Result: 0 pairs
   of every kind, in every block including fitting-against-legacy; the closest pair anywhere is
   at hash distance 5; all 146 components have size 1
   [lock\screen_summary.json; lock\components.csv, from input-lock.zip]. Opus's option 1 would
   mostly recompute a locked result. Its new parts (pHash, SSIM) need packages the venv does
   not have (`skimage`, `imagehash` missing; `cv2` 5.0.0 present) [run: import check].
2. **The rare images came from the web.** Yoo et al. describe them as "extracted from Google
   Images and Google search engine by searching for keywords", say ethics approval was not
   needed because they "used open web-based and deidentified data", and say the split was
   random and "exhibited no overlap" [opened: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7829497/fullTextXML,
   read through a summarising tool; Lam should check the three quotes in the PDF]. The files
   agree with this: all 146 are RGBA PNG, hand-cropped, with almost every image a different
   size (CSC 28 sizes for 32 images, widths 410-644; RP 29 for 31), no EXIF, file names
   `01.PNG`..`32.PNG` [run: check1.py]. The repository does not mention the web origin
   [run: grep of README.md, CONTEXT.md, protocols.md]. The same source gives per-class totals of
   30/30/24/19/16 (CSC/MH/MacTel/Stargardt/RP); the release has 32/31/29/23/31 [run: check1.py]
   - to verify, but if true the release is not the paper's exact set.
   Consequence: "leakage" here means (a) the same picture posted twice at different crops or
   sizes, (b) several panels of one case report (same patient, neighbouring slices or
   before/after treatment). It is not "two visits to one clinic".
3. **The binding protocol already says similarity cannot establish patients.** "Similar
   filenames or a perceptual hash do not establish patients; unknown rare patients remain
   unknown" [FM_PROTOCOL.md:73-74]; "This screen is not a validated duplicate or patient
   detector" [FM_PROTOCOL.md:85]. Any P10 group must be called a "similarity group" or "same
   picture", never a patient.

### Is the Kermany calibration sound? No, and I measured why

- **It barely works inside Kermany itself.** On the 5,000 Kermany images in the locked features
  (2,064 patients; 504 to 4,733 same-patient pairs per class), cosine similarity separates
  same-patient from different-patient pairs within a class with AUROC 0.72-0.83 (MIRAGE),
  0.69-0.73 (DINOv2), 0.68-0.72 (Inception-v3), 0.61-0.67 (RETFound). At a strict cut-off that
  only 0.1% of different-patient pairs exceed, only 13-22% (MIRAGE) or 4-8% (other models) of
  same-patient pairs are caught [run: check2.py; feasibility pilot, see disclosure]. A "nothing
  flagged" result would therefore mean almost nothing.
- **Opus's threshold ("the cosine above which 95% of pairs are same-patient") does not
  transfer.** That is a precision rule, and precision depends on how common same-patient pairs
  are. In the Kermany sample they are about 0.5% of within-class pairs (4,055 of 780,625 for
  CNV) [run: check2.py]. Among 20-30 web images per class the share is unknown and could be ten
  times higher or zero.
- **Different pictures.** Kermany images are native 512x496 or 768x496 greyscale JPEGs from one
  export pipeline [run: check1.py, 200 sampled test images]; rare images are web crops of mixed
  style. In such a set, feature similarity will largely follow source style (device, colour
  map, overlays), so it groups by website or device, not by patient. Published work with frozen
  foundation-model features on OCT reports 78% rank-1 re-identification within one device and
  "near chance across devices", and that needed a trained pair classifier
  [opened: https://www-live.dfki.de/web/forschung/projekte-publikationen/publikation/17187, abstract only].
- **Same volume against two visits.** Kermany names are CLASS-patient-index; the median patient
  has 5 images, the 99th percentile 184, the maximum over 800 [run: check1.py]. Nothing in the
  name says volume or visit. The literature notes that consecutive slices are "very similar in
  both visible structures and noise" and that 92% of Kermany v2 test images share a subject
  with training [opened: https://ar5iv.labs.arxiv.org/html/2202.12267]; in our manifest 559 of
  657 test patients are also in train or val [run: check1.py; compare CONTEXT.md:95, which
  counts 566 of 633 by bare ID]. The measured weak separability suggests Kermany same-patient
  pairs are a mix of near-identical slices and unrelated-looking ones; one threshold cannot
  describe both.
- **A defensible threshold, if one is wanted anyway:** fix the false-alarm rate, not the
  precision (for example "at most 1 in 1,000 known-different pairs exceeds it"), state the
  sensitivity measured on labelled data (4-22% above), and use the threshold only to choose
  pairs for review, not to declare patients. OCTDL also has patient IDs and its features are
  in the lock [protocols.md:216]; it could serve as a second calibration set, but it has the
  same transfer problem.

### Pseudo-patient groups (option 4): drop

Single-link connected components chain: A-B and B-C above the threshold put A and C together
even if they are far apart, and 20-30 images of one disease form a dense cloud. The locked
protocol used components only for confirmed or unresolved duplicate pairs
[FM_PROTOCOL.md:86-88], where chaining is harmless (the same picture is transitive). Options 5a
and 5b need only pairs, not groups.

### Sensitivity re-analysis (5a, 5b) and "the splits never move"

- **Compatible with rule 3** if it is a labelled re-score: `data/manifest.csv` is not touched,
  nothing is re-sampled, locked numbers are not replaced [AGENTS.md rule 3;
  FM_PROTOCOL.md:91 "Canonical test rows never move"; FM_AMENDMENTS.md:46-47]. P1's
  leave-one-image-out already re-scores with images left out [protocols.md:214]. 5b needs the
  empty-class rule of P4 [protocols.md:217].
- **But it is biased, and Opus does not say so.** The prototype classifier is right when a
  query is close to its class's examples. Removing the queries that are closest to an example
  removes the easy ones, so recall falls whether or not any patient is shared. A drop is not
  evidence of leakage; no drop is the only informative outcome. If computed from
  feature-similarity flags it must be called a "worst-case bound", and the flags should come
  from pixels or from a model other than the one scored.
- With 4-5 hold-out images per class, removing even two leaves a class at 2-3 images; the
  macro recall then moves in steps of 0.07-0.10 per image.

### Options Opus missed

- **Same-picture screen that survives cropping.** Keypoint matching (ORB features, then a
  geometric consistency check) finds one picture inside another at a different crop or scale;
  the locked hash cannot. Two crops of one B-scan share the same speckle grain; two different
  patients never do. So a high match count has an almost unambiguous meaning ("same picture"),
  and the cut-off can be set without Kermany: positives are each rare image against a re-cropped,
  re-scaled copy of itself; negatives are pairs from different disease folders. Runtime: all
  10,585 pairs in about one minute on the laptop [run: ORB timing on 40 Kermany images, 780
  pairs in 4.1 s]. It does not catch neighbouring slices or a second visit.
- **Image header data:** nothing usable. No EXIF; 2 of 146 files carry a creation time
  [run: check1.py].
- **Equal pixel sizes within a class** (two crops saved from one figure often share a size)
  and **neighbouring file numbers**: weak hints, usable only to add pairs to the review sheet.
- **Curvature, vessel shadows, lesion position:** these are what the reviewer looks at; they
  identify neighbouring slices of one scan fairly well and nothing beyond that.

### What a non-expert visual review can honestly claim

Can claim: "pair X shows the same picture or an overlapping crop" and, for obvious cases,
"neighbouring slices of one scan" (same outline, same lesion, same overlay style). Cannot
claim: same patient at another visit, the other eye, or "different patients". The locked
protocol already frames it this way: "A non-expert duplicate reviewer is not diagnosing
disease" and unresolved pairs count as linked [FM_PROTOCOL.md:86-90]. To make it defensible:
the sheet hides which image is hold-out and shows no model score; a fixed number of random
same-class pairs is mixed in as decoys; answers are yes / unsure / no, saved as a CSV; "unsure"
counts as linked.

### Minimum viable version before 21 Nov

1. **Wording (no computation):** hold-out = "our own random seed-42 hold-out from the same
   web-collected image set; patient and source of each image unknown"; cite the locked screen
   (0 pairs, closest hash distance 5) and Yoo's own "no overlap" statement as image-level only;
   add the web origin to W3 [protocols.md:228-229].
2. **P10 (about half a day, laptop):** rule committed first. (i) crop-robust same-picture
   screen on all within-class pairs; (ii) review sheet = every flagged pair + the N closest
   development-to-hold-out pairs per class by pixel match (N fixed in the rule, e.g. 5) +
   decoys; Lam reviews blind; (iii) only if a pair is marked yes or unsure: re-score hold-out
   without those hold-out images and development without those query-example pairs, labelled
   sensitivity, beside the locked numbers. If nothing is marked, no re-score exists.
3. Not before 21 Nov: Kermany-calibrated thresholds, pseudo-patients, feature-based worst-case
   bound.

### Choices for Lam (A4)

| Choice | Cost | What it can claim |
|---|---|---|
| 1. Wording only, citing the locked screen and the web origin | 1 h | "No re-encoded duplicate among the 146; whether images share a patient or a source is unknown" |
| 2. Choice 1 + P10 same-picture screen + blind review of a fixed pair list; re-score only for confirmed or unsure pairs | half a day, laptop, before 21 Nov | "No (or n) overlapping pictures or obvious neighbouring slices between development and hold-out; here is what the numbers become without them". Not "no shared patients" |
| 3. Choice 2 + feature-based worst-case bound (remove each hold-out image's closest development neighbours) | + half a day | "Even without its closest neighbours the result is X": an upper bound on any leakage effect, biased downwards by design |
| 4. Opus's full plan (Kermany-calibrated patient threshold, pseudo-patients, 5a/5b) | a day or more | Little: the calibration catches 4-22% of known same-patient pairs in Kermany itself and does not carry over to web images |

**Recommendation: choice 2.** Choice 1 is the floor and should be done whatever else happens.
Choice 3 only if time remains after G1, and only with the "worst-case, biased" label. Not
choice 4.

Sources opened:
- https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7829497/fullTextXML (Yoo et al. 2021, full text)
- https://ar5iv.labs.arxiv.org/html/2202.12267 (data leakage in OCT classification, Scientific Data 2022)
- https://www-live.dfki.de/web/forschung/projekte-publikationen/publikation/17187 (re-identification from retinal OCT with foundation models, BVM 2026, abstract)
