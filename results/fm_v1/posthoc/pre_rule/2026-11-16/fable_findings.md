# Fable 5.2 findings: pretraining exposure, domain shift, interpretation

Date: 16 November 2026. Archive read: `results-20261114T093033Z.zip` (read-only).
Scripts: `scratchpad\fable_inv\` (`extract.py`, `conf.py`, `outs.py`, `conf5.py`, `grab.py`, `peek.py`).
Nothing in the repo, `data/`, `results/` or `drive/` was changed. No P1-P6 number was computed
(no bounds, no leave-one-out, no October heads, no clinic-to-clinic test, no size-right/wrong split,
no score gaps or cut-offs, no prototype-versus-logistic table).

Tags: **[pre-specified output]** = a number the locked run wrote; **[derived by me]** = my own sum
or difference of locked numbers (none falls under P1-P6; each is flagged so Lam can decide whether
it needs a post hoc label); **[paper, opened]** = I downloaded the full text and quote it;
**[unverified]** = not confirmed in a source I opened.

Sources opened (Europe PMC full-text XML unless noted): RETFound PMC10550819; MIRAGE PMC12462498;
OCTDL PMC11009408; Yoo et al. PMC7829497; DINOv2 arXiv 2304.07193v2 (HTML). Not opened
successfully: the Mendeley Kermany page gave no counts; the Keras Inception-v3 page gave no text.

---

## 1. Pretraining exposure (W1)

| Model | Pretraining data | Kermany 2018 | OCTID | OCTDL | Web images like the Yoo rare set | Source and quote |
|---|---|---|---|---|---|---|
| RETFound (Nature OCT) | ImageNet-1k, then 736,442 OCT: 627,133 Moorfields MEH-MIDAS + 109,309 from Kermany | **Yes** | No (downstream test set only) | No (not mentioned; paper predates OCTDL) | No (by the paper's data list) | [paper, opened] "627,133 (85.2%) OCTs are from Moorfields Diabetic imAge dataSet (MEH-MIDAS) ... and 109,309 (14.8%) OCTs that come from ref. 34"; ref. 34 = Kermany et al., Cell 2018. "successively on natural images (ImageNet-1k) followed by retinal images" |
| MIRAGE-Base | 261,184 OCT-SLO-layer triplets from the in-house VIBES registry, Medical University of Vienna, 2007-2021 | **No** | **No** | **No** | No (by the paper's data list) | [paper, opened] "MIRAGE was pretrained on a large in-house dataset of multimodal retinal images, the Vienna Imaging Biomarker Eye Study (VIBES) registry of the Macula Clinic at the Medical University of Vienna"; "consists of 261,184 samples from 42,082 patients and 75,653 unique eyes" |
| DINOv2-Base | LVD-142M: curated sets (ImageNet-22k, ImageNet-1k train, Google Landmarks, fine-grained sets) plus images retrieved from 1.2 billion crawled web images | Unknown | Unknown | Unknown (LVD-142M predates OCTDL's 2024 release, so unlikely) | **Unknown, cannot be excluded** | [paper, opened] "we collect a raw unfiltered dataset of images from a publicly available repository of crawled web data. From each web page in the repository, we extract URL links of images from <img> tags"; curated list "contains ImageNet-22k, the train split of ImageNet-1k, Google Landmarks and several fine-grained datasets". The word "medical" does not occur in the paper text |
| Inception-v3 (Keras) | ImageNet (ILSVRC-2012, 1,000 classes) | No | No | No | No | Repo code: `stage3_classifier.py:105` `weights="imagenet"`; `fm_extract.py:36`. That ImageNet-1k is from 2012 and has no OCT class is my background knowledge: [unverified], Keras page not readable |

Further points.

- **RETFound saw the whole Kermany release, test folder included (inference, confidence high).**
  RETFound's 109,309 equals the total MIRAGE's Table 9 gives for the full Kermany dataset:
  "Kermany ... 109,309 [samples] 4686 [patients]" [paper, opened, both]. A count equal to the whole
  release leaves no room for a held-out test folder. What stays [unverified]: file-level overlap
  with our manifest. We use Mendeley v2 (84,484 common images, `CONTEXT.md:95`); that the v2 files
  are a subset of the 109,309-image release is likely but I did not check it.
- RETFound's exposure is label-free (masked-image reconstruction). The checkpoint never saw
  Kermany's labels [paper, opened: "unannotated retinal images for SSL"].
- **MIRAGE's own benchmark used Kermany, OCTID and OCTDL as downstream tasks** [paper, opened: "8
  public and 1 in-house: Duke iAMD, GAMMA, Harvard Glaucoma, Kermany, Noor Eye Hospital, OCTDL,
  OCTID, OLIVES, and OPTIMA9C"]. The released weights are the pretraining weights, so this is not
  image exposure. Whether pretraining choices were tuned on those benchmark results is [unverified].
- **MIRAGE's pretraining devices are Cirrus and Spectralis only** [paper, opened: Table 7 "Cirrus
  (54.4%) Spectralis (45.6%)"; "Images were acquired with Cirrus ... and Spectralis devices"].
  Kermany is Spectralis, OCTID is Cirrus, OCTDL is Optovue Avanti RTVue XR [paper, opened: MIRAGE
  Table 9; OCTDL paper "obtained with an Optovue Avanti RTVue XR"; RETFound "Reference 34 contains
  images from SPECTRALIS"]. So MIRAGE has no data overlap with any test source, but two of the
  three sources come from device families it was pretrained on and OCTDL does not.
- **MIRAGE's pretraining contains our rare diseases, unlabelled** [paper, opened: Table 8,
  "Chorioretinopathia centralis serosa 4.61 ... Macular telangiectasia 1.39 ... Macular hole,
  lamellar hole, NH defect 1.12 ... Coat, Best, and Stargardt diseases, Terson syndrome 0.33";
  labels available for 30.8% of samples]. Retinitis pigmentosa is not in that table. This is
  disease familiarity, which is what a retinal model is for, not leakage.
- RETFound's MEH-MIDAS devices: "topcon 3DOCT-2000SA (Topcon), CLARUS (ZEISS) and Triton (Topcon)"
  [paper, opened]. The cohort is patients with diabetes.
- Yoo rare images: "extracted from Google Images and Google search engine by searching for
  keywords" [paper, opened]. Their devices are [unverified] and probably mixed.

Correction for the repo notes: `FM_EXTERNAL_MAPPING.md:130` and `FM_RULE_TRACKER.md:46` still say
OCTID/OCTDL exposure is "unverified". On the papers' own data lists it is **no** for RETFound and
MIRAGE. `protocols.md` section 8's W1 gap row can be closed with the table above.

---

## 2. Domain shift, from pre-specified outputs only

All counts below are the `confusion` rows of `runs/<arm>/.../nine_way/metrics.json`
[pre-specified output]; the percentages and the "common/rare" subtotals are [derived by me] as
plain sums. The aggregate destination table that P5 plans already exists in the locked output;
only the per-scan score gaps and cut-off would be new, and I did not compute them.

### 2a. Where normal scans go (nine-way audit)

Columns: NORMAL / CNV / DME / DRUSEN / CSC / MH / MacTel / Stargardt / RP.

| Source (n) | Model | NORMAL | CNV | DME | DRUSEN | CSC | MH | MacTel | Stargardt | RP | Common disease | Rare |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OCTDL (332) | MIRAGE | 26 | 148 | 118 | 10 | 0 | 6 | 24 | 0 | 0 | 276 (83%) | 30 (9%) |
| | Inception-v3 | 66 | 196 | 21 | 38 | 0 | 0 | 0 | 0 | 11 | 255 (77%) | 11 (3%) |
| | RETFound | 137 | 8 | 167 | 2 | 0 | 0 | 12 | 1 | 5 | 177 (53%) | 18 (5%) |
| | DINOv2 | 145 | 29 | 1 | 16 | 6 | 0 | 10 | 58 | 67 | 46 (14%) | 141 (42%) |
| OCTID (206) | MIRAGE | 202 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 4 |
| | Inception-v3 | 182 | 0 | 9 | 0 | 12 | 0 | 3 | 0 | 0 | 9 | 15 |
| | RETFound | 189 | 0 | 3 | 14 | 0 | 0 | 0 | 0 | 0 | 17 | 0 |
| | DINOv2 | 188 | 1 | 0 | 0 | 0 | 0 | 3 | 3 | 11 | 1 | 17 |
| Kermany (250, all) | MIRAGE | 250 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| | Inception-v3 | 223 | 0 | 9 | 11 | 0 | 0 | 5 | 1 | 1 | 20 | 7 |
| | RETFound | 215 | 1 | 2 | 31 | 0 | 0 | 1 | 0 | 0 | 34 | 1 |
| | DINOv2 | 218 | 5 | 3 | 14 | 2 | 0 | 1 | 0 | 7 | 22 | 10 |

**Answer.** For three of four models the OCTDL failure is mostly **normal -> common disease**:
MIRAGE 276 of 306 errors, Inception-v3 255 of 266, RETFound 177 of 195. DINOv2 is the exception:
141 of 187 errors go to a rare class, almost all Stargardt (58) or RP (67). The common class that
absorbs the errors differs by model: MIRAGE CNV and DME, Inception-v3 CNV, RETFound DME.
Confidence: high for the counts; they are single deterministic outputs on 332 images from 110
patients, so the per-patient picture may differ.

### 2b. Where external rare images go

Nine-way audit [pre-specified output]:

| True class (n) | Model | NORMAL | CNV | DME | DRUSEN | CSC | MH | MacTel | Stargardt | RP |
|---|---|---|---|---|---|---|---|---|---|---|
| OCTID CSC (102) | MIRAGE | 8 | 28 | 23 | 2 | **23** | 1 | 1 | 15 | 1 |
| | DINOv2 | 11 | 19 | 1 | 2 | **30** | 0 | 3 | 30 | 6 |
| | RETFound | 13 | 12 | 72 | 5 | **0** | 0 | 0 | 0 | 0 |
| | Inception-v3 | 3 | 16 | 26 | 11 | **22** | 0 | 2 | 14 | 8 |
| OCTID MH (102) | MIRAGE | 8 | 10 | 72 | 1 | 0 | **10** | 0 | 1 | 0 |
| | DINOv2 | 4 | 2 | 6 | 5 | 2 | **48** | 23 | 2 | 10 |
| | RETFound | 8 | 3 | 80 | 11 | 0 | **0** | 0 | 0 | 0 |
| | Inception-v3 | 6 | 5 | 45 | 2 | 2 | **27** | 0 | 1 | 14 |
| OCTDL MH (39) | MIRAGE | 0 | 6 | 30 | 0 | 0 | **3** | 0 | 0 | 0 |
| | DINOv2 | 0 | 0 | 0 | 0 | 1 | **26** | 8 | 2 | 2 |
| | RETFound | 0 | 0 | 36 | 0 | 0 | **0** | 0 | 0 | 3 |
| | Inception-v3 | 2 | 25 | 11 | 0 | 1 | **0** | 0 | 0 | 0 |

Five-way output shares, prototype head, from `reports/external/external_transfer_outputs.csv`
[pre-specified output; shares = mean count / n, derived by me]:

| True class | Model | K=1 correct | K=8 correct | K=16 correct | Largest wrong class at K=16 |
|---|---|---|---|---|---|
| OCTID CSC | MIRAGE | .44 | .57 | .61 | Stargardt .26 |
| | DINOv2 | .16 | .25 | .29 | Stargardt .54 |
| | Inception-v3 | .27 | .41 | .45 | Stargardt .32 |
| | RETFound | .12 | .02 | .00 | Stargardt .61 |
| OCTID MH | MIRAGE | .39 | .70 | .77 | MacTel .10 |
| | DINOv2 | .24 | .37 | .42 | MacTel .30 |
| | Inception-v3 | .31 | .45 | .47 | RP .30 |
| | RETFound | .14 | .15 | .01 | Stargardt .45, MacTel .43 |
| OCTDL MH | MIRAGE | .20 | .25 | .26 | **MacTel .51** (was .26 at K=1) |
| | DINOv2 | .38 | .67 | .69 | MacTel .18 |
| | Inception-v3 | .24 | .38 | .39 | CSC .43 |
| | RETFound | .18 | .18 | .03 | Stargardt .63 |

Reading:

- In the five-way task MIRAGE transfers to OCTID (Cirrus) and improves with K, but on OCTDL
  macular holes it stays near chance (0.20-0.26) while its MacTel share doubles with K. More
  examples make the wrong answer more stable, which is a systematic offset and not noise.
  Confidence: medium-high (39 images, 28 patients).
- In the nine-way audit, the two OCT-pretrained models send external macular holes to **DME**:
  MIRAGE 72/102 and 30/39, RETFound 80/102 and 36/39. Cystoid spaces at the edge of a macular
  hole resemble diabetic macular oedema, so this may be a morphological confusion and not only a
  source effect. That clinical reading is mine and [unverified]; it is a question for a clinician
  (Q5).

### 2c. Which explanations the locked numbers rule in or out

Devices [paper, opened]: Kermany Spectralis; OCTID Cirrus HD-OCT (one device, per
`FM_EXTERNAL_MAPPING.md:43` and MIRAGE Table 9); OCTDL Optovue Avanti RTVue XR; Yoo rare images
unknown [unverified]. Geometry from `reports/geometry/*/geometry_summary.csv` [pre-specified
output]: OCTDL normals median 882 x 321 (aspect 2.75), OCTID 750 x 500 (1.5), Kermany 512 x 496
(1.03). MIRAGE and Inception-v3 squash the whole image to a square (retained fraction 1.0);
DINOv2 and RETFound centre-crop and keep a median 28% of an OCTDL normal against 74% of a Kermany
normal and 51% of an OCTID normal.

| Explanation | Status | Evidence |
|---|---|---|
| The failure is specific to OCTDL, not to "any external source" | **Ruled in** (high) | OCTID normal recall is 182-202 of 206 for all four models; OCTDL is 26-145 of 332 for all four |
| It needs retinal pretraining or MIRAGE's missing device (RTVue not in VIBES) | **Not necessary** (high) | Inception-v3, with no OCT pretraining, also collapses (66/332) and in the same direction (normal -> CNV). Device absence may add to MIRAGE's degree; the numbers cannot size that |
| MIRAGE's per-image min-max stretch is the cause | **Not necessary; contribution open** (medium) | Inception-v3 has no stretch and fails the same way. MIRAGE with the same stretch is perfect on Kermany (250/250) and near-perfect on OCTID (202/206), so the stretch alone does not break normals. An interaction with OCTDL's intensity range cannot be excluded without re-extraction |
| Aspect-ratio handling (squash versus crop) | **Open, suggestive** (low-medium) | The two squash models have the lowest OCTDL normal recall (MIRAGE 8%, Inception-v3 20%) and both send normals to CNV; the two crop models keep 41-44%. Four models only, and the split is confounded with everything else that differs between them |
| Kermany-only common supports | **Open, plausible; cannot be separated here** (medium) | Every NORMAL/CNV/DME/DRUSEN prototype is the mean of 1,000 Spectralis images. All four models lose OCTDL normals, each to a different class, which fits "far from the Kermany normal prototype, nearest class is arbitrary". P3 is the planned test of this |
| Label definitions (OCTDL "normal" is not normal) | **Unlikely as the main cause** (medium) | The destination class differs by model (CNV, DME, Stargardt/RP); a real shared pathology would pull the models toward the same class. OCTDL labels were confirmed by clinicians [paper, opened via repo note]. Not excluded for a minority of scans |
| Pretraining exposure | **Ruled out as a cause of the OCTDL result** (high) | No model pretrained on OCTDL (section 1) |
| Patient clustering inflates the counts | **Open** | 332 images from 110 patients; the locked interval for MIRAGE normal recall is 0.049-0.113 [pre-specified output], so clustering does not rescue it |

---

## 3. Exposure versus shift

Direction: exposure would raise in-source scores; shift lowers external ones.

1. **RETFound and Kermany: exposure is real but shows no inflation.** RETFound pretrained on
   Kermany images, yet it has the *lowest* Kermany common-class recall of the four: CNV 147/250,
   DME 99/250, DRUSEN 104/250, against MIRAGE 229/200/230, DINOv2 209/152/162, Inception-v3
   170/138/163 [pre-specified output]. Its NORMAL recall (215/250) is also the lowest.
   Confidence high that exposure did not give it a visible advantage with frozen prototypes.
2. **RETFound's low Kermany false-rare rate (1/169) does not need exposure to explain it.** It
   also gives 0/206 on OCTID, which it never pretrained on, and it assigns a rare class to 0 of
   204 OCTID rare images and 3 of 39 OCTDL macular holes [pre-specified output]. It simply almost
   never outputs a rare class for a clinical OCT scan. A contribution from exposure cannot be
   excluded, but it is not needed. Confidence medium-high. The disclosure in section 10 of
   `protocols.md` should stay.
3. **MIRAGE: no data exposure, but a device pattern.** Normal recall 250/250 (Spectralis),
   202/206 (Cirrus), 26/332 (RTVue) lines up with its pretraining devices. Inception-v3 shows the
   same ordering with no OCT pretraining (223, 182, 66), so device coverage is at most part of the
   story. MIRAGE's zero Kermany false-rare count cannot be an exposure artefact. Confidence
   medium.
4. **DINOv2 and web images: no sign of inflation.** If the Yoo web images were in LVD-142M, the
   in-source five-way score would be raised. DINOv2's development score (0.558) is close to
   Inception-v3 (0.534) and below the size-only reference (0.569). Exposure remains unknown; it
   explains nothing we see.
5. **MIRAGE saw CSC, MacTel, macular hole and Stargardt during pretraining; RP is not listed.**
   Its best class is RP (0.73 development, 0.81 hold-out) and its weakest is macular hole (0.54),
   so per-class results do not follow pretraining prevalence [derived by me from the P0 numbers
   and Table 8]. Do not use pretraining content to explain per-class ranks.

Net: no locked result is plausibly explained by pretraining exposure. The external drops are
shift (or support-source) effects, and the only confirmed exposure (RETFound, Kermany) sits
beside RETFound's worst results.

---

## 4. Other interpretations worth Lam's attention

1. **Where MIRAGE's five-way errors go: CSC and macular hole trade places.** Summing the
   per-episode confusion counts, prototype head, K = 1-8 [derived by me from pre-specified
   per-episode `metrics.json`; the diagonal reproduces P0 exactly: 0.705/0.540/0.684/0.591/0.728]:
   development, true MH -> CSC 0.235; true CSC -> MH 0.162; every other cell is <= 0.12. On the
   hold-out the weak class changes: MacTel 0.50 (-> MH 0.21, CSC 0.14, Stargardt 0.14) and MH
   rises to 0.79. With 4-5 hold-out images per class, the hold-out does not confirm or refute the
   MH weakness. Benchmark CSC and MH images differ in size (median 414 x 413 against 327 x 326),
   so this pair is one that size alone would separate and MIRAGE still confuses.

2. **In the one test where size carries no information, MIRAGE's lead mostly disappears.** OCTID
   CSC-versus-MH: size-only scores exactly 0.50 at every K. Mean over K = 1-8 [pre-specified
   output]: DINOv2 0.768, MIRAGE 0.737, Inception-v3 0.711, RETFound 0.512. MIRAGE minus
   Inception-v3 is +0.026 there, against +0.116 on development and +0.197 on the hold-out
   [derived by me]. MIRAGE leads only at K = 16 (0.898; DINOv2 0.863; Inception-v3 0.844). Also
   on the hold-out at K = 16, size-only (0.727) beats DINOv2 (0.680) and Inception-v3 (0.621)
   [pre-specified output], and the hold-out classes barely overlap in width (CSC 413-417, MH
   324-330, MacTel 481-484, Stargardt 374-376) [pre-specified geometry table]. Caveat: the OCTID
   test is two-class and cross-source, so it is not the same task. Confidence medium-high that the
   in-source margin overstates MIRAGE's advantage.

3. **The curves have not flattened by K = 8, and K = 16 is only partly informative.** Development
   gains per doubling [derived from pre-specified `nomination.json`]: MIRAGE +0.120, +0.098,
   +0.065; DINOv2 +0.090, +0.120, +0.094; Inception-v3 +0.091, +0.081, +0.067; RETFound +0.057,
   +0.066, +0.047. Hold-out K = 8 -> 16: MIRAGE 0.867 -> 0.894, DINOv2 0.657 -> 0.680,
   Inception-v3 0.614 -> 0.621, RETFound 0.422 -> 0.476. On OCTID two-way MIRAGE still climbs
   0.837 -> 0.898. So MIRAGE and Inception-v3 look near a ceiling on the 22 hold-out images only.
   At K = 16 Stargardt uses 16 of its 19 images, so the draws overlap heavily and the smaller SD
   at K = 16 is not comparable with lower K.

4. **RETFound gets worse externally as K grows.** Its five-way share of correct answers at K = 16
   is 0.00 (OCTID CSC), 0.01 (OCTID MH), 0.03 (OCTDL MH), down from 0.12-0.18 at K = 1; the mass
   moves to Stargardt and MacTel [pre-specified output]. In the nine-way audit it gives a rare
   label to 21 of 781 external images. Its development score (0.406) is below size-only (0.569).
   This looks like a frozen-feature recipe that separates "benchmark image" from "clinical scan"
   and little else; it is not evidence about RETFound when fine-tuned. Confidence medium.

5. **A low false-rare rate tracks a low tendency to say "rare" at all.** Per source, share of
   true rare images given any rare label against share of normals given any rare label [derived
   by me: sums of the locked nine-way counts in 2a and 2b; not in the P list, needs a post hoc
   label if used]:

   | | OCTID rare (204) | OCTID normal (206) | OCTDL MH (39) | OCTDL normal (332) |
   |---|---|---|---|---|
   | MIRAGE | 52 (25%) | 4 (2%) | 3 (8%) | 30 (9%) |
   | Inception-v3 | 90 (44%) | 15 (7%) | 1 (3%) | 11 (3%) |
   | DINOv2 | 154 (75%) | 17 (8%) | 39 (100%) | 141 (42%) |
   | RETFound | 0 (0%) | 0 (0%) | 3 (8%) | 18 (5%) |

   MIRAGE's low false-rare rate comes with missing three quarters of OCTID's rare images in the
   nine-way setting; RETFound's 0% is a model that never says rare. The protocol already warns
   "Low FRR alone does not imply useful diagnosis"; these counts make that concrete and argue for
   showing rare recall next to the false-rare rate. Placement is Lam's decision (rule 2).

---

## 5. Further analysis that would most change the interpretation (ranked)

| Rank | Analysis | Status | Allowed? | What it would change |
|---|---|---|---|---|
| 1 | **P3, clinic-to-clinic macular hole and normal** | Planned, "if time", first in the drop order | Yes, after rule 1; outside the protocol by rule 4 | The only planned test that separates "encoder cannot read RTVue scans" from "Kermany/web prototypes do not transfer". If MIRAGE separates OCTDL MH from OCTDL normal with OCTID examples, the OCTDL failure is a support-source effect. Lam may want to reconsider dropping it first |
| 2 | **OCTDL normals by aspect ratio or width** (laptop): normal recall per model in width or aspect bands, from locked predictions and image sizes | **New** | Only as a labelled post hoc analysis: rule written into section 8 before computing, after rule 1, started by 21 Nov. Close to P4 in spirit but a different split, so it needs its own line | Tests the squash-versus-crop explanation in 2c without a GPU. OCTDL widths range 614-1270, so there is spread to use |
| 3 | **Input check on OCTDL normals** (GPU): aspect-preserving input, and MIRAGE without min-max | **New** | Changes the A2 A-11 input recipe, so it is outside the protocol; the plan's GPU check covers only the 389 rare images and only if the mentor asks. Needs Lam's go-ahead and a dated change line; may not replace any locked number | The direct test of the min-max and geometry explanations. Higher cost; do only if rank 2 points that way |
| 4 | **P5, per-scan destinations and top-two gaps** | Planned, "always" | Yes, after rule 1 | Destinations are already in the locked output (2a). The gaps show whether OCTDL normals are near misses or confident errors, which decides whether an abstain rule is worth describing |
| 5 | **P4, size-right against size-wrong**, and **P1, leave-one-out** | Planned | Yes, after rule 1 | Item 4.2 suggests the in-source margin leans on size; P4 sizes that, P1 shows how much of the hold-out margin rests on single images |
| 6 | **Rare recall beside the false-rare rate, per source** (table in 4.5) | **New**, sums of locked counts only | Yes as a labelled post hoc table; no new model output | Changes how the false-rare result reads; cheap |
| 7 | **P2, October heads on OCTID and OCTDL** | Planned, "if time" | Yes, after rule 1, outside the protocol by rule 4 | Shows whether a head trained on all rare images also fails on OCTDL normals; affects the Yoo comparison more than the shift question |
| 8 | **W1 remainder**: check that our Kermany v2 files are within the 109,309-image release | Planned (W1) | Yes, no scores involved | Low. Only firms up one sentence of disclosure |

Not recommended: fitting a NORMAL prototype from external normals inside the nine-way audit. A1
makes external images query-only, and P3 already covers the question in a labelled form.

## Limits of this report

- All interpretation rests on single deterministic nine-way heads and on 22, 39, 102 and 332-image
  groups; no patient-level uncertainty beyond the locked intervals.
- Clinical readings (MH resembling DME, CSC resembling CNV) are mine and unverified.
- I read the papers' main text, not their supplements. RETFound's Supplementary Table 1 and
  MIRAGE's supplement were not opened.
- The DINOv2 web-image question cannot be settled from any public source.
