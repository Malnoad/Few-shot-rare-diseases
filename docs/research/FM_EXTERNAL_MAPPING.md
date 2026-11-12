# External rare-class mapping: data balance, MH/CSC evidence and working definitions

Prepared 12 November 2026 for Lam, pending the mentor's feedback. The label decision is now
amendment A1 (`FM_AMENDMENTS.md`, 12 Nov); this note is its evidence. Otherwise nothing here changes
protocol v1.0; options B (documented-label amendment) and C (clinician check) are deferred
by Lam. No encoder was run on any image below and no score exists.

## 1. Data balance after adding OCTID and OCTDL

Benchmark counts are from `data/manifest.csv`. External counts are from the OCTID Borealis
file lists (11 Nov) and `OCTDL_labels.csv` (Zenodo 10839556, MD5 `2fced789…c289`, matching
the earlier record).

| Class | Benchmark (dev + legacy test) | OCTID | OCTDL | Sources | Patients known? |
|---|---|---|---|---|---|
| CSC | 27 + 5 = 32 | CSR 102 | none | 2 | Benchmark no; OCTID no |
| MH | 26 + 5 = 31 | MH 102 | VID/MH 39 (28 patients, 28 eyes) | 3 | OCTDL only |
| MacTel | 25 + 4 = 29 | none | none | 1 | no |
| Stargardt | 19 + 4 = 23 | none | none | 1 | no |
| RP | 27 + 4 = 31 | none | none | 1 | no |
| NORMAL | 23,536 train, 2,779 val, 250 test (Kermany) | 206 | 332 (110 patients) | 3 | Kermany and OCTDL |

Not mapped: OCTID AMD (55) and DR (107); OCTDL AMD, DME, ERM, RAO, RVO, and the VID
subcategories LH (26 images, 14 patients) and VMT (11 images, 9 patients).

Consequences:
- External data adds rare images for only two of the five rare classes. MacTel, Stargardt
  and RP stay single-source.
- If external images were added to fitting, MH would grow from 31 to 172 and CSC from 32 to
  134, while the other three classes stay at 23-31. The five-way task would become unbalanced
  in both size and source.
- OCTDL's `condition` column says "MH" for all 76 VID images, including lamellar holes and
  traction. Only `subcategory` separates true MH (39). The protocol's warning ("condition=MH
  alone is unsafe") is confirmed.

## 2. What the sources say about MH and CSC

| Source | MH | CSC / CSR | Who labelled |
|---|---|---|---|
| Benchmark (Mendeley btv6yrdbmv, released with Yoo et al. 2021) | Definition unverified | Definition unverified | Unverified |
| OCTID (Gholami et al., arXiv:1812.07056, PDF shipped with each Borealis dataset) | "MH retinal defects can vary from small hypoflective breaks to full-thickness intra-retinal spaces" (broad) | Listed as "Central Serous Retinopathy", described under the heading "Central Serous Chorioretinopathy (CSC)": "focal detachments of the retina and the RPE layer ... hypo reflective spaces at the sub-retinal and sub-RPE levels" | "image class labeling was done based on the diagnosis of retinal clinical experts at SN hospital"; one Cirrus HD-OCT, Sankara Nethralaya, Chennai; fovea-centred B-scan picked by an optometrist; resized to 500x750 |
| OCTDL (Kulyabin et al., Sci Data 2024, PMC11009408) | "a complete defect in the inner layers of the retina that extends to the RPE"; lamellar tear treated as a separate variant (`subcategory` LH) | none | Seven trained medical students, then two clinical specialists independently with consensus; head of clinic confirmed every diagnosis; Optovue RTVue XR |
| IVTS classification (Duker et al., Ophthalmology 2013, doi:10.1016/j.ophtha.2013.07.042) | "Full-thickness macular hole (FTMH) is defined as a foveal lesion with interruption of all retinal layers from the internal limiting membrane to the retinal pigment epithelium" | — | Expert panel consensus |
| MeSH D056833 | — | "Central Serous Retinopathy" is an entry term of "Central Serous Chorioretinopathy" | NLM |

Earlier papers that used OCTID treat its CSR class as CSC: EyeDiff (doi:10.1038/s41746-026-02560-2)
lists "CSCR"; a 2026 TVST external validation (doi:10.1167/tvst.15.1.6) lists "AMD, CSC, FTMH,
and Healthy"; a 2025 BMC Ophthalmology review (doi:10.1186/s12886-025-04372-6) calls it CSCR.
Note the TVST paper's "FTMH" is that paper's reading; OCTID's own text is broader.

## 3. Image inspection (non-clinical)

The agent viewed 6 benchmark MH, 6 OCTID MH, 6 OCTDL MH (6 patients), 3 OCTDL LH, 4 benchmark
CSC and 4 OCTID CSR images. These are lay observations on a tiny sample, not labels. The sheets
are in the agent's scratchpad (`ext/mh_sheet.png`, `ext/csc_sheet.png`) and are not committed,
because the repository holds no third-party images.

- MH: the benchmark and OCTDL MH samples all looked like full-thickness holes. OCTID MH looked
  mostly full-thickness, but MH1 looked like a smaller defect and MH4 showed traction and
  detachment, consistent with OCTID's broader definition. OCTDL LH looked clearly different
  (shallow, no full-thickness gap).
- CSC: the four benchmark CSC samples showed large dome-shaped subretinal fluid. OCTID CSR1
  showed a dome under the RPE; CSR2-4 looked flatter, with subtle changes. The OCTID class may
  include milder or chronic cases. This is a possible spectrum mismatch, not a label error.
- Source cues are visible to the eye: OCTID images carry a scan-orientation icon in the
  bottom-left corner; benchmark images are crops with black padding, and their size differs by
  class (median width CSC 414, MH 327, MacTel 482, Stargardt 385, RP 393 px; all RGBA);
  OCTDL images are wide (about 740-1100 x 265-410 px).

## 4. Working definitions until the mentor replies

These are proposals for Lam to adopt or change; they decide eligibility, never single images.

- **MH (working):** full-thickness macular hole as defined by the IVTS: "interruption of all
  retinal layers from the internal limiting membrane to the retinal pigment epithelium".
  - OCTDL: include `disease=VID` and `subcategory=MH` only (its own definition matches);
    exclude LH and VMT.
  - OCTID MH: eligible as "MH, subtype not verified", reported separately from OCTDL MH,
    because OCTID's definition also covers partial defects.
- **CSC (working):** serous detachment of the neurosensory retina and/or RPE at the macula
  (subretinal or sub-RPE fluid), as OCTID describes its CSC class. OCTID CSR is eligible as CSC
  on the dataset authors' own heading and the MeSH synonym. The possible acute/chronic spectrum
  difference is reported as a limitation.
- **Benchmark:** Yoo et al.'s definitions are unverified. Ask the mentor to confirm that the
  benchmark MH images are full-thickness and the CSC images serous detachments.
- **No per-image relabelling by the team.** If the mentor or a clinician rejects a mapping,
  the whole class drops back to the protocol default (NORMAL FRR only for that source).

## 5. Grouping and splitting: what it fixes and what it does not

Grouping (duplicates and verified patients) with grouped folds fixes **leakage**: the same eye
or patient in fitting and evaluation. It is already in the protocol and stays essential, for
example OCTDL MH has 39 images from 28 patients (up to 3 per patient).

It cannot fix **confounding**, where a cue is present in every image of a class in every split.
Two such cues exist:
1. Within the benchmark, crop size differs by rare class (section 3), and each class comes from
   one source. Any split keeps that cue on both sides.
2. Across sources, each external cohort has its own look (OCTID icon and size, OCTDL width).
   If external images are mixed into fitting for MH and CSC only, a new shortcut appears:
   "looks like OCTID, so CSC or MH", because MacTel, Stargardt and RP never come from OCTID.

What does help, all before any score:
- **A within-source test where source cannot separate the classes.** OCTID contains both CSC
  (102) and MH (102), at one size, from one device. A two-way CSC-versus-MH test on OCTID
  queries, with supports from the benchmark, measures disease transfer while the source is held
  constant. This is the cleanest external rare test available.
- **Leave-one-source-out**, for the classes with several sources: fit on two sources, test on
  the third (MH: benchmark, OCTID, OCTDL).
- **A source probe as a diagnostic:** how well frozen features predict the source alone. The
  old C0 audit already found near-perfect rare-versus-common separation (AUC about 0.995).
- Geometry harmonization (cropping the OCTID icon, fixed aspect) reduces the cues, but the C0
  audit found that format and framing normalization did not remove the separation (0.995 to
  0.991).

Any of these is a dated amendment to protocol v1.0, allowed only before the lock.

## 6. A clear baseline from previous studies

| Baseline | What it is | Why | Limits |
|---|---|---|---|
| Historical method (in the protocol) | Yoo et al.'s frozen ImageNet Inception-v3, same heads and episodes | Connects to the reproduced study (Table I) | Not K-shot; same images reused |
| Standard few-shot head (in the protocol) | L2-normalized nearest centroid, as in SimpleShot (Wang et al. 2019) | The usual frozen-feature few-shot baseline; our primary head | Already the method, not an outside number |
| Published frozen-encoder benchmark | MIRAGE paper (Morano et al. 2025, PMC12462498): RETFound, DINOv2 and MIRAGE with linear probing, frozen encoder plus one linear layer, five seeds, on OCTID (Control 206, MH 102, AMD 55, CSR 102, DR 107) and OCTDL | The same three encoder families with a frozen-feature protocol on the same external datasets; a published anchor for comparison | Their models are ViT-Large (ours: DINOv2 and MIRAGE Base); their splits and code availability are unverified; full-data, not K-shot. Reproducing it on OCTID would expose OCTID outcomes before our locked external reporting, so it should use a dataset outside our endpoints or come after reporting |

MIRAGE reports its pretraining used an in-house dataset; whether OCTID or OCTDL images were in
RETFound's or MIRAGE's pretraining remains unverified.

## 7. Open for Lam

1. Adopt, change or reject the working definitions in section 4.
2. Whether to add the OCTID CSC-versus-MH test and leave-one-source-out (section 5) as a dated
   amendment before the lock.
3. Which baseline row(s) in section 6 to name in the protocol and paper.
4. Options B and C (deferred).
