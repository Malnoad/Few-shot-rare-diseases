# Lit scan: GAN vs basic augmentation for few-shot medical classification, and who used Yoo et al. 2021

Scoping review, single reviewer (agent), searched 2026-10-23. It answers two questions:

- **Q1.** Since 2021, has GAN or CycleGAN augmentation been shown to beat ordinary (basic)
  augmentation for few-shot or rare-class medical image classification? OCT first, then
  other modalities.
- **Q2.** Has anyone reproduced, re-used or critically evaluated Yoo, Choi & Kim 2021
  (MBEC 59:401-415, doi:10.1007/s11517-021-02321-1)?

Every source cited here was opened: the full text where marked **FT**, otherwise the
abstract only (**Abs**). Anything I did not open is marked *unverified*.

## Conclusion for this project (5 lines)

1. I found no study since 2021 showing that GAN augmentation beats basic augmentation for few-shot **OCT** classification across several training seeds. Yoo 2021 compared against basic augmentation on one independent test split. The only re-implementation (Tai 2025) has no basic-only arm.
2. Outside OCT, three studies compare GANs head-to-head with basic augmentation (CT hemorrhage, two histopathology datasets). All three favour the GAN on the minority class. None reports a paired comparison across seeds: they use one training run with bootstrap CIs, 5-fold CV, or 5 seeds pooled into one confusion matrix.
3. The large 2026 systematic review (160 studies) reports modest median gains from adding synthetic images to real data (+4.3% accuracy). It measures those gains against real data alone, not against basic augmentation. Statistical reporting in the field was rated poor.
4. So our finding (over seeds 1-5, gan ties basic and both beat none) does not contradict the literature, because the literature never tested this claim properly. The seed sweep (C1) would be the first multi-seed test of gan vs basic on this benchmark that I could find.
5. For L7: in Tai 2025, the no-augmentation baseline recovers 0% of 4 of the 5 rare classes, as in Yoo, while our `none` reaches 0.75-0.80. That points to a difference in our `none` setup (frozen features + head, class balance), not in the data.

## Search log

All searches ran on 2026-10-23 through public APIs (no keys, no paid services). The raw
JSON/XML responses were saved under `/tmp/litscan/` (not committed). "Count" is the
total the API reported. All API records were retrieved unless noted.

### Q2: citation searches (forward citations of Yoo 2021)

| # | Database | Query / endpoint | Count | Note |
|---|---|---|---|---|
| C1 | OpenAlex | `works?filter=cites:W3122440835` (2 pages × 100) | 132 | W3122440835 = Yoo 2021 |
| C2 | Semantic Scholar | `paper/DOI:10.1007/s11517-021-02321-1/citations?limit=1000` | 127 | |
| C3 | Europe PMC | `MED/33492598/citations` | 66 | |
| C4 | Crossref | `works/10.1007/s11517-021-02321-1` → `is-referenced-by-count` | 131 | count only, no list |
| C5 | OpenCitations | `index/api/v2/citation-count/doi:...` | n/a | returned "Moved Permanently"; not retried |
| C6 | Europe PMC | `CITES:33492598_MED` | 66 | 40 had a PMCID; full text fetched for all 40, 37 had a body |
| C7 | Europe PMC full text | `"btv6yrdbmv"` (rare-disease dataset DOI) | 2 | Yoo 2021 itself; Rozhyna 2024 |
| C8 | Europe PMC full text | `"10.17632/btv6yrdbmv"` | 1 | Rozhyna 2024 |
| C9 | OpenAlex full text | `search=btv6yrdbmv` | 2 | Rozhyna 2024; OmniMedSeg-OCT 2026 (a dataset record whose listed datasets do not include it) |
| C10 | OpenAlex | `title_and_abstract.search:("rare retinal" OR "rare retinal diseases") AND (OCT OR "optical coherence tomography") AND ("few-shot" OR "few shot" OR GAN OR CycleGAN)`, from 2021 | 5 | catches re-use without citation |

After deduplicating C1-C3 by normalized title: **150 citing records**. I screened all titles.
I screened abstracts for the OCT/GAN/few-shot candidates. For the 37 open-access full texts,
I extracted every sentence that cites Yoo 2021 (a script maps the reference ID to its
in-text citations) and read each one.

### Q1: topic searches (2021-01-01 to 2026-10-23)

| # | Database | Query | Count |
|---|---|---|---|
| T1 | PubMed | `("generative adversarial"[tiab] OR GAN[tiab] OR GANs[tiab] OR CycleGAN[tiab] OR "cycle-consistent"[tiab]) AND (augment*[tiab] OR synthetic[tiab] OR synthesi*[tiab]) AND ("optical coherence tomography"[tiab] OR OCT[tiab]) AND (classif*[tiab] OR diagnos*[tiab] OR detect*[tiab]) AND 2021:2026[dp]` | 37 |
| T2 | PubMed | `("generative adversarial"[tiab] OR GAN[tiab] OR GANs[tiab] OR CycleGAN[tiab]) AND augment*[tiab] AND ("few-shot"[tiab] OR "few shot"[tiab] OR "rare disease*"[tiab] OR "rare class*"[tiab] OR "minority class*"[tiab] OR "class imbalance"[tiab] OR imbalanced[tiab] OR "small dataset*"[tiab] OR "limited data"[tiab]) AND classif*[tiab] AND (traditional[tiab] OR conventional[tiab] OR basic[tiab] OR classical[tiab] OR standard[tiab] OR geometric[tiab]) AND 2021:2026[dp]` | 82 |
| T3 | PubMed | `("generative adversarial"[tiab] OR GAN[tiab] OR CycleGAN[tiab] OR "generative model*"[tiab]) AND ("data augmentation"[tiab]) AND (medical[tiab] OR clinical[tiab] OR radiolog*[tiab] OR ophthalm*[tiab]) AND (systematic review[pt] OR meta-analysis[pt] OR "systematic review"[tiab] OR "meta-analysis"[tiab] OR "scoping review"[tiab]) AND 2021:2026[dp]` | 16 |
| T4 | Europe PMC | `ABSTRACT:("generative adversarial" OR CycleGAN OR GAN OR GANs) AND ABSTRACT:(augmentation OR augmented OR synthetic) AND ABSTRACT:("optical coherence tomography" OR OCT) AND ABSTRACT:(rare OR "few-shot" OR "few shot" OR "small dataset" OR "limited data" OR imbalance OR imbalanced) AND PUB_YEAR:[2021 TO 2026]` | 5 |
| T5 | Europe PMC | `ABSTRACT:("generative adversarial" OR CycleGAN OR GAN OR GANs) AND ABSTRACT:(augmentation) AND ABSTRACT:("few-shot" OR "few shot" OR "rare disease" OR "rare diseases" OR "rare class" OR "minority class") AND ABSTRACT:(classification) AND ABSTRACT:(traditional OR conventional OR basic OR classical OR standard OR geometric) AND ABSTRACT:(image OR images OR imaging) AND PUB_YEAR:[2021 TO 2026]` | 7 |
| T6 | arXiv | `(abs:GAN OR abs:CycleGAN OR abs:"generative adversarial") AND abs:augmentation AND (abs:OCT OR abs:"optical coherence tomography") AND abs:classification AND submittedDate:[202101010000 TO 202610232359]` | 1 |
| T7 | arXiv | `abs:OCT AND abs:augmentation AND abs:GAN` (no date filter; filtered locally) | 4 (2 from 2021+) |
| T8 | arXiv | `abs:OCT AND abs:CycleGAN` (filtered locally) | 6 (5 from 2021+) |
| T9 | arXiv | `abs:GAN AND abs:augmentation AND abs:rare AND abs:medical` (filtered locally) | 5 (3 from 2021+) |
| T10 | arXiv | `abs:GAN AND abs:augmentation AND abs:"few-shot" AND abs:medical AND submittedDate:[...]` | 1 |
| T11 | OpenAlex | `title_and_abstract.search:(GAN OR CycleGAN OR "generative adversarial") AND augmentation AND ("optical coherence tomography" OR OCT) AND (rare OR "few-shot" OR "few shot" OR "small dataset" OR imbalanced)`, from 2021 | 13 |
| T12 | OpenAlex | `title_and_abstract.search:(GAN OR CycleGAN OR "generative adversarial") AND augmentation AND ("few-shot" OR "rare disease" OR "rare diseases" OR "minority class") AND classification AND (traditional OR conventional OR basic OR classical) AND (medical OR clinical)`, from 2021 | 40 |

arXiv's `abs:` search matches exact tokens without stemming, so its counts are small. T6
missed Tai 2025, whose abstract says "classifying", not "classification"; T7 found it.
Overlap between searches was removed by eye, not counted, so I report no deduplicated total.
I screened titles for all records and abstracts for plausible ones. I fetched full text for
the open-access candidates.

## Inclusion rule

**Q1 primary studies:** all of the following.
1. Published or posted 2021-01-01 to 2026-10-23.
2. Medical **image classification**, meaning a diagnostic label per image or patient.
   Segmentation-only, image-quality-only and non-image studies (ECG, tabular, text) are excluded.
3. GAN-generated images (any GAN, including CycleGAN and StyleGAN) are added to real
   training data.
4. A rare or few-shot setting: the target class is a rare disease, is described as
   few- or low-shot, is ≤1% of the training data, or has ≤1,000 real training images.
5. A quantitative comparison against at least one non-GAN arm (no augmentation or basic
   augmentation).
6. I could open at least the abstract.

Studies that use only diffusion models are noted but not included, because the question
is about GANs. Q1 also includes systematic or scoping reviews of generative augmentation in
medical imaging, kept in a separate table.

**Q2:** any record that cites Yoo 2021 or uses its rare-disease dataset (Mendeley
btv6yrdbmv) **and** re-implements the method, re-uses the data or code, or makes a specific
critical or quantitative statement about it. Background-only citations were screened out.

## Q1 results: primary studies

"Basic arm?" asks whether there is an arm with basic augmentation only (flips, rotations,
etc.), so GAN vs basic is a direct comparison. "Seeds / CV" records how training randomness
or data splits were repeated.

| Study | Src | Modality / task | Rare/few-shot target | GAN | Basic arm? | Seeds / CV | Reported result |
|---|---|---|---|---|---|---|---|
| Yoo, Choi, Kim 2021, MBEC 59:401, doi:10.1007/s11517-021-02321-1 | FT (PMC7829497) | OCT, 9 classes | 5 rare diseases, 23-32 images each (our `data/manifest.csv`: 19-27 train + 4-5 test) | CycleGAN per disease | **Yes**: none / basic / basic + 3,000 GAN per class | 5-fold CV on all data (93.9%) + one independent test split; no seeds reported | Independent test 92.1 (gan) vs 91.3 (basic) vs 88.4 (none) %. No significance test for gan vs basic in the text I extracted. The GAN arm also adds 3,000 images per class; I did not verify whether the basic arm was size-matched |
| Tai et al. 2025, arXiv:2505.20149 (IEEE GCCE 2025, doi:10.1109/gcce65946.2025.11275047) | FT | OCT, 9 classes, Yoo's data | same 5 rare diseases | CycleGAN, U-GAT-IT | **No**: arms are no-aug vs (2,000 basic + 3,000 GAN) | 5-fold CV, mean ± SD; per-class TPR on an "independent test" | CycleGAN: balanced accuracy 0.550 → 0.856, overall accuracy 97.18 → 93.76%. U-GAT-IT beats CycleGAN. Basic alone never tested |
| Yoo et al. 2021, CMPB 205:106086, doi:10.1016/j.cmpb.2021.106086 | Abs | Ocular surface photos, 4 classes | 75-136 images per class | 2 GANs | Not stated in abstract | One 70/10/20 split (abstract) | "Significant improvement" with GAN images; best 87.5% (4-class) |
| Zhao J et al. 2026, Retina 46:1759 doi:10.1097/IAE.0000000000004919 | Abs | Fundus, DR grading (IDRiD) | proliferative DR, limited data | StyleGAN3 (+ Medfusion diffusion) | Not stated in abstract (baseline = "real data") | **5 random seeds**, 3 architectures | QWK 0.635 → 0.719 (StyleGAN3), 0.770 (diffusion). Diffusion beat the GAN |
| Jeong, Patel, Banerjee 2022, J Med Imaging 9:035504, doi:10.1117/1.JMI.9.3.035504 | FT (PMC HTML) | Head CT, hemorrhage, 6 classes | epidural = 0.41% of data | cDCGAN (+ others) | **Yes**: traditional, traditional + balancing, cDCGAN | One training run per arm; bootstrap CIs | Binary: "did not differ much". Multiclass epidural recall 0.047 → 0.172 and F1 0.082 → 0.276 vs traditional |
| Giuste et al. 2023, J Biomed Inform 139:104303, doi:10.1016/j.jbi.2023.104303 | FT (PMC10031799) | Histopathology tiles, rare pediatric heart-transplant rejection | 26 expert tiles | PGAN + IGAN (+ diffusion) | **Yes**: conventional (rotation/flip) vs conventional + GAN | **5 seeds**, pooled into one confusion matrix; no test | GAN and diffusion "boosted" VGG-19 and ResNet-152 (2 of 4 classifiers) |
| Ruiz-Casado et al. 2024, Sensors 24:3777, doi:10.3390/s24123777 | FT (PMC11207853) | Breast histopathology, binary | 588 benign vs 1,233 malignant | ProGAN, SNGAN, BEGAN, ReGAN | **Yes**: original, traditional, downsampling, GAN-balanced | **5-fold CV**, mean ± SD | GAN-balanced "consistently" highest mean accuracy across 3 CNNs; no significance test seen |
| Hassini et al. 2025, Sci Rep, doi:10.1038/s41598-025-26429-8 | FT (PMC12660895) | Fourier ptychographic microscopy, 5 WBC classes | basophils, 54 real images | physics-inspired GAN | **No**: with vs without synthetic basophils | **50 initializations** per arm | Basophil precision 94.00% → 99.37% |
| Deo, Mehta, Jadhav 2024, arXiv:2409.19945 | FT | Dermoscopy (HAM10000), 7 classes | dermatofibroma (tail class) | One-Shot GAN vs WGAN | **No**: no-aug and WGAN baselines | None reported | Tail-class accuracy up vs no-aug and WGAN |

**Near misses (screened, not included):** Zhao M et al. 2022, Med Phys 49:7357 (OCT; mixed
real + GAN beat real by 5.56% accuracy, but the per-class training size is not given in the
abstract; Abs). Two corneal-topography and skin-lesion studies (doi:10.1088/2057-1976/adabea;
doi:10.1109/EMBC58623.2025.11254065; Abs) say they compare against "traditional" methods
but give no sizes. Kermany-based OCT studies that balance DRUSEN/DME with GANs (e.g.
doi:10.1007/s00417-022-05919-9; Abs) fail rule 4, because the minority class has ~8,900
images. The EyeDiff text-to-image study (npj Digit Med 2026, doi:10.1038/s41746-026-02560-2;
Abs) is diffusion-based.

### Q1 results: evidence syntheses

| Review | Src | Scope | What it says about GAN vs basic |
|---|---|---|---|
| Folguera-Profitós et al. 2026, JMIR AI, doi:10.2196/81472 | FT (PMC13671675) | Systematic review, 160 studies 2020-2025, synthetic medical images; 74% GAN | Adding synthetic images to real data: median +4.3% accuracy, +5.3% recall, +4.0% AUC. These gains are measured against **real data alone**, and the authors call them "unweighted descriptive summaries rather than pooled causal effects". Training data was stratified (original / traditionally augmented / synthetic), but the main text reports no synthetic-vs-traditional estimate (supplement not opened). Only 38% of studies had low risk of bias; random seeds were "rarely made available" |
| Machado et al. 2025, Appl Sci 15:3084, doi:10.3390/app15063084 | Abs (MDPI blocked download) | Systematic review, generative augmentation for inherited retinal diseases; 32 studies, only 2 on IRD | "Generative models effectively augment small datasets". The abstract gives no head-to-head with basic augmentation |
| Finetti et al. 2025, Med Sci 13:260, doi:10.3390/medsci13040260 | FT (PMC12641889) | Scoping review, 118 rare-disease studies 2018-2025 | Classical augmentation is the most frequent approach. It cites Yoo 2021 as GAN images that "doubled the size of training sets and enhanced the performance" |
| Kugelman et al. 2022, J Optom, doi:10.1016/j.optom.2022.09.004 | FT (PMC9732473) | Review of GANs in OCT | Lists Yoo 2021 under GAN-based augmentation for classification; no re-analysis |

## Q2 results: re-use, reproduction and critical evaluation of Yoo 2021

No independent reproduction has been published. One group re-implemented the pipeline and
extended it. Others re-used the dataset or made one critical remark. Several papers copy
Yoo's headline number into comparison tables incorrectly.

| Citing work | Src | Kind | What it does with Yoo 2021 | Basic arm? | Seeds / CV |
|---|---|---|---|---|---|
| Tai et al. 2025 (Yuan Ze Univ.; arXiv:2505.20149 / GCCE 2025) | FT | **Re-implementation + extension** | "This study builds upon the framework proposed by Yoo et al.": same data, same basic-aug parameters, 2,000 NORMAL as domain A, one GAN per disease. Their CycleGAN arm: 93.76 ± 1.44% (5-fold CV), close to Yoo's 93.9 ± 4.5%. Their no-aug baseline gets 0% TPR on CSC, MacTel, RP and Stargardt (MH 100%). They cite Yoo's ~20% clinician rejection rate as a weakness | No | 5-fold CV |
| Wu et al. 2025, ICCE-Berlin, doi:10.1109/icce-berlin67488.2025.11277693 | Abs | Same group, follow-up | "Building on our previous FSL training strategy"; lightweight CNN, 97.87%. Whether it uses Yoo's data: *unverified* (closed access) | unknown | unknown |
| Tseng et al. 2026, ICCE, doi:10.1109/icce67443.2026.11449635 | Abs | Same group, follow-up | 8-class imbalanced OCT with rare diseases; "modified from the baseline method in [1]" plus basic augmentation, 78.35%. Identity of [1]: *unverified* | unknown | unknown |
| Pasupuleti & Girish 2021, arXiv:2112.06031 (CCIS 2023) | FT | **Dataset re-use** | Uses Yoo's rare-disease data (reports 30 CSC, 30 MH, 24 MacTel, 19 RP, 16 Stargardt; our manifest has 32 / 31 / 29 / 31 / 23, possibly a different dataset version, *unverified*) to train a multi-domain image-to-image GAN against CycleGAN and MUNIT. Image quality only (FID/LPIPS); no classifier | n/a | n/a |
| Rozhyna et al. 2024, Diagnostics, doi:10.3390/diagnostics14151668 | FT | Dataset catalogue | Lists the rare-disease OCT dataset and Yoo's use of it | n/a | n/a |
| Verma et al. 2024, TVST, doi:10.1167/tvst.13.11.2 | FT | **Critical remark** | Notes that "many of the generated OCT images contained artifacts", which makes them "unsuitable for a segmentation task" | n/a | n/a |
| Li et al. 2023, J Imaging, doi:10.3390/jimaging9100203 | FT | Misquote | Comparison table lists "GAN [Yoo] 0.939" under the OCT-C8 dataset. Yoo did not use OCT-C8; 93.9% is Yoo's 9-class 5-fold CV | n/a | n/a |
| Subramanian et al. 2022, Comput Intell Neurosci, doi:10.1155/2022/8014979 | FT | Misquote | Lists Yoo as "DME, CNV, MH and Drusen 93.9" | n/a | n/a |
| Kaya et al. 2025, World J Psychiatry, doi:10.5498/wjp.v15.i9.108359 | FT | Misquote | Lists Yoo as "OCT2017, accuracy 93.90" | n/a | n/a |
| Devi & Grace 2026, J Appl Clin Med Phys, doi:10.1002/acm2.70573 | FT | Misquote | Lists Yoo as "GAN, 4 classes, 0.912". Yoo reports 9-class results; I found no 4-class experiment in the text I extracted | n/a | n/a |
| You, Kim, Ryu, Yoo 2022, Eye Vis, doi:10.1186/s40662-022-00277-3 | FT | Review (Yoo is a co-author) | Summarises Yoo 2021 as GAN augmentation that "increased the accuracy of diagnosing rare retinal diseases" | n/a | n/a |

Of the 37 open-access full texts, the 28 not listed above cite Yoo only as background
(few-shot learning, GANs in ophthalmology, rare diseases). For 3 of them, no in-text
citation sentence could be extracted. None questions the gan-vs-basic result, and none
reports re-running it.

## Limitations

- One reviewer, no second screener. Screening counts are per search, not a reconciled
  PRISMA flow.
- No access to Scopus, Web of Science, IEEE Xplore or Google Scholar. IEEE conference
  papers were found only through OpenAlex and Semantic Scholar. Closed-access papers were
  read as abstracts only, so their "Basic arm?" and "Seeds / CV" values are often unknown.
- The PubMed T2 query requires comparison words ("traditional", "basic"...) in the
  abstract. That favours studies with a basic-augmentation arm, but misses studies that do
  not mention one in the abstract.
- Semantic Scholar rate-limited (HTTP 429) one lookup (IEEE Access 2026,
  doi:10.1109/ACCESS.2026.3680450, "Hybrid Few-Shot Learning With Attention..."). It was not
  opened, so whether it re-uses Yoo's data is *unverified*. Mai et al. 2021 (MICCAI,
  doi:10.1007/978-3-030-87237-3_10) and DMRD-Net 2026 (doi:10.1016/j.ins.2026.123578) also
  surfaced in C10 and were not opened beyond metadata or abstract.
