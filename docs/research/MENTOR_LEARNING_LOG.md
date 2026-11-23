# Learning log: what each mentor feedback taught us

Lam's mentor feedback shapes how this project thinks. After each feedback we write one entry:
what was said, what we changed, and the lesson in two lenses — **computer science**
(experimental design, shortcuts, controls, failure analysis, leakage, reproducibility) and
**mathematics and protocol** (what exactly is estimated, decision rules fixed in advance, paired
designs, uncertainty, sensitivity analyses). The goal is a slow, steady research intuition, not
more ideas. Decisions and numbers live in `protocols.md`; this file keeps only the lessons.

## Feedback 1 (17–18 Nov 2026): what "rare" means; transparency over comparability

**Said.** "Rare" should be judged by how few labelled images exist, not by disease prevalence.
Methods are simple. Later: our protocol and results are acceptable because they are transparent;
on comparing with Yoo et al., "even the reviewer cannot identify clearly what it is because no one
can compare" (their test split was never published).

**Changed.** Rare defined as B1 (at most 5% of a dataset's labelled images; few-shot pool at most
30; tiers A–D). External OCTID CSC/MH re-labelled by tier, not called "rare". Hold-out kept, with
honest wording ("a hold-out matching their class counts", never "their test set"), the Mendeley
v1/v2 provenance recorded, and the novelty claim narrowed after finding Sun et al. 2024.

**Computer-science lesson.** Define the category before measuring it: a benchmark is only
meaningful if its class definition is operational and checkable on every dataset.

**Mathematics/protocol lesson.** When a comparison cannot be made fairly, do not force it.
State what is not identifiable and report what is: a transparent protocol with fixed rules is
worth more than a number placed next to someone else's number.

**Question to ask next time.** "Is this term defined so that someone else could apply it to a new
dataset and get the same answer?"

## Feedback 2 (20 Nov 2026): reliability before ranking

**Said** (verbatim, as Lam relayed it on 20 Nov): "At this stage, I have a positive assessment of
your research direction, particularly the connection between the two phases: from validating the
effectiveness of GAN augmentation to evaluating the few-shot capabilities of foundation models. The
findings regarding source confounding and performance degradation on external OCT datasets are
results with real scientific value. However, if you are aiming for a research paper, the focus
should be on the reliability and generalizability of the models, rather than merely proving that
MIRAGE outperforms other backbones. There are three things that need to be done: Control for the
effects of image size and framing, then check whether MIRAGE still maintains its advantage. This
is the most critical experiment. Perform an error analysis on OCTID and OCTDL, specifically looking
at cases where normal images are misclassified as other diseases. Test the sensitivity of MIRAGE
and RETFound to preprocessing to ensure that the conclusions do not depend on the feature
extraction method. You do not need to add more models or continue with GAN experiments at this
time. If the main results still hold after these controls, this is a direction with strong
potential to be developed into a solid research paper."

**Changed.** Phase 1 (to 3 Dec) = controls E1–E4, rules pushed before any number (`4b4e883`):
one main control (crop the black bands, centre square, 256 px), one success rule (paired
MIRAGE − Inception-v3, 5th percentile over 100 draws > 0), fixed verdict sentences, a fixed
preprocessing grid, an error analysis, a source-separability test. Phase 2 (after 3 Dec) =
few-shot adaptation, with the Phase-1 audit as a standard check.

**Computer-science lesson.** A model "winning" means little until the strongest cheap
explanation is ruled out. Here the cheap explanations are image size, black letterbox bands and
the image source. Remove the cue, then re-measure; look at the failures, not only the average.

**Mathematics/protocol lesson.** Turn "does the advantage survive?" into one estimand and one
decision rule fixed before the data: a paired difference on the same draws, judged by its 5th
percentile, under one designated control. Secondary variants describe; they never rescue or
overturn the verdict. Write the possible sentences before seeing the numbers.

**Question to ask next time.** "What is the simplest thing other than the disease that could
produce this result, and have we removed it?"

## Feedback 3 (23 Nov 2026, first of two reviews on 23–24 Nov): one audit paper; a mechanism, not a sweep

**Said** (verbatim, as Lam relayed it on 23 Nov): "First, regarding the two phases and the
comparison table: Phase 1: The two pieces of feedback do not conflict. Feedback 1 (defining "rare"
based on label scarcity rather than forcing a comparison with Yoo) and Feedback 2 (focusing on
reliability/generalizability instead of ranking) both lead to the exact same paper: an audit.
He/She is currently treating them as two directions that need to be reconciled, whereas in
reality, a single paper satisfies both. The original direction—"FMs beat Inception-v3"—clearly does
not survive. Phase 1 is now an audit paper with a different claim. Phase 2, as proposed, expands in
scope but lacks a mechanism. "Adding more datasets, more rare diseases, and going beyond
ophthalmology" will simply reproduce this exact same problem on a larger scale, because every
benchmark has its own shortcuts, and there is no theory dictating which cues a frozen encoder picks
up. If you want a mechanism, you should package the shortcut audit itself into a portable
tool—"how to determine whether a few-shot benchmark is measuring pathology or data source"—and then
validate it across multiple datasets and modalities. That way, you achieve both breadth and a
underlying mechanism, which I think makes for a far stronger publication than just another
benchmark sweep. Regarding Provenance: v1 (119 images, matching the paper) vs. v2 (146 images,
which was used), along with the possibility that the RP and Stargardt labels were swapped, is a
provenance finding about a published benchmark and belongs in the main text, not a footnote. There
is one consistent pattern that hasn't been emphasized enough: MIRAGE only wins on the exact
benchmark that contains letterbox borders and size–class correlation; outside of it, the rankings
invert. Creating a main-text table that places benchmark vs. external results side-by-side to
highlight this rank inversion will be much more convincing than putting R2 as Table I. G1:
MIRAGE's rare-vs-normal AUROC on OCTID drops from 0.781 to 0.198 just from changing how the images
are fed into the model. Correctly classified OCTID normal cases drop from 202 to 0. Combined with
the fact that the median black border ratio is 0.34–0.50 in rare images versus ~0 in
Kermany/OCTID/OCTDL, this implies the "rare" signal is largely acting as a letterbox border
detector. This is also a main-text result rather than an appendix item. Most importantly, E1's
success criterion does not measure what Dr. Hieu asked; instead, it measures stability when
drawing support sets from a fixed pool of images. It already met that bar under uncontrolled
conditions (+0.001 at K=1). This means E1 is almost guaranteed to "pass" even if the margin of
advantage collapses from +0.116 down to +0.02 (which should actually be a fail). The real question
is: after controlling for these confounders, does MIRAGE still outperform the size-only baseline,
and by how much does the margin shrink compared to the locked protocol?"

**Checked against the files of record (23 Nov; no new number).** A first pass by the coordinator,
then a read-only cross-check by three models (GPT-6.2 Sol recomputed every number from the
per-draw archives; Gemini 4 Neon listed every ranking and placement rule; Claude Fable 5.2 reviewed
as a hostile referee). The first pass got three readings wrong; the corrected position:

- *One audit paper.* Right, and approved on 22 Nov. "FMs beat Inception-v3" fails in the plural
  (DINOv2 minus Inception-v3 goes from +0.024 to −0.044 under E1-M); MIRAGE's registered lead
  survives narrowly (+0.116 → +0.063). The draft still reads as a ranking paper (title, abstract,
  contribution 1, conclusion).
- *Provenance.* Already in the main text in five places (Related Work, Methods, three Limitations
  paragraphs); missing only from the abstract, introduction and results. The reviewer missed the
  stronger provenance finding, P10 (14 same-picture pairs; 4 of the 22 hold-out images copied in
  development). The v1/v2 mapping was computed before any rule.
- *Rank inversion.* Overstated. The order depends on task and class, not only on the source.
  Against Inception-v3, MIRAGE leads on CSC and trails on macular hole inside and outside the
  benchmark: P8 CSC vs MH −0.083; OCTID CSC vs normal +0.072; OCTID MH vs normal −0.100; OCTDL MH vs
  normal −0.277. DINOv2 leads on external disease against normal but calls the most external
  normals rare (OCTID 17/206, OCTDL 141/332); MIRAGE keeps the most OCTID normals NORMAL (202/206;
  206/206 under all three E1 controls) and is the most source-separable arm (E4). From P8 to the
  OCTID two-way test, MIRAGE is the only arm that does not fall.
- *G1.* The numbers are right (0.781 → 0.198; 202 → 0). "Largely a letterbox border detector" is
  forbidden by the 19 Nov G1 decision ("bars caused it"; "anything about what the features
  contain") and is contradicted as a general claim: cropping the bands from the benchmark images
  (E1-C, no bars anywhere) sent MIRAGE's OCTDL normals called rare from 30 to 228 of 332, a centre
  crop to 329; padding moved Inception-v3 the same way (OCTID normals 182 → 8 NORMAL); padded
  Kermany normals barely moved (169 → 166). The allowed reading is stronger than a hedge: report the
  three recipes' counts and say "external nine-way results depend on input geometry; we assign no
  cause".
- *E1.* The reviewer is right about what the rule measures (stability over support draws on 124
  fixed queries; no minimum margin; no paired comparison with raw size-only) and wrong that a pass
  was "almost guaranteed". Recomputed: with every draw shifted, the bar is a mean of 0.0606, and a
  +0.02 margin fails (p05 −0.041); with every draw scaled down, any positive mean passes. The
  observed controls behaved like a shift (paired SD 0.033–0.037 in all four feature sets; scaling
  would predict 0.020–0.025), so the bar of about 0.06 is fair at the observed spread; the valid
  form of the reviewer's worry is that a cue shared by both arms would shrink the spread. The pass
  is close to a coin flip: the mean sits 0.004 above the bar, and a 5th percentile from 100 draws
  has a sampling error of about 0.008 (normal approximation, never estimated in the rule). The
  reviewer's "real question", from frozen rows: the lead fell from 0.116 to 0.063 (−45.5%). On
  development this is arithmetically all Inception-v3's rise (0.534 → 0.586), whose whole input
  chain changed (the decoder check covered no rare image); on the hold-out MIRAGE also fell (0.722 →
  0.667) and under crop-only by 0.027. The rule's paired companion lines: MIRAGE E1-M minus the
  content-size reference +0.203 (p05 +0.149), minus the E1-M thumbnail +0.290. Against raw
  size-only, unpaired means only: 0.649 vs 0.569 on the K mean, 0.486 vs 0.578 at K = 1; p05 was
  below 0 at K = 1 and 2 and on the matched-size queries.
- *Found on the way.* `main.tex` still says "The narrow pass does not rest on the same-picture
  pairs", which the 22 Nov corrections called too strong; it is fixed in step 2.

**Changed.** Proposed, for Lam's decision; no number changes, but three recorded placement
commitments do (R2 as Table 1; A3's OCTID two-way last; G1 in the supplement, 19 Nov), so each
needs a dated decision by Lam and the mentor and a deviation row:
- a benchmark-against-external table built by a completeness rule (every comparison on file, the
  P8 row, CSC and MH split, false-rare counts), each row tagged registered, post hoc (rule date) or
  pre-rule, described as "the order differs by task and class"; the locked Table I stays in the
  main text (A3);
- G1 in the main text with the three recipes' counts and the allowed sentences only;
- provenance and P10 together as a short results subsection, labelled pre-rule;
- beside the unchanged E1 verdict: what the rule measures, the ~0.06 bar, the shrink and its split,
  the companion lines, the p05 below 0 at K = 1 and 2, MIRAGE's hold-out fall;
- the abstract and introduction say the audit framing was adopted after the locked results, at the
  reviewers' request.
Phase 2: the audit direction, validated where the truth is known; other modalities not yet. The
smallest first experiment (Fable): planted black bands on Kermany four-way K-shot, on Rule A's
patient-disjoint pools, at the rare images' measured border share, as a dose series, with a
negative control (bands on every class) and the size-only and thumbnail references; one Colab
session; it is the planted-cue test already listed in section 8. Rule A's pilot second; B parked.
Decided after the second review (24 Nov).

**Computer-science lesson.** Check a claimed pattern against every comparison on file before
building a table around it, and split by class before calling a rank change a shortcut: here the
"inversion" is mostly macular hole. A packaged audit is an instrument; planting a cue where the
truth is known is what turns it into evidence about a mechanism.

**Mathematics/protocol lesson.** Know what a decision rule's uncertainty covers: a percentile over
support draws from a fixed pool measures support-sampling stability, not image-sampling
uncertainty, and not the size of the advantage. A rule's detectable margin depends on how the
spread moves with the effect (shift: bar ≈ mean − p05; scaling: any positive mean passes); check
which one the data follow. When a paired difference shrinks, split it into its arms on every
evaluation set, not only the one that tells the cleaner story.

**What the coordinator learned about its own first pass.** Three errors survived until the
cross-check: a pattern confirmed on the comparisons that came to mind first, a fact taken from one
set (development) and stated generally, and a softened cause offered as a safe phrase. One
independent check per claim before writing it into a log.

**Question to ask next time.** "Does this rule's pass answer the question that was asked, and what
margin would it still call a pass?"

**Rebuttal outcome (23 Nov, two rounds; GPT-6.2 Sol for the mentor's view, Claude Fable 5.2 for the
project's; read-only).** Both sides ended on nearly the same positions:
- *Headline:* about the benchmark, not the models: image size alone predicts this benchmark's
  rare-disease labels better than any model at K = 1, and external "rare" calls move with input
  geometry. The model order by class (MIRAGE ahead on CSC, behind on macular hole) comes second.
  "No single story" was dropped as unfalsifiable; "border detector" stays out.
- *E1:* the registered pass stays, printed after the shrink (0.116 → 0.063) and the reference
  comparisons, never as a headline.
- *G1:* plain counts, not "we assign no cause": padding is an intervention, so "under black
  padding MIRAGE called 0 of 206 OCTID normals NORMAL (202 unpadded)" is stated as observed.
- *Table:* the locked table stays Table I; the complete synthesis is Table II beside it and is cited
  in the abstract; a swap needs a dated decision by Lam and the mentor.
- *Provenance:* one results subsection: 146 vs 119 images, the RP/Stargardt count mismatch, and 4
  of the 22 hold-out images with a same-picture copy in development (5 pairs; Stargardt/03 matches
  two).
- *Phase 2 is one line, not two:* Phase 1 measured non-pathology signal; the controls become a
  fixed audit; a planted cue with known truth calibrates it; the same-patient leak (rule A) is a
  second known-truth cue; then the audit goes to a second source. Patient diversity (question B)
  is off the line. Section 12 as filed inverts this (patients as the topic, the audit as a
  "standard check", and B promoted if A is null), so it needs reordering.
Lam's reading, "one straight line forward", holds once section 12 is reordered this way.

**Open items after the rebuttal (23 Nov; Claude Fable 5.2 leading, GPT-6.2 Sol and Gemini 4 Neon on
the frozen files, a Claude Opus cold read of the summary; read-only, no new number).**
- *Why outside normals are called rare.* No single cue fits every recipe on file. A border
  detector fits padding (G1, R-pad) and fails on band-free inputs (E1-C 228, E1-M 261, M-crop 329
  of 332 OCTDL normals called rare). The model-free thumbnail does the same (259 of 332; 331 under
  E1-M). Smallest account: outside scans change their similarity to the prototypes with every
  recipe and the nine-way head must choose; layout and zoom are candidates, not separated. The crop
  controls also re-processed the Kermany fitting images, so they never isolated the rare examples.
- *Macular hole: disease or size?* Cannot be told. The cross-check's "class effect, mostly macular
  hole" is withdrawn along with the reviewer's "wins only where the cue is". MIRAGE's 0.000 on
  size-wrong macular-hole queries at K = 8 rests on one fold and one outlier image (`MH/31.PNG`,
  491 × 387 among 25 images near 327 × 326); MIRAGE leads five-way macular-hole transfer to OCTID
  (0.702 at K = 8) and still trails on size-right macular-hole queries (0.708 against 0.804 and
  0.796).
- *The size-only question.* Under E1-M every input is a 256-pixel square, so the raw size rule has
  nothing to read; the same-input reference is the E1-M thumbnail (0.360; MIRAGE 0.649, paired
  +0.290, p05 +0.243). Raw size-only stays a finding about the benchmark (0.578 at K = 1, above all
  four models). No new analysis is needed.
- *One experiment for the first two items.* On Kermany, with patient-disjoint pools: borders and
  zoom planted on the supports and on the queries independently, in doses, with sham and all-class
  controls. It is step 3 of the Phase-2 line.
Lesson: a single striking cell (0.000) is a lead, not a finding, until its denominator is read.

## Working habits these feedbacks point to

- One main question per phase; extra ideas go to a list, not into the current plan.
- Few experiments, each with its rule written first.
- Prefer one careful analysis plus one independent check over many parallel opinions.
- When a result looks good, look for the shortcut first (P10 found 14 same-picture pairs the
  locked duplicate screen had missed).
- When a reviewer names a pattern, test it on every comparison on file, split by class, before
  agreeing or disagreeing (Feedback 3: "rankings invert" was mostly macular hole).
