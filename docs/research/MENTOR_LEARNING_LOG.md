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

## Working habits these two feedbacks point to

- One main question per phase; extra ideas go to a list, not into the current plan.
- Few experiments, each with its rule written first.
- Prefer one careful analysis plus one independent check over many parallel opinions.
- When a result looks good, look for the shortcut first (P10 found 14 same-picture pairs the
  locked duplicate screen had missed).
