# Three paper discrepancies: what Lam needs for D11

Verified 27 October 2026, without training. L8 and L10 have mathematical
incompatibilities under stated assumptions; L6 remains an unexplained baseline
gap. The investigation is complete at this scope, but the authors' missing
implementation details remain open. D11, whether to contact them, is Lam's decision.

**How to find the evidence.** Paper citations below refer to
[Yoo et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7829497/), checked through
the [Europe PMC full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7829497/fullTextXML).
The local PDF directory was unavailable. References marked **L8 branch** mean
`task/l8-rci` at commit `6277e64b7a667fdf5ef647815576611a6b2433fa`, readable with
`git show task/l8-rci:<path>`. These files are now on main (PRs #2 and #8). All file line numbers refer to the versions checked
for this report.

## RCI (L8): our calculation is sound, but its paper comparison is unresolved

**Answer:** Our relative classifier information (RCI) implements
`I(true; predicted) / H(true)`, the definition identified in L8's citation trail;
the paper's GAN RCI of **0.983** cannot come from that formula with its own
Table 4 recalls and stated test counts, which allow at most **0.970242**.

RCI asks how much knowing the prediction reduces uncertainty about the true
disease. `H(true)` means the uncertainty before seeing a prediction;
`H(true | predicted)` means what remains afterward. Their difference is the
information gained, written `I(true; predicted)`. Thus:

```text
RCI = (starting uncertainty - remaining uncertainty) / starting uncertainty
```

**A small worked example.** Imagine two diseases with four images each. Rows
are true diseases; columns are predictions. This is an illustrative calculation,
not research data.

```text
                 Predicted A    Predicted B
True A                3              1
True B                1              3
```

Before seeing a prediction, A and B are equally likely. Uncertainty is
`H(true) = -0.5 log2(0.5) - 0.5 log2(0.5) = 1 bit`.
Among the four predictions of A, three really are A and one is B; predictions
of B have the same proportions. Remaining uncertainty for either prediction is
`-0.75 log2(0.75) - 0.25 log2(0.25) = 0.811278 bits`.
So `I = 1 - 0.811278 = 0.188722 bits`, and **RCI = 0.188722**.
Accuracy is **6/8 = 75%**. RCI is therefore a different quantity from accuracy;
75% accuracy does not mean RCI should be near 0.75. Using natural logarithms
instead of base two changes the units but leaves the ratio unchanged.

**Our nine-class results.** For `gan`, the saved confusion matrix contains
**950/1,022** correct predictions, or **92.95%** accuracy. In the natural-log
units used by our code, starting uncertainty is **1.494888**, remaining
uncertainty is **0.277988**, and information gained is **1.216900**. Therefore
`RCI = 1.216900 / 1.494888 = 0.814041`, with rounding only for display.
Sources: [confusion_gan.csv](../../results/stage4/confusion_gan.csv), lines 2–10;
[metrics.csv](../../results/stage4/metrics.csv), line 4;
[stage4_evaluate.py](../../stage4_evaluate.py), lines 47–55.

The saved and independently recomputed RCIs, followed by the paper's Table 3
values, are:

- `none`: **0.787521**, paper **0.916**.
- `basic`: **0.760097**, paper **0.953**.
- `gan`: **0.814041**, paper **0.983**.
- `knn`: **0.661864**, paper **0.972**.
- `proto`: **0.429951**, paper **0.933**. Our `proto` is a class-mean baseline;
  the paper's is a trained prototypical network, so these are different models.

Sources: `results/analysis/rci_definitions.csv`, lines 2–6 (**L8 branch**);
[metrics.csv](../../results/stage4/metrics.csv), lines 2–6; paper Table 3.
The existing Codex audit, independently reviewed here, also agrees:
`C:/Users/ThinkBook/.t3/worktrees/Few-shot/t3-61cd05e7/docs/research/L8_RCI_AUDIT.md`,
lines 6–29 (“Supported conclusion”).

**What is proven.** Our function agrees with an independent calculation of
the remaining uncertainty for all five saved confusion matrices. The L8 bound
asks a different question: keeping the paper's class counts and recalls fixed,
could *any* arrangement of its mistakes give RCI 0.983? Its exhaustive maximum
is **0.9702419109**, reverified here. Intuitively, even arranging every class's
mistakes in the most informative possible way cannot reach the reported value.
This statement concerns the paper's inputs, not our lower-performing classifier.
Sources: `analysis.py:133` and `results/analysis/rci_paper_bounds.csv:4`
(**L8 branch**); paper Table 4 and Methods, “Characteristics of the datasets”.

**What is likely, and what is unknown.** A definition, aggregation, input or
reporting difference is a plausible explanation. The paper names RCI but does
not print its equation. L8's bibliography traces the normalized definition
through earlier literature; the earlier Yoo paper's full text was not opened,
so this trail does not establish the authors' actual implementation. The Codex
audit likewise distinguishes agreement with the documented definition from
agreement with the OCT authors' code. Sources: paper “Statistical analysis”;
`docs/research/REPLICATION.md:66` and `docs/research/references.bib:4`
(**L8 branch**); the audit's lines 31–50 (“Source boundary”).

L8 tested ten formulas or normalizations, including swapping the true and
predicted axes. Rechecking their saved calculations found no choice that
reproduces the paper's RCI column on our matrices. This rules out those simple
reinterpretations as an explanation of our comparison; it does not test every
possible formula or recover the paper's unshared confusion matrices.
Source: `analysis.py:114` and `results/analysis/rci_definitions.csv:1`
(**L8 branch**).

**What would settle it.** Ask for the exact RCI code or equation, its inputs
and aggregation rule, and the independent-test confusion matrices or
predictions. Calculate both versions from those same predictions. Until then,
retain our explicitly defined RCI as descriptive and avoid treating its gap
from Table 3 as evidence of worse classification. No training is needed.

## Table 3 vs Table 4 (L10): two entries cannot describe the same predictions

**Answer:** With the paper's stated test counts, Table 4 implies **86.1%**
accuracy for the prototypical network and **92.0%** for GAN, whereas Table 3
reports **80.8%** and **92.1%**, respectively.

**Worked example.** Overall accuracy counts correct images, so multiply each
class's recall by its test-image count and add the results. The paper states
**250 images per common class**, plus **5 CSC, 5 macular hole, 4 MacTel,
4 retinitis pigmentosa and 4 Stargardt images**, totaling **1,022**.
For example, normal-class GAN recall **0.912** means `250 × 0.912 = 228`
correct images. Source: paper Methods, “Characteristics of the datasets”, and
Table 4. In the paper's column order, the full calculation is:

```text
                     Normal CNV DME Drusen CSC MH MacTel RP Stargardt
Prototypical correct:   218  229 208    205   5  5      4  3         3
GAN correct:            228  237 233    222   5  5      4  3         3

Prototypical: 880 / 1,022 × 100 = 86.105675% -> 86.1%
GAN:          940 / 1,022 × 100 = 91.976517% -> 92.0%
```

Every count above was recalculated directly from Table 4. The resulting
accuracies also match `results/analysis/rci_paper_bounds.csv`, lines 6 and 4
(**L8 branch**). Conversely, Table 3's rounded **80.8%** corresponds to
**826** correct images, **54 fewer** than Table 4 implies; **92.1%** corresponds
to **941**, just **one more** than Table 4 implies.

**What is proven.** Under the stated class counts and ordinary overall
accuracy, both pairs cannot simultaneously describe one set of predictions.
Rounding does not fix this: with 250 images, one correct image changes recall
by **0.004**, exceeding the precision of the three-decimal recalls. The rare
class recalls likewise determine integer counts. This is a consistency check
inside the paper, independent of our reproduction.

**What is likely, and what is unknown.** A transcription mistake or values
from different runs could explain the mismatch. The GAN difference is small;
the prototypical-network difference is substantial. We cannot identify which
entry needs correction or whether different runs were actually used. Neither
table should silently be substituted for the other.

**What would settle it.** Obtain the confusion matrices or predictions used
for each table, confirmation that both describe the same run and split, and
any corrected values. Training our models cannot repair this internal mismatch.
It also makes the RCI bound conditional on Table 4 being the intended input.

## kNN (L6): the gap is real, but its cause is not established

**Answer:** Our **86.40%** versus the paper's **91.2%** remains unexplained;
changing to cosine distance and K=1 gives **91.10%** without training, but
changes the paper's stated method and therefore does not resolve the gap.

**Worked example.** k-nearest neighbours (kNN) classifies an image using nearby
training examples. Euclidean distance compares feature-vector positions;
cosine distance compares their directions. K determines how many neighbours
vote. The saved sweep on our frozen ImageNet features gives:

- Euclidean, **K=3**: **86.399217%**, displayed as **86.40%**.
- Cosine, **K=3**: **86.986301%**, displayed as **86.99%**.
- Cosine, **K=1**: **91.095890%**, displayed as **91.10%**.

Sources: [knn_sweep.csv](../../results/analysis/knn_sweep.csv), lines 3, 7 and 6;
paper Table 3 for **91.2%**. Our K=3 Euclidean confusion matrix independently
gives **883/1,022 = 86.399217%**: [confusion_knn.csv](../../results/stage4/confusion_knn.csv),
lines 2–10. Cosine K=1 is close to the target, but inspecting test performance
to select a different method cannot validate the specified baseline.

**What is proven about the wording.** In its metric-learning paragraph, the
paper says it used the **“Inception-v3 model trained without data augmentation”**
as the backbone for kNN and the prototypical network. It specifies Euclidean
distance and K=3. In the classifier-training paragraph, however, it explicitly
states that convolutional layers stayed frozen and only the final fully
connected layer was trained. These statements must be read together.

Our kNN reads average-pooled ImageNet backbone features, before our trained
softmax layer. Sources: [stage3_classifier.py](../../stage3_classifier.py),
lines 68–72 and 183–184; [stage4_evaluate.py](../../stage4_evaluate.py),
lines 87–108. Training that final layer changes its outputs, but cannot change
features extracted before it. Thus the word “trained” alone does **not** prove
that the authors used learned convolutional features different from ours.

**What is likely, and what is unknown.** The sweep shows sensitivity to the
neighbour rule, so implementation details are plausible contributors. A
different feature layer or normalization could matter, but the feature layer,
any postprocessing and the exact checkpoint used for the paper's kNN are
unverified. The present evidence does not show which detail explains the gap.

**What would settle it, without training.** Ask for the exact feature layer,
checkpoint, preprocessing, normalization, training pool and kNN settings.
Re-evaluate K=3 Euclidean using that recipe if the necessary artifacts are
available. As a separate hypothesis check, our existing `none` head and cached
features could generate final-layer outputs for kNN without new training;
that check has not been performed here and would not establish the authors'
feature choice by itself.

**What training would test.** If the authors used a learned final-layer
representation unavailable in their artifacts, reproduce their no-augmentation
classifier training on Colab and run K=3 Euclidean on the confirmed layer.
If they confirm that convolutional layers were updated for this baseline,
training that backbone and re-extracting features would test the learned-feature
explanation. That would need clarification of the paper's frozen-layer statement
and Lam's scope decision: end-to-end fine-tuning is currently out of scope
([CONTEXT.md](CONTEXT.md), lines 18–21). Merely retraining a head while still taking
the same frozen backbone features cannot test that explanation. No training was
run for this report.

**Proposed README edits for T1; not applied here.**

1. Link this report from “Differences from the paper” and the RCI footnote.
   Suggested wording: “Our RCI calculation is verified. Under
   I(true; predicted)/H(true), the paper's Table 4 GAN recalls permit at most
   0.970242, below its reported 0.983. The authors' implementation is unknown.”
2. Add: “Tables 3 and 4 disagree internally: Table 4 implies prototypical-network
   accuracy 86.1% versus 80.8%, and GAN accuracy 92.0% versus 92.1%.”
3. Clarify the kNN row: “86.40% versus 91.2%; feature layer and postprocessing
   unresolved. Cosine K=1 reaches 91.10% but changes the specified baseline.
   The paper says the classifier backbone was frozen, so a learned-backbone
   explanation requires confirmation.”

**Verification and scope.** Only this report was added. Independent conditional
entropy calculations matched our RCI function and saved values; the L8 maximum
enumeration reproduced the GAN bound; all ten saved definition calculations
were checked; paper test counts and Tables 3–4 were read directly; sweep values
were checked against the saved CSV. No evaluation outputs were overwritten.
Repository check output, using the requested interpreter:

```text
.agents/check exit=0 (Python: D:/Few-shot/.venv/Scripts/python.exe)
```

Pre-completion surfaces: entry points, notebook, reverse actions, data schemas
and feature-cache path lists are unchanged. Docs apply through this new report
and the README proposals above; method, data and recorded results are unchanged,
so no CONTEXT or COLAB update is needed. Checks pass; raw-data verification was
not exercised because this worktree lacks the raw OCT2017 directory.

## Options for D11

**Email now.** The benefit is a short, answerable request for information only
the authors can provide: RCI implementation, table inputs and kNN feature
details. We have enough evidence to ask without alleging a faulty study. A
reply could resolve all three issues without additional training.

**Keep D11 on hold.** This avoids correspondence and lets Lam focus on the
reproduction's main accuracy results. The cost is that RCI comparability and
the baseline details remain unresolved. Further local formula hunting cannot
identify unshared code, and matching one headline number would not settle it.

**Recommendation for Lam to consider:** email once the distinction is clear:
we have verified our RCI calculation and a conditional contradiction in the
published numbers; we have not established what the authors actually computed.
No message has been sent. Lam should check the worked examples and decide
whether resolving these points would change the project's interpretation or
next experiment.

### DRAFT, NOT SENT

Subject: Clarification of RCI, independent-test tables and kNN features

Dear Dr Yoo and colleagues,

I am a student reproducing your 2021 OCT few-shot classification study
(DOI: 10.1007/s11517-021-02321-1). Could you help clarify three details?

First, what equation or code calculated relative classifier information, and
was it computed from pooled predictions or averaged across classes? Using
I(true; predicted)/H(true), the test counts and GAN recalls in Table 4 give a
maximum possible RCI of about 0.970, below the 0.983 in Table 3. I may be
missing a definition or input difference. Would you be able to share the RCI
implementation and independent-test confusion matrices?

Second, multiplying Table 4's recalls by the stated class counts gives
880/1,022 correct for the prototypical network (86.1%) and 940/1,022 for GAN
(92.0%), whereas Table 3 reports 80.8% and 92.1%. Do the tables use the same
run and split, and are any values corrected elsewhere?

Finally, for K=3 Euclidean kNN, which layer supplied features from the
no-augmentation Inception-v3 model? Were these frozen backbone features or
trained final-layer outputs, and was any normalization applied?

Thank you for your time and for making the study available.

Best regards,
Lam
