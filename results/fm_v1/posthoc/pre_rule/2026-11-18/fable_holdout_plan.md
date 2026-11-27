# Hold-out and comparability with Yoo et al.: rebuttal and plan (Fable 5.2, 18 Nov 2026)

Read-only review. No number was computed on rare images or model outputs. "SP" = the session scratchpad, `C:\Users\THINKB~1\AppData\Local\Temp\claude\c38f92eaa1d8ed7a\34630c3a-5fe8-47a5-a7e3-d2b9d1ac6536\scratchpad`.

## A. Rebuttal of the Opus position

1. **The biggest gap is novelty, not the hold-out.** Sun et al. 2024 already ran K-shot (K = 1, 3, 5) five-way tasks on Yoo's rare images (SP `cites\agentB.md:7`). Our positioning says "no K-shot budget" for these images (`docs/research/FM_AMENDMENTS.md:311`) and "nobody in the studies we read" (`protocols.md:30-33`). That was true of the studies read on 13 Nov. It is not true now. Opus treats Sun only as a comparator. Sun is first a correction to claim 1. A3 cannot be edited (`protocols.md:364-365`), so the paper and a dated line in `protocols.md` must narrow the claim.
2. **"Bridge to Yoo's method" overclaims.** A3 says Yoo's numbers and our replication's are "context only" (`FM_AMENDMENTS.md:237-238`). The replication heads were fitted on all 124 development images plus generated ones; the FMs see K per class. Same 22 images, different information. That comparison already exists as P2 (17/18/19 of 22; `protocols.md:215`, `:302`) and is labelled outside the locked study (rule 4, `protocols.md:204-206`). Nothing new is needed; do not rename it.
3. **"Development is already a one-set design like Yoo's 5-fold" is half right.** It is like Sun's design. It is not like Yoo's 5-fold: nine-way, every rare image plus GAN images, accuracy dominated by 1,000 common test images (SP `yoo_ctx.txt:9-13`). Say "resampling design, as in Sun et al.", not "as in Yoo".
4. **Moving to one set is not available in v1.** A3 puts "legacy 22 with exact counts per class and arm" in the main text (`FM_AMENDMENTS.md:187`, `:137-139`). Rule 2 forbids replacing it (`protocols.md:199-202`). "Direction only" was already adopted (`protocols.md:348`). So Opus is right on the lock, but for the placement reason, not only "splits never move".
5. **P11 is weaker than presented.** Sun's pool cannot be rebuilt: they report MacTel 30, version 1 has 24, version 2 has 29 (SP `cites\agentB.md:7`; task fact 1). They use 3 random tasks and a trained ResNet-12. We already know our development level at K = 4 and 8, so choosing to add P11 now is outcome-informed (`protocols.md:135-136`). It would be the 12th post hoc item with three days left, while P10's review and G1 scoring are open (`protocols.md:381`).
6. **Provenance finding: reportable, with a label.** It is file identity, like the duplicate screen (`protocols.md:102-104`); no score is involved. But it was done after P10's rule was committed (`dcbb323`) and before it was pushed (branch is 3 commits ahead of origin: `git status -sb`). P10 step 1 names Yoo's source links, not Mendeley versions (`protocols.md:223`). So: push first, then record it as "found 18 Nov, before P10's rule was pushed, not in P10's text", release the mapping in `results/fm_v1/posthoc/pre_rule/` (precedent `protocols.md:351-355`). The mapping now lives only in the scratchpad, which can be wiped. Do not store Mendeley version 1 images in `data/raw` (AGENTS.md rule 1).
7. **Missing: the RP/Stargardt label question.** Paper text gives Stargardt 15+4 = 19 and RP 12+4 = 16; folders give RP 19, Stargardt 16 (task fact 3). Either the text or the folder names are swapped. Macro recall is unaffected. Per-class names for these two (P0, `protocols.md:213`) need a footnote.
8. **Missing: "Yoo's model" is not one thing.** Supplement code (mixed7, 256 px, sigmoid) differs from the paper text and from our arm (`stage3_classifier.py:108,278`; SP `web_yoo\esm_text.txt`). A3's sentence "It is the backbone of Yoo et al." (`FM_AMENDMENTS.md:225`) holds only as "same backbone family". This is the mentor's Q3 option (`protocols.md:262`).
9. **Overreach to avoid:** "21 of 22 are in version 1" does not mean overlap with Yoo's test. It only means one image (RP/20) cannot have been in it.

## B. Handling plan

**Stays locked:** manifest, both sets, nominee, every locked number, A3 placement (hold-out in main text, "historically inspected").

**Reworded (no new number):** W3 hold-out text (D1); novelty claim narrowed for Sun; Inception-v3 called "the backbone family of Yoo et al."; Yoo's 20/22 and 92.1/93.9 cited as published context only; RP/Stargardt footnote; `README.md:336-337` ("matching the paper's Fig. 2") fixed under W3 (`protocols.md:230`).

**Added, labelled:** one provenance paragraph in Data (version 1 = 119, version 2 = 146, 28 additions, one replaced MacTel image), with the mapping file committed. A version column joins P10 step 1 by a dated change line. No score is recomputed by version. P1's leave-one-image-out already covers removing any single hold-out image (`protocols.md:214`); report its min and max as the rule says, do not single out RP/20.

**Housekeeping:** add Sun 2024 and Tai 2025 to section 10 with a reason (`protocols.md:399-400`); add a dated correction to `docs/research/lit-scan-gan-vs-basic.md` Q2 (`:89`, `:129`), which missed Sun.

**Moves to v2 (pre-specified):** one pool with repeated episodes; Sun-style arm (K = 1, 3, 5, query = all remaining); version-1-only pool as a planned sensitivity; GAN arm only if cheap.

## C. Comparisons, ranked

| # | Comparator | Lets us claim | Cannot claim | Cost | Rule status | Before 21 Nov? |
|---|---|---|---|---|---|---|
| 1 | Inception-v3 prototype arm | Each FM against Yoo's backbone family, same supports and queries | Anything about Yoo's trained pipeline | None | Locked (`FM_AMENDMENTS.md:225-227`) | Yes, exists |
| 2 | Size-only reference | Whether a model beats image shape | Why | None | Locked | Yes, exists |
| 3 | Our GAN-pipeline heads on the 22 (P2) | Direction: full-data heads beside K-shot FMs, identical images | Ranking; n = 22, inspected | None | Post hoc, done, rule 4 | Yes, exists |
| 4 | Sun 2024 as cited context | Prior K-shot work exists on these images; design is alike | Any numeric win or loss: other pool, trained net, 3 tasks | Wording | No rule; section 10 line | Yes |
| 5 | Yoo's S3 matrix (20/22; both misses called Normal) | Their published rare counts, as context | Same images; class names of the misses (column order, SP `web_yoo\esm_p7_0_Image138.png`) | Wording | Context only (A3) | Yes, cite only |
| 6 | Yoo's 92.1 / 93.9 | That these are nine-way, common-class-dominated | Any comparison with five-way macro recall (`protocols.md:90-91`) | Wording | Context only | Yes, cite only |
| 7 | External OCTID / OCTDL | False-rare rate; agreement with published labels | Rare-disease transfer (A1 rule 7) | None | Locked | Yes, exists |
| 8 | Sun protocol as P11 | A number on Sun's design | Same pool; a fair contest | About 1 day with tests | Needs new rule, pushed first | No. v2 |
| 9 | Version-1-only (119) rerun | Sensitivity to the 28 additions | Yoo's split | Half a day | Needs new rule | No. v2 |
| 10 | Tai 2025 | An example of loose comparison to avoid | Anything: test set undefined (SP `cites\agentA.md:13`) | Wording | — | One clause at most |

## D. Draft wording

**(1) Hold-out (W3).** "The hold-out is 22 rare images (5 CSC, 5 MH, 4 MacTel, 4 Stargardt, 4 RP) drawn once with a fixed seed from the public release. It matches the class counts of the test in Yoo et al.; which images they tested on was never published. These 22 images were inspected in our earlier work, so the test is not blind. We report exact counts and read them for direction only."

**(2) Comparability.** "Yoo et al. reported 92.1% accuracy on an independent test and 93.9% in five-fold cross-validation for a nine-way classifier trained on every rare image plus generated ones; most test images were common classes. Their test files are not public, and the public release changed after submission (119 images in version 1, 146 in version 2, which we use). Their figures are therefore context, not a baseline. Later work reusing these images did not rebuild that test either. Sun et al. (2024) drew repeated five-way K-shot tasks from the rare images with a trained network; our development protocol follows the same kind of design, with frozen encoders, grouped folds and 100 paired draws. Pools, draws and training differ, so we do not rank our results against theirs. Within this study, every comparison uses identical supports and queries, including frozen Inception-v3, the backbone family of Yoo et al."

**(3) Limitation.** "The rare images come from web searches, carry no patient identifiers, and total 19 to 27 development images per class; our results describe these images, cannot be compared number-for-number with earlier studies, and the folder labels for RP and Stargardt disagree with the counts in the original paper."

## E. Choices for Lam

1. **Wording and provenance only (recommended).** Do section B. No new score. Push `dcbb323` first. Fits 21 Nov; fixes the novelty claim, which is the real exposure.
2. **Option 1 plus P11.** Rule written and pushed by 19 Nov, supplement, labelled "outside the protocol, added after the development results were known". Only if the mentor asks for a number beside Sun's.
3. **Option 1 plus version-1-only sensitivity.** Low value: it does not recover Yoo's split.
4. **Retitle** as a frozen-encoder benchmark on the Yoo images, Yoo comparison in the discussion (`protocols.md:260`). Compatible with option 1; worth raising with the mentor.

Unverified by me: Mendeley API contents and hashes (task fact 1), Sun's 77.96 and Wilcoxon (not in SP `cites\sun_ctx.txt` excerpts I read; taken from `cites\agentB.md:7`), and the S3 image itself.
