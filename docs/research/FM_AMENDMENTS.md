# Dated amendments to protocol v1.0

`FM_PROTOCOL.md` stays byte-identical (SHA256 `4e8df8d9…c5df`, pinned in `fm_prepare.py`).
Amendments are recorded here, dated, before the input lock and before any score they affect.
Each one names what it replaces; everything else in v1.0 stands.

## A1. External rare labels (12 November 2026, before the input lock)

**Lam's decision, verbatim:** "We adopt published dataset labels for exploratory external
evaluation only. Results measure agreement with those labels, not clinically validated disease
transfer. Cohort eligibility is fixed before inference, and all planned results will be
reported."

**Replaces**, for the cohorts below only: protocol section 8, "CSR-to-CSC needs clinical
label-equivalence confirmation", and the fallback "If rare mapping fails, retain NORMAL FRR and
omit that disease-transfer claim". No clinical equivalence is claimed, so no confirmation is
required; the transfer claim is replaced by a narrower agreement claim.

**Eligible cohorts, fixed now:**

| Cohort | Images | Basis | Use |
|---|---|---|---|
| OCTDL `disease=VID` and `subcategory=MH` | 39 (28 patients) | OCTDL's MH definition ("a complete defect in the inner layers of the retina that extends to the RPE") matches the IVTS full-thickness MH definition | Query only; grouped by OCTDL patient ID |
| OCTID CSR | 102 | OCTID's paper describes the class under "Central Serous Chorioretinopathy (CSC)"; MeSH D056833 lists "Central Serous Retinopathy" as an entry term | Query only |
| OCTID MH | 102 | OCTID's own broad definition ("small hypoflective breaks to full-thickness intra-retinal spaces") | Query only; a separate cohort, never pooled with OCTDL MH, never called full-thickness MH |
| OCTID NORMAL, OCTDL NO | 206; 332 (110 patients) | Unchanged from v1.0 | Nine-way NORMAL FRR |

Excluded: OCTDL VID `subcategory` LH and VMT (their `condition` column also reads "MH"),
OCTID AMD and DR, OCTDL AMD, DME, ERM, RAO and RVO.

**Rules:**
1. External images are evaluation queries only; none enters fitting, supports or nomination.
   MacTel, Stargardt and RP receive no external claim.
2. External rare results are exploratory and descriptive. Prior studies (TVST 2026, MIRAGE
   2025, EyeDiff 2026) are cited as precedent for evaluating published categories, not as
   evidence of label equivalence. The benchmark's own CSC and MH definitions remain unverified.
3. OCTID two-way CSC-versus-MH test: benchmark-only CSC and MH supports, cosine prototypes only
   (no logistic probe). Report per-class recall, macro recall, confusion counts and denominators
   for every locked K and arm, favourable or not. OCTDL MH and the five-way head's outputs on
   external rare images are reported as output shares with denominators.
4. No claim that acquisition shortcuts are removed. Record image geometry and padding by class
   and source (benchmark crop width differs by class, e.g. CSC about 414 px, MH about 327 px),
   and report the source probe as a diagnostic, not a gate.
5. Intervals: OCTDL patient-cluster intervals (5,000 draws, seed 0, as v1.0); OCTID intervals are
   image-level and labelled conditional, because OCTID gives no patient IDs.
6. Eligibility is fixed before inference. Later criticism of a mapping (for example from the
   mentor) is reported as a limitation; scored results are never withdrawn or replaced.
7. Required wording: "We evaluated agreement with OCTID's and OCTDL's published CSC/CSR and MH
   labels using benchmark-only K-shot supports. Clinical label equivalence, patient
   independence and freedom from acquisition shortcuts were not established."

**Provenance:** draft and evidence in `FM_LABEL_RULE_PROPOSAL.md` and `FM_EXTERNAL_MAPPING.md`;
independent rebuttals by GPT-6.1-Sol (Codex) and Gemini 4 Argon High, 12 November 2026, both
"accept with changes"; revised rule above approved by Lam the same day.

**Still to do before the lock:** implement rules 3-5 in the external code and notebook; hash this
file into the input lock with the protocol; update paper paragraph C25, which still says
"clinical confirmation", in the next draft.
