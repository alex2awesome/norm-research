# ROBIS: A new tool to assess risk of bias in systematic reviews was developed
SOURCE_URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4687950/
DOMAIN: academic

ROBIS (Whiting et al., J Clin Epidemiol 2016) is, distinctively among appraisal instruments, designed specifically to judge **risk of bias** in a systematic review — as opposed to AMSTAR-style general "methodological quality" — with its final and most directly relevant output for claim-matching being an explicit judgment of whether **the review's interpretation/conclusions are actually supported by the evidence it synthesized**. It targets reviews of interventions, diagnosis, prognosis, and etiology in healthcare, and its primary audience is guideline developers and authors of "reviews of reviews" who need to decide how much to trust a given review's stated claims.

## Three-phase structure

**Phase 1 — Assess relevance (optional).** The assessor defines the question they actually care about using PICO (or an equivalent framework) and rates how well the systematic review's own question aligns with it ("yes"/"partial"/"no"). This phase can be skipped when a review is being assessed on its own terms rather than against an external question.

**Phase 2 — Identify concerns about bias in the review process.** Four sequential domains, each composed of signalling questions answered "yes," "probably yes," "probably no," "no," or "no information" (with "yes" indicating low concern). Each domain then receives a single judgment of **"low," "high,"** or **"unclear"** risk of bias.

**Phase 3 — Judge overall risk of bias**, assessing whether the review's stated interpretation/conclusions address all the concerns identified across the four Phase 2 domains.

## Phase 2 domains (the actionable core)

### Domain 1: Study eligibility criteria
Focus: were the review's inclusion criteria prespecified, clear, and appropriate to the review question? Key considerations: criteria should be explicitly defined *before* study selection began (ideally documented in a protocol/registration predating the review), and the criteria must have been applied consistently across candidate studies rather than adjusted post hoc to fit a desired conclusion.

### Domain 2: Identification and selection of studies
Focus: were eligible studies comprehensively identified and selected without bias? Key requirements: sensitive searches across appropriate databases and both published and unpublished sources; supplementary search methods such as reference-list checking, citation searching, and hand-searching; a search strategy combining free-text terms and controlled vocabulary/subject indexing; and independent screening by at least two reviewers at both the title/abstract and full-text stages.

### Domain 3: Data collection and study appraisal
Focus: was bias introduced during data extraction or risk-of-bias assessment of the primary studies? Key elements: a planned, structured data-collection form piloted before use; comprehensive collection of numerical, statistical, and study-characteristic data (not just outcomes favorable to a hypothesis); duplicate data extraction, or rigorous single extraction with independent verification; risk-of-bias assessment of primary studies using validated tools or clearly stated criteria; and at least two reviewers independently assessing (or verifying) study validity.

### Domain 4: Synthesis and findings
Focus: whether the synthesis method itself was appropriate and honestly reflects the underlying evidence. Considerations quoted from the tool: "whether the analytic approach is appropriate for the research question posed; whether between-study variation (heterogeneity) is taken into account; whether biases in the primary studies are taken into account; whether the information from the primary studies being synthesized is complete..." Critical aspects include choosing an appropriate synthesis method (quantitative meta-analysis vs. narrative synthesis) for the data at hand, accounting for study-level bias when interpreting pooled results, addressing publication/reporting bias, and avoiding computational errors in meta-analytic calculations.

## Phase 3: Judging whether conclusions are supported by the evidence

This is the phase most directly about claim-evidence matching. Signalling questions evaluate: whether the review's interpretation addressed all concerns identified in Phases 2's four domains; whether the review's stated conclusions are actually supported by the evidence that was synthesized; and the overall quality of the interpretation despite whatever limitations were identified. The overall judgment is again "low," "high," or "unclear" risk of bias.

The tool's key structural principle: **a systematic review can be judged low risk of bias even if its included primary studies are individually high risk of bias — provided the review explicitly identified, assessed, and accounted for that risk when drawing its conclusions.** In other words, ROBIS distinguishes "the underlying evidence is weak" from "the review claims more than the evidence supports" — a review can honestly caveat weak evidence and still pass; a review can also draw an overclaimed conclusion from strong-looking evidence and fail. The claim-evidence match being tested is specifically whether the stated conclusion's confidence level tracks the actual limitations of the review process and the underlying evidence, not whether the evidence itself is flawless.

## Constraints and cautions

- ROBIS explicitly avoids generating a **summary numeric quality score**, which the authors flag as methodologically unreliable and liable to obscure which specific concern actually undermines the review's conclusions.
- Applying the tool requires both methodological and content-area expertise from the assessor.
- The four Phase-2 domains are meant to be considered **sequentially** — problems identified early (e.g., a biased eligibility criterion) can propagate into and compound concerns found later (e.g., in synthesis) — rather than as four independent, siloed checks.
- Reporting quality of the review under assessment directly affects how feasible ROBIS is to apply: Cochrane reviews typically provide sufficient methodological detail; many non-Cochrane reviews do not, which itself becomes evidence of possible unassessed risk of bias (many "no information" answers to signalling questions should push a domain toward "unclear" or "high" rather than being ignored).

## Actionable takeaway for claim-matching

ROBIS operationalizes the claim-matching question at the level of the *review* rather than the *study*: given the identified weaknesses in eligibility criteria, search/selection, data collection/appraisal, and synthesis method, does the review's stated conclusion still follow? A conclusion that ignores or downplays known limitations in its own evidence-gathering process is exactly the failure mode ROBIS Phase 3 is built to catch, independent of whether the underlying primary studies themselves were well or poorly conducted.
