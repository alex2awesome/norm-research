# Empirically testing whether a metric is verifiable / articulable / dense

Date: 2026-05-22. Discussion + design notes. Companion to
`2026-05-16__validity-experiments-plan.md` (which lays out the E1–E6
experiment plan); this file is the higher-level framing of *what* we are
testing and *why* the approaches we have don't quite suffice on their own.
References `[[project_verifiability_explainability_gaps]]` in memory.

## The problem

We want to assign each rubric / R1 rule one of:

- **L1 Verifiable** — a program / regex / lint / test resolves it.
- **L2 Articulable** — code can't, but an LLM-judge given the rubric can.
- **L3 Defensible-judgment** — neither code nor LLM-judge alone, but briefed
  humans agree above chance.
- **L4 Fully tacit / taste** — even briefed expert raters disagree.

The catch: **we do not have per-rubric gold-truth feature labels.** We have
labeled datasets for the *ultimate outcome* (e.g., PR accept/reject,
peer-review accept/reject), but not for the individual feature values
("does this PR have a 23-line function?"). If we had per-feature gold, the
test would be trivial: ask a coding model to implement the rubric, check
held-out fidelity against gold — high fidelity = verifiable; if the code
fails but LLM-judge nails it = articulable; etc.

So we need *proxies* for the level assignment.

## Alex's initial ideas (and a structural concern)

1. **Consistency** — does a rubric give the same score across prompt
   variations (LLM-as-judge case) or across coding-model variations
   (code case)?
2. **Predictive ablation** — try to code up every rubric. The ones that
   actually help predict outcome are the "verifiable" batch. Take everything
   left, run as LLM-judge; the ones whose LLM-judge scores help predict are
   "articulable." Whatever helps a dense reward model on top of these is the
   "taste" residual.
3. **LLM self-report** — already implemented (the gpt-5-mini v6–v9 prompts).
   Acknowledged as tangential; introspection on level doesn't pin down the
   L2/L3 boundary (scores swung 11→6→17→17 on the same 33 clusters across
   prompt versions).

**The structural concern with (2).** It conflates two different properties:

- *Measurability* — can we get an agreed score from this rubric at all?
- *Predictive utility* — does the score help predict the outcome?

A rubric like "use Oxford commas" is perfectly verifiable but probably
uncorrelated with PR acceptance — it would be wrongly excluded from the
"verifiable" batch by a pure predictive-ablation test. Conversely, a metric
might predict outcomes only because it correlates with a more fundamental
cause we didn't measure.

So predictive-ablation actually measures **verifiable AND useful** (the
intersection). For the project's gap framing
(`Outcome = f(V) + g(A) + h(T)`), that intersection *is* what we want for
the AUC decomposition — but for the *taxonomy* of which rubric is at which
level, we want measurability decoupled from predictive utility.

## Proposed two-axis frame

Decouple the two properties and run them as independent axes:

```
                       PREDICTIVE      NON-PREDICTIVE
  MEASURABLE          [matters]       [noise: Oxford commas]
  NOT MEASURABLE      [method gap]    [irrelevant / fully tacit]
```

- The **level** (V / A / T) is determined by **which kind of rater** achieves
  measurability — independent of outcome labels.
- The **predictive-utility axis** is layered on top, on tasks where we have
  outcome labels (code-review, peer-review, news, etc.), and gives the AUC
  decomposition the paper actually claims.

This separation has a nice side benefit: the level can be assigned even on
tasks **without** outcome labels (humor, creative-writing, math, math
elegance, …) — which is exactly the tasks where the L3/L4 mass lives.

## Operationalising the level (no outcome labels needed)

Per rubric, on a sample of ~50–100 datapoints, collect scores from:

- **K=3 code implementations** — three different coding models each
  generate code from the rubric, run on the sample.
- **K=4 LLM judges** — four different models (Llama / Claude / GPT /
  Gemini) given the rubric NL, score the sample.
- **K=4 paraphrased LLM judges** — same model, rubric paraphrased 4 ways
  (Mathur 2020-style robustness check).
- *(optional, expensive)* **K=3 briefed human raters** — for the boundary
  L2 / L3 / L4 cases.

Compute pairwise agreement (Krippendorff's α or ICC(2,k) on ordinal scores):

- `ρ(code, code)` — do code implementations converge?
- `ρ(code, LLM)` — does code agree with LLM-judge?
- `ρ(LLM, LLM)` — do LLM judges converge?
- `ρ(paraphrase, paraphrase)` — is the LLM stable under rubric wording?
- `ρ(human, human)` — do briefed humans converge?

The level falls out from the pattern:

| level | criterion (with thresholds T₁, T₂ to be calibrated) |
|---|---|
| **V** | ρ(code, code) > T₂ **and** ρ(code, LLM) > T₁ |
| **A** | ρ(LLM, LLM) > T₂ and ρ(paraphrase) > T₂ and ρ(code, LLM) < T₁ |
| **T₁** (defensible) | ρ(human, human) > T₂ and ρ(LLM, LLM) < T₂ |
| **T₂** (fully tacit) | nothing reaches the agreement threshold |

This is essentially **Generalizability theory** (Cronbach et al.) under the
hood: decompose score variance into rater / method / item / occasion
components. If `σ²(method)` is large between code and LLM, the rubric is
near the V / A boundary; if `σ²(rater | LLM judge)` is large, the rubric is
poorly articulated.

## Faithfulness checks (does the score measure what the rubric intends?)

Agreement gives you reliability but not validity. Two complementary checks
for whether a measure is *measuring what we think it's measuring*:

### (a) Recoverability (data-driven check)

E2 in the validity-experiments plan. Score N datapoints with the rubric →
hide the rubric → give the labeled datapoints to a model → ask it to infer
the rule → apply to held-out → correlate inferred-score with original-score.

- High held-out correlation → the **score distribution itself reveals the
  rule**, i.e., a reverse-engineerer with no rubric can recover it. Strong
  evidence the measure is well-defined.
- Low correlation → either the rubric is doing something the model can't
  infer from labelled examples (tacit / hidden), or the original scoring
  was noisy.

### (b) Code-interpretation with obfuscation (implementation-driven check)

Alex's proposal, 2026-05-22. Have a coding model generate code for the
rubric, then **strip all self-documenting content** (rename functions to
`a`, `b`, `c`; variables to `x, y, z`; remove docstrings and comments;
anonymise string constants where semantically possible). Then give the
obfuscated code to a fresh LLM and ask: "what rubric is this code trying to
measure?" Compare to the original rubric.

Why obfuscation matters: `is_passive_voice(sentence)` reveals the rubric in
the function name — that tests whether LLMs can read English, not whether
the *implementation is faithful*. With anonymised names, the inferrer has
to read the actual logic (token counts, regex patterns, conditional
structure) and reconstruct the underlying concept. That is a real
faithfulness check.

The four-way truth table for a single rubric:

|                                 | code-interp guesses rubric | code-interp fails |
|---|---|---|
| code-score matches LLM-judge   | **V**: faithful + verifiable     | LLM-judge calibration check needed; could be V where code happens to track |
| code-score doesn't match LLM   | **A** with attempted-but-failed code; code tried but couldn't | **A or T**; code is irrelevant, not even attempting |

### Recoverability + obfuscated-code-interpretation together

These two checks attack faithfulness from different sides:

- Recoverability tests whether the *output label distribution* is
  informative about the underlying rule.
- Code-interpretation tests whether the *implementation* is faithful to the
  underlying rule.

Both are independent of outcome labels. Both are independent of agreement.
Together they triangulate the level assignment — and a rubric that passes
both is the most confidently-verifiable kind. A rubric that fails both but
has high LLM-judge agreement is the most confidently-articulable kind.

## Related work I want to read

In rough order of relevance:

1. **Generalizability theory (G-theory)** — Cronbach, Gleser, Nanda, Rajaratnam,
   *The Dependability of Behavioral Measurements* (1972). Variance
   decomposition into rater × item × method × occasion components.
   Educational measurement's standard framework for exactly the question
   we're asking.

2. **Automated Essay Scoring (AES) literature** — Foltz, Landauer,
   Shermis. Decades of work on rule-based scorers vs latent-semantic
   features vs ML scorers vs (now) LLM judges. Direct parallel to our
   verifiable / articulable / dense decomposition.

3. **Mathur et al. 2020 "Tangled up in BLEU"** — paraphrase-robustness and
   rater-stability for evaluation metrics; methods translate one-to-one to
   our "is this rubric stable under rewording?" test.

4. **Krippendorff's α / Cohen's κ / ICC(2,k)** — standard inter-annotator-
   reliability metrics. NLP convention α > 0.67 ≈ "tentatively reliable",
   α > 0.8 ≈ "reliable." For ordinal scores ICC(2,k) is more honest than κ.

5. **Consensual Assessment Technique (Amabile 1982)** — creativity research
   acceptance that some constructs (creativity, voice) have no rubric, only
   expert consensus. Direct framework for L3/L4: if briefed domain experts
   agree above chance without a written rubric, L3; if they don't, L4.
   **Important caveat (Alex, 2026-05-22):** Amabile does CAT at the
   *product* level -- experts holistically judge whether a poem is creative.
   Our move is at the *metric* level -- different measurement methods
   triangulate the same latent rubric score. The analogy is clean but the
   more *direct* precedents are concurrent / convergent validity and MTMM
   (entries below), where the unit being agreed on is the measurement, not
   the product itself.

5a. **Cronbach 1955, "Construct validity in psychological tests"** --
    multiple measurement methods of the same construct should agree;
    inter-method agreement IS the validity evidence. The frame for treating
    inter-method convergence as the legitimacy criterion for our V/A/T
    assignment.

5b. **Campbell & Fiske 1959, "Convergent and discriminant validation by the
    multitrait-multimethod matrix"** -- the most direct operational fit.
    Lay out N rubrics x M methods, compute the N x N x M x M correlation
    matrix:
    - **same-rubric, different-method** correlations (V-shaped diagonals)
      = convergent validity for the rubric -- this is exactly our V/A
      level signal.
    - **different-rubric, same-method** correlations capture method bias:
      if all rubrics correlate within Llama-judge but not within code, the
      method (Llama) has a systematic bias, not the rubrics.
    - **same-rubric, same-method, different-occasion** = reliability.
    Our 30-rubric x {3 code, 4 LLM, 4 paraphrase} pilot IS an MTMM design;
    we should report the standard Campbell-Fiske diagnostics on it. Cite
    this instead of (or in addition to) Amabile when framing the protocol.

6. **Daston, *Rules: A Short History of What We Live By*** (2022) —
   philosophical thin/thick rules distinction; conceptual grounding but not
   operational (already in `[[project_thin_thick_rules_philosophy]]`).

7. **The "elusive annotation" / "subjective NLP" literature** — papers on
   getting reliable annotations for inherently subjective tasks (humour,
   story quality). Pavlick & Kwiatkowski 2019 "Inherent Disagreements in
   Human Textual Inferences" is a key entry point.

## My honest take on what to actually run first

A **scaled-down G-theory study** on ~30 rubrics × 50 datapoints × 4 LLM
judges × 3 code implementations. ~6,000 LLM calls + 90 code generations —
totally doable on sk3. Output: per-rubric variance-decomposition vector
(σ² for each source). That vector *is* the V / A / T characterization, and
on the subset where outcome labels exist, validate the level assignment
against held-out outcome AUCs (E5-style).

This replaces the gpt-5-mini self-report with an empirical 4-coordinate
profile per rubric. The paper deliverable becomes "introspected L1–L4 was
unstable across prompts; we replaced it with a variance-decomposition profile
validated against outcome AUC." Stronger move.

Then layer the **faithfulness checks** (recoverability + obfuscated-code-
interpretation) on a smaller sample to confirm the variance profile means
what we think it means.

## Open questions / sensitivity

- **Rater independence.** Are Llama / Claude / GPT / Gemini *independent*
  samples from "the population of articulable LLM judges"? They share
  training data and may correlate spuriously. Same for code generators.
  Worth a small inter-model-family correlation sanity check.
- **Threshold calibration (T₁, T₂).** ICC > 0.7 ≈ reliable; tighter
  numbers need a held-out calibration set.
- **Sample size per rubric.** 50 datapoints is the minimum where ICC is
  stable. May want 100 for the boundary cases.
- **What counts as "obfuscation done well"** for code-interpretation. Regex
  patterns themselves leak intent (`/\bbe\s+\w+ed\b/` is recognisably about
  passive voice). Hard to fully anonymise without breaking the code. Pilot
  this on 10 rubrics first.
- **Outcome-axis confound.** On tasks without clean outcome labels (humor,
  creative-writing) we can only do the level assignment, not the predictive
  AUC decomposition. The paper has to acknowledge that.

## Schedule — Sat 2026-05-23 → Fri 2026-05-29

Anchored on two external deadlines: **Thu May 28 meeting with Noah Goodman**
and **Fri May 29 presentation in Sanmi Koyejo's group.** Alex is travelling
Sat–Wed for a wedding (sparse work windows, some plane time). Claude (me)
does most of the autonomous implementation + runs; Alex reviews + frames.

Goal for Thu: have at least *one* task end-to-end with real variance-
decomposition numbers, plus the faithfulness-check pilot results, so the
Noah conversation is "here's what we measured" not "here's what we plan."

### Sat May 23 — implementation day (Claude solo while Alex flies)

- [ ] Code-gen pipeline: rubric → Python code via Llama-3.3-70B and one
  coder model (DeepSeek-Coder or Qwen-Coder if available on sk3).
  Includes execution harness on a datapoint, score normalization to [0,1].
- [ ] LLM-judge pipeline: 4 judges (Llama-70B, Qwen-122B if available, plus
  via OpenRouter Claude + Gemini), rubric + datapoint → score [0,1].
- [ ] Paraphrase pipeline: rubric → 4 NL paraphrases (one model).
- [ ] Agreement-statistics module: ICC(2,k), Krippendorff's α, simple
  Pearson — output the per-rubric ρ-matrix table.
- [ ] Data prep: select 30 R1 rules from the locked clustering, stratified
  to cover the existing v7 articulability labels (rough V/A/T mix). Select
  50 datapoints from code-review PR corpus (cleanest labels, simplest
  datapoint structure).
- [ ] Smoke test E2E on 5 rules × 10 datapoints. Verify no obvious bugs.

### Sun May 24 — full code-review G-theory pilot

- [ ] Run: 30 rules × 50 datapoints × {3 code impls, 4 LLM judges,
  4 paraphrase variants}. ~7K LLM calls + 90 code generations + 4.5K code
  executions. Should fit in a single sk3 GPU session (Llama-70B, ~30–60
  min) plus OpenRouter calls for the non-Llama judges.
- [ ] Compute the per-rule ρ-matrix + variance decomposition.
- [ ] Produce: a 30-row table of {rule, σ²(method), σ²(rater | LLM),
  σ²(rater | code), inferred-level, v7-self-reported-level} for cross-check.

### Mon May 25 — faithfulness check pilots

- [ ] 10-rule code-interpretation pilot: pick 10 rules from the 30 (mix V/
  A/T per the Sunday inferred-level). Generate code, obfuscate (strip
  function/variable names, comments, docstrings; replace where possible),
  give to a fresh LLM, ask it to guess the underlying rubric. Manually
  inspect each output for leakage (especially regex / string-literal
  leaks) and rate the inferred-vs-actual similarity.
- [ ] 10-rule recoverability pilot: score 50 datapoints with each rubric,
  hide rubric, give labelled datapoints to a fresh LLM, ask it to infer
  the rule and apply to held-out 50. Correlate inferred-score with
  original-score per rule.
- [ ] Decision: is obfuscation working well enough to use code-interp at
  scale? Pilot result tells us.

### Tue May 26 — second task + cross-task synthesis

- [ ] Run G-theory pilot on a second task. Two candidates:
  - **peer-review**: clean outcome labels, complementary domain to code.
  - **creative-writing**: no outcome labels, but where the L4 mass lives;
    most informative for the level-only branch of the protocol.
  Default to creative-writing unless Alex says otherwise — it's where the
  decomposition is most novel.
- [ ] Cross-task comparison: does the rule-level variance profile transfer?
  Or are levels task-specific? (Hypothesis: levels are largely
  rule-intrinsic but a rule can shift levels between tasks if the
  surrounding evidence base differs.)

### Wed May 27 — synthesis + writeup (Alex travels back)

- [ ] Claude: write `notes/2026-05-27__validity-pilot-results.md` with:
  the 30-rule variance table, the pilot results, key examples (rules that
  flipped levels under empirical test vs introspection), recoverability +
  code-interp scatter plots, the V/A/T mass per task.
- [ ] Alex: review on the plane / Wednesday evening. Mark questions for
  Noah.

### Thu May 28 — Noah meeting

- [ ] Morning: Alex incorporates pilot results into talking points
  (Claude available for last-minute fixes / re-runs).
- [ ] Meeting: real-time iteration on the framework based on Noah's
  feedback.
- [ ] Post-meeting: Claude logs Noah's feedback to
  `notes/2026-05-28__noah-feedback.md`; we adjust the framework if needed.

### Fri May 29 — Sanmi group presentation

- [ ] Slides built from the pilot results + Noah's feedback.
- [ ] Sanity check the cross-task comparison numbers.
- [ ] Present.

### Risk register / fallbacks

- **GPU contention on sk3** — if 70B is unavailable for hours-long stretches,
  fall back to single-judge runs (just Llama-70B, no Qwen-122B). The
  4-judge G-theory weakens but the protocol still produces a number.
- **OpenRouter cost / rate-limits** for non-Llama judges — cap to 30 rules
  × 50 datapoints = 1500 calls per judge per run. Should be < $5 per judge
  at GPT-4o-mini / Claude-Haiku rates.
- **Obfuscation pilot finds the test is gameable** — drop code-interp from
  the framework, rely on recoverability + agreement decomposition alone.
  Worse but still defensible.
- **Time slip on travel days** — Mon and Tue are the most fungible. If
  Sunday's G-theory pilot reveals bugs, slip the second-task run to Tue
  and skip cross-task synthesis. The Thu deliverable still has
  one task's variance decomposition + the faithfulness pilots.
- **The variance decomposition doesn't separate V from A cleanly** — fall
  back to the simpler "predictive ablation" framing for the Noah meeting;
  use the agreement results as evidence for the *taxonomy*, not the
  decomposition.

### Owner conventions

- **Claude**: implementation, sk3 runs, agreement statistics, draft writeup,
  re-runs on demand.
- **Alex**: rule selection final call, datapoint corpus final call,
  thresholds / cut-offs interpretation, talk framing.

## Relationship to existing notes

- `2026-05-16__validity-experiments-plan.md` — E1–E6 experimental plan.
  Most relevant: **E1 (Consistency)**, **E2 (Recoverability)**, **E3
  (Rearticulation)**, **E5 (code-vs-rubric predictive)**. This document is
  the higher-level framing; the E1–E6 are the concrete experiments that
  populate the framework above.
- `2026-05-14__metric-taxonomy-and-two-axis-setup.md` — L1–L4 definitions
  and the original four-axis classification setup.
- `[[project_verifiability_explainability_gaps]]` (memory) — the gap-
  framework `Outcome = f(V) + g(A) + h(T)` this all serves.


## References (auto-verified BibTeX, 2026-06-15)

> Extracted from this document and web-verified + independently audited by an automated fact-check pass (search → fetch → resolvable id; attributed claim checked against the located paper). 9 entries. Real located works; not hand-checked. See "needs manual review" for 0 contradicted-claim and 0 unlocatable/rejected items.

```bibtex
@article{amabile1982social,
  title={Social psychology of creativity: A consensual assessment technique},
  author={Amabile, Teresa M.},
  journal={Journal of Personality and Social Psychology},
  volume={43},
  number={5},
  pages={997--1013},
  year={1982},
  doi={10.1037/0022-3514.43.5.997}
}

@article{campbell1959convergent,
  title={Convergent and discriminant validation by the multitrait-multimethod matrix},
  author={Campbell, Donald T. and Fiske, Donald W.},
  journal={Psychological Bulletin},
  volume={56},
  number={2},
  pages={81--105},
  year={1959},
  doi={10.1037/h0046016}
}

@article{cronbach1955construct,
  title={Construct validity in psychological tests},
  author={Cronbach, Lee J. and Meehl, Paul E.},
  journal={Psychological Bulletin},
  volume={52},
  number={4},
  pages={281--302},
  year={1955},
  doi={10.1037/h0040957}
}

@book{cronbach1972dependability,
  author    = {Lee J. Cronbach and Goldine C. Gleser and Harinder Nanda and Nageswari Rajaratnam},
  title     = {The Dependability of Behavioral Measurements: Theory of Generalizability for Scores and Profiles},
  publisher = {John Wiley \& Sons},
  address   = {New York},
  year      = {1972},
  isbn      = {9780471188506}
}

@book{daston2022rules,
  title={Rules: A Short History of What We Live By},
  author={Daston, Lorraine},
  series={The Lawrence Stone Lectures},
  year={2022},
  publisher={Princeton University Press},
  isbn={9780691156989}
}

@book{krippendorff2004content,
  title={Content Analysis: An Introduction to Its Methodology},
  author={Krippendorff, Klaus},
  edition={2nd},
  year={2004},
  publisher={Sage Publications},
  isbn={9780761915454}
}

@inproceedings{mathur2020tangled,
  title={Tangled up in {BLEU}: Reevaluating the Evaluation of Automatic Machine Translation Evaluation Metrics},
  author={Mathur, Nitika and Baldwin, Timothy and Cohn, Trevor},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2020},
  eprint={2006.06264},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{pavlick2019inherent,
  title={Inherent Disagreements in Human Textual Inferences},
  author={Pavlick, Ellie and Kwiatkowski, Tom},
  journal={Transactions of the Association for Computational Linguistics},
  volume={7},
  pages={677--694},
  year={2019},
  doi={10.1162/tacl_a_00293}
}

@book{shermis2013handbook,
  editor    = {Mark D. Shermis and Jill Burstein},
  title     = {Handbook of Automated Essay Evaluation: Current Applications and New Directions},
  publisher = {Routledge},
  address   = {New York},
  year      = {2013},
  isbn      = {9780415810968}
}

```

### Citations needing manual review

**Partial claim-match (3)** — spot-check exact numbers/wording:

- `cronbach1955construct`; `mathur2020tangled`; `shermis2013handbook`

