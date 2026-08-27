# Metric-seam pilot v0 — press releases, 10 aspects × 250 items (results)

*2026-07-01. First empirical run of the seam survey (proposal:
`2026-07-01__metric-seam-proposal.md`; scripts: `methods/metric_seam/pilot/`). This is the PRE-GATE
survey (description-compiled programs vs one LLM channel), NOT the MIGRATE gate — code channel here is
a LOWER bound on codability.*

## Setup
- **LLM channel:** Gemma-4-31B-it, offline batch vLLM (gemma4 env, sk3 GPU 1 only, released after),
  greedy, 0–10/NA single-line readout, 2,500 verdicts in ~8 min after ~2 min engine init. 1 run;
  earlier crash = URL-dense text >6144 tok → fix: TRUNC 8000 chars + max_model_len 10240.
- **Code channel:** existing `codegen_claude` flavors v0/v1/v2 per aspect (description-compiled,
  blind to the LLM channel), CPU. 28/30 ran; a86_v1/v2 have mangled-quote SyntaxErrors (recorded as
  failed rungs).
- **Items:** 250 sampled (seed 0, len≥1000) from `runs/validity_full/v2/press_releases/datapoints.json`.
- Outputs: `outputs/metric_seam_pilot/` (items, prompts, results.jsonl, code_scores.json,
  seam_table.json, adjudicate_a80/a86.json, adjudication_summary.json).

## Seam table (Spearman ρ code-vs-LLM; κ at median split in seam_table.json)

| aspect | name | expected | NA | LLM sd | ρ v0 | ρ v1 | ρ v2 | pre-gate verdict |
|---|---|---|---|---|---|---|---|---|
| a79 | Wire-ready format | V | 0 | 1.03 (compressed 0–4) | .58 | .57 | .48 | boundary |
| a80 | Media contacts | V | 0 | 3.08 (bimodal 0/10) | .55 | .54 | .52 | boundary |
| a110 | Boilerplate/metadata | V | 1 | 3.66 | .48 | .57 | .53 | boundary |
| a100 | Lede 5Ws | boundary | 1 | 4.05 | .26 | .16 | .21 | A-layer |
| a101 | Inverted pyramid | boundary | 0 | 3.94 | .22 | .04 | .41 | boundary |
| a86 | Quote quality | boundary | 3 | 3.09 | **.63** | broke | broke | "codable-now" → **REFUTED by adjudication** |
| a105 | Plain language | boundary | 8 | 2.84 | .16 | .31 | .22 | A-layer |
| a118 | Timeliness signaled | boundary | 0 | 4.81 (bimodal 0/10) | .27 | .40 | .34 | boundary |
| a117 | Newsworthiness/hook | A | 0 | 3.87 | .37 | .34 | .38 | boundary-ish (proxies) |
| a73 | Empathy/sensitivity | A | **149** | 2.77 | .11 | .25 | .20 | A-layer; NA-heavy = narrow applicability |

No judge collapse (all sd > 1, modal ≤ 0.8); a80/a118 near-binary (presence-style criteria).

## Adjudication of disagreement cells (cross-family: Claude; n=10 balanced cells each)

- **a80 (contacts): llm_right 8 / code_right 1 / both 1.** Program fires on page chrome (nav
  "Contact Us", subscription emails) on NON-releases; misses real contact blocks with OCR-dropped `@`,
  spaced phone digits, email-free named blocks. → residual is **input thickness + document scoping**,
  not rule thickness (thin-rule/thick-input quadrant; v_struct sandwich applies).
- **a86 (quote quality): llm_right 10 / 0 / 0.** Presence-proxy fails both directions (scare
  quotes/JSON artifacts vs mojibake curly quotes/'said'-free attribution). **The ρ=.63 top-of-table
  score would have MIGRATED this criterion on correlation alone; adjudication kills it** — the pilot's
  cleanest evidence that the gate needs direction-adjudication/CF-validity, not just κ.

## Readings (per §3.4)

1. **On this corpus the binding residual for "structural" criteria is input normalization + scoping**
   (scraper chrome, mojibake, truncated blocks, non-release contamination — blogs/news articles inside
   the corpus), NOT open-texture rules. Echoes patents "parsing-dominated" + legal v_struct. Expect
   a79/a80/a110 to migrate in the real MIGRATE loop (channel-targeted codegen fixes normalization
   easily); a86 splits presence (codable) from quality (A-residual).
2. **Description-compiled ≠ channel-targeted**: current programs never saw the judge's verdicts; ρ ≈
   .5–.6 for structural aspects is the *floor*, i.e. pre-gate ρ understates the V share.
3. **A-layer calls stable**: lede-5Ws quality, plain-language, empathy resisted all rungs (ρ ≤ .31) —
   consistent with §3.4's expected-A row; a73's 60% NA is the applicability guard, live in the wild.
4. **Corpus action item**: add a document-scoping channel ("is this actually a press release?") before
   any per-criterion scoring; contamination inflates apparent input-thickness for every aspect.

## Caveats
Single greedy pass (no T_train reliability), one judge family for the channel, median-split κ crude,
n=10 adjudications/aspect by one adjudicator, broken a86 programs limit its ladder to one rung.

## Next (maps to proposal E-S1/E-S2)
- Channel-targeted MIGRATE attempt on a80 (codegen against (x, verdict) pairs with normalization) —
  prediction: passes the gate; the seam moves.
- CF-validity probe for a86 (perturb quotes only) to formalize the adjudication kill.
- Scoping channel + re-run; add second judge pass for reliability.

---

# v1 UPDATE (2026-07-02): fixes, robustness, and the hybrid round — RUN

*All "next steps" above executed same session. Scripts: `methods/metric_seam/pilot/{build_v1,
gemma_score_v1,run_code_flavors_v1,analyze_v1}.py`, `methods/metric_seam/hybrids/` (ops, harness,
improver machinery, programs). Data: `outputs/metric_seam_pilot/v1/`.*

## Fixes over v0 (errors the run exposed)
- **F1 (real bug): apples-to-apples violation** — v0 code channel scored FULL text, LLM saw first-8k.
  v1: one canonical text (head 5000 + tail 2500 — footers survive) for BOTH channels.
- F2 two judge passes (paraphrased template) → per-aspect reliability 0.79–0.96; exception
  **a118 timeliness rel=0.59** (form-unstable channel — bimodal criterion sensitive to wording).
- F3 **scoping channel: only 105/250 items are press releases** (contamination confirmed, mean
  scope 4.5/10).
- F4 CF probe quantified: injected boilerplate-CEO quote → v0 presence-code Δ **+0.167** vs judge
  Δ **+0.047**.

## v1 seam table (canonical text; ρ full / best-scoped)
Structural criteria RISE when scoped (a80 .63→.76, a110 .55→.76, a86 .71→.72); soft criteria FALL
(lede-5Ws .25→**−0.01**, newsworthiness .40→.20, empathy .28→.32-ish) — their full-set code signal
was mostly the shared "is-this-even-a-release" variance. **Scoping deconfounds the seam table.**

## Hybrid round (mixed code + LLM-field + tool ops; reflective improvers, gated)
Contract: `LLM_FIELDS` (≤2 thick-input extractors, Gemma-batched) + `score(text, extracted, ops)`;
ops = normalize/dates/sent_stats (computation) + TF-IDF corpus retrieval (evidence).
Gate: G1 test-ρ ≥ max(baseline+0.10, 0.60) on held-out 100; G3 (a86) CF-Δ ≤ judge-Δ+0.05.

| aspect | code-only test ρ | +LLM fields test ρ (scoped) | baseline | gate |
|---|---|---|---|---|
| a86 quote quality | .868 | **.895** (.656) | .720 | **G1 PASS, G3 PASS (Δ .004 vs judge .047)** |
| a110 boilerplate | .635 | **.743** (.829) | .631 | **G1 PASS** |
| a80 media contacts | .551 | .709 (**.835**) | .634 | fail by .025 full-set; scoped ≈ +.08 over scoped baseline |
| a105 plain language | .396 | .535 (.398) | .421 | fail — **A-layer confirmed** (its author predicted the shrinkage) |

- **a86 is the headline**: the criterion refuted as a presence-proxy in v0 now has a certified
  mixed implementation — quote-presence in code, quote-QUALITY via specificity predicate over
  LLM-extracted quote spans, CF-flat by construction. Thick-input/thin-rule, demonstrated.

**BOOTSTRAP CORRECTION (2026-07-02, B=2000 item resamples of the held-out test; this supersedes
the point-estimate gate verdicts above):**

| aspect | ρ_hyb [95% CI] | P(gate G1) | P(beats baseline) | corrected verdict |
|---|---|---|---|---|
| a86 | .892 [.852, .928] | **0.90** | **1.00** | beats-baseline CERTIFIED; joint gate at 90% (marginal at δ=.05) |
| a110 | .742 [.607, .840] | 0.59 | 0.95 | gate UNRESOLVED at n=100 (earlier "PASS" overclaimed) |
| a80 | .704 [.594, .787] | 0.31 | 0.92 | gate fail UNRESOLVED — likely better than baseline, strict gate undecidable |
| a105 | .530 [.351, .692] | 0.18 | 0.85 | consistent with A-layer; even superiority only 85% |

a86 judge CF delta: +.047 [+.027, +.067] (n=30) — the judge's rise on injected quotes is real but
small; hybrid Δ=.004 sits below the CI's lower edge. Point-estimate gate margins of ±.01–.03 at
n_test=100 are inside Spearman sampling noise (SE≈.05–.07) — all future gate decisions need this
bootstrap form (Rung-3 discipline), or n_test scaled up (~2,500 more Gemma verdicts ≈ 10 min GPU).
- **Round-1 reflective refinement (a80): a cautionary success.** 12/12 train feedback cells
  improved, train ρ held, but full-set test ρ DROPPED .709→.579 (scoped unchanged .82) — the
  train-selected non-release damping didn't transfer. The held-out gate caught it; h0 stays HEAD.
  Lesson: single-round feedback on 12 cells overfits the out-of-scope tail; the non-release
  handling belongs in the SCOPING channel (composition), not inside each criterion channel.
- LLM-field extraction is itself applicability data: media_contact non-empty on only 59/250 docs.

## Clean claims after v1
1. Mixed code/prompt implementations beat both pure forms: fields add +.05–.16 test ρ over
   code-only on 3/4 aspects; 2/4 aspects pass the strict gate as hybrids.
2. The MIGRATE gate needs CF/adjudication, not κ alone (v0 a86 refutation + v1 CF numbers).
3. Scoping = deconfounding for the seam; without it, soft criteria borrow code signal and
   structural criteria are under-credited.
4. Plain language resists hybrid codification even with a jargon-extraction LLM field (scoped
   .398) — the A-layer call survives its strongest challenge yet.
5. Held-out gating catches reflective-loop overfitting within one round (a80 h1).

Open for E-S2 proper (user sign-off): define the gate on the scoped subset (needs scoped baseline
on the same split); scoping as its own channel composed before criterion channels; second
adjudication round on a110/a80 disagreements; math task next (sympy computation ops).

## Mixedness measurement (2026-07-02, `ablate_mixedness.py` → `mixedness_report.json`)

Ablation lattice {LLM fields on/off} × {ops full/null} on held-out test; Shapley shares of test ρ
(base = code core on raw text); per-item touch = fraction of test items a medium moves >0.02:

| aspect | code core | LLM share | tool share | LLM-touch | seam string |
|---|---|---|---|---|---|
| a80 contacts | .55 | **+.16** | .00 | 47% | C→(T_c)→L→C, LLM-dominant |
| a86 quotes | .60 | +.14 | **+.17** | 36% | C→T_c→L→C, three-way mixed (normalize recovers mojibake quotes: tools-only already .87) |
| a110 boilerplate | .64 | **+.11** | −.00 | 24% | C→L→C; retrieval evidence op net-zero on test |
| a105 plain lang | .37 | +.14 | +.04 | **81%** | LLM-dependent AND still sub-gate — the A-layer signature |

Readings: (1) mixedness is a measurable spectrum, not a binary — LLM-touch share orders the aspects
.24→.36→.47→.81 tracking increasing A-ness; (2) a86's fidelity is mostly COMPUTATION-op work
(mojibake normalize), i.e. the "codable" verdict was blocked by input encoding, not rules —
thick-input diagnosis confirmed causally; (3) a110's evidence op (TF-IDF retrieval) is net-zero →
prune (no silent op-bloat); (4) all four dataflows are C→L→C sandwiches (Daston thick-thin-thick,
literal). Developmental order from lineage: judge(prompt) → MIGRATE(code+LLM-fields, one step) →
round-1 code refinement (rejected by held-out gate). Static shares: 170-215 code LOC vs 32-62
LLM-instruction tokens vs 1-3 ops call sites per channel.

---

# OVERNIGHT EXPANSION (2026-07-02 night) — 4-task certified survey, in flight

User goal: expansive, honest, statistically/theoretically grounded seam measurement; ≥40 metrics
per task on press_releases, math, code_review, patents.

**Done before overnight run:** certificates library (`methods/metric_seam/certificates.py`) with
14/14 green planted tests (`tests_certificates.py`); full theory folded into proposal §5bis
(attenuation ceiling S1, codability bracket S2, matroid-U₂ S3, tightening S4, op-value S5).

**In flight (sk3 GPU 1, sequential queue `outputs/metric_seam_pilot/queue/`):**
1. PR wave-2: 20 aspects × 250 × 2 passes (judge batch finishing) → then 20 hybrid improver
   agents + field batch + certified eval.
2. Queue jobs: math (40 aspects, 20,250 prompts) → code_review (40) → patents (40) → PR wave-3
   (10 more aspects → PR reaches 40). Each = 2 judge passes + scope channel; code flavors already
   run locally (math 119/120, cr 120/120, patents 112/120, wave3 29/30; breaks recorded).
3. Analyses: `analyze_v2.py` pattern with certificates (reliability, ceilings,
   ceiling-normalized ρ, scoped ρ) per task.

**Deliberately deferred for user sign-off:** hybrid evolution for math/code_review/patents
(120+ improver agents = cost decision); CF probes + adjudication rounds for new tasks
(per-criterion design); scoped-gate definition.

## MATH survey (overnight, 40 aspects × 250 items × 2 passes; `tasks/math/seam_table.json`)

- **Corpus is clean**: 237/250 in-scope (mean scope 9.5) — no contamination excuse, unlike PR.
- **Description-compiled code is nearly useless on math**: best pre-gate ρ = .41 (visuals/examples);
  citations .39, equation-numbering .38, axiomatic-presentation .33 — and the mass of the
  distribution sits at ρ ≤ .3 with several ≈0 or negative INCLUDING mechanical-sounding criteria
  (notation consistency −.01, microtypography −.03, inline-$-delimiters .17, notation-vs-code .11).
- Judge reliability mid (0.36–0.93, mostly .6–.8; ceilings .73–.98) — Gemma is less stable on math
  criteria than PR ones; 4/40 aspects degenerate (constant passes / all-NA).
- ρ/ceiling < .3 for ~85% of aspects ⇒ at the description-compiled tier **math is A-layer-dominant
  while PR is mixed** — the cross-task frontier difference is large and in the predicted direction
  EXCEPT that even LaTeX-mechanical criteria resist description-compiled regex (candidate
  explanations: the checkable predicate needs parsing/AST (computation ops), or judge scores
  holistic quality the mechanical check only weakly proxies — hybrid round would separate these).
- Applicability channels visibly working: existential-proofs NA=101/250, citations NA=74/250.

## CODE_REVIEW survey (overnight, 40 aspects; `tasks/code_review/seam_table.json`)
- Scope clean (246/250). **18/40 aspects DEGENERATE by NA** — stride-sampled bank includes
  narrow-scope criteria (K8s hardening, PHP PSR, Java GC, stylelint); applicability channel
  correctly NAs them. Finding: the code_review R2 bank has a fat tail of narrow-applicability
  metrics; NA-rate is an applicability profile of the bank itself.
- Usable 22: A-dominant at description-compiled tier (testability .62/ceil .64 is the outlier;
  code-smells .42, component-catalog .42; median ρ/ceil ≈ .2).
- **Structural cause: the datapoints are title+comments WITHOUT the diff** — the judged object is
  absent from X. This is input-channel starvation = the evidence-op diagnosis at task scale
  (matches project_code_review_verifiability_plan's diff-enriched re-score TODO).

## PATENTS survey (overnight, 40 aspects; `tasks/patents/seam_table.json`)
- Corpus fully in-scope (250/250, mean 10.0). 16/40 degenerate (NA-heavy: design-drawing,
  sequence-listing niches) + several unreliable channels (rel1 ≤ 0 — judge anti-correlates across
  templates, pure form-noise; flagged, ceiling undefined).
- **Description-compiled code carries ~zero seam signal on patents**: usable-aspect ρ mostly ∈
  [−.3, +.2]; only amendment-practice .58 (n=49, NA=201) and §112(f)-trigger .34 positive.
- Reading: patents' binding criteria (novelty/obviousness/eligibility) reference EXTERNAL evidence
  (prior art) — evidence-op-dominant by construction, exactly the cross-task prediction (E-S3) and
  consistent with the parsing-dominated §102 memory. Confirms the op-type taxonomy at task scale:
  PR = normalization (computation), math = parsing/AST (computation), code_review = missing diff
  (evidence), patents = prior-art retrieval (evidence).

## OVERNIGHT COMPLETE (2026-07-02 ~04:30): 160 metrics surveyed, 20 hybrids certified

**Survey coverage (goal ≥40/task): press_releases 40 (10 v0/v1 + 20 v2 + 10 v3), math 40,
code_review 40, patents 40 — each with 2-pass reliability, attenuation ceilings, scope channel,
description-compiled 3-rung code baselines.** GPU total ≈ 3.5h on one B200 (sequential queue).

**Wave-2 hybrid fleet (20 aspects, held-out n=100, B=2000; `v2/hybrid_eval_v2.json`):**
- Strong gate passes P(gate)≥.95: a76 lede (1.0), a42 audience (.989), a87 tone (.998),
  a65 context (.976), a66 data-support (.966), a103 style (.958), a97 checkable-facts (.95).
- 16/20 beat their description-compiled baselines at P≥.96. Clean null: a111 CTAs (P>base .55).
  a115 unresolved (.83). a119 originality: gate numbers are ARTIFACTS (constant baseline → NaN);
  its own CI [.754,.892] clears the .60 floor decisively; used the retrieval EVIDENCE op.
- Median test ρ .67–.70; ρ/ceiling .45–.86. Expected-A criteria reached high fidelity via LLM
  fields (tone .806, audience .732) — mixedness shifted toward the LLM medium, NOT a refutation of
  A-ness; ablation Shapley pending to quantify shares.
- PR wave-3 (10 more): description-tier ρ .01–.42 — consistent with wave-1/2 pre-gate floor.

**Cross-task op-type map (the headline):** PR = computation (normalization) + mixed seam;
math = A-dominant, mechanical criteria need parsing/AST computation ops; code_review = evidence-
starved (no diff in X; 18/40 NA-degenerate = bank applicability profile); patents = evidence-
dominant (prior art external; near-zero code signal, several unreliable judge channels flagged).

**Open (user sign-off):** mixedness ablation on the 20 new hybrids; hybrid evolution for
math/code_review/patents (needs diff-enrichment for CR and retrieval corpus for patents to be
meaningful); consolidated cross-task figure; second-family judge replication.

---

## CODE_REVIEW re-survey on FULL-PR corpora (2026-07-02, corpus 1 landed)

**Setup.** Same 40 code_review aspects, same protocol (2 judge passes + scope, Gemma-4-31B),
but text = PR title + description + **unified diff** (300 PRs from
`code_review_dense_4096tok_with_reasoning.csv.gz`, diff-bearing rows). Directly A/B-able against
the comments-only survey. Code baseline = the code_review codegen programs (v0/v1/v2 flavors) via
symlinked codegen dir. Corpus 2 (`code_competition`, 300 submissions w/ verdict metadata) still
scoring.

**Two orthogonal findings — the "evidence-starved" story splits in half:**

1. **The diff rescues the JUDGE's degeneracy — input starvation was real.** NA-degenerate aspects
   dropped **20/40 → 8/40**. The mechanical criteria that were unmeasurable on comments (import
   hygiene, control-flow clarity, overflow safety, secure deserialization, Python exception
   handling, …) become judge-measurable once the code is in X (rel₁ 0.53–0.96). So yes — those 12
   metrics were degenerate purely because the diff wasn't in the input channel. ✓ prediction held.

2. **But the code BASELINE got WORSE, not better** (median best-flavor ρ **0.225 → 0.068**; mean
   0.227 → 0.095). The mechanical metrics I expected code to nail with the diff present are
   near-zero or negative: import hygiene ρ=−0.11, secure-deserialize −0.17, control-flow 0.06,
   overflow 0.08. Best diff-corpus aspect is PR-hygiene/metadata (0.42) — a title/label property,
   not a code property.

**Interpretation (with the load-bearing caveat).** Evidence-presence ≠ verifiability. Having the
code in front of you is *necessary but not sufficient* for a program to score it. Two candidates,
not yet separated:
  (a) **OOD-programs confound [must flag]:** the codegen programs were written for the
      *comments-only* representation (`score(text)` where text = review threads); run on diff text
      they pattern-match on things no longer present, so this baseline is a LOWER bound and an
      underestimate. (Explains why comments-era "design-for-testability" 0.62 → diff 0.23: the
      programs keyed on reviewer *discussion*, which the diff corpus dilutes — construct
      substitution, again.)
  (b) **Genuine A-layer:** reading "is this deserialization secure / is this control flow clear"
      from a raw diff may simply be tacit for a description-compiled Python program even when the
      evidence is present — exactly the articulability gap, now localized to the code executor.
Separating (a) from (b) needs **diff-native codegen** (regenerate programs against the diff
representation) and/or the **reconstruction R** (recovers the rule from the judge's own behavior,
representation-agnostic). Both are follow-ups; the current diff seam table stands as: *judge
degeneracy halved, code-reproducibility unresolved and bounded below by an OOD baseline.*

**Presentation line:** "evidence-starved" was correct but incomplete — the input starvation was
real (judge could not apply 12/40 criteria without the code), yet putting the code in the input
does NOT by itself make those criteria code-verifiable. Verifiability is a property of the
(criterion, executor, representation) triple, not of evidence-presence alone.

## CODE_COMPETITION corpus + the FIRST EXTERNAL GROUND-TRUTH ANCHOR (2026-07-02)

**Corpus 2** = 300 competitive-programming submissions (competition_unified), scored on the same
40 code_review aspects. More degeneracy (~18/40) — expected: PR-oriented aspects (deployment, HTTP
API, version-control literacy, PR-hygiene) don't apply to a self-contained contest solution. The
aspects that DO apply give a clean code signal (testability 0.42, comment-style 0.31,
one-stmt-per-line 0.30) AND consistent NEGATIVES (import hygiene −0.33, source-org −0.29,
secure-deserialize −0.26) — the code program's notion of "good" is anti-correlated with the judge's
in this domain.

**The anchor (verdict_anchor.json).** Competition submissions carry an execution verdict
(AC/WA/TLE/RE/CE) the judge never saw — the project's FIRST external ground truth (everything else
certifies reproduction of the judge, not correctness). 255 graded, 47% AC. For each measurable
quality aspect we correlate judge-score and best-code-program against is_correct(AC):

**Headline: judged "code quality" is ~orthogonal to actual correctness, and the single strongest
relationship is NEGATIVE.**
- median |judge~AC| = **0.103** — quality judgments barely predict whether the code runs.
- most negative: **a180 "one statement per line / clean formatting" judge~AC = −0.437** — solutions
  the LLM rates as cleanly formatted are substantially LESS likely to be correct. Competitive
  programming inverts SWE aesthetics: terse/golfed/dense code wins; clean readable code correlates
  with slower/wrong/novice submissions.
- most positive: a36 code-smells +0.285, a153 hot-path-performance +0.257 (perf hygiene ≈ avoiding
  TLE — the one quality axis that IS about correctness here).
- **9 aspects where judge~AC and code~AC disagree in SIGN** (a63: judge −0.24 vs code +0.34;
  a207, a171, a216, …) — the LLM judge and the description-compiled program point opposite
  directions on whether the property even relates to correctness.

**Why this is the money slide.**
1. First anchored measurement in the project — verdict = did it actually pass, not "does a bigger
   model like it." Directly answers the standing "how can we trust the judge?" worry: on this
   corpus the judge's quality scores are NOT a correctness oracle, and we can prove it.
2. Empirically validates the V/A/Taste split: correctness (V = verdict) and judged quality (A) are
   near-orthogonal, sometimes anti-correlated — not collapsible. A faithful quality-judge is still
   not a correctness judge.
3. Construct-inversion: the judge carries generic SWE aesthetic priors that MISMATCH the domain
   objective (correctness-under-time-pressure), so normal quality signals flip sign. This is
   exactly the executor/construct-relativity the theory warns about, caught with ground truth.

Caveat: the code programs are code_review-comments-era (OOD on both new corpora), so code~AC is a
lower bound; and these are quality-scorers not correctness-predictors, so neither channel is
expected to be a strong AC predictor — the FINDING is precisely that judged quality ≠ correctness.

---

## SKEPTICAL AUDIT of the 2026-07-02 code findings (2026-07-03, audit_competition_claims.py) — RETRACTIONS

User pushed back; line-by-line re-derivation with confound checks. Verdicts:

**RETRACTED — the a180 "clean formatting anti-correlates with correctness / competitive
programming inverts SWE aesthetics" headline.** The verdict×language composition is maximally
entangled: ALL 62 C++ submissions in the graded sample are AC; ALL 135 non-AC items are
Python-on-Codeforces (harvesting artifact: other platforms only publish accepted solutions).
Within Python — the only stratum with verdict variance — a180 judge~AC = **+0.039 (perm p=.62)**.
The pooled −0.437 was pure language-composition (judge scores Python formatting high; Python
carries all the failures). a135's sign flips too (−0.189 pooled → +0.151 within-Python). This is
the SAME Simpson trap as the F2P split-signal memo — caught in our own anchor analysis.

**RETRACTED — "9 aspects where judge~AC and code~AC disagree in sign."** Zero survive requiring
both sides Bonferroni-significant (17 aspects, α=.0029, |r|≳.185 at n=255). Raw sign flips among
noise-level correlations.

**CORRECTED — the orthogonality claim.** Sample AC-rate 47% is BY CONSTRUCTION (stratified);
population ≈69%. Within-Python the surviving relationships are weak POSITIVE: a36 code-smells
+0.204 (p=.007), a0 control-flow +0.195 (p=.0055), a9 testability +0.186 (p=.010), a153 perf
+0.173 (p=.017, drops from .257 pooled once TLE excluded — partially TLE-driven as suspected);
none Bonferroni-proof. Honest summary: *judged quality is weakly positively related to
correctness where verdict variance exists at all; no strong relationship in either direction.*
Clustering is fine (244 unique problems/255 items). The anchor DESIGN remains sound; this corpus's
verdict coverage is platform-entangled — a real anchor result needs a verdict-balanced
within-(Python×CF) resample (cheap follow-up; candidates.parquet has plenty).

**SURVIVES — corpus-1 (diffs) code-baseline drop.** On the honest intersection (same 20 aspects
measurable in both corpora): comments median ρ .225 → diffs .105, negative delta on 16/20 aspects.
(The .068 previously reported mixed aspect sets; conclusion unchanged, magnitude softer.)
Dominant open confound stands: comments-era programs are OOD on diff text (lower bound).

**Context number (new):** on corpus-1 the judge aspects barely track ACCEPTANCE either — max
|judge~accept| = .18 (a171), accept-rate 86/14 — consistent with acceptance being hard from text.

**Meta-lesson for the deck:** the audit machinery (stratify → permute → Bonferroni → both-sides
significance) killed 2 of 3 headline claims within an hour. That discipline IS the project's
pitch — point estimates and pooled correlations overclaim; certified pipelines catch it.

### Post-audit: the DECISIVE CELL (CF-only × Python-only) — the surviving insight (2026-07-03)

User note: AtCoder/CodeChef/Codeforces are the trusted competition corpora; luogu/codewars/hr/
aizu/usaco are not (they contributed 61/120 of the sampled AC items — future runs restrict to
ac/cc/cf). The clean anchored cell = Codeforces × Python (trusted platform, one language, real
verdict variance): n=168, 33 AC. Also: the 45 'unknown' verdicts are mostly LeetCode (23).

**The substance/surface split (all 16 aspects, same items, same confounds):**
- SUBSTANTIVE quality dimensions carry weak-but-real positive correctness signal:
  a153 hot-path perf **+0.267 (perm p=.0007 — the only Bonferroni-proof aspect)**, a0
  control-flow +0.199 (p=.010), a171 numeric/overflow +0.187 (p=.0097), a135 import hygiene
  +0.173 (p=.022), a36 code smells +0.163 (p=.035).
- SURFACE/style dimensions are exactly flat: a180 one-stmt-per-line −0.009, a162 comments −0.02,
  a81 literate +0.007, a108 naming +0.04, a216 organization, a225 encoding, a207 ≈ 0.
- a153 is not purely TLE-mechanical (excl-TLE +0.217) and survives rough difficulty partialing
  (judge~diff −0.167, diff~AC −0.214 ⇒ partial ≈ +0.24).

**Honest framing:** not "quality anti-predicts correctness" (retracted) and not "quality is
orthogonal to correctness" (overclaimed) — rather: *the judge's quality construct has a thin
correctness-relevant core (substantive dimensions, r≈0.16–0.27) and a thick correctness-irrelevant
shell (style dimensions, r≈0)*. The contrast is internally controlled — same items, same
composition, same difficulty distribution across all 16 aspects; a confound would have to
selectively attack substance aspects only (measured judge~difficulty is small, 0.04–0.17).
Caveats: 33 AC positives, difficulty imbalance (diff~AC −0.21); the definitive version is a
difficulty-matched verdict-balanced resample within CF×Python (cheap; 800K candidates available).
Reliability≠validity stands: judge rel₁ up to .96 on aspects whose correctness signal is zero.

### "Does competition code go V?" — the corpus-native probe (2026-07-03, native_competition_probe.py)

Under the shipped survey rung: ~nothing seam-split to V (0/19 measured aspects above half-ceiling;
3/19 with ρ≥.30; best .42) — but that rung is comments-era OOD. Two 30-line CORPUS-NATIVE programs
(same canonical text the judge saw):
- **a180 one-statement-per-line: .30 → .754 = 77% of ceiling** (within-lang: python .58, cpp .72;
  pooled > per-lang because the construct legitimately differs by language). The mechanical band
  DOES go V the moment the executor matches the representation — user's expectation confirmed.
- **a135 import hygiene: −.33 → −.03 overall; cpp +.58 but python −.07** against a rel₁=.92 judge —
  "mechanical-looking" ≠ "one obvious program": the judge's operationalization has free parameters
  (what counts as hygiene in competitive python) that a first-guess checklist misses. This is
  exactly the gap the gated train-split refinement loop exists to close.

**Chiasmus (deck-ready):** the style aspects — most codable (V) — carry ZERO correctness signal
(decisive-cell surface median r=.007); the substantive aspects — the correctness-relevant core
(median r=.173, max .267) — are least codable so far. What's codable isn't what matters, and what
matters isn't (yet) codable: V and validity are different axes, and this corpus measures both.
Notebook §4 (2026-07-03 rebuild) has the full section: 3-corpora table, decisive cell, probe chart.

---

## PR_EXEC — the F2P-mock experiment (2026-07-03, eval_pr_exec.json)

**Design.** 539 PRs = transplant-consolidated set ∩ diffs corpus; 6 test/correctness aspects
judged by Gemma (rel₁ .80–.96, all measurable); 3 hand-written v0 hybrids whose "code up
F2P/P2F" step calls the MOCKED evidence op `ops.test_transition(dpid)` = stored transplant
outcomes (nothing built/executed, per user directive); NullExecOps ablation twin isolates the
machinery's certified marginal; anchors = accept/reject + days_open, always pooled AND
within-batch (13 usable batches).

**Findings, skepticism applied:**

1. **The evidence op's certified marginal is real but criterion-specific.** Adding the mocked
   transplant outcome to an otherwise-identical program: a128 testing-adequacy ρ .234→.337,
   P(op helps)=.995 (certified); a104 tests-presence .133→.178, P=.962; a67 correctness-risk
   .132→.059, P=.05 — **certified harmful as v0-wired**. Execution verdicts inform "is the change
   adequately tested"; they do not straightforwardly encode "is this correct/low-risk" (label
   coverage: 71% indeterminate; pinned only 29/800).
2. **Honest hybrid readout:** blind-written v0 hybrids (no train-feedback round, unlike PR wave-2)
   are weak as wholes — a104 hybrid .178 LOSES to the description-compiled text baseline .417
   (P(beats)=.003; test-file paths in diffs are structurally easy for text programs). The
   op-marginal comparison is immune (same program ± op).
3. **exec_label ~ accept within-batch = −.006** (pooled .097 = composition, again): the transplant
   F2P/P2F label alone carries NO within-repo acceptance signal — independently replicates the
   F2P split-signal memo's MH-stratified null via a different estimator on a different pipeline.
4. **★ The standout: judge a67 (functional correctness & defect risk) ~ acceptance = .235
   within-batch** (13 batches, n=538, ≈5 SE), **robust to diff-size** (judge~size .009,
   partial | size = .236), not reducible to test-presence (r=.27 vs a104 judge), while days_open
   anchors are all ~0. An untrained single-aspect LLM judge reading title+diff carries real
   within-repo merge signal — larger than the execution machinery's label. (Context: trained
   dense models reach AUC≈.68 on 141K PRs; this is zero-shot and modest — no overclaim. Candidate
   residual confound to note: PR age/style drift within long-lived repos.)

**Extended chiasmus (deck):** for the ACCEPTANCE construct, the A-channel (LLM's holistic
correctness read) out-signals the V-channel (actual test execution) within-repo — merge decisions
are a social/communicative act more than an execution-verified one. Combined with the competition
finding (most-codable aspects carry zero correctness signal), the two anchors now bracket the
thesis from both sides: V ≠ validity, A ≠ noise.

## RECONSTRUCTION objective (R), GLM-5.2 sweep — COMPLETE (2026-07-03, recon_results.jsonl, 527 rows / 479 channels)

**Protocol.** Per channel: GLM-5.2 sees 24 blinded (doc, score) exemplars → states the rule →
rule re-executed on 40 held-out items → R = ρ(execute(m̂), channel). Channels: judge (blinded),
judge_truedesc (no recovery; GLM executes the TRUE description = executor reference),
code (best rung), hybrid (PR only), judge_null (permuted scores = identifiability floor).
Dedup: last non-error row per (task,aspect,channel); 48 first-launch 5xx errors all retried clean;
26 channels GLM-degenerate (constant executor output, R undefined) — flagged, not dropped.

**Median R by task (blinded judge / truedesc reference / code / hybrid / null):**

| task | judge | truedesc | code | hybrid | null floor |
|---|---|---|---|---|---|
| press_releases | .471 (39) | .713 (39) | .254 (37) | .564 (24) | +.03 (5) |
| math | .318 (37) | .681 (36) | .155 (33) | — | +.07 (5) |
| patents | .197 (20) | .498 (16) | .174 (16) | — | −.17 (3) |
| code_review (comments) | .240 (22) | .548 (22) | .189 (23) | — | −.24 (3) |

**Findings (descriptive):**
1. Controls behave: null floor ~0 (med −.03, max .23, n=16); truedesc reference tops at .50–.71 —
   executor-relativity measured directly (same words, different executor ⇒ different metric).
2. Blinded recovery = 40–66% of reference, ranked exactly like §2 codability
   (PR > math > CR ≈ patents) — two independent operationalizations of articulability agree.
3. PAIRED on shared aspects (Simpson guard): hybrid med .564 vs judge .541, higher 11/24 —
   hybrids recover as well as the judge channel (no recoverability penalty for migration).
   The unpaired .564-vs-.471 gap is aspect composition — do not quote it.
4. Code rungs recover WORST (code > judge in only 14/37 paired) despite determinism:
   R tracks natural-language articulability of the rule, not reproducibility. Different axis.
5. Coding read: CR has lowest blinded R (.240) AND lowest reference (.548) — matches §4's
   fidelity-side picture (real but hard-to-articulate construct on comments-era text).

**Notebook updated (2026-07-03):** `notebooks/2026-07-02__metric-seam-certificates-and-overnight-report.ipynb`
(+ .html) now carries: redone 7-corpus codability-vs-ceiling figure + per-corpus summary table (§2),
fleet-anatomy tables — op distribution, shape census, stage-index composition (§3b: 27 programs,
100% open+close in code, LLM at median slot 4, predicate slot 27/27 code, evidence ops 4/27),
pr_exec F2P-mock section with a128 flow diagram + op-marginal and Simpson-guarded anchor tables (§4d),
and the recon section above (§5). 36 cells, 0 errors.

---

## GATE RESOLUTION at n_test=500 (2026-07-03, expansion/gate_expansion_report.json)

The v1 bootstrap left a110/a80 UNRESOLVED at n_test=100 (P(gate)=.59/.31). Fix: 400 fresh
held-out items (same sampling frame len>=1000, disjoint from the v1 250, seed 101, same
canonical head5000+tail2500), judged 2-pass + scope + the h0 hybrids' LLM fields (Gemma,
queue jobs 61/62). NOTHING retrained — frozen h0 hybrids vs frozen best-train-flavor
baselines, PAIRED bootstrap B=2000, test = 100 v1-test + 400 new. Expansion channel health:
rel1 .83-.97, scope 226/400.

| aspect | full n=500: hyb/base -> P(gate) | scoped n=267: hyb/base -> P(gate) | P(beats base) | verdict at n=500 |
|---|---|---|---|---|
| a86 quotes | .869/.669 -> **1.00** | .765/.638 -> .71 | 1.00 | **G1 PASS certified (full)** |
| a110 boilerplate | .755/.590 -> **.989** | .815/.657 -> **.97** | 1.00 | **G1 PASS certified (was unresolved .59)** |
| a80 contacts | .736/.635 -> .53 | .834/.648 -> **.99** | .9995 | **scoped-gate PASS certified**; full-set sits exactly on the +.10 margin (delta=+.101) |
| a105 plain lang | .362/.266 -> .00 | .342/.182 -> .00 | .98 | **A-layer confirmed at 5x data**: beats baseline, decisively below gate |

Readings: (1) all four v1 hybrid verdicts are now RESOLVED in the Rung-3 sense; (2) a80's
full-vs-scoped split quantitatively confirms the v1 diagnosis (residual = out-of-scope
handling, which belongs to the scoping channel, not the criterion channel); (3) a105's
"better than code but far below gate" at n=500 is the cleanest A-layer certificate we have.

---

## E-S1 PLANTED KILL-SWITCH — first end-to-end run (2026-07-03; methods/metric_seam/killswitch/)

**Design pre-registered in killswitch/DESIGN.md BEFORE any evaluation** (7 plants of known
placement on the v1 250 items, laundered as p901-p907; Arm S = synthetic rel-calibrated
channels, Arm J = real Gemma 2-pass on the same descriptions; blind clean-room codegen
(3 rungs x 7) + blind improver hybrids (1 round); a first-fleet blinding breach — one agent
read plants.py — was caught via the agent's own disclosure, the whole fleet quarantined to
codegen_disclosed/, and everything regenerated clean-room; incident logged in DESIGN.md).

**Scorecard vs pre-registered bars (Arm S hard; reports killswitch_report.json +
killswitch_hybrid_report.json):**

| plant (designed) | outcome vs bar | reading |
|---|---|---|
| p901 code | hybrid .796 vs bar .814 (83% of ceil, P=.30) | NEAR-MISS: one evolution round lands 2pp under the 85% bar |
| p902 code+comp-op | **CODE certified** P(.85c)=.99 | PASS |
| p903 evidence-op | code-only .21 P=0 (no false cert ✓); op marginal **P=1.00** (ablation → constant: op carries ALL signal); hybrid 76% of ceil — but with id-exclusion **.944 = 98% of ceil** | PASS on op recovery; placement bar missed ONLY by contract lacking dpid (28/100 truncated docs slip the .995 self-drop) — flagged as risk BEFORE eval; harness upgrade queued |
| p904 "mixed" | pipeline says **CODE** (fields earn ~0, P=.01; code alone 88-95% of ceil both arms) | PLANT MIS-DESIGNED: attribution counting is genuinely code-reachable; pipeline correctly REFUSED the designed mixedness |
| p905 a-layer | no code cert (code .37, hybrid .56 < gate floor) | PASS (directional) |
| p906 null | rel1 −.02 → DEGENERATE, nothing certified ✓; **the p906 improver independently refused to fit the noise** (read rel from the pack, found the planted twin-doc contradictions, declined) | PASS — the fleet itself caught the plant |
| p907 ceiling check | **oracle = 0.999 of predicted S1 ceiling** (pre-reg tol ±.05); no rung exceeds ceiling; placement knife-edge (.734 vs bar .736, P=.47) | S1 FORMULA CHECK PASSED to 3 decimals |

Op-type readouts (strong-executor-without-tool = Arm-J judge; 50%-of-ceiling bar): p902 →
computation (.66), p903 → evidence (.28) — **2/2 correct**. Arm-J extra readouts:
ρ(judge,truth) = p904 .95 / p902 .65 / p901 .53 / p905 .47 / p903 .28 — the judge cannot
know corpus distinctiveness without retrieval (evidence starvation at judge level), and its
p901 construct differs from raw density (oracle only .47 of its ceiling) while staying
rel=.98: operationalization slack measured directly. Arm-J p901 hard bar FAILED as
pre-registered — diagnosis: that bar conflated placement recovery with judge≡designed
construct; the slack is real and is what §3.4 CF machinery exists for.

**Headline: ZERO false certifications in 14 arm×plant cells** — every miss is conservative
(under-certification at strict bars after ONE improver round, which empirically reaches
76-83% of ceiling). Improver recoveries were sharp: p901's agent reverse-engineered "the
judge is a naive numeric-density rater" and p907's "log substantive-prose volume" from 30
noisy examples — both are the actual generating functions.

Follow-ups queued: h1 evolution round for p901/p907 (do the near-misses close?); add
datapoint_id (or exclude-self retrieval) to the hybrid contract; n_test scale-up for the
knife-edge cells; fold placement-accuracy table into the report notebook §6.

---

## CREATIVE WRITING seam survey — the taste pole (2026-07-03; tasks/creative_writing/)

40 aspects x 250 WritingPrompts stories x 2 passes + scope (job 70), code flavors 120/120 ran.
Corpus pristine: 249/250 in-scope (mean 10.0). **37/40 usable** (only 3 degenerate — the CW
bank is broad-applicability, unlike code_review 18/40 / patents 16/40).

- **Median judge reliability .90 — the HIGHEST of any surveyed task.**
- **Median ρ/ceiling = .128 — the LOWEST of any surveyed task** (frac ≥.3 of ceiling: 14%;
  max .59 = sensory immersion a90; then character dimensionality .36, compression .36).
- Anti-correlated / ~zero mechanical-sounding criteria: sentence rhythm .08, line-level
  clarity .14, mechanical correctness .15, causal coherence −.04, earned payoff −.09.

Cross-task frontier (description-compiled tier, median ρ/ceiling): **PR (mixed, hybrids
certify) > math ≈ code_review ≈ patents (op-starved) > CW .128** — with reliability
ORDERING REVERSED (CW judges are the most stable). The taste pole behaves exactly as the
V/A/Taste decomposition predicts: highly reliable judgments with almost no description-
compiled surface signature. (Hybrid/fields evolution for CW = W3.1 fleet, next.)

---

## Kill-switch h1 round + dpid contract fix (2026-07-04)

**h1 refinement closes both near-misses.** One feedback round (top-12 residual cells + 6
well-fit anchors, same protocol as the v1 a80-h1 lesson) on p901/p907, evaluated on the
SAME held-out test split (n=100, B=2000 paired bootstrap):

| plant | h0 held-out | h1 held-out | P(h1 >= 85% ceiling) | P(h1 > h0) |
|---|---|---|---|---|
| p901 | .796 (83% ceil) | **.826 (86% ceil)** | .587 | .9995 |
| p907 | .734 (85% ceil) | **.784 (90% ceil)** | .845 | .999 |

Both now certify at the pre-registered 85%-of-ceiling bar. p901 clears narrowly (P=.59);
p907 clears comfortably (P=.85). No overfitting collapse (train and test both improved in
step — unlike the a80-h1 cautionary case).

**dpid contract fix.** The p903 self-hit gap flagged as a risk before evaluation was real:
the `score(text, extracted, ops)` contract has no datapoint_id, so p903_h0 could only
approximate self-exclusion via a similarity threshold (>=.995), which 28/100 truncated/
edited docs slip past. Fix implemented as a non-breaking wrapper in
`eval_killswitch_hybrids.py` (`_SelfExcludingOps`): `ops` is bound to the true current
datapoint_id per document and always passes it as `exclude_id` to `retrieve_similar`,
overriding whatever the hybrid program itself passes — no existing hybrid program's
signature has to change. Effect on the S6 verdict table (re-run in full):

- **S:p903 flips from UNCERTIFIED(near-miss) to CERTIFIED CODE+EVIDENCE_OP** — hyb_te
  .732 -> **.944** (98% of ceiling), P(>=85% ceil) .05 -> **1.00**. Op marginal unchanged
  at P=1.00 (op carries all signal, confirmed evidence-type).
- J:p903 (real-judge arm) moves .162 -> .292 but stays UNCERTIFIED — consistent with the
  already-documented Arm-J construct slack (judge's p903 correlate with designed truth was
  only .28 of its own ceiling; the judge itself is evidence-starved, not just the harness).
- All other 12 cells unchanged; **zero false certifications still holds across all 14
  arm x plant cells**, and the fix only ever *raises* a true-positive score (self-similarity
  can only depress a distinctiveness estimate, never inflate one) — no risk of the wrapper
  manufacturing a false CODE verdict elsewhere.

Updated final kill-switch scorecard (Arm S, all pre-registered bars): p901 CERTIFIED (h1),
p902 CERTIFIED, p903 CERTIFIED (dpid fix), p904 CODE (plant mis-designed, as before), p905
correctly uncertified (A-layer), p906 correctly DEGENERATE, p907 CERTIFIED (h1). **6/7
plants now certify exactly as their designed type predicts; the 7th (p904) was diagnosed
as a mis-designed plant whose "mixedness" is genuinely code-reachable — the pipeline's
refusal to manufacture a MIXED verdict there was itself validating.**

Remaining follow-up: fold this placement table into notebook §6 (not yet done).

---

## Second-judge-family replication, Llama-3.3-70B (2026-07-04; methods/metric_seam/pilot/eval_llama_replication.py, eval_llama_gate_expansion.py)

Same prompts verbatim, only the judge model swapped (Gemma-31B -> Llama-3.3-70B BF16),
across PR waves v1/v2/v3 + the math survey (queue2 jobs 80-83/90, all rc=0, <0.6% parse
failures per wave).

**Instrument stability replicates well.** Median cross-judge channel agreement: PR v1 .80,
v2 .80, v3 .76, math .55 (math lower — expected, math judge reliability itself is lower,
median rel1 .73 vs PR's .83-.92). Disattenuated agreement (rho / sqrt(rel1_G * rel1_L))
sits near 1.0 for the v1 focus aspects (a80 .83, a86 .98, a105 .91, a110 1.08) — the two
judge families are reading the SAME underlying construct, not different constructs that
happen to correlate.

**Codability ordering replicates strongly across judge families** — the code-rung Spearman
between Gemma-ranked and Llama-ranked flavor cells: PR v1 **+.93** (28 cells), v2 **+.84**
(53 cells), v3 **+.79** (28 cells), math **+.72** (101 cells). Whatever makes an aspect
codable under one judge makes it codable under the other; this is the load-bearing claim
for "codability is a property of the aspect, not an artifact of the judge."

**Gate replication is judge-dependent, and n=500 has the power to show a real split.**
Frozen h0 hybrids + frozen baseline flavor (selected on Gemma train, never re-picked),
re-bootstrapped under the Llama channel at full n=500 (100 v1-test + 400 expansion items,
B=2000):

| aspect | Gemma n=500 (from gate_expansion_report.json) | Llama n=500 | reading |
|---|---|---|---|
| a86 | P(gate)=1.00 | **P(gate)=1.00** | clean replication |
| a105 | P(gate)=0.00 (A-layer) | **P(gate)=0.00** | clean replication (correctly non-codable under BOTH judges) |
| a80 | full .53 / scoped .99 | full .31 / **scoped .64** | scoping-channel diagnosis replicates directionally, weaker under Llama |
| a110 | **CERTIFIED** P(gate)=.989 (full), .97 (scoped) | **P(gate)=0.00 both** (rho_hybrid .50 vs baseline .50, beat margin only +.015 vs the +.10 required) | **DOES NOT REPLICATE** |

a110 is the one real divergence: it certifies cleanly under Gemma at n=500 but the same
frozen hybrid fails to beat the same frozen baseline by the required margin under Llama at
identical n and identical items. Since a86/a105 replicate exactly and the code-rung
ordering replicates at rho .93 on this same wave, this isn't generic judge noise — it's a
specific instrument-dependent result for a110 that needs flagging in the paper as a
limitation (or a target for its own h1 round under the Llama channel specifically).

Remaining: math-survey gate replication not yet run (no v1-style hybrid gates exist for
math aspects yet — blocked on the math hybrid fleet below).

---

## W3.1 hybrid fleets — Creative Writing RESULTS (2026-07-04; methods/metric_seam/hybrids/eval_hybrids_task.py)

37/37 CW aspects evolved (code + up to 2 LLM-extracted fields via `build_packs_task.py` ->
one improver agent per aspect, isolated single-pack reads) and evaluated on the SAME
150/100 train/test split as the pack builder, Rung-3 paired bootstrap B=2000, frozen
baseline (best of v0/v1/v2 description-compiled flavors by TRAIN rho, never re-picked).

**5/37 clear the strict PR-calibrated gate** (rho_test >= max(rho_baseline+0.10, 0.60)):
a144 P=1.00 (rho .842), a72 P=.88 (rho .678), a99 P=.89 (rho .682), a90 P=.68
(rho .740), a342 P=.53 (rho .606). (CW aspect IDs are local to this task, not shared
with the math or PR aspect namespaces.)

**But nearly every hybrid substantially beats its baseline anyway** — median rho_hybrid
**.481** vs. median rho_baseline **.096** (median improvement **+.327**); P(beats
baseline) >= .95 for **25/36** valid aspects, >= .80 for **33/36**. This is the same
taste-pole signature the CW seam survey already found (median rho/ceiling .128 for the
description-compiled tier, lowest of any task): the fields+code intervention closes most
of the gap in effect size, but the absolute 0.60 floor calibrated on PR's much more
codable aspects is rarely reached even after hybrid evolution — codability ceiling is a
property of the criterion class (taste vs. craft vs. mechanical), not just a fixable
baseline weakness. One genuine near-miss worth flagging for a future h1 round: a342 P=.53
(barely over) and several P=.15-.35 cases (a153, a117, a225, a45, a261) sit in a "clear
lift, not-yet-certified" band that h1 refinement (the protocol that closed p901/p907 in
the kill-switch) would likely move.

**Two bugs found and fixed during eval** (both a324-class blind-write defects, not
protocol issues): (1) a324 ("Character Dimensionality") referenced an undefined
`_RELATION_RE` regex, causing a silent `NameError` swallowed by the outer
`except Exception: return 0.5` — every document scored an identical constant 0.5,
Spearman undefined. Fixed by adding the missing relationship-noun lexicon; rho_test then
resolves to +.335 (P beats baseline .55). (2) a288 ("has the story got a title") remains
undefined (NaN) on this test split — confirmed NOT a bug: only 3/250 corpus documents
have a genuine title, and by chance none landed in the 100-item held-out test set. This
is a legitimate degenerate-split case for an extremely rare binary signal, not a program
defect (verified the 3 positive documents all score 0.93-1.00 when run directly).

Full per-aspect table: `outputs/metric_seam_pilot/tasks/creative_writing/hybrid_gate_report.json`.

---

## W3.1 hybrid fleets — MATH RESULTS (2026-07-04; LaTeX-aware AST-style ops, ops_math.py)

35/35 math aspects evolved using the new `MathOps` op set (`extract_math_spans`,
`latex_tokens`, `notation_census`, `equation_stats`, `proof_skeleton`, `delimiter_health`
— pure-stdlib LaTeX span/delimiter parsing, no sympy) instead of raw regex over
mixed prose+markup, motivated directly by the seam-survey finding that math's
description-compiled tier was op-starved (best pre-gate rho .41, several checkable-
predicate criteria at .01-.17). Same protocol as CW: 150/100 split, frozen baseline
(best of v0/v1/v2 by TRAIN rho), Rung-3 B=2000 paired bootstrap.

**0/35 clear the strict gate** (rho_test >= max(rho_baseline+0.10, 0.60)) — closest are
a132 P=.46 (rho .590), a198 P=.43 (rho .597), a42 P=.18, a108 P=.16, a144 P=.15. Median
rho_hybrid **.369** vs. median rho_baseline **.133** (median lift **+.196** — smaller
than CW's +.327, consistent with math already having a higher description-compiled floor
per the seam survey, so there is less headroom for the LaTeX ops to add). P(beats
baseline)>=.95 for 16/34; P(beats baseline)<.5 for 4/34 (a234, a36, a6, a222 — genuinely
weak hybrids, verified NOT bugs: all produce varied, error-free scores, they simply
under-perform their own baseline on held-out; a0 also regresses, -0.170). One legitimate
degenerate case: a210's frozen baseline flavor collapses to a CONSTANT on the 98-item
test split (Spearman undefined) even though its TRAIN rho was already negative (-0.10,
i.e. "least-bad of three bad flavors") — the hybrid itself scores a valid .531, so this
reads as "baseline degenerate, hybrid fine," not a hybrid failure.

**Reading:** math hybrids improve on baseline less dramatically and less consistently
than CW's — a mixed picture where the LaTeX-aware ops clearly help some criteria a lot
(a66 +.615, a156 +.53, a72 +.51, a168 +.47, a60 +.40 — several of these are exactly the
notation/delimiter-hygiene criteria the ops were built for) while leaving a genuine
weak tail (4-5 aspects where the hybrid underperforms baseline). This is a different
signature from CW's uniformly-positive-but-under-gate pattern, and consistent with
math sitting between PR (op-rich, gates readily) and CW (taste, gates almost never) on
the codability spectrum — median rho_hybrid .369 here vs CW's .481 and PR's substantially
higher certified rates. No h1 refinement round has been run yet for math (unlike the CW
near-misses); given 5 aspects sit within P=.15-.46 of the gate, an h1 round targeting
those specifically would be the natural next step, following the same protocol that
closed p901/p907 in the kill-switch.

Full per-aspect table: `outputs/metric_seam_pilot/tasks/math/hybrid_gate_report.json`.

## R2: TVD-MI↔Spearman bridge calibration + lemma-gap closures (2026-07-04; methods/metric_seam/pilot/bridge_calibration.py)

Roadmap-v2 R2 (notes/2026-07-04__metric-seam-roadmap-v2.md). 609 (aspect × channel) pairs — every
survey source with a 2-pass Gemma target (PR v1/v2/v3 incl. saved hybrid columns, all 7 task
surveys), each pair scored with Spearman ρ, Pearson r, and TVD-MI (vinfo 2-bin, perm-debiased)
against the same 2-pass judge mean → `outputs/metric_seam_pilot/bridge_calibration.json`.

**Bridge (lemma-note Gap 9):** monotone in the mean — TVD-MI decile means rise .015 → .666 as
|ρ| goes 0 → .9; Spearman(|ρ|, TVD-MI) = .71 over pairs — but per-pair spread is wide
(|ρ| .6–.7 maps to TVD .21–.55). Licensed: the decile table as a directional lookup for
cross-stack prose. Not licensed: per-channel inversion. One residual outlier (code_competition
a63_v2_holistic, n=39).

**Gap 3 at scale:** median |ρ_S − r_P| = .023 (matches T5) but p90 = .080 / p99 = .198 /
max = .286; p90 = .111 even at |ρ| ≥ .5. Adopted wording: gates = empirical-bootstrap Spearman
statements (Rung-3 only, never invoke Lemma A2); ceiling-normalized readings must quote the
per-channel Pearson companion.

**★ Estimator bug found & fixed (`methods/metric_implementer/vinfo.py::_binize`):** the 2-bin
rank-median split broke ties by stable sort order = item position; with both sides sharing item
order, two INDEPENDENT ~90%-tied vectors read TVD-MI ≈ .7–.85 after debias (the permutation
floor destroys exactly the order coupling it should calibrate). 21/609 calibration pairs were
inflated this way (all heavily-tied judge or code columns: pattern = high 2-bin TVD, 5-bin TVD
= 0, ρ ≈ 0). Fix: independent seeded random tie-breaking per side (≡ independent infinitesimal
jitter); artifact cases now ~0, continuous perfect-dependence .95, 19/19 tests green. Blast
radius: `tvd_recovery`/`tvd_transmission` are closed-form (no binning) — headline R̂/T̂ safe;
`measures.py` scorecard diagnostics (`tvd_mi` reliability/invariance/applicability fields) on
heavily-tied metrics are suspect pre-fix (no contaminated local artifacts found; sk3-side
scorecard runs need recompute where used). Post-fix semantics: TVD-MI on heavily-tied data
reads dependence-after-jitter (a lower bound), by design.

**certificates.py hardening:** `enumerate_stump_class` docstring restated to gridded-class
scope (Gap 4); `u2_matroid_bound` gains per-part tightening (lemma B1 remark R2, strictly
tighter, still valid) + anti-conservative-γ̂ warning; new `u2_matroid_bound_curve` reports U₂
as a curve in γ (Gap 5a discipline: never quote a plug-in γ̂ point). 14/14 planted tests green.

**Scoped-gate rule ADOPTED (user sign-off 2026-07-04):** a scoped certificate is legitimate iff
(a) scope predicate criterion-independent (document-type applicability), (b) frozen before
gating, (c) applied symmetrically to hybrid and baseline, (d) certificate stamped as scoped
with coverage fraction. Under this rule a80 is SCOPED-CERTIFIED (n=267, P(gate)=.99, scope =
"is actually a press release" = criterion-independent, symmetric, frozen).

## CAM profile — the per-task complexity takeaway (2026-07-04; methods/metric_seam/pilot/cam_profile.py)

Answer to "what does this line of work tell us about the underlying complexity of these
metrics, per task, as a number/certificate": per criterion, the one-sided certificate is
r̃ = clip[0,1](ρ_test / attenuation ceiling) of its best MATERIALIZED implementation; per task,
the survival curve of r̃ over judge-measurable criteria; scalar = CAM (mean r̃ = area under the
survival curve), with depth marks frac≥.5 / frac≥.8. Semantics: LOWER-bound object (monotone
under more search; 1−CAM = uncertified residual, NOT proven-tacit); plants calibrate the search
(one blind round reaches ~76–90% of a known ceiling → r̃/0.8–0.9 = calibrated reachable-share
band); Gap-3 rule applies (Pearson companion near ceiling). → cam_profile.json

| task | CAM base → certified | frac≥.5 | frac≥.8 | reading |
|---|---|---|---|---|
| press_releases (n=20) | .369 → .697 | .95 | .25 | mostly compilable; a quarter compiles DEEP |
| creative_writing (n=36) | .131 → .466 | .47 | .03 | broad SHALLOW compilability — everything lifts, almost nothing compiles deep (taste pole) |
| math (n=34) | .173 → .377 | .27 | .00 | bimodal: LaTeX-hygiene band compiles hard, reasoning band resists; lowest certified mass |

Note the axis split: on the strict absolute gate CW(5/37) > math(0/35), but on certified MASS
CW(.466) > math(.377) with neither near PR — the two rankings answer different questions
(clear-a-fixed-bar vs how-much-of-the-judge-signal-is-certified-articulable). Recon-R's
independent language-articulability ordering (PR .471 > math .318 > CR .240 ≈ patents .197)
agrees with CAM's task ordering where they overlap.

## R1: money figure LANDED (2026-07-04; methods/metric_seam/pilot/money_figure.py)

`outputs/metric_seam_pilot/figures/money_bracket.{png,pdf}` + notebook §8. Panel A: per-criterion
brackets floor→certified (ceiling-normalized, held-out) for PR/math/CW, colored by gate status
(PR 12/20 certified at P≥.5, math 0/34, CW 5/36). Panel B: CAM survival curves + recon-R
annotations. Panel C: lower-arm-only corpora (survey-grade) with op-type diagnoses. Honest
artifact made visible: a few low-reliability PR criteria have r̃≈.85 but stay red because the
G1 absolute .60 bar exceeds their attenuation ceiling — the fixed bar is unreachable for
low-rel judges; flag when quoting per-criterion gate verdicts (ceiling-relative reading is what
Panel A's axis shows).

## R4: h1 refinement fleet — held-out gates (2026-07-04; build_h1_packs_task.py / eval_h1_task.py)

10 blind Sonnet improvers (math a132/a198/a42/a108/a144, CW a153/a117/a225/a45/a261), packs =
criterion + h0 source + 12 worst TRAIN residual cells (+6 anchors) + the a80 anti-overfit
warning; all revised their LLM fields → 3,500 field prompts re-extracted (namespaced
`<aid>.h1__<field>`, same Gemma env/scorer as h0; one-off on sk3 GPU 7 after queue jobs failed
on GPU-1 contention with the other thread's alpha-probe run — untouched). Promotion rule
pre-registered: h1 replaces h0 iff P(h1>h0) ≥ .8, paired bootstrap B=2000, same held-out items.

**RESULT: 0/10 promotions — h0 stays HEAD everywhere.** (plumbing verified: 0 missing fields,
full boot; a117-h1 produces 96 distinct scores — regressions are construct failures, not bugs)

| aspect | ρ h0 → h1 | P(gate) h0 → h1 | P(h1>h0) |
|---|---|---|---|
| math a42 | .550 → **.634** | .18 → **.52** | **.79** (bar .80) |
| math a108 | .537 → .564 | .18 → .31 | .64 |
| CW a153 | .565 → .560 | .33 → .29 | .48 |
| CW a261 | .534 → .532 | .14 → .15 | .48 |
| CW a225 | .565 → .525 | .33 → .20 | .26 |
| math a198 | .597 → .492 | .42 → .13 | .17 |
| math a144 | .483 → .390 | .15 → .03 | .19 |
| CW a45 | .544 → .475 | .21 → .04 | .03 |
| math a132 | .590 → .398 | .47 → .00 | .01 |
| CW a117 | .573 → .143 | .36 → .00 | .00 |

Readings. (1) **The a80 lesson replicates at fleet scale**: train-residual-guided single rounds
mostly do NOT transfer — 2 gains, 3 ties, 5 regressions (a117 catastrophic: redesigned fields
measure a different construct). The held-out promotion discipline caught 10/10. (2) **Plant-vs-
real contrast is the finding**: the kill-switch plants (codable by design) reached 86–90% of
ceiling with the SAME h1 protocol; real taste/craft near-gate criteria did not move. h1 rounds
are the only currency the "belongs as prompt" direction can pay in — and they mostly came back
empty → saturation evidence that the residual on these criteria is genuine A-layer, not search
shortfall. Feeds claim (vii). (3) **a42 disposition**: h1 crosses the certification gate
(P_gate=.52) but misses promotion by .011 — do NOT rule-bend; resolve on fresh items
(gate-expansion protocol, as a110/a80 at n=500) if we want it. (4) Improver field-redesign is
the risky move: all 10 changed fields; the 2 winners (a42/a108) added a *discriminating second
field* to an existing design; the big losers (a117/a132) replaced the construct wholesale.

## R3: same-f (TVD) headroom T−R on press releases (2026-07-04; methods/metric_seam/pilot/headroom_pr.py)

Both legs in ONE f (lemma A1/C3 discipline), 29 PR aspects (v1 10 + v2 19 usable), 250 items:
T̂ = tvd_transmission of the (N,2) binarized pass matrix (shared per-aspect pooled-median
threshold, strict >; no degenerate splits at the .10–.90 bar); R̂ = tvd_recovery(m̂, pass-p
binary verdict) averaged over passes, reported BOTH graded (calibration-sensitive lower bound)
and rank-median-binarized (scale-free; tie-safe split). → headroom_pr.json

Results. (1) **DPI guardrail: 0/29 violations** in either reading — R̂ ≤ T̂ everywhere (post-fix
tvd stack behaving; a violation would have flagged leakage per Gap 1). (2) T̂ ∈ .29–.48 of the
.5 binary cap (T_norm in JSON). (3) **Hybrids close TVD headroom**: hybrid R_bin > best-code
R_bin on 22/23 aspects with both; flagship a86 hybrid R_bin=.423 vs T=.480 → h=+.057, smallest
in the table. (4) ★ cross-stack replication of the R4 verdict: for a80, TVD ranks h1 (R_bin
.198) BELOW h0 (.222) — the TVD functional independently agrees with the Spearman gate that
rejected a80-h1. (5) Interpretation discipline: h magnitudes are NOT pure articulability
(attainability/rounding, Gap 11; graded-R calibration sensitivity); certified content = h ≥ 0
+ the relative ordering. Blinded-judge recon channels remain Spearman-only (no per-item preds
in the recon store) — flagged, not silently mixed.

## R7.2: HUMOR seam survey — second taste-pole point (2026-07-04; tasks/humor/)

New domain (user-approved expansion). build_task.py gained a humor ROLE + per-task MIN_LEN=150
(jokes are complete documents at median ~483 chars; the default 600 floor would have biased the
pool to long-form). 40 aspects × 250 items (label balance 133/250), 20,250 prompts judged
Gemma 2-pass + scope on sk3 GPU 7 one-off (GPU 1 still held by the other thread's alpha-probe);
scope 248/250; all 120 pre-existing codegen flavors ran clean.

**Result: humor sits ON the taste pole with CW.** 34/40 judge-measurable; the 6 degenerate are
all performance-modality criteria (heckler management, stage presence, musical branding,
production craft, comics-form, timing/delivery) that the judge correctly NA's on text — the
NA channel doing applicability work. Median rel1 .85 (high, CW-like), median best-code
ρ/ceiling **.102**, frac ≥.3 = **12%** (CW: .128/14% by the same survey-grade measure).
Most-codable: a171 single-panel cartoon economy .77 (heavily NA-scoped, n=83), a36 writing
economy .45, a135 platform standards .38 — the "mechanical band" again. Soft-construct criteria
(misdirection/reveal −.02, SSTH/GTVH .02, cross-cultural translatability .01, economy/brevity-
as-judged −.17!) sit at zero code signal despite rel1 .88–.94 — highly reliable judgments with
no compiled floor, the taste signature.

Upper arm in flight: 31 improver packs (build_packs_task.py + humor hazard note), 31 blind
Sonnet improvers → programs_humor/<aid>_h0.py; then field extraction + held-out gates + humor
CAM (4th task in the money figure).

## R7.2b: HUMOR fleet gates + CAM — humor completes the 4-task frontier (2026-07-04)

Fleet landed: 31/31 blind Sonnet improvers returned (`programs_humor/<aid>_h0.py`), all
compile, all declare exactly 2 LLM fields, 31/31 smoke-pass on real items with empty fields.
Field extraction: 15,500 prompts (62 fields × 250 items) on sk3 GPU 7 one-off, 15,500/15,500
rows, 0 empty. NOTE an infra failure worth remembering: the first launch died at engine init
with `nvcc fatal: Failed to preprocess host compiler properties` during a flashinfer JIT build
of the sampling op — the bare non-login SSH shell (AFS home unreadable → no profile) lacks the
CUDA env. Fix: replicate the queue2 runner's env (PATH+=/usr/local/cuda-12.8/bin, CUDA_HOME,
LD_LIBRARY_PATH, TMPDIR=/lfs/.../tmp, CUDA_CACHE_PATH=/lfs/.../.cache/cuda). This is a THIRD
cause of "EngineCore failed to start" on sk3, distinct from zombie-EngineCore and
teardown-lag contention.

**Gates (held-out n≈100, G1 = max(base+.10, .60), B=2000): 4/31 certified.**

| aid | criterion | rel1 | base | hyb | r~ | P(gate) |
|---|---|---|---|---|---|---|
| a351 | Representation ethics & harm minimization | .91 | +.30 | **+.76** | .78 | .991 |
| a135 | Platform/broadcast standards | .94 | +.34 | **+.66** | .67 | .803 |
| a153 | Cross-cultural translatability | .92 | **−.18** | **+.61** | .62 | .583 |
| a81 | Topical angle and anchoring | .85 | **−.10** | **+.61** | .64 | .547 |

Near-misses: a36 writing economy (+.45→+.57, P=.31), a315 punchline-last placement
(+.14→+.50, P=.13), a333 rule-of-three (+.25→+.49, P=.11). 19/31 beat baseline at
P≥.95 — the absolute .60 bar (not lack of lift) blocks most, the same low-reliability
artifact documented for PR in §R1.

**Regressions (construct-replacement pattern, replicating R4):** a297 timing/delivery
(+.38→+.13), a90 storytelling/personal material (+.15→−.04), a306 SSTH/GTVH-based analysis
(+.15→+.09), a216 narrative coherence (flat ~0). The R4 lesson holds out-of-domain: improvers
that REPLACE the baseline construct with an LLM-field construct lose; winners ADD a
discriminating field on top of real code signal.

**The certified/uncertified split is the takeaway.** What certifies in humor is NOT comedic
craft — it is the standards/framing band: ethics-compliance, platform standards, audience/
reference-frame criteria (translatability, topical anchoring — both with NEGATIVE code
floors that the 2 LLM fields flip to +.61). The craft core (timing, storytelling, incongruity
theory, coherence) is where hybrids stall or regress despite judge rel1 .85–.94. Humor's
tacit residual is the comedy itself; its articulable mass is the compliance shell around it.

**CAM (4-task frontier, money figure regenerated):**

| task | n | CAM base→cert | frac≥.5 | frac≥.8 |
|---|---|---|---|---|
| press_releases | 20 | .369→.697 | .95 | .25 |
| creative_writing | 36 | .131→.466 | .47 | .03 |
| math | 34 | .173→.377 | .27 | .00 |
| **humor** | **31** | **.120→.351** | **.23** | **.00** |

Humor posts the LOWEST certified CAM of the four — more taste-pole than CW on the certified
measure (survey floors had them tied at ~.10 median). One blind round; kill-switch calibration
(~.8–.9 of reachable ceiling per round) puts humor's reachable-share band at ≈.39–.44.
Money figure now 4 tasks (Panel A n=121 brackets total; humor bracket column added; Panel B
humor curve #bc4c00). Caveats: a252 gate underpowered (n=43 test cells, borderline judge
coverage); humor h0-only (no h1 round) — same one-round footing as CW/math, so cross-task
CAM comparisons are apples-to-apples.

## R7.1: patents_pa — prior-art evidence op vs a doc-only judge is a FORCED NULL (2026-07-04)

Task built pr_exec-style over the other-thread prior-art corpus: 250 apps (125/125 final
outcome) sampled from the 8,314 apps in BOTH the v3 rejections CSV (doc text) and
option3_claims_gemma_scale.jsonl (per-claim K=8 refs + Gemma disclosure verdicts, 21,447 apps).
Payload (pa_features.json) strips ALL supervision (label, rejection_type, gold_*, rejected_*);
`PriorArtOps.prior_art(dpid)` + `NullPriorArtOps` ablation twin (ops_pa.py). 8 aspects × 2
passes + scope + 7 LLM fields = 6,000 prompts, one sk3 GPU-7 pass; 24/24 code flavors;
4 Sonnet-written evidence-op hybrids (a26 non-obviousness, a34 novelty bars, a60 prior-art
differentiation, a35 triad). Eval: eval_patents_pa.py → pa_eval.json.

**Survey replicates evidence-starvation on the new corpus:** a26 rel1 .774 / all flavors ≈ 0
(r̃ = .00); best floors a35 .26, a25 .24, a60 .21 (rel1 .13!), a16 .19, a36 .08; a22 all-NA.

**Op marginal (hybrid_full vs hybrid_noop, held-out): a26 P=.03 (op HURTS: −.065 vs +.124),
a60 P=.24 (−.150 vs −.085), a34 P=.34 (≈0 either way; judge n=95, 4 distinct test values →
bootstrap-degenerate, point ρ +.07), a35 P=.96 — but the "win" is +(−.118) vs (−.319): the op
attenuated an anticorrelated text arm toward zero, not toward the judge.** No gate certifies
(P(gate)=0 across the board).

**Reading (this is the point, not a failure):** the judge target M̄_E(x) is a function of X
alone, so I(M̄(X); Z | X) = 0 exactly — a Z whose content is orthogonal to the document text
CANNOT help reconstruct a doc-only judge; any observed marginal is noise or the X-correlated
part of Z. pr_exec's mild positive marginals now read correctly: exec outcomes are partially
X-recoverable (test files visible in the diff); prior-art disclosure state is nearly
X-orthogonal (anchors: pa_exposure⊥outcome ρ=.003 in this all-§102-rejected pool, while the
doc-only judge's a26 read correlates .262 with final outcome — selection effect noted).
Certified seam verdict for the patents corner: **evidence-dominant criteria are
unreconstructible from X for every executor — including the judge**; the evidence op's value
is only well-posed against a Z-aware target. Honest forward path (NOT run; flagged for
sign-off): an evidence-aware judge — same Gemma, prompt contains x AND the disclosure record
— giving M̄(x, Z) as target; still label-free, reconstruction-only compliant. Theory hook:
Prop 5.1 lattice monotonicity + DPI — widening the CHANNEL's executor class cannot help when
the TARGET lives at a lower level (worked example added to the binding/provenance note §7).

## R7.2c: LEGAL (Title VII) seam survey — the SPECIFIED pole (2026-07-04)

Second user-approved new domain, built lean: aspects = 20 stride-sampled from the 50
adversarially-verified Title VII doctrine rubrics (THIN_METRIC_REGISTRY workflow lineage,
online-rubrics/by-law/title_vii/doctrine.json); items = 250 of the 1,172 balanced ex-ante
case-facts narratives (title_vii_balanced_v2.jsonl, all ≥1,240 chars); 60 codegen modules by
5 Sonnet agents (batch 3 caught+fixed a real date-anchor bug: prefer nearest date AT-OR-AFTER
the anchor phrase); 10,250 prompts, one GPU-7 pass; scope 204/250.

**Result: Title VII doctrine sits at the CODED pole — the highest survey floor of any domain
outside PR.** Judge rel1 .68–.97 (median .89); median best-flavor ρ/ceiling **≈.41**, frac
≥.3 = **65%** (13/20). Compare survey-grade medians: patents .10, humor .10, CW .13,
code_review .23. Top: federal-sector flag .76, procedural exhaustion/timeliness .73 (the
stdlib date-arithmetic extractors applying the 180/300-day windows), constructive discharge
.68, protected-class membership .57, retaliation adverse action .57, protected activity .55.
Bottom: same-actor hire-to-fire gap −.02 and 15-employee threshold .05 (the governing
quantities are rarely stated in the facts — evidence-starved-lite, not construct-hard),
majority-group flag .18.

**Reading:** doctrinal criteria are *specified by institutional design* — statute and case
law are a community that has ALREADY compiled its norms most of the way to code (provenance
ladder rung 1 in notes/2026-07-04__where-meaning-lives-binding-provenance.md). The
articulability spectrum across our domains now runs **law > PR > math > CW ≈ humor**, which
is exactly the anthropological prediction: articulable mass tracks how much norm-compilation
work the source community has institutionally done. Caveats: survey-grade (full-sample, no
held-out, lower arm only — no improver fleet run, token-conservation directive); single legal
domain (title_vii); the two floor-zero criteria fail on missing evidence in X, not tacitness.
Panel C of the money figure now carries legal_title_vii with diagnosis "specified pole".

## TRANSPORT TEST — interpreter-swap on all field-bearing hybrids (2026-07-05 overnight)

Design (theory note §7): re-extract every hybrid's LLM fields with Llama-3.3-70B (the W1.2
second judge family) — 56,750 prompts, 5 tasks, one GPU-7 sequential pass, 0 failures — then
re-score the FROZEN programs on the FROZEN held-out splits under three conditions: gemma
(certified), llama (family swap), blank (fields ablated). Per criterion: field_marginal
fm = ρ_g − ρ_blank (borrowed-meaning weight), transport_delta td = ρ_g − ρ_llama (certificate
loss under swap), ratio td/fm, P_degrade (paired bootstrap). Harness:
hybrids/transport_eval_task.py (+ transport_eval_v2.py, f2p_mock/transport_eval_pa.py by two
Sonnet adapters; PR ρ_g reproduces certified gate numbers within .008 except a25 +.078).

**Both retrieval-theory predictions confirmed (pooled n=120 criteria):**

| task | n | ρ_S(fm, td) | med fm | med td |
|---|---|---|---|---|
| press_releases | 20 | .607 | .085 | .013 |
| creative_writing | 36 | .668 | .289 | .065 |
| math | 34 | .505 | .148 | .076 |
| humor | 30 | .537 | .172 | .065 |
| **pooled** | **120** | **.591** | | **ratio med .30** (n=101, \|fm\|>.05) |

(1) **Certificate loss tracks borrowed meaning** — pooled Spearman(fm, td) = .59: swap the
interpreter and you lose in proportion to what the program borrowed. Direct evidence the
field content is retrieval-from-the-interpreter, not incidental. (2) **Most borrowed meaning
is SHARED culture** — median ratio .30: ~70% of the field signal survives the family swap.
If the payload were checkpoint-idiosyncratic, ratios would cluster at 1; fully shared, at 0.
The enculturated payload is mostly common training culture — "enculturation" over "model
quirk."

**But binding is graded and has real certificate consequences.** 34/120 criteria degrade at
P≥.95; **3 of PR's 12 certified gates do not survive the swap** (fall below the .60 bar):
a87 "Human, humble spokesperson tone" .813→.591 (td=.222, ratio 1.07 — fully
extractor-bound; fittingly the most taste-flavored certified PR criterion), a104 scannable
formatting .633→.571, a112 links/navigation .612→.588. Math is the least transportable
(median ratio ~.51; a108 td=.544) — its field constructs are the most family-idiosyncratic.
Conversely **6/120 criteria genuinely IMPROVE under Llama** (2 further P=0 cases are inert-field ties) (humor a9 sensitive-content
.431→.652 P=.008, humor a90 storytelling, CW a171/a54): when the construct is shared,
extractor competence varies and swap can be an upgrade.

**patents_pa: null contributor by program design** — all 4 programs gate fields behind
case-sensitive verbatim-containment grounding that paraphrased extractions never pass →
fm ≈ 0, fields inert (Sonnet adapter traced it; a60's lowercased check is the lone tiny
exception). Lesson: over-strict grounding silently disables borrowed meaning.

**Consequences adopted:** certificates should be stamped as (criterion, judge-family,
executor level, field-extractor family) with transport_ratio as the extractor-boundness
coordinate; the same-family discipline is now a measured necessity for ~1/4 of certified
gates and a measured over-caution for the rest. Artifacts: transport_eval.json in v2/ and
tasks/{creative_writing,math,humor,patents_pa}/.

## §R7.3 — Legal (Title VII) UPPER ARM: 9/20 certified, CAM .372→.621 (2026-07-05)

Fleet completed the survey (§R7.2c): 20 blind Sonnet improvers → programs_legal/, 9,750
Gemma field extractions (GPU-7 one-off), eval_hybrids_task on the frozen 150/100 split.
**9/20 gate-certified at P(gate)≥.5** — a44 protected-activity-present .993, a46
specific-neutral-practice .993, a39 discipline-postdates-protected-activity .966, a15
HWE-elements .959, a0 protected-class .946, a23 procedural-exhaustion .936 (ρ .825, task
max), a18 materially-adverse-action .914, a36 same-actor-gap .823, a5
replacement-outside-class .608; 15/20 beat baseline P≥.95. **CAM .372→.621, frac≥.5 .80,
frac≥.8 .15 — 2nd of 5 fleet tasks**; money figure regenerated 5-task
(legal color #1a7f37; removed from survey panel).

Reading: the certified set is exactly doctrine's ELEMENTS layer (checkable claim elements +
temporal/procedural structure); the uncertified tail is doctrinal gestalt (constructive
discharge, cat's-paw, direct-evidence characterization, discrete-act-vs-HWE
classification). Baseline CAM .372 ≈ PR .369 (the two institutional domains tie at the
description-compiled floor); after one evolution round PR pulls ahead (.697 vs .621) —
spectrum now stated as **law ≈ PR at the compiled floor; PR > law > math > CW ≈ humor
certified**.

★ TWO SURVEY DIAGNOSES REVISED: §R7.2c called same-actor-gap and 15-employee-threshold
"missing evidence in X, not tacitness" (survey r̃≈0). The upper arm RECOVERED both — a36
−.046→+.693 (surname matching + calendar arithmetic across hire/fire events), a28
−.037→+.467 (gate-fails only on the absolute .60 bar). Survey-zero ≠ evidence-absent:
one improver round distinguishes extraction shortfall from genuinely missing evidence.
One improver regression: a13 constructive-discharge .700→.458 (P(beats)=.10) — h0 keyword
baseline stays HEAD (construct-replacement failure mode again).

## §TRANSPORT-3FAM (E6) — Qwen third family: THINKING-LEAK INCIDENT, re-run in flight

First Qwen-3.5-122B extraction pass (54,750 prompts, all 4 corpora) completed with full
row counts but was UNUSABLE: llm.chat used the model's default chat template, whose
thinking mode ON meant every field burned its 48-token budget on "Thinking Process:..."
and truncated before any answer (median raw 183 chars vs 8 for Gemma/Llama). Caught by
the check-the-distribution discipline: humor 3fam preliminary showed rho_qwen ≈ rho_blank
(fields inert) — an artifact reading "Qwen transports terribly" that raw inspection
exposed in one look. Scorer patched (chat_template_kwargs={"enable_thinking": False} +
defensive think-strip); bad outputs quarantined (*_thinking_bad.jsonl, both machines);
humor transport_eval_3fam_BAD_thinking.json void. Harness transport_eval_3fam.py ready
(adds ratio_l↔ratio_q criterion-boundness correlation + degrade-agreement counts);
smoke test (30 prompts) verifying the patch before the full re-run.

## §CODA — Codability priors from criterion phrasing: pre-registered probe, mostly NULL (2026-07-05)

Pre-reg: seam-position note §3 (features + signs fixed before data). Probe over 167
criteria with valid y_code (6 tasks; ceiling>0.3 filter), blind annotation with anchors in
every batch (pass 1 failed anchor bounds → one permitted redo; pass 2 3/6 batches clean,
residuals = off-by-1/2 on guess bounds, within report). Artifacts:
outputs/metric_seam_pilot/coda_probe/{features.csv, outcomes.csv, analysis.json}. Note:
the annotation covered the FULL aspect banks (1,571 criteria) — a standing feature bank
for prospective use, not just the 167 with outcomes.

RESULTS (descriptive): (a) 6/8 pre-registered feature signs correct vs y_code, but tiny —
only F2 span-locality (+.152, p=.050) and F4 reader-effect (−.154, p=.047) reach ~.05;
F5 rule-shape and F8 cross-positional ≈ 0 (sign misses). (b) LOTO combined model pooled
Spearman **−.178** — NEGATIVE via a between-task offset: mechanically-phrased domains
(patents) read as highly codable but realize low r̃. Within-task ≈ 0 except PR (+.377).
(c) Zero-shot codability guess = best prior overall (+.180; PR +.446) and beats the
fitted feature model out-of-task; rel1 alone +.148. Per pre-reg §3.3 verdict scale
(≥.5 bar): **phrasing underdetermines the seam in this data — you must run the pipeline.**

READING (the twist that makes the null informative): the poster failure is patents
"claim count compliance" — guess=10, perfectly mechanical phrasing, y_code=.124, because
the counted object isn't in X. Codability factorizes as phrasing-type ×
evidence-availability × judge-reliability, and criterion TEXT only reveals the first
factor. This is the mirror image of today's §R7.3 legal reversal (survey-zero ≠
evidence-absent): ex-ante reads in BOTH directions — "looks mechanical" and "floor is
zero" — are unreliable without running the evolved arm. Where the evidence regime is
homogeneous (within PR), phrasing DOES carry signal (+.38–.45). Caveats: y_code is a
lower-bound object (search shortfall attenuates any true phrasing signal); n per task
20–37; no heavy model warranted (probe already bounds the ceiling).

### §TRANSPORT-3FAM RESULTS — clean Qwen re-run: both predictions replicate; boundness is criterion-level (2026-07-05)

Re-run complete (54,750 rows, 0 thinking leaks, median raw len 7–10 ≈ Gemma/Llama's 8).
transport_eval_3fam on all 4 hybrid corpora (frozen programs/splits; files
{v2,tasks/*}/transport_eval_3fam.json). Pooled n=123 hybrids / 120 with fm / 101 with
defined ratios:

| swap | fm↔td ρ | med td | med ratio | degrade P≥.95 | improve P≤.05 |
|---|---|---|---|---|---|
| Gemma→Llama-70B | .591 | .054 | .299 | 35 | 6 |
| Gemma→Qwen-122B | .377 | .023 | **.230** | 21 | 8 |

Per task median ratio_q: PR .026, CW .232, math .371, humor .129 (ratio_l: .163/.256/.567/.434).

1. **Replication:** third family confirms both retrieval predictions — loss tracks fm and
   median ratio ≪ 1 (~77% of borrowed signal survives). On PR, Qwen loses essentially
   NOTHING (med ratio .026); fm↔td_q degenerates there for lack of loss variance — the
   most norm-compiled task is also the most transportable, as shared-culture predicts.
2. **Capacity direction:** the larger extractor loses less on ALL 4 tasks (21 vs 35
   degrades; 8 vs 6 improvements) — consistent with enculturation-depth-grows-with-
   capacity, though cross-family; E3 same-family staircase stays the clean test.
3. **★ E6 (criterion-level boundness): pooled Spearman(ratio_l, ratio_q) = .295 (n=101)**,
   positive in each task (PR .104, CW .177, math .233, humor .366) — WHICH criteria lose
   signal is partially a property of the criterion, not the swap pair. 11/120 degrade
   under BOTH swaps = candidate judge-family-bound constructs, headed by a87
   humble-spokesperson-tone (r_l 1.07, r_q .96 — both families lose ~all of it;
   two-family replication that its certified content is Gemma-idiosyncratic); also PR
   a119 (the evidence-op aspect), CW a117/a225/a252/a315/a99 (taste core), humor a351
   representation-ethics (partially bound despite certifying), math a48/a60.
4. **3 convergent improvements** (both families beat Gemma P≤.05): PR a115, CW a54, humor
   a90 storytelling (−.04 → +.22/+.18) — shared construct, weak original extractor;
   the improvement direction also transports.

Certificate stamping unchanged (family + transport_ratio); 3fam adds ratio_qwen as a
second boundness coordinate. Paper C7 updated.

## §BATTERY-PROBE — local API probe of E1/E2/E3 (2026-07-05 evening)

While sk3 was down: PR only, the 4 e2-bearing certified criteria (a76/a87/a104/a112),
100 test items, docs ≤12k chars. Extractors via API: GLM-4.7 (all conditions, 2763/2800),
GLM-5.2 (full only, 781/800, 2.4% transient 529s), Llama-3.2-3B + 3.1-8B via OpenRouter
(full only, 800/800 each, 0 err). Within-GLM-4.7 comparisons; frozen h0 programs.
Eval: methods/metric_seam/battery/probe_eval.py -> outputs/.../battery/probe/probe_eval.json.

E1-KEY (frac = surviving fraction of within-GLM fm):
  a104: frac_name .98  frac_nonce −.07  P(name>nonce)=1.00   — textbook key signature
  a112: frac_name 1.10 frac_nonce .22   P=1.00               — key signature
  a76:  frac_name 1.06 frac_nonce .83   P=.83                — mixed; definition carries most
  a87:  unusable (within-GLM fm denominator .066, barely over the .05 guard; fracs unstable)
  -> constructs vary key-like vs spec-like; 2/4 clean retrieval signature.

E2-STIP (conflict set = stipulated answer ≠ model's own full-condition answer):
  doc_type (a87): comply .36 / snapback .46 (n=39)
  doc_kind (a104): comply .25 / snapback .57 (n=56)
  page_kind (a112): comply .53 / snapback .42 (n=19)
  median comply .36 vs snapback .46 — in-prompt deviant definition loses to the
  community meaning about half the time (H_spec predicts comply ~1).

E3-SCALE (median fm = rho_full − rho_blank over the 4 criteria):
  Llama family (sanctioned staircase): 3B .032 -> 8B .056 -> 70B .122 (monotone;
  3B/8B negative on a87/a104 — small models inject noise, not weaker signal)
  Unpooled replication points: Gemma-31B .207, GLM-4.7 .194, Qwen-122B .144, GLM-5.2 .133.
  (GLM-5.2 < GLM-4.7 on these 4 criteria; n=4, PR-only — noted, not interpreted.)

Caveats: single task, 4 criteria, ~100 items, within-extractor E1/E2. Full-scale
E1/E2 (4 tasks, 58.5k Gemma prompts) + E5 SEAM-POS (27k) + E3 3B/8B full re-extraction
(5 tasks) queued on sk3 GPU 1 queue2 jobs 200-224, launched 2026-07-05 ~20:34 PDT.

## §BATTERY-FULL — E1/E2/E3/E4/E5 at scale (sk3 GPU-1 queue, 2026-07-05 night)

All 23 queue jobs rc=0 (~3.5 h wall): Gemma battery 58.5k prompts (4 tasks) + SEAM-POS
27k + Llama-3B/8B field re-extraction (5 tasks) + E4 8B base/instr few-shot pairs (22k).
Raw-output spot-checks clean (no template leaks; base-model doc-continuation noise on
math noted). Evals: eval_key/eval_stip/eval_seampos/eval_scale ->
outputs/metric_seam_pilot/battery/{key,stip}_eval_<task>.json, seampos_eval.json, eval_scale.json.

E1-KEY medians per task (within-Gemma; frac = surviving fraction of fm):
  task      frac_name  frac_nonce  n_P95(name>nonce)/n
  PR          1.00       0.58        3/8
  CW          0.98       0.98        0/15
  math        0.75       1.00        1/15   (INVERTED: name-only LOSES signal, d_name .079)
  humor       0.92       0.86        3/15
  -> key-likeness is DOMAIN-GRADED: PR institutional constructs most key-like;
  CW definitions fully sufficient (name adds nothing); math names underdetermine
  (definition carries the content). Probe's clean 2/4 PR signature (a104/a112) sits at
  the key-like extreme, not the rule. Per-criterion spread is wide in every task
  (e.g. humor a333 P=.999 textbook key; CW a9 P=.91).

E2-STIP (Gemma, deviant stipulation, conflict-set readout):
  median compliance: PR .82 / CW .96 / math .68 / humor .92; snapback .00-.06;
  0 fields with snapback > compliance (n_fields 3-5 per task; ~7/10 candidate fields
  per task dropped — conflict set <15 or checker error).
  ** DIVERGES from local probe: GLM-4.7 on the SAME PR fields snapped back .46
  (a87 doc_type: Gemma comply .82/snap .18 vs GLM comply .36/snap .46). Same
  manipulation, opposite behavior by extractor family — stipulation-override
  ("semantic gravity") is an EXTRACTOR property, not a construct property. Gemma-31B
  executes the deviant spec; GLM-4.7 (reasoning-tuned) reverts to community meaning.
  Needs like-for-like replication before any claim (same conflict definition; add a
  3rd family; scale within family predicted by flipped-label ICL lit).

E5 SEAM-POS (PR certified 12):
  CLC aperture: median frac_kept(digest) .588 (range .02 [a87] - .96 [a97]);
  fm_R .117 -> fm_digest .039. Positional views uneven (a87 head .69 vs mid .29).
  A code-built digest keeps over half the field contribution on most criteria, but
  aperture-fragile criteria exist (a87 loses ~all).
  CCL valuation: LLM aggregator over code signals NEVER beats fitted ridge on same
  inputs (0/12 at P>=.95; median fm_A -.068; a87 ridge significantly better, P=.03).
  -> borrowed judgment localizes to the READ stage; no evidence of borrowed
  valuation at Aggregate.

E3-SCALE (fm-bearing criteria, gemma fm>=.10; Llama family only):
  task    3B     8B     70B    frac monotone(per-criterion)
  PR      .002   .073   .162   .63
  CW      .070   .023   .248   .23   (8B dip below 3B)
  math    .013   .034   .115   .29
  humor   .028   .112   .205   .47
  legal   -.001  .125   (70B llama extraction not yet run for legal)
  -> median staircase rises 3B->70B in all four complete tasks; per-criterion
  monotonicity noisy (median-level, not criterion-level, regularity).

E4-LOCUS (8B rung, identical few-shot completion prompts, base vs instruct):
  median fm: PR -.013/.041, CW .012/.033, math .000/.007, humor .022/.041.
  At 8B most of the (small) field signal appears only with instruction tuning;
  8B sits near the fm noise floor — decisive rung is 70B pair.
  Llama-3.1-70B BASE downloading on sk3 (~130GB, pid 717807); E4-70B jobs queue on landing.

## §CODIF — segment annotation of the 143-program fleet (R11, 2026-07-05 night)

Scheme: seam-position note §5 (C1-C8, decompression-rung-aligned). 12 Sonnet batches,
anchors a104(PR)/a42(math) blinded per batch. QC: every INDEPENDENT anchor read scored
7/8+ vs hand truth (no degenerate passes); modal tag agreement a104 .906 (n=4), a42 1.0
(n=6). METHOD INCIDENT: shared output dir let late batches read early batches' files —
8/12 batches copied or "harmonized to convention" their anchor rows (all self-reported;
one byte-identical). Copies excluded via NONINDEP list + signature detector
(codif_eval.py); native-program rows unaffected but conventions propagated across
batches (C4 boundary, "C8 never dominant"), so cross-batch consistency is partly
convention-following, not independent convergence. Next pass: isolated output dirs +
nonce-named anchors (memory updated).
NaN incident: eval_scale.json fm values NaN when rho_blank NaN (guard missed blank side);
NaN poisoned sorted() -> garbage medians on first codif_eval run (C4 fm_without ".607"
> global max = tell). Fixed both scripts; all medians below are post-fix.

Files: battery/codif/codif_merged.jsonl (143), codif_summary.json.

Tag prevalence (fraction of task's programs): C1/C7/C8 ~universal everywhere.
  C2 signifier-match: legal 1.00, PR 1.00, CW .95, humor .71, math .60
  C3 form-measure:    math .94, CW .89, humor .81, PR .80, LEGAL .00
  C4 placement:       math .71, humor .58, PR .55, CW .46, legal .25
  C5 extract-compute: PR .75, math .57, legal .55, humor .29, CW .22
  C6 exemplar-match:  humor .42, CW .08, PR .05, math .00, legal .00
  -> legal codifies with NO form measurement (all signifier+extraction+dates);
     exemplar-match (kNN) is essentially a humor phenomenon.

c8_share (how much of the score the LLM fields set):
  PR 1/20 HIGH (16 MED, 3 LOW)  — most code-carried fleet
  math 17/35 HIGH (+2 NONE, 7 LOW); CW 18/37 HIGH; legal 14/20 HIGH; humor 22/31 HIGH
  — matches the CAM ordering (PR most codable) from the opposite direction.

Outcome contrasts (medians, descriptive):
  C2 vs transport: programs WITH signifier-match have criterion transport ratio .256
    vs .706 without (lower=more transports) — lexicalized constructs are the ones whose
    borrowed fields survive family swap; direct code-side echo of the lexicalization story.
  C6: exemplar-match programs transport WORSE (.47 vs .277) — rule-less constructs are
    more culture-bound.
  C3: fm .153 with form-measure vs .258 without — where code can measure form, the
    field burden drops.
  frac_nonce by c8_share: HIGH .92 / MED .89 / LOW .80 (weak gradient, small LOW n).

Thick-predicate census (what the fields borrow, per task):
  PR: KIND 12 + GROUNDING 12 + THIN-EXTRACT 10 (classification + evidence-quoting)
  legal: GROUNDING 16 + THIN-EXTRACT 15 (fact-witnessing, near-zero aesthetics)
  CW: STRUCTURE-JUDGMENT 21 + CRAFT 9 + TONE 6 + the novelty/theme/stakes OTHERs
  humor: GROUNDING 17 + STRUCTURE-JUDGMENT 17 + STANCE/SEVERITY OTHERs
  math: STRUCTURE-JUDGMENT 11 + epistemic OTHERs (relevance/insight/calibration)
  -> the OTHER overflow is domain-diagnostic: evidence domains (PR/legal) fit the
  starter vocab; CW/humor overflow into aesthetic-stance predicates, math into
  epistemic ones. Scheme v3 should add NOVELTY-APTNESS, STANCE, RELEVANCE-SALIENCE,
  EPISTEMIC-CALIBRATION.

## §E7-PILOT — TF-IDF distill-the-field arm (label-clean redesign, run as PILOT overnight 2026-07-06)

Design per seam note §6 (selector trained on the certified field's OWN train outputs;
judge only in final rho). Near-categorical fields only (<=8 values, >=90% cover).
Script battery/e7_sel_pilot.py -> battery/e7_sel_pilot.json. RUN WITHOUT EXPLICIT
SIGN-OFF under the user's overnight "keep going" — flagged for morning veto; no
measurement target touched, no labels in training.

  task    n_fields  med_agree  med_frac_distilled
  PR         14       .64          .225
  CW         41       .63          .586
  math       24       .76          .534
  humor      30       .76          .633

Reading (descriptive): even where a TF-IDF selector reproduces field VALUES at .6-.76,
it keeps only ~22-63% of the field's program-level contribution — and several PR fields
go NEGATIVE when substituted (a42.doc_type -1.19, a87.quote_tone -.55, a119.doc_kind
-1.12): the errors the selector makes are exactly the items where the field earns its fm.
H_leak takes its first construct-level hit: fields are not bag-of-ngram shortcuts.
Surprise: PR (the most code-carried fleet per CODIF) has the LEAST distillable fields —
its few borrowed judgments are the most surface-irreducible; CW/humor fields half-distill,
possibly via topical lexical correlates in those corpora. Next (not run): BGE-embedding
selector arm; cross frac_distilled x CODIF tags x E6 transport (provenance grid §6.3).

## §E2-3FAM — stipulation-override, three extractors, same PR fields (2026-07-06 00:0x)

GLM-5.2 all-condition probe complete (2003/2019, 16 transient 529). Same fields, same
deviant stipulations, same conflict rule (checker-truth != model's own full answer):

  field            Gemma-31B        GLM-4.7          GLM-5.2
  a87  doc_type    .82 / .18        .36 / .46        .63 / .38     (comply / snapback)
  a104 doc_kind    .82 / .11        .25 / .57        .43 / .49
  a112 page_kind   .81 / .00        .53 / .42        (conflict<15)
  median (avail)   .82 / .04        .36 / .46        .63 / .49

Override capacity spans .25-.90 compliance ON THE SAME FIELDS: an extractor property
with big between- AND within-family variation (GLM-5.2 complies ~1.7x GLM-4.7's rate).
No simple scale story; Gemma-31B (smallest) complies most. Candidate driver to test:
reasoning-style post-training (GLM-4.7 re-derives the community concept and overrides
the stipulation) vs direct instruction-following — E2 on Qwen thinking-on vs off would
isolate it (same weights, toggle reasoning).

E1 addendum: construct key-likeness is EXTRACTOR-STABLE — a104 shows the textbook key
signature for Gemma AND GLM-4.7 AND GLM-5.2 (frac_nonce -.07/.06/.06, P=1.0 all), a76
is spec-like for all three (frac_nonce .78-.85). Key-vs-spec is a property of the
CONSTRUCT, replicated across 3 families; the E2 override is a property of the EXTRACTOR.
Clean double dissociation for the paper.

## §E2-TOGGLE — Qwen-122B thinking on/off, same weights, same stipulations (2026-07-06 ~01:30, PRELIMINARY)

Isolating run for §E2-3FAM's extractor-dependence. Scorer qwen_thinking_score_sk3.py
(post-</think> parse, unclosed flagged; Mamba fix max_num_seqs=512). 3 jobs rc=0.

AUDIT FIRST: 1007/2750 (37%) thinking-on rows hit the 1536-token cap without closing
</think> -> excluded as missing data (first-pass eval counted them "other" — garbage;
re-eval excludes). Re-run of ALL unclosed rows at 4096 tokens queued (jobs 310/311)
to kill the easy-case selection confound before any final claim.

On CLOSED rows (preliminary), compliance/snapback:
  field           toff           ton(closed)      n_ton
  a25             .14 / .45      1.00 / .00        25
  a42             .43 / .43      1.00 / .00        49
  a104            .43 / .44       .96 / .04        49
  a112            .54 / .41      1.00 / .00        16
  a87             .63 / .37       .15 / .00 (o=.85) 94   <- "other"=long-form answers,
                                                            string-match artifact, refine
Same weights, toggle reasoning: thinking-ON complies ~1.0 with the deviant stipulation;
thinking-OFF snaps back .37-.45. INVERTS my §E2-3FAM guess (reasoning != snap-back
driver): snap-back looks like a FAST-PATH DEFAULT to the enculturated meaning, and
deliberation is what enables executing an in-context spec that conflicts with it.
Fits flipped-label ICL findings (deliberation overrides priors). Also note Qwen-toff
snaps back at GLM-4.7 levels while Gemma complies .82 — so mode explains only part of
the extractor spread; caveat: GLM API modes are server-side and unauditable.
Pending: ton2 4k-token completion run -> final numbers + a87 fuzzy-match refinement.

### §E2-TOGGLE addendum — FINAL after 4k-token completion run (2026-07-06 ~02:55)

All 1007 unclosed rows re-run at 4096 tokens (jobs 310/311 rc=0); 471 STILL unclosed =
deliberative STALLS, not truncation. Stall rates: 24.2% under deviant stipulation vs
8.6% under normal instruction (2.8x) — the norm conflict is metabolically visible in
generation length. Stalled rows excluded as missing; fuzzy containment match added.

FINAL compliance/snapback (thinking-on n now 41-115/field):
  a25  toff .14/.46  ton .98/.00 (n=115)
  a42  toff .43/.43  ton 1.00/.00 (n=82)
  a104 toff .43/.44  ton .97/.03 (n=102)
  a112 toff .54/.41  ton 1.00/.00 (n=41)
  a87  toff .63/.37  ton .28/.00 other .72 (n=101) — TON answers match neither string
       even with containment; unresolved artifact or genuine third-category answers,
       needs eyeballing before use.
Headline unchanged and strengthened: same weights, deliberation ON -> executes the
deviant spec ~perfectly; OFF -> snaps back to enculturated meaning ~40%. Deliberation
is the spec-following enabler; snap-back is the fast-path default. Selection caveat
narrowed (stalls excluded on BOTH readouts; stall asymmetry itself now a reported
outcome, not silent missingness).

## §E7-FULL — BGE arm + provenance grid (2026-07-06 morning, E7 user-approved)

BGE arm (bge-small-en-v1.5 + logreg, laptop CPU) REPLICATES TF-IDF almost exactly:
med frac_distilled PR .225/.225, CW .541/.586, math .418/.534, humor .633/.633
(bge/tfidf). Surface-SEMANTIC recovers no more than surface-LEXICAL — the undistillable
residue is invisible to both surface representations, sharpening the anti-H_leak read.

Provenance grid (97 fields with distill=max(arms) x transport ratio; battery/e7_provenance_grid.json):
  CODIFIED-SURFACE (dist>=.5, ratio<=.5)  n=34   surface-living constructs
  ENCULTURATED     (dist<.5,  ratio<=.5)  n=25   shared, surface-irreducible — T-RET home
  IDIOSYNCRATIC    (dist<.5,  ratio>.5)   n=15
  OVERFIT-SURFACE  (dist>=.5, ratio>.5)   n=23   larger than the "rare" §6.3 guess
KEY NULL: Spearman(distill, transport)= -.011 (n=97); distill x frac_nonce -.087 (n=50).
Distillability, shareability, and key-likeness are (pairwise) near-orthogonal — the
provenance grid is genuinely 2D, not one "tacitness" axis renamed. Surprise: CRAFT (.75)
and GROUNDING (.75) most distillable; the OTHER novel predicates least (.42). Caveat:
frac_distilled=fm_S/fm_F is ratio-noisy at small fm_F; treat per-cell membership, not
rankings, as the stable readout.

## §KEY-CONCEPTS — per-metric key term + definition extraction (2026-07-06, AS-directed)

Sonnet fan-out over all 143 fleet criteria (improver_packs metadata), saved alongside
the metrics: <task outdir>/key_concepts.json, schema {key_term, term_source
community|coined, community_term, definition}; synced to sk3. Substrate for scaling the
E2-KIND grid to all concepts (name-material now available per criterion).

Lexicalization census (term_source=coined):
  humor 0/31, PR 1/20, legal 1/20, CW 2/37, MATH 7/35 (20%)
  -> humor's craft culture is fully lexicalized (punching up, rule of three, callback);
  math-WRITING norms often have no native term of art (agent borrowed BLUF/pinpoint-
  citation/Occam or coined; e.g. "formula run-on"). Converges with E1's math inversion
  (names underdetermine; definitions carry) and the tacit-stream's codified-not-
  lexicalized (MECH) cell. term_source now joins tags as a standing Face-2 covariate.
  Extraction agents also flagged construct-duplicate families within tasks (CW rhythm
  x3, humor rule-of-three x2, ethics-of-harm x3; legal a5/a41 nested) — same-key-term
  criteria should show CORRELATED name-gravity in E2-KIND; free replication structure.

### E2 CORRECTION (2026-07-06 pm) — checker-eval builtins bug, all E2 numbers re-issued

The E2-KIND variant author discovered that eval'ing checker_exprs with
{"__builtins__": {}} silently raises NameError wherever an expr's evaluation path
touches len/any -> those items were skipped by try/except. Bias audit: a112__page_kind
55% of items skipped, a42__doc_type 32%, a87__quote_tone 100% (why it "errored"),
a104/a25/a87__doc_type clean. Fixed with SAFE_BUILTINS in eval_stip/probe_eval/
eval_qwen_toggle; corrected numbers (all conclusions SURVIVE, most strengthen):
  Gemma (6/6 fields now): median comply .819 / snap .065; quote_tone 1.0/.0.
  GLM-4.7: median comply .359 / snap .554 (snapback UP from .46).
  GLM-5.2: median comply .625 / snap .375.
  Qwen toggle: toff snapback now .37-.59 (a112 .506, a42 .585); ton comply .97-1.0
  unchanged. Toggle contrast STRONGER.
New observation: quote_tone's deviant rule is complied with by EVERY extractor
(Gemma 1.0, GLM-5.2 .99, GLM-4.7 .82) — TONE-REGISTER constructs may exert weaker
gravity than KIND/genre constructs; conflict-kind salience varies by thick-predicate
type (feeds E2-KIND analysis).

## §E2-KIND — nonce+deviant grid, deficit control (2026-07-06)

Design (seam note §7): {name, nonce} x {deviant X', neutral fresh-label X''} per
e2-bearing PR field. cell4 = nonce+deviant (KEY), cell5/6 = neutral-rule execution
controls (kills the instruction-following-deficit confound). gravity_effect =
acc(cell6) − comply(cell4): same nonce framing, only the rule's relation to the
community concept changes. 5 fields graded (a87__quote_tone never reaches n_conflict
≥15 — no gradable conflict set); qwen thinking-ON still generating (harvest pending).

Medians over gradable cells (per-cell n_conflict 29–191):

| extractor | exec5 name+neut | exec6 nonce+neut | comply4 | phantom4 | gravity |
|---|---|---|---|---|---|
| gemma-31B  | .980 | .980 | .865 | .052 | .101 |
| llama-3B   | .472 | .532 | .286 | .384 | −.046 |
| llama-8B   | .972 | .972 | .273 | .186 | .567 |
| llama-70B  | .876 | .864 | .460 | .340 | .404 |
| qwen-toff  | .916 | .912 | .343 | .327 | .657 |

Readings (descriptive; single task, 5 fields):
1. DEFICIT CONFOUND REMOVED at ≥8B: neutral fresh-label rules execute at .86–1.0
   for 8B/70B/qwen/gemma. Only 3B genuinely can't execute (.47–.53), so its cell4 is
   uninterpretable — exactly the confound the user flagged; cells 5/6 earn their keep.
2. SEMANTIC GRAVITY is real and family-dependent: swapping neutral→deviant under the
   SAME nonce framing costs llama-8B .57 median compliance, qwen-toff .66, llama-70B
   .40 — Gemma only .10. Matches §E2-3FAM extractor-property finding on a harder
   control (community name fully absent).
3. PHANTOM-SNAP WITHOUT THE NAME (new): models emit their community-concept answer
   under nonce+deviant framing — llama-8B a42 .986, qwen a25 .759, llama-70B a25
   .612 (Gemma .052). The concept reasserts itself with no lexical trigger, through
   document + answer-vocabulary shape alone.
4. nonce_locus covariate: the extreme phantom cells are label-locus (nonce replaces
   the OUTPUT labels): a42/a104 phantom .72–.99 in 4 of 8 non-Gemma label cells;
   question_term cells drift to "other" instead. Gravity strongest where the model
   must emit the community's own answer vocabulary.
5. Llama staircase: comply 8B .273 → 70B .460 (gravity .567 → .404): capacity to
   HOLD a deviant rule grows with scale but 70B stays far below Gemma .865.

-> battery/e2kind_eval.json. E4-70B instruct done (all 4 tasks, expected counts);
eval_scale running — §E4 addendum next. Lit-positioning: full factorial name x rule
+ deficit controls + (pending) reasoning toggle in an EXTRACTION setting = the open
cell vs Verbalizer-Manipulation/MAGNIFICo/WinoDict (§7.1 dedup).

## §E4-70B addendum — LOCUS at the decisive rung (2026-07-06)

Both 70B arms done (Llama-3.1-70B base, Llama-3.3-70B-Instruct), same few-shot
completion prompts as the 8B pair (format controlled). Median fm = rho_cond −
rho_blank per task (completion mode; chat-mode llama70 column for reference):

| task | 8B base | 8B instr | 70B base | 70B instr | 70B chat ref |
|---|---|---|---|---|---|
| PR    | −.013 | .041 | .006 | .063 | .077 |
| CW    | .012  | .033 | .106 | .119 | .174 |
| math  | .000  | .007 | .000 | .025 | .077 |
| humor | .022  | .041 | .083 | .139 | .108 |

Readings (descriptive):
1. Instruct > base at BOTH scales, all 4 tasks (8/8 comparisons) — the tuned
   layer carries real extraction capacity even at 70B.
2. DOMAIN ASYMMETRY in where capacity lives: base-model capacity grows with
   scale for the taste domains — CW 70B base .106 = 89% of its instruct arm,
   humor .083 = 60% — but stays ≈0 for PR (.006) and math (.000) even at 70B.
   For CW/humor the field signal is largely in the pretrained distribution
   (the community's own texts); for PR/math extraction the capacity is almost
   entirely from the instruction-tuned layer.
3. Completion-mode fm sits below chat-mode fm throughout except humor 70B
   instruct (.139 vs .108) — prompt-format cost is real; base-vs-instruct
   contrasts above are within-format and unaffected.
Caveat: math completion-mode fms are tiny overall (many criteria at 0), so its
row mostly says "completion prompting fails for math fields," not locus.

-> battery/eval_scale.json (rev with e4_70bbase/e4_70binstr columns).

## §AGENTIC-COMPILE — held-out certification of the 13-criterion fleet (2026-07-06)

Design per seam note §8: one Sonnet agent per tail criterion, <=6 reflective rounds
against TRAIN residuals, LLM fields FROZEN byte-identical to h0 (audited all 13),
no dpid hacks (audited), test split untouched until this pass. Certification =
same G1 gate machinery as the h0 fleet (paired bootstrap vs frozen codegen
baseline, B=2000). battery/agentic_cert.json.

| criterion | Δrho TRAIN | Δrho TEST | P(c>h0) | gap(cand) | note |
|---|---|---|---|---|---|
| math a198   | +.195 | **+.079** | .83 | +.108 | (n)-backref bug fix; P_gate .43→.78 |
| math a42    | +.264 | **+.062** | .90 | +.070 | digit-proxy deletion; P_gate .19→.44 |
| math a132   | +.133 | −.030 | .30 | +.049 | tier recalibration OVERFIT — inverted |
| humor a135  | +.159 | **+.075** | 1.00 | +.062 | substring-bug fixes; P_gate .80→.98 CERT-FLIP |
| humor a153  | +.125 | **+.077** | .95 | −.030 | pun-severity gate; P_gate .56→.90; test>train |
| humor a351  | +.068 | +.012 | .78 | −.011 | field-dom; small real gain |
| PR a119     | +.013 | −.020 | .14 | −.006 | train gain gone; h0 generalized UP (.80→.84) |
| PR a115     | +.024 | −.002 | .50 | +.122 | worst winner's curse |
| PR a87      | +.096 | −.017 | .18 |  .000 | field-dom recode: ALL of +.096 was train-fit |
| CW a90      | +.013 | −.012 | .09 | +.116 | |
| CW a72      | +.033 | −.007 | .32 | +.050 | |
| CW a99      | +.030 | −.013 | .33 | +.019 | |
| CW a342     | +.002 |  .000 | .00 | −.040 | |

Readings:
1. THE BOUNDARY HOLDS. Survivors (4/13, all P(c>h0) >= .83) are exactly the
   objective-code-defect fixes: substring/regex/ordering bugs + deletion of a
   saturating proxy (math a198/a42, humor a135/a153). Every calibration-type gain
   (retuned weights, bucket-mean remaps, tier boundaries: a132, a87, a115, CW all)
   EVAPORATED or inverted on held-out. Bug fixes transfer; recalibration is
   winner's curse.
2. Gate movement: 1 strict cert-flip (a135 crosses .95), a198 flips at .5
   (.43→.78), a153 .56→.90. So flexible compilation moves the code-sufficiency
   boundary by ~1-2 criteria out of 10 gate-fail attempts — a few points, not a
   re-partition. Field-dominated share = domain fact, not compiler artifact
   (a87 the sharpest: +.096 train → −.017 test, gap literally .000 vs h0's
   train-underfit .700→.813).
3. Winner's-curse gaps concentrate where deltas died (a115 +.122, a90 +.116)
   and are ~0/negative where gains were real (a153 −.030, a351 −.011) — the
   gap statistic works as the §8 diagnostic predicted.
4. Fleet-hygiene payoff regardless: the substring bug class (spic/spice,
   cock/cockpit, punch\w*/punchline, descend\w*/descendants) is confirmed in
   4 h0s across humor+CW; h0 baselines there were systematically depressed.
   Sweep remaining h0s for \w*-suffix and bare-substring matching.

### §E2-KIND addendum — qwen thinking-ON (2026-07-06 pm)

Same weights as qwen_toff row above; 4k think budget; unclosed-think rows dropped
(missing data, not verdicts).

| extractor | exec5 | exec6 | comply4 | phantom4 | gravity |
|---|---|---|---|---|---|
| qwen_toff | .916 | .912 | .343 | .327 | .657 |
| qwen_ton  | 1.0  | 1.0  | **.964–1.0 (med 1.0)** | **.000** | .000 |

1. TOGGLE INVERSION SURVIVES — AND SHARPENS — UNDER NONCE NAMING: thinking-on
   complies perfectly with the deviant rule in every gradable cell (comply .964,
   .991, 1.0, 1.0; phantom 0.0), same weights that phantom-snap .327 with
   thinking off. Deliberation fully brackets semantic gravity; habit does not.
2. STALLS TRACK CONFLICT, NOT LOAD: unclosed-think = 14.2% on cell4
   (nonce+deviant) vs 3.7%/4.5% on the neutral-rule cells — and 24.2% on the
   original NAME+deviant stip set. Gradient 24% (name+deviant) → 14% (nonce+
   deviant) → 4% (neutral): deliberative stalls scale with the strength of the
   community-concept trigger, the lexical trigger being the strongest. The stall
   is the cost of holding a rule against gravity; remove the name and half the
   cost disappears.
3. Deficit control also perfect (exec5/6 = 1.0) — ton's cell4 compliance is
   rule-execution, not chance.

## §GEPA-H2H — direct-prompt arm vs certified hybrid, held-out (2026-07-06)

Setup (seam note §9; harness battery/gepa_h2h/): arm G = single Gemma-31B scoring
prompt (0-10), GEPA loop = 3 dev-scored rounds (seed + 2 revisions), GLM-5.2
proposer 24 calls total, dev = 40 fixed TRAIN items/criterion, judge scores never
in any LLM prompt (rank-residual feedback only). Arm H = certified hybrid
(max(cand,h0) from agentic_cert.json, recomputed same split). Frozen argmax-dev
prompt per criterion evaluated once on TEST (n≈100).

| criterion | G test | H test | P(G>H) | GEPA best round |
|---|---|---|---|---|
| PR a119 | .912 | .840 | .98 | r2 (seed .794→.966 dev, real GEPA lift) |
| PR a115 | .910 | .749 | 1.0 | r2 ≈ seed |
| PR a87 | .950 | .813 | 1.0 | r0 seed |
| CW a90 | .962 | .740 | 1.0 | r0 |
| CW a72 | .927 | .678 | 1.0 | r0 |
| CW a99 | .848 | .682 | 1.0 | r2 ≈ seed |
| CW a342 | .975 | .606 | 1.0 | r0 |
| math a198 | .877 | .676 | 1.0 | r0 (GEPA r1 INVERTED construct, dev −.10) |
| math a42 | .885 | .612 | 1.0 | r0 |
| humor a351 | .924 | .776 | 1.0 | r1 (real lift .766→.869 dev) |
| humor a153 | .902 | .688 | 1.0 | r0 |
| humor a135 | **.487** | **.735** | **.004** | r0 (GEPA collapsed r1 all-0s, r2 dev −.18) |

Readings:
1. STRUCTURAL, NOT A CONTEST: judge = Gemma-31B two-pass mean; arm G is (nearly)
   re-running the judge, so G sits at/near the judge-noise ceiling everywhere
   (.85–.98). The G−H gap is therefore the measured FIDELITY COST OF COMPRESSION
   into code + <=2 typed fields: median ≈ .19 rho (range .07 PR a119 → .37 CW
   a342). "Hybrid beats GEPA('s bound)" is not claimable and never was — the
   right axis for arm H is typedness/determinism/auditability/transport at a
   quantified fidelity discount.
2. Compression cost is domain-graded, largest exactly where code-sufficiency is
   lowest (CW .17–.37) and smallest on the most bureaucratic criteria (PR a119
   .07) — §IV gradient again.
3. GEPA itself barely matters at this ceiling: argmax-dev = the SEED prompt for
   8/12 criteria; real lifts only where seed underperformed (a119 +.17, a351
   +.10 dev); and the loop is NON-MONOTONE and construct-unsafe — one reflective
   rewrite collapsed a135 to constant-0 scoring and inverted a198 (dev −.10);
   raw rank-residual feedback alone does not protect the construct. Our gate
   machinery catches exactly this failure class; GEPA has no equivalent.
4. The one H win (a135, P(G>H)=.004): the criterion where the judge construct is
   most gameable-by-elaboration (platform standards). Monolithic prompting is
   not uniformly dominant even at matched executor.
5. Cost column: G = 1 full-doc call; H = code + 1-2 short extraction calls
   (reusable across all criteria sharing a field). Comparable inference; H's
   marginal cost per additional criterion is near zero once fields exist.

GLM spend: 24 calls. GPU 1 released (queue STOP). -> gepa_h2h/gepa_h2h_final.json

### §E2 checker-infrastructure closure (2026-07-06 pm)

Two harness defects found while scaling E2-KIND (CW variant-author caught #1):
1. GENEXPR NAMESPACE BUG: `eval(expr, {builtins}, {"text": text})` breaks any
   checker using a generator expression (`any(w in text for w in [...])`) — the
   genexpr frame can't see locals-passed `text` → NameError → item silently
   skipped. 16/36 e2 checkers across the 4 tasks are genexpr-based. FIXED in all
   4 harnesses (text moved into eval globals). No harvested number changes: the
   only affected PR field was a87__quote_tone (excluded from all medians); the
   15 CW/math/humor genexpr checkers had never been run.
2. a87__quote_tone mystery CLOSED: with the namespace bug fixed, its deviant
   checker turns out to be CONSTANT ('boastful' on 300/300 docs) → conflict set
   empty by construction → ungradable under any extractor. Was masked by #1.
   Lesson institutionalized in the scale-up authoring spec: checkers must emit
   ≥2 distinct values over 100 real docs (validated per field, recorded in
   _validation keys).

## §HYGIENE — fleet-wide substring-bug sweep + recertification (2026-07-06 eve)

Sweep: AST+corpus-grounded detector (battery/detect_substring_bugs.py) over all 143
h0/h1 + 13 agentic programs -> 684 flags; 4 Sonnet triage agents verified each flag
BY EXECUTION (not trusting the static class): 165 TRUE_BUG patched across 49 programs
(PR 36 / CW 104 / humor 19 / math 1 / legal 5), ~360 INTENTIONAL_STEM, ~200 scanner
false-alarms (anchored regexes, set-membership `in`). Patches = \b-anchored inflection
whitelists in programs_hygiene/ (originals untouched); manifest audited — zero
cross-task clobbering. Greatest hits: "wit"⊂with (237 docs), "fortunate"⊂unfortunately
(sentiment inversion), "story"⊂history, "mission"⊂emissions, "asian"⊂Caucasian (Title
VII race-tagging — patched incl. bonus finding), "cock"⊂cockpit, "punch"⊂punchline.

RECERTIFICATION (held-out, all 48 gradable patched programs, battery/hygiene_cert.json):
median Δrho_test = 0.000, range [−.006, +.012]; only mover humor a135 +.012 w/ gate
P .80→.84; no gate flips anywhere; a105 ungradable (v1-era, no judge channel).

Reading — BASELINE VALIDITY RESOLVED, and a finding in its own right:
1. The bug class is REAL at item level (spice-rack jokes tagged as slur-bearing;
   plaintiff "Caucasian" tagged asian) but has ~ZERO aggregate rank impact at
   observed incidence: metric-level Spearman is robust to lexical false positives
   at these rates. Published h0-anchored numbers (code-sufficiency census, agentic
   deltas, GEPA gap) stand unchanged; my earlier "humor h0 baselines depressed"
   worry was wrong on held-out.
2. Decomposition for the agentic story: of humor a135's certified +.075, only
   ~+.012 is the substring bugs — the rest was severity-tier recalibration. The
   agentic gains were restructuring, more than bug repair, on test.
3. TRAIN deltas the patch agents measured (e.g. a153_h1 wit-fix −.070 train) mostly
   vanish on test (0.000) — one more instance of train-side movement being unreliable.
4. Construct-validity note for the paper: fixes are kept regardless (a metric that
   calls "spice rack" a slur is wrong even if its rank order survives); rank
   robustness is why the gates never caught this class.

## §E2-KIND-SCALE (preliminary) — 4 domains, 26 gradable fields (2026-07-06 eve)

Scale-up per seam note §7: 30 new checkable fields authored (CW 10 / math 5 of 10,
5 skipped for corpus-degenerate e2 checkers / humor 10), leak-audited (10 cell4
concept-name leaks found by the builder guard and hand-fixed, incl. label renames
where the answer label WAS the concept stem: a180 EARNED→TRENVIK, a9hm
MITIGATED→BREVOLE). 15/18 extraction jobs harvested; qwen_ton (3) still running —
ton rows + own-answer-proxy caveat to follow.

Deficit-CONDITIONED medians (cells with exec6 >= .7 only; n_ok = gradable cells):

| task | ext | n_ok | comply | phantom | gravity |
|---|---|---|---|---|---|
| PR    | gemma | 5 | .865 | .052 | .101 |
| PR    | llama70 | 3 | .460 | .340 | .457 |
| PR    | qwen_toff | 4 | .343 | .627 | .661 |
| CW    | gemma | 10 | .972 | .033 | .023 |
| CW    | llama70 | 8 | .688 | .165 | .406 |
| CW    | qwen_toff | 9 | .428 | .202 | .364 |
| math  | gemma | 4 | .765 | .409 | .292 |
| math  | llama70 | 2 | .800 | .836* | .598 |
| math  | qwen_toff | 1 | .446 | .385 | .314 |
| humor | gemma | 7 | .905 | .034 | .021 |
| humor | llama70 | 4 | .844 | .479 | .371 |
| humor | qwen_toff | 4 | .767 | .600 | .516 |
(*math llama70: only 2 cells clear the exec bar — cell-level, not robust)

Readings (PRELIMINARY, ton pending):
1. GEMMA RULE-FOLLOWER REPLICATES 4/4 domains: comply .77–.97, gravity ≈ 0
   (.02–.10) everywhere. The certified extractor treats stipulated rules as
   autonomous across every domain we have.
2. SEMANTIC GRAVITY REPLICATES for Llama/Qwen in all 4 domains (gravity .29–.66
   where exec-conditioned), but the magnitude ordering is NOT a simple
   taste-gradient: math shows the HEAVIEST phantom-snap for big Llama (.836) —
   models re-derive community answers (completeness/on-topic judgments) from
   content despite deviant definitions. Tension with E1 (math = definitions
   suffice): definitions are USED when they match the community concept,
   OVERRIDDEN when they conflict with it — the definitional domain respects
   definitions conditionally, not unconditionally.
3. DEFICIT CONTROL BITES HARDER AT SCALE-UP: many neutral rules (char-count
   thresholds) are themselves hard to execute (CW 8B exec5 .33) — gravity is
   only interpretable exec-conditioned; n_ok column is load-bearing. Rule-
   executability varies by rule TYPE (label-reading easy, counting hard) —
   worth standardizing neutral-rule types if we extend again.
4. Coverage: 26 exec-gradable of 31 authored fields; math thin (5 fields, 1-4
   clearing exec bar per extractor) — math rows are directional only.

### §E2-KIND-SCALE addendum — qwen thinking-ON, CW + math + humor (2026-07-06 night)

(ton own-answers proxied from qwen_toff — same weights, different mode.)

| task | toff comply/phantom/gravity | TON comply/phantom/gravity |
|---|---|---|
| CW    | .428 / .202 / .364 | **1.0 / .000 / .004** (10/10 fields, exec 1.0) |
| math  | .446 / .385 / .314 | **1.0 / .000 / .000** (5/5 fields, exec .98+) |
| humor | .767 / .600 / .516 | **1.0 / .000 / .000** (3 gradable c4, exec ~1.0) |

1. TOGGLE INVERSION REPLICATES 4/4 domains (PR, CW, math, humor): deliberation
   fully brackets semantic gravity in every domain, including math where big
   Llama phantom-snaps hardest (.836). Humor gives the cleanest field-level
   flip: a36__padding_verdict toff comply .117 / phantom .883 (heaviest snap
   in humor) → ton **1.0 / .000**; a324__resolution_mode .767→1.0;
   a288__pattern_break .907→.959. Weakest ton cells fleet-wide remain the CW
   both-locus craft fields (a234 prose_craft .872, a72 mode .860) — even
   deliberation retains a whisper of gravity on thick craft vocabulary, the
   only sub-1.0 ton compliance anywhere (humor's 3 gradable are all ≥ .959).
2. STALL-GRADIENT AMENDMENT: the PR finding "stalls track conflict" holds only
   at MATCHED rule difficulty. Scale-up cells: CW stalls flat (~14-18% all
   cells), math INVERTED (cell4 5% vs neutral 29-33%) — the scale-up's neutral
   rules are counting rules that thinking-mode genuinely tries to compute.
   Humor is the CLEANEST conflict-stall case yet: cell4 **39.2%** vs cell5
   3.4% / cell6 3.3% — its neutral rules are easy (label-reading), so the
   conflict term shows undiluted, steeper than PR's 24%→4%. Stall =
   f(conflict, computational difficulty), each term now isolated in at least
   one domain (humor isolates conflict, math isolates difficulty).
3. Humor ton failure mode is stall-not-snap: 39% of cell4 attempts never close
   thinking; a9__harm_mitigated loses c4 gradability under ton (toff n=55 →
   ton n<15) because stalls decimate the conflict set. Under deliberation the
   model either follows the deviant rule perfectly or declines to answer —
   discretion suppressed, dissent expressed as silence.
4. sk3 note: root disk at 85% (353G/438G) — one /tmp git-clone present; not
   urgent, watch on next login failure.

### §CROSSFAM — cross-family gate recertification, CW + humor certified sets (2026-07-06 night)

Closes the fleet-wide swap question: PR's 12 certified gates were swap-tested in
§TRANSPORT (3/12 fail); this re-runs the G1 gate for the CW 5 + humor 4 certified
criteria with LLM_FIELDS swapped to Llama-70B / Qwen-122B extractions (frozen
programs, splits, judge, train-best codegen baseline). Script
battery/cert_crossfam.py → battery/crossfam_cert.json. Legal untested (no swap
extractions — known thin spot). Gemma column reproduces the published gates
exactly (a144 1.0, a351 .9875, a72 .884, a99 .9085 — validates the harness).

| criterion | gemma ρ / P_gate | llama ρ / P_gate | qwen ρ / P_gate |
|---|---|---|---|
| CW a144 | .842 / **1.00** | .660 / .81 | .834 / **1.00** |
| CW a72  | .678 / .88 | .592 / .44 | .700 / **.94** |
| CW a99  | .682 / .91 | .391 / **.008** | .604 / .52 |
| CW a90  | .740 / .69 | .723 / .59 | .712 / .52 |
| CW a342 | .606 / .54 | .463 / .03 | .559 / .28 |
| humor a351 | .764 / **.99** | .616 / .61 | .596 / .50 |
| humor a135 | .660 / .80 | .721 / **.97** | .595 / .49 |
| humor a153 | .611 / .56 | .284 / **.0005** | .539 / .20 |
| humor a81  | .609 / .57 | .502 / .14 | .428 / .05 |

Readings:
1. TASTE-POLE CERTIFICATES ARE MORE EXTRACTOR-BOUND THAN PR's. Of the two
   strong (P≥.95) taste-pole gates, CW a144 survives 1/2 families (qwen
   lossless at 1.00, llama drops to .81) and humor a351 survives 0/2
   (.61/.50). PR by contrast kept 9/12 under swap. The field-extractor-family
   coordinate in the certificate stamp matters MOST exactly where the fields
   carry enculturated craft payload — consistent with §TRANSPORT's fm↔td
   coupling and the E1/E7 story.
2. Family asymmetry is criterion-level, not uniform: qwen preserves CW
   (a144 1.00, a72 IMPROVES to .94) but is the weaker family for humor
   (all 4 ≤ .50); llama flips humor a135 UP to .969 while collapsing CW a99
   (.008) and humor a153 (.0005, ρ .61→.28). Echoes E6: boundness is a
   criterion property.
3. Marginal gemma gates (a90 .69, a342 .54, a153 .56, a81 .57) stay marginal
   or fail under swap — no swap rescues a below-bar gate except a135 (llama).
4. Paper wording: quote taste-pole certificates ONLY with the 4-tuple stamp
   (criterion, judge-family, executor, extractor-family); "certified" without
   the extractor coordinate overclaims at the taste pole specifically.

### §R20 — SIV compression waterfall (figure/consolidation pass, 2026-07-06 night)

`battery/fig_s4_waterfall.py` -> `figures/s4_compression_waterfall.{png,pdf}` +
`battery/s4_waterfall.json`. Rungs per GEPA-H2H criterion (held-out test): frozen
codegen baseline -> certified hybrid HEAD (H; * = agentic candidate) -> GEPA
single prompt (G, ~= re-running the judge) -> attenuation ceiling. Panel B is the
ceiling-normalized (r~ = clip01(rho/ceiling)) domain summary — the SIV gradient
as one picture.

| domain | med r~ code | med r~ hybrid | med r~ prompt | fidelity cost (r~_G − r~_H) |
|---|---|---|---|---|
| PR    | .399 | .825 | .934 | **.109** |
| humor | .306 | .747 | .921 | **.174** |
| math  | .543 | .733 | .984 | **.251** |
| CW    | .140 | .700 | .972 | **.272** |

Readings:
1. The prompt rung is ~flat (.92–.98 everywhere): a single free-text prompt
   recovers the judge at its noise ceiling in every domain. The DOMAIN
   information is in the hybrid rung — what survives compression into
   code + ≤2 typed fields.
2. Ceiling normalization REORDERS the gradient: on raw rho, math's gap looked
   small (~.20 vs CW .17–.37) because its ceilings are low (.849–.914);
   normalized, math's fidelity cost (.251) sits just under CW's (.272) and
   well above PR's (.109). Quote the NORMALIZED gradient for SIV: CW ≈ math >
   humor > PR. Caveats: math n=2; humor median hides the a135 inversion
   (G .487 < H .735 — the one criterion where compression BEAT the prompt).
3. The code rung tells the Daston story on its own: what doctrine/convention
   pre-compiled (PR .399, math .543 — templates and notation) vs what craft
   never wrote down (CW .140). The hybrid rung then shows the LLM-field
   channel buying .35–.56 of ceiling — the seam is where the thick residue
   concentrates.

### §E2-KIND-SCALE math-ext addendum — 5 re-authored fields (R18, 2026-07-07)

The 5 math fields skipped in the scale-up (corpus-degenerate original e2 checkers)
were re-authored with fresh deviant rules + a per-field community_checker_expr
(eval reads it in place of the dead original), validated non-degenerate on the
real corpus (cell4 minority 24–44%, sim-conflict 61–161) before any GPU. Now run
across all 6 extractors. Fields: a150 engagement, a168 claim_quality, a24
is_complete, a42 grounds_with_example, a60 lead_pattern.

| extractor | comply (med, new-5) | phantom (med) | verdict |
|---|---|---|---|
| gemma     | .852 | .148 | RULE-FOLLOWER (4/5; a60 partial) |
| llama70   | .571 | .429 | SEMANTIC GRAVITY |
| qwen_toff | .728 | .272 | mixed (a24 heavy snap) |
| qwen_ton  | **1.000** | **.000** | TOGGLE INVERSION |
| llama3b/8b | deficit (exec .3–.7) | — | instruction-following deficit |

All three main-run findings REPLICATE on the previously-dead fields:
1. Gemma rule-follower (comply .85, gravity med .035); Llama-70B phantom-snaps
   (comply .571 / phantom .429); qwen thinking-ON snaps to perfect compliance
   (1.0 / .000, exec ~1.0) — 5th math confirmation of the toggle inversion, now
   on fields that carry a genuine community concept rather than counting rules.
2. ★ NEW TEXTURE — the deviant's THICKNESS is graded by community-concept pull,
   visible field-by-field:
   - a150 (deviant = "\frac present", a pure surface property with no concept to
     retrieve) is FOLLOWED even when deviant by every non-thinking extractor
     (gemma .949, llama70 .937, qwen .911) — a THIN deviant, near-zero gravity.
   - a24 is_complete (deviant collides with the retrievable "Therefore/Thus"
     completeness concept) phantom-snaps qwen_toff hard (.234 comply / .766
     phantom) and pulls llama70 to .571 — a THICK deviant.
   - a60 lead_pattern (deviant collides with length-quartile concept) snaps even
     GEMMA (.448 comply / .552 phantom) — the one field where the rule-follower
     retrieves the concept, because the community label is itself computed from a
     salient surface feature (length).
   This sharpens the main-run reading "definitions used when matching the
   community concept, overridden when conflicting": overriding strength scales
   with how retrievable the colliding concept is. Surface-mechanical deviants are
   thin (followed); concept-colliding deviants are thick (snapped), gradient
   running deeper into the model the thicker the collision.
3. No thick-craft ton-whisper cells here (all math qwen_ton = 1.0/.000) —
   consistent: math has no craft vocabulary; the sub-1.0 ton cells remain
   CW-only (a234 prose_craft, a72 mode). Deliberation fully brackets even the
   thick concept-colliding deviants.

### §CROSSFAM legal rows (R19, 2026-07-07) — the SPECIFIED pole is swap-robust

Extended cert_crossfam.py to legal's 9 certified gates under Llama-70B / Qwen-122B
field extractions (gemma column reproduces the published §R7.3 gates exactly).

| criterion | gemma ρ / P_gate | llama ρ / P_gate | qwen ρ / P_gate |
|---|---|---|---|
| a44 federal-sector flag | .769 / **.99** | .741 / **.96** | .757 / **.98** |
| a46 | .777 / **.99** | .605 / .47 | .695 / .87 |
| a39 | .728 / **.97** | .641 / .66 | .680 / .84 |
| a15 | .783 / **.96** | .783 / **.95** | .691 / .76 |
| a0  | .715 / **.94** | .732 / **.96** | .741 / **.98** |
| a23 exhaustion | .825 / **.94** | .809 / .85 | .785 / .81 |
| a18 | .730 / .90 | .657 / .60 | .593 / .23 |
| a36 | .693 / .83 | .435 / .27 | .435 / .28 |
| a5  | .630 / .59 | .607 / .53 | .455 / .08 |

Reading — legal certificates are the LEAST extractor-bound of the taste-to-doctrine
span. Of the 6 strong gemma gates (P≥.95): a44 and a0 survive BOTH family swaps at
P≥.96; a15 survives llama; a46/a39/a23 stay high (ρ .60–.81) though under the .95
bar. Contrast: PR kept 9/12, CW a144 1/2, humor a351 0/2. The certified content at
the specified pole = doctrinal ELEMENTS (federal-sector status, exhaustion dates,
15-employee facts) whose extraction is a near-invariant text property — so the
field-extractor coordinate barely moves the gate. The two marginal gemma gates
(a36 .83, a5 .59) fail under swap, as at every pole. Fleet ordering of certificate
extractor-boundness now: taste (CW/humor) ≫ math > PR > law. The 4-tuple stamp
(criterion, judge-family, executor, extractor-family) is load-bearing at the taste
pole and nearly slack at the doctrine pole — a graded property, matching the
transport-ratio and E6 story. Legal removes the last single-family certificate set.

### §R19 legal judge-rep + transport (2026-07-07) — the doctrine pole closes

Second judge (Llama-70B, results_llama.jsonl) + legal transport ratios, closing
the last single-family / single-judge thin spot.

JUDGE-FAMILY REPLICATION (legal_judge_rep.py → legal_judge_rep.json):
- Cross-judge agreement (Gemma vs Llama-70B judge, per aspect): median raw
  Spearman **0.71**, disattenuated **0.79** — legal item-level judgments are
  judge-family-robust (doctrine is the specified pole).
- Codability-ordering replication: Spearman(code-ρ|gemma, code-ρ|llama) over 20
  aspects = **0.591** — which legal criteria are more codable replicates across
  judges (weaker than PR's ~.93 but clearly positive).
- Gate replication under the Llama judge: **1/9 at P≥.95** (a46 .9955); a23 .89,
  a0/a15 .77, a39 .70 hold moderately; a44 DROPS to .55. Same phenomenon as PR
  a110 / the Night-2 Llama replication: absolute gates are (criterion,
  judge-family) statements, and the stamp already says so. ★ a44 dissociates —
  field-swap-ROBUST (§CROSSFAM llama .96/qwen .98) but judge-swap-FRAGILE (.55):
  extractor-boundness ⊥ judge-boundness, two independent certificate coordinates.

TRANSPORT / E6 (legal_transport.py → tasks/legal_title_vii/transport_eval_3fam.json):
- median ratio (fraction of field signal LOST under swap): llama **.134**, qwen
  **.336** — legal fields TRANSPORT WELL (66–87% survives), the portable/specified
  signature, consistent with the doctrine-pole reading.
- Spearman(fm, td_llama) = **+0.366** — loss tracks borrowed-meaning magnitude,
  replicating the main transport prediction.
- ★ E6 Spearman(ratio_llama, ratio_qwen) = **+0.473** (n=17) — boundness is
  criterion-level in legal too; legal is the 5th task confirming E6 (pooled was
  .295). Legal now populates the E6 panel and (per §11.1) the E8 P1/P2 test.

### §FLEET-BOUNDNESS — cross-family check scaled to ALL metrics (2026-07-07)

User ask: scale the checks to all corpora + all metrics. crossfam_cert.py reported
only the ~30 CERTIFIED gates; the transport_eval_3fam.json files already carry the
family-swap degradation readout (ratio = td/fm = fraction of field signal LOST;
P_degrade) for EVERY gradable hybrid aspect. consolidate_boundness.py rolls all 5
corpora (118 gradable metrics) into one table (battery/fleet_boundness.json).

| task | n_asp | n_gates | med ratio_L | med ratio_Q | %both-swap-bound | E6 within |
|---|---|---|---|---|---|---|
| legal | 20 | 9 | .134 | .336 | .10 | .473 |
| PR    | 20 | 12 | .163 | .026 | .10 | .104 |
| CW    | 37 | 5 | .256 | .232 | .14 | .177 |
| humor | 31 | 4 | .434 | .129 | .07 | .366 |
| math  | 35 | 0 | .567 | .371 | .06 | .233 |
| **pooled E6** | | | | | | **.294 (n=118)** |

Readings:
1. ★ ROBUST FLEET RESULT: E6 Spearman(ratio_llama, ratio_qwen) is POSITIVE in
   every one of the 5 corpora (.10–.47) and pooled **.294 (n=118)** — boundness
   is a CRITERION-level property fleet-wide, now over ALL metrics, not just the
   certified subset (pooled .294 reproduces the earlier certified-panel .295
   exactly). This is the clean "scaled to all metrics" confirmation.
2. ★ RECONCILE two boundness orderings — they answer DIFFERENT questions and are
   NOT in tension:
   - ALL-METRICS median loss (this table): math HIGHEST (.567), legal/PR lowest.
     Matches the §TRANSPORT-3FAM finding "math least transportable" — math
     field-meaning is tied to the extractor's notation handling.
   - CERTIFIED-GATE fragility (§CROSSFAM): taste-pole gates most fragile, law
     least. That is the SELECTED high-signal subset (the aspects that gate).
   The median-ratio-over-all-metrics is additionally noise-inflated at small fm
   (ratio=td/fm), so quote it as a distribution statistic, NOT as a
   certificate-boundness claim; the gate table is the certificate statement.
3. %both-swap-bound (P_degrade≥.95 under BOTH families) = 6–14% per task — a
   minority of metrics are strongly bound to the extractor in both directions;
   the rest transport at least partially.

COVERAGE STATED HONESTLY: transport/boundness + E8-ARTIC now cover ALL gradable
metrics in the 5 hybrid-fleet corpora (PR/CW/math/humor/legal). Survey-only
aspects have NO fields to swap; patents_pa is design-null (verbatim grounding →
fm≈0); code_review/code_competition/news never had hybrid fleets (nothing to
transport). Remaining check-coverage gap = judge-family replication (2nd judge):
present for PR/math/legal, MISSING for CW/humor → GPU-queued.

### §JUDGE-REP-FLEET — second-judge replication scaled to all 5 corpora (R21, 2026-07-07)

CW + humor got their Llama-70B second judge (jobs 600/601, 0.80-util wrapper after a
contention-race fix; gpu_waiter grabbed a GPU at 05:02, done 05:41). With PR/math
(W1.2) + legal (R19), judge-family replication now covers ALL 5 fleet corpora.
judge_rep_task.py (validated: reproduces legal; fresh math raw .546 ≈ W1.2's .55).

| task | raw agree (median) | disatt agree | codability-order ρ | gates re-gate P≥.95 |
|---|---|---|---|---|
| PR (W1.2) | .80 | ~1.0 (focus) | **.93** | a86 yes / a110 NO / a80 dir. |
| legal | .71 | **.789** | .591 | 1/9 |
| CW | .639 | .747 | .573 | **0/5** |
| math | .546 | .742 | .761 | 0/0 (no gates) |
| humor | .542 | .683 | .68 | **0/4** |

Readings (taste-vs-doctrine):
1. COVERAGE COMPLETE: every fleet corpus now has a 2nd-judge replication. The
   instrument-stability claim ("certificates are (criterion, judge-family)
   statements") is now fleet-wide, not PR/math-only.
2. DISATTENUATED cross-judge agreement is the clean read (raw conflates with judge
   reliability — math's raw .546 rises to disatt .742 once its low rel1 is divided
   out). On disatt, the TASTE POLE is lowest (humor .683) and the specified/doctrine
   pole highest (legal .789, PR focus ~1.0): the two judge families agree LEAST on
   the underlying construct exactly where the metric is thickest. A rough thick→thin
   gradient, though CW (.747) sits with math, so it is graded not monotone.
3. CODABILITY-ORDERING replication is POSITIVE in all 5 (ρ .57–.93) — whatever makes
   an aspect codable under Gemma makes it codable under Llama, fleet-wide; highest
   for the most codable tasks (PR .93, math .761), lower at the taste pole.
4. ★ GATE replication across judges MIRRORS extractor-fragility: taste-pole gates
   replicate WORST (CW 0/5, humor 0/4 at the strict P≥.95 bar), legal 1/9, PR partial.
   The taste-pole certificates are DOUBLY bound — fragile on both the extractor axis
   (§CROSSFAM) AND the judge axis. They don't vanish, they slip below the absolute
   bar (a144 P=.887, a351 P=.294) — same "(criterion, judge-family) property" story.
5. RECONCILE with R19: pole-level, taste is doubly-fragile (both axes elevated); but
   criterion-level the two boundness axes are INDEPENDENT (legal a44 = field-robust
   yet judge-fragile). Pole-correlation + criterion-independence both hold: the thick
   end concentrates fragility on average, individual criteria can still dissociate.
   → The 4-tuple certificate stamp (criterion, judge-family, executor, extractor-
   family) is load-bearing at the taste pole on BOTH judge and extractor coordinates.

### §REVIEW-PASS (2026-07-07 pm) — three re-issues from a full-thread audit

Systematic audit of every battery output vs. its harness version. All published
JSONs reproduce their notes tables; three gaps found and closed (all CPU-only,
existing extractions — no new measurement targets):

**1. PR judge-rep row re-issued with the SAME machinery as the other 4 corpora**
(judge_rep_task.py taught the v2 loader; v2/results_llama.jsonl was already on
disk). The W1.2-cited row was focus-aspect/different-leg. Same-machinery PR row:
raw agreement **.799** (validates W1.2's .80), disatt **.958**, codability-order
ρ **.967** (n=19), gates **1/7 gradable at P≥.95** (a76 .995; a87 .894 near;
a115/a104/a112 collapse .004–.045; a119 gradable but its codegen baseline is the
known constant artifact; a86/a110/a105/a128/a67 not in v2 machinery — W1.2 n=500
leg still governs those: a86 replicates, a110 does not). REVISED FLEET TABLE:

| task | raw | disatt | codability ρ | strict gates re-gate |
|---|---|---|---|---|
| PR    | .799 | **.958** | **.967** | 1/7 (+a86 W1.2; a110 fails) |
| legal | .710 | .789 | .591 | 1/9 (a46; a23 .89 near) |
| CW    | .639 | .747 | .573 | 0/5 (a144 .887 near) |
| math  | .546 | .742 | .761 | 0/0 (no gates) |
| humor | .542 | .683 | .680 | 0/4 (best .542) |

READING REVISED vs §JUDGE-REP-FLEET: (i) the agreement gradient SHARPENS — disatt
now runs PR .958 ≫ legal .789 > CW .747 ≈ math .742 > humor .683 on full fleets,
the cleanest thin→thick ordering any battery leg has produced; (ii) the earlier
claim "taste-pole gates replicate worst, PR partial" SOFTENS — strict-bar absolute
gates are judge-family-bound at EVERY pole (PR 1/7 too). What remains graded is
how FAR below the bar gates fall (legal keeps 4 gates ≥.70, PR keeps 2 ≥.89;
CW keeps 1; humor none above .55). Doubly-bound taste-pole claim stands, but
state it as "farther below the bar," not "uniquely fails."

**2. Stale E2 stip evals re-issued** — stip_eval_{CW,math,humor}.json predated
the SAFE_BUILTINS + genexpr-namespace fixes (only PR had been re-run). Fixed-
harness re-run recovers the silently-dropped fields: coverage 9 → 28 fields.
Medians ~unchanged (CW .956/.042, math .693/.087, humor .849/.084 — humor's
published .92 was a 3-field median, now .849 over 10), Gemma rule-follower
conclusion SURVIVES. ★ New texture: 3 recovered cells show snapback > compliance
under GEMMA — CW a234__prose_craft (.29/.57), math a150__engagement (.39/.45),
math a60__lead_pattern (.23/.26) — precisely the thick fields flagged elsewhere
(a234 = the sole sub-1.0 qwen_ton whisper; a60 = the field that snaps Gemma in
the math-ext grid). Even the rule-follower's compliance cracks exactly where the
colliding community concept is thickest; three independent readouts now converge
on the same fields.

**3. E3 legal 70B rung filled** (blocked pre-R19; legal field_results_llama.jsonl
now exists from the transport build). Legal staircase: 3B −.001 → 8B .125 → 70B
**.187** (fm-bearing medians), per-criterion monotone fraction .529 — second-
highest in the fleet. E3 within-family staircase now rises at the median in
**5/5 tasks**; no task-level exception remains.

Files: battery/judge_rep_press_releases.json (new), stip_eval_{creative_writing,
math,humor}.json (re-issued), eval_scale.json (rev: legal llama70 column).

---

## §CODA — a-priori codability probe + explanatory join (2026-07-08)

*Pre-registered 2026-07-05 (priors note §3.1); run 2026-07-08 after AS pushed for an
explanatory analysis of why some units code while others stay prompted. CPU + Sonnet
only. 141 fleet criteria, F1–F8 annotated blind (task identity hidden, ids shuffled,
6 synthetic anchors interleaved per batch): 0/32 hard anchor failures — clean pass.
Outputs: battery/coda_eval.json; scripts build_coda_batches.py / eval_coda.py.*

**Headline: the CODA verdict lands on the pre-registered LOW branch — within a
community, the seam is NOT legible in the criterion's phrasing; you must run the
pipeline.** Pooled across tasks every predicted sign holds (F1 quantifiability +.23,
F2 span-locality +.26, F5 rule-shape +.18 on y_code; F3 norm-deixis −.22, F4
reader-effect −.18, F6 aggregation-breadth −.23) — but that is the domain gradient
(C2) wearing phrasing as a proxy. Within-task the effects collapse (only F1 survives,
+.26), and the honest test is LOTO:

| predictor of y_code (code floor r̃) | LOTO/pooled Spearman | 95% CI |
|---|---|---|
| rank-ridge on F1–F8, leave-one-task-out | **.031** | [−.13, .20] |
| zero-shot "rate codability 0–10" (Sonnet) | .139 | [−.03, .29] |
| rel1 alone | .232 | — |

Per-task LOTO: PR .45, everything else ≤ .01 — only the most institutionally
compiled domain has phrasing-legible seams. Zero-shot matches the feature model, so
the null is about the *description text*, not the feature choice. y_fm is slightly
more phrasing-legible (LOTO .185; within-task F4 .23, F7 .24, F3/F6 .20) — where
enculturation load LANDS is more predictable than how far code GETS.

**But the mediation chain shows exactly where prediction breaks.** Phrasing robustly
predicts the compiler's *explication strategy* (within-task Spearman, CODIF join):
F1 quantifiability → C5 extract-compute **.49** and C3 form-measure .41; F6
aggregation-breadth → C6 exemplar-match **.38**; F8 cross-positional → C3 .36/C5 .34;
F3 norm-deixis → C6 .31. So: phrasing → how the program tries (strong), phrasing →
whether it reaches the judge's signal (null, within domain). The compiler is not
confused about strategy; the failure is in whether the community's norm has
surface-instantiated correlates the ops can reach — a fact about the practice, not
about the wording. (Daston, made measurable: thinning fails where the world wasn't
standardized, not where the rule was badly phrased.)

**Ex-post separators — what DOES distinguish coded from prompted, within task:**

| coordinate | vs y_code | vs certified | n |
|---|---|---|---|
| distill_max (E7: field's surface-distillability) | .20 | **.40** | 72 |
| C6 exemplar-match present | .04 | **.38** | 141 |
| rel1 | .24 | .34 | 141 |
| ratio_mean (extractor-boundness) | .05 | .32 | 117 |
| conv (E8 articulation convergence) | −.01 | .19 | 139 |

The best available answer to "why does THIS criterion code?": (i) its borrowed
judgment is distillable into surface features (the enculturated reading has learnable
textual correlates), and (ii) the community has stable paradigm cases (retrieve_similar
viable). Articulation convergence — how alike families *gloss* the construct — says
nothing (−.01), echoing the E8 dissociation: verbalization ≠ codability. ratio_mean
+.32 on certified replicates the doubly-bound-certs pattern from the transport leg.

Caveats: one Sonnet annotator pass (anchor-validated); n=141 clustered in 5 tasks;
y_code has a floor-clip pileup at 0; distill covers the E7 subset only (n=72). All
descriptive — no gates keyed to CODA.

---

## §EXPANSION — five new arms (2026-07-08): more agentic flow, more metrics, more seams, unit→code

*User directive: run everything in Part-1 (adaptive agentic flow, more metrics, more seams,
direct semantic-unit→code) + the recommended next steps (rewrite intervention, budget ladder),
THEN return to predictability. One consolidated GPU pass (Gemma-4-31B, GPU 7, ~75k prompts:
judging + field extraction). All arms below. Files under battery/ + tasks/{peer_review,
legal_ss_disability,humor_units}/.*

### 1. REWRITE — does checklist re-articulation move the code floor? (battery/rewrite_eval.json)
Causal test of the CODA null. 18 near-miss criteria, three code arms (train-selected flavor,
held-out ρ): **orig** (stored flavors) / **ctl** (fresh Sonnet recompile of the SAME
description) / **rewrite** (fresh compile of a gestalt→checklist rewrite holding the referent).
- **rewrite − ctl: median +0.058, only 3/18 at P≥.95.** Re-wording helps a little, not reliably.
- **ctl − orig: ~12/18 at P≥.95** — the *compiler-version* upgrade (same description, fresher
  codegen) moves the floor MORE than the rewording does.
Reading: consistent with CODA — phrasing underdetermines the seam; you gain more from a better
compiler than from a better-worded criterion. A few genuine rewrite wins (PR a119 .30→.60 P.998;
humor a135 .27→.37 P.998; legal a39/a15 +.25) show checklist form helps where the construct had
locatable checkable sub-parts, i.e. exactly the F1/F5-high criteria.

### 2. AGENTIC-R2 — flexible compile with 6 rounds (battery/agentic_cert_r2.json)
Reverses the round-1 "boundary is a domain fact" read. 12 tail criteria (rescue gate-FAILs +
recode field-dominated), Sonnet agent ≤6 reflective rounds, free to invent code ops; certified
held-out with the SAME gate machinery.
- **median Δρ_test = +0.05; 2 NET-NEW gate certs (CW a225, humor a333); gate-cert cand/h0 = 3/1.**
- Real held-out recodes of field-dominated criteria: legal a0 pure-CODE ρ .69→.75 (signal
  moved INTO code), a15 .745→.831, humor a315 .364→.645 (+.15 test), a333 .49→.75.
- Winner's curse visible: math a144 train +.64 but test −.09 vs h0 (gap +.25); a72 train +.22
  test flat. The single-round static compiler missed these; 6 rounds + op-invention finds them —
  but overfits on a minority. **Refined verdict: the code-sufficiency boundary is partly a
  compiler-effort artifact, not purely domain-fixed — flexibility recovers real mass on
  recodable/field-dominated criteria, with honest overfit on some.**
- ★ a108 honesty check: its 2nd field `technique_novelty` was declared but the agent kept it
  "neutral". EXTRACTED for real, it HURTS (.609→.435). Certified a108 = code-recode + the one
  real field `creative_trick` (.537→.609). Reconfirms the fleet rule: winners ADD code signal;
  borrowed pointers that replace the construct lose.

### 3. BUDGET LADDER — seam depth 0/1/2/4 fields (battery/budget_ladder.json)
15→12 criteria stratified by c8_share, fm>0. Budget-0 = blanked; b1 = best single field
(train-selected); b2 = full 2-field h0; b4 = h0 + 2 NEW construct-grounded fields (GPU-extracted).
- **b0→b1 is the dominant jump everywhere** (one borrowed field does most of the work).
- **b2→b4 lifts the FIELD-DOMINATED criteria specifically**: CW a117 .573→.722 (+.15),
  math a180 .23→.43, humor a0 .43→.54, a333 .49→.58, a153 .61→.66; taste/craft-shell criteria
  plateau or regress (a144, a18, a234, a36).
- **Seam depth (min budget for 95% of max ρ) by c8_share: HIGH=4, MED=2, LOW=4 (median).**
  The ≤2-field contract was TRUNCATING the most enculturated (c8=HIGH) criteria — they keep
  absorbing borrowed judgment out to 4 fields. Seam depth is a graded outcome, not binary.

### 4. NEW METRICS — peer_review + legal_ss_disability (cam_profile.json, 8 tasks now)
Full 20-aspect surveys, same pipeline (2-form Gemma judge, code flavors, h0 hybrids, gates).
- **peer_review (ML paper submissions): CAM .296→.359, frac≥.5 .31.** Lands at the TASTE pole.
  Approved 5-cluster splits cleanly: a25 impact/dissemination (hyb .644, surface DOI/github
  markers) + a45 scalability (.547, scale claims) code/certify; a163 reasoning-quality,
  a130 abstract, a214 quotation-usage all FAIL (thick judgment on paper reasoning; a214 rel1
  .55 also judge-noisy).
- **legal_ss_disability (SSA disability appeals, drop-in): CAM .362→.524, frac≥.5 .61** —
  REPLICATES the title_vii "law = institutionally pre-compiled" shape (.372→.621) on a second,
  independent legal domain. Doctrinal elements (step-5 burden, duty-to-develop, harmless-error)
  code moderately; the certified layer is again the elements layer.
- **Updated cross-domain spectrum (CAM_certified):** PR .697 ≈ law_title_vii .621 >
  law_ss_disability .524 > CW .466 > math .377 ≈ peer_review .359 ≈ humor_units .352 ≈
  humor .351. Two legal domains now bracket the institutional pole; peer_review joins the
  taste cluster despite being technical — its criteria are reviewer-tacit, not doctrine.

### 5. UNIT→CODE — census lexicon units through the seam (battery/units_eval.json)
The direct semantic-unit→code test. 30 humor census R1 constructs (author-verbatim units:
4 MECHANICAL + 13 CRAFT + 13 TASTE, n_sources≥3) formatted as criteria, judged + hybridized
on the SAME humor items.
- **humor_units CAM .122→.352 ≈ curated humor .120→.351.** ★ Bottom-up author-extracted units
  land at the SAME seam as top-down curated aspects — a strong construct-validity check on the
  whole census program: the seam is a property of the domain's norms, not of who wrote the rubric.
- Code floor by census type (median r̃): **MECHANICAL .259 > TASTE .132 > CRAFT .038.**
  The naive two-faces prediction (TASTE lowest) INVERTS between CRAFT and TASTE — because the
  codable TASTE units are content-RATING (u22 Purposeful-Taboo .308, u2 Audience-Rating .259:
  surface lexical markers — slurs, profanity), while the uncodable ones are craft-TECHNIQUE
  (u10 tone-consistency, u12 tension-build, u25 self-deprecation: all code_rt 0.0 — need to
  read the joke). So what codes among real author units is surface content-marking, not
  technique — the same split the fleet shows, now at the author-lexicon grain.
- fm substantial across all types (.18–.27): the borrowed field carries most of what code can't,
  MECHANICAL included.

Files: battery/{rewrite_eval,agentic_cert_r2,budget_ladder,units_eval,coda_eval}.json;
tasks/{peer_review,legal_ss_disability,humor_units}/{results,field_results,hybrid_gate_report}.json;
programs_{peer,ssdis,units,b4}/, programs_agentic/*_agentic_r2.py; scripts eval_{rewrite,units,
budget_ladder,budget_b4}.py, cert_agentic_r2.py, build_{coda_batches,unit_task,unit_field_prompts,
b4_field_prompts}.py. GPU: one Gemma-4-31B pass, GPU 7, 75,750 prompts.

**NEXT (user-gated): return to predictability** — re-run CODA/LOTO with the expanded panel
(peer_review + ss_disability + 30 units = +66 criteria) and the new outcomes (seam depth,
agentic-r2 recodability, b4 marginal). Question: does phrasing predict seam DEPTH better than
it predicted the binary floor?

---

## §PREDICTABILITY-2 — does phrasing predict seam DEPTH better than the binary floor? (2026-07-08)

*Return-to-predictability on the EXPANDED panel: CODA-1 fleet (141) + 55 new criteria
(peer_review 16, legal_ss_disability 13, humor_units 26), F1-F8 blind-annotated + anchored
(0 hard-anchor fails). 196 criteria across 8 tasks -> more LOTO folds + a continuous
seam-depth target. battery/coda2_eval.json; eval_coda2.py.*

**Answer: YES. Phrasing predicts the continuous seam depth ~1.7× better than the binary
code floor, and via DIFFERENT features.**

LOTO rank-ridge (F1-F8, 8-task leave-one-task-out):

| target | LOTO pooled | 95% CI | zero-shot |
|---|---|---|---|
| **y_code** (binary code floor r̃) | 0.178 | [.03, .31] | **0.249** |
| **y_fm** (seam depth = borrowed-judgment marginal) | **0.301** | [.15, .42] | 0.168 |
| **y_seampos** (frac of hybrid signal that is borrowed) | 0.233 | [.08, .38] | 0.155 |

Three findings:
1. **Depth is more legible than the floor.** y_fm LOTO .30 ≫ y_code .18 — how much borrowed
   judgment a criterion will need is more predictable from its wording than whether pure code
   clears the bar. (The binary floor rose from CODA-1's .03 to .18 mostly from POWER — 8 tasks
   = 7 folds vs 5 — so don't over-read the floor jump; the DEPTH>floor gap is the result.)
2. **Different features predict the two.** Within-task, the FLOOR is driven by the
   "mechanically checkable" features: F1 quantifiability .36, F2 span-locality .26. DEPTH is
   driven by the THICKNESS features: F7 specialized-world-knowledge .34, F4 reader-effect .31,
   F3 norm-deixis .27, F8 cross-positional .27, F6 aggregation .25. The phrasing that tells you
   "code can reach it" is not the phrasing that tells you "how deep the borrowed seam sits."
3. **Structure beats holistic for depth; holistic beats structure for the floor.** Zero-shot
   "rate codability 0-10" beats the F1-F8 model on y_code (.25 vs .18) but LOSES on y_fm
   (.17 vs .30). Predicting whether it codes at all is a gestalt call; predicting depth needs
   the decomposed thickness features.

Budget-depth validation (12 ladder criteria, min-fields depth): F5 rule-shape .46, F2
span-locality .34, F8 .24 track actual seam depth directionally (small n; y_fm itself only .08
vs budget-depth — total marginal ≠ diminishing-returns depth, different operationalizations).

**Throughline update:** CODA-1 said the seam's BINARY location is barely legible within a
community. CODA-2 sharpens it: the binary floor stays hard to call, but the GRADED depth of
borrowed judgment IS legible from the criterion's thickness vocabulary — Daston's "how far did
thinning get" is written in the words, even when "can you thin it at all" is not. The
enculturation-load features (F3/F4/F7) that CODA-1 found predict y_fm within-task now survive
leave-one-task-out on 8 tasks.

---

## §GLM-TACIT — semantic gravity across the GLM family (2026-07-08)

*User: expand LLMs to the GLM family (z.ai subscription, two sk3 keys, free Anthropic
endpoint), test tacitity across all GLM endpoints. Infra: api_field_runner.py now sees both
sk3 keys (.z-ai-api-key.txt + .z-ai-api-key-alexander-spangher.txt) with --key-file toggle;
GLM is a first-class extractor family, 0-GPU pure HTTP. Probe: build_glm_tacit.py +
eval_glm_tacit.py; battery/glm_tacit/. 4 endpoints × 2000 prompts, split across the two
quotas.*

**Design (self-contained, quota-sparing).** E2 semantic-gravity: for each e2-bearing field,
each GLM version answers BOTH its OWN community-condition (full field instruction) and the
DEVIANT stipulation, on 50 stable-hashed items. Snap-back measured against THAT version's own
community answer (cleaner than the Gemma-anchored fleet stip eval). Thick task (humor, 10
fields) vs thin task (math, 10). Answer distributions healthy (37–58 distinct comm answers/
version, 0 empty).

| endpoint | humor comply / snap | math comply / snap | thick>thin snap? |
|---|---|---|---|
| glm-4.5 | .462 / **.538** | .793 / .207 | yes (2.6×) |
| glm-4.6 | .500 / **.552** | .793 / .098 | yes (5.6×) |
| glm-4.7 | .452 / **.407** | .571 / .209 | yes (1.9×) |
| glm-5.2 | .735 / .188 | .741 / .259 | no (flat/inverted) |

**Findings:**
1. **The tacitness signature replicates in a THIRD model family.** On thick humor constructs,
   GLM snaps back to community meaning against the deviant rule at substantial rates
   (.41–.55 for 4.5/4.6/4.7), often exceeding compliance; on thin math constructs it complies
   with the rule (.57–.79) and barely snaps back (.10–.26). Semantic gravity is thick-selective
   in GLM just as in Gemma / the Llama ladder / the Qwen toggle. Cross-family generality of the
   phenomenon is now 4 families.
2. **★ Within GLM, snap-back declines with capability.** Median humor snap-back falls 4.5 .54
   → 4.6 .55 → 4.7 .41 → 5.2 .19, while humor compliance rises to .74 at 5.2; the thick/thin
   gap collapses at the top (5.2 snaps back slightly MORE on math than humor). The strongest
   endpoint follows the deviant rule most and resists least.
3. **CAVEAT (confound, stated up front).** This compact probe has NO neutral-rule execution
   control (E2-KIND cell 6), so it CANNOT subtract the instruction-following-deficit: 5.2's
   higher compliance conflates "less semantic gravity" with "better at executing the mechanical
   rule" (cf. Wei et al. 2023, prior-override is scale-dependent). The clean, capability-held-
   fixed result is the WITHIN-version thick>thin contrast (finding 1), which holds for 3/4.
   The scale trend (finding 2) is suggestive; a clean "gravity vs GLM scale" claim needs the
   full E2-KIND grid (cells 5/6) run on GLM — a natural, still-sparing follow-up.
4. Per-field heterogeneity: most humor fields lose snap-back 4.5→5.2 (phrasing_sharpness
   .60→.03, twist_quality .50→.19, harm_mitigated .54→.27), but a few GAIN (resolution_mode
   .63→.80, padding_verdict .60→.81) — 5.2 resists the deviant rule MORE on those. Not a
   uniform capability effect; construct-specific.

Files: battery/glm_tacit_eval.json, glm_tacit/{prompts,checkers,results_glm-*}.jsonl;
build_glm_tacit.py, eval_glm_tacit.py; api_field_runner.py (two-key support).

### §GLM-TACIT.2 — E2-KIND deficit control resolves the scale confound (2026-07-08)

*Full E2-KIND grid on all 4 GLM endpoints (cell4 nonce+deviant, cell5 name+neutral,
cell6 nonce+neutral), 50-item subsample, own comm baseline. Answers: is glm-5.2's low
raw snap-back genuine gravity loss or just better rule-following? battery/glm_e2kind_eval.json;
build_glm_e2kind.py, eval_glm_e2kind.py.*

Deficit-corrected gravity subtracts rule-following capacity. Two rows:
NAME-row gravity = cap5(name+neutral exec) − comply2(name+deviant, from §GLM-TACIT);
NONCE-row gravity = cap6(nonce+neutral exec) − comply4(nonce+deviant). HUMOR (thick):

| endpoint | cap5 name+neutral | comply2 name+deviant | **NAME-gravity** | nonce-gravity |
|---|---|---|---|---|
| glm-4.5 | .68 | .462 | **0.218** | −0.01 |
| glm-4.6 | .68 | .500 | **0.180** | −0.06 |
| glm-4.7 | .72 | .452 | **0.268** | +0.05 |
| glm-5.2 | .80 | .735 | **0.065** | +0.03 |

**Resolution: glm-5.2 genuinely has less semantic gravity — the scale decline survives the
deficit correction.** Rule-following capacity DOES rise with scale (cap5 .68→.68→.72→.80,
5.2 IS the best rule-follower), but subtracting it does not explain away the effect: NAME-row
gravity still collapses from .22 (4.5) / .27 (4.7) to .065 (5.2). 5.2 follows the deviant
redefinition (.735) almost as readily as a neutral rule (.80) — barely any construct-specific
resistance — whereas 4.7 follows the deviant rule at .45 but a neutral one at .72, a .27
resistance gap. So the drop is real gravity loss, not a capacity artifact. (Both effects
coexist: 5.2 is more capable AND less gravity-bound.)

**★ Bonus: gravity is NAME-locked (lexical), not concept-locked.** NONCE-row gravity is ≈0
for EVERY GLM version, small and large alike (−.06 to +.05). Replace the community WORD with
a nonce and the gravity well vanishes entirely — all versions just comply with the deviant
rule. The tacit pull lives in the community's lexical key, not in a concept the model infers
from the deviant definition's vocabulary — a clean within-GLM confirmation of the
name-gravity hypothesis (E1/E2 name-vs-concept dissociation), the phantom-snap-back this grid
was built to detect does not fire in GLM.

MATH (thin): nonce-gravity ≤ 0 for all versions (no gravity, as expected; the negatives are
checker/rule-complexity asymmetry — deviant math rules are more mechanically checkable than
the neutral controls — not anti-gravity).

**Combined GLM-TACIT verdict:** (i) the thick-selective semantic-gravity signature replicates
in a 4th model family; (ii) it is lexically keyed (nonce kills it); (iii) it genuinely weakens
as GLM capability rises, above and beyond the concurrent rise in rule-following capacity — the
strongest endpoint is both the best rule-follower and the least community-anchored. This is the
Wei-2023 prior-override-scales story, but with the deficit control it is a statement about
reduced ANCHORING, not merely improved compliance.

---

### §PREDICTABILITY-2 CORRECTION — pooled-LOTO fold-size artifact; headline retracted (2026-07-08 pm)

*Full-audit pass over the battery ("check everything") caught a construction bug in the
pooled LOTO statistic of eval_coda2.py (and eval_coda.py, shared code): within-fold ranks
(raw 0..n−1) were concatenated across folds of unequal size (13–36 criteria). Mean rank
scales with fold size in BOTH vectors, so pooling injects a shared fold-size signal
regardless of prediction quality. Within-fold permutation of the predictions (zero true
signal) yields pooled ρ ≈ +.16 to +.19 on this panel — i.e., most of the reported pooled
numbers was mechanical. Fixed by normalizing within-fold ranks to [0,1] before pooling +
within-fold permutation p-values (eval_coda2.py v2; coda2_eval.json regenerated; audit
script $CLAUDE_JOB_DIR/tmp/coda2_pooling_audit.py reproduces old numbers, null, and fix;
size-weighted per-fold means agree with the fixed pooled to ±.004).*

**RETRACTED: "depth is ~1.7× more legible than the floor" (y_fm .30 vs y_code .18).**
Corrected LOTO (196 criteria, 8 tasks):

| target | v1 (artifact) | v2 corrected | perm p | zero-shot v2 | zs perm p |
|---|---|---|---|---|---|
| y_code (binary floor) | .178 | **.091** | .11 | **.136** | **.029** |
| y_fm (seam depth) | .301 | **.121** | **.047** | −.021 | .60 |
| y_seampos | .233 | **.041** | .31 | −.076 | .85 |

- The y_fm-vs-y_code separation is NOT interpretable (.12 vs .09); phrasing-legibility of
  the seam is weak everywhere across tasks. The CODA-1 → CODA-2 floor "rise" (.03→.18) was
  the artifact growing with more/less-equal folds, not power. CODA-1's null verdict was
  computed under the same construction, so it was optimistic — its conclusion (seam not
  phrasing-legible) only strengthens.
- **What SURVIVES (the honest headline): the double dissociation.** Zero-shot holistic
  "rate codability 0-10" predicts the FLOOR (.136, p=.029 — the only significant floor
  predictor) and has literally nothing on depth (−.02). The decomposed F1-F8 model's only
  signal is on DEPTH (.121, p=.047). Per-feature (corrected within-task): checkability
  F1 .19 / F2 .13 correlate with the floor and F1 anti-correlates with depth (−.09);
  thickness F7 .18 / F4 .14 correlate with depth and F7 is .00 on the floor. Same pattern
  as v1 at roughly half amplitude; F3/F6/F8 evaporate.
- Budget-depth validation (n=12) unchanged (raw values, no rank pooling): F5 .46, F2 .34.

**Other audit flags from the same pass (no re-issues needed):**
- UNIT→CODE: 26–27 of the 30 built units survive scoring thresholds; the per-type medians
  rest on n=2 MECHANICAL / 11 CRAFT / 13 TASTE. The "MECHANICAL > TASTE > CRAFT" ordering
  should not be leaned on (MECH n=2); the robust reading is the within-TASTE split —
  codable units are content-RATING (surface lexical markers), uncodable are technique.
  The headline (humor_units CAM .122→.352 ≈ curated .120→.351) is unaffected (26 units).
- GLM E2-KIND math: cell6 "capacity" < cell4 compliance for 4.5/4.6/4.7 (gravity −.17 to
  −.33) = the neutral control rules are mechanically HARDER than the deviant ones in math;
  cell6 is not a clean capacity control there. Humor cells are well-behaved (cap6 ≥
  comply4 where gravity > 0), so the thick-task conclusions stand; math rows are
  interpretable only as "no positive gravity," as already stated.
