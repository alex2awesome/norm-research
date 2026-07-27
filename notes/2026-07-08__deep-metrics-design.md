# Deep metrics: multi-step / seam-oriented metric discovery (design, 2026-07-08)

**Problem.** Every metric the infilling arms can currently propose is a *one-shot rubric*: a
frozen sentence applied in a single judge pass. The validated discoveries so far (clarity
family, Q&A practice norms) are exactly the properties visible in one pass. Deeper preferences
— "the algebra in the key step actually checks out", "the counterexample really satisfies the
claimed conditions", "this answer dominates its sibling answers" — are PROCEDURES, not
descriptions. The metric-seam thread predicts these live in the seam between the language
channel (one-shot rubric) and the code channel (deterministic computation): expressible only
as typed multi-step programs.

## Method ideas (ideation)

1. **Metric programs (typed-step DSL)** — a metric = a short program of typed steps:
   `judge_score` (batched LLM sub-judgment), `judge_extract` (batched extraction of a short
   span/claim), `code` (whitelisted pure-Python on prior step outputs), + `aggregate`.
   Step-synchronous execution keeps offline-batch vLLM discipline (each step = ONE batch over
   all items). This instantiates the seam proposal's typed channels (L-steps, C-steps) inside
   one metric.
2. **Verification-chain metrics** (the fact-check family; tests the verifiability conjecture
   from the other side): extract the key computation/claim → independently redo/check it →
   score agreement. Prediction: in conventionalized fields the residual articulable value
   hides in CHECK-work, not in new descriptions.
3. **Comparative/contextual metrics**: score the item RELATIVE to sibling items (answers to
   the same question; we hold question grouping). Pairwise-A already beats absolute rubrics
   (.723 vs .661 on MathlibPR) — proposals of the form "criterion X, judged relative to the
   other answers".
4. **Perturbation/ablation metrics**: would the judgment survive deleting step k / paraphrase?
   Fragility of the answer's value under targeted ablation as a metric (counterfactual-edit
   pool machinery exists).
5. **Conjecture-then-operationalize**: strong proposer states an abstract community norm
   ("this community rewards answers that meet the asker where they are"), then a second pass
   compiles it into a program; the compile step is itself a seam probe (what fails to compile
   into ≤4 steps is a tacitness certificate candidate).
6. **Depth premium (the seam measurement)**: for every KEPT program, auto-distill a one-shot
   rubric version (same semantics, single prompt) and score it. `depth_premium =
   bits(program) − bits(flattened)`. Positive premium = the metric's value is genuinely
   procedural — a THIRD articulability rung between "rubric-articulable" and "tacit":
   *procedure-articulable*. This is the new quantity the paper gets from this thread.

## Pilot implementation (v0, this session)

- `methods/metrics_tree_infilling/deep_metrics.py`: `DeepMetricProgram` schema (JSON,
  proposer-emittable), AST-whitelisted safe executor for code steps, step-synchronous batched
  executor with judge callback + JSONL cache + temp>0 reliability rerun.
- `scripts/tools/deep_metric_pilot.py`: standalone pilot through the SAME gate arithmetic
  (paired-CV bits gain over the 48-metric bank, NB-corrected confirm, Bonferroni over planned
  programs, redundancy R², content guard on final semantics) + the flattening comparator.
  NOT yet threaded through global_infill (surgery deferred until the pilot pays).
- Pilot legs: math-general-topology + math-probability (hottest tails), GLM-5.2 proposer of
  programs, 70B executor, k=8 programs/leg. Cost: ~3-5 judge batches per program per split.
- Gate additions for programs: report n_steps + judge-calls/item (depth cost); reliability =
  rerun of judge steps with salted temp>0 draws; flattening loss.

## Pilot results (2026-07-08 18:36, topology + probability)

Machinery works end-to-end: 16/16 GLM-proposed programs parsed, executed step-synchronously,
gated. **Topology: `explicit_question_resolution` = the first procedure-articulable candidate**
— guard +0.0190 bits, confirm +0.0166 @ p=0.00334 (missed the pilot Bonferroni-16 bar 0.0031
by 8%!), retest 0.96, **depth_premium +0.0096 bits** (the program beats its own flattened
one-shot rubric by ~0.01 bits — procedural value a single reading cannot capture). Runner-up:
logical_completeness_vs_brevity (confirm +0.0053, premium +0.0018). Probability: GLM proposed
a homogeneous verify-* family — all null (confirm −0.002..+0.003), retests 0.66-0.86 (programs
execute LESS reliably than one-shot rubrics there), premiums negative. Reading: depth pays
where the community's residual is procedural (topology's resolve-the-question-completely norm),
not as a universal recipe; program reliability is the executor bottleneck to watch (#65 tie-in).
NEXT: fresh-sample stage-2 for explicit_question_resolution; diversity constraint on program
proposals (probability's 8 near-identical verify-* programs wasted the leg).

## Risks / guards
- Proposer-generated code: AST whitelist (no imports, no attributes, no dunder, call-list
  {math.*, re.search/findall, len, min, max, abs, sum, float, int, round}); failures → step
  output NaN → program viability gate.
- Leakage: programs see only the item text (+ siblings for comparative kind, never labels).
- Multiple comparisons: programs are PLANNED proposals — same Bonferroni-m discipline.
