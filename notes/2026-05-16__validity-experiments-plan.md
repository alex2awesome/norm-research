# Validity experiments — "metrics for metrics"

Date: 2026-05-16

## Purpose

So far the articulability scale (L1 code / L2 LLM-judge / L3 expert / L4 tacit) is
**introspected** — gpt-5-mini reads a rubric and self-reports a level. Four prompt
versions (v6→v9) could not pin down the L2/L3 boundary; the score swung 11→6→17→17
on the same 33 clusters. The conclusion from that saga: stop perfecting the
self-report instrument and **measure rubric properties empirically on data**.

These six experiments give every rubric an empirical profile. The profile, not a
prompt, decides where it lands on the scale. This is task #44 made concrete and is
the experimental grounding for `Outcome = f(Verifiable) + g(Articulable) + h(Taste)`.

## Unit of analysis

A **rubric cluster** (R1-refined dedup unit) — same unit as the 2-axis
classification, so results join directly to the existing articulability labels.
Sample stratified: per task, ~40–60 clusters spanning the four self-reported
articulability levels (so we can check empirical vs. introspected agreement).
Total first pass ≈ 400–600 clusters.

## Shared infrastructure

**Datapoint corpora** (real submissions to score against a rubric):
- code-review — `code_review_modeling_dataset.csv.gz` (141K PRs, accept/reject)
- creative-writing — LitBench (preference pairs + rationales)
- humor — newyorker caption ratings; reddit jokes
- peer-review — review corpus (accept/reject labels)
- news-homepages — homepage dataset + `storysniffer_labeled.csv`
- patents — first-draft approval set
- math-stackexchange — binary-judged QA set
- notice-and-comment / grant-funding / legal — corpora exist; label quality TBC

**Models.** OpenAI out of credits on the main account — confirm whether the SALT
lab key (sk3) still has budget for *small* samples. Open judges on sk3 (1 GPU,
stack processes): Llama-70B-FP8, Qwen3.5-122B-FP8, Llama-8B, plus one more family
(Gemma/Mistral) for judge diversity in E1.

**Ground-truth labels** (E5 needs these): code-review, peer-review, news-homepages,
patents, creative-writing (preference). Tasks without clean outcome labels skip E5.

---

## E1 — Consistency

**Q.** Across several LMs and across repeated trials, which rubrics yield the most
stable scores?

**Method.** For each sampled rubric: score N≈100 task datapoints with K judges
(≥4 models) × T trials (≥3, temperature>0). Decompose score variance:
within-judge-across-trial, across-judge, residual. Report per rubric ICC and
Krippendorff's α.

**Output.** `rubric_consistency.parquet` — per rubric: trial-stability,
judge-agreement, total reliability.

**Reads as.** High consistency → the construct is stable and shared (articulable).
Low consistency even within one model across trials → ambiguous or tacit (L4).

**Cost.** ~600 rubrics × 100 pts × 4 judges × 3 trials open-model inference. Heavy
but all on sk3. Start with 1 task end-to-end.

## E2 — Recoverability (from labeled data)

**Q.** Given datapoints labeled with a rubric's scores, can a model re-derive the
rule?

**Method.** Score M datapoints with the rubric → labels. Hide the rubric. Give a
model the labeled datapoints, ask it to state the latent rule. Apply the recovered
rule to held-out datapoints; correlate recovered-scores with original-scores.

**Output.** `rubric_recoverability.parquet` — recovery fidelity (held-out corr).

**Reads as.** High fidelity → learnable from examples (L1/L2). Low → tacit (L3/L4).

## E3 — Rearticulation stability (code + paraphrase)

**Q.** How many ways can the metric be said? Do diverse restatements (a) get
judged "the same rubric" and (b) assign the same values to datapoints?

**Method.** Per rubric generate K NL paraphrases + K code implementations (flag
rubrics where code generation fails — itself a signal). (a) Pairwise LLM
"same rubric?" → equivalence-agreement rate. (b) Score datapoints with each
restatement → cross-restatement score agreement.

**Output.** `rubric_rearticulation.parquet` — code-implementable (bool),
paraphrase-verdict-agreement, code-vs-NL agreement.

**Reads as.** Restatements converge → stable well-defined construct. Restatements
drift → slippery construct (higher on the scale).

## E4 — Describability via synthesized contrastive pairs

**Q.** Can we synthesize minimal contrasting example pairs (high vs low on the
rubric dimension) from which a fresh LLM infers the rubric?

**Method.** Synthesize C contrastive pairs per rubric. Give only the pairs to a
fresh model; have it (a) state the rubric, (b) score held-out datapoints. Compare
to original. This is E2 by ostension rather than by labeled real data.

**Output.** `rubric_describability.parquet` — example-recoverability fidelity.

**Reads as.** Demonstrable by examples → articulable. Cannot be shown → tacit.

## E5 — Rubric vs. code: which predicts ground truth better?

**Q.** For rubrics with both an NL form and a code form, which version's scores
correlate higher with the **task outcome label**?

**Method.** On label-bearing tasks, score datapoints three ways — NL rubric (LLM
judge), code implementation, restatement-consensus — and correlate each with the
ground-truth outcome (AUC / Spearman).

**Output.** `rubric_vs_code_predictive.parquet` — per rubric: AUC_code, AUC_rubric,
AUC_consensus, delta.

**Reads as.** code ≥ rubric → genuinely codifiable (L1). rubric > code → the LLM
judge captures nuance code cannot (L2/L3). This is the direct test of the
verifiable/articulable boundary.

## E6 — Corpus analysis: inarticulability acknowledgments

**Q.** What fraction of the human-written guidebooks explicitly invoke
inarticulability ("you know it when you see it", "hard to define", "taste",
"intuition", "ineffable")?

**Method.** Lexical search over source pages for an inarticulability lexicon +
an LLM pass classifying whether a page concedes some quality is inarticulable.
Aggregate per task.

**Output.** `inarticulability_by_task.parquet` — % of source docs per task.

**Reads as.** External validation of the L4 finding — creative-writing/humor
should acknowledge inarticulability far more than code-review/patents. Cheap, no
labels, no datapoint scoring; can run first.

---

## Synthesis — empirical placement on the scale

Each rubric ends with a vector: (E1 consistency, E2/E4 recoverability, E3
rearticulation stability, E5 code-vs-rubric delta). The L1–L4 levels become
*empirically defined regions* of that space, replacing the introspected label:

- **L1** code ≥ rubric (E5), high recoverability, high consistency, code-implementable.
- **L2** rubric > code, LLM-recoverable, high consistency.
- **L3** recoverable only with expert framing, moderate consistency, restatements drift.
- **L4** low consistency across trials, low recoverability, restatements diverge.

Then compare this empirical placement to the introspected v6–v9 labels — that
agreement (or disagreement) is itself a headline result.

## Open questions for Alex

1. SALT-lab OpenAI key — any budget for small samples, or fully open-model?
2. Unit = rubric cluster (R1-refined), or go finer to leaf rubrics?
3. Start with which task end-to-end? (code-review has the cleanest labels.)
4. E1 needs a judge-model roster — confirm which open models to stand up on sk3.

## Suggested sequencing

E6 first (cheap, no infra). Then E1+E3 on one task end-to-end (code-review) to
validate the harness. Then E2/E4. E5 last (needs the code implementations from E3
plus ground-truth joins).
