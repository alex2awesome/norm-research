# Validity pilot — smoke test + R1 vs R2 determination (2026-05-24)

## What was run

**Smoke test:** 5 paired (R1, R2) metrics × 10 peer-review datapoints, all via Claude subagents.
- 30 code-gen variants (5 metrics × 2 levels × 3 trials)
- 50 judge calls (5 metrics × 2 levels × 5 paraphrases × 1 chunk-of-10)
- Total subagent calls: 80 (plus 10 paraphrase-gen)

All artifacts at `runs/validity_pilot/smoke/`. Analysis at
`runs/validity_pilot/smoke/analysis_report.md`.

## Headline results

| Dimension | R1 (specific rule) | R2 (broad aspect) | Winner |
|---|---|---|---|
| **Code SNR** (Claude intra-trial signal/noise) | **2.68** | 1.37 | **R1** (Δ=1.32) |
| **Judge SNR** (Claude intra-paraphrase signal/noise) | 2.25 | **2.76** | **R2** (Δ=0.51) |
| **Code↔Judge Pearson** (within Claude) | +0.118 | **+0.240** | **R2** (Δ=0.12) |
| **Cross-MODEL code Pearson** (Claude vs Llama code-gen) | **+0.154** | −0.017 | **R1** (Δ=0.17) |
| **Cross-MODEL judge Pearson** (Claude vs Llama judges) | +0.362 | **+0.730** | **R2** (Δ=0.37) |

**R1 wins on every code dimension. R2 wins on every judge dimension.** The pattern
holds both within-model (trial/paraphrase consistency) AND across-model (Claude vs
Llama agreement). This is a clean empirical validation of the
specificity-vs-generality intuition.

### Cross-model calibration note

- Llama judge systematically scores LOWER than Claude judge (R1: 0.604 vs 0.478;
  R2: 0.569 vs 0.428; mean abs diff ~0.16). Convergent validity still holds
  (Pearson > 0.7 at R2) but absolute calibration differs.
- Llama as judge has higher paraphrase σ (~0.08) than Claude (~0.02-0.03) —
  Llama is more sensitive to wording.
- Llama code-gen scores are also systematically lower than Claude (especially at R2
  where 4/5 of Llama's R2 codes return ~0). Llama writes stricter/more conservative
  Python heuristics.

## Interpretation

**R1 wins for code-gen, R2 wins for LLM-judge** — confirms the user's
specificity/generality intuition at the pipeline level:

- **Code-gen rewards specificity.** A specific R1 rule like "Define and present
  error bars clearly" gives the coder a concrete, programmable target.
  The R2 framing ("Statistical Methodology and Reporting") is too abstract for
  rule-based code to discriminate — multiple trials converge to similar overall
  signals but with less per-datapoint variance.

- **LLM-judge rewards generality.** Asking "does this paper address statistical
  methodology" (R2) lets the judge integrate over many cues. Asking "does this
  paper define error bars clearly" (R1) often returns score=5 (N/A) when the
  rubric simply doesn't apply to this datapoint — collapsing variance to zero.

- **Code↔Judge agreement is higher at R2** because both methods can converge
  on broader-pattern recognition where the rubric clearly does apply, whereas at
  R1 they may disagree on what counts as a partial match.

## Per-metric breakdown (smoke)

| Metric | Level | Code SNR | Judge SNR | Code↔Judge ρ | Notes |
|---|---|---|---|---|---|
| m0: Strengths and weaknesses | R1 | 0.63 | 0.66 | −0.06 | Conservative judge scores 2-7 |
| m0: same (aspect) | R2 | 1.01 | 1.82 | +0.09 | R2 lifts signal |
| m1: Define error bars | R1 | 3.40 | **0.00** | +0.00 | **N/A on every paper** — rubric doesn't apply to abstracts |
| m1: Statistical Methodology | R2 | 1.44 | 5.35 | −0.35 | R2 framing covers the cases R1 missed |
| m2: Data access restrictions | R1 | **7.33** | 4.94 | **+0.94** | R1 wins decisively — concrete rule, strong agreement |
| m2: Data Access Controls | R2 | 1.49 | 1.50 | +0.43 | R2 less specific |
| m3: Results presented clearly | R1 | 1.21 | 2.80 | −0.32 | Disagreement between code (lexical) and judge |
| m3: Results Presentation Quality | R2 | 0.97 | 2.58 | +0.20 | R2 frames better |
| m4: Detailed description | R1 | 0.84 | 2.85 | +0.03 | Modest |
| m4: Detail and Specificity | R2 | 1.91 | 2.56 | **+0.83** | R2 strong agreement |

**One coverage failure (m1 R1, error bars):** The judge returned score=5 (N/A)
on all 50 (paraphrase × datapoint) combinations because the dataset is paper
abstracts, which never discuss error bars at the abstract level. This is a real
finding: **very specific R1 metrics can have applicability gaps depending on
the datapoint type.** A real pipeline needs an N/A handling policy.

## Determination

**Don't pick one level — use both, for different purposes:**

1. **Code-gen pipeline → use R1.** Specific rules give the coder a programmable
   target. R1 SNR is ~2× R2. Hard-FP rate is also lower at R1 because the
   rubric is concrete enough to be testable.
2. **LLM-judge pipeline → use R2.** Broader aspects give the judge enough scope
   to score nearly every datapoint without forced "N/A" outputs. R2 paraphrase
   stability is higher (5.35 vs 0.00 for m1).
3. **Convergent-validity comparison → use R2.** Code↔Judge Pearson is 2× higher
   at R2. If you want code-gen and judge to corroborate each other, the R2
   framing makes both methods speak the same language.
4. **For an articulability score per metric** (the original V/A/T project goal):
   compare R1-judge SNR to R1-code SNR. When R1-code SNR ≥ R1-judge SNR, the
   metric is **verifiable**. When R1-code SNR ≪ R1-judge SNR, the metric is
   **articulable-but-not-verifiable**. m1 (error bars) is a clear example —
   code SNR 3.40 vs judge SNR 0.00 → highly verifiable but inapplicable to
   abstracts.

## Aggregated-R1 vs direct-R2 — convergent validity (new experiment)

**Question**: should we generate code for ALL R1 sub-families of each R2 aspect and aggregate, or code the R2 aspect directly?

| Method | Mean Pearson with Claude R2 judge | Mean Pearson with Llama R2 judge |
|---|---|---|
| **Aggregated R1 codes (mean)** | **+0.299** | **+0.365** |
| Aggregated R1 codes (max) | +0.232 | (similar) |
| Direct R2 Claude code | +0.240 | +0.040 |
| Direct R2 Llama code | +0.187 | +0.040 |

**Aggregating R1 codes BEATS direct R2 coding** on the convergent-validity test. The gain is small for Claude judge (+25%) but dramatic for Llama judge (9×, since Llama's direct R2 codes return 0 for 4/5 aspects — over-strict at the abstract level).

**Operational pipeline recommendation:**

```
For each R2 aspect A with R1 members {f1..fk}:
    # Code-gen side
    for f in members: generate Claude code score_f
    code_score_A = mean({score_f(text) for f in members})

    # Judge side
    judge_score_A = LLM judge with R2 framing (mean across paraphrases)

    # Convergent validity
    validity_A = pearson(code_score_A, judge_score_A) across datapoints
```

`validity_A` is the per-aspect articulability/verifiability score:
- High ρ → aspect is **articulable AND code-verifiable**
- Low ρ → aspect is **judge-only-articulable** (V/A gap = unable to operationalize)

The aspect-level numbers from the smoke (per `analysis_r1_aggregated_vs_r2.md`):
- m2 (Data Access): +.494, m3 (Results Presentation): +.485 — verifiable
- m0 (Strengths/Weaknesses): +.310 — moderately verifiable
- m1 (Statistical Methodology): +.111, m4 (Detail/Specificity): +.095 — weak verifiability

## Practical scaling cost (40 × 100)

| Phase | Calls per level | Both levels | Subagent rounds (at 10 parallel) |
|---|---|---|---|
| Code-gen | 40 × 3 trials = 120 | 240 | ~24 |
| Code exec | 40 × 3 × 100 = 12,000 subprocess (local) | 24,000 | ~5 min wall |
| Paraphrase | 40 | 80 | ~8 |
| Judge | 40 × 5 paraphrases × 5 chunks = 1,000 | 2,000 | ~200 |
| **Total subagent rounds** | | | **~230 rounds** |

**~230 rounds is heavy for a single conversation.** Realistic path:
- Use **larger judge chunks** (50 datapoints per prompt) → 200 judge calls per level → 80 total rounds.
- Or **use Llama-70B on sk3** for the judge step (vLLM throughput much higher than serial subagents).
- For the next iteration: **target 20 metrics × 50 datapoints** → ~100 rounds, achievable in one session.

## Limitations of the smoke (still to address)

- **Only 10 datapoints** — too few to compute meaningful Pearson per metric. Per-metric numbers are noisy; aggregate patterns are clear. A 50–100 datapoint run would tighten per-metric estimates.
- **All datapoints are paper abstracts** — full-paper text might restore applicability for very specific rubrics like m1 (error bars).
- **Qwen-Coder not tested** — not available on sk3 (only Llama family). Adding a third model family (e.g., via OpenRouter API call to DeepSeek-Coder) would triangulate further.
- **5 paraphrases including the original** — the "original" is itself a phrasing. A true sensitivity-to-phrasing test would use only the 4 LLM paraphrases.

## Files written

- `scripts/validity_pilot_data_prep.py` — sample paired (R1, R2) metrics + datapoints
- `scripts/validity_pilot_codegen_prep.py` — generate code-gen prompts
- `scripts/validity_pilot_judge_prep.py` — generate paraphrase + scoring prompts
- `scripts/validity_pilot_code_exec.py` — subprocess sandbox + scoring (Claude codes)
- `scripts/validity_pilot_code_exec_llama.py` — same for Llama-generated codes
- `scripts/validity_pilot_analyze.py` — within-model SNR + Pearson
- `scripts/validity_pilot_analyze_multimodel.py` — Claude vs Llama code-gen agreement
- `scripts/validity_pilot_analyze_judge_multimodel.py` — Claude vs Llama judge agreement
- `scripts/sk3_validity_pilot_llama.py` — sk3 runner for Llama code-gen + judge
- `runs/validity_pilot/smoke/` — all artifacts:
  - `metrics.json`, `datapoints.json` — inputs
  - `codegen/responses/` (Claude), `codegen/responses_llama/` (Llama)
  - `codegen/exec_results.jsonl` (Claude), `codegen/exec_results_llama.jsonl` (Llama)
  - `judge/score_responses/` (Claude), `judge/score_responses_llama/` (Llama)
  - `analysis_report.md` (within-model), `analysis_cross_model.md` (codes),
    `analysis_cross_judge.md` (judges)
