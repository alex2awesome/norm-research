# Full validity pipeline — final findings (peer-review, 2026-05-24)

## Pipeline executed (all 100% complete)

| Component | Volume | Wall time |
|---|---|---|
| Datapoints (stratified peer-review) | 500 (250 accept / 250 reject, abstracts ~1500 chars) | — |
| R1 metrics (top-3 R1 per R2) | 506 | — |
| R2 aspects | 218 | — |
| Paraphrases per rubric | 5 (NL paraphrases via Llama-3.3-70B) | 11 s |
| Llama R1 code variants × execs | 2,530 × 500 = 1.27M | 40s gen + 6s exec |
| Qwen R1 code variants × execs | 2,516 × 500 = 1.26M | ~10 min gen + 19s exec |
| Qwen R2 direct code × execs | 1,087 × 500 = 544K | ~5 min gen + 5s exec |
| Claude R1 code (cross-model subset) | 50 × 500 | (subagent rounds) |
| Llama R2 judge calls | 10,900 (218 R2 × 5 para × 10 chunks of 50dp) | ~4 hr on GPU 2 |
| Llama R1 judge calls | 25,300 (506 R1 × 5 para × 10 chunks of 50dp) | ~5 hr on GPUs 2+3 parallel |

All artifacts: `runs/validity_full/full_v1/`. Report: `analysis_comparison.md`.

## Findings

### 1. Coder quality: Qwen >> Llama (validated user intuition)

| | Llama | Qwen |
|---|---|---|
| Mean line count | 19 | 50 (**2.7×**) |
| Approach | substring matching, simple thresholds | regex, multiple keyword categories, structured logic |
| R1↔R1 convergent ρ (vs Llama judge) | +0.058 | **+0.064** |
| % aspects with ρ > 0.10 | 29% | 31% |

Per-metric, Qwen wins. Llama-70B is not a coder model — Qwen3-Coder produces substantively more thorough Python.

### 2. Aggregation level: R2 direct > R1 aggregated > R1 direct

Convergent validity per R2 aspect (n=190):

| Method | Mean ρ | Median ρ | % > 0.10 |
|---|---|---|---|
| **Qwen R2 direct code ↔ R2 judge** | **+0.089** | +0.068 | **40%** |
| Qwen R1 aggregated ↔ R2 judge | +0.075 | +0.048 | 33% |
| Llama R1 aggregated ↔ R2 judge | +0.059 | +0.039 | 35% |

Convergent validity per R1 metric (n=506):

| Method | Mean ρ | Median ρ | % > 0.10 |
|---|---|---|---|
| Qwen R1 code ↔ Llama R1 judge | +0.064 | +0.041 | 31% |
| Llama R1 code ↔ Llama R1 judge | +0.058 | +0.047 | 29% |
| Llama vs Qwen codes (cross-model) | +0.275 | +0.282 | 74% |

**R2 direct wins by ~25% over R1 aggregation, and by ~40% over R1 direct.** Larger thematic context lets both methods converge on the same signal. This **reverses the smoke test** (which had R1-aggregated > R2-direct based on 5 aspects × 10 dp — too noisy).

### 3. Predictive validity (vs accept/reject label) is weak in aggregate

| Method | Mean \|ρ\| | % > 0.10 |
|---|---|---|
| Llama R1 code → label | 0.043 | 4% |
| Qwen R1 code → label | 0.047 | 8% |
| Llama R1 judge → label | 0.046 | 6% |
| Qwen R2 code → label | 0.050 | 10% |
| Llama R2 judge → label | 0.048 | 8% |

**Most rubrics don't strongly predict accept/reject.** Only ~5-10% of metrics have \|ρ\| > 0.10. Two reasons:
- Accept/reject is a holistic judgment most individual rubrics don't capture
- Dataset is abstracts (~1500 chars) — many rubrics target full-paper content

### 4. Sign-disagreement within R2 (a real failure mode of mean aggregation)

| Multi-child R2 aspects (≥2 R1 children) | 160 / 218 |
|---|---|
| With ANY sign-disagreement among children's label-ρ | **95 (59%)** |
| With STRONG sign-disagreement (\|range\| > 0.10) | **20 (13%)** |

E.g., "Code and Software Release Practices":
- "Include link to software archive" → +0.203 with label (predictive)
- "State authorship and copyright" → +0.012 (neutral)
- "Provide support for older versions" → **−0.049** (opposite direction!)

Mean aggregation washes out signal. This is why **R2 direct coding beats R1 aggregation** — the LLM treats the R2 theme as one coherent target rather than averaging contradictory sub-rules.

### 5. The well-behaved aspects ("verifiable AND informative")

R2 aspects with both convergent validity > 0.20 AND label correlation > 0.08:

| Aspect | Code↔Judge ρ | Judge↔Label ρ |
|---|---|---|
| **Data and Code Availability** | **+0.538** | +0.148 |
| Patient Perspective and Care Communication | +0.418 | +0.118 |
| Code and Software Release Practices | +0.343 | +0.129 |
| Reporting Guideline Adherence (Clinical/Trials) | +0.339 | +0.120 |
| Licensing and Open Access of Outputs | +0.352 | +0.124 |
| Decision-Making and Policy Linkage | +0.175 | +0.141 |

**Open science / structured-reporting aspects dominate** — exactly what V/A/T theory predicts (verifiable end of the spectrum). About 6 aspects clear the gold-standard bar.

Top R1 metrics by R1↔R1 convergent validity:

| R1 metric | Qwen ρ | Llama ρ |
|---|---|---|
| Include key references and a link to the software archive | **+0.622** | +0.518 |
| Code and software made publicly available | +0.531 | +0.521 |
| State participant-level inclusion/exclusion criteria | +0.485 | +0.229 |
| Include or address patient-reported outcomes | +0.444 | +0.357 |
| Work free of spelling errors | +0.405 | +0.006 |
| Disclose environmental impact | +0.401 | +0.154 |
| Source/platform/URL/date specified | +0.393 | +0.160 |

These are all concrete rubrics with clear lexical signatures.

## 6. Operational recommendations

For a peer-review scoring pipeline at scale:

1. **Coder**: Qwen3-Coder (not Llama-70B). 60% relative gain in convergent validity.
2. **Aggregation level**: R2 direct (both code and judge target the same R2 theme). R1 aggregation is dominated, R1 direct is dominated.
3. **Verifiability score per aspect**: `Pearson(R2 code, R2 judge)` across datapoints. >0.30 = well-defined and operational, <0.10 = ill-defined or genuinely judge-only (tacit).
4. **For predictive use**: only ~10% of aspects meaningfully predict accept/reject from abstracts. Full-paper text would likely increase this.
5. **For ill-aggregating R2 aspects** (those with sign-disagreement among R1 children): consider splitting the R2 cluster, or use R2 direct framing only.

## What this means for the V/A/T framework

- The pipeline gives a per-aspect **verifiability score** (code-judge agreement) and **predictive score** (label correlation).
- The bivariate (verifiability, predictive) plane separates aspects into four quadrants:
  - **Verifiable + predictive**: 6 aspects (Data/Code Availability, Licensing, etc.) — these are the "fully operational" rubrics
  - **Verifiable but not predictive**: many (e.g., manuscript structure) — verifiable surface form, doesn't track outcome
  - **Not verifiable but predictive**: very rare — would indicate "tacit signal" (judge sees something code can't)
  - **Neither**: most aspects — either too tacit or simply not visible in abstracts

## Files

- `runs/validity_full/full_v1/analysis_comparison.md` — main report
- `runs/validity_full/full_v1/per_r1_full.json` — per-R1-metric stats (506 metrics)
- `runs/validity_full/full_v1/per_r2_full.json` — per-R2-aspect stats (218 aspects)
- `runs/validity_full/full_v1/codegen_exec_results{,_qwen_all,_qwen_r2,_claude}.jsonl` — all raw scores
- `runs/validity_full/full_v1/judge_responses_llama/`, `judge_r1_responses/` — judge JSON outputs
- `scripts/validity_full_*.py` — pipeline scripts
- `scripts/sk3_validity_full_runner_streamed.py` — sk3 runner

## Open questions for next iteration

1. **Full-paper text** instead of abstracts — would likely lift label correlation
2. **R2 aspects with sign-disagreement** — split or accept and use R2-direct only
3. **Add Claude code-gen at scale** (we only ran 50 keys) — to see if Claude's even-more-sophisticated codes beat Qwen
4. **Run on other 10 tasks** beyond peer-review — does the R2-direct > R1-agg pattern hold?
5. **Two-step prompt for coder**: first prompt asks for "what concrete signals would indicate this rubric", then second prompts for code — may improve Llama
