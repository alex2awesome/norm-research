# Full validity pipeline — progress (2026-05-24, mid-run)

## What's run

| Component | Status | Volume |
|---|---|---|
| Paraphrase generation (Llama on sk3) | ✅ Done in 11s | 724 prompts (506 R1 + 218 R2) × 5 paraphrases each |
| Llama code-gen (all R1 × 5 paraphrases) | ✅ Done in 40s | 2,530 codes |
| Qwen code-gen (cross-model subset) | ✅ Done | 50 codes (10 R2 × top-1 R1 × 5 paraphrases) |
| Claude code-gen (cross-model subset) | ✅ Done | 50 codes (same keys as Qwen) |
| Code execution (in-process) | ✅ Done in 6s | 1.26M Llama + 24K Qwen + 25K Claude executions |
| Llama judge (218 R2 × 5 paraphrases × 10 chunks of 50dp) | ⏳ Running (streamed) | 10,900 prompts, ETA ~2.5 hr |
| Final convergent + predictive analysis | ⏳ Pending judge | — |

All artifacts at `runs/validity_full/full_v1/`. Datapoints: 500 stratified peer-review papers (50/50 accept/reject).

## Cross-model code-gen findings (3-way subset, 48 keys)

| Model pair | Mean Pearson | Median |
|---|---|---|
| Llama vs Qwen | +0.227 | +0.212 |
| **Llama vs Claude** | **−0.006** | +0.037 |
| Qwen vs Claude | +0.159 | +0.124 |

Mean score per model on the subset:
- Llama: 0.163 (most conservative)
- Qwen: 0.350
- Claude: 0.379 (most permissive)

**Llama is dramatically more conservative than Claude/Qwen**, and the Llama↔Claude correlation is essentially zero — they extract structurally different operationalizations from the same rubric. This is a real cross-model bias finding that the smoke test had hinted at.

## Code-only paraphrase convergence

Across 5 paraphrases of the same rubric (aggregated across R1 children per aspect):

| σ_para bin | Count | What it means |
|---|---|---|
| [0.00, 0.05) | 18 | Highly stable rubric |
| [0.05, 0.10) | 60 | Stable |
| [0.10, 0.15) | 71 | Moderate |
| [0.15, 0.20) | 35 | Sensitive |
| [0.20, 1.00) | 34 | Very sensitive |

Mean σ_para = 0.130; median = 0.116. **Most rubrics are paraphrase-stable on the code side** (78/218 in lowest two bins).

## Code-only predictive validity (vs accept/reject label)

Across 218 aspects:
- Mean |Pearson(code, label)| = 0.044
- Mean AUC = 0.507 (weak overall — expected at the abstract level)
- 9 aspects with |ρ| > 0.1 (real signal)
- 18 aspects with AUC > 0.55 or < 0.45 (informative)

### Top 20 aspects by |code↔label| Pearson

| Aspect | n R1 | code↔label ρ | AUC |
|---|---|---|---|
| **Code and Software Release Practices** | 3 | **+0.173** | 0.552 |
| **Data and Code Availability** | 3 | **+0.148** | 0.543 |
| Conceptual Connections to Related Fields | 1 | −0.124 | 0.429 |
| Introduction and Background | 1 | +0.121 | 0.564 |
| Findability and Accessibility of Outputs | 3 | +0.117 | 0.569 |
| Supporting Material Use | 2 | +0.112 | 0.553 |
| Author Response and Revision Practices | 3 | −0.108 | 0.440 |
| Language Eligibility | 1 | −0.103 | 0.460 |
| Title, Abstract, and Keywords | 3 | +0.102 | 0.559 |
| Sample Size and Power | 3 | −0.100 | 0.440 |
| Strengths, Weaknesses and Limitations | 2 | +0.099 | 0.559 |
| Replication Study Requirements | 1 | −0.097 | 0.483 |
| Interdisciplinarity and Disciplinary Voices | 3 | +0.097 | 0.545 |
| Trial Design and Controls | 3 | +0.094 | 0.564 |
| Machine Readability | 1 | −0.092 | 0.444 |
| Citation Specificity and Formats | 3 | +0.090 | 0.530 |
| AI Use and Disclosure | 3 | +0.088 | 0.559 |
| Author Response and Revision | 3 | −0.087 | 0.440 |
| Data Leakage and Bias Prevention | 2 | −0.087 | 0.454 |
| Hardware and Capture Setup Documentation | 1 | −0.087 | 0.469 |

Sensible signals:
- **Positive**: papers that discuss code/data sharing, have clear intro/title/abstract, mention controls → more likely accepted.
- **Negative**: papers that emphasize "conceptual connections" (often handwavy), discuss author-response context (probably rejection-history bias), or hit "machine readability" issues → more likely rejected.

## Pending (when judge finishes)

- Convergent validity per aspect (code-aggregated vs Llama-judge Pearson)
- Judge↔label correlation per aspect (compare to code↔label)
- Judge paraphrase convergence per aspect
- Final R1-vs-R2 determination at scale

## Scripts ready (for repeat / extend)

| Script | Role |
|---|---|
| `scripts/validity_full_data_prep.py` | Sample R2 aspects + R1 sub-families + datapoints |
| `scripts/validity_full_paraphrase_prep.py` | 5 NL paraphrases per rubric |
| `scripts/validity_full_codegen_prep.py` | Code-gen prompt per (R1, paraphrase) |
| `scripts/validity_full_judge_prep.py` | Judge prompt per (R2, paraphrase, chunk) |
| `scripts/validity_full_qwen_codegen.py` | Async OpenRouter Qwen runner |
| `scripts/sk3_validity_full_runner.py` | Llama on sk3 (single-pass) |
| `scripts/sk3_validity_full_runner_streamed.py` | Llama on sk3 (incremental flush) |
| `scripts/validity_full_exec_codes.py` | Subprocess sandbox exec (slow, safe) |
| `scripts/validity_full_exec_inproc.py` | In-process exec (~100× faster, less safe) |
| `scripts/validity_full_analyze.py` | Convergent + predictive validity per aspect |
| `scripts/validity_full_code_only_report.py` | Intermediate code-only stats |
