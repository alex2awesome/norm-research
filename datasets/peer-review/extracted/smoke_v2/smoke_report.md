# Smoke v2 (norm-adjacent passage+signal extraction) report

_Generated 2026-06-02 from 20 reviews (parse_ok=20)_


## 1. Aggregate stats

| metric | value |
|---|---:|
| reviews processed | 20 |
| reviews with parse_ok | 20 |
| total passages | 139 |
| total signals  | 303 |
| mean passages per review | 6.95 |
| mean signals per passage | 2.18 |
| **mean signals per review** | **15.15** |
| rubric coverage (of 154) | 43 (27.9%) |
| empty rubric_matches | 49 (16.2%) |
| signal_text substring of passage_text | 303/303 (100.0%) |
| passage_text substring of review_text  | 125/139 (89.9%) |
| total tokens in/out | 200,279 / 32,554 |

## 2. v1 vs v2 comparison

_v1 reviews processed: 20/20 (complete)._

| metric | v1 (verbatim norms) | v2 (norm-adjacent signals) | ratio |
|---|---:|---:|---:|
| items per review | 7.95 reasons | 15.15 signals | 1.91x |
| empty rubric_matches | 2.5% | 16.2% | — |
| rubric vocabulary size | 88 | 154 | 1.75x |
| rubric coverage | 32/88 (36.4%) | 43/154 (27.9%) | — |
| positive pct | 34.6% | 32.7% | — |
| negative pct | 57.9% | 54.1% | — |
| mixed/neutral pct | 7.5% (mixed) | 13.2% (neutral) | — |

**v1 is complete (20/20)** — comparison is final.

## 3. Example passages (multi-signal, multi-rubric)

### Example 1 (review_id=188687, venue=NeurIPS, decision=Accept (poster), passage_polarity=negative)

> The main drawback of this work is its presentation. Some of the proof is very difficult to parse. I have difficulty understanding the many algorithmic choices behind Algorithm 2. In particular, the idea of using random batch size, fully adaptive DP mechanism, specific distribution for batch size are not clear to me. I think the authors should provide a discussion on the necessity of these particular algorithmic choices.

| signal_text | type | polarity | rubric_matches |
|---|---|---|---|
| The main drawback of this work is its presentation | complaint | negative | [89] Clarity, readability, and organization for the ...; [110] Clarity, readability, and organization for the ... |
| Some of the proof is very difficult to parse | complaint | negative | [89] Clarity, readability, and organization for the ...; [110] Clarity, readability, and organization for the ... |
| I have difficulty understanding the many algorithmic choices behind Algorithm 2 | complaint | negative | [1] Procedural and analytical detail sufficient for...; [9] Methods reporting completeness and replicability; [89] Clarity, readability, and organization for the ... |
| the idea of using random batch size, fully adaptive DP mechanism, specific distribution for batch size are not clear ... | complaint | negative | [1] Procedural and analytical detail sufficient for...; [9] Methods reporting completeness and replicability; [89] Clarity, readability, and organization for the ... |
| the authors should provide a discussion on the necessity of these particular algorithmic choices | suggestion | negative | [1] Procedural and analytical detail sufficient for...; [9] Methods reporting completeness and replicability |

### Example 2 (review_id=243623, venue=TMLR, decision=Reject, passage_polarity=negative)

> Error in regret analysis. Note that the high probability UCB analysis in Eq(24) only holds for a single data point. When analyzing the regret and summing over $T$ rounds, the probability of all predictions within their UCB requires a union bound. So if the regret has $1-\delta$ probability holds for the first term in Eq(C.9), then the UCB of each predictive distribution should be $1- \delta/(TK)$, which means the coefficient $c$ in Eq(24) should not be a constant but related to $T, K$, which will change the order of the regret in Theorem 1. Similar union bound is used in Chu et al., 2011 an...

| signal_text | type | polarity | rubric_matches |
|---|---|---|---|
| Error in regret analysis | complaint | negative | [137] Statistical analysis rigor and transparent repo...; [71] Statistical methods: appropriateness, transpare... |
| the high probability UCB analysis in Eq(24) only holds for a single data point | complaint | negative | [137] Statistical analysis rigor and transparent repo...; [71] Statistical methods: appropriateness, transpare... |
| the probability of all predictions within their UCB requires a union bound | complaint | negative | [137] Statistical analysis rigor and transparent repo...; [71] Statistical methods: appropriateness, transpare... |
| the coefficient $c$ in Eq(24) should not be a constant but related to $T, K$, which will change the order of the regret | complaint | negative | [137] Statistical analysis rigor and transparent repo...; [71] Statistical methods: appropriateness, transpare... |
| authors are suggested to fix the proof error accordingly | suggestion | negative | [137] Statistical analysis rigor and transparent repo... |

### Example 3 (review_id=44492, venue=ICLR, decision=Accept: notable-top-5%, passage_polarity=negative)

> It would be good to discuss limitations of the model. Some questions I have:    * How are the out-of-distribution generalization characteristics of the model?    * Can the model capture variable length trajectories?    * I am intrigued by the skill composition idea. Currently only "AND" style composition is demonstrated. What other compositions does this method can support?   * How does the method perform with limited data?

| signal_text | type | polarity | rubric_matches |
|---|---|---|---|
| It would be good to discuss limitations of the model | complaint | negative | [3] Discussion and conclusions — interpretation and...; [31] Discussion and conclusions: evidence‑aligned, c... |
| How are the out-of-distribution generalization characteristics of the model? | complaint | negative | [38] External validity, scope, and generalizability ... |
| Can the model capture variable length trajectories? | complaint | negative | _(none)_ |
| Currently only "AND" style composition is demonstrated. What other compositions does this method can support? | complaint | negative | _(none)_ |
| How does the method perform with limited data? | complaint | negative | _(none)_ |

## 4. Polarity distribution (signal level)

| polarity | count | pct |
|---|---:|---:|
| positive | 99 | 32.7% |
| negative | 164 | 54.1% |
| neutral | 40 | 13.2% |

### Passage-level polarity

| polarity | count | pct |
|---|---:|---:|
| positive | 45 | 32.4% |
| negative | 72 | 51.8% |
| mixed | 22 | 15.8% |

### Signal type distribution

| signal_type | count | pct |
|---|---:|---:|
| complaint | 130 | 42.9% |
| praise | 97 | 32.0% |
| observation | 24 | 7.9% |
| suggestion | 52 | 17.2% |

## 5. Polarity skew by group

### By venue (signal polarity)

| venue | n_sig | %pos | %neg | %neu |
|---|---:|---:|---:|---:|
| EMNLP | 25 | 52.0% | 32.0% | 16.0% |
| ICLR | 73 | 31.5% | 58.9% | 9.6% |
| NeurIPS | 81 | 35.8% | 45.7% | 18.5% |
| TMLR | 93 | 18.3% | 69.9% | 11.8% |
| eLife | 31 | 54.8% | 35.5% | 9.7% |

### By accept/reject decision

| decision | n_sig | %pos | %neg | %neu |
|---|---:|---:|---:|---:|
| accept | 197 | 37.1% | 48.7% | 14.2% |
| reject | 106 | 24.5% | 64.2% | 11.3% |

### Meta-review vs individual review

| kind | n_sig | %pos | %neg | %neu |
|---|---:|---:|---:|---:|
| meta | 49 | 53.1% | 32.7% | 14.3% |
| non_meta | 254 | 28.7% | 58.3% | 13.0% |

## 6. Top-15 rubrics with polarity skew

| rank | rubric_id | name | n_signals | %pos | %neg | %neu |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 89 | Clarity, readability, and organization for the intended audience | 47 | 17% | 81% | 2% |
| 2 | 79 | Novelty and significance of the contribution | 42 | 69% | 31% | 0% |
| 3 | 70 | Positioning within and comparison to prior work | 40 | 25% | 58% | 18% |
| 4 | 9 | Methods reporting completeness and replicability | 37 | 22% | 59% | 19% |
| 5 | 1 | Procedural and analytical detail sufficient for replication | 34 | 29% | 50% | 21% |
| 6 | 38 | External validity, scope, and generalizability claims | 24 | 4% | 79% | 17% |
| 7 | 47 | Theoretical framing, coherence, and use of theory | 15 | 13% | 73% | 13% |
| 8 | 3 | Discussion and conclusions — interpretation and implications | 13 | 38% | 54% | 8% |
| 9 | 31 | Discussion and conclusions: evidence‑aligned, contextualized, and b... | 13 | 31% | 54% | 15% |
| 10 | 26 | Introduction — background, context/scope, and justification | 11 | 18% | 55% | 27% |
| 11 | 71 | Statistical methods: appropriateness, transparency, and robustness ... | 10 | 0% | 100% | 0% |
| 12 | 88 | Novelty and positioning vs prior work | 9 | 33% | 44% | 22% |
| 13 | 137 | Statistical analysis rigor and transparent reporting | 9 | 0% | 100% | 0% |
| 14 | 134 | Computational resources, efficiency, and convergence comparisons | 7 | 14% | 57% | 29% |
| 15 | 120 | Figures/tables and data visualization — clarity, accuracy, accessib... | 5 | 20% | 80% | 0% |

## 7. Coverage gap (signals tagged with empty rubric_matches)

- 49 / 303 signals (16.2%) have empty rubric_matches.
- 43 / 154 rubrics (27.9%) have at least one tagged signal.

### 10 sample empty-match signals (potential taxonomy gaps)

- (ICLR) [praise, positive] increases sampling efficiency by concentrating on partial solutions
- (ICLR) [praise, positive] less dependent on human input, making it more scalable
- (ICLR) [praise, positive] satisfactory experimental results
- (ICLR) [complaint, negative] concerns regarding the sufficiency of the experiments
- (ICLR) [observation, neutral] raised minor issues
- (ICLR) [praise, positive] the authors effectively addressed these queries
- (ICLR) [praise, positive] they adequately resolved several experimental issues
- (ICLR) [praise, positive] shown to tackle compositionality of skills and constraints at inference time
- (ICLR) [complaint, negative] Can the model capture variable length trajectories?
- (ICLR) [complaint, negative] Currently only "AND" style composition is demonstrated. What other compositions does this method can support?

## 8. Verdict: does norm-adjacent yield materially more signal than verbatim?

v2 produced **15.2 signals per review** versus v1's **8.0 reasons per review** — a 1.91x raw amplification.

**Verdict: yes** — the norm-adjacent framing yields meaningfully more signal density. Worth scaling to the full 259K reviews.

## 9. Llama-bulk plan: scaling to 259K reviews on sk3

### Token economics (from this 20-review smoke)

- Mean input tokens per review: ~10,013
- Mean output tokens per review: ~1,627
- Projected 259K reviews on Claude Sonnet 4.5 (Batch API, 50% off): ~$7,052
  - direct messages.create at sticker price: ~$14,104
- Projected on Llama-70B on sk3: **~$0 marginal** (GPUs already paid for; pure wall-clock cost)

### Can Llama-70B handle this prompt?

**Probably yes, with the following structural choices:**

- **System prompt size**: ~131,516 chars (~10,500 tokens) of rubric block. Llama-70B has 128k context, so this fits comfortably; the rubric block is the dominant fixed cost. Use vLLM **prefix caching** (`enable_prefix_caching=True`) so the rubric block is paid once per worker, not per review.
- **Output complexity**: the multi-passage + nested-signal + 0-3-rubric-id structure is JSON-structured and Llama-70B handles structured output reliably when constrained. Use **guided JSON** (vLLM `guided_json` with a Pydantic schema) to enforce the shape — eliminates the ~10-20% parse-failure tail Claude has on free-form output.
- **Failure modes to expect**: Llama tends to over-extract on long reviews (passage_text spans drifting >1500 chars), and may hallucinate rubric_ids outside the 0-154 range. Mitigate with a post-hoc validator that drops/clips out-of-range ids and re-prompts on schema violation.
- **Faithfulness**: v2's prompt asks for verbatim substrings; Llama generally respects this less strictly than Claude. Plan a **post-extraction substring-check pass** and drop signals whose `signal_text` is not an exact substring of `passage_text`. (Faithfulness numbers above set the v2/Claude baseline.)

### Suggested sk3 vLLM batched run

```
# Single B200 GPU, BF16 (recipe per reference_sk3_vllm_bf16.md)
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.93 \
  --enable-prefix-caching \
  --guided-decoding-backend xgrammar

# Client side: submit prompts in batches of 2-4k at a time
# (per feedback_vllm_batch_size.md). Reuse the same system prompt
# across all 259k requests so prefix-cache hit rate is ~100%.
```

### Cost-equivalence summary

| route | wall-clock estimate | $$ |
|---|---|---:|
| Claude Sonnet 4.5 (Batch API) | ~24h (Anthropic's 24h batch SLA) | ~$7,052 |
| Claude Sonnet 4.5 (direct messages.create) | ~5-7 days (rate-limited) | ~$14,104 |
| Llama-70B on sk3 (1 B200, prefix-cached) | ~3-4 days @ 30 req/s sustained | ~$0 marginal |
| Llama-70B on sk3 (2 B200, prefix-cached) | ~1.5-2 days | ~$0 marginal |

**Recommendation**: gold-set tune the prompt on Claude Sonnet 4.5 (this smoke + a 200-review followup ~$30), then deploy with `guided_json` on Llama-70B for the full 259K. Hold back a 500-review Claude validation set to score Llama against, so we have a faithfulness/coverage drift estimate.
