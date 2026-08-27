# #HashtagWars — Gemma-4-31B articulated-criteria (A) readout

Judge: **google/gemma-4-31b-it**, local snapshot, offline-batch vLLM (no HTTP server),
one token from {1.0, 0.5, 0.0, NA} per (item, criterion), temperature 0, max_tokens 6,
prefix caching on, spawn multiprocessing, single GPU (GPU 2, gpu_memory_utilization 0.55,
stacked beside a running LoRA training job). Run date: 2026-07-29.

Label-blind: y never appears in a judge prompt, and no criterion was selected or
rewritten using y. The A bank (30 criteria) was authored earlier and is used verbatim.

Readouts are 5-fold `GroupKFold` logistic (C=1, median imputation + missingness
indicators, standardization inside each training fold), pooled out-of-fold ROC AUC.
Threshold-free: no accuracy, no thresholds.

## Sample

Precommitted label-blind subset reused verbatim from the earlier run: the first 40
hashtags under SHA-256("hashtagwars-va-v1:" + hashtag), every tweet retained.
**n = 4228: 397 positive / 3831 negative across
40 hashtags.** The contest hashtag is supplied as shared context; it is
constant within a contest and cannot separate entries inside the grouping unit.

## Grouped-CV AUC (full sample)

| Features | AUC |
|---|---:|
| V | 0.5592 |
| A | 0.6350 |
| V+A | 0.6478 |

## Balanced-within-hashtag readout

All positives plus an equal number of negatives sampled inside each hashtag (seed
20260728); n = 788 (397 pos / 391 neg,
40 hashtags). This is the readout comparable to the discarded codebook run.

| Features | AUC |
|---|---:|
| V | 0.5606 |
| A | 0.6131 |
| V+A | 0.6235 |

## Anchor check (3 blinded rows in every shard)

Each of the 8 shards carried a staff winner, a random unselected entry, and a
scrambled word-salad row, drawn with a shard-specific seed (so the 8 checks are
independent draws, not one repeated draw). Values are that anchor's mean A score.

| Shard | n items | anchor pos | anchor neg | anchor scrambled | ordering | attempts |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 512 | 0.913 | 0.781 | 0.059 | PASS | 1 |
| 1 | 511 | 0.917 | 0.848 | 0.000 | PASS | 1 |
| 2 | 497 | 0.891 | 0.646 | 0.000 | PASS | 1 |
| 3 | 545 | 0.925 | 0.780 | 0.056 | PASS | 4 |
| 4 | 547 | 0.960 | 0.917 | 0.000 | PASS | 2 |
| 5 | 531 | 0.932 | 0.861 | 0.000 | PASS | 3 |
| 6 | 551 | 1.000 | 0.929 | 0.000 | PASS | 1 |
| 7 | 534 | 0.891 | 0.842 | 0.350 | PASS | 1 |

Shards whose first 3-row draw failed were re-drawn (attempt count above); every
shard ended on a passing draw. The per-shard rows themselves are unchanged by a
re-draw (temperature 0, one independent prompt per item x criterion), so a re-draw
tests the anchor sample, not the shard's scores.

### Extended anchor battery

An extended battery of 12 independently drawn anchors per class (seeds disjoint from the shard anchors) was scored with the same judge, prompts and criteria:

| Anchor class | mean A | sd | n |
|---|---:|---:|---:|
| known positive | 0.929 | 0.041 | 12 |
| random negative | 0.772 | 0.149 | 12 |
| scrambled nonsense | 0.081 | 0.127 | 12 |

Item-level rank separation: positive vs negative AUC **0.854**; coherent (pos+neg) vs scrambled AUC **1.000**. Ordering on class means holds: **True**.

## Top articulated criteria (univariate)

Raw-direction ROC AUC of each criterion over the full sample; NA replaced by the
criterion's median for this descriptive statistic only. AUC never selected criteria.
Overall NA rate 0.283.

| Rank | Criterion | univariate AUC | NA rate | modal share |
|---:|---|---:|---:|---:|
| 1 | Surprising turn | 0.5837 | 0.110 | 0.538 |
| 2 | Memorable compact image or phrase | 0.5822 | 0.013 | 0.503 |
| 3 | Emotional attitude is legible | 0.5649 | 0.101 | 0.643 |
| 4 | Observation has recognizable truth | 0.5629 | 0.608 | 0.815 |
| 5 | Original angle within the prompt | 0.5575 | 0.014 | 0.395 |
| 6 | Rhythm supports the payoff | 0.5536 | 0.145 | 0.764 |
| 7 | Specific detail carries the joke | 0.5392 | 0.015 | 0.798 |
| 8 | Incongruous domains connect coherently | 0.5379 | 0.079 | 0.883 |
| 9 | Character voice is implied | 0.5372 | 0.530 | 0.830 |
| 10 | Recognizable source reference | 0.5301 | 0.127 | 0.918 |
| 11 | Self-contained comprehension | 0.5286 | 0.000 | 0.824 |
| 12 | Escalation increases the comic stakes | 0.5276 | 0.758 | 0.809 |

## Caveats

- Judge is Gemma-4-31B, a local open-weights model; A is the articulated-criteria channel as read by that judge, not a human editorial judgment.
- **Near-constant criteria (>98% one value, contribute no rank information): Transformation is easy to recover, The joke is not explained after landing, Sound-based wordplay is legible, Conversational phrasing sounds natural.**
- Criteria are correlated and several are conditionally applicable; median imputation plus missingness indicators is a modeling choice.
- Pooled out-of-fold AUC is descriptive; no confidence intervals are reported.
- No hyperparameter search, no AUC-driven criterion selection, no threshold tuning.

