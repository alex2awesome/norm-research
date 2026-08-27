# Creative writing (r/WritingPrompts) — Gemma-4-31B articulated-criteria (A) readout

Judge: **google/gemma-4-31b-it**, local snapshot, offline-batch vLLM (no HTTP server),
one token from {1.0, 0.5, 0.0, NA} per (item, criterion), temperature 0, max_tokens 6,
prefix caching on, spawn multiprocessing, single GPU (GPU 2, gpu_memory_utilization 0.55,
stacked beside a running LoRA training job). Run date: 2026-07-29.

Label-blind: y never appears in a judge prompt, and no criterion was selected or
rewritten using y. The A bank (45 criteria) was authored earlier and is used verbatim.

Readouts are 5-fold `GroupKFold` logistic (C=1, median imputation + missingness
indicators, standardization inside each training fold), pooled out-of-fold ROC AUC.
Threshold-free: no accuracy, no thresholds.

## Sample

Source: **canonical** `datasets/creative-writing/writingprompts_modeling_clean.csv.gz`
(the real grouped 50/50 score>=10 build present on sk3), not the local reconstruction
in this directory. Whole prompt groups taken in SHA-256("cw-va-v2-sample|" + prompt_id)
order until n >= 2000, exactly the rule recorded in `sample_inventory.json`.
**n = 2000: 1009 positive / 991 negative across
1500 prompt groups.** Grouping unit = prompt.

Deterministic truncation: stories longer than 6000 characters are shown
as the first 3600 characters + `[... DETERMINISTIC MIDDLE OMISSION ...]` + the
last 2400 characters. Writing prompts are truncated to 1200
characters. No randomness is involved.

## Grouped-CV AUC

| Features | AUC |
|---|---:|
| V | 0.6039 |
| A | 0.6053 |
| V+A | 0.6266 |

## Anchor check (3 blinded rows in every shard)

Per shard: a random positive story, a random negative story, and a word-salad scramble
of two stories, shard-specific seeds. Values are that anchor's mean A score.

| Shard | n items | anchor pos | anchor neg | anchor scrambled | ordering | attempts |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 533 | 0.722 | 0.233 | 0.000 | PASS | 2 |
| 1 | 492 | 0.644 | 0.631 | 0.023 | PASS | 1 |
| 2 | 477 | 0.739 | 0.678 | 0.000 | PASS | 2 |
| 3 | 498 | 0.844 | 0.463 | 0.024 | PASS | 2 |

### Extended anchor battery

An extended battery of 12 independently drawn anchors per class (seeds disjoint from the shard anchors) was scored with the same judge, prompts and criteria:

| Anchor class | mean A | sd | n |
|---|---:|---:|---:|
| known positive | 0.686 | 0.181 | 12 |
| random negative | 0.744 | 0.141 | 12 |
| scrambled nonsense | 0.008 | 0.010 | 12 |

Item-level rank separation: positive vs negative AUC **0.403**; coherent (pos+neg) vs scrambled AUC **1.000**. Ordering on class means holds: **False**.

## Top articulated criteria (univariate)

| Rank | Criterion | univariate AUC | NA rate | modal share |
|---:|---|---:|---:|---:|
| 1 | Prose economy | 0.5672 | 0.000 | 0.562 |
| 2 | Fresh premise or central angle | 0.5667 | 0.000 | 0.583 |
| 3 | Ending resonance | 0.5652 | 0.000 | 0.648 |
| 4 | Opening narrative traction | 0.5617 | 0.000 | 0.754 |
| 5 | Tonal control | 0.5605 | 0.000 | 0.640 |
| 6 | Controlled pacing | 0.5547 | 0.001 | 0.602 |
| 7 | Scene-level purpose | 0.5493 | 0.000 | 0.894 |
| 8 | Precise diction | 0.5482 | 0.000 | 0.594 |
| 9 | Exposition is integrated | 0.5469 | 0.000 | 0.477 |
| 10 | Focused dramatic through-line | 0.5454 | 0.000 | 0.786 |
| 11 | Sentence-level clarity | 0.5453 | 0.000 | 0.527 |
| 12 | Ending resolves or productively reframes | 0.5452 | 0.000 | 0.764 |

## Caveats

- Judge is Gemma-4-31B, a local open-weights model; A is the articulated-criteria channel as read by that judge, not a human editorial judgment.
- No criterion collapsed to a near-constant value (all modal shares < 0.98).
- Criteria are correlated and several are conditionally applicable; median imputation plus missingness indicators is a modeling choice.
- Pooled out-of-fold AUC is descriptive; no confidence intervals are reported.
- No hyperparameter search, no AUC-driven criterion selection, no threshold tuning.

- **The positive-vs-negative half of the anchor check does NOT hold for this task.**
  In the 12-per-class battery the positive anchors mean 0.686 and the negative anchors
  mean 0.744 (pos-vs-neg AUC 0.403, i.e. slightly reversed), and 3 of the 4 per-shard
  3-row checks passed only on the second anchor draw. What the anchors do establish
  robustly is coherence sensitivity: coherent-vs-scrambled AUC is 1.000 and the
  scrambled anchors mean 0.008. Read the A numbers below as "the judge applies these
  criteria to real prose and is not collapsed", not as "the anchor protocol certified
  positive/negative separation on this task". Both anchor classes are drawn from the
  same balanced pool, where a single score>=10 story is genuinely hard to tell from a
  single score<10 story; that is consistent with the modest A AUC of 0.605.
- Long stories are middle-omitted, so criteria about middle-of-story structure are
  judged on partial evidence.
