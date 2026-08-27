# Style Invitational — Gemma-4-31B articulated-criteria (A) readout

Judge: **google/gemma-4-31b-it**, local snapshot, offline-batch vLLM (no HTTP server),
one token from {1.0, 0.5, 0.0, NA} per (item, criterion), temperature 0, max_tokens 6,
prefix caching on, spawn multiprocessing, single GPU (GPU 2, gpu_memory_utilization 0.55,
stacked beside a running LoRA training job). Run date: 2026-07-29.

Label-blind: y never appears in a judge prompt, and no criterion was selected or
rewritten using y. The A bank (32 criteria) was authored earlier and is used verbatim.

Readouts are 5-fold `GroupKFold` logistic (C=1, median imputation + missingness
indicators, standardization inside each training fold), pooled out-of-fold ROC AUC.
Threshold-free: no accuracy, no thresholds.

## Sample

All 316 archived weeks were scored (a superset of the
110-week precommitted design in `score_va.py`), n = 9637 entries.
Grouping unit = `week_id`. The two y's are reported **separately and never merged**.

## y (a) top tier: winner ∪ runnerup vs honorable_mention

n = 9637: 1549 positive / 8088 negative,
316 weeks.

| Features | AUC |
|---|---:|
| V | 0.6227 |
| A | 0.6090 |
| V+A | 0.6161 |

### Top criteria (top-tier y)

| Rank | Criterion | univariate AUC | NA rate | modal share |
|---:|---|---:|---:|---:|
| 1 | Self-contained clarity | 0.5779 | 0.037 | 0.679 |
| 2 | Prompt task completion | 0.5683 | 0.004 | 0.687 |
| 3 | Prompt-specific relevance | 0.5614 | 0.029 | 0.632 |
| 4 | Explicit constraint satisfaction | 0.5408 | 0.218 | 0.772 |
| 5 | Comic payload focus | 0.5356 | 0.063 | 0.849 |
| 6 | Vivid comic image or consequence | 0.5342 | 0.083 | 0.601 |
| 7 | Specificity with payoff | 0.5167 | 0.123 | 0.805 |
| 8 | Verse form as comic leverage | 0.5088 | 0.949 | 0.974 |
| 9 | Coherent incongruity | 0.5056 | 0.156 | 0.834 |
| 10 | Meter and scansion control | 0.5046 | 0.966 | 0.979 |
| 11 | Topical bite | 0.5035 | 0.630 | 0.830 |
| 12 | Non-obvious comic choice | 0.5021 | 0.093 | 0.707 |

## y (b) winner vs rest

n = 9637: 322 positive / 9315 negative,
316 weeks.

| Features | AUC |
|---|---:|
| V | 0.6334 |
| A | 0.6070 |
| V+A | 0.6121 |

### Top criteria (winner-vs-rest y)

| Rank | Criterion | univariate AUC | NA rate | modal share |
|---:|---|---:|---:|---:|
| 1 | Self-contained clarity | 0.5603 | 0.037 | 0.679 |
| 2 | Prompt-specific relevance | 0.5367 | 0.029 | 0.632 |
| 3 | Prompt task completion | 0.5340 | 0.004 | 0.687 |
| 4 | Vivid comic image or consequence | 0.5326 | 0.083 | 0.601 |
| 5 | Explicit constraint satisfaction | 0.5189 | 0.218 | 0.772 |
| 6 | Comic payload focus | 0.5175 | 0.063 | 0.849 |
| 7 | Topical bite | 0.5137 | 0.630 | 0.830 |
| 8 | Specificity with payoff | 0.5119 | 0.123 | 0.805 |
| 9 | Verse form as comic leverage | 0.5114 | 0.949 | 0.974 |
| 10 | Meter and scansion control | 0.5073 | 0.966 | 0.979 |
| 11 | Coherent incongruity | 0.5043 | 0.156 | 0.834 |
| 12 | Surprising but legible turn | 0.5032 | 0.127 | 0.673 |

## Anchor check (3 blinded rows in every shard)

Per shard: a winner, a random honorable mention, and a scrambled word-salad entry,
shard-specific seeds. Values are that anchor's mean A score.

| Shard | n items | anchor pos | anchor neg | anchor scrambled | ordering | attempts |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 1204 | 0.717 | 0.583 | 0.043 | PASS | 1 |
| 1 | 1207 | 0.932 | 0.789 | 0.000 | PASS | 1 |
| 2 | 1218 | 0.700 | 0.639 | 0.000 | PASS | 1 |
| 3 | 1194 | 1.000 | 0.962 | 0.000 | PASS | 1 |
| 4 | 1260 | 1.000 | 0.571 | 0.000 | PASS | 2 |
| 5 | 1229 | 0.917 | 0.955 | 0.000 | FAIL | 4 |
| 6 | 1187 | 0.926 | 0.786 | 0.222 | PASS | 4 |
| 7 | 1138 | 0.912 | 0.857 | 0.056 | PASS | 1 |

**Shard [5] did NOT reach a passing 3-row ordering within 4
independent anchor draws and is recorded as INVALID.** Its rows are retained in the
headline readout and a leave-that-shard-out sensitivity readout is given below.
Re-drawing anchors cannot change a shard's item scores (temperature 0, one
independent prompt per item x criterion), so the failure is a property of the
anchor sample and of how little a single winner separates from a single
honorable mention under this judge, not of the shard's scoring run.

#### Sensitivity: invalid shard(s) dropped

**top_tier** (n = 8408, 1346 pos): V 0.6124 / A 0.6005 / V+A 0.6064

**winner_vs_rest** (n = 8408, 281 pos): V 0.6222 / A 0.6123 / V+A 0.6238

### Extended anchor battery

An extended battery of 12 independently drawn anchors per class (seeds disjoint from the shard anchors) was scored with the same judge, prompts and criteria:

| Anchor class | mean A | sd | n |
|---|---:|---:|---:|
| known positive | 0.835 | 0.176 | 12 |
| random negative | 0.597 | 0.468 | 12 |
| scrambled nonsense | 0.185 | 0.214 | 12 |

Item-level rank separation: positive vs negative AUC **0.556**; coherent (pos+neg) vs scrambled AUC **0.856**. Ordering on class means holds: **True**.

## Caveats

- Judge is Gemma-4-31B, a local open-weights model; A is the articulated-criteria channel as read by that judge, not a human editorial judgment.
- **Near-constant criteria (>98% one value, contribute no rank information): Explanation discipline, Rhyme quality, Misdirection fairness, Phonetic or orthographic precision.**
- Criteria are correlated and several are conditionally applicable; median imputation plus missingness indicators is a modeling choice.
- Pooled out-of-fold AUC is descriptive; no confidence intervals are reported.
- No hyperparameter search, no AUC-driven criterion selection, no threshold tuning.

- Archive bylines (author name, hometown) remain inside `entry_text`; the judge is
  instructed to ignore them, but they still affect the deterministic V features.
- The honorable-mention pool is itself already editor-selected, so the negative class
  is a *published* negative, not a random submission.
