# Math.SE v3.3 (`math_se_v3_3_propensity_balanced`) Leakage / Confound Audit

Date: 2026-06-10. Script: `v3_3_audit.py` (this folder). Dataset:
`../math_se_v3_3_propensity_balanced.csv.gz` (99,722 rows; 98,770 unique
questions; train=79,873 / test=9,854; question-disjoint splits verified — 0
train/test question overlap). All models fit on `train` only; all reported
AUCs are on `test`. Pos rates: train 0.5003, test 0.4975.

v3.3 = v3 rebalanced within **cross-fitted question-propensity deciles × year**
(`propensity_balance_v3_3.py`). The dataset carries the build-time `propensity`
and `decile` columns; both are probed below.

## 0. v3 vs v3.3 comparison (all test AUCs)

| Probe | v3 | v3.3 | Δ |
|---|---|---|---|
| Answer-only word 1-2 grams TF-IDF+LR | 0.6584 | **0.6212** | -0.037 |
| Answer-only char 3-5 grams TF-IDF+LR | 0.6812 | **0.6428** | -0.038 |
| Full text (Q+A) word TF-IDF+LR | 0.6832 | 0.5900 | -0.093 |
| **Question-only word TF-IDF+LR** | **0.6496** | **0.4690** | **-0.181** |
| answer_year (single feature, abs) | 0.5978 | 0.5073 | -0.090 |
| All-structural LR (10 features) | 0.6106 | 0.5608 | -0.050 |
| propensity column (single feature) | n/a | 0.5003 | — |
| Pre-2014 pos rate | 0.7535 (14.5% of rows) | 0.5906 (12.3% of rows) | — |
| Answer-word margin over question floor | +0.009 | **+0.152** | — |

Headline: **the question-only floor is dead** (0.6496 → 0.4690, i.e. at/below
chance), **the year confound is dead** (0.5978 → 0.5073), and the
**propensity column itself is dead on test (0.5003 — ~0.5 by construction, as
expected)**. Answer-text AUCs dropped only ~0.04, so most of what the v3
answer models were reading was the question-side/era confound; the surviving
0.62-0.64 answer-only signal now stands on its own — the answer-over-floor
margin grows from +0.01 to +0.15.

## 1. TF-IDF + Logistic Regression test AUCs

| Model | Input | Train AUC | Test AUC | # features |
|---|---|---|---|---|
| Word 1-2 grams (min_df=5, ≤200K) | ANSWER only | 0.8715 | **0.6212** | 168,017 |
| Char 3-5 grams | ANSWER only | 0.8091 | **0.6428** | 200,000 |
| Word 1-2 grams | Full text (Q+A) | 0.8856 | 0.5900 | 200,000 |
| Word 1-2 grams | QUESTION only | 0.8798 | **0.4690** | 166,104 |

Key reading: the question-only model now generalizes at/below chance (train
0.88 → test 0.469) — question topic/wording no longer predicts the label.
The slightly-below-0.5 value is the known mild in-family overcorrection from
balancing against a TF-IDF+LR propensity model (matches the build manifest's
0.461 floor). Curiously the full-text model (0.5900) is now *worse* than the
answer-only model (0.6212): question tokens act as noise/anti-signal rather
than free AUC. Caveat from the README stands: the floor guarantee is relative
to the TF-IDF+LR family; a stronger question-side model (LLM) may recover
some floor — re-check when judges enter the picture.

## 2. Top ±40 answer-word features — leakage scrutiny

Full lists: `top_features_answer_word.csv`, `top_features_question_word.csv`.

| Feature | Coef | Verdict | Reasoning |
|---|---|---|---|
| `added` / `edit` | +2.02 / +1.89 | **leakage** | Post-publication edit markers; edits accrue after votes/acceptance (carried over from v3) |
| `my comment` | +1.65 | **leakage** | References the comment thread — engagement artifact, post-hoc |
| `my` / `my question` | -3.67 / -1.82 | ambiguous→leakage | Asker-voice in the answer slot = self-answers |
| `found` / `figured` / `answer` / `question` | -1.68 / -1.71 / -3.15 / -2.68 | ambiguous→leakage | "I found/figured out" self-answer discourse |
| `https` | -1.68 | ambiguous | Link-only answers genuinely low quality, but also era/style-correlated (weakened from v3's -3.05) |
| `nice` / `tag` | +2.07 / +2.22 | ambiguous | Politeness/meta-reference engagement markers |
| `think` / `not sure` / `maybe` / `sorry` / `seems` / `believe` | -3.67 / -2.22 / -2.08 / -1.96 / -1.70 / -1.59 | quality-signal | Genuine hedging |
| `boxed` / `align` / `operatorname` / `ldots` / `underbrace` / `color` | +2.33 / +1.94 / +1.73 / +1.67 / +1.67 / +1.60 | quality-signal | LaTeX-richness / careful typesetting |
| `no` / `in fact` / `indeed` / `note that` / `counterexample` / `proof of` | +2.22 / +1.96 / +1.82 / +1.58 / +1.66 / +1.61 | quality-signal | Direct-answer openings and mathematical discourse |

Same picture as v3: no gross leakage tokens (`thanks`, `upvote`, `accepted`
absent); the residual leakage is second-order (edit markers, comment
references, self-answer voice) with coefficients comparable to genuine quality
terms. The v3 recommendation to strip "Added:"/"EDIT:" suffixes and comment
references in the modeling pipeline still applies. Question-word features are
not meaningfully interpretable here (the model is below chance on test); note
the year token `2015` (+1.29) appears, consistent with memorization rather
than signal.

## 3. Single-feature test AUCs (structural probes)

| Feature | Raw AUC | abs AUC | Higher predicts | Flag (>0.55) |
|---|---|---|---|---|
| n_answers_on_question | 0.4518 | 0.5482 | negative | — (was 0.5311 in v3) |
| n_display_math | 0.5356 | 0.5356 | positive | — |
| n_latex_blocks | 0.5296 | 0.5296 | positive | — |
| answer_age_gap_days | 0.4796 | 0.5204 | negative | — |
| answer_position | 0.4886 | 0.5114 | negative | — |
| answer_year | 0.4927 | 0.5073 | negative | — (was **0.5978** in v3) |
| answer_len_chars | 0.5029 | 0.5029 | positive | — |
| answer_len_tokens | 0.4996 | 0.5004 | negative | — |
| has_numbered_list | 0.4996 | 0.5004 | negative | — |
| propensity | 0.5003 | 0.5003 | positive | — (dead by construction ✓) |
| n_paragraphs | 0.5000 | 0.5000 | positive | — |

**No single feature exceeds the 0.55 flag.** All-structural-features LR
(10 features, propensity excluded): test AUC **0.5608** (v3: 0.6106) — the
year driver is gone; what remains is mostly `n_answers_on_question` (0.5482,
mildly up from v3 — more answers on a question means more competition for the
chosen slot, an inherent property of the sampling unit, not the answer text).

### Propensity column check

AUC of the raw `propensity` column on test = **0.5003**. By construction the
v3.3 build balances classes within propensity deciles, so propensity should
carry zero label information — confirmed; the column is dead on test.

## 4. Class balance by answer_year (drift check)

| Year | n | pos_rate |
|---|---|---|
| 2010 | 160 | 0.6938 |
| 2011 | 1,001 | 0.6154 |
| 2012 | 2,513 | 0.4417 |
| 2013 | 8,543 | 0.6295 |
| 2014 | 10,846 | 0.4739 |
| 2015 | 12,015 | 0.4315 |
| 2016 | 10,375 | 0.5296 |
| 2017 | 11,643 | 0.4856 |
| 2018 | 9,892 | 0.4858 |
| 2019 | 8,539 | 0.5208 |
| 2020 | 8,655 | 0.4756 |
| 2021 | 6,392 | 0.5053 |
| 2022 | 4,222 | 0.5261 |
| 2023 | 3,906 | 0.4916 |
| 2024 | 1,020 | 0.4245 |

The v3 monotone drift (0.94 → 0.33 across years) is gone. Pre-2014 pos rate
fell from 0.7535 to 0.5906 (12.3% of rows); residual wobble is non-monotone
(2013 at 0.63 vs 2012 at 0.44) and nets out: answer_year single-feature AUC
is 0.5073. Max abs deviation from 0.5 among years with n≥200 is 0.1295
(2013). Year is no longer a usable shortcut, though per-year balance is not
exact — the balancing was decile×year jointly, and within-cell parity doesn't
force exact marginal year parity. Decile-by-class balance
(`decile_balance.csv`) is exactly 0.5000 in all 10 deciles.

## 5. Within answer_position == 1 (position residual)

- n (all splits) = 78,037, pos_rate = 0.5047; n (test) = 7,742, pos_rate = 0.5044
- Length (chars) AUC within position-1 test rows: **0.5042** — chance.

Per-position balance (`position_balance.csv`): positions 1-2 (95.3% of data)
at 0.50/0.48; positions ≥5 drift negative but cover <0.6% of rows. Unchanged
from v3 — the position controls continue to hold.

## 6. Question-only model

Test AUC 0.4690 (see §1) — the floor the v3.3 build was designed to kill, and
it is killed (slightly below chance = mild in-family overcorrection,
acceptable per the build gate: input floor 0.665 → output floor 0.461; this
audit independently reproduces 0.469 with its own vectorizer config).

## Verdict

**v3.3 resolves both mandatory caveats from the v3 audit and is clean for V/A
metric work:**

1. **Year drift fixed.** answer_year AUC 0.5978 → 0.5073; pre-2014 pos rate
   0.75 → 0.59 with no monotone trend. No need to drop pre-2014 rows.
2. **Question-only floor killed.** 0.6496 → 0.4690. Metric/judge AUCs on v3.3
   no longer need to clear a ~0.65 topic-predictability baseline; the
   answer-only linear signal (0.62-0.64) is now attributable to the answer.
3. **Propensity/decile columns are inert.** propensity AUC 0.5003 on test;
   all 10 deciles exactly 50/50. Safe to leave the columns in the file.
4. **No structural feature above 0.55.** Largest is n_answers_on_question
   (0.5482), an inherent sampling property; length, LaTeX, paragraphs,
   position all ≤0.536.

Remaining caveats (carried forward, not blockers):

- **Second-order text leakage persists**: `added`/`edit` markers, `my comment`
  references, self-answer voice (`my question`, `found`, `figured`) still rank
  in the top ±40 coefficients. Strip edit suffixes / comment references in the
  modeling pipeline if a metric suspiciously keys on them.
- **The floor guarantee is TF-IDF+LR-family-relative.** An LLM question-side
  baseline may recover part of the floor; report a question-only LLM baseline
  alongside any LLM judge result.

## Artifacts

- `results.json` — all AUCs
- `top_features_answer_word.csv`, `top_features_question_word.csv` — ±40 LR coefficients
- `single_feature_aucs.csv`, `year_balance.csv`, `decile_balance.csv`, `position_balance.csv`
