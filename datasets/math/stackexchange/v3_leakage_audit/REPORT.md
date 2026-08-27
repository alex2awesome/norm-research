# Math.SE v3 (`math_se_v3_position_matched`) Leakage / Confound Audit

Date: 2026-06-10. Script: `audit.py` (this folder). Dataset:
`../math_se_v3_position_matched.csv.gz` (100,000 rows; 98,914 unique questions;
train=80,057 / eval=9,999 / test=9,944; question-disjoint splits verified — 0
train/test question overlap). All models fit on `train` only; all reported
AUCs are on `test`. Pos rates: train 0.5001, test 0.5054, eval 0.4940.

## 1. TF-IDF + Logistic Regression test AUCs

| Model | Input | Train AUC | Test AUC | # features |
|---|---|---|---|---|
| Word 1-2 grams (min_df=5, ≤200K) | ANSWER only | 0.8669 | **0.6584** | 174,260 |
| Char 3-5 grams | ANSWER only | 0.8082 | **0.6812** | 200,000 |
| Word 1-2 grams | Full text (Q+A) | 0.8760 | 0.6832 | 200,000 |
| Word 1-2 grams | QUESTION only | 0.8668 | **0.6496** | 170,252 |

Key reading: the **question-only model reaches 0.6496** despite splits being
question-disjoint — question topic/wording alone predicts the label almost as
well as the answer text (word-level 0.6584). The answer adds only ~+0.03 AUC
over the question-only floor at the linear level. This is not leakage per se
(the label is partly determined by what kind of question is being answered),
but it means **any V/A metric evaluated on v3 must beat the ~0.65
question-only baseline before claiming it measures answer quality.**

## 2. Top ±40 answer-word features — leakage scrutiny

Full lists: `top_features_answer_word.csv`, `top_features_question_word.csv`.

Top suspicious features and verdicts:

| Feature | Coef | Verdict | Reasoning |
|---|---|---|---|
| `added` | +2.29 | **leakage** | "Added:" post-publication edit marker; edits accrue after votes/acceptance |
| `my comment` | +1.59 | **leakage** | References the comment thread under the post — engagement artifact, post-hoc |
| `https` | -3.05 | ambiguous | Link-only answers are genuinely low-quality, but URL density is also era-/style-correlated |
| `my` / `my question` | -2.82 / -1.69 | ambiguous→leakage | Asker-voice in the answer slot = self-answers; if negatives oversample self-answers it's a sampling artifact |
| `found` / `figured` / `an answer` / `answered` | -2.17 / -1.73 / -1.96 / -1.51 | ambiguous→leakage | "I found/figured out the answer" self-answer discourse, same concern as above |
| `help` | -2.00 | ambiguous | "hope this helps" / "any help" meta-discourse, not math content |
| `nice` | +1.63 | ambiguous | "nice question/proof" politeness/engagement marker |
| `tag` | +1.83 | ambiguous | Meta reference to question tags, not answer substance |
| `maybe` / `think` / `not sure` | -2.83 / -2.69 / -1.93 | quality-signal | Genuine hedging — uncertain answers are genuinely worse |
| `oeis` / `the wikipedia` / `paper` / `page` | +1.94 / +1.75 / +2.21 / +1.89 | quality-signal | Citation of references — legitimate quality behavior |
| `yes` / `no` / `note that` / `hence` / `counterexample` | +2.27 / +2.38 / +2.18 / +1.77 / +2.14 | quality-signal | Direct-answer openings and mathematical discourse |
| `align` / `frac1` / `int_0` / `leqslant` | +1.65 / +1.81 / +1.68 / +1.75 | quality-signal | LaTeX-richness; structural but content-correlated |

Notably absent from the top ±40: `thanks`, `edit`, `update`, `upvote`,
`accepted` — the classic gross-leakage tokens are not dominant. The leakage
that exists is second-order (edit markers, comment references, self-answer
voice) with coefficients comparable to genuine quality terms.

## 3. Single-feature test AUCs (structural probes)

| Feature | Raw AUC | abs AUC | Higher predicts | Flag (>0.55) |
|---|---|---|---|---|
| answer_year | 0.4022 | 0.5978 | negative | **FLAG** |
| n_answers_on_question | 0.4689 | 0.5311 | negative | — |
| n_display_math | 0.5299 | 0.5299 | positive | — |
| answer_len_tokens | 0.4821 | 0.5179 | negative | — |
| answer_age_gap_days | 0.4826 | 0.5174 | negative | — |
| answer_position | 0.4926 | 0.5074 | negative | — |
| n_latex_blocks | 0.5073 | 0.5073 | positive | — |
| answer_len_chars | 0.4944 | 0.5056 | negative | — |
| has_numbered_list | 0.4999 | 0.5001 | negative | — |
| n_paragraphs | 0.5001 | 0.5001 | positive | — |

All-structural-features LR (10 features, combined): test AUC **0.6106** —
driven almost entirely by `answer_year`. Length, LaTeX density, paragraph
count, and position are all ≤0.53 individually: **the position-matching and
length controls in the v3 build worked.**

## 4. Class balance by answer_year (drift check)

| Year | n | pos_rate |
|---|---|---|
| 2010 | 423 | 0.9409 |
| 2011 | 2,154 | 0.8969 |
| 2012 | 4,288 | 0.8041 |
| 2013 | 7,676 | 0.6747 |
| 2014 | 9,779 | 0.5224 |
| 2015 | 10,509 | 0.4761 |
| 2016 | 10,130 | 0.4792 |
| 2017 | 11,347 | 0.4421 |
| 2018 | 9,581 | 0.4438 |
| 2019 | 8,510 | 0.4515 |
| 2020 | 9,135 | 0.4090 |
| 2021 | 6,561 | 0.4681 |
| 2022 | 4,730 | 0.4478 |
| 2023 | 4,112 | 0.4110 |
| 2024 | 1,065 | 0.3305 |

**This is the one real confound.** Pre-2014 rows (14,541 = 14.5% of the
dataset) have a 0.7535 pos rate; 2010-2012 are 80-94% positive. Era-specific
style (older LaTeX conventions, link formats, community discourse) therefore
carries label signal, and plausibly accounts for part of both the char-gram
AUC (0.6812) and the question-only AUC (0.6496).

## 5. Within answer_position == 1 (position residual)

- n (all splits) = 77,125, pos_rate = 0.4998; n (test) = 7,668, pos_rate = 0.5100
- Length (chars) AUC within position-1 test rows: **0.4922** — chance.

Per-position balance (`position_balance.csv`): positions 1-2 (94.7% of data)
are at 0.50; positions ≥5 drift negative but cover <0.6% of rows.

## 6. Question-only model

Test AUC 0.6496 (see §1). Top question features (e.g. `wikipedia`,
`counterexample`, `puzzle`, `motivation`, `martingale` positive; `help`,
`struggling`, `determine` negative) show this is **topic/register
predictability**: conceptual "is it true that…" questions attract
better-judged answers than computational "help me solve" questions. Expected
given the label construction; must be treated as the baseline floor.

## Verdict

**v3 is clean enough for V/A metric work, with three mandatory caveats:**

1. **Gross confounds are fixed.** Length, position, LaTeX density, paragraph
   structure are all at chance or ≤0.53 individually; within-position-1
   subset is balanced with length at chance. No classic leakage tokens
   (thanks/edit/update/accepted) dominate the linear model.
2. **Year drift must be handled.** answer_year alone scores 0.598 and
   pre-2014 rows are 75% positive. Recommended: either drop pre-2014 rows
   (loses 14.5%) or rebalance labels within year before any headline claim;
   at minimum report metric AUC with and without pre-2014 rows.
3. **Report the question-only floor.** Any metric/judge AUC on v3 should be
   compared against the 0.6496 question-only TF-IDF baseline; only the margin
   above it is evidence of answer-quality measurement. Linear answer-text
   models clear it by only +0.01-0.03, so the articulable answer-specific
   signal at the bag-of-words level is small — this is the gap V/A metrics
   need to explain, not a reason to discard the dataset.

Minor: consider stripping "Added:"-style edit suffixes and comment references
from answer text in the modeling pipeline (second-order post-hoc leakage).

## Artifacts

- `results.json` — all AUCs
- `top_features_answer_word.csv`, `top_features_question_word.csv` — ±40 LR coefficients
- `single_feature_aucs.csv`, `year_balance.csv`, `position_balance.csv`
