# law-stackexchange (law-C: practitioner-crowd revealed preference)

Law Stack Exchange answer-quality task — the **law-C cell** of the taste
taxonomy (practitioner/enthusiast **crowd**, **revealed** via community votes).
Direct legal parallel to Math.SE and CodeReview.SE: the same SE answer-quality
signal applied to legal exposition.

This is the **SMALL contrast SE site**: ~29K questions / ~43K answers in the
April-2024 dump (vs Math.SE's millions). It exists to confirm the SE
answer-quality signal generalizes to a legal domain under the **identical**
deconfounding recipe — not to be a large modeling corpus.

## Label — PRACTITIONER-CROWD revealed preference

- **Positive (judgement=1):** answer was *accepted* by the asker **and** has
  Score >= 3 (community upvotes minus downvotes).
- **Negative (judgement=0):** answer was *not accepted* **and** has
  Score <= `neg_max_score`.

`neg_max_score = 0` is the strict Math.SE recipe. On this small, score-right-
shifted site the strict pool was too thin (2,038 matched rows < 4,000 floor),
so the build **auto-fell-back to `neg_max_score = 1`** (the CR.SE widening),
documented in the manifest (`chosen.fallback_triggered = true`). The ambiguous
middle is dropped; answers < 50 chars are dropped as stubs.

The signal proxies how the Law.SE community judges a legal answer — a
*practitioner/enthusiast crowd* preference, **revealed** through votes, not an
explicit rubric. It should not be over-interpreted as legal correctness:
confident, well-cited exposition reads authoritative whether or not it is right.

## Design — 1:1 port of Math.SE v3.3 / CR.SE v2

The whole point is methodological identity with Math.SE and CR.SE so the three
SE cells are comparable. Both stages are byte-for-byte ports of the canonical
scripts (`datasets/math/stackexchange/build_v3_position_matched.py` +
`propensity_balance_v3_3.py`; `datasets/code-review/crse_balanced_v2/`).

| Step | Math.SE v3.3 / CR.SE v2 | Law.SE (this dir) | Identical? |
|---|---|---|---|
| Label | accepted ∧ score≥3 vs not-accepted ∧ score≤`neg_max` | same | yes |
| Stub filter | answer < 50 chars dropped | same | yes |
| Question-disjoint | drop questions with both classes | same | yes |
| Position metadata | rank by CreationDate over ALL answers, before any filter | same | yes |
| **Position matching** | per-tag × (q_len_bin × a_len_bin × position{1,2,3+} × year3) downsample | same | yes |
| **Propensity balance** | 5-fold OOF TF-IDF+LR p(y\|question) → balance within (decile × year3) → 50/50 | same | yes |
| Floor gate | FAIL LOUD if OUTPUT question-only floor > 0.55 | same | yes |
| Split | md5(question_id)[:8]/0xFFFFFFFF → 80/10/10, all answers of a Q on ONE side | same (`splits.py`, hash matches) | yes |
| `neg_max` fallback | CR.SE: 0→1 if pool < 15K | 0→1 if pool < 4K (smaller site) | same mechanism |

**The dominant confound is answer POSITION** (earlier answers accrue more
votes). Pre-matching, the positive answer was posted *earlier* in **69.4%** of
within-question pos/neg pairs (Math.SE: 67.3%). Position matching drives the
position distribution to be **identical** between classes
(85.5% / 12.5% / 2.0% at positions 1 / 2 / 3+ for both); the propensity step
then balances within (question-propensity-decile × 3-year-bin), which also
absorbs question-popularity / question-age / topic and time drift. Answerer
reputation is not used as a feature (no leakage); it is controlled implicitly
via the position/propensity matching.

## Floors (5-fold grouped CV, balanced output)

| Floor | INPUT (pool) | OUTPUT (balanced) |
|---|---|---|
| question-only TF-IDF+LR | 0.581 | **0.450** (gate ≤0.55 passed) |
| position-only LR | 0.494 | 0.504 |
| answer-text TF-IDF+LR (signal) | 0.576 | 0.554 |
| OOF propensity AUC | — | 0.582 |

Output question-only floor **0.450** lands slightly below chance — mild
in-family overcorrection, the *same* documented outcome as Math.SE v3.3
(0.461). The position confound is at chance (0.504). The answer-text margin
above the question floor (~0.62 answer-only train→test vs 0.45 question floor)
is the legitimate answer-quality signal. **Report every downstream metric/judge
AUC against the 0.450 question-only floor** — only the margin above it is
evidence of answer-quality measurement.

## Top TF-IDF features (answer text, balanced output, AUC 0.624)

- **→ Positive:** `section`, `shall`, `court`, `right to`, `written`,
  `criminal`, case/statute citations (`v.`, `u.s.c`, `§`) — confident
  statute/citation register.
- **→ Negative:** `if the`, `can`, `what`, `how`, `you are`, `know`,
  `one thing`, `outside`, plus topic words (`australia`, `international`) —
  conditional/speculative register + small-site topic imbalance.

## Length confound — neutralized

Answer-length point-biserial corr with label = **0.016 (p=0.33, n.s.)**;
length-only LR floor = **0.504** (chance). Median answer length pos=962 /
neg=956 chars. Length is in the position-matching key, so the classic SE
"long answers win" shortcut is removed; **no length-controlled floor needed**
(corr ≪ 0.25).

## Manual spurious-feature analysis

Per-1000-chars rates (pos vs neg) on the balanced output:

| Feature | pos | neg | verdict |
|---|---|---|---|
| Citation: USC/section/act | 0.533 | 0.422 | **mixed** — real signal (grounding in authority is valued) *and* confound (cited answers read authoritative regardless of correctness) |
| Citation: case (`v.`) | 0.131 | 0.089 | same as above |
| IANAL disclaimer | 0.008 | 0.017 | **confound** — low-confidence register marker; 2× more common in neg (1.5% vs 0.7% of docs), modest |
| Hedging (maybe/think/guess) | 0.293 | 0.340 | **confound** — weak here (near-even), unlike Math.SE where hedging was a strong negative signal |
| imperative legal (shall/must) | 0.321 | 0.267 | **signal-leaning** — register of definitive legal statements |
| URL/link | 0.066 | 0.065 | **no effect** — equal across classes |
| rhetorical `?` | 0.295 | 0.388 | **confound** — speculative answers ask more questions back |

Real **confounds** (presentation register, not legal quality): IANAL
disclaimers, hedging, rhetorical questions. Real **quality signal** entangled
with register: citation/authority density and definitive legal phrasing —
genuinely valued by the community but also a confident-tone proxy. Length and
links are non-signals. None of these are *position/time* artifacts (those are
handled by the matching); they are answer-content register effects that a
downstream answer-quality model is expected to pick up — the point of the
0.450 floor is that they sit *above* the question-side floor.

## Known residuals

- **Topic imbalance** (`australia`, `international` as negative features) — the
  small site has uneven tag coverage; the propensity step balances within
  question-propensity deciles, which absorbs most of it, but a few topic words
  survive in the answer-text model. Acceptable at this size.
- **`neg_max_score = 1` fallback** widens the negative class to score-1 answers
  (vs strict 0). These are still *not accepted*; the score-1 negatives are
  low-engagement, not necessarily wrong. Same tradeoff CR.SE accepted.
- **Small N (3,606 rows).** This is a contrast cell, not a modeling corpus;
  test split is 405 rows. Treat AUCs as indicative.

## Files

```
datasets/law_se/
├── README.md                       (this file)
├── splits.py                       shared md5 group-split (matches Math.SE/CR.SE)
├── build_law_se_pool.py            Stage 1: label + position-match → pool
├── propensity_balance_law_se.py    Stage 2: OOF-propensity balance + floor gate
├── law_confound_check.py           manual spurious-feature lexicon audit
└── built/                          (data; sk3 + copied to laptop)
    ├── law_se_pool.csv.gz          position-matched pool (4,192 rows)
    ├── law_se_balanced.csv.gz      final balanced, with split column (3,606 rows)
    ├── train.csv.gz / eval.csv.gz / test.csv.gz   2,832 / 369 / 405 rows
    └── *.manifest.json             self-audit manifests for both stages
```

Raw dump (`law.stackexchange.com.7z`, ~99 MB) and `raw_dump/` extraction live
on sk3 only:
`/lfs/skampere3/0/alexspan/norm-research/datasets/law_se/`. The raw 7z is never
deleted.

## Output schema (`{train,eval,test}.csv.gz`)

| Column | Description |
|---|---|
| `text` | `"Question: <title>\n\n<question body>\n\nAnswer: <answer body>"`, HTML stripped, whitespace collapsed (identical stripper to Math.SE/CR.SE) |
| `judgement` | binary label (0/1) |
| `split` | train / eval / test (group-split by question_id) |
| `question_id`, `answer_id` | SE post IDs |
| `answer_position`, `n_answers_on_question`, `answer_age_gap_days`, `answer_year` | position/time metadata (computed over ALL answers before filtering) |
| `score`, `accepted`, `primary_tag`, `question_tags` | label provenance |
| `answer_text_norm` | presentation-normalized answer (NFKC, curly→straight quotes, dash/ellipsis, whitespace) — task-spec variant; `text` keeps the canonical raw-stripped form for floor comparability |

## Reproduce (on sk3, no GPU)

```bash
export HOME=/lfs/skampere3/0/alexspan
cd /lfs/skampere3/0/alexspan/norm-research/datasets/law_se
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
# (raw_dump/Posts.xml already extracted from law.stackexchange.com.7z; never delete the 7z)
$PY build_law_se_pool.py            # Stage 1 → built/law_se_pool.csv.gz
$PY propensity_balance_law_se.py    # Stage 2 → built/{law_se_balanced,train,eval,test}.csv.gz
```
