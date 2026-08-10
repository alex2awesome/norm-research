# LLM approach-judge results: elegance is the first live A-signal on AoPS

2026-06-12. Judge: Qwen3.5-122B-A10B-FP8, sk3 GPU 3, 6.57h.
Script `scripts/judge_approach_sk3.py`; verdicts `approach_verdicts.jsonl`
(31,250 rows: 27,122 usable verdicts, 3,608 terminal failures (13%, after 4
retry passes), 520 no-wiki-solutions). Per forum solution: problem + up to 6
wiki solutions → {matched_solution, same_approach, approach_label,
novel_approach, elegance 1–5, reason}. Analysis:
`scripts/analyze_approach_verdicts.py`; joined table
`approach_verdicts_joined.parquet` (dedup to 26,757).

## Distributions

- elegance: 1: 14%, 2: 12%, 3: 24%, 4: 44%, 5: 8%
- same_approach (forum solution uses a wiki approach): **70%**
- novel_approach: 23.5%
- `approach_label` top buckets: 'not a solution' 1,671 (6%) — meta/comment
  posts; then coordinate geometry, complementary counting, p-adic valuation,
  power-of-a-point, barycentric coordinates… (real approach taxonomy emerges)

## Main tests vs de-confounded thanks (`thanks_resid`), within-problem

| field | within-prob mean ρ | wilcoxon p | AUC | verdict |
|---|---|---|---|---|
| **elegance** | **+0.066** | **5e-11** | 0.544 (4–5 vs 1–2) | **LIVE** |
| novel_approach | −0.016 | 0.15 | 0.507 | null |
| same_approach (≈ matched-to-wiki) | +0.011 | 0.28 | 0.500 | null — replicates lexical-similarity null exactly |

Verified-correct subset (n=6,956): elegance ρ +0.076 (p=6.6e-05); novel and
same_approach still null. So it is not correctness leakage.

## Robustness of the elegance signal

| cut | within-prob ρ | p | AUC(4–5 vs 1–2) |
|---|---|---|---|
| dedup all (n=26,757) | +0.066 | 5e-11 | 0.544 |
| excl 'not a solution' | +0.073 | 9e-13 | 0.547 |
| elegance≥2 only | +0.080 | 4.2e-14 | 0.549 |
| excl salvaged JSON | +0.066 | 4e-10 | 0.545 |
| verified-correct ∧ excl not-a-sol | +0.078 | 4.5e-05 | 0.545 |

Length confound: elegance ~ log_len ρ only +0.10 (judge is NOT just length);
partial ρ (elegance ~ thanks_resid | length) = +0.063 (p=1.2e-26) — elegance
carries signal **independent of** the length signal.

## Interpretation (vs the established nulls)

1. **Approach choice doesn't earn thanks**: matching a wiki/editorial approach
   is dead at AUC 0.500 — the LLM-judged version of similarity agrees
   perfectly with the lexical version. Novelty of approach is also null.
2. **Judged elegance is the first above-floor A-signal** on this leg: small
   (AUC 0.545–0.549, vs length 0.594) but extremely robust and
   length-independent. The community rewards a quality the judge can see in
   the writing, not which approach was taken.
3. V/A/T reading: V≈0.50–0.58 (correctness saturates), A gains a real but thin
   increment (elegance, +~0.05 AUC over chance, additive with length), and the
   bulk of within-problem thanks variance remains unexplained → taste/noise.

Caveats: thanks_resid noise floor ±0.02–0.06 means ρ≈0.07 is near the
measurable edge though the p-values are unambiguous; 13% terminal judge
failures are not random (longer/messier posts); elegance scale is judge-anchored
(44% of mass on 4).

## Code-parallel framing: y = same-approach-as-editorial (user-requested)

2026-06-12, `scripts/v_ladder_same_approach.py`. Flip the design to match
LC/CC/CF: y=1 = LLM judged the forum solution substantially the same approach
as a wiki solution (base rate 0.696, n=26,757), group-CV LR by problem.

**EX-ANTE RULING (user, 2026-06-12 — same as the code-editorial debate):
similarity-to-the-wiki is computed against the label's own reference object,
so it is NOT an admissible predictor.** (For the record it scores 0.757 alone
/ 0.771 with V — useful only as label-validity evidence that "same approach"
is heavily lexically determined.)

Admissible ex-ante ladder (n=26,405, base 0.699, group-CV by problem):

| layer (ex-ante only) | AUC |
|---|---|
| V-struct (length, display math, latex density) | 0.544 |
| V-answer (has_answer, is_correct) | 0.602 |
| V-struct + V-answer | 0.636 |
| A-cheap register p_ed (post text only) | 0.653 |
| V + register | 0.678 |
| **post-only word TF-IDF (1–2gram, fold-fit)** | **0.727** |
| TF-IDF + V + register | **0.735** |

Reading: deterministic ex-ante V tops out ~0.64 (correctness the best single
V feature — right answers skew editorial-approach). The post's own vocabulary
is the strongest admissible signal (0.727): mainstream-technique vocabulary
predicts matching the editorial without ever seeing it — the approach
fingerprint in clean ex-ante form. Residual 0.735→1 is the contextual part
(does THIS problem's editorial take the mainstream route) plus judge noise.
One dataset, two labels: agreement-with-editorial is substantially
predictable ex-ante (0.74); community preference is not (0.50–0.59 + thin
elegance). Caveat: y is a single-judge label; 13% messiest posts unjudged.

## Transcription cleaning + TF-IDF-inspired V features (2026-06-12 PM)

User flagged `boxed`/`hide solution` as possible editorial transcriptions.
Quantified: **955 posts (3.6%) sit in a transcription band (sim_word≥0.5 to
wiki; 94–96% y=1)** — dropped from all analyses below. But boxed/hide are NOT
transcription artifacts: only 317/8,647 hide-posts are high-sim, and among
clean posts y|boxed=0.847 vs 0.618 — it's the **complete-solution register**
(forum convention: full solutions get [hide] + \boxed answer). Post-cleaning
TF-IDF top features unchanged (boxed, hide solution, we, implies, thus...).

`scripts/v_features_from_tfidf.py`: 11 deterministic ex-ante lexicon/regex
features (boxed, hide_block, answer_stmt, deductive, meta_doubt,
first_person, heavy_machinery, standard_tech, proof_framing,
numeral_density, question_marks). Spot-checked. Best singles: numeral
density 0.650, boxed 0.617, first_person 0.597(−), hide 0.574.

Cleaned-pool ladder (n=25,454, base 0.687):

| layer | AUC |
|---|---|
| V-old (struct+answer) | 0.634 |
| **V-new (11 lexicons)** | **0.706** |
| V-old+new | 0.714 |
| +register | 0.716 |
| post-only TF-IDF (reference) | 0.726 |
| everything | 0.737 |

**11 interpretable deterministic features recover ~90% of the TF-IDF signal**
(0.714/0.726); cleaning didn't move TF-IDF (0.727→0.726). Same-approach ≈
complete-solution register + concrete-numeric computation − meta-doubt −
heavy machinery (inversion/Pascal/genfuncs/determinants/code). Same shape as
code: approach-match is articulable convention; preference is not.
Feature table: `v_features_same_approach.parquet`.

## TWO-REGIME caveat (2026-06-12 PM) — how to state the result

Restricting to verified-correct genuine solutions (n=6,140; excl
not-a-solution): **base rate jumps 0.687→0.882 and predictability collapses
— TF-IDF 0.726→0.594, V-new 0.706→0.560** (V+register 0.571).

| pool | base | V-new | TF-IDF |
|---|---|---|---|
| all judged (cleaned) | 0.687 | 0.706 | 0.726 |
| verified-correct only | 0.882 | 0.560 | 0.594 |

So the headline 0.71–0.74 is substantially the *solution-completeness
gradient* (complete worked solutions vs fragments/meta/wrong attempts), which
is highly articulable. True approach choice among correct solvers: deviation
from editorial is rare (12%) and only weakly predictable ex-ante (0.59) —
and unrewarded (thanks nulls). State the result as two regimes, never as a
flat 0.74. Open items: (a) ~50-row human audit of judge same/different calls
(only unvalidated label in the chain; 70% base rate could include judge
yes-bias); (b) check whether code (CC/LC) similarity numbers show the same
within-correct collapse.

## Judge audit (2026-06-12 PM): 50-case blind re-judgment

3 independent Claude auditors re-judged 50 stratified cases (25/25 by judge
verdict; cleaned pool; blind to judge answers; problem + wiki solutions +
forum post). Files: notes/audit_blind.jsonl, audit_judge_answers.jsonl,
audit_claude_*.json.

- Agreement on same/different: **0.72 (κ≈0.44)**; **0.85 on
  auditor-high-confidence cases**; matched wiki-solution index agrees 0.86
  when both say same.
- **Judge is conservative, not yes-biased**: 10/14 disagreements are
  judge=different, auditor=same (judge under-credits stylistically distant
  matches). Implied corpus base rate ≈0.71 vs observed 0.70 — no inflation.
- Auditors flagged 10/50 as non-solutions (consistent with judge's 6%
  not-a-solution + cleaning).
- Implication: label noise attenuates both pools' AUCs ≈ equally → the
  two-regime contrast stands; absolute AUCs are lower bounds.

## Code-leg comparison (2026-06-12 PM): convergence confirmed, regimes aligned

Code similarity-to-editorial judgments exist (sk3
`outputs/v2_analysis/comp_unified_editorial_labels.parquet`, Qwen, 3,795
pairs cc/luogu/usaco; `lc_editorial_judging_claude_1500.parquet`, Claude,
1,480 LC pairs). Key structural fact: **the judged code pools are ~94%
accepted-only** (luogu/usaco scraped AC-only; LC discuss posts have no
verdicts) — code was always measured INSIDE regime 2. Like-for-like:

| within-correct, post-only TF-IDF | AUC |
|---|---|
| AoPS verified-correct | 0.594 |
| cc+luogu AC-only | 0.608 |
| luogu alone (100% AC) | 0.507 |
| LeetCode real-code | 0.53–0.56 |

**Approach identity among correct solutions ≈ 0.5–0.6 everywhere — math and
code converge on the same quantity.** Out-of-regime slice agrees: 211
unverified cc rows have same-approach base 0.483 vs 0.730 for AC (regime-1
direction). ARTIFACT WARNING: pooled 3-platform code AUC 0.722 is platform
composition — usaco "candidates" are the editorial's own code (base 0.985);
drop/separate usaco in any code-leg reporting. To build a true full-pool
regime for code: judge the Codeforces WA/RE candidates (86K/21K, ~155
problems with editorials in candidates.parquet) — only platform with
rejected submissions.
