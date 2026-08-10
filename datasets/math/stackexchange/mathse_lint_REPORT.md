# Math.SE deterministic linter — feature report (v1, 2026-06-11)

Module: `mathse_lint.py` (+ `resources/theorem_names.txt`,
`resources/math_jargon_allowlist.txt`). Run over the full canonical
`math_se_v3_3_propensity_balanced.csv.gz` (99,722 rows, y 50/50) on sk3 with
24 CPU workers. **No LLM anywhere** — regex + sympy (lark LaTeX backend) +
pylatexenc + pyspellchecker + a cached Wikipedia theorem-name list. Purpose:
deterministic verification coverage *independent* of the LLM claim-extraction
pipeline (`verification/`, v15–v17).

Features CSV (sk3): `…/datasets/math/stackexchange/mathse_lint_features.csv`.

## What shipped (9 metric families)

| family | features | how |
|---|---|---|
| step_chain | n_steps_total/checked/verified/refuted/unparseable, frac_steps_verified | display+inline math split into `=`/`≤`/`≥` chains at top bracket level; each adjacent pair parsed with sympy's **lark** LaTeX backend and checked symbolically (`simplify(a−b)=0`) then numerically (8 samples on (0.15,2.5)); fork-isolated, 2s/step, ≤30s/row |
| literal_arith | n_arith, n_arith_wrong | pure-numeric `=`-pairs (e.g. `12·13=156`, `\frac82=4`, prose `3+4=7`) verified exactly (rationals) or to 1% (decimals) |
| latex_parse | n_latex_errors, frac_blocks_with_errors, n_math_blocks | brace balance, `\left/\right` balance, `\begin/\end` matching, strict pylatexenc parse, odd-`$` check |
| symbol_hygiene | n_undefined_symbols, n_unused_definitions | bind sites = let/define/denote/noun-phrases ("the line $l$") + question math + LHS-of-`=` + big-operator subscripts (`\sum_{n=…}`, `\lim_{h\to…}`); skip e,i,d,O,C,π,ε,δ and i…n as sub/superscripts; flag unbound symbols used ≥2×, and let/define-bound symbols never used again |
| dangling_refs | n_refs, n_dangling | `\eqref/\ref` vs `\label`; "equation (N)/by (N)" vs `\tag{N}`/line anchors (question anchors count as referents); "the lemma/claim above" vs an earlier lemma/claim |
| typo_density | typos_per_100_words, n_prose_words | pyspellchecker on math/code/URL-stripped prose, lowercase words ≥4 chars only (proper nouns excluded), minus a 273-word corpus-derived jargon allowlist (doc-freq ≥2 OOV terms, eyeballed; true misspellings like *wich/lenght/beacuse/continous* kept OUT of the allowlist) |
| theorem_names | n_theorem_mentions, n_misspelled_theorem_mentions | 1,303 names from Wikipedia "List of theorems"+"List of lemmas" (cached in repo); pre-keyword suffix match + full-name substring match; misspelling = capitalized candidate at Levenshtein 1–2 from a known name but not exact |
| near_dup | max_jaccard_to_sibling | 5-gram char-shingle jaccard between answers of the same question_id (0 if no sibling) |
| form_contract | question_type, contract_met, has_boxed_or_final_numeric, has_proof_markers, has_example_marker | question speech-act regex (prove > compute > reference > why); answer-form regexes (\boxed/"the answer is" final numeric; QED/∎/"as desired"; "for example/intuitively"; URL/book for reference) |

## Validation (mandatory 30-row eyeball + targeted samples, before the run)

Two validation rounds on a 30-row random sample with full artifact dumps,
plus targeted samples for the sparse metrics (rows containing `\tag`/`\eqref`,
"theorem/lemma"). Examples of what the final version produces:

**step_chain** (verification is real, refutation is conservative):
- `[VERIFIED] x^2+bx = x^2+bx+\frac{b^2}{4}-\frac{b^2}{4} = (x+\frac b2)^2-(\frac b2)^2` (completing the square, symbolic)
- `[VERIFIED] 2 e^{-2e^y}\cdot e^y = 2e^{-2e^y+y}` (symbolic)
- `[VERIFIED] \int_0^1\sqrt{1-x^2}\,dx = \frac{\frac12!\cdot\frac12!}{(\frac12+\frac12)!}` (lark parses integrals/factorials)
- `[INCONCLUSIVE] x = -3`, `M = 7`, `Y = \log X` — contextual assignments are *not* refuted (refutation requires identical free-symbol sets + consistent numeric mismatch, or exact nonzero constant difference)
- synthetic checks: `x^2+1 = (x-1)(x+1)` → REFUTED; `\sin^2x+\cos^2x = 1`, `\sum_{i=1}^n i = \frac{n(n+1)}2` → VERIFIED

**literal_arith**:
- `[EQUAL] 0\cdot\frac{1}{2}+2\cdot\frac{1}{2} = 1`, `[EQUAL] -3 -(-7) = 4`
- synthetic: `\frac{8}{2} = 5` → WRONG, `2^{10} = 1024` → EQUAL

**symbol_hygiene** (after tightening): `Var(X)` no longer tokenized as V,a,r;
`\lim_{h\to0}` binds `h`; "lines $l,l'$" binds via noun phrase. Surviving
flags look genuinely sloppy, e.g. undefined `x,y` in an answer that introduces
an ellipse equation without binding them (judgement=0).

**dangling_refs**: "equation $(2)$" resolving to a `\tag{2}` in the *question*
is no longer dangling; flagged cases are real ("the claim" with no claim
stated anywhere).

**typo_density**: catches *ineuqlity, continous, beacuse, wich*; passes
*holomorphic, abelian, surjective, wlog*; proper nouns (Rudin, Munkres,
usernames) excluded by the lowercase-only rule.

**theorem_names**: mentions "Fubini's theorem", "monotone convergence
theorem", "Squeeze theorem" found; misspellings detected on synthetic
"Zorns lemma" (d=1), "Pythagoren theorem" (d=1).

## Heuristics weakened / dropped during validation

- **sympy `parse_latex` antlr backend unusable** in the norm-scraper env
  (antlr4-python3-runtime pinned at 4.9 by omegaconf; sympy 1.14 wants 4.11).
  Used the pure-python **lark backend** instead — works, but its grammar has
  no subscripted symbols and no implicit multiplication after exponents; we
  textually map subscripted tokens to fresh letters per chain and insert
  explicit `\cdot` after exponents. lark also returns ambiguity trees
  (`n(n+1)` = apply-or-multiply); resolved charitably — verify if *any*
  interpretation pair verifies, refute only if *all* refute.
- **Inequality-step refutation** initially fired on context-dependent
  inequalities with different symbols on each side (`3\delta(\delta+2) <
  \varepsilon`-type, where δ=min{1,ε/9} in context) — now requires identical
  free-symbol sets, like the equality path.
- **"reference" question type** initially keyed on bare "book"/"textbook"
  (fired on "my book says…"); tightened to recommendation/where-can-I-find
  phrasings.
- **`!=` is factorial-then-equals in LaTeX** (`\tfrac12! = \sqrt{\pi}/2`),
  not "not-equal" — an early guard treating it as ≠ was removed.
- **E[X]-style bracket notation, norms `\|·\|`, `\sqrt[n]{}`** stay
  unparseable (counted in n_steps_unparseable) — no silent guessing.

## Results (full run, 99,722 rows, 0 lint errors)

Splits: train 79,873 / eval 9,995 / test 9,854; y-balance 0.500.
Analysis artifacts on sk3: `…/stackexchange/lint_analysis/`
(`auc_results.txt`, `audit_results.txt`, scripts).

### Per-feature AUC (train) + coverage

| feature | AUC (train) | coverage (defined) | nonzero |
|---|---|---|---|
| n_steps_total | 0.5291 | 100.0% | 74.8% |
| n_steps_checked | 0.5182 | 100.0% | 50.2% |
| n_steps_verified | 0.5180 | 100.0% | 16.1% |
| n_steps_refuted | 0.5032 | 100.0% | 4.1% |
| n_steps_unparseable | 0.5228 | 100.0% | 60.6% |
| frac_steps_verified | 0.5266 | 50.2% | 32.1% |
| n_arith | 0.5039 | 100.0% | 4.6% |
| n_arith_wrong | 0.4994 | 100.0% | 0.4% |
| n_latex_errors | 0.4975 | 100.0% | 1.1% |
| frac_blocks_with_errors | 0.4982 | 100.0% | 0.9% |
| n_undefined_symbols | 0.4965 | 100.0% | 22.6% |
| n_unused_definitions | 0.4999 | 100.0% | 5.5% |
| n_refs | 0.5014 | 100.0% | 2.6% |
| n_dangling | 0.5002 | 100.0% | 1.2% |
| typos_per_100_words | 0.4886 | 100.0% | 18.7% |
| n_theorem_mentions | 0.5057 | 100.0% | 6.3% |
| n_misspelled_theorem_mentions | 0.4999 | 100.0% | 0.1% |
| max_jaccard_to_sibling | 0.4814 | 100.0% | 1.9% |
| contract_met | 0.5071 | 60.2% | 9.4% |
| has_boxed_or_final_numeric | 0.5044 | 100.0% | 5.2% |
| has_proof_markers | 0.5018 | 100.0% | 3.2% |
| has_example_marker | 0.5075 | 100.0% | 11.9% |

No single feature is strong alone; the best individual signals are
volume/parseability of math (n_steps_total 0.529), typo density (0.489 —
i.e. typos predict judgement=0) and near-dup (0.481 — copying a sibling
answer predicts judgement=0).

### Combined LR (all lint features; StandardScaler + LogisticRegression(max_iter=2000), fit on train)

| split | AUC | n |
|---|---|---|
| train | 0.5661 | 79,873 |
| eval | 0.5672 | 9,995 |
| test | 0.5693 | 9,854 |

Reference points: question-only floor **0.461**; LLM-extraction sympy
claim-V **0.541**. The pure-deterministic linter beats claim-V by ~2.5–2.8
AUC points with 100% row coverage (claim-V covers only the ~10% of rows
with checkable claims).

Top standardized coefficients: `max_jaccard_to_sibling` −3.86 (dominant;
near-dup answers lose), `frac_steps_verified` +0.08,
`typos_per_100_words` −0.06, `has_example_marker` +0.05,
`n_theorem_mentions` +0.05, `n_latex_errors` −0.04,
`n_undefined_symbols` −0.03. Ablation **without** near-dup:
train 0.5484 / eval 0.5462 / test 0.5518 — so near-dup contributes
~+0.02 and the rest of the linter ~+0.09 over the floor.

### Super-combined (lint + claim-V on the joined subset)

Join on `answer_id == row_id` against
`verification/v_features_eval_full.csv`: 9,851 rows
(train 7,859 / eval 1,005 / test 987; y-balance 0.500/0.513/0.489).

| feature set | train AUC | eval AUC | test AUC |
|---|---|---|---|
| lint-only | 0.5715 | 0.5347 | 0.5542 |
| claim-only | 0.5435 | 0.5459 | 0.5487 |
| lint+claim | 0.5827 | 0.5545 | 0.5785 |

**Answer to the headline question: yes — all-deterministic V beats 0.541.**
On the full data the lint-only LR is 0.567/0.569 (eval/test); on the joined
subset lint+claim reaches 0.579 test (eval 0.555 — n=1,005, noisy), beating
claim-only (0.549) on both held-out splits. The two signal families are
complementary: adding lint to claim gains ~+1–3 points held-out.

### Precision audit (15 flagged rows per violation metric, eyeballed)

Verdicts use "FP" = flag fires but is not a genuine author error.

**n_steps_refuted (4.1% of rows). FP impression ≈ 70–80% as an
*error* detector.** Genuine catches exist —
`-2x^2+16x = -2(x^2-8)` *("Do you know how to complete the square? …
$$-2x^2+16x=-2(x^2-8)=-2(x-4)^2+32$$", a real dropped-x error,
judgement=0)* — but most refutations are not errors:
(a) **restated hypotheses / equations-to-solve**, e.g. `4A = A^7` *("If
$4A=A^7$, then $\det(4A)=\det(A^7)$…")* and `P^2 = P` (definition of a
projection); (b) **context the linter can't see**: modular identities
without `mod` in the chain (`\overline{17}=\overline{-768}` mod 157),
custom binary operations (`a*(b*a)=\big((b*a)*b\big)*(b*a)` for a
user-defined `*`), finite-field algebra; (c) **lark parser misreads of
true identities**, e.g. `r\cos(\phi+\theta) = r\cos\phi\cos\theta -
r\sin\phi\sin\theta` (angle addition — implicit function application
swallows the product) and the geometric-series identity
`q+\sum_{k=1}^n\frac2{3^k} = q+\frac23\frac{1-\frac1{3^n}}{1-\frac13}`.
A few flags catch *deliberate sloppiness* stated as equality
(`\theta=\sin\theta=\tan\theta` small-angle; `\frac{\pi}{3}=60`
radians-vs-degrees).

**n_arith_wrong (0.43%). FP impression ≈ 70–80%.** Genuine:
`\frac{13-1}{52-1} = \frac{12}{52}` *(should be 12/51; "…you are
interested in the variable $X_{52}$…", judgement=0)*. FP modes: numbers
in **other bases / modular context** (`101+110=1011` binary;
`1+1=0` in $\mathbb{Z}_2$; `3^{11}=1` mod 23), **floor division**
(`7/10=0` in a long-division walkthrough), **intended approximation**
(`\sqrt{3137}=56`), and **source-corrupted superscripts**
(`234 = 2\cdot32\cdot13` where SE stripped `3^2` to `32`).

**n_latex_errors (1.1%). FP impression ≈ 50%.** Genuine author
sloppiness is caught (a set literally closed with `\}` against an open
`(`: `P=(\{0,\frac{1}{n},…,1\},\{0,…,1\}\}`; formulas split across two
`$…$` pairs mid-set: `$A=\{cId_{2\times 2}$, $c\in\mathbb{C}\}$`). But
three extractor artifacts inflate it: nested `$` inside `\text{…}`
(legal on MathJax: `P(k \text{ bits are sent and $4-k$ bits…})`), the
`<DMATH>` sentinel landing inside `\left\{…\right.` cases blocks, and
`\begin {array}` (space before brace — MathJax-tolerated, regex-missed).

**n_undefined_symbols (22.5%). FP impression ≈ 90% — the weakest
metric.** Fires on conventional implicit binding the binder regexes miss:
integration dummies (`\int_0^{F(x)}t^{k-1}(1-t)^{n-k}\,dt` flags `t`),
transform variables (`\mathcal{L}[J_0(x)](p)` flags `x,p`), prose
binders not in the lexicon ("give $\lambda$ a small imaginary part
$\lambda\mapsto\lambda+i\mu$, **for** $\mu>0$" flags `\mu`; "with $D$
**being** the region" flags `D`), and "whenever"-bound variables. Its
AUC (0.4965) is correspondingly noise. Tightening would need a much
richer binder grammar.

**n_dangling (1.2%). FP impression ≈ 70%.** The dominant FP: "the
claim/proposition" referring to the *question's statement-to-prove*
when the question never uses the literal word ("I'll try to prove this
by induction… **The claim**…"; "Let $P(n)$ be **the proposition** that
states…" — a definition, not a reference). Also: trailing `(*)` anchors
at the end of an align line (not `\tag`, so unrecognized:
"…$f'(z_0)$$ and **by (*)**" where `(*)` *is* defined two lines up).
Genuine danglers found: references to formulas in *other answers*
("mentioned in that answer… **Formula (1)**", "per the strong form
**formula (5)**" citing @Noah's answer) — real self-containedness
violations, arguably fine in context.

**n_misspelled_theorem_mentions (0.13%). FP impression ≈ 45% — the
highest-precision error flag.** Genuine: *Parserval* (~Parseval),
*Nielson Schreier* (~Nielsen–Schreier), *Borell Cantelli's* (~Borel–
Cantelli), *Mashke's* (~Maschke), *Radon Nikodyn* (~Radon–Nikodym),
*Monotone Convergent Theorem* (~monotone convergence). FP mode is
**legitimate name variants** at Levenshtein ≤2: "Pythagoras(') theorem"
(≁ misspelling of "Pythagorean", 6 of 15 samples) and "Rational Roots
theorem" (~rational root). An alias list for ~10 common variants would
push precision to ~90%; coverage is tiny regardless.

**typos_per_100_words (bonus, 8 samples at ≥5/100). ≈ half genuine.**
Real typos caught (*floting, righ, tbefore, untill, montonely*); noise
from code identifiers in prose (*matplotlib, numpy, linspace* — only
4-space-indented code is stripped), @usernames in lowercase
(*ancientmathematician*), Latin etymology (*numerus*), and valid rare
words (*equational, monotonely*).

### Bottom line

- Deterministic lint features beat the LLM claim-V pipeline as a
  *label* signal (test 0.569 vs 0.541) at 100% coverage and zero LLM
  cost, and combine with it for 0.579 test on the joint subset. All of
  this is far above the 0.461 question-only floor but far below the
  dense ceiling — consistent with "verifiable surface quality is a thin
  slice of what the community rewards."
- As *error detectors*, only `n_misspelled_theorem_mentions` (~55%
  precision, fixable to ~90%) and `n_arith_wrong`/`n_steps_refuted` on
  *plain real-number arithmetic* are usable; modular/contextual math is
  the systematic confounder for both sympy families.
- `max_jaccard_to_sibling` (near-dup) is the single most useful feature
  in the LR despite 1.9% nonzero — duplicated sibling answers are
  strongly judgement=0.
