# Math.SE Tier-1 Verification Pilot (n=30)

**Date:** 2026-06-10
**Sample:** 30 rows from `datasets/math/stackexchange/math_se_modeling.csv.gz` (pandas seed 42; 15 with judgement=1, 15 with judgement=0). Saved in `sample30.jsonl`.
**Pipeline:** manual claim extraction by Claude (acting as the extractor, writing sympy-ready expressions into `claims.jsonl`) -> `verify_claims.py` (sympy 1.14: symbolic simplify first, then 20-point random numeric evaluation respecting assumptions, 10s timeout per symbolic op) -> `results.json`.

## Headline numbers

| Metric | Value |
|---|---|
| Answers with >=1 checkable claim | **23/30 (76.7%)** |
| ... among judgement=1 | 13/15 (86.7%) |
| ... among judgement=0 | 10/15 (66.7%) |
| Checkable claims extracted | 60 (mean 2.0/answer overall; 2.6 over covered answers; max 5) |
| Claim verdicts | **58 VERIFIED, 2 REFUTED, 0 INCONCLUSIVE, 0 PARSE_FAIL** |
| Verification methods (VERIFIED) | 46 symbolic, 8 numeric-sample, 3 numeric-quadrature, 1 boolean-SAT |

Claim-type mix: 27 EQUALITY, 9 NUMERIC_VALUE, 7 LIMIT, 5 DERIVATIVE, 6 INTEGRAL, 4 SUM, 1 INEQUALITY, 1 BOOLEAN_EQUIV (+7 NONE rows).

**Important caveat on PARSE_FAIL=0:** I wrote the sympy expressions myself during extraction, so all parse burden was absorbed into the extraction step (typo repair, notation disambiguation, assumption annotation). A mechanical LaTeX->sympy parser would have failed on many of these; see failure modes.

## The 2 refuted claims (both row 20, judgement=0)

The answer (order-statistics rewrite of an integral) states
`F(1-F)^2 = (F_{2,3} - F_{1,3})/3` and `F^2(1-F) = (F_{3,3} - F_{2,3})/3`.
With its own definitions (F_{1,3}=F^3 etc.) these are **swapped**: symbolic residual `F(2F^2-3F+1) != 0`, 20/20 numeric mismatches. The two errors partially cancel downstream but flip a sign in the derivation. Fittingly, the answer ends with "I would be happy if someone could verify this" and has judgement=0. This is a genuine catch: a real algebra error in real Math.SE content found by the harness.

## Verification outcome vs. judgement label (n=30, counts only)

| Row outcome | judgement=1 | judgement=0 |
|---|---|---|
| ALL_VERIFIED (>=1 claim, 0 refuted) | 13 | 9 |
| HAS_REFUTED | 0 | 1 |
| NO_CLAIMS (pure prose) | 2 | 5 |

Signal exists but is weak and asymmetric: refutation only ever hit a judgement=0 row, and prose-only answers skew judgement=0 (5 vs 2). But 9 of 10 checkable judgement=0 answers verified cleanly -- e.g. row 15 is a flawlessly correct antiderivative check that the community still scored 0. **The judgement label measures community reception/acceptance, not mathematical correctness; Tier-1 verification is a precision tool for catching algebra errors, not a proxy for the label.**

## Failure modes (of coverage, in order of importance)

1. **Pure-prose/abstract answers (7/30):** topology (compactness arguments), differential geometry (holonomy), functional analysis (completeness counterexample), topos theory, infinite-dimensional operator counterexamples, history-of-notation. Nothing for a CAS to bite on.
2. **Checkable claims are often peripheral.** For proof-style answers, what I could check is the *identity being proven* or side computations -- not the argument that constitutes the answer's value (e.g. row 1: the combinatorial proof's correctness is untouched by verifying the identity; row 17: the answer's content is about why delta=min{1,eps/4}, not the limit value). Verified-claims != verified-answer.
3. **Quantified/forall-claims need instantiation.** "Converges iff p>1/2" (row 26), "diverges for all C!=3" (row 21), limits split by parameter ranges -- I checked hand-picked instances (p=1/2,1,2; C=1,3). A pipeline must do this instantiation explicitly and honestly report it as spot-checking.
4. **Formalization judgment calls** that an automated extractor would get wrong silently: repairing a LaTeX typo ("sin(theta + cos theta)" -> sin theta + cos theta, row 0); disambiguating vector notation P0P1 into coordinates (row 16); encoding a cdf F and density f=F' as an undefined sympy Function (row 20); choosing one-sided limit directions (row 14).
5. **Asymptotics are out of scope:** row 12's content is Laplace-method expansions with o(1)/~ -- only its two derivative steps were checkable. INEQUALITY and several identity verifications are sampling-based (evidence, not proof; flagged in `method`).

## Honest assessment

**(a) Fraction of Math.SE coverable by Tier-1 (sympy/numeric) at scale.**
~75% of answers contain *some* checkable claim, but I estimate only **~40-50%** of answers have their *load-bearing* content checkable this way (computational calculus, identities, determinate "the answer is X" questions). Another ~25% get partial signal (proof answers with checkable side-computations -- useful for catching slips, as row 20 shows). The remaining ~25-30% (abstract algebra, topology, category theory, analysis arguments, soft questions) get zero coverage. Coverage is strongly topic-dependent, so any scaled metric will be confounded by topic unless stratified.

**(b) What an LLM-extraction + sympy-check pipeline needs.**
- Extraction is the hard 90%: LaTeX->sympy translation with assumption tracking (domains, integer-ness, branch cuts), bound-variable handling, typo repair, and instantiation of quantified claims. Expect meaningful PARSE_FAIL and -- worse -- silent mis-formalization rates; needs a round-trip check (render formalized claim back to LaTeX, LLM-verify it matches the quoted source) before trusting verdicts.
- Claim-importance weighting: distinguish "headline result" from "side computation"; a REFUTED side identity with a partially-cancelling error (row 20) must not be naively scored as a wrong final answer.
- Calibration of verdict semantics: numeric-sample VERIFIED is evidence not proof; refutation should require a symbolic residual or multiple solid numeric witnesses (this harness requires both kinds of margin and got clean 20/20 separations).
- Cost at scale is fine: one LLM extraction call per answer + cheap sympy; the harness ran 60 claims in under a minute. The binding constraint is extraction fidelity, not compute.

**(c) Would Lean autoformalization add anything today?**
Mostly no, for this corpus. The rows where Lean could in principle help are exactly the prose-proof rows (2, 18, 22, 24) -- and those are the *hardest* to autoformalize: statement autoformalization success rates are modest even on clean competition statements, and full proof autoformalization of free-form Math.SE prose (point-set topology with ad-hoc definitions, pre-Hilbert space counterexamples, topos theory with thin mathlib coverage) is well beyond current tooling. For the rows Lean *could* handle (algebraic identities, determinant expansions), sympy already gives the same verdict at a tiny fraction of the cost. Lean's genuine marginal value would be (i) turning sampling-based identity/inequality "VERIFIED" into proofs (e.g. polyrith/nlinarith on polynomial claims), and (ii) eventually checking proof *structure*, which is the actual judgement-relevant content of most answers. Today that is a research project, not a pipeline component. The more practical Tier-2 is: stronger CAS strategies + interval arithmetic for inequalities + LLM-judged proof-step entailment.

## Artifacts

- `sample30.jsonl` -- the 30 sampled rows (row_id, orig_index, text, judgement)
- `claims.jsonl` -- 67 records: 60 structured claims + 7 NONE records with reasons
- `verify_claims.py` -- harness (symbolic -> numeric fallback, timeouts, complex-safe sampling)
- `results.json` -- per-claim verdict, method, diagnostics (witness points, residuals, elapsed)
