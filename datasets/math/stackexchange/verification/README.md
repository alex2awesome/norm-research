# Math.SE Tier-1 Claim Verification (production pipeline)

Production version of the 30-answer pilot in `../verification_pilot/`
(REPORT.md there has the full motivation and failure-mode analysis).
Goal: for 10K-100K Math.SE answers, extract mechanically checkable claims
with a local LLM and verify them with sympy, producing per-answer
verifiability (V) features.

## Pipeline

```
math_se_v3_position_matched.csv.gz  (100K rows: text, split, answer_id, ...)
        |
        v
[1] EXTRACT   run_extraction_sk3.py        GPU, sk3, offline vLLM batch
        |       Qwen3.5-122B-A10B-FP8, prompt = extraction_prompt.SYSTEM_PROMPT
        |       retry-with-new-seed on invalid output (never repetition_penalty)
        v
[2] VALIDATE  extraction_prompt.validate_extraction   (inline in step 1)
        |       strict JSON + one repair pass; schema check; round-trip
        |       fidelity check (source_quote must occur in the answer)
        v
    claims_{split}.jsonl     one claim per line, row ids carried
        |
        v
[3] VERIFY    run_verification.py -> harness.py        CPU, anywhere
        |       per-claim subprocess with hard timeout (10s default)
        |       tiers: symbolic simplify -> numeric sampling (20 pts,
        |       assumption-respecting, complex-safe) -> quadrature
        v
    results_{split}.jsonl    verdict + method + diagnostics per claim
        |
        v
[4] AGGREGATE (inside run_verification.py)
        v
    features_{split}.csv     per-answer V features
```

Steps 1-2 run on sk3 (launch separately; this repo only provides the
script). Steps 3-4 are pure CPU.

## Claim schema

Documented in full in `extraction_prompt.SCHEMA_DOC`. Claim types:

| type | checked as |
|---|---|
| EQUALITY / NUMERIC_VALUE / DERIVATIVE / LIMIT / SUM / INTEGRAL | lhs == rhs (or !=, is_finite, is_infinite) |
| INEQUALITY | lhs <op> rhs at sampled admissible points |
| BOOLEAN_EQUIV | SAT check of Not(Equivalent(lhs, rhs)) |
| MODULAR | lhs == rhs (mod modulus), exact integer arithmetic |
| DIVISIBILITY | lhs (divisor) divides rhs (dividend) |
| COMBINATORIAL_COUNT | expression equals a concrete integer, exact |
| MATRIX | elementwise identity of small concrete matrices |
| RECURRENCE_CHECK | closed form satisfies a recurrence at sampled n |
| NONE | answer has nothing a CAS can check (reason recorded) |

Every claim carries fidelity fields that the harness passes through
untouched: `source_quote` (verbatim substring of the answer),
`fidelity_note` (what was changed when formalizing), `fidelity`
(round-trip quote-check result), `load_bearing` (is this the answer's
main result or a side computation).

## Verdict semantics

| verdict | meaning |
|---|---|
| VERIFIED_SYMBOLIC | proof-grade: simplify(lhs-rhs)==0, SAT-unsat, or exact integer arithmetic |
| VERIFIED_NUMERIC | **evidence, not proof**: 20-point sampling or quadrature agreement |
| REFUTED | symbolic nonzero constant residual, exact integer counterexample, or solid numeric mismatches (20/20-style margins) |
| INCONCLUSIVE | could not decide within budget (timeouts, sampling errors) |
| PARSE_FAIL | malformed claim: schema violation, banned tokens, sympy parse error |
| NO_CLAIM | claim_type NONE (prose answer) |
| EXTRACTION_FAILED | LLM produced no valid claims after all retries (carried from step 1) |

Numeric REFUTED requires the mismatch margin (rel diff > 1e-5) with no
matching points; mixed match/mismatch is INCONCLUSIVE. For MODULAR /
DIVISIBILITY / COMBINATORIAL_COUNT the arithmetic is exact, so a single
sampled counterexample is a genuine refutation.

## Known failure modes (from the pilot, n=30)

1. **Coverage is topic-dependent.** ~75% of answers contain some checkable
   claim but only ~40-50% have their *load-bearing* content checkable;
   abstract algebra / topology / category theory get zero coverage. Any
   scaled metric must stratify by topic (`primary_tag`).
2. **Verified-claims != verified-answer.** Side computations verify while
   the actual argument goes unchecked; `load_bearing` exists to separate
   these, and `has_refuted_load_bearing` is the high-precision error signal.
3. **Silent mis-formalization is the dominant extraction risk** (typo
   repair, notation disambiguation, one-sided limit direction, encoding a
   cdf as F:function). Mitigations: mandatory `source_quote` round-trip
   check, mandatory `fidelity_note`, retry on validation failure. Residual
   risk remains — REFUTED claims at scale should be spot-audited before use.
4. **Quantified claims are spot-checked via instantiation** (the prompt
   requires instantiations be declared in `fidelity_note`); a "VERIFIED"
   instance is not a proof of the universal claim.
5. **Asymptotics (o(1), ~) are out of scope** for Tier-1.
6. **The judgement label measures community reception, not correctness**
   (pilot: 9/10 checkable judgement=0 answers verified cleanly). V features
   are a precision tool for catching algebra errors, not a label proxy.

## Aggregation plan (features_{split}.csv)

Per answer: `n_claims`, `n_verified` (+ symbolic/numeric split),
`n_refuted`, `n_inconclusive`, `n_parse_fail`,
`frac_checkable` = (n_verified + n_refuted) / n_claims,
`has_refuted_load_bearing` (any REFUTED claim marked load-bearing),
`no_claims`, `extraction_failed`.

## Regression test

`pytest test_harness.py` re-runs the pilot's 67 hand-extracted claims and
requires the pilot tally: 58 VERIFIED (47 symbolic + 11 numeric under the
default 10s budget), 2 REFUTED (claim_ids 46/47 — the genuine order-statistics
algebra error in row 20), 7 NO_CLAIM, 0 INCONCLUSIVE, 0 PARSE_FAIL.
Plus unit tests for the five new claim types, malformed-input handling,
the hard timeout, and fidelity passthrough. 13 tests, ~20s.

## Running at scale

```bash
# on sk3 (GPU), one free GPU, offline batch:
CUDA_VISIBLE_DEVICES=<free> nohup python3 run_extraction_sk3.py --split eval \
    > extraction_eval.log 2>&1 &

# anywhere (CPU):
python3 run_verification.py --claims claims_eval.jsonl --workers 8
```

Both steps checkpoint after every chunk/claim and resume from existing
output, so they can be killed and relaunched freely.
