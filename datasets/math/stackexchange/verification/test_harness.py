#!/usr/bin/env python3
"""Tests for the production verification harness.

The regression test re-runs the 30-answer pilot's hand-extracted claims
(../verification_pilot/claims.jsonl) through the new harness and requires
the pilot tally to be reproduced exactly: 58 VERIFIED (any split between
VERIFIED_SYMBOLIC and VERIFIED_NUMERIC), 2 REFUTED (claim_ids 46 and 47),
0 INCONCLUSIVE, 0 PARSE_FAIL, 7 NO_CLAIM.

Run:  pytest test_harness.py -v
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

PILOT_CLAIMS = HERE.parent / "verification_pilot" / "claims.jsonl"


# ----------------------------------------------------------- pilot regression
def test_regression_pilot_claims():
    claims = [json.loads(l) for l in PILOT_CLAIMS.open()]
    assert len(claims) == 67
    tally = Counter()
    refuted_ids = []
    slow = []
    for c in claims:
        t0 = time.time()
        r = harness.verify_claim(c)  # default 10s budget, subprocess hard kill
        dt = time.time() - t0
        tally[r["verdict"]] += 1
        if r["verdict"] == "REFUTED":
            refuted_ids.append(c["claim_id"])
        if dt > harness.DEFAULT_TIMEOUT_S + harness.HARD_KILL_GRACE_S + 2:
            slow.append((c["claim_id"], dt))

    n_verified = tally["VERIFIED_SYMBOLIC"] + tally["VERIFIED_NUMERIC"]
    assert n_verified == 58, f"expected 58 VERIFIED, got {n_verified}: {dict(tally)}"
    assert tally["REFUTED"] == 2, f"expected 2 REFUTED: {dict(tally)}"
    assert sorted(refuted_ids) == [46, 47], f"wrong refuted claims: {refuted_ids}"
    assert tally["NO_CLAIM"] == 7
    assert tally["PARSE_FAIL"] == 0
    assert tally["INCONCLUSIVE"] == 0
    assert not slow, f"claims exceeded the hard timeout envelope: {slow}"


# ----------------------------------------------------------- malformed input
def test_malformed_claim_is_parse_fail_not_crash():
    bad = {"claim_id": 1, "row_id": 1, "claim_type": "EQUALITY",
           "sympy_lhs": "x ++ )( nonsense", "sympy_rhs": "0",
           "relation": "==", "free_symbols": ["x"], "assumptions": []}
    r = harness.verify_claim(bad)
    assert r["verdict"] == "PARSE_FAIL"


def test_schema_rejects_missing_fields_and_bad_types():
    assert harness.validate_claim_schema({"claim_type": "WIBBLE"})
    assert harness.validate_claim_schema({"claim_type": "EQUALITY"})  # no lhs
    assert harness.validate_claim_schema(
        {"claim_type": "EQUALITY", "sympy_lhs": "x", "sympy_rhs": "x",
         "relation": ">>"})
    assert harness.validate_claim_schema("not even a dict")
    r = harness.verify_claim({"claim_type": "EQUALITY"})
    assert r["verdict"] == "PARSE_FAIL"
    assert "schema_errors" in r["detail"]


def test_banned_tokens_rejected():
    bad = {"claim_type": "EQUALITY", "relation": "==",
           "sympy_lhs": "__import__('os').system('true')", "sympy_rhs": "0",
           "free_symbols": [], "assumptions": []}
    r = harness.verify_claim(bad)
    assert r["verdict"] == "PARSE_FAIL"


def test_hard_timeout_on_pathological_claim():
    # an integral mpmath grinds on; must come back INCONCLUSIVE within budget
    c = {"claim_id": 999, "row_id": 999, "claim_type": "INTEGRAL",
         "sympy_lhs": "Integral(Integral(exp(sin(x*y))*log(x + y + 2), "
                      "(x, 0, oo)), (y, 0, oo))",
         "sympy_rhs": "5", "relation": "==",
         "free_symbols": ["x", "y"], "assumptions": []}
    t0 = time.time()
    r = harness.verify_claim(c, timeout_s=4.0)
    assert time.time() - t0 < 4.0 + harness.HARD_KILL_GRACE_S + 2
    assert r["verdict"] == "INCONCLUSIVE"


# ----------------------------------------------------------- new claim types
def test_modular_concrete():
    ok = {"claim_type": "MODULAR", "sympy_lhs": "2**100", "sympy_rhs": "6",
          "modulus": "10", "relation": "==", "free_symbols": [],
          "assumptions": []}
    r = harness.verify_claim(ok)
    assert r["verdict"] == "VERIFIED_SYMBOLIC" and r["method"] == "exact_arithmetic"
    bad = dict(ok, sympy_rhs="7")
    assert harness.verify_claim(bad)["verdict"] == "REFUTED"


def test_modular_symbolic_sampling():
    c = {"claim_type": "MODULAR", "sympy_lhs": "n**5", "sympy_rhs": "n",
         "modulus": "5", "relation": "==",
         "free_symbols": ["n:integer,positive"], "assumptions": []}
    r = harness.verify_claim(c)
    assert r["verdict"] in ("VERIFIED_SYMBOLIC", "VERIFIED_NUMERIC")
    wrong = dict(c, modulus="7")  # n**7 == n mod 7 holds, n**5 == n mod 7 doesn't
    assert harness.verify_claim(wrong)["verdict"] == "REFUTED"


def test_divisibility():
    ok = {"claim_type": "DIVISIBILITY", "sympy_lhs": "6",
          "sympy_rhs": "n**3 - n", "relation": "divides",
          "free_symbols": ["n:integer,positive"], "assumptions": []}
    r = harness.verify_claim(ok)
    assert r["verdict"] in ("VERIFIED_SYMBOLIC", "VERIFIED_NUMERIC")
    bad = {"claim_type": "DIVISIBILITY", "sympy_lhs": "4",
           "sympy_rhs": "n**2", "relation": "divides",
           "free_symbols": ["n:integer,positive"], "assumptions": []}
    assert harness.verify_claim(bad)["verdict"] == "REFUTED"


def test_combinatorial_count():
    ok = {"claim_type": "COMBINATORIAL_COUNT",
          "sympy_lhs": "binomial(10, 3)", "sympy_rhs": "120",
          "relation": "==", "free_symbols": [], "assumptions": []}
    r = harness.verify_claim(ok)
    assert r["verdict"] == "VERIFIED_SYMBOLIC" and r["method"] == "exact_arithmetic"
    bad = dict(ok, sympy_rhs="121")
    assert harness.verify_claim(bad)["verdict"] == "REFUTED"
    big = {"claim_type": "COMBINATORIAL_COUNT",
           "sympy_lhs": "factorial(20)/(factorial(10)*factorial(10))",
           "sympy_rhs": "184756", "relation": "==", "free_symbols": [],
           "assumptions": []}
    assert harness.verify_claim(big)["verdict"] == "VERIFIED_SYMBOLIC"


def test_matrix():
    ok = {"claim_type": "MATRIX",
          "sympy_lhs": "Matrix([[1, 2], [3, 4]])**2",
          "sympy_rhs": "Matrix([[7, 10], [15, 22]])",
          "relation": "==", "free_symbols": [], "assumptions": []}
    r = harness.verify_claim(ok)
    assert r["verdict"] == "VERIFIED_SYMBOLIC"
    bad = dict(ok, sympy_rhs="Matrix([[7, 10], [15, 23]])")
    assert harness.verify_claim(bad)["verdict"] == "REFUTED"
    symbolic = {"claim_type": "MATRIX",
                "sympy_lhs": "Matrix([[a, b], [0, a]])*Matrix([[a, -b], [0, a]])",
                "sympy_rhs": "Matrix([[a**2, 0], [0, a**2]])",
                "relation": "==", "free_symbols": ["a", "b"], "assumptions": []}
    r = harness.verify_claim(symbolic)
    assert r["verdict"] in ("VERIFIED_SYMBOLIC", "VERIFIED_NUMERIC")
    not_a_matrix = dict(ok, sympy_lhs="x + 1", free_symbols=["x"])
    assert harness.verify_claim(not_a_matrix)["verdict"] == "PARSE_FAIL"


def test_recurrence_check():
    ok = {"claim_type": "RECURRENCE_CHECK",
          "sympy_lhs": "2**n + 1",
          "recurrence": "a(n+1) = 2*a(n) - 1",
          "relation": "satisfies",
          "free_symbols": ["n:integer,positive"], "assumptions": []}
    r = harness.verify_claim(ok)
    assert r["verdict"] in ("VERIFIED_SYMBOLIC", "VERIFIED_NUMERIC")
    bad = dict(ok, recurrence="a(n+1) = 2*a(n)")
    assert harness.verify_claim(bad)["verdict"] == "REFUTED"
    # the pilot's a_{k+1} = a_k**2 - 2 closed form (claim 28, as RECURRENCE_CHECK)
    pilot28 = {"claim_type": "RECURRENCE_CHECK",
               "sympy_lhs": "2**(2**n) + 2**(-(2**n))",
               "recurrence": "a(n+1) = a(n)**2 - 2",
               "relation": "satisfies",
               "free_symbols": ["n:integer,positive"], "assumptions": []}
    r = harness.verify_claim(pilot28)
    assert r["verdict"] in ("VERIFIED_SYMBOLIC", "VERIFIED_NUMERIC")


# ----------------------------------------------------------- fidelity passthrough
def test_fidelity_fields_passed_through():
    c = {"claim_id": "123:0", "row_id": 123, "question_id": 9, "split": "eval",
         "judgement": 1, "claim_type": "NUMERIC_VALUE",
         "sympy_lhs": "1 + 1", "sympy_rhs": "2", "relation": "==",
         "free_symbols": [], "assumptions": [],
         "source_quote": "one plus one is two",
         "fidelity_note": "faithful transcription",
         "fidelity": {"quote_found": True, "match_kind": "exact"},
         "load_bearing": True}
    r = harness.verify_claim(c)
    assert r["verdict"] == "VERIFIED_SYMBOLIC"
    assert r["source_quote"] == "one plus one is two"
    assert r["fidelity_note"] == "faithful transcription"
    assert r["fidelity"] == {"quote_found": True, "match_kind": "exact"}
    assert r["load_bearing"] is True
    assert r["row_id"] == 123 and r["claim_id"] == "123:0"
    assert r["split"] == "eval" and r["judgement"] == 1


# ----------------------------------------------------------- extraction validation
def test_validate_extraction_roundtrip():
    from extraction_prompt import validate_extraction
    source = "The answer is $x^2 + 1 = 5$ when $x = 2$."
    good = json.dumps({"claims": [{
        "claim_type": "NUMERIC_VALUE", "sympy_lhs": "2**2 + 1",
        "sympy_rhs": "5", "relation": "==", "free_symbols": [],
        "assumptions": [], "source_quote": "$x^2 + 1 = 5$ when $x = 2$",
        "fidelity_note": "instantiated x=2 as stated", "load_bearing": True}]})
    claims, errors = validate_extraction(good, source_text=source)
    assert not errors and len(claims) == 1
    assert claims[0]["fidelity"]["quote_found"] is True

    fenced = "```json\n" + good + "\n```"
    claims, errors = validate_extraction(fenced, source_text=source)
    assert not errors and len(claims) == 1

    bad_quote = good.replace("$x^2 + 1 = 5$ when $x = 2$",
                             "this never appears in the source")
    claims, errors = validate_extraction(bad_quote, source_text=source)
    assert errors and not claims

    garbage, errors = validate_extraction("total nonsense, no json here")
    assert not garbage and errors

    none_case = json.dumps({"claims": [{"claim_type": "NONE",
                                        "reason": "pure prose"}]})
    claims, errors = validate_extraction(none_case, source_text=source)
    assert not errors and claims[0]["claim_type"] == "NONE"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
