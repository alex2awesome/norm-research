#!/usr/bin/env python3
"""Tier-1 verification harness for the Math.SE verification pilot.

Reads claims.jsonl (manually extracted, structured claims), attempts to verify
each with sympy, and writes results.json.

Verdicts:
  VERIFIED      symbolic proof (simplify(lhs-rhs)==0, sum/limit/integral evaluates
                to claimed value) OR all numeric sample points agree
  REFUTED       symbolic nonzero constant difference, or numeric counterexample(s)
  INCONCLUSIVE  could not decide (simplify failed AND sampling failed/errored)
  PARSE_FAIL    sympy could not parse the claim

Methods recorded: symbolic | numeric_sample | numeric_quadrature | boolean_sat | none.
NOTE: numeric_sample VERIFIED for inequalities/identities is sampling evidence
(20 points), not a proof; this is flagged via the method field.
"""
import json
import os
import random
import signal
import time

import sympy as sp
from sympy import Function, Symbol
from sympy.logic.boolalg import Equivalent, Not
from sympy.logic.inference import satisfiable

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT_S = 10        # per symbolic operation (simplify / doit), as specified
SAMPLING_CAP_S = 60   # cap for the whole 20-point sampling loop
N_SAMPLES = 20
MATCH_TOL = 1e-7      # relative tolerance to count a sample point as "equal"
REFUTE_TOL = 1e-5     # relative difference must exceed this to count as mismatch

EVAL_TYPES = (sp.Sum, sp.Integral, sp.Derivative, sp.Limit)


class HarnessTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise HarnessTimeout()


signal.signal(signal.SIGALRM, _alarm)


def with_timeout(fn, seconds=TIMEOUT_S):
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)


# ---------------------------------------------------------------- namespaces
def build_ns(claim):
    """Per-claim parse namespace: declared free symbols with assumptions."""
    ns = {}
    for spec in claim.get("free_symbols") or []:
        if ":" in spec:
            name, kindstr = spec.split(":", 1)
            kinds = set(kindstr.split(","))
        else:
            name, kinds = spec, set()
        if "function" in kinds:
            ns[name] = Function(name)
            continue
        kwargs = {"real": True}
        for k in ("integer", "positive", "nonnegative"):
            if k in kinds:
                kwargs[k] = True
        ns[name] = Symbol(name, **kwargs)
    return ns


def prepare(claim, ns):
    """Parse lhs/rhs; split assumptions into equality constraints (substituted)
    and relational constraints (used to steer sampling)."""
    lhs = sp.sympify(claim["sympy_lhs"], locals=ns)
    rhs = sp.sympify(claim["sympy_rhs"], locals=ns) if claim.get("sympy_rhs") is not None else None
    rels = []
    for a in claim.get("assumptions") or []:
        a = a.strip()
        is_eq_constraint = ("=" in a and not any(t in a for t in ("==", "<=", ">=", "!=", "<", ">")))
        if is_eq_constraint:
            L, R = a.split("=")
            eq = sp.Eq(sp.sympify(L, locals=ns), sp.sympify(R, locals=ns))
            target = sorted(eq.free_symbols, key=lambda s: s.name)[-1]
            sol = sp.solve(eq, target)[0]
            lhs = lhs.subs(target, sol)
            if rhs is not None:
                rhs = rhs.subs(target, sol)
        else:
            rels.append(sp.sympify(a, locals=ns))
    return lhs, rhs, rels


# ---------------------------------------------------------------- sampling
def sample_value(s, rng):
    if s.is_integer:
        if s.is_positive:
            return sp.Integer(rng.randint(1, 6))
        if s.is_nonnegative:
            return sp.Integer(rng.randint(0, 6))
        return sp.Integer(rng.randint(-6, 6))
    if s.is_positive:
        return sp.Float(rng.uniform(0.05, 6.0))
    if s.is_nonnegative:
        return sp.Float(rng.uniform(0.0, 6.0))
    return sp.Float(rng.uniform(-6.0, 6.0))


def sample_point(frees, rels, rng):
    """Rejection sampling respecting relational assumptions."""
    for _ in range(500):
        point = {s: sample_value(s, rng) for s in frees}
        ok = True
        for rel in rels:
            try:
                v = rel.subs(point)
                if v is not sp.true and bool(v) is not True:
                    ok = False
                    break
            except (TypeError, AttributeError):
                ok = False
                break
        if ok:
            return point
    return None


def num_eval(expr, point):
    """Complex-safe numeric evaluation at a sample point."""
    e = expr.subs(point)
    if e.atoms(*EVAL_TYPES):
        e = e.doit()
    return complex(sp.N(e, 25))


# ---------------------------------------------------------------- checkers
def numeric_equality(lhs, rhs, rels, seed):
    rng = random.Random(seed)
    frees = sorted(lhs.free_symbols | rhs.free_symbols, key=lambda s: s.name)
    matches = mismatches = errors = 0
    witness = None
    for _ in range(N_SAMPLES):
        point = sample_point(frees, rels, rng)
        if point is None:
            return None
        try:
            lv = num_eval(lhs, point)
            rv = num_eval(rhs, point)
        except Exception:
            errors += 1
            continue
        scale = max(1.0, abs(lv), abs(rv))
        d = abs(lv - rv) / scale
        if d < MATCH_TOL:
            matches += 1
        elif d > REFUTE_TOL:
            mismatches += 1
            if witness is None:
                witness = {"point": {str(k): str(v) for k, v in point.items()},
                           "rel_diff": float(d)}
        # diffs in the dead zone [MATCH_TOL, REFUTE_TOL] are counted as neither
    return {"matches": matches, "mismatches": mismatches, "errors": errors,
            "witness": witness}


def check_equality_like(lhs, rhs, rels, relation, seed):
    """EQUALITY / NUMERIC_VALUE / DERIVATIVE / LIMIT / SUM / INTEGRAL with == or !=."""
    detail = {}
    # 1) symbolic
    try:
        def sym():
            L = lhs.doit(deep=True)
            R = rhs.doit(deep=True)
            return sp.simplify(sp.expand(L - R))
        d = with_timeout(sym)
        if d == 0:
            v = "VERIFIED" if relation == "==" else "REFUTED"
            return v, "symbolic", {"simplified_diff": "0"}
        if d.is_number and (d.is_zero is False):
            v = "REFUTED" if relation == "==" else "VERIFIED"
            return v, "symbolic", {"simplified_diff": str(d)[:120]}
        detail["simplified_diff"] = str(d)[:160]
    except HarnessTimeout:
        detail["symbolic"] = "timeout"
    except Exception as e:
        detail["symbolic"] = f"{type(e).__name__}: {e}"[:160]

    # special case: no free symbols and lhs is an unevaluated Integral/Sum/Limit
    # -> numeric quadrature comparison
    frees = lhs.free_symbols | rhs.free_symbols
    if not frees:
        try:
            lv = with_timeout(lambda: complex(sp.N(lhs.doit() if not lhs.atoms(sp.Integral) else lhs.evalf(), 20)), 30)
            rv = with_timeout(lambda: complex(sp.N(rhs, 20)))
            scale = max(1.0, abs(lv), abs(rv))
            d = abs(lv - rv) / scale
            detail["lhs_num"] = repr(lv)
            detail["rhs_num"] = repr(rv)
            if d < 1e-6:
                v = "VERIFIED" if relation == "==" else "REFUTED"
                return v, "numeric_quadrature", detail
            if d > 1e-4:
                v = "REFUTED" if relation == "==" else "VERIFIED"
                return v, "numeric_quadrature", detail
        except Exception as e:
            detail["quadrature"] = f"{type(e).__name__}: {e}"[:120]
        return "INCONCLUSIVE", "none", detail

    # 2) numeric sampling over free symbols
    try:
        res = with_timeout(lambda: numeric_equality(lhs, rhs, rels, seed), SAMPLING_CAP_S)
    except HarnessTimeout:
        res = None
        detail["sampling"] = "timeout"
    if res is None:
        return "INCONCLUSIVE", "none", detail
    detail.update(res)
    m, mm = res["matches"], res["mismatches"]
    if relation == "==":
        if mm > 0 and mm >= m:
            return "REFUTED", "numeric_sample", detail
        if mm > 0:
            return "INCONCLUSIVE", "numeric_sample", detail  # mixed: suspicious
        if m >= 10:
            return "VERIFIED", "numeric_sample", detail
    else:  # "!=" claim: equation should FAIL at every admissible point
        if m > 0:
            return "REFUTED", "numeric_sample", detail
        if mm >= 10:
            return "VERIFIED", "numeric_sample", detail
    return "INCONCLUSIVE", "numeric_sample", detail


def check_inequality(lhs, rhs, rels, op, seed):
    detail = {}
    # cheap symbolic attempt
    try:
        d = with_timeout(lambda: sp.simplify(rhs - lhs))
        if op in ("<=", "<") and d.is_nonnegative:
            return "VERIFIED", "symbolic", {"rhs_minus_lhs": str(d)[:120]}
        if op in (">=", ">") and d.is_nonpositive:
            return "VERIFIED", "symbolic", {"rhs_minus_lhs": str(d)[:120]}
    except (HarnessTimeout, Exception):
        pass
    rng = random.Random(seed)
    frees = sorted(lhs.free_symbols | rhs.free_symbols, key=lambda s: s.name)
    holds = violations = errors = 0
    witness = None
    for _ in range(N_SAMPLES):
        point = sample_point(frees, rels, rng)
        if point is None:
            return "INCONCLUSIVE", "none", {"sampling": "no admissible point"}
        try:
            lv = num_eval(lhs, point)
            rv = num_eval(rhs, point)
            if abs(lv.imag) > 1e-9 or abs(rv.imag) > 1e-9:
                errors += 1
                continue
            lvr, rvr = lv.real, rv.real
        except Exception:
            errors += 1
            continue
        margin = REFUTE_TOL * max(1.0, abs(lvr), abs(rvr))
        ok = {"<=": lvr <= rvr + margin, "<": lvr < rvr + margin,
              ">=": lvr >= rvr - margin, ">": lvr > rvr - margin}[op]
        if ok:
            holds += 1
        else:
            violations += 1
            if witness is None:
                witness = {str(k): str(v) for k, v in point.items()}
    detail = {"holds": holds, "violations": violations, "errors": errors,
              "witness": witness}
    if violations > 0:
        return "REFUTED", "numeric_sample", detail
    if holds >= 10:
        return "VERIFIED", "numeric_sample", detail
    return "INCONCLUSIVE", "numeric_sample", detail


def check_boolean_equiv(lhs, rhs):
    try:
        model = with_timeout(lambda: satisfiable(Not(Equivalent(lhs, rhs))))
        if model is False:
            return "VERIFIED", "boolean_sat", {}
        return "REFUTED", "boolean_sat", {"countermodel": str(model)[:160]}
    except (HarnessTimeout, Exception) as e:
        return "INCONCLUSIVE", "none", {"error": str(e)[:120]}


def check_finiteness(lhs, relation):
    detail = {}
    val = None
    try:
        val = with_timeout(lambda: lhs.doit())
    except HarnessTimeout:
        detail["doit"] = "timeout"
    except Exception as e:
        detail["doit"] = f"{type(e).__name__}: {e}"[:120]
    if val is not None and not val.atoms(sp.Integral) and val.is_number:
        detail["value"] = str(val)[:120]
        if val.is_finite is True:
            return ("VERIFIED" if relation == "is_finite" else "REFUTED"), "symbolic", detail
        if val.is_finite is False:
            return ("VERIFIED" if relation == "is_infinite" else "REFUTED"), "symbolic", detail
    # numeric quadrature can only support finiteness, never divergence
    if relation == "is_finite":
        try:
            v = with_timeout(lambda: complex(lhs.evalf(20)), 30)
            detail["quadrature_value"] = repr(v)
            if abs(v) < 1e10:
                return "VERIFIED", "numeric_quadrature", detail
        except Exception as e:
            detail["quadrature"] = f"{type(e).__name__}: {e}"[:120]
    return "INCONCLUSIVE", "none", detail


# ---------------------------------------------------------------- main
def run_claim(claim):
    t0 = time.time()
    out = {"claim_id": claim["claim_id"], "row_id": claim["row_id"],
           "judgement": claim["judgement"], "claim_type": claim["claim_type"],
           "relation": claim.get("relation")}
    if claim["claim_type"] == "NONE":
        out.update(verdict="NO_CLAIM", method="none", detail={"reason": claim.get("reason", "")})
        return out
    try:
        ns = build_ns(claim)
        lhs, rhs, rels = prepare(claim, ns)
    except Exception as e:
        out.update(verdict="PARSE_FAIL", method="none",
                   detail={"error": f"{type(e).__name__}: {e}"[:200]})
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out
    seed = 4242 + claim["claim_id"]
    try:
        if claim["claim_type"] == "BOOLEAN_EQUIV":
            v, m, d = check_boolean_equiv(lhs, rhs)
        elif claim["relation"] in ("is_finite", "is_infinite"):
            v, m, d = check_finiteness(lhs, claim["relation"])
        elif claim["claim_type"] == "INEQUALITY":
            v, m, d = check_inequality(lhs, rhs, rels, claim["relation"], seed)
        else:
            v, m, d = check_equality_like(lhs, rhs, rels, claim["relation"], seed)
    except Exception as e:
        v, m, d = "INCONCLUSIVE", "none", {"harness_error": f"{type(e).__name__}: {e}"[:200]}
    out.update(verdict=v, method=m, detail=d)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def main():
    claims = [json.loads(line) for line in open(os.path.join(HERE, "claims.jsonl"))]
    results = []
    for c in claims:
        r = run_claim(c)
        results.append(r)
        print(f"claim {r['claim_id']:2d} row {r['row_id']:2d} "
              f"{r['claim_type']:<14} -> {r['verdict']:<13} ({r['method']}, {r.get('elapsed_s', 0)}s)",
              flush=True)
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    # quick tally
    from collections import Counter
    tally = Counter(r["verdict"] for r in results if r["verdict"] != "NO_CLAIM")
    print("\nVerdict tally (checkable claims):", dict(tally))


if __name__ == "__main__":
    main()
