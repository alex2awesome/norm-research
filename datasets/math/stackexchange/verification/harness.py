#!/usr/bin/env python3
"""Tier-1 verification harness for Math.SE claims (production version).

Refactor of ../verification_pilot/verify_claims.py into an importable module,
hardened for LLM-extracted (rather than hand-extracted) claims:

  * strict claim-schema validation: malformed LLM output yields a clean
    PARSE_FAIL verdict, never a crash;
  * per-claim hard timeout (default 10s): each claim is verified in a forked
    child process that is SIGKILLed if it exceeds the budget, so pathological
    simplifications / huge-integer expansions cannot hang the pipeline.
    Inside the child, SIGALRM-based phase budgets (symbolic vs numeric) make
    sure the numeric fallback still gets a slice of the budget;
  * verdict tiers kept from the pilot:
        symbolic simplify -> equals()/residual -> 20-point numeric sampling
        (assumption-respecting, complex-safe) -> quadrature for definite
        integrals with no free symbols;
  * five claim types added on top of the pilot's:
        MODULAR             a == b (mod n)
        DIVISIBILITY        d | a   (lhs = divisor, rhs = dividend)
        COMBINATORIAL_COUNT expression == concrete integer (exact arithmetic)
        MATRIX              small concrete matrix identity (elementwise)
        RECURRENCE_CHECK    closed form satisfies a recurrence at sampled n
  * extraction-side fidelity fields (source_quote, fidelity_note, fidelity,
    load_bearing) are passed through verbatim into every result record.

Verdicts: VERIFIED_SYMBOLIC | VERIFIED_NUMERIC | REFUTED | INCONCLUSIVE |
PARSE_FAIL (plus NO_CLAIM for claim_type == "NONE" records).
VERIFIED_NUMERIC is sampling/quadrature *evidence*, not proof; the `method`
field records which tier decided.

Public API:
    validate_claim_schema(claim) -> list[str]            # [] means valid
    verify_claim(claim, timeout_s=10.0, subprocess=True) -> dict (result)
    verify_claims(claims, timeout_s=10.0) -> list[dict]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import random
import re
import signal
import time

import sympy as sp
from sympy import Function, Symbol
from sympy.core.function import AppliedUndef
from sympy.logic.boolalg import Equivalent, Not
from sympy.logic.inference import satisfiable
from sympy.matrices import MatrixBase

# ---------------------------------------------------------------- constants
DEFAULT_TIMEOUT_S = 10.0   # per-claim budget (wall clock inside the child)
HARD_KILL_GRACE_S = 3.0    # extra slack before the parent SIGKILLs the child
SYM_FRACTION = 0.4         # share of the budget the symbolic phase may consume
N_SAMPLES = 20
MATCH_TOL = 1e-7           # relative tolerance to count a sample as "equal"
REFUTE_TOL = 1e-5          # relative difference must exceed this to mismatch

EVAL_TYPES = (sp.Sum, sp.Integral, sp.Derivative, sp.Limit)

EQUALITY_LIKE_TYPES = frozenset(
    {"EQUALITY", "NUMERIC_VALUE", "LIMIT", "DERIVATIVE", "INTEGRAL", "SUM"})
CLAIM_TYPES = EQUALITY_LIKE_TYPES | frozenset(
    {"INEQUALITY", "BOOLEAN_EQUIV", "MODULAR", "DIVISIBILITY",
     "COMBINATORIAL_COUNT", "MATRIX", "RECURRENCE_CHECK", "NONE"})

RELATIONS_BY_TYPE = {
    **{t: {"==", "!=", "is_finite", "is_infinite"} for t in EQUALITY_LIKE_TYPES},
    "INEQUALITY": {"<=", "<", ">=", ">"},
    "BOOLEAN_EQUIV": {"equiv"},
    "MODULAR": {"==", "!="},
    "DIVISIBILITY": {"divides", "not_divides"},
    "COMBINATORIAL_COUNT": {"=="},
    "MATRIX": {"==", "!="},
    "RECURRENCE_CHECK": {"satisfies"},
}
DEFAULT_RELATION = {
    "MODULAR": "==", "DIVISIBILITY": "divides", "COMBINATORIAL_COUNT": "==",
    "MATRIX": "==", "RECURRENCE_CHECK": "satisfies",
}

SYMBOL_KINDS = {"integer", "positive", "nonnegative", "nonzero", "negative",
                "real", "function"}
_SYMSPEC_RE = re.compile(r"^[A-Za-z_]\w*(:[a-z_,]+)?$")

# sympify() ultimately eval()s; reject anything that smells like code, not math.
_BANNED_RE = re.compile(
    r"(__|\bimport\b|\bopen\b|\bexec\b|\beval\b|\bcompile\b|\bglobals\b|"
    r"\blocals\b|\bgetattr\b|\bsetattr\b|\bos\b|\bsys\b|\bsubprocess\b|"
    r"\blambda\b|;|\bwhile\b|\bfor\b\s)")

# Extraction-side fields the harness must carry through untouched.
PASSTHROUGH_FIELDS = ("claim_id", "row_id", "question_id", "split",
                      "judgement", "source_quote", "fidelity_note",
                      "fidelity", "load_bearing", "note")

MAX_EXPR_CHARS = 2000


# ---------------------------------------------------------------- schema
def _expr_field_errors(claim, field, required=True):
    v = claim.get(field)
    if v is None:
        return [f"missing {field}"] if required else []
    if not isinstance(v, str) or not v.strip():
        return [f"{field} must be a non-empty string"]
    if len(v) > MAX_EXPR_CHARS:
        return [f"{field} longer than {MAX_EXPR_CHARS} chars"]
    if _BANNED_RE.search(v):
        return [f"{field} contains a banned token"]
    return []


def validate_claim_schema(claim) -> list:
    """Return a list of schema errors ([] = valid). Never raises."""
    errs = []
    if not isinstance(claim, dict):
        return ["claim is not a JSON object"]
    ct = claim.get("claim_type")
    if ct not in CLAIM_TYPES:
        return [f"invalid claim_type: {ct!r}"]
    if ct == "NONE":
        return []

    rel = claim.get("relation") or DEFAULT_RELATION.get(ct)
    allowed = RELATIONS_BY_TYPE[ct]
    if rel not in allowed:
        errs.append(f"invalid relation {rel!r} for {ct} (allowed: {sorted(allowed)})")

    errs += _expr_field_errors(claim, "sympy_lhs", required=True)
    rhs_required = (rel not in ("is_finite", "is_infinite")
                    and ct != "RECURRENCE_CHECK")
    errs += _expr_field_errors(claim, "sympy_rhs", required=rhs_required)
    if ct == "MODULAR":
        errs += _expr_field_errors(claim, "modulus", required=True)
    if ct == "RECURRENCE_CHECK":
        errs += _expr_field_errors(claim, "recurrence", required=True)
        rv = claim.get("rec_var", "n")
        if not isinstance(rv, str) or not _SYMSPEC_RE.match(rv) or ":" in rv:
            errs.append(f"invalid rec_var: {rv!r}")

    fs = claim.get("free_symbols") or []
    if not isinstance(fs, list):
        errs.append("free_symbols must be a list")
    else:
        for spec in fs:
            if not isinstance(spec, str) or not _SYMSPEC_RE.match(spec):
                errs.append(f"invalid free_symbols entry: {spec!r}")
                continue
            if ":" in spec:
                kinds = set(spec.split(":", 1)[1].split(","))
                bad = kinds - SYMBOL_KINDS
                if bad:
                    errs.append(f"unknown symbol kind(s) {sorted(bad)} in {spec!r}")
    asm = claim.get("assumptions") or []
    if not isinstance(asm, list):
        errs.append("assumptions must be a list")
    else:
        for a in asm:
            if not isinstance(a, str) or not a.strip():
                errs.append(f"invalid assumption entry: {a!r}")
            elif _BANNED_RE.search(a):
                errs.append(f"assumption contains a banned token: {a!r}")
    return errs


# ---------------------------------------------------------------- timeouts
class HarnessTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise HarnessTimeout()


def _with_timeout(fn, seconds):
    """Run fn() under a SIGALRM budget. Only call in the verifier child."""
    if seconds <= 0:
        raise HarnessTimeout()
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class Budget:
    """Wall-clock budget for one claim, split into phases."""

    def __init__(self, total_s):
        self.total_s = float(total_s)
        self.t0 = time.time()

    def left(self):
        return max(0.0, self.total_s - (time.time() - self.t0))

    def sym_cap(self):
        return min(max(1.0, SYM_FRACTION * self.total_s), self.left())


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
        for k in ("integer", "positive", "nonnegative", "nonzero", "negative"):
            if k in kinds:
                kwargs[k] = True
        ns[name] = Symbol(name, **kwargs)
    return ns


def prepare(claim, ns):
    """Parse lhs/rhs; split assumptions into equality constraints (substituted)
    and relational constraints (used to steer sampling). Kept from pilot."""
    lhs = sp.sympify(claim["sympy_lhs"], locals=ns)
    rhs = (sp.sympify(claim["sympy_rhs"], locals=ns)
           if claim.get("sympy_rhs") is not None else None)
    rels = []
    for a in claim.get("assumptions") or []:
        a = a.strip()
        is_eq_constraint = ("=" in a and not any(
            t in a for t in ("==", "<=", ">=", "!=", "<", ">")))
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
        if s.is_negative:
            return sp.Integer(rng.randint(-6, -1))
        if s.is_nonzero:
            return sp.Integer(rng.choice([-1, 1]) * rng.randint(1, 6))
        return sp.Integer(rng.randint(-6, 6))
    if s.is_positive:
        return sp.Float(rng.uniform(0.05, 6.0))
    if s.is_nonnegative:
        return sp.Float(rng.uniform(0.0, 6.0))
    if s.is_negative:
        return sp.Float(rng.uniform(-6.0, -0.05))
    if s.is_nonzero:
        return sp.Float(rng.choice([-1, 1]) * rng.uniform(0.05, 6.0))
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
# Each checker returns (verdict_base, method, detail) with verdict_base in
# {"VERIFIED", "REFUTED", "INCONCLUSIVE", "PARSE_FAIL"}.

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
        # diffs in the dead zone [MATCH_TOL, REFUTE_TOL] count as neither
    return {"matches": matches, "mismatches": mismatches, "errors": errors,
            "witness": witness}


def check_equality_like(lhs, rhs, rels, relation, seed, budget):
    """EQUALITY / NUMERIC_VALUE / DERIVATIVE / LIMIT / SUM / INTEGRAL, == or !=."""
    detail = {}
    # tier 1: symbolic
    try:
        def sym():
            L = lhs.doit(deep=True)
            R = rhs.doit(deep=True)
            return sp.simplify(sp.expand(L - R))
        d = _with_timeout(sym, budget.sym_cap())
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

    frees = lhs.free_symbols | rhs.free_symbols
    # tier 3 (no frees): numeric quadrature for unevaluated Integral/Sum/Limit
    if not frees:
        try:
            def quad():
                lv = complex(sp.N(
                    lhs.doit() if not lhs.atoms(sp.Integral) else lhs.evalf(), 20))
                rv = complex(sp.N(rhs, 20))
                return lv, rv
            lv, rv = _with_timeout(quad, budget.left())
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
        except HarnessTimeout:
            detail["quadrature"] = "timeout"
        except Exception as e:
            detail["quadrature"] = f"{type(e).__name__}: {e}"[:120]
        return "INCONCLUSIVE", "none", detail

    # tier 2: numeric sampling over free symbols
    try:
        res = _with_timeout(lambda: numeric_equality(lhs, rhs, rels, seed),
                            budget.left())
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


def check_inequality(lhs, rhs, rels, op, seed, budget):
    # cheap symbolic attempt
    try:
        d = _with_timeout(lambda: sp.simplify(rhs - lhs), budget.sym_cap())
        if op in ("<=", "<") and d.is_nonnegative:
            return "VERIFIED", "symbolic", {"rhs_minus_lhs": str(d)[:120]}
        if op in (">=", ">") and d.is_nonpositive:
            return "VERIFIED", "symbolic", {"rhs_minus_lhs": str(d)[:120]}
    except (HarnessTimeout, Exception):
        pass

    def sample_loop():
        rng = random.Random(seed)
        frees = sorted(lhs.free_symbols | rhs.free_symbols, key=lambda s: s.name)
        holds = violations = errors = 0
        witness = None
        for _ in range(N_SAMPLES):
            point = sample_point(frees, rels, rng)
            if point is None:
                return None
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
        return {"holds": holds, "violations": violations, "errors": errors,
                "witness": witness}

    try:
        res = _with_timeout(sample_loop, budget.left())
    except HarnessTimeout:
        return "INCONCLUSIVE", "none", {"sampling": "timeout"}
    if res is None:
        return "INCONCLUSIVE", "none", {"sampling": "no admissible point"}
    if res["violations"] > 0:
        return "REFUTED", "numeric_sample", res
    if res["holds"] >= 10:
        return "VERIFIED", "numeric_sample", res
    return "INCONCLUSIVE", "numeric_sample", res


def check_boolean_equiv(lhs, rhs, budget):
    try:
        model = _with_timeout(lambda: satisfiable(Not(Equivalent(lhs, rhs))),
                              budget.sym_cap())
        if model is False:
            return "VERIFIED", "boolean_sat", {}
        return "REFUTED", "boolean_sat", {"countermodel": str(model)[:160]}
    except (HarnessTimeout, Exception) as e:
        return "INCONCLUSIVE", "none", {"error": str(e)[:120]}


def check_finiteness(lhs, relation, budget):
    detail = {}
    val = None
    try:
        val = _with_timeout(lambda: lhs.doit(), budget.sym_cap())
    except HarnessTimeout:
        detail["doit"] = "timeout"
    except Exception as e:
        detail["doit"] = f"{type(e).__name__}: {e}"[:120]
    if val is not None and not val.atoms(sp.Integral) and val.is_number:
        detail["value"] = str(val)[:120]
        if val.is_finite is True:
            return (("VERIFIED" if relation == "is_finite" else "REFUTED"),
                    "symbolic", detail)
        if val.is_finite is False:
            return (("VERIFIED" if relation == "is_infinite" else "REFUTED"),
                    "symbolic", detail)
    # numeric quadrature can only support finiteness, never divergence
    if relation == "is_finite":
        try:
            v = _with_timeout(lambda: complex(lhs.evalf(20)), budget.left())
            detail["quadrature_value"] = repr(v)
            if abs(v) < 1e10:
                return "VERIFIED", "numeric_quadrature", detail
        except HarnessTimeout:
            detail["quadrature"] = "timeout"
        except Exception as e:
            detail["quadrature"] = f"{type(e).__name__}: {e}"[:120]
    return "INCONCLUSIVE", "none", detail


# ------------------------------------------------ new claim-type checkers
def _as_exact_int(expr, cap):
    """Try to reduce expr to an exact sympy Integer within a time budget."""
    def ev():
        v = expr.doit(deep=True) if hasattr(expr, "doit") else expr
        if not getattr(v, "is_Integer", False):
            v = sp.simplify(v)
        if not getattr(v, "is_Integer", False):
            vn = sp.nsimplify(v, rational=True)
            if getattr(vn, "is_Integer", False):
                v = vn
        return v
    v = _with_timeout(ev, cap)
    return v if getattr(v, "is_Integer", False) else None


def check_modular(diff_expr, modulus, rels, relation, seed, budget):
    """(diff_expr == 0) mod modulus. relation '==' asserts congruence,
    '!=' asserts non-congruence. Integer arithmetic is exact, so a concrete
    evaluation is a proof and a sampled counterexample is a real refutation."""
    detail = {}
    frees = diff_expr.free_symbols | modulus.free_symbols
    if not frees:
        try:
            d = _as_exact_int(diff_expr, budget.sym_cap())
            m = _as_exact_int(modulus, budget.sym_cap())
            if d is not None and m is not None and int(m) != 0:
                cong = (int(d) % int(m) == 0)
                detail["residue"] = str(int(d) % int(m))[:60]
                hit = cong if relation == "==" else not cong
                return (("VERIFIED" if hit else "REFUTED"),
                        "exact_arithmetic", detail)
            detail["exact"] = "could not reduce to integers"
        except HarnessTimeout:
            detail["exact"] = "timeout"
        except Exception as e:
            detail["exact"] = f"{type(e).__name__}: {e}"[:120]
        return "INCONCLUSIVE", "none", detail

    # symbolic attempt on the free-symbol case
    try:
        r = _with_timeout(lambda: sp.simplify(sp.Mod(diff_expr, modulus)),
                          budget.sym_cap())
        if r == 0:
            v = "VERIFIED" if relation == "==" else "REFUTED"
            return v, "symbolic", {"mod_simplified": "0"}
        detail["mod_simplified"] = str(r)[:120]
    except HarnessTimeout:
        detail["symbolic"] = "timeout"
    except Exception as e:
        detail["symbolic"] = f"{type(e).__name__}: {e}"[:120]

    # exact integer sampling
    def sample_loop():
        rng = random.Random(seed)
        matches = mismatches = errors = 0
        witness = None
        for _ in range(N_SAMPLES):
            point = {}
            ok = True
            for s in sorted(frees, key=lambda x: x.name):
                lo, hi = 1, 30
                if s.is_positive:
                    lo, hi = 1, 30
                elif s.is_nonnegative:
                    lo, hi = 0, 30
                else:
                    lo, hi = -15, 15
                point[s] = sp.Integer(rng.randint(lo, hi))
            for rel in rels:
                try:
                    v = rel.subs(point)
                    if v is not sp.true and bool(v) is not True:
                        ok = False
                        break
                except (TypeError, AttributeError):
                    ok = False
                    break
            if not ok:
                errors += 1
                continue
            try:
                dv = diff_expr.subs(point).doit()
                mv = modulus.subs(point).doit()
                if not dv.is_Integer:
                    dv = sp.simplify(dv)
                if not mv.is_Integer:
                    mv = sp.simplify(mv)
                if not (dv.is_Integer and mv.is_Integer) or int(mv) == 0:
                    errors += 1
                    continue
                cong = (int(dv) % int(mv) == 0)
            except Exception:
                errors += 1
                continue
            hit = cong if relation == "==" else not cong
            if hit:
                matches += 1
            else:
                mismatches += 1
                if witness is None:
                    witness = {str(k): str(v) for k, v in point.items()}
        return {"matches": matches, "mismatches": mismatches,
                "errors": errors, "witness": witness}

    try:
        res = _with_timeout(sample_loop, budget.left())
    except HarnessTimeout:
        detail["sampling"] = "timeout"
        return "INCONCLUSIVE", "none", detail
    detail.update(res)
    if res["mismatches"] > 0:
        # exact integer counterexample -> genuine refutation
        return "REFUTED", "numeric_sample", detail
    if res["matches"] >= 10:
        return "VERIFIED", "numeric_sample", detail
    return "INCONCLUSIVE", "numeric_sample", detail


def check_comb_count(lhs, rhs, budget):
    """COMBINATORIAL_COUNT: lhs must equal the concrete integer rhs exactly."""
    detail = {}
    if lhs.free_symbols | rhs.free_symbols:
        return "INCONCLUSIVE", "none", {
            "error": "COMBINATORIAL_COUNT requires a concrete (symbol-free) claim"}
    try:
        lv = _as_exact_int(lhs, budget.sym_cap())
        rv = _as_exact_int(rhs, budget.sym_cap())
        if lv is not None and rv is not None:
            detail["lhs_value"] = str(int(lv))[:80]
            return (("VERIFIED" if int(lv) == int(rv) else "REFUTED"),
                    "exact_arithmetic", detail)
        detail["exact"] = "could not reduce to integers"
    except HarnessTimeout:
        detail["exact"] = "timeout"
    except Exception as e:
        detail["exact"] = f"{type(e).__name__}: {e}"[:120]
    # numeric fallback
    try:
        def quad():
            return (complex(sp.N(lhs.doit(deep=True), 30)),
                    complex(sp.N(rhs, 30)))
        lv, rv = _with_timeout(quad, budget.left())
        detail["lhs_num"], detail["rhs_num"] = repr(lv), repr(rv)
        rel = abs(lv - rv) / max(1.0, abs(rv))
        if rel < 1e-10:
            return "VERIFIED", "numeric_quadrature", detail
        if rel > 1e-6:
            return "REFUTED", "numeric_quadrature", detail
    except HarnessTimeout:
        detail["quadrature"] = "timeout"
    except Exception as e:
        detail["quadrature"] = f"{type(e).__name__}: {e}"[:120]
    return "INCONCLUSIVE", "none", detail


def check_matrix(lhs, rhs, rels, relation, seed, budget):
    """Elementwise identity between small concrete matrices."""
    detail = {}
    try:
        L = _with_timeout(lambda: lhs.doit(deep=True), budget.sym_cap())
        R = _with_timeout(lambda: rhs.doit(deep=True), budget.sym_cap())
    except HarnessTimeout:
        return "INCONCLUSIVE", "none", {"doit": "timeout"}
    except Exception as e:
        return "INCONCLUSIVE", "none", {"doit": f"{type(e).__name__}: {e}"[:120]}
    if not (isinstance(L, MatrixBase) and isinstance(R, MatrixBase)):
        return "PARSE_FAIL", "none", {
            "error": "MATRIX claim sides did not parse to matrices"}
    if L.shape != R.shape:
        v = "REFUTED" if relation == "==" else "VERIFIED"
        return v, "symbolic", {"shape_lhs": str(L.shape), "shape_rhs": str(R.shape)}
    # tier 1: symbolic, elementwise
    try:
        D = _with_timeout(lambda: sp.simplify(sp.expand(L - R)), budget.sym_cap())
        if D == sp.zeros(*L.shape):
            v = "VERIFIED" if relation == "==" else "REFUTED"
            return v, "symbolic", {"simplified_diff": "0"}
        nonzero_const = [e for e in D if e.is_number and (e.is_zero is False)]
        if nonzero_const:
            v = "REFUTED" if relation == "==" else "VERIFIED"
            return v, "symbolic", {"nonzero_entry": str(nonzero_const[0])[:120]}
        detail["simplified_diff"] = str(D)[:160]
    except HarnessTimeout:
        detail["symbolic"] = "timeout"
    except Exception as e:
        detail["symbolic"] = f"{type(e).__name__}: {e}"[:160]

    # tier 2: numeric sampling, max relative diff across entries
    def sample_loop():
        rng = random.Random(seed)
        frees = sorted(L.free_symbols | R.free_symbols, key=lambda s: s.name)
        matches = mismatches = errors = 0
        witness = None
        entries = list(zip(list(L), list(R)))
        for _ in range(N_SAMPLES):
            point = sample_point(frees, rels, rng)
            if point is None:
                return None
            try:
                worst = 0.0
                for le, re_ in entries:
                    lv = num_eval(le, point)
                    rv = num_eval(re_, point)
                    scale = max(1.0, abs(lv), abs(rv))
                    worst = max(worst, abs(lv - rv) / scale)
            except Exception:
                errors += 1
                continue
            if worst < MATCH_TOL:
                matches += 1
            elif worst > REFUTE_TOL:
                mismatches += 1
                if witness is None:
                    witness = {"point": {str(k): str(v) for k, v in point.items()},
                               "max_rel_diff": float(worst)}
        return {"matches": matches, "mismatches": mismatches,
                "errors": errors, "witness": witness}

    try:
        res = _with_timeout(sample_loop, budget.left())
    except HarnessTimeout:
        detail["sampling"] = "timeout"
        return "INCONCLUSIVE", "none", detail
    if res is None:
        detail["sampling"] = "no admissible point"
        return "INCONCLUSIVE", "none", detail
    detail.update(res)
    m, mm = res["matches"], res["mismatches"]
    if relation == "==":
        if mm > 0 and mm >= m:
            return "REFUTED", "numeric_sample", detail
        if mm > 0:
            return "INCONCLUSIVE", "numeric_sample", detail
        if m >= 10:
            return "VERIFIED", "numeric_sample", detail
    else:
        if m > 0:
            return "REFUTED", "numeric_sample", detail
        if mm >= 10:
            return "VERIFIED", "numeric_sample", detail
    return "INCONCLUSIVE", "numeric_sample", detail


def check_recurrence(claim, ns, rels, seed, budget):
    """RECURRENCE_CHECK: closed form (sympy_lhs, in rec_var) satisfies the
    recurrence (written with an undefined function, default a(...)) — checked
    symbolically, else at sampled integer indices."""
    detail = {}
    var_name = claim.get("rec_var", "n")
    func_name = claim.get("func_name", "a")
    if var_name not in ns:
        ns[var_name] = Symbol(var_name, integer=True, positive=True)
    if func_name not in ns:
        ns[func_name] = Function(func_name)
    var = ns[var_name]
    closed = sp.sympify(claim["sympy_lhs"], locals=ns)

    rec = claim["recurrence"].strip()
    if "==" in rec:
        Ls, Rs = rec.split("==", 1)
    elif "=" in rec and not any(t in rec for t in ("<=", ">=", "!=")):
        Ls, Rs = rec.split("=", 1)
    else:
        Ls, Rs = rec, "0"
    expr = sp.sympify(Ls, locals=ns) - sp.sympify(Rs, locals=ns)

    apps = [f for f in expr.atoms(AppliedUndef)
            if f.func == ns[func_name]]
    if not apps:
        return "PARSE_FAIL", "none", {
            "error": f"recurrence contains no {func_name}(...) terms"}
    expr = expr.subs({app: closed.subs(var, app.args[0]) for app in apps})

    # tier 1: symbolic
    try:
        d = _with_timeout(lambda: sp.simplify(sp.expand(expr)), budget.sym_cap())
        if d == 0:
            return "VERIFIED", "symbolic", {"simplified_residual": "0"}
        if d.is_number and (d.is_zero is False):
            return "REFUTED", "symbolic", {"simplified_residual": str(d)[:120]}
        detail["simplified_residual"] = str(d)[:160]
    except HarnessTimeout:
        detail["symbolic"] = "timeout"
    except Exception as e:
        detail["symbolic"] = f"{type(e).__name__}: {e}"[:160]

    # tier 2: sample integer indices (small, to dodge double-exponential blowup)
    n_start = claim.get("n_start", 1)
    try:
        n_start = int(n_start)
    except (TypeError, ValueError):
        n_start = 1

    def sample_loop():
        rng = random.Random(seed)
        other = sorted(expr.free_symbols - {var}, key=lambda s: s.name)
        matches = mismatches = errors = 0
        witness = None
        for _ in range(N_SAMPLES):
            point = sample_point(other, rels, rng) if other else {}
            if point is None:
                return None
            point = dict(point)
            point[var] = sp.Integer(rng.randint(n_start, n_start + 9))
            try:
                rv = num_eval(expr, point)
                ref = num_eval(closed, {var: point[var],
                                        **{k: v for k, v in point.items()
                                           if k != var}})
                scale = max(1.0, abs(ref))
                d = abs(rv) / scale
            except Exception:
                errors += 1
                continue
            if d < MATCH_TOL:
                matches += 1
            elif d > REFUTE_TOL:
                mismatches += 1
                if witness is None:
                    witness = {"point": {str(k): str(v) for k, v in point.items()},
                               "rel_residual": float(d)}
        return {"matches": matches, "mismatches": mismatches,
                "errors": errors, "witness": witness}

    try:
        res = _with_timeout(sample_loop, budget.left())
    except HarnessTimeout:
        detail["sampling"] = "timeout"
        return "INCONCLUSIVE", "none", detail
    if res is None:
        detail["sampling"] = "no admissible point"
        return "INCONCLUSIVE", "none", detail
    detail.update(res)
    m, mm = res["matches"], res["mismatches"]
    if mm > 0 and mm >= m:
        return "REFUTED", "numeric_sample", detail
    if mm > 0:
        return "INCONCLUSIVE", "numeric_sample", detail
    if m >= 10:
        return "VERIFIED", "numeric_sample", detail
    return "INCONCLUSIVE", "numeric_sample", detail


# ---------------------------------------------------------------- dispatch
SYMBOLIC_METHODS = frozenset({"symbolic", "boolean_sat", "exact_arithmetic"})


def _seed_for(claim):
    cid = claim.get("claim_id")
    if isinstance(cid, int):
        return 4242 + cid       # keep pilot parity
    if cid is None:
        cid = json.dumps(claim.get("sympy_lhs", ""), sort_keys=True)
    import hashlib
    return 4242 + int(hashlib.md5(str(cid).encode()).hexdigest()[:8], 16)


def _base_result(claim):
    out = {}
    for k in PASSTHROUGH_FIELDS:
        if k in claim:
            out[k] = claim[k]
    out["claim_type"] = claim.get("claim_type")
    out["relation"] = claim.get("relation") or DEFAULT_RELATION.get(
        claim.get("claim_type"), None)
    return out


def _verify_in_process(claim, timeout_s):
    """Verify one schema-valid, non-NONE claim. Must run in a process whose
    main thread we own (SIGALRM). Returns a JSON-safe result dict."""
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.time()
    out = _base_result(claim)
    budget = Budget(timeout_s)
    relation = out["relation"]
    ct = claim["claim_type"]
    seed = _seed_for(claim)

    try:
        ns = build_ns(claim)
        if ct == "RECURRENCE_CHECK":
            # parsing happens inside the checker (needs the recurrence string)
            lhs = rhs = None
            rels = []
            for a in claim.get("assumptions") or []:
                if not ("=" in a and not any(
                        t in a for t in ("==", "<=", ">=", "!=", "<", ">"))):
                    rels.append(sp.sympify(a, locals=ns))
        else:
            lhs, rhs, rels = prepare(claim, ns)
        modulus = (sp.sympify(claim["modulus"], locals=ns)
                   if ct == "MODULAR" else None)
    except HarnessTimeout:
        out.update(verdict="INCONCLUSIVE", method="none",
                   detail={"error": "timeout during parse"})
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out
    except Exception as e:
        out.update(verdict="PARSE_FAIL", method="none",
                   detail={"error": f"{type(e).__name__}: {e}"[:200]})
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    try:
        if ct == "BOOLEAN_EQUIV":
            v, m, d = check_boolean_equiv(lhs, rhs, budget)
        elif ct == "MODULAR":
            v, m, d = check_modular(lhs - rhs, modulus, rels, relation,
                                    seed, budget)
        elif ct == "DIVISIBILITY":
            # lhs = divisor, rhs = dividend; "d divides a" <=> a == 0 (mod d)
            mod_rel = "==" if relation == "divides" else "!="
            v, m, d = check_modular(rhs, lhs, rels, mod_rel, seed, budget)
        elif ct == "COMBINATORIAL_COUNT":
            v, m, d = check_comb_count(lhs, rhs, budget)
        elif ct == "MATRIX":
            v, m, d = check_matrix(lhs, rhs, rels, relation, seed, budget)
        elif ct == "RECURRENCE_CHECK":
            v, m, d = check_recurrence(claim, ns, rels, seed, budget)
        elif relation in ("is_finite", "is_infinite"):
            v, m, d = check_finiteness(lhs, relation, budget)
        elif ct == "INEQUALITY":
            v, m, d = check_inequality(lhs, rhs, rels, relation, seed, budget)
        else:
            v, m, d = check_equality_like(lhs, rhs, rels, relation, seed, budget)
    except HarnessTimeout:
        v, m, d = "INCONCLUSIVE", "none", {"error": "claim budget exhausted"}
    except Exception as e:
        v, m, d = "INCONCLUSIVE", "none", {
            "harness_error": f"{type(e).__name__}: {e}"[:200]}

    if v == "VERIFIED":
        v = "VERIFIED_SYMBOLIC" if m in SYMBOLIC_METHODS else "VERIFIED_NUMERIC"
    out.update(verdict=v, method=m, detail=d)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def _child_main(claim, timeout_s, conn):
    try:
        res = _verify_in_process(claim, timeout_s)
        conn.send(json.loads(json.dumps(res, default=str)))
    except Exception as e:
        conn.send({**_base_result(claim), "verdict": "INCONCLUSIVE",
                   "method": "none",
                   "detail": {"worker_error": f"{type(e).__name__}: {e}"[:200]}})
    finally:
        conn.close()


def _mp_context():
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else "spawn")


def verify_claim(claim, timeout_s=DEFAULT_TIMEOUT_S, subprocess=True):
    """Verify one claim. Returns a result dict; never raises, never hangs
    longer than timeout_s + HARD_KILL_GRACE_S (when subprocess=True)."""
    out = _base_result(claim) if isinstance(claim, dict) else {}
    errs = validate_claim_schema(claim)
    if errs:
        out.update(verdict="PARSE_FAIL", method="none",
                   detail={"schema_errors": errs[:10]}, elapsed_s=0.0)
        return out
    if claim["claim_type"] == "NONE":
        out.update(verdict="NO_CLAIM", method="none",
                   detail={"reason": str(claim.get("reason", ""))[:300]},
                   elapsed_s=0.0)
        return out
    if not subprocess:
        return _verify_in_process(claim, timeout_s)

    ctx = _mp_context()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child_main, args=(claim, timeout_s, child_conn),
                    daemon=False)
    t0 = time.time()
    p.start()
    child_conn.close()
    result = None
    try:
        if parent_conn.poll(timeout_s + HARD_KILL_GRACE_S):
            result = parent_conn.recv()
    except (EOFError, OSError):
        result = None
    finally:
        parent_conn.close()
    p.join(timeout=1.0)
    if p.is_alive():
        p.kill()
        p.join()
    if result is None:
        out.update(verdict="INCONCLUSIVE", method="none",
                   detail={"error": "hard timeout (worker killed)"
                           if time.time() - t0 >= timeout_s
                           else f"worker died (exitcode={p.exitcode})"},
                   elapsed_s=round(time.time() - t0, 2))
        return out
    return result


def verify_claims(claims, timeout_s=DEFAULT_TIMEOUT_S, subprocess=True):
    """Verify an iterable of claims sequentially; returns a list of results."""
    return [verify_claim(c, timeout_s=timeout_s, subprocess=subprocess)
            for c in claims]


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="Verify a claims.jsonl file")
    ap.add_argument("claims_jsonl")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()
    tally = Counter()
    with open(args.claims_jsonl) as fh:
        for line in fh:
            c = json.loads(line)
            r = verify_claim(c, timeout_s=args.timeout)
            tally[r["verdict"]] += 1
            print(f"claim {str(c.get('claim_id')):>8} row "
                  f"{str(c.get('row_id')):>8} {str(c.get('claim_type')):<18} "
                  f"-> {r['verdict']:<18} ({r['method']}, {r.get('elapsed_s')}s)")
    print("\nVerdict tally:", dict(tally))
