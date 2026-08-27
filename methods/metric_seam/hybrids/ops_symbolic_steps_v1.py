"""Exact, relation-local certificates for rational algebra equality steps.

This is an additive math capability.  It emits no a12 score and makes no
whole-proof rigor claim.  The operation parses bounded LaTeX expressions with
SymPy's Lark backend, verifies rational-function identities exactly on their
reported denominator-nonzero domain, and can exhibit a rational assignment
that refutes a *universal-identity* reading.  Claim scope remains external.

The existing, manually selected :mod:`ops_math` span extractor is reused as a
retrospective seed.  Nothing here claims that capability selection was
automatically discovered.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import itertools
from pathlib import Path
import re
import signal
import sys
from typing import Any, Iterable

import sympy as sp
from sympy.core.relational import Relational
from sympy.parsing.latex import parse_latex

try:
    from .ops_math import MathOps
except ImportError:  # pragma: no cover - direct-module/legacy import compatibility
    _HYBRIDS = Path(__file__).resolve().parent
    if str(_HYBRIDS) not in sys.path:
        sys.path.insert(0, str(_HYBRIDS))
    from ops_math import MathOps  # type: ignore[no-redef]


SCHEMA = "metric-seam.math-symbolic-step-certificates.v1"
MAX_EXPRESSION_CHARS = 220
MAX_PAIR_CANDIDATES = 40
MAX_OPERATIONS = 80
MAX_SYMBOLS = 6
PARSE_SECONDS = 0.5
SIMPLIFY_SECONDS = 0.5

_ANSWER_SPLIT_RE = re.compile(r"(?im)^\s*Answer\s*:\s*")
_ROW_BREAK_RE = re.compile(r"\\\\(?:\s*\[[^]]*\])?")
_ALIGN_WRAPPER_RE = re.compile(
    r"\\(?:begin|end)\{(?:align|aligned|alignat|gather|gathered|split|eqnarray)\*?\}"
)
_ANNOTATION_RE = re.compile(r"\\(?:tag|label)\{[^{}]*\}")
_DROP_SPACING_RE = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|quad|qquad)\b|\\[,;:!]"
)
_UNSUPPORTED_RE = re.compile(
    r"\\(?:begin|end|text|mbox|operatorname|cases|matrix|pmatrix|bmatrix|"
    r"vmatrix|array|sum|prod|int|lim|sqrt|log|ln|sin|cos|tan|exp|"
    r"leq?|geq?|neq|approx|sim|equiv|in|subset|supset|to|rightarrow|"
    r"Rightarrow|implies)\b"
)
_TRAILING_PUNCTUATION_RE = re.compile(r"[\s,;.]+$")


class _BudgetExceeded(RuntimeError):
    pass


class _TimeBudget:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.previous_handler: Any = None

    def __enter__(self):
        if not hasattr(signal, "setitimer"):
            return self
        self.previous_handler = signal.getsignal(signal.SIGALRM)

        def alarm(_signum, _frame):
            raise _BudgetExceeded("symbolic operation exceeded its CPU budget")

        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def _answer_only(text: str) -> str:
    parts = _ANSWER_SPLIT_RE.split(text or "")
    return parts[-1] if len(parts) > 1 else (text or "")


def _clean_expression(text: str) -> str:
    value = _DROP_SPACING_RE.sub("", text).strip()
    value = _TRAILING_PUNCTUATION_RE.sub("", value)
    return value


def _equation_rows(span: str) -> Iterable[list[str]]:
    normalized = _ALIGN_WRAPPER_RE.sub("", span)
    normalized = _ANNOTATION_RE.sub("", normalized)
    for raw_row in _ROW_BREAK_RE.split(normalized):
        row = raw_row.replace("&", " ").strip()
        if not row or len(row) > 4 * MAX_EXPRESSION_CHARS:
            continue
        if ":=" in row or "=:" in row or _UNSUPPORTED_RE.search(row):
            continue
        parts = [_clean_expression(value) for value in row.split("=")]
        if 2 <= len(parts) <= 4 and all(
            value and len(value) <= MAX_EXPRESSION_CHARS for value in parts
        ):
            yield parts


@lru_cache(maxsize=4096)
def _parse_rational_expression(source: str) -> sp.Basic:
    with _TimeBudget(PARSE_SECONDS):
        expression = parse_latex(source, backend="lark")
    if not isinstance(expression, sp.Basic) or isinstance(expression, Relational):
        raise ValueError("LaTeX parse is ambiguous or relational")
    if expression.has(sp.Float) or expression.atoms(sp.Function):
        raise ValueError("only exact rational algebra is supported")
    symbols = sorted(expression.free_symbols, key=str)
    if len(symbols) > MAX_SYMBOLS:
        raise ValueError("symbol budget exceeded")
    if sp.count_ops(expression) > MAX_OPERATIONS:
        raise ValueError("operation budget exceeded")
    if expression.is_rational_function(*symbols) is not True:
        raise ValueError("expression is outside rational algebra")
    return expression


def _domain_obligations(expressions: Iterable[sp.Basic]) -> list[str]:
    obligations: set[str] = set()
    for expression in expressions:
        denominator = sp.factor(sp.denom(sp.together(expression)))
        if denominator not in (sp.Integer(1), sp.Integer(-1)):
            obligations.add(f"{sp.sstr(denominator)} != 0")
    return sorted(obligations)


def _counterexample(
    lhs: sp.Basic, rhs: sp.Basic, symbols: list[sp.Symbol]
) -> dict[str, int] | None:
    if not symbols:
        difference = sp.simplify(lhs - rhs)
        return {} if difference != 0 else None
    values = (-2, -1, 0, 1, 2, 3, 5)
    assignments = itertools.islice(itertools.product(values, repeat=len(symbols)), 96)
    for tuple_values in assignments:
        substitution = dict(zip(symbols, tuple_values))
        try:
            left_value = sp.cancel(lhs.subs(substitution))
            right_value = sp.cancel(rhs.subs(substitution))
            if any(value.has(sp.zoo, sp.nan, sp.oo, -sp.oo) for value in (left_value, right_value)):
                continue
            if sp.simplify(left_value - right_value) != 0:
                return {str(symbol): int(value) for symbol, value in substitution.items()}
        except Exception:
            continue
    return None


def verify_expression_pair(
    lhs_source: str,
    rhs_source: str,
    *,
    declared_universal_scope: bool = False,
) -> dict[str, Any]:
    """Verify one expression pair without inferring its claim scope."""

    pair_hash = hashlib.sha256(
        (lhs_source + "\0" + rhs_source).encode("utf-8", errors="replace")
    ).hexdigest()
    try:
        lhs = _parse_rational_expression(_clean_expression(lhs_source))
        rhs = _parse_rational_expression(_clean_expression(rhs_source))
        symbols = sorted(lhs.free_symbols | rhs.free_symbols, key=str)
        with _TimeBudget(SIMPLIFY_SECONDS):
            difference = sp.cancel(sp.together(lhs - rhs))
        obligations = _domain_obligations((lhs, rhs))
    except (_BudgetExceeded, Exception) as error:
        return {
            "pair_sha256": pair_hash,
            "status": "parse_noncoverage",
            "reason": type(error).__name__,
            "positive_code_witness": False,
            "criterion_defect_witness": False,
            "whole_criterion_fidelity": "UNAVAILABLE",
        }

    common = {
        "pair_sha256": pair_hash,
        "symbol_count": len(symbols),
        "domain_nonzero_obligations": obligations,
        "declared_universal_scope": declared_universal_scope,
        "whole_criterion_fidelity": "UNAVAILABLE",
    }
    if difference == 0:
        return {
            **common,
            "status": "verified_rational_identity",
            "positive_code_witness": True,
            "criterion_defect_witness": False,
            "claim_scope_required": False,
        }

    with _TimeBudget(SIMPLIFY_SECONDS):
        witness = _counterexample(lhs, rhs, symbols)
    if witness is not None:
        status = (
            "universal_identity_counterexample"
            if declared_universal_scope
            else "exact_nonidentity_witness"
        )
        return {
            **common,
            "status": status,
            "positive_code_witness": True,
            "criterion_defect_witness": declared_universal_scope,
            "claim_scope_required": not declared_universal_scope,
            "counterexample_assignment": witness,
        }
    return {
        **common,
        "status": "symbolically_unresolved",
        "positive_code_witness": False,
        "criterion_defect_witness": False,
        "claim_scope_required": True,
    }


def analyze_document(text: str) -> dict[str, Any]:
    """Return count-only relation evidence for the answer, never a quality score."""

    results: list[dict[str, Any]] = []
    equality_rows = 0
    budget_exhausted = False
    for _kind, span in MathOps.extract_math_spans(_answer_only(text)):
        for parts in _equation_rows(span):
            equality_rows += 1
            for lhs, rhs in zip(parts, parts[1:]):
                if len(results) >= MAX_PAIR_CANDIDATES:
                    budget_exhausted = True
                    break
                results.append(verify_expression_pair(lhs, rhs))
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    statuses = Counter(row["status"] for row in results)
    parsed_count = len(results) - statuses["parse_noncoverage"]
    return {
        "schema": SCHEMA,
        "relation_id": "explicit_rational_equality_preservation",
        "equality_rows_seen": equality_rows,
        "pair_candidate_count": len(results),
        "parsed_rational_pair_count": parsed_count,
        "verified_rational_identity_count": statuses["verified_rational_identity"],
        "exact_nonidentity_witness_count": statuses["exact_nonidentity_witness"],
        "universal_identity_counterexample_count": statuses[
            "universal_identity_counterexample"
        ],
        "symbolically_unresolved_count": statuses["symbolically_unresolved"],
        "parse_noncoverage_count": statuses["parse_noncoverage"],
        "positive_code_witness_count": sum(
            row.get("positive_code_witness") is True for row in results
        ),
        "criterion_defect_witness_count": 0,
        "abstained": parsed_count == 0,
        "pair_budget_exhausted": budget_exhausted,
        "whole_criterion_fidelity": "UNAVAILABLE",
        "whole_criterion_scalar": None,
    }


class SymbolicStepOpsV1:
    verify_expression_pair = staticmethod(verify_expression_pair)
    analyze_document = staticmethod(analyze_document)
