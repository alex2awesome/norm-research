"""Deterministic, relation-local rational-algebra certificates for new runs.

This module is the additive successor to :mod:`ops_symbolic_steps_v1`.  V1 used
``ITIMER_REAL`` around each SymPy/Lark parse.  Because the timer also covered
Lark's cold parser initialization, an identical expression could be a verified
identity on one process and ``parse_noncoverage`` on another solely because the
host crossed a 0.5 second wall-clock boundary.

V2 makes that operational boundary explicit:

* the parser is prewarmed once, before any item is classified;
* bounded item parsing and simplification have no wall-clock classification
  threshold; and
* callers that require containment use :func:`analyze_documents_isolated`.
  Its process timeout produces an execution-level ``process_timeout`` receipt
  and **no relation classifications**.  A timeout is never converted into
  parse noncoverage.

Frozen v1 sources and artifacts remain historical.  New symbolic-step runs
should import this module and, for untrusted batches, call
``analyze_documents_isolated`` rather than the v1 evaluator/worker pair.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Mapping, Sequence

import sympy as sp
from sympy.core.relational import Relational
from sympy.parsing.latex import parse_latex

from .ops_symbolic_steps_v1 import (
    MAX_EXPRESSION_CHARS,
    MAX_OPERATIONS,
    MAX_PAIR_CANDIDATES,
    MAX_SYMBOLS,
    MathOps,
    _answer_only,
    _clean_expression,
    _counterexample,
    _domain_obligations,
    _equation_rows,
)


SCHEMA = "metric-seam.math-symbolic-step-certificates.v2"
REQUEST_SCHEMA = "metric-seam.math-symbolic-step-process-request.v2"
RESULT_SCHEMA = "metric-seam.math-symbolic-step-process-result.v2"
RELATION_ID = "explicit_rational_equality_preservation"
PREWARM_EXPRESSION = r"x + 1"
DEFAULT_PROCESS_TIMEOUT_SECONDS = 120.0

ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("_ops_symbolic_steps_v2_worker.py")

_PREWARMED = False
_PREWARM_LOCK = threading.Lock()


class SymbolicWorkerError(RuntimeError):
    """The isolated worker failed for a reason other than its outer timeout."""


def _parse_backend(source: str) -> sp.Basic:
    """Single indirection used by both prewarm and bounded item parsing."""

    return parse_latex(source, backend="lark")


def prewarm_parser() -> dict[str, Any]:
    """Initialize the Lark parser once, outside the item-classification path.

    There is intentionally no local wall-clock cutoff here.  An isolated caller
    places one timeout around the complete worker process.  If initialization
    exceeds it, the caller emits a process-timeout receipt without item labels.
    """

    global _PREWARMED
    if not _PREWARMED:
        with _PREWARM_LOCK:
            if not _PREWARMED:
                probe = _parse_backend(PREWARM_EXPRESSION)
                if not isinstance(probe, sp.Basic) or isinstance(probe, Relational):
                    raise SymbolicWorkerError("symbolic parser prewarm returned invalid output")
                _PREWARMED = True
    return {
        "completed": True,
        "expression": PREWARM_EXPRESSION,
        "classification_emitted": False,
    }


@lru_cache(maxsize=4096)
def _parse_rational_expression(source: str) -> sp.Basic:
    """Parse one deterministically bounded rational expression.

    Surface and semantic complexity guards are deterministic.  Process-level
    containment, not elapsed time, handles a pathological parser invocation.
    """

    if not isinstance(source, str) or not source or len(source) > MAX_EXPRESSION_CHARS:
        raise ValueError("expression exceeds the deterministic surface bound")
    prewarm_parser()
    expression = _parse_backend(source)
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


def verify_expression_pair(
    lhs_source: str,
    rhs_source: str,
    *,
    declared_universal_scope: bool = False,
) -> dict[str, Any]:
    """Verify one pair without converting execution time into relation evidence."""

    # Prewarm failure is an execution failure, not parse noncoverage.  Keep it
    # outside the relation-level exception boundary for direct callers too.
    prewarm_parser()
    pair_hash = hashlib.sha256(
        (lhs_source + "\0" + rhs_source).encode("utf-8", errors="replace")
    ).hexdigest()
    try:
        lhs = _parse_rational_expression(_clean_expression(lhs_source))
        rhs = _parse_rational_expression(_clean_expression(rhs_source))
        symbols = sorted(lhs.free_symbols | rhs.free_symbols, key=str)
        difference = sp.cancel(sp.together(lhs - rhs))
        obligations = _domain_obligations((lhs, rhs))
    except Exception as error:
        return {
            "pair_sha256": pair_hash,
            "status": "parse_noncoverage",
            "reason": type(error).__name__,
            "positive_code_witness": False,
            "criterion_defect_witness": False,
            "whole_criterion_fidelity": "UNAVAILABLE",
            "execution_timeout": False,
        }

    common = {
        "pair_sha256": pair_hash,
        "symbol_count": len(symbols),
        "domain_nonzero_obligations": obligations,
        "declared_universal_scope": declared_universal_scope,
        "whole_criterion_fidelity": "UNAVAILABLE",
        "execution_timeout": False,
    }
    if difference == 0:
        return {
            **common,
            "status": "verified_rational_identity",
            "positive_code_witness": True,
            "criterion_defect_witness": False,
            "claim_scope_required": False,
        }

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
    """Return count-only relation evidence after deterministic parser prewarm."""

    prewarm_parser()
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
        "relation_id": RELATION_ID,
        "execution_status": "completed",
        "execution_timeout": False,
        "parser_prewarmed_before_item": True,
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


def execute_process_request(
    request: Mapping[str, Any],
    *,
    prewarm: Callable[[], Mapping[str, Any]] = prewarm_parser,
    analyzer: Callable[[str], Mapping[str, Any]] = analyze_document,
) -> dict[str, Any]:
    """Execute a validated worker request; dependency hooks support exact tests."""

    if not isinstance(request, Mapping) or set(request) != {"schema", "items"}:
        raise ValueError("process request must contain exactly schema and items")
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unexpected process request schema")
    items = request.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("process request needs a nonempty item list")
    expected_keys = [f"item_{index:04d}" for index in range(1, len(items) + 1)]
    for expected, row in zip(expected_keys, items):
        if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
            raise ValueError("worker rows are limited to item_key and ctext")
        if row.get("item_key") != expected or not isinstance(row.get("ctext"), str):
            raise ValueError("worker item keys must be ordered opaque aliases")

    prewarm_receipt = dict(prewarm())
    if prewarm_receipt.get("completed") is not True:
        raise SymbolicWorkerError("parser prewarm did not complete")
    outputs = [
        {"item_key": expected, "analysis": dict(analyzer(row["ctext"]))}
        for expected, row in zip(expected_keys, items)
    ]
    return {
        "schema": RESULT_SCHEMA,
        "execution_status": "completed",
        "timeout_scope": "outer_worker_process",
        "timeouts_are_relation_noncoverage": False,
        "prewarm": prewarm_receipt,
        "n_requested": len(items),
        "n_completed": len(outputs),
        "relation_classifications_emitted": True,
        "outputs": outputs,
    }


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_batch_items(items: Sequence[str]) -> list[dict[str, str]]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or not items:
        raise ValueError("items must be a nonempty sequence of ctext strings")
    if not all(isinstance(text, str) for text in items):
        raise ValueError("every batch item must be a ctext string")
    return [
        {"item_key": f"item_{index:04d}", "ctext": text}
        for index, text in enumerate(items, 1)
    ]


def analyze_documents_isolated(
    items: Sequence[str],
    *,
    process_timeout_seconds: float = DEFAULT_PROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Analyze a batch in one prewarmed subprocess.

    The returned timeout receipt deliberately contains ``outputs=None`` and
    ``relation_classifications_emitted=False``.  Callers must retry or report
    execution unavailability; they may not count the timeout as parse
    noncoverage or negative relation evidence.
    """

    rows = _validate_batch_items(items)
    if (
        isinstance(process_timeout_seconds, bool)
        or not isinstance(process_timeout_seconds, (int, float))
        or not math.isfinite(float(process_timeout_seconds))
        or process_timeout_seconds <= 0
    ):
        raise ValueError("process_timeout_seconds must be finite and positive")
    request = {"schema": REQUEST_SCHEMA, "items": rows}
    with tempfile.TemporaryDirectory(prefix="metric_seam_symbolic_steps_v2_") as name:
        temporary = Path(name)
        request_path = temporary / "request.json"
        result_path = temporary / "result.json"
        request_path.write_bytes(_canonical_bytes(request))
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(temporary),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
        }
        try:
            process = subprocess.run(
                [sys.executable, str(WORKER), str(request_path), str(result_path)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=float(process_timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "schema": RESULT_SCHEMA,
                "execution_status": "process_timeout",
                "timeout_scope": "outer_worker_process",
                "timeout_seconds": float(process_timeout_seconds),
                "timeouts_are_relation_noncoverage": False,
                "n_requested": len(rows),
                "n_completed": 0,
                "relation_classifications_emitted": False,
                "outputs": None,
            }
        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "worker failed")[-2000:]
            raise SymbolicWorkerError(
                f"symbolic v2 worker exited {process.returncode}: {detail}"
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SymbolicWorkerError("symbolic v2 worker returned no valid result") from error

    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("execution_status") != "completed"
        or result.get("n_requested") != len(rows)
        or result.get("n_completed") != len(rows)
        or result.get("relation_classifications_emitted") is not True
        or not isinstance(result.get("outputs"), list)
        or [row.get("item_key") for row in result["outputs"]]
        != [row["item_key"] for row in rows]
    ):
        raise SymbolicWorkerError("symbolic v2 worker result violates its contract")
    return result


class SymbolicStepOpsV2:
    """Capability facade for new-run callers."""

    prewarm_parser = staticmethod(prewarm_parser)
    verify_expression_pair = staticmethod(verify_expression_pair)
    analyze_document = staticmethod(analyze_document)
    analyze_documents_isolated = staticmethod(analyze_documents_isolated)
