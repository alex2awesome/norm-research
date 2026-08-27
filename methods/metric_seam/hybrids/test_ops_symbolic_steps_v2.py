from __future__ import annotations

import subprocess
import time

import pytest
import sympy as sp

from methods.metric_seam.hybrids import ops_symbolic_steps_v2 as symbolic


def test_v2_keeps_relation_local_identity_and_nonidentity_witnesses() -> None:
    identity = symbolic.verify_expression_pair(r"\frac{1}{x}+1", r"\frac{x+1}{x}")
    nonidentity = symbolic.verify_expression_pair("x+x", "x")

    assert identity["status"] == "verified_rational_identity"
    assert identity["domain_nonzero_obligations"] == ["x != 0"]
    assert nonidentity["status"] == "exact_nonidentity_witness"
    assert nonidentity["criterion_defect_witness"] is False
    assert identity["execution_timeout"] is False
    assert nonidentity["execution_timeout"] is False


def test_cold_prewarm_occurs_once_before_item_classification(monkeypatch) -> None:
    calls: list[str] = []

    def cold_backend(source: str) -> sp.Basic:
        calls.append(source)
        if len(calls) == 1:
            # Simulate cold initialization cost.  V1's 0.5-second alarm made
            # this elapsed time part of the relation label; v2 has no such timer.
            time.sleep(0.01)
        return sp.Symbol("x") + 1 if source == symbolic.PREWARM_EXPRESSION else sp.Symbol("x")

    monkeypatch.setattr(symbolic, "_parse_backend", cold_backend)
    monkeypatch.setattr(symbolic, "_PREWARMED", False)
    symbolic._parse_rational_expression.cache_clear()

    first = symbolic.verify_expression_pair("x", "x")
    second = symbolic.verify_expression_pair("x", "x")

    assert first["status"] == second["status"] == "verified_rational_identity"
    assert calls[0] == symbolic.PREWARM_EXPRESSION
    assert calls.count(symbolic.PREWARM_EXPRESSION) == 1


def test_prewarm_failure_is_not_relabelled_as_parse_noncoverage(monkeypatch) -> None:
    def broken_backend(_source: str) -> sp.Basic:
        raise RuntimeError("parser initialization failed")

    monkeypatch.setattr(symbolic, "_parse_backend", broken_backend)
    monkeypatch.setattr(symbolic, "_PREWARMED", False)
    symbolic._parse_rational_expression.cache_clear()

    with pytest.raises(RuntimeError, match="initialization failed"):
        symbolic.verify_expression_pair("x", "x")


def test_worker_contract_prewarm_precedes_every_analysis() -> None:
    events: list[str] = []

    def prewarm() -> dict[str, object]:
        events.append("prewarm")
        return {"completed": True, "classification_emitted": False}

    def analyze(text: str) -> dict[str, object]:
        assert events == ["prewarm"] or events[-1] == "analysis"
        events.append("analysis")
        return {"text_length": len(text)}

    result = symbolic.execute_process_request(
        {
            "schema": symbolic.REQUEST_SCHEMA,
            "items": [
                {"item_key": "item_0001", "ctext": "one"},
                {"item_key": "item_0002", "ctext": "two"},
            ],
        },
        prewarm=prewarm,
        analyzer=analyze,
    )

    assert events == ["prewarm", "analysis", "analysis"]
    assert result["execution_status"] == "completed"
    assert result["timeouts_are_relation_noncoverage"] is False


def test_process_timeout_emits_no_relation_classification(monkeypatch) -> None:
    def timeout(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=kwargs["timeout"])

    monkeypatch.setattr(symbolic.subprocess, "run", timeout)
    result = symbolic.analyze_documents_isolated(
        ["Question: q\nAnswer: $$x+x=2x$$"], process_timeout_seconds=1.25
    )

    assert result["execution_status"] == "process_timeout"
    assert result["timeouts_are_relation_noncoverage"] is False
    assert result["relation_classifications_emitted"] is False
    assert result["outputs"] is None
    assert result["n_completed"] == 0


def test_fresh_isolated_processes_agree_after_cold_prewarm() -> None:
    text = "Question: q\nAnswer: $$x+x=2x$$"
    results = [
        symbolic.analyze_documents_isolated([text], process_timeout_seconds=30)
        for _ in range(2)
    ]
    analyses = [result["outputs"][0]["analysis"] for result in results]

    assert analyses[0] == analyses[1]
    assert analyses[0]["verified_rational_identity_count"] == 1
    assert analyses[0]["parse_noncoverage_count"] == 0
    assert all(result["prewarm"]["completed"] is True for result in results)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), True])
def test_process_timeout_must_be_finite_positive(timeout) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        symbolic.analyze_documents_isolated(["ctext"], process_timeout_seconds=timeout)


def test_process_request_rejects_nonopaque_or_extra_fields() -> None:
    with pytest.raises(ValueError, match="limited"):
        symbolic.execute_process_request(
            {
                "schema": symbolic.REQUEST_SCHEMA,
                "items": [
                    {"item_key": "d01", "ctext": "text", "judgement": 1},
                ],
            }
        )
