from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.metric_seam.compile_math_a12_llm_requests import compile_bundle
from methods.metric_seam.run_math_a12_llm_verifier import (
    StopProduction,
    import_completed,
    load_bundle,
    recover_smoke,
    run_phase,
)


def _write_bundle(path: Path) -> list[dict]:
    rows = [
        {
            "item_key": f"train_{index:04d}",
            "ctext": f"Answer:\n$$x+{index}=x+{index}$$",
        }
        for index in range(1, 7)
    ]
    requests = compile_bundle(rows=rows, model="pinned-sonnet")
    path.write_text("".join(json.dumps(row) + "\n" for row in requests))
    return requests


def _valid(request: dict, _timeout: float):
    span = request["pair"]["lhs"]["span"]
    raw = json.dumps({"applies": True, "violated": False, "witnesses": [span]})
    return 0, raw, "", False


def test_exact_smoke_then_production(tmp_path: Path) -> None:
    bundle = tmp_path / "requests.jsonl"
    expected = _write_bundle(bundle)
    requests = load_bundle(bundle, model="pinned-sonnet")
    assert len(requests) == len(expected) == 12
    responses = tmp_path / "responses.jsonl"
    smoke = run_phase(
        requests=requests,
        responses_path=responses,
        phase="smoke",
        concurrency=2,
        timeout_seconds=1,
        invoker=_valid,
    )
    assert smoke == {
        "phase": "smoke",
        "selected": 10,
        "valid": 10,
        "contract_error": 0,
        "process_error": 0,
    }
    production = run_phase(
        requests=requests,
        responses_path=responses,
        phase="production",
        concurrency=2,
        timeout_seconds=1,
        invoker=_valid,
    )
    assert production["selected"] == 2
    assert len(responses.read_text().splitlines()) == 12


def test_any_smoke_failure_stops_production(tmp_path: Path) -> None:
    bundle = tmp_path / "requests.jsonl"
    requests = load_bundle(bundle, model="pinned-sonnet") if bundle.exists() else None
    _write_bundle(bundle)
    requests = load_bundle(bundle, model="pinned-sonnet")
    responses = tmp_path / "responses.jsonl"

    def bad(request: dict, timeout: float):
        if request == requests[0]:
            return 0, "not json", "", False
        return _valid(request, timeout)

    smoke = run_phase(
        requests=requests,
        responses_path=responses,
        phase="smoke",
        concurrency=1,
        timeout_seconds=1,
        invoker=bad,
    )
    assert smoke["valid"] == 9
    with pytest.raises(StopProduction, match="9/10"):
        run_phase(
            requests=requests,
            responses_path=responses,
            phase="production",
            concurrency=1,
            timeout_seconds=1,
            invoker=_valid,
        )


def test_parser_recovery_reuses_raw_smoke_without_model_calls(tmp_path: Path) -> None:
    bundle = tmp_path / "requests.jsonl"
    _write_bundle(bundle)
    requests = load_bundle(bundle, model="pinned-sonnet")
    source = tmp_path / "source.jsonl"
    run_phase(
        requests=requests,
        responses_path=source,
        phase="smoke",
        concurrency=2,
        timeout_seconds=1,
        invoker=_valid,
    )
    recovered = tmp_path / "recovered.jsonl"
    summary = recover_smoke(
        requests=requests,
        source_responses_path=source,
        recovered_responses_path=recovered,
    )
    assert summary == {"recovered": 10, "valid": 10, "model_calls": 0}
    assert all(json.loads(line)["recovery"]["kind"].endswith("no_model_call") for line in recovered.read_text().splitlines())


def test_import_completed_reindexes_identical_single_pass_requests(tmp_path: Path) -> None:
    bundle = tmp_path / "requests.jsonl"
    rows = [
        {"item_key": f"train_{index:04d}", "ctext": f"Answer:\n$$x+{index}=x+{index}$$"}
        for index in range(1, 12)
    ]
    two_pass = compile_bundle(rows=rows, model="pinned-sonnet")
    bundle.write_text("".join(json.dumps(row) + "\n" for row in two_pass))
    source = tmp_path / "source.jsonl"
    run_phase(
        requests=two_pass,
        responses_path=source,
        phase="smoke",
        concurrency=2,
        timeout_seconds=1,
        invoker=_valid,
    )
    # Add the remaining valid responses as production in the source bundle.
    run_phase(
        requests=two_pass,
        responses_path=source,
        phase="production",
        concurrency=2,
        timeout_seconds=1,
        invoker=_valid,
    )
    one_pass = compile_bundle(rows=rows, model="pinned-sonnet", pass_indices=(1,))
    imported = tmp_path / "imported.jsonl"
    summary = import_completed(
        requests=one_pass,
        source_responses_path=source,
        imported_responses_path=imported,
    )
    assert summary == {"imported": 11, "model_calls": 0}
    assert [json.loads(line)["request_index"] for line in imported.read_text().splitlines()] == list(range(11))
