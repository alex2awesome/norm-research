from __future__ import annotations

from copy import deepcopy
import json
import threading
import time

import pytest

from methods.metric_seam.family_scale.decomposition_requests import (
    BUNDLE_SCHEMA,
    CliResult,
    FLEET_IDS,
    RequestSchemaError,
    build_metric_submissions,
    compile_requests,
    parse_response,
    run_phase,
    validate_bundle,
)
from methods.metric_seam.family_scale.decomposition_stability import (
    SCHEMA as SUBMISSION_SCHEMA,
    load_submission,
)


def _sha(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _study(n_metrics: int = 3) -> dict:
    cells = []
    for index in range(n_metrics):
        metric_text = {
            "construct": f"Metric {index}",
            "description": f"Assess articulated relation {index} without corpus data.",
        }
        cells.append(
            {
                "metric_id": f"metric-{index}",
                "task": "code-review",
                "domain": "code",
                "level": "R1",
                "metric_text": metric_text,
                "metric_text_sha256": _sha(metric_text),
                "decomposition_input_fields": ["construct", "description"],
            }
        )
    study = {
        "schema": "metric-seam.family-scale-study.v1",
        "status": "frozen_before_decomposition_or_corpus_contact",
        "decomposition": {
            "independent_fleets": 3,
            "relations_per_metric_guidance": [2, 5],
        },
        "cells": cells,
    }
    study["study_content_sha256"] = _sha(study)
    return study


VALID_RESPONSE = json.dumps(
    {
        "relations": [
            {
                "op_class": "evidence",
                "witness_kind": "claim limitation set",
                "relation": "extract every required limitation",
            },
            {
                "op_class": "computation",
                "witness_kind": "single reference mapping",
                "relation": "map all limitations to one disclosure",
            },
        ]
    }
)


def _valid_executor(_: object) -> CliResult:
    return CliResult(0, VALID_RESPONSE, "")


def test_compile_three_neutral_ordered_requests_per_metric() -> None:
    bundle = compile_requests(_study(3), model="sonnet")
    assert bundle["schema"] == BUNDLE_SCHEMA
    assert bundle["corpus_accessed"] is False
    assert bundle["model_calls_executed"] is False
    assert len(bundle["requests"]) == 9
    assert sum(row["phase"] == "smoke" for row in bundle["requests"]) == 6
    validate_bundle(bundle)

    by_metric = {}
    for request in bundle["requests"]:
        by_metric.setdefault(request["metric_id"], []).append(request)
        assert request["metric_id"] not in request["prompt"]
        assert "code-review" not in request["prompt"]
        assert request["request_sha256"] == _sha(
            {"model": request["model"], "prompt": request["prompt"]}
        )
    for requests in by_metric.values():
        assert {row["fleet_id"] for row in requests} == set(FLEET_IDS)
        assert len({row["prompt"] for row in requests}) == 3
        assert len({row["semantic_payload_sha256"] for row in requests}) == 1
        assert {tuple(row["fleet_order"]) for row in requests} == {
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1),
        }


def test_parser_accepts_bare_or_single_fence_and_rejects_prose_or_bad_counts() -> None:
    assert len(parse_response(VALID_RESPONSE)) == 2
    assert len(parse_response(f"```json\n{VALID_RESPONSE}\n```")) == 2
    with pytest.raises(RequestSchemaError, match="not one JSON"):
        parse_response("Here is the result: " + VALID_RESPONSE)
    with pytest.raises(RequestSchemaError, match="malformed or multiple"):
        parse_response(f"```json\n{VALID_RESPONSE}\n```\n```json\n{VALID_RESPONSE}\n```")
    with pytest.raises(RequestSchemaError, match="2 to 5"):
        parse_response(json.dumps({"relations": json.loads(VALID_RESPONSE)["relations"][:1]}))
    with pytest.raises(RequestSchemaError, match="key mismatch"):
        parse_response(json.dumps({"relations": json.loads(VALID_RESPONSE)["relations"], "score": 1}))


def test_smoke_requires_six_of_six_before_production() -> None:
    bundle = compile_requests(_study(3))
    calls = 0

    def one_invalid(_: object) -> CliResult:
        nonlocal calls
        calls += 1
        return CliResult(0, VALID_RESPONSE if calls < 6 else "not-json", "")

    smoke_rows, smoke = run_phase(
        bundle, [], phase="smoke", executor=one_invalid, max_concurrency=1
    )
    assert smoke["phase_completed"] == 6
    assert smoke["phase_valid"] == 5
    assert smoke["production_unblocked"] is False
    production_called = False

    def forbidden(_: object) -> CliResult:
        nonlocal production_called
        production_called = True
        return CliResult(0, VALID_RESPONSE, "")

    with pytest.raises(RequestSchemaError, match="5/6"):
        run_phase(
            bundle,
            smoke_rows,
            phase="production",
            executor=forbidden,
            max_concurrency=4,
        )
    assert production_called is False


def test_six_of_six_smoke_unblocks_resumable_production() -> None:
    bundle = compile_requests(_study(4))
    smoke_rows, smoke = run_phase(
        bundle, [], phase="smoke", executor=_valid_executor, max_concurrency=4
    )
    assert smoke["smoke"] == {
        "required": 6,
        "completed": 6,
        "valid": 6,
        "passed": True,
    }
    production_rows, production = run_phase(
        bundle,
        smoke_rows,
        phase="production",
        executor=_valid_executor,
        max_concurrency=4,
    )
    assert production["scheduled"] == 6
    combined = smoke_rows + production_rows
    resumed_rows, resumed = run_phase(
        bundle,
        combined,
        phase="production",
        executor=lambda _: (_ for _ in ()).throw(AssertionError("must not rerun")),
        max_concurrency=4,
    )
    assert resumed_rows == []
    assert resumed["scheduled"] == 0


def test_runner_never_exceeds_four_concurrent_calls() -> None:
    bundle = compile_requests(_study(4))
    active = 0
    maximum = 0
    lock = threading.Lock()

    def measured(_: object) -> CliResult:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return CliResult(0, VALID_RESPONSE, "")

    rows, _ = run_phase(
        bundle, [], phase="smoke", executor=measured, max_concurrency=4
    )
    assert len(rows) == 6
    assert 2 <= maximum <= 4
    with pytest.raises(RequestSchemaError, match="between 1 and 4"):
        run_phase(bundle, [], phase="smoke", executor=measured, max_concurrency=5)


def test_response_and_request_digests_fail_closed() -> None:
    bundle = compile_requests(_study(2))
    rows, _ = run_phase(
        bundle, [], phase="smoke", executor=_valid_executor, max_concurrency=2
    )
    assert all(row["response_sha256"] for row in rows)
    tampered = deepcopy(rows)
    tampered[0]["raw_response"] += " "
    with pytest.raises(RequestSchemaError, match="response_sha256"):
        run_phase(
            bundle,
            tampered,
            phase="smoke",
            executor=_valid_executor,
            max_concurrency=1,
        )

    broken_bundle = deepcopy(bundle)
    broken_bundle["requests"][0]["prompt"] += "changed"
    content = dict(broken_bundle)
    content.pop("bundle_content_sha256")
    broken_bundle["bundle_content_sha256"] = _sha(content)
    with pytest.raises(RequestSchemaError, match="request_sha256"):
        validate_bundle(broken_bundle)


def test_complete_metric_emits_stability_consumable_submission() -> None:
    bundle = compile_requests(_study(2))
    rows, _ = run_phase(
        bundle, [], phase="smoke", executor=_valid_executor, max_concurrency=3
    )
    submissions = build_metric_submissions(bundle, rows)
    assert set(submissions) == {"metric-0", "metric-1"}
    for submission in submissions.values():
        assert submission["schema"] == SUBMISSION_SCHEMA
        assert len(submission["fleets"]) == 3
        report = load_submission(submission)
        assert report["input_scope"] == "metric_text_only"


def test_compile_rejects_scope_or_study_hash_drift() -> None:
    study = _study(2)
    study["cells"][0]["metric_text"]["corpus"] = "forbidden"
    study["study_content_sha256"] = _sha(
        {key: value for key, value in study.items() if key != "study_content_sha256"}
    )
    with pytest.raises(RequestSchemaError, match="extra=.*corpus"):
        compile_requests(study)

    study = _study(2)
    study["study_content_sha256"] = "0" * 64
    with pytest.raises(RequestSchemaError, match="study_content_sha256"):
        compile_requests(study)
