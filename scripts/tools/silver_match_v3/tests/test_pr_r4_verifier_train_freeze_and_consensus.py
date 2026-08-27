from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.build_three_order_consensus_proposals import build
from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.freeze_pr_r4_verifier_train_outputs import (
    LOCK_SCHEMA,
    freeze,
)


ORDERS = ("original", "hashed", "reverse")


def _artifact(path: Path, **extra: object) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256_file(path), **extra}


def _prediction(
    uid: str,
    order: str,
    decision: str,
    metric_id: str | None,
    prompt_sha: str,
) -> dict:
    return {
        "norm_uid": uid,
        "task": "press-releases",
        "order_mode": order,
        "model": "test-model",
        "prompt_sha256": prompt_sha,
        "candidate_bank_source_sha256": "bank-sha",
        "candidate_ids": ["m1", "m2"],
        "decision": decision,
        "metric_id": metric_id,
        "confidence": "high",
        "reason": f"reason for {uid} in {order}",
        "raw_response": "valid response",
        "parse_error": None,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    candidates = tmp_path / "candidates.jsonl"
    prompt = tmp_path / "prompt.txt"
    write_jsonl(
        candidates,
        [
            {"norm_uid": uid, "candidates": [{"metric_id": "m1"}, {"metric_id": "m2"}]}
            for uid in ("u1", "u2", "u3")
        ],
    )
    prompt.write_text("frozen R4 prompt\n", encoding="utf-8")
    prompt_sha = sha256_file(prompt)

    outputs: dict[str, Path] = {}
    for order in ORDERS:
        path = tmp_path / f"{order}.jsonl"
        write_jsonl(
            path,
            [
                _prediction("u1", order, "MATCH", "m1", prompt_sha),
                _prediction("u2", order, "NO_CANDIDATE_FITS", None, prompt_sha),
                _prediction(
                    "u3",
                    order,
                    "MATCH",
                    "m2" if order == "reverse" else "m1",
                    prompt_sha,
                ),
            ],
        )
        meta = {
            "selection_role": "train",
            "order_mode": order,
            "model": "test-model",
            "eligible_count": 3,
            "new_count": 3,
            "invalid_count": 0,
            "api_request_count": 3,
            "input_candidates_sha256": sha256_file(candidates),
            "prompt_sha256": prompt_sha,
            "output_sha256": sha256_file(path),
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        outputs[order] = path

    lock = {
        "schema_version": LOCK_SCHEMA,
        "status": "FROZEN_BEFORE_R4_OPTIMIZE_INFERENCE",
        "task": "press-releases",
        "role": "optimize_only_for_future_verifier_training",
        "adjudicator_select_already_consumed": True,
        "adjudicator_prompt_frozen_before_select": True,
        "verifier_selection_requires_a_new_source-disjoint_panel": True,
        "inputs": {
            "candidates": _artifact(candidates, count=3, k=2),
            "frozen_r4_prompt": _artifact(prompt),
        },
        "inference": {
            "orders": list(ORDERS),
            "model": "test-model",
            "maximum_requests_per_order": 4,
            "maximum_total_requests": 12,
        },
        "outputs": {order: str(path) for order, path in outputs.items()},
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock_path, outputs


def test_freeze_binds_complete_three_order_outputs_before_verifier_authoring(
    tmp_path: Path,
) -> None:
    lock_path, outputs = _fixture(tmp_path)

    frozen = freeze(lock_path)

    assert frozen["status"] == "FROZEN_COMPLETE_BEFORE_VERIFIER_AUTHORING"
    assert frozen["row_count"] == 3
    assert frozen["total_api_requests"] == 9
    assert frozen["truth_content_read_by_freezer"] is False
    assert frozen["verifier_selection_requires_new_source_disjoint_panel"] is True
    assert {
        order: frozen["outputs"][order]["predictions"]["sha256"]
        for order in ORDERS
    } == {order: sha256_file(outputs[order]) for order in ORDERS}


def test_freeze_fails_closed_on_prediction_or_metadata_drift(tmp_path: Path) -> None:
    lock_path, outputs = _fixture(tmp_path)
    with outputs["hashed"].open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="metadata drift/incompleteness"):
        freeze(lock_path)


def test_three_order_builder_keeps_only_exact_match_consensus(tmp_path: Path) -> None:
    lock_path, outputs = _fixture(tmp_path)
    assert freeze(lock_path)["status"] == "FROZEN_COMPLETE_BEFORE_VERIFIER_AUTHORING"

    selected, report = build(outputs, "press-releases")

    assert [row["norm_uid"] for row in selected] == ["u1"]
    assert selected[0]["consensus_metric_id"] == "m1"
    assert selected[0]["consensus_order_modes"] == list(ORDERS)
    assert report["input_count"] == 3
    assert report["decision_agreement_count"] == 3
    assert report["exact_decision_and_id_agreement_count"] == 2
    assert report["consensus_match_count"] == 1
    assert all(path.is_file() for path in outputs.values())
    assert len(list(read_jsonl(outputs["original"]))) == 3


def test_three_order_builder_rejects_task_or_coverage_mismatch(tmp_path: Path) -> None:
    _, outputs = _fixture(tmp_path)
    reverse = list(read_jsonl(outputs["reverse"]))[:-1]
    write_jsonl(outputs["reverse"], reverse)

    with pytest.raises(ValueError, match="different coverage"):
        build(outputs, "press-releases")
