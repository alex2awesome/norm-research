#!/usr/bin/env python3
"""Provider payload and response contract for the matched a407 augmentation arms.

This module contains no network client.  It makes the final model-visible
contrast testable before a transport runner is allowed to execute it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from . import a407_dual_channel_pipeline_v1 as v1
    from . import prepare_a407_matched_prompt_arms_v2 as matched
except ImportError:  # pragma: no cover - direct-script compatibility
    import a407_dual_channel_pipeline_v1 as v1  # type: ignore[no-redef]
    import prepare_a407_matched_prompt_arms_v2 as matched  # type: ignore[no-redef]


RESPONSE_SCHEMA_NAME = "a407_matched_name_expressiveness_v2"


def _score_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "number", "minimum": 0.0, "maximum": 1.0},
            {"type": "null"},
        ]
    }


def response_json_schema(spec: dict[str, Any]) -> dict[str, Any]:
    reasons = spec.get("abstention_reasons")
    if not isinstance(reasons, list) or not all(
        isinstance(reason, str) for reason in reasons
    ):
        raise ValueError("matched abstention reasons are invalid")
    relations = list(matched.RELATIONS)
    relation_scores = {
        "type": "object",
        "additionalProperties": False,
        "required": relations,
        "properties": {relation: _score_schema() for relation in relations},
    }
    relation_abstentions = {
        "type": "object",
        "additionalProperties": False,
        "required": relations,
        "properties": {
            relation: {"type": "boolean"} for relation in relations
        },
    }
    return {
        "name": RESPONSE_SCHEMA_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "item_key",
                "abstained",
                "abstention_reason",
                "declared_holistic_score",
                "relation_scores",
                "relation_abstentions",
            ],
            "properties": {
                "item_key": {"type": "string", "pattern": "^heldout_[0-9]{4}$"},
                "abstained": {"type": "boolean"},
                "abstention_reason": {"type": "string", "enum": reasons},
                "declared_holistic_score": _score_schema(),
                "relation_scores": relation_scores,
                "relation_abstentions": relation_abstentions,
            },
        },
    }


def api_payload_for_request(
    request: dict[str, Any], model: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    if request.get("schema") != matched.REQUEST_SCHEMA:
        raise ValueError("matched request schema mismatch")
    if request.get("arm") not in {"raw_prompt", "hybrid"}:
        raise ValueError("matched request arm mismatch")
    if request.get("response_relations") != list(matched.RELATIONS):
        raise ValueError("matched request relation contract mismatch")
    return {
        "model": model["model"],
        "temperature": model["temperature"],
        "max_tokens": model["max_output_tokens"],
        "reasoning": model["reasoning"],
        "provider": {"require_parameters": model["provider_require_parameters"]},
        "messages": [
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": request["user_prompt"]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": response_json_schema(spec),
        },
    }


def validate_response(
    value: Any, *, request: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "item_key",
        "abstained",
        "abstention_reason",
        "declared_holistic_score",
        "relation_scores",
        "relation_abstentions",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("response keys differ from the matched contract")
    if value["item_key"] != request["item_key"]:
        raise ValueError("response item key mismatch")
    if not isinstance(value["abstained"], bool):
        raise ValueError("response abstention flag is invalid")
    if value["abstention_reason"] not in set(spec["abstention_reasons"]):
        raise ValueError("response abstention reason is invalid")
    if not v1._unit_interval_or_none(value["declared_holistic_score"]):
        raise ValueError("response holistic score is invalid")
    scores = value["relation_scores"]
    abstentions = value["relation_abstentions"]
    if not isinstance(scores, dict) or set(scores) != set(matched.RELATIONS):
        raise ValueError("response relation scores differ from the matched contract")
    if not isinstance(abstentions, dict) or set(abstentions) != set(
        matched.RELATIONS
    ):
        raise ValueError("response relation abstentions differ from the matched contract")
    for relation in matched.RELATIONS:
        if not v1._unit_interval_or_none(scores[relation]):
            raise ValueError("response relation score is invalid")
        if not isinstance(abstentions[relation], bool):
            raise ValueError("response relation abstention is invalid")
        if abstentions[relation] != (scores[relation] is None):
            raise ValueError("response score/abstention pairing is inconsistent")
    if value["abstained"]:
        if value["abstention_reason"] == "none":
            raise ValueError("global abstention requires a reason")
        if value["declared_holistic_score"] is not None or not all(
            abstentions.values()
        ):
            raise ValueError("global abstention must null every score")
    elif (
        value["abstention_reason"] != "none"
        or value["declared_holistic_score"] is None
        or all(abstentions.values())
    ):
        raise ValueError("nonabstained response is internally inconsistent")
    return value


def load_prepared(
    preparation_dir: Path = matched.OUT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest_path = preparation_dir / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != matched.MANIFEST_SCHEMA:
        raise ValueError("matched preparation manifest schema mismatch")
    for name, record in manifest.get("artifacts", {}).items():
        path = preparation_dir / name
        if not path.is_file() or v1.hash_file(path) != record.get("sha256"):
            raise ValueError("matched preparation artifact mismatch")
    raw = v1.read_jsonl(preparation_dir / "raw_prompt_requests.jsonl")
    hybrid = v1.read_jsonl(preparation_dir / "hybrid_seam_requests.jsonl")
    if len(raw) != 100 or len(hybrid) != 100:
        raise ValueError("matched preparation request count mismatch")
    spec = json.loads(
        (preparation_dir / "matched_prompt_spec.json").read_text(encoding="utf-8")
    )
    model = json.loads(matched.MODEL_SPEC_PATH.read_text(encoding="utf-8"))
    for left, right in zip(raw, hybrid):
        left_payload = api_payload_for_request(left, model, spec)
        right_payload = api_payload_for_request(right, model, spec)
        if left_payload["messages"][0] != right_payload["messages"][0]:
            raise ValueError("matched provider system messages differ")
        if left_payload["response_format"] != right_payload["response_format"]:
            raise ValueError("matched provider response contracts differ")
        if left["ctext_sha256"] != right["ctext_sha256"]:
            raise ValueError("matched provider ctext identities differ")
    return raw, hybrid, spec, model


__all__ = [
    "api_payload_for_request",
    "load_prepared",
    "response_json_schema",
    "validate_response",
]
