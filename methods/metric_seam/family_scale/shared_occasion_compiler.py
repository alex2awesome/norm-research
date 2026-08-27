"""Deterministic shared-occasion prompt compilation for verifier families.

The compiler makes batching a property of the occasion, not of a prompt
channel: every relation-conditioned channel sees byte-identical occasion IDs,
payloads, and relation batches.  Ten percent of occasions are held out from
batching as a calibration arm.  The optional blind-discovery arm is emitted in
a separate request collection and never receives the relation inventory.

This module compiles and validates JSON only.  It performs no model calls and
does not read or write experiment artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal, Mapping, Sequence


BUNDLE_SCHEMA = "metric-seam.shared-occasion-prompt-bundle.v1"
REQUEST_SCHEMA = "metric-seam.shared-occasion-relation-request.v1"
DISCOVERY_REQUEST_SCHEMA = "metric-seam.shared-occasion-blind-discovery-request.v1"
RESPONSE_PARSER_VERSION = "metric-seam.shared-occasion-response-parser.v1"
DISCOVERY_PARSER_VERSION = "metric-seam.shared-occasion-discovery-parser.v1"
CALIBRATION_PERCENT = 10
PASSES = (1, 2)

_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.DOTALL | re.IGNORECASE
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

RELATION_SYSTEM_PROMPT = """You are an independent relation verifier.
Judge every supplied relation on the one supplied occasion. Return only a JSON array with one flat object per relation, in the supplied order. Every object must have exactly these keys:
{"occasion_id": string, "relation_id": string, "applies": boolean, "violated": boolean, "witness": string or null}
Use applies=false, violated=false, witness=null when the relation has no assessable occasion. An applicable judgment requires a concise nonempty witness grounded only in the payload. A non-applicable judgment cannot be violated. Do not emit scores, confidence, rationale fields, Markdown, or additional keys."""

DISCOVERY_SYSTEM_PROMPT = """You are an independent blind relation discoverer.
Inspect only the supplied occasion. Return only a JSON array of zero or more flat objects. Every object must have exactly these keys:
{"occasion_id": string, "candidate_id": string, "witness_kind": string, "relation": string}
Candidate IDs must be unique within the response. Use short normalized noun phrases for witness_kind and short normalized descriptions for relation. Do not emit scores, confidence, Markdown, nested objects, or additional keys."""


class ContractError(ValueError):
    """Raised when compilation input or a retained response breaks contract."""


def _validated_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ContractError(f"{field} must match {_ID.pattern}")
    return value


def _validated_text(value: object, field: str, *, limit: int = 8000) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > limit
        or any(ord(char) < 32 and char not in "\n\t" for char in value)
    ):
        raise ContractError(f"{field} must be bounded nonempty text")
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError("value is not finite JSON") from exc


def _detached_json(value: object) -> object:
    return json.loads(_canonical_json(value))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Occasion:
    occasion_id: str
    payload: object

    def __post_init__(self) -> None:
        _validated_id(self.occasion_id, "occasion_id")
        _canonical_json(self.payload)

    def to_json_value(self) -> dict[str, object]:
        return {
            "occasion_id": self.occasion_id,
            "payload": _detached_json(self.payload),
        }


@dataclass(frozen=True)
class Relation:
    relation_id: str
    name: str
    description: str

    def __post_init__(self) -> None:
        _validated_id(self.relation_id, "relation_id")
        _validated_text(self.name, "relation name", limit=256)
        _validated_text(self.description, "relation description")

    def to_json_value(self) -> dict[str, str]:
        return {
            "relation_id": self.relation_id,
            "name": self.name,
            "description": self.description,
        }


@dataclass(frozen=True)
class PromptChannel:
    channel_id: str
    instruction: str

    def __post_init__(self) -> None:
        _validated_id(self.channel_id, "channel_id")
        _validated_text(self.instruction, "channel instruction")

    def to_json_value(self) -> dict[str, str]:
        return {"channel_id": self.channel_id, "instruction": self.instruction}


@dataclass(frozen=True)
class BlindDiscoveryArm:
    """Configuration for a genuinely relation-blind, separately emitted arm."""

    channel_id: str
    instruction: str
    max_candidates: int = 12

    def __post_init__(self) -> None:
        _validated_id(self.channel_id, "blind discovery channel_id")
        _validated_text(self.instruction, "blind discovery instruction")
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 50:
            raise ContractError("max_candidates must be an integer from 1 to 50")


@dataclass(frozen=True)
class ParsedRelationResponse:
    rows: tuple[dict[str, object], ...]
    parser_version: str
    parse_mode: Literal["strict_json", "fence_unwrapped"]


@dataclass(frozen=True)
class ParsedDiscoveryResponse:
    rows: tuple[dict[str, str], ...]
    parser_version: str
    parse_mode: Literal["strict_json", "fence_unwrapped"]


def _require_unique_ids(values: Sequence[object], field: str) -> None:
    ids = [getattr(value, field) for value in values]
    if len(ids) != len(set(ids)):
        raise ContractError(f"duplicate {field}")


def _rank(seed: str, namespace: str, identifier: str) -> str:
    raw = f"{seed}\0{namespace}\0{identifier}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _calibration_ids(occasions: Sequence[Occasion], seed: str) -> frozenset[str]:
    # Nearest integer to ten percent, with a one-occasion minimum.  The
    # manifest reports both target and realized counts for small samples.
    count = min(len(occasions), max(1, (len(occasions) + 5) // 10))
    ordered = sorted(
        (occasion.occasion_id for occasion in occasions),
        key=lambda occasion_id: (_rank(seed, "calibration", occasion_id), occasion_id),
    )
    return frozenset(ordered[:count])


def _batch_sizes(count: int) -> tuple[int, ...]:
    if count < 2:
        raise ContractError("at least two relations are required for batched occasions")
    sizes: list[int] = []
    remaining = count
    while remaining:
        if remaining == 4:
            sizes.extend((2, 2))
            break
        size = min(3, remaining)
        if size < 2:  # defensive; the remaining==4 branch prevents this
            raise AssertionError("invalid 2-3 partition")
        sizes.append(size)
        remaining -= size
    return tuple(sizes)


def _relation_batches(
    *,
    occasion_id: str,
    relations: Sequence[Relation],
    seed: str,
    calibration: bool,
) -> tuple[tuple[str, ...], ...]:
    ordered = sorted(
        (relation.relation_id for relation in relations),
        key=lambda relation_id: (
            _rank(seed, f"cooccurrence:{occasion_id}", relation_id),
            relation_id,
        ),
    )
    if calibration:
        return tuple((relation_id,) for relation_id in ordered)
    sizes = _batch_sizes(len(ordered))
    batches: list[tuple[str, ...]] = []
    offset = 0
    for size in sizes:
        batches.append(tuple(ordered[offset : offset + size]))
        offset += size
    return tuple(batches)


def _relation_user_prompt(
    *,
    occasion: Mapping[str, object],
    relations: Sequence[Mapping[str, str]],
    channel: PromptChannel,
) -> str:
    prompt_input = {
        "channel_instruction": channel.instruction,
        "occasion": occasion,
        "relations": list(relations),
    }
    return "INPUT_JSON:\n" + _canonical_json(prompt_input)


def _discovery_user_prompt(
    *, occasion: Mapping[str, object], arm: BlindDiscoveryArm
) -> str:
    # No relation inventory, relation ID, relation name, or relation description
    # enters this projection.
    prompt_input = {
        "discovery_instruction": arm.instruction,
        "max_candidates": arm.max_candidates,
        "occasion": occasion,
    }
    return "BLIND_DISCOVERY_INPUT_JSON:\n" + _canonical_json(prompt_input)


def compile_shared_occasion_bundle(
    *,
    occasions: Sequence[Occasion],
    relations: Sequence[Relation],
    channels: Sequence[PromptChannel],
    model: str,
    randomization_seed: str,
    blind_discovery: BlindDiscoveryArm | None = None,
) -> dict[str, object]:
    """Compile a deterministic two-pass shared-occasion request bundle."""

    if not occasions:
        raise ContractError("at least one occasion is required")
    if len(relations) < 2:
        raise ContractError("at least two relations are required")
    if not channels:
        raise ContractError("at least one relation-conditioned channel is required")
    _require_unique_ids(occasions, "occasion_id")
    _require_unique_ids(relations, "relation_id")
    _require_unique_ids(channels, "channel_id")
    _validated_text(model, "model", limit=256)
    _validated_text(randomization_seed, "randomization_seed", limit=1024)
    if blind_discovery is not None and blind_discovery.channel_id in {
        channel.channel_id for channel in channels
    }:
        raise ContractError("blind discovery channel must be separate from prompt channels")

    occasion_values = sorted(
        (occasion.to_json_value() for occasion in occasions),
        key=lambda row: str(row["occasion_id"]),
    )
    relation_values = sorted(
        (relation.to_json_value() for relation in relations),
        key=lambda row: row["relation_id"],
    )
    channel_values = sorted(
        (channel.to_json_value() for channel in channels),
        key=lambda row: row["channel_id"],
    )
    relation_by_id = {row["relation_id"]: row for row in relation_values}
    channel_by_id = {channel.channel_id: channel for channel in channels}
    calibration = _calibration_ids(occasions, randomization_seed)

    plan: list[dict[str, object]] = []
    for occasion in occasion_values:
        occasion_id = str(occasion["occasion_id"])
        batches = _relation_batches(
            occasion_id=occasion_id,
            relations=relations,
            seed=randomization_seed,
            calibration=occasion_id in calibration,
        )
        for batch_index, relation_ids in enumerate(batches):
            plan.append(
                {
                    "occasion_id": occasion_id,
                    "batch_index": batch_index,
                    "calibration_unbatched": occasion_id in calibration,
                    "relation_ids": list(relation_ids),
                }
            )

    occasion_by_id = {str(row["occasion_id"]): row for row in occasion_values}
    conditioned_requests: list[dict[str, object]] = []
    for plan_row in plan:
        occasion = occasion_by_id[str(plan_row["occasion_id"])]
        batch_relations = [
            relation_by_id[str(relation_id)]
            for relation_id in plan_row["relation_ids"]  # type: ignore[union-attr]
        ]
        for channel_id in sorted(channel_by_id):
            channel = channel_by_id[channel_id]
            for pass_index in PASSES:
                identity: dict[str, object] = {
                    "schema": REQUEST_SCHEMA,
                    "arm": "relation_conditioned",
                    "channel_id": channel_id,
                    "pass_index": pass_index,
                    "model": model,
                    "occasion": occasion,
                    "batch_index": plan_row["batch_index"],
                    "calibration_unbatched": plan_row["calibration_unbatched"],
                    "relations": batch_relations,
                    "system_prompt": RELATION_SYSTEM_PROMPT,
                    "user_prompt": _relation_user_prompt(
                        occasion=occasion,
                        relations=batch_relations,
                        channel=channel,
                    ),
                }
                conditioned_requests.append(
                    {
                        **identity,
                        "request_sha256": _sha256(identity),
                        "response_contract": {
                            "parser_version": RESPONSE_PARSER_VERSION,
                            "top_level": "array",
                            "row_keys": [
                                "occasion_id",
                                "relation_id",
                                "applies",
                                "violated",
                                "witness",
                            ],
                            "floats_allowed": False,
                        },
                    }
                )

    discovery_requests: list[dict[str, object]] = []
    if blind_discovery is not None:
        for occasion in occasion_values:
            for pass_index in PASSES:
                identity = {
                    "schema": DISCOVERY_REQUEST_SCHEMA,
                    "arm": "blind_discovery",
                    "channel_id": blind_discovery.channel_id,
                    "pass_index": pass_index,
                    "model": model,
                    "occasion": occasion,
                    "max_candidates": blind_discovery.max_candidates,
                    "system_prompt": DISCOVERY_SYSTEM_PROMPT,
                    "user_prompt": _discovery_user_prompt(
                        occasion=occasion, arm=blind_discovery
                    ),
                }
                discovery_requests.append(
                    {
                        **identity,
                        "request_sha256": _sha256(identity),
                        "response_contract": {
                            "parser_version": DISCOVERY_PARSER_VERSION,
                            "top_level": "array",
                            "row_keys": [
                                "occasion_id",
                                "candidate_id",
                                "witness_kind",
                                "relation",
                            ],
                            "floats_allowed": False,
                            "max_rows": blind_discovery.max_candidates,
                        },
                    }
                )

    conditioned_requests.sort(
        key=lambda row: (
            str(row["occasion"]["occasion_id"]),  # type: ignore[index]
            int(row["batch_index"]),
            str(row["channel_id"]),
            int(row["pass_index"]),
        )
    )
    discovery_requests.sort(
        key=lambda row: (
            str(row["occasion"]["occasion_id"]),  # type: ignore[index]
            int(row["pass_index"]),
        )
    )
    manifest_without_hash: dict[str, object] = {
        "schema": BUNDLE_SCHEMA,
        "model": model,
        "randomization_seed": randomization_seed,
        "passes": list(PASSES),
        "batch_relation_count": {"minimum": 2, "maximum": 3},
        "calibration": {
            "target_percent": CALIBRATION_PERCENT,
            "occasion_count": len(calibration),
            "total_occasion_count": len(occasions),
            "occasion_ids": sorted(calibration),
            "batch_size": 1,
        },
        "blind_discovery": {
            "enabled": blind_discovery is not None,
            "separate_from_relation_conditioned": True,
            "channel_id": blind_discovery.channel_id if blind_discovery else None,
        },
        "occasion_set_sha256": _sha256(occasion_values),
        "relation_set_sha256": _sha256(relation_values),
        "channel_set_sha256": _sha256(channel_values),
        "batch_plan_sha256": _sha256(plan),
        "conditioned_request_count": len(conditioned_requests),
        "conditioned_request_set_sha256": _sha256(
            [row["request_sha256"] for row in conditioned_requests]
        ),
        "discovery_request_count": len(discovery_requests),
        "discovery_request_set_sha256": _sha256(
            [row["request_sha256"] for row in discovery_requests]
        ),
    }
    manifest = {**manifest_without_hash, "manifest_sha256": _sha256(manifest_without_hash)}
    return {
        "schema": BUNDLE_SCHEMA,
        "manifest": manifest,
        "relation_conditioned_requests": conditioned_requests,
        "blind_discovery_requests": discovery_requests,
    }


def _decode_array(raw: str) -> tuple[list[object], Literal["strict_json", "fence_unwrapped"]]:
    if not isinstance(raw, str) or not raw.strip():
        raise ContractError("empty response")
    mode: Literal["strict_json", "fence_unwrapped"] = "strict_json"
    candidate = raw
    fenced = _FENCE.match(raw)
    if fenced:
        mode = "fence_unwrapped"
        candidate = fenced.group(1)

    def reject_float(_: str) -> None:
        raise ContractError("floating-point values are forbidden")

    try:
        value = json.loads(
            candidate,
            parse_float=reject_float,
            parse_constant=reject_float,
        )
    except ContractError:
        raise
    except json.JSONDecodeError as exc:
        raise ContractError("response is not valid JSON") from exc
    if not isinstance(value, list):
        raise ContractError("response must be a JSON array")
    return value, mode


def parse_relation_response(
    raw: str, *, request: Mapping[str, object]
) -> ParsedRelationResponse:
    """Validate one relation-conditioned response against its exact request."""

    if request.get("schema") != REQUEST_SCHEMA or request.get("arm") != "relation_conditioned":
        raise ContractError("request is not relation-conditioned")
    occasion = request.get("occasion")
    relations = request.get("relations")
    if not isinstance(occasion, dict) or not isinstance(occasion.get("occasion_id"), str):
        raise ContractError("request has no valid occasion")
    if not isinstance(relations, list) or not relations:
        raise ContractError("request has no relation batch")
    expected_ids = []
    for relation in relations:
        if not isinstance(relation, dict) or not isinstance(relation.get("relation_id"), str):
            raise ContractError("request has an invalid relation")
        expected_ids.append(relation["relation_id"])
    rows, mode = _decode_array(raw)
    if len(rows) != len(expected_ids):
        raise ContractError("response must contain exactly one row per relation")
    validated: list[dict[str, object]] = []
    keys = {"occasion_id", "relation_id", "applies", "violated", "witness"}
    for index, (row, relation_id) in enumerate(zip(rows, expected_ids)):
        if not isinstance(row, dict) or set(row) != keys:
            raise ContractError(f"response row {index} has invalid keys")
        if row["occasion_id"] != occasion["occasion_id"]:
            raise ContractError(f"response row {index} has wrong occasion_id")
        if row["relation_id"] != relation_id:
            raise ContractError(f"response row {index} has wrong relation_id or order")
        if type(row["applies"]) is not bool or type(row["violated"]) is not bool:
            raise ContractError(f"response row {index} requires boolean states")
        witness = row["witness"]
        if not row["applies"]:
            if row["violated"] or witness is not None:
                raise ContractError(f"response row {index} has invalid non-applicable state")
        elif not isinstance(witness, str) or not witness.strip() or len(witness) > 4000:
            raise ContractError(f"response row {index} requires a bounded witness")
        validated.append(dict(row))
    return ParsedRelationResponse(tuple(validated), RESPONSE_PARSER_VERSION, mode)


def parse_discovery_response(
    raw: str, *, request: Mapping[str, object]
) -> ParsedDiscoveryResponse:
    """Validate one response from the explicitly separate blind arm."""

    if request.get("schema") != DISCOVERY_REQUEST_SCHEMA or request.get("arm") != "blind_discovery":
        raise ContractError("request is not a blind-discovery request")
    occasion = request.get("occasion")
    maximum = request.get("max_candidates")
    if not isinstance(occasion, dict) or not isinstance(occasion.get("occasion_id"), str):
        raise ContractError("request has no valid occasion")
    if type(maximum) is not int:
        raise ContractError("request has no max_candidates")
    rows, mode = _decode_array(raw)
    if len(rows) > maximum:
        raise ContractError("discovery response exceeds max_candidates")
    keys = {"occasion_id", "candidate_id", "witness_kind", "relation"}
    validated: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != keys:
            raise ContractError(f"discovery row {index} has invalid keys")
        if row["occasion_id"] != occasion["occasion_id"]:
            raise ContractError(f"discovery row {index} has wrong occasion_id")
        candidate_id = _validated_id(row["candidate_id"], "candidate_id")
        if candidate_id in seen:
            raise ContractError("discovery candidate IDs must be unique")
        seen.add(candidate_id)
        _validated_text(row["witness_kind"], "witness_kind", limit=256)
        _validated_text(row["relation"], "relation", limit=1000)
        validated.append(dict(row))  # type: ignore[arg-type]
    return ParsedDiscoveryResponse(tuple(validated), DISCOVERY_PARSER_VERSION, mode)
