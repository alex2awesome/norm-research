"""Frozen request/response contract for independently authored LLM verifiers.

This module compiles requests and validates retained responses; it does not
perform API calls.  Parser recovery is versioned and recorded so fenced JSON or
literal control characters cannot silently change the analysis population.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Literal, Mapping

from .diff_lines import validate_verdict_addresses
from .schema import SchemaError, Verdict, validate_json_no_floats


PARSER_VERSION = "metric-seam.verifier-response-parser.v1"
REQUEST_SCHEMA = "metric-seam.verifier-llm-request.v2"
RequestSplit = Literal["compiler_train", "sealed_heldout"]
_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """You are one independent implementation of a binary verifier.
Judge only the articulated sub-relation supplied by the user. Return exactly one JSON object:
{"applies": boolean, "violated": boolean, "witnesses": [{"path": string, "start_line": integer, "end_line": integer}]}
Use applies=false, violated=false, witnesses=[] when the diff contains no occasion to judge.
For applies=true, violated=false means the relation is satisfied; violated=true means violated.
Every applies=true verdict requires load-bearing witness lines that ground that judgment. Paths and one-based new-file line numbers must be visible in the unified diff. Do not emit a score, confidence, rationale, Markdown, or any extra keys."""


@dataclass(frozen=True)
class UnitContract:
    unit_id: str
    metric_name: str
    relation: str
    cuf_executor: str
    cuf_node_id: str

    def __post_init__(self) -> None:
        for name in ("unit_id", "metric_name", "relation", "cuf_executor", "cuf_node_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class ParsedResponse:
    verdict: Verdict
    parser_version: str
    parse_mode: Literal["strict_json", "fence_unwrapped", "control_char_recovery"]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compile_request(
    *,
    contract: UnitContract,
    item_key: str,
    ctext: str,
    pass_index: int,
    model: str,
    split: RequestSplit,
) -> dict:
    if not item_key or not ctext.startswith("diff --git "):
        raise ValueError("request requires an item key and unified diff")
    if pass_index not in (1, 2):
        raise ValueError("pass_index must be 1 or 2")
    if not model:
        raise ValueError("model must be nonempty")
    if split not in ("compiler_train", "sealed_heldout"):
        raise ValueError("split must be compiler_train or sealed_heldout")
    user_prompt = (
        f"METRIC: {contract.metric_name}\n"
        f"SUB-RELATION: {contract.relation}\n\n"
        "ITEM (unified diff):\n"
        f"{ctext}"
    )
    identity = {
        "schema": REQUEST_SCHEMA,
        "unit": asdict(contract),
        "item_key": item_key,
        "pass_index": pass_index,
        "model": model,
        "split": split,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        **identity,
        "request_sha256": _sha256(canonical),
        "response_contract": {
            "parser_version": PARSER_VERSION,
            "floats_allowed": False,
            "witnesses_must_address_visible_new_side_lines": True,
        },
    }


def _decode(candidate: str, *, strict: bool) -> object:
    def reject_float(_: str):
        raise SchemaError("floating-point values are forbidden in verifier JSON")

    try:
        value = json.loads(
            candidate,
            strict=strict,
            parse_float=reject_float,
            parse_constant=reject_float,
        )
    except SchemaError:
        raise
    except json.JSONDecodeError as exc:
        raise SchemaError("invalid verifier JSON") from exc
    validate_json_no_floats(value)
    return value


def parse_response(raw: str, *, ctext: str) -> ParsedResponse:
    """Parse with explicit provenance, then bind every witness to the item."""

    if not isinstance(raw, str) or not raw.strip():
        raise SchemaError("empty verifier response")
    candidates: list[tuple[str, str, bool]] = [("strict_json", raw, True)]
    fenced = _FENCE.match(raw)
    if fenced:
        candidates.append(("fence_unwrapped", fenced.group(1), True))
    # The final replay mode is intentionally last and always disclosed.  It
    # recovers literal tabs/newlines inside strings retained by provider APIs.
    candidates.append(("control_char_recovery", fenced.group(1) if fenced else raw, False))
    last_error: Exception | None = None
    for mode, candidate, strict in candidates:
        try:
            verdict = Verdict.from_json(_decode(candidate, strict=strict))
            validate_verdict_addresses(ctext, verdict)
            return ParsedResponse(verdict, PARSER_VERSION, mode)  # type: ignore[arg-type]
        except (SchemaError, ValueError) as exc:
            last_error = exc
    raise SchemaError(f"verifier response failed {PARSER_VERSION}") from last_error


def validate_response_envelope(envelope: Mapping[str, object], request: Mapping[str, object]) -> dict:
    """Bind a retained raw response to one digest-pinned request."""

    if envelope.get("request_sha256") != request.get("request_sha256"):
        raise SchemaError("response/request digest mismatch")
    raw = envelope.get("raw_response")
    if not isinstance(raw, str):
        raise SchemaError("response envelope has no raw_response string")
    user_prompt = request.get("user_prompt")
    if not isinstance(user_prompt, str) or "ITEM (unified diff):\n" not in user_prompt:
        raise SchemaError("request has no recoverable item projection")
    ctext = user_prompt.split("ITEM (unified diff):\n", 1)[1]
    parsed = parse_response(raw, ctext=ctext)
    return {
        "request_sha256": request["request_sha256"],
        "item_key": request["item_key"],
        "unit_id": request["unit"]["unit_id"],  # type: ignore[index]
        "pass_index": request["pass_index"],
        "split": request["split"],
        "verdict": parsed.verdict.to_json_value(),
        "parser_version": parsed.parser_version,
        "parse_mode": parsed.parse_mode,
    }


def smoke_passes(rows: list[Mapping[str, object]], *, expected: int = 10) -> bool:
    """Production may start only after an exact all-valid smoke batch."""

    return len(rows) == expected and all(row.get("status") == "valid" for row in rows)
