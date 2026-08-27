#!/usr/bin/env python3
"""Freeze a matched raw/hybrid a407 contrast before reference access.

The original v1 request arms changed both the evidence surface and the prompted
relation/output contract.  They remain useful exploratory arms, but do not
isolate seam placement.  This additive preparation gives both arms the same
system prompt, user-object shape, six relations, and response contract.  The
only prompt-visible intervention is ``codescope_v3_facts``: null for raw and the
deterministic fact object for hybrid.

This module reads only the already-sanitized clean preparation.  It performs no
historical-reference access, evaluation, model/API call, or GPU operation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

try:
    from . import a407_dual_channel_pipeline_v1 as v1
except ImportError:  # pragma: no cover - direct-script compatibility
    import a407_dual_channel_pipeline_v1 as v1  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[3]
SOURCE_PREPARATION = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "a407_dual_channel_prepare_002_clean"
)
OUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "a407_matched_prompt_prepare_003_blind"
)
SPEC_PATH = Path(__file__).with_name("a407_matched_prompt_spec_v2.json")
MODEL_SPEC_PATH = Path(__file__).with_name(
    "a407_glm47_openrouter_reasoning_off_model_v1.json"
)

REQUEST_SCHEMA = "metric-seam.a407-matched-prompt-request.v2"
MANIFEST_SCHEMA = "metric-seam.a407-matched-prompt-preparation.v2"
RELATIONS = v1.ALL_RELATIONS


def _write_readonly(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _json_bytes(value: Any) -> bytes:
    return v1.canonical_bytes(value)


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _load_spec() -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("schema") != "metric-seam.a407-matched-prompt-spec.v2":
        raise ValueError("matched prompt spec schema mismatch")
    if tuple(spec.get("relation_semantics", {}).keys()) != RELATIONS:
        raise ValueError("matched prompt relation order/content mismatch")
    if spec.get("output_source_text_allowed") is not False:
        raise ValueError("matched prompt must forbid source-text output")
    return spec


def _render_system_prompt(spec: dict[str, Any]) -> str:
    """Render the actual construct and relation definitions into model-visible text."""

    return (
        str(spec["system_prompt"])
        + "\n\nARTICULATED_CONSTRUCT\n"
        + str(spec["construct"])
        + "\n\nRELATION_DEFINITIONS_JSON\n"
        + json.dumps(
            spec["relation_semantics"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _request(
    *,
    arm: str,
    ordinal: int,
    item_key: str,
    ctext: str,
    facts: dict[str, Any] | None,
    spec: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    if arm not in {"raw_prompt", "hybrid"}:
        raise ValueError("unknown matched arm")
    if (arm == "raw_prompt") != (facts is None):
        raise ValueError("facts must be null only for the raw arm")
    prompt_input = {
        "codescope_v3_facts": facts,
        "ctext": ctext,
        "item_key": item_key,
    }
    material = {
        "schema": REQUEST_SCHEMA,
        "arm": arm,
        "heldout_ordinal": ordinal,
        "item_key": item_key,
        "ctext_sha256": v1.hash_value(ctext),
        "codescope_v3_facts_present": facts is not None,
        "codescope_v3_facts_sha256": (
            None if facts is None else v1.hash_value(facts)
        ),
        "prompt_spec_sha256": v1.hash_value(spec),
        "model_spec_sha256": v1.hash_value(model),
        "system_prompt": _render_system_prompt(spec),
        "user_prompt": "INPUT_JSON\n" + json.dumps(
            prompt_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "response_relations": list(RELATIONS),
        "output_source_text_allowed": False,
        "historical_reference_available": False,
    }
    request_sha = v1.hash_value(material)
    return {
        **material,
        "request_id": f"a407_matched_{arm}_{ordinal:04d}_{request_sha[:16]}",
        "request_sha256": request_sha,
    }


def build_matched_arms(
    raw_v1: list[dict[str, Any]],
    hybrid_v1: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    model: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(raw_v1) != len(hybrid_v1) or not raw_v1:
        raise ValueError("source request arms have unequal or empty support")
    raw_out: list[dict[str, Any]] = []
    hybrid_out: list[dict[str, Any]] = []
    for ordinal, (raw, hybrid) in enumerate(zip(raw_v1, hybrid_v1), 1):
        if raw.get("item_key") != hybrid.get("item_key"):
            raise ValueError("source arm item keys differ")
        if raw.get("heldout_ordinal") != ordinal or hybrid.get(
            "heldout_ordinal"
        ) != ordinal:
            raise ValueError("source arm ordinal drift")
        raw_input = raw.get("input")
        hybrid_input = hybrid.get("input")
        if not isinstance(raw_input, dict) or raw_input != hybrid_input:
            raise ValueError("source arm ctext inputs differ")
        if set(raw_input) != {"ctext", "item_key"}:
            raise ValueError("source arm input exceeds the sanitized allowlist")
        item_key = raw_input["item_key"]
        ctext = raw_input["ctext"]
        facts = hybrid.get("codescope_v3_facts")
        if not isinstance(item_key, str) or not isinstance(ctext, str):
            raise ValueError("source input types are invalid")
        if not isinstance(facts, dict) or facts.get("schema") != (
            "metric-seam.code-scope-declaration-use-graph.v3"
        ):
            raise ValueError("source hybrid facts are invalid")
        raw_request = _request(
            arm="raw_prompt",
            ordinal=ordinal,
            item_key=item_key,
            ctext=ctext,
            facts=None,
            spec=spec,
            model=model,
        )
        hybrid_request = _request(
            arm="hybrid",
            ordinal=ordinal,
            item_key=item_key,
            ctext=ctext,
            facts=facts,
            spec=spec,
            model=model,
        )
        if raw_request["system_prompt"] != hybrid_request["system_prompt"]:
            raise AssertionError("matched system prompts differ")
        if raw_request["response_relations"] != hybrid_request["response_relations"]:
            raise AssertionError("matched response relations differ")
        raw_out.append(raw_request)
        hybrid_out.append(hybrid_request)
    return raw_out, hybrid_out


def prepare(source_dir: Path = SOURCE_PREPARATION, out_dir: Path = OUT) -> Path:
    if out_dir.exists():
        raise FileExistsError("refusing to overwrite matched prompt preparation")
    source_manifest = json.loads(
        (source_dir / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("execution_status", {}).get(
        "historical_reference_accessed"
    ) is not False:
        raise ValueError("source preparation is not reference-blind")
    spec = _load_spec()
    model = json.loads(MODEL_SPEC_PATH.read_text(encoding="utf-8"))
    raw_v1 = v1.read_jsonl(source_dir / "raw_prompt_requests.jsonl")
    hybrid_v1 = v1.read_jsonl(source_dir / "hybrid_seam_requests.jsonl")
    raw, hybrid = build_matched_arms(raw_v1, hybrid_v1, spec=spec, model=model)
    if len(raw) != 100 or len(hybrid) != 100:
        raise ValueError("matched arms must each contain 100 heldout requests")

    out_dir.mkdir(parents=True)
    spec_out = out_dir / "matched_prompt_spec.json"
    raw_out = out_dir / "raw_prompt_requests.jsonl"
    hybrid_out = out_dir / "hybrid_seam_requests.jsonl"
    addendum_out = out_dir / "design_addendum.json"
    report_out = out_dir / "REPORT.md"
    _write_readonly(spec_out, _json_bytes(spec))
    _write_readonly(raw_out, _jsonl_bytes(raw))
    _write_readonly(hybrid_out, _jsonl_bytes(hybrid))
    addendum = {
        "schema": "metric-seam.a407-matched-prompt-design-addendum.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "historical_reference_accessed": False,
        "historical_reference_values_available": False,
        "superseded_claim": (
            "The v1 raw/hybrid contrast does not isolate seam placement because its "
            "relation sets and prompt instructions differ."
        ),
        "registered_contrast": (
            "Same prompt program and output contract; CodeScope-v3 facts are null "
            "for raw and present for hybrid."
        ),
        "eligible_claim": (
            "Difference attributable to exposing structured facts under this prompt "
            "and representation, not correctness, full isomorphism, or code superiority."
        ),
        "raw_request_count": len(raw),
        "hybrid_request_count": len(hybrid),
        "model_calls": False,
        "api_calls": False,
        "gpu_used": False,
    }
    _write_readonly(addendum_out, _json_bytes(addendum))
    report = """# a407 matched prompt/hybrid preparation

Status: **prepared before historical-reference access; not executed**.

The original v1 arms changed both the structured evidence surface and the prompted
relation/output contract, so their difference cannot cleanly identify seam placement.
This additive contrast renders the articulated construct and all six relation definitions,
then holds that system prompt, response contract, sanitized ctext, model specification,
and user-object shape fixed. The only
prompt-visible intervention is `codescope_v3_facts`: null for raw and the deterministic
fact object for hybrid. Each arm contains 100 requests.

No historical judgement, API/model call, evaluation, or GPU was used. A later result may
estimate the effect of structured facts under this prompt program; it cannot by itself
establish correctness, full isomorphism, or code superiority.
"""
    _write_readonly(report_out, report.encode("utf-8"))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_preparation": str(source_dir.relative_to(ROOT)),
        "source_preparation_manifest_sha256": v1.hash_file(
            source_dir / "preparation_manifest.json"
        ),
        "system_prompt_identical_count": sum(
            left["system_prompt"] == right["system_prompt"]
            for left, right in zip(raw, hybrid)
        ),
        "response_contract_identical_count": sum(
            left["response_relations"] == right["response_relations"]
            for left, right in zip(raw, hybrid)
        ),
        "ctext_identical_count": sum(
            left["ctext_sha256"] == right["ctext_sha256"]
            for left, right in zip(raw, hybrid)
        ),
        "raw_fact_null_count": sum(
            row["codescope_v3_facts_present"] is False for row in raw
        ),
        "hybrid_fact_present_count": sum(
            row["codescope_v3_facts_present"] is True for row in hybrid
        ),
        "artifacts": {
            path.name: {"sha256": v1.hash_file(path), "bytes": path.stat().st_size}
            for path in (spec_out, raw_out, hybrid_out, addendum_out, report_out)
        },
        "execution": {
            "historical_reference_accessed": False,
            "evaluation_performed": False,
            "model_calls": False,
            "api_calls": False,
            "gpu_used": False,
        },
    }
    manifest_out = out_dir / "preparation_manifest.json"
    _write_readonly(manifest_out, _json_bytes(manifest))
    out_dir.chmod(0o555)
    return manifest_out


def main() -> int:
    manifest = prepare()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "ctext_identical_count": value["ctext_identical_count"],
                "hybrid_fact_present_count": value["hybrid_fact_present_count"],
                "raw_fact_null_count": value["raw_fact_null_count"],
                "response_contract_identical_count": value[
                    "response_contract_identical_count"
                ],
                "system_prompt_identical_count": value[
                    "system_prompt_identical_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
