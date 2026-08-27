#!/usr/bin/env python3
"""Hardened, additive science-v8 source-addressed articulability instrument.

V8 preserves v7's exact source-address surfaces while making three distinctions
machine-readable:

* semantic support/contradiction selected by the model is a
  ``prompt_asserted_relation_certificate`` (articulability);
* a corrected, deterministic relation parser is a separate, non-gating
  ``code_relation_audit`` (verifiability evidence local to that parser); and
* the historical full-paper code verifier is only a comparator.

Preparation and verification make no API or GPU calls.  The trusted loader must
deserialize each JSON source row, but its fresh projection indexes, retains, renders,
and emits only ``paper_id``, ``abstract``, and ``body``.  No label value is accessed by
the projection or admitted to a request.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from . import addressed_pipeline as v7
from . import core as frozen_code
from . import core_corrected as corrected_code


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/peer-review/peer_review_cv_evidence.jsonl"
DEFAULT_SPEC = Path(__file__).with_name("articulability_addressed_prompt_v8.json")
DEFAULT_MODEL = Path(__file__).with_name(
    "articulability_model_glm47_openrouter_addressed_v8.json"
)
DEFAULT_V7_BUNDLE = (
    ROOT / "outputs/metric_seam_pilot/science_articulability_v7_source_addressed_prepared"
)
DEFAULT_OUT = (
    ROOT / "outputs/metric_seam_pilot/science_articulability_v8_hardened_prepared"
)
DEFAULT_HISTORICAL_CODE_COMPARATOR = (
    ROOT / "outputs/metric_seam_pilot/science_claims_v2_corrected_v2/results.json"
)

REQUEST_SCHEMA = "science-articulability-addressed-request-v8"
RESULT_SCHEMA = "science-articulability-addressed-bound-result-v8"
NORMALIZED_SCHEMA = "science-articulability-addressed-normalized-result-v8"
MANIFEST_SCHEMA = "science-articulability-addressed-bundle-v8"
STRUCTURAL_ABSTENTION_SCHEMA = "science-articulability-structural-abstention-v8"
CROSSWALK_SCHEMA = "science-articulability-source-crosswalk-v8"
CODE_AUDIT_SCHEMA = "science-articulability-code-relation-audit-v8"
PROMPT_CERTIFICATE_TYPE = "prompt_asserted_relation_certificate"
CODE_AUDIT_STATUSES = {"verified", "diverged", "abstained"}

EXPECTED_CORPUS_COUNT = 2400
EXPECTED_ELIGIBLE_COUNT = 1957
EXPECTED_STRUCTURAL_ABSTENTION_COUNT = 443
EXPECTED_FIRST_FIVE_ELIGIBILITY = [True, True, True, False, True]

REQUESTS_NAME = "requests.jsonl"
ABSTENTIONS_NAME = "structural_abstentions.jsonl"
CROSSWALK_NAME = "source_crosswalk.jsonl"
REPORT_NAME = "PREPARATION_REPORT.md"

_REQUEST_MATERIAL_KEYS = (
    "source_index",
    "paper_input",
    "paper_input_sha256",
    "v7_request_id",
    "v7_request_sha256",
    "v7_segmentation_contract_sha256",
    "source_map",
    "source_map_sha256",
    "source_crosswalk_sha256",
    "prompt_spec_sha256",
    "model_manifest_sha256",
    "system_prompt",
    "user_prompt",
)


canonical_bytes = v7.canonical_bytes
hash_value = v7.hash_value
hash_file = v7.hash_file
display_path = v7.display_path


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _numbered_lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            yield line_number, line


def allowed_projection(raw: dict[str, Any], line_number: int) -> dict[str, str]:
    """Project a deserialized trusted row without indexing any non-allowlisted key."""

    return {
        "paper_id": str(raw.get("paper_id") or f"line_{line_number}"),
        "abstract": str(raw.get("abstract") or ""),
        "body": str(raw.get("body") or ""),
    }


def load_corpus(path: Path) -> list[dict[str, str]]:
    """Deserialize trusted rows, immediately retaining only the fresh projection."""

    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                # json.loads necessarily deserializes the complete trusted source row.
                # `allowed_projection` is the only consumer and never indexes labels.
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"source line {line_number} is not an object")
                records.append(allowed_projection(raw, line_number))
    ids = [row["paper_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("paper_id values must be unique")
    return records


def _validate_spec(spec: dict[str, Any]) -> None:
    v7._validate_spec(spec)
    if spec.get("semantic_output_type") != PROMPT_CERTIFICATE_TYPE:
        raise ValueError("v8 semantic output must be prompt_asserted_relation_certificate")
    notice = str(spec.get("non_gating_code_audit_notice") or "")
    if "never gates" not in notice:
        raise ValueError("v8 prompt must state that the code audit never gates")


def _validate_model(model: dict[str, Any]) -> None:
    v7._validate_model(model)
    if model.get("response_format") != "json_schema":
        raise ValueError("v8 transport requires json_schema response format")
    reasoning = model.get("reasoning")
    if reasoning != {"effort": "none"}:
        raise ValueError("v8 model identity must request no reasoning trace")
    if model.get("provider_require_parameters") is not True:
        raise ValueError("v8 OpenRouter transport must require declared parameters")
    if model.get("request_timeout_seconds") != 120 or model.get("max_attempts") != 2:
        raise ValueError("v8 bounded transport policy changed")


def render_system_prompt(spec: dict[str, Any]) -> str:
    _validate_spec(spec)
    return v7.render_system_prompt(spec) + (
        "\n\nV8_TYPE_GUARD: A supported or contradicted selection is a "
        "prompt_asserted_relation_certificate only. It is not executable verification. "
        "A separate non-gating code parser may later audit the hydrated source spans."
    )


def render_user_prompt(paper_id: str, source_map: dict[str, Any]) -> str:
    # Intentionally preserve the v7 addressed surface byte-for-byte.
    return v7.render_user_prompt(paper_id, source_map)


def corrected_segmenter_contract() -> dict[str, Any]:
    """Bind the frozen segmenter used by the corrected full-paper code verifier."""

    source = inspect.getsource(frozen_code.segment_sentences)
    return {
        "algorithm": "frozen_science_claims_v2.segment_sentences",
        "function_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "core_path": display_path(Path(frozen_code.__file__)),
        "core_sha256": hash_file(Path(frozen_code.__file__)),
        "corrected_wrapper_path": display_path(Path(corrected_code.__file__)),
        "corrected_wrapper_sha256": hash_file(Path(corrected_code.__file__)),
        "surface_policy": "whitespace-normalized sentence text with original offsets",
    }


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def crosswalk_span(
    span: dict[str, Any], corrected_sentences: list[Any]
) -> dict[str, Any]:
    """Crosswalk one exact v7 span without changing its ID, text, or offsets."""

    candidates = [
        sentence
        for sentence in corrected_sentences
        if _overlaps(span["start"], span["end"], sentence.start, sentence.end)
    ]
    exact = [
        sentence
        for sentence in candidates
        if sentence.start == span["start"] and sentence.end == span["end"]
    ]
    if len(exact) == 1:
        status = "matched"
    elif candidates:
        status = "ambiguous"
    else:
        status = "unmatched"
    return {
        "sentence_id": span["sentence_id"],
        "v7_start": span["start"],
        "v7_end": span["end"],
        "v7_text_sha256": span["text_sha256"],
        "status": status,
        "corrected_candidates": [
            {
                "sentence_index": sentence.index,
                "start": sentence.start,
                "end": sentence.end,
                "text_sha256": hash_value(sentence.text),
            }
            for sentence in candidates
        ],
    }


def build_source_crosswalk(
    source_index: int,
    paper: dict[str, str],
    source_map: dict[str, Any],
    v7_request: dict[str, Any],
) -> dict[str, Any]:
    abstract_code = corrected_code.segment_sentences(paper["abstract"])
    body_code = corrected_code.segment_sentences(paper["body"])
    return {
        "schema_version": CROSSWALK_SCHEMA,
        "source_index": source_index,
        "paper_id": paper["paper_id"],
        "v7_request_id": v7_request["request_id"],
        "v7_request_sha256": v7_request["request_sha256"],
        "v7_source_map_sha256": v7_request["source_map_sha256"],
        "abstract": [crosswalk_span(span, abstract_code) for span in source_map["abstract"]],
        "body": [crosswalk_span(span, body_code) for span in source_map["body"]],
    }


def _crosswalk_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for section in ("abstract", "body"):
            for span in row[section]:
                counts[span["status"]] += 1
    return {status: counts[status] for status in ("matched", "ambiguous", "unmatched")}


def build_request(
    source_index: int,
    paper: dict[str, str],
    v7_request: dict[str, Any],
    crosswalk: dict[str, Any],
    spec: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    if not paper["body"].strip():
        raise ValueError("missing-body rows are structural abstentions, not requests")
    source_map = v7.build_source_map(paper)
    if source_map != v7_request["source_map"]:
        raise ValueError("v8 must preserve the exact v7 source map")
    material = {
        "source_index": source_index,
        "paper_input": paper,
        "paper_input_sha256": hash_value(paper),
        "v7_request_id": v7_request["request_id"],
        "v7_request_sha256": v7_request["request_sha256"],
        "v7_segmentation_contract_sha256": v7_request[
            "segmentation_contract_sha256"
        ],
        "source_map": source_map,
        "source_map_sha256": hash_value(source_map),
        "source_crosswalk_sha256": hash_value(crosswalk),
        "prompt_spec_sha256": hash_value(spec),
        "model_manifest_sha256": hash_value(model),
        "system_prompt": render_system_prompt(spec),
        "user_prompt": render_user_prompt(paper["paper_id"], source_map),
    }
    request_sha = hash_value(material)
    return {
        "schema_version": REQUEST_SCHEMA,
        "request_id": f"science_v8_addressed_{source_index:04d}_{request_sha[:16]}",
        "paper_id": paper["paper_id"],
        **material,
        "request_sha256": request_sha,
    }


def build_structural_abstention(
    source_index: int,
    paper: dict[str, str],
    v7_request: dict[str, Any],
    crosswalk: dict[str, Any],
) -> dict[str, Any]:
    if paper["body"].strip():
        raise ValueError("only missing-body rows may enter the structural ledger")
    material = {
        "source_index": source_index,
        "paper_id": paper["paper_id"],
        "paper_input_sha256": hash_value(paper),
        "source_map_sha256": v7_request["source_map_sha256"],
        "source_crosswalk_sha256": hash_value(crosswalk),
        "v7_request_id": v7_request["request_id"],
        "v7_request_sha256": v7_request["request_sha256"],
        "prompt_eligible": False,
        "status": "structural_abstention",
        "reason": "missing_fullpaper_body",
        "api_call_required": False,
    }
    return {
        "schema_version": STRUCTURAL_ABSTENTION_SCHEMA,
        **material,
        "abstention_sha256": hash_value(material),
    }


def build_artifacts(
    records: list[dict[str, str]],
    spec: dict[str, Any],
    model: dict[str, Any],
    v7_requests: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    crosswalks: list[dict[str, Any]] = []
    for source_index, paper in enumerate(records):
        v7_request = v7_requests[source_index]
        v7_paper = v7_request.get("paper_input")
        if v7_paper != paper:
            raise ValueError(
                f"exact v7 paper_input/current projection drift at source index {source_index}"
            )
        if v7_request.get("paper_input_sha256") != hash_value(paper):
            raise ValueError(
                f"v7 paper_input hash/current projection drift at source index {source_index}"
            )
        source_map = v7.build_source_map(paper)
        if source_map != v7_request["source_map"]:
            raise ValueError(f"v7 source-map drift at source index {source_index}")
        crosswalk = build_source_crosswalk(
            source_index, paper, source_map, v7_request
        )
        crosswalks.append(crosswalk)
        if paper["body"].strip():
            requests.append(
                build_request(source_index, paper, v7_request, crosswalk, spec, model)
            )
        else:
            abstentions.append(
                build_structural_abstention(
                    source_index, paper, v7_request, crosswalk
                )
            )
    return requests, abstentions, crosswalks


def _quantity_audit(claim_text: str, evidence_text: str) -> dict[str, Any]:
    claims = corrected_code.extract_quantities(claim_text)
    evidence = corrected_code.extract_quantities(evidence_text)
    if not claims and not evidence:
        parser_relation_state = "abstained"
        reason = "no_quantity_parsed_on_either_side"
        matched = 0
    elif not claims or not evidence:
        parser_relation_state = "mismatch"
        reason = "quantity_present_on_only_one_side"
        matched = 0
    else:
        matched = sum(
            any(
                corrected_code.quantity_relation_equal(
                    claim_text, claim_quantity, evidence_text, evidence_quantity
                )
                for evidence_quantity in evidence
            )
            for claim_quantity in claims
        )
        parser_relation_state = "aligned" if matched == len(claims) else "mismatch"
        reason = (
            "all_claim_quantities_units_and_entity_bindings_match"
            if parser_relation_state == "aligned"
            else "claim_quantity_unit_or_entity_binding_diverged"
        )
    return {
        "parser_relation_state": parser_relation_state,
        "reason": reason,
        "claim_quantity_count": len(claims),
        "evidence_quantity_count": len(evidence),
        "matched_claim_quantity_count": matched,
    }


def _comparison_audit(claim_text: str, evidence_text: str) -> dict[str, Any]:
    claim = corrected_code.extract_comparison(claim_text)
    evidence = corrected_code.extract_comparison(evidence_text)
    relation_state = frozen_code._comparison_state(claim, evidence)
    if claim is None or evidence is None or relation_state == "missing":
        parser_availability = "abstained"
    elif relation_state in {"aligned", "aligned_reversed"}:
        parser_availability = "parsed"
    else:
        parser_availability = "parsed"
    return {
        "parser_availability": parser_availability,
        "parser_relation_state": relation_state,
        "claim_comparison_present": claim is not None,
        "evidence_comparison_present": evidence is not None,
    }


def code_relation_audit(
    selection: dict[str, Any], claim_text: str, evidence_text: str | None
) -> dict[str, Any]:
    """Audit a prompt assertion without accepting/rejecting the prompt response."""

    relation = selection["relation"]
    decision = selection["decision"]
    quantity = None
    comparison = None
    status = "abstained"
    reason = "prompt_selection_is_not_a_numeric_or_comparative_assertion"
    if evidence_text is None:
        reason = "no_hydrated_body_evidence"
    elif decision not in {"supported", "contradicted"}:
        reason = "prompt_decision_does_not_assert_support_or_contradiction"
    elif relation == "numeric":
        quantity = _quantity_audit(claim_text, evidence_text)
        parser_state = quantity["parser_relation_state"]
        expected_state = "aligned" if decision == "supported" else "mismatch"
        if parser_state == "abstained":
            status = "abstained"
            reason = "numeric_parser_abstained"
        elif parser_state == expected_state:
            status = "verified"
            reason = "numeric_parser_agrees_with_prompt_decision"
        else:
            status = "diverged"
            reason = "numeric_parser_diverges_from_prompt_decision"
    elif relation == "comparative":
        comparison = _comparison_audit(claim_text, evidence_text)
        # Quantity applicability is derived only from the hydrated source spans.  The
        # prompt's quantity_count/state are retained as articulability output, never as
        # the code audit's execution switch or count.
        quantity = _quantity_audit(claim_text, evidence_text)
        state = comparison["parser_relation_state"]
        expected = (
            {"reversed_roles", "direction_mismatch"}
            if decision == "contradicted"
            else {"aligned", "aligned_reversed"}
        )
        if comparison["parser_availability"] == "abstained":
            status = "abstained"
            reason = "comparison_parser_missing_relation"
        elif state in expected:
            status = "verified"
            reason = "comparison_parser_agrees_with_prompt_assertion"
        else:
            status = "diverged"
            reason = "comparison_parser_diverges_from_prompt_assertion"
        quantity_state = quantity["parser_relation_state"]
        quantity_present = (
            quantity["claim_quantity_count"] > 0
            or quantity["evidence_quantity_count"] > 0
        )
        # A supported comparison that carries quantities in either source span must
        # reproduce those quantities/units/entity bindings.  Prompt `not_required/0`
        # cannot suppress this executable obligation.  Comparative contradiction is
        # licensed by parsed role/direction reversal under the current response contract;
        # quantity state is still reported but does not create that contradiction.
        if decision == "supported" and quantity_present:
            if quantity_state != "aligned":
                status = "diverged"
                reason = "comparison_source_quantity_relation_diverges"
    if status not in CODE_AUDIT_STATUSES:
        raise AssertionError("invalid code audit status")
    return {
        "schema_version": CODE_AUDIT_SCHEMA,
        "status": status,
        "reason": reason,
        "non_gating": True,
        "prompt_decision": decision,
        "prompt_relation": relation,
        "prompt_audit_status": status,
        "quantity_parser": quantity,
        "comparison_parser": comparison,
        "implementation": {
            "core_corrected_sha256": hash_file(Path(corrected_code.__file__)),
            "core_sha256": hash_file(Path(frozen_code.__file__)),
        },
    }


def validate_response(response: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    # Reuse the audited v7 exact schema/address/coherence validator.  V8 request rows
    # deliberately retain all fields it consumes.
    return v7.validate_response(response, request)


def hydrate_response(response: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    response = validate_response(response, request)
    abstract, body = v7._address_index(request)
    matches: list[dict[str, Any]] = []
    for selection in response["selections"]:
        claim_span = abstract[selection["claim_sentence_id"]]
        evidence_id = selection["evidence_sentence_id"]
        evidence_span = body[evidence_id] if evidence_id is not None else None
        decision = selection["decision"]
        if decision in {"supported", "contradicted"}:
            witness_kind = PROMPT_CERTIFICATE_TYPE
        elif decision == "evidence_link":
            witness_kind = "prompt_asserted_evidence_link"
        else:
            witness_kind = "none"
        audit = code_relation_audit(
            selection,
            claim_span["text"],
            evidence_span["text"] if evidence_span is not None else None,
        )
        hybrid_witness_kind = (
            "prompt_selected_code_confirmed_hybrid_witness"
            if witness_kind == PROMPT_CERTIFICATE_TYPE
            and audit["status"] == "verified"
            else None
        )
        matches.append({
            "decision": decision,
            "witness_kind": witness_kind,
            "semantic_channel": "prompt_articulability",
            "hybrid_witness_kind": hybrid_witness_kind,
            "reason": "prompt_source_addressed_relation_judgment",
            "claim": v7._hydrated_span(claim_span, relation=selection["relation"]),
            "evidence": (
                v7._hydrated_span(evidence_span) if evidence_span is not None else None
            ),
            "prompt_typed_assertion": {
                key: selection[key]
                for key in (
                    "quantity_state", "comparison_state", "evidence_kind",
                    "quantity_count", "comparison_present",
                )
            },
            "source_addresses": {
                "claim": selection["claim_sentence_id"],
                "evidence": evidence_id,
            },
            "code_relation_audit": audit,
        })
    decisions = [match["decision"] for match in matches]
    prompt_certificates = [
        match for match in matches if match["witness_kind"] == PROMPT_CERTIFICATE_TYPE
    ]
    evidence_links = [
        match
        for match in matches
        if match["witness_kind"] == "prompt_asserted_evidence_link"
    ]
    audit_counts = Counter(match["code_relation_audit"]["status"] for match in matches)
    hybrid_witnesses = [
        match for match in matches if match["hybrid_witness_kind"] is not None
    ]
    return {
        "paper_id": request["paper_id"],
        "status": v7._derived_status(decisions),
        "reason": "derived_from_validated_prompt_assertions",
        "claim_count": len(matches),
        "prompt_asserted_relation_certificate_count": len(prompt_certificates),
        "prompt_asserted_evidence_link_count": len(evidence_links),
        "prompt_asserted_relation_certificates": prompt_certificates,
        "prompt_asserted_evidence_links": evidence_links,
        "prompt_selected_code_confirmed_hybrid_witness_count": len(
            hybrid_witnesses
        ),
        "prompt_selected_code_confirmed_hybrid_witnesses": hybrid_witnesses,
        "matches": matches,
        "code_relation_audit_counts": {
            status: audit_counts[status]
            for status in ("verified", "diverged", "abstained")
        },
        "seam": {
            "articulability_channel": "prompt_assertions",
            "verifiability_channel": "non_gating_code_relation_audit",
            "code_audit_can_reject_prompt_response": False,
        },
        "transport": {
            "model_returned_source_text": False,
            "source_hydration": "deterministic_exact_v7_bound_spans",
            "status_and_counts": "derived_in_code",
        },
    }


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _v7_requests_by_source_index(
    v7_bundle: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest, requests = v7.verify_bundle(v7_bundle)
    rows = sorted(requests.values(), key=lambda row: row["sequence_index"])
    by_index = {row["sequence_index"]: row for row in rows}
    if set(by_index) != set(range(len(rows))):
        raise ValueError("v7 request sequences are not contiguous")
    return manifest, by_index


def implementation_bindings() -> dict[str, dict[str, str]]:
    files = {
        "v8_pipeline": Path(__file__),
        "v8_runner": Path(__file__).with_name("addressed_runner_v8.py"),
        "v8_evaluator": Path(__file__).with_name("evaluate_addressed_v8.py"),
        "v7_pipeline": Path(v7.__file__),
        "corrected_code": Path(corrected_code.__file__),
        "frozen_code_segmenter": Path(frozen_code.__file__),
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"v8 implementation dependency missing: {missing}")
    return {
        key: {"path": display_path(path), "sha256": hash_file(path)}
        for key, path in files.items()
    }


def historical_comparator_binding(
    path: Path, *, expected_input_source_sha256: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise ValueError("historical code comparator lacks a schema_version")
    comparator_input = payload.get("input")
    if not isinstance(comparator_input, dict):
        raise ValueError("historical code comparator lacks an input binding")
    comparator_source_sha = comparator_input.get("sha256")
    comparator_source_path = comparator_input.get("path")
    if comparator_source_sha != expected_input_source_sha256:
        raise ValueError(
            "historical comparator payload input SHA differs from v8 source SHA"
        )
    if not isinstance(comparator_source_path, str) or not comparator_source_path:
        raise ValueError("historical comparator input binding lacks a source path")
    return {
        "path": display_path(path),
        "sha256": hash_file(path),
        "schema_version": schema,
        "input_source_path": comparator_source_path,
        "input_source_sha256": comparator_source_sha,
        "source_artifact_provenance": payload.get("provenance"),
        "source_artifact_pipeline_status": payload.get("pipeline_status"),
        "v8_analysis_pipeline_status": "selected",
        "selection_mode": "retrospective_seed",
        "original_decomposition_discovery": "manual_historical",
        "automatically_discovered_by_v8": False,
    }


def _stats(
    records: list[dict[str, str]],
    requests: list[dict[str, Any]],
    abstentions: list[dict[str, Any]],
    crosswalks: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_lengths = [
        len(row["system_prompt"]) + len(row["user_prompt"]) for row in requests
    ]
    abstract_counts = [len(row["source_map"]["abstract"]) for row in requests]
    body_counts = [len(row["source_map"]["body"]) for row in requests]
    corpus_abstract_spans = sum(len(row["abstract"]) for row in crosswalks)
    corpus_body_spans = sum(len(row["body"]) for row in crosswalks)
    return {
        "corpus_records": len(records),
        "body_present_prompt_eligible": len(requests),
        "missing_body_structural_abstentions": len(abstentions),
        "prompt_characters": {
            "min": min(prompt_lengths, default=0),
            "max": max(prompt_lengths, default=0),
            "total": sum(prompt_lengths),
        },
        "eligible_address_counts": {
            "abstract_total": sum(abstract_counts),
            "body_total": sum(body_counts),
            "abstract_min": min(abstract_counts, default=0),
            "abstract_max": max(abstract_counts, default=0),
            "body_min": min(body_counts, default=0),
            "body_max": max(body_counts, default=0),
        },
        "crosswalk_span_counts": _crosswalk_counts(crosswalks),
        "corpus_address_counts": {
            "abstract": corpus_abstract_spans,
            "body": corpus_body_spans,
            "total": corpus_abstract_spans + corpus_body_spans,
        },
    }


def _first_five_crosswalk(
    records: list[dict[str, str]],
    requests: list[dict[str, Any]],
    abstentions: list[dict[str, Any]],
    v7_requests: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    request_by_index = {row["source_index"]: row for row in requests}
    abstention_by_index = {row["source_index"]: row for row in abstentions}
    rows = []
    for index, paper in enumerate(records[:5]):
        eligible = bool(paper["body"].strip())
        rows.append({
            "source_index": index,
            "paper_id": paper["paper_id"],
            "v7_request_id": v7_requests[index]["request_id"],
            "body_present_prompt_eligible": eligible,
            "v8_request_id": (
                request_by_index[index]["request_id"] if eligible else None
            ),
            "v8_structural_abstention_sha256": (
                None
                if eligible
                else abstention_by_index[index]["abstention_sha256"]
            ),
        })
    return rows


def build_manifest(
    *,
    input_path: Path,
    spec_path: Path,
    model_path: Path,
    v7_bundle: Path,
    historical_comparator_path: Path,
    v7_manifest: dict[str, Any],
    records: list[dict[str, str]],
    requests: list[dict[str, Any]],
    abstentions: list[dict[str, Any]],
    crosswalks: list[dict[str, Any]],
    v7_requests: dict[int, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    stats = _stats(records, requests, abstentions, crosswalks)
    production = input_path.resolve() == DEFAULT_INPUT.resolve()
    count_contract = {
        "mode": "production_exact" if production else "test_fixture_derived",
        "expected_corpus_records": EXPECTED_CORPUS_COUNT if production else len(records),
        "expected_body_present_prompt_eligible": (
            EXPECTED_ELIGIBLE_COUNT if production else len(requests)
        ),
        "expected_missing_body_structural_abstentions": (
            EXPECTED_STRUCTURAL_ABSTENTION_COUNT if production else len(abstentions)
        ),
    }
    first_five = _first_five_crosswalk(
        records, requests, abstentions, v7_requests
    )
    if production and [
        row["body_present_prompt_eligible"] for row in first_five
    ] != EXPECTED_FIRST_FIVE_ELIGIBILITY:
        raise ValueError("production first-five paired-smoke eligibility drift")
    corpus_sha = hash_value([hash_value(record) for record in records])
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "prepared_not_run_no_api_calls",
        "objective": (
            "unsupervised_prompt_articulability_same_evidence_content_source_addressed"
        ),
        "axes": {
            "articulability": "prompt_based",
            "verifiability": "code_based",
            "isomorphism": "separate_reconstruction_comparison",
        },
        "execution_policy": {
            "api_calls_made_by_prepare": 0,
            "physical_model_attempts_made_by_prepare": 0,
            "gpu_used": False,
            "external_supervision": "none",
            "external_scientific_knowledge": "forbidden",
            "future_runner_is_serial_ready": True,
        },
        "label_policy": {
            "trusted_loader_deserializes_full_source_row": True,
            "projection_indexes_only": ["paper_id", "abstract", "body"],
            "projection_retains_only": ["paper_id", "abstract", "body"],
            "projection_renders_only": ["paper_id", "abstract", "body"],
            "projection_emits_only": ["paper_id", "abstract", "body"],
            "label_values_indexed_retained_rendered_or_emitted": False,
        },
        "input": {
            "path": display_path(input_path),
            "source_file_sha256": hash_file(input_path),
            "allowlisted_corpus_sha256": corpus_sha,
            "record_count": len(records),
        },
        "v7_checkpoint": {
            "bundle_path": display_path(v7_bundle),
            "manifest_sha256": hash_file(v7_bundle / "manifest.json"),
            "requests_sha256": hash_file(v7_bundle / REQUESTS_NAME),
            "manifest_schema": v7_manifest["schema_version"],
            "request_count": v7_manifest["requests"]["count"],
            "segmentation_contract_sha256": v7_manifest[
                "segmentation_contract"
            ]["canonical_sha256"],
        },
        "prompt_spec": {
            "path": display_path(spec_path),
            "file_sha256": hash_file(spec_path),
            "canonical_sha256": hash_value(spec),
            "identity": spec,
        },
        "model_manifest": {
            "path": display_path(model_path),
            "file_sha256": hash_file(model_path),
            "canonical_sha256": hash_value(model),
            "identity": model,
        },
        "implementation_dependencies": implementation_bindings(),
        "historical_code_comparator": historical_comparator_binding(
            historical_comparator_path,
            expected_input_source_sha256=hash_file(input_path),
        ),
        "segmentation": {
            "prompt_surface": v7_manifest["segmentation_contract"],
            "corrected_code_segmenter": corrected_segmenter_contract(),
            "surface_policy": (
                "A####/B#### IDs, exact texts, and exact offsets are unchanged from v7; "
                "the corrected-code crosswalk never alters evidence shown to the prompt"
            ),
            "crosswalk_status_semantics": {
                "matched": "exactly one corrected-code span has identical offsets",
                "ambiguous": (
                    "one or more corrected-code spans overlap, but none has identical offsets"
                ),
                "unmatched": "no corrected-code span overlaps the exact v7 span",
            },
        },
        "isomorphism_scope": {
            "same_evidence_content": True,
            "same_input_representation": False,
            "input_representation_fidelity": "partial_not_estimable_from_this_bundle",
            "semantic_reconstruction_comparison_estimable_after_execution": True,
            "full_isomorphism_licensed": False,
            "reason": (
                "The prompt sees addressed JSONL with IDs, JSON escaping, and omitted "
                "inter-sentence whitespace; historical code sees continuous abstract/body."
            ),
        },
        "strata": {
            "count_contract": count_contract,
            "observed": {
                "corpus_records": stats["corpus_records"],
                "body_present_prompt_eligible": stats[
                    "body_present_prompt_eligible"
                ],
                "missing_body_structural_abstentions": stats[
                    "missing_body_structural_abstentions"
                ],
            },
            "wording_guard": (
                "Only the body-present stratum is full-paper prompt-eligible; the "
                "2,400-record corpus must not be described as 2,400 full-paper prompts"
            ),
            "first_five_paired_smoke_source_index_crosswalk": first_five,
        },
        "files": {
            "requests": {
                "path": REQUESTS_NAME,
                "sha256": hash_file(output_dir / REQUESTS_NAME),
                "count": len(requests),
                "future_remote_eligible_calls": len(requests),
            },
            "structural_abstentions": {
                "path": ABSTENTIONS_NAME,
                "sha256": hash_file(output_dir / ABSTENTIONS_NAME),
                "count": len(abstentions),
                "future_remote_calls": 0,
            },
            "source_crosswalk": {
                "path": CROSSWALK_NAME,
                "sha256": hash_file(output_dir / CROSSWALK_NAME),
                "count": len(crosswalks),
                "span_status_counts": stats["crosswalk_span_counts"],
                "corpus_address_counts": stats["corpus_address_counts"],
            },
            "preparation_report": {
                "path": REPORT_NAME,
                "verification": "exact deterministic render from this manifest",
            },
        },
        "request_statistics": {
            "prompt_characters": stats["prompt_characters"],
            "eligible_address_counts": stats["eligible_address_counts"],
        },
        "semantic_contract": {
            "primary_prompt_output": PROMPT_CERTIFICATE_TYPE,
            "contradiction_scope": {
                "accepted_response_relation": "comparative_only",
                "numeric_contradiction_symmetry": (
                    "primitive_only_future_schema_guard_not_reachable_in_v8_responses"
                ),
            },
            "code_relation_audit": {
                "statuses": ["verified", "diverged", "abstained"],
                "non_gating": True,
                "can_reject_schema_valid_prompt_response": False,
                "bound_parsers": [
                    "complete-token quantity/unit parsing",
                    "local quantity-entity binding",
                    "comparison direction/role parsing",
                ],
            },
            "historical_code_verifier_role": "comparison_only",
            "verified_selected_instance_output": {
                "type": "prompt_selected_code_confirmed_hybrid_witness",
                "scope": "document_local_relation_local_parser_scoped",
                "external_scientific_truth": False,
                "effect_on_prompt_acceptance": "none_non_gating",
            },
            "external_truth_claim": "none",
        },
        "result_contract": {
            "schema_version": RESULT_SCHEMA,
            "normalized_schema_version": NORMALIZED_SCHEMA,
            "required_transport_bindings": [
                "request_id", "request_sha256", "model_manifest_sha256",
                "bundle_manifest_sha256", "runner_sha256",
                "api_payload_sha256",
            ],
            "resume_conflict_policy": (
                "an existing normalized row must match the complete bound transport "
                "result/hash (including retry telemetry), parsed response hash, and "
                "deterministic hydration plus non-gating code audit exactly"
            ),
            "physical_attempt_count_includes_retries": True,
            "normalized_resume_replays_full_bound_transport_result": True,
        },
    }


def render_preparation_report(manifest: dict[str, Any]) -> str:
    observed = manifest["strata"]["observed"]
    crosswalk = manifest["files"]["source_crosswalk"]["span_status_counts"]
    first_five = manifest["strata"]["first_five_paired_smoke_source_index_crosswalk"]
    first_five_text = ", ".join(
        f"{row['source_index']}={'prompt' if row['body_present_prompt_eligible'] else 'missing-body-control'}"
        for row in first_five
    )
    return f"""# Science articulability v8 — hardened source-addressed preparation

Status: **prepared, not run**. Preparation made **0 API calls**, **0 physical model
attempts**, and used **no GPU**. The objective remains unsupervised reconstruction.

## Explicit corpus strata

- Body-present, full-paper prompt-eligible: **{observed['body_present_prompt_eligible']:,}**
- Missing-body deterministic structural abstentions: **{observed['missing_body_structural_abstentions']:,}**
- Total corpus accounting: **{observed['corpus_records']:,}**

Only the first stratum is full-paper prompt-eligible. This report does not describe all
{observed['corpus_records']:,} records as full-paper prompts. The paired first-five smoke
crosswalk is `{first_five_text}` (four prompt-eligible rows and one missing-body control in
the production corpus).

## Typed seam

The primary semantic output is explicitly
`prompt_asserted_relation_certificate`: an articulability result produced by a prompt.
After code hydrates exact v7 A####/B#### source spans, the corrected quantity/unit/entity
and comparison direction/role parsers produce a separate `code_relation_audit` with
`verified`, `diverged`, or `abstained`. The audit is **non-gating**: parser disagreement
does not reject a schema-valid prompt response and is not external scientific truth.
The audit primitive is decision-symmetric for numeric relations, but numeric contradiction
is a future-schema guard only: the current v8 response contract accepts contradiction for
comparative relations only.

## Exact-source and corrected-segmenter crosswalk

V8 preserves every v7 address, exact text, and offset. A deterministic crosswalk to the
frozen corrected-code segmenter covers every span: matched={crosswalk['matched']:,},
ambiguous={crosswalk['ambiguous']:,}, unmatched={crosswalk['unmatched']:,}. Ambiguity is
reported rather than changing the evidence surface. Both segmenter implementations and
the crosswalk file are SHA-bound in `manifest.json`.

The prepared bundle contains {observed['body_present_prompt_eligible']:,} remote-eligible
requests and a separate {observed['missing_body_structural_abstentions']:,}-row abstention
ledger. `addressed_runner_v8.py` is serial-ready but was not invoked.

The historical code comparator and v8 prompt receive the same evidence content, but not
the same input representation: addressed JSONL adds IDs/escaping and changes whitespace
layout relative to continuous abstract/body. A future semantic reconstruction comparison
therefore cannot by itself license full input-representation isomorphism.
"""


def _assert_count_contract(manifest: dict[str, Any]) -> None:
    expected = manifest["strata"]["count_contract"]
    observed = manifest["strata"]["observed"]
    mapping = {
        "corpus_records": "expected_corpus_records",
        "body_present_prompt_eligible": "expected_body_present_prompt_eligible",
        "missing_body_structural_abstentions": (
            "expected_missing_body_structural_abstentions"
        ),
    }
    for observed_key, expected_key in mapping.items():
        if observed[observed_key] != expected[expected_key]:
            raise ValueError(f"stratum count contract mismatch: {observed_key}")


def prepare(
    input_path: Path,
    spec_path: Path,
    model_path: Path,
    v7_bundle: Path,
    output_dir: Path,
    historical_comparator_path: Path = DEFAULT_HISTORICAL_CODE_COMPARATOR,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty v8 bundle: {output_dir}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    _validate_spec(spec)
    _validate_model(model)
    records = load_corpus(input_path)
    v7_manifest, v7_requests = _v7_requests_by_source_index(v7_bundle)
    if len(v7_requests) != len(records):
        raise ValueError("v7 checkpoint and v8 source corpus counts differ")
    requests, abstentions, crosswalks = build_artifacts(
        records, spec, model, v7_requests
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / REQUESTS_NAME, requests)
    _write_jsonl(output_dir / ABSTENTIONS_NAME, abstentions)
    _write_jsonl(output_dir / CROSSWALK_NAME, crosswalks)
    manifest = build_manifest(
        input_path=input_path,
        spec_path=spec_path,
        model_path=model_path,
        v7_bundle=v7_bundle,
        historical_comparator_path=historical_comparator_path,
        v7_manifest=v7_manifest,
        records=records,
        requests=requests,
        abstentions=abstentions,
        crosswalks=crosswalks,
        v7_requests=v7_requests,
        output_dir=output_dir,
    )
    _assert_count_contract(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_NAME).write_text(
        render_preparation_report(manifest), encoding="utf-8"
    )
    return manifest


def verify_bundle(
    bundle: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    """Recompute every manifest field and every prepared artifact exactly."""

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported science v8 bundle schema")
    if manifest.get("status") != "prepared_not_run_no_api_calls":
        raise ValueError("v8 bundle is not prepared-not-run")
    if manifest.get("objective") != (
        "unsupervised_prompt_articulability_same_evidence_content_source_addressed"
    ):
        raise ValueError("v8 objective changed")
    if manifest.get("execution_policy") != {
        "api_calls_made_by_prepare": 0,
        "physical_model_attempts_made_by_prepare": 0,
        "gpu_used": False,
        "external_supervision": "none",
        "external_scientific_knowledge": "forbidden",
        "future_runner_is_serial_ready": True,
    }:
        raise ValueError("v8 execution policy/status/API/GPU contract changed")

    input_path = _resolve_recorded_path(manifest["input"]["path"])
    spec_path = _resolve_recorded_path(manifest["prompt_spec"]["path"])
    model_path = _resolve_recorded_path(manifest["model_manifest"]["path"])
    v7_bundle = _resolve_recorded_path(manifest["v7_checkpoint"]["bundle_path"])
    historical_comparator_path = _resolve_recorded_path(
        manifest["historical_code_comparator"]["path"]
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    _validate_spec(spec)
    _validate_model(model)
    records = load_corpus(input_path)
    v7_manifest, v7_requests = _v7_requests_by_source_index(v7_bundle)
    regenerated = build_artifacts(records, spec, model, v7_requests)
    regenerated_requests, regenerated_abstentions, regenerated_crosswalks = regenerated

    actual_requests = _read_jsonl(bundle / REQUESTS_NAME)
    actual_abstentions = _read_jsonl(bundle / ABSTENTIONS_NAME)
    actual_crosswalks = _read_jsonl(bundle / CROSSWALK_NAME)
    if actual_requests != regenerated_requests:
        raise ValueError("v8 request rows differ from deterministic regeneration")
    if actual_abstentions != regenerated_abstentions:
        raise ValueError("v8 structural abstention ledger differs from regeneration")
    if actual_crosswalks != regenerated_crosswalks:
        raise ValueError("v8 source crosswalk differs from deterministic regeneration")

    expected_manifest = build_manifest(
        input_path=input_path,
        spec_path=spec_path,
        model_path=model_path,
        v7_bundle=v7_bundle,
        historical_comparator_path=historical_comparator_path,
        v7_manifest=v7_manifest,
        records=records,
        requests=regenerated_requests,
        abstentions=regenerated_abstentions,
        crosswalks=regenerated_crosswalks,
        v7_requests=v7_requests,
        output_dir=bundle,
    )
    _assert_count_contract(expected_manifest)
    if manifest != expected_manifest:
        differing = sorted(
            key
            for key in set(manifest) | set(expected_manifest)
            if manifest.get(key) != expected_manifest.get(key)
        )
        raise ValueError(f"v8 manifest full recomputation mismatch: {differing}")
    expected_report = render_preparation_report(expected_manifest)
    if (bundle / REPORT_NAME).read_text(encoding="utf-8") != expected_report:
        raise ValueError("v8 preparation report differs from deterministic render")

    request_by_id: dict[str, dict[str, Any]] = {}
    for row in actual_requests:
        material = {key: row.get(key) for key in _REQUEST_MATERIAL_KEYS}
        if hash_value(material) != row.get("request_sha256"):
            raise ValueError("v8 request material hash mismatch")
        expected_id = (
            f"science_v8_addressed_{row['source_index']:04d}_"
            f"{row['request_sha256'][:16]}"
        )
        if row.get("request_id") != expected_id:
            raise ValueError("v8 request id is not source-index/material derived")
        if expected_id in request_by_id:
            raise ValueError("duplicate v8 request id")
        request_by_id[expected_id] = row
    abstention_by_index = {row["source_index"]: row for row in actual_abstentions}
    if len(abstention_by_index) != len(actual_abstentions):
        raise ValueError("duplicate source index in structural abstention ledger")
    return manifest, request_by_id, abstention_by_index


def _extract_json(response: Any) -> dict[str, Any]:
    return v7._extract_json(response)


def _runner_module():
    from . import addressed_runner_v8

    return addressed_runner_v8


_BOUND_RESULT_KEYS = {
    "schema_version", "request_id", "request_sha256",
    "model_manifest_sha256", "bundle_manifest_sha256", "runner_sha256",
    "api_payload_sha256", "provider", "model", "response",
    "parsed_response_sha256", "telemetry",
}


def verify_bound_result(
    raw: dict[str, Any],
    *,
    request: dict[str, Any],
    manifest: dict[str, Any],
    bundle_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one runner result and deterministically replay hydration/audit."""

    if set(raw) != _BOUND_RESULT_KEYS:
        raise ValueError("bound result keys differ from the exact v8 result schema")
    runner = _runner_module()
    runner_sha = hash_file(Path(runner.__file__))
    payload = runner.api_payload_for_request(request, manifest["model_manifest"]["identity"])
    expected = {
        "schema_version": RESULT_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "model_manifest_sha256": manifest["model_manifest"]["canonical_sha256"],
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "runner_sha256": runner_sha,
        "api_payload_sha256": hash_value(payload),
        "provider": manifest["model_manifest"]["identity"]["backend"],
        "model": manifest["model_manifest"]["identity"]["model"],
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"v8 bound result binding mismatch: {key}")
    telemetry = raw.get("telemetry")
    runner.validate_telemetry(
        telemetry,
        max_attempts=manifest["model_manifest"]["identity"]["max_attempts"],
    )
    parsed = _extract_json(raw.get("response"))
    if raw.get("parsed_response_sha256") != hash_value(parsed):
        raise ValueError("v8 parsed response hash mismatch")
    hydrated = hydrate_response(parsed, request)
    return parsed, hydrated


_NORMALIZED_KEYS = {
    "schema_version", "request_id", "request_sha256", "paper_input_sha256",
    "source_map_sha256", "source_crosswalk_sha256", "prompt_spec_sha256",
    "model_manifest_sha256", "bundle_manifest_sha256", "runner_sha256",
    "api_payload_sha256", "transport_result_sha256", "bound_transport_result",
    "parsed_response_sha256", "validated_response", "result_sha256", "result",
}


def _verify_normalized_row(
    row: dict[str, Any],
    *,
    request: dict[str, Any],
    manifest: dict[str, Any],
    bundle_manifest_sha256: str,
) -> None:
    if not isinstance(row, dict) or set(row) != _NORMALIZED_KEYS:
        raise ValueError("v8 normalized row keys differ from the exact contract")
    expected = {
        "schema_version": NORMALIZED_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "paper_input_sha256": request["paper_input_sha256"],
        "source_map_sha256": request["source_map_sha256"],
        "source_crosswalk_sha256": request["source_crosswalk_sha256"],
        "prompt_spec_sha256": request["prompt_spec_sha256"],
        "model_manifest_sha256": manifest["model_manifest"]["canonical_sha256"],
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "runner_sha256": hash_file(Path(_runner_module().__file__)),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"v8 normalized resume binding mismatch: {key}")
    bound = row.get("bound_transport_result")
    if not isinstance(bound, dict):
        raise ValueError("v8 normalized row lacks its bound transport result")
    if row.get("transport_result_sha256") != hash_value(bound):
        raise ValueError("v8 normalized transport result hash mismatch")
    bound_response, bound_result = verify_bound_result(
        bound,
        request=request,
        manifest=manifest,
        bundle_manifest_sha256=bundle_manifest_sha256,
    )
    if row.get("api_payload_sha256") != bound.get("api_payload_sha256"):
        raise ValueError("v8 normalized API payload binding mismatch")
    response = row.get("validated_response")
    if not isinstance(response, dict):
        raise ValueError("v8 normalized row lacks validated_response")
    if row.get("parsed_response_sha256") != hash_value(response):
        raise ValueError("v8 normalized parsed response hash mismatch")
    if response != bound_response:
        raise ValueError("v8 normalized response differs from bound transport response")
    replay = hydrate_response(response, request)
    if replay != bound_result:
        raise ValueError("v8 normalized bound transport replay mismatch")
    if row.get("result") != replay:
        raise ValueError("v8 normalized hydration/code-audit replay mismatch")
    if row.get("result_sha256") != hash_value(replay):
        raise ValueError("v8 normalized result hash mismatch")


def ingest(
    bundle: Path,
    raw_results_path: Path,
    normalized_path: Path,
    rejects_path: Path,
) -> dict[str, int]:
    manifest, requests, _ = verify_bundle(bundle)
    manifest_sha = hash_file(bundle / "manifest.json")
    existing_rows = _read_jsonl(normalized_path) if normalized_path.exists() else []
    existing: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        rid = row.get("request_id")
        if rid in existing:
            raise ValueError(f"duplicate request_id in v8 normalized output: {rid}")
        if rid not in requests:
            raise ValueError(f"v8 normalized row outside bundle: {rid}")
        _verify_normalized_row(
            row,
            request=requests[rid],
            manifest=manifest,
            bundle_manifest_sha256=manifest_sha,
        )
        existing[rid] = row

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    already_present = 0
    seen_raw: set[str] = set()
    for line_number, raw_line in _numbered_lines(raw_results_path):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            rejected.append({
                "line_number": line_number,
                "request_id": None,
                "reason": f"malformed JSONL result: {exc}",
                "raw_line_sha256": hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
            })
            continue
        rid = raw.get("request_id") if isinstance(raw, dict) else None
        try:
            if not isinstance(raw, dict):
                raise ValueError("v8 raw result JSON must be an object")
            if rid in seen_raw:
                raise ValueError("duplicate request_id in v8 raw results")
            seen_raw.add(rid)
            if rid not in requests:
                raise ValueError("request_id is not in the v8 bundle")
            request = requests[rid]
            parsed, hydrated = verify_bound_result(
                raw,
                request=request,
                manifest=manifest,
                bundle_manifest_sha256=manifest_sha,
            )
            parsed_sha = hash_value(parsed)
            result_sha = hash_value(hydrated)
            if rid in existing:
                prior = existing[rid]
                raw_sha = hash_value(raw)
                if (
                    prior["transport_result_sha256"] != raw_sha
                    or prior["bound_transport_result"] != raw
                ):
                    raise ValueError(
                        "changed-raw resume conflict: full bound transport result differs"
                    )
                if prior["parsed_response_sha256"] != parsed_sha:
                    raise ValueError(
                        "changed-raw resume conflict: parsed response hash differs"
                    )
                if prior["result_sha256"] != result_sha or prior["result"] != hydrated:
                    raise ValueError(
                        "changed-raw resume conflict: deterministic hydration/code audit differs"
                    )
                already_present += 1
                continue
            accepted.append({
                "schema_version": NORMALIZED_SCHEMA,
                "request_id": rid,
                "request_sha256": request["request_sha256"],
                "paper_input_sha256": request["paper_input_sha256"],
                "source_map_sha256": request["source_map_sha256"],
                "source_crosswalk_sha256": request["source_crosswalk_sha256"],
                "prompt_spec_sha256": request["prompt_spec_sha256"],
                "model_manifest_sha256": manifest["model_manifest"][
                    "canonical_sha256"
                ],
                "bundle_manifest_sha256": manifest_sha,
                "runner_sha256": raw["runner_sha256"],
                "api_payload_sha256": raw["api_payload_sha256"],
                "transport_result_sha256": hash_value(raw),
                "bound_transport_result": raw,
                "parsed_response_sha256": parsed_sha,
                "validated_response": parsed,
                "result_sha256": result_sha,
                "result": hydrated,
            })
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({
                "line_number": line_number,
                "request_id": rid,
                "reason": str(exc),
                "raw_result_sha256": hash_value(raw),
            })
    if accepted:
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(normalized_path, accepted, mode="a")
    if rejected:
        rejects_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(rejects_path, rejected, mode="a")
    return {
        "accepted_new": len(accepted),
        "already_present_exact": already_present,
        "rejected": len(rejected),
        "remaining_prompt_eligible": len(requests) - len(existing) - len(accepted),
        "structural_abstentions": manifest["strata"]["observed"][
            "missing_body_structural_abstentions"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    prepare_parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    prepare_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    prepare_parser.add_argument("--v7-bundle", type=Path, default=DEFAULT_V7_BUNDLE)
    prepare_parser.add_argument(
        "--historical-code-comparator",
        type=Path,
        default=DEFAULT_HISTORICAL_CODE_COMPARATOR,
    )
    prepare_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    ingest_parser.add_argument("--raw-results", type=Path, required=True)
    ingest_parser.add_argument("--normalized", type=Path)
    ingest_parser.add_argument("--rejects", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(
            args.input.resolve(), args.spec.resolve(), args.model.resolve(),
            args.v7_bundle.resolve(), args.out.resolve(),
            args.historical_code_comparator.resolve(),
        )
        print(json.dumps({
            "status": manifest["status"],
            "prompt_eligible": manifest["strata"]["observed"][
                "body_present_prompt_eligible"
            ],
            "structural_abstentions": manifest["strata"]["observed"][
                "missing_body_structural_abstentions"
            ],
            "api_calls_made": 0,
            "gpu_used": False,
        }, sort_keys=True))
    elif args.command == "verify":
        manifest, requests, abstentions = verify_bundle(args.bundle.resolve())
        print(json.dumps({
            "status": "verified_full_manifest_and_artifacts",
            "schema_version": manifest["schema_version"],
            "prompt_eligible": len(requests),
            "structural_abstentions": len(abstentions),
        }, sort_keys=True))
    else:
        normalized = args.normalized or args.bundle / "normalized_results_v8.jsonl"
        rejects = args.rejects or args.bundle / "rejected_results_v8.jsonl"
        print(json.dumps(ingest(
            args.bundle.resolve(), args.raw_results.resolve(), normalized.resolve(),
            rejects.resolve(),
        ), sort_keys=True))


if __name__ == "__main__":
    main()
