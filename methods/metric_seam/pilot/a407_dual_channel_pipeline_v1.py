#!/usr/bin/env python3
"""Offline preparation pipeline for the clean a407 dual-channel evaluation.

This module never reads the historical source item file.  Its only item input
is the readonly opaque heldout bundle produced by the trusted sealer.  It
verifies the frozen code candidate, executes it deterministically over the
heldout representation, freezes explicit relation/fact coverage and target-free
ablations, and builds two API request arms without sending them.

The raw-prompt arm consumes the exact sanitized ctext.  The hybrid arm consumes
that byte-identical ctext plus deterministic facts from the candidate-bound
CodeScope-v3 helper.  Model outputs are constrained to numeric, abstention, and
relation fields; no rationale or source-text field exists in either schema.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
CANONICAL_OUT_BASENAME = "a407_dual_channel_prepare_002_clean"
CANDIDATE_DIR = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "blind_code_a407_candidate_002_scope_v3"
)
CONTRACT_PATH = (
    ROOT
    / "methods/metric_seam/contracts/"
    "code_review_a407_projected_relation_contract_v2.json"
)
RAW_PROMPT_SPEC_PATH = Path(__file__).with_name("a407_raw_prompt_spec_v1.json")
HYBRID_PROMPT_SPEC_PATH = Path(__file__).with_name(
    "a407_hybrid_seam_prompt_spec_v1.json"
)
MODEL_SPEC_PATH = Path(__file__).with_name(
    "a407_glm47_openrouter_reasoning_off_model_v1.json"
)
PREREGISTRATION_SPEC_PATH = Path(__file__).with_name(
    "a407_dual_channel_evaluation_preregistration_v1.json"
)
RUNNER_PATH = Path(__file__).with_name("a407_dual_channel_runner_v1.py")

EXPECTED_CANDIDATE_FREEZE_SHA256 = (
    "594491db3db4cdcee61f2896f6bfbd8d08a3b98ee852cce21a7cc1ea8beab059"
)
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "97a691d7b57ac8f7db39b9374e53365edf8262029fe33b4145b22163e159c8a5"
)
EXPECTED_CONTRACT_SHA256 = (
    "fdd77fc8a4930842afd6110af279ddbba032eca294cc88b59f083a6e1e31688c"
)
EXPECTED_HELDOUT_COUNT = 100

BUNDLE_SCHEMA = "metric-seam.a407-dual-channel-preparation.v1"
RAW_REQUEST_SCHEMA = "metric-seam.a407-raw-prompt-request.v1"
HYBRID_REQUEST_SCHEMA = "metric-seam.a407-hybrid-seam-request.v1"
CANDIDATE_COVERAGE_ROW_SCHEMA = "metric-seam.a407-code-coverage-row.v1"
CANDIDATE_COVERAGE_SUMMARY_SCHEMA = "metric-seam.a407-code-coverage-summary.v1"
ABLATION_SCHEMA = "metric-seam.a407-code-feature-family-ablation.v1"

STRUCTURAL_RELATIONS = (
    "identifier_morpheme_information_density",
    "scope_weighted_specificity",
    "placeholder_avoidance",
    "declaration_use_consistency",
    "collision_and_shadowing",
)
ALL_RELATIONS = (*STRUCTURAL_RELATIONS, "semantic_context_fit")
HYBRID_RELATIONS = ("semantic_context_fit", "placeholder_appropriateness")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_exclusive_readonly(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def write_json_exclusive(path: Path, value: Any) -> None:
    _write_bytes_exclusive_readonly(path, canonical_bytes(value))


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_bytes(dict(row)) for row in rows)
    _write_bytes_exclusive_readonly(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL rows must be objects")
                rows.append(value)
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _unit_interval_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _validate_heldout_bundle(bundle_path: Path) -> list[dict[str, str]]:
    try:
        from methods.metric_seam.battery.sanitized_ctext_projection_v1 import (
            require_projected_ctext,
        )
        from methods.metric_seam.battery.seal_sanitized_ctext_heldout_v1 import (
            BUNDLE_SCHEMA as HELDOUT_BUNDLE_SCHEMA,
        )
    except ImportError as exc:  # pragma: no cover - repository import invariant
        raise RuntimeError("canonical projection modules are unavailable") from exc

    bundle = _load_json_object(bundle_path)
    if bundle.get("schema") != HELDOUT_BUNDLE_SCHEMA:
        raise ValueError("heldout bundle schema mismatch")
    if bundle.get("task") != "code_review" or bundle.get("criterion_id") != "a407":
        raise ValueError("heldout bundle task identity mismatch")
    items = bundle.get("heldout_items")
    if not isinstance(items, list) or len(items) != EXPECTED_HELDOUT_COUNT:
        raise ValueError("heldout bundle row count mismatch")
    expected_aliases = [
        f"heldout_{index:04d}" for index in range(1, EXPECTED_HELDOUT_COUNT + 1)
    ]
    rows: list[dict[str, str]] = []
    for expected_alias, row in zip(expected_aliases, items):
        if not isinstance(row, dict) or set(row) != {"ctext", "item_key"}:
            raise ValueError("heldout item exceeds the frozen allowlist")
        if row.get("item_key") != expected_alias:
            raise ValueError("heldout aliases are not canonical")
        ctext = row.get("ctext")
        if not isinstance(ctext, str):
            raise ValueError("heldout ctext must be a string")
        require_projected_ctext(ctext)
        rows.append({"item_key": expected_alias, "ctext": ctext})
    return rows


def _validate_specs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = _load_json_object(RAW_PROMPT_SPEC_PATH)
    hybrid = _load_json_object(HYBRID_PROMPT_SPEC_PATH)
    model = _load_json_object(MODEL_SPEC_PATH)
    if raw.get("arm") != "raw_prompt_articulability":
        raise ValueError("raw prompt arm identity mismatch")
    if set(raw.get("relation_semantics", {})) != set(ALL_RELATIONS):
        raise ValueError("raw relation contract mismatch")
    if raw.get("output_source_text_allowed") is not False:
        raise ValueError("raw prompt must forbid source text in output")
    if hybrid.get("arm") != "hybrid_seam":
        raise ValueError("hybrid prompt arm identity mismatch")
    if set(hybrid.get("relation_semantics", {})) != set(HYBRID_RELATIONS):
        raise ValueError("hybrid relation contract mismatch")
    if hybrid.get("fact_boundary", {}).get("schema") != (
        "metric-seam.code-scope-declaration-use-graph.v3"
    ):
        raise ValueError("hybrid fact schema mismatch")
    if hybrid.get("output_source_text_allowed") is not False:
        raise ValueError("hybrid prompt must forbid source text in output")
    exact_model_identity = {
        "backend": "openrouter",
        "protocol": "openai_chat_completions",
        "model": "z-ai/glm-4.7",
        "temperature": 0.0,
        "response_format": "json_schema",
        "provider_require_parameters": True,
        "reasoning": {"effort": "none"},
        "request_timeout_seconds": 120,
        "max_attempts": 2,
        "runner_mode": "strictly_serial_append_only_resume",
        "execution_status": "prepared_not_run",
    }
    if any(model.get(key) != value for key, value in exact_model_identity.items()):
        raise ValueError("GLM-4.7 OpenRouter reasoning-off identity drift")
    if model.get("response_source_text_allowed") is not False:
        raise ValueError("model contract must forbid source text in responses")
    return raw, hybrid, model


def _candidate_paths() -> dict[str, Path]:
    return {
        "candidate_freeze": CANDIDATE_DIR / "candidate_freeze.json",
        "candidate_source": CANDIDATE_DIR / "candidate.py",
        "candidate_config": CANDIDATE_DIR / "candidate_config.json",
        "compiler_log": CANDIDATE_DIR / "compiler_log.md",
        "train_fact_summary": CANDIDATE_DIR / "train_fact_summary.json",
    }


def verify_frozen_candidate() -> tuple[Any, dict[str, Any]]:
    """Verify candidate bindings and self-tests without mutating the candidate."""

    paths = _candidate_paths()
    if hash_file(paths["candidate_freeze"]) != EXPECTED_CANDIDATE_FREEZE_SHA256:
        raise RuntimeError("candidate freeze hash mismatch")
    if hash_file(paths["candidate_source"]) != EXPECTED_CANDIDATE_SOURCE_SHA256:
        raise RuntimeError("candidate source hash mismatch")
    if hash_file(CONTRACT_PATH) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("projected relation contract hash mismatch")
    freeze = _load_json_object(paths["candidate_freeze"])

    checked_hashes = 0
    matching_hashes = 0
    for section in ("frozen_outputs", "allowed_authorship_inputs"):
        records = freeze.get(section)
        if not isinstance(records, dict):
            raise RuntimeError("candidate freeze binding section is missing")
        for record in records.values():
            if not isinstance(record, dict):
                raise RuntimeError("candidate freeze binding is invalid")
            path_text = record.get("path")
            expected_hash = record.get("sha256")
            if not isinstance(path_text, str) or not isinstance(expected_hash, str):
                raise RuntimeError("candidate freeze binding is invalid")
            bound_path = (ROOT / path_text).resolve()
            checked_hashes += 1
            if bound_path.is_file() and hash_file(bound_path) == expected_hash:
                matching_hashes += 1
    if checked_hashes != matching_hashes:
        raise RuntimeError("a candidate freeze binding did not verify")

    # Exercise the physical CLI self-test while retaining neither stdout nor
    # stderr.  Only byte counts and hashes enter the receipt.
    completed = subprocess.run(
        [sys.executable, str(paths["candidate_source"]), "--self-test"],
        check=False,
        capture_output=True,
    )
    cli_output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError("candidate CLI self-test failed with hidden output")

    spec = importlib.util.spec_from_file_location(
        "_frozen_a407_scope_v3_candidate", paths["candidate_source"]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate import specification is unavailable")
    module = importlib.util.module_from_spec(spec)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        spec.loader.exec_module(module)
        self_test = module.self_test()
    if captured.getvalue():
        raise RuntimeError("candidate import/self-test emitted hidden output")
    if not isinstance(self_test, dict) or self_test.get("passed") is not True:
        raise RuntimeError("candidate embedded self-test failed")

    count_true = lambda value: sum(item is True for item in value.values())
    receipt = {
        "schema": "metric-seam.a407-frozen-candidate-verification.v1",
        "candidate_freeze_sha256": hash_file(paths["candidate_freeze"]),
        "candidate_source_sha256": hash_file(paths["candidate_source"]),
        "candidate_config_sha256": hash_file(paths["candidate_config"]),
        "projected_relation_contract_sha256": hash_file(CONTRACT_PATH),
        "freeze_binding_count": checked_hashes,
        "freeze_binding_match_count": matching_hashes,
        "freeze_bindings_passed": checked_hashes == matching_hashes,
        "cli_self_test_exit_code": completed.returncode,
        "cli_self_test_output_byte_count": len(cli_output),
        "cli_self_test_output_sha256": hashlib.sha256(cli_output).hexdigest(),
        "cli_self_test_output_retained": False,
        "embedded_self_test_passed": True,
        "morpheme_checks_passed": count_true(self_test["morpheme_checks"]),
        "fact_contract_checks_passed": count_true(self_test["fact_contract_checks"]),
        "scalar_contrast_checks_passed": sum(
            row.get("passed") is True
            for row in self_test["scalar_contrast_checks"].values()
        ),
        "semantic_boundary_checks_passed": sum(
            row.get("boundary_preserved") is True
            for row in self_test["semantic_boundary_counterexamples"].values()
        ),
        "candidate_edited": False,
        "reference_accessed": False,
        "evaluation_performed": False,
        "model_calls": False,
        "gpu_used": False,
    }
    return module, receipt


def _validate_candidate_output(row: dict[str, Any], expected_alias: str) -> None:
    if row.get("item_key") != expected_alias:
        raise ValueError("candidate output alias mismatch")
    if not _unit_interval_or_none(row.get("score")) or row.get("score") is None:
        raise ValueError("candidate score must be a finite unit-interval scalar")
    relations = row.get("relation_scores")
    if not isinstance(relations, dict) or set(relations) != set(ALL_RELATIONS):
        raise ValueError("candidate relation output mismatch")
    if any(not _unit_interval_or_none(value) for value in relations.values()):
        raise ValueError("candidate relation score is out of range")
    if relations.get("semantic_context_fit") is not None:
        raise ValueError("candidate semantic_context_fit must remain null")
    facts = row.get("fact_counts")
    if not isinstance(facts, dict):
        raise ValueError("candidate fact counts are missing")
    if any(
        not isinstance(value, (int, dict)) or isinstance(value, bool)
        for value in facts.values()
    ):
        raise ValueError("candidate fact-count output has an invalid type")


def run_frozen_candidate(
    items: list[dict[str, str]], candidate: Any
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    outputs: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    weights = candidate.CONFIG.get("relation_weights")
    if not isinstance(weights, dict) or set(weights) != set(STRUCTURAL_RELATIONS):
        raise RuntimeError("candidate frozen structural weights are unavailable")
    if abs(math.fsum(float(value) for value in weights.values()) - 1.0) > 1e-12:
        raise RuntimeError("candidate structural weights do not sum to one")

    counters: Counter[str] = Counter()
    relation_coverage_counts: Counter[str] = Counter()
    fact_coverage_counts: Counter[str] = Counter()
    parse_totals: Counter[str] = Counter()
    for item in items:
        first = candidate.score_item(dict(item))
        second = candidate.score_item(dict(item))
        if canonical_bytes(first) != canonical_bytes(second):
            raise RuntimeError("candidate output is not exactly repeatable")
        if not isinstance(first, dict):
            raise ValueError("candidate output must be an object")
        _validate_candidate_output(first, item["item_key"])
        outputs.append(first)

        facts = first["fact_counts"]
        declarations = int(facts["declarations"])
        uses = int(facts["uses"])
        supported = int(facts["supported_files_analyzed"])
        unsupported = int(facts["unsupported_files_with_added_code"])
        supported_empty = int(facts["supported_files_without_added_code"])
        truncated = int(facts["truncated_input"]) > 0
        orphaned = int(facts["orphan_fragments"]) > 0
        parse_error = any(
            int(facts[key]) > 0
            for key in ("parse_error_files", "parse_error_nodes", "parse_missing_nodes")
        )
        declarations_present = declarations > 0
        neutral_noncoverage = (
            not declarations_present and float(first["score"]) == 0.5
        )
        if not declarations_present and not neutral_noncoverage:
            raise RuntimeError("no-declaration candidate row drifted from frozen neutral")

        relation_coverage = {
            relation: declarations_present for relation in STRUCTURAL_RELATIONS
        }
        relation_coverage["semantic_context_fit"] = False
        fact_coverage = {
            "parse_and_file_diagnostics": True,
            "scope_graph": supported > 0,
            "declarations_and_morphemes": declarations_present,
            "declaration_use_edges": declarations_present or uses > 0,
            "collision_and_shadowing_absence_or_events": declarations_present,
        }
        parse_coverage = {
            "truncated_input": truncated,
            "orphan_fragments_present": orphaned,
            "parse_error_or_missing_nodes": parse_error,
            "unsupported_added_files_present": unsupported > 0,
            "supported_files_without_added_code_present": supported_empty > 0,
            "no_supported_file_analyzed": supported == 0,
            "no_declarations": not declarations_present,
            "runtime_error": False,
        }
        for relation, covered in relation_coverage.items():
            relation_coverage_counts[relation] += int(covered)
        for family, covered in fact_coverage.items():
            fact_coverage_counts[family] += int(covered)
        for state, present in parse_coverage.items():
            parse_totals[state] += int(present)
        counters["repeatability_match_count"] += 1
        counters["finite_unit_interval_score_count"] += 1
        counters["declaration_covered_count"] += int(declarations_present)
        counters["neutral_no_declaration_noncoverage_count"] += int(neutral_noncoverage)
        counters["semantic_context_fit_null_count"] += int(
            first["relation_scores"]["semantic_context_fit"] is None
        )

        output_hash = hash_value(first)
        coverage_rows.append({
            "schema": CANDIDATE_COVERAGE_ROW_SCHEMA,
            "item_key": item["item_key"],
            "candidate_output_sha256": output_hash,
            "runtime_status": "success",
            "relation_coverage": relation_coverage,
            "fact_family_coverage": fact_coverage,
            "parse_and_input_coverage": parse_coverage,
            "neutral_no_declaration_is_noncoverage": neutral_noncoverage,
            "semantic_context_fit_available": False,
            "whole_construct_code_scalar_fidelity": "UNAVAILABLE",
        })

        relation_scores = first["relation_scores"]
        if declarations_present:
            leave_one_out: dict[str, float] = {}
            for omitted in STRUCTURAL_RELATIONS:
                denominator = 1.0 - float(weights[omitted])
                value = math.fsum(
                    float(weights[relation]) * float(relation_scores[relation])
                    for relation in STRUCTURAL_RELATIONS
                    if relation != omitted
                ) / denominator
                leave_one_out[omitted] = round(min(1.0, max(0.0, value)), 12)
            exposed = {
                relation: float(relation_scores[relation])
                for relation in STRUCTURAL_RELATIONS
            }
            base_score: float | None = float(first["score"])
        else:
            leave_one_out = {relation: None for relation in STRUCTURAL_RELATIONS}  # type: ignore[assignment]
            exposed = {relation: None for relation in STRUCTURAL_RELATIONS}  # type: ignore[assignment]
            base_score = None
        ablations.append({
            "schema": ABLATION_SCHEMA,
            "item_key": item["item_key"],
            "candidate_output_sha256": output_hash,
            "declaration_coverage": declarations_present,
            "structural_partial_aggregate": base_score,
            "exposed_feature_family_scores": exposed,
            "leave_one_feature_family_out_renormalized": leave_one_out,
            "frozen_weight_source": "candidate_config.relation_weights",
            "target_or_reference_informed_weights": False,
            "semantic_context_fit": None,
            "whole_construct_code_scalar_fidelity": "UNAVAILABLE",
        })

    summary = {
        "schema": CANDIDATE_COVERAGE_SUMMARY_SCHEMA,
        "heldout_count": len(items),
        "candidate_output_count": len(outputs),
        "repeatability_match_count": counters["repeatability_match_count"],
        "finite_unit_interval_score_count": counters[
            "finite_unit_interval_score_count"
        ],
        "declaration_covered_count": counters["declaration_covered_count"],
        "neutral_no_declaration_noncoverage_count": counters[
            "neutral_no_declaration_noncoverage_count"
        ],
        "semantic_context_fit_null_count": counters[
            "semantic_context_fit_null_count"
        ],
        "relation_covered_row_counts": {
            relation: relation_coverage_counts[relation]
            for relation in ALL_RELATIONS
        },
        "fact_family_covered_row_counts": dict(sorted(fact_coverage_counts.items())),
        "parse_truncation_error_row_counts": dict(sorted(parse_totals.items())),
        "neutral_score_without_declarations": 0.5,
        "neutral_without_declarations_is_positive_verifier_evidence": False,
        "semantic_context_fit_available": False,
        "independent_whole_scalar_adversary_available": False,
        "whole_construct_code_scalar_fidelity": "UNAVAILABLE",
        "subrelation_evidence_emitted": True,
        "parent_collapsed_without_frozen_aggregation": False,
        "reference_accessed": False,
        "correlation_computed": False,
        "evaluation_performed": False,
    }
    return outputs, coverage_rows, summary, ablations


def _request_material_hash(request: Mapping[str, Any]) -> str:
    return hash_value({
        key: value
        for key, value in request.items()
        if key not in {"request_id", "request_sha256"}
    })


def _build_raw_request(
    ordinal: int,
    item: dict[str, str],
    spec: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    input_value = dict(item)
    material = {
        "schema": RAW_REQUEST_SCHEMA,
        "arm": "raw_prompt_articulability",
        "heldout_ordinal": ordinal,
        "item_key": item["item_key"],
        "input": input_value,
        "ctext_sha256": hash_value(item["ctext"]),
        "prompt_spec_sha256": hash_value(spec),
        "model_spec_sha256": hash_value(model),
        "system_prompt": spec["system_prompt"],
        "user_prompt": "INPUT_JSON\n" + json.dumps(
            input_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "output_source_text_allowed": False,
        "historical_reference_available": False,
    }
    request_hash = hash_value(material)
    return {
        **material,
        "request_id": f"a407_raw_{ordinal:04d}_{request_hash[:16]}",
        "request_sha256": request_hash,
    }


def _build_hybrid_request(
    ordinal: int,
    item: dict[str, str],
    facts: dict[str, Any],
    spec: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    input_value = dict(item)
    prompt_input = {
        "item_key": item["item_key"],
        "ctext": item["ctext"],
        "codescope_v3_facts": facts,
    }
    material = {
        "schema": HYBRID_REQUEST_SCHEMA,
        "arm": "hybrid_seam",
        "heldout_ordinal": ordinal,
        "item_key": item["item_key"],
        "input": input_value,
        "ctext_sha256": hash_value(item["ctext"]),
        "codescope_v3_facts": facts,
        "codescope_v3_facts_sha256": hash_value(facts),
        "prompt_spec_sha256": hash_value(spec),
        "model_spec_sha256": hash_value(model),
        "system_prompt": spec["system_prompt"],
        "user_prompt": "INPUT_AND_FACTS_JSON\n" + json.dumps(
            prompt_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "output_source_text_allowed": False,
        "historical_reference_available": False,
    }
    request_hash = hash_value(material)
    return {
        **material,
        "request_id": f"a407_hybrid_{ordinal:04d}_{request_hash[:16]}",
        "request_sha256": request_hash,
    }


def build_request_arms(
    items: list[dict[str, str]],
    candidate: Any,
    raw_spec: dict[str, Any],
    hybrid_spec: dict[str, Any],
    model_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_requests: list[dict[str, Any]] = []
    hybrid_requests: list[dict[str, Any]] = []
    for ordinal, item in enumerate(items, 1):
        facts = candidate.CodeScopeOpsV3.declaration_use_graph(item["ctext"])
        if facts.get("schema") != "metric-seam.code-scope-declaration-use-graph.v3":
            raise RuntimeError("CodeScope-v3 fact schema mismatch")
        # CodeScope facts may contain identifier/path surfaces but never the
        # complete ctext input as a field or value.
        if any(value == item["ctext"] for value in _iter_strings(facts)):
            raise RuntimeError("CodeScope-v3 facts unexpectedly embed the full ctext")
        raw_requests.append(
            _build_raw_request(ordinal, item, raw_spec, model_spec)
        )
        hybrid_requests.append(
            _build_hybrid_request(
                ordinal, item, facts, hybrid_spec, model_spec
            )
        )
    return raw_requests, hybrid_requests


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "sha256": hash_file(path),
        "byte_count": path.stat().st_size,
        "readonly": path.stat().st_mode & 0o222 == 0,
    }


def _implementation_bindings() -> dict[str, dict[str, str]]:
    paths = {
        "pipeline": Path(__file__),
        "runner": RUNNER_PATH,
        "raw_prompt_spec": RAW_PROMPT_SPEC_PATH,
        "hybrid_prompt_spec": HYBRID_PROMPT_SPEC_PATH,
        "model_spec": MODEL_SPEC_PATH,
        "preregistration_spec": PREREGISTRATION_SPEC_PATH,
        "projected_relation_contract": CONTRACT_PATH,
        "candidate_freeze": CANDIDATE_DIR / "candidate_freeze.json",
        "candidate_source": CANDIDATE_DIR / "candidate.py",
        "candidate_config": CANDIDATE_DIR / "candidate_config.json",
        "codescope_v3": ROOT / "methods/metric_seam/hybrids/ops_code_scope_v3.py",
        "codescope_v2": ROOT / "methods/metric_seam/hybrids/ops_code_scope_v2.py",
        "unified_diff_helper": ROOT / "methods/metric_seam/hybrids/ops_code.py",
        "sanitized_ctext_projection": (
            ROOT / "methods/metric_seam/battery/sanitized_ctext_projection_v1.py"
        ),
        "heldout_sealer": (
            ROOT / "methods/metric_seam/battery/seal_sanitized_ctext_heldout_v1.py"
        ),
        "heldout_steward_audits": (
            ROOT / "methods/metric_seam/battery/audit_sanitized_ctext_heldout_v1.py"
        ),
        "detect_secrets_counts_audit": (
            ROOT / "methods/metric_seam/battery/audit_detect_secrets_counts_v1.py"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("an implementation binding is unavailable")
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for name, path in sorted(paths.items())
    }


def prepare_downstream_bundle(
    *,
    heldout_bundle_path: Path,
    heldout_seal_manifest_path: Path,
    privacy_receipt_path: Path,
    replay_receipt_path: Path,
    detect_secrets_receipt_path: Path,
    out_dir: Path,
) -> Path:
    """Freeze all downstream artifacts without source/reference/API access."""

    if out_dir.name != CANONICAL_OUT_BASENAME:
        raise ValueError("output directory is not the canonical clean preparation")
    if not out_dir.is_dir():
        raise ValueError("trusted sealer must create the output directory first")
    for path in (
        heldout_bundle_path,
        heldout_seal_manifest_path,
        privacy_receipt_path,
        replay_receipt_path,
        detect_secrets_receipt_path,
    ):
        if path.parent.resolve() != out_dir.resolve() or not path.is_file():
            raise ValueError("a required trusted preparation artifact is unavailable")

    items = _validate_heldout_bundle(heldout_bundle_path)
    raw_spec, hybrid_spec, model_spec = _validate_specs()
    preregistration = _load_json_object(PREREGISTRATION_SPEC_PATH)
    if preregistration.get("historical_reference_accessed_at_registration") is not False:
        raise RuntimeError("preregistration is not reference-blind")

    # Freeze preregistration before candidate outputs or request artifacts.  No
    # historical reference path is accepted anywhere in this module.
    preregistration = {
        **preregistration,
        "registration_binding": {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "heldout_bundle_sha256": hash_file(heldout_bundle_path),
            "projected_relation_contract_sha256": hash_file(CONTRACT_PATH),
            "raw_prompt_spec_sha256": hash_file(RAW_PROMPT_SPEC_PATH),
            "hybrid_prompt_spec_sha256": hash_file(HYBRID_PROMPT_SPEC_PATH),
            "model_spec_sha256": hash_file(MODEL_SPEC_PATH),
            "historical_reference_path_recorded": False,
            "historical_reference_value_recorded": False,
        },
    }
    preregistration_path = out_dir / "evaluation_preregistration.json"
    write_json_exclusive(preregistration_path, preregistration)

    candidate, candidate_receipt = verify_frozen_candidate()
    candidate_receipt_path = out_dir / "candidate_verification_receipt.json"
    write_json_exclusive(candidate_receipt_path, candidate_receipt)

    outputs, coverage_rows, coverage_summary, ablations = run_frozen_candidate(
        items, candidate
    )
    candidate_outputs_path = out_dir / "candidate_outputs.jsonl"
    coverage_rows_path = out_dir / "code_coverage_rows.jsonl"
    coverage_summary_path = out_dir / "code_coverage_summary.json"
    ablations_path = out_dir / "code_feature_family_ablations.jsonl"
    write_jsonl_exclusive(candidate_outputs_path, outputs)
    write_jsonl_exclusive(coverage_rows_path, coverage_rows)
    write_json_exclusive(coverage_summary_path, coverage_summary)
    write_jsonl_exclusive(ablations_path, ablations)

    raw_requests, hybrid_requests = build_request_arms(
        items, candidate, raw_spec, hybrid_spec, model_spec
    )
    raw_requests_path = out_dir / "raw_prompt_requests.jsonl"
    hybrid_requests_path = out_dir / "hybrid_seam_requests.jsonl"
    write_jsonl_exclusive(raw_requests_path, raw_requests)
    write_jsonl_exclusive(hybrid_requests_path, hybrid_requests)

    if len(raw_requests) != EXPECTED_HELDOUT_COUNT or len(hybrid_requests) != (
        EXPECTED_HELDOUT_COUNT
    ):
        raise AssertionError("request arm row count mismatch")
    for item, raw_request, hybrid_request in zip(
        items, raw_requests, hybrid_requests
    ):
        expected_hash = hash_value(item["ctext"])
        if not (
            raw_request["input"] == item
            and hybrid_request["input"] == item
            and raw_request["ctext_sha256"] == expected_hash
            and hybrid_request["ctext_sha256"] == expected_hash
        ):
            raise AssertionError("ctext identity across prepared arms drifted")

    artifact_paths = {
        "candidate_outputs.jsonl": candidate_outputs_path,
        "candidate_verification_receipt.json": candidate_receipt_path,
        "code_coverage_rows.jsonl": coverage_rows_path,
        "code_coverage_summary.json": coverage_summary_path,
        "code_feature_family_ablations.jsonl": ablations_path,
        "detect_secrets_counts_only_receipt.json": detect_secrets_receipt_path,
        "evaluation_preregistration.json": preregistration_path,
        "full_corpus_heldout_replay_receipt.json": replay_receipt_path,
        "heldout_bundle.json": heldout_bundle_path,
        "heldout_seal_manifest.json": heldout_seal_manifest_path,
        "hybrid_seam_requests.jsonl": hybrid_requests_path,
        "raw_prompt_requests.jsonl": raw_requests_path,
        "steward_heldout_privacy_receipt.json": privacy_receipt_path,
    }
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preparation_id": CANONICAL_OUT_BASENAME,
        "task": "code_review",
        "criterion_id": "a407",
        "input_boundary": {
            "only_item_input": "readonly heldout_bundle.json",
            "heldout_bundle_sha256": hash_file(heldout_bundle_path),
            "source_item_file_opened_by_downstream_pipeline": False,
            "historical_reference_opened": False,
            "original_identifiers_available": False,
            "source_identifier_map_available": False,
            "sanitized_ctext_identity_across_code_raw_hybrid_count": len(items),
        },
        "candidate_arm": {
            "candidate_id": "blind_code_a407_candidate_002_scope_v3",
            "candidate_freeze_sha256": EXPECTED_CANDIDATE_FREEZE_SHA256,
            "candidate_source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
            "exact_output_count": len(outputs),
            "repeatability_match_count": coverage_summary[
                "repeatability_match_count"
            ],
            "semantic_context_fit": None,
            "whole_construct_code_scalar_fidelity": "UNAVAILABLE",
            "structural_partial_aggregate_emitted": True,
            "subrelation_coverage_emitted": True,
            "parse_truncation_error_coverage_emitted": True,
            "neutral_no_declaration_is_positive_evidence": False,
        },
        "request_arms": {
            "raw_prompt": {
                "request_count": len(raw_requests),
                "prompt_spec_sha256": hash_file(RAW_PROMPT_SPEC_PATH),
                "request_artifact": "raw_prompt_requests.jsonl",
            },
            "hybrid": {
                "request_count": len(hybrid_requests),
                "prompt_spec_sha256": hash_file(HYBRID_PROMPT_SPEC_PATH),
                "request_artifact": "hybrid_seam_requests.jsonl",
                "fact_schema": "metric-seam.code-scope-declaration-use-graph.v3",
                "facts_derived_from_identical_ctext_count": len(hybrid_requests),
            },
            "model_spec_sha256": hash_file(MODEL_SPEC_PATH),
            "model": "z-ai/glm-4.7",
            "backend": "openrouter",
            "reasoning": {"effort": "none"},
            "runner": "strictly_serial_append_only_resume",
            "output_source_text_allowed": False,
        },
        "preregistration": {
            "artifact": "evaluation_preregistration.json",
            "sha256": hash_file(preregistration_path),
            "historical_reference_role": "prompt_reference_replay_not_truth",
            "expected_exact_input_primary_count": 99,
            "expected_sanitizer_changed_exclusion_count": 1,
            "full_isomorphism_requires_four_fidelity_dimensions": True,
        },
        "authorship_boundary": {
            "procedural_bundle_only": True,
            "os_isolated": False,
            "description": "procedural_not_os_isolated",
        },
        "execution_status": {
            "candidate_executed": True,
            "request_arms_prepared": True,
            "model_calls": False,
            "api_calls": False,
            "gpu_used": False,
            "historical_reference_accessed": False,
            "reference_values_recorded": False,
            "correlation_computed": False,
            "evaluation_performed": False,
        },
        "implementation": _implementation_bindings(),
        "artifacts": {
            name: _artifact_record(path)
            for name, path in sorted(artifact_paths.items())
        },
    }
    manifest_path = out_dir / "preparation_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    return manifest_path


def verify_preparation_bundle(
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Replay hashes and ctext/fact bindings for the future serial runner."""

    manifest_path = out_dir / "preparation_manifest.json"
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("preparation manifest schema mismatch")
    if manifest.get("preparation_id") != CANONICAL_OUT_BASENAME:
        raise ValueError("preparation identity mismatch")
    for name, record in manifest.get("artifacts", {}).items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("artifact name is not a basename")
        path = out_dir / name
        if not path.is_file() or hash_file(path) != record.get("sha256"):
            raise ValueError("preparation artifact hash mismatch")
        if path.stat().st_mode & 0o222:
            raise ValueError("preparation artifact is not readonly")
    items = _validate_heldout_bundle(out_dir / "heldout_bundle.json")
    raw_requests = read_jsonl(out_dir / "raw_prompt_requests.jsonl")
    hybrid_requests = read_jsonl(out_dir / "hybrid_seam_requests.jsonl")
    if not (
        len(items) == len(raw_requests) == len(hybrid_requests) == EXPECTED_HELDOUT_COUNT
    ):
        raise ValueError("prepared request counts differ")
    seen: set[str] = set()
    for ordinal, (item, raw, hybrid) in enumerate(
        zip(items, raw_requests, hybrid_requests), 1
    ):
        for request, schema, arm in (
            (raw, RAW_REQUEST_SCHEMA, "raw_prompt_articulability"),
            (hybrid, HYBRID_REQUEST_SCHEMA, "hybrid_seam"),
        ):
            if request.get("schema") != schema or request.get("arm") != arm:
                raise ValueError("prepared request identity mismatch")
            if request.get("heldout_ordinal") != ordinal:
                raise ValueError("prepared request order mismatch")
            if request.get("input") != item or request.get("item_key") != item["item_key"]:
                raise ValueError("prepared request input drift")
            if request.get("ctext_sha256") != hash_value(item["ctext"]):
                raise ValueError("prepared request ctext hash drift")
            if request.get("request_sha256") != _request_material_hash(request):
                raise ValueError("prepared request hash drift")
            request_id = request.get("request_id")
            if not isinstance(request_id, str) or request_id in seen:
                raise ValueError("prepared request IDs are invalid")
            seen.add(request_id)
        facts = hybrid.get("codescope_v3_facts")
        if not isinstance(facts, dict) or hybrid.get(
            "codescope_v3_facts_sha256"
        ) != hash_value(facts):
            raise ValueError("hybrid fact binding drift")
    model = _load_json_object(MODEL_SPEC_PATH)
    if hash_value(model) != raw_requests[0].get("model_spec_sha256"):
        raise ValueError("request/model identity drift")
    return manifest, {"raw_prompt": raw_requests, "hybrid": hybrid_requests}, model
