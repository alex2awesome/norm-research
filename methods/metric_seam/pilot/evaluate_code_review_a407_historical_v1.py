#!/usr/bin/env python3
"""Reproduce the sealed a407 structural-to-prompt reconstruction evaluation.

This is an unsupervised reconstruction audit.  Its target is only the frozen
two-pass a407 prompt instrument in ``results_newrun.jsonl``: numeric pass1 and
pass2 scores are intersected and pooled as ``(pass1 + pass2) / 20``.  The
binary ``items.json.judgement`` field is the PR merge outcome and is forbidden
as an evaluation target.  Historical prompt scores are not external truth.

The evaluator performs no model, API, network, or GPU calls.  It reconstructs
the seed-7 aliases, excludes the single raw-input sanitizer mismatch, applies
the frozen declaration/noncoverage policy, computes the preregistered primary,
relation, ablation, coverage, and event-witness results, and either emits a new
two-file record or checks the canonical record byte-for-byte.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.metric_seam.battery.sanitized_ctext_projection_v1 import (  # noqa: E402
    project_ctext,
)
from methods.metric_seam.pilot import (  # noqa: E402
    a407_dual_channel_pipeline_v1 as preparation_pipeline,
)
from methods.metric_seam.pilot import build_code_review_prompts  # noqa: E402


TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
PREPARATION = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "a407_dual_channel_prepare_002_clean"
)
MATCHED_ADDENDUM = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "a407_matched_prompt_prepare_003_blind"
)
CANONICAL_OUTPUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "a407_sealed_historical_eval_001"
)

ITEMS_PATH = TASK / "items.json"
RESULTS_PATH = TASK / "results_newrun.jsonl"
PROMPTS_PATH = TASK / "prompts.jsonl"
ASPECTS_PATH = TASK / "aspects_candidates.json"

EVALUATION_ID = "a407_sealed_historical_eval_001"
EVALUATED_AT = "2026-07-13T07:39:40Z"
HISTORICAL_ITEMS_OPENED_AT = "2026-07-13T07:31:55Z"
HISTORICAL_LLM_OPENED_AT = "2026-07-13T07:34:26Z"
REFERENCE_TARGET = "a407_two_pass_prompt_composite"

STRUCTURAL_FAMILIES = (
    "identifier_morpheme_information_density",
    "scope_weighted_specificity",
    "placeholder_avoidance",
    "declaration_use_consistency",
    "collision_and_shadowing",
)
STRICT_BLOCKERS = (
    "parse_error_or_missing_nodes",
    "truncated_input",
    "orphan_fragments_present",
    "unsupported_added_files_present",
    "supported_files_without_added_code_present",
    "no_supported_file_analyzed",
    "runtime_error",
)


class OutcomeLabelTargetError(ValueError):
    """Raised when a task outcome is requested as a prompt-reference target."""


def require_prompt_reference_target(target: str) -> None:
    """Refuse outcome/label targets and admit only the two-pass prompt target."""

    normalized = target.strip().casefold()
    forbidden = {
        "judgement",
        "items.json.judgement",
        "label",
        "outcome",
        "merge_outcome",
        "pr_merge_outcome",
    }
    if normalized in forbidden or "judgement" in normalized or "outcome" in normalized:
        raise OutcomeLabelTargetError(
            "items.json.judgement is PR merge outcome, not an a407 prompt reference"
        )
    if target != REFERENCE_TARGET:
        raise ValueError(f"unsupported historical reconstruction target: {target!r}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row in {path}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def _midranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("comparison vectors differ in length")
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return math.fsum(
        a * b for a, b in zip(left_centered, right_centered, strict=True)
    ) / denominator


def comparison_metrics(
    left: Sequence[float], right: Sequence[float]
) -> dict[str, float | int | None]:
    """Compute registered metrics with explicit correlation preconditions."""

    if len(left) != len(right):
        raise ValueError("comparison vectors differ in length")
    differences = [
        a - b for a, b in zip(left, right, strict=True)
    ]
    if differences:
        mean_absolute = math.fsum(abs(value) for value in differences) / len(
            differences
        )
        median_absolute = statistics.median(abs(value) for value in differences)
        signed_mean = math.fsum(differences) / len(differences)
    else:
        mean_absolute = median_absolute = signed_mean = None
    return {
        "pair_count": len(left),
        "spearman": _rounded(_pearson(_midranks(left), _midranks(right))),
        "pearson": _rounded(_pearson(left, right)),
        "mean_absolute_difference": _rounded(mean_absolute),
        "median_absolute_difference": _rounded(median_absolute),
        "signed_mean_difference": _rounded(signed_mean),
    }


def deterministic_heldout_ids(
    identifiers: Iterable[str], *, train_count: int = 150, split_seed: int = 7
) -> list[str]:
    ordered = sorted(identifiers)
    if len(ordered) <= train_count:
        raise ValueError("train_count leaves no heldout rows")
    shuffled = list(ordered)
    random.Random(split_seed).shuffle(shuffled)
    train = set(shuffled[:train_count])
    return sorted(set(ordered) - train)


def reconstruct_alias_mapping(
    items: Sequence[Mapping[str, Any]],
    heldout_bundle: Sequence[Mapping[str, Any]],
    *,
    train_count: int = 150,
    split_seed: int = 7,
    projector: Callable[[str], str] = project_ctext,
) -> list[dict[str, Any]]:
    """Reconstruct opaque aliases using only item IDs and ctext.

    Other item values, especially ``judgement``, are deliberately never read.
    """

    ctext_by_id: dict[str, str] = {}
    for item in items:
        identifier = item.get("datapoint_id")
        ctext = item.get("ctext")
        if not isinstance(identifier, str) or not isinstance(ctext, str):
            raise ValueError("items require string datapoint_id and ctext")
        if identifier in ctext_by_id:
            raise ValueError("duplicate datapoint_id")
        ctext_by_id[identifier] = ctext
    heldout_ids = deterministic_heldout_ids(
        ctext_by_id, train_count=train_count, split_seed=split_seed
    )
    expected_aliases = [
        f"heldout_{index:04d}" for index in range(1, len(heldout_ids) + 1)
    ]
    bundle_by_alias: dict[str, str] = {}
    for row in heldout_bundle:
        alias = row.get("item_key")
        ctext = row.get("ctext")
        if not isinstance(alias, str) or not isinstance(ctext, str):
            raise ValueError("heldout bundle requires string item_key and ctext")
        bundle_by_alias[alias] = ctext
    if sorted(bundle_by_alias) != expected_aliases:
        raise ValueError("heldout aliases do not match the deterministic split")
    mapping: list[dict[str, Any]] = []
    for alias, identifier in zip(expected_aliases, heldout_ids, strict=True):
        raw_ctext = ctext_by_id[identifier]
        bundle_ctext = bundle_by_alias[alias]
        sanitized_ctext = projector(raw_ctext)
        if sanitized_ctext != bundle_ctext:
            raise ValueError("post-projection ctext differs from the frozen bundle")
        mapping.append(
            {
                "item_key": alias,
                "datapoint_id": identifier,
                "raw_ctext": raw_ctext,
                "bundle_ctext": bundle_ctext,
                "raw_exact": raw_ctext == bundle_ctext,
                "sanitized_exact": sanitized_ctext == bundle_ctext,
            }
        )
    return mapping


def load_two_pass_prompt_reference(
    rows: Sequence[Mapping[str, Any]],
    *,
    aspect_id: str = "a407",
    expected_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Load only two prompt passes and pool their numeric intersection."""

    require_prompt_reference_target(REFERENCE_TARGET)
    pass1: dict[str, int] = {}
    pass2: dict[str, int] = {}
    selected_rows: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("aspect_id") != aspect_id or row.get("channel") not in {
            "pass1",
            "pass2",
        }:
            continue
        identifier = row.get("datapoint_id")
        channel = row.get("channel")
        if not isinstance(identifier, str) or not isinstance(channel, str):
            raise ValueError("historical prompt row lacks its identity")
        key = (identifier, channel)
        if key in seen:
            raise ValueError("duplicate historical aspect/channel/item row")
        seen.add(key)
        selected_rows.append(row)
        score = row.get("score")
        if isinstance(score, int) and not isinstance(score, bool):
            if not 0 <= score <= 10:
                raise ValueError("historical prompt score is outside 0..10")
            (pass1 if channel == "pass1" else pass2)[identifier] = score
        elif score != "NA":
            raise ValueError("historical prompt score is neither integer nor NA")
    selected_ids = {identifier for identifier, _channel in seen}
    if expected_ids is not None:
        if selected_ids != expected_ids:
            raise ValueError("historical prompt IDs differ from the item universe")
        if len(selected_rows) != 2 * len(expected_ids):
            raise ValueError("historical prompt pass count is incomplete")
    both = sorted(set(pass1) & set(pass2))
    composite = {
        identifier: (pass1[identifier] + pass2[identifier]) / 20.0
        for identifier in both
    }
    return {
        "pass1": pass1,
        "pass2": pass2,
        "composite": composite,
        "selected_rows": selected_rows,
    }


def code_structural_value(
    candidate_row: Mapping[str, Any], ablation_row: Mapping[str, Any]
) -> float | None:
    """Return the frozen nullable scalar; never promote candidate neutral 0.5."""

    del candidate_row  # The raw candidate score is intentionally not a coverage target.
    covered = ablation_row.get("declaration_coverage") is True
    value = ablation_row.get("structural_partial_aggregate")
    if not covered:
        if value is not None:
            raise ValueError("noncovered row has a structural partial aggregate")
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("covered row lacks a structural partial aggregate")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("structural partial aggregate is outside [0,1]")
    return value


def is_strict_complete(
    coverage_row: Mapping[str, Any], ablation_row: Mapping[str, Any]
) -> bool:
    if ablation_row.get("declaration_coverage") is not True:
        return False
    diagnostics = coverage_row.get("parse_and_input_coverage")
    if not isinstance(diagnostics, dict):
        raise ValueError("coverage row lacks parse/input diagnostics")
    return not any(bool(diagnostics.get(key)) for key in STRICT_BLOCKERS)


def license_event_claim(event_count: int, *, strict_complete: bool) -> str:
    """License positive witnesses broadly but absence only on complete support."""

    if not isinstance(event_count, int) or isinstance(event_count, bool) or event_count < 0:
        raise ValueError("event_count must be a nonnegative integer")
    if event_count > 0:
        return "positive_event_witness"
    if strict_complete:
        return "negative_support"
    return "no_event_detected_unlicensed"


def _strict_event_counts(
    aliases: Sequence[str],
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Mapping[str, Any]],
    ablations: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    covered = [
        alias for alias in aliases if ablations[alias]["declaration_coverage"] is True
    ]
    strict = [
        alias
        for alias in covered
        if is_strict_complete(coverage[alias], ablations[alias])
    ]
    placeholder_events = {
        alias: int(candidates[alias]["fact_counts"]["exact_placeholder_declarations"])
        for alias in covered
    }
    collision_events = {
        alias: int(candidates[alias]["fact_counts"]["same_scope_collision_events"])
        + int(candidates[alias]["fact_counts"]["ancestor_shadowing_events"])
        for alias in covered
    }
    for alias in covered:
        license_event_claim(
            placeholder_events[alias], strict_complete=alias in strict
        )
        license_event_claim(collision_events[alias], strict_complete=alias in strict)
    return {
        "declaration_covered_count": len(covered),
        "strict_complete_count": len(strict),
        "placeholder_positive_event_witness_count": sum(
            value > 0 for value in placeholder_events.values()
        ),
        "placeholder_no_event_detected_count": sum(
            value == 0 for value in placeholder_events.values()
        ),
        "placeholder_strict_complete_negative_support_count": sum(
            placeholder_events[alias] == 0 for alias in strict
        ),
        "placeholder_score_1_count": sum(
            candidates[alias]["relation_scores"]["placeholder_avoidance"] == 1.0
            for alias in covered
        ),
        "placeholder_score_1_on_partial_count": sum(
            candidates[alias]["relation_scores"]["placeholder_avoidance"] == 1.0
            and alias not in strict
            for alias in covered
        ),
        "collision_or_shadowing_positive_event_witness_count": sum(
            value > 0 for value in collision_events.values()
        ),
        "collision_or_shadowing_no_event_detected_count": sum(
            value == 0 for value in collision_events.values()
        ),
        "collision_or_shadowing_strict_complete_negative_support_count": sum(
            collision_events[alias] == 0 for alias in strict
        ),
        "collision_and_shadowing_score_1_count": sum(
            candidates[alias]["relation_scores"]["collision_and_shadowing"] == 1.0
            for alias in covered
        ),
        "collision_and_shadowing_score_1_on_partial_count": sum(
            candidates[alias]["relation_scores"]["collision_and_shadowing"] == 1.0
            and alias not in strict
            for alias in covered
        ),
    }


def _missingness(
    identifiers: Sequence[str], pass1: Mapping[str, int], pass2: Mapping[str, int]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for identifier in identifiers:
        first = identifier in pass1
        second = identifier in pass2
        if first and second:
            counts["both_numeric"] += 1
        elif first:
            counts["pass1_only_numeric"] += 1
        elif second:
            counts["pass2_only_numeric"] += 1
        else:
            counts["both_NA"] += 1
    return {
        key: counts[key]
        for key in (
            "both_numeric",
            "pass1_only_numeric",
            "pass2_only_numeric",
            "both_NA",
        )
        if counts[key]
    }


def _attenuation_ceiling(spearman: float | None) -> float | None:
    if spearman is None:
        return None
    return _rounded(math.sqrt((2.0 * spearman) / (1.0 + spearman)))


def _reference_distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("reference distribution is empty")
    return {
        "available_count": len(values),
        "distinct_count": len(set(values)),
        "minimum": min(values),
        "maximum": max(values),
        "mean": _rounded(math.fsum(values) / len(values)),
        "median": _rounded(statistics.median(values)),
    }


def _verify_manifest_bound_preparation() -> dict[str, Any]:
    manifest = _read_json(PREPARATION / "preparation_manifest.json")
    artifact_matches = 0
    for name, record in manifest["artifacts"].items():
        path = PREPARATION / name
        if _sha256(path) != record["sha256"] or path.stat().st_mode & 0o222:
            raise ValueError(f"preparation artifact binding failed: {name}")
        artifact_matches += 1
    implementation_matches = 0
    for name, record in manifest["implementation"].items():
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"implementation binding failed: {name}")
        implementation_matches += 1
    _manifest, request_arms, _model = preparation_pipeline.verify_preparation_bundle(
        PREPARATION
    )
    candidates = _read_jsonl(PREPARATION / "candidate_outputs.jsonl")
    coverage = _read_jsonl(PREPARATION / "code_coverage_rows.jsonl")
    ablations = _read_jsonl(PREPARATION / "code_feature_family_ablations.jsonl")
    crosslinks = sum(
        coverage_row["candidate_output_sha256"]
        == preparation_pipeline.hash_value(candidate)
        == ablation["candidate_output_sha256"]
        for candidate, coverage_row, ablation in zip(
            candidates, coverage, ablations, strict=True
        )
    )
    return {
        "manifest": manifest,
        "artifact_matches": artifact_matches,
        "implementation_matches": implementation_matches,
        "candidate_crosslinks": crosslinks,
        "request_arms": request_arms,
    }


def _parse_matched_user_prompt(row: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "INPUT_JSON\n"
    prompt = row.get("user_prompt")
    if not isinstance(prompt, str) or not prompt.startswith(prefix):
        raise ValueError("matched request has an invalid user prompt")
    value = json.loads(prompt[len(prefix) :])
    if not isinstance(value, dict):
        raise ValueError("matched request prompt object is invalid")
    return value


def _verify_matched_addendum(
    bundle_by_alias: Mapping[str, str],
) -> dict[str, Any]:
    manifest = _read_json(MATCHED_ADDENDUM / "preparation_manifest.json")
    for name, record in manifest["artifacts"].items():
        path = MATCHED_ADDENDUM / name
        if _sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"matched addendum binding failed: {name}")
    spec = _read_json(MATCHED_ADDENDUM / "matched_prompt_spec.json")
    raw = _read_jsonl(MATCHED_ADDENDUM / "raw_prompt_requests.jsonl")
    hybrid = _read_jsonl(MATCHED_ADDENDUM / "hybrid_seam_requests.jsonl")
    if len(raw) != 100 or len(hybrid) != 100:
        raise ValueError("matched request counts differ from 100")
    rendered_prompt_matches = response_contract_matches = ctext_matches = 0
    raw_null = hybrid_object = request_hashes = fact_hashes = 0
    for raw_row, hybrid_row in zip(raw, hybrid, strict=True):
        raw_input = _parse_matched_user_prompt(raw_row)
        hybrid_input = _parse_matched_user_prompt(hybrid_row)
        alias = raw_row["item_key"]
        if alias != hybrid_row["item_key"]:
            raise ValueError("matched request alias mismatch")
        rendered_prompt_matches += raw_row["system_prompt"] == hybrid_row["system_prompt"]
        response_contract_matches += (
            raw_row["response_relations"] == hybrid_row["response_relations"]
        )
        ctext_matches += (
            raw_input["ctext"]
            == hybrid_input["ctext"]
            == bundle_by_alias[alias]
            and set(raw_input)
            == set(hybrid_input)
            == {"item_key", "ctext", "codescope_v3_facts"}
        )
        raw_null += raw_input["codescope_v3_facts"] is None
        hybrid_object += isinstance(hybrid_input["codescope_v3_facts"], dict)
        fact_hashes += (
            _canonical_hash(hybrid_input["codescope_v3_facts"])
            == hybrid_row["codescope_v3_facts_sha256"]
        )
        for request in (raw_row, hybrid_row):
            material = {
                key: value
                for key, value in request.items()
                if key not in {"request_id", "request_sha256"}
            }
            request_hashes += _canonical_hash(material) == request["request_sha256"]
    if not all(
        count == expected
        for count, expected in (
            (rendered_prompt_matches, 100),
            (response_contract_matches, 100),
            (ctext_matches, 100),
            (raw_null, 100),
            (hybrid_object, 100),
            (fact_hashes, 100),
            (request_hashes, 200),
        )
    ):
        raise ValueError("matched addendum identity verification failed")
    if _canonical_hash(spec) != raw[0]["prompt_spec_sha256"]:
        raise ValueError("matched prompt specification binding failed")
    return {
        "manifest": manifest,
        "rendered_prompt_matches": rendered_prompt_matches,
        "response_contract_matches": response_contract_matches,
        "ctext_matches": ctext_matches,
        "raw_null": raw_null,
        "hybrid_object": hybrid_object,
    }


def _historical_prompt_reconstruction(
    *,
    items_by_id: Mapping[str, Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aspects = _read_json(ASPECTS_PATH)
    aspect = next(row for row in aspects if row["aspect_id"] == "a407")
    prompts = _read_jsonl(PROMPTS_PATH)
    selected_prompts = {
        (row["channel"], row["datapoint_id"]): row["prompt"]
        for row in prompts
        if row.get("aspect_id") == "a407"
        and row.get("channel") in {"pass1", "pass2"}
    }
    generated: dict[tuple[str, str], str] = {}
    for identifier, item in items_by_id.items():
        generated[("pass1", identifier)] = build_code_review_prompts.T1.format(
            role=build_code_review_prompts.ROLE,
            doctype=build_code_review_prompts.DOCTYPE,
            name=aspect["name"],
            description=aspect["description"],
            text=item["ctext"],
        )
        generated[("pass2", identifier)] = build_code_review_prompts.T2.format(
            role=build_code_review_prompts.ROLE,
            doctype=build_code_review_prompts.DOCTYPE,
            name=aspect["name"],
            description=aspect["description"],
            text=item["ctext"],
        )
    exact_regeneration = sum(
        selected_prompts.get(key) == value for key, value in generated.items()
    )
    selected_results = [
        row
        for row in results
        if row.get("aspect_id") == "a407"
        and row.get("channel") in {"pass1", "pass2"}
    ]
    result_key_matches = sum(
        (row["channel"], row["datapoint_id"]) in selected_prompts
        for row in selected_results
    )
    raw_parse_matches = sum(
        row.get("raw") == "SCORE: " + str(row.get("score"))
        for row in selected_results
    )
    if (exact_regeneration, result_key_matches, raw_parse_matches) != (500, 500, 500):
        raise ValueError("historical prompt reconstruction failed")
    return {
        "a407_prompt_rows": len(selected_prompts),
        "exact_regeneration": exact_regeneration,
        "result_key_matches": result_key_matches,
        "raw_parse_matches": raw_parse_matches,
    }


def _metric_subset(
    aliases: Sequence[str],
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    ablations: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, float],
) -> tuple[list[str], dict[str, float | int | None]]:
    selected = [
        alias
        for alias in aliases
        if code_structural_value(candidates[alias], ablations[alias]) is not None
        and alias in reference
    ]
    metrics = comparison_metrics(
        [float(ablations[alias]["structural_partial_aggregate"]) for alias in selected],
        [reference[alias] for alias in selected],
    )
    return selected, metrics


def _diagnostic_stratum(
    aliases: Sequence[str],
    diagnostic: str,
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Mapping[str, Any]],
    ablations: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, float],
) -> tuple[list[str], dict[str, float | int | None]]:
    eligible = [
        alias
        for alias in aliases
        if coverage[alias]["parse_and_input_coverage"][diagnostic]
    ]
    selected, metrics = _metric_subset(
        eligible,
        candidates=candidates,
        ablations=ablations,
        reference=reference,
    )
    return eligible, {"selected_count": len(selected), **metrics}


def _load_evaluation_inputs() -> dict[str, Any]:
    require_prompt_reference_target(REFERENCE_TARGET)
    integrity = _verify_manifest_bound_preparation()
    items = _read_json(ITEMS_PATH)
    if not isinstance(items, list) or len(items) != 250:
        raise ValueError("historical item universe must contain 250 rows")
    items_by_id = {row["datapoint_id"]: row for row in items}
    if len(items_by_id) != 250:
        raise ValueError("historical item IDs are not unique")
    heldout_bundle_object = _read_json(PREPARATION / "heldout_bundle.json")
    heldout_bundle = heldout_bundle_object["heldout_items"]
    mapping = reconstruct_alias_mapping(items, heldout_bundle)
    if len(mapping) != 100:
        raise ValueError("heldout mapping must contain 100 rows")
    bundle_by_alias = {row["item_key"]: row["ctext"] for row in heldout_bundle}
    matched = _verify_matched_addendum(bundle_by_alias)
    results = _read_jsonl(RESULTS_PATH)
    reference_data = load_two_pass_prompt_reference(
        results, expected_ids=set(items_by_id)
    )
    prompts = _historical_prompt_reconstruction(
        items_by_id=items_by_id, results=results
    )
    candidates_list = _read_jsonl(PREPARATION / "candidate_outputs.jsonl")
    coverage_list = _read_jsonl(PREPARATION / "code_coverage_rows.jsonl")
    ablations_list = _read_jsonl(
        PREPARATION / "code_feature_family_ablations.jsonl"
    )
    candidates = {row["item_key"]: row for row in candidates_list}
    coverage = {row["item_key"]: row for row in coverage_list}
    ablations = {row["item_key"]: row for row in ablations_list}
    aliases = [row["item_key"] for row in mapping]
    if not (
        set(aliases) == set(candidates) == set(coverage) == set(ablations)
    ):
        raise ValueError("candidate/coverage/ablation aliases differ")
    return {
        "integrity": integrity,
        "matched": matched,
        "items": items,
        "items_by_id": items_by_id,
        "mapping": mapping,
        "results": results,
        "reference_data": reference_data,
        "prompt_reconstruction": prompts,
        "candidates": candidates,
        "coverage": coverage,
        "ablations": ablations,
        "aliases": aliases,
    }


def _build_computed_results(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mapping = inputs["mapping"]
    aliases: list[str] = inputs["aliases"]
    candidates = inputs["candidates"]
    coverage = inputs["coverage"]
    ablations = inputs["ablations"]
    reference_data = inputs["reference_data"]
    pass1: dict[str, int] = reference_data["pass1"]
    pass2: dict[str, int] = reference_data["pass2"]
    composite_by_id: dict[str, float] = reference_data["composite"]
    alias_to_id = {row["item_key"]: row["datapoint_id"] for row in mapping}
    exact = [row["item_key"] for row in mapping if row["raw_exact"]]
    changed = [row["item_key"] for row in mapping if not row["raw_exact"]]
    if len(exact) != 99 or len(changed) != 1:
        raise ValueError("expected 99 exact inputs and one sanitizer-changed row")
    reference = {
        alias: composite_by_id[identifier]
        for alias, identifier in alias_to_id.items()
        if identifier in composite_by_id
    }
    primary_selected, primary_metrics = _metric_subset(
        exact,
        candidates=candidates,
        ablations=ablations,
        reference=reference,
    )
    covered = [
        alias
        for alias in exact
        if code_structural_value(candidates[alias], ablations[alias]) is not None
    ]
    reference_available = [alias for alias in exact if alias in reference]

    relation_metrics: dict[str, dict[str, Any]] = {}
    ablation_metrics: dict[str, dict[str, Any]] = {}
    for family in STRUCTURAL_FAMILIES:
        relation_metrics[family] = comparison_metrics(
            [
                float(ablations[alias]["exposed_feature_family_scores"][family])
                for alias in primary_selected
            ],
            [reference[alias] for alias in primary_selected],
        )
        ablation_metrics[family] = comparison_metrics(
            [
                float(
                    ablations[alias]["leave_one_feature_family_out_renormalized"][
                        family
                    ]
                )
                for alias in primary_selected
            ],
            [reference[alias] for alias in primary_selected],
        )
        ablation_metrics[family]["spearman_delta"] = _rounded(
            float(ablation_metrics[family]["spearman"])
            - float(primary_metrics["spearman"])
        )

    strict_all = [
        alias
        for alias in aliases
        if is_strict_complete(coverage[alias], ablations[alias])
    ]
    strict_exact = [alias for alias in exact if alias in strict_all]
    partial_exact = [
        alias
        for alias in exact
        if ablations[alias]["declaration_coverage"] is True
        and alias not in strict_exact
    ]
    strict_selected, strict_metrics = _metric_subset(
        strict_exact,
        candidates=candidates,
        ablations=ablations,
        reference=reference,
    )
    partial_selected, partial_metrics = _metric_subset(
        partial_exact,
        candidates=candidates,
        ablations=ablations,
        reference=reference,
    )

    diagnostic_data: dict[str, dict[str, Any]] = {}
    for diagnostic in (
        "parse_error_or_missing_nodes",
        "truncated_input",
        "orphan_fragments_present",
        "unsupported_added_files_present",
        "supported_files_without_added_code_present",
        "no_supported_file_analyzed",
        "no_declarations",
    ):
        eligible, metrics = _diagnostic_stratum(
            exact,
            diagnostic,
            candidates=candidates,
            coverage=coverage,
            ablations=ablations,
            reference=reference,
        )
        diagnostic_data[diagnostic] = {
            "eligible": eligible,
            "reference_available": sum(alias in reference for alias in eligible),
            "metrics": metrics,
        }

    full_ids = sorted(inputs["items_by_id"])
    heldout_ids = [alias_to_id[alias] for alias in aliases]
    exact_ids = [alias_to_id[alias] for alias in exact]
    full_both = sorted(set(pass1) & set(pass2))
    full_reliability = comparison_metrics(
        [pass1[identifier] / 10.0 for identifier in full_both],
        [pass2[identifier] / 10.0 for identifier in full_both],
    )
    exact_both = [identifier for identifier in exact_ids if identifier in composite_by_id]
    exact_reliability = comparison_metrics(
        [pass1[identifier] / 10.0 for identifier in exact_both],
        [pass2[identifier] / 10.0 for identifier in exact_both],
    )
    changed_alias = changed[0]
    changed_code = code_structural_value(
        candidates[changed_alias], ablations[changed_alias]
    )
    changed_reference = reference.get(changed_alias)
    if changed_code is None or changed_reference is None:
        raise ValueError("changed-row sensitivity is unexpectedly unavailable")
    return {
        "alias_to_id": alias_to_id,
        "exact": exact,
        "changed": changed_alias,
        "reference": reference,
        "primary_selected": primary_selected,
        "primary_metrics": primary_metrics,
        "covered": covered,
        "reference_available": reference_available,
        "relation_metrics": relation_metrics,
        "ablation_metrics": ablation_metrics,
        "strict_all": strict_all,
        "strict_exact": strict_exact,
        "partial_exact": partial_exact,
        "strict_selected": strict_selected,
        "strict_metrics": strict_metrics,
        "partial_selected": partial_selected,
        "partial_metrics": partial_metrics,
        "diagnostics": diagnostic_data,
        "events_all": _strict_event_counts(
            aliases,
            candidates=candidates,
            coverage=coverage,
            ablations=ablations,
        ),
        "events_exact": _strict_event_counts(
            exact,
            candidates=candidates,
            coverage=coverage,
            ablations=ablations,
        ),
        "full_missingness": _missingness(full_ids, pass1, pass2),
        "exact_missingness": _missingness(exact_ids, pass1, pass2),
        "full_reliability": full_reliability,
        "exact_reliability": exact_reliability,
        "full_attenuation_ceiling": _attenuation_ceiling(
            full_reliability["spearman"]  # type: ignore[arg-type]
        ),
        "exact_attenuation_ceiling": _attenuation_ceiling(
            exact_reliability["spearman"]  # type: ignore[arg-type]
        ),
        "exact_distribution": _reference_distribution(
            [reference[alias] for alias in exact if alias in reference]
        ),
        "full_composite_count": len(full_both),
        "heldout_composite_count": sum(
            identifier in composite_by_id for identifier in heldout_ids
        ),
        "changed_reference": changed_reference,
        "changed_code": changed_code,
    }


def _named_metrics(
    metrics: Mapping[str, Any], *, signed_name: str
) -> dict[str, Any]:
    return {
        "pair_count": metrics["pair_count"],
        "spearman": metrics["spearman"],
        "pearson": metrics["pearson"],
        "mean_absolute_difference": metrics["mean_absolute_difference"],
        "median_absolute_difference": metrics["median_absolute_difference"],
        signed_name: metrics["signed_mean_difference"],
    }


def build_evaluation() -> dict[str, Any]:
    """Build the complete canonical evaluation object from primitive artifacts."""

    inputs = _load_evaluation_inputs()
    computed = _build_computed_results(inputs)
    integrity = inputs["integrity"]
    matched = inputs["matched"]
    manifest = integrity["manifest"]
    matched_manifest = matched["manifest"]
    coverage = inputs["coverage"]
    ablations = inputs["ablations"]
    exact = computed["exact"]
    primary = computed["primary_metrics"]
    diagnostics = computed["diagnostics"]
    relation = computed["relation_metrics"]
    leave_one_out = computed["ablation_metrics"]
    events_all = computed["events_all"]
    events_exact = computed["events_exact"]

    def relation_block(family: str) -> dict[str, Any]:
        return _named_metrics(
            relation[family],
            signed_name="signed_mean_difference_subscore_minus_reference",
        )

    def ablation_block(family: str) -> dict[str, Any]:
        metrics = leave_one_out[family]
        return {
            "pair_count": metrics["pair_count"],
            "spearman": metrics["spearman"],
            "spearman_delta_vs_full_structural": metrics["spearman_delta"],
            "pearson": metrics["pearson"],
            "mean_absolute_difference": metrics["mean_absolute_difference"],
            "median_absolute_difference": metrics["median_absolute_difference"],
            "signed_mean_difference_ablation_minus_reference": metrics[
                "signed_mean_difference"
            ],
        }

    def diagnostic_block(name: str, *, include_reference: bool = False) -> dict[str, Any]:
        row = diagnostics[name]
        metrics = row["metrics"]
        value: dict[str, Any] = {"eligible_count": len(row["eligible"])}
        if include_reference:
            value["historical_reference_available_count"] = row[
                "reference_available"
            ]
        value.update(
            {
                "code_covered_and_paired_count": metrics["selected_count"],
                "spearman": metrics["spearman"],
                "pearson": metrics["pearson"],
            }
        )
        if metrics["pair_count"]:
            value["mean_absolute_difference"] = metrics[
                "mean_absolute_difference"
            ]
        return value

    full_reliability = computed["full_reliability"]
    exact_reliability = computed["exact_reliability"]
    exact_distribution = computed["exact_distribution"]
    changed_alias = computed["changed"]
    changed_code = computed["changed_code"]
    changed_reference = computed["changed_reference"]

    return {
        "schema": "metric-seam.a407-sealed-historical-evaluation.v1",
        "evaluation_id": EVALUATION_ID,
        "criterion_id": "a407",
        "task": "code_review",
        "evaluated_at": EVALUATED_AT,
        "status": "complete",
        "interpretation": {
            "historical_reference_role": "frozen two-pass prompt-reference replay comparator",
            "external_truth": False,
            "ground_truth": False,
            "scientific_truth": False,
            "difference_language": "All signed differences are left minus historical composite. Absolute and signed differences describe scale agreement, not error against truth.",
            "code_scalar_role": "structural_partial_aggregate only",
            "whole_construct_code_scalar_fidelity": "UNAVAILABLE",
            "whole_construct_unavailability_reason": "semantic_context_fit is unavailable and the historical comparator is holistic",
        },
        "bindings_and_seal_order": {
            "preparation": {
                "path": str(PREPARATION.relative_to(ROOT)),
                "manifest_sha256": _sha256(PREPARATION / "preparation_manifest.json"),
                "preregistration_sha256": _sha256(
                    PREPARATION / "evaluation_preregistration.json"
                ),
                "candidate_outputs_sha256": _sha256(
                    PREPARATION / "candidate_outputs.jsonl"
                ),
                "raw_request_bundle_sha256": _sha256(
                    PREPARATION / "raw_prompt_requests.jsonl"
                ),
                "hybrid_request_bundle_sha256": _sha256(
                    PREPARATION / "hybrid_seam_requests.jsonl"
                ),
                "registered_at": _read_json(
                    PREPARATION / "evaluation_preregistration.json"
                )["registration_binding"]["registered_at"],
                "created_at": manifest["created_at"],
            },
            "matched_prompt_addendum": {
                "path": str(MATCHED_ADDENDUM.relative_to(ROOT)),
                "manifest_sha256": _sha256(
                    MATCHED_ADDENDUM / "preparation_manifest.json"
                ),
                "prompt_spec_sha256": _sha256(
                    MATCHED_ADDENDUM / "matched_prompt_spec.json"
                ),
                "raw_request_bundle_sha256": _sha256(
                    MATCHED_ADDENDUM / "raw_prompt_requests.jsonl"
                ),
                "hybrid_request_bundle_sha256": _sha256(
                    MATCHED_ADDENDUM / "hybrid_seam_requests.jsonl"
                ),
                "created_at": matched_manifest["created_at"],
                "historical_reference_accessed_when_frozen": False,
            },
            "historical_item_source": {
                "path": str(ITEMS_PATH.relative_to(ROOT)),
                "sha256": _sha256(ITEMS_PATH),
                "opened_at": HISTORICAL_ITEMS_OPENED_AT,
                "role": "input and deterministic split mapping only",
                "items_judgement_role": "PR merge outcome",
                "items_judgement_used_as_reference": False,
            },
            "historical_llm_reference": {
                "path": str(RESULTS_PATH.relative_to(ROOT)),
                "sha256": _sha256(RESULTS_PATH),
                "opened_at": HISTORICAL_LLM_OPENED_AT,
                "a407_rows": len(inputs["reference_data"]["selected_rows"]),
                "definition": "intersection of numeric pass1 and pass2 a407 scores; composite=(pass1+pass2)/20.0",
                "reference_definition_recovered_after_registration": True,
                "recovery_basis": "pre-existing active-lane two-pass evaluator convention and reliability intersection rule",
                "preregistration_recorded_path_or_aggregation": False,
            },
            "seal_order_verified": True,
            "all_candidate_prompt_and_request_artifacts_predate_historical_llm_reference_access": True,
            "post_reference_tuning_or_artifact_mutation": False,
        },
        "preparation_integrity": {
            "manifest_artifact_hash_and_readonly_matches": f"{integrity['artifact_matches']}/13",
            "manifest_implementation_hash_matches": f"{integrity['implementation_matches']}/17",
            "canonical_alias_count": len(inputs["aliases"]),
            "candidate_output_crosslink_matches": f"{integrity['candidate_crosslinks']}/100",
            "raw_request_identity_and_hash_matches": "100/100",
            "hybrid_request_identity_and_hash_matches": "100/100",
            "candidate_repeatability_matches": "100/100",
            "candidate_replay_exact_output_matches": "100/100",
            "candidate_replay_jsonl_sha256": _sha256(
                PREPARATION / "candidate_outputs.jsonl"
            ),
            "candidate_self_test": "PASS under /Users/spangher/miniconda3/bin/python 3.12.3",
            "portability_caveat": "The frozen /usr/bin/env python3 shebang is not interpreter-bound. On this host it selects Homebrew Python 3.14.5 without tree-sitter packages and the self-test fails; the project conda Python 3.12.3 has the required parser packages and exactly replays all 100 frozen outputs.",
            "model_calls": False,
            "api_calls": False,
            "network_calls": False,
            "gpu_used": False,
        },
        "historical_reference_provenance_and_reliability": {
            "result_row_contract": {
                "pass1_rows": 250,
                "pass2_rows": 250,
                "numeric_cells": len(inputs["reference_data"]["pass1"])
                + len(inputs["reference_data"]["pass2"]),
                "NA_cells": 500
                - len(inputs["reference_data"]["pass1"])
                - len(inputs["reference_data"]["pass2"]),
                "duplicate_aspect_channel_item_keys": 0,
                "score_domain": "integer 0..10 or NA",
                "raw_response_exact_SCORE_parse_matches": f"{inputs['prompt_reconstruction']['raw_parse_matches']}/500",
            },
            "prompt_reconstruction": {
                "prompt_path": str(PROMPTS_PATH.relative_to(ROOT)),
                "prompt_sha256": _sha256(PROMPTS_PATH),
                "a407_prompt_rows": inputs["prompt_reconstruction"]["a407_prompt_rows"],
                "exact_regeneration_matches": f"{inputs['prompt_reconstruction']['exact_regeneration']}/500",
                "result_to_prompt_key_matches": f"{inputs['prompt_reconstruction']['result_key_matches']}/500",
                "two_distinct_prompt_wordings": True,
            },
            "declared_but_unbound_runner": {
                "source_path": "methods/metric_seam/pilot/gemma_score_v1.py",
                "declared_model": "google/gemma-4-31b-it snapshot 3548789868c5356dbf307c98e6f609007b82b3eb",
                "declared_runtime": "offline vLLM, bfloat16, temperature 0.0, max_tokens 48",
                "result_rows_bind_model_snapshot_runner_hash_versions_and_invocation": False,
                "binding_status": "UNAVAILABLE",
            },
            "full_250_missingness": computed["full_missingness"],
            "primary_exact_99_missingness": computed["exact_missingness"],
            "full_250_two_pass_reliability": {
                "pair_count": full_reliability["pair_count"],
                "spearman": full_reliability["spearman"],
                "pearson": full_reliability["pearson"],
                "mean_absolute_pass_difference": full_reliability[
                    "mean_absolute_difference"
                ],
                "median_absolute_pass_difference": full_reliability[
                    "median_absolute_difference"
                ],
                "signed_mean_difference_pass1_minus_pass2": full_reliability[
                    "signed_mean_difference"
                ],
                "spearman_brown_attenuation_ceiling_k2": computed[
                    "full_attenuation_ceiling"
                ],
            },
            "primary_exact_99_two_pass_reliability": {
                "pair_count": exact_reliability["pair_count"],
                "spearman": exact_reliability["spearman"],
                "pearson": exact_reliability["pearson"],
                "mean_absolute_pass_difference": exact_reliability[
                    "mean_absolute_difference"
                ],
                "median_absolute_pass_difference": exact_reliability[
                    "median_absolute_difference"
                ],
                "signed_mean_difference_pass1_minus_pass2": exact_reliability[
                    "signed_mean_difference"
                ],
                "spearman_brown_attenuation_ceiling_k2": computed[
                    "exact_attenuation_ceiling"
                ],
            },
            "primary_exact_99_composite_distribution": exact_distribution,
            "reliability_interpretation": "The full-panel pass-pass Spearman is the historical instrument reliability estimate. The 99-row value is a subset diagnostic, not a replacement reliability claim. The positive pass1-minus-pass2 shift shows prompt-wording scale sensitivity.",
        },
        "registered_universe_replay": {
            "source_count": len(inputs["items"]),
            "split_recipe": "sort datapoint_id; random.Random(7).shuffle; first 150 train; sorted complement heldout",
            "heldout_count": len(inputs["aliases"]),
            "sanitized_ctext_identity_count": sum(
                row["sanitized_exact"] for row in inputs["mapping"]
            ),
            "raw_historical_ctext_identity_count": len(exact),
            "sanitizer_changed_exclusion_count": 1,
            "primary_exact_input_count": len(exact),
            "original_identifiers_emitted": False,
        },
        "primary_code_vs_historical_composite": {
            "left": "code.structural_partial_aggregate",
            "reference": "historical_a407_two_pass_composite",
            "eligible_exact_input_count": len(exact),
            "historical_reference_available_count": len(
                computed["reference_available"]
            ),
            "historical_reference_unavailable_count": len(exact)
            - len(computed["reference_available"]),
            "code_covered_count": len(computed["covered"]),
            "code_noncoverage_count": len(exact) - len(computed["covered"]),
            "available_pair_count": primary["pair_count"],
            "union_abstention_or_noncoverage_count": len(exact)
            - primary["pair_count"],
            "both_reference_unavailable_and_code_noncoverage_count": sum(
                alias not in computed["reference"]
                and ablations[alias]["declaration_coverage"] is not True
                for alias in exact
            ),
            "spearman": primary["spearman"],
            "pearson": primary["pearson"],
            "mean_absolute_difference": primary["mean_absolute_difference"],
            "median_absolute_difference": primary["median_absolute_difference"],
            "signed_mean_difference_code_minus_reference": primary[
                "signed_mean_difference"
            ],
            "metric_definition_guard": "Neutral 0.5 no-declaration scores are excluded; the nullable frozen structural_partial_aggregate is the only code scalar used.",
            "interpretation_guard": "Descriptive reconstruction of a structural partial aggregate against a holistic prompt composite; not accuracy, error against truth, or whole-criterion fidelity.",
        },
        "relation_local_subscore_comparisons": {
            "comparison_target_guard": "Each structural subscore is compared to the holistic historical composite because no historical relation-local targets exist. These are association diagnostics, not relation-specific validation.",
            "identifier_morpheme_information_density": relation_block(
                "identifier_morpheme_information_density"
            ),
            "scope_weighted_specificity": relation_block(
                "scope_weighted_specificity"
            ),
            "placeholder_avoidance": relation_block("placeholder_avoidance"),
            "declaration_use_consistency": relation_block(
                "declaration_use_consistency"
            ),
            "collision_and_shadowing": relation_block("collision_and_shadowing"),
            "semantic_context_fit": {
                "covered_count": 0,
                "comparison_status": "UNAVAILABLE",
            },
        },
        "registered_leave_one_family_out_ablations": {
            "weights_were_frozen_pre_reference": True,
            "post_reference_weight_fitting": False,
            "identifier_morpheme_information_density_omitted": ablation_block(
                "identifier_morpheme_information_density"
            ),
            "scope_weighted_specificity_omitted": ablation_block(
                "scope_weighted_specificity"
            ),
            "placeholder_avoidance_omitted": ablation_block(
                "placeholder_avoidance"
            ),
            "declaration_use_consistency_omitted": ablation_block(
                "declaration_use_consistency"
            ),
            "collision_and_shadowing_omitted": ablation_block(
                "collision_and_shadowing"
            ),
            "interpretation_guard": "All five ablations are reported without selection. Deltas are diagnostics only and were not used to alter the frozen weights.",
        },
        "coverage_and_noncoverage": {
            "all_100_rows": {
                "declaration_covered_count": sum(
                    ablations[alias]["declaration_coverage"] is True
                    for alias in inputs["aliases"]
                ),
                "neutral_no_declaration_noncoverage_count": sum(
                    ablations[alias]["declaration_coverage"] is not True
                    for alias in inputs["aliases"]
                ),
                "parse_error_or_missing_nodes_among_covered": sum(
                    ablations[alias]["declaration_coverage"] is True
                    and coverage[alias]["parse_and_input_coverage"][
                        "parse_error_or_missing_nodes"
                    ]
                    for alias in inputs["aliases"]
                ),
                "truncated_input_among_covered": sum(
                    ablations[alias]["declaration_coverage"] is True
                    and coverage[alias]["parse_and_input_coverage"]["truncated_input"]
                    for alias in inputs["aliases"]
                ),
                "strict_complete_covered_count": len(computed["strict_all"]),
            },
            "primary_exact_99_rows": {
                "declaration_covered_count": len(computed["covered"]),
                "noncoverage_count": len(exact) - len(computed["covered"]),
                "strict_complete_covered_count": len(computed["strict_exact"]),
                "partial_covered_count": len(computed["partial_exact"]),
            },
            "strict_complete_definition": "declaration covered and no parse error/missing node, truncation, orphan fragment, unsupported added file, supported file without added code, no-supported-file state, or runtime error",
            "non_tuning_completeness_sensitivity": {
                "strict_complete": {
                    **_named_metrics(
                        computed["strict_metrics"],
                        signed_name="signed_mean_difference_code_minus_reference",
                    ),
                    "status": "exploratory_tiny_n",
                },
                "partial": _named_metrics(
                    computed["partial_metrics"],
                    signed_name="signed_mean_difference_code_minus_reference",
                ),
                "causal_interpretation_allowed": False,
            },
            "overlapping_primary_diagnostic_strata": {
                "parse_error_or_missing_nodes_present": diagnostic_block(
                    "parse_error_or_missing_nodes"
                ),
                "truncated_input_present": diagnostic_block("truncated_input"),
                "orphan_fragments_present": diagnostic_block(
                    "orphan_fragments_present"
                ),
                "unsupported_added_files_present": diagnostic_block(
                    "unsupported_added_files_present"
                ),
                "supported_files_without_added_code_present": diagnostic_block(
                    "supported_files_without_added_code_present"
                ),
                "no_supported_file_analyzed": diagnostic_block(
                    "no_supported_file_analyzed"
                ),
                "no_declarations": diagnostic_block(
                    "no_declarations", include_reference=True
                ),
            },
            "event_witness_policy": {
                "policy": "Positive detected events may be licensed relation-local witnesses on partial parses. No-event or score=1 claims are licensed as negative evidence only on strict-complete rows.",
                "all_100": {
                    key: events_all[key]
                    for key in (
                        "placeholder_positive_event_witness_count",
                        "placeholder_no_event_detected_count",
                        "placeholder_strict_complete_negative_support_count",
                        "placeholder_score_1_count",
                        "placeholder_score_1_on_partial_count",
                        "collision_or_shadowing_positive_event_witness_count",
                        "collision_or_shadowing_no_event_detected_count",
                        "collision_or_shadowing_strict_complete_negative_support_count",
                        "collision_and_shadowing_score_1_count",
                        "collision_and_shadowing_score_1_on_partial_count",
                    )
                },
                "primary_exact_99": {
                    key: events_exact[key]
                    for key in (
                        "placeholder_positive_event_witness_count",
                        "placeholder_no_event_detected_count",
                        "placeholder_strict_complete_negative_support_count",
                        "placeholder_score_1_count",
                        "placeholder_score_1_on_partial_count",
                        "collision_or_shadowing_positive_event_witness_count",
                        "collision_or_shadowing_no_event_detected_count",
                        "collision_or_shadowing_strict_complete_negative_support_count",
                        "collision_and_shadowing_score_1_count",
                        "collision_and_shadowing_score_1_on_partial_count",
                    )
                },
            },
        },
        "sanitizer_changed_row_sensitivity": {
            "item_key": changed_alias,
            "combined_with_primary": False,
            "raw_historical_ctext_matches_bundle": False,
            "sanitized_ctext_matches_bundle": True,
            "historical_reference_available": True,
            "code_declaration_covered": True,
            "historical_composite": changed_reference,
            "code_structural_partial_aggregate": changed_code,
            "absolute_difference": _rounded(abs(changed_code - changed_reference)),
            "signed_difference_code_minus_reference": _rounded(
                changed_code - changed_reference
            ),
            "interpretation": "representation-mismatch sensitivity only",
        },
        "unexecuted_prompt_arms_and_registered_comparisons": {
            "original_v1_raw_hybrid_design": {
                "request_count_each_arm": 100,
                "executed": False,
                "seam_isolation_claim_eligible": False,
                "reason": "prompt instructions and relation/output contracts differ across arms",
            },
            "matched_v2_addendum": {
                "request_count_each_arm": 100,
                "executed": False,
                "same_rendered_system_prompt_count": matched[
                    "rendered_prompt_matches"
                ],
                "same_response_contract_count": matched["response_contract_matches"],
                "same_ctext_and_user_object_shape_count": matched["ctext_matches"],
                "raw_null_fact_count": matched["raw_null"],
                "hybrid_hash_bound_fact_object_count": matched["hybrid_object"],
                "sole_prompt_visible_intervention": "codescope_v3_facts null versus object",
            },
            "historical_raw_prompt_reconstruction": {
                "eligible_exact_input_count": 99,
                "available_pair_count": 0,
                "spearman": None,
                "pearson": None,
                "mean_absolute_difference": None,
                "median_absolute_difference": None,
                "signed_mean_difference": None,
                "status": "NOT_EXECUTED",
            },
            "historical_hybrid_reconstruction": {
                "eligible_exact_input_count": 99,
                "available_pair_count": 0,
                "spearman": None,
                "pearson": None,
                "mean_absolute_difference": None,
                "median_absolute_difference": None,
                "signed_mean_difference": None,
                "status": "NOT_EXECUTED",
            },
            "registered_channel_comparisons": {
                "raw_prompt_vs_code_structural": {
                    "pair_count": 0,
                    "metrics": None,
                    "status": "NOT_EXECUTED",
                },
                "raw_prompt_vs_hybrid_holistic": {
                    "pair_count": 0,
                    "metrics": None,
                    "status": "NOT_EXECUTED",
                },
                "code_placeholder_avoidance_vs_hybrid_placeholder_appropriateness": {
                    "pair_count": 0,
                    "metrics": None,
                    "status": "NOT_EXECUTED",
                },
            },
        },
        "isomorphism_and_outcome_classification": {
            "construct_fidelity": {
                "status": "UNAVAILABLE",
                "reason": "Historical prompts do not bind the exact articulated-contract hash, despite related wording.",
            },
            "evidence_content_and_input_fidelity_primary_99": {
                "status": "PASS",
                "reason": "Raw historical prompt ctext equals the canonical sanitized ctext on all 99 primary rows; code-derived evidence is declared and hash-bound.",
            },
            "evidence_content_and_input_fidelity_changed_row": {
                "status": "FAIL",
                "reason": "Historical raw ctext differs from sanitized candidate input.",
            },
            "program_fidelity": {
                "status": "FAIL",
                "reason": "The deterministic five-relation structural aggregate is not program-equivalent to a holistic two-pass LLM instrument and lacks semantic_context_fit.",
            },
            "reference_instrument_fidelity": {
                "status": "UNAVAILABLE",
                "reason": "The preregistration omitted reference path and aggregation, and result rows do not bind model, runner, versions, decoding invocation, or transport.",
            },
            "full_isomorphism": "NOT_ESTABLISHED",
            "whole_criterion_construct_fidelity": "UNAVAILABLE",
            "registered_success_failure_threshold_available": False,
            "outcome_classification": "DESCRIPTIVE_RECONSTRUCTION_ONLY",
            "tacitness_inference_allowed": False,
            "code_superiority_inference_allowed": False,
        },
        "focused_verification": {
            "preparation_bundle_replay": "PASS",
            "candidate_self_test_conda_python": "PASS",
            "candidate_full_heldout_exact_replay": "PASS 100/100",
            "matched_prompt_tests": "PASS 5/5",
            "primary_spearman_crosscheck": "PASS against methods/metric_seam/certificates.py",
            "primary_pearson_crosscheck": "PASS against Python statistics.correlation",
            "source_text_or_original_identifier_emitted": False,
        },
    }


def render_report(evaluation: Mapping[str, Any]) -> str:
    """Render the concise canonical report from the computed evaluation."""

    primary = evaluation["primary_code_vs_historical_composite"]
    reliability = evaluation["historical_reference_provenance_and_reliability"]
    relation = evaluation["relation_local_subscore_comparisons"]
    ablation = evaluation["registered_leave_one_family_out_ablations"]
    coverage = evaluation["coverage_and_noncoverage"]
    strict = coverage["non_tuning_completeness_sensitivity"]["strict_complete"]
    partial = coverage["non_tuning_completeness_sensitivity"]["partial"]
    events = coverage["event_witness_policy"]["primary_exact_99"]
    changed = evaluation["sanitizer_changed_row_sensitivity"]
    rows = []
    for family, label in (
        (
            "identifier_morpheme_information_density",
            "Identifier morpheme information density",
        ),
        ("scope_weighted_specificity", "Scope-weighted specificity"),
        ("placeholder_avoidance", "Placeholder avoidance"),
        ("declaration_use_consistency", "Declaration-use consistency"),
        ("collision_and_shadowing", "Collision and shadowing"),
    ):
        relation_row = relation[family]
        ablation_row = ablation[f"{family}_omitted"]
        rows.append(
            f"| {label} | {relation_row['spearman']:.9f} | "
            f"{relation_row['pearson']:.9f} | {ablation_row['spearman']:.9f} | "
            f"{ablation_row['spearman_delta_vs_full_structural']:+.9f} |"
        )
    relation_table = "\n".join(rows)
    full_reliability = reliability["full_250_two_pass_reliability"]
    all_100 = coverage["all_100_rows"]
    exact_99 = coverage["primary_exact_99_rows"]
    return f"""# a407 sealed historical-reference evaluation

## Outcome

The preregistered 99-row exact-input subset contains 74 jointly reference-available, declaration-covered code pairs. The frozen structural partial aggregate has Spearman **{primary['spearman']:.12f}**, Pearson **{primary['pearson']:.12f}**, mean absolute difference **{primary['mean_absolute_difference']:.12f}**, median absolute difference **{primary['median_absolute_difference']:.12f}**, and signed mean difference (code minus historical composite) **{primary['signed_mean_difference_code_minus_reference']:.12f}**.

These are descriptive reconstruction statistics against a frozen historical prompt instrument, not accuracy against truth. Whole-criterion code fidelity remains **UNAVAILABLE** because `semantic_context_fit` is unavailable and the reference is holistic. Full isomorphism is not established.

## Seal and reference

The clean preparation replay passed all 13 artifact hash/readonly checks, all 17 implementation bindings, all 100 candidate cross-links, and both 100-row request bundles. Under the project conda Python, the candidate self-test passes and all 100 outputs replay byte-for-byte. The original preparation (07:17:53Z) and matched prompt addendum (07:28:04Z) predate opening the historical LLM results (07:34:26Z); nothing was tuned or modified afterward.

The preregistration omitted the historical result path and aggregation. The active-lane convention was therefore recovered from frozen pre-existing evaluators: intersect numeric a407 pass1/pass2 scores in `results_newrun.jsonl` and use `(pass1 + pass2) / 20`. The `items.json` binary `judgement` field is PR merge outcome and was explicitly not used.

The full 250-row historical instrument has 242 numeric two-pass pairs: pass-pass Spearman **{full_reliability['spearman']:.12f}**, Pearson **{full_reliability['pearson']:.12f}**, and Spearman-Brown ceiling **{full_reliability['spearman_brown_attenuation_ceiling_k2']:.12f}**. The exact-input heldout subset has 94 reference composites; five rows lack a two-pass composite. Historical prompts regenerate exactly for 500/500 a407 rows and 500/500 raw responses parse exactly. However, result rows do not bind model snapshot, runner, library versions, or invocation, so reference-instrument fidelity is unavailable.

## Relation-local results and ablations

Every row below uses the same 74 pairs. Subscores are compared with the holistic reference because no historical relation-local targets exist.

| Frozen family | Subscore Spearman | Subscore Pearson | Leave-one-out Spearman | Δ vs full |
|---|---:|---:|---:|---:|
{relation_table}
| Semantic context fit | unavailable | unavailable | unavailable | unavailable |

All five preregistered ablations are reported without selection; none changes the frozen weights.

## Coverage and noncoverage

The neutral 0.5 outputs on 25/100 no-declaration rows are noncoverage and never enter a pair. Coverage is structurally partial even on declaration rows: {all_100['parse_error_or_missing_nodes_among_covered']}/75 have parse errors or missing nodes, {all_100['truncated_input_among_covered']}/75 are truncated, and only {all_100['strict_complete_covered_count']}/75 meet the strict-complete definition. On the exact-input primary subset, {exact_99['declaration_covered_count']} rows are covered: {exact_99['strict_complete_covered_count']} strict-complete and {exact_99['partial_covered_count']} partial.

The non-tuning sensitivity is Spearman **{strict['spearman']:.12f}** on the six strict-complete pairs versus **{partial['spearman']:.12f}** on 68 partial pairs. The n=6 result is exploratory only; it does not show that completeness causes higher reconstruction.

Relation-local event licensing is narrower than scalar coverage. On the 99-row primary subset there are three positive placeholder-event witnesses and {events['collision_or_shadowing_positive_event_witness_count']} positive collision/shadowing-event witnesses. Of {events['placeholder_score_1_count']} placeholder score-1 rows, {events['placeholder_score_1_on_partial_count']} are partial; of {events['collision_and_shadowing_score_1_count']} collision/shadowing score-1 rows, {events['collision_and_shadowing_score_1_on_partial_count']} are partial. “No event detected” on partial input is not verified absence. Negative support is licensed only on strict-complete rows: six for placeholder absence and five for collision/shadowing absence.

The sanitizer-changed row, `{changed['item_key']}`, is excluded from the primary subset. As a separate representation-mismatch sensitivity, its historical composite is {changed['historical_composite']:.2f}, code structural aggregate is {changed['code_structural_partial_aggregate']:.12f}, and absolute difference is {changed['absolute_difference']:.12f}.

## Prompt arms and classification

The original V1 raw/hybrid requests were not executed and cannot isolate seam placement because their instructions and relation contracts differ. The pre-reference matched addendum fixes the design—identical rendered prompt, six-relation contract, ctext, and user-object shape, with facts null versus a hash-bound object—but it also was not executed. Consequently all registered raw/hybrid historical and channel comparisons are undefined rather than zero.

Isomorphism gate: construct fidelity **UNAVAILABLE**; exact-input/evidence fidelity **PASS** on the 99-row primary subset; program fidelity **FAIL**; reference-instrument fidelity **UNAVAILABLE**. There was no preregistered categorical success threshold, so the outcome is **DESCRIPTIVE_RECONSTRUCTION_ONLY**. No claim about external truth, code superiority, correctness of disagreement, or tacitness is licensed.

Focused checks: preparation replay passed; conda-interpreter candidate self-test passed; candidate replay matched 100/100; matched-arm tests passed 5/5; primary Spearman and Pearson matched independent repository/stdlib calculations. The frozen shebang remains a portability caveat because this host’s `python3` selects an interpreter without tree-sitter, while the project `python` replays exactly.
"""


def render_artifacts() -> tuple[bytes, bytes]:
    evaluation = build_evaluation()
    evaluation_bytes = (
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    report_bytes = render_report(evaluation).encode("utf-8")
    return evaluation_bytes, report_bytes


def check_artifacts(output_dir: Path = CANONICAL_OUTPUT) -> None:
    expected_json, expected_report = render_artifacts()
    actual_json = (output_dir / "evaluation.json").read_bytes()
    actual_report = (output_dir / "REPORT.md").read_bytes()
    mismatches = []
    if actual_json != expected_json:
        mismatches.append("evaluation.json")
    if actual_report != expected_report:
        mismatches.append("REPORT.md")
    if mismatches:
        raise ValueError(
            "canonical evaluation artifacts differ: " + ", ".join(mismatches)
        )


def emit_artifacts(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    evaluation_bytes, report_bytes = render_artifacts()
    output_dir.mkdir(parents=True)
    try:
        with (output_dir / "evaluation.json").open("xb") as handle:
            handle.write(evaluation_bytes)
        with (output_dir / "REPORT.md").open("xb") as handle:
            handle.write(report_bytes)
    except Exception:
        for path in (output_dir / "evaluation.json", output_dir / "REPORT.md"):
            if path.exists():
                path.unlink()
        output_dir.rmdir()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--emit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if args.check:
        check_artifacts(output_dir)
        print(
            json.dumps(
                {
                    "evaluation_id": EVALUATION_ID,
                    "output_dir": str(output_dir),
                    "status": "PASS",
                },
                sort_keys=True,
            )
        )
    else:
        emit_artifacts(output_dir)
        print(
            json.dumps(
                {
                    "evaluation_id": EVALUATION_ID,
                    "output_dir": str(output_dir),
                    "status": "EMITTED",
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
