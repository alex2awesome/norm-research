"""Normalize real metric-seam artifacts into one typed technical-entry receipt.

This module is an additive integration boundary.  It does not certify a batch, alter
``certify_batch_v2.py``, or turn agreement with a frozen LLM reference into external
ground truth.  Its purpose is narrower: make the artifacts emitted by the blind/sealed
lane explicit and mutually checkable before a later, separately preregistered batch
analysis consumes them.

The receipt keeps three questions separate:

* coverage: which registered held-out items received finite candidate/reference values;
* construct fidelity: which channel-faithful contract and independent-adversary gates pass;
* reconstruction: how closely the candidate orders the frozen LLM reference on common
  support, including a declared absolute-correlation floor.

Code-native certificates and executable relation depth are additional planes.  Missing
certificate or depth artifacts are ``unavailable``, never zero.  Hashes appear only where
they bind artifacts or identities; they are not scientific outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "metric-seam.technical-entry.v1"
DEPTH_SCHEMA_VERSION = "metric-seam.relation-depth.v1"
CERTIFICATE_SCHEMA_VERSION = "metric-seam.certificate-summary.v1"
DEPTH_SCALE = {
    0: "surface_lexical",
    1: "parsed_document_structure",
    2: "cross_span_or_cross_section_relation",
    3: "formal_solver_or_evidence_graph_execution",
    4: "environment_or_world_execution",
}


class TechnicalEntryError(ValueError):
    """Base class for a malformed or mutually inconsistent technical entry."""


class BindingError(TechnicalEntryError):
    """Two artifacts purporting to describe one entry do not share an identity."""


class SchemaError(TechnicalEntryError):
    """An artifact cannot be interpreted without guessing."""


class EvidenceStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PreflightConfig:
    """Thresholds used only for preflight eligibility and descriptive floor fields."""

    candidate_coverage_min: float = 0.90
    reference_availability_min: float = 0.50
    common_given_reference_min: float = 0.90
    minimum_common_pairs: int = 20
    absolute_rho_min: float = 0.30
    require_discrimination_gate: bool = True

    def validate(self) -> None:
        for name in (
            "candidate_coverage_min",
            "reference_availability_min",
            "common_given_reference_min",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SchemaError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise SchemaError(f"{name} must be finite and in [0, 1]")
        if (
            isinstance(self.minimum_common_pairs, bool)
            or not isinstance(self.minimum_common_pairs, int)
            or self.minimum_common_pairs < 3
        ):
            raise SchemaError("minimum_common_pairs must be an integer >= 3")
        if (
            isinstance(self.absolute_rho_min, bool)
            or not isinstance(self.absolute_rho_min, (int, float))
            or not math.isfinite(float(self.absolute_rho_min))
            or not -1.0 <= float(self.absolute_rho_min) <= 1.0
        ):
            raise SchemaError("absolute_rho_min must be finite and in [-1, 1]")


@dataclass(frozen=True)
class LoadedArtifact:
    value: Mapping[str, Any]
    path: Path | None
    file_sha256: str | None


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"value is not canonicalizable JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(value: Mapping[str, Any] | Path | str, label: str) -> LoadedArtifact:
    if isinstance(value, Mapping):
        return LoadedArtifact(value=value, path=None, file_sha256=None)
    path = Path(value).resolve()
    if not path.is_file():
        raise SchemaError(f"{label} does not exist: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise SchemaError(f"{label} must contain a JSON object")
    return LoadedArtifact(value=parsed, path=path, file_sha256=_file_sha256(path))


def _optional_load(
    value: Mapping[str, Any] | Path | str | None, label: str
) -> LoadedArtifact | None:
    return _load(value, label) if value is not None else None


def _expected_task_aspect(criterion_id: str) -> tuple[str, str]:
    if not isinstance(criterion_id, str) or "__" not in criterion_id:
        raise SchemaError("criterion_id must have the form task__aspect")
    task, aspect = criterion_id.rsplit("__", 1)
    if not task or not aspect:
        raise SchemaError("criterion_id must have non-empty task and aspect components")
    return task, aspect


def _identity_candidates(payload: Mapping[str, Any], key: str) -> list[Any]:
    values: list[Any] = []
    if key in payload:
        values.append(payload[key])
    for container_name in ("bindings", "identity", "entry"):
        container = payload.get(container_name)
        if isinstance(container, Mapping) and key in container:
            values.append(container[key])
    return values


def _bind_identity(
    payload: Mapping[str, Any], *, criterion_id: str, relation_id: str, label: str
) -> None:
    task, aspect = _expected_task_aspect(criterion_id)
    for observed in _identity_candidates(payload, "criterion_id"):
        if observed != criterion_id:
            raise BindingError(
                f"{label} criterion_id mismatch: expected {criterion_id!r}, got {observed!r}"
            )
    for observed in _identity_candidates(payload, "relation_id"):
        if observed != relation_id:
            raise BindingError(
                f"{label} relation_id mismatch: expected {relation_id!r}, got {observed!r}"
            )
    for observed in _identity_candidates(payload, "task"):
        if observed != task:
            raise BindingError(
                f"{label} task mismatch: expected {task!r}, got {observed!r}"
            )
    for observed in _identity_candidates(payload, "aspect_id"):
        if observed != aspect:
            raise BindingError(
                f"{label} aspect_id mismatch: expected {aspect!r}, got {observed!r}"
            )


def _artifact_manifest_record(
    sealed_manifest: Mapping[str, Any], artifact_name: str
) -> Mapping[str, Any] | None:
    artifacts = sealed_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    record = artifacts.get(artifact_name)
    return record if isinstance(record, Mapping) else None


def _bind_file_to_sealed_manifest(
    artifact: LoadedArtifact,
    *,
    sealed_manifest: Mapping[str, Any],
    artifact_name: str,
    label: str,
) -> None:
    if artifact.file_sha256 is None:
        return
    record = _artifact_manifest_record(sealed_manifest, artifact_name)
    if record is None:
        raise BindingError(f"sealed manifest does not bind {artifact_name}")
    expected = record.get("sha256")
    if expected != artifact.file_sha256:
        raise BindingError(
            f"{label} file digest differs from the sealed manifest: "
            f"expected {expected!r}, got {artifact.file_sha256!r}"
        )


def _extract_candidate_hashes(payload: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for value in _identity_candidates(payload, "candidate_sha256"):
        if isinstance(value, str):
            found.add(value)
    candidate = payload.get("candidate")
    if isinstance(candidate, Mapping) and isinstance(candidate.get("sha256"), str):
        found.add(candidate["sha256"])
    verified = payload.get("verified_sha256")
    if isinstance(verified, Mapping):
        for path, value in verified.items():
            name = str(path).lower()
            if "candidate" in name and isinstance(value, str):
                found.add(value)
    return found


def _bind_candidate(
    payload: Mapping[str, Any], *, candidate_sha256: str, label: str
) -> None:
    observed = _extract_candidate_hashes(payload)
    if observed and candidate_sha256 not in observed:
        raise BindingError(
            f"{label} is bound to a different candidate: expected {candidate_sha256}, "
            f"observed {sorted(observed)}"
        )


def _extract_universe_hashes(payload: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for key in ("heldout_universe_sha256", "universe_sha256"):
        for value in _identity_candidates(payload, key):
            if isinstance(value, str):
                found.add(value)
    return found


def _bind_universe(
    payload: Mapping[str, Any], *, universe_sha256: str, label: str
) -> None:
    observed = _extract_universe_hashes(payload)
    if observed and observed != {universe_sha256}:
        raise BindingError(
            f"{label} held-out universe mismatch: expected {universe_sha256}, "
            f"observed {sorted(observed)}"
        )


def _raw_score_map(payload: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    for key in ("score_map", "scores"):
        raw = payload.get(key)
        if isinstance(raw, Mapping):
            return raw
    # Bare maps remain useful for small normalized fixtures, but metadata-bearing objects
    # must never be guessed to be score maps.
    if payload and all(isinstance(key, str) for key in payload) and all(
        value is None or (
            not isinstance(value, bool) and isinstance(value, (int, float))
        )
        for value in payload.values()
    ):
        return payload
    raise SchemaError(f"{label} must contain score_map or scores")


def _score_map(payload: Mapping[str, Any], label: str) -> dict[str, float | None]:
    raw = _raw_score_map(payload, label)
    if not raw:
        raise SchemaError(f"{label} score map must not be empty")
    out: dict[str, float | None] = {}
    for item_id, value in raw.items():
        if not isinstance(item_id, str) or not item_id:
            raise SchemaError(f"{label} contains an invalid item identifier")
        if value is None:
            out[item_id] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"{label}[{item_id!r}] must be numeric or null")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise SchemaError(f"{label}[{item_id!r}] must be finite and in [0, 1]")
        out[item_id] = number
    return out


def _finite_scores(scores: Mapping[str, float | None]) -> dict[str, float]:
    return {key: value for key, value in scores.items() if value is not None}


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        for offset in range(start, end + 1):
            result[order[offset]] = rank
        start = end + 1
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    dl = [value - mean_left for value in left]
    dr = [value - mean_right for value in right]
    denominator = math.sqrt(sum(x * x for x in dl) * sum(y * y for y in dr))
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(dl, dr)) / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _evidence_status(raw_status: Any, raw_passed: Any = None) -> EvidenceStatus:
    status = str(raw_status or "").upper()
    if status in {"PASS", "ACCEPT", "ACCEPTED"} and raw_passed is not False:
        return EvidenceStatus.PASS
    if status in {"FAIL", "REJECT", "REJECTED"} or raw_passed is False:
        return EvidenceStatus.FAIL
    return EvidenceStatus.UNAVAILABLE


def normalize_contract_check(
    payload: Mapping[str, Any] | None,
    *,
    candidate_channel: str,
    expected_contract_canonical_sha256: str | None,
) -> dict[str, Any]:
    """Select the channel-appropriate gate without relabeling authored probes."""

    if payload is None:
        return {
            "status": EvidenceStatus.UNAVAILABLE.value,
            "selected_gate": None,
            "selected_gate_result": None,
            "discrimination_gate": None,
            "reason": "no channel-faithful contract check supplied",
        }
    route = {
        "code": "code_gate",
        "hybrid": "hybrid_gate",
        "hybrid code+llm": "hybrid_gate",
        "hybrid_code_llm": "hybrid_gate",
    }
    selected_name = route.get(candidate_channel.lower())
    if selected_name is None:
        raise SchemaError(
            "candidate_channel must be code or hybrid for a technical contract check"
        )
    selected = payload.get(selected_name)
    if not isinstance(selected, Mapping):
        raise SchemaError(f"contract check is missing {selected_name}")
    observed_contract = payload.get("contract_sha256")
    if (
        expected_contract_canonical_sha256 is not None
        and isinstance(observed_contract, str)
        and observed_contract != expected_contract_canonical_sha256
    ):
        raise BindingError(
            "contract check is bound to a different canonical contract: "
            f"expected {expected_contract_canonical_sha256}, got {observed_contract}"
        )
    selected_status = _evidence_status(selected.get("status"), selected.get("passed"))
    discrimination = payload.get("discrimination_gate")
    discrimination_status = (
        _evidence_status(discrimination.get("status"), discrimination.get("passed"))
        if isinstance(discrimination, Mapping)
        else EvidenceStatus.UNAVAILABLE
    )
    return {
        "status": selected_status.value,
        "selected_gate": selected_name,
        "selected_gate_result": dict(selected),
        "discrimination_status": discrimination_status.value,
        "discrimination_gate": dict(discrimination) if isinstance(discrimination, Mapping) else None,
        "contract_sha256": observed_contract,
        "reason": (
            "channel-faithful gate selected without changing frozen probe channel labels"
        ),
    }


def normalize_adversary(
    payload: Mapping[str, Any] | None,
    *,
    candidate_sha256: str,
    criterion_id: str,
    relation_id: str,
    universe_sha256: str,
) -> dict[str, Any]:
    """Normalize generic, a144-style, and a216-style adversary result schemas."""

    if payload is None:
        return {
            "status": EvidenceStatus.UNAVAILABLE.value,
            "schema_family": None,
            "reason": "no independent adversary result supplied",
        }
    _bind_identity(payload, criterion_id=criterion_id, relation_id=relation_id, label="adversary")
    _bind_candidate(payload, candidate_sha256=candidate_sha256, label="adversary")
    _bind_universe(payload, universe_sha256=universe_sha256, label="adversary")

    if "suite_pass" in payload:
        suite_pass = payload["suite_pass"]
        if not isinstance(suite_pass, bool):
            raise SchemaError("adversary suite_pass must be boolean")
        status = EvidenceStatus.PASS if suite_pass else EvidenceStatus.FAIL
        schema_family = "suite_pass"
    elif "decision" in payload:
        status = _evidence_status(payload.get("decision"))
        schema_family = "decision"
    elif "verdict" in payload:
        status = _evidence_status(payload.get("verdict"))
        schema_family = "verdict"
    else:
        status = EvidenceStatus.UNAVAILABLE
        schema_family = "unknown"

    failed_integrity = []
    for key in ("integrity_ok", "validity_pass", "freeze_verified"):
        if key in payload:
            if not isinstance(payload[key], bool):
                raise SchemaError(f"adversary {key} must be boolean")
            if payload[key] is False:
                failed_integrity.append(key)
    if failed_integrity:
        status = EvidenceStatus.FAIL

    summary = payload.get("summary")
    metrics = payload.get("metrics")
    return {
        "status": status.value,
        "schema_family": schema_family,
        "raw_decision": payload.get("decision", payload.get("verdict", payload.get("suite_pass"))),
        "failed_integrity_fields": failed_integrity,
        "summary": dict(summary) if isinstance(summary, Mapping) else None,
        "metrics": dict(metrics) if isinstance(metrics, Mapping) else None,
        "candidate_binding_observed": bool(_extract_candidate_hashes(payload)),
        "reason": (
            "independent adversary accepted"
            if status is EvidenceStatus.PASS
            else "independent adversary rejected or failed integrity"
            if status is EvidenceStatus.FAIL
            else "adversary schema did not provide a decision"
        ),
    }


def _combine_construct_fidelity(
    contract: Mapping[str, Any], adversary: Mapping[str, Any], *, require_discrimination: bool
) -> tuple[EvidenceStatus, list[str]]:
    reasons: list[str] = []
    components = [EvidenceStatus(contract["status"]), EvidenceStatus(adversary["status"])]
    if require_discrimination:
        discrimination = EvidenceStatus(
            contract.get("discrimination_status", EvidenceStatus.UNAVAILABLE.value)
        )
        components.append(discrimination)
    labels = ["contract_gate", "adversary"] + (["discrimination_gate"] if require_discrimination else [])
    for label, status in zip(labels, components):
        if status is not EvidenceStatus.PASS:
            reasons.append(f"{label}_{status.value}")
    if EvidenceStatus.FAIL in components:
        return EvidenceStatus.FAIL, reasons
    if all(status is EvidenceStatus.PASS for status in components):
        return EvidenceStatus.PASS, reasons
    return EvidenceStatus.UNAVAILABLE, reasons


def normalize_certificates(
    payload: Mapping[str, Any] | None,
    *,
    heldout_count: int,
    candidate_sha256: str,
    criterion_id: str,
    relation_id: str,
    universe_sha256: str,
) -> dict[str, Any]:
    if payload is None:
        return {
            "status": EvidenceStatus.UNAVAILABLE.value,
            "schema": None,
            "verified_positive_n": None,
            "verified_absence_n": None,
            "abstain_n": None,
            "error_n": None,
            "unclassified_n": None,
            "certificate_coverage": None,
            "reason": "no certificate summary supplied",
        }
    _bind_identity(payload, criterion_id=criterion_id, relation_id=relation_id, label="certificate")
    _bind_candidate(payload, candidate_sha256=candidate_sha256, label="certificate")
    _bind_universe(payload, universe_sha256=universe_sha256, label="certificate")
    schema = payload.get("schema") or payload.get("schema_version")
    if schema not in (None, CERTIFICATE_SCHEMA_VERSION):
        raise SchemaError(f"unsupported certificate summary schema: {schema!r}")
    counts = payload.get("counts", payload.get("certificate_counts"))
    if not isinstance(counts, Mapping):
        raise SchemaError("certificate summary must contain counts")

    aliases = {
        "verified_positive_n": ("verified_positive", "verified_positive_n", "positive"),
        "verified_absence_n": ("verified_absence", "verified_absence_n", "absence"),
        "abstain_n": ("abstain", "abstain_n"),
        "error_n": ("error", "errors", "error_n"),
    }
    normalized: dict[str, int] = {}
    for output_name, names in aliases.items():
        present = [counts[name] for name in names if name in counts]
        if not present:
            raise SchemaError(
                f"certificate summary must explicitly declare {output_name}; "
                "missing is not the same as zero"
            )
        if len(set(present)) != 1:
            raise SchemaError(f"conflicting certificate count aliases for {output_name}")
        value = present[0]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError(f"certificate count {output_name} must be a non-negative integer")
        normalized[output_name] = value
    classified = sum(normalized.values())
    if classified > heldout_count:
        raise SchemaError("certificate item counts exceed the registered held-out universe")
    unclassified = heldout_count - classified
    replay_verified = payload.get("replay_verified")
    if replay_verified is not None and not isinstance(replay_verified, bool):
        raise SchemaError("certificate replay_verified must be boolean or null")
    return {
        "status": "available",
        "schema": schema or CERTIFICATE_SCHEMA_VERSION,
        **normalized,
        "unclassified_n": unclassified,
        "replay_verified": replay_verified,
        "certificate_coverage": (
            normalized["verified_positive_n"] + normalized["verified_absence_n"]
        )
        / heldout_count,
        "reason": "relation-local certificate item outcomes; no missed-event completeness inferred",
    }


def normalize_depth(
    payload: Mapping[str, Any] | None,
    *,
    heldout_count: int,
    candidate_sha256: str,
    criterion_id: str,
    relation_id: str,
    universe_sha256: str,
) -> dict[str, Any]:
    if payload is None:
        return {
            "status": EvidenceStatus.UNAVAILABLE.value,
            "scale": DEPTH_SCHEMA_VERSION,
            "static_max_relation_depth": None,
            "dynamic_contributing_depth_histogram": None,
            "reason": "no audited relation-depth profile supplied",
        }
    _bind_identity(payload, criterion_id=criterion_id, relation_id=relation_id, label="depth")
    _bind_candidate(payload, candidate_sha256=candidate_sha256, label="depth")
    _bind_universe(payload, universe_sha256=universe_sha256, label="depth")
    scale = payload.get("scale", payload.get("schema"))
    if scale != DEPTH_SCHEMA_VERSION:
        raise SchemaError(f"depth scale must be {DEPTH_SCHEMA_VERSION!r}")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise SchemaError("depth profile nodes must be a non-empty list")
    seen: set[str] = set()
    normalized_nodes = []
    contributing_depths = []
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise SchemaError(f"depth node {index} must be an object")
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id or node_id in seen:
            raise SchemaError("depth node ids must be unique non-empty strings")
        seen.add(node_id)
        implementation = node.get("implementation", "code")
        if implementation not in {"code", "prompt", "aggregation"}:
            raise SchemaError(f"depth node {node_id} has invalid implementation")
        depth = node.get("relation_depth")
        if implementation == "code":
            if isinstance(depth, bool) or not isinstance(depth, int) or depth not in DEPTH_SCALE:
                raise SchemaError(
                    f"code depth node {node_id} needs integer relation_depth in [0, 4]"
                )
        elif depth is not None and (
            isinstance(depth, bool) or not isinstance(depth, int) or depth not in DEPTH_SCALE
        ):
            raise SchemaError(f"depth node {node_id} relation_depth must be null or in [0, 4]")
        contributes = node.get("contributes_to_output")
        if not isinstance(contributes, bool):
            raise SchemaError(f"depth node {node_id} contributes_to_output must be boolean")
        if implementation == "code" and contributes:
            contributing_depths.append(depth)
        normalized_nodes.append(
            {
                "node_id": node_id,
                "implementation": implementation,
                "relation_depth": depth,
                "relation_depth_label": DEPTH_SCALE.get(depth),
                "contributes_to_output": contributes,
            }
        )
    static_max = max(contributing_depths) if contributing_depths else None
    declared_max = payload.get("static_max_relation_depth")
    if declared_max is not None and declared_max != static_max:
        raise BindingError(
            f"declared static max relation depth {declared_max!r} != computed {static_max!r}"
        )
    longest_path = payload.get("longest_path_edges")
    if longest_path is not None and (
        isinstance(longest_path, bool) or not isinstance(longest_path, int) or longest_path < 0
    ):
        raise SchemaError("longest_path_edges must be a non-negative integer or null")

    histogram_raw = payload.get("dynamic_contributing_depth_histogram")
    histogram = None
    if histogram_raw is not None:
        if not isinstance(histogram_raw, Mapping):
            raise SchemaError("dynamic contributing depth histogram must be an object")
        histogram = {str(level): 0 for level in DEPTH_SCALE}
        for key, value in histogram_raw.items():
            try:
                level = int(key)
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"invalid dynamic depth level {key!r}") from exc
            if level not in DEPTH_SCALE:
                raise SchemaError(f"dynamic depth level must lie in [0, 4], got {level}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaError("dynamic depth histogram counts must be non-negative integers")
            histogram[str(level)] = value
        if sum(histogram.values()) != heldout_count:
            raise SchemaError(
                "dynamic contributing depth histogram must account for every held-out item"
            )
    return {
        "status": EvidenceStatus.PASS.value,
        "scale": DEPTH_SCHEMA_VERSION,
        "scale_labels": {str(key): value for key, value in DEPTH_SCALE.items()},
        "nodes": normalized_nodes,
        "static_max_relation_depth": static_max,
        "longest_path_edges": longest_path,
        "dynamic_contributing_depth_histogram": histogram,
        "reason": (
            "audited relation depth; depth is descriptive and does not establish construct fidelity"
        ),
    }


def _metrics_candidate(metrics: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if metrics is None:
        return None
    candidate = metrics.get("candidate")
    return candidate if isinstance(candidate, Mapping) else None


def _assert_metric_equal(
    observed: Any, expected: Any, label: str, *, tolerance: float = 1e-12
) -> None:
    if observed is None:
        return
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise BindingError(f"sealed metric {label} mismatch: {observed!r} != {expected!r}")
    elif observed != expected:
        raise BindingError(f"sealed metric {label} mismatch: {observed!r} != {expected!r}")


def normalize_technical_entry(
    *,
    criterion_id: str,
    relation_id: str,
    candidate_channel: str,
    sealed_manifest: Mapping[str, Any] | Path | str,
    candidate_scores: Mapping[str, Any] | Path | str,
    frozen_llm_reference: Mapping[str, Any] | Path | str,
    metrics: Mapping[str, Any] | Path | str | None = None,
    contract: Mapping[str, Any] | Path | str | None = None,
    contract_check: Mapping[str, Any] | Path | str | None = None,
    adversary: Mapping[str, Any] | Path | str | None = None,
    comparator_scores: Mapping[str, Any] | Path | str | None = None,
    certificate_summary: Mapping[str, Any] | Path | str | None = None,
    depth_profile: Mapping[str, Any] | Path | str | None = None,
    expected_candidate_sha256: str | None = None,
    config: PreflightConfig = PreflightConfig(),
) -> dict[str, Any]:
    """Return a preflight receipt; no batch-level p-value or claim is produced."""

    config.validate()
    if not isinstance(relation_id, str) or not relation_id:
        raise SchemaError("relation_id must be a non-empty string")
    manifest_artifact = _load(sealed_manifest, "sealed_manifest")
    candidate_artifact = _load(candidate_scores, "candidate_scores")
    reference_artifact = _load(frozen_llm_reference, "frozen_llm_reference")
    metrics_artifact = _optional_load(metrics, "metrics")
    contract_artifact = _optional_load(contract, "contract")
    contract_check_artifact = _optional_load(contract_check, "contract_check")
    adversary_artifact = _optional_load(adversary, "adversary")
    comparator_artifact = _optional_load(comparator_scores, "comparator_scores")
    certificate_artifact = _optional_load(certificate_summary, "certificate_summary")
    depth_artifact = _optional_load(depth_profile, "depth_profile")

    manifest = manifest_artifact.value
    if manifest.get("schema") != "metric-seam.blind-reconstruction.sealed-evaluation-manifest.v2":
        raise SchemaError("unexpected sealed evaluation manifest schema")
    _bind_identity(manifest, criterion_id=criterion_id, relation_id=relation_id, label="manifest")
    policy = manifest.get("policy")
    if isinstance(policy, Mapping):
        required_true = (
            "heldout_exact_complement_reconstructed",
            "candidate_execution_preceded_reference_load",
        )
        for key in required_true:
            if key in policy and policy[key] is not True:
                raise BindingError(f"sealed manifest policy does not establish {key}")
        if policy.get("reference_values_sent_to_candidate") is True:
            raise BindingError("sealed manifest says reference values were sent to candidate")
    for label, artifact in (
        ("candidate_scores", candidate_artifact),
        ("frozen_llm_reference", reference_artifact),
        ("metrics", metrics_artifact),
    ):
        if artifact is not None:
            _bind_identity(artifact.value, criterion_id=criterion_id, relation_id=relation_id, label=label)

    if metrics_artifact is not None:
        observed_channel = metrics_artifact.value.get("candidate_channel")
        channel_equivalents = {
            "code": {"code"},
            "hybrid": {"hybrid", "hybrid code+LLM", "hybrid code+llm", "hybrid_code_llm"},
        }
        expected_channels = channel_equivalents.get(candidate_channel.lower())
        if expected_channels is None:
            raise SchemaError("candidate_channel must be code or hybrid")
        if isinstance(observed_channel, str) and observed_channel not in expected_channels:
            raise BindingError(
                f"metrics candidate channel mismatch: expected {candidate_channel!r}, "
                f"got {observed_channel!r}"
            )
    for label, artifact in (
        ("frozen_llm_reference", reference_artifact),
        ("metrics", metrics_artifact),
    ):
        if artifact is not None and artifact.value.get("external_ground_truth") is True:
            raise BindingError(f"{label} is marked as external ground truth")

    _bind_file_to_sealed_manifest(
        candidate_artifact,
        sealed_manifest=manifest,
        artifact_name="candidate_scores.json",
        label="candidate_scores",
    )
    _bind_file_to_sealed_manifest(
        reference_artifact,
        sealed_manifest=manifest,
        artifact_name="llm_reference_scores.json",
        label="frozen_llm_reference",
    )
    if metrics_artifact is not None:
        _bind_file_to_sealed_manifest(
            metrics_artifact,
            sealed_manifest=manifest,
            artifact_name="metrics.json",
            label="metrics",
        )

    candidate_hash = candidate_artifact.value.get("candidate_sha256")
    if not isinstance(candidate_hash, str) or len(candidate_hash) != 64:
        raise SchemaError("candidate score artifact must bind candidate_sha256")
    if expected_candidate_sha256 is not None and candidate_hash != expected_candidate_sha256:
        raise BindingError(
            f"candidate mismatch: expected {expected_candidate_sha256}, got {candidate_hash}"
        )
    frozen_candidate_record = _artifact_manifest_record(manifest, "candidate_frozen.py")
    if frozen_candidate_record is not None and frozen_candidate_record.get("sha256") != candidate_hash:
        raise BindingError("candidate score artifact differs from sealed candidate bytes")

    candidate_map = _score_map(candidate_artifact.value, "candidate_scores")
    reference_map = _score_map(reference_artifact.value, "frozen_llm_reference")
    comparator_map = (
        _score_map(comparator_artifact.value, "comparator_scores")
        if comparator_artifact is not None
        else None
    )
    partition = manifest.get("partition")
    if not isinstance(partition, Mapping):
        raise SchemaError("sealed manifest must contain partition")
    heldout_count = partition.get("heldout_count")
    if isinstance(heldout_count, bool) or not isinstance(heldout_count, int) or heldout_count <= 0:
        raise SchemaError("sealed partition heldout_count must be a positive integer")
    universe = sorted(candidate_map)
    if len(universe) != heldout_count:
        raise BindingError(
            "candidate score artifact must enumerate the full held-out universe, including nulls: "
            f"expected {heldout_count}, observed {len(universe)}"
        )
    universe_set = set(universe)
    if not set(reference_map).issubset(universe_set):
        raise BindingError("frozen reference contains ids outside candidate held-out universe")
    if comparator_map is not None and not set(comparator_map).issubset(universe_set):
        raise BindingError("comparator contains ids outside candidate held-out universe")
    universe_sha256 = _canonical_sha256(universe)

    for label, artifact in (
        ("candidate_scores", candidate_artifact),
        ("frozen_llm_reference", reference_artifact),
        ("metrics", metrics_artifact),
        ("contract_check", contract_check_artifact),
        ("adversary", adversary_artifact),
        ("comparator_scores", comparator_artifact),
        ("certificate_summary", certificate_artifact),
        ("depth_profile", depth_artifact),
    ):
        if artifact is not None:
            _bind_candidate(artifact.value, candidate_sha256=candidate_hash, label=label)
            _bind_universe(artifact.value, universe_sha256=universe_sha256, label=label)

    contract_file_sha = contract_artifact.file_sha256 if contract_artifact else None
    contract_canonical_sha = (
        _canonical_sha256(contract_artifact.value) if contract_artifact else None
    )
    manifest_contract = manifest.get("inputs", {}).get("contract")
    if contract_artifact is not None and isinstance(manifest_contract, Mapping):
        expected_file_sha = manifest_contract.get("sha256")
        if contract_file_sha is not None and expected_file_sha != contract_file_sha:
            raise BindingError("supplied contract differs from sealed manifest input")

    contract_result = normalize_contract_check(
        contract_check_artifact.value if contract_check_artifact else None,
        candidate_channel=candidate_channel,
        expected_contract_canonical_sha256=contract_canonical_sha,
    )
    adversary_result = normalize_adversary(
        adversary_artifact.value if adversary_artifact else None,
        candidate_sha256=candidate_hash,
        criterion_id=criterion_id,
        relation_id=relation_id,
        universe_sha256=universe_sha256,
    )
    construct_status, construct_reasons = _combine_construct_fidelity(
        contract_result,
        adversary_result,
        require_discrimination=config.require_discrimination_gate,
    )

    candidate_finite = _finite_scores(candidate_map)
    reference_finite = _finite_scores(reference_map)
    comparator_finite = _finite_scores(comparator_map or {})
    common = sorted(set(candidate_finite) & set(reference_finite))
    candidate_coverage = len(candidate_finite) / heldout_count
    reference_availability = len(reference_finite) / heldout_count
    common_given_reference = len(common) / len(reference_finite) if reference_finite else 0.0
    coverage_reasons = []
    if candidate_coverage < config.candidate_coverage_min:
        coverage_reasons.append("candidate_coverage_below_minimum")
    if reference_availability < config.reference_availability_min:
        coverage_reasons.append("reference_availability_below_minimum")
    if common_given_reference < config.common_given_reference_min:
        coverage_reasons.append("common_given_reference_below_minimum")
    if len(common) < config.minimum_common_pairs:
        coverage_reasons.append("common_pairs_below_minimum")
    coverage_status = EvidenceStatus.FAIL if coverage_reasons else EvidenceStatus.PASS

    rho_candidate = _spearman(
        [candidate_finite[item_id] for item_id in common],
        [reference_finite[item_id] for item_id in common],
    )
    absolute_floor_met = (
        rho_candidate >= config.absolute_rho_min if rho_candidate is not None else None
    )
    comparator_common = sorted(
        set(candidate_finite) & set(comparator_finite) & set(reference_finite)
    )
    rho_candidate_on_comparator_support = None
    rho_comparator = None
    delta_spearman = None
    if comparator_artifact is not None:
        rho_candidate_on_comparator_support = _spearman(
            [candidate_finite[item_id] for item_id in comparator_common],
            [reference_finite[item_id] for item_id in comparator_common],
        )
        rho_comparator = _spearman(
            [comparator_finite[item_id] for item_id in comparator_common],
            [reference_finite[item_id] for item_id in comparator_common],
        )
        if rho_candidate_on_comparator_support is not None and rho_comparator is not None:
            delta_spearman = rho_candidate_on_comparator_support - rho_comparator

    metrics_payload = metrics_artifact.value if metrics_artifact else None
    metric_candidate = _metrics_candidate(metrics_payload)
    if metric_candidate is not None:
        _assert_metric_equal(metric_candidate.get("heldout_count"), heldout_count, "heldout_count")
        _assert_metric_equal(
            metric_candidate.get("n_scoreable"), len(candidate_finite), "n_scoreable"
        )
        _assert_metric_equal(
            metric_candidate.get("reference_available_count"),
            len(reference_finite),
            "reference_available_count",
        )
        _assert_metric_equal(metric_candidate.get("common_count"), len(common), "common_count")
        if rho_candidate is not None:
            _assert_metric_equal(
                metric_candidate.get("spearman_reconstruction"),
                rho_candidate,
                "spearman_reconstruction",
            )

    certificate_result = normalize_certificates(
        certificate_artifact.value if certificate_artifact else None,
        heldout_count=heldout_count,
        candidate_sha256=candidate_hash,
        criterion_id=criterion_id,
        relation_id=relation_id,
        universe_sha256=universe_sha256,
    )
    depth_result = normalize_depth(
        depth_artifact.value if depth_artifact else None,
        heldout_count=heldout_count,
        candidate_sha256=candidate_hash,
        criterion_id=criterion_id,
        relation_id=relation_id,
        universe_sha256=universe_sha256,
    )
    eligibility_reasons = list(coverage_reasons)
    if construct_status is not EvidenceStatus.PASS:
        eligibility_reasons.append(f"construct_fidelity_{construct_status.value}")

    task, aspect = _expected_task_aspect(criterion_id)
    return {
        "schema": SCHEMA_VERSION,
        "objective": "unsupervised reconstruction of a frozen prompt/LLM reference",
        "external_ground_truth": False,
        "criterion_id": criterion_id,
        "relation_id": relation_id,
        "task": task,
        "aspect_id": aspect,
        "candidate_channel": candidate_channel,
        "terminology": {
            "articulability": "prompt/LLM implementation",
            "verifiability": "replayable executable/code certificate",
            "reference_reconstruction": "agreement with the frozen LLM reference",
            "isomorphism": "not established by this preflight",
        },
        "bindings": {
            "candidate_sha256": candidate_hash,
            "contract_file_sha256": contract_file_sha,
            "contract_canonical_sha256": contract_canonical_sha,
            "heldout_universe_sha256": universe_sha256,
            "sealed_manifest_sha256": manifest_artifact.file_sha256,
        },
        "config": asdict(config),
        "coverage": {
            "status": coverage_status.value,
            "reasons": coverage_reasons,
            "heldout_n": heldout_count,
            "candidate_enumerated_n": len(candidate_map),
            "candidate_finite_n": len(candidate_finite),
            "candidate_null_n": heldout_count - len(candidate_finite),
            "candidate_fraction": candidate_coverage,
            "reference_enumerated_n": len(reference_map),
            "reference_available_n": len(reference_finite),
            "reference_null_n": len(reference_map) - len(reference_finite),
            "reference_fraction": reference_availability,
            "common_n": len(common),
            "common_given_reference": common_given_reference,
            "comparator_finite_n": (
                len(comparator_finite) if comparator_artifact is not None else None
            ),
            "comparator_common_n": (
                len(comparator_common) if comparator_artifact is not None else None
            ),
        },
        "construct_fidelity": {
            "status": construct_status.value,
            "reasons": construct_reasons,
            "contract": contract_result,
            "adversary": adversary_result,
        },
        "reference_reconstruction": {
            "status": "observed" if rho_candidate is not None else "undefined",
            "common_n": len(common),
            "rho_candidate": rho_candidate,
            "absolute_rho_min": config.absolute_rho_min,
            "absolute_floor_met": absolute_floor_met,
            "comparator_common_n": (
                len(comparator_common) if comparator_artifact is not None else None
            ),
            "rho_candidate_on_comparator_support": rho_candidate_on_comparator_support,
            "rho_comparator": rho_comparator,
            "delta_spearman": delta_spearman,
            "p_value": None,
            "bh_q_value": None,
            "note": "preflight only; multiplicity-aware inference belongs to a frozen batch",
        },
        "certificate_plane": certificate_result,
        "program_depth": depth_result,
        "inferential_preflight": {
            "eligible": not eligibility_reasons,
            "reasons": eligibility_reasons,
            "absolute_floor_met": absolute_floor_met,
            "absolute_floor_is_separate_from_eligibility": True,
        },
        "claim_permissions": {
            "may_claim_confirmatory_batch": False,
            "may_claim_external_ground_truth_accuracy": False,
            "may_claim_isomorphism": False,
            "may_claim_tacitness": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion-id", required=True)
    parser.add_argument("--relation-id", required=True)
    parser.add_argument("--candidate-channel", choices=("code", "hybrid"), required=True)
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--frozen-llm-reference", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-check", type=Path)
    parser.add_argument("--adversary", type=Path)
    parser.add_argument("--comparator-scores", type=Path)
    parser.add_argument("--certificate-summary", type=Path)
    parser.add_argument("--depth-profile", type=Path)
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--candidate-coverage-min", type=float, default=0.90)
    parser.add_argument("--reference-availability-min", type=float, default=0.50)
    parser.add_argument("--common-given-reference-min", type=float, default=0.90)
    parser.add_argument("--minimum-common-pairs", type=int, default=20)
    parser.add_argument("--absolute-rho-min", type=float, default=0.30)
    parser.add_argument("--allow-missing-discrimination", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PreflightConfig(
        candidate_coverage_min=args.candidate_coverage_min,
        reference_availability_min=args.reference_availability_min,
        common_given_reference_min=args.common_given_reference_min,
        minimum_common_pairs=args.minimum_common_pairs,
        absolute_rho_min=args.absolute_rho_min,
        require_discrimination_gate=not args.allow_missing_discrimination,
    )
    receipt = normalize_technical_entry(
        criterion_id=args.criterion_id,
        relation_id=args.relation_id,
        candidate_channel=args.candidate_channel,
        sealed_manifest=args.sealed_manifest,
        candidate_scores=args.candidate_scores,
        frozen_llm_reference=args.frozen_llm_reference,
        metrics=args.metrics,
        contract=args.contract,
        contract_check=args.contract_check,
        adversary=args.adversary,
        comparator_scores=args.comparator_scores,
        certificate_summary=args.certificate_summary,
        depth_profile=args.depth_profile,
        expected_candidate_sha256=args.expected_candidate_sha256,
        config=config,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
