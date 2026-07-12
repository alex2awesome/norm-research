"""Manifest validation and evaluation for the technical retrospective replay.

This module deliberately does not run an LLM, train a predictor, or consult an external
label.  It resolves measurements already present in repo artifacts, records their
provenance, and prevents four different objectives from being silently collapsed:

* articulability: prompt/LLM operationalization;
* verifiability: executable code checking a relation;
* isomorphic reconstruction: agreement with the sealed LLM reference instrument; and
* constructive extension: code-native invariants that need not agree with that reference.

The replay is an accounting instrument, not a claim that manually or mock-constructed
decompositions were discovered autonomously.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from ..reconstruction_v2 import (
        AxisEvidence,
        DiscoveryMode,
        PipelineStatus,
        ReconstructionEvidence,
        SelectionMode,
        Status,
        claim_permissions,
        classify,
    )
except ImportError:  # support direct execution through evaluate.py
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reconstruction_v2 import (  # type: ignore[no-redef]
        AxisEvidence,
        DiscoveryMode,
        PipelineStatus,
        ReconstructionEvidence,
        SelectionMode,
        Status,
        claim_permissions,
        classify,
    )


DISCOVERY_MODES = frozenset({"agentic", "manual", "mock", "oracle", "replay"})
OBJECTIVES = (
    "articulability",
    "verifiability",
    "isomorphic_reconstruction",
    "constructive_extension",
)
ASSESSMENTS = frozenset(
    {"not_evaluated", "ineligible", "descriptive", "supported", "unsupported", "mixed"}
)
REFERENCE_ACCESS = frozenset({"sealed", "seen", "unknown", "not_applicable"})
MEASUREMENT_KINDS = frozenset({"scalar", "length", "count_where", "difference"})
PIPELINE_STATUSES = frozenset({"selected", "candidate", "not_selected"})
SELECTION_MODES = frozenset({"blind_agentic", "retrospective_seed", "predeclared"})
DEPTH_LABELS = {
    0: "surface_lexical",
    1: "parsed_document_structure",
    2: "cross_span_or_section_relation",
    3: "formal_solver_or_evidence_graph",
    4: "environment_or_world_execution",
}


class ManifestError(ValueError):
    """Raised when a replay manifest would permit an ambiguous or invalid result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _validate_repo_path(raw: Any, where: str) -> None:
    _require(isinstance(raw, str) and raw, f"{where}: path must be a non-empty string")
    path = Path(raw)
    _require(not path.is_absolute(), f"{where}: path must be repo-relative")
    _require(".." not in path.parts, f"{where}: path may not escape the repo")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate structural and claim-separation invariants.

    Validation is intentionally stricter than a loose JSON schema.  Most importantly,
    every case must represent all four objectives explicitly, every artifact must carry a
    discovery mode, and an isomorphic result must say whether the reference was sealed.
    """

    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest.get("schema_version") == "technical-replay-v2", "unsupported schema")
    _require(manifest.get("external_supervision") == "none", "replay must remain unsupervised")
    definitions = manifest.get("objective_definitions")
    _require(isinstance(definitions, dict), "objective_definitions must be an object")
    _require(set(definitions) == set(OBJECTIVES), "all four objective definitions are required")
    for objective, definition in definitions.items():
        _require(isinstance(definition, str) and definition.strip(), f"empty {objective} definition")

    cases = manifest.get("cases")
    _require(isinstance(cases, list) and cases, "cases must be a non-empty list")
    case_ids: set[str] = set()
    for ci, case in enumerate(cases):
        at = f"cases[{ci}]"
        _require(isinstance(case, dict), f"{at}: case must be an object")
        case_id = case.get("case_id")
        _require(isinstance(case_id, str) and case_id, f"{at}: case_id required")
        _require(case_id not in case_ids, f"{at}: duplicate case_id {case_id}")
        case_ids.add(case_id)
        _require(case.get("discovery_mode") in DISCOVERY_MODES, f"{at}: invalid discovery_mode")
        _require(
            case.get("reference_access_during_discovery") in REFERENCE_ACCESS,
            f"{at}: invalid reference access",
        )
        _require(isinstance(case.get("domain"), str) and case["domain"], f"{at}: domain required")
        lineage = case.get("program_lineage", "metric_seam_active_or_domain_replay")
        _require(
            isinstance(lineage, str) and lineage,
            f"{at}: program_lineage must be a non-empty string when supplied",
        )
        _require(
            isinstance(case.get("relation"), str) and case["relation"], f"{at}: relation required"
        )
        _require(case.get("pipeline_status") in PIPELINE_STATUSES, f"{at}: invalid pipeline_status")
        _require(case.get("selection_mode") in SELECTION_MODES, f"{at}: invalid selection_mode")
        depth = case.get("relation_depth")
        _require(isinstance(depth, dict), f"{at}: relation_depth must be an object")
        level = depth.get("level")
        _require(level in DEPTH_LABELS, f"{at}: relation depth must be an integer from 0 through 4")
        _require(depth.get("label") == DEPTH_LABELS[level], f"{at}: relation-depth label mismatch")
        _require(
            isinstance(depth.get("mechanism"), str) and depth["mechanism"],
            f"{at}: relation-depth mechanism required",
        )

        corpus = case.get("corpus")
        _require(isinstance(corpus, dict), f"{at}: corpus must be an object")
        for key in ("observed_sections", "required_sections", "limitations"):
            _require(isinstance(corpus.get(key), list), f"{at}.corpus.{key} must be a list")
            _require(
                all(isinstance(v, str) and v for v in corpus[key]),
                f"{at}.corpus.{key} must contain strings",
            )

        artifacts = case.get("artifacts")
        _require(isinstance(artifacts, list) and artifacts, f"{at}: artifacts required")
        artifact_ids: set[str] = set()
        for ai, artifact in enumerate(artifacts):
            aw = f"{at}.artifacts[{ai}]"
            _require(isinstance(artifact, dict), f"{aw}: artifact must be an object")
            artifact_id = artifact.get("artifact_id")
            _require(isinstance(artifact_id, str) and artifact_id, f"{aw}: artifact_id required")
            _require(artifact_id not in artifact_ids, f"{aw}: duplicate artifact_id")
            artifact_ids.add(artifact_id)
            _validate_repo_path(artifact.get("path"), aw)
            _require(artifact.get("discovery_mode") in DISCOVERY_MODES, f"{aw}: invalid mode")
            _require(isinstance(artifact.get("role"), str) and artifact["role"], f"{aw}: role required")

        axes = case.get("axes")
        _require(isinstance(axes, dict), f"{at}: axes must be an object")
        _require(set(axes) == set(OBJECTIVES), f"{at}: axes must contain exactly {OBJECTIVES}")
        for objective, axis in axes.items():
            ow = f"{at}.axes.{objective}"
            _require(isinstance(axis, dict), f"{ow}: axis must be an object")
            _require(axis.get("assessment") in ASSESSMENTS, f"{ow}: invalid assessment")
            _require(isinstance(axis.get("claim"), str), f"{ow}: claim must be a string")
            _require(isinstance(axis.get("limitations"), list), f"{ow}: limitations must be a list")
            used = axis.get("artifacts_used")
            _require(isinstance(used, list), f"{ow}: artifacts_used must be a list")
            _require(set(used) <= artifact_ids, f"{ow}: unknown artifact in artifacts_used")
            measurements = axis.get("measurements")
            _require(isinstance(measurements, list), f"{ow}: measurements must be a list")
            mids: set[str] = set()
            for mi, measurement in enumerate(measurements):
                mw = f"{ow}.measurements[{mi}]"
                _require(isinstance(measurement, dict), f"{mw}: measurement must be an object")
                mid = measurement.get("measurement_id")
                _require(isinstance(mid, str) and mid, f"{mw}: measurement_id required")
                _require(mid not in mids, f"{mw}: duplicate measurement_id")
                mids.add(mid)
                kind = measurement.get("kind")
                _require(kind in MEASUREMENT_KINDS, f"{mw}: invalid kind")
                if kind == "difference":
                    _require(
                        measurement.get("left") in mids and measurement.get("right") in mids,
                        f"{mw}: difference operands must precede the derived measurement",
                    )
                else:
                    _require(
                        measurement.get("artifact_id") in artifact_ids,
                        f"{mw}: unknown artifact_id",
                    )
                    _require(isinstance(measurement.get("pointer", ""), str), f"{mw}: pointer required")
                if kind == "count_where":
                    _require(isinstance(measurement.get("field_pointer"), str), f"{mw}: field_pointer required")
                    predicates = [k for k in ("equals", "in", "gt", "ge", "lt", "le") if k in measurement]
                    _require(len(predicates) == 1, f"{mw}: count_where needs exactly one predicate")

        utility = case.get("utility")
        uw = f"{at}.utility"
        _require(isinstance(utility, dict), f"{uw}: utility must be an object")
        _require(utility.get("assessment") in ASSESSMENTS, f"{uw}: invalid assessment")
        _require(isinstance(utility.get("claim"), str), f"{uw}: claim must be a string")
        _require(isinstance(utility.get("limitations"), list), f"{uw}: limitations must be a list")
        used = utility.get("artifacts_used")
        _require(isinstance(used, list), f"{uw}: artifacts_used must be a list")
        _require(set(used) <= artifact_ids, f"{uw}: unknown artifact in artifacts_used")
        measurements = utility.get("measurements")
        _require(isinstance(measurements, list), f"{uw}: measurements must be a list")
        mids: set[str] = set()
        for mi, measurement in enumerate(measurements):
            mw = f"{uw}.measurements[{mi}]"
            _require(isinstance(measurement, dict), f"{mw}: measurement must be an object")
            mid = measurement.get("measurement_id")
            _require(isinstance(mid, str) and mid, f"{mw}: measurement_id required")
            _require(mid not in mids, f"{mw}: duplicate measurement_id")
            mids.add(mid)
            kind = measurement.get("kind")
            _require(kind in MEASUREMENT_KINDS, f"{mw}: invalid kind")
            if kind == "difference":
                _require(
                    measurement.get("left") in mids and measurement.get("right") in mids,
                    f"{mw}: difference operands must precede the derived measurement",
                )
            else:
                _require(measurement.get("artifact_id") in artifact_ids, f"{mw}: unknown artifact_id")
                _require(isinstance(measurement.get("pointer", ""), str), f"{mw}: pointer required")
            if kind == "count_where":
                _require(isinstance(measurement.get("field_pointer"), str), f"{mw}: field_pointer required")
                predicates = [k for k in ("equals", "in", "gt", "ge", "lt", "le") if k in measurement]
                _require(len(predicates) == 1, f"{mw}: count_where needs exactly one predicate")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pointer(value: Any, pointer: str) -> Any:
    """Resolve a small RFC-6901-compatible JSON pointer."""

    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ManifestError(f"JSON pointer must be empty or start with '/': {pointer!r}")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ManifestError(f"pointer {pointer!r} missing object key {token!r}")
            current = current[token]
        elif isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as exc:
                raise ManifestError(f"pointer {pointer!r} has invalid list index {token!r}") from exc
        else:
            raise ManifestError(f"pointer {pointer!r} traverses a scalar at {token!r}")
    return current


def _members(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        return value.values()
    if isinstance(value, list):
        return value
    raise ManifestError("length/count_where target must be an object or array")


def _predicate_matches(value: Any, spec: dict[str, Any]) -> bool:
    if "equals" in spec:
        return value == spec["equals"]
    if "in" in spec:
        return value in spec["in"]
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    if "gt" in spec:
        return value > spec["gt"]
    if "ge" in spec:
        return value >= spec["ge"]
    if "lt" in spec:
        return value < spec["lt"]
    if "le" in spec:
        return value <= spec["le"]
    raise ManifestError("count_where predicate missing")


def _resolve_measurement(
    spec: dict[str, Any], artifact_data: dict[str, Any], resolved: dict[str, Any]
) -> Any:
    kind = spec["kind"]
    if kind == "difference":
        left, right = resolved[spec["left"]], resolved[spec["right"]]
        if isinstance(left, bool) or isinstance(right, bool):
            raise ManifestError("difference operands must be numeric, not boolean")
        return left - right
    selected = resolve_pointer(artifact_data[spec["artifact_id"]], spec.get("pointer", ""))
    if kind == "scalar":
        if isinstance(selected, float) and not math.isfinite(selected):
            return None
        if not isinstance(selected, (str, int, float, bool)) and selected is not None:
            raise ManifestError(f"scalar measurement resolved to {type(selected).__name__}")
        return selected
    if kind == "length":
        return len(list(_members(selected)))
    if kind == "count_where":
        count = 0
        for member in _members(selected):
            try:
                value = resolve_pointer(member, spec["field_pointer"])
            except ManifestError:
                continue
            count += int(_predicate_matches(value, spec))
        return count
    raise ManifestError(f"unimplemented measurement kind {kind}")


def _axis_permissions(case: dict[str, Any], axis: dict[str, Any], modes: set[str]) -> dict[str, bool]:
    objective = axis["objective"]
    corpus_eligible = case["corpus_eligibility"]["eligible"]
    return {
        "may_claim_historical_automatic_selection": case["selection_mode"] == "blind_agentic",
        "may_claim_selected_pipeline_result": (
            case["pipeline_status"] == "selected" and corpus_eligible
        ),
        "may_claim_confirmatory_isomorphic_reconstruction": (
            objective == "isomorphic_reconstruction"
            and axis["assessment"] == "supported"
            and corpus_eligible
            and case["reference_access_during_discovery"] == "sealed"
            and case["selection_mode"] == "blind_agentic"
            and not ({"mock", "oracle"} & modes)
        ),
        "may_claim_provenance_conditioned_constructive_extension": (
            objective == "constructive_extension"
            and axis["assessment"] == "supported"
            and corpus_eligible
            and case["pipeline_status"] == "selected"
        ),
        "may_claim_unconditioned_constructive_extension": (
            objective == "constructive_extension"
            and axis["assessment"] == "supported"
            and corpus_eligible
            and not ({"mock", "oracle"} & modes)
        ),
    }


def _canonical_status(assessment: str) -> Status:
    """Map replay assessments into the canonical three-valued v2 vocabulary."""

    if assessment == "supported":
        return Status.PASS
    if assessment == "unsupported":
        return Status.FAIL
    return Status.UNAVAILABLE


def _canonical_record(case: dict[str, Any]) -> dict[str, Any]:
    """Emit a conservative record compatible with ``reconstruction_v2.py``.

    Mixed and descriptive historical evidence intentionally maps to UNAVAILABLE rather
    than being promoted to PASS.  Constructive extension is always false in this replay:
    none of the selected artifacts freezes a verifier-adjudicated disagreement set.
    """

    axes = case["axes"]

    def evidence(objective: str) -> AxisEvidence:
        axis = axes[objective]
        return AxisEvidence(
            status=_canonical_status(axis["assessment"]),
            metric="retrospective_replay_assessment",
            note=axis["claim"],
        )

    iso = evidence("isomorphic_reconstruction")
    record = ReconstructionEvidence(
        criterion_id=case["case_id"],
        relation_id=case["relation"],
        discovery_mode=DiscoveryMode(case["discovery_mode"]),
        pipeline_status=PipelineStatus(case["pipeline_status"]),
        selection_mode=SelectionMode(case["selection_mode"]),
        articulability=evidence("articulability"),
        verifiability=evidence("verifiability"),
        hybrid=iso,
        reference_isomorphism=iso,
        construct_fidelity=AxisEvidence(
            status=_canonical_status(axes["verifiability"]["assessment"]),
            metric="retrospective_relation_certificate",
            note="Descriptive/mixed replay evidence is unavailable for a canonical PASS.",
        ),
        verified_reference_disagreement=False,
        verifier_certificate=None,
        provenance_note=(
            "Retrospective provenance is preserved; this record does not assert blind "
            "agentic discovery."
        ),
    )
    payload = record.as_dict()
    payload["outcome"] = classify(record).value
    payload["claim_permissions"] = claim_permissions(record)
    return payload


def evaluate_manifest(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Resolve a validated manifest into a provenance- and claim-bounded result."""

    validate_manifest(manifest)
    result_cases: list[dict[str, Any]] = []
    all_modes: Counter[str] = Counter()

    for source_case in manifest["cases"]:
        artifacts: dict[str, dict[str, Any]] = {}
        artifact_data: dict[str, Any] = {}
        for source_artifact in source_case["artifacts"]:
            artifact = dict(source_artifact)
            path = (repo_root / artifact["path"]).resolve()
            try:
                path.relative_to(repo_root.resolve())
            except ValueError as exc:
                raise ManifestError(f"artifact escapes repo: {artifact['path']}") from exc
            if not path.is_file():
                raise ManifestError(f"artifact does not exist: {artifact['path']}")
            artifact["sha256"] = _sha256(path)
            artifact["bytes"] = path.stat().st_size
            artifacts[artifact["artifact_id"]] = artifact
            all_modes[artifact["discovery_mode"]] += 1
            if path.suffix == ".json":
                with path.open() as handle:
                    artifact_data[artifact["artifact_id"]] = json.load(handle)

        required = set(source_case["corpus"]["required_sections"])
        observed = set(source_case["corpus"]["observed_sections"])
        missing = sorted(required - observed)
        case = {
            "case_id": source_case["case_id"],
            "domain": source_case["domain"],
            "program_lineage": source_case.get(
                "program_lineage", "metric_seam_active_or_domain_replay"
            ),
            "relation": source_case["relation"],
            "discovery_mode": source_case["discovery_mode"],
            "pipeline_status": source_case["pipeline_status"],
            "selection_mode": source_case["selection_mode"],
            "relation_depth": source_case["relation_depth"],
            "reference_access_during_discovery": source_case["reference_access_during_discovery"],
            "corpus_eligibility": {
                "eligible": not missing,
                "missing_sections": missing,
                "limitations": source_case["corpus"]["limitations"],
            },
            "artifacts": list(artifacts.values()),
            "axes": {},
        }

        for objective in OBJECTIVES:
            source_axis = source_case["axes"][objective]
            resolved: dict[str, Any] = {}
            for spec in source_axis["measurements"]:
                resolved[spec["measurement_id"]] = _resolve_measurement(spec, artifact_data, resolved)
            modes = {artifacts[a]["discovery_mode"] for a in source_axis["artifacts_used"]}
            axis = {
                "objective": objective,
                "assessment": source_axis["assessment"],
                "claim": source_axis["claim"],
                "measurements": resolved,
                "limitations": source_axis["limitations"],
                "provenance_modes": sorted(modes),
            }
            if missing and axis["assessment"] not in {"not_evaluated", "ineligible"}:
                axis["assessment"] = "ineligible"
                axis["claim"] = "Corpus requirements are not met; the declared result is not evaluated."
                axis["limitations"] = axis["limitations"] + [f"Missing corpus sections: {', '.join(missing)}"]
            axis["claim_permissions"] = _axis_permissions(case, axis, modes)
            case["axes"][objective] = axis
        source_utility = source_case["utility"]
        utility_values: dict[str, Any] = {}
        for spec in source_utility["measurements"]:
            utility_values[spec["measurement_id"]] = _resolve_measurement(
                spec, artifact_data, utility_values
            )
        utility_modes = {
            artifacts[artifact_id]["discovery_mode"]
            for artifact_id in source_utility["artifacts_used"]
        }
        utility_assessment = source_utility["assessment"]
        utility_claim = source_utility["claim"]
        utility_limitations = list(source_utility["limitations"])
        if missing and utility_assessment not in {"not_evaluated", "ineligible"}:
            utility_assessment = "ineligible"
            utility_claim = "Corpus requirements are not met; selected-pipeline utility is unavailable."
            utility_limitations.append(f"Missing corpus sections: {', '.join(missing)}")
        case["utility"] = {
            "assessment": utility_assessment,
            "claim": utility_claim,
            "measurements": utility_values,
            "limitations": utility_limitations,
            "provenance_modes": sorted(utility_modes),
            "may_claim_selected_pipeline_utility": (
                source_case["pipeline_status"] == "selected"
                and not missing
                and utility_assessment in {"supported", "mixed", "descriptive"}
            ),
        }
        case["canonical_v2"] = _canonical_record(case)
        result_cases.append(case)

    return {
        "schema_version": manifest["schema_version"],
        "experiment_id": manifest["experiment_id"],
        "external_supervision": manifest["external_supervision"],
        "objective_definitions": manifest["objective_definitions"],
        "summary": {
            "n_cases": len(result_cases),
            "domains": sorted({case["domain"] for case in result_cases}),
            "artifact_discovery_modes": dict(sorted(all_modes.items())),
            "n_corpus_eligible": sum(case["corpus_eligibility"]["eligible"] for case in result_cases),
            "n_selected_pipeline_cases": sum(
                case["pipeline_status"] == "selected" for case in result_cases
            ),
            "relation_depth_counts": dict(
                sorted(Counter(case["relation_depth"]["level"] for case in result_cases).items())
            ),
            "n_selected_utility_claims_permitted": sum(
                case["utility"]["may_claim_selected_pipeline_utility"] for case in result_cases
            ),
            "n_automatic_decomposition_claims_permitted": sum(
                case["selection_mode"] == "blind_agentic" for case in result_cases
            ),
            "n_confirmatory_isomorphic_claims_permitted": sum(
                axis["claim_permissions"]["may_claim_confirmatory_isomorphic_reconstruction"]
                for case in result_cases
                for axis in case["axes"].values()
            ),
            "n_unconditioned_extension_claims_permitted": sum(
                axis["claim_permissions"]["may_claim_unconditioned_constructive_extension"]
                for case in result_cases
                for axis in case["axes"].values()
            ),
            "n_provenance_conditioned_extension_claims_permitted": sum(
                axis["claim_permissions"][
                    "may_claim_provenance_conditioned_constructive_extension"
                ]
                for case in result_cases
                for axis in case["axes"].values()
            ),
            "n_canonical_code_verifiability_claims_permitted": sum(
                case["canonical_v2"]["claim_permissions"]["may_claim_code_verifiability"]
                for case in result_cases
            ),
            "n_canonical_constructive_extension_claims_permitted": sum(
                case["canonical_v2"]["claim_permissions"][
                    "may_claim_constructive_extension"
                ]
                for case in result_cases
            ),
            "n_tacitness_claims_permitted": sum(
                case["canonical_v2"]["claim_permissions"]["may_claim_tacitness"]
                for case in result_cases
            ),
        },
        "cases": result_cases,
    }
