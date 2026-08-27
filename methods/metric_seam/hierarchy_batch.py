"""Prepare source-only R1/R2/R3 metrics for metric-seam program authoring.

The canonical hierarchy cohort is compiled by
``methods.codability.experiments.compile_fresh_name_arm_bank``.  This module
does not select a second panel.  It adds the missing metric-seam layer:

* a label-free compiler brief for every hierarchy metric;
* explicit candidate depth and provenance requirements;
* a readiness ledger that keeps prompt articulability, code verifiability,
  sealed reconstruction, and isomorphism as separate gates.

No prompt result, executable score, outcome label, or external anchor is read
when these artifacts are built.  A compiler brief is an authoring input, not a
successful decomposition or a verifiability result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from methods.codability.experiments.compile_fresh_name_arm_bank import (
    validate_bank,
)
from methods.metric_seam.hierarchy_panel_compat import validate_hierarchy_panel


BRIEF_SCHEMA = "metric-seam.hierarchy-compiler-brief.v1"
READINESS_SCHEMA = "metric-seam.hierarchy-readiness.v1"
COMPLETION_RECEIPT_SCHEMA = "metric-seam.hierarchy-completion-receipt.v1"
TERMINAL_ATTEMPT_RECEIPT_SCHEMA = "metric-seam.hierarchy-terminal-attempt-receipt.v1"
ROOT = Path(__file__).resolve().parents[2]

COMPLETION_ARTIFACT_FIELDS = {
    "candidate": "candidate_path",
    "decomposition": "decomposition_path",
    "depth_record": "depth_record_path",
    "candidate_execution": "candidate_execution_path",
    "construct_fidelity": "construct_fidelity_path",
    "certificate": "certificate_path",
    "frozen_reference": "frozen_reference_path",
    "sealed_evaluation": "sealed_evaluation_path",
    "isomorphism": "isomorphism_path",
}
COMPLETION_GATES = (
    "construct_fidelity",
    "input_same_byte_fidelity",
    "executed_program_fidelity",
    "reference_instrument_fidelity",
    "reference_reconstruction",
    "code_certificate_and_abstention_validity",
    "sealed_evaluation_validated",
)
FIDELITY_GATES = COMPLETION_GATES[:5]
ADJUDICATED_GATE_STATUSES = frozenset({"pass", "fail", "unavailable"})

DEPTH_VOCABULARY = {
    "0": "surface lexical matching",
    "1": "parsed document structure",
    "2": "cross-span or cross-section relation checking",
    "3": "formal solver or evidence-graph execution",
    "4": "environment or world execution",
}

TASK_CAPABILITIES = {
    "code-review": [
        "diff/file parser",
        "language AST or tree-sitter",
        "control/data-flow graph",
        "linter or type checker",
        "sandboxed compile/test execution when a repository snapshot exists",
    ],
    "math-stackexchange": [
        "LaTeX/math-fragment parser",
        "SymPy normalization and equivalence",
        "equation/proof-step dependency graph",
        "counterexample or domain-condition checker",
    ],
    "patents": [
        "claim and antecedent parser",
        "claim-dependency graph",
        "specification-support evidence graph",
        "prior-art retrieval with explicit oracle/non-oracle provenance",
    ],
    "peer-review": [
        "full-paper section parser",
        "claim-to-method/result evidence graph",
        "quantity/statistical-report parser",
        "citation and artifact-link resolver",
    ],
    "press-releases": [
        "sentence/dependency parser",
        "claim-attribution graph",
        "date and quantity normalizer",
        "source-link and article-evidence matcher",
    ],
    "news-homepages": [
        "page/block structure parser",
        "headline-to-story claim matcher",
        "source/attribution graph",
        "date and quantity normalizer",
    ],
    "legal-outcome-prediction": [
        "citation and authority parser",
        "element/exhaustion dependency graph",
        "timeline and date computation",
        "claim-to-record evidence matcher",
    ],
    "notice-and-comment": [
        "rule/comment section parser",
        "issue-to-response evidence graph",
        "citation and statutory-authority parser",
        "quantity, cost, and deadline normalization",
    ],
    "grant-funding": [
        "proposal section and table parser",
        "budget arithmetic and constraint checker",
        "eligibility/deadline rule engine",
        "claim-to-citation evidence graph",
    ],
    "creative-writing": [
        "sentence/discourse graph",
        "dependency and coreference parser",
        "character/event recurrence graph",
        "document-position and structural segmentation",
    ],
    "humor": [
        "sentence/discourse graph",
        "dependency and coreference parser",
        "setup/payoff and refrain recurrence graph",
        "document-position and structural segmentation",
    ],
}

TECHNICAL_FIRST_ORDER = (
    "math-stackexchange",
    "code-review",
    "patents",
    "peer-review",
    "press-releases",
    "news-homepages",
    "legal-outcome-prediction",
    "notice-and-comment",
    "grant-funding",
    "creative-writing",
    "humor",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _component_rows(cell: Mapping, *, maximum: int = 24) -> list[dict]:
    """Project hierarchy components into hypotheses, never channel verdicts."""
    rows = []
    seen = set()
    for source, values in (
        ("immediate_component", cell.get("components", [])),
        ("raw_leaf_example", cell.get("children", [])),
    ):
        for value in values or []:
            if not isinstance(value, Mapping):
                continue
            name = str(value.get("name") or value.get("medoid_name") or "").strip()
            description = str(value.get("description") or "").strip()
            key = " ".join(name.casefold().split())
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "hypothesis_id": f"h{len(rows) + 1:02d}",
                    "source": source,
                    "name": name,
                    "description": description or None,
                    "requested_relation": None,
                    "implemented_relation": None,
                    "channel_hypothesis": "unresolved",
                    "depth_hypothesis": None,
                    "status": "author_must_adjudicate",
                }
            )
            if len(rows) >= maximum:
                return rows
    return rows


def compile_brief(cell: Mapping) -> dict:
    task = str(cell["task"])
    return {
        "schema": BRIEF_SCHEMA,
        "metric": {
            "cell_id": str(cell["id"]),
            "metric_id": str(cell["metric_id"]),
            "task": task,
            "level": str(cell["level"]),
            "bucket": str(cell["bucket"]),
            "name": str(cell["construct"]),
            "description": str(cell["description"]),
            "hierarchy_source": {
                "path": str(cell["source_path"]),
                "node_id": str(cell["node_id"]),
                "source_kind": str(cell["source_kind"]),
                "source_index": int(cell["source_index"]),
            },
        },
        "objective": {
            "name": "unsupervised reconstruction of an articulated hierarchy metric",
            "articulability": "prompt/LLM implementation",
            "verifiability": "executable/code implementation with scoped replayable evidence",
            "reference_reconstruction": "agreement with a later-opened frozen LLM reference",
            "isomorphism": (
                "joint construct, input, executed-program, reference-instrument, and "
                "reference-reconstruction fidelity"
            ),
            "external_supervision": False,
        },
        "compiler_view": {
            "reference_values_available": False,
            "outcome_labels_available": False,
            "heldout_identifiers_available": False,
            "representation": "task ctext/probe text plus explicitly declared evidence only",
        },
        "candidate_subrelations": _component_rows(cell),
        "available_capability_families": TASK_CAPABILITIES[task],
        "program_contract": {
            "score_signature": "score(ctext: str, extracted: dict, ops) -> float | None",
            "required_declarations": [
                "DISCOVERY_MODE",
                "RELATIONS (requested relation, implemented relation, channel, depth)",
                "AGGREGATION_RULE or explicit None",
                "CAPABILITIES_USED",
                "ABSTENTION_CONDITIONS",
            ],
            "depth_vocabulary": DEPTH_VOCABULARY,
            "minimum_nonlexical_requirement": (
                "At least one attempted relation must use depth >=1, or the author must record "
                "bounded non-discovery after checking the declared task capabilities."
            ),
            "negative_result_policy": (
                "Failure means bounded non-discovery within the frozen program class, "
                "capabilities, representation, and budget; it never establishes tacitness."
            ),
        },
        "provenance_policy": {
            "allowed": ["agentic", "manual", "mock", "oracle", "replay"],
            "historical_deep_programs": (
                "may enter as retrospective seeds with original provenance unchanged"
            ),
            "automatic_discovery_claim_requires": "blind agentic construction in this run",
        },
        "evaluation_gates": [
            "source identity frozen",
            "construct/subrelation contract independently checked",
            "candidate source and declarations frozen",
            "candidate executed before reference load",
            "certificate coverage and abstention reported",
            "reference reconstruction reported separately",
            "all five isomorphism checks pass before using the term isomorphic",
        ],
    }


def compile_briefs(panel: Mapping) -> list[dict]:
    errors = validate_hierarchy_panel(panel)
    if errors:
        raise ValueError(f"invalid hierarchy metric panel: {errors}")
    return [compile_brief(cell) for cell in panel["cells"]]


def _arm_counts(bank: Mapping) -> dict[str, int]:
    return {str(cell["id"]): len(cell.get("arms", [])) for cell in bank.get("cells", [])}


def _resolve_local_artifact(raw_path: object, artifact_root: Path) -> tuple[Path | None, str | None]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, None
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (artifact_root / path).resolve()
    try:
        resolved.relative_to(artifact_root.resolve())
    except ValueError:
        return None, f"artifact path escapes the declared root: {raw_path}"
    if not resolved.is_file():
        return None, f"declared artifact is not a local file: {raw_path}"
    return resolved, None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_completion_receipt(
    *,
    program: Mapping,
    cell_id: str,
    panel_content_sha256: str,
    artifact_root: Path,
) -> tuple[bool, str | None, dict[str, str]]:
    """Validate one explicit receipt; path declarations alone are never completion."""

    receipt_path, path_error = _resolve_local_artifact(
        program.get("completion_receipt_path"), artifact_root
    )
    if path_error:
        return False, path_error, {}
    if receipt_path is None:
        return False, None, {}
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"completion receipt is not valid UTF-8 JSON: {exc}", {}
    if not isinstance(receipt, Mapping):
        return False, "completion receipt must be a JSON object", {}
    if receipt.get("schema") != COMPLETION_RECEIPT_SCHEMA:
        return False, "completion receipt has the wrong schema", {}
    if receipt.get("cell_id") != cell_id:
        return False, "completion receipt is bound to another cell", {}
    if receipt.get("panel_content_sha256") != panel_content_sha256:
        return False, "completion receipt is bound to another hierarchy panel", {}
    if receipt.get("status") != "validated_complete":
        return False, "completion receipt does not declare validated_complete", {}
    if receipt.get("external_supervised_target_used") is not False:
        return False, "completion receipt does not preserve the unsupervised objective", {}

    gates = receipt.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(COMPLETION_GATES):
        return False, "completion receipt does not enumerate the exact fidelity gates", {}
    gate_statuses = {}
    for gate in COMPLETION_GATES:
        record = gates.get(gate)
        if not isinstance(record, Mapping):
            return False, f"completion receipt gate {gate} is not an evidence record", {}
        status = record.get("status")
        evidence = record.get("evidence")
        if status not in ADJUDICATED_GATE_STATUSES or not isinstance(evidence, str) or not evidence:
            return False, f"completion receipt gate {gate} lacks status/evidence", {}
        gate_statuses[gate] = str(status)
    if gate_statuses["sealed_evaluation_validated"] != "pass":
        return False, "sealed evaluation must pass for a completed run", gate_statuses
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(COMPLETION_ARTIFACT_FIELDS):
        return (
            False,
            "completion receipt does not bind the exact completion artifact set",
            gate_statuses,
        )
    for label, registry_field in COMPLETION_ARTIFACT_FIELDS.items():
        record = artifacts.get(label)
        if not isinstance(record, Mapping):
            return False, f"completion receipt has a malformed {label} artifact record", gate_statuses
        declared_path = program.get(registry_field)
        if record.get("path") != declared_path:
            return False, f"completion receipt {label} path differs from the registry", gate_statuses
        resolved, error = _resolve_local_artifact(declared_path, artifact_root)
        if error or resolved is None:
            return False, error or f"completion artifact is missing: {label}", gate_statuses
        expected_sha = record.get("sha256")
        if not isinstance(expected_sha, str) or _file_sha256(resolved) != expected_sha:
            return (
                False,
                f"completion receipt {label} content binding does not match",
                gate_statuses,
            )
    expected_outcome = (
        "isomorphic"
        if all(gate_statuses[gate] == "pass" for gate in FIDELITY_GATES)
        else "non_isomorphic"
        if any(gate_statuses[gate] == "fail" for gate in FIDELITY_GATES)
        else "indeterminate"
    )
    if receipt.get("isomorphism_outcome") != expected_outcome:
        return False, "completion receipt isomorphism outcome disagrees with fidelity gates", gate_statuses
    return True, None, gate_statuses


def _validate_terminal_attempt_receipt(
    *,
    program: Mapping,
    cell_id: str,
    panel_content_sha256: str,
    artifact_root: Path,
) -> tuple[bool, str | None]:
    """Validate bounded non-discovery separately from a completed candidate run."""

    receipt_path, path_error = _resolve_local_artifact(
        program.get("terminal_attempt_receipt_path"), artifact_root
    )
    if path_error:
        return False, path_error
    if receipt_path is None:
        return False, None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"terminal attempt receipt is not valid UTF-8 JSON: {exc}"
    if not isinstance(receipt, Mapping) or receipt.get("schema") != TERMINAL_ATTEMPT_RECEIPT_SCHEMA:
        return False, "terminal attempt receipt has the wrong schema"
    if (
        receipt.get("cell_id") != cell_id
        or receipt.get("panel_content_sha256") != panel_content_sha256
    ):
        return False, "terminal attempt receipt identity binding differs"
    if receipt.get("status") != "bounded_non_discovery":
        return False, "terminal attempt receipt has the wrong status"
    if receipt.get("external_supervised_target_used") is not False:
        return False, "terminal attempt receipt does not preserve the unsupervised objective"
    scope = receipt.get("search_scope")
    required_scope = ("program_class", "capabilities", "representation", "budget", "reason")
    if not isinstance(scope, Mapping) or any(
        not isinstance(scope.get(field), str) or not scope[field].strip()
        for field in required_scope
    ):
        return False, "terminal attempt receipt does not bound the full search scope"
    if receipt.get("tacitness_claimed") is not False:
        return False, "bounded non-discovery may not be relabeled as tacitness"
    return True, None


def build_readiness(panel: Mapping, *, prompt_bank: Mapping | None = None,
                    program_registry: Mapping[str, Mapping] | None = None,
                    artifact_root: Path | None = None) -> dict:
    errors = validate_hierarchy_panel(panel)
    if errors:
        raise ValueError(f"invalid hierarchy metric panel: {errors}")
    if prompt_bank is not None:
        if prompt_bank.get("metric_panel_content_sha256") != panel.get("panel_content_sha256"):
            raise ValueError("prompt bank was compiled from a different hierarchy panel")
        bank_errors = validate_bank(dict(prompt_bank))
        if bank_errors:
            raise ValueError(f"invalid hierarchy prompt bank: {bank_errors}")
    arms = _arm_counts(prompt_bank or {})
    registry = dict(program_registry or {})
    artifact_root = (artifact_root or ROOT).resolve()
    panel_ids = {str(cell["id"]) for cell in panel["cells"]}
    extras = set(registry) - panel_ids
    if extras:
        raise ValueError(f"program registry contains cells outside the panel: {sorted(extras)[:5]}")
    malformed = sorted(cell_id for cell_id, row in registry.items() if not isinstance(row, Mapping))
    if malformed:
        raise ValueError(f"program registry has non-object rows: {malformed[:5]}")

    rows = []
    order = {task: index for index, task in enumerate(TECHNICAL_FIRST_ORDER)}
    level_order = {"R1": 0, "R2": 1, "R3": 2}
    sorted_cells = sorted(
        panel["cells"],
        key=lambda cell: (
            order[str(cell["task"])],
            level_order[str(cell["level"])],
            int(cell["selection_rank"]),
        ),
    )
    for queue_index, cell in enumerate(sorted_cells, 1):
        cell_id = str(cell["id"])
        program = registry.get(cell_id)
        declared_fields = sorted(
            field
            for field in COMPLETION_ARTIFACT_FIELDS.values()
            if program and isinstance(program.get(field), str) and program.get(field).strip()
        )
        local_fields = []
        local_errors = {}
        for field in COMPLETION_ARTIFACT_FIELDS.values():
            resolved, error = _resolve_local_artifact(
                program.get(field) if program else None, artifact_root
            )
            if resolved is not None:
                local_fields.append(field)
            elif error is not None:
                local_errors[field] = error
        has_program = "candidate_path" in local_fields
        has_decomposition = "decomposition_path" in local_fields
        has_depth = "depth_record_path" in local_fields
        has_execution = "candidate_execution_path" in local_fields
        has_reference = "frozen_reference_path" in local_fields
        has_sealed = "sealed_evaluation_path" in local_fields
        has_construct_fidelity = "construct_fidelity_path" in local_fields
        has_certificate = "certificate_path" in local_fields
        has_isomorphism_adjudication = "isomorphism_path" in local_fields
        completion_receipt_valid, completion_receipt_error, completion_gate_statuses = (
            _validate_completion_receipt(
                program=program,
                cell_id=cell_id,
                panel_content_sha256=str(panel["panel_content_sha256"]),
                artifact_root=artifact_root,
            )
            if program
            else (False, None, {})
        )
        terminal_attempt_valid, terminal_attempt_error = (
            _validate_terminal_attempt_receipt(
                program=program,
                cell_id=cell_id,
                panel_content_sha256=str(panel["panel_content_sha256"]),
                artifact_root=artifact_root,
            )
            if program
            else (False, None)
        )
        audited_depth = program.get("audited_depth") if program else None
        completed_decomposition = all(
            (has_program, has_decomposition, has_depth, has_execution, has_construct_fidelity)
        )
        operational_relation_local = bool(
            program and program.get("train_operational_relation_local_witness")
        )
        confirmatory_ready = bool(
            program and program.get("heldout_confirmatory_reconstruction_ready")
        )
        rows.append(
            {
                "queue_index": queue_index,
                "cell_id": cell_id,
                "metric_id": str(cell["metric_id"]),
                "task": str(cell["task"]),
                "level": str(cell["level"]),
                "source_identity_ready": True,
                "compiler_brief_ready": True,
                "prompt_arms_compiled": arms.get(cell_id, 0) > 0,
                "prompt_arm_count": arms.get(cell_id, 0),
                "prompt_reference_scored": has_reference,
                "prompt_reference_compiled_unscored": bool(
                    program and program.get("prompt_reference_compiled_unscored")
                ),
                "candidate_program_declared": "candidate_path" in declared_fields,
                "candidate_program_present": has_program,
                "subrelation_decomposition_present": has_decomposition,
                "depth_record_present": has_depth,
                "label_free_candidate_execution_complete": has_execution,
                "sealed_reference_evaluation_complete": has_sealed,
                "construct_fidelity_complete": bool(
                    has_construct_fidelity
                ),
                "construct_fidelity_verdict": (
                    program.get("construct_fidelity_verdict") if program else None
                ),
                "audited_depth": audited_depth,
                "completed_decomposition": completed_decomposition,
                "operational_relation_local_witness": operational_relation_local,
                "heldout_confirmatory_reconstruction_ready": confirmatory_ready,
                "decision_contributing_depth_ge_2_operational_witness": bool(
                    operational_relation_local
                    and isinstance(audited_depth, int)
                    and not isinstance(audited_depth, bool)
                    and audited_depth >= 2
                ),
                "whole_construct_code_fidelity": bool(
                    program and program.get("whole_construct_code_fidelity")
                ),
                "certificate_plane_complete": has_certificate,
                "isomorphism_complete": has_isomorphism_adjudication,
                "artifact_evidence": {
                    "declared_fields": declared_fields,
                    "locally_present_fields": sorted(local_fields),
                    "local_errors": local_errors,
                    "all_completion_artifacts_declared": (
                        set(declared_fields) == set(COMPLETION_ARTIFACT_FIELDS.values())
                    ),
                    "all_completion_artifacts_locally_present": (
                        set(local_fields) == set(COMPLETION_ARTIFACT_FIELDS.values())
                    ),
                    "completion_receipt_declared": bool(
                        program and program.get("completion_receipt_path")
                    ),
                    "completion_receipt_valid": completion_receipt_valid,
                    "completion_receipt_error": completion_receipt_error,
                    "completion_gate_statuses": completion_gate_statuses,
                    "terminal_attempt_receipt_declared": bool(
                        program and program.get("terminal_attempt_receipt_path")
                    ),
                    "terminal_attempt_receipt_valid": terminal_attempt_valid,
                    "terminal_attempt_receipt_error": terminal_attempt_error,
                },
                "bounded_non_discovery_recorded": terminal_attempt_valid,
                "completed_deep_metric_seam_run": completion_receipt_valid,
            }
        )

    counts = Counter(
        (row["task"], row["level"])
        for row in rows
        if row["completed_deep_metric_seam_run"]
    )
    matrix = {
        task: {
            level: {
                "target": int(panel["n_per_task_level"]),
                "complete": counts[(task, level)],
                "remaining": int(panel["n_per_task_level"]) - counts[(task, level)],
            }
            for level in panel["levels"]
        }
        for task in panel["tasks"]
    }
    progress_fields = (
        "completed_decomposition",
        "bounded_non_discovery_recorded",
        "operational_relation_local_witness",
        "heldout_confirmatory_reconstruction_ready",
        "decision_contributing_depth_ge_2_operational_witness",
        "whole_construct_code_fidelity",
        "prompt_reference_compiled_unscored",
        "prompt_reference_scored",
        "isomorphism_complete",
    )
    progress_matrix = {
        task: {
            level: {
                field: sum(
                    row["task"] == task and row["level"] == level and bool(row[field])
                    for row in rows
                )
                for field in progress_fields
            }
            for level in panel["levels"]
        }
        for task in panel["tasks"]
    }
    return {
        "schema": READINESS_SCHEMA,
        "panel_content_sha256": panel["panel_content_sha256"],
        "hierarchy_frame": panel.get("hierarchy_frame"),
        "n_cells": len(rows),
        "goal": "30 completed deep metric-seam runs per task and R1/R2/R3 level",
        "completed_definition": (
            "a cell-bound validated receipt covering program/decomposition/depth/execution, "
            "certificate and abstention validity, sealed evaluation, and explicit construct, "
            "same-byte input, executed-program, reference-instrument, and reference-"
            "reconstruction fidelity statuses"
        ),
        "bounded_non_discovery_definition": (
            "a separate terminal-attempt receipt bounded by program class, capabilities, "
            "representation, and budget; it does not count as a completed deep run or tacitness"
        ),
        "prompt_only_never_counts_as_completed_deep_metric_seam_run": True,
        "path_declarations_never_count_as_completed_deep_metric_seam_run": True,
        "completion_receipt_schema": COMPLETION_RECEIPT_SCHEMA,
        "terminal_attempt_receipt_schema": TERMINAL_ATTEMPT_RECEIPT_SCHEMA,
        "matrix": matrix,
        "progress_matrix": progress_matrix,
        "rows": rows,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path)
    parser.add_argument("--program-registry", type=Path)
    parser.add_argument("--briefs-out", type=Path, required=True)
    parser.add_argument("--readiness-out", type=Path, required=True)
    args = parser.parse_args(argv)
    panel = _load_json(args.panel)
    prompt_bank = _load_json(args.prompt_bank) if args.prompt_bank else None
    registry_payload = _load_json(args.program_registry) if args.program_registry else {}
    program_registry = registry_payload.get("registry", registry_payload)
    briefs = compile_briefs(panel)
    readiness = build_readiness(
        panel, prompt_bank=prompt_bank, program_registry=program_registry
    )
    _write_jsonl(args.briefs_out, briefs)
    _write_json(args.readiness_out, readiness)
    print(
        json.dumps(
            {
                "n_briefs": len(briefs),
                "n_readiness_rows": len(readiness["rows"]),
                "n_completed": sum(
                    row["completed_deep_metric_seam_run"] for row in readiness["rows"]
                ),
                "briefs_out": str(args.briefs_out),
                "readiness_out": str(args.readiness_out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
