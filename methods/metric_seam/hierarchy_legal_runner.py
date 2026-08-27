"""Execute accepted legal-writing relation projections on one sealed text split.

The runner reads only the shared item manifest, opaque ``item_key`` values,
exact ``ctext``, the pre-execution fidelity audit, and (for heldout) a frozen
compiler-train gate.  It does not read prompt outputs, references, judge
scores, outcomes, source identifiers, historical item packs, APIs, network
resources, or accelerator devices.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.adjudicate_legal_hierarchy_construct_fidelity import (
    SCHEMA as FIDELITY_SCHEMA,
)
from methods.metric_seam.legal_hierarchy_projection import (
    PROGRAM_VERSION,
    RELATION_BY_ID,
    SCHEMA as PROGRAM_SCHEMA,
    analyze_legal_writing_ctext,
    load_cpu_parser,
)


SCHEMA = "metric-seam.hierarchy-legal-execution.v1"
TASK = "legal-outcome-prediction"
PHASES = {"compiler_train", "heldout_pre_reference"}
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ITEMS_ROOT = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/legal-outcome-prediction"
PROJECTION_PATH = ROOT / "methods/metric_seam/legal_hierarchy_projection.py"


class LegalExecutionError(ValueError):
    """Raised when an input breaks the legal pre-reference execution contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def validate_items(items: object, *, phase: str) -> list[dict[str, str]]:
    if phase not in PHASES:
        raise LegalExecutionError(f"unsupported phase: {phase}")
    if not isinstance(items, list) or not items:
        raise LegalExecutionError("items must be a nonempty JSON list")
    prefix = "train_" if phase == "compiler_train" else "heldout_"
    seen = set()
    result = []
    for index, row in enumerate(items):
        if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
            raise LegalExecutionError(f"item {index} must expose exactly item_key and ctext")
        item_key, ctext = row["item_key"], row["ctext"]
        if not isinstance(item_key, str) or not item_key.startswith(prefix) or item_key in seen:
            raise LegalExecutionError(f"item {index} has invalid/duplicate opaque key")
        if not isinstance(ctext, str) or not ctext.strip():
            raise LegalExecutionError(f"item {index} has invalid ctext")
        seen.add(item_key)
        result.append({"item_key": item_key, "ctext": ctext})
    return result


def validate_manifest(manifest: Mapping[str, Any], items: Sequence[Mapping[str, str]], *, phase: str) -> int:
    if manifest.get("schema") != "metric-seam.hierarchy-shared-items.v1" or manifest.get("task") != TASK:
        raise LegalExecutionError("unexpected shared-item manifest")
    representation = manifest.get("representation", {})
    if (
        representation.get("field") != "ctext"
        or representation.get("same_bytes_required_for_prompt_and_code") is not True
        or representation.get("projection") != "source_text[:max_chars] before exact deduplication"
    ):
        raise LegalExecutionError("manifest does not bind exact shared ctext")
    max_chars = representation.get("max_chars")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars != 4000:
        raise LegalExecutionError("legal hierarchy v2 must retain the frozen 4,000-character ctext cap")
    selection = manifest.get("selection", {})
    expected_n = selection.get("train_n" if phase == "compiler_train" else "heldout_n")
    if expected_n != len(items) or len(items) != 150:
        raise LegalExecutionError("split count does not match the frozen 150-row panel")
    if selection.get("outcome_or_reference_values_used") is not False:
        raise LegalExecutionError("item selection used outcomes/references")
    policy = manifest.get("policy", {})
    for field in ("outcome_columns_emitted", "source_identifiers_emitted", "external_supervision_used"):
        if policy.get(field) is not False:
            raise LegalExecutionError(f"manifest violates {field}")
    if any(len(row["ctext"]) > max_chars for row in items):
        raise LegalExecutionError("item exceeds representation cap")
    return max_chars


def validate_fidelity(fidelity: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    if fidelity.get("schema") != FIDELITY_SCHEMA or fidelity.get("task") != TASK:
        raise LegalExecutionError("unexpected legal construct-fidelity audit")
    design = fidelity.get("audit_design", {})
    required_false = (
        "program_source_executed",
        "items_loaded",
        "prompt_outputs_loaded",
        "reference_or_outcome_values_loaded",
        "external_supervision_used",
        "historical_600_character_constructs_modified",
        "historical_programs_modified",
    )
    if any(design.get(field) is not False for field in required_false):
        raise LegalExecutionError("fidelity audit violates pre-execution/provenance separation")
    rows = fidelity.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise LegalExecutionError("fidelity audit must have 90 rows")
    mappings = []
    seen_cells = set()
    for row in rows:
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in seen_cells:
            raise LegalExecutionError("invalid/duplicate fidelity cell")
        seen_cells.add(cell_id)
        matches = row.get("matched_relations")
        if not isinstance(matches, list):
            raise LegalExecutionError("fidelity matches are not a list")
        for match in matches:
            relation_id = match.get("relation_id")
            if relation_id not in RELATION_BY_ID:
                raise LegalExecutionError(f"unknown audited relation: {relation_id}")
            if (
                match.get("construct_fidelity") != "partial_relation_local"
                or match.get("whole_construct_fidelity") is not False
                or match.get("execution_eligibility") != "relation_local_only"
                or match.get("effective_code_depth") != RELATION_BY_ID[relation_id]["effective_code_depth"]
            ):
                raise LegalExecutionError("fidelity relation scope/depth drifted")
            mappings.append(
                {
                    "cell_id": cell_id,
                    "level": row["level"],
                    "selection_rank": row["selection_rank"],
                    "construct": row["construct"],
                    "relation_id": relation_id,
                    "effective_code_depth": match["effective_code_depth"],
                }
            )
    return sorted({row["relation_id"] for row in mappings}), mappings


def _validate_gate(
    gate: Mapping[str, Any],
    *,
    fidelity_source: Path,
    relation_ids: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    # Local import avoids a runner->gate->runner import cycle at module load.
    from methods.metric_seam.hierarchy_legal_train_gate import SCHEMA as GATE_SCHEMA

    if gate.get("schema") != GATE_SCHEMA or gate.get("status") != "frozen-before-heldout-pre-reference-execution":
        raise LegalExecutionError("unexpected/unfrozen compiler-train gate")
    source = gate.get("source_fidelity", {})
    if source.get("sha256") != _sha256(fidelity_source) or source.get("path") != _relative(fidelity_source):
        raise LegalExecutionError("compiler-train gate is not bound to this fidelity audit")
    separation = gate.get("separation", {})
    for field in ("heldout_items_loaded", "prompt_outputs_loaded", "reference_or_outcome_values_loaded", "external_supervision_used"):
        if separation.get(field) is not False:
            raise LegalExecutionError(f"gate violates {field}")
    selected_relations = gate.get("selected_relation_ids")
    selected_mappings = gate.get("selected_mappings")
    if (
        not isinstance(selected_relations, list)
        or not selected_relations
        or not set(selected_relations) <= set(relation_ids)
        or not isinstance(selected_mappings, list)
    ):
        raise LegalExecutionError("gate selection is invalid")
    return selected_relations, selected_mappings


def _numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _relation_summary(rows: Sequence[Mapping[str, Any]], relation_id: str) -> dict[str, Any]:
    values = []
    certificates = 0
    for row in rows:
        result = row.get("result")
        if not isinstance(result, Mapping):
            continue
        relation = result.get("relation_values", {}).get(relation_id)
        if not isinstance(relation, Mapping):
            continue
        value = relation.get("value")
        if _numeric(value):
            values.append(float(value))
        certs = relation.get("certificates")
        if isinstance(certs, list):
            certificates += len(certs)
    value_counts = Counter(values)
    largest_tie_count = max(value_counts.values(), default=0)
    return {
        "n_items": len(rows),
        "n_measured": len(values),
        "n_abstained": len(rows) - len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "n_unique_values": len(set(values)),
        "nonconstant": bool(values and min(values) < max(values)),
        "largest_tie_count": largest_tie_count,
        "n_off_mode": len(values) - largest_tie_count,
        "n_certificates": certificates,
    }


def execute_split(
    items: Sequence[Mapping[str, str]],
    *,
    phase: str,
    relation_ids: Sequence[str],
    relation_mappings: Sequence[Mapping[str, Any]],
    representation_max_chars: int,
    parser: Any | None = None,
) -> dict[str, Any]:
    items = validate_items(list(items), phase=phase)
    if not relation_ids or not set(relation_ids) <= set(RELATION_BY_ID):
        raise LegalExecutionError("invalid empty/unknown execution relation set")
    nlp = load_cpu_parser() if parser is None else parser
    if nlp is None:
        raise LegalExecutionError("frozen CPU dependency/entity parser is unavailable")
    rows = []
    failures = Counter()
    for item in items:
        try:
            result = analyze_legal_writing_ctext(item["ctext"], relation_ids=relation_ids, nlp=nlp)
        except Exception as exc:  # measured, fail-closed receipt
            failures[type(exc).__name__] += 1
            rows.append(
                {
                    "item_key": item["item_key"],
                    "ctext_sha256": hashlib.sha256(item["ctext"].encode("utf-8")).hexdigest(),
                    "ctext_chars": len(item["ctext"]),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "result": None,
                }
            )
            continue
        rows.append(
            {
                "item_key": item["item_key"],
                "ctext_sha256": hashlib.sha256(item["ctext"].encode("utf-8")).hexdigest(),
                "ctext_chars": len(item["ctext"]),
                "status": "measured_exact_presented_ctext",
                "error_type": None,
                "result": result,
            }
        )
    summaries = {relation_id: _relation_summary(rows, relation_id) for relation_id in relation_ids}
    status_counts = Counter(row["status"] for row in rows)
    return {
        "schema": SCHEMA,
        "program_schema": PROGRAM_SCHEMA,
        "program_version": PROGRAM_VERSION,
        "phase": phase,
        "task": TASK,
        "design": {
            "input_fields": ["item_key", "ctext"],
            "exact_presented_ctext_used": True,
            "representation_max_chars": representation_max_chars,
            "source_text_beyond_presented_ctext_loaded": False,
            "historical_600_character_construct_definitions_modified": False,
            "historical_h0_programs_modified_or_executed": False,
            "outcome_or_reference_values_loaded": False,
            "prompt_outputs_loaded": False,
            "external_supervision_used": False,
            "network_or_api_used": False,
            "accelerator_used": False,
            "cpu_parser_required": True,
            "whole_criterion_score_emitted": False,
            "code_verifiability_scope": "named relation projections only",
            "prompt_articulability_measured": False,
            "reconstruction_measured": False,
            "isomorphism_measured": False,
        },
        "relation_ids": list(relation_ids),
        "relation_mappings": list(relation_mappings),
        "summary": {
            "n_items": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_types": dict(sorted(failures.items())),
            "n_relation_programs": len(relation_ids),
            "n_relation_mappings": len(relation_mappings),
            "relation_measurement": summaries,
        },
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-root", type=Path, default=DEFAULT_ITEMS_ROOT)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--selection-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.items_root.resolve() != DEFAULT_ITEMS_ROOT.resolve():
        raise LegalExecutionError("official execution requires the frozen legal items root")
    if args.phase == "compiler_train" and args.selection_gate is not None:
        raise LegalExecutionError("compiler train may not receive a selection gate")
    if args.phase == "heldout_pre_reference" and args.selection_gate is None:
        raise LegalExecutionError("heldout execution requires the frozen compiler-train gate")
    manifest_path = args.items_root / "manifest.json"
    items_path = args.items_root / ("compiler_train.json" if args.phase == "compiler_train" else "sealed_heldout.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = validate_items(json.loads(items_path.read_text(encoding="utf-8")), phase=args.phase)
    max_chars = validate_manifest(manifest, items, phase=args.phase)
    fidelity = json.loads(args.fidelity.read_text(encoding="utf-8"))
    relation_ids, relation_mappings = validate_fidelity(fidelity)
    gate_binding = None
    if args.phase == "heldout_pre_reference":
        gate = json.loads(args.selection_gate.read_text(encoding="utf-8"))
        relation_ids, relation_mappings = _validate_gate(
            gate,
            fidelity_source=args.fidelity,
            relation_ids=relation_ids,
        )
        gate_binding = {"path": _relative(args.selection_gate), "sha256": _sha256(args.selection_gate)}
    # Environment receipt: callers mask devices, while the program also calls
    # spacy.require_cpu(). Missing variables do not grant accelerator use.
    result = execute_split(
        items,
        phase=args.phase,
        relation_ids=relation_ids,
        relation_mappings=relation_mappings,
        representation_max_chars=max_chars,
    )
    result["sources"] = {
        "manifest": {"path": _relative(manifest_path), "sha256": _sha256(manifest_path)},
        "items": {"path": _relative(items_path), "sha256": _sha256(items_path)},
        "fidelity": {"path": _relative(args.fidelity), "sha256": _sha256(args.fidelity)},
        "projection": {"path": _relative(PROJECTION_PATH), "sha256": _sha256(PROJECTION_PATH)},
        "selection_gate": gate_binding,
    }
    result["environment_receipt"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "spacy_cpu_required_in_program": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
