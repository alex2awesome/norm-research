"""Run the grant relation program on compiler-train only.

This execution is deliberately pre-audit and exploratory.  It establishes
whether the frozen program emits measurable/nonconstant code-side relations;
it cannot promote any hierarchy mapping until an independent construct audit
accepts that mapping.  The sealed held-out path is never accepted by this CLI.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from methods.metric_seam.grant_structure_v1 import RELATION_DEPTHS, analyze


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPOSAL = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/grant_structure_static_proposal_v1.json"
)
DEFAULT_MANIFEST = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/grant-funding/manifest.json"
)
DEFAULT_TRAIN = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/grant-funding/compiler_train.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/grant_structure_compiler_train_v1.json"
)
PROGRAM_SOURCE = ROOT / "methods/metric_seam/grant_structure_v1.py"
ALLOWED_IMPORT_ROOTS = {"__future__", "dataclasses", "decimal", "re", "typing"}
FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "open"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_source_ast(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        ):
            raise ValueError(f"forbidden call in grant program: {node.func.id}")
    roots = {name.split(".", 1)[0] for name in imports}
    if not roots <= ALLOWED_IMPORT_ROOTS:
        raise ValueError(f"grant program import allowlist violated: {sorted(roots)}")
    return {
        "ast_parsed": True,
        "import_roots": sorted(roots),
        "file_io_calls_allowed": False,
        "network_process_environment_imports_allowed": False,
    }


def _validate_inputs(proposal: Mapping, manifest: Mapping, rows: object) -> list[dict]:
    if (
        proposal.get("schema") != "metric-seam.grant-structure-static-proposal.v1"
        or proposal.get("status")
        != "author_proposal_complete_pending_independent_construct_audit"
        or proposal.get("summary", {}).get("eligible_for_execution_before_independent_audit")
        != 0
    ):
        raise ValueError("grant authoring proposal is not the expected pre-audit artifact")
    program = proposal.get("program", {})
    if program.get("source_sha256") != _sha256(PROGRAM_SOURCE):
        raise ValueError("grant program source changed after proposal")
    if (
        manifest.get("schema") != "metric-seam.hierarchy-shared-items.v1"
        or manifest.get("task") != "grant-funding"
        or manifest.get("selection", {}).get("outcome_or_reference_values_used") is not False
        or manifest.get("policy", {}).get("outcome_columns_emitted") is not False
        or manifest.get("policy", {}).get("external_supervision_used") is not False
    ):
        raise ValueError("grant item manifest violates the label-free contract")
    if not isinstance(rows, list) or len(rows) != manifest["selection"]["train_n"] != 103:
        raise ValueError("grant compiler-train rows drifted")
    checked = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"item_key", "ctext"}:
            raise ValueError("grant compiler-train rows must contain only item_key and ctext")
        if not isinstance(row["item_key"], str) or not isinstance(row["ctext"], str):
            raise ValueError("grant compiler-train row types are invalid")
        if not row["item_key"].startswith("train_") or len(row["ctext"]) > 4000:
            raise ValueError("grant compiler-train split or representation drifted")
        checked.append(row)
    return checked


def _relation_summary(item_rows: list[dict], relation: str) -> dict:
    outputs = [row["relations"][relation] for row in item_rows]
    statuses = Counter(output["status"] for output in outputs)
    finite = [
        float(output["score"])
        for output in outputs
        if output["status"] == "measured"
        and isinstance(output["score"], (int, float))
        and math.isfinite(float(output["score"]))
    ]
    unique = sorted(set(finite))
    return {
        "depth": RELATION_DEPTHS[relation],
        "status_counts": dict(sorted(statuses.items())),
        "measured": len(finite),
        "unique_finite_scores": len(unique),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "nondegenerate": len(unique) >= 2,
    }


def run(proposal: Mapping, manifest: Mapping, rows: object) -> dict:
    checked = _validate_inputs(proposal, manifest, rows)
    ast_receipt = _validate_source_ast(PROGRAM_SOURCE)
    item_rows = []
    for row in checked:
        output = analyze(row["ctext"])
        item_rows.append(
            {
                "item_key": row["item_key"],
                "input_characters": len(row["ctext"]),
                "relations": output["relations"],
            }
        )
    by_relation = {
        relation: _relation_summary(item_rows, relation)
        for relation in sorted(RELATION_DEPTHS)
    }
    nondegenerate = [
        relation for relation, summary in by_relation.items() if summary["nondegenerate"]
    ]
    return {
        "schema": "metric-seam.grant-structure-train-execution.v1",
        "status": "compiler_train_exploratory_complete_pending_independent_construct_audit",
        "phase": "compiler_train",
        "proposal_status_at_execution": proposal["status"],
        "program": {
            "source": "methods/metric_seam/grant_structure_v1.py",
            "source_sha256": _sha256(PROGRAM_SOURCE),
            "ast_restriction_receipt": ast_receipt,
        },
        "blindness": {
            "input_fields_passed_to_program": ["ctext"],
            "reference_fields_passed_to_program": False,
            "outcome_fields_passed_to_program": False,
            "heldout_items_or_outputs_loaded": False,
            "external_supervised_anchor_used": False,
            "model_or_api_used": False,
            "accelerator_used": False,
            "credentials_required": False,
        },
        "summary": {
            "items": len(item_rows),
            "relations_executed": len(by_relation),
            "nondegenerate_relations": len(nondegenerate),
            "nondegenerate_relation_ids": nondegenerate,
            "hierarchy_mappings_promoted": 0,
            "heldout_execution_authorized": False,
            "prompt_articulability_measurements": 0,
            "reconstruction_measurements": 0,
            "isomorphism_measurements": 0,
        },
        "by_relation": by_relation,
        "rows": item_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compiler-train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = json.loads(args.compiler_train.read_text(encoding="utf-8"))
    payload = run(proposal, manifest, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

