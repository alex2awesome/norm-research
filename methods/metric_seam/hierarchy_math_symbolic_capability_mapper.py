"""Source-only retrieval for a rational-equality capability expansion.

This is an additive sensitivity over the frozen 90-cell math hierarchy.  It
statically inspects the manually designed ``ops_symbolic_steps_v1`` capability
and its v2 successor, then retrieves possible construct matches from panel
construct/description text alone.  It never imports or executes either
capability and never reads items, certificates, prompts, references, outcomes,
scores, correlations, models, or reconstruction artifacts.

Retrieval is deliberately broader than construct fidelity.  An independent
ledger must adjudicate object, relation, polarity, applicability, and
aggregation before any cell receives relation-local credit.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "metric-seam.math-symbolic-capability-source-retrieval.v1"
TASK = "math-stackexchange"
LEVELS = ("R1", "R2", "R3")
RELATION_ID = "explicit_rational_equality_preservation"

_OBJECT_CUES = {
    "argument",
    "arguments",
    "claims",
    "computation",
    "computations",
    "computational",
    "inference",
    "inferences",
    "manuscript",
    "mathematical",
    "proof",
    "proofs",
    "reasoning",
    "step",
    "steps",
    "stepwise",
}
_VALIDITY_CUES = {
    "airtight",
    "check",
    "checkability",
    "checkable",
    "checked",
    "correct",
    "correctness",
    "follows",
    "logic",
    "logical",
    "rigor",
    "rigorous",
    "rigour",
    "valid",
    "validity",
    "verification",
    "verifiability",
    "verify",
}
_EXPLICIT_SCOPE_CUES = {
    "airtight",
    "claims",
    "computational",
    "deductive",
    "each",
    "every",
    "formal",
    "manuscript",
    "mechanical",
    "proof",
    "proofs",
    "step",
    "steps",
    "stepwise",
}


class SymbolicCapabilityMapError(ValueError):
    """Raised when the frozen source-only mapping contract is violated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_sha256(construct: str, description: str) -> str:
    return _sha256_bytes((construct + "\0" + description).encode("utf-8"))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).casefold()))


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == name and value is not None:
            try:
                return ast.literal_eval(value)
            except (TypeError, ValueError) as error:
                raise SymbolicCapabilityMapError(
                    f"{name} is not a static literal"
                ) from error
    raise SymbolicCapabilityMapError(f"missing static assignment: {name}")


def _function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def inspect_symbolic_capability(
    v1_source: str,
    v2_source: str,
    *,
    v1_path: str = "ops_symbolic_steps_v1.py",
    v2_path: str = "ops_symbolic_steps_v2.py",
    v1_test_path: str = "test_ops_symbolic_steps_v1.py",
    v2_test_path: str = "test_ops_symbolic_steps_v2.py",
    v1_tests_present: bool = False,
    v2_tests_present: bool = False,
) -> dict[str, Any]:
    """Inspect source syntax only; importing the verifier is forbidden here."""

    try:
        v1_tree = ast.parse(v1_source)
        v2_tree = ast.parse(v2_source)
    except SyntaxError as error:
        raise SymbolicCapabilityMapError("symbolic capability source does not parse") from error
    required_v1 = {"verify_expression_pair", "analyze_document"}
    required_v2 = {
        "verify_expression_pair",
        "analyze_document",
        "analyze_documents_isolated",
    }
    if not required_v1.issubset(_function_names(v1_tree)):
        raise SymbolicCapabilityMapError("v1 symbolic relation functions are missing")
    if not required_v2.issubset(_function_names(v2_tree)):
        raise SymbolicCapabilityMapError("v2 symbolic/isolation functions are missing")
    if _literal_assignment(v2_tree, "RELATION_ID") != RELATION_ID:
        raise SymbolicCapabilityMapError("v2 symbolic relation identity drifted")
    v2_imports_v1 = any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "ops_symbolic_steps_v1"
        for node in v2_tree.body
    )
    if not v2_imports_v1:
        raise SymbolicCapabilityMapError("v2 no longer retains explicit v1 provenance")
    v1_imports_sympy = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name == "sympy" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module == "sympy.parsing.latex"
        )
        for node in v1_tree.body
    )
    if not v1_imports_sympy:
        raise SymbolicCapabilityMapError("v1 is no longer statically bound to SymPy/Lark parsing")
    return {
        "capability_id": "rational_equality_steps_v2",
        "selection_provenance": (
            "manually_designed_pipeline_seed_retrospectively_applied_not_automatically_discovered"
        ),
        "preferred_source": v2_path,
        "preferred_source_sha256": _sha256_bytes(v2_source.encode("utf-8")),
        "historical_v1_source": v1_path,
        "historical_v1_source_sha256": _sha256_bytes(v1_source.encode("utf-8")),
        "v2_explicitly_imports_v1": True,
        "relation_id": RELATION_ID,
        "static_functions": {
            "v1": sorted(required_v1),
            "v2": sorted(required_v2),
        },
        "relation_contract": {
            "object": (
                "adjacent expressions in an explicitly presented answer-side LaTeX equality row"
            ),
            "relation": (
                "exact rational-algebra equality preservation or exact nonidentity evidence"
            ),
            "polarity": (
                "identity supports the displayed equality; nonidentity is typed relation evidence "
                "but requires external claim-scope adjudication before becoming a criterion defect"
            ),
            "applicability": (
                "bounded parseable rational algebra only; unsupported expressions abstain"
            ),
            "aggregation": (
                "count-only document receipt; no scalar and no every-step or whole-proof claim"
            ),
        },
        "frozen_matched_relation_depth": 3,
        "frozen_depth_meaning": "positive formal relation via computer algebra",
        "isolation_and_test_receipts": {
            "isolation_wrapper_present_by_static_source": True,
            "isolation_wrapper_executed_for_this_sensitivity": False,
            "isolation_does_not_increment_matched_relation_depth": True,
            "v1_test_path": v1_test_path,
            "v1_tests_present": bool(v1_tests_present),
            "v2_test_path": v2_test_path,
            "v2_tests_present": bool(v2_tests_present),
            "capability_tests_executed_for_this_sensitivity": False,
        },
    }


def _validate_panel(panel: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise SymbolicCapabilityMapError("unexpected hierarchy panel schema")
    if not isinstance(panel.get("panel_content_sha256"), str):
        raise SymbolicCapabilityMapError("hierarchy panel has no content identity")
    raw_cells = panel.get("cells")
    if not isinstance(raw_cells, list):
        raise SymbolicCapabilityMapError("hierarchy panel cells are missing")
    cells = [cell for cell in raw_cells if cell.get("task") == TASK]
    if len(cells) != 90 or Counter(cell.get("level") for cell in cells) != {
        "R1": 30,
        "R2": 30,
        "R3": 30,
    }:
        raise SymbolicCapabilityMapError("math panel must contain 30 R1/R2/R3 cells")
    ids = [cell.get("id") for cell in cells]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in ids):
        raise SymbolicCapabilityMapError("math panel cell identity is invalid")
    if len(set(ids)) != 90:
        raise SymbolicCapabilityMapError("math panel cell identities are not unique")
    for cell in cells:
        if not isinstance(cell.get("construct"), str) or not isinstance(
            cell.get("description"), str
        ):
            raise SymbolicCapabilityMapError("math panel construct text is invalid")
    return cells


def _retrieval_features(construct: str, description: str) -> dict[str, list[str]]:
    tokens = _tokens(f"{construct} {description}")
    return {
        "object_cues": sorted(tokens & _OBJECT_CUES),
        "validity_cues": sorted(tokens & _VALIDITY_CUES),
        "explicit_scope_cues": sorted(tokens & _EXPLICIT_SCOPE_CUES),
    }


def build_symbolic_capability_map(
    panel: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    cells = _validate_panel(panel)
    if capability.get("relation_id") != RELATION_ID:
        raise SymbolicCapabilityMapError("unexpected inspected capability relation")
    if capability.get("frozen_matched_relation_depth") != 3:
        raise SymbolicCapabilityMapError("formal relation must retain frozen depth 3")
    rows: list[dict[str, Any]] = []
    for cell in cells:
        features = _retrieval_features(cell["construct"], cell["description"])
        retrieved = all(features.values())
        rows.append(
            {
                "cell_id": cell["id"],
                "task": TASK,
                "level": cell["level"],
                "metric_name": cell["construct"],
                "metric_description": cell["description"],
                "metric_source_text_sha256": _source_sha256(
                    cell["construct"], cell["description"]
                ),
                "retrieved_candidate": retrieved,
                "retrieval_features": features,
                "candidate_capability_id": (
                    capability["capability_id"] if retrieved else None
                ),
            }
        )
    retrieved_rows = [row for row in rows if row["retrieved_candidate"]]
    return {
        "schema": SCHEMA,
        "status": "static_source_only_candidate_retrieval_complete",
        "task": TASK,
        "design_scope": "construct_and_description_only_plus_static_capability_syntax",
        "sources": dict(sources or {}),
        "panel_content_sha256": panel["panel_content_sha256"],
        "capability": dict(capability),
        "retrieval_rule": {
            "tokenization": "casefolded ASCII alphabetic tokens",
            "candidate_if": (
                "at least one object cue AND one validity cue AND one explicit-scope cue"
            ),
            "construct_fidelity_inferred": False,
        },
        "programs_imported_or_executed": False,
        "items_or_articles_loaded": False,
        "certificate_counts_loaded": False,
        "prompt_outputs_loaded": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "correlations_or_reconstruction_loaded": False,
        "models_apis_or_gpus_used": False,
        "summary": {
            "n_cells": len(rows),
            "n_retrieved_candidates": len(retrieved_rows),
            "retrieved_by_level": dict(Counter(row["level"] for row in retrieved_rows)),
            "n_construct_fidelity_decisions": 0,
        },
        "rows": rows,
        "interpretation": (
            "Broad deterministic source-only retrieval for an existing manual capability seed; "
            "not automatic discovery, construct fidelity, execution, codability, or reconstruction."
        ),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v1-test", type=Path, required=True)
    parser.add_argument("--v2-test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    v1_source = args.v1.read_text(encoding="utf-8")
    v2_source = args.v2.read_text(encoding="utf-8")
    capability = inspect_symbolic_capability(
        v1_source,
        v2_source,
        v1_path=str(args.v1),
        v2_path=str(args.v2),
        v1_test_path=str(args.v1_test),
        v2_test_path=str(args.v2_test),
        v1_tests_present=args.v1_test.is_file(),
        v2_tests_present=args.v2_test.is_file(),
    )
    payload = build_symbolic_capability_map(
        _load(args.panel),
        capability,
        sources={
            "panel": str(args.panel),
            "preferred_v2_capability": str(args.v2),
            "historical_v1_capability": str(args.v1),
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
