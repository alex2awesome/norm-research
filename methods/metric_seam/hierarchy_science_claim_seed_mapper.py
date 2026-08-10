"""Inventory the full-article science claim verifier against the R1/R2/R3 panel.

The inventory is deliberately static and outcome blind.  It parses the hierarchy
cell's source text and the Python syntax of the existing science verifier; it never
imports or executes that verifier and never reads articles, item identifiers,
judgements, program outputs, correlations, or reconstruction results.

The historical verifier is one manually designed pure-code capability.  It extracts
result-bearing claims from an abstract, retrieves distinct sentences from the same
article's full-paper body with a document-local BM25 index, applies numeric and
directed-comparison predicates, and chooses an exact one-to-one matching.  A retrieved
seed is only a candidate for a later relation-local audit.  Even an executed
certificate would establish document-internal consistency, not external scientific
truth or a whole peer-review judgement.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence


SCHEMA = "metric-seam.hierarchy-science-claim-seed-map.v1"
TASK = "peer-review"
CAPABILITY_ID = "science_claims_v2_relation_strict_full_article"
DESIGN_SCOPE = "outcome_blind_static_panel_and_python_source_only"

DEPTH_MEANINGS = {
    0: "non-executable catalog entry / bounded non-discovery",
    1: "surface text, regex, or generic document statistics",
    2: "document parser, structural aggregation, or stateful local algorithm",
    3: "retrieval or external-evidence pipeline",
    4: "sandboxed execution, test-running, or formal proof checking",
}

_CLAIM_OBJECTS = {"claim", "claims", "conclusion", "conclusions"}
_SUPPORT_RELATIONS = {
    "align",
    "aligned",
    "alignment",
    "calibrated",
    "calibration",
    "data",
    "evidence",
    "support",
    "supported",
    "supports",
    "warranted",
}
_FRONT_OBJECTS = {"abstract"}
_FRONT_ACCURACY = {"accurate", "accurately", "accuracy", "fidelity"}
_FRONT_CONTENT = {"conclusions", "content", "contribution", "findings", "scope"}

_FORBIDDEN_IMPORT_ROOTS = {
    "anthropic",
    "datasets",
    "huggingface_hub",
    "openai",
    "requests",
    "sklearn",
    "torch",
    "transformers",
}


@dataclass(frozen=True)
class SourceModule:
    """Static syntax inventory for one module; reading it cannot execute it."""

    role: str
    path: str
    source_sha256: str
    module_docstring: str
    functions: tuple[str, ...]
    classes: tuple[str, ...]
    imports: tuple[str, ...]
    constants: tuple[str, ...]
    ast_node_count: int
    control_node_count: int
    forbidden_import_roots: tuple[str, ...]
    dynamic_execution_calls: tuple[str, ...]


def _source_module_payload(module: SourceModule) -> dict:
    """Return a JSON-native representation stable before and after serialization."""

    payload = asdict(module)
    for field in (
        "functions",
        "classes",
        "imports",
        "constants",
        "forbidden_import_roots",
        "dynamic_execution_calls",
    ):
        payload[field] = list(payload[field])
    return payload


def _relative_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _import_names(tree: ast.Module) -> tuple[str, ...]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            names.update(f"{prefix}:{alias.name}" for alias in node.names)
    return tuple(sorted(names))


def _import_root(name: str) -> str:
    clean = name.lstrip(".").split(":", 1)[0]
    return clean.split(".", 1)[0]


def read_source_module(
    path: Path, *, role: str, repo_root: Path | None = None
) -> SourceModule:
    """Parse one Python file without importing or executing it."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports = _import_names(tree)
    dynamic = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec", "compile", "__import__"}
        }
    )
    controls = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    constants: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        constants.update(
            target.id
            for target in targets
            if isinstance(target, ast.Name) and target.id.isupper()
        )
    roots = {_import_root(name) for name in imports}
    return SourceModule(
        role=role,
        path=_relative_path(path, repo_root),
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        module_docstring=" ".join((ast.get_docstring(tree) or "").split()),
        functions=tuple(
            sorted(node.name for node in tree.body if isinstance(node, ast.FunctionDef))
        ),
        classes=tuple(
            sorted(node.name for node in tree.body if isinstance(node, ast.ClassDef))
        ),
        imports=imports,
        constants=tuple(sorted(constants)),
        ast_node_count=sum(1 for _ in ast.walk(tree)),
        control_node_count=sum(isinstance(node, controls) for node in ast.walk(tree)),
        forbidden_import_roots=tuple(sorted(roots & _FORBIDDEN_IMPORT_ROOTS)),
        dynamic_execution_calls=tuple(dynamic),
    )


def build_capability_inventory(
    core_path: Path,
    strict_path: Path,
    *,
    corrected_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Build and validate the source-only inventory for the manual capability."""

    if corrected_path is None:
        corrected_path = strict_path.with_name("core_corrected.py")
    modules = [
        read_source_module(core_path, role="base_full_article_pipeline", repo_root=repo_root),
        read_source_module(
            corrected_path,
            role="strict_quantity_dependency",
            repo_root=repo_root,
        ),
        read_source_module(strict_path, role="strict_relation_layer", repo_root=repo_root),
    ]
    base, corrected, strict = modules
    required_base_functions = {
        "segment_sentences",
        "extract_claims",
        "_max_weight_matching",
        "verify_document",
    }
    required_base_classes = {"DocumentBM25"}
    required_strict_functions = {
        "extract_quantities",
        "quantity_relation_equal",
        "_quantity_matching",
        "extract_comparison",
        "evaluate_edge",
        "verify_document",
    }
    required_corrected_functions = {"extract_quantities"}
    missing = {
        "base_functions": sorted(required_base_functions - set(base.functions)),
        "base_classes": sorted(required_base_classes - set(base.classes)),
        "corrected_functions": sorted(
            required_corrected_functions - set(corrected.functions)
        ),
        "strict_functions": sorted(required_strict_functions - set(strict.functions)),
    }
    if any(missing.values()):
        raise ValueError(f"science capability source is missing required symbols: {missing}")
    unsafe_imports = sorted(
        {name for module in modules for name in module.forbidden_import_roots}
    )
    dynamic_calls = sorted(
        {name for module in modules for name in module.dynamic_execution_calls}
    )
    if unsafe_imports or dynamic_calls:
        raise ValueError(
            "science capability source is not within the static pure-code envelope: "
            f"imports={unsafe_imports}, dynamic_calls={dynamic_calls}"
        )
    return {
        "capability_id": CAPABILITY_ID,
        "historical_construction": "retrospective manually designed pipeline seed",
        "automatic_discovery": False,
        "channel": "pure_code",
        "source_modules": [_source_module_payload(module) for module in modules],
        "relation_chain": [
            {
                "operation": "abstract sentence segmentation and executable-claim extraction",
                "effective_depth": 2,
                "basis": "stateful document parsing and typed claim classification",
            },
            {
                "operation": "document-local BM25 retrieval over distinct full-paper body sentences",
                "effective_depth": 3,
                "basis": "retrieval is fit independently inside each presented article",
            },
            {
                "operation": "numeric/unit/entity/direction and comparative-role predicates",
                "effective_depth": 2,
                "basis": "executable relation predicates with explicit abstention paths",
            },
            {
                "operation": "exact maximum-weight one-to-one claim/evidence matching",
                "effective_depth": 2,
                "basis": "stateful bipartite matching prevents evidence-sentence reuse",
            },
        ],
        "maximum_relation_chain_depth": 3,
        "maximum_relation_chain_depth_meaning": DEPTH_MEANINGS[3],
        "applicability": (
            "requires an abstract, an independently segmented full-paper body, and at least one "
            "extractable result-bearing relation; otherwise the executable pipeline abstains"
        ),
        "aggregation": (
            "select at most five abstract claims, retrieve at most eight candidates per claim, "
            "then choose distinct body sentences by exact maximum-weight matching"
        ),
        "certificate_scope": (
            "document-internal support or contradiction for an extracted numeric/comparative "
            "relation; empirical/theoretical lexical links remain weaker evidence links"
        ),
        "external_scientific_truth_established": False,
        "whole_peer_review_judgement_established": False,
    }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).casefold()))


def _source_only_gate(cell: Mapping) -> dict:
    title = _tokens(str(cell["construct"]))
    description = _tokens(str(cell["description"]))
    query = title | description
    claim_objects = sorted(query & _CLAIM_OBJECTS)
    support_relations = sorted(query & _SUPPORT_RELATIONS)
    abstract_objects = sorted(query & _FRONT_OBJECTS)
    # "Front matter" is admitted only when both words are present.
    if {"front", "matter"} <= query:
        abstract_objects.append("front matter")
    front_accuracy = sorted(query & _FRONT_ACCURACY)
    front_content = sorted(query & _FRONT_CONTENT)
    claim_support_gate = bool(claim_objects and support_relations)
    abstract_fidelity_gate = bool(
        abstract_objects and front_accuracy and front_content
    )
    reasons: list[str] = []
    if claim_support_gate:
        reasons.append("explicit_claim_or_conclusion_plus_support_relation_language")
    if abstract_fidelity_gate:
        reasons.append("explicit_abstract_or_front_matter_fidelity_language")
    return {
        "passes": bool(reasons),
        "reasons": reasons,
        "matched_source_terms": {
            "claim_or_conclusion": claim_objects,
            "support_relation": support_relations,
            "abstract_or_front_matter": abstract_objects,
            "accuracy_or_fidelity": front_accuracy,
            "front_content": front_content,
        },
    }


def retrieve_for_cell(cell: Mapping, capability: Mapping) -> dict:
    if cell.get("task") != TASK:
        raise ValueError(f"this mapper is intentionally limited to {TASK} cells")
    gate = _source_only_gate(cell)
    selected = None
    if gate["passes"]:
        selected = {
            "capability_id": capability["capability_id"],
            "source_paths": [
                module["path"] for module in capability["source_modules"]
            ],
            "retrieval_gate": gate,
            "channel": capability["channel"],
            "historical_construction": capability["historical_construction"],
            "maximum_relation_chain_depth": capability[
                "maximum_relation_chain_depth"
            ],
            "certificate_scope": capability["certificate_scope"],
            "candidate_status": (
                "pending_independent_object_relation_polarity_applicability_aggregation_audit"
            ),
        }
    return {
        "cell_id": str(cell["id"]),
        "task": TASK,
        "level": str(cell["level"]),
        "metric_name": str(cell["construct"]),
        "metric_description": str(cell["description"]),
        "decision": (
            "candidate_seed_pending_independent_construct_fidelity_audit"
            if selected is not None
            else "abstain"
        ),
        "selected_seed": selected,
        "source_only_gate": gate,
        "interpretation": (
            "retrospective source-only candidate retrieval; not execution, construct fidelity, "
            "codability, reconstruction, isomorphism, or scientific truth"
        ),
    }


def build_seed_map(panel: Mapping, capability: Mapping) -> dict:
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise ValueError("expected tacit_breadth_metric_panel/v1")
    if capability.get("capability_id") != CAPABILITY_ID:
        raise ValueError(f"expected capability {CAPABILITY_ID}")
    cells = [cell for cell in panel.get("cells", []) if cell.get("task") == TASK]
    counts = Counter(str(cell.get("level")) for cell in cells)
    if counts != Counter({"R1": 30, "R2": 30, "R3": 30}):
        raise ValueError(
            f"expected 30 peer-review cells at each R1/R2/R3 level; found {counts}"
        )
    required = {"id", "task", "level", "construct", "description"}
    for cell in cells:
        missing = sorted(required - cell.keys())
        if missing:
            raise ValueError(f"panel cell is missing source metadata: {missing}")
    rows = [retrieve_for_cell(cell, capability) for cell in cells]
    selected = [row for row in rows if row["selected_seed"] is not None]
    by_level = {
        level: {
            "n_cells": counts[level],
            "n_candidate_seeds": sum(
                row["level"] == level and row["selected_seed"] is not None
                for row in rows
            ),
            "n_abstentions": sum(
                row["level"] == level and row["selected_seed"] is None
                for row in rows
            ),
        }
        for level in ("R1", "R2", "R3")
    }
    return {
        "schema": SCHEMA,
        "status": (
            "retrospective-candidate-seeds-pending-independent-construct-fidelity-audit"
        ),
        "design_scope": DESIGN_SCOPE,
        "task": TASK,
        "panel_schema": panel["schema"],
        "panel_content_sha256": panel.get("panel_content_sha256"),
        "hierarchy_frame": panel.get("hierarchy_frame"),
        "n_cells": len(rows),
        "n_historical_capability_families": 1,
        "input_fields_used": ["id", "task", "level", "construct", "description"],
        "forbidden_inputs": [
            "articles, items, item identifiers, or train/heldout membership",
            "reference judgements, decisions, acceptance labels, or reviewer scores",
            "historical program outputs, certificates, correlations, or result artifacts",
            "prompt outputs, reconstruction results, or isomorphism outcomes",
            "external scientific databases or external truth labels",
        ],
        "provenance": {
            "retrieval": "deterministic source-only exact lexical relation gate",
            "seed_source": "existing full-article science claim verifier",
            "original_pipeline_authorship": "manual; not automatic discovery",
            "candidate_execution": False,
            "construct_fidelity_adjudication": False,
            "prompt_articulability_evaluation": False,
            "external_supervision_used_for_this_inventory": False,
        },
        "retrieval_policy": {
            "claim_support_gate": (
                "requires an explicit claim/conclusion object and an explicit support, evidence, "
                "data, alignment, calibration, or warrant relation term"
            ),
            "abstract_fidelity_gate": (
                "requires explicit abstract/front-matter language, accuracy/fidelity language, "
                "and a scope/contribution/finding/conclusion/content object"
            ),
            "post_retrieval_gate": (
                "a separate static audit must align object, relation, polarity, applicability, "
                "and aggregation before any relation-local execution is eligible"
            ),
        },
        "capability_inventory": dict(capability),
        "summary": {
            "decision_counts": dict(
                sorted(Counter(row["decision"] for row in rows).items())
            ),
            "by_level": by_level,
            "n_unique_selected_capability_families": len(
                {
                    row["selected_seed"]["capability_id"]
                    for row in selected
                    if row["selected_seed"] is not None
                }
            ),
            "exact_whole_construct_code_fidelity_established": 0,
            "relation_local_fidelity_established": 0,
            "execution_witnesses_established": 0,
            "external_scientific_truth_claims": 0,
        },
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--corrected", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    capability = build_capability_inventory(
        args.core,
        args.strict,
        corrected_path=args.corrected,
        repo_root=repo_root,
    )
    result = build_seed_map(panel, capability)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
