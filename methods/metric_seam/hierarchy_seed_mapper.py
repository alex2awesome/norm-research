"""Retrieve outcome-blind code-review program seeds for the R1/R2/R3 panel.

This module is deliberately a *candidate retriever*, not an evaluator.  It
compares the source-side hierarchy name/description with static metadata from
the historical code-review metric library.  It never imports or executes a
metric module and never reads judgments, labels, held-out identifiers, metric
outputs, correlations, or reconstruction results.

Every accepted match remains a retrospective seed that requires an independent
construct-fidelity audit.  A match is not evidence that the hierarchy metric is
verifiable, reconstructed, or automatically discovered.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


SCHEMA = "metric-seam.hierarchy-code-review-seed-map.v1"
DESIGN_SCOPE = "outcome_blind_source_metadata_only"

TIER_MEANINGS = {
    0: "non-executable catalog entry / bounded non-discovery",
    1: "stdlib or surface-text operation",
    2: "AST, tokenizer, parser, or structural algorithm",
    3: "external static-analysis or formatting tool",
    4: "sandboxed execution or test-running",
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "avoid", "be", "by", "can", "code",
    "does", "for", "from", "how", "in", "into", "is", "it", "its", "of",
    "on", "or", "should", "that", "the", "their", "these", "this", "to",
    "use", "uses", "using", "when", "where", "which", "with", "without",
}

# These frequent umbrella words can support a match but cannot, by themselves,
# justify one.  For example, ``design system`` must not retrieve an unrelated
# parser merely because its title also says ``design``.
_WEAK_CONCEPTS = {
    "best", "change", "clarity", "consistency", "content", "control", "design", "global",
    "guideline", "management", "practice", "process", "quality", "scope",
    "resource", "standard", "strategy", "system", "tool", "type", "user",
}

# Small morphology/concept normalizer.  These are source-semantic aliases, not
# learned associations: no empirical outcomes or reference judgments enter it.
_ALIASES = {
    "accessible": "accessibility",
    "architectural": "architecture",
    "architectures": "architecture",
    "behaviour": "behavior",
    "behavioural": "behavior",
    "behavioral": "behavior",
    "builders": "builder",
    "classes": "class",
    "clear": "clarity",
    "cohesive": "cohesion",
    "comments": "comment",
    "commenting": "comment",
    "complexities": "complexity",
    "constructors": "constructor",
    "conventions": "convention",
    "correct": "correctness",
    "dependencies": "dependency",
    "diagnostics": "diagnostic",
    "docs": "documentation",
    "docstrings": "documentation",
    "documenting": "documentation",
    "duplicate": "duplication",
    "duplicated": "duplication",
    "dry": "duplication",
    "efficient": "performance",
    "efficiency": "performance",
    "errors": "error",
    "formats": "formatting",
    "formatted": "formatting",
    "functions": "function",
    "guidelines": "guideline",
    "idiomatic": "idiom",
    "idioms": "idiom",
    "interfaces": "interface",
    "libraries": "library",
    "maintainable": "maintainability",
    "methods": "method",
    "modular": "module",
    "modules": "module",
    "names": "naming",
    "named": "naming",
    "observability": "observable",
    "patterns": "pattern",
    "performant": "performance",
    "refactoring": "refactor",
    "refactorings": "refactor",
    "reliable": "reliability",
    "requirements": "requirement",
    "resources": "resource",
    "secure": "security",
    "simpler": "simplicity",
    "simple": "simplicity",
    "standards": "standard",
    "strategies": "strategy",
    "tests": "test",
    "testing": "test",
    "tools": "tool",
    "types": "type",
    "users": "user",
    "verification": "verify",
    "verified": "verify",
    "warnings": "warning",
}


@dataclass(frozen=True)
class ProgramMetadata:
    """Static, outcome-free metadata read from one metric module."""

    aspect_id: str
    aspect_name: str
    path: str
    declared_tier: int
    tools: tuple[str, ...]
    languages: tuple[str, ...]
    classification: str
    docstring_summary: str
    imported_roots: tuple[str, ...]
    ast_node_count: int
    function_count: int
    control_node_count: int
    uses_regex_module: bool
    program_shape: str

    @property
    def executable(self) -> bool:
        return self.classification != "THICK" and self.declared_tier > 0

    @property
    def preferred_nonlexical(self) -> bool:
        return self.executable and self.declared_tier >= 2


def _canonical_token(token: str) -> str:
    token = token.casefold()
    if token in _ALIASES:
        return _ALIASES[token]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return _ALIASES.get(token, token)


def _tokens(text: str) -> list[str]:
    result = []
    for raw in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", str(text)):
        token = _canonical_token(raw.strip(".+#-"))
        if len(token) >= 2 and token not in _STOPWORDS:
            result.append(token)
    return result


def _first_paragraph(docstring: str) -> str:
    paragraphs = re.split(r"\n\s*\n", docstring.strip(), maxsplit=1)
    return " ".join(paragraphs[0].split()) if paragraphs else ""


def _literal_constants(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    wanted = {
        "ASPECT_ID", "ASPECT_NAME", "TIER", "TOOLS", "APPLIES_TO_LANGS",
        "CLASSIFICATION",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value_node = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value_node = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name) or target.id not in wanted or value_node is None:
            continue
        try:
            values[target.id] = ast.literal_eval(value_node)
        except (TypeError, ValueError):
            continue
    return values


def _imported_roots(tree: ast.Module) -> tuple[str, ...]:
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return tuple(sorted(roots))


def _program_shape(*, tier: int, tools: tuple[str, ...], imports: tuple[str, ...],
                   control_nodes: int) -> str:
    if tier <= 0:
        return "non_executable_catalog_entry"
    if tier >= 4:
        return "environment_or_test_execution"
    if tier == 3:
        return "external_static_tool"
    parser_signal = any(
        tool.startswith("tree-sitter") or tool in {"ast", "tokenize"}
        for tool in tools
    ) or bool({"ast", "tokenize", "tree_sitter"} & set(imports))
    if parser_signal:
        return "parsed_structure"
    if tier == 2 and control_nodes >= 4:
        return "stateful_structural_algorithm"
    if tier == 2:
        return "declared_structural_operation"
    return "surface_or_stdlib_operation"


def read_program_metadata(path: Path, *, repo_root: Path | None = None) -> ProgramMetadata:
    """Parse one metric module without importing or executing it."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    values = _literal_constants(tree)
    required = {
        "ASPECT_ID", "ASPECT_NAME", "TIER", "TOOLS", "APPLIES_TO_LANGS",
        "CLASSIFICATION",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"{path}: missing literal metadata constants {missing}")
    tools = tuple(str(value) for value in values["TOOLS"])
    languages = tuple(str(value) for value in values["APPLIES_TO_LANGS"])
    tier = int(values["TIER"])
    imports = _imported_roots(tree)
    control_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    control_nodes = sum(isinstance(node, control_types) for node in ast.walk(tree))
    relative = path
    if repo_root is not None:
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            pass
    return ProgramMetadata(
        aspect_id=str(values["ASPECT_ID"]),
        aspect_name=str(values["ASPECT_NAME"]),
        path=str(relative),
        declared_tier=tier,
        tools=tools,
        languages=languages,
        classification=str(values["CLASSIFICATION"]),
        docstring_summary=_first_paragraph(ast.get_docstring(tree) or ""),
        imported_roots=imports,
        ast_node_count=sum(1 for _ in ast.walk(tree)),
        function_count=sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                           for node in ast.walk(tree)),
        control_node_count=control_nodes,
        uses_regex_module="re" in imports,
        program_shape=_program_shape(
            tier=tier, tools=tools, imports=imports, control_nodes=control_nodes
        ),
    )


def load_program_library(metrics_dir: Path, *, repo_root: Path | None = None) -> list[ProgramMetadata]:
    paths = sorted(metrics_dir.glob("a[0-9]*_*.py"))
    programs = [read_program_metadata(path, repo_root=repo_root) for path in paths]
    seen = Counter(program.aspect_id for program in programs)
    duplicates = sorted(aspect_id for aspect_id, count in seen.items() if count != 1)
    if duplicates:
        raise ValueError(f"duplicate aspect ids in program library: {duplicates}")
    if not programs:
        raise ValueError(f"no metric modules found in {metrics_dir}")
    return programs


def _idf(programs: Iterable[ProgramMetadata]) -> dict[str, float]:
    documents = []
    for program in programs:
        documents.append(set(_tokens(program.aspect_name + " " + program.docstring_summary)))
    frequency = Counter(token for document in documents for token in document)
    n = len(documents)
    return {
        token: math.log((n + 1.0) / (count + 1.0)) + 1.0
        for token, count in frequency.items()
    }


def _weighted_sum(tokens: Iterable[str], idf: Mapping[str, float]) -> float:
    return sum(idf.get(token, math.log(2.0)) for token in set(tokens))


def _weighted_f1(left: set[str], right: set[str], idf: Mapping[str, float]) -> float:
    shared = left & right
    if not shared:
        return 0.0
    shared_weight = _weighted_sum(shared, idf)
    precision = shared_weight / max(_weighted_sum(right, idf), 1e-12)
    recall = shared_weight / max(_weighted_sum(left, idf), 1e-12)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _coverage(query: set[str], document: set[str], idf: Mapping[str, float]) -> float:
    if not query:
        return 0.0
    return _weighted_sum(query & document, idf) / max(_weighted_sum(query, idf), 1e-12)


def _normalize_phrase(text: str) -> str:
    return " ".join(_tokens(text))


def _score_candidate(cell: Mapping, program: ProgramMetadata,
                     idf: Mapping[str, float]) -> dict:
    query_title = set(_tokens(str(cell["construct"])))
    query_description = set(_tokens(str(cell["description"])))
    query_all = query_title | query_description
    candidate_title = set(_tokens(program.aspect_name))
    candidate_summary = set(_tokens(program.docstring_summary))
    candidate_all = candidate_title | candidate_summary

    title_f1 = _weighted_f1(query_title, candidate_title, idf)
    description_support = _coverage(query_description, candidate_all, idf)
    candidate_support = _coverage(candidate_title, query_all, idf)
    query_phrase = _normalize_phrase(str(cell["construct"]))
    candidate_phrase = _normalize_phrase(program.aspect_name)
    phrase_match = bool(
        query_phrase and candidate_phrase
        and (query_phrase in candidate_phrase or candidate_phrase in query_phrase)
    )
    semantic_score = min(
        1.0,
        0.74 * title_f1
        + 0.18 * description_support
        + 0.08 * candidate_support
        + (0.08 if phrase_match else 0.0),
    )
    shape_bonus = {
        "environment_or_test_execution": 0.025,
        "external_static_tool": 0.020,
        "parsed_structure": 0.015,
        "stateful_structural_algorithm": 0.010,
        "declared_structural_operation": 0.005,
    }.get(program.program_shape, 0.0)
    rank_score = semantic_score + shape_bonus
    shared_title = sorted(query_title & candidate_title)
    candidate_title_in_query = sorted(query_all & candidate_title)
    strong_shared_title = sorted(set(shared_title) - _WEAK_CONCEPTS)
    strong_candidate_title_in_query = sorted(
        set(candidate_title_in_query) - _WEAK_CONCEPTS
    )
    defensible = bool(
        program.executable
        and program.preferred_nonlexical
        and (
            (semantic_score >= 0.12 and phrase_match)
            or (
                semantic_score >= 0.20
                and len(shared_title) >= 2
                and strong_shared_title
            )
            or (
                semantic_score >= 0.18
                and len(candidate_title_in_query) >= 2
                and strong_candidate_title_in_query
            )
            or (semantic_score >= 0.31 and strong_shared_title)
        )
    )
    return {
        "aspect_id": program.aspect_id,
        "aspect_name": program.aspect_name,
        "source_path": program.path,
        "semantic_score": round(semantic_score, 6),
        "rank_score": round(rank_score, 6),
        "score_components": {
            "title_weighted_f1": round(title_f1, 6),
            "description_support": round(description_support, 6),
            "candidate_title_support": round(candidate_support, 6),
            "exact_normalized_phrase_match": phrase_match,
            "shared_title_concepts": shared_title,
            "strong_shared_title_concepts": strong_shared_title,
            "candidate_title_concepts_in_query": candidate_title_in_query,
            "strong_candidate_title_concepts_in_query": strong_candidate_title_in_query,
        },
        "depth_provenance": {
            "declared_tool_tier": program.declared_tier,
            "declared_tool_tier_meaning": TIER_MEANINGS.get(
                program.declared_tier, "undeclared tier semantics"
            ),
            "derived_program_shape": program.program_shape,
            "preferred_nonlexical": program.preferred_nonlexical,
            "derivation": (
                "static AST shape plus literal TIER/TOOLS/import metadata; candidate was not run"
            ),
        },
        "tool_provenance": {
            "declared_tools": list(program.tools),
            "declared_languages": list(program.languages),
            "declared_classification": program.classification,
            "imported_module_roots": list(program.imported_roots),
            "uses_regex_module": program.uses_regex_module,
            "static_ast_node_count": program.ast_node_count,
            "static_function_count": program.function_count,
            "static_control_node_count": program.control_node_count,
        },
        "docstring_summary": program.docstring_summary,
        "passes_source_only_seed_gate": defensible,
    }


def retrieve_for_cell(cell: Mapping, programs: Sequence[ProgramMetadata], *, top_k: int = 5) -> dict:
    """Retrieve provisional candidates for one source-side hierarchy cell."""
    if cell.get("task") != "code-review":
        raise ValueError("this mapper is intentionally limited to code-review cells")
    idf = _idf(programs)
    candidates = [_score_candidate(cell, program, idf) for program in programs]
    # Construct relevance dominates.  The depth preference is deliberately a
    # small tie-break so an unrelated deep tool cannot outrank a close construct.
    candidates.sort(
        key=lambda row: (
            row["passes_source_only_seed_gate"],
            row["rank_score"],
            row["semantic_score"],
            row["depth_provenance"]["declared_tool_tier"],
            row["aspect_id"],
        ),
        reverse=True,
    )
    gated = [row for row in candidates if row["passes_source_only_seed_gate"]]
    selected = gated[0] if gated else None
    alternatives = candidates[:top_k]
    if selected is not None and selected not in alternatives:
        alternatives = [selected, *alternatives[: max(0, top_k - 1)]]
    return {
        "cell_id": str(cell["id"]),
        "task": "code-review",
        "level": str(cell["level"]),
        "metric_name": str(cell["construct"]),
        "metric_description": str(cell["description"]),
        "decision": (
            "candidate_seed_pending_construct_fidelity_audit" if selected else "abstain"
        ),
        "selected_seed": selected,
        "top_source_only_candidates": alternatives,
        "interpretation": (
            "retrospective candidate retrieval only; not a verified program, reconstruction "
            "result, isomorphism result, or automatic-discovery claim"
        ),
    }


def build_seed_map(panel: Mapping, programs: Sequence[ProgramMetadata], *, top_k: int = 5) -> dict:
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise ValueError("expected tacit_breadth_metric_panel/v1")
    cells = [cell for cell in panel.get("cells", []) if cell.get("task") == "code-review"]
    counts = Counter(str(cell.get("level")) for cell in cells)
    if counts != Counter({"R1": 30, "R2": 30, "R3": 30}):
        raise ValueError(f"expected 30 code-review cells at each R1/R2/R3 level; found {counts}")
    required = {"id", "task", "level", "construct", "description"}
    for cell in cells:
        missing = sorted(required - cell.keys())
        if missing:
            raise ValueError(f"panel cell is missing source metadata: {missing}")

    rows = [retrieve_for_cell(cell, programs, top_k=top_k) for cell in cells]
    decision_counts = Counter(row["decision"] for row in rows)
    selected = [row["selected_seed"] for row in rows if row["selected_seed"] is not None]
    by_level = {
        level: {
            "n_cells": counts[level],
            "n_candidate_seeds": sum(
                row["level"] == level and row["selected_seed"] is not None for row in rows
            ),
            "n_abstentions": sum(
                row["level"] == level and row["selected_seed"] is None for row in rows
            ),
        }
        for level in ("R1", "R2", "R3")
    }
    return {
        "schema": SCHEMA,
        "status": "retrospective-candidate-seeds-pending-construct-fidelity-audit",
        "design_scope": DESIGN_SCOPE,
        "panel_schema": panel["schema"],
        "panel_content_sha256": panel.get("panel_content_sha256"),
        "hierarchy_frame": panel.get("hierarchy_frame"),
        "task": "code-review",
        "levels": ["R1", "R2", "R3"],
        "n_cells": len(rows),
        "n_library_modules": len(programs),
        "input_fields_used": ["id", "task", "level", "construct", "description"],
        "forbidden_inputs": [
            "reference judgments", "outcome labels", "heldout identifiers",
            "program outputs", "correlations", "reconstruction results",
        ],
        "provenance": {
            "retrieval": "deterministic static-metadata retrieval in this run",
            "seed_source": "retrospective reuse from the historical code-review A-bank",
            "original_program_authorship": (
                "not encoded uniformly in the module interface; no automatic-discovery "
                "provenance is inferred"
            ),
            "candidate_execution": False,
            "construct_fidelity_adjudication": False,
        },
        "retrieval_policy": {
            "semantic_evidence": (
                "deterministic weighted concept overlap between hierarchy construct/description "
                "and ASPECT_NAME/first module-docstring paragraph"
            ),
            "depth_preference": (
                "only executable Tier>=2 candidates can pass; static program shape adds at most "
                "0.025 as a tie-break after construct relevance"
            ),
            "abstention": (
                "abstain unless a candidate has explicit normalized concept/phrase support and "
                "passes the frozen source-only score gate"
            ),
            "post_retrieval_gate": (
                "independent construct-fidelity audit required before execution or any "
                "verifiability/reconstruction claim"
            ),
        },
        "tier_vocabulary_source": "methods/existing_metrics_runner/coded/GUIDE.md",
        "summary": {
            "decision_counts": dict(sorted(decision_counts.items())),
            "by_level": by_level,
            "n_unique_selected_programs": len({row["aspect_id"] for row in selected}),
            "selected_declared_tier_counts": dict(sorted(Counter(
                str(row["depth_provenance"]["declared_tool_tier"])
                for row in selected
            ).items())),
            "selected_program_shape_counts": dict(sorted(Counter(
                row["depth_provenance"]["derived_program_shape"]
                for row in selected
            ).items())),
            "selected_classification_counts": dict(sorted(Counter(
                row["tool_provenance"]["declared_classification"]
                for row in selected
            ).items())),
        },
        "rows": rows,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >=1")
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --force to replace")
    repo_root = Path(__file__).resolve().parents[2]
    programs = load_program_library(args.metrics_dir, repo_root=repo_root)
    payload = build_seed_map(_load_json(args.panel), programs, top_k=args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
