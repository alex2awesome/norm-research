"""Retrieve outcome-blind historical math seeds for the R1/R2/R3 panel.

This is a static candidate-inventory pass.  It reads only source-side panel
names/descriptions and Python syntax from ``programs_math``/``ops_math.py``.
It never imports or executes a historical program and never reads items,
judgments, labels, program outputs, correlations, or reconstruction results.

The historical modules are *hybrids*: their code predicates are mixed with
``LLM_FIELDS``, and their docstrings often disclose manual, train-residual-
informed construction.  Retrieval therefore establishes only that an existing
historical family is worth an independent relation-local fidelity audit.  It
does not establish whole-criterion code fidelity, verifiability, isomorphism,
codability, or automatic discovery.
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


SCHEMA = "metric-seam.hierarchy-math-seed-map.v1"
DESIGN_SCOPE = "outcome_blind_static_source_and_capability_metadata_only"
TASK = "math-stackexchange"

# Ops defined in ops_math.py.  They are deterministic document-local parsers
# or parsed-structure aggregators, rather than proof checkers or a CAS.
MATH_STRUCTURE_OPS = {
    "extract_math_spans",
    "latex_tokens",
    "notation_census",
    "equation_stats",
    "proof_skeleton",
    "delimiter_health",
}
GENERIC_SURFACE_OPS = {"normalize", "sent_stats"}

DEPTH_MEANINGS = {
    0: "non-executable catalog entry / bounded non-discovery",
    1: "surface text, regex, or generic document statistics",
    2: "LaTeX parsing, token streams, structural aggregation, or stateful local algorithm",
    3: "computer algebra, proof assistant, or external static tool",
    4: "sandboxed execution, test-running, or formal proof checking",
}

_STOPWORDS = {
    "a", "an", "and", "answer", "answers", "appropriate", "as", "at",
    "be", "by", "channel", "criterion", "ensure", "for", "from", "good",
    "h0", "h1", "hybrid", "in", "is", "it", "its", "math", "mathematical",
    "metric", "of", "on", "or", "should", "stackexchange", "that", "the",
    "their", "these", "this", "to", "use", "uses", "using", "when", "which",
    "with", "without",
}

_WEAK_CONCEPTS = {
    "accuracy", "clarity", "communication", "complete", "completeness",
    "content", "correctness", "effective", "general", "integration", "method",
    "presentation", "quality", "reasoning", "statement", "structure", "support",
}

_ALIASES = {
    "abstracts": "abstract",
    "algorithms": "algorithm",
    "analogical": "analogy",
    "analogies": "analogy",
    "attributed": "attribution",
    "attributions": "attribution",
    "captions": "caption",
    "checkable": "checkability",
    "citations": "citation",
    "cited": "citation",
    "concise": "concision",
    "consistent": "consistency",
    "constructions": "construction",
    "constructive": "construction",
    "counterexamples": "counterexample",
    "definitions": "definition",
    "diagrams": "visual",
    "diagrammatic": "visual",
    "display": "layout",
    "displays": "layout",
    "economical": "economy",
    "examples": "example",
    "figures": "visual",
    "formal": "rigor",
    "formulae": "formula",
    "formulas": "formula",
    "graphics": "visual",
    "graphical": "visual",
    "heuristics": "heuristic",
    "hypotheses": "hypothesis",
    "introductions": "introduction",
    "labels": "labeling",
    "labeled": "labeling",
    "labelled": "labeling",
    "logical": "logic",
    "notations": "notation",
    "proofs": "proof",
    "references": "reference",
    "rigorous": "rigor",
    "scaffolding": "organization",
    "signposting": "organization",
    "simple": "simplicity",
    "sources": "citation",
    "symbols": "symbol",
    "theorems": "theorem",
    "typeset": "typesetting",
    "typography": "typesetting",
    "typographical": "typesetting",
    "typographic": "typesetting",
    "visuals": "visual",
    "witnesses": "witness",
}


@dataclass(frozen=True)
class CapabilityMetadata:
    """One public MathOps capability parsed without importing ops_math."""

    name: str
    summary: str
    ast_node_count: int
    control_node_count: int


@dataclass(frozen=True)
class ProgramVariant:
    """Static metadata for one historical ``aN_hM.py`` variant."""

    aspect_id: str
    revision: int
    path: str
    source_heading: str
    llm_field_names: tuple[str, ...]
    invoked_ops: tuple[str, ...]
    deep_math_ops: tuple[str, ...]
    generic_surface_ops: tuple[str, ...]
    unknown_ops: tuple[str, ...]
    imported_roots: tuple[str, ...]
    regex_constant_names: tuple[str, ...]
    ast_node_count: int
    function_count: int
    control_node_count: int
    code_depth: int
    program_shape: str


@dataclass(frozen=True)
class ProgramFamily:
    """A historical aspect family, represented by its latest source revision."""

    aspect_id: str
    selected_variant: ProgramVariant
    variants: tuple[ProgramVariant, ...]


def _canonical_token(token: str) -> str:
    token = token.casefold().strip("._+#-")
    if token in _ALIASES:
        return _ALIASES[token]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return _ALIASES.get(token, token)


def _tokens(text: str) -> list[str]:
    result: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*", str(text)):
        token = _canonical_token(raw)
        if len(token) >= 2 and token not in _STOPWORDS:
            result.append(token)
    return result


def _first_paragraph(text: str) -> str:
    chunks = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    return " ".join(chunks[0].split()) if chunks and chunks[0].strip() else ""


def _source_heading(source: str, tree: ast.Module, *, aspect_id: str, revision: int) -> str:
    """Extract only a source heading, excluding residual/performance rationale."""
    paragraph = _first_paragraph(ast.get_docstring(tree) or "")
    if paragraph:
        # Quoted criterion names are the cleanest declarations in this bank.
        quoted = [value.strip() for value in re.findall(r"[\"']([^\"']{8,160})[\"']", paragraph)]
        if quoted:
            return max(quoted, key=len)
        heading = paragraph
        heading = re.sub(
            rf"^(?:Hybrid\s+)?metric(?:\s+channel)?\s+(?:for\s+)?{re.escape(aspect_id)}\s*:?\s*",
            "",
            heading,
            flags=re.I,
        )
        heading = re.sub(
            rf"^{re.escape(aspect_id)}(?:\s+h{revision})?\s*(?:(?::|--?)+\s*)?",
            "",
            heading,
            flags=re.I,
        )
        heading = re.split(
            r"\s+(?:--?|:)\s+(?:LaTeX-aware\s+)?hybrid\b",
            heading,
            maxsplit=1,
            flags=re.I,
        )[0]
        heading = re.split(r"\s*\((?:Math|build from)\b", heading, maxsplit=1, flags=re.I)[0]
        return heading.strip(" .:-")

    # Two legacy files have no module docstring.  Use their leading source
    # comment when it declares a title; otherwise the LLM field/constant names
    # below provide the bounded fallback semantics.
    leading = source[:5000]
    comment_match = re.search(
        rf"^\s*#.*?{re.escape(aspect_id)}\s+[\"']([^\"']+)[\"']",
        leading,
        flags=re.I | re.M,
    )
    return comment_match.group(1).strip() if comment_match else ""


def _literal_llm_field_names(tree: ast.Module) -> tuple[str, ...]:
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if not isinstance(target, ast.Name) or target.id != "LLM_FIELDS" or value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (TypeError, ValueError):
            return ()
        if isinstance(literal, dict):
            return tuple(sorted(str(key) for key in literal))
    return ()


def _invoked_ops(tree: ast.Module) -> tuple[str, ...]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ops"
        ):
            names.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "ops"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return tuple(sorted(names))


def _imported_roots(tree: ast.Module) -> tuple[str, ...]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return tuple(sorted(roots))


def _regex_constants(tree: ast.Module) -> tuple[str, ...]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name)
            and ("RE" in target.id or "PAT" in target.id or "REGEX" in target.id)
            for target in targets
        ):
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return tuple(sorted(names))


def read_capability_catalog(path: Path) -> dict[str, CapabilityMetadata]:
    """Parse MathOps public methods without importing the capability module."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    math_ops = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MathOps"),
        None,
    )
    if math_ops is None:
        raise ValueError(f"{path}: MathOps class not found")
    control_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    catalog: dict[str, CapabilityMetadata] = {}
    for node in math_ops.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
            continue
        catalog[node.name] = CapabilityMetadata(
            name=node.name,
            summary=_first_paragraph(ast.get_docstring(node) or ""),
            ast_node_count=sum(1 for _ in ast.walk(node)),
            control_node_count=sum(isinstance(item, control_types) for item in ast.walk(node)),
        )
    missing = sorted(MATH_STRUCTURE_OPS - catalog.keys())
    if missing:
        raise ValueError(f"{path}: missing expected MathOps capabilities {missing}")
    return catalog


def read_program_variant(
    path: Path,
    *,
    capability_names: Iterable[str],
    repo_root: Path | None = None,
) -> ProgramVariant:
    """Parse one historical program without importing or executing it."""
    match = re.fullmatch(r"(a\d+)_h(\d+)\.py", path.name)
    if match is None:
        raise ValueError(f"unexpected historical math filename: {path}")
    aspect_id, revision_text = match.groups()
    revision = int(revision_text)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    invoked = _invoked_ops(tree)
    capability_set = set(capability_names)
    deep = tuple(sorted(set(invoked) & capability_set))
    surface = tuple(sorted(set(invoked) & GENERIC_SURFACE_OPS))
    unknown = tuple(sorted(set(invoked) - capability_set - GENERIC_SURFACE_OPS))
    control_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    control_count = sum(isinstance(node, control_types) for node in ast.walk(tree))
    regex_names = _regex_constants(tree)
    if deep:
        code_depth = 2
        program_shape = (
            "multi_stage_parsed_math_pipeline" if len(deep) >= 3 else "parsed_math_features"
        )
    elif control_count >= 5 and regex_names:
        code_depth = 2
        program_shape = "stateful_local_structural_algorithm"
    else:
        code_depth = 1
        program_shape = "surface_regex_or_document_statistics"
    relative = path
    if repo_root is not None:
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            pass
    return ProgramVariant(
        aspect_id=aspect_id,
        revision=revision,
        path=str(relative),
        source_heading=_source_heading(source, tree, aspect_id=aspect_id, revision=revision),
        llm_field_names=_literal_llm_field_names(tree),
        invoked_ops=invoked,
        deep_math_ops=deep,
        generic_surface_ops=surface,
        unknown_ops=unknown,
        imported_roots=_imported_roots(tree),
        regex_constant_names=regex_names,
        ast_node_count=sum(1 for _ in ast.walk(tree)),
        function_count=sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)
        ),
        control_node_count=control_count,
        code_depth=code_depth,
        program_shape=program_shape,
    )


def load_program_families(
    programs_dir: Path,
    *,
    capability_catalog: Mapping[str, CapabilityMetadata],
    repo_root: Path | None = None,
) -> list[ProgramFamily]:
    paths = sorted(programs_dir.glob("a[0-9]*_h[0-9]*.py"))
    if not paths:
        raise ValueError(f"no historical math programs found in {programs_dir}")
    variants = [
        read_program_variant(
            path,
            capability_names=capability_catalog.keys(),
            repo_root=repo_root,
        )
        for path in paths
    ]
    grouped: dict[str, list[ProgramVariant]] = {}
    for variant in variants:
        grouped.setdefault(variant.aspect_id, []).append(variant)
    families: list[ProgramFamily] = []
    for aspect_id, members in sorted(
        grouped.items(), key=lambda item: int(item[0].removeprefix("a"))
    ):
        members.sort(key=lambda member: (member.revision, member.path))
        revisions = [member.revision for member in members]
        if len(revisions) != len(set(revisions)):
            raise ValueError(f"duplicate revisions in {aspect_id}: {revisions}")
        families.append(
            ProgramFamily(
                aspect_id=aspect_id,
                selected_variant=members[-1],
                variants=tuple(members),
            )
        )
    return families


def _family_semantic_text(family: ProgramFamily) -> str:
    variant = family.selected_variant
    fallback = " ".join(name.replace("_", " ") for name in variant.llm_field_names)
    code_terms = " ".join(name.replace("_", " ") for name in variant.regex_constant_names)
    return " ".join(part for part in (variant.source_heading, fallback, code_terms) if part)


def _idf(families: Iterable[ProgramFamily]) -> dict[str, float]:
    docs = [set(_tokens(_family_semantic_text(family))) for family in families]
    frequency = Counter(token for doc in docs for token in doc)
    n = len(docs)
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
    overlap = _weighted_sum(shared, idf)
    precision = overlap / max(_weighted_sum(right, idf), 1e-12)
    recall = overlap / max(_weighted_sum(left, idf), 1e-12)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _coverage(query: set[str], document: set[str], idf: Mapping[str, float]) -> float:
    if not query:
        return 0.0
    return _weighted_sum(query & document, idf) / max(_weighted_sum(query, idf), 1e-12)


def _normalize_phrase(text: str) -> str:
    return " ".join(_tokens(text))


def _variant_record(variant: ProgramVariant) -> dict:
    return {
        "revision": variant.revision,
        "source_path": variant.path,
        "source_heading": variant.source_heading,
        "llm_field_names": list(variant.llm_field_names),
        "static_code": {
            "invoked_ops": list(variant.invoked_ops),
            "deep_math_ops": list(variant.deep_math_ops),
            "generic_surface_ops": list(variant.generic_surface_ops),
            "unknown_ops": list(variant.unknown_ops),
            "regex_constant_names": list(variant.regex_constant_names),
            "imported_module_roots": list(variant.imported_roots),
            "ast_node_count": variant.ast_node_count,
            "function_count": variant.function_count,
            "control_node_count": variant.control_node_count,
            "derived_depth": variant.code_depth,
            "derived_depth_meaning": DEPTH_MEANINGS[variant.code_depth],
            "program_shape": variant.program_shape,
        },
    }


def _score_candidate(cell: Mapping, family: ProgramFamily, idf: Mapping[str, float]) -> dict:
    variant = family.selected_variant
    query_title = set(_tokens(str(cell["construct"])))
    query_description = set(_tokens(str(cell["description"])))
    query_all = query_title | query_description
    candidate_title = set(_tokens(variant.source_heading))
    candidate_support = set(_tokens(_family_semantic_text(family)))
    title_f1 = _weighted_f1(query_title, candidate_title, idf)
    description_support = _coverage(query_description, candidate_support, idf)
    candidate_title_support = _coverage(candidate_title, query_all, idf)
    query_phrase = _normalize_phrase(str(cell["construct"]))
    candidate_phrase = _normalize_phrase(variant.source_heading)
    phrase_match = bool(
        query_phrase
        and candidate_phrase
        and (query_phrase in candidate_phrase or candidate_phrase in query_phrase)
    )
    semantic_score = min(
        1.0,
        0.60 * title_f1
        + 0.25 * description_support
        + 0.15 * candidate_title_support
        + (0.08 if phrase_match else 0.0),
    )
    depth_tiebreak = 0.015 if variant.code_depth >= 2 else 0.0
    shared_title = sorted(query_title & candidate_title)
    title_in_query = sorted(query_all & candidate_title)
    strong_shared_title = sorted(set(shared_title) - _WEAK_CONCEPTS)
    strong_title_in_query = sorted(set(title_in_query) - _WEAK_CONCEPTS)
    defensible = bool(
        variant.code_depth >= 2
        and candidate_title
        and (
            (semantic_score >= 0.12 and phrase_match)
            or (semantic_score >= 0.20 and len(shared_title) >= 2 and strong_shared_title)
            or (semantic_score >= 0.19 and len(title_in_query) >= 2 and strong_title_in_query)
            or (
                semantic_score >= 0.32
                and strong_shared_title
                and description_support >= 0.10
            )
        )
    )
    return {
        "aspect_id": family.aspect_id,
        "source_heading": variant.source_heading,
        "selected_revision": variant.revision,
        "source_path": variant.path,
        "semantic_score": round(semantic_score, 6),
        "rank_score": round(semantic_score + depth_tiebreak, 6),
        "score_components": {
            "title_weighted_f1": round(title_f1, 6),
            "description_support": round(description_support, 6),
            "candidate_title_support": round(candidate_title_support, 6),
            "exact_normalized_phrase_match": phrase_match,
            "shared_title_concepts": shared_title,
            "strong_shared_title_concepts": strong_shared_title,
            "candidate_title_concepts_in_query": title_in_query,
            "strong_candidate_title_concepts_in_query": strong_title_in_query,
        },
        "depth_provenance": {
            "derived_code_depth": variant.code_depth,
            "derived_code_depth_meaning": DEPTH_MEANINGS[variant.code_depth],
            "derived_program_shape": variant.program_shape,
            "deep_math_ops": list(variant.deep_math_ops),
            "derivation": (
                "static AST calls cross-referenced to public MathOps methods; program was not run"
            ),
            "limit": (
                "MathOps parses LaTeX/document structure; it is not SymPy, a proof assistant, "
                "formal proof checking, or answer execution"
            ),
        },
        "hybrid_provenance": {
            "llm_field_names": list(variant.llm_field_names),
            "has_prompt_based_subrelations": bool(variant.llm_field_names),
            "historical_construction": (
                "manual historical hybrid; source docs commonly disclose train-residual-informed "
                "construction, so no automatic-discovery provenance is inferred"
            ),
            "candidate_scope": (
                "historical hybrid family with static code subrelations; code-only whole-construct "
                "fidelity is not inferred"
            ),
        },
        "available_variants": [_variant_record(member) for member in family.variants],
        "variant_selection_policy": (
            "latest filename revision, selected without reading execution or performance results"
        ),
        "passes_source_only_seed_gate": defensible,
    }


def retrieve_for_cell(
    cell: Mapping,
    families: Sequence[ProgramFamily],
    *,
    top_k: int = 5,
) -> dict:
    if cell.get("task") != TASK:
        raise ValueError(f"this mapper is intentionally limited to {TASK} cells")
    idf = _idf(families)
    candidates = [_score_candidate(cell, family, idf) for family in families]
    candidates.sort(
        key=lambda row: (
            row["passes_source_only_seed_gate"],
            row["rank_score"],
            row["semantic_score"],
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
        "task": TASK,
        "level": str(cell["level"]),
        "metric_name": str(cell["construct"]),
        "metric_description": str(cell["description"]),
        "decision": (
            "candidate_seed_pending_independent_construct_fidelity_audit"
            if selected
            else "abstain"
        ),
        "selected_seed": selected,
        "top_source_only_candidates": alternatives,
        "interpretation": (
            "retrospective candidate retrieval only; not code fidelity, verifiability, "
            "codability, reconstruction, isomorphism, or automatic discovery"
        ),
    }


def build_seed_map(
    panel: Mapping,
    families: Sequence[ProgramFamily],
    *,
    capability_catalog: Mapping[str, CapabilityMetadata],
    top_k: int = 5,
) -> dict:
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise ValueError("expected tacit_breadth_metric_panel/v1")
    cells = [cell for cell in panel.get("cells", []) if cell.get("task") == TASK]
    counts = Counter(str(cell.get("level")) for cell in cells)
    if counts != Counter({"R1": 30, "R2": 30, "R3": 30}):
        raise ValueError(f"expected 30 {TASK} cells at each R1/R2/R3 level; found {counts}")
    required = {"id", "task", "level", "construct", "description"}
    for cell in cells:
        missing = sorted(required - cell.keys())
        if missing:
            raise ValueError(f"panel cell is missing source metadata: {missing}")
    if not families:
        raise ValueError("historical program family inventory is empty")

    rows = [retrieve_for_cell(cell, families, top_k=top_k) for cell in cells]
    selected = [row["selected_seed"] for row in rows if row["selected_seed"]]
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
        "status": "retrospective-candidate-seeds-pending-independent-construct-fidelity-audit",
        "design_scope": DESIGN_SCOPE,
        "panel_schema": panel["schema"],
        "panel_content_sha256": panel.get("panel_content_sha256"),
        "hierarchy_frame": panel.get("hierarchy_frame"),
        "task": TASK,
        "levels": ["R1", "R2", "R3"],
        "n_cells": len(rows),
        "n_historical_program_families": len(families),
        "n_historical_program_variants": sum(len(family.variants) for family in families),
        "input_fields_used": ["id", "task", "level", "construct", "description"],
        "forbidden_inputs": [
            "items or item identifiers",
            "reference judgments",
            "outcome labels",
            "heldout identifiers",
            "program outputs",
            "correlations or performance summaries",
            "reconstruction or isomorphism results",
        ],
        "provenance": {
            "retrieval": "deterministic source-only static retrieval in this run",
            "seed_source": "retrospective historical math hybrid bank",
            "original_program_authorship": (
                "manual and commonly train-residual-informed; not automatic discovery"
            ),
            "variant_policy": (
                "latest source revision by filename; no result/performance fields consulted"
            ),
            "candidate_execution": False,
            "construct_fidelity_adjudication": False,
            "prompt_articulability_evaluation": False,
        },
        "retrieval_policy": {
            "semantic_evidence": (
                "weighted concept overlap between panel construct/description and the frozen "
                "first source heading plus static field/constant identifiers"
            ),
            "deep_code_gate": (
                "candidate must derive depth>=2 from MathOps parser calls or a stateful local "
                "structural algorithm; depth is a source-shape claim, not a fidelity claim"
            ),
            "abstention": (
                "abstain unless explicit normalized concept/phrase support passes a frozen gate"
            ),
            "post_retrieval_gate": (
                "independent sub-relation construct-fidelity audit required before execution or "
                "any verifiability/reconstruction claim"
            ),
        },
        "capability_library": {
            "source_path": "methods/metric_seam/hybrids/ops_math.py",
            "public_math_ops": {
                name: {
                    "summary": metadata.summary,
                    "ast_node_count": metadata.ast_node_count,
                    "control_node_count": metadata.control_node_count,
                    "derived_depth": 2,
                }
                for name, metadata in sorted(capability_catalog.items())
            },
            "scope_limit": (
                "document-local LaTeX parsing/tokenization and structural aggregation only; "
                "no CAS, proof assistant, formal checking, or answer execution in this bank"
            ),
        },
        "summary": {
            "decision_counts": dict(sorted(Counter(row["decision"] for row in rows).items())),
            "by_level": by_level,
            "n_unique_selected_program_families": len(
                {seed["aspect_id"] for seed in selected}
            ),
            "selected_depth_counts": dict(sorted(Counter(
                str(seed["depth_provenance"]["derived_code_depth"]) for seed in selected
            ).items())),
            "selected_program_shape_counts": dict(sorted(Counter(
                seed["depth_provenance"]["derived_program_shape"] for seed in selected
            ).items())),
            "selected_with_prompt_subrelations": sum(
                seed["hybrid_provenance"]["has_prompt_based_subrelations"] for seed in selected
            ),
            "exact_whole_construct_code_fidelity_established": 0,
            "relation_local_code_fidelity_established": 0,
        },
        "rows": rows,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--programs-dir", type=Path, required=True)
    parser.add_argument("--ops-math", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >=1")
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --force to replace")
    repo_root = Path(__file__).resolve().parents[2]
    catalog = read_capability_catalog(args.ops_math)
    families = load_program_families(
        args.programs_dir,
        capability_catalog=catalog,
        repo_root=repo_root,
    )
    payload = build_seed_map(
        _load_json(args.panel),
        families,
        capability_catalog=catalog,
        top_k=args.top_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
