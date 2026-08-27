"""Retrieve outcome-blind historical patent seeds for the R1/R2/R3 panel.

This is a deliberately narrow inventory over the four programs in
``f2p_mock/programs_pa``.  It parses panel source text and Python syntax; it
does not import or execute the programs and does not read items, outcomes,
judge scores, correlations, or reconstruction results.

The historical programs are manual hybrids.  In addition to prompt fields,
they call a precomputed prior-art evidence operation whose candidate pool was
examiner/oracle conditioned and whose disclosure decisions were produced by a
reading model.  A retrieved seed is therefore only a candidate for a later
relation-local audit.  It is not a pure-code witness, autonomous retrieval,
whole-criterion fidelity, codability, or isomorphism.
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


SCHEMA = "metric-seam.hierarchy-patent-seed-map.v1"
TASK = "patents"
DESIGN_SCOPE = "outcome_blind_static_source_and_capability_metadata_only"

DEPTH_MEANINGS = {
    0: "non-executable catalog entry / bounded non-discovery",
    1: "surface text, regex, or generic document statistics",
    2: "document parser, structural aggregation, or stateful local algorithm",
    3: "retrieval or external-evidence pipeline",
    4: "sandboxed execution, test-running, or formal proof checking",
}

_STOPWORDS = {
    "a", "an", "and", "application", "applications", "as", "at", "be",
    "by", "claim", "claims", "criterion", "document", "documents", "for",
    "from", "h0", "hybrid", "in", "is", "it", "its", "of", "on", "or",
    "patent", "patents", "program", "requirement", "requirements", "score",
    "that", "the", "their", "this", "to", "use", "uses", "using", "when",
    "which", "with", "without",
}

_WEAK_CONCEPTS = {
    "appropriate", "assessment", "compliance", "content", "correct",
    "correctly", "ensure", "general", "include", "including", "practice",
    "quality", "specific", "substantive",
}

_ALIASES = {
    "anticipated": "anticipation",
    "anticipates": "anticipation",
    "anticipating": "anticipation",
    "bars": "bar",
    "disclosed": "disclosure",
    "discloses": "disclosure",
    "disclosing": "disclosure",
    "disclosures": "disclosure",
    "differentiate": "differentiation",
    "differentiated": "differentiation",
    "differentiates": "differentiation",
    "distinguish": "differentiation",
    "distinguished": "differentiation",
    "distinguishing": "differentiation",
    "industrial": "industry",
    "industrially": "industry",
    "inventive": "invention",
    "inventions": "invention",
    "new": "novelty",
    "non-obvious": "nonobviousness",
    "nonobvious": "nonobviousness",
    "non-obviousness": "nonobviousness",
    "novel": "novelty",
    "obvious": "obviousness",
    "publicly": "public",
    "references": "reference",
    "statutory": "statute",
    "steps": "step",
}


@dataclass(frozen=True)
class ProgramSeed:
    aspect_id: str
    revision: int
    path: str
    heading: str
    semantic_source: str
    llm_field_names: tuple[str, ...]
    invoked_ops: tuple[str, ...]
    regex_constant_names: tuple[str, ...]
    imported_roots: tuple[str, ...]
    ast_node_count: int
    control_node_count: int


def _first_paragraph(text: str) -> str:
    chunks = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    return " ".join(chunks[0].split()) if chunks and chunks[0].strip() else ""


def _canonical_token(raw: str) -> str:
    token = raw.casefold().strip("._+#-–—")
    token = _ALIASES.get(token, token)
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return _ALIASES.get(token, token)


def _tokens(text: str) -> list[str]:
    result: list[str] = []
    normalized = str(text).replace("§", " statute ")
    normalized = re.sub(r"[-‐‑‒–—]+", " ", normalized)
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9-]*", normalized):
        token = _canonical_token(raw)
        if len(token) >= 2 and token not in _STOPWORDS:
            result.append(token)
    return result


def _literal_llm_fields(tree: ast.Module) -> tuple[str, ...]:
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == "LLM_FIELDS" and value is not None:
            try:
                literal = ast.literal_eval(value)
            except (TypeError, ValueError):
                return ()
            if isinstance(literal, dict):
                return tuple(sorted(str(name) for name in literal))
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
    return tuple(sorted(names))


def _regex_constants(tree: ast.Module) -> tuple[str, ...]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and (
                "RE" in target.id or "PAT" in target.id or "REGEX" in target.id
            ):
                names.add(target.id)
    return tuple(sorted(names))


def _imported_roots(tree: ast.Module) -> tuple[str, ...]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return tuple(sorted(roots))


def _heading(docstring: str, aspect_id: str) -> str:
    paragraph = _first_paragraph(docstring)
    paragraph = re.sub(
        rf"^{re.escape(aspect_id)}\s*(?:(?:--?|—)\s*)?",
        "",
        paragraph,
        flags=re.I,
    )
    paragraph = re.split(r"\s*\(hybrid\b", paragraph, maxsplit=1, flags=re.I)[0]
    return paragraph.strip(" .:–—-")


def read_program_seed(path: Path, *, repo_root: Path | None = None) -> ProgramSeed:
    """Parse one historical program without importing or executing it."""
    match = re.fullmatch(r"(a\d+)_h(\d+)\.py", path.name)
    if match is None:
        raise ValueError(f"unexpected patent-program filename: {path}")
    aspect_id, revision_text = match.groups()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    docstring = ast.get_docstring(tree) or ""
    relative = path
    if repo_root is not None:
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            pass
    controls = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    return ProgramSeed(
        aspect_id=aspect_id,
        revision=int(revision_text),
        path=str(relative),
        heading=_heading(docstring, aspect_id),
        semantic_source=docstring,
        llm_field_names=_literal_llm_fields(tree),
        invoked_ops=_invoked_ops(tree),
        regex_constant_names=_regex_constants(tree),
        imported_roots=_imported_roots(tree),
        ast_node_count=sum(1 for _ in ast.walk(tree)),
        control_node_count=sum(isinstance(node, controls) for node in ast.walk(tree)),
    )


def load_program_seeds(
    programs_dir: Path, *, repo_root: Path | None = None
) -> list[ProgramSeed]:
    paths = sorted(programs_dir.glob("a[0-9]*_h[0-9]*.py"))
    if not paths:
        raise ValueError(f"no patent programs found in {programs_dir}")
    seeds = [read_program_seed(path, repo_root=repo_root) for path in paths]
    ids = [seed.aspect_id for seed in seeds]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate patent program families are not supported: {ids}")
    return seeds


def _idf(programs: Iterable[ProgramSeed]) -> dict[str, float]:
    docs = [set(_tokens(program.semantic_source)) for program in programs]
    frequency = Counter(token for doc in docs for token in doc)
    n_docs = len(docs)
    return {
        token: math.log((n_docs + 1.0) / (count + 1.0)) + 1.0
        for token, count in frequency.items()
    }


def _weight(tokens: Iterable[str], idf: Mapping[str, float]) -> float:
    return sum(idf.get(token, math.log(2.0)) for token in set(tokens))


def _coverage(query: set[str], document: set[str], idf: Mapping[str, float]) -> float:
    if not query:
        return 0.0
    return _weight(query & document, idf) / max(_weight(query, idf), 1e-12)


def _weighted_f1(left: set[str], right: set[str], idf: Mapping[str, float]) -> float:
    shared = left & right
    if not shared:
        return 0.0
    overlap = _weight(shared, idf)
    precision = overlap / max(_weight(right, idf), 1e-12)
    recall = overlap / max(_weight(left, idf), 1e-12)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _normalize_phrase(text: str) -> str:
    return " ".join(_tokens(text))


def _score(cell: Mapping, program: ProgramSeed, idf: Mapping[str, float]) -> dict:
    title = set(_tokens(str(cell["construct"])))
    description = set(_tokens(str(cell["description"])))
    query = title | description
    program_title = set(_tokens(program.heading))
    program_document = set(_tokens(program.semantic_source))
    title_f1 = _weighted_f1(title, program_title, idf)
    description_coverage = _coverage(description, program_document, idf)
    heading_coverage = _coverage(program_title, query, idf)
    query_phrase = _normalize_phrase(str(cell["construct"]))
    heading_phrase = _normalize_phrase(program.heading)
    phrase_match = bool(
        query_phrase
        and heading_phrase
        and (query_phrase in heading_phrase or heading_phrase in query_phrase)
    )
    shared_heading = sorted(title & program_title)
    heading_in_query = sorted(query & program_title)
    strong_shared = sorted(set(shared_heading) - _WEAK_CONCEPTS)
    strong_in_query = sorted(set(heading_in_query) - _WEAK_CONCEPTS)
    score = min(
        1.0,
        0.62 * title_f1
        + 0.23 * description_coverage
        + 0.15 * heading_coverage
        + (0.08 if phrase_match else 0.0),
    )
    # The bank has only four highly specific families.  Require heading-level
    # relation support so a generic mention of "prior art" in a long panel
    # description cannot manufacture a candidate.
    defensible = bool(
        "prior_art" in program.invoked_ops
        and (
            (phrase_match and score >= 0.18)
            or (score >= 0.25 and len(strong_shared) >= 2)
            or (score >= 0.27 and len(strong_in_query) >= 2)
            or (
                score >= 0.31
                and len(strong_shared) >= 1
                and description_coverage >= 0.20
            )
        )
    )
    return {
        "aspect_id": program.aspect_id,
        "source_path": program.path,
        "source_heading": program.heading,
        "semantic_score": round(score, 6),
        "score_components": {
            "title_weighted_f1": round(title_f1, 6),
            "description_coverage": round(description_coverage, 6),
            "program_heading_coverage": round(heading_coverage, 6),
            "exact_normalized_phrase_match": phrase_match,
            "shared_heading_concepts": shared_heading,
            "strong_shared_heading_concepts": strong_shared,
            "program_heading_concepts_in_query": heading_in_query,
            "strong_program_heading_concepts_in_query": strong_in_query,
        },
        "static_program": {
            "revision": program.revision,
            "llm_field_names": list(program.llm_field_names),
            "invoked_ops": list(program.invoked_ops),
            "regex_constant_names": list(program.regex_constant_names),
            "imported_module_roots": list(program.imported_roots),
            "ast_node_count": program.ast_node_count,
            "control_node_count": program.control_node_count,
        },
        "depth_provenance": {
            "derived_program_depth": 3,
            "derived_program_depth_meaning": DEPTH_MEANINGS[3],
            "local_text_code_depth": 1,
            "evidence_operation": "prior_art",
            "derivation": (
                "static AST call to prior_art plus regex/local aggregation; program was not run"
            ),
            "channel_warning": (
                "depth 3 belongs to a precomputed retrieval-plus-reading-model evidence channel, "
                "not a pure-code/CAS/formal-verification channel"
            ),
        },
        "provenance": {
            "historical_construction": "retrospective manual hybrid seed",
            "prompt_subrelations_present": bool(program.llm_field_names),
            "evidence_subrelation_present": True,
            "evidence_candidate_pool": "examiner/oracle conditioned",
            "evidence_relation_labels": "precomputed reading-model disclosure verdicts",
            "autonomous_retrieval": False,
            "pure_code_witness": False,
        },
        "passes_source_only_seed_gate": defensible,
    }


def retrieve_for_cell(
    cell: Mapping, programs: Sequence[ProgramSeed], *, top_k: int = 4
) -> dict:
    if cell.get("task") != TASK:
        raise ValueError(f"this mapper is intentionally limited to {TASK} cells")
    idf = _idf(programs)
    candidates = [_score(cell, program, idf) for program in programs]
    candidates.sort(
        key=lambda row: (
            row["passes_source_only_seed_gate"],
            row["semantic_score"],
            row["aspect_id"],
        ),
        reverse=True,
    )
    gated = [row for row in candidates if row["passes_source_only_seed_gate"]]
    selected = gated[0] if gated else None
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
        "top_source_only_candidates": candidates[:top_k],
        "interpretation": (
            "retrospective candidate retrieval only; not relation fidelity, pure-code "
            "verifiability, codability, autonomous discovery, reconstruction, or isomorphism"
        ),
    }


def build_seed_map(panel: Mapping, programs: Sequence[ProgramSeed], *, top_k: int = 4) -> dict:
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise ValueError("expected tacit_breadth_metric_panel/v1")
    cells = [cell for cell in panel.get("cells", []) if cell.get("task") == TASK]
    counts = Counter(str(cell.get("level")) for cell in cells)
    if counts != Counter({"R1": 30, "R2": 30, "R3": 30}):
        raise ValueError(f"expected 30 patent cells at each R1/R2/R3 level; found {counts}")
    required = {"id", "task", "level", "construct", "description"}
    for cell in cells:
        missing = sorted(required - cell.keys())
        if missing:
            raise ValueError(f"panel cell is missing source metadata: {missing}")
    if not programs:
        raise ValueError("historical patent program inventory is empty")

    rows = [retrieve_for_cell(cell, programs, top_k=top_k) for cell in cells]
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
        "status": "retrospective-candidate-seeds-pending-independent-construct-fidelity-audit",
        "design_scope": DESIGN_SCOPE,
        "task": TASK,
        "panel_schema": panel["schema"],
        "panel_content_sha256": panel.get("panel_content_sha256"),
        "hierarchy_frame": panel.get("hierarchy_frame"),
        "n_cells": len(rows),
        "n_historical_program_families": len(programs),
        "input_fields_used": ["id", "task", "level", "construct", "description"],
        "forbidden_inputs": [
            "items or item identifiers",
            "reference judgments or outcome labels",
            "heldout identifiers",
            "program outputs or correlations",
            "reconstruction or isomorphism results",
        ],
        "provenance": {
            "retrieval": "deterministic source-only static retrieval in this run",
            "seed_source": "retrospective four-program patent prior-art hybrid bank",
            "original_program_authorship": "manual; not automatic discovery",
            "candidate_execution": False,
            "construct_fidelity_adjudication": False,
            "prompt_articulability_evaluation": False,
            "oracle_conditioning_retained": True,
        },
        "retrieval_policy": {
            "semantic_evidence": (
                "weighted overlap between panel construct/description and program headings/docstrings"
            ),
            "strict_heading_gate": (
                "a candidate requires phrase match or multiple strong heading concepts; generic "
                "prior-art mentions in descriptions are insufficient"
            ),
            "post_retrieval_gate": (
                "independent relation-local construct-fidelity audit required before execution "
                "or any verifiability/reconstruction claim"
            ),
        },
        "capability_scope": {
            "operation": "prior_art",
            "program_depth": 3,
            "depth_meaning": DEPTH_MEANINGS[3],
            "implementation": (
                "lookup of already-computed BM25/dense candidates and reading-model "
                "claim-element disclosure records"
            ),
            "known_oracle_caveat": (
                "examiner-cited documents were force-included for targeted claims; stripping "
                "gold fields does not make candidate construction autonomous"
            ),
            "known_multiplicity_caveat": (
                "historical feature payload contains duplicate claim-element rows and weights "
                "summaries by extraction multiplicity"
            ),
            "channel_class": "external_evidence_plus_model_assisted_relation_labels",
            "pure_code": False,
        },
        "summary": {
            "decision_counts": dict(sorted(Counter(row["decision"] for row in rows).items())),
            "by_level": by_level,
            "n_unique_selected_program_families": len(
                {seed["aspect_id"] for seed in selected}
            ),
            "selected_with_prompt_subrelations": sum(
                seed["provenance"]["prompt_subrelations_present"] for seed in selected
            ),
            "selected_with_oracle_conditioned_evidence": len(selected),
            "exact_whole_construct_code_fidelity_established": 0,
            "relation_local_fidelity_established": 0,
            "pure_code_witnesses_established": 0,
        },
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--programs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    programs = load_program_seeds(args.programs_dir, repo_root=repo_root)
    result = build_seed_map(panel, programs, top_k=args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
