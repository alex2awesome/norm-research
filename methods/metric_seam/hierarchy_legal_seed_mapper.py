"""Build an outcome-blind source map from frozen Title VII programs to legal panel cells.

This is retrieval, not construct-fidelity adjudication.  It parses source files
and operative criterion text only.  In particular it never imports a historical
program, loads items, prompt outputs, judge/reference values, or outcomes.

The historical source criteria and the hierarchy target criteria are different:
``programs_legal`` implements Title VII fact-pattern predicates, while the
hierarchy panel asks about legal writing.  Candidate transfer is therefore
capability-local (dates, quantities, concrete facts, actor/quotation structure),
not aspect-ID or whole-score transfer.
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

from methods.metric_seam.legal_hierarchy_projection import RELATIONS


SCHEMA = "metric-seam.hierarchy-legal-capability-seed-map.v1"
TASK = "legal-outcome-prediction"
DESIGN_SCOPE = "source_only_manual_retrospective_capability_retrieval"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"
DEFAULT_PROGRAMS = ROOT / "methods/metric_seam/hybrids/programs_legal"
DEFAULT_ASPECTS = ROOT / "runs/validity_full/v2/legal_title_vii/aspects.json"
DEFAULT_ASPECTS_USED = ROOT / "outputs/metric_seam_pilot/tasks/legal_title_vii/aspects_used.json"
DEFAULT_OUTPUT = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/legal_capability_seed_map_v1.json"


_TRIGGERS: dict[str, tuple[str, ...]] = {
    "plain_language_surface": (
        "plain english", "plain language", "plainness", "legalese", "diction", "familiar words",
        "plain-language", "wordy", "economy", "concise single-word", "non-academic voice",
    ),
    "sentence_clarity_parse": (
        "short sentence", "sentence clarity", "sentence mechanics", "sentence length",
        "complex syntax", "controlled embedding", "clear sentences", "readability",
        "respect limited attention", "economy and judicial efficiency",
    ),
    "active_voice_parse": ("active prose", "active verb", "active voice", "nominalization", "buried verb", "actor-forward"),
    "negation_stack_parse": ("multiple negatives", "stacking negatives", "positive phrasing"),
    "concrete_fact_anchors": (
        "concrete specificity", "concrete details", "concrete fact", "concrete examples",
        "specific dates", "concrete nouns", "numbers", "fact presentation",
    ),
    "temporal_order_graph": ("chronological", "chronology", "timeline", "temporal", "sequence"),
    "numeric_consistency_check": ("numbers", "dates", "details consistent", "mechanical correctness", "accuracy"),
    "definition_use_graph": ("defined terms", "definitions", "acronyms", "terminology", "definition completeness"),
    "citation_format_structure": (
        "citation", "citations", "authority", "authorities", "precedent", "quotation", "footnote",
        "parenthetical", "record citations",
    ),
    "quote_attribution_parse": ("quotation", "quote", "speaker", "attribution"),
    "discourse_cohesion_graph": (
        "coherent", "coherence", "logical flow", "central theme", "theme reinforcement",
        "case theory", "narrative structure", "narrative and persuasion", "narrative nonfiction",
        "controlling theme", "reinforce it", "transitions",
    ),
    "paragraph_cohesion_graph": ("paragraph-level cohesion", "topic sentences", "one point per", "one idea paragraphs"),
    "frontloaded_disposition_structure": (
        "front-load", "front loading", "up front", "point-first", "bluf", "opening strongest",
        "requested relief", "relief statement", "end decisively", "primacy", "clearly and immediately",
        "early and succinctly",
    ),
    "counterposition_structure": (
        "counterargument", "counterarguments", "opposing arguments", "refutation", "rebuttal",
        "points of dispute", "adversarial awareness",
    ),
    "tone_restraint_surface": (
        "professional tone", "civil tone", "civility", "restrained", "hyperbole", "personal attacks",
        "incendiary", "respectful", "dignified", "professional voice", "plain, professional",
        "calibrate tone/professionalism",
    ),
    "heading_roadmap_structure": (
        "headings", "point headings", "subheadings", "roadmap", "summary of the argument",
        "signposting", "labeled sections", "parallel structure", "arrange sections",
    ),
    "question_frame_structure": (
        "question presented", "questions presented", "issue framing", "frame the issue", "yes/no",
        "single, clear sentence", "crisp issue", "state the legal issue", "issue clearly and immediately",
    ),
    "inclusive_language_surface": ("inclusive language", "bias-free", "gender-neutral"),
    "deadline_remedy_consequence_structure": (
        "demand letters", "cease-and-desist", "remedy", "deadline", "consequences",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _literal_llm_fields(tree: ast.Module) -> list[str]:
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == "LLM_FIELDS" and value is not None:
            try:
                literal = ast.literal_eval(value)
            except (TypeError, ValueError):
                return []
            return sorted(str(key) for key in literal) if isinstance(literal, dict) else []
    return []


def _invoked_ops(tree: ast.Module) -> list[str]:
    return sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ops"
        }
    )


def read_historical_programs(programs_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(programs_dir.glob("a[0-9]*_h0.py"), key=lambda value: int(re.search(r"a(\d+)", value.name).group(1))):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        aspect_id = path.stem.split("_", 1)[0]
        rows.append(
            {
                "aspect_id": aspect_id,
                "source_path": _relative(path),
                "source_sha256": _sha256(path),
                "docstring_first_paragraph": " ".join((ast.get_docstring(tree) or "").split("\n\n", 1)[0].split()),
                "llm_field_names": _literal_llm_fields(tree),
                "invoked_ops": _invoked_ops(tree),
                "ast_node_count": sum(1 for _ in ast.walk(tree)),
                "source_executed": False,
            }
        )
    if len(rows) != 20 or len({row["aspect_id"] for row in rows}) != 20:
        raise ValueError("expected exactly 20 unique frozen legal h0 programs")
    return rows


def read_operative_aspects(aspects_path: Path, used_path: Path) -> list[dict[str, Any]]:
    used = json.loads(used_path.read_text(encoding="utf-8"))
    aspects = json.loads(aspects_path.read_text(encoding="utf-8"))
    by_id = {row["aspect_id"]: row for row in aspects}
    if len(used) != 20 or set(used) - set(by_id):
        raise ValueError("historical aspects_used does not bind 20 source definitions")
    rows = []
    for aspect_id in used:
        row = by_id[aspect_id]
        description = row["description"]
        rows.append(
            {
                "aspect_id": aspect_id,
                "name": row["name"],
                "description": description,
                "description_chars": len(description),
                "description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
                "exactly_600_chars": len(description) == 600,
                "operative_construct_preserved_verbatim": True,
            }
        )
    if sum(row["exactly_600_chars"] for row in rows) != 10:
        raise ValueError("known legal 600-character construct boundary drifted")
    return rows


def _score_candidates(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = f"{cell['construct']} {cell['description']}".casefold().replace("‑", "-")
    candidates = []
    by_relation = {row["relation_id"]: row for row in RELATIONS}
    for relation_id, triggers in _TRIGGERS.items():
        matched = sorted({trigger for trigger in triggers if trigger in text})
        if not matched:
            continue
        relation = by_relation[relation_id]
        candidates.append(
            {
                "relation_id": relation_id,
                "matched_source_phrases": matched,
                "retrieval_score": len(matched),
                "historical_seed_ids": relation["historical_seed_ids"],
                "requires_independent_construct_fidelity_audit": True,
            }
        )
    return sorted(candidates, key=lambda row: (-row["retrieval_score"], row["relation_id"]))


def build_seed_map(
    panel: Mapping[str, Any],
    *,
    historical_programs: Sequence[Mapping[str, Any]],
    operative_aspects: Sequence[Mapping[str, Any]],
    projection_path: Path,
) -> dict[str, Any]:
    cells = [row for row in panel.get("cells", []) if row.get("task") == TASK]
    if len(cells) != 90 or Counter(row["level"] for row in cells) != Counter({"R1": 30, "R2": 30, "R3": 30}):
        raise ValueError("legal hierarchy source map requires exactly 30 R1/R2/R3 cells")
    program_by_id = {row["aspect_id"]: row for row in historical_programs}
    if set(program_by_id) != {row["aspect_id"] for row in operative_aspects}:
        raise ValueError("historical programs and operative constructs are not one-to-one")
    rows = []
    for cell in cells:
        candidates = _score_candidates(cell)
        for candidate in candidates:
            candidate["historical_sources"] = [
                {
                    "aspect_id": aspect_id,
                    "source_path": program_by_id[aspect_id]["source_path"],
                    "source_sha256": program_by_id[aspect_id]["source_sha256"],
                }
                for aspect_id in candidate["historical_seed_ids"]
            ]
        rows.append(
            {
                "cell_id": cell["id"],
                "level": cell["level"],
                "selection_rank": cell["selection_rank"],
                "construct": cell["construct"],
                "description": cell["description"],
                "candidates": candidates,
                "retrieval_status": "candidate_relations_found" if candidates else "bounded_non_discovery_in_frozen_source_map",
            }
        )
    relation_counts = Counter(
        candidate["relation_id"] for row in rows for candidate in row["candidates"]
    )
    return {
        "schema": SCHEMA,
        "status": "source-map-complete-awaiting-independent-fidelity-audit",
        "task": TASK,
        "design_scope": DESIGN_SCOPE,
        "panel_content_sha256": panel.get("panel_content_sha256"),
        "historical_construct_binding": {
            "source_aspects_path": _relative(DEFAULT_ASPECTS),
            "source_aspects_sha256": _sha256(DEFAULT_ASPECTS),
            "source_aspects_used_path": _relative(DEFAULT_ASPECTS_USED),
            "source_aspects_used_sha256": _sha256(DEFAULT_ASPECTS_USED),
            "n_operative_aspects": len(operative_aspects),
            "n_exactly_600_character_descriptions": sum(row["exactly_600_chars"] for row in operative_aspects),
            "truncated_descriptions_repaired": False,
            "interpretation": "the ten 600-character source definitions remain the operative historical constructs because the original judge saw those same bytes",
            "rows": list(operative_aspects),
        },
        "historical_program_inventory": {
            "n_programs": len(historical_programs),
            "programs_imported_or_executed": False,
            "rows": list(historical_programs),
        },
        "additive_projection": {
            "source_path": _relative(projection_path),
            "source_sha256": _sha256(projection_path),
            "historical_files_modified": False,
            "whole_historical_scores_transferred": False,
            "relation_catalog": list(RELATIONS),
        },
        "separation": {
            "prompt_articulability_measured": False,
            "code_verifiability_measured": False,
            "reconstruction_measured": False,
            "isomorphism_measured": False,
            "outcomes_or_reference_values_loaded": False,
            "items_or_heldout_identifiers_loaded": False,
            "external_supervision_used": False,
        },
        "summary": {
            "n_cells": len(rows),
            "by_level": dict(sorted(Counter(row["level"] for row in rows).items())),
            "n_cells_with_candidates": sum(bool(row["candidates"]) for row in rows),
            "n_bounded_non_discovery": sum(not row["candidates"] for row in rows),
            "candidate_relation_counts": dict(sorted(relation_counts.items())),
        },
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--programs", type=Path, default=DEFAULT_PROGRAMS)
    parser.add_argument("--aspects", type=Path, default=DEFAULT_ASPECTS)
    parser.add_argument("--aspects-used", type=Path, default=DEFAULT_ASPECTS_USED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    programs = read_historical_programs(args.programs)
    aspects = read_operative_aspects(args.aspects, args.aspects_used)
    projection_path = ROOT / "methods/metric_seam/legal_hierarchy_projection.py"
    payload = build_seed_map(
        panel,
        historical_programs=programs,
        operative_aspects=aspects,
        projection_path=projection_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
