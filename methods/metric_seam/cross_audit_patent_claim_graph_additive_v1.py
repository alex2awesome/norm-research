"""Independent cross-audit of the additive patent claim-graph lane.

This module is additive: it does not rewrite the audited program, its source
audit, freeze, executions, or operational summary.  It replays those artifacts
from their bound text-only sources, checks relation certificates against the
parsed claim graph, runs adversarial safety probes, and separates two questions:

1. Does the proposed relation map narrowly to the requested panel construct?
2. Is the current implementation safe enough to count as executable coverage?

No references, outcomes, prompts, models, APIs, prior art, examiner evidence,
or external supervision are loaded.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from methods.metric_seam.hierarchy_patent_claim_graph_additive_freeze_v1 import (
    build_freeze,
)
from methods.metric_seam.hierarchy_patent_claim_graph_additive_operational_v1 import (
    build_summary,
)
from methods.metric_seam.hierarchy_patent_claim_graph_additive_runner_v1 import (
    build_execution,
)
from methods.metric_seam.patent_claim_graph_additive_v1 import (
    _MARKUSH_RE,
    _ancestor_numbers,
    analyze_patent_claim_graph,
)
from methods.metric_seam.patent_claim_structure import parse_claims


SCHEMA = "metric-seam.patent-claim-graph-additive-cross-audit.v1"

OUT_REL = Path("outputs/metric_seam_pilot/hierarchy_r123")
PATHS = {
    "panel": OUT_REL / "panel_v3.json",
    "canonical": OUT_REL / "patents_claim_structure_construct_fidelity_v1.json",
    "historical": OUT_REL / "patents_construct_fidelity_v1.json",
    "construct_audit": OUT_REL / "patents_claim_graph_additive_construct_fidelity_v1.json",
    "train": OUT_REL / "patents_claim_graph_additive_compiler_train_v2.json",
    "superseded_train": OUT_REL / "patents_claim_graph_additive_compiler_train_v1.json",
    "freeze": OUT_REL / "patents_claim_graph_additive_train_freeze_v1.json",
    "heldout": OUT_REL / "patents_claim_graph_additive_heldout_pre_reference_v1.json",
    "operational": OUT_REL / "patents_claim_graph_additive_operational_summary_v1.json",
    "output": OUT_REL / "patents_claim_graph_additive_cross_audit_v1.json",
}

RETAINED_RELATIONS = {
    "claim_status_and_local_listing_witnesses",
    "two_part_or_jepson_structure",
    "markush_closed_group_structure",
    "bounded_antecedent_term_reference_graph",
}
QUARANTINED_RELATIONS = {
    "numeric_constraint_definition_graph",
    "formula_variable_definition_alignment",
}

RELATION_ADJUDICATION = {
    "claim_status_and_local_listing_witnesses": {
        "implementation_disposition": "retain",
        "audited_depth": 2,
        "object": "pass",
        "relation": "pass",
        "polarity": "pass",
        "applicability": "pass",
        "aggregation": "pass_nonaggregating_certificates",
        "depth": "pass",
        "reason": (
            "recognized status parentheticals and duplicate presented ordinals are finite, "
            "local, correctly polarized witnesses; no list-completeness conclusion is made"
        ),
    },
    "two_part_or_jepson_structure": {
        "implementation_disposition": "retain",
        "audited_depth": 2,
        "object": "pass",
        "relation": "pass",
        "polarity": "pass",
        "applicability": "pass",
        "aggregation": "pass_nonaggregating_certificates",
        "depth": "pass",
        "reason": (
            "exact boundaries inside independently parsed claims support the narrow structural "
            "claim; they do not decide appropriateness, admitted prior art, or novelty"
        ),
    },
    "markush_closed_group_structure": {
        "implementation_disposition": "retain_with_fail_closed_truncation_filter",
        "audited_depth": 2,
        "object": "pass_after_filter",
        "relation": "pass_after_filter",
        "polarity": "pass_positive_only",
        "applicability": "fail_current_program_on_undelimited_capped_tail",
        "aggregation": "pass_nonaggregating_certificates",
        "depth": "pass",
        "reason": (
            "the exact opener and finite-list relation are sound, but the program emits a node "
            "when a 4k-capped final claim ends mid-list; cross-audit counts exclude such nodes"
        ),
    },
    "bounded_antecedent_term_reference_graph": {
        "implementation_disposition": "retain_with_duplicate_ordinal_scope_caveat",
        "audited_depth": 3,
        "object": "pass",
        "relation": "pass_on_replayed_receipts",
        "polarity": "pass_bounded_grammar_only",
        "applicability": "pass_on_replayed_receipts_adversarial_duplicate_bug_recorded",
        "aggregation": "fail_declared_none_but_unused_resolution_ratio_emitted",
        "depth": "pass",
        "reason": (
            "actual candidate edges replay within the same claim instance or an explicit "
            "ancestor.  A synthetic duplicate ordinal can cross-link instances, so duplicate "
            "ordinals must fail closed.  The downstream pipeline uses certificates, not the "
            "undeclared diagnostic resolution ratio"
        ),
    },
    "numeric_constraint_definition_graph": {
        "implementation_disposition": "quarantine_pending_relation_scope_repair",
        "audited_depth": 2,
        "object": "pass_constraint_and_definition_nodes",
        "relation": "fail_global_definition_candidate_pool",
        "polarity": "pass_positive_only",
        "applicability": "fail_no_same_claim_or_ancestor_restriction",
        "aggregation": "pass_nonaggregating_certificates",
        "depth": "fail_declared_3_current_algorithm_is_span_parse_plus_global_match",
        "reason": (
            "two of four train links already point outside the constraint claim's explicit "
            "dependency ancestry, and an adversarial example links unrelated claims solely by "
            "parameter key.  The code performs no dependency traversal, so current depth is 2 "
            "rather than 3"
        ),
    },
    "formula_variable_definition_alignment": {
        "implementation_disposition": "quarantine_pending_parser_repair",
        "audited_depth": 3,
        "object": "fail_separator_letters_and_indexed_symbol_collapse",
        "relation": "fail_certificate_class_not_adversarially_safe",
        "polarity": "fail_numeric_string_equivalence_can_create_false_conflicts",
        "applicability": "pass_explicit_assignment_definition_grammar",
        "aggregation": "pass_nonaggregating_certificates",
        "depth": "pass",
        "reason": (
            "the only observed positive link is valid, but definition tokenization treats the "
            "letters in 'and/or' as symbols, indexed variables collapse to their first letter, "
            "and numeric string variants can become false contradiction witnesses"
        ),
    },
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partial_ids(artifact: Mapping) -> set[str]:
    return {
        row["cell_id"]
        for row in artifact["rows"]
        if row["verdict"] == "partial_relation_local"
    }


def _verify_bound_sources(root: Path, execution: Mapping) -> dict:
    checks = {}
    for name in ("items", "manifest", "program", "runner"):
        source = execution["sources"][name]
        path = root / source["path"]
        actual = _sha256(path)
        checks[name] = {
            "path": source["path"],
            "recorded_sha256": source["sha256"],
            "actual_sha256": actual,
            "matches": actual == source["sha256"],
        }
    return checks


def _replay_execution(root: Path, receipt: Mapping) -> bool:
    item_path = root / receipt["sources"]["items"]["path"]
    manifest_path = root / receipt["sources"]["manifest"]["path"]
    item_bytes = item_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    replay = build_execution(
        json.loads(item_bytes),
        json.loads(manifest_bytes),
        phase=receipt["phase"],
        item_source_bytes=item_bytes,
        manifest_source_bytes=manifest_bytes,
        item_source_path=receipt["sources"]["items"]["path"],
        manifest_source_path=receipt["sources"]["manifest"]["path"],
    )
    return replay == receipt


def _mapping_rows(audit: Mapping) -> list[dict]:
    rows = []
    for cell in audit["rows"]:
        for mapping in cell.get("matched_relations", []):
            adjudication = RELATION_ADJUDICATION[mapping["relation_id"]]
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "level": cell["level"],
                    "selection_rank": cell["selection_rank"],
                    "construct": cell["construct"],
                    "relation_id": mapping["relation_id"],
                    "requested_subrelation": mapping["requested_subrelation"],
                    "original_depth": mapping["depth"],
                    "construct_mapping_disposition": "retain_relation_local_mapping",
                    **adjudication,
                    "counts_as_current_executable_coverage": (
                        mapping["relation_id"] in RETAINED_RELATIONS
                    ),
                }
            )
    return rows


def _claim_ancestors(claims: Sequence, number: int) -> set[int]:
    counts = Counter(claim.number for claim in claims)
    by_number = {claim.number: claim for claim in claims if counts[claim.number] == 1}
    return _ancestor_numbers(number, by_number)


_MENTION_ID_RE = re.compile(r"p(?P<instance>\d+):c(?P<claim>\d+):m\d+")


def _term_receipt_validation(execution: Mapping, items: Mapping[str, str]) -> dict:
    n_edges = 0
    violations = []
    duplicate_items = []
    for row in execution["rows"]:
        if not row.get("result"):
            continue
        claims = parse_claims(items[row["item_key"]])[1]
        counts = Counter(claim.number for claim in claims)
        duplicates = sorted(number for number, count in counts.items() if count > 1)
        if duplicates:
            duplicate_items.append(
                {"item_key": row["item_key"], "duplicate_ordinals": duplicates}
            )
        graph = row["result"]["graphs"]["term_reference"]
        nodes = {node["mention_id"]: node for node in graph["nodes"]}
        for edge in graph["edges"]:
            n_edges += 1
            ref_match = _MENTION_ID_RE.fullmatch(edge["reference_id"])
            if ref_match is None:
                violations.append(
                    {"item_key": row["item_key"], "reason": "unparseable_reference_id"}
                )
                continue
            ref_instance = int(ref_match.group("instance"))
            ref_claim = int(ref_match.group("claim"))
            ref_node = nodes[edge["reference_id"]]
            ancestors = _claim_ancestors(claims, ref_claim)
            for candidate_id in edge["candidate_introduction_ids"]:
                candidate_match = _MENTION_ID_RE.fullmatch(candidate_id)
                candidate = nodes[candidate_id]
                same_instance_earlier = (
                    candidate_match is not None
                    and int(candidate_match.group("instance")) == ref_instance
                    and candidate["span"][0] < ref_node["span"][0]
                )
                ancestor = candidate["claim"] in ancestors
                if not (same_instance_earlier or ancestor):
                    violations.append(
                        {
                            "item_key": row["item_key"],
                            "reference_id": edge["reference_id"],
                            "candidate_id": candidate_id,
                            "reason": "candidate_is_not_same_instance_earlier_or_ancestor",
                        }
                    )
    return {
        "n_edges_replayed": n_edges,
        "n_candidate_scope_violations": len(violations),
        "violations": violations,
        "duplicate_ordinal_items": duplicate_items,
    }


def _numeric_receipt_validation(execution: Mapping, items: Mapping[str, str]) -> dict:
    n_links = 0
    violations = []
    for row in execution["rows"]:
        if not row.get("result"):
            continue
        claims = parse_claims(items[row["item_key"]])[1]
        graph = row["result"]["graphs"]["numeric_constraint_definition"]
        constraints = {node["constraint_id"]: node for node in graph["constraint_nodes"]}
        definitions = {node["definition_id"]: node for node in graph["definition_nodes"]}
        for link in graph["links"]:
            n_links += 1
            constraint = constraints[link["constraint_id"]]
            definition = definitions[link["definition_id"]]
            allowed = _claim_ancestors(claims, constraint["claim"]) | {
                constraint["claim"]
            }
            if definition["claim"] not in allowed:
                violations.append(
                    {
                        "item_key": row["item_key"],
                        "constraint_id": constraint["constraint_id"],
                        "definition_id": definition["definition_id"],
                        "reason": "definition_claim_is_not_same_or_explicit_ancestor",
                    }
                )
    return {
        "n_links_replayed": n_links,
        "n_observed_scope_violations": len(violations),
        "violations": violations,
        "program_still_quarantined_by_adversarial_probe": True,
    }


def _valid_definition_symbols(surface: str) -> set[str]:
    lead = re.split(r"\s+(?:is|are)\s+", surface, maxsplit=1, flags=re.I)[0]
    return {
        token.casefold()
        for token in re.split(r"\s*(?:,|\band\b|\bor\b)\s*", lead, flags=re.I)
        if re.fullmatch(r"[A-Za-z]", token)
    }


def _formula_receipt_validation(execution: Mapping, items: Mapping[str, str]) -> dict:
    links = 0
    link_scope_violations = []
    ghost_nodes = []
    seen_definition_symbols = set()
    for row in execution["rows"]:
        if not row.get("result"):
            continue
        claims = parse_claims(items[row["item_key"]])[1]
        graph = row["result"]["graphs"]["formula_variable_definition"]
        assignments = {node["assignment_id"]: node for node in graph["assignment_nodes"]}
        definitions = {node["definition_id"]: node for node in graph["definition_nodes"]}
        for definition in definitions.values():
            identity = (
                row["item_key"],
                definition["claim"],
                tuple(definition["span"]),
                definition["symbol"],
            )
            invalid_separator_letter = (
                definition["symbol"]
                not in _valid_definition_symbols(definition["surface"])
            )
            duplicate_from_same_clause = identity in seen_definition_symbols
            if invalid_separator_letter or duplicate_from_same_clause:
                ghost_nodes.append(
                    {
                        "item_key": row["item_key"],
                        "definition_id": definition["definition_id"],
                        "symbol": definition["symbol"],
                        "surface": definition["surface"],
                        "reason": (
                            "separator_letters_parsed_as_symbols"
                            if invalid_separator_letter
                            else "duplicate_symbol_from_same_definition_clause"
                        ),
                    }
                )
            seen_definition_symbols.add(identity)
        for link in graph["links"]:
            links += 1
            assignment = assignments[link["assignment_id"]]
            definition = definitions[link["definition_id"]]
            allowed = _claim_ancestors(claims, assignment["claim"]) | {
                assignment["claim"]
            }
            if (
                definition["claim"] not in allowed
                or definition["symbol"] != assignment["symbol"]
                or definition["symbol"]
                not in _valid_definition_symbols(definition["surface"])
            ):
                link_scope_violations.append(
                    {"item_key": row["item_key"], "link": link}
                )
    return {
        "n_links_replayed": links,
        "n_observed_link_violations": len(link_scope_violations),
        "link_violations": link_scope_violations,
        "n_ghost_definition_nodes": len(ghost_nodes),
        "ghost_definition_nodes": ghost_nodes,
        "program_still_quarantined_by_adversarial_probes": True,
    }


def _markush_receipt_validation(execution: Mapping, items: Mapping[str, str]) -> dict:
    total = 0
    safe = 0
    unsafe = []
    safe_items = set()
    unsafe_items = set()
    for row in execution["rows"]:
        if not row.get("result"):
            continue
        ctext = items[row["item_key"]]
        claims = parse_claims(ctext)[1]
        nodes = row["result"]["graphs"]["markush"]["nodes"]
        by_key = {(node["claim"], node["span"][0]): node for node in nodes}
        for claim_index, claim in enumerate(claims):
            for match in _MARKUSH_RE.finditer(claim.text):
                node = by_key.get((claim.number, match.start()))
                if node is None:
                    continue
                total += 1
                at_cap = len(ctext) == 4000
                final_claim = claim_index == len(claims) - 1
                has_closing_delimiter = re.search(r"[;.]", claim.text[match.end() :]) is not None
                truncated_tail = at_cap and final_claim and not has_closing_delimiter
                if truncated_tail:
                    unsafe_items.add(row["item_key"])
                    unsafe.append(
                        {
                            "item_key": row["item_key"],
                            "claim": claim.number,
                            "span": node["span"],
                            "reason": "4k_capped_final_claim_has_no_closing_delimiter",
                        }
                    )
                else:
                    safe += 1
                    safe_items.add(row["item_key"])
    return {
        "n_original_certificates": total,
        "n_retained_certificates": safe,
        "n_excluded_truncated_certificates": len(unsafe),
        "n_items_with_retained_certificates": len(safe_items),
        "n_items_with_excluded_certificates": len(unsafe_items),
        "excluded": unsafe,
    }


def adversarial_probes() -> dict:
    numeric = analyze_patent_claim_graph(
        "CLAIMS\n1. A system wherein a temperature is measured using a probe.\n"
        "2. An unrelated composition having a temperature of at least 5 degrees."
    )["graphs"]["numeric_constraint_definition"]
    formula = analyze_patent_claim_graph(
        "CLAIMS\n1. A system wherein n and y are independently an integer.\n"
        "2. The system of claim 1, wherein a=1.\n"
        "3. The system of claim 1, wherein x_1=0 and x_2=0.0."
    )["graphs"]["formula_variable_definition"]
    duplicate = analyze_patent_claim_graph(
        "CLAIMS\n1. A sensor.\n"
        "1. A method using the sensor."
    )["graphs"]["term_reference"]
    markush = analyze_patent_claim_graph(
        "CLAIMS\n1. A composition selected from the group consisting of alpha, beta and gam"
    )["graphs"]["markush"]

    numeric_cross_claim = any(
        link["constraint_id"].startswith("c2:")
        and link["definition_id"].startswith("c1:")
        for link in numeric["links"]
    )
    ghost_symbols = sorted(
        {
            node["symbol"]
            for node in formula["definition_nodes"]
            if node["symbol"] not in {"n", "y"}
        }
    )
    duplicate_cross_instance = any(
        edge["reference_id"].startswith("p2:c1:")
        and any(candidate.startswith("p1:c1:") for candidate in edge["candidate_introduction_ids"])
        for edge in duplicate["edges"]
    )
    truncated_markush_emitted = bool(markush["nodes"])
    return {
        "numeric_unrelated_claim_link": {
            "defect_reproduced": numeric_cross_claim,
            "n_links": len(numeric["links"]),
        },
        "formula_separator_letters": {
            "defect_reproduced": bool(ghost_symbols),
            "ghost_symbols": ghost_symbols,
        },
        "formula_indexed_symbol_and_numeric_normalization": {
            "defect_reproduced": bool(formula["conflicts"]),
            "conflicts": formula["conflicts"],
        },
        "term_duplicate_ordinal_cross_instance": {
            "defect_reproduced": duplicate_cross_instance,
            "edges": duplicate["edges"],
        },
        "markush_undelimited_tail": {
            "defect_reproduced": truncated_markush_emitted,
            "nodes": markush["nodes"],
        },
    }


def _cap_certificate_stats(execution: Mapping) -> dict:
    by_relation: dict[str, dict[str, set | int]] = {}
    for row in execution["rows"]:
        if not row.get("result"):
            continue
        at_cap = row["representation"]["at_declared_character_cap"]
        for certificate in row["result"]["certificates"]:
            relation = certificate["relation"]
            entry = by_relation.setdefault(
                relation,
                {
                    "certificates": 0,
                    "certificates_at_cap": 0,
                    "items": set(),
                    "items_at_cap": set(),
                },
            )
            entry["certificates"] += 1
            entry["items"].add(row["item_key"])
            if at_cap:
                entry["certificates_at_cap"] += 1
                entry["items_at_cap"].add(row["item_key"])
    return {
        relation: {
            "n_certificates": entry["certificates"],
            "n_certificates_on_capped_items": entry["certificates_at_cap"],
            "n_items": len(entry["items"]),
            "n_capped_items": len(entry["items_at_cap"]),
        }
        for relation, entry in sorted(by_relation.items())
    }


def build_cross_audit(root: Path) -> dict:
    artifacts = {name: _load(root / path) for name, path in PATHS.items() if name != "output"}
    train = artifacts["train"]
    heldout = artifacts["heldout"]
    train_items = {
        row["item_key"]: row["ctext"]
        for row in _load(root / Path(train["sources"]["items"]["path"]))
    }
    heldout_items = {
        row["item_key"]: row["ctext"]
        for row in _load(root / Path(heldout["sources"]["items"]["path"]))
    }

    replay = {
        "train_execution_exact": _replay_execution(root, train),
        "heldout_execution_exact": _replay_execution(root, heldout),
        "freeze_exact": build_freeze(
            artifacts["construct_audit"], train, artifacts["superseded_train"]
        )
        == artifacts["freeze"],
        "operational_summary_exact": build_summary(
            artifacts["construct_audit"], artifacts["freeze"], train, heldout
        )
        == artifacts["operational"],
    }

    canonical_ids = _partial_ids(artifacts["canonical"])
    historical_ids = _partial_ids(artifacts["historical"])
    additive_ids = _partial_ids(artifacts["construct_audit"])
    mapping_rows = _mapping_rows(artifacts["construct_audit"])
    trusted_ids = {
        row["cell_id"]
        for row in mapping_rows
        if row["counts_as_current_executable_coverage"]
    }
    trusted_mapping_count = sum(
        row["counts_as_current_executable_coverage"] for row in mapping_rows
    )
    trusted_cell_depth = {}
    trusted_cell_level = {}
    for cell_id in trusted_ids:
        trusted_cell_level[cell_id] = next(
            row["level"] for row in mapping_rows if row["cell_id"] == cell_id
        )
        trusted_cell_depth[cell_id] = max(
            row["audited_depth"]
            for row in mapping_rows
            if row["cell_id"] == cell_id
            and row["counts_as_current_executable_coverage"]
        )

    markush_train = _markush_receipt_validation(train, train_items)
    markush_heldout = _markush_receipt_validation(heldout, heldout_items)
    probes = adversarial_probes()
    if not all(row["defect_reproduced"] for row in probes.values()):
        raise AssertionError("one or more expected adversarial defects did not reproduce")

    source_hashes = {
        "construct_audit": {
            "path": str(PATHS["construct_audit"]),
            "sha256": _sha256(root / PATHS["construct_audit"]),
        },
        "freeze": {
            "path": str(PATHS["freeze"]),
            "sha256": _sha256(root / PATHS["freeze"]),
        },
        "operational": {
            "path": str(PATHS["operational"]),
            "sha256": _sha256(root / PATHS["operational"]),
        },
    }
    all_bound_hashes = {
        "train": _verify_bound_sources(root, train),
        "heldout": _verify_bound_sources(root, heldout),
    }

    return {
        "schema": SCHEMA,
        "status": "independent_additive_cross_audit_complete_with_quarantines",
        "task": "patents",
        "design": {
            "input_channel": "exact frozen ctext plus source code and provenance artifacts",
            "references_or_outcomes_loaded": False,
            "prompt_outputs_loaded": False,
            "external_supervision_used": False,
            "models_apis_or_accelerators_used": False,
            "canonical_artifacts_modified": False,
            "adjudication_rule": (
                "retain construct mapping separately from executable certification; quarantine "
                "a program family when a finite adversarial counterexample defeats its stated "
                "relation, applicability, polarity, or depth contract"
            ),
        },
        "sources": source_hashes,
        "bound_source_hash_checks": all_bound_hashes,
        "replay": replay,
        "mapping_adjudications": mapping_rows,
        "receipt_validation": {
            "train": {
                "term": _term_receipt_validation(train, train_items),
                "numeric": _numeric_receipt_validation(train, train_items),
                "formula": _formula_receipt_validation(train, train_items),
                "markush": markush_train,
                "certificate_cap_stats": _cap_certificate_stats(train),
            },
            "heldout": {
                "term": _term_receipt_validation(heldout, heldout_items),
                "numeric": _numeric_receipt_validation(heldout, heldout_items),
                "formula": _formula_receipt_validation(heldout, heldout_items),
                "markush": markush_heldout,
                "certificate_cap_stats": _cap_certificate_stats(heldout),
            },
        },
        "adversarial_probes": probes,
        "descriptive_union_check": {
            "canonical_cells": len(canonical_ids),
            "historical_cells": len(historical_ids),
            "original_additive_cells": len(additive_ids),
            "canonical_historical_overlap": len(canonical_ids & historical_ids),
            "canonical_additive_overlap": len(canonical_ids & additive_ids),
            "historical_additive_overlap": len(historical_ids & additive_ids),
            "original_three_lane_union": len(canonical_ids | historical_ids | additive_ids),
            "trusted_current_additive_cells": len(trusted_ids),
            "trusted_three_lane_union": len(canonical_ids | historical_ids | trusted_ids),
            "interpretation": (
                "descriptive, provenance-separated cell coverage only; neither number is a "
                "codability, articulability, reconstruction, or isomorphism estimate"
            ),
        },
        "summary": {
            "n_construct_mappings_retained_as_relation_local": len(mapping_rows),
            "n_original_additive_cells": len(additive_ids),
            "n_original_additive_mappings": len(mapping_rows),
            "n_current_executable_cells_after_cross_audit": len(trusted_ids),
            "n_current_executable_mappings_after_cross_audit": trusted_mapping_count,
            "n_quarantined_cells": len(additive_ids - trusted_ids),
            "n_quarantined_mappings": len(mapping_rows) - trusted_mapping_count,
            "quarantined_relations": sorted(QUARANTINED_RELATIONS),
            "current_executable_cells_by_level": dict(
                sorted(
                    Counter(trusted_cell_level.values()).items()
                )
            ),
            "current_executable_cell_depth_counts": dict(
                sorted(Counter(str(depth) for depth in trusted_cell_depth.values()).items())
            ),
            "heldout_markush_original_certificates": markush_heldout[
                "n_original_certificates"
            ],
            "heldout_markush_retained_certificates": markush_heldout[
                "n_retained_certificates"
            ],
            "heldout_markush_retained_items": markush_heldout[
                "n_items_with_retained_certificates"
            ],
            "all_execution_source_hashes_match": all(
                check["matches"]
                for phase in all_bound_hashes.values()
                for check in phase.values()
            ),
            "all_pipeline_replays_exact": all(replay.values()),
        },
        "provenance_correction": {
            "execution_receipts_hash_bound": True,
            "construct_audit_hash_bound_by_freeze": False,
            "freeze_hash_bound_by_operational_summary": False,
            "finding": (
                "train and heldout receipts strongly bind items, manifest, program, and runner. "
                "Freeze and operational validation compare schemas/content relations but do not "
                "record SHA-256 identities for the construct audit and freeze themselves; this "
                "cross-audit records those hashes without rewriting canonical artifacts"
            ),
        },
        "claim_limits": {
            "whole_construct_cells": 0,
            "codability_claim_permitted": False,
            "prompt_articulability_measured": False,
            "reference_reconstruction_measured": False,
            "isomorphism_measured": False,
            "negative_result_establishes_tacitness": False,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    artifact = build_cross_audit(root)
    output = root / PATHS["output"]
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **artifact["summary"]}, indent=2))


if __name__ == "__main__":
    main()
