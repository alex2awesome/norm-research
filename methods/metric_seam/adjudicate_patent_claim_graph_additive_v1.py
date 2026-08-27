"""Cell-level construct audit for the additive patent claim graphs.

This adjudication is source-only.  It reads panel text and frozen provenance
ledgers, never item text, program outputs, references, outcomes, prompts, prior
art, or examiner evidence.  Accepted rows are narrow subrelations, not whole
criteria.  Every other panel cell records bounded non-discovery within this
six-relation program class.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.patent_claim_graph_additive_v1 import (
    RELATIONS,
    SCHEMA as PROGRAM_SCHEMA,
)


SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-fidelity.v1"
PANEL_SCHEMA = "tacit_breadth_metric_panel/v1"
CANONICAL_SCHEMA = "metric-seam.hierarchy-patent-claim-structure-fidelity.v1"
HISTORICAL_SCHEMA = "metric-seam.hierarchy-patent-construct-fidelity.v1"
TASK = "patents"


def _match(
    relation_id: str,
    requested_subrelation: str,
    partial_scope: str,
    exclusions: Sequence[str],
    certificate_policy: str,
) -> dict:
    return {
        "relation_id": relation_id,
        "requested_subrelation": requested_subrelation,
        "partial_scope": partial_scope,
        "exclusions": list(exclusions),
        "certificate_policy": certificate_policy,
    }


# Exact ids make panel/rank drift fail closed.  These are the seven-cell
# high-confidence audit envelope plus the separately requested repaired
# Markush relation.  Multiple relations in a cell do not create extra cells.
_ACCEPTED = {
    "TB::patents::specific::R1::parented_tree::252::f491e1d963d7235b9f55": [
        _match(
            "numeric_constraint_definition_graph",
            "numerical ranges linked to explicit measurement/definition language",
            "finite presented numeric constraint nodes and positive in-ctext definition links",
            [
                "support, enablement, divergent-technique convergence, accepted-method status, and definiteness",
                "definitions or disclosure outside the 4,000-character ctext prefix",
                "an inference that an unlinked constraint lacks a definition",
            ],
            "positive links only; unlinked nodes are applicability witnesses, not defects",
        )
    ],
    "TB::patents::specific::R1::merged_tree::254::1e6c67e300daccfa0331": [
        _match(
            "formula_variable_definition_alignment",
            "formula symbols remain definition-linked and incorporated equalities are not contradictory",
            "single-symbol numeric assignments linked to definitions plus finite incompatible-equality counter-witnesses along an explicit dependency path",
            [
                "general technical sense, prose contradiction, dimensional analysis, and formula correctness",
                "implicit definitions, algebra beyond scalar equality, and missing material outside ctext",
            ],
            "positive definition links and exact numeric-equality conflicts only",
        )
    ],
    "TB::patents::specific::R2::grandparent::19::80d1e59043d4c57aac24": [
        _match(
            "two_part_or_jepson_structure",
            "two-part/Jepson claim format",
            "finite independent-claim boundary witness for a Jepson improvement or EPC characterising portion",
            [
                "whether the form is appropriate in view of prior art",
                "what is actually admitted prior art or novel",
                "section 112(f) implications and drafting quality",
            ],
            "positive structural boundary witnesses only",
        )
    ],
    "TB::patents::specific::R2::merged_group::89::8b609440fc3c85acf8c3": [
        _match(
            "bounded_antecedent_term_reference_graph",
            "antecedent-basis and precise-reference hygiene",
            "bounded article-led term introductions and references resolved across explicit claim ancestors",
            [
                "whole terminology consistency, coined-term clarity, public notice, or legal definiteness",
                "semantic coreference, implicit antecedents, specification-wide vocabulary, and verified absence",
            ],
            "finite resolved/ambiguous/unresolved nodes under the declared grammar only",
        ),
        _match(
            "formula_variable_definition_alignment",
            "formula symbols are explicitly defined and equality constraints are internally consistent",
            "single-symbol definition links and finite incompatible-equality counter-witnesses",
            [
                "all prose terminology, formula correctness, meaning equivalence, and legal definiteness",
                "symbols or definitions outside the presented prefix",
            ],
            "positive definition links and exact numeric-equality conflicts only",
        ),
    ],
    "TB::patents::specific::R2::grandparent::20::d5241fc9bf0f24e2d9fc": [
        _match(
            "markush_closed_group_structure",
            "Markush closed-group syntax and explicit mixtures/combinations qualifier",
            "exact closed-group opener followed by a finite presented alternative list, with any explicit mixture/combination qualifier separately witnessed",
            [
                "Markush scope construction, unity, disclosure, enablement, and legal validity",
                "product-by-process, apparatus, composition, and structural-characterization doctrines",
                "whether mixtures are implicitly encompassed when no explicit qualifier appears",
            ],
            "positive exact-syntax/list witnesses only; generic 'selected from' does not apply",
        )
    ],
    "TB::patents::specific::R3::merged_group::14::53724d3a12eba0780d15": [
        _match(
            "claim_status_and_local_listing_witnesses",
            "claim-list status marking and local ordinal integrity",
            "recognized status parentheticals and duplicate presented-ordinal counter-witnesses",
            [
                "amendment timing/admissibility, new matter, broadening, proper complete listing, and lifecycle procedure",
                "claim material after the presented prefix and jurisdiction-specific status requirements",
            ],
            "finite recognized markers and local duplicate ordinals only; no list-completeness claim",
        )
    ],
    "TB::patents::specific::R3::grandparent::1::4eef90e0e376a03bdf9b": [
        _match(
            "bounded_antecedent_term_reference_graph",
            "proper antecedent basis and article usage",
            "bounded a/an introduction to the/said/such reference graph across explicit ancestors",
            [
                "reasonable certainty, general clarity, support, essential features, and legal indefiniteness",
                "implicit/semantic coreference and ambiguity outside the bounded grammar",
            ],
            "finite graph resolutions/counter-witnesses only; no whole-claim pass/fail",
        ),
        _match(
            "numeric_constraint_definition_graph",
            "objective numeric boundaries linked to explicit measurement/definition language",
            "finite numeric constraint nodes and positive definition/measurement links",
            [
                "subjective terms generally, accepted-method status, convergence, reproducibility, and clarity",
                "absence of definitions outside the presented prefix",
            ],
            "positive links only; unlinked applicability is not a defect certificate",
        ),
        _match(
            "formula_variable_definition_alignment",
            "formula-variable definitions and equality consistency",
            "single-symbol definition links and exact incorporated equality conflicts",
            [
                "formula correctness, dimensional consistency, broader terminology, and legal definiteness",
                "unparsed math or definitions outside ctext",
            ],
            "positive definition links and exact numeric-equality conflicts only",
        ),
    ],
    "TB::patents::specific::R3::merged_group::2::dc0365c77ceff8c35701": [
        _match(
            "numeric_constraint_definition_graph",
            "parameter/range nodes linked to explicit measurement or definition language",
            "finite presented numeric constraints and positive in-ctext measurement/definition links",
            [
                "accepted-method status, common general knowledge, method convergence, guidance, effects, and reproducibility",
                "trade-name ambiguity, support/enablement, and legal clarity/definiteness",
                "an inference from missing links in the capped representation",
            ],
            "positive links only; constraint incidence alone is applicability, not quality",
        )
    ],
}


def _validate_inputs(panel: Mapping, canonical: Mapping, historical: Mapping) -> list[Mapping]:
    if panel.get("schema") != PANEL_SCHEMA:
        raise ValueError("unexpected panel schema")
    if canonical.get("schema") != CANONICAL_SCHEMA or canonical.get("task") != TASK:
        raise ValueError("unexpected canonical patent fidelity artifact")
    if historical.get("schema") != HISTORICAL_SCHEMA or historical.get("task") != TASK:
        raise ValueError("unexpected historical patent fidelity artifact")
    cells = [row for row in panel.get("cells", []) if row.get("task") == TASK]
    if len(cells) != 90 or len({row.get("id") for row in cells}) != 90:
        raise ValueError("patent panel must contain exactly 90 unique cells")
    if set(_ACCEPTED) - {row["id"] for row in cells}:
        raise ValueError("accepted cell id is absent from the frozen panel")
    relation_ids = {row["relation_id"] for row in RELATIONS}
    used = {
        match["relation_id"] for matches in _ACCEPTED.values() for match in matches
    }
    if used != relation_ids:
        raise ValueError("accepted mapping and additive relation catalog drifted")
    return cells


def _partial_ids(artifact: Mapping) -> set[str]:
    return {
        row["cell_id"]
        for row in artifact["rows"]
        if row["verdict"] == "partial_relation_local"
    }


def build_audit(panel: Mapping, canonical: Mapping, historical: Mapping) -> dict:
    cells = _validate_inputs(panel, canonical, historical)
    relation_by_id = {row["relation_id"]: row for row in RELATIONS}
    rows = []
    for cell in cells:
        matches = []
        for match in _ACCEPTED.get(cell["id"], []):
            relation = relation_by_id[match["relation_id"]]
            matches.append({**match, "channel": relation["channel"], "depth": relation["depth"]})
        rows.append(
            {
                "cell_id": cell["id"],
                "task": TASK,
                "level": cell["level"],
                "selection_rank": cell["selection_rank"],
                "construct": cell["construct"],
                "description": cell["description"],
                "verdict": "partial_relation_local" if matches else "bounded_non_discovery",
                "matched_relations": matches,
                "eligible_relation_local_depths": sorted({row["depth"] for row in matches}),
                "maximum_matching_relation_depth": max(
                    (row["depth"] for row in matches), default=None
                ),
                "exact_whole_construct_fidelity": False,
                "bounded_non_discovery_reason": (
                    None
                    if matches
                    else (
                        "no direct requested subrelation survived the cell-level source audit "
                        "within the frozen six-relation claim-graph class; lexical or applicability "
                        "overlap was not credited"
                    )
                ),
            }
        )

    accepted_rows = [row for row in rows if row["verdict"] == "partial_relation_local"]
    depths = Counter(str(row["maximum_matching_relation_depth"]) for row in accepted_rows)
    relation_counts = Counter(
        relation["relation_id"]
        for row in accepted_rows
        for relation in row["matched_relations"]
    )
    by_level = {}
    for level in ("R1", "R2", "R3"):
        level_rows = [row for row in rows if row["level"] == level]
        by_level[level] = {
            "n_cells": len(level_rows),
            "n_partial_relation_local": sum(
                row["verdict"] == "partial_relation_local" for row in level_rows
            ),
            "n_bounded_non_discovery": sum(
                row["verdict"] == "bounded_non_discovery" for row in level_rows
            ),
        }

    additive_ids = set(_ACCEPTED)
    canonical_ids = _partial_ids(canonical)
    historical_ids = _partial_ids(historical)
    return {
        "schema": SCHEMA,
        "status": "source_audited_additive_static_partial_relations",
        "task": TASK,
        "source_panel_content_sha256": panel.get("panel_content_sha256"),
        "program_schema": PROGRAM_SCHEMA,
        "program_relation_catalog": list(RELATIONS),
        "audit_design": {
            "item_text_loaded": False,
            "program_outputs_loaded": False,
            "outcome_or_reference_values_loaded": False,
            "prompt_outputs_loaded": False,
            "prior_art_or_examiner_evidence_loaded": False,
            "external_supervision_used": False,
            "fidelity_rule": (
                "accept only a directly requested relation, preserve exclusions and finite "
                "certificate policy, and never promote applicability to whole-criterion fidelity"
            ),
            "negative_result_policy": (
                "bounded non-discovery applies only to this frozen relation catalog, panel, "
                "representation, and audit budget; it is not evidence of tacitness"
            ),
        },
        "rows": rows,
        "summary": {
            "verdict_counts": dict(Counter(row["verdict"] for row in rows)),
            "by_level": by_level,
            "n_partial_relation_local_cells": len(accepted_rows),
            "n_relation_mappings": sum(len(row["matched_relations"]) for row in accepted_rows),
            "n_exact_whole_construct_cells": 0,
            "maximum_matching_relation_depth_counts": dict(sorted(depths.items())),
            "relation_cell_counts": dict(sorted(relation_counts.items())),
            "provenance_separate_descriptive_union": {
                "canonical_pure_code_cells": len(canonical_ids),
                "historical_oracle_hybrid_cells": len(historical_ids),
                "additive_claim_graph_cells": len(additive_ids),
                "additive_overlap_with_canonical": len(additive_ids & canonical_ids),
                "additive_overlap_with_historical": len(additive_ids & historical_ids),
                "three_lane_union_cells": len(
                    additive_ids | canonical_ids | historical_ids
                ),
                "interpretation": (
                    "descriptive coverage union only; provenance and channels remain separate "
                    "and this is not a codability, reconstruction, or isomorphism estimate"
                ),
            },
        },
        "claim_limits": {
            "whole_patent_score_emitted": False,
            "codability_claim_permitted": False,
            "prompt_articulability_measured": False,
            "reference_reconstruction_measured": False,
            "isomorphism_measured": False,
            "verified_absence_from_full_patent_permitted": False,
        },
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "metric_seam_pilot" / "hierarchy_r123"
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=out / "panel_v3.json")
    parser.add_argument(
        "--canonical",
        type=Path,
        default=out / "patents_claim_structure_construct_fidelity_v1.json",
    )
    parser.add_argument(
        "--historical",
        type=Path,
        default=out / "patents_construct_fidelity_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=out / "patents_claim_graph_additive_construct_fidelity_v1.json",
    )
    args = parser.parse_args()
    artifact = build_audit(_load(args.panel), _load(args.canonical), _load(args.historical))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
