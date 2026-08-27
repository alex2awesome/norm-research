"""Adjudicate the pure-code patent claim-structure seed against panel_v3.

This is a source-and-contract audit, not a fit audit.  It reads the frozen
panel, the executable relation catalogue, and (optionally) the compiler-train
execution receipt.  It must not read held-out items, reference judgements,
prompt outputs, patent outcomes, prior art, or examiner records.

The headline mapping is deliberately conservative.  A relation is accepted
only when the panel criterion explicitly contains the implemented relation;
mere applicability detection is retained as a named sensitivity near-miss.
Train execution establishes only operational applicability on presented bytes.
It cannot repair a static construct mismatch.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.patent_claim_structure import (
    RELATIONS,
    SCHEMA as PROGRAM_SCHEMA,
)


SCHEMA = "metric-seam.hierarchy-patent-claim-structure-fidelity.v1"
PANEL_SCHEMA = "tacit_breadth_metric_panel/v1"
TRAIN_SCHEMA = "metric-seam.hierarchy-patent-claim-structure-execution.v3"
HISTORICAL_SCHEMA = "metric-seam.hierarchy-patent-construct-fidelity.v1"
TASK = "patents"


def _match(
    relation_id: str,
    partial_scope: str,
    exclusions: Sequence[str],
    *,
    certificate_policy: str,
) -> dict:
    return {
        "relation_id": relation_id,
        "partial_scope": partial_scope,
        "exclusions": list(exclusions),
        "certificate_policy": certificate_policy,
    }


# Exact cell ids prevent a rank or name drift from silently changing the audit.
_ACCEPTED = {
    "TB::patents::specific::R1::merged_tree::151::a6737bddab8d451d7ae9": [
        _match(
            "application_section_presence",
            "named ABSTRACT and CLAIMS section presence only",
            [
                "request, description, drawings, oath/declaration, fees, and jurisdictional variants",
                "document sufficiency, order, form, and filing-date entitlement",
            ],
            certificate_policy="static relation match only; train value is formatter-constant and non-operational",
        )
    ],
    "TB::patents::specific::R2::grandparent::10::41a099074657b4acc7f5": [
        _match(
            "functional_limitation_incidence",
            "positive recognition of a bounded set of functional-language markers",
            [
                "whether section 112(f) is legally invoked",
                "corresponding structure or algorithm, definiteness, breadth, and drafting quality",
                "verified absence of functional language",
            ],
            certificate_policy="positive finite marker witnesses only",
        )
    ],
    "TB::patents::specific::R2::merged_group::40::bb89d6d56dcc9ea9c238": [
        _match(
            "application_section_presence",
            "named ABSTRACT and CLAIMS section presence only",
            [
                "all other required application parts",
                "section order, jurisdiction-specific form, content adequacy, and fees",
            ],
            certificate_policy="static relation match only; train value is formatter-constant and non-operational",
        )
    ],
    "TB::patents::specific::R3::grandparent::0::ed76386d4408681be502": [
        _match(
            "functional_limitation_incidence",
            "positive recognition of a bounded set of possible functional limitations",
            [
                "section 112(f) invocation and claim construction",
                "linked structure or algorithms, compliance, avoidance, vagueness, and result-only scope",
                "verified absence of functional language",
            ],
            certificate_policy="positive finite marker witnesses only",
        )
    ],
    "TB::patents::specific::R3::merged_group::12::4a62e79af29087e6ff96": [
        _match(
            "application_section_presence",
            "named ABSTRACT and CLAIMS section presence only",
            [
                "request, title, description sufficiency, drawings, oath/declaration, fees, and language",
                "proper order/form, filing-date entitlement, and examination entry",
            ],
            certificate_policy="static relation match only; train value is formatter-constant and non-operational",
        )
    ],
    "TB::patents::specific::R3::merged_group::3::6d907639386384acc1da": [
        _match(
            "abstract_word_count",
            "replayable word count of the named presented abstract",
            [
                "clarity, technical representativeness, identifiers, tone, and prohibited language",
                "a universal office-independent compliance threshold",
            ],
            certificate_policy="exact presented-section count; interpret thresholds under a separately frozen office rule",
        )
    ],
    "TB::patents::specific::R3::grandparent::14::b26fd00c6c47f2854678": [
        _match(
            "claim_dependency_well_formedness",
            "finite explicit dependency edges resolve to earlier-presented lower-numbered claims",
            [
                "global claim-set completeness under the character cap",
                "support, breadth, design-around resistance, single-sentence form, and divided infringement",
            ],
            certificate_policy="finite positive edges and local counter-witnesses only; no global absence/compliance claim",
        ),
        _match(
            "claim_set_layering",
            "positive existence of a parsed independent root plus a valid dependent fallback edge",
            [
                "strategic quality or adequacy of fallback positions",
                "verified absence of layering in a capped or incomplete source claim set",
            ],
            certificate_policy="positive finite existence witness only",
        ),
    ],
    "TB::patents::specific::R3::merged_group::7::ac30b4e148a5c6a11ec7": [
        _match(
            "statutory_category_surface_coverage",
            "positive independent-claim preamble witness for process, machine/apparatus, manufacture/article, or composition",
            [
                "legal eligibility and governing-definition analysis",
                "design and plant categories and exhaustive category coverage",
                "verified absence of a statutory category",
            ],
            certificate_policy="positive surface-and-span certificates only; do not use zero/coverage as absence",
        )
    ],
}


# These are plausible broader mappings, preserved so the conservative headline
# is auditable rather than silently discarding sensitivity choices.
_SENSITIVITY_NEAR_MISSES = {
    "TB::patents::specific::R1::parented_tree::252::f491e1d963d7235b9f55": {
        "relation_id": "numerical_limitation_incidence",
        "reason": (
            "numeric/range incidence identifies an applicability object but does not assess "
            "measurement clarity, support, divergent techniques, or definiteness; more incidence "
            "has no aligned quality direction"
        ),
    },
    "TB::patents::specific::R1::merged_tree::254::1e6c67e300daccfa0331": {
        "relation_id": "claim_dependency_well_formedness",
        "reason": (
            "syntactic dependency resolution is not internal technical consistency or technical "
            "sense, and therefore does not preserve this criterion's evaluative relation"
        ),
    },
    "TB::patents::specific::R2::grandparent::20::d5241fc9bf0f24e2d9fc": {
        "relation_id": "statutory_category_surface_coverage",
        "reason": (
            "generic statutory-category recognition does not implement the doctrine-specific "
            "product-by-process, Markush, mixture, or structural-characterization issues"
        ),
    },
    "TB::patents::specific::R3::merged_group::2::dc0365c77ceff8c35701": {
        "relation_id": "numerical_limitation_incidence",
        "reason": (
            "numeric/range incidence does not verify definitions, accepted measurement methods, "
            "convergence, reproducibility, effects, or clarity; it is an applicability gate only"
        ),
    },
}


_REJECTION_REASON_BY_SLOT = {
    # R1
    ("R1", 0): "dependency topology does not test a single general inventive concept or special technical relation",
    ("R1", 1): "no POSITA, make/use teaching, or undue-experimentation analysis",
    ("R1", 2): "no drawing defect, correction procedure, or submission-format analysis",
    ("R1", 3): "no terminal disclaimer, ownership, patent term, or prosecution-history input",
    ("R1", 4): "no design ornamentality, article relationship, originality, novelty, or registrability analysis",
    ("R1", 5): "generic claim topology does not encode restriction, election, species, or reissue rules",
    ("R1", 6): "no inventor-state, best-mode, or disclosure-sufficiency analysis",
    ("R1", 7): "no prior art, enablement, public disclosure, use, or filing-date comparison",
    ("R1", 8): "no novelty, inventive-step, or industrial-applicability determination",
    ("R1", 9): "category or functional surfaces do not test a judicial exception or practical application",
    ("R1", 10): "no accused product/process, limitation segmentation, or element-by-element mapping",
    ("R1", 11): "no empirical results, validation method, or replication evidence analysis",
    ("R1", 12): "dependency syntax does not assess reasonable-certainty claim scope for a POSITA",
    ("R1", 13): "no claim-to-description semantic support or undue-generalization analysis",
    ("R1", 15): "no specification architecture, alternatives, mechanisms, technical effects, or completeness analysis",
    ("R1", 16): "no biological deposit, depository, sequence-listing, or plant-disclosure analysis",
    ("R1", 18): "no semantic claim-clarity analysis under the named legal standard",
    ("R1", 19): "no drawings or necessity/completeness/new-matter analysis",
    ("R1", 20): "generic formatter headings are not prescribed EPC forms or a formalities examination; train presence is constant",
    ("R1", 21): "no full-scope written-description or enablement comparison between claims and specification",
    ("R1", 22): "no scope/public-notice analysis; the antecedent heuristic is diagnostic-only and excluded",
    ("R1", 23): "no language-of-proceedings, nationality/residence, treaty-reference, or national-form analysis",
    ("R1", 24): "functional/numeric incidence only flags applicability and does not test concrete support or possession",
    ("R1", 25): "no biological-material description, deposit timing, depository, or deposit-information analysis",
    ("R1", 27): "no original-disclosure baseline, amendment history, support, responsiveness, or scope change",
    ("R1", 28): "no Hague procedure, reproductions, US-specific design rules, or restriction analysis",
    ("R1", 29): "no affidavit, signature, response sufficiency, revival, or delay-statement analysis",
    # R2
    ("R2", 0): "no technological-improvement, implementation-specificity, mental-step, or generic-computer analysis",
    ("R2", 1): "root/dependent layering does not encode restriction, species, linking, rejoinder, election, or divisionals",
    ("R2", 2): "no two-part/Jepson parser, prior-art partition, improvement identification, or side-effect analysis",
    ("R2", 3): "no terminology identity/definition/ambiguity analysis; the antecedent heuristic is diagnostic-only",
    ("R2", 4): "no prior-art, public-disclosure, sale/use, temporal, or design-novelty evidence",
    ("R2", 5): "no jurisdiction, language, nationality/residence, or treaty-reference analysis",
    ("R2", 6): "no design images, views, broken lines, shading, or scope-signaling analysis",
    ("R2", 7): "no figures, shading, orientation, reference numerals, sheet numbering, or perspective analysis",
    ("R2", 8): "generic statutory categories do not distinguish pharmaceutical claim forms or combination strategies",
    ("R2", 9): "no ST.25/ST.26 sequence validation, deposit, viability, deadline, or new-matter analysis",
    ("R2", 10): "no model, algorithm, training-data, reproducibility, technical-effect, or full-scope disclosure analysis",
    ("R2", 11): "no CPC G16H semantic classification or scope distinction",
    ("R2", 13): "formatter headings do not test cross-jurisdiction forms, signatures, language, or formal examination",
    ("R2", 14): "no medical/diagnostic subject identification, jurisdictional law, eligibility, or claim strategy",
    ("R2", 16): "no earlier-filing references, priority chain, timing, entitlement, or Article 76 analysis",
    ("R2", 17): "the program analyzes patent claims, not trademark signatures, specimens, use, or section 2(d)",
    ("R2", 18): "no PTAB petition, grounds, record citations, expert support, or evidence mapping",
    ("R2", 19): "no wood/plant treatment purpose, material, process, apparatus, or classification analysis",
    ("R2", 21): "no claim-to-specification possession analysis across claimed breadth",
    ("R2", 22): "no background, prior-approach, technical-problem, or achievement/advantage analysis",
    ("R2", 23): "surface system/category markers do not determine conventionality or eligibility",
    ("R2", 24): "no deposits, sequences, plant particulars, enablement, or written-description analysis",
    ("R2", 25): "no design classifier, ornamental-scope analysis, drawing analysis, or single-design-claim relation",
    ("R2", 26): "no geographical-indication definition, geography, product, or protection-regime analysis",
    ("R2", 27): "functional/numeric incidence only flags applicability and does not test possession or enablement",
    ("R2", 28): "claims are optional in provisionals, so generic section presence has the wrong applicability and direction",
    ("R2", 29): "no rights, prohibited acts, geography, publication timing, or related-application analysis",
    # R3
    ("R3", 1): "no natural-product, diagnostic/treatment, marked-difference, or practical-application analysis",
    ("R3", 2): "no ST.26 XML structure, symbol, identifier, validation, or cross-reference parser",
    ("R3", 3): "no semantic possession, representative-species, structural-feature, or blaze-mark analysis",
    ("R3", 4): "no priority references, chain, timing, copendency, correction, or earlier-date support analysis",
    ("R3", 5): "no terminal disclaimer, double patenting, ownership, term, recordation, or prosecution input",
    ("R3", 6): "no claim-to-specification possession or enablement analysis across scope",
    ("R3", 7): "no description text comparison, contradiction, embodiment, allowance, or added-matter analysis",
    ("R3", 8): "no claim chart, prior art, accused product, pinpoint citation, KSR, or admission analysis",
    ("R3", 9): "no design images, view consistency, broken lines, shading, or formal drawing analysis",
    ("R3", 10): "no judicial-exception, practical-application, conventionality, inventive-concept, or preemption analysis",
    ("R3", 11): "ordinal contiguity is not amendment compliance: valid claim listings retain canceled numbers and gaps",
    ("R3", 12): "no objective-boundary or semantic clarity analysis; antecedent output is diagnostic-only and excluded",
    ("R3", 13): "no operability, utility, credibility, substantiality, or industrial-application analysis",
    ("R3", 14): "no drawings, reference-character consistency, lead-line, caption, or legibility analysis",
    ("R3", 18): "no design ornamentality, article relationship, images, views, or Hague/formality analysis",
    ("R3", 20): "no double-patenting comparison, terminal disclaimer, ownership, term, or enforceability analysis",
    ("R3", 21): "dependency topology does not establish semantic unity or restriction/election compliance",
    ("R3", 22): "no PTAB petition, challenged claim, ground, evidence map, prior-art status, or institution analysis",
    ("R3", 23): "no original-disclosure baseline, amendment history, support comparison, or filing-date analysis",
    ("R3", 24): "no deposit, sequence listing, viability, timing, availability, or biotech-sufficiency analysis",
    ("R3", 25): "no intrinsic record, BRI/Phillips reasoning, lexicography, disclaimer, or construction impact",
    ("R3", 26): "no prior art, anticipation mapping, temporal event, grace-period, or exception analysis",
    ("R3", 28): "generic headings/categories do not establish a design application, single claim, drawings, or unity",
    ("R3", 29): "claim topology does not encode unity, restriction propriety, election responses, or divisional strategy",
}


def _relation_catalog() -> dict[str, dict]:
    return {str(row["relation_id"]): dict(row) for row in RELATIONS}


def _validate_train_receipt(receipt: Mapping) -> dict[str, dict]:
    if receipt.get("schema") != TRAIN_SCHEMA:
        raise ValueError(f"expected train schema {TRAIN_SCHEMA}")
    if receipt.get("program_schema") != PROGRAM_SCHEMA:
        raise ValueError("train receipt does not bind to the current program schema")
    if receipt.get("phase") != "compiler_train":
        raise ValueError("only compiler_train receipts are allowed")
    design = receipt.get("design", {})
    required_false = (
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_used",
        "whole_patent_score_emitted",
        "absence_certificate_permitted",
    )
    if any(design.get(key) is not False for key in required_false):
        raise ValueError("train receipt violates the blind, non-aggregated contract")
    summary = receipt.get("summary", {})
    if summary.get("n_items") != 150 or summary.get("failure_types"):
        raise ValueError("unexpected compiler-train execution population or failures")
    measurement = summary.get("relation_measurement", {})
    if set(measurement) != set(_relation_catalog()):
        raise ValueError("train relation catalogue does not match the program")
    return {str(key): dict(value) for key, value in measurement.items()}


def _train_applicability(
    relation_id: str,
    measurement: Mapping[str, Mapping] | None,
) -> dict | None:
    if measurement is None:
        return None
    observed = dict(measurement[relation_id])
    if relation_id == "application_section_presence":
        classification = "measured_but_constant_non_operational"
        allowed = "static construct fidelity only"
    elif relation_id == "statutory_category_surface_coverage":
        classification = "nonconstant_positive_certificate_channel"
        allowed = "positive surface-and-span certificates only"
    elif relation_id in {
        "functional_limitation_incidence",
        "claim_set_layering",
    }:
        classification = "nonconstant_positive_finite_witness_channel"
        allowed = "positive finite witnesses only"
    elif relation_id == "claim_dependency_well_formedness":
        classification = "nonconstant_finite_edge_witness_channel"
        allowed = "finite positive edges and local counter-witnesses only"
    elif relation_id == "abstract_word_count":
        classification = "nonconstant_presented_section_measurement"
        allowed = "exact word count on the named presented abstract"
    else:
        classification = "not_headline_eligible"
        allowed = "no headline construct-fidelity credit"
    return {
        **observed,
        "classification": classification,
        "allowed_interpretation": allowed,
        "absence_or_whole_source_inference_permitted": False,
    }


def _weighted_sensitivity(
    panel_rows: Sequence[Mapping],
    included_ids: set[str],
) -> dict:
    total = sum(float(row["design_weight"]) for row in panel_rows)
    included = sum(
        float(row["design_weight"])
        for row in panel_rows
        if str(row["id"]) in included_ids
    )
    return {
        "n_selected_cells": len(included_ids),
        "design_weight_numerator": included,
        "design_weight_denominator": total,
        "weighted_fraction": included / total,
        "by_level": {
            level: {
                "n_selected_cells": sum(
                    str(row["id"]) in included_ids and row["level"] == level
                    for row in panel_rows
                ),
                "weighted_fraction": (
                    sum(
                        float(row["design_weight"])
                        for row in panel_rows
                        if row["level"] == level and str(row["id"]) in included_ids
                    )
                    / sum(
                        float(row["design_weight"])
                        for row in panel_rows
                        if row["level"] == level
                    )
                ),
            }
            for level in ("R1", "R2", "R3")
        },
    }


def build_audit(
    panel: Mapping,
    train_receipt: Mapping | None = None,
    historical_audit: Mapping | None = None,
) -> dict:
    if panel.get("schema") != PANEL_SCHEMA or panel.get("n_cells") != 990:
        raise ValueError("expected the frozen 990-cell panel_v3")
    panel_rows = [row for row in panel.get("cells", []) if row.get("task") == TASK]
    if len(panel_rows) != 90 or len({row["id"] for row in panel_rows}) != 90:
        raise ValueError("expected 90 unique patent cells")
    panel_ids = {str(row["id"]) for row in panel_rows}
    named_ids = set(_ACCEPTED) | set(_SENSITIVITY_NEAR_MISSES)
    if not named_ids <= panel_ids:
        raise ValueError(f"stale named cell ids: {sorted(named_ids - panel_ids)}")

    measurement = (
        _validate_train_receipt(train_receipt)
        if train_receipt is not None
        else None
    )
    catalogue = _relation_catalog()
    rows = []
    for source in panel_rows:
        cell_id = str(source["id"])
        level = str(source["level"])
        rank = int(source["selection_rank"])
        if cell_id in _ACCEPTED:
            matches = []
            for frozen in _ACCEPTED[cell_id]:
                relation_id = str(frozen["relation_id"])
                if relation_id not in catalogue:
                    raise ValueError(f"unknown relation in accepted map: {relation_id}")
                contract = catalogue[relation_id]
                matches.append(
                    {
                        **frozen,
                        "implemented_relation": contract["implemented_relation"],
                        "channel": contract["channel"],
                        "effective_code_depth": int(contract["depth"]),
                        "train_operational_applicability": _train_applicability(
                            relation_id, measurement
                        ),
                    }
                )
            verdict = "partial_relation_local"
            rejection_reason = None
            sensitivity = None
        elif cell_id in _SENSITIVITY_NEAR_MISSES:
            matches = []
            verdict = "sensitivity_near_miss_not_accepted"
            sensitivity = dict(_SENSITIVITY_NEAR_MISSES[cell_id])
            relation_id = str(sensitivity["relation_id"])
            sensitivity["effective_code_depth"] = int(catalogue[relation_id]["depth"])
            sensitivity["train_operational_applicability"] = _train_applicability(
                relation_id, measurement
            )
            rejection_reason = sensitivity["reason"]
        else:
            matches = []
            verdict = "no_faithful_relation"
            sensitivity = None
            try:
                rejection_reason = _REJECTION_REASON_BY_SLOT[(level, rank)]
            except KeyError as exc:
                raise ValueError(f"missing rejection adjudication for {(level, rank)}") from exc
        depths = sorted({row["effective_code_depth"] for row in matches})
        rows.append(
            {
                "cell_id": cell_id,
                "task": TASK,
                "level": level,
                "selection_rank": rank,
                "construct": str(source["construct"]),
                "description": str(source["description"]),
                "verdict": verdict,
                "matched_relations": matches,
                "eligible_relation_local_depths": depths,
                "maximum_matching_relation_depth": max(depths) if depths else None,
                "exact_whole_construct_fidelity": False,
                "rejection_or_demotion_reason": rejection_reason,
                "sensitivity_near_miss": sensitivity,
            }
        )

    verdict_counts = Counter(row["verdict"] for row in rows)
    accepted = [row for row in rows if row["verdict"] == "partial_relation_local"]
    relation_cell_counts = Counter(
        relation["relation_id"]
        for row in accepted
        for relation in row["matched_relations"]
    )
    max_depth_counts = Counter(str(row["maximum_matching_relation_depth"]) for row in accepted)
    accepted_ids = {row["cell_id"] for row in accepted}
    sensitivity_ids = set(_SENSITIVITY_NEAR_MISSES)
    historical_summary = None
    if historical_audit is not None:
        if (
            historical_audit.get("schema") != HISTORICAL_SCHEMA
            or historical_audit.get("task") != TASK
            or historical_audit.get("n_cells") != 90
        ):
            raise ValueError("unexpected historical patent fidelity audit")
        historical_rows = [
            row
            for row in historical_audit.get("rows", [])
            if row.get("verdict") == "partial_relation_local"
        ]
        historical_ids = {str(row["cell_id"]) for row in historical_rows}
        overlap = accepted_ids & historical_ids
        union_ids = accepted_ids | historical_ids
        union_depth_counts = Counter(
            str(row["maximum_matching_relation_depth"]) for row in accepted
        )
        union_depth_counts.update(
            str(row["maximum_matching_relation_depth"]) for row in historical_rows
        )
        historical_summary = {
            "historical_schema": HISTORICAL_SCHEMA,
            "n_current_partial_cells": len(accepted_ids),
            "n_historical_partial_cells": len(historical_ids),
            "n_overlapping_cells": len(overlap),
            "overlapping_cell_ids": sorted(overlap),
            "n_additive_union_cells": len(union_ids),
            "by_level": {
                level: {
                    "n_current": sum(row["level"] == level for row in accepted),
                    "n_historical": sum(
                        row["level"] == level for row in historical_rows
                    ),
                    "n_union": len(
                        {
                            str(row["cell_id"])
                            for row in [*accepted, *historical_rows]
                            if row["level"] == level
                        }
                    ),
                }
                for level in ("R1", "R2", "R3")
            },
            "maximum_matching_relation_depth_counts": dict(
                sorted(union_depth_counts.items())
            ),
            "provenance_warning": (
                "the union is descriptive only: the historical six are manual "
                "oracle-conditioned hybrids, whereas the current eight are pure-code "
                "static partial relation matches"
            ),
        }
    else:
        historical_ids = set()
        union_ids = accepted_ids

    weighted = {
        "label": "post_hoc_design_weighted_conditional_sensitivity",
        "not_a_codability_or_prevalence_certification": True,
        "conditioning": (
            "conditional on the frozen patent panel, this manual program class, and the "
            "conservative static-fidelity adjudication"
        ),
        "conservative_eight": _weighted_sensitivity(panel_rows, accepted_ids),
        "broader_twelve_including_near_misses": _weighted_sensitivity(
            panel_rows, accepted_ids | sensitivity_ids
        ),
    }
    if historical_audit is not None:
        weighted["historical_six"] = _weighted_sensitivity(
            panel_rows, historical_ids
        )
        weighted["additive_union_fourteen"] = _weighted_sensitivity(
            panel_rows, union_ids
        )
    return {
        "schema": SCHEMA,
        "status": "conservative-static-adjudication-complete",
        "task": TASK,
        "n_cells": len(rows),
        "program_schema": PROGRAM_SCHEMA,
        "train_receipt_schema": train_receipt.get("schema") if train_receipt else None,
        "source_panel_content_sha256": panel.get("panel_content_sha256"),
        "design_scope": "static_contract_fidelity_plus_train_operational_applicability",
        "forbidden_inputs_loaded": False,
        "execution_performed_by_this_audit": False,
        "fidelity_rule": (
            "accept only when the panel criterion explicitly contains the executable relation; "
            "applicability detection, thematic proximity, and a successful train execution cannot "
            "repair a relation mismatch"
        ),
        "cap_rule": (
            "119/150 compiler-train texts meet the declared character cap; finite positive witnesses "
            "and local counter-witnesses may be replayed, but zero/absence, whole-source completeness, "
            "and global compliance may not be inferred"
        ),
        "summary": {
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "n_partial_relation_local_cells": len(accepted),
            "n_exact_whole_construct_cells": 0,
            "n_sensitivity_near_miss_cells": len(_SENSITIVITY_NEAR_MISSES),
            "relation_cell_counts": dict(sorted(relation_cell_counts.items())),
            "maximum_matching_relation_depth_counts": dict(sorted(max_depth_counts.items())),
            "by_level": {
                level: {
                    "n_cells": sum(row["level"] == level for row in rows),
                    "n_partial_relation_local": sum(
                        row["level"] == level
                        and row["verdict"] == "partial_relation_local"
                        for row in rows
                    ),
                    "n_sensitivity_near_miss": sum(
                        row["level"] == level
                        and row["verdict"] == "sensitivity_near_miss_not_accepted"
                        for row in rows
                    ),
                }
                for level in ("R1", "R2", "R3")
            },
            "additive_union_with_historical": historical_summary,
            "posthoc_design_weighted_conditional_sensitivity": weighted,
        },
        "interpretation_limits": [
            "eight partial cells are static relation matches, not eight reconstructed metrics",
            "three section-presence matches are constant in compiler train and operationally non-discriminating",
            "no accepted cell has exact whole-construct fidelity",
            "no prompt/reference reconstruction or isomorphism result is produced here",
            "the manual additive seed is not autonomous discovery",
            "bounded non-discovery is never evidence of tacitness",
        ],
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--train-receipt", type=Path)
    parser.add_argument("--historical-audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    train = (
        json.loads(args.train_receipt.read_text(encoding="utf-8"))
        if args.train_receipt
        else None
    )
    historical = (
        json.loads(args.historical_audit.read_text(encoding="utf-8"))
        if args.historical_audit
        else None
    )
    result = build_audit(panel, train, historical)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
