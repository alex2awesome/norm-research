"""Descriptive static science-claim relation rates over the hierarchy panel.

This module joins the frozen 90-cell peer-review panel to the source-only science
claim construct-fidelity audit.  It never executes the verifier and never reads
articles, items, outcomes, historical certificates, program outputs, prompt outputs,
correlations, or reconstruction results.

The conditional expansion is a point estimate over eligible native action-node
records under within-stratum exchangeability of the deterministic outcome-blind SHA
sample.  It is not design-unbiased without that assumption, has no confidence
interval, and is not a codability, reconstruction, isomorphism, or hierarchy-trend
estimate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.adjudicate_science_claim_construct_fidelity import (
    SCHEMA as FIDELITY_SCHEMA,
)


SCHEMA = "metric-seam.science-claim-static-witness-prevalence.v1"
TASK = "peer-review"
LEVELS = ("R1", "R2", "R3")
EXPANSION_KEY = "eligible_inventory_stratum_expansion"
OUTCOMES = (
    "retrieved_candidate",
    "relation_local_static_fidelity",
    "depth3_relation_local_static_fidelity",
    "whole_construct_exact",
)


class ScienceClaimPrevalenceError(ValueError):
    """Raised when the frozen panel/audit descriptive join fails closed."""


def _rate(rows: Sequence[Mapping], outcome: str, *, weighted: bool) -> dict:
    if not rows:
        return {
            "n_sampled_nodes": 0,
            "expanded_population_nodes": 0.0,
            "expanded_positive_nodes": 0.0,
            "rate": None,
        }
    weights = [float(row["design_weight"]) if weighted else 1.0 for row in rows]
    denominator = sum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ScienceClaimPrevalenceError(
            "descriptive denominator must be finite and positive"
        )
    numerator = sum(
        weight * bool(row[outcome]) for weight, row in zip(weights, rows)
    )
    return {
        "n_sampled_nodes": len(rows),
        "expanded_population_nodes": round(denominator, 6),
        "expanded_positive_nodes": round(numerator, 6),
        "rate": round(numerator / denominator, 6),
    }


def _scope(rows: Sequence[Mapping]) -> dict:
    return {
        "n_sampled_nodes": len(rows),
        "balanced_panel": {
            outcome: _rate(rows, outcome, weighted=False) for outcome in OUTCOMES
        },
        EXPANSION_KEY: {
            outcome: _rate(rows, outcome, weighted=True) for outcome in OUTCOMES
        },
    }


def _validate_sampling_frame(panel: Mapping, cells: Mapping[str, Mapping]) -> dict:
    inventory_rows = [
        row for row in panel.get("inventory", []) if row.get("task") == TASK
    ]
    if {row.get("level") for row in inventory_rows} != set(LEVELS):
        raise ScienceClaimPrevalenceError(
            "peer-review inventory must contain exactly R1/R2/R3"
        )
    inventory = {str(row["level"]): row for row in inventory_rows}
    strata: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for cell in cells.values():
        strata[
            (
                str(cell["level"]),
                str(cell["source_kind"]),
                str(cell["breadth_stratum"]),
            )
        ].append(cell)
    for key, rows in strata.items():
        try:
            populations = {int(row["stratum_population_n"]) for row in rows}
            selected = {int(row["stratum_selected_n"]) for row in rows}
            probabilities = {float(row["inclusion_probability"]) for row in rows}
            weights = {float(row["design_weight"]) for row in rows}
        except (KeyError, TypeError, ValueError) as error:
            raise ScienceClaimPrevalenceError(
                f"invalid peer-review stratum metadata for {key}"
            ) from error
        if len(populations) != 1 or len(selected) != 1:
            raise ScienceClaimPrevalenceError(
                f"inconsistent peer-review stratum counts for {key}"
            )
        population_n, selected_n = next(iter(populations)), next(iter(selected))
        if selected_n != len(rows) or not 0 < selected_n <= population_n:
            raise ScienceClaimPrevalenceError(
                f"invalid peer-review selected count for {key}"
            )
        if (
            len(probabilities) != 1
            or len(weights) != 1
            or not math.isclose(
                next(iter(probabilities)),
                selected_n / population_n,
                abs_tol=1e-12,
            )
            or not math.isclose(
                next(iter(weights)),
                population_n / selected_n,
                abs_tol=1e-12,
            )
        ):
            raise ScienceClaimPrevalenceError(
                f"peer-review inclusion fraction/design weight drifted for {key}"
            )
    stratum_population = Counter()
    for (level, _kind, _breadth), rows in strata.items():
        stratum_population[level] += int(rows[0]["stratum_population_n"])
    eligible_by_level = {
        level: int(inventory[level]["n_eligible_nodes"]) for level in LEVELS
    }
    complete_by_level = {
        level: int(inventory[level]["n_complete_nodes"]) for level in LEVELS
    }
    if dict(stratum_population) != eligible_by_level:
        raise ScienceClaimPrevalenceError(
            "peer-review strata do not sum to the eligible inventory"
        )
    weighted_total = sum(float(cell["design_weight"]) for cell in cells.values())
    eligible_total = sum(eligible_by_level.values())
    if not math.isclose(weighted_total, eligible_total, abs_tol=1e-9):
        raise ScienceClaimPrevalenceError(
            "peer-review weights do not expand to the eligible inventory"
        )
    complete_total = sum(complete_by_level.values())
    return {
        "n_complete_action_node_records": complete_total,
        "n_eligible_action_node_records": eligible_total,
        "n_excluded_by_frozen_eligibility_rule": complete_total - eligible_total,
        "complete_by_level": complete_by_level,
        "eligible_by_level": eligible_by_level,
        "n_sampling_strata": len(strata),
        "selected_per_stratum": sorted({len(rows) for rows in strata.values()}),
        "eligibility_rule": (
            "nonempty name, at least 8 description words, and at least 1 child"
        ),
    }


def _validate_audit_summary(fidelity: Mapping, audit_rows: Sequence[Mapping]) -> None:
    verdicts = Counter(str(row.get("verdict")) for row in audit_rows)
    summary = fidelity.get("summary")
    if not isinstance(summary, Mapping):
        raise ScienceClaimPrevalenceError("science fidelity summary is missing")
    if summary.get("verdict_counts") != dict(sorted(verdicts.items())):
        raise ScienceClaimPrevalenceError("science fidelity verdict summary drifted")
    retrieved = sum(row.get("candidate_capability_id") is not None for row in audit_rows)
    partial = verdicts["partial_relation_local"]
    if summary.get("n_retrieved") != retrieved:
        raise ScienceClaimPrevalenceError("science retrieved summary drifted")
    if summary.get("n_partial_relation_local") != partial:
        raise ScienceClaimPrevalenceError("science relation-local summary drifted")
    if summary.get("n_with_depth3_matching_relation") != partial:
        raise ScienceClaimPrevalenceError("science depth-3 summary drifted")
    for zero_field in (
        "n_exact_whole_construct",
        "n_execution_witnesses",
        "n_external_scientific_truth_claims",
        "n_automatic_discoveries",
    ):
        if summary.get(zero_field) != 0:
            raise ScienceClaimPrevalenceError(
                f"science static summary crossed forbidden boundary: {zero_field}"
            )


def build_science_claim_prevalence(
    panel: Mapping, fidelity: Mapping, *, sources: Mapping | None = None
) -> dict:
    """Build descriptive static rates after strict task-local validation."""

    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise ScienceClaimPrevalenceError("unexpected hierarchy panel schema")
    if not isinstance(panel.get("panel_content_sha256"), str):
        raise ScienceClaimPrevalenceError("hierarchy panel has no content identity")
    if fidelity.get("schema") != FIDELITY_SCHEMA or fidelity.get("task") != TASK:
        raise ScienceClaimPrevalenceError(
            "unexpected science claim construct-fidelity artifact"
        )
    if (
        fidelity.get("status")
        != "static-relation-local-adjudication-complete-pre-execution"
    ):
        raise ScienceClaimPrevalenceError("science fidelity audit is not complete")
    if fidelity.get("source_panel_content_sha256") != panel.get(
        "panel_content_sha256"
    ):
        raise ScienceClaimPrevalenceError(
            "science fidelity audit is bound to another panel"
        )
    for field in (
        "execution_performed",
        "articles_or_items_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "historical_certificates_or_program_outputs_loaded",
        "prompt_or_reconstruction_outputs_loaded",
        "external_supervision_loaded_for_this_audit",
    ):
        if fidelity.get(field) is not False:
            raise ScienceClaimPrevalenceError(
                f"science static audit crossed forbidden boundary: {field}"
            )
    audit_rows = fidelity.get("rows")
    if not isinstance(audit_rows, list) or len(audit_rows) != 90:
        raise ScienceClaimPrevalenceError(
            "science fidelity audit must contain exactly 90 rows"
        )
    audits = {str(row.get("cell_id")): row for row in audit_rows}
    if len(audits) != 90:
        raise ScienceClaimPrevalenceError(
            "science fidelity rows contain duplicate IDs"
        )
    _validate_audit_summary(fidelity, audit_rows)
    cells = {
        str(cell["id"]): cell
        for cell in panel.get("cells", [])
        if cell.get("task") == TASK
    }
    if len(cells) != 90 or set(cells) != set(audits):
        raise ScienceClaimPrevalenceError(
            "peer-review panel/fidelity identities do not match exactly"
        )
    frame = _validate_sampling_frame(panel, cells)

    joined = []
    for cell_id, cell in cells.items():
        audit = audits[cell_id]
        if audit.get("level") != cell.get("level") or audit.get(
            "metric_name"
        ) != cell.get("construct"):
            raise ScienceClaimPrevalenceError(
                f"{cell_id}: panel/audit source metadata drifted"
            )
        verdict = audit.get("verdict")
        if verdict not in {
            "no_candidate",
            "partial_relation_local",
            "relation_mismatch",
        }:
            raise ScienceClaimPrevalenceError(
                f"{cell_id}: unexpected science fidelity verdict"
            )
        retrieved = audit.get("candidate_capability_id") is not None
        if retrieved != (verdict != "no_candidate"):
            raise ScienceClaimPrevalenceError(
                f"{cell_id}: candidate/verdict mismatch"
            )
        relation_local = verdict == "partial_relation_local"
        depths = audit.get("eligible_relation_local_depths")
        if not isinstance(depths, list) or any(
            isinstance(depth, bool) or not isinstance(depth, int) or depth not in range(5)
            for depth in depths
        ):
            raise ScienceClaimPrevalenceError(
                f"{cell_id}: invalid matched-relation depths"
            )
        if depths != ([3] if relation_local else []):
            raise ScienceClaimPrevalenceError(
                f"{cell_id}: science relation/depth contract drifted"
            )
        if relation_local != bool(
            audit.get("eligible_for_later_relation_local_execution", False)
        ):
            raise ScienceClaimPrevalenceError(
                f"{cell_id}: static relation eligibility drifted"
            )
        for field in (
            "exact_whole_construct_fidelity",
            "execution_witness_established",
            "external_scientific_truth_established",
            "automatic_discovery",
        ):
            if audit.get(field, False) is not False:
                raise ScienceClaimPrevalenceError(
                    f"{cell_id}: static row crossed forbidden boundary: {field}"
                )
        if retrieved and audit.get("static_pure_code_capability") is not True:
            raise ScienceClaimPrevalenceError(
                f"{cell_id}: manual pure-code capability provenance drifted"
            )
        joined.append(
            {
                "cell_id": cell_id,
                "level": cell["level"],
                "source_kind": cell["source_kind"],
                "breadth_stratum": cell["breadth_stratum"],
                "design_weight": cell["design_weight"],
                "retrieved_candidate": retrieved,
                "relation_local_static_fidelity": relation_local,
                "depth3_relation_local_static_fidelity": relation_local
                and depths == [3],
                "whole_construct_exact": False,
            }
        )

    by_level = {
        level: _scope([row for row in joined if row["level"] == level])
        for level in LEVELS
    }
    source_kind_specific = {
        level: {
            kind: _scope(
                [
                    row
                    for row in joined
                    if row["level"] == level and row["source_kind"] == kind
                ]
            )
            for kind in sorted(
                {row["source_kind"] for row in joined if row["level"] == level}
            )
        }
        for level in LEVELS
    }
    merged = [
        row for row in joined if row["source_kind"] in {"merged_tree", "merged_group"}
    ]
    return {
        "schema": SCHEMA,
        "status": "static_descriptive_rates_complete_pre_execution",
        "task": TASK,
        "sources": dict(sources or {}),
        "panel_content_sha256": panel["panel_content_sha256"],
        "sampling_frame": frame,
        "outcome_definitions": {
            "retrieved_candidate": (
                "source-only candidate retrieval; not construct fidelity"
            ),
            "relation_local_static_fidelity": (
                "static partial match to the document-internal claim/body relation; not "
                "codability or scientific truth"
            ),
            "depth3_relation_local_static_fidelity": (
                "the same partial relation match with document-local BM25 retrieval as the "
                "maximum matched-chain depth"
            ),
            "whole_construct_exact": (
                "exact static whole-peer-review-construct relation fidelity"
            ),
        },
        "estimands": {
            "balanced_panel": (
                "unweighted descriptive rate in the balanced 30-node-per-level panel"
            ),
            EXPANSION_KEY: (
                "conditional stratum-expansion point estimate over 675 eligible native "
                "action-node records, treating deterministic outcome-blind SHA rank as "
                "pseudo-random/exchangeable within source-kind x breadth x level strata"
            ),
            "sampling_uncertainty": (
                "not estimated; no randomized-design replicate weights or audited alternate "
                "samples exist"
            ),
        },
        "pooled_eligible_action_nodes": _scope(joined),
        "by_level": by_level,
        "matched_relation_depth": {
            "depth": 3,
            "depth_meaning": "document-local retrieval relation chain",
            "all_and_only_relation_local_static_matches_at_this_depth": True,
        },
        "point_sensitivities": {
            "source_kind_specific": source_kind_specific,
            "merged_only": _scope(merged),
        },
        "channel_provenance": {
            "historical_pipeline": "manually designed full-article pure-code verifier",
            "evidence_scope": "distinct body sentences within the same presented article",
            "retrieval_scope": "document-local BM25; no corpus or external retrieval",
            "certificate_scope": (
                "numeric/comparative document-internal consistency, not external scientific truth"
            ),
            "automatic_discovery": False,
        },
        "uncertainty_intervals_emitted": False,
        "execution_or_outcome_stages_emitted": False,
        "prompt_or_model_stages_emitted": False,
        "reconstruction_or_isomorphism_stages_emitted": False,
        "claim_limits": [
            "No science verifier was executed for this hierarchy pass.",
            "No articles, items, outcomes, historical certificates, program outputs, prompt outputs, correlations, reconstruction, or isomorphism results were loaded.",
            "Rates describe static relation matches to one manual pure-code capability, not peer-review metric codability.",
            "Document-internal consistency is not external scientific truth or evidence reliability.",
            "No-candidate and mismatch rows are bounded non-discovery, never evidence of tacitness.",
            "The expansion covers eligible action-node records, not unique constructs or raw rubrics.",
            "R1/R2/R3 point differences do not establish an abstraction or hierarchy-round trend.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    fidelity = json.loads(args.fidelity.read_text(encoding="utf-8"))
    payload = build_science_claim_prevalence(
        panel,
        fidelity,
        sources={
            "panel": str(args.panel),
            "construct_fidelity": str(args.fidelity),
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
