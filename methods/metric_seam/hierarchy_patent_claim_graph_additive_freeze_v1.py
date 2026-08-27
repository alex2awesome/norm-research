"""Freeze additive patent mappings after source audit and train-only execution.

This gate has no reference, outcome, prompt, heldout, prior-art, examiner, API,
model, or accelerator input.  It does not fit thresholds.  It merely records
which source-audited relation mappings produced at least one finite certificate
on compiler-train and therefore proceed unchanged to pre-reference heldout
execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping


SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-train-freeze.v1"
AUDIT_SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-fidelity.v1"
EXECUTION_SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-execution.v2"
PROGRAM_SCHEMA = "metric-seam.patent-claim-graph-additive.v2"
SUPERSEDED_EXECUTION_SCHEMA = (
    "metric-seam.hierarchy-patent-claim-graph-additive-execution.v1"
)


def _classify(n_items: int) -> str:
    if n_items >= 30:
        return "train_observed_dense"
    if n_items >= 5:
        return "train_observed"
    if n_items >= 1:
        return "train_observed_sparse"
    return "train_not_observed_static_only"


def build_freeze(audit: Mapping, train: Mapping, superseded_train: Mapping) -> dict:
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("task") != "patents"
        or audit.get("program_schema") != PROGRAM_SCHEMA
    ):
        raise ValueError("unexpected additive construct audit")
    if (
        train.get("schema") != EXECUTION_SCHEMA
        or train.get("program_schema") != PROGRAM_SCHEMA
        or train.get("phase") != "compiler_train"
    ):
        raise ValueError("unexpected compiler-train execution")
    design = train.get("design", {})
    forbidden_false = (
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_used",
        "model_or_api_calls_made",
        "accelerators_used",
        "whole_patent_score_emitted",
        "codability_reconstruction_or_isomorphism_measured",
    )
    if any(design.get(field) is not False for field in forbidden_false):
        raise ValueError("compiler-train execution violates blind pure-code design")
    if design.get("exact_frozen_ctext_used") is not True:
        raise ValueError("compiler-train execution did not use exact frozen ctext")
    if train.get("summary", {}).get("failure_types") != {}:
        raise ValueError("compiler-train execution has program failures")
    if (
        superseded_train.get("schema") != SUPERSEDED_EXECUTION_SCHEMA
        or superseded_train.get("phase") != "compiler_train"
    ):
        raise ValueError("unexpected superseded train receipt")

    relation_summaries = train["summary"]["relation_certificates"]
    relation_status = {
        relation_id: {
            "n_train_items_with_finite_certificates": summary[
                "n_items_with_finite_certificates"
            ],
            "n_train_certificates": summary["n_certificates"],
            "status": _classify(summary["n_items_with_finite_certificates"]),
        }
        for relation_id, summary in sorted(relation_summaries.items())
    }
    accepted = [
        row for row in audit["rows"] if row["verdict"] == "partial_relation_local"
    ]
    mappings = []
    for row in accepted:
        for relation in row["matched_relations"]:
            status = relation_status[relation["relation_id"]]
            mappings.append(
                {
                    "cell_id": row["cell_id"],
                    "level": row["level"],
                    "selection_rank": row["selection_rank"],
                    "relation_id": relation["relation_id"],
                    "depth": relation["depth"],
                    "train_status": status["status"],
                    "n_train_items_with_finite_certificates": status[
                        "n_train_items_with_finite_certificates"
                    ],
                    "selected_for_heldout_pre_reference": status[
                        "n_train_items_with_finite_certificates"
                    ]
                    > 0,
                }
            )
    selected = [row for row in mappings if row["selected_for_heldout_pre_reference"]]
    selected_cells = {row["cell_id"] for row in selected}
    return {
        "schema": SCHEMA,
        "task": "patents",
        "program_schema": PROGRAM_SCHEMA,
        "design": {
            "selection_inputs": [
                "source-only construct audit",
                "compiler-train finite-certificate incidence",
            ],
            "heldout_text_loaded": False,
            "outcome_or_reference_values_loaded": False,
            "prompt_outputs_loaded": False,
            "external_supervision_used": False,
            "selection_rule": (
                "a source-audited relation mapping proceeds unchanged when its relation emitted "
                "at least one finite compiler-train certificate; sparse status is retained"
            ),
            "threshold_or_weight_fitting_performed": False,
        },
        "sources": {
            "construct_audit_schema": audit["schema"],
            "compiler_train_schema": train["schema"],
            "compiler_train_sources": train["sources"],
        },
        "relation_train_status": relation_status,
        "mappings": mappings,
        "summary": {
            "n_static_partial_cells": len(accepted),
            "n_static_relation_mappings": len(mappings),
            "n_selected_cells": len(selected_cells),
            "n_selected_relation_mappings": len(selected),
            "selected_cells_by_level": dict(
                sorted(
                    Counter(
                        row["level"]
                        for row in accepted
                        if row["cell_id"] in selected_cells
                    ).items()
                )
            ),
            "selected_cell_maximum_depth_counts": dict(
                sorted(
                    Counter(
                        str(
                            max(
                                mapping["depth"]
                                for mapping in selected
                                if mapping["cell_id"] == cell_id
                            )
                        )
                        for cell_id in selected_cells
                    ).items()
                )
            ),
            "train_status_mapping_counts": dict(
                sorted(Counter(row["train_status"] for row in mappings).items())
            ),
            "whole_construct_cells": 0,
        },
        "prefreeze_supersession_incident": {
            "superseded_schema": superseded_train["schema"],
            "superseded_program_schema": superseded_train["program_schema"],
            "heldout_was_run_under_superseded_program": False,
            "defect": (
                "two-part node boundary kind overwrote the certificate kind field; the graph "
                "witness itself was unchanged, but the receipt schema was not certificate-safe"
            ),
            "disposition": (
                "preserved as a pre-freeze development receipt; v2 renames boundary_kind, "
                "tightens head-only term matching, binds program/runner source hashes, and is "
                "the sole input to this freeze"
            ),
        },
        "claim_limits": {
            "codability_claim_permitted": False,
            "prompt_articulability_measured": False,
            "reference_reconstruction_measured": False,
            "isomorphism_measured": False,
            "negative_result_establishes_tacitness": False,
        },
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "metric_seam_pilot" / "hierarchy_r123"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=out / "patents_claim_graph_additive_construct_fidelity_v1.json",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=out / "patents_claim_graph_additive_compiler_train_v2.json",
    )
    parser.add_argument(
        "--superseded-train",
        type=Path,
        default=out / "patents_claim_graph_additive_compiler_train_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=out / "patents_claim_graph_additive_train_freeze_v1.json",
    )
    args = parser.parse_args()
    artifact = build_freeze(
        _load(args.audit), _load(args.train), _load(args.superseded_train)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
