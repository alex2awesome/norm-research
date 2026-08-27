"""Summarize frozen train and pre-reference heldout patent claim-graph execution."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping


SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-operational.v1"
AUDIT_SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-fidelity.v1"
FREEZE_SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-train-freeze.v1"
EXECUTION_SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-execution.v2"
PROGRAM_SCHEMA = "metric-seam.patent-claim-graph-additive.v2"


def _validate_execution(execution: Mapping, *, phase: str) -> None:
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("program_schema") != PROGRAM_SCHEMA
        or execution.get("phase") != phase
    ):
        raise ValueError(f"unexpected {phase} execution")
    design = execution.get("design", {})
    for field in (
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_used",
        "model_or_api_calls_made",
        "accelerators_used",
        "whole_patent_score_emitted",
        "codability_reconstruction_or_isomorphism_measured",
    ):
        if design.get(field) is not False:
            raise ValueError(f"{phase} violates {field}")
    if design.get("exact_frozen_ctext_used") is not True:
        raise ValueError(f"{phase} did not use exact frozen ctext")
    if execution.get("summary", {}).get("failure_types") != {}:
        raise ValueError(f"{phase} contains execution failures")


def _heldout_status(n_items: int) -> str:
    if n_items >= 30:
        return "heldout_observed_dense"
    if n_items >= 5:
        return "heldout_observed"
    if n_items >= 1:
        return "heldout_observed_sparse"
    return "heldout_bounded_non_discovery"


def build_summary(
    audit: Mapping,
    freeze: Mapping,
    train: Mapping,
    heldout: Mapping,
) -> dict:
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("program_schema") != PROGRAM_SCHEMA
        or audit.get("task") != "patents"
    ):
        raise ValueError("unexpected construct audit")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("task") != "patents":
        raise ValueError("unexpected train freeze")
    _validate_execution(train, phase="compiler_train")
    _validate_execution(heldout, phase="heldout_pre_reference")
    for source in ("program", "runner", "manifest"):
        if train["sources"][source] != heldout["sources"][source]:
            raise ValueError(f"train/heldout {source} provenance differs")
    if freeze["sources"]["compiler_train_sources"] != train["sources"]:
        raise ValueError("freeze is not bound to the supplied compiler-train receipt")
    if freeze["design"]["heldout_text_loaded"] is not False:
        raise ValueError("train freeze was not sealed from heldout text")

    heldout_relations = heldout["summary"]["relation_certificates"]
    relation_status = {}
    for relation_id, train_status in freeze["relation_train_status"].items():
        observed = heldout_relations[relation_id][
            "n_items_with_finite_certificates"
        ]
        relation_status[relation_id] = {
            **train_status,
            "n_heldout_items_with_finite_certificates": observed,
            "n_heldout_certificates": heldout_relations[relation_id]["n_certificates"],
            "heldout_status": _heldout_status(observed),
        }

    mappings = []
    for mapping in freeze["mappings"]:
        status = relation_status[mapping["relation_id"]]
        mappings.append(
            {
                **mapping,
                "n_heldout_items_with_finite_certificates": status[
                    "n_heldout_items_with_finite_certificates"
                ],
                "heldout_status": status["heldout_status"],
                "heldout_relation_operational": status[
                    "n_heldout_items_with_finite_certificates"
                ]
                > 0,
            }
        )
    operational_mappings = [row for row in mappings if row["heldout_relation_operational"]]
    static_rows = [
        row for row in audit["rows"] if row["verdict"] == "partial_relation_local"
    ]
    operational_cells = {row["cell_id"] for row in operational_mappings}
    cell_rows = []
    for row in static_rows:
        cell_mappings = [item for item in mappings if item["cell_id"] == row["cell_id"]]
        live = [item for item in cell_mappings if item["heldout_relation_operational"]]
        cell_rows.append(
            {
                "cell_id": row["cell_id"],
                "level": row["level"],
                "selection_rank": row["selection_rank"],
                "static_maximum_depth": row["maximum_matching_relation_depth"],
                "heldout_operational": bool(live),
                "heldout_operational_relation_ids": [
                    item["relation_id"] for item in live
                ],
                "heldout_operational_maximum_depth": max(
                    (item["depth"] for item in live), default=None
                ),
                "status": (
                    "heldout_finite_witness_operational"
                    if live
                    else "heldout_bounded_non_discovery"
                ),
            }
        )

    return {
        "schema": SCHEMA,
        "task": "patents",
        "program_schema": PROGRAM_SCHEMA,
        "design": {
            "compiler_train_selection_frozen_before_heldout": True,
            "same_program_and_runner_source_hashes_train_to_heldout": True,
            "exact_shared_ctext_contract": True,
            "outcome_or_reference_values_loaded": False,
            "prompt_outputs_loaded": False,
            "external_supervision_used": False,
            "model_or_api_calls_made": False,
            "accelerators_used": False,
            "heldout_readout": "finite relation witness incidence only",
        },
        "sources": {
            "construct_audit_schema": audit["schema"],
            "train_freeze_schema": freeze["schema"],
            "compiler_train": train["sources"],
            "heldout_pre_reference": heldout["sources"],
        },
        "relation_status": relation_status,
        "mappings": mappings,
        "cells": cell_rows,
        "summary": {
            "n_static_partial_cells": len(static_rows),
            "n_train_selected_cells": freeze["summary"]["n_selected_cells"],
            "n_heldout_operational_cells": len(operational_cells),
            "n_static_relation_mappings": len(mappings),
            "n_heldout_operational_relation_mappings": len(operational_mappings),
            "heldout_operational_cells_by_level": dict(
                sorted(
                    Counter(
                        row["level"] for row in cell_rows if row["heldout_operational"]
                    ).items()
                )
            ),
            "heldout_operational_cell_depth_counts": dict(
                sorted(
                    Counter(
                        str(row["heldout_operational_maximum_depth"])
                        for row in cell_rows
                        if row["heldout_operational"]
                    ).items()
                )
            ),
            "heldout_relation_status_mapping_counts": dict(
                sorted(Counter(row["heldout_status"] for row in mappings).items())
            ),
            "train_items": train["summary"]["n_items"],
            "heldout_items": heldout["summary"]["n_items"],
            "train_items_at_character_cap": train["summary"][
                "items_at_declared_character_cap"
            ],
            "heldout_items_at_character_cap": heldout["summary"][
                "items_at_declared_character_cap"
            ],
            "whole_construct_cells": 0,
        },
        "claim_limits": {
            "finite_witness_operational_is_not_reconstruction": True,
            "codability_claim_permitted": False,
            "prompt_articulability_measured": False,
            "reference_reconstruction_measured": False,
            "isomorphism_measured": False,
            "absence_from_full_patent_or_claim_set_established": False,
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
        "--freeze",
        type=Path,
        default=out / "patents_claim_graph_additive_train_freeze_v1.json",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=out / "patents_claim_graph_additive_compiler_train_v2.json",
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=out / "patents_claim_graph_additive_heldout_pre_reference_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=out / "patents_claim_graph_additive_operational_summary_v1.json",
    )
    args = parser.parse_args()
    result = build_summary(
        _load(args.audit),
        _load(args.freeze),
        _load(args.train),
        _load(args.heldout),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
