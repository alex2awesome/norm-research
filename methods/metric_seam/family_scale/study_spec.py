#!/usr/bin/env python3
"""Freeze the outcome-blind 60-metric technical family-scale pilot.

This compiler samples metric texts from the already-frozen hierarchy panel.  It
deliberately emits no corpus text, prior program, historical score, child
rubric, or outcome.  Its output is the only input licensed for blind relation
decomposition fleets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "metric-seam.family-scale-study.v1"
DOMAINS = {
    "math-stackexchange": "math",
    "code-review": "code",
    "peer-review": "science_full_article",
    "patents": "patents",
}
LEVELS = ("R1", "R2", "R3")
SALT = "metric-seam-family-scale-technical-pilot-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _rank(cell: Mapping[str, object]) -> str:
    return hashlib.sha256(f"{SALT}\0{cell['id']}".encode()).hexdigest()


def compile_study(panel: Mapping[str, object], *, per_domain_level: int = 5) -> dict[str, object]:
    if panel.get("status") != "frozen-outcome-blind-hierarchy-sample":
        raise ValueError("study requires the frozen outcome-blind hierarchy panel")
    if per_domain_level < 1:
        raise ValueError("per_domain_level must be positive")
    cells = panel.get("cells")
    if not isinstance(cells, list):
        raise ValueError("panel cells missing")

    selected: list[dict[str, object]] = []
    for task, domain in DOMAINS.items():
        for level in LEVELS:
            candidates = [
                cell for cell in cells
                if cell.get("task") == task and cell.get("level") == level
                and isinstance(cell.get("construct"), str)
                and isinstance(cell.get("description"), str)
            ]
            candidates.sort(key=lambda cell: (_rank(cell), str(cell["id"])))
            if len(candidates) < per_domain_level:
                raise ValueError(f"insufficient cells for {task}/{level}")
            for cell in candidates[:per_domain_level]:
                metric_text = {
                    "construct": cell["construct"],
                    "description": cell["description"],
                }
                selected.append({
                    "metric_id": cell["id"],
                    "task": task,
                    "domain": domain,
                    "level": level,
                    "metric_text": metric_text,
                    "metric_text_sha256": _sha(metric_text),
                    "decomposition_input_fields": ["construct", "description"],
                })

    selected.sort(key=lambda row: (str(row["domain"]), str(row["level"]), str(row["metric_id"])))
    study = {
        "schema": SCHEMA,
        "status": "frozen_before_decomposition_or_corpus_contact",
        "objective": "Estimate relation-family incidence, operational support, and prompt/code reconstruction in four technical domains.",
        "sampling": {
            "frame": "frozen hierarchy panel v3",
            "rule": f"stable-hash select {per_domain_level} cells per domain x R1/R2/R3",
            "salt": SALT,
            "n_metrics": len(selected),
            "n_per_domain": per_domain_level * len(LEVELS),
            "symmetric_metric_budget_not_symmetric_yield_assumption": True,
        },
        "decomposition": {
            "blind_to": ["corpus", "items", "programs", "program_outputs", "prior_judgments", "heldout", "child_rubrics"],
            "independent_fleets": 3,
            "relations_per_metric_guidance": [2, 5],
            "stability_subsample": "all 60 pilot metrics",
            "structural_reconstruction_is_separate_from_behavioral_reconstruction": True,
        },
        "pipeline": [
            "PROPOSE", "BASE_RATE_PROBE", "AUTHOR_OR_IMPORT", "CONSTRUCT_CHALLENGE",
            "PER_NODE_GATE", "SELECT", "FREEZE", "TRANSCRIBE", "EVALUATE",
        ],
        "stop_rules": {
            "base_rate_probe_n": 30,
            "base_rate_decision": "provisional authorship kill only; repeat discrimination on full TRAIN execution",
            "maximum_authorship_rounds": 2,
            "failed_unit_reading": "no certified witness in frozen family/corpus/budget; never tacitness",
        },
        "family_certificate": {
            "unit": "relation family x domain-corpus",
            "G1": "two independently authored implementations agree on applicability and verdict and identify overlapping witnesses",
            "G2": "blind proxy traps include proxy-on/construct-satisfied and proxy-off/construct-violated controls",
            "reuse_risk": "all metric cells using a failed family/corpus certificate are uncertified together",
        },
        "occasion": {
            "shared_proposer_required": True,
            "identical_occasion_ids_and_payloads_across_channels": True,
            "whole_document_discovery": "out of scope except separately labeled blind-discovery subsample",
        },
        "prompt_execution": {
            "relations_per_call": [2, 3],
            "cooccurrence_assignment": "deterministic randomized",
            "unbatched_calibration_fraction": 0.10,
            "passes": 2,
            "call_id_is_an_analysis_cluster": True,
        },
        "primary_estimands": [
            "relation_decomposition_stability",
            "pre_authoring_corpus_support_incidence",
            "family_certification_incidence",
            "full_train_operational_witness_incidence",
            "tie_robust_prompt_code_reconstruction_fidelity",
            "program_depth_and_typed_seam_position",
        ],
        "primary_fidelity": {
            "statistic": "pairwise concordance/c-index conditional on joint applicability",
            "normalization": "two-pass prompt-target reliability ceiling",
            "mandatory_companions": ["tie_fraction", "mode_fraction", "distinct_values", "entropy", "all_funnel_denominators"],
            "uncertainty": "cluster bootstrap over metric, document, and prompt call identifiers",
            "spearman": "secondary sensitivity only",
        },
        "domain_constraints": {
            "code": "AST/dataflow/static analysis on diffs only; execution/tests excluded until pre-review corpus is approved and re-mined",
            "science_full_article": "full article source; abstract-only artifacts excluded",
            "patents": "precomputed/mock retrieval machinery permitted only with explicit provenance",
            "math": "formal parsers and symbolic execution permitted; occasion role remains a separate relation",
        },
        "cells": selected,
    }
    study["study_content_sha256"] = _sha(study)
    return study


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-domain-level", type=int, default=5)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    result = compile_study(panel, per_domain_level=args.per_domain_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
