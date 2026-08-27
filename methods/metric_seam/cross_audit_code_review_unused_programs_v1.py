"""Independently adjudicate the nine unused-program code-review proposals.

This additive audit is deliberately narrower than execution.  It checks the
requested/implemented relation, current candidate source, applicability and
abstention contract, and the maximum decision-contributing program depth.  It
never reads task items, outputs, references, outcomes, or prompt responses.

The first source pass exposed four program defects.  Those programs were
repaired before this artifact was frozen; executable counterexamples live in
``test_code_review_candidate_program_repairs_v1.py``.  Historical hierarchy
artifacts remain untouched, so accepted rows extend rather than rewrite the
canonical corrected 50/90 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/"
    "code_review_unused_program_feasibility_inventory_v1.json"
)
OUT = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/"
    "code_review_unused_program_construct_cross_audit_v1.json"
)
SCHEMA = "metric-seam.code-review-unused-program-construct-cross-audit.v1"


class CrossAuditError(ValueError):
    pass


# These are independent source-level judgments, keyed to the frozen hierarchy
# cell rather than to aspect name similarity.  Depth is the maximum
# decision-contributing path under hierarchy_batch.DEPTH_VOCABULARY.
JUDGMENTS: dict[str, dict[str, Any]] = {
    "TB::code-review::general::R1::merged_tree::31::f239fc227b096b1638ef": {
        "depth": 2,
        "rationale": "AST relations among public fields, accessors, and return annotations implement stable-abstraction leakage, but not API size or version evolution.",
    },
    "TB::code-review::general::R1::merged_tree::55::742ea284277cb2d01283": {
        "depth": 4,
        "rationale": "Reconstructed files are executed through canonical formatter environments; this implements formatting conformance only, not naming, idioms, or project-policy authority.",
    },
    "TB::code-review::general::R1::parented_tree::21::e1c3bc938276569dc4bc": {
        "depth": 4,
        "rationale": "Canonical ecosystem formatter execution is a faithful convention sub-relation; repository-local comparison and justified deviations remain unimplemented.",
    },
    "TB::code-review::general::R2::grandparent::3::2e742c97fa237c1e3aa6": {
        "depth": 2,
        "rationale": "Cross-node AST analysis of loops, recursion, memoization, binary-search shape, and sorting implements a bounded algorithmic-complexity relation, not empirical performance discipline.",
    },
    "TB::code-review::general::R2::grandparent::43::12b78f2174bf884b965b": {
        "depth": 4,
        "rationale": "Ruff is actually executed and its typed diagnostic set is normalized by added lines, implementing a Python static-defect-detection slice only.",
    },
    "TB::code-review::general::R2::merged_group::123::a2a43bff0666769d0822": {
        "depth": 2,
        "rationale": "The program relates changed non-test source paths to changed test paths and now abstains without a source denominator; test behavior and broader readiness remain unimplemented.",
    },
    "TB::code-review::general::R2::merged_group::40::46a2dd9793d0881557cd": {
        "depth": 4,
        "rationale": "Canonical ecosystem formatter execution implements ecosystem formatting adherence only, not local conventions or deviation justification.",
    },
    "TB::code-review::general::R3::grandparent::11::2c562440277fde02d164": {
        "depth": 4,
        "rationale": "Config parsing plus external lint/secret-scanner execution implements managed and bounded-size configuration, not general simplicity, magic, or speculative abstraction.",
    },
    "TB::code-review::general::R3::grandparent::12::090af4e25717489168bb": {
        "depth": 2,
        "rationale": "Cross-node AST complexity classification implements the efficient-algorithm source-shape slice, not profiles, workloads, or empirical trade-offs.",
    },
}

REPAIR_EVIDENCE = {
    "a35": "Final-annotated Python class fields no longer count as mutable public leaks.",
    "a72": "Formatter inputs restore the projection-stripped terminal newline and aggregate exact conforming/measurable added-line counts.",
    "a181": "Ruff JSON diagnostics are counted; human success/summary text is never a finding.",
    "a309": "Test-only changes abstain because source-to-test correspondence has no source denominator.",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrossAuditError(f"expected object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    inventory = _load(INVENTORY)
    if (
        inventory.get("schema")
        != "metric-seam.code-review-unused-program-feasibility-inventory.v1"
        or inventory.get("status")
        != "developmental_source_only_proposals_pending_independent_audit"
    ):
        raise CrossAuditError("unexpected feasibility inventory")
    proposals = {
        row["cell_id"]: row
        for row in inventory["rows"]
        if row.get("decision") == "propose_partial_mapping"
    }
    if set(proposals) != set(JUDGMENTS):
        raise CrossAuditError("proposal set differs from the independent audit set")

    rows = []
    for cell_id in sorted(proposals):
        proposal = proposals[cell_id]
        judgment = JUDGMENTS[cell_id]
        source_rel = proposal["candidate"]["source_path"]
        source = ROOT / source_rel
        source_sha = _sha256(source)
        if source_sha != proposal["candidate"]["source_sha256"]:
            raise CrossAuditError(f"candidate source binding drifted: {cell_id}")
        aspect_id = proposal["candidate"]["aspect_id"]
        depth = int(judgment["depth"])
        rows.append(
            {
                "cell_id": cell_id,
                "level": proposal["level"],
                "metric_name": proposal["metric_name"],
                "candidate_aspect_id": aspect_id,
                "candidate_source": source_rel,
                "candidate_source_sha256": source_sha,
                "verdict": "accepted_partial_relation_local",
                "scope": "subrelation_only",
                "whole_construct_exact": False,
                "requested_relation": proposal["requested_relation"],
                "implemented_relation": proposal["implemented_relation"],
                "applicability": proposal["applicability"],
                "abstention": proposal["abstention"],
                "audited_depth": depth,
                "audited_depth_meaning": {
                    2: "cross-span or cross-section relation checking",
                    4: "environment or world execution",
                }[depth],
                "proposal_depth": proposal["proposed_matched_relation_depth"],
                "depth_corrected": depth
                != proposal["proposed_matched_relation_depth"],
                "repair_counterexample_evidence": REPAIR_EVIDENCE.get(aspect_id),
                "independent_rationale": judgment["rationale"],
            }
        )

    by_level = {
        level: sum(row["level"] == level for row in rows)
        for level in ("R1", "R2", "R3")
    }
    by_depth = {
        str(depth): sum(row["audited_depth"] == depth for row in rows)
        for depth in (2, 4)
    }
    current = {"R1": 14, "R2": 15, "R3": 21}
    projected = {level: current[level] + by_level[level] for level in current}
    return {
        "schema": SCHEMA,
        "status": "independent_static_construct_audit_complete_pre_execution",
        "task": "code-review",
        "objective": "independent relation-local construct and program-depth adjudication of nine additive unused-program proposals",
        "sources": {
            str(INVENTORY.relative_to(ROOT)): {"sha256": _sha256(INVENTORY)}
        },
        "sealed_inputs": {
            "task_items_loaded": False,
            "program_outputs_loaded": False,
            "outcomes_loaded": False,
            "references_loaded": False,
            "prompt_responses_loaded": False,
            "models_or_apis_called": False,
            "gpu_used": False,
            "external_supervision_used": False,
        },
        "method_origin": {
            "candidate_programs": "existing/manual retrospective seeds",
            "proposal_author": "prior independent agent inventory",
            "construct_cross_audit": "separate source inspection after adversarial synthetic repair probes",
            "automatic_discovery_claimed": False,
        },
        "summary": {
            "n_proposals_audited": len(rows),
            "n_accepted_partial_relation_local": len(rows),
            "n_rejected": 0,
            "n_unique_programs": len({row["candidate_aspect_id"] for row in rows}),
            "accepted_by_level": by_level,
            "accepted_by_audited_depth": by_depth,
            "depth_corrections": sum(row["depth_corrected"] for row in rows),
            "canonical_corrected_static_unchanged": 50,
            "additive_static_union_if_adopted": 59,
            "additive_static_union_by_level_if_adopted": projected,
        },
        "claim_limits": [
            "This promotes nine mappings through static construct fidelity only; no task-item execution occurred in this audit.",
            "The canonical historical corrected funnel remains 50/90 and is not rewritten.",
            "Train measurability, heldout execution, prompt articulability, reconstruction, isomorphism, codability, and whole-construct verifiability remain unmeasured.",
            "Depth records maximum decision-contributing program path, not construct difficulty or performance quality.",
        ],
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != serialized:
            raise CrossAuditError(f"checked artifact differs: {args.out}")
        return
    args.out.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
