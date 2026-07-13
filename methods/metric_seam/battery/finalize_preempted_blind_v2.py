#!/usr/bin/env python3
"""Finalize a blind candidate rejected by a pre-frozen construct adversary.

This path deliberately does not open a held-out reference. Construct-fidelity failure
preempts a correlation claim and yields ``proxy_mismatch`` with isomorphism unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from methods.metric_seam.reconstruction_v2 import (
        AxisEvidence,
        DiscoveryMode,
        PipelineStatus,
        ReconstructionEvidence,
        SelectionMode,
        Status,
        claim_permissions,
        classify,
    )
except ModuleNotFoundError:  # Support direct ``python path/to/script.py`` execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from methods.metric_seam.reconstruction_v2 import (  # type: ignore[no-redef]
        AxisEvidence,
        DiscoveryMode,
        PipelineStatus,
        ReconstructionEvidence,
        SelectionMode,
        Status,
        claim_permissions,
        classify,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_preempted(
    *,
    criterion_id: str,
    relation_id: str,
    candidate: Path,
    prepare_manifest: Path,
    adversary_result: Path,
) -> dict[str, Any]:
    """Build a preempted record without accepting or reading a reference path."""

    adversary = json.loads(adversary_result.read_text())
    if adversary.get("suite_pass") is not False:
        raise ValueError("preemption finalizer requires a failed adversary suite")
    if adversary.get("freeze_verified") is not True:
        raise ValueError("adversary freeze was not verified")
    if (adversary.get("conditions") or {}).get("pair_category_floor") is not False:
        raise ValueError("preemption requires a failed construct-category floor")
    recorded_candidate_sha = (adversary.get("candidate") or {}).get("sha256")
    actual_candidate_sha = sha256(candidate)
    if recorded_candidate_sha != actual_candidate_sha:
        raise ValueError("candidate does not match the adversary result")
    metrics = adversary["metrics"]
    evidence = ReconstructionEvidence(
        criterion_id=criterion_id,
        relation_id=relation_id,
        discovery_mode=DiscoveryMode.AGENTIC,
        pipeline_status=PipelineStatus.SELECTED,
        selection_mode=SelectionMode.BLIND_AGENTIC,
        articulability=AxisEvidence(
            Status.UNAVAILABLE,
            metric="prompt_channel_not_used_by_candidate",
            note="The candidate requested no prompt fields.",
        ),
        verifiability=AxisEvidence(
            Status.FAIL,
            score=metrics["pair_pass_rate"],
            metric="independent_adversary_pair_pass_rate",
            artifact=str(adversary_result),
            note=(
                "The code executed on every case but failed the frozen category-floor rule; "
                "high aggregate pair sensitivity is not a construct witness."
            ),
        ),
        hybrid=AxisEvidence(
            Status.UNAVAILABLE,
            metric="candidate_hybrid_channel_not_implemented",
        ),
        reference_isomorphism=AxisEvidence(
            Status.UNAVAILABLE,
            metric="heldout_reference_deliberately_unopened",
            note=(
                "Construct-fidelity failure preempted held-out reference access; the sealed "
                "split remains reusable only under a separately registered future protocol."
            ),
        ),
        construct_fidelity=AxisEvidence(
            Status.FAIL,
            score=metrics["minimum_pair_category_pass_rate"],
            metric="minimum_frozen_adversary_category_pass_rate",
            artifact=str(adversary_result),
            note=(
                "The candidate missed the frozen semantic-target category and failed the "
                "subequation-grouping category floor."
            ),
        ),
        provenance_note=(
            "Clean-room compiler saw only the ctext-only TRAIN bundle, projected contract, and "
            "allowed base/math operations. Candidate and independent adversary were frozen "
            "before execution; no held-out reference was opened."
        ),
    )
    payload = evidence.as_dict()
    payload["outcome"] = classify(evidence).value
    payload["claim_permissions"] = claim_permissions(evidence)
    payload["heldout_reference_opened"] = False
    payload["preemption_reason"] = "construct_fidelity_failed_before_reference_access"
    payload["adversary_summary"] = {
        "pair_cases": adversary["case_counts"]["pair_cases"],
        "range_cases": adversary["case_counts"]["range_cases"],
        "pair_pass_rate": metrics["pair_pass_rate"],
        "range_pass_rate": metrics["range_pass_rate"],
        "minimum_pair_category_pass_rate": metrics["minimum_pair_category_pass_rate"],
        "failed_conditions": [
            key for key, passed in adversary["conditions"].items() if not passed
        ],
    }
    payload["artifact_sha256"] = {
        "candidate": actual_candidate_sha,
        "prepare_manifest": sha256(prepare_manifest),
        "adversary_result": sha256(adversary_result),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion-id", required=True)
    parser.add_argument("--relation-id", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--prepare-manifest", type=Path, required=True)
    parser.add_argument("--adversary-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = finalize_preempted(
        criterion_id=args.criterion_id,
        relation_id=args.relation_id,
        candidate=args.candidate,
        prepare_manifest=args.prepare_manifest,
        adversary_result=args.adversary_result,
    )
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "outcome": payload["outcome"],
        "heldout_reference_opened": False,
        "adversary_summary": payload["adversary_summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
