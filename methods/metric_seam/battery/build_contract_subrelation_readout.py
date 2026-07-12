#!/usr/bin/env python3
"""Translate a channel-faithful contract check into per-sub-relation v2 evidence.

Each synthetic contrast remains its own relation-local record. A successful CODE probe is
a scoped verifiability witness; a successful L probe is a scoped articulability witness.
The command never infers a whole-criterion outcome and never promotes a probe-only result
to frozen-reference isomorphism or corpus discrimination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ..reconstruction_v2 import (
        AxisEvidence,
        DiscoveryMode,
        PipelineStatus,
        ReconstructionEvidence,
        RelationMatchVerdict,
        SelectionMode,
        Status,
        SubrelationEvidence,
        build_decomposition,
        decomposition_readout,
    )
except ImportError:  # direct-file execution
    import sys

    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from methods.metric_seam.reconstruction_v2 import (  # type: ignore[no-redef]
        AxisEvidence,
        DiscoveryMode,
        PipelineStatus,
        ReconstructionEvidence,
        RelationMatchVerdict,
        SelectionMode,
        Status,
        SubrelationEvidence,
        build_decomposition,
        decomposition_readout,
    )


def build_readout(
    contract: dict,
    check: dict,
    *,
    criterion_id: str,
    discovery_mode: DiscoveryMode,
) -> dict:
    observations = {row["index"]: row for row in check["probes"]}
    subrelations = []
    for index, probe in enumerate(contract["cf_probes"]):
        observed = observations[index]
        passed = observed["outcome"] == "SEPARATED"
        channel = probe["channel"]
        if channel == "CODE":
            articulability = AxisEvidence(Status.UNAVAILABLE, note="not tested by CODE probe")
            verifiability = AxisEvidence(
                Status.PASS if passed else Status.FAIL,
                score=observed.get("delta"),
                metric="signed_probe_margin",
                note="probe-local executable contrast",
            )
            relation_match = (
                RelationMatchVerdict.CODE_NATIVE
                if passed
                else RelationMatchVerdict.CAPABILITY_MISMATCH
            )
            program_relation = "frozen code path over independently presented probe texts"
        else:
            articulability = AxisEvidence(
                Status.PASS if passed else Status.FAIL,
                score=observed.get("delta"),
                metric="signed_probe_margin",
                note="probe-local frozen prompt-field contrast",
            )
            verifiability = AxisEvidence(Status.UNAVAILABLE, note="not tested by L probe")
            relation_match = (
                RelationMatchVerdict.PROMPT_NATIVE
                if passed
                else RelationMatchVerdict.CAPABILITY_MISMATCH
            )
            program_relation = "frozen LLM_FIELDS extraction plus frozen code aggregation"
        evidence = ReconstructionEvidence(
            criterion_id=criterion_id,
            relation_id=f"contract_probe_{index}",
            discovery_mode=discovery_mode,
            pipeline_status=PipelineStatus.SELECTED,
            selection_mode=SelectionMode.RETROSPECTIVE_SEED,
            articulability=articulability,
            verifiability=verifiability,
            hybrid=AxisEvidence(Status.UNAVAILABLE, note="reported only at parent gate"),
            reference_isomorphism=AxisEvidence(
                Status.UNAVAILABLE,
                note="synthetic contract check has no frozen-LLM reference comparison",
            ),
            construct_fidelity=AxisEvidence(
                Status.PASS if passed else Status.FAIL,
                score=observed.get("delta"),
                metric="probe_local_relation_fidelity",
                artifact="contract_check.json",
            ),
            provenance_note=(
                "retrospective frozen candidate; current L texts were extracted independently "
                "without pair polarity or labels"
            ),
        )
        subrelations.append(
            SubrelationEvidence(
                evidence=evidence,
                construct_relation=probe.get("why") or f"contract contrast {index}",
                program_relation=program_relation,
                relation_match=relation_match,
                note=f"contract channel={channel}; probe type={probe.get('probe_type')}",
            )
        )
    decomposition = build_decomposition(
        criterion_id,
        subrelations,
        provenance_note=(
            "probe-local readout only; corpus discrimination and whole-criterion aggregation "
            "remain separate"
        ),
    )
    result = decomposition_readout(decomposition)
    result["contract_sha256"] = check["contract_sha256"]
    result["extraction_sha256"] = check.get("extraction_sha256")
    result["code_gate"] = check["code_gate"]
    result["hybrid_gate"] = check["hybrid_gate"]
    result["discrimination_gate"] = check["discrimination_gate"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--criterion-id", required=True)
    parser.add_argument(
        "--discovery-mode", choices=[mode.value for mode in DiscoveryMode], default="manual"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_readout(
        json.loads(args.contract.read_text()),
        json.loads(args.check.read_text()),
        criterion_id=args.criterion_id,
        discovery_mode=DiscoveryMode(args.discovery_mode),
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
