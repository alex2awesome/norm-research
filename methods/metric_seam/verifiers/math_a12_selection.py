"""Freeze the retrospective Math-a12 verifier unit against the CUF snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .cuf_snapshot import normalize_metric_name


SCHEMA = "metric-seam.math-a12-verifier-selection.v1"
CRITERION = "Precision and rigor in statements and proofs"
RELATION_ID = "explicit_rational_equality_preservation"
CUF_NODE_ID = 2
OPERATIONAL_ESTIMAND = (
    "Given one structurally proposed adjacent equality pair, determine whether "
    "both sides are identical or nonidentical rational expressions on the "
    "algebraic domain inferred by the bounded parser."
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_selection(*, snapshot_manifest: Path, relation_contract: Path) -> dict:
    snapshot = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    contract = json.loads(relation_contract.read_text(encoding="utf-8"))
    if snapshot.get("schema") != "metric-seam.cuf-bank-snapshot.v1":
        raise ValueError("unsupported CUF snapshot schema")
    if snapshot.get("task") != "math" or snapshot.get("executor") != "llama8b":
        raise ValueError("selection requires the Math llama8b CUF bank")
    if contract.get("criterion_name") != CRITERION:
        raise ValueError("relation contract criterion drift")
    relation = next(
        (
            row
            for row in contract.get("relation_decomposition", [])
            if row.get("relation_id") == RELATION_ID and row.get("channel") == "code_verifiable"
        ),
        None,
    )
    if relation is None:
        raise ValueError("relation contract lacks the code-verifiable a12 relation")

    bank_path = Path(snapshot["snapshot"]["bank_path"])
    matches = []
    for line in bank_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if normalize_metric_name(row["metric"]) == normalize_metric_name(CRITERION):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("criterion does not have one conservative CUF equality join")
    metric = matches[0]
    units = [row for row in metric["rows"] if row.get("node_id") == CUF_NODE_ID]
    if len(units) != 1 or units[0].get("verdict") != "CERTIFIED-UNIT":
        raise ValueError("selected CUF unit is absent or uncertified")
    unit = units[0]

    return {
        "schema": SCHEMA,
        "status": "retrospective_existing_capability_selection_frozen",
        "task": "math",
        "criterion_id": "a12",
        "criterion_name": CRITERION,
        "relation_id": RELATION_ID,
        "relation_scope": "presented adjacent equality pair only",
        "implemented_relation": OPERATIONAL_ESTIMAND,
        "manual_parent_decomposition": relation["construct_match"],
        "source_cuf": {
            "executor": "llama8b",
            "metric_k": metric["k"],
            "node_id": unit["node_id"],
            "level": unit["level"],
            "span": unit["span"],
            "verdict": unit["verdict"],
        },
        "inputs": {
            "snapshot_manifest": {
                "path": str(snapshot_manifest),
                "sha256": _sha(snapshot_manifest),
            },
            "bank": {"path": str(bank_path), "sha256": _sha(bank_path)},
            "relation_contract": {
                "path": str(relation_contract),
                "sha256": _sha(relation_contract),
            },
        },
        "selection_provenance": {
            "criterion_join": "unique equality after conservative Unicode/whitespace normalization",
            "subrelation_decomposition": "pre-existing manually authored relation contract",
            "automatic_discovery_claimed": False,
            "pristine_blind_authorship_claimed": False,
            "legacy_program_and_aggregate_train_summary_preexisted": True,
            "selected_cuf_span_certifies_operational_estimand_directly": False,
            "heldout_or_prompt_reference_used_for_this_adapter": False,
        },
        "claim_limits": [
            "CUF certification concerns prompt-addressability for llama8b, not code verification.",
            "The selected CUF span is broad; the equality-pair relation is a manual decomposition beneath it.",
            "Exact nonidentity is a relation-instance violation, not a document error without claim scope.",
            "The operation infers denominator obligations and does not recover a document-declared domain.",
            "This selection reuses a manually constructed deep-code capability as a pipeline seed.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--relation-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_selection(
        snapshot_manifest=args.snapshot_manifest,
        relation_contract=args.relation_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
