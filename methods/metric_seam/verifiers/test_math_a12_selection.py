from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam.verifiers.math_a12_selection import (
    CRITERION,
    RELATION_ID,
    build_selection,
)


def test_selection_requires_exact_criterion_join_and_certified_unit(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    bank.write_text(
        json.dumps(
            {
                "metric": CRITERION,
                "k": 12,
                "rows": [
                    {
                        "node_id": 2,
                        "level": 2,
                        "span": "eliminate ambiguity and errors.",
                        "verdict": "CERTIFIED-UNIT",
                    }
                ],
            }
        )
        + "\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "metric-seam.cuf-bank-snapshot.v1",
                "task": "math",
                "executor": "llama8b",
                "snapshot": {"bank_path": str(bank)},
            }
        )
    )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "criterion_name": CRITERION,
                "relation_decomposition": [
                    {
                        "relation_id": RELATION_ID,
                        "channel": "code_verifiable",
                        "construct_match": "preserve a rational equality",
                    }
                ],
            }
        )
    )
    result = build_selection(snapshot_manifest=manifest, relation_contract=contract)
    assert result["source_cuf"]["node_id"] == 2
    assert result["selection_provenance"]["automatic_discovery_claimed"] is False
    assert result["selection_provenance"]["selected_cuf_span_certifies_operational_estimand_directly"] is False
    assert "inferred" in result["implemented_relation"]
