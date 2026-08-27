from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.finalize_semantic_resolver import main


def _label(uid: str, decision: str, metric: str | None, confidence: str = "high") -> dict:
    return {
        "norm_uid": uid,
        "task": "humor",
        "decision": decision,
        "metric_id": metric,
        "confidence": confidence,
        "reason": "reason",
        "current_bank_source_sha256": "bank-sha",
    }


def test_finalizer_requires_two_exact_sources_and_audits_strict_corrections(
    tmp_path: Path, monkeypatch
) -> None:
    pack, resolver_pack = tmp_path / "pack", tmp_path / "resolver"
    items = [
        {
            "norm_uid": f"u{index}",
            "task": "humor",
            "corpus": "humor_multi",
            "row": index,
            "source_group": f"g{index}",
            "norm": f"norm {index}",
            "context": f"context {index}",
        }
        for index in range(4)
    ]
    bank = {
        "task": "humor",
        "source_sha256": "bank-sha",
        "metrics": [{"metric_id": "m0"}, {"metric_id": "m1"}],
    }
    write_jsonl(pack / "items.jsonl", items)
    (pack / "bank.json").parent.mkdir(parents=True, exist_ok=True)
    (pack / "bank.json").write_text(json.dumps(bank))
    (pack / "validation.json").write_text(
        json.dumps(
            {
                "task": "humor",
                "bank_source_sha256": "bank-sha",
                "outputs": {
                    "items": {"sha256": sha256_file(pack / "items.jsonl")},
                    "bank": {"sha256": sha256_file(pack / "bank.json")},
                },
            }
        )
    )
    write_jsonl(resolver_pack / "items.jsonl", items[1:])
    (resolver_pack / "validation.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "items": {"sha256": sha256_file(resolver_pack / "items.jsonl")}
                }
            }
        )
    )
    semantic, strict, resolver = (
        tmp_path / "semantic.jsonl",
        tmp_path / "strict.jsonl",
        tmp_path / "resolver.jsonl",
    )
    write_jsonl(
        semantic,
        [
            _label("u0", "MATCH", "m0"),
            _label("u1", "MATCH", "m0"),
            _label("u2", "MATCH", "m0"),
            _label("u3", "NO_EXPLICIT_CRITERION", None, "medium"),
        ],
    )
    write_jsonl(
        strict,
        [
            _label("u0", "MATCH", "m0"),
            _label("u2", "MATCH", "m1"),
        ],
    )
    write_jsonl(
        resolver,
        [
            _label("u1", "MATCH", "m0"),
            _label("u2", "MATCH", "m1"),
            _label("u3", "GENERIC_VERDICT", None),
        ],
    )
    outputs = {
        "resolved": tmp_path / "resolved.jsonl",
        "unresolved": tmp_path / "unresolved.jsonl",
        "disagreements": tmp_path / "disagreements.jsonl",
        "report": tmp_path / "report.json",
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_semantic_resolver",
            "--pack-root",
            str(pack),
            "--semantic-labels",
            str(semantic),
            "--strict-key",
            str(strict),
            "--resolver-pack-root",
            str(resolver_pack),
            "--resolver-labels",
            str(resolver),
            "--output",
            str(outputs["resolved"]),
            "--unresolved-output",
            str(outputs["unresolved"]),
            "--disagreements-output",
            str(outputs["disagreements"]),
            "--report",
            str(outputs["report"]),
        ],
    )
    main()
    resolved = {row["norm_uid"]: row for row in read_jsonl(outputs["resolved"])}
    assert set(resolved) == {"u0", "u1", "u2"}
    assert resolved["u2"]["metric_id"] == "m1"
    assert [row["norm_uid"] for row in read_jsonl(outputs["unresolved"])] == ["u3"]
    report = json.loads(outputs["report"].read_text())
    assert report["strict78_audit"]["correction_opportunities"] == 1
    assert report["strict78_audit"]["strict_correction_precision"] == 1.0
