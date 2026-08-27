from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.prepare_semantic_resolver_pack import main


def _label(uid: str, decision: str, metric: str | None, confidence: str) -> dict:
    return {
        "norm_uid": uid,
        "task": "humor",
        "decision": decision,
        "metric_id": metric,
        "confidence": confidence,
        "current_bank_source_sha256": "bank-sha",
    }


def test_resolver_union_hides_predictions_and_applies_all_rules(
    tmp_path: Path, monkeypatch
) -> None:
    pack = tmp_path / "pack"
    items = [
        {"norm_uid": f"u{index}", "task": "humor", "source_group": f"g{index}"}
        for index in range(6)
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
    semantic = tmp_path / "semantic.jsonl"
    write_jsonl(
        semantic,
        [
            _label("u0", "MATCH", "m0", "high"),  # corroborated: omit
            _label("u1", "MATCH", "m0", "high"),  # key absent: include
            _label("u2", "MATCH", "m0", "high"),  # key mismatch: include
            _label("u3", "NO_EXPLICIT_CRITERION", None, "medium"),  # include
            _label("u4", "NO_EXPLICIT_CRITERION", None, "high"),  # key conflict: include
            _label("u5", "NO_EXPLICIT_CRITERION", None, "high"),  # no key: omit
        ],
    )
    key = tmp_path / "key.jsonl"
    write_jsonl(
        key,
        [
            _label("u0", "MATCH", "m0", "high"),
            _label("u2", "MATCH", "m1", "high"),
            _label("u4", "GENERIC_VERDICT", None, "high"),
        ],
    )
    output = tmp_path / "resolver"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_semantic_resolver_pack",
            "--pack-root",
            str(pack),
            "--semantic-labels",
            str(semantic),
            "--strict-key",
            str(key),
            "--output-root",
            str(output),
            "--seed",
            "41",
        ],
    )
    main()
    rows = list(read_jsonl(output / "items.jsonl"))
    assert {row["norm_uid"] for row in rows} == {"u1", "u2", "u3", "u4"}
    assert all("decision" not in row and "metric_id" not in row for row in rows)
    report = json.loads((output / "validation.json").read_text())
    assert report["truth_hidden"] is True
    assert report["count"] == 4


def test_exact_disagreements_only_excludes_agreement_and_low_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    pack = tmp_path / "pack"
    items = [
        {"norm_uid": f"u{index}", "task": "humor", "source_group": f"g{index}"}
        for index in range(4)
    ]
    bank = {
        "task": "humor",
        "source_sha256": "bank-sha",
        "metrics": [{"metric_id": "m0"}, {"metric_id": "m1"}],
    }
    write_jsonl(pack / "items.jsonl", items)
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
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    write_jsonl(
        first,
        [
            _label("u0", "MATCH", "m0", "high"),
            _label("u1", "MATCH", "m0", "medium"),
            _label("u2", "MATCH", "m0", "high"),
            _label("u3", "NO_EXPLICIT_CRITERION", None, "high"),
        ],
    )
    write_jsonl(
        second,
        [
            _label("u0", "MATCH", "m0", "high"),
            _label("u1", "MATCH", "m0", "high"),
            _label("u2", "MATCH", "m1", "high"),
            _label("u3", "GENERIC_VERDICT", None, "high"),
        ],
    )
    output = tmp_path / "resolver"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_semantic_resolver_pack",
            "--pack-root",
            str(pack),
            "--semantic-labels",
            str(first),
            "--strict-key",
            str(second),
            "--output-root",
            str(output),
            "--seed",
            "43",
            "--selection-mode",
            "exact_disagreements_only",
        ],
    )
    main()
    rows = list(read_jsonl(output / "items.jsonl"))
    assert {row["norm_uid"] for row in rows} == {"u2", "u3"}
    report = json.loads((output / "validation.json").read_text())
    assert report["selection_rule"]["mode"] == "exact_disagreements_only"
    assert report["selection_rule"]["all_semantic_medium_or_low_any_decision"] is False
    assert (
        report["selection_rule"][
            "all_semantic_matches_without_exact_strict_corroboration"
        ]
        is False
    )
