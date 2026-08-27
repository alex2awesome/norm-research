from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.finalize_exact_multi_pass_truth import (
    _validate_role_freeze,
    main,
)


def _make_pack(root: Path, items: list[dict], bank: dict) -> None:
    write_jsonl(root / "items.jsonl", items)
    root.mkdir(parents=True, exist_ok=True)
    (root / "bank.json").write_text(json.dumps(bank))
    (root / "validation.json").write_text(
        json.dumps(
            {
                "task": "humor",
                "bank_source_sha256": "bank-sha",
                "truth_hidden": True,
                "outputs": {
                    "items": {"sha256": sha256_file(root / "items.jsonl")},
                    "bank": {"sha256": sha256_file(root / "bank.json")},
                },
            }
        )
    )


def _label(uid: str, decision: str, metric_id: str | None) -> dict:
    return {
        "norm_uid": uid,
        "task": "humor",
        "decision": decision,
        "metric_id": metric_id,
        "confidence": "high",
        "reason": "independent judgment",
        "current_bank_source_sha256": "bank-sha",
    }


def test_legacy_audit_pack_can_be_bound_to_clean_role_freeze(tmp_path: Path) -> None:
    freeze_path = tmp_path / "FREEZE.json"
    validation = {"task": "press-releases", "count": 2, "input_hashes": {"items": "ids-sha"}}
    freeze = {
        "schema_version": "silver-match-v3-clean-gepa-panel-freeze-v1",
        "status": "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
        "task": "press-releases",
        "role": "optimize",
        "required_upstream_split": "train",
        "selected_count": 2,
        "outputs": {"identities": {"sha256": "ids-sha"}},
        "content_contract": {
            "selection_uses_identity_and_source_group_only": True,
            "downstream_outcomes_read": False,
            "metric_ids_read": False,
            "model_prediction_fields_read": False,
            "truth_fields_read": False,
        },
    }
    freeze_path.write_text(json.dumps(freeze))
    result = _validate_role_freeze(freeze_path, validation, "optimize")
    assert result["identity_sha256"] == "ids-sha"
    assert result["selected_count"] == 2


def test_modern_panel_pack_can_be_bound_to_named_role_freeze(tmp_path: Path) -> None:
    identities_sha = "c" * 64
    freeze_path = tmp_path / "FREEZE.json"
    freeze = {
        "schema_version": "silver-match-v3-clean-gepa-panel-freeze-v1",
        "status": "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
        "task": "humor",
        "role": "adjudicator_dev",
        "required_upstream_split": "train",
        "selected_count": 300,
        "outputs": {"identities": {"sha256": identities_sha}},
        "content_contract": {
            "selection_uses_identity_and_source_group_only": True,
            "downstream_outcomes_read": False,
            "metric_ids_read": False,
            "model_prediction_fields_read": False,
            "truth_fields_read": False,
        },
    }
    freeze_path.write_text(json.dumps(freeze))
    validation = {
        "task": "humor",
        "count": 300,
        "inputs": {
            "identities": {"sha256": identities_sha},
            "identity_freeze": {"sha256": sha256_file(freeze_path)},
        },
    }
    result = _validate_role_freeze(
        freeze_path, validation, "adjudicator_dev"
    )
    assert result["role"] == "adjudicator_dev"
    assert result["identity_sha256"] == identities_sha


def test_sequential_resolvers_cover_only_current_disagreements(
    tmp_path: Path, monkeypatch
) -> None:
    items = [
        {
            "norm_uid": f"u{i}",
            "task": "humor",
            "corpus": "c",
            "row": i,
            "source_group": f"g{i}",
        }
        for i in range(4)
    ]
    bank = {
        "task": "humor",
        "source_sha256": "bank-sha",
        "metrics": [{"metric_id": "m0"}, {"metric_id": "m1"}],
    }
    full_a, full_b = tmp_path / "pack-a", tmp_path / "pack-b"
    resolver_c, resolver_d = tmp_path / "pack-c", tmp_path / "pack-d"
    _make_pack(full_a, items, bank)
    _make_pack(full_b, list(reversed(items)), {**bank, "metrics": list(reversed(bank["metrics"]))})
    _make_pack(resolver_c, items[1:], bank)
    _make_pack(resolver_d, items[2:], {**bank, "metrics": list(reversed(bank["metrics"]))})
    labels = {
        "a": [
            _label("u0", "MATCH", "m0"),
            _label("u1", "MATCH", "m0"),
            _label("u2", "MATCH", "m0"),
            _label("u3", "NO_EXPLICIT_CRITERION", None),
        ],
        "b": [
            _label("u0", "MATCH", "m0"),
            _label("u1", "MATCH", "m1"),
            _label("u2", "MATCH", "m1"),
            _label("u3", "GENERIC_VERDICT", None),
        ],
        "c": [
            _label("u1", "MATCH", "m0"),
            _label("u2", "NO_CANDIDATE_FITS", None),
            _label("u3", "MATCH", "m0"),
        ],
        "d": [
            _label("u2", "MATCH", "m1"),
            _label("u3", "MATCH", "m1"),
        ],
    }
    paths = {}
    for name, rows in labels.items():
        path = tmp_path / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    output, unresolved = tmp_path / "truth.jsonl", tmp_path / "unresolved.jsonl"
    disagreements, report = tmp_path / "disagreements.jsonl", tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_exact_multi_pass_truth",
            "--pack-root",
            str(full_a),
            *sum(
                (
                    ["--label-pass", f"{name}={paths[name]}"]
                    for name in ("a", "b", "c", "d")
                ),
                [],
            ),
            *sum(
                (
                    ["--pass-pack", f"{name}={pack}"]
                    for name, pack in (
                        ("a", full_a),
                        ("b", full_b),
                        ("c", resolver_c),
                        ("d", resolver_d),
                    )
                ),
                [],
            ),
            "--output",
            str(output),
            "--unresolved-output",
            str(unresolved),
            "--disagreements-output",
            str(disagreements),
            "--report",
            str(report),
        ],
    )
    main()
    resolved = {row["norm_uid"]: row for row in read_jsonl(output)}
    assert set(resolved) == {"u0", "u1", "u2"}
    assert resolved["u1"]["metric_id"] == "m0"
    assert resolved["u2"]["metric_id"] == "m1"
    assert [row["norm_uid"] for row in read_jsonl(unresolved)] == ["u3"]
    summary = json.loads(report.read_text())
    assert summary["rounds"][2]["labeled_count"] == 3
    assert summary["rounds"][3]["labeled_count"] == 2
    assert summary["resolved_count"] == 3
    assert summary["unresolved_count"] == 1
