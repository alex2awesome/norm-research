from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.partition_independent_pack import main as partition
from scripts.tools.silver_match_v3.subset_labels_to_pack import main as subset_labels


def _make_pack(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    root = tmp_path / "source"
    items = [
        {
            "schema_version": "silver-match-v3.0",
            "norm_uid": f"{index:064x}",
            "task": "math-stackexchange",
            "corpus": "math_se",
            "row": index,
            "split": "train",
            "split_group": f"post:{index}",
            "norm": f"norm {index}",
        }
        for index in range(6)
    ]
    bank = {
        "task": "math-stackexchange",
        "source_sha256": "bank-sha",
        "metrics": [{"metric_id": "m0", "name": "Metric"}],
    }
    items_path, bank_path = root / "items.jsonl", root / "bank.json"
    write_jsonl(items_path, items)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(bank))
    (root / "validation.json").write_text(
        json.dumps(
            {
                "task": "math-stackexchange",
                "bank_source_sha256": "bank-sha",
                "outputs": {
                    "items": {"sha256": sha256_file(items_path)},
                    "bank": {"sha256": sha256_file(bank_path)},
                },
            }
        )
    )
    reference = tmp_path / "roles.jsonl"
    write_jsonl(
        reference,
        [
            {
                **row,
                "predeclared_split": "train",
                "split": "train" if index < 4 else "dev",
            }
            for index, row in enumerate(items)
        ],
    )
    return root, reference, items


def test_partition_and_label_projection_are_disjoint_and_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    source, reference, items = _make_pack(tmp_path)
    output = tmp_path / "partitioned"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "partition_independent_pack",
            "--pack-root",
            str(source),
            "--reference",
            str(reference),
            "--role-map",
            "train=training",
            "--role-map",
            "dev=blind_audit",
            "--output-root",
            str(output),
            "--chunk-size",
            "3",
        ],
    )
    partition()
    training = list(read_jsonl(output / "training" / "items.jsonl"))
    audit = list(read_jsonl(output / "blind_audit" / "items.jsonl"))
    assert len(training) == 4
    assert len(audit) == 2
    assert not ({row["norm_uid"] for row in training} & {row["norm_uid"] for row in audit})
    assert all(row["canonical_predeclared_split"] == "train" for row in audit)
    assert json.loads((output / "blind_audit" / "validation.json").read_text())[
        "permanently_excluded_from_gradients"
    ]

    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(
        labels_path,
        [
            {
                **{key: row[key] for key in ("norm_uid", "task", "corpus", "row", "split_group")},
                "decision": "MATCH",
                "metric_id": "m0",
                "confidence": "high",
                "current_bank_source_sha256": "bank-sha",
            }
            for row in items
        ],
    )
    audit_labels = tmp_path / "audit.labels.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subset_labels_to_pack",
            "--labels",
            str(labels_path),
            "--pack-root",
            str(output / "blind_audit"),
            "--output",
            str(audit_labels),
        ],
    )
    subset_labels()
    projected = list(read_jsonl(audit_labels))
    assert len(projected) == 2
    assert all(row["training_eligible"] is False for row in projected)
    assert all(row["audit_permanently_excluded_from_gradients"] for row in projected)


def test_partition_rejects_incomplete_reference(tmp_path: Path, monkeypatch) -> None:
    source, reference, _ = _make_pack(tmp_path)
    rows = list(read_jsonl(reference))[:-1]
    incomplete = tmp_path / "incomplete.jsonl"
    write_jsonl(incomplete, rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "partition_independent_pack",
            "--pack-root",
            str(source),
            "--reference",
            str(incomplete),
            "--role-map",
            "train=training",
            "--role-map",
            "dev=blind_audit",
            "--output-root",
            str(tmp_path / "bad"),
        ],
    )
    with pytest.raises(ValueError, match="cover the source pack exactly"):
        partition()
