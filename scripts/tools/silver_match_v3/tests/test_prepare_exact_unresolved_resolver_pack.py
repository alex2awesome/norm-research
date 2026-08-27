import json
import sys
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.prepare_exact_unresolved_resolver_pack import main


def _build_source(root: Path) -> None:
    items = [
        {"norm_uid": f"u{i}", "task": "demo", "norm": f"norm {i}", "truth_hidden": True}
        for i in range(4)
    ]
    bank = {
        "task": "demo",
        "source_sha256": "bank-sha",
        "metrics": [{"metric_id": f"m{i}"} for i in range(3)],
    }
    write_jsonl(root / "items.jsonl", items)
    (root / "bank.json").write_text(json.dumps(bank) + "\n", encoding="utf-8")
    validation = {
        "task": "demo",
        "bank_source_sha256": "bank-sha",
        "outputs": {
            "items": {"sha256": sha256_file(root / "items.jsonl")},
            "bank": {"sha256": sha256_file(root / "bank.json")},
        },
    }
    (root / "validation.json").write_text(json.dumps(validation) + "\n", encoding="utf-8")


def test_builds_uid_exact_truth_hidden_pack(tmp_path: Path, monkeypatch) -> None:
    source, output = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    _build_source(source)
    unresolved = tmp_path / "unresolved.jsonl"
    write_jsonl(
        unresolved,
        [
            {
                "norm_uid": "u3",
                "task": "demo",
                "unresolved_reason": "three distinct exact votes",
                "source_predictions": {"A": {"metric_id": "m0"}},
            },
            {
                "norm_uid": "u1",
                "task": "demo",
                "unresolved_reason": "three distinct exact votes",
                "source_predictions": {"B": {"metric_id": "m1"}},
            },
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_exact_unresolved_resolver_pack",
            "--pack-root",
            str(source),
            "--unresolved",
            str(unresolved),
            "--output-root",
            str(output),
            "--seed",
            "17",
            "--chunk-size",
            "1",
        ],
    )
    main()
    rows = [json.loads(line) for line in (output / "items.jsonl").read_text().splitlines()]
    assert {row["norm_uid"] for row in rows} == {"u1", "u3"}
    assert all("source_predictions" not in row and "metric_id" not in row for row in rows)
    report = json.loads((output / "validation.json").read_text())
    assert report["count"] == 2
    assert report["chunk_count"] == 2
    assert report["prior_decisions_and_metric_ids_hidden"] is True


def test_rejects_uid_outside_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _build_source(source)
    unresolved = tmp_path / "unresolved.jsonl"
    write_jsonl(
        unresolved,
        [{"norm_uid": "outside", "task": "demo", "unresolved_reason": "x"}],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_exact_unresolved_resolver_pack",
            "--pack-root",
            str(source),
            "--unresolved",
            str(unresolved),
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "17",
        ],
    )
    with pytest.raises(ValueError, match="outside source pack"):
        main()
