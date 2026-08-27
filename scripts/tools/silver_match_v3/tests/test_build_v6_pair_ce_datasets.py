from __future__ import annotations

import json
from pathlib import Path

from scripts.tools.silver_match_v3.build_v6_pair_ce_datasets import (
    _adaptive_exposure_budgets,
    _build_task,
    _norm_uid,
    _source_group,
)
from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.train_nemotron_cross_encoder import (
    load_pair_examples,
)


def _write_pairs(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pair(
    *, split: str, left: str, right: str, score: int, cosine: float = 0.8
) -> dict[str, object]:
    return {
        "task": "code-review",
        "split": split,
        "key_a": left,
        "key_b": right,
        "canonical_a": f"Criterion {left}",
        "canonical_b": f"Criterion {right}",
        "score": score,
        "cos": cosine,
    }


def _bank() -> dict[str, object]:
    keys = {
        "code-review::raw::train_a::0": {"a0"},
        "code-review::raw::train_a2::0": {"a0"},
        "code-review::raw::train_b::0": {"a1"},
        "code-review::raw::train_c::0": {"a2"},
        "code-review::raw::dev_a::0": {"a0"},
        "code-review::raw::dev_a2::0": {"a0"},
        "code-review::raw::dev_b::0": {"a1"},
        "code-review::raw::dev_c::0": {"a2"},
    }
    return {
        "bank_source_sha256": "b" * 64,
        "key_to_ids": keys,
        "metrics": {
            f"a{i}": {"metric_id": f"a{i}", "metric_card": f"Metric {i}. Definition: d{i}"}
            for i in range(3)
        },
    }


def test_task_builder_emits_loader_compatible_three_way_disjoint_data(
    tmp_path: Path,
) -> None:
    pair_path = tmp_path / "pairs.jsonl"
    rows = [
        _pair(
            split="train",
            left="code-review::raw::train_a::0",
            right="code-review::raw::train_a2::0",
            score=2,
        ),
        _pair(
            split="train",
            left="code-review::raw::train_a::0",
            right="code-review::raw::train_b::0",
            score=1,
        ),
        _pair(
            split="train",
            left="code-review::raw::train_a::0",
            right="code-review::raw::train_c::0",
            score=0,
        ),
        _pair(
            split="eval",
            left="code-review::raw::dev_a::0",
            right="code-review::raw::dev_a2::0",
            score=2,
        ),
        _pair(
            split="eval",
            left="code-review::raw::dev_a::0",
            right="code-review::raw::dev_b::0",
            score=1,
        ),
        _pair(
            split="eval",
            left="code-review::raw::dev_a::0",
            right="code-review::raw::dev_c::0",
            score=0,
        ),
    ]
    _write_pairs(pair_path, rows)
    output = tmp_path / "stage"
    published = tmp_path / "published"
    output.mkdir()
    audit_records = {
        ("code-review", "train"): {
            "source_id": "pairs/code/train",
            "row_count": 3,
            "rejected_row_count": 0,
        },
        ("code-review", "eval"): {
            "source_id": "pairs/code/eval",
            "row_count": 3,
            "rejected_row_count": 0,
        },
    }
    eval_groups = {
        _source_group(str(row[field]))
        for row in rows
        if row["split"] == "eval"
        for field in ("key_a", "key_b")
    }
    result = _build_task(
        task="code-review",
        pair_path=pair_path,
        pair_hash=sha256_file(pair_path),
        audit_records=audit_records,
        bank=_bank(),
        eval_groups=eval_groups,
        output_dir=output,
        published_output_dir=published,
        norm_merge_audit={"excluded_sources": []},
    )
    assert result["class_counts"] == {
        "train": {"EXACT": 2, "FAMILY": 2, "REJECT": 2},
        "dev": {"EXACT": 2, "FAMILY": 2, "REJECT": 2},
    }
    train = load_pair_examples([output / "code-review" / "train.jsonl"])
    dev = load_pair_examples([output / "code-review" / "dev.jsonl"])
    assert {row.source_group for row in train}.isdisjoint(
        {row.source_group for row in dev}
    )
    assert {row.label for row in train} == {"EXACT", "FAMILY", "REJECT"}
    queue = json.loads((output / "code-review" / "TRAIN_QUEUE.json").read_text())
    assert queue["task_local_lora_only"] is True
    assert queue["cross_task_pooling"] is False
    assert str(published / "code-review" / "train.jsonl") in queue["commands"][0]


def test_train_pair_touching_dev_source_group_is_quarantined(tmp_path: Path) -> None:
    pair_path = tmp_path / "pairs.jsonl"
    rows = [
        _pair(
            split="train",
            left="code-review::raw::train_a::0",
            right="code-review::raw::train_a2::0",
            score=2,
        ),
        _pair(
            split="train",
            left="code-review::raw::train_a::0",
            right="code-review::raw::train_b::0",
            score=1,
        ),
        _pair(
            split="train",
            left="code-review::raw::train_a::0",
            right="code-review::raw::train_c::0",
            score=0,
        ),
        _pair(
            split="train",
            left="code-review::raw::dev_a::0",
            right="code-review::raw::train_b::0",
            score=1,
        ),
        _pair(
            split="eval",
            left="code-review::raw::dev_a::0",
            right="code-review::raw::dev_a2::0",
            score=2,
        ),
        _pair(
            split="eval",
            left="code-review::raw::dev_a::0",
            right="code-review::raw::dev_b::0",
            score=1,
        ),
        _pair(
            split="eval",
            left="code-review::raw::dev_a::0",
            right="code-review::raw::dev_c::0",
            score=0,
        ),
    ]
    _write_pairs(pair_path, rows)
    output = tmp_path / "stage"
    output.mkdir()
    audit_records = {
        ("code-review", "train"): {
            "source_id": "train",
            "row_count": 4,
            "rejected_row_count": 0,
        },
        ("code-review", "eval"): {
            "source_id": "dev",
            "row_count": 3,
            "rejected_row_count": 0,
        },
    }
    _build_task(
        task="code-review",
        pair_path=pair_path,
        pair_hash=sha256_file(pair_path),
        audit_records=audit_records,
        bank=_bank(),
        eval_groups={"code-review::raw::dev_a"},
        output_dir=output,
        published_output_dir=tmp_path / "published",
        norm_merge_audit={"excluded_sources": []},
    )
    report = json.loads((output / "code-review" / "report.json").read_text())
    assert report["quarantine_reason_counts"]["train_pair_touches_dev_source_group"] == 1
    assert report["source_coverage"]["source_group_overlap"] == 0


def test_adaptive_budgets_limit_repeated_exposure() -> None:
    assert _adaptive_exposure_budgets(668) == [16704, 33400, 66800]
    assert _adaptive_exposure_budgets(5000) == [125000, 250000, 400000]
    assert _adaptive_exposure_budgets(1) == [10000]


def test_norm_uid_is_task_and_key_stable() -> None:
    key = "code-review::raw::doc::0"
    assert _norm_uid("code-review", key) == _norm_uid("code-review", key)
    assert _norm_uid("code-review", key) != _norm_uid("peer-review", key)
