import json
from pathlib import Path

from scripts.tools.silver_match_v3.build_retriever_teacher_set import build_teacher_set


def dump_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_blocks_frozen_groups_and_human_overrides(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"metrics": [{"metric_id": "a0"}, {"metric_id": "a1"}]}))
    norms = tmp_path / "norms.jsonl"
    rows = [
        {"norm_uid": "train", "task": "humor", "corpus": "c", "row": 0, "source_id": "s0", "norm": "n0"},
        {"norm_uid": "same-source", "task": "humor", "corpus": "c", "row": 1, "source_id": "frozen", "norm": "n1"},
        {"norm_uid": "test", "task": "humor", "corpus": "c", "row": 2, "source_id": "frozen", "norm": "n2"},
        {"norm_uid": "dev", "task": "humor", "corpus": "c", "row": 3, "source_id": "dev", "norm": "n3"},
    ]
    dump_jsonl(norms, rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "banks": {"humor": {"path": str(bank), "source_sha256": "bankhash"}},
        "corpora": {"c": {"task": "humor", "path": str(norms)}},
    }))
    trusted = tmp_path / "trusted.jsonl"
    dump_jsonl(trusted, [
        {"norm_uid": "train", "task": "humor", "decision": "MATCH", "metric_id": "a0", "current_bank_source_sha256": "bankhash"},
        {"norm_uid": "same-source", "task": "humor", "decision": "MATCH", "metric_id": "a0", "current_bank_source_sha256": "bankhash"},
    ])
    human = tmp_path / "human.jsonl"
    dump_jsonl(human, [
        {"norm_uid": "train", "task": "humor", "decision": "MATCH", "metric_id": "a1", "current_bank_source_sha256": "bankhash", "split": "train"},
        {"norm_uid": "test", "task": "humor", "decision": "MATCH", "metric_id": "a0", "current_bank_source_sha256": "bankhash", "split": "test"},
        {"norm_uid": "dev", "task": "humor", "decision": "NO_EXPLICIT_CRITERION", "metric_id": None, "current_bank_source_sha256": "bankhash", "split": "dev"},
    ])
    teachers, external, report = build_teacher_set(
        manifest_path=manifest,
        task="humor",
        trusted_paths=[trusted],
        human_paths=[human],
    )
    assert [(row["norm_uid"], row["metric_id"]) for row in teachers] == [("train", "a1")]
    assert {row["norm_uid"] for row in external} == {"dev", "test"}
    assert report["selection_counts"]["trusted_frozen_source_blocked"] == 1
    assert report["selection_counts"]["human_overrode_trusted_conflict"] == 1
    assert report["source_group_overlap"] == {"teacher_external": 0, "dev_test": 0}


def test_forced_top3_uses_rank_one_and_preserves_alternatives(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"metrics": [{"metric_id": f"a{i}"} for i in range(3)]}))
    norms = tmp_path / "norms.jsonl"
    dump_jsonl(norms, [
        {"norm_uid": "forced", "task": "humor", "corpus": "c", "row": 0,
         "source_id": "s0", "norm": "n0"},
        {"norm_uid": "dev", "task": "humor", "corpus": "c", "row": 1,
         "source_id": "s1", "norm": "n1"},
    ])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "banks": {"humor": {"path": str(bank), "source_sha256": "bankhash"}},
        "corpora": {"c": {"task": "humor", "path": str(norms)}},
    }))
    trusted = tmp_path / "trusted.jsonl"
    dump_jsonl(trusted, [
        {"norm_uid": "forced", "task": "humor", "decision": "MATCH",
         "metric_id": f"a{i - 1}", "forced_rank": i,
         "label_source": "sonnet_forced_top3",
         "supervision_strength": "weak_forced_positive",
         "current_bank_source_sha256": "bankhash"}
        for i in (1, 2, 3)
    ])
    human = tmp_path / "human.jsonl"
    dump_jsonl(human, [
        {"norm_uid": "dev", "task": "humor", "decision": "MATCH",
         "metric_id": "a0", "current_bank_source_sha256": "bankhash", "split": "dev"}
    ])
    teachers, _, report = build_teacher_set(
        manifest_path=manifest,
        task="humor",
        trusted_paths=[trusted],
        human_paths=[human],
    )
    assert len(teachers) == 1
    assert teachers[0]["metric_id"] == "a0"
    assert teachers[0]["acceptable_metric_ids"] == ["a0", "a1", "a2"]
    assert teachers[0]["forced_group_rows"] == 3
    assert report["selection_counts"]["trusted_weak_forced_alternative_rows_merged"] == 2
