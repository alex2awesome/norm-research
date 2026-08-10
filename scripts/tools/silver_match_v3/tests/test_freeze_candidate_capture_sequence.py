import json

import pytest

from scripts.tools.silver_match_v3.freeze_candidate_capture_sequence import (
    evaluate_sequence,
    select_sequence,
)


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path):
    labels = tmp_path / "labels.jsonl"
    system_a = tmp_path / "a.jsonl"
    system_b = tmp_path / "b.jsonl"
    label_rows = []
    candidate_a, candidate_b = [], []
    for split, count in (("dev", 100), ("test", 80)):
        for index in range(count):
            uid = f"{split}-{index}"
            gold = f"m{index % 6}"
            label_rows.append({
                "norm_uid": uid,
                "task": "humor",
                "corpus": "c",
                "split": split,
                "decision": "MATCH",
                "metric_id": gold,
            })
            rows_a, rows_b = [], []
            for metric_index in range(6):
                row_a = {"metric_id": f"m{metric_index}"}
                row_b = {"metric_id": f"m{metric_index}"}
                for lane_index, lane in enumerate((
                    "rank", "dense_statement_rank", "char_rank", "dense_rank",
                    "word_rank", "char_statement_rank", "word_statement_rank",
                )):
                    # a:rank captures the first 80%; b:rank supplies the rest.
                    a_hit = index < int(count * 0.8) and lane_index == 0
                    b_hit = index >= int(count * 0.8) and lane_index == 0
                    row_a[lane] = 1 if a_hit and metric_index == index % 6 else metric_index + 2
                    row_b[lane] = 1 if b_hit and metric_index == index % 6 else metric_index + 2
                rows_a.append(row_a)
                rows_b.append(row_b)
            candidate_a.append({"norm_uid": uid, "task": "humor", "candidates": rows_a})
            candidate_b.append({"norm_uid": uid, "task": "humor", "candidates": rows_b})
    _write(labels, label_rows)
    _write(system_a, candidate_a)
    _write(system_b, candidate_b)
    return labels, {"a": system_a, "b": system_b}


def test_dev_only_selection_then_fixed_test_evaluation(tmp_path):
    labels, candidates = _fixture(tmp_path)
    selection = select_sequence([labels], candidates, k=1, target=0.05)
    assert selection["selection_split"] == "dev"
    assert selection["test_labels_used_for_selection"] is False
    assert selection["selected_sequence"] == ["a:rank", "b:rank"]
    assert selection["selection_result"]["under_target_supported"] is True

    evaluation = evaluate_sequence(selection, split="test")
    assert evaluation["selected_sequence"] == ["a:rank", "b:rank"]
    assert evaluation["test_selection_performed"] is False
    assert evaluation["evaluation_result"]["union_capture_rate"] == 1.0


def test_hash_mutation_fails_closed(tmp_path):
    labels, candidates = _fixture(tmp_path)
    selection = select_sequence([labels], candidates, k=1)
    labels.write_text(labels.read_text() + "\n")
    with pytest.raises(ValueError, match="label hash mismatch"):
        evaluate_sequence(selection, split="test")


def test_selection_and_evaluation_splits_must_differ(tmp_path):
    labels, candidates = _fixture(tmp_path)
    selection = select_sequence([labels], candidates, k=1)
    with pytest.raises(ValueError, match="must differ"):
        evaluate_sequence(selection, split="dev")
