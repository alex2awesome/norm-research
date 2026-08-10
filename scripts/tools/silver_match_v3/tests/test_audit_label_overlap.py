from scripts.tools.silver_match_v3.audit_label_overlap import audit
from scripts.tools.silver_match_v3.common import write_jsonl


def test_overlap_detects_fully_recycled_left_labels(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_jsonl(
        left,
        [
            {"norm_uid": "u1", "source_group": "g1"},
            {"norm_uid": "u2", "source_group": "g2"},
        ],
    )
    write_jsonl(
        right,
        [
            {"norm_uid": "u1", "source_group": "g1"},
            {"norm_uid": "u2", "source_group": "g2"},
            {"norm_uid": "u3", "source_group": "g3"},
        ],
    )
    report = audit(left, right)
    assert report["status"] == "LEFT_FULLY_RECYCLED_FROM_RIGHT"
    assert report["left_adds_new_exact_labels"] is False
    assert report["overlap"]["uids"] == 2


def test_overlap_detects_new_left_label(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_jsonl(left, [{"norm_uid": "new", "source_group": "new-group"}])
    write_jsonl(right, [{"norm_uid": "old", "source_group": "old-group"}])
    report = audit(left, right)
    assert report["status"] == "LEFT_ADDS_NEW_EXACT_LABELS"
    assert report["left_only"]["uids"] == 1
