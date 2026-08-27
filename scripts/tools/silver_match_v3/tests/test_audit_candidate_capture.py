import json

import pytest

from scripts.tools.silver_match_v3.audit_candidate_capture import (
    CAPTURE_LANES,
    audit_candidate_capture,
)


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_capture_union_and_unique_margins(tmp_path):
    labels, candidates = tmp_path / "labels.jsonl", tmp_path / "candidates.jsonl"
    gold_rows, candidate_rows = [], []
    for i in range(3):
        uid, gold = f"u{i}", f"a{i}"
        gold_rows.append({"norm_uid": uid, "task": "t", "corpus": "c", "split": "test", "decision": "MATCH", "metric_id": gold})
        rows = []
        for j in range(8):
            row = {"metric_id": f"a{j}"}
            for lane_idx, lane in enumerate(CAPTURE_LANES):
                # Different lanes uniquely capture different gold rows at k=1.
                if lane_idx == i:
                    row[lane] = 1 if j == i else j + 2
                else:
                    row[lane] = 1 if j == 7 else j + 2
            rows.append(row)
        candidate_rows.append({"norm_uid": uid, "task": "t", "corpus": "c", "candidates": rows})
    _write(labels, gold_rows)
    _write(candidates, candidate_rows)
    report = audit_candidate_capture([labels], [candidates], k=1)
    assert report["overall"]["gold_matches"] == 3
    assert report["overall"]["union_capture_count"] == 3
    assert report["overall"]["union_miss_count"] == 0
    assert sum(report["overall"]["unique_marginal_capture_by_lane"].values()) == 3
    assert report["overall"]["unique_candidate_union_size"] == {
        "min": 2,
        "p50": 2,
        "p90": 2,
        "max": 2,
        "mean": 2.0,
    }
    assert report["groups"]["task_split:t:test"]["union_capture_rate"] == 1.0


def test_multiple_encoder_systems_are_union_captured(tmp_path):
    labels = tmp_path / "labels.jsonl"
    bge = tmp_path / "candidates_bge.jsonl"
    nemo = tmp_path / "candidates_nemotron_base.jsonl"
    _write(labels, [{"norm_uid": "u", "task": "t", "corpus": "c", "decision": "MATCH", "metric_id": "a1"}])
    base_rows = []
    nemo_rows = []
    for i in range(3):
        b = {"metric_id": f"a{i}"}
        n = {"metric_id": f"a{i}"}
        for lane in CAPTURE_LANES:
            b[lane] = 1 if i == 0 else i + 1
            n[lane] = 1 if i == 1 else i + 2
        base_rows.append(b)
        nemo_rows.append(n)
    _write(bge, [{"norm_uid": "u", "task": "t", "corpus": "c", "candidates": base_rows}])
    _write(nemo, [{"norm_uid": "u", "task": "t", "corpus": "c", "candidates": nemo_rows}])
    report = audit_candidate_capture([labels], [bge, nemo], k=1)
    assert report["overall"]["union_capture_rate"] == 1.0
    assert "bge:rank" in report["lanes"]
    assert "nemotron_base:rank" in report["lanes"]


def test_disjoint_shards_with_same_system_name_are_merged(tmp_path):
    labels = tmp_path / "labels.jsonl"
    shard1 = tmp_path / "foo.adapter.dev.jsonl"
    shard2 = tmp_path / "foo.adapter.test.jsonl"
    gold = [
        {"norm_uid": uid, "task": "t", "corpus": "c", "decision": "MATCH", "metric_id": "a1"}
        for uid in ("u1", "u2")
    ]
    _write(labels, gold)
    candidates = []
    for uid in ("u1", "u2"):
        rows = []
        for i in range(2):
            row = {"metric_id": f"a{i}"}
            for lane in CAPTURE_LANES:
                row[lane] = 1 if i == 1 else 2
            rows.append(row)
        candidates.append({"norm_uid": uid, "task": "t", "corpus": "c", "candidates": rows})
    _write(shard1, candidates[:1])
    _write(shard2, candidates[1:])
    report = audit_candidate_capture([labels], [shard1, shard2], k=1)
    assert report["overall"]["gold_matches"] == 2
    assert report["overall"]["union_capture_rate"] == 1.0


def test_prefix_lane_can_count_absent_gold_as_a_miss(tmp_path):
    labels = tmp_path / "labels.jsonl"
    candidates = tmp_path / "foo.adapter.jsonl"
    _write(
        labels,
        [
            {
                "norm_uid": "u",
                "task": "t",
                "corpus": "c",
                "decision": "MATCH",
                "metric_id": "gold",
            }
        ],
    )
    _write(
        candidates,
        [
            {
                "norm_uid": "u",
                "task": "t",
                "corpus": "c",
                "candidates": [{"metric_id": "other", "rank": 1}],
            },
            {
                "norm_uid": "ignored",
                "task": "t",
                "corpus": "c",
                "candidates": [{"metric_id": "gold", "rank": 1}],
            },
        ],
    )
    with pytest.raises(ValueError, match="gold metric absent"):
        audit_candidate_capture([labels], [candidates], k=1)
    report = audit_candidate_capture(
        [labels], [candidates], k=1, allow_prefix_missing_gold=True
    )
    assert report["allow_prefix_missing_gold"] is True
    assert report["overall"]["union_miss_count"] == 1
    rank = report["overall"]["candidate_rank_capture_by_system"]["adapter"]
    assert rank["candidate_depth"] == 1
    assert rank["gold_present_count"] == 0
    assert rank["gold_absent_count"] == 1
    assert rank["capture_at_depth"]["1"] == {"count": 0, "rate": 0.0}


def test_candidate_rank_curve_tracks_progressive_depth(tmp_path):
    labels = tmp_path / "labels.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write(
        labels,
        [
            {
                "norm_uid": f"u{i}",
                "task": "t",
                "corpus": "c",
                "decision": "MATCH",
                "metric_id": f"a{rank}",
            }
            for i, rank in enumerate((0, 4, 19, 49))
        ],
    )
    _write(
        candidates,
        [
            {
                "norm_uid": f"u{i}",
                "task": "t",
                "corpus": "c",
                "candidates": [
                    {
                        "metric_id": f"a{rank}",
                        "rank": rank + 1,
                        "component_lane_ranks": {
                            "left": {"rank": rank + 1},
                            "right": {"rank": 50 - rank},
                        },
                    }
                    for rank in range(50)
                ],
            }
            for i in range(4)
        ],
    )
    report = audit_candidate_capture([labels], [candidates], k=50)
    curve = report["overall"]["candidate_rank_capture_by_system"]["system0"]
    assert curve["candidate_depth"] == 50
    assert curve["gold_rank_quantiles_when_present"]["max"] == 50
    assert curve["capture_at_depth"]["1"]["count"] == 1
    assert curve["capture_at_depth"]["5"]["count"] == 2
    assert curve["capture_at_depth"]["20"]["count"] == 3
    assert curve["capture_at_depth"]["50"]["count"] == 4
    component = report["overall"]["component_union_capture_by_system"]["system0"]
    assert component["curve"]["1"]["unique_candidate_count"]["mean"] == 2.0
    assert component["curve"]["30"]["gold_capture_count"] == 4
