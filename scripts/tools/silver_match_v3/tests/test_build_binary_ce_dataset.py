import argparse
import json

from scripts.tools.silver_match_v3.build_binary_ce_dataset import build


def _row(uid, metric, relation, group, *, lane=None, rank=None, split="train"):
    provenance = []
    lanes = []
    if lane:
        lanes.append(lane)
        item = {"lane": lane, "artifact_sha256": "a" * 64}
        if rank is not None:
            item["rank"] = rank
        provenance.append(item)
    return {
        "norm_uid": uid,
        "metric_id": metric,
        "source_group": group,
        "query": f"query {uid}",
        "metric_card": f"card {metric}",
        "relation": relation,
        "split": split,
        "candidate_lanes": lanes,
        "candidate_provenance": provenance,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_builder_keeps_all_positives_and_exact_provenance_quotas(tmp_path):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    _write(
        train,
        [
            _row("p", "m1", "EXACT", "train:p"),
            _row("p", "m2", "EXACT", "train:p"),
            _row("f1", "m3", "FAMILY", "train:f1"),
            _row("f2", "m4", "FAMILY", "train:f2"),
            _row("h1", "m5", "REJECT", "train:h1", lane="retrieval", rank=2),
            _row("h2", "m6", "REJECT", "train:h2", lane="retrieval", rank=9),
            _row(
                "e1",
                "m7",
                "REJECT",
                "train:e1",
                lane="global_balanced_negative",
            ),
            _row(
                "e2",
                "m8",
                "REJECT",
                "train:e2",
                lane="global_balanced_negative",
            ),
        ],
    )
    _write(dev, [_row("d", "m1", "REJECT", "dev:d", split="dev")])
    _write(test, [_row("t", "m1", "REJECT", "test:t", split="test")])
    args = argparse.Namespace(
        train=str(train),
        dev=str(dev),
        test=str(test),
        output=str(tmp_path / "out"),
        seed=7,
        hard_family=1,
        hard_retrieval=1,
        easy_global=1,
    )
    first = build(args)
    assert first["output_counts"]["train_binary"] == {"1": 2, "0": 3}
    assert first["split_audit"]["source_group_overlap_count"] == 0
    output_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "binary.train.pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(row["binary_label"] == 1 for row in output_rows) == 2
    assert {
        row["binary_negative_type"]
        for row in output_rows
        if row["binary_label"] == 0
    } == {"hard_family", "hard_retrieval_top10", "easy_global"}
