import json

from scripts.tools.silver_match_v3.build_gemma_consensus_teachers import build_consensus
from scripts.tools.silver_match_v3.make_calibration import split_for, split_group_for


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_consensus_requires_same_metric_and_confidence(tmp_path):
    # Find deterministic source IDs on the train split.
    source_ids = []
    index = 0
    while len(source_ids) < 3:
        norm = {
            "norm_uid": f"u{index}",
            "task": "peer-review",
            "corpus": "reviews",
            "source_id": f"s{index}",
            "row": index,
        }
        if split_for(split_group_for(norm)) == "train":
            source_ids.append(norm)
        index += 1
    norms = tmp_path / "norms.jsonl"
    write_jsonl(norms, source_ids)
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "source_sha256": "bank-sha",
                "metrics": [{"metric_id": "a1"}, {"metric_id": "a2"}],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "v3",
                "banks": {
                    "peer-review": {
                        "path": str(bank),
                        "source_sha256": "bank-sha",
                    }
                },
                "corpora": {
                    "reviews": {"path": str(norms), "task": "peer-review"}
                },
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(
        candidates,
        [
            {
                "norm_uid": row["norm_uid"],
                "task": "peer-review",
                "bank_source_sha256": "bank-sha",
                "candidates": [{"metric_id": "a1"}, {"metric_id": "a2"}],
            }
            for row in source_ids
        ],
    )

    def prediction(row, order, metric, confidence):
        return {
            "norm_uid": row["norm_uid"],
            "task": "peer-review",
            "decision": "MATCH",
            "metric_id": metric,
            "confidence": confidence,
            "reason": "reason",
            "model": "gemma",
            "prompt_sha256": "prompt",
            "order_mode": order,
            "candidate_bank_source_sha256": "bank-sha",
        }

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(
        first,
        [
            prediction(source_ids[0], "original", "a1", "high"),
            prediction(source_ids[1], "original", "a1", "high"),
            prediction(source_ids[2], "original", "a1", "medium"),
        ],
    )
    write_jsonl(
        second,
        [
            prediction(source_ids[0], "hashed", "a1", "high"),
            prediction(source_ids[1], "hashed", "a2", "high"),
            prediction(source_ids[2], "hashed", "a1", "high"),
        ],
    )
    rows, report = build_consensus(
        manifest_path=manifest,
        candidates_path=candidates,
        first_path=first,
        second_path=second,
        task="peer-review",
        human_panel_paths=[],
        min_confidence="high",
    )
    assert [row["norm_uid"] for row in rows] == [source_ids[0]["norm_uid"]]
    assert report["counts"]["excluded_metric_disagreement"] == 1
    assert report["counts"]["excluded_low_confidence"] == 1
