from scripts.tools.silver_match_v3.build_lora_teacher_set import build_teacher_rows
from scripts.tools.silver_match_v3.train_nemotron_lora import source_group_key


def norm(uid: str, source_id: str) -> dict:
    return {
        "norm_uid": uid,
        "task": "peer-review",
        "corpus": "reviews",
        "source_id": source_id,
        "norm": uid,
    }


def label(uid: str, split: str, decision: str = "MATCH", metric_id: str = "a1") -> dict:
    return {
        "norm_uid": uid,
        "task": "peer-review",
        "corpus": "reviews",
        "split": split,
        "decision": decision,
        "metric_id": metric_id if decision == "MATCH" else None,
        "current_bank_source_sha256": "bank",
    }


def test_builder_excludes_entire_group_that_touches_frozen_split():
    norms = {
        "train-leak": norm("train-leak", "shared"),
        "dev": norm("dev", "shared"),
        "clean": norm("clean", "clean"),
        "abstain": norm("abstain", "other"),
    }
    rows, audit = build_teacher_rows(
        [
            (
                "labels.jsonl",
                [
                    label("train-leak", "train"),
                    label("dev", "dev"),
                    label("clean", "train"),
                    label("abstain", "train", "NO_EXPLICIT_CRITERION"),
                ],
            )
        ],
        norms,
        "peer-review",
        "bank",
    )
    assert [row["norm_uid"] for row in rows] == ["clean"]
    assert rows[0]["source_group"] == source_group_key(norms["clean"])
    assert audit["source_group_overlap_with_dev_test"] == 0
    assert audit["excluded"]["source_group_touches_dev_or_test"] == 1


def test_builder_rejects_stale_bank_hash():
    norms = {"x": norm("x", "x")}
    bad = label("x", "train")
    bad["current_bank_source_sha256"] = "stale"
    try:
        build_teacher_rows([("x", [bad])], norms, "peer-review", "bank")
    except ValueError as exc:
        assert "bank hash mismatch" in str(exc)
    else:
        raise AssertionError("stale teacher should fail")
