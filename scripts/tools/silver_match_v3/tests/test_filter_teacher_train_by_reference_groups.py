import pytest

from scripts.tools.silver_match_v3.filter_teacher_train_by_reference_groups import (
    filter_rows,
)
from scripts.tools.silver_match_v3.train_nemotron_lora import source_group_key


def _norm(uid: str, source_id: str) -> dict:
    return {
        "norm_uid": uid,
        "task": "humor",
        "corpus": "jokes",
        "source_id": source_id,
        "norm": uid,
    }


def _teacher(uid: str, source_group: str | None = None) -> dict:
    row = {
        "norm_uid": uid,
        "task": "humor",
        "decision": "MATCH",
        "metric_id": "m1",
        "current_bank_source_sha256": "bank",
    }
    if source_group is not None:
        row["source_group"] = source_group
    return row


def test_recomputes_groups_and_excludes_every_row_in_reference_group():
    norms = {
        "direct": _norm("direct", "shared"),
        "sibling": _norm("sibling", "shared"),
        "keep": _norm("keep", "other"),
    }
    rows, audit = filter_rows(
        teacher_rows=[
            _teacher("direct"),
            _teacher("sibling", "stale-group"),
            _teacher("keep"),
        ],
        reference_inputs=[("dev", [{"norm_uid": "direct", "task": "humor"}])],
        norms=norms,
        task="humor",
        bank_hash="bank",
    )
    assert [row["norm_uid"] for row in rows] == ["keep"]
    assert rows[0]["source_group"] == source_group_key(norms["keep"])
    assert rows[0]["ce_source_disjoint_filter"] is True
    assert audit["excluded"]["reference_source_group"] == 2
    assert audit["output_reference_source_group_overlap"] == 0


def test_rejects_reference_or_teacher_uid_absent_from_manifest():
    norms = {"keep": _norm("keep", "other")}
    with pytest.raises(ValueError, match="UID absent"):
        filter_rows(
            teacher_rows=[_teacher("keep")],
            reference_inputs=[("dev", [{"norm_uid": "missing", "task": "humor"}])],
            norms=norms,
            task="humor",
            bank_hash="bank",
        )


def test_rejects_stale_bank_hash():
    norms = {"keep": _norm("keep", "other"), "dev": _norm("dev", "dev")}
    bad = _teacher("keep")
    bad["current_bank_source_sha256"] = "stale"
    with pytest.raises(ValueError, match="bank hash mismatch"):
        filter_rows(
            teacher_rows=[bad],
            reference_inputs=[("dev", [{"norm_uid": "dev", "task": "humor"}])],
            norms=norms,
            task="humor",
            bank_hash="bank",
        )


def test_rejects_duplicate_teacher_uid():
    norms = {"keep": _norm("keep", "other"), "dev": _norm("dev", "dev")}
    with pytest.raises(ValueError, match="duplicate UID"):
        filter_rows(
            teacher_rows=[_teacher("keep"), _teacher("keep")],
            reference_inputs=[("dev", [{"norm_uid": "dev", "task": "humor"}])],
            norms=norms,
            task="humor",
            bank_hash="bank",
        )
