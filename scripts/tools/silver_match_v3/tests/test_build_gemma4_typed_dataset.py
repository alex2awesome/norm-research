import argparse
import hashlib
import json

import pytest

from scripts.tools.silver_match_v3.build_gemma4_typed_dataset import (
    _target_for_slate,
    build,
    structured_target,
)
from scripts.tools.silver_match_v3.build_nemotron_ce_pairs import _load_truth


def test_unanchored_family_only_is_allowed_only_for_generative_adjudicator(tmp_path):
    truth = tmp_path / "truth.jsonl"
    truth.write_text(json.dumps({
        "norm_uid": "u", "task": "humor", "corpus": "c",
        "source_group": "g", "split": "train",
        "decision": "MATCH_FAMILY_ONLY", "metric_id": None,
        "current_bank_source_sha256": "bank",
    }) + "\n")
    kwargs = {
        "task": "humor", "bank_hash": "bank", "bank_ids": {"m0"},
        "split_assignments": {},
    }
    with pytest.raises(ValueError, match="lacks family anchors"):
        _load_truth([truth], **kwargs)
    rows, indexed, _ = _load_truth(
        [truth], allow_unanchored_family_only=True, **kwargs
    )
    assert len(rows) == 1 and indexed["u"]["decision"] == "MATCH_FAMILY_ONLY"
    relation, decision, metric = _target_for_slate(indexed["u"], ["m0"], {})
    assert (relation, decision, metric) == ("FAMILY", "MATCH_FAMILY_ONLY", None)


def test_structured_target_spans_exactly_cover_named_fields():
    rendered, spans = structured_target(
        decision="MATCH", metric_id="m0", confidence="high", reason="why"
    )
    assert json.loads(rendered) == {
        "decision": "MATCH", "metric_id": "m0", "confidence": "high", "reason": "why"
    }
    for name, span in spans.items():
        segment = rendered[span["start"]:span["end"]]
        assert segment.startswith(json.dumps(name) + ":")


def test_full_builder_uses_two_train_views_one_dev_view_and_no_dev_injection(tmp_path):
    bank_hash = hashlib.sha256(b"bank").hexdigest()
    bank = {
        "task": "humor", "source_sha256": bank_hash,
        "metrics": [{
            "task": "humor", "metric_id": f"m{i}", "name": f"metric {i}",
            "description": f"definition {i}", "examples": [],
        } for i in range(3)],
    }
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank) + "\n")
    norms = tmp_path / "norms.jsonl"
    norms.write_text("".join(json.dumps({
        "task": "humor", "corpus": "jokes", "norm_uid": uid,
        "source_id": source, "row": i, "norm": f"statement {uid}", "context": "evidence",
    }) + "\n" for i, (uid, source) in enumerate((("train", "s-train"), ("dev", "s-dev")))))
    groups = {
        "train": "humor\x1fjokes\x1fsource\x1fs-train",
        "dev": "humor\x1fjokes\x1fsource\x1fs-dev",
    }
    truth = tmp_path / "truth.jsonl"
    truth.write_text("".join(json.dumps(row) + "\n" for row in ({
        "task": "humor", "corpus": "jokes", "norm_uid": "train",
        "source_group": groups["train"], "split": "train", "decision": "MATCH",
        "metric_id": "m0", "confidence": "high", "reason": "exact",
        "current_bank_source_sha256": bank_hash,
    }, {
        "task": "humor", "corpus": "jokes", "norm_uid": "dev",
        "source_group": groups["dev"], "split": "dev", "decision": "MATCH_FAMILY_ONLY",
        "metric_id": None, "confidence": "medium", "reason": "ambiguous family",
        "current_bank_source_sha256": bank_hash,
    })))
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps({
        "task": "humor", "norm_uid": uid, "bank_source_sha256": bank_hash,
        "candidates": [{"metric_id": metric, "rank": rank} for rank, metric in enumerate(ids, 1)],
    }) + "\n" for uid, ids in (("train", ("m1", "m2", "m0")), ("dev", ("m2", "m1", "m0")))))
    hierarchy = tmp_path / "hierarchy.json"
    hierarchy.write_text(json.dumps({
        "task": "humor", "n_r2_clusters_in": 3, "n_merged_groups": 1,
        "merged_groups": [{"metric_ids": ["m0", "m1", "m2"]}],
    }) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "banks": {"humor": {"path": str(bank_path), "source_sha256": bank_hash, "count": 3}},
        "corpora": {"jokes": {"task": "humor", "path": str(norms), "count": 2}},
    }) + "\n")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Choose the exact metric or abstain.\n")
    args = argparse.Namespace(
        manifest=str(manifest), bank=str(bank_path), hierarchy=str(hierarchy),
        prompt=str(prompt), truth=[str(truth)], split_assignments=None,
        candidates=[f"ce={candidates}"], task="humor", max_candidates=3,
        order_seed=7, context_chars=100, description_chars=100,
        example_chars=100, max_examples=1,
    )
    buckets, report = build(args)
    assert len(buckets["train"]) == 2
    assert len(buckets["dev"]) == 1
    assert not buckets["test"] and not buckets["blind"]
    assert buckets["dev"][0]["decision"] == "MATCH_FAMILY_ONLY"
    assert buckets["dev"][0]["gradient_eligible"] is False
    assert report["candidate_injections_outside_train"] == 0
    assert report["source_groups_crossing_splits"] == 0
