import json

import pytest

from scripts.tools.silver_match_v3.adjudicate_gemma import ordered_candidates
from scripts.tools.silver_match_v3.run_paired_gemma_lora_batch import (
    EXPECTED_MODEL_CONTENT_SHA256,
    SCHEMA,
    infer_manifest_task,
    paired_rows_for_batch,
    validate_model_inventory,
)
from scripts.tools.silver_match_v3.score_humor_gemma_lora_select import (
    clopper_pearson,
    paired_exact_change,
    score_rows,
)


def test_paired_gemma_infers_non_humor_task_from_manifest_routing():
    rows = [
        {"norm_uid": "u1", "task": "legal", "corpus": "court_a"},
        {"norm_uid": "u2", "task": "legal", "corpus": "court_b"},
    ]
    manifest = {
        "corpora": {
            "court_a": {"task": "legal"},
            "court_b": {"task": "legal"},
        }
    }
    assert infer_manifest_task(rows, manifest) == "legal"


def test_paired_gemma_rejects_cross_task_candidate_input():
    rows = [
        {"norm_uid": "u1", "task": "legal", "corpus": "court"},
        {"norm_uid": "u2", "task": "humor", "corpus": "jokes"},
    ]
    manifest = {
        "corpora": {"court": {"task": "legal"}, "jokes": {"task": "humor"}}
    }
    with pytest.raises(ValueError, match="exactly one manifest task"):
        infer_manifest_task(rows, manifest)


def _truth(uid, decision, metric_id=None):
    return {
        "norm_uid": uid,
        "task": "humor",
        "corpus": "humor_multi",
        "row": int(uid[1:]),
        "decision": decision,
        "metric_id": metric_id,
        "gepa_role": "select",
        "evaluation_only": True,
        "training_eligible": False,
    }


def _candidate(uid, ids):
    return {
        "norm_uid": uid,
        "corpus": "humor_multi",
        "bank_source_sha256": "bank",
        "candidates": [{"metric_id": value} for value in ids],
    }


def _prediction(decision, metric_id=None):
    return {
        "decision": decision,
        "metric_id": metric_id,
        "confidence": "high",
        "reason": "synthetic",
    }


def _paired_row(candidate, order, base, lora):
    uid = candidate["norm_uid"]
    ids = [
        row["metric_id"]
        for row in ordered_candidates(candidate["candidates"], order, uid)
    ]
    return {
        "schema_version": SCHEMA,
        "norm_uid": uid,
        "order_mode": order,
        "candidate_ids": ids,
        "candidate_bank_source_sha256": "bank",
        "prompt_sha256": "prompt",
        "base_item_prompt_sha256": f"item-{order}-{uid}",
        "lora_item_prompt_sha256": f"item-{order}-{uid}",
        "base": base,
        "lora": lora,
    }


def _fixture():
    truth = [
        _truth("u1", "MATCH", "m1"),
        _truth("u2", "MATCH_FAMILY_ONLY"),
        _truth("u3", "NO_CANDIDATE_FITS"),
        _truth("u4", "MATCH", "m2"),
    ]
    candidates = [
        _candidate("u1", ["m1", "x1", "x2"]),
        _candidate("u2", ["x1", "x2", "x3"]),
        _candidate("u3", ["x3", "x2", "x1"]),
        # The second gold leaf is deliberately absent.
        _candidate("u4", ["x1", "x2", "x3"]),
    ]
    base_original = {
        "u1": _prediction("MATCH", "x1"),
        "u2": _prediction("MATCH_FAMILY_ONLY"),
        "u3": _prediction("NO_CANDIDATE_FITS"),
        "u4": _prediction("NO_CANDIDATE_FITS"),
    }
    base_hashed = {
        **base_original,
        "u3": _prediction("GENERIC_VERDICT"),
    }
    lora = {
        "u1": _prediction("MATCH", "m1"),
        "u2": _prediction("MATCH_FAMILY_ONLY"),
        "u3": _prediction("NO_CANDIDATE_FITS"),
        "u4": _prediction("NO_CANDIDATE_FITS"),
    }
    original = [
        _paired_row(row, "original", base_original[row["norm_uid"]], lora[row["norm_uid"]])
        for row in candidates
    ]
    hashed = [
        _paired_row(row, "hashed", base_hashed[row["norm_uid"]], lora[row["norm_uid"]])
        for row in candidates
    ]
    return truth, candidates, original, hashed


def test_scores_conservative_paired_gain_and_candidate_recall():
    truth, candidates, original, hashed = _fixture()
    report, audit = score_rows(
        truth,
        candidates,
        original,
        hashed,
        minimum_stability=0.75,
    )
    assert report["candidate_retrieval"] == {
        "candidate_k": 3,
        "gold_match_candidate_present_count": 1,
        "gold_match_candidate_absent_count": 1,
        "candidate_recall_of_gold_match": 0.5,
    }
    paired = report["paired_primary_exact_decision_and_leaf"]
    assert paired["base_mean_accuracy"] == 0.375
    assert paired["lora_mean_accuracy"] == 0.75
    assert paired["gain"] == 0.375
    assert paired["per_order_paired_exact"]["original"]["lora_only_correct"] == 1
    assert paired["per_order_paired_exact"]["hashed"]["lora_only_correct"] == 2
    assert report["systems"]["base"]["two_order"][
        "valid_exact_output_stability"
    ] == 0.75
    assert report["systems"]["lora"]["two_order"][
        "valid_exact_output_stability"
    ] == 1.0
    assert report["systems"]["lora"]["orders"]["original"][
        "gold_match_candidate_present"
    ]["conditional_exact_leaf_accuracy"] == 1.0
    assert report["promotion_gate"]["passed"] is True
    assert len(audit) == 4
    assert {row["paired_exact_order_count_transition"] for row in audit} == {
        "0->0",
        "0->2",
        "1->2",
        "2->2",
    }


def test_invalid_output_is_counted_and_cannot_be_stable():
    truth, candidates, original, hashed = _fixture()
    original[0]["lora"] = _prediction("INVALID_OUTPUT")
    hashed[0]["lora"] = _prediction("INVALID_OUTPUT")
    report, _ = score_rows(truth, candidates, original, hashed)
    assert report["systems"]["lora"]["orders"]["original"]["invalid_rate"] == 0.25
    assert report["systems"]["lora"]["two_order"][
        "valid_exact_output_stability"
    ] == 0.75
    assert report["promotion_gate"]["checks"][
        "maximum_per_order_invalid_rate_at_most_limit"
    ] is False


def test_rejects_slate_or_paired_prompt_drift():
    truth, candidates, original, hashed = _fixture()
    original[0]["candidate_ids"] = list(reversed(original[0]["candidate_ids"]))
    with pytest.raises(ValueError, match="slate drift"):
        score_rows(truth, candidates, original, hashed)

    truth, candidates, original, hashed = _fixture()
    hashed[0]["lora_item_prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="prompt mismatch"):
        score_rows(truth, candidates, original, hashed)


def test_exact_binomial_and_paired_interval_helpers():
    lower, upper = clopper_pearson(5, 10)
    assert lower == pytest.approx(0.187086, abs=1e-5)
    assert upper == pytest.approx(0.812914, abs=1e-5)
    paired = paired_exact_change(
        [True, False, False, True],
        [True, True, True, True],
    )
    assert paired["gain"] == 0.5
    assert paired["discordant_count"] == 2
    assert paired["exact_one_sided_mcnemar_p_lora_better"] == 0.25
    assert paired["conditional_exact_gain_interval"][1] == 0.5


def test_cpu_materializer_keeps_base_and_lora_in_one_shared_prompt_row():
    candidate_row = _candidate("u1", ["m1", "m2"])
    norm = {
        "norm_uid": "u1",
        "corpus": "humor_multi",
        "task": "humor",
        "row": 1,
    }
    batch = [(candidate_row, norm, candidate_row["candidates"], "rendered")]
    values = [(_prediction("MATCH", "m1"), None, '{"decision":"MATCH"}')]
    rows = paired_rows_for_batch(
        batch,
        values,
        values,
        [0],
        [0],
        order="original",
        prompt_hash="prompt",
        freeze_hash="freeze",
        model="model",
        adapter="adapter",
        adapter_name="humor",
        keep_raw=False,
    )
    assert len(rows) == 1
    assert rows[0]["base_item_prompt_sha256"] == rows[0]["lora_item_prompt_sha256"]
    assert rows[0]["base"]["metric_id"] == rows[0]["lora"]["metric_id"] == "m1"
    assert rows[0]["inference_freeze_sha256"] == "freeze"
    assert rows[0]["base"]["raw_response"] is None


def test_model_inventory_is_exactly_bound(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    inventory = tmp_path / "inventory.json"
    payload = {
        "status": "FROZEN_CONTENT_HASH_INVENTORY",
        "root": str(model),
        "file_count": 12,
        "content_inventory_sha256": EXPECTED_MODEL_CONTENT_SHA256,
    }
    inventory.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_model_inventory(inventory, model) == payload
    payload["file_count"] = 11
    inventory.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content inventory mismatch"):
        validate_model_inventory(inventory, model)
