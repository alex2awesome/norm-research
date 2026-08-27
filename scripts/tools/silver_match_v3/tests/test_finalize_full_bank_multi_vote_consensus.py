import json
import sys

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.finalize_full_bank_multi_vote_consensus import (
    LABEL_SOURCE,
    finalize_consensus,
    main,
)


def _row(decision, metric_id=None, confidence="high"):
    return {
        "decision": decision,
        "metric_id": metric_id,
        "confidence": confidence,
    }


def test_gemma_pair_is_one_vote_and_resolver_breaks_disagreement():
    items = {
        "u1": {"corpus": "c", "task": "t", "row": 1},
        "u2": {"corpus": "c", "task": "t", "row": 2},
        "u3": {"corpus": "c", "task": "t", "row": 3},
    }
    codex_a = {
        "u1": _row("MATCH", "a", "medium"),
        "u2": _row("NO_CANDIDATE_FITS"),
        "u3": _row("MATCH", "b"),
    }
    gemma_original = {
        "u1": _row("MATCH", "a"),
        "u2": _row("NO_CANDIDATE_FITS"),
        "u3": _row("MATCH", "a"),
    }
    gemma_hashed = {
        "u1": _row("MATCH", "a"),
        "u2": _row("MATCH", "a"),
        "u3": _row("MATCH", "a"),
    }

    accepted, unresolved, report = finalize_consensus(
        original_items=items,
        bank_ids={"a", "b"},
        bank_sha="bank",
        codex_passes=[("codex_a", codex_a, "sha-a")],
        gemma_original=gemma_original,
        gemma_hashed=gemma_hashed,
    )
    assert [row["norm_uid"] for row in accepted] == ["u1"]
    assert accepted[0]["confidence"] == "medium"
    assert accepted[0]["label_source"] == LABEL_SOURCE
    assert {row["norm_uid"] for row in unresolved} == {"u2", "u3"}
    assert report["complete"] is False

    codex_b = {
        "u2": _row("NO_CANDIDATE_FITS", confidence="medium"),
        "u3": _row("MATCH", "b", confidence="medium"),
    }
    accepted, unresolved, report = finalize_consensus(
        original_items=items,
        bank_ids={"a", "b"},
        bank_sha="bank",
        codex_passes=[
            ("codex_a", codex_a, "sha-a"),
            ("codex_b", codex_b, "sha-b"),
        ],
        gemma_original=gemma_original,
        gemma_hashed=gemma_hashed,
    )
    rows = {row["norm_uid"]: row for row in accepted}
    assert not unresolved
    assert report["complete"] is True
    assert rows["u2"]["consensus_vote_count"] == 2
    # The Gemma pair is one competing vote; two isolated Codex votes resolve it.
    assert rows["u3"]["consensus_vote_count"] == 2
    assert rows["u3"]["consensus_total_eligible_votes"] == 3
    assert rows["u3"]["metric_id"] == "b"


def test_gemma_low_confidence_pair_does_not_promote():
    items = {"u": {"corpus": "c", "task": "t", "row": 1}}
    accepted, unresolved, _ = finalize_consensus(
        original_items=items,
        bank_ids={"a"},
        bank_sha="bank",
        codex_passes=[("codex", {"u": _row("MATCH", "a")}, "sha")],
        gemma_original={"u": _row("MATCH", "a", "low")},
        gemma_hashed={"u": _row("MATCH", "a", "low")},
        min_gemma_confidence="medium",
    )
    assert not accepted
    assert unresolved[0]["unresolved_reason"] == "insufficient_exact_independent_votes"


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_validates_frozen_lineage_and_writes_complete_consensus(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    items = [
        {"schema_version": "v3", "norm_uid": "u", "corpus": "c", "task": "t", "row": 1,
         "split_group": "g", "split": "test"}
    ]
    bank = {"task": "t", "source_sha256": "bank", "metrics": [{"metric_id": "a"}]}
    _write_jsonl(pack / "items.jsonl", items)
    (pack / "bank.json").write_text(json.dumps(bank), encoding="utf-8")
    pack_validation = {
        "schema_version": "silver-match-v3-unresolved-label-pack-v1",
        "task": "t",
        "count": 1,
        "bank_source_sha256": "bank",
        "truth_hidden": True,
        "system_key_excluded_from_label_pack": True,
        "outputs": {
            "items": {"sha256": sha256_file(pack / "items.jsonl")},
            "bank": {"sha256": sha256_file(pack / "bank.json")},
        },
    }
    (pack / "validation.json").write_text(json.dumps(pack_validation), encoding="utf-8")

    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [{"norm_uid": "u", "bank_source_sha256": "bank",
                              "candidates": [{"metric_id": "a"}]}])
    candidate_freeze = tmp_path / "candidate.freeze.json"
    candidate_freeze.write_text(json.dumps({
        "status": "FROZEN_BEFORE_INFERENCE", "truth_hidden": True, "count": 1,
        "candidate_depth": 1,
        "inputs": {"pack_validation": {"sha256": sha256_file(pack / "validation.json")}},
        "output": {"sha256": sha256_file(candidates)},
    }), encoding="utf-8")

    prompt, addon = tmp_path / "prompt.txt", tmp_path / "addon.txt"
    prompt.write_text("prompt\n", encoding="utf-8")
    addon.write_text("addon\n", encoding="utf-8")
    combined_sha = __import__("hashlib").sha256(b"prompt\n\naddon\n").hexdigest()
    gemma_freeze = tmp_path / "gemma.freeze.json"
    gemma_freeze.write_text(json.dumps({
        "status": "FROZEN_BEFORE_INFERENCE",
        "scientific_contract": {
            "gemma_outputs_may_not_directly_enter_the_release": True,
            "gemma_two_order_exact_consensus_counts_as_one_model_vote": True,
            "promotion_requires_exact_agreement_with_an_independent_codex_full_bank_label_or_further_resolver": True,
            "disagreements_remain_unresolved": True,
        },
        "inputs": {
            "candidate_freeze": {"sha256": sha256_file(candidate_freeze)},
            "candidates": {"sha256": sha256_file(candidates)},
            "prompt": {"path": str(prompt), "sha256": sha256_file(prompt)},
            "prompt_addon": {"path": str(addon), "sha256": sha256_file(addon)},
        },
        "runtime": {"model": "gemma"},
    }), encoding="utf-8")
    original, hashed = tmp_path / "gemma.original.v2.jsonl", tmp_path / "gemma.hashed.v2.jsonl"
    base_gemma = {
        "schema_version": "v3", "norm_uid": "u", "corpus": "c", "task": "t", "row": 1,
        "decision": "MATCH", "metric_id": "a", "confidence": "high", "reason": "exact",
        "candidate_ids": ["a"], "candidate_bank_source_sha256": "bank",
        "prompt_sha256": combined_sha, "model": "gemma", "parse_error": None,
    }
    _write_jsonl(original, [{**base_gemma, "order_mode": "original"}])
    _write_jsonl(hashed, [{**base_gemma, "order_mode": "hashed"}])
    for path, order in ((original, "original"), (hashed, "hashed")):
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps({
            "output_sha256": sha256_file(path),
            "input_candidates_sha256": sha256_file(candidates),
            "model": "gemma", "prompt_sha256": combined_sha, "order_mode": order,
        }), encoding="utf-8")
    retry = tmp_path / "retry.json"
    retry.write_text(json.dumps({
        "status": "FROZEN_BEFORE_RETRY_INFERENCE",
        "unchanged_inputs_and_scientific_contract": True,
        "retry_change": {"candidate_depth": 1},
        "outputs": {"original": str(original), "hashed": str(hashed)},
    }), encoding="utf-8")

    codex_labels = tmp_path / "codex.jsonl"
    _write_jsonl(codex_labels, [{
        **items[0], "decision": "MATCH", "metric_id": "a", "confidence": "medium",
        "reason": "exact", "label_source": "independent_codex_full_bank",
        "current_bank_source_sha256": "bank",
    }])
    codex_validation = tmp_path / "codex.validation.json"
    codex_transcript = tmp_path / "codex.transcript.json"
    codex_transcript.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    codex_validation.write_text(json.dumps({
        "complete": True, "count": 1, "bank_source_sha256": "bank",
        "output": {"sha256": sha256_file(codex_labels)},
        "pack_validation": {"sha256": sha256_file(pack / "validation.json")},
        "transcript_audit": {"status": "PASS", "path": str(codex_transcript),
                             "sha256": sha256_file(codex_transcript)},
    }), encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "finalize_full_bank_multi_vote_consensus",
        "--pack-root", str(pack), "--candidates", str(candidates),
        "--candidate-freeze", str(candidate_freeze), "--gemma-freeze", str(gemma_freeze),
        "--gemma-retry-freeze", str(retry), "--gemma-original", str(original),
        "--gemma-hashed", str(hashed), "--codex-pack-root", str(pack),
        "--codex-labels", str(codex_labels), "--codex-validation", str(codex_validation),
        "--output-root", str(output),
    ])
    main()
    report = json.loads((output / "validation.json").read_text())
    row = json.loads((output / "labels.jsonl").read_text())
    assert report["complete"] is True and report["unresolved_count"] == 0
    assert row["decision"] == "MATCH" and row["metric_id"] == "a"
    assert row["confidence"] == "medium"
