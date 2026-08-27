import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.run_isolated_pr_verifier_prompt_author import (
    EXPECTED_CONFIDENCES,
    EXPECTED_DECISIONS,
    audit_tool_free_transcript,
    validate_and_sanitize_packet,
    validate_author_output,
)


def _json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _packet(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for index in range(20):
        target = "CONFIRM_MATCH" if index < 10 else "REJECT"
        rows.append(
            {
                "schema_version": "silver-match-v3-verifier-author-example-v1",
                "norm_uid": f"u{index}",
                "source_group": f"g{index}",
                "task": "press-releases",
                "gepa_role": "optimize",
                "predeclared_split": "train",
                "norm": f"norm {index}",
                "context": f"context {index}",
                "proposal": {"decision": "MATCH", "metric_id": "m1", "reason": "r"},
                "gold": {"decision": "MATCH", "metric_id": "m1" if index < 10 else "m2"},
                "target": target,
                "metric_cards": {"m1": {"metric_id": "m1"}},
                "use_contract": {
                    "verifier_prompt_authorship_or_gepa_optimize_only": True,
                    "verifier_selection": False,
                    "final_blind_audit": False,
                    "mi_or_outcome_estimation": False,
                    "retriever_training": False,
                },
            }
        )
    examples = _jsonl(tmp_path / "examples.jsonl", rows)
    report = _json(
        tmp_path / "REPORT.json",
        {
            "schema_version": "silver-match-v3-verifier-author-training-packet-v1",
            "status": "FROZEN_OPTIMIZE_ONLY_AUTHORSHIP_EVIDENCE",
            "task": "press-releases",
            "count": 20,
            "source_groups": 20,
            "fresh_verifier_dev_truth_read": False,
            "blind_audit_truth_read": False,
            "target_counts": {"CONFIRM_MATCH": 10, "REJECT": 10},
            "outputs": {"examples": {"sha256": sha256_file(examples)}},
        },
    )
    return report, examples


def test_packet_is_identity_stripped_and_balanced(tmp_path):
    report, examples = _packet(tmp_path)
    _, payload = validate_and_sanitize_packet(report, examples)
    assert payload["count"] == 20
    assert payload["target_counts"] == {"CONFIRM_MATCH": 10, "REJECT": 10}
    assert all("norm_uid" not in row and "source_group" not in row for row in payload["examples"])


def test_packet_with_dev_truth_fails_closed(tmp_path):
    report, examples = _packet(tmp_path)
    value = json.loads(report.read_text())
    value["fresh_verifier_dev_truth_read"] = True
    report.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="unsupported"):
        validate_and_sanitize_packet(report, examples)


def test_author_output_and_tool_free_transcript_contract():
    hashes = {key: key[0] * 64 for key in ("report", "examples", "evidence", "input_freeze")}
    value = {
        "schema_version": "silver-match-v3-pr-verifier-fresh-author-v1",
        "prompt_name": "one",
        "prompt_text": (
            "Confidence must be exactly one of high, medium, or low. "
            "Reason must be at most 24 words. " + "x" * 900
        ),
        "parse_rule": {
            "allowed_decisions": EXPECTED_DECISIONS,
            "confirm_metric_id_is_proposal": True,
            "better_candidate_metric_id_is_supplied_alternative": True,
            "other_decisions_metric_id_null": True,
            "confidence_values": EXPECTED_CONFIDENCES,
            "confirm_requires_explicit_criterion": True,
            "confirm_requires_exact_leaf_contrast": True,
            "reason_max_words": 24,
        },
        "selection_rule": {
            "variant_count": 1,
            "choose_without_verifier_dev_truth": True,
            "optimize_only_authoring": True,
            "promotion_requires_frozen_fresh_dev_gates": True,
            "precision_dominates_yield": True,
            "no_candidate_fits_route": "full_bank_rescue",
        },
        "provenance": {
            "used_only_inline_identity_stripped_optimize_evidence": True,
            "tools_called": False,
            "verifier_dev_truth_read": False,
            "select_test_or_blind_material_read": False,
            "mi_or_outcomes_read": False,
            "training_packet_report_sha256": hashes["report"],
            "training_examples_sha256": hashes["examples"],
            "sanitized_evidence_sha256": hashes["evidence"],
            "input_freeze_sha256": hashes["input_freeze"],
        },
    }
    validate_author_output(value, expected_hashes=hashes)
    assert audit_tool_free_transcript("model: gpt\nfinal\n{}\n")["tool_event_count"] == 0
    with pytest.raises(ValueError, match="used tools"):
        audit_tool_free_transcript("model: gpt\nexec\n/bin/zsh -lc pwd\n")
