import json

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.merge_rescue_decisions import (
    _reason_map_sha256,
    _reason_transitions,
    _uid_set_sha256,
    _validate_unresolved_reconciliation,
    merge_rescue,
)


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path, possible=False):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "v3"}))
    primary = tmp_path / "primary.jsonl"
    _write(primary, [
        {"norm_uid": "m", "task": "t", "corpus": "c", "decision": "NO_CANDIDATE_FITS", "metric_id": None, "candidate_bank_source_sha256": "sha"},
        {"norm_uid": "a", "task": "t", "corpus": "c", "decision": "NOISE", "metric_id": None, "candidate_bank_source_sha256": "sha"},
        {"norm_uid": "keep", "task": "t", "corpus": "c", "decision": "MATCH", "metric_id": "a0", "candidate_bank_source_sha256": "sha"},
    ])
    finalists = tmp_path / "finalists.jsonl"
    _write(finalists, [{"norm_uid": "m", "task": "t", "corpus": "c", "bank_source_sha256": "sha", "rescue_proposed_metric_ids": ["a1"], "candidates": [{"metric_id": "a1"}]}])
    final_results = tmp_path / "final-results.jsonl"
    _write(final_results, [{"norm_uid": "m", "task": "t", "corpus": "c", "decision": "MATCH", "metric_id": "a1", "confidence": "high", "reason": "exact", "candidate_ids": ["a1"], "candidate_bank_source_sha256": "sha", "prompt_sha256": "p", "model": "g"}])
    audits = tmp_path / "audits.jsonl"
    _write(audits, [{
        "norm_uid": "a", "task": "t", "corpus": "c",
        "bank_source_sha256": "sha", "rescue_coverage_repeats": 2,
        "rescue_reincludes_primary": True,
    }])
    verifications = tmp_path / "verifications.jsonl"
    _write(verifications, [{
        "norm_uid": "a", "task": "t", "corpus": "c",
        "decision": "POSSIBLE_EXACT_BANK_MATCH" if possible else "NOISE",
        "confirmed_decision": None if possible else "NOISE",
        "possible_exact_bank_match": possible,
        "confidence": "high", "reason": "garbled", "bank_source_sha256": "sha",
        "prompt_sha256": "q", "model": "g",
        "schema_version": "silver-match-v3-two-order-abstention-verification-v1",
        "verification_orders": ["original", "hashed"],
        "strict_two_order_abstention": not possible,
        "rescue_coverage_repeats": 2, "rescue_reincludes_primary": True,
    }])
    return manifest, primary, finalists, final_results, audits, verifications


def test_merge_upgrades_match_and_confirms_typed_abstention(tmp_path):
    paths = _fixture(tmp_path)
    output = tmp_path / "merged.jsonl"
    report = merge_rescue(
        manifest_path=paths[0], primary_paths=[paths[1]],
        finalist_candidate_paths=[paths[2]], finalist_adjudication_paths=[paths[3]],
        no_match_audit_paths=[paths[4]], abstention_verification_paths=[paths[5]],
        output_path=output,
    )
    rows = {row["norm_uid"]: row for row in map(json.loads, output.read_text().splitlines())}
    assert rows["m"]["decision"] == "MATCH" and rows["m"]["metric_id"] == "a1"
    assert rows["a"]["decision"] == "NOISE" and rows["a"]["metric_id"] is None
    assert rows["keep"]["rescue_status"] == "NOT_APPLICABLE_PRIMARY_MATCH"
    assert report["unresolved_rows"] == 0


def test_possible_exact_match_fails_closed(tmp_path):
    paths = _fixture(tmp_path, possible=True)
    unresolved = tmp_path / "unresolved.jsonl"
    with pytest.raises(ValueError, match="possible_exact_bank_match"):
        merge_rescue(
            manifest_path=paths[0], primary_paths=[paths[1]],
            finalist_candidate_paths=[paths[2]], finalist_adjudication_paths=[paths[3]],
            no_match_audit_paths=[paths[4]], abstention_verification_paths=[paths[5]],
            output_path=tmp_path / "merged.jsonl",
            unresolved_output_path=unresolved,
        )
    row = json.loads(unresolved.read_text())
    assert row["norm_uid"] == "a"
    assert row["source"] == "typed_abstention"
    assert row["unresolved_reason"] == "possible_exact_bank_match"


def test_strict_rescue_match_requires_two_orders_and_selected_verifier(tmp_path):
    paths = _fixture(tmp_path)
    final_row = json.loads(paths[3].read_text().strip())
    final_row["order_mode"] = "original"
    _write(paths[3], [final_row])
    order = tmp_path / "order.jsonl"
    _write(
        order,
        [
            {
                **final_row,
                "order_mode": "hashed",
            }
        ],
    )
    verifier = tmp_path / "match-verifier.jsonl"
    _write(
        verifier,
        [
            {
                "norm_uid": "m",
                "decision": "CONFIRM_MATCH",
                "metric_id": "a1",
                "confidence": "high",
                "reason": "exact under contrast",
                "prompt_sha256": "v" * 64,
                "primary_prompt_sha256": "p",
                "candidate_bank_source_sha256": "sha",
            }
        ],
    )
    adjudicator_selection = tmp_path / "adj-selection.json"
    adjudicator_selection.write_text(
        json.dumps(
            {
                "task": "t",
                "selection_split": "dev",
                "chosen": {"prompt_sha256": "p"},
            }
        )
    )
    # Selection records carry real SHA-256 strings; adjust both inference rows.
    adj_sha = "a" * 64
    final_row["prompt_sha256"] = adj_sha
    _write(paths[3], [final_row])
    order_row = {**final_row, "order_mode": "hashed"}
    _write(order, [order_row])
    adjudicator_selection.write_text(
        json.dumps(
            {
                "task": "t",
                "selection_split": "dev",
                "chosen": {"prompt_sha256": adj_sha},
            }
        )
    )
    verifier_selection = tmp_path / "verifier-selection.json"
    verifier_selection.write_text(
        json.dumps(
            {
                "task": "t",
                "selection_split": "dev",
                "chosen": {"prompt_sha256": "v" * 64},
            }
        )
    )
    verifier_policy = tmp_path / "verifier-policy.json"
    verifier_policy.write_text(
        json.dumps(
            {
                "task": "t",
                "inputs": {
                    "selection": {"sha256": sha256_file(verifier_selection)}
                },
                "may_run_on_production_unlabeled_norms": True,
                "dev_gate": {"cleared": True},
                "order_policy": {"orders": ["original", "hashed"]},
            }
        )
    )
    verifier_row = json.loads(verifier.read_text().strip())
    verifier_row["primary_prompt_sha256"] = adj_sha
    verifier_row.update(
        {
            "schema_version": "silver-match-v3-two-order-production-verification-v1",
            "verification_orders": ["original", "hashed"],
            "strict_two_order_acceptance": True,
            "verifier_selection_sha256": sha256_file(verifier_selection),
            "verifier_policy_sha256": sha256_file(verifier_policy),
        }
    )
    _write(verifier, [verifier_row])
    report = merge_rescue(
        manifest_path=paths[0],
        primary_paths=[paths[1]],
        finalist_candidate_paths=[paths[2]],
        finalist_adjudication_paths=[paths[3]],
        finalist_order_paths=[order],
        finalist_verification_paths=[verifier],
        no_match_audit_paths=[paths[4]],
        abstention_verification_paths=[paths[5]],
        output_path=tmp_path / "strict-merged.jsonl",
        adjudicator_selection_path=adjudicator_selection,
        verifier_selection_path=verifier_selection,
        verifier_policy_path=verifier_policy,
        strict_production=True,
    )
    assert report["strict_production"] is True

    policy_payload = json.loads(verifier_policy.read_text())
    policy_payload["order_policy"]["orders"] = ["original", "hashed", "reverse"]
    verifier_policy.write_text(json.dumps(policy_payload))
    verifier_row.update(
        {
            "schema_version": "silver-match-v3-multi-order-production-verification-v1",
            "verification_orders": ["original", "hashed", "reverse"],
            "strict_two_order_acceptance": None,
            "strict_all_order_acceptance": True,
            "accepted_by_order": {
                "original": True,
                "hashed": True,
                "reverse": True,
            },
            "verifier_policy_sha256": sha256_file(verifier_policy),
        }
    )
    _write(verifier, [verifier_row])
    report = merge_rescue(
        manifest_path=paths[0],
        primary_paths=[paths[1]],
        finalist_candidate_paths=[paths[2]],
        finalist_adjudication_paths=[paths[3]],
        finalist_order_paths=[order],
        finalist_verification_paths=[verifier],
        no_match_audit_paths=[paths[4]],
        abstention_verification_paths=[paths[5]],
        output_path=tmp_path / "strict-multi-order-merged.jsonl",
        adjudicator_selection_path=adjudicator_selection,
        verifier_selection_path=verifier_selection,
        verifier_policy_path=verifier_policy,
        strict_production=True,
    )
    assert report["strict_production"] is True


@pytest.mark.parametrize(
    ("label_source", "confidence", "label_extra", "validation_extra"),
    [
        ("independent_codex_full_bank", "high", {}, {}),
        (
            "independent_full_bank_multi_vote_consensus",
            "medium",
            {
                "consensus_vote_count": 2,
                "consensus_total_eligible_votes": 3,
                "consensus_vote_sources": ["codex_isolated_pass_01", "gemma4_gepa_two_order"],
                "permanently_excluded_from_gradients": True,
                "training_eligible_preverification": False,
            },
            {
                "schema_version": "silver-match-v3-full-bank-multi-vote-consensus-v1",
                "unresolved": {"count": 0},
                "policy": {
                    "minimum_independent_votes": 2,
                    "winner_must_be_unique": True,
                    "source_confidences_are_preserved_not_upgraded": True,
                },
            },
        ),
    ],
)
def test_frozen_blind_high_label_resolves_exact_unresolved_set(
    tmp_path, label_source, confidence, label_extra, validation_extra
):
    paths = list(_fixture(tmp_path, possible=True))
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "task": "t",
                "source_sha256": "sha",
                "metrics": [
                    {"metric_id": "a0", "name": "zero"},
                    {"metric_id": "a1", "name": "one"},
                ],
            }
        )
    )
    paths[0].write_text(
        json.dumps(
            {
                "schema_version": "v3",
                "banks": {"t": {"path": str(bank), "source_sha256": "sha"}},
            }
        )
    )
    unresolved = tmp_path / "unresolved.jsonl"
    with pytest.raises(ValueError, match="possible_exact_bank_match"):
        merge_rescue(
            manifest_path=paths[0],
            primary_paths=[paths[1]],
            finalist_candidate_paths=[paths[2]],
            finalist_adjudication_paths=[paths[3]],
            no_match_audit_paths=[paths[4]],
            abstention_verification_paths=[paths[5]],
            output_path=tmp_path / "first.jsonl",
            unresolved_output_path=unresolved,
        )

    pack_validation = tmp_path / "pack.validation.json"
    pack_validation.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-unresolved-label-pack-v1",
                "truth_hidden": True,
                "system_key_excluded_from_label_pack": True,
                "permanently_excluded_from_gradients": True,
                "inputs": {"unresolved": {"sha256": sha256_file(unresolved)}},
            }
        )
    )
    labels = tmp_path / "manual.jsonl"
    _write(
        labels,
        [
            {
                "norm_uid": "a",
                "task": "t",
                "corpus": "c",
                "row": None,
                "decision": "NOISE",
                "metric_id": None,
                "confidence": confidence,
                "reason": "independently judged garble",
                "label_source": label_source,
                "current_bank_source_sha256": "sha",
                **label_extra,
            }
        ],
    )
    validation = tmp_path / "manual.validation.json"
    validation.write_text(
        json.dumps(
            {
                "complete": True,
                "count": 1,
                "output": {"sha256": sha256_file(labels)},
                "pack_validation": {
                    "path": str(pack_validation),
                    "sha256": sha256_file(pack_validation),
                },
                **validation_extra,
            }
        )
    )
    output = tmp_path / "resolved.jsonl"
    report = merge_rescue(
        manifest_path=paths[0],
        primary_paths=[paths[1]],
        finalist_candidate_paths=[paths[2]],
        finalist_adjudication_paths=[paths[3]],
        no_match_audit_paths=[paths[4]],
        abstention_verification_paths=[paths[5]],
        output_path=output,
        unresolved_output_path=unresolved,
        manual_unresolved_labels_path=labels,
        manual_unresolved_validation_path=validation,
    )
    rows = {row["norm_uid"]: row for row in map(json.loads, output.read_text().splitlines())}
    assert rows["a"]["decision"] == "NOISE"
    assert rows["a"]["rescue_status"] == "EXHAUSTIVE_RESCUE_MANUAL_RESOLVED"
    assert rows["a"]["verification_status"] == "independent_blind_unresolved_resolution"
    assert report["manual_unresolved_resolution"]["resolved_rows"] == 1


def test_reason_taxonomy_reconciliation_is_exact_uid_and_hash_bound(tmp_path):
    frozen_path = tmp_path / "frozen.jsonl"
    recomputed_path = tmp_path / "recomputed.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    validation_path = tmp_path / "validation.json"
    reconciliation_path = tmp_path / "reconciliation.json"
    frozen = {"u": "legacy_reason"}
    recomputed = {"u": "current_reason"}
    _write(
        frozen_path,
        [{"norm_uid": "u", "unresolved_reason": frozen["u"]}],
    )
    _write(
        recomputed_path,
        [{"norm_uid": "u", "unresolved_reason": recomputed["u"]}],
    )
    _write(labels_path, [{"norm_uid": "u", "decision": "NOISE"}])
    validation_path.write_text(json.dumps({"complete": True}), encoding="utf-8")
    report = {
        "schema_version": "silver-match-v3-unresolved-ledger-reason-taxonomy-reconciliation-v1",
        "status": "PASS_EXACT_UID_SET_REASON_TAXONOMY_ONLY",
        "exact_uid_set_equality": True,
        "changed_fields": ["unresolved_reason"],
        "frozen_ledger": {
            "path": str(frozen_path),
            "sha256": sha256_file(frozen_path),
            "count": 1,
            "uid_set_canonical_sha256": _uid_set_sha256(frozen),
            "reason_map_canonical_sha256": _reason_map_sha256(frozen),
        },
        "recomputed_ledger": {
            "path": str(recomputed_path),
            "sha256": sha256_file(recomputed_path),
            "count": 1,
            "uid_set_canonical_sha256": _uid_set_sha256(recomputed),
            "reason_map_canonical_sha256": _reason_map_sha256(recomputed),
        },
        "reason_transitions": _reason_transitions(frozen, recomputed),
        "changed_reason_rows": 1,
        "unchanged_reason_rows": 0,
        "manual_consensus": {
            "labels_path": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "validation_path": str(validation_path),
            "validation_sha256": sha256_file(validation_path),
        },
        "scientific_effect": {
            "uid_membership_changed": False,
            "manual_label_coverage_changed": False,
            "only_internal_failure_reason_taxonomy_changed": True,
        },
    }
    reconciliation_path.write_text(json.dumps(report), encoding="utf-8")
    result = _validate_unresolved_reconciliation(
        path=reconciliation_path,
        frozen_path=frozen_path,
        frozen=frozen,
        recomputed=recomputed,
        manual_labels_path=labels_path,
        manual_validation_path=validation_path,
    )
    assert result["exact_uid_set_equality"] is True

    report["reason_transitions"][0]["count"] = 2
    reconciliation_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic hashes"):
        _validate_unresolved_reconciliation(
            path=reconciliation_path,
            frozen_path=frozen_path,
            frozen=frozen,
            recomputed=recomputed,
            manual_labels_path=labels_path,
            manual_validation_path=validation_path,
        )

    report["reason_transitions"] = _reason_transitions(frozen, recomputed)
    reconciliation_path.write_text(json.dumps(report), encoding="utf-8")
    labels_path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manual consensus binding"):
        _validate_unresolved_reconciliation(
            path=reconciliation_path,
            frozen_path=frozen_path,
            frozen=frozen,
            recomputed=recomputed,
            manual_labels_path=labels_path,
            manual_validation_path=validation_path,
        )
