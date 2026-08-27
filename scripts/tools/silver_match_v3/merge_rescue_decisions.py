#!/usr/bin/env python3
"""Merge exhaustive-rescue outcomes into primary adjudications, failing closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .adjudicate_gemma import DECISIONS
from .common import read_jsonl, sha256_file, write_jsonl
from .finalize_adjudications import final_match_decision, selected_prompt_sha
from .verify_abstention_gemma import TYPED_DECISIONS


MULTI_VOTE_LABEL_SOURCE = "independent_full_bank_multi_vote_consensus"
MULTI_VOTE_SCHEMA = "silver-match-v3-full-bank-multi-vote-consensus-v1"
UNRESOLVED_RECONCILIATION_SCHEMA = (
    "silver-match-v3-unresolved-ledger-reason-taxonomy-reconciliation-v1"
)


def _unique(paths: Iterable[Path], kind: str) -> dict[str, dict[str, Any]]:
    output = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in output:
                raise ValueError(f"missing/duplicate {kind} norm_uid: {uid!r}")
            output[uid] = row
    return output


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _uid_set_sha256(values: dict[str, str]) -> str:
    return _canonical_sha256(sorted(values))


def _reason_map_sha256(values: dict[str, str]) -> str:
    return _canonical_sha256(
        [
            {"norm_uid": uid, "unresolved_reason": values[uid]}
            for uid in sorted(values)
        ]
    )


def _reason_transitions(
    frozen: dict[str, str], recomputed: dict[str, str]
) -> list[dict[str, Any]]:
    counts = Counter(
        (frozen[uid], recomputed[uid])
        for uid in frozen
        if frozen[uid] != recomputed[uid]
    )
    return [
        {"from": old, "to": new, "count": count}
        for (old, new), count in sorted(counts.items())
    ]


def _validate_unresolved_reconciliation(
    *,
    path: Path,
    frozen_path: Path,
    frozen: dict[str, str],
    recomputed: dict[str, str],
    manual_labels_path: Path,
    manual_validation_path: Path,
) -> dict[str, Any]:
    """Accept an old blind ledger only when membership is exactly unchanged.

    This is deliberately narrower than an override flag.  A frozen report must
    bind both ledgers, both reason maps, the exact UID set, the sole changed
    field, and the manual consensus inputs.  Any membership or non-taxonomic
    change remains a hard failure.
    """

    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != UNRESOLVED_RECONCILIATION_SCHEMA
        or report.get("status") != "PASS_EXACT_UID_SET_REASON_TAXONOMY_ONLY"
        or report.get("exact_uid_set_equality") is not True
        or report.get("changed_fields") != ["unresolved_reason"]
        or (report.get("scientific_effect") or {}).get("uid_membership_changed")
        is not False
        or (report.get("scientific_effect") or {}).get(
            "manual_label_coverage_changed"
        )
        is not False
        or (report.get("scientific_effect") or {}).get(
            "only_internal_failure_reason_taxonomy_changed"
        )
        is not True
    ):
        raise ValueError("unresolved reconciliation lacks the exact taxonomy-only gate")
    if set(frozen) != set(recomputed):
        raise ValueError("unresolved reconciliation cannot change UID membership")

    frozen_ref = report.get("frozen_ledger") or {}
    recomputed_ref = report.get("recomputed_ledger") or {}
    recomputed_path = Path(str(recomputed_ref.get("path") or ""))
    if (
        Path(str(frozen_ref.get("path") or "")).resolve()
        != frozen_path.resolve()
        or frozen_ref.get("sha256") != sha256_file(frozen_path)
        or int(frozen_ref.get("count", -1)) != len(frozen)
        or not recomputed_path.is_file()
        or recomputed_ref.get("sha256") != sha256_file(recomputed_path)
        or int(recomputed_ref.get("count", -1)) != len(recomputed)
    ):
        raise ValueError("unresolved reconciliation ledger binding changed")
    recomputed_artifact = {
        uid: str(row.get("unresolved_reason") or "")
        for uid, row in _unique([recomputed_path], "recomputed-unresolved").items()
    }
    if recomputed_artifact != recomputed:
        raise ValueError("reconciliation recomputed ledger differs from live failures")

    uid_sha = _uid_set_sha256(frozen)
    if (
        frozen_ref.get("uid_set_canonical_sha256") != uid_sha
        or recomputed_ref.get("uid_set_canonical_sha256") != uid_sha
        or frozen_ref.get("reason_map_canonical_sha256")
        != _reason_map_sha256(frozen)
        or recomputed_ref.get("reason_map_canonical_sha256")
        != _reason_map_sha256(recomputed)
        or report.get("reason_transitions")
        != _reason_transitions(frozen, recomputed)
        or int(report.get("changed_reason_rows", -1))
        != sum(frozen[uid] != recomputed[uid] for uid in frozen)
        or int(report.get("unchanged_reason_rows", -1))
        != sum(frozen[uid] == recomputed[uid] for uid in frozen)
    ):
        raise ValueError("unresolved reconciliation semantic hashes changed")

    manual = report.get("manual_consensus") or {}
    if (
        Path(str(manual.get("labels_path") or "")).resolve()
        != manual_labels_path.resolve()
        or manual.get("labels_sha256") != sha256_file(manual_labels_path)
        or Path(str(manual.get("validation_path") or "")).resolve()
        != manual_validation_path.resolve()
        or manual.get("validation_sha256") != sha256_file(manual_validation_path)
    ):
        raise ValueError("unresolved reconciliation manual consensus binding changed")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "changed_reason_rows": report["changed_reason_rows"],
        "reason_transitions": report["reason_transitions"],
        "exact_uid_set_equality": True,
    }


def merge_rescue(
    *,
    manifest_path: Path,
    primary_paths: list[Path],
    finalist_candidate_paths: list[Path],
    finalist_adjudication_paths: list[Path],
    finalist_order_paths: list[Path] | None = None,
    finalist_verification_paths: list[Path] | None = None,
    no_match_audit_paths: list[Path],
    abstention_verification_paths: list[Path],
    output_path: Path,
    adjudicator_selection_path: Path | None = None,
    verifier_selection_path: Path | None = None,
    verifier_policy_path: Path | None = None,
    rescue_plan_path: Path | None = None,
    unresolved_output_path: Path | None = None,
    manual_unresolved_labels_path: Path | None = None,
    manual_unresolved_validation_path: Path | None = None,
    unresolved_reconciliation_path: Path | None = None,
    strict_production: bool = False,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary = _unique(primary_paths, "primary")
    finalists = _unique(finalist_candidate_paths, "finalist-candidate")
    finalist_results = _unique(finalist_adjudication_paths, "finalist-adjudication")
    finalist_order = _unique(finalist_order_paths or [], "finalist-order")
    finalist_verification = _unique(
        finalist_verification_paths or [], "finalist-verification"
    )
    no_match = _unique(no_match_audit_paths, "no-match-audit")
    abstention_results = _unique(abstention_verification_paths, "abstention-verification")
    if set(finalists) != set(finalist_results):
        raise ValueError(
            f"finalist coverage mismatch: candidates={len(finalists)}, results={len(finalist_results)}"
        )
    expected_adjudicator_prompt = expected_verifier_prompt = None
    expected_verifier_orders: list[str] | None = None
    expected_rescue_plan_sha: str | None = None
    if strict_production:
        if (
            adjudicator_selection_path is None
            or verifier_selection_path is None
            or verifier_policy_path is None
        ):
            raise ValueError(
                "strict rescue merge requires adjudicator/verifier selections and policy"
            )
        tasks = {str(row.get("task") or "") for row in finalists.values()}
        if len(tasks) != 1:
            raise ValueError("strict rescue merge must contain exactly one task")
        task = next(iter(tasks))
        expected_adjudicator_prompt, _ = selected_prompt_sha(
            adjudicator_selection_path, task, "rescue adjudicator"
        )
        expected_verifier_prompt, _ = selected_prompt_sha(
            verifier_selection_path, task, "rescue verifier"
        )
        verifier_policy = json.loads(
            verifier_policy_path.read_text(encoding="utf-8")
        )
        selection_ref = (verifier_policy.get("inputs") or {}).get("selection") or {}
        if (
            verifier_policy.get("task") != task
            or selection_ref.get("sha256") != sha256_file(verifier_selection_path)
            or verifier_policy.get("may_run_on_production_unlabeled_norms") is not True
            or (verifier_policy.get("dev_gate") or {}).get("cleared") is not True
        ):
            raise ValueError("strict rescue verifier policy is not linked/dev-cleared")
        expected_verifier_orders = [
            str(value)
            for value in (verifier_policy.get("order_policy") or {}).get("orders") or []
        ]
        if expected_verifier_orders not in (
            ["original", "hashed"],
            ["original", "hashed", "reverse"],
        ):
            raise ValueError("strict rescue verifier policy has unsupported topology")
        if rescue_plan_path is not None:
            rescue_plan = json.loads(rescue_plan_path.read_text(encoding="utf-8"))
            expected_rescue_plan_sha = sha256_file(rescue_plan_path)
            if (
                rescue_plan.get("schema_version")
                != "silver-match-v3-task-rescue-plan-v3"
                or rescue_plan.get("status")
                != "FROZEN_READY_FOR_REPEATED_FULL_BANK_RESCUE"
                or rescue_plan.get("task") != task
                or (rescue_plan.get("verifier") or {}).get("orders")
                != expected_verifier_orders
                or (rescue_plan.get("verifier") or {})
                .get("selection", {})
                .get("sha256")
                != sha256_file(verifier_selection_path)
                or (rescue_plan.get("verifier") or {})
                .get("production_policy", {})
                .get("sha256")
                != sha256_file(verifier_policy_path)
            ):
                raise ValueError("strict rescue plan is not linked to selected verifier")
        if set(finalist_order) != set(finalists):
            raise ValueError("strict rescue finalist order coverage mismatch")
        expected_verification = {
            uid
            for uid, row in finalist_results.items()
            if row.get("decision") == "MATCH"
        }
        if set(finalist_verification) != expected_verification:
            raise ValueError("strict rescue finalist verifier coverage mismatch")
    if set(no_match) != set(abstention_results):
        raise ValueError(
            f"abstention verification mismatch: audits={len(no_match)}, results={len(abstention_results)}"
        )
    rescued = set(finalists) | set(no_match)
    if set(finalists) & set(no_match):
        raise ValueError("rescue finalist and no-match sets overlap")
    if not rescued <= set(primary):
        raise ValueError(f"rescue contains {len(rescued-set(primary))} UIDs absent from primary")

    unresolved = []
    replacement: dict[str, dict[str, Any]] = {}
    replacement_counts = Counter()
    for uid, result in finalist_results.items():
        decision = str(result.get("decision") or "")
        if decision not in DECISIONS:
            unresolved.append((uid, "invalid_finalist_decision"))
            continue
        if decision != "MATCH" and result.get("confidence") == "low":
            unresolved.append((uid, "low_confidence_finalist_abstention"))
            continue
        bank_sha = str(result.get("candidate_bank_source_sha256") or "")
        if bank_sha != str(finalists[uid].get("bank_source_sha256") or ""):
            unresolved.append((uid, "finalist_bank_hash_mismatch"))
            continue
        metric_id = result.get("metric_id")
        if strict_production:
            order = finalist_order[uid]
            if (
                result.get("prompt_sha256") != expected_adjudicator_prompt
                or order.get("prompt_sha256") != expected_adjudicator_prompt
            ):
                unresolved.append((uid, "unselected_finalist_adjudicator_prompt"))
                continue
            if result.get("model") != order.get("model"):
                unresolved.append((uid, "finalist_order_model_mismatch"))
                continue
            if str(order.get("candidate_bank_source_sha256") or "") != str(
                finalists[uid].get("bank_source_sha256") or ""
            ):
                unresolved.append((uid, "finalist_order_bank_hash_mismatch"))
                continue
            if {
                str(result.get("order_mode")), str(order.get("order_mode"))
            } != {"original", "hashed"}:
                unresolved.append((uid, "finalist_order_modes_not_original_hashed"))
                continue
            result_ids = [str(value) for value in result.get("candidate_ids") or []]
            order_ids = [str(value) for value in order.get("candidate_ids") or []]
            if len(result_ids) != len(order_ids) or set(result_ids) != set(order_ids):
                unresolved.append((uid, "finalist_order_candidate_mismatch"))
                continue
            if decision != "MATCH":
                unresolved.append((uid, "finalist_abstention_requires_independent_typed_audit"))
                continue
            verification = finalist_verification.get(uid)
            if verification is None:
                unresolved.append((uid, "missing_finalist_contrastive_verifier"))
                continue
            if (
                verification.get("prompt_sha256") != expected_verifier_prompt
                or verification.get("primary_prompt_sha256")
                != result.get("prompt_sha256")
            ):
                unresolved.append((uid, "unselected_or_unlinked_finalist_verifier"))
                continue
            verification_schema = verification.get("schema_version")
            if verification_schema == "silver-match-v3-two-order-production-verification-v1":
                topology_valid = bool(
                    expected_verifier_orders == ["original", "hashed"]
                    and verification.get("strict_two_order_acceptance") is True
                )
            elif verification_schema == "silver-match-v3-multi-order-production-verification-v1":
                accepted_by_order = verification.get("accepted_by_order")
                topology_valid = bool(
                    isinstance(accepted_by_order, dict)
                    and list(accepted_by_order) == expected_verifier_orders
                    and all(value is True for value in accepted_by_order.values())
                    and verification.get("strict_all_order_acceptance") is True
                )
            else:
                topology_valid = False
            if (
                not topology_valid
                or verification.get("verification_orders") != expected_verifier_orders
                or verification.get("confidence") != "high"
                or verification.get("verifier_selection_sha256")
                != sha256_file(verifier_selection_path)
                or verification.get("verifier_policy_sha256")
                != sha256_file(verifier_policy_path)
                or (
                    expected_rescue_plan_sha is not None
                    and verification.get("rescue_plan_sha256")
                    != expected_rescue_plan_sha
                )
            ):
                unresolved.append((uid, "finalist_verifier_not_strict_selected_orders"))
                continue
            if str(verification.get("candidate_bank_source_sha256") or "") != str(
                finalists[uid].get("bank_source_sha256") or ""
            ):
                unresolved.append((uid, "finalist_verifier_bank_hash_mismatch"))
                continue
            final_decision, final_metric, final_status = final_match_decision(
                result, order, verification
            )
            if final_decision != "MATCH" or final_metric != metric_id:
                unresolved.append((uid, f"finalist_not_strictly_verified:{final_status}"))
                continue
        if decision == "MATCH":
            if metric_id not in finalists[uid].get("rescue_proposed_metric_ids", []) and metric_id not in [
                row["metric_id"] for row in finalists[uid].get("candidates") or []
            ]:
                unresolved.append((uid, "finalist_match_outside_slate"))
                continue
        elif metric_id is not None:
            unresolved.append((uid, "metric_on_finalist_abstention"))
            continue
        replacement[uid] = result
        replacement_counts[f"finalist:{decision}"] += 1

    for uid, result in abstention_results.items():
        confirmed = result.get("confirmed_decision")
        if result.get("possible_exact_bank_match"):
            unresolved.append((uid, "possible_exact_bank_match"))
            continue
        if confirmed not in TYPED_DECISIONS:
            unresolved.append((uid, "unconfirmed_abstention"))
            continue
        if strict_production and (
            result.get("schema_version")
            != "silver-match-v3-two-order-abstention-verification-v1"
            or result.get("verification_orders") != ["original", "hashed"]
            or result.get("strict_two_order_abstention") is not True
            or int(result.get("rescue_coverage_repeats", 0)) < 2
            or result.get("rescue_reincludes_primary") is not True
        ):
            unresolved.append((uid, "abstention_not_repeated_full_bank_two_order"))
            continue
        if result.get("confidence") == "low":
            unresolved.append((uid, "low_confidence_abstention_verification"))
            continue
        if str(result.get("bank_source_sha256") or "") != str(
            no_match[uid].get("bank_source_sha256") or ""
        ):
            unresolved.append((uid, "abstention_bank_hash_mismatch"))
            continue
        replacement[uid] = {
            **result,
            "decision": confirmed,
            "metric_id": None,
            "candidate_bank_source_sha256": result["bank_source_sha256"],
            "candidate_ids": [],
        }
        replacement_counts[f"abstention:{confirmed}"] += 1
    manual_resolution_meta = None
    if unresolved and manual_unresolved_labels_path is not None:
        if manual_unresolved_validation_path is None or unresolved_output_path is None:
            raise ValueError(
                "manual unresolved resolution requires labels, validation, and the frozen unresolved ledger"
            )
        unresolved_by_uid = dict(unresolved)
        if len(unresolved_by_uid) != len(unresolved):
            raise ValueError("multiple unresolved reasons for one UID")
        if not unresolved_output_path.exists():
            raise FileNotFoundError(
                "manual unresolved resolution requires the first-pass unresolved ledger"
            )
        frozen_unresolved = _unique([unresolved_output_path], "frozen-unresolved")
        observed_reasons = {
            uid: str(row.get("unresolved_reason") or "")
            for uid, row in frozen_unresolved.items()
        }
        reconciliation_meta = None
        if observed_reasons != unresolved_by_uid:
            if unresolved_reconciliation_path is None:
                raise ValueError(
                    "frozen unresolved ledger differs from recomputed failures"
                )
            reconciliation_meta = _validate_unresolved_reconciliation(
                path=unresolved_reconciliation_path,
                frozen_path=unresolved_output_path,
                frozen=observed_reasons,
                recomputed=unresolved_by_uid,
                manual_labels_path=manual_unresolved_labels_path,
                manual_validation_path=manual_unresolved_validation_path,
            )
        elif unresolved_reconciliation_path is not None:
            raise ValueError(
                "unresolved reconciliation supplied without a live ledger discrepancy"
            )

        validation = json.loads(
            manual_unresolved_validation_path.read_text(encoding="utf-8")
        )
        label_sha = sha256_file(manual_unresolved_labels_path)
        if (
            validation.get("complete") is not True
            or int(validation.get("count", -1)) != len(unresolved_by_uid)
            or (validation.get("output") or {}).get("sha256") != label_sha
        ):
            raise ValueError("manual unresolved label validation is incomplete or unlinked")
        pack_ref = validation.get("pack_validation") or {}
        pack_validation_path = Path(str(pack_ref.get("path") or ""))
        if (
            not pack_validation_path.exists()
            or sha256_file(pack_validation_path) != pack_ref.get("sha256")
        ):
            raise ValueError("manual unresolved pack validation changed or is missing")
        pack_validation = json.loads(
            pack_validation_path.read_text(encoding="utf-8")
        )
        if (
            pack_validation.get("schema_version")
            != "silver-match-v3-unresolved-label-pack-v1"
            or pack_validation.get("truth_hidden") is not True
            or pack_validation.get("system_key_excluded_from_label_pack") is not True
            or pack_validation.get("permanently_excluded_from_gradients") is not True
            or ((pack_validation.get("inputs") or {}).get("unresolved") or {}).get(
                "sha256"
            )
            != sha256_file(unresolved_output_path)
        ):
            raise ValueError("manual unresolved labels are not from the frozen blind pack")

        manual = _unique([manual_unresolved_labels_path], "manual-unresolved-label")
        if set(manual) != set(unresolved_by_uid):
            raise ValueError("manual unresolved labels do not exactly cover frozen failures")
        manual_sources = {str(row.get("label_source") or "") for row in manual.values()}
        if MULTI_VOTE_LABEL_SOURCE in manual_sources:
            policy = validation.get("policy") or {}
            if (
                validation.get("schema_version") != MULTI_VOTE_SCHEMA
                or int((validation.get("unresolved") or {}).get("count", -1)) != 0
                or int(policy.get("minimum_independent_votes", -1)) < 2
                or policy.get("winner_must_be_unique") is not True
                or policy.get("source_confidences_are_preserved_not_upgraded") is not True
            ):
                raise ValueError("multi-vote manual resolution lacks a complete strict consensus gate")
        bank_cache: dict[str, tuple[str, set[str]]] = {}
        for uid, label in manual.items():
            task = str(label.get("task") or "")
            if task not in manifest.get("banks", {}):
                raise ValueError(f"manual unresolved task absent from manifest: {uid}")
            if task not in bank_cache:
                bank_meta = manifest["banks"][task]
                bank_path = Path(str(bank_meta["path"]))
                if not bank_path.is_absolute():
                    bank_path = manifest_path.parent / bank_path
                bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
                bank_cache[task] = (
                    str(bank_meta["source_sha256"]),
                    {str(row["metric_id"]) for row in bank_payload["metrics"]},
                )
            bank_sha, bank_ids = bank_cache[task]
            decision = str(label.get("decision") or "")
            metric_id = label.get("metric_id")
            confidence = str(label.get("confidence") or "")
            label_source = str(label.get("label_source") or "")
            legacy_decisive = (
                label_source == "independent_codex_full_bank" and confidence == "high"
            )
            consensus_decisive = (
                label_source == MULTI_VOTE_LABEL_SOURCE
                and confidence in {"high", "medium", "low"}
                and int(label.get("consensus_vote_count", 0)) >= 2
                and int(label.get("consensus_total_eligible_votes", 0))
                >= int(label.get("consensus_vote_count", 0))
                and len(label.get("consensus_vote_sources") or [])
                == int(label.get("consensus_vote_count", 0))
                and label.get("permanently_excluded_from_gradients") is True
                and label.get("training_eligible_preverification") is False
            )
            base = finalists.get(uid) or no_match.get(uid) or primary[uid]
            if (
                decision not in DECISIONS
                or not (legacy_decisive or consensus_decisive)
                or label.get("current_bank_source_sha256") != bank_sha
                or label.get("corpus") != base.get("corpus")
                or label.get("task") != base.get("task")
                or label.get("row") != base.get("row")
            ):
                raise ValueError(f"manual unresolved label lacks decisive blind provenance: {uid}")
            if decision == "MATCH":
                if str(metric_id) not in bank_ids:
                    raise ValueError(f"manual unresolved MATCH metric absent from bank: {uid}")
            elif metric_id is not None:
                raise ValueError(f"manual unresolved abstention carries metric: {uid}")
            replacement.pop(uid, None)
            replacement[uid] = {
                **label,
                "candidate_bank_source_sha256": bank_sha,
                "candidate_ids": [str(metric_id)] if decision == "MATCH" else [],
                "manual_unresolved_resolution": True,
                "manual_unresolved_validation_sha256": sha256_file(
                    manual_unresolved_validation_path
                ),
                "frozen_unresolved_sha256": sha256_file(unresolved_output_path),
            }
            replacement_counts[f"manual_unresolved:{decision}"] += 1
        manual_resolution_meta = {
            "labels": {
                "path": str(manual_unresolved_labels_path),
                "sha256": label_sha,
            },
            "validation": {
                "path": str(manual_unresolved_validation_path),
                "sha256": sha256_file(manual_unresolved_validation_path),
            },
            "pack_validation": {
                "path": str(pack_validation_path),
                "sha256": sha256_file(pack_validation_path),
            },
            "frozen_unresolved": {
                "path": str(unresolved_output_path),
                "sha256": sha256_file(unresolved_output_path),
            },
            "resolved_rows": len(manual),
            "accepted_label_sources": sorted(manual_sources),
            "unresolved_reason_taxonomy_reconciliation": reconciliation_meta,
            "required_evidence": (
                "high-confidence independent Codex label or provenance-bound unique "
                "multi-vote full-bank consensus"
            ),
        }
        unresolved = []

    if unresolved:
        if unresolved_output_path:
            rows = []
            seen_unresolved = set()
            for uid, reason in unresolved:
                if uid in seen_unresolved:
                    raise ValueError(f"multiple unresolved reasons for {uid}")
                seen_unresolved.add(uid)
                source = "finalist" if uid in finalists else "typed_abstention"
                base = finalists.get(uid) or no_match.get(uid) or primary[uid]
                rows.append(
                    {
                        "schema_version": "silver-match-v3-unresolved-rescue-v1",
                        "norm_uid": uid,
                        "corpus": base.get("corpus"),
                        "task": base.get("task"),
                        "row": base.get("row"),
                        "source": source,
                        "unresolved_reason": reason,
                        "bank_source_sha256": base.get("bank_source_sha256")
                        or base.get("candidate_bank_source_sha256"),
                    }
                )
            rows.sort(key=lambda row: (str(row["task"]), str(row["corpus"]), str(row["norm_uid"])))
            if unresolved_output_path.exists():
                existing = list(read_jsonl(unresolved_output_path))
                if existing != rows:
                    raise ValueError("existing unresolved ledger differs from recomputed failures")
            else:
                write_jsonl(unresolved_output_path, rows)
                unresolved_output_path.with_suffix(
                    unresolved_output_path.suffix + ".report.json"
                ).write_text(
                    json.dumps(
                        {
                            "schema_version": "silver-match-v3-unresolved-rescue-report-v1",
                            "manifest_sha256": sha256_file(manifest_path),
                            "count": len(rows),
                            "reason_counts": dict(
                                sorted(Counter(row["unresolved_reason"] for row in rows).items())
                            ),
                            "output_sha256": sha256_file(unresolved_output_path),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        raise ValueError(
            f"unresolved rescue outcomes: {len(unresolved)}; first={unresolved[:5]}"
        )
    if set(replacement) != rescued:
        raise AssertionError("resolved rescue replacement set is incomplete")

    output = []
    final_counts = Counter()
    for uid, row in primary.items():
        if uid not in replacement:
            merged = dict(row)
            merged["rescue_status"] = "NOT_APPLICABLE_PRIMARY_MATCH"
        else:
            result = replacement[uid]
            merged = dict(row)
            merged.update(
                {
                    "decision": result["decision"],
                    "metric_id": result.get("metric_id"),
                    "confidence": result.get("confidence"),
                    "reason": result.get("reason"),
                    "candidate_ids": result.get("candidate_ids") or [],
                    "candidate_bank_source_sha256": result.get(
                        "candidate_bank_source_sha256"
                    ),
                    "prompt_sha256": result.get("prompt_sha256"),
                    "model": result.get("model"),
                    "parse_error": result.get("parse_error"),
                    "rescue_status": (
                        "EXHAUSTIVE_RESCUE_MANUAL_RESOLVED"
                        if result.get("manual_unresolved_resolution")
                        else "EXHAUSTIVE_RESCUE_RESOLVED"
                    ),
                    "verification_status": (
                        "independent_blind_unresolved_resolution"
                        if result.get("manual_unresolved_resolution")
                        else (
                            "rescued_verified_exact_match"
                            if result["decision"] == "MATCH"
                            else "rescued_repeated_full_bank_typed_abstention"
                        )
                    ),
                    "pre_rescue": {
                        "decision": row.get("decision"),
                        "metric_id": row.get("metric_id"),
                        "confidence": row.get("confidence"),
                        "reason": row.get("reason"),
                    },
                    "rescue_resolution": result,
                }
            )
        final_counts[str(merged.get("decision"))] += 1
        output.append(merged)
    write_jsonl(output_path, output)
    report = {
        "schema_version": "silver-match-v3-rescue-merge-v1",
        "manifest_sha256": sha256_file(manifest_path),
        "primary_inputs": {str(path): sha256_file(path) for path in primary_paths},
        "finalist_candidate_inputs": {
            str(path): sha256_file(path) for path in finalist_candidate_paths
        },
        "finalist_adjudication_inputs": {
            str(path): sha256_file(path) for path in finalist_adjudication_paths
        },
        "finalist_order_inputs": {
            str(path): sha256_file(path) for path in (finalist_order_paths or [])
        },
        "finalist_verification_inputs": {
            str(path): sha256_file(path)
            for path in (finalist_verification_paths or [])
        },
        "no_match_audit_inputs": {
            str(path): sha256_file(path) for path in no_match_audit_paths
        },
        "abstention_verification_inputs": {
            str(path): sha256_file(path) for path in abstention_verification_paths
        },
        "primary_rows": len(primary),
        "rescued_rows": len(rescued),
        "replacement_counts": dict(sorted(replacement_counts.items())),
        "final_decision_counts": dict(sorted(final_counts.items())),
        "unresolved_rows": 0,
        "manual_unresolved_resolution": manual_resolution_meta,
        "strict_production": strict_production,
        "adjudicator_selection": (
            {
                "path": str(adjudicator_selection_path),
                "sha256": sha256_file(adjudicator_selection_path),
                "prompt_sha256": expected_adjudicator_prompt,
            }
            if adjudicator_selection_path
            else None
        ),
        "verifier_selection": (
            {
                "path": str(verifier_selection_path),
                "sha256": sha256_file(verifier_selection_path),
                "prompt_sha256": expected_verifier_prompt,
            }
            if verifier_selection_path
            else None
        ),
        "verifier_policy": (
            {
                "path": str(verifier_policy_path),
                "sha256": sha256_file(verifier_policy_path),
            }
            if verifier_policy_path
            else None
        ),
        "rescue_plan": (
            {
                "path": str(rescue_plan_path),
                "sha256": sha256_file(rescue_plan_path),
            }
            if rescue_plan_path
            else None
        ),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primary", action="append", required=True)
    parser.add_argument("--finalist-candidates", action="append", required=True)
    parser.add_argument("--finalist-adjudications", action="append", required=True)
    parser.add_argument("--finalist-order-check", action="append", default=[])
    parser.add_argument("--finalist-verification", action="append", default=[])
    parser.add_argument("--no-match-audits", action="append", required=True)
    parser.add_argument("--abstention-verifications", action="append", required=True)
    parser.add_argument("--adjudicator-selection")
    parser.add_argument("--verifier-selection")
    parser.add_argument("--verifier-policy")
    parser.add_argument("--rescue-plan")
    parser.add_argument("--unresolved-output")
    parser.add_argument("--manual-unresolved-labels")
    parser.add_argument("--manual-unresolved-validation")
    parser.add_argument("--unresolved-reconciliation")
    parser.add_argument("--strict-production", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = merge_rescue(
        manifest_path=Path(args.manifest).resolve(),
        primary_paths=[Path(path).resolve() for path in args.primary],
        finalist_candidate_paths=[Path(path).resolve() for path in args.finalist_candidates],
        finalist_adjudication_paths=[Path(path).resolve() for path in args.finalist_adjudications],
        finalist_order_paths=[Path(path).resolve() for path in args.finalist_order_check],
        finalist_verification_paths=[
            Path(path).resolve() for path in args.finalist_verification
        ],
        no_match_audit_paths=[Path(path).resolve() for path in args.no_match_audits],
        abstention_verification_paths=[Path(path).resolve() for path in args.abstention_verifications],
        output_path=Path(args.output).resolve(),
        adjudicator_selection_path=(
            Path(args.adjudicator_selection).resolve()
            if args.adjudicator_selection
            else None
        ),
        verifier_selection_path=(
            Path(args.verifier_selection).resolve()
            if args.verifier_selection
            else None
        ),
        verifier_policy_path=(
            Path(args.verifier_policy).resolve() if args.verifier_policy else None
        ),
        rescue_plan_path=(
            Path(args.rescue_plan).resolve() if args.rescue_plan else None
        ),
        unresolved_output_path=(
            Path(args.unresolved_output).resolve() if args.unresolved_output else None
        ),
        manual_unresolved_labels_path=(
            Path(args.manual_unresolved_labels).resolve()
            if args.manual_unresolved_labels
            else None
        ),
        manual_unresolved_validation_path=(
            Path(args.manual_unresolved_validation).resolve()
            if args.manual_unresolved_validation
            else None
        ),
        unresolved_reconciliation_path=(
            Path(args.unresolved_reconciliation).resolve()
            if args.unresolved_reconciliation
            else None
        ),
        strict_production=args.strict_production,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
