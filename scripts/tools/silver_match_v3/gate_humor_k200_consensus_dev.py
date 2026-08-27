#!/usr/bin/env python3
"""Gate Humor K200 retrieval on untouched exact-consensus development truth.

This is a CPU-only, create-only bridge between truth consensus and the final
training handoff.  It extracts only frozen development MATCH rows, audits K200
capture with an exact one-sided Clopper-Pearson bound, and routes every miss to
the independently frozen full-bank candidate artifact.  A passing run also
freezes the diverse candidate bundle consumed by the final CE/Gemma builders.

The module never trains, scores a model, or reads test/blind outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from .audit_candidate_capture import audit_candidate_capture
from .audit_false_abstentions import clopper_pearson_upper
from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .freeze_humor_final_stack_handoff import _bank_source_hash


TASK = "humor"
SCHEMA = "silver-match-v3-humor-k200-untouched-dev-gate-v1"
PASS_STATUS = "K200_UNTOUCHED_DEV_GATE_PASSED"
FAIL_STATUS = "K200_UNTOUCHED_DEV_GATE_FAILED_FULLBANK_RESCUE_REQUIRED"
BUNDLE_SCHEMA = "silver-match-v3-candidate-capture-sequence-v1"
COMPONENT_DEPTHS = (1, 2, 5, 10, 20, 30, 40, 50)


def _ref(path: Path, *, count: int | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    ref: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if count is not None:
        ref["count"] = count
    return ref


def _write_json_create(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _manifest_dev_truth(
    manifest_path: Path, dev_truth_path: Path, *, bank_hash: str
) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    dev_truth_path = dev_truth_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = (manifest.get("outputs") or {}).get("dev") or {}
    output_path = Path(str(output.get("path") or ""))
    if not output_path.is_absolute():
        output_path = (manifest_path.parent / output_path).resolve()
    else:
        output_path = output_path.resolve()
    rows = list(read_jsonl(dev_truth_path))
    if (
        manifest.get("schema_version")
        != "silver-match-v3-consensus-training-truth-manifest-v1"
        or manifest.get("status")
        != "COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS"
        or manifest.get("task") != TASK
        or int(manifest.get("source_group_cross_split_count", -1)) != 0
        or int(manifest.get("blind_rows_training_eligible", -1)) != 0
        or output_path != dev_truth_path
        or output.get("sha256") != sha256_file(dev_truth_path)
        or int(output.get("count", -1)) != len(rows)
    ):
        raise ValueError("exact Humor consensus manifest/truth contract differs")
    seen: set[str] = set()
    groups: dict[str, str] = {}
    for row in rows:
        uid = normalize_space(row.get("norm_uid"))
        split = normalize_space(row.get("split"))
        role = normalize_space(row.get("collection_role"))
        supplied_hash = normalize_space(
            row.get("current_bank_source_sha256")
            or row.get("bank_source_sha256")
        )
        group = normalize_space(row.get("source_group") or row.get("split_group"))
        if (
            not uid
            or uid in seen
            or row.get("task") != TASK
            or supplied_hash != bank_hash
            or role != "dev"
            or split != "dev"
            or not group
        ):
            raise ValueError(f"invalid/duplicate exact consensus row: {uid}")
        seen.add(uid)
        prior = groups.setdefault(group, role)
        if prior != role:
            raise ValueError(f"consensus source group crosses roles: {group}")
    return rows


def _dev_matches(rows: list[dict[str, Any]], *, bank_hash: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if normalize_space(row.get("collection_role")) != "dev":
            continue
        if row.get("dev_selection_eligible") is not True:
            raise ValueError("frozen dev truth lacks dev-selection eligibility")
        if row.get("training_eligible") is True or row.get("blind_evaluation_only") is True:
            raise ValueError("frozen dev truth escaped its role firewall")
        if normalize_space(row.get("decision")) != "MATCH":
            continue
        metric_id = normalize_space(row.get("metric_id"))
        if not metric_id:
            raise ValueError("dev MATCH lacks exact metric_id")
        output.append(
            {
                **row,
                "split": "dev",
                "collection_role": "dev",
                "current_bank_source_sha256": bank_hash,
                "gate_use": "untouched_development_retrieval_capture_only",
            }
        )
    output.sort(key=lambda row: normalize_space(row["norm_uid"]))
    if not output:
        raise ValueError("exact consensus contains no untouched dev MATCH truth")
    return output


def _candidate_rows(path: Path, wanted: set[str]) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for row in read_jsonl(path):
        uid = normalize_space(row.get("norm_uid"))
        if uid not in wanted:
            continue
        if uid in seen:
            raise ValueError(f"duplicate candidate norm_uid: {uid}")
        seen.add(uid)
        yield row
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(f"candidate artifact misses dev truth UIDs: {missing[:5]}")


def _component_ids(candidate: Mapping[str, Any], depth: int) -> set[str]:
    selected = set()
    for row in candidate.get("candidates") or []:
        ranks = row.get("lane_ranks") or {}
        if any(
            value is not None and int(value) <= depth
            for value in ranks.values()
        ):
            selected.add(normalize_space(row.get("metric_id")))
    return selected


def _progressive_policies(
    labels: list[dict[str, Any]], k200_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    label_by_uid = {normalize_space(row["norm_uid"]): row for row in labels}
    candidates = {
        normalize_space(row["norm_uid"]): row
        for row in _candidate_rows(k200_path, set(label_by_uid))
    }
    # The policy grid is fixed before looking at dev.  Bonferroni bounds make
    # selecting the first passing member a simultaneous one-sided 95% claim.
    policy_count = len(COMPONENT_DEPTHS) + 1
    simultaneous_alpha = 0.05 / policy_count
    policies = []
    for depth in COMPONENT_DEPTHS:
        misses = 0
        sizes = []
        for uid, label in label_by_uid.items():
            ids = _component_ids(candidates[uid], depth)
            sizes.append(len(ids))
            misses += int(normalize_space(label["metric_id"]) not in ids)
        policies.append(
            {
                "name": f"component_union_depth_{depth}",
                "kind": "component_union",
                "component_depth": depth,
                "gold_matches": len(labels),
                "capture_count": len(labels) - misses,
                "capture_rate": (len(labels) - misses) / len(labels),
                "miss_count": misses,
                "miss_rate": misses / len(labels),
                "miss_upper_bound_pointwise_one_sided_95": clopper_pearson_upper(
                    misses, len(labels), alpha=0.05
                ),
                "miss_upper_bound_simultaneous_one_sided_95": clopper_pearson_upper(
                    misses, len(labels), alpha=simultaneous_alpha
                ),
                "candidate_union_size": {
                    "min": min(sizes),
                    "max": max(sizes),
                    "mean": sum(sizes) / len(sizes),
                },
            }
        )
    fused_misses = 0
    fused_sizes = []
    for uid, label in label_by_uid.items():
        ids = {
            normalize_space(row.get("metric_id"))
            for row in (candidates[uid].get("candidates") or [])[:200]
        }
        fused_sizes.append(len(ids))
        fused_misses += int(normalize_space(label["metric_id"]) not in ids)
    policies.append(
        {
            "name": "fused_k200",
            "kind": "fused_rank_prefix",
            "fused_k": 200,
            "gold_matches": len(labels),
            "capture_count": len(labels) - fused_misses,
            "capture_rate": (len(labels) - fused_misses) / len(labels),
            "miss_count": fused_misses,
            "miss_rate": fused_misses / len(labels),
            "miss_upper_bound_pointwise_one_sided_95": clopper_pearson_upper(
                fused_misses, len(labels), alpha=0.05
            ),
            "miss_upper_bound_simultaneous_one_sided_95": clopper_pearson_upper(
                fused_misses, len(labels), alpha=simultaneous_alpha
            ),
            "candidate_union_size": {
                "min": min(fused_sizes),
                "max": max(fused_sizes),
                "mean": sum(fused_sizes) / len(fused_sizes),
            },
        }
    )
    for policy in policies:
        policy["target_upper_bound"] = 0.05
        policy["simultaneous_family_size"] = policy_count
        policy["passed_simultaneous_gate"] = (
            policy["miss_upper_bound_simultaneous_one_sided_95"] < 0.05
        )
    selected = next(
        (
            policy
            for policy in policies
            if policy["kind"] == "component_union"
            and policy["passed_simultaneous_gate"]
        ),
        None,
    )
    if selected is not None:
        selected["deployment_gate_method"] = (
            "predeclared_component_grid_bonferroni_simultaneous_one_sided_95"
        )
        selected["passed_deployment_gate"] = True
    else:
        fused = policies[-1]
        fused["passed_fixed_fallback_gate"] = (
            fused["miss_upper_bound_pointwise_one_sided_95"] < 0.05
        )
        if fused["passed_fixed_fallback_gate"]:
            fused["deployment_gate_method"] = (
                "predeclared_fixed_fused_k200_fallback_pointwise_one_sided_95"
            )
            fused["passed_deployment_gate"] = True
            selected = fused
    return policies, selected


def _route_fullbank_misses(
    labels: list[dict[str, Any]],
    k200_path: Path,
    fullbank_path: Path,
    *,
    selected_policy: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    label_by_uid = {normalize_space(row["norm_uid"]): row for row in labels}
    wanted = set(label_by_uid)
    k200 = {normalize_space(row["norm_uid"]): row for row in _candidate_rows(k200_path, wanted)}
    fullbank = {
        normalize_space(row["norm_uid"]): row
        for row in _candidate_rows(fullbank_path, wanted)
    }
    misses = []
    for uid in sorted(wanted):
        gold = normalize_space(label_by_uid[uid]["metric_id"])
        if (selected_policy or {}).get("kind") == "component_union":
            primary_ids = _component_ids(
                k200[uid], int(selected_policy["component_depth"])
            )
            policy_name = str(selected_policy["name"])
        else:
            primary_ids = {
                normalize_space(row.get("metric_id"))
                for row in (k200[uid].get("candidates") or [])[:200]
            }
            policy_name = "fused_k200"
        if gold in primary_ids:
            continue
        rescue = [
            (rank, row)
            for rank, row in enumerate(fullbank[uid].get("candidates") or [], 1)
            if normalize_space(row.get("metric_id")) == gold
        ]
        if len(rescue) != 1:
            raise ValueError(f"full-bank rescue does not contain gold exactly once: {uid}/{gold}")
        rank, candidate = rescue[0]
        misses.append(
            {
                "schema_version": "silver-match-v3-humor-k200-dev-miss-rescue-v1",
                "task": TASK,
                "split": "dev",
                "norm_uid": uid,
                "metric_id": gold,
                "primary_policy": policy_name,
                "primary_policy_captured": False,
                "rescue": "fullbank285",
                "fullbank_rank": rank,
                "fullbank_candidate": candidate,
            }
        )
    return misses


def _freeze_candidate_bundle(
    *,
    prior_path: Path,
    k200_path: Path,
    output_path: Path,
    gate_report_path: Path,
) -> dict[str, Any]:
    prior_path = prior_path.resolve()
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    inputs = dict(prior.get("candidate_inputs") or {})
    available = list(prior.get("available_lanes") or [])
    selected = list(prior.get("selected_sequence") or [])
    if (
        prior.get("schema_version") != BUNDLE_SCHEMA
        or prior.get("selection_split") != "dev"
        or prior.get("test_labels_used_for_selection") is not False
        or len(inputs) < 2
        or not available
        or not selected
    ):
        raise ValueError("prior diverse candidate bundle contract differs")
    for name, ref in inputs.items():
        path = Path(str((ref or {}).get("path") or "")).resolve()
        if sha256_file(path) != (ref or {}).get("sha256"):
            raise ValueError(f"prior candidate bundle input changed: {name}")
    if "production_k200" in inputs:
        raise ValueError("prior candidate bundle already has production_k200")
    inputs["production_k200"] = {
        "path": str(k200_path.resolve()),
        "sha256": sha256_file(k200_path),
    }
    available.append("production_k200:rank")
    selected.append("production_k200:rank")
    gate = json.loads(gate_report_path.read_text(encoding="utf-8"))
    gate_passed = (gate.get("gate") or {}).get("passed") is True
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "status": (
            "FROZEN_DIVERSE_BUNDLE_PLUS_UNTOUCHED_DEV_VALIDATED_K200"
            if gate_passed
            else "FROZEN_DIVERSE_BUNDLE_K200_NOT_PROMOTED_FULLBANK_REQUIRED"
        ),
        "task": TASK,
        "selection_split": "dev",
        "test_labels_used_for_selection": False,
        "candidate_inputs": inputs,
        "available_lanes": available,
        "selected_sequence": selected,
        "prior_bundle": _ref(prior_path),
        "k200_dev_gate": _ref(gate_report_path),
        "k200_primary_promoted": gate_passed,
        "fullbank_required_for_production": not gate_passed,
        "full_candidate_inputs_required_by_final_builders": True,
    }
    _write_json_create(output_path, payload)
    return payload


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    bank_path = Path(args.bank).resolve()
    dev_truth_path = Path(args.consensus_dev_truth).resolve()
    manifest_path = Path(args.consensus_manifest).resolve()
    k200_path = Path(args.k200_candidates).resolve()
    fullbank_path = Path(args.fullbank_candidates).resolve()
    labels_path = Path(args.dev_match_labels).resolve()
    capture_path = Path(args.capture_report).resolve()
    misses_path = Path(args.rescue_misses).resolve()
    gate_path = Path(args.gate_report).resolve()
    bundle_path = Path(args.candidate_bundle_output).resolve()
    outputs = (labels_path, capture_path, misses_path, gate_path, bundle_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite Humor dev-gate outputs")

    bank_hash = _bank_source_hash(bank_path)
    consensus = _manifest_dev_truth(
        manifest_path, dev_truth_path, bank_hash=bank_hash
    )
    labels = _dev_matches(consensus, bank_hash=bank_hash)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(labels_path, labels)
    capture = audit_candidate_capture(
        [labels_path],
        [k200_path],
        k=200,
        alpha=0.05,
        target=0.05,
        allow_prefix_missing_gold=True,
    )
    _write_json_create(capture_path, capture)
    policies, selected_policy = _progressive_policies(labels, k200_path)
    misses = _route_fullbank_misses(
        labels,
        k200_path,
        fullbank_path,
        selected_policy=selected_policy,
    )
    write_jsonl(misses_path, misses)
    dev = (capture.get("groups") or {}).get("task_split:humor:dev") or {}
    fused_passed = (
        dev.get("under_target_supported") is True
        and float(dev.get("union_miss_upper_bound", 1.0)) < 0.05
    )
    selected_miss_count = (
        int(selected_policy["miss_count"]) if selected_policy is not None else -1
    )
    passed = (
        fused_passed
        and selected_policy is not None
        and selected_miss_count == len(misses)
    )
    selected_component_depth = (
        int(selected_policy["component_depth"])
        if selected_policy is not None
        and selected_policy.get("kind") == "component_union"
        else None
    )
    progressive_depths = [
        depth
        for depth in COMPONENT_DEPTHS
        if selected_component_depth is not None and depth <= selected_component_depth
    ]
    report = {
        "schema_version": SCHEMA,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "task": TASK,
        "selection_role": "untouched_development_only",
        "test_or_blind_labels_used_for_policy_selection": False,
        "training_labels_used_for_promotion": False,
        "inputs": {
            "bank": _ref(bank_path),
            "consensus_manifest": _ref(manifest_path),
            "consensus_dev_truth": _ref(dev_truth_path, count=len(consensus)),
            "k200_candidates": _ref(k200_path),
            "fullbank285_candidates": _ref(fullbank_path),
        },
        "outputs": {
            "dev_match_labels": _ref(labels_path, count=len(labels)),
            "capture_report": _ref(capture_path),
            "fullbank_rescue_misses": _ref(misses_path, count=len(misses)),
        },
        "gate": {
            "gold_matches": int(dev.get("gold_matches", -1)),
            "k200_capture_count": int(dev.get("union_capture_count", -1)),
            "k200_capture_rate": dev.get("union_capture_rate"),
            "k200_miss_count": len(misses),
            "k200_miss_upper_bound_one_sided_95": dev.get("union_miss_upper_bound"),
            "target_upper_bound": 0.05,
            "passed": passed,
        },
        "progressive_policy_selection": {
            "predeclared_component_depths": list(COMPONENT_DEPTHS),
            "fused_k200_final_candidate": True,
            "multiple_policy_correction": "bonferroni_simultaneous_one_sided_95",
            "policies": policies,
            "selected_policy": selected_policy,
            "selected_before_test_or_blind_opening": True,
            "passed": selected_policy is not None,
        },
        "production_scoring_contract": {
            "progressive_component_trial_depths": progressive_depths,
            "maximum_authorized_primary_policy": (
                selected_policy.get("name") if selected_policy else None
            ),
            "score_all_fused_k200_pairs_unconditionally": False,
            "pre_ce_early_stopping_authorized": False,
            "ce_confidence_early_stopping_requires_separate_untouched_dev_audit": True,
            "fullbank285_rescue_required_for_primary_abstentions": True,
        },
        "rescue": {
            "strategy": "exact_fullbank285_for_every_selected_primary_policy_miss",
            "misses_routed": len(misses),
            "misses_not_found_in_fullbank": 0,
            "actionable_delta": (
                "promote_selected_progressive_primary_with_fullbank285_abstention_rescue"
                if passed
                else "do_not_promote_primary; score_or_label exact fused-K200 misses from fullbank285"
            ),
        },
        "gpu_processes_launched": 0,
        "training_or_model_scoring_executed": False,
        "release_ready": False,
    }
    _write_json_create(gate_path, report)
    # The candidate bundle is also a training-data input.  A failed production
    # retrieval gate must prevent K200 promotion, but must not silently block
    # task-local CE training on the already frozen truth.  The bundle carries
    # an explicit full-bank-required state which downstream release code can
    # distinguish from a production-validated K200 policy.
    _freeze_candidate_bundle(
        prior_path=Path(args.prior_candidate_bundle),
        k200_path=k200_path,
        output_path=bundle_path,
        gate_report_path=gate_path,
    )
    report["candidate_bundle"] = _ref(bundle_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--consensus-dev-truth", required=True)
    parser.add_argument("--consensus-manifest", required=True)
    parser.add_argument("--k200-candidates", required=True)
    parser.add_argument("--fullbank-candidates", required=True)
    parser.add_argument("--prior-candidate-bundle", required=True)
    parser.add_argument("--dev-match-labels", required=True)
    parser.add_argument("--capture-report", required=True)
    parser.add_argument("--rescue-misses", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument("--candidate-bundle-output", required=True)
    args = parser.parse_args()
    report = run_gate(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
