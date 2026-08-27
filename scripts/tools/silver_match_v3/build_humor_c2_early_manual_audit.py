#!/usr/bin/env python3
"""Build a deterministic truth-blind audit packet from the first sealed c2 shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .common import read_jsonl, sha256_file
from .finalize_humor_c2_full285_deployment import (
    CE_MARGIN_THRESHOLD,
    DEPLOYMENT_CLAIM,
    HYBRID_THRESHOLD,
)
from .merge_package_humor_full285_ce import PACKAGE_SCHEMA, PROMPT_SCHEMA
from .run_humor_c2_production_paired_vllm import META_SCHEMA, PREDICTION_SCHEMA


PACKET_SCHEMA = "silver-match-v3-humor-c2-early-manual-audit-row-v1"
REPORT_SCHEMA = "silver-match-v3-humor-c2-early-manual-audit-report-v1"
SELECTION_SEED = "humor-c2-early-audit-v1"
EXPECTED_BANK = 285
TARGETS = {
    "hybrid_accepted_match": 8,
    "stable_typed_abstention": 5,
    "order_disagreement_or_context_needed": 5,
    "ce_multi_positive": 4,
    "all_zero_full285": 4,
    "k85_only_case": 4,
}
CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def artifact(path: Path, **extra: Any) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, **extra}


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def _prompt_text(row: Mapping[str, Any]) -> tuple[str, str]:
    conversation = row.get("conversation") or []
    if len(conversation) != 1 or conversation[0].get("role") != "user":
        raise ValueError("unexpected production conversation")
    text = str(conversation[0].get("content") or "")
    statement_marker = "HUMAN STATEMENT (verbatim):\n"
    context_marker = "\nCONTEXT (capped at 600 characters):\n"
    cards_marker = "\n\nCANDIDATE METRIC CARDS (no examples):\n"
    if statement_marker not in text or context_marker not in text or cards_marker not in text:
        raise ValueError("production prompt text markers differ")
    statement_and_rest = text.split(statement_marker, 1)[1]
    statement, context_and_rest = statement_and_rest.split(context_marker, 1)
    context = context_and_rest.split(cards_marker, 1)[0]
    return statement, context


def _policy(package: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    ldecision, rdecision = str(left.get("decision")), str(right.get("decision"))
    lmetric, rmetric = left.get("metric_id"), right.get("metric_id")
    lconf, rconf = str(left.get("confidence")), str(right.get("confidence"))
    if lconf not in CONF_RANK or rconf not in CONF_RANK:
        raise ValueError("invalid typed confidence")
    confidence = lconf if CONF_RANK[lconf] <= CONF_RANK[rconf] else rconf
    ce_metric = str(package["ce_top1_metric_id"])
    ce_score = float(package["ce_top1_exact_probability"])
    ce_margin = float(package["ce_top_margin"])
    stable_match = ldecision == rdecision == "MATCH" and lmetric is not None and lmetric == rmetric
    accepted = bool(
        stable_match and confidence == "high" and lmetric == ce_metric
        and ce_score >= HYBRID_THRESHOLD and ce_margin >= CE_MARGIN_THRESHOLD
    )
    if accepted:
        decision, metric_id, status = "MATCH", lmetric, DEPLOYMENT_CLAIM
    elif "INVALID_OUTPUT" in {ldecision, rdecision}:
        decision, metric_id, confidence, status = "INVALID_OUTPUT", None, "low", "PRODUCTION_C2_INVALID_OUTPUT"
    elif ldecision == rdecision and ldecision != "MATCH":
        decision, metric_id, status = ldecision, None, "PRODUCTION_C2_STABLE_PAIRED_ABSTENTION"
    else:
        decision, metric_id, confidence, status = "UNSTABLE_MATCH", None, "low", "PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE"
    return {
        "decision": decision, "metric_id": metric_id, "confidence": confidence,
        "verification_status": status, "accepted_by_frozen_hybrid": accepted,
        "frozen_rule": {
            "same_exact_leaf_both_orders": stable_match,
            "minimum_typed_confidence": "high", "ce_top1_must_equal_typed_leaf": True,
            "minimum_ce_exact_probability": HYBRID_THRESHOLD,
            "minimum_ce_top_margin": CE_MARGIN_THRESHOLD,
        },
    }


def _stable_abstention(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    decision = str(left.get("decision"))
    return decision == str(right.get("decision")) and decision not in {"MATCH", "INVALID_OUTPUT", "CONTEXT_NEEDED"}


def _disagreement_or_context(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        str(left.get("decision")) != str(right.get("decision"))
        or left.get("metric_id") != right.get("metric_id")
        or "CONTEXT_NEEDED" in {str(left.get("decision")), str(right.get("decision"))}
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    package_path, prompts_path, bank_path, typed_root, output_root = map(
        lambda value: Path(value).resolve(),
        (args.candidate_package, args.prompts, args.bank, args.typed_root, args.output_root),
    )
    if output_root.exists():
        raise FileExistsError(output_root)
    meta_path = typed_root / "INFERENCE_META.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("schema_version") != META_SCHEMA
        or meta.get("status") != "COMPLETE_C2_PRODUCTION_PAIRED_INFERENCE"
        or meta.get("deployment_claim") != DEPLOYMENT_CLAIM
        or int(meta.get("test_or_blind_rows_read", -1)) != 0
    ):
        raise ValueError("typed shard is not a sealed truth-blind production shard")
    typed: dict[str, dict[str, dict[str, Any]]] = {}
    for order in ("original", "reordered"):
        path = typed_root / f"typed.{order}.jsonl"
        if (meta.get("outputs") or {}).get(order, {}).get("sha256") != sha256_file(path):
            raise ValueError("typed output SHA differs from sealed meta")
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if (
                row.get("schema_version") != PREDICTION_SCHEMA or row.get("split") != "production"
                or row.get("order_mode") != order or not uid or order in typed.get(uid, {})
            ):
                raise ValueError(f"invalid typed row: {uid}/{order}")
            typed.setdefault(uid, {})[order] = row
    if not typed or any(set(rows) != {"original", "reordered"} for rows in typed.values()):
        raise ValueError("typed shard paired coverage differs")

    packages: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(package_path):
        uid = str(row.get("norm_uid") or "")
        if uid in typed:
            if row.get("schema_version") != PACKAGE_SCHEMA or uid in packages:
                raise ValueError(f"invalid/duplicate candidate package row: {uid}")
            packages[uid] = row
    prompts: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(prompts_path):
        uid = str(row.get("norm_uid") or "")
        if uid in typed and row.get("order_mode") == "original":
            if row.get("schema_version") != PROMPT_SCHEMA or uid in prompts:
                raise ValueError(f"invalid/duplicate prompt row: {uid}")
            prompts[uid] = row
    if set(packages) != set(typed) or set(prompts) != set(typed):
        raise ValueError("audit inputs differ on typed shard coverage")

    bank_doc = json.loads(bank_path.read_text(encoding="utf-8"))
    bank = {str(row["metric_id"]): row for row in bank_doc.get("metrics") or []}
    if len(bank) != EXPECTED_BANK:
        raise ValueError("Humor bank cardinality differs")

    records: dict[str, dict[str, Any]] = {}
    memberships: dict[str, set[str]] = {}
    for uid in sorted(typed):
        package = packages[uid]; left, right = typed[uid]["original"], typed[uid]["reordered"]
        policy = _policy(package, left, right)
        positives = list(package.get("ce_positive_metric_ids") or [])
        top1_source = str(package.get("ce_top1_surface_source") or "")
        memberships[uid] = set()
        if policy["accepted_by_frozen_hybrid"]: memberships[uid].add("hybrid_accepted_match")
        if _stable_abstention(left, right): memberships[uid].add("stable_typed_abstention")
        if _disagreement_or_context(left, right): memberships[uid].add("order_disagreement_or_context_needed")
        if len(positives) >= 2: memberships[uid].add("ce_multi_positive")
        if not positives: memberships[uid].add("all_zero_full285")
        if top1_source == "k85": memberships[uid].add("k85_only_case")
        statement, context = _prompt_text(prompts[uid])
        candidates = []
        for candidate in package.get("candidates") or []:
            metric_id = str(candidate.get("metric_id") or "")
            card = bank.get(metric_id)
            if card is None:
                raise ValueError(f"candidate absent from bank: {metric_id}")
            candidates.append({
                **candidate,
                "metric_name": card.get("name"),
                "metric_description": card.get("description"),
            })
        records[uid] = {
            "schema_version": PACKET_SCHEMA, "task": "humor", "corpus": "humor_multi",
            "norm_uid": uid, "source_group": package.get("source_group"),
            "truth_data_read": False, "gold_fields_present": False,
            "human_statement_verbatim": statement,
            "context_verbatim_capped": context,
            "ce_full285_summary": {
                "bank_cardinality": EXPECTED_BANK,
                "top1_metric_id": package.get("ce_top1_metric_id"),
                "top1_exact_probability": package.get("ce_top1_exact_probability"),
                "top1_surface_source": top1_source,
                "top2_exact_probability": package.get("ce_top2_exact_probability"),
                "top_margin": package.get("ce_top_margin"),
                "frozen_positive_threshold": package.get("ce_positive_threshold"),
                "positive_metric_ids": positives,
                "positive_count": len(positives),
                "all_zero_at_frozen_threshold": not positives,
                "ranked_top16_plus_all_positives": candidates,
            },
            "typed_original_raw": {key: left.get(key) for key in ("decision", "metric_id", "confidence", "reason", "parse_error", "raw_response")},
            "typed_reordered_raw": {key: right.get(key) for key in ("decision", "metric_id", "confidence", "reason", "parse_error", "raw_response")},
            "final_frozen_policy_result": policy,
            "provenance": {
                "deployment_claim": DEPLOYMENT_CLAIM,
                "typed_shard_id": meta.get("shard_id"), "typed_num_shards": meta.get("num_shards"),
                "selection_seed": SELECTION_SEED, "test_or_blind_rows_read": 0,
            },
        }

    selected: list[dict[str, Any]] = []; seen: set[str] = set(); counts: Counter[str] = Counter()
    eligible_counts = {category: sum(category in values for values in memberships.values()) for category in TARGETS}
    for category, target in TARGETS.items():
        eligible = sorted(
            (uid for uid, values in memberships.items() if category in values and uid not in seen),
            key=lambda uid: hashlib.sha256(f"{SELECTION_SEED}\0{category}\0{uid}".encode()).hexdigest(),
        )
        for uid in eligible[:target]:
            seen.add(uid); counts[category] += 1
            selected.append({**records[uid], "audit_stratum": category, "audit_stratum_index": counts[category] - 1})

    output_root.mkdir(parents=True, exist_ok=False)
    packet_path = output_root / "EARLY_MANUAL_AUDIT_PACKET.jsonl"
    temporary = output_root / f".{packet_path.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("xb") as handle:
            for row in selected:
                handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, packet_path)
    except BaseException:
        temporary.unlink(missing_ok=True); raise
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE_TRUTH_BLIND_FIRST_SEALED_SHARD_AUDIT_PACKET",
        "task": "humor", "corpus": "humor_multi", "deployment_claim": DEPLOYMENT_CLAIM,
        "test_or_blind_or_truth_rows_read": 0, "gold_fields_written": 0,
        "selection_seed": SELECTION_SEED, "first_sealed_shard": meta.get("shard_id"),
        "targets": TARGETS, "eligible_counts": eligible_counts,
        "selected_counts": dict(sorted(counts.items())), "selected_total": len(selected),
        "selection_is_unique_across_strata": len(seen) == len(selected),
        "inputs": {
            "candidate_package": artifact(package_path), "prompts": artifact(prompts_path),
            "bank": artifact(bank_path), "typed_meta": artifact(meta_path),
        },
        "output": artifact(packet_path, rows=len(selected)),
    }
    _write_json_new(output_root / "REPORT.json", report)
    print(json.dumps(report, sort_keys=True)); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-package", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--typed-root", required=True)
    parser.add_argument("--output-root", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
