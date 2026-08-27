#!/usr/bin/env python3
"""Freeze clean typed Humor SFT examples for one task-specific Gemma-4 LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adjudicate_gemma import build_item_prompt
from .common import read_jsonl, sha256_file, write_jsonl


TASK = "humor"
BANK_SHA = "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9"
DECISIONS = {
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}


def _ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if not rows or "" in uids or len(uids) != len(set(uids)):
        raise ValueError(f"empty/missing/duplicate UIDs: {path}")
    return rows


def _support_reason(row: dict[str, Any]) -> str:
    decision = str(row["decision"])
    metric_id = row.get("metric_id")
    predictions = row.get("source_predictions") or {}
    for name in row.get("agreement_sources") or []:
        prediction = predictions.get(name) or {}
        if prediction.get("decision") == decision and prediction.get("metric_id") == metric_id:
            reason = str(prediction.get("reason") or "").strip()
            if reason:
                return reason
    reason = str(row.get("reason") or "").strip()
    if not reason:
        raise ValueError(f"truth row lacks a target reason: {row['norm_uid']}")
    return reason


def _hash_order(uid: str, metric_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{uid}\x1f{metric_id}".encode()).hexdigest()


def _insert_at_hash_position(values: list[str], value: str, uid: str) -> list[str]:
    if value in values:
        return values
    position = int(hashlib.sha256(f"inject\x1f{uid}\x1f{value}".encode()).hexdigest()[:8], 16)
    position %= len(values) + 1
    return values[:position] + [value] + values[position:]


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {
        name: Path(value).resolve()
        for name, value in vars(args).items()
        if name not in {"max_candidates", "order_seed", "output", "report"}
    }
    bank_payload = json.loads(paths["bank"].read_text(encoding="utf-8"))
    if bank_payload.get("task") != TASK or bank_payload.get("source_sha256") != BANK_SHA:
        raise ValueError("unexpected Humor bank")
    metrics = list(bank_payload.get("metrics") or [])
    metric_by_id = {str(row["metric_id"]): row for row in metrics}
    if len(metric_by_id) != 285 or len(metric_by_id) != len(metrics):
        raise ValueError("Humor bank is not the exact 285-leaf bank")

    norm_rows = _rows(paths["norms"])
    norms = {str(row["norm_uid"]): row for row in norm_rows}
    optimize = _rows(paths["optimize_truth"])
    historical = _rows(paths["historical_truth"])
    truth_rows = optimize + historical
    if len(optimize) != 296 or len(historical) != 600 or len(truth_rows) != 896:
        raise ValueError("expected exact 296 optimize + 600 historical clean truth")
    truth = {str(row["norm_uid"]): row for row in truth_rows}
    if len(truth) != 896 or set(truth) - set(norms):
        raise ValueError("clean truth does not map one-to-one into strict norms")
    decisions = Counter(str(row.get("decision") or "") for row in truth_rows)
    if set(decisions) - DECISIONS or decisions["MATCH"] != 388:
        raise ValueError(f"unexpected clean truth decisions: {dict(decisions)}")
    for row in truth_rows:
        if row.get("task") != TASK or row.get("current_bank_source_sha256") != BANK_SHA:
            raise ValueError(f"truth task/bank mismatch: {row['norm_uid']}")
        if row["decision"] == "MATCH":
            if str(row.get("metric_id") or "") not in metric_by_id:
                raise ValueError(f"MATCH metric absent from bank: {row['norm_uid']}")
        elif row.get("metric_id") is not None:
            raise ValueError(f"typed nonmatch carries metric: {row['norm_uid']}")

    fresh = _rows(paths["fresh_select_identities"])
    fresh_freeze = json.loads(paths["fresh_select_freeze"].read_text(encoding="utf-8"))
    if (
        fresh_freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or int(fresh_freeze.get("selected_count", -1)) != 300
        or ((fresh_freeze.get("outputs") or {}).get("identities") or {}).get("sha256")
        != sha256_file(paths["fresh_select_identities"])
    ):
        raise ValueError("fresh select firewall was not frozen before labeling")
    truth_groups = {str(row.get("source_group") or "") for row in truth_rows}
    fresh_uids = {str(row["norm_uid"]) for row in fresh}
    fresh_groups = {str(row.get("source_group") or "") for row in fresh}
    if set(truth) & fresh_uids or truth_groups & fresh_groups:
        raise ValueError("Gemma clean truth overlaps fresh select")

    candidate_rows = _rows(paths["candidates"])
    candidates = {str(row["norm_uid"]): row for row in candidate_rows}
    if set(candidates) != set(truth):
        raise ValueError("frozen candidate rows do not exactly cover clean truth")
    for uid, row in candidates.items():
        ids = [str(value["metric_id"]) for value in row.get("candidates") or []]
        if (
            row.get("bank_source_sha256") != BANK_SHA
            or len(ids) != 50
            or len(set(ids)) != 50
            or set(ids) - set(metric_by_id)
        ):
            raise ValueError(f"invalid frozen K50 row: {uid}")

    error_rows = list(read_jsonl(paths["prior_errors"]))
    wrong_by_uid: dict[str, set[str]] = defaultdict(set)
    error_counts: Counter[str] = Counter()
    for row in error_rows:
        uid = str(row.get("norm_uid") or "")
        if uid not in truth:
            continue
        expected = truth[uid]
        declared_truth = row.get("truth") or {}
        if (
            declared_truth.get("decision") != expected.get("decision")
            or declared_truth.get("metric_id") != expected.get("metric_id")
        ):
            raise ValueError(f"prior error truth drift: {uid}")
        for order in ("original", "hashed"):
            prediction = row.get(order) or {}
            predicted = prediction.get("metric_id") if prediction.get("decision") == "MATCH" else None
            if predicted and predicted in metric_by_id and predicted != expected.get("metric_id"):
                wrong_by_uid[uid].add(str(predicted))
                if expected["decision"] == "MATCH":
                    error_counts["wrong_leaf_metric"] += 1
                else:
                    error_counts["false_match_metric"] += 1
    prompt = "\n\n".join(
        paths[name].read_text(encoding="utf-8").rstrip()
        for name in ("base_prompt", "prompt_addon")
    ) + "\n"

    # The bank cards are the 285 R2 merged groups.  Their
    # ``source_r2_cluster_ids`` fields point *down* to raw clusters and do not
    # define sibling metric families.  The frozen pre-existing R3 hierarchy is
    # the auditable parent map: its bare ``source_r2_cluster_ids`` are the
    # numeric suffixes of the current ``a{N}`` metric IDs.  R3 intentionally
    # lists only multi-member merges; omitted R2 metrics are singleton
    # families and therefore have no same-family negative.
    hierarchy = json.loads(paths["frozen_r3_hierarchy"].read_text(encoding="utf-8"))
    if (
        hierarchy.get("task") != TASK
        or int(hierarchy.get("n_r2_clusters_in", -1)) != 285
        or int(hierarchy.get("n_merged_groups", -1)) != len(hierarchy.get("merged_groups") or [])
    ):
        raise ValueError("unexpected frozen Humor R3 hierarchy")
    family_members: dict[int, set[str]] = defaultdict(set)
    metric_family: dict[str, int] = {}
    for family_id, group in enumerate(hierarchy.get("merged_groups") or []):
        ids = [f"a{int(value)}" for value in group.get("source_r2_cluster_ids") or []]
        names = [str(value) for value in group.get("source_r2_cluster_names") or []]
        if len(ids) < 2 or len(ids) != len(names) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid frozen R3 family {family_id}")
        for metric_id, expected_name in zip(ids, names):
            if metric_id not in metric_by_id:
                raise ValueError(f"frozen R3 metric absent from current bank: {metric_id}")
            if str(metric_by_id[metric_id].get("name") or "") != expected_name:
                raise ValueError(f"frozen R3 name drift for {metric_id}")
            if metric_id in metric_family:
                raise ValueError(f"metric appears in two frozen R3 families: {metric_id}")
            metric_family[metric_id] = family_id
            family_members[family_id].add(metric_id)
    if len(metric_family) != 175 or len(family_members) != 60:
        raise ValueError(
            f"unexpected frozen R3 coverage: metrics={len(metric_family)} families={len(family_members)}"
        )

    examples: list[dict[str, Any]] = []
    selected_lane_counts: Counter[str] = Counter()
    injected_targets = 0
    same_family_available = same_family_selected = 0
    wrong_metric_rows = 0
    prompt_chars: list[int] = []
    for uid in sorted(truth):
        row = truth[uid]
        norm = norms[uid]
        k50 = [str(value["metric_id"]) for value in candidates[uid]["candidates"]]
        target = str(row["metric_id"]) if row["decision"] == "MATCH" else None
        wrong = sorted(wrong_by_uid.get(uid, set()))
        wrong_metric_rows += int(bool(wrong))
        family: set[str] = set()
        if target and target in metric_family:
            family.update(family_members[metric_family[target]])
            family.discard(target)
            same_family_available += int(bool(family))
        family_ordered = sorted(
            family,
            key=lambda mid: (k50.index(mid) if mid in k50 else 10**9, mid),
        )
        chosen: list[str] = []
        lanes: dict[str, str] = {}

        def add(metric_id: str, lane: str) -> None:
            if metric_id in metric_by_id and metric_id not in chosen and len(chosen) < args.max_candidates:
                chosen.append(metric_id)
                lanes[metric_id] = lane
                selected_lane_counts[lane] += 1

        if target:
            add(target, "exact_target")
        for metric_id in wrong[:4]:
            add(metric_id, "prior_adjudicator_wrong_leaf_or_false_match")
        before_family = len(chosen)
        for metric_id in family_ordered[:5]:
            add(metric_id, "same_frozen_r3_sibling_leaf")
        same_family_selected += int(len(chosen) > before_family)
        for metric_id in k50:
            add(metric_id, "retriever_top_confusion")
        if len(chosen) != args.max_candidates:
            raise ValueError(f"could not build exact K={args.max_candidates} hard slate: {uid}")
        if target and target not in chosen:
            raise AssertionError(f"target missing from training slate: {uid}")

        retrieval_order = sorted(
            chosen,
            key=lambda mid: (k50.index(mid) if mid in k50 else 10**9, mid),
        )
        if target and target not in k50:
            injected_targets += 1
            retrieval_order.remove(target)
            retrieval_order = _insert_at_hash_position(retrieval_order, target, uid)
        orderings = {
            "retrieval_hardmix": retrieval_order,
            "sha256_permutation": sorted(
                chosen, key=lambda mid: (_hash_order(uid, mid, args.order_seed), mid)
            ),
        }
        if set(orderings["retrieval_hardmix"]) != set(orderings["sha256_permutation"]):
            raise AssertionError(f"candidate order views differ in set: {uid}")
        if len(chosen) > 1 and orderings["retrieval_hardmix"] == orderings["sha256_permutation"]:
            # A second deterministic seed gives an actual permutation without
            # using any label or model result.
            orderings["sha256_permutation"] = sorted(
                chosen,
                key=lambda mid: (_hash_order(uid, mid, args.order_seed + 1), mid),
            )

        target_json = json.dumps(
            {
                "decision": row["decision"],
                "metric_id": target,
                "confidence": str(row.get("confidence") or "medium"),
                "reason": _support_reason(row),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for view, ordered_ids in orderings.items():
            candidate_cards = [{"metric_id": metric_id} for metric_id in ordered_ids]
            rendered = build_item_prompt(
                prompt,
                norm,
                candidate_cards,
                metric_by_id,
                context_chars=1400,
                description_chars=520,
                example_chars=180,
                max_examples=2,
            )
            prompt_chars.append(len(rendered))
            examples.append(
                {
                    "schema_version": "silver-match-v3-humor-gemma4-lora-example-v1",
                    "task": TASK,
                    "corpus": str(norm["corpus"]),
                    "norm_uid": uid,
                    "source_group": str(row["source_group"]),
                    "view": view,
                    "split": "train",
                    "decision": row["decision"],
                    "metric_id": target,
                    "candidate_metric_ids": ordered_ids,
                    "candidate_lanes": {mid: lanes[mid] for mid in ordered_ids},
                    "gradient_eligible": True,
                    "messages": [
                        {"role": "user", "content": rendered},
                        {"role": "assistant", "content": target_json},
                    ],
                }
            )

    if len(examples) != 1792:
        raise ValueError(f"expected two views for all 896 truth rows: {len(examples)}")
    report = {
        "schema_version": "silver-match-v3-humor-gemma4-lora-dataset-report-v1",
        "status": "FROZEN_CLEAN_TYPED_SFT_READY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "truth_rows": 896,
        "example_rows": len(examples),
        "views_per_truth_row": 2,
        "truth_decision_counts": dict(sorted(decisions.items())),
        "match_rows": decisions["MATCH"],
        "typed_nonmatch_rows": 896 - decisions["MATCH"],
        "max_candidates": args.max_candidates,
        "candidate_view_counts": dict(sorted(Counter(row["view"] for row in examples).items())),
        "candidate_lane_selections": dict(sorted(selected_lane_counts.items())),
        "prior_error_metric_events": dict(sorted(error_counts.items())),
        "rows_with_prior_wrong_or_false_match_metric": wrong_metric_rows,
        "same_family_available_match_rows": same_family_available,
        "same_family_selected_match_rows": same_family_selected,
        "frozen_r3_family_count": len(family_members),
        "frozen_r3_family_metric_count": len(metric_family),
        "frozen_r3_singleton_metric_count": len(metric_by_id) - len(metric_family),
        "target_absent_from_frozen_k50_but_injected": injected_targets,
        "prompt_chars": {
            "min": min(prompt_chars),
            "max": max(prompt_chars),
            "mean": sum(prompt_chars) / len(prompt_chars),
        },
        "fresh_select_uid_overlap": 0,
        "fresh_select_source_group_overlap": 0,
        "select_labels_or_consensus_read": False,
        "label_or_metric_targets_changed": False,
        "gepa_prompt_role": "fixed_inference_scaffold_not_gradient_source",
        "inputs": {name: _ref(path) for name, path in sorted(paths.items())},
    }
    return examples, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--norms", required=True)
    parser.add_argument("--optimize-truth", required=True)
    parser.add_argument("--historical-truth", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prior-errors", required=True)
    parser.add_argument("--frozen-r3-hierarchy", required=True)
    parser.add_argument("--fresh-select-identities", required=True)
    parser.add_argument("--fresh-select-freeze", required=True)
    parser.add_argument("--base-prompt", required=True)
    parser.add_argument("--prompt-addon", required=True)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--order-seed", type=int, default=202607131401)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite frozen Gemma LoRA dataset")
    examples, report = build(args)
    write_jsonl(output, examples)
    report["output"] = {**_ref(output), "count": len(examples)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
