#!/usr/bin/env python3
"""Apply the unchanged dev-frozen Humor CE+c2 rule to production outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import read_jsonl, sha256_file
from .run_humor_c2_production_paired_vllm import META_SCHEMA, PREDICTION_SCHEMA


SCHEMA = "silver-match-v3-humor-c2-full285-deployment-prediction-v1"
REPORT_SCHEMA = "silver-match-v3-humor-c2-full285-deployment-report-v1"
PACKAGE_SCHEMA = "silver-match-v3-humor-ce-top16-plus-positives-v1"
EXPECTED_UIDS = 55_288
MINIMUM_SLATE_DEPTH = 16
HYBRID_THRESHOLD = 0.9871788024902344
CE_MARGIN_THRESHOLD = 0.0
DEPLOYMENT_CLAIM = "DEV_FROZEN_DEPLOYMENT_BLIND_P855"
CONF_RANK = {"low": 0, "medium": 1, "high": 2}
FINAL_DECISIONS = {
    "MATCH", "MATCH_FAMILY_ONLY", "NO_EXPLICIT_CRITERION", "CONTEXT_NEEDED",
    "GENERIC_VERDICT", "NO_CANDIDATE_FITS", "NOISE", "INVALID_OUTPUT",
    "UNSTABLE_MATCH",
}


def artifact(path: Path, **extra: Any) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, **extra}


def _load_typed(roots: list[Path]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    metas = []
    shard_ids: set[int] = set(); num_shards: int | None = None
    for root in roots:
        meta_path = root / "INFERENCE_META.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")); metas.append(meta)
        if (
            meta.get("schema_version") != META_SCHEMA
            or meta.get("status") != "COMPLETE_C2_PRODUCTION_PAIRED_INFERENCE"
            or meta.get("deployment_claim") != DEPLOYMENT_CLAIM
            or meta.get("test_or_blind_rows_read") != 0
        ):
            raise ValueError(f"invalid typed production shard: {root}")
        shard_id, current_total = int(meta.get("shard_id", -1)), int(meta.get("num_shards", -1))
        if shard_id in shard_ids or current_total < 1 or not 0 <= shard_id < current_total:
            raise ValueError("invalid/duplicate typed shard coordinates")
        shard_ids.add(shard_id)
        if num_shards is None: num_shards = current_total
        elif num_shards != current_total: raise ValueError("typed shards disagree on total")
        for order in ("original", "reordered"):
            path = root / f"typed.{order}.jsonl"
            ref = (meta.get("outputs") or {}).get(order) or {}
            if ref.get("sha256") != sha256_file(path):
                raise ValueError(f"typed shard output SHA differs: {path}")
            for row in read_jsonl(path):
                uid = str(row.get("norm_uid") or "")
                if (
                    row.get("schema_version") != PREDICTION_SCHEMA or row.get("split") != "production"
                    or row.get("order_mode") != order or not uid or order in result.get(uid, {})
                ):
                    raise ValueError(f"invalid/duplicate typed prediction: {uid}/{order}")
                result.setdefault(uid, {})[order] = row
    if num_shards is None or shard_ids != set(range(num_shards)):
        raise ValueError("typed shard set is incomplete")
    if len(result) != EXPECTED_UIDS or any(set(values) != {"original", "reordered"} for values in result.values()):
        raise ValueError("typed paired coverage differs")
    return result, metas


def _minimum_confidence(left: str, right: str) -> str:
    if left not in CONF_RANK or right not in CONF_RANK:
        raise ValueError("invalid typed confidence")
    return left if CONF_RANK[left] <= CONF_RANK[right] else right


def build(args: argparse.Namespace) -> dict[str, Any]:
    package_path, output, report_path = map(Path, (args.candidate_package, args.output, args.report_output))
    package_path, output, report_path = package_path.resolve(), output.resolve(), report_path.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite deployment finalization")
    typed, typed_metas = _load_typed([Path(value).resolve() for value in args.typed_root])
    counts: Counter[str] = Counter(); status_counts: Counter[str] = Counter()
    accepted_metric_counts: Counter[str] = Counter(); seen: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for index, package in enumerate(read_jsonl(package_path)):
                uid = str(package.get("norm_uid") or "")
                candidates = [str(value) for value in package.get("candidate_metric_ids") or []]
                if (
                    package.get("schema_version") != PACKAGE_SCHEMA or package.get("split") != "production"
                    or not uid or uid in seen or len(candidates) < MINIMUM_SLATE_DEPTH or len(candidates) != len(set(candidates))
                ):
                    raise ValueError(f"invalid candidate package: {uid}")
                seen.add(uid)
                left, right = typed[uid]["original"], typed[uid]["reordered"]
                if (
                    [str(value) for value in left.get("candidate_metric_ids") or []] != candidates
                    or set(str(value) for value in right.get("candidate_metric_ids") or []) != set(candidates)
                ):
                    raise ValueError(f"typed/package slate differs: {uid}")
                ldecision, rdecision = str(left.get("decision") or ""), str(right.get("decision") or "")
                lmetric = str(left.get("metric_id") or "") or None
                rmetric = str(right.get("metric_id") or "") or None
                lconf, rconf = str(left.get("confidence") or "").lower(), str(right.get("confidence") or "").lower()
                confidence = _minimum_confidence(lconf, rconf)
                ce_metric = str(package.get("ce_top1_metric_id") or "")
                ce_score = float(package.get("ce_top1_exact_probability", -1))
                ce_margin = float(package.get("ce_top_margin", -1))
                stable_match = ldecision == rdecision == "MATCH" and lmetric is not None and lmetric == rmetric
                accepted = bool(
                    stable_match and confidence == "high" and lmetric == ce_metric
                    and ce_score >= HYBRID_THRESHOLD and ce_margin >= CE_MARGIN_THRESHOLD
                )
                if accepted:
                    decision, metric_id = "MATCH", lmetric
                    status = DEPLOYMENT_CLAIM
                    reason = "Two-order c2 exact leaf equals CE top-1 under the unchanged dev-frozen hybrid gate."
                    accepted_metric_counts[str(metric_id)] += 1
                elif "INVALID_OUTPUT" in {ldecision, rdecision}:
                    decision, metric_id, confidence = "INVALID_OUTPUT", None, "low"
                    status = "PRODUCTION_C2_INVALID_OUTPUT"
                    reason = "At least one paired-order c2 decode was invalid."
                elif ldecision == rdecision and ldecision != "MATCH" and ldecision in FINAL_DECISIONS:
                    decision, metric_id = ldecision, None
                    status = "PRODUCTION_C2_STABLE_PAIRED_ABSTENTION"
                    reason = str(left.get("reason") or "Paired-order c2 abstention.")
                else:
                    decision, metric_id, confidence = "UNSTABLE_MATCH", None, "low"
                    status = "PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE"
                    reason = "Paired orders disagreed or the proposed match failed the unchanged CE+c2 gate."
                if decision not in FINAL_DECISIONS or (decision == "MATCH") != (metric_id is not None):
                    raise ValueError(f"invalid normalized decision: {uid}/{decision}/{metric_id}")
                counts[decision] += 1; status_counts[status] += 1
                row = {
                    "schema_version": SCHEMA, "task": "humor", "corpus": "humor_multi",
                    "row": index, "norm_uid": uid, "decision": decision, "metric_id": metric_id,
                    "confidence": confidence, "reason": reason, "verification_status": status,
                    "deployment_claim": DEPLOYMENT_CLAIM,
                    "blind_gate": {"precision": 0.855, "wilson_lower": 0.753, "promotion_passed": False},
                    "bank_source_sha256": package.get("bank_source_sha256"),
                    "candidate_ids": candidates, "proposed_metric_id": lmetric,
                    "frozen_hybrid_evidence": {
                        "minimum_ce_exact_probability": HYBRID_THRESHOLD,
                        "minimum_ce_top_margin": CE_MARGIN_THRESHOLD,
                        "minimum_typed_confidence": "high", "ce_top1_metric_id": ce_metric,
                        "ce_top1_exact_probability": ce_score, "ce_top_margin": ce_margin,
                        "typed_original": {key: left.get(key) for key in ("decision", "metric_id", "confidence", "reason", "parse_error")},
                        "typed_reordered": {key: right.get(key) for key in ("decision", "metric_id", "confidence", "reason", "parse_error")},
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        if len(seen) != EXPECTED_UIDS or set(typed) != seen:
            raise ValueError("normalized production coverage differs")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True); raise
    matches = counts["MATCH"]
    report = {
        "schema_version": REPORT_SCHEMA, "status": "COMPLETE_DEV_FROZEN_DEPLOYMENT_BLIND_P855",
        "task": "humor", "corpus": "humor_multi", "test_or_blind_rows_read": 0,
        "deployment_claim": DEPLOYMENT_CLAIM,
        "blind_gate": {"precision": 0.855, "wilson_lower": 0.753, "promotion_passed": False,
                       "validated_high_precision_claim_allowed": False},
        "frozen_hybrid_rule": {"typed_same_exact_leaf_both_orders": True,
                               "minimum_typed_confidence": "high", "ce_top1_must_equal_typed_leaf": True,
                               "minimum_ce_exact_probability": HYBRID_THRESHOLD,
                               "minimum_ce_top_margin": CE_MARGIN_THRESHOLD, "retuned_on_production": False},
        "coverage": {"norm_uids": len(seen), "normalized_rows": len(seen),
                     "match_rows": matches, "match_rate": matches / EXPECTED_UIDS,
                     "abstention_or_unstable_rows": EXPECTED_UIDS - matches,
                     "abstention_or_unstable_rate": (EXPECTED_UIDS - matches) / EXPECTED_UIDS},
        "decision_counts": dict(sorted(counts.items())),
        "verification_status_counts": dict(sorted(status_counts.items())),
        "accepted_metric_concentration": {"unique_metrics": len(accepted_metric_counts),
                                           "largest_share": max(accepted_metric_counts.values(), default=0) / matches if matches else 0,
                                           "top20": accepted_metric_counts.most_common(20)},
        "inputs": {"candidate_package": artifact(package_path, rows=EXPECTED_UIDS),
                   "typed_shards": [artifact(Path(root).resolve() / "INFERENCE_META.json") for root in args.typed_root]},
        "output": artifact(output, rows=EXPECTED_UIDS),
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({key: report[key] for key in ("status", "coverage", "decision_counts", "verification_status_counts", "output")}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-package", required=True); parser.add_argument("--typed-root", action="append", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--report-output", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
