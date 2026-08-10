#!/usr/bin/env python3
"""Freeze Humor hybrid deployment risk and sample production-only audit strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


RISK_SCHEMA = "silver-match-v3-humor-hybrid-postblind-risk-v1"
SAMPLE_SCHEMA = "silver-match-v3-humor-production-risk-sample-v1"
LANE_SIZES = {
    "HYBRID_ACCEPTED_MATCH": 400,
    "TYPED_STABLE_ABSTENTION": 200,
    "TYPED_MATCH_HYBRID_REJECTED": 200,
    "TYPED_ORDER_UNSTABLE": 200,
    "CE_UNCOVERED": 200,
}


def _audit_lane(row: Mapping[str, Any]) -> tuple[str, str]:
    """Read the explicit lane or derive its exact legacy C2 equivalent."""
    provenance = row.get("provenance") or {}
    explicit = str(provenance.get("hybrid_policy_lane") or "")
    if explicit:
        return explicit, "provenance.hybrid_policy_lane"
    decision = str(row.get("decision") or "")
    verification = str(row.get("verification_status") or "")
    mapping = {
        ("MATCH", "DEV_FROZEN_DEPLOYMENT_BLIND_P855"): "HYBRID_ACCEPTED_MATCH",
        ("UNSTABLE_MATCH", "PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE"): "TYPED_ORDER_UNSTABLE",
        ("INVALID_OUTPUT", "PRODUCTION_C2_INVALID_OUTPUT"): "CE_UNCOVERED",
    }
    lane = mapping.get((decision, verification))
    if lane:
        return lane, "derived_from_frozen_decision_and_verification_status"
    if verification == "PRODUCTION_C2_STABLE_PAIRED_ABSTENTION" and decision != "MATCH":
        return "TYPED_STABLE_ABSTENTION", "derived_from_frozen_decision_and_verification_status"
    return "", "unresolved"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON: {path}:{line_number}") from exc


def freeze(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    gate_path = Path(args.dev_gate).resolve()
    blind_path = Path(args.blind_score).resolve()
    if sha256_file(gate_path) != args.dev_gate_sha256:
        raise ValueError("dev gate SHA mismatch")
    if sha256_file(blind_path) != args.blind_score_sha256:
        raise ValueError("blind score SHA mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "PASS_VALIDATED_DEV_HYBRID_GATE"
        or blind.get("status")
        != "COMPLETE_ONE_SHOT_TRUTH_FIREWALLED_BLIND_HYBRID_EVALUATION"
        or blind["truth_firewall"].get("test_rows_read") != 0
        or blind["truth_firewall"].get("threshold_or_model_selection_on_blind") is not False
    ):
        raise ValueError("inputs do not establish the frozen dev/blind contract")
    selected = gate["selected_gate"]
    metrics = blind["metrics"]
    blind_passed = bool(
        metrics["exact_precision"] >= gate["policy"]["minimum_precision"]
        and metrics["precision_wilson_95"][0] >= gate["policy"]["minimum_wilson_lower"]
    )
    if blind_passed:
        raise ValueError("risk record is only for the observed blind contract miss")

    record = {
        "schema_version": RISK_SCHEMA,
        "status": "FROZEN_ADVISORY_ONLY_AFTER_BLIND_PRECISION_CONTRACT_MISS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bound_evidence": {"dev_gate": artifact(gate_path), "blind_score": artifact(blind_path)},
        "frozen_rule": {
            "minimum_typed_confidence": selected["minimum_typed_confidence"],
            "minimum_ce_exact_probability": selected["minimum_ce_exact_probability"],
            "minimum_ce_top_margin": selected["minimum_ce_top_margin"],
            "acceptance": "same typed MATCH leaf both orders and CE same top1",
        },
        "external_validity": {
            "dev": {
                "accepted": selected["accepted"], "correct": selected["correct"],
                "precision": selected["precision"],
                "precision_wilson_95": selected["precision_wilson_95"],
                "recall": selected["recall_of_overlap_gold_matches"],
            },
            "blind": {
                **blind["counts"], **metrics,
                "precision_contract_passed": False,
                "wilson_contract_passed": False,
            },
            "conclusion": (
                "The frozen hybrid is useful as an advisory precision-enrichment lane, "
                "but is not validated as a >=0.90 precision production truth source."
            ),
        },
        "canonical_row_policy": {
            "HYBRID_ACCEPTED_MATCH": {
                "decision": "preserve MATCH and exact metric_id",
                "confidence": "medium",
                "verification_status": "HYBRID_ACCEPTED_BLIND_CONTRACT_FAILED_ADVISORY",
                "analysis_role": "advisory_silver; include only in explicit sensitivity analyses",
                "required_provenance": [
                    "typed_original", "typed_reordered", "ce_top1_metric_id",
                    "ce_exact_probability", "ce_top_margin", "frozen_rule_sha256",
                ],
            },
            "TYPED_STABLE_ABSTENTION": {
                "decision": "preserve the identical typed abstention decision",
                "metric_id": None,
                "verification_status": "TYPED_TWO_ORDER_STABLE_ABSTENTION",
                "analysis_role": "abstention; never coerce to a bank metric",
            },
            "TYPED_MATCH_HYBRID_REJECTED": {
                "decision": "CONTEXT_NEEDED",
                "metric_id": None,
                "verification_status": "OPERATIONAL_ABSTENTION_TYPED_MATCH_FAILED_HYBRID",
                "analysis_role": "abstention; preserve raw typed prediction in provenance",
            },
            "TYPED_ORDER_UNSTABLE": {
                "decision": "CONTEXT_NEEDED",
                "metric_id": None,
                "verification_status": "OPERATIONAL_ABSTENTION_ORDER_UNSTABLE",
                "analysis_role": "abstention; preserve both raw order predictions",
            },
            "CE_UNCOVERED": {
                "decision": "preserve stable typed abstention, otherwise CONTEXT_NEEDED",
                "metric_id": None,
                "verification_status": "CE_UNCOVERED_FORCED_ABSTENTION",
                "analysis_role": "coverage gap, not evidence of bank absence",
            },
            "required_field": "provenance.hybrid_policy_lane",
            "never_claim": [
                "blind-validated >=0.90 precision", "expert truth", "bank absence from abstention",
            ],
        },
        "postproduction_audit": {
            "sampling_seed": "humor-postprod-audit-v1",
            "requested_by_lane": LANE_SIZES,
            "selection": "ascending sha256(seed + NUL + lane + NUL + norm_uid)",
            "labels": "two independent strong-model labels, adjudicate disagreements; no model under audit labels its own sample",
            "primary_estimands": [
                "HYBRID_ACCEPTED_MATCH exact precision and Wilson 95% interval",
                "false-abstention rate for stable typed abstentions and hybrid rejections",
                "error rates stratified by order instability and CE coverage",
            ],
            "promotion_rule": "none predeclared here; this audit cannot retroactively alter the blind result",
            "forbidden_inputs": ["blind", "test"],
        },
        "no_retraining_or_retuning_performed": True,
        "blind_or_test_rows_read_by_this_freeze": 0,
    }
    write_json_new(output, record)
    print(json.dumps({"status": record["status"], "blind": record["external_validity"]["blind"]}, sort_keys=True))


def sample(args: argparse.Namespace) -> None:
    risk_path = Path(args.risk_record).resolve()
    predictions_path = Path(args.production_predictions).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    risk = json.loads(risk_path.read_text(encoding="utf-8"))
    if risk.get("schema_version") != RISK_SCHEMA:
        raise ValueError("invalid risk record")
    rows_by_lane: dict[str, list[dict[str, Any]]] = {key: [] for key in LANE_SIZES}
    seen: set[str] = set()
    lane_sources: Counter[str] = Counter()
    for row in read_jsonl(predictions_path):
        uid = str(row.get("norm_uid") or "")
        lane, lane_source = _audit_lane(row)
        if not uid or uid in seen:
            raise ValueError(f"missing/duplicate production norm_uid: {uid}")
        if str(row.get("split") or "production").lower() in {"blind", "test"}:
            raise ValueError("blind/test row supplied to production audit sampler")
        if lane not in rows_by_lane:
            raise ValueError(f"missing/unknown provenance.hybrid_policy_lane: {uid}/{lane}")
        if (row.get("decision") == "MATCH") != bool(row.get("metric_id")):
            raise ValueError(f"invalid production decision/metric contract: {uid}")
        seen.add(uid)
        rows_by_lane[lane].append(row)
        lane_sources[lane_source] += 1
    if not seen:
        raise ValueError("empty production prediction file")

    output_root.mkdir(parents=True, exist_ok=False)
    sample_path = output_root / "production_risk_audit_sample.jsonl"
    selected_counts: Counter[str] = Counter()
    with sample_path.open("x", encoding="utf-8") as handle:
        for lane, requested in LANE_SIZES.items():
            values = sorted(
                rows_by_lane[lane],
                key=lambda row: hashlib.sha256(
                    f"humor-postprod-audit-v1\0{lane}\0{row['norm_uid']}".encode()
                ).hexdigest(),
            )
            for rank, row in enumerate(values[:requested], 1):
                handle.write(json.dumps({
                    "schema_version": SAMPLE_SCHEMA, "audit_lane": lane,
                    "audit_rank": rank, "prediction": row,
                }, ensure_ascii=False, sort_keys=True) + "\n")
                selected_counts[lane] += 1
        handle.flush(); os.fsync(handle.fileno())
    report = {
        "schema_version": "silver-match-v3-humor-production-risk-sample-report-v1",
        "status": "COMPLETE_CREATE_ONLY_PRODUCTION_AUDIT_SAMPLE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "risk_record": artifact(risk_path), "production_predictions": artifact(predictions_path),
        "population": {lane: len(rows) for lane, rows in rows_by_lane.items()},
        "lane_assignment": {
            "sources": dict(sorted(lane_sources.items())),
            "legacy_c2_derivation": {
                "MATCH+DEV_FROZEN_DEPLOYMENT_BLIND_P855": "HYBRID_ACCEPTED_MATCH",
                "non-MATCH+PRODUCTION_C2_STABLE_PAIRED_ABSTENTION": "TYPED_STABLE_ABSTENTION",
                "UNSTABLE_MATCH+PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE": "TYPED_ORDER_UNSTABLE",
                "INVALID_OUTPUT+PRODUCTION_C2_INVALID_OUTPUT": "CE_UNCOVERED",
            },
            "prediction_rows_mutated": False,
        },
        "requested": LANE_SIZES, "selected": dict(selected_counts),
        "sample": artifact(sample_path),
        "blind_or_test_rows_read": 0,
    }
    write_json_new(output_root / "REPORT.json", report)
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--dev-gate", required=True); p.add_argument("--dev-gate-sha256", required=True)
    p.add_argument("--blind-score", required=True); p.add_argument("--blind-score-sha256", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("sample-production")
    p.add_argument("--risk-record", required=True); p.add_argument("--production-predictions", required=True)
    p.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "freeze": freeze(args)
    else: sample(args)


if __name__ == "__main__":
    main()
