#!/usr/bin/env python3
"""Sample production-only audit strata from the sealed full-285 c2 deployment.

The frozen v1 sampler expected ``provenance.hybrid_policy_lane`` on each
production row.  The sealed 55,288-row normalized artifact
(``silver-match-v3-humor-c2-full285-deployment-prediction-v1``) instead carries
the full paired-order evidence in ``frozen_hybrid_evidence`` and collapses the
policy lanes into ``verification_status``/``decision``.  This v2 sampler
derives the audit lanes deterministically from that sealed evidence -- it does
not modify the artifact and it fails closed on any row it cannot classify.

Lane derivation (deterministic, evidence-only):

- decision MATCH + status DEV_FROZEN_DEPLOYMENT_BLIND_P855
    -> HYBRID_ACCEPTED_MATCH
- status PRODUCTION_C2_STABLE_PAIRED_ABSTENTION
    -> TYPED_STABLE_ABSTENTION
- status PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE where BOTH typed orders decided
  MATCH on the SAME leaf (so only the frozen CE+confidence gate rejected it)
    -> TYPED_MATCH_HYBRID_REJECTED
- status PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE otherwise (order disagreement)
    -> TYPED_ORDER_UNSTABLE
- status PRODUCTION_C2_INVALID_OUTPUT -> INVALID_OUTPUT (all sampled)
- CE_UNCOVERED is structurally empty: production scored all 285 bank metrics
  for every norm, so no norm lacks CE coverage.  It is retained with
  population 0 so the release freezer sees the full frozen lane plan.

Sampling seed string, per-lane ordering rule, sample schema, and report schema
are identical to v1 so downstream release freezing is unchanged.
"""

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
REPORT_SCHEMA = "silver-match-v3-humor-production-risk-sample-report-v1"
PREDICTION_SCHEMA = "silver-match-v3-humor-c2-full285-deployment-prediction-v1"
LANE_SIZES = {
    "HYBRID_ACCEPTED_MATCH": 400,
    "TYPED_STABLE_ABSTENTION": 200,
    "TYPED_MATCH_HYBRID_REJECTED": 200,
    "TYPED_ORDER_UNSTABLE": 200,
    "CE_UNCOVERED": 200,
    "INVALID_OUTPUT": 2,
}
SEED = "humor-postprod-audit-v1"


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
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON: {path}:{line_number}") from exc


def derive_lane(row: Mapping[str, Any]) -> str:
    uid = row.get("norm_uid")
    status = str(row.get("verification_status") or "")
    decision = str(row.get("decision") or "")
    if status == "DEV_FROZEN_DEPLOYMENT_BLIND_P855":
        if decision != "MATCH" or not row.get("metric_id"):
            raise ValueError(f"accepted row without exact MATCH contract: {uid}")
        return "HYBRID_ACCEPTED_MATCH"
    if status == "PRODUCTION_C2_STABLE_PAIRED_ABSTENTION":
        if decision == "MATCH" or row.get("metric_id"):
            raise ValueError(f"stable abstention carries a MATCH/metric: {uid}")
        return "TYPED_STABLE_ABSTENTION"
    if status == "PRODUCTION_C2_INVALID_OUTPUT":
        return "INVALID_OUTPUT"
    if status == "PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE":
        evidence = row.get("frozen_hybrid_evidence") or {}
        original = evidence.get("typed_original") or {}
        reordered = evidence.get("typed_reordered") or {}
        if not original or not reordered:
            raise ValueError(f"unstable row lacks paired typed evidence: {uid}")
        if (
            original.get("decision") == "MATCH"
            and reordered.get("decision") == "MATCH"
            and original.get("metric_id")
            and original.get("metric_id") == reordered.get("metric_id")
        ):
            return "TYPED_MATCH_HYBRID_REJECTED"
        return "TYPED_ORDER_UNSTABLE"
    raise ValueError(f"unknown production verification_status: {uid}/{status}")


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
    for row in read_jsonl(predictions_path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in seen:
            raise ValueError(f"missing/duplicate production norm_uid: {uid}")
        if row.get("schema_version") != PREDICTION_SCHEMA:
            raise ValueError(f"unexpected prediction schema: {uid}")
        if str(row.get("split") or "production").lower() in {"blind", "test"}:
            raise ValueError("blind/test row supplied to production audit sampler")
        if (row.get("decision") == "MATCH") != bool(row.get("metric_id")):
            raise ValueError(f"invalid production decision/metric contract: {uid}")
        seen.add(uid)
        rows_by_lane[derive_lane(row)].append(row)
    if not seen:
        raise ValueError("empty production prediction file")
    if rows_by_lane["CE_UNCOVERED"]:
        raise ValueError("CE_UNCOVERED must be empty under full-285 scoring")

    output_root.mkdir(parents=True, exist_ok=False)
    sample_path = output_root / "production_risk_audit_sample.jsonl"
    selected_counts: Counter[str] = Counter()
    with sample_path.open("x", encoding="utf-8") as handle:
        for lane, requested in LANE_SIZES.items():
            values = sorted(
                rows_by_lane[lane],
                key=lambda row: hashlib.sha256(
                    f"{SEED}\0{lane}\0{row['norm_uid']}".encode()
                ).hexdigest(),
            )
            for rank, row in enumerate(values[:requested], 1):
                handle.write(json.dumps({
                    "schema_version": SAMPLE_SCHEMA, "audit_lane": lane,
                    "audit_rank": rank, "prediction": row,
                }, ensure_ascii=False, sort_keys=True) + "\n")
                selected_counts[lane] += 1
        handle.flush()
        os.fsync(handle.fileno())
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE_CREATE_ONLY_PRODUCTION_AUDIT_SAMPLE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "risk_record": artifact(risk_path),
        "production_predictions": artifact(predictions_path),
        "population": {lane: len(rows) for lane, rows in rows_by_lane.items()},
        "requested": dict(LANE_SIZES),
        "selected": dict(selected_counts),
        "sample": artifact(sample_path),
        "blind_or_test_rows_read": 0,
        "lane_derivation": {
            "version": "v2-derived-from-sealed-full285-evidence",
            "reason": (
                "sealed deployment schema carries paired typed evidence instead "
                "of provenance.hybrid_policy_lane; lanes derived deterministically"
            ),
            "HYBRID_ACCEPTED_MATCH": "decision=MATCH, status=DEV_FROZEN_DEPLOYMENT_BLIND_P855",
            "TYPED_STABLE_ABSTENTION": "status=PRODUCTION_C2_STABLE_PAIRED_ABSTENTION",
            "TYPED_MATCH_HYBRID_REJECTED": (
                "status=PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE and both typed orders "
                "decided MATCH on the same leaf (CE/confidence gate rejection)"
            ),
            "TYPED_ORDER_UNSTABLE": (
                "status=PRODUCTION_C2_OR_HYBRID_GATE_UNSTABLE with order disagreement"
            ),
            "INVALID_OUTPUT": "status=PRODUCTION_C2_INVALID_OUTPUT (all sampled)",
            "CE_UNCOVERED": "structurally empty: all 285 metrics CE-scored per norm",
        },
    }
    write_json_new(output_root / "REPORT.json", report)
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample-production")
    p.add_argument("--risk-record", required=True)
    p.add_argument("--production-predictions", required=True)
    p.add_argument("--output-root", required=True)
    args = parser.parse_args()
    sample(args)


if __name__ == "__main__":
    main()
