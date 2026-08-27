#!/usr/bin/env python3
"""Create the canonical Humor final by overlaying truth on production predictions.

The 22,090-row authoritative joined truth and 55,288-row production complement
must be disjoint and must exactly partition the frozen 77,378-row canonical
Humor corpus. Output follows canonical manifest order and is immediately run
through the standard final-output audit. All artifacts are create-only.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .audit_final_outputs import audit_outputs
from .common import normalize_space, read_jsonl, sha256_file
from .finalize_adjudications import FINAL_DECISIONS


SCHEMA = "silver-match-v3-humor-authoritative-overlay-final-v1"
REPORT_SCHEMA = "silver-match-v3-humor-authoritative-overlay-report-v1"


def _resolve(raw: str | Path, anchor: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA256 differs: {observed}")
    return observed


def _index(
    path: Path,
    *,
    label: str,
    expected_count: int,
    allowed_decisions: set[str],
    canonical_uids: set[str],
    metric_ids: set[str],
    bank_sha: str,
    allow_missing_confidence: bool = False,
) -> tuple[dict[str, dict[str, Any]], Counter[str], int]:
    indexed: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    defaulted_confidence = 0
    for line_no, raw in enumerate(read_jsonl(path), 1):
        row = dict(raw)
        uid = normalize_space(row.get("norm_uid"))
        if not uid or uid in indexed:
            raise ValueError(f"{label} missing/duplicate UID at line {line_no}: {uid!r}")
        if uid not in canonical_uids:
            raise ValueError(f"{label} contains foreign UID: {uid}")
        decision = normalize_space(row.get("decision")).upper()
        if decision not in allowed_decisions:
            raise ValueError(f"{label} invalid decision for {uid}: {decision!r}")
        metric_id = normalize_space(row.get("metric_id")) or None
        if decision == "MATCH":
            if metric_id not in metric_ids:
                raise ValueError(f"{label} MATCH metric outside bank for {uid}: {metric_id}")
        elif metric_id is not None:
            raise ValueError(f"{label} non-MATCH carries metric for {uid}: {metric_id}")
        confidence = normalize_space(row.get("confidence")).lower()
        if not confidence and allow_missing_confidence:
            # The authoritative joined-truth release predates the confidence
            # field.  Preserve that fact explicitly and use the conservative
            # valid release value only for schema compatibility.  These rows
            # are excluded from all downstream MI/outcome estimation.
            confidence = "medium"
            row["confidence_defaulted_from_missing"] = True
            row["confidence_default_reason"] = (
                "AUTHORITATIVE_JOINED_TRUTH_PREDATES_CONFIDENCE_FIELD"
            )
            defaulted_confidence += 1
        if confidence not in CONFIDENCES:
            raise ValueError(f"{label} invalid confidence for {uid}: {confidence!r}")
        supplied_bank = normalize_space(
            row.get("bank_source_sha256") or row.get("current_bank_source_sha256")
        )
        if supplied_bank and supplied_bank != bank_sha:
            raise ValueError(f"{label} bank provenance differs for {uid}")
        row["decision"] = decision
        row["metric_id"] = metric_id
        row["confidence"] = confidence
        indexed[uid] = row
        counts[decision] += 1
    if len(indexed) != expected_count:
        raise ValueError(f"{label} expected {expected_count} UIDs, got {len(indexed)}")
    return indexed, counts, defaulted_confidence


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    truth_path = Path(args.truth).resolve()
    predictions_path = Path(args.predictions).resolve()
    output_path = Path(args.output).resolve()
    audit_path = Path(args.audit_output).resolve()
    report_path = Path(args.report_output).resolve()
    for path in (output_path, audit_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite release artifact: {path}")

    manifest_sha = _require_sha(manifest_path, args.manifest_sha256, "manifest")
    truth_sha = _require_sha(truth_path, args.truth_sha256, "truth")
    predictions_sha = _require_sha(
        predictions_path, args.predictions_sha256, "predictions"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_meta = (manifest.get("corpora") or {}).get(args.corpus) or {}
    bank_meta = (manifest.get("banks") or {}).get(args.task) or {}
    if corpus_meta.get("task") != args.task:
        raise ValueError("manifest does not bind requested task/corpus")
    canonical_path = _resolve(str(corpus_meta.get("path") or ""), manifest_path)
    bank_path = _resolve(str(bank_meta.get("path") or ""), manifest_path)
    canonical = list(read_jsonl(canonical_path))
    if len(canonical) != args.expected_canonical:
        raise ValueError("canonical Humor row count differs")
    canonical_uids = [normalize_space(row.get("norm_uid")) for row in canonical]
    if "" in canonical_uids or len(set(canonical_uids)) != len(canonical_uids):
        raise ValueError("canonical Humor UIDs are missing or duplicated")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metric_ids = {
        normalize_space(row.get("metric_id")) for row in bank.get("metrics") or []
    }
    if "" in metric_ids or len(metric_ids) != int(bank_meta.get("count", -1)):
        raise ValueError("Humor bank identities/count differ")
    bank_sha = normalize_space(bank_meta.get("source_sha256"))
    if not bank_sha or bank.get("source_sha256") not in (None, bank_sha):
        raise ValueError("Humor bank source identity differs")

    canonical_set = set(canonical_uids)
    truth, truth_counts, truth_defaulted_confidence = _index(
        truth_path,
        label="authoritative truth",
        expected_count=args.expected_truth,
        allowed_decisions=set(DECISIONS),
        canonical_uids=canonical_set,
        metric_ids=metric_ids,
        bank_sha=bank_sha,
        allow_missing_confidence=True,
    )
    predictions, prediction_counts, prediction_defaulted_confidence = _index(
        predictions_path,
        label="production predictions",
        expected_count=args.expected_predictions,
        allowed_decisions=set(FINAL_DECISIONS),
        canonical_uids=canonical_set,
        metric_ids=metric_ids,
        bank_sha=bank_sha,
    )
    if prediction_defaulted_confidence:
        raise AssertionError("production confidence defaulting is forbidden")
    overlap = set(truth) & set(predictions)
    union = set(truth) | set(predictions)
    if overlap or union != canonical_set:
        raise ValueError(
            "truth/prediction partition differs from canonical: "
            f"overlap={len(overlap)} missing={len(canonical_set-union)} "
            f"extra={len(union-canonical_set)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    created: list[Path] = []
    output_counts: Counter[str] = Counter()
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for canonical_row in canonical:
                uid = normalize_space(canonical_row["norm_uid"])
                authoritative = uid in truth
                source = truth[uid] if authoritative else predictions[uid]
                decision = str(source["decision"])
                row = {
                    **source,
                    "schema_version": SCHEMA,
                    "norm_uid": uid,
                    "task": args.task,
                    "corpus": args.corpus,
                    "row": canonical_row.get("row"),
                    "bank_source_sha256": bank_sha,
                    "decision": decision,
                    "metric_id": source["metric_id"],
                    "confidence": source["confidence"],
                    "verification_status": (
                        "AUTHORITATIVE_JOINED_TRUTH_OVERLAY"
                        if authoritative
                        else normalize_space(source.get("verification_status"))
                        or "PRODUCTION_PREDICTION_COMPLEMENT"
                    ),
                    "release_source": (
                        "AUTHORITATIVE_JOINED_TRUTH"
                        if authoritative
                        else "PRODUCTION_PREDICTION"
                    ),
                    "release_source_sha256": truth_sha if authoritative else predictions_sha,
                    "authoritative_truth_overlay": authoritative,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                output_counts[decision] += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        created.append(output_path)

        audit = audit_outputs(
            manifest_path,
            [output_path],
            tasks={args.task},
            corpora={args.corpus},
        )
        _write_json_new(audit_path, audit)
        created.append(audit_path)
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "COMPLETE_CANONICAL_AUTHORITATIVE_OVERLAY_AUDITED",
            "task": args.task,
            "corpus": args.corpus,
            "inputs": {
                "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
                "canonical": {
                    "path": str(canonical_path),
                    "sha256": sha256_file(canonical_path),
                    "count": len(canonical),
                },
                "bank": {
                    "path": str(bank_path),
                    "sha256": sha256_file(bank_path),
                    "source_sha256": bank_sha,
                    "metric_count": len(metric_ids),
                },
                "authoritative_truth": {
                    "path": str(truth_path),
                    "sha256": truth_sha,
                    "count": len(truth),
                    "decision_counts": dict(sorted(truth_counts.items())),
                    "missing_confidence_defaulted_to_medium": truth_defaulted_confidence,
                },
                "production_predictions": {
                    "path": str(predictions_path),
                    "sha256": predictions_sha,
                    "count": len(predictions),
                    "decision_counts": dict(sorted(prediction_counts.items())),
                },
            },
            "partition_audit": {
                "truth_prediction_uid_overlap": 0,
                "union_equals_canonical": True,
                "canonical_order_preserved": True,
                "all_match_metrics_in_bank": True,
                "all_nonmatch_metrics_null": True,
                "decision_taxonomy_valid": True,
            },
            "output": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "count": len(canonical),
                "decision_counts": dict(sorted(output_counts.items())),
            },
            "final_audit": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
                "complete": audit["complete"],
                "audited_rows": audit["audited_rows"],
            },
            "freeze_inputs_ready": {
                "manifest": str(manifest_path),
                "final": str(output_path),
                "final_audit": str(audit_path),
            },
        }
        _write_json_new(report_path, report)
        return report
    except BaseException:
        temporary.unlink(missing_ok=True)
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--truth-sha256", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--task", default="humor")
    parser.add_argument("--corpus", default="humor_multi")
    parser.add_argument("--expected-canonical", type=int, default=77378)
    parser.add_argument("--expected-truth", type=int, default=22090)
    parser.add_argument("--expected-predictions", type=int, default=55288)
    return parser.parse_args(argv)


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
