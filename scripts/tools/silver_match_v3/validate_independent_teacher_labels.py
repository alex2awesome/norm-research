#!/usr/bin/env python3
"""Validate raw structured Codex labels against an immutable teacher pack."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .common import read_jsonl, sha256_file, write_jsonl


COMPOSITE_TRANSCRIPT_SCHEMA = (
    "silver-match-v3-composite-openrouter-direct-vllm-transcript-audit-v1"
)


def _bound_file(reference: dict[str, Any]) -> bool:
    """Return whether an absolute path/hash audit reference is still exact."""

    path_value = str(reference.get("path") or "")
    expected = str(reference.get("sha256") or "")
    if not path_value or not expected:
        return False
    path = Path(path_value).resolve()
    return path.is_file() and sha256_file(path) == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--raw-label-dir", required=True)
    parser.add_argument("--retrieval-candidates")
    parser.add_argument(
        "--transcript-audit",
        help="Optional PASS audit from audit_isolated_labeler_transcripts; when supplied it is hash-bound to every chunk/raw label.",
    )
    parser.add_argument(
        "--split-role",
        choices=("train", "dev", "test"),
        help="Required when audit-pack items intentionally omit a canonical split field.",
    )
    parser.add_argument("--annotator", default="codex-gpt-5.6-sol-ultra")
    parser.add_argument(
        "--label-source",
        default="independent_codex_full_bank",
        help="Explicit provenance string for the independent labeling backend.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    pack_root = Path(args.pack_root).resolve()
    raw_root = Path(args.raw_label_dir).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite validated independent labels")
    pack_report_path = pack_root / "validation.json"
    pack_report = json.loads(pack_report_path.read_text(encoding="utf-8"))
    items_path, bank_path = pack_root / "items.jsonl", pack_root / "bank.json"
    if sha256_file(items_path) != pack_report["outputs"]["items"]["sha256"]:
        raise ValueError("teacher-pack items hash mismatch")
    if sha256_file(bank_path) != pack_report["outputs"]["bank"]["sha256"]:
        raise ValueError("teacher-pack bank hash mismatch")
    task = str(pack_report["task"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("task") != task or bank.get("source_sha256") != pack_report.get(
        "bank_source_sha256"
    ):
        raise ValueError("teacher-pack bank identity mismatch")
    bank_ids = {str(row["metric_id"]) for row in bank["metrics"]}
    items = list(read_jsonl(items_path))
    item_by_uid = {str(row["norm_uid"]): row for row in items}
    if len(item_by_uid) != len(items):
        raise ValueError("teacher pack has duplicate item UIDs")

    retrieval_ranks = None
    retrieval_hash = None
    if args.retrieval_candidates:
        retrieval_path = Path(args.retrieval_candidates).resolve()
        retrieval_hash = sha256_file(retrieval_path)
        retrieval_ranks = {}
        for row in read_jsonl(retrieval_path):
            uid = str(row["norm_uid"])
            if uid not in item_by_uid:
                continue
            if uid in retrieval_ranks:
                raise ValueError(f"duplicate retrieval UID: {uid}")
            retrieval_ranks[uid] = {
                str(value["metric_id"]): index
                for index, value in enumerate(row.get("candidates") or [], 1)
            }
        if set(retrieval_ranks) != set(item_by_uid):
            missing = sorted(set(item_by_uid) - set(retrieval_ranks))
            raise ValueError(f"retrieval candidates are incomplete: {missing[:3]}")

    labels_by_uid: dict[str, dict[str, Any]] = {}
    raw_hashes = {}
    chunk_reports = {}
    chunk_paths = sorted((pack_root / "chunks").glob("part-*.jsonl"))
    for chunk_path in chunk_paths:
        chunk = chunk_path.stem
        expected = [str(row["norm_uid"]) for row in read_jsonl(chunk_path)]
        raw_path = raw_root / f"{chunk}.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw label chunk missing: {raw_path}")
        raw_hashes[str(raw_path)] = sha256_file(raw_path)
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if payload.get("task") != task or payload.get("chunk_id") != chunk:
            raise ValueError(f"raw label task/chunk mismatch: {raw_path}")
        values = payload.get("labels")
        if not isinstance(values, list) or len(values) != len(expected):
            raise ValueError(f"raw label count mismatch: {raw_path}")
        observed = [str(row.get("norm_uid") or "") for row in values]
        if len(set(observed)) != len(observed) or set(observed) != set(expected):
            raise ValueError(f"raw label UID coverage mismatch: {raw_path}")
        for raw in values:
            uid = str(raw["norm_uid"])
            if uid in labels_by_uid:
                raise ValueError(f"duplicate label across chunks: {uid}")
            decision = str(raw.get("decision") or "").upper()
            confidence = str(raw.get("confidence") or "").lower()
            reason = str(raw.get("reason") or "").strip()
            metric_id = raw.get("metric_id")
            metric_id = None if metric_id is None else str(metric_id)
            if decision not in DECISIONS or confidence not in CONFIDENCES or not reason:
                raise ValueError(f"invalid decision/confidence/reason: {uid}")
            if decision == "MATCH":
                if metric_id not in bank_ids:
                    raise ValueError(f"MATCH metric absent from frozen bank: {uid}/{metric_id}")
            elif metric_id is not None:
                raise ValueError(f"abstention carries a metric ID: {uid}")
            rank = (
                retrieval_ranks[uid].get(metric_id)
                if retrieval_ranks is not None and metric_id is not None
                else None
            )
            item = item_by_uid[uid]
            split_group = item.get("split_group") or item.get("source_group")
            if not split_group:
                raise ValueError(f"teacher-pack item lacks source-group provenance: {uid}")
            split = item.get("split") or args.split_role
            if not split:
                raise ValueError(
                    f"teacher-pack item lacks split; pass --split-role explicitly: {uid}"
                )
            labels_by_uid[uid] = {
                "schema_version": item["schema_version"],
                "norm_uid": uid,
                "corpus": item["corpus"],
                "task": task,
                "row": item["row"],
                "split_group": split_group,
                "split": split,
                "decision": decision,
                "metric_id": metric_id,
                "current_bank_source_sha256": pack_report["bank_source_sha256"],
                "confidence": confidence,
                "reason": reason,
                "label_source": args.label_source,
                "annotator": args.annotator,
                "retrieved_rank": rank,
                "training_eligible_preverification": (
                    decision == "MATCH" and confidence == "high"
                ),
                "raw_label_chunk": str(raw_path),
                "raw_label_chunk_sha256": raw_hashes[str(raw_path)],
            }
        chunk_reports[chunk] = {"count": len(values), "raw_sha256": raw_hashes[str(raw_path)]}

    if set(labels_by_uid) != set(item_by_uid):
        missing = sorted(set(item_by_uid) - set(labels_by_uid))
        raise ValueError(f"validated label coverage incomplete: {missing[:3]}")
    transcript_audit_ref = None
    if args.transcript_audit:
        audit_path = Path(args.transcript_audit).resolve()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_chunks = {str(row.get("chunk") or ""): row for row in audit.get("chunks") or []}
        expected_chunks = {path.stem: path for path in chunk_paths}
        transcript_schema = audit.get("schema_version")
        common_valid = (
            audit.get("status") == "PASS"
            and audit.get("complete") is True
            and int(audit.get("expected_chunks", -1)) == len(chunk_paths)
            and int(audit.get("audited_chunks", -1)) == len(chunk_paths)
            and set(audit_chunks) == set(expected_chunks)
            and (audit.get("bank") or {}).get("sha256") == sha256_file(bank_path)
        )
        if transcript_schema == "silver-match-v3-isolated-labeler-transcript-audit-v1":
            backend_valid = audit.get("violations") == []
        elif transcript_schema == "silver-match-v3-openrouter-labeler-transcript-audit-v1":
            contract = audit.get("contract") or {}
            backend_valid = (
                audit.get("truth_hidden") is True
                and audit.get("violations") == []
                and (audit.get("items") or {}).get("sha256") == sha256_file(items_path)
                and (audit.get("pack_validation") or {}).get("sha256")
                == sha256_file(pack_report_path)
                and contract.get(
                    "exact_request_body_reconstructed_from_only_frozen_bank_chunk_and_guides"
                )
                is True
                and contract.get(
                    "sample_keys_predictions_proposals_mi_and_outcomes_absent"
                )
                is True
                and contract.get("api_credentials_not_logged") is True
            )
        elif transcript_schema == "silver-match-v3-claude-labeler-transcript-audit-v1":
            contract = audit.get("contract") or {}
            backend_valid = (
                audit.get("truth_hidden") is True
                and audit.get("violations") == []
                and (audit.get("items") or {}).get("sha256") == sha256_file(items_path)
                and (audit.get("pack_validation") or {}).get("sha256")
                == sha256_file(pack_report_path)
                and contract.get("only_read_and_structured_output_tools_observed")
                is True
                and contract.get(
                    "every_chunk_read_only_its_guides_bank_and_assigned_items"
                )
                is True
                and contract.get("network_and_mcp_use_absent") is True
                and contract.get("final_payload_exactly_bound_to_raw_labels") is True
            )
        elif transcript_schema == COMPOSITE_TRANSCRIPT_SCHEMA:
            contract = audit.get("contract") or {}
            backend_valid = (
                audit.get("truth_hidden") is True
                and audit.get("violations") == []
                and (audit.get("items") or {}).get("sha256") == sha256_file(items_path)
                and (audit.get("pack_validation") or {}).get("sha256")
                == sha256_file(pack_report_path)
                and contract.get("every_chunk_has_exactly_one_verified_backend") is True
                and contract.get("openrouter_chunks_retain_exact_request_transcripts")
                is True
                and contract.get("direct_vllm_rows_reparsed_from_raw_responses") is True
                and contract.get("direct_vllm_item_prompts_independently_reconstructed")
                is True
                and contract.get(
                    "sample_keys_predictions_proposals_mi_and_outcomes_absent"
                )
                is True
            )
        else:
            backend_valid = False
        if not common_valid or not backend_valid:
            raise ValueError("transcript audit is incomplete, failed, or bound to another pack")
        for chunk, chunk_path in expected_chunks.items():
            raw_path = raw_root / f"{chunk}.json"
            row = audit_chunks[chunk]
            event_count = (
                int(row.get("command_count", 0))
                if transcript_schema == "silver-match-v3-isolated-labeler-transcript-audit-v1"
                else (
                    int(row.get("request_count", 0))
                    if transcript_schema
                    == "silver-match-v3-openrouter-labeler-transcript-audit-v1"
                    else int(row.get("event_count", 0))
                )
            )
            transcript_bound = True
            if transcript_schema == "silver-match-v3-openrouter-labeler-transcript-audit-v1":
                transcript_path = pack_root / "api_transcripts" / f"{chunk}.json"
                transcript_bound = (
                    transcript_path.is_file()
                    and row.get("transcript_sha256") == sha256_file(transcript_path)
                )
            elif transcript_schema == "silver-match-v3-claude-labeler-transcript-audit-v1":
                transcript_path = Path(str(row.get("transcript_path") or "")).resolve()
                audited_raw_path = Path(str(row.get("raw_label_path") or "")).resolve()
                transcript_bound = (
                    transcript_path.is_file()
                    and audited_raw_path == raw_path.resolve()
                    and row.get("transcript_sha256") == sha256_file(transcript_path)
                )
            elif transcript_schema == COMPOSITE_TRANSCRIPT_SCHEMA:
                audited_raw_path = Path(str(row.get("raw_label_path") or "")).resolve()
                backend = str(row.get("backend") or "")
                backend_artifacts = row.get("backend_artifacts") or {}
                if backend == "openrouter_api":
                    transcript_ref = backend_artifacts.get("api_transcript") or {}
                    backend_bound = _bound_file(transcript_ref)
                elif backend == "direct_vllm":
                    required = (
                        "direct_output",
                        "direct_meta",
                        "candidates",
                        "runner",
                        "runner_log",
                        "frozen_plan",
                    )
                    backend_bound = all(
                        _bound_file(backend_artifacts.get(key) or {}) for key in required
                    )
                else:
                    backend_bound = False
                transcript_bound = (
                    audited_raw_path == raw_path.resolve()
                    and row.get("raw_label_sha256") == sha256_file(raw_path)
                    and backend_bound
                )
            if (
                row.get("chunk_sha256") != sha256_file(chunk_path)
                or row.get("raw_label_sha256") != sha256_file(raw_path)
                or event_count < 1
                or not transcript_bound
            ):
                raise ValueError(f"transcript audit chunk binding mismatch: {chunk}")
        transcript_audit_ref = {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "status": "PASS",
            "schema_version": transcript_schema,
            "model": audit.get("model"),
            "audited_chunks": len(chunk_paths),
        }
    labels = [labels_by_uid[str(item["norm_uid"])] for item in items]
    write_jsonl(output_path, labels)
    decision_counts = Counter(row["decision"] for row in labels)
    confidence_counts = Counter(row["confidence"] for row in labels)
    matches = [row for row in labels if row["decision"] == "MATCH"]
    report = {
        "schema_version": "silver-match-v3-independent-label-validation-v1",
        "task": task,
        "complete": True,
        "count": len(labels),
        "unique_uids": len(labels_by_uid),
        "unique_source_groups": len({row["split_group"] for row in labels}),
        "train_split_count": sum(row["split"] == "train" for row in labels),
        "decision_counts": dict(sorted(decision_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "match_count": len(matches),
        "high_confidence_match_count": sum(
            row["confidence"] == "high" for row in matches
        ),
        "retrieval_candidate_sha256": retrieval_hash,
        "retrieval_miss_count": (
            sum(row["retrieved_rank"] is None for row in matches)
            if retrieval_ranks is not None
            else None
        ),
        "retrieval_miss_at_50_count": (
            sum(
                row["retrieved_rank"] is None or row["retrieved_rank"] > 50
                for row in matches
            )
            if retrieval_ranks is not None
            else None
        ),
        "bank_source_sha256": pack_report["bank_source_sha256"],
        "pack_validation": {
            "path": str(pack_report_path),
            "sha256": sha256_file(pack_report_path),
        },
        "transcript_audit": transcript_audit_ref,
        "raw_chunks": chunk_reports,
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
