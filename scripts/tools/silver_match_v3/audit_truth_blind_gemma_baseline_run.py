#!/usr/bin/env python3
"""Audit completed truth-blind Gemma outputs and their raw-response transcripts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .adjudicate_gemma import (
    build_item_prompt,
    ordered_candidates,
    parse_response,
    prompt_sha256,
)
from .common import read_jsonl, sha256_file


RUN_SCHEMA = "silver-match-v3-truth-blind-gemma-baseline-run-v1"
RUN_STATUS = "TRUTH_BLIND_MULTI_ORDER_FULL_BANK_BASELINE_COMPLETE_UNSCORED"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _bound(path: Path, sha256: str, *, label: str) -> Path:
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != sha256:
        raise ValueError(f"missing or hash-drifted {label}: {path}")
    return path


def _resolve(raw: str, anchor: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _report_by_order(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("order_reports") or []
    result = {str(row.get("order") or ""): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError("run report has missing or duplicate order reports")
    return result


def audit(
    queue_path: Path,
    run_report_path: Path,
    failure_record_path: Path | None = None,
) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    run_report_path = run_report_path.resolve()
    queue = _json(queue_path)
    report = _json(run_report_path)
    queue_sha = sha256_file(queue_path)
    orders = list((queue.get("runtime") or {}).get("orders") or [])
    if (
        report.get("schema_version") != RUN_SCHEMA
        or report.get("status") != RUN_STATUS
        or (report.get("queue") or {}).get("sha256") != queue_sha
        or report.get("truth_select_predictions_mi_and_outcomes_read") is not False
        or report.get("scored_against_optimize_truth") is not False
        or report.get("eligible_for_truth_consensus") is not False
        or report.get("orders") != orders
    ):
        raise ValueError("run report is not the frozen completed unscored baseline")

    inputs = queue.get("inputs") or {}
    candidate_path = _bound(
        Path(inputs["candidates"]["path"]),
        str(inputs["candidates"]["sha256"]),
        label="candidate slate",
    )
    bank_path = _bound(
        Path(inputs["bank"]["path"]),
        str(inputs["bank"]["sha256"]),
        label="bank",
    )
    manifest_path = _bound(
        Path(inputs["manifest"]["path"]),
        str(inputs["manifest"]["sha256"]),
        label="manifest",
    )
    prompt_paths = [
        _bound(Path(row["path"]), str(row["sha256"]), label="prompt component")
        for row in inputs.get("prompt_components") or []
    ]
    if not prompt_paths:
        raise ValueError("queue has no prompt components")

    candidates = list(read_jsonl(candidate_path))
    candidate_uids = [str(row.get("norm_uid") or "") for row in candidates]
    candidate_by_uid = {str(row["norm_uid"]): row for row in candidates}
    if (
        "" in candidate_uids
        or len(candidate_uids) != len(candidate_by_uid)
        or len(candidates) != int(report.get("row_count", -1))
    ):
        raise ValueError("candidate UID universe differs from completed run")
    bank = _json(bank_path)
    metrics = list(bank.get("metrics") or [])
    metric_by_id = {str(row.get("metric_id") or ""): row for row in metrics}
    bank_ids = list(metric_by_id)
    if (
        "" in metric_by_id
        or len(metric_by_id) != len(metrics)
        or len(metrics) != int(report.get("bank_metric_count", -1))
    ):
        raise ValueError("bank universe differs from completed run")

    manifest = _json(manifest_path)
    wanted = set(candidate_uids)
    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in (manifest.get("corpora") or {}).items():
        if meta.get("task") != queue.get("task"):
            continue
        corpus_path = _resolve(str(meta.get("path") or ""), manifest_path)
        for row in read_jsonl(corpus_path):
            uid = str(row.get("norm_uid") or "")
            if uid in wanted:
                if uid in norms or row.get("corpus") != corpus:
                    raise ValueError(f"canonical norm identity drift: {uid}")
                norms[uid] = row
    if set(norms) != wanted:
        raise ValueError("canonical manifest does not cover the run UID universe")

    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    if prompt_sha256(system_prompt) != (queue.get("prompt") or {}).get(
        "combined_sha256"
    ):
        raise ValueError("combined prompt hash drift")
    runtime = queue.get("runtime") or {}
    report_orders = _report_by_order(report)
    if set(report_orders) != set(orders):
        raise ValueError("run report order topology differs from queue")

    order_audits: list[dict[str, Any]] = []
    for order in orders:
        output_path = Path(queue["outputs"][order]["path"]).resolve()
        order_report = report_orders[order]
        output_ref = order_report.get("output") or {}
        _bound(output_path, str(output_ref.get("sha256") or ""), label=f"{order} output")
        rows = list(read_jsonl(output_path))
        if [str(row.get("norm_uid") or "") for row in rows] != candidate_uids:
            raise ValueError(f"{order} output is not in exact frozen candidate order")

        decision_counts: Counter[str] = Counter()
        parsed_retry_safe_count = 0
        for row, candidate in zip(rows, candidates):
            uid = str(row["norm_uid"])
            cards = ordered_candidates(list(candidate["candidates"]), order, uid)
            candidate_ids = [str(card["metric_id"]) for card in cards]
            rendered = build_item_prompt(
                system_prompt,
                norms[uid],
                cards,
                metric_by_id,
                context_chars=int(runtime["context_chars"]),
                description_chars=int(runtime["description_chars"]),
                example_chars=int(runtime["example_chars"]),
                max_examples=int(runtime["max_examples"]),
            )
            raw = row.get("raw_response")
            if (
                row.get("task") != queue.get("task")
                or row.get("corpus") != norms[uid].get("corpus")
                or row.get("row") != norms[uid].get("row")
                or row.get("order_mode") != order
                or row.get("candidate_ids") != candidate_ids
                or set(candidate_ids) != set(bank_ids)
                or len(candidate_ids) != len(bank_ids)
                or row.get("candidate_bank_source_sha256")
                != (queue.get("scientific_contract") or {}).get("bank_source_sha256")
                or row.get("prompt_sha256")
                != (queue.get("prompt") or {}).get("combined_sha256")
                or row.get("item_prompt_sha256") != prompt_sha256(rendered)
                or row.get("model") != runtime.get("model")
                or not isinstance(raw, str)
                or not raw.strip()
            ):
                raise ValueError(f"{order} transcript lineage drift: {uid}")
            parsed, error = parse_response(raw, set(candidate_ids))
            if parsed is None:
                expected = {
                    "decision": "INVALID_OUTPUT",
                    "metric_id": None,
                    "confidence": "low",
                    "reason": error,
                    "parse_error": error,
                }
            else:
                expected = {
                    "decision": parsed["decision"],
                    "metric_id": parsed["metric_id"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "parse_error": None,
                }
                parsed_retry_safe_count += 1
            observed = {key: row.get(key) for key in expected}
            if observed != expected:
                raise ValueError(f"{order} raw-response reparse mismatch: {uid}")
            decision_counts[str(row["decision"])] += 1

        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta_ref = order_report.get("meta") or {}
        _bound(meta_path, str(meta_ref.get("sha256") or ""), label=f"{order} metadata")
        meta = _json(meta_path)
        if (
            meta.get("output_sha256") != sha256_file(output_path)
            or meta.get("input_candidates_sha256") != sha256_file(candidate_path)
            or meta.get("prompt_sha256") != (queue.get("prompt") or {}).get(
                "combined_sha256"
            )
            or meta.get("order_mode") != order
            or int(meta.get("invalid_count", -1))
            != int(order_report.get("final_invalid_output_count", -2))
        ):
            raise ValueError(f"{order} metadata/report binding drift")
        runner_log_ref = order_report.get("runner_log") or {}
        runner_log = _bound(
            Path(str(runner_log_ref.get("path") or "")),
            str(runner_log_ref.get("sha256") or ""),
            label=f"{order} runner transcript",
        )
        log_text = runner_log.read_text(encoding="utf-8", errors="replace")
        if (
            f"adjudicated={len(rows)}/{len(rows)}" not in log_text
            or "Traceback (most recent call last)" in log_text
            or "cannot find -lcuda" in log_text
        ):
            raise ValueError(f"{order} runner transcript is not a clean completion")
        order_audits.append(
            {
                "order": order,
                "rows": len(rows),
                "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
                "metadata": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
                "runner_transcript": {
                    "path": str(runner_log),
                    "sha256": sha256_file(runner_log),
                },
                "raw_response_present_count": len(rows),
                "raw_response_exact_reparse_count": parsed_retry_safe_count,
                "invalid_output_count": decision_counts.get("INVALID_OUTPUT", 0),
                "decision_counts": dict(sorted(decision_counts.items())),
            }
        )

    failure_record = None
    if failure_record_path is not None:
        failure_record_path = failure_record_path.resolve()
        failure_record = _json(failure_record_path)
        failed_queue = failure_record.get("frozen_queue") or {}
        failed_attempt = failure_record.get("failed_attempt") or {}
        failed_log = failed_attempt.get("runner_log") or {}
        _bound(
            Path(str(failed_log.get("path") or "")),
            str(failed_log.get("sha256") or ""),
            label="failed attempt transcript",
        )
        if (
            failed_queue.get("sha256") != queue_sha
            or failure_record.get("status")
            != "FAILED_CLOSED_ZERO_ROWS_RUNTIME_LINKER_REPAIR_BOUND"
            or (failure_record.get("scientific_state") or {}).get(
                "predictions_written"
            )
            != 0
            or failed_attempt.get("root_cause")
            != "system linker could not resolve -lcuda"
        ):
            raise ValueError("failed-attempt record does not bind the zero-row retry")
        failure_record = {
            "path": str(failure_record_path),
            "sha256": sha256_file(failure_record_path),
        }

    return {
        "schema_version": "silver-match-v3-truth-blind-gemma-run-transcript-audit-v1",
        "status": "COMPLETE_RAW_TRANSCRIPTS_EXACT_REPARSE_AND_LINEAGE_PASS",
        "task": queue.get("task"),
        "queue": {"path": str(queue_path), "sha256": queue_sha},
        "run_report": {
            "path": str(run_report_path),
            "sha256": sha256_file(run_report_path),
        },
        "failure_record": failure_record,
        "orders": order_audits,
        "truth_labels_select_rows_prior_predictions_mi_and_outcomes_read": False,
        "scoring_performed": False,
        "eligible_for_truth_consensus": False,
        "auditor": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--failure-record")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(
        Path(args.queue),
        Path(args.run_report),
        Path(args.failure_record) if args.failure_record else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "output": str(output), "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
