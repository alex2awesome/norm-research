#!/usr/bin/env python3
"""Run or safely resume an audited truth-blind Gemma full-bank queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES, DECISIONS, ordered_candidates
from .audit_exact_directory_inventory import assert_exact_inventory
from .common import read_jsonl, sha256_file
from .freeze_python_runtime_dependency_inventory import (
    assert_exact_runtime_dependencies,
)


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bank_source_sha256(queue: dict[str, Any]) -> str:
    contract_value = str(
        (queue.get("scientific_contract") or {}).get("bank_source_sha256") or ""
    )
    input_value = str(((queue.get("inputs") or {}).get("bank") or {}).get("source_sha256") or "")
    if not contract_value or contract_value != input_value:
        raise ValueError("queue contract and bound bank input do not share one source hash")
    return contract_value


def _rehash_all_bound_inputs(queue: dict[str, Any]) -> dict[str, Any]:
    inputs = queue.get("inputs") or {}
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("queue has no bound input topology")
    rows: list[dict[str, Any]] = []

    def validate(name: str, entry: dict[str, Any]) -> None:
        path = Path(str(entry.get("path") or "")).resolve()
        if (
            not path.is_file()
            or sha256_file(path) != str(entry.get("sha256") or "")
            or path.stat().st_size != int(entry.get("bytes", -1))
        ):
            raise ValueError(f"bound input missing or hash/size drifted: {name}")
        rows.append(
            {
                "name": name,
                "path": str(path),
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
            }
        )

    for name, entry in sorted(inputs.items()):
        if name == "prompt_components":
            if not isinstance(entry, list) or not entry:
                raise ValueError("bound prompt component topology is empty")
            for index, component in enumerate(entry):
                validate(f"prompt_components[{index}]", component)
        elif isinstance(entry, dict):
            validate(name, entry)
        else:
            raise ValueError(f"unrecognized bound input entry: {name}")
    return {
        "status": "ALL_BOUND_INPUT_PATH_HASH_SIZE_RECHECK_PASS",
        "input_file_count": len(rows),
        "inputs": rows,
    }


def _validate_completed_order(
    *,
    queue: dict[str, Any],
    order: str,
    candidates_by_uid: dict[str, dict[str, Any]],
    expected_bank_ids: list[str],
) -> dict[str, Any]:
    bank_source_sha256 = _bank_source_sha256(queue)
    expected_uid_set = set(candidates_by_uid)
    expected_bank_id_set = set(expected_bank_ids)
    output = Path(str(queue["outputs"][order]["path"])).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if not output.is_file() or not meta_path.is_file():
        raise ValueError(f"{order} did not produce both output and metadata")
    rows = list(read_jsonl(output))
    row_uids = [str(row.get("norm_uid") or "") for row in rows]
    if (
        len(rows) != len(expected_uid_set)
        or set(row_uids) != expected_uid_set
        or len(row_uids) != len(set(row_uids))
    ):
        raise ValueError(f"{order} output is not an exact full-bank completion")
    for row in rows:
        uid = str(row["norm_uid"])
        expected_order = [
            str(card["metric_id"])
            for card in ordered_candidates(
                list(candidates_by_uid[uid]["candidates"]), order, uid
            )
        ]
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        confidence = str(row.get("confidence") or "")
        parse_error = row.get("parse_error")
        if (
            row.get("task") != queue["task"]
            or row.get("order_mode") != order
            or row.get("candidate_bank_source_sha256") != bank_source_sha256
            or [str(value) for value in row.get("candidate_ids") or []]
            != expected_order
            or len(expected_order) != len(expected_bank_ids)
            or set(expected_order) != expected_bank_id_set
            or row.get("model") != queue["runtime"]["model"]
            or row.get("prompt_sha256") != queue["prompt"]["combined_sha256"]
        ):
            raise ValueError(f"{order} row lineage/order drift for {uid}")
        if decision == "INVALID_OUTPUT":
            if metric_id is not None or confidence != "low" or not parse_error:
                raise ValueError(f"{order} malformed INVALID_OUTPUT row for {uid}")
        elif decision in DECISIONS:
            if (
                confidence not in CONFIDENCES
                or not str(row.get("reason") or "").strip()
                or parse_error is not None
                or (decision == "MATCH" and metric_id not in expected_bank_id_set)
                or (decision != "MATCH" and metric_id is not None)
            ):
                raise ValueError(f"{order} invalid decision schema for {uid}")
        else:
            raise ValueError(f"{order} unknown decision for {uid}: {decision}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prompt_entries = queue["inputs"]["prompt_components"]
    expected_prompt_component_sha256 = {
        str(Path(entry["path"]).resolve()): entry["sha256"] for entry in prompt_entries
    }
    expected_rendering = {
        key: queue["runtime"][key]
        for key in ("context_chars", "description_chars", "example_chars", "max_examples")
    }
    expected_runtime = {
        key: queue["runtime"][key]
        for key in (
            "temperature",
            "seed",
            "batch_size",
            "max_model_len",
            "max_tokens",
            "gpu_memory_utilization",
            "keep_raw",
            "resume",
        )
    }
    if (
        Path(str(meta.get("input_candidates") or "")).resolve()
        != Path(queue["inputs"]["candidates"]["path"]).resolve()
        or meta.get("input_candidates_sha256")
        != queue["inputs"]["candidates"]["sha256"]
        or Path(str(meta.get("output") or "")).resolve() != output
        or meta.get("output_sha256") != sha256_file(output)
        or Path(str(meta.get("prompt") or "")).resolve()
        != Path(prompt_entries[0]["path"]).resolve()
        or [Path(value).resolve() for value in meta.get("prompt_addons") or []]
        != [Path(entry["path"]).resolve() for entry in prompt_entries[1:]]
        or meta.get("prompt_component_sha256") != expected_prompt_component_sha256
        or meta.get("prompt_sha256") != queue["prompt"]["combined_sha256"]
        or meta.get("model") != queue["runtime"]["model"]
        or Path(str(meta.get("python_executable") or "")).resolve()
        != Path(queue["runtime"]["python"]).resolve()
        or meta.get("order_mode") != order
        or int(meta.get("max_candidates", -1)) != len(expected_bank_ids)
        or meta.get("prompt_rendering") != expected_rendering
        or meta.get("runtime") != expected_runtime
        or int(meta.get("shard_id", -1)) != 0
        or int(meta.get("num_shards", -1)) != 1
    ):
        raise ValueError(f"{order} metadata does not bind the completed output")
    final_invalid_output_count = sum(
        row.get("decision") == "INVALID_OUTPUT" for row in rows
    )
    final_parse_error_count = sum(row.get("parse_error") is not None for row in rows)
    if final_invalid_output_count != final_parse_error_count:
        raise ValueError(f"{order} has inconsistent final invalid/parse-error rows")
    return {
        "order": order,
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": len(rows)},
        "meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
        "final_invalid_output_count": final_invalid_output_count,
        "final_parse_error_count": final_parse_error_count,
        "invalid_count_in_final_invocation_metadata": int(meta.get("invalid_count", -1)),
        "elapsed_seconds_in_final_invocation": meta.get("elapsed_seconds"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    queue_path = Path(args.queue).resolve()
    prelaunch_path = Path(args.prelaunch_audit).resolve()
    inventory_path = Path(args.implementation_inventory).resolve()
    report_path = Path(args.output).resolve()
    if report_path.exists():
        raise FileExistsError(report_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    _bank_source_sha256(queue)
    bound_inputs_before = _rehash_all_bound_inputs(queue)
    prelaunch = json.loads(prelaunch_path.read_text(encoding="utf-8"))
    if (
        prelaunch.get("status")
        != "EXACT_HASH_PINNED_MULTI_ORDER_QUEUE_PASS_WAITING_FOR_GPU"
        or (prelaunch.get("queue") or {}).get("sha256") != sha256_file(queue_path)
        or prelaunch.get("outputs_absent_before_launch") is not True
        or prelaunch.get("implementation_snapshot_exact_recursive_pycache_free") is not True
        or prelaunch.get("model_snapshot_exact_recursive_hash_bound") is not True
        or prelaunch.get("python_runtime_dependencies_exactly_bound") is not True
        or prelaunch.get("interruption_safe_append_only_resume") is not True
    ):
        raise ValueError("queue lacks the exact clean prelaunch attestation")
    expected_implementation_inventory = Path(
        queue["inputs"]["implementation_inventory"]["path"]
    ).resolve()
    model_inventory_path = Path(queue["inputs"]["model_inventory"]["path"]).resolve()
    dependency_inventory_path = Path(
        queue["inputs"]["runtime_dependency_inventory"]["path"]
    ).resolve()
    if inventory_path != expected_implementation_inventory:
        raise ValueError("launcher implementation inventory differs from frozen queue")
    inventory_audit_before = assert_exact_inventory(inventory_path)
    model_inventory_audit_before = assert_exact_inventory(model_inventory_path)
    dependency_audit_before = assert_exact_runtime_dependencies(
        dependency_inventory_path
    )
    environment = {str(k): str(v) for k, v in (queue["runtime"]["environment"] or {}).items()}
    if environment.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("runtime is not bytecode-write-disabled")
    env = os.environ.copy()
    env.update(environment)

    candidates = list(read_jsonl(Path(queue["inputs"]["candidates"]["path"])))
    candidates_by_uid = {str(row["norm_uid"]): row for row in candidates}
    bank = json.loads(Path(queue["inputs"]["bank"]["path"]).read_text(encoding="utf-8"))
    expected_bank_ids = [str(row["metric_id"]) for row in bank["metrics"]]
    contract = queue.get("scientific_contract") or {}
    if (
        len(candidates) != int(contract.get("row_count", -1))
        or len(candidates_by_uid) != len(candidates)
        or len(expected_bank_ids) != int(contract.get("candidate_depth", -1))
        or len(expected_bank_ids) != len(set(expected_bank_ids))
        or contract.get("truth_labels_select_rows_prior_predictions_mi_and_outcomes_hidden")
        is not True
    ):
        raise ValueError("runtime inputs violate the frozen truth-blind contract")

    order_reports: list[dict[str, Any]] = []
    for order in queue["runtime"]["orders"]:
        output = Path(queue["outputs"][order]["path"]).resolve()
        meta = output.with_suffix(output.suffix + ".meta.json")
        if output.exists() and meta.exists():
            order_reports.append(
                {**_validate_completed_order(
                    queue=queue,
                    order=order,
                    candidates_by_uid=candidates_by_uid,
                    expected_bank_ids=expected_bank_ids,
                ), "already_complete_before_invocation": True}
            )
            continue
        if meta.exists() and not output.exists():
            raise ValueError(f"orphan metadata without output for {order}")
        attempt = 1
        while True:
            log_path = report_path.parent / f"{order}.runner.attempt-{attempt:04d}.log"
            if not log_path.exists():
                break
            attempt += 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("xb") as log:
            completed = subprocess.run(
                [str(value) for value in queue["commands"][order]],
                cwd=environment["PYTHONPATH"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{order} exited {completed.returncode}; partial JSONL is preserved for --resume"
            )
        order_reports.append(
            {
                **_validate_completed_order(
                    queue=queue,
                    order=order,
                    candidates_by_uid=candidates_by_uid,
                    expected_bank_ids=expected_bank_ids,
                ),
                "already_complete_before_invocation": False,
                "runner_log": {"path": str(log_path), "sha256": sha256_file(log_path)},
            }
        )
        _rehash_all_bound_inputs(queue)
        assert_exact_inventory(inventory_path)
        assert_exact_inventory(model_inventory_path)
        assert_exact_runtime_dependencies(dependency_inventory_path)

    bound_inputs_after = _rehash_all_bound_inputs(queue)
    inventory_audit_after = assert_exact_inventory(inventory_path)
    model_inventory_audit_after = assert_exact_inventory(model_inventory_path)
    dependency_audit_after = assert_exact_runtime_dependencies(
        dependency_inventory_path
    )
    result = {
        "schema_version": "silver-match-v3-truth-blind-gemma-baseline-run-v1",
        "status": "TRUTH_BLIND_MULTI_ORDER_FULL_BANK_BASELINE_COMPLETE_UNSCORED",
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "prelaunch_audit": {"path": str(prelaunch_path), "sha256": sha256_file(prelaunch_path)},
        "task": queue["task"],
        "role": queue["role"],
        "row_count": len(candidates),
        "bank_metric_count": len(expected_bank_ids),
        "orders": list(queue["runtime"]["orders"]),
        "order_reports": order_reports,
        "truth_select_predictions_mi_and_outcomes_read": False,
        "scored_against_optimize_truth": False,
        "eligible_for_truth_consensus": False,
        "implementation_inventory_before": inventory_audit_before,
        "implementation_inventory_after": inventory_audit_after,
        "model_inventory_before": model_inventory_audit_before,
        "model_inventory_after": model_inventory_audit_after,
        "runtime_dependencies_before": dependency_audit_before,
        "runtime_dependencies_after": dependency_audit_after,
        "bound_inputs_before": bound_inputs_before,
        "bound_inputs_after": bound_inputs_after,
    }
    _write_once(report_path, result)
    return {**result, "report_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--prelaunch-audit", required=True)
    parser.add_argument("--implementation-inventory", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
