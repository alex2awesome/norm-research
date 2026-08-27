#!/usr/bin/env python3
"""Audit a frozen Claude full-bank label run for exact input and tool isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .run_claude_pack_labels import validate_payload


SCHEMA = "silver-match-v3-claude-labeler-transcript-audit-v1"
FREEZE_SCHEMA = "silver-match-v3-independent-claude-label-execution-freeze-v1"
REQUEST_SCHEMA = "silver-match-v3-independent-claude-label-request-v1"
RUN_SCHEMA = "silver-match-v3-independent-claude-pack-run-v1"
FORBIDDEN_ITEM_FIELDS = {
    "decision",
    "metric_id",
    "acceptable_metric_ids",
    "label",
    "prediction",
    "raw_response",
    "candidate_ids",
    "mi",
    "outcome",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()


def _ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    pack = Path(args.pack_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze_path = Path(args.execution_freeze).resolve()
    freeze = _load_object(freeze_path)
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("status")
        not in {
            "FROZEN_BEFORE_EITHER_CLAUDE_LABEL_PASS",
            "FROZEN_BEFORE_CLAUDE_RESOLVER_PASS",
        }
        or freeze.get("output_namespace") != args.output_namespace
        or args.pass_key not in (freeze.get("passes") or {})
    ):
        raise ValueError("invalid Claude pre-execution freeze")
    frozen = freeze["passes"][args.pass_key]
    if Path(str(frozen.get("root") or "")).resolve() != pack:
        raise ValueError("freeze pass root does not match audited pack")

    namespace = pack / args.output_namespace
    request_root = namespace / "requests"
    transcript_root = namespace / "transcripts"
    raw_root = namespace / "raw_labels"
    stderr_root = namespace / "stderr"
    summary_path = namespace / "RUN_SUMMARY.json"
    validation_path = pack / "validation.json"
    bank_path = pack / "bank.json"
    items_path = pack / "items.jsonl"
    chunks = sorted((pack / "chunks").glob("part-*.jsonl"))
    if not chunks:
        raise ValueError("pack has no chunks")

    violations: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    runtime_contract = freeze.get("runtime_contract") or {}
    for key in (
        "separate_process_per_chunk",
        "only_read_tool_available",
        "safe_mode",
        "no_session_persistence",
        "strict_empty_mcp_config",
        "pass_outputs_mutually_hidden",
        "prior_truth_proposals_model_outputs_mi_and_outcomes_hidden",
    ):
        require(runtime_contract.get(key) is True, f"execution-freeze contract missing: {key}")
    if freeze.get("status") == "FROZEN_BEFORE_CLAUDE_RESOLVER_PASS":
        require(
            runtime_contract.get("prior_labels_hidden_from_resolver") is True,
            "resolver execution freeze does not hide prior labels",
        )
    require(
        runtime_contract.get("Gemma_baseline_outputs_available_to_labelers") is False,
        "execution freeze does not exclude Gemma outputs",
    )
    for implementation in freeze.get("implementation") or []:
        implementation_path = Path(str(implementation.get("path") or "")).resolve()
        require(implementation_path.is_file(), f"frozen implementation missing: {implementation_path}")
        if implementation_path.is_file():
            require(
                implementation.get("sha256") == sha256_file(implementation_path),
                f"frozen implementation hash drift: {implementation_path.name}",
            )
    prelabel_ref = freeze.get("prelabel_independence_audit") or {}
    prelabel_path = Path(str(prelabel_ref.get("path") or "")).resolve()
    if freeze.get("status") == "FROZEN_BEFORE_EITHER_CLAUDE_LABEL_PASS":
        require(prelabel_path.is_file(), "prelabel independence audit missing")
    if prelabel_path.is_file():
        require(
            str(prelabel_ref.get("sha256") or "") == sha256_file(prelabel_path),
            "prelabel independence audit hash drift",
        )
        prelabel = _load_object(prelabel_path)
        require(
            prelabel.get("status") == "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING",
            "prelabel independence audit is not clean",
        )
        for key in (
            "candidate_proposals_exposed_to_either_pass",
            "prior_truth_or_predictions_exposed_to_either_pass",
            "pass_predictions_mutually_visible",
        ):
            require(prelabel.get(key) is False, f"prelabel independence violation: {key}")
    else:
        frontier_ref = freeze.get("prior_consensus_frontier") or {}
        consensus_ref = freeze.get("source_consensus_report") or {}
        frontier_path = Path(str(frontier_ref.get("path") or "")).resolve()
        consensus_path = Path(str(consensus_ref.get("path") or "")).resolve()
        require(frontier_path.is_file(), "resolver consensus frontier missing")
        require(consensus_path.is_file(), "resolver source consensus report missing")
        if frontier_path.is_file():
            require(
                frontier_ref.get("sha256") == sha256_file(frontier_path),
                "resolver frontier hash drift",
            )
        if consensus_path.is_file():
            require(
                consensus_ref.get("sha256") == sha256_file(consensus_path),
                "resolver consensus report hash drift",
            )

    # Bind the current pack to the before-labeling freeze.
    for name, path in (
        ("validation", validation_path),
        ("bank", bank_path),
        ("items", items_path),
    ):
        require(path.is_file(), f"missing pack {name}")
        if path.is_file():
            require(
                (frozen.get(name) or {}).get("sha256") == sha256_file(path),
                f"frozen {name} hash drift",
            )
    frozen_chunks = {
        Path(str(row.get("path") or "")).name: row for row in frozen.get("chunks") or []
    }
    require(len(chunks) == len(frozen_chunks), "frozen chunk count drift")
    for chunk in chunks:
        require(
            (frozen_chunks.get(chunk.name) or {}).get("sha256") == sha256_file(chunk),
            f"frozen chunk hash drift: {chunk.stem}",
        )

    validation = _load_object(validation_path)
    bank = _load_object(bank_path)
    bank_rows = list(bank.get("metrics") or bank.get("bank") or [])
    bank_ids = {str(row.get("metric_id") or "") for row in bank_rows}
    task = str(freeze.get("task") or "")
    require(validation.get("truth_hidden") is True, "pack is not truth-hidden")
    require(
        validation.get("status") == "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING"
        or (
            validation.get("schema_version")
            == "silver-match-v3-exact-unresolved-resolver-pack-v1"
            and validation.get("prior_decisions_and_metric_ids_hidden") is True
        ),
        "pack is not a frozen source or exact resolver pack",
    )
    require(validation.get("task") == task, "task mismatch")
    require("" not in bank_ids and len(bank_ids) == len(bank_rows), "invalid bank IDs")
    item_rows = list(read_jsonl(items_path))
    require(
        not any(FORBIDDEN_ITEM_FIELDS & set(row) for row in item_rows),
        "pack items expose forbidden truth/proposal/outcome fields",
    )

    # The staged guide/schema copies are content-bound to the implementation freeze.
    implementation_by_name = {
        Path(str(row.get("path") or "")).name: str(row.get("sha256") or "")
        for row in freeze.get("implementation") or []
    }
    implementation_paths_by_name = {
        Path(str(row.get("path") or "")).name: Path(
            str(row.get("path") or "")
        ).resolve()
        for row in freeze.get("implementation") or []
    }
    required_names = {
        "INDEPENDENT_LABELING_GUIDE.md",
        "ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md",
        "independent_labels_1_to_25.schema.json",
    }
    require(required_names <= set(implementation_by_name), "freeze lacks guide/schema hashes")

    summary = _load_object(summary_path) if summary_path.is_file() else {}
    require(summary.get("schema_version") == RUN_SCHEMA, "missing or invalid run summary")
    require(summary.get("status") == "COMPLETE", "Claude run is incomplete")
    require(int(summary.get("chunks", -1)) == len(chunks), "run summary chunk count drift")
    require(int(summary.get("rows", -1)) == int(frozen.get("count", -2)), "run summary row count drift")
    require(int(summary.get("failed", -1)) == 0, "run summary reports failures")
    require(summary.get("model") == freeze.get("model"), "run model differs from freeze")
    require(summary.get("effort") == freeze.get("effort"), "run effort differs from freeze")
    require(
        (summary.get("pack") or {}).get("validation_sha256") == sha256_file(validation_path),
        "run summary validation hash drift",
    )

    reports: list[dict[str, Any]] = []
    total_events = total_reads = total_structured = 0
    model_names: set[str] = set()
    for chunk_path in chunks:
        chunk = chunk_path.stem
        request_path = request_root / f"{chunk}.json"
        raw_path = raw_root / f"{chunk}.json"
        stderr_candidates = sorted(stderr_root.glob(f"{chunk}.attempt-*.log"))
        transcript_candidates = sorted(transcript_root.glob(f"{chunk}.attempt-*.jsonl"))
        prefix = f"{chunk}: "
        if not request_path.is_file() or not raw_path.is_file():
            violations.append(prefix + "missing request or raw label")
            continue
        if len(transcript_candidates) != 1 or len(stderr_candidates) != 1:
            violations.append(prefix + "expected exactly one transcript and stderr attempt")
            continue
        transcript_path, stderr_path = transcript_candidates[0], stderr_candidates[0]
        require(stderr_path.stat().st_size == 0, prefix + "non-empty stderr")

        request = _load_object(request_path)
        require(request.get("schema_version") == REQUEST_SCHEMA, prefix + "request schema drift")
        require(request.get("status") == "FROZEN_BEFORE_REQUEST", prefix + "request not frozen")
        require(request.get("task") == task and request.get("chunk_id") == chunk, prefix + "request identity drift")
        require(request.get("model") == freeze.get("model"), prefix + "request model drift")
        require(request.get("effort") == freeze.get("effort"), prefix + "request effort drift")
        contract = request.get("tool_contract") or {}
        require(
            contract
            == {
                "available": ["Read"],
                "network_tools_available": False,
                "write_or_shell_tools_available": False,
                "safe_mode": True,
                "session_persistence": False,
            },
            prefix + "request tool contract drift",
        )
        cwd = Path(str(request.get("cwd") or "")).resolve()
        require(cwd == pack.parent.resolve(), prefix + "request cwd drift")

        expected_uids = [str(row["norm_uid"]) for row in read_jsonl(chunk_path)]
        raw = _load_object(raw_path)
        try:
            validate_payload(
                raw,
                task=task,
                chunk_id=chunk,
                expected_uids=expected_uids,
                bank_ids=bank_ids,
            )
        except Exception as exc:
            violations.append(prefix + f"raw label invalid: {type(exc).__name__}: {exc}")

        input_rows = request.get("inputs") or []
        request_input_paths = [
            Path(str(row.get("path") or "")).resolve() for row in input_rows
        ]
        request_inputs: dict[Path, str] = {}
        for row in input_rows:
            path = Path(str(row.get("path") or "")).resolve()
            if path in request_inputs:
                violations.append(prefix + f"duplicate request input: {path}")
            request_inputs[path] = str(row.get("sha256") or "")
        guide_paths = {
            path
            for path in request_inputs
            if path.name
            in {"INDEPENDENT_LABELING_GUIDE.md", "ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md"}
        }
        guide_aliases_by_name = {
            path.name: {
                path,
                *(
                    [implementation_paths_by_name[path.name]]
                    if path.name in implementation_paths_by_name
                    else []
                ),
            }
            for path in guide_paths
        }
        allowed_guide_paths = set().union(*guide_aliases_by_name.values()) if guide_aliases_by_name else set()
        schema_paths = {
            path for path in request_inputs if path.name == "independent_labels_1_to_25.schema.json"
        }
        expected_inputs = guide_paths | schema_paths | {bank_path, chunk_path, validation_path}
        require(len(guide_paths) == 2 and len(schema_paths) == 1, prefix + "guide/schema input set drift")
        require(set(request_inputs) == expected_inputs, prefix + "request includes non-frozen inputs")
        expected_input_order = [
            next(
                path
                for path in request_input_paths
                if path.name == "INDEPENDENT_LABELING_GUIDE.md"
            ),
            next(
                path
                for path in request_input_paths
                if path.name == "ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md"
            ),
            bank_path,
            chunk_path,
            next(iter(schema_paths)),
            validation_path,
        ] if len(guide_paths) == 2 and len(schema_paths) == 1 else []
        require(request_input_paths == expected_input_order, prefix + "request input order drift")
        for path, recorded_hash in request_inputs.items():
            require(path.is_file(), prefix + f"request input missing: {path}")
            if path.is_file():
                require(recorded_hash == sha256_file(path), prefix + f"request input hash drift: {path.name}")
                if path.name in implementation_by_name:
                    require(
                        recorded_hash == implementation_by_name[path.name],
                        prefix + f"request input differs from execution freeze: {path.name}",
                    )
        if expected_input_order:
            guides_in_order = expected_input_order[:2]
            guide_clause = " ".join(
                f"Read {path.relative_to(cwd)} for audited exact-leaf boundary guidance."
                for path in guides_in_order[1:]
            )
            expected_prompt = (
                f"Act as independent hidden-ID full-bank annotator {request.get('pass_name')} for a "
                "high-precision silver norm-to-metric pool. Read "
                f"{guides_in_order[0].relative_to(cwd)}. {guide_clause} Read the frozen, "
                f"order-permuted {task} bank at {bank_path.relative_to(cwd)} and "
                f"label all {len(expected_uids)} items in {chunk_path.relative_to(cwd)} from scratch. "
                "Consider the entire bank for every item. Distinguish exact MATCH from "
                "related-family-only and typed bank-gap/no-criterion abstentions; never force "
                f"a leaf. Set task to {task} and chunk_id to {chunk}. Return only "
                "schema-conforming JSON. Do not search for any prior labels, proposals, "
                "audits, truth, model outputs, MI, or outcomes; use only the item text/context, "
                "guides, and this bank."
            )
            require(
                request.get("prompt_sha256")
                == hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest(),
                prefix + "request prompt hash is not exactly reconstructible from frozen inputs",
            )

        events: list[dict[str, Any]] = []
        try:
            for line_number, line in enumerate(
                transcript_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"event {line_number} is not an object")
                events.append(value)
        except Exception as exc:
            violations.append(prefix + f"invalid transcript JSONL: {type(exc).__name__}: {exc}")
            continue
        total_events += len(events)
        init = [row for row in events if row.get("type") == "system" and row.get("subtype") == "init"]
        finals = [row for row in events if row.get("type") == "result"]
        require(len(init) == 1, prefix + "expected one init event")
        if len(init) == 1:
            require(set(init[0].get("tools") or []) == {"Read", "StructuredOutput"}, prefix + "unexpected available tools")
            require(init[0].get("permissionMode") == "dontAsk", prefix + "permission mode drift")
            require(init[0].get("mcp_servers") == [], prefix + "MCP server present")
            require(Path(str(init[0].get("cwd") or "")).resolve() == cwd, prefix + "runtime cwd drift")
            model_names.add(str(init[0].get("model") or ""))
        require(len(finals) == 1, prefix + "expected one result event")

        read_paths: list[Path] = []
        read_path_by_tool_id: dict[str, Path] = {}
        read_lines_by_path: dict[Path, set[int]] = {}
        structured_inputs: list[dict[str, Any]] = []
        unexpected_tools: list[str] = []
        for event in events:
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = str(block.get("name") or "")
                tool_input = block.get("input") or {}
                if name == "Read":
                    path = _resolve(str(tool_input.get("file_path") or ""), cwd)
                    read_paths.append(path)
                    tool_id = str(block.get("id") or "")
                    if not tool_id or tool_id in read_path_by_tool_id:
                        violations.append(prefix + "Read call lacks a unique tool-use ID")
                    else:
                        read_path_by_tool_id[tool_id] = path
                    if path not in allowed_guide_paths | {bank_path, chunk_path}:
                        violations.append(prefix + f"Read accessed forbidden path: {path}")
                elif name == "StructuredOutput":
                    if isinstance(tool_input, dict):
                        structured_inputs.append(tool_input)
                    else:
                        violations.append(prefix + "StructuredOutput input is not an object")
                else:
                    unexpected_tools.append(name)
        observed_read_results: set[str] = set()
        for event in events:
            message = event.get("message")
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = str(block.get("tool_use_id") or "")
                if tool_id not in read_path_by_tool_id:
                    continue
                observed_read_results.add(tool_id)
                require(block.get("is_error") is not True, prefix + "Read tool result is an error")
                content = block.get("content")
                require(isinstance(content, str), prefix + "Read tool result is not text")
                if isinstance(content, str):
                    lines = {
                        int(match.group(1))
                        for line in content.splitlines()
                        if (match := re.match(r"^(\d+)\t", line))
                    }
                    read_lines_by_path.setdefault(read_path_by_tool_id[tool_id], set()).update(lines)
        require(
            observed_read_results == set(read_path_by_tool_id),
            prefix + "Read call/result coverage mismatch",
        )
        total_reads += len(read_paths)
        total_structured += len(structured_inputs)
        require(not unexpected_tools, prefix + f"unexpected tool use: {unexpected_tools}")
        require({bank_path, chunk_path} <= set(read_paths), prefix + "bank or assigned chunk was not read")
        for required_guide in guide_paths:
            aliases = guide_aliases_by_name[required_guide.name]
            require(bool(aliases & set(read_paths)), prefix + f"required guide was not read: {required_guide.name}")
            observed_lines = set().union(
                *(read_lines_by_path.get(alias, set()) for alias in aliases)
            )
            expected_line_count = len(required_guide.read_text(encoding="utf-8").splitlines())
            expected_lines = set(range(1, expected_line_count + 1))
            require(
                expected_lines <= observed_lines,
                prefix + f"Read did not expose every frozen line: {required_guide.name}",
            )
        for required_path in (bank_path, chunk_path):
            expected_line_count = len(required_path.read_text(encoding="utf-8").splitlines())
            expected_lines = set(range(1, expected_line_count + 1))
            require(
                expected_lines <= read_lines_by_path.get(required_path, set()),
                prefix + f"Read did not expose every frozen line: {required_path.name}",
            )
        require(len(structured_inputs) == 1, prefix + "expected one StructuredOutput use")

        final = finals[0] if len(finals) == 1 else {}
        require(final.get("subtype") == "success" and final.get("is_error") is False, prefix + "result not successful")
        require(final.get("permission_denials") == [], prefix + "permission denial present")
        structured = final.get("structured_output")
        require(isinstance(structured, dict) and structured == raw, prefix + "final structured output differs from raw label")
        if isinstance(final.get("result"), str):
            try:
                require(json.loads(final["result"]) == raw, prefix + "result string differs from raw label")
            except json.JSONDecodeError:
                violations.append(prefix + "result string is not JSON")
        else:
            violations.append(prefix + "result string absent")
        require(structured_inputs == [raw], prefix + "StructuredOutput call differs from raw label")
        usage = final.get("usage") or {}
        server_usage = usage.get("server_tool_use") or {}
        require(int(server_usage.get("web_search_requests", 0) or 0) == 0, prefix + "web search request observed")
        require(int(server_usage.get("web_fetch_requests", 0) or 0) == 0, prefix + "web fetch request observed")
        model_usage = final.get("modelUsage") or {}
        for model, row in model_usage.items():
            model_names.add(str(model))
            require(int((row or {}).get("webSearchRequests", 0) or 0) == 0, prefix + "model web search observed")

        reports.append(
            {
                "chunk": chunk,
                "chunk_sha256": sha256_file(chunk_path),
                "request_sha256": sha256_file(request_path),
                "transcript_path": str(transcript_path),
                "transcript_sha256": sha256_file(transcript_path),
                "stderr_path": str(stderr_path),
                "stderr_sha256": sha256_file(stderr_path),
                "raw_label_path": str(raw_path),
                "raw_label_sha256": sha256_file(raw_path),
                "event_count": len(events),
                "read_count": len(read_paths),
                "structured_output_count": len(structured_inputs),
                "row_count": len(expected_uids),
            }
        )

    result = {
        "schema_version": SCHEMA,
        "status": "PASS" if not violations else "FAIL",
        "complete": not violations and len(reports) == len(chunks),
        "truth_hidden": True,
        "task": task,
        "pass_key": args.pass_key,
        "model": freeze.get("model"),
        "runtime_models": sorted(model_names),
        "effort": freeze.get("effort"),
        "output_namespace": args.output_namespace,
        "expected_chunks": len(chunks),
        "audited_chunks": len(reports),
        "expected_rows": int(frozen.get("count", -1)),
        "audited_rows": sum(row["row_count"] for row in reports),
        "event_count": total_events,
        "read_count": total_reads,
        "structured_output_count": total_structured,
        "violations": violations,
        "auditor": _ref(Path(__file__).resolve()),
        "execution_freeze": _ref(freeze_path),
        "prelabel_independence_audit": _ref(prelabel_path)
        if prelabel_path.is_file()
        else None,
        "pack_validation": _ref(validation_path),
        "items": _ref(items_path),
        "bank": _ref(bank_path),
        "run_summary": _ref(summary_path) if summary_path.is_file() else None,
        "contract": {
            "only_read_and_structured_output_tools_observed": not any(
                "unexpected tool" in row or "forbidden path" in row for row in violations
            ),
            "every_chunk_read_only_its_guides_bank_and_assigned_items": not any(
                "required input was not read" in row or "forbidden path" in row for row in violations
            ),
            "sample_keys_predictions_proposals_mi_outcomes_and_gemma_absent": True,
            "network_and_mcp_use_absent": not any(
                "web " in row or "MCP" in row for row in violations
            ),
            "final_payload_exactly_bound_to_raw_labels": not any(
                "differs from raw label" in row for row in violations
            ),
        },
        "chunks": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if violations:
        raise ValueError(f"Claude transcript audit failed with {len(violations)} violation(s)")
    return {**result, "output": str(output), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--execution-freeze", required=True)
    parser.add_argument("--pass-key", required=True)
    parser.add_argument("--output-namespace", default="claude_sonnet_v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), sort_keys=True))


if __name__ == "__main__":
    main()
