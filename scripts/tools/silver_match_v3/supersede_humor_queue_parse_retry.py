#!/usr/bin/env python3
"""Freeze an append-only Humor queue retry for sealed parse failures.

Only the failed cell's output path and maximum generation budget may change.
Every scientific input, model, prompt, seed, order, and downstream cell stays
byte-identical. The failed output remains immutable evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .run_humor_fresh_select_gpu_queue import validate_queue


def _replace_arg(argv: list[str], name: str, old: str, new: str) -> None:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"retry cell must contain exactly one {name}")
    index = positions[0] + 1
    if argv[index] != old:
        raise ValueError(f"retry cell {name} differs from expected {old}")
    argv[index] = new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-queue", required=True)
    parser.add_argument("--failed-output", required=True)
    parser.add_argument("--retry-output", required=True)
    parser.add_argument("--old-max-tokens", type=int, required=True)
    parser.add_argument("--new-max-tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.new_max_tokens <= args.old_max_tokens:
        parser.error("retry token budget must increase")

    parent_path = Path(args.parent_queue).resolve()
    failed_path = Path(args.failed_output).resolve()
    retry_path = Path(args.retry_output).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists() or retry_path.exists() or retry_path.with_suffix(
        retry_path.suffix + ".meta.json"
    ).exists():
        raise FileExistsError("retry queue/output must be append-only")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    validate_queue(parent)

    failed_meta_path = failed_path.with_suffix(failed_path.suffix + ".meta.json")
    failed_meta = json.loads(failed_meta_path.read_text(encoding="utf-8"))
    failed_rows = list(read_jsonl(failed_path))
    invalid = [row for row in failed_rows if row.get("parse_error")]
    if (
        failed_meta.get("output_sha256") != sha256_file(failed_path)
        or int(failed_meta.get("eligible_count", -1)) != len(failed_rows)
        or int(failed_meta.get("invalid_count", -1)) != len(invalid)
        or not invalid
        or any(
            row.get("decision") != "INVALID_OUTPUT"
            or row.get("parse_error") not in {"no_json", "invalid_json"}
            for row in invalid
        )
        or int(failed_meta.get("max_tokens", -1)) != args.old_max_tokens
    ):
        raise ValueError("failed output is not a sealed JSON parse failure at the old cap")

    queue = copy.deepcopy(parent)
    matched_cells: list[tuple[str, dict[str, Any]]] = []
    for stage in queue["stages"]:
        for cell in stage["cells"]:
            argv = [str(value) for value in cell["argv"]]
            output_positions = [i for i, value in enumerate(argv) if value == "--output"]
            if len(output_positions) == 1 and Path(
                argv[output_positions[0] + 1]
            ).resolve() == failed_path:
                matched_cells.append((str(stage["stage"]), cell))
    if len(matched_cells) != 1:
        raise ValueError("failed output does not identify exactly one frozen queue cell")
    stage_name, cell = matched_cells[0]
    argv = [str(value) for value in cell["argv"]]
    _replace_arg(argv, "--output", str(failed_path), str(retry_path))
    _replace_arg(
        argv,
        "--max-tokens",
        str(args.old_max_tokens),
        str(args.new_max_tokens),
    )
    cell["argv"] = argv

    output_keys = [
        key
        for key, value in queue["outputs"].items()
        if Path(str(value)).resolve() == failed_path
    ]
    if len(output_keys) != 1:
        raise ValueError("failed output does not identify exactly one output key")
    queue["outputs"][output_keys[0]] = str(retry_path)
    queue.setdefault("inputs", {})["parse_retry_supersession"] = {
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "parent_queue": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
        },
        "failed_output": {
            "path": str(failed_path),
            "sha256": sha256_file(failed_path),
        },
        "failed_meta": {
            "path": str(failed_meta_path),
            "sha256": sha256_file(failed_meta_path),
        },
    }
    queue["runtime_supersession"] = {
        "schema_version": "silver-match-v3-humor-queue-parse-retry-v1",
        "status": "FROZEN_BEFORE_PARSE_RETRY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "parent_queue_sha256": sha256_file(parent_path),
        "stage": stage_name,
        "output_key": output_keys[0],
        "failed_output_sha256": sha256_file(failed_path),
        "failed_invalid_count": len(invalid),
        "failed_parse_error_counts": {
            key: sum(row.get("parse_error") == key for row in invalid)
            for key in sorted({str(row.get("parse_error")) for row in invalid})
        },
        "retry_output": str(retry_path),
        "only_runtime_mutations": {
            "output_path": [str(failed_path), str(retry_path)],
            "max_tokens": [args.old_max_tokens, args.new_max_tokens],
        },
        "model_prompt_seed_order_or_panel_changed": False,
        "truth_read": False,
        "blind_read": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(queue, handle, indent=2, sort_keys=True)
        handle.write("\n")
    validate_queue(queue)
    print(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_PARSE_RETRY",
                "queue": str(output_path),
                "queue_sha256": sha256_file(output_path),
                "parent_queue_sha256": sha256_file(parent_path),
                "stage": stage_name,
                "retry_output": str(retry_path),
                "invalid_count": len(invalid),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
