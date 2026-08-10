#!/usr/bin/env python3
"""Seal a read-only exact-completeness audit for a Humor Gemma queue."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file
from .run_humor_fresh_select_gpu_queue import _cell_complete, validate_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--complete-marker", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    marker_path = Path(args.complete_marker).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    validate_queue(queue)
    queue_sha = sha256_file(queue_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("schema_version")
        != "silver-match-v3-humor-queue-run-complete-v1"
        or marker.get("queue_sha256") != queue_sha
        or marker.get("all_cells_exact_complete") is not True
    ):
        raise ValueError("complete marker does not bind an exact-complete queue")

    cells = []
    for stage in queue["stages"]:
        for index, cell in enumerate(stage["cells"]):
            if not _cell_complete(cell):
                raise ValueError(
                    f"queue cell is not exact-complete: {stage['stage']}[{index}]"
                )
            argv = [str(value) for value in cell["argv"]]
            output_index = argv.index("--output") + 1
            cell_output = Path(argv[output_index])
            cells.append(
                {
                    "stage": str(stage["stage"]),
                    "cell_index": index,
                    "module": str(cell["module"]),
                    "output": str(cell_output),
                    "output_sha256": sha256_file(cell_output),
                }
            )

    report = {
        "schema_version": "silver-match-v3-humor-queue-exact-complete-audit-v1",
        "status": "PASS",
        "task": "humor",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "queue": {"path": str(queue_path), "sha256": queue_sha},
        "complete_marker": {
            "path": str(marker_path),
            "sha256": sha256_file(marker_path),
        },
        "stage_count": len(queue["stages"]),
        "cell_count": len(cells),
        "all_cells_exact_complete": True,
        "cells": cells,
        "truth_read": False,
        "blind_read": False,
        "permanent_blind_consumed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage_count": len(queue["stages"]),
                "cell_count": len(cells),
                "queue_sha256": queue_sha,
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
