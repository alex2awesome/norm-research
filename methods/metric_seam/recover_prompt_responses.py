"""Re-derive prompt-response status from retained raw text, without new model calls.

The 2026-07-13 ``code_review_glm52_impl_summary_v1`` run recorded 4,500/4,500
``contract_error`` because the runner deserialized ``raw_response`` with a bare
``json.loads``: it did not unwrap Markdown fences, and its strict mode rejected
literal tabs inside evidence strings.  GLM-5.2 had in fact honored the response
schema on essentially every row.

Because the runner retains ``raw_response`` verbatim, the run is recoverable on
CPU.  This tool replays the *fixed* deserialize/validate path over an existing
responses file and writes a new one.  It issues no provider request, reads no
outcome label, and never edits the input in place.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from methods.metric_seam.hierarchy_prompt_batch import validate_prompt_response
from methods.metric_seam.run_hierarchy_prompt_jobs import deserialize_response

RECOVERABLE = ("contract_error",)


def recover_row(row: dict) -> dict:
    """Return ``row`` with status/parsed_response re-derived from ``raw_response``."""

    if row.get("status") not in RECOVERABLE:
        return dict(row)
    recovered = dict(row)
    try:
        parsed = deserialize_response(row.get("raw_response", ""))
        recovered["parsed_response"] = validate_prompt_response(parsed)
        recovered["status"] = "valid"
    except Exception:  # noqa: BLE001 - any failure leaves the row a contract error
        recovered["parsed_response"] = {}
        recovered["status"] = "contract_error"
    return recovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.output.resolve() == args.responses.resolve():
        raise SystemExit("refusing to overwrite the input responses artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    status_after: Counter[str] = Counter()
    rows = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for line in args.responses.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            before[row.get("status", "")] += 1
            recovered = recover_row(row)
            after[recovered["status"]] += 1
            if recovered["status"] == "valid":
                status_after[recovered["parsed_response"]["measurement_status"]] += 1
            handle.write(json.dumps(recovered, sort_keys=True) + "\n")

    print(f"rows                : {rows}")
    print(f"status before       : {dict(before)}")
    print(f"status after        : {dict(after)}")
    print(f"recovered valid     : {after['valid']} ({100 * after['valid'] / max(rows, 1):.2f}%)")
    print(f"true contract errors: {after['contract_error']}")
    print(f"measurement_status  : {dict(status_after)}")
    print(f"wrote               : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
