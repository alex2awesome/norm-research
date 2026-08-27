#!/usr/bin/env python3
"""Repair only the frozen PR R4 prompt's confidence output contract."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from .common import sha256_file


OLD = "confidence is a number from 0 to 1."
NEW = 'confidence is exactly one of "high", "medium", or "low".'


def repair(source: Path, output: Path, audit: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    audit = audit.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or audit.exists():
        raise FileExistsError("refusing to overwrite a prompt or repair audit")

    before = source.read_text(encoding="utf-8")
    if before.count(OLD) != 1 or NEW in before:
        raise ValueError("source does not have the one expected numeric-confidence clause")
    after = before.replace(OLD, NEW, 1)
    if after.replace(NEW, OLD, 1) != before:
        raise AssertionError("repair is not exactly reversible")
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    changed = [
        index + 1
        for index, (old_line, new_line) in enumerate(zip(before_lines, after_lines))
        if old_line != new_line
    ]
    if len(before_lines) != len(after_lines) or changed != [6]:
        raise AssertionError(f"unexpected diff surface: {changed}")

    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(after, encoding="utf-8")
    unified_diff = "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=str(source),
            tofile=str(output),
            n=1,
        )
    )
    record = {
        "schema_version": "silver-match-v3-pr-r4-prompt-interface-repair-v1",
        "status": "INTERFACE_ONLY_REPAIR_BEFORE_TRUTH_JOIN",
        "task": "press-releases",
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "allowed_replacement": {"old": OLD, "new": NEW, "occurrences": 1},
        "machine_checks": {
            "reverse_substitution_recovers_source_bytes": True,
            "line_count_unchanged": True,
            "only_changed_lines": changed,
            "decision_vocabulary_unchanged": True,
            "semantic_adjudication_instructions_unchanged": True,
            "reason_contract": "nonempty contrastive reason, at most 24 words; parser-compatible",
            "inference_max_tokens": 220,
        },
        "unified_diff": unified_diff,
        "truth_or_predictions_read": False,
    }
    audit.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**record, "audit_sha256": sha256_file(audit)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    print(json.dumps(repair(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
