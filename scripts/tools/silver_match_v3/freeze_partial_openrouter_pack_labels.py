#!/usr/bin/env python3
"""Freeze valid completed chunks from an interrupted OpenRouter label pass.

This artifact is deliberately non-promotional.  It proves which completed
chunks still pass the strict request-reconstruction audit while recording the
exact missing frontier.  A later production gold pass must be complete and
independently audited; partial chunks cannot silently become a complete pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit_openrouter_labeler_transcripts import audit
from .common import read_jsonl, sha256_file


SCHEMA = "silver-match-v3-partial-openrouter-label-freeze-v1"


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _named_roots(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in output:
            raise ValueError("--pass-root must be unique NAME=PATH values")
        output[name] = Path(raw_path).resolve()
    if not output:
        raise ValueError("at least one --pass-root is required")
    return output


def _freeze_pass(
    *,
    name: str,
    root: Path,
    guides: list[Path],
    schema_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("truth_hidden") is not True:
        raise ValueError(f"pass is not truth-hidden: {name}")
    chunks = sorted((root / "chunks").glob("part-*.jsonl"))
    expected = {path.stem for path in chunks}
    recorded = {
        Path(path).stem
        for path in ((validation.get("outputs") or {}).get("chunks") or {})
    }
    if not chunks or expected != recorded:
        raise ValueError(f"chunk inventory differs from frozen validation: {name}")
    raw = {path.stem: path for path in (root / "raw_labels").glob("part-*.json")}
    transcripts = {
        path.stem: path for path in (root / "api_transcripts").glob("part-*.json")
    }
    if set(raw) != set(transcripts) or not set(raw).issubset(expected):
        raise ValueError(f"raw/transcript completed-prefix mismatch: {name}")

    strict = audit(
        pack_root=root,
        guides=guides,
        schema_path=schema_path,
        runner_path=runner_path,
    )
    completed = set(raw)
    audited = {str(row.get("chunk") or "") for row in strict.get("chunks") or []}
    if audited != completed:
        raise ValueError(f"strict audit did not accept every completed chunk: {name}")
    missing = expected - completed
    observed_missing: set[str] = set()
    unexpected: list[dict[str, Any]] = []
    for violation in strict.get("violations") or []:
        chunk = str(violation.get("chunk") or "")
        kind = str(violation.get("kind") or "")
        if chunk in missing and kind == "FileNotFoundError":
            observed_missing.add(chunk)
        elif (
            not completed
            and chunk == "*"
            and kind == "MIXED_OR_MISSING_BACKEND_IDENTITY"
        ):
            continue
        else:
            unexpected.append(violation)
    if observed_missing != missing or unexpected:
        raise ValueError(
            f"partial pass has violations beyond its exact missing frontier: {name}: "
            f"missing={sorted(missing - observed_missing)} unexpected={unexpected[:2]}"
        )
    if completed and (
        strict.get("model") != "google/gemma-4-31b-it"
        or not strict.get("pass_name")
    ):
        raise ValueError(f"completed chunks lack frozen Gemma identity: {name}")

    rows = sum(len(list(read_jsonl(root / "chunks" / f"{chunk}.jsonl"))) for chunk in completed)
    return {
        "name": name,
        "root": str(root),
        "task": validation.get("task"),
        "model": strict.get("model"),
        "pass_name": strict.get("pass_name"),
        "pack_validation": _ref(validation_path),
        "bank": _ref(root / "bank.json"),
        "items": _ref(root / "items.jsonl"),
        "expected_chunk_count": len(expected),
        "completed_chunk_count": len(completed),
        "completed_row_count": rows,
        "completed_chunks": [
            {
                "chunk": chunk,
                "chunk_input": _ref(root / "chunks" / f"{chunk}.jsonl"),
                "raw_label": _ref(raw[chunk]),
                "api_transcript": _ref(transcripts[chunk]),
            }
            for chunk in sorted(completed)
        ],
        "missing_chunks": sorted(missing),
        "strict_partial_audit": strict,
        "promoted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-root", action="append", required=True)
    parser.add_argument("--guide", action="append", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--reason", default="OPENROUTER_SPEND_STOP")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    guides = [Path(value).resolve() for value in args.guide]
    schema_path = Path(args.schema).resolve()
    runner_path = Path(args.runner).resolve()
    passes = {
        name: _freeze_pass(
            name=name,
            root=root,
            guides=guides,
            schema_path=schema_path,
            runner_path=runner_path,
        )
        for name, root in _named_roots(args.pass_root).items()
    }
    tasks = {row["task"] for row in passes.values()}
    payload = {
        "schema_version": SCHEMA,
        "status": "FROZEN_PARTIAL_TRANSCRIPT_AUDITED_INTERRUPTED_NO_PROMOTION",
        "reason": args.reason,
        "task": next(iter(tasks)) if len(tasks) == 1 else None,
        "processes_terminated": True,
        "passes": passes,
        "total_expected_rows": sum(
            int(row["expected_chunk_count"])
            * int(json.loads(Path(row["pack_validation"]["path"]).read_text())["chunk_size"])
            for row in passes.values()
        ),
        "total_completed_rows": sum(
            int(row["completed_row_count"]) for row in passes.values()
        ),
        "contracts": {
            "every_preserved_chunk_passes_exact_request_reconstruction": True,
            "missing_frontier_is_explicit": True,
            "partial_labels_are_not_a_complete_gold_pass": True,
            "partial_labels_are_not_training_or_prompt_gradient_eligible": True,
            "next_gold_pass_requires_complete_independently_audited_coverage": True,
            "no_further_openrouter_spend_authorized_for_this_continuation": True,
        },
    }
    if len(tasks) != 1:
        raise ValueError("all partial passes must belong to one task")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "task": payload["task"],
                "completed_rows": payload["total_completed_rows"],
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
