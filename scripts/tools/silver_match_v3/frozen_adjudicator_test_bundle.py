#!/usr/bin/env python3
"""Atomically reserve and seal a two-order frozen adjudicator test bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import read_jsonl, sha256_file


def combined_prompt(paths: list[Path]) -> tuple[str, str]:
    text = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in paths) + "\n"
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def reserve(args: argparse.Namespace) -> None:
    marker = Path(args.marker).resolve()
    completed = marker.with_name(marker.stem + ".completed.json")
    if marker.exists() or completed.exists():
        raise FileExistsError(f"frozen adjudicator test already consumed: {marker}")
    selection_path = Path(args.selection).resolve()
    candidate_path = Path(args.candidates).resolve()
    manifest_path = Path(args.manifest).resolve()
    outputs = {
        "original": Path(args.original_output).resolve(),
        "hashed": Path(args.hashed_output).resolve(),
    }
    if any(path.exists() or path.with_suffix(path.suffix + ".meta.json").exists() for path in outputs.values()):
        raise FileExistsError("a frozen adjudicator output already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "external_dev_only":
        raise ValueError("adjudicator selection was not external-dev-only")
    if selection.get("adjudicator_test_consumed") is not False:
        raise ValueError("selection record does not declare an unconsumed test")
    prompt_paths = [Path(args.prompt).resolve(), *[Path(path).resolve() for path in args.prompt_addon]]
    _, prompt_hash = combined_prompt(prompt_paths)
    chosen = selection.get("chosen") or {}
    if chosen.get("prompt_sha256") != prompt_hash:
        raise ValueError("supplied prompt differs from dev-selected adjudicator")
    depth = int(selection.get("candidate_depth", 0))
    if args.max_candidates != depth:
        raise ValueError("test depth differs from dev-selected depth")
    candidate_meta_path = candidate_path.with_suffix(candidate_path.suffix + ".meta.json")
    candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    if not candidate_meta.get("source_is_frozen_test"):
        raise ValueError("candidate slate is not derived from the frozen retriever test")
    if candidate_meta.get("output_k") != depth:
        raise ValueError("frozen candidate materialization has the wrong depth")
    if candidate_meta.get("output_sha256") != sha256_file(candidate_path):
        raise ValueError("frozen candidate metadata hash mismatch")
    count = sum(1 for _ in read_jsonl(candidate_path))
    payload = {
        "status": "STARTED_TWO_ORDER_FROZEN_ADJUDICATOR_TEST",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "task": selection.get("task"),
        "candidate_depth": depth,
        "candidate_count": count,
        "required_order_modes": ["original", "hashed"],
        "prompt_sha256": prompt_hash,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "selection": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
            "candidates": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "candidate_meta": {
                "path": str(candidate_meta_path),
                "sha256": sha256_file(candidate_meta_path),
            },
            "prompt_components": {str(path): sha256_file(path) for path in prompt_paths},
        },
        "outputs": {mode: str(path) for mode, path in outputs.items()},
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**payload, "marker": str(marker), "marker_sha256": sha256_file(marker)}, sort_keys=True))


def complete(args: argparse.Namespace) -> None:
    marker = Path(args.marker).resolve()
    completed = marker.with_name(marker.stem + ".completed.json")
    if not marker.exists():
        raise FileNotFoundError(marker)
    if completed.exists():
        raise FileExistsError(completed)
    started = json.loads(marker.read_text(encoding="utf-8"))
    checks = {}
    for mode in started["required_order_modes"]:
        output = Path(started["outputs"][mode])
        meta_path = output.with_suffix(output.suffix + ".meta.json")
        if not output.exists() or not meta_path.exists():
            raise FileNotFoundError(f"frozen {mode} output is incomplete")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("order_mode") != mode:
            raise ValueError(f"frozen output order mismatch: {mode}")
        if meta.get("input_candidates_sha256") != started["inputs"]["candidates"]["sha256"]:
            raise ValueError(f"frozen output candidate mismatch: {mode}")
        if meta.get("prompt_sha256") != started["prompt_sha256"]:
            raise ValueError(f"frozen output prompt mismatch: {mode}")
        if meta.get("max_candidates") != started["candidate_depth"]:
            raise ValueError(f"frozen output depth mismatch: {mode}")
        if meta.get("new_count") != started["candidate_count"]:
            raise ValueError(f"frozen output count mismatch: {mode}")
        checks[mode] = {
            "output": {"path": str(output), "sha256": sha256_file(output)},
            "meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
        }
    payload = {
        "status": "COMPLETED_TWO_ORDER_FROZEN_ADJUDICATOR_TEST",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "started_marker": str(marker),
        "started_marker_sha256": sha256_file(marker),
        "checks": checks,
    }
    with completed.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**payload, "completed_marker": str(completed), "completed_sha256": sha256_file(completed)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    reserve_parser = sub.add_parser("reserve")
    reserve_parser.add_argument("--marker", required=True)
    reserve_parser.add_argument("--selection", required=True)
    reserve_parser.add_argument("--manifest", required=True)
    reserve_parser.add_argument("--candidates", required=True)
    reserve_parser.add_argument("--prompt", required=True)
    reserve_parser.add_argument("--prompt-addon", action="append", default=[])
    reserve_parser.add_argument("--max-candidates", type=int, required=True)
    reserve_parser.add_argument("--original-output", required=True)
    reserve_parser.add_argument("--hashed-output", required=True)
    complete_parser = sub.add_parser("complete")
    complete_parser.add_argument("--marker", required=True)
    args = parser.parse_args()
    reserve(args) if args.action == "reserve" else complete(args)


if __name__ == "__main__":
    main()
