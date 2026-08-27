#!/usr/bin/env python3
"""Audit a label-plan repartition that changes chunking but no scientific inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def source(plan: dict[str, Any], key: str) -> dict[str, Any]:
    return plan["inputs"][f"source_pass_{key.lower()}"]


def chunk_paths(value: dict[str, Any]) -> list[Path]:
    return [Path(row["path"]) for row in value["chunks"]]


def concatenated_chunks(value: dict[str, Any]) -> bytes:
    return b"".join(path.read_bytes() for path in chunk_paths(value))


def row_counts(value: dict[str, Any]) -> list[int]:
    return [len(path.read_bytes().splitlines()) for path in chunk_paths(value)]


def implementation_hashes(plan: dict[str, Any]) -> dict[str, Any]:
    impl = plan["implementation"]
    return {
        "runner": impl["runner"]["sha256"],
        "labeling_guide": impl["labeling_guide"]["sha256"],
        "isolation_guide": impl["isolation_guide"]["sha256"],
        "output_schema": impl["output_schema"]["sha256"],
        "boundary_guides": [row["sha256"] for row in impl["boundary_guides"]],
    }


def normalized_command(plan: dict[str, Any], key: str) -> dict[str, Any]:
    command = plan["commands"][key]
    argv = list(command["argv"])
    pack_index = argv.index("--pack-root") + 1
    argv[pack_index] = "<PASS_WORKSPACE>/pack"
    return {
        "argv": argv,
        "pythonpath": command["environment"]["PYTHONPATH"],
    }


def forbidden_runtime_artifacts(root: Path) -> list[str]:
    candidates = [
        root / "runtime/LAUNCH_RECEIPT.json",
        root / "runtime/COMPLETION_RECEIPT.json",
    ]
    for pass_name in ("pass_a", "pass_b"):
        pack = root / f"workspaces/{pass_name}/pack"
        candidates.extend(
            [
                pack / "raw_labels",
                pack / "logs",
                pack / "labels.validated.jsonl",
                pack / "predictions.jsonl",
            ]
        )
    return [str(path.resolve()) for path in candidates if path.exists()]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)

    old_plan_path = Path(args.old_plan).resolve()
    new_plan_path = Path(args.new_plan).resolve()
    old_plan = load_json(old_plan_path)
    new_plan = load_json(new_plan_path)
    old_root = Path(args.old_root).resolve()
    new_root = Path(args.new_root).resolve()

    for plan in (old_plan, new_plan):
        if plan.get("schema_version") != "silver-match-v3-independent-codex-label-execution-plan-v1":
            raise ValueError("unexpected execution-plan schema")
        if plan.get("status") != "FROZEN_BEFORE_EITHER_INDEPENDENT_LABEL_PASS":
            raise ValueError("execution plan was not frozen before labeling")

    equalities = {
        "task": old_plan["task"] == new_plan["task"],
        "row_count": old_plan["row_count"] == new_plan["row_count"],
        "pass_count": old_plan["pass_count"] == new_plan["pass_count"],
        "runtime": old_plan["runtime"] == new_plan["runtime"],
        "contracts": old_plan["contracts"] == new_plan["contracts"],
        "implementation_hashes": implementation_hashes(old_plan)
        == implementation_hashes(new_plan),
        "external_policy_hash": old_plan["inputs"]["external_policy"]["sha256"]
        == new_plan["inputs"]["external_policy"]["sha256"],
        "command_a_except_workspace": normalized_command(old_plan, "A")
        == normalized_command(new_plan, "A"),
        "command_b_except_workspace": normalized_command(old_plan, "B")
        == normalized_command(new_plan, "B"),
    }

    passes: dict[str, Any] = {}
    for key in ("A", "B"):
        old_source = source(old_plan, key)
        new_source = source(new_plan, key)
        old_items = Path(old_source["items"]["path"]).read_bytes()
        new_items = Path(new_source["items"]["path"]).read_bytes()
        old_bank = Path(old_source["bank"]["path"]).read_bytes()
        new_bank = Path(new_source["bank"]["path"]).read_bytes()
        old_concat = concatenated_chunks(old_source)
        new_concat = concatenated_chunks(new_source)
        old_counts = row_counts(old_source)
        new_counts = row_counts(new_source)
        checks = {
            "seed_equal": old_source["seed"] == new_source["seed"],
            "source_pack_equal": old_source["source_pack"] == new_source["source_pack"],
            "items_byte_identical": old_items == new_items,
            "bank_byte_identical": old_bank == new_bank,
            "old_chunks_reproduce_items": old_concat == old_items,
            "new_chunks_reproduce_items": new_concat == new_items,
            "concatenated_chunks_byte_identical": old_concat == new_concat,
            "old_chunk_rows_exactly_25": bool(old_counts) and set(old_counts) == {25},
            "new_chunk_rows_exactly_5": bool(new_counts) and set(new_counts) == {5},
        }
        if not all(checks.values()):
            failed = [name for name, value in checks.items() if not value]
            raise ValueError(f"pass {key} repartition invariants failed: {failed}")
        passes[key] = {
            "checks": checks,
            "seed": old_source["seed"],
            "row_count": len(old_items.splitlines()),
            "items_sha256": hashlib.sha256(old_items).hexdigest(),
            "bank_sha256": hashlib.sha256(old_bank).hexdigest(),
            "old_chunk_count": len(old_counts),
            "new_chunk_count": len(new_counts),
            "old_chunk_row_counts": sorted(set(old_counts)),
            "new_chunk_row_counts": sorted(set(new_counts)),
        }

    if not all(equalities.values()):
        failed = [name for name, value in equalities.items() if not value]
        raise ValueError(f"plan invariants failed: {failed}")

    old_forbidden = forbidden_runtime_artifacts(old_root)
    new_forbidden = forbidden_runtime_artifacts(new_root)
    if old_forbidden or new_forbidden:
        raise ValueError(
            "label/runtime artifacts exist before supersession audit: "
            + json.dumps({"old": old_forbidden, "new": new_forbidden})
        )

    evidence = [file_ref(Path(path)) for path in args.external_evidence]
    payload = {
        "schema_version": "silver-match-v3-independent-label-repartition-supersession-audit-v1",
        "status": "FROZEN_REPARTITION_ONLY; EXECUTION_BLOCKED_ON_SK3_AUTH",
        "task": old_plan["task"],
        "old_plan": file_ref(old_plan_path),
        "new_plan": file_ref(new_plan_path),
        "old_plan_was_unlaunched_and_unlabeled": True,
        "new_plan_was_unlaunched_and_unlabeled": True,
        "scientific_contract_changed": False,
        "only_intentional_changes": [
            "chunk_size_25_to_5",
            "chunk_inventory_16_to_80_per_pass",
            "source_validation_and_workspace_paths",
        ],
        "plan_equalities": equalities,
        "passes": passes,
        "external_runtime_evidence": evidence,
        "execution_disposition": {
            "old_plan_eligible": False,
            "new_plan_eligible_on_sk3": False,
            "next_action": "relocate frozen plan and byte-identical packs to an authenticated lane",
            "labels_created_by_either_code_plan": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**payload, "audit_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-plan", required=True)
    parser.add_argument("--new-plan", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--external-evidence", action="append", default=[])
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(parser.parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
