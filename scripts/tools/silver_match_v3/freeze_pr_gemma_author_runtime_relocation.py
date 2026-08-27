#!/usr/bin/env python3
"""Freeze and validate a hash-identical host relocation for PR Gemma authorship.

The freeze is deliberately limited to the optimize-only author packet, the
identity-free aggregate taxonomy, the already-frozen base prompt, executable
code, and a complete resolved-file manifest of the Gemma snapshot.  It does
not accept or inspect truth, test, select, or MI artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "silver-match-v3-pr-gemma-author-runtime-source-binding-v1"
SOURCE_STATUS = "FROZEN_SOURCE_BINDING_BEFORE_RELOCATION"
SCHEMA = "silver-match-v3-pr-gemma-author-runtime-relocation-freeze-v1"
STATUS = "FROZEN_HASH_IDENTICAL_RELOCATION_BEFORE_INFERENCE"
INPUT_NAMES = (
    "training_report",
    "training_examples",
    "aggregate_taxonomy",
    "base_prompt",
    "author_script",
    "common_module",
)
FORBIDDEN_PATH_TOKENS = ("/test", "test_", "/select", "select_", "/mi_", "/mi-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for path in sorted(value for value in root.iterdir() if value.is_file()):
        resolved = path.resolve(strict=True)
        rows.append(
            {
                "name": path.name,
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    if not rows or not any(row["name"].endswith(".safetensors") for row in rows):
        raise ValueError("Gemma snapshot manifest is incomplete")
    return rows


def manifest_sha(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUT_NAMES:
        parser.add_argument(f"--source-{name.replace('_', '-')}")
        parser.add_argument(f"--target-{name.replace('_', '-')}")
    parser.add_argument("--source-model")
    parser.add_argument("--target-model")
    parser.add_argument("--source-output")
    parser.add_argument("--target-output")
    parser.add_argument("--source-binding")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--examples-per-class", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target_paths = {}
    for name in INPUT_NAMES:
        value = getattr(args, f"target_{name}")
        if value is not None:
            target_paths[name] = Path(value).resolve()
    source_paths = {}
    for name in INPUT_NAMES:
        value = getattr(args, f"source_{name}")
        if value is not None:
            source_paths[name] = Path(value).resolve()
    for name, path in {**source_paths, **target_paths}.items():
        lower = str(path).lower()
        if any(token in lower for token in FORBIDDEN_PATH_TOKENS):
            raise ValueError(f"forbidden test/select/MI path in relocation allowlist: {name}")

    settings = {
            "seed": args.seed,
            "examples_per_class": args.examples_per_class,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "direct_batch_vllm": True,
    }
    if args.source_binding is None:
        if (
            set(source_paths) != set(INPUT_NAMES)
            or args.source_model is None
            or args.source_output is None
            or any(value is None for value in settings.values())
        ):
            raise ValueError("source binding requires every source input, model, output, and setting")
        source_output = Path(args.source_output).resolve()
        if source_output.exists():
            raise FileExistsError("canonical source inference output already exists")
        source_model = Path(args.source_model).resolve()
        source_model_rows = model_manifest(source_model)
        payload = {
            "schema_version": SOURCE_SCHEMA,
            "status": SOURCE_STATUS,
            "task": "press-releases",
            "scientific_settings": settings,
            "source_inputs": {name: ref(path) for name, path in source_paths.items()},
            "gemma_snapshot": {
                "source_path": str(source_model),
                "manifest": source_model_rows,
                "manifest_sha256": manifest_sha(source_model_rows),
            },
            "canonical_output_path": str(source_output),
            "gate": {
                "frozen_before_relocation_and_inference": True,
                "allowlisted_files_opened_only": True,
                "truth_test_select_mi_inputs_opened": False,
                "fresh_test_drawn_or_read": False,
                "append_only_outputs": True,
            },
        }
    else:
        source_binding_path = Path(args.source_binding).resolve()
        source = json.loads(source_binding_path.read_text(encoding="utf-8"))
        if (
            source.get("schema_version") != SOURCE_SCHEMA
            or source.get("status") != SOURCE_STATUS
            or source.get("task") != "press-releases"
            or set(target_paths) != set(INPUT_NAMES)
            or args.target_model is None
            or args.target_output is None
        ):
            raise ValueError("invalid source binding or incomplete target relocation")
        source_refs = source.get("source_inputs") or {}
        target_refs = {name: ref(path) for name, path in target_paths.items()}
        for name in INPUT_NAMES:
            if (
                (source_refs.get(name) or {}).get("sha256") != target_refs[name]["sha256"]
                or int((source_refs.get(name) or {}).get("size", -1))
                != target_refs[name]["size"]
            ):
                raise ValueError(f"relocated input differs from source: {name}")
        target_model = Path(args.target_model).resolve()
        target_model_rows = model_manifest(target_model)
        source_model_rows = (source.get("gemma_snapshot") or {}).get("manifest") or []
        if (
            source_model_rows != target_model_rows
            or (source.get("gemma_snapshot") or {}).get("manifest_sha256")
            != manifest_sha(target_model_rows)
        ):
            raise ValueError("relocated Gemma snapshot differs from source")
        source_output = Path(str(source.get("canonical_output_path") or "")).resolve()
        target_output = Path(args.target_output).resolve()
        if source_output.exists() or target_output.exists():
            raise FileExistsError(
                "source and target inference outputs must not exist at prewrite gate"
            )
        payload = {
            "schema_version": SCHEMA,
            "status": STATUS,
            "task": "press-releases",
            "scientific_settings": source.get("scientific_settings"),
            "source_binding": {
                "path": str(source_binding_path),
                "sha256": sha256_file(source_binding_path),
            },
            "source_inputs": source_refs,
            "target_inputs": target_refs,
            "gemma_snapshot": {
                "source_path": (source.get("gemma_snapshot") or {}).get("source_path"),
                "target_path": str(target_model),
                "manifest": source_model_rows,
                "manifest_sha256": manifest_sha(source_model_rows),
            },
            "outputs": {
                "canonical_source_path": str(source_output),
                "relocated_target_path": str(target_output),
                "source_preexisting": False,
                "target_preexisting": False,
            },
            "gate": {
                "validated_before_inference": True,
                "only_absolute_host_paths_and_runtime_may_differ": True,
                "all_scientific_inputs_hash_identical": True,
                "complete_model_snapshot_hash_identical": True,
                "allowlisted_files_opened_only": True,
                "truth_test_select_mi_inputs_opened": False,
                "fresh_test_drawn_or_read": False,
                "append_only_outputs": True,
            },
        }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload}))


if __name__ == "__main__":
    main()
