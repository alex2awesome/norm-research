#!/usr/bin/env python3
"""Freeze the backend-pure direct-batch queue for Humor fresh select-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _command(module: str, *argv: object, gpu: int | None = None) -> dict[str, Any]:
    return {
        "module": module,
        "argv": [str(value) for value in argv],
        "cuda_visible_devices": None if gpu is None else str(gpu),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--identity-freeze", required=True)
    parser.add_argument("--adjudicator-selection", required=True)
    parser.add_argument("--adjudicator-prompt", required=True)
    parser.add_argument("--adjudicator-addon", required=True)
    parser.add_argument("--r5-freeze", required=True)
    parser.add_argument("--verifier-prompt", action="append", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument(
        "--gpu-id",
        action="append",
        type=int,
        required=True,
        help=(
            "Physical GPU ID available to this queue. Pass one, two, or three "
            "distinct IDs. Cells that share one physical GPU are frozen as "
            "sequential backend-pure stages."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--queue-output", required=True)
    args = parser.parse_args()

    repo, python, pack, manifest, identity, selection, r5, model, output, queue = map(
        lambda value: Path(value).resolve(),
        (
            args.repo, args.python, args.pack_root, args.execution_manifest,
            args.identity_freeze, args.adjudicator_selection, args.r5_freeze,
            args.model_snapshot, args.output_root, args.queue_output,
        ),
    )
    if queue.exists():
        raise FileExistsError(queue)
    gpu_ids = list(args.gpu_id)
    if len(gpu_ids) not in (1, 2, 3) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-id must provide one, two, or three distinct GPU IDs")
    if any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("GPU IDs must be non-negative")
    for path in (repo, python, pack, manifest, identity, selection, r5, model):
        if not path.exists():
            raise FileNotFoundError(path)
    validation_path = pack / "validation.json"
    candidates_path = pack / "candidates.top50.jsonl"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("task") != "humor"
        or validation.get("gepa_role") != "select"
        or validation.get("truth_hidden") is not True
        or validation.get("count") != 300
        or validation.get("candidate_k") != 50
    ):
        raise ValueError("pack is not the frozen 300-row Humor select pack")
    if sha256_file(candidates_path) != validation["outputs"]["candidates"]["sha256"]:
        raise ValueError("candidate pack hash mismatch")
    candidate_rows = list(read_jsonl(candidates_path))
    candidate_uids = [str(row.get("norm_uid") or "") for row in candidate_rows]
    if (
        len(candidate_rows) != 300
        or "" in candidate_uids
        or len(set(candidate_uids)) != 300
        or any(len(row.get("candidates") or []) != 50 for row in candidate_rows)
    ):
        raise ValueError("candidate pack lacks exact unique 300xK50 coverage")
    identity_payload = json.loads(identity.read_text(encoding="utf-8"))
    if (
        identity_payload.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or identity_payload.get("role") != "select"
        or sha256_file(identity) != validation["inputs"]["identity_freeze"]["sha256"]
    ):
        raise ValueError("identity freeze mismatch")
    local_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        local_manifest.get("truth_or_label_fields_in_manifest") is not False
        or local_manifest.get("source_pack", {}).get("validation_sha256")
        != sha256_file(validation_path)
    ):
        raise ValueError("execution manifest is not bound to the truth-hidden pack")

    adjudicator = json.loads(selection.read_text(encoding="utf-8"))
    chosen = adjudicator.get("chosen") or {}
    adj_paths = [Path(args.adjudicator_prompt).resolve(), Path(args.adjudicator_addon).resolve()]
    expected_adj = set((chosen.get("prompt_component_sha256") or {}).values())
    if (
        adjudicator.get("task") != "humor"
        or adjudicator.get("adjudicator_test_consumed") is not False
        or chosen.get("name") != "r1"
        or {sha256_file(path) for path in adj_paths} != expected_adj
    ):
        raise ValueError("adjudicator prompt differs from frozen dev-only selection")

    r5_payload = json.loads(r5.read_text(encoding="utf-8"))
    verifier_paths = [Path(value).resolve() for value in args.verifier_prompt]
    expected_verifier = {
        str(row["sha256"]) for row in (r5_payload.get("prompt") or {}).get("components") or []
    }
    if (
        r5_payload.get("status") != "FROZEN_BEFORE_VERIFIER_GEPA_INFERENCE"
        or r5_payload.get("permanent_blind_consumed") is not False
        or {sha256_file(path) for path in verifier_paths} != expected_verifier
    ):
        raise ValueError("verifier prompt differs from frozen R5 candidate")

    model_identity = [
        model / "config.json", model / "generation_config.json",
        model / "model.safetensors.index.json", model / "tokenizer.json",
    ]
    index_payload = json.loads(model_identity[2].read_text(encoding="utf-8"))
    shard_names = sorted(set((index_payload.get("weight_map") or {}).values()))
    if not shard_names or any(Path(name).name != name for name in shard_names):
        raise ValueError("Gemma weight index has an invalid or empty shard inventory")
    weight_shards = [model / name for name in shard_names]
    required_model = [*model_identity, *weight_shards]
    if any(not path.is_file() for path in required_model):
        raise FileNotFoundError("Gemma snapshot is incomplete")
    outputs = {
        "adj_original": output / "adjudicator" / "original.jsonl",
        "adj_hashed": output / "adjudicator" / "hashed.jsonl",
        "consensus": output / "adjudicator" / "exact_consensus.proposals.jsonl",
        "ver_original": output / "verifier_r5" / "original.jsonl",
        "ver_hashed": output / "verifier_r5" / "hashed.jsonl",
        "ver_reverse": output / "verifier_r5" / "reverse.jsonl",
    }
    base_adj = [
        "--manifest", manifest, "--candidates", candidates_path,
        "--prompt", adj_paths[0], "--prompt-addon", adj_paths[1],
        "--model", model, "--max-candidates", 50, "--batch-size", 128,
        "--gpu-memory-utilization", 0.88, "--max-model-len", 8192,
        "--max-tokens", 160, "--seed", 17, "--context-chars", 1200,
        "--description-chars", 260, "--example-chars", 80,
        "--max-examples", 0, "--resume",
    ]
    base_ver = [
        "--manifest", manifest, "--candidates", candidates_path,
        "--primary", outputs["consensus"], "--prompt", verifier_paths[0],
    ]
    for path in verifier_paths[1:]:
        base_ver.extend(["--prompt-addon", path])
    base_ver.extend(
        [
            "--model", model, "--max-alternatives", 49, "--batch-size", 128,
            "--gpu-memory-utilization", 0.88, "--max-model-len", 8192,
            "--max-tokens", 180, "--seed", 29, "--context-chars", 1200,
            "--description-chars", 260, "--example-chars", 80,
            "--max-examples", 0, "--resume",
        ]
    )
    module_root = "scripts.tools.silver_match_v3"
    if len(gpu_ids) == 1:
        stages = [
            {
                "stage": "fresh_select_adjudicator_original",
                "parallel": False,
                "cells": [
                    _command(
                        f"{module_root}.adjudicate_gemma", *base_adj,
                        "--output", outputs["adj_original"], "--order-mode", "original",
                        gpu=gpu_ids[0],
                    )
                ],
            },
            {
                "stage": "fresh_select_adjudicator_hashed",
                "depends_on": ["fresh_select_adjudicator_original"],
                "parallel": False,
                "cells": [
                    _command(
                        f"{module_root}.adjudicate_gemma", *base_adj,
                        "--output", outputs["adj_hashed"], "--order-mode", "hashed",
                        gpu=gpu_ids[0],
                    )
                ],
            },
        ]
        adjudicator_dependency = "fresh_select_adjudicator_hashed"
    else:
        stages = [
            {
                "stage": "fresh_select_adjudicator_all_orders",
                "parallel": True,
                "cells": [
                    _command(f"{module_root}.adjudicate_gemma", *base_adj, "--output", outputs["adj_original"], "--order-mode", "original", gpu=gpu_ids[0]),
                    _command(f"{module_root}.adjudicate_gemma", *base_adj, "--output", outputs["adj_hashed"], "--order-mode", "hashed", gpu=gpu_ids[1]),
                ],
            }
        ]
        adjudicator_dependency = "fresh_select_adjudicator_all_orders"
    stages.append(
        {
            "stage": "exact_adjudicator_consensus",
            "depends_on": [adjudicator_dependency],
            "parallel": False,
            "cells": [
                _command(
                    f"{module_root}.build_two_order_consensus_proposals",
                    "--original", outputs["adj_original"], "--hashed", outputs["adj_hashed"],
                    "--task", "humor", "--output", outputs["consensus"],
                )
            ],
        }
    )
    if len(gpu_ids) == 1:
        dependency = "exact_adjudicator_consensus"
        for order, output_key in (
            ("original", "ver_original"),
            ("hashed", "ver_hashed"),
            ("reverse", "ver_reverse"),
        ):
            stage_name = f"fresh_select_verifier_r5_candidate_{order}"
            stages.append(
                {
                    "stage": stage_name,
                    "depends_on": [dependency],
                    "parallel": False,
                    "cells": [
                        _command(
                            f"{module_root}.verify_gemma", *base_ver,
                            "--output", outputs[output_key], "--order-mode", order,
                            gpu=gpu_ids[0],
                        )
                    ],
                }
            )
            dependency = stage_name
    else:
        stages.append(
            {
                "stage": "fresh_select_verifier_r5_candidate_original_hashed",
                "depends_on": ["exact_adjudicator_consensus"],
                "parallel": True,
                "cells": [
                    _command(f"{module_root}.verify_gemma", *base_ver, "--output", outputs["ver_original"], "--order-mode", "original", gpu=gpu_ids[0]),
                    _command(f"{module_root}.verify_gemma", *base_ver, "--output", outputs["ver_hashed"], "--order-mode", "hashed", gpu=gpu_ids[1]),
                ],
            }
        )
    if len(gpu_ids) == 3:
        stages[-1]["cells"].append(
            _command(
                f"{module_root}.verify_gemma", *base_ver,
                "--output", outputs["ver_reverse"], "--order-mode", "reverse",
                gpu=gpu_ids[2],
            )
        )
        stages[-1]["stage"] = "fresh_select_verifier_r5_candidate_all_orders"
    elif len(gpu_ids) == 2:
        stages.append(
            {
                "stage": "fresh_select_verifier_r5_candidate_reverse",
                "depends_on": ["fresh_select_verifier_r5_candidate_original_hashed"],
                "parallel": False,
                "cells": [
                    _command(
                        f"{module_root}.verify_gemma", *base_ver,
                        "--output", outputs["ver_reverse"], "--order-mode", "reverse",
                        gpu=gpu_ids[0],
                    )
                ],
            }
        )
    payload = {
        "schema_version": "silver-match-v3-humor-fresh-select-gpu-queue-v1",
        "status": "FROZEN_BEFORE_FRESH_SELECT_MODEL_PREDICTIONS",
        "task": "humor",
        "backend": "direct_vllm_batch",
        "backend_purity": {
            "openai_server_forbidden": True,
            "never_mix_backends_within_cell": True,
            "r5_optimize_openrouter_cells_not_retried_unless_failed_or_incomplete": True,
        },
        "gpu_policy": {
            "physical_gpu_ids": gpu_ids,
            "maximum_concurrent_gpus": len(gpu_ids),
            "global_gpu_count_gate_applied": False,
        },
        "python": _artifact(python),
        "repo": str(repo),
        "model_snapshot": {
            "path": str(model),
            "revision": model.name,
            "identity_files": {_path.name: _artifact(_path) for _path in model_identity},
            "weight_shard_bytes": {path.name: path.stat().st_size for path in weight_shards},
        },
        "inputs": {
            "pack_validation": _artifact(validation_path),
            "candidates": _artifact(candidates_path),
            "execution_manifest": _artifact(manifest),
            "identity_freeze": _artifact(identity),
            "adjudicator_selection": _artifact(selection),
            "r5_optimize_freeze": _artifact(r5),
            "adjudicator_prompts": [_artifact(path) for path in adj_paths],
            "verifier_prompts": [_artifact(path) for path in verifier_paths],
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "stages": stages,
        "fresh_select_truth_read": False,
        "permanent_blind_consumed": False,
    }
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "queue_sha256": sha256_file(queue)}, sort_keys=True))


if __name__ == "__main__":
    main()
