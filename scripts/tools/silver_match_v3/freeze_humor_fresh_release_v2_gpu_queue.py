#!/usr/bin/env python3
"""Freeze the truth-hidden Humor fresh-release-v2 Gemma evaluation queue.

The queue evaluates both predeclared adjudicators in two candidate orders on
the adjudicator panel.  It also evaluates every predeclared verifier in three
orders for proposals from *each* adjudicator on the disjoint verifier panel.
Freezing the complete cross-product before either truth release is opened
prevents a later adjudicator choice from leaking into verifier selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


SCHEMA = "silver-match-v3-humor-fresh-select-gpu-queue-v1"
STATUS = "FROZEN_BEFORE_FRESH_SELECT_MODEL_PREDICTIONS"


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _command(module: str, *argv: object, gpu: int | None = None) -> dict[str, Any]:
    return {
        "module": module,
        "argv": [str(value) for value in argv],
        "cuda_visible_devices": None if gpu is None else str(gpu),
    }


def _load_pack(
    *,
    root: Path,
    manifest: Path,
    identity_freeze: Path,
    expected_role: str,
) -> dict[str, Any]:
    validation_path = root / "validation.json"
    candidates_path = root / "candidates.top50.jsonl"
    items_path = root / "items.jsonl"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("task") != "humor"
        or validation.get("gepa_role") != expected_role
        or validation.get("truth_hidden") is not True
        or int(validation.get("count", -1)) != 300
        or int(validation.get("candidate_k", -1)) != 50
    ):
        raise ValueError(f"pack is not the frozen Humor {expected_role} panel")
    expected_outputs = validation.get("outputs") or {}
    for name, path in (("candidates", candidates_path), ("items", items_path)):
        if sha256_file(path) != (expected_outputs.get(name) or {}).get("sha256"):
            raise ValueError(f"{expected_role} {name} hash mismatch")

    candidate_rows = list(read_jsonl(candidates_path))
    item_rows = list(read_jsonl(items_path))
    candidate_uids = [str(row.get("norm_uid") or "") for row in candidate_rows]
    item_uids = [str(row.get("norm_uid") or "") for row in item_rows]
    if (
        len(candidate_rows) != 300
        or len(item_rows) != 300
        or "" in candidate_uids
        or len(set(candidate_uids)) != 300
        or set(candidate_uids) != set(item_uids)
        or any(len(row.get("candidates") or []) != 50 for row in candidate_rows)
        or any(row.get("truth_hidden") is not True for row in item_rows)
    ):
        raise ValueError(f"{expected_role} pack lacks exact unique 300xK50 coverage")

    freeze = json.loads(identity_freeze.read_text(encoding="utf-8"))
    expected_freeze_sha = (
        (validation.get("inputs") or {}).get("identity_freeze") or {}
    ).get("sha256")
    if (
        freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or freeze.get("task") != "humor"
        or freeze.get("role") != expected_role
        or int(freeze.get("selected_count", -1)) != 300
        or sha256_file(identity_freeze) != expected_freeze_sha
    ):
        raise ValueError(f"{expected_role} identity freeze mismatch")

    execution = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        execution.get("truth_or_label_fields_in_manifest") is not False
        or (execution.get("source_pack") or {}).get("truth_hidden") is not True
        or (execution.get("source_pack") or {}).get("validation_sha256")
        != sha256_file(validation_path)
    ):
        raise ValueError(f"{expected_role} inference manifest is not truth-hidden")
    return {
        "validation": validation_path,
        "candidates": candidates_path,
        "items": items_path,
        "manifest": manifest,
        "identity_freeze": identity_freeze,
        "uids": set(item_uids),
        "source_groups": {str(row.get("source_group") or "") for row in item_rows},
    }


def _prompt_variants(
    *, repo: Path, policy: dict[str, Any], key: str
) -> list[dict[str, Any]]:
    variants = policy.get(key) or []
    if not variants:
        raise ValueError(f"prelabel policy has no {key}")
    seen: set[str] = set()
    result = []
    for variant in variants:
        name = str(variant.get("name") or "")
        if not name or name in seen:
            raise ValueError(f"missing or duplicate variant in {key}: {name!r}")
        seen.add(name)
        components = []
        for frozen in variant.get("prompt_components") or []:
            relative = Path(str(frozen["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"prompt path is not repo-relative: {relative}")
            path = (repo / relative).resolve()
            if sha256_file(path) != str(frozen["sha256"]):
                raise ValueError(f"prompt hash mismatch: {path}")
            components.append(path)
        if not components:
            raise ValueError(f"variant lacks prompt components: {name}")
        result.append({"name": name, "components": components})
    return result


def _adjudicator_argv(
    *, manifest: Path, candidates: Path, prompts: list[Path], model: Path
) -> list[object]:
    argv: list[object] = [
        "--manifest", manifest,
        "--candidates", candidates,
        "--prompt", prompts[0],
    ]
    for path in prompts[1:]:
        argv.extend(("--prompt-addon", path))
    argv.extend(
        (
            "--model", model,
            "--max-candidates", 50,
            "--batch-size", 128,
            "--gpu-memory-utilization", 0.88,
            "--max-model-len", 8192,
            "--max-tokens", 160,
            "--seed", 17,
            "--context-chars", 1200,
            "--description-chars", 260,
            "--example-chars", 80,
            "--max-examples", 0,
            "--resume",
        )
    )
    return argv


def _verifier_argv(
    *,
    manifest: Path,
    candidates: Path,
    primary: Path,
    prompts: list[Path],
    model: Path,
) -> list[object]:
    argv: list[object] = [
        "--manifest", manifest,
        "--candidates", candidates,
        "--primary", primary,
        "--prompt", prompts[0],
    ]
    for path in prompts[1:]:
        argv.extend(("--prompt-addon", path))
    argv.extend(
        (
            "--model", model,
            "--max-alternatives", 49,
            "--batch-size", 128,
            "--gpu-memory-utilization", 0.88,
            "--max-model-len", 8192,
            "--max-tokens", 180,
            "--seed", 29,
            "--context-chars", 1200,
            "--description-chars", 260,
            "--example-chars", 80,
            "--max-examples", 0,
            "--resume",
        )
    )
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--adjudicator-pack-root", required=True)
    parser.add_argument("--verifier-pack-root", required=True)
    parser.add_argument("--adjudicator-manifest", required=True)
    parser.add_argument("--verifier-manifest", required=True)
    parser.add_argument("--adjudicator-identity-freeze", required=True)
    parser.add_argument("--verifier-identity-freeze", required=True)
    parser.add_argument("--prelabel-policy", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--queue-output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    python = Path(args.python).resolve()
    adjudicator_pack = Path(args.adjudicator_pack_root).resolve()
    verifier_pack = Path(args.verifier_pack_root).resolve()
    adjudicator_manifest = Path(args.adjudicator_manifest).resolve()
    verifier_manifest = Path(args.verifier_manifest).resolve()
    adjudicator_freeze = Path(args.adjudicator_identity_freeze).resolve()
    verifier_freeze = Path(args.verifier_identity_freeze).resolve()
    policy_path = Path(args.prelabel_policy).resolve()
    model = Path(args.model_snapshot).resolve()
    output = Path(args.output_root).resolve()
    queue = Path(args.queue_output).resolve()
    gpu = int(args.gpu_id)
    if queue.exists():
        raise FileExistsError(queue)
    if gpu < 0:
        raise ValueError("GPU ID must be non-negative")
    for path in (repo, python, adjudicator_pack, verifier_pack, policy_path, model):
        if not path.exists():
            raise FileNotFoundError(path)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        policy.get("schema_version")
        != "silver-match-v3-humor-fresh-release-v2-prelabel-policy-v1"
        or policy.get("status")
        != "FROZEN_BEFORE_INDEPENDENT_LABELS_OR_MODEL_PREDICTIONS"
        or policy.get("task") != "humor"
        or policy.get("blind_status") != "SEALED_UNCONSUMED"
        or (policy.get("independent_truth_labels") or {}).get(
            "truth_may_be_opened_before_both_prediction_families_are_frozen"
        )
        is not False
    ):
        raise ValueError("unsupported or contaminated Humor prelabel policy")

    adjudicator = _load_pack(
        root=adjudicator_pack,
        manifest=adjudicator_manifest,
        identity_freeze=adjudicator_freeze,
        expected_role="adjudicator_dev",
    )
    verifier = _load_pack(
        root=verifier_pack,
        manifest=verifier_manifest,
        identity_freeze=verifier_freeze,
        expected_role="verifier_dev",
    )
    if (
        adjudicator["uids"] & verifier["uids"]
        or adjudicator["source_groups"] & verifier["source_groups"]
        or "" in adjudicator["source_groups"]
        or "" in verifier["source_groups"]
    ):
        raise ValueError("adjudicator and verifier panels are not source-disjoint")

    adjudicator_variants = _prompt_variants(
        repo=repo, policy=policy, key="adjudicator_variants"
    )
    verifier_variants = _prompt_variants(
        repo=repo, policy=policy, key="verifier_variants"
    )

    model_identity = [
        model / "config.json",
        model / "generation_config.json",
        model / "model.safetensors.index.json",
        model / "tokenizer.json",
    ]
    index_payload = json.loads(model_identity[2].read_text(encoding="utf-8"))
    shard_names = sorted(set((index_payload.get("weight_map") or {}).values()))
    if not shard_names or any(Path(name).name != name for name in shard_names):
        raise ValueError("Gemma weight index has an invalid or empty shard inventory")
    weight_shards = [model / name for name in shard_names]
    required_model = [*model_identity, *weight_shards]
    if any(not path.is_file() for path in required_model):
        raise FileNotFoundError("Gemma snapshot is incomplete")

    module_root = "scripts.tools.silver_match_v3"
    stages: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}

    def add_adjudicator_family(
        *, panel_name: str, pack: dict[str, Any], variant: dict[str, Any]
    ) -> Path:
        dependency: str | None = None
        order_outputs: dict[str, Path] = {}
        base = _adjudicator_argv(
            manifest=pack["manifest"],
            candidates=pack["candidates"],
            prompts=variant["components"],
            model=model,
        )
        for order in ("original", "hashed"):
            stage_name = f"{panel_name}.{variant['name']}.adjudicator.{order}"
            order_output = output / panel_name / "adjudicator" / variant["name"] / f"{order}.jsonl"
            order_outputs[order] = order_output
            stages.append(
                {
                    "stage": stage_name,
                    **({"depends_on": [dependency]} if dependency else {}),
                    "parallel": False,
                    "cells": [
                        _command(
                            f"{module_root}.adjudicate_gemma",
                            *base,
                            "--output", order_output,
                            "--order-mode", order,
                            gpu=gpu,
                        )
                    ],
                }
            )
            dependency = stage_name
            outputs[f"{panel_name}.adjudicator.{variant['name']}.{order}"] = str(order_output)
        consensus = output / panel_name / "adjudicator" / variant["name"] / "exact_consensus.proposals.jsonl"
        consensus_stage = f"{panel_name}.{variant['name']}.adjudicator.exact_consensus"
        stages.append(
            {
                "stage": consensus_stage,
                "depends_on": [dependency],
                "parallel": False,
                "cells": [
                    _command(
                        f"{module_root}.build_two_order_consensus_proposals",
                        "--original", order_outputs["original"],
                        "--hashed", order_outputs["hashed"],
                        "--task", "humor",
                        "--output", consensus,
                    )
                ],
            }
        )
        outputs[f"{panel_name}.adjudicator.{variant['name']}.consensus"] = str(consensus)
        return consensus

    for variant in adjudicator_variants:
        add_adjudicator_family(
            panel_name="adjudicator_dev", pack=adjudicator, variant=variant
        )

    verifier_proposals: dict[str, Path] = {}
    for variant in adjudicator_variants:
        verifier_proposals[variant["name"]] = add_adjudicator_family(
            panel_name="verifier_dev", pack=verifier, variant=variant
        )

    dependency = stages[-1]["stage"]
    for adjudicator_variant in adjudicator_variants:
        primary = verifier_proposals[adjudicator_variant["name"]]
        for verifier_variant in verifier_variants:
            base = _verifier_argv(
                manifest=verifier["manifest"],
                candidates=verifier["candidates"],
                primary=primary,
                prompts=verifier_variant["components"],
                model=model,
            )
            for order in ("original", "hashed", "reverse"):
                stage_name = (
                    f"verifier_dev.{adjudicator_variant['name']}."
                    f"{verifier_variant['name']}.verifier.{order}"
                )
                order_output = (
                    output
                    / "verifier_dev"
                    / "verifier"
                    / adjudicator_variant["name"]
                    / verifier_variant["name"]
                    / f"{order}.jsonl"
                )
                stages.append(
                    {
                        "stage": stage_name,
                        "depends_on": [dependency],
                        "parallel": False,
                        "cells": [
                            _command(
                                f"{module_root}.verify_gemma",
                                *base,
                                "--output", order_output,
                                "--order-mode", order,
                                gpu=gpu,
                            )
                        ],
                    }
                )
                dependency = stage_name
                outputs[
                    f"verifier_dev.verifier.{adjudicator_variant['name']}."
                    f"{verifier_variant['name']}.{order}"
                ] = str(order_output)

    implementation_paths = [
        repo / "scripts/tools/silver_match_v3/adjudicate_gemma.py",
        repo / "scripts/tools/silver_match_v3/verify_gemma.py",
        repo / "scripts/tools/silver_match_v3/build_two_order_consensus_proposals.py",
        repo / "scripts/tools/silver_match_v3/run_humor_fresh_select_gpu_queue.py",
        Path(__file__).resolve(),
    ]
    payload = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "task": "humor",
        "release": "fresh_release_v2",
        "backend": "direct_vllm_batch",
        "backend_purity": {
            "openai_server_forbidden": True,
            "never_mix_backends_within_cell": True,
            "all_predeclared_model_outputs_frozen_before_truth_join": True,
        },
        "scientific_contract": {
            "adjudicator_variants": [row["name"] for row in adjudicator_variants],
            "verifier_variants": [row["name"] for row in verifier_variants],
            "adjudicator_orders": ["original", "hashed"],
            "verifier_orders": ["original", "hashed", "reverse"],
            "verifier_crosses_every_predeclared_adjudicator": True,
            "truth_opened": False,
            "outcomes_or_mi_read": False,
        },
        "gpu_policy": {
            "physical_gpu_ids": [gpu],
            "maximum_concurrent_gpus": 1,
            "global_gpu_count_gate_applied": False,
        },
        "python": _artifact(python),
        "repo": str(repo),
        "model_snapshot": {
            "path": str(model),
            "revision": model.name,
            "identity_files": {path.name: _artifact(path) for path in model_identity},
            "weight_shard_bytes": {
                path.name: path.stat().st_size for path in weight_shards
            },
        },
        "inputs": {
            "prelabel_policy": _artifact(policy_path),
            "adjudicator_panel": {
                name: _artifact(adjudicator[name])
                for name in ("validation", "candidates", "items", "manifest", "identity_freeze")
            },
            "verifier_panel": {
                name: _artifact(verifier[name])
                for name in ("validation", "candidates", "items", "manifest", "identity_freeze")
            },
            "prompt_components": {
                f"adjudicator.{variant['name']}": [
                    _artifact(path) for path in variant["components"]
                ]
                for variant in adjudicator_variants
            }
            | {
                f"verifier.{variant['name']}": [
                    _artifact(path) for path in variant["components"]
                ]
                for variant in verifier_variants
            },
            "implementations": [_artifact(path) for path in implementation_paths],
        },
        "outputs": outputs,
        "stages": stages,
        "fresh_select_truth_read": False,
        "permanent_blind_consumed": False,
    }
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "queue": str(queue),
                "queue_sha256": sha256_file(queue),
                "stage_count": len(stages),
                "gpu_cell_count": sum(
                    cell["cuda_visible_devices"] is not None
                    for stage in stages
                    for cell in stage["cells"]
                ),
                "status": STATUS,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
