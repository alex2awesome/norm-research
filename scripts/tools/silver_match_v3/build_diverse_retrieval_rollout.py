#!/usr/bin/env python3
"""Freeze the Humor-first, all-corpus diverse retrieval rollout inventory.

This is a non-executing bridge artifact.  It pins the canonical coverage and
known existing K50 lanes, then states the complete-bank lanes and deterministic
K50 union that must be materialized before CE pair generation.  Missing remote
runtime identities stay explicit blockers rather than being guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


TASK_ORDER = (
    "humor",
    "code-review",
    "creative-writing",
    "math-stackexchange",
    "peer-review",
    "press-releases",
    "legal-outcome-prediction",
    "notice-and-comment",
)


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _candidate_lane(meta: dict[str, Any]) -> dict[str, Any]:
    inputs = meta.get("inputs") or {}
    if len(inputs) != 1:
        raise ValueError("existing candidate coverage must bind exactly one artifact")
    path, identity = next(iter(inputs.items()))
    return {
        "name": "existing-selected-k50",
        "kind": "audited_prefix",
        "expected_k": int(meta["expected_k"]),
        "candidate": {
            "path": path,
            "sha256": identity["sha256"],
            "meta_path": identity["meta"],
            "meta_sha256": identity["meta_sha256"],
            "audit_path": meta["audit"],
            "audit_sha256": meta["audit_sha256"],
        },
    }


def build(
    *,
    coverage_path: Path,
    bank_manifest_path: Path,
    humor_selection_path: Path,
    humor_capture_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    coverage_path = coverage_path.resolve()
    bank_manifest_path = bank_manifest_path.resolve()
    humor_selection_path = humor_selection_path.resolve()
    humor_capture_path = humor_capture_path.resolve()
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    humor_selection = json.loads(humor_selection_path.read_text(encoding="utf-8"))
    capture = json.loads(humor_capture_path.read_text(encoding="utf-8"))

    manifest = coverage.get("manifest") or {}
    corpora = (coverage.get("extractions") or {}).get("corpora") or {}
    candidate_corpora = (coverage.get("candidate_retrieval") or {}).get("corpora") or {}
    banks = bank_manifest.get("banks") or {}
    if (
        coverage.get("extractions", {}).get("complete") is not True
        or int(manifest.get("total_tasks", -1)) != len(TASK_ORDER)
        or int(manifest.get("total_corpora", -1)) != len(corpora)
        or len(corpora) != 23
        or set(banks) != set(TASK_ORDER)
    ):
        raise ValueError("coverage/bank manifest is not the canonical eight-task inventory")
    if humor_selection.get("task") != "humor" or humor_selection.get("status") != "SELECTED_FOR_PRODUCTION_RETRIEVAL":
        raise ValueError("Humor v2 retriever selection is not production-selected")
    chosen = humor_selection.get("chosen") or {}
    if (
        chosen.get("kind") != "nemotron_lora_adapter"
        or humor_selection.get("selection_split") != "external_dev_only"
        or humor_selection.get("frozen_external_test_consumed") is not False
        or not chosen.get("external_dev_metrics", {}).get("promotion_gate", {}).get("passed")
    ):
        raise ValueError("Humor v2 selection did not pass the sealed external-dev gate")

    task_rows: list[dict[str, Any]] = []
    for priority, task in enumerate(TASK_ORDER, start=1):
        task_corpora = {
            name: {
                "count": int(meta["count"]),
                "path": meta["path"],
                "sha256": meta["sha256"],
            }
            for name, meta in sorted(corpora.items())
            if meta.get("task") == task
        }
        if not task_corpora:
            raise ValueError(f"missing task corpora: {task}")
        bank = banks[task]
        existing = {
            corpus: _candidate_lane(candidate_corpora[corpus])
            for corpus in task_corpora
            if corpus in candidate_corpora
        }
        selected = (
            {
                "kind": "nemotron_lora_adapter",
                "name": chosen["name"],
                "selection_evidence": _artifact(humor_selection_path),
                "base_model": chosen["base_model"],
                "adapter": chosen["adapter"],
                "retrieval_geometry": chosen["retrieval_geometry"],
                "external_test_consumed": False,
            }
            if task == "humor"
            else (coverage.get("retriever_selections", {}).get("tasks", {}).get(task))
        )
        task_rows.append(
            {
                "priority": priority,
                "task": task,
                "bank": {
                    "count": int(bank["count"]),
                    "path": bank["path"],
                    "source_sha256": bank["source_sha256"],
                },
                "corpora": task_corpora,
                "corpus_count": len(task_corpora),
                "norm_count": sum(row["count"] for row in task_corpora.values()),
                "selected_retriever": selected,
                "existing_audited_prefix_lanes": existing,
                "required_materialization": {
                    "generated_complete_bank_lanes": [
                        {
                            "name": "selected-primary-fullbank",
                            "depth": int(bank["count"]),
                            "selection": "selected_retriever",
                        },
                        {
                            "name": "independent-bge-fullbank",
                            "depth": int(bank["count"]),
                            "encoder": "BAAI/bge-large-en-v1.5 immutable snapshot",
                            "selection": "task/dev/full-bank fusion binding required",
                        },
                    ],
                    "ce_union": {
                        "name": "selected-bge-existing-rrf-k50-v1",
                        "algorithm": "weighted-complete-bank-rrf-v1",
                        "output_k": min(50, int(bank["count"])),
                        "generated_full_bank_lane_count": 2,
                        "include_existing_audited_prefix_when_present": True,
                        "one_exact_row_per_canonical_norm": True,
                    },
                    "post_materialization_audit_required": True,
                    "diagnostic_subset_substitution_forbidden": True,
                },
                "release_ready": False,
            }
        )

    inventory = {
        "schema_version": "silver-match-v3-diverse-retrieval-rollout-v1",
        "status": "FROZEN_NON_EXECUTING_ROLLOUT_INVENTORY",
        "release_ready": False,
        "coverage": _artifact(coverage_path),
        "bank_manifest": _artifact(bank_manifest_path),
        "canonical_manifest": manifest,
        "task_order": list(TASK_ORDER),
        "task_count": len(task_rows),
        "corpus_count": sum(row["corpus_count"] for row in task_rows),
        "norm_count": sum(row["norm_count"] for row in task_rows),
        "bank_leaf_count": sum(row["bank"]["count"] for row in task_rows),
        "tasks": task_rows,
        "global_contract": {
            "minimum_independent_complete_bank_lanes_per_corpus": 2,
            "ce_union_output_k": 50,
            "batched_encoder_inference": True,
            "openai_compatible_server": False,
            "append_only": True,
            "release_requires_hash_and_shape_audit": True,
            "notice_and_comment_scheduled_last": True,
        },
    }

    humor = next(row for row in task_rows if row["task"] == "humor")
    humor_existing = humor["existing_audited_prefix_lanes"].get("humor_multi")
    if humor["norm_count"] != 77378 or humor_existing is None:
        raise ValueError("Humor inventory is not the exact 77,378-row production corpus")
    humor_template = {
        "schema_version": "silver-match-v3-humor-full-corpus-retrieval-template-v1",
        "status": "FROZEN_TEMPLATE_NOT_EXECUTABLE",
        "release_ready": False,
        "task": "humor",
        "coverage": _artifact(coverage_path),
        "canonical_manifest": manifest,
        "bank": humor["bank"],
        "corpora": humor["corpora"],
        "corpus_count": 1,
        "norm_count": 77378,
        "selected_retriever": humor["selected_retriever"],
        "existing_human_k50_lane": humor_existing,
        "diagnostic_capture_evidence": {
            **_artifact(humor_capture_path),
            "scope": "diagnostic-only; never substitutes for the 77,378-row corpus",
            "reported_capture": capture.get("overall") or capture.get("test") or {},
        },
        "queue_spec_template": {
            "task": "humor",
            "manifest": manifest["path"],
            "selection": str(humor_selection_path),
            "output_root": "${REMOTE_APPEND_ONLY_HUMOR_RETRIEVAL_ROOT}",
            "repo_root": "${REMOTE_REPO_ROOT}",
            "python": "${REMOTE_PYTHON}",
            "gpu_index": "${ALLOWED_IDLE_GPU}",
            "full_k": 285,
            "primary_k": 50,
            "systems": [
                {
                    "name": chosen["name"],
                    "selection_name": chosen["name"],
                    "role": "primary",
                    "encoder": chosen["base_model"]["path"],
                    "adapter": chosen["adapter"]["path"],
                    "query_format": "nemotron",
                    "fusion": "${HASHED_HUMOR_DIRECT_DENSE_FUSION_V2}",
                },
                {
                    "name": "bge-large-en-v1.5",
                    "role": "diverse",
                    "encoder": "/lfs/skampere3/0/shared_hf_cache/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                    "query_format": "raw",
                    "fusion": "${HASHED_HUMOR_BGE_DEV_FULLBANK_FUSION}",
                },
            ],
            "existing_lanes": [
                {
                    "name": "human-only-v1",
                    "expected_k": 50,
                    "candidates": {
                        "humor_multi": humor_existing["candidate"]["path"]
                    },
                    "audits": {
                        "humor_multi": humor_existing["candidate"]["audit_path"]
                    },
                }
            ],
            "union": {
                "name": "nemotron-bge-human-rrf-k50-v1",
                "output_k": 50,
                "rank_constant": 60.0,
                "lane_weights": {
                    chosen["name"]: 1.0,
                    "bge-large-en-v1.5": 1.0,
                    "human-only-v1": 1.0,
                },
            },
        },
        "expected_materialization": {
            "generated_complete_bank_artifacts": 2,
            "generated_rows_per_artifact": 77378,
            "generated_candidates_per_row": 285,
            "reused_audited_human_prefix_artifacts": 1,
            "union_artifacts": 1,
            "union_rows": 77378,
            "union_candidates_per_row": 50,
            "minimum_complete_bank_lane_identities_in_union_meta": 2,
        },
        "blockers_to_executable_freeze": [
            "normalize the v2 Nemotron LoRA selection into the generic freezer selection contract",
            "bind and hash the selected direct-dense Humor fusion artifact",
            "bind and hash the Humor BGE task/dev/full-bank fusion artifact",
            "rehash remote encoder, adapter, canonical corpus, bank, and Human K50 artifacts",
            "choose an allowed genuinely idle GPU only at launch time",
        ],
        "release_blockers": [
            "materialize and audit both complete-bank lanes over all 77,378 norms",
            "materialize and audit the deterministic K50 union",
            "run the final adjudicator with typed abstention/noise outputs",
            "pass independent release-risk and MI-join audits",
        ],
    }
    return inventory, humor_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--bank-manifest", required=True)
    parser.add_argument("--humor-selection", required=True)
    parser.add_argument("--humor-capture", required=True)
    parser.add_argument("--inventory-output", required=True)
    parser.add_argument("--humor-output", required=True)
    args = parser.parse_args()
    inventory, humor = build(
        coverage_path=Path(args.coverage),
        bank_manifest_path=Path(args.bank_manifest),
        humor_selection_path=Path(args.humor_selection),
        humor_capture_path=Path(args.humor_capture),
    )
    for raw_path, payload in (
        (args.inventory_output, inventory),
        (args.humor_output, humor),
    ):
        path = Path(raw_path).resolve()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(path), "sha256": sha256_file(path)}))


if __name__ == "__main__":
    main()
