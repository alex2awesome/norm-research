#!/usr/bin/env python3
"""Freeze a non-executable task retrieval template from an all-task audit.

This is the bridge used when canonical data live on a remote host but a queue
must be specified and reviewed locally.  It pins the authoritative manifest
identity and every task corpus/count/path/hash from the supplied coverage audit,
but deliberately leaves model-selection paths unresolved.  Consequently its
status can never be mistaken for a launchable or release-ready queue.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .common import sha256_file


IMPLEMENTATIONS = (
    "retrieve.py",
    "freeze_retrieval_queue.py",
    "run_frozen_retrieval_queue.py",
    "materialize_retrieval_lane_union.py",
    "audit_candidate_outputs.py",
)


def freeze_template(
    *,
    coverage_path: Path,
    task: str,
    bank_count: int,
    bank_source_sha256: str,
    output_root: str,
    repo_root: Path,
) -> dict[str, Any]:
    coverage_path = coverage_path.resolve()
    repo_root = repo_root.resolve()
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("schema_version") != "silver-match-v3-alltask-release-coverage-audit-v1":
        raise ValueError("unsupported all-task coverage audit")
    manifest = coverage.get("manifest") or {}
    manifest_sha = str(manifest.get("sha256") or "")
    if not manifest.get("path") or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
        raise ValueError("coverage audit lacks an authoritative manifest identity")
    if coverage.get("extractions", {}).get("complete") is not True:
        raise ValueError("canonical extraction inventory is incomplete")
    if bank_count < 1 or not re.fullmatch(r"[0-9a-f]{64}", bank_source_sha256):
        raise ValueError("invalid bank identity")

    corpora = {
        name: {
            "task": task,
            "count": int(meta["count"]),
            "path": str(meta["path"]),
            "sha256": str(meta["sha256"]),
        }
        for name, meta in sorted(coverage["extractions"]["corpora"].items())
        if meta.get("task") == task
    }
    if not corpora:
        raise ValueError(f"coverage audit has no corpora for task: {task}")
    if any(
        value["count"] < 1 or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        for value in corpora.values()
    ):
        raise ValueError("task corpus inventory has an invalid count or hash")
    missing = set(coverage.get("candidate_retrieval", {}).get("missing_corpora") or [])
    already_retrieved = set(corpora) - missing
    selection = (
        (coverage.get("retriever_selections") or {}).get("tasks") or {}
    ).get(task)
    if selection is not None:
        required = {
            "chosen_kind",
            "chosen_name",
            "path",
            "sha256",
            "fusion_path",
            "fusion_sha256",
        }
        if (
            not isinstance(selection, dict)
            or not required <= set(selection)
            or selection["chosen_kind"] not in {"nemotron_base", "adapter"}
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(selection[key]))
                for key in ("sha256", "fusion_sha256")
            )
        ):
            raise ValueError("coverage retriever selection is incomplete or invalid")
        frozen_selection = {key: selection[key] for key in sorted(required)}
    else:
        frozen_selection = None

    primary_name = str(
        frozen_selection["chosen_name"] if frozen_selection else "selected-primary"
    )
    primary_system = {
        "name": primary_name,
        "selection_name": primary_name,
        "role": "primary",
        "encoder": "${IMMUTABLE_PRIMARY_ENCODER_SNAPSHOT}",
        "query_format": "nemotron",
        "fusion": (
            frozen_selection["fusion_path"]
            if frozen_selection
            else "${DEV_SELECTED_PRIMARY_FUSION}"
        ),
    }
    if frozen_selection and frozen_selection["chosen_kind"] == "adapter":
        primary_system["adapter"] = "${FROZEN_SELECTED_TASK_ADAPTER}"

    implementation_root = repo_root / "scripts/tools/silver_match_v3"
    implementations = {
        name: {
            "path": str((implementation_root / name).resolve()),
            "sha256": sha256_file(implementation_root / name),
        }
        for name in IMPLEMENTATIONS
    }
    return {
        "schema_version": "silver-match-v3-task-retrieval-queue-template-v1",
        "status": "FROZEN_TEMPLATE_NOT_EXECUTABLE",
        "release_ready": False,
        "task": task,
        "coverage_audit": {
            "path": str(coverage_path),
            "sha256": sha256_file(coverage_path),
        },
        "authoritative_manifest": {
            "path": str(manifest["path"]),
            "sha256": manifest_sha,
        },
        "bank": {
            "count": bank_count,
            "source_sha256": bank_source_sha256,
        },
        "selected_retriever": frozen_selection,
        "corpora": corpora,
        "corpus_count": len(corpora),
        "norm_count": sum(value["count"] for value in corpora.values()),
        "retrieval_gap_at_freeze": {
            "missing_corpora": sorted(set(corpora) & missing),
            "already_retrieved_corpora": sorted(already_retrieved),
        },
        "queue_spec_template": {
            "task": task,
            "manifest": str(manifest["path"]),
            "selection": (
                frozen_selection["path"]
                if frozen_selection
                else "${EXTERNAL_DEV_ONLY_RETRIEVER_SELECTION}"
            ),
            "output_root": output_root,
            "repo_root": "${REMOTE_REPO_ROOT}",
            "python": "${REMOTE_PYTHON}",
            "gpu_index": "${ALLOWED_IDLE_GPU}",
            "full_k": bank_count,
            "primary_k": min(50, bank_count),
            "systems": [
                primary_system,
                {
                    "name": "diverse-base",
                    "role": "diverse",
                    "encoder": "${IMMUTABLE_DIVERSE_ENCODER_SNAPSHOT}",
                    "query_format": "raw",
                    "fusion": "${DEV_SELECTED_DIVERSE_FUSION}",
                },
            ],
            "union": {
                "name": "diverse-fullbank-rrf-v1",
                "output_k": min(50, bank_count),
                "rank_constant": 60.0,
                "lane_weights": {primary_name: 1.0, "diverse-base": 1.0},
            },
        },
        "expected_materialization": {
            "complete_bank_lane_artifacts": len(corpora) * 2,
            "primary_topk_artifacts": len(corpora),
            "diverse_union_artifacts": len(corpora),
            "candidate_rows_per_artifact": {
                name: value["count"] for name, value in corpora.items()
            },
            "one_exact_row_per_norm_required": True,
            "post_materialization_audit_required": True,
            "diagnostic_subset_reuse_forbidden": True,
        },
        "inference_contract": {
            "retrieval_runtime": "sentence-transformers batched encoder inference",
            "openai_compatible_server": False,
            "union_runtime": "deterministic CPU streaming projection",
        },
        "implementations": implementations,
        "blockers_to_executable_freeze": [
            (
                "validate the frozen retriever selection and fusion hashes on the live host"
                if frozen_selection
                else "bind and hash an external-dev-only retriever selection"
            ),
            "bind immutable primary and diverse encoder snapshots",
            (
                "bind a task/dev/full-bank fusion report for the diverse lane"
                if frozen_selection
                else "bind task/dev/full-bank fusion reports for both lanes"
            ),
            "choose an allowed genuinely idle GPU only at launch time",
            "run freeze_retrieval_queue.py against the live authoritative manifest",
        ],
        "release_blockers": [
            "materialize and audit every complete-bank lane",
            "materialize and audit every deterministic top-K union",
            "downstream adjudication and typed abstention remain incomplete",
            "independent final risk audit remains incomplete",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bank-count", type=int, required=True)
    parser.add_argument("--bank-source-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = freeze_template(
        coverage_path=Path(args.coverage),
        task=args.task,
        bank_count=args.bank_count,
        bank_source_sha256=args.bank_source_sha256,
        output_root=args.output_root,
        repo_root=Path(args.repo_root),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "status": payload["status"],
                "corpus_count": payload["corpus_count"],
                "norm_count": payload["norm_count"],
                "release_ready": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
