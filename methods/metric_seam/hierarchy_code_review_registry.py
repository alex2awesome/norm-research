"""Join code-review hierarchy artifacts into the 990-cell readiness registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.hierarchy_code_runner import EXECUTION_SCHEMA, TRAIN_GATE_SCHEMA
from methods.metric_seam.hierarchy_fidelity_merge import SCHEMA as FIDELITY_SCHEMA
from methods.metric_seam.hierarchy_heldout_readiness import SCHEMA as HELDOUT_SCHEMA
from methods.metric_seam.hierarchy_prompt_batch import (
    CURRENT_SCHEMA as CURRENT_PROMPT_SCHEMA,
    SCHEMA as LEGACY_PROMPT_SCHEMA,
)


SCHEMA = "metric-seam.hierarchy-code-review-registry.v2"


class RegistryError(ValueError):
    """Raised when code-review progress artifacts do not join exactly."""


def build_registry(audit: Mapping, train_execution: Mapping, train_gate: Mapping,
                   heldout_readiness: Mapping, prompt_manifest: Mapping, *,
                   sources: Mapping[str, str] | None = None) -> dict:
    if audit.get("schema") != FIDELITY_SCHEMA or len(audit.get("rows", [])) != 90:
        raise RegistryError("expected canonical 90-row construct-fidelity audit")
    if train_execution.get("schema") != EXECUTION_SCHEMA or train_execution.get("phase") != "compiler_train":
        raise RegistryError("expected compiler-train execution")
    if train_gate.get("schema") != TRAIN_GATE_SCHEMA:
        raise RegistryError("expected compiler-train gate")
    if heldout_readiness.get("schema") != HELDOUT_SCHEMA:
        raise RegistryError("expected heldout readiness")
    prompt_schema = prompt_manifest.get("schema")
    if (
        prompt_schema not in {LEGACY_PROMPT_SCHEMA, CURRENT_PROMPT_SCHEMA}
        or prompt_manifest.get("status") != "compiled_unscored"
    ):
        raise RegistryError("expected unscored reconstruction prompt manifest")
    if prompt_schema == CURRENT_PROMPT_SCHEMA:
        if prompt_manifest.get("panel_content_sha256") != audit.get("panel_content_sha256"):
            raise RegistryError("prompt manifest and construct audit use different panels")
        if prompt_manifest.get("scope_statements", {}).get(
            "selected_construct_fidelity_verdict_counts"
        ) != {"partial": 21}:
            raise RegistryError("current prompt batch must preserve partial-only code scope")

    train_programs = {program["aspect_id"]: program for program in train_execution["programs"]}
    operational_cells = {
        cell_id for program in train_gate["selected_programs"] for cell_id in program["cell_ids"]
    }
    confirmatory_cells = {
        cell_id for program in heldout_readiness["confirmatory_programs"]
        for cell_id in program["cell_ids"]
    }
    prompt_cells = set(prompt_manifest["cell_ids"])
    if prompt_cells != confirmatory_cells:
        raise RegistryError("prompt cells do not equal pre-frozen confirmatory cells")

    source_paths = dict(sources or {})
    rows = []
    for audit_row in audit["rows"]:
        cell_id = audit_row["cell_id"]
        candidate = audit_row["candidate"]
        eligible = audit_row["eligible_for_relation_local_execution"]
        if eligible:
            aspect_id = candidate["aspect_id"]
            if aspect_id not in train_programs:
                raise RegistryError(f"eligible cell has no training execution: {cell_id}")
            execution_path = source_paths.get("train_execution")
            candidate_path = candidate["source_path"]
            depth_path = source_paths.get("construct_fidelity")
        else:
            execution_path = None
            candidate_path = None
            depth_path = None
        rows.append({
            "cell_id": cell_id,
            "task": "code-review",
            "level": audit_row["level"],
            "candidate_path": candidate_path,
            "decomposition_path": source_paths.get("construct_fidelity"),
            "depth_record_path": depth_path,
            "candidate_execution_path": execution_path,
            "construct_fidelity_path": source_paths.get("construct_fidelity"),
            "construct_fidelity_verdict": audit_row["verdict"],
            "audited_depth": audit_row["audited_depth"],
            "train_operational_relation_local_witness": cell_id in operational_cells,
            "heldout_confirmatory_reconstruction_ready": cell_id in confirmatory_cells,
            "prompt_reference_compiled_unscored": cell_id in prompt_cells,
            "prompt_batch_path": source_paths.get("prompt_manifest") if cell_id in prompt_cells else None,
            "whole_construct_code_fidelity": audit_row["verdict"] == "exact",
            "program_provenance": "retrospective historical code-review A-bank seed" if eligible else None,
            "frozen_reference_path": None,
            "sealed_evaluation_path": None,
            "certificate_path": None,
            "isomorphism_path": None,
        })

    return {
        "schema": SCHEMA,
        "status": "code_arm_complete_prompt_reference_unscored",
        "task": "code-review",
        "sources": source_paths,
        "summary": {
            "n_cells": len(rows),
            "n_construct_fidelity_audited": len(rows),
            "n_relation_local_static_fidelity": sum(row["candidate_path"] is not None for row in rows),
            "n_train_operational_relation_mappings": len(operational_cells),
            "n_heldout_confirmatory_reconstruction_ready": len(confirmatory_cells),
            "n_prompt_references_compiled_unscored": len(prompt_cells),
            "prompt_manifest_schema": prompt_schema,
            "n_prompt_channels_compiled": (
                int(prompt_manifest.get("n_channels", 0))
                if prompt_schema == CURRENT_PROMPT_SCHEMA
                else len(prompt_manifest.get("channels", []))
            ),
            "n_unique_prompt_program_vectors": prompt_manifest.get(
                "n_unique_program_vectors"
            ),
            "n_prompt_jobs_compiled_unscored": int(prompt_manifest.get("n_jobs", 0)),
            "n_whole_construct_code_fidelity": sum(row["whole_construct_code_fidelity"] for row in rows),
            "n_prompt_references_scored": 0,
            "n_isomorphism_adjudications": 0,
        },
        "registry": {row["cell_id"]: {key: value for key, value in row.items() if key != "cell_id"}
                     for row in rows},
        "rows": rows,
    }


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train-execution", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--heldout-readiness", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --force")
    paths = {
        "construct_fidelity": str(args.audit),
        "train_execution": str(args.train_execution),
        "train_gate": str(args.train_gate),
        "heldout_readiness": str(args.heldout_readiness),
        "prompt_manifest": str(args.prompt_manifest),
    }
    payload = build_registry(
        _load(args.audit), _load(args.train_execution), _load(args.train_gate),
        _load(args.heldout_readiness), _load(args.prompt_manifest), sources=paths,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
