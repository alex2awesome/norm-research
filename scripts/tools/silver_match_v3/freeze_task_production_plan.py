#!/usr/bin/env python3
"""Freeze one task's validated retriever/adjudicator/verifier production plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


def _artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _resolve_component(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _validate_prompt_components(
    chosen: dict[str, Any], repo_root: Path, role: str
) -> dict[str, dict[str, str]]:
    expected = chosen.get("prompt_component_sha256") or {}
    if not expected:
        raise ValueError(f"{role} selection has no prompt-component hashes")
    output = {}
    for raw_path, digest in expected.items():
        path = _resolve_component(str(raw_path), repo_root).resolve()
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"{role} prompt component changed: {path}")
        output[str(path)] = {"sha256": actual}
    return output


def _validate_adjudicator_dev_runs(
    chosen: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Recover and bind the exact rendering used by the selected dev runs."""
    inputs = chosen.get("inputs") or {}
    expected_components = chosen.get("prompt_component_sha256") or {}
    metas: dict[str, dict[str, str]] = {}
    models = set()
    renderings = []
    for order in ("original", "hashed"):
        ref = inputs.get(order) or {}
        output = Path(str(ref.get("path") or ""))
        expected_output_sha = str(ref.get("sha256") or "")
        if not output.is_absolute():
            output = repo_root / output
        output = output.resolve()
        if not output.exists() or sha256_file(output) != expected_output_sha:
            raise ValueError(f"selected adjudicator {order} output changed: {output}")
        meta_path = output.with_suffix(output.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("order_mode") != order
            or meta.get("output_sha256") != expected_output_sha
            or meta.get("prompt_sha256") != chosen.get("prompt_sha256")
            or int(meta.get("max_candidates", -1)) != 50
            or (meta.get("prompt_component_sha256") or {}) != expected_components
        ):
            raise ValueError(f"selected adjudicator {order} metadata is inconsistent")
        model = str(meta.get("model") or "")
        rendering = meta.get("prompt_rendering") or {}
        if not model or not rendering:
            raise ValueError(f"selected adjudicator {order} metadata lacks rendering")
        models.add(model)
        renderings.append(rendering)
        metas[order] = _artifact(meta_path)
    if len(models) != 1 or renderings[0] != renderings[1]:
        raise ValueError("selected adjudicator orders used different runtime rendering")
    return {
        "model": models.pop(),
        "prompt_rendering": renderings[0],
        "selected_dev_run_meta": metas,
    }


def freeze_plan(
    *,
    manifest_path: Path,
    task: str,
    candidate_path: Path,
    candidate_audit_paths: list[Path],
    retriever_selection_path: Path,
    adjudicator_selection_path: Path,
    verifier_selection_path: Path,
    verifier_policy_path: Path,
    repo_root: Path,
    adjudicator_max_model_len: int = 8192,
    adjudicator_max_tokens: int = 160,
    adjudicator_seed: int = 17,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in manifest.get("banks", {}):
        raise KeyError(task)
    expected_corpora = {
        corpus for corpus, meta in manifest["corpora"].items() if meta["task"] == task
    }
    expected_count = sum(
        int(manifest["corpora"][corpus]["count"]) for corpus in expected_corpora
    )
    bank_sha = str(manifest["banks"][task]["source_sha256"])

    audited_candidates: dict[str, str] = {}
    audited_corpora = set()
    audited_count = 0
    audit_records = {}
    for path in candidate_audit_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if (
            audit.get("complete") is not True
            or audit.get("task") != task
            or audit.get("manifest_sha256") != sha256_file(manifest_path)
            or str(audit.get("bank_source_sha256")) != bank_sha
        ):
            raise ValueError(f"candidate audit is not valid for frozen task: {path}")
        corpus = str(audit["corpus"])
        if corpus in audited_corpora:
            raise ValueError(f"duplicate corpus audit: {corpus}")
        audited_corpora.add(corpus)
        audited_count += int(audit["observed_count"])
        for candidate, value in audit["candidate_inputs"].items():
            if candidate in audited_candidates:
                raise ValueError(f"duplicate audited candidate artifact: {candidate}")
            audited_candidates[str(Path(candidate).resolve())] = str(value["sha256"])
        audit_records[str(path)] = sha256_file(path)
    if audited_corpora != expected_corpora or audited_count != expected_count:
        raise ValueError(
            f"task candidate audits are incomplete: corpora={audited_corpora ^ expected_corpora}, "
            f"count={audited_count}/{expected_count}"
        )

    candidate_meta_path = candidate_path.with_suffix(candidate_path.suffix + ".meta.json")
    candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    if candidate_meta.get("sha256") != sha256_file(candidate_path):
        raise ValueError("combined candidate hash differs from metadata")
    combined_inputs = {
        str(Path(path).resolve()): str(value["sha256"])
        for path, value in candidate_meta.get("inputs", {}).items()
    }
    if combined_inputs != audited_candidates or int(candidate_meta.get("count", -1)) != expected_count:
        raise ValueError("combined candidate artifact is not exactly the audited task union")

    retriever = json.loads(retriever_selection_path.read_text(encoding="utf-8"))
    if retriever.get("task") != task or retriever.get("selection_split") not in {
        "dev",
        "external_dev_only",
    }:
        raise ValueError("retriever is not a task-matched dev selection")
    adjudicator = json.loads(adjudicator_selection_path.read_text(encoding="utf-8"))
    if (
        adjudicator.get("task") != task
        or adjudicator.get("selection_split") not in {"dev", "external_dev_only"}
        or int(adjudicator.get("candidate_depth", -1)) != 50
    ):
        raise ValueError("adjudicator is not the task-matched K50 dev selection")
    adjudicator_chosen = adjudicator.get("chosen") or {}
    adjudicator_components = _validate_prompt_components(
        adjudicator_chosen, repo_root, "adjudicator"
    )
    adjudicator_dev_runtime = _validate_adjudicator_dev_runs(
        adjudicator_chosen, repo_root
    )

    verifier = json.loads(verifier_selection_path.read_text(encoding="utf-8"))
    verifier_chosen = verifier.get("chosen") or {}
    if (
        verifier.get("task") != task
        or verifier.get("selection_split") not in {"dev", "external_dev_only"}
        or verifier.get("calibration_power_status") != "supported"
        or verifier_chosen.get("statistically_supported") is not True
    ):
        raise ValueError("verifier selection lacks task-matched supported dev calibration")
    verifier_components = _validate_prompt_components(
        verifier_chosen, repo_root, "verifier"
    )
    policy = json.loads(verifier_policy_path.read_text(encoding="utf-8"))
    selection_ref = (policy.get("inputs") or {}).get("selection") or {}
    if (
        policy.get("task") != task
        or policy.get("selection_split") not in {"dev", "external_dev_only"}
        or (policy.get("dev_gate") or {}).get("cleared") is not True
        or policy.get("may_run_on_production_unlabeled_norms") is not True
        or selection_ref.get("sha256") != sha256_file(verifier_selection_path)
    ):
        raise ValueError("verifier production policy is not linked and dev-cleared")
    policy_rendered = str((policy.get("prompt") or {}).get("rendered_prompt_sha256") or "")
    if policy_rendered != str(verifier_chosen.get("prompt_sha256") or ""):
        raise ValueError("verifier policy and selection prompt hashes differ")

    return {
        "schema_version": "silver-match-v3-task-production-plan-v1",
        "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
        "task": task,
        "manifest": _artifact(manifest_path),
        "bank_source_sha256": bank_sha,
        "corpora": sorted(expected_corpora),
        "expected_count": expected_count,
        "candidate_union": _artifact(candidate_path),
        "candidate_union_meta": _artifact(candidate_meta_path),
        "candidate_audits": audit_records,
        "retriever_selection": _artifact(retriever_selection_path),
        "adjudicator": {
            "selection": _artifact(adjudicator_selection_path),
            "prompt_sha256": adjudicator_chosen["prompt_sha256"],
            "prompt": adjudicator_chosen["prompt"],
            "prompt_addons": adjudicator_chosen.get("prompt_addons") or [],
            "prompt_components": adjudicator_components,
            "candidate_depth": 50,
            "orders": ["original", "hashed"],
            **adjudicator_dev_runtime,
            "production_sampling": {
                "temperature": 0.0,
                "max_model_len": adjudicator_max_model_len,
                "max_tokens": adjudicator_max_tokens,
                "seed": adjudicator_seed,
            },
            "implementation": _artifact(
                repo_root / "scripts/tools/silver_match_v3/adjudicate_gemma.py"
            ),
        },
        "verifier": {
            "selection": _artifact(verifier_selection_path),
            "production_policy": _artifact(verifier_policy_path),
            "prompt_sha256": verifier_chosen["prompt_sha256"],
            "prompt": verifier_chosen["prompt"],
            "prompt_addons": verifier_chosen.get("prompt_addons") or [],
            "prompt_components": verifier_components,
            "rendering": policy["rendering"],
            "orders": ["original", "hashed"],
            "acceptance": policy["order_policy"]["retain_only_if"],
            "blind_final_match_audit_required": True,
            "implementation": _artifact(
                repo_root / "scripts/tools/silver_match_v3/verify_gemma.py"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-audit", action="append", required=True)
    parser.add_argument("--retriever-selection", required=True)
    parser.add_argument("--adjudicator-selection", required=True)
    parser.add_argument("--verifier-selection", required=True)
    parser.add_argument("--verifier-policy", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--adjudicator-max-model-len", type=int, default=8192)
    parser.add_argument("--adjudicator-max-tokens", type=int, default=160)
    parser.add_argument("--adjudicator-seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = freeze_plan(
        manifest_path=Path(args.manifest).resolve(),
        task=args.task,
        candidate_path=Path(args.candidates).resolve(),
        candidate_audit_paths=[Path(path).resolve() for path in args.candidate_audit],
        retriever_selection_path=Path(args.retriever_selection).resolve(),
        adjudicator_selection_path=Path(args.adjudicator_selection).resolve(),
        verifier_selection_path=Path(args.verifier_selection).resolve(),
        verifier_policy_path=Path(args.verifier_policy).resolve(),
        repo_root=Path(args.repo_root).resolve(),
        adjudicator_max_model_len=args.adjudicator_max_model_len,
        adjudicator_max_tokens=args.adjudicator_max_tokens,
        adjudicator_seed=args.adjudicator_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**plan, "output": _artifact(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
