#!/usr/bin/env python3
"""Freeze production from a clean explicit-role GEPA selection.

The legacy production-plan freezer expects older selection schemas and always
hard-codes a two-order verifier.  This freezer preserves the exact adjudicator
branch and the two- or three-order verifier policy selected by the pre-frozen
explicit-role GEPA experiment while retaining exact all-corpus candidate-union
coverage checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


PLAN_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-plan-v1"
ROLE_FREEZE_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-freeze-v1"
ADJ_SCHEMA = "silver-match-v3-explicit-role-adjudicator-selection-v1"
VERIFIER_SCHEMA = "silver-match-v3-explicit-role-verifier-selection-v1"
POLICY_SCHEMA = "silver-match-v3-verifier-production-policy-v2"


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_ref(ref: dict[str, Any], label: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
        raise ValueError(f"{label} is missing or hash-drifted: {path}")
    return path


def _arg(argv: list[str], flag: str, default: str | None = None) -> str:
    count = argv.count(flag)
    if count == 0 and default is not None:
        return default
    if count != 1:
        raise ValueError(f"frozen command must contain exactly one {flag}")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"frozen command lacks value for {flag}")
    return str(argv[index + 1])


def _prompt_block(variant: dict[str, Any], expected_sha: str) -> dict[str, Any]:
    if str(variant.get("combined_prompt_sha256") or "") != expected_sha:
        raise ValueError("selected prompt hash differs from frozen variant")
    components: list[dict[str, str]] = []
    for ref in variant.get("components") or []:
        path = _validate_ref(ref, "selected prompt component")
        components.append({"path": str(path), "sha256": sha256_file(path)})
    if not components:
        raise ValueError("selected prompt variant has no components")
    return {
        "prompt": components[0]["path"],
        "prompt_addons": [row["path"] for row in components[1:]],
        "prompt_components": {
            row["path"]: {"sha256": row["sha256"]} for row in components
        },
    }


def _candidate_scope(
    *,
    manifest_path: Path,
    task: str,
    candidate_path: Path,
    candidate_audit_paths: list[Path],
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
    audited_corpora: set[str] = set()
    audited_count = 0
    audit_records: dict[str, str] = {}
    for path in candidate_audit_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if (
            audit.get("complete") is not True
            or audit.get("task") != task
            or audit.get("manifest_sha256") != sha256_file(manifest_path)
            or str(audit.get("bank_source_sha256")) != bank_sha
        ):
            raise ValueError(f"candidate audit is not valid for frozen task: {path}")
        corpus = str(audit.get("corpus") or "")
        if not corpus or corpus in audited_corpora:
            raise ValueError(f"duplicate/missing corpus candidate audit: {corpus!r}")
        audited_corpora.add(corpus)
        audited_count += int(audit.get("observed_count") or 0)
        for raw_candidate, value in (audit.get("candidate_inputs") or {}).items():
            candidate = str(Path(raw_candidate).resolve())
            if candidate in audited_candidates:
                raise ValueError(f"duplicate audited candidate artifact: {candidate}")
            audited_candidates[candidate] = str(value.get("sha256") or "")
        audit_records[str(path.resolve())] = sha256_file(path)
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
        str(Path(path).resolve()): str(value.get("sha256") or "")
        for path, value in (candidate_meta.get("inputs") or {}).items()
    }
    if (
        combined_inputs != audited_candidates
        or int(candidate_meta.get("count", -1)) != expected_count
    ):
        raise ValueError("combined candidate artifact is not exactly the audited task union")
    return {
        "manifest": manifest,
        "expected_corpora": sorted(expected_corpora),
        "expected_count": expected_count,
        "bank_sha": bank_sha,
        "candidate_meta_path": candidate_meta_path,
        "candidate_audits": audit_records,
    }


def _adjudicator_runtime(
    *,
    plan: dict[str, Any],
    variant_name: str,
    prompt_sha: str,
    prompt_components: dict[str, dict[str, str]],
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for cell in plan.get("commands") or []:
        if (
            cell.get("stage") == "adjudicator"
            and cell.get("role") == "prompt_dev"
            and cell.get("variant") == variant_name
        ):
            order = str(cell.get("order") or "")
            if order in cells:
                raise ValueError(f"duplicate selected adjudicator cell: {order}")
            cells[order] = cell
    if set(cells) != {"original", "hashed"}:
        raise ValueError("selected adjudicator lacks original+hashed prompt-dev cells")

    expected_component_map = {
        path: value["sha256"] for path, value in prompt_components.items()
    }
    metas: dict[str, dict[str, str]] = {}
    frozen_commands: dict[str, dict[str, Any]] = {}
    models: set[str] = set()
    renderings: list[dict[str, Any]] = []
    effective_sampling: list[dict[str, Any]] = []
    for order in ("original", "hashed"):
        direct = cells[order].get("direct_batch_command") or {}
        if direct.get("module") != "scripts.tools.silver_match_v3.adjudicate_gemma":
            raise ValueError("selected adjudicator was not a direct batch cell")
        argv = [str(value) for value in direct.get("argv") or []]
        output = Path(_arg(argv, "--output")).resolve()
        meta_path = output.with_suffix(output.suffix + ".meta.json")
        if not output.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"selected adjudicator {order} dev output is incomplete")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("order_mode") != order
            or _arg(argv, "--order-mode") != order
            or meta.get("output_sha256") != sha256_file(output)
            or meta.get("prompt_sha256") != prompt_sha
            or int(meta.get("max_candidates", -1)) != int(plan.get("candidate_k", -1))
            or (meta.get("prompt_component_sha256") or {}) != expected_component_map
            or int(meta.get("invalid_count", -1)) != 0
        ):
            raise ValueError(f"selected adjudicator {order} dev runtime is inconsistent")
        model = str(meta.get("model") or "")
        rendering = meta.get("prompt_rendering") or {}
        if not model or not rendering or _arg(argv, "--model") != model:
            raise ValueError(f"selected adjudicator {order} lacks bound rendering")
        sampling = {
            "temperature": 0.0,
            "max_model_len": int(_arg(argv, "--max-model-len", "8192")),
            "max_tokens": int(_arg(argv, "--max-tokens", "160")),
            "seed": int(_arg(argv, "--seed", "17")),
            "batch_size": int(_arg(argv, "--batch-size", "256")),
            "gpu_memory_utilization": float(
                _arg(argv, "--gpu-memory-utilization", "0.9")
            ),
        }
        models.add(model)
        renderings.append(rendering)
        effective_sampling.append(sampling)
        metas[order] = _artifact(meta_path)
        frozen_commands[order] = direct
    if len(models) != 1 or renderings[0] != renderings[1] or effective_sampling[0] != effective_sampling[1]:
        raise ValueError("selected adjudicator orders used different runtime settings")
    return {
        "model": models.pop(),
        "prompt_rendering": renderings[0],
        "production_sampling": effective_sampling[0],
        "selected_dev_run_meta": metas,
        "selected_dev_direct_commands": frozen_commands,
    }


def freeze_plan(
    *,
    manifest_path: Path,
    task: str,
    candidate_path: Path,
    candidate_audit_paths: list[Path],
    retriever_selection_path: Path,
    explicit_plan_path: Path,
    role_freeze_path: Path,
    adjudicator_selection_path: Path,
    verifier_selection_path: Path,
    verifier_policy_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    scope = _candidate_scope(
        manifest_path=manifest_path,
        task=task,
        candidate_path=candidate_path,
        candidate_audit_paths=candidate_audit_paths,
    )
    retriever = json.loads(retriever_selection_path.read_text(encoding="utf-8"))
    if retriever.get("task") != task or retriever.get("selection_split") not in {
        "dev",
        "external_dev_only",
    }:
        raise ValueError("retriever is not a task-matched dev selection")

    explicit_plan = json.loads(explicit_plan_path.read_text(encoding="utf-8"))
    role_freeze = json.loads(role_freeze_path.read_text(encoding="utf-8"))
    explicit_sha = sha256_file(explicit_plan_path)
    if (
        explicit_plan.get("schema_version") != PLAN_SCHEMA
        or explicit_plan.get("status") != "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE"
        or explicit_plan.get("task") != task
        or explicit_plan.get("candidate_bank_source_sha256") != scope["bank_sha"]
        or (explicit_plan.get("inputs") or {}).get("manifest", {}).get("sha256")
        != sha256_file(manifest_path)
        or int(explicit_plan.get("candidate_k", -1)) != 50
        or (explicit_plan.get("scientific_scope") or {}).get("test_or_blind_audit_consumed")
        is not False
        or (explicit_plan.get("scientific_scope") or {}).get("production_consumed") is not False
        or (explicit_plan.get("scientific_scope") or {}).get("outcomes_or_mi_used") is not False
        or role_freeze.get("schema_version") != ROLE_FREEZE_SCHEMA
        or role_freeze.get("task") != task
        or (role_freeze.get("command_plan") or {}).get("sha256") != explicit_sha
        or role_freeze.get("test_or_blind_audit_consumed") is not False
        or role_freeze.get("production_consumed") is not False
    ):
        raise ValueError("explicit-role plan/FREEZE is not clean, task-matched, and linked")

    adjudicator = json.loads(adjudicator_selection_path.read_text(encoding="utf-8"))
    verifier = json.loads(verifier_selection_path.read_text(encoding="utf-8"))
    policy = json.loads(verifier_policy_path.read_text(encoding="utf-8"))
    if (
        adjudicator.get("schema_version") != ADJ_SCHEMA
        or adjudicator.get("task") != task
        or adjudicator.get("status") != "selected"
        or adjudicator.get("selection_role") != "prompt_dev"
        or adjudicator.get("test_or_blind_audit_consumed") is not False
        or adjudicator.get("production_consumed") is not False
        or adjudicator.get("outcomes_or_mi_used") is not False
        or (adjudicator.get("inputs") or {}).get("command_plan", {}).get("sha256")
        != explicit_sha
    ):
        raise ValueError("adjudicator is not a clean explicit-role selection")
    if (
        verifier.get("schema_version") != VERIFIER_SCHEMA
        or verifier.get("task") != task
        or verifier.get("status") != "selected"
        or verifier.get("selection_role") != "prompt_dev"
        or verifier.get("test_or_blind_audit_consumed") is not False
        or verifier.get("production_consumed") is not False
        or verifier.get("outcomes_or_mi_used") is not False
        or (verifier.get("inputs") or {}).get("plan", {}).get("sha256") != explicit_sha
    ):
        raise ValueError("verifier is not a clean explicit-role selection")
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or policy.get("task") != task
        or policy.get("selection_role") != "prompt_dev"
        or policy.get("test_or_blind_audit_consumed") is not False
        or policy.get("production_consumed") is not False
        or policy.get("outcomes_or_mi_used") is not False
        or policy.get("may_run_on_production_unlabeled_norms") is not True
        or (policy.get("dev_gate") or {}).get("cleared") is not True
        or (policy.get("inputs") or {}).get("selection", {}).get("sha256")
        != sha256_file(verifier_selection_path)
        or (policy.get("inputs") or {}).get("command_plan", {}).get("sha256")
        != explicit_sha
    ):
        raise ValueError("verifier production policy is not clean, linked, and dev-cleared")

    chosen_adj = adjudicator.get("chosen") or {}
    chosen_verifier = verifier.get("chosen") or {}
    adj_name = str(chosen_adj.get("name") or "")
    verifier_name = str(chosen_verifier.get("verifier_variant") or "")
    adj_prompt_sha = str(chosen_adj.get("prompt_sha256") or "")
    verifier_prompt_sha = str(chosen_verifier.get("verifier_prompt_sha256") or "")
    adj_variants = {
        str(row.get("name") or ""): row for row in explicit_plan.get("adjudicator_variants") or []
    }
    verifier_variants = {
        str(row.get("name") or ""): row for row in explicit_plan.get("verifier_variants") or []
    }
    if adj_name not in adj_variants or verifier_name not in verifier_variants:
        raise ValueError("selected prompt variant is absent from explicit-role plan")
    adj_prompt = _prompt_block(adj_variants[adj_name], adj_prompt_sha)
    verifier_prompt = _prompt_block(verifier_variants[verifier_name], verifier_prompt_sha)
    if str((policy.get("prompt") or {}).get("rendered_prompt_sha256") or "") != verifier_prompt_sha:
        raise ValueError("verifier selection and production policy prompt differ")
    if policy.get("selected_policy") != chosen_verifier.get("policy"):
        raise ValueError("verifier selection and production policy name differ")
    policy_components = {
        str(row.get("path") or ""): str(row.get("sha256") or "")
        for row in (policy.get("prompt") or {}).get("components") or []
    }
    expected_verifier_components = {
        path: value["sha256"] for path, value in verifier_prompt["prompt_components"].items()
    }
    if policy_components != expected_verifier_components:
        raise ValueError("verifier policy prompt components differ from selected variant")

    adj_runtime = _adjudicator_runtime(
        plan=explicit_plan,
        variant_name=adj_name,
        prompt_sha=adj_prompt_sha,
        prompt_components=adj_prompt["prompt_components"],
    )
    orders = list((policy.get("order_policy") or {}).get("orders") or [])
    if orders not in (["original", "hashed"], ["original", "hashed", "reverse"]):
        raise ValueError("verifier production policy has unsupported order topology")
    rendering = policy.get("rendering") or {}
    if not rendering.get("model") or not rendering.get("max_alternatives"):
        raise ValueError("verifier production policy lacks exact rendering")
    for order, refs in (policy.get("selected_prompt_dev_runs") or {}).items():
        if order not in orders:
            raise ValueError("policy binds a prompt-dev order outside selected topology")
        _validate_ref(refs.get("output") or {}, f"verifier selected dev {order} output")
        _validate_ref(refs.get("meta") or {}, f"verifier selected dev {order} meta")
    if set((policy.get("selected_prompt_dev_runs") or {})) != set(orders):
        raise ValueError("policy does not bind every selected prompt-dev verifier order")

    implementation_root = repo_root / "scripts/tools/silver_match_v3"
    return {
        "schema_version": "silver-match-v3-task-production-plan-v2",
        "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
        "task": task,
        "manifest": _artifact(manifest_path),
        "bank_source_sha256": scope["bank_sha"],
        "corpora": scope["expected_corpora"],
        "expected_count": scope["expected_count"],
        "candidate_union": _artifact(candidate_path),
        "candidate_union_meta": _artifact(scope["candidate_meta_path"]),
        "candidate_audits": scope["candidate_audits"],
        "retriever_selection": _artifact(retriever_selection_path),
        "lineage_artifacts": {
            "explicit_role_command_plan": _artifact(explicit_plan_path),
            "explicit_role_freeze": _artifact(role_freeze_path),
        },
        "adjudicator": {
            "selection": _artifact(adjudicator_selection_path),
            "selected_variant": adj_name,
            "prompt_sha256": adj_prompt_sha,
            **adj_prompt,
            "candidate_depth": 50,
            "orders": ["original", "hashed"],
            **adj_runtime,
            "implementation": _artifact(implementation_root / "adjudicate_gemma.py"),
        },
        "verifier": {
            "selection": _artifact(verifier_selection_path),
            "production_policy": _artifact(verifier_policy_path),
            "selected_variant": verifier_name,
            "selected_policy": chosen_verifier["policy"],
            "prompt_sha256": verifier_prompt_sha,
            **verifier_prompt,
            "rendering": rendering,
            "orders": orders,
            "acceptance": (policy.get("order_policy") or {}).get("retain_only_if"),
            "acceptance_mode": (policy.get("order_policy") or {}).get("acceptance_mode"),
            "selected_dev_run_meta": {
                order: refs["meta"]
                for order, refs in policy["selected_prompt_dev_runs"].items()
            },
            "blind_final_match_audit_required": True,
            "implementation": _artifact(implementation_root / "verify_gemma.py"),
            "combiner_implementation": _artifact(
                implementation_root / "combine_ordered_verifications.py"
            ),
        },
        "finalizer_implementation": _artifact(
            implementation_root / "finalize_adjudications.py"
        ),
        "runner_implementation": _artifact(
            implementation_root / "run_task_production.py"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-audit", action="append", required=True)
    parser.add_argument("--retriever-selection", required=True)
    parser.add_argument("--explicit-role-plan", required=True)
    parser.add_argument("--role-freeze")
    parser.add_argument("--adjudicator-selection", required=True)
    parser.add_argument("--verifier-selection", required=True)
    parser.add_argument("--verifier-policy", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    explicit_plan_path = Path(args.explicit_role_plan).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = freeze_plan(
        manifest_path=Path(args.manifest).resolve(),
        task=args.task,
        candidate_path=Path(args.candidates).resolve(),
        candidate_audit_paths=[Path(path).resolve() for path in args.candidate_audit],
        retriever_selection_path=Path(args.retriever_selection).resolve(),
        explicit_plan_path=explicit_plan_path,
        role_freeze_path=(
            Path(args.role_freeze).resolve()
            if args.role_freeze
            else explicit_plan_path.with_name("FREEZE.json")
        ),
        adjudicator_selection_path=Path(args.adjudicator_selection).resolve(),
        verifier_selection_path=Path(args.verifier_selection).resolve(),
        verifier_policy_path=Path(args.verifier_policy).resolve(),
        repo_root=Path(args.repo_root).resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
