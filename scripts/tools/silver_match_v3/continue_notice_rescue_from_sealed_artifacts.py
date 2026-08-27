#!/usr/bin/env python3
"""Finish N&C rescue from revalidated sealed inference artifacts, CPU only.

The canonical final audit is published last as the completion sentinel awaited
by the existing closure wrapper.  Every downstream command refuses overwrite;
canonical targets must all be absent at freeze time.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .audit_notice_rescue_sealed_artifacts import (
    CPU_IMPLEMENTATIONS,
    EXPECTED_BANK_COUNT,
    TASK,
    audit as audit_sealed,
    _validate_artifact_lock,
    _validate_bank_binding,
)
from .common import sha256_file


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(
    command: list[str],
    *,
    cwd: Path,
    log: Path,
    verify_implementations: Any,
) -> None:
    if log.exists():
        raise FileExistsError(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    verify_implementations()
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"CPU continuation command failed ({completed.returncode}); see {log}"
        )
    verify_implementations()


def _ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _module_path(repo: Path, module: str) -> Path | None:
    if not module.startswith("scripts.tools.silver_match_v3"):
        return None
    file_path = repo / (module.replace(".", "/") + ".py")
    if file_path.is_file():
        return file_path.resolve()
    package = repo / module.replace(".", "/") / "__init__.py"
    return package.resolve() if package.is_file() else None


def _module_name(path: Path, repo: Path) -> str:
    relative = path.resolve().relative_to(repo.resolve())
    values = list(relative.with_suffix("").parts)
    if values[-1] == "__init__":
        values.pop()
    return ".".join(values)


def _local_imports(path: Path, repo: Path) -> set[Path]:
    module = _module_name(path, repo)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    output: set[Path] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                values = package.split(".")
                if node.level > len(values):
                    raise ValueError(f"invalid relative import in {path}")
                base = ".".join(values[: len(values) - node.level + 1])
                imported = f"{base}.{node.module}" if node.module else base
            else:
                imported = str(node.module or "")
            names = [imported]
            if node.module is None:
                names.extend(f"{imported}.{alias.name}" for alias in node.names)
        for name in names:
            candidate = _module_path(repo, name)
            if candidate is not None:
                output.add(candidate)
    return output


def _dependency_inventory(entrypoints: list[Path], repo: Path) -> list[Path]:
    pending = [path.resolve() for path in entrypoints]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_file() or repo.resolve() not in path.parents:
            raise ValueError(f"dependency is absent/outside repository: {path}")
        observed.add(path)
        pending.extend(sorted(_local_imports(path, repo) - observed))
        parent = path.parent
        while parent != repo.resolve():
            init = parent / "__init__.py"
            if init.is_file() and init.resolve() not in observed:
                pending.append(init.resolve())
            parent = parent.parent
    return sorted(observed)


def _snapshot_dependencies(
    *, entrypoints: list[Path], repo: Path, snapshot_root: Path
) -> tuple[Path, dict[str, dict[str, Any]]]:
    if snapshot_root.exists():
        raise FileExistsError(snapshot_root)
    inventory: dict[str, dict[str, Any]] = {}
    for source in _dependency_inventory(entrypoints, repo):
        relative = source.relative_to(repo)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_sha = sha256_file(source)
        snapshot_sha = sha256_file(destination)
        if source_sha != snapshot_sha:
            raise ValueError(f"dependency snapshot copy mismatch: {source}")
        inventory[str(relative)] = {
            "source": str(source),
            "source_sha256": source_sha,
            "snapshot": str(destination),
            "snapshot_sha256": snapshot_sha,
        }
    freeze = snapshot_root / "FREEZE.json"
    _write_new_json(
        freeze,
        {
            "schema_version": "silver-match-v3-python-dependency-snapshot-v1",
            "status": "FROZEN_TASK_LOCAL_IMPLEMENTATION_UNIVERSE",
            "entrypoints": [str(path.resolve()) for path in entrypoints],
            "dependency_count": len(inventory),
            "dependencies": inventory,
        },
    )
    return freeze, inventory


def _verify_dependency_snapshot(
    *, entrypoints: list[Path], repo: Path, inventory: dict[str, dict[str, Any]]
) -> None:
    observed = {
        str(path.relative_to(repo)) for path in _dependency_inventory(entrypoints, repo)
    }
    if observed != set(inventory):
        raise ValueError("recursive Python dependency inventory changed")
    for relative, ref in inventory.items():
        source = repo / relative
        snapshot = Path(ref["snapshot"])
        if (
            sha256_file(source) != ref["source_sha256"]
            or sha256_file(snapshot) != ref["snapshot_sha256"]
            or ref["source_sha256"] != ref["snapshot_sha256"]
        ):
            raise ValueError(f"Python implementation dependency drift: {relative}")


def _validate_blind_bank_binding(
    *, manifest_path: Path, blind_root: Path, task: str, expected_count: int
) -> None:
    manifest = _json(manifest_path)
    bank_meta = (manifest.get("banks") or {}).get(task) or {}
    source_sha = str(bank_meta.get("source_sha256") or "")
    bank_path = Path(str(bank_meta.get("path") or ""))
    if not bank_path.is_absolute():
        bank_path = manifest_path.parent / bank_path
    if not source_sha or not bank_path.is_file():
        raise ValueError(f"manifest lacks authoritative bank linkage: {task}")
    sample = _json(blind_root / "sample_report.json")
    sample_bank = (sample.get("bank_outputs") or {}).get(task) or {}
    sample_bank_path = Path(str(sample_bank.get("path") or ""))
    if (
        sample.get("manifest_sha256") != sha256_file(manifest_path)
        or sample_bank.get("source_sha256") != source_sha
        or not sample_bank_path.is_file()
        or sample_bank.get("sha256") != sha256_file(sample_bank_path)
        or sha256_file(sample_bank_path) != sha256_file(bank_path)
    ):
        raise ValueError(f"blind sample authoritative bank mismatch: {blind_root}")
    validation = _json(blind_root / f"task__{task}.label_pack/validation.json")
    canonical_ref = (validation.get("inputs") or {}).get("canonical_bank") or {}
    if (
        int(validation.get("bank_metric_count", -1)) != expected_count
        or validation.get("bank_source_sha256") != source_sha
        or canonical_ref.get("source_sha256") != source_sha
        or canonical_ref.get("sha256") != sha256_file(bank_path)
    ):
        raise ValueError(f"blind label-pack authoritative bank mismatch: {blind_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-lock", required=True)
    parser.add_argument("--rescue-root", required=True)
    parser.add_argument("--primary", action="append", required=True)
    parser.add_argument("--adjudicator-selection", required=True)
    parser.add_argument("--verifier-selection", required=True)
    parser.add_argument("--verifier-policy", required=True)
    parser.add_argument("--manual-labels", required=True)
    parser.add_argument("--manual-validation", required=True)
    parser.add_argument("--prewrapper-audit", required=True)
    parser.add_argument("--postwrapper-audit", required=True)
    parser.add_argument("--current-sealed-audit", required=True)
    parser.add_argument("--failed-wrapper-record", required=True)
    parser.add_argument("--unresolved-reconciliation", required=True)
    parser.add_argument("--prior-continuation-failure", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    artifact_lock_path = Path(args.artifact_lock).resolve()
    rescue = Path(args.rescue_root).resolve()
    primary = [Path(path).resolve() for path in args.primary]
    adjudicator_selection = Path(args.adjudicator_selection).resolve()
    verifier_selection = Path(args.verifier_selection).resolve()
    verifier_policy = Path(args.verifier_policy).resolve()
    manual_labels = Path(args.manual_labels).resolve()
    manual_validation = Path(args.manual_validation).resolve()
    prewrapper_audit_path = Path(args.prewrapper_audit).resolve()
    postwrapper_audit_path = Path(args.postwrapper_audit).resolve()
    current_sealed_audit_path = Path(args.current_sealed_audit).resolve()
    failed_wrapper_record_path = Path(args.failed_wrapper_record).resolve()
    unresolved_reconciliation_path = Path(args.unresolved_reconciliation).resolve()
    prior_continuation_failure_path = Path(args.prior_continuation_failure).resolve()
    output_root = Path(args.output_root).resolve()
    python = Path(args.python).resolve()
    repo = Path(args.repo).resolve()
    if len(primary) != 2:
        parser.error("exactly two --primary inputs are required")
    for path in [
        manifest,
        artifact_lock_path,
        *primary,
        adjudicator_selection,
        verifier_selection,
        verifier_policy,
        manual_labels,
        manual_validation,
        prewrapper_audit_path,
        postwrapper_audit_path,
        current_sealed_audit_path,
        failed_wrapper_record_path,
        unresolved_reconciliation_path,
        prior_continuation_failure_path,
        python,
        repo,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    freeze_path = output_root / "CPU_CONTINUATION_FREEZE.json"
    ready_path = output_root / "CPU_CONTINUATION_READY_TO_PUBLISH.json"
    published_path = output_root / "CPU_CONTINUATION_PUBLISHED.json"
    candidate_audit_path = output_root / "final.audit.candidate.json"
    for path in (freeze_path, ready_path, published_path, candidate_audit_path):
        if path.exists():
            raise FileExistsError(path)

    canonical_all = rescue / "final.all-corpora.jsonl"
    canonical_all_report = canonical_all.with_suffix(canonical_all.suffix + ".report.json")
    final_by_corpus = rescue / "final_by_corpus"
    final_notice = final_by_corpus / "notice_and_comment.jsonl"
    final_notice_meta = final_notice.with_suffix(final_notice.suffix + ".meta.json")
    final_comments = final_by_corpus / "nc_public_comments.jsonl"
    final_comments_meta = final_comments.with_suffix(final_comments.suffix + ".meta.json")
    blind_match = rescue / "blind_audit_match"
    blind_abstention = rescue / "blind_audit_abstention"
    canonical_audit = rescue / "final.audit.json"
    canonical_targets = [
        canonical_all,
        canonical_all_report,
        final_by_corpus,
        blind_match,
        blind_abstention,
        canonical_audit,
    ]
    existing = [str(path) for path in canonical_targets if path.exists()]
    if existing:
        raise FileExistsError(f"canonical continuation targets are not fresh: {existing}")

    prewrapper = _json(prewrapper_audit_path)
    postwrapper = _json(postwrapper_audit_path)
    current_sealed = _json(current_sealed_audit_path)
    for name, report in (
        ("prewrapper", prewrapper),
        ("postwrapper", postwrapper),
        ("current", current_sealed),
    ):
        if (
            report.get("status")
            != "PASS_SEALED_GPU_ARTIFACTS_CONTENT_REVALIDATED"
            or report.get("complete") is not True
            or report.get("task") != TASK
            or int((report.get("authoritative_bank") or {}).get("metric_count", -1))
            != EXPECTED_BANK_COUNT
        ):
            raise ValueError(f"{name} sealed-artifact audit did not pass")
    if (
        prewrapper.get("artifacts") != postwrapper.get("artifacts")
        or prewrapper.get("trial_summaries") != postwrapper.get("trial_summaries")
        or prewrapper.get("lifecycle_metadata_defects")
        != postwrapper.get("lifecycle_metadata_defects")
    ):
        raise ValueError("sealed inference artifacts drifted across stale wrapper attempt")
    if (
        postwrapper.get("artifacts") != current_sealed.get("artifacts")
        or postwrapper.get("trial_summaries")
        != current_sealed.get("trial_summaries")
        or postwrapper.get("lifecycle_metadata_defects")
        != current_sealed.get("lifecycle_metadata_defects")
    ):
        raise ValueError("sealed inference artifacts drifted before CPU continuation")

    # Recompute once more at execution time, eliminating an audit-to-use race.
    live = audit_sealed(
        manifest_path=manifest,
        artifact_lock_path=artifact_lock_path,
        rescue_root=rescue,
        primary_paths=primary,
        adjudicator_selection=adjudicator_selection,
        verifier_selection=verifier_selection,
        verifier_policy=verifier_policy,
    )
    if live.get("artifacts") != current_sealed.get("artifacts"):
        raise ValueError("sealed inference artifacts changed after post-wrapper audit")

    wrapper_record = _json(failed_wrapper_record_path)
    if (
        wrapper_record.get("status")
        not in {
            "STALE_WRAPPER_FAILED_BEFORE_INFERENCE",
            "STALE_WRAPPER_TERMINATED_BEFORE_INFERENCE",
        }
        or wrapper_record.get("gpu_inference_started") is not False
    ):
        raise ValueError("failed-wrapper record does not prove no new inference")

    unresolved = rescue / "unresolved.jsonl"
    if not unresolved.is_file():
        raise FileNotFoundError(unresolved)
    reconciliation = _json(unresolved_reconciliation_path)
    frozen_ref = reconciliation.get("frozen_ledger") or {}
    manual_ref = reconciliation.get("manual_consensus") or {}
    if (
        reconciliation.get("schema_version")
        != "silver-match-v3-unresolved-ledger-reason-taxonomy-reconciliation-v1"
        or reconciliation.get("status")
        != "PASS_EXACT_UID_SET_REASON_TAXONOMY_ONLY"
        or reconciliation.get("exact_uid_set_equality") is not True
        or reconciliation.get("changed_fields") != ["unresolved_reason"]
        or Path(str(frozen_ref.get("path") or "")).resolve() != unresolved
        or frozen_ref.get("sha256") != sha256_file(unresolved)
        or Path(str(manual_ref.get("labels_path") or "")).resolve()
        != manual_labels
        or manual_ref.get("labels_sha256") != sha256_file(manual_labels)
        or Path(str(manual_ref.get("validation_path") or "")).resolve()
        != manual_validation
        or manual_ref.get("validation_sha256") != sha256_file(manual_validation)
    ):
        raise ValueError("unresolved reason-taxonomy reconciliation is invalid or unlinked")
    prior_failure = _json(prior_continuation_failure_path)
    failure_reconciliation = (
        (prior_failure.get("inputs") or {}).get("reconciliation") or {}
    )
    if (
        prior_failure.get("status")
        != "FAILED_CLOSED_BEFORE_CANONICAL_WRITE_LEDGER_REASON_TAXONOMY_ONLY"
        or prior_failure.get("gpu_inference_run") is not False
        or prior_failure.get("canonical_targets_present") != []
        or failure_reconciliation.get("sha256")
        != sha256_file(unresolved_reconciliation_path)
    ):
        raise ValueError("prior continuation failure is not safely frozen")

    manual_report = _json(manual_validation)
    manual_sha = sha256_file(manual_labels)
    if (
        manual_report.get("schema_version")
        != "silver-match-v3-full-bank-multi-vote-consensus-v1"
        or manual_report.get("complete") is not True
        or int((manual_report.get("unresolved") or {}).get("count", -1)) != 0
        or (manual_report.get("output") or {}).get("sha256") != manual_sha
    ):
        raise ValueError("manual consensus labels are incomplete, unresolved, or unlinked")

    implementations_root = Path(__file__).resolve().parent
    implementations = {
        name: {
            "path": str(implementations_root / name),
            "sha256": sha256_file(implementations_root / name),
        }
        for name in CPU_IMPLEMENTATIONS
    }
    if implementations != current_sealed.get("cpu_continuation_implementations"):
        raise ValueError("CPU continuation implementations changed after artifact audit")

    entrypoints = [
        *(implementations_root / name for name in CPU_IMPLEMENTATIONS),
        Path(__file__).resolve(),
    ]
    snapshot_freeze, dependency_inventory = _snapshot_dependencies(
        entrypoints=entrypoints,
        repo=repo,
        snapshot_root=output_root / "implementation_snapshot_v1",
    )

    def verify_implementations() -> None:
        _verify_dependency_snapshot(
            entrypoints=entrypoints,
            repo=repo,
            inventory=dependency_inventory,
        )

    manifest_payload = _json(manifest)
    locked_bank_path, _, _, _, _ = _validate_bank_binding(
        manifest, manifest_payload
    )
    frozen_lock_binding = _validate_artifact_lock(
        artifact_lock_path=artifact_lock_path,
        manifest_path=manifest,
        manifest=manifest_payload,
        bank_path=locked_bank_path,
    )
    if frozen_lock_binding != current_sealed.get("artifact_lock"):
        raise ValueError("artifact-lock binding differs from post-wrapper sealed audit")

    def verify_execution_environment() -> None:
        verify_implementations()
        current_manifest = _json(manifest)
        current_bank, _, _, _, _ = _validate_bank_binding(manifest, current_manifest)
        current_lock = _validate_artifact_lock(
            artifact_lock_path=artifact_lock_path,
            manifest_path=manifest,
            manifest=current_manifest,
            bank_path=current_bank,
        )
        if current_lock != frozen_lock_binding:
            raise ValueError("artifact-lock/bank/norm inputs changed at point of use")

    verify_execution_environment()

    freeze = {
        "schema_version": "silver-match-v3-notice-cpu-continuation-freeze-v1",
        "status": "FROZEN_BEFORE_CPU_CONTINUATION",
        "task": TASK,
        "gpu_inference_required": False,
        "authoritative_bank_metric_count": EXPECTED_BANK_COUNT,
        "canonical_targets_proven_absent": [str(path) for path in canonical_targets],
        "inputs": {
            "manifest": _ref(manifest),
            "artifact_lock": _ref(artifact_lock_path),
            "artifact_lock_task_binding": frozen_lock_binding,
            "primary": [_ref(path) for path in primary],
            "sealed_pre_wrapper_audit": _ref(prewrapper_audit_path),
            "sealed_post_wrapper_audit": _ref(postwrapper_audit_path),
            "current_sealed_artifact_audit": _ref(current_sealed_audit_path),
            "failed_wrapper_record": _ref(failed_wrapper_record_path),
            "unresolved_reason_taxonomy_reconciliation": _ref(
                unresolved_reconciliation_path
            ),
            "prior_continuation_failure": _ref(prior_continuation_failure_path),
            "manual_consensus_labels": _ref(manual_labels),
            "manual_consensus_validation": _ref(manual_validation),
            "frozen_unresolved": _ref(unresolved),
            "adjudicator_selection": _ref(adjudicator_selection),
            "verifier_selection": _ref(verifier_selection),
            "verifier_policy": _ref(verifier_policy),
        },
        "implementations": implementations,
        "recursive_python_dependency_snapshot": _ref(snapshot_freeze),
        "orchestrator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "python": _ref(python),
        "publication_rule": (
            "the structural final.audit release-candidate sentinel is hard-linked "
            "last after every CPU output and blind audit pack is prepared; the blind "
            "audit itself remains pending independent returned labels"
        ),
    }
    _write_new_json(freeze_path, freeze)

    base = [str(python), "-u", "-m"]
    commands: list[dict[str, Any]] = []
    merge = [
        *base,
        "scripts.tools.silver_match_v3.merge_rescue_decisions",
        "--manifest",
        str(manifest),
    ]
    for path in primary:
        merge += ["--primary", str(path)]
    merge += [
        "--finalist-candidates",
        str(rescue / "aggregate/match_finalists.jsonl"),
        "--finalist-adjudications",
        str(rescue / "finalists/adjudicate.original.jsonl"),
        "--finalist-order-check",
        str(rescue / "finalists/adjudicate.hashed.jsonl"),
        "--finalist-verification",
        str(rescue / "finalists/verify.strict-combined.jsonl"),
        "--no-match-audits",
        str(rescue / "aggregate/no_match_provisional.jsonl"),
        "--abstention-verifications",
        str(rescue / "typed_abstentions/verify.strict-combined.jsonl"),
        "--adjudicator-selection",
        str(adjudicator_selection),
        "--verifier-selection",
        str(verifier_selection),
        "--verifier-policy",
        str(verifier_policy),
        "--unresolved-output",
        str(unresolved),
        "--manual-unresolved-labels",
        str(manual_labels),
        "--manual-unresolved-validation",
        str(manual_validation),
        "--unresolved-reconciliation",
        str(unresolved_reconciliation_path),
        "--strict-production",
        "--output",
        str(canonical_all),
    ]
    commands.append({"name": "merge", "argv": merge})

    filter_notice = [
        *base,
        "scripts.tools.silver_match_v3.filter_labels",
        "--input",
        str(canonical_all),
        "--output",
        str(final_notice),
        "--where",
        "corpus=notice_and_comment",
    ]
    filter_comments = [
        *base,
        "scripts.tools.silver_match_v3.filter_labels",
        "--input",
        str(canonical_all),
        "--output",
        str(final_comments),
        "--where",
        "corpus=nc_public_comments",
    ]
    commands += [
        {"name": "filter_notice", "argv": filter_notice},
        {"name": "filter_comments", "argv": filter_comments},
    ]

    def blind_command(kind: str, root: Path, seed: str) -> list[str]:
        return [
            *base,
            "scripts.tools.silver_match_v3.prepare_final_decision_audit",
            "--manifest",
            str(manifest),
            "--final",
            str(final_notice),
            "--final",
            str(final_comments),
            "--output-root",
            str(root),
            "--global-n",
            "300",
            "--per-task-n",
            "200",
            "--seed",
            seed,
            "--sample-kind",
            kind,
        ]

    blind_match_command = blind_command("match", blind_match, "271828")
    blind_abstention_command = blind_command(
        "abstention", blind_abstention, "314159"
    )
    commands += [
        {"name": "blind_match", "argv": blind_match_command},
        {"name": "blind_abstention", "argv": blind_abstention_command},
    ]
    final_audit_command = [
        *base,
        "scripts.tools.silver_match_v3.audit_final_outputs",
        "--manifest",
        str(manifest),
        "--task",
        TASK,
        "--final",
        str(final_notice),
        "--final",
        str(final_comments),
        "--output",
        str(candidate_audit_path),
    ]
    commands.append({"name": "audit", "argv": final_audit_command})
    command_plan_path = output_root / "CPU_CONTINUATION_COMMANDS.json"
    _write_new_json(
        command_plan_path,
        {
            "schema_version": "silver-match-v3-notice-cpu-continuation-commands-v1",
            "freeze": _ref(freeze_path),
            "commands": commands,
        },
    )

    for item in commands:
        _run(
            item["argv"],
            cwd=repo,
            log=output_root / "logs" / f"{item['name']}.log",
            verify_implementations=verify_execution_environment,
        )

    final_audit = _json(candidate_audit_path)
    if (
        final_audit.get("complete") is not True
        or final_audit.get("scope", {}).get("tasks") != [TASK]
        or int(final_audit.get("corpora_audited", -1)) != 2
        or int(
            final_audit.get("by_task", {})
            .get(TASK, {})
            .get("matched_metric_coverage", {})
            .get("bank_count", -1)
        )
        != EXPECTED_BANK_COUNT
    ):
        raise ValueError("candidate final audit did not pass exact N&C scope/bank gates")
    for root in (blind_match, blind_abstention):
        _validate_blind_bank_binding(
            manifest_path=manifest,
            blind_root=root,
            task=TASK,
            expected_count=EXPECTED_BANK_COUNT,
        )

    ready = {
        "schema_version": "silver-match-v3-notice-cpu-continuation-ready-v1",
        "status": "READY_TO_PUBLISH_STRUCTURALLY_AUDITED_RELEASE_CANDIDATE",
        "freeze": _ref(freeze_path),
        "command_plan": _ref(command_plan_path),
        "sealed_artifact_audit": _ref(postwrapper_audit_path),
        "manual_consensus": {
            "labels": _ref(manual_labels),
            "validation": _ref(manual_validation),
        },
        "outputs": {
            "final_all": _ref(canonical_all),
            "final_all_report": _ref(canonical_all_report),
            "notice_and_comment": _ref(final_notice),
            "notice_and_comment_meta": _ref(final_notice_meta),
            "nc_public_comments": _ref(final_comments),
            "nc_public_comments_meta": _ref(final_comments_meta),
            "blind_match_report": _ref(blind_match / "sample_report.json"),
            "blind_abstention_report": _ref(blind_abstention / "sample_report.json"),
            "candidate_final_audit": _ref(candidate_audit_path),
        },
        "authoritative_bank_metric_count": EXPECTED_BANK_COUNT,
        "gpu_inference_run": False,
        "claim_boundary": (
            "final.audit proves structural completeness and the prepared blind packs "
            "are release-candidate audit instruments; independent hidden labels have "
            "not yet been returned or scored"
        ),
    }
    _write_new_json(ready_path, ready)

    # Same filesystem: hard-linking is atomic and refuses an existing target.
    verify_execution_environment()
    os.link(candidate_audit_path, canonical_audit)
    verify_execution_environment()
    published = {
        "schema_version": "silver-match-v3-notice-cpu-continuation-published-v1",
        "status": "PUBLISHED_STRUCTURAL_RELEASE_CANDIDATE_PENDING_INDEPENDENT_BLIND_LABELS",
        "ready": _ref(ready_path),
        "canonical_final_audit": _ref(canonical_audit),
        "candidate_final_audit": _ref(candidate_audit_path),
        "same_inode": candidate_audit_path.stat().st_ino == canonical_audit.stat().st_ino,
        "independent_blind_audit_passed": False,
        "claim_boundary": (
            "publication of the structural final.audit sentinel is not a passed "
            "independent final-production blind audit"
        ),
    }
    _write_new_json(published_path, published)
    print(json.dumps(published, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
