#!/usr/bin/env python3
"""Fail-closed inventory audit for an all-task silver-match release.

Component-specific audits prove that one retrieval or one final corpus is
valid.  They do not prove that every corpus in the canonical manifest is
present.  This auditor closes that gap: it verifies the frozen extraction
lock, one selected retriever and one complete candidate audit per corpus, one
canonical final file per corpus, and one passed production blind audit per
task.  Missing evidence is reported explicitly and makes the CLI exit nonzero
unless ``--allow-incomplete`` is used for a progress snapshot.

The caller supplies evidence paths explicitly.  Discovery by filename is
intentionally avoided because stale attempts and rejected variants commonly
live beside selected artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .audit_final_outputs import audit_outputs
from .common import sha256_file
from .freeze_task_final_risk_release import verify_task_final_risk_release


SCHEMA = "silver-match-v3-alltask-release-coverage-audit-v2"


def _resolve(value: str | Path, anchor: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (anchor.parent / path).resolve()


def _parse_bindings(values: Iterable[str], *, name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key.strip() or not raw_path.strip():
            raise ValueError(f"invalid {name} binding {value!r}; expected KEY=PATH")
        key = key.strip()
        if key in result:
            raise ValueError(f"duplicate {name} binding for {key}")
        result[key] = Path(raw_path).resolve()
    return result


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _verify_extractions(
    manifest_path: Path, artifact_lock_path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    lock = json.loads(artifact_lock_path.read_text(encoding="utf-8"))
    expected_manifest_sha = str((lock.get("manifest") or {}).get("sha256") or "")
    observed_manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha != observed_manifest_sha:
        raise ValueError("artifact lock does not pin the supplied manifest")
    lock_norms = lock.get("norms") or {}
    manifest_corpora = manifest.get("corpora") or {}
    if set(lock_norms) != set(manifest_corpora):
        raise ValueError("artifact-lock/manifest corpus set mismatch")

    corpora: dict[str, Any] = {}
    for corpus, meta in sorted(manifest_corpora.items()):
        frozen = lock_norms[corpus]
        path = _resolve(str(frozen["path"]), artifact_lock_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        actual_count = _line_count(path)
        expected_path = _resolve(str(meta["path"]), manifest_path)
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"canonical extraction path mismatch for {corpus}")
        if actual_sha != str(frozen["sha256"]):
            raise ValueError(f"canonical extraction hash mismatch for {corpus}")
        if actual_count != int(frozen["count"]) or actual_count != int(meta["count"]):
            raise ValueError(f"canonical extraction count mismatch for {corpus}")
        corpora[corpus] = {
            "task": str(meta["task"]),
            "path": str(path),
            "count": actual_count,
            "sha256": actual_sha,
        }
    return {
        "complete": True,
        "manifest_sha256": observed_manifest_sha,
        "artifact_lock_sha256": sha256_file(artifact_lock_path),
        "count": sum(row["count"] for row in corpora.values()),
        "corpora": corpora,
    }


def _verify_selections(
    manifest: dict[str, Any], bindings: dict[str, Path]
) -> dict[str, Any]:
    expected = set(manifest.get("banks") or {})
    unknown = set(bindings) - expected
    if unknown:
        raise ValueError(f"selection bindings name unknown tasks: {sorted(unknown)}")
    records: dict[str, Any] = {}
    for task, path in sorted(bindings.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task") != task:
            raise ValueError(f"selection task mismatch for {task}: {path}")
        chosen = payload.get("chosen") or {}
        selection = payload.get("selection") or {}
        chosen_kind = chosen.get("kind") or selection.get("chosen_kind")
        chosen_name = chosen.get("name") or selection.get("chosen_name")
        if not chosen_kind or not chosen_name:
            raise ValueError(f"selection lacks an explicit chosen retriever for {task}")
        if payload.get("selection_split") != "external_dev_only":
            raise ValueError(f"selection is not external-dev-only for {task}")
        records[task] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "chosen_kind": str(chosen_kind),
            "chosen_name": str(chosen_name),
            "fusion_path": chosen.get("fusion_report"),
            "fusion_sha256": chosen.get("fusion_report_sha256"),
        }
    missing = sorted(expected - set(records))
    return {
        "complete": not missing,
        "complete_tasks": len(records),
        "expected_tasks": len(expected),
        "missing_tasks": missing,
        "tasks": records,
    }


def _verify_candidates(
    manifest_path: Path,
    manifest: dict[str, Any],
    audit_paths: Iterable[Path],
) -> dict[str, Any]:
    expected = manifest.get("corpora") or {}
    manifest_sha = sha256_file(manifest_path)
    records: dict[str, Any] = {}
    for path in audit_paths:
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        corpus = str(payload.get("corpus") or "")
        if corpus not in expected:
            raise ValueError(f"candidate audit has unknown corpus {corpus!r}: {path}")
        if corpus in records:
            raise ValueError(f"multiple candidate audits supplied for {corpus}")
        meta = expected[corpus]
        task = str(meta["task"])
        if (
            payload.get("complete") is not True
            or payload.get("task") != task
            or int(payload.get("expected_count", -1)) != int(meta["count"])
            or int(payload.get("observed_count", -1)) != int(meta["count"])
            or payload.get("manifest_sha256") != manifest_sha
            or payload.get("bank_source_sha256")
            != manifest["banks"][task]["source_sha256"]
        ):
            raise ValueError(f"candidate audit is not canonical and complete: {path}")
        inputs = payload.get("candidate_inputs") or {}
        if not inputs:
            raise ValueError(f"candidate audit has no inputs: {path}")
        verified_inputs: dict[str, Any] = {}
        for raw_candidate, identity in sorted(inputs.items()):
            candidate = Path(raw_candidate)
            if not candidate.is_absolute():
                candidate = (path.parent / candidate).resolve()
            if not candidate.is_file() or sha256_file(candidate) != identity.get(
                "sha256"
            ):
                raise ValueError(f"candidate artifact differs from audit: {candidate}")
            if _line_count(candidate) != int(identity.get("count", -1)):
                raise ValueError(f"candidate row count differs from audit: {candidate}")
            meta_path = Path(str(identity.get("meta") or ""))
            if not meta_path.is_absolute():
                meta_path = (path.parent / meta_path).resolve()
            if not meta_path.is_file() or sha256_file(meta_path) != identity.get(
                "meta_sha256"
            ):
                raise ValueError(f"candidate metadata differs from audit: {meta_path}")
            verified_inputs[str(candidate)] = {
                "count": int(identity["count"]),
                "sha256": str(identity["sha256"]),
                "meta": str(meta_path),
                "meta_sha256": str(identity["meta_sha256"]),
            }
        records[corpus] = {
            "task": task,
            "count": int(meta["count"]),
            "expected_k": int(payload["expected_k"]),
            "materialized_k": int(payload["materialized_k"]),
            "audit": str(path),
            "audit_sha256": sha256_file(path),
            "inputs": verified_inputs,
        }
    missing = sorted(set(expected) - set(records))
    return {
        "complete": not missing,
        "covered_count": sum(row["count"] for row in records.values()),
        "expected_count": sum(int(row["count"]) for row in expected.values()),
        "complete_corpora": len(records),
        "expected_corpora": len(expected),
        "missing_corpora": missing,
        "corpora": records,
    }


def _verify_finals(
    manifest_path: Path,
    manifest: dict[str, Any],
    bindings: dict[str, Path],
) -> dict[str, Any]:
    expected = manifest.get("corpora") or {}
    unknown = set(bindings) - set(expected)
    if unknown:
        raise ValueError(f"final bindings name unknown corpora: {sorted(unknown)}")
    records: dict[str, Any] = {}
    for corpus, path in sorted(bindings.items()):
        report = audit_outputs(manifest_path, [path], corpora={corpus})
        records[corpus] = {
            "task": str(expected[corpus]["task"]),
            "count": int(expected[corpus]["count"]),
            "path": str(path),
            "sha256": sha256_file(path),
            "decision_counts": report["by_corpus"][corpus]["decision_counts"],
        }
    missing = sorted(set(expected) - set(records))
    return {
        "complete": not missing,
        "covered_count": sum(row["count"] for row in records.values()),
        "expected_count": sum(int(row["count"]) for row in expected.values()),
        "complete_corpora": len(records),
        "expected_corpora": len(expected),
        "missing_corpora": missing,
        "corpora": records,
    }


def _verify_blind_audits(
    manifest_path: Path,
    manifest: dict[str, Any],
    bindings: dict[str, Path],
    final_bindings: dict[str, Path],
) -> dict[str, Any]:
    expected = set(manifest.get("banks") or {})
    unknown = set(bindings) - expected
    if unknown:
        raise ValueError(f"blind-audit bindings name unknown tasks: {sorted(unknown)}")
    records: dict[str, Any] = {}
    for task, path in sorted(bindings.items()):
        task_corpora = {
            corpus
            for corpus, meta in (manifest.get("corpora") or {}).items()
            if str(meta.get("task") or "") == task
        }
        if not task_corpora.issubset(final_bindings):
            raise ValueError(
                f"blind audit supplied before all task finals: {task}; "
                f"missing={sorted(task_corpora - set(final_bindings))}"
            )
        recomputed = verify_task_final_risk_release(
            path,
            expected_manifest_path=manifest_path,
            expected_final_paths=[final_bindings[corpus] for corpus in task_corpora],
        )
        if (
            recomputed.get("task") != task
            or recomputed.get("production_final_blind_audit") is not True
            or recomputed.get("status") != "PASS"
            or recomputed.get("complete") is not True
            or not all((recomputed.get("gates") or {}).values())
        ):
            raise ValueError(f"non-passing recomputed production blind audit: {path}")
        records[task] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "status": "PASS",
            "gates": recomputed["gates"],
            "match_statistics": recomputed["match_audit"]["statistics"],
            "abstention_statistics": recomputed["abstention_audit"]["statistics"],
        }
    missing = sorted(expected - set(records))
    return {
        "complete": not missing,
        "complete_tasks": len(records),
        "expected_tasks": len(expected),
        "missing_tasks": missing,
        "tasks": records,
    }


def audit_alltask_coverage(
    *,
    manifest_path: Path,
    artifact_lock_path: Path,
    selection_bindings: dict[str, Path],
    candidate_audits: Iterable[Path],
    final_bindings: dict[str, Path],
    blind_audit_bindings: dict[str, Path],
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    artifact_lock_path = artifact_lock_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        int(manifest.get("total_norms", -1))
        != sum(int(row["count"]) for row in (manifest.get("corpora") or {}).values())
        or int(manifest.get("total_corpora", -1)) != len(manifest.get("corpora") or {})
        or int(manifest.get("total_tasks", -1)) != len(manifest.get("banks") or {})
    ):
        raise ValueError("manifest aggregate counts are inconsistent")
    extractions = _verify_extractions(manifest_path, artifact_lock_path, manifest)
    selections = _verify_selections(manifest, selection_bindings)
    candidates = _verify_candidates(manifest_path, manifest, candidate_audits)
    finals = _verify_finals(manifest_path, manifest, final_bindings)
    blind = _verify_blind_audits(
        manifest_path, manifest, blind_audit_bindings, final_bindings
    )
    gates = {
        "extractions": bool(extractions["complete"]),
        "retriever_selections": bool(selections["complete"]),
        "candidate_retrieval": bool(candidates["complete"]),
        "canonical_final_outputs": bool(finals["complete"]),
        "production_final_blind_audits": bool(blind["complete"]),
    }
    return {
        "schema_version": SCHEMA,
        "complete": all(gates.values()),
        "gates": gates,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "total_norms": int(manifest["total_norms"]),
            "total_corpora": int(manifest["total_corpora"]),
            "total_tasks": int(manifest["total_tasks"]),
        },
        "extractions": extractions,
        "retriever_selections": selections,
        "candidate_retrieval": candidates,
        "canonical_final_outputs": finals,
        "production_final_blind_audits": blind,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-lock", required=True)
    parser.add_argument(
        "--selection-record", action="append", default=[], metavar="TASK=PATH"
    )
    parser.add_argument("--candidate-audit", action="append", default=[])
    parser.add_argument(
        "--final-output", action="append", default=[], metavar="CORPUS=PATH"
    )
    parser.add_argument(
        "--blind-audit", action="append", default=[], metavar="TASK=PATH"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = audit_alltask_coverage(
        manifest_path=Path(args.manifest),
        artifact_lock_path=Path(args.artifact_lock),
        selection_bindings=_parse_bindings(
            args.selection_record, name="selection-record"
        ),
        candidate_audits=[Path(path) for path in args.candidate_audit],
        final_bindings=_parse_bindings(args.final_output, name="final-output"),
        blind_audit_bindings=_parse_bindings(args.blind_audit, name="blind-audit"),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"complete": report["complete"], "output": str(output)}, sort_keys=True
        )
    )
    if not report["complete"] and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
