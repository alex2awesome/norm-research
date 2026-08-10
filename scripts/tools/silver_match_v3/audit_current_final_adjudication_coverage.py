#!/usr/bin/env python3
"""Audit the current final-adjudication boundary for every task and corpus.

This inventory deliberately distinguishes retrieval coverage from final-label
coverage. A spec must classify every canonical task either as having no bound
production final or as an explicitly hash-pinned partial component partition.
The latter is rejoined against canonical UIDs and must account for every row as
MATCH, a separately named typed non-match, extraction NOISE, or unresolved. Partial components
are never promoted to a canonical final by this audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import sha256_file


SCHEMA = "silver-match-v3-current-final-adjudication-coverage-v2"
NO_FINAL = "no_canonical_final"
PARTIAL = "partial_component_partition_v1"
NOISE_DECISION = "NOISE"
TYPED_NONMATCH_DECISIONS = {
    "GENERIC_VERDICT",
    "MATCH_FAMILY_ONLY",
    "NO_CANDIDATE_FITS",
    "NO_EXPLICIT_CRITERION",
}
DECISION_COUNT_FIELDS = {
    "GENERIC_VERDICT": "generic_verdict_count",
    "MATCH_FAMILY_ONLY": "match_family_only_count",
    "NO_CANDIDATE_FITS": "no_candidate_fits_count",
    "NO_EXPLICIT_CRITERION": "no_explicit_criterion_count",
    NOISE_DECISION: "noise_count",
}
ROLLUP_COUNT_FIELDS = (
    "expected_count",
    "resolved_count",
    "match_count",
    "typed_nonmatch_count",
    *DECISION_COUNT_FIELDS.values(),
    "unresolved_count",
)


def _resolve(value: str | Path, anchor: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _artifact(entry: dict[str, Any], anchor: Path) -> dict[str, Any]:
    path = _resolve(entry["path"], anchor)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != entry.get("sha256"):
        raise ValueError(f"artifact hash mismatch: {path}")
    return {"path": str(path), "sha256": observed}


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{number}") from exc


def _canonical_task(
    manifest: dict[str, Any], manifest_path: Path, task: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    corpora = {
        corpus: meta
        for corpus, meta in (manifest.get("corpora") or {}).items()
        if meta.get("task") == task
    }
    canonical: dict[str, str] = {}
    for corpus, meta in corpora.items():
        path = _resolve(meta["path"], manifest_path)
        count = 0
        for row in _rows(path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in canonical:
                raise ValueError(f"missing/duplicate canonical UID for {corpus}: {uid!r}")
            if row.get("corpus") not in (None, corpus) or row.get("task") not in (
                None,
                task,
            ):
                raise ValueError(f"canonical routing mismatch: {uid}")
            canonical[uid] = corpus
            count += 1
        if count != int(meta["count"]):
            raise ValueError(f"canonical count mismatch for {corpus}")
    return corpora, canonical


def _load_component(
    entry: dict[str, Any], *, anchor: Path, canonical: dict[str, str], task: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    identity = _artifact(entry, anchor)
    path = Path(identity["path"])
    rows: dict[str, dict[str, Any]] = {}
    for row in _rows(path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in rows:
            raise ValueError(f"missing/duplicate component UID in {path}: {uid!r}")
        if uid not in canonical:
            raise ValueError(f"component UID outside canonical task {task}: {uid}")
        if row.get("task") != task or row.get("corpus") != canonical[uid]:
            raise ValueError(f"component routing mismatch: {uid}")
        rows[uid] = row
    report_entry = entry.get("report")
    report_identity = None
    if report_entry:
        report_identity = _artifact(report_entry, anchor)
        report = json.loads(Path(report_identity["path"]).read_text(encoding="utf-8"))
        if report.get("output_sha256") != identity["sha256"]:
            raise ValueError(f"component report does not bind output: {path}")
    return rows, {
        **identity,
        "count": len(rows),
        "report": report_identity,
    }


def _audit_partial(
    *,
    task: str,
    mode: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    spec_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    corpora, canonical = _canonical_task(manifest, manifest_path, task)
    bank_sha = str(manifest["banks"][task]["source_sha256"])
    artifacts: dict[str, Any] = {"primary_finals": []}
    primary_match: dict[str, dict[str, Any]] = {}
    primary_seen: set[str] = set()
    bound_corpora: set[str] = set()
    for entry in mode.get("primary_finals") or []:
        corpus = str(entry.get("corpus") or "")
        if corpus not in corpora or corpus in bound_corpora:
            raise ValueError(f"invalid/duplicate primary-final corpus: {corpus}")
        rows, identity = _load_component(
            entry, anchor=spec_path, canonical=canonical, task=task
        )
        if set(rows) != {uid for uid, value in canonical.items() if value == corpus}:
            raise ValueError(f"primary final is not complete for {corpus}")
        if any(row.get("corpus") != corpus for row in rows.values()):
            raise ValueError(f"primary final crosses corpora: {corpus}")
        primary_seen.update(rows)
        bound_corpora.add(corpus)
        for uid, row in rows.items():
            if row.get("bank_source_sha256") != bank_sha:
                raise ValueError(f"primary-final bank mismatch: {uid}")
            if row.get("decision") == "MATCH":
                if not row.get("metric_id"):
                    raise ValueError(f"primary MATCH lacks metric_id: {uid}")
                primary_match[uid] = row
        artifacts["primary_finals"].append({"corpus": corpus, **identity})
    if bound_corpora != set(corpora) or primary_seen != set(canonical):
        raise ValueError("partial mode must bind a full primary artifact for every task corpus")

    rescue_rows, artifacts["rescue_matches"] = _load_component(
        mode["rescue_matches"], anchor=spec_path, canonical=canonical, task=task
    )
    typed_rows, artifacts["typed_abstentions"] = _load_component(
        mode["typed_abstentions"], anchor=spec_path, canonical=canonical, task=task
    )
    unresolved_rows, artifacts["unresolved"] = _load_component(
        mode["unresolved"], anchor=spec_path, canonical=canonical, task=task
    )
    for rows in (rescue_rows, typed_rows, unresolved_rows):
        for uid, row in rows.items():
            observed_bank = row.get("bank_source_sha256") or row.get(
                "candidate_bank_source_sha256"
            )
            if observed_bank != bank_sha:
                raise ValueError(f"partial-component bank mismatch: {uid}")

    raw_rescue_match = {
        uid: row
        for uid, row in rescue_rows.items()
        if row.get("strict_two_order_acceptance") is True
        and row.get("decision") == "CONFIRM_MATCH"
        and row.get("metric_id")
    }
    strict_typed = {
        uid: row
        for uid, row in typed_rows.items()
        if row.get("strict_two_order_abstention") is True
    }
    unresolved = set(unresolved_rows)
    rescue_match = set(raw_rescue_match) - unresolved
    match = set(primary_match) | rescue_match
    typed = set(strict_typed) - unresolved
    if set(primary_match) & rescue_match or match & typed or match & unresolved or typed & unresolved:
        raise ValueError("trusted partial-component categories overlap")

    typed_decisions = {
        uid: str(strict_typed[uid].get("confirmed_decision") or "") for uid in typed
    }
    unknown = set(typed_decisions.values()) - TYPED_NONMATCH_DECISIONS - {
        NOISE_DECISION
    }
    if unknown:
        raise ValueError(f"unclassified typed decisions: {sorted(unknown)}")
    noise = {
        uid for uid, decision in typed_decisions.items() if decision == NOISE_DECISION
    }
    typed_nonmatch = typed - noise
    partition = match | noise | typed_nonmatch | unresolved
    if partition != set(canonical):
        missing = set(canonical) - partition
        raise ValueError(f"partial components do not partition canonical task; missing={len(missing)}")

    corpus_rows: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted(corpora.items()):
        universe = {uid for uid, value in canonical.items() if value == corpus}
        current_match = match & universe
        current_noise = noise & universe
        current_typed_nonmatch = typed_nonmatch & universe
        current_unresolved = unresolved & universe
        decision_counts = Counter(typed_decisions[uid] for uid in typed & universe)
        reason_counts = Counter(
            str(unresolved_rows[uid].get("unresolved_reason") or "")
            for uid in current_unresolved
        )
        corpus_rows[corpus] = {
            "task": task,
            "expected_count": int(meta["count"]),
            "resolved_count": len(
                current_match | current_noise | current_typed_nonmatch
            ),
            "match_count": len(current_match),
            "typed_nonmatch_count": len(current_typed_nonmatch),
            **{
                field: int(decision_counts.get(decision, 0))
                for decision, field in DECISION_COUNT_FIELDS.items()
            },
            "unresolved_count": len(current_unresolved),
            "typed_decision_counts": dict(sorted(decision_counts.items())),
            "unresolved_reason_counts": dict(sorted(reason_counts.items())),
            "artifact_status": "PARTIAL_COMPONENTS_NOT_CANONICAL_FINAL",
            "validation_status": "EXACT_COMPONENT_PARTITION_PASS",
            "canonical_final_ready": False,
            "remaining_action": mode["remaining_action"],
        }
    totals = Counter()
    for row in corpus_rows.values():
        for key in ROLLUP_COUNT_FIELDS:
            totals[key] += int(row[key])
    task_decision_counts = Counter()
    for row in corpus_rows.values():
        task_decision_counts.update(row["typed_decision_counts"])
    task_row = {
        "mode": PARTIAL,
        **dict(totals),
        "corpora": sorted(corpus_rows),
        "artifact_status": "PARTIAL_COMPONENTS_NOT_CANONICAL_FINAL",
        "validation_status": "EXACT_COMPONENT_PARTITION_PASS",
        "canonical_final_ready": False,
        "raw_rescue_match_count": len(raw_rescue_match),
        "raw_rescue_match_overridden_by_unresolved": len(set(raw_rescue_match) & unresolved),
        "typed_decision_counts": dict(sorted(task_decision_counts.items())),
        "artifacts": artifacts,
        "remaining_action": mode["remaining_action"],
    }
    return task_row, corpus_rows


def audit_current_coverage(spec_path: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    request_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    effective_spec_path = spec_path
    effective_spec_identity = None
    if request_spec.get("base_spec"):
        effective_spec_identity = _artifact(request_spec["base_spec"], spec_path)
        effective_spec_path = Path(effective_spec_identity["path"])
        spec = json.loads(effective_spec_path.read_text(encoding="utf-8"))
    else:
        spec = request_spec
    manifest_identity = _artifact(spec["manifest"], effective_spec_path)
    manifest_path = Path(manifest_identity["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_tasks = set(manifest.get("banks") or {})
    modes = spec.get("tasks") or {}
    if set(modes) != expected_tasks:
        raise ValueError(
            f"final-adjudication spec task mismatch: "
            f"missing={sorted(expected_tasks - set(modes))} "
            f"unknown={sorted(set(modes) - expected_tasks)}"
        )

    tasks: dict[str, Any] = {}
    corpora: dict[str, Any] = {}
    for task, mode in sorted(modes.items()):
        kind = mode.get("mode")
        task_corpora = {
            corpus: meta
            for corpus, meta in (manifest.get("corpora") or {}).items()
            if meta.get("task") == task
        }
        if kind == PARTIAL:
            task_row, corpus_rows = _audit_partial(
                task=task,
                mode=mode,
                manifest=manifest,
                manifest_path=manifest_path,
                spec_path=effective_spec_path,
            )
        elif kind == NO_FINAL:
            if mode.get("artifacts"):
                raise ValueError(f"no-final task unexpectedly binds artifacts: {task}")
            supporting_evidence = [
                _artifact(entry, effective_spec_path)
                for entry in (mode.get("supporting_evidence") or [])
            ]
            corpus_rows = {
                corpus: {
                    "task": task,
                    "expected_count": int(meta["count"]),
                    "resolved_count": 0,
                    "match_count": 0,
                    "typed_nonmatch_count": 0,
                    "generic_verdict_count": 0,
                    "match_family_only_count": 0,
                    "no_candidate_fits_count": 0,
                    "no_explicit_criterion_count": 0,
                    "noise_count": 0,
                    "unresolved_count": int(meta["count"]),
                    "typed_decision_counts": {},
                    "unresolved_reason_counts": {
                        "no_canonical_final_artifact_bound": int(meta["count"])
                    },
                    "artifact_status": "NO_CANONICAL_FINAL_ARTIFACT_BOUND",
                    "validation_status": "NOT_RUN",
                    "canonical_final_ready": False,
                    "remaining_action": mode["remaining_action"],
                }
                for corpus, meta in sorted(task_corpora.items())
            }
            expected_count = sum(row["expected_count"] for row in corpus_rows.values())
            task_row = {
                "mode": NO_FINAL,
                "expected_count": expected_count,
                "resolved_count": 0,
                "match_count": 0,
                "typed_nonmatch_count": 0,
                "generic_verdict_count": 0,
                "match_family_only_count": 0,
                "no_candidate_fits_count": 0,
                "no_explicit_criterion_count": 0,
                "noise_count": 0,
                "unresolved_count": expected_count,
                "corpora": sorted(corpus_rows),
                "artifact_status": "NO_CANONICAL_FINAL_ARTIFACT_BOUND",
                "validation_status": "NOT_RUN",
                "canonical_final_ready": False,
                "typed_decision_counts": {},
                "artifacts": [],
                "supporting_evidence": supporting_evidence,
                "upstream_status": mode.get("upstream_status") or {},
                "remaining_action": mode["remaining_action"],
            }
        else:
            raise ValueError(f"unsupported final-adjudication mode for {task}: {kind}")
        tasks[task] = task_row
        corpora.update(corpus_rows)

    if set(corpora) != set(manifest.get("corpora") or {}):
        raise ValueError("final-adjudication ledger does not cover every manifest corpus")
    summary = Counter()
    for row in corpora.values():
        for key in ROLLUP_COUNT_FIELDS:
            summary[key] += int(row[key])
    if summary["expected_count"] != int(manifest["total_norms"]):
        raise ValueError("final-adjudication ledger total differs from manifest")
    summary_decision_counts = Counter()
    for row in corpora.values():
        summary_decision_counts.update(row["typed_decision_counts"])
    predecessor = request_spec.get("supersedes")
    supersedes = None
    if predecessor:
        supersedes = {
            **_artifact(predecessor, spec_path),
            "status": predecessor.get("status"),
            "reason": predecessor.get("reason"),
        }
        if supersedes["status"] != "SUPERSEDED_TERMINOLOGY_ERROR":
            raise ValueError("superseded ledger must carry terminology-error status")
    return {
        "schema_version": SCHEMA,
        "inventory_complete": True,
        "release_complete": all(row["canonical_final_ready"] for row in corpora.values()),
        "manifest": manifest_identity,
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "base_spec": effective_spec_identity,
        "supersedes": supersedes,
        "summary": {
            **dict(summary),
            "typed_decision_counts": dict(sorted(summary_decision_counts.items())),
            "expected_tasks": len(expected_tasks),
            "expected_corpora": len(corpora),
            "canonical_final_ready_tasks": sum(
                bool(row["canonical_final_ready"]) for row in tasks.values()
            ),
            "canonical_final_ready_corpora": sum(
                bool(row["canonical_final_ready"]) for row in corpora.values()
            ),
        },
        "tasks": tasks,
        "corpora": dict(sorted(corpora.items())),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_current_coverage(Path(args.spec))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output), **report["summary"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
