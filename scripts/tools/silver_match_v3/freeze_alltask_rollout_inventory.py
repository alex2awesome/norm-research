#!/usr/bin/env python3
"""Freeze the exact canonical scope for the eight-task silver-match rollout.

This is a scope freezer, not a release-readiness audit.  It reads only the
manifest supplied by the caller and the bank/norm artifacts referenced by that
manifest.  It does not discover artifacts by filename and it never treats a
historical local output as evidence that a task is ready.

The resulting inventory is the handoff contract for a uniform all-task run:
every canonical norm must receive exactly one final typed decision against the
bank frozen here.  Candidate, truth, production, rescue, final-output, and
blind-risk evidence remain explicitly required and unevaluated until their
dedicated fail-closed auditors validate them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import normalize_space, read_jsonl, sha256_file


SCHEMA = "silver-match-v3-alltask-rollout-inventory-v1"

# User-prioritized execution order.  The middle tasks retain a deterministic
# order; Humor is intentionally first and Notice & Comment intentionally last.
ROLLOUT_TASK_ORDER = (
    "humor",
    "code-review",
    "creative-writing",
    "legal-outcome-prediction",
    "math-stackexchange",
    "peer-review",
    "press-releases",
    "notice-and-comment",
)

FINAL_DECISION_TAXONOMY = (
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
    "UNSTABLE_MATCH",
    "INVALID_OUTPUT",
)

UID_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_json_no_duplicate_keys(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )


def _resolve(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (manifest_path.parent / path).resolve()
    )


def _identity_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"manifest {name} must be a nonempty object")
    return value


def _validate_manifest_shape(
    manifest: dict[str, Any], manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if manifest.get("source_mode") != "canonical":
        raise ValueError("rollout requires a canonical source_mode manifest")
    banks = _require_object(manifest.get("banks"), name="banks")
    corpora = _require_object(manifest.get("corpora"), name="corpora")
    routing = _require_object(manifest.get("routing"), name="routing")

    expected_tasks = set(ROLLOUT_TASK_ORDER)
    observed_tasks = set(banks)
    if observed_tasks != expected_tasks:
        raise ValueError(
            "manifest task set mismatch: "
            f"missing={sorted(expected_tasks - observed_tasks)} "
            f"foreign={sorted(observed_tasks - expected_tasks)}"
        )
    if set(routing) != set(corpora):
        raise ValueError(
            "manifest routing/corpus set mismatch: "
            f"missing={sorted(set(corpora) - set(routing))} "
            f"foreign={sorted(set(routing) - set(corpora))}"
        )

    task_corpus_counts: Counter[str] = Counter()
    seen_artifact_paths: dict[Path, str] = {}
    for task, meta in banks.items():
        if not isinstance(meta, dict):
            raise ValueError(f"bank metadata must be an object: {task}")
        raw_path = normalize_space(meta.get("path"))
        if not raw_path:
            raise ValueError(f"bank path missing: {task}")
        path = _resolve(raw_path, manifest_path)
        prior = seen_artifact_paths.setdefault(path, f"bank:{task}")
        if prior != f"bank:{task}":
            raise ValueError(
                f"duplicate canonical artifact path: {prior} and bank:{task}"
            )

    for corpus, meta in corpora.items():
        if not corpus or not isinstance(meta, dict):
            raise ValueError(f"invalid corpus metadata: {corpus!r}")
        task = normalize_space(meta.get("task"))
        routed_task = normalize_space(routing.get(corpus))
        if task not in expected_tasks:
            raise ValueError(f"foreign task routing for corpus {corpus}: {task!r}")
        if routed_task != task:
            raise ValueError(
                f"manifest routing mismatch for corpus {corpus}: "
                f"routing={routed_task!r} metadata={task!r}"
            )
        if meta.get("coverage_complete") is not True:
            raise ValueError(f"canonical corpus coverage is incomplete: {corpus}")
        if meta.get("missing_optional_segments") not in (None, []):
            raise ValueError(
                f"canonical corpus has missing optional segments: {corpus}"
            )
        raw_path = normalize_space(meta.get("path"))
        if not raw_path:
            raise ValueError(f"norm path missing: {corpus}")
        path = _resolve(raw_path, manifest_path)
        label = f"corpus:{corpus}"
        prior = seen_artifact_paths.setdefault(path, label)
        if prior != label:
            raise ValueError(f"duplicate canonical artifact path: {prior} and {label}")
        task_corpus_counts[task] += 1

    tasks_without_corpora = sorted(expected_tasks - set(task_corpus_counts))
    if tasks_without_corpora:
        raise ValueError(f"tasks lack canonical corpora: {tasks_without_corpora}")

    aliases = manifest.get("aliases") or {}
    if not isinstance(aliases, dict):
        raise ValueError("manifest aliases must be an object")
    for alias, target in aliases.items():
        if alias in corpora:
            raise ValueError(f"alias duplicates canonical corpus: {alias}")
        if target not in corpora:
            raise ValueError(f"alias targets foreign corpus: {alias} -> {target}")

    if (
        int(manifest.get("total_tasks", -1)) != len(banks)
        or int(manifest.get("total_corpora", -1)) != len(corpora)
        or int(manifest.get("total_norms", -1))
        != sum(int(meta.get("count", -1)) for meta in corpora.values())
    ):
        raise ValueError("manifest aggregate counts are inconsistent")
    return banks, corpora, {str(key): str(value) for key, value in routing.items()}


def _freeze_bank(
    *, task: str, metadata: dict[str, Any], manifest_path: Path
) -> tuple[dict[str, Any], set[str]]:
    path = _resolve(str(metadata["path"]), manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load_json_no_duplicate_keys(path)
    if not isinstance(payload, dict) or payload.get("task") != task:
        raise ValueError(f"bank payload task mismatch: {task}: {path}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"bank has no metrics: {task}: {path}")
    if len(metrics) != int(metadata.get("count", -1)):
        raise ValueError(f"bank count mismatch: {task}: {path}")
    for index, row in enumerate(metrics):
        if not isinstance(row, dict):
            raise ValueError(f"non-object bank metric: {task}:{index}")
    metric_ids = [normalize_space(row.get("metric_id")) for row in metrics]
    if not all(metric_ids) or len(metric_ids) != len(set(metric_ids)):
        raise ValueError(f"bank has missing/duplicate metric IDs: {task}: {path}")
    for index, row in enumerate(metrics):
        if row.get("task") not in (None, task):
            raise ValueError(f"bank metric task mismatch: {task}:{metric_ids[index]}")
        if row.get("metric_index") not in (None, index):
            raise ValueError(f"bank metric index mismatch: {task}:{metric_ids[index]}")
    manifest_source_sha = normalize_space(metadata.get("source_sha256"))
    payload_source_sha = normalize_space(payload.get("source_sha256"))
    if not manifest_source_sha or payload_source_sha != manifest_source_sha:
        raise ValueError(f"bank source provenance mismatch: {task}: {path}")
    return (
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "metric_count": len(metrics),
            "metric_ids_sha256": _identity_sha256(metric_ids),
            "source_path": str(
                metadata.get("source_path") or payload.get("source_path") or ""
            ),
            "source_sha256": manifest_source_sha,
        },
        set(metric_ids),
    )


def _freeze_corpus(
    *,
    corpus: str,
    task: str,
    metadata: dict[str, Any],
    manifest_path: Path,
    globally_seen_uids: set[str],
) -> dict[str, Any]:
    path = _resolve(str(metadata["path"]), manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    count = 0
    ordered_uids: list[str] = []
    for row in read_jsonl(path):
        uid = normalize_space(row.get("norm_uid"))
        if not UID_RE.fullmatch(uid):
            raise ValueError(f"malformed norm_uid in {corpus}: {uid!r}")
        if uid in globally_seen_uids:
            raise ValueError(f"duplicate global norm_uid: {uid}")
        globally_seen_uids.add(uid)
        if row.get("corpus") != corpus or row.get("task") != task:
            raise ValueError(f"foreign corpus routing in canonical norm {uid}")
        if not normalize_space(row.get("norm")):
            raise ValueError(f"empty canonical norm: {uid}")
        ordered_uids.append(uid)
        count += 1
    if count != int(metadata.get("count", -1)):
        raise ValueError(
            f"canonical norm count mismatch for {corpus}: "
            f"observed={count} manifest={metadata.get('count')}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "norm_count": count,
        "norm_uids_sha256": _identity_sha256(ordered_uids),
        "source_paths": [str(value) for value in metadata.get("source_paths") or []],
        "source_sha256": metadata.get("source_sha256") or {},
        "coverage_complete": True,
    }


def _required_artifacts(corpus_count: int, norm_count: int) -> dict[str, Any]:
    """Return uniform unfulfilled contracts without inferring current readiness."""

    return {
        "truth": {
            "status": "REQUIRED_NOT_EVALUATED",
            "requirements": [
                "source-group-disjoint train/dev/blind identities frozen before labeling",
                "at least two transcript-isolated full-bank labels per accepted truth row",
                "disagreement-only resolver labels; unresolved rows remain unresolved",
                "blind identities excluded from training, GEPA, thresholds, and model selection",
            ],
        },
        "candidate_capture": {
            "status": "REQUIRED_NOT_EVALUATED",
            "expected_corpora": corpus_count,
            "expected_norms": norm_count,
            "requirements": [
                "one canonical K50 production-candidate audit per corpus",
                "multiple diverse capture trials plus exact full-bank rescue for unresolved rows",
                "all candidates use only metric IDs from the manifest-pinned task bank",
            ],
        },
        "production_inference": {
            "status": "REQUIRED_NOT_EVALUATED",
            "requirements": [
                "external-dev-only retriever/adjudicator/verifier selection",
                "frozen task production plan bound to exact candidates and prompt/model hashes",
                "one pre-rescue typed decision in canonical UID order for every norm",
            ],
        },
        "abstention_rescue": {
            "status": "REQUIRED_NOT_EVALUATED",
            "requirements": [
                "all non-MATCH and low-confidence rows enter repeated full-bank rescue",
                "at least two distinct full-bank candidate systems with exact corpus coverage",
                "strict multi-order adjudication/verification; disagreement fails closed",
            ],
        },
        "release": {
            "status": "REQUIRED_NOT_EVALUATED",
            "expected_corpora": corpus_count,
            "expected_norms": norm_count,
            "requirements": [
                "one canonical typed final row for every manifest norm",
                "all nine decision categories reported including zero-valued categories",
                "passed independent final-production MATCH and abstention blind-risk audits",
                "passed fail-closed all-task release-coverage audit",
            ],
        },
    }


def freeze_alltask_rollout_inventory(manifest_path: Path) -> dict[str, Any]:
    """Validate and hash-lock the eight-task canonical rollout scope."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json_no_duplicate_keys(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    banks, corpora, routing = _validate_manifest_shape(manifest, manifest_path)

    frozen_banks: dict[str, dict[str, Any]] = {}
    for task in ROLLOUT_TASK_ORDER:
        frozen_banks[task], _ = _freeze_bank(
            task=task, metadata=banks[task], manifest_path=manifest_path
        )

    globally_seen_uids: set[str] = set()
    tasks: dict[str, Any] = {}
    for rollout_index, task in enumerate(ROLLOUT_TASK_ORDER, 1):
        task_corpus_names = sorted(
            corpus for corpus, routed_task in routing.items() if routed_task == task
        )
        frozen_corpora = {
            corpus: _freeze_corpus(
                corpus=corpus,
                task=task,
                metadata=corpora[corpus],
                manifest_path=manifest_path,
                globally_seen_uids=globally_seen_uids,
            )
            for corpus in task_corpus_names
        }
        task_norm_count = sum(row["norm_count"] for row in frozen_corpora.values())
        task_record = {
            "rollout_index": rollout_index,
            "bank": frozen_banks[task],
            "corpus_count": len(frozen_corpora),
            "norm_count": task_norm_count,
            "corpora": frozen_corpora,
            "required_artifacts": _required_artifacts(
                len(frozen_corpora), task_norm_count
            ),
        }
        task_record["scope_sha256"] = _json_sha256(
            {
                "task": task,
                "bank": {
                    key: task_record["bank"][key]
                    for key in (
                        "sha256",
                        "metric_count",
                        "metric_ids_sha256",
                        "source_sha256",
                    )
                },
                "corpora": {
                    corpus: {
                        key: row[key]
                        for key in ("sha256", "norm_count", "norm_uids_sha256")
                    }
                    for corpus, row in frozen_corpora.items()
                },
            }
        )
        tasks[task] = task_record

    observed_norms = sum(row["norm_count"] for row in tasks.values())
    if observed_norms != int(manifest["total_norms"]):
        raise ValueError(
            f"manifest total_norms mismatch: observed={observed_norms} "
            f"manifest={manifest['total_norms']}"
        )
    if len(globally_seen_uids) != observed_norms:
        raise ValueError("global canonical norm identity count mismatch")

    manifest_sha256 = sha256_file(manifest_path)
    scope_sha256 = _json_sha256(
        {
            "manifest_sha256": manifest_sha256,
            "tasks": {task: row["scope_sha256"] for task, row in tasks.items()},
        }
    )
    return {
        "schema_version": SCHEMA,
        "status": "FROZEN_CANONICAL_SCOPE",
        "scope_frozen": True,
        "release_ready": False,
        "readiness_evidence_evaluated": False,
        "readiness_warning": (
            "This inventory proves canonical scope only. It does not discover or "
            "validate candidate, truth, inference, rescue, final, or blind-audit outputs."
        ),
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "schema_version": manifest.get("schema_version"),
            "source_mode": "canonical",
        },
        "rollout_order": list(ROLLOUT_TASK_ORDER),
        "scope_sha256": scope_sha256,
        "final_decision_taxonomy": list(FINAL_DECISION_TAXONOMY),
        "totals": {
            "tasks": len(tasks),
            "corpora": sum(row["corpus_count"] for row in tasks.values()),
            "norms": observed_norms,
            "metrics": sum(row["bank"]["metric_count"] for row in tasks.values()),
        },
        "tasks": tasks,
        "authoritative_contracts": {
            "canonical_scope": "validate_manifest.py + this exact inventory",
            "candidate_union": "audit_candidate_outputs.py and audit_task_candidate_coverage.py",
            "truth_panels": "validate_independent_teacher_labels.py and finalize_exact_multi_pass_truth.py",
            "production_inference": "freeze_task_production_plan.py and run_task_production.py",
            "abstention_rescue": "freeze_task_rescue_plan.py, run_task_rescue.py, and merge_rescue_decisions.py",
            "canonical_final": "audit_final_outputs.py",
            "release_risk": "freeze_task_final_risk_release.py",
            "alltask_release": "audit_alltask_release_coverage.py",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = freeze_alltask_rollout_inventory(Path(args.manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "status": payload["status"],
                "totals": payload["totals"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
