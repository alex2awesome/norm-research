#!/usr/bin/env python3
"""Validate the hash-bound all-task MI↔silver execution inventory.

The inventory is a readiness ledger, not a correlation result.  This validator
rechecks every locally available source byte, the frozen task/corpus universe,
the recorded certificate metadata, and the fail-closed readiness logic.  It
does not claim that remote certificate files exist when SSH is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .common import sha256_file


SCHEMA = "silver-match-v3-mi-execution-matrix-v1"
TASK_ORDER = (
    "humor",
    "press-releases",
    "math-stackexchange",
    "peer-review",
    "code-review",
    "creative-writing",
    "legal-outcome-prediction",
    "notice-and-comment",
)
CERTIFICATES: dict[str, tuple[str | None, int, int]] = {
    "humor": (
        "c18a765d68529a9986a02c41012d54f4672deac790898d27dac57010b1b014c3",
        285,
        285,
    ),
    "press-releases": (
        "254ad20bd97603ae6eb3ad819ef2760c6fb5f6bdd2ede47bf194620b4897c56a",
        221,
        221,
    ),
    "math-stackexchange": (
        "46ac755e28224adb67d644c75a22b542ea2af20d378a822339cfffd50113217b",
        141,
        141,
    ),
    "peer-review": (
        "4214e334e39c5bf3fa9baa9535c6168de0e6dce4ab73c926e79d94d645a051f7",
        88,
        88,
    ),
    "code-review": (
        "b3ce76b25cadbee5a98e677d2876423d03d806e59b72cbed4b3fd990112a0b6a",
        133,
        133,
    ),
    "creative-writing": (
        "8ff7c1088309a0183d3a393d2610155df6c3f9d99926072d0247abb03ca0bea6",
        368,
        371,
    ),
    "legal-outcome-prediction": (None, 0, 104),
    "notice-and-comment": (
        "4dacd71da39a29a2a43971358d4706084ed471a228263b147c4fee2554a3fbaf",
        18,
        88,
    ),
}


def _resolve(raw: str, root: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_local_artifacts(
    references: Mapping[str, Any], repo_root: Path
) -> dict[str, Path]:
    output = {}
    for name, ref in references.items():
        if not isinstance(ref, Mapping) or not ref.get("path") or not ref.get("sha256"):
            raise ValueError(f"local evidence lacks path/hash: {name}")
        path = _resolve(str(ref["path"]), repo_root)
        if not path.is_file() or sha256_file(path) != str(ref["sha256"]):
            raise ValueError(f"local evidence changed: {name}={path}")
        output[name] = path
    return output


def validate_matrix(path: Path, repo_root: Path) -> dict[str, Any]:
    path = path.resolve()
    repo_root = repo_root.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported MI execution matrix schema")
    if payload.get("status") != "AUDITED_NO_PRODUCTION_CORRELATIONS_RUN":
        raise ValueError("matrix improperly claims an executed correlation status")
    local = _verify_local_artifacts(payload.get("local_evidence") or {}, repo_root)
    manifest = json.loads(local["canonical_manifest"].read_text(encoding="utf-8"))
    coverage = json.loads(local["current_final_coverage"].read_text(encoding="utf-8"))
    tasks = payload.get("tasks") or []
    if [row.get("task") for row in tasks] != list(TASK_ORDER):
        raise ValueError("task order/coverage differs; Humor must be first and N&C last")
    if len({row["task"] for row in tasks}) != len(TASK_ORDER):
        raise ValueError("duplicate MI matrix task")
    coverage_tasks = coverage.get("tasks") or {}
    norm_total = corpus_total = bank_total = conditional = current = 0
    for row in tasks:
        task = str(row["task"])
        if task not in manifest.get("banks", {}) or task not in coverage_tasks:
            raise ValueError(f"task absent from frozen sources: {task}")
        bank = manifest["banks"][task]
        expected_corpora = [
            corpus
            for corpus, meta in manifest["corpora"].items()
            if meta.get("task") == task
        ]
        expected_norms = sum(
            int(manifest["corpora"][corpus]["count"])
            for corpus in expected_corpora
        )
        expected_corpus_counts = {
            corpus: int(manifest["corpora"][corpus]["count"])
            for corpus in expected_corpora
        }
        if (
            row.get("canonical_norm_count") != expected_norms
            or row.get("corpora") != expected_corpora
            or row.get("corpus_count") != len(expected_corpora)
            or row.get("corpus_norm_counts") != expected_corpus_counts
            or row.get("bank_metric_count") != int(bank["count"])
            or row.get("bank_source_sha256") != bank["source_sha256"]
        ):
            raise ValueError(f"task universe drift: {task}")
        frozen_coverage = coverage_tasks[task]
        if (
            row.get("canonical_final_currently_ready")
            != bool(frozen_coverage["canonical_final_ready"])
            or row.get("canonical_final_currently_ready") is not False
        ):
            raise ValueError(f"matrix final readiness is stale or unsafe: {task}")
        cert_sha, joined, bank_count = CERTIFICATES[task]
        certificate = row.get("mi_certificate") or {}
        if (
            certificate.get("recorded_sha256") != cert_sha
            or int(certificate.get("joined_bank_metrics", -1)) != joined
            or int(certificate.get("bank_metrics", -1)) != bank_count
            or certificate.get("locally_verified_this_audit") is not False
            or certificate.get("remote_exists_verified_this_audit") is not False
        ):
            raise ValueError(f"recorded certificate metadata drift: {task}")
        expected_coverage = joined / bank_count if bank_count else 0.0
        if abs(float(certificate.get("recorded_bank_coverage", -1)) - expected_coverage) > 1e-12:
            raise ValueError(f"certificate coverage arithmetic drift: {task}")
        conditional_ready = cert_sha is not None and joined >= 4
        if row.get("conditionally_runnable_after_final_and_cert_reverify") is not conditional_ready:
            raise ValueError(f"conditional readiness drift: {task}")
        if row.get("runnable_now") is not False:
            raise ValueError(f"production correlation was prematurely authorized: {task}")
        norm_total += expected_norms
        corpus_total += len(expected_corpora)
        bank_total += int(bank["count"])
        conditional += int(conditional_ready)
        current += int(bool(row["runnable_now"]))
    summary = payload.get("summary") or {}
    if (
        norm_total != int(manifest["total_norms"])
        or corpus_total != int(manifest["total_corpora"])
        or bank_total != sum(int(value["count"]) for value in manifest["banks"].values())
        or summary.get("canonical_norms") != norm_total
        or summary.get("corpora") != corpus_total
        or summary.get("tasks") != len(TASK_ORDER)
        or summary.get("bank_metrics") != bank_total
        or summary.get("runnable_now_tasks") != current
        or summary.get("conditionally_runnable_after_final_and_cert_reverify_tasks")
        != conditional
        or summary.get("tasks_missing_compatible_certificate") != 1
    ):
        raise ValueError("MI matrix summary totals drift")
    contract = payload.get("correlation_contract") or {}
    if (
        contract.get("primary_estimand") != "source_presence.OPT.spearman_rho"
        or "exact task/corpus/norm coverage"
        not in str(contract.get("upstream_ce_universe") or "")
        or "canonical finals remain"
        not in str(contract.get("upstream_ce_universe") or "")
        or contract.get("canonical_row_join") != "norm_uid in exact manifest row order"
        or contract.get("certificate_primary_join")
        != "metric_index parsed from certificate file plus normalized-name equality"
        or contract.get("certificate_fallback_join") != "unique normalized metric name"
        or contract.get("source_group_key")
        != "corpus + source_id; else corpus + paper_id; else corpus + norm_uid"
    ):
        raise ValueError("correlation join/estimand contract drift")
    return {
        "schema_version": "silver-match-v3-mi-execution-matrix-validation-v1",
        "status": "PASS",
        "matrix": {"path": str(path), "sha256": sha256_file(path)},
        "tasks": len(TASK_ORDER),
        "canonical_norms": norm_total,
        "corpora": corpus_total,
        "bank_metrics": bank_total,
        "runnable_now_tasks": current,
        "conditional_tasks": conditional,
        "remote_certificate_files_reverified": 0,
        "production_correlations_run": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate_matrix(Path(args.matrix), Path(args.repo_root))
    if args.output:
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
