#!/usr/bin/env python3
"""Unsupervised classifier over frozen telemetry from prior PR transplant executions.

The consolidated CSV includes a historical accept/reject ``judgement`` column.  This probe
never reads it. It does not run repositories or tests in the current replay. It reports only
the stored execution/certificate coverage and abstention/failure modes emitted by the legacy
prototype transplant runner. This artifact is not the active metric-seam coding census.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BASE = REPO_ROOT / "datasets" / "code-review" / "pr_test_execution"
DEFAULT_CSV = BASE / "outputs" / "transplant_consolidated_2026_07_12_canonical.csv"
DEFAULT_REPORT = BASE / "REPORT_consolidated_dataset_2026_07_12.md"
DEFAULT_OUT = (
    REPO_ROOT / "outputs" / "metric_seam_pilot" / "technical_replay_v2" / "code_execution_probe.json"
)

CERTIFICATE_LABELS = frozenset({"pinned", "partial_pinned", "vacuous"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_number(row: dict[str, str], field: str) -> bool:
    raw = row.get(field, "").strip()
    if not raw:
        return False
    try:
        return float(raw) > 0
    except ValueError:
        return False


def run_probe(csv_path: Path, report_path: Path) -> dict[str, Any]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = Counter(row["transplant_pr_label"] for row in rows)
    languages = Counter(row["language"] for row in rows)
    certificate_rows = sum(labels[label] for label in CERTIFICATE_LABELS)
    row_ids = Counter(row["row_id"] for row in rows)
    duplicate_row_ids = sorted(row_id for row_id, count in row_ids.items() if count > 1)
    integrity = {
        "row_ids_unique": not duplicate_row_ids,
        "label_counts_sum_to_total": sum(labels.values()) == len(rows),
        "pinned_rows_have_assertion_failure": all(
            _positive_number(row, "n_assertion_fail")
            for row in rows
            if row["transplant_pr_label"] == "pinned"
        ),
        "vacuous_rows_have_vacuous_pass": all(
            _positive_number(row, "n_vacuous_pass")
            for row in rows
            if row["transplant_pr_label"] == "vacuous"
        ),
        "partial_rows_have_both_signals": all(
            _positive_number(row, "n_assertion_fail")
            and _positive_number(row, "n_vacuous_pass")
            for row in rows
            if row["transplant_pr_label"] == "partial_pinned"
        ),
    }
    hard_checks = {key: value for key, value in integrity.items() if key != "row_ids_unique"}
    if not all(hard_checks.values()):
        raise AssertionError(f"execution artifact integrity failure: {integrity}")
    return {
        "schema_version": "technical-code-execution-probe-v1",
        "external_supervision": "none",
        "ignored_source_fields": ["judgement", "judgement_source"],
        "sources": {
            "csv": {
                "path": str(csv_path.relative_to(REPO_ROOT)),
                "sha256": sha256(csv_path),
                "bytes": csv_path.stat().st_size,
            },
            "report": {
                "path": str(report_path.relative_to(REPO_ROOT)),
                "sha256": sha256(report_path),
                "bytes": report_path.stat().st_size,
            },
        },
        "pipeline_status": "selected",
        "selection_mode": "retrospective_seed",
        "program_lineage": "legacy_prototype_not_active_coding_census",
        "current_replay_executes_repositories": False,
        "relation_depth": 4,
        "relation_depth_label": "environment_or_world_execution",
        "n_rows": len(rows),
        "n_unique_row_ids": len(row_ids),
        "duplicate_row_ids": duplicate_row_ids,
        "language_counts": dict(sorted(languages.items())),
        "label_counts": dict(sorted(labels.items())),
        "certificate_coverage": {
            "n_any_transition_certificate": certificate_rows,
            "fraction_any_transition_certificate": certificate_rows / len(rows) if rows else None,
            "n_pinned": labels["pinned"],
            "n_partial_pinned": labels["partial_pinned"],
            "n_vacuous": labels["vacuous"],
            "n_none_abstention": labels["none"],
            "n_indeterminate": labels["indeterminate"],
            "n_other_execution_errors": len(rows)
            - certificate_rows
            - labels["none"]
            - labels["indeterminate"],
        },
        "signal_coverage": {
            "n_rows_with_assertion_failure": sum(
                _positive_number(row, "n_assertion_fail") for row in rows
            ),
            "n_rows_with_vacuous_pass": sum(
                _positive_number(row, "n_vacuous_pass") for row in rows
            ),
            "n_rows_with_compile_failure": sum(
                _positive_number(row, "n_compile_fail") for row in rows
            ),
        },
        "integrity_checks": integrity,
        "interpretation": {
            "utility": (
                "stored telemetry from prior environment execution contains narrow behavioral "
                "transition certificates on a minority of rows and honest indeterminate/abstention "
                "outcomes elsewhere; the current replay only classifies those frozen rows"
            ),
            "constructive_extension": (
                "pinned/partial/vacuous transitions are evidence unavailable from diff text alone; "
                "no accept/reject outcome is used to score or tune the certificate"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run_probe(args.csv.resolve(), args.report.resolve())
    if args.check:
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
