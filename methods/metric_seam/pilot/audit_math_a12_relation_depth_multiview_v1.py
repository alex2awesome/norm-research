#!/usr/bin/env python3
"""Additive multi-view relation-depth audit for the Math-a12 projection.

The original pair-certificate projection used a successful-output convention:
depth 3 only when at least one rational pair parsed.  That convention is useful
for positive evidence, but it understates execution depth on rows whose formal
parse attempt directly produced an abstention.  This audit retains the original
artifact unchanged and emits three explicitly named row-level views:

* deepest attempted depth;
* deepest decision-contributing depth, including formal coverage failures; and
* positive relation-evidence depth, with no-evidence rows counted separately.

This is a post-reference audit of frozen code outputs.  It does not load a prompt
reference, execute a model, estimate reconstruction/isomorphism, or alter v1.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.battery.seal_ctext_items_v2 import canonical_bytes, sha256
from methods.metric_seam.battery.technical_entry_v1 import normalize_depth


PROJECTION_MANIFEST_SCHEMA = (
    "metric-seam.math-a12-pair-certificate-projection-manifest.v1"
)
AUDIT_SCHEMA = "metric-seam.math-a12-relation-depth-multiview-audit.v1"
ROW_SCHEMA = "metric-seam.math-a12-relation-depth-multiview-rows.v1"
MANIFEST_SCHEMA = "metric-seam.math-a12-relation-depth-multiview-manifest.v1"
DEPTH_SCHEMA = "metric-seam.relation-depth.v1"
CRITERION_ID = "math__a12"
RELATION_ID = "explicit_rational_equality_preservation"
REQUIRED_INPUT_ARTIFACTS = (
    "pair_certificates.jsonl",
    "row_projection.json",
    "relation_depth.json",
    "projection_summary.json",
)


class DepthAuditError(ValueError):
    """A frozen projection or proposed depth view is internally inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DepthAuditError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DepthAuditError(f"{path} must contain a JSON object")
    return value


def _verify_bound_file(record: Mapping[str, Any], *, label: str) -> Path:
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise DepthAuditError(f"{label} lacks a path/sha256 binding")
    path = Path(path_value).resolve()
    if not path.is_file() or sha256(path) != expected:
        raise DepthAuditError(f"{label} no longer matches its frozen binding")
    return path


def _verify_projection(
    projection_dir: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    manifest = _load_json(projection_dir / "projection_manifest.json")
    if manifest.get("schema") != PROJECTION_MANIFEST_SCHEMA:
        raise DepthAuditError("unexpected pair-projection manifest schema")
    if manifest.get("reference_loaded_or_used") is not False:
        raise DepthAuditError("pair projection does not assert reference absence")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DepthAuditError("pair-projection manifest has no artifact bindings")
    for name in REQUIRED_INPUT_ARTIFACTS:
        record = artifacts.get(name)
        path = projection_dir / name
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("sha256"), str)
            or not path.is_file()
            or sha256(path) != record["sha256"]
        ):
            raise DepthAuditError(f"frozen projection artifact mismatch: {name}")

    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise DepthAuditError("pair-projection manifest has no execution bindings")
    for name in ("worker", "replay"):
        record = execution.get(name)
        if not isinstance(record, Mapping):
            raise DepthAuditError(f"pair-projection manifest has no {name} binding")
        _verify_bound_file(record, label=f"frozen pair-projection {name}")

    summary = _load_json(projection_dir / "projection_summary.json")
    depth = _load_json(projection_dir / "relation_depth.json")
    row_payload = _load_json(projection_dir / "row_projection.json")
    rows_raw = row_payload.get("rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise DepthAuditError("row projection must contain a non-empty rows list")
    rows = []
    for index, row in enumerate(rows_raw):
        if not isinstance(row, Mapping):
            raise DepthAuditError(f"row projection entry {index} is not an object")
        rows.append(row)

    certificates: list[Mapping[str, Any]] = []
    try:
        lines = (projection_dir / "pair_certificates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        for index, line in enumerate(lines):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise DepthAuditError(f"pair certificate {index} is not an object")
            certificates.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DepthAuditError(f"cannot load pair certificates: {exc}") from exc

    if len(rows) != summary.get("heldout_count"):
        raise DepthAuditError("row count differs from projection summary")
    if len(certificates) != summary.get("pair_certificate_count"):
        raise DepthAuditError("certificate count differs from projection summary")
    return manifest, summary, depth, rows, certificates


def _row_depths(
    rows: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    certificate_counts = Counter()
    positive_counts = Counter()
    status_counts: dict[str, Counter[str]] = {}
    known_ids: set[str] = set()
    for row in rows:
        datapoint_id = row.get("datapoint_id")
        if not isinstance(datapoint_id, str) or not datapoint_id or datapoint_id in known_ids:
            raise DepthAuditError("row projection has invalid or duplicate datapoint ids")
        known_ids.add(datapoint_id)
        status_counts[datapoint_id] = Counter()
    for certificate in certificates:
        datapoint_id = certificate.get("datapoint_id")
        status = certificate.get("status")
        if datapoint_id not in known_ids or not isinstance(status, str):
            raise DepthAuditError("certificate has an unknown row or invalid status")
        certificate_counts[datapoint_id] += 1
        positive_counts[datapoint_id] += int(
            certificate.get("positive_code_witness") is True
        )
        status_counts[datapoint_id][status] += 1

    audited: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for row in rows:
        datapoint_id = row["datapoint_id"]
        analysis = row.get("analysis")
        if not isinstance(analysis, Mapping):
            raise DepthAuditError(f"{datapoint_id} has no analysis object")
        candidate_n = analysis.get("pair_candidate_count")
        parsed_n = analysis.get("parsed_rational_pair_count")
        positive_n = analysis.get("positive_code_witness_count")
        abstained = analysis.get("abstained")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (candidate_n, parsed_n, positive_n)
        ) or not isinstance(abstained, bool):
            raise DepthAuditError(f"{datapoint_id} has malformed analysis counts")
        if candidate_n != row.get("certificate_count") or candidate_n != certificate_counts[
            datapoint_id
        ]:
            raise DepthAuditError(f"{datapoint_id} certificate counts disagree")
        if positive_n != positive_counts[datapoint_id]:
            raise DepthAuditError(f"{datapoint_id} positive-witness counts disagree")
        if parsed_n != candidate_n - status_counts[datapoint_id]["parse_noncoverage"]:
            raise DepthAuditError(f"{datapoint_id} parsed-pair counts disagree")
        if abstained != (parsed_n == 0):
            raise DepthAuditError(f"{datapoint_id} abstention is inconsistent")

        if candidate_n == 0:
            category = "parser_structure_only_no_pair_candidate"
            attempted_depth = 1
            contributing_depth = 1
            positive_evidence_depth = None
        elif positive_n > 0:
            category = "formal_positive_relation_evidence"
            attempted_depth = 3
            contributing_depth = 3
            positive_evidence_depth = 3
        elif parsed_n == 0:
            category = "formal_parse_noncoverage_abstention"
            attempted_depth = 3
            contributing_depth = 3
            positive_evidence_depth = None
        else:
            category = "formal_execution_without_positive_relation_evidence"
            attempted_depth = 3
            contributing_depth = 3
            positive_evidence_depth = None
        category_counts[category] += 1
        audited.append(
            {
                "datapoint_id": datapoint_id,
                "pair_candidate_count": candidate_n,
                "parsed_rational_pair_count": parsed_n,
                "positive_code_witness_count": positive_n,
                "abstained": abstained,
                "depth_category": category,
                "deepest_attempted_depth": attempted_depth,
                "deepest_decision_contributing_depth": contributing_depth,
                "positive_relation_evidence_depth": positive_evidence_depth,
            }
        )
    return audited, dict(sorted(category_counts.items()))


def _histogram(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row[key]) for row in rows if row.get(key) is not None)
    return dict(sorted(counts.items()))


def _render_report(summary: Mapping[str, Any]) -> str:
    views = summary["depth_views"]
    attempted = views["deepest_attempted"]["histogram"]
    positive = views["positive_relation_evidence"]
    categories = summary["row_category_counts"]
    pair_counts = summary["pair_counts"]
    return f"""# Math a12 relation-depth multi-view audit

## Corrected interpretation

The earlier projection placed 74 abstained rows at depth 1 because its dynamic rule counted
depth 3 only when the formal operation returned a parsed result. That is a successful-output
view, not a complete execution-depth view. Of those 74 rows, **{categories.get('formal_parse_noncoverage_abstention', 0)}** formed equation pairs and invoked the formal parser; all
of their pair attempts returned parse noncoverage. Those failures directly determined the
row abstention.

Under the audited execution/decision convention, **{attempted.get('1', 0)} rows
({100 * attempted.get('1', 0) / summary['heldout_count']:.1f}%)** stop at parsed document
structure (depth 1), while **{attempted.get('3', 0)} rows
({100 * attempted.get('3', 0) / summary['heldout_count']:.1f}%)** attempt the formal path
(depth 3). The deepest decision-contributing histogram is identical: formal parse failure is
an abstention-producing result, not evidence that formal code was never executed.

Positive relation evidence remains narrower: **{positive['evidence_rows']} rows
({100 * positive['evidence_rows'] / summary['heldout_count']:.1f}%)** have depth-3 positive
code witnesses; **{positive['no_positive_evidence_rows']}** have none. Among the
{attempted.get('3', 0)} rows reaching the formal path, the positive-evidence rate is
{100 * positive['evidence_rows'] / attempted.get('3', 1):.1f}%.

At pair level, the frozen operation made **{pair_counts['formal_attempt_n']}** formal attempts:
**{pair_counts['positive_evidence_n']}** produced relation-local evidence and
**{pair_counts['parse_noncoverage_n']}** returned parse noncoverage.

## Claim boundary

This additive audit supersedes only the interpretation of the old dynamic depth histogram;
it does not alter the frozen pair certificates, per-row classifications, or aggregates. A
depth-3 attempt is not a successful solver result, positive evidence, construct fidelity, or
reconstruction. The positive-evidence view is therefore reported separately rather than
crediting coverage failures as witnesses.

This is a post-reference audit of already-frozen code outputs. It loaded no prompt reference
and used no model, API, external supervised anchor, or GPU.
"""


def audit_relation_depth(
    *, projection_dir: Path, output_dir: Path
) -> tuple[Path, Path, Path]:
    """Verify the frozen projection and write a separate multi-view audit."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite depth audit {output_dir}")
    manifest, projection_summary, old_depth, rows, certificates = _verify_projection(
        projection_dir
    )
    audited_rows, category_counts = _row_depths(rows, certificates)
    heldout_count = len(audited_rows)
    heldout_ids = sorted(row["datapoint_id"] for row in audited_rows)
    universe_sha256 = _canonical_sha256(heldout_ids)
    candidate_sha256 = old_depth.get("candidate_sha256")
    if not isinstance(candidate_sha256, str):
        raise DepthAuditError("old depth artifact has no candidate binding")
    normalize_depth(
        old_depth,
        heldout_count=heldout_count,
        candidate_sha256=candidate_sha256,
        criterion_id=CRITERION_ID,
        relation_id=RELATION_ID,
        universe_sha256=universe_sha256,
    )

    attempted_histogram = _histogram(audited_rows, "deepest_attempted_depth")
    contributing_histogram = _histogram(
        audited_rows, "deepest_decision_contributing_depth"
    )
    positive_histogram = _histogram(
        audited_rows, "positive_relation_evidence_depth"
    )
    evidence_rows = sum(positive_histogram.values())
    depth_views = {
        "deepest_attempted": {
            "histogram": attempted_histogram,
            "accounted_rows": heldout_count,
            "semantics": (
                "deepest operation invoked; a pair sent to the formal verification path "
                "counts at depth 3 even when parsing fails"
            ),
        },
        "deepest_decision_contributing": {
            "histogram": contributing_histogram,
            "accounted_rows": heldout_count,
            "semantics": (
                "deepest operation whose result determines evidence or abstention; formal "
                "parse noncoverage contributes to abstention"
            ),
        },
        "positive_relation_evidence": {
            "histogram": positive_histogram,
            "evidence_rows": evidence_rows,
            "no_positive_evidence_rows": heldout_count - evidence_rows,
            "semantics": "depth only for rows with at least one positive code witness",
        },
    }
    corrected_depth = {
        "scale": DEPTH_SCHEMA,
        "criterion_id": CRITERION_ID,
        "relation_id": RELATION_ID,
        "candidate_sha256": candidate_sha256,
        "universe_sha256": universe_sha256,
        "nodes": old_depth["nodes"],
        "static_max_relation_depth": 3,
        "longest_path_edges": old_depth.get("longest_path_edges"),
        "dynamic_contributing_depth_histogram": contributing_histogram,
        "dynamic_rule": (
            "deepest decision-contributing operation; formal parse noncoverage is a "
            "depth-3 abstention result, while rows with no pair candidate stop at depth 1"
        ),
        "dynamic_deepest_attempted_depth_histogram": attempted_histogram,
        "dynamic_positive_evidence_depth_histogram": positive_histogram,
        "dynamic_positive_evidence_no_evidence_n": heldout_count - evidence_rows,
        "depth_views": depth_views,
        "supersedes_interpretation_of": {
            "path": str((projection_dir / "relation_depth.json").resolve()),
            "sha256": sha256(projection_dir / "relation_depth.json"),
            "prior_dynamic_histogram": old_depth.get(
                "dynamic_contributing_depth_histogram"
            ),
            "scope": "dynamic depth interpretation only; frozen classifications unchanged",
        },
    }
    normalize_depth(
        corrected_depth,
        heldout_count=heldout_count,
        candidate_sha256=candidate_sha256,
        criterion_id=CRITERION_ID,
        relation_id=RELATION_ID,
        universe_sha256=universe_sha256,
    )

    certificate_statuses = Counter(
        certificate["status"] for certificate in certificates
    )
    pair_positive_n = sum(
        certificate.get("positive_code_witness") is True
        for certificate in certificates
    )
    summary = {
        "schema": AUDIT_SCHEMA,
        "task": "math",
        "criterion_id": "a12",
        "relation_id": RELATION_ID,
        "temporal_status": "post_reference_frozen_projection_audit",
        "reference_loaded_or_used": False,
        "model_calls": False,
        "gpu_used": False,
        "new_blind_result": False,
        "new_reconstruction_result": False,
        "new_isomorphism_result": False,
        "heldout_count": heldout_count,
        "source_projection_verified": True,
        "source_projection_unchanged": True,
        "source_row_and_aggregate_results_unchanged": True,
        "prior_successful_output_histogram": old_depth.get(
            "dynamic_contributing_depth_histogram"
        ),
        "depth_views": depth_views,
        "row_category_counts": category_counts,
        "pair_counts": {
            "formal_attempt_n": len(certificates),
            "positive_evidence_n": pair_positive_n,
            "parse_noncoverage_n": certificate_statuses["parse_noncoverage"],
        },
        "adjudication": (
            "the prior histogram was internally precise under its success-only rule but "
            "incomplete as an execution-depth headline; multi-view depth is required"
        ),
        "claim_boundary": (
            "attempted/contributing depth includes abstention-producing formal failures; "
            "positive evidence remains a separate, narrower view"
        ),
    }
    row_payload = {"schema": ROW_SCHEMA, "rows": audited_rows}

    output_dir.mkdir(parents=True, exist_ok=False)
    depth_path = output_dir / "relation_depth_multiview.json"
    _write_exclusive(depth_path, corrected_depth)
    rows_path = output_dir / "row_depth_audit.json"
    _write_exclusive(rows_path, row_payload)
    summary_path = output_dir / "audit_summary.json"
    _write_exclusive(summary_path, summary)
    report_path = output_dir / "AUDIT_REPORT.md"
    _write_exclusive(report_path, _render_report(summary).encode("utf-8"))
    audit_manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": _utc_now(),
        "reference_loaded_or_used": False,
        "gpu_used": False,
        "inputs": {
            "projection_manifest": {
                "path": str((projection_dir / "projection_manifest.json").resolve()),
                "sha256": sha256(projection_dir / "projection_manifest.json"),
            },
            **{
                name: {
                    "path": str((projection_dir / name).resolve()),
                    "sha256": sha256(projection_dir / name),
                }
                for name in REQUIRED_INPUT_ARTIFACTS
            },
            "source_projection_manifest_schema": manifest["schema"],
            "source_projection_summary_schema": projection_summary.get("schema"),
        },
        "implementation": {
            "audit": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__)),
            }
        },
        "artifacts": {
            path.name: {"sha256": sha256(path)}
            for path in (depth_path, rows_path, summary_path, report_path)
        },
    }
    manifest_path = output_dir / "audit_manifest.json"
    _write_exclusive(manifest_path, audit_manifest)
    _write_exclusive(
        output_dir / "audit_manifest.sha256",
        (sha256(manifest_path) + "  audit_manifest.json\n").encode("utf-8"),
    )
    return summary_path, depth_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, depth, report = audit_relation_depth(
        projection_dir=args.projection_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"summary": str(summary), "depth": str(depth), "report": str(report)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
