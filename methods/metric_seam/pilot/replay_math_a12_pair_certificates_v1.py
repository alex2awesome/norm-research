#!/usr/bin/env python3
"""Post-reference pair-certificate projection for sealed Math-a12 v1.

This additive replay leaves the frozen v1 execution and finalization untouched.
It reconstructs the identical sanitized held-out rows, runs the identical frozen
symbolic relation without loading a prompt reference, and retains inspectable
pair-level outputs that the original count-only interface discarded.  It writes
nothing until every row and aggregate exactly match the sealed v1 execution.

Because the v1 prompt reference has already been opened, this is explicitly a
post-reference projection/audit replay.  It is not a new blind run, reconstruction
estimate, isomorphism result, or opportunity to tune the operation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from methods.metric_seam.battery.seal_ctext_items_v2 import canonical_bytes, sha256
from methods.metric_seam.battery.technical_entry_v1 import normalize_depth
from methods.metric_seam.pilot._math_a12_pair_certificate_worker_v1 import (
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
)
from methods.metric_seam.pilot.evaluate_math_a12_symbolic_step_heldout_v1 import (
    HeldoutIntegrityError,
    _reconstruct_partition,
    _verify_execution,
    _verify_preparation,
)


ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("_math_a12_pair_certificate_worker_v1.py")
SUMMARY_SCHEMA = "metric-seam.math-a12-pair-certificate-projection.v1"
MANIFEST_SCHEMA = "metric-seam.math-a12-pair-certificate-projection-manifest.v1"
DEPTH_SCHEMA = "metric-seam.relation-depth.v1"
CRITERION_ID = "math__a12"
RELATION_ID = "explicit_rational_equality_preservation"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _run_worker(request: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="metric_seam_math_a12_pair_projection_") as name:
        temporary = Path(name)
        request_path = temporary / "request.json"
        result_path = temporary / "result.json"
        request_path.write_bytes(canonical_bytes(request))
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(temporary),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
        }
        process = subprocess.run(
            [sys.executable, str(WORKER), str(request_path), str(result_path)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if process.returncode != 0:
            message = (process.stderr or process.stdout or "pair projection failed")[-2000:]
            raise HeldoutIntegrityError(
                f"pair-certificate worker exited {process.returncode}: {message}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != RESULT_SCHEMA:
        raise HeldoutIntegrityError("pair-certificate worker returned an unknown schema")
    return result


def _request(heldout: Sequence[tuple[str, str]]) -> tuple[dict, dict[str, str]]:
    alias_to_id = {
        f"heldout_{index:04d}": datapoint_id
        for index, (datapoint_id, _) in enumerate(heldout, 1)
    }
    return (
        {
            "schema": REQUEST_SCHEMA,
            "relation_id": RELATION_ID,
            "reference_values_present": False,
            "source_identifiers_present": False,
            "eval_items": [
                {"item_key": alias, "ctext": ctext}
                for alias, (_, ctext) in zip(alias_to_id, heldout)
            ],
        },
        alias_to_id,
    )


def _flatten_certificates(
    outputs: Sequence[Mapping[str, Any]], alias_to_id: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    for output in outputs:
        alias = output["item_key"]
        datapoint_id = alias_to_id[alias]
        projection = output["projection"]
        first_index = len(certificates) + 1 if projection["certificates"] else None
        for certificate in projection["certificates"]:
            certificates.append(
                {
                    "certificate_index": len(certificates) + 1,
                    "datapoint_id": datapoint_id,
                    **certificate,
                }
            )
        rows.append(
            {
                "datapoint_id": datapoint_id,
                "analysis": projection["analysis"],
                "certificate_count": len(projection["certificates"]),
                "first_certificate_index": first_index,
                "dynamic_max_contributing_depth": projection[
                    "dynamic_max_contributing_depth"
                ],
            }
        )
    return rows, certificates


def _aggregate_certificate_statuses(certificates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(certificate["status"] for certificate in certificates)
    return dict(sorted(counts.items()))


def _depth_profile(
    *, rows: Sequence[Mapping[str, Any]], operation_sha256: str
) -> dict[str, Any]:
    heldout_ids = sorted(row["datapoint_id"] for row in rows)
    universe_sha256 = _canonical_sha256(heldout_ids)
    histogram = Counter(str(row["dynamic_max_contributing_depth"]) for row in rows)
    profile = {
        "scale": DEPTH_SCHEMA,
        "criterion_id": CRITERION_ID,
        "relation_id": RELATION_ID,
        "candidate_sha256": operation_sha256,
        "universe_sha256": universe_sha256,
        "nodes": [
            {
                "node_id": "math_span_and_equality_structure_parser",
                "implementation": "code",
                "relation_depth": 1,
                "contributes_to_output": True,
            },
            {
                "node_id": "exact_sympy_rational_solver",
                "implementation": "code",
                "relation_depth": 3,
                "contributes_to_output": True,
            },
        ],
        "static_max_relation_depth": 3,
        "longest_path_edges": 1,
        "dynamic_contributing_depth_histogram": dict(sorted(histogram.items())),
        "dynamic_rule": (
            "depth 3 iff at least one rational pair parsed and exact solver output "
            "contributed; otherwise depth 1 parser/structure determined abstention"
        ),
    }
    normalize_depth(
        profile,
        heldout_count=len(rows),
        candidate_sha256=operation_sha256,
        criterion_id=CRITERION_ID,
        relation_id=RELATION_ID,
        universe_sha256=universe_sha256,
    )
    return profile


def _render_report(summary: Mapping[str, Any]) -> str:
    counts = summary["sealed_v1_summary"]
    statuses = summary["pair_status_counts"]
    depth = summary["relation_depth"]["dynamic_contributing_depth_histogram"]
    return f"""# Math a12 pair-certificate projection replay

## Outcome

This post-reference, code-only audit replay projected **{counts['pair_candidate_count']}**
inspectable pair records from the same 100 held-out rows used by sealed v1. Every per-row
analysis object and the full aggregate are exactly equal to sealed v1 before any artifact was
written.

Pair statuses: **{statuses.get('verified_rational_identity', 0)}** exact identities,
**{statuses.get('exact_nonidentity_witness', 0)}** exact nonidentities,
**{statuses.get('parse_noncoverage', 0)}** parse noncoverage, and
**{statuses.get('symbolically_unresolved', 0)}** unresolved. Parsed certificates retain the
sanitized bounded expression pair, canonical SymPy expressions, denominator-nonzero
obligations, and counterexample assignments where present.

Domain obligations are a faithful projection of the frozen v1 operation, not a new
singularity analysis. SymPy simplification can erase a cancelled denominator before the
operation records it, so an empty obligation list does not certify equality over the total
domain.

The audited relation-depth profile has static maximum depth 3. Dynamic maximum-contributing
depth is depth 1 on **{depth.get('1', 0)}** rows and depth 3 on **{depth.get('3', 0)}** rows.

## Claim boundary

This projection was run after sealed v1 had already opened its prompt reference. The replay did
not load or use that reference, but it is **not** a new blind reconstruction or isomorphism
result. It only makes the already-frozen relation-local code classifications inspectable.
Nonidentity remains a relation witness, not a document defect, because universal claim scope
was not supplied. Parse noncoverage remains abstention.

No model, API, external supervised anchor, or GPU was used.
"""


def replay_pair_certificates(
    *,
    preparation_dir: Path,
    execution_dir: Path,
    output_dir: Path,
    process_timeout: float = 900.0,
) -> tuple[Path, Path, Path]:
    """Replay pair projection and write only after exact sealed-v1 equivalence."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite pair projection {output_dir}")
    sealed_execution, sealed_manifest = _verify_execution(execution_dir)
    prep_manifest, bundle, source_path = _verify_preparation(preparation_dir)
    recorded_prep = sealed_manifest.get("inputs", {}).get("preparation_manifest", {})
    if (
        Path(recorded_prep.get("path", "")).resolve()
        != (preparation_dir / "prepare_manifest.json").resolve()
        or recorded_prep.get("sha256") != sha256(preparation_dir / "prepare_manifest.json")
    ):
        raise HeldoutIntegrityError("sealed v1 execution binds a different preparation")
    _, heldout, heldout_redactions = _reconstruct_partition(
        prep_manifest, bundle, source_path
    )
    request, alias_to_id = _request(heldout)
    started_at = _utc_now()
    worker_result = _run_worker(request, timeout=process_timeout)
    completed_at = _utc_now()
    outputs = worker_result.get("outputs")
    expected_aliases = list(alias_to_id)
    if not isinstance(outputs, list) or [row.get("item_key") for row in outputs] != expected_aliases:
        raise HeldoutIntegrityError("pair worker returned unexpected held-out aliases")
    rows, certificates = _flatten_certificates(outputs, alias_to_id)

    sealed_rows = {
        row["datapoint_id"]: row["analysis"] for row in sealed_execution["rows"]
    }
    replayed_rows = {row["datapoint_id"]: row["analysis"] for row in rows}
    if replayed_rows != sealed_rows:
        raise HeldoutIntegrityError("pair projection row classifications differ from sealed v1")
    if sealed_execution["summary"] != _sealed_summary(rows):
        raise HeldoutIntegrityError("pair projection aggregate differs from sealed v1")
    if len(certificates) != sealed_execution["summary"]["pair_candidate_count"]:
        raise HeldoutIntegrityError("pair projection count differs from sealed v1")

    operation_sha = prep_manifest["implementation"]["symbolic_operation"]["sha256"]
    depth_profile = _depth_profile(rows=rows, operation_sha256=operation_sha)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "task": "math",
        "criterion_id": "a12",
        "relation_id": RELATION_ID,
        "temporal_status": "post_reference_projection_replay",
        "reference_loaded_or_used_by_replay": False,
        "new_blind_result": False,
        "new_reconstruction_result": False,
        "new_isomorphism_result": False,
        "external_supervised_anchor": False,
        "model_calls": False,
        "gpu_used": False,
        "heldout_count": len(rows),
        "sealed_v1_row_classifications_exact": True,
        "sealed_v1_aggregate_exact": True,
        "sealed_v1_summary": sealed_execution["summary"],
        "pair_certificate_count": len(certificates),
        "pair_status_counts": _aggregate_certificate_statuses(certificates),
        "representation_policy": (
            "sanitized bounded equation-side LaTeX; canonical SymPy form only when parsed"
        ),
        "domain_obligation_boundary": (
            "faithful frozen-v1 output only; cancelled singularities may be simplified "
            "away, and an empty list does not certify total-domain equality"
        ),
        "relation_depth": depth_profile,
        "claim_boundary": (
            "inspectable projection of frozen relation-local classifications; nonidentity "
            "is not a document defect without universal scope"
        ),
    }
    row_projection = {
        "schema": "metric-seam.math-a12-pair-certificate-rows.v1",
        "rows": rows,
    }

    # All equivalence/depth checks above complete before the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=False)
    certificates_path = output_dir / "pair_certificates.jsonl"
    certificate_bytes = b"".join(canonical_bytes(row) for row in certificates)
    _write_exclusive(certificates_path, certificate_bytes)
    rows_path = output_dir / "row_projection.json"
    _write_exclusive(rows_path, row_projection)
    depth_path = output_dir / "relation_depth.json"
    _write_exclusive(depth_path, depth_profile)
    summary_path = output_dir / "projection_summary.json"
    _write_exclusive(summary_path, summary)
    report_path = output_dir / "REPORT.md"
    _write_exclusive(report_path, _render_report(summary).encode("utf-8"))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": _utc_now(),
        "replay_started_at": started_at,
        "replay_completed_at": completed_at,
        "temporal_status": "post_reference_projection_replay",
        "reference_path_accepted_by_cli": False,
        "reference_loaded_or_used": False,
        "inputs": {
            "sealed_v1_execution": {
                "path": str((execution_dir / "candidate_execution.json").resolve()),
                "sha256": sha256(execution_dir / "candidate_execution.json"),
            },
            "sealed_v1_manifest": {
                "path": str((execution_dir / "execution_manifest.json").resolve()),
                "sha256": sha256(execution_dir / "execution_manifest.json"),
            },
            "preparation_manifest": {
                "path": str((preparation_dir / "prepare_manifest.json").resolve()),
                "sha256": sha256(preparation_dir / "prepare_manifest.json"),
            },
            "compiler_bundle": {
                "path": str((preparation_dir / "compiler_bundle.json").resolve()),
                "sha256": sha256(preparation_dir / "compiler_bundle.json"),
            },
            "symbolic_operation": prep_manifest["implementation"]["symbolic_operation"],
        },
        "execution": {
            "heldout_identifiers_sent_to_worker": False,
            "reference_values_sent_to_worker": False,
            "request_sha256": _sha256_bytes(canonical_bytes(request)),
            "sanitizer_heldout_receipt": heldout_redactions,
            "worker": {"path": str(WORKER.resolve()), "sha256": sha256(WORKER)},
            "replay": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__)),
            },
        },
        "equivalence": {
            "row_classifications_exact": True,
            "aggregate_exact": True,
            "checked_before_write": True,
        },
        "artifacts": {
            path.name: {"sha256": sha256(path)}
            for path in (
                certificates_path,
                rows_path,
                depth_path,
                summary_path,
                report_path,
            )
        },
    }
    manifest_path = output_dir / "projection_manifest.json"
    _write_exclusive(manifest_path, manifest)
    _write_exclusive(
        output_dir / "projection_manifest.sha256",
        (sha256(manifest_path) + "  projection_manifest.json\n").encode("utf-8"),
    )
    return summary_path, certificates_path, report_path


def _sealed_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    fields = (
        "equality_rows_seen",
        "pair_candidate_count",
        "parsed_rational_pair_count",
        "verified_rational_identity_count",
        "exact_nonidentity_witness_count",
        "universal_identity_counterexample_count",
        "symbolically_unresolved_count",
        "parse_noncoverage_count",
        "positive_code_witness_count",
        "criterion_defect_witness_count",
    )
    counts: Counter[str] = Counter()
    for row in rows:
        analysis = row["analysis"]
        counts["rows"] += 1
        counts["rows_abstained"] += int(analysis["abstained"])
        counts["rows_with_executable_pair"] += int(not analysis["abstained"])
        counts["rows_with_identity_witness"] += int(
            analysis["verified_rational_identity_count"] > 0
        )
        counts["rows_with_exact_nonidentity_witness"] += int(
            analysis["exact_nonidentity_witness_count"] > 0
        )
        counts["rows_pair_budget_exhausted"] += int(
            analysis["pair_budget_exhausted"]
        )
        for field in fields:
            counts[field] += int(analysis[field])
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--process-timeout", type=float, default=900.0)
    args = parser.parse_args()
    summary, certificates, report = replay_pair_certificates(
        preparation_dir=args.preparation_dir,
        execution_dir=args.execution_dir,
        output_dir=args.output_dir,
        process_timeout=args.process_timeout,
    )
    print(
        json.dumps(
            {
                "summary": str(summary),
                "certificates": str(certificates),
                "report": str(report),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
