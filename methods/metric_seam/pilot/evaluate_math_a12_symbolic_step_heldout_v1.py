#!/usr/bin/env python3
"""Two-phase held-out execution for the frozen Math-a12 symbolic relation.

``execute`` reconstructs the seed-7 held-out complement, applies the same
credential sanitizer used at preparation time, and runs the already-pinned
symbolic operation in a subprocess that sees only opaque aliases and ctext.
It cannot accept or read a prompt-reference path.

``finalize`` first verifies that immutable execution, then opens the stored
two-pass a12 prompt reference.  It reports reference availability and prompt
reliability only.  There is intentionally no candidate scalar, correlation,
parent aggregation, or isomorphism verdict: the executable output is a set of
relation-local identity/nonidentity witnesses plus explicit abstentions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from methods.metric_seam.battery.seal_ctext_items_v2 import canonical_bytes, sha256
from methods.metric_seam.battery.seal_ctext_train_view_v3 import (
    CREDENTIAL_PATTERNS,
    sanitize_ctext,
)
from methods.metric_seam.pilot._math_a12_symbolic_step_worker_v1 import (
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("_math_a12_symbolic_step_worker_v1.py")

PREP_SCHEMA = "metric-seam.sanitized-ctext-train-preparation-manifest.v2"
BUNDLE_SCHEMA = "metric-seam.sanitized-ctext-train-compiler-view.v2"
EXECUTION_SCHEMA = "metric-seam.math-a12-symbolic-heldout-execution.v1"
MANIFEST_SCHEMA = "metric-seam.math-a12-symbolic-heldout-manifest.v1"
FINALIZATION_SCHEMA = "metric-seam.math-a12-symbolic-heldout-finalization.v1"

COUNT_FIELDS = (
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


class HeldoutIntegrityError(RuntimeError):
    """The frozen relation cannot be evaluated without violating an invariant."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _path_from_record(record: Mapping[str, Any]) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise HeldoutIntegrityError("manifest has no recorded input path")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _verify_record(record: Mapping[str, Any], *, expected: Path | None = None) -> Path:
    path = _path_from_record(record)
    if expected is not None and path != expected.resolve():
        raise HeldoutIntegrityError(f"recorded path differs from expected input: {path}")
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise HeldoutIntegrityError(f"recorded input changed or is unavailable: {path}")
    return path


def _verify_preparation(preparation_dir: Path) -> tuple[dict, dict, Path]:
    manifest_path = preparation_dir / "prepare_manifest.json"
    bundle_path = preparation_dir / "compiler_bundle.json"
    manifest = _load_json(manifest_path)
    bundle = _load_json(bundle_path)
    if manifest.get("schema") != PREP_SCHEMA or bundle.get("schema") != BUNDLE_SCHEMA:
        raise HeldoutIntegrityError("unexpected a12 preparation schema")
    if manifest.get("task") != "math" or manifest.get("criterion_id") != "a12":
        raise HeldoutIntegrityError("preparation is not Math a12")
    if bundle.get("task") != "math" or bundle.get("criterion_id") != "a12":
        raise HeldoutIntegrityError("compiler bundle is not Math a12")
    artifact = manifest.get("artifacts", {}).get("compiler_bundle.json", {})
    if sha256(bundle_path) != artifact.get("sha256"):
        raise HeldoutIntegrityError("compiler bundle changed after preparation")
    if artifact.get("allowed_item_keys") != ["ctext", "item_key"]:
        raise HeldoutIntegrityError("compiler row allowlist changed")

    policy = manifest.get("policy", {})
    required_policy = {
        "compiler_receives_heldout": False,
        "compiler_receives_reference_values": False,
        "external_supervised_anchor": None,
        "model_calls": False,
        "gpu_used": False,
    }
    for key, expected in required_policy.items():
        if expected is not None and policy.get(key) is not expected:
            raise HeldoutIntegrityError(f"preparation policy failed: {key}")
    objective = bundle.get("objective", {})
    if objective.get("external_supervised_anchor") is not False:
        raise HeldoutIntegrityError("preparation objective admits an external anchor")
    interface = bundle.get("interface", {})
    if interface.get("reference_values_available") is not False:
        raise HeldoutIntegrityError("compiler bundle exposes reference values")
    if interface.get("heldout_items_available") is not False:
        raise HeldoutIntegrityError("compiler bundle exposes held-out rows")

    implementation = manifest.get("implementation", {})
    _verify_record(
        implementation.get("symbolic_operation", {}),
        expected=ROOT / "methods/metric_seam/hybrids/ops_symbolic_steps_v1.py",
    )
    _verify_record(
        implementation.get("relation_contract", {}),
        expected=(
            ROOT
            / "methods/metric_seam/contracts/"
            "math_a12_symbolic_step_relation_contract_v1.json"
        ),
    )
    source_path = _verify_record(manifest.get("inputs", {}).get("source", {}))
    _verify_record(manifest.get("inputs", {}).get("projected_contract", {}))
    return manifest, bundle, source_path


def _summarize_redactions(
    redactions: Mapping[str, Mapping[str, int]], ids: Iterable[str]
) -> dict[str, Any]:
    selected = list(ids)
    categories = [pattern.category for pattern in CREDENTIAL_PATTERNS]
    totals = {
        category: sum(redactions[datapoint_id][category] for datapoint_id in selected)
        for category in categories
    }
    return {
        "row_count": len(selected),
        "changed_row_count": sum(any(redactions[datapoint_id].values()) for datapoint_id in selected),
        "category_counts": totals,
        "total_matches": sum(totals.values()),
    }


def _reconstruct_partition(
    manifest: Mapping[str, Any], bundle: Mapping[str, Any], source_path: Path
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Any]]:
    raw = _load_json(source_path)
    if not isinstance(raw, list) or not raw:
        raise HeldoutIntegrityError("pinned Math source is not a non-empty list")
    by_id: dict[str, str] = {}
    redactions: dict[str, dict[str, int]] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise HeldoutIntegrityError(f"source row {index} is not an object")
        datapoint_id, ctext = row.get("datapoint_id"), row.get("ctext")
        if not isinstance(datapoint_id, str) or not datapoint_id or datapoint_id in by_id:
            raise HeldoutIntegrityError("source identifiers must be unique strings")
        if not isinstance(ctext, str):
            raise HeldoutIntegrityError(f"source row {datapoint_id!r} has no ctext")
        sanitized, counts = sanitize_ctext(ctext)
        by_id[datapoint_id] = sanitized
        redactions[datapoint_id] = counts

    partition = manifest.get("partition", {})
    expected_algorithm = (
        "sorted datapoint_id; random.Random(seed).shuffle; first train_count"
    )
    if partition.get("algorithm") != expected_algorithm:
        raise HeldoutIntegrityError("unrecognized frozen split algorithm")
    ids = sorted(by_id)
    random.Random(int(partition["seed"])).shuffle(ids)
    train_count = int(partition["train_count"])
    if len(ids) != int(partition["corpus_count"]):
        raise HeldoutIntegrityError("source count changed after preparation")
    train_ids = set(ids[:train_count])
    heldout_ids = set(ids[train_count:])
    train = [(dpid, by_id[dpid]) for dpid in sorted(train_ids)]
    heldout = [(dpid, by_id[dpid]) for dpid in sorted(heldout_ids)]
    if len(heldout) != int(partition["heldout_count"]):
        raise HeldoutIntegrityError("held-out complement has the wrong size")

    compiler_rows = bundle.get("train_items")
    if not isinstance(compiler_rows, list) or len(compiler_rows) != len(train):
        raise HeldoutIntegrityError("compiler TRAIN count differs from reconstructed split")
    for index, (compiler_row, (_, ctext)) in enumerate(zip(compiler_rows, train), 1):
        if (
            not isinstance(compiler_row, dict)
            or set(compiler_row) != {"ctext", "item_key"}
            or compiler_row.get("item_key") != f"train_{index:04d}"
            or compiler_row.get("ctext") != ctext
        ):
            raise HeldoutIntegrityError("reconstructed TRAIN view differs from frozen bundle")

    heldout_redactions = _summarize_redactions(redactions, heldout_ids)
    prepared_redactions = manifest.get("credential_redaction", {}).get("heldout")
    if heldout_redactions != prepared_redactions:
        raise HeldoutIntegrityError("held-out sanitizer receipt differs from preparation")
    return train, heldout, heldout_redactions


def _worker_request(heldout: Sequence[tuple[str, str]]) -> tuple[dict, dict[str, str]]:
    alias_to_id = {
        f"heldout_{index:04d}": datapoint_id
        for index, (datapoint_id, _) in enumerate(heldout, 1)
    }
    request = {
        "schema": REQUEST_SCHEMA,
        "relation_id": "explicit_rational_equality_preservation",
        "reference_values_present": False,
        "source_identifiers_present": False,
        "parent_scalar_requested": False,
        "eval_items": [
            {"item_key": alias, "ctext": ctext}
            for alias, (_, ctext) in zip(alias_to_id, heldout)
        ],
    }
    return request, alias_to_id


def _run_worker(request: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="metric_seam_math_a12_heldout_") as name:
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
            message = (process.stderr or process.stdout or "symbolic worker failed")[-2000:]
            raise HeldoutIntegrityError(
                f"symbolic held-out worker exited {process.returncode}: {message}"
            )
        result = _load_json(result_path)
    if result.get("schema") != RESULT_SCHEMA:
        raise HeldoutIntegrityError("symbolic worker returned an unexpected schema")
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        for field in COUNT_FIELDS:
            counts[field] += int(analysis[field])
    return dict(sorted(counts.items()))


def execute_heldout(
    *, preparation_dir: Path, output_dir: Path, process_timeout: float = 900.0
) -> tuple[Path, Path]:
    """Execute and seal candidate evidence without accepting a reference path."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite held-out execution {output_dir}")
    manifest, bundle, source_path = _verify_preparation(preparation_dir)
    _, heldout, heldout_redactions = _reconstruct_partition(
        manifest, bundle, source_path
    )
    request, alias_to_id = _worker_request(heldout)
    started_at = _utc_now()
    worker_result = _run_worker(request, timeout=process_timeout)
    completed_at = _utc_now()

    outputs = worker_result.get("outputs")
    expected_aliases = list(alias_to_id)
    if not isinstance(outputs, list) or [row.get("item_key") for row in outputs] != expected_aliases:
        raise HeldoutIntegrityError("worker output aliases differ from the sealed request")
    public_rows = [
        {
            "datapoint_id": alias_to_id[row["item_key"]],
            "analysis": row["analysis"],
        }
        for row in outputs
    ]
    summary = _aggregate(public_rows)
    if summary["criterion_defect_witness_count"] != 0:
        raise HeldoutIntegrityError("scope-free run emitted a criterion-level defect")
    if summary["universal_identity_counterexample_count"] != 0:
        raise HeldoutIntegrityError("scope-free run emitted a universal counterexample")

    execution = {
        "schema": EXECUTION_SCHEMA,
        "task": "math",
        "criterion_id": "a12",
        "relation_id": "explicit_rational_equality_preservation",
        "objective": "unsupervised reconstruction; relation-local code verification",
        "heldout_count": len(heldout),
        "candidate_started_at": started_at,
        "candidate_completed_at": completed_at,
        "reference_accessed": False,
        "model_calls": False,
        "gpu_used": False,
        "whole_criterion_fidelity": "UNAVAILABLE",
        "whole_criterion_scalar": None,
        "candidate_reference_correlation": None,
        "summary": summary,
        "rows": public_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    execution_path = output_dir / "candidate_execution.json"
    _write_exclusive(execution_path, execution)
    execution_manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": _utc_now(),
        "preparation_run_id": manifest.get("run_id"),
        "inputs": {
            "preparation_manifest": {
                "path": str((preparation_dir / "prepare_manifest.json").resolve()),
                "sha256": sha256(preparation_dir / "prepare_manifest.json"),
            },
            "compiler_bundle": {
                "path": str((preparation_dir / "compiler_bundle.json").resolve()),
                "sha256": sha256(preparation_dir / "compiler_bundle.json"),
            },
            "source": {
                "path": str(source_path.resolve()),
                "sha256": sha256(source_path),
            },
            "symbolic_operation": manifest["implementation"]["symbolic_operation"],
            "relation_contract": manifest["implementation"]["relation_contract"],
        },
        "execution": {
            "candidate_started_at": started_at,
            "candidate_completed_at": completed_at,
            "heldout_identifiers_sent_to_candidate": False,
            "reference_values_sent_to_candidate": False,
            "parent_scalar_requested": False,
            "request_sha256": _sha256_bytes(canonical_bytes(request)),
            "worker": {
                "path": str(WORKER.resolve()),
                "sha256": sha256(WORKER),
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__)),
            },
        },
        "sanitizer": {
            "same_as_preparation": True,
            "heldout_receipt": heldout_redactions,
        },
        "policy": {
            "reference_accessed": False,
            "external_supervised_anchor": False,
            "model_calls": False,
            "gpu_used": False,
            "whole_criterion_scalar": None,
            "candidate_reference_correlation": None,
            "projection_boundary_not_os_sandbox": True,
        },
        "artifacts": {
            execution_path.name: {
                "sha256": sha256(execution_path),
                "heldout_count": len(heldout),
            }
        },
    }
    manifest_path = output_dir / "execution_manifest.json"
    _write_exclusive(manifest_path, execution_manifest)
    _write_exclusive(
        output_dir / "execution_manifest.sha256",
        (sha256(manifest_path) + "  execution_manifest.json\n").encode("utf-8"),
    )
    return execution_path, manifest_path


def _verify_execution(execution_dir: Path) -> tuple[dict, dict]:
    execution_path = execution_dir / "candidate_execution.json"
    manifest_path = execution_dir / "execution_manifest.json"
    execution = _load_json(execution_path)
    manifest = _load_json(manifest_path)
    if execution.get("schema") != EXECUTION_SCHEMA or manifest.get("schema") != MANIFEST_SCHEMA:
        raise HeldoutIntegrityError("unexpected symbolic held-out execution schema")
    record = manifest.get("artifacts", {}).get(execution_path.name, {})
    if sha256(execution_path) != record.get("sha256"):
        raise HeldoutIntegrityError("candidate execution changed after sealing")
    receipt = (execution_dir / "execution_manifest.sha256").read_text(encoding="utf-8")
    if receipt != f"{sha256(manifest_path)}  execution_manifest.json\n":
        raise HeldoutIntegrityError("execution manifest receipt does not match")
    policy = manifest.get("policy", {})
    if policy.get("reference_accessed") is not False:
        raise HeldoutIntegrityError("execution claims prior reference access")
    if execution.get("reference_accessed") is not False:
        raise HeldoutIntegrityError("candidate result claims prior reference access")
    if execution.get("whole_criterion_scalar") is not None:
        raise HeldoutIntegrityError("candidate execution contains a parent scalar")
    if execution.get("candidate_reference_correlation") is not None:
        raise HeldoutIntegrityError("candidate execution contains a correlation")
    return execution, manifest


def _load_prompt_reference(
    reference_path: Path, heldout_ids: set[str]
) -> tuple[dict[str, int], dict[str, int]]:
    pass1: dict[str, int] = {}
    pass2: dict[str, int] = {}
    with reference_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("aspect_id") != "a12" or row.get("datapoint_id") not in heldout_ids:
                continue
            channel, score = row.get("channel"), row.get("score")
            if channel not in {"pass1", "pass2"} or type(score) is not int:
                continue
            if not 0 <= score <= 10:
                raise HeldoutIntegrityError(
                    f"invalid a12 prompt score at reference line {line_number}"
                )
            target = pass1 if channel == "pass1" else pass2
            datapoint_id = row["datapoint_id"]
            if datapoint_id in target:
                raise HeldoutIntegrityError(
                    f"duplicate {channel} a12 reference for {datapoint_id}"
                )
            target[datapoint_id] = score
    return pass1, pass2


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    left = 0
    while left < len(order):
        right = left
        while right + 1 < len(order) and values[order[right + 1]] == values[order[left]]:
            right += 1
        rank = (left + right) / 2 + 1
        for index in range(left, right + 1):
            ranks[order[index]] = rank
        left = right + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if not left_scale or not right_scale:
        return None
    return numerator / (left_scale * right_scale)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _reference_intersections(
    rows: Sequence[Mapping[str, Any]], available: set[str]
) -> dict[str, int]:
    predicates = {
        "all_heldout": lambda analysis: True,
        "executable_relation_rows": lambda analysis: not analysis["abstained"],
        "abstention_rows": lambda analysis: analysis["abstained"],
        "identity_witness_rows": lambda analysis: (
            analysis["verified_rational_identity_count"] > 0
        ),
        "exact_nonidentity_witness_rows": lambda analysis: (
            analysis["exact_nonidentity_witness_count"] > 0
        ),
    }
    return {
        name: sum(
            row["datapoint_id"] in available and predicate(row["analysis"])
            for row in rows
        )
        for name, predicate in predicates.items()
    }


def _render_report(finalization: Mapping[str, Any]) -> str:
    summary = finalization["candidate_execution_summary"]
    reference = finalization["prompt_reference"]
    reliability = reference["two_pass_spearman"]
    reliability_text = "UNAVAILABLE" if reliability is None else f"{reliability:.3f}"
    return f"""# Math a12 symbolic relation — sealed held-out report

## Outcome

The frozen symbolic operation executed on all **{finalization['heldout_count']}** seed-7
held-out rows before the stored prompt reference was opened. It found at least one executable
rational equality pair on **{summary['rows_with_executable_pair']}** rows and abstained on
**{summary['rows_abstained']}**. It emitted **{summary['verified_rational_identity_count']}**
exact identity witnesses and **{summary['exact_nonidentity_witness_count']}** exact
nonidentity witnesses. Those occurred on {summary['rows_with_identity_witness']} and
{summary['rows_with_exact_nonidentity_witness']} rows, respectively.

The prompt reference is available on **{reference['available_both_passes']} / {finalization['heldout_count']}**
held-out rows; its two-pass Spearman reliability is **{reliability_text}**.
Reference loading began only after the sealed candidate execution completed.

## Licensed claim

This is positive **relation-instance code verification** for bounded rational-algebra equality
steps. An identity witness certifies equality on the emitted denominator-nonzero domain. An
exact nonidentity witness establishes only that the pair is not a rational identity; without a
separately frozen universal-scope judgment it is not a document defect.

No parent a12 scalar was defined, fitted, or emitted. No candidate/reference correlation was
computed. Whole-proof rigor, whole-criterion reconstruction, and isomorphism are therefore
**NOT ESTIMATED**. Parse noncoverage and rows without an executable pair are abstentions, never
negative evidence or evidence of tacitness. The stored prompt judgement is a reconstruction
reference, not external ground truth.

No model, API, external supervised anchor, or GPU was used.
"""


def finalize_reference(
    *, execution_dir: Path, reference_path: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Open the prompt reference only after verifying sealed candidate execution."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite finalization {output_dir}")
    execution, manifest = _verify_execution(execution_dir)
    rows = execution.get("rows")
    if not isinstance(rows, list) or len(rows) != execution.get("heldout_count"):
        raise HeldoutIntegrityError("candidate execution rows are incomplete")
    heldout_ids = {row["datapoint_id"] for row in rows}
    completed_at = datetime.fromisoformat(execution["candidate_completed_at"])

    reference_load_started_at = _utc_now()
    if datetime.fromisoformat(reference_load_started_at) < completed_at:
        raise HeldoutIntegrityError("reference load precedes candidate completion")
    pass1, pass2 = _load_prompt_reference(reference_path, heldout_ids)
    reference_loaded_at = _utc_now()
    common = sorted(set(pass1) & set(pass2))
    reliability = _spearman(
        [float(pass1[datapoint_id]) for datapoint_id in common],
        [float(pass2[datapoint_id]) for datapoint_id in common],
    )

    finalization = {
        "schema": FINALIZATION_SCHEMA,
        "task": "math",
        "criterion_id": "a12",
        "relation_id": "explicit_rational_equality_preservation",
        "objective": "unsupervised reconstruction with a stored LLM reference",
        "heldout_count": execution["heldout_count"],
        "candidate_execution_summary": execution["summary"],
        "prompt_reference": {
            "reference_load_started_at": reference_load_started_at,
            "reference_loaded_at": reference_loaded_at,
            "candidate_completed_before_reference_load": True,
            "available_pass1": len(pass1),
            "available_pass2": len(pass2),
            "available_both_passes": len(common),
            "two_pass_spearman": reliability,
            "relation_row_intersections": _reference_intersections(rows, set(common)),
            "role": "frozen prompt-judgement reconstruction reference; not external truth",
        },
        "candidate_parent_scalar": None,
        "candidate_reference_correlation": None,
        "whole_criterion_reconstruction": "NOT_ESTIMATED",
        "isomorphism": "NOT_ESTIMATED",
        "whole_criterion_fidelity": "UNAVAILABLE",
        "verifiability": "relation_instance_witnesses_only",
        "articulability": "prompt reference available; relation-matched prompt arm not run",
        "claim_boundary": (
            "Exact identity/nonidentity witnesses are relation-local. Nonidentity is not a "
            "document defect without a separately frozen universal-scope judgment."
        ),
        "negative_result_policy": (
            "parse noncoverage and no executable pair are abstentions; neither supports "
            "a tacitness claim"
        ),
        "external_supervised_anchor": False,
        "model_calls": False,
        "gpu_used": False,
        "inputs": {
            "candidate_execution": {
                "path": str((execution_dir / "candidate_execution.json").resolve()),
                "sha256": sha256(execution_dir / "candidate_execution.json"),
            },
            "execution_manifest": {
                "path": str((execution_dir / "execution_manifest.json").resolve()),
                "sha256": sha256(execution_dir / "execution_manifest.json"),
                "preparation_run_id": manifest.get("preparation_run_id"),
            },
            "prompt_reference": {
                "path": str(reference_path.resolve()),
                "sha256": sha256(reference_path),
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    finalization_path = output_dir / "finalization.json"
    _write_exclusive(finalization_path, finalization)
    report_path = output_dir / "REPORT.md"
    _write_exclusive(report_path, _render_report(finalization).encode("utf-8"))
    _write_exclusive(
        output_dir / "finalization.sha256",
        (sha256(finalization_path) + "  finalization.json\n").encode("utf-8"),
    )
    return finalization_path, report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--preparation-dir", type=Path, required=True)
    execute_parser.add_argument("--output-dir", type=Path, required=True)
    execute_parser.add_argument("--process-timeout", type=float, default=900.0)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--execution-dir", type=Path, required=True)
    finalize_parser.add_argument("--reference", type=Path, required=True)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "execute":
        execution, manifest = execute_heldout(
            preparation_dir=args.preparation_dir,
            output_dir=args.output_dir,
            process_timeout=args.process_timeout,
        )
        print(json.dumps({"execution": str(execution), "manifest": str(manifest)}))
    else:
        finalization, report = finalize_reference(
            execution_dir=args.execution_dir,
            reference_path=args.reference,
            output_dir=args.output_dir,
        )
        print(json.dumps({"finalization": str(finalization), "report": str(report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
