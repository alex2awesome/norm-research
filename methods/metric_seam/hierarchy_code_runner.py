"""Execute audited hierarchy-program seeds without loading reference judgments.

This is the code arm of the hierarchy reconstruction lane.  It accepts only
construct-fidelity rows already adjudicated as ``exact`` or ``partial`` and the
official shared item panel containing exactly ``item_key`` and ``ctext``.
Trusted historical candidates run in separate processes with an allowlisted
environment, credentials removed, and accelerator devices masked.  This is
process isolation, not an OS filesystem/network sandbox.  The resulting scalar
measurements are relation-local evidence; they are not, by themselves,
whole-construct verification or reconstruction.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import numbers
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

from methods.existing_metrics_runner.coded.sandbox import ALLOWED_TOOLS
from methods.metric_seam.hierarchy_fidelity_merge import SCHEMA as FIDELITY_SCHEMA
from methods.metric_seam.hierarchy_items import validate_task_items


ROOT = Path(__file__).resolve().parents[2]
METRICS_ROOT = ROOT / "methods/existing_metrics_runner/coded/metrics"
CANONICAL_ITEMS_ROOT = (
    ROOT / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/code-review"
)
EXECUTION_SCHEMA = "metric-seam.hierarchy-code-execution.v1"
TRAIN_GATE_SCHEMA = "metric-seam.hierarchy-code-train-gate.v1"
ALLOWED_PHASES = {"compiler_train", "heldout_pre_reference"}
PHASE_FILE = {
    "compiler_train": "compiler_train.json",
    "heldout_pre_reference": "sealed_heldout.json",
}
ELIGIBLE_VERDICTS = {"exact", "partial"}

_AUDIT_TOP_FIELDS = {
    "schema", "status", "task", "design_scope", "source_seed_map",
    "source_level_audits", "source_cross_audit_adjudication", "n_adjudicated_changes",
    "panel_content_sha256", "hierarchy_frame",
    "execution_performed", "reference_values_loaded", "outcome_labels_loaded",
    "external_supervision", "depth_vocabulary", "interpretation", "summary", "rows",
}
_AUDIT_ROW_FIELDS = {
    "cell_id", "level", "metric_name", "metric_description", "candidate",
    "requested_relation", "implemented_relations", "verdict", "scope",
    "eligible_for_relation_local_execution", "audited_depth",
    "dependency_applicability_caveats", "rationale", "interpretation",
}


class ExecutionInputError(ValueError):
    """Raised when an input would break the pre-reference replay contract."""


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_items(payload: object) -> list[dict[str, str]]:
    """Require a label-free item list and return a normalized copy."""

    if not isinstance(payload, list) or not payload:
        raise ExecutionInputError("items must be a non-empty JSON list")
    rows: list[dict[str, str]] = []
    seen = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ExecutionInputError(f"item {index} is not an object")
        if set(row) != {"item_key", "ctext"}:
            raise ExecutionInputError(
                f"item {index} must contain exactly item_key and ctext; found {sorted(row)}"
            )
        item_key, ctext = row["item_key"], row["ctext"]
        if not isinstance(item_key, str) or not item_key or item_key in seen:
            raise ExecutionInputError(f"item {index} has an invalid or duplicate item_key")
        if not isinstance(ctext, str) or not ctext:
            raise ExecutionInputError(f"item {index} has empty/non-string ctext")
        seen.add(item_key)
        rows.append({"item_key": item_key, "ctext": ctext})
    return rows


def load_bound_items(items_root: Path, phase: str, *, require_canonical: bool = True) -> tuple[list[dict], Path]:
    """Validate both official splits, then return the phase-bound split."""

    if phase not in ALLOWED_PHASES:
        raise ExecutionInputError(f"phase must be one of {sorted(ALLOWED_PHASES)}")
    resolved = items_root.resolve()
    if require_canonical and resolved != CANONICAL_ITEMS_ROOT.resolve():
        raise ExecutionInputError(
            f"official replay requires canonical items root {CANONICAL_ITEMS_ROOT}"
        )
    manifest_path = resolved / "manifest.json"
    train_path = resolved / "compiler_train.json"
    heldout_path = resolved / "sealed_heldout.json"
    manifest = _load_json(manifest_path)
    train = validate_items(_load_json(train_path))
    heldout = validate_items(_load_json(heldout_path))
    if manifest.get("task") != "code-review":
        raise ExecutionInputError("item manifest is not the code-review panel")
    try:
        validate_task_items(manifest, train, heldout)
    except ValueError as exc:
        raise ExecutionInputError(str(exc)) from exc
    selected_path = train_path if phase == "compiler_train" else heldout_path
    return (train if phase == "compiler_train" else heldout), selected_path


def canonical_source_path(source_path: str) -> str:
    """Return one repository-relative identity for an allowed metric module."""

    path = Path(source_path)
    absolute = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        relative = absolute.relative_to(METRICS_ROOT.resolve())
    except ValueError as exc:
        raise ExecutionInputError(f"candidate is outside the allowed metric library: {path}") from exc
    if relative.parent != Path(".") or relative.suffix != ".py":
        raise ExecutionInputError(f"candidate must be one top-level metric module: {relative}")
    if not relative.stem.startswith("a"):
        raise ExecutionInputError(f"candidate module has an invalid name: {relative.name}")
    return str(absolute.relative_to(ROOT.resolve()))


def module_name_for_source(source_path: str) -> str:
    """Resolve an allowed canonical source path to its package import name."""

    relative = Path(canonical_source_path(source_path)).relative_to(
        "methods/existing_metrics_runner/coded/metrics"
    )
    return f"methods.existing_metrics_runner.coded.metrics.{relative.stem}"


def _normalize_candidate(row: Mapping) -> tuple[str, str, str]:
    candidate = row.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ExecutionInputError(f"eligible row {row.get('cell_id')} has no candidate object")
    aspect_id = candidate.get("aspect_id")
    source_path = candidate.get("source_path")
    source_sha256 = candidate.get("source_sha256")
    if not all(isinstance(value, str) and value for value in (
        aspect_id, source_path, source_sha256
    )):
        raise ExecutionInputError(f"eligible row {row.get('cell_id')} has invalid candidate metadata")
    canonical = canonical_source_path(source_path)
    observed_sha256 = hashlib.sha256((ROOT / canonical).read_bytes()).hexdigest()
    if observed_sha256 != source_sha256:
        raise ExecutionInputError(
            f"eligible row {row.get('cell_id')} source changed after static audit"
        )
    return aspect_id, canonical, source_sha256


def validate_canonical_audit(audit: Mapping) -> None:
    """Reject permissive or outcome-bearing substitutes for the merged audit."""

    if audit.get("schema") != FIDELITY_SCHEMA:
        raise ExecutionInputError(f"expected canonical audit schema {FIDELITY_SCHEMA}")
    if set(audit) != _AUDIT_TOP_FIELDS:
        raise ExecutionInputError(
            f"canonical audit top-level fields differ: {sorted(set(audit) ^ _AUDIT_TOP_FIELDS)}"
        )
    if audit.get("execution_performed") is not False:
        raise ExecutionInputError("audit must have been completed before execution")
    if audit.get("reference_values_loaded") is not False or audit.get("outcome_labels_loaded") is not False:
        raise ExecutionInputError("audit may not contain loaded references or outcomes")
    rows = audit.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise ExecutionInputError("canonical audit must contain 90 rows")
    cell_ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _AUDIT_ROW_FIELDS:
            raise ExecutionInputError(f"audit row {index} does not match the canonical row schema")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ExecutionInputError(f"audit row {index} has invalid cell_id")
        cell_ids.append(cell_id)
        verdict, scope = row.get("verdict"), row.get("scope")
        expected_scope = {
            "exact": "whole_construct",
            "partial": "subrelation_only",
            "mismatch": "none",
            "no_candidate_bounded_non_discovery": "none",
        }.get(verdict)
        if expected_scope is None or scope != expected_scope:
            raise ExecutionInputError(f"{cell_id}: verdict/scope mismatch")
        eligible = row.get("eligible_for_relation_local_execution")
        if not isinstance(eligible, bool) or eligible != (verdict in ELIGIBLE_VERDICTS):
            raise ExecutionInputError(f"{cell_id}: verdict/eligibility mismatch")
        depth = row.get("audited_depth")
        if row.get("candidate") is None:
            if depth is not None:
                raise ExecutionInputError(f"{cell_id}: no-candidate row has a depth")
        elif isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 4:
            raise ExecutionInputError(f"{cell_id}: invalid audited depth")
        if not isinstance(row.get("requested_relation"), str) or not row["requested_relation"].strip():
            raise ExecutionInputError(f"{cell_id}: invalid requested relation")
        relations = row.get("implemented_relations")
        if not isinstance(relations, list) or not all(
            isinstance(value, str) and value.strip() for value in relations
        ):
            raise ExecutionInputError(f"{cell_id}: invalid implemented relations")
    if len(set(cell_ids)) != len(cell_ids):
        raise ExecutionInputError("canonical audit contains duplicate cell ids")


def build_execution_plan(audit: Mapping) -> list[dict]:
    """Group independently accepted relation-local rows by executable module."""

    validate_canonical_audit(audit)
    rows = audit["rows"]
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("eligible_for_relation_local_execution"):
            continue
        verdict = row.get("verdict")
        if verdict not in ELIGIBLE_VERDICTS:
            raise ExecutionInputError(
                f"row {row.get('cell_id')} is eligible but has verdict {verdict!r}"
            )
        aspect_id, source_path, source_sha256 = _normalize_candidate(row)
        relation = {
            "cell_id": row["cell_id"],
            "level": row["level"],
            "metric_name": row["metric_name"],
            "construct_fidelity_verdict": verdict,
            "scope": row["scope"],
            "requested_relation": row["requested_relation"],
            "implemented_relations": list(row["implemented_relations"]),
            "audited_depth": row["audited_depth"],
        }
        grouped.setdefault((aspect_id, source_path, source_sha256), []).append(relation)
    return [
        {
            "aspect_id": aspect_id,
            "source_path": source_path,
            "source_sha256": source_sha256,
            "relations": sorted(relations, key=lambda row: row["cell_id"]),
        }
        for (aspect_id, source_path, source_sha256), relations in sorted(grouped.items())
    ]


def apply_program_selection(plans: Sequence[Mapping], selection: Mapping) -> list[dict]:
    """Apply a training-only program gate before heldout replay."""

    if selection.get("schema") != TRAIN_GATE_SCHEMA:
        raise ExecutionInputError("heldout replay requires the canonical training gate")
    if selection.get("selection_basis") != "compiler_train_outputs_only":
        raise ExecutionInputError("program selection was not based only on compiler-train outputs")
    selected = selection.get("selected_programs")
    if not isinstance(selected, list) or not selected:
        raise ExecutionInputError("training gate selected no programs")
    plan_by_identity = {
        (plan["aspect_id"], plan["source_path"], plan["source_sha256"]): plan
        for plan in plans
    }
    observed = []
    for row in selected:
        if not isinstance(row, Mapping):
            raise ExecutionInputError("invalid selected-program row")
        identity = (row.get("aspect_id"), row.get("source_path"), row.get("source_sha256"))
        plan = plan_by_identity.get(identity)
        if plan is None:
            raise ExecutionInputError(f"selected program does not match canonical audit: {identity[0]}")
        if row.get("cell_ids") != [relation["cell_id"] for relation in plan["relations"]]:
            raise ExecutionInputError(f"selected relation mapping drift for {identity[0]}")
        observed.append(plan)
    if len({plan["aspect_id"] for plan in observed}) != len(observed):
        raise ExecutionInputError("training gate contains duplicate programs")
    return observed


def _score_status(score) -> tuple[str, float | None]:
    if score is None:
        return "abstained", None
    if isinstance(score, bool) or not isinstance(score, numbers.Real):
        return "contract_error", None
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return "contract_error", None
    return "scored", value


def execute_one_program(source_path: str, aspect_id: str, source_sha256: str,
                        items: Sequence[Mapping]) -> dict:
    """Run one trusted historical module in the current worker process."""

    module_name = module_name_for_source(source_path)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # import failure is a measured bounded failure
        return {
            "aspect_id": aspect_id,
            "source_path": source_path,
            "worker_status": "import_error",
            "measurement_status": "not_measured",
            "error_type": type(exc).__name__,
            "rows": [],
            "summary": {"n_items": len(items), "n_import_errors": 1},
        }
    imported_path = Path(module.__file__).resolve()
    expected_path = (ROOT / canonical_source_path(source_path)).resolve()
    if imported_path != expected_path:
        raise ExecutionInputError(
            f"candidate import resolved to {imported_path}, expected {expected_path}"
        )
    if hashlib.sha256(imported_path.read_bytes()).hexdigest() != source_sha256:
        raise ExecutionInputError(f"{aspect_id} source changed after construct audit")
    if getattr(module, "ASPECT_ID", None) != aspect_id:
        raise ExecutionInputError(
            f"candidate id mismatch: audit says {aspect_id}, module says "
            f"{getattr(module, 'ASPECT_ID', None)!r}"
        )
    if not callable(getattr(module, "applies", None)) or not callable(
        getattr(module, "score", None)
    ):
        raise ExecutionInputError(f"{aspect_id} does not implement applies()/score()")

    outputs = []
    for item in items:
        try:
            applies_raw = module.applies(item["ctext"])
            if not isinstance(applies_raw, bool):
                outputs.append(
                    {
                        "item_key": item["item_key"],
                        "applies": None,
                        "score": None,
                        "status": "contract_error",
                        "error_type": "NonBooleanApplies",
                    }
                )
                continue
            if not applies_raw:
                outputs.append(
                    {
                        "item_key": item["item_key"],
                        "applies": False,
                        "score": None,
                        "status": "not_applicable",
                    }
                )
                continue
            status, value = _score_status(module.score(item["ctext"]))
            outputs.append(
                {
                    "item_key": item["item_key"],
                    "applies": True,
                    "score": value,
                    "status": status,
                }
            )
        except Exception as exc:  # one bad item must not poison other rows
            outputs.append(
                {
                    "item_key": item["item_key"],
                    "applies": None,
                    "score": None,
                    "status": "execution_error",
                    "error_type": type(exc).__name__,
                }
            )

    statuses = Counter(row["status"] for row in outputs)
    scores = [row["score"] for row in outputs if row["status"] == "scored"]
    summary = {
        "n_items": len(outputs),
        "status_counts": dict(sorted(statuses.items())),
        "n_applies": sum(row["applies"] is True for row in outputs),
        "n_scored": len(scores),
        "coverage": round(len(scores) / len(outputs), 6) if outputs else 0.0,
        "n_unique_scores": len(set(scores)),
        "nonconstant_when_scored": len(set(scores)) >= 2,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }
    failure_count = statuses["execution_error"] + statuses["contract_error"]
    if failure_count:
        measurement_status = (
            "item_failures_with_measurements" if scores else "item_failures_no_measurements"
        )
    elif not scores:
        measurement_status = "no_usable_scores"
    elif len(set(scores)) < 2:
        measurement_status = "constant_measurement"
    else:
        measurement_status = "nondegenerate_measurement"
    return {
        "aspect_id": aspect_id,
        "source_path": source_path,
        "worker_status": "completed",
        "measurement_status": measurement_status,
        "rows": outputs,
        "summary": summary,
    }


def _controlled_tool_path() -> str:
    directories = {str(Path(sys.executable).resolve().parent), "/usr/bin", "/bin", "/usr/sbin", "/sbin"}
    for tool in sorted(ALLOWED_TOOLS):
        found = shutil.which(tool)
        if found:
            directories.add(str(Path(found).resolve().parent))
    return os.pathsep.join(sorted(directories))


def worker_environment(home: Path) -> dict[str, str]:
    """Build an allowlisted environment; inherit no provider credentials."""

    env = {
        "HOME": str(home),
        "PATH": _controlled_tool_path(),
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MPLCONFIGDIR": str(home / ".matplotlib"),
        "TMPDIR": str(home / "tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "HIP_VISIBLE_DEVICES": "-1",
        "ROCR_VISIBLE_DEVICES": "-1",
    }
    (home / "tmp").mkdir(parents=True, exist_ok=True)
    return env


def _worker_failure(plan: Mapping, status: str, n_items: int, error_type: str | None = None) -> dict:
    result = {
        "aspect_id": plan["aspect_id"],
        "source_path": plan["source_path"],
        "worker_status": status,
        "measurement_status": "not_measured",
        "rows": [],
        "summary": {"n_items": n_items},
    }
    if error_type:
        result["error_type"] = error_type
    return result


def _validate_worker_result(result: Mapping, plan: Mapping,
                            expected_item_keys: Sequence[str]) -> None:
    if result.get("aspect_id") != plan["aspect_id"] or result.get("source_path") != plan["source_path"]:
        raise ExecutionInputError(f"worker identity mismatch for {plan['aspect_id']}")
    if result.get("worker_status") != "completed":
        return
    rows = result.get("rows")
    if not isinstance(rows, list) or [row.get("item_key") for row in rows] != list(expected_item_keys):
        raise ExecutionInputError(f"worker item-set mismatch for {plan['aspect_id']}")


def _run_worker(plan: Mapping, items: Sequence[Mapping], *, timeout_seconds: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="metric-seam-code-") as directory:
        worker_home = Path(directory)
        out = worker_home / "result.json"
        command = [
            sys.executable,
            "-m",
            "methods.metric_seam.hierarchy_code_runner",
            "worker",
            "--source-path",
            str(plan["source_path"]),
            "--aspect-id",
            str(plan["aspect_id"]),
            "--source-sha256",
            str(plan["source_sha256"]),
            "--out",
            str(out),
        ]
        payload = json.dumps(list(items), sort_keys=True, ensure_ascii=False)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=worker_environment(worker_home),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(input=payload, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            return _worker_failure(plan, "program_timeout", len(items))
        if process.returncode != 0 or not out.exists():
            return _worker_failure(
                plan, "worker_error", len(items), f"WorkerExit{process.returncode}"
            )
        result = _load_json(out)
        _validate_worker_result(result, plan, [row["item_key"] for row in items])
        return result


def execute_audit(audit: Mapping, items_root: Path, *, phase: str,
                  timeout_seconds: float = 600.0,
                  require_canonical_items: bool = True,
                  audit_source_path: str | None = None,
                  program_selection: Mapping | None = None,
                  program_selection_source: str | None = None) -> dict:
    """Execute every unique accepted program and attach its audited relations."""

    items, items_path = load_bound_items(
        items_root, phase, require_canonical=require_canonical_items
    )
    plans = build_execution_plan(audit)
    if phase == "heldout_pre_reference":
        if program_selection is None:
            raise ExecutionInputError("heldout replay requires a frozen compiler-train gate")
        plans = apply_program_selection(plans, program_selection)
    elif program_selection is not None:
        raise ExecutionInputError("compiler-train execution may not use an output-derived program gate")
    programs = []
    for plan in plans:
        result = _run_worker(plan, items, timeout_seconds=timeout_seconds)
        result["source_sha256"] = plan["source_sha256"]
        result["relations"] = plan["relations"]
        programs.append(result)
    worker_statuses = Counter(program["worker_status"] for program in programs)
    measurement_statuses = Counter(program["measurement_status"] for program in programs)
    relation_rows = [relation for plan in plans for relation in plan["relations"]]
    nondegenerate_ids = {
        program["aspect_id"] for program in programs
        if program["measurement_status"] == "nondegenerate_measurement"
    }
    vector_identities = {
        json.dumps(
            [(row.get("status"), row.get("score")) for row in program.get("rows", [])],
            separators=(",", ":"),
        )
        for program in programs if program.get("rows")
    }
    if not programs:
        overall_status = "no_eligible_programs"
    elif worker_statuses["completed"] != len(programs):
        overall_status = "replay_finished_with_worker_failures"
    else:
        overall_status = "worker_replay_complete"
    return {
        "schema": EXECUTION_SCHEMA,
        "status": overall_status,
        "phase": phase,
        "items_path": str(items_path),
        "n_items": len(items),
        "construct_fidelity_source": audit_source_path,
        "program_selection_source": program_selection_source,
        "reference_fields_passed_to_worker": False,
        "outcome_fields_passed_to_worker": False,
        "audit_fields_passed_to_worker": [
            "candidate aspect_id/source identity"
        ],
        "credentials_inherited_by_worker": False,
        "accelerators_visible_to_worker": False,
        "worker_process_isolated": True,
        "worker_filesystem_isolated": False,
        "worker_network_isolated": False,
        "candidate_trust_model": "trusted historical repository programs, not adversarial code",
        "external_tool_environment": (
            "PATH reduced to directories containing the allowlisted tools; executable versions "
            "and candidate dependency closure are not pinned by this artifact"
        ),
        "execution_provenance": "retrospective historical program seed replay",
        "interpretation": (
            "Scalar outputs are relation-local code measurements. Partial matches do not verify "
            "the whole construct, and replay does not establish prompt reconstruction or isomorphism. "
            "This runner establishes official-input and process provenance, not an adversarial OS sandbox."
        ),
        "summary": {
            "n_unique_programs": len(programs),
            "n_planned_relation_mappings": len(relation_rows),
            "n_unique_scalar_vectors": len(vector_identities),
            "n_relation_mappings_with_nondegenerate_measurement": sum(
                plan["aspect_id"] in nondegenerate_ids
                for plan in plans for _relation in plan["relations"]
            ),
            "relation_verdict_counts": dict(sorted(Counter(
                row["construct_fidelity_verdict"] for row in relation_rows
            ).items())),
            "worker_status_counts": dict(sorted(worker_statuses.items())),
            "measurement_status_counts": dict(sorted(measurement_statuses.items())),
            "n_nondegenerate_programs": len(nondegenerate_ids),
        },
        "programs": programs,
    }


def _write_new(path: Path, payload: Mapping, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--source-path", required=True)
    worker.add_argument("--aspect-id", required=True)
    worker.add_argument("--source-sha256", required=True)
    worker.add_argument("--out", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--audit", type=Path, required=True)
    run.add_argument("--items-root", type=Path, default=CANONICAL_ITEMS_ROOT)
    run.add_argument("--phase", choices=sorted(ALLOWED_PHASES), required=True)
    run.add_argument("--program-selection", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--program-timeout", type=float, default=600.0)
    run.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "worker":
        items = validate_items(json.load(sys.stdin))
        result = execute_one_program(
            args.source_path, args.aspect_id, args.source_sha256, items
        )
        _write_new(args.out, result)
        return 0
    if args.program_timeout <= 0:
        parser.error("--program-timeout must be positive")
    audit = _load_json(args.audit)
    selection = _load_json(args.program_selection) if args.program_selection else None
    result = execute_audit(
        audit,
        args.items_root,
        phase=args.phase,
        timeout_seconds=args.program_timeout,
        audit_source_path=str(args.audit),
        program_selection=selection,
        program_selection_source=str(args.program_selection) if args.program_selection else None,
    )
    _write_new(args.out, result, force=args.force)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
