"""Sealed held-out evaluation for blind metric reconstruction programs.

This additive evaluator closes the compiler/evaluator separation begun by
``blind_reconstruction_v2.py``.  It reconstructs the held-out complement from the
hash-pinned item source and split recipe, runs the exact candidate bytes from a prior
label-free execution in a fresh isolated process, and *only then* opens the frozen LLM
reference file.  No reference value is sent to candidate execution.

Terminology is intentionally strict:

* articulability is prompt/LLM implementation of an articulated construct;
* verifiability is executable/code implementation of that construct;
* candidate/reference correlation is reconstruction (isomorphism) evidence, not an
  external-ground-truth accuracy claim.

The output directory is exclusive and every artifact is made read-only.  A manifest
uses repository-relative paths and pins all inputs and outputs by SHA-256; a separate
receipt pins the manifest itself.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
METRIC_SEAM = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(METRIC_SEAM) not in sys.path:
    sys.path.insert(0, str(METRIC_SEAM))

from environment_v2 import environment_fingerprint  # noqa: E402
from blind_reconstruction_v2 import (  # noqa: E402
    IntegrityError,
    SCHEMA_EXECUTION,
    _audit_candidate_text,
    verify_prepared,
)


WORKER = HERE / "_sealed_worker_v2.py"
SCHEMA_MANIFEST = "metric-seam.blind-reconstruction.sealed-evaluation-manifest.v2"
SCHEMA_METRICS = "metric-seam.blind-reconstruction.sealed-metrics.v2"

HISTORICAL_PROGRAM_DIRS = {
    "press_releases": "programs_v2",
    "creative_writing": "programs_cw",
    "math": "programs_math",
    "humor": "programs_humor",
    "legal_title_vii": "programs_legal",
    "peer_review": "programs_peer",
    "legal_ss_disability": "programs_ssdis",
    "humor_units": "programs_units",
}


class SealedEvaluationError(RuntimeError):
    """The frozen run cannot be evaluated without violating an invariant."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: pathlib.Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_exclusive(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _repo_path(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise SealedEvaluationError(
            f"artifact is outside the declared repository root: {path}"
        ) from exc


def _resolve_recorded_path(value: str, repo_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _input_record(path: pathlib.Path, repo_root: pathlib.Path) -> dict:
    return {"path": _repo_path(path, repo_root), "sha256": sha256_file(path)}


def _manifest_result_path(execution_manifest_path: pathlib.Path,
                          execution_manifest: Mapping[str, Any]) -> pathlib.Path:
    artifacts = execution_manifest.get("artifacts", {})
    if not isinstance(artifacts, dict) or len(artifacts) != 1:
        raise IntegrityError("execution manifest must pin exactly one result artifact")
    name = next(iter(artifacts))
    if pathlib.Path(name).name != name:
        raise IntegrityError("execution result artifact must be a sibling filename")
    return execution_manifest_path.parent / name


def _verify_prior_execution(
    *,
    bundle_path: pathlib.Path,
    execution_manifest_path: pathlib.Path,
    candidate_path: pathlib.Path | None,
    repo_root: pathlib.Path,
) -> tuple[dict, dict, dict, pathlib.Path, pathlib.Path, bytes]:
    """Validate the prepared bundle, prior execution, exact source, and train output."""
    bundle, prepare_manifest = verify_prepared(bundle_path)
    execution_manifest = _load_json(execution_manifest_path)
    if execution_manifest.get("schema") != SCHEMA_EXECUTION:
        raise IntegrityError("unexpected blind execution manifest schema")
    if execution_manifest.get("prepared_run_id") != prepare_manifest.get("run_id"):
        raise IntegrityError("execution and preparation run ids differ")
    recorded_bundle = execution_manifest.get("inputs", {}).get("compiler_bundle", {})
    if recorded_bundle.get("sha256") != sha256_file(bundle_path):
        raise IntegrityError("execution manifest does not pin the supplied compiler bundle")
    recorded_bundle_path = _resolve_recorded_path(
        str(recorded_bundle.get("path", "")), repo_root
    )
    if recorded_bundle_path != bundle_path.resolve():
        raise IntegrityError("execution manifest points at a different compiler bundle")

    candidate_record = execution_manifest.get("inputs", {}).get("candidate", {})
    recorded_candidate = _resolve_recorded_path(
        str(candidate_record.get("path", "")), repo_root
    )
    supplied_candidate = candidate_path.resolve() if candidate_path else recorded_candidate
    if supplied_candidate != recorded_candidate:
        raise IntegrityError("supplied candidate differs from the previously executed source")
    candidate_bytes = supplied_candidate.read_bytes()
    if sha256_bytes(candidate_bytes) != candidate_record.get("sha256"):
        raise IntegrityError("candidate changed after the label-free execution")
    try:
        source = candidate_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError("candidate source is no longer valid UTF-8") from exc
    _audit_candidate_text(source, str(supplied_candidate))
    if not execution_manifest.get("execution", {}).get("source_policy_checked"):
        raise IntegrityError("prior execution did not attest source-policy checking")

    # A prepared implementation drift would already fail verify_prepared.  The prior
    # execution must additionally have recorded that same transitive implementation.
    if execution_manifest.get("implementation") != prepare_manifest.get("implementation"):
        raise IntegrityError("execution implementation differs from prepared implementation")
    if execution_manifest.get("environment", {}).get("sha256") != (
        prepare_manifest.get("environment", {}).get("sha256")
    ):
        raise IntegrityError("execution environment differs from prepared environment")

    execution_result_path = _manifest_result_path(
        execution_manifest_path, execution_manifest
    )
    expected_result_hash = execution_manifest["artifacts"][execution_result_path.name].get(
        "sha256"
    )
    if not execution_result_path.exists() or sha256_file(execution_result_path) != expected_result_hash:
        raise IntegrityError("label-free execution result changed after it was recorded")
    execution_result = _load_json(execution_result_path)
    if execution_result.get("schema") != (
        "metric-seam.blind-reconstruction.execution-result.v2"
    ):
        raise IntegrityError("unexpected label-free execution result schema")
    if execution_result.get("n_items") != len(bundle.get("train_items", [])):
        raise IntegrityError("label-free result does not cover the compiler TRAIN bundle")

    return (
        bundle,
        prepare_manifest,
        execution_manifest,
        execution_result_path,
        supplied_candidate,
        candidate_bytes,
    )


def _reconstruct_partition(bundle: Mapping[str, Any], prepare: Mapping[str, Any],
                           items_path: pathlib.Path) -> tuple[list[dict], list[dict]]:
    rows = _load_json(items_path)
    if not isinstance(rows, list):
        raise IntegrityError("pinned item source is not a JSON list")
    by_id: dict[str, dict] = {}
    for row in rows:
        dpid = row.get("datapoint_id") if isinstance(row, dict) else None
        if not isinstance(dpid, str) or dpid in by_id:
            raise IntegrityError("pinned items need unique string datapoint_id values")
        if not isinstance(row.get("ctext"), str):
            raise IntegrityError(f"item {dpid!r} lacks frozen ctext")
        by_id[dpid] = row

    partition = prepare.get("partition", {})
    if partition.get("algorithm") != (
        "sorted datapoint_id; random.Random(seed).shuffle; first train_count"
    ):
        raise IntegrityError("unrecognized split algorithm; exact complement is unavailable")
    ids = sorted(by_id)
    random.Random(int(partition["seed"])).shuffle(ids)
    train_count = int(partition["train_count"])
    if len(ids) != int(partition["corpus_count"]):
        raise IntegrityError("pinned corpus count differs from prepare manifest")
    train_ids = set(ids[:train_count])
    train_rows = [by_id[dpid] for dpid in sorted(train_ids)]
    heldout_rows = [by_id[dpid] for dpid in sorted(set(ids) - train_ids)]
    if len(heldout_rows) != int(partition["heldout_count"]):
        raise IntegrityError("reconstructed held-out count differs from prepare manifest")

    compiler_rows = bundle.get("train_items", [])
    if len(compiler_rows) != len(train_rows):
        raise IntegrityError("compiler TRAIN count differs from reconstructed split")
    for index, (compiler_row, source_row) in enumerate(zip(compiler_rows, train_rows), 1):
        if compiler_row.get("item_key") != f"train_{index:04d}":
            raise IntegrityError("compiler TRAIN aliases are not canonical")
        if compiler_row.get("ctext") != source_row["ctext"]:
            raise IntegrityError("compiler TRAIN text differs from pinned split reconstruction")
    return train_rows, heldout_rows


def _load_cached_fields(path: pathlib.Path, *, aspect_id: str,
                        heldout_ids: set[str], field_names: set[str]) -> dict[str, dict[str, str]]:
    if not field_names:
        return {}
    out: dict[str, dict[str, str]] = {}
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("channel") != "field":
                continue
            tagged = str(row.get("aspect_id", ""))
            if "__" not in tagged:
                continue
            aid, field = tagged.split("__", 1)
            dpid = row.get("datapoint_id")
            if aid != aspect_id or field not in field_names or dpid not in heldout_ids:
                continue
            key = (dpid, field)
            if key in seen:
                raise SealedEvaluationError(f"duplicate cached field at line {line_no}: {key}")
            seen.add(key)
            raw = str(row.get("raw") or "").strip()
            out.setdefault(dpid, {})[field] = "" if raw.upper() == "NONE" else raw
    return out


def _extract_literal_llm_fields(source: str, filename: str) -> dict[str, str]:
    """Read historical prompt provenance without importing the historical program."""
    tree = ast.parse(source, filename=filename)
    found = None
    for node in tree.body:
        value_node = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "LLM_FIELDS"
            for target in node.targets
        ):
            value_node = node.value
        elif (isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name)
              and node.target.id == "LLM_FIELDS"):
            value_node = node.value
        if value_node is not None:
            try:
                found = ast.literal_eval(value_node)
            except (ValueError, TypeError) as exc:
                raise SealedEvaluationError(
                    "historical LLM_FIELDS must be a literal prompt mapping"
                ) from exc
    if found is None:
        return {}
    if (not isinstance(found, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in found.items())):
        raise SealedEvaluationError("historical LLM_FIELDS has invalid provenance")
    return found


def _audit_historical_prompt_source(
    path: pathlib.Path,
    *,
    aspect_id: str,
    heldout_rows: Sequence[dict],
    field_prompts: Mapping[str, str],
) -> dict:
    """Bind cached historical fields to their per-item prompt requests.

    The historical prompt JSONL contains the complete rendered prompt, including the
    field instruction from ``LLM_FIELDS`` and the item ctext.  Checking both components
    prevents a same-name cached field produced by a different prompt or representation
    from silently entering the comparison.
    """
    text_by_id = {row["datapoint_id"]: row["ctext"] for row in heldout_rows}
    expected = {
        (dpid, field) for dpid in text_by_id for field in field_prompts
    }
    observed = set()
    instruction_matches = 0
    ctext_matches = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("channel") != "field":
                continue
            tagged = str(row.get("aspect_id", ""))
            if "__" not in tagged:
                continue
            aid, field = tagged.split("__", 1)
            dpid = row.get("datapoint_id")
            key = (dpid, field)
            if aid != aspect_id or key not in expected:
                continue
            if key in observed:
                raise SealedEvaluationError(
                    f"duplicate historical field prompt at line {line_no}: {key}"
                )
            observed.add(key)
            rendered = row.get("prompt")
            if not isinstance(rendered, str):
                continue
            instruction_matches += field_prompts[field] in rendered
            ctext_matches += text_by_id[dpid] in rendered
    return {
        "expected_item_field_pairs": len(expected),
        "observed_item_field_pairs": len(observed),
        "exact_instruction_matches": instruction_matches,
        "exact_ctext_matches": ctext_matches,
        "complete_exact_binding": (
            len(observed) == len(expected)
            and instruction_matches == len(expected)
            and ctext_matches == len(expected)
        ),
    }


def _run_worker(*, candidate_bytes: bytes, bundle: Mapping[str, Any],
                heldout_rows: Sequence[dict], fields: Mapping[str, Mapping[str, str]],
                expected_fields: Mapping[str, str], timeout_per_item: float,
                process_timeout: float | None) -> tuple[dict[str, float | None], dict]:
    opaque_to_id = {
        f"heldout_{index:04d}": row["datapoint_id"]
        for index, row in enumerate(heldout_rows, 1)
    }
    eval_items = []
    for opaque, row in zip(opaque_to_id, heldout_rows):
        eval_items.append({
            "item_key": opaque,
            "ctext": row["ctext"],
            "fields": dict(fields.get(row["datapoint_id"], {})),
        })
    request = {
        "schema": "metric-seam.blind-reconstruction.sealed-worker-request.v2",
        "train_items": [
            {"item_key": row["item_key"], "ctext": row["ctext"]}
            for row in bundle["train_items"]
        ],
        "eval_items": eval_items,
        "capabilities": list(bundle["allowed"]["capabilities"]),
        "expected_fields": dict(expected_fields),
        "timeout_per_item": timeout_per_item,
        "reference_values_present": False,
    }

    with tempfile.TemporaryDirectory(prefix="metric_seam_sealed_v2_") as tmp_name:
        tmp = pathlib.Path(tmp_name)
        candidate_copy = tmp / "candidate.py"
        candidate_copy.write_bytes(candidate_bytes)
        request_path = tmp / "request.json"
        request_path.write_bytes(_canonical_bytes(request))
        result_path = tmp / "result.json"
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONHASHSEED": "0",
        }
        timeout = process_timeout or max(
            60.0, len(heldout_rows) * timeout_per_item + 30.0
        )
        process = subprocess.run(
            [sys.executable, "-I", str(WORKER), str(request_path),
             str(candidate_copy), str(result_path)],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if process.returncode != 0:
            message = (process.stderr or process.stdout or "sealed worker failed")[-2000:]
            raise SealedEvaluationError(
                f"sealed worker exited {process.returncode}: {message}"
            )
        worker_result = _load_json(result_path)

    expected_keys = list(opaque_to_id)
    rows = worker_result.get("outputs", [])
    if [row.get("item_key") for row in rows] != expected_keys:
        raise SealedEvaluationError("sealed worker returned unexpected held-out aliases")
    score_map = {
        opaque_to_id[row["item_key"]]: row.get("score") for row in rows
    }
    # Errors retain only opaque aliases in the public artifact.  The post-evaluation
    # score map carries real ids, but the candidate process never did.
    return score_map, worker_result


def _load_llm_reference(path: pathlib.Path, *, aspect_id: str,
                        heldout_ids: set[str]) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    pass1: dict[str, int] = {}
    pass2: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("aspect_id") != aspect_id or row.get("datapoint_id") not in heldout_ids:
                continue
            channel = row.get("channel")
            score = row.get("score")
            if channel not in {"pass1", "pass2"} or type(score) is not int:
                continue
            if not 0 <= score <= 10:
                raise SealedEvaluationError(f"invalid LLM score at line {line_no}")
            target = pass1 if channel == "pass1" else pass2
            dpid = row["datapoint_id"]
            if dpid in target:
                raise SealedEvaluationError(
                    f"duplicate {channel} LLM reference for {dpid}"
                )
            target[dpid] = score
    common = sorted(set(pass1) & set(pass2))
    reference = {dpid: (pass1[dpid] + pass2[dpid]) / 20.0 for dpid in common}
    return reference, pass1, pass2


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    left = 0
    while left < len(order):
        right = left
        while (right + 1 < len(order)
               and values[order[right + 1]] == values[order[left]]):
            right += 1
        rank = (left + right) / 2.0 + 1.0
        for position in range(left, right + 1):
            ranks[order[position]] = rank
        left = right + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right)
    )
    denom_left = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    denom_right = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    if denom_left == 0.0 or denom_right == 0.0:
        return None
    return numerator / (denom_left * denom_right)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _comparison(score_map: Mapping[str, float | None],
                reference: Mapping[str, float], heldout_count: int) -> dict:
    finite_scores = {
        key: float(value) for key, value in score_map.items()
        if value is not None and math.isfinite(float(value))
    }
    common = sorted(set(finite_scores) & set(reference))
    left = [finite_scores[key] for key in common]
    right = [reference[key] for key in common]
    rho = spearman(left, right)
    return {
        "heldout_count": heldout_count,
        "reference_available_count": len(reference),
        "common_count": len(common),
        "n_scoreable": len(finite_scores),
        "candidate_coverage_all_heldout": (
            len(finite_scores) / heldout_count if heldout_count else 0.0
        ),
        "candidate_coverage_conditional_on_reference": (
            len(common) / len(reference) if reference else 0.0
        ),
        "reference_count": len(reference),
        "reference_availability_all_heldout": (
            len(reference) / heldout_count if heldout_count else 0.0
        ),
        "common_support_n": len(common),
        "common_support_ids": common,
        "spearman_reconstruction": rho,
        "mean_absolute_difference": (
            statistics.mean(abs(x - y) for x, y in zip(left, right)) if common else None
        ),
        "score_min": min(finite_scores.values()) if finite_scores else None,
        "score_max": max(finite_scores.values()) if finite_scores else None,
        "score_std_population": (
            statistics.pstdev(finite_scores.values()) if finite_scores else None
        ),
    }


def _historical_program_path(task: str, aspect_id: str,
                             repo_root: pathlib.Path) -> pathlib.Path | None:
    directory = HISTORICAL_PROGRAM_DIRS.get(task)
    if directory is None:
        return None
    path = repo_root / "methods/metric_seam/hybrids" / directory / f"{aspect_id}_h0.py"
    return path if path.exists() else None


def evaluate_sealed(
    *,
    bundle_path: pathlib.Path,
    execution_manifest_path: pathlib.Path,
    out_dir: pathlib.Path,
    repo_root: pathlib.Path = ROOT,
    candidate_path: pathlib.Path | None = None,
    reference_path: pathlib.Path | None = None,
    include_historical: bool = True,
    historical_h0_path: pathlib.Path | None = None,
    historical_fields_path: pathlib.Path | None = None,
    historical_prompts_path: pathlib.Path | None = None,
    code_scores_path: pathlib.Path | None = None,
    timeout_per_item: float = 20.0,
    process_timeout: float | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Run one immutable evaluation and return ``(metrics, manifest)`` paths."""
    repo_root = repo_root.resolve()
    bundle_path = bundle_path.resolve()
    execution_manifest_path = execution_manifest_path.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"sealed evaluation directory already exists: {out_dir}")

    (
        bundle,
        prepare,
        execution_manifest,
        execution_result_path,
        candidate_source_path,
        candidate_bytes,
    ) = _verify_prior_execution(
        bundle_path=bundle_path,
        execution_manifest_path=execution_manifest_path,
        candidate_path=candidate_path,
        repo_root=repo_root,
    )
    prepare_path = bundle_path.parent / "prepare_manifest.json"

    items_record = prepare.get("inputs", {}).get("items", {})
    items_path = _resolve_recorded_path(str(items_record.get("path", "")), repo_root)
    if sha256_file(items_path) != items_record.get("sha256"):
        raise IntegrityError("pinned item source changed after preparation")
    _train_rows, heldout_rows = _reconstruct_partition(bundle, prepare, items_path)
    heldout_ids = {row["datapoint_id"] for row in heldout_rows}

    allowed_specs = bundle["allowed"]["fields"]
    expected_candidate_fields = {
        name: spec["prompt"] for name, spec in allowed_specs.items()
    }
    candidate_fields: dict[str, dict[str, str]] = {}
    compiler_fields_path = None
    if expected_candidate_fields:
        compiler_fields_record = prepare.get("inputs", {}).get("fields")
        if not compiler_fields_record:
            raise IntegrityError("candidate fields lack a prepared source record")
        compiler_fields_path = _resolve_recorded_path(
            str(compiler_fields_record.get("path", "")), repo_root
        )
        if sha256_file(compiler_fields_path) != compiler_fields_record.get("sha256"):
            raise IntegrityError("frozen compiler field source changed")
        candidate_fields = _load_cached_fields(
            compiler_fields_path,
            aspect_id=bundle["aspect_id"],
            heldout_ids=heldout_ids,
            field_names=set(expected_candidate_fields),
        )

    # Critical ordering: execute exact candidate bytes before opening results.jsonl.
    candidate_started_at = _utc_now()
    candidate_scores, candidate_worker_result = _run_worker(
        candidate_bytes=candidate_bytes,
        bundle=bundle,
        heldout_rows=heldout_rows,
        fields=candidate_fields,
        expected_fields=expected_candidate_fields,
        timeout_per_item=timeout_per_item,
        process_timeout=process_timeout,
    )
    candidate_completed_at = _utc_now()

    # Off-label diagnostic only: the frozen channel annotations remain untouched.
    # A code-only candidate can nevertheless challenge an all-L allocation if it
    # separates the already-frozen probe pairs.  This is neither contract certification
    # nor permission to relabel the contract after seeing the outputs.
    contract_record = prepare.get("inputs", {}).get("contract", {})
    contract_path = _resolve_recorded_path(str(contract_record.get("path", "")), repo_root)
    if sha256_file(contract_path) != contract_record.get("sha256"):
        raise IntegrityError("frozen contract source changed after preparation")
    probes = bundle.get("construct", {}).get("cf_probes", [])
    channel_challenge: dict[str, Any]
    if expected_candidate_fields:
        channel_challenge = {
            "status": "not_evaluated",
            "reason": "candidate is not code-only; frozen probe extraction fields are unavailable",
            "is_contract_pass": False,
            "frozen_channel_labels_changed": False,
        }
    else:
        probe_rows = []
        for index, probe in enumerate(probes, 1):
            probe_rows.extend([
                {"datapoint_id": f"probe_{index:02d}_pos", "ctext": probe["text_pos"]},
                {"datapoint_id": f"probe_{index:02d}_neg", "ctext": probe["text_neg"]},
            ])
        probe_scores, probe_worker = _run_worker(
            candidate_bytes=candidate_bytes,
            bundle=bundle,
            heldout_rows=probe_rows,
            fields={},
            expected_fields={},
            timeout_per_item=timeout_per_item,
            process_timeout=process_timeout,
        )
        challenge_rows = []
        for index, probe in enumerate(probes, 1):
            positive = probe_scores.get(f"probe_{index:02d}_pos")
            negative = probe_scores.get(f"probe_{index:02d}_neg")
            delta = (
                positive - negative
                if positive is not None and negative is not None else None
            )
            challenge_rows.append({
                "probe_index": index - 1,
                "frozen_channel": probe.get("channel"),
                "probe_type": probe.get("probe_type"),
                "positive_score": positive,
                "negative_score": negative,
                "positive_minus_negative": delta,
                "directionally_separated_at_0_05": delta is not None and delta >= 0.05,
            })
        n_separated = sum(
            row["directionally_separated_at_0_05"] for row in challenge_rows
        )
        channel_challenge = {
            "schema": "metric-seam.blind-reconstruction.channel-challenge.v2",
            "status": "off_label_diagnostic",
            "task": bundle["task"],
            "aspect_id": bundle["aspect_id"],
            "candidate_channel": "code",
            "is_contract_pass": False,
            "frozen_channel_labels_changed": False,
            "claim": (
                "Probe separation by a frozen code-only candidate is evidence that the "
                "original seam allocation may be pessimistic/challengeable. It does not "
                "retroactively turn an L-tagged probe into a CODE contract probe."
            ),
            "diagnostic_threshold": "positive_score - negative_score >= 0.05",
            "n_probes": len(challenge_rows),
            "n_directionally_separated": n_separated,
            "all_frozen_channels": sorted({str(row["frozen_channel"])
                                            for row in challenge_rows}),
            "candidate_sha256": sha256_bytes(candidate_bytes),
            "contract": _input_record(contract_path, repo_root),
            "probes": challenge_rows,
            "worker_errors": probe_worker.get("errors", []),
        }

    historical: dict[str, Any] = {}
    historical_inputs: dict[str, pathlib.Path] = {}
    h0_scores = None
    task_dir = repo_root / "outputs/metric_seam_pilot/tasks" / bundle["task"]
    if include_historical:
        h0_path = historical_h0_path or _historical_program_path(
            bundle["task"], bundle["aspect_id"], repo_root
        )
        if h0_path is not None and h0_path.exists():
            h0_bytes = h0_path.read_bytes()
            h0_prompts = _extract_literal_llm_fields(
                h0_bytes.decode("utf-8"), str(h0_path)
            )
            h0_fields: dict[str, dict[str, str]] = {}
            field_path = historical_fields_path or task_dir / "field_results.jsonl"
            prompt_path = historical_prompts_path or task_dir / "field_prompts.jsonl"
            field_record = None
            prompt_provenance = None
            if h0_prompts:
                if not field_path.exists():
                    historical["h0"] = {
                        "status": "not_run",
                        "reason": "historical LLM fields file is missing",
                        "channel": "hybrid code+LLM",
                    }
                else:
                    h0_fields = _load_cached_fields(
                        field_path,
                        aspect_id=bundle["aspect_id"],
                        heldout_ids=heldout_ids,
                        field_names=set(h0_prompts),
                    )
                    field_record = field_path
                    if prompt_path.exists():
                        prompt_provenance = {
                            "source": _input_record(prompt_path, repo_root),
                            "binding_audit": _audit_historical_prompt_source(
                                prompt_path,
                                aspect_id=bundle["aspect_id"],
                                heldout_rows=heldout_rows,
                                field_prompts=h0_prompts,
                            ),
                        }
                    else:
                        prompt_provenance = {
                            "source": None,
                            "binding_audit": {
                                "complete_exact_binding": False,
                                "reason": "historical field prompt source is missing",
                            },
                        }
            if not h0_prompts or field_record is not None:
                try:
                    h0_scores, h0_worker = _run_worker(
                        candidate_bytes=h0_bytes,
                        bundle=bundle,
                        heldout_rows=heldout_rows,
                        fields=h0_fields,
                        expected_fields=h0_prompts,
                        timeout_per_item=timeout_per_item,
                        process_timeout=process_timeout,
                    )
                    historical["h0"] = {
                        "status": "executed",
                        "channel": "hybrid code+LLM" if h0_prompts else "code",
                        "score_map": h0_scores,
                        "program": _repo_path(h0_path, repo_root),
                        "program_sha256": sha256_bytes(h0_bytes),
                        "llm_fields": {
                            name: {
                                "prompt": prompt,
                                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                            }
                            for name, prompt in sorted(h0_prompts.items())
                        },
                        "cached_field_source": (
                            _input_record(field_record, repo_root) if field_record else None
                        ),
                        "rendered_prompt_provenance": prompt_provenance,
                        "worker_summary": {
                            key: value for key, value in h0_worker.items()
                            if key not in {"outputs", "errors"}
                        },
                        "errors": h0_worker.get("errors", []),
                    }
                    historical_inputs["historical_h0"] = h0_path
                    if field_record:
                        historical_inputs["historical_h0_fields"] = field_record
                    if h0_prompts and prompt_path.exists():
                        historical_inputs["historical_h0_prompts"] = prompt_path
                except Exception as exc:  # comparison failure must not erase sealed result
                    historical["h0"] = {
                        "status": "not_run",
                        "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                        "channel": "hybrid code+LLM" if h0_prompts else "code",
                    }

    # Only now is the prompt/LLM reference opened.
    reference_path = (reference_path or task_dir / "results.jsonl").resolve()
    reference_load_started_at = _utc_now()
    reference, pass1, pass2 = _load_llm_reference(
        reference_path, aspect_id=bundle["aspect_id"], heldout_ids=heldout_ids
    )
    reference_loaded_at = _utc_now()
    if reference_load_started_at < candidate_completed_at:
        # ISO UTC timestamps are lexicographically ordered.  Fail closed if the trace
        # cannot establish compiler/reference ordering.
        raise SealedEvaluationError("reference was opened before candidate execution ended")

    candidate_metrics = _comparison(candidate_scores, reference, len(heldout_rows))
    pass_common = sorted(set(pass1) & set(pass2))
    reference_reliability = spearman(
        [float(pass1[key]) for key in pass_common],
        [float(pass2[key]) for key in pass_common],
    )

    if include_historical and h0_scores is not None:
        historical["h0"]["metrics"] = _comparison(
            h0_scores, reference, len(heldout_rows)
        )

    if include_historical:
        scores_path = code_scores_path or task_dir / "code_scores.json"
        if scores_path.exists():
            code_payload = _load_json(scores_path)
            prefix = f"{bundle['aspect_id']}_"
            columns = {
                name: values for name, values in code_payload.items()
                if name.startswith(prefix) and isinstance(values, dict)
            }
            historical["preexisting_code_columns"] = {
                "status": "loaded",
                "channel": "code",
                "selection": "all named columns reported; no held-out selection",
                "source": _input_record(scores_path, repo_root),
                "columns": {
                    name: {
                        "score_map": {
                            dpid: values.get(dpid) for dpid in sorted(heldout_ids)
                            if dpid in values
                        },
                        "metrics": _comparison(
                            {dpid: values.get(dpid) for dpid in heldout_ids},
                            reference,
                            len(heldout_rows),
                        ),
                    }
                    for name, values in sorted(columns.items())
                },
            }
            historical_inputs["historical_code_scores"] = scores_path

    metrics = {
        "schema": SCHEMA_METRICS,
        "task": bundle["task"],
        "aspect_id": bundle["aspect_id"],
        "objective": "unsupervised reconstruction of the articulated LLM metric",
        "external_ground_truth": False,
        "terminology": {
            "articulability": "prompt/LLM implementation",
            "verifiability": "executable/code implementation",
            "isomorphism_evidence": (
                "held-out Spearman correlation between the code score and the frozen "
                "two-pass LLM reference on common support"
            ),
        },
        "candidate_channel": "code" if not expected_candidate_fields else "hybrid code+LLM",
        "candidate": candidate_metrics,
        "reference": {
            "channel": "prompt/LLM",
            "aggregation": "mean(pass1, pass2) / 10 on integer-score intersection",
            "n_pass1": len(pass1),
            "n_pass2": len(pass2),
            "n_two_pass": len(reference),
            "two_pass_spearman_reliability": reference_reliability,
        },
        "interpretation": (
            "Correlation estimates reconstruction/isomorphism to an articulated LLM "
            "metric. It is not accuracy against supervised external ground truth. "
            "Divergence may be verifier underperformance, construct mismatch, or a "
            "constructive extension that requires separate construct-validity evidence."
        ),
    }

    candidate_artifact = {
        "schema": "metric-seam.blind-reconstruction.sealed-candidate-scores.v2",
        "task": bundle["task"],
        "aspect_id": bundle["aspect_id"],
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "score_map": candidate_scores,
        "worker_summary": {
            key: value for key, value in candidate_worker_result.items()
            if key not in {"outputs", "errors"}
        },
        "errors": candidate_worker_result.get("errors", []),
    }
    reference_artifact = {
        "schema": "metric-seam.blind-reconstruction.sealed-llm-reference.v2",
        "task": bundle["task"],
        "aspect_id": bundle["aspect_id"],
        "channel": "prompt/LLM",
        "external_ground_truth": False,
        "aggregation": "mean(pass1, pass2) / 10 on integer-score intersection",
        "score_map": reference,
    }

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        artifacts: dict[str, bytes] = {
            "candidate_frozen.py": candidate_bytes,
            "candidate_scores.json": _canonical_bytes(candidate_artifact),
            "llm_reference_scores.json": _canonical_bytes(reference_artifact),
            "metrics.json": _canonical_bytes(metrics),
            "channel_challenge.json": _canonical_bytes(channel_challenge),
        }
        if historical:
            artifacts["historical_comparisons.json"] = _canonical_bytes({
                "schema": "metric-seam.blind-reconstruction.historical-comparisons.v2",
                "task": bundle["task"],
                "aspect_id": bundle["aspect_id"],
                "comparisons": historical,
            })
        artifact_records = {}
        for name, payload in artifacts.items():
            path = out_dir / name
            _write_exclusive(path, payload)
            artifact_records[name] = {
                "path": _repo_path(path, repo_root),
                "sha256": sha256_bytes(payload),
                "readonly": True,
            }

        input_records = {
            "compiler_bundle": _input_record(bundle_path, repo_root),
            "prepare_manifest": _input_record(prepare_path, repo_root),
            "label_free_execution_manifest": _input_record(
                execution_manifest_path, repo_root
            ),
            "label_free_execution_result": _input_record(
                execution_result_path, repo_root
            ),
            "candidate_source_at_seal": _input_record(
                candidate_source_path, repo_root
            ),
            "items": _input_record(items_path, repo_root),
            "contract": _input_record(contract_path, repo_root),
            "llm_reference_results": _input_record(reference_path, repo_root),
        }
        if compiler_fields_path:
            input_records["compiler_cached_fields"] = _input_record(
                compiler_fields_path, repo_root
            )
        for name, path in historical_inputs.items():
            input_records[name] = _input_record(path, repo_root)

        manifest = {
            "schema": SCHEMA_MANIFEST,
            "created_at": _utc_now(),
            "task": bundle["task"],
            "aspect_id": bundle["aspect_id"],
            "prepared_run_id": prepare["run_id"],
            "policy": {
                "objective": "unsupervised reconstruction",
                "external_ground_truth": False,
                "candidate_exact_prior_bytes_verified": True,
                "candidate_frozen_copy_is_read_only": True,
                "prepared_hashes_verified": True,
                "label_free_execution_hashes_verified": True,
                "heldout_exact_complement_reconstructed": True,
                "heldout_identifiers_sent_to_candidate": False,
                "reference_values_sent_to_candidate": False,
                "candidate_execution_preceded_reference_load": True,
                "historical_columns_used_for_tuning": False,
                "outputs_are_read_only": True,
            },
            "evaluation_order": {
                "candidate_execution_started_at": candidate_started_at,
                "candidate_execution_completed_at": candidate_completed_at,
                "llm_reference_load_started_at": reference_load_started_at,
                "llm_reference_loaded_at": reference_loaded_at,
            },
            "partition": {
                **prepare["partition"],
                "heldout_ids_materialized_only_in_evaluator": True,
                "candidate_process_item_keys": "opaque heldout_NNNN aliases",
            },
            "candidate_execution": {
                "fresh_process": True,
                "python_isolated_flag": True,
                "cwd": "ephemeral temporary directory",
                "minimal_environment": True,
                "same_declared_capabilities": bundle["allowed"]["capabilities"],
                "retrieval_scope_if_enabled": "frozen compiler TRAIN ctext only",
                "timeout_per_item_seconds": timeout_per_item,
                "os_security_boundary": False,
            },
            "inputs": input_records,
            "implementation": {
                "prepared": prepare["implementation"],
                "sealed_evaluator": _input_record(pathlib.Path(__file__), repo_root),
                "sealed_worker": _input_record(WORKER, repo_root),
            },
            "environment": environment_fingerprint(),
            "artifacts": artifact_records,
        }
        manifest_bytes = _canonical_bytes(manifest)
        manifest_path = out_dir / "sealed_manifest.json"
        _write_exclusive(manifest_path, manifest_bytes)
        receipt = (
            f"{sha256_bytes(manifest_bytes)}  sealed_manifest.json\n"
        ).encode("ascii")
        receipt_path = out_dir / "sealed_manifest.sha256"
        _write_exclusive(receipt_path, receipt)
        return out_dir / "metrics.json", manifest_path
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--execution-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--repo-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--no-historical", action="store_true")
    parser.add_argument("--historical-h0", type=pathlib.Path)
    parser.add_argument("--historical-fields", type=pathlib.Path)
    parser.add_argument("--historical-prompts", type=pathlib.Path)
    parser.add_argument("--code-scores", type=pathlib.Path)
    parser.add_argument("--timeout-per-item", type=float, default=20.0)
    parser.add_argument("--process-timeout", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics, manifest = evaluate_sealed(
        bundle_path=args.bundle,
        execution_manifest_path=args.execution_manifest,
        out_dir=args.out,
        repo_root=args.repo_root,
        candidate_path=args.candidate,
        reference_path=args.reference,
        include_historical=not args.no_historical,
        historical_h0_path=args.historical_h0,
        historical_fields_path=args.historical_fields,
        historical_prompts_path=args.historical_prompts,
        code_scores_path=args.code_scores,
        timeout_per_item=args.timeout_per_item,
        process_timeout=args.process_timeout,
    )
    report = _load_json(metrics)
    print(json.dumps({
        "metrics": str(metrics),
        "manifest": str(manifest),
        "candidate": report["candidate"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
