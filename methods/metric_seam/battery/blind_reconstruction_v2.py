"""Blind, reconstruction-only development lane for metric-seam programs.

Historical ``agentic_run.py`` intentionally exposed TRAIN correlations and residuals.
This additive v2 lane answers a different question: can an agent reconstruct an
articulated construct without seeing any judge value, residual, test identifier, or
label?  It prepares an auditable compiler bundle and executes a candidate in a fresh
Python process.  Feedback is limited to outputs, coverage, range, and runtime errors.

The lane does not introduce external supervision.  The sealed LLM judgement remains
the later evaluation instrument.  In this module:

  * articulability means a prompt/LLM implementation;
  * verifiability means an executable/code implementation;
  * isomorphism means that both reconstruct the same articulated construct and see the
    same ctext/evidence representation, except for their declared channel capability.

Examples
--------
Prepare a new immutable run directory::

    python blind_reconstruction_v2.py prepare \
      --task math --aspect-id a144 --items .../tasks/math/items.json \
      --contract .../contracts_v3/math__a144.json --out RUN_DIR \
      --capability base --capability math

Run a candidate without judge feedback::

    python blind_reconstruction_v2.py run \
      --bundle RUN_DIR/compiler_bundle.json --candidate candidate.py

``prepare`` refuses to reuse a directory and writes read-only artifacts. ``run``
refuses to overwrite an execution name and verifies the prepared hashes first.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


HERE = pathlib.Path(__file__).resolve().parent
METRIC_SEAM = HERE.parent
HYBRIDS = METRIC_SEAM / "hybrids"
if str(METRIC_SEAM) not in sys.path:
    sys.path.insert(0, str(METRIC_SEAM))

from environment_v2 import environment_fingerprint  # noqa: E402

WORKER = HERE / "_blind_worker_v2.py"
SPLIT_OPS = HERE / "split_ops_v2.py"

SCHEMA_BUNDLE = "metric-seam.blind-reconstruction.compiler-bundle.v2"
SCHEMA_MANIFEST = "metric-seam.blind-reconstruction.prepare-manifest.v2"
SCHEMA_EXECUTION = "metric-seam.blind-reconstruction.execution-manifest.v2"

_DPID_RE = re.compile(r"\bd\d{4,8}\b", re.I)
_LABEL_ASSIGNMENT_RE = re.compile(
    r"\b(?:judge(?:ment)?|gold|label|ground[_ -]?truth|score)\s*(?:=|:)\s*"
    r"(?:[-+]?\d+(?:\.\d+)?|true|false|yes|no)\b",
    re.I,
)

_SAFE_IMPORT_ROOTS = frozenset(
    {"__future__", "re", "math", "statistics", "collections", "itertools",
     "functools", "decimal", "fractions", "typing", "datetime", "calendar"}
)
_FORBIDDEN_CALLS = frozenset(
    {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}
)


class IntegrityError(RuntimeError):
    """A prepared artifact no longer matches its immutable manifest."""


class CandidatePolicyError(ValueError):
    """Candidate source violates the best-effort blind execution policy."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_exclusive(path: pathlib.Path, payload: bytes, *, readonly: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if readonly:
        path.chmod(0o444)


def _redact_contract_string(value: str) -> str:
    value = _DPID_RE.sub("[CORPUS_ITEM]", value)
    return _LABEL_ASSIGNMENT_RE.sub("[LABEL_REDACTED]", value)


def _forbidden_contract_labels(value: Any, path: str = "$") -> list[str]:
    """Locate embedded outcome values, including provenance fields we later omit.

    We reject rather than silently clean these contracts.  That keeps a contaminated
    historical contract from being mistaken for a valid blind input merely because its
    outcome-bearing provenance happened to be outside the projected compiler view.
    """
    found = []
    if isinstance(value, str):
        for match in _LABEL_ASSIGNMENT_RE.finditer(value):
            found.append(f"{path}: {match.group(0)!r}")
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_forbidden_contract_labels(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_contract_labels(child, f"{path}[{index}]"))
    return found


def _safe_contract(raw: Mapping[str, Any]) -> dict:
    """Project a provenance-heavy contract into its articulated, label-free part."""
    probes = []
    for probe in raw.get("cf_probes", []):
        if not isinstance(probe, dict):
            raise ValueError("every cf_probes entry must be an object")
        row = {}
        for key in ("text_pos", "text_neg", "why", "probe_type", "channel"):
            if key in probe:
                value = probe[key]
                row[key] = _redact_contract_string(value) if isinstance(value, str) else value
        if "text_pos" not in row or "text_neg" not in row:
            raise ValueError("each contract probe needs text_pos and text_neg")
        probes.append(row)
    return {
        "construct_definition": _redact_contract_string(str(raw.get("construct_definition", ""))),
        "boundary_notes": _redact_contract_string(str(raw.get("boundary_notes", ""))),
        "cf_probes": probes,
        "discrimination_checks": dict(raw.get("discrimination_checks", {})),
    }


def _implementation_files(capabilities: Iterable[str]) -> dict[str, pathlib.Path]:
    """Transitive local implementation files used by a capability configuration."""
    caps = set(capabilities)
    files = {
        "blind_harness": pathlib.Path(__file__).resolve(),
        "worker": WORKER.resolve(),
        "split_ops": SPLIT_OPS.resolve(),
        "environment_fingerprint": (METRIC_SEAM / "environment_v2.py").resolve(),
    }
    if "base" in caps:
        files["base_ops"] = (HYBRIDS / "ops.py").resolve()
    if "math" in caps:
        files["math_ops"] = (HYBRIDS / "ops_math.py").resolve()
    if "capability" in caps:
        files["capability_ops_v2"] = (HYBRIDS / "ops_capability_v2.py").resolve()
        files["capability_ops_v1_dependency"] = (HYBRIDS / "ops_capability.py").resolve()
    return files


def _implementation_manifest(capabilities: Iterable[str]) -> dict:
    return {name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in _implementation_files(capabilities).items()}


def audit_compiler_bundle(bundle: Mapping[str, Any]) -> None:
    """Fail closed if outcome metadata or non-opaque identifiers enter the bundle.

    Corpus text itself is intentionally not lexically filtered: an article may discuss
    a score or label as content.  The structural interface, construct contract, and
    cached prompt fields are audited because those are the possible supervision paths.
    """
    violations = []
    contract_text = json.dumps(bundle.get("construct", {}), ensure_ascii=False)
    if _LABEL_ASSIGNMENT_RE.search(contract_text):
        violations.append("construct contains an outcome-label assignment")
    if _DPID_RE.search(contract_text):
        violations.append("construct contains a non-opaque corpus identifier")

    field_specs = bundle.get("allowed", {}).get("fields", {})
    if not isinstance(field_specs, dict):
        violations.append("allowed.fields is not a prompt-provenance mapping")
        field_specs = {}
    items = bundle.get("train_items", [])
    expected_keys = [f"train_{i:04d}" for i in range(1, len(items) + 1)]
    observed_keys = []
    for index, item in enumerate(items):
        if set(item) != {"item_key", "ctext", "fields"}:
            violations.append(f"train_items[{index}] has unexpected structural keys")
        key = item.get("item_key")
        observed_keys.append(key)
        if not isinstance(item.get("ctext"), str):
            violations.append(f"train_items[{index}].ctext is not a string")
        fields = item.get("fields", {})
        if not isinstance(fields, dict) or not set(fields).issubset(field_specs):
            violations.append(f"train_items[{index}].fields exceeds its allowlist")
            continue
        for field, value in fields.items():
            if not isinstance(value, str):
                violations.append(f"train_items[{index}].fields.{field} is not text")
            elif _LABEL_ASSIGNMENT_RE.search(value):
                violations.append(
                    f"train_items[{index}].fields.{field} contains an outcome-label assignment"
                )
    if observed_keys != expected_keys:
        violations.append("TRAIN item keys are not the canonical opaque sequence")
    if violations:
        raise ValueError("compiler bundle violates blind policy: " + "; ".join(violations))


def _load_json(path: pathlib.Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _choose_train(items: Sequence[dict], *, train_count: int, seed: int) -> list[dict]:
    """Return TRAIN rows only; no held-out identifier leaves this function."""
    if train_count <= 0 or train_count >= len(items):
        raise ValueError("train_count must be positive and smaller than the corpus")
    by_id = {}
    for row in items:
        dpid = row.get("datapoint_id")
        if not isinstance(dpid, str) or dpid in by_id:
            raise ValueError("items need unique string datapoint_id values")
        if not isinstance(row.get("ctext"), str):
            raise ValueError(f"item {dpid!r} has no string ctext; raw text fallback is forbidden")
        by_id[dpid] = row
    shuffled = sorted(by_id)
    random.Random(seed).shuffle(shuffled)
    selected = set(shuffled[:train_count])
    # A deterministic, opaque enumeration.  Original identifiers are never emitted.
    return [by_id[dpid] for dpid in sorted(selected)]


def _load_allowed_fields(path: pathlib.Path | None, *, aspect_id: str,
                         selected_ids: set[str], allowed_fields: set[str]) -> dict:
    if not allowed_fields:
        return {}
    if path is None:
        raise ValueError("--fields-jsonl is required when --allow-field is used")
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
            if aid != aspect_id or field not in allowed_fields or dpid not in selected_ids:
                continue
            key = (dpid, field)
            if key in seen:
                raise ValueError(f"duplicate cached field at line {line_no}: {key}")
            seen.add(key)
            raw = (row.get("raw") or "").strip()
            out.setdefault(dpid, {})[field] = "" if raw.upper() == "NONE" else raw
    return out


def build_bundle(*, task: str, aspect_id: str, items_path: pathlib.Path,
                 contract_path: pathlib.Path, train_count: int = 150, split_seed: int = 7,
                 capabilities: Iterable[str] = ("base",), allowed_fields: Iterable[str] = (),
                 fields_path: pathlib.Path | None = None,
                 field_prompts: Mapping[str, str] | None = None) -> tuple[dict, dict]:
    """Build a compiler bundle and provenance payload without reading judge data."""
    from split_ops_v2 import SUPPORTED_CAPABILITIES

    capability_set = set(capabilities)
    unknown = capability_set - SUPPORTED_CAPABILITIES
    if unknown:
        raise ValueError(f"unknown capabilities: {sorted(unknown)}")
    field_set = set(allowed_fields)
    prompts = dict(field_prompts or {})
    if set(prompts) != field_set:
        raise ValueError(
            "field prompt provenance must be supplied for exactly every allowed field"
        )
    if any(not isinstance(name, str) or not isinstance(prompt, str) or not prompt.strip()
           for name, prompt in prompts.items()):
        raise ValueError("field prompts must be non-empty strings")

    all_items = _load_json(items_path)
    if not isinstance(all_items, list):
        raise ValueError("items source must contain a JSON list")
    selected = _choose_train(all_items, train_count=train_count, seed=split_seed)
    selected_ids = {row["datapoint_id"] for row in selected}
    fields = _load_allowed_fields(fields_path, aspect_id=aspect_id,
                                  selected_ids=selected_ids, allowed_fields=field_set)
    raw_contract = _load_json(contract_path)
    forbidden = _forbidden_contract_labels(raw_contract)
    if forbidden:
        preview = "; ".join(forbidden[:5])
        suffix = f" (+{len(forbidden) - 5} more)" if len(forbidden) > 5 else ""
        raise ValueError(
            "contract contains forbidden outcome labels and is not eligible for blind "
            f"reconstruction: {preview}{suffix}"
        )
    contract = _safe_contract(raw_contract)

    train_items = []
    for index, row in enumerate(selected, 1):
        dpid = row["datapoint_id"]
        train_items.append(
            {
                "item_key": f"train_{index:04d}",
                "ctext": row["ctext"],
                "fields": fields.get(dpid, {}),
            }
        )

    bundle = {
        "schema": SCHEMA_BUNDLE,
        "objective": {
            "name": "unsupervised reconstruction of an articulated metric",
            "articulability": "prompt/LLM implementation of the articulated construct",
            "verifiability": "executable/code implementation of the articulated construct",
            "external_ground_truth": False,
        },
        "task": task,
        "aspect_id": aspect_id,
        "construct": contract,
        "allowed": {
            "fields": {
                name: {"prompt": prompts[name],
                       "prompt_sha256": sha256_bytes(prompts[name].encode("utf-8"))}
                for name in sorted(field_set)
            },
            "capabilities": sorted(capability_set),
        },
        "interface": {
            "score_signature": "score(ctext: str, extracted: dict, ops) -> float in [0,1]",
            "representation": "ctext only",
            "item_keys": "opaque TRAIN-only aliases",
            "development_feedback": ["candidate outputs", "coverage", "range", "runtime errors"],
            "judge_values_available": False,
            "residuals_available": False,
            "heldout_identifiers_available": False,
        },
        "train_items": train_items,
    }
    audit_compiler_bundle(bundle)
    provenance = {
        "inputs": {
            "items": {"path": str(items_path.resolve()), "sha256": sha256_file(items_path)},
            "contract": {"path": str(contract_path.resolve()),
                         "sha256": sha256_file(contract_path)},
            "fields": ({"path": str(fields_path.resolve()), "sha256": sha256_file(fields_path)}
                       if fields_path else None),
        },
        "partition": {
            "algorithm": "sorted datapoint_id; random.Random(seed).shuffle; first train_count",
            "seed": split_seed,
            "train_count": train_count,
            "corpus_count": len(all_items),
            "heldout_count": len(all_items) - train_count,
            "identifiers_emitted": False,
        },
    }
    return bundle, provenance


def prepare_run(*, out_dir: pathlib.Path, **kwargs) -> tuple[pathlib.Path, pathlib.Path]:
    if out_dir.exists():
        raise FileExistsError(f"run directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    try:
        bundle, provenance = build_bundle(**kwargs)
        bundle_bytes = _canonical_bytes(bundle)
        bundle_path = out_dir / "compiler_bundle.json"
        _write_exclusive(bundle_path, bundle_bytes)
        manifest = {
            "schema": SCHEMA_MANIFEST,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": sha256_bytes(bundle_bytes)[:20],
            "policy": {
                "objective": "unsupervised reconstruction",
                "loads_judge_data": False,
                "loads_heldout_items_into_compiler_bundle": False,
                "ctext_only": True,
                "bundle_policy_audited": True,
                "prepared_artifacts_are_read_only": True,
            },
            **provenance,
            "implementation": _implementation_manifest(bundle["allowed"]["capabilities"]),
            "environment": environment_fingerprint(),
            "artifacts": {"compiler_bundle.json": {"sha256": sha256_bytes(bundle_bytes)}},
        }
        manifest_path = out_dir / "prepare_manifest.json"
        _write_exclusive(manifest_path, _canonical_bytes(manifest))
        return bundle_path, manifest_path
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def verify_prepared(bundle_path: pathlib.Path) -> tuple[dict, dict]:
    manifest_path = bundle_path.parent / "prepare_manifest.json"
    if not manifest_path.exists():
        raise IntegrityError("prepare_manifest.json is missing")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != SCHEMA_MANIFEST:
        raise IntegrityError("unexpected prepare manifest schema")
    expected = manifest.get("artifacts", {}).get(bundle_path.name, {}).get("sha256")
    if not expected or sha256_file(bundle_path) != expected:
        raise IntegrityError("compiler bundle hash does not match prepare manifest")
    bundle = _load_json(bundle_path)
    if bundle.get("schema") != SCHEMA_BUNDLE:
        raise IntegrityError("unexpected compiler bundle schema")
    try:
        audit_compiler_bundle(bundle)
    except ValueError as exc:
        raise IntegrityError(str(exc)) from exc
    expected_files = _implementation_files(bundle["allowed"]["capabilities"])
    for key, path in expected_files.items():
        frozen = manifest.get("implementation", {}).get(key, {}).get("sha256")
        if not frozen or sha256_file(path) != frozen:
            raise IntegrityError(f"{key} implementation changed since preparation")
    if manifest.get("environment", {}).get("sha256") != environment_fingerprint()["sha256"]:
        raise IntegrityError("execution environment changed since preparation")
    return bundle, manifest


def _audit_candidate_text(source: str, filename: str) -> None:
    """Reject common filesystem/network/process escape routes.

    This is a reproducibility and accidental-leak guard, not an OS security boundary.
    The execution manifest states that limitation explicitly.
    """
    tree = ast.parse(source, filename=filename)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in _SAFE_IMPORT_ROOTS:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level or root not in _SAFE_IMPORT_ROOTS:
                violations.append(f"from {node.module or '.'} import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                violations.append(f"call {node.func.id}()")
    if violations:
        raise CandidatePolicyError("candidate uses forbidden operations: " + ", ".join(sorted(set(violations))))


def audit_candidate_source(path: pathlib.Path) -> None:
    _audit_candidate_text(path.read_text(encoding="utf-8"), str(path))


def run_candidate(*, bundle_path: pathlib.Path, candidate_path: pathlib.Path,
                  execution_name: str | None = None, timeout_per_item: float = 20.0,
                  process_timeout: float | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    bundle, prepared = verify_prepared(bundle_path)
    candidate_bytes = candidate_path.read_bytes()
    try:
        candidate_source = candidate_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CandidatePolicyError("candidate source must be UTF-8") from exc
    _audit_candidate_text(candidate_source, str(candidate_path))
    candidate_hash = sha256_bytes(candidate_bytes)
    name = execution_name or f"execution_{candidate_hash[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("execution_name may contain only letters, digits, dot, dash, underscore")
    result_path = bundle_path.parent / f"{name}.json"
    manifest_path = bundle_path.parent / f"{name}.manifest.json"
    if result_path.exists() or manifest_path.exists():
        raise FileExistsError(f"execution already exists: {name}")

    with tempfile.TemporaryDirectory(prefix="metric_seam_blind_v2_") as tmp:
        temp = pathlib.Path(tmp)
        candidate_copy = temp / "candidate.py"
        # Execute the exact bytes that were audited and hashed above (no path TOCTOU).
        candidate_copy.write_bytes(candidate_bytes)
        request = {
            "schema": "metric-seam.blind-reconstruction.worker-request.v2",
            "bundle": bundle,
            "timeout_per_item": timeout_per_item,
        }
        request_path = temp / "request.json"
        request_path.write_bytes(_canonical_bytes(request))
        worker_output = temp / "result.json"
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(temp),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONHASHSEED": "0",
        }
        timeout = process_timeout or max(60.0, len(bundle["train_items"]) * timeout_per_item + 30.0)
        proc = subprocess.run(
            [sys.executable, "-I", str(WORKER), str(request_path),
             str(candidate_copy), str(worker_output)],
            cwd=temp, env=env, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or "worker failed")[-2000:]
            raise RuntimeError(f"blind worker exited {proc.returncode}: {message}")
        result_bytes = worker_output.read_bytes()

    _write_exclusive(result_path, result_bytes)
    execution_manifest = {
        "schema": SCHEMA_EXECUTION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prepared_run_id": prepared["run_id"],
        "inputs": {
            "compiler_bundle": {"path": str(bundle_path.resolve()),
                                "sha256": sha256_file(bundle_path)},
            "candidate": {"path": str(candidate_path.resolve()), "sha256": candidate_hash},
        },
        "execution": {
            "fresh_process": True,
            "python_isolated_flag": True,
            "cwd": "ephemeral temporary directory",
            "environment": "minimal allowlist",
            "source_policy_checked": True,
            "os_security_boundary": False,
            "note": "AST policy prevents accidental leakage; it is not an adversarial sandbox.",
            "timeout_per_item_seconds": timeout_per_item,
        },
        "implementation": _implementation_manifest(bundle["allowed"]["capabilities"]),
        "environment": environment_fingerprint(),
        "artifacts": {result_path.name: {"sha256": sha256_bytes(result_bytes)}},
    }
    _write_exclusive(manifest_path, _canonical_bytes(execution_manifest))
    return result_path, manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="create an immutable label-free compiler bundle")
    prep.add_argument("--task", required=True)
    prep.add_argument("--aspect-id", required=True)
    prep.add_argument("--items", type=pathlib.Path, required=True)
    prep.add_argument("--contract", type=pathlib.Path, required=True)
    prep.add_argument("--out", type=pathlib.Path, required=True)
    prep.add_argument("--train-count", type=int, default=150)
    prep.add_argument("--split-seed", type=int, default=7)
    prep.add_argument("--capability", action="append", default=[])
    prep.add_argument("--allow-field", action="append", default=[])
    prep.add_argument("--fields-jsonl", type=pathlib.Path)
    prep.add_argument(
        "--field-specs", type=pathlib.Path,
        help="JSON object mapping every --allow-field name to its frozen extraction prompt",
    )

    run = sub.add_parser("run", help="execute a candidate with label-free feedback")
    run.add_argument("--bundle", type=pathlib.Path, required=True)
    run.add_argument("--candidate", type=pathlib.Path, required=True)
    run.add_argument("--execution-name")
    run.add_argument("--timeout-per-item", type=float, default=20.0)
    run.add_argument("--process-timeout", type=float)

    verify = sub.add_parser("verify", help="verify prepared artifact hashes")
    verify.add_argument("--bundle", type=pathlib.Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        capabilities = args.capability or ["base"]
        field_prompts = _load_json(args.field_specs) if args.field_specs else {}
        if not isinstance(field_prompts, dict):
            raise ValueError("--field-specs must contain a JSON object")
        bundle, manifest = prepare_run(
            out_dir=args.out, task=args.task, aspect_id=args.aspect_id,
            items_path=args.items, contract_path=args.contract,
            train_count=args.train_count, split_seed=args.split_seed,
            capabilities=capabilities, allowed_fields=args.allow_field,
            fields_path=args.fields_jsonl, field_prompts=field_prompts,
        )
        print(json.dumps({"bundle": str(bundle), "manifest": str(manifest)}))
    elif args.command == "run":
        result, manifest = run_candidate(
            bundle_path=args.bundle, candidate_path=args.candidate,
            execution_name=args.execution_name, timeout_per_item=args.timeout_per_item,
            process_timeout=args.process_timeout,
        )
        print(json.dumps({"result": str(result), "manifest": str(manifest)}))
    else:
        _bundle, manifest = verify_prepared(args.bundle)
        print(json.dumps({"verified": True, "run_id": manifest["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
