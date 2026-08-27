"""Compile and run blind metric-text-only decomposition fleets.

The compiler consumes the frozen family-scale study manifest but places only a
cell's ``metric_text`` in a model prompt.  Three fresh Sonnet requests differ
only by a neutral nonce and the order of otherwise identical prompt sections.
The runner is local, resumable, and phase-gated: all six requests for the two
smoke metrics must parse before any production subprocess can be created.

This module never reads a corpus and does not execute calls at import or
compile time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from .decomposition_stability import (
    CanonicalRelation,
    DecompositionSchemaError,
    SCHEMA as SUBMISSION_SCHEMA,
)


STUDY_SCHEMA = "metric-seam.family-scale-study.v1"
REQUEST_SCHEMA = "metric-seam.decomposition-request.v1"
BUNDLE_SCHEMA = "metric-seam.decomposition-request-bundle.v1"
RESPONSE_SCHEMA = "metric-seam.decomposition-response.v1"
FLEET_IDS = ("fleet_1", "fleet_2", "fleet_3")
SMOKE_METRICS = 2
MIN_RELATIONS = 2
MAX_RELATIONS = 5


class RequestSchemaError(ValueError):
    """Raised when a request bundle or response violates the closed schema."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RequestSchemaError(
            f"{path}: key mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestSchemaError(f"{path}: expected nonempty string")
    return value


def _validate_metric_text(value: object, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise RequestSchemaError(f"{path}: expected object")
    _exact_keys(value, {"construct", "description"}, path)
    return {
        "construct": _nonempty(value["construct"], f"{path}.construct"),
        "description": _nonempty(value["description"], f"{path}.description"),
    }


_TASK_SECTION = """TASK
Decompose the supplied articulated metric into the minimal executable DAG node TYPES needed to assess it. Propose relation types only: do not implement, score, retrieve evidence, assume a corpus, or infer any outcome. Return between 2 and 5 relations."""

_CONTRACT_SECTION = """OUTPUT CONTRACT
Return exactly one JSON object and no prose. Its only key is \"relations\". The value is an array of 2 to 5 distinct objects. Every relation object has exactly these keys:
{"op_class":"computation|evidence|individuation","witness_kind":"short normalized noun phrase","relation":"short normalized relation description"}"""


def _metric_section(metric_text: Mapping[str, str]) -> str:
    return (
        "METRIC TEXT\n"
        f"Name: {metric_text['construct']}\n"
        f"Description: {metric_text['description']}"
    )


# Neutral Latin-square rotations.  The section strings themselves are shared
# exactly across fleets; the nonce carries no task information.
_SECTION_ORDERS = {
    "fleet_1": (0, 1, 2),
    "fleet_2": (1, 2, 0),
    "fleet_3": (2, 0, 1),
}


def _prompt(metric_text: Mapping[str, str], fleet_id: str, nonce: str) -> tuple[str, str]:
    sections = (_TASK_SECTION, _metric_section(metric_text), _CONTRACT_SECTION)
    order = _SECTION_ORDERS[fleet_id]
    prompt = "\n\n".join(sections[index] for index in order)
    prompt += f"\n\nNEUTRAL REQUEST NONCE\n{nonce}"
    # This digest mechanically establishes that fleet semantic content is the
    # same even though presentation order and neutral nonce differ.
    semantic_payload_sha256 = _sha(sorted(sections))
    return prompt, semantic_payload_sha256


def _validate_study(study: object) -> tuple[dict[str, object], ...]:
    if not isinstance(study, dict):
        raise RequestSchemaError("study: expected object")
    if study.get("schema") != STUDY_SCHEMA:
        raise RequestSchemaError(f"study.schema: expected {STUDY_SCHEMA!r}")
    if study.get("status") != "frozen_before_decomposition_or_corpus_contact":
        raise RequestSchemaError("study is not frozen before decomposition")
    decomposition = study.get("decomposition")
    if not isinstance(decomposition, dict):
        raise RequestSchemaError("study.decomposition: expected object")
    if decomposition.get("independent_fleets") != 3:
        raise RequestSchemaError("study must preregister exactly three independent fleets")
    if decomposition.get("relations_per_metric_guidance") != [2, 5]:
        raise RequestSchemaError("study relation guidance must be [2, 5]")
    cells = study.get("cells")
    if not isinstance(cells, list) or len(cells) < SMOKE_METRICS:
        raise RequestSchemaError("study must contain at least two metric cells")

    seen: set[str] = set()
    validated: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        path = f"study.cells[{index}]"
        if not isinstance(cell, dict):
            raise RequestSchemaError(f"{path}: expected object")
        metric_id = _nonempty(cell.get("metric_id"), f"{path}.metric_id")
        if metric_id in seen:
            raise RequestSchemaError(f"{path}.metric_id: duplicate")
        seen.add(metric_id)
        metric_text = _validate_metric_text(cell.get("metric_text"), f"{path}.metric_text")
        expected_sha = _sha(metric_text)
        if cell.get("metric_text_sha256") != expected_sha:
            raise RequestSchemaError(f"{path}.metric_text_sha256: digest mismatch")
        if cell.get("decomposition_input_fields") != ["construct", "description"]:
            raise RequestSchemaError(f"{path}: decomposition input scope drift")
        validated.append(
            {
                "metric_id": metric_id,
                "metric_text": metric_text,
                "metric_text_sha256": expected_sha,
            }
        )
    return tuple(validated)


def compile_requests(study: object, *, model: str = "sonnet") -> dict[str, object]:
    """Compile three fresh requests per metric without contacting a model."""

    if "sonnet" not in _nonempty(model, "model").casefold():
        raise RequestSchemaError("decomposition fleet model must be Sonnet")
    cells = _validate_study(study)
    if not isinstance(study, dict):  # narrowed by _validate_study
        raise AssertionError
    claimed_study_sha = study.get("study_content_sha256")
    study_without_sha = dict(study)
    study_without_sha.pop("study_content_sha256", None)
    computed_study_sha = _sha(study_without_sha)
    if claimed_study_sha != computed_study_sha:
        raise RequestSchemaError("study_content_sha256 mismatch")

    metric_rows: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []
    for metric_index, cell in enumerate(cells):
        metric_id = str(cell["metric_id"])
        metric_text = cell["metric_text"]
        metric_sha = str(cell["metric_text_sha256"])
        phase = "smoke" if metric_index < SMOKE_METRICS else "production"
        metric_rows.append(
            {
                "metric_id": metric_id,
                "metric_text": metric_text,
                "metric_text_sha256": metric_sha,
                "phase": phase,
            }
        )
        semantic_digests: set[str] = set()
        for fleet_id in FLEET_IDS:
            nonce = hashlib.sha256(
                f"metric-seam-decomposition-v1\0{metric_sha}\0{fleet_id}".encode()
            ).hexdigest()[:20]
            prompt, semantic_sha = _prompt(metric_text, fleet_id, nonce)
            semantic_digests.add(semantic_sha)
            request_core = {"model": model, "prompt": prompt}
            request_sha = _sha(request_core)
            request_id = "dreq_" + hashlib.sha256(
                f"{metric_id}\0{fleet_id}\0{request_sha}".encode()
            ).hexdigest()[:24]
            requests.append(
                {
                    "schema": REQUEST_SCHEMA,
                    "request_id": request_id,
                    "metric_id": metric_id,
                    "metric_text_sha256": metric_sha,
                    "fleet_id": fleet_id,
                    "fleet_order": list(_SECTION_ORDERS[fleet_id]),
                    "neutral_nonce": nonce,
                    "semantic_payload_sha256": semantic_sha,
                    "phase": phase,
                    "model": model,
                    "prompt": prompt,
                    "request_sha256": request_sha,
                }
            )
        if len(semantic_digests) != 1:
            raise AssertionError("fleet semantic payloads diverged")

    bundle: dict[str, object] = {
        "schema": BUNDLE_SCHEMA,
        "status": "compiled_no_calls_executed",
        "source_study_content_sha256": computed_study_sha,
        "input_scope": "metric_text_only",
        "model": model,
        "fleet_ids": list(FLEET_IDS),
        "smoke_contract": {
            "metric_count": SMOKE_METRICS,
            "request_count": SMOKE_METRICS * len(FLEET_IDS),
            "required_valid": SMOKE_METRICS * len(FLEET_IDS),
            "production_blocked_until_complete": True,
        },
        "metrics": metric_rows,
        "requests": requests,
        "corpus_accessed": False,
        "model_calls_executed": False,
    }
    bundle["bundle_content_sha256"] = _sha(bundle)
    return bundle


def _validate_request(request: object, path: str = "request") -> dict[str, object]:
    if not isinstance(request, dict):
        raise RequestSchemaError(f"{path}: expected object")
    expected = {
        "schema",
        "request_id",
        "metric_id",
        "metric_text_sha256",
        "fleet_id",
        "fleet_order",
        "neutral_nonce",
        "semantic_payload_sha256",
        "phase",
        "model",
        "prompt",
        "request_sha256",
    }
    _exact_keys(request, expected, path)
    if request["schema"] != REQUEST_SCHEMA:
        raise RequestSchemaError(f"{path}.schema: unexpected")
    if request["fleet_id"] not in FLEET_IDS:
        raise RequestSchemaError(f"{path}.fleet_id: unexpected")
    if request["fleet_order"] != list(_SECTION_ORDERS[str(request["fleet_id"])]):
        raise RequestSchemaError(f"{path}.fleet_order: drift")
    if request["phase"] not in {"smoke", "production"}:
        raise RequestSchemaError(f"{path}.phase: unexpected")
    model = _nonempty(request["model"], f"{path}.model")
    prompt = _nonempty(request["prompt"], f"{path}.prompt")
    if "sonnet" not in model.casefold():
        raise RequestSchemaError(f"{path}.model: expected Sonnet")
    if request["request_sha256"] != _sha({"model": model, "prompt": prompt}):
        raise RequestSchemaError(f"{path}.request_sha256: digest mismatch")
    return request


def validate_bundle(bundle: object) -> dict[str, object]:
    if not isinstance(bundle, dict):
        raise RequestSchemaError("bundle: expected object")
    expected = {
        "schema",
        "status",
        "source_study_content_sha256",
        "input_scope",
        "model",
        "fleet_ids",
        "smoke_contract",
        "metrics",
        "requests",
        "corpus_accessed",
        "model_calls_executed",
        "bundle_content_sha256",
    }
    _exact_keys(bundle, expected, "bundle")
    if bundle["schema"] != BUNDLE_SCHEMA or bundle["input_scope"] != "metric_text_only":
        raise RequestSchemaError("bundle schema or input scope drift")
    if bundle["corpus_accessed"] is not False or bundle["model_calls_executed"] is not False:
        raise RequestSchemaError("compiled bundle incorrectly claims side effects")
    content = dict(bundle)
    claimed = content.pop("bundle_content_sha256")
    if claimed != _sha(content):
        raise RequestSchemaError("bundle_content_sha256 mismatch")
    metrics = bundle["metrics"]
    requests = bundle["requests"]
    if not isinstance(metrics, list) or not isinstance(requests, list):
        raise RequestSchemaError("bundle metrics/requests must be arrays")
    if len(requests) != 3 * len(metrics):
        raise RequestSchemaError("bundle must have three requests per metric")
    validated = [_validate_request(row, f"bundle.requests[{index}]") for index, row in enumerate(requests)]
    request_ids = [row["request_id"] for row in validated]
    if len(set(request_ids)) != len(request_ids):
        raise RequestSchemaError("bundle request_id values must be unique")
    smoke = [row for row in validated if row["phase"] == "smoke"]
    if len(smoke) != 6:
        raise RequestSchemaError("bundle smoke phase must contain exactly six requests")
    by_metric: dict[str, list[dict[str, object]]] = {}
    for row in validated:
        by_metric.setdefault(str(row["metric_id"]), []).append(row)
    for metric_id, rows in by_metric.items():
        if {row["fleet_id"] for row in rows} != set(FLEET_IDS):
            raise RequestSchemaError(f"metric {metric_id}: fleet coverage drift")
        if len({row["semantic_payload_sha256"] for row in rows}) != 1:
            raise RequestSchemaError(f"metric {metric_id}: semantic prompt drift")
    return bundle


_FENCE = re.compile(r"\A```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.DOTALL | re.IGNORECASE)


def parse_response(raw: str) -> tuple[CanonicalRelation, ...]:
    """Parse exactly one bare or fenced response object."""

    if not isinstance(raw, str) or not raw.strip():
        raise RequestSchemaError("response is empty")
    text = raw.strip()
    if text.startswith("```"):
        if text.count("```") != 2:
            raise RequestSchemaError("response has malformed or multiple fences")
        match = _FENCE.fullmatch(text)
        if match is None:
            raise RequestSchemaError("response has malformed or multiple fences")
        text = match.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RequestSchemaError("response is not one JSON value") from exc
    if not isinstance(value, dict):
        raise RequestSchemaError("response root must be one object")
    _exact_keys(value, {"relations"}, "response")
    rows = value["relations"]
    if not isinstance(rows, list) or not MIN_RELATIONS <= len(rows) <= MAX_RELATIONS:
        raise RequestSchemaError("response.relations must contain 2 to 5 objects")
    try:
        relations = tuple(
            CanonicalRelation.from_value(row, f"response.relations[{index}]")
            for index, row in enumerate(rows)
        )
    except DecompositionSchemaError as exc:
        raise RequestSchemaError(str(exc)) from exc
    if len(set(relations)) != len(relations):
        raise RequestSchemaError("response contains duplicate canonical relations")
    return relations


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


Executor = Callable[[Mapping[str, object]], CliResult]


def local_claude_executor(
    request: Mapping[str, object], *, claude_bin: str = "claude", timeout: int = 180
) -> CliResult:
    """Execute one fresh non-resuming local Claude process."""

    completed = subprocess.run(
        [
            claude_bin,
            "-p",
            "--model",
            str(request["model"]),
            "--output-format",
            "text",
        ],
        input=str(request["prompt"]),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CliResult(completed.returncode, completed.stdout, completed.stderr)


def _validate_response_row(
    row: object, requests: Mapping[str, Mapping[str, object]], path: str
) -> dict[str, object]:
    if not isinstance(row, dict):
        raise RequestSchemaError(f"{path}: expected object")
    expected = {
        "schema",
        "request_id",
        "request_sha256",
        "response_sha256",
        "returncode",
        "raw_response",
        "stderr",
        "valid",
        "parse_error",
        "relations",
    }
    _exact_keys(row, expected, path)
    request_id = row["request_id"]
    if request_id not in requests:
        raise RequestSchemaError(f"{path}.request_id: not in bundle")
    request = requests[str(request_id)]
    if row["request_sha256"] != request["request_sha256"]:
        raise RequestSchemaError(f"{path}.request_sha256: mismatch")
    raw = row["raw_response"]
    if not isinstance(raw, str) or row["response_sha256"] != _text_sha(raw):
        raise RequestSchemaError(f"{path}.response_sha256: mismatch")
    return row


def load_response_ledger(
    rows: Iterable[object], bundle: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    requests = {
        str(request["request_id"]): request
        for request in bundle["requests"]  # type: ignore[index]
    }
    result: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        validated = _validate_response_row(row, requests, f"ledger[{index}]")
        request_id = str(validated["request_id"])
        if request_id in result:
            raise RequestSchemaError(f"ledger: duplicate response for {request_id}")
        result[request_id] = validated
    return result


def _response_row(request: Mapping[str, object], result: CliResult) -> dict[str, object]:
    parse_error: str | None = None
    relations: list[dict[str, str]] | None = None
    if result.returncode == 0:
        try:
            relations = [relation.to_value() for relation in parse_response(result.stdout)]
        except RequestSchemaError as exc:
            parse_error = str(exc)
    else:
        parse_error = f"claude CLI exited {result.returncode}"
    return {
        "schema": RESPONSE_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "response_sha256": _text_sha(result.stdout),
        "returncode": result.returncode,
        "raw_response": result.stdout,
        "stderr": result.stderr,
        "valid": parse_error is None,
        "parse_error": parse_error,
        "relations": relations,
    }


def _smoke_status(
    bundle: Mapping[str, object], ledger: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    smoke = [request for request in bundle["requests"] if request["phase"] == "smoke"]  # type: ignore[index]
    completed = [ledger.get(str(request["request_id"])) for request in smoke]
    valid = sum(row is not None and row["valid"] is True for row in completed)
    return {
        "required": 6,
        "completed": sum(row is not None for row in completed),
        "valid": valid,
        "passed": len(completed) == 6 and valid == 6,
    }


def run_phase(
    bundle: object,
    existing_rows: Sequence[object],
    *,
    phase: Literal["smoke", "production"],
    executor: Executor,
    max_concurrency: int = 4,
    row_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run missing requests for one phase; production is smoke-gated."""

    validated_bundle = validate_bundle(bundle)
    if phase not in {"smoke", "production"}:
        raise RequestSchemaError("phase must be smoke or production")
    if not 1 <= max_concurrency <= 4:
        raise RequestSchemaError("max_concurrency must be between 1 and 4")
    ledger = load_response_ledger(existing_rows, validated_bundle)
    before = _smoke_status(validated_bundle, ledger)
    if phase == "production" and before["passed"] is not True:
        raise RequestSchemaError(
            f"production blocked: smoke must be 6/6 valid, observed {before['valid']}/6"
        )
    pending = [
        request
        for request in validated_bundle["requests"]
        if request["phase"] == phase and request["request_id"] not in ledger
    ]
    new_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {pool.submit(executor, request): request for request in pending}
        for future in as_completed(futures):
            request = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # execution failure is recorded, not hidden
                result = CliResult(-1, "", f"{type(exc).__name__}: {exc}")
            if not isinstance(result, CliResult):
                raise TypeError("executor must return CliResult")
            response = _response_row(request, result)
            new_rows.append(response)
            if row_sink is not None:
                row_sink(response)
    new_rows.sort(key=lambda row: str(row["request_id"]))
    combined = list(existing_rows) + new_rows
    final_ledger = load_response_ledger(combined, validated_bundle)
    after = _smoke_status(validated_bundle, final_ledger)
    phase_rows = [
        final_ledger[str(request["request_id"])]
        for request in validated_bundle["requests"]
        if request["phase"] == phase and str(request["request_id"]) in final_ledger
    ]
    summary = {
        "phase": phase,
        "scheduled": len(pending),
        "phase_expected": sum(
            request["phase"] == phase for request in validated_bundle["requests"]
        ),
        "phase_completed": len(phase_rows),
        "phase_valid": sum(row["valid"] is True for row in phase_rows),
        "smoke": after,
        "production_unblocked": after["passed"],
    }
    return new_rows, summary


def build_metric_submissions(
    bundle: object, response_rows: Sequence[object], *, require_all: bool = True
) -> dict[str, dict[str, object]]:
    """Build strict decomposition-stability submissions for complete metrics."""

    validated_bundle = validate_bundle(bundle)
    ledger = load_response_ledger(response_rows, validated_bundle)
    requests_by_metric: dict[str, list[Mapping[str, object]]] = {}
    for request in validated_bundle["requests"]:
        requests_by_metric.setdefault(str(request["metric_id"]), []).append(request)
    metric_rows = {
        str(metric["metric_id"]): metric for metric in validated_bundle["metrics"]
    }
    submissions: dict[str, dict[str, object]] = {}
    incomplete: list[str] = []
    for metric_id, requests in requests_by_metric.items():
        response_group = [ledger.get(str(request["request_id"])) for request in requests]
        if any(row is None or row["valid"] is not True for row in response_group):
            incomplete.append(metric_id)
            continue
        metric = metric_rows[metric_id]
        metric_text = metric["metric_text"]
        fleets = []
        for fleet_id in FLEET_IDS:
            request = next(request for request in requests if request["fleet_id"] == fleet_id)
            response = ledger[str(request["request_id"])]
            fleets.append({"fleet_id": fleet_id, "relations": response["relations"]})
        submissions[metric_id] = {
            "schema": SUBMISSION_SCHEMA,
            "metric": {
                "name": metric_text["construct"],
                "text": metric_text["description"],
            },
            "fleets": fleets,
        }
    if require_all and incomplete:
        raise RequestSchemaError(
            f"cannot emit submissions; incomplete metrics={len(incomplete)}"
        )
    return submissions


def _read_jsonl(path: Path) -> list[object]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RequestSchemaError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def derive_first_response_ledger(
    bundle: object, rows: Sequence[object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Derive a deterministic ledger after a disclosed concurrent-run race.

    The raw append-only ledger remains untouched.  First occurrence is chosen
    by raw ledger order; duplicate responses are counted and never compared or
    selected using their content or validity.
    """

    validated_bundle = validate_bundle(bundle)
    requests = {
        str(request["request_id"]): request
        for request in validated_bundle["requests"]
    }
    selected: dict[str, dict[str, object]] = {}
    duplicate_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        validated = _validate_response_row(row, requests, f"raw_ledger[{index}]")
        request_id = str(validated["request_id"])
        if request_id in selected:
            duplicate_counts[request_id] = duplicate_counts.get(request_id, 0) + 1
        else:
            selected[request_id] = validated
    derived = list(selected.values())
    report = {
        "schema": "metric-seam.decomposition-ledger-derivation.v1",
        "policy": "first occurrence in append-only raw ledger; outcome-blind",
        "raw_rows": len(rows),
        "derived_rows": len(derived),
        "duplicate_rows_removed": len(rows) - len(derived),
        "duplicate_request_counts": dict(sorted(duplicate_counts.items())),
        "raw_rows_sha256": _sha(rows),
        "derived_rows_sha256": _sha(derived),
    }
    return derived, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--study", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--model", default="sonnet")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--requests", type=Path, required=True)
    run_parser.add_argument("--ledger", type=Path, required=True)
    run_parser.add_argument("--phase", choices=("smoke", "production"), required=True)
    run_parser.add_argument("--max-concurrency", type=int, default=4)
    run_parser.add_argument("--claude-bin", default="claude")
    run_parser.add_argument("--timeout", type=int, default=180)

    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--requests", type=Path, required=True)
    emit_parser.add_argument("--ledger", type=Path, required=True)
    emit_parser.add_argument("--output-dir", type=Path, required=True)
    emit_parser.add_argument("--allow-partial", action="store_true")

    dedupe_parser = subparsers.add_parser("derive-first-ledger")
    dedupe_parser.add_argument("--requests", type=Path, required=True)
    dedupe_parser.add_argument("--raw-ledger", type=Path, required=True)
    dedupe_parser.add_argument("--output", type=Path, required=True)
    dedupe_parser.add_argument("--report", type=Path, required=True)

    recover_parser = subparsers.add_parser("recover-one")
    recover_parser.add_argument("--requests", type=Path, required=True)
    recover_parser.add_argument("--request-id", required=True)
    recover_parser.add_argument("--output", type=Path, required=True)
    recover_parser.add_argument("--claude-bin", default="claude")
    recover_parser.add_argument("--timeout", type=int, default=180)

    merge_parser = subparsers.add_parser("apply-recovery")
    merge_parser.add_argument("--requests", type=Path, required=True)
    merge_parser.add_argument("--ledger", type=Path, required=True)
    merge_parser.add_argument("--recovery", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--report", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "compile":
        study = json.loads(args.study.read_text(encoding="utf-8"))
        bundle = compile_requests(study, model=args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(args.output)
        return 0

    bundle = json.loads(args.requests.read_text(encoding="utf-8"))
    if args.command == "recover-one":
        validated = validate_bundle(bundle)
        request = next(
            (row for row in validated["requests"] if row["request_id"] == args.request_id),
            None,
        )
        if request is None:
            raise RequestSchemaError("recovery request_id is not in the bundle")
        response = _response_row(
            request,
            local_claude_executor(request, claude_bin=args.claude_bin, timeout=args.timeout),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"request_id": args.request_id, "valid": response["valid"]}, sort_keys=True))
        return 0 if response["valid"] else 2

    if args.command == "apply-recovery":
        base = _read_jsonl(args.ledger)
        ledger = load_response_ledger(base, validate_bundle(bundle))
        recovery = json.loads(args.recovery.read_text(encoding="utf-8"))
        requests = {str(row["request_id"]): row for row in bundle["requests"]}
        replacement = _validate_response_row(recovery, requests, "recovery")
        request_id = str(replacement["request_id"])
        if request_id not in ledger or ledger[request_id]["valid"] is not False:
            raise RequestSchemaError("recovery may replace exactly one retained invalid row")
        if replacement["valid"] is not True:
            raise RequestSchemaError("recovery response is not valid")
        merged = [replacement if row["request_id"] == request_id else row for row in base]
        load_response_ledger(merged, validate_bundle(bundle))
        report = {
            "schema": "metric-seam.decomposition-recovery-merge.v1",
            "policy": "same frozen request retried once; original invalid ledger retained",
            "request_id": request_id,
            "original_response_sha256": ledger[request_id]["response_sha256"],
            "recovery_response_sha256": replacement["response_sha256"],
            "base_rows_sha256": _sha(base),
            "merged_rows_sha256": _sha(merged),
        }
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged),
            encoding="utf-8",
        )
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0

    if args.command == "derive-first-ledger":
        rows = _read_jsonl(args.raw_ledger)
        derived, report = derive_first_response_ledger(bundle, rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in derived),
            encoding="utf-8",
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True))
        return 0

    rows = _read_jsonl(args.ledger)
    if args.command == "run":
        executor = lambda request: local_claude_executor(  # noqa: E731
            request, claude_bin=args.claude_bin, timeout=args.timeout
        )
        new_rows, summary = run_phase(
            bundle,
            rows,
            phase=args.phase,
            executor=executor,
            max_concurrency=args.max_concurrency,
            row_sink=lambda row: _append_jsonl(args.ledger, [row]),
        )
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["phase_valid"] == summary["phase_expected"] else 2

    submissions = build_metric_submissions(
        bundle, rows, require_all=not args.allow_partial
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric_id, submission in submissions.items():
        filename = "metric_" + hashlib.sha256(metric_id.encode()).hexdigest()[:20] + ".json"
        (args.output_dir / filename).write_text(
            json.dumps(submission, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"emitted": len(submissions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
