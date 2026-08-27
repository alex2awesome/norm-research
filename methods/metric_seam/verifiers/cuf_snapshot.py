"""Import a CUF bank immutably and prepare conservative code-review joins.

This module is deliberately limited to Stage 0 of the verifier roadmap.  It
copies an already-produced, executor-indexed CUF bank into a content-addressed
local snapshot and joins current code-review program names to CUF metric names.
Only a unique equality after the documented Unicode/whitespace normalization
is accepted automatically.  Ambiguous and unmatched names are emitted to a
review queue; this module never performs fuzzy or semantic matching.

Examples::

    ssh sk3 'cat /lfs/skampere3/0/alexspan/outputs/unit_cert/bank/code-review/llama8b/bank_units.jsonl' \
      | python -m methods.metric_seam.verifiers.cuf_snapshot snapshot \
          --source - \
          --source-label ssh://sk3/lfs/skampere3/0/alexspan/outputs/unit_cert/bank/code-review/llama8b/bank_units.jsonl \
          --snapshot-root outputs/metric_seam_pilot/hierarchy_r123/cuf_snapshots \
          --task code-review --executor llama8b

    python -m methods.metric_seam.verifiers.cuf_snapshot join \
      --snapshot-manifest outputs/metric_seam_pilot/hierarchy_r123/cuf_snapshots/code-review/llama8b/<sha256>/manifest.json \
      --cell-manifest outputs/metric_seam_pilot/hierarchy_r123/code_review_reconstruction_prompt_manifest_v3.json \
      --construct-fidelity outputs/metric_seam_pilot/hierarchy_r123/code_review_construct_fidelity_v2.json \
      --output outputs/metric_seam_pilot/hierarchy_r123/code_review_cuf_llama8b_join_candidates_v1.json
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import BinaryIO, Iterable, Mapping, Sequence
import unicodedata


SNAPSHOT_SCHEMA = "metric-seam.cuf-bank-snapshot.v1"
JOIN_SCHEMA = "metric-seam.code-review-cuf-join-candidates.v1"

BANK_REQUIRED_FIELDS = frozenset({"metric", "k", "rows", "meta"})
UNIT_REQUIRED_FIELDS = frozenset(
    {
        "node_id",
        "level",
        "span",
        "delta_free",
        "p_free",
        "delta_M",
        "p_M",
        "sign_stability",
        "kappa",
        "eps_ctx",
        "verdict",
        "atom",
        "detect_free",
        "detect_M",
        "certified_lo",
    }
)
NUMERIC_UNIT_FIELDS = frozenset(
    {
        "delta_free",
        "p_free",
        "delta_M",
        "p_M",
        "sign_stability",
        "kappa",
        "eps_ctx",
        "certified_lo",
    }
)

_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
    }
)


class CufValidationError(ValueError):
    """Raised when an imported CUF bank violates its frozen row contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_metric_name(value: str) -> str:
    """Return the conservative equality key used for automatic joins.

    This is not a similarity transform: it applies Unicode NFKC, case folding,
    typographic-dash canonicalization, and whitespace collapse only.  It does
    not remove punctuation, words, parentheticals, or conjunctions.
    """

    if not isinstance(value, str):
        raise TypeError("metric name must be a string")
    normalized = unicodedata.normalize("NFKC", value).translate(_DASH_TRANSLATION)
    return " ".join(normalized.casefold().split())


def _is_finite_number_or_none(value: object) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_unit(unit: object, *, line_number: int, unit_index: int) -> dict:
    where = f"line {line_number} unit {unit_index}"
    if not isinstance(unit, dict):
        raise CufValidationError(f"{where}: expected an object")
    missing = sorted(UNIT_REQUIRED_FIELDS - unit.keys())
    if missing:
        raise CufValidationError(f"{where}: missing fields {missing}")
    if not isinstance(unit["node_id"], int) or isinstance(unit["node_id"], bool):
        raise CufValidationError(f"{where}: node_id must be an integer")
    if (
        not isinstance(unit["level"], int)
        or isinstance(unit["level"], bool)
        or unit["level"] < 1
    ):
        raise CufValidationError(f"{where}: level must be a positive integer")
    if not isinstance(unit["span"], str) or not unit["span"].strip():
        raise CufValidationError(f"{where}: span must be a nonempty string")
    if not isinstance(unit["verdict"], str) or not unit["verdict"].strip():
        raise CufValidationError(f"{where}: verdict must be a nonempty string")
    if unit["atom"] is not None and not isinstance(unit["atom"], str):
        raise CufValidationError(f"{where}: atom must be a string or null")
    for field in ("detect_free", "detect_M"):
        if not isinstance(unit[field], bool):
            raise CufValidationError(f"{where}: {field} must be boolean")
    for field in NUMERIC_UNIT_FIELDS:
        if not _is_finite_number_or_none(unit[field]):
            raise CufValidationError(f"{where}: {field} must be finite numeric or null")
    return unit


def validate_bank_bytes(data: bytes) -> tuple[list[dict], dict]:
    """Decode and validate CUF JSONL, returning rows and deterministic counts."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CufValidationError(f"CUF bank is not valid UTF-8: {exc}") from exc

    rows: list[dict] = []
    k_counts: Counter[int] = Counter()
    metric_counts: Counter[str] = Counter()
    normalized_metric_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    certified_units = 0
    unit_count = 0
    blank_lines = 0

    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            blank_lines += 1
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CufValidationError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise CufValidationError(f"line {line_number}: expected an object")
        missing = sorted(BANK_REQUIRED_FIELDS - row.keys())
        if missing:
            raise CufValidationError(f"line {line_number}: missing fields {missing}")
        metric = row["metric"]
        if not isinstance(metric, str) or not metric.strip():
            raise CufValidationError(f"line {line_number}: metric must be a nonempty string")
        if not isinstance(row["k"], int) or isinstance(row["k"], bool) or row["k"] < 0:
            raise CufValidationError(f"line {line_number}: k must be a nonnegative integer")
        if not isinstance(row["rows"], list):
            raise CufValidationError(f"line {line_number}: rows must be an array")
        if not isinstance(row["meta"], dict):
            raise CufValidationError(f"line {line_number}: meta must be an object")

        for unit_index, unit in enumerate(row["rows"]):
            checked = _validate_unit(unit, line_number=line_number, unit_index=unit_index)
            unit_count += 1
            verdict = checked["verdict"]
            verdict_counts[verdict] += 1
            level_counts[str(checked["level"])] += 1
            # CUF certification is the bank's exact categorical verdict.  The
            # detector flags and diagnostic strings are not substitutes for it.
            if verdict == "CERTIFIED-UNIT":
                certified_units += 1

        rows.append(row)
        k_counts[row["k"]] += 1
        metric_counts[metric] += 1
        normalized_metric_counts[normalize_metric_name(metric)] += 1

    if not rows:
        raise CufValidationError("CUF bank contains no nonblank rows")

    counts = {
        "bytes": len(data),
        "blank_lines_ignored_during_validation": blank_lines,
        "bank_rows": len(rows),
        "unique_metric_names": len(metric_counts),
        "unique_normalized_metric_names": len(normalized_metric_counts),
        "unit_rows": unit_count,
        "certified_unit_rows": certified_units,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "level_counts": dict(sorted(level_counts.items())),
        "duplicate_k_values": sorted(k for k, count in k_counts.items() if count > 1),
        "duplicate_metric_names": sorted(
            metric for metric, count in metric_counts.items() if count > 1
        ),
        "duplicate_normalized_metric_names": sorted(
            name for name, count in normalized_metric_counts.items() if count > 1
        ),
    }
    return rows, counts


def _read_source(source: str, stdin: BinaryIO) -> tuple[bytes, str, str]:
    if source == "-":
        data = stdin.read()
        if not isinstance(data, bytes):
            raise TypeError("stdin must be opened in binary mode")
        return data, "stdin", "stdin"
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_bytes(), "local_path", str(path)


def snapshot_bank(
    *,
    source: str,
    source_label: str | None,
    snapshot_root: Path,
    task: str,
    executor: str,
    stdin: BinaryIO | None = None,
) -> dict:
    """Validate and content-address a CUF bank without overwriting prior data."""

    if not task.strip() or not executor.strip():
        raise ValueError("task and executor must be nonempty")
    data, transport, default_locator = _read_source(source, stdin or sys.stdin.buffer)
    if source == "-" and not source_label:
        raise ValueError("--source-label is required when --source=-")
    locator = source_label or default_locator
    _, counts = validate_bank_bytes(data)
    digest = sha256_bytes(data)
    snapshot_dir = snapshot_root / task / executor / digest
    bank_path = snapshot_dir / "bank_units.jsonl"
    manifest_path = snapshot_dir / "manifest.json"
    index_path = snapshot_root / "index.jsonl"

    manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "status": "validated_immutable_snapshot",
        "task": task,
        "executor": executor,
        "snapshot_id": digest,
        "source": {
            "locator": locator,
            "transport": transport,
            "sha256": digest,
        },
        "snapshot": {
            "bank_path": str(bank_path),
            "manifest_path": str(manifest_path),
            "sha256": digest,
        },
        "validation": {
            "required_bank_fields": sorted(BANK_REQUIRED_FIELDS),
            "required_unit_fields": sorted(UNIT_REQUIRED_FIELDS),
            **counts,
        },
        "claim_limits": {
            "semantic_join_performed": False,
            "model_or_gpu_used": False,
            "executor_banks_pooled": False,
        },
    }

    if snapshot_dir.exists():
        if not bank_path.is_file() or not manifest_path.is_file():
            raise FileExistsError(f"incomplete existing snapshot: {snapshot_dir}")
        if sha256_file(bank_path) != digest:
            raise FileExistsError(f"existing snapshot bytes disagree with its id: {snapshot_dir}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("snapshot_id") != digest:
            raise FileExistsError(f"existing manifest disagrees with snapshot id: {manifest_path}")
        return existing

    snapshot_dir.mkdir(parents=True, exist_ok=False)
    with bank_path.open("xb") as handle:
        handle.write(data)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    snapshot_root.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": "metric-seam.cuf-bank-snapshot-index-row.v1",
                    "task": task,
                    "executor": executor,
                    "snapshot_id": digest,
                    "manifest_path": str(manifest_path),
                    "source_locator": locator,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    return manifest


def _literal_module_constants(path: Path, names: Iterable[str]) -> dict[str, object]:
    wanted = set(names)
    found: dict[str, object] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    found[target.id] = ast.literal_eval(value)
                except (TypeError, ValueError):
                    continue
    return found


def _certified_unit_count(row: Mapping) -> int:
    # Certification is a bank verdict, not an inference from correlated
    # diagnostics such as detect_free/detect_M.  Keeping this exact prevents a
    # future subthreshold row with a positive detector bit from being promoted
    # during the Stage-0 join.
    return sum(unit.get("verdict") == "CERTIFIED-UNIT" for unit in row["rows"])


def build_code_review_join(
    *,
    snapshot_manifest: Path,
    cell_manifest: Path,
    construct_fidelity: Path,
    repo_root: Path,
    expected_cells: int = 18,
) -> dict:
    """Build exact-name joins plus a complete non-semantic review queue."""

    snapshot = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"unsupported snapshot schema: {snapshot.get('schema')!r}")
    bank_path = Path(snapshot["snapshot"]["bank_path"])
    if not bank_path.is_absolute():
        bank_path = (repo_root / bank_path).resolve()
    bank_bytes = bank_path.read_bytes()
    if sha256_bytes(bank_bytes) != snapshot["snapshot_id"]:
        raise ValueError("snapshot bank SHA-256 does not match snapshot manifest")
    bank_rows, bank_counts = validate_bank_bytes(bank_bytes)

    cells_payload = json.loads(cell_manifest.read_text(encoding="utf-8"))
    cell_ids = cells_payload.get("cell_ids")
    if not isinstance(cell_ids, list) or not all(isinstance(x, str) for x in cell_ids):
        raise ValueError("cell manifest must contain string array cell_ids")
    if len(cell_ids) != expected_cells or len(set(cell_ids)) != expected_cells:
        raise ValueError(
            f"expected {expected_cells} unique current cells, found {len(set(cell_ids))}"
        )

    fidelity_payload = json.loads(construct_fidelity.read_text(encoding="utf-8"))
    fidelity_by_cell = {row["cell_id"]: row for row in fidelity_payload.get("rows", [])}
    missing_cells = sorted(set(cell_ids) - fidelity_by_cell.keys())
    if missing_cells:
        raise ValueError(f"construct-fidelity artifact omits cells: {missing_cells}")

    bank_by_normalized: dict[str, list[dict]] = defaultdict(list)
    for bank_row in bank_rows:
        bank_by_normalized[normalize_metric_name(bank_row["metric"])].append(bank_row)

    result_rows = []
    review_queue = []
    status_counts: Counter[str] = Counter()
    for cell_id in cell_ids:
        fidelity = fidelity_by_cell[cell_id]
        candidate = fidelity.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"current cell {cell_id} has no candidate binding")
        source_path = Path(candidate["source_path"])
        if not source_path.is_absolute():
            source_path = (repo_root / source_path).resolve()
        constants = _literal_module_constants(source_path, ("ASPECT_ID", "ASPECT_NAME"))
        aspect_name = constants.get("ASPECT_NAME")
        if not isinstance(aspect_name, str) or not aspect_name.strip():
            raise ValueError(f"candidate source has no literal ASPECT_NAME: {source_path}")
        declared_aspect = constants.get("ASPECT_ID")
        if declared_aspect is not None and str(declared_aspect) != str(candidate["aspect_id"]):
            raise ValueError(f"candidate ASPECT_ID disagrees for {cell_id}")

        key = normalize_metric_name(aspect_name)
        matches = bank_by_normalized.get(key, [])
        if len(matches) == 1:
            status = "auto_accepted_exact_normalized_unique"
            selected = matches[0]
            exact_candidates = [selected["metric"]]
            selected_summary = {
                "metric": selected["metric"],
                "k": selected["k"],
                "unit_rows": len(selected["rows"]),
                "certified_unit_rows": _certified_unit_count(selected),
            }
        elif len(matches) > 1:
            status = "queued_ambiguous_exact_normalized_for_sonnet"
            exact_candidates = [row["metric"] for row in matches]
            selected_summary = None
        else:
            status = "queued_unmatched_for_sonnet"
            exact_candidates = []
            selected_summary = None
        status_counts[status] += 1

        result = {
            "cell_id": cell_id,
            "level": fidelity.get("level"),
            "parent_metric_name": fidelity.get("metric_name"),
            "candidate_aspect_id": candidate["aspect_id"],
            "candidate_metric_name": aspect_name,
            "candidate_metric_name_normalized": key,
            "candidate_source_path": candidate["source_path"],
            "join_status": status,
            "exact_normalized_bank_candidates": exact_candidates,
            "selected_bank_metric": selected_summary,
            "semantic_adjudication_performed": False,
        }
        result_rows.append(result)
        if selected_summary is None:
            review_queue.append(
                {
                    "cell_id": cell_id,
                    "candidate_aspect_id": candidate["aspect_id"],
                    "candidate_metric_name": aspect_name,
                    "parent_metric_name": fidelity.get("metric_name"),
                    "queue_reason": status,
                    "exact_normalized_bank_candidates": exact_candidates,
                    "required_next_action": "Sonnet-or-better semantic name adjudication",
                }
            )

    return {
        "schema": JOIN_SCHEMA,
        "status": "exact_normalized_join_complete_semantic_review_pending",
        "task": "code-review",
        "executor": snapshot["executor"],
        "source_snapshot": {
            "manifest_path": str(snapshot_manifest),
            "snapshot_id": snapshot["snapshot_id"],
            "bank_path": str(bank_path),
            "bank_counts": bank_counts,
        },
        "source_cells": {
            "cell_manifest_path": str(cell_manifest),
            "cell_manifest_sha256": sha256_file(cell_manifest),
            "construct_fidelity_path": str(construct_fidelity),
            "construct_fidelity_sha256": sha256_file(construct_fidelity),
            "expected_cells": expected_cells,
        },
        "join_policy": {
            "normalization": "Unicode NFKC + typographic-dash folding + casefold + whitespace collapse",
            "automatic_acceptance": "exactly one CUF bank row has the same normalized metric name",
            "ambiguous_policy": "queue_for_sonnet",
            "unmatched_policy": "queue_for_sonnet",
            "fuzzy_matching_performed": False,
            "semantic_adjudication_performed": False,
        },
        "summary": {
            "cells": len(result_rows),
            "unique_candidate_aspects": len({row["candidate_aspect_id"] for row in result_rows}),
            "status_counts": dict(sorted(status_counts.items())),
            "review_queue_cells": len(review_queue),
        },
        "rows": result_rows,
        "review_queue": review_queue,
        "claim_limits": {
            "cuf_units_are_executor_indexed": True,
            "executor_banks_pooled": False,
            "semantic_metric_identity_established_for_queued_rows": False,
            "verifier_authored_or_certified": False,
            "model_or_gpu_used": False,
        },
    }


def write_new_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _snapshot_command(args: argparse.Namespace) -> int:
    manifest = snapshot_bank(
        source=args.source,
        source_label=args.source_label,
        snapshot_root=args.snapshot_root,
        task=args.task,
        executor=args.executor,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _join_command(args: argparse.Namespace) -> int:
    payload = build_code_review_join(
        snapshot_manifest=args.snapshot_manifest,
        cell_manifest=args.cell_manifest,
        construct_fidelity=args.construct_fidelity,
        repo_root=args.repo_root.resolve(),
        expected_cells=args.expected_cells,
    )
    write_new_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="validate and immutably snapshot CUF JSONL")
    snapshot.add_argument("--source", required=True, help="local bank_units.jsonl path, or - for stdin")
    snapshot.add_argument("--source-label", help="required provenance locator for stdin")
    snapshot.add_argument("--snapshot-root", type=Path, required=True)
    snapshot.add_argument("--task", required=True)
    snapshot.add_argument("--executor", required=True)
    snapshot.set_defaults(func=_snapshot_command)

    join = subparsers.add_parser("join", help="prepare exact-name code-review CUF joins")
    join.add_argument("--snapshot-manifest", type=Path, required=True)
    join.add_argument("--cell-manifest", type=Path, required=True)
    join.add_argument("--construct-fidelity", type=Path, required=True)
    join.add_argument("--repo-root", type=Path, default=Path.cwd())
    join.add_argument("--expected-cells", type=int, default=18)
    join.add_argument("--output", type=Path, required=True)
    join.set_defaults(func=_join_command)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
