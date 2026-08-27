"""Measure code-review target resolution with tie-aware tercile AUC.

This is an additive, CPU-only reanalysis of the frozen code-review ceiling run.
It does not call a model, reparse model text, or alter any historical artifact.

The code score is the target.  Items are sorted by ``(score, item_key)`` and
partitioned into three deterministic, near-equal nominal chunks.  The bottom
and top chunks are then expanded to include every item tied at their respective
cut points.  A mapping has usable target spread only when the two cut values
differ.  On the intersection where both prompt passes emitted a scalar score,
the readout is the Mann-Whitney probability that a top-tercile prediction is
larger than a bottom-tercile prediction; prediction ties contribute one half.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_EXECUTION = BASE / "code_review_heldout_execution_v1.json"
DEFAULT_JOBS = BASE / "code_review_reconstruction_ceiling_jobs_v1.jsonl.gz"
DEFAULT_RESPONSES = (
    BASE / "results/code_review_glm52_ceiling_v1/responses.jsonl"
)
DEFAULT_SOURCE_READOUT = (
    BASE / "results/code_review_glm52_ceiling_v1/readout.json"
)
DEFAULT_RECOVERED_JOBS = BASE / "code_review_reconstruction_prompt_jobs_v3.jsonl.gz"
DEFAULT_RECOVERED_RESPONSES = (
    BASE / "results/code_review_glm52_impl_summary_v2_recovered/responses.jsonl"
)
DEFAULT_OUTPUT = (
    BASE / "results/code_review_target_resolution_v1/readout.json"
)
CHANNEL = "full_executable_contract"
COMPARISON_CHANNEL = "implementation_disclosed"
SCHEMA = "metric-seam.code-review-target-resolution.v1"
EXPECTED_CELLS = 18
EXPECTED_ITEMS = 125
EXPECTED_PASSES = (1, 2)


class TargetResolutionError(ValueError):
    """Raised when frozen inputs do not satisfy the analysis contract."""


@dataclass(frozen=True)
class TercilePartition:
    bottom: tuple[str, ...]
    middle_nominal: tuple[str, ...]
    top: tuple[str, ...]
    bottom_nominal_n: int
    top_nominal_n: int
    bottom_boundary: float
    top_boundary: float

    @property
    def has_spread(self) -> bool:
        return self.bottom_boundary < self.top_boundary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Prefer a repository-relative path so the readout is portable."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _request_sha256(request: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(request))).hexdigest()


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TargetResolutionError(f"expected JSON object: {path}")
    return value


def _split_sizes(n: int) -> tuple[int, int, int]:
    quotient, remainder = divmod(n, 3)
    return tuple(
        quotient + (1 if index < remainder else 0) for index in range(3)
    )  # type: ignore[return-value]


def tie_expanded_terciles(targets: Mapping[str, float]) -> TercilePartition:
    """Return deterministic nominal thirds with boundary-score ties preserved."""

    if len(targets) < 3:
        raise TargetResolutionError("at least three scored targets are required")
    if any(not math.isfinite(float(value)) for value in targets.values()):
        raise TargetResolutionError("target values must be finite")
    ordered = sorted(targets, key=lambda item: (float(targets[item]), item))
    n_bottom, n_middle, n_top = _split_sizes(len(ordered))
    bottom_nominal = ordered[:n_bottom]
    middle_nominal = ordered[n_bottom : n_bottom + n_middle]
    top_nominal = ordered[n_bottom + n_middle :]
    if not bottom_nominal or not top_nominal:
        raise TargetResolutionError("tercile partition has an empty extreme")
    bottom_boundary = float(targets[bottom_nominal[-1]])
    top_boundary = float(targets[top_nominal[0]])
    bottom = tuple(
        item for item in ordered if float(targets[item]) <= bottom_boundary
    )
    top = tuple(item for item in ordered if float(targets[item]) >= top_boundary)
    return TercilePartition(
        bottom=bottom,
        middle_nominal=tuple(middle_nominal),
        top=top,
        bottom_nominal_n=len(bottom_nominal),
        top_nominal_n=len(top_nominal),
        bottom_boundary=bottom_boundary,
        top_boundary=top_boundary,
    )


def mann_whitney_auc(
    bottom_predictions: Sequence[float], top_predictions: Sequence[float]
) -> float | None:
    """P(top > bottom) with prediction ties worth one half."""

    if not bottom_predictions or not top_predictions:
        return None
    wins = 0.0
    for bottom in bottom_predictions:
        for top in top_predictions:
            wins += float(top > bottom) + 0.5 * float(top == bottom)
    return wins / (len(bottom_predictions) * len(top_predictions))


def bootstrap_auc(
    bottom_predictions: Sequence[float],
    top_predictions: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Stratified item bootstrap of the Mann-Whitney AUC."""

    if draws < 1:
        raise TargetResolutionError("bootstrap draws must be positive")
    if not bottom_predictions or not top_predictions:
        return np.asarray([], dtype=float)
    bottom = np.asarray(bottom_predictions, dtype=float)
    top = np.asarray(top_predictions, dtype=float)
    outcome = (top[None, :] > bottom[:, None]).astype(float)
    outcome += 0.5 * (top[None, :] == bottom[:, None])
    rng = np.random.default_rng(seed)
    bottom_counts = rng.multinomial(
        len(bottom), np.full(len(bottom), 1.0 / len(bottom)), size=draws
    )
    top_counts = rng.multinomial(
        len(top), np.full(len(top), 1.0 / len(top)), size=draws
    )
    numerators = np.einsum(
        "bi,ij,bj->b", bottom_counts, outcome, top_counts, optimize=True
    )
    return numerators / float(len(bottom) * len(top))


def _interval(values: np.ndarray) -> list[float | None]:
    if not len(values):
        return [None, None]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _cell_seed(seed: int, cell_id: str) -> int:
    material = f"{seed}\0{cell_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _validate_scalar(payload: object) -> dict:
    if not isinstance(payload, Mapping):
        raise TargetResolutionError("parsed response is not an object")
    status = payload.get("measurement_status")
    if status not in {"not_applicable", "applicable_abstain", "scored"}:
        raise TargetResolutionError("invalid response measurement_status")
    expected = {"measurement_status", "evidence", "rationale"}
    if status == "scored":
        expected.add("score")
    if set(payload) != expected:
        raise TargetResolutionError("response fields do not match status")
    if not isinstance(payload.get("evidence"), list) or not all(
        isinstance(value, str) for value in payload["evidence"]
    ):
        raise TargetResolutionError("response evidence is not a string list")
    if not isinstance(payload.get("rationale"), str) or not payload["rationale"].strip():
        raise TargetResolutionError("response rationale is empty")
    result = dict(payload)
    if status == "scored":
        score = payload.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise TargetResolutionError("response score is outside [0,1]")
        result["score"] = float(score)
    return result


def _load_jobs(path: Path, channel: str) -> tuple[dict[str, dict], dict[str, dict]]:
    jobs: dict[str, dict] = {}
    cells: dict[str, dict] = {}
    slots: set[tuple[str, str, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            metadata = row.get("audit_metadata") if isinstance(row, Mapping) else None
            if not isinstance(metadata, Mapping) or metadata.get("channel") != channel:
                continue
            request_id = row.get("request_id")
            request = row.get("request")
            if (
                not isinstance(request_id, str)
                or request_id in jobs
                or not isinstance(request, Mapping)
                or set(request) != {"system", "user"}
            ):
                raise TargetResolutionError(f"invalid job at {path}:{line_number}")
            cell_id = metadata.get("cell_id")
            item_key = metadata.get("item_key")
            pass_id = metadata.get("pass_id")
            aspect_id = metadata.get("aspect_id")
            if (
                not isinstance(cell_id, str)
                or not isinstance(item_key, str)
                or pass_id not in EXPECTED_PASSES
                or not isinstance(aspect_id, str)
            ):
                raise TargetResolutionError("job identity is incomplete")
            slot = (cell_id, item_key, int(pass_id))
            if slot in slots:
                raise TargetResolutionError(f"duplicate job slot: {slot}")
            slots.add(slot)
            jobs[request_id] = row
            cell = cells.setdefault(
                cell_id,
                {
                    "aspect_id": aspect_id,
                    "level": metadata.get("level"),
                    "source_path": metadata.get("source_path"),
                    "source_sha256": metadata.get("source_sha256"),
                    "item_keys": set(),
                    "passes": set(),
                },
            )
            for key in ("aspect_id", "level", "source_path", "source_sha256"):
                if cell[key] != metadata.get(key):
                    raise TargetResolutionError(f"inconsistent cell metadata: {cell_id}")
            cell["item_keys"].add(item_key)
            cell["passes"].add(pass_id)
    if len(cells) != EXPECTED_CELLS:
        raise TargetResolutionError(
            f"expected {EXPECTED_CELLS} cells, observed {len(cells)}"
        )
    for cell_id, cell in cells.items():
        if len(cell["item_keys"]) != EXPECTED_ITEMS or cell["passes"] != set(EXPECTED_PASSES):
            raise TargetResolutionError(f"incomplete cell panel: {cell_id}")
    expected_jobs = EXPECTED_CELLS * EXPECTED_ITEMS * len(EXPECTED_PASSES)
    if len(jobs) != expected_jobs:
        raise TargetResolutionError(
            f"expected {expected_jobs} jobs, observed {len(jobs)}"
        )
    return jobs, cells


def _load_predictions(
    path: Path, jobs: Mapping[str, dict]
) -> tuple[dict[str, dict[int, dict[str, float]]], dict]:
    predictions: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: {1: {}, 2: {}}
    )
    status_counts: Counter[str] = Counter()
    measurement_counts: Counter[str] = Counter()
    seen: set[str] = set()
    fenced = valid_fenced = literal_tab = valid_literal_tab = 0
    explicit_parse_policy_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, Mapping) or not isinstance(row.get("request_id"), str):
                raise TargetResolutionError(f"invalid response row {line_number}")
            request_id = row["request_id"]
            if request_id in seen:
                raise TargetResolutionError(f"duplicate response: {request_id}")
            seen.add(request_id)
            if request_id not in jobs:
                raise TargetResolutionError(f"response outside selected jobs: {request_id}")
            job = jobs[request_id]
            if row.get("request_sha256") != _request_sha256(job["request"]):
                raise TargetResolutionError(f"request hash mismatch: {request_id}")
            status = str(row.get("status"))
            status_counts[status] += 1
            raw = row.get("raw_response")
            raw = raw if isinstance(raw, str) else ""
            is_fenced = raw.lstrip().startswith("```")
            has_tab = "\t" in raw
            fenced += int(is_fenced)
            literal_tab += int(has_tab)
            valid_fenced += int(status == "valid" and is_fenced)
            valid_literal_tab += int(status == "valid" and has_tab)
            explicit_parse_policy_rows += int("parse_policy" in row)
            if status != "valid":
                continue
            parsed = _validate_scalar(row.get("parsed_response"))
            measurement_counts[parsed["measurement_status"]] += 1
            if parsed["measurement_status"] != "scored":
                continue
            metadata = job["audit_metadata"]
            predictions[metadata["cell_id"]][metadata["pass_id"]][
                metadata["item_key"]
            ] = parsed["score"]
    if seen != set(jobs):
        raise TargetResolutionError(
            f"response coverage mismatch: missing={len(set(jobs) - seen)}"
        )
    return predictions, {
        "response_rows": len(seen),
        "status_counts": dict(sorted(status_counts.items())),
        "valid_measurement_status_counts": dict(sorted(measurement_counts.items())),
        "raw_markdown_fenced_rows": fenced,
        "valid_markdown_fenced_rows": valid_fenced,
        "raw_literal_tab_rows": literal_tab,
        "valid_literal_tab_rows": valid_literal_tab,
        "rows_with_explicit_parse_policy": explicit_parse_policy_rows,
    }


def _execution_by_aspect(execution: Mapping[str, object]) -> dict[str, dict]:
    programs = execution.get("programs")
    if not isinstance(programs, list):
        raise TargetResolutionError("execution has no program list")
    result = {}
    for program in programs:
        if not isinstance(program, dict) or not isinstance(program.get("aspect_id"), str):
            raise TargetResolutionError("invalid execution program")
        if program["aspect_id"] in result:
            raise TargetResolutionError(
                f"duplicate execution program: {program['aspect_id']}"
            )
        result[program["aspect_id"]] = program
    return result


def _target_rows(program: Mapping[str, object]) -> tuple[dict[str, float], Counter[str]]:
    rows = program.get("rows")
    if not isinstance(rows, list):
        raise TargetResolutionError("execution program has no rows")
    targets: dict[str, float] = {}
    statuses: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("item_key"), str):
            raise TargetResolutionError("invalid execution row")
        status = str(row.get("status"))
        statuses[status] += 1
        if row["item_key"] in targets:
            raise TargetResolutionError(
                f"duplicate scored target item: {row['item_key']}"
            )
        score = row.get("score")
        if status != "scored":
            continue
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise TargetResolutionError("invalid scored target")
        targets[row["item_key"]] = float(score)
    return targets, statuses


def _metric_name(program: Mapping[str, object], cell_id: str) -> str:
    relations = program.get("relations")
    if not isinstance(relations, list):
        raise TargetResolutionError("execution program has no relations")
    names = [
        row.get("metric_name")
        for row in relations
        if isinstance(row, Mapping) and row.get("cell_id") == cell_id
    ]
    if len(names) != 1 or not isinstance(names[0], str):
        raise TargetResolutionError(f"missing metric relation for {cell_id}")
    return names[0]


def _analyze_source(
    *,
    execution: Mapping[str, object],
    jobs_path: Path,
    responses_path: Path,
    channel: str,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    jobs, cells = _load_jobs(jobs_path, channel)
    predictions, accounting = _load_predictions(responses_path, jobs)
    programs = _execution_by_aspect(execution)
    per_cell: list[dict] = []
    bootstrap_by_cell: dict[str, np.ndarray] = {}
    for cell_id in sorted(cells):
        metadata = cells[cell_id]
        aspect_id = metadata["aspect_id"]
        program = programs.get(aspect_id)
        if not isinstance(program, Mapping):
            raise TargetResolutionError(f"missing execution program: {aspect_id}")
        if (
            program.get("source_sha256") != metadata["source_sha256"]
            or program.get("source_path") != metadata["source_path"]
        ):
            raise TargetResolutionError(f"source binding mismatch: {cell_id}")
        targets, target_statuses = _target_rows(program)
        partition = tie_expanded_terciles(targets)
        pass1 = predictions.get(cell_id, {1: {}, 2: {}})[1]
        pass2 = predictions.get(cell_id, {1: {}, 2: {}})[2]
        bottom_items = [
            item for item in partition.bottom if item in pass1 and item in pass2
        ]
        top_items = [item for item in partition.top if item in pass1 and item in pass2]
        bottom_predictions = [(pass1[item] + pass2[item]) / 2.0 for item in bottom_items]
        top_predictions = [(pass1[item] + pass2[item]) / 2.0 for item in top_items]
        auc = (
            mann_whitney_auc(bottom_predictions, top_predictions)
            if partition.has_spread
            else None
        )
        samples = (
            bootstrap_auc(
                bottom_predictions,
                top_predictions,
                draws=bootstrap_draws,
                seed=_cell_seed(bootstrap_seed, cell_id),
            )
            if auc is not None
            else np.asarray([], dtype=float)
        )
        bootstrap_by_cell[cell_id] = samples
        counts = Counter(targets.values())
        mode_count = max(counts.values())
        per_cell.append(
            {
                "cell_id": cell_id,
                "aspect_id": aspect_id,
                "level": metadata["level"],
                "metric_name": _metric_name(program, cell_id),
                "source_path": metadata["source_path"],
                "source_sha256": metadata["source_sha256"],
                "target": {
                    "panel_n": EXPECTED_ITEMS,
                    "scored_n": len(targets),
                    "status_counts": dict(sorted(target_statuses.items())),
                    "unique_value_count": len(counts),
                    "mode_count": mode_count,
                    "mode_fraction": mode_count / len(targets),
                    "minimum": min(targets.values()),
                    "maximum": max(targets.values()),
                },
                "terciles": {
                    "construction": (
                        "stable (target,item_key) nominal thirds; expand each extreme "
                        "to include all target ties at its cut point"
                    ),
                    "nominal_sizes": {
                        "bottom": partition.bottom_nominal_n,
                        "middle": len(partition.middle_nominal),
                        "top": partition.top_nominal_n,
                    },
                    "tie_expanded_sizes": {
                        "bottom": len(partition.bottom),
                        "top": len(partition.top),
                    },
                    "bottom_boundary": partition.bottom_boundary,
                    "top_boundary": partition.top_boundary,
                    "target_spread": partition.has_spread,
                    "extreme_overlap_n": len(set(partition.bottom) & set(partition.top)),
                },
                "prediction_support": {
                    "rule": "mean of passes 1 and 2 only where both are scored",
                    "bottom_n": len(bottom_predictions),
                    "top_n": len(top_predictions),
                },
                "tercile_auc": auc,
                "item_bootstrap_ci95": _interval(samples),
                "item_bootstrap_draws": len(samples),
            }
        )
    return per_cell, accounting, bootstrap_by_cell


def _headline_checks(per_cell: Sequence[Mapping[str, object]]) -> dict:
    expected = {"a0": 0.720, "a37": 0.711, "a92": 0.710}
    rows = {}
    for aspect_id, claimed in expected.items():
        matches = [
            {
                "cell_id": row["cell_id"],
                "level": row["level"],
                "observed_auc": row["tercile_auc"],
                "rounded_3dp": (
                    round(float(row["tercile_auc"]), 3)
                    if row["tercile_auc"] is not None
                    else None
                ),
            }
            for row in per_cell
            if row["aspect_id"] == aspect_id
        ]
        rows[aspect_id] = {
            "claimed_approximate_auc": claimed,
            "mappings": matches,
            "any_mapping_reproduces_to_3dp": any(
                match["rounded_3dp"] == claimed for match in matches
            ),
        }
    return rows


def analyze(
    *,
    execution_path: Path = DEFAULT_EXECUTION,
    jobs_path: Path = DEFAULT_JOBS,
    responses_path: Path = DEFAULT_RESPONSES,
    source_readout_path: Path = DEFAULT_SOURCE_READOUT,
    recovered_jobs_path: Path = DEFAULT_RECOVERED_JOBS,
    recovered_responses_path: Path = DEFAULT_RECOVERED_RESPONSES,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260713,
) -> dict:
    """Build the frozen target-resolution readout without remote calls."""

    execution = _load_object(execution_path)
    source_readout = _load_object(source_readout_path)
    per_cell, accounting, bootstrap = _analyze_source(
        execution=execution,
        jobs_path=jobs_path,
        responses_path=responses_path,
        channel=CHANNEL,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    spread_rows = [
        row
        for row in per_cell
        if row["terciles"]["target_spread"] and row["tercile_auc"] is not None
    ]
    aucs = [float(row["tercile_auc"]) for row in spread_rows]
    median_auc = float(statistics.median(aucs))
    aggregate_samples = np.asarray(
        [
            statistics.median(
                float(bootstrap[row["cell_id"]][draw]) for row in spread_rows
            )
            for draw in range(bootstrap_draws)
        ],
        dtype=float,
    )

    # The recovered implementation-summary arm is checked under the identical
    # estimator only to adjudicate source provenance.  It is not pooled into the
    # target-resolution result and receives no bootstrap computation.
    recovered_cells, recovered_accounting, _ = _analyze_source(
        execution=execution,
        jobs_path=recovered_jobs_path,
        responses_path=recovered_responses_path,
        channel=COMPARISON_CHANNEL,
        bootstrap_draws=1,
        bootstrap_seed=bootstrap_seed,
    )
    recovered_headlines = _headline_checks(recovered_cells)

    claimed_median = 0.547
    current_runner = ROOT / "methods/metric_seam/run_hierarchy_prompt_jobs.py"
    return {
        "schema": SCHEMA,
        "status": "cpu_only_reanalysis_complete",
        "estimand": (
            "For each mapping with separated code-target extreme terciles, the "
            "Mann-Whitney probability that the mean two-pass GLM-5.2 prediction "
            "on a target-top item exceeds that on a target-bottom item."
        ),
        "scope": {
            "task": "code-review",
            "prediction_channel": CHANNEL,
            "relation_mappings": len(per_cell),
            "items_per_panel": EXPECTED_ITEMS,
            "model_calls_made": 0,
            "gpu_or_accelerator_used": False,
        },
        "sources": {
            "execution": {"path": _display_path(execution_path), "sha256": _sha256_file(execution_path)},
            "jobs": {"path": _display_path(jobs_path), "sha256": _sha256_file(jobs_path)},
            "responses": {"path": _display_path(responses_path), "sha256": _sha256_file(responses_path)},
            "source_readout": {
                "path": _display_path(source_readout_path),
                "sha256": _sha256_file(source_readout_path),
                "recorded_channel": source_readout.get("scope", {}).get("channel"),
            },
        },
        "execution_accounting": accounting,
        "parser_provenance": {
            "stored_parsed_response_used": True,
            "raw_response_reparsed": False,
            "posthoc_recovery_applied_to_primary_source": False,
            "response_rows_self_declare_parse_policy": (
                accounting["rows_with_explicit_parse_policy"] == accounting["response_rows"]
            ),
            "observed_transport_evidence": {
                "valid_markdown_fenced_rows": accounting["valid_markdown_fenced_rows"],
                "valid_literal_tab_rows": accounting["valid_literal_tab_rows"],
            },
            "current_runner": {
                "path": _display_path(current_runner),
                "sha256": _sha256_file(current_runner),
                "observed_policy": "sole Markdown-fence unwrap plus json.loads(strict=False)",
            },
            "limitation": (
                "Response rows do not bind a parser implementation hash or explicit "
                "parse_policy. Fenced and literal-tab rows marked valid prove tolerant "
                "deserialization occurred, but the current runner hash is contextual "
                "provenance rather than a contemporaneously sealed execution binding."
            ),
        },
        "per_cell": per_cell,
        "aggregate": {
            "target_spread_mappings": len(spread_rows),
            "target_no_spread_mappings": len(per_cell) - len(spread_rows),
            "median_tercile_auc": median_auc,
            "descriptive_item_bootstrap_ci95": _interval(aggregate_samples),
            "auc_below_0_5_count": sum(value < 0.5 for value in aucs),
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
        },
        "headline_reproduction": {
            "per_aspect": _headline_checks(per_cell),
            "claimed_eight_mapping_median": claimed_median,
            "observed_eight_mapping_median": median_auc,
            "absolute_median_discrepancy": abs(median_auc - claimed_median),
            "median_reproduces_to_3dp": round(median_auc, 3) == claimed_median,
            "disposition": (
                "The named a0/a37/a92 values reproduce from the full-contract "
                "source under one declared tie-preserving estimator. The claimed "
                "eight-mapping median does not reproduce from that same source and "
                "is not substituted into this artifact."
            ),
        },
        "source_adjudication": {
            "selected": (
                "full_executable_contract responses: this is the only inspected "
                "frozen response source that reproduces all three named values"
            ),
            "recovered_implementation_summary": {
                "jobs": {
                    "path": _display_path(recovered_jobs_path),
                    "sha256": _sha256_file(recovered_jobs_path),
                },
                "responses": {
                    "path": _display_path(recovered_responses_path),
                    "sha256": _sha256_file(recovered_responses_path),
                },
                "accounting": recovered_accounting,
                "headline_checks_under_identical_estimator": recovered_headlines,
                "used_in_primary_result": False,
            },
        },
        "claim_limits": [
            "Tercile AUC measures reconstruction of a frozen code target; it is not whole-metric codability.",
            "A non-spreading target is reported as unresolved rather than assigned an AUC.",
            "The aggregate interval resamples items within the eight observed mappings; it is descriptive and does not generalize over a population of metrics.",
            "The full-contract prompt asks a model to simulate code. Its AUC is a target-resolution diagnostic, not an articulability upper bound or external correctness result.",
            "R1/R2/R3 are descriptive labels only.",
        ],
    }


def _fmt(value: object, digits: int = 3) -> str:
    return "undefined" if value is None else f"{float(value):.{digits}f}"


def render_report(readout: Mapping[str, object]) -> str:
    aggregate = readout["aggregate"]
    reproduction = readout["headline_reproduction"]
    rows = readout["per_cell"]
    lines = [
        "# Code-review target resolution",
        "",
        (
            f"**{aggregate['target_spread_mappings']} of 18 mappings had separated "
            f"target terciles. Their median AUC was "
            f"{_fmt(aggregate['median_tercile_auc'])}; the previously quoted 0.547 "
            f"median does not reproduce (difference "
            f"{_fmt(reproduction['absolute_median_discrepancy'])}).**"
        ),
        "",
        "The named values do reproduce: a0 = 0.720, a37 = 0.711, and the R3 a92 mapping = 0.710.",
        "",
        "| level | aspect | scored n | unique targets | mode fraction | bottom/top target n | bottom/top prediction n | AUC | 95% item-bootstrap CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda value: (value["level"], value["aspect_id"], value["cell_id"])):
        target = row["target"]
        terciles = row["terciles"]
        support = row["prediction_support"]
        ci = row["item_bootstrap_ci95"]
        lines.append(
            f"| {row['level']} | {row['aspect_id']} | {target['scored_n']} | "
            f"{target['unique_value_count']} | {target['mode_fraction']:.3f} | "
            f"{terciles['tie_expanded_sizes']['bottom']}/{terciles['tie_expanded_sizes']['top']} | "
            f"{support['bottom_n']}/{support['top_n']} | {_fmt(row['tercile_auc'])} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] |"
        )
    lines.extend(["", "## Parser provenance", ""])
    parser = readout["parser_provenance"]
    lines.append(
        f"The analysis used stored parsed responses and did not reparse raw text. "
        f"The primary source contains "
        f"{parser['observed_transport_evidence']['valid_markdown_fenced_rows']:,} "
        f"valid fenced rows and "
        f"{parser['observed_transport_evidence']['valid_literal_tab_rows']:,} valid "
        "rows with literal tabs. Response rows do not self-declare or hash-bind their parser policy."
    )
    lines.extend(["", "## Claim limits", ""])
    lines.extend(f"- {value}" for value in readout["claim_limits"])
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--source-readout", type=Path, default=DEFAULT_SOURCE_READOUT)
    parser.add_argument("--recovered-jobs", type=Path, default=DEFAULT_RECOVERED_JOBS)
    parser.add_argument("--recovered-responses", type=Path, default=DEFAULT_RECOVERED_RESPONSES)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = args.output.parent / "report.md"
    if args.output.exists() or report.exists():
        raise FileExistsError(
            f"refusing to overwrite append-only analysis outputs: {args.output}, {report}"
        )
    readout = analyze(
        execution_path=args.execution,
        jobs_path=args.jobs,
        responses_path=args.responses,
        source_readout_path=args.source_readout,
        recovered_jobs_path=args.recovered_jobs,
        recovered_responses_path=args.recovered_responses,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(readout, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    with report.open("x", encoding="utf-8") as handle:
        handle.write(render_report(readout))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(report),
                "target_spread_mappings": readout["aggregate"]["target_spread_mappings"],
                "median_tercile_auc": readout["aggregate"]["median_tercile_auc"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
