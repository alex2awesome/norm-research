"""Analyze frozen code-review relation-local prompt reconstruction responses."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from methods.metric_seam.hierarchy_prompt_batch import validate_prompt_response
from methods.metric_seam.run_hierarchy_prompt_jobs import (
    iter_selected_jobs,
    preflight_jobs,
    request_sha256,
)


SCHEMA = "metric-seam.code-review-reconstruction-readout.v1"
CHANNEL = "implementation_disclosed"
EXPECTED_RELATION_MAPPINGS = 18
EXPECTED_VECTOR_CLUSTERS = 10
EXPECTED_ITEMS = 125
EXPECTED_RESPONSES = EXPECTED_RELATION_MAPPINGS * EXPECTED_ITEMS * 2
MIN_EXPLORATORY_SUPPORT = 10
MIN_CONFIRMATORY_SUPPORT = 30


class ReconstructionAnalysisError(ValueError):
    """Raised when frozen artifacts cannot support the registered readout."""


@dataclass(frozen=True)
class MappingSeries:
    cell_id: str
    aspect_id: str
    vector_cluster_id: str
    level: str
    metric_name: str
    item_keys: tuple[str, ...]
    code_scores: Mapping[str, float]
    pass1_scores: Mapping[str, float]
    pass2_scores: Mapping[str, float]


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Raw signed Spearman correlation with average ranks and explicit constants."""

    if len(x) != len(y) or len(x) < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start
            while (
                end + 1 < len(order)
                and values[order[end + 1]] == values[order[start]]
            ):
                end += 1
            average_rank = (start + end) / 2.0 + 1.0
            for position in range(start, end + 1):
                result[order[position]] = average_rank
            start = end + 1
        return result

    rank_x = ranks(x)
    rank_y = ranks(y)
    mean_x = statistics.fmean(rank_x)
    mean_y = statistics.fmean(rank_y)
    centered_x = [value - mean_x for value in rank_x]
    centered_y = [value - mean_y for value in rank_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def _support_tier(n: int) -> str:
    if n >= MIN_CONFIRMATORY_SUPPORT:
        return "confirmatory_estimate"
    if n >= MIN_EXPLORATORY_SUPPORT:
        return "exploratory_estimate"
    return "no_correlation_estimate"


def _common_support(series: MappingSeries) -> list[str]:
    return [
        item_key
        for item_key in series.item_keys
        if item_key in series.code_scores
        and item_key in series.pass1_scores
        and item_key in series.pass2_scores
    ]


def _rho_on_items(
    series: MappingSeries,
    item_keys: Sequence[str],
    *,
    prompt: str = "mean",
) -> float | None:
    code = [series.code_scores[item_key] for item_key in item_keys]
    if prompt == "pass1":
        scores = [series.pass1_scores[item_key] for item_key in item_keys]
    elif prompt == "pass2":
        scores = [series.pass2_scores[item_key] for item_key in item_keys]
    elif prompt == "mean":
        scores = [
            (series.pass1_scores[item_key] + series.pass2_scores[item_key]) / 2.0
            for item_key in item_keys
        ]
    else:
        raise ReconstructionAnalysisError(f"unknown prompt projection: {prompt}")
    return spearman(code, scores)


def mapping_statistics(series: MappingSeries) -> dict:
    support = _common_support(series)
    n = len(support)
    tier = _support_tier(n)
    estimate_allowed = n >= MIN_EXPLORATORY_SUPPORT
    raw_rho = _rho_on_items(series, support) if estimate_allowed else None
    pass1_rho = (
        _rho_on_items(series, support, prompt="pass1") if estimate_allowed else None
    )
    pass2_rho = (
        _rho_on_items(series, support, prompt="pass2") if estimate_allowed else None
    )
    pass_reliability = (
        spearman(
            [series.pass1_scores[item_key] for item_key in support],
            [series.pass2_scores[item_key] for item_key in support],
        )
        if estimate_allowed
        else None
    )
    two_pass_reliability = None
    ceiling = None
    if pass_reliability is not None and pass_reliability > 0.0:
        two_pass_reliability = 2.0 * pass_reliability / (1.0 + pass_reliability)
        ceiling = math.sqrt(two_pass_reliability)
    normalized = (
        raw_rho / ceiling
        if raw_rho is not None and ceiling is not None and ceiling > 0.0
        else None
    )
    code_values = [series.code_scores[item_key] for item_key in support]
    tie_fraction = (
        max(Counter(code_values).values()) / n if code_values else None
    )
    denominator = len(series.item_keys)
    return {
        "cell_id": series.cell_id,
        "aspect_id": series.aspect_id,
        "vector_cluster_id": series.vector_cluster_id,
        "level": series.level,
        "metric_name": series.metric_name,
        "support_interpretation": tier,
        "common_support_n": n,
        "raw_rho": raw_rho,
        "pass1_rho": pass1_rho,
        "pass2_rho": pass2_rho,
        "pass_to_pass_reliability": pass_reliability,
        "two_pass_spearman_brown_reliability": two_pass_reliability,
        "attenuation_ceiling": ceiling,
        "ceiling_normalized_rho": normalized,
        "code_unique_score_count": len(set(code_values)),
        "largest_code_tie_fraction": tie_fraction,
        "code_scored_count": len(series.code_scores),
        "code_scored_coverage": len(series.code_scores) / denominator,
        "prompt_pass1_scored_count": len(series.pass1_scores),
        "prompt_pass1_scored_coverage": len(series.pass1_scores) / denominator,
        "prompt_pass2_scored_count": len(series.pass2_scores),
        "prompt_pass2_scored_coverage": len(series.pass2_scores) / denominator,
        "prompt_both_passes_scored_count": len(
            set(series.pass1_scores) & set(series.pass2_scores)
        ),
        "prompt_both_passes_scored_coverage": len(
            set(series.pass1_scores) & set(series.pass2_scores)
        )
        / denominator,
    }


def wrong_relation_statistics(
    correct: MappingSeries,
    wrong_prompt: MappingSeries,
) -> dict:
    support = [
        item_key
        for item_key in correct.item_keys
        if item_key in correct.code_scores
        and item_key in correct.pass1_scores
        and item_key in correct.pass2_scores
        and item_key in wrong_prompt.pass1_scores
        and item_key in wrong_prompt.pass2_scores
    ]
    tier = _support_tier(len(support))
    rho_correct = None
    rho_wrong = None
    if len(support) >= MIN_EXPLORATORY_SUPPORT:
        code = [correct.code_scores[item_key] for item_key in support]
        correct_mean = [
            (correct.pass1_scores[item_key] + correct.pass2_scores[item_key]) / 2.0
            for item_key in support
        ]
        wrong_mean = [
            (
                wrong_prompt.pass1_scores[item_key]
                + wrong_prompt.pass2_scores[item_key]
            )
            / 2.0
            for item_key in support
        ]
        rho_correct = spearman(code, correct_mean)
        rho_wrong = spearman(code, wrong_mean)
    return {
        "cell_id": correct.cell_id,
        "aspect_id": correct.aspect_id,
        "vector_cluster_id": correct.vector_cluster_id,
        "level": correct.level,
        "control_prompt_cell_id": wrong_prompt.cell_id,
        "control_prompt_aspect_id": wrong_prompt.aspect_id,
        "support_interpretation": tier,
        "identical_common_support_n": len(support),
        "rho_correct": rho_correct,
        "rho_wrong": rho_wrong,
        "rho_correct_minus_wrong": (
            rho_correct - rho_wrong
            if rho_correct is not None and rho_wrong is not None
            else None
        ),
        # Kept in the readout to make the identical-support assertion auditable.
        "identical_common_support_item_keys": support,
    }


def _bootstrap_rho(
    series: MappingSeries,
    sampled_items: Sequence[str],
) -> float | None:
    support = set(_common_support(series))
    retained = [item_key for item_key in sampled_items if item_key in support]
    return _rho_on_items(series, retained) if len(retained) >= 2 else None


def _bootstrap_wrong_delta(
    correct: MappingSeries,
    wrong: MappingSeries,
    sampled_items: Sequence[str],
) -> float | None:
    support = {
        item_key
        for item_key in correct.item_keys
        if item_key in correct.code_scores
        and item_key in correct.pass1_scores
        and item_key in correct.pass2_scores
        and item_key in wrong.pass1_scores
        and item_key in wrong.pass2_scores
    }
    retained = [item_key for item_key in sampled_items if item_key in support]
    if len(retained) < 2:
        return None
    code = [correct.code_scores[item_key] for item_key in retained]
    correct_mean = [
        (correct.pass1_scores[item_key] + correct.pass2_scores[item_key]) / 2.0
        for item_key in retained
    ]
    wrong_mean = [
        (wrong.pass1_scores[item_key] + wrong.pass2_scores[item_key]) / 2.0
        for item_key in retained
    ]
    rho_correct = spearman(code, correct_mean)
    rho_wrong = spearman(code, wrong_mean)
    if rho_correct is None or rho_wrong is None:
        return None
    return rho_correct - rho_wrong


def hierarchical_bootstrap(
    *,
    series_by_cell: Mapping[str, MappingSeries],
    cluster_to_cells: Mapping[str, Sequence[str]],
    item_keys: Sequence[str],
    draws: int,
    seed: int,
    wrong_prompt_cells: Mapping[str, str] | None = None,
) -> list[float]:
    """Cluster mappings by vector and share one item resample in every draw."""

    if draws < 1:
        raise ReconstructionAnalysisError("bootstrap draws must be positive")
    cluster_ids = sorted(cluster_to_cells)
    if not cluster_ids or not item_keys:
        raise ReconstructionAnalysisError("bootstrap requires clusters and item keys")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    original_eligible = {
        cell_id
        for cell_id, series in series_by_cell.items()
        if len(_common_support(series)) >= MIN_EXPLORATORY_SUPPORT
    }
    if wrong_prompt_cells is not None:
        original_eligible = {
            cell_id
            for cell_id in original_eligible
            if wrong_relation_statistics(
                series_by_cell[cell_id], series_by_cell[wrong_prompt_cells[cell_id]]
            )["rho_correct_minus_wrong"]
            is not None
        }

    for _ in range(draws):
        drawn_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        drawn_items = rng.choice(item_keys, size=len(item_keys), replace=True).tolist()
        mapping_values: list[float] = []
        for cluster_id in drawn_clusters:
            for cell_id in cluster_to_cells[str(cluster_id)]:
                if cell_id not in original_eligible:
                    continue
                series = series_by_cell[cell_id]
                if wrong_prompt_cells is None:
                    value = _bootstrap_rho(series, drawn_items)
                else:
                    value = _bootstrap_wrong_delta(
                        series,
                        series_by_cell[wrong_prompt_cells[cell_id]],
                        drawn_items,
                    )
                if value is not None:
                    mapping_values.append(value)
        if mapping_values:
            estimates.append(float(statistics.median(mapping_values)))
    if not estimates:
        raise ReconstructionAnalysisError("no defined bootstrap estimates")
    return estimates


def _percentile_interval(values: Sequence[float]) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconstructionAnalysisError(f"expected an object in {path}")
    return value


def resolve_analysis_channel(jobs_path: Path, requested: str | None = None) -> str:
    """Resolve one analysis channel without binding the analyzer to the v3 arm.

    The original three-channel v3 bundle keeps its historical default of
    ``implementation_disclosed``.  Additive single-channel bundles such as the
    full-executable-contract ceiling infer their sole channel automatically.
    """

    channels: set[str] = set()
    with gzip.open(jobs_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReconstructionAnalysisError(
                    f"invalid job JSON at {jobs_path}:{line_number}"
                ) from exc
            metadata = row.get("audit_metadata") if isinstance(row, Mapping) else None
            channel = metadata.get("channel") if isinstance(metadata, Mapping) else None
            if not isinstance(channel, str) or not channel:
                raise ReconstructionAnalysisError(
                    f"job at {jobs_path}:{line_number} has no channel"
                )
            channels.add(channel)
    if requested is not None:
        if requested not in channels:
            raise ReconstructionAnalysisError(
                f"requested channel {requested!r} is absent; found {sorted(channels)}"
            )
        return requested
    if len(channels) == 1:
        return next(iter(channels))
    if CHANNEL in channels:
        return CHANNEL
    raise ReconstructionAnalysisError(
        f"multiple job channels require --channel; found {sorted(channels)}"
    )


def _load_jobs(
    jobs_path: Path, channel: str
) -> tuple[dict[str, dict], dict[str, dict]]:
    preflight_jobs(jobs_path, channel=channel, expected_jobs=EXPECTED_RESPONSES)
    by_request: dict[str, dict] = {}
    by_slot: dict[tuple[str, str, int], dict] = {}
    cell_metadata: dict[str, dict] = {}
    for row in iter_selected_jobs(jobs_path, channel):
        metadata = row["audit_metadata"]
        request_id = row["request_id"]
        pass_id = metadata.get("pass_id", metadata.get("pass_index"))
        slot = (metadata["cell_id"], metadata["item_key"], pass_id)
        if slot in by_slot:
            raise ReconstructionAnalysisError(f"duplicate prompt job slot: {slot}")
        by_slot[slot] = row
        by_request[request_id] = row
        existing = cell_metadata.setdefault(
            metadata["cell_id"],
            {
                "aspect_id": metadata["aspect_id"],
                "level": metadata["level"],
                "item_keys": set(),
            },
        )
        if (
            existing["aspect_id"] != metadata["aspect_id"]
            or existing["level"] != metadata["level"]
        ):
            raise ReconstructionAnalysisError("inconsistent prompt cell metadata")
        existing["item_keys"].add(metadata["item_key"])
    return by_request, cell_metadata


def _load_responses(
    path: Path,
    jobs_by_request: Mapping[str, dict],
) -> tuple[dict[str, dict], dict]:
    responses: dict[str, dict] = {}
    status_counts: Counter[str] = Counter()
    measurement_counts: Counter[str] = Counter()
    returned_models: Counter[str] = Counter()
    total_attempts = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReconstructionAnalysisError(
                    f"invalid response JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict) or not isinstance(row.get("request_id"), str):
                raise ReconstructionAnalysisError(f"invalid response row {line_number}")
            request_id = row["request_id"]
            if request_id in responses:
                raise ReconstructionAnalysisError(f"duplicate response: {request_id}")
            responses[request_id] = row
            status_counts[str(row.get("status"))] += 1
            attempts = row.get("attempts")
            if isinstance(attempts, int):
                total_attempts += attempts
            model = row.get("returned_model")
            if isinstance(model, str) and model:
                returned_models[model] += 1
            job = jobs_by_request.get(request_id)
            if job is None:
                continue
            expected_sha = request_sha256(job["request"])
            if row.get("request_sha256") != expected_sha:
                raise ReconstructionAnalysisError(
                    f"request hash mismatch for response {request_id}"
                )
            if row.get("status") == "valid":
                parsed = validate_prompt_response(row.get("parsed_response"))
                measurement_counts[parsed["measurement_status"]] += 1
    requested = set(jobs_by_request)
    observed = set(responses)
    accounting = {
        "expected_requests": len(requested),
        "response_rows": len(responses),
        "missing_request_count": len(requested - observed),
        "unexpected_request_count": len(observed - requested),
        "status_counts": dict(sorted(status_counts.items())),
        "valid_measurement_status_counts": dict(sorted(measurement_counts.items())),
        "total_attempts": total_attempts,
        "returned_model_counts": dict(sorted(returned_models.items())),
    }
    return responses, accounting


def _prompt_scores_by_cell(
    jobs_by_request: Mapping[str, dict],
    responses: Mapping[str, dict],
) -> dict[str, dict[int, dict[str, float]]]:
    result: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: {1: {}, 2: {}}
    )
    for request_id, job in jobs_by_request.items():
        response = responses.get(request_id)
        if not response or response.get("status") != "valid":
            continue
        parsed = validate_prompt_response(response.get("parsed_response"))
        if parsed["measurement_status"] != "scored":
            continue
        metadata = job["audit_metadata"]
        pass_id = metadata.get("pass_id", metadata.get("pass_index"))
        result[metadata["cell_id"]][pass_id][metadata["item_key"]] = parsed["score"]
    return result


def _execution_programs(code_execution: Mapping[str, object]) -> dict[str, dict]:
    programs = code_execution.get("programs")
    if not isinstance(programs, list):
        raise ReconstructionAnalysisError("code execution has no programs")
    by_aspect = {}
    for program in programs:
        if not isinstance(program, dict) or not isinstance(program.get("aspect_id"), str):
            raise ReconstructionAnalysisError("invalid code execution program")
        by_aspect[program["aspect_id"]] = program
    return by_aspect


def _build_series(
    *,
    manifest: Mapping[str, object],
    cell_metadata: Mapping[str, dict],
    prompt_scores: Mapping[str, dict[int, dict[str, float]]],
    code_execution: Mapping[str, object],
) -> tuple[dict[str, MappingSeries], dict[str, list[str]], dict[str, str]]:
    clustered = manifest.get("clustered_inference")
    if not isinstance(clustered, Mapping):
        raise ReconstructionAnalysisError("manifest lacks clustered inference")
    clusters = clustered.get("vector_clusters")
    if not isinstance(clusters, list) or len(clusters) != EXPECTED_VECTOR_CLUSTERS:
        raise ReconstructionAnalysisError("manifest does not contain 10 vector clusters")
    programs = _execution_programs(code_execution)
    cluster_to_cells: dict[str, list[str]] = {}
    cluster_by_cell: dict[str, dict] = {}
    for cluster in clusters:
        cluster_id = cluster["vector_cluster_id"]
        cells = list(cluster["cell_ids"])
        cluster_to_cells[cluster_id] = cells
        for cell_id in cells:
            if cell_id in cluster_by_cell:
                raise ReconstructionAnalysisError("cell appears in multiple vector clusters")
            cluster_by_cell[cell_id] = cluster
    if set(cluster_by_cell) != set(cell_metadata):
        raise ReconstructionAnalysisError("manifest vector clusters do not match prompt cells")

    relations_by_cell: dict[str, dict] = {}
    for program in programs.values():
        for relation in program.get("relations", []):
            if relation["cell_id"] in cell_metadata:
                relations_by_cell[relation["cell_id"]] = relation

    series_by_cell: dict[str, MappingSeries] = {}
    for cell_id, metadata in cell_metadata.items():
        cluster = cluster_by_cell[cell_id]
        aspect_id = metadata["aspect_id"]
        if cluster["aspect_id"] != aspect_id:
            raise ReconstructionAnalysisError(f"aspect mismatch for {cell_id}")
        program = programs.get(aspect_id)
        if program is None or program.get("source_sha256") != cluster["source_sha256"]:
            raise ReconstructionAnalysisError(f"code vector identity mismatch for {cell_id}")
        rows = program.get("rows")
        if not isinstance(rows, list):
            raise ReconstructionAnalysisError(f"code program {aspect_id} has no rows")
        code_scores = {
            row["item_key"]: float(row["score"])
            for row in rows
            if row.get("status") == "scored"
            and isinstance(row.get("score"), (int, float))
            and not isinstance(row.get("score"), bool)
            and math.isfinite(float(row["score"]))
        }
        relation = relations_by_cell.get(cell_id)
        if relation is None:
            raise ReconstructionAnalysisError(f"missing execution relation for {cell_id}")
        cell_prompt = prompt_scores.get(cell_id, {1: {}, 2: {}})
        series_by_cell[cell_id] = MappingSeries(
            cell_id=cell_id,
            aspect_id=aspect_id,
            vector_cluster_id=cluster["vector_cluster_id"],
            level=metadata["level"],
            metric_name=relation["metric_name"],
            item_keys=tuple(sorted(metadata["item_keys"])),
            code_scores=code_scores,
            pass1_scores=cell_prompt[1],
            pass2_scores=cell_prompt[2],
        )

    prereg = manifest.get("analysis_preregistration")
    wrong_contract = prereg.get("wrong_relation_control") if isinstance(prereg, Mapping) else None
    wrong_rows = wrong_contract.get("rows") if isinstance(wrong_contract, Mapping) else None
    if not isinstance(wrong_rows, list) or len(wrong_rows) != EXPECTED_RELATION_MAPPINGS:
        raise ReconstructionAnalysisError("manifest lacks frozen wrong-relation rows")
    wrong_prompt_cells = {row["cell_id"]: row["control_prompt_cell_id"] for row in wrong_rows}
    if set(wrong_prompt_cells) != set(series_by_cell):
        raise ReconstructionAnalysisError("wrong-relation assignment does not cover all cells")
    return series_by_cell, cluster_to_cells, wrong_prompt_cells


def analyze(
    *,
    prompt_manifest_path: Path,
    prompt_jobs_path: Path,
    responses_path: Path,
    code_execution_path: Path,
    bootstrap_draws: int,
    bootstrap_seed: int,
    channel: str | None = None,
) -> dict:
    manifest = _load_json(prompt_manifest_path)
    code_execution = _load_json(code_execution_path)
    if manifest.get("n_cells") != EXPECTED_RELATION_MAPPINGS:
        raise ReconstructionAnalysisError("prompt manifest is not the frozen 18-cell scope")
    if manifest.get("n_unique_program_vectors") != EXPECTED_VECTOR_CLUSTERS:
        raise ReconstructionAnalysisError("prompt manifest is not the frozen 10-vector scope")
    if manifest.get("n_items_per_cell") != EXPECTED_ITEMS:
        raise ReconstructionAnalysisError("prompt manifest is not the frozen 125-item scope")

    analysis_channel = resolve_analysis_channel(prompt_jobs_path, channel)
    jobs_by_request, cell_metadata = _load_jobs(prompt_jobs_path, analysis_channel)
    responses, accounting = _load_responses(responses_path, jobs_by_request)
    prompt_scores = _prompt_scores_by_cell(jobs_by_request, responses)
    series_by_cell, cluster_to_cells, wrong_prompt_cells = _build_series(
        manifest=manifest,
        cell_metadata=cell_metadata,
        prompt_scores=prompt_scores,
        code_execution=code_execution,
    )
    per_mapping = [
        mapping_statistics(series_by_cell[cell_id])
        for cell_id in sorted(series_by_cell)
    ]
    raw_values = [row["raw_rho"] for row in per_mapping if row["raw_rho"] is not None]
    normalized_values = [
        row["ceiling_normalized_rho"]
        for row in per_mapping
        if row["ceiling_normalized_rho"] is not None
    ]
    ceilings = [
        row["attenuation_ceiling"]
        for row in per_mapping
        if row["attenuation_ceiling"] is not None
    ]
    item_keys = next(iter(series_by_cell.values())).item_keys
    raw_bootstrap = (
        hierarchical_bootstrap(
            series_by_cell=series_by_cell,
            cluster_to_cells=cluster_to_cells,
            item_keys=item_keys,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        if raw_values
        else []
    )

    wrong_per_mapping = [
        wrong_relation_statistics(
            series_by_cell[cell_id], series_by_cell[wrong_prompt_cells[cell_id]]
        )
        for cell_id in sorted(series_by_cell)
    ]
    wrong_values = [
        row["rho_correct_minus_wrong"]
        for row in wrong_per_mapping
        if row["rho_correct_minus_wrong"] is not None
    ]
    wrong_bootstrap = (
        hierarchical_bootstrap(
            series_by_cell=series_by_cell,
            cluster_to_cells=cluster_to_cells,
            item_keys=item_keys,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
            wrong_prompt_cells=wrong_prompt_cells,
        )
        if wrong_values
        else []
    )

    return {
        "schema": SCHEMA,
        "estimand": (
            "Median within-mapping signed Spearman agreement between frozen code "
            "scores and mean two-pass GLM-5.2 prompt scores, conditional on an "
            "already-discovered executable relation-local witness."
        ),
        "scope": {
            "relation_mappings": EXPECTED_RELATION_MAPPINGS,
            "unique_program_vectors": EXPECTED_VECTOR_CLUSTERS,
            "items": EXPECTED_ITEMS,
            "model": "glm-5.2",
            "channel": analysis_channel,
        },
        "execution_accounting": accounting,
        "per_mapping": per_mapping,
        "aggregate": {
            "median_raw_rho": (
                float(statistics.median(raw_values)) if raw_values else None
            ),
            "ci95": (
                _percentile_interval(raw_bootstrap) if raw_bootstrap else [None, None]
            ),
            "bootstrap_draws_requested": bootstrap_draws,
            "bootstrap_draws_defined": len(raw_bootstrap),
            "bootstrap_seed": bootstrap_seed,
            "mappings_with_defined_raw_rho": len(raw_values),
            "mappings_with_confirmatory_support": sum(
                row["support_interpretation"] == "confirmatory_estimate"
                for row in per_mapping
            ),
            "median_reliability_ceiling": (
                float(statistics.median(ceilings)) if ceilings else None
            ),
            "median_ceiling_normalized_rho": (
                float(statistics.median(normalized_values))
                if normalized_values
                else None
            ),
        },
        "wrong_relation_control": {
            "support_rule": (
                "Each correct and assigned wrong correlation uses the identical "
                "intersection of code, both correct passes, and both wrong passes."
            ),
            "per_mapping": wrong_per_mapping,
            "mappings_with_defined_contrast": len(wrong_values),
            "median_rho_correct_minus_wrong": (
                float(statistics.median(wrong_values)) if wrong_values else None
            ),
            "ci95": (
                _percentile_interval(wrong_bootstrap)
                if wrong_bootstrap
                else [None, None]
            ),
            "bootstrap_draws_requested": bootstrap_draws,
            "bootstrap_draws_defined": len(wrong_bootstrap),
            "bootstrap_seed": bootstrap_seed,
        },
        "claim_limits": (
            [
                "This is conditional relation-local reconstruction after an executable witness was already discovered; it is not whole-metric codability.",
                "Full executable source disclosure is still model simulation, not literal program execution or external validation of the code.",
                "The ceiling result alone does not establish tacitness or external correctness; it determines whether lower-disclosure arms are interpretable.",
                "Code/prompt disagreement is not automatically code underperformance; the code may implement a different or stronger relation.",
                "R1/R2/R3 are descriptive labels only; no R-level trend is estimated.",
            ]
            if analysis_channel == "full_executable_contract"
            else [
                "This is conditional relation-local reconstruction after an executable witness was already discovered; it is not whole-metric codability.",
                "The implementation-disclosed channel is an incomplete relation summary, not a literal source-code or full executable-contract reconstruction task.",
                "The result does not establish tacitness or external correctness.",
                "Code/prompt disagreement is not automatically code underperformance; the code may implement a different or stronger relation.",
                "R1/R2/R3 are descriptive labels only; no R-level trend is estimated.",
            ]
        ),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def render_report(readout: Mapping[str, object]) -> str:
    accounting = readout["execution_accounting"]
    aggregate = readout["aggregate"]
    wrong = readout["wrong_relation_control"]
    per_mapping = readout["per_mapping"]
    valid = accounting["status_counts"].get("valid", 0)
    confirmatory = aggregate["mappings_with_confirmatory_support"]
    ci = aggregate["ci95"]
    wrong_ci = wrong["ci95"]
    scope = readout.get("scope")
    channel = (
        scope.get("channel", CHANNEL) if isinstance(scope, Mapping) else CHANNEL
    )
    if aggregate["median_raw_rho"] is None:
        interpretation = (
            "No reconstruction correlation can be estimated because no response "
            "satisfied the frozen strict JSON contract. This is a bounded "
            "GLM-5.2 output-contract failure for the incomplete implementation-summary "
            "channel; it is not an observed rho of zero or evidence of behavioral "
            "disagreement between code and prompt scores."
        )
    elif channel == "full_executable_contract":
        rho = aggregate["median_raw_rho"]
        if rho >= 0.70:
            interpretation = (
                "The pre-declared high-ceiling branch applies (median raw rho >= "
                ".70): GLM-5.2 can simulate the disclosed programs well enough for "
                "the lower-disclosure ladder to be readable. The recovered "
                "implementation-summary rho=.146 can therefore be interpreted as "
                "disclosure loss localized to withheld applicability, polarity, and "
                "aggregation, subject to the specificity-control and support limits."
            )
        elif rho < 0.40:
            interpretation = (
                "The pre-declared low-ceiling branch applies (median raw rho < .40): "
                "GLM-5.2 cannot reliably simulate these programs even with complete "
                "source disclosure. Lower-disclosure reconstruction is therefore "
                "not interpretable as tacitness; this is an executor/item-panel "
                "instrument limit."
            )
        else:
            interpretation = (
                "The pre-declared intermediate-ceiling branch applies (.40 <= median "
                "raw rho < .70). Report the disclosure ladder descriptively with "
                "uncertainty; it licenses no tacitness or instrument-limit verdict."
            )
    else:
        interpretation = (
            "The signed estimate measures how much relation-local code behavior "
            "GLM-5.2 reconstructs from the incomplete implementation summary. "
            "A positive result supports some prompt articulation of that local "
            "behavior; a low result is a bounded reconstruction failure for this "
            "model and channel."
        )
    lines = [
        (
            f"**{valid:,} responses were valid; {confirmatory} of 18 mappings had "
            f"confirmatory support. Median raw Spearman rho was "
            f"{_fmt(aggregate['median_raw_rho'])} (95% clustered bootstrap CI "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}]).**"
        ),
        "",
        (
            "# Code-review full-contract ceiling reconstruction"
            if channel == "full_executable_contract"
            else "# Code-review relation-local reconstruction"
        ),
        "",
        (
            f"The median two-pass reliability ceiling was "
            f"{_fmt(aggregate['median_reliability_ceiling'])}. The median "
            f"correct-minus-wrong-relation contrast was "
            f"{_fmt(wrong['median_rho_correct_minus_wrong'])} (95% clustered "
            f"bootstrap CI [{_fmt(wrong_ci[0])}, {_fmt(wrong_ci[1])}])."
        ),
        "",
        "## Mapping-level distribution",
        "",
        "| level | aspect | mapping | common n | support | raw rho | ceiling | rho/ceiling |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    sorted_rows = sorted(
        per_mapping,
        key=lambda row: (
            row["raw_rho"] is None,
            row["raw_rho"] if row["raw_rho"] is not None else math.inf,
        ),
    )
    for row in sorted_rows:
        metric = str(row["metric_name"]).replace("|", "\\|")
        lines.append(
            f"| {row['level']} | {row['aspect_id']} | {metric} | "
            f"{row['common_support_n']} | {row['support_interpretation']} | "
            f"{_fmt(row['raw_rho'])} | {_fmt(row['attenuation_ceiling'])} | "
            f"{_fmt(row['ceiling_normalized_rho'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "## Claim limits",
            "",
        ]
    )
    lines.extend(f"- {limit}" for limit in readout["claim_limits"])
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", required=True, type=Path)
    parser.add_argument("--prompt-jobs", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--code-execution", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument(
        "--channel",
        help=(
            "analysis channel; inferred for single-channel bundles and defaults to "
            "implementation_disclosed for the historical multi-channel v3 bundle"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    readout = analyze(
        prompt_manifest_path=args.prompt_manifest,
        prompt_jobs_path=args.prompt_jobs,
        responses_path=args.responses,
        code_execution_path=args.code_execution,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
        channel=args.channel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(readout, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path = args.output.parent / "report.md"
    report_path.write_text(render_report(readout), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(report_path),
                "valid_responses": readout["execution_accounting"]["status_counts"].get(
                    "valid", 0
                ),
                "median_raw_rho": readout["aggregate"]["median_raw_rho"],
                "ci95": readout["aggregate"]["ci95"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
