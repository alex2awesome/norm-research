"""Validate the frozen hierarchy-v1 panel without weakening the live v3 contract.

The metric-seam hierarchy artifacts were frozen against
``tacit_breadth_metric_panel/v1``.  The panel compiler has since advanced its live
validator to v3, whose additional task-global provenance fields cannot be added to a
frozen v1 artifact without changing the scientific sample.  Legacy metric-seam
consumers therefore dispatch v1 panels to the bounded validator below and all other
panels to the compiler's current, strict validator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.compile_fresh_name_arm_bank import (
    BREADTH_LEVELS,
    DEFAULT_BREADTH_BUCKETS,
    DEFAULT_BREADTH_TASKS,
    validate_metric_panel,
)
from methods.metric_implementer.experiments.mine_clusters import (
    hierarchy_leaf_support_ids,
)


FROZEN_V1_SCHEMA = "tacit_breadth_metric_panel/v1"
ROOT = Path(__file__).resolve().parents[2]

_LEGACY_CELL_FIELDS = {
    "id",
    "node_id",
    "metric_id",
    "task",
    "domain",
    "level",
    "bucket",
    "source_kind",
    "source_index",
    "immediate_source_ids",
    "immediate_source_sha256",
    "leaf_support_count",
    "leaf_support_sha256",
    "dependency_component_id",
    "dependency_component_size",
    "dependency_degree",
    "source_assignment_multiplicity_max",
    "provenance_component_id",
    "provenance_component_size",
    "provenance_overlap_degree",
    "provenance_assignment_multiplicity_max",
    "construct",
    "description",
    "children",
    "breadth_stratum",
    "selection_rank",
    "stratum_population_n",
    "stratum_selected_n",
    "inclusion_probability",
    "design_weight",
    "source_path",
    "source_sha256",
}


def _is_plain_int(value: object, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _canonical_sha256(payload: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_frozen_v1_panel(panel: Mapping) -> list[str]:
    """Return errors for the fields that define the frozen v1 sampling contract.

    This intentionally does not reconstruct v3 task-global provenance.  It does
    validate the v1 content binding, source files, identities, 30-per-level frame,
    level-local dependency metadata, and sampling-design fields used downstream.
    More specific design equalities remain the responsibility of the prevalence
    consumer so its fail-closed diagnostics retain their scientific meaning.
    """

    errors: list[str] = []
    if panel.get("schema") != FROZEN_V1_SCHEMA:
        return ["metric panel is not the frozen hierarchy v1 schema"]

    declared = panel.get("panel_content_sha256")
    core = {key: value for key, value in panel.items() if key != "panel_content_sha256"}
    if declared != _canonical_sha256(core):
        errors.append("metric panel content hash mismatch")

    tasks = panel.get("tasks")
    levels = panel.get("levels")
    if tasks != list(DEFAULT_BREADTH_TASKS):
        errors.append("frozen v1 panel changes the declared task identities or order")
    if levels != list(BREADTH_LEVELS):
        errors.append("frozen v1 panel changes the declared hierarchy levels or order")
    if panel.get("task_buckets") != DEFAULT_BREADTH_BUCKETS:
        errors.append("frozen v1 panel changes the task-to-bucket binding")
    if panel.get("n_per_task_level") != 30:
        errors.append("frozen v1 panel does not declare 30 cells per task and level")

    cells = panel.get("cells")
    if not isinstance(cells, list):
        return [*errors, "frozen v1 panel cells are not a list"]
    expected_n = len(DEFAULT_BREADTH_TASKS) * len(BREADTH_LEVELS) * 30
    if panel.get("n_cells") != expected_n or len(cells) != expected_n:
        errors.append(
            f"frozen v1 panel expected {expected_n} cells, found {len(cells)}"
        )

    sources = panel.get("sources")
    if not isinstance(sources, list):
        sources = []
        errors.append("frozen v1 panel sources are not a list")
    source_by_frame: dict[tuple[str, str], Mapping] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            errors.append("frozen v1 panel has a malformed hierarchy source record")
            continue
        task = str(source.get("task"))
        level = str(source.get("level"))
        key = (task, level)
        if key in source_by_frame:
            errors.append(f"{task}/{level}: duplicate hierarchy source binding")
            continue
        source_by_frame[key] = source
        if task not in DEFAULT_BREADTH_TASKS or level not in BREADTH_LEVELS:
            errors.append(f"{task}/{level}: source is outside the frozen frame")
            continue
        if source.get("bucket") != DEFAULT_BREADTH_BUCKETS[task]:
            errors.append(f"{task}/{level}: hierarchy source bucket changed")
        relative = source.get("path")
        expected_sha = source.get("sha256")
        path = ROOT / str(relative or "")
        if not relative or not path.is_file() or sha256_file(path) != expected_sha:
            errors.append(f"hierarchy source changed: {relative}")

    expected_frames = {
        (task, level) for task in DEFAULT_BREADTH_TASKS for level in BREADTH_LEVELS
    }
    if set(source_by_frame) != expected_frames:
        errors.append("frozen v1 hierarchy sources do not cover every task and level")

    ids: list[object] = []
    node_ids: list[object] = []
    metric_ids: list[object] = []
    frame_counts: dict[tuple[str, str], int] = {}
    frame_ranks: dict[tuple[str, str], set[int]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            errors.append("frozen v1 panel has a malformed cell record")
            continue
        cell_id = cell.get("id")
        missing = sorted(_LEGACY_CELL_FIELDS - set(cell))
        if missing:
            errors.append(f"{cell_id}: missing frozen v1 fields {missing}")
            continue

        ids.append(cell_id)
        node_ids.append(cell.get("node_id"))
        metric_ids.append(cell.get("metric_id"))
        task = str(cell.get("task"))
        level = str(cell.get("level"))
        frame = (task, level)
        frame_counts[frame] = frame_counts.get(frame, 0) + 1
        rank = cell.get("selection_rank")
        if _is_plain_int(rank):
            frame_ranks.setdefault(frame, set()).add(rank)

        if task not in DEFAULT_BREADTH_TASKS or level not in BREADTH_LEVELS:
            errors.append(f"{cell_id}: cell is outside the frozen task/level frame")
            continue
        if (
            cell.get("domain") != task
            or cell.get("bucket") != DEFAULT_BREADTH_BUCKETS[task]
            or cell_id != f"TB::{cell.get('node_id')}"
            or cell.get("metric_id") != cell.get("node_id")
        ):
            errors.append(f"{cell_id}: frozen v1 cell identity changed")

        source = source_by_frame.get(frame)
        if source is None or (
            cell.get("source_path") != source.get("path")
            or cell.get("source_sha256") != source.get("sha256")
            or cell.get("bucket") != source.get("bucket")
        ):
            errors.append(f"{cell_id}: cell/source binding changed")

        immediate_ids = cell.get("immediate_source_ids")
        if not isinstance(immediate_ids, list) or not immediate_ids:
            errors.append(f"{cell_id}: invalid immediate-source identities")
        else:
            immediate_sha = hashlib.sha256(
                json.dumps(immediate_ids, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if immediate_sha != cell.get("immediate_source_sha256"):
                errors.append(f"{cell_id}: immediate-source hash mismatch")

        children = cell.get("children")
        try:
            leaf_ids = sorted(hierarchy_leaf_support_ids(children))
        except (TypeError, ValueError):
            leaf_ids = []
        if not leaf_ids:
            errors.append(f"{cell_id}: invalid raw-leaf support")
        else:
            leaf_sha = hashlib.sha256(
                json.dumps(leaf_ids, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if (
                len(leaf_ids) != cell.get("leaf_support_count")
                or leaf_sha != cell.get("leaf_support_sha256")
            ):
                errors.append(f"{cell_id}: raw-leaf support binding changed")

        integer_bounds = {
            "dependency_component_size": 1,
            "dependency_degree": 0,
            "source_assignment_multiplicity_max": 1,
            "provenance_component_size": 1,
            "provenance_overlap_degree": 0,
            "provenance_assignment_multiplicity_max": 1,
            "selection_rank": 0,
            "stratum_population_n": 1,
            "stratum_selected_n": 1,
        }
        if (
            not cell.get("dependency_component_id")
            or not cell.get("provenance_component_id")
            or any(
                not _is_plain_int(cell.get(field), minimum=minimum)
                for field, minimum in integer_bounds.items()
            )
        ):
            errors.append(f"{cell_id}: invalid legacy dependency/design metadata")
        probability = cell.get("inclusion_probability")
        weight = cell.get("design_weight")
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(float(probability))
            or not 0 < float(probability) <= 1
            or not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0
            or not str(cell.get("breadth_stratum")).strip()
        ):
            errors.append(f"{cell_id}: invalid legacy sampling-design metadata")

    for label, values in (
        ("cell ids", ids),
        ("node ids", node_ids),
        ("metric ids", metric_ids),
    ):
        if None in values or len(values) != len(set(values)):
            errors.append(f"frozen v1 panel has missing or duplicate {label}")

    for frame in sorted(expected_frames):
        if frame_counts.get(frame) != 30:
            errors.append(
                f"{frame[0]}/{frame[1]}: expected 30 cells, "
                f"found {frame_counts.get(frame, 0)}"
            )
        if frame_ranks.get(frame) != set(range(30)):
            errors.append(f"{frame[0]}/{frame[1]}: selection ranks are not 0..29")

    inventory = panel.get("inventory")
    if not isinstance(inventory, list):
        errors.append("frozen v1 panel inventory is not a list")
    else:
        inventory_frames = {
            (str(row.get("task")), str(row.get("level")))
            for row in inventory
            if isinstance(row, Mapping)
            and row.get("bucket") == DEFAULT_BREADTH_BUCKETS.get(str(row.get("task")))
            and row.get("n_selected") == 30
        }
        if len(inventory) != len(expected_frames) or inventory_frames != expected_frames:
            errors.append("frozen v1 inventory does not cover the 30-per-level frame")

    return errors


def validate_hierarchy_panel(panel: Mapping) -> list[str]:
    """Validate v1 with its frozen contract and otherwise retain the live validator."""

    if panel.get("schema") == FROZEN_V1_SCHEMA:
        return validate_frozen_v1_panel(panel)
    return validate_metric_panel(dict(panel))
