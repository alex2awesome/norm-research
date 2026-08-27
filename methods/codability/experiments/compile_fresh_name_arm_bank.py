#!/usr/bin/env python
"""Compile source-only name experiment arms and matched controls before executor scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import (
    BREADTH_FIRST_ALLOCATION_STRATEGY,
    sha256_file,
    text_sha256,
)
from methods.codability.experiments.policy_data import (
    ANALYSIS_IMPLEMENTATION_PATHS,
    _resolve_declared_path,
    validate_policy_articulation_selection_provenance,
)
from methods.codability.experiments.score_fresh_name_arms import load_lockbox_selection
from methods.codability.experiments.score_fresh_target_views import load_manifest as load_target_manifest
from methods.codability.experiments.validate_fresh_item_partitions import validate_packet
from methods.codability.unit_count_grid import leaf_units
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.experiments.mine_clusters import (
    hierarchy_groups,
    hierarchy_leaf_support_ids,
    hierarchy_terminal_frontier,
    hierarchy_terminal_frontier_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = Path("notebooks/data/two_faces_20260702")
SOURCE_FILES = {
    "humor": DATA / "r3_humor/grid_humor_v1/messages.json",
    "cw": DATA / "r3_cw/grid_cw_v1/messages.json",
    "pr": DATA / "r3_pr/grid_pr_v1/messages.json",
}
CELL_TARGETS = {
    ("humor", 23): ["llama70_n_target"],
    ("humor", 49): ["llama70_n_target", "gemma31_target"],
    ("pr", 8): ["llama70_n_target"],
    ("cw", 27): ["qwen7_target"],
}
RUNG_CHANNEL = {
    "definition": "declarative", "explanation": "explanatory",
    "full_rubric": "procedural", "exemplars_v2": "ostensive",
    "dossier_v2": "composed",
}
NEUTRAL_WORDS = (
    "The document is presented here for routine review. Read the material in order and consider "
    "only what appears on the page. The pages may contain sections, sentences, examples, and "
    "ordinary formatting. Complete the review consistently using the same process for every item. "
    "No additional background information is supplied. The file was assembled for this evaluation "
    "and its placement does not imply any particular judgment."
).split()

DEFAULT_BREADTH_TASKS = (
    "code-review",
    "creative-writing",
    "grant-funding",
    "humor",
    "legal-outcome-prediction",
    "math-stackexchange",
    "news-homepages",
    "notice-and-comment",
    "patents",
    "peer-review",
    "press-releases",
)
# General is the shared hierarchy when it has enough complete nodes. Notice-and-comment and
# patents need their specific hierarchy: unlike the historical merged-only accessor, the complete
# specific inventories (merged nodes + grandparents) contain 34 and 42 R3 nodes respectively.
DEFAULT_BREADTH_BUCKETS = {
    **{task: "general" for task in DEFAULT_BREADTH_TASKS},
    "notice-and-comment": "specific",
    "patents": "specific",
}
BREADTH_LEVELS = ("R1", "R2", "R3")
# Select the broadest round first, then fill progressively finer rounds with raw-rubric-disjoint
# records.  Across all six fixed permutations this outcome-blind order maximizes task-global raw
# provenance components (812/990 in the source snapshot, versus 638 for R1→R2→R3) and minimizes
# the largest inherited block, while final artifact order remains canonical R1/R2/R3.
BREADTH_SELECTION_LEVEL_ORDER = ("R3", "R2", "R1")
BREADTH_READOUT_MANIFEST = Path(__file__).with_name(
    "tacit_breadth_readout_manifest_v1.json")
SAME_VERSION_MODEL_TEMPLATE = Path(__file__).with_name(
    "same_version_upper_execution_manifest_v1.json")
BREADTH_CONFIRMATION_ROOT = DATA / "tacit_breadth_confirmation_v3"
BREADTH_CALIBRATION_REPORT = BREADTH_CONFIRMATION_ROOT / "calibration_report.json"
BREADTH_LOCKBOX_RELEASE = BREADTH_CONFIRMATION_ROOT / "calibration_release.json"
# The production confirmation keeps one logical arm/form row per backend call.  A real BF16
# smoke found deterministic batch-shape drift at eight rows (maximum 3.06e-5), while explicit
# row-batch one was bit-identical to the historical scalar path.  Keep the optimization available
# in the scorer, but do not redefine the frozen readout numerics to buy scheduling throughput.
BREADTH_TEACHER_FORCED_ROW_BATCH_SIZE = 1
# Breadth production does not inherit undeclared cluster-shell tuning.  Spawn is required by the
# vLLM offline worker layout used by the integrated launcher; the architecture override was found
# ambiently set to an A100 target on the B200 host and is therefore explicitly removed.
BREADTH_RUNTIME_ENVIRONMENT_OVERRIDES = {
    "VLLM_GPU_MEM_UTIL": None,
    "VLLM_BLOCK_SIZE": None,
    "VLLM_ENFORCE_EAGER": None,
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "FLASHINFER_CUDA_ARCHS": None,
}

BREADTH_SCORING_IMPLEMENTATION = (
    "methods/codability/__init__.py",
    "methods/codability/experiments/__init__.py",
    # The breadth launcher fixes the exact environment, GPU-cap preflight, domain shards, and
    # authenticated scorer invocations. It deliberately has no embedded execution-manifest hash,
    # avoiding a manifest↔launcher hash cycle while allowing the manifest to bind this file.
    "methods/codability/experiments/run_tacit_breadth_search_sk3.sh",
    "methods/codability/experiments/score_fresh_name_arms.py",
    "methods/codability/experiments/score_adaptive_ostensive_orbits.py",
    "methods/codability/experiments/score_fresh_target_views.py",
    "methods/codability/experiments/validate_fresh_item_partitions.py",
    "methods/codability/experiments/build_fresh_item_partitions.py",
    "methods/codability/experiments/policy_data.py",
    "methods/codability/experiments/shard_fresh_score_artifact.py",
    "methods/metric_implementer/__init__.py",
    "methods/metric_implementer/config.py",
    "methods/metric_implementer/vllm_backend.py",
    "methods/metric_implementer/manifest.py",
    "methods/metric_implementer/artifact.py",
)
# The runner imports the shared arm selector from the scorer, so that transitive dependency is
# bound here as well as in the scoring closure.  The path list itself lives in policy_data so the
# manifest declaration and the report generator's self-record can never drift apart again.
BREADTH_ANALYSIS_IMPLEMENTATION = ANALYSIS_IMPLEMENTATION_PATHS
BREADTH_COMPILATION_IMPLEMENTATION = (
    "methods/codability/__init__.py",
    "methods/codability/experiments/__init__.py",
    "methods/codability/experiments/compile_fresh_name_arm_bank.py",
    "methods/codability/experiments/build_fresh_item_partitions.py",
    "methods/codability/experiments/validate_fresh_item_partitions.py",
    "methods/codability/experiments/policy_data.py",
    "methods/codability/experiments/score_fresh_name_arms.py",
    "methods/codability/experiments/run_policy_isomorphism.py",
    "methods/metric_implementer/__init__.py",
    "methods/metric_implementer/experiments/__init__.py",
    "methods/metric_implementer/experiments/mine_clusters.py",
    "methods/metric_implementer/config.py",
    "methods/codability/unit_count_grid.py",
    "methods/metric_implementer/experiments/unit_certificate.py",
)
CONCLUDING_SCORING_IMPLEMENTATION = tuple(
    "methods/codability/experiments/run_concluding_policy_confirmation_sk3.sh"
    if path == "methods/codability/experiments/run_tacit_breadth_search_sk3.sh"
    else path
    for path in BREADTH_SCORING_IMPLEMENTATION
)
BREADTH_SELECTION_POLICY = {
    "schema": "tacit_breadth_selection_policy/v2",
    "maximum_candidates_per_cell": 4,
    "minimum_candidates_per_cell": 1,
    "rank_diversity_tolerance": 0.10,
    "roles_in_order": [
        "best_functional_rank",
        "best_vector_identity",
        "best_component_distinct_route_within_rank_tolerance",
        "best_address_dose",
    ],
    "primary_order": (
        "minimum of adverse-form and quotient Spearman descending; observed functional "
        "substitution; content-specific point superiority to both exact controls; adverse MAE, "
        "flip rate, absolute bias, semantic words, and arm id ascending"
    ),
    "vector_order": (
        "maximum normalized excess beyond the target self-form band ascending, then the primary "
        "order; all four policy coordinates are retained"
    ),
    "diversity_rule": (
        "prefer a content-specific different-channel route whose declared nonempty component set "
        "is incomparable with the rank-optimal route, provided its rank floor is within 0.10; "
        "fall back to the best such component-distinct route, with primary order breaking ties"
    ),
    "dose_rule": (
        "retain the best address_dose arm by the primary order so validation estimates the "
        "unit/full-text dose frontier even when another channel wins"
    ),
    "null_cell_rule": (
        "never drop a metric because search is null: retain its best explicit arm and controls "
        "so validation estimates the unconditional prevalence denominator"
    ),
    "control_rule": (
        "each selected content arm carries exactly one exact-added-word inert control and one "
        "same-task/level wrong-construct control with the identical form orbit"
    ),
}


def _hierarchy_source_path(task: str, bucket: str, level: str) -> Path:
    stem = "r1_refined" if level == "R1" else f"{level.lower()}_expanded"
    return REPO_ROOT / "outputs" / "hierarchy" / f"{task}_{bucket}_{stem}.json"


def _breadth_bin(value: int, ordered_values: list[int]) -> str:
    if not ordered_values:
        return "unknown"
    lower = ordered_values[(len(ordered_values) - 1) // 3]
    upper = ordered_values[(2 * (len(ordered_values) - 1)) // 3]
    if value <= lower:
        return "narrow"
    if value <= upper:
        return "middle"
    return "broad"


def _construct_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


class _TaskRawProvenanceGraph:
    """Track inherited raw-rubric dependence across all three rounds of one task."""

    def __init__(self, *, task: str, bucket: str):
        self.task = task
        self.bucket = bucket
        self.parents: list[int] = []
        self.sizes: list[int] = []
        self.supports: list[frozenset[str]] = []
        self.node_ids: list[str] = []
        self.raw_owners: dict[str, list[int]] = {}
        self.construct_keys: set[str] = set()

    def _root(self, index: int) -> int:
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def _union(self, left: int, right: int) -> None:
        left_root, right_root = self._root(left), self._root(right)
        if left_root == right_root:
            return
        # Attach the older component to the newly selected node's component.  Component identity
        # is recomputed from sorted node ids after selection, so root direction is not semantic.
        self.parents[left_root] = right_root
        self.sizes[right_root] += self.sizes[left_root]

    def overlap_roots(self, support: frozenset[str]) -> set[int]:
        return {
            self._root(owner)
            for raw_id in support
            for owner in self.raw_owners.get(raw_id, ())
        }

    def priority(self, support: frozenset[str]) -> tuple[int, int, int]:
        """Prefer no inherited collision, then avoid merging or enlarging large components."""
        roots = self.overlap_roots(support)
        component_sizes = [self.sizes[root] for root in roots]
        return (
            len(roots),
            sum(component_sizes),
            max(component_sizes, default=0),
        )

    def add(self, *, node_id: str, construct: str,
            support: frozenset[str]) -> None:
        if not support:
            raise ValueError(f"{node_id}: task-level raw provenance support is empty")
        construct_key = _construct_key(construct)
        if not construct_key or construct_key in self.construct_keys:
            raise ValueError(f"{node_id}: duplicate or empty task-level construct name")
        overlapping = self.overlap_roots(support)
        index = len(self.parents)
        self.parents.append(index)
        self.sizes.append(1)
        self.supports.append(support)
        self.node_ids.append(node_id)
        for root in overlapping:
            self._union(root, index)
        for raw_id in support:
            self.raw_owners.setdefault(raw_id, []).append(index)
        self.construct_keys.add(construct_key)

    def annotations(self) -> dict[str, dict]:
        members: dict[int, list[int]] = {}
        for index in range(len(self.parents)):
            members.setdefault(self._root(index), []).append(index)
        result = {}
        for indices in members.values():
            node_ids = sorted(self.node_ids[index] for index in indices)
            digest = hashlib.sha256(json.dumps(
                node_ids, ensure_ascii=False).encode()).hexdigest()[:20]
            component_id = (
                f"{self.task}::{self.bucket}::all-rounds::raw-provenance::{digest}"
            )
            for index in indices:
                neighbours = set()
                multiplicity = 1
                for raw_id in self.supports[index]:
                    owners = self.raw_owners[raw_id]
                    multiplicity = max(multiplicity, len(owners))
                    neighbours.update(owners)
                neighbours.discard(index)
                result[self.node_ids[index]] = {
                    "task_raw_provenance_component_id": component_id,
                    "task_raw_provenance_component_size": len(indices),
                    "task_raw_provenance_overlap_degree": len(neighbours),
                    "task_raw_provenance_assignment_multiplicity_max": multiplicity,
                }
        return result


def _stable_stratified_panel(
        nodes: list[dict], *, n: int, salt: str,
        task_provenance_graph: _TaskRawProvenanceGraph | None = None) -> list[dict]:
    """Outcome-blind, dependence-diverse sample within the declared design strata.

    Source kind and breadth tertile determine the allocation exactly as before.  Within each
    stratum, however, a stable-hash candidate that adds both a new immediate-dependency component
    and a new inherited-provenance component is preferred before one that repeats either source
    of dependence.  This improves the effective metric sample without deleting overlapping nodes
    from the native action-node target population or pretending that the resulting panel is a
    partition.
    """
    eligible = [
        node for node in nodes
        if node.get("merged_name")
        and len(_words(node.get("merged_description", ""))) >= 8
        and node.get("all_leaves")
        and (
            task_provenance_graph is None
            or _construct_key(node["merged_name"])
            not in task_provenance_graph.construct_keys
        )
    ]
    if len(eligible) < n:
        raise ValueError(
            f"breadth quota requires {n} eligible hierarchy nodes; found {len(eligible)}"
        )
    sizes = sorted(max(1, int(node.get("total_leaf_rubrics", 0))) for node in eligible)
    strata: dict[tuple[str, str], list[dict]] = {}
    for node in eligible:
        key = (
            str(node["source_kind"]),
            _breadth_bin(max(1, int(node.get("total_leaf_rubrics", 0))), sizes),
        )
        strata.setdefault(key, []).append(node)
    for key, rows in strata.items():
        rows.sort(key=lambda row: text_sha256(f"{salt}|{key}|{row['node_id']}"))
    raw_support_by_node = {
        node["node_id"]: frozenset(hierarchy_leaf_support_ids(node["all_leaves"]))
        for node in eligible
    }
    if any(not values for values in raw_support_by_node.values()):
        raise ValueError("breadth sampler requires nonempty task-level raw provenance support")
    population_counts = {key: len(rows) for key, rows in strata.items()}

    selected = []
    selected_dependency_components: set[str] = set()
    selected_provenance_components: set[str] = set()
    keys = sorted(strata)
    while len(selected) < n:
        progressed = False
        for key in keys:
            rows = strata[key]
            if task_provenance_graph is not None:
                rows[:] = [
                    row for row in rows
                    if _construct_key(row["merged_name"])
                    not in task_provenance_graph.construct_keys
                ]
            if rows and len(selected) < n:
                for row in rows:
                    if (not isinstance(row.get("dependency_component_id"), str)
                            or not row["dependency_component_id"]
                            or not isinstance(row.get("provenance_component_id"), str)
                            or not row["provenance_component_id"]):
                        raise ValueError(
                            "breadth sampler requires complete dependency/provenance identities"
                        )

                def dependence_priority(row: dict) -> tuple[int, int, int, int, int, int]:
                    dependency_new = (
                        row["dependency_component_id"]
                        not in selected_dependency_components
                    )
                    provenance_new = (
                        row["provenance_component_id"]
                        not in selected_provenance_components
                    )
                    # Raw inherited overlap is the broader dependence relation, so a candidate
                    # that adds only provenance diversity precedes one that adds only immediate
                    # dependency diversity.  ``rows`` is already stable-hash ordered; the index
                    # is the final deterministic tie-breaker.
                    category = (
                        0 if dependency_new and provenance_new else
                        1 if provenance_new else
                        2 if dependency_new else
                        3
                    )
                    task_priority = (
                        task_provenance_graph.priority(
                            raw_support_by_node[row["node_id"]])
                        if task_provenance_graph is not None else (0, 0, 0)
                    )
                    return (
                        *task_priority,
                        category,
                        int(row["provenance_component_size"]),
                        int(row["dependency_component_size"]),
                    )

                best_index = min(
                    range(len(rows)),
                    key=lambda index: (*dependence_priority(rows[index]), index),
                )
                row = dict(rows.pop(best_index))
                row["breadth_stratum"] = key[1]
                row["selection_rank"] = len(selected)
                selected.append(row)
                selected_dependency_components.add(row["dependency_component_id"])
                selected_provenance_components.add(row["provenance_component_id"])
                if task_provenance_graph is not None:
                    task_provenance_graph.add(
                        node_id=row["node_id"],
                        construct=row["merged_name"],
                        support=raw_support_by_node[row["node_id"]],
                    )
                progressed = True
        if not progressed:
            break
    if len(selected) != n:
        raise ValueError(f"stratified sampler produced {len(selected)}/{n} nodes")
    selected_counts = {
        key: sum(
            (row["source_kind"], row["breadth_stratum"]) == key
            for row in selected
        )
        for key in strata
    }
    for row in selected:
        key = (row["source_kind"], row["breadth_stratum"])
        population_n = population_counts[key]
        selected_n = selected_counts[key]
        row["stratum_population_n"] = population_n
        row["stratum_selected_n"] = selected_n
        # The diversity-prioritized within-stratum choice is deliberately not a simple
        # random sample.  These quantities describe frozen stratum coverage and a nominal
        # post-stratification sensitivity; they are not node-level inclusion probabilities
        # or Horvitz-Thompson design weights.
        row["stratum_coverage_fraction"] = selected_n / population_n
        row["nominal_poststratification_weight"] = population_n / selected_n
    return selected


def compile_metric_panel(*, tasks: tuple[str, ...] = DEFAULT_BREADTH_TASKS,
                         task_buckets: dict[str, str] = DEFAULT_BREADTH_BUCKETS,
                         n_per_task_level: int = 30,
                         salt: str = "tacit-breadth-panel-v3-task-global-diverse") -> dict:
    """Freeze a level-safe, source-bound 30 x task x R1/R2/R3 metric panel."""
    cells, sources, inventory, terminal_sensitivities = [], [], [], []
    for task in tasks:
        if task not in task_buckets:
            raise ValueError(f"no hierarchy bucket declared for {task!r}")
        bucket = task_buckets[task]
        task_cell_start = len(cells)
        task_inventory_start = len(inventory)
        task_provenance_graph = _TaskRawProvenanceGraph(task=task, bucket=bucket)
        for level in BREADTH_SELECTION_LEVEL_ORDER:
            source_path = _hierarchy_source_path(task, bucket, level)
            if not source_path.is_file():
                raise ValueError(f"missing hierarchy source: {source_path}")
            nodes = hierarchy_groups(task, bucket, level)
            eligible = [
                node for node in nodes
                if node.get("merged_name")
                and len(_words(node.get("merged_description", ""))) >= 8
                and node.get("all_leaves")
            ]
            chosen = _stable_stratified_panel(
                nodes,
                n=n_per_task_level,
                salt=f"{salt}|{task}|{bucket}|{level}",
                task_provenance_graph=task_provenance_graph,
            )
            source_rel = str(source_path.relative_to(REPO_ROOT))
            source_sha = sha256_file(source_path)
            sources.append({
                "task": task,
                "bucket": bucket,
                "level": level,
                "path": source_rel,
                "sha256": source_sha,
            })
            inventory.append({
                "task": task,
                "bucket": bucket,
                "level": level,
                "n_complete_nodes": len(nodes),
                "n_eligible_nodes": len(eligible),
                "n_sampling_frame_nodes": sum({
                    (node["source_kind"], node["breadth_stratum"]):
                        node["stratum_population_n"]
                    for node in chosen
                }.values()),
                "n_excluded_prior_round_exact_name_duplicates": (
                    len(eligible) - sum({
                        (node["source_kind"], node["breadth_stratum"]):
                            node["stratum_population_n"]
                        for node in chosen
                    }.values())
                ),
                "n_selected": len(chosen),
                "selected_dependency_components": len({
                    node["dependency_component_id"] for node in chosen
                }),
                "selected_raw_provenance_components": len({
                    node["provenance_component_id"] for node in chosen
                }),
                "selected_jointly_dependence_disjoint_nodes": sum(
                    node["dependency_degree"] == 0
                    and node["provenance_overlap_degree"] == 0
                    for node in chosen
                ),
                "source_kind_counts": {
                    kind: sum(node["source_kind"] == kind for node in nodes)
                    for kind in sorted({node["source_kind"] for node in nodes})
                },
                "dependency_components": len({
                    node["dependency_component_id"] for node in nodes
                }),
                "largest_dependency_component": max(
                    node["dependency_component_size"] for node in nodes
                ),
                "nodes_with_reused_immediate_sources": sum(
                    node["dependency_degree"] > 0 for node in nodes
                ),
                "maximum_source_assignment_multiplicity": max(
                    node["source_assignment_multiplicity_max"] for node in nodes
                ),
                "raw_provenance_components": len({
                    node["provenance_component_id"] for node in nodes
                }),
                "largest_raw_provenance_component": max(
                    node["provenance_component_size"] for node in nodes
                ),
                "nodes_with_raw_provenance_overlap": sum(
                    node["provenance_overlap_degree"] > 0 for node in nodes
                ),
                "maximum_raw_provenance_assignment_multiplicity": max(
                    node["provenance_assignment_multiplicity_max"] for node in nodes
                ),
            })
            if level == "R1":
                terminal_sensitivities.append({
                    "task": task,
                    "bucket": bucket,
                    "level": level,
                    "available": False,
                    "reason": (
                        "exact complete-linkage input snapshot is not reproducible from the "
                        "local refined action file; native action-node frame only"
                    ),
                })
            else:
                terminal_nodes = hierarchy_terminal_frontier(task, bucket, level)
                terminal_audit = hierarchy_terminal_frontier_audit(task, bucket, level)
                terminal_eligible = [
                    node for node in terminal_nodes
                    if node.get("merged_name")
                    and len(_words(node.get("merged_description", ""))) >= 8
                    and node.get("all_leaves")
                ]
                if len(terminal_eligible) < n_per_task_level:
                    raise ValueError(
                        f"{task}/{level}: terminal-frontier sensitivity has only "
                        f"{len(terminal_eligible)} eligible nodes"
                    )
                terminal_identity = [{
                    "node_id": node["node_id"],
                    "name": node["merged_name"],
                    "description": node["merged_description"],
                    "frontier_role": node["frontier_role"],
                    "frontier_source_ids": node["frontier_source_ids"],
                } for node in terminal_nodes]
                terminal_audit = {
                    **terminal_audit,
                    "source_path": source_rel,
                    "n_eligible_nodes": len(terminal_eligible),
                    "frontier_content_sha256": hashlib.sha256(json.dumps(
                        terminal_identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                }
                terminal_sensitivities.append(terminal_audit)
            for node in chosen:
                cells.append({
                    "id": f"TB::{node['node_id']}",
                    "node_id": node["node_id"],
                    "task": task,
                    "domain": task,
                    "level": level,
                    "bucket": bucket,
                    "metric_id": node["node_id"],
                    "source_kind": node["source_kind"],
                    "source_index": node["source_index"],
                    "legacy_group_idx": node["group_idx"],
                    "immediate_source_ids": node["immediate_source_ids"],
                    "immediate_source_sha256": node["immediate_source_sha256"],
                    "leaf_support_count": node["leaf_support_count"],
                    "leaf_support_sha256": node["leaf_support_sha256"],
                    "dependency_component_id": node["dependency_component_id"],
                    "dependency_component_size": node["dependency_component_size"],
                    "dependency_degree": node["dependency_degree"],
                    "source_assignment_multiplicity_max": node[
                        "source_assignment_multiplicity_max"],
                    "provenance_component_id": node["provenance_component_id"],
                    "provenance_component_size": node["provenance_component_size"],
                    "provenance_overlap_degree": node["provenance_overlap_degree"],
                    "provenance_assignment_multiplicity_max": node[
                        "provenance_assignment_multiplicity_max"],
                    "construct": node["merged_name"],
                    "description": node["merged_description"],
                    "total_leaf_rubrics": node["total_leaf_rubrics"],
                    "components": node["component_children"],
                    "children": node["all_leaves"],
                    "breadth_stratum": node["breadth_stratum"],
                    "selection_rank": node["selection_rank"],
                    "stratum_population_n": node["stratum_population_n"],
                    "stratum_selected_n": node["stratum_selected_n"],
                    "stratum_coverage_fraction": node[
                        "stratum_coverage_fraction"],
                    "nominal_poststratification_weight": node[
                        "nominal_poststratification_weight"],
                    "source_path": source_rel,
                    "source_sha256": source_sha,
                })
        task_cells = cells[task_cell_start:]
        annotations = task_provenance_graph.annotations()
        if (len(task_cells) != len(BREADTH_LEVELS) * n_per_task_level
                or set(annotations) != {cell["node_id"] for cell in task_cells}):
            raise ValueError(
                f"{task}: task-level raw-provenance graph does not cover the panel"
            )
        for cell in task_cells:
            cell.update(annotations[cell["node_id"]])
        for row in inventory[task_inventory_start:]:
            level_cells = [cell for cell in task_cells if cell["level"] == row["level"]]
            row["selected_task_raw_provenance_components"] = len({
                cell["task_raw_provenance_component_id"] for cell in level_cells
            })
            row["largest_selected_task_raw_provenance_component"] = max(
                cell["task_raw_provenance_component_size"] for cell in level_cells
            )
        level_rank = {level: index for index, level in enumerate(BREADTH_LEVELS)}
        cells[task_cell_start:] = sorted(
            task_cells,
            key=lambda cell: (level_rank[cell["level"]], cell["selection_rank"]),
        )
        sources[-len(BREADTH_LEVELS):] = sorted(
            sources[-len(BREADTH_LEVELS):], key=lambda row: level_rank[row["level"]])
        inventory[-len(BREADTH_LEVELS):] = sorted(
            inventory[-len(BREADTH_LEVELS):], key=lambda row: level_rank[row["level"]])
        terminal_sensitivities[-len(BREADTH_LEVELS):] = sorted(
            terminal_sensitivities[-len(BREADTH_LEVELS):],
            key=lambda row: level_rank[row["level"]],
        )
    payload = {
        "schema": "tacit_breadth_metric_panel/v3",
        "status": "frozen-outcome-blind-hierarchy-sample",
        "objective": "at least 30 metrics per task in each R1/R2/R3 hierarchy level",
        "hierarchy_frame": {
            "generation": "legacy-expanded-source-action-node-dag-v1",
            "sampling_unit": (
                "one native named action-node record emitted by the frozen hierarchy round"
            ),
            "R1_source": "native r1_refined actions (parented_trees + merged_trees)",
            "R2_source": "native r2_expanded actions (merged_groups + grandparents)",
            "R3_source": "native r3_expanded actions (merged_groups + grandparents)",
            "is_partition": False,
            "overlap_handling": (
                "Immediate-source reuse is frozen as a dependency-component identifier. Metric-"
                "level inference must resample whole dependency components. Within each frozen "
                "source-kind x breadth stratum, outcome-blind selection prefers nodes that add new "
                "dependency and inherited-provenance components before repeats; overlap remains "
                "in the target population and is never silently trimmed. Source-kind-specific and "
                "merged-only results are mandatory sensitivities."
            ),
            "inherited_provenance_handling": (
                "Raw-rubric key reuse is frozen separately both within each hierarchy file and "
                "across all R1/R2/R3 records selected for a task. It is not equated with construct "
                "identity. The task-global raw-provenance block is the conservative task and "
                "aggregate inference unit; level-local dependency/provenance blocks remain "
                "diagnostics. Nominal weights never estimate prevalence over raw human rubrics."
            ),
            "why_not_terminal_carry_forward_as_primary": (
                "A disjoint terminal frontier would require propagating untouched or lower-round "
                "nodes into R2/R3 and would therefore confound hierarchy round with semantic grain. "
                "The native DAG is retained as the construct population; a tightest-first terminal "
                "frontier is a declared secondary sensitivity, not the primary metric frame."
            ),
            "not_the_rebuilt_lexicon_partition": True,
            "interpretation": (
                "R1/R2/R3 are operational round strata of this legacy expanded-source DAG, not "
                "certified same-construct/theme/category partitions. They must not be merged with "
                "the newer lexicon partition lineage or described as a paired ancestry ladder."
            ),
        },
        "tasks": list(tasks),
        "levels": list(BREADTH_LEVELS),
        "selection_level_order": list(BREADTH_SELECTION_LEVEL_ORDER),
        "task_buckets": {task: task_buckets[task] for task in tasks},
        "n_per_task_level": int(n_per_task_level),
        "n_cells": len(cells),
        "sampling_rule": (
            "Require nonempty name, >=8-word source description, and >=1 child; then select "
            "outcome-blind by stable-hash round-robin over source materialization kind and "
            "leaf-count breadth tertile. Across a task, require exact construct-name uniqueness "
            "and select R3 then R2 then R1 so broad supports are fixed before finer records. "
            "Greedily prefer raw-rubric support that does not collide with already selected "
            "rounds; when collision is unavoidable, avoid merging or enlarging large components. "
            "Within each level-local stratum, also prefer a new immediate-dependency "
            "and inherited-provenance component, using stable hash and component size only as "
            "outcome-blind tie-breakers. Never pad, duplicate, or sample with replacement."
        ),
        "prevalence_estimands": {
            "balanced_panel": (
                "unweighted prevalence over the deliberately balanced 30-node panel"
            ),
            "source_inventory_poststratified_sensitivity": (
                "descriptive source_kind x breadth-tertile post-stratification of the frozen "
                "dependence-diverse panel. Because within-stratum component diversity is "
                "prioritized, the nominal weights are not node-level inclusion-probability or "
                "Horvitz-Thompson weights and cannot identify native-inventory prevalence. "
                "Uncertainty must resample frozen dependency components and never treat 30 "
                "overlapping nodes as independent"
            ),
            "mandatory_sensitivities": [
                "source_kind_specific",
                "merged_only",
                "tightest_first_terminal_frontier",
            ],
        },
        "salt": salt,
        "sources": sources,
        "inventory": inventory,
        "terminal_frontier_sensitivities": terminal_sensitivities,
        "cells": cells,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["panel_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_metric_panel(panel: dict) -> list[str]:
    errors = []
    if panel.get("schema") != "tacit_breadth_metric_panel/v3":
        errors.append("metric panel is not the task-global-dependence v3 schema")
    if panel.get("selection_level_order") != list(BREADTH_SELECTION_LEVEL_ORDER):
        errors.append("metric panel changes the frozen cross-round selection order")
    declared = panel.get("panel_content_sha256")
    core = {key: value for key, value in panel.items() if key != "panel_content_sha256"}
    observed = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if declared != observed:
        errors.append("metric panel content hash mismatch")
    cells = panel.get("cells", [])
    ids = [cell.get("id") for cell in cells]
    node_ids = [cell.get("node_id") for cell in cells]
    if None in ids or len(ids) != len(set(ids)):
        errors.append("metric panel has missing or duplicate cell ids")
    if None in node_ids or len(node_ids) != len(set(node_ids)):
        errors.append("metric panel has missing or duplicate node ids")
    for source in panel.get("sources", []):
        path = REPO_ROOT / source.get("path", "")
        if not path.is_file() or sha256_file(path) != source.get("sha256"):
            errors.append(f"hierarchy source changed: {source.get('path')}")
    required_dependency_fields = {
        "source_kind", "source_index", "immediate_source_ids",
        "immediate_source_sha256", "leaf_support_count", "leaf_support_sha256",
        "dependency_component_id", "dependency_component_size", "dependency_degree",
        "source_assignment_multiplicity_max",
        "provenance_component_id", "provenance_component_size",
        "provenance_overlap_degree", "provenance_assignment_multiplicity_max",
        "task_raw_provenance_component_id", "task_raw_provenance_component_size",
        "task_raw_provenance_overlap_degree",
        "task_raw_provenance_assignment_multiplicity_max",
    }
    for cell in cells:
        missing = sorted(required_dependency_fields - set(cell))
        if missing:
            errors.append(f"{cell.get('id')}: missing hierarchy dependency fields {missing}")
            continue
        immediate_hash = hashlib.sha256(json.dumps(
            cell["immediate_source_ids"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if immediate_hash != cell["immediate_source_sha256"]:
            errors.append(f"{cell.get('id')}: immediate-source hash mismatch")
        if (not cell["dependency_component_id"]
                or int(cell["dependency_component_size"]) < 1
                or int(cell["dependency_degree"]) < 0
                or int(cell["source_assignment_multiplicity_max"]) < 1
                or not cell["provenance_component_id"]
                or int(cell["provenance_component_size"]) < 1
                or int(cell["provenance_overlap_degree"]) < 0
                or int(cell["provenance_assignment_multiplicity_max"]) < 1
                or not cell["task_raw_provenance_component_id"]
                or int(cell["task_raw_provenance_component_size"]) < 1
                or int(cell["task_raw_provenance_overlap_degree"]) < 0
                or int(cell[
                    "task_raw_provenance_assignment_multiplicity_max"]) < 1):
            errors.append(f"{cell.get('id')}: invalid hierarchy dependency metadata")
    expected_n = int(panel.get("n_per_task_level", 0))
    for task in panel.get("tasks", []):
        for level in panel.get("levels", []):
            count = sum(
                cell.get("task") == task and cell.get("level") == level
                for cell in cells
            )
            if count != expected_n:
                errors.append(f"{task}/{level}: expected {expected_n} cells, found {count}")
        task_cells = [cell for cell in cells if cell.get("task") == task]
        buckets = {cell.get("bucket") for cell in task_cells}
        if len(buckets) != 1 or None in buckets:
            errors.append(f"{task}: task panel has invalid bucket identities")
            continue
        graph = _TaskRawProvenanceGraph(task=task, bucket=next(iter(buckets)))
        try:
            for cell in task_cells:
                graph.add(
                    node_id=cell["node_id"],
                    construct=cell["construct"],
                    support=frozenset(hierarchy_leaf_support_ids(cell["children"])),
                )
            expected_annotations = graph.annotations()
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{task}: task raw-provenance reconstruction failed: {exc}")
            continue
        for cell in task_cells:
            expected = expected_annotations[cell["node_id"]]
            changed = sorted(
                key for key, value in expected.items() if cell.get(key) != value
            )
            if changed:
                errors.append(
                    f"{cell.get('id')}: task raw-provenance metadata changed {changed}"
                )
    return errors


def _normalize_content(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _truncate_words(text: str, maximum: int) -> str:
    return " ".join(_words(text)[:maximum])


def _balanced_join(sections: list[str], maximum: int) -> str:
    """Retain content from every nonempty channel before spending residual word budget."""
    active = [section.strip() for section in sections if section.strip()]
    if not active:
        return ""
    lengths = [len(_words(section)) for section in active]
    allocations = [min(length, maximum // len(active)) for length in lengths]
    remaining = maximum - sum(allocations)
    while remaining:
        progressed = False
        for index, length in enumerate(lengths):
            if allocations[index] < length and remaining:
                allocations[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return "\n\n".join(
        _truncate_words(section, allocation)
        for section, allocation in zip(active, allocations)
        if allocation
    )


def _child_rule(child: dict) -> str:
    name = str(child.get("name") or "").strip()
    description = str(child.get("description") or "").strip()
    if description and _normalize_content(description) != _normalize_content(name):
        return f"{name}: {description}"
    return name


def _address_id(text: str) -> str:
    return f"address::{text_sha256(_normalize_content(text))[:20]}"


def _contained_component_ids(rendered: str, candidates: list[str]) -> list[str]:
    rendered_key = _normalize_content(rendered)
    result = []
    for candidate in candidates:
        candidate_key = _normalize_content(candidate)
        if candidate_key and candidate_key in rendered_key:
            component_id = _address_id(candidate)
            if component_id not in result:
                result.append(component_id)
    return result


def _explicit_units(cell: dict, *, max_added_words: int) -> list[dict]:
    """Deterministic source address lattice with stable, content-bound component identities."""
    candidates = [
        ("definition", candidate) for candidate in leaf_units(cell["description"])
    ]
    candidates.extend(
        ("component_rule", _child_rule(child)) for child in cell.get("components", []))
    candidates.extend(
        ("leaf_signal", _child_rule(child)) for child in cell.get("children", []))
    seen, units, total = set(), [], 0
    for source, candidate in candidates:
        candidate = str(candidate).strip()
        key = _normalize_content(candidate)
        if not key or key in seen:
            continue
        remaining = max_added_words - total
        if remaining <= 0:
            break
        candidate = _truncate_words(candidate, remaining)
        if not candidate:
            break
        seen.add(key)
        units.append({
            "id": _address_id(candidate),
            "source": source,
            "text": candidate,
            "word_count": len(_words(candidate)),
        })
        total += len(_words(candidate))
    return units


def _breadth_added_contents(cell: dict, *, max_added_words: int) -> list[dict]:
    definition = _truncate_words(cell["description"], max_added_words)
    component_rules = [_child_rule(child) for child in cell.get("components", [])]
    component_rules = [rule for rule in component_rules if rule]
    rules = (
        _truncate_words(
            "Recognition rules:\n" + "\n".join(
                f"{index + 1}. {rule}"
                for index, rule in enumerate(component_rules)),
            max_added_words,
        )
        if component_rules else ""
    )
    leaf_signals = [_child_rule(child) for child in cell.get("children", [])]
    leaf_signals = [signal for signal in leaf_signals if signal]
    evidence = (
        _truncate_words(
            "Source criteria and boundary signals:\n" + "\n".join(
                f"{index + 1}. {signal}"
                for index, signal in enumerate(leaf_signals)),
            max_added_words,
        )
        if leaf_signals else ""
    )
    combined = _balanced_join(
        [f"Definition: {definition}", rules], max_added_words)
    dossier = _balanced_join(
        [f"Definition: {definition}", rules, evidence], max_added_words)
    units = _explicit_units(cell, max_added_words=max_added_words)
    definition_components = _contained_component_ids(
        definition, leaf_units(cell["description"]))
    rule_components = _contained_component_ids(rules, component_rules)
    evidence_components = _contained_component_ids(evidence, leaf_signals)
    combined_components = _contained_component_ids(
        combined, leaf_units(cell["description"]) + component_rules)
    dossier_components = _contained_component_ids(
        dossier, leaf_units(cell["description"]) + component_rules + leaf_signals)
    arms = [
        {"id": "source_definition", "channel": "declarative", "added": definition,
         "provenance": "source_hierarchy_definition",
         "components": definition_components},
        {"id": "source_rules", "channel": "procedural", "added": rules,
         "provenance": "source_hierarchy_immediate_children",
         "components": rule_components},
        {"id": "source_leaf_inventory", "channel": "ostensive", "added": evidence,
         "provenance": "source_hierarchy_leaf_signals",
         "components": evidence_components},
        {"id": "source_definition_rules", "channel": "composed", "added": combined,
         "provenance": "source_definition_plus_children",
         "components": combined_components},
        {"id": "source_dossier", "channel": "composed", "added": dossier,
         "provenance": "source_definition_children_and_leaf_signals",
         "components": dossier_components},
    ]
    if not units:
        raise ValueError(f"{cell['id']}: no explicit source units")
    for size in (1, 2, 4, 8):
        if size >= len(units):
            continue
        arms.append({
            "id": f"source_units_{size}",
            "channel": "address_dose",
            "added": " ".join(unit["text"] for unit in units[:size]),
            "provenance": "source_address_prefix",
            "components": [unit["id"] for unit in units[:size]],
            "n_address_units": size,
        })
    arms.append({
        "id": "source_units_full",
        "channel": "address_dose",
        "added": " ".join(unit["text"] for unit in units),
        "provenance": "source_address_prefix_full",
        "components": [unit["id"] for unit in units],
        "n_address_units": len(units),
    })
    # A few very small hierarchy nodes can render two source routes identically. Preserve the
    # first declared channel and never count byte-equivalent prompts as distinct articulations.
    unique, seen = [], set()
    for arm in arms:
        key = _normalize_content(arm["added"])
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(arm)
    return unique


def _fit_added_words(text: str, n_words: int) -> str:
    words = _words(text)
    if len(words) >= n_words:
        return " ".join(words[:n_words])
    padding = [
        NEUTRAL_WORDS[index % len(NEUTRAL_WORDS)]
        for index in range(n_words - len(words))
    ]
    return " ".join(words + padding)


def _target_plus_added(construct: str, added: str) -> str:
    return f"{construct}\n\n{added}".strip()


def _wrong_breadth_added(*, cell: dict, source_arm: dict,
                         cells_by_id: dict[str, dict],
                         arm_specs: dict[str, dict[str, dict]]) -> tuple[str, dict]:
    target_n = len(_words(source_arm["added"]))
    candidates = []
    for other_id, other in cells_by_id.items():
        if other_id == cell["id"]:
            continue
        if (other["task"], other["bucket"], other["level"]) != (
                cell["task"], cell["bucket"], cell["level"]):
            continue
        other_arm = arm_specs[other_id].get(source_arm["id"])
        if other_arm is None:
            continue
        other_added = other_arm["added"]
        candidates.append((
            abs(len(_words(other_added)) - target_n),
            _jaccard(source_arm["added"], other_added),
            other["node_id"],
            other_added,
            other["construct"],
        ))
    if not candidates:
        raise ValueError(f"{cell['id']}/{source_arm['id']}: no matched wrong construct")
    length_gap, overlap, node_id, added, construct = min(candidates)
    return _fit_added_words(added, target_n), {
        "wrong_node_id": node_id,
        "wrong_construct": construct,
        "source_added_word_count": len(_words(added)),
        "target_added_word_count": target_n,
        "pre_fit_length_gap": length_gap,
        "lexical_jaccard_before_fit": overlap,
    }


def compile_breadth_bank(*, panel: dict, target_model_jobs: tuple[str, ...] = (
        "llama31_70b_name_target",), max_added_words: int = 360) -> dict:
    """Compile full-text and unit-dose articulations into the integrated arm-bank schema."""
    errors = validate_metric_panel(panel)
    if errors:
        raise ValueError(errors)
    cells_by_id = {cell["id"]: cell for cell in panel["cells"]}
    arm_specs = {
        cell_id: {
            arm["id"]: arm
            for arm in _breadth_added_contents(cell, max_added_words=max_added_words)
        }
        for cell_id, cell in cells_by_id.items()
    }
    cells = []
    for cell in panel["cells"]:
        construct = cell["construct"]
        sparse_forms = _forms(construct)
        arms = [{
            "id": "name",
            "channel": "sparse",
            "provenance": "construct_name",
            "control_for": None,
            "semantic_content_word_count": len(_words(construct)),
            "added_content_word_count": 0,
            "content_sha256": text_sha256(construct),
            "components": [],
            "forms": sparse_forms,
        }]
        for source_arm in arm_specs[cell["id"]].values():
            added = source_arm["added"]
            content = _target_plus_added(construct, added)
            source_id = source_arm["id"]
            source = _arm(
                source_id,
                channel=source_arm["channel"],
                content=content,
                provenance=source_arm["provenance"],
                added_content_word_count=len(_words(added)),
                components=source_arm.get("components"),
                n_address_units=source_arm.get("n_address_units"),
            )
            arms.append(source)
            wrong_added, wrong_meta = _wrong_breadth_added(
                cell=cell,
                source_arm=source_arm,
                cells_by_id=cells_by_id,
                arm_specs=arm_specs,
            )
            wrong_content = _target_plus_added(construct, wrong_added)
            inert_content = _target_plus_added(
                construct, _inert(len(_words(added))))
            arms.append(_arm(
                f"control_wrong_{source_id.removeprefix('source_')}",
                channel=source_arm["channel"],
                content=wrong_content,
                provenance="wrong_construct_control",
                control_for=source_id,
                added_content_word_count=len(_words(wrong_added)),
                components=[],
                n_address_units=source_arm.get("n_address_units"),
                control_meta=wrong_meta,
            ))
            arms.append(_arm(
                f"control_inert_{source_id.removeprefix('source_')}",
                channel=source_arm["channel"],
                content=inert_content,
                provenance="inert_length_control",
                control_for=source_id,
                added_content_word_count=len(_words(added)),
                components=[],
                n_address_units=source_arm.get("n_address_units"),
            ))
        cells.append({
            "id": cell["id"],
            "domain": cell["domain"],
            "task": cell["task"],
            "level": cell["level"],
            "bucket": cell["bucket"],
            "metric_id": cell["metric_id"],
            "node_id": cell["node_id"],
            "source_kind": cell["source_kind"],
            "source_index": cell["source_index"],
            "gi": cell["legacy_group_idx"],
            "construct": construct,
            "target_model_jobs": list(target_model_jobs),
            "source_path": cell["source_path"],
            "source_sha256": cell["source_sha256"],
            "breadth_stratum": cell["breadth_stratum"],
            "leaf_support_count": cell["leaf_support_count"],
            "leaf_support_sha256": cell["leaf_support_sha256"],
            "dependency_component_id": cell["dependency_component_id"],
            "dependency_component_size": cell["dependency_component_size"],
            "dependency_degree": cell["dependency_degree"],
            "source_assignment_multiplicity_max": cell[
                "source_assignment_multiplicity_max"],
            "provenance_component_id": cell["provenance_component_id"],
            "provenance_component_size": cell["provenance_component_size"],
            "provenance_overlap_degree": cell["provenance_overlap_degree"],
            "provenance_assignment_multiplicity_max": cell[
                "provenance_assignment_multiplicity_max"],
            "task_raw_provenance_component_id": cell[
                "task_raw_provenance_component_id"],
            "task_raw_provenance_component_size": cell[
                "task_raw_provenance_component_size"],
            "task_raw_provenance_overlap_degree": cell[
                "task_raw_provenance_overlap_degree"],
            "task_raw_provenance_assignment_multiplicity_max": cell[
                "task_raw_provenance_assignment_multiplicity_max"],
            "stratum_population_n": cell["stratum_population_n"],
            "stratum_selected_n": cell["stratum_selected_n"],
            "stratum_coverage_fraction": cell["stratum_coverage_fraction"],
            "nominal_poststratification_weight": cell[
                "nominal_poststratification_weight"],
            "arms": arms,
        })
    payload = {
        "schema": "tacit_breadth_arm_bank/v3",
        "status": "source-only-frozen-before-model-outcomes",
        "metric_panel_content_sha256": panel["panel_content_sha256"],
        "metric_panel_n_cells": panel["n_cells"],
        "target_model_jobs": list(target_model_jobs),
        "max_added_words": max_added_words,
        "cells": cells,
        "arm_strategy": (
            "Every content arm retains the target construct name and adds source-only hierarchy "
            "content. Declarative definition, procedural child rules, their composition, and "
            "nested address-unit prefixes share the same form orbit. Each has exact added-word "
            "inert and wrong-construct controls."
        ),
        "unit_status": (
            "deterministic CUF address-lattice segments; dose units are not certified Omega units"
        ),
        "selection_rule": (
            "Selection may use calibration only. Confirmation must preserve source/control trios "
            "and may not rewrite source content from target outcomes."
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _words(text: str) -> list[str]:
    return str(text).split()


def _fit_words(text: str, n_words: int) -> str:
    words = _words(text)
    if len(words) >= n_words:
        return " ".join(words[:n_words])
    padding = [NEUTRAL_WORDS[i % len(NEUTRAL_WORDS)]
               for i in range(n_words - len(words))]
    return " ".join(words + padding)


def _inert(n_words: int) -> str:
    return " ".join(NEUTRAL_WORDS[i % len(NEUTRAL_WORDS)] for i in range(n_words))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    return len(left & right) / len(left | right) if left and right else 0.0


def _corrected_rung(message: dict, rung: str) -> str:
    rungs = message["rungs"]
    if rung == "exemplars_v2":
        value = rungs["exemplars"]
        return value.replace("Judge by these examples ONLY.\n", "Illustrative examples:\n", 1)
    if rung == "dossier_v2":
        if "dossier_v2" in rungs:
            return rungs["dossier_v2"]
        return rungs["dossier"].replace(
            "Judge by these examples ONLY.\n", "Illustrative examples:\n", 1)
    return rungs[rung]


def _forms(content: str) -> list[dict]:
    values = [
        ("canonical", content),
        ("question", "Does the item satisfy the guidance below? Apply the guidance when "
                     f"answering.\n\n{content}"),
        ("boilerplate", "You are an expert evaluator. Apply the following guidance consistently "
                        f"and only as relevant.\n\n{content}"),
    ]
    return [{"id": form_id, "prompt": prompt, "prompt_sha256": text_sha256(prompt),
             "total_word_count": len(_words(prompt))} for form_id, prompt in values]


def _wrong_construct(messages: dict, *, gi: int, rung: str,
                     target_content: str) -> tuple[int, str, dict]:
    target_n = len(_words(target_content))
    candidates = []
    for candidate_gi, message in messages.items():
        if int(candidate_gi) == gi:
            continue
        try:
            content = _corrected_rung(message, rung)
        except KeyError:
            continue
        overlap = _jaccard(target_content, content)
        length_gap = abs(len(_words(content)) - target_n)
        # Channel integrity comes first: heavily truncating an exemplar/dossier turns it into a
        # malformed control. Prefer source content already within 10% of the selected dose, then
        # minimize the exact gap and lexical overlap. Common template words in example blocks make
        # a hard Jaccard cutoff inappropriate.
        candidates.append((length_gap / max(target_n, 1) > 0.10, overlap, length_gap,
                           int(candidate_gi), content, message["name"]))
    if not candidates:
        raise ValueError(f"no wrong-construct candidate for gi={gi}, rung={rung}")
    _, overlap, length_gap, wrong_gi, content, name = min(candidates)
    fitted = _fit_words(content, target_n)
    return wrong_gi, fitted, {"source_name": name, "source_word_count": len(_words(content)),
                              "target_word_count": target_n, "pre_fit_length_gap": length_gap,
                              "lexical_jaccard_before_fit": overlap}


def _arm(arm_id: str, *, channel: str, content: str, provenance: str,
         control_for: str | None = None, control_meta: dict | None = None,
         added_content_word_count: int | None = None,
         components: list[str] | None = None,
         n_address_units: int | None = None) -> dict:
    result = {
        "id": arm_id, "channel": channel, "provenance": provenance,
        "control_for": control_for, "semantic_content_word_count": len(_words(content)),
        "content_sha256": text_sha256(content), "forms": _forms(content),
        **({"control_meta": control_meta} if control_meta else {}),
    }
    if added_content_word_count is not None:
        result["added_content_word_count"] = int(added_content_word_count)
    if components is not None:
        result["components"] = list(components)
    if n_address_units is not None:
        result["n_address_units"] = int(n_address_units)
    return result


def compile_bank(*, source_files: dict[str, Path] = SOURCE_FILES,
                 target_manifest_path: str | Path | None = None,
                 cell_targets: dict[tuple[str, int], list[str]] | None = None,
                 domain_tasks: dict[str, str] | None = None) -> dict:
    """Compile legacy source messages for an explicitly declared construct panel.

    ``cell_targets`` and ``domain_tasks`` make the established compiler reusable by batched
    confirmations without changing the historical four-cell default artifact.  In particular,
    callers using canonical packet-domain names such as ``press-releases`` can bind the task name
    explicitly instead of creating a second alias directory or a one-off compiler.
    """
    selected_targets = CELL_TARGETS if cell_targets is None else cell_targets
    if not selected_targets:
        raise ValueError("cell_targets must declare at least one construct")
    domain_tasks = {} if domain_tasks is None else domain_tasks
    target_manifest = load_target_manifest(target_manifest_path) if target_manifest_path else \
        load_target_manifest()
    name_forms = {cell["construct"]: cell["forms"] for cell in target_manifest["cells"]
                  if cell["view"] == "N"}
    source_meta, cells = {}, []
    for domain, source_path in source_files.items():
        messages = json.loads(Path(source_path).read_text())
        source_meta[domain] = {"path": str(source_path), "sha256": sha256_file(source_path)}
        for (cell_domain, gi), target_jobs in selected_targets.items():
            if cell_domain != domain:
                continue
            message = messages[str(gi)]
            construct = message["name"]
            sparse_forms = [{**form, "prompt_sha256": text_sha256(form["prompt"]),
                             "total_word_count": len(_words(form["prompt"]))}
                            for form in name_forms[construct]]
            arms = [{"id": "name", "channel": "sparse", "provenance": "construct_name",
                     "control_for": None,
                     "semantic_content_word_count": len(_words(construct)),
                     "content_sha256": text_sha256(construct), "forms": sparse_forms}]
            for rung in RUNG_CHANNEL:
                content = _corrected_rung(message, rung)
                source_id = f"source_{rung}"
                arms.append(_arm(source_id, channel=RUNG_CHANNEL[rung], content=content,
                                 provenance="source_telling"))
                wrong_gi, wrong, wrong_meta = _wrong_construct(
                    messages, gi=gi, rung=rung, target_content=content)
                arms.append(_arm(f"control_wrong_{rung}", channel=RUNG_CHANNEL[rung],
                                 content=wrong, provenance="wrong_construct_control",
                                 control_for=source_id,
                                 control_meta={"wrong_gi": wrong_gi, **wrong_meta}))
                arms.append(_arm(f"control_inert_{rung}", channel=RUNG_CHANNEL[rung],
                                 content=_inert(len(_words(content))),
                                 provenance="inert_length_control", control_for=source_id))
            cell = {"id": f"N_{domain}_{gi}", "domain": domain, "gi": gi,
                    "construct": construct, "target_model_jobs": target_jobs,
                    "arms": arms}
            if domain in domain_tasks:
                cell["task"] = domain_tasks[domain]
            cells.append(cell)
    payload = {
        "schema": "fresh_name_arm_bank/v1",
        "status": "source-only-frozen-before-fresh-executor-outcomes",
        "source_messages": source_meta, "cells": cells,
        "selection_rule": ("On residual_prompt_selection, maximize the paired-bootstrap lower "
                           "confidence bound of adverse-form oriented recovery among source-only "
                           "arms that have positive polarity and meet the signature floor; break "
                           "ties within 0.01 by fewer semantic-content words, then frozen arm id. "
                           "Controls never enter selection. This selects a confirmation candidate, "
                           "not a certified minimum debt."),
        "specificity_rule": ("On the lockbox, a source arm must beat both its same-channel, "
                             "same-content-word-count wrong-construct and inert controls."),
        "cost_rule": ("semantic_content_word_count excludes the frozen generic form wrapper; total "
                      "prompt words are also recorded. These are legacy word costs, not CUF units."),
    }
    if cell_targets is not None:
        payload["declared_cell_targets"] = [
            {
                "domain": domain,
                "gi": gi,
                "target_model_jobs": list(target_jobs),
                **({"task": domain_tasks[domain]} if domain in domain_tasks else {}),
            }
            for (domain, gi), target_jobs in selected_targets.items()
        ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_bank(bank: dict) -> list[str]:
    errors = []
    declared_hash = bank.get("bank_content_sha256")
    core = {key: value for key, value in bank.items() if key != "bank_content_sha256"}
    observed_hash = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if declared_hash != observed_hash:
        errors.append("arm bank content hash mismatch")
    for cell in bank.get("cells", []):
        ids = {arm["id"] for arm in cell["arms"]}
        if len(ids) != len(cell["arms"]) or "name" not in ids:
            errors.append(f"{cell['id']}: duplicate arms or missing name")
        by_id = {arm["id"]: arm for arm in cell["arms"]}
        for arm in cell["arms"]:
            if len(arm["forms"]) != 3:
                errors.append(f"{cell['id']}/{arm['id']}: form orbit is not size three")
            if arm["control_for"]:
                source = by_id.get(arm["control_for"])
                if source is None:
                    errors.append(f"{cell['id']}/{arm['id']}: missing controlled source arm")
                elif source["semantic_content_word_count"] != arm["semantic_content_word_count"]:
                    errors.append(f"{cell['id']}/{arm['id']}: content length mismatch")
    return errors


def _recorded_path(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _artifact_binding(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"frozen artifact is missing: {path}")
    return {"path": _recorded_path(path), "sha256": sha256_file(path)}


def _implementation_binding(paths: tuple[str, ...], *, semantics: str) -> dict:
    files = [_artifact_binding(REPO_ROOT / path) for path in paths]
    return {"semantics": semantics, "files": files}


def _assert_full_breadth_closure(panel: dict, bank: dict) -> None:
    expected = len(DEFAULT_BREADTH_TASKS) * len(BREADTH_LEVELS) * 30
    if (panel.get("schema") != "tacit_breadth_metric_panel/v3"
            or bank.get("schema") != "tacit_breadth_arm_bank/v3"):
        raise ValueError("breadth execution requires the task-global-dependence v3 design")
    if (panel.get("tasks") != list(DEFAULT_BREADTH_TASKS)
            or panel.get("levels") != list(BREADTH_LEVELS)
            or panel.get("n_per_task_level") != 30
            or panel.get("n_cells") != expected):
        raise ValueError("metric panel is not the declared 30 x 11 x R1/R2/R3 breadth panel")
    if validate_metric_panel(panel):
        raise ValueError(validate_metric_panel(panel))
    if validate_bank(bank):
        raise ValueError(validate_bank(bank))
    if (bank.get("metric_panel_content_sha256") != panel.get("panel_content_sha256")
            or len(bank.get("cells", [])) != expected
            or [cell.get("id") for cell in bank.get("cells", [])]
            != [cell.get("id") for cell in panel.get("cells", [])]):
        raise ValueError("arm bank does not close exactly over the frozen metric panel")
    identity_fields = {
        "id": "id",
        "domain": "domain",
        "task": "task",
        "level": "level",
        "bucket": "bucket",
        "metric_id": "metric_id",
        "node_id": "node_id",
        "source_kind": "source_kind",
        "source_index": "source_index",
        "legacy_group_idx": "gi",
        "construct": "construct",
        "source_path": "source_path",
        "source_sha256": "source_sha256",
        "breadth_stratum": "breadth_stratum",
        "leaf_support_count": "leaf_support_count",
        "leaf_support_sha256": "leaf_support_sha256",
        "dependency_component_id": "dependency_component_id",
        "dependency_component_size": "dependency_component_size",
        "dependency_degree": "dependency_degree",
        "source_assignment_multiplicity_max": "source_assignment_multiplicity_max",
        "provenance_component_id": "provenance_component_id",
        "provenance_component_size": "provenance_component_size",
        "provenance_overlap_degree": "provenance_overlap_degree",
        "provenance_assignment_multiplicity_max": (
            "provenance_assignment_multiplicity_max"),
        "task_raw_provenance_component_id": "task_raw_provenance_component_id",
        "task_raw_provenance_component_size": "task_raw_provenance_component_size",
        "task_raw_provenance_overlap_degree": "task_raw_provenance_overlap_degree",
        "task_raw_provenance_assignment_multiplicity_max": (
            "task_raw_provenance_assignment_multiplicity_max"),
        "stratum_population_n": "stratum_population_n",
        "stratum_selected_n": "stratum_selected_n",
        "stratum_coverage_fraction": "stratum_coverage_fraction",
        "nominal_poststratification_weight": "nominal_poststratification_weight",
    }
    for panel_cell, bank_cell in zip(panel["cells"], bank["cells"]):
        mismatches = {
            panel_key: {
                "panel": panel_cell.get(panel_key),
                "bank": bank_cell.get(bank_key),
            }
            for panel_key, bank_key in identity_fields.items()
            if panel_cell.get(panel_key) != bank_cell.get(bank_key)
        }
        if mismatches:
            raise ValueError(
                f"arm bank hierarchy identity differs for {panel_cell.get('id')}: {mismatches}"
            )


def _authenticate_stored_source_membership(
        *, integrity: dict, packet: dict, packet_manifest_path: Path,
        domains: list[str], required_partitions: set[str]) -> dict:
    """Verify a raw-source certificate already bound to the immutable packet.

    This is the explicit offline-host route: packet structure and every item file are recomputed
    locally, while source-row membership is inherited only from the hash-bound certificate that
    was produced on the data host.  It is not used by production scoring authorization, and it
    cannot turn a partial/failed certificate into a valid one.
    """
    if integrity.get("source_membership_verified") is not True:
        raise ValueError("stored integrity lacks verified raw-source membership")
    rows = integrity.get("source_membership")
    if not isinstance(rows, list):
        raise ValueError("stored raw-source membership certificate is not a list")
    by_domain = {
        row.get("domain"): row for row in rows
        if isinstance(row, dict) and isinstance(row.get("domain"), str)
    }
    if len(by_domain) != len(rows) or set(by_domain) != set(domains):
        raise ValueError("stored raw-source membership domain panel is incomplete or duplicated")
    packet_by_domain = {row.get("domain"): row for row in packet.get("domains", [])}
    checked = []
    for domain in domains:
        certificate = by_domain[domain]
        packet_domain = packet_by_domain.get(domain)
        if not isinstance(packet_domain, dict):
            raise ValueError(f"packet omits source-bound domain {domain!r}")
        partitions = [
            row for row in packet_domain.get("partitions", [])
            if row.get("id") in required_partitions
        ]
        expected_n = sum(int(row.get("n", 0)) for row in partitions)
        packet_hashes = []
        for partition in partitions:
            item_path = _resolve_declared_path(
                partition.get("items_path", ""), manifest_path=packet_manifest_path)
            if not item_path.is_file():
                raise ValueError(
                    f"stored source certificate cannot resolve {domain!r} item file"
                )
            packet_hashes.extend(
                json.loads(line)["text_sha256"]
                for line in item_path.read_text().splitlines() if line.strip()
            )
        item_set_sha256 = hashlib.sha256(
            "\n".join(sorted(packet_hashes)).encode()).hexdigest()
        expected_projection = packet_domain.get("source_io_projection", {})
        checks = {
            "valid": certificate.get("valid") is True,
            "errors_empty": certificate.get("errors") == [],
            "dataset_path": certificate.get("dataset_path") == packet_domain.get("dataset_path"),
            "dataset_sha256": certificate.get("dataset_sha256") == packet_domain.get("dataset_sha256"),
            "n_source_rows": certificate.get("n_source_rows") == packet_domain.get("n_dataset_rows"),
            "n_packet_items": certificate.get("n_packet_items") == expected_n,
            "n_matched_items": certificate.get("n_matched_items") == expected_n,
            "packet_item_set_sha256": certificate.get("packet_item_set_sha256") == item_set_sha256,
            "projected_columns": certificate.get("projected_columns") == expected_projection.get("loaded_columns"),
            "projection_grade": certificate.get("projection_grade") == expected_projection.get("projection_grade"),
            "declared_outcome_column": certificate.get("declared_outcome_column") == expected_projection.get("declared_outcome_column"),
            "outcome_column_retained": certificate.get("outcome_column_retained") is False,
            "source_group_identity_recomputed": certificate.get(
                "source_group_identity_recomputed") is True,
            "canonical_first_occurrence_checked": certificate.get(
                "canonical_first_occurrence_checked") is True,
        }
        failed = [key for key, valid in checks.items() if not valid]
        if failed:
            raise ValueError(
                f"stored raw-source membership certificate failed for {domain!r}: {failed}"
            )
        checked.append({
            "domain": domain,
            "dataset_sha256": certificate["dataset_sha256"],
            "n_packet_items": expected_n,
            "packet_item_set_sha256": item_set_sha256,
        })
    canonical = json.dumps(checked, sort_keys=True, separators=(",", ":"))
    return {
        "valid": True,
        "n_domains": len(checked),
        "n_items": sum(row["n_packet_items"] for row in checked),
        "checked_rows_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def compile_breadth_execution_manifest(
        *, stage: str, metric_panel_path: str | Path, arm_bank_path: str | Path,
        protocol_manifest_path: str | Path, packet_manifest_path: str | Path,
        partition_integrity_path: str | Path,
        readout_manifest_path: str | Path = BREADTH_READOUT_MANIFEST,
        model_template_path: str | Path = SAME_VERSION_MODEL_TEMPLATE,
        selection_artifact_path: str | Path | None = None,
        source_validation_mode: str = "recompute") -> dict:
    """Compile the acyclic search or validation manifest for the integrated breadth run."""
    if stage not in {"search", "validation"}:
        raise ValueError("breadth execution stage must be 'search' or 'validation'")
    if source_validation_mode not in {"recompute", "authenticated-certificate"}:
        raise ValueError(
            "source validation mode must be 'recompute' or 'authenticated-certificate'"
        )
    paths = {
        "metric_panel": Path(metric_panel_path),
        "arm_bank": Path(arm_bank_path),
        "protocol": Path(protocol_manifest_path),
        "packet": Path(packet_manifest_path),
        "integrity": Path(partition_integrity_path),
        "readout": Path(readout_manifest_path),
        "model_template": Path(model_template_path),
    }
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"breadth manifest input is missing: {path}")
    panel = json.loads(paths["metric_panel"].read_text())
    bank = json.loads(paths["arm_bank"].read_text())
    _assert_full_breadth_closure(panel, bank)

    protocol = json.loads(paths["protocol"].read_text())
    packet = json.loads(paths["packet"].read_text())
    integrity = json.loads(paths["integrity"].read_text())
    domains = list(DEFAULT_BREADTH_TASKS)
    required_partitions = {"tacit_breadth_search", "tacit_breadth_validation"}
    packet_domains = {row.get("domain") for row in packet.get("domains", [])}
    if set(domains) != packet_domains:
        raise ValueError("breadth packet domain panel differs from the metric panel")
    protocol_partitions = {row.get("id") for row in protocol.get("partitions", [])}
    if not required_partitions <= protocol_partitions:
        raise ValueError("breadth protocol omits search or validation")
    if (integrity.get("valid") is not True
            or not required_partitions <= set(integrity.get("validated_partitions", []))
            or integrity.get("packet_manifest_sha256") != sha256_file(paths["packet"])
            or integrity.get("protocol_manifest_sha256") != sha256_file(paths["protocol"])):
        raise ValueError("breadth partition-integrity certificate is invalid or incomplete")
    require_source_membership = (
        protocol.get("allocation_strategy") == BREADTH_FIRST_ALLOCATION_STRATEGY)
    if (require_source_membership
            and integrity.get("source_membership_verified") is not True):
        raise ValueError(
            "breadth partition-integrity certificate lacks raw-source membership validation")
    revalidate_sources = source_validation_mode == "recompute"
    revalidated = validate_packet(
        paths["packet"],
        protocol_path=paths["protocol"],
        domains=set(domains),
        partitions=required_partitions,
        verify_source_membership=bool(
            integrity.get("source_membership_verified") and revalidate_sources),
        verify_dataset_files=revalidate_sources,
    )
    if (revalidated.get("valid") is not True
            or set(revalidated.get("validated_partitions", [])) != required_partitions
            or revalidated.get("n_domains") != len(domains)):
        raise ValueError(
            f"breadth packet fails full compile-time validation: "
            f"{revalidated.get('errors', [])}"
        )
    integrity_keys = [
        "packet_manifest_sha256", "protocol_manifest_sha256", "n_domains", "n_items",
        "n_domain_scoped_source_groups", "validated_partitions", "domains",
    ]
    if require_source_membership and revalidate_sources:
        integrity_keys.extend(("source_membership_verified", "source_membership"))
    for key in integrity_keys:
        if integrity.get(key) != revalidated.get(key):
            raise ValueError(
                f"stored partition-integrity certificate differs from recomputation at {key}"
            )
    if require_source_membership and not revalidate_sources:
        source_certificate = _authenticate_stored_source_membership(
            integrity=integrity,
            packet=packet,
            packet_manifest_path=paths["packet"],
            domains=domains,
            required_partitions=required_partitions,
        )
    else:
        source_certificate = {
            "valid": True,
            "n_domains": len(domains),
            "n_items": revalidated["n_items"],
            "checked_rows_sha256": hashlib.sha256(json.dumps(
                revalidated.get("source_membership", []),
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }
    if protocol.get("emit_practice_targets") is not False:
        raise ValueError("breadth execution cannot bind a practice-target packet")

    readout = json.loads(paths["readout"].read_text())
    readout_template = readout.get("readout_template", "")
    if not readout_template or readout.get("label_support") != ["YES", "NO"]:
        raise ValueError("breadth readout manifest lacks the frozen YES/NO readout")
    template = json.loads(paths["model_template"].read_text())
    if sha256_file(paths["model_template"]) != sha256_file(SAME_VERSION_MODEL_TEMPLATE):
        raise ValueError(
            "breadth model template is not byte-identical to the canonical Llama-3.1 "
            "8B/70B BF16 template"
        )
    jobs = deepcopy(template.get("model_jobs", []))
    required_jobs = {"llama31_70b_name_target", "llama31_8b_executor"}
    if {job.get("id") for job in jobs} != required_jobs:
        raise ValueError("model template is not the exact Llama-3.1 8B/70B pair")
    for job in jobs:
        job["required_repetitions"] = [0] if stage == "search" else [0, 1]
        # A 70B BF16 checkpoint fits on one B200 (validated by the H49 run).  Keep every breadth
        # process TP=1 so the campaign never initializes a cross-device communicator or reaches
        # a prohibited physical GPU through tensor parallelism.
        job["tensor_parallel_size"] = 1

    template_runner = deepcopy(template.get("analysis", {}).get("runner"))
    if not isinstance(template_runner, dict) or not template_runner:
        raise ValueError("model template omits the frozen policy-isomorphism runner")
    expected_runner_jobs = {
        "small_job": "llama31_8b_executor",
        "big_job": "llama31_70b_name_target",
        "target_arm_id": "name",
    }
    for key, value in expected_runner_jobs.items():
        if template_runner.get(key) != value:
            raise ValueError(
                f"model template runner {key} differs from the breadth design: "
                f"{template_runner.get(key)!r} != {value!r}"
            )
    runner = {
        **template_runner,
        "n_boot": 2000 if stage == "search" else 10000,
        "cell_ids": [cell["id"] for cell in panel["cells"]],
        "functional_rho_floor": 0.7,
        "confidence": 0.95,
        "include_controls": True,
        "source_group_inference": True,
        "allow_fake_inputs": False,
    }

    phase = "calibration" if stage == "search" else "lockbox"
    partition = (
        "tacit_breadth_search" if stage == "search" else "tacit_breadth_validation")
    if stage == "search" and selection_artifact_path is not None:
        raise ValueError("search manifest must not bind a future selection")
    selection_binding = None
    if stage == "validation":
        if selection_artifact_path is None:
            raise ValueError("validation manifest requires a frozen selection artifact")
        selection_binding = _artifact_binding(selection_artifact_path)
        selection = json.loads(Path(selection_artifact_path).read_text())
        if (selection.get("schema") != "policy_articulation_selection/v1"
                or selection.get("selected_phase") != phase
                or selection.get("selected_partition") != partition
                or selection.get("arm_bank_sha256") != sha256_file(paths["arm_bank"])
                or selection.get("packet_manifest_sha256") != sha256_file(paths["packet"])):
            raise ValueError("breadth selection is not bound to this validation stage")
        allowed = load_lockbox_selection(
            selection,
            arm_bank_sha256=sha256_file(paths["arm_bank"]),
            packet_manifest_sha256=sha256_file(paths["packet"]),
            expected_phase=phase,
            expected_partition=partition,
            arm_bank=bank,
        )
        if set(allowed) != {cell["id"] for cell in bank["cells"]}:
            raise ValueError("breadth selection does not deeply cover every arm-bank cell")

        # Do not trust a hand-authored selection that merely has plausible hashes and arm IDs.
        # Replay the frozen role policy from its authenticated search report and require exact
        # canonical equality before the validation manifest can exist.
        from methods.codability.experiments.run_policy_isomorphism import (
            build_policy_articulation_selection,
        )

        selection_path = Path(selection_artifact_path)
        required_selection_paths = {
            key: selection.get(key)
            for key in (
                "search_execution_manifest_path", "search_report_path",
                "arm_bank_path", "packet_manifest_path", "metric_panel_path",
            )
        }
        if any(not isinstance(value, str) or not value for value in required_selection_paths.values()):
            raise ValueError("breadth selection omits canonical replay paths")
        resolved_selection_paths = {
            key: _resolve_declared_path(value, manifest_path=selection_path)
            for key, value in required_selection_paths.items()
        }
        additional_rows = selection.get("additional_artifacts")
        if not isinstance(additional_rows, list):
            raise ValueError("breadth selection additional artifacts are invalid")
        additional_paths = tuple(
            _resolve_declared_path(row.get("path", ""), manifest_path=selection_path)
            for row in additional_rows if isinstance(row, dict)
        )
        if len(additional_paths) != len(additional_rows):
            raise ValueError("breadth selection additional artifacts contain invalid rows")
        replayed_selection = build_policy_articulation_selection(
            search_execution_manifest_path=resolved_selection_paths[
                "search_execution_manifest_path"],
            search_report_path=resolved_selection_paths["search_report_path"],
            arm_bank_path=resolved_selection_paths["arm_bank_path"],
            packet_manifest_path=resolved_selection_paths["packet_manifest_path"],
            metric_panel_path=resolved_selection_paths["metric_panel_path"],
            additional_artifact_paths=additional_paths,
            selected_phase=phase,
            selected_partition=partition,
        )
        if replayed_selection != selection:
            raise ValueError(
                "breadth selection differs from canonical frozen-policy replay"
            )

    domain_tasks = {task: task for task in domains}
    max_chars = {
        task: int(cfgmod.apply_task_preset(
            cfgmod.ImplementerConfig(), task).max_text_chars)
        for task in domains
    }
    execution_environment = deepcopy(template["execution_environment"])
    execution_environment["runtime_environment_overrides"] = deepcopy(
        BREADTH_RUNTIME_ENVIRONMENT_OVERRIDES)
    manifest = {
        "schema": "fresh_name_execution_manifest/v2",
        "status": f"frozen-before-tacit-breadth-{stage}-model-outcomes",
        "objective": (
            "within-family Llama-3.1 8B reconstruction of the Llama-3.1 70B "
            "name-only policy over 30 metrics per task at R1/R2/R3"
        ),
        "anchor_policy": (
            "unsupervised model-to-model reconstruction only; no dataset outcome, human label, "
            "community target, compiler, or external ground truth"
        ),
        "domains": domains,
        "domain_tasks": domain_tasks,
        "protocol_manifest_path": _recorded_path(paths["protocol"]),
        "protocol_manifest_sha256": sha256_file(paths["protocol"]),
        "packet_manifest_path": _recorded_path(paths["packet"]),
        "packet_manifest_sha256": sha256_file(paths["packet"]),
        "partition_integrity_path": _recorded_path(paths["integrity"]),
        "partition_integrity_sha256": sha256_file(paths["integrity"]),
        "source_validation_at_freeze": {
            "mode": source_validation_mode,
            "raw_datasets_recomputed_on_compiler_host": revalidate_sources,
            "packet_structure_and_item_files_recomputed": True,
            "partition_integrity_sha256": sha256_file(paths["integrity"]),
            **source_certificate,
        },
        "arm_bank_path": _recorded_path(paths["arm_bank"]),
        "arm_bank_sha256": sha256_file(paths["arm_bank"]),
        "target_prompt_manifest_path": _recorded_path(paths["readout"]),
        "target_prompt_manifest_sha256": sha256_file(paths["readout"]),
        "readout_template_sha256": text_sha256(readout_template),
        "binary_readout": "teacher_forced_declared_labels",
        "label_support": ["YES", "NO"],
        # Freeze the real-GPU-validated scalar-equivalent schedule.  The optional scorer batching
        # implementation remains available, but eight-row batching failed the pre-outcome 1e-6
        # invariance gate and therefore does not enter this confirmation.
        "teacher_forced_row_batch_size": BREADTH_TEACHER_FORCED_ROW_BATCH_SIZE,
        "teacher_forced_batching_audit": {
            "status": "eight-row-batching-rejected-before-model-outcomes",
            "production_row_batch_size": BREADTH_TEACHER_FORCED_ROW_BATCH_SIZE,
            "invariance_absolute_tolerance": 1e-6,
            "harness_sha256": (
                "d4c0483df60ace1457903a764c84c8706bcf56383163c2c3e004d873d1e2052b"
            ),
            "ordered_rendered_prompt_sha256": (
                "86ee49341f6052579b09a36659aa66130336a31eb362ae14dc30b6c9b9c35053"
            ),
            "model_snapshot_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
            "environment": {
                "gpu": "NVIDIA B200",
                "driver": "580.82.07",
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "vllm": "0.17.0",
                "torch": "2.10.0+cu128",
            },
            "n_logical_rows": 9,
            "n_items": 8,
            "n_probabilities": 72,
            "scalar_repeat": {
                "n_exact": 72, "max_absolute_delta": 0.0,
                "binary_decision_flips": 0,
            },
            "scalar_vs_explicit_row_batch_one": {
                "n_exact": 72, "max_absolute_delta": 0.0,
                "binary_decision_flips": 0,
            },
            "scalar_vs_row_batch_eight": {
                "n_exact": 71,
                "max_absolute_delta": 3.0607528489601243e-5,
                "mean_absolute_delta": 4.2510456235557283e-7,
                "binary_decision_flips": 0,
            },
            "eager_scalar_vs_row_batch_eight": {
                "n_exact": 69,
                "max_absolute_delta": 0.009208505763685482,
                "mean_absolute_delta": 0.00012848048909831457,
                "binary_decision_flips": 0,
            },
            "decision": (
                "freeze explicit row-batch one because it is bit-identical to scalar; reject "
                "eight-row and eager schedules for the confirmatory readout"
            ),
        },
        "item_text_max_chars_by_task": max_chars,
        "execution_environment": execution_environment,
        "teacher_forced_label_validation": deepcopy(
            template["teacher_forced_label_validation"]),
        "phases": {phase: [partition]},
        "phase_access": {
            phase: (
                "open_development" if stage == "search" else "sealed_confirmation"
            ),
        },
        "selection_required_phases": [] if stage == "search" else [phase],
        # The paths and release rule are frozen in the search manifest before any model outcome.
        # The later validation manifest is required to retain this block byte-for-byte; its
        # release artifact binds the authenticated search report, frozen selection, and validation
        # manifest before the held-out item partition can be scored.
        "lockbox_release": {
            "required": True,
            "schema": "policy_isomorphism_calibration_release/v1",
            "artifact_path": str(BREADTH_LOCKBOX_RELEASE),
            "calibration_report_path": str(BREADTH_CALIBRATION_REPORT),
            "calibration_report_schema": "policy_isomorphism_experiment/v5",
            "calibration_partition": "tacit_breadth_search",
            "lockbox_partition": "tacit_breadth_validation",
            "rule": (
                "the held-out breadth partition remains inaccessible until the exact frozen "
                "search report, canonical selection, and validation manifest jointly emit and "
                "pass the production-only release gate"
            ),
        },
        "model_family": "Meta Llama 3.1 Instruct",
        "model_jobs": jobs,
        "execution_sharding": {
            "axis": "domain",
            "permitted_subset": "any nonempty subset of the 11 frozen domains",
            "output_identity": "model_job x phase x repetition x domain",
            "completion_rule": (
                "a stage/repetition is complete only when one authenticated score artifact exists "
                "for every declared domain; disjoint domain subsets may run in separate processes"
            ),
            "scientific_effect": (
                "none; domain sharding changes scheduling only and never item, arm, model, or "
                "readout identity"
            ),
        },
        "resource_policy": {
            **deepcopy(template["resource_policy"]),
            "maximum_gpus_for_any_job": 1,
            "maximum_total_gpus": 1,
            "permitted_physical_gpu_indices": [0],
            "forbidden_physical_gpu_indices": [1, 2, 3, 4],
            "launch_condition": (
                "launch only on a free physical GPU 0; never fall back to another device"
            ),
            "launch_status": (
                "authorized on physical GPU 0 only; GPUs 1 through 4 are explicitly prohibited"
            ),
        },
        "analysis": {
            "functional_rho_floor": 0.7,
            "confidence": 0.95,
            "n_boot": 2000 if stage == "search" else 10000,
            "include_controls": True,
            "source_group_inference": True,
            "cell_ids": [cell["id"] for cell in panel["cells"]],
            # The CLI invocation is authenticated field-for-field against this nested block by
            # run_policy_isomorphism.  Flat descriptive fields above are not an execution lock.
            "runner": runner,
            "strata": ["R1", "R2", "R3"],
            "prevalence_denominator": "30 frozen metrics per task and hierarchy level",
            "primary_grade": (
                "content-specific direct and fixed-target functional substitution at adverse-form "
                "and quotient Spearman >= 0.70"
            ),
            "fiber_grade": (
                "multiple surface- and component-distinct articulations independently pass the "
                "primary grade and their mutual quotient-Spearman gate"
            ),
        },
        "selection_policy": deepcopy(BREADTH_SELECTION_POLICY),
        "implementation": {
            "scoring": _implementation_binding(
                BREADTH_SCORING_IMPLEMENTATION,
                semantics="integrated multi-task teacher-forced scoring and immutable sharding",
            ),
            "analysis": _implementation_binding(
                BREADTH_ANALYSIS_IMPLEMENTATION,
                semantics=(
                    "complete fixed-target/direct-endpoint certification, paired item inference, "
                    "ordinal and quotient-vector equal-but-different fibers, and common-target "
                    "ladder implementation"
                ),
            ),
            "compilation": _implementation_binding(
                BREADTH_COMPILATION_IMPLEMENTATION,
                semantics="source-bound hierarchy sampling and full-text/address decomposition",
            ),
        },
        "additional_artifacts": [
            {"role": "metric_panel", **_artifact_binding(paths["metric_panel"])},
            {"role": "model_environment_template",
             **_artifact_binding(paths["model_template"])},
        ],
    }
    if selection_binding is not None:
        manifest["selection_artifact_path"] = selection_binding["path"]
        manifest["selection_artifact_sha256"] = selection_binding["sha256"]
        provenance = validate_policy_articulation_selection_provenance(
            selection,
            selection_path=selection_artifact_path,
            execution_manifest=manifest,
            # Validation paths are repo-relative and already present before the output manifest
            # is written; the selection location supplies a stable resolution base at freeze time.
            execution_manifest_path=selection_artifact_path,
        )
        manifest["selection_provenance_validation_at_freeze"] = {
            "valid": provenance["valid"],
            "search_execution_manifest_sha256": provenance[
                "search_execution_manifest_sha256"],
            "search_report_sha256": provenance["search_report_sha256"],
            "n_cells": provenance["n_cells"],
        }
    return manifest


def compile_concluding_confirmation_manifest(
        *, template_manifest_path: str | Path, construct_panel_path: str | Path,
        arm_bank_path: str | Path, target_manifest_path: str | Path,
        selection_artifact_path: str | Path) -> dict:
    """Bind a prior-selected multi-construct existence batch to the hardened breadth runtime."""
    paths = {
        "template": Path(template_manifest_path),
        "panel": Path(construct_panel_path),
        "arm_bank": Path(arm_bank_path),
        "target": Path(target_manifest_path),
        "selection": Path(selection_artifact_path),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [label for label, path in paths.items() if not path.is_file()]
        raise ValueError(f"concluding confirmation inputs are missing: {missing}")
    template = json.loads(paths["template"].read_text())
    panel = json.loads(paths["panel"].read_text())
    bank = json.loads(paths["arm_bank"].read_text())
    target = json.loads(paths["target"].read_text())
    selection = json.loads(paths["selection"].read_text())
    if template.get("schema") != "fresh_name_execution_manifest/v2":
        raise ValueError("concluding confirmation template is not a frozen v2 manifest")
    if panel.get("schema") != "legacy_construct_panel/v1" or validate_bank(bank):
        raise ValueError("concluding construct panel or arm bank is invalid")
    panel_ids = [f"N_{row['domain']}_{row['gi']}" for row in panel.get("cells", [])]
    bank_ids = [cell.get("id") for cell in bank.get("cells", [])]
    target_ids = [cell.get("id") for cell in target.get("cells", [])]
    if not panel_ids or panel_ids != bank_ids or panel_ids != target_ids:
        raise ValueError("concluding panel, arm bank, and target cells differ")
    if any(
            (panel_row.get("construct"), panel_row.get("domain"), panel_row.get("task"))
            != (bank_row.get("construct"), bank_row.get("domain"), bank_row.get("task"))
            or (panel_row.get("construct"), panel_row.get("domain"), panel_row.get("task"))
            != (target_row.get("construct"), target_row.get("domain"), target_row.get("task"))
            for panel_row, bank_row, target_row in zip(
                panel["cells"], bank["cells"], target["cells"])
    ):
        raise ValueError("concluding construct identity changes across frozen artifacts")
    packet_sha = template.get("packet_manifest_sha256")
    if selection.get("arm_bank_sha256") != sha256_file(paths["arm_bank"]):
        raise ValueError("concluding selection changes the arm bank")
    if selection.get("packet_manifest_sha256") != packet_sha:
        raise ValueError("concluding selection changes the item packet")
    allowed = load_lockbox_selection(
        selection,
        arm_bank_sha256=sha256_file(paths["arm_bank"]),
        packet_manifest_sha256=packet_sha,
        expected_partition="tacit_breadth_validation",
        arm_bank=bank,
    )
    if list(allowed) != panel_ids:
        raise ValueError("concluding selection does not cover the exact construct panel")
    domains = list(dict.fromkeys(row["domain"] for row in panel["cells"]))
    domain_tasks = {row["domain"]: row["task"] for row in panel["cells"]}
    readout_template = target.get("readout_template")
    if (not isinstance(readout_template, str) or not readout_template
            or target.get("label_support") != ["YES", "NO"]):
        raise ValueError("concluding target manifest changes the teacher-forced readout")

    manifest = deepcopy(template)
    manifest.update({
        "status": "frozen-before-concluding-policy-calibration-or-lockbox-outcomes",
        "objective": (
            "prior-selected three-construct existence confirmation of same-version Llama-3.1 "
            "8B reconstruction of the Llama-3.1 70B name-only policy"
        ),
        "anchor_policy": (
            "unsupervised model-to-model reconstruction only; no dataset outcome, human label, "
            "community target, compiler, or external ground truth"
        ),
        "domains": domains,
        "domain_tasks": domain_tasks,
        "arm_bank_path": _recorded_path(paths["arm_bank"]),
        "arm_bank_sha256": sha256_file(paths["arm_bank"]),
        "target_prompt_manifest_path": _recorded_path(paths["target"]),
        "target_prompt_manifest_sha256": sha256_file(paths["target"]),
        "readout_template_sha256": text_sha256(readout_template),
        "selection_artifact_path": _recorded_path(paths["selection"]),
        "selection_artifact_sha256": sha256_file(paths["selection"]),
        "phases": {
            "calibration": ["tacit_breadth_search"],
            "lockbox": ["tacit_breadth_validation"],
        },
        "phase_access": {
            "calibration": "open_development",
            "lockbox": "sealed_confirmation",
        },
        "selection_required_phases": ["calibration", "lockbox"],
        "item_text_max_chars_by_task": {
            task: int(cfgmod.apply_task_preset(
                cfgmod.ImplementerConfig(), task).max_text_chars)
            for task in dict.fromkeys(domain_tasks.values())
        },
        "lockbox_release": {
            "required": True,
            "schema": "policy_isomorphism_calibration_release/v1",
            "artifact_path": (
                "notebooks/data/two_faces_20260702/concluding_policy_confirmation_v2/"
                "calibration_release.json"
            ),
            "calibration_report_path": (
                "notebooks/data/two_faces_20260702/concluding_policy_confirmation_v2/"
                "calibration_report.json"
            ),
            "calibration_report_schema": "policy_isomorphism_experiment/v5",
            "calibration_partition": "tacit_breadth_search",
            "lockbox_partition": "tacit_breadth_validation",
            "rule": (
                "the validation packet remains inaccessible until the exact production-only "
                "calibration report emits and passes the hash-bound release gate"
            ),
        },
        "execution_sharding": {
            "axis": "domain",
            "permitted_subset": "either or both frozen domains",
            "output_identity": "model_job x phase x repetition x domain",
            "completion_rule": "both domains must be present before analysis",
            "scientific_effect": "none; domain sharding changes scheduling only",
        },
        "resource_policy": {
            "execution": "sequential one-model-per-process",
            "maximum_gpus_for_any_job": 1,
            "maximum_total_gpus": 4,
            "permitted_physical_gpu_indices": [5, 6, 7],
            "forbidden_physical_gpu_indices": [0, 1, 2, 3, 4],
            "launch_condition": "launch on one free permitted GPU after busy-process preflight",
            "launch_status": "authorized by the user for this independent concluding batch",
        },
        "additional_artifacts": [{
            **_artifact_binding(paths["panel"]),
            "role": "prior_selected_construct_panel",
        }],
        "implementation": {
            "scoring": _implementation_binding(
                CONCLUDING_SCORING_IMPLEMENTATION,
                semantics=(
                    "multi-construct scoring, partition authorization, teacher-forced readout, "
                    "single-GPU launch safety, and immutable sharding"
                ),
            ),
            "analysis": _implementation_binding(
                BREADTH_ANALYSIS_IMPLEMENTATION,
                semantics=(
                    "multi-cell fixed-target/direct-endpoint certification and per-construct "
                    "matched-control inference"
                ),
            ),
            "compilation": _implementation_binding(
                BREADTH_COMPILATION_IMPLEMENTATION,
                semantics="source-arm, exact-control, construct-panel, and manifest compilation",
            ),
        },
    })
    for job in manifest["model_jobs"]:
        job["required_repetitions"] = [0, 1]
        job["tensor_parallel_size"] = 1
    runner = manifest["analysis"]["runner"]
    runner.update({
        "n_boot": 10000,
        "cell_ids": panel_ids,
        "functional_rho_floor": 0.7,
        "confidence": 0.95,
        "include_controls": True,
        "source_group_inference": True,
        "allow_fake_inputs": False,
    })
    manifest["analysis"] = {
        "functional_rho_floor": 0.7,
        "confidence": 0.95,
        "n_boot": 10000,
        "include_controls": True,
        "source_group_inference": True,
        "cell_ids": panel_ids,
        "runner": runner,
        "claim_scope": "prior-selected existence batch; not prevalence",
        "multiplicity": (
            "one six-member Bonferroni union family per construct; construct families and their "
            "selection-biased X-of-3 count remain separate"
        ),
        "primary_grade": (
            "content-specific direct and fixed-target functional substitution at adverse-form "
            "and quotient Spearman >= 0.70"
        ),
    }
    manifest.pop("selection_policy", None)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=(
        "legacy-bank", "metric-panel", "breadth-bank",
        "breadth-search-manifest", "breadth-validation-manifest",
        "concluding-confirmation-manifest"),
                        default="legacy-bank")
    parser.add_argument("--target-manifest", default=None)
    parser.add_argument(
        "--legacy-panel",
        default=None,
        help=(
            "optional legacy_construct_panel/v1 JSON declaring source files, exact cells, "
            "canonical tasks, and target jobs for --mode legacy-bank"
        ),
    )
    parser.add_argument("--metric-panel", default=None,
                        help="frozen panel JSON for breadth bank/manifest modes")
    parser.add_argument("--arm-bank", default=None,
                        help="frozen breadth bank JSON for manifest modes")
    parser.add_argument("--protocol-manifest", default=None)
    parser.add_argument("--packet-manifest", default=None)
    parser.add_argument("--partition-integrity", default=None)
    parser.add_argument("--selection-artifact", default=None)
    parser.add_argument(
        "--source-validation-mode",
        choices=("recompute", "authenticated-certificate"),
        default="recompute",
        help=(
            "recompute raw-source membership, or authenticate the packet's already-frozen "
            "source-membership certificate when compiling off the data host"
        ),
    )
    parser.add_argument("--readout-manifest", default=str(BREADTH_READOUT_MANIFEST))
    parser.add_argument("--model-template", default=str(SAME_VERSION_MODEL_TEMPLATE))
    parser.add_argument("--template-manifest", default=None,
                        help="frozen v2 structural template for concluding confirmation")
    parser.add_argument("--tasks", default=",".join(DEFAULT_BREADTH_TASKS),
                        help="comma-separated canonical tasks for --mode metric-panel")
    parser.add_argument("--n-per-task-level", type=int, default=30)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "metric-panel":
        tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
        panel = compile_metric_panel(tasks=tasks, n_per_task_level=args.n_per_task_level)
        out.write_text(json.dumps(panel, indent=1))
        print(json.dumps({
            "out": str(out),
            "sha256": sha256_file(out),
            "panel_content_sha256": panel["panel_content_sha256"],
            "n_cells": panel["n_cells"],
            "n_tasks": len(panel["tasks"]),
            "levels": panel["levels"],
        }, indent=1))
        return
    if args.mode == "breadth-bank":
        if not args.metric_panel:
            parser.error("--mode breadth-bank requires --metric-panel")
        panel = json.loads(Path(args.metric_panel).read_text())
        bank = compile_breadth_bank(panel=panel)
        errors = validate_bank(bank)
        if errors:
            raise ValueError(errors)
        out.write_text(json.dumps(bank, indent=1))
        print(json.dumps({
            "out": str(out),
            "sha256": sha256_file(out),
            "bank_content_sha256": bank["bank_content_sha256"],
            "n_cells": len(bank["cells"]),
            "n_arms": sum(len(cell["arms"]) for cell in bank["cells"]),
        }, indent=1))
        return
    if args.mode in {"breadth-search-manifest", "breadth-validation-manifest"}:
        required = {
            "--metric-panel": args.metric_panel,
            "--arm-bank": args.arm_bank,
            "--protocol-manifest": args.protocol_manifest,
            "--packet-manifest": args.packet_manifest,
            "--partition-integrity": args.partition_integrity,
        }
        missing = [flag for flag, value in required.items() if not value]
        if missing:
            parser.error(f"{args.mode} requires {', '.join(missing)}")
        stage = "search" if args.mode == "breadth-search-manifest" else "validation"
        manifest = compile_breadth_execution_manifest(
            stage=stage,
            metric_panel_path=args.metric_panel,
            arm_bank_path=args.arm_bank,
            protocol_manifest_path=args.protocol_manifest,
            packet_manifest_path=args.packet_manifest,
            partition_integrity_path=args.partition_integrity,
            readout_manifest_path=args.readout_manifest,
            model_template_path=args.model_template,
            selection_artifact_path=args.selection_artifact,
            source_validation_mode=args.source_validation_mode,
        )
        out.write_text(json.dumps(manifest, indent=1))
        print(json.dumps({
            "out": str(out),
            "sha256": sha256_file(out),
            "stage": stage,
            "n_cells": len(manifest["analysis"]["cell_ids"]),
            "selection_artifact_sha256": manifest.get("selection_artifact_sha256"),
        }, indent=1))
        return
    if args.mode == "concluding-confirmation-manifest":
        required = {
            "--template-manifest": args.template_manifest,
            "--legacy-panel": args.legacy_panel,
            "--arm-bank": args.arm_bank,
            "--target-manifest": args.target_manifest,
            "--selection-artifact": args.selection_artifact,
        }
        missing = [flag for flag, value in required.items() if not value]
        if missing:
            parser.error(f"{args.mode} requires {', '.join(missing)}")
        manifest = compile_concluding_confirmation_manifest(
            template_manifest_path=args.template_manifest,
            construct_panel_path=args.legacy_panel,
            arm_bank_path=args.arm_bank,
            target_manifest_path=args.target_manifest,
            selection_artifact_path=args.selection_artifact,
        )
        out.write_text(json.dumps(manifest, indent=1))
        print(json.dumps({
            "out": str(out),
            "sha256": sha256_file(out),
            "n_cells": len(manifest["analysis"]["cell_ids"]),
            "domains": manifest["domains"],
        }, indent=1))
        return

    legacy_kwargs = {}
    if args.legacy_panel:
        panel_path = Path(args.legacy_panel)
        panel = json.loads(panel_path.read_text())
        if panel.get("schema") != "legacy_construct_panel/v1":
            raise ValueError("legacy panel must use legacy_construct_panel/v1")
        source_rows = panel.get("source_files")
        cell_rows = panel.get("cells")
        if not isinstance(source_rows, dict) or not isinstance(cell_rows, list) or not cell_rows:
            raise ValueError("legacy panel requires source_files and a nonempty cells list")
        source_files = {domain: Path(path) for domain, path in source_rows.items()}
        cell_targets = {}
        domain_tasks = {}
        for row in cell_rows:
            domain = row.get("domain")
            gi = row.get("gi")
            task = row.get("task")
            target_jobs = row.get("target_model_jobs")
            if (domain not in source_files or not isinstance(gi, int)
                    or not isinstance(task, str) or not task
                    or not isinstance(target_jobs, list) or not target_jobs
                    or any(not isinstance(job, str) or not job for job in target_jobs)):
                raise ValueError(f"invalid legacy panel cell: {row}")
            key = (domain, gi)
            if key in cell_targets:
                raise ValueError(f"duplicate legacy panel cell: {key}")
            if domain in domain_tasks and domain_tasks[domain] != task:
                raise ValueError(f"legacy panel changes canonical task for {domain}")
            cell_targets[key] = target_jobs
            domain_tasks[domain] = task
        legacy_kwargs = {
            "source_files": source_files,
            "cell_targets": cell_targets,
            "domain_tasks": domain_tasks,
        }
    bank = compile_bank(target_manifest_path=args.target_manifest, **legacy_kwargs)
    errors = validate_bank(bank)
    if errors:
        raise ValueError(errors)
    out.write_text(json.dumps(bank, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "bank_content_sha256": bank["bank_content_sha256"],
                      "n_cells": len(bank["cells"]),
                      "n_arms": sum(len(cell["arms"]) for cell in bank["cells"])}, indent=1))


if __name__ == "__main__":
    main()
