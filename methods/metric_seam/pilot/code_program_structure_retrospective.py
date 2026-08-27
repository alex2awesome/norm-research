"""Measure common source-structure descriptors for the active-code programs.

The active coding lane calls its manually engineered static/AST checkers
``deep`` and its prompt-generated executable programs ``shallow``.  This
retrospective asks whether those authoring labels correspond to measurable
source structure.  It never executes a metric, reads an item outcome, or turns
syntax into a construct-fidelity claim.

Descriptors are computed with Python's AST over the exact entry-module source
receipts used to produce ``code_scores.json``.  A conservative scope-qualified
local call graph is condensed into a DAG before longest-path calculation, so
recursion is recorded rather than silently breaking the depth calculation.
Shared library internals are intentionally outside the measurement.  These are
syntactic entry-module descriptors, not semantic relation depth or dynamic
contributing-path depth.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

from methods.metric_seam.battery.certify_batch_v2 import spearman


SCHEMA = "metric-seam.code-program-structure-retrospective.v1"
ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
DEPTH_RESULT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "code_depth_full_panel_retrospective_002/results.json"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "code_program_structure_retrospective_001"
)

CONTROL_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match)
LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)


class StructureError(RuntimeError):
    """Raised when the frozen source panel cannot be measured faithfully."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_ast_depth(tree: ast.AST) -> int:
    def visit(node: ast.AST, depth: int) -> int:
        children = list(ast.iter_child_nodes(node))
        return depth if not children else max(visit(child, depth + 1) for child in children)

    return visit(tree, 0)


def _max_control_nesting(tree: ast.AST) -> int:
    def visit(node: ast.AST, depth: int) -> int:
        next_depth = depth + int(isinstance(node, CONTROL_NODES))
        return max(
            [next_depth]
            + [visit(child, next_depth) for child in ast.iter_child_nodes(node)]
        )

    return visit(tree, 0)


class _DirectCallCollector(ast.NodeVisitor):
    """Collect calls in one body without attributing nested definitions."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - AST visitor API
        if isinstance(node.func, ast.Name):
            self.names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.names.add(node.func.attr)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return


    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        return


@dataclass(frozen=True)
class _FunctionRecord:
    qualified_name: str
    parent_scope: str
    node: ast.FunctionDef | ast.AsyncFunctionDef


def _function_records(tree: ast.AST) -> list[_FunctionRecord]:
    records: list[_FunctionRecord] = []

    def descend(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                descend(child, (*scope, child.name))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*scope, child.name))
                records.append(
                    _FunctionRecord(
                        qualified_name=qualified,
                        parent_scope=".".join(scope),
                        node=child,
                    )
                )
                descend(child, (*scope, child.name))
            else:
                descend(child, scope)

    descend(tree, ())
    return records


def _local_call_graph(tree: ast.AST) -> dict[str, set[str]]:
    records = _function_records(tree)
    qualified = {record.qualified_name for record in records}
    by_basename: dict[str, list[str]] = {}
    for record in records:
        by_basename.setdefault(record.node.name, []).append(record.qualified_name)
    graph: dict[str, set[str]] = {record.qualified_name: set() for record in records}
    for record in records:
        collector = _DirectCallCollector()
        for statement in record.node.body:
            collector.visit(statement)
        for called_name in collector.names:
            candidates = by_basename.get(called_name, [])
            resolved: str | None = None
            # Prefer a nested function, then a lexical sibling/ancestor, then a
            # unique global basename. Ambiguous attribute/name matches are omitted.
            lexical = record.qualified_name.split(".")
            for end in range(len(lexical), -1, -1):
                candidate = ".".join((*lexical[:end], called_name))
                if candidate in qualified:
                    resolved = candidate
                    break
            if resolved is None and len(candidates) == 1:
                resolved = candidates[0]
            if resolved is not None:
                graph[record.qualified_name].add(resolved)
    return graph


def _strongly_connected_components(
    graph: Mapping[str, set[str]],
) -> list[tuple[str, ...]]:
    """Tarjan SCCs in deterministic node/edge order."""

    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def connect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for child in sorted(graph[node]):
            if child not in indices:
                connect(child)
                lowlink[node] = min(lowlink[node], lowlink[child])
            elif child in on_stack:
                lowlink[node] = min(lowlink[node], indices[child])
        if lowlink[node] == indices[node]:
            members: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                members.append(member)
                if member == node:
                    break
            components.append(tuple(sorted(members)))

    for node in sorted(graph):
        if node not in indices:
            connect(node)
    return components


def _call_graph_descriptors(graph: Mapping[str, set[str]]) -> dict[str, int]:
    if not graph:
        return {
            "local_call_graph_nodes": 0,
            "local_call_graph_edges": 0,
            "local_call_graph_sccs": 0,
            "recursive_components": 0,
            "condensed_longest_path_edges": 0,
        }
    components = _strongly_connected_components(graph)
    component_of = {
        member: index for index, component in enumerate(components) for member in component
    }
    dag: dict[int, set[int]] = {index: set() for index in range(len(components))}
    for parent, children in graph.items():
        for child in children:
            left = component_of[parent]
            right = component_of[child]
            if left != right:
                dag[left].add(right)

    cache: dict[int, int] = {}

    def longest(node: int) -> int:
        if node not in cache:
            cache[node] = 0 if not dag[node] else 1 + max(longest(child) for child in dag[node])
        return cache[node]

    recursive = sum(
        len(component) > 1 or component[0] in graph[component[0]]
        for component in components
    )
    return {
        "local_call_graph_nodes": len(graph),
        "local_call_graph_edges": sum(len(children) for children in graph.values()),
        "local_call_graph_sccs": len(components),
        "recursive_components": recursive,
        "condensed_longest_path_edges": max(longest(node) for node in dag),
    }


def source_descriptors(source: str) -> dict[str, Any]:
    """Return language-level descriptors for one parseable Python source."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise StructureError(f"Python source does not parse: {exc}") from exc
    nodes = list(ast.walk(tree))
    imports: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    graph = _local_call_graph(tree)
    lines = source.splitlines()
    return {
        "physical_lines": len(lines),
        "nonblank_noncomment_lines": sum(
            bool(line.strip()) and not line.lstrip().startswith("#") for line in lines
        ),
        "ast_nodes": len(nodes),
        "ast_max_depth": _max_ast_depth(tree),
        "statement_nodes": sum(isinstance(node, ast.stmt) for node in nodes),
        "expression_nodes": sum(isinstance(node, ast.expr) for node in nodes),
        "function_defs": sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in nodes
        ),
        "class_defs": sum(isinstance(node, ast.ClassDef) for node in nodes),
        "call_nodes": sum(isinstance(node, ast.Call) for node in nodes),
        "control_nodes": sum(isinstance(node, CONTROL_NODES) for node in nodes),
        "loop_nodes": sum(isinstance(node, LOOP_NODES) for node in nodes),
        "max_control_nesting": _max_control_nesting(tree),
        "imported_top_level_modules": sorted(imports),
        "imported_top_level_module_count": len(imports),
        **_call_graph_descriptors(graph),
    }


def _median(values: Iterable[float]) -> float:
    return round(float(statistics.median(values)), 3)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate(
    *,
    task_dir: Path = TASK_DIR,
    depth_result: Path = DEPTH_RESULT,
) -> dict[str, Any]:
    manifest_path = task_dir / "code_scores_cpu_manifest.json"
    manifest = _read_json(manifest_path)
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise StructureError("code-score manifest has no source mapping")
    depth = _read_json(depth_result)
    depth_inputs = depth.get("inputs") or {}
    code_score_receipt = depth_inputs.get("code_scores") or {}
    item_receipt = depth_inputs.get("items") or {}
    if manifest.get("output_sha256") != code_score_receipt.get("sha256"):
        raise StructureError("code-score manifest is not bound to the depth result")
    if manifest.get("input_items_sha256") != item_receipt.get("sha256"):
        raise StructureError("item manifest is not bound to the depth result")
    code_scores_path = task_dir / "code_scores.json"
    items_path = task_dir / "items.json"
    if not code_scores_path.is_file() or _sha256(code_scores_path) != manifest.get(
        "output_sha256"
    ):
        raise StructureError("current code scores differ from the bound score manifest")
    if not items_path.is_file() or _sha256(items_path) != manifest.get(
        "input_items_sha256"
    ):
        raise StructureError("current items differ from the bound score manifest")
    criteria = depth.get("criteria")
    if not isinstance(criteria, list) or len(criteria) != 18:
        raise StructureError("active-code depth result must contain all 18 criteria")

    programs: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    for criterion in criteria:
        criterion_id = criterion["criterion_id"]
        selected = (criterion.get("train_shallow_selection") or {}).get("selected")
        names = [("deep", criterion["deep_program"])]
        if selected:
            names.append(("shallow", selected))
        for arm, program_name in names:
            receipt = sources.get(program_name)
            if not isinstance(receipt, Mapping):
                raise StructureError(f"no frozen source receipt for {program_name}")
            path = ROOT / str(receipt.get("path"))
            if not path.is_file():
                raise StructureError(f"source is missing: {path}")
            observed_sha = _sha256(path)
            if observed_sha != receipt.get("sha256"):
                raise StructureError(f"source changed since score execution: {program_name}")
            programs[program_name] = {
                "criterion_id": criterion_id,
                "arm": arm,
                "program": program_name,
                "source_path": path.relative_to(ROOT).as_posix(),
                "descriptors": source_descriptors(path.read_text(encoding="utf-8")),
            }
        if selected:
            deep = programs[criterion["deep_program"]]["descriptors"]
            shallow = programs[selected]["descriptors"]
            comparison = criterion.get("heldout_comparison") or {}
            pairs.append(
                {
                    "criterion_id": criterion_id,
                    "deep_program": criterion["deep_program"],
                    "shallow_program": selected,
                    "structure_difference": {
                        key: deep[key] - shallow[key]
                        for key in (
                            "physical_lines",
                            "nonblank_noncomment_lines",
                            "ast_nodes",
                            "ast_max_depth",
                            "statement_nodes",
                            "function_defs",
                            "call_nodes",
                            "control_nodes",
                            "max_control_nesting",
                            "condensed_longest_path_edges",
                        )
                    },
                    "reconstruction_delta_spearman": comparison.get("delta_spearman"),
                    "reconstruction_common_n": comparison.get("n_paired"),
                    "comparison_support_eligible": comparison.get(
                        "inferential_eligible"
                    )
                    is True,
                }
            )

    paired_deep = [programs[row["deep_program"]]["descriptors"] for row in pairs]
    paired_shallow = [programs[row["shallow_program"]]["descriptors"] for row in pairs]
    metric_names = (
        "physical_lines",
        "nonblank_noncomment_lines",
        "ast_nodes",
        "ast_max_depth",
        "statement_nodes",
        "function_defs",
        "call_nodes",
        "control_nodes",
        "max_control_nesting",
        "condensed_longest_path_edges",
    )
    paired_summary = {
        key: {
            "deep_median": _median(row[key] for row in paired_deep),
            "shallow_median": _median(row[key] for row in paired_shallow),
            "deep_greater_count": sum(
                deep[key] > shallow[key]
                for deep, shallow in zip(paired_deep, paired_shallow, strict=True)
            ),
            "ties": sum(
                deep[key] == shallow[key]
                for deep, shallow in zip(paired_deep, paired_shallow, strict=True)
            ),
            "pair_n": len(pairs),
        }
        for key in metric_names
    }

    defined_rows = [row for row in pairs if _finite(row["reconstruction_delta_spearman"])]
    association_strata = {
        "all_defined": defined_rows,
        "minimum_common_n_20": [
            row
            for row in defined_rows
            if int(row.get("reconstruction_common_n") or 0) >= 20
        ],
        "comparison_support_eligible": [
            row for row in defined_rows if row["comparison_support_eligible"]
        ],
    }
    association_sensitivity: dict[str, dict[str, Any]] = {}
    for stratum, selected_rows in association_strata.items():
        association_sensitivity[stratum] = {
            "n": len(selected_rows),
            "metrics": {},
        }
        for key in metric_names:
            structure_delta = [
                row["structure_difference"][key] for row in selected_rows
            ]
            signal_delta = [
                float(row["reconstruction_delta_spearman"])
                for row in selected_rows
            ]
            rho = spearman(structure_delta, signal_delta)
            association_sensitivity[stratum]["metrics"][key] = {
                "spearman_structure_delta_vs_reconstruction_delta": (
                    rho if math.isfinite(rho) else None
                ),
                "inference": "descriptive_post_hoc_no_p_value",
            }

    return {
        "schema": SCHEMA,
        "scope": {
            "active_criteria": 18,
            "deep_programs": 18,
            "train_selected_shallow_programs": len(pairs),
            "runtime_channel_both_arms": "code",
            "selection": "retrospective_full_family",
        },
        "descriptor_semantics": {
            "kind": "python_entry_module_source_structure",
            "transitive_dependencies_included": False,
            "call_graph": (
                "conservative scope-qualified lexical graph; ambiguous name matches "
                "omitted; SCC-condensed longest path; not a dynamic execution trace"
            ),
            "guard": (
                "Entry-module syntax excludes shared parser/library internals and is not "
                "semantic relation depth, construct fidelity, reconstruction agreement, "
                "or automatic discovery."
            ),
        },
        "paired_summary": paired_summary,
        "association_sensitivity": association_sensitivity,
        "association_interpretation": (
            "Signs change across support strata; no directional structure-signal "
            "association is supported."
        ),
        "upstream_reconstruction_family": {
            "criteria": depth["summary"]["active_criteria"],
            "comparison_support_eligible": depth["summary"][
                "inferentially_eligible"
            ],
            "multiplicity_controlled_improvements": depth["summary"][
                "multiplicity_controlled_improvements"
            ],
        },
        "programs": [programs[name] for name in sorted(programs)],
        "pairs": pairs,
        "inputs": {
            "code_score_manifest": {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(manifest_path),
            },
            "full_family_depth_result": {
                "path": depth_result.relative_to(ROOT).as_posix(),
                "sha256": _sha256(depth_result),
            },
            "code_scores": dict(code_score_receipt),
            "items": dict(item_receipt),
        },
    }


def render_report(result: Mapping[str, Any]) -> str:
    summary = result["paired_summary"]
    rows = []
    for key in (
        "nonblank_noncomment_lines",
        "ast_nodes",
        "ast_max_depth",
        "function_defs",
        "control_nodes",
        "max_control_nesting",
        "condensed_longest_path_edges",
    ):
        value = summary[key]
        rows.append(
            f"| {key} | {value['deep_median']:.1f} | {value['shallow_median']:.1f} | "
            f"{value['deep_greater_count']}/{value['pair_n']} |"
        )
    table = "\n".join(rows)
    sensitivity_rows = []
    for stratum, record in result["association_sensitivity"].items():
        metrics = record["metrics"]
        sensitivity_rows.append(
            f"| {stratum} | {record['n']} | "
            f"{metrics['ast_nodes']['spearman_structure_delta_vs_reconstruction_delta']:+.3f} | "
            f"{metrics['function_defs']['spearman_structure_delta_vs_reconstruction_delta']:+.3f} | "
            f"{metrics['control_nodes']['spearman_structure_delta_vs_reconstruction_delta']:+.3f} |"
        )
    sensitivity = "\n".join(sensitivity_rows)
    upstream = result["upstream_reconstruction_family"]
    return f"""# Active-code source-structure retrospective

The authoring labels do correspond to materially different program structure on the
15 criteria with a TRAIN-selected shallow executable comparator. Both arms are runtime
code. These entry-module descriptors exclude shared parser and library internals.

| source descriptor | deep median | shallow median | deep > shallow |
|---|---:|---:|---:|
{table}

Post-hoc association signs are subset-sensitive:

| comparison stratum | n | AST-node delta rho | function delta rho | control delta rho |
|---|---:|---:|---:|---:|
{sensitivity}

No directional association is supported: the signs change when the upstream comparison
support gates are applied, and the eligible stratum has only four criteria.

These descriptors establish only that the manually engineered arm is structurally different.
They do not establish semantic relation depth, construct fidelity, or better reconstruction.
The bound full-family reconstruction analysis has
{upstream['multiplicity_controlled_improvements']} BH-FDR improvements across
{upstream['comparison_support_eligible']} support-eligible comparisons, so a syntactically
deeper program must not be reported as a more successful verifier without separate evidence.
"""


def write_result(result: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(render_report(result), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=TASK_DIR)
    parser.add_argument("--depth-result", type=Path, default=DEPTH_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = evaluate(task_dir=args.task_dir, depth_result=args.depth_result)
    if args.check:
        existing = _read_json(args.out_dir / "results.json")
        if existing != result:
            raise StructureError("stored structure result differs from deterministic rerun")
    else:
        write_result(result, args.out_dir)
    print(json.dumps({"scope": result["scope"], "paired_summary": result["paired_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
