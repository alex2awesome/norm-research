#!/usr/bin/env python3
"""Fail-closed, offline contracts for applying similarity adapters to hierarchies.

This module does not load a model, build a graph, or update a canonical
partition.  It validates the immutable records passed between those stages and
materializes a hash-bound candidate manifest at an explicit, new path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from methods.codability.lexicon_distill.dataset import (
    LABELS,
    LEVELS,
    PROTOCOL_IDS,
    TASKS,
)


PAIR_INPUT_SCHEMA = "gemma-similarity-pair-input-v1"
PAIR_OUTPUT_SCHEMA = "gemma-similarity-pair-output-v1"
NODE_INVENTORY_SCHEMA = "gemma-hierarchy-node-inventory-v1"
PARTITION_SCHEMA = "gemma-hierarchy-candidate-partition-v1"
NAMES_SCHEMA = "gemma-hierarchy-candidate-names-v1"
CANDIDATE_MANIFEST_SCHEMA = "gemma-hierarchy-candidate-manifest-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_LABEL_SET = frozenset(LABELS)
_PROTOCOL_LEVEL = {
    PROTOCOL_IDS["R1"]: "R1",
    PROTOCOL_IDS["R2"]: "R2",
    PROTOCOL_IDS["R2_V2"]: "R2",
    PROTOCOL_IDS["R2_V2_1"]: "R2",
    PROTOCOL_IDS["R3"]: "R3",
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], context: str) -> None:
    expected_set = frozenset(expected)
    actual = frozenset(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        raise ValueError(f"{context} keys mismatch: missing={missing}, extra={extra}")


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _sha256(value: object, context: str) -> str:
    token = _nonempty_string(value, context)
    if not _SHA256.fullmatch(token):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return token


def _task_level_protocol(row: Mapping[str, Any], context: str) -> tuple[str, str, str]:
    task = _nonempty_string(row.get("task"), f"{context}.task")
    level = _nonempty_string(row.get("level"), f"{context}.level")
    protocol = _nonempty_string(row.get("protocol_id"), f"{context}.protocol_id")
    if task not in TASKS:
        raise ValueError(f"{context}.task is not canonical: {task}")
    if level not in LEVELS:
        raise ValueError(f"{context}.level is invalid: {level}")
    if protocol not in _PROTOCOL_LEVEL:
        raise ValueError(f"{context}.protocol_id is unknown: {protocol}")
    if _PROTOCOL_LEVEL[protocol] != level:
        raise ValueError(f"{context}: protocol {protocol} is not valid for {level}")
    return task, level, protocol


def _probabilities(value: object, context: str) -> dict[str, float]:
    row = _object(value, context)
    _exact_keys(row, LABELS, context)
    result: dict[str, float] = {}
    for label in LABELS:
        probability = row[label]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"{context}.{label} must be numeric")
        number = float(probability)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{context}.{label} must be finite and in [0, 1]")
        result[label] = number
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{context} must sum to 1")
    return result


def _winning_label(probabilities: Mapping[str, float], context: str) -> str:
    """Return a deterministic, precision-preserving winner.

    BF16 inference can produce exact equal logits.  LABELS is ordered from the
    least to the most merge-permissive relation, so selecting its first maximum
    makes an exact tie conservative instead of making a completed inference run
    unwriteable.
    """
    maximum = max(probabilities.values())
    winners = [label for label in LABELS if probabilities[label] == maximum]
    return winners[0]


def validate_pair_input(value: object, *, context: str = "pair input") -> dict[str, Any]:
    row = dict(_object(value, context))
    _exact_keys(
        row,
        (
            "schema_version", "pair_id", "task", "level", "protocol_id",
            "node_a", "node_b", "text_a", "text_b",
            "source_node_a_sha256", "source_node_b_sha256",
        ),
        context,
    )
    if row["schema_version"] != PAIR_INPUT_SCHEMA:
        raise ValueError(f"{context}.schema_version must be {PAIR_INPUT_SCHEMA}")
    _nonempty_string(row["pair_id"], f"{context}.pair_id")
    _task_level_protocol(row, context)
    node_a = _nonempty_string(row["node_a"], f"{context}.node_a")
    node_b = _nonempty_string(row["node_b"], f"{context}.node_b")
    if node_a == node_b:
        raise ValueError(f"{context} cannot compare a node with itself")
    _nonempty_string(row["text_a"], f"{context}.text_a")
    _nonempty_string(row["text_b"], f"{context}.text_b")
    _sha256(row["source_node_a_sha256"], f"{context}.source_node_a_sha256")
    _sha256(row["source_node_b_sha256"], f"{context}.source_node_b_sha256")
    return row


def pair_input_sha256(value: object) -> str:
    return canonical_json_sha256(validate_pair_input(value))


def _validate_order_view(value: object, context: str) -> dict[str, Any]:
    row = dict(_object(value, context))
    _exact_keys(row, ("prediction", "probabilities"), context)
    probabilities = _probabilities(row["probabilities"], f"{context}.probabilities")
    prediction = _nonempty_string(row["prediction"], f"{context}.prediction")
    if prediction not in _LABEL_SET:
        raise ValueError(f"{context}.prediction is invalid: {prediction}")
    if prediction != _winning_label(probabilities, f"{context}.probabilities"):
        raise ValueError(f"{context}.prediction does not match its probabilities")
    return row


def validate_pair_output(value: object, *, context: str = "pair output") -> dict[str, Any]:
    row = dict(_object(value, context))
    _exact_keys(
        row,
        (
            "schema_version", "pair_id", "task", "level", "protocol_id",
            "input_sha256", "prediction", "probabilities", "order_views",
            "order_consistent", "adapter_sha256", "protocol_sha256",
        ),
        context,
    )
    if row["schema_version"] != PAIR_OUTPUT_SCHEMA:
        raise ValueError(f"{context}.schema_version must be {PAIR_OUTPUT_SCHEMA}")
    _nonempty_string(row["pair_id"], f"{context}.pair_id")
    _task_level_protocol(row, context)
    _sha256(row["input_sha256"], f"{context}.input_sha256")
    _sha256(row["adapter_sha256"], f"{context}.adapter_sha256")
    _sha256(row["protocol_sha256"], f"{context}.protocol_sha256")
    probabilities = _probabilities(row["probabilities"], f"{context}.probabilities")
    prediction = _nonempty_string(row["prediction"], f"{context}.prediction")
    if prediction not in _LABEL_SET or prediction != _winning_label(
        probabilities, f"{context}.probabilities"
    ):
        raise ValueError(f"{context}.prediction does not match its probabilities")
    views = _object(row["order_views"], f"{context}.order_views")
    _exact_keys(views, ("ab", "ba"), f"{context}.order_views")
    ab = _validate_order_view(views["ab"], f"{context}.order_views.ab")
    ba = _validate_order_view(views["ba"], f"{context}.order_views.ba")
    if not isinstance(row["order_consistent"], bool):
        raise ValueError(f"{context}.order_consistent must be boolean")
    expected_consistency = ab["prediction"] == ba["prediction"]
    if row["order_consistent"] != expected_consistency:
        raise ValueError(f"{context}.order_consistent is incorrect")
    averaged = {
        label: (
            float(ab["probabilities"][label]) + float(ba["probabilities"][label])
        ) / 2.0
        for label in LABELS
    }
    if any(
        not math.isclose(averaged[label], probabilities[label], abs_tol=1e-6, rel_tol=0.0)
        for label in LABELS
    ):
        raise ValueError(f"{context}.probabilities are not the mean of order views")
    return row


def _read_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: JSONL file is empty")
    return rows


def validate_pair_files(input_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    inputs = [
        validate_pair_input(row, context=f"{input_path}:{index}")
        for index, row in enumerate(_read_jsonl(input_path), 1)
    ]
    by_id = {row["pair_id"]: row for row in inputs}
    if len(by_id) != len(inputs):
        raise ValueError(f"{input_path}: duplicate pair_id")
    cells = {(row["task"], row["level"], row["protocol_id"]) for row in inputs}
    if len(cells) != 1:
        raise ValueError(
            f"{input_path}: a scoring batch must contain exactly one task/level/protocol cell; "
            f"found {sorted(cells)}"
        )
    node_pairs = {tuple(sorted((row["node_a"], row["node_b"]))) for row in inputs}
    if len(node_pairs) != len(inputs):
        raise ValueError(f"{input_path}: duplicate unordered node pair")
    task, level, protocol_id = next(iter(cells))
    result: dict[str, Any] = {
        "task": task,
        "level": level,
        "protocol_id": protocol_id,
        "n_pairs": len(inputs),
        "inputs_sha256": sha256_file(input_path),
    }
    if output_path is None:
        return result
    outputs = [
        validate_pair_output(row, context=f"{output_path}:{index}")
        for index, row in enumerate(_read_jsonl(output_path), 1)
    ]
    output_by_id = {row["pair_id"]: row for row in outputs}
    if len(output_by_id) != len(outputs):
        raise ValueError(f"{output_path}: duplicate pair_id")
    if set(output_by_id) != set(by_id):
        raise ValueError(f"{output_path}: output pair IDs do not exactly cover input pair IDs")
    adapter_hashes: set[str] = set()
    protocol_hashes: set[str] = set()
    for pair_id, output in output_by_id.items():
        source = by_id[pair_id]
        if (output["task"], output["level"], output["protocol_id"]) != (
            source["task"], source["level"], source["protocol_id"]
        ):
            raise ValueError(f"{output_path}: cell mismatch for {pair_id}")
        if output["input_sha256"] != pair_input_sha256(source):
            raise ValueError(f"{output_path}: input hash mismatch for {pair_id}")
        adapter_hashes.add(output["adapter_sha256"])
        protocol_hashes.add(output["protocol_sha256"])
    if len(adapter_hashes) != 1 or len(protocol_hashes) != 1:
        raise ValueError(f"{output_path}: a scoring batch must use one adapter and one protocol hash")
    result.update(
        outputs_sha256=sha256_file(output_path),
        adapter_sha256=next(iter(adapter_hashes)),
        protocol_sha256=next(iter(protocol_hashes)),
    )
    return result


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    return dict(_object(value, context))


def _artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_node_inventory(value: object) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    row = dict(_object(value, "node inventory"))
    _exact_keys(
        row,
        ("schema_version", "task", "level", "protocol_id", "parent_partition_sha256", "nodes"),
        "node inventory",
    )
    if row["schema_version"] != NODE_INVENTORY_SCHEMA:
        raise ValueError(f"node inventory schema must be {NODE_INVENTORY_SCHEMA}")
    _task_level_protocol(row, "node inventory")
    _sha256(row["parent_partition_sha256"], "node inventory.parent_partition_sha256")
    if not isinstance(row["nodes"], list) or not row["nodes"]:
        raise ValueError("node inventory.nodes must be a nonempty array")
    nodes: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(row["nodes"]):
        node = dict(_object(value, f"node inventory.nodes[{index}]"))
        _exact_keys(node, ("node_id", "text", "source_node_sha256"), f"node inventory.nodes[{index}]")
        node_id = _nonempty_string(node["node_id"], f"node inventory.nodes[{index}].node_id")
        _nonempty_string(node["text"], f"node inventory.nodes[{index}].text")
        _sha256(node["source_node_sha256"], f"node inventory.nodes[{index}].source_node_sha256")
        if node_id in nodes:
            raise ValueError(f"duplicate node inventory ID: {node_id}")
        nodes[node_id] = node
    return row, nodes


def _validate_partition(value: object) -> tuple[dict[str, Any], dict[str, str]]:
    row = dict(_object(value, "candidate partition"))
    _exact_keys(row, ("schema_version", "task", "level", "protocol_id", "partition"), "candidate partition")
    if row["schema_version"] != PARTITION_SCHEMA:
        raise ValueError(f"candidate partition schema must be {PARTITION_SCHEMA}")
    _task_level_protocol(row, "candidate partition")
    raw = _object(row["partition"], "candidate partition.partition")
    if not raw:
        raise ValueError("candidate partition.partition must be nonempty")
    partition: dict[str, str] = {}
    for node_id, group_id in raw.items():
        _nonempty_string(node_id, "candidate partition node ID")
        partition[node_id] = _nonempty_string(group_id, f"candidate partition[{node_id}]")
    return row, partition


def _validate_names(value: object) -> tuple[dict[str, Any], set[str]]:
    row = dict(_object(value, "candidate names"))
    _exact_keys(
        row,
        ("schema_version", "task", "level", "protocol_id", "partition_sha256", "names"),
        "candidate names",
    )
    if row["schema_version"] != NAMES_SCHEMA:
        raise ValueError(f"candidate names schema must be {NAMES_SCHEMA}")
    _task_level_protocol(row, "candidate names")
    _sha256(row["partition_sha256"], "candidate names.partition_sha256")
    raw = _object(row["names"], "candidate names.names")
    if not raw:
        raise ValueError("candidate names.names must be nonempty")
    for group_id, value in raw.items():
        _nonempty_string(group_id, "candidate names group ID")
        name = _object(value, f"candidate names.names[{group_id}]")
        _exact_keys(name, ("name", "gloss"), f"candidate names.names[{group_id}]")
        _nonempty_string(name["name"], f"candidate names.names[{group_id}].name")
        _nonempty_string(name["gloss"], f"candidate names.names[{group_id}].gloss")
    return row, set(raw)


def _validate_config(path: Path, kind: str) -> dict[str, Any]:
    value = _read_json_object(path, f"{kind} config")
    schema = value.get("schema_version")
    config_id = value.get("config_id")
    if schema != f"gemma-hierarchy-{kind}-config-v1":
        raise ValueError(f"{kind} config has invalid schema_version: {schema}")
    _nonempty_string(config_id, f"{kind} config.config_id")
    return value


def build_candidate_manifest(
    *,
    parent_partition_path: Path,
    node_inventory_path: Path,
    pair_inputs_path: Path,
    pair_outputs_path: Path,
    retrieval_config_path: Path,
    scorer_config_path: Path,
    threshold_config_path: Path,
    graph_config_path: Path,
    partition_path: Path,
    names_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate and write one immutable, non-canonical candidate manifest."""
    paths = {
        "parent_partition": parent_partition_path,
        "node_inventory": node_inventory_path,
        "pair_inputs": pair_inputs_path,
        "pair_outputs": pair_outputs_path,
        "retrieval_config": retrieval_config_path,
        "scorer_config": scorer_config_path,
        "threshold_config": threshold_config_path,
        "graph_config": graph_config_path,
        "partition": partition_path,
        "names": names_path,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)

    inventory, nodes = _validate_node_inventory(
        _read_json_object(node_inventory_path, "node inventory")
    )
    partition_row, partition = _validate_partition(
        _read_json_object(partition_path, "candidate partition")
    )
    names_row, name_groups = _validate_names(_read_json_object(names_path, "candidate names"))
    pair_summary = validate_pair_files(pair_inputs_path, pair_outputs_path)
    retrieval = _validate_config(retrieval_config_path, "retrieval")
    scorer = _validate_config(scorer_config_path, "scorer")
    thresholds = _validate_config(threshold_config_path, "threshold")
    graph = _validate_config(graph_config_path, "graph")

    parent_hash = sha256_file(parent_partition_path)
    if inventory["parent_partition_sha256"] != parent_hash:
        raise ValueError("node inventory is not bound to the supplied parent partition")
    cell = (inventory["task"], inventory["level"], inventory["protocol_id"])
    for context, value in (
        ("candidate partition", partition_row),
        ("candidate names", names_row),
        ("pair scores", pair_summary),
    ):
        other = (value["task"], value["level"], value["protocol_id"])
        if other != cell:
            raise ValueError(f"{context} cell {other} does not match node inventory cell {cell}")
    scorer_protocol = scorer.get("protocol_id")
    if scorer_protocol != cell[2]:
        raise ValueError(f"scorer config protocol {scorer_protocol} does not match {cell[2]}")
    if scorer.get("adapter_sha256") != pair_summary["adapter_sha256"]:
        raise ValueError("scorer config adapter hash does not match pair outputs")
    if scorer.get("protocol_sha256") != pair_summary["protocol_sha256"]:
        raise ValueError("scorer config protocol hash does not match pair outputs")
    if set(partition) != set(nodes):
        missing = sorted(set(nodes) - set(partition))[:10]
        extra = sorted(set(partition) - set(nodes))[:10]
        raise ValueError(f"partition does not exactly cover node inventory: missing={missing}, extra={extra}")
    groups = set(partition.values())
    if name_groups != groups:
        raise ValueError(
            "candidate names do not exactly cover partition groups: "
            f"missing={sorted(groups - name_groups)}, extra={sorted(name_groups - groups)}"
        )
    if names_row["partition_sha256"] != sha256_file(partition_path):
        raise ValueError("candidate names are not bound to the supplied partition")

    for index, raw in enumerate(_read_jsonl(pair_inputs_path), 1):
        pair = validate_pair_input(raw, context=f"{pair_inputs_path}:{index}")
        for side in ("a", "b"):
            node_id = pair[f"node_{side}"]
            if node_id not in nodes:
                raise ValueError(f"pair input references unknown node: {node_id}")
            if pair[f"source_node_{side}_sha256"] != nodes[node_id]["source_node_sha256"]:
                raise ValueError(f"pair input source hash does not match inventory node: {node_id}")
            if pair[f"text_{side}"] != nodes[node_id]["text"]:
                raise ValueError(f"pair input text does not match inventory node: {node_id}")

    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": canonical_json_sha256(
            {name: _artifact_ref(path)["sha256"] for name, path in sorted(paths.items())}
        ),
        "task": cell[0],
        "level": cell[1],
        "protocol_id": cell[2],
        "n_nodes": len(nodes),
        "n_groups": len(groups),
        "n_scored_pairs": pair_summary["n_pairs"],
        "configs": {
            "retrieval": retrieval["config_id"],
            "scorer": scorer["config_id"],
            "threshold": thresholds["config_id"],
            "graph": graph["config_id"],
        },
        "artifacts": {name: _artifact_ref(path) for name, path in sorted(paths.items())},
        "canonical": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-pairs", help="validate pair input/output JSONL")
    validate.add_argument("--inputs", type=Path, required=True)
    validate.add_argument("--outputs", type=Path)
    manifest = commands.add_parser("build-manifest", help="write a new immutable candidate manifest")
    for name in (
        "parent-partition", "node-inventory", "pair-inputs", "pair-outputs",
        "retrieval-config", "scorer-config", "threshold-config", "graph-config",
        "partition", "names", "output",
    ):
        manifest.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate-pairs":
        print(json.dumps(validate_pair_files(args.inputs, args.outputs), sort_keys=True))
        return
    result = build_candidate_manifest(
        parent_partition_path=args.parent_partition,
        node_inventory_path=args.node_inventory,
        pair_inputs_path=args.pair_inputs,
        pair_outputs_path=args.pair_outputs,
        retrieval_config_path=args.retrieval_config,
        scorer_config_path=args.scorer_config,
        threshold_config_path=args.threshold_config,
        graph_config_path=args.graph_config,
        partition_path=args.partition,
        names_path=args.names,
        output_path=args.output,
    )
    print(json.dumps({"candidate_id": result["candidate_id"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
