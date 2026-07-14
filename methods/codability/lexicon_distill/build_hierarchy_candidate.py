#!/usr/bin/env python3
"""Build immutable Gemma-scored R1/R2/R3 hierarchy candidates.

Embeddings retrieve pairs; they never authorize a semantic edge.  Gemma probabilities authorize
graph edges under a development-calibrated threshold.  The resulting partition is always a named
candidate artifact and can become canonical only through the separate promotion gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from methods.codability.lexicon import build_level
from methods.codability.lexicon.semantic_group_merge import _embed_bge, _validate_vectors
from .dataset import PROTOCOL_IDS, TASKS
from .hierarchy_contracts import (
    NODE_INVENTORY_SCHEMA,
    PAIR_INPUT_SCHEMA,
    PARTITION_SCHEMA,
    canonical_json_sha256,
    pair_input_sha256,
    sha256_file,
    validate_pair_files,
)


_CANONICAL = re.compile(r"partition_.+_(?:L0v\d+|R[123])\.json$")


def _write_new(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _json_new(path: Path, payload: object) -> None:
    _write_new(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _protocol_sha(protocols_path: Path, protocol_id: str) -> str:
    payload = json.loads(protocols_path.read_text(encoding="utf-8"))
    row = payload.get(protocol_id)
    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
        raise ValueError(f"protocol bundle lacks {protocol_id}")
    observed = hashlib.sha256(row["text"].encode()).hexdigest()
    if row.get("sha256") != observed:
        raise ValueError(f"protocol hash mismatch for {protocol_id}")
    return observed


def materialize_nodes(
    *, task: str, level: str, protocol_id: str, output_path: str | Path
) -> dict[str, Any]:
    """Freeze lineage and materialize the exact semantic node inventory for one rung."""
    allowed_protocols = {
        "R1": {PROTOCOL_IDS["R1"]},
        "R2": {PROTOCOL_IDS["R2"], PROTOCOL_IDS["R2_V2"], PROTOCOL_IDS["R2_V2_1"]},
        "R3": {PROTOCOL_IDS["R3"]},
    }
    if task not in TASKS or level not in allowed_protocols:
        raise ValueError(f"invalid hierarchy cell: {task}/{level}")
    if protocol_id not in allowed_protocols[level]:
        raise ValueError(f"protocol {protocol_id} is not valid for {level}")
    # Application is read-only with respect to the validated legacy engine.
    # A parent must already have been frozen explicitly; silently resolving
    # "latest" here would both mutate canonical state and revive parent drift.
    manifest_path = Path(build_level.OUT) / f"level_manifest_{task}_{level}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no frozen parent manifest for {task}/{level}: {manifest_path}"
        )
    parent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not parent_manifest.get("parent_partition_path") or not parent_manifest.get(
        "parent_partition_sha256"
    ):
        raise ValueError(f"incomplete frozen parent manifest: {manifest_path}")
    build_level._validate_level_manifest(task, level)
    nodes, _keys = build_level.nodes_from_level(task, level)
    parent_path, parent_sha = build_level._parent_partition(task, level)
    rendered = []
    for node in sorted(nodes, key=lambda row: str(row["node_id"])):
        node_id = str(node["node_id"])
        text = build_level.rep_text(node).strip()
        if not text:
            raise ValueError(f"node {node_id} lacks a semantic representation")
        source_hash = canonical_json_sha256({"node_id": node_id, "text": text})
        rendered.append({"node_id": node_id, "text": text, "source_node_sha256": source_hash})
    inventory = {
        "schema_version": NODE_INVENTORY_SCHEMA,
        "task": task,
        "level": level,
        "protocol_id": protocol_id,
        "parent_partition_sha256": parent_sha,
        "nodes": rendered,
    }
    _json_new(Path(output_path), inventory)
    return {
        "inventory_path": str(Path(output_path).expanduser().resolve()),
        "inventory_sha256": sha256_file(Path(output_path).expanduser().resolve()),
        "parent_partition_path": parent_path,
        "parent_partition_sha256": parent_sha,
        "n_nodes": len(rendered),
    }


def _read_inventory(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from .hierarchy_contracts import _validate_node_inventory

    payload = json.loads(path.read_text(encoding="utf-8"))
    return _validate_node_inventory(payload)


def prepare_pairs(
    *,
    inventory_path: str | Path,
    pairs_path: str | Path,
    report_path: str | Path,
    k: int = 50,
    exhaustive_limit: int = 250,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Retrieve a symmetric BGE kNN candidate graph, exhaustive for small upper rungs."""
    if k < 1 or exhaustive_limit < 2 or batch_size < 1:
        raise ValueError("k, exhaustive_limit, and batch_size must be positive")
    source = Path(inventory_path).expanduser().resolve()
    if Path(pairs_path).expanduser().resolve().exists():
        raise FileExistsError(Path(pairs_path).expanduser().resolve())
    if Path(report_path).expanduser().resolve().exists():
        raise FileExistsError(Path(report_path).expanduser().resolve())
    inventory, by_id = _read_inventory(source)
    ordered = sorted(by_id)
    if len(ordered) < 2:
        raise ValueError("at least two hierarchy nodes are required")
    if len(ordered) <= exhaustive_limit:
        pairs = {(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))}
        retrieval = {"method": "exhaustive", "model": None, "k": None}
    else:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        vectors, model = _embed_bge([by_id[node]["text"] for node in ordered], batch_size=batch_size)
        vectors = _validate_vectors(vectors, len(ordered))
        neighbors = min(k + 1, len(ordered))
        distances, indices = NearestNeighbors(n_neighbors=neighbors, metric="cosine").fit(vectors).kneighbors(vectors)
        pairs = set()
        for i in range(len(ordered)):
            for position in range(indices.shape[1]):
                j = int(indices[i, position])
                if i == j:
                    continue
                a, b = sorted((ordered[i], ordered[j]))
                pairs.add((a, b))
        retrieval = {
            "method": "symmetric_semantic_knn",
            "model": model,
            "k": k,
            "semantic_decider": False,
            "distance_minimum": float(np.min(distances)),
            "distance_maximum": float(np.max(distances)),
        }
    rows = []
    for a, b in sorted(pairs):
        material = {
            "task": inventory["task"], "level": inventory["level"],
            "protocol_id": inventory["protocol_id"], "node_a": a, "node_b": b,
            "inventory_sha256": sha256_file(source),
        }
        row = {
            "schema_version": PAIR_INPUT_SCHEMA,
            "pair_id": hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:24],
            "task": inventory["task"], "level": inventory["level"],
            "protocol_id": inventory["protocol_id"],
            "node_a": a, "node_b": b,
            "text_a": by_id[a]["text"], "text_b": by_id[b]["text"],
            "source_node_a_sha256": by_id[a]["source_node_sha256"],
            "source_node_b_sha256": by_id[b]["source_node_sha256"],
        }
        pair_input_sha256(row)
        rows.append(row)
    destination = Path(pairs_path).expanduser().resolve()
    _write_new(destination, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))
    validation = validate_pair_files(destination)
    report = {
        "schema_version": "gemma-hierarchy-retrieval-report-v1",
        "inventory": {"path": str(source), "sha256": sha256_file(source)},
        "pairs": {"path": str(destination), "sha256": sha256_file(destination)},
        "n_nodes": len(ordered), "n_pairs": len(rows), "retrieval": retrieval,
        "validation": validation,
        "semantic_truth": "none; embeddings retrieve and Gemma later decides relation probabilities",
    }
    _json_new(Path(report_path), report)
    return report


def _read_outputs(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_partition(
    *,
    inventory_path: str | Path,
    pair_inputs_path: str | Path,
    pair_outputs_path: str | Path,
    calibration_path: str | Path,
    partition_path: str | Path,
    report_path: str | Path,
    resolution: float = 1.0,
    related_weight: float = 0.0,
) -> dict[str, Any]:
    """Apply a certified threshold and deterministic weighted Louvain to Gemma scores."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if not 0 <= related_weight < 1:
        raise ValueError("related_weight must be in [0, 1)")
    inventory_file = Path(inventory_path).expanduser().resolve()
    inputs_file = Path(pair_inputs_path).expanduser().resolve()
    outputs_file = Path(pair_outputs_path).expanduser().resolve()
    calibration_file = Path(calibration_path).expanduser().resolve()
    destination = Path(partition_path).expanduser().resolve()
    report_destination = Path(report_path).expanduser().resolve()
    if destination.exists() or report_destination.exists():
        raise FileExistsError(destination if destination.exists() else report_destination)
    if _CANONICAL.fullmatch(destination.name):
        raise ValueError("candidate builder refuses a canonical-looking partition path")
    inventory, nodes = _read_inventory(inventory_file)
    validation = validate_pair_files(inputs_file, outputs_file)
    if (validation["task"], validation["level"], validation["protocol_id"]) != (
        inventory["task"], inventory["level"], inventory["protocol_id"]
    ):
        raise ValueError("pair-scoring cell differs from node inventory")
    input_by_id = {}
    with inputs_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for side in ("a", "b"):
                node_id = row[f"node_{side}"]
                if node_id not in nodes:
                    raise ValueError(f"pair input references unknown inventory node: {node_id}")
                if row[f"source_node_{side}_sha256"] != nodes[node_id]["source_node_sha256"]:
                    raise ValueError(f"pair input source hash differs from inventory node: {node_id}")
                if row[f"text_{side}"] != nodes[node_id]["text"]:
                    raise ValueError(f"pair input text differs from inventory node: {node_id}")
            input_by_id[row["pair_id"]] = row
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    if calibration.get("schema_version") != "gemma4-similarity-threshold-calibration-v1":
        raise ValueError("invalid threshold calibration artifact")
    if calibration.get("certified") is not True:
        raise ValueError("uncertified SAME threshold cannot build a hierarchy candidate")
    if (calibration.get("level"), calibration.get("protocol_id")) != (
        inventory["level"], inventory["protocol_id"]
    ):
        raise ValueError("calibrated threshold belongs to a different level/protocol")
    if calibration.get("adapter_sha256") != validation["adapter_sha256"]:
        raise ValueError("calibrated threshold belongs to a different adapter")
    if calibration.get("protocol_sha256") != validation["protocol_sha256"]:
        raise ValueError("calibrated threshold belongs to different protocol text")
    threshold = calibration.get("selected_same_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("invalid calibrated SAME threshold")
    calibrated_related_weight = calibration.get("selected_related_weight", 0.0)
    if (
        isinstance(calibrated_related_weight, bool)
        or not isinstance(calibrated_related_weight, (int, float))
        or not 0 <= float(calibrated_related_weight) < 1
    ):
        raise ValueError("invalid calibrated RELATED weight")
    if not abs(float(calibrated_related_weight) - related_weight) <= 1e-12:
        raise ValueError("RELATED graph weight differs from its calibrated value")

    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    same_edges = 0
    related_edges = 0
    for output in _read_outputs(outputs_file):
        source = input_by_id[output["pair_id"]]
        a, b = source["node_a"], source["node_b"]
        probabilities = output["probabilities"]
        same = float(probabilities["SAME"])
        related = float(probabilities["RELATED"])
        if same >= float(threshold):
            graph.add_edge(a, b, weight=same)
            same_edges += 1
        elif related_weight and output["prediction"] == "RELATED":
            graph.add_edge(a, b, weight=related_weight * related)
            related_edges += 1
    assignment = {}
    for index, community in enumerate(
        louvain_communities(graph, seed=0, resolution=resolution, weight="weight")
    ):
        for node in community:
            assignment[str(node)] = f"{inventory['task']}_{inventory['level']}_gemma_g{index}"
    if set(assignment) != set(nodes):
        raise AssertionError("Louvain did not assign every frozen node")
    payload = {
        "schema_version": PARTITION_SCHEMA,
        "task": inventory["task"], "level": inventory["level"],
        "protocol_id": inventory["protocol_id"], "partition": assignment,
    }
    _json_new(destination, payload)
    sizes = {}
    for group in assignment.values():
        sizes[group] = sizes.get(group, 0) + 1
    report = {
        "schema_version": "gemma-hierarchy-graph-build-report-v1",
        "inventory": {"path": str(inventory_file), "sha256": sha256_file(inventory_file)},
        "pair_inputs": {"path": str(inputs_file), "sha256": sha256_file(inputs_file)},
        "pair_outputs": {"path": str(outputs_file), "sha256": sha256_file(outputs_file)},
        "calibration": {"path": str(calibration_file), "sha256": sha256_file(calibration_file)},
        "partition": {"path": str(destination), "sha256": sha256_file(destination)},
        "graph": {"same_threshold": float(threshold), "resolution": resolution,
                  "related_weight": related_weight, "same_edges": same_edges,
                  "related_edges": related_edges},
        "groups": {"n": len(sizes), "singletons": sum(size == 1 for size in sizes.values()),
                   "over_30": sum(size > 30 for size in sizes.values()),
                   "maximum_size": max(sizes.values(), default=0)},
        "semantic_decider": "Gemma similarity adapter probabilities only; retrieval does not authorize edges",
        "canonical_write_authorized": False,
    }
    _json_new(report_destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    nodes = sub.add_parser("materialize-nodes")
    nodes.add_argument("--task", required=True)
    nodes.add_argument("--level", required=True, choices=("R1", "R2", "R3"))
    nodes.add_argument("--protocol-id", required=True)
    nodes.add_argument("--output", required=True)
    pairs = sub.add_parser("prepare-pairs")
    pairs.add_argument("--inventory", required=True)
    pairs.add_argument("--pairs", required=True)
    pairs.add_argument("--report", required=True)
    pairs.add_argument("--k", type=int, default=50)
    pairs.add_argument("--exhaustive-limit", type=int, default=250)
    pairs.add_argument("--batch-size", type=int, default=64)
    build = sub.add_parser("build-partition")
    build.add_argument("--inventory", required=True)
    build.add_argument("--pair-inputs", required=True)
    build.add_argument("--pair-outputs", required=True)
    build.add_argument("--calibration", required=True)
    build.add_argument("--partition", required=True)
    build.add_argument("--report", required=True)
    build.add_argument("--resolution", type=float, default=1.0)
    build.add_argument("--related-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "materialize-nodes":
        result = materialize_nodes(task=args.task, level=args.level, protocol_id=args.protocol_id,
                                   output_path=args.output)
    elif args.command == "prepare-pairs":
        result = prepare_pairs(inventory_path=args.inventory, pairs_path=args.pairs,
                               report_path=args.report, k=args.k,
                               exhaustive_limit=args.exhaustive_limit, batch_size=args.batch_size)
    else:
        result = build_partition(
            inventory_path=args.inventory, pair_inputs_path=args.pair_inputs,
            pair_outputs_path=args.pair_outputs, calibration_path=args.calibration,
            partition_path=args.partition, report_path=args.report,
            resolution=args.resolution, related_weight=args.related_weight,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
