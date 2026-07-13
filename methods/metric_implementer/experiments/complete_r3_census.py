"""Materialize a complete R3 partition from the historical merge-only output.

The R2-to-R3 meta-merge artifact stores only multi-node merges. R2 metrics that
were not merged are omitted, so reading ``merged_groups`` directly selects for
mergeable metrics rather than representing the R3 level. This module preserves
the existing merge order and appends every untouched R2 input as a singleton
carry-forward. The result is a disjoint, exhaustive partition of the frozen R2
input and remains compatible with consumers that read ``merged_groups``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


SCHEMA = "cr3-complete-r3-partition-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_group(parent: Mapping[str, object], source_index: int) -> dict:
    children = list(parent.get("children") or [])
    leaves = []
    for child in children:
        leaves.extend(list((child or {}).get("rubrics") or []))
    name = str(parent.get("parent_name") or "").strip()
    description = str(parent.get("parent_description") or "").strip()
    if not name or not description:
        raise ValueError(f"R2 input {source_index} lacks a name or description")
    return {
        "merged_name": name,
        "merged_description": description,
        "source_r2_cluster_ids": [int(source_index)],
        "source_r2_cluster_names": [name],
        "total_leaf_rubrics": len(leaves),
        "all_leaves": leaves,
        "r3_membership_type": "singleton_carry_forward",
        "source_r3_merged_group_index": None,
    }


def build_complete_r3_partition(
    r2_as_r3_input: Mapping[str, object],
    r3_expanded: Mapping[str, object],
    *,
    r2_input_sha256: str,
    r3_expanded_sha256: str,
) -> dict:
    """Return a disjoint complete partition, rejecting ambiguous source maps."""
    task = str(r3_expanded.get("task") or "")
    bucket = str(r3_expanded.get("bucket") or "")
    if not task or not bucket:
        raise ValueError("R3 artifact lacks task or bucket identity")
    if (str(r2_as_r3_input.get("task") or "") != task
            or str(r2_as_r3_input.get("bucket") or "") != bucket):
        raise ValueError("R2 input and R3 artifact identify different strata")

    parents = list(r2_as_r3_input.get("parented_trees") or [])
    n_input = int(r3_expanded.get("n_r2_clusters_in", -1))
    if n_input < 1 or len(parents) != n_input:
        raise ValueError(
            f"R3 declares {n_input} R2 inputs but the frozen input contains {len(parents)}")

    complete_groups = []
    covered: set[int] = set()
    for merge_index, raw_group in enumerate(r3_expanded.get("merged_groups") or []):
        group = dict(raw_group)
        source_ids = [int(value) for value in group.get("source_r2_cluster_ids") or []]
        if len(source_ids) < 2 or len(set(source_ids)) != len(source_ids):
            raise ValueError(f"R3 merge {merge_index} is not a unique multi-node merge")
        if any(value < 0 or value >= n_input for value in source_ids):
            raise ValueError(f"R3 merge {merge_index} references an out-of-range R2 input")
        overlap = covered.intersection(source_ids)
        if overlap:
            raise ValueError(f"R3 merges overlap on R2 inputs {sorted(overlap)}")
        covered.update(source_ids)
        group["r3_membership_type"] = "multi_r2_merge"
        group["source_r3_merged_group_index"] = int(merge_index)
        complete_groups.append(group)

    untouched = [index for index in range(n_input) if index not in covered]
    complete_groups.extend(_source_group(parents[index], index) for index in untouched)

    partition_ids = [
        int(source_id)
        for group in complete_groups
        for source_id in group["source_r2_cluster_ids"]
    ]
    if sorted(partition_ids) != list(range(n_input)) or len(partition_ids) != n_input:
        raise ValueError("complete R3 groups do not partition the frozen R2 input exactly once")

    core = {
        "schema": SCHEMA,
        "task": task,
        "bucket": bucket,
        "round": r3_expanded.get("round"),
        "model": r3_expanded.get("model"),
        "source_r2_as_r3_input_sha256": str(r2_input_sha256),
        "source_merge_only_r3_expanded_sha256": str(r3_expanded_sha256),
        "n_r2_clusters_in": n_input,
        "n_multi_r2_merges": len(r3_expanded.get("merged_groups") or []),
        "n_r2_inputs_covered_by_merges": len(covered),
        "n_singleton_carry_forwards": len(untouched),
        "n_merged_groups": len(complete_groups),
        "merged_groups": complete_groups,
        "n_grandparents": len(r3_expanded.get("grandparents") or []),
        "grandparents": list(r3_expanded.get("grandparents") or []),
        "grandparents_role": "higher_order_auxiliary_not_part_of_the_R3_partition",
        "partition_contract": {
            "disjoint": True,
            "exhaustive_over_frozen_r2_input": True,
            "existing_multi_merge_indices_preserved": True,
            "singleton_indices_appended_in_source_order": True,
            "uses_reconstruction_or_certificate_outcomes": False,
            "uses_external_labels": False,
        },
    }
    return {**core, "partition_sha256": _payload_sha256(core)}


def materialize_complete_r3_partition(
    r2_input_path: str | Path,
    r3_expanded_path: str | Path,
    output_path: str | Path,
) -> dict:
    r2_path = Path(r2_input_path).resolve()
    r3_path = Path(r3_expanded_path).resolve()
    out_path = Path(output_path).resolve()
    r2_payload = json.loads(r2_path.read_text(encoding="utf-8"))
    r3_payload = json.loads(r3_path.read_text(encoding="utf-8"))
    result = build_complete_r3_partition(
        r2_payload,
        r3_payload,
        r2_input_sha256=_file_sha256(r2_path),
        r3_expanded_sha256=_file_sha256(r3_path),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, out_path)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-as-r3-input", required=True)
    parser.add_argument("--r3-expanded", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = materialize_complete_r3_partition(
        args.r2_as_r3_input, args.r3_expanded, args.output)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "n_complete_r3": result["n_merged_groups"],
        "n_multi_merge": result["n_multi_r2_merges"],
        "n_singleton": result["n_singleton_carry_forwards"],
        "partition_sha256": result["partition_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
