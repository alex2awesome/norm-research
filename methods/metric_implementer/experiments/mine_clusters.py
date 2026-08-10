"""mine_clusters — harvest a rich candidate-criteria pool from the curated rubric-cluster hierarchy.

The leaf-rubric clustering (notes/2026-05-19__structural-metrics.md) organizes 53K human-authored
rubric forms (11 tasks) into a granularity hierarchy. Two levels are materialized on disk and are
DIRECT candidate-criteria sources — no decomposition, no GPU:

  * **L0** = "same rubric, restated"   — `clusters_<task>.json`  ({provenance_key: cluster_id}).
    One representative canonical text per L0 cluster → the FINEST atomic units.
  * **R1** = "different rubrics, same principle" — `r1_families_<task>.json`
    ({families: [{family_id, name, description, cluster_ids}], cluster_to_family}). Each R1 family
    is an LLM-WRITTEN name+description → a READY-MADE atomic criterion, the coarse/legible backbone.

WHY THIS EXISTS (theory §6.5 / missing-impact §6.7c): GEPA-only mining decomposes a single optimized
prompt and collapses Ω to ~3-6 (the under-mining bug). The cluster hierarchy is a CORPUS-COVERAGE
source — every human-authored norm is represented — so unioning it with GEPA's OPTIMIZATION-RELEVANCE
source lets |Ω| reach the ~8-25 the tail-bound needs. The two are complementary: GEPA anchors in what
is causally relevant to THIS executor X; the clusters guarantee coverage of the normative universe.
(Side-channel criteria — length/format — are NOT in the corpus; those still need adversarial_saturation.)

ZERO GPU. CLI:
  python -m methods.metric_implementer.experiments.mine_clusters --task creative-writing
  python -m methods.metric_implementer.experiments.mine_clusters --task peer-review --levels R1,L0 --pool-max 120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Local mirror of sk3 /lfs/.../norm_embed/match_out (see notes/2026-05-19__structural-metrics.md).
_STRUCT_DIR = str(_REPO_ROOT / "outputs" / "analyses" / "structural_metrics")
_CANON = str(_REPO_ROOT / "outputs" / "analyses" / "canon_all_real_forms.jsonl")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def _dedup(crits):
    """Same 8-token-prefix dedup the GEPA harvester uses, so union sources de-dupe consistently."""
    seen, out = set(), []
    for c in crits:
        k = " ".join(_norm(c).split()[:8])
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def r1_criteria(task: str):
    """R1 families → 'name: description' strings. The coverage backbone (LLM-written, legible)."""
    fn = os.path.join(_STRUCT_DIR, f"r1_families_{task}.json")
    if not os.path.exists(fn):
        return []
    d = json.load(open(fn))
    fams = d.get("families", []) if isinstance(d, dict) else []
    out = []
    for f in fams:
        name, desc = (f.get("name") or "").strip(), (f.get("description") or "").strip()
        if not desc:
            continue
        out.append(f"{name}: {desc}".strip(": ").strip() if name else desc)
    return out


# --------------------------------------------------------------------------------------------
# R2 = "different principles, same metric" — `outputs/hierarchy/<task>_<bucket>_r2_expanded.json`.
# Each merged_group is ONE R2 METRIC: a family with a holistic merged_description (→ GEPA seed) and
# all_leaves (→ the family's child criteria). Buckets general/specific/hyper_specific OVERLAP — fix
# ONE bucket per run to avoid double-counting the same cluster.
# --------------------------------------------------------------------------------------------

_HIER_DIR = str(_REPO_ROOT / "outputs" / "hierarchy")


def _hierarchy_node_id(*, task: str, bucket: str, level: str, source_kind: str,
                       source_index: int, name: str, description: str) -> str:
    """Content-bound identity for a hierarchy node.

    The historical grid code used the integer ``group_idx`` alone.  That integer is only local to
    one level and bucket, so it collides as soon as R1/R2/R3 are analyzed together.  Keep
    ``group_idx`` as a compatibility/display index, but make this composite identity the key for
    all new breadth experiments.
    """
    identity = {
        "task": task,
        "bucket": bucket,
        "level": level,
        "source_kind": source_kind,
        "source_index": int(source_index),
        "name": str(name).strip(),
        "description": str(description).strip(),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return f"{task}::{bucket}::{level}::{source_kind}::{source_index}::{digest}"


def _clean_child_records(children: list[dict] | None) -> list[dict]:
    clean_children = []
    for child in children or []:
        if not isinstance(child, dict):
            continue
        child_name = str(
            child.get("name") or child.get("medoid_name") or "").strip()
        if not child_name:
            continue
        clean = {
            "name": child_name,
            "description": str(
                child.get("description") or child.get("medoid_description") or ""
            ).strip(),
        }
        for field in ("key", "cluster_id", "r2_cluster_id", "n_leaves", "size"):
            if field in child and child[field] is not None:
                clean[field] = child[field]
        clean_children.append(clean)
    return clean_children


def _ordered_unique(values: list) -> list:
    """JSON-stable de-duplication that also tolerates historical mixed identifier types."""
    seen, out = set(), []
    for value in values:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if marker not in seen:
            seen.add(marker)
            out.append(value)
    return out


def _leaf_support_ids(leaves: list[dict]) -> list[str]:
    """Best available content-bound leaf identities for overlap auditing.

    The expanded legacy hierarchy is a DAG rather than a partition.  Immediate source identifiers
    are preferable when present, but raw rubric keys let us audit records whose historical source
    IDs were omitted.  A content hash is the fail-closed fallback; the display name itself is never
    treated as globally unique.
    """
    ids = []
    for leaf in leaves or []:
        if not isinstance(leaf, dict):
            continue
        for field in ("key", "cluster_id", "r2_cluster_id"):
            if leaf.get(field) is not None:
                ids.append(f"{field}:{leaf[field]}")
                break
        else:
            payload = {
                "name": str(leaf.get("name") or leaf.get("medoid_name") or "").strip(),
                "description": str(
                    leaf.get("description") or leaf.get("medoid_description") or ""
                ).strip(),
            }
            if payload["name"]:
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()[:24]
                ids.append(f"content:{digest}")
    return _ordered_unique(ids)


def hierarchy_leaf_support_ids(leaves: list[dict]) -> list[str]:
    """Public, content-bound raw-leaf identities for cross-round dependence audits.

    ``hierarchy_groups`` annotates components within one R1/R2/R3 source file.  Experiments that
    combine rounds must additionally connect records that inherit the same raw rubric across
    files; this wrapper exposes the exact identity rule without duplicating it downstream.
    """
    return _leaf_support_ids(leaves)


def _node_record(*, task: str, bucket: str, level: str, source_kind: str,
                 source_index: int, group_idx: int, name: str, description: str,
                 total_leaf_rubrics: int, leaves: list[dict],
                 components: list[dict] | None = None,
                 immediate_source_ids: list | None = None) -> dict:
    clean_leaves = []
    for leaf in leaves or []:
        if not isinstance(leaf, dict):
            continue
        leaf_name = str(leaf.get("name") or leaf.get("medoid_name") or "").strip()
        if not leaf_name:
            continue
        clean = {"name": leaf_name}
        for field in ("description", "key", "r2_cluster_id", "cluster_id", "n_leaves"):
            if field in leaf and leaf[field] is not None:
                clean[field] = leaf[field]
        clean_leaves.append(clean)
    name = str(name or "").strip()
    description = str(description or "").strip()
    immediate_source_ids = _ordered_unique(list(immediate_source_ids or []))
    leaf_support_ids = _leaf_support_ids(clean_leaves)
    return {
        "node_id": _hierarchy_node_id(
            task=task,
            bucket=bucket,
            level=level,
            source_kind=source_kind,
            source_index=source_index,
            name=name,
            description=description,
        ),
        "task": task,
        "bucket": bucket,
        "level": level,
        "source_kind": source_kind,
        "source_index": int(source_index),
        # Compatibility only. New artifacts must key by node_id, never this local integer.
        "group_idx": int(group_idx),
        "merged_name": name,
        "merged_description": description,
        "total_leaf_rubrics": int(total_leaf_rubrics or len(clean_leaves)),
        "all_leaves": clean_leaves,
        # Immediate child concepts are the procedural decomposition. ``all_leaves`` remains the
        # provenance/ostensive inventory and may be much finer grained.
        "component_children": _clean_child_records(components),
        # These fields expose the dependency structure that the old merged+grandparent readers
        # silently discarded.  They are provenance/audit fields, never model inputs.
        "immediate_source_ids": immediate_source_ids,
        "immediate_source_sha256": hashlib.sha256(json.dumps(
            immediate_source_ids, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "leaf_support_count": len(leaf_support_ids),
        "leaf_support_sha256": hashlib.sha256(json.dumps(
            sorted(leaf_support_ids), ensure_ascii=False).encode()).hexdigest(),
        "_leaf_support_ids": leaf_support_ids,
    }


def _annotate_dependency_components(records: list[dict]) -> list[dict]:
    """Annotate the native action-node DAG without pretending it is a partition.

    Two node records are in the same dependency component when they reuse at least one immediate
    input node.  If a legacy record lacks immediate IDs, raw-leaf support is used as a conservative
    fallback.  The component is the metric-level resampling block for breadth inference; it does
    not merge the constructs or alter their prompt text.
    """
    if not records:
        return records
    parents = list(range(len(records)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    supports: list[set[str]] = []
    raw_supports: list[set[str]] = []
    first_owner: dict[str, int] = {}
    source_owners: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        immediate = record.get("immediate_source_ids", []) or []
        values = (
            {f"immediate:{json.dumps(value, sort_keys=True, ensure_ascii=False)}"
             for value in immediate}
            if immediate
            else {f"leaf:{value}" for value in record.get("_leaf_support_ids", []) or []}
        )
        supports.append(values)
        raw_supports.append({
            f"leaf:{value}" for value in (
                record.get("_leaf_support_ids")
                or _leaf_support_ids(record.get("all_leaves", []) or [])
            )
        })
        for value in values:
            source_owners.setdefault(value, []).append(index)
            if value in first_owner:
                union(index, first_owner[value])
            else:
                first_owner[value] = index

    members: dict[int, list[int]] = {}
    for index in range(len(records)):
        members.setdefault(root(index), []).append(index)
    for indices in members.values():
        node_ids = sorted(records[index]["node_id"] for index in indices)
        component_digest = hashlib.sha256(
            json.dumps(node_ids, ensure_ascii=False).encode()).hexdigest()[:20]
        component_id = (
            f"{records[indices[0]]['task']}::{records[indices[0]]['bucket']}::"
            f"{records[indices[0]]['level']}::dependency::{component_digest}"
        )
        for index in indices:
            neighbours = set()
            multiplicity = 1
            for value in supports[index]:
                owners = source_owners[value]
                multiplicity = max(multiplicity, len(owners))
                neighbours.update(owners)
            neighbours.discard(index)
            records[index]["dependency_component_id"] = component_id
            records[index]["dependency_component_size"] = len(indices)
            records[index]["dependency_degree"] = len(neighbours)
            records[index]["source_assignment_multiplicity_max"] = multiplicity

    # A round-input frontier can still inherit overlapping raw-rubric provenance from an earlier
    # legacy action round.  Keep that dependence separate: it blocks prevalence-over-leaves claims
    # but does not make two named evaluative constructs identical.
    raw_parents = list(range(len(records)))

    def raw_root(index: int) -> int:
        while raw_parents[index] != index:
            raw_parents[index] = raw_parents[raw_parents[index]]
            index = raw_parents[index]
        return index

    def raw_union(left: int, right: int) -> None:
        left_root, right_root = raw_root(left), raw_root(right)
        if left_root != right_root:
            raw_parents[right_root] = left_root

    raw_owners: dict[str, list[int]] = {}
    raw_first_owner: dict[str, int] = {}
    for index, values in enumerate(raw_supports):
        for value in values:
            raw_owners.setdefault(value, []).append(index)
            if value in raw_first_owner:
                raw_union(index, raw_first_owner[value])
            else:
                raw_first_owner[value] = index
    raw_members: dict[int, list[int]] = {}
    for index in range(len(records)):
        raw_members.setdefault(raw_root(index), []).append(index)
    for indices in raw_members.values():
        node_ids = sorted(records[index]["node_id"] for index in indices)
        digest = hashlib.sha256(
            json.dumps(node_ids, ensure_ascii=False).encode()).hexdigest()[:20]
        component_id = (
            f"{records[indices[0]]['task']}::{records[indices[0]]['bucket']}::"
            f"{records[indices[0]]['level']}::raw-provenance::{digest}"
        )
        for index in indices:
            neighbours = set()
            multiplicity = 1
            for value in raw_supports[index]:
                owners = raw_owners[value]
                multiplicity = max(multiplicity, len(owners))
                neighbours.update(owners)
            neighbours.discard(index)
            records[index]["provenance_component_id"] = component_id
            records[index]["provenance_component_size"] = len(indices)
            records[index]["provenance_overlap_degree"] = len(neighbours)
            records[index]["provenance_assignment_multiplicity_max"] = multiplicity
            records[index].pop("_leaf_support_ids", None)
    return records


def _upper_round_inputs(task: str, bucket: str, level: str) -> list[dict]:
    """Materialize the exact node order consumed by the R2/R3 meta-merge round."""
    if level == "R2":
        path = Path(_HIER_DIR) / f"{task}_{bucket}_r1_refined.json"
        if not path.is_file():
            return []
        payload = json.loads(path.read_text())
        inputs = []
        for group in payload.get("parented_trees", []) or []:
            leaves = [
                rubric
                for child in group.get("children", []) or []
                for rubric in child.get("rubrics", []) or []
            ]
            inputs.append({
                "name": group.get("parent_name", ""),
                "description": group.get("parent_description", ""),
                "all_leaves": leaves,
                "components": [{
                    "name": child.get("medoid_name", ""),
                    "description": child.get("medoid_description", ""),
                    "cluster_id": child.get("cluster_id"),
                    "n_leaves": len(child.get("rubrics", []) or []),
                } for child in group.get("children", []) or []],
                "source_kind": "parented_tree",
            })
        for group in payload.get("merged_trees", []) or []:
            inputs.append({
                "name": group.get("merged_name", ""),
                "description": group.get("merged_description", ""),
                "all_leaves": group.get("all_rubrics", []) or [],
                "components": [{
                    "name": medoid,
                    "cluster_id": cluster_id,
                } for cluster_id, medoid in zip(
                    group.get("source_cluster_ids", []) or [],
                    group.get("source_cluster_medoids", []) or [],
                )],
                "source_kind": "merged_tree",
            })
        return inputs
    if level == "R3":
        path = Path(_HIER_DIR) / f"{task}_{bucket}_r2_expanded.json"
        if not path.is_file():
            return []
        payload = json.loads(path.read_text())
        # r2_to_r3_input.py deliberately feeds only R2 merged_groups, in file order.
        prior_inputs = _upper_round_inputs(task, bucket, "R2")
        return [{
            "name": group.get("merged_name", ""),
            "description": group.get("merged_description", ""),
            "all_leaves": group.get("all_leaves", []) or [],
            "components": _source_components(group, prior_inputs),
            "source_kind": "merged_group",
        } for group in payload.get("merged_groups", []) or []]
    return []


def _source_components(group: dict, round_inputs: list[dict]) -> list[dict]:
    ids = group.get("source_r2_cluster_ids", []) or []
    names = group.get("source_r2_cluster_names", []) or []
    components = []
    for position, source_id in enumerate(ids):
        prior = (
            round_inputs[source_id]
            if isinstance(source_id, int) and 0 <= source_id < len(round_inputs)
            else {}
        )
        components.append({
            "name": names[position] if position < len(names) else prior.get("name", ""),
            "description": prior.get("description", ""),
            "r2_cluster_id": source_id,
            "n_leaves": len(prior.get("all_leaves", []) or []),
        })
    return components


def hierarchy_groups(task: str, bucket: str, level: str, *,
                     include_grandparents: bool = True) -> list[dict]:
    """Return the complete refined hierarchy inventory for one task/bucket/level.

    R1 is materialized as ``parented_trees + merged_trees``. R2/R3 are materialized as
    ``merged_groups + grandparents``. The older accessors exposed only ``merged_groups`` at the
    upper levels, silently dropping a substantial part of the taxonomy (including enough R3 nodes
    to change breadth feasibility). Existing merged-group indices remain unchanged; grandparents
    are appended, so old checkpoints retain their meaning.
    """
    level = str(level).upper()
    if level not in {"R1", "R2", "R3"}:
        raise ValueError(f"unsupported hierarchy level: {level!r}")
    stem = "r1_refined" if level == "R1" else f"{level.lower()}_expanded"
    path = Path(_HIER_DIR) / f"{task}_{bucket}_{stem}.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    records: list[dict] = []

    if level == "R1":
        for source_index, group in enumerate(payload.get("parented_trees", []) or []):
            leaves = []
            for child in group.get("children", []) or []:
                rubrics = child.get("rubrics", []) or []
                leaves.extend(rubrics)
                if not rubrics and child.get("medoid_name"):
                    leaves.append({
                        "name": child["medoid_name"],
                        "description": child.get("medoid_description", ""),
                        "cluster_id": child.get("cluster_id"),
                    })
            records.append(_node_record(
                task=task,
                bucket=bucket,
                level=level,
                source_kind="parented_tree",
                source_index=source_index,
                group_idx=len(records),
                name=group.get("parent_name", ""),
                description=group.get("parent_description", ""),
                total_leaf_rubrics=group.get("total_leaf_rubrics", len(leaves)),
                leaves=leaves,
                components=[{
                    "name": child.get("medoid_name", ""),
                    "description": child.get("medoid_description", ""),
                    "cluster_id": child.get("cluster_id"),
                    "n_leaves": len(child.get("rubrics", []) or []),
                } for child in group.get("children", []) or []],
                immediate_source_ids=[
                    child.get("cluster_id")
                    for child in group.get("children", []) or []
                    if child.get("cluster_id") is not None
                ],
            ))
        for source_index, group in enumerate(payload.get("merged_trees", []) or []):
            leaves = group.get("all_rubrics", []) or []
            records.append(_node_record(
                task=task,
                bucket=bucket,
                level=level,
                source_kind="merged_tree",
                source_index=source_index,
                group_idx=len(records),
                name=group.get("merged_name", ""),
                description=group.get("merged_description", ""),
                total_leaf_rubrics=group.get("total_rubric_count", len(leaves)),
                leaves=leaves,
                components=[{
                    "name": medoid,
                    "cluster_id": cluster_id,
                } for cluster_id, medoid in zip(
                    group.get("source_cluster_ids", []) or [],
                    group.get("source_cluster_medoids", []) or [],
                )],
                immediate_source_ids=group.get("source_cluster_ids", []) or [],
            ))
        return _annotate_dependency_components(records)

    round_inputs = _upper_round_inputs(task, bucket, level)
    for source_index, group in enumerate(payload.get("merged_groups", []) or []):
        leaves = group.get("all_leaves", []) or []
        records.append(_node_record(
            task=task,
            bucket=bucket,
            level=level,
            source_kind="merged_group",
            source_index=source_index,
            group_idx=len(records),
            name=group.get("merged_name", ""),
            description=group.get("merged_description", ""),
            total_leaf_rubrics=group.get("total_leaf_rubrics", len(leaves)),
            leaves=leaves,
            components=_source_components(group, round_inputs),
            immediate_source_ids=group.get("source_r2_cluster_ids", []) or [],
        ))
    if include_grandparents:
        for source_index, group in enumerate(payload.get("grandparents", []) or []):
            components = group.get("children", []) or []
            leaves = []
            for child in components:
                source_id = child.get("r2_cluster_id")
                if isinstance(source_id, int) and 0 <= source_id < len(round_inputs):
                    leaves.extend(round_inputs[source_id].get("all_leaves", []) or [])
            records.append(_node_record(
                task=task,
                bucket=bucket,
                level=level,
                source_kind="grandparent",
                source_index=source_index,
                group_idx=len(records),
                name=group.get("grandparent_name", ""),
                description=group.get("grandparent_description", ""),
                total_leaf_rubrics=group.get("total_leaf_rubrics", len(leaves)),
                leaves=leaves,
                components=components,
                immediate_source_ids=[
                    child.get("r2_cluster_id")
                    for child in components
                    if child.get("r2_cluster_id") is not None
                ],
            ))
    return _annotate_dependency_components(records)


def _terminal_frontier_decode(task: str, bucket: str, level: str) -> tuple[list[dict], dict]:
    """Decode a disjoint, round-local frontier from the legacy R2/R3 action DAG.

    ``meta_merge.py`` emits two *action lists*, not a partition: MERGE actions canonicalize the
    same construct while PARENT actions introduce a broader construct.  The prompt did not forbid
    source reuse, and untouched inputs were omitted.  A naive ``merged_groups + grandparents``
    union therefore co-selects ancestors and descendants and loses untouched concepts.

    This decoder is deliberately conservative and never edits an LLM-written action:

    * quarantine both endpoints of every non-laminar conflict;
    * among compatible nested parents, retain only inclusion-maximal parents;
    * absorb a strict merge subset beneath a retained parent;
    * retain every other compatible merge; and
    * carry every uncovered recorded input forward verbatim.

    The result covers each *recorded immediate input ID* exactly once.  It is not a global taxonomy
    partition: R2 inputs already overlap in raw rubric support, and the historical R3 builder only
    consumed R2 merged groups.  Callers must label it a legacy cross-sectional sensitivity.
    """
    level = str(level).upper()
    if level not in {"R2", "R3"}:
        raise ValueError("terminal-frontier decoding is available only for R2/R3 action rounds")
    path = Path(_HIER_DIR) / f"{task}_{bucket}_{level.lower()}_expanded.json"
    if not path.is_file():
        return [], {
            "available": False,
            "reason": "missing expanded hierarchy action file",
            "task": task,
            "bucket": bucket,
            "level": level,
        }
    payload = json.loads(path.read_text())
    round_inputs = _upper_round_inputs(task, bucket, level)
    declared_n = payload.get("n_r2_clusters_in")
    header_errors = []
    if payload.get("task") != task:
        header_errors.append(f"task={payload.get('task')!r}")
    if payload.get("bucket") != bucket:
        header_errors.append(f"bucket={payload.get('bucket')!r}")
    if payload.get("n_merged_groups") != len(payload.get("merged_groups", []) or []):
        header_errors.append("n_merged_groups")
    if payload.get("n_grandparents") != len(payload.get("grandparents", []) or []):
        header_errors.append("n_grandparents")
    if header_errors:
        raise ValueError(
            f"{task}/{bucket}/{level}: expanded action header mismatch: {header_errors}"
        )
    if not isinstance(declared_n, int) or declared_n != len(round_inputs):
        raise ValueError(
            f"{task}/{bucket}/{level}: exact round-input count mismatch "
            f"({declared_n!r} declared, {len(round_inputs)} reconstructed)"
        )
    for source_index, source in enumerate(round_inputs):
        if (not str(source.get("name") or "").strip()
                or not str(source.get("description") or "").strip()):
            raise ValueError(
                f"{task}/{bucket}/{level}: recorded input {source_index} lacks authenticated text"
            )
    native = hierarchy_groups(task, bucket, level, include_grandparents=True)
    native_by_key = {
        (record["source_kind"], record["source_index"]): record for record in native
    }

    actions: list[dict] = []
    for kind, source_kind, rows, id_getter in (
        ("merge", "merged_group", payload.get("merged_groups", []) or [],
         lambda row: row.get("source_r2_cluster_ids", []) or []),
        ("parent", "grandparent", payload.get("grandparents", []) or [],
         lambda row: [
             child.get("r2_cluster_id") for child in row.get("children", []) or []
         ]),
    ):
        for source_index, row in enumerate(rows):
            source_ids = id_getter(row)
            if (not source_ids or any(not isinstance(value, int) for value in source_ids)
                    or len(source_ids) != len(set(source_ids))
                    or len(set(source_ids)) < 2
                    or any(value < 0 or value >= declared_n for value in source_ids)):
                raise ValueError(
                    f"{task}/{bucket}/{level}/{source_kind}/{source_index}: invalid source IDs"
                )
            if source_kind == "merged_group":
                source_names = row.get("source_r2_cluster_names", []) or []
            else:
                source_names = [
                    child.get("name") for child in row.get("children", []) or []
                ]
            expected_names = [round_inputs[value]["name"] for value in source_ids]
            if ([str(value).strip() for value in source_names]
                    != [str(value).strip() for value in expected_names]):
                raise ValueError(
                    f"{task}/{bucket}/{level}/{source_kind}/{source_index}: "
                    "source name/ID alignment mismatch"
                )
            actions.append({
                "id": f"{source_kind}:{source_index}",
                "kind": kind,
                "source_kind": source_kind,
                "source_index": source_index,
                "sources": frozenset(source_ids),
                "record": native_by_key[(source_kind, source_index)],
            })

    merges = [action for action in actions if action["kind"] == "merge"]
    parents = [action for action in actions if action["kind"] == "parent"]
    quarantined: dict[str, set[str]] = {}

    def quarantine(left: dict, right: dict, reason: str) -> None:
        quarantined.setdefault(left["id"], set()).add(f"{reason}:{right['id']}")
        quarantined.setdefault(right["id"], set()).add(f"{reason}:{left['id']}")

    for left_index, left in enumerate(merges):
        for right in merges[left_index + 1:]:
            if left["sources"] & right["sources"]:
                quarantine(left, right, "merge_merge_overlap")
    for left_index, left in enumerate(parents):
        for right in parents[left_index + 1:]:
            overlap = left["sources"] & right["sources"]
            if overlap and not (
                    left["sources"] < right["sources"]
                    or right["sources"] < left["sources"]):
                quarantine(left, right, "parent_parent_non_laminar")
    for merge in merges:
        for parent in parents:
            overlap = merge["sources"] & parent["sources"]
            if overlap and not merge["sources"] < parent["sources"]:
                quarantine(merge, parent, "merge_parent_non_laminar")

    surviving_parents = [
        action for action in parents if action["id"] not in quarantined
    ]
    retained_parents = [
        action for action in surviving_parents
        if not any(
            action["sources"] < other["sources"]
            for other in surviving_parents if other is not action
        )
    ]
    surviving_merges = [action for action in merges if action["id"] not in quarantined]
    retained_merges = [
        action for action in surviving_merges
        if not any(action["sources"] < parent["sources"] for parent in retained_parents)
    ]
    retained = retained_parents + retained_merges
    for left_index, left in enumerate(retained):
        for right in retained[left_index + 1:]:
            if left["sources"] & right["sources"]:
                raise AssertionError(
                    f"{task}/{bucket}/{level}: frontier actions remain overlapping: "
                    f"{left['id']} vs {right['id']}"
                )

    records: list[dict] = []
    covered: set[int] = set()
    for action in sorted(
            retained, key=lambda row: (row["kind"] != "parent", row["source_index"])):
        record = dict(action["record"])
        absorbed = sorted(
            merge["id"] for merge in surviving_merges
            if merge["sources"] < action["sources"]
        ) if action["kind"] == "parent" else []
        record.update({
            "group_idx": len(records),
            "frontier_role": f"retained_{action['kind']}",
            "frontier_source_ids": sorted(action["sources"]),
            "absorbed_descendant_action_ids": absorbed,
            "carried_from_level": None,
        })
        records.append(record)
        covered.update(action["sources"])

    prior_level = "R1" if level == "R2" else "R2"
    for source_index, source in enumerate(round_inputs):
        if source_index in covered:
            continue
        record = _node_record(
            task=task,
            bucket=bucket,
            level=level,
            source_kind="carry_forward",
            source_index=source_index,
            group_idx=len(records),
            name=source.get("name", ""),
            description=source.get("description", ""),
            total_leaf_rubrics=len(source.get("all_leaves", []) or []),
            leaves=source.get("all_leaves", []) or [],
            components=source.get("components", []) or [],
            immediate_source_ids=[source_index],
        )
        record.update({
            "frontier_role": "carried_uncovered_input",
            "frontier_source_ids": [source_index],
            "absorbed_descendant_action_ids": [],
            "carried_from_level": prior_level,
            "carried_from_source_kind": source.get("source_kind"),
        })
        records.append(record)
        covered.add(source_index)

    coverage_counts = [0] * declared_n
    for record in records:
        for source_id in record["frontier_source_ids"]:
            coverage_counts[source_id] += 1
    if any(count != 1 for count in coverage_counts):
        raise AssertionError(
            f"{task}/{bucket}/{level}: terminal frontier does not cover inputs exactly once"
        )
    records = _annotate_dependency_components(records)
    if any(record["dependency_component_size"] != 1 for record in records):
        raise AssertionError(
            f"{task}/{bucket}/{level}: decoded terminal frontier is not source-disjoint"
        )
    dropped_nested_parents = [
        action["id"] for action in surviving_parents if action not in retained_parents
    ]
    absorbed_merges = [
        action["id"] for action in surviving_merges if action not in retained_merges
    ]
    audit = {
        "available": True,
        "schema": "legacy_round_local_terminal_frontier_audit/v1",
        "task": task,
        "bucket": bucket,
        "level": level,
        "source_path": str(path),
        "n_recorded_inputs": declared_n,
        "n_native_merge_actions": len(merges),
        "n_native_parent_actions": len(parents),
        "quarantined_actions": {
            action_id: sorted(reasons) for action_id, reasons in sorted(quarantined.items())
        },
        "dropped_nested_parent_actions": sorted(dropped_nested_parents),
        "absorbed_merge_actions": sorted(absorbed_merges),
        "n_retained_parent_actions": len(retained_parents),
        "n_retained_merge_actions": len(retained_merges),
        "n_carried_inputs": sum(
            record["frontier_role"] == "carried_uncovered_input" for record in records
        ),
        "n_frontier_nodes": len(records),
        "exact_once_input_coverage": True,
        "global_partition_claim": False,
        "interpretation": (
            "disjoint only over the exact recorded inputs to this legacy action round; not a "
            "global raw-rubric partition or a clean paired modern R1/R2/R3 hierarchy"
        ),
    }
    return records, audit


def hierarchy_terminal_frontier(task: str, bucket: str, level: str) -> list[dict]:
    """Public round-local terminal-frontier sensitivity accessor."""
    records, _ = _terminal_frontier_decode(task, bucket, level)
    return records


def hierarchy_terminal_frontier_audit(task: str, bucket: str, level: str) -> dict:
    """Machine-readable audit for :func:`hierarchy_terminal_frontier`."""
    _, audit = _terminal_frontier_decode(task, bucket, level)
    return audit


def _expanded_groups(task: str, bucket: str, level: str):
    """Legacy merged-group inventory used by historical grid/checkpoint indices.

    The breadth experiment opts into the complete round inventory through
    :func:`hierarchy_groups`.  Keeping this accessor merged-only prevents an otherwise silent
    expansion of every older sweep that iterates ``r2_groups``/``r3_groups``.
    """
    return hierarchy_groups(task, bucket, level.upper(), include_grandparents=False)


def r2_groups(task: str, bucket: str = "specific"):
    """Historical R2 merged groups; integer indices retain their frozen meaning."""
    return _expanded_groups(task, bucket, "r2")


def r3_groups(task: str, bucket: str = "general"):
    """Historical R3 merged groups; use ``hierarchy_groups`` for the complete round."""
    return _expanded_groups(task, bucket, "r3")


def r2_criteria(task: str, bucket: str = "specific"):
    """R2 groups → 'merged_name: merged_description' strings (parallels r1_criteria)."""
    return [f"{g['merged_name']}: {g['merged_description']}".strip(": ").strip()
            for g in r2_groups(task, bucket) if g["merged_description"]]


def _expanded_children(task: str, bucket: str, level: str, group_idx: int):
    """Scoped child-criteria pool for ONE R2/R3 metric = its `all_leaves` names (the family's own
    criteria). Source (c) scoped to the family — the coverage backbone for the certificate."""
    groups = _expanded_groups(task, bucket, level)
    if not (0 <= group_idx < len(groups)):
        return []
    return [lf.get("name", "").strip() for lf in groups[group_idx]["all_leaves"]
            if lf.get("name") and len(lf.get("name", "")) > 4]


def r2_children(task: str, bucket: str, group_idx: int):
    """Scoped child-criteria pool for ONE R2 metric = its `all_leaves` names."""
    return _expanded_children(task, bucket, "r2", group_idx)


def r3_children(task: str, bucket: str, group_idx: int):
    """Scoped child-criteria pool for ONE R3 metric = its `all_leaves` names."""
    return _expanded_children(task, bucket, "r3", group_idx)


def _l0_key_to_canon(task: str):
    """Stream canon_all_real_forms.jsonl, yield (key, canonical) for `task` only."""
    if not os.path.exists(_CANON):
        return
    with open(_CANON) as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("task") != task:
                continue
            key, canon = o.get("key"), o.get("canonical")
            if key and canon:
                yield key, canon


def l0_reps(task: str):
    """One representative canonical text per L0 cluster (the median-length member — avoids
    degenerate short/long restatements). Joins clusters_<task>.json (key→cid) × canon (key→text)."""
    fn = os.path.join(_STRUCT_DIR, f"clusters_{task}.json")
    if not os.path.exists(fn) or not os.path.exists(_CANON):
        return []
    key2cid = json.load(open(fn))            # {provenance_key: cluster_id}
    cid2texts: dict = {}
    n_hit = 0
    for key, canon in _l0_key_to_canon(task):
        cid = key2cid.get(key)
        if cid is None:
            continue
        cid2texts.setdefault(cid, []).append(canon)
        n_hit += 1
    reps = []
    for cid, texts in cid2texts.items():
        if not texts:
            continue
        texts.sort(key=len)
        reps.append(texts[len(texts) // 2])  # median-length member = the representative
    return reps, n_hit


def _cid2rep(task: str) -> dict:
    """``{cluster_id: representative canonical text}`` — the median-length member per L0 cluster.
    Used to materialize R1 family children from their `cluster_ids`."""
    fn = os.path.join(_STRUCT_DIR, f"clusters_{task}.json")
    if not os.path.exists(fn) or not os.path.exists(_CANON):
        return {}
    key2cid = json.load(open(fn))
    cid2texts: dict = {}
    for key, canon in _l0_key_to_canon(task):
        cid = key2cid.get(key)
        if cid is None:
            continue
        cid2texts.setdefault(cid, []).append(canon)
    out = {}
    for cid, texts in cid2texts.items():
        texts.sort(key=len)
        out[cid] = texts[len(texts) // 2]
    return out


# --------------------------------------------------------------------------------------------
# R1 = "different rubrics, same principle" — `r1_families_<task>.json`
# ({families:[{family_id, name, description, cluster_ids}]}). Each family is an LLM-WRITTEN
# name+description → a metric whose children are its `cluster_ids`' L0 representative texts.
# NOTE for CW: R1 over-fragmented (~1700 families, median 1 leaf) → degenerate as standalone
# metrics (≈ L0 atoms); SAMPLE rather than enumerate, or use R3 for a real coarse level.
# --------------------------------------------------------------------------------------------

def r1_groups(task: str):
    """R1 families as metric groups: [{group_idx, merged_name, merged_description,
    total_leaf_rubrics, all_leaves:[{name}]}]. `merged_description` is the family description (→ M_i /
    GEPA seed); children = the L0 representative texts of the family's `cluster_ids`."""
    fn = os.path.join(_STRUCT_DIR, f"r1_families_{task}.json")
    if not os.path.exists(fn):
        return []
    d = json.load(open(fn))
    fams = d.get("families", []) if isinstance(d, dict) else []
    cid2rep = _cid2rep(task)
    out = []
    for i, f in enumerate(fams):
        name = (f.get("name") or "").strip()
        desc = (f.get("description") or "").strip()
        cids = f.get("cluster_ids", []) or []
        leaves = [{"name": cid2rep[c]} for c in cids if c in cid2rep]
        if not leaves and name:                 # no canon rep resolved → seed with the family name
            leaves = [{"name": name}]
        out.append({"group_idx": i, "merged_name": name, "merged_description": desc,
                    "total_leaf_rubrics": len(cids), "all_leaves": leaves})
    return out


def r1_children(task: str, group_idx: int):
    """Scoped child-criteria pool for ONE R1 family = its `cluster_ids`' L0 representative texts."""
    groups = r1_groups(task)
    if not (0 <= group_idx < len(groups)):
        return []
    return [lf.get("name", "").strip() for lf in groups[group_idx]["all_leaves"]
            if lf.get("name") and len(lf.get("name", "")) > 4]


def mine(task: str, levels=("R1", "L0"), pool_max: int | None = None, want_provenance: bool = False,
         bucket: str = "specific"):
    """Union the requested levels → dedup'd candidate pool. `want_provenance` returns
    [{criterion, level}] so the pool COMPOSITION (which source each came from) is inspectable —
    useful for diagnosing whether Ω is GEPA- or corpus-driven and for relevance-weighting. `bucket`
    selects the R2 specificity hierarchy (general/specific/hyper_specific — they OVERLAP, fix one)."""
    pool, l0_hit = [], None
    if "R2" in levels:
        for c in r2_criteria(task, bucket):
            pool.append((c, "R2"))
    if "R1" in levels:
        for c in r1_criteria(task):
            pool.append((c, "R1"))
    if "L0" in levels:
        reps, l0_hit = l0_reps(task)
        for c in reps:
            pool.append((c, "L0"))
    # de-dupe by text (across levels too — an R1 family often restates its member L0 cluster)
    seen, dedup = set(), []
    for c, lvl in pool:
        k = " ".join(_norm(c).split()[:8])
        if k and k not in seen:
            seen.add(k)
            dedup.append((c, lvl))
    if pool_max:
        dedup = dedup[:pool_max]
    if want_provenance:
        return [{"criterion": c, "level": lvl} for c, lvl in dedup], {"l0_keys_matched": l0_hit}
    return [c for c, _ in dedup], {"l0_keys_matched": l0_hit, "n_levels": list(levels)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mine_clusters", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--levels", default="R1,L0", help="comma list of {R1,L0}")
    ap.add_argument("--pool-max", type=int, default=None)
    ap.add_argument("--out", default=None, help="write criteria one-per-line (else just print stats)")
    a = ap.parse_args(argv)
    levels = tuple(x.strip().upper() for x in a.levels.split(",") if x.strip())
    pool, meta = mine(a.task, levels=levels, pool_max=a.pool_max, want_provenance=False)
    by = {}
    # recount per level (post-dedup) for the readout
    prov, _ = mine(a.task, levels=levels, pool_max=a.pool_max, want_provenance=True)
    by = {}
    for p in prov:
        by[p["level"]] = by.get(p["level"], 0) + 1
    print(f"[{a.task}] levels={list(levels)} → {len(pool)} candidate criteria "
          f"(by level: {by}; L0 canon-keys matched: {meta.get('l0_keys_matched')})")
    for c in pool[:6]:
        print(f"  - {c[:110]}")
    if a.out:
        with open(a.out, "w") as f:
            for c in pool:
                f.write(f"- {c}\n")
        print(f"wrote {len(pool)} criteria to {a.out}")


if __name__ == "__main__":
    main()
