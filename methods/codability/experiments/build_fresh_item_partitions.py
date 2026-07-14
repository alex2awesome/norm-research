#!/usr/bin/env python
"""Build hash-frozen, leakage-aware fresh item partitions for confirmatory experiments."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from methods.metric_implementer.manifest import _load_subsampled_df, full_manifest


MANIFEST_PATH = Path(__file__).with_name("fresh_item_partition_manifest_v1.json")

LEGACY_ALLOCATION_STRATEGY = "legacy_greedy_group_drain_v1"
BREADTH_FIRST_ALLOCATION_STRATEGY = "breadth_first_group_round_robin_v2"
ALLOCATION_STRATEGIES = frozenset({
    LEGACY_ALLOCATION_STRATEGY,
    BREADTH_FIRST_ALLOCATION_STRATEGY,
})


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return sha256_bytes(str(text).encode())


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def validate_protocol(protocol: dict) -> list[str]:
    """Static validation before any corpus rows or sealed packet files are opened."""
    errors = []
    domains = protocol.get("domains")
    if not isinstance(domains, dict) or not domains:
        return ["protocol has no domains"]
    known_tasks = {entry.task for entry in full_manifest().datasets}
    for domain, specification in domains.items():
        if specification.get("task") not in known_tasks:
            errors.append(f"{domain}: unknown canonical task {specification.get('task')!r}")
        strategy = specification.get("source_group")
        if isinstance(strategy, str):
            if strategy not in {
                    "writing_prompt_prefix", "question_prefix", "group_column",
                    "content_hash_only", "item_hash"}:
                errors.append(f"{domain}: unsupported source-group strategy {strategy!r}")
        elif isinstance(strategy, dict):
            kind = strategy.get("strategy")
            if kind not in {"text_prefix", "columns", "item_hash"}:
                errors.append(f"{domain}: unsupported source-group strategy {kind!r}")
            if kind == "text_prefix" and not strategy.get("separator"):
                errors.append(f"{domain}: text_prefix strategy lacks separator")
            if kind == "columns" and not strategy.get("columns"):
                errors.append(f"{domain}: columns strategy lacks columns")
        else:
            errors.append(f"{domain}: source-group strategy is missing")
    partition_ids = []
    allocation_strategy = protocol.get(
        "allocation_strategy", LEGACY_ALLOCATION_STRATEGY)
    if allocation_strategy not in ALLOCATION_STRATEGIES:
        errors.append(
            f"unsupported allocation strategy {allocation_strategy!r}")
    coverage = {domain: 0 for domain in domains}
    for partition in protocol.get("partitions", []):
        partition_id = partition.get("id")
        partition_ids.append(partition_id)
        for domain in partition.get("domains", []):
            if domain not in domains:
                errors.append(f"{partition_id}: unknown domain {domain!r}")
                continue
            coverage[domain] += 1
            requested = partition.get("n_by_domain", {}).get(domain, partition.get("n"))
            if not isinstance(requested, int) or requested <= 0:
                errors.append(f"{partition_id}/{domain}: invalid partition size {requested!r}")
    if None in partition_ids or len(partition_ids) != len(set(partition_ids)):
        errors.append("partition ids are missing or duplicated")
    for domain, count in coverage.items():
        if count == 0:
            errors.append(f"{domain}: not covered by any partition")
    if (str(protocol.get("schema", "")).endswith("tacit_breadth")
            and protocol.get("emit_practice_targets") is not False):
        errors.append("tacit breadth protocol must explicitly disable practice targets")
    for prior in protocol.get("exclude_packet_manifests", []):
        path = Path(prior.get("path", ""))
        if not path.is_file() or sha256_file(path) != prior.get("sha256"):
            errors.append(f"prior packet manifest changed or is missing: {path}")
        aliases = prior.get("domain_aliases", {})
        if not set(aliases) <= set(prior.get("domains", [])):
            errors.append(f"prior packet aliases exceed declared domains: {path}")
    return errors


def source_group(domain: str, text: str, row: dict, content_hash: str,
                 strategy: str | dict | None = None) -> str:
    """Recover the strongest stable source group available without using outcome labels."""
    legacy = {
        "writing_prompt_prefix": {
            "strategy": "text_prefix", "separator": "\n\nSTORY:", "tag": "prompt"},
        "question_prefix": {
            "strategy": "text_prefix", "separator": "\n\nAnswer:", "tag": "question"},
        "group_column": {"strategy": "columns", "columns": ["group"], "tag": "company"},
        "content_hash_only": {"strategy": "item_hash"},
    }
    if isinstance(strategy, str):
        strategy = legacy.get(strategy, {"strategy": strategy})
    if strategy is None:
        # Compatibility for packets predating declarative source-group strategies.
        if domain == "cw":
            strategy = legacy["writing_prompt_prefix"]
        elif domain == "math":
            strategy = legacy["question_prefix"]
        elif domain == "pr":
            strategy = legacy["group_column"]
        else:
            strategy = legacy["content_hash_only"]
    if not isinstance(strategy, dict):
        raise ValueError(f"{domain}: invalid source-group strategy {strategy!r}")
    kind = strategy.get("strategy")
    if kind == "text_prefix":
        separator = strategy.get("separator")
        if not isinstance(separator, str) or not separator:
            raise ValueError(f"{domain}: text_prefix source group requires a separator")
        prefix = text.split(separator, 1)[0]
        return f"{strategy.get('tag', 'prefix')}:" + text_sha256(prefix)
    if kind == "columns":
        columns = strategy.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ValueError(f"{domain}: columns source group requires column names")
        values = [str(row.get(column, "")).strip() for column in columns]
        values = [value for value in values if value and value.lower() != "nan"]
        if values:
            return f"{strategy.get('tag', 'column')}:" + text_sha256("||".join(values))
        if strategy.get("required"):
            raise ValueError(f"{domain}: required source-group columns are empty: {columns}")
        return "item:" + content_hash
    if kind != "item_hash":
        raise ValueError(f"{domain}: unsupported source-group strategy {kind!r}")
    return "item:" + content_hash


def records_from_frame(domain: str, frame: pd.DataFrame, *, text_column: str,
                       id_column: str | None, label_column: str | None,
                       source_group_strategy: str | dict | None = None) -> list[dict]:
    records = []
    for index, row in frame.iterrows():
        text = str(row[text_column])
        if not text.strip():
            continue
        content_hash = text_sha256(text)
        item_id = str(row[id_column]) if id_column and id_column in frame.columns else str(index)
        label = row[label_column] if label_column and label_column in frame.columns else None
        if hasattr(label, "item"):
            label = label.item()
        record = {"item_id": item_id, "text": text, "text_sha256": content_hash,
                  "source_group": source_group(
                      domain, text, row.to_dict(), content_hash,
                      strategy=source_group_strategy),
                  "source_split": (str(row["split"]) if "split" in frame.columns else None),
                  "practice_target": label}
        records.append(record)
    # Keep one stable representative for exact duplicates.
    by_hash = {}
    for record in records:
        by_hash.setdefault(record["text_sha256"], record)
    return list(by_hash.values())


def reconstruct_legacy_exclusions(entry, protocol: dict) -> set[str]:
    exclusion = protocol["legacy_exclusion"]
    if exclusion.get("enabled", True) is not True:
        return set()
    reserve = int(exclusion["gepa_reserve"])
    n_probes = int(exclusion["n_probes"])
    frame = _load_subsampled_df(entry, reserve + n_probes,
                                seed=int(exclusion["sampling_seed"]))
    texts = frame[entry.text_column].astype(str).tolist()[reserve: reserve + n_probes]
    return {text_sha256(text) for text in texts}


def _requested_partition_size(spec: dict, domain: str) -> int:
    requested_n = int(spec.get("n_by_domain", {}).get(domain, spec.get("n", 0)))
    if requested_n <= 0:
        raise ValueError(f"{domain}/{spec.get('id')}: partition size must be positive")
    return requested_n


def _allowed_source_splits(spec: dict, domain: str) -> set[str]:
    return set(spec.get("source_split", {}).get(domain, []))


def _eligible_group_rows(
        rows: list[dict], *, allowed_splits: set[str],
        used_hashes: set[str]) -> list[dict]:
    return [
        row for row in rows
        if row["text_sha256"] not in used_hashes
        and (not allowed_splits or row["source_split"] in allowed_splits)
    ]


def _future_capacity_is_sufficient(
        *, groups: dict[str, list[dict]], excluded_group_ids: set[str],
        future_specs: list[dict], domain: str, used_hashes: set[str]) -> bool:
    """Conservative Hall-style capacity check for indivisible source groups.

    A source group may belong to only one future partition.  For every subset of future
    partitions, the union of remaining groups must contain at least their total demand, where a
    group contributes only its largest eligible capacity to that subset.  The bound is exact for
    the two-partition breadth protocol and catches the common greedy-starvation failure for more
    general protocols before any rows are emitted.
    """
    if not future_specs:
        return True
    demands = [_requested_partition_size(spec, domain) for spec in future_specs]
    n_specs = len(future_specs)
    # Protocols in this repository have few partitions.  Avoid an accidental exponential walk
    # for a future large manifest while retaining all singleton/pair/global necessary bounds.
    masks = range(1, 1 << n_specs) if n_specs <= 12 else (
        *(1 << index for index in range(n_specs)),
        (1 << n_specs) - 1,
    )
    masks = tuple(masks)
    indices_by_mask = {
        mask: tuple(index for index in range(n_specs) if mask & (1 << index))
        for mask in masks
    }
    capacities_by_mask = {mask: 0 for mask in masks}
    for group_id, rows in groups.items():
        if group_id in excluded_group_ids:
            continue
        row_capacities = [
            sum(
                row["text_sha256"] not in used_hashes
                and (not _allowed_source_splits(spec, domain)
                     or row["source_split"] in _allowed_source_splits(spec, domain))
                for row in rows
            )
            for spec in future_specs
        ]
        for mask, indices in indices_by_mask.items():
            capacities_by_mask[mask] += max(row_capacities[index] for index in indices)
    for mask, indices in indices_by_mask.items():
        demand = sum(demands[index] for index in indices)
        if capacities_by_mask[mask] < demand:
            return False
    return True


def _allocate_legacy_greedy_group_drain(
        groups: dict[str, list[dict]], specs: list[dict], *, domain: str,
        salt: str) -> dict[str, list[dict]]:
    """Reproduce the original packet allocator byte-for-byte.

    This deliberately drains an early hash-ordered group before visiting the next group.  It is
    retained only so already-frozen packets remain reproducible; new breadth protocols must name
    the round-robin strategy explicitly.
    """
    used_hashes, used_groups = set(), set()
    allocated = {}
    for spec in specs:
        requested_n = _requested_partition_size(spec, domain)
        allowed_splits = _allowed_source_splits(spec, domain)
        group_ids = []
        for group_id, rows in groups.items():
            if group_id in used_groups:
                continue
            if allowed_splits and not any(
                    row["source_split"] in allowed_splits for row in rows):
                continue
            order_key = text_sha256(f"{salt}|{domain}|{spec['id']}|{group_id}")
            group_ids.append((order_key, group_id))
        selected = []
        for _, group_id in sorted(group_ids):
            candidates = _eligible_group_rows(
                groups[group_id], allowed_splits=allowed_splits,
                used_hashes=used_hashes)
            if not candidates:
                continue
            need = requested_n - len(selected)
            selected.extend(candidates[:need])
            used_groups.add(group_id)
            used_hashes.update(row["text_sha256"] for row in candidates)
            if len(selected) == requested_n:
                break
        if len(selected) != requested_n:
            raise ValueError(
                f"{domain}/{spec['id']}: need {requested_n} items, found {len(selected)}")
        allocated[spec["id"]] = selected
    return allocated


def _allocate_breadth_first_group_round_robin(
        groups: dict[str, list[dict]], specs: list[dict], *, domain: str,
        salt: str) -> dict[str, list[dict]]:
    """Allocate one row per touched source group before taking a second.

    Groups are still indivisible across partitions: once any row from a group is selected, every
    row in that group is reserved to the same partition.  A deterministic look-ahead preserves
    enough untouched group capacity for later partitions instead of allowing an early fold to
    make an otherwise feasible protocol fail.
    """
    used_hashes: set[str] = set()
    used_groups: set[str] = set()
    allocated: dict[str, list[dict]] = {}
    for spec_index, spec in enumerate(specs):
        requested_n = _requested_partition_size(spec, domain)
        allowed_splits = _allowed_source_splits(spec, domain)
        future_specs = specs[spec_index + 1:]
        def group_is_candidate(group_id: str) -> bool:
            return group_id not in used_groups and any(
                row["text_sha256"] not in used_hashes
                and (not allowed_splits or row["source_split"] in allowed_splits)
                for row in groups[group_id]
            )

        def candidate_group_ids():
            return (group_id for group_id in groups if group_is_candidate(group_id))

        candidate_count = sum(1 for _ in candidate_group_ids())
        maximum_breadth = min(requested_n, candidate_count)

        def order_key(group_id: str) -> str:
            return text_sha256(f"{salt}|{domain}|{spec['id']}|{group_id}")

        hash_order = heapq.nsmallest(
            maximum_breadth, candidate_group_ids(), key=order_key)

        def selected_rows_by_group(selected_group_ids: list[str]) -> dict[str, list[dict]]:
            return {
                group_id: _eligible_group_rows(
                    groups[group_id], allowed_splits=allowed_splits,
                    used_hashes=used_hashes)
                for group_id in selected_group_ids
            }

        def allocation_is_viable(selected_group_ids: list[str]) -> bool:
            selected_rows = selected_rows_by_group(selected_group_ids)
            if sum(len(value) for value in selected_rows.values()) < requested_n:
                return False
            return _future_capacity_is_sufficient(
                groups=groups,
                excluded_group_ids=used_groups | set(selected_group_ids),
                future_specs=future_specs,
                domain=domain,
                used_hashes=used_hashes,
            )

        selected_group_ids = hash_order[:maximum_breadth]
        if not allocation_is_viable(selected_group_ids):
            # Preserve groups in proportion to their best capacity for any later partition.
            # Groups unavailable to every later fold have zero opportunity cost and are consumed
            # first.  Hash order is the deterministic tie-breaker.
            def future_opportunity_cost(group_id: str) -> int:
                return max((
                    sum(
                        row["text_sha256"] not in used_hashes
                        and (not _allowed_source_splits(future, domain)
                             or row["source_split"] in _allowed_source_splits(
                                 future, domain))
                        for row in groups[group_id]
                    )
                    for future in future_specs
                ), default=0)

            preservation_order = heapq.nsmallest(
                maximum_breadth,
                candidate_group_ids(),
                key=lambda group_id: (future_opportunity_cost(group_id),
                                      order_key(group_id)),
            )
            selected_group_ids = []
            for breadth in range(maximum_breadth, 0, -1):
                proposal = preservation_order[:breadth]
                if allocation_is_viable(proposal):
                    selected_group_ids = proposal
                    break
            if not selected_group_ids:
                raise ValueError(
                    f"{domain}/{spec['id']}: cannot allocate {requested_n} items with "
                    f"source-group-disjoint future capacity under "
                    f"{BREADTH_FIRST_ALLOCATION_STRATEGY}")

        # Pseudorandomize emitted group order independently of the capacity-preservation ranking.
        selected_group_ids = sorted(selected_group_ids, key=order_key)
        candidates_by_group = selected_rows_by_group(selected_group_ids)
        selected: list[dict] = []
        round_index = 0
        while len(selected) < requested_n:
            progressed = False
            for group_id in selected_group_ids:
                rows = candidates_by_group[group_id]
                if round_index >= len(rows):
                    continue
                selected.append(rows[round_index])
                progressed = True
                if len(selected) == requested_n:
                    break
            if not progressed:
                break
            round_index += 1
        if len(selected) != requested_n:
            raise ValueError(
                f"{domain}/{spec['id']}: need {requested_n} items, found {len(selected)} "
                f"under {BREADTH_FIRST_ALLOCATION_STRATEGY}")
        touched = set(selected_group_ids)
        used_groups.update(touched)
        # Reserve every item in every touched group, including rows outside this fold's native
        # split.  ``used_groups`` is the primary barrier; the hashes make that reservation
        # explicit and robust to future allocator changes.
        used_hashes.update(
            row["text_sha256"]
            for group_id in touched
            for row in groups[group_id]
        )
        allocated[spec["id"]] = selected
    return allocated


def _allocate_item_hash_partitions_from_frame(
        frame: pd.DataFrame, specs: list[dict], *, domain: str,
        text_column: str, id_column: str | None, salt: str,
        excluded_hashes: set[str], excluded_groups: set[str],
        ) -> tuple[dict[str, list[dict]], int]:
    """Memory-bounded breadth allocation when every content hash is its own group.

    Large text corpora (notably the 500k-row patent corpus) cannot portably retain a Python dict
    plus a singleton list plus the full text for every source group.  This path scans the already
    loaded frame once per partition, retains only a set of compact hashes and the requested
    hash-minimal records, and is exactly equivalent to group round-robin because every group has
    capacity one after exact-text deduplication.
    """
    allocated: dict[str, list[dict]] = {}
    used_hashes: set[str] = set()
    used_groups: set[str] = set()
    n_unique_texts = 0
    ids = frame[id_column] if id_column and id_column in frame.columns else None
    splits = frame["split"] if "split" in frame.columns else None
    for spec_index, spec in enumerate(specs):
        requested_n = _requested_partition_size(spec, domain)
        allowed_splits = _allowed_source_splits(spec, domain)
        seen_hashes: set[str] = set()

        def candidates():
            for index, value in frame[text_column].items():
                text = str(value)
                if not text.strip():
                    continue
                content_hash = text_sha256(text)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
                group_id = "item:" + content_hash
                source_split = str(splits.at[index]) if splits is not None else None
                if (content_hash in excluded_hashes or group_id in excluded_groups
                        or content_hash in used_hashes or group_id in used_groups
                        or (allowed_splits and source_split not in allowed_splits)):
                    continue
                item_id = str(ids.at[index]) if ids is not None else str(index)
                order_key = text_sha256(
                    f"{salt}|{domain}|{spec['id']}|{group_id}")
                yield order_key, {
                    "item_id": item_id,
                    "text": text,
                    "text_sha256": content_hash,
                    "source_group": group_id,
                    "source_split": source_split,
                    "practice_target": None,
                }

        selected = [
            row for _, row in heapq.nsmallest(
                requested_n, candidates(), key=lambda value: value[0])
        ]
        if spec_index == 0:
            n_unique_texts = len(seen_hashes)
        if len(selected) != requested_n:
            raise ValueError(
                f"{domain}/{spec['id']}: need {requested_n} items, found {len(selected)} "
                f"under {BREADTH_FIRST_ALLOCATION_STRATEGY}")
        used_hashes.update(row["text_sha256"] for row in selected)
        used_groups.update(row["source_group"] for row in selected)
        allocated[spec["id"]] = selected
    return allocated, n_unique_texts


def _iter_projected_dataset_chunks(
        path: str | Path, *, columns: list[str], chunk_size: int = 500):
    """Read only identity/text columns for memory-bounded label-free packet construction."""
    path = Path(path)
    name = path.name
    if name.endswith(".csv.gz"):
        yield from pd.read_csv(
            path, compression="gzip", usecols=columns, chunksize=chunk_size)
        return
    if name.endswith(".tsv.gz"):
        yield from pd.read_csv(
            path, compression="gzip", sep="\t", usecols=columns,
            chunksize=chunk_size)
        return
    if path.suffix == ".csv":
        yield from pd.read_csv(path, usecols=columns, chunksize=chunk_size)
        return
    # None of the current item-hash breadth domains uses these formats.  Retain a projected
    # fallback so the strategy remains manifest-driven without silently loading outcome columns.
    if path.suffix == ".parquet":
        yield pd.read_parquet(path, columns=columns)
        return
    if name.endswith(".jsonl.gz") or name.endswith(".json.gz") or (
            path.suffix == ".gz" and not name.endswith((".csv.gz", ".tsv.gz"))):
        handle = gzip.open(path, "rt")
    elif path.suffix == ".jsonl":
        handle = path.open()
    else:
        handle = None
    if handle is not None:
        batch: list[dict] = []
        start_index = 0
        with handle as source:
            for line in source:
                if not line.strip():
                    continue
                value = json.loads(line)
                batch.append({column: value.get(column) for column in columns})
                if len(batch) == chunk_size:
                    yield pd.DataFrame(
                        batch, index=range(start_index, start_index + len(batch)))
                    start_index += len(batch)
                    batch = []
        if batch:
            yield pd.DataFrame(
                batch, index=range(start_index, start_index + len(batch)))
        return
    if path.suffix == ".json":
        values = json.loads(path.read_text())
        if not isinstance(values, list):
            raise ValueError(f"projected JSON dataset must be a list: {path}")
        for start in range(0, len(values), chunk_size):
            batch = [
                {column: value.get(column) for column in columns}
                for value in values[start:start + chunk_size]
            ]
            yield pd.DataFrame(batch, index=range(start, start + len(batch)))
        return
    raise ValueError(
        f"memory-bounded item-hash allocation does not support dataset format {path}")


def source_projection_grade(path: str | Path) -> str:
    """Describe when undeclared row fields are discarded by the source reader."""
    path = Path(path)
    name = path.name
    if name.endswith((".csv.gz", ".tsv.gz")) or path.suffix in {".csv", ".parquet"}:
        return "parser_column_projection"
    if (
        name.endswith((".jsonl.gz", ".json.gz"))
        or path.suffix in {".jsonl", ".json"}
        or (path.suffix == ".gz" and not name.endswith((".csv.gz", ".tsv.gz")))
    ):
        return "post_decode_key_projection"
    raise ValueError(f"unsupported projected source format {path}")


def projected_source_columns(
        *, text_column: str, id_column: str | None,
        source_group_strategy: str | dict | None,
        partition_specs: list[dict], domain: str) -> list[str]:
    """Columns packet construction may physically load from the outcome-bearing source file."""
    columns = [text_column]
    if id_column:
        columns.append(id_column)
    if isinstance(source_group_strategy, dict):
        if source_group_strategy.get("strategy") == "columns":
            columns.extend(source_group_strategy.get("columns", []))
    if any(_allowed_source_splits(spec, domain) for spec in partition_specs):
        columns.append("split")
    return list(dict.fromkeys(columns))


def load_projected_source_frame(
        dataset_path: str | Path, *, columns: list[str]) -> pd.DataFrame:
    """Load a source corpus without materializing any undeclared outcome column."""
    chunks = list(_iter_projected_dataset_chunks(dataset_path, columns=columns))
    if not chunks:
        return pd.DataFrame(columns=columns)
    if len(chunks) == 1:
        return chunks[0]
    return pd.concat(chunks, axis=0)


def _allocate_item_hash_partitions_from_dataset(
        dataset_path: str | Path, specs: list[dict], *, domain: str,
        text_column: str, id_column: str | None, salt: str,
        excluded_hashes: set[str], excluded_groups: set[str],
        ) -> tuple[dict[str, list[dict]], int, int]:
    """Streaming counterpart to ``_allocate_item_hash_partitions_from_frame``."""
    allocated: dict[str, list[dict]] = {}
    used_hashes: set[str] = set()
    used_groups: set[str] = set()
    n_dataset_rows = 0
    n_unique_texts = 0
    needs_split = any(_allowed_source_splits(spec, domain) for spec in specs)
    columns = [text_column]
    if id_column:
        columns.append(id_column)
    if needs_split:
        columns.append("split")
    columns = list(dict.fromkeys(columns))
    for spec_index, spec in enumerate(specs):
        requested_n = _requested_partition_size(spec, domain)
        allowed_splits = _allowed_source_splits(spec, domain)
        seen_hashes: set[str] = set()

        def candidates():
            nonlocal n_dataset_rows
            for chunk in _iter_projected_dataset_chunks(
                    dataset_path, columns=columns):
                if spec_index == 0:
                    n_dataset_rows += len(chunk)
                ids = chunk[id_column] if id_column else None
                splits = chunk["split"] if needs_split else None
                for index, value in chunk[text_column].items():
                    text = str(value)
                    if not text.strip():
                        continue
                    content_hash = text_sha256(text)
                    if content_hash in seen_hashes:
                        continue
                    seen_hashes.add(content_hash)
                    group_id = "item:" + content_hash
                    source_split = (
                        str(splits.at[index]) if splits is not None else None)
                    if (content_hash in excluded_hashes or group_id in excluded_groups
                            or content_hash in used_hashes or group_id in used_groups
                            or (allowed_splits and source_split not in allowed_splits)):
                        continue
                    item_id = str(ids.at[index]) if ids is not None else str(index)
                    order_key = text_sha256(
                        f"{salt}|{domain}|{spec['id']}|{group_id}")
                    yield order_key, {
                        "item_id": item_id,
                        "text": text,
                        "text_sha256": content_hash,
                        "source_group": group_id,
                        "source_split": source_split,
                        "practice_target": None,
                    }

        selected = [
            row for _, row in heapq.nsmallest(
                requested_n, candidates(), key=lambda value: value[0])
        ]
        if spec_index == 0:
            n_unique_texts = len(seen_hashes)
        if len(selected) != requested_n:
            raise ValueError(
                f"{domain}/{spec['id']}: need {requested_n} items, found {len(selected)} "
                f"under {BREADTH_FIRST_ALLOCATION_STRATEGY}")
        used_hashes.update(row["text_sha256"] for row in selected)
        used_groups.update(row["source_group"] for row in selected)
        allocated[spec["id"]] = selected
    return allocated, n_dataset_rows, n_unique_texts


def allocate_partitions(records: list[dict], specs: list[dict], *, domain: str,
                        salt: str, excluded_hashes: set[str],
                        excluded_groups: set[str] | None = None,
                        allocation_strategy: str = LEGACY_ALLOCATION_STRATEGY,
                        ) -> dict[str, list[dict]]:
    """Allocate exact-size item/source-group-disjoint partitions deterministically."""
    if allocation_strategy not in ALLOCATION_STRATEGIES:
        raise ValueError(f"unsupported allocation strategy {allocation_strategy!r}")
    excluded_groups = excluded_groups or set()
    groups = defaultdict(list)
    for record in records:
        if (record["text_sha256"] not in excluded_hashes
                and record["source_group"] not in excluded_groups):
            groups[record["source_group"]].append(record)
    for rows in groups.values():
        rows.sort(key=lambda row: row["text_sha256"])
    if allocation_strategy == LEGACY_ALLOCATION_STRATEGY:
        return _allocate_legacy_greedy_group_drain(
            groups, specs, domain=domain, salt=salt)
    return _allocate_breadth_first_group_round_robin(
        groups, specs, domain=domain, salt=salt)


def _ordered_set_hash(rows: list[dict]) -> str:
    return sha256_bytes("\n".join(row["text_sha256"] for row in rows).encode())


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload)
    return sha256_file(path)


def _resolve_recorded_path(path: str | Path, *, manifest_path: Path) -> Path:
    """Resolve a packet-recorded path after allowing the repository to move as a unit."""
    value = Path(path)
    candidates = [value]
    if not value.is_absolute():
        candidates.extend((manifest_path.parent / value, Path.cwd() / value))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"prior packet file is missing: {path}")


def load_prior_packet_exclusions(protocol: dict, *, domain: str) -> dict:
    """Authenticate earlier packets and recover hashes/groups without exposing item text."""
    hashes: set[str] = set()
    groups: set[str] = set()
    packets = []
    for spec in protocol.get("exclude_packet_manifests", []):
        declared_domains = spec.get("domains")
        if declared_domains is not None and domain not in set(declared_domains):
            continue
        manifest_path = Path(spec["path"])
        if not manifest_path.is_file():
            raise ValueError(f"prior packet manifest is missing: {manifest_path}")
        observed_manifest_sha256 = sha256_file(manifest_path)
        if observed_manifest_sha256 != spec.get("sha256"):
            raise ValueError(
                "prior packet manifest SHA-256 mismatch: "
                f"observed={observed_manifest_sha256} expected={spec.get('sha256')!r}"
            )
        packet = json.loads(manifest_path.read_text())
        prior_domain = spec.get("domain_aliases", {}).get(domain, domain)
        domain_rows = [
            row for row in packet.get("domains", [])
            if row.get("domain") == prior_domain
        ]
        if len(domain_rows) != 1:
            raise ValueError(
                f"prior packet must contain exactly one domain row for {prior_domain!r}"
            )
        n_items_before = len(hashes)
        n_groups_before = len(groups)
        for partition in domain_rows[0].get("partitions", []):
            item_path = _resolve_recorded_path(
                partition["items_path"], manifest_path=manifest_path)
            observed_items_sha256 = sha256_file(item_path)
            if observed_items_sha256 != partition.get("items_sha256"):
                raise ValueError(
                    "prior packet item SHA-256 mismatch for "
                    f"{domain}/{partition.get('id')}"
                )
            n_rows = 0
            with item_path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    item_hash = row.get("text_sha256")
                    source = row.get("source_group")
                    if not isinstance(item_hash, str) or not item_hash:
                        raise ValueError(
                            f"prior packet lacks item hash in {domain}/{partition.get('id')}"
                        )
                    if not isinstance(source, str) or not source:
                        raise ValueError(
                            f"prior packet lacks source group in {domain}/{partition.get('id')}"
                        )
                    # Deliberately do not retain or return the text field.  The allocator needs
                    # only leakage-blocking identities from sealed predecessor packets.
                    hashes.add(item_hash)
                    groups.add(source)
                    n_rows += 1
            if n_rows != int(partition.get("n", -1)):
                raise ValueError(
                    f"prior packet row count mismatch for {domain}/{partition.get('id')}"
                )
        packets.append({
            "path": str(manifest_path),
            "sha256": observed_manifest_sha256,
            "n_new_item_hashes": len(hashes) - n_items_before,
            "n_new_source_groups": len(groups) - n_groups_before,
        })
    return {"hashes": hashes, "groups": groups, "packets": packets}


def build_domain(domain: str, protocol: dict, out_dir: Path) -> dict:
    domain_spec = protocol["domains"][domain]
    entry = next(row for row in full_manifest().datasets if row.task == domain_spec["task"])
    emit_practice_targets = bool(protocol.get("emit_practice_targets", False))
    legacy_exclusions = reconstruct_legacy_exclusions(entry, protocol)
    prior = load_prior_packet_exclusions(protocol, domain=domain)
    exclusions = legacy_exclusions | prior["hashes"]
    partition_specs = [row for row in protocol["partitions"] if domain in row["domains"]]
    allocation_strategy = protocol.get(
        "allocation_strategy", LEGACY_ALLOCATION_STRATEGY)
    source_group_strategy = domain_spec.get("source_group")
    projected_columns = projected_source_columns(
        text_column=entry.text_column, id_column=entry.id_column,
        source_group_strategy=source_group_strategy,
        partition_specs=partition_specs, domain=domain)
    projection_grade = source_projection_grade(entry.path)
    if (not emit_practice_targets and entry.label_column
            and entry.label_column in projected_columns):
        raise ValueError(
            f"{domain}: outcome column {entry.label_column!r} enters label-free projection")
    if (allocation_strategy == BREADTH_FIRST_ALLOCATION_STRATEGY
            and not emit_practice_targets
            and isinstance(source_group_strategy, dict)
            and source_group_strategy.get("strategy") == "item_hash"):
        allocated, n_dataset_rows, n_unique_texts = (
            _allocate_item_hash_partitions_from_dataset(
            entry.path, partition_specs, domain=domain,
            text_column=entry.text_column, id_column=entry.id_column,
            salt=protocol["salt"], excluded_hashes=exclusions,
            excluded_groups=prior["groups"])
        )
    else:
        frame = (
            load_projected_source_frame(entry.path, columns=projected_columns)
            if (allocation_strategy == BREADTH_FIRST_ALLOCATION_STRATEGY
                and not emit_practice_targets)
            else _load_subsampled_df(
                entry, 0, seed=protocol["legacy_exclusion"]["sampling_seed"])
        )
        n_dataset_rows = len(frame)
        records = records_from_frame(
            domain, frame, text_column=entry.text_column,
            id_column=entry.id_column,
            label_column=(entry.label_column if emit_practice_targets else None),
            source_group_strategy=source_group_strategy)
        n_unique_texts = len(records)
        allocated = allocate_partitions(
            records, partition_specs, domain=domain,
            salt=protocol["salt"], excluded_hashes=exclusions,
            excluded_groups=prior["groups"],
            allocation_strategy=allocation_strategy)
    partition_rows = []
    for spec in partition_specs:
        rows = allocated[spec["id"]]
        item_rows = [{key: row[key] for key in
                      ("item_id", "text_sha256", "source_group", "source_split", "text")}
                     for row in rows]
        target_rows = [{"text_sha256": row["text_sha256"],
                        "practice_target": row["practice_target"]} for row in rows]
        item_path = out_dir / domain / "items" / f"{spec['id']}.jsonl"
        target_path = (
            out_dir / domain / "practice_targets" / f"{spec['id']}.jsonl"
            if emit_practice_targets else None
        )
        partition_row = {
            "id": spec["id"], "n": len(rows), "visibility": spec["visibility"],
            "items_path": str(item_path), "items_sha256": _write_jsonl(item_path, item_rows),
            "targets_path": str(target_path) if target_path else None,
            "targets_sha256": (
                _write_jsonl(target_path, target_rows) if target_path else None
            ),
            "ordered_item_set_sha256": _ordered_set_hash(rows),
            "n_source_groups": len({row["source_group"] for row in rows}),
        }
        partition_rows.append(partition_row)
    return {
        "domain": domain, "task": domain_spec["task"],
        "dataset_path": str(entry.path), "dataset_sha256": sha256_file(entry.path),
        "n_dataset_rows": n_dataset_rows, "n_unique_texts": n_unique_texts,
        "legacy_exclusion_count": len(legacy_exclusions),
        "legacy_exclusion_set_sha256": sha256_bytes(
            "\n".join(sorted(legacy_exclusions)).encode()),
        "prior_packet_exclusion_count": len(prior["hashes"]),
        "prior_packet_source_group_exclusion_count": len(prior["groups"]),
        "prior_packet_exclusion_set_sha256": sha256_bytes(
            "\n".join(sorted(prior["hashes"])).encode()),
        "prior_packet_source_group_set_sha256": sha256_bytes(
            "\n".join(sorted(prior["groups"])).encode()),
        "excluded_packets": prior["packets"],
        "source_group_method": domain_spec["source_group"],
        **({
            "allocation_strategy": allocation_strategy,
            "source_io_projection": {
                "enabled": bool(
                    allocation_strategy == BREADTH_FIRST_ALLOCATION_STRATEGY
                    and not emit_practice_targets),
                "loaded_columns": projected_columns,
                "projection_grade": projection_grade,
                "outcome_column_retained": bool(
                    entry.label_column and entry.label_column in projected_columns),
                "declared_outcome_column": entry.label_column,
            },
            "source_io_policy": (
                (
                    "parser-level identity/text/source-group column projection; declared "
                    "outcome columns are excluded before tabular materialization"
                    if projection_grade == "parser_column_projection"
                    else "row objects are decoded before identity/text/source-group keys are "
                    "retained; declared outcome keys are not retained, emitted, selected on, "
                    "or used"
                )
                if (allocation_strategy == BREADTH_FIRST_ALLOCATION_STRATEGY
                    and not emit_practice_targets)
                else "legacy full-frame loading policy"
            ),
        } if "allocation_strategy" in protocol else {}),
        "holdout_grade": domain_spec["holdout_grade"], "partitions": partition_rows,
    }


def build(*, domains: list[str], out_dir: str | Path,
          manifest_path: str | Path = MANIFEST_PATH) -> dict:
    protocol = load_manifest(manifest_path)
    protocol_errors = validate_protocol(protocol)
    if protocol_errors:
        raise ValueError(protocol_errors)
    undeclared = sorted(set(domains) - set(protocol["domains"]))
    if undeclared:
        raise ValueError(f"requested domains are absent from protocol: {undeclared}")
    out_dir = Path(out_dir)
    domain_rows = [build_domain(domain, protocol, out_dir) for domain in domains]
    report = {
        "schema": "fresh_item_partitions/v1", "protocol_manifest": str(manifest_path),
        "protocol_manifest_sha256": sha256_file(manifest_path),
        **({"allocation_strategy": protocol["allocation_strategy"]}
           if "allocation_strategy" in protocol else {}),
        "domains": domain_rows,
        "anchor_policy": (
            "model-to-model only; no practice targets were emitted"
            if not protocol.get("emit_practice_targets", False)
            else "practice targets are stored separately and are not implied analysis inputs"
        ),
        "access_rule": ("Item text and practice targets are deliberately separate. Do not expose "
                        "sealed-until-final-read files to teachers, selectors, or optimizers."),
    }
    path = out_dir / "packet_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    report["packet_manifest_path"] = str(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domains", default="humor,cw,pr,math")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()
    domains = [value.strip() for value in args.domains.split(",") if value.strip()]
    report = build(domains=domains, out_dir=args.out_dir, manifest_path=args.manifest)
    # Never print item contents or practice targets.
    print(json.dumps({"packet_manifest": report["packet_manifest_path"],
                      "protocol_manifest_sha256": report["protocol_manifest_sha256"],
                      "domains": [{"domain": row["domain"],
                                   "holdout_grade": row["holdout_grade"],
                                   "partitions": {part["id"]: part["n"]
                                                  for part in row["partitions"]}}
                                  for row in report["domains"]]}, indent=1))


if __name__ == "__main__":
    main()
