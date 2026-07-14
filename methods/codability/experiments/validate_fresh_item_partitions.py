#!/usr/bin/env python
"""Validate fresh confirmatory partitions without exposing sealed text or targets."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import (
    LEGACY_ALLOCATION_STRATEGY,
    MANIFEST_PATH,
    _iter_projected_dataset_chunks,
    load_prior_packet_exclusions,
    load_manifest,
    projected_source_columns,
    reconstruct_legacy_exclusions,
    sha256_bytes,
    sha256_file,
    source_group,
    source_projection_grade,
    text_sha256,
)
from methods.metric_implementer.manifest import full_manifest


def _resolve(raw_path: str | None, manifest_path: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path if path.exists() else None
    for base in (Path.cwd(), manifest_path.parent, *manifest_path.parents):
        candidate = base / path
        if candidate.exists():
            return candidate
    return None


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _target_valid(value) -> bool:
    if value is None or isinstance(value, bool):
        return value is not None
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return bool(str(value).strip())


def _source_membership_certificate(
    *,
    domain: str,
    entry,
    domain_spec: dict,
    partition_specs: list[dict],
    packet_items: list[dict],
) -> dict:
    """Recompute packet row membership and group identity from projected raw-source columns."""
    target_rows = {
        row.get("text_sha256"): row for row in packet_items
        if isinstance(row.get("text_sha256"), str)
    }
    text_column = getattr(entry, "text_column", None)
    id_column = getattr(entry, "id_column", None)
    label_column = getattr(entry, "label_column", None)
    if not text_column:
        return {
            "schema": "fresh_item_source_membership/v1",
            "valid": False,
            "domain": domain,
            "n_packet_items": len(packet_items),
            "n_matched_items": 0,
            "allocation_replay_verified": False,
            "errors": ["canonical dataset has no declared text column"],
        }
    projected_columns = projected_source_columns(
        text_column=text_column,
        id_column=id_column,
        source_group_strategy=domain_spec.get("source_group"),
        partition_specs=partition_specs,
        domain=domain,
    )
    mismatch_counts: Counter[str] = Counter()
    if len(target_rows) != len(packet_items):
        mismatch_counts["duplicate_or_missing_packet_hash"] += (
            len(packet_items) - len(target_rows))
    if label_column and label_column in projected_columns:
        mismatch_counts["outcome_column_entered_projection"] += 1
    first_source_occurrence_seen: set[str] = set()
    matched_hashes: set[str] = set()
    n_source_rows = 0
    for chunk in _iter_projected_dataset_chunks(entry.path, columns=projected_columns):
        n_source_rows += len(chunk)
        for index, value in chunk[text_column].items():
            text = str(value)
            if not text.strip():
                continue
            content_hash = text_sha256(text)
            if (content_hash not in target_rows
                    or content_hash in first_source_occurrence_seen):
                continue
            # ``records_from_frame`` keeps the first exact-text occurrence.  Comparing the first
            # matching raw row therefore verifies both source membership and canonical dedup
            # identity without retaining the rest of the outcome-bearing corpus.
            first_source_occurrence_seen.add(content_hash)
            packet_row = target_rows[content_hash]
            source_row = chunk.loc[index]
            expected_item_id = (
                str(source_row[id_column])
                if id_column and id_column in chunk.columns
                else str(index)
            )
            expected_group = source_group(
                domain, text, source_row.to_dict(), content_hash,
                strategy=domain_spec.get("source_group"))
            expected_split = (
                str(source_row["split"]) if "split" in chunk.columns else None)
            local_mismatch = False
            for field, expected in (
                    ("text", text),
                    ("item_id", expected_item_id),
                    ("source_group", expected_group),
                    ("source_split", expected_split)):
                if packet_row.get(field) != expected:
                    mismatch_counts[field] += 1
                    local_mismatch = True
            if not local_mismatch:
                matched_hashes.add(content_hash)
    missing = set(target_rows) - first_source_occurrence_seen
    if missing:
        mismatch_counts["source_row_missing"] += len(missing)
    item_set_sha256 = sha256_bytes("\n".join(sorted(target_rows)).encode())
    errors = [
        f"{field}: {count} row(s)"
        for field, count in sorted(mismatch_counts.items()) if count
    ]
    return {
        "schema": "fresh_item_source_membership/v1",
        "valid": not errors and len(matched_hashes) == len(target_rows),
        "domain": domain,
        "dataset_path": str(entry.path),
        "dataset_sha256": sha256_file(entry.path),
        "projected_columns": projected_columns,
        "projection_grade": source_projection_grade(entry.path),
        "declared_outcome_column": label_column,
        "outcome_column_retained": bool(label_column and label_column in projected_columns),
        "n_source_rows": n_source_rows,
        "n_packet_items": len(packet_items),
        "n_matched_items": len(matched_hashes),
        "packet_item_set_sha256": item_set_sha256,
        "source_group_identity_recomputed": True,
        "canonical_first_occurrence_checked": True,
        "allocation_replay_verified": False,
        "errors": errors,
    }


def validate_packet(packet_manifest_path: str | Path, *,
                    protocol_path: str | Path = MANIFEST_PATH,
                    domains: set[str] | None = None,
                    partitions: set[str] | None = None,
                    verify_source_membership: bool = False,
                    verify_dataset_files: bool = True) -> dict:
    if verify_source_membership and not verify_dataset_files:
        raise ValueError(
            "source membership cannot be recomputed without verifying dataset files"
        )
    packet_manifest_path = Path(packet_manifest_path)
    packet = json.loads(packet_manifest_path.read_text())
    protocol = load_manifest(protocol_path)
    errors: list[str] = []
    expected_protocol_hash = sha256_file(protocol_path)
    if packet.get("protocol_manifest_sha256") != expected_protocol_hash:
        errors.append("protocol manifest SHA-256 mismatch")
    expected_allocation_strategy = protocol.get(
        "allocation_strategy", LEGACY_ALLOCATION_STRATEGY)
    observed_allocation_strategy = packet.get(
        "allocation_strategy", LEGACY_ALLOCATION_STRATEGY)
    if observed_allocation_strategy != expected_allocation_strategy:
        errors.append("allocation strategy differs from protocol")
    protocol_domains = protocol.get("domains", {})
    if not isinstance(protocol_domains, dict):
        errors.append("protocol domains must be an object")
        protocol_domains = {}
    spec_rows = protocol.get("partitions", [])
    specs = {row.get("id"): row for row in spec_rows if row.get("id") is not None}
    if len(specs) != len(spec_rows):
        errors.append("protocol partition ids are missing or duplicated")

    packet_domain_rows = packet.get("domains", [])
    packet_domains: dict[str, dict] = {}
    for row in packet_domain_rows:
        domain = row.get("domain")
        if not isinstance(domain, str) or not domain:
            errors.append("packet contains a domain row without an id")
        elif domain in packet_domains:
            errors.append(f"packet contains duplicate domain row {domain!r}")
        else:
            packet_domains[domain] = row

    selected_domains = (
        set(packet_domains) if domains is None else set(domains)
    )
    for domain in sorted(selected_domains - set(protocol_domains)):
        errors.append(f"requested domain {domain!r} is absent from protocol")
    for domain in sorted(selected_domains - set(packet_domains)):
        errors.append(f"requested domain {domain!r} is absent from packet")

    requested_partitions = None if partitions is None else set(partitions)
    if requested_partitions is not None:
        for partition_id in sorted(requested_partitions - set(specs)):
            errors.append(
                f"requested partition {partition_id!r} is absent from protocol")
        packet_partition_ids = {
            partition.get("id")
            for row in packet_domain_rows
            for partition in row.get("partitions", [])
        }
        for partition_id in sorted(requested_partitions - packet_partition_ids):
            errors.append(
                f"requested partition {partition_id!r} is absent from packet")

    all_hashes: set[str] = set()
    all_groups: set[str] = set()
    checked_partitions: set[str] = set()
    domain_reports = []
    source_membership_reports = []
    # Retain packet order for legacy reports; append explicitly requested missing domains so
    # their absence is also visible in the domain-level audit trail.
    ordered_domains = [
        row.get("domain") for row in packet_domain_rows
        if row.get("domain") in selected_domains
    ]
    ordered_domains.extend(sorted(selected_domains - set(ordered_domains)))
    for domain in ordered_domains:
        domain_row = packet_domains.get(domain)
        if domain_row is None or domain not in protocol_domains:
            continue
        local_errors = []
        if domain_row.get(
                "allocation_strategy", LEGACY_ALLOCATION_STRATEGY
        ) != expected_allocation_strategy:
            local_errors.append("domain allocation strategy differs from protocol")
        entries = [
            row for row in full_manifest().datasets
            if row.task == protocol_domains[domain]["task"]
        ]
        if len(entries) != 1:
            local_errors.append(
                f"canonical dataset lookup returned {len(entries)} rows")
            errors.extend(local_errors)
            domain_reports.append({
                "domain": domain, "valid": False, "n_items": 0,
                "n_source_groups": 0, "errors": local_errors,
            })
            continue
        entry = entries[0]
        if expected_allocation_strategy != LEGACY_ALLOCATION_STRATEGY:
            text_column = getattr(entry, "text_column", None)
            id_column = getattr(entry, "id_column", None)
            label_column = getattr(entry, "label_column", None)
            partition_specs_for_domain = [
                row for row in spec_rows if domain in row.get("domains", [])
            ]
            if not text_column:
                local_errors.append("canonical dataset has no declared text column")
                expected_projection = None
            else:
                expected_projection = projected_source_columns(
                    text_column=text_column,
                    id_column=id_column,
                    source_group_strategy=protocol_domains[domain].get("source_group"),
                    partition_specs=partition_specs_for_domain,
                    domain=domain,
                )
            projection = domain_row.get("source_io_projection", {})
            if (
                expected_projection is None
                or projection.get("enabled") is not True
                or projection.get("loaded_columns") != expected_projection
                or projection.get("projection_grade")
                != source_projection_grade(entry.path)
                or projection.get("outcome_column_retained") is not False
                or projection.get("declared_outcome_column") != label_column
            ):
                local_errors.append(
                    "source I/O projection certificate differs from protocol")
        dataset_path = Path(entry.path)
        if verify_dataset_files:
            if not dataset_path.exists():
                local_errors.append("dataset file is missing")
            elif sha256_file(dataset_path) != domain_row.get("dataset_sha256"):
                local_errors.append("dataset SHA-256 mismatch")
        legacy = reconstruct_legacy_exclusions(entry, protocol)
        legacy_hash = sha256_bytes("\n".join(sorted(legacy)).encode())
        if legacy_hash != domain_row.get("legacy_exclusion_set_sha256"):
            local_errors.append("legacy exclusion reconstruction mismatch")
        prior = load_prior_packet_exclusions(protocol, domain=domain)
        if len(prior["hashes"]) != domain_row.get("prior_packet_exclusion_count", 0):
            local_errors.append("prior packet item-exclusion count mismatch")
        if len(prior["groups"]) != domain_row.get(
                "prior_packet_source_group_exclusion_count", 0):
            local_errors.append("prior packet source-group exclusion count mismatch")
        prior_hash = sha256_bytes("\n".join(sorted(prior["hashes"])).encode())
        if prior_hash != domain_row.get(
                "prior_packet_exclusion_set_sha256", sha256_bytes(b"")):
            local_errors.append("prior packet item-exclusion hash mismatch")
        prior_group_hash = sha256_bytes("\n".join(sorted(prior["groups"])).encode())
        if prior_group_hash != domain_row.get(
                "prior_packet_source_group_set_sha256", sha256_bytes(b"")):
            local_errors.append("prior packet source-group exclusion hash mismatch")

        domain_hashes: set[str] = set()
        domain_groups: set[str] = set()
        n_items = 0
        membership_items: list[dict] = []
        packet_partition_rows = domain_row.get("partitions", [])
        packet_partitions: dict[str, dict] = {}
        for partition_row in packet_partition_rows:
            partition_id = partition_row.get("id")
            if not isinstance(partition_id, str) or not partition_id:
                local_errors.append("packet contains a partition row without an id")
            elif partition_id in packet_partitions:
                local_errors.append(
                    f"{domain}/{partition_id}: duplicate packet partition")
            else:
                packet_partitions[partition_id] = partition_row
        selected_partition_ids = (
            list(packet_partitions)
            if requested_partitions is None
            else sorted(requested_partitions)
        )
        for partition_id in selected_partition_ids:
            label = f"{domain}/{partition_id}"
            spec = specs.get(partition_id)
            if spec is None:
                local_errors.append(f"{label}: undeclared partition")
                continue
            if domain not in spec.get("domains", []):
                local_errors.append(f"{label}: partition is absent from protocol domain")
                continue
            partition = packet_partitions.get(partition_id)
            if partition is None:
                local_errors.append(f"{label}: requested partition is absent from packet")
                continue
            checked_partitions.add(partition_id)
            item_path = _resolve(partition["items_path"], packet_manifest_path)
            target_path = _resolve(partition.get("targets_path"), packet_manifest_path)
            emit_practice_targets = bool(protocol.get("emit_practice_targets", True))
            if item_path is None:
                local_errors.append(f"{label}: missing item file")
                continue
            if emit_practice_targets and target_path is None:
                local_errors.append(f"{label}: missing target file")
                continue
            if not emit_practice_targets and (
                    target_path is not None or partition.get("targets_sha256") is not None):
                local_errors.append(f"{label}: practice target unexpectedly emitted")
            if sha256_file(item_path) != partition.get("items_sha256"):
                local_errors.append(f"{label}: item file SHA-256 mismatch")
            if target_path and sha256_file(target_path) != partition.get("targets_sha256"):
                local_errors.append(f"{label}: target file SHA-256 mismatch")
            items = _read_jsonl(item_path)
            membership_items.extend(items)
            targets = _read_jsonl(target_path) if target_path else []
            expected_n = spec.get("n_by_domain", {}).get(domain, spec.get("n"))
            if not isinstance(expected_n, int) or isinstance(expected_n, bool) or expected_n <= 0:
                local_errors.append(f"{label}: protocol has invalid row count {expected_n!r}")
                expected_n = None
            if (expected_n is not None and partition.get("n") is not None
                    and partition.get("n") != expected_n):
                local_errors.append(f"{label}: packet row count declaration differs from protocol")
            if expected_n is not None and (len(items) != expected_n or (
                    emit_practice_targets and len(targets) != expected_n)):
                local_errors.append(f"{label}: row count differs from protocol")
            item_hashes = [row.get("text_sha256") for row in items]
            target_hashes = [row.get("text_sha256") for row in targets]
            if emit_practice_targets and item_hashes != target_hashes:
                local_errors.append(f"{label}: item and target rows are not aligned")
            if any("practice_target" in row for row in items):
                local_errors.append(f"{label}: practice target leaked into item file")
            if any("text" in row for row in targets):
                local_errors.append(f"{label}: item text leaked into target file")
            if any(text_sha256(row.get("text", "")) != row.get("text_sha256") for row in items):
                local_errors.append(f"{label}: item content hash mismatch")
            if emit_practice_targets and any(
                    not _target_valid(row.get("practice_target")) for row in targets):
                local_errors.append(f"{label}: missing or non-finite practice target")
            ordered_hash = sha256_bytes("\n".join(item_hashes).encode())
            if ordered_hash != partition.get("ordered_item_set_sha256"):
                local_errors.append(f"{label}: ordered item-set hash mismatch")
            hashes = set(item_hashes)
            groups = {row.get("source_group") for row in items}
            if len(hashes) != len(items):
                local_errors.append(f"{label}: duplicate item hash")
            if hashes & domain_hashes:
                local_errors.append(f"{label}: item reused across partitions")
            if groups & domain_groups:
                local_errors.append(f"{label}: source group reused across partitions")
            if hashes & legacy:
                local_errors.append(f"{label}: legacy probe reused")
            if hashes & prior["hashes"]:
                local_errors.append(f"{label}: prior packet item reused")
            if groups & prior["groups"]:
                local_errors.append(f"{label}: prior packet source group reused")
            allowed_splits = set(spec.get("source_split", {}).get(domain, []))
            observed_splits = {row.get("source_split") for row in items}
            if allowed_splits and not observed_splits <= allowed_splits:
                local_errors.append(f"{label}: native source split violation")
            domain_hashes.update(hashes)
            domain_groups.update(groups)
            n_items += len(items)
        if verify_source_membership:
            if dataset_path.exists() and membership_items:
                membership = _source_membership_certificate(
                    domain=domain,
                    entry=entry,
                    domain_spec=protocol_domains[domain],
                    partition_specs=[
                        specs[value] for value in selected_partition_ids
                        if value in specs
                    ],
                    packet_items=membership_items,
                )
            else:
                membership = {
                    "schema": "fresh_item_source_membership/v1",
                    "valid": False,
                    "domain": domain,
                    "n_packet_items": len(membership_items),
                    "n_matched_items": 0,
                    "errors": ["source dataset or packet items are unavailable"],
                }
            source_membership_reports.append(membership)
            if membership.get("valid") is not True:
                local_errors.append(
                    "source membership/group identity verification failed")
        if domain_hashes & all_hashes:
            local_errors.append("text hash appears in another domain")
        # Cross-domain source-group strings are not comparable; only item hashes are global.
        all_hashes.update(domain_hashes)
        all_groups.update(f"{domain}:{value}" for value in domain_groups)
        errors.extend(local_errors)
        domain_reports.append({"domain": domain, "valid": not local_errors,
                               "n_items": n_items,
                               "n_source_groups": len(domain_groups),
                               "errors": local_errors})

    return {
        "schema": "fresh_item_partition_integrity/v1", "valid": not errors,
        "packet_manifest_path": str(packet_manifest_path),
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "protocol_manifest_sha256": expected_protocol_hash,
        "n_domains": len(domain_reports), "n_items": len(all_hashes),
        "n_domain_scoped_source_groups": len(all_groups),
        "dataset_files_verified": bool(verify_dataset_files and not errors),
        "source_membership_verified": bool(verify_source_membership and not errors),
        "source_membership": source_membership_reports,
        # Authorization accepts a certificate only when ``valid`` is true, so an invalid
        # partial audit deliberately certifies no partitions even if some files happened to
        # pass.  This prevents a requested-but-empty slice from authorizing model scoring.
        "validated_partitions": sorted(checked_partitions) if not errors else [],
        "domains": domain_reports, "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--protocol", default=str(MANIFEST_PATH))
    parser.add_argument("--domains", default=None,
                        help="optional comma-separated domain subset")
    parser.add_argument("--partitions", default=None,
                        help="optional comma-separated partition subset")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--verify-source-membership", action="store_true",
        help="stream raw projected source columns and recompute row/group identity")
    args = parser.parse_args()
    domains = ({value.strip() for value in args.domains.split(",") if value.strip()}
               if args.domains else None)
    partitions = (
        {value.strip() for value in args.partitions.split(",") if value.strip()}
        if args.partitions else None
    )
    report = validate_packet(
        args.packet_manifest, protocol_path=args.protocol, domains=domains,
        partitions=partitions,
        verify_source_membership=args.verify_source_membership,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
