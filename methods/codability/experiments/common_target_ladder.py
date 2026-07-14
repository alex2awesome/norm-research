#!/usr/bin/env python
"""Validate and summarize a scale ladder measured against one fixed target policy.

Pairwise larger-reader targets cannot be added: ``D(1B -> 3B; target=3B)`` and
``D(3B -> 8B; target=8B)`` are debts for different random variables.  This module rejects that
mistake mechanically.  It also refuses a triangle/potential calculation when the reported cost is
legacy word count rather than a composable certified-unit measure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "common_target_articulation_ladder/v1"
POLICY_EXECUTOR_SCHEMA = "fixed_target_policy_executor_ladder/v4"
POLICY_RESPONSE_SCHEMA = "fixed_target_policy_executor_response_surface/v4"
POLICY_CROSSFOLD_SCHEMAS = frozenset({
    "crossfold_policy_isomorphism_fibers/v4",
    "crossfold_policy_isomorphism_fibers/v5",
})
POLICY_SOURCE_SCHEMAS = frozenset({
    "policy_isomorphism_experiment/v4",
    "policy_isomorphism_experiment/v5",
})
POLICY_FIXED_CONFIG_KEYS = (
    "big_job",
    "target_arm_id",
    "mae_margin",
    "rho_margin",
    "flip_margin",
    "bias_margin",
    "functional_rho_floor",
)

# ``gi`` is a materialization-local legacy row number.  It is not a metric identity once the
# hierarchy is expanded across R1/R2/R3.  Breadth artifacts therefore use the frozen cell id plus
# the source node coordinates below; legacy H49 reports continue to use (domain, gi, construct).
BREADTH_CELL_IDENTITY_KEYS = ("task", "level", "bucket", "node_id")
BREADTH_CELL_IDENTITY_TRIGGERS = ("level", "bucket", "node_id", "metric_id")


def uses_breadth_cell_identity(cell: Mapping) -> bool:
    """Return whether a cell declares any hierarchy-specific identity coordinate.

    ``task`` alone is deliberately not a trigger because newly scored legacy H49 rows also carry
    their canonical task name while retaining the frozen legacy cell schema.
    """
    return any(cell.get(key) is not None for key in BREADTH_CELL_IDENTITY_TRIGGERS)


def policy_cell_identity(cell: Mapping, *, context: str = "cell") -> dict:
    """Extract one lossless policy-cell identity with a legacy-compatible fallback."""
    cell_id = cell.get("cell_id", cell.get("id"))
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError(f"{context} has a missing or invalid cell_id")
    result = {"cell_id": cell_id}
    for key in ("domain", "construct"):
        value = cell.get(key)
        if value is not None:
            result[key] = value
    if uses_breadth_cell_identity(cell):
        missing = [
            key for key in BREADTH_CELL_IDENTITY_KEYS
            if not isinstance(cell.get(key), str) or not cell.get(key)
        ]
        if missing:
            raise ValueError(
                f"{context}/{cell_id} has partial breadth identity; missing={missing}"
            )
        result.update({key: cell[key] for key in BREADTH_CELL_IDENTITY_KEYS})
        if cell.get("metric_id") is not None:
            if not isinstance(cell["metric_id"], str) or not cell["metric_id"]:
                raise ValueError(f"{context}/{cell_id} has an invalid metric_id")
            result["metric_id"] = cell["metric_id"]
        # Retain the source-local number for auditability, but never use it as a breadth key.
        if cell.get("gi") is not None:
            result["gi"] = cell["gi"]
        result["identity_mode"] = "hierarchy_node"
    else:
        if cell.get("gi") is not None:
            result["gi"] = cell["gi"]
        result["identity_mode"] = "legacy_domain_gi"
    return result


def validate_policy_cell_panel(cells: Sequence[Mapping], *, context: str) -> dict:
    """Reject duplicate ids and duplicate hierarchy-node coordinates without consulting ``gi``."""
    identities = [policy_cell_identity(cell, context=context) for cell in cells]
    cell_ids = [identity["cell_id"] for identity in identities]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError(f"{context} has duplicate cell ids")
    breadth = [
        identity for identity in identities
        if identity["identity_mode"] == "hierarchy_node"
    ]
    breadth_keys = [
        tuple(identity[key] for key in BREADTH_CELL_IDENTITY_KEYS)
        for identity in breadth
    ]
    if len(breadth_keys) != len(set(breadth_keys)):
        raise ValueError(f"{context} has duplicate breadth node identities")
    return {
        "valid": True,
        "n_cells": len(identities),
        "n_breadth_cells": len(breadth),
        "n_legacy_cells": len(identities) - len(breadth),
        "breadth_key": ["cell_id", *BREADTH_CELL_IDENTITY_KEYS],
        "legacy_key": ["cell_id", "domain", "gi"],
    }


def require_same_policy_cell_identity(
    cells: Sequence[Mapping], *, context: str, extra_keys: Sequence[str] = ()
) -> dict:
    """Require repeated reports for one cell id to preserve all identity coordinates."""
    if not cells:
        raise ValueError(f"{context} has no cells")
    identities = [policy_cell_identity(cell, context=context) for cell in cells]
    modes = {identity["identity_mode"] for identity in identities}
    if len(modes) != 1:
        raise ValueError(f"{context} mixes legacy and breadth cell identity")
    required = (
        ("domain", "construct", *BREADTH_CELL_IDENTITY_KEYS)
        if next(iter(modes)) == "hierarchy_node"
        else ("domain", "construct")
    )
    optional = (
        ("metric_id", "gi")
        if next(iter(modes)) == "hierarchy_node" else ("gi",)
    )
    for key in (*required, *extra_keys):
        values = {cell.get(key) for cell in cells}
        if None in values or len(values) != 1:
            raise ValueError(f"{context} changes or omits identity field {key!r}")
    for key in optional:
        values = {cell.get(key) for cell in cells}
        if values != {None} and (None in values or len(values) != 1):
            raise ValueError(f"{context} changes or omits identity field {key!r}")
    if len({identity["cell_id"] for identity in identities}) != 1:
        raise ValueError(f"{context} changes cell_id")
    return identities[0]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cell_binary_protocol(cell: Mapping, *, context: str) -> str | None:
    """Return one declared binary protocol, retaining compatibility with legacy v4 reports."""
    target = cell.get("target_binary_readout")
    small = cell.get("small_binary_readout")
    comparator = cell.get("scale_comparator_binary_readout")
    if target is None and small is None and comparator is None:
        return None
    if target is None or small is None or target != small:
        raise ValueError(f"binary readout identity fails for {context}")
    if comparator is not None and comparator != target:
        raise ValueError(f"scale-comparator binary readout identity fails for {context}")
    return str(target)


def _ladder_row_key(domain: str, row: Mapping) -> tuple:
    """Build a non-colliding row key while retaining old (domain, gi) semantics."""
    if row.get("cell_id") is not None:
        return ("cell_id", domain, str(row["cell_id"]))
    if uses_breadth_cell_identity(row):
        identity = policy_cell_identity(
            {**row, "cell_id": row.get("node_id")},
            context=f"fixed-target ladder/{domain}",
        )
        return (
            "hierarchy_node",
            identity["task"],
            identity["level"],
            identity["bucket"],
            identity["node_id"],
        )
    return ("legacy_domain_gi", domain, int(row["gi"]))


def _ladder_key_description(key: tuple) -> dict:
    if key[0] == "cell_id":
        return {"identity_mode": key[0], "domain": key[1], "cell_id": key[2]}
    if key[0] == "hierarchy_node":
        return {
            "identity_mode": key[0],
            "task": key[1],
            "level": key[2],
            "bucket": key[3],
            "node_id": key[4],
        }
    return {"identity_mode": key[0], "domain": key[1], "gi": key[2]}


def _ladder_key_domain(key: tuple) -> str:
    # Breadth task ids are the canonical domain labels used by the breadth packet.
    return str(key[1])


def _rows(artifact: Mapping) -> dict[tuple, Mapping]:
    rows = {}
    for domain, result in artifact.get("by_domain", {}).items():
        for row in result.get("per_metric", []):
            if ("error" not in row and "ineligible" not in row
                    and row.get("heldout", {}).get("valid")):
                key = _ladder_row_key(domain, row)
                if key in rows:
                    raise ValueError(
                        "fixed-target ladder contains duplicate metric identity: "
                        f"{_ladder_key_description(key)}"
                    )
                rows[key] = row
    return rows


def validate_common_target(artifacts: Sequence[Mapping]) -> dict:
    if len(artifacts) < 2:
        raise ValueError("a ladder needs at least two hop artifacts")
    schemas = {artifact.get("schema") for artifact in artifacts}
    allowed = {"fixed_target_name_substitution/v1", "fixed_target_name_substitution/v2"}
    if not schemas or not schemas.issubset(allowed):
        raise ValueError(f"unexpected artifact schemas: {sorted(map(str, schemas))}")
    target_tags = {artifact.get("config", {}).get("target_tag") for artifact in artifacts}
    if len(target_tags) != 1 or None in target_tags:
        raise ValueError(f"ladder hops do not declare one common target tag: {target_tags}")

    maps = [_rows(artifact) for artifact in artifacts]
    common_cells = set(maps[0])
    for rows in maps[1:]:
        common_cells &= set(rows)
    if not common_cells:
        raise ValueError("ladder artifacts have no common metric cells")
    mismatches = []
    for cell in sorted(common_cells):
        target_ids = {rows[cell]["target"]["target_id"] for rows in maps}
        split_seeds = {rows[cell]["probe_split"]["seed"] for rows in maps}
        heldout_sizes = {rows[cell]["probe_split"]["n_heldout"] for rows in maps}
        if len(target_ids) != 1 or len(split_seeds) != 1 or len(heldout_sizes) != 1:
            mismatches.append({**_ladder_key_description(cell),
                               "target_ids": sorted(target_ids),
                               "split_seeds": sorted(split_seeds),
                               "heldout_sizes": sorted(heldout_sizes)})
    if mismatches:
        raise ValueError(f"common-target/split validation failed for {len(mismatches)} cells")

    domain_target_hashes = {}
    for artifact in artifacts:
        for domain, result in artifact.get("by_domain", {}).items():
            target_input = result.get("inputs", {}).get("target_grid", {})
            digest = target_input.get("sha256")
            if digest:
                domain_target_hashes.setdefault(domain, set()).add(digest)
    bad_hashes = {domain: sorted(values) for domain, values in domain_target_hashes.items()
                  if len(values) != 1}
    if bad_hashes:
        raise ValueError(f"target grid hashes differ across hops: {bad_hashes}")
    return {
        "valid": True,
        "input_schemas": sorted(schemas),
        "target_tag": next(iter(target_tags)),
        "n_common_cells": len(common_cells),
        "common_cells_by_domain": {
            domain: sum(_ladder_key_domain(cell) == domain for cell in common_cells)
            for domain in sorted({_ladder_key_domain(cell) for cell in common_cells})},
        "target_grid_sha256_by_domain": {
            domain: next(iter(values)) for domain, values in sorted(domain_target_hashes.items())},
        "same_target_id_split_seed_and_heldout_size": True,
    }


def summarize_hop(artifact: Mapping) -> dict:
    rows = list(_rows(artifact).values())
    gaps = [row for row in rows if row["heldout"]["gates"]["baseline_gap_confirmed"]]
    gates = ("articulation_improvement_confirmed", "noninferior_to_big_sparse",
             "equivalent_to_big_sparse", "signature_improved", "signature_noninferior_to_big")
    return {
        "small_reader": artifact["config"]["small_tag"],
        "big_reader": artifact["config"]["big_tag"],
        "target_reader": artifact["config"]["target_tag"],
        "n_evaluable": len(rows),
        "n_confirmed_baseline_gaps": len(gaps),
        "gate_success_among_confirmed_gaps": {
            gate: sum(row["heldout"]["gates"][gate] for row in gaps) for gate in gates},
        "methodological_substitution": sum(
            row["heldout"]["methodological_substitution"] for row in gaps),
        "equivalent_methodological_substitution": sum(
            row["heldout"]["equivalent_methodological_substitution"] for row in gaps),
        "debt_status": {
            "finite": sum(row["heldout"]["methodological_substitution"] for row in gaps),
            "right_censored_within_bank": sum(
                not row["heldout"]["methodological_substitution"] for row in gaps),
            "cost_basis": "legacy message words; not certified articulation units",
        },
    }


def build_ladder_report(artifacts: Sequence[Mapping], *, labels: Sequence[str] | None = None) -> dict:
    validation = validate_common_target(artifacts)
    if labels is not None and len(labels) != len(artifacts):
        raise ValueError("labels must align with artifacts")
    maps = [_rows(artifact) for artifact in artifacts]
    common = set.intersection(*(set(rows) for rows in maps))
    all_finite = [cell for cell in common if all(
        rows[cell]["heldout"]["methodological_substitution"] for rows in maps)]
    hops = {}
    for index, artifact in enumerate(artifacts):
        label = labels[index] if labels is not None else f"hop_{index}"
        hops[label] = summarize_hop(artifact)
    return {
        "schema": SCHEMA,
        "validation": validation,
        "hops": hops,
        "potential_test": {
            "n_common_cells_finite_on_every_hop": len(all_finite),
            "triangle_evaluable": False,
            "reasons": (["no common cell has finite debt on every hop"] if not all_finite else [])
                       + ["legacy word-count costs are not composable certified units"],
            "status": "right_censored_and_cost_basis_ineligible",
        },
        "claim_scope": ("The report validates a common target and summarizes held-out hop results. "
                        "It does not turn legacy words into articulation units or infer a scalar "
                        "potential from censored debts."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_policy_executor_ladder(
        crossfold_paths: Sequence[str | Path], *, labels: Sequence[str] | None = None) -> dict:
    """Compare executor scales only after proving target, item, prompt, and readout identity."""
    if len(crossfold_paths) < 2:
        raise ValueError("a fixed-target executor ladder needs at least two crossfold reports")
    if labels is not None and len(labels) != len(crossfold_paths):
        raise ValueError("labels must align with crossfold reports")
    entries = []
    for index, path_value in enumerate(crossfold_paths):
        path = Path(path_value)
        crossfold = json.loads(path.read_text())
        if crossfold.get("schema") not in POLICY_CROSSFOLD_SCHEMAS:
            raise ValueError(f"unexpected policy crossfold schema in {path}")
        arm_bank_sha256 = crossfold.get("arm_bank_sha256")
        if not arm_bank_sha256:
            raise ValueError(f"crossfold report does not declare an arm bank in {path}")
        crossfold_floor = crossfold.get("functional_rho_floor")
        if crossfold_floor is None:
            raise ValueError(f"crossfold report does not declare a functional floor in {path}")
        crossfold_cells = crossfold.get("cells")
        if not isinstance(crossfold_cells, list) or not crossfold_cells:
            raise ValueError(f"crossfold report has no cells in {path}")
        validate_policy_cell_panel(
            crossfold_cells, context=f"crossfold report {path}"
        )
        crossfold_cell_ids = [cell.get("cell_id") for cell in crossfold_cells]
        if None in crossfold_cell_ids or len(set(crossfold_cell_ids)) != len(
                crossfold_cell_ids):
            raise ValueError(f"crossfold report has missing or duplicate cell ids in {path}")
        crossfold_cell_map = {cell["cell_id"]: cell for cell in crossfold_cells}

        references = crossfold.get("reports")
        if not isinstance(references, list) or not references:
            raise ValueError(f"crossfold report has no source reports in {path}")
        source_reports = []
        source_partitions = []
        for reference in references:
            source_path = Path(reference["path"])
            if _sha256(source_path) != reference["sha256"]:
                raise ValueError(f"crossfold source report changed: {source_path}")
            report = json.loads(source_path.read_text())
            if report.get("schema") not in POLICY_SOURCE_SCHEMAS:
                raise ValueError(f"unexpected source policy schema in {source_path}")
            partition = report.get("partition")
            if reference.get("partition") != partition:
                raise ValueError(
                    f"crossfold reference/source partition identity fails in {source_path}"
                )
            if partition in source_partitions:
                raise ValueError(f"duplicate source partition {partition!r} in {path}")
            source_partitions.append(partition)
            if report.get("arm_bank_sha256") != arm_bank_sha256:
                raise ValueError(
                    f"source/crossfold arm bank identity fails in {source_path}"
                )
            config = report.get("config", {})
            required_config = ("small_job", *POLICY_FIXED_CONFIG_KEYS)
            missing_config = [key for key in required_config if config.get(key) is None]
            if missing_config:
                raise ValueError(
                    f"source report lacks required config {missing_config} in {source_path}"
                )
            if float(config["functional_rho_floor"]) != float(crossfold_floor):
                raise ValueError(
                    f"source/crossfold functional floor identity fails in {source_path}"
                )
            source_cells = report.get("cells")
            if not isinstance(source_cells, list) or not source_cells:
                raise ValueError(f"source report has no cells in {source_path}")
            validate_policy_cell_panel(
                source_cells, context=f"source policy report {source_path}"
            )
            source_cell_ids = [cell.get("cell_id") for cell in source_cells]
            if None in source_cell_ids or len(set(source_cell_ids)) != len(source_cell_ids):
                raise ValueError(
                    f"source report has missing or duplicate cell ids in {source_path}"
                )
            if set(source_cell_ids) != set(crossfold_cell_ids):
                raise ValueError(
                    f"source/crossfold cell identity fails for partition {partition!r}"
                )
            source_reports.append(report)

        configs = [report["config"] for report in source_reports]
        executor_jobs = {config["small_job"] for config in configs}
        if len(executor_jobs) != 1:
            raise ValueError("one crossfold report mixes executor small_job values")
        for key in POLICY_FIXED_CONFIG_KEYS:
            if len({config[key] for config in configs}) != 1:
                raise ValueError(f"one crossfold report mixes config {key!r}")

        source_maps = {
            report["partition"]: {cell["cell_id"]: cell for cell in report["cells"]}
            for report in source_reports
        }
        readout_hashes = set()
        binary_protocols = set()
        for cell_id, crossfold_cell in crossfold_cell_map.items():
            source_cells = [source_map[cell_id] for source_map in source_maps.values()]
            # Retain the established error surface for legacy reports, then apply the richer
            # hierarchy identity contract below.
            for key in ("domain", "construct"):
                values = {cell.get(key) for cell in source_cells}
                if None in values or len(values) != 1:
                    raise ValueError(
                        f"source cell {key} identity fails for {cell_id!r} in {path}"
                    )
            require_same_policy_cell_identity(
                source_cells,
                context=f"source cell {cell_id!r} in {path}",
                extra_keys=("target_job",),
            )
            for partition, cell in zip(source_maps, source_cells):
                if cell.get("small_job") != next(iter(executor_jobs)):
                    raise ValueError(
                        f"source cell small_job identity fails for {cell_id!r} in {path}"
                    )
                if cell.get("target_job") != configs[0]["big_job"]:
                    raise ValueError(
                        f"source cell target_job identity fails for {cell_id!r} in {path}"
                    )
                if not cell.get("executor_prompt_bank_validation", {}).get("valid"):
                    raise ValueError(
                        f"prompt bank was not hash-validated for "
                        f"{partition}/{cell_id}"
                    )
                target_readout = cell.get("target_readout_template_sha256")
                small_readout = cell.get("small_readout_template_sha256")
                if target_readout is None or target_readout != small_readout:
                    raise ValueError(
                        f"readout identity fails for {partition}/{cell_id}"
                    )
                readout_hashes.add(target_readout)
                binary_protocols.add(_cell_binary_protocol(
                    cell, context=f"{partition}/{cell_id}"))
            if (uses_breadth_cell_identity(crossfold_cell)
                    or uses_breadth_cell_identity(source_cells[0])):
                require_same_policy_cell_identity(
                    [crossfold_cell, source_cells[0]],
                    context=f"source/crossfold cell {cell_id!r}",
                )
            else:
                for key in ("domain", "construct"):
                    if crossfold_cell.get(key) != source_cells[0].get(key):
                        raise ValueError(
                            f"source/crossfold cell {key} identity fails for {cell_id!r}"
                        )
            capacity_rows = crossfold_cell.get("functional_capacity_by_arm", [])
            capacity_arm_ids = [row.get("arm_id") for row in capacity_rows]
            if (None in capacity_arm_ids
                    or len(set(capacity_arm_ids)) != len(capacity_arm_ids)):
                raise ValueError(
                    f"crossfold cell has missing or duplicate arm ids for {cell_id!r}"
                )
            common_arms = crossfold_cell.get("common_arms")
            if not isinstance(common_arms, list) or set(capacity_arm_ids) != set(common_arms):
                raise ValueError(
                    f"crossfold arm identity fails for cell {cell_id!r}"
                )
            control_arms = crossfold_cell.get(
                "control_arms_excluded_from_membership", [])
            if (not isinstance(control_arms, list)
                    or set(control_arms).intersection(common_arms)):
                raise ValueError(
                    f"crossfold control-arm identity fails for cell {cell_id!r}"
                )
            all_reported_arms = set(common_arms) | set(control_arms)
            for partition, cell in zip(source_maps, source_cells):
                source_rows = cell.get("rows", [])
                source_arm_ids = [row.get("arm_id") for row in source_rows]
                if (None in source_arm_ids
                        or len(set(source_arm_ids)) != len(source_arm_ids)
                        or set(source_arm_ids) != all_reported_arms):
                    raise ValueError(
                        f"source/crossfold arm identity fails for "
                        f"{partition!r}/{cell_id!r}"
                    )
            for row in capacity_rows:
                fold_partitions = [fold.get("partition") for fold in row.get("folds", [])]
                if (len(set(fold_partitions)) != len(fold_partitions)
                        or set(fold_partitions) != set(source_partitions)):
                    raise ValueError(
                        f"crossfold fold/partition identity fails for "
                        f"{cell_id!r}/{row.get('arm_id')!r}"
                    )
        if len(readout_hashes) != 1:
            raise ValueError(f"one crossfold report mixes readout templates in {path}")
        if len(binary_protocols) != 1:
            raise ValueError(f"one crossfold report mixes binary readout protocols in {path}")

        entries.append({
            "label": labels[index] if labels is not None else next(iter(executor_jobs)),
            "path": str(path),
            "sha256": _sha256(path),
            "crossfold": crossfold,
            "sources": source_reports,
            "source_by_partition": source_maps,
            "crossfold_cell_by_id": crossfold_cell_map,
            "executor_job": next(iter(executor_jobs)),
            "config": {key: configs[0][key] for key in POLICY_FIXED_CONFIG_KEYS},
            "readout_template_sha256": next(iter(readout_hashes)),
            "binary_readout": next(iter(binary_protocols)),
        })

    if len({entry["label"] for entry in entries}) != len(entries):
        raise ValueError("executor ladder labels must be unique")
    if len({entry["crossfold"]["arm_bank_sha256"] for entry in entries}) != 1:
        raise ValueError("executor ladder arm banks differ")
    for key in POLICY_FIXED_CONFIG_KEYS:
        if len({entry["config"][key] for entry in entries}) != 1:
            raise ValueError(f"executor ladder config {key!r} differs")
    if len({entry["readout_template_sha256"] for entry in entries}) != 1:
        raise ValueError("executor ladder readout templates differ")
    if len({entry["binary_readout"] for entry in entries}) != 1:
        raise ValueError("executor ladder binary readout protocols differ")
    partition_sets = [{report["partition"] for report in entry["sources"]} for entry in entries]
    if any(values != partition_sets[0] for values in partition_sets[1:]):
        raise ValueError("executor ladder partition sets differ")
    cell_sets = [{cell["cell_id"] for cell in entry["crossfold"]["cells"]}
                 for entry in entries]
    if any(values != cell_sets[0] for values in cell_sets[1:]):
        raise ValueError("executor ladder cell sets differ")

    source_by_entry = [entry["source_by_partition"] for entry in entries]
    for partition in sorted(partition_sets[0]):
        for cell_id in sorted(cell_sets[0]):
            cells = [source_map[partition][cell_id] for source_map in source_by_entry]
            require_same_policy_cell_identity(
                cells,
                context=f"source partition/cell identity for {partition}/{cell_id}",
                extra_keys=("target_job",),
            )
            if len({tuple(cell["target_shards"]) for cell in cells}) != 1:
                raise ValueError(
                    f"fixed target shard identity fails for {partition}/{cell_id}")
            if len({cell["n_items"] for cell in cells}) != 1:
                raise ValueError(f"item counts differ for {partition}/{cell_id}")
            readout_hashes = {
                value for cell in cells for value in (
                    cell.get("target_readout_template_sha256"),
                    cell.get("small_readout_template_sha256"),
                )
            }
            if None in readout_hashes or len(readout_hashes) != 1:
                raise ValueError(f"readout identity fails for {partition}/{cell_id}")
            binary_protocols = {
                _cell_binary_protocol(cell, context=f"{partition}/{cell_id}")
                for cell in cells
            }
            if len(binary_protocols) != 1:
                raise ValueError(
                    f"binary readout identity fails for {partition}/{cell_id}")

    floor = float(entries[0]["config"]["functional_rho_floor"])
    cell_rows = []
    for cell_id in sorted(cell_sets[0]):
        crossfold_cells = [
            {cell["cell_id"]: cell for cell in entry["crossfold"]["cells"]}[cell_id]
            for entry in entries
        ]
        capacity_maps = [{row["arm_id"]: row for row in cell["functional_capacity_by_arm"]}
                         for cell in crossfold_cells]
        common_arms = sorted(set.intersection(*(set(rows) for rows in capacity_maps)))
        per_arm = []
        for arm_id in common_arms:
            executor_values = {}
            for entry, capacity_map in zip(entries, capacity_maps):
                row = capacity_map[arm_id]
                fold_rhos = [fold["adverse_rho_point"] for fold in row["folds"]]
                fold_maes = [fold["adverse_mae_tvd"] for fold in row["folds"]]
                fold_gains = [fold.get("mae_gain_over_small_sparse") for fold in row["folds"]]
                fold_rank_gains = [fold.get("rho_gain_over_small_sparse")
                                   for fold in row["folds"]]
                fold_rank_closure = [fold.get("fraction_rank_scale_gap_closed")
                                     for fold in row["folds"]]
                fold_mae_closure = [fold.get("fraction_mae_scale_gap_closed")
                                    for fold in row["folds"]]
                executor_values[entry["label"]] = {
                    "worst_fold_adverse_rho": float(min(fold_rhos)),
                    "worst_fold_adverse_mae_tvd": float(max(fold_maes)),
                    "descriptive_observed_worst_fold_mae_gain_over_name": (
                        None if any(value is None for value in fold_gains)
                        else float(min(fold_gains))
                    ),
                    "descriptive_observed_worst_fold_rho_gain_over_name": (
                        None if any(value is None for value in fold_rank_gains)
                        else float(min(fold_rank_gains))
                    ),
                    "descriptive_observed_worst_fold_fraction_rank_target_self_gap_closed": (
                        None if any(value is None for value in fold_rank_closure)
                        else float(min(fold_rank_closure))
                    ),
                    "descriptive_observed_worst_fold_fraction_mae_target_self_gap_closed": (
                        None if any(value is None for value in fold_mae_closure)
                        else float(min(fold_mae_closure))
                    ),
                    "observed_worst_fold_functional_rank_capacity": row[
                        "stable_observed_max_rho_floor"],
                    "certified_rank_capacity": row["stable_certified_max_rho_floor"],
                }
            per_arm.append({
                "arm_id": arm_id,
                "components": capacity_maps[0][arm_id]["components"],
                "executors": executor_values,
            })

        executor_summaries = {}
        for entry, crossfold_cell, source_map in zip(
                entries, crossfold_cells, source_by_entry):
            source_cells = []
            for partition in sorted(partition_sets[0]):
                source_cells.append(source_map[partition][cell_id])
            sparse_points = [cell["rows"][0]["certificate"]["small_sparse_point"][
                "candidate_robust"] for cell in source_cells]
            target_points = [cell["rows"][0]["certificate"]["point"][
                "target_self_robust"] for cell in source_cells]
            capacities = [
                row["stable_observed_max_rho_floor"]
                for row in crossfold_cell["functional_capacity_by_arm"]
                if row["stable_observed_max_rho_floor"] is not None
            ]
            reported_profile = next(
                row for row in crossfold_cell["functional_floor_profile"]
                if abs(row["rho_floor"] - floor) < 1e-12
            )
            executor_summaries[entry["label"]] = {
                "executor_job": entry["executor_job"],
                "name_only_worst_fold": {
                    "adverse_rho": float(min(row["spearman"] for row in sparse_points)),
                    "adverse_mae_tvd": float(max(row["mae_tvd"] for row in sparse_points)),
                },
                "target_self_worst_fold": {
                    "adverse_rho": float(min(row["spearman"] for row in target_points)),
                    "adverse_mae_tvd": float(max(row["mae_tvd"] for row in target_points)),
                },
                "best_observed_worst_fold_functional_rank_capacity": max(
                    capacities, default=None),
                "observed_functional_members_at_floor": reported_profile[
                    "observed_functional_members"],
                "component_minimal_members_at_floor": reported_profile[
                    "observed_component_minimal_members"],
            }
        cell_rows.append({
            **{
                key: value
                for key, value in policy_cell_identity(
                    crossfold_cells[0], context=f"executor ladder/{cell_id}"
                ).items()
                if key != "identity_mode"
            },
            "executors": executor_summaries,
            "per_arm": per_arm,
        })

    return {
        "schema": POLICY_EXECUTOR_SCHEMA,
        "estimand": "same-prompt executor-scale reconstruction of one fixed target policy",
        "validation": {
            "valid": True,
            "target_job": entries[0]["config"]["big_job"],
            "target_arm_id": entries[0]["config"]["target_arm_id"],
            "identity_margins": {
                key: entries[0]["config"][key]
                for key in ("mae_margin", "rho_margin", "flip_margin", "bias_margin")
            },
            "partitions": sorted(partition_sets[0]),
            "n_cells": len(cell_rows),
            "arm_bank_sha256": entries[0]["crossfold"]["arm_bank_sha256"],
            "same_target_shards": True,
            "same_item_partitions": True,
            "same_prompt_hash_bank": True,
            "same_readout_template": True,
            "same_binary_readout_protocol": True,
            "binary_readout": entries[0]["binary_readout"],
            "source_reports_bound_to_crossfold_bank": True,
            "same_target_arm": True,
            "same_scoring_margins": True,
            "same_partition_cell_identity": True,
        },
        "functional_rho_floor": floor,
        "inputs": [{"label": entry["label"], "executor_job": entry["executor_job"],
                    "path": entry["path"], "sha256": entry["sha256"]}
                   for entry in entries],
        "cells": cell_rows,
        "gap_closure_scope": (
            "Gap-closure fields are descriptive point estimates, reduced to the adverse "
            "(minimum-benefit) observed fold. They are neither confidence-bound certificates "
            "nor evidence that a scalar scale gap exists outside the fixed target channel."
        ),
        "claim_boundary": (
            "This fixes target policy, items, articulation text, forms, and readout across "
            "executors. It identifies an executor contrast, not a pure parameter law when model "
            "versions, training, or quantization differ; public folds remain exploratory."
        ),
    }


def _candidate_robust(row: Mapping) -> Mapping:
    return row["certificate"]["point"]["candidate_robust"]


def _sparse_robust(row: Mapping) -> Mapping:
    return row["certificate"]["small_sparse_point"]["candidate_robust"]


def _candidate_point(row: Mapping) -> Mapping:
    return row["certificate"]["point"]


def _sparse_point(row: Mapping) -> Mapping:
    return row["certificate"]["small_sparse_point"]


def build_policy_executor_response_surface(
        report_paths: Sequence[str | Path], *, labels: Sequence[str] | None = None) -> dict:
    """Build a point-estimate scale/articulation surface for one fixed target and prompt bank.

    Input order declares executor order.  Unlike :func:`build_policy_executor_ladder`, this mode
    accepts one source report per executor on one common partition, which permits an exact saved
    rung to enter even when its reciprocal public fold was never executed.  It deliberately does
    not turn the resulting point comparisons into interval certificates.
    """
    if len(report_paths) < 2:
        raise ValueError("a fixed-target response surface needs at least two executor reports")
    if labels is not None and len(labels) != len(report_paths):
        raise ValueError("labels must align with executor reports")

    entries = []
    for index, path_value in enumerate(report_paths):
        path = Path(path_value)
        report = json.loads(path.read_text())
        if report.get("schema") not in POLICY_SOURCE_SCHEMAS:
            raise ValueError(f"unexpected source policy schema in {path}")
        config = report.get("config", {})
        required = ("small_job", *POLICY_FIXED_CONFIG_KEYS)
        missing = [key for key in required if config.get(key) is None]
        if missing:
            raise ValueError(f"source report lacks required config {missing} in {path}")
        cells = report.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError(f"source report has no cells in {path}")
        validate_policy_cell_panel(
            cells, context=f"executor response-surface source {path}"
        )
        cell_ids = [cell.get("cell_id") for cell in cells]
        if None in cell_ids or len(cell_ids) != len(set(cell_ids)):
            raise ValueError(f"source report has missing or duplicate cells in {path}")
        label = labels[index] if labels is not None else config["small_job"]
        entries.append({
            "label": label,
            "path": str(path),
            "sha256": _sha256(path),
            "report": report,
            "config": config,
            "cells": {cell["cell_id"]: cell for cell in cells},
        })
    if len({entry["label"] for entry in entries}) != len(entries):
        raise ValueError("executor response-surface labels must be unique")
    if len({entry["report"].get("partition") for entry in entries}) != 1:
        raise ValueError("executor response-surface partitions differ")
    if len({entry["report"].get("arm_bank_sha256") for entry in entries}) != 1:
        raise ValueError("executor response-surface arm banks differ")
    for key in POLICY_FIXED_CONFIG_KEYS:
        if len({entry["config"][key] for entry in entries}) != 1:
            raise ValueError(f"executor response-surface config {key!r} differs")
    cell_sets = [set(entry["cells"]) for entry in entries]
    if any(values != cell_sets[0] for values in cell_sets[1:]):
        raise ValueError("executor response-surface cell panels differ")

    rho_margin = float(entries[0]["config"]["rho_margin"])
    mae_margin = float(entries[0]["config"]["mae_margin"])
    flip_margin = float(entries[0]["config"]["flip_margin"])
    bias_margin = float(entries[0]["config"]["bias_margin"])
    functional_floor = float(entries[0]["config"]["functional_rho_floor"])
    cell_rows = []
    surface_binary_protocols = set()
    for cell_id in sorted(cell_sets[0]):
        cells = [entry["cells"][cell_id] for entry in entries]
        try:
            require_same_policy_cell_identity(
                cells,
                context=f"executor response-surface cell {cell_id}",
                extra_keys=("target_job", "n_items"),
            )
        except ValueError as exc:
            # Keep this high-level prefix stable for callers while retaining the precise field in
            # the shared identity validator's message.
            raise ValueError(
                f"executor response-surface cell {cell_id} changes identity: {exc}"
            ) from exc
        if not all(cell.get("executor_prompt_bank_validation", {}).get("valid")
                   for cell in cells):
            raise ValueError(f"executor response-surface prompt validation fails for {cell_id}")
        for entry, cell in zip(entries, cells):
            if cell.get("small_job") != entry["config"]["small_job"]:
                raise ValueError(
                    f"executor response-surface small-job identity fails for {cell_id}")
            if cell.get("target_job") != entry["config"]["big_job"]:
                raise ValueError(
                    f"executor response-surface target-job identity fails for {cell_id}")
        if len({tuple(cell.get("target_shards", ())) for cell in cells}) != 1:
            raise ValueError(f"executor response-surface target shards differ for {cell_id}")
        readouts = {
            value for cell in cells for value in (
                cell.get("small_readout_template_sha256"),
                cell.get("target_readout_template_sha256"),
            )
        }
        if None in readouts or len(readouts) != 1:
            raise ValueError(f"executor response-surface readouts differ for {cell_id}")
        binary_protocols = {
            _cell_binary_protocol(cell, context=f"response-surface/{cell_id}")
            for cell in cells
        }
        if len(binary_protocols) != 1:
            raise ValueError(
                f"executor response-surface binary readouts differ for {cell_id}")
        surface_binary_protocols.update(binary_protocols)

        row_maps = []
        for cell in cells:
            arm_ids = [row.get("arm_id") for row in cell.get("rows", [])]
            if not arm_ids or None in arm_ids:
                raise ValueError("executor response-surface has an empty/missing arm panel")
            if len(arm_ids) != len(set(arm_ids)):
                raise ValueError(
                    f"executor response-surface has duplicate arms for {cell_id}")
            row_maps.append({row["arm_id"]: row for row in cell["rows"]})
        arm_sets = [set(rows) for rows in row_maps]
        if any(values != arm_sets[0] for values in arm_sets[1:]):
            raise ValueError(f"executor response-surface arm panels differ for {cell_id}")
        control_provenances = {"wrong_construct_control", "inert_length_control"}
        eligible_arm_ids = sorted(
            arm_id for arm_id in arm_sets[0]
            if not row_maps[0][arm_id].get("control_for")
            and row_maps[0][arm_id].get("provenance") not in control_provenances
        )
        if not eligible_arm_ids:
            raise ValueError(
                f"executor response-surface has no eligible articulation arms for {cell_id}"
            )
        control_arm_ids = sorted(set(arm_sets[0]) - set(eligible_arm_ids))
        eligible_arm_set = set(eligible_arm_ids)
        executor_rows = {}
        for entry, rows in zip(entries, row_maps):
            for arm_id in sorted(arm_sets[0]):
                metadata = [(
                    tuple(row_map[arm_id].get("components", ())),
                    row_map[arm_id].get("channel"),
                    row_map[arm_id].get("provenance"),
                    row_map[arm_id].get("control_for"),
                    row_map[arm_id].get("semantic_content_word_count"),
                ) for row_map in row_maps]
                if len(set(metadata)) != 1:
                    raise ValueError(
                        f"executor response-surface changes arm metadata for {cell_id}/{arm_id}")
            sparse_values = [_sparse_robust(row) for row in rows.values()]
            if any(value != sparse_values[0] for value in sparse_values[1:]):
                raise ValueError(
                    f"executor name baseline changes by arm for {entry['label']}/{cell_id}")
            sparse = sparse_values[0]
            sparse_points = [_sparse_point(row) for row in rows.values()]
            if any(value != sparse_points[0] for value in sparse_points[1:]):
                raise ValueError(
                    f"executor name point changes by arm for {entry['label']}/{cell_id}")
            executor_rows[entry["label"]] = {
                "executor_job": entry["config"]["small_job"],
                "name_only": sparse,
                "name_only_forms": sparse_points[0]["candidate_forms"],
                "name_only_quotient": sparse_points[0]["quotient"],
                "arms": {
                    arm_id: {
                        "candidate": _candidate_robust(row),
                        "candidate_forms": _candidate_point(row)["candidate_forms"],
                        "candidate_quotient": _candidate_point(row)["quotient"],
                        "articulation_gain": {
                            "rho": float(_candidate_robust(row)["spearman"]
                                         - sparse["spearman"]),
                            "mae_tvd": float(sparse["mae_tvd"]
                                             - _candidate_robust(row)["mae_tvd"]),
                        },
                    }
                    for arm_id, row in sorted(rows.items())
                    if arm_id in eligible_arm_set
                },
            }

        target_self_values = [{
            "robust": _candidate_point(next(iter(rows.values())))["target_self_robust"],
            "forms": _candidate_point(next(iter(rows.values())))["target_self_forms"],
        } for rows in row_maps]
        if any(value != target_self_values[0] for value in target_self_values[1:]):
            raise ValueError(
                f"executor response-surface target-self statistics differ for {cell_id}")

        steps = []
        for left_index in range(len(entries) - 1):
            small_entry, large_entry = entries[left_index:left_index + 2]
            small = executor_rows[small_entry["label"]]
            large = executor_rows[large_entry["label"]]
            rank_gap = float(large["name_only"]["spearman"]
                             - small["name_only"]["spearman"])
            mae_gap = float(small["name_only"]["mae_tvd"]
                            - large["name_only"]["mae_tvd"])
            native_joint_advantage = rank_gap > 0 and mae_gap > 0
            arm_steps = []
            for arm_id in eligible_arm_ids:
                candidate = small["arms"][arm_id]["candidate"]
                candidate_quotient = small["arms"][arm_id]["candidate_quotient"]
                small_gain = small["arms"][arm_id]["articulation_gain"]
                large_gain = large["arms"][arm_id]["articulation_gain"]
                rank_with_margin = bool(
                    candidate["spearman"] >= large["name_only"]["spearman"] - rho_margin)
                mae_with_margin = bool(
                    candidate["mae_tvd"] <= large["name_only"]["mae_tvd"] + mae_margin)
                flip_with_margin = bool(
                    candidate["binary_flip_rate"]
                    <= large["name_only"]["binary_flip_rate"] + flip_margin)
                bias_with_margin = bool(
                    candidate["absolute_bias"]
                    <= large["name_only"]["absolute_bias"] + bias_margin)
                candidate_forms = small["arms"][arm_id]["candidate_forms"]
                large_forms = large["name_only_forms"]
                if set(candidate_forms) != set(large_forms):
                    raise ValueError(
                        f"executor response-surface form panels differ for {cell_id}/{arm_id}")
                matched_forms = {}
                for form in sorted(candidate_forms):
                    candidate_form = candidate_forms[form]
                    large_form = large_forms[form]
                    matched_forms[form] = {
                        "rank_at_least_large_name": bool(
                            candidate_form["spearman"] >= large_form["spearman"]),
                        "mae_no_worse_than_large_name": bool(
                            candidate_form["mae_tvd"] <= large_form["mae_tvd"]),
                        "rank_noninferior_with_margin": bool(
                            candidate_form["spearman"]
                            >= large_form["spearman"] - rho_margin),
                        "mae_noninferior_with_margin": bool(
                            candidate_form["mae_tvd"]
                            <= large_form["mae_tvd"] + mae_margin),
                    }
                matched_point = all(
                    row["rank_at_least_large_name"]
                    and row["mae_no_worse_than_large_name"]
                    for row in matched_forms.values())
                matched_margin = all(
                    row["rank_noninferior_with_margin"]
                    and row["mae_noninferior_with_margin"]
                    for row in matched_forms.values())
                endpoint_point = bool(
                    candidate.get("all_positive_polarity") is True
                    and candidate["spearman"] >= large["name_only"]["spearman"]
                    and candidate["mae_tvd"] <= large["name_only"]["mae_tvd"])
                endpoint_margin = bool(
                    candidate.get("all_positive_polarity") is True
                    and rank_with_margin and mae_with_margin)
                direct_gain = small_gain["rho"] > 0 and small_gain["mae_tvd"] > 0
                arm_steps.append({
                    "arm_id": arm_id,
                    "components": row_maps[0][arm_id].get("components", []),
                    "small_articulation": candidate,
                    "large_name_only": large["name_only"],
                    "small_articulation_gain": small_gain,
                    "descriptive_executor_by_articulation_difference_in_differences": {
                        "rho": float(large_gain["rho"] - small_gain["rho"]),
                        "mae_tvd": float(large_gain["mae_tvd"] - small_gain["mae_tvd"]),
                    },
                    "descriptive_step_closure": {
                        "rho": (float(small_gain["rho"] / rank_gap)
                                if rank_gap > 0 else None),
                        "mae_tvd": (float(small_gain["mae_tvd"] / mae_gap)
                                    if mae_gap > 0 else None),
                    },
                    "adverse_envelope_point_gates": {
                        "positive_polarity": candidate.get("all_positive_polarity") is True,
                        "rank_at_least_large_name": bool(
                            candidate["spearman"] >= large["name_only"]["spearman"]),
                        "mae_no_worse_than_large_name": bool(
                            candidate["mae_tvd"] <= large["name_only"]["mae_tvd"]),
                        "rank_noninferior_with_margin": rank_with_margin,
                        "mae_noninferior_with_margin": mae_with_margin,
                        "rank_mae_point_dominance": endpoint_point,
                        "rank_mae_margin_match": endpoint_margin,
                        "four_coordinate_margin_match": bool(
                            endpoint_margin and flip_with_margin and bias_with_margin),
                        "functional_target_reconstruction": bool(
                            candidate.get("all_positive_polarity") is True
                            and candidate["spearman"] >= functional_floor
                            and candidate_quotient.get("spearman") is not None
                            and candidate_quotient["spearman"] >= functional_floor
                            and small_gain["mae_tvd"] > 0),
                        "functional_floor_rescue": bool(
                            candidate.get("all_positive_polarity") is True
                            and small["name_only"]["spearman"] < functional_floor
                            and candidate["spearman"] >= functional_floor
                            and candidate_quotient.get("spearman") is not None
                            and candidate_quotient["spearman"] >= functional_floor
                            and small_gain["mae_tvd"] > 0),
                        "local_adverse_rank_mae_scale_step_candidate": bool(
                            native_joint_advantage and direct_gain and endpoint_margin),
                    },
                    "matched_form_rank_mae_point_gates": {
                        "forms": matched_forms,
                        "all_forms_point_dominance": bool(
                            candidate.get("all_positive_polarity") is True
                            and matched_point),
                        "all_forms_margin_match": bool(
                            candidate.get("all_positive_polarity") is True
                            and matched_margin),
                    },
                })
            steps.append({
                "small_executor": small_entry["label"],
                "large_executor": large_entry["label"],
                "native_name_scale_advantage": {
                    "rho": rank_gap,
                    "mae_tvd": mae_gap,
                },
                "native_step_point_eligibility": {
                    "rank_advantage_positive": rank_gap > 0,
                    "mae_advantage_positive": mae_gap > 0,
                    "joint_rank_mae_advantage_positive": native_joint_advantage,
                    "confidence_grade": "not_estimated",
                },
                "per_arm": arm_steps,
                "summary": {
                    "n_adverse_envelope_rank_mae_point_dominance": sum(
                        row["adverse_envelope_point_gates"]["rank_mae_point_dominance"]
                        for row in arm_steps),
                    "n_adverse_envelope_rank_mae_margin_matches": sum(
                        row["adverse_envelope_point_gates"]["rank_mae_margin_match"]
                        for row in arm_steps),
                    "n_matched_form_rank_mae_point_dominance": sum(
                        row["matched_form_rank_mae_point_gates"][
                            "all_forms_point_dominance"] for row in arm_steps),
                    "n_matched_form_rank_mae_margin_matches": sum(
                        row["matched_form_rank_mae_point_gates"]["all_forms_margin_match"]
                        for row in arm_steps),
                    "n_four_coordinate_adverse_envelope_margin_matches": sum(
                        row["adverse_envelope_point_gates"][
                            "four_coordinate_margin_match"] for row in arm_steps),
                    "n_functional_target_reconstructions": sum(
                        row["adverse_envelope_point_gates"][
                            "functional_target_reconstruction"] for row in arm_steps),
                    "n_functional_floor_rescues": sum(
                        row["adverse_envelope_point_gates"]["functional_floor_rescue"]
                        for row in arm_steps),
                    "n_local_adverse_rank_mae_scale_step_candidates": sum(
                        row["adverse_envelope_point_gates"][
                            "local_adverse_rank_mae_scale_step_candidate"]
                        for row in arm_steps),
                },
            })
        cell_rows.append({
            **{
                key: value
                for key, value in policy_cell_identity(
                    cells[0], context=f"executor response-surface/{cell_id}"
                ).items()
                if key != "identity_mode"
            },
            "control_arms_excluded_from_response_surface": control_arm_ids,
            "executors": executor_rows,
            "adjacent_steps": steps,
        })

    if len(surface_binary_protocols) != 1:
        raise ValueError("executor response surface mixes binary readout protocols")

    return {
        "schema": POLICY_RESPONSE_SCHEMA,
        "estimand": (
            "fixed-panel adverse-form executor-by-articulation response surface; local large-name "
            "envelope emulation is separate from target-policy reconstruction"),
        "validation": {
            "valid": True,
            "partition": entries[0]["report"]["partition"],
            "target_job": entries[0]["config"]["big_job"],
            "target_arm_id": entries[0]["config"]["target_arm_id"],
            "arm_bank_sha256": entries[0]["report"]["arm_bank_sha256"],
            "same_target_shards": True,
            "same_item_partition": True,
            "same_prompt_hash_bank": True,
            "same_readout_template": True,
            "same_binary_readout_protocol": True,
            "binary_readout": next(iter(surface_binary_protocols)),
            "executor_order_is_caller_declared": True,
        },
        "margins": {"rho": rho_margin, "mae_tvd": mae_margin,
                    "binary_flip_rate": flip_margin, "absolute_bias": bias_margin},
        "functional_rho_floor": functional_floor,
        "inputs": [{
            "label": entry["label"], "executor_job": entry["config"]["small_job"],
            "path": entry["path"], "sha256": entry["sha256"],
        } for entry in entries],
        "cells": cell_rows,
        "claim_boundary": (
            "This is an exact fixed-target, fixed-item, fixed-prompt response surface, but all "
            "endpoint matches, scale gaps, direct gains, closure ratios, and interactions are "
            "retrospective public-fold point estimates. Local large-name envelope matching is not "
            "70B target-policy isomorphism, and none is a paired confidence certificate. The "
            "primary envelope is minimax across forms; matched-form and four-coordinate "
            "sensitivities are reported separately. Input order declares executor order; model "
            "version, training, and quantization remain confounded with parameter count. Hashed "
            "shards authenticate outputs, but caller labels are not checkpoint registries."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default="",
                        help="comma-separated fixed_target_name_substitution JSON files")
    parser.add_argument("--policy-crossfold-artifacts", default="",
                        help="comma-separated fixed-target crossfold policy reports")
    parser.add_argument("--policy-fold-artifacts", default="",
                        help="comma-separated same-fold fixed-target policy reports")
    parser.add_argument("--labels", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    labels = [value for value in args.labels.split(",") if value] or None
    modes = (bool(args.artifacts), bool(args.policy_crossfold_artifacts),
             bool(args.policy_fold_artifacts))
    if sum(modes) != 1:
        parser.error("provide exactly one artifact mode")
    if args.policy_crossfold_artifacts:
        paths = [Path(value) for value in args.policy_crossfold_artifacts.split(",") if value]
        report = build_policy_executor_ladder(paths, labels=labels)
    elif args.policy_fold_artifacts:
        paths = [Path(value) for value in args.policy_fold_artifacts.split(",") if value]
        report = build_policy_executor_response_surface(paths, labels=labels)
    else:
        paths = [Path(value) for value in args.artifacts.split(",") if value]
        artifacts = [json.loads(path.read_text()) for path in paths]
        report = build_ladder_report(artifacts, labels=labels)
        report["inputs"] = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
