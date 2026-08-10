#!/usr/bin/env python
"""Verify and record the metadata-only hierarchy-panel v2 -> v3 migration.

The crosswalk has a deliberately narrow purpose.  It proves that v3 did not
change panel membership, construct text, hierarchy children, or any generated
prompt arm.  V3 adds the dependency, provenance, sampling-weight, and
overlap-estimand metadata needed to analyze the legacy hierarchy as the
overlapping action-node DAG it actually is.

This is a CPU-only local comparison.  It does not execute a candidate program,
read model outcomes, call an API, or use an accelerator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"

DEFAULT_PANEL_V2 = BASE / "panel_v2.json"
DEFAULT_PANEL_V3 = BASE / "panel_v3.json"
DEFAULT_BANK_V2 = BASE / "prompt_arm_bank_v2.json"
DEFAULT_BANK_V3 = BASE / "prompt_arm_bank_v3.json"
DEFAULT_OUTPUT = BASE / "hierarchy_panel_prompt_v2_to_v3_crosswalk_v1.json"

EXPECTED_PANEL_CELLS = 990
EXPECTED_PROMPT_ARMS = 28_335

PANEL_ADDED_CELL_METADATA = (
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
    "stratum_population_n",
    "stratum_selected_n",
    "inclusion_probability",
    "design_weight",
)

INVENTORY_ADDED_METADATA = (
    "dependency_components",
    "largest_dependency_component",
    "nodes_with_reused_immediate_sources",
    "maximum_source_assignment_multiplicity",
    "raw_provenance_components",
    "largest_raw_provenance_component",
    "nodes_with_raw_provenance_overlap",
    "maximum_raw_provenance_assignment_multiplicity",
)

BANK_PROPAGATED_IDENTITY_METADATA = (
    "source_kind",
    "source_index",
)

BANK_ADDED_ANALYSIS_METADATA = (
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
    "stratum_population_n",
    "stratum_selected_n",
    "inclusion_probability",
    "design_weight",
)

FRAME_ADDED_FIELDS = (
    "sampling_unit",
    "is_partition",
    "overlap_handling",
    "inherited_provenance_handling",
    "why_not_terminal_carry_forward_as_primary",
)

FRAME_REFRAMED_FIELDS = (
    "generation",
    "R1_source",
    "R2_source",
    "R3_source",
    "interpretation",
)


class CrosswalkError(ValueError):
    """Raised when v3 is not a metadata-only extension of v2."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrosswalkError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _json_content_digest(payload: dict[str, Any], digest_field: str) -> str:
    # The source compiler's frozen-content hash uses json.dumps' default
    # ensure_ascii=True.  Keep that recipe distinct from this artifact's UTF-8
    # canonicalization so the historical declarations can be checked exactly.
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != digest_field},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_digest(rows: Iterable[Any]) -> str:
    """Digest an ordered sequence as canonical JSON records separated by newlines."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _recorded_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _binding(path: Path, payload: dict[str, Any], content_field: str) -> dict[str, Any]:
    return {
        "path": _recorded_path(path),
        "file_sha256": _file_digest(path),
        "declared_content_sha256": payload[content_field],
    }


def _assert_declared_digest(
    payload: dict[str, Any], *, digest_field: str, label: str
) -> None:
    observed = _json_content_digest(payload, digest_field)
    _require(
        payload.get(digest_field) == observed,
        f"{label}: declared {digest_field} does not match its JSON content",
    )


def _assert_projection(
    old_rows: Sequence[dict[str, Any]],
    new_rows: Sequence[dict[str, Any]],
    *,
    added_fields: Sequence[str],
    identity_field: str,
    label: str,
) -> None:
    _require(len(old_rows) == len(new_rows), f"{label}: row count changed")
    expected_added = set(added_fields)
    for index, (old, new) in enumerate(zip(old_rows, new_rows)):
        identity = old.get(identity_field, index)
        _require(
            new.get(identity_field) == identity,
            f"{label}: row order/identity changed at index {index}",
        )
        _require(
            set(new) - set(old) == expected_added,
            f"{label}/{identity}: unexpected added or missing fields: "
            f"{sorted(set(new) - set(old))}",
        )
        _require(
            not (set(old) - set(new)),
            f"{label}/{identity}: v2 fields were removed",
        )
        changed = [key for key, value in old.items() if new.get(key) != value]
        _require(
            not changed,
            f"{label}/{identity}: v2 scientific/content fields changed: {changed}",
        )


def _panel_checks(panel_v2: dict[str, Any], panel_v3: dict[str, Any]) -> dict[str, Any]:
    _assert_declared_digest(
        panel_v2, digest_field="panel_content_sha256", label="panel_v2"
    )
    _assert_declared_digest(
        panel_v3, digest_field="panel_content_sha256", label="panel_v3"
    )

    expected_added_top = {"prevalence_estimands", "terminal_frontier_sensitivities"}
    _require(
        set(panel_v3) - set(panel_v2) == expected_added_top,
        "panel: unexpected v3 top-level fields",
    )
    _require(not (set(panel_v2) - set(panel_v3)), "panel: v2 top-level fields removed")

    expected_changed_top = {
        "hierarchy_frame",
        "inventory",
        "cells",
        "panel_content_sha256",
    }
    changed_top = {
        key for key in panel_v2 if panel_v2[key] != panel_v3.get(key)
    }
    _require(
        changed_top == expected_changed_top,
        f"panel: unexpected common top-level changes: {sorted(changed_top)}",
    )

    cells_v2 = panel_v2["cells"]
    cells_v3 = panel_v3["cells"]
    _require(
        len(cells_v2) == EXPECTED_PANEL_CELLS,
        f"panel: expected {EXPECTED_PANEL_CELLS} v2 cells, found {len(cells_v2)}",
    )
    _assert_projection(
        cells_v2,
        cells_v3,
        added_fields=PANEL_ADDED_CELL_METADATA,
        identity_field="id",
        label="panel_cells",
    )
    _assert_projection(
        panel_v2["inventory"],
        panel_v3["inventory"],
        added_fields=INVENTORY_ADDED_METADATA,
        identity_field="task",
        label="panel_inventory",
    )

    frame_v2 = panel_v2["hierarchy_frame"]
    frame_v3 = panel_v3["hierarchy_frame"]
    _require(
        set(frame_v3) - set(frame_v2) == set(FRAME_ADDED_FIELDS),
        "panel hierarchy frame: unexpected added fields",
    )
    _require(
        not (set(frame_v2) - set(frame_v3)),
        "panel hierarchy frame: v2 fields removed",
    )
    reframed = {
        key for key in frame_v2 if frame_v2[key] != frame_v3.get(key)
    }
    _require(
        reframed == set(FRAME_REFRAMED_FIELDS),
        f"panel hierarchy frame: unexpected reframed fields: {sorted(reframed)}",
    )
    _require(
        frame_v3["generation"] == "legacy-expanded-source-action-node-dag-v1",
        "panel hierarchy frame: v3 is not the declared action-node DAG",
    )
    _require(
        frame_v3["is_partition"] is False,
        "panel hierarchy frame: overlapping DAG must not be called a partition",
    )
    _require(
        "dependency" in frame_v3["overlap_handling"].lower(),
        "panel hierarchy frame: overlap handling does not name dependency components",
    )

    sensitivities = panel_v3["terminal_frontier_sensitivities"]
    _require(len(sensitivities) == 33, "panel: expected 33 task/level sensitivities")
    sensitivity_keys = {
        (row["task"], row["level"]) for row in sensitivities
    }
    _require(
        len(sensitivity_keys) == 33,
        "panel: terminal-frontier sensitivities repeat task/level cells",
    )
    prevalence = panel_v3["prevalence_estimands"]
    _require(
        set(prevalence) == {"balanced_panel", "source_inventory", "mandatory_sensitivities"},
        "panel: prevalence estimand metadata is incomplete",
    )

    counts = []
    for task in panel_v3["tasks"]:
        for level in panel_v3["levels"]:
            selected = [
                row for row in cells_v3
                if row["task"] == task and row["level"] == level
            ]
            _require(
                len(selected) == 30,
                f"panel: {task}/{level} does not retain 30 cells",
            )
            counts.append({"task": task, "level": level, "n_cells": len(selected)})

    for cell in cells_v3:
        population = cell["stratum_population_n"]
        selected = cell["stratum_selected_n"]
        _require(
            population >= selected >= 1,
            f"panel/{cell['id']}: invalid sampling-stratum counts",
        )
        _require(
            abs(cell["inclusion_probability"] - selected / population) < 1e-12,
            f"panel/{cell['id']}: inclusion probability is inconsistent",
        )
        _require(
            abs(cell["design_weight"] - population / selected) < 1e-12,
            f"panel/{cell['id']}: design weight is inconsistent",
        )

    scientific_fields = list(cells_v2[0])
    scientific_projection = _sequence_digest(
        {key: cell[key] for key in scientific_fields} for cell in cells_v3
    )
    return {
        "n_cells": len(cells_v3),
        "n_task_level_strata": len(counts),
        "cells_per_task_level": counts,
        "identity_rule": "v2 cell index and id map to the same v3 cell index and id",
        "cell_order_identical": True,
        "node_order_identical": [row["node_id"] for row in cells_v2]
        == [row["node_id"] for row in cells_v3],
        "all_v2_cell_fields_identical": True,
        "construct_scientific_content_unchanged": True,
        "unchanged_v2_cell_fields": scientific_fields,
        "scientific_cell_projection_canonical_jsonl_sha256": scientific_projection,
        "v3_added_cell_analysis_metadata": list(PANEL_ADDED_CELL_METADATA),
        "v3_added_inventory_analysis_metadata": list(INVENTORY_ADDED_METADATA),
        "v3_added_top_level_analysis_metadata": sorted(expected_added_top),
    }


def _bank_checks(
    bank_v2: dict[str, Any],
    bank_v3: dict[str, Any],
    panel_v2: dict[str, Any],
    panel_v3: dict[str, Any],
) -> dict[str, Any]:
    _assert_declared_digest(
        bank_v2, digest_field="bank_content_sha256", label="prompt_arm_bank_v2"
    )
    _assert_declared_digest(
        bank_v3, digest_field="bank_content_sha256", label="prompt_arm_bank_v3"
    )
    _require(
        set(bank_v2) == set(bank_v3),
        "prompt bank: v3 top-level field set changed",
    )
    changed_top = {
        key for key in bank_v2 if bank_v2[key] != bank_v3.get(key)
    }
    _require(
        changed_top == {"metric_panel_content_sha256", "cells", "bank_content_sha256"},
        f"prompt bank: unexpected top-level changes: {sorted(changed_top)}",
    )
    _require(
        bank_v2["metric_panel_content_sha256"] == panel_v2["panel_content_sha256"],
        "prompt bank v2 does not bind panel v2",
    )
    _require(
        bank_v3["metric_panel_content_sha256"] == panel_v3["panel_content_sha256"],
        "prompt bank v3 does not bind panel v3",
    )

    cells_v2 = bank_v2["cells"]
    cells_v3 = bank_v3["cells"]
    added = BANK_PROPAGATED_IDENTITY_METADATA + BANK_ADDED_ANALYSIS_METADATA
    _assert_projection(
        cells_v2,
        cells_v3,
        added_fields=added,
        identity_field="id",
        label="prompt_bank_cells",
    )
    _require(
        [row["id"] for row in cells_v3]
        == [row["id"] for row in panel_v3["cells"]],
        "prompt bank v3 cell order differs from panel v3",
    )

    panel_by_id = {row["id"]: row for row in panel_v3["cells"]}
    propagated = BANK_PROPAGATED_IDENTITY_METADATA + BANK_ADDED_ANALYSIS_METADATA
    for cell in cells_v3:
        panel_cell = panel_by_id[cell["id"]]
        changed = [key for key in propagated if cell[key] != panel_cell[key]]
        _require(
            not changed,
            f"prompt bank/{cell['id']}: propagated panel metadata differs: {changed}",
        )

    n_arms_v2 = sum(len(cell["arms"]) for cell in cells_v2)
    n_arms_v3 = sum(len(cell["arms"]) for cell in cells_v3)
    _require(
        n_arms_v2 == n_arms_v3 == EXPECTED_PROMPT_ARMS,
        f"prompt bank: expected {EXPECTED_PROMPT_ARMS} arms; found "
        f"v2={n_arms_v2}, v3={n_arms_v3}",
    )
    arm_ids_v2 = [
        (cell["id"], arm["id"])
        for cell in cells_v2
        for arm in cell["arms"]
    ]
    arm_ids_v3 = [
        (cell["id"], arm["id"])
        for cell in cells_v3
        for arm in cell["arms"]
    ]
    _require(arm_ids_v2 == arm_ids_v3, "prompt bank: semantic arm order changed")
    _require(
        all(old["arms"] == new["arms"] for old, new in zip(cells_v2, cells_v3)),
        "prompt bank: one or more semantic prompt-arm objects changed",
    )
    arm_projection = _sequence_digest(
        {"cell_id": cell["id"], "arm": arm}
        for cell in cells_v3
        for arm in cell["arms"]
    )
    return {
        "n_cells": len(cells_v3),
        "n_semantic_prompt_arms": n_arms_v3,
        "identity_rule": (
            "v2 (cell index, cell id, arm index, arm id) maps to the identical v3 tuple"
        ),
        "cell_order_identical": True,
        "semantic_arm_order_identical": True,
        "semantic_arm_objects_identical": True,
        "semantic_prompt_arm_projection_canonical_jsonl_sha256": arm_projection,
        "v3_propagated_identity_metadata": list(BANK_PROPAGATED_IDENTITY_METADATA),
        "v3_added_analysis_metadata": list(BANK_ADDED_ANALYSIS_METADATA),
    }


def build_crosswalk(
    panel_v2: dict[str, Any],
    panel_v3: dict[str, Any],
    bank_v2: dict[str, Any],
    bank_v3: dict[str, Any],
    *,
    input_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the verified canonical migration record or raise ``CrosswalkError``."""

    panel = _panel_checks(panel_v2, panel_v3)
    bank = _bank_checks(bank_v2, bank_v3, panel_v2, panel_v3)
    frame_v2 = panel_v2["hierarchy_frame"]
    frame_v3 = panel_v3["hierarchy_frame"]
    payload: dict[str, Any] = {
        "schema": "hierarchy_panel_prompt_v2_to_v3_crosswalk/v1",
        "status": "verified-metadata-only-scientific-content-preserved",
        "purpose": (
            "Separate the unchanged metric/prompt scientific objects from v3 analysis metadata "
            "and the corrected overlapping-DAG inference frame."
        ),
        "cpu_only": True,
        "outcomes_or_external_supervision_consumed": False,
        "inputs": input_bindings or {},
        "panel_result": panel,
        "prompt_bank_result": bank,
        "inference_frame_change": {
            "v2_generation_label": frame_v2["generation"],
            "v3_generation_label": frame_v3["generation"],
            "v3_sampling_unit": frame_v3["sampling_unit"],
            "v3_is_partition": frame_v3["is_partition"],
            "primary_frame": "overlapping native action-node DAG",
            "added_frame_fields": list(FRAME_ADDED_FIELDS),
            "reframed_text_fields": list(FRAME_REFRAMED_FIELDS),
            "analysis_consequence": (
                "Balanced-panel prevalence remains descriptive. Source-inventory prevalence uses "
                "the frozen design weights, and uncertainty must respect dependency/provenance "
                "components with the declared source-kind, merged-only, and terminal-frontier "
                "sensitivities."
            ),
        },
        "scientific_disposition": {
            "metric_selection_rerun_required": False,
            "prompt_generation_rerun_required": False,
            "model_scoring_rerun_required": False,
            "claim": (
                "V3 preserves all 990 selected metric objects and all 28,335 semantic prompt "
                "arms. It changes the analysis metadata and inference frame, not the construct "
                "or prompt content."
            ),
        },
    }
    payload["crosswalk_content_sha256"] = hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def compile_from_paths(
    *,
    panel_v2_path: Path = DEFAULT_PANEL_V2,
    panel_v3_path: Path = DEFAULT_PANEL_V3,
    bank_v2_path: Path = DEFAULT_BANK_V2,
    bank_v3_path: Path = DEFAULT_BANK_V3,
) -> dict[str, Any]:
    panel_v2 = _load_json(panel_v2_path)
    panel_v3 = _load_json(panel_v3_path)
    bank_v2 = _load_json(bank_v2_path)
    bank_v3 = _load_json(bank_v3_path)
    bindings = {
        "panel_v2": _binding(panel_v2_path, panel_v2, "panel_content_sha256"),
        "panel_v3": _binding(panel_v3_path, panel_v3, "panel_content_sha256"),
        "prompt_arm_bank_v2": _binding(bank_v2_path, bank_v2, "bank_content_sha256"),
        "prompt_arm_bank_v3": _binding(bank_v3_path, bank_v3, "bank_content_sha256"),
    }
    return build_crosswalk(
        panel_v2,
        panel_v3,
        bank_v2,
        bank_v3,
        input_bindings=bindings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-v2", type=Path, default=DEFAULT_PANEL_V2)
    parser.add_argument("--panel-v3", type=Path, default=DEFAULT_PANEL_V3)
    parser.add_argument("--bank-v2", type=Path, default=DEFAULT_BANK_V2)
    parser.add_argument("--bank-v3", type=Path, default=DEFAULT_BANK_V3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    artifact = compile_from_paths(
        panel_v2_path=args.panel_v2,
        panel_v3_path=args.panel_v3,
        bank_v2_path=args.bank_v2,
        bank_v3_path=args.bank_v3,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"verified {artifact['panel_result']['n_cells']} cells and "
        f"{artifact['prompt_bank_result']['n_semantic_prompt_arms']} semantic prompt arms; "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
