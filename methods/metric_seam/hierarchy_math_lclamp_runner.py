"""Execute audited math hybrids under fixed LLM-field sentinel profiles.

For a historical hybrid ``f(x, L)``, this runner measures conditional slices
``g_c(x) = f(x, c)`` by holding every declared ``LLM_FIELDS`` value fixed to a
constant sentinel profile while ``x`` varies.  Variation within one profile is
therefore executable/code-attributable, but the slice is not the original
hybrid, a pure-code rewrite, whole-construct fidelity, reconstruction, or
isomorphism.

Compiler-train runs use the complete, fixed Cartesian sentinel grid.  Heldout
runs require a frozen train-only profile gate and can run only its selected
profile for each program.  Items contain exactly ``item_key`` and ``ctext``;
references, outcomes, and extracted LLM values are rejected.  Trusted frozen
programs run in separate credential-stripped processes with accelerator devices
masked.  This is process isolation, not a filesystem or network sandbox.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
import math
import numbers
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import types
from typing import Any, Mapping, Sequence

from methods.existing_metrics_runner.coded.sandbox import ALLOWED_TOOLS
from methods.metric_seam.hierarchy_items import validate_task_items


ROOT = Path(__file__).resolve().parents[2]
HYBRIDS_ROOT = ROOT / "methods/metric_seam/hybrids"
PROGRAMS_ROOT = HYBRIDS_ROOT / "programs_math"
CANONICAL_ITEMS_ROOT = (
    ROOT / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/math-stackexchange"
)

MERGED_AUDIT_SCHEMA = "metric-seam.math-construct-fidelity-merged.v1"
MERGED_AUDIT_STATUS = "static_construct_fidelity_complete_pre_execution"
MERGED_AUDIT_DESIGN_SCOPE = "blind_static_construct_fidelity"
EXECUTION_SCHEMA = "metric-seam.math-lclamp-execution.v1"
PROFILE_GATE_SCHEMA = "metric-seam.math-lclamp-train-profile-gate.v1"
ALLOWED_PHASES = {"compiler_train", "heldout_pre_reference"}
ELIGIBLE_VERDICTS = {"exact", "partial"}

SENTINELS: tuple[dict[str, str], ...] = (
    {"sentinel_id": "empty", "value": ""},
    {"sentinel_id": "none", "value": "NONE"},
    {"sentinel_id": "yes", "value": "YES"},
    {"sentinel_id": "no", "value": "NO"},
    {"sentinel_id": "present", "value": "PRESENT"},
)

CAPABILITY_PATHS = (
    "methods/metric_seam/hybrids/ops.py",
    "methods/metric_seam/hybrids/ops_math.py",
)

_AUDIT_TOP_FIELDS = {
    "schema",
    "status",
    "task",
    "design_scope",
    "cross_audit",
    "sources",
    "panel_content_sha256",
    "hierarchy_frame",
    "ops_math_source",
    "ops_math_sha256",
    "execution_performed",
    "items_loaded",
    "reference_values_loaded",
    "outcome_labels_loaded",
    "program_outputs_loaded",
    "external_supervision",
    "depth_vocabulary",
    "capability_limit",
    "provenance",
    "interpretation",
    "summary",
    "rows",
}
_AUDIT_ROW_FIELDS = {
    "cell_id",
    "task",
    "level",
    "metric_name",
    "metric_description",
    "candidate",
    "requested_relation",
    "implemented_relations",
    "residual_construct",
    "verdict",
    "scope",
    "eligible_for_relation_local_execution",
    "audited_depth",
    "polarity_aggregation_applicability_caveats",
    "justification",
    "interpretation",
}
_CANDIDATE_FIELDS = {
    "aspect_id",
    "source_heading",
    "selected_revision",
    "source_path",
    "program_sha256",
    "historical_hybrid_provenance",
    "llm_fields_excluded_from_implemented_relations",
}
_PROFILE_FIELDS = {"profile_id", "profile_index", "assignments"}
_ASSIGNMENT_FIELDS = {"field_name", "sentinel_id", "value"}
_PROFILE_GATE_TOP_FIELDS = {
    "schema", "status", "selection_basis", "training_execution_source",
    "construct_fidelity_source", "construct_fidelity_fingerprint", "thresholds",
    "capability_runtime",
    "reference_values_used", "outcome_labels_used", "heldout_items_or_outputs_used",
    "prompt_or_llm_values_used", "score_direction_or_target_used", "interpretation",
    "summary", "selected_program_profiles", "programs",
}
_GATE_SELECTED_FIELDS = {
    "aspect_id", "source_path", "program_sha256", "selected_revision",
    "llm_field_names", "cell_ids", "profile",
}
_GATE_PROGRAM_FIELDS = {
    "aspect_id", "source_path", "program_sha256", "selected_revision",
    "llm_field_names", "cell_ids", "n_relation_mappings", "n_grid_profiles",
    "selected_for_heldout_pre_reference", "selected_profile", "selection_rule", "profiles",
}
_GATE_PROFILE_DECISION_FIELDS = {
    "profile", "n_measured", "coverage", "n_unique_scores", "n_failed",
    "n_abstained", "eligible_by_train_measurability", "decision",
}
_ASPECT_RE = re.compile(r"a\d+")
_CANONICAL_PROGRAM_IMPORTS = {"re", "math", "collections", "statistics"}
_BANNED_DYNAMIC_CALLS = {"open", "exec", "eval", "compile", "__import__"}
_BANNED_CALL_ROOTS = {
    "os", "subprocess", "socket", "urllib", "requests", "httpx", "openai", "anthropic"
}
_AUDIT_SOURCE_FIELDS = {
    "panel", "seed_map", "level_audit_1", "level_audit_2", "cross_audit_overlay"
}


class MathLClampInputError(ValueError):
    """Raised when an input violates the conditional-slice contract."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _content_fingerprint(value: Mapping) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_items(payload: object) -> list[dict[str, str]]:
    """Accept only a nonempty label-free ``item_key``/``ctext`` panel."""

    if not isinstance(payload, list) or not payload:
        raise MathLClampInputError("items must be a non-empty JSON list")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
            found = sorted(row) if isinstance(row, Mapping) else type(row).__name__
            raise MathLClampInputError(
                f"item {index} must contain exactly item_key and ctext; found {found}"
            )
        item_key, ctext = row["item_key"], row["ctext"]
        if not isinstance(item_key, str) or not item_key or item_key in seen:
            raise MathLClampInputError(f"item {index} has invalid or duplicate item_key")
        if not isinstance(ctext, str) or not ctext.strip():
            raise MathLClampInputError(f"item {index} has empty/non-string ctext")
        seen.add(item_key)
        normalized.append({"item_key": item_key, "ctext": ctext})
    return normalized


def load_bound_items(
    items_root: Path,
    phase: str,
    *,
    require_canonical: bool = True,
) -> tuple[list[dict[str, str]], Path]:
    """Validate both official splits before returning the phase-bound split."""

    if phase not in ALLOWED_PHASES:
        raise MathLClampInputError(f"phase must be one of {sorted(ALLOWED_PHASES)}")
    resolved = items_root.resolve()
    if require_canonical and resolved != CANONICAL_ITEMS_ROOT.resolve():
        raise MathLClampInputError(
            f"official math replay requires canonical items root {CANONICAL_ITEMS_ROOT}"
        )
    manifest_path = resolved / "manifest.json"
    train_path = resolved / "compiler_train.json"
    heldout_path = resolved / "sealed_heldout.json"
    manifest = _load_json(manifest_path)
    train = validate_items(_load_json(train_path))
    heldout = validate_items(_load_json(heldout_path))
    if manifest.get("task") != "math-stackexchange":
        raise MathLClampInputError("item manifest is not the math-stackexchange panel")
    try:
        validate_task_items(manifest, train, heldout)
    except ValueError as exc:
        raise MathLClampInputError(str(exc)) from exc
    return (
        (train, train_path)
        if phase == "compiler_train"
        else (heldout, heldout_path)
    )


def _validate_field_names(field_names: object) -> tuple[str, ...]:
    if not isinstance(field_names, list):
        raise MathLClampInputError("llm_field_names must be a list")
    if len(field_names) > 2:
        raise MathLClampInputError("historical program declares more than two LLM_FIELDS")
    if any(not isinstance(name, str) or not name for name in field_names):
        raise MathLClampInputError("LLM field names must be nonempty strings")
    if len(set(field_names)) != len(field_names):
        raise MathLClampInputError("LLM field names must be unique")
    return tuple(field_names)


def build_sentinel_profiles(field_names: Sequence[str]) -> list[dict[str, Any]]:
    """Return the fixed Cartesian grid in declared field order."""

    fields = _validate_field_names(list(field_names))
    products = itertools.product(SENTINELS, repeat=len(fields)) if fields else [()]
    profiles = []
    for index, combination in enumerate(products):
        assignments = [
            {
                "field_name": field_name,
                "sentinel_id": sentinel["sentinel_id"],
                "value": sentinel["value"],
            }
            for field_name, sentinel in zip(fields, combination)
        ]
        profiles.append(
            {
                "profile_id": f"profile_{index:03d}",
                "profile_index": index,
                "assignments": assignments,
            }
        )
    return profiles


def validate_profiles(
    profiles: object,
    field_names: Sequence[str],
    *,
    require_complete_grid: bool,
) -> list[dict[str, Any]]:
    """Require exact canonical grid records, or an ordered subset for heldout."""

    if not isinstance(profiles, list) or not profiles:
        raise MathLClampInputError("profiles must be a non-empty list")
    expected = build_sentinel_profiles(field_names)
    expected_by_id = {profile["profile_id"]: profile for profile in expected}
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping) or set(profile) != _PROFILE_FIELDS:
            raise MathLClampInputError(f"profile {index} has noncanonical fields")
        profile_id = profile.get("profile_id")
        if not isinstance(profile_id, str) or profile_id in seen:
            raise MathLClampInputError("profile ids must be unique strings")
        canonical = expected_by_id.get(profile_id)
        if canonical is None or dict(profile) != canonical:
            raise MathLClampInputError(f"profile {profile_id} is not a fixed sentinel profile")
        for assignment in profile["assignments"]:
            if not isinstance(assignment, Mapping) or set(assignment) != _ASSIGNMENT_FIELDS:
                raise MathLClampInputError(f"profile {profile_id} has invalid assignments")
        seen.add(profile_id)
        observed.append(canonical)
    if [p["profile_index"] for p in observed] != sorted(p["profile_index"] for p in observed):
        raise MathLClampInputError("profiles must be in canonical order")
    if require_complete_grid and observed != expected:
        raise MathLClampInputError("compiler-train requires the complete sentinel grid")
    return observed


def canonical_program_source(
    source_path: str,
    *,
    program_root: Path = PROGRAMS_ROOT,
    require_canonical_programs: bool = True,
) -> str:
    """Bind a source to one top-level ``aN_hR.py`` program under the allowed root."""

    root = program_root.resolve()
    if require_canonical_programs and root != PROGRAMS_ROOT.resolve():
        raise MathLClampInputError("production execution requires the canonical math program root")
    path = Path(source_path)
    absolute = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise MathLClampInputError(
            f"candidate is outside the allowed math program library: {path}"
        ) from exc
    if relative.parent != Path(".") or relative.suffix != ".py":
        raise MathLClampInputError(f"candidate must be one top-level Python module: {relative}")
    if not re.fullmatch(r"a\d+_h\d+\.py", relative.name):
        raise MathLClampInputError(f"candidate module has invalid identity: {relative.name}")
    if require_canonical_programs:
        return str(absolute.relative_to(ROOT.resolve()))
    return str(absolute)


def read_program_contract(path: Path) -> dict[str, Any]:
    """Read literal field identity and score signature without importing code."""

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise MathLClampInputError(f"cannot parse candidate source: {path}") from exc
    field_nodes = []
    score_nodes = []
    import_roots: set[str] = set()
    banned_calls: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "score":
            score_nodes.append(node)
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "LLM_FIELDS" for target in node.targets):
                field_nodes.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "LLM_FIELDS":
                field_nodes.append(node.value)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BANNED_DYNAMIC_CALLS:
                banned_calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                root = node.func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in _BANNED_CALL_ROOTS:
                    banned_calls.add(f"{root.id}.{node.func.attr}")
    if len(field_nodes) > 1:
        raise MathLClampInputError(f"{path}: multiple top-level LLM_FIELDS assignments")
    if field_nodes:
        try:
            fields = ast.literal_eval(field_nodes[0])
        except Exception as exc:
            raise MathLClampInputError(f"{path}: LLM_FIELDS must be a literal mapping") from exc
    else:
        fields = {}
    if not isinstance(fields, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, str)
        for key, value in fields.items()
    ):
        raise MathLClampInputError(f"{path}: invalid LLM_FIELDS mapping")
    field_names = _validate_field_names(list(fields))
    if len(score_nodes) != 1:
        raise MathLClampInputError(f"{path}: expected exactly one top-level score function")
    score_node = score_nodes[0]
    positional = [arg.arg for arg in score_node.args.posonlyargs + score_node.args.args]
    if positional != ["text", "extracted", "ops"] or score_node.args.vararg or score_node.args.kwarg:
        raise MathLClampInputError(
            f"{path}: score signature must be score(text, extracted, ops)"
        )
    return {
        "llm_field_names": list(field_names),
        "llm_field_prompts": fields,
        "import_roots": sorted(import_roots),
        "banned_runtime_calls": sorted(banned_calls),
    }


def _validate_canonical_program_runtime_policy(contract: Mapping, path: Path) -> None:
    imports = set(contract.get("import_roots", []))
    unexpected = imports - _CANONICAL_PROGRAM_IMPORTS
    if unexpected:
        raise MathLClampInputError(
            f"{path}: canonical L-clamp program imports non-allowlisted modules {sorted(unexpected)}"
        )
    if contract.get("banned_runtime_calls"):
        raise MathLClampInputError(
            f"{path}: canonical L-clamp program contains forbidden runtime calls"
        )


def _validate_capability_sources(records: object) -> list[dict[str, str]]:
    if not isinstance(records, list) or len(records) != len(CAPABILITY_PATHS):
        raise MathLClampInputError("audit must bind the complete math capability runtime")
    normalized = []
    for record, expected_path in zip(records, CAPABILITY_PATHS):
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise MathLClampInputError("invalid capability source identity")
        if record.get("path") != expected_path:
            raise MathLClampInputError("capability source path/order drift")
        observed = _sha256(ROOT / expected_path)
        if record.get("sha256") != observed:
            raise MathLClampInputError(f"capability source changed after audit: {expected_path}")
        normalized.append({"path": expected_path, "sha256": observed})
    return normalized


def _capability_sources_from_audit(audit: Mapping) -> list[dict[str, str]]:
    """Build the complete runtime identity from the merger's narrower binding.

    The static fidelity merger binds ``ops_math.py`` directly.  Its subclass
    imports ``ops.py``, so execution additionally records the current base-op
    bytes and the train gate freezes both records before heldout.  This does not
    retroactively claim that the static audit inspected ``ops.py``.
    """

    if audit.get("ops_math_source") != CAPABILITY_PATHS[1]:
        raise MathLClampInputError("merged audit is bound to an unexpected math capability")
    observed_math_sha = _sha256(ROOT / CAPABILITY_PATHS[1])
    if audit.get("ops_math_sha256") != observed_math_sha:
        raise MathLClampInputError("ops_math source changed after construct audit")
    return _validate_capability_sources(
        [
            {"path": path, "sha256": _sha256(ROOT / path)}
            for path in CAPABILITY_PATHS
        ]
    )


def _validate_audit_sources(records: object) -> None:
    """Require the five source records emitted by the completed merger."""

    if not isinstance(records, Mapping) or set(records) != _AUDIT_SOURCE_FIELDS:
        raise MathLClampInputError("merged audit source identities are incomplete")
    root = ROOT.resolve()
    for label, record in records.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise MathLClampInputError(f"merged audit has invalid {label} source identity")
        source = Path(record.get("path", ""))
        source = source.resolve() if source.is_absolute() else (ROOT / source).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise MathLClampInputError(f"merged audit {label} source escapes repository") from exc
        if not source.is_file() or record.get("sha256") != _sha256(source):
            raise MathLClampInputError(f"merged audit {label} source changed or is missing")


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _summarize_audit_rows(rows: Sequence[Mapping]) -> dict[str, Any]:
    verdicts = Counter(str(row["verdict"]) for row in rows)
    depths = Counter(
        "null" if row["audited_depth"] is None else str(row["audited_depth"])
        for row in rows
    )
    eligible_rows = [row for row in rows if row["eligible_for_relation_local_execution"]]
    eligible_depths = Counter(str(row["audited_depth"]) for row in eligible_rows)
    retrieved = sum(row["candidate"] is not None for row in rows)
    return {
        "n_cells": len(rows),
        "n_retrieved_candidates": retrieved,
        "verdicts": dict(sorted(verdicts.items())),
        "eligible_for_relation_local_execution": len(eligible_rows),
        "eligible_fraction_of_cells": _fraction(len(eligible_rows), len(rows)),
        "eligible_fraction_of_retrieved_candidates": _fraction(
            len(eligible_rows), retrieved
        ),
        "audited_depths": dict(sorted(depths.items())),
        "eligible_audited_depths": dict(sorted(eligible_depths.items())),
    }


def _expected_audit_summary(rows: Sequence[Mapping]) -> dict[str, Any]:
    summary = _summarize_audit_rows(rows)
    summary.update(
        {
            "whole_construct_exact_count": sum(
                row["verdict"] == "exact" for row in rows
            ),
            "n_unique_eligible_programs": len(
                {
                    row["candidate"]["aspect_id"]
                    for row in rows
                    if row["eligible_for_relation_local_execution"]
                }
            ),
            "by_level": {
                level: _summarize_audit_rows(
                    [row for row in rows if row["level"] == level]
                )
                for level in ("R1", "R2", "R3")
            },
        }
    )
    return summary


def _normalize_candidate(
    row: Mapping,
    *,
    program_root: Path,
    require_canonical_programs: bool,
) -> dict[str, Any]:
    candidate = row.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_FIELDS:
        raise MathLClampInputError(f"{row.get('cell_id')}: invalid candidate schema")
    aspect_id = candidate.get("aspect_id")
    revision = candidate.get("selected_revision")
    if not isinstance(aspect_id, str) or not _ASPECT_RE.fullmatch(aspect_id):
        raise MathLClampInputError(f"{row.get('cell_id')}: invalid aspect id")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise MathLClampInputError(f"{row.get('cell_id')}: invalid selected revision")
    canonical = canonical_program_source(
        candidate.get("source_path"),
        program_root=program_root,
        require_canonical_programs=require_canonical_programs,
    )
    expected_stem = f"{aspect_id}_h{revision}"
    if Path(canonical).stem != expected_stem:
        raise MathLClampInputError(f"{row.get('cell_id')}: aspect/revision/path mismatch")
    absolute = Path(canonical) if Path(canonical).is_absolute() else ROOT / canonical
    observed_sha = _sha256(absolute)
    if candidate.get("program_sha256") != observed_sha:
        raise MathLClampInputError(f"{row.get('cell_id')}: program changed after audit")
    field_names = _validate_field_names(
        candidate.get("llm_fields_excluded_from_implemented_relations")
    )
    static_contract = read_program_contract(absolute)
    if require_canonical_programs:
        _validate_canonical_program_runtime_policy(static_contract, absolute)
    static_field_names = static_contract["llm_field_names"]
    if set(field_names) != set(static_field_names):
        raise MathLClampInputError(f"{row.get('cell_id')}: LLM field identity drift")
    for field in ("source_heading", "historical_hybrid_provenance"):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            raise MathLClampInputError(f"{row.get('cell_id')}: invalid {field}")
    return {
        "aspect_id": aspect_id,
        "source_heading": candidate["source_heading"],
        "selected_revision": revision,
        "source_path": canonical,
        "program_sha256": observed_sha,
        "historical_hybrid_provenance": candidate["historical_hybrid_provenance"],
        # The merger records excluded field identity, not a profile-order
        # contract.  The literal program declaration supplies the only valid
        # Cartesian profile order.
        "llm_field_names": list(static_field_names),
    }


def validate_merged_audit(
    audit: Mapping,
    *,
    program_root: Path = PROGRAMS_ROOT,
    require_canonical_programs: bool = True,
) -> None:
    """Strictly validate the canonical completed 90-cell math artifact."""

    if audit.get("schema") != MERGED_AUDIT_SCHEMA:
        raise MathLClampInputError(f"expected merged audit schema {MERGED_AUDIT_SCHEMA}")
    if set(audit) != _AUDIT_TOP_FIELDS:
        raise MathLClampInputError(
            f"merged audit top-level fields differ: {sorted(set(audit) ^ _AUDIT_TOP_FIELDS)}"
        )
    if audit.get("task") != "math-stackexchange":
        raise MathLClampInputError("merged audit task mismatch")
    if audit.get("status") != MERGED_AUDIT_STATUS:
        raise MathLClampInputError("merged audit is not a completed cross-audited static artifact")
    if audit.get("design_scope") != MERGED_AUDIT_DESIGN_SCOPE:
        raise MathLClampInputError("merged audit design scope mismatch")
    cross_audit = audit.get("cross_audit")
    if not isinstance(cross_audit, Mapping) or set(cross_audit) != {
        "status", "n_guarded_changes", "provisional_until_complete"
    }:
        raise MathLClampInputError("merged audit has invalid cross-audit metadata")
    changes = cross_audit.get("n_guarded_changes")
    if (
        cross_audit.get("status") != "complete"
        or cross_audit.get("provisional_until_complete") is not False
        or isinstance(changes, bool)
        or not isinstance(changes, int)
        or changes < 0
    ):
        raise MathLClampInputError("merged audit cross-audit is not complete")
    _validate_audit_sources(audit.get("sources"))
    panel_sha = audit.get("panel_content_sha256")
    if not isinstance(panel_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", panel_sha):
        raise MathLClampInputError("merged audit has invalid panel digest")
    if not isinstance(audit.get("depth_vocabulary"), Mapping) or not audit["depth_vocabulary"]:
        raise MathLClampInputError("merged audit has invalid depth vocabulary")
    if not isinstance(audit.get("hierarchy_frame"), Mapping) or not audit["hierarchy_frame"]:
        raise MathLClampInputError("merged audit has invalid hierarchy frame")
    if not isinstance(audit.get("capability_limit"), str) or not audit[
        "capability_limit"
    ].strip():
        raise MathLClampInputError("merged audit has invalid capability limit")
    if not isinstance(audit.get("provenance"), Mapping):
        raise MathLClampInputError("merged audit has invalid provenance")
    if not isinstance(audit.get("summary"), Mapping):
        raise MathLClampInputError("merged audit has invalid summary")
    if not isinstance(audit.get("interpretation"), str) or not audit["interpretation"].strip():
        raise MathLClampInputError("merged audit has invalid interpretation")
    false_flags = (
        "execution_performed",
        "items_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "program_outputs_loaded",
        "external_supervision",
    )
    if any(audit.get(field) is not False for field in false_flags):
        raise MathLClampInputError("merged static audit contains forbidden post-audit state")
    _capability_sources_from_audit(audit)
    rows = audit.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise MathLClampInputError("merged audit must contain exactly 90 rows")
    if Counter(row.get("level") for row in rows if isinstance(row, Mapping)) != {
        "R1": 30,
        "R2": 30,
        "R3": 30,
    }:
        raise MathLClampInputError("merged audit must contain 30 rows per level")
    seen: set[str] = set()
    aspect_identities: dict[str, tuple[Any, ...]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _AUDIT_ROW_FIELDS:
            raise MathLClampInputError(f"audit row {index} does not match canonical schema")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            raise MathLClampInputError(f"audit row {index} has invalid/duplicate cell_id")
        seen.add(cell_id)
        if row.get("task") != "math-stackexchange" or row.get("level") not in {"R1", "R2", "R3"}:
            raise MathLClampInputError(f"{cell_id}: task/level mismatch")
        verdict = row.get("verdict")
        expected_scope = {
            "exact": "whole_construct",
            "partial": "subrelation_only",
            "mismatch": "none",
            "no_candidate_bounded_non_discovery": "none",
        }.get(verdict)
        if expected_scope is None or row.get("scope") != expected_scope:
            raise MathLClampInputError(f"{cell_id}: verdict/scope mismatch")
        eligible = row.get("eligible_for_relation_local_execution")
        if not isinstance(eligible, bool) or eligible != (verdict in ELIGIBLE_VERDICTS):
            raise MathLClampInputError(f"{cell_id}: verdict/eligibility mismatch")
        for field in (
            "metric_name",
            "metric_description",
            "requested_relation",
            "residual_construct",
            "justification",
            "interpretation",
        ):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise MathLClampInputError(f"{cell_id}: invalid {field}")
        relations = row.get("implemented_relations")
        if not isinstance(relations, list) or not all(
            isinstance(value, str) and value.strip() for value in relations
        ):
            raise MathLClampInputError(f"{cell_id}: invalid implemented relations")
        if eligible and not relations:
            raise MathLClampInputError(f"{cell_id}: eligible row has no code relation")
        caveats = row.get("polarity_aggregation_applicability_caveats")
        if not isinstance(caveats, list) or not caveats or not all(
            isinstance(value, str) and value.strip() for value in caveats
        ):
            raise MathLClampInputError(f"{cell_id}: invalid caveats")
        candidate = row.get("candidate")
        depth = row.get("audited_depth")
        if candidate is None:
            if verdict != "no_candidate_bounded_non_discovery" or depth is not None or relations:
                raise MathLClampInputError(f"{cell_id}: no-candidate row is inconsistent")
            continue
        if not relations:
            raise MathLClampInputError(f"{cell_id}: candidate row has no implemented relation")
        if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 4:
            raise MathLClampInputError(f"{cell_id}: invalid audited depth")
        normalized = _normalize_candidate(
            row,
            program_root=program_root,
            require_canonical_programs=require_canonical_programs,
        )
        identity = (
            normalized["source_path"],
            normalized["program_sha256"],
            normalized["selected_revision"],
            tuple(normalized["llm_field_names"]),
        )
        previous = aspect_identities.setdefault(normalized["aspect_id"], identity)
        if previous != identity:
            raise MathLClampInputError(
                f"{normalized['aspect_id']}: multiple executable source identities"
            )
    if audit.get("summary") != _expected_audit_summary(rows):
        raise MathLClampInputError("merged audit summary does not match its rows")


def build_execution_plan(
    audit: Mapping,
    *,
    program_root: Path = PROGRAMS_ROOT,
    require_canonical_programs: bool = True,
) -> list[dict[str, Any]]:
    """Group shared accepted rows by exact program/revision/field identity."""

    validate_merged_audit(
        audit,
        program_root=program_root,
        require_canonical_programs=require_canonical_programs,
    )
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in audit["rows"]:
        if not row["eligible_for_relation_local_execution"]:
            continue
        candidate = _normalize_candidate(
            row,
            program_root=program_root,
            require_canonical_programs=require_canonical_programs,
        )
        identity = (
            candidate["aspect_id"],
            candidate["source_path"],
            candidate["program_sha256"],
            candidate["selected_revision"],
            tuple(candidate["llm_field_names"]),
        )
        if identity not in grouped:
            grouped[identity] = {**candidate, "relations": []}
        grouped[identity]["relations"].append(
            {
                "cell_id": row["cell_id"],
                "level": row["level"],
                "metric_name": row["metric_name"],
                "construct_fidelity_verdict": row["verdict"],
                "scope": row["scope"],
                "requested_relation": row["requested_relation"],
                "implemented_relations": list(row["implemented_relations"]),
                "residual_construct": row["residual_construct"],
                "audited_depth": row["audited_depth"],
            }
        )
    plans = list(grouped.values())
    for plan in plans:
        plan["relations"] = sorted(plan["relations"], key=lambda relation: relation["cell_id"])
    return sorted(plans, key=lambda plan: (plan["aspect_id"], plan["source_path"]))


def _score_state(score: object) -> tuple[str, str, float | None, str | None]:
    """Return the required three-state accounting plus a detailed status."""

    if score is None:
        return "abstained", "abstained", None, None
    if isinstance(score, bool) or not isinstance(score, numbers.Real):
        return "failed", "contract_error", None, "NonNumericScore"
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return "failed", "contract_error", None, "OutOfRangeScore"
    return "measured", "scored", value, None


def _load_math_ops(capability_sources: Sequence[Mapping]) -> Any:
    """Execute the exact audited capability bytes, without loading a corpus."""

    normalized = _validate_capability_sources(capability_sources)
    source_by_path = {record["path"]: (ROOT / record["path"]).read_bytes() for record in normalized}
    base_module = types.ModuleType("ops")
    base_module.__file__ = str(ROOT / CAPABILITY_PATHS[0])
    exec(compile(source_by_path[CAPABILITY_PATHS[0]], base_module.__file__, "exec"), base_module.__dict__)
    prior_ops = sys.modules.get("ops")
    sys.modules["ops"] = base_module
    try:
        math_module = types.ModuleType("ops_math")
        math_module.__file__ = str(ROOT / CAPABILITY_PATHS[1])
        exec(
            compile(source_by_path[CAPABILITY_PATHS[1]], math_module.__file__, "exec"),
            math_module.__dict__,
        )
    finally:
        if prior_ops is None:
            sys.modules.pop("ops", None)
        else:
            sys.modules["ops"] = prior_ops
    math_ops = getattr(math_module, "MathOps", None)
    if not isinstance(math_ops, type):
        raise MathLClampInputError("audited ops_math source has no MathOps class")
    return math_ops(corpus_path=None)


def _load_exact_program(path: Path, expected_sha256: str) -> types.ModuleType:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise MathLClampInputError("program source changed before worker execution")
    module = types.ModuleType(f"metric_seam_lclamp_{path.stem}")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _profile_summary(rows: Sequence[Mapping]) -> tuple[dict[str, Any], str]:
    states = Counter(row["measurement_state"] for row in rows)
    statuses = Counter(row["status"] for row in rows)
    scores = [row["score"] for row in rows if row["measurement_state"] == "measured"]
    summary = {
        "n_items": len(rows),
        "state_counts": {state: states[state] for state in ("measured", "abstained", "failed")},
        "status_counts": dict(sorted(statuses.items())),
        "n_measured": len(scores),
        "coverage": round(len(scores) / len(rows), 6) if rows else 0.0,
        "n_unique_scores": len(set(scores)),
        "nonconstant_when_measured": len(set(scores)) >= 2,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }
    if states["failed"]:
        status = "item_failures_with_measurements" if scores else "item_failures_no_measurements"
    elif not scores:
        status = "no_usable_measurements"
    elif len(set(scores)) < 2:
        status = "constant_measurement"
    else:
        status = "nondegenerate_measurement"
    return summary, status


def execute_one_program(
    plan: Mapping,
    items: Sequence[Mapping],
    profiles: Sequence[Mapping],
    capability_sources: Sequence[Mapping],
    *,
    program_root: Path = PROGRAMS_ROOT,
    require_canonical_programs: bool = True,
    require_complete_grid: bool,
) -> dict[str, Any]:
    """Execute one trusted exact-source program under each supplied profile."""

    normalized_items = validate_items(list(items))
    canonical = canonical_program_source(
        plan["source_path"],
        program_root=program_root,
        require_canonical_programs=require_canonical_programs,
    )
    absolute = Path(canonical) if Path(canonical).is_absolute() else ROOT / canonical
    if Path(canonical).stem != f"{plan['aspect_id']}_h{plan['selected_revision']}":
        raise MathLClampInputError("worker aspect/revision/source identity mismatch")
    if _sha256(absolute) != plan["program_sha256"]:
        raise MathLClampInputError("worker program digest mismatch")
    static_contract = read_program_contract(absolute)
    if require_canonical_programs:
        _validate_canonical_program_runtime_policy(static_contract, absolute)
    field_names = _validate_field_names(plan["llm_field_names"])
    if static_contract["llm_field_names"] != list(field_names):
        raise MathLClampInputError("worker LLM field identity mismatch")
    normalized_profiles = validate_profiles(
        list(profiles), field_names, require_complete_grid=require_complete_grid
    )
    try:
        module = _load_exact_program(absolute, plan["program_sha256"])
        runtime_fields = getattr(module, "LLM_FIELDS", {}) or {}
        if not isinstance(runtime_fields, Mapping) or list(runtime_fields) != list(field_names):
            raise MathLClampInputError("runtime LLM field identity differs from static contract")
        scorer = getattr(module, "score", None)
        if not callable(scorer):
            raise MathLClampInputError("runtime candidate has no callable score")
        ops = _load_math_ops(capability_sources)
    except MathLClampInputError:
        raise
    except Exception as exc:
        return {
            "aspect_id": plan["aspect_id"],
            "selected_revision": plan["selected_revision"],
            "source_path": canonical,
            "worker_status": "program_import_error",
            "measurement_status": "not_measured",
            "error_type": type(exc).__name__,
            "profiles": [],
            "summary": {"n_items": len(normalized_items), "n_profiles": len(normalized_profiles)},
        }

    profile_results = []
    for profile in normalized_profiles:
        constants = {
            assignment["field_name"]: assignment["value"]
            for assignment in profile["assignments"]
        }
        outputs = []
        for item in normalized_items:
            try:
                # A fresh plain dict preserves historical isinstance(..., dict)
                # behavior while preventing mutations from leaking across items.
                state, status, value, error_type = _score_state(
                    scorer(item["ctext"], dict(constants), ops)
                )
            except Exception as exc:
                state, status, value, error_type = (
                    "failed",
                    "execution_error",
                    None,
                    type(exc).__name__,
                )
            output = {
                "item_key": item["item_key"],
                "measurement_state": state,
                "status": status,
                "score": value,
            }
            if error_type:
                output["error_type"] = error_type
            outputs.append(output)
        summary, measurement_status = _profile_summary(outputs)
        profile_results.append(
            {
                **profile,
                "constant_values_frozen_within_profile": True,
                "measurement_status": measurement_status,
                "rows": outputs,
                "summary": summary,
            }
        )

    measurement_statuses = Counter(
        profile["measurement_status"] for profile in profile_results
    )
    state_totals = Counter()
    for profile in profile_results:
        state_totals.update(profile["summary"]["state_counts"])
    return {
        "aspect_id": plan["aspect_id"],
        "selected_revision": plan["selected_revision"],
        "source_path": canonical,
        "worker_status": "completed",
        "measurement_status": (
            "at_least_one_nondegenerate_profile"
            if measurement_statuses["nondegenerate_measurement"]
            else "no_nondegenerate_profile"
        ),
        "llm_field_names": list(field_names),
        "profiles": profile_results,
        "summary": {
            "n_items": len(normalized_items),
            "n_profiles": len(profile_results),
            "profile_measurement_status_counts": dict(sorted(measurement_statuses.items())),
            "three_state_totals": {
                state: state_totals[state] for state in ("measured", "abstained", "failed")
            },
        },
    }


def _controlled_tool_path() -> str:
    directories = {
        str(Path(sys.executable).resolve().parent),
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    }
    for tool in sorted(ALLOWED_TOOLS):
        found = shutil.which(tool)
        if found:
            directories.add(str(Path(found).resolve().parent))
    return os.pathsep.join(sorted(directories))


def worker_environment(home: Path) -> dict[str, str]:
    """Construct an allowlisted CPU environment and inherit no credentials."""

    env = {
        "HOME": str(home),
        "PATH": _controlled_tool_path(),
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MPLCONFIGDIR": str(home / ".matplotlib"),
        "TMPDIR": str(home / "tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "HIP_VISIBLE_DEVICES": "-1",
        "ROCR_VISIBLE_DEVICES": "-1",
    }
    (home / "tmp").mkdir(parents=True, exist_ok=True)
    return env


def _worker_failure(
    plan: Mapping,
    status: str,
    n_items: int,
    n_profiles: int,
    error_type: str | None = None,
) -> dict[str, Any]:
    result = {
        "aspect_id": plan["aspect_id"],
        "selected_revision": plan["selected_revision"],
        "source_path": plan["source_path"],
        "worker_status": status,
        "measurement_status": "not_measured",
        "profiles": [],
        "summary": {"n_items": n_items, "n_profiles": n_profiles},
    }
    if error_type:
        result["error_type"] = error_type
    return result


def _validate_worker_result(
    result: Mapping,
    plan: Mapping,
    expected_profiles: Sequence[Mapping],
    expected_item_keys: Sequence[str],
) -> None:
    for field in ("aspect_id", "selected_revision", "source_path"):
        if result.get(field) != plan[field]:
            raise MathLClampInputError(f"worker {field} mismatch for {plan['aspect_id']}")
    if result.get("worker_status") != "completed":
        if result.get("profiles") != []:
            raise MathLClampInputError("failed worker may not emit partial profile results")
        return
    profiles = result.get("profiles")
    if not isinstance(profiles, list) or [p.get("profile_id") for p in profiles] != [
        p["profile_id"] for p in expected_profiles
    ]:
        raise MathLClampInputError(f"worker profile-set mismatch for {plan['aspect_id']}")
    for observed, expected in zip(profiles, expected_profiles):
        if any(observed.get(field) != expected[field] for field in _PROFILE_FIELDS):
            raise MathLClampInputError(f"worker profile identity drift for {plan['aspect_id']}")
        rows = observed.get("rows")
        if not isinstance(rows, list) or [row.get("item_key") for row in rows] != list(
            expected_item_keys
        ):
            raise MathLClampInputError(f"worker item-set mismatch for {plan['aspect_id']}")
        if any(row.get("measurement_state") not in {"measured", "abstained", "failed"} for row in rows):
            raise MathLClampInputError(f"worker emitted invalid measurement state for {plan['aspect_id']}")


def _run_worker(
    plan: Mapping,
    items: Sequence[Mapping],
    profiles: Sequence[Mapping],
    capability_sources: Sequence[Mapping],
    *,
    timeout_seconds: float,
    require_complete_grid: bool,
    program_root: Path = PROGRAMS_ROOT,
    require_canonical_programs: bool = True,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="metric-seam-math-lclamp-") as directory:
        worker_home = Path(directory)
        out = worker_home / "result.json"
        command = [
            sys.executable,
            "-m",
            "methods.metric_seam.hierarchy_math_lclamp_runner",
            "worker",
            "--source-path",
            str(plan["source_path"]),
            "--aspect-id",
            str(plan["aspect_id"]),
            "--selected-revision",
            str(plan["selected_revision"]),
            "--program-sha256",
            str(plan["program_sha256"]),
            "--llm-field-names-json",
            json.dumps(plan["llm_field_names"], separators=(",", ":")),
            "--profile-mode",
            "complete" if require_complete_grid else "subset",
            "--out",
            str(out),
        ]
        if not require_canonical_programs:
            command.extend(
                ["--test-program-root", str(program_root), "--allow-test-program-root"]
            )
        payload = json.dumps(
            {
                "items": list(items),
                "profiles": list(profiles),
                "capability_sources": list(capability_sources),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=worker_environment(worker_home),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(input=payload, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            return _worker_failure(
                plan, "program_timeout", len(items), len(profiles)
            )
        if process.returncode != 0 or not out.exists():
            return _worker_failure(
                plan,
                "worker_error",
                len(items),
                len(profiles),
                f"WorkerExit{process.returncode}",
            )
        result = _load_json(out)
        _validate_worker_result(
            result,
            plan,
            profiles,
            [row["item_key"] for row in items],
        )
        return result


def apply_profile_selection(
    plans: Sequence[Mapping],
    selection: Mapping,
    *,
    audit_fingerprint: str,
    capability_runtime: Sequence[Mapping],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Bind heldout execution to one train-selected canonical profile/program."""

    if set(selection) != _PROFILE_GATE_TOP_FIELDS:
        raise MathLClampInputError(
            f"profile gate fields differ: {sorted(set(selection) ^ _PROFILE_GATE_TOP_FIELDS)}"
        )
    if selection.get("schema") != PROFILE_GATE_SCHEMA:
        raise MathLClampInputError("heldout requires the canonical math profile gate")
    if selection.get("status") != "frozen_before_heldout_profile_execution":
        raise MathLClampInputError("profile gate was not frozen before heldout")
    if selection.get("selection_basis") != "compiler_train_profile_measurability_only":
        raise MathLClampInputError("profile gate selection basis is not train-only measurability")
    if selection.get("construct_fidelity_fingerprint") != audit_fingerprint:
        raise MathLClampInputError("profile gate is bound to a different construct audit")
    if selection.get("capability_runtime") != list(capability_runtime):
        raise MathLClampInputError("profile gate is bound to a different capability runtime")
    for field in (
        "reference_values_used",
        "outcome_labels_used",
        "heldout_items_or_outputs_used",
        "prompt_or_llm_values_used",
        "score_direction_or_target_used",
    ):
        if selection.get(field) is not False:
            raise MathLClampInputError(f"profile gate used forbidden field: {field}")
    thresholds = selection.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "min_measured", "min_coverage", "min_unique_scores", "max_failed",
        "profile_tie_break",
    } or not isinstance(
        selection.get("summary"), Mapping
    ) or not isinstance(selection.get("interpretation"), str):
        raise MathLClampInputError("profile gate has invalid metadata")
    for key in ("min_measured", "min_unique_scores", "max_failed"):
        if isinstance(thresholds.get(key), bool) or not isinstance(thresholds.get(key), int):
            raise MathLClampInputError("profile gate has invalid integer threshold")
    if (
        thresholds["min_measured"] < 2
        or thresholds["min_unique_scores"] < 2
        or thresholds["max_failed"] < 0
        or isinstance(thresholds.get("min_coverage"), bool)
        or not isinstance(thresholds.get("min_coverage"), (int, float))
        or not 0 <= thresholds["min_coverage"] <= 1
        or thresholds.get("profile_tie_break") != "lowest fixed profile_index"
    ):
        raise MathLClampInputError("profile gate threshold policy drift")
    selected = selection.get("selected_program_profiles")
    program_records = selection.get("programs")
    if not isinstance(selected, list) or not isinstance(program_records, list):
        raise MathLClampInputError("profile gate selected-program list is invalid")
    plan_by_identity = {
        (
            plan["aspect_id"],
            plan["source_path"],
            plan["program_sha256"],
            plan["selected_revision"],
            tuple(plan["llm_field_names"]),
        ): dict(plan)
        for plan in plans
    }
    plan_by_aspect = {plan["aspect_id"]: plan for plan in plans}
    if len(program_records) != len(plans):
        raise MathLClampInputError("profile gate program count differs from audit plan")
    derived_selected = []
    seen_programs = set()
    for record in program_records:
        if not isinstance(record, Mapping) or set(record) != _GATE_PROGRAM_FIELDS:
            raise MathLClampInputError("invalid profile gate program record")
        aspect_id = record.get("aspect_id")
        plan = plan_by_aspect.get(aspect_id)
        if plan is None or aspect_id in seen_programs:
            raise MathLClampInputError("profile gate program identity drift or duplicate")
        seen_programs.add(aspect_id)
        for field in ("source_path", "program_sha256", "selected_revision", "llm_field_names"):
            if record.get(field) != plan[field]:
                raise MathLClampInputError(f"profile gate {field} drift for {aspect_id}")
        cell_ids = [relation["cell_id"] for relation in plan["relations"]]
        if record.get("cell_ids") != cell_ids or record.get("n_relation_mappings") != len(cell_ids):
            raise MathLClampInputError(f"profile gate relation mapping drift for {aspect_id}")
        expected_grid = build_sentinel_profiles(plan["llm_field_names"])
        if record.get("n_grid_profiles") != len(expected_grid):
            raise MathLClampInputError(f"profile gate grid size drift for {aspect_id}")
        decisions = record.get("profiles")
        if not isinstance(decisions, list) or len(decisions) != len(expected_grid):
            raise MathLClampInputError(f"profile gate decision grid drift for {aspect_id}")
        eligible_profiles = []
        for decision, expected_profile in zip(decisions, expected_grid):
            if not isinstance(decision, Mapping) or set(decision) != _GATE_PROFILE_DECISION_FIELDS:
                raise MathLClampInputError("invalid gate profile decision record")
            if decision.get("profile") != expected_profile:
                raise MathLClampInputError(f"profile gate sentinel identity drift for {aspect_id}")
            eligible = decision.get("eligible_by_train_measurability")
            if not isinstance(eligible, bool):
                raise MathLClampInputError("profile gate eligibility must be boolean")
            for key in ("n_measured", "n_unique_scores", "n_failed", "n_abstained"):
                if isinstance(decision.get(key), bool) or not isinstance(decision.get(key), int):
                    raise MathLClampInputError("profile gate count is invalid")
            if isinstance(decision.get("coverage"), bool) or not isinstance(
                decision.get("coverage"), (int, float)
            ) or not 0 <= decision["coverage"] <= 1:
                raise MathLClampInputError("profile gate coverage is invalid")
            if not isinstance(decision.get("decision"), str):
                raise MathLClampInputError("profile gate decision label is invalid")
            expected_eligible = bool(
                decision["n_failed"] <= thresholds["max_failed"]
                and decision["n_measured"] >= thresholds["min_measured"]
                and decision["coverage"] >= thresholds["min_coverage"]
                and decision["n_unique_scores"] >= thresholds["min_unique_scores"]
            )
            if eligible != expected_eligible:
                raise MathLClampInputError(
                    f"profile gate eligibility is not implied by train measurability for {aspect_id}"
                )
            if eligible:
                eligible_profiles.append(expected_profile)
        expected_selected_profile = eligible_profiles[0] if eligible_profiles else None
        selected_flag = record.get("selected_for_heldout_pre_reference")
        if selected_flag != (expected_selected_profile is not None) or record.get(
            "selected_profile"
        ) != expected_selected_profile:
            raise MathLClampInputError(f"profile gate selection/tie-break drift for {aspect_id}")
        if expected_selected_profile is not None:
            derived_selected.append(
                {
                    "aspect_id": plan["aspect_id"],
                    "source_path": plan["source_path"],
                    "program_sha256": plan["program_sha256"],
                    "selected_revision": plan["selected_revision"],
                    "llm_field_names": plan["llm_field_names"],
                    "cell_ids": cell_ids,
                    "profile": expected_selected_profile,
                }
            )
    if seen_programs != set(plan_by_aspect) or selected != derived_selected:
        raise MathLClampInputError("profile gate selected list does not match train decisions")

    applied = []
    seen: set[tuple[Any, ...]] = set()
    for row in selected:
        if not isinstance(row, Mapping) or set(row) != _GATE_SELECTED_FIELDS:
            raise MathLClampInputError("invalid selected program/profile row")
        identity = (
            row.get("aspect_id"),
            row.get("source_path"),
            row.get("program_sha256"),
            row.get("selected_revision"),
            tuple(row.get("llm_field_names", [])),
        )
        plan = plan_by_identity.get(identity)
        if plan is None or identity in seen:
            raise MathLClampInputError("selected program/profile identity drift or duplicate")
        seen.add(identity)
        if row.get("cell_ids") != [relation["cell_id"] for relation in plan["relations"]]:
            raise MathLClampInputError(f"selected relation mapping drift for {plan['aspect_id']}")
        profile = row.get("profile")
        normalized_profile = validate_profiles(
            [profile], plan["llm_field_names"], require_complete_grid=False
        )
        applied.append((plan, normalized_profile))
    return applied


def execute_audit(
    audit: Mapping,
    items_root: Path,
    *,
    phase: str,
    timeout_seconds: float = 600.0,
    require_canonical_items: bool = True,
    program_root: Path = PROGRAMS_ROOT,
    require_canonical_programs: bool = True,
    audit_source_path: str | None = None,
    profile_selection: Mapping | None = None,
    profile_selection_source: str | None = None,
) -> dict[str, Any]:
    """Execute the full train grid or only frozen train-selected heldout profiles."""

    items, items_path = load_bound_items(
        items_root, phase, require_canonical=require_canonical_items
    )
    plans = build_execution_plan(
        audit,
        program_root=program_root,
        require_canonical_programs=require_canonical_programs,
    )
    audit_fingerprint = _content_fingerprint(audit)
    capability_sources = _capability_sources_from_audit(audit)
    if phase == "compiler_train":
        if profile_selection is not None:
            raise MathLClampInputError("compiler-train may not use an output-derived profile gate")
        scheduled = [
            (plan, build_sentinel_profiles(plan["llm_field_names"])) for plan in plans
        ]
        complete_grid = True
    else:
        if profile_selection is None:
            raise MathLClampInputError("heldout execution requires a frozen train-only profile gate")
        scheduled = apply_profile_selection(
            plans,
            profile_selection,
            audit_fingerprint=audit_fingerprint,
            capability_runtime=capability_sources,
        )
        complete_grid = False

    programs = []
    for plan, profiles in scheduled:
        result = _run_worker(
            plan,
            items,
            profiles,
            capability_sources,
            timeout_seconds=timeout_seconds,
            require_complete_grid=complete_grid,
            program_root=program_root,
            require_canonical_programs=require_canonical_programs,
        )
        result["program_sha256"] = plan["program_sha256"]
        result["llm_field_names"] = plan["llm_field_names"]
        result["relations"] = plan["relations"]
        programs.append(result)

    worker_statuses = Counter(program["worker_status"] for program in programs)
    profile_statuses = Counter(
        profile["measurement_status"]
        for program in programs
        for profile in program.get("profiles", [])
    )
    three_states = Counter()
    for program in programs:
        for profile in program.get("profiles", []):
            three_states.update(profile["summary"]["state_counts"])
    relation_rows = [relation for plan, _profiles in scheduled for relation in plan["relations"]]
    n_profiles = sum(len(profiles) for _plan, profiles in scheduled)
    if not programs:
        status = "no_train-measurable_profiles_selected" if phase == "heldout_pre_reference" else "no_eligible_programs"
    elif worker_statuses["completed"] != len(programs):
        status = "execution_finished_with_worker_failures"
    else:
        status = "conditional_slice_execution_complete"
    return {
        "schema": EXECUTION_SCHEMA,
        "status": status,
        "phase": phase,
        "items_path": str(items_path),
        "n_items": len(items),
        "construct_fidelity_source": audit_source_path,
        "construct_fidelity_fingerprint": audit_fingerprint,
        "profile_selection_source": profile_selection_source,
        "scientific_object": "conditional L-clamp slices g_c(x)=f(x,c)",
        "original_hybrid_execution": False,
        "pure_code_rewrite_claimed": False,
        "whole_construct_fidelity_claimed": False,
        "constant_profiles_frozen_within_each_run": True,
        "sentinel_grid": list(SENTINELS),
        "sentinel_grid_rule": (
            "fixed Cartesian product in declared LLM_FIELDS order; compiler-train runs all "
            "profiles and heldout runs only the frozen train-selected profile"
        ),
        "reference_fields_passed_to_worker": False,
        "outcome_fields_passed_to_worker": False,
        "actual_llm_extractions_passed_to_worker": False,
        "models_or_apis_called_by_runner": False,
        "credentials_inherited_by_worker": False,
        "accelerators_visible_to_worker": False,
        "worker_process_isolated": True,
        "worker_filesystem_isolated": False,
        "worker_network_isolated": False,
        "candidate_trust_model": "trusted frozen historical repository programs, not adversarial code",
        "capability_runtime": capability_sources,
        "ops_corpus_or_retrieval_state_loaded": False,
        "execution_provenance": "retrospective manual historical hybrid under constant LLM-field sentinels",
        "interpretation": (
            "Within-profile variation is executable/code-attributable conditional on the fixed sentinel. "
            "A selected profile means only train measurability and nondegeneracy. Neither a slice nor its "
            "failure establishes the original hybrid, whole-construct verifiability, prompt articulability, "
            "reconstruction, isomorphism, tacitness, or non-verifiability."
        ),
        "summary": {
            "n_unique_programs": len(programs),
            "n_profile_runs": n_profiles,
            "n_relation_mappings": len(relation_rows),
            "worker_status_counts": dict(sorted(worker_statuses.items())),
            "profile_measurement_status_counts": dict(sorted(profile_statuses.items())),
            "three_state_totals": {
                state: three_states[state] for state in ("measured", "abstained", "failed")
            },
        },
        "programs": programs,
    }


def _write_new(path: Path, payload: Mapping, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--source-path", required=True)
    worker.add_argument("--aspect-id", required=True)
    worker.add_argument("--selected-revision", type=int, required=True)
    worker.add_argument("--program-sha256", required=True)
    worker.add_argument("--llm-field-names-json", required=True)
    worker.add_argument("--profile-mode", choices=("complete", "subset"), required=True)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--test-program-root", type=Path)
    worker.add_argument("--allow-test-program-root", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--audit", type=Path, required=True)
    run.add_argument("--items-root", type=Path, default=CANONICAL_ITEMS_ROOT)
    run.add_argument("--phase", choices=sorted(ALLOWED_PHASES), required=True)
    run.add_argument("--profile-selection", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--program-timeout", type=float, default=600.0)
    run.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "worker":
        if args.test_program_root is not None and not args.allow_test_program_root:
            parser.error("--test-program-root requires --allow-test-program-root")
        program_root = args.test_program_root or PROGRAMS_ROOT
        require_canonical = args.test_program_root is None
        payload = json.load(sys.stdin)
        if not isinstance(payload, Mapping) or set(payload) != {
            "items",
            "profiles",
            "capability_sources",
        }:
            raise MathLClampInputError("worker payload has forbidden or missing fields")
        plan = {
            "aspect_id": args.aspect_id,
            "selected_revision": args.selected_revision,
            "source_path": args.source_path,
            "program_sha256": args.program_sha256,
            "llm_field_names": json.loads(args.llm_field_names_json),
        }
        result = execute_one_program(
            plan,
            payload["items"],
            payload["profiles"],
            payload["capability_sources"],
            program_root=program_root,
            require_canonical_programs=require_canonical,
            require_complete_grid=args.profile_mode == "complete",
        )
        _write_new(args.out, result)
        return 0

    if args.program_timeout <= 0:
        parser.error("--program-timeout must be positive")
    audit = _load_json(args.audit)
    selection = _load_json(args.profile_selection) if args.profile_selection else None
    result = execute_audit(
        audit,
        args.items_root,
        phase=args.phase,
        timeout_seconds=args.program_timeout,
        audit_source_path=str(args.audit),
        profile_selection=selection,
        profile_selection_source=str(args.profile_selection) if args.profile_selection else None,
    )
    _write_new(args.out, result, force=args.force)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
