"""Strictly merge the independent static math construct-fidelity audits.

This module performs no candidate execution and reads no items, judgments,
outputs, correlations, or reconstruction results. With no guarded cross-audit
overlay, its output is explicitly provisional and pending cross-audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

SCHEMA = "metric-seam.math-construct-fidelity-merged.v1"
AUDIT_SCHEMA = "metric-seam.math-static-construct-fidelity.v1"
SEED_SCHEMA = "metric-seam.hierarchy-math-seed-map.v1"
OVERLAY_SCHEMA = "metric-seam.math-static-construct-fidelity-cross-adjudication-merged.v1"
TASK = "math-stackexchange"
LEVELS = ("R1", "R2", "R3")
ROOT = Path(__file__).resolve().parents[2]
VERDICTS = {"exact", "partial", "mismatch", "no_candidate_bounded_non_discovery"}
SCOPE_BY_VERDICT = {
    "exact": "whole_construct",
    "partial": "subrelation_only",
    "mismatch": "none",
    "no_candidate_bounded_non_discovery": "none",
}
AUDIT_TOP_KEYS = {
    "schema", "status", "design_scope", "task", "levels", "source_candidate_map",
    "panel_content_sha256", "n_rows", "audit_inputs", "forbidden_inputs",
    "execution_performed", "ops_math_source", "ops_math_sha256", "provenance",
    "adjudication_policy", "audited_depth_vocabulary", "capability_limit", "counts",
    "interpretation", "rows",
}
ROW_KEYS = {
    "cell_id", "task", "level", "metric_name", "metric_description", "candidate",
    "requested_relation", "implemented_relations", "residual_construct", "verdict",
    "scope", "eligible_for_relation_local_execution", "audited_depth",
    "polarity_aggregation_applicability_caveats", "justification", "interpretation",
}
CANDIDATE_KEYS = {
    "aspect_id", "source_heading", "selected_revision", "source_path",
    "program_sha256", "historical_hybrid_provenance",
    "llm_fields_excluded_from_implemented_relations",
}
ALLOWED_OVERLAY_FIELDS = {
    "verdict", "scope", "eligible_for_relation_local_execution", "audited_depth",
    "implemented_relations", "residual_construct",
    "polarity_aggregation_applicability_caveats", "justification",
}
EXPECTED_SEED_INPUT_FIELDS = ["id", "task", "level", "construct", "description"]
EXPECTED_SEED_FORBIDDEN_INPUTS = [
    "items or item identifiers",
    "reference judgments",
    "outcome labels",
    "heldout identifiers",
    "program outputs",
    "correlations or performance summaries",
    "reconstruction or isomorphism results",
]
EXPECTED_AUDIT_INPUTS = [
    "hierarchy construct name and description",
    "selected historical program revision source",
    "methods/metric_seam/hybrids/ops_math.py implementation",
]
EXPECTED_AUDIT_FORBIDDEN_INPUTS = [
    "candidate program execution",
    "items or item identifiers",
    "reference judgments or outcome labels",
    "heldout identifiers or outputs",
    "program outputs or correlations",
    "reconstruction or isomorphism results",
    "model/API calls",
]


class MathFidelityError(ValueError):
    """Raised when a frozen static source artifact cannot be joined strictly."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _summarize(rows: Sequence[Mapping]) -> dict:
    verdicts = Counter(str(row["verdict"]) for row in rows)
    depths = Counter(
        "null" if row["audited_depth"] is None else str(row["audited_depth"])
        for row in rows
    )
    eligible_depths = Counter(
        str(row["audited_depth"])
        for row in rows if row["eligible_for_relation_local_execution"]
    )
    n = len(rows)
    retrieved = sum(row["candidate"] is not None for row in rows)
    eligible = sum(bool(row["eligible_for_relation_local_execution"]) for row in rows)
    return {
        "n_cells": n,
        "n_retrieved_candidates": retrieved,
        "verdicts": dict(sorted(verdicts.items())),
        "eligible_for_relation_local_execution": eligible,
        "eligible_fraction_of_cells": _fraction(eligible, n),
        "eligible_fraction_of_retrieved_candidates": _fraction(eligible, retrieved),
        "audited_depths": dict(sorted(depths.items())),
        "eligible_audited_depths": dict(sorted(eligible_depths.items())),
    }


def _audit_counts(rows: Sequence[Mapping], levels: Sequence[str]) -> dict:
    result = {
        level: _summarize([row for row in rows if row["level"] == level])
        for level in levels
    }
    result["overall"] = _summarize(rows)
    return result


def _safe_source(relative: str, *, label: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise MathFidelityError(f"{label} escapes the repository: {relative}") from error
    if not path.is_file():
        raise MathFidelityError(f"{label} is missing: {relative}")
    return path


def _validate_seed_and_panel(panel: Mapping, seed_map: Mapping) -> tuple[dict, dict]:
    # The global panel validator evolves with newer generators.  This merger
    # instead validates the frozen math task and its exact seed binding below,
    # so another task's later metadata migration cannot change this result.
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise MathFidelityError("unexpected hierarchy panel schema")
    if not isinstance(panel.get("panel_content_sha256"), str):
        raise MathFidelityError("hierarchy panel has no content identity")
    if set(seed_map) != {
        "schema", "status", "design_scope", "panel_schema", "panel_content_sha256",
        "hierarchy_frame", "task", "levels", "n_cells", "n_historical_program_families",
        "n_historical_program_variants", "input_fields_used", "forbidden_inputs",
        "provenance", "retrieval_policy", "capability_library", "summary", "rows",
    }:
        raise MathFidelityError("math seed-map top-level shape drifted")
    if (
        seed_map.get("schema") != SEED_SCHEMA
        or seed_map.get("status")
        != "retrospective-candidate-seeds-pending-independent-construct-fidelity-audit"
        or seed_map.get("design_scope")
        != "outcome_blind_static_source_and_capability_metadata_only"
        or seed_map.get("panel_schema") != panel.get("schema")
        or seed_map.get("task") != TASK
        or seed_map.get("levels") != list(LEVELS)
        or seed_map.get("n_cells") != 90
    ):
        raise MathFidelityError("unexpected math seed-map schema or scope")
    if seed_map.get("panel_content_sha256") != panel.get("panel_content_sha256"):
        raise MathFidelityError("math seed map is bound to another panel")
    provenance = seed_map.get("provenance", {})
    if seed_map.get("input_fields_used") != EXPECTED_SEED_INPUT_FIELDS:
        raise MathFidelityError("math seed retrieval input fields drifted")
    if seed_map.get("forbidden_inputs") != EXPECTED_SEED_FORBIDDEN_INPUTS:
        raise MathFidelityError("math seed retrieval forbidden-input contract drifted")
    for field in (
        "candidate_execution", "construct_fidelity_adjudication",
        "prompt_articulability_evaluation",
    ):
        if provenance.get(field) is not False:
            raise MathFidelityError(f"math seed retrieval crossed forbidden boundary: {field}")
    rows = seed_map.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise MathFidelityError("math seed map must contain exactly 90 rows")
    seeds = {str(row.get("cell_id")): row for row in rows}
    if len(seeds) != 90:
        raise MathFidelityError("math seed map contains duplicate cell IDs")
    cells = {
        str(cell["id"]): cell for cell in panel["cells"] if cell.get("task") == TASK
    }
    if len(cells) != 90 or set(cells) != set(seeds):
        raise MathFidelityError("math seed-map/panel cell identities do not match exactly")
    for cell_id, seed in seeds.items():
        cell = cells[cell_id]
        for seed_field, panel_field in (
            ("task", "task"), ("level", "level"), ("metric_name", "construct"),
            ("metric_description", "description"),
        ):
            if seed.get(seed_field) != cell.get(panel_field):
                raise MathFidelityError(f"{cell_id}: seed/panel {seed_field} drifted")
    return seeds, cells


def _validate_candidate(
    cell_id: str, candidate: Mapping | None, selected_seed: Mapping | None,
) -> dict | None:
    if selected_seed is None:
        if candidate is not None:
            raise MathFidelityError(f"{cell_id}: abstained seed acquired a candidate")
        return None
    if not isinstance(candidate, Mapping) or set(candidate) != CANDIDATE_KEYS:
        raise MathFidelityError(f"{cell_id}: candidate shape drifted")
    expected = {
        "aspect_id": selected_seed.get("aspect_id"),
        "source_heading": selected_seed.get("source_heading"),
        "selected_revision": selected_seed.get("selected_revision"),
        "source_path": selected_seed.get("source_path"),
        "historical_hybrid_provenance": selected_seed.get("hybrid_provenance", {}).get(
            "historical_construction"
        ),
        "llm_fields_excluded_from_implemented_relations": selected_seed.get(
            "hybrid_provenance", {}
        ).get("llm_field_names"),
    }
    for field, value in expected.items():
        if candidate.get(field) != value:
            raise MathFidelityError(f"{cell_id}: candidate/seed {field} drifted")
    source = _safe_source(str(candidate["source_path"]), label=f"{cell_id} candidate source")
    if candidate.get("program_sha256") != _sha256(source):
        raise MathFidelityError(f"{cell_id}: candidate source digest drifted")
    return copy.deepcopy(dict(candidate))


def _validate_row(row: Mapping, seed: Mapping) -> dict:
    cell_id = str(seed["cell_id"])
    if set(row) != ROW_KEYS:
        raise MathFidelityError(f"{cell_id}: audit row shape drifted")
    for field in ("cell_id", "task", "level", "metric_name", "metric_description"):
        expected = {
            "cell_id": seed["cell_id"],
            "task": TASK,
            "level": seed["level"],
            "metric_name": seed["metric_name"],
            "metric_description": seed["metric_description"],
        }[field]
        if row.get(field) != expected:
            raise MathFidelityError(f"{cell_id}: row {field} drifted")
    candidate = _validate_candidate(cell_id, row.get("candidate"), seed.get("selected_seed"))
    verdict = row.get("verdict")
    if verdict not in VERDICTS or row.get("scope") != SCOPE_BY_VERDICT.get(verdict):
        raise MathFidelityError(f"{cell_id}: invalid verdict/scope combination")
    eligible = row.get("eligible_for_relation_local_execution")
    if not isinstance(eligible, bool) or eligible != (verdict in {"exact", "partial"}):
        raise MathFidelityError(f"{cell_id}: eligibility contradicts verdict")
    if candidate is None and verdict != "no_candidate_bounded_non_discovery":
        raise MathFidelityError(f"{cell_id}: no-candidate seed was relabeled")
    if candidate is not None and verdict == "no_candidate_bounded_non_discovery":
        raise MathFidelityError(f"{cell_id}: selected candidate was relabeled no-candidate")
    depth = row.get("audited_depth")
    if candidate is None:
        if depth is not None:
            raise MathFidelityError(f"{cell_id}: no-candidate row has a depth")
    elif isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 4:
        raise MathFidelityError(f"{cell_id}: candidate depth must be an integer 0..4")
    for field in ("requested_relation", "residual_construct", "justification", "interpretation"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise MathFidelityError(f"{cell_id}: missing {field}")
    implemented = row.get("implemented_relations")
    caveats = row.get("polarity_aggregation_applicability_caveats")
    if not isinstance(implemented, list) or not all(
        isinstance(value, str) and value.strip() for value in implemented
    ):
        raise MathFidelityError(f"{cell_id}: implemented_relations shape drifted")
    if (candidate is None) != (implemented == []):
        raise MathFidelityError(f"{cell_id}: candidate/relation inventory contradicts")
    if not isinstance(caveats, list) or not caveats or not all(
        isinstance(value, str) and value.strip() for value in caveats
    ):
        raise MathFidelityError(f"{cell_id}: caveat shape drifted")
    if candidate is not None:
        relation_text = " ".join(implemented)
        excluded = candidate["llm_fields_excluded_from_implemented_relations"]
        if any(str(field) in relation_text for field in excluded):
            raise MathFidelityError(f"{cell_id}: implemented relation names an excluded LLM field")
    normalized = copy.deepcopy(dict(row))
    normalized["candidate"] = candidate
    return normalized


def _validate_level_audits(
    audits: Sequence[Mapping], seeds: Mapping[str, Mapping],
) -> tuple[list[dict], str, str]:
    if len(audits) != 2:
        raise MathFidelityError("expected the frozen R1/R2 audit and independent R3 audit")
    common_fields = (
        "source_candidate_map", "audit_inputs", "forbidden_inputs", "ops_math_source", "ops_math_sha256",
        "provenance", "adjudication_policy", "audited_depth_vocabulary", "capability_limit",
    )
    baseline = audits[0]
    observed_levels: set[str] = set()
    raw_rows: list[Mapping] = []
    for audit in audits:
        if set(audit) != AUDIT_TOP_KEYS:
            raise MathFidelityError("math level-audit top-level shape drifted")
        if (
            audit.get("schema") != AUDIT_SCHEMA
            or audit.get("status") != "complete_static_code_only_adjudication_pre_execution"
            or audit.get("design_scope") != "outcome_blind_static_construct_fidelity"
            or audit.get("task") != TASK
            or audit.get("execution_performed") is not False
        ):
            raise MathFidelityError("unexpected math audit schema, scope, or execution state")
        if audit.get("audit_inputs") != EXPECTED_AUDIT_INPUTS:
            raise MathFidelityError("math audit input contract drifted")
        if audit.get("forbidden_inputs") != EXPECTED_AUDIT_FORBIDDEN_INPUTS:
            raise MathFidelityError("math audit forbidden-input contract drifted")
        if Path(str(audit.get("source_candidate_map", ""))).name != (
            "math_stackexchange_seed_map_v1.json"
        ):
            raise MathFidelityError("math audit is bound to another seed map")
        if audit.get("panel_content_sha256") != baseline.get("panel_content_sha256"):
            raise MathFidelityError("math level audits use different panels")
        for field in common_fields:
            if audit.get(field) != baseline.get(field):
                raise MathFidelityError(f"math level audits disagree on {field}")
        levels = audit.get("levels")
        if not isinstance(levels, list) or not levels or not set(levels) <= set(LEVELS):
            raise MathFidelityError("math audit declares invalid levels")
        if observed_levels & set(levels):
            raise MathFidelityError("math level audits overlap")
        observed_levels.update(levels)
        rows = audit.get("rows")
        if not isinstance(rows, list) or audit.get("n_rows") != len(rows):
            raise MathFidelityError("math audit row count drifted")
        if Counter(str(row.get("level")) for row in rows) != Counter({
            level: 30 for level in levels
        }):
            raise MathFidelityError("math audit does not have 30 disjoint rows per level")
        if audit.get("counts") != _audit_counts(rows, levels):
            raise MathFidelityError("math source-audit counts drifted")
        raw_rows.extend(rows)
    if observed_levels != set(LEVELS):
        raise MathFidelityError("math audits do not cover exactly R1/R2/R3")
    raw_by_id = {str(row.get("cell_id")): row for row in raw_rows}
    if len(raw_rows) != 90 or len(raw_by_id) != 90 or set(raw_by_id) != set(seeds):
        raise MathFidelityError("math audit/seed identities do not close at 90 cells")
    rows = [_validate_row(raw_by_id[cell_id], seed) for cell_id, seed in seeds.items()]
    ops_source = str(baseline["ops_math_source"])
    ops_path = _safe_source(ops_source, label="ops_math source")
    ops_sha = _sha256(ops_path)
    if baseline.get("ops_math_sha256") != ops_sha:
        raise MathFidelityError("ops_math source digest drifted")
    return rows, ops_source, ops_sha


def _overlay_counts(rows: Sequence[Mapping]) -> dict:
    retrieved = [row for row in rows if row.get("candidate") is not None]
    eligible = [row for row in retrieved if row["eligible_for_relation_local_execution"]]
    return {
        "retrieved_candidates": len(retrieved),
        "retrieved_verdicts": dict(sorted(Counter(row["verdict"] for row in retrieved).items())),
        "retrieved_depths": dict(
            sorted(Counter(str(row["audited_depth"]) for row in retrieved).items())
        ),
        "eligible_depths": dict(
            sorted(Counter(str(row["audited_depth"]) for row in eligible).items())
        ),
        "eligible_for_relation_local_execution": len(eligible),
    }


def _apply_overlay(rows: Sequence[Mapping], overlay: Mapping | None) -> tuple[list[dict], int]:
    output = {str(row["cell_id"]): copy.deepcopy(dict(row)) for row in rows}
    if overlay is None:
        return list(output.values()), 0
    if set(overlay) != {
        "schema", "status", "task", "levels", "design_scope", "source_records",
        "forbidden_inputs", "forbidden_inputs_used", "candidate_execution_performed",
        "candidate_import_performed", "model_or_api_calls_performed", "accelerators_used",
        "review_coverage", "before_counts", "after_counts_if_overlay_applied",
        "changes", "interpretation",
    }:
        raise MathFidelityError("math cross-audit overlay shape drifted")
    if (
        overlay.get("schema") != OVERLAY_SCHEMA
        or overlay.get("status") != "complete_guarded_static_cross_audit"
        or overlay.get("task") != TASK
        or overlay.get("levels") != list(LEVELS)
        or overlay.get("design_scope")
        != "outcome_blind_static_code_only_cross_adjudication"
        or overlay.get("forbidden_inputs") != EXPECTED_AUDIT_FORBIDDEN_INPUTS
        or overlay.get("forbidden_inputs_used") is not False
        or overlay.get("candidate_execution_performed") is not False
        or overlay.get("candidate_import_performed") is not False
        or overlay.get("model_or_api_calls_performed") is not False
        or overlay.get("accelerators_used") is not False
    ):
        raise MathFidelityError("math cross-audit overlay is not a sealed static audit")
    records = overlay.get("source_records")
    if not isinstance(records, list) or len(records) != 2:
        raise MathFidelityError("math cross-audit source records are incomplete")
    if [record.get("levels") for record in records if isinstance(record, Mapping)] != [
        ["R1", "R2"], ["R3"]
    ]:
        raise MathFidelityError("math cross-audit source levels drifted")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "levels", "source_audit", "source_overlay"
        }:
            raise MathFidelityError("invalid math cross-audit source record")
        for label in ("source_audit", "source_overlay"):
            identity = record.get(label)
            if not isinstance(identity, Mapping) or set(identity) != {"path", "sha256"}:
                raise MathFidelityError("invalid math cross-audit source identity")
            path = _safe_source(str(identity["path"]), label=f"cross-audit {label}")
            if identity.get("sha256") != _sha256(path):
                raise MathFidelityError(f"math cross-audit {label} digest drifted")
    coverage = overlay.get("review_coverage")
    if coverage != {
        "source_rows": 90,
        "retrieved_candidates_reviewed": 47,
        "changed_rows": 21,
        "unchanged_retrieved_rows": 26,
        "all_retrieved_candidates_reviewed": True,
    }:
        raise MathFidelityError("math cross-audit coverage is incomplete")
    if overlay.get("before_counts") != _overlay_counts(list(output.values())):
        raise MathFidelityError("math cross-audit before counts drifted")
    changes = overlay.get("changes")
    if not isinstance(changes, list) or len(changes) != coverage["changed_rows"]:
        raise MathFidelityError("math cross-audit overlay has an invalid changes list")
    seen: set[str] = set()
    for change in changes:
        if set(change) != {
            "cell_id", "candidate_guard", "change_kind", "before", "after", "reason"
        }:
            raise MathFidelityError("math cross-audit change shape drifted")
        cell_id = str(change.get("cell_id"))
        before, after = change.get("before"), change.get("after")
        if cell_id in seen or cell_id not in output:
            raise MathFidelityError(f"invalid/duplicate overlay cell {cell_id}")
        seen.add(cell_id)
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or not after
            or set(before) != set(after)
            or not set(after) <= ALLOWED_OVERLAY_FIELDS
            or before == after
            or not isinstance(change.get("change_kind"), str)
            or not change["change_kind"].strip()
            or not isinstance(change.get("reason"), str)
            or not change["reason"].strip()
        ):
            raise MathFidelityError(f"{cell_id}: invalid guarded before/after change")
        candidate = output[cell_id].get("candidate")
        guard = change.get("candidate_guard")
        if candidate is None or not isinstance(guard, Mapping) or guard != {
            "aspect_id": candidate["aspect_id"],
            "source_path": candidate["source_path"],
            "program_sha256": candidate["program_sha256"],
        }:
            raise MathFidelityError(f"{cell_id}: candidate guard drifted")
        for field, old in before.items():
            if output[cell_id].get(field) != old:
                raise MathFidelityError(f"{cell_id}: overlay before-value drift for {field}")
        output[cell_id].update(after)
        # Recheck the coupled adjudication fields after every accepted change.
        verdict = output[cell_id]["verdict"]
        if (
            verdict not in VERDICTS
            or output[cell_id]["scope"] != SCOPE_BY_VERDICT[verdict]
            or output[cell_id]["eligible_for_relation_local_execution"]
            != (verdict in {"exact", "partial"})
        ):
            raise MathFidelityError(f"{cell_id}: overlay leaves an incoherent verdict state")
        depth = output[cell_id]["audited_depth"]
        if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 4:
            raise MathFidelityError(f"{cell_id}: overlay leaves an invalid candidate depth")
    if overlay.get("after_counts_if_overlay_applied") != _overlay_counts(list(output.values())):
        raise MathFidelityError("math cross-audit after counts drifted")
    return list(output.values()), len(changes)


def merge_math_audits(
    panel: Mapping, seed_map: Mapping, level_audits: Sequence[Mapping], *,
    overlay: Mapping | None = None, sources: Mapping | None = None,
) -> dict:
    seeds, _cells = _validate_seed_and_panel(panel, seed_map)
    rows, ops_source, ops_sha = _validate_level_audits(level_audits, seeds)
    if seed_map.get("panel_content_sha256") != level_audits[0].get("panel_content_sha256"):
        raise MathFidelityError("seed map and math audits are bound to different panels")
    rows, n_changes = _apply_overlay(rows, overlay)
    rows = [_validate_row(row, seeds[str(row["cell_id"])]) for row in rows]
    rows.sort(key=lambda row: (LEVELS.index(row["level"]), row["cell_id"]))
    summary = _summarize(rows)
    summary.update({
        "whole_construct_exact_count": sum(row["verdict"] == "exact" for row in rows),
        "n_unique_eligible_programs": len({
            row["candidate"]["aspect_id"] for row in rows
            if row["eligible_for_relation_local_execution"]
        }),
        "by_level": {
            level: _summarize([row for row in rows if row["level"] == level])
            for level in LEVELS
        },
    })
    cross_status = "complete" if overlay is not None else "pending_independent_cross_audit"
    return {
        "schema": SCHEMA,
        "status": (
            "static_construct_fidelity_complete_pre_execution"
            if overlay is not None else "provisional_static_merge_pending_cross_audit"
        ),
        "task": TASK,
        "design_scope": "blind_static_construct_fidelity",
        "cross_audit": {
            "status": cross_status,
            "n_guarded_changes": n_changes,
            "provisional_until_complete": overlay is None,
        },
        "sources": dict(sources or {}),
        "panel_content_sha256": panel["panel_content_sha256"],
        "hierarchy_frame": seed_map["hierarchy_frame"],
        "ops_math_source": ops_source,
        "ops_math_sha256": ops_sha,
        "execution_performed": False,
        "items_loaded": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "program_outputs_loaded": False,
        "external_supervision": False,
        "depth_vocabulary": level_audits[0]["audited_depth_vocabulary"],
        "capability_limit": level_audits[0]["capability_limit"],
        "provenance": level_audits[0]["provenance"],
        "summary": summary,
        "interpretation": (
            "This is a static code-only relation-fidelity audit over a retrospective manual "
            "historical hybrid bank. Partial denotes only a named subrelation. It is not code "
            "execution, verifiability performance, prompt articulability, reconstruction, "
            "isomorphism, codability, automatic discovery, or evidence of tacitness."
        ),
        "rows": rows,
    }


def _source_record(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha256(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--seed-map", type=Path, required=True)
    parser.add_argument("--level-audit", type=Path, action="append", required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    source_paths = {
        "panel": args.panel,
        "seed_map": args.seed_map,
        **{f"level_audit_{index + 1}": path for index, path in enumerate(args.level_audit)},
    }
    if args.overlay:
        source_paths["cross_audit_overlay"] = args.overlay
    payload = merge_math_audits(
        _load(args.panel), _load(args.seed_map), [_load(path) for path in args.level_audit],
        overlay=_load(args.overlay) if args.overlay else None,
        sources={name: _source_record(path) for name, path in source_paths.items()},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
