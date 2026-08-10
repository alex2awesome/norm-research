"""Validate and merge blind static construct-fidelity audits.

The level audits were intentionally written independently and may use either a
nested ``candidate`` object or the equivalent flat candidate fields.  This
module normalizes them into the one schema consumed by the sealed code runner,
while checking every row against the frozen source-only seed map.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "metric-seam.code-review-construct-fidelity-merged.v1"
ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("R1", "R2", "R3")
VERDICTS = {"exact", "partial", "mismatch", "no_candidate_bounded_non_discovery"}
SCOPE_BY_VERDICT = {
    "exact": "whole_construct",
    "partial": "subrelation_only",
    "mismatch": "none",
    "no_candidate_bounded_non_discovery": "none",
}


class FidelityAuditError(ValueError):
    """Raised when a level audit is inconsistent with the frozen seed map."""


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(row: Mapping) -> dict[str, str] | None:
    nested = row.get("candidate")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise FidelityAuditError(f"{row.get('cell_id')}: candidate must be an object or null")
        aspect_id = nested.get("aspect_id")
        source_path = nested.get("source_path")
    else:
        aspect_id = row.get("candidate_aspect_id")
        source_path = row.get("candidate_source_path")
    if aspect_id is None and source_path is None:
        return None
    if not isinstance(aspect_id, str) or not isinstance(source_path, str):
        raise FidelityAuditError(f"{row.get('cell_id')}: incomplete candidate identity")
    return {"aspect_id": aspect_id, "source_path": source_path}


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _apply_adjudication(rows_by_id: Mapping[str, Mapping], adjudication: Mapping | None) -> tuple[dict, int]:
    rows = {cell_id: copy.deepcopy(row) for cell_id, row in rows_by_id.items()}
    if adjudication is None:
        return rows, 0
    if adjudication.get("design_scope") != "blind_static_construct_fidelity_cross_audit":
        raise FidelityAuditError("unexpected cross-audit adjudication scope")
    if adjudication.get("forbidden_inputs_used") is not False:
        raise FidelityAuditError("cross-audit adjudication used forbidden inputs")
    if adjudication.get("candidate_execution_performed") is not False:
        raise FidelityAuditError("cross-audit adjudication must precede execution")
    changes = adjudication.get("changes")
    if not isinstance(changes, list):
        raise FidelityAuditError("cross-audit adjudication has no changes list")
    allowed = {"verdict", "scope", "eligible_for_relation_local_execution", "audited_depth"}
    seen = set()
    for change in changes:
        cell_id = str(change.get("cell_id"))
        if cell_id in seen or cell_id not in rows:
            raise FidelityAuditError(f"invalid/duplicate adjudication cell: {cell_id}")
        seen.add(cell_id)
        if change.get("adjudication") != "accepted":
            raise FidelityAuditError(f"non-accepted change cannot enter canonical audit: {cell_id}")
        before, after = change.get("before"), change.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise FidelityAuditError(f"{cell_id}: adjudication needs before/after objects")
        if not set(before) <= allowed or not set(after) <= allowed or not after:
            raise FidelityAuditError(f"{cell_id}: adjudication changes unsupported fields")
        for field, expected in before.items():
            if rows[cell_id].get(field) != expected:
                raise FidelityAuditError(
                    f"{cell_id}: adjudication before-value drift for {field}"
                )
        rows[cell_id].update(after)
    return rows, len(changes)


def _normalize_row(row: Mapping, seed: Mapping) -> dict:
    cell_id = str(seed["cell_id"])
    if str(row.get("cell_id")) != cell_id:
        raise FidelityAuditError(f"row identity mismatch for {cell_id}")
    for field, expected in (
        ("level", seed["level"]),
        ("metric_name", seed["metric_name"]),
    ):
        if row.get(field) != expected:
            raise FidelityAuditError(
                f"{cell_id}: {field} mismatch: {row.get(field)!r} != {expected!r}"
            )

    verdict = row.get("verdict")
    if verdict not in VERDICTS:
        raise FidelityAuditError(f"{cell_id}: invalid verdict {verdict!r}")
    scope = row.get("scope")
    if scope != SCOPE_BY_VERDICT[verdict]:
        raise FidelityAuditError(
            f"{cell_id}: verdict {verdict} requires scope {SCOPE_BY_VERDICT[verdict]}"
        )
    eligible = row.get("eligible_for_relation_local_execution")
    if not isinstance(eligible, bool) or eligible != (verdict in {"exact", "partial"}):
        raise FidelityAuditError(f"{cell_id}: eligibility contradicts verdict {verdict}")

    selected = seed.get("selected_seed")
    candidate = _candidate(row)
    expected_candidate = None if selected is None else {
        "aspect_id": selected["aspect_id"],
        "source_path": selected["source_path"],
    }
    if candidate != expected_candidate:
        raise FidelityAuditError(
            f"{cell_id}: audited candidate {candidate!r} does not match seed {expected_candidate!r}"
        )
    if selected is None and verdict != "no_candidate_bounded_non_discovery":
        raise FidelityAuditError(f"{cell_id}: abstained seed must remain bounded non-discovery")
    if selected is not None and verdict == "no_candidate_bounded_non_discovery":
        raise FidelityAuditError(f"{cell_id}: selected seed cannot be relabeled no-candidate")

    depth = row.get("audited_depth")
    if candidate is None:
        if depth is not None:
            raise FidelityAuditError(f"{cell_id}: no-candidate row cannot have audited depth")
    elif isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 4:
        raise FidelityAuditError(f"{cell_id}: candidate row requires audited depth 0..4")

    requested = row.get("requested_relation")
    implemented = row.get("implemented_relations")
    caveats = row.get("dependency_applicability_caveats")
    if not isinstance(requested, str) or not requested.strip():
        raise FidelityAuditError(f"{cell_id}: missing requested relation")
    if not isinstance(implemented, list) or not all(
        isinstance(value, str) and value.strip() for value in implemented
    ):
        raise FidelityAuditError(f"{cell_id}: implemented_relations must be a string list")
    if candidate is not None and not implemented:
        raise FidelityAuditError(f"{cell_id}: candidate audit must describe implemented relations")
    if candidate is None and implemented:
        raise FidelityAuditError(f"{cell_id}: no-candidate row cannot claim implemented relations")
    if not isinstance(caveats, list) or not all(isinstance(value, str) for value in caveats):
        raise FidelityAuditError(f"{cell_id}: caveats must be a string list")

    normalized_candidate = candidate
    if candidate is not None:
        source = (ROOT / candidate["source_path"]).resolve()
        if not source.is_file():
            raise FidelityAuditError(f"{cell_id}: candidate source is missing: {source}")
        normalized_candidate = {
            **candidate,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }

    return {
        "cell_id": cell_id,
        "level": str(seed["level"]),
        "metric_name": str(seed["metric_name"]),
        "metric_description": str(seed["metric_description"]),
        "candidate": normalized_candidate,
        "requested_relation": requested.strip(),
        "implemented_relations": implemented,
        "verdict": verdict,
        "scope": scope,
        "eligible_for_relation_local_execution": eligible,
        "audited_depth": depth,
        "dependency_applicability_caveats": caveats,
        "rationale": str(row.get("rationale") or "").strip(),
        "interpretation": str(row.get("interpretation") or "").strip(),
    }


def merge_audits(seed_map: Mapping, level_audits: Sequence[Mapping], *,
                 audit_sources: Sequence[str] | None = None,
                 adjudication: Mapping | None = None,
                 adjudication_source: str | None = None,
                 seed_map_source: str | None = None) -> dict:
    """Return one canonical 90-row construct-fidelity artifact."""

    seed_rows = seed_map.get("rows")
    if not isinstance(seed_rows, list) or len(seed_rows) != 90:
        raise FidelityAuditError("expected a 90-row code-review seed map")
    seeds = {str(row["cell_id"]): row for row in seed_rows}
    if len(seeds) != 90:
        raise FidelityAuditError("seed map has duplicate cell ids")

    raw_rows = []
    observed_levels = Counter()
    for audit in level_audits:
        if audit.get("design_scope") != "blind_static_construct_fidelity":
            raise FidelityAuditError("every audit must declare blind_static_construct_fidelity")
        if audit.get("execution_performed") not in {None, False}:
            raise FidelityAuditError("construct-fidelity audit must precede candidate execution")
        rows = audit.get("rows")
        if not isinstance(rows, list):
            raise FidelityAuditError("level audit has no rows list")
        raw_rows.extend(rows)
        observed_levels.update(str(row.get("level")) for row in rows)
    if observed_levels != Counter({level: 30 for level in LEVELS}):
        raise FidelityAuditError(f"expected 30 rows per level; found {observed_levels}")
    raw_by_id = {str(row.get("cell_id")): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise FidelityAuditError("level audits contain duplicate cell ids")
    missing, extra = set(seeds) - set(raw_by_id), set(raw_by_id) - set(seeds)
    if missing or extra:
        raise FidelityAuditError(
            f"audit/seed identity mismatch: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
        )
    raw_by_id, n_adjudicated = _apply_adjudication(raw_by_id, adjudication)

    rows = [_normalize_row(raw_by_id[cell_id], seed) for cell_id, seed in seeds.items()]
    rows.sort(key=lambda row: (LEVELS.index(row["level"]), row["cell_id"]))
    verdicts = Counter(row["verdict"] for row in rows)
    retrieved = sum(row["candidate"] is not None for row in rows)
    def summarize(subset: Sequence[Mapping]) -> dict:
        local_verdicts = Counter(row["verdict"] for row in subset)
        local_depths = Counter(
            "null" if row["audited_depth"] is None else str(row["audited_depth"])
            for row in subset
        )
        local_eligible_depths = Counter(
            str(row["audited_depth"])
            for row in subset if row["eligible_for_relation_local_execution"]
        )
        n = len(subset)
        local_retrieved = sum(row["candidate"] is not None for row in subset)
        local_eligible = sum(row["eligible_for_relation_local_execution"] for row in subset)
        return {
            "n_metrics": n,
            "verdict_counts": dict(sorted(local_verdicts.items())),
            "retrieved_candidate_count": local_retrieved,
            "retrieved_candidate_fraction": _fraction(local_retrieved, n),
            "relation_local_static_fidelity_count": local_eligible,
            "relation_local_static_fidelity_fraction": _fraction(local_eligible, n),
            "whole_construct_exact_count": local_verdicts["exact"],
            "whole_construct_exact_fraction": _fraction(local_verdicts["exact"], n),
            "audited_depth_counts_all": dict(sorted(local_depths.items())),
            "audited_depth_counts_eligible": dict(sorted(local_eligible_depths.items())),
        }

    return {
        "schema": SCHEMA,
        "status": "static_construct_fidelity_complete_pre_execution",
        "task": "code-review",
        "design_scope": "blind_static_construct_fidelity",
        "source_seed_map": seed_map_source,
        "source_level_audits": list(audit_sources or []),
        "source_cross_audit_adjudication": adjudication_source,
        "n_adjudicated_changes": n_adjudicated,
        "panel_content_sha256": seed_map.get("panel_content_sha256"),
        "hierarchy_frame": seed_map.get("hierarchy_frame"),
        "execution_performed": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "external_supervision": False,
        "depth_vocabulary": {
            "0": "surface lexical operation",
            "1": "parsed document/code structure",
            "2": "cross-span, cross-file, or cross-section relation",
            "3": "formal solver or evidence-graph execution",
            "4": "environment or test execution",
        },
        "interpretation": (
            "Static fidelity is an execution gate, not a result. Partial means a named executable "
            "subrelation may be replayed; it does not establish whole-construct verifiability, "
            "prompt reconstruction, isomorphism, or tacitness."
        ),
        "summary": {
            **summarize(rows),
            "retrieved_mismatch_count": verdicts["mismatch"],
            "retrieved_mismatch_fraction": _fraction(verdicts["mismatch"], retrieved),
            "n_unique_eligible_programs": len({
                row["candidate"]["aspect_id"] for row in rows
                if row["eligible_for_relation_local_execution"]
            }),
            "by_level": {
                level: summarize([row for row in rows if row["level"] == level])
                for level in LEVELS
            },
        },
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-map", type=Path, required=True)
    parser.add_argument("--level-audit", type=Path, action="append", required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --force")
    payload = merge_audits(
        _load(args.seed_map),
        [_load(path) for path in args.level_audit],
        audit_sources=[str(path) for path in args.level_audit],
        adjudication=_load(args.adjudication) if args.adjudication else None,
        adjudication_source=str(args.adjudication) if args.adjudication else None,
        seed_map_source=str(args.seed_map),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
