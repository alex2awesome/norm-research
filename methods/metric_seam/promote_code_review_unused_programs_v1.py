"""Build an additive 90-row code-review fidelity gate with nine new mappings.

The historical merged audit is retained byte-for-byte.  This builder first
applies the already-certified six independent demotions, then overlays the
nine separately audited unused-program mappings.  The result is a new static
execution gate (59/90), not a rewrite of the canonical 50/90 funnel and not an
execution or reconstruction result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from methods.metric_seam.hierarchy_code_runner import validate_canonical_audit


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
CANONICAL = BASE / "code_review_construct_fidelity_v2.json"
CORRECTED = BASE / "code_review_corrected_funnel_v1.json"
ADDITIVE_AUDIT = BASE / "code_review_unused_program_construct_cross_audit_v1.json"
OUT = BASE / "code_review_construct_fidelity_additive_unused_programs_v1.json"


class PromotionError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PromotionError(f"expected object: {path}")
    return value


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        verdicts = Counter(row["verdict"] for row in group)
        retrieved = [row for row in group if row["candidate"] is not None]
        eligible = [row for row in group if row["eligible_for_relation_local_execution"]]
        all_depths = Counter(
            "null" if row["candidate"] is None else str(row["audited_depth"])
            for row in group
        )
        eligible_depths = Counter(str(row["audited_depth"]) for row in eligible)
        return {
            "n_metrics": len(group),
            "verdict_counts": dict(sorted(verdicts.items())),
            "retrieved_candidate_count": len(retrieved),
            "retrieved_candidate_fraction": round(len(retrieved) / len(group), 6),
            "relation_local_static_fidelity_count": len(eligible),
            "relation_local_static_fidelity_fraction": round(len(eligible) / len(group), 6),
            "whole_construct_exact_count": sum(row["verdict"] == "exact" for row in group),
            "whole_construct_exact_fraction": round(
                sum(row["verdict"] == "exact" for row in group) / len(group), 6
            ),
            "audited_depth_counts_all": dict(sorted(all_depths.items())),
            "audited_depth_counts_eligible": dict(sorted(eligible_depths.items())),
        }

    result = summarize(rows)
    result["retrieved_mismatch_count"] = sum(row["verdict"] == "mismatch" for row in rows)
    result["retrieved_mismatch_fraction"] = round(
        result["retrieved_mismatch_count"] / result["retrieved_candidate_count"], 6
    )
    result["n_unique_eligible_programs"] = len(
        {
            row["candidate"]["aspect_id"]
            for row in rows
            if row["eligible_for_relation_local_execution"]
        }
    )
    result["by_level"] = {
        level: summarize([row for row in rows if row["level"] == level])
        for level in ("R1", "R2", "R3")
    }
    return result


def build() -> dict[str, Any]:
    canonical = _load(CANONICAL)
    corrected = _load(CORRECTED)
    additive = _load(ADDITIVE_AUDIT)
    validate_canonical_audit(canonical)
    if (
        corrected.get("schema") != "metric-seam.code-review-corrected-funnel.v1"
        or additive.get("schema")
        != "metric-seam.code-review-unused-program-construct-cross-audit.v1"
        or additive.get("status")
        != "independent_static_construct_audit_complete_pre_execution"
    ):
        raise PromotionError("unexpected corrected/additive audit input")
    demoted = {row["cell_id"] for row in corrected["removed_mappings"]["static"]}
    promoted = {row["cell_id"]: row for row in additive["rows"]}
    if len(demoted) != 6 or len(promoted) != 9:
        raise PromotionError("expected six demotions and nine promotions")

    rows = []
    for original in canonical["rows"]:
        row = json.loads(json.dumps(original))
        cell_id = row["cell_id"]
        if cell_id in demoted:
            row.update(
                {
                    "verdict": "mismatch",
                    "scope": "none",
                    "eligible_for_relation_local_execution": False,
                    "rationale": (
                        "Superseded by the independent corrected-funnel audit: the "
                        "historical program misses the requested relation."
                    ),
                    "interpretation": (
                        "Rejected historical mapping; this is bounded instrument "
                        "noncoverage, not tacitness."
                    ),
                }
            )
        if cell_id in promoted:
            accepted = promoted[cell_id]
            row.update(
                {
                    "candidate": {
                        "aspect_id": accepted["candidate_aspect_id"],
                        "source_path": accepted["candidate_source"],
                        "source_sha256": accepted["candidate_source_sha256"],
                    },
                    "requested_relation": accepted["requested_relation"],
                    "implemented_relations": [accepted["implemented_relation"]],
                    "verdict": "partial",
                    "scope": "subrelation_only",
                    "eligible_for_relation_local_execution": True,
                    "audited_depth": accepted["audited_depth"],
                    "dependency_applicability_caveats": [
                        accepted["applicability"],
                        accepted["abstention"],
                        accepted["independent_rationale"],
                    ],
                    "rationale": accepted["independent_rationale"],
                    "interpretation": (
                        "Additive independently accepted relation-local mapping; "
                        "partial never establishes whole-construct verifiability."
                    ),
                }
            )
        rows.append(row)

    if len(rows) != 90 or len({row["cell_id"] for row in rows}) != 90:
        raise PromotionError("promoted audit does not cover exactly 90 cells")
    result = {
        **{key: value for key, value in canonical.items() if key not in {"rows", "summary"}},
        "status": "additive_static_construct_fidelity_complete_pre_execution",
        "design_scope": "corrected_canonical_plus_independently_audited_additive_programs",
        "source_seed_map": str(ADDITIVE_AUDIT.relative_to(ROOT)),
        "source_level_audits": [
            str(CANONICAL.relative_to(ROOT)),
            str(CORRECTED.relative_to(ROOT)),
        ],
        "source_cross_audit_adjudication": str(ADDITIVE_AUDIT.relative_to(ROOT)),
        "n_adjudicated_changes": 15,
        "execution_performed": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "external_supervision": False,
        "interpretation": (
            "Additive static execution gate: six historical wrong-relation mappings "
            "remain demoted and nine repaired unused-program mappings are admitted as "
            "partial subrelations. The canonical 50/90 artifact remains unchanged."
        ),
        "summary": _summary(rows),
        "rows": rows,
    }
    validate_canonical_audit(result)
    if result["summary"]["relation_local_static_fidelity_count"] != 59:
        raise PromotionError("additive static union must be 59/90")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != serialized:
            raise PromotionError(f"checked artifact differs: {args.out}")
        return
    args.out.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
