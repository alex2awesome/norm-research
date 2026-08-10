"""CPU-only resolution audit for existing Math, Science, and Patent targets.

This diagnostic asks only whether a frozen code-side target varies enough on
TRAIN to support a later comparison.  It does not certify a construct, select
held-out programs, or call a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Hashable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
SCHEMA = "metric-seam.technical-target-discrimination.v1"


class DiscriminationAuditError(ValueError):
    pass


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiscriminationAuditError(f"expected object: {path}")
    return value


def summarize_vector(
    *, task: str, vector_id: str, values: Sequence[Hashable | None], depth: int | None
) -> dict:
    if not values:
        raise DiscriminationAuditError("target vector is empty")
    measured = [value for value in values if value is not None]
    for value in measured:
        if isinstance(value, float) and not math.isfinite(value):
            raise DiscriminationAuditError("target vector contains a non-finite value")
        if isinstance(value, (dict, list, set)):
            raise DiscriminationAuditError("target values must be hashable scalars")
    counts = Counter(measured)
    mode_count = counts.most_common(1)[0][1] if counts else 0
    coverage = len(measured) / len(values)
    mode_fraction = mode_count / len(measured) if measured else None
    unique = len(counts)
    failed = []
    if coverage < 0.90:
        failed.append("coverage_below_0.90")
    if unique < 3:
        failed.append("fewer_than_3_values")
    if mode_fraction is None or mode_fraction > 0.85:
        failed.append("mode_fraction_above_0.85")
    return {
        "task": task,
        "vector_id": vector_id,
        "decision_contributing_depth": depth,
        "n_items": len(values),
        "n_measured": len(measured),
        "coverage": coverage,
        "n_unique_values": unique,
        "mode_count": mode_count,
        "mode_fraction": mode_fraction,
        "minimum": min(measured) if measured and all(isinstance(x, (int, float)) for x in measured) else None,
        "maximum": max(measured) if measured and all(isinstance(x, (int, float)) for x in measured) else None,
        "passes_resolution_diagnostic": not failed,
        "failed_checks": failed,
    }


def math_vectors(execution: Mapping, gate: Mapping) -> list[dict]:
    selected = {
        row["aspect_id"]: row["profile"]["profile_id"]
        for row in gate.get("selected_program_profiles", [])
    }
    rows = []
    for program in execution.get("programs", []):
        aspect = program["aspect_id"]
        profile_id = selected.get(aspect)
        if profile_id is None:
            continue
        profile = next(
            (row for row in program["profiles"] if row["profile_id"] == profile_id), None
        )
        if profile is None:
            raise DiscriminationAuditError(f"selected Math profile missing: {aspect}/{profile_id}")
        values = [
            row.get("score") if row.get("measurement_state") == "measured" else None
            for row in profile["rows"]
        ]
        depths = [relation.get("audited_depth") for relation in program.get("relations", [])]
        depth = max((value for value in depths if isinstance(value, int)), default=None)
        rows.append(
            summarize_vector(
                task="math-stackexchange",
                vector_id=f"{aspect}:{profile_id}",
                values=values,
                depth=depth,
            )
        )
    return rows


def patent_vectors(payload: Mapping, *, lane: str) -> list[dict]:
    relation_names: set[str] = set()
    for row in payload.get("rows", []):
        relation_names.update(row.get("result", {}).get("relation_values", {}))
    output = []
    for relation in sorted(relation_names):
        values = [
            row.get("result", {}).get("relation_values", {}).get(relation, {}).get("value")
            for row in payload["rows"]
        ]
        depths = [
            row.get("result", {}).get("maximum_decision_contributing_depth")
            for row in payload["rows"]
        ]
        depth = max((value for value in depths if isinstance(value, int)), default=None)
        output.append(
            summarize_vector(
                task="patents",
                vector_id=f"{lane}:{relation}",
                values=values,
                depth=depth,
            )
        )
    return output


def science_vectors(payload: Mapping) -> list[dict]:
    rows = payload.get("rows", [])
    depth = max(
        (row.get("effective_code_depth", 0) for row in payload.get("relation_mappings", [])),
        default=None,
    )
    return [
        summarize_vector(
            task="peer-review-fullarticle",
            vector_id="numeric_comparative:verifier_status",
            values=[row.get("verifier_status") for row in rows],
            depth=depth,
        ),
        summarize_vector(
            task="peer-review-fullarticle",
            vector_id="numeric_comparative:certificate_count",
            values=[row.get("certificate_count") for row in rows],
            depth=depth,
        ),
    ]


def build_audit(
    *, math_execution: Path, math_gate: Path, patent_structure: Path,
    patent_graph: Path, science_execution: Path,
) -> dict:
    vectors = [
        *math_vectors(_load(math_execution), _load(math_gate)),
        *patent_vectors(_load(patent_structure), lane="claim_structure"),
        *patent_vectors(_load(patent_graph), lane="claim_graph"),
        *science_vectors(_load(science_execution)),
    ]
    by_task = {}
    for task in sorted({row["task"] for row in vectors}):
        task_rows = [row for row in vectors if row["task"] == task]
        by_task[task] = {
            "n_vectors": len(task_rows),
            "n_passing_resolution_diagnostic": sum(
                row["passes_resolution_diagnostic"] for row in task_rows
            ),
            "n_failing_resolution_diagnostic": sum(
                not row["passes_resolution_diagnostic"] for row in task_rows
            ),
        }
    return {
        "schema": SCHEMA,
        "status": "cpu_train_diagnostic_complete",
        "estimand": "target resolution on frozen compiler-TRAIN outputs",
        "model_calls_performed": False,
        "gpu_used": False,
        "thresholds": {
            "minimum_coverage": 0.90,
            "minimum_unique_values": 3,
            "maximum_mode_fraction": 0.85,
        },
        "summary_by_task": by_task,
        "vectors": vectors,
        "claim_limits": [
            "Passing is not construct fidelity or verifiability.",
            "Failing means the target is unresolved on this TRAIN corpus, not tacit or uncodable.",
            "Multiple hierarchy metrics can share one executable vector and are not duplicated here.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-execution", type=Path, default=BASE / "math_stackexchange_lclamp_compiler_train_v1.json")
    parser.add_argument("--math-gate", type=Path, default=BASE / "math_stackexchange_lclamp_train_profile_gate_v1.json")
    parser.add_argument("--patent-structure", type=Path, default=BASE / "patents_claim_structure_compiler_train_v14.json")
    parser.add_argument("--patent-graph", type=Path, default=BASE / "patents_claim_graph_additive_compiler_train_v2.json")
    parser.add_argument("--science-execution", type=Path, default=BASE / "peer_review_science_fullarticle_compiler_train_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_audit(
        math_execution=args.math_execution,
        math_gate=args.math_gate,
        patent_structure=args.patent_structure,
        patent_graph=args.patent_graph,
        science_execution=args.science_execution,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary_by_task"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
