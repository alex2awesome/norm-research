"""Execute the patent claim-structure seed on one frozen text-only split.

The runner accepts only opaque item keys plus ``ctext``.  It neither loads nor
joins prompt/reference values, outcomes, source identifiers, examiner records,
or prior-art evidence.  Compiler-train and heldout-pre-reference are explicit
phases so relation selection can be frozen before the heldout command is run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.patent_claim_structure import (
    RELATIONS,
    SCHEMA as PROGRAM_SCHEMA,
    analyze_patent_ctext,
)


SCHEMA = "metric-seam.hierarchy-patent-claim-structure-execution.v3"
PHASES = {"compiler_train", "heldout_pre_reference"}


class PatentExecutionError(ValueError):
    """Raised when a split exposes fields outside the frozen text-only contract."""


def validate_items(items: Sequence[Mapping], *, phase: str) -> None:
    if phase not in PHASES:
        raise PatentExecutionError(f"unsupported phase: {phase}")
    if not isinstance(items, list) or not items:
        raise PatentExecutionError("items must be a nonempty JSON list")
    expected_prefix = "train_" if phase == "compiler_train" else "heldout_"
    seen = set()
    for index, row in enumerate(items):
        if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
            raise PatentExecutionError(
                f"item {index} must expose exactly item_key and ctext"
            )
        item_key = row["item_key"]
        if not isinstance(item_key, str) or not item_key.startswith(expected_prefix):
            raise PatentExecutionError(f"item {index} has an invalid opaque split key")
        if item_key in seen:
            raise PatentExecutionError(f"duplicate item key: {item_key}")
        seen.add(item_key)
        if not isinstance(row["ctext"], str) or not row["ctext"].strip():
            raise PatentExecutionError(f"item {index} has invalid ctext")


def validate_manifest(manifest: Mapping, items: Sequence[Mapping], *, phase: str) -> int:
    """Validate the representation contract and return its character cap."""

    if manifest.get("schema") != "metric-seam.hierarchy-shared-items.v1":
        raise PatentExecutionError("unexpected shared-item manifest schema")
    if manifest.get("task") != "patents":
        raise PatentExecutionError("shared-item manifest is not for patents")
    representation = manifest.get("representation", {})
    if (
        representation.get("field") != "ctext"
        or representation.get("same_bytes_required_for_prompt_and_code") is not True
    ):
        raise PatentExecutionError("manifest does not require shared prompt/code ctext")
    max_chars = representation.get("max_chars")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise PatentExecutionError("manifest has invalid representation max_chars")
    policy = manifest.get("policy", {})
    if (
        policy.get("outcome_columns_emitted") is not False
        or policy.get("external_supervision_used") is not False
    ):
        raise PatentExecutionError("manifest violates the outcome-blind text-only policy")
    expected_key = "train_n" if phase == "compiler_train" else "heldout_n"
    if manifest.get("selection", {}).get(expected_key) != len(items):
        raise PatentExecutionError("manifest split count does not match items")
    if any(len(row["ctext"]) > max_chars for row in items):
        raise PatentExecutionError("item exceeds the declared ctext character cap")
    return max_chars


def _relation_summary(rows: Sequence[Mapping], relation_id: str) -> dict:
    values = []
    for row in rows:
        result = row.get("result")
        if not isinstance(result, Mapping):
            continue
        relation = result.get("relation_values", {}).get(relation_id, {})
        value = relation.get("value") if isinstance(relation, Mapping) else None
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return {
        "n_measured": len(values),
        "n_abstained": len(rows) - len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "nonconstant": bool(values and min(values) < max(values)),
    }


def execute_split(
    items: Sequence[Mapping],
    *,
    phase: str,
    representation_max_chars: int | None = None,
) -> dict:
    validate_items(items, phase=phase)
    rows = []
    failures = Counter()
    for item in items:
        try:
            result = analyze_patent_ctext(item["ctext"])
        except Exception as exc:  # measured fail-closed execution receipt
            failures[type(exc).__name__] += 1
            rows.append(
                {
                    "item_key": item["item_key"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "result": None,
                }
            )
            continue
        claims = result["claims"]
        at_declared_cap = bool(
            representation_max_chars is not None
            and len(item["ctext"]) == representation_max_chars
        )
        rows.append(
            {
                "item_key": item["item_key"],
                "status": (
                    "relation_abstained"
                    if not claims
                    else "measured_with_possible_truncation"
                    if at_declared_cap
                    else "measured"
                ),
                "error_type": None,
                "representation": {
                    "ctext_chars": len(item["ctext"]),
                    "declared_max_chars": representation_max_chars,
                    "at_declared_character_cap": at_declared_cap,
                    "possibly_truncated_by_declared_character_cap": at_declared_cap,
                    "whole_source_claim_set_completeness_established": False,
                },
                "relation_applicability": {
                    "finite_witnesses_replayable_on_presented_bytes": True,
                    "absence_or_whole_claim_set_inference_permitted": False,
                    "train_gate_scope": (
                        "finite_witnesses_only"
                        if at_declared_cap
                        else "presented_text_relations_and_finite_witnesses"
                    ),
                },
                "result": result,
            }
        )

    relation_ids = [relation["relation_id"] for relation in RELATIONS]
    status_counts = Counter(row["status"] for row in rows)
    at_cap = sum(
        bool(row.get("representation", {}).get("at_declared_character_cap"))
        for row in rows
    )
    certificate_counts = Counter(
        certificate["relation"]
        for row in rows
        if isinstance(row.get("result"), Mapping)
        for certificate in row["result"]["certificates"]
    )
    return {
        "schema": SCHEMA,
        "program_schema": PROGRAM_SCHEMA,
        "phase": phase,
        "design": {
            "input_fields": ["item_key", "ctext"],
            "outcome_or_reference_values_loaded": False,
            "prompt_outputs_loaded": False,
            "prior_art_or_examiner_evidence_loaded": False,
            "external_supervision_used": False,
            "whole_patent_score_emitted": False,
            "declared_representation_max_chars": representation_max_chars,
            "absence_certificate_permitted": False,
            "finite_local_counter_witness_permitted": True,
            "at_cap_is_treated_as_possible_truncation": True,
        },
        "summary": {
            "n_items": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_types": dict(sorted(failures.items())),
            "items_at_declared_character_cap": at_cap,
            "items_measured_with_possible_truncation": status_counts.get(
                "measured_with_possible_truncation", 0
            ),
            "relation_measurement": {
                relation_id: _relation_summary(rows, relation_id)
                for relation_id in relation_ids
            },
            "certificate_counts": dict(sorted(certificate_counts.items())),
        },
        "rows": rows,
    }


def _write_new(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    items = json.loads(args.items.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_items(items, phase=args.phase)
    max_chars = validate_manifest(manifest, items, phase=args.phase)
    result = execute_split(
        items,
        phase=args.phase,
        representation_max_chars=max_chars,
    )
    _write_new(args.output, result)
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
