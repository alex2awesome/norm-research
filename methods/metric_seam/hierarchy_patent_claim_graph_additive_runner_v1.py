"""Execute the additive patent claim graphs on one frozen text-only split.

The runner accepts exactly ``item_key`` and ``ctext``.  Compiler-train and
heldout-pre-reference are separate phases; no reference, outcome, prompt,
prior-art, examiner, model, API, accelerator, or external supervision channel
is available here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.patent_claim_graph_additive_v1 import (
    RELATIONS,
    SCHEMA as PROGRAM_SCHEMA,
    analyze_patent_claim_graph,
)


SCHEMA = "metric-seam.hierarchy-patent-claim-graph-additive-execution.v2"
MANIFEST_SCHEMA = "metric-seam.hierarchy-shared-items.v1"
PHASES = {"compiler_train", "heldout_pre_reference"}


class PatentClaimGraphExecutionError(ValueError):
    """Raised when a split violates the frozen text-only contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_items(items: Sequence[Mapping], *, phase: str) -> None:
    if phase not in PHASES:
        raise PatentClaimGraphExecutionError(f"unsupported phase: {phase}")
    if not isinstance(items, list) or not items:
        raise PatentClaimGraphExecutionError("items must be a nonempty JSON list")
    prefix = "train_" if phase == "compiler_train" else "heldout_"
    seen = set()
    for index, row in enumerate(items):
        if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
            raise PatentClaimGraphExecutionError(
                f"item {index} must expose exactly item_key and ctext"
            )
        if (
            not isinstance(row["item_key"], str)
            or not row["item_key"].startswith(prefix)
            or row["item_key"] in seen
        ):
            raise PatentClaimGraphExecutionError(
                f"item {index} has invalid or duplicate opaque split key"
            )
        seen.add(row["item_key"])
        if not isinstance(row["ctext"], str) or not row["ctext"].strip():
            raise PatentClaimGraphExecutionError(f"item {index} has invalid ctext")


def validate_manifest(manifest: Mapping, items: Sequence[Mapping], *, phase: str) -> int:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("task") != "patents":
        raise PatentClaimGraphExecutionError("unexpected patent shared-item manifest")
    representation = manifest.get("representation")
    if not isinstance(representation, Mapping) or (
        representation.get("field") != "ctext"
        or representation.get("projection")
        != "source_text[:max_chars] before exact deduplication"
        or representation.get("same_bytes_required_for_prompt_and_code") is not True
    ):
        raise PatentClaimGraphExecutionError("shared-ctext representation contract drifted")
    max_chars = representation.get("max_chars")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise PatentClaimGraphExecutionError("invalid ctext cap")
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(key) is not expected
        for key, expected in {
            "outcome_columns_emitted": False,
            "source_identifiers_emitted": False,
            "compiler_receives_heldout_text": False,
            "external_supervision_used": False,
        }.items()
    ):
        raise PatentClaimGraphExecutionError("manifest violates blind text-only policy")
    selection = manifest.get("selection")
    expected_count = "train_n" if phase == "compiler_train" else "heldout_n"
    if not isinstance(selection, Mapping) or selection.get(expected_count) != len(items):
        raise PatentClaimGraphExecutionError("manifest split count does not match items")
    if selection.get("outcome_or_reference_values_used") is not False:
        raise PatentClaimGraphExecutionError("manifest selection was not reference blind")
    if any(len(row["ctext"]) > max_chars for row in items):
        raise PatentClaimGraphExecutionError("ctext exceeds frozen character cap")
    return max_chars


def _relation_summary(rows: Sequence[Mapping], relation_id: str) -> dict:
    certificates = [
        certificate
        for row in rows
        if isinstance(row.get("result"), Mapping)
        for certificate in row["result"]["certificates"]
        if certificate["relation"] == relation_id
    ]
    items_with = sum(
        any(
            certificate["relation"] == relation_id
            for certificate in row.get("result", {}).get("certificates", [])
        )
        for row in rows
        if isinstance(row.get("result"), Mapping)
    )
    kinds = Counter(row["kind"] for row in certificates)
    return {
        "n_items_with_finite_certificates": items_with,
        "n_items_without_finite_certificates": len(rows) - items_with,
        "n_certificates": len(certificates),
        "certificate_kind_counts": dict(sorted(kinds.items())),
    }


def execute_split(
    items: Sequence[Mapping],
    *,
    phase: str,
    representation_max_chars: int,
) -> dict:
    validate_items(items, phase=phase)
    rows = []
    failures = Counter()
    for item in items:
        at_cap = len(item["ctext"]) == representation_max_chars
        try:
            result = analyze_patent_claim_graph(item["ctext"])
        except Exception as exc:  # fail-closed measured execution receipt
            failures[type(exc).__name__] += 1
            rows.append(
                {
                    "item_key": item["item_key"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "representation": {
                        "ctext_chars": len(item["ctext"]),
                        "declared_max_chars": representation_max_chars,
                        "at_declared_character_cap": at_cap,
                    },
                    "result": None,
                }
            )
            continue
        rows.append(
            {
                "item_key": item["item_key"],
                "status": (
                    "relation_abstained"
                    if result["claim_count"] == 0
                    else "measured_with_possible_truncation"
                    if at_cap
                    else "measured"
                ),
                "error_type": None,
                "representation": {
                    "ctext_chars": len(item["ctext"]),
                    "declared_max_chars": representation_max_chars,
                    "at_declared_character_cap": at_cap,
                    "possibly_truncated_by_declared_character_cap": at_cap,
                    "whole_source_claim_set_completeness_established": False,
                },
                "result": result,
            }
        )

    status_counts = Counter(row["status"] for row in rows)
    relation_ids = [row["relation_id"] for row in RELATIONS]
    return {
        "schema": SCHEMA,
        "program_schema": PROGRAM_SCHEMA,
        "phase": phase,
        "design": {
            "input_fields": ["item_key", "ctext"],
            "exact_frozen_ctext_used": True,
            "outcome_or_reference_values_loaded": False,
            "prompt_outputs_loaded": False,
            "prior_art_or_examiner_evidence_loaded": False,
            "external_supervision_used": False,
            "model_or_api_calls_made": False,
            "accelerators_used": False,
            "whole_patent_score_emitted": False,
            "codability_reconstruction_or_isomorphism_measured": False,
            "absence_outside_local_recognized_grammar_permitted": False,
            "declared_representation_max_chars": representation_max_chars,
        },
        "summary": {
            "n_items": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_types": dict(sorted(failures.items())),
            "items_at_declared_character_cap": sum(
                row["representation"]["at_declared_character_cap"] for row in rows
            ),
            "relation_certificates": {
                relation_id: _relation_summary(rows, relation_id)
                for relation_id in relation_ids
            },
        },
        "rows": rows,
    }


def build_execution(
    items: Sequence[Mapping],
    manifest: Mapping,
    *,
    phase: str,
    item_source_bytes: bytes,
    manifest_source_bytes: bytes,
    item_source_path: str,
    manifest_source_path: str,
) -> dict:
    try:
        source_items = json.loads(item_source_bytes)
        source_manifest = json.loads(manifest_source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatentClaimGraphExecutionError("source bytes are not valid JSON") from exc
    if source_items != items or source_manifest != manifest:
        raise PatentClaimGraphExecutionError(
            "parsed inputs do not equal the exact bound source bytes"
        )
    validate_items(items, phase=phase)
    max_chars = validate_manifest(manifest, items, phase=phase)
    artifact = execute_split(
        items, phase=phase, representation_max_chars=max_chars
    )
    artifact["sources"] = {
        "items": {
            "path": item_source_path,
            "sha256": _sha256_bytes(item_source_bytes),
            "n_items": len(items),
        },
        "manifest": {
            "path": manifest_source_path,
            "sha256": _sha256_bytes(manifest_source_bytes),
            "schema": manifest["schema"],
        },
        "program": {
            "path": "methods/metric_seam/patent_claim_graph_additive_v1.py",
            "sha256": _sha256_bytes(
                Path(__file__).with_name("patent_claim_graph_additive_v1.py").read_bytes()
            ),
            "schema": PROGRAM_SCHEMA,
        },
        "runner": {
            "path": "methods/metric_seam/hierarchy_patent_claim_graph_additive_runner_v1.py",
            "sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "schema": SCHEMA,
        },
    }
    return artifact


def _write_new(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    item_bytes = args.items.read_bytes()
    manifest_bytes = args.manifest.read_bytes()
    items = json.loads(item_bytes)
    manifest = json.loads(manifest_bytes)
    artifact = build_execution(
        items,
        manifest,
        phase=args.phase,
        item_source_bytes=item_bytes,
        manifest_source_bytes=manifest_bytes,
        item_source_path=str(args.items),
        manifest_source_path=str(args.manifest),
    )
    _write_new(args.output, artifact)
    print(json.dumps({"output": str(args.output), **artifact["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
