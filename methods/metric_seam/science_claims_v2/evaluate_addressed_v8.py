#!/usr/bin/env python3
"""Evaluate science-v8 prompt assertions against two explicitly non-truth code views.

The corrected relation parser is a non-gating audit of the model-selected spans.  The
historical corrected full-paper verifier is a comparator.  Neither is an external anchor
or supervised ground truth.  Tiny or constant-support cells are reported but not
estimated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from . import addressed_pipeline_v8 as pipeline


ROOT = pipeline.ROOT
DEFAULT_OLD_CODE = (
    pipeline.DEFAULT_HISTORICAL_CODE_COMPARATOR
)
EVALUATION_SCHEMA = "science-articulability-addressed-evaluation-v8"
MIN_ESTIMATING_N = 20


def _binary_comparison(
    left: list[bool], right: list[bool], *, minimum_n: int = MIN_ESTIMATING_N
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("binary comparison inputs differ in length")
    counts = {
        "both_positive": sum(a and b for a, b in zip(left, right)),
        "left_only": sum(a and not b for a, b in zip(left, right)),
        "right_only": sum(not a and b for a, b in zip(left, right)),
        "both_negative": sum(not a and not b for a, b in zip(left, right)),
    }
    descriptive = {
        "n": len(left),
        "left_positive": sum(left),
        "right_positive": sum(right),
        "agreement_count": counts["both_positive"] + counts["both_negative"],
        "confusion": counts,
    }
    if len(left) < minimum_n:
        return {
            **descriptive,
            "estimate_status": "not_estimated",
            "estimate_reason": f"tiny_support_n_below_{minimum_n}",
            "phi": None,
        }
    if len(set(left)) < 2 or len(set(right)) < 2:
        return {
            **descriptive,
            "estimate_status": "not_estimated",
            "estimate_reason": "all_negative_all_positive_or_constant_support",
            "phi": None,
        }
    a = counts["both_positive"]
    b = counts["left_only"]
    c = counts["right_only"]
    d = counts["both_negative"]
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return {
        **descriptive,
        "estimate_status": "estimated",
        "estimate_reason": None,
        "phi": (a * d - b * c) / denominator if denominator else None,
    }


def _old_code_map(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema_version: str,
    expected_source_sha256: str,
) -> tuple[str, dict[str, bool], dict[str, Any]]:
    actual_sha = pipeline.hash_file(path)
    if actual_sha != expected_sha256:
        raise ValueError("old code comparator SHA does not match its explicit binding")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema_version:
        raise ValueError("old code comparator schema does not match its explicit binding")
    comparator_input = payload.get("input")
    if (
        not isinstance(comparator_input, dict)
        or comparator_input.get("sha256") != expected_source_sha256
    ):
        raise ValueError(
            "old code comparator payload input SHA does not match the v8 source SHA"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("old code comparator lacks a records list")
    by_id: dict[str, bool] = {}
    for row in records:
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str) or paper_id in by_id:
            raise ValueError("old code comparator paper IDs are invalid or duplicated")
        by_id[paper_id] = bool(row.get("certificate_count", 0) > 0)
    provenance = {
        "original_decomposition_discovery": "manual_historical",
        "source_artifact_provenance": payload.get("provenance"),
        "source_artifact_pipeline_status": payload.get("pipeline_status"),
        "v8_analysis_pipeline_status": "selected",
        "selection_mode": "retrospective_seed",
        "automatically_discovered_by_v8": False,
        "interpretation": (
            "The existing deep full-paper verifier is analyzed as if selected by the "
            "pipeline while preserving that its original decomposition was manual/historical."
        ),
    }
    return str(payload.get("schema_version") or "unknown"), by_id, provenance


def evaluate(
    bundle: Path,
    normalized_path: Path,
    old_code_path: Path,
    *,
    expected_old_code_sha256: str | None = None,
    expected_old_code_schema_version: str | None = None,
    expected_old_code_source_sha256: str | None = None,
) -> dict[str, Any]:
    manifest, requests, abstentions = pipeline.verify_bundle(bundle)
    manifest_sha = pipeline.hash_file(bundle / "manifest.json")
    normalized = (
        pipeline._read_jsonl(normalized_path) if normalized_path.exists() else []
    )
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in normalized:
        rid = row.get("request_id")
        if rid in seen or rid not in requests:
            raise ValueError("normalized evaluation rows are duplicated or outside bundle")
        pipeline._verify_normalized_row(
            row,
            request=requests[rid],
            manifest=manifest,
            bundle_manifest_sha256=manifest_sha,
        )
        seen.add(rid)
        result = row["result"]
        prompt_asserted = result["prompt_asserted_relation_certificate_count"] > 0
        code_audit_verified = any(
            match["witness_kind"] == pipeline.PROMPT_CERTIFICATE_TYPE
            and match["code_relation_audit"]["status"] == "verified"
            for match in result["matches"]
        )
        audit_statuses = [
            match["code_relation_audit"]["status"]
            for match in result["matches"]
            if match["witness_kind"] == pipeline.PROMPT_CERTIFICATE_TYPE
        ]
        rows.append({
            "paper_id": result["paper_id"],
            "request_id": rid,
            "source_index": requests[rid]["source_index"],
            "prompt_asserted_relation_certificate_present": prompt_asserted,
            "non_gating_code_relation_audit_verified_present": code_audit_verified,
            "prompt_certificate_code_audit_statuses": audit_statuses,
        })
    bound_old = manifest["historical_code_comparator"]
    bound_path = pipeline._resolve_recorded_path(bound_old["path"])
    if old_code_path.resolve() == bound_path.resolve():
        expected_sha = bound_old["sha256"]
        expected_schema = bound_old["schema_version"]
        expected_source_sha = bound_old["input_source_sha256"]
    else:
        if (
            not expected_old_code_sha256
            or not expected_old_code_schema_version
            or not expected_old_code_source_sha256
        ):
            raise ValueError(
                "an arbitrary old-code path requires explicit expected file SHA, "
                "schema, and source SHA"
            )
        expected_sha = expected_old_code_sha256
        expected_schema = expected_old_code_schema_version
        expected_source_sha = expected_old_code_source_sha256
    if expected_source_sha != manifest["input"]["source_file_sha256"]:
        raise ValueError(
            "old code comparator expected source SHA differs from bundle source SHA"
        )
    old_schema, old_code, old_code_provenance = _old_code_map(
        old_code_path,
        expected_sha256=expected_sha,
        expected_schema_version=expected_schema,
        expected_source_sha256=expected_source_sha,
    )
    missing_old = [row["paper_id"] for row in rows if row["paper_id"] not in old_code]
    if missing_old:
        raise ValueError("completed v8 rows are absent from the old code comparator")
    for row in rows:
        row["old_corrected_code_relation_certificate_present"] = old_code[
            row["paper_id"]
        ]

    prompt = [row["prompt_asserted_relation_certificate_present"] for row in rows]
    old = [row["old_corrected_code_relation_certificate_present"] for row in rows]
    audit_status_counts: dict[str, int] = {
        status: sum(
            row["prompt_certificate_code_audit_statuses"].count(status)
            for row in rows
        )
        for status in ("verified", "diverged", "abstained")
    }
    return {
        "schema_version": EVALUATION_SCHEMA,
        "objective": "unsupervised_reconstruction_comparison_no_external_anchor",
        "interpretation": {
            "prompt": "articulability implementation",
            "non_gating_code_relation_audit": (
                "conditional construct-fidelity diagnostic over model-selected exact "
                "spans; not a response gate and not an independent reconstruction arm"
            ),
            "old_corrected_code": (
                "historical full-paper same-evidence-content code comparator; not "
                "supervised ground truth"
            ),
            "isomorphism": (
                "semantic reconstruction may be estimated, but full/input-representation "
                "isomorphism is unavailable because addressed JSONL and continuous text differ"
            ),
        },
        "bindings": {
            "bundle_manifest_sha256": manifest_sha,
            "normalized_results_sha256": (
                pipeline.hash_file(normalized_path) if normalized_path.exists() else None
            ),
            "old_code_path": pipeline.display_path(old_code_path),
            "old_code_sha256": pipeline.hash_file(old_code_path),
            "old_code_schema_version": old_schema,
            "old_code_expected_sha256": expected_sha,
            "old_code_expected_schema_version": expected_schema,
            "old_code_expected_source_sha256": expected_source_sha,
            "evaluator_sha256": pipeline.hash_file(Path(__file__)),
        },
        "old_fullpaper_code_comparator_provenance": old_code_provenance,
        "support": {
            "corpus_total": manifest["strata"]["observed"]["corpus_records"],
            "prompt_eligible_total": len(requests),
            "structural_abstention_total": len(abstentions),
            "completed_prompt_results": len(rows),
            "uncompleted_prompt_eligible": len(requests) - len(rows),
        },
        "prompt_certificate_code_relation_audit_status_counts": audit_status_counts,
        "conditional_prompt_certificate_construct_fidelity": {
            "diagnostic_status": "descriptive_conditional_not_isomorphism_estimate",
            "conditioning_event": "prompt_asserted_relation_certificate",
            "prompt_asserted_certificate_count": sum(audit_status_counts.values()),
            "code_parser_verified_count": audit_status_counts["verified"],
            "code_parser_diverged_count": audit_status_counts["diverged"],
            "code_parser_abstained_count": audit_status_counts["abstained"],
            "verified_fraction_of_prompt_assertions": (
                audit_status_counts["verified"] / sum(audit_status_counts.values())
                if sum(audit_status_counts.values())
                else None
            ),
            "right_only_is_structurally_undefined": True,
            "phi_is_not_reported": True,
        },
        "prompt_selected_code_confirmed_hybrid_witnesses": {
            "count": audit_status_counts["verified"],
            "type": "prompt_selected_code_confirmed_hybrid_witness",
            "scope": "document_local_relation_local_parser_scoped",
            "external_scientific_truth": False,
            "effect_on_prompt_acceptance": "none_non_gating",
            "claim_licensed": (
                "On these selected instances, the prompt assertion is separately "
                "code-confirmed conditional on prompt-selected spans by the frozen "
                "local relation parser."
            ),
        },
        "comparisons": {
            "prompt_assertion_vs_old_corrected_code_comparator": _binary_comparison(
                prompt, old
            ),
        },
        "rows": rows,
    }


def render_report(payload: dict[str, Any]) -> str:
    support = payload["support"]
    comparisons = payload["comparisons"]
    fidelity = payload["conditional_prompt_certificate_construct_fidelity"]
    provenance = payload["old_fullpaper_code_comparator_provenance"]
    hybrid = payload["prompt_selected_code_confirmed_hybrid_witnesses"]
    lines = []
    for name, value in comparisons.items():
        lines.append(
            f"- `{name}`: n={value['n']}, left+={value['left_positive']}, "
            f"right+={value['right_positive']}, status={value['estimate_status']}, "
            f"phi={value['phi']}"
        )
    return f"""# Science articulability v8 — reconstruction evaluation

This evaluation uses no supervised or external scientific anchor. Prompt assertions are
the articulability channel. The corrected relation parser over hydrated selected spans is
a separate, non-gating **conditional construct-fidelity diagnostic**. It is defined only
after a prompt selects a certificate, so it is not a symmetric reconstruction arm and no
phi/isomorphism statistic is computed for prompt-versus-audit. The old corrected full-paper
verifier is the independent same-evidence-content comparator, not ground truth. It sees
continuous abstract/body while the prompt sees addressed JSONL, so semantic reconstruction
does not license full input-representation isomorphism.

Comparator provenance remains explicit: the deep full-paper decomposition was
`{provenance['original_decomposition_discovery']}`; v8 gives it pipeline status
`{provenance['v8_analysis_pipeline_status']}` under selection mode
`{provenance['selection_mode']}`. V8 does not claim it automatically discovered that
decomposition.

- Corpus: {support['corpus_total']:,}
- Prompt-eligible body-present stratum: {support['prompt_eligible_total']:,}
- Missing-body structural abstentions: {support['structural_abstention_total']:,}
- Completed prompt results: {support['completed_prompt_results']:,}
- Conditional parser confirmation: {fidelity['code_parser_verified_count']} verified /
  {fidelity['prompt_asserted_certificate_count']} prompt-asserted certificates
  ({fidelity['code_parser_diverged_count']} diverged,
  {fidelity['code_parser_abstained_count']} abstained)
- Prompt-selected, code-confirmed hybrid witnesses: **{hybrid['count']}**
  (document-local, relation-local, parser-scoped; not external scientific truth and not a
  prompt-response gate)

{chr(10).join(lines)}

Cells with fewer than {MIN_ESTIMATING_N} paired observations or constant/all-negative
support are explicitly `not_estimated`; descriptive counts remain available.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=pipeline.DEFAULT_OUT)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--old-code", type=Path, default=DEFAULT_OLD_CODE)
    parser.add_argument("--expected-old-code-sha256")
    parser.add_argument("--expected-old-code-schema-version")
    parser.add_argument("--expected-old-code-source-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(
        args.bundle.resolve(), args.normalized.resolve(), args.old_code.resolve(),
        expected_old_code_sha256=args.expected_old_code_sha256,
        expected_old_code_schema_version=args.expected_old_code_schema_version,
        expected_old_code_source_sha256=args.expected_old_code_source_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({
        "completed_prompt_results": payload["support"]["completed_prompt_results"],
        "comparisons": {
            key: value["estimate_status"]
            for key, value in payload["comparisons"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
