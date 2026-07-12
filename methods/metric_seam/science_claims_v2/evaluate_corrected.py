#!/usr/bin/env python3
"""Produce the additive, audit-corrected science certificate artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .core_corrected import (
        extract_quantities,
        metamorphic_self_check,
        quantity_anchor_terms,
        verify_document,
    )
    from .evaluate import _summarize, load_unlabelled, sha256
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from methods.metric_seam.science_claims_v2.core_corrected import (
        extract_quantities,
        metamorphic_self_check,
        quantity_anchor_terms,
        verify_document,
    )
    from methods.metric_seam.science_claims_v2.evaluate import _summarize, load_unlabelled, sha256


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/peer-review/peer_review_cv_evidence.jsonl"
DEFAULT_BASELINE = ROOT / "outputs/metric_seam_pilot/science_claims_v2/results.json"
DEFAULT_OUTPUT = ROOT / "outputs/metric_seam_pilot/science_claims_v2_corrected_v2/results.json"
DEFAULT_REPORT = ROOT / "outputs/metric_seam_pilot/science_claims_v2_corrected_v2/REPORT.md"

_AUDIT_CASES = {
    "compact_100k_retained_with_complete_token": "iclr_cYksYKbf6K",
    "compact_6_7B_and_33B_retained_with_complete_tokens": "iclr_FCCeBaFa8M",
    "compact_1_5B_retained_with_complete_token": "iclr_ZPkNrs6aNO",
    "stage_index_removed": "iclr_UKZqSYB2ya",
    "math_superscript_collision_removed": "iclr_DRKkO2Tejc",
    "adapter_task_collision_removed": "iclr_yOOJwR15xg",
}


def _certificate_map(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (record["paper_id"], cert["claim"]["text"], cert["evidence"]["text"]): cert
        for record in records
        for cert in record.get("certificates", [])
    }


def _status_for(records: list[dict[str, Any]], paper_id: str) -> dict[str, Any]:
    record = next(record for record in records if record["paper_id"] == paper_id)
    return {
        "status": record["status"],
        "certificate_count": record.get("certificate_count", 0),
        "certificate_claims": [c["claim"]["text"] for c in record.get("certificates", [])],
    }


def _removed_reasons(
    baseline: dict[tuple[str, str, str], dict[str, Any]],
    corrected_records: list[dict[str, Any]],
) -> dict[str, int]:
    corrected = _certificate_map(corrected_records)
    records_by_id = {record["paper_id"]: record for record in corrected_records}
    reasons: Counter[str] = Counter()
    for paper_id, claim_text, evidence_text in baseline.keys() - corrected.keys():
        record = records_by_id[paper_id]
        exact = next(
            (
                match for match in record.get("matches", [])
                if match["claim"]["text"] == claim_text
                and match["evidence"]["text"] == evidence_text
            ),
            None,
        )
        same_claim = next(
            (match for match in record.get("matches", []) if match["claim"]["text"] == claim_text),
            None,
        )
        if exact:
            reasons[exact["reason"]] += 1
        elif same_claim:
            reasons["matching_changed:" + same_claim["reason"]] += 1
        else:
            reasons["claim_no_longer_selected_as_executable"] += 1
    return dict(sorted(reasons.items()))


def _binding_diagnostic(cert: dict[str, Any]) -> dict[str, Any]:
    claim_text = cert["claim"]["text"]
    evidence_text = cert["evidence"]["text"]
    claim_quantities = extract_quantities(claim_text)
    evidence_quantities = extract_quantities(evidence_text)
    rows = []
    for claim_quantity in claim_quantities:
        equal = [q for q in evidence_quantities if q.unit == claim_quantity.unit and abs(q.value - claim_quantity.value) <= max(1e-9, .005 * max(abs(q.value), abs(claim_quantity.value), 1e-12))]
        rows.append({
            "claim_raw": claim_quantity.raw.strip(),
            "normalized_value": claim_quantity.value,
            "unit": claim_quantity.unit,
            "claim_anchors": list(quantity_anchor_terms(claim_text, claim_quantity)),
            "evidence_candidates": [
                {
                    "raw": q.raw.strip(),
                    "anchors": list(quantity_anchor_terms(evidence_text, q)),
                }
                for q in equal
            ],
        })
    return {"relation": cert["claim"]["relation"], "quantities": rows}


def _diagnostic_sample(certificates: dict[tuple[str, str, str], dict[str, Any]], n: int = 10):
    # Hash ordering makes the sample deterministic without favoring input order or paper ID.
    ordered = sorted(
        certificates.items(),
        key=lambda item: hashlib.sha256("\0".join(item[0]).encode()).hexdigest(),
    )[:n]
    return [
        {
            "paper_id": key[0],
            "claim": key[1],
            "evidence": key[2],
            "diagnostic": _binding_diagnostic(cert),
        }
        for key, cert in ordered
    ]


def _comparison(
    baseline_payload: dict[str, Any],
    corrected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_records = baseline_payload["records"]
    baseline = _certificate_map(baseline_records)
    corrected = _certificate_map(corrected_records)
    before, after = baseline_payload["summary"], _summarize(corrected_records)
    return {
        "before_after": {
            "strong_relation_certificates": [sum(before["certificate_decisions"].values()), sum(after["certificate_decisions"].values())],
            "certified_documents": [sum(r.get("certificate_count", 0) > 0 for r in baseline_records), sum(r.get("certificate_count", 0) > 0 for r in corrected_records)],
            "numeric_certificates": [before["certificate_relations"].get("numeric", 0), after["certificate_relations"].get("numeric", 0)],
            "comparative_certificates": [before["certificate_relations"].get("comparative", 0), after["certificate_relations"].get("comparative", 0)],
            "weak_evidence_links": [before["evidence_link_count"], after["evidence_link_count"]],
            "weak_evidence_link_documents": [before["evidence_link_documents"], after["evidence_link_documents"]],
            "graph_edges": [before["graph_edges"], after["graph_edges"]],
        },
        "certificate_identity_diff": {
            "retained": len(baseline.keys() & corrected.keys()),
            "removed": len(baseline.keys() - corrected.keys()),
            "added": len(corrected.keys() - baseline.keys()),
            "net_change": len(corrected) - len(baseline),
            "key": "paper_id + exact claim text + exact evidence text",
        },
        "removed_certificate_reasons": _removed_reasons(baseline, corrected_records),
        "audit_corpus_cases": {
            name: {
                "paper_id": paper_id,
                "before": _status_for(baseline_records, paper_id),
                "corrected": _status_for(corrected_records, paper_id),
            }
            for name, paper_id in _AUDIT_CASES.items()
        },
        "corrected_strong_diagnostic_sample": _diagnostic_sample(corrected),
    }


def _report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    comparison = payload["audit_comparison"]
    before_after = comparison["before_after"]
    diff = comparison["certificate_identity_diff"]
    rows = "\n".join(
        f"| {label.replace('_', ' ')} | {values[0]} | {values[1]} | {values[1] - values[0]:+d} |"
        for label, values in before_after.items()
    )
    cases = "\n".join(
        f"| {name} | `{value['paper_id']}` | {value['before']['status']} / {value['before']['certificate_count']} | "
        f"{value['corrected']['status']} / {value['corrected']['certificate_count']} |"
        for name, value in comparison["audit_corpus_cases"].items()
    )
    artifacts = "\n".join(
        f"- `{path}`: `{digest}`" for path, digest in sorted(payload["artifacts"].items())
    )
    return f"""# Science full-paper claim verifier — audit-corrected v2.2

**Current corrected result.** The frozen, audited v2 result remains at
`outputs/metric_seam_pilot/science_claims_v2/results.json` and still reproduces its historical
171 count. The intermediate v2.1 correction is also retained for provenance. Neither is
presented as the current certificate count.

This run uses the same 2,400 records and reads only `paper_id`, `abstract`, and `body`; it never
reads `y`. It fixes complete-token quantity parsing, process/document index leakage, TeX
superscript leakage, named-version identifier leakage, and entity binding for small bare
unitless integers.

## Corrected headline

- Strong numeric/comparative relation certificates: **{sum(summary['certificate_decisions'].values())}** across **{sum(r.get('certificate_count', 0) > 0 for r in payload['records'])}** papers
- Numeric: {summary['certificate_relations'].get('numeric', 0)}; comparative: {summary['certificate_relations'].get('comparative', 0)}
- Weaker evidence-link witnesses are preserved as a separate tier: **{summary['evidence_link_count']}** across **{summary['evidence_link_documents']}** papers
- Status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`
- Metamorphic/regression checks: {payload['metamorphic_checks']['passed']} / {payload['metamorphic_checks']['total']}

“Strong” here means that the corrected executable relation fired. It is not external scientific
ground truth and is not an independent semantic-validation label.

## Before / after

| Measure | Audited v2 | Corrected v2.2 | Change |
|---|---:|---:|---:|
{rows}

At exact certificate identity, {diff['retained']} were retained, {diff['removed']} removed, and
{diff['added']} newly exposed by complete suffix parsing (net {diff['net_change']:+d}). Removal
reasons are recorded structurally in `results.json`; no removed certificate is deleted from the
historical artifact.

## Audit counterexamples

| Case | Paper | Before status / certificates | Corrected status / certificates |
|---|---|---:|---:|
{cases}

The five token regressions (`100k`, `33B`, `6.7B`, `1.5B`, `30nm`) and synthetic perturbations
are also executable tests. A deterministic 10-certificate diagnostic sample, including normalized
values and local claim/evidence anchors, is stored under
`audit_comparison.corrected_strong_diagnostic_sample` for independent inspection.

## Frozen inputs and implementation

- Input `{payload['input']['path']}`: `{payload['input']['sha256']}`
- Historical baseline `{payload['baseline']['path']}`: `{payload['baseline']['sha256']}`
{artifacts}
"""


def run(input_path: Path, baseline_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    metamorphic = metamorphic_self_check()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != "science-claims-v2":
        raise ValueError("expected the frozen science-claims-v2 baseline")
    records = [
        verify_document(record["paper_id"], record["abstract"], record["body"])
        for record in load_unlabelled(input_path)
    ]
    source_files = [
        Path(__file__).with_name("core.py"),
        Path(__file__).with_name("core_corrected.py"),
        Path(__file__).with_name("evaluate.py"),
        Path(__file__),
        Path(__file__).with_name("test_science_claims_v2.py"),
        Path(__file__).with_name("test_science_claims_corrected.py"),
    ]
    payload = {
        "schema_version": "science-claims-v2.2-audit-corrected",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "code_verifiability",
        "provenance": "retrospective_seed_audit_correction",
        "pipeline_status": "selected_corrected",
        "external_supervision": "none",
        "label_policy": {
            "forbidden": ["y", "judgement", "acceptance"],
            "loader_allowlist": ["paper_id", "abstract", "body"],
        },
        "correction_scope": [
            "complete quantity token parsing and normalization",
            "process/document/list index filtering",
            "TeX superscript filtering",
            "named-version and slash-composed identifier filtering",
            "small bare integer local-entity binding",
        ],
        "input": {"path": str(input_path.relative_to(ROOT)), "sha256": sha256(input_path)},
        "baseline": {"path": str(baseline_path.relative_to(ROOT)), "sha256": sha256(baseline_path)},
        "artifacts": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "metamorphic_checks": {
            "passed": sum(metamorphic.values()),
            "total": len(metamorphic),
            "checks": metamorphic,
        },
        "summary": _summarize(records),
        "audit_comparison": _comparison(baseline, records),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(
        args.input.resolve(), args.baseline.resolve(), args.output.resolve(), args.report.resolve()
    )
    print(json.dumps({
        "summary": payload["summary"],
        "audit_comparison": payload["audit_comparison"]["before_after"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
