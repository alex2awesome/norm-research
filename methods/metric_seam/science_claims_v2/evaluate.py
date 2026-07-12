#!/usr/bin/env python3
"""Run the selected science claim verifier over full-paper evidence without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .core import metamorphic_self_check, verify_document
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from methods.metric_seam.science_claims_v2.core import metamorphic_self_check, verify_document


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/peer-review/peer_review_cv_evidence.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/metric_seam_pilot/science_claims_v2/results.json"
DEFAULT_REPORT = ROOT / "outputs/metric_seam_pilot/science_claims_v2/REPORT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_unlabelled(path: Path):
    """Yield only authorized text fields; ``y`` and every other field are ignored."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            yield {
                "paper_id": str(raw.get("paper_id") or f"line_{line_number}"),
                "abstract": str(raw.get("abstract") or ""),
                "body": str(raw.get("body") or ""),
            }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(r["status"] for r in records)
    reasons = Counter(r["reason"] for r in records)
    matched_decisions: Counter[str] = Counter()
    matched_relations: Counter[str] = Counter()
    certificate_decisions: Counter[str] = Counter()
    certificate_relations: Counter[str] = Counter()
    quantity_claims = 0
    quantity_matched = 0
    strong_quantity_certificates = 0
    strong_quantity_fully_matched = 0
    graph_edges = 0
    evidence_link_relations: Counter[str] = Counter()
    evidence_link_count = 0
    for record in records:
        graph_edges += record.get("graph", {}).get("edges", 0)
        for cert in record.get("matches", []):
            decision = cert["decision"]
            relation = cert["claim"]["relation"]
            matched_decisions[decision] += 1
            matched_relations[relation] += 1
            if decision in {"supported", "contradicted"}:
                certificate_decisions[decision] += 1
                certificate_relations[relation] += 1
                if cert["checks"]["quantity_required"]:
                    strong_quantity_certificates += 1
                    if cert["checks"]["quantity_matches"] == cert["checks"]["quantity_required"]:
                        strong_quantity_fully_matched += 1
            elif decision == "evidence_link":
                evidence_link_count += 1
                evidence_link_relations[relation] += 1
            if cert["checks"]["quantity_required"]:
                quantity_claims += 1
                if cert["checks"]["quantity_matches"] == cert["checks"]["quantity_required"]:
                    quantity_matched += 1
    total = len(records)
    covered = total - statuses["abstain"]
    body_available = total - reasons["missing_fullpaper_body"]
    certified_docs = sum(1 for r in records if r.get("certificate_count", 0) > 0)
    evidence_link_docs = sum(1 for r in records if r.get("evidence_link_count", 0) > 0)
    return {
        "papers": total,
        "status_counts": dict(sorted(statuses.items())),
        "abstention_reasons": dict(sorted((k, v) for k, v in reasons.items() if k.startswith("missing_") or k.startswith("no_") or k.startswith("abstract_only"))),
        "coverage": round(covered / total, 6) if total else 0.0,
        "body_available": body_available,
        "coverage_given_body": round(covered / body_available, 6) if body_available else 0.0,
        "certified_document_rate": round(certified_docs / total, 6) if total else 0.0,
        "certified_rate_given_body": round(certified_docs / body_available, 6) if body_available else 0.0,
        "certified_rate_given_coverage": round(certified_docs / covered, 6) if covered else 0.0,
        "evidence_link_documents": evidence_link_docs,
        "evidence_link_document_rate": round(evidence_link_docs / total, 6) if total else 0.0,
        "evidence_link_rate_given_body": round(evidence_link_docs / body_available, 6) if body_available else 0.0,
        "evidence_link_count": evidence_link_count,
        "evidence_link_relations": dict(sorted(evidence_link_relations.items())),
        "matched_edge_decisions": dict(sorted(matched_decisions.items())),
        "matched_edge_relations": dict(sorted(matched_relations.items())),
        "certificate_decisions": dict(sorted(certificate_decisions.items())),
        "certificate_relations": dict(sorted(certificate_relations.items())),
        "quantity_bearing_matched_edges": quantity_claims,
        "fully_matched_quantity_edges": quantity_matched,
        "quantity_bearing_relation_certificates": strong_quantity_certificates,
        "fully_matched_quantity_relation_certificates": strong_quantity_fully_matched,
        "graph_edges": graph_edges,
    }


def _report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    artifact_lines = "\n".join(
        f"- `{path}`: `{digest}`" for path, digest in sorted(payload["artifacts"].items())
    )
    meta = payload["metamorphic_checks"]
    return f"""# Science full-paper claim verifier v2

This is an additive replay of the already-selected claim → body-evidence decomposition. It does
not claim that the decomposition was originally discovered automatically. Provenance is
`retrospective_seed`; pipeline status is `selected`.

## Run

- Papers: {s['papers']} (the evaluator never reads `y`)
- Papers with full-paper-derived body sections: {s['body_available']} / {s['papers']}
- Code-verification coverage: {s['coverage']:.1%} of all records; {s['coverage_given_body']:.1%} where a body is available
- Papers with a **strong numeric/comparative relation certificate**: {s['certified_document_rate']:.1%} of all records; {s['certified_rate_given_body']:.1%} where a body is available; {s['certified_rate_given_coverage']:.1%} of covered records
- Papers with a weaker **surface evidence-link witness**: {s['evidence_link_document_rate']:.1%} of all records; {s['evidence_link_rate_given_body']:.1%} where a body is available
- Status counts: `{json.dumps(s['status_counts'], sort_keys=True)}`
- Certified decisions: `{json.dumps(s['certificate_decisions'], sort_keys=True)}`
- Certificate relations: `{json.dumps(s['certificate_relations'], sort_keys=True)}`
- Evidence-link witnesses: {s['evidence_link_count']}; relations: `{json.dumps(s['evidence_link_relations'], sort_keys=True)}`
- All matched-edge decisions (including honest insufficiency): `{json.dumps(s['matched_edge_decisions'], sort_keys=True)}`
- Strong quantity-bearing relation certificates with every normalized obligation matched: {s['fully_matched_quantity_relation_certificates']} / {s['quantity_bearing_relation_certificates']}
- All quantity-bearing matched edges (including insufficient candidates): {s['fully_matched_quantity_edges']} / {s['quantity_bearing_matched_edges']}
- Candidate claim↔evidence graph edges evaluated: {s['graph_edges']}
- Metamorphic checks passed before evaluation: {meta['passed']} / {meta['total']}

## What executes

Abstract result sentences are segmented into relation-bearing claims. Body sentences are indexed
with per-document BM25 (no cross-paper fit), then joined to claims through an exact maximum-weight
bipartite matching. Certificates record normalized quantities/units, entity-role and comparison
direction checks, lexical coverage, source offsets, and explicit abstentions. Repeated abstract
sentences are not allowed to serve as independent body evidence.

Only normalized numeric identity and comparison entity/baseline/direction matches are counted as
strong relation certificates. Empirical, theoretical, and qualitative lexical+artifact matches
are explicitly reported as weaker `evidence_link` witnesses: they locate relevant evidence but do
not establish semantic support. Neither tier is external scientific ground truth or an isomorphism
score against an LLM reference. No LLM, acceptance label, external anchor, or supervised tuning is
used.

The channel-matched articulability prompt is frozen at
`methods/metric_seam/science_claims_v2/articulability_prompt.json`, but its run result is
`unavailable_not_run`.

## Frozen artifact hashes

- Input `{payload['input']['path']}`: `{payload['input']['sha256']}`
{artifact_lines}
"""


def run(input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    metamorphic = metamorphic_self_check()
    records = [verify_document(r["paper_id"], r["abstract"], r["body"])
               for r in load_unlabelled(input_path)]
    source_files = [
        Path(__file__).with_name("core.py"),
        Path(__file__),
        Path(__file__).with_name("test_science_claims_v2.py"),
        Path(__file__).with_name("articulability_prompt.json"),
        ROOT / "methods/metric_seam/hybrids/programs_peer_review/cv1_supported_h0.py",
        ROOT / "methods/metric_seam/hybrids/programs_peer_review/cv2_beats_baselines_h0.py",
        ROOT / "methods/metric_seam/hybrids/programs_peer_review/cv3_has_evidence_h0.py",
        ROOT / "datasets/peer-review/build_cv_evidence.py",
    ]
    payload = {
        "schema_version": "science-claims-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "code_verifiability",
        "provenance": "retrospective_seed",
        "pipeline_status": "selected",
        "selection_note": "Existing cv1/cv2/cv3 claim-to-body decomposition retained and evolved; no claim of original automatic discovery.",
        "articulability": "not_evaluated",
        "articulability_counterpart": {
            "specification": "methods/metric_seam/science_claims_v2/articulability_prompt.json",
            "run_result": "unavailable_not_run",
            "input_representation": "identical paper_id+abstract+body allowlist",
            "certificate_semantics": "channel-matched support/contradiction/insufficient/abstain",
        },
        "isomorphic_reconstruction": "not_evaluated",
        "external_supervision": "none",
        "label_policy": {
            "forbidden_for_training_selection_tuning_and_headline_evaluation": ["y", "judgement", "acceptance"],
            "loader_allowlist": ["paper_id", "abstract", "body"],
            "implementation": "load_unlabelled constructs a new allowlisted mapping and ignores all other JSON fields",
        },
        "retrieval_fit_scope": "within_each_presented_paper_body_only",
        "metamorphic_checks": {
            "passed": sum(metamorphic.values()),
            "total": len(metamorphic),
            "checks": metamorphic,
        },
        "input": {"path": str(input_path.relative_to(ROOT)), "sha256": sha256(input_path)},
        "artifacts": {str(p.relative_to(ROOT)): sha256(p) for p in source_files if p.exists()},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "summary": _summarize(records),
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args.input.resolve(), args.output.resolve(), args.report.resolve())
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
