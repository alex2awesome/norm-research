#!/usr/bin/env python3
"""Additive exact-address science comparator with strict relation fidelity.

This applies :mod:`core_relation_strict` to the same A####/B#### source maps used by
the v8 prompt arm.  It intentionally leaves the historical v8 output untouched.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import addressed_code_comparator_v8 as v8
from . import core_relation_strict as strict


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = (
    ROOT / "outputs/metric_seam_pilot/science_articulability_v8_hardened_prepared"
)
DEFAULT_CONTINUOUS = (
    ROOT / "outputs/metric_seam_pilot/science_claims_v2_relation_strict_v23/results.json"
)
DEFAULT_OUT = (
    ROOT / "outputs/metric_seam_pilot/science_verifiability_v9_relation_strict_addressed"
)


@contextmanager
def _addressed_strict_bindings() -> Iterator[None]:
    old_quantities = v8.corrected.extract_quantities
    old_comparison = v8.extract_comparison
    old_edge = v8._evaluate_edge
    v8.corrected.extract_quantities = strict.extract_quantities
    v8.extract_comparison = strict.extract_comparison
    v8._evaluate_edge = strict.evaluate_edge
    try:
        yield
    finally:
        v8.corrected.extract_quantities = old_quantities
        v8.extract_comparison = old_comparison
        v8._evaluate_edge = old_edge


def verify_addressed_document(paper_id: str, source_map: dict[str, Any]) -> dict[str, Any]:
    with _addressed_strict_bindings():
        return v8.verify_addressed_document(paper_id, source_map)


def _normal_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _hydrate(
    certificate: dict[str, Any],
    source_map: dict[str, Any],
    side: str,
) -> str:
    address = certificate[side]["source_address"]
    span = source_map[address["section"]][address["sentence_index"]]
    if (
        span["sentence_id"] != address["sentence_id"]
        or span["text_sha256"] != address["text_sha256"]
    ):
        raise ValueError("certificate address no longer resolves to its exact source span")
    return span["text"]


def _continuous_key(
    paper_id: str,
    certificate: dict[str, Any],
    *,
    normalize: bool,
) -> tuple[str, ...]:
    transform = _normal_text if normalize else lambda value: value
    return (
        paper_id,
        transform(certificate["claim"]["text"]),
        transform(certificate["evidence"]["text"]),
        certificate["claim"]["relation"],
        certificate["decision"],
    )


def _addressed_key(
    paper_id: str,
    certificate: dict[str, Any],
    source_map: dict[str, Any],
    *,
    normalize: bool,
) -> tuple[str, ...]:
    transform = _normal_text if normalize else lambda value: value
    return (
        paper_id,
        transform(_hydrate(certificate, source_map, "claim")),
        transform(_hydrate(certificate, source_map, "evidence")),
        certificate["claim"]["relation"],
        certificate["decision"],
    )


def _identity_counts(left: Counter, right: Counter) -> dict[str, int]:
    return {
        "continuous": sum(left.values()),
        "addressed": sum(right.values()),
        "intersection": sum((left & right).values()),
        "continuous_only": sum((left - right).values()),
        "addressed_only": sum((right - left).values()),
    }


def representation_comparison(
    continuous_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    source_maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    continuous_records = continuous_payload["records"]
    addressed_records = {row["paper_id"]: row["result"] for row in rows}
    continuous_by_id = {row["paper_id"]: row for row in continuous_records}
    if set(continuous_by_id) != set(addressed_records):
        raise ValueError("continuous and addressed arms cover different paper IDs")

    def continuous_counter(kind: str, normalize: bool) -> Counter:
        return Counter(
            _continuous_key(record["paper_id"], certificate, normalize=normalize)
            for record in continuous_records
            for certificate in record.get(kind, [])
        )

    def addressed_counter(kind: str, normalize: bool) -> Counter:
        return Counter(
            _addressed_key(
                row["paper_id"],
                certificate,
                source_maps[row["paper_id"]],
                normalize=normalize,
            )
            for row in rows
            for certificate in row["result"].get(kind, [])
        )

    strong_exact = _identity_counts(
        continuous_counter("certificates", False),
        addressed_counter("certificates", False),
    )
    strong_normalized = _identity_counts(
        continuous_counter("certificates", True),
        addressed_counter("certificates", True),
    )
    weak_normalized = _identity_counts(
        continuous_counter("evidence_links", True),
        addressed_counter("evidence_links", True),
    )
    status_deltas = Counter(
        f"{continuous_by_id[paper_id]['status']}->{addressed_records[paper_id]['status']}"
        for paper_id in continuous_by_id
        if continuous_by_id[paper_id]["status"] != addressed_records[paper_id]["status"]
    )
    supported_continuous = {
        paper_id
        for paper_id, result in continuous_by_id.items()
        if result["status"] == "supported"
    }
    supported_addressed = {
        paper_id
        for paper_id, result in addressed_records.items()
        if result["status"] == "supported"
    }
    return {
        "comparison_key": (
            "paper_id + claim text + evidence text + relation + decision"
        ),
        "strong_exact_text": strong_exact,
        "strong_whitespace_normalized_text": strong_normalized,
        "weak_whitespace_normalized_text": weak_normalized,
        "paper_status_agreement": len(continuous_by_id) - sum(status_deltas.values()),
        "paper_status_total": len(continuous_by_id),
        "paper_status_deltas": dict(sorted(status_deltas.items())),
        "supported_paper_sets": {
            "continuous": len(supported_continuous),
            "addressed": len(supported_addressed),
            "intersection": len(supported_continuous & supported_addressed),
            "continuous_only": len(supported_continuous - supported_addressed),
            "addressed_only": len(supported_addressed - supported_continuous),
        },
        "interpretation": (
            "Whitespace-normalized identity measures representation robustness of the "
            "same strict executable relation program. Exact-text identity is separately "
            "reported because addressed spans preserve source line breaks while the "
            "continuous segmenter normalizes whitespace."
        ),
    }


def run(
    bundle: Path,
    continuous_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    checks = strict.metamorphic_self_check()
    _, requests, abstentions, _ = v8._verified_bundle_snapshot(bundle)
    source_maps = {request["paper_id"]: request["source_map"] for request in requests}
    rows = [
        {
            "schema_version": "science-verifiability-addressed-result-v9",
            "source_index": request["source_index"],
            "paper_id": request["paper_id"],
            "result": verify_addressed_document(
                request["paper_id"], request["source_map"]
            ),
        }
        for request in requests
    ]
    for abstention in abstentions:
        legacy = v8._structural_result(abstention, bundle_snapshot={
            "manifest_sha256": "not_repeated_in_minimal_v9_result",
            "requests_sha256": "not_repeated_in_minimal_v9_result",
            "implementation_bindings": {
                "exact_address_comparator": {"sha256": "v9_strict"}
            },
        })
        rows.append(
            {
                "schema_version": "science-verifiability-addressed-result-v9",
                "source_index": abstention["source_index"],
                "paper_id": abstention["paper_id"],
                "result": legacy["result"],
            }
        )
    rows.sort(key=lambda row: row["source_index"])
    if [row["source_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("addressed strict output does not cover each corpus row once")

    continuous_payload = json.loads(continuous_path.read_text(encoding="utf-8"))
    comparison = representation_comparison(continuous_payload, rows, source_maps)
    summary = v8.summarize(rows)
    payload = {
        "schema_version": "science-verifiability-addressed-v9-relation-strict",
        "status": "completed_cpu_no_api_no_gpu",
        "objective": "unsupervised_code_reconstruction_relation_local",
        "external_supervision": "none",
        "method_origin": "manually_constructed_retrospective_seed",
        "certificate_scope": (
            "document_local_parser_witness_not_external_scientific_truth"
        ),
        "corrections": continuous_payload["corrections"],
        "metamorphic_checks": checks,
        "summary": summary,
        "representation_comparison": comparison,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "code_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    comparison = payload["representation_comparison"]
    normalized = comparison["strong_whitespace_normalized_text"]
    exact = comparison["strong_exact_text"]
    weak = comparison["weak_whitespace_normalized_text"]
    return f"""# Science exact-address relation comparator v9

Status: **completed on CPU**, with no model/API calls and no GPU use.

This additive arm runs the same v2.3 strict relation program over the prompt arm's exact A/B
source addresses. It leaves the v2.2 and v8 artifacts unchanged.

## Corrected addressed result

- Parser-accepted strong relation witnesses: **{summary['certificates']}**
- Numeric: **{summary['certificate_relation_counts'].get('numeric', 0)}**;
  comparative: **{summary['certificate_relation_counts'].get('comparative', 0)}**
- Weak evidence links (separate tier): **{summary['evidence_links']}**
- Statuses: `{json.dumps(summary['status_counts'], sort_keys=True)}`

## Continuous ↔ addressed representation test

- Strong identities after whitespace normalization: **{normalized['intersection']} /
  {normalized['continuous']} continuous** and **{normalized['intersection']} /
  {normalized['addressed']} addressed**
- Strong strict-text identities: **{exact['intersection']}**; continuous-only:
  **{exact['continuous_only']}**; addressed-only: **{exact['addressed_only']}**
- Supported-paper sets: `{json.dumps(comparison['supported_paper_sets'], sort_keys=True)}`
- Paper statuses agree: **{comparison['paper_status_agreement']} /
  {comparison['paper_status_total']}**; deltas:
  `{json.dumps(comparison['paper_status_deltas'], sort_keys=True)}`
- Weak-link normalized identities: intersection **{weak['intersection']}**,
  continuous-only **{weak['continuous_only']}**, addressed-only **{weak['addressed_only']}**

Strict text is not expected to match: the continuous segmenter replaces source line breaks with
spaces, whereas exact-address spans retain them. The whitespace-normalized comparison is the
licensed representation test.

The earlier 136/136 finding remains valid as **v2.2/v8 parser-output invariance**, but the audit
found at least five concrete relation-fidelity counterexamples in that parser class. The present result
tests representation robustness after those defects are corrected. It is still a relation-local
executable witness, not external scientific truth, full semantic isomorphism, or automatic
discovery of the decomposition.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--continuous", type=Path, default=DEFAULT_CONTINUOUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = run(
        args.bundle.resolve(), args.continuous.resolve(), args.out.resolve()
    )
    print(
        json.dumps(
            {
                "certificates": payload["summary"]["certificates"],
                "relations": payload["summary"]["certificate_relation_counts"],
                "representation": payload["representation_comparison"][
                    "strong_whitespace_normalized_text"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
