#!/usr/bin/env python3
"""Score raw vLLM alignments and compare them with frozen Sonnet alignments."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import random
import statistics

from .response_normalization import normalize_alignment_raw
from .semantic_alignment import AlignmentError, build_alignment_report


def links(report: dict) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for cluster in report["semantic_clusters"]:
        ids = sorted(str(unit["unit_id"]) for unit in cluster["units"])
        values.update(combinations(ids, 2))
    return values


def jaccard(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def median_summary(values: list[float], *, seed: int, draws: int = 10_000) -> dict:
    """Metric-bootstrap interval for the median; metrics are the sampling unit."""

    if not values:
        return {"n_metrics": 0, "median": None, "metric_bootstrap_ci95": None}
    rng = random.Random(seed)
    boot = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)
    )
    return {
        "n_metrics": len(values),
        "median": statistics.median(values),
        "metric_bootstrap_ci95": [boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]],
        "bootstrap_draws": draws,
        "bootstrap_unit": "metric",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--sonnet-dir", type=Path, required=True)
    parser.add_argument("--study-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.raw_dir / "manifest.json").read_text())
    study = json.loads(args.study_readout.read_text())
    domain_by_name = {str(row["construct"]): str(row["domain"]) for row in study["records"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for request_path in sorted(args.requests_dir.glob("*_request.json")):
        stem = request_path.name.removesuffix("_request.json")
        raw_path = args.raw_dir / f"{stem}_raw.txt"
        sonnet_path = args.sonnet_dir / f"{stem}_response.json"
        try:
            request = json.loads(request_path.read_text())
            raw_text = raw_path.read_text(encoding="utf-8")
            normalized_text = normalize_alignment_raw(raw_text)
            was_normalized = normalized_text != raw_text
            report = build_alignment_report(
                request=request, raw_response=normalized_text
            )
            report["source_request_model"] = report["model"]
            report["model"] = manifest["model"]
            report["instrument"] = "independent_open_model_replication"
            report["response_was_normalized"] = was_normalized
            sonnet = json.loads(sonnet_path.read_text())
            pairwise_mean = statistics.mean(
                value["semantic_jaccard"]["decimal"] for value in report["pairwise"]
            )
            link_j = jaccard(links(report), links(sonnet))
            output_path = args.output_dir / f"{stem}_response.json"
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            rows.append({
                "metric_file_stem": stem,
                "metric_name": report["metric"]["name"],
                "domain": domain_by_name[report["metric"]["name"]],
                "open_model_mean_pairwise_semantic_jaccard": pairwise_mean,
                "sonnet_open_model_link_jaccard": link_j,
                "response_was_normalized": was_normalized,
                "output": str(output_path),
            })
        except (AlignmentError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            failures.append({"metric_file_stem": stem, "error": f"{type(exc).__name__}: {exc}"})
    link_values = [row["sonnet_open_model_link_jaccard"] for row in rows]
    open_values = [row["open_model_mean_pairwise_semantic_jaccard"] for row in rows]
    by_domain = {}
    for index, domain in enumerate(sorted({row["domain"] for row in rows})):
        domain_rows = [row for row in rows if row["domain"] == domain]
        by_domain[domain] = {
            "sonnet_open_model_link_jaccard": median_summary(
                [row["sonnet_open_model_link_jaccard"] for row in domain_rows],
                seed=99173 + index,
            ),
            "open_model_decomposition_stability": median_summary(
                [row["open_model_mean_pairwise_semantic_jaccard"] for row in domain_rows],
                seed=88163 + index,
            ),
        }
    result = {
        "schema": "metric-seam.semantic-alignment-cross-instrument.v1",
        "status": "complete" if not failures else "partial",
        "model": manifest["model"],
        "n_expected": len(list(args.requests_dir.glob("*_request.json"))),
        "n_valid": len(rows),
        "n_invalid": len(failures),
        "n_normalized": sum(1 for row in rows if row["response_was_normalized"]),
        "overall": {
            "sonnet_open_model_link_jaccard": median_summary(link_values, seed=77171),
            "open_model_decomposition_stability": median_summary(open_values, seed=66161),
        },
        "by_domain": by_domain,
        "rows": rows,
        "failures": failures,
        "claim_limits": [
            "Cross-instrument agreement is unsupervised and does not establish construct truth.",
            "Both instruments see the same frozen metric text and relation units.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "n_valid": result["n_valid"],
        "n_invalid": result["n_invalid"],
        "n_normalized": result["n_normalized"],
        "link_agreement": result["overall"]["sonnet_open_model_link_jaccard"],
    }))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
