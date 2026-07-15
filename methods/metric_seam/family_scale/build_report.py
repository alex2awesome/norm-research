#!/usr/bin/env python3
"""Build the first complete family-scale structural reconstruction report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Sequence


SCHEMA = "metric-seam.family-scale-structural-readout.v1"


def _q(values: list[float], p: float) -> float:
    xs = sorted(values)
    x = (len(xs) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (hi - x) + xs[hi] * (x - lo)


def _summary(values: list[float], *, seed: int, draws: int = 10000) -> dict[str, object]:
    if not values:
        raise ValueError("cannot summarize empty values")
    rng = random.Random(seed)
    boots = [statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)]
    return {
        "n_metrics": len(values),
        "median_metric_mean_pairwise_semantic_jaccard": statistics.median(values),
        "iqr": [_q(values, 0.25), _q(values, 0.75)],
        "metric_bootstrap_ci95_for_median": [_q(boots, 0.025), _q(boots, 0.975)],
        "bootstrap_draws": draws,
        "bootstrap_unit": "metric",
    }


def build(study: dict, stability_dir: Path, semantic_dir: Path) -> dict[str, object]:
    cells = study["cells"]
    if len(cells) != 60:
        raise ValueError("first structural pilot requires exactly 60 frozen metrics")
    by_filename = {
        "metric_" + hashlib.sha256(row["metric_id"].encode()).hexdigest()[:20] + ".json": row
        for row in cells
    }
    records = []
    relation_counts = Counter()
    for filename, cell in sorted(by_filename.items()):
        stability_path = stability_dir / filename
        semantic_path = semantic_dir / filename.replace(".json", "_response.json")
        if not stability_path.exists() or not semantic_path.exists():
            raise ValueError(f"missing result for {cell['metric_id']}")
        lexical = json.loads(stability_path.read_text())
        semantic = json.loads(semantic_path.read_text())
        if semantic.get("request_sha256") is None or semantic.get("alignment_call_count") != 1:
            raise ValueError("invalid semantic result")
        pairwise = semantic["pairwise"]
        semantic_js = [row["semantic_jaccard"]["decimal"] for row in pairwise]
        lexical_js = [row["lexical_exact_lower_bound"]["jaccard"]["decimal"] for row in pairwise]
        counts = [row["relation_count"] for row in lexical["fleets"]]
        relation_counts.update(counts)
        records.append({
            "metric_id": cell["metric_id"],
            "domain": cell["domain"],
            "level": cell["level"],
            "construct": cell["metric_text"]["construct"],
            "fleet_relation_counts": counts,
            "mean_pairwise_lexical_jaccard": statistics.mean(lexical_js),
            "mean_pairwise_semantic_jaccard": statistics.mean(semantic_js),
            "pairwise_semantic_jaccard": semantic_js,
            "semantic_cluster_count": len(semantic["semantic_clusters"]),
            "unmatched_unit_count": len(semantic["unmatched_units"]),
        })
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_level: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_domain[row["domain"]].append(row["mean_pairwise_semantic_jaccard"])
        by_level[row["level"]].append(row["mean_pairwise_semantic_jaccard"])
    return {
        "schema": SCHEMA,
        "status": "complete_60_metric_structural_reconstruction",
        "study_content_sha256": study["study_content_sha256"],
        "calls": {
            "blind_decomposition": 180,
            "semantic_alignment": 60,
            "decomposition_recovery_calls": 1,
            "alignment_recovery_calls": 1,
        },
        "coverage": {"metrics_expected": 60, "metrics_complete": len(records), "domains": 4},
        "overall": _summary([row["mean_pairwise_semantic_jaccard"] for row in records], seed=20260714),
        "by_domain": {
            key: _summary(values, seed=20260714 + i)
            for i, (key, values) in enumerate(sorted(by_domain.items()))
        },
        "by_hierarchy_round_descriptive": {
            key: _summary(values, seed=20260814 + i)
            for i, (key, values) in enumerate(sorted(by_level.items()))
        },
        "instrument_diagnostics": {
            "metrics_with_any_exact_lexical_match": sum(row["mean_pairwise_lexical_jaccard"] > 0 for row in records),
            "fleet_decompositions_at_five_relation_cap": relation_counts[5],
            "fleet_decompositions_total": sum(relation_counts.values()),
            "cap_saturation_rate": relation_counts[5] / sum(relation_counts.values()),
            "reading": "semantic alignment is primary; exact lexical matching is an observed-zero lower bound; width/capture-recapture is censored by the five-relation cap",
        },
        "claim_limits": [
            "This is prompt-side articulation/decomposition stability, not code-based verifiability.",
            "One Sonnet alignment pass per metric is an unsupervised semantic matching instrument, not external ground truth.",
            "The symmetric metric budget does not imply equal population prevalence or equal domain yield.",
            "No corpus support, family certificate, operational code witness, behavioral isomorphism, or tacitness claim is established here.",
            "R1/R2/R3 are operational generation rounds and not a certified ancestry partition; round summaries are descriptive only.",
        ],
        "next_stage": "base-rate probe recurring semantic relation families before authoring/importing code verifiers",
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--stability-dir", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(json.loads(args.study.read_text()), args.stability_dir, args.semantic_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Family-scale structural reconstruction — first 60-metric result", "",
        f"Complete: **{result['coverage']['metrics_complete']}/{result['coverage']['metrics_expected']} metrics**.", "",
        f"Overall median metric-level mean semantic Jaccard: **{result['overall']['median_metric_mean_pairwise_semantic_jaccard']:.3f}** "
        f"(metric-bootstrap 95% CI {result['overall']['metric_bootstrap_ci95_for_median']}).", "",
        "## By technical domain", "",
    ]
    for domain, row in result["by_domain"].items():
        lines.append(f"- {domain}: median **{row['median_metric_mean_pairwise_semantic_jaccard']:.3f}**, CI {row['metric_bootstrap_ci95_for_median']} (n={row['n_metrics']})")
    diag = result["instrument_diagnostics"]
    lines += ["", "## Instrument reading", "",
              f"Exact lexical overlap occurred in {diag['metrics_with_any_exact_lexical_match']}/60 metrics. "
              f"Semantic alignment is therefore the primary structural readout. {diag['fleet_decompositions_at_five_relation_cap']}/{diag['fleet_decompositions_total']} fleet outputs hit the five-relation cap, so width is censored and capture–recapture is not interpreted.", "",
              "This result measures prompt-side articulation stability only. Code-family base-rate probes, certificates, execution, and behavioral reconstruction are the next stages."]
    args.report.write_text("\n".join(lines) + "\n")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
