#!/usr/bin/env python3
"""Resolve the technical replay manifest and emit JSON plus a compact Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_MANIFEST = HERE / "initial_manifest.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "metric_seam_pilot" / "technical_replay_v2"

try:
    from .core import OBJECTIVES, evaluate_manifest
    from ..environment_v2 import environment_fingerprint
except ImportError:  # direct ``python evaluate.py`` execution
    import sys

    sys.path.insert(0, str(HERE.parent))
    from core import OBJECTIVES, evaluate_manifest
    from environment_v2 import environment_fingerprint


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_report(result: dict[str, Any], manifest_sha256: str) -> str:
    lines = [
        "# Technical-domain retrospective replay v2",
        "",
        f"Manifest SHA-256: `{manifest_sha256}`",
        "",
        "This is an unsupervised reconstruction audit over existing artifacts. It does not use an "
        "external anchor, and it does not relabel manual, mock, oracle-conditioned, or retrospective "
        "decompositions as automatically discovered. Those artifacts may nevertheless be selected "
        "pipeline seeds whose utility, depth, and certificate yield are measured with provenance intact.",
        "",
        "## Objective separation",
        "",
    ]
    for objective in OBJECTIVES:
        lines.append(f"- **{objective.replace('_', ' ')}:** {result['objective_definitions'][objective]}")
    lines.extend(
        [
            "",
            "## Selected-pipeline utility and relation depth",
            "",
            "| Domain / case | Pipeline role | Selection | Origin | Depth | Utility |",
            "|---|---|---|---|---|---|",
        ]
    )
    for case in result["cases"]:
        utility = case["utility"]
        metrics = ", ".join(f"{k}={_fmt(v)}" for k, v in utility["measurements"].items())
        utility_cell = f"{utility['assessment']}: {metrics}" if metrics else utility["assessment"]
        depth = case["relation_depth"]
        lineage = case.get("program_lineage", "")
        label = f"{case['domain']} / `{case['case_id']}`"
        if lineage == "legacy_prototype_not_active_coding_census":
            label += " (legacy prototype; active coding census pending)"
        lines.append(
            "| " + " | ".join(
                [
                    label,
                    case["pipeline_status"],
                    case["selection_mode"],
                    case["discovery_mode"],
                    f"{depth['level']} ({depth['label']})",
                    utility_cell,
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Case readout",
            "",
            "| Domain / case | Provenance | Corpus | Prompt articulability | Code verifiability | LLM-reference reconstruction | Constructive extension |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for case in result["cases"]:
        axes = case["axes"]
        corpus = "eligible" if case["corpus_eligibility"]["eligible"] else "ineligible"
        cells = []
        for objective in OBJECTIVES:
            axis = axes[objective]
            metrics = ", ".join(f"{k}={_fmt(v)}" for k, v in axis["measurements"].items())
            cells.append(f"{axis['assessment']}: {metrics}" if metrics else axis["assessment"])
        lineage = case.get("program_lineage", "")
        label = f"{case['domain']} / `{case['case_id']}`"
        if lineage == "legacy_prototype_not_active_coding_census":
            label += " (legacy prototype; active coding census pending)"
        lines.append(
            "| " + " | ".join(
                [
                    label,
                    case["discovery_mode"],
                    corpus,
                    *cells,
                ]
            ) + " |"
        )
    lines.extend(["", "## Bounded conclusions", ""])
    for case in result["cases"]:
        lines.append(f"### {case['case_id']}")
        lines.append("")
        depth = case["relation_depth"]
        lines.append(
            f"- selected pipeline: `{case['pipeline_status']}` via `{case['selection_mode']}`; "
            f"historical origin `{case['discovery_mode']}`."
        )
        lines.append(
            f"- relation depth {depth['level']} ({depth['label']}): {depth['mechanism']}"
        )
        lines.append(
            f"- selected-pipeline utility ({case['utility']['assessment']}): "
            f"{case['utility']['claim']}"
        )
        for objective in OBJECTIVES:
            axis = case["axes"][objective]
            lines.append(f"- {objective.replace('_', ' ')} ({axis['assessment']}): {axis['claim']}")
        if case["corpus_eligibility"]["limitations"]:
            lines.append(
                "- Corpus limits: " + "; ".join(case["corpus_eligibility"]["limitations"])
            )
        lines.append("")
    summary = result["summary"]
    lines.extend(
        [
            "## Claim hygiene",
            "",
            f"- Selected pipeline cases: {summary['n_selected_pipeline_cases']}.",
            f"- Selected-pipeline utility claims permitted: {summary['n_selected_utility_claims_permitted']}.",
            f"- Historical automatic-selection claims permitted: {summary['n_automatic_decomposition_claims_permitted']}.",
            f"- Confirmatory isomorphic-reconstruction claims permitted: {summary['n_confirmatory_isomorphic_claims_permitted']}.",
            f"- Provenance-conditioned selected-pipeline evidence extensions: {summary['n_provenance_conditioned_extension_claims_permitted']}.",
            f"- Unconditioned selected-pipeline evidence extensions: {summary['n_unconditioned_extension_claims_permitted']}.",
            f"- Canonical code-verifiability claims permitted: {summary['n_canonical_code_verifiability_claims_permitted']}.",
            f"- Canonical verifier-dominant constructive-extension claims permitted: {summary['n_canonical_constructive_extension_claims_permitted']}.",
            f"- Tacitness claims permitted: {summary['n_tacitness_claims_permitted']}.",
            "- Retrospective selection is an experimental pipeline decision, not a claim about how the historical "
            "artifact was created. Manual/mock/oracle provenance qualifies interpretation but does not erase measured utility.",
            "- A later blind agentic run is still required for automatic-selection or confirmatory reconstruction claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="evaluate without writing outputs")
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    result = evaluate_manifest(manifest, REPO_ROOT)
    result["environment"] = environment_fingerprint()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    result["manifest_sha256"] = manifest_sha

    if args.check:
        print(json.dumps(result["summary"], indent=2, allow_nan=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.snapshot.json").write_bytes(manifest_bytes)
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.out_dir / "REPORT.md").write_text(render_report(result, manifest_sha))
    print(f"wrote {args.out_dir / 'results.json'}")
    print(f"wrote {args.out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
