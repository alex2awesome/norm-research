#!/usr/bin/env python3
"""Bind and byte-replay the current exact-address Science code arm.

The v9 artifact predates a durable receipt for its input corpus, prompt-side
address representation, implementation modules, and rendered outputs.  This
additive audit re-executes the CPU-only code arm in a temporary directory and
records whether every archived byte is reproduced.  It never executes the
prepared prompt jobs or loads an external target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from . import addressed_code_comparator_v9 as v9


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = (
    ROOT / "outputs/metric_seam_pilot/science_articulability_v8_hardened_prepared"
)
DEFAULT_CONTINUOUS = (
    ROOT / "outputs/metric_seam_pilot/science_claims_v2_relation_strict_v23/results.json"
)
DEFAULT_ARCHIVED = (
    ROOT / "outputs/metric_seam_pilot/science_verifiability_v9_relation_strict_addressed"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/metric_seam_pilot/science_verifiability_v9_relation_strict_addressed_replay_v1.json"
)

OUTPUT_FILES = ("manifest.json", "code_results.jsonl", "REPORT.md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit(
    *,
    bundle: Path = DEFAULT_BUNDLE,
    continuous: Path = DEFAULT_CONTINUOUS,
    archived: Path = DEFAULT_ARCHIVED,
) -> dict[str, Any]:
    prompt_manifest = _load_json(bundle / "manifest.json")
    archived_manifest = _load_json(archived / "manifest.json")
    if (
        prompt_manifest.get("schema_version")
        != "science-articulability-addressed-bundle-v8"
        or prompt_manifest.get("status") != "prepared_not_run_no_api_calls"
        or prompt_manifest.get("files", {}).get("requests", {}).get("count") != 1957
        or prompt_manifest.get("files", {})
        .get("structural_abstentions", {})
        .get("count")
        != 443
        or prompt_manifest.get("execution_policy", {}).get("api_calls_made_by_prepare")
        != 0
        or prompt_manifest.get("execution_policy", {}).get("gpu_used") is not False
    ):
        raise ValueError("current Science prompt bundle drifted")
    if (
        archived_manifest.get("schema_version")
        != "science-verifiability-addressed-v9-relation-strict"
        or archived_manifest.get("status") != "completed_cpu_no_api_no_gpu"
        or archived_manifest.get("summary", {}).get("certificates") != 100
        or archived_manifest.get("representation_comparison", {})
        .get("strong_whitespace_normalized_text", {})
        .get("intersection")
        != 100
    ):
        raise ValueError("archived Science v9 artifact drifted")

    with tempfile.TemporaryDirectory(prefix="metric-seam-science-v9-replay-") as temp:
        replay = Path(temp) / "replay"
        v9.run(bundle, continuous, replay)
        output_comparison = {}
        for name in OUTPUT_FILES:
            archived_path = archived / name
            replay_path = replay / name
            output_comparison[name] = {
                "archived_sha256": _sha256(archived_path),
                "replay_sha256": _sha256(replay_path),
                "byte_exact": archived_path.read_bytes() == replay_path.read_bytes(),
            }

    byte_exact = all(row["byte_exact"] for row in output_comparison.values())
    if not byte_exact:
        raise ValueError("Science v9 replay diverged from the archived output")

    prompt_files = {
        name: _bound_file(bundle / prompt_manifest["files"][name]["path"])
        for name in ("requests", "source_crosswalk", "structural_abstentions")
    }
    return {
        "schema_version": "metric-seam.science-addressed-v9-replay-freeze.v1",
        "status": "byte_exact_cpu_replay_complete",
        "objective": "unsupervised_relation_local_prompt_code_reconstruction_scaffold",
        "method_origin": "manually_constructed_retrospective_pipeline_seed",
        "bound_inputs": {
            "prompt_manifest": _bound_file(bundle / "manifest.json"),
            "prompt_files": prompt_files,
            "continuous_code_input": _bound_file(continuous),
            "addressed_comparator_module": _bound_file(Path(v9.__file__).resolve()),
            "strict_relation_module": _bound_file(
                ROOT / "methods/metric_seam/science_claims_v2/core_relation_strict.py"
            ),
            "addressed_base_comparator_module": _bound_file(
                ROOT
                / "methods/metric_seam/science_claims_v2/addressed_code_comparator_v8.py"
            ),
            "address_builder_module": _bound_file(
                ROOT / "methods/metric_seam/science_claims_v2/addressed_pipeline.py"
            ),
        },
        "archived_outputs": {
            name: _bound_file(archived / name) for name in OUTPUT_FILES
        },
        "replay": {
            "output_comparison": output_comparison,
            "byte_exact_all_outputs": byte_exact,
            "records": int(archived_manifest["summary"]["records"]),
            "strong_relation_witnesses": int(
                archived_manifest["summary"]["certificates"]
            ),
            "strong_whitespace_normalized_witnesses_shared": int(
                archived_manifest["representation_comparison"]
                ["strong_whitespace_normalized_text"]["intersection"]
            ),
            "supported_papers_shared": int(
                archived_manifest["representation_comparison"]
                ["supported_paper_sets"]["intersection"]
            ),
        },
        "prompt_plane": {
            "compiled_unscored_jobs": 1957,
            "structural_abstentions_without_remote_call": 443,
            "prompt_responses_in_current_v8_bundle": 0,
            "prompt_articulability_measured": False,
            "semantic_prompt_code_comparison_measured": False,
        },
        "channel_contract": {
            "same_source_address_text_available_to_prompt_and_code": True,
            "prompt_jobs_executed_by_this_audit": False,
            "reference_or_outcome_values_loaded": False,
            "external_supervision_used": False,
            "models_or_apis_called": False,
            "accelerators_used": False,
            "external_scientific_truth_estimated": False,
            "whole_review_score_emitted": False,
        },
        "temporal_disposition": {
            "current_bundle_role": "instrument_development_exploratory_unscored",
            "fresh_split_required_for_confirmatory_prompt_code_claim": True,
            "reason": (
                "The current prompt contract follows earlier transport smoke tests and "
                "the fixed code decomposition; it can support an exploratory comparison "
                "after execution, not a temporally blind confirmatory claim."
            ),
        },
        "claim_boundary": (
            "The exact-address code representation and archived code results are now "
            "bound and byte-replayable. This establishes a same-source-address scaffold, "
            "not prompt articulability, prompt/code semantic isomorphism, whole-criterion "
            "codability, or external scientific truth."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--continuous", type=Path, default=DEFAULT_CONTINUOUS)
    parser.add_argument("--archived", type=Path, default=DEFAULT_ARCHIVED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = audit(
        bundle=args.bundle,
        continuous=args.continuous,
        archived=args.archived,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "records": payload["replay"]["records"],
        "strong_relation_witnesses": payload["replay"][
            "strong_relation_witnesses"
        ],
        "compiled_unscored_prompt_jobs": payload["prompt_plane"]
        ["compiled_unscored_jobs"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
