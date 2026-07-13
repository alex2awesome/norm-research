#!/usr/bin/env python3
"""Freeze or verify the 2026-07-12 reconstruction-v2 continuation artifacts.

This is additive to ``freeze_reconstruction_v2.py`` and its already-published
manifest.  It deliberately has a narrower, explicit scope: the fresh blind a216
run, the active code-review a104 CPU lane, and the science same-input prompt
counterpart.  Future work must write a new manifest rather than replacing this one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    / "FREEZE_MANIFEST_2026-07-12_CONTINUATION_01.json"
)

EXACT_PATHS = (
    "methods/metric_seam/freeze_reconstruction_v2_continuation.py",
    "methods/metric_seam/battery/finalize_preempted_blind_v2.py",
    "methods/metric_seam/battery/seal_ctext_items_v2.py",
    "methods/metric_seam/battery/test_finalize_preempted_blind_v2.py",
    "methods/metric_seam/battery/test_seal_ctext_items_v2.py",
    "methods/metric_seam/certificates.py",
    "methods/metric_seam/reconstruction_v2.py",
    "methods/metric_seam/RECONSTRUCTION_V2.md",
    "methods/metric_seam/environment_v2.py",
    "methods/metric_seam/battery/blind_reconstruction_v2.py",
    "methods/metric_seam/battery/_blind_worker_v2.py",
    "methods/metric_seam/battery/split_ops_v2.py",
    "methods/metric_seam/hybrids/ops.py",
    "methods/metric_seam/hybrids/ops_math.py",
    "methods/metric_seam/hybrids/ops_code.py",
    "methods/metric_seam/hybrids/test_ops_code.py",
    "methods/metric_seam/hybrids/eval_hybrids_task.py",
    "methods/metric_seam/hybrids/programs_code_review",
    # The V3 independent audit reruns these exact pre-existing comparison
    # programs.  Their hashes are also recorded in the receipts, but keeping
    # the sources in this continuation freeze makes the rerun self-contained.
    "methods/existing_metrics_runner/coded/metrics/a104_test_presence.py",
    "methods/existing_metrics_runner/coded/sandbox.py",
    "runs/validity_full/v2/code_review/codegen_claude/a104_v0_keyword.py",
    "runs/validity_full/v2/code_review/codegen_claude/a104_v1_structure.py",
    "runs/validity_full/v2/code_review/codegen_claude/a104_v2_holistic.py",
    "methods/metric_seam/pilot/build_code_review_cpu_baselines.py",
    "methods/metric_seam/pilot/build_code_review_blind_h0.py",
    "methods/metric_seam/pilot/evaluate_code_review_a104_cpu.py",
    "methods/metric_seam/pilot/correct_code_review_a104_provenance_v3.py",
    "methods/metric_seam/pilot/audit_code_review_a104_v3.py",
    "methods/metric_seam/pilot/test_code_review_a104_provenance_v3.py",
    "methods/metric_seam/science_claims_v2",
    "notes/2026-07-12__metric-seam-reconstruction-v2-progress.md",
    "notes/2026-07-10__seam-agentic-program-runbook.md",
    "outputs/metric_seam_pilot/reconstruction_v2/blind_math_a216_001",
    "outputs/metric_seam_pilot/battery/effort_ladder/contracts_v3/math__a216.json",
    "outputs/metric_seam_pilot/tasks/math/items.json",
    "outputs/metric_seam_pilot/reconstruction_v2/sealed_inputs/math_items_ctext_only_v1.json",
    "outputs/metric_seam_pilot/reconstruction_v2/sealed_inputs/math_items_ctext_only_v1.manifest.json",
    # Bind only the active a104 V3 dependency graph.  In particular, do not
    # sweep archive_pre_e2ladder, superseded V1 artifacts, raw prompt batches,
    # or unrelated future coding criteria into this additive freeze.
    "outputs/metric_seam_pilot/tasks/code_review/items.json",
    "outputs/metric_seam_pilot/tasks/code_review/results.jsonl",
    "outputs/metric_seam_pilot/tasks/code_review/aspects_used.json",
    "outputs/metric_seam_pilot/tasks/code_review/code_scores.json",
    "outputs/metric_seam_pilot/tasks/code_review/code_scores_cpu_manifest.json",
    "outputs/metric_seam_pilot/tasks/code_review/blind_h0_cpu_v2",
    "outputs/metric_seam_pilot/tasks/code_review/a104_cpu_sealed_eval_v2.json",
    "outputs/metric_seam_pilot/tasks/code_review/A104_CPU_SEALED_REPORT_V2.md",
    "outputs/metric_seam_pilot/tasks/code_review/A104_CPU_SEALED_REPORT_V2_CORRECTION_NOTICE.md",
    "outputs/metric_seam_pilot/tasks/code_review/a104_cpu_sealed_eval_v3.json",
    "outputs/metric_seam_pilot/tasks/code_review/A104_CPU_SEALED_REPORT_V3.md",
    "outputs/metric_seam_pilot/tasks/code_review/a104_cpu_v3_independent_audit_v1.json",
    "outputs/metric_seam_pilot/tasks/code_review/A104_CPU_V3_INDEPENDENT_AUDIT_V1.md",
    "outputs/metric_seam_pilot/tasks/code_review/a104_repo_grouped_sensitivity_v1.json",
    # Resolve the science trial family explicitly.  A broad v* glob would
    # make this old freeze fail merely because a future additive v7 is added.
    "outputs/metric_seam_pilot/science_articulability_v1_prepared",
    "outputs/metric_seam_pilot/science_articulability_v2_prepared",
    "outputs/metric_seam_pilot/science_articulability_v3_prepared",
    "outputs/metric_seam_pilot/science_articulability_v4_json_prepared",
    "outputs/metric_seam_pilot/science_articulability_v5_openrouter_prepared",
    "outputs/metric_seam_pilot/science_articulability_v6_openrouter_reasoning_off_prepared",
)

GLOB_PATHS: tuple[str, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_under(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from (
        child
        for child in path.rglob("*")
        if child.is_file() and "__pycache__" not in child.parts
    )


def selected_files(output: Path) -> list[Path]:
    selected: set[Path] = set()
    for relative in EXACT_PATHS:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(relative)
        selected.update(_files_under(path))
    for pattern in GLOB_PATHS:
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        for path in matches:
            selected.update(_files_under(path))
    selected.discard(output.resolve())
    return sorted(selected, key=lambda path: str(path.relative_to(ROOT)))


def build_manifest(output: Path) -> dict:
    files = selected_files(output.resolve())
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "schema_version": "metric-seam-reconstruction-v2-continuation-freeze-v1",
        "date": "2026-07-12",
        "git_parent_at_freeze": head,
        "scope": (
            "fresh blind Math a216 construct-preempted run; active code-review a104 CPU "
            "reconstruction and provenance correction; science full-paper same-input "
            "prompt-articulability instrument trials"
        ),
        "scope_paths": {"exact": list(EXACT_PATHS), "globs": list(GLOB_PATHS)},
        "policy": {
            "historical_artifacts_overwritten": False,
            "external_supervised_anchor_added": False,
            "unsupervised_reconstruction_objective_unchanged": True,
            "gpu_used_by_scoped_runs": False,
            "ignored_outputs_bound_by_full_sha256": True,
            "future_changes_require_new_manifest": True,
        },
        "n_files": len(files),
        "files": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }


def verify_manifest(output: Path) -> None:
    frozen = json.loads(output.read_text(encoding="utf-8"))
    current = build_manifest(output)
    invariant_keys = ("schema_version", "date", "scope", "scope_paths", "policy", "n_files")
    drifted = [key for key in invariant_keys if frozen.get(key) != current.get(key)]
    if drifted:
        raise SystemExit(f"freeze verification failed: manifest metadata drifted={drifted}")
    if frozen["files"] != current["files"]:
        frozen_paths, current_paths = set(frozen["files"]), set(current["files"])
        changed = sorted(
            path
            for path in frozen_paths & current_paths
            if frozen["files"][path] != current["files"][path]
        )
        raise SystemExit(
            "freeze verification failed: "
            f"added={sorted(current_paths - frozen_paths)}, "
            f"removed={sorted(frozen_paths - current_paths)}, changed={changed}"
        )
    print(f"PASS: {len(current['files'])} continuation files match full SHA-256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.out.resolve()
    if args.verify:
        verify_manifest(output)
        return 0
    if output.exists():
        raise SystemExit(
            f"refusing to overwrite frozen manifest {output}; use --verify or a new path"
        )
    manifest = build_manifest(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} ({manifest['n_files']} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
