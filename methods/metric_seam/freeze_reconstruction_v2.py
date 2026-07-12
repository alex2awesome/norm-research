#!/usr/bin/env python3
"""Write or verify the scoped reconstruction-v2 evidence freeze manifest."""

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
    / "outputs"
    / "metric_seam_pilot"
    / "reconstruction_v2"
    / "FREEZE_MANIFEST_2026-07-12.json"
)

SCOPED_PATHS = (
    "methods/metric_seam/RECONSTRUCTION_V2.md",
    "methods/metric_seam/reconstruction_v2.py",
    "methods/metric_seam/test_reconstruction_v2.py",
    "methods/metric_seam/freeze_reconstruction_v2.py",
    "methods/metric_seam/battery/_blind_worker_v2.py",
    "methods/metric_seam/battery/_sealed_worker_v2.py",
    "methods/metric_seam/battery/blind_reconstruction_v2.py",
    "methods/metric_seam/battery/evaluate_blind_v2.py",
    "methods/metric_seam/battery/split_ops_v2.py",
    "methods/metric_seam/battery/contract_check_isomorphic.py",
    "methods/metric_seam/battery/build_probe_extractions_v2.py",
    "methods/metric_seam/battery/build_contract_subrelation_readout.py",
    "methods/metric_seam/battery/dag_schema_enforced.py",
    "methods/metric_seam/battery/dag_schema_hardened.py",
    "methods/metric_seam/battery/verify_retrieval_scope_v2.py",
    "methods/metric_seam/battery/certify_batch_v2.py",
    "methods/metric_seam/battery/test_blind_reconstruction_v2.py",
    "methods/metric_seam/battery/test_evaluate_blind_v2.py",
    "methods/metric_seam/battery/test_contract_check_isomorphic.py",
    "methods/metric_seam/battery/test_build_probe_extractions_v2.py",
    "methods/metric_seam/battery/test_dag_schema_enforced.py",
    "methods/metric_seam/battery/test_dag_schema_hardened.py",
    "methods/metric_seam/battery/test_certify_batch_v2.py",
    "methods/metric_seam/hybrids/ops_capability.py",
    "methods/metric_seam/hybrids/ops_capability_v2.py",
    "methods/metric_seam/hybrids/test_ops_capability.py",
    "methods/metric_seam/hybrids/test_ops_capability_v2.py",
    "methods/metric_seam/science_claims_v2",
    "methods/metric_seam/technical_replay",
    "notes/2026-07-12__metric-seam-reconstruction-v2-progress.md",
    "notes/2026-07-12__reconstruction-v2-audit.md",
    "notes/2026-07-12__metric-seam-verification-handoff.md",
    "notes/2026-07-10__seam-agentic-program-runbook.md",
    "outputs/metric_seam_pilot/reconstruction_v2",
    "outputs/metric_seam_pilot/technical_replay_v2",
    "outputs/metric_seam_pilot/science_claims_v2/results.json",
    "outputs/metric_seam_pilot/science_claims_v2/CORRECTION_NOTICE.md",
    "outputs/metric_seam_pilot/science_claims_v2_corrected_v2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(paths: Iterable[str], output: Path) -> list[Path]:
    selected: set[Path] = set()
    for relative in paths:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(relative)
        if path.is_file():
            selected.add(path)
        else:
            selected.update(
                child
                for child in path.rglob("*")
                if child.is_file() and "__pycache__" not in child.parts
            )
    selected.discard(output)
    return sorted(selected, key=lambda path: str(path.relative_to(ROOT)))


def build_manifest(output: Path) -> dict:
    files = iter_files(SCOPED_PATHS, output)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema_version": "metric-seam-reconstruction-v2-freeze-v1",
        "date": "2026-07-12",
        "git_parent_at_freeze": head,
        "scope": (
            "verified blind-a144 core, audit-remediated instruments, technical replay, live "
            "L-probe runs, and current corrected science output"
        ),
        "policy": {
            "historical_artifacts_overwritten": False,
            "external_supervised_anchor_added": False,
            "ignored_outputs_are_bound_by_full_sha256": True,
            "future_changes_require_new_manifest": True,
        },
        "n_files": len(files),
        "files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }


def verify_manifest(output: Path) -> None:
    frozen = json.loads(output.read_text())
    current = build_manifest(output)
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
    print(f"PASS: {len(current['files'])} frozen files match full SHA-256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_manifest(args.out)
        return 0
    manifest = build_manifest(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({manifest['n_files']} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
