#!/usr/bin/env python3
"""Freeze the label-free preparation artifacts for active code criterion a407.

This script performs no model call, candidate execution, or evaluation.  It is
the trusted one-way projection step from the active code corpus into an opaque,
TRAIN-only compiler view.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.metric_seam.battery.seal_ctext_train_view_v3 import (  # noqa: E402
    prepare_train_view,
    sha256,
)
from methods.metric_seam.battery.audit_ctext_compiler_view_v1 import (  # noqa: E402
    write_receipt,
)


SOURCE = ROOT / "outputs/metric_seam_pilot/tasks/code_review/items.json"
CONTRACT = (
    ROOT
    / "methods/metric_seam/contracts/code_review_a407_projected_relation_contract_v1.json"
)
OUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/blind_code_a407_prepare_002_sanitized"
)
REJECTED_BUNDLE = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/blind_code_a407_prepare_001/"
    "compiler_bundle.json"
)

DEPENDENCIES = {
    "prepare_script": Path(__file__),
    "privacy_interface_auditor": (
        ROOT / "methods/metric_seam/battery/audit_ctext_compiler_view_v1.py"
    ),
    "sealer_v3": ROOT / "methods/metric_seam/battery/seal_ctext_train_view_v3.py",
    "ctext_projection_v2_dependency": (
        ROOT / "methods/metric_seam/battery/seal_ctext_items_v2.py"
    ),
    "scope_graph_v2": ROOT / "methods/metric_seam/hybrids/ops_code_scope_v2.py",
    "unified_diff_parser_dependency": ROOT / "methods/metric_seam/hybrids/ops_code.py",
}

TREE_SITTER_PACKAGES = {
    "tree-sitter",
    "tree-sitter-go",
    "tree-sitter-java",
    "tree-sitter-javascript",
    "tree-sitter-python",
    "tree-sitter-typescript",
}


def main() -> int:
    if REJECTED_BUNDLE.exists():
        raise RuntimeError("rejected unsanitized compiler bundle still exists")
    bundle_path, manifest_path = prepare_train_view(
        source=SOURCE,
        contract_path=CONTRACT,
        out_dir=OUT,
        task="code_review",
        criterion_id="a407",
        train_count=150,
        split_seed=7,
        dependency_files=DEPENDENCIES,
        dependency_packages=TREE_SITTER_PACKAGES,
    )
    receipt_path = OUT / "steward_privacy_interface_receipt.json"
    receipt = write_receipt(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        source_path=SOURCE,
        receipt_path=receipt_path,
    )
    if not receipt["compiler_handoff_allowed"]:
        raise RuntimeError(
            "sanitized compiler view failed steward audit; matching content was not printed"
        )
    # Diff bodies and source identifiers are deliberately never printed.
    print(
        json.dumps(
            {
                "criterion_id": "a407",
                "phase": "label_free_preparation_only",
                "compiler_bundle": str(bundle_path),
                "compiler_bundle_sha256": sha256(bundle_path),
                "prepare_manifest": str(manifest_path),
                "steward_receipt": str(receipt_path),
                "privacy_scan_total_matches": receipt["credential_scan"]["total_matches"],
                "source_identifier_occurrence_count": (
                    receipt["interface_audit"]["source_identifier_occurrence_count"]
                ),
                "structural_outcome_key_count": (
                    receipt["interface_audit"]["structural_outcome_key_count"]
                ),
                "train_count": 150,
                "heldout_count": 100,
                "candidate_authored": False,
                "candidate_executed": False,
                "reference_opened": False,
                "model_calls": False,
                "gpu_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
