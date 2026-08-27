#!/usr/bin/env python3
"""Freeze the v3 sanitized, TRAIN-only preparation for active code a407.

This trusted projection performs no candidate execution, reference access,
model call, or evaluation.  It binds the corrected generic scope capability,
the common representation hook, two deterministic steward audits, and an
independent detect-secrets counts-only audit.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.metric_seam.battery.audit_ctext_compiler_view_v1 import (  # noqa: E402
    write_receipt as write_privacy_receipt,
)
from methods.metric_seam.battery.audit_detect_secrets_counts_v1 import (  # noqa: E402
    write_receipt as write_independent_secret_receipt,
)
from methods.metric_seam.battery.audit_full_corpus_sanitizer_v1 import (  # noqa: E402
    write_replay_receipt,
)
from methods.metric_seam.battery.seal_ctext_train_view_v3 import (  # noqa: E402
    prepare_train_view,
    sha256,
)


SOURCE = ROOT / "outputs/metric_seam_pilot/tasks/code_review/items.json"
CONTRACT = (
    ROOT
    / "methods/metric_seam/contracts/code_review_a407_projected_relation_contract_v2.json"
)
OUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "blind_code_a407_prepare_003_sanitized_scope_v3"
)
REJECTED_BUNDLE = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/blind_code_a407_prepare_001/"
    "compiler_bundle.json"
)

DEPENDENCIES = {
    "prepare_script_v3": Path(__file__),
    "privacy_interface_auditor": (
        ROOT / "methods/metric_seam/battery/audit_ctext_compiler_view_v1.py"
    ),
    "full_corpus_replay_auditor": (
        ROOT / "methods/metric_seam/battery/audit_full_corpus_sanitizer_v1.py"
    ),
    "independent_secret_counts_auditor": (
        ROOT / "methods/metric_seam/battery/audit_detect_secrets_counts_v1.py"
    ),
    "sealer_v3": ROOT / "methods/metric_seam/battery/seal_ctext_train_view_v3.py",
    "same_representation_hook": (
        ROOT / "methods/metric_seam/battery/sanitized_ctext_projection_v1.py"
    ),
    "ctext_projection_v2_dependency": (
        ROOT / "methods/metric_seam/battery/seal_ctext_items_v2.py"
    ),
    "scope_graph_v3": ROOT / "methods/metric_seam/hybrids/ops_code_scope_v3.py",
    "scope_graph_v2_transitive_helpers": (
        ROOT / "methods/metric_seam/hybrids/ops_code_scope_v2.py"
    ),
    "unified_diff_parser_dependency": ROOT / "methods/metric_seam/hybrids/ops_code.py",
}

DEPENDENCY_PACKAGES = {
    "detect-secrets",
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
        dependency_packages=DEPENDENCY_PACKAGES,
    )

    privacy_path = OUT / "steward_privacy_interface_receipt.json"
    privacy = write_privacy_receipt(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        source_path=SOURCE,
        receipt_path=privacy_path,
    )
    replay_path = OUT / "full_corpus_sanitizer_replay_receipt.json"
    replay = write_replay_receipt(
        source_path=SOURCE,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        receipt_path=replay_path,
    )
    independent_path = OUT / "detect_secrets_counts_only_receipt.json"
    independent = write_independent_secret_receipt(
        artifact_path=bundle_path,
        receipt_path=independent_path,
    )

    if not privacy["compiler_handoff_allowed"]:
        raise RuntimeError("compiler view failed the hidden steward interface audit")
    if not replay["replay_passed"]:
        raise RuntimeError("compiler view failed the hidden full-corpus replay audit")
    if not independent["scan_passed"]:
        raise RuntimeError("compiler view failed the hidden independent secret audit")

    # Never print ctext, source identifiers, scanner findings, or source paths.
    print(
        json.dumps(
            {
                "criterion_id": "a407",
                "phase": "corrected_capability_preparation_only",
                "compiler_bundle_sha256": sha256(bundle_path),
                "prepare_manifest_sha256": sha256(manifest_path),
                "privacy_receipt_sha256": sha256(privacy_path),
                "full_replay_receipt_sha256": sha256(replay_path),
                "independent_secret_receipt_sha256": sha256(independent_path),
                "privacy_scan_total_matches": privacy["credential_scan"][
                    "total_matches"
                ],
                "independent_aggregate_finding_count": independent[
                    "aggregate_finding_count"
                ],
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

