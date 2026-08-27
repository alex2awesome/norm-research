#!/usr/bin/env python3
"""Create the clean, offline a407 heldout/code/raw/hybrid preparation.

The historical item file is accessible only to the trusted heldout sealer and
its two steward replay/privacy auditors.  Downstream preparation receives the
readonly opaque heldout bundle only.  This command executes the frozen code
candidate but performs no historical-reference access, correlation,
evaluation, API/model call, or GPU operation.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.metric_seam.battery.audit_detect_secrets_counts_v1 import (  # noqa: E402
    write_receipt as write_detect_secrets_receipt,
)
from methods.metric_seam.battery.audit_sanitized_ctext_heldout_v1 import (  # noqa: E402
    write_privacy_receipt,
    write_replay_receipt,
)
from methods.metric_seam.battery.seal_ctext_items_v2 import sha256  # noqa: E402
from methods.metric_seam.battery.seal_sanitized_ctext_heldout_v1 import (  # noqa: E402
    seal_heldout_view,
)
from methods.metric_seam.pilot.a407_dual_channel_pipeline_v1 import (  # noqa: E402
    prepare_downstream_bundle,
    verify_preparation_bundle,
)


SOURCE = ROOT / "outputs/metric_seam_pilot/tasks/code_review/items.json"
OUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "a407_dual_channel_prepare_002_clean"
)


def _cleanup_owned_partial_output() -> None:
    if not OUT.exists():
        return
    for path in OUT.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
        elif path.is_dir():
            path.chmod(0o755)
    OUT.chmod(0o755)
    shutil.rmtree(OUT)


def main() -> int:
    if OUT.exists():
        raise FileExistsError("refusing to overwrite the canonical clean preparation")

    bundle_path = OUT / "heldout_bundle.json"
    seal_manifest_path = OUT / "heldout_seal_manifest.json"
    privacy_path = OUT / "steward_heldout_privacy_receipt.json"
    replay_path = OUT / "full_corpus_heldout_replay_receipt.json"
    detect_path = OUT / "detect_secrets_counts_only_receipt.json"
    try:
        seal_heldout_view(
            source_path=SOURCE,
            bundle_path=bundle_path,
            manifest_path=seal_manifest_path,
            task="code_review",
            criterion_id="a407",
            train_count=150,
            heldout_count=100,
            split_seed=7,
        )
        privacy = write_privacy_receipt(
            source_path=SOURCE,
            bundle_path=bundle_path,
            manifest_path=seal_manifest_path,
            receipt_path=privacy_path,
        )
        replay = write_replay_receipt(
            source_path=SOURCE,
            bundle_path=bundle_path,
            manifest_path=seal_manifest_path,
            receipt_path=replay_path,
        )
        independent = write_detect_secrets_receipt(
            artifact_path=bundle_path,
            receipt_path=detect_path,
        )
        if not privacy.get("audit_passed"):
            raise RuntimeError("heldout interface failed the hidden steward privacy audit")
        if not replay.get("replay_passed"):
            raise RuntimeError("heldout view failed the hidden steward replay audit")
        if not independent.get("scan_passed"):
            raise RuntimeError("heldout view failed the hidden independent secret audit")
        if replay["replay"]["heldout"]["row_count"] != 100:
            raise RuntimeError("heldout replay count differs from the frozen design")
        if replay["replay"]["heldout"]["changed_row_count"] != 1:
            raise RuntimeError("heldout changed-row count differs from the preregistration")

        manifest_path = prepare_downstream_bundle(
            heldout_bundle_path=bundle_path,
            heldout_seal_manifest_path=seal_manifest_path,
            privacy_receipt_path=privacy_path,
            replay_receipt_path=replay_path,
            detect_secrets_receipt_path=detect_path,
            out_dir=OUT,
        )
        manifest, arms, _model = verify_preparation_bundle(OUT)
        if manifest["execution_status"]["api_calls"] is not False:
            raise AssertionError("offline preparation unexpectedly records API calls")
    except Exception:
        _cleanup_owned_partial_output()
        raise

    # Counts and hashes only.  Never print source paths, keys, IDs, ctext,
    # candidate values, reference values, prompts, scanner findings, or excerpts.
    print(json.dumps({
        "artifact_count": len(manifest["artifacts"]) + 1,
        "candidate_output_count": manifest["candidate_arm"]["exact_output_count"],
        "heldout_count": 100,
        "heldout_bundle_sha256": sha256(bundle_path),
        "hybrid_request_count": len(arms["hybrid"]),
        "manifest_sha256": sha256(manifest_path),
        "raw_prompt_request_count": len(arms["raw_prompt"]),
        "sanitizer_changed_heldout_count": 1,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
