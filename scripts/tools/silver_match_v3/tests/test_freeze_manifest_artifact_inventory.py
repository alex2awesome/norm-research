from __future__ import annotations

import json
from pathlib import Path

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_manifest_artifact_inventory import (
    build_inventory,
)


def test_inventory_binds_manifest_and_artifacts(tmp_path: Path) -> None:
    bank = tmp_path / "bank.json"
    norms = tmp_path / "norms.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    norms.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {"t": {"path": str(bank)}},
                "corpora": {"c": {"path": str(norms)}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = build_inventory(manifest)
    assert result["source_manifest_sha256"] == sha256_file(manifest)
    assert result["artifact_count"] == 2
    assert {row["sha256"] for row in result["artifacts"]} == {
        sha256_file(bank),
        sha256_file(norms),
    }
