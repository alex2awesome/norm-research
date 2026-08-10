from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.relocate_consensus_truth_handoff import relocate


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_relocation_preserves_truth_bytes_and_only_rebinds_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    validation = source / "validation.json"
    _json(validation, {"task": "humor"})
    rows = {
        "train": [{"norm_uid": "t", "split": "train"}],
        "dev": [{"norm_uid": "d", "split": "dev"}],
        "test": [{"norm_uid": "x", "split": "test", "decision": "HIDDEN"}],
    }
    paths = {}
    all_rows = []
    for split, values in rows.items():
        path = source / f"truth.{split}.jsonl"
        write_jsonl(path, values)
        paths[split] = path
        all_rows.extend(values)
    paths["all"] = source / "truth.all.jsonl"
    write_jsonl(paths["all"], all_rows)
    manifest = source / "MANIFEST.json"
    _json(
        manifest,
        {
            "schema_version": "silver-match-v3-consensus-training-truth-manifest-v1",
            "status": "COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS",
            "task": "humor",
            "source_group_cross_split_count": 0,
            "blind_rows_training_eligible": 0,
            "inputs": {
                "pack_validation": {
                    "path": str(validation),
                    "sha256": sha256_file(validation),
                }
            },
            "outputs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "count": len(all_rows) if name == "all" else len(rows[name]),
                }
                for name, path in paths.items()
            },
        },
    )
    eligible = source / "eligible.jsonl"
    typed = source / "typed.jsonl"
    write_jsonl(eligible, rows["train"])
    write_jsonl(typed, rows["dev"])
    ce_report = source / "CE_REPORT.json"
    _json(
        ce_report,
        {
            "schema_version": "silver-match-v3-ce-eligible-truth-report-v1",
            "status": "PARTITIONED_WITHOUT_INFERRED_FAMILY_ANCHORS",
            "task": "humor",
            "source_groups_crossing_splits": 0,
            "input": {
                "path": str(paths["all"]),
                "sha256": sha256_file(paths["all"]),
                "count": len(all_rows),
            },
            "outputs": {
                "eligible": {
                    "path": str(eligible),
                    "sha256": sha256_file(eligible),
                    "count": 1,
                },
                "typed_only": {
                    "path": str(typed),
                    "sha256": sha256_file(typed),
                    "count": 1,
                },
            },
        },
    )
    destination = tmp_path / "relocated"
    published = Path("/published/consensus")
    report = relocate(
        argparse.Namespace(
            manifest=str(manifest),
            ce_report=str(ce_report),
            source_validation=str(validation),
            output_root=str(destination),
            published_output_root=str(published),
        )
    )
    assert report["truth_or_ce_output_bytes_changed"] is False
    for name, source_path in paths.items():
        relocated = destination / f"truth.{name}.jsonl"
        assert sha256_file(relocated) == sha256_file(source_path)
    relocated_manifest = json.loads((destination / "MANIFEST.json").read_text())
    assert relocated_manifest["outputs"]["dev"]["path"] == str(
        published / "truth.dev.jsonl"
    )
    assert (destination / "truth.test.jsonl").read_bytes() == paths[
        "test"
    ].read_bytes()
