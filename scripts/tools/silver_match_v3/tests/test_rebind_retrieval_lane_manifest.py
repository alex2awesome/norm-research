import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.rebind_retrieval_lane_manifest import rebind_lane


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source"
    target = tmp_path / "target"
    bank_payload = {
        "metrics": [
            {"metric_id": "a0", "name": "zero"},
            {"metric_id": "a1", "name": "one"},
            {"metric_id": "a2", "name": "two"},
        ]
    }
    canonical = [
        {"norm_uid": "u0", "row": 0, "task": "humor", "corpus": "c"},
        {"norm_uid": "u1", "row": 1, "task": "humor", "corpus": "c"},
    ]
    for root in (source, target):
        _write_json(root / "bank.json", bank_payload)
        (root / "norms.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in canonical), encoding="utf-8"
        )
    corpus_science = {"task": "humor", "count": 2, "coverage_complete": True}
    bank_science = {"count": 3, "source_sha256": "bank-source"}
    source_manifest = tmp_path / "source.manifest.json"
    target_manifest = tmp_path / "target.manifest.json"
    _write_json(
        source_manifest,
        {
            "corpora": {"c": {**corpus_science, "path": str(source / "norms.jsonl")}},
            "banks": {"humor": {**bank_science, "path": str(source / "bank.json")}},
        },
    )
    _write_json(
        target_manifest,
        {
            "corpora": {"c": {**corpus_science, "path": str(target / "norms.jsonl")}},
            "banks": {"humor": {**bank_science, "path": str(target / "bank.json")}},
        },
    )
    candidate = tmp_path / "lane.jsonl"
    candidate.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "bank_source_sha256": "bank-source",
                    "candidates": [
                        {"metric_id": "a1", "rank": 1},
                        {"metric_id": "a0", "rank": 2},
                    ],
                }
            )
            + "\n"
            for row in canonical
        ),
        encoding="utf-8",
    )
    meta = tmp_path / "lane.jsonl.meta.json"
    _write_json(
        meta,
        {
            "task": "humor",
            "corpus": "c",
            "manifest": str(source_manifest),
            "manifest_sha256": sha256_file(source_manifest),
            "output_path": str(candidate),
            "output_sha256": sha256_file(candidate),
            "input_count": 2,
            "output_k": 2,
            "bank_source_sha256": "bank-source",
            "canonical_corpus": {
                "path": str(source / "norms.jsonl"),
                "sha256": sha256_file(source / "norms.jsonl"),
            },
            "bank_artifact": {
                "path": str(source / "bank.json"),
                "sha256": sha256_file(source / "bank.json"),
            },
        },
    )
    return {
        "candidate": candidate,
        "meta": meta,
        "source_manifest": source_manifest,
        "target_manifest": target_manifest,
        "output": tmp_path / "rebound.jsonl",
    }


def test_rebinds_only_manifest_and_paths_without_changing_candidate_bytes(tmp_path: Path) -> None:
    f = _fixture(tmp_path)
    result = rebind_lane(
        source_candidate=f["candidate"],
        source_meta=f["meta"],
        source_manifest=f["source_manifest"],
        target_manifest=f["target_manifest"],
        corpus="c",
        output_candidate=f["output"],
    )
    assert result["status"] == "CANONICAL_MANIFEST_REBOUND"
    assert result["candidate"]["rows"] == 2
    assert sha256_file(f["output"]) == sha256_file(f["candidate"])
    rebound = json.loads(f["output"].with_suffix(".jsonl.meta.json").read_text())
    assert rebound["manifest_sha256"] == sha256_file(f["target_manifest"])
    assert rebound["runtime_relocation"]["candidate_bytes_changed"] is False
    assert rebound["runtime_relocation"]["canonical_rows_verified"] == 2


def test_rejects_scientific_manifest_change(tmp_path: Path) -> None:
    f = _fixture(tmp_path)
    target = json.loads(f["target_manifest"].read_text())
    target["banks"]["humor"]["source_sha256"] = "different"
    _write_json(f["target_manifest"], target)
    with pytest.raises(ValueError, match="bank metadata"):
        rebind_lane(
            source_candidate=f["candidate"],
            source_meta=f["meta"],
            source_manifest=f["source_manifest"],
            target_manifest=f["target_manifest"],
            corpus="c",
            output_candidate=f["output"],
        )


def test_rejects_candidate_route_or_rank_change(tmp_path: Path) -> None:
    f = _fixture(tmp_path)
    rows = [json.loads(line) for line in f["candidate"].read_text().splitlines()]
    rows[0]["candidates"][0]["rank"] = 2
    f["candidate"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    meta = json.loads(f["meta"].read_text())
    meta["output_sha256"] = sha256_file(f["candidate"])
    _write_json(f["meta"], meta)
    with pytest.raises(ValueError, match="universe/ranks"):
        rebind_lane(
            source_candidate=f["candidate"],
            source_meta=f["meta"],
            source_manifest=f["source_manifest"],
            target_manifest=f["target_manifest"],
            corpus="c",
            output_candidate=f["output"],
        )


def test_relocates_fusion_and_adapter_only_when_bytes_match(tmp_path: Path) -> None:
    f = _fixture(tmp_path)
    source_fusion = tmp_path / "missing" / "fusion.json"
    runtime_fusion = tmp_path / "runtime.fusion.json"
    runtime_fusion.write_text("fusion")
    runtime_adapter = tmp_path / "runtime.adapter"
    runtime_adapter.mkdir()
    (runtime_adapter / "adapter.bin").write_text("adapter")
    meta = json.loads(f["meta"].read_text())
    meta["fusion_weights"] = str(source_fusion)
    meta["fusion_weights_sha256"] = sha256_file(runtime_fusion)
    meta["adapter"] = str(tmp_path / "missing" / "adapter")
    meta["adapter_hashes"] = {
        "adapter.bin": sha256_file(runtime_adapter / "adapter.bin")
    }
    _write_json(f["meta"], meta)
    rebind_lane(
        source_candidate=f["candidate"],
        source_meta=f["meta"],
        source_manifest=f["source_manifest"],
        target_manifest=f["target_manifest"],
        corpus="c",
        output_candidate=f["output"],
        runtime_fusion=runtime_fusion,
        runtime_adapter=runtime_adapter,
    )
    rebound = json.loads(f["output"].with_suffix(".jsonl.meta.json").read_text())
    assert rebound["fusion_weights"] == str(runtime_fusion.resolve())
    assert rebound["adapter"] == str(runtime_adapter.resolve())
    assert rebound["runtime_relocation"]["runtime_adapter"]["bytes_changed"] is False


def test_legacy_meta_can_use_source_artifact_inventory(tmp_path: Path) -> None:
    f = _fixture(tmp_path)
    meta = json.loads(f["meta"].read_text())
    del meta["canonical_corpus"]
    del meta["bank_artifact"]
    _write_json(f["meta"], meta)
    source_manifest = json.loads(f["source_manifest"].read_text())
    inventory = tmp_path / "inventory.json"
    _write_json(
        inventory,
        {
            "source_manifest_sha256": sha256_file(f["source_manifest"]),
            "artifacts": [
                {
                    "section": "corpora",
                    "name": "c",
                    "sha256": sha256_file(Path(source_manifest["corpora"]["c"]["path"])),
                },
                {
                    "section": "banks",
                    "name": "humor",
                    "sha256": sha256_file(Path(source_manifest["banks"]["humor"]["path"])),
                },
            ],
        },
    )
    rebind_lane(
        source_candidate=f["candidate"],
        source_meta=f["meta"],
        source_manifest=f["source_manifest"],
        target_manifest=f["target_manifest"],
        corpus="c",
        output_candidate=f["output"],
        source_artifact_inventory=inventory,
    )
    rebound = json.loads(f["output"].with_suffix(".jsonl.meta.json").read_text())
    assert rebound["runtime_relocation"]["source_artifact_inventory"]["sha256"] == sha256_file(inventory)


def test_legacy_unpinned_meta_requires_frozen_source_audit(tmp_path: Path) -> None:
    f = _fixture(tmp_path)
    meta = json.loads(f["meta"].read_text())
    meta["manifest_sha256"] = None
    _write_json(f["meta"], meta)
    audit = tmp_path / "source.audit.json"
    _write_json(
        audit,
        {
            "complete": True,
            "manifest_sha256": sha256_file(f["source_manifest"]),
            "corpus": "c",
            "candidate_inputs": {
                str(f["candidate"]): {
                    "sha256": sha256_file(f["candidate"]),
                    "meta_sha256": sha256_file(f["meta"]),
                }
            },
        },
    )
    rebind_lane(
        source_candidate=f["candidate"],
        source_meta=f["meta"],
        source_manifest=f["source_manifest"],
        target_manifest=f["target_manifest"],
        corpus="c",
        output_candidate=f["output"],
        source_audit=audit,
        source_audit_sha256=sha256_file(audit),
    )
    rebound = json.loads(f["output"].with_suffix(".jsonl.meta.json").read_text())
    assert rebound["runtime_relocation"]["source_audit"]["sha256"] == sha256_file(audit)
