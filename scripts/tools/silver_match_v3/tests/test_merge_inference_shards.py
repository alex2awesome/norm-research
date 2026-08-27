import json

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file
from scripts.tools.silver_match_v3.merge_inference_shards import merge_shards
from scripts.tools.silver_match_v3.retrieve import stable_shard


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _candidate_fixture(tmp_path, *, changed_prompt=False, omit_uid=None):
    candidates = tmp_path / "candidates.jsonl"
    candidate_rows = [
        {"norm_uid": f"{index:064x}", "candidates": []} for index in range(12)
    ]
    _write_jsonl(candidates, candidate_rows)
    paths = []
    for shard_id in range(3):
        path = tmp_path / f"shard-{shard_id}.jsonl"
        rows = [
            {
                "norm_uid": row["norm_uid"],
                "decision": "MATCH" if shard_id else "INVALID_OUTPUT",
                "inference_representative_norm_uid": row["norm_uid"],
            }
            for row in candidate_rows
            if stable_shard(row["norm_uid"], 3) == shard_id
            and row["norm_uid"] != omit_uid
        ]
        _write_jsonl(path, rows)
        meta = {
            "schema_version": "v3",
            "input_candidates": str(candidates),
            "input_candidates_sha256": sha256_file(candidates),
            "output": str(path),
            "output_sha256": sha256_file(path),
            "prompt_sha256": "b" * 64 if changed_prompt and shard_id == 2 else "a" * 64,
            "model": "/model/snapshot",
            "order_mode": "original",
            "max_candidates": 50,
            "prompt_rendering": {"context_chars": 1200},
            "new_count": len(rows),
            "eligible_count": len(rows),
            "unique_prompt_inferences": len(rows),
            "deduplicated_prompt_count": 0,
            "retry_prompt_inferences": 0,
            "invalid_count": sum(row["decision"] == "INVALID_OUTPUT" for row in rows),
            "shard_id": shard_id,
            "num_shards": 3,
            "elapsed_seconds": 1.0,
        }
        path.with_suffix(".jsonl.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        paths.append(path)
    return paths, candidate_rows


def test_merge_shards_proves_coverage_and_emits_audit_compatible_meta(tmp_path):
    paths, candidates = _candidate_fixture(tmp_path)
    output = tmp_path / "combined.jsonl"
    meta = merge_shards(input_paths=paths, output_path=output)
    rows = list(read_jsonl(output))
    assert [row["norm_uid"] for row in rows] == sorted(
        row["norm_uid"] for row in candidates
    )
    assert meta["output_sha256"] == sha256_file(output)
    assert meta["new_count"] == meta["eligible_count"] == 12
    assert meta["invalid_count"] == len(paths[0].read_text().splitlines())
    assert meta["shard_id"] == 0 and meta["num_shards"] == 1
    assert meta["combined_from_num_shards"] == 3


def test_merge_shards_rejects_runtime_drift(tmp_path):
    paths, _ = _candidate_fixture(tmp_path, changed_prompt=True)
    with pytest.raises(ValueError, match="runtime metadata differs"):
        merge_shards(input_paths=paths, output_path=tmp_path / "combined.jsonl")


def test_merge_shards_rejects_incomplete_source_coverage(tmp_path):
    omitted = f"{3:064x}"
    paths, candidates = _candidate_fixture(tmp_path, omit_uid=omitted)
    assert any(row["norm_uid"] == omitted for row in candidates)
    with pytest.raises(ValueError, match="merged coverage differs"):
        merge_shards(input_paths=paths, output_path=tmp_path / "combined.jsonl")
