import json

import pytest

from scripts.tools.silver_match_v3.audit_candidate_outputs import audit_candidates
from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.truncate_candidate_depth import (
    truncate_candidate_depth,
)


def make_fixture(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "metrics": [
                    {"metric_id": f"m{i}", "name": str(i), "description": str(i)}
                    for i in range(1, 4)
                ]
            }
        )
    )
    norms = tmp_path / "norms.jsonl"
    write_jsonl(norms, [{"norm_uid": "a" * 64, "row": 0}])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpora": {"c": {"task": "t", "count": 1, "path": str(norms)}},
                "banks": {
                    "t": {"count": 3, "path": str(bank), "source_sha256": "bank-sha"}
                },
            }
        )
    )
    source = tmp_path / "source.jsonl"
    write_jsonl(
        source,
        [
            {
                "norm_uid": "a" * 64,
                "corpus": "c",
                "task": "t",
                "row": 0,
                "bank_source_sha256": "bank-sha",
                "candidates": [
                    {"metric_id": f"m{i}", "rank": i, "rrf_score": 1.0 / i}
                    for i in range(1, 4)
                ],
            }
        ],
    )
    fusion = tmp_path / "fusion.json"
    fusion.write_text(json.dumps({"selection_split": "dev", "task": "t"}))
    source_meta = source.with_suffix(".jsonl.meta.json")
    source_meta.write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "corpus": "c",
                "task": "t",
                "input_count": 1,
                "output_path": str(source),
                "output_sha256": sha256_file(source),
                "bank_source_sha256": "bank-sha",
                "encoder": "/models/frozen",
                "adapter": None,
                "query_format": "nemotron",
                "dense_query_instruction": True,
                "query_views": "evidence+statement",
                "fusion_weights": str(fusion),
                "fusion_weights_sha256": sha256_file(fusion),
                "output_k": 3,
            }
        )
    )
    return manifest, source, source_meta


def test_projection_is_an_exact_auditable_prefix(tmp_path):
    manifest, source, source_meta = make_fixture(tmp_path)
    output = tmp_path / "top2.jsonl"
    meta = truncate_candidate_depth(
        input_path=source, output_path=output, output_k=2
    )
    row = next(iter(json.loads(line) for line in output.read_text().splitlines()))
    assert [candidate["metric_id"] for candidate in row["candidates"]] == ["m1", "m2"]
    assert meta["projection"]["source_candidates_sha256"] == sha256_file(source)
    assert meta["projection"]["source_meta_sha256"] == sha256_file(source_meta)
    report = audit_candidates(
        manifest_path=manifest,
        corpus="c",
        candidate_paths=[output],
        expected_k=2,
    )
    assert report["complete"] is True


def test_projection_rejects_noncontiguous_source_ranks(tmp_path):
    _, source, source_meta = make_fixture(tmp_path)
    row = json.loads(source.read_text())
    row["candidates"][1]["rank"] = 9
    write_jsonl(source, [row])
    payload = json.loads(source_meta.read_text())
    payload["output_sha256"] = sha256_file(source)
    source_meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not contiguous"):
        truncate_candidate_depth(
            input_path=source, output_path=tmp_path / "top2.jsonl", output_k=2
        )
