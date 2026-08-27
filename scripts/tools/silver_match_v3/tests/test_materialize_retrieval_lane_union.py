import json

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.materialize_retrieval_lane_union import (
    materialize_union,
)


def _fixture(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps({"metrics": [{"metric_id": metric} for metric in ("m0", "m1", "m2")]})
    )
    norms = tmp_path / "norms.jsonl"
    uid = "a" * 64
    write_jsonl(norms, [{"norm_uid": uid, "row": 0}])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {
                    "legal": {
                        "count": 3,
                        "path": str(bank),
                        "source_sha256": "frozen-bank-source",
                    }
                },
                "corpora": {
                    "court": {"task": "legal", "count": 1, "path": str(norms)}
                },
            }
        )
    )

    def lane(name, order):
        path = tmp_path / f"{name}.jsonl"
        write_jsonl(
            path,
            [
                {
                    "norm_uid": uid,
                    "corpus": "court",
                    "task": "legal",
                    "row": 0,
                    "bank_source_sha256": "frozen-bank-source",
                    "candidates": [
                        {"metric_id": metric, "rank": rank, "dense_score": 1 / rank}
                        for rank, metric in enumerate(order, 1)
                    ],
                }
            ],
        )
        meta = {
            "manifest_sha256": sha256_file(manifest),
            "output_sha256": sha256_file(path),
            "corpus": "court",
            "task": "legal",
            "bank_source_sha256": "frozen-bank-source",
            "output_k": 3,
            "input_count": 1,
            "encoder": name,
            "query_format": "raw",
            "query_views": "evidence+statement",
        }
        path.with_suffix(".jsonl.meta.json").write_text(json.dumps(meta))
        return path

    return manifest, lane("primary", ["m0", "m1", "m2"]), lane(
        "diverse", ["m2", "m1", "m0"]
    )


def test_materializes_deterministic_complete_lane_rrf_union(tmp_path):
    manifest, primary, diverse = _fixture(tmp_path)
    output = tmp_path / "union.jsonl"
    meta = materialize_union(
        manifest_path=manifest,
        corpus="court",
        lanes=[("primary", primary, 2.0), ("diverse", diverse, 1.0)],
        output_path=output,
        output_k=2,
    )
    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert [value["metric_id"] for value in rows[0]["candidates"]] == ["m0", "m1"]
    assert rows[0]["candidates"][0]["lane_ranks"] == {"diverse": 3, "primary": 1}
    assert meta["union"]["algorithm"] == "weighted-complete-bank-rrf-v1"
    assert meta["output_sha256"] == sha256_file(output)
    assert meta["input_count"] == 1


def test_fails_closed_on_lane_bank_universe_mismatch(tmp_path):
    manifest, primary, diverse = _fixture(tmp_path)
    row = next(read_jsonl(diverse))
    row["candidates"][2]["metric_id"] = "foreign"
    write_jsonl(diverse, [row])
    meta_path = diverse.with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["output_sha256"] = sha256_file(diverse)
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="bank universe"):
        materialize_union(
            manifest_path=manifest,
            corpus="court",
            lanes=[("primary", primary, 1.0), ("diverse", diverse, 1.0)],
            output_path=tmp_path / "bad.jsonl",
            output_k=2,
        )
    assert not (tmp_path / "bad.jsonl").exists()


def test_requires_two_distinct_positive_lanes(tmp_path):
    manifest, primary, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="at least two"):
        materialize_union(
            manifest_path=manifest,
            corpus="court",
            lanes=[("primary", primary, 1.0)],
            output_path=tmp_path / "bad.jsonl",
            output_k=2,
        )


def test_full_bank_mode_preserves_every_requested_component_prefix(tmp_path):
    manifest, primary, diverse = _fixture(tmp_path)
    for path, dense_order in (
        (primary, ["m2", "m1", "m0"]),
        (diverse, ["m0", "m1", "m2"]),
    ):
        row = next(read_jsonl(path))
        dense_rank = {metric: rank for rank, metric in enumerate(dense_order, 1)}
        for candidate in row["candidates"]:
            candidate["dense_rank"] = dense_rank[candidate["metric_id"]]
        write_jsonl(path, [row])
        meta_path = path.with_suffix(".jsonl.meta.json")
        meta = json.loads(meta_path.read_text())
        meta["output_sha256"] = sha256_file(path)
        meta_path.write_text(json.dumps(meta))

    output = tmp_path / "preserved.jsonl"
    meta = materialize_union(
        manifest_path=manifest,
        corpus="court",
        lanes=[("primary", primary, 1.0), ("diverse", diverse, 1.0)],
        output_path=output,
        output_k=3,
        preserve_components={
            "primary": ["rank", "dense_rank"],
            "diverse": ["rank", "dense_rank"],
        },
        preserve_k=1,
    )
    row = next(read_jsonl(output))
    assert row["preserved_prefix_union_size"] == 2
    preserved = [
        value["metric_id"]
        for value in row["candidates"]
        if value["preserved_prefix_member"]
    ]
    assert set(preserved) == {"m0", "m2"}
    assert [value["preserved_prefix_member"] for value in row["candidates"]] == [
        True,
        True,
        False,
    ]
    assert (
        meta["union"]["algorithm"]
        == "coverage-preserving-component-prefix-rrf-v1"
    )


def test_preserves_an_existing_audited_prefix_lane_with_full_bank_lanes(tmp_path):
    manifest, primary, prefix = _fixture(tmp_path)
    prefix_row = next(read_jsonl(prefix))
    prefix_row["candidates"] = prefix_row["candidates"][:2]
    write_jsonl(prefix, [prefix_row])
    meta_path = prefix.with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["output_sha256"] = sha256_file(prefix)
    meta["output_k"] = 2
    meta_path.write_text(json.dumps(meta))

    output = tmp_path / "prefix-preserved.jsonl"
    result = materialize_union(
        manifest_path=manifest,
        corpus="court",
        lanes=[("primary", primary, 1.0), ("existing", prefix, 1.0)],
        output_path=output,
        output_k=3,
        preserve_components={"primary": ["rank"], "existing": ["rank"]},
        preserve_k=1,
        prefix_lanes={"existing"},
    )
    row = next(read_jsonl(output))
    assert row["preserved_prefix_union_size"] == 2
    assert set(
        value["metric_id"]
        for value in row["candidates"]
        if value["preserved_prefix_member"]
    ) == {"m0", "m2"}
    lanes = result["union"]["lanes"]
    assert {lane["name"]: lane["kind"] for lane in lanes} == {
        "primary": "complete-bank",
        "existing": "preserved-prefix",
    }
