import json

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs import (
    PAIR_SCHEMA,
    materialize,
)
from scripts.tools.silver_match_v3.run_nemotron_ce import score_pair_from_row


def _fixture(tmp_path):
    bank_hash = "frozen-bank-source"
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "task": "legal",
                "source_sha256": bank_hash,
                "metrics": [
                    {"metric_id": "m0", "name": "Reasoning", "description": "sound"},
                    {"metric_id": "m1", "name": "Support", "description": "supported"},
                ],
            }
        )
    )
    corpora = {}
    candidates = {}
    for index, corpus in enumerate(("court_b", "court_a")):
        uid = str(index + 1) * 64
        norms = tmp_path / f"{corpus}.norms.jsonl"
        write_jsonl(
            norms,
            [
                {
                    "schema_version": "silver-match-v3.0",
                    "task": "legal",
                    "corpus": corpus,
                    "norm_uid": uid,
                    "row": 0,
                    "source_id": f"source-{index}",
                    "norm": f"The reasoning criterion {index} was explicit.",
                    "context": "A longer passage containing the evaluative criterion.",
                }
            ],
        )
        corpora[corpus] = {"task": "legal", "count": 1, "path": str(norms)}
        union = tmp_path / f"{corpus}.union.jsonl"
        write_jsonl(
            union,
            [
                {
                    "schema_version": "silver-match-v3.0",
                    "task": "legal",
                    "corpus": corpus,
                    "norm_uid": uid,
                    "row": 0,
                    "bank_source_sha256": bank_hash,
                    "candidates": [
                        {"metric_id": "m0", "rank": 1},
                        {"metric_id": "m1", "rank": 2},
                    ],
                }
            ],
        )
        candidates[corpus] = union
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {
                    "legal": {
                        "count": 2,
                        "path": str(bank),
                        "source_sha256": bank_hash,
                    }
                },
                "corpora": corpora,
            }
        )
    )
    for corpus, union in candidates.items():
        union.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "output_sha256": sha256_file(union),
                    "manifest_sha256": sha256_file(manifest),
                    "corpus": corpus,
                    "task": "legal",
                    "bank_source_sha256": bank_hash,
                    "input_count": 1,
                    "output_k": 2,
                    "union": {
                        "lanes": [
                            {
                                "name": "primary",
                                "kind": "complete-bank",
                                "sha256": "a" * 64,
                            },
                            {
                                "name": "diverse",
                                "kind": "complete-bank",
                                "sha256": "b" * 64,
                            },
                        ]
                    },
                }
            )
        )
    return manifest, candidates


def test_materializes_unlabeled_pairs_and_exact_task_universe(tmp_path):
    manifest, candidates = _fixture(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    universe = tmp_path / "universe.jsonl"
    report = materialize(
        manifest_path=manifest,
        task="legal",
        candidates=candidates,
        output_path=pairs,
        universe_path=universe,
        expected_k=2,
    )
    pair_rows = list(read_jsonl(pairs))
    universe_rows = list(read_jsonl(universe))
    assert report["corpus_order"] == ["court_b", "court_a"]
    assert report["norm_count"] == 2
    assert report["pair_count"] == 4
    assert len(pair_rows) == 4
    assert len(universe_rows) == 2
    assert {row["schema_version"] for row in pair_rows} == {PAIR_SCHEMA}
    assert all("label" not in row and "relation" not in row for row in pair_rows)
    assert [row["corpus"] for row in universe_rows] == ["court_b", "court_a"]
    assert all(row["split"] == "production" for row in universe_rows)
    parsed = score_pair_from_row(pair_rows[0], "fixture")
    assert parsed.gold_relation is None
    assert parsed.example.label == "REJECT"  # inference-only collator placeholder
    assert report["pairs"]["sha256"] == sha256_file(pairs)


def test_rejects_single_lane_union_and_removes_partial_outputs(tmp_path):
    manifest, candidates = _fixture(tmp_path)
    first = next(iter(candidates.values()))
    meta_path = first.with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["union"]["lanes"] = [{"name": "primary"}]
    meta_path.write_text(json.dumps(meta))
    pairs = tmp_path / "pairs.jsonl"
    universe = tmp_path / "universe.jsonl"
    with pytest.raises(ValueError, match="candidate union contract"):
        materialize(
            manifest_path=manifest,
            task="legal",
            candidates=candidates,
            output_path=pairs,
            universe_path=universe,
            expected_k=2,
        )
    assert not pairs.exists()
    assert not universe.exists()


def test_rejects_two_prefix_lanes_without_complete_bank_capture(tmp_path):
    manifest, candidates = _fixture(tmp_path)
    first = next(iter(candidates.values()))
    meta_path = first.with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    for lane in meta["union"]["lanes"]:
        lane["kind"] = "preserved-prefix"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="two complete-bank lanes"):
        materialize(
            manifest_path=manifest,
            task="legal",
            candidates=candidates,
            output_path=tmp_path / "pairs.jsonl",
            universe_path=tmp_path / "universe.jsonl",
            expected_k=2,
        )


def test_requires_exact_all_corpus_bindings(tmp_path):
    manifest, candidates = _fixture(tmp_path)
    candidates.pop("court_a")
    with pytest.raises(ValueError, match="candidate corpus bindings differ"):
        materialize(
            manifest_path=manifest,
            task="legal",
            candidates=candidates,
            output_path=tmp_path / "pairs.jsonl",
            universe_path=tmp_path / "universe.jsonl",
            expected_k=2,
        )


def test_rejects_diagnostic_subset_count(tmp_path):
    manifest, candidates = _fixture(tmp_path)
    first = next(iter(candidates.values()))
    meta_path = first.with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["input_count"] = 0
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="candidate union contract"):
        materialize(
            manifest_path=manifest,
            task="legal",
            candidates=candidates,
            output_path=tmp_path / "pairs.jsonl",
            universe_path=tmp_path / "universe.jsonl",
            expected_k=2,
        )


def test_accepts_complete_bank_union_and_projects_requested_prefix(tmp_path):
    manifest, candidates = _fixture(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    report = materialize(
        manifest_path=manifest,
        task="legal",
        candidates=candidates,
        output_path=pairs,
        universe_path=tmp_path / "universe.jsonl",
        expected_k=1,
    )
    rows = list(read_jsonl(pairs))
    assert report["candidate_depth"] == 1
    assert report["pair_count"] == 2
    assert len(rows) == 2
    assert {row["metric_id"] for row in rows} == {"m0"}
    assert all(
        corpus["candidate_union"]["output_k"] == 2
        for corpus in report["corpora"].values()
    )
