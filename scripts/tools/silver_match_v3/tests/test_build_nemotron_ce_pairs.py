import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.build_nemotron_ce_pairs import build, write_release


def _write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path, *, include_candidates: bool = True):
    bank_hash = hashlib.sha256(b"frozen-bank-source").hexdigest()
    bank = {
        "task": "humor",
        "source_sha256": bank_hash,
        "metrics": [
            {
                "task": "humor",
                "metric_id": f"m{index}",
                "name": f"metric {index}",
                "description": f"definition {index}",
                "examples": [],
            }
            for index in range(4)
        ],
    }
    source_groups = {
        "train-1": "humor\x1fjokes\x1fsource\x1fsource-train-1",
        "dev-1": "humor\x1fjokes\x1fsource\x1fsource-dev-1",
    }
    norms = [
        {
            "task": "humor",
            "corpus": "jokes",
            "norm_uid": uid,
            "source_id": source_id,
            "row": index,
            "norm": f"human statement {uid}",
            "context": f"evidence {uid}",
        }
        for index, (uid, source_id) in enumerate(
            (
                ("train-1", "source-train-1"),
                ("dev-1", "source-dev-1"),
            )
        )
    ]
    truth = [
        {
            "task": "humor",
            "corpus": "jokes",
            "norm_uid": "train-1",
            "source_group": source_groups["train-1"],
            "split": "train",
            "decision": "MATCH",
            "metric_id": "m0",
            "acceptable_metric_ids": ["m0", "m1"],
            "current_bank_source_sha256": bank_hash,
            "gradient_eligible": True,
        },
        {
            "task": "humor",
            "corpus": "jokes",
            "norm_uid": "dev-1",
            "source_group": source_groups["dev-1"],
            "split": "dev",
            "decision": "MATCH",
            "metric_id": "m1",
            "acceptable_metric_ids": ["m1"],
            "current_bank_source_sha256": bank_hash,
        },
    ]
    hierarchy = {
        "task": "humor",
        "n_r2_clusters_in": 4,
        "n_merged_groups": 1,
        "merged_groups": [{"metric_ids": ["m0", "m1", "m2"]}],
    }
    candidate_rows = [
        {
            "task": "humor",
            "norm_uid": uid,
            "bank_source_sha256": bank_hash,
            "candidates": candidates,
        }
        for uid, candidates in (
            (
                "train-1",
                [
                    {"metric_id": "m1", "rank": 1, "score": 0.8},
                    {"metric_id": "m2", "rank": 2, "score": 0.7},
                    {"metric_id": "m3", "rank": 3, "score": 0.6},
                ],
            ),
            ("dev-1", [{"metric_id": "m3", "rank": 1, "score": 0.5}]),
        )
    ]
    candidate_rows.append(
        {
            "task": "humor",
            "norm_uid": "outside-truth",
            "bank_source_sha256": bank_hash,
            "candidates": [{"metric_id": "m0", "rank": 1}],
        }
    )
    lexical_rows = [
        {
            "task": "humor",
            "norm_uid": "train-1",
            "bank_source_sha256": bank_hash,
            "candidates": [
                {"metric_id": "m3", "rank": 1, "retrieval_lane": "word"},
                {"metric_id": "m2", "rank": 2, "retrieval_lane": "word"},
            ],
        }
    ]
    bank_path = _write_json(tmp_path / "bank.json", bank)
    norms_path = _write_jsonl(tmp_path / "norms.jsonl", norms)
    truth_path = _write_jsonl(tmp_path / "truth.jsonl", truth)
    hierarchy_path = _write_json(tmp_path / "hierarchy.json", hierarchy)
    dense_path = _write_jsonl(tmp_path / "dense.jsonl", candidate_rows)
    lexical_path = _write_jsonl(tmp_path / "lexical.jsonl", lexical_rows)
    manifest = {
        "banks": {
            "humor": {
                "path": str(bank_path),
                "source_sha256": bank_hash,
                "count": 4,
            }
        },
        "corpora": {
            "jokes": {
                "task": "humor",
                "path": str(norms_path),
                "count": len(norms),
            }
        },
    }
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)
    args = argparse.Namespace(
        manifest=str(manifest_path),
        task="humor",
        bank=str(bank_path),
        truth=[str(truth_path)],
        split_assignments=None,
        candidates=(
            [f"dense={dense_path}", f"lexical={lexical_path}"]
            if include_candidates
            else []
        ),
        hierarchy=str(hierarchy_path),
        maximum_pairs=400_000,
        global_negatives_per_norm=0,
        context_chars=1600,
        seed=713,
        output=str(tmp_path / "pairs.jsonl"),
        report=str(tmp_path / "report.json"),
    )
    return args, manifest_path, bank_path, truth_path, dense_path


def test_relations_union_provenance_and_no_nontrain_gold_injection(tmp_path: Path):
    args, *_ = _fixture(tmp_path)
    rows, report = build(args)
    by_pair = {(row["norm_uid"], row["metric_id"]): row for row in rows}

    assert by_pair[("train-1", "m0")]["relation"] == "EXACT"
    assert by_pair[("train-1", "m0")]["candidate_lanes"] == ["train_gold_injection"]
    assert by_pair[("train-1", "m1")]["relation"] == "EXACT"
    assert by_pair[("train-1", "m2")]["relation"] == "FAMILY"
    assert by_pair[("train-1", "m3")]["relation"] == "REJECT"
    assert by_pair[("train-1", "m3")]["candidate_lanes"] == ["dense", "lexical"]
    assert ("dev-1", "m1") not in by_pair
    assert by_pair[("dev-1", "m3")]["relation"] == "REJECT"
    assert by_pair[("dev-1", "m3")]["gradient_eligible"] is False
    assert report["acceptable_as_reject_count"] == 0
    assert report["derived_candidate_rows_outside_train"] == 0
    assert report["candidate_union_audit"]["outside_truth_rows_ignored"] == 1


def test_global_negative_lane_is_deterministic_and_metric_balanced(tmp_path: Path):
    args, manifest_path, bank_path, truth_path, _ = _fixture(
        tmp_path, include_candidates=False
    )
    bank_hash = json.loads(bank_path.read_text())["source_sha256"]
    norms = []
    truth = []
    candidate_rows = []
    for index in range(8):
        uid = f"u{index}"
        source_id = f"s{index}"
        group = f"humor\x1fjokes\x1fsource\x1f{source_id}"
        norms.append(
            {
                "task": "humor",
                "corpus": "jokes",
                "norm_uid": uid,
                "source_id": source_id,
                "row": index,
                "norm": f"statement {index}",
            }
        )
        truth.append(
            {
                "task": "humor",
                "corpus": "jokes",
                "norm_uid": uid,
                "source_group": group,
                "split": "train",
                "decision": "NOISE",
                "metric_id": None,
                "acceptable_metric_ids": [],
                "current_bank_source_sha256": bank_hash,
            }
        )
        candidate_rows.append(
            {
                "task": "humor",
                "norm_uid": uid,
                "bank_source_sha256": bank_hash,
                "candidates": [{"metric_id": "m0", "rank": 1}],
            }
        )
    norms_path = tmp_path / "norms.jsonl"
    _write_jsonl(norms_path, norms)
    _write_jsonl(truth_path, truth)
    manifest = json.loads(manifest_path.read_text())
    manifest["corpora"]["jokes"]["path"] = str(norms_path)
    manifest["corpora"]["jokes"]["count"] = len(norms)
    _write_json(manifest_path, manifest)
    candidates_path = _write_jsonl(tmp_path / "cover.jsonl", candidate_rows)
    args.candidates = [f"retriever={candidates_path}"]
    args.global_negatives_per_norm = 1

    first, _ = build(args)
    second, _ = build(args)
    assert first == second
    assert len(first) == 16
    assert {row["relation"] for row in first} == {"REJECT"}
    global_rows = [
        row for row in first if row["candidate_lanes"] == ["global_balanced_negative"]
    ]
    assert len(global_rows) == 8
    exposure = {}
    for row in global_rows:
        exposure[row["metric_id"]] = exposure.get(row["metric_id"], 0) + 1
    assert max(exposure.values()) - min(exposure.values()) <= 1


def test_cap_is_deterministic_and_report_binds_output_hash(tmp_path: Path):
    args, *_ = _fixture(tmp_path)
    args.maximum_pairs = 5
    first, first_report = build(args)
    second, second_report = build(args)
    assert first == second
    assert first_report == second_report
    assert len(first) == 5
    assert all(
        row["relation"] == "EXACT"
        for row in first
        if row["metric_id"] in row["acceptable_metric_ids"]
    )

    release = write_release(args)
    output = Path(args.output)
    assert release["output"]["count"] == 5
    assert (
        release["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    saved = json.loads(Path(args.report).read_text())
    assert (
        saved["inputs"]["bank"]["sha256"]
        == hashlib.sha256(Path(args.bank).read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_release(args)


def test_fails_closed_on_bank_hash_drift(tmp_path: Path):
    args, _, bank_path, *_ = _fixture(tmp_path)
    bank = json.loads(bank_path.read_text())
    bank["source_sha256"] = "drifted"
    _write_json(bank_path, bank)
    with pytest.raises(ValueError, match="bank task/source hash differs"):
        build(args)


def test_fails_closed_on_source_group_split_leakage(tmp_path: Path):
    args, _, _, truth_path, _ = _fixture(tmp_path)
    truth = [json.loads(line) for line in truth_path.read_text().splitlines()]
    truth[1]["source_group"] = truth[0]["source_group"]
    _write_jsonl(truth_path, truth)
    with pytest.raises(ValueError, match="source group crosses truth splits"):
        build(args)


def test_fails_closed_on_out_of_bank_candidate(tmp_path: Path):
    args, _, _, _, dense_path = _fixture(tmp_path)
    rows = [json.loads(line) for line in dense_path.read_text().splitlines()]
    rows[0]["candidates"][0]["metric_id"] = "not-in-bank"
    _write_jsonl(dense_path, rows)
    with pytest.raises(ValueError, match="duplicate/out-of-bank metric"):
        build(args)


def test_authoritative_split_join_fills_missing_truth_splits(tmp_path: Path):
    args, _, bank_path, truth_path, _ = _fixture(tmp_path)
    truth = [json.loads(line) for line in truth_path.read_text().splitlines()]
    for row in truth:
        row.pop("split")
    _write_jsonl(truth_path, truth)
    bank_hash = json.loads(bank_path.read_text())["source_sha256"]
    assignments = [
        {
            "task": "humor",
            "norm_uid": "train-1",
            "source_group": "humor:jokes:source:source-train-1",
            "split": "train",
            "current_bank_source_sha256": bank_hash,
        },
        {
            "task": "humor",
            "norm_uid": "dev-1",
            "source_group": "humor:jokes:source:source-dev-1",
            "split": "dev",
            "current_bank_source_sha256": bank_hash,
        },
        {
            "task": "humor",
            "norm_uid": "outside-truth",
            "source_group": "humor:jokes:source:outside",
            "split": "test",
            "current_bank_source_sha256": bank_hash,
        },
    ]
    assignment_path = _write_jsonl(tmp_path / "splits.jsonl", assignments)
    args.split_assignments = str(assignment_path)

    rows, report = build(args)
    assert {row["norm_uid"]: row["split"] for row in rows} == {
        "train-1": "train",
        "dev-1": "dev",
    }
    assert report["split_assignment_audit"] == {
        "assignment_rows": 3,
        "assignment_rows_outside_truth": 1,
        "truth_rows_joined_to_assignments": 2,
    }


def test_fails_closed_when_candidate_union_misses_truth_uid(tmp_path: Path):
    args, _, _, _, dense_path = _fixture(tmp_path)
    rows = [json.loads(line) for line in dense_path.read_text().splitlines()]
    _write_jsonl(
        dense_path,
        [row for row in rows if row["norm_uid"] != "dev-1"],
    )
    with pytest.raises(ValueError, match="candidate lane union misses 1 truth UIDs"):
        build(args)
