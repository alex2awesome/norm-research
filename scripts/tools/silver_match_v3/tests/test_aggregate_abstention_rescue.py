import json

from scripts.tools.silver_match_v3.aggregate_abstention_rescue import aggregate_rescue
from scripts.tools.silver_match_v3.build_abstention_rescue import build_rescue


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exhaustive_trials_produce_contrastive_finalists(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"metrics": [{"metric_id": f"a{i}"} for i in range(5)]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "v3",
        "banks": {"t": {"path": str(bank), "source_sha256": "sha"}},
        "corpora": {},
    }))
    full = tmp_path / "full.jsonl"
    candidates = []
    for i in range(5):
        candidates.append({
            "metric_id": f"a{i}", "rank": i + 1, "dense_rank": i + 1,
            "dense_statement_rank": 5 - i, "word_rank": i + 1,
            "word_statement_rank": i + 1, "char_rank": i + 1,
            "char_statement_rank": i + 1,
        })
    _write(full, [{"norm_uid": "u", "task": "t", "corpus": "c", "row": 1, "bank_source_sha256": "sha", "candidates": candidates}])
    primary = tmp_path / "primary.jsonl"
    _write(primary, [{"norm_uid": "u", "task": "t", "corpus": "c", "decision": "NO_CANDIDATE_FITS", "confidence": "medium", "candidate_ids": ["a0", "a1"]}])
    rescue_root = tmp_path / "rescue"
    build_rescue(
        manifest_path=manifest,
        candidate_paths=[full],
        primary_paths=[primary],
        output_root=rescue_root,
        block_size=2,
        primary_k=2,
        eligible_decisions={"NO_CANDIDATE_FITS"},
        include_all_abstentions=False,
        include_low_confidence=True,
    )
    adjudications = []
    for trial_path in sorted(rescue_root.glob("trial-*.jsonl")):
        trial = json.loads(trial_path.read_text().splitlines()[0])
        ids = [row["metric_id"] for row in trial["candidates"]]
        out = tmp_path / f"out-{trial['rescue_trial']}.jsonl"
        decision = "MATCH" if "a2" in ids else "NO_CANDIDATE_FITS"
        _write(out, [{
            "norm_uid": "u", "task": "t", "corpus": "c",
            "rescue_trial": trial["rescue_trial"],
            "decision": decision, "metric_id": "a2" if decision == "MATCH" else None,
            "confidence": "high", "reason": "audit", "candidate_ids": ids,
            "candidate_bank_source_sha256": "sha",
        }])
        adjudications.append(out)
    aggregate_root = tmp_path / "aggregate"
    report = aggregate_rescue(
        manifest_path=manifest,
        rescue_manifest_path=rescue_root / "rescue_manifest.json",
        primary_paths=[primary],
        adjudication_paths=adjudications,
        output_root=aggregate_root,
        max_finalists=4,
    )
    assert report["expected_trial_rows"] == 2
    assert report["status_counts"] == {"MATCH_FINALISTS": 1}
    finalist = json.loads((aggregate_root / "match_finalists.jsonl").read_text())
    assert finalist["rescue_exhaustive"] is True
    assert finalist["rescue_bank_count"] == 5
    assert finalist["rescue_proposed_metric_ids"] == ["a2"]
    assert finalist["candidates"][0]["metric_id"] == "a2"


def test_repeated_full_bank_capture_validates_every_metric_twice(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"metrics": [{"metric_id": f"a{i}"} for i in range(3)]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "v3",
        "banks": {"t": {"path": str(bank), "source_sha256": "sha"}},
        "corpora": {},
    }))
    candidates = [
        {
            "metric_id": f"a{i}", "rank": i + 1, "dense_rank": i + 1,
            "dense_statement_rank": 3 - i, "word_rank": i + 1,
            "word_statement_rank": i + 1, "char_rank": i + 1,
            "char_statement_rank": i + 1,
        }
        for i in range(3)
    ]
    full = tmp_path / "full.jsonl"
    _write(full, [{
        "norm_uid": "u", "task": "t", "corpus": "c", "row": 1,
        "bank_source_sha256": "sha", "candidates": candidates,
    }])
    primary = tmp_path / "primary.jsonl"
    _write(primary, [{
        "norm_uid": "u", "task": "t", "corpus": "c",
        "decision": "NO_CANDIDATE_FITS", "confidence": "medium",
        "candidate_ids": ["a0", "a1"],
    }])
    rescue_root = tmp_path / "rescue"
    build_rescue(
        manifest_path=manifest,
        candidate_paths=[full],
        primary_paths=[primary],
        output_root=rescue_root,
        block_size=2,
        primary_k=2,
        eligible_decisions={"NO_CANDIDATE_FITS"},
        include_all_abstentions=False,
        include_low_confidence=True,
        coverage_repeats=2,
        reinclude_primary=True,
    )
    adjudications = []
    for trial_path in sorted(rescue_root.glob("trial-*.jsonl")):
        trial = json.loads(trial_path.read_text())
        ids = [row["metric_id"] for row in trial["candidates"]]
        out = tmp_path / f"out-{trial['rescue_trial']}.jsonl"
        _write(out, [{
            "norm_uid": "u", "task": "t", "corpus": "c",
            "rescue_trial": trial["rescue_trial"],
            "decision": "NO_CANDIDATE_FITS", "metric_id": None,
            "confidence": "high", "reason": "none", "candidate_ids": ids,
            "candidate_bank_source_sha256": "sha",
        }])
        adjudications.append(out)
    aggregate_root = tmp_path / "aggregate"
    report = aggregate_rescue(
        manifest_path=manifest,
        rescue_manifest_path=rescue_root / "rescue_manifest.json",
        primary_paths=[primary],
        adjudication_paths=adjudications,
        output_root=aggregate_root,
    )
    assert report["expected_trial_rows"] == 4
    assert report["coverage_repeats"] == 2
    assert report["proposal_capture_pattern_counts"] == {"00": 1}
    assert report["capture_recapture_diagnostic"]["observed_union"] == 0
    row = json.loads((aggregate_root / "no_match_provisional.jsonl").read_text())
    assert row["rescue_coverage_repeats"] == 2
    assert row["rescue_reincludes_primary"] is True
