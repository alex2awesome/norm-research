import json

from scripts.tools.silver_match_v3.score_three_order_verifier import main


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_all_three_exact_high(tmp_path, monkeypatch):
    truth = [
        {"norm_uid": "good", "decision": "MATCH", "metric_id": "a1"},
        {"norm_uid": "bad", "decision": "MATCH", "metric_id": "a2"},
    ]
    primary = [{"norm_uid": uid, "metric_id": "a1"} for uid in ("good", "bad")]
    confirmed = [
        {"norm_uid": uid, "decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high", "parse_error": None}
        for uid in ("good", "bad")
    ]
    reverse = [confirmed[0], {"norm_uid": "bad", "decision": "AMBIGUOUS_MATCH", "metric_id": None, "confidence": "high", "parse_error": None}]
    for name, rows in (("truth", truth), ("primary", primary), ("original", confirmed), ("hashed", confirmed), ("reverse", reverse)):
        _write(tmp_path / f"{name}.jsonl", rows)
    output = tmp_path / "score.json"
    monkeypatch.setattr("sys.argv", ["score", "--truth", str(tmp_path / "truth.jsonl"), "--primary", str(tmp_path / "primary.jsonl"), "--original", str(tmp_path / "original.jsonl"), "--hashed", str(tmp_path / "hashed.jsonl"), "--reverse", str(tmp_path / "reverse.jsonl"), "--output", str(output)])
    main()
    policy = json.loads(output.read_text())["policy"]
    assert policy["retained"] == policy["retained_true"] == 1
