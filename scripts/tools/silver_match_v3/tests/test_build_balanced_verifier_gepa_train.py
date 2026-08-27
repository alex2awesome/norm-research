import json

from scripts.tools.silver_match_v3.build_balanced_verifier_gepa_train import main


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_balances_gold_and_negative_proposals(tmp_path, monkeypatch):
    truth, primary, candidates = [], [], []
    for index in range(6):
        uid = f"u{index}"
        truth.append({"norm_uid": uid, "decision": "MATCH", "metric_id": "a1"})
        primary.append({"norm_uid": uid, "decision": "MATCH", "metric_id": "a1" if index < 5 else "a2"})
        candidates.append({"norm_uid": uid, "candidates": [{"metric_id": "a1"}, {"metric_id": "a2"}]})
    for name, values in (("truth", truth), ("primary", primary), ("candidates", candidates)):
        _write(tmp_path / f"{name}.jsonl", values)
    output = tmp_path / "balanced"
    monkeypatch.setattr(
        "sys.argv",
        ["build", "--truth", str(tmp_path / "truth.jsonl"), "--primary", str(tmp_path / "primary.jsonl"), "--candidates", str(tmp_path / "candidates.jsonl"), "--output-root", str(output)],
    )
    main()
    targets = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert sum(row["target"] == "CONFIRM_MATCH" for row in targets) == 3
    assert sum(row["target"] == "REJECT" for row in targets) == 3
