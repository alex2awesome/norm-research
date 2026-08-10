import json

from scripts.tools.silver_match_v3.freeze_verifier_gepa_split import main


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_freezes_stratified_disjoint_split(tmp_path, monkeypatch):
    truth = []
    primary = []
    candidates = []
    for index in range(10):
        uid = f"u{index}"
        truth.append(
            {
                "norm_uid": uid,
                "decision": "MATCH" if index < 8 else "NO_CANDIDATE_FITS",
                "metric_id": "a1" if index < 8 else None,
            }
        )
        primary.append({"norm_uid": uid, "metric_id": "a1" if index % 2 else "a2"})
        candidates.append({"norm_uid": uid, "candidates": []})
    for name, rows in (("truth", truth), ("primary", primary), ("candidates", candidates)):
        _write(tmp_path / f"{name}.jsonl", rows)
    output = tmp_path / "split"
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--truth", str(tmp_path / "truth.jsonl"),
            "--primary", str(tmp_path / "primary.jsonl"),
            "--candidates", str(tmp_path / "candidates.jsonl"),
            "--output-root", str(output),
        ],
    )
    main()
    optimize = {json.loads(line)["norm_uid"] for line in (output / "optimize/truth.jsonl").read_text().splitlines()}
    select = {json.loads(line)["norm_uid"] for line in (output / "select/truth.jsonl").read_text().splitlines()}
    assert optimize.isdisjoint(select)
    assert optimize | select == {row["norm_uid"] for row in truth}
    assert (output / "FREEZE.json").exists()
