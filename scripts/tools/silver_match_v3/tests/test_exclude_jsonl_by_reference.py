import json

from scripts.tools.silver_match_v3.exclude_jsonl_by_reference import main


def test_excludes_entire_source_group(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    exclude = tmp_path / "exclude.jsonl"
    output = tmp_path / "out.jsonl"
    source.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"norm_uid": "a", "source_group": "g1"},
                {"norm_uid": "b", "source_group": "g1"},
                {"norm_uid": "c", "source_group": "g2"},
            ]
        )
    )
    exclude.write_text(json.dumps({"norm_uid": "a", "source_group": "g1"}) + "\n")
    monkeypatch.setattr(
        "sys.argv",
        ["exclude", "--input", str(source), "--exclude", str(exclude), "--output", str(output)],
    )
    main()
    assert [json.loads(line)["norm_uid"] for line in output.read_text().splitlines()] == ["c"]
