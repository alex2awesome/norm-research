from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import write_jsonl
from scripts.tools.silver_match_v3.rechunk_independent_raw_labels import main


def test_rechunks_exact_uid_union_in_target_order(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "part-005.jsonl"
    write_jsonl(target, [{"norm_uid": "u2"}, {"norm_uid": "u1"}, {"norm_uid": "u3"}])
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(
        json.dumps(
            {"task": "humor", "chunk_id": "part-000", "labels": [{"norm_uid": "u1"}]}
        )
    )
    second.write_text(
        json.dumps(
            {
                "task": "humor",
                "chunk_id": "part-001",
                "labels": [{"norm_uid": "u3"}, {"norm_uid": "u2"}],
            }
        )
    )
    output = tmp_path / "raw" / "part-005.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rechunk_independent_raw_labels",
            "--target-chunk",
            str(target),
            "--input",
            str(first),
            "--input",
            str(second),
            "--task",
            "humor",
            "--output",
            str(output),
        ],
    )
    main()
    payload = json.loads(output.read_text())
    assert payload["chunk_id"] == "part-005"
    assert [row["norm_uid"] for row in payload["labels"]] == ["u2", "u1", "u3"]
    report = json.loads(output.with_suffix(".json.meta.json").read_text())
    assert report["exact_uid_coverage"] is True
    assert report["label_rows_unmodified"] is True
