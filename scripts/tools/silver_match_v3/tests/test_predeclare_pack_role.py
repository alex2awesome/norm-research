from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.predeclare_pack_role import main


def test_predeclare_blind_audit_role_is_immutable_and_preserves_upstream_split(
    tmp_path: Path, monkeypatch
) -> None:
    pack = tmp_path / "pack"
    items = [
        {
            "norm_uid": f"{index:064x}",
            "task": "math-stackexchange",
            "split": "train",
            "split_group": f"post:{index}",
        }
        for index in range(3)
    ]
    items_path = pack / "items.jsonl"
    write_jsonl(items_path, items)
    (pack / "validation.json").write_text(
        json.dumps(
            {
                "task": "math-stackexchange",
                "outputs": {"items": {"sha256": sha256_file(items_path)}},
            }
        )
    )
    output = tmp_path / "roles.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predeclare_pack_role",
            "--pack-root",
            str(pack),
            "--role",
            "blind_audit",
            "--output",
            str(output),
        ],
    )
    main()
    rows = list(read_jsonl(output))
    assert all(row["predeclared_split"] == "train" for row in rows)
    assert all(row["teacher_partition_role"] == "blind_audit" for row in rows)
    assert all(row["audit_permanently_excluded_from_gradients"] for row in rows)
    assert json.loads(output.with_suffix(".jsonl.meta.json").read_text())[
        "permanently_excluded_from_gradients"
    ]
