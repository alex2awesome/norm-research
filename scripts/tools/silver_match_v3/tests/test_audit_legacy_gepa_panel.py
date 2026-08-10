import json

from scripts.tools.silver_match_v3.audit_legacy_gepa_panel import audit


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_recomputes_canonical_split_and_rejects_relabelled_nontrain(tmp_path):
    # Find deterministic source IDs for distinct canonical roles without
    # coupling this test to hard-coded hash outputs.
    from scripts.tools.silver_match_v3.make_calibration import split_for

    source_ids = {}
    value = 0
    while set(source_ids) != {"train", "dev", "test"}:
        candidate = f"c:source:s{value}"
        source_ids.setdefault(split_for(candidate), f"s{value}")
        value += 1
    norms = tmp_path / "norms.jsonl"
    rows = [
        {
            "norm_uid": role,
            "task": "task",
            "corpus": "c",
            "source_id": source_id,
        }
        for role, source_id in source_ids.items()
    ]
    _jsonl(norms, rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"corpora": {"c": {"task": "task", "path": norms.name}}})
    )
    panel = tmp_path / "panel.jsonl"
    _jsonl(
        panel,
        [
            {
                "norm_uid": row["norm_uid"],
                "task": "task",
                "split": "train" if index < 2 else "dev",
                "predeclared_split": "train",
            }
            for index, row in enumerate(rows)
        ],
    )
    strong = tmp_path / "strong.jsonl"
    _jsonl(
        strong,
        [
            {
                "norm_uid": row["norm_uid"],
                "task": "task",
                "supervision_strength": "strong",
                "split": row["norm_uid"],
            }
            for row in rows
        ],
    )
    result = audit(
        manifest_path=manifest,
        task="task",
        panel_path=panel,
        strong_labels_path=strong,
    )
    assert result["status"] == "INVALID_FOR_TRAIN_ONLY_GEPA"
    assert result["authoritative_nontrain_rows_in_panel"] == 2
    assert result["false_predeclared_train_rows"] == 2
    assert result["strong_label_universe"]["panel_uid_set_equals_strong_uid_set"]
    assert not result["scientific_contract"]["may_reuse_panel_for_prompt_selection"]


def test_accepts_genuinely_train_only_panel(tmp_path):
    from scripts.tools.silver_match_v3.make_calibration import split_for

    source_ids = []
    value = 0
    while len(source_ids) < 2:
        source_id = f"s{value}"
        if split_for(f"c:source:{source_id}") == "train":
            source_ids.append(source_id)
        value += 1
    norms = tmp_path / "norms.jsonl"
    rows = [
        {"norm_uid": f"u{i}", "task": "task", "corpus": "c", "source_id": source_id}
        for i, source_id in enumerate(source_ids)
    ]
    _jsonl(norms, rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"corpora": {"c": {"task": "task", "path": norms.name}}})
    )
    panel = tmp_path / "panel.jsonl"
    _jsonl(
        panel,
        [
            {
                "norm_uid": row["norm_uid"],
                "task": "task",
                "split": "dev",
                "predeclared_split": "train",
            }
            for row in rows
        ],
    )
    result = audit(
        manifest_path=manifest,
        task="task",
        panel_path=panel,
        strong_labels_path=None,
    )
    assert result["status"] == "VALID_TRAIN_ONLY_GEPA"
    assert result["authoritative_upstream_split"] == {"train": 2}
