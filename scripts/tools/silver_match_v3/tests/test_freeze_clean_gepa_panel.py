import json

import pytest

from scripts.tools.silver_match_v3.freeze_clean_gepa_panel import main
from scripts.tools.silver_match_v3.make_calibration import split_for


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path):
    rows = []
    for corpus in ("left", "right"):
        for index in range(80):
            rows.append(
                {
                    "norm_uid": f"{corpus}-{index}",
                    "task": "task",
                    "corpus": corpus,
                    "source_id": f"source-{index}",
                }
            )
    paths = {}
    for corpus in ("left", "right"):
        path = tmp_path / f"{corpus}.jsonl"
        _write_jsonl(path, [row for row in rows if row["corpus"] == corpus])
        paths[corpus] = path
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpora": {
                    corpus: {"task": "task", "path": path.name}
                    for corpus, path in paths.items()
                }
            }
        )
    )
    train_rows = [
        row
        for row in rows
        if split_for(f"{row['corpus']}:source:{row['source_id']}") == "train"
    ]
    return manifest, train_rows


def test_freezes_identity_only_disjoint_panel(tmp_path, monkeypatch):
    manifest, train_rows = _fixture(tmp_path)
    excluded = train_rows[:4]
    exclusion = tmp_path / "exclude.jsonl"
    _write_jsonl(
        exclusion,
        [
            {
                "norm_uid": row["norm_uid"],
                "source_group": f"{row['corpus']}:source:{row['source_id']}",
            }
            for row in excluded
        ],
    )
    output = tmp_path / "frozen"
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--role",
            "select",
            "--count",
            "12",
            "--min-per-corpus",
            "3",
            "--exclude-panel",
            str(exclusion),
            "--output-root",
            str(output),
        ],
    )
    main()
    rows = [json.loads(line) for line in (output / "identities.jsonl").read_text().splitlines()]
    freeze = json.loads((output / "FREEZE.json").read_text())
    assert len(rows) == len({row["source_group"] for row in rows}) == 12
    assert all(set(row) == {
        "schema_version", "norm_uid", "task", "corpus", "source_group",
        "upstream_split", "gepa_role",
        "permanently_excluded_from_retriever_gradients",
        "permanently_excluded_from_mi_and_outcome_estimation",
    } for row in rows)
    assert set(freeze["selected_by_corpus"]) == {"left", "right"}
    assert min(freeze["selected_by_corpus"].values()) >= 3
    assert freeze["exclusion_union"]["selected_source_group_overlap"] == 0
    assert freeze["content_contract"]["metric_ids_read"] is False


def test_rejects_source_group_mismatch(tmp_path, monkeypatch):
    manifest, train_rows = _fixture(tmp_path)
    exclusion = tmp_path / "exclude.jsonl"
    _write_jsonl(
        exclusion,
        [{"norm_uid": train_rows[0]["norm_uid"], "source_group": "wrong"}],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--role",
            "select",
            "--count",
            "2",
            "--exclude-panel",
            str(exclusion),
            "--output-root",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="source_group mismatch"):
        main()


def test_minimum_applies_to_corpus_with_no_remaining_groups(tmp_path, monkeypatch):
    manifest, train_rows = _fixture(tmp_path)
    excluded = tmp_path / "exclude.jsonl"
    right = [row for row in train_rows if row["corpus"] == "right"]
    _write_jsonl(
        excluded,
        [
            {
                "norm_uid": row["norm_uid"],
                "source_group": f"right:source:{row['source_id']}",
            }
            for row in right
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--role",
            "select",
            "--count",
            "2",
            "--min-per-corpus",
            "1",
            "--exclude-panel",
            str(excluded),
            "--output-root",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="right has 0 eligible groups"):
        main()


def test_newline_uid_exclusion_is_group_aware(tmp_path, monkeypatch):
    manifest, train_rows = _fixture(tmp_path)
    excluded = tmp_path / "exclude.uids.txt"
    excluded.write_text(train_rows[0]["norm_uid"] + "\n")
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--role",
            "optimize",
            "--count",
            "3",
            "--exclude-uid-file",
            str(excluded),
            "--output-root",
            str(output),
        ],
    )
    main()
    freeze = json.loads((output / "FREEZE.json").read_text())
    entry = freeze["inputs"]["exclusions"][str(excluded.resolve())]
    assert entry["format"] == "newline_delimited_uids"
    assert entry["uids"] == entry["source_groups"] == 1


def test_authoritative_upstream_role_reference_overrides_calibration_hash(tmp_path, monkeypatch):
    manifest, _ = _fixture(tmp_path)
    canonical = []
    for corpus in ("left", "right"):
        canonical.extend(
            json.loads(line)
            for line in (tmp_path / f"{corpus}.jsonl").read_text().splitlines()
        )
    # Deliberately choose rows whose legacy calibration-hash role is not train;
    # the retriever/LoRA role reference is the authoritative split here.
    desired = [
        row
        for row in canonical
        if split_for(f"{row['corpus']}:source:{row['source_id']}") != "train"
    ][:4]
    eligible = tmp_path / "eligible.jsonl"
    roles = tmp_path / "roles.jsonl"
    _write_jsonl(
        eligible,
        [
            {
                "norm_uid": row["norm_uid"],
                "source_group": (
                    f"task\x1f{row['corpus']}\x1fsource\x1f{row['source_id']}"
                ),
            }
            for row in desired
        ],
    )
    _write_jsonl(
        roles,
        [
            {"norm_uid": row["norm_uid"], "task": "task", "split": "train"}
            for row in desired
        ],
    )
    excluded = tmp_path / "excluded.jsonl"
    fallback = next(row for row in canonical if row not in desired)
    _write_jsonl(excluded, [{"norm_uid": fallback["norm_uid"]}])
    output = tmp_path / "authoritative"
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--role",
            "select",
            "--count",
            "4",
            "--eligible-reference",
            str(eligible),
            "--upstream-role-reference",
            str(roles),
            "--exclude-panel",
            str(excluded),
            "--output-root",
            str(output),
        ],
    )
    main()
    rows = [json.loads(line) for line in (output / "identities.jsonl").read_text().splitlines()]
    freeze = json.loads((output / "FREEZE.json").read_text())
    assert {row["norm_uid"] for row in rows} == {row["norm_uid"] for row in desired}
    assert freeze["inputs"]["upstream_role_reference"]["authoritative"] is True
    assert freeze["inputs"]["upstream_role_reference"]["role_counts"] == {"train": 4}
    assert freeze["inputs"]["eligible_reference"][
        "legacy_namespaced_source_group_rows"
    ] == 4
