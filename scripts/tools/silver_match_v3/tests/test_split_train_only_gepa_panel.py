from scripts.tools.silver_match_v3.split_train_only_gepa_panel import split_panel


def test_split_panel_is_group_safe_and_train_only():
    norms = {}
    labels = []
    # Find enough deterministic canonical-train groups for both local roles.
    from scripts.tools.silver_match_v3.make_calibration import split_for

    i = 0
    while len(labels) < 120:
        group = f"source-{i}"
        i += 1
        if split_for(f"c:source:{group}") != "train":
            continue
        uid = f"u-{i}"
        norms[uid] = {
            "norm_uid": uid,
            "source_id": group,
            "corpus": "c",
            "task": "code-review",
        }
        labels.append(
            {
                "norm_uid": uid,
                "task": "code-review",
                "decision": "MATCH",
                "metric_id": "a0",
            }
        )
    rows, report = split_panel(
        labels, norms, task="code-review", seed=7, dev_percent=25
    )
    assert {row["split"] for row in rows} == {"train", "dev"}
    assert all(row["predeclared_split"] == "train" for row in rows)
    assert report["source_group_overlap"] == 0


def test_split_panel_rejects_nontrain_upstream_group():
    from scripts.tools.silver_match_v3.make_calibration import split_for

    i = 0
    while split_for(f"c:source:bad-{i}") == "train":
        i += 1
    group = f"bad-{i}"
    labels = [{"norm_uid": "u", "task": "code-review"}]
    norms = {
        "u": {
            "norm_uid": "u",
            "source_id": group,
            "corpus": "c",
            "task": "code-review",
        }
    }
    try:
        split_panel(labels, norms, task="code-review", seed=7, dev_percent=25)
    except ValueError as exc:
        assert "non-train source group" in str(exc)
    else:
        raise AssertionError("expected non-train source-group rejection")


def test_split_panel_excludes_entire_source_group():
    from scripts.tools.silver_match_v3.make_calibration import split_for, split_group_for

    norms = {}
    labels = []
    i = 0
    while len({row["source_id"] for row in norms.values()}) < 120:
        source = f"source-{i}"
        i += 1
        if split_for(f"c:source:{source}") != "train":
            continue
        for suffix in ("a", "b"):
            uid = f"u-{i}-{suffix}"
            norms[uid] = {
                "norm_uid": uid,
                "source_id": source,
                "corpus": "c",
                "task": "code-review",
            }
            labels.append(
                {
                    "norm_uid": uid,
                    "task": "code-review",
                    "decision": "MATCH",
                    "metric_id": "a0",
                }
            )
    first_uid = labels[0]["norm_uid"]
    excluded = {split_group_for(norms[first_uid])}
    rows, report = split_panel(
        labels,
        norms,
        task="code-review",
        seed=7,
        dev_percent=25,
        excluded_groups=excluded,
    )
    assert all(split_group_for(norms[row["norm_uid"]]) not in excluded for row in rows)
    assert report["excluded_labeled_uids"] == 2
    assert report["selected_source_group_overlap_with_exclusions"] == 0


def test_split_panel_enforces_predeclared_minimum_support():
    from scripts.tools.silver_match_v3.make_calibration import split_for

    norms = {}
    labels = []
    i = 0
    while len(labels) < 30:
        source = f"source-{i}"
        i += 1
        if split_for(f"c:source:{source}") != "train":
            continue
        uid = f"u-{i}"
        norms[uid] = {
            "norm_uid": uid,
            "source_id": source,
            "corpus": "c",
            "task": "code-review",
        }
        labels.append({"norm_uid": uid, "task": "code-review"})
    try:
        split_panel(
            labels,
            norms,
            task="code-review",
            seed=7,
            dev_percent=25,
            minimum_train=100,
            minimum_dev=100,
        )
    except ValueError as exc:
        assert "minimum support" in str(exc)
    else:
        raise AssertionError("expected minimum-support rejection")
