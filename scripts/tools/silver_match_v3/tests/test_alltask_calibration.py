from scripts.tools.silver_match_v3.make_alltask_calibration import select_rows


def _row(uid, corpus, task, rank):
    return {
        "norm_uid": uid,
        "corpus": corpus,
        "task": task,
        "split_group": f"{corpus}:source:{rank}",
    }


def test_selection_covers_corpora_then_fills_task():
    rows = {
        "c1": [_row(f"a{i}", "c1", "task", i) for i in range(8)],
        "c2": [_row(f"b{i}", "c2", "task", i) for i in range(8)],
    }
    selected = select_rows(
        rows, {"c1": "task", "c2": "task"}, per_task=6, min_per_corpus=2
    )
    assert len(selected) == 6
    assert sum(row["corpus"] == "c1" for row in selected) >= 2
    assert sum(row["corpus"] == "c2" for row in selected) >= 2
    assert len({row["split_group"] for row in selected}) == 6
