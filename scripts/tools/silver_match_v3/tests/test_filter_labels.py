from scripts.tools.silver_match_v3.filter_labels import filter_rows


def test_filter_rows_supports_exact_field_predicates():
    rows = [
        {"task": "x", "split": "train", "label_source": "human"},
        {"task": "x", "split": "train", "label_source": "weak"},
        {"task": "y", "split": "train", "label_source": "human"},
    ]
    assert filter_rows(
        rows,
        task="x",
        split="train",
        where={"label_source": "human"},
    ) == [rows[0]]
