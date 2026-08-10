from scripts.tools.silver_match_v3.prepare_independent_teacher_pack import balanced_sample


def test_balanced_sample_round_robins_corpora():
    rows = [
        {"norm_uid": f"a{i}", "corpus": "a"} for i in range(5)
    ] + [{"norm_uid": f"b{i}", "corpus": "b"} for i in range(5)]
    selected = balanced_sample(rows, 6)
    assert sum(row["corpus"] == "a" for row in selected) == 3
    assert sum(row["corpus"] == "b" for row in selected) == 3
    assert balanced_sample(rows, 6) == selected
