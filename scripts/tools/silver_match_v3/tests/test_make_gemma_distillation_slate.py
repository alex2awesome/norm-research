from scripts.tools.silver_match_v3.make_gemma_distillation_slate import select_balanced


def row(uid: str, corpus: str) -> dict:
    return {"norm_uid": uid, "corpus": corpus, "split_group": uid}


def test_select_balanced_round_robins_corpora_deterministically():
    values = {
        "a": [row("a1", "a"), row("a2", "a"), row("a3", "a")],
        "b": [row("b1", "b")],
    }
    selected = select_balanced(values, 3)
    assert len(selected) == 3
    assert sum(item["corpus"] == "b" for item in selected) == 1
    assert sum(item["corpus"] == "a" for item in selected) == 2
    assert select_balanced(values, 3) == selected
