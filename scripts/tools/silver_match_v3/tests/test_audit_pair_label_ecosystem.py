from collections import Counter

from scripts.tools.silver_match_v3.audit_pair_label_ecosystem import heldout_related_graph


def test_heldout_related_graph_exposes_same_contradiction():
    aggregates = {
        "train": {
            (1, 2): Counter({1: 5}),
            (3, 4): Counter({1: 5, 0: 1}),
        },
        "eval": {(1, 2): Counter({1: 1, 2: 2})},
    }
    report = heldout_related_graph(aggregates, 5)
    assert report["selected_train_edges"] == 1
    assert report["selected_metric_pairs"] == [["a1", "a2"]]
    assert report["eval_label_counts"] == {
        "unrelated_0": 0,
        "related_1": 1,
        "same_2": 2,
    }
