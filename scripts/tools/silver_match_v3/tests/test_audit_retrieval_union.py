from scripts.tools.silver_match_v3.audit_retrieval_union import audit


WEIGHTS = {
    "dense_rank": 1,
    "dense_statement_rank": 0,
    "word_rank": 0,
    "word_statement_rank": 0,
    "char_rank": 0,
    "char_statement_rank": 0,
}


def slate(uid, order):
    candidates = []
    for index, metric_id in enumerate(order):
        candidates.append(
            {
                "metric_id": metric_id,
                "metric_index": int(metric_id[1:]),
                "dense_rank": index + 1,
                "dense_statement_rank": index + 1,
                "word_rank": index + 1,
                "word_statement_rank": index + 1,
                "char_rank": index + 1,
                "char_statement_rank": index + 1,
            }
        )
    return {"norm_uid": uid, "candidates": candidates}


def test_audit_reports_paired_union_and_unique_rescue():
    labels = [
        {"norm_uid": "u1", "metric_id": "m1", "corpus": "c"},
        {"norm_uid": "u2", "metric_id": "m2", "corpus": "c"},
    ]
    systems = {
        "a": {
            "weights": WEIGHTS,
            "rank_constant": 60,
            "candidates": {
                "u1": slate("u1", ["m1", "m0"]),
                "u2": slate("u2", ["m0", "m2"]),
            },
        },
        "b": {
            "weights": WEIGHTS,
            "rank_constant": 60,
            "candidates": {
                "u1": slate("u1", ["m0", "m1"]),
                "u2": slate("u2", ["m2", "m0"]),
            },
        },
    }
    report, items = audit(labels, systems, top_k=1)
    assert report["all"]["systems"]["a"]["recall"] == 0.5
    assert report["all"]["systems"]["b"]["recall"] == 0.5
    assert report["all"]["unions"]["a+b"]["recall"] == 1.0
    assert report["all"]["unique_rescues"] == {"a": 1, "b": 1}
    assert len(items) == 2
