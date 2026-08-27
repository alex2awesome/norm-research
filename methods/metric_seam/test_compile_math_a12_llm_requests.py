from __future__ import annotations

from methods.metric_seam.compile_math_a12_llm_requests import compile_bundle


def test_compiler_emits_two_split_bound_requests_per_structural_pair() -> None:
    requests = compile_bundle(
        rows=[{"item_key": "train_0001", "ctext": "Answer:\n$$x=x=2x$$"}],
        model="pinned-sonnet",
    )
    assert len(requests) == 4
    assert {row["pass_index"] for row in requests} == {1, 2}
    assert {row["split"] for row in requests} == {"compiler_train"}
    assert all(row["model"] == "pinned-sonnet" for row in requests)
    assert all("verdict" not in row["user_prompt"].lower() for row in requests)


def test_single_pass_bundle_reuses_the_identical_pass_one_requests() -> None:
    rows = [{"item_key": "train_0001", "ctext": "Answer:\n$$x=x=2x$$"}]
    both = compile_bundle(rows=rows, model="pinned-sonnet")
    one = compile_bundle(rows=rows, model="pinned-sonnet", pass_indices=(1,))
    assert one == [request for request in both if request["pass_index"] == 1]
