import json

import pytest

from scripts.tools.silver_match_v3.build_abstention_rescue import (
    complementary_blocks,
    complementary_blocks_multi,
    repeated_blocks_multi,
)


def _candidates(n):
    return [
        {
            "metric_id": f"a{i}",
            "rank": i + 1,
            "dense_rank": n - i,
            "dense_statement_rank": i + 1,
            "word_rank": (i * 3) % n + 1,
            "word_statement_rank": (i * 5) % n + 1,
            "char_rank": (i * 7) % n + 1,
            "char_statement_rank": (i * 11) % n + 1,
        }
        for i in range(n)
    ]


def test_complementary_trials_exhaust_bank_without_duplicates():
    candidates = _candidates(121)
    bank = {row["metric_id"] for row in candidates}
    primary = [f"a{i}" for i in range(21)]
    blocks = complementary_blocks(candidates, bank, primary, block_size=50)
    assert len(blocks) == 2
    seen = set(primary)
    assert [block["lane"] for block in blocks] == [
        "dense_statement_rank",
        "char_rank",
    ]
    for block in blocks:
        ids = {row["metric_id"] for row in block["candidates"]}
        assert not ids & seen
        assert block["coverage_before"] == len(seen)
        seen |= ids
        assert block["coverage_after"] == len(seen)
    assert seen == bank
    assert blocks[-1]["coverage_complete"] is True


def test_full_bank_input_is_required():
    candidates = _candidates(10)
    with pytest.raises(ValueError, match="full-bank candidates required"):
        complementary_blocks(candidates[:-1], {f"a{i}" for i in range(10)}, [], block_size=5)


def test_primary_ids_must_be_in_bank():
    candidates = _candidates(10)
    bank = {row["metric_id"] for row in candidates}
    with pytest.raises(ValueError, match="outside bank"):
        complementary_blocks(candidates, bank, ["not-a-metric"], block_size=5)


def test_multiple_systems_are_interleaved_and_still_exhaustive():
    first = _candidates(121)
    second = list(reversed(_candidates(121)))
    # Reassign component ranks after reversing to create genuinely different
    # retrieval order while preserving the same frozen metric universe.
    for rank, row in enumerate(second, 1):
        row["rank"] = rank
        row["dense_rank"] = rank
    bank = {row["metric_id"] for row in first}
    blocks = complementary_blocks_multi(
        [("bge", first), ("adapter", second)],
        bank,
        ["a0"],
        block_size=50,
    )
    assert len(blocks) == 3
    assert blocks[0]["lane"] == "bge:rank"
    assert blocks[1]["lane"] == "adapter:rank"
    seen = {"a0"}
    for block in blocks:
        ids = {row["metric_id"] for row in block["candidates"]}
        assert not ids & seen
        seen |= ids
        assert all("retrieval_system" in row for row in block["candidates"])
    assert seen == bank


def test_repeated_capture_reincludes_primary_and_repartitions_bank():
    first = _candidates(88)
    second = list(reversed(_candidates(88)))
    for rank, row in enumerate(second, 1):
        row["rank"] = rank
        row["dense_rank"] = rank
    bank = {row["metric_id"] for row in first}
    blocks = repeated_blocks_multi(
        [("bge", first), ("adapter", second)],
        bank,
        [f"a{i}" for i in range(50)],
        block_size=50,
        coverage_repeats=2,
        reinclude_primary=True,
    )
    assert len(blocks) == 4
    assert [block["capture"] for block in blocks] == [0, 0, 1, 1]
    exposures = {metric_id: 0 for metric_id in bank}
    for block in blocks:
        for row in block["candidates"]:
            exposures[row["metric_id"]] += 1
    assert set(exposures.values()) == {2}
    assert blocks[0]["lane"] != blocks[2]["lane"]
