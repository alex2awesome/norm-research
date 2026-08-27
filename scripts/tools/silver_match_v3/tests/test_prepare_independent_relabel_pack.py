import json
from pathlib import Path

from scripts.tools.silver_match_v3.prepare_independent_relabel_pack import _key


def test_deterministic_relabel_order_key() -> None:
    values = ["b", "a", "c"]
    assert sorted(values, key=lambda value: _key(37, "item", value)) == sorted(
        values, key=lambda value: _key(37, "item", value)
    )
    assert {_key(37, "item", value) for value in values}.isdisjoint(
        {_key(37, "metric", value) for value in values}
    )
