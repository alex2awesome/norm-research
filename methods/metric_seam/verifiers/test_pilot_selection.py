import json
from pathlib import Path

import pytest

from methods.metric_seam.verifiers.pilot_selection import build_selection


JOIN = Path("outputs/metric_seam_pilot/hierarchy_r123/code_review_cuf_llama8b_join_candidates_v1.json")


def test_real_selection_is_bounded_exact_and_deterministic():
    first = build_selection(join_path=JOIN)
    second = build_selection(join_path=JOIN)
    assert first == second
    assert [row["candidate_aspect_id"] for row in first["real_units"]] == [
        "a0", "a18", "a38", "a92"
    ]
    assert all(row["cuf"]["level"] == 1 for row in first["real_units"])
    assert not first["selection_policy"]["semantic_join_queue_used"]
    assert not first["selection_policy"]["heldout_items_or_outputs_loaded_by_builder"]
    assert first["selection_policy"]["selection_author_blinding_not_mechanically_established"]


def test_unadjudicated_join_cannot_enter_pilot():
    with pytest.raises(ValueError, match="auto-accepted"):
        build_selection(join_path=JOIN, aspects=["a15"])
