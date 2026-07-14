from __future__ import annotations

import copy

import numpy as np
import pytest

from methods.metric_implementer.experiments.v14_panel_design import (
    _eligible_subset,
    build_panel_design,
    freeze_probe_split,
    identification_diagnostic,
    validate_panel_design,
    validate_probe_split,
)


def test_eligible_subset_is_hash_stable_and_balance_feasible():
    target = np.zeros(120, dtype=np.uint8)
    target[[3, 40, 117]] = 1
    kwargs = dict(
        teaching_indices=list(range(120)), run_sha="release", metric_key="metric",
        trial=0, attempt=0, fraction=0.4, panel_size=8, target=target,
    )
    first = _eligible_subset(**kwargs)
    assert first == _eligible_subset(**kwargs)
    assert len(first) == 48
    assert int(np.sum(target[first])) >= 3


def _fixture():
    rng = np.random.default_rng(1401)
    signatures = rng.integers(0, 2, size=(60, 300), dtype=np.uint8)
    probe_ids = [f"probe-{index:03d}" for index in range(300)]
    return signatures, probe_ids


def test_v14_split_and_panel_family_are_deterministic_balanced_unique_and_covering():
    signatures, probe_ids = _fixture()
    first_split = freeze_probe_split(probe_ids, run_sha="freeze", metric_key="metric")
    second_split = freeze_probe_split(probe_ids, run_sha="freeze", metric_key="metric")
    assert first_split == second_split
    validate_probe_split(first_split)
    assert [len(first_split[name]["indices"]) for name in (
        "teaching", "decoder_development", "heldout"
    )] == [120, 30, 150]

    first = build_panel_design(
        signatures, target_index=0,
        teaching_indices=first_split["teaching"]["indices"],
        run_sha="freeze", metric_key="metric", probe_ids=probe_ids,
    )
    second = build_panel_design(
        signatures, target_index=0,
        teaching_indices=first_split["teaching"]["indices"],
        run_sha="freeze", metric_key="metric", probe_ids=probe_ids,
    )
    assert first == second
    validate_panel_design(first)
    assert len(first["panels"]) == len({tuple(row["indices"]) for row in first["panels"]}) == 50
    assert all(3 <= row["target_yes"] <= 5 for row in first["panels"])
    assert min(first["probe_usage"].values()) >= 2
    assert {family: sum(row["decoder_family"] == family for row in first["panels"])
            for family in ("qwen", "llama", "mistral")} == {
                "qwen": 17, "llama": 17, "mistral": 16,
            }

    damaged = copy.deepcopy(first)
    damaged["probe_usage"][next(iter(damaged["probe_usage"]))] = 0
    with pytest.raises(ValueError, match="coverage"):
        validate_panel_design(damaged)


def test_identification_entropy_is_explicitly_not_a_behavioral_ceiling():
    signatures, _ = _fixture()
    result = identification_diagnostic(signatures, range(8), target_index=0)
    assert 0.0 <= result["identification_mi_bits"] <= 8.0
    assert result["scope"] == "identification_only_not_a_behavioral_value_ceiling"
