"""Scope and fail-closed tests for auxiliary reconstruction replay bounds."""
from __future__ import annotations

import numpy as np
import pytest

from methods.metric_implementer.experiments.reconstruction_certificate import (
    mcq_reconstruction_certificate,
    reconstruction_global_certificate,
)


def test_mcq_replay_certificate_is_anchor_free_secondary_and_bounds_recovery():
    target = np.tile(np.asarray([0, 1], dtype=np.uint8), 1000)
    options = np.vstack([target, 1 - target, np.zeros_like(target), np.ones_like(target)])
    choices = np.zeros(4, dtype=int)
    assignment = np.arange(len(target), dtype=int) % len(choices)

    certificate = mcq_reconstruction_certificate(
        target,
        options,
        choices,
        assignment,
        option_ids=("target", "inverse", "never", "always"),
        codebook_frozen_before_candidate_optimization=True,
        choices_frozen_before_lockbox=True,
        assignments_iid_uniform_and_predeclared=True,
        lockbox_unused_for_optimization=True,
    )

    assert certificate["scope"]["production_cr3_role"].startswith(
        "secondary behavioral replay")
    assert certificate["empirical"]["recovery_bits"] == pytest.approx(1.0)
    assert certificate["certified"]["codebook_prompt_optimum_UCB_bits"] >= certificate[
        "certified"]["candidate_recovery_LCB_bits"]
    assert certificate["scope"]["explicitly_not_required"] == [
        "silver labels",
        "ground-truth labels",
        "human annotations",
        "archival outcomes",
    ]


def test_mcq_replay_certificate_fails_closed_when_lockbox_choice_is_not_frozen():
    target = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="choices_frozen_before_lockbox"):
        mcq_reconstruction_certificate(
            target,
            np.vstack([target, 1 - target]),
            np.zeros(2, dtype=int),
            np.asarray([0, 1, 0, 1], dtype=int),
            option_ids=("target", "inverse"),
            codebook_frozen_before_candidate_optimization=True,
            choices_frozen_before_lockbox=False,
            assignments_iid_uniform_and_predeclared=True,
            lockbox_unused_for_optimization=True,
        )


def test_free_reconstruction_replay_certificate_rejects_consumed_lockbox():
    target = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="unused for optimization"):
        reconstruction_global_certificate(
            target,
            target[:, None],
            prompt_frozen_before_lockbox=True,
            reconstructions_frozen_before_lockbox=True,
            lockbox_unused_for_optimization=False,
        )
