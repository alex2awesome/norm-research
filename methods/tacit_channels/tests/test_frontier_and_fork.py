"""CPU-safe tests: frontier extraction end-to-end on fixtures; teacher-forced fork guards."""
import json

import numpy as np
import pytest

from methods.tacit_channels.channels.eval.teacher_forced_lora import (
    check_upstream_drift, score_declared_binary_lora, upstream_source_sha256,
)
from methods.tacit_channels.channels.frontier_probe.extract_rescue_articulations import extract
from methods.tacit_channels.channels.gepa_bridge.fitness_reconstruction import (
    per_instance_scores,
)


def _bank(tmp_path, cell_id, construct, articulation):
    bank = {"schema": "t", "cells": [{
        "id": cell_id, "domain": "humor", "construct": construct,
        "arms": [
            {"id": "name", "channel": "sparse", "control_for": None,
             "forms": [{"id": "canonical", "prompt": construct}]},
            {"id": "source_definition", "channel": "declarative", "control_for": None,
             "forms": [{"id": "canonical",
                        "prompt": f"{construct}\n\n{articulation}"}]},
        ]}]}
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bank))
    return str(p)


def _grid(base, job, domain, rows):
    d = base / job
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"grid_{domain}_fix_rep0.npz",
        scores=np.vstack([v for _m, v in rows]),
        meta=np.array([json.dumps(m) for m, _v in rows], dtype=object))


def test_extract_finds_planted_rescue(tmp_path):
    rng = np.random.default_rng(3)
    cell = "TB::humor::planted"
    articulation = "The punchline must reverse the setup's frame in the final clause."
    bank_path = _bank(tmp_path, cell, "Frame reversal", articulation)

    base = tmp_path / "scores"
    fam = base / "family_scores_test"
    target = rng.normal(size=50)
    m = lambda arm, cf=None: {"cell_id": cell, "arm_id": arm, "form": "canonical",
                              "control_for": cf}
    _grid(fam, "test_70b_name_target", "humor", [(m("name"), target)])
    _grid(fam, "test_8b_executor", "humor", [
        (m("name"), target + rng.normal(scale=3.0, size=50)),
        (m("source_definition"), target + rng.normal(scale=0.2, size=50)),
    ])
    rows = extract(bank_path, str(base),
                   families={"test": ("family_scores_test", "test_70b_name_target")},
                   domains=("humor",))
    assert len(rows) == 1
    row = rows[0]
    assert row["rescued"] is True
    assert row["best_arm"] == "source_definition"
    assert articulation in row["articulation_text"]
    assert row["construct_name_text"] == "Frame reversal"


def test_fork_drift_guard():
    sha = upstream_source_sha256()
    assert len(sha) == 64
    check_upstream_drift(sha)  # current source passes
    with pytest.raises(RuntimeError, match="changed upstream"):
        check_upstream_drift("0" * 64)


def test_fork_fake_backend_branch():
    class FakeVLLM:  # class NAME is the dispatch contract (mirrors frozen impl)
        def score_binary(self, prompts, pos="YES", neg="NO", seed=0):
            return [0.25] * len(prompts)

    out = score_declared_binary_lora(FakeVLLM(), ["a", "b", "c"])
    assert out.shape == (3,)
    assert np.allclose(out, 0.25)


def test_per_instance_scores_rank_equivalence():
    target = np.array([0.1, 0.2, 0.5, 0.9])
    perfect = per_instance_scores(np.array([0.0, 0.3, 0.6, 1.0]), target)
    reversed_ = per_instance_scores(np.array([1.0, 0.6, 0.3, 0.0]), target)
    assert np.mean(perfect) == pytest.approx(1.0)
    assert np.mean(perfect) > np.mean(reversed_)
