"""Frozen-codebook Reconstruction-MCQ value bridge tests."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from methods.metric_implementer.experiments.cr3_reconstruction_values import (
    CachedChoiceReconstructor,
    build_codebook_panel_plan,
    build_frozen_codebook_manifest,
    evaluate_scored_prompt_values,
    load_value_artifact,
    score_codebook_panel_priors,
    select_prior_balanced_panels,
    validate_codebook_manifest,
    write_value_artifact,
)
from scripts.tools.cr3_mining_worker import stage_value


class _ConditionSensitiveSelector:
    def score_choices(self, prompts, choices, **kwargs):
        rows = []
        seeds = kwargs.get("seed")
        if not isinstance(seeds, (list, tuple, np.ndarray)):
            seeds = [seeds] * len(prompts)
        for prompt, seed in zip(prompts, seeds):
            target_position = next(
                int(number) - 1 for number, description in
                re.findall(r"(?m)^(\d+)\. (.+)$", prompt)
                if "target metric" in description
            )
            target_probability = 0.8 if int(seed) // 10_000 == 1 else 0.2
            row = np.full(len(choices), (1.0 - target_probability) / (len(choices) - 1))
            row[target_position] = target_probability
            rows.append(row.tolist())
        return rows


class _DriftingSelector:
    def __init__(self, target_probability):
        self.target_probability = float(target_probability)
        self.calls = 0

    def score_choices(self, prompts, choices, **_kwargs):
        self.calls += 1
        return [[self.target_probability, 1.0 - self.target_probability]
                for _ in prompts]


class _SemanticPriorSelector:
    def score_choices(self, prompts, choices, **_kwargs):
        rows = []
        for prompt in prompts:
            descriptions = [description for _, description in
                            re.findall(r"(?m)^(\d+)\. (.+)$", prompt)]
            weights = np.asarray([
                4.0 if "salient" in description else 1.0
                for description in descriptions
            ])
            rows.append((weights / weights.sum()).tolist())
        return rows


def test_choice_probability_cache_freezes_cross_process_numeric_drift(tmp_path):
    path = tmp_path / "choice.sqlite"
    first_backend = _DriftingSelector(0.7)
    first = CachedChoiceReconstructor(
        first_backend, path, model="model", revision="revision")
    observed = first.score_choices(["prompt"], ["1", "2"], seed=[4])
    assert first_backend.calls == 1

    second_backend = _DriftingSelector(0.6)
    second = CachedChoiceReconstructor(
        second_backend, path, model="model", revision="revision")
    replayed = second.score_choices(["prompt"], ["1", "2"], seed=[4])
    assert replayed == observed
    assert second_backend.calls == 0


def _write_bootstrap(root, key, target, description):
    directory = root / key / "bootstrap"
    directory.mkdir(parents=True)
    path = directory / "scored.npz"
    probes = [f"probe {i}" for i in range(len(target))]
    np.savez_compressed(
        path,
        sigs=np.asarray([target]),
        texts=np.asarray([f"seed prompt {key}"], object),
        target=np.asarray(target, float),
        metric_description=np.asarray(description),
        probe_texts=np.asarray(probes, object),
        probe_sha256=np.asarray("shared-probes"),
        executor_model=np.asarray("fixed-executor"),
        executor_model_revision=np.asarray("revision"),
        readout_id=np.asarray("hard-readout"),
    )
    return path


def _bootstraps(tmp_path):
    base = np.tile([0.0, 1.0], 10)
    vectors = []
    for offset in range(4):
        vector = base.copy()
        vector[np.arange(offset, len(base), 4)] = 1.0 - vector[np.arange(offset, len(base), 4)]
        vectors.append(vector)
    return [
        _write_bootstrap(tmp_path, f"metric_{index}", vector,
                         "target metric" if index == 0 else f"distractor metric {index}")
        for index, vector in enumerate(vectors)
    ]


def test_codebook_is_bootstrap_only_frozen_and_hash_validated(tmp_path):
    manifest = build_frozen_codebook_manifest(
        _bootstraps(tmp_path), n_options=4, design_size=20,
        min_design_disagreements=2, seed=4)
    validate_codebook_manifest(manifest)
    assert manifest["premises"]["built_from_bootstrap_only"] is True
    assert manifest["premises"]["uses_external_labels"] is False
    for entry in manifest["entries"].values():
        assert 0.0 <= entry["target_design_yes_rate"] <= 1.0
        if entry["valid"]:
            assert entry["selected_distractor_kappa_min"] == min(
                row["kappa"] for row in entry["distractor_design_statistics"])
            assert entry["selected_distractor_disagreements_min"] >= 1
    assert all(entry["valid"] for entry in manifest["entries"].values())
    assert len(manifest["entries"]["metric_0"]["distractor_metric_keys"]) == 3

    mutated = dict(manifest)
    mutated["design_seed"] = 99
    with pytest.raises(ValueError, match="mutated"):
        validate_codebook_manifest(mutated)


def test_prior_calibration_selects_a_balanced_menu_before_prompt_search(tmp_path):
    base = np.tile([0.0, 1.0], 20)
    descriptions = [
        "salient target",
        "salient neighbor one",
        "salient neighbor two",
        "salient neighbor three",
        "obscure neighbor four",
        "obscure neighbor five",
        "obscure neighbor six",
    ]
    paths = []
    for index, description in enumerate(descriptions):
        vector = base.copy()
        vector[np.arange(index, len(base), 11)] = 1.0 - vector[
            np.arange(index, len(base), 11)]
        paths.append(_write_bootstrap(tmp_path, f"metric_{index}", vector, description))

    plan = build_codebook_panel_plan(
        paths,
        target_metric_keys=["metric_0"],
        n_options=4,
        design_size=40,
        min_design_disagreements=2,
        candidate_pool_size=6,
        max_panels_per_target=20,
        seed=9,
    )
    calibration = score_codebook_panel_priors(
        _SemanticPriorSelector(),
        panel_plan=plan,
        noun="story",
        n_draws=4,
        reconstructor_model="semantic-selector",
        reconstructor_revision="revision",
    )
    selections = select_prior_balanced_panels(
        plan,
        calibration,
        maximum_option_probability=0.35,
        target_probability_tolerance=0.10,
        minimum_normalized_entropy=0.90,
    )
    selected = selections["metric_0"]
    assert selected["prior_calibration"]["passes_prior_balance"] is True
    selected_prior = selected["prior_calibration"]["prior"]
    assert selected_prior["maximum_option_probability"] <= 0.35
    assert abs(selected_prior["target_probability"] - 0.25) <= 0.10
    assert selected_prior["normalized_entropy"] >= 0.90

    passing_rows = [
        row for row in calibration["rows"]["metric_0"]
        if row["panel_id"] == selected["prior_calibration"]["panel_id"]
    ]
    assert len(passing_rows) == 1

    manifest = build_frozen_codebook_manifest(
        paths,
        n_options=4,
        design_size=40,
        min_design_disagreements=2,
        seed=9,
        panel_selections=selections,
    )
    validate_codebook_manifest(manifest)
    entry = manifest["entries"]["metric_0"]
    assert entry["selection_method"].startswith("blind_no_demo")
    assert entry["prior_calibration"]["passes_prior_balance"] is True


def test_every_scored_row_receives_an_anchor_free_mcq_value(tmp_path):
    paths = _bootstraps(tmp_path)
    manifest = build_frozen_codebook_manifest(
        paths, n_options=4, design_size=20, min_design_disagreements=2, seed=4)
    target = np.load(paths[0], allow_pickle=True)["target"]
    scored = tmp_path / "audit_scored.npz"
    rows = np.vstack([target, 1.0 - target, target])
    np.savez_compressed(
        scored,
        sigs=rows,
        texts=np.asarray(["prompt a", "prompt b", "prompt a"], object),
        families=np.asarray(["f1", "f2", "f1"], object),
        probe_sha256=np.asarray("shared-probes"),
    )
    payload = evaluate_scored_prompt_values(
        _ConditionSensitiveSelector(),
        codebook_manifest=manifest,
        target_metric_key="metric_0",
        scored_path=scored,
        noun="story",
        n_examples=8,
        n_reconstruction_draws=4,
        choice_readout="logits",
        choice_probabilities_content_cached=True,
    )
    assert payload["n_rows"] == 3
    assert payload["values"].tolist() == pytest.approx([0.6, 0.6, 0.6])
    assert payload["raw_target_option_probability"].tolist() == pytest.approx([0.8, 0.8, 0.8])
    assert payload["premises"]["every_scored_row_valued"] is True
    assert payload["premises"]["uses_external_labels"] is False
    assert payload["premises"]["value_determined_by_exact_behavior"] is True
    assert payload["no_demonstration_target_probability"] == pytest.approx(0.2)
    assert payload["value_cap"] == pytest.approx(0.8)
    assert payload["batched_logit_path"] is True

    artifact = tmp_path / "values.npz"
    write_value_artifact(
        artifact, payload, reconstructor_model="fixed-reconstructor",
        reconstructor_revision="revision")
    loaded = load_value_artifact(
        artifact,
        expected_source_scored_sha256=payload["source_scored_sha256"],
        expected_codebook_manifest_sha256=manifest["manifest_sha256"],
    )
    assert loaded["values"].tolist() == pytest.approx([0.6, 0.6, 0.6])
    assert loaded["reconstructor_model"] == "fixed-reconstructor"
    assert loaded["value_cap"] == pytest.approx(0.8)


def test_value_worker_writes_every_row_with_one_resident_fake_backend(tmp_path):
    paths = _bootstraps(tmp_path)
    manifest = build_frozen_codebook_manifest(
        paths, n_options=4, design_size=20, min_design_disagreements=2, seed=4)
    manifest_path = tmp_path / "codebook.json"
    manifest_path.write_text(json.dumps(manifest))
    target = np.load(paths[0], allow_pickle=True)["target"]
    scored = tmp_path / "worker_scored.npz"
    np.savez_compressed(
        scored,
        sigs=np.vstack([target, 1.0 - target]),
        texts=np.asarray(["prompt a", "prompt b"], object),
        families=np.asarray(["f1", "f2"], object),
        probe_sha256=np.asarray("shared-probes"),
    )
    output = tmp_path / "worker_values.npz"
    jobs = tmp_path / "value_jobs.json"
    jobs.write_text(json.dumps([{
        "codebook_manifest": str(manifest_path),
        "target_metric_key": "metric_0",
        "scored": str(scored),
        "noun": "story",
        "n_examples": 8,
        "n_reconstruction_draws": 4,
        "choice_readout": "auto",
        "out": str(output),
    }]))
    stage_value(SimpleNamespace(
        jobs=str(jobs), model="fake-reconstructor", fake=True))
    loaded = load_value_artifact(
        output,
        expected_source_scored_sha256=None,
        expected_codebook_manifest_sha256=manifest["manifest_sha256"],
    )
    assert len(loaded["values"]) == 2
    assert np.all((loaded["values"] >= 0.0) & (loaded["values"] <= 1.0))
    assert loaded["premises"]["every_scored_row_valued"] is True
