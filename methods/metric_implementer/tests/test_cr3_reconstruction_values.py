"""Frozen-codebook Reconstruction-MCQ value bridge tests."""
from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from methods.metric_implementer.experiments.cr3_reconstruction_values import (
    CachedChoiceReconstructor,
    TEACHING_LIBRARY_SIZE,
    build_teaching_finalist_lock,
    build_teaching_panel_library,
    build_task_centralness_reference_plan,
    build_finite_state_envelope,
    build_codebook_panel_plan,
    build_frozen_codebook_manifest,
    evaluate_scored_prompt_values,
    import_choice_probability_cache,
    load_finite_state_scored_artifact,
    load_value_artifact,
    lookup_scored_prompt_values,
    prior_balanced_panel_rows,
    score_codebook_panel_priors,
    score_task_centralness_reference,
    select_prior_balanced_panels,
    select_full24_teaching_instrument,
    select_state_capable_panels,
    teaching_panel_selection_from_library,
    validate_teaching_panel_library,
    validate_codebook_manifest,
    validate_finite_state_envelope,
    validate_task_centralness_calibration,
    write_finite_state_scored_artifact,
    write_value_artifact,
)
from methods.metric_implementer.experiments.run_cr3_mining_loop import (
    _frozen_prior_selection,
    _mcq_calibration_matches_order_design,
)
from methods.metric_implementer.recon_channel import (
    mcq_no_demo_choice_probabilities,
    mcq_no_demo_choice_probabilities_many,
    mcq_option_order_design,
)
from methods.metric_implementer.vllm_backend import (
    CHOICE_READOUT_ID,
    FAKE_CHOICE_READOUT_ID,
)
import scripts.tools.cr3_reconstruction_calibration_worker as centralness_worker
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
    choice_readout_id = CHOICE_READOUT_ID

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


class _CountingUniformSelector:
    choice_readout_id = CHOICE_READOUT_ID

    def __init__(self):
        self.calls = 0
        self.rows = 0

    def score_choices(self, prompts, choices, **_kwargs):
        self.calls += 1
        self.rows += len(prompts)
        return [[1.0 / len(choices)] * len(choices) for _ in prompts]


def test_batched_blind_menu_scoring_is_exactly_scalar_query_compatible():
    menus = [
        ["target metric", "distractor one", "distractor two", "distractor three"],
        ["salient target", "obscure one", "salient two", "obscure three"],
    ]
    scalar_backend = _SemanticPriorSelector()
    scalar = [
        mcq_no_demo_choice_probabilities(
            scalar_backend, noun="story", option_descriptions=menu, n_draws=4)
        for menu in menus
    ]
    batched = mcq_no_demo_choice_probabilities_many(
        _SemanticPriorSelector(), noun="story",
        option_description_batches=menus, n_draws=4)
    assert batched == scalar


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

    imported_path = tmp_path / "private_copy.sqlite"
    report = import_choice_probability_cache(path, imported_path)
    assert report["source_rows"] == 1
    assert report["writable_database_shared"] is False
    imported_backend = _DriftingSelector(0.4)
    imported = CachedChoiceReconstructor(
        imported_backend, imported_path, model="model", revision="revision")
    assert imported.score_choices(["prompt"], ["1", "2"], seed=[4]) == observed
    assert imported_backend.calls == 0
    assert imported.rows == {}


def test_v12_choice_cache_never_admits_a_v11_probability_row(tmp_path):
    path = tmp_path / "choice.sqlite"
    old_payload = {
        "schema": "cr3-choice-probability-cache-v1",
        "model": "model",
        "revision": "revision",
        "prompt": "prompt",
        "choices": ["1", "2"],
        "system": None,
        "seed": 4,
    }
    old_key = hashlib.sha256(json.dumps(
        old_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE choice_rows (cache_key TEXT PRIMARY KEY, "
        "probabilities_json TEXT NOT NULL) WITHOUT ROWID")
    connection.execute(
        "INSERT INTO choice_rows(cache_key, probabilities_json) VALUES (?, ?)",
        (old_key, "[0.99,0.01]"),
    )
    connection.commit()
    connection.close()

    backend = _DriftingSelector(0.6)
    current = CachedChoiceReconstructor(
        backend, path, model="model", revision="revision")
    observed = current.score_choices(["prompt"], ["1", "2"], seed=[4])
    assert np.allclose(observed, [[0.6, 0.4]])
    assert backend.calls == 1


def test_four_row_calibration_is_not_transplanted_but_its_cache_prefix_is_reused(
    tmp_path,
):
    paths = _bootstraps(tmp_path / "bootstraps")
    centralness_plan, centralness_calibration = _centralness_artifacts(paths)
    plan = build_codebook_panel_plan(
        paths,
        centralness_reference_plan=centralness_plan,
        centralness_calibration=centralness_calibration,
        target_metric_keys=["metric_0"],
        n_options=4,
        design_size=12,
        min_design_disagreements=2,
        candidate_pool_size=3,
        max_panels_per_target=1,
        seed=9,
    )
    legacy_prefix_sha = hashlib.sha256(json.dumps(
        plan["panels"]["metric_0"][:1], sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    assert legacy_prefix_sha == "ea00e6dcb52328a053b646b11924b4df866e076d18c4e6bcc64bd07ef17c9b15"
    old_backend = _CountingUniformSelector()
    old_cache_path = tmp_path / "old" / "choice.sqlite"
    old_cache = CachedChoiceReconstructor(
        old_backend, old_cache_path, model="model", revision="revision")
    old_calibration = score_codebook_panel_priors(
        old_cache,
        panel_plan=plan,
        noun="story",
        n_draws=4,
        reconstructor_model="model",
        reconstructor_revision="revision",
    )
    assert old_backend.rows == 4
    expected_design = mcq_option_order_design(4, 24)
    assert not _mcq_calibration_matches_order_design(
        old_calibration, expected_design)
    old_prior = old_calibration["rows"]["metric_0"][0]["prior"]
    old_cache.connection.close()

    new_cache_path = tmp_path / "new" / "choice.sqlite"
    report = import_choice_probability_cache(old_cache_path, new_cache_path)
    assert report["source_rows"] == 4
    new_backend = _CountingUniformSelector()
    new_cache = CachedChoiceReconstructor(
        new_backend, new_cache_path, model="model", revision="revision")
    new_calibration = score_codebook_panel_priors(
        new_cache,
        panel_plan=plan,
        noun="story",
        n_draws=24,
        reconstructor_model="model",
        reconstructor_revision="revision",
    )
    new_prior = new_calibration["rows"]["metric_0"][0]["prior"]
    assert new_backend.calls == 1
    assert new_backend.rows == 20
    assert new_prior["query_sha256"][:4] == old_prior["query_sha256"]
    assert new_prior["canonical_choice_probabilities"][:4] == (
        old_prior["canonical_choice_probabilities"])
    assert _mcq_calibration_matches_order_design(
        new_calibration, expected_design)
    new_cache.connection.close()


def _write_bootstrap(root, key, target, description, *, probe_texts=None):
    directory = root / key / "bootstrap"
    directory.mkdir(parents=True)
    path = directory / "scored.npz"
    probes = (list(probe_texts) if probe_texts is not None
              else [f"probe {i}" for i in range(len(target))])
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


def _teaching_bootstraps(tmp_path):
    n = 80
    indices = np.arange(n)
    probes = [
        (f"Scene {index // 2:02d}: {'kind' if index % 2 else 'rude'} reply "
         + "context " * (1 + (index // 8) % 4)).strip()
        for index in range(n)
    ]
    base = (indices % 2).astype(float)
    vectors = [
        base,
        np.where(indices % 5 == 0, 1.0 - base, base),
        np.where(indices % 7 < 2, 1.0 - base, base),
        np.where((indices // 3) % 4 == 0, 1.0 - base, base),
    ]
    return [
        _write_bootstrap(
            tmp_path, f"teaching_metric_{index}", vector,
            "target teaching metric" if index == 0 else f"teaching distractor {index}",
            probe_texts=probes,
        )
        for index, vector in enumerate(vectors)
    ]


def _centralness_artifacts(paths, selector=None):
    plan = build_task_centralness_reference_plan(paths, seed=19)
    calibration = score_task_centralness_reference(
        selector or _CountingUniformSelector(),
        reference_plan=plan,
        noun="story",
        reconstructor_model="centralness-selector",
        reconstructor_revision="revision",
    )
    return plan, calibration


def test_teaching_library_is_deterministic_unique_and_preserves_exact_baseline(tmp_path):
    paths = _teaching_bootstraps(tmp_path)
    kwargs = dict(
        n_options=4,
        design_size=24,
        min_design_disagreements=2,
        seed=31,
        reconstruction_noun="story",
    )
    forward = build_frozen_codebook_manifest(paths, **kwargs)
    reverse = build_frozen_codebook_manifest(list(reversed(paths)), **kwargs)
    assert forward == reverse
    target_key = "teaching_metric_0"
    library = build_teaching_panel_library(forward, target_metric_key=target_key)
    assert library == build_teaching_panel_library(reverse, target_metric_key=target_key)
    validate_teaching_panel_library(library, forward, target_metric_key=target_key)
    assert library["library_size"] == TEACHING_LIBRARY_SIZE
    assert library["panels"][0]["role"] == "baseline_exact"
    assert library["panels"][0]["fixed_teaching_indices"] == forward["entries"][
        target_key]["fixed_teaching_indices"]
    assert library["panels"][0]["fixed_teaching_target_scores"] == forward["entries"][
        target_key]["fixed_teaching_target_scores"]

    design = set(forward["design_indices"])
    panels = [tuple(panel["fixed_teaching_indices"]) for panel in library["panels"]]
    assert len(set(panels)) == TEACHING_LIBRARY_SIZE
    assert all(len(set(panel)) == 8 and design.isdisjoint(panel) for panel in panels)
    assert all(panel["design_statistics"]["target_is_balanced"]
               for panel in library["panels"])
    roles = {panel["role"] for panel in library["panels"]}
    assert {"baseline_exact", "matched_pair_surface", "matched_pair_behavior",
            "nuisance_balanced_separation", "pattern_diverse", "tfidf_diverse"} <= roles
    nuisance = next(panel for panel in library["panels"]
                    if panel["role"] == "nuisance_balanced_separation")
    assert nuisance["design_statistics"]["absolute_log_length_mean_difference"] < 0.15
    assert nuisance["design_statistics"]["minimum_distractor_separation"] >= 1

    selected = teaching_panel_selection_from_library(library, library_index=1)
    selection = {
        target_key: {
            "distractor_metric_keys": forward["entries"][target_key][
                "distractor_metric_keys"],
            "prior_calibration": forward["entries"][target_key]["prior_calibration"],
            "teaching_panel_selection": selected,
        },
    }
    variant = build_frozen_codebook_manifest(paths, panel_selections=selection, **kwargs)
    assert variant["entries"][target_key]["fixed_teaching_indices"] == library["panels"][1][
        "fixed_teaching_indices"]
    assert forward == build_frozen_codebook_manifest(paths, **kwargs)

    mutated = json.loads(json.dumps(library))
    mutated["panels"][1]["fixed_teaching_indices"][0] += 1
    with pytest.raises(ValueError, match="invalid or mutated"):
        validate_teaching_panel_library(mutated, forward, target_metric_key=target_key)


def _screen_candidate(index, *, live=False, canonical_value=0.0, upper=0.2):
    core = {
        "schema": "cr3-reconstruction-teaching-screen-candidate-v1",
        "target_metric_key": "target",
        "prior_panel_id": "menu",
        "library_sha256": "a" * 64,
        "library_index": index,
        "teaching_panel_sha256": f"{index + 1:064x}",
        "teaching_panel_selection": {"selection_sha256": f"{index + 2:064x}"},
        "codebook_manifest_sha256": "b" * 64,
        "instrument_sha256": f"{index + 10:064x}",
        "envelope_summary_sha256": f"{index + 20:064x}",
        "state_function_semantic_sha256": f"{index + 30:064x}",
        "screen_option_order_design": mcq_option_order_design(4, 4),
        "screen_finite_state_upper_bound": upper,
        "canonical_identification": {
            "canonical_state": 3,
            "canonical_value": canonical_value,
            "canonical_target_posterior": 0.4 if live else 0.2,
            "canonical_posterior_margin": 0.1 if live else -0.1,
            "positive_canonical_lift": canonical_value > 0.0,
            "unique_target_posterior_argmax": live,
            "canonical_live": live,
            "design_only_not_achieved_prompt_evidence": True,
        },
        "synthetic_envelope_live_diagnostic": False,
        "uses_candidate_prompt_behavior": False,
        "uses_external_labels": False,
    }
    return {**core, "screen_candidate_sha256": hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def _full_envelope(candidate, *, canonical_live, upper, synthetic_live):
    posterior = [0.4, 0.2, 0.2, 0.2] if canonical_live else [0.2, 0.3, 0.25, 0.25]
    return {
        "target_metric_key": "target",
        "instrument_sha256": candidate["instrument_sha256"],
        "summary_sha256": f"{candidate['library_index'] + 40:064x}",
        "state_function_semantic_sha256": f"{candidate['library_index'] + 50:064x}",
        "finite_state_upper_bound": upper,
        "option_order_design": mcq_option_order_design(4, 24),
        "operational_target_diagnostic": {
            "state": 3,
            "value": 0.1 if canonical_live else 0.0,
            "mean_annotation_option_posterior": posterior,
        },
        "state_envelope_capability": {
            "has_positive_unique_target_maximizer": synthetic_live,
        },
    }


def test_f2_lock_is_hash_bound_and_full24_canonical_gate_is_independent_of_synthetic_max():
    candidates = [_screen_candidate(index, upper=0.3 - index * 0.01)
                  for index in range(TEACHING_LIBRARY_SIZE)]
    lock = build_teaching_finalist_lock(candidates)
    assert lock == build_teaching_finalist_lock(list(reversed(candidates)))
    assert len(lock["finalists"]) == 2

    mutated_candidate = json.loads(json.dumps(candidates[0]))
    mutated_candidate["screen_finite_state_upper_bound"] = 0.99
    with pytest.raises(ValueError, match="invalid teaching screen candidate"):
        build_teaching_finalist_lock([mutated_candidate, *candidates[1:]])
    mutated_lock = json.loads(json.dumps(lock))
    mutated_lock["finalists"].reverse()
    with pytest.raises(ValueError, match="invalid or mutated teaching finalist lock"):
        select_full24_teaching_instrument(mutated_lock, {})

    finalists = lock["finalists"]
    all_dead = {
        row["screen_candidate_sha256"]: _full_envelope(
            row, canonical_live=False, upper=0.4, synthetic_live=True)
        for row in finalists
    }
    formal = select_full24_teaching_instrument(lock, all_dead)
    assert formal["passes_canonical_identification"] is False
    assert formal["reporting_status"] == "FORMAL_CERTIFICATE_ONLY"

    live_sha = finalists[1]["screen_candidate_sha256"]
    mixed = {
        finalists[0]["screen_candidate_sha256"]: _full_envelope(
            finalists[0], canonical_live=False, upper=0.8, synthetic_live=True),
        live_sha: _full_envelope(
            finalists[1], canonical_live=True, upper=0.2, synthetic_live=False),
    }
    selected = select_full24_teaching_instrument(lock, mixed)
    assert selected["passes_canonical_identification"] is True
    assert selected["chosen_screen_candidate_sha256"] == live_sha
    assert selected["chosen_synthetic_envelope_live_diagnostic"] is False


def test_four_order_teaching_screen_is_an_exact_cache_prefix_of_full24(tmp_path):
    paths = _teaching_bootstraps(tmp_path / "bank")
    manifest = build_frozen_codebook_manifest(
        paths, n_options=4, design_size=24, min_design_disagreements=2, seed=31,
        reconstruction_noun="story")
    target = np.load(paths[0], allow_pickle=True)["target"]
    scored = tmp_path / "two_candidates.npz"
    np.savez_compressed(
        scored,
        sigs=np.vstack([target, 1.0 - target]),
        texts=np.asarray(["candidate a", "candidate b"], object),
        probe_sha256=np.asarray("shared-probes"),
    )
    uniform4 = np.full((4, 4), 0.25)
    uniform24 = np.full((24, 4), 0.25)
    backend = _CountingUniformSelector()
    cached = CachedChoiceReconstructor(
        backend, tmp_path / "shared.sqlite", model="model", revision="revision")
    screen = evaluate_scored_prompt_values(
        cached, codebook_manifest=manifest, target_metric_key="teaching_metric_0",
        scored_path=scored, noun="story", n_examples=8, n_reconstruction_draws=4,
        choice_readout="logits", fixed_no_demo_canonical_probabilities=uniform4,
        choice_probabilities_content_cached=True)
    screen_rows = backend.rows
    full = evaluate_scored_prompt_values(
        cached, codebook_manifest=manifest, target_metric_key="teaching_metric_0",
        scored_path=scored, noun="story", n_examples=8, n_reconstruction_draws=24,
        choice_readout="logits", fixed_no_demo_canonical_probabilities=uniform24,
        choice_probabilities_content_cached=True)
    incremental_rows = backend.rows - screen_rows

    fresh_backend = _CountingUniformSelector()
    fresh = CachedChoiceReconstructor(
        fresh_backend, tmp_path / "fresh.sqlite", model="model", revision="revision")
    evaluate_scored_prompt_values(
        fresh, codebook_manifest=manifest, target_metric_key="teaching_metric_0",
        scored_path=scored, noun="story", n_examples=8, n_reconstruction_draws=24,
        choice_readout="logits", fixed_no_demo_canonical_probabilities=uniform24,
        choice_probabilities_content_cached=True)
    assert screen_rows > 0
    assert incremental_rows < fresh_backend.rows
    assert screen_rows + incremental_rows == fresh_backend.rows
    assert screen["option_order_design"]["canonical_option_orders"] == full[
        "option_order_design"]["canonical_option_orders"][:4]


def test_task_centralness_reference_is_deterministic_batched_and_hash_bound(tmp_path):
    paths = _bootstraps(tmp_path)
    forward = build_task_centralness_reference_plan(paths, seed=23)
    reverse = build_task_centralness_reference_plan(list(reversed(paths)), seed=23)
    assert forward == reverse
    assert forward["contexts_per_metric"] == 4
    assert forward["premises"][
        "uses_bootstrap_descriptions_and_frozen_provenance_only"] is True
    assert forward["premises"][
        "centralness_scalar_is_not_a_certificate_premise"] is True
    assert forward["option_order_design"] == mcq_option_order_design(4, 4)
    assert all(
        len(rows) == 4
        and all(len(set(row["option_metric_keys"])) == 4 for row in rows)
        for rows in forward["contexts"].values()
    )

    backend = _CountingUniformSelector()
    calibration = score_task_centralness_reference(
        backend,
        reference_plan=forward,
        noun="story",
        query_batch_size=512,
        reconstructor_model="model",
        reconstructor_revision="revision",
    )
    assert backend.calls == 1
    assert backend.rows == len(paths) * 4 * 4
    scores = validate_task_centralness_calibration(forward, calibration)
    assert scores == pytest.approx({f"metric_{index}": 0.0 for index in range(4)})

    mutated = json.loads(json.dumps(calibration))
    mutated["centralness"]["metric_0"]["score"] = 1.0
    with pytest.raises(ValueError, match="invalid or mismatched"):
        validate_task_centralness_calibration(forward, mutated)

    displayed_mutation = json.loads(json.dumps(calibration))
    displayed = displayed_mutation["rows"]["metric_0"][0]["prior"][
        "displayed_choice_probabilities"]
    displayed[0][0], displayed[0][1] = 0.30, 0.20
    core = dict(displayed_mutation)
    core.pop("calibration_sha256")
    displayed_mutation["calibration_sha256"] = hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="displayed rows or rendered queries"):
        validate_task_centralness_calibration(forward, displayed_mutation)

    query_mutation = json.loads(json.dumps(calibration))
    query_mutation["rows"]["metric_0"][0]["prior"]["query_sha256"][0] = "0" * 64
    core = dict(query_mutation)
    core.pop("calibration_sha256")
    query_mutation["calibration_sha256"] = hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="displayed rows or rendered queries"):
        validate_task_centralness_calibration(forward, query_mutation)


def test_centralness_worker_writes_an_immutable_validated_artifact(
    tmp_path, monkeypatch,
):
    plan = build_task_centralness_reference_plan(_bootstraps(tmp_path / "bank"), seed=23)
    plan_path = tmp_path / "centralness_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output_path = tmp_path / "centralness_calibration.json"
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps([{
        "expected_reconstructor_model": "fake/reconstructor",
        "expected_reconstructor_revision": "revision",
        "expected_choice_readout_id": CHOICE_READOUT_ID,
        "n_draws": 4,
        "centralness_plan": str(plan_path),
        "noun": "story",
        "query_batch_size": 512,
        "out": str(output_path),
    }]), encoding="utf-8")
    backend = _CountingUniformSelector()
    monkeypatch.setattr(
        centralness_worker, "make_judge_backend",
        lambda model, config, temperature: backend,
    )
    monkeypatch.setattr(
        centralness_worker, "ImplementerConfig",
        lambda: SimpleNamespace(vllm_lfs_home=None, vllm_fake=False),
    )
    monkeypatch.setattr(
        centralness_worker, "model_revision_id", lambda model: "revision")

    centralness_worker.run(jobs_path, model="fake/reconstructor")
    observed = json.loads(output_path.read_text(encoding="utf-8"))
    assert backend.calls == 1
    assert backend.rows == 4 * 4 * 4
    assert validate_task_centralness_calibration(plan, observed) == pytest.approx(
        {f"metric_{index}": 0.0 for index in range(4)})

    # Resume recomputes deterministically and accepts only the byte-identical artifact.
    centralness_worker.run(jobs_path, model="fake/reconstructor")
    assert backend.calls == 2
    with pytest.raises(RuntimeError, match="differs from the frozen run manifest"):
        centralness_worker.run(jobs_path, model="different/reconstructor")


def test_prelocked_centralness_fallback_repairs_zero_pass_without_changing_primary(
    tmp_path,
):
    n = 80
    target = np.tile([0.0, 1.0], n // 2)
    paths = [_write_bootstrap(tmp_path, "metric_0", target, "salient target")]
    for index in range(16):
        vector = target.copy()
        vector[np.arange(index % 10, n, 10)] = 1.0 - vector[np.arange(index % 10, n, 10)]
        paths.append(_write_bootstrap(
            tmp_path, f"metric_{index + 1}", vector, f"obscure near metric {index}"))
    for index in range(6):
        vector = target.copy()
        vector[np.arange(index % 5, n, 5)] = 1.0 - vector[np.arange(index % 5, n, 5)]
        paths.append(_write_bootstrap(
            tmp_path, f"metric_{17 + index}", vector, f"salient alternative {index}"))

    centralness_plan, centralness_calibration = _centralness_artifacts(
        paths, selector=_SemanticPriorSelector())
    primary_only = build_codebook_panel_plan(
        paths,
        centralness_reference_plan=centralness_plan,
        centralness_calibration=centralness_calibration,
        target_metric_keys=["metric_0"],
        n_options=4,
        design_size=60,
        min_design_disagreements=2,
        candidate_pool_size=16,
        max_panels_per_target=560,
        centralness_candidate_pool_size=22,
        centralness_fallback_panels_per_target=0,
        seed=9,
    )
    repaired = build_codebook_panel_plan(
        paths,
        centralness_reference_plan=centralness_plan,
        centralness_calibration=centralness_calibration,
        target_metric_keys=["metric_0"],
        n_options=4,
        design_size=60,
        min_design_disagreements=2,
        candidate_pool_size=16,
        max_panels_per_target=560,
        centralness_candidate_pool_size=22,
        centralness_fallback_panels_per_target=64,
        seed=9,
    )
    arms = repaired["panel_arms"]["metric_0"]
    assert arms["n_primary"] == 560
    assert arms["n_fallback"] == 64
    assert repaired["panels"]["metric_0"][:560] == primary_only["panels"]["metric_0"]
    assert repaired["premises"]["fallback_is_not_conditioned_on_primary_gate_results"]

    calibration = score_codebook_panel_priors(
        _SemanticPriorSelector(),
        panel_plan=repaired,
        noun="story",
        n_draws=24,
        reconstructor_model="semantic-selector",
        reconstructor_revision="revision",
    )
    ranked = prior_balanced_panel_rows(repaired, calibration)
    rows = ranked["rows"]["metric_0"]
    primary = [row for row in rows
               if row["panel_plan_provenance"]["arm"] == "legacy_behavior_primary"]
    fallback = [row for row in rows
                if row["panel_plan_provenance"]["arm"] == "blind_centralness_fallback"]
    assert not any(row["passes_prior_balance"] for row in primary)
    assert any(row["passes_prior_balance"] for row in fallback)
    assert all(row["prior"]["option_order_design"] == mcq_option_order_design(4, 24)
               for row in rows)
    chosen = next(row for row in fallback if row["passes_prior_balance"])
    selection = _frozen_prior_selection("metric_0", chosen, rows, ranked)
    selected_manifest = build_frozen_codebook_manifest(
        paths,
        n_options=4,
        design_size=60,
        min_design_disagreements=2,
        seed=9,
        panel_selections=selection,
        reconstruction_noun="story",
    )
    provenance = selected_manifest["entries"]["metric_0"]["panel_plan_selection"]
    assert provenance["arm"] == "blind_centralness_fallback"
    assert provenance["arm_rank"] >= 0
    assert provenance["centralness_reference_plan_sha256"] == centralness_plan[
        "plan_sha256"]
    assert provenance["centralness_calibration_sha256"] == centralness_calibration[
        "calibration_sha256"]
    validate_codebook_manifest(selected_manifest)


def test_codebook_is_bootstrap_only_frozen_and_hash_validated(tmp_path):
    manifest = build_frozen_codebook_manifest(
        _bootstraps(tmp_path), n_options=4, design_size=12,
        min_design_disagreements=2, seed=4, reconstruction_noun="story")
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
    design = set(manifest["design_indices"])
    for entry in manifest["entries"].values():
        assert len(entry["fixed_teaching_indices"]) == 8
        assert design.isdisjoint(entry["fixed_teaching_indices"])

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

    centralness_plan, centralness_calibration = _centralness_artifacts(paths)
    plan = build_codebook_panel_plan(
        paths,
        centralness_reference_plan=centralness_plan,
        centralness_calibration=centralness_calibration,
        target_metric_keys=["metric_0"],
        n_options=4,
        design_size=24,
        min_design_disagreements=2,
        candidate_pool_size=6,
        max_panels_per_target=20,
        seed=9,
    )
    calibration = score_codebook_panel_priors(
        _SemanticPriorSelector(),
        panel_plan=plan,
        noun="story",
        n_draws=24,
        reconstructor_model="semantic-selector",
        reconstructor_revision="revision",
    )
    assert calibration["option_order_design"]["exact_full_factorial"] is True
    assert calibration["option_order_design"]["n_unique_orders"] == 24
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
        design_size=24,
        min_design_disagreements=2,
        seed=9,
        panel_selections=selections,
        reconstruction_noun="story",
    )
    validate_codebook_manifest(manifest)
    entry = manifest["entries"]["metric_0"]
    assert entry["selection_method"].startswith("blind_no_demo")
    assert entry["prior_calibration"]["passes_prior_balance"] is True

    ranked = prior_balanced_panel_rows(
        plan, calibration,
        maximum_option_probability=1.0,
        target_probability_tolerance=1.0,
        minimum_normalized_entropy=0.0,
    )
    rows = ranked["rows"]["metric_0"]
    assert len(rows) >= 3
    envelopes = {"metric_0": {}}
    for index, row in enumerate(rows):
        envelopes["metric_0"][row["panel_id"]] = {
            "target_metric_key": "metric_0",
            "prior_panel_id": row["panel_id"],
            "distractor_metric_keys": row["distractor_metric_keys"],
            "finite_state_upper_bound": 0.1 + 0.01 * index,
            "state_envelope_capability": {
                "has_positive_unique_target_maximizer": index != len(rows) - 1,
            },
            "summary_sha256": f"summary-{index}",
            "state_function_semantic_sha256": f"semantic-{index}",
        }
    chosen = select_state_capable_panels(
        plan, calibration, envelopes,
        maximum_option_probability=1.0,
        target_probability_tolerance=1.0,
        minimum_normalized_entropy=0.0,
    )["metric_0"]
    assert chosen["prior_calibration"]["panel_id"] == rows[-2]["panel_id"]
    assert chosen["state_envelope_selection"]["passes_state_capability"] is True
    assert chosen["state_envelope_selection"]["n_live_panels"] == len(rows) - 1
    assert chosen["state_envelope_selection"][
        "chosen_state_function_semantic_sha256"] == f"semantic-{len(rows) - 2}"


def test_every_scored_row_receives_an_anchor_free_mcq_value(tmp_path):
    paths = _bootstraps(tmp_path)
    manifest = build_frozen_codebook_manifest(
        paths, n_options=4, design_size=12, min_design_disagreements=2, seed=4,
        reconstruction_noun="story")
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
    assert payload["option_order_design"]["exact_full_factorial"] is False

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
    assert loaded["option_order_design"] == payload["option_order_design"]

    tampered = dict(payload)
    tampered["option_order_design"] = json.loads(json.dumps(
        payload["option_order_design"]))
    tampered["option_order_design"]["canonical_option_orders"][0] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="mutated option-order block"):
        write_value_artifact(
            tmp_path / "tampered_values.npz",
            tampered,
            reconstructor_model="fixed-reconstructor",
            reconstructor_revision="revision",
        )


def test_exhaustive_fixed_transcript_table_is_an_all_prompt_upper_envelope(tmp_path):
    paths = _bootstraps(tmp_path)
    manifest = build_frozen_codebook_manifest(
        paths, n_options=4, design_size=12, min_design_disagreements=2, seed=4,
        reconstruction_noun="story")
    state_path = tmp_path / "states.npz"
    table = write_finite_state_scored_artifact(
        state_path, codebook_manifest=manifest, target_metric_key="metric_0")
    assert table["n_states"] == 256
    assert np.array_equal(table["state_integers"], np.arange(256))
    assert len({tuple(row) for row in table["state_bits"]}) == 256

    state_values = evaluate_scored_prompt_values(
        _ConditionSensitiveSelector(),
        codebook_manifest=manifest,
        target_metric_key="metric_0",
        scored_path=state_path,
        noun="story",
        n_examples=8,
        n_reconstruction_draws=4,
        choice_readout="logits",
        choice_probabilities_content_cached=True,
    )
    value_path = tmp_path / "state_values.npz"
    write_value_artifact(
        value_path, state_values,
        reconstructor_model="fixed-reconstructor", reconstructor_revision="revision")
    loaded_values = load_value_artifact(
        value_path,
        expected_source_scored_sha256=table["sha256"],
        expected_codebook_manifest_sha256=manifest["manifest_sha256"],
    )
    envelope = build_finite_state_envelope(
        codebook_manifest=manifest,
        target_metric_key="metric_0",
        state_scored_path=state_path,
        value_payload=loaded_values,
    )
    assert envelope["finite_state_upper_bound"] == pytest.approx(0.6)
    assert envelope["coarse_no_demo_range_cap"] == pytest.approx(0.8)
    assert envelope["state_envelope_capability"][
        "has_positive_unique_target_maximizer"] is True
    assert envelope["operational_target_diagnostic"]["is_headline_gate"] is True
    assert envelope["operational_target_diagnostic"][
        "eligible_as_achieved_prompt_lower_bound"] is False
    assert len(envelope["state_function_semantic_sha256"]) == 64

    target = np.load(paths[0], allow_pickle=True)["target"]
    candidate_path = tmp_path / "arbitrary_candidates.npz"
    np.savez_compressed(
        candidate_path,
        sigs=np.vstack([target, 1.0 - target, np.zeros_like(target)]),
        texts=np.asarray(["target", "inverse", "constant"], object),
        probe_sha256=np.asarray("shared-probes"),
    )
    candidates = evaluate_scored_prompt_values(
        _ConditionSensitiveSelector(),
        codebook_manifest=manifest,
        target_metric_key="metric_0",
        scored_path=candidate_path,
        noun="story",
        n_examples=8,
        n_reconstruction_draws=4,
        choice_readout="logits",
        choice_probabilities_content_cached=True,
    )
    assert np.all(candidates["values"] <= envelope["finite_state_upper_bound"] + 1e-12)
    looked_up = lookup_scored_prompt_values(
        codebook_manifest=manifest,
        target_metric_key="metric_0",
        scored_path=candidate_path,
        state_scored_path=state_path,
        state_value_payload=loaded_values,
        envelope_summary=envelope,
    )
    assert looked_up["finite_state_lookup"] is True
    assert looked_up["values"].tolist() == pytest.approx(candidates["values"].tolist())
    assert looked_up["raw_target_option_probability"].tolist() == pytest.approx(
        candidates["raw_target_option_probability"].tolist())
    expected_indices = manifest["entries"]["metric_0"]["fixed_teaching_indices"]
    assert all(
        detail["design"]["indices_in_prompt_order"] == expected_indices
        for detail in candidates["details"]
    )

    validate_finite_state_envelope(
        envelope,
        codebook_manifest=manifest,
        target_metric_key="metric_0",
        state_scored_path=state_path,
        value_payload=loaded_values,
    )
    mutated_summary = json.loads(json.dumps(envelope))
    mutated_summary["finite_state_upper_bound"] += 0.01
    with pytest.raises(ValueError, match="mutated"):
        validate_finite_state_envelope(
            mutated_summary,
            codebook_manifest=manifest,
            target_metric_key="metric_0",
            state_scored_path=state_path,
            value_payload=loaded_values,
        )

    inconsistent_values = dict(loaded_values)
    inconsistent_values["details"] = json.loads(json.dumps(loaded_values["details"]))
    inconsistent_values["details"][0]["value_mark"] += 0.01
    with pytest.raises(ValueError, match="inconsistent value_mark"):
        build_finite_state_envelope(
            codebook_manifest=manifest,
            target_metric_key="metric_0",
            state_scored_path=state_path,
            value_payload=inconsistent_values,
        )

    inconsistent_matrix = dict(loaded_values)
    inconsistent_matrix["details"] = json.loads(json.dumps(loaded_values["details"]))
    probabilities = inconsistent_matrix["details"][0]["identification"]["conditions"][
        "annotations"]["canonical_choice_probabilities"]
    probabilities[0][0] += 0.01
    probabilities[0][1] -= 0.01
    with pytest.raises(ValueError, match="scalar summary inconsistent"):
        build_finite_state_envelope(
            codebook_manifest=manifest,
            target_metric_key="metric_0",
            state_scored_path=state_path,
            value_payload=inconsistent_matrix,
        )

    forged_transcript = dict(loaded_values)
    forged_transcript["details"] = json.loads(json.dumps(loaded_values["details"]))
    forged_transcript["details"][0]["design"]["teaching_transcript_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="forged transcript hash"):
        build_finite_state_envelope(
            codebook_manifest=manifest,
            target_metric_key="metric_0",
            state_scored_path=state_path,
            value_payload=forged_transcript,
        )

    with np.load(state_path, allow_pickle=True) as z:
        mutated_payload = {name: z[name] for name in z.files}
    mutated_payload["state_bits"] = mutated_payload["state_bits"].copy()
    mutated_payload["state_bits"][3, 0] ^= 1
    mutated_path = tmp_path / "mutated_states.npz"
    np.savez_compressed(mutated_path, **mutated_payload)
    with pytest.raises(ValueError, match="mutated"):
        load_finite_state_scored_artifact(
            mutated_path,
            codebook_manifest=manifest,
            target_metric_key="metric_0",
        )


def test_value_worker_writes_every_row_with_one_resident_fake_backend(tmp_path):
    paths = _bootstraps(tmp_path)
    manifest = build_frozen_codebook_manifest(
        paths, n_options=4, design_size=12, min_design_disagreements=2, seed=4,
        reconstruction_noun="story")
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
        "expected_reconstructor_model": "fake-reconstructor",
        "expected_reconstructor_revision": "fake-reconstructor",
        "expected_choice_readout_id": FAKE_CHOICE_READOUT_ID,
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
    with pytest.raises(ValueError, match="reconstructor revision"):
        load_value_artifact(
            output,
            expected_reconstructor_model="fake-reconstructor",
            expected_reconstructor_revision="different-revision",
        )

    bad_jobs = tmp_path / "bad_revision_jobs.json"
    bad_jobs.write_text(json.dumps([{
        "expected_reconstructor_model": "fake-reconstructor",
        "expected_reconstructor_revision": "different-revision",
        "expected_choice_readout_id": FAKE_CHOICE_READOUT_ID,
        "out": str(tmp_path / "must_not_exist.npz"),
    }]))
    with pytest.raises(RuntimeError, match="revision differs from the frozen run manifest"):
        stage_value(SimpleNamespace(
            jobs=str(bad_jobs), model="fake-reconstructor", fake=True))
    assert not (tmp_path / "must_not_exist.npz").exists()
