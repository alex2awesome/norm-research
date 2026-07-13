"""CPU tests for the resampled-teaching-panel value certificate driver (CR-3 v13)."""
from __future__ import annotations

import re

import numpy as np
import pytest

from methods.metric_implementer.experiments.cr3_reconstruction_values import (
    TEACHING_LIBRARY_SIZE,
    build_frozen_codebook_manifest,
    build_teaching_panel_library,
)
from methods.metric_implementer.experiments.cr3_sampled_value_certify import (
    DEGENERATE_CONTROL_FAMILY,
    HEADLINE_MAX_ACHIEVED_VALUE_CI_WIDTH,
    HEADLINE_MIN_BLIND_HEADROOM,
    PLANTED_CONTROL_FAMILY,
    PRIMARY_CI_PERCENTILES,
    SAMPLED_VALUE_SCHEMA,
    SENSITIVITY_CI_PERCENTILES,
    VALUE_STATUS_CERTIFIED_PRIMARY,
    VALUE_STATUS_FORMAL_ONLY,
    VALUE_STATUS_SUGGESTIVE_SENSITIVITY,
    _allocate_horizon,
    _blind_menu_prior,
    _headline_eligible,
    _panel_index_plan,
    _percentile_interval,
    _value_status,
    certify_sampled_value,
    main,
    write_sampled_value_certificate,
)
from methods.metric_implementer.experiments.cr_audit import (
    dkw_expected_max_lower,
    dkw_expected_max_upper,
)

CHOICE_READOUT_ID = "allowed-exact-single-token-choice-processed-posterior-v2"


# --- synthetic instrument fixtures -----------------------------------------------------

def _write_bootstrap(root, key, target, description, probes):
    directory = root / key / "bootstrap"
    directory.mkdir(parents=True)
    path = directory / "scored.npz"
    np.savez_compressed(
        path,
        sigs=np.asarray([target]),
        texts=np.asarray([f"seed prompt {key}"], object),
        target=np.asarray(target, float),
        metric_description=np.asarray(description),
        probe_texts=np.asarray(probes, object),
        probe_sha256=np.asarray("shared-probes"),
        executor_model=np.asarray("meta-llama/Llama-3.1-8B-Instruct"),
        executor_model_revision=np.asarray("executor-revision"),
        readout_id=np.asarray("hard-readout"),
    )
    return path


def _marked_bootstraps(root, *, n=120):
    """Bootstraps whose probe texts carry the target's verdict as a ``TB=<bit>`` marker."""
    indices = np.arange(n)
    target = (indices % 2).astype(float)
    probes = [f"TB={int(target[i])} Scene {i:03d} reply context" for i in range(n)]
    vectors = [
        target,
        np.where(indices % 5 == 0, 1.0 - target, target),
        np.where(indices % 7 < 2, 1.0 - target, target),
        np.where((indices // 3) % 4 == 0, 1.0 - target, target),
    ]
    descriptions = [
        "SALIENTTARGET salient property of the reply",
        "distractor criterion alpha",
        "distractor criterion beta",
        "distractor criterion gamma",
    ]
    return [
        _write_bootstrap(root, f"metric_{index}", vector, descriptions[index], probes)
        for index, vector in enumerate(vectors)
    ]


def _manifest(root, *, n=120):
    return build_frozen_codebook_manifest(
        _marked_bootstraps(root, n=n), n_options=4, design_size=24,
        min_design_disagreements=2, seed=31, reconstruction_noun="story")


def _write_pool(path, *, sigs, texts, families):
    np.savez_compressed(
        path,
        sigs=np.asarray(sigs, float),
        texts=np.asarray(list(texts), object),
        families=np.asarray(list(families), object),
        probe_sha256=np.asarray("shared-probes"),
    )
    return path


# --- fake reconstructors (no GPU) ------------------------------------------------------

class _ConditionSensitiveSelector:
    """Annotation queries recover the target (0.8); every control is blind chance (0.2)."""

    choice_readout_id = CHOICE_READOUT_ID

    def score_choices(self, prompts, choices, **kwargs):
        seeds = kwargs.get("seed")
        if not isinstance(seeds, (list, tuple, np.ndarray)):
            seeds = [seeds] * len(prompts)
        rows = []
        for prompt, seed in zip(prompts, seeds):
            target_position = next(
                int(number) - 1 for number, description in
                re.findall(r"(?m)^(\d+)\. (.+)$", prompt)
                if "SALIENTTARGET" in description)
            target_probability = 0.8 if int(seed) // 10_000 == 1 else 0.2
            row = [(1.0 - target_probability) / (len(choices) - 1)] * len(choices)
            row[target_position] = target_probability
            rows.append(row)
        return rows


class _CoherenceSelector:
    """Target probability rises only when EVERY shown label matches the text's ``TB`` marker.

    A prompt whose verdicts equal the target's are perfectly coherent on every panel; a
    constant-verdict prompt and any shuffled-label control are not. The blind no-demo query
    returns exact chance, so identity never leaks through the prior.
    """

    choice_readout_id = CHOICE_READOUT_ID

    def score_choices(self, prompts, choices, **_kwargs):
        n_options = len(choices)
        rows = []
        for prompt in prompts:
            target_position = next(
                int(number) - 1 for number, description in
                re.findall(r"(?m)^(\d+)\. (.+)$", prompt)
                if "SALIENTTARGET" in description)
            if "(No scored examples are provided.)" in prompt:
                rows.append([1.0 / n_options] * n_options)
                continue
            examples = re.findall(r"\[score=(\d)\]\n```\n(.*?)\n```", prompt, re.DOTALL)
            matches = []
            for score, text in examples:
                marker = re.search(r"TB=(\d)", text)
                if marker is not None:
                    matches.append(int(score) == int(marker.group(1)))
            alignment = float(np.mean(matches)) if matches else 0.5
            target_probability = 0.2 + 0.6 * (1.0 if alignment > 0.99 else 0.0)
            row = [(1.0 - target_probability) / (n_options - 1)] * n_options
            row[target_position] = target_probability
            rows.append(row)
        return rows


class _HighBlindPriorSelector:
    """Blind menu already near-certain about the target: no headroom for articulation."""

    choice_readout_id = CHOICE_READOUT_ID

    def score_choices(self, prompts, choices, **_kwargs):
        n_options = len(choices)
        rows = []
        for prompt in prompts:
            target_position = next(
                int(number) - 1 for number, description in
                re.findall(r"(?m)^(\d+)\. (.+)$", prompt)
                if "SALIENTTARGET" in description)
            row = [0.05 / (n_options - 1)] * n_options
            row[target_position] = 0.95
            rows.append(row)
        return rows


class _RecordingSelector:
    """Records every rendered query; returns uniform choice mass."""

    choice_readout_id = CHOICE_READOUT_ID

    def __init__(self):
        self.prompts = []

    def score_choices(self, prompts, choices, **_kwargs):
        self.prompts.extend(prompts)
        return [[1.0 / len(choices)] * len(choices) for _ in prompts]


# --- tests -----------------------------------------------------------------------------

def test_panel_plan_is_deterministic_records_R_and_changes_with_R(tmp_path):
    manifest = _manifest(tmp_path / "bank")
    plan_a = _panel_index_plan(manifest, "metric_0", n_panels=12)
    plan_b = _panel_index_plan(manifest, "metric_0", n_panels=12)
    assert plan_a == plan_b
    assert plan_a["n_panels_R"] == 12
    assert len(plan_a["panels"]) == 12
    design_split = set(int(index) for index in manifest["design_indices"])
    for panel in plan_a["panels"]:
        indices = panel["fixed_teaching_indices"]
        assert len(set(indices)) == 8
        assert design_split.isdisjoint(indices)

    plan_c = _panel_index_plan(manifest, "metric_0", n_panels=20)
    assert plan_c["n_panels_R"] == 20
    assert len(plan_c["panels"]) == 20
    assert plan_c["library_sha256"] != plan_a["library_sha256"]
    # The first panel is the exact v12 baseline regardless of R.
    assert plan_a["panels"][0]["role"] == plan_c["panels"][0]["role"] == "baseline_exact"


def test_library_size_parameterization_default_and_bounds(tmp_path):
    manifest = _manifest(tmp_path / "bank")
    default_library = build_teaching_panel_library(manifest, target_metric_key="metric_0")
    assert default_library["library_size"] == TEACHING_LIBRARY_SIZE == 8
    assert default_library == build_teaching_panel_library(
        manifest, target_metric_key="metric_0", library_size=8)

    twelve = build_teaching_panel_library(
        manifest, target_metric_key="metric_0", library_size=12)
    assert twelve["library_size"] == 12
    assert len({tuple(p["fixed_teaching_indices"]) for p in twelve["panels"]}) == 12

    small = build_teaching_panel_library(
        manifest, target_metric_key="metric_0", library_size=2)
    assert small["library_size"] == 2 and len(small["panels"]) == 2
    assert small["panels"][0]["role"] == "baseline_exact"

    for bad in (1, 0, 65, -3):
        with pytest.raises(ValueError, match=r"\[2, 64\]"):
            build_teaching_panel_library(
                manifest, target_metric_key="metric_0", library_size=bad)


def test_percentile_ci_and_mean_match_hand_built_matrix():
    matrix = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.30, 0.30, 0.30, 0.30, 0.30, 0.30],
    ])
    primary = _percentile_interval(matrix, PRIMARY_CI_PERCENTILES)
    assert primary[0].tolist() == pytest.approx([0.025, 0.30])
    assert primary[1].tolist() == pytest.approx([0.975, 0.30])
    sensitivity = _percentile_interval(matrix, SENSITIVITY_CI_PERCENTILES)
    assert sensitivity[0].tolist() == pytest.approx([0.05, 0.30])
    assert sensitivity[1].tolist() == pytest.approx([0.95, 0.30])
    assert matrix.mean(axis=1).tolist() == pytest.approx([0.5, 0.30])
    # The percentile wrapper is exactly numpy's per-row percentile over the panel axis.
    assert primary[0].tolist() == pytest.approx(
        np.percentile(matrix, PRIMARY_CI_PERCENTILES[0], axis=1).tolist())


def test_allocate_horizon_is_deterministic_and_sums_exactly():
    allocation = _allocate_horizon({"fA": 2, "fB": 1}, 100)
    assert allocation == {"fA": 67, "fB": 33}
    assert sum(allocation.values()) == 100
    assert _allocate_horizon({"a": 1, "b": 1, "c": 1}, 10) == {"a": 4, "b": 3, "c": 3}
    assert _allocate_horizon({"a": 3}, 0) == {"a": 0}


def test_dkw_gain_upper_bound_is_conservative_against_monte_carlo():
    rng = np.random.default_rng(20260713)
    b_cap = 0.8
    marks = np.clip(rng.beta(2.0, 5.0, size=200) * b_cap, 0.0, b_cap)
    horizon = 50
    alpha = 0.05
    upper, epsilon = dkw_expected_max_upper({"f": marks}, {"f": horizon}, b_cap, alpha)
    lower, _ = dkw_expected_max_lower({"f": marks}, {"f": horizon}, b_cap, alpha)
    draws = rng.integers(0, len(marks), size=(40_000, horizon))
    monte_carlo_expected_max = float(np.mean(marks[draws].max(axis=1)))
    assert upper >= monte_carlo_expected_max
    assert lower <= monte_carlo_expected_max
    assert epsilon["f"] > 0.0
    # The gain over any observed best is at least the true expected-max gain.
    observed_best = float(np.max(marks))
    assert max(0.0, upper - observed_best) >= max(0.0, monte_carlo_expected_max - observed_best)


def test_blind_no_demo_query_hides_target_identity_and_shows_no_demonstrations(tmp_path):
    manifest = _manifest(tmp_path / "bank")
    entry = manifest["entries"]["metric_0"]
    from methods.metric_implementer.experiments.cr3_reconstruction_values import _bootstrap
    metric_meta = manifest["metrics"]
    option_descriptions = [entry["target_description"]] + [
        _bootstrap(metric_meta[key]["bootstrap_path"])["description"]
        for key in entry["distractor_metric_keys"]
    ]
    recorder = _RecordingSelector()
    _blind_menu_prior(
        recorder, noun="story", option_descriptions=option_descriptions, n_perms=8)
    assert recorder.prompts, "blind prior must render at least one query"
    for query in recorder.prompts:
        assert "(No scored examples are provided.)" in query
        assert "[score=" not in query          # no labeled teaching demonstrations
        assert "metric_0" not in query          # the target's identity key never leaks
        assert "TB=" not in query               # no per-item verdict markers disclosed


def test_headline_gates_use_both_frozen_constants():
    assert HEADLINE_MIN_BLIND_HEADROOM == 0.10
    assert HEADLINE_MAX_ACHIEVED_VALUE_CI_WIDTH == 0.15
    # Both thresholds must pass for eligibility.
    assert _headline_eligible(blind_headroom=0.50, achieved_value_ci_width=0.10) is True
    assert _headline_eligible(blind_headroom=0.09, achieved_value_ci_width=0.00) is False
    assert _headline_eligible(blind_headroom=0.50, achieved_value_ci_width=0.16) is False
    assert _value_status(
        headline_eligible=True, achieved_value=0.3,
        positive_label=VALUE_STATUS_CERTIFIED_PRIMARY) == VALUE_STATUS_CERTIFIED_PRIMARY
    assert _value_status(
        headline_eligible=True, achieved_value=0.0,
        positive_label=VALUE_STATUS_CERTIFIED_PRIMARY) == VALUE_STATUS_FORMAL_ONLY
    assert _value_status(
        headline_eligible=False, achieved_value=0.9,
        positive_label=VALUE_STATUS_CERTIFIED_PRIMARY) == VALUE_STATUS_FORMAL_ONLY


def test_planted_control_certifies_positive_value(tmp_path):
    manifest = _manifest(tmp_path / "bank")
    pool = _write_pool(
        tmp_path / "pool.npz",
        sigs=[np.ones(120)], texts=["noise constant one"], families=["mined"])
    result = certify_sampled_value(
        _CoherenceSelector(), codebook_manifest=manifest, target_metric_key="metric_0",
        scored_pool_path=pool, n_panels=12, n_perms=8, mcq_n_options=4, alpha=0.05,
        horizons=[100, 300], reconstructor_model="google/gemma-4-31b-it",
        reconstructor_revision="rev", planted_control=True)
    certificate = result["certificate"]
    assert certificate["schema"] == SAMPLED_VALUE_SCHEMA
    primary = certificate["reporting"]["primary_95"]
    assert primary["value_status"] == VALUE_STATUS_CERTIFIED_PRIMARY
    achieved = certificate["achieved_value"]["primary_95"]
    assert achieved["achieved_value"] > 0.0
    assert achieved["achieved_value_prompt_family"] == PLANTED_CONTROL_FAMILY
    assert certificate["achieved_value"]["per_family_best_mean_value"][
        PLANTED_CONTROL_FAMILY]["best_mean_value"] > 0.0
    assert certificate["achieved_value"]["per_family_best_mean_value"][
        "mined"]["best_mean_value"] == pytest.approx(0.0)
    assert certificate["calibration_controls"]["calibration_run"] is True
    # Panel-invariant recovery ⇒ a tight interval and an eligible headline.
    assert achieved["achieved_value_ci_width"] <= HEADLINE_MAX_ACHIEVED_VALUE_CI_WIDTH
    assert certificate["reporting"]["sensitivity_90"]["value_status"] == (
        VALUE_STATUS_SUGGESTIVE_SENSITIVITY)


def test_degenerate_control_is_formal_certificate_only(tmp_path):
    manifest = _manifest(tmp_path / "bank")
    pool = _write_pool(
        tmp_path / "pool.npz",
        sigs=[np.ones(120)], texts=["noise constant one"], families=["mined"])
    result = certify_sampled_value(
        _CoherenceSelector(), codebook_manifest=manifest, target_metric_key="metric_0",
        scored_pool_path=pool, n_panels=12, n_perms=8, mcq_n_options=4, alpha=0.05,
        horizons=[100], reconstructor_model="google/gemma-4-31b-it",
        reconstructor_revision="rev", degenerate_control=True)
    certificate = result["certificate"]
    assert certificate["reporting"]["primary_95"]["value_status"] == VALUE_STATUS_FORMAL_ONLY
    assert certificate["reporting"]["sensitivity_90"]["value_status"] == VALUE_STATUS_FORMAL_ONLY
    assert certificate["achieved_value"]["primary_95"]["achieved_value"] == pytest.approx(0.0)
    assert certificate["achieved_value"]["per_family_best_mean_value"][
        DEGENERATE_CONTROL_FAMILY]["best_mean_value"] == pytest.approx(0.0)


def test_low_blind_headroom_forces_formal_certificate_only(tmp_path):
    manifest = _manifest(tmp_path / "bank")
    target = np.load(_marked_bootstraps(tmp_path / "unused")[0], allow_pickle=True)["target"]
    pool = _write_pool(
        tmp_path / "pool.npz",
        sigs=[(target > 0.5).astype(float)], texts=["recovers target"], families=["mined"])
    result = certify_sampled_value(
        _HighBlindPriorSelector(), codebook_manifest=manifest, target_metric_key="metric_0",
        scored_pool_path=pool, n_panels=12, n_perms=8, mcq_n_options=4, alpha=0.05,
        horizons=[100], reconstructor_model="google/gemma-4-31b-it",
        reconstructor_revision="rev")
    certificate = result["certificate"]
    gates = certificate["headline_gates"]
    assert gates["observed_blind_headroom"] < HEADLINE_MIN_BLIND_HEADROOM
    assert gates["headline_eligible_primary_95"] is False
    assert certificate["reporting"]["primary_95"]["value_status"] == VALUE_STATUS_FORMAL_ONLY


def test_certificate_is_deterministic_and_write_is_immutable(tmp_path):
    manifest = _manifest(tmp_path / "bank")
    pool = _write_pool(
        tmp_path / "pool.npz",
        sigs=[np.tile([0.0, 1.0], 60), np.zeros(120)],
        texts=["prompt a", "prompt b"], families=["fA", "fB"])
    kwargs = dict(
        codebook_manifest=manifest, target_metric_key="metric_0", scored_pool_path=pool,
        n_panels=12, n_perms=8, mcq_n_options=4, alpha=0.05, horizons=[100],
        reconstructor_model="google/gemma-4-31b-it", reconstructor_revision="rev")
    first = certify_sampled_value(_ConditionSensitiveSelector(), **kwargs)
    second = certify_sampled_value(_ConditionSensitiveSelector(), **kwargs)
    assert first["certificate"]["panel_plan_sha256"] == second["certificate"]["panel_plan_sha256"]
    assert first["certificate"]["certificate_sha256"] == second["certificate"]["certificate_sha256"]

    out_dir = tmp_path / "out" / "metric_0"
    written = write_sampled_value_certificate(out_dir, first)
    assert (out_dir / "certificate.json").exists()
    assert (out_dir / "per_prompt_values.npz").exists()
    assert len(written["per_prompt_table_sha256"]) == 64
    with pytest.raises(FileExistsError):
        write_sampled_value_certificate(out_dir, first)


def test_cli_fake_backends_end_to_end(tmp_path):
    manifest = _manifest(tmp_path / "bank")
    assets_root = tmp_path / "assets"
    metric_dir = assets_root / "metric_0"
    metric_dir.mkdir(parents=True)
    (metric_dir / "codebook.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    _write_pool(
        metric_dir / "pool.npz",
        sigs=[np.tile([0.0, 1.0], 60), np.zeros(120)],
        texts=["prompt a", "prompt b"], families=["fA", "fB"])
    out_root = tmp_path / "out"
    exit_code = main([
        "--assets-root", str(assets_root),
        "--task", "story",
        "--metrics", "metric_0",
        "--out-root", str(out_root),
        "--n-panels", "12",
        "--n-perms", "8",
        "--horizons", "100,300",
        "--fake-backends",
    ])
    assert exit_code == 0
    import json
    certificate = json.loads((out_root / "metric_0" / "certificate.json").read_text())
    assert certificate["schema"] == SAMPLED_VALUE_SCHEMA
    assert certificate["n_panels_R"] == 12
    assert certificate["reconstructor"]["choice_readout_id"] == "fake-hash-choice-probabilities-v1"
    run_manifest = json.loads((out_root / "run_manifest.json").read_text())
    assert run_manifest["n_panels_R"] == 12
    assert len(run_manifest["run_panel_plan_sha256"]) == 64
    # The out-root is fail-closed against reuse.
    with pytest.raises(FileExistsError):
        main([
            "--assets-root", str(assets_root), "--task", "story", "--metrics", "metric_0",
            "--out-root", str(out_root), "--fake-backends",
        ])
