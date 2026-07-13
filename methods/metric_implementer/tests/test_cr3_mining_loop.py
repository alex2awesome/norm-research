"""Production-path invariants for CR-3 sampling and resume behavior."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from methods.metric_implementer.experiments.run_cr3_mining_loop import (
    _combined_tier_status,
    _apply_publication_gate,
    _metric_task,
    _retryable_worker_failure,
    _validate_proposal,
    _validate_level_matched_codebook_banks,
    _worker_environment,
    attach_reporting_tiers,
    main as mining_main,
    mcq_instrument_quality,
    mcq_reported_global_status,
    reporting_alpha_tiers,
    validate_reconstructor_artifact_contract,
    validate_numeric_reuse_manifest,
)
from methods.metric_implementer.experiments.cr3_evidence_store import build_evidence_store
from scripts.tools.cr3_mining_worker import (
    READOUT_ID,
    _checked_signature,
    _content_cached_signature,
    draw_valid_rows,
    score_unique_texts,
)


class _RecordingBackend:
    def __init__(self):
        self.seeds = []

    def generate_batch(self, prompts, *, seed, **kwargs):
        seeds = [int(x) for x in seed]
        self.seeds.extend(seeds)
        outputs = []
        for value in seeds:
            if value % 7 == 0:
                outputs.append("invalid output")
            else:
                outputs.append(f"Does independently seeded property {value} hold?")
        return outputs


class _HolisticBackend:
    def generate_batch(self, prompts, *, seed, **kwargs):
        return [
            ("Judge the full metric by considering its central construct, supporting evidence, "
             "important exclusions, and edge cases. Require the text to satisfy the construct "
             f"coherently rather than matching one isolated cue. Independent draw {int(value)}.")
            for value in seed
        ]


def _status_certificate(mass_interval, gain_interval, *, pool_best=0.2):
    return {
        "scope": {"iid_provenance_established": True},
        "certified": {
            "pool_best_prompt_value": pool_best,
            "pool_best_prompt_recovery_bits": pool_best,
            "future_draws_per_family": {"family": 100},
        },
        "status_evidence": {
            "simultaneous_confidence": 0.95,
            "behavioral_missing_mass_interval": list(mass_interval),
            "finite_horizon_expected_best_gain_interval": list(gain_interval),
        },
        "estimand": {"value_unit": "probability"},
    }


def test_predeclared_reporting_tiers_allocate_across_metrics_and_slots():
    args = SimpleNamespace(
        alpha=0.05,
        study_alpha=0.05,
        checkpoint_iters="0,2",
        max_iter=3,
    )
    tiers = reporting_alpha_tiers(args, n_metrics=4)
    assert tiers["primary_95"]["cell_alpha"] == 0.05 / 12
    assert tiers["sensitivity_90"]["cell_alpha"] == 0.10 / 12
    assert tiers["primary_95"]["scope"]["overall_simultaneous_confidence"] == 0.95
    assert tiers["sensitivity_90"]["scope"]["overall_simultaneous_confidence"] == 0.90


def test_90_percent_only_resolution_is_suggestive_and_never_relabels_primary():
    primary = _status_certificate((0.05, 0.15), (0.0, 0.03))
    sensitivity = _status_certificate((0.06, 0.09), (0.0, 0.015))
    attach_reporting_tiers(
        primary,
        sensitivity,
        primary_scope={"overall_simultaneous_confidence": 0.95},
        sensitivity_scope={"overall_simultaneous_confidence": 0.90},
        plateau_epsilon=0.02,
        saturation_missing_mass=0.10,
    )
    assert primary["prompt_evolution_status"]["headline_status"] == "UNRESOLVED"
    secondary = primary["reporting_tiers"]["sensitivity_90"]["status"]
    assert secondary["behavior_status"] == "SUGGESTIVE_SATURATED"
    assert secondary["value_status"] == "SUGGESTIVE_PLATEAUED"
    combined = primary["reporting_tiers"]["combined_reporting_status"]
    assert combined["behavior_status"] == "SUGGESTIVE_SATURATED"
    assert combined["value_status"] == "SUGGESTIVE_PLATEAUED"
    assert primary["reporting_tiers"][
        "all_prompt_cap_is_exact_and_not_confidence_tiered"] is True


def test_primary_axis_conclusion_dominates_matching_sensitivity_conclusion():
    primary = {
        "behavior_status": "CERTIFIED_UNSATURATED",
        "value_status": "UNRESOLVED",
    }
    sensitivity = {
        "behavior_status": "SUGGESTIVE_UNSATURATED",
        "value_status": "SUGGESTIVE_RISING",
    }
    combined = _combined_tier_status(primary, sensitivity)
    assert combined["behavior_status"] == "CERTIFIED_UNSATURATED"
    assert combined["value_status"] == "SUGGESTIVE_RISING"


def test_formal_only_mcq_panel_demotes_value_status_but_preserves_behavior_status():
    primary = _status_certificate((0.11, 0.20), (0.0, 0.01))
    sensitivity = _status_certificate((0.12, 0.18), (0.0, 0.008))
    for certificate in (primary, sensitivity):
        certificate["all_finite_prompt_certificate"] = {
            "instrument_quality": {
                "headline_eligible": False,
                "reasons": ["prior-degenerate panel"],
            },
        }
    attach_reporting_tiers(
        primary,
        sensitivity,
        primary_scope={"overall_simultaneous_confidence": 0.95},
        sensitivity_scope={"overall_simultaneous_confidence": 0.90},
        plateau_epsilon=0.02,
        saturation_missing_mass=0.10,
    )
    status = primary["prompt_evolution_status"]
    assert status["behavior_status"] == "CERTIFIED_UNSATURATED"
    assert status["formal_mathematical_value_status"] == "CERTIFIED_PLATEAUED"
    assert status["value_status"] == "FORMAL_CERTIFICATE_ONLY"
    assert status["headline_status"] == (
        "CERTIFIED_BEHAVIORALLY_UNSATURATED_VALUE_FORMAL_ONLY")
    assert status["value_headline_eligible"] is False


def test_bad_mcq_panel_does_not_turn_unresolved_value_evidence_into_a_conclusion():
    primary = _status_certificate((0.11, 0.20), (0.0, 0.03))
    sensitivity = _status_certificate((0.12, 0.18), (0.0, 0.025))
    for certificate in (primary, sensitivity):
        certificate["all_finite_prompt_certificate"] = {
            "instrument_quality": {
                "headline_eligible": False,
                "reasons": ["prior-degenerate panel"],
            },
        }
    attach_reporting_tiers(
        primary,
        sensitivity,
        primary_scope={"overall_simultaneous_confidence": 0.95},
        sensitivity_scope={"overall_simultaneous_confidence": 0.90},
        plateau_epsilon=0.02,
        saturation_missing_mass=0.10,
    )
    status = primary["prompt_evolution_status"]
    assert status["value_status"] == "UNRESOLVED"
    assert status["formal_mathematical_value_status"] == "UNRESOLVED"
    assert "VALUE_FORMAL_ONLY" not in status["headline_status"]
    combined = primary["reporting_tiers"]["combined_reporting_status"]
    assert combined["value_status"] == "UNRESOLVED"
    assert "FORMAL_CERTIFICATE_ONLY" not in combined["conclusions"]


def test_bad_mcq_panel_keeps_90_percent_value_direction_formal_and_non_headline():
    primary = _status_certificate((0.11, 0.20), (0.0, 0.03))
    sensitivity = _status_certificate((0.12, 0.18), (0.0, 0.015))
    for certificate in (primary, sensitivity):
        certificate["all_finite_prompt_certificate"] = {
            "instrument_quality": {
                "headline_eligible": False,
                "reasons": ["prior-degenerate panel"],
            },
        }
    attach_reporting_tiers(
        primary,
        sensitivity,
        primary_scope={"overall_simultaneous_confidence": 0.95},
        sensitivity_scope={"overall_simultaneous_confidence": 0.90},
        plateau_epsilon=0.02,
        saturation_missing_mass=0.10,
    )
    combined = primary["reporting_tiers"]["combined_reporting_status"]
    assert combined["value_status"] == "UNRESOLVED"
    assert combined["formal_mathematical_value_status"] == "SUGGESTIVE_PLATEAUED"
    assert "FORMAL_CERTIFICATE_ONLY" not in combined["conclusions"]


def test_numeric_reuse_rejects_pre_v12_or_unpinned_executor_manifest(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({
        "schema": "cr3-run-v11",
        "executor": "fake-executor",
        "dry_run": True,
    }))
    args = SimpleNamespace(
        executor="fake-executor",
        dry_run=True,
        worker="scripts/tools/cr3_mining_worker.py",
        worker_home=str(tmp_path),
    )
    with np.testing.assert_raises_regex(ValueError, "outside the current executor namespace"):
        validate_numeric_reuse_manifest(root, args, role="test reuse")


def test_publication_gate_fails_closed_when_iid_provenance_is_missing():
    gated = _apply_publication_gate({
        "headline_status": "CERTIFIED_PLATEAUED",
        "behavior_status": "UNRESOLVED",
        "value_status": "CERTIFIED_PLATEAUED",
    }, {})
    assert gated["headline_status"] == "SYNTHETIC_TEST_ONLY"
    assert gated["publication_eligible"] is False


def test_mcq_codebook_banks_reject_mixed_hierarchy_levels():
    _validate_level_matched_codebook_banks({
        "a": {"task": "humor", "level": "R3"},
        "b": {"task": "news-homepages", "level": "R2"},
        "c": {"task": "news-homepages", "level": "R2"},
    })
    with np.testing.assert_raises_regex(ValueError, "mixes hierarchy levels"):
        _validate_level_matched_codebook_banks({
            "a": {"task": "humor", "level": "R3"},
            "b": {"task": "humor", "level": "R2"},
        })
    with np.testing.assert_raises_regex(ValueError, "no explicit R1/R2/R3 level"):
        _validate_level_matched_codebook_banks({
            "a": {"task": "humor", "level": None},
            "b": {"task": "humor", "level": None},
        })
    _validate_level_matched_codebook_banks({
        "a": {"task": "humor", "level": None},
        "b": {"task": "humor", "level": None},
    }, allow_all_unknown=True)


def test_bound_grade_mcq_rejects_sampled_choice_readout_before_any_run():
    with np.testing.assert_raises(SystemExit):
        mining_main([
            "--metrics", "does-not-need-to-exist.npz",
            "--value-mode", "reconstruction_mcq",
            "--mcq-choice-readout", "sampled",
            "--dry-run",
        ])


def test_metric_task_accepts_every_existing_r3_bank_prefix_and_rejects_substrings():
    tasks = (
        "creative-writing",
        "humor",
        "news-homepages",
        "press-releases",
        "code-review",
        "math-stackexchange",
        "grant-funding",
        "peer-review",
        "legal-outcome-prediction",
    )
    for task in tasks:
        assert _metric_task(f"{task}_R3_metric7") == task

    try:
        _metric_task("prefix-creative-writing_R3_metric7")
    except ValueError:
        pass
    else:
        raise AssertionError("task inference must require an exact supported prefix")


def test_worker_environment_is_pinned_before_gpu_subprocess_start(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", "/afs/cs.stanford.edu/u/alexspan")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/afs/stale/triton")
    environment = _worker_environment(SimpleNamespace(worker_home=str(tmp_path)))
    assert environment["HOME"] == str(tmp_path.resolve())
    assert environment["METRIC_IMPLEMENTER_LFS_HOME"] == str(tmp_path.resolve())
    assert environment["XDG_CACHE_HOME"] == str(tmp_path.resolve() / ".cache")
    assert environment["TRITON_CACHE_DIR"] == str(tmp_path.resolve() / ".triton" / "cache")
    assert environment["VLLM_CONFIG_ROOT"] == str(tmp_path.resolve() / ".config" / "vllm")
    assert environment["VLLM_NO_USAGE_STATS"] == "1"
    assert os.environ["HOME"] == "/afs/cs.stanford.edu/u/alexspan"


def test_worker_retry_filter_is_narrow_to_vllm_memory_profile_race():
    transient = (
        "AssertionError: Error in memory profiling. Initial free memory 161.62 GiB, "
        "current free memory 162.24 GiB.")
    assert _retryable_worker_failure(transient)
    assert not _retryable_worker_failure("CUDA out of memory")
    assert not _retryable_worker_failure("PermissionError: inaccessible HOME")


def test_mcq_instrument_quality_separates_formal_bound_from_headline_gate():
    args = SimpleNamespace(
        mcq_min_headline_value_cap=0.10,
        mcq_min_headline_distractor_kappa=0.50,
        target_value_gap=0.02,
    )
    selected = [
        {"metric_key": "d1", "kappa": 0.20, "n_disagree": 5},
        {"metric_key": "d2", "kappa": 0.30, "n_disagree": 4},
        {"metric_key": "d3", "kappa": 0.40, "n_disagree": 3},
    ]
    state = {
        "fixed_no_demo_canonical_choice_probabilities": np.tile(
            np.asarray([[0.25, 0.25, 0.25, 0.25]]), (4, 1)),
        "value_cap": 0.60,
        "coarse_range_cap": 0.75,
        "finite_state_envelope": {
            "state_envelope_capability": {
                "has_positive_unique_target_maximizer": True,
            },
            "operational_target_diagnostic": {"value": 0.0, "is_headline_gate": False},
        },
        "mcq_codebook_entry": {
            "target_design_yes_rate": 0.5,
            "distractor_design_statistics": selected,
            "prior_calibration": {
                "passes_prior_balance": True,
                "prior": {"canonical_mean_prior": [0.25, 0.25, 0.25, 0.25]},
            },
        },
    }
    diagnostic = mcq_instrument_quality(state, args)
    assert diagnostic["status"] == "HEADLINE_ELIGIBLE"
    assert diagnostic["formal_all_prompt_bound_valid"] is True
    assert diagnostic["selected_distractor_kappa_is_headline_gate"] is False
    assert mcq_reported_global_status(
        "CERTIFIED_EPSILON_GLOBAL_OPTIMUM", diagnostic
    ) == "CERTIFIED_EPSILON_GLOBAL_OPTIMUM"

    state["finite_state_envelope"]["state_envelope_capability"][
        "has_positive_unique_target_maximizer"] = False
    diagnostic = mcq_instrument_quality(state, args)
    assert diagnostic["status"] == "FORMAL_CERTIFICATE_ONLY"
    assert diagnostic["headline_eligible"] is False
    assert len(diagnostic["reasons"]) == 1
    assert mcq_reported_global_status(
        "CERTIFIED_EPSILON_GLOBAL_OPTIMUM", diagnostic
    ) == "FORMAL_CERTIFICATE_ONLY"


def test_production_sampler_uses_unique_per_request_seeds_and_exact_quota():
    backend = _RecordingBackend()
    rows, attempts = draw_valid_rows(
        backend,
        "Propose one criterion?",
        n=40,
        base_seed=123,
        family="family_a",
        model="fake/model",
        temperature=0.9,
    )
    assert len(rows) == 40
    assert len({row["seed"] for row in rows}) == 40
    assert len(backend.seeds) == len(set(backend.seeds))
    assert len(attempts) >= len(rows)
    assert all(row["family"] == "family_a" for row in rows)


def test_proposal_transaction_is_bound_to_manifest_model_revision_and_config(tmp_path):
    rows, _ = draw_valid_rows(
        _RecordingBackend(),
        "Propose one criterion?",
        n=4,
        base_seed=123,
        family="family_a",
        model="fake/model",
        model_revision="resolved-revision",
        temperature=0.9,
    )
    path = tmp_path / "proposal.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _validate_proposal(
        path,
        "family_a",
        4,
        "atomic",
        expected_model="fake/model",
        expected_model_revision="resolved-revision",
        expected_temperature=0.9,
    )

    tampered = [dict(row) for row in rows]
    tampered[0]["model_revision"] = "other-revision"
    path.write_text("".join(json.dumps(row) + "\n" for row in tampered))
    with np.testing.assert_raises_regex(RuntimeError, "model revision changed"):
        _validate_proposal(
            path,
            "family_a",
            4,
            "atomic",
            expected_model="fake/model",
            expected_model_revision="resolved-revision",
            expected_temperature=0.9,
        )

    tampered = [dict(row) for row in rows]
    tampered[0]["generator_config_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in tampered))
    with np.testing.assert_raises_regex(RuntimeError, "configuration hash mismatch"):
        _validate_proposal(
            path,
            "family_a",
            4,
            "atomic",
            expected_model="fake/model",
            expected_model_revision="resolved-revision",
            expected_temperature=0.9,
        )


def test_reconstructor_artifact_contract_rejects_revision_or_readout_drift():
    manifest = {
        "mcq_reconstructor": "fake-reconstructor",
        "mcq_reconstructor_revision": "resolved-revision",
        "mcq_choice_readout_protocol": "choice-readout-v1",
    }
    payload = {
        "reconstructor_model": "fake-reconstructor",
        "reconstructor_revision": "resolved-revision",
        "choice_readout_id": "choice-readout-v1",
    }
    validate_reconstructor_artifact_contract(payload, manifest, role="test artifact")
    payload["reconstructor_revision"] = "other-revision"
    with np.testing.assert_raises_regex(RuntimeError, "frozen Reconstruction-MCQ namespace"):
        validate_reconstructor_artifact_contract(payload, manifest, role="test artifact")


def test_holistic_sampler_accepts_complete_long_rubrics_and_records_mode():
    rows, attempts = draw_valid_rows(
        _HolisticBackend(),
        "Write one full rubric",
        n=12,
        base_seed=321,
        family="family_holistic",
        model="fake/model",
        temperature=0.9,
        proposal_mode="holistic",
    )
    assert len(rows) == 12
    assert len(attempts) >= 12
    assert all(row["proposal_mode"] == "holistic" for row in rows)
    assert all(row["prompt_template_id"] == "holistic-rubric-v1-description" for row in rows)
    assert all(row["validator_id"] == "holistic-rubric-80-8000-v1" for row in rows)
    assert all(80 <= len(row["text"]) <= 8000 for row in rows)


def test_cr3_signature_uses_only_the_total_constrained_binary_readout():
    class Executor:
        def score_binary(self, *_args, **_kwargs):
            raise AssertionError("CR3 must not use the legacy top-logprob readout")

        def score_binary_constrained(self, prompts, *, pos, neg, seed):
            assert (pos, neg) == ("YES", "NO")
            assert len(prompts) == len(seed) == 3
            assert len(set(seed)) == 3
            return [0.1, 0.5, 0.9]

    signature = _checked_signature(
        Executor(), "Use a conflicting 0/1 rubric.", ["a", "b", "c"], 4000, "namespace")

    assert np.array_equal(signature, np.asarray([0.1, 0.5, 0.9]))
    assert "allowed-two-token" in READOUT_ID


def test_duplicate_prompt_text_is_scored_once_and_reuses_identical_signature():
    calls = []

    def score(text):
        calls.append(text)
        return np.asarray([len(text) % 2, 1, 0], float)

    criteria = [{"text": "same?"}, {"text": "other?"}, {"text": "same?"}]
    signatures, n_new = score_unique_texts(
        criteria, cache_namespace="task", score_fn=score, cache={})
    assert n_new == 2
    assert calls == ["same?", "other?"]
    assert np.array_equal(signatures[0], signatures[2])


def test_content_cache_reuses_signature_across_process_local_caches(tmp_path):
    calls = []

    def score(text):
        calls.append(text)
        return np.asarray([0.1, 0.9, 0.2], float)

    first, created = _content_cached_signature(
        str(tmp_path), "namespace", "criterion?", 3, score)
    second, created_again = _content_cached_signature(
        str(tmp_path), "namespace", "criterion?", 3,
        lambda _: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert created and not created_again
    assert calls == ["criterion?"]
    assert np.array_equal(first, second)


def _checkpoint(path):
    rng = np.random.default_rng(22)
    n_probes = 80
    target = np.tile([0, 1], n_probes // 2).astype(float)
    rows, tags, prompts = [], [], []
    for family in ("glm_a", "glm_b", "glm_c"):
        for i in range(6):
            col = target.copy() if i == 0 else rng.integers(0, 2, n_probes).astype(float)
            rows.append(col * 0.98 + 0.01)
            tags.append(family)
            prompts.append(f"{family} prompt {i}?")
    np.savez_compressed(
        path,
        sigs=np.asarray(rows),
        tags=np.asarray(tags, object),
        prompts=np.asarray(prompts, object),
        M_i=target,
        name=np.asarray("Synthetic metric"),
        description=np.asarray("Whether the synthetic target property holds"),
    )


def _small_checkpoint(path, target, index):
    rng = np.random.default_rng(100 + index)
    rows = [target * 0.98 + 0.01]
    rows.extend(rng.integers(0, 2, len(target)).astype(float) * 0.98 + 0.01 for _ in range(2))
    np.savez_compressed(
        path,
        sigs=np.asarray(rows),
        tags=np.asarray(["glm_a", "glm_b", "glm_c"], object),
        prompts=np.asarray([f"metric {index} seed {i}?" for i in range(3)], object),
        M_i=np.asarray(target, float),
        name=np.asarray(f"Synthetic metric {index}"),
        description=np.asarray(f"Whether target metric {index} holds"),
    )


def test_dry_loop_resume_never_absorbs_or_overwrites_confirmation(tmp_path):
    checkpoint = tmp_path / "creative-writing_metric0_sigs.npz"
    output = tmp_path / "run"
    _checkpoint(checkpoint)
    argv = [
        "--metrics", str(checkpoint),
        "--out-root", str(output),
        "--dry-run",
        "--batch-per-family", "4",
        "--confirm-per-family", "5",
        "--checkpoint-per-family", "5",
        "--checkpoint-iters", "0",
        "--study-alpha", "0.05",
        "--ceiling-horizon-per-family", "2",
        "--max-iter", "1",
        "--patience", "1",
    ]
    assert mining_main(argv) == 0
    metric_dir = output / "creative-writing_metric0"
    ledger_path = metric_dir / "absorption_ledger.jsonl"
    certificate_path = metric_dir / "confirmation" / "certificate.json"
    checkpoint_path = metric_dir / "checkpoint" / "iter_000" / "certificate.json"
    trajectory_path = metric_dir / "certified_trajectory.json"
    first_certificate = certificate_path.read_bytes()
    first_checkpoint = checkpoint_path.read_bytes()
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(ledger) == 1
    assert "confirmation" not in ledger[0]["scored_path"]
    payload = json.loads(certificate_path.read_text())
    assert payload["run"]["never_absorbed"] is True
    assert payload["publication_eligible"] is False
    assert payload["prompt_evolution_status"]["headline_status"] == "SYNTHETIC_TEST_ONLY"
    assert payload["prompt_evolution_status"]["publication_eligible"] is False
    assert payload["run"]["alpha_scope"]["scope"].startswith("familywise simultaneous")
    assert "prompt_evolution_status" in payload
    assert payload["estimand"]["prompt_class"] == (
        "single prompts in the frozen pool union the declared proposer-process support")
    global_cert = payload["all_finite_prompt_certificate"]
    assert global_cert["estimand"]["prompt_class"] == (
        "all finite prompts Sigma*; no prompt-length budget")
    assert global_cert["publication_eligible"] is False
    assert global_cert["certificate"]["status"] == "SYNTHETIC_TEST_ONLY"
    assert global_cert["certificate"]["synthetic_diagnostic_status"] == (
        "PROVABLY_OPTIMAL_DPI_ATTAINED_FIXED_PANEL")
    assert global_cert["certificate"]["certified_optimization_gap_UCB_bits"] == 0.0
    assert global_cert["proof_scope"]["population_exact_by_construction"] is False
    checkpoint_payload = json.loads(checkpoint_path.read_text())
    assert checkpoint_payload["run"]["never_absorbed"] is True
    assert checkpoint_payload["run"]["iterations_absorbed"] == 0
    trajectory = json.loads(trajectory_path.read_text())
    assert [point["phase"] for point in trajectory["points"]] == [
        "checkpoint", "final_confirmation"]
    assert trajectory["monitor_rows_are_certificates"] is False
    assert trajectory["publication_eligible"] is False

    # A completed resume is a no-op: no second absorption and no confirmation rewrite.
    assert mining_main(argv) == 0
    assert certificate_path.read_bytes() == first_certificate
    assert checkpoint_path.read_bytes() == first_checkpoint
    assert len(ledger_path.read_text().splitlines()) == 1


def test_dry_reconstruction_mcq_mode_values_every_prompt_and_uses_external_value_certificate(tmp_path):
    n = 80
    base = np.tile([0.0, 1.0], n // 2)
    metrics = []
    # The fifth metric is a codebook-only near-neighbor of metric 0. It must be
    # eligible as a hard distractor without becoming a mining target.
    flip_sets = [set(), {0, 1}, {2, 3}, {4, 5}, {0}]
    for index, residues in enumerate(flip_sets):
        target = base.copy()
        mask = np.asarray([i % 8 in residues for i in range(n)])
        target[mask] = 1.0 - target[mask]
        path = tmp_path / f"creative-writing_metric{index}_sigs.npz"
        _small_checkpoint(path, target, index)
        metrics.append(str(path))

    mining_targets = metrics[:4]

    output = tmp_path / "mcq_run"
    argv = [
        "--metrics", *mining_targets,
        "--mcq-codebook-metrics", *metrics,
        "--out-root", str(output),
        "--dry-run",
        "--value-mode", "reconstruction_mcq",
        "--batch-per-family", "2",
        "--confirm-per-family", "2",
        "--checkpoint-per-family", "2",
        "--ceiling-horizon-per-family", "1",
        "--max-iter", "1",
        "--patience", "1",
        "--mcq-n-options", "4",
        "--family-modes", "atomic", "holistic", "atomic",
        "--mcq-design-size", "40",
        "--mcq-n-examples", "8",
        "--mcq-reconstruction-draws", "4",
    ]
    assert mining_main(argv) == 0
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["family_modes"] == ["atomic", "holistic", "atomic"]
    assert run_manifest["family_model_revisions"] == run_manifest["families"]
    assert run_manifest["mcq_reconstructor_revision"] == run_manifest["mcq_reconstructor"]
    codebook = json.loads((output / "mcq_codebooks" / "creative-writing.json").read_text())
    assert codebook["premises"]["uses_external_labels"] is False
    assert len(codebook["metrics"]) == 5
    panel_plan = json.loads(
        (output / "mcq_codebooks" / "creative-writing.panel_plan.json").read_text())
    assert any(
        "creative-writing_metric4" in panel["distractor_metric_keys"]
        for panel in panel_plan["panels"]["creative-writing_metric0"]
    )
    assert (output / "mcq_codebooks" / "creative-writing.prior_calibration.json").exists()
    assert codebook["entries"]["creative-writing_metric0"]["prior_calibration"] is not None
    candidate = (output / "mcq_codebook_candidates" / "creative-writing_metric4"
                 / "bootstrap" / "scored.npz")
    assert candidate.exists()
    candidate_payload = np.load(candidate, allow_pickle=True)
    assert str(candidate_payload["schema"]) == "cr3-codebook-bootstrap-v1"
    assert candidate_payload["sigs"].shape == (1, n)
    qwen_proposal = json.loads(
        (output / "creative-writing_metric0" / "monitor" / "iter_000"
         / "proposal_qwen14.jsonl").read_text().splitlines()[0])
    assert qwen_proposal["proposal_mode"] == "holistic"
    bank_identity = json.loads((output / "mcq_identity_final.json").read_text())
    assert bank_identity["uses_external_labels"] is False
    assert bank_identity["status"] == "SYNTHETIC_TEST_ONLY"
    assert bank_identity["publication_eligible"] is False
    assert bank_identity["tasks"]["creative-writing"]["channels"]["annotations"]["valid"] is True
    assert "hierarchy_level" in bank_identity["tasks"]["creative-writing"]
    assert bank_identity["tasks"]["creative-writing"][
        "unfiltered_channels_publication_eligible"] is False
    assert bank_identity["tasks"]["creative-writing"]["headline_eligible_only"][
        "publication_eligible"] is False
    assert "headline_eligible_only" in bank_identity["tasks"]["creative-writing"]
    assert all(
        "headline_eligible" in selected
        for selected in bank_identity["selected_prompts"].values()
    )
    immutable_before_resume = {}
    for index in range(4):
        metric_dir = output / f"creative-writing_metric{index}"
        assert (metric_dir / "bootstrap" / "values.npz").exists()
        assert (metric_dir / "monitor" / "iter_000" / "values.npz").exists()
        certificate = json.loads((metric_dir / "confirmation" / "certificate.json").read_text())
        assert certificate["estimand"]["target_mode"] == (
            "supplied bounded anchor-free reconstruction values")
        assert certificate["estimand"]["value_name"] == (
            "annotation-attributable Reconstruction-MCQ target-option lift")
        global_certificate = certificate["all_finite_prompt_certificate"]
        assert global_certificate["synthetic_diagnostic_status"] in {
            "CERTIFIED_GLOBAL_GAP_BOUND",
            "CERTIFIED_EPSILON_GLOBAL_OPTIMUM",
            "CERTIFIED_GLOBAL_OPTIMUM",
        }
        assert global_certificate["status"] == "SYNTHETIC_TEST_ONLY"
        assert global_certificate["publication_eligible"] is False
        assert certificate["publication_eligible"] is False
        assert certificate["prompt_evolution_status"]["headline_status"] == (
            "SYNTHETIC_TEST_ONLY")
        assert np.isclose(
            global_certificate["coarse_no_demo_range_cap"],
            1.0 - global_certificate["no_demonstration_target_probability"])
        assert global_certificate["anchor_free_global_upper_bound"] <= (
            global_certificate["coarse_no_demo_range_cap"] + 1e-12)
        assert global_certificate["n_fixed_teaching_items"] == 8
        assert global_certificate["n_exhaustive_binary_states"] == 256
        assert (output / "mcq_state_tables" / f"creative-writing_metric{index}"
                / "envelope.json").exists()
        assert global_certificate["identified_interval"][0] <= (
            global_certificate["identified_interval"][1])
        assert global_certificate["best_evaluated_lower_bound"] == max(
            global_certificate["absorbed_pool_best_lower_bound"],
            global_certificate["current_audit_best_lower_bound"],
        )
        assert global_certificate["instrument_quality"]["formal_all_prompt_bound_valid"] is True
        assert certificate["scope"]["external_supervision_used"] is False
        ledger = json.loads((metric_dir / "absorption_ledger.jsonl").read_text())
        assert "value_path" in ledger and "value_sha256" in ledger
        immutable_before_resume[index] = {
            "certificate": (metric_dir / "confirmation" / "certificate.json").read_bytes(),
            "ledger": (metric_dir / "absorption_ledger.jsonl").read_bytes(),
        }

    # Resume must reconstruct the valued pool from its ledger without changing any
    # adaptive or confirmation artifact.
    assert mining_main(argv) == 0
    for index in range(4):
        metric_dir = output / f"creative-writing_metric{index}"
        assert (metric_dir / "confirmation" / "certificate.json").read_bytes() == (
            immutable_before_resume[index]["certificate"])
        assert (metric_dir / "absorption_ledger.jsonl").read_bytes() == (
            immutable_before_resume[index]["ledger"])

    # A separate run may hard-link the validated canonical candidate bank without
    # admitting any of those candidate-only prompt pools into its target search.
    reused_output = tmp_path / "mcq_reused"
    reused_argv = list(argv)
    reused_argv[reused_argv.index(str(output))] = str(reused_output)
    reused_argv.extend(["--reuse-mcq-codebook-root", str(output)])
    assert mining_main(reused_argv) == 0
    for index in range(5):
        original = (output / "mcq_codebook_candidates" / f"creative-writing_metric{index}"
                    / "bootstrap" / "scored.npz")
        reused = (reused_output / "mcq_codebook_candidates"
                  / f"creative-writing_metric{index}" / "bootstrap" / "scored.npz")
        assert original.stat().st_ino == reused.stat().st_ino

    # Expensive historical generations can enter a new run only as achieved-value
    # candidates. They precede the adaptive ledger and never count as confirmation.
    evidence_root = tmp_path / "evidence"
    build_evidence_store([output], evidence_root)
    evidence_output = tmp_path / "mcq_evidence_reused"
    evidence_argv = list(argv)
    evidence_argv[evidence_argv.index(str(output))] = str(evidence_output)
    evidence_argv.extend(["--reuse-evidence-root", str(evidence_root)])
    assert mining_main(evidence_argv) == 0
    for index in range(4):
        metric_dir = evidence_output / f"creative-writing_metric{index}"
        imported = json.loads((metric_dir / "historical" / "import.json").read_text())
        assert imported["evidence_role"] == "candidate_only"
        assert imported["eligible_as_fresh_audit"] is False
        assert (metric_dir / "historical" / "values.npz").exists()
        ledger = json.loads((metric_dir / "absorption_ledger.jsonl").read_text())
        assert ledger["pool_n_before"] == 3 + imported["n_candidates"]
        certificate = json.loads((metric_dir / "confirmation" / "certificate.json").read_text())
        assert certificate["run"]["never_absorbed"] is True
        assert "historical" not in certificate["run"]["confirmation_scored_path"]

    # Resume reconstructs bootstrap + historical candidates + adaptive ledger in
    # the same order and leaves the immutable confirmation untouched.
    before = (evidence_output / "creative-writing_metric0" / "confirmation"
              / "certificate.json").read_bytes()
    assert mining_main(evidence_argv) == 0
    assert (evidence_output / "creative-writing_metric0" / "confirmation"
            / "certificate.json").read_bytes() == before
