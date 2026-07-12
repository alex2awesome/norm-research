"""Production-path invariants for CR-3 sampling and resume behavior."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from methods.metric_implementer.experiments.run_cr3_mining_loop import (
    _metric_task,
    _retryable_worker_failure,
    _worker_environment,
    main as mining_main,
    mcq_instrument_quality,
)
from methods.metric_implementer.experiments.cr3_evidence_store import build_evidence_store
from scripts.tools.cr3_mining_worker import (
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
    )
    selected = [
        {"metric_key": "d1", "kappa": 0.20, "n_disagree": 5},
        {"metric_key": "d2", "kappa": 0.30, "n_disagree": 4},
        {"metric_key": "d3", "kappa": 0.40, "n_disagree": 3},
    ]
    state = {
        "fixed_no_demo_canonical_choice_probabilities": np.tile(
            np.asarray([[0.99, 0.005, 0.003, 0.002]]), (4, 1)),
        "value_cap": 0.01,
        "mcq_codebook_entry": {
            "target_design_yes_rate": 0.5,
            "distractor_design_statistics": selected,
            "prior_calibration": {
                "passes_prior_balance": True,
                "prior": {"canonical_mean_prior": [0.99, 0.005, 0.003, 0.002]},
            },
        },
    }
    diagnostic = mcq_instrument_quality(state, args)
    assert diagnostic["status"] == "FORMAL_CERTIFICATE_ONLY"
    assert diagnostic["formal_all_prompt_bound_valid"] is True
    assert len(diagnostic["reasons"]) == 2

    state["fixed_no_demo_canonical_choice_probabilities"] = np.tile(
        np.asarray([[0.25, 0.25, 0.25, 0.25]]), (4, 1))
    state["value_cap"] = 0.75
    state["mcq_codebook_entry"]["prior_calibration"]["prior"][
        "canonical_mean_prior"] = [0.25, 0.25, 0.25, 0.25]
    for row in selected:
        row["kappa"] += 0.50
    diagnostic = mcq_instrument_quality(state, args)
    assert diagnostic["status"] == "HEADLINE_ELIGIBLE"
    assert diagnostic["headline_eligible"] is True


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
    assert payload["run"]["alpha_scope"]["scope"].startswith("familywise simultaneous")
    assert "prompt_evolution_status" in payload
    assert payload["estimand"]["prompt_class"] == (
        "single prompts in the frozen pool union the declared proposer-process support")
    global_cert = payload["all_finite_prompt_certificate"]
    assert global_cert["estimand"]["prompt_class"] == (
        "all finite prompts Sigma*; no prompt-length budget")
    assert global_cert["certificate"]["status"] == (
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
        "--mcq-n-examples", "4",
        "--mcq-reconstruction-draws", "4",
    ]
    assert mining_main(argv) == 0
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["family_modes"] == ["atomic", "holistic", "atomic"]
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
    assert bank_identity["tasks"]["creative-writing"]["channels"]["annotations"]["valid"] is True
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
        assert global_certificate["status"] == "CERTIFIED_GLOBAL_GAP_BOUND"
        assert np.isclose(
            global_certificate["anchor_free_global_upper_bound"],
            1.0 - global_certificate["no_demonstration_target_probability"])
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
