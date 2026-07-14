from __future__ import annotations

import re
import json
from pathlib import Path

import numpy as np
import pytest

from methods.metric_implementer.config import ImplementerConfig
from methods.metric_implementer.experiments.cr3_evidence_store import EvidenceCellStore
from methods.metric_implementer.experiments.run_v14_value_campaign import (
    _metric_context,
    _qualification_panel,
    assert_gpu_authorized,
    build_designs,
    validate_metric_design,
)
from methods.metric_implementer.experiments.v14_behavioral_channel import (
    BEHAVIORAL_ARMS,
    evaluate_behavioral_state_tables_v14,
    no_verbatim_violations,
)
from methods.metric_implementer.experiments.v14_mcq_channel import (
    evaluate_mcq_state_tables_v14,
)
from methods.metric_implementer.experiments.v14_decoder_tuning import (
    template_sha256,
    tune_shared_template_batched,
)
from methods.metric_implementer.experiments.v14_preregistration import (
    evaluate_decoder_qualification,
    evaluate_sentinel_liveness,
)
from methods.metric_implementer.vllm_backend import FakeVLLM


def _one_panel_design(target):
    indices = list(range(8))
    bits = np.asarray(target)[indices].astype(int).tolist()
    return {
        "design_sha256": "d" * 64,
        "teaching_indices": list(range(8)),
        "panel_size": 8,
        "panels": [{
            "trial": 0, "indices": indices, "panel_sha256": "1" * 64,
            "decoder_family": "qwen", "target_state_bits": bits,
        }],
    }


class _PlantedConstructor:
    def __init__(self, target_by_text):
        self.target_by_text = target_by_text

    def generate_batch(self, prompts, **_kwargs):
        output = []
        for prompt in prompts:
            examples = re.findall(r"\[label=(\d)\]\n```\n(.*?)\n```", prompt, re.DOTALL)
            correct = bool(examples) and all(
                int(label) == self.target_by_text[text] for label, text in examples
            )
            output.append(
                "Does the item meet criterion alpha?" if correct
                else "Does the item meet criterion beta?"
            )
        return output


class _PlantedExecutor:
    def __init__(self, target_by_text):
        self.target_by_text = target_by_text

    def score_binary_constrained(self, prompts, **_kwargs):
        output = []
        for prompt in prompts:
            rule = re.search(r"Criterion:\n(.*?)\n\nText:", prompt, re.DOTALL).group(1)
            text = re.search(r"\n\nText:\n(.*?)\n\nDoes the text", prompt, re.DOTALL).group(1)
            output.append(float(self.target_by_text[text]) if "alpha" in rule else 0.0)
        return output


def test_behavioral_fake_backend_end_to_end_both_arms_and_cell_resume(tmp_path):
    texts = [f"ordinary sample {index}" for index in range(28)]
    target = np.asarray([(index % 2) for index in range(28)], dtype=int)
    target_by_text = dict(zip(texts, target.tolist()))
    design = _one_panel_design(target)
    embeddings = np.random.default_rng(14).normal(size=(len(texts), 12))
    with EvidenceCellStore(tmp_path / "cells.sqlite") as store:
        first = evaluate_behavioral_state_tables_v14(
            _PlantedConstructor(target_by_text), _PlantedExecutor(target_by_text),
            design_manifest=design, probe_texts=texts, heldout_indices=list(range(8, 28)),
            heldout_target=target[8:], noun="item", decoder_revision="fake-decoder",
            executor_revision="fake-executor", readout_id="fake-readout", store=store,
            probe_embeddings=embeddings,
        )
        before = store.count()
        second = evaluate_behavioral_state_tables_v14(
            _PlantedConstructor(target_by_text), _PlantedExecutor(target_by_text),
            design_manifest=design, probe_texts=texts, heldout_indices=list(range(8, 28)),
            heldout_target=target[8:], noun="item", decoder_revision="fake-decoder",
            executor_revision="fake-executor", readout_id="fake-readout", store=store,
            probe_embeddings=embeddings,
        )
        assert store.count() == before
    assert set(first["arms"]) == set(BEHAVIORAL_ARMS)
    for arm in BEHAVIORAL_ARMS:
        assert first["arms"][arm]["clipped_value"].shape == (1, 256)
        assert np.all(np.isfinite(first["arms"][arm]["raw_lift"]))
        assert first["arms"][arm]["near_raw_lift"].shape == (1, 256)
        assert first["arms"][arm]["far_raw_lift"].shape == (1, 256)
        assert np.array_equal(
            first["arms"][arm]["clipped_value"], second["arms"][arm]["clipped_value"]
        )


def test_mcq_fake_backend_end_to_end_and_non_disclosure(tmp_path):
    cfg = ImplementerConfig()
    cfg.vllm_fake = True
    backend = FakeVLLM("fake", "judge", cfg, 0.0)
    texts = [f"sample {index}" for index in range(20)]
    target = np.asarray([(index % 2) for index in range(20)], dtype=float)
    distractors = [
        {"metric_id": f"d{index}", "description": f"distractor {index}",
         "scores": np.roll(target, index + 1), "body": f"distractor {index}"}
        for index in range(3)
    ]
    with EvidenceCellStore(tmp_path / "mcq.sqlite") as store:
        result = evaluate_mcq_state_tables_v14(
            backend, design_manifest=_one_panel_design(target), noun="item",
            target_metric_id="target", target_description="target criterion",
            distractors=distractors, probe_texts=texts, constructor_revision="fake",
            store=store, n_reconstruction_draws=4, query_batch_size=512,
        )
    assert result["raw_lift"].shape == (1, 256)
    assert np.all(np.isfinite(result["clipped_value"]))
    assert result["non_disclosure"]["candidate_prompt_text_passed_to_query_builder"] is False


def test_gpu_guard_and_no_verbatim_hard_reject():
    with pytest.raises(RuntimeError, match="permanently forbidden"):
        assert_gpu_authorized([1], hostname="sk3", fake_backends=False)
    assert_gpu_authorized([0, 5], hostname="sk3", fake_backends=False)
    violations = no_verbatim_violations(
        "Does this contain alpha beta gamma delta epsilon zeta eta theta?",
        ["alpha beta gamma delta epsilon zeta eta theta extra"],
        corpus_token_counts={word: 10 for word in "alpha beta gamma delta epsilon zeta eta theta".split()},
    )
    assert "eight_word_demo_shingle" in violations


def test_qualification_and_sentinel_gate_only_declared_liveness_failures():
    qualification = evaluate_decoder_qualification([
        {"metric_key": str(index), "canonical_lift_bits": 0.1 if index < 4 else 0.0,
         "shuffled_lift_bits": 0.0}
        for index in range(6)
    ])
    assert qualification["passed"] is True
    live = evaluate_sentinel_liveness([{
        "metric_key": "m", "channel": "behavioral", "arm": "no_verbatim_examples",
        "structurally_valid": True, "planted_positive_value": 0.1,
        "degenerate_control_value": 0.0, "blind_value": 0.0,
        "annotated_canonical_value": 0.1, "cap": 0.2,
    }])
    assert live["passed"] is True
    dead = evaluate_sentinel_liveness([{
        "metric_key": "m", "channel": "mcq", "arm": None,
        "structurally_valid": True, "planted_positive_value": 0.0,
        "degenerate_control_value": 0.0, "blind_value": 0.2,
        "annotated_canonical_value": 0.1, "cap": 0.0,
    }])
    assert dead["fanout_blocked"] is True


def test_real_fixture_design_serializes_all_fifty_panels(tmp_path):
    fixture = (Path(__file__).parent / "fixtures" / "cr3_v12_humor_metric50_subset").resolve()
    manifest = {
        "schema": "cr3-value-bound-metrics-v13.1",
        "metrics": [{
            "task": "humor", "level": "R3", "metric": "50",
            "metric_key": "humor_R3_metric50",
            "codebook_path": str(fixture / "mcq_codebooks" / "humor.json"),
            "codebook_layout": "production", "assets_root": str(fixture),
            "candidate_bank_path": str(
                fixture / "humor_R3_metric50" / "historical" / "scored.npz"
            ),
        }],
    }
    manifest_path = tmp_path / "metrics.json"
    manifest_path.write_text(json.dumps(manifest))
    rows = build_designs(
        metrics_manifest_path=manifest_path, out_root=tmp_path / "out", run_sha="test-freeze",
    )
    assert len(rows) == 1
    validate_metric_design(rows[0])
    assert len(rows[0]["panel_design"]["panels"]) == 50
    context = _metric_context(rows[0])
    menu = context["menu"]
    cfg = ImplementerConfig()
    cfg.vllm_fake = True
    backend = FakeVLLM("fake", "judge", cfg, 0.0)
    with EvidenceCellStore(tmp_path / "real_metric_cells.sqlite") as store:
        result = evaluate_mcq_state_tables_v14(
            backend, design_manifest=_qualification_panel(rows[0], "qwen"),
            noun=str(context["codebook"]["reconstruction_noun"]),
            target_metric_id=str(rows[0]["metric_key"]),
            target_description=str(menu["entry"]["target_description"]),
            distractors=menu["distractors"], probe_texts=context["probe_texts"],
            constructor_revision="fake-real-metric", store=store,
            n_reconstruction_draws=4, query_batch_size=512,
        )
    state_path = tmp_path / "real_metric_one_panel.npz"
    np.savez_compressed(
        state_path, raw_lift=result["raw_lift"], clipped_value=result["clipped_value"],
    )
    with np.load(state_path, allow_pickle=False) as artifact:
        assert artifact["raw_lift"].shape == (1, 256)


def test_batched_gepa_is_hard_capped_at_four_rounds_and_eight_candidates():
    batch_sizes = []

    def propose(incumbent, _feedback, round_index, count):
        return [f"{incumbent} improve-{round_index}-{index}" for index in range(count)]

    def evaluate(candidates):
        batch_sizes.append(len(candidates))
        return {
            template_sha256(candidate): {
                "pooled_fitness": len(candidate) / 100.0,
                "heldout_prompt_transfer_ok": True,
                "far_near_transfer_ok": True,
                "feedback": [],
            }
            for candidate in candidates
        }

    result = tune_shared_template_batched(
        propose, evaluate,
        seed_template="Use {noun} {examples} {choices} {labels}",
        channel="mcq", arm="mcq", forbidden_strings=[],
        required_fields=("noun", "examples", "choices", "labels"),
    )
    assert batch_sizes == [1, 8, 8, 8, 8]
    assert result["round_cap"] == 4
    assert max(row["round"] for row in result["trace"]) == 4
