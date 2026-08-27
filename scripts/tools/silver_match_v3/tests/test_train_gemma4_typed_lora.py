from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from scripts.tools.silver_match_v3.build_gemma4_typed_dataset import structured_target
from scripts.tools.silver_match_v3.train_gemma4_typed_lora import (
    DEFAULT_FIELD_LOSS_WEIGHTS,
    audit_gradient_roles,
    checkpoint_selection_key,
    directory_ref,
    parse_args,
    resolve_language_model_target_scope,
    tokenize_example,
    tune_dev_confidence_threshold,
    validate_trainable_lora_scope,
    weighted_causal_lm_loss,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        prompt = f"USER:{messages[0]['content']}\nASSISTANT:"
        if add_generation_prompt:
            return prompt
        return prompt + messages[-1]["content"] + "<eos>"

    def __call__(self, text, **kwargs):
        values = [2 + (ord(char) % 97) for char in text]
        result = {"input_ids": values}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def _typed_row(*, uid="n1", group="g1", split="train", gradient=True):
    target, spans = structured_target(
        decision="MATCH",
        metric_id="a7",
        confidence="high",
        reason="The human explicitly names the criterion.",
    )
    return {
        "norm_uid": uid,
        "source_group": group,
        "split": split,
        "gradient_eligible": gradient,
        "view": "retrieval_order",
        "decision": "MATCH",
        "metric_id": "a7",
        "candidate_metric_ids": ["a7", "a9"],
        "target_field_char_spans": spans,
        "messages": [
            {"role": "user", "content": "choose one"},
            {"role": "assistant", "content": target},
        ],
    }


def test_v2_character_spans_become_exact_field_token_weights():
    row = _typed_row()
    encoded = tokenize_example(CharacterTokenizer(), row, 4096)
    target = row["messages"][-1]["content"]
    start = encoded["prompt_tokens"]
    assert all(label == -100 for label in encoded["labels"][:start])
    assert all(weight == 0 for weight in encoded["loss_weights"][:start])
    for field, span in row["target_field_char_spans"].items():
        observed = encoded["loss_weights"][start + span["start"] : start + span["end"]]
        assert observed and set(observed) == {DEFAULT_FIELD_LOSS_WEIGHTS[field]}
    # Braces and comma separators deliberately retain the structural weight.
    assert encoded["loss_weights"][start] == pytest.approx(0.25)
    assert encoded["loss_weights"][start + len(target) - 1] == pytest.approx(0.25)
    assert encoded["has_target_field_char_spans"] is True


def test_legacy_row_retains_uniform_assistant_only_loss():
    row = _typed_row()
    del row["target_field_char_spans"]
    encoded = tokenize_example(CharacterTokenizer(), row, 4096)
    assert set(encoded["loss_weights"][encoded["prompt_tokens"] :]) == {1.0}


def test_weighted_loss_excludes_prompt_and_final_nonpredicting_position_gradients():
    logits = torch.randn(1, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3]])
    weights = torch.tensor([[0.0, 0.0, 4.0, 0.25]])
    loss = weighted_causal_lm_loss(logits, labels, weights)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 0]).item() == 0
    assert torch.count_nonzero(logits.grad[0, 1]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 2]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 3]).item() == 0


def test_role_audit_excludes_train_opt_out_and_fails_on_heldout_leakage():
    eligible = _typed_row(uid="train", group="train-g")
    excluded = _typed_row(uid="skip", group="skip-g", gradient=False)
    dev = _typed_row(uid="dev", group="dev-g", split="dev", gradient=False)
    selected, report = audit_gradient_roles([eligible, excluded], [dev])
    assert [row["norm_uid"] for row in selected] == ["train"]
    assert report["explicit_train_rows_excluded"] == 1
    assert report["heldout_gradient_eligible_count"] == 0

    with pytest.raises(ValueError, match="source-group leakage"):
        audit_gradient_roles(
            [eligible], [_typed_row(uid="other", group="train-g", split="dev", gradient=False)]
        )
    with pytest.raises(ValueError, match="explicitly gradient-ineligible"):
        audit_gradient_roles(
            [eligible], [_typed_row(uid="other", group="dev-g", split="dev", gradient=True)]
        )


def test_dev_threshold_and_checkpoint_selection_are_dev_only():
    rows = [
        {"decision": "MATCH", "metric_id": "a1"},
        {"decision": "MATCH", "metric_id": "a2"},
        {"decision": "NO_CANDIDATE_FITS", "metric_id": None},
    ]
    predictions = [
        {"decision": "MATCH", "metric_id": "a1", "confidence": "high"},
        {"decision": "MATCH", "metric_id": "wrong", "confidence": "low"},
        {"decision": "NO_CANDIDATE_FITS", "metric_id": None, "confidence": "high"},
    ]
    tuned = tune_dev_confidence_threshold(
        rows,
        predictions,
        min_precision=0.9,
        min_wilson_lower=0.0,
        min_predictions=1,
    )
    assert tuned["minimum_confidence"] == "high"
    assert tuned["selection_split"] == "dev"
    first = {"cumulative_exposure": 10, "weighted_dev_loss": 1.0, "confidence_gate": tuned}
    second = {
        "cumulative_exposure": 20,
        "weighted_dev_loss": 0.5,
        "confidence_gate": {**tuned, "exact_f_beta_0_5": 0.1},
    }
    assert checkpoint_selection_key(first) > checkpoint_selection_key(second)


def test_new_recipe_defaults_and_exposure_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trainer",
            "--dataset",
            "train.jsonl",
            "--model",
            "model",
            "--report",
            "report.json",
            "--preflight-only",
            "--exposure-checkpoints",
            "100,250,500",
        ],
    )
    args = parse_args()
    assert (args.lora_r, args.lora_alpha, args.learning_rate) == (8, 16, 2e-5)
    assert args.exposure_checkpoints == (100, 250, 500)
    assert [
        args.decision_loss_weight,
        args.metric_id_loss_weight,
        args.confidence_loss_weight,
        args.reason_loss_weight,
    ] == [4.0, 4.0, 1.0, 0.25]


def test_directory_content_hash_and_qkvo_scope_contract(tmp_path: Path):
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "adapter_config.json").write_text(json.dumps({"r": 8}))
    first = directory_ref(root)
    assert first["file_count"] == 1
    (root / "adapter_config.json").write_text(json.dumps({"r": 16}))
    assert directory_ref(root)["content_manifest_sha256"] != first["content_manifest_sha256"]

    class FakeModel:
        def named_parameters(self):
            yield (
                "base_model.model.model.language_model.layers.0.self_attn.q_proj."
                "lora_A.default.weight",
                torch.nn.Parameter(torch.zeros(1), requires_grad=True),
            )

    names, modules = validate_trainable_lora_scope(
        FakeModel(), {"model.language_model.layers.0.self_attn.q_proj"}
    )
    assert len(names) == 1
    assert modules == {"model.language_model.layers.0.self_attn.q_proj"}


def test_llama_qkvo_scope_is_exactly_detected_and_validated():
    class FakeLlamaBase:
        def named_modules(self):
            yield "model.layers.0.self_attn.q_proj", torch.nn.Linear(2, 2)
            yield "model.layers.0.self_attn.k_proj", torch.nn.Linear(2, 2)
            yield "model.layers.0.mlp.up_proj", torch.nn.Linear(2, 2)

    architecture, regex, modules = resolve_language_model_target_scope(FakeLlamaBase())
    assert architecture == "llama_causal_language_model"
    assert regex.startswith("^model\\.layers")
    assert modules == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    }

    class FakePeftLlama:
        def named_parameters(self):
            for target in ("q_proj", "k_proj"):
                yield (
                    f"base_model.model.model.layers.0.self_attn.{target}."
                    "lora_A.default.weight",
                    torch.nn.Parameter(torch.zeros(1), requires_grad=True),
                )

    names, adapted = validate_trainable_lora_scope(FakePeftLlama(), modules)
    assert len(names) == 2
    assert adapted == modules


def test_gemma_qkvo_scope_autodetection_remains_unchanged():
    class FakeGemmaBase:
        def named_modules(self):
            yield (
                "model.language_model.layers.0.self_attn.o_proj",
                torch.nn.Linear(2, 2),
            )
            yield "model.vision_tower.layers.0.self_attn.o_proj", torch.nn.Linear(2, 2)

    architecture, regex, modules = resolve_language_model_target_scope(FakeGemmaBase())
    assert architecture == "gemma4_multimodal_language_model"
    assert "language_model" in regex
    assert modules == {"model.language_model.layers.0.self_attn.o_proj"}
