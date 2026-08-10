#!/usr/bin/env python3
"""Train one fixed, task-local Gemma-4 LoRA on typed silver-match decisions.

This deliberately avoids TRL and ``datasets`` so the training runtime is
small and auditable.  Loss is applied only to the final assistant turn.  A
preflight-only mode performs the exact chat-template tokenization and refuses
examples whose target would require truncation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import socket
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
GEMMA_LANGUAGE_MODEL_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
)
LLAMA_LANGUAGE_MODEL_TARGET_REGEX = (
    r"^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
)
# Backward-compatible public constant for the original Gemma-only contract.
LANGUAGE_MODEL_TARGET_REGEX = GEMMA_LANGUAGE_MODEL_TARGET_REGEX
TARGET_SCOPE_REGEXES = {
    "gemma4_multimodal_language_model": GEMMA_LANGUAGE_MODEL_TARGET_REGEX,
    "llama_causal_language_model": LLAMA_LANGUAGE_MODEL_TARGET_REGEX,
}
FIELD_NAMES = ("decision", "metric_id", "confidence", "reason")
DEFAULT_FIELD_LOSS_WEIGHTS = {
    "decision": 4.0,
    "metric_id": 4.0,
    "confidence": 1.0,
    "reason": 0.25,
}
DEFAULT_STRUCTURAL_LOSS_WEIGHT = 0.25
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
TYPED_DECISIONS = {
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def directory_ref(path: Path) -> dict[str, Any]:
    """Return an exact, portable content manifest for a directory tree."""

    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        files.append(
            {
                "relative_path": child.relative_to(path).as_posix(),
                "sha256": sha256_file(child),
                "bytes": child.stat().st_size,
            }
        )
    if not files:
        raise ValueError(f"directory contains no files: {path}")
    canonical = json.dumps(files, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return {
        "path": str(path),
        "content_manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "file_count": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def read_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) < 2
                or messages[-1].get("role") != "assistant"
                or not str(messages[-1].get("content") or "").strip()
            ):
                raise ValueError(f"invalid messages at dataset line {line_number}")
            examples.append(row)
    if not examples:
        raise ValueError("empty training dataset")
    return examples


def audit_gradient_roles(
    train_examples: Sequence[dict[str, Any]],
    dev_examples: Sequence[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail closed on role leakage and return gradient-eligible train rows.

    Legacy v1 train files did not carry ``split``.  They remain accepted when
    no dev file is supplied.  A v2 train/dev run, however, must provide a
    non-empty source group for every row so source disjointness is provable.
    """

    eligible: list[dict[str, Any]] = []
    excluded = 0
    strict_v2 = bool(dev_examples)
    for index, row in enumerate(train_examples, 1):
        split = str(row.get("split") or "").strip().lower()
        if (strict_v2 and split != "train") or (split and split != "train"):
            raise ValueError(f"held-out row in gradient dataset at line {index}: {split}")
        gradient = row.get("gradient_eligible")
        if gradient is not None and not isinstance(gradient, bool):
            raise ValueError(f"gradient_eligible is not boolean at train line {index}")
        if strict_v2 and gradient is None:
            raise ValueError(f"v2 train row lacks gradient_eligible at line {index}")
        if gradient is False:
            excluded += 1
            continue
        eligible.append(row)
    if not eligible:
        raise ValueError("training dataset has no gradient-eligible rows")

    for index, row in enumerate(dev_examples, 1):
        split = str(row.get("split") or "").strip().lower()
        if split != "dev":
            raise ValueError(f"non-dev row in dev dataset at line {index}: {split}")
        if row.get("gradient_eligible") is not False:
            raise ValueError(f"dev row is not explicitly gradient-ineligible at line {index}")

    train_uids = {
        str(row.get("norm_uid") or "").strip() for row in train_examples
    }
    gradient_uids = {str(row.get("norm_uid") or "").strip() for row in eligible}
    dev_uids = {str(row.get("norm_uid") or "").strip() for row in dev_examples}
    if "" in train_uids or "" in dev_uids:
        raise ValueError("dataset role audit requires non-empty norm_uid")
    uid_overlap = sorted(train_uids & dev_uids)
    if uid_overlap:
        raise ValueError(f"train/dev norm UID leakage: {uid_overlap[:10]}")

    train_groups = {
        str(row.get("source_group") or "").strip() for row in train_examples
    }
    dev_groups = {str(row.get("source_group") or "").strip() for row in dev_examples}
    if dev_examples and ("" in train_groups or "" in dev_groups):
        raise ValueError("source-disjoint train/dev audit requires every source_group")
    uid_groups: dict[str, str] = {}
    for row in (*train_examples, *dev_examples):
        uid = str(row.get("norm_uid") or "").strip()
        group = str(row.get("source_group") or "").strip()
        prior = uid_groups.setdefault(uid, group)
        if prior != group:
            raise ValueError(f"norm UID maps to multiple source groups: {uid}")
    group_overlap = sorted((train_groups - {""}) & (dev_groups - {""}))
    if group_overlap:
        raise ValueError(f"train/dev source-group leakage: {group_overlap[:10]}")
    return eligible, {
        "status": "PASS_SOURCE_DISJOINT_HELDOUT_GRADIENT_EXCLUDED",
        "input_train_rows": len(train_examples),
        "gradient_train_rows": len(eligible),
        "explicit_train_rows_excluded": excluded,
        "dev_rows": len(dev_examples),
        "train_norm_uids": len(train_uids),
        "gradient_train_norm_uids": len(gradient_uids),
        "dev_norm_uids": len(dev_uids),
        "train_source_groups": len(train_groups - {""}),
        "dev_source_groups": len(dev_groups - {""}),
        "norm_uid_overlap_count": 0,
        "source_group_overlap_count": 0,
        "heldout_gradient_eligible_count": 0,
        "legacy_missing_source_groups_accepted_without_dev": bool(
            not dev_examples and "" in train_groups
        ),
    }


def _token_list(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("unexpected batched chat-template result")
        value = value[0]
    return [int(token) for token in value]


def _validated_target_spans(
    row: Mapping[str, Any], assistant_content: str
) -> dict[str, tuple[int, int]] | None:
    raw = row.get("target_field_char_spans")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != set(FIELD_NAMES):
        raise ValueError(
            f"target_field_char_spans must cover exactly {FIELD_NAMES}: "
            f"{row.get('norm_uid')}/{row.get('view')}"
        )
    spans: dict[str, tuple[int, int]] = {}
    prior_end = -1
    for field in FIELD_NAMES:
        value = raw[field]
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid target span object: {field}")
        try:
            start, end = int(value["start"]), int(value["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid target span coordinates: {field}") from exc
        if start < 0 or end <= start or end > len(assistant_content) or start < prior_end:
            raise ValueError(f"invalid/overlapping target span: {field}/{start}:{end}")
        spans[field] = (start, end)
        prior_end = end
    return spans


def tokenize_example(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    *,
    field_loss_weights: Mapping[str, float] | None = None,
    structural_loss_weight: float = DEFAULT_STRUCTURAL_LOSS_WEIGHT,
) -> dict[str, Any]:
    """Render a conversation and construct assistant-only weighted targets.

    The v2 builder supplies character spans relative to the raw assistant JSON.
    Fast-tokenizer offsets are measured over the exact inference prefix plus
    that JSON, so a token receives the maximum weight of every field it
    overlaps.  This is conservative at subword boundaries and never assigns a
    held-out/prompt token a gradient.
    """

    weights_by_field = dict(field_loss_weights or DEFAULT_FIELD_LOSS_WEIGHTS)
    if set(weights_by_field) != set(FIELD_NAMES) or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in weights_by_field.values()
    ):
        raise ValueError(f"field loss weights must be finite positive values for {FIELD_NAMES}")
    if not math.isfinite(structural_loss_weight) or structural_loss_weight <= 0.0:
        raise ValueError("structural loss weight must be finite and positive")
    messages = row["messages"]
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    canonical_full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    uid = str(row.get("norm_uid") or "")
    view = str(row.get("view") or "")
    if not isinstance(prompt_text, str) or not isinstance(canonical_full_text, str):
        raise ValueError(f"chat template did not render text: {uid}/{view}")
    assistant_content = str(messages[-1]["content"])
    target_spans = _validated_target_spans(row, assistant_content)
    content_start = canonical_full_text.rfind(assistant_content)
    if content_start < 0:
        raise ValueError(f"assistant content absent from canonical rendering: {uid}/{view}")
    canonical_suffix = canonical_full_text[content_start + len(assistant_content) :]
    if not prompt_text or not canonical_suffix:
        raise ValueError(f"empty inference prefix or canonical assistant suffix: {uid}/{view}")
    # Gemma-4's inference generation prefix includes its configured channel
    # header, while rendering an already-populated assistant message omits
    # that header.  Training must match the actual vLLM ``chat`` inference
    # prefix, so concatenate that exact prefix with the target and only reuse
    # the canonical assistant turn terminator from the populated rendering.
    full_text = prompt_text + assistant_content + canonical_suffix
    rendered_target = full_text[len(prompt_text) :]
    if not rendered_target.startswith(assistant_content):
        raise ValueError(f"assistant content does not begin at rendered boundary: {uid}/{view}")

    # Tokenize the complete rendered string once and locate the character
    # boundary with fast-tokenizer offsets.  Encoding the prompt and full text
    # separately is not safe: SentencePiece/BPE can retokenize several tokens
    # at the concatenation boundary.  If one token straddles the boundary it is
    # conservatively assigned to the target, which loses no assistant signal.
    encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = _token_list(encoding)
    offsets = encoding.get("offset_mapping")
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        if len(offsets) != 1:
            raise ValueError(f"unexpected batched offsets: {uid}/{view}")
        offsets = offsets[0]
    offsets = [(int(start), int(end)) for start, end in offsets]
    if not full_ids or len(offsets) != len(full_ids):
        raise ValueError(f"empty template tokenization: {uid}/{view}")
    boundary = len(prompt_text)
    target_start = next(
        (index for index, (_start, end) in enumerate(offsets) if end > boundary),
        None,
    )
    if target_start is None:
        raise ValueError(f"could not locate assistant offset boundary: {uid}/{view}")
    target_tokens = len(full_ids) - target_start
    if target_tokens <= 0:
        raise ValueError(f"assistant target has no tokens: {uid}/{view}")
    if len(full_ids) > max_length:
        raise ValueError(
            f"example exceeds max_length without safe target-preserving truncation: "
            f"{uid}/{view} full={len(full_ids)} max={max_length}"
        )
    labels = [-100] * target_start + full_ids[target_start:]
    if len(labels) != len(full_ids) or not any(value != -100 for value in labels):
        raise AssertionError(f"loss-mask construction failed: {uid}/{view}")
    loss_weights = [0.0] * target_start + [float(structural_loss_weight)] * target_tokens
    field_token_counts: Counter[str] = Counter()
    if target_spans is None:
        # Legacy v1 examples did not expose field spans.  Preserve their
        # historical uniform assistant-only objective exactly.
        loss_weights[target_start:] = [1.0] * target_tokens
    else:
        absolute_spans = {
            field: (boundary + start, boundary + end)
            for field, (start, end) in target_spans.items()
        }
        for index in range(target_start, len(offsets)):
            token_start, token_end = offsets[index]
            touched = [
                field
                for field, (start, end) in absolute_spans.items()
                if token_end > start and token_start < end
            ]
            if touched:
                loss_weights[index] = max(float(weights_by_field[field]) for field in touched)
                field_token_counts.update(touched)
        untouched = sorted(set(FIELD_NAMES) - set(field_token_counts))
        if untouched:
            raise ValueError(f"target spans mapped to no tokens: {uid}/{view}/{untouched}")
    if len(loss_weights) != len(labels) or any(
        weight != 0.0 for weight, label in zip(loss_weights, labels) if label == -100
    ):
        raise AssertionError(f"weighted loss mask construction failed: {uid}/{view}")

    prompt_encoding = tokenizer(prompt_text, add_special_tokens=False)
    prompt_input_ids = _token_list(prompt_encoding)
    if not prompt_input_ids:
        raise ValueError(f"inference prompt tokenization is empty: {uid}/{view}")
    return {
        "input_ids": full_ids,
        "labels": labels,
        "loss_weights": loss_weights,
        "prompt_input_ids": prompt_input_ids,
        "length": len(full_ids),
        "prompt_tokens": target_start,
        "target_tokens": target_tokens,
        "norm_uid": uid,
        "view": view,
        "decision": str(row.get("decision") or ""),
        "metric_id": row.get("metric_id"),
        "candidate_metric_ids": list(row.get("candidate_metric_ids") or []),
        "source_group": str(row.get("source_group") or ""),
        "split": str(row.get("split") or ""),
        "has_target_field_char_spans": target_spans is not None,
        "field_weight_token_counts": dict(sorted(field_token_counts.items())),
    }


def tokenize_dataset(
    tokenizer: Any,
    examples: Sequence[dict[str, Any]],
    max_length: int,
    *,
    field_loss_weights: Mapping[str, float] | None = None,
    structural_loss_weight: float = DEFAULT_STRUCTURAL_LOSS_WEIGHT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = [
        tokenize_example(
            tokenizer,
            row,
            max_length,
            field_loss_weights=field_loss_weights,
            structural_loss_weight=structural_loss_weight,
        )
        for row in examples
    ]
    lengths = [row["length"] for row in encoded]
    prompt_lengths = [row["prompt_tokens"] for row in encoded]
    target_lengths = [row["target_tokens"] for row in encoded]
    unique_uids = {row["norm_uid"] for row in encoded}
    view_counts = Counter(row["view"] for row in encoded)
    decision_counts = Counter(row["decision"] for row in encoded)
    weighted_tokens = [sum(row["loss_weights"]) for row in encoded]
    span_rows = sum(row["has_target_field_char_spans"] for row in encoded)
    field_token_counts: Counter[str] = Counter()
    for row in encoded:
        field_token_counts.update(row["field_weight_token_counts"])
    return encoded, {
        "example_count": len(encoded),
        "unique_norm_uid_count": len(unique_uids),
        "view_counts": dict(sorted(view_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "target_field_span_rows": span_rows,
        "legacy_uniform_loss_rows": len(encoded) - span_rows,
        "field_weight_token_counts": dict(sorted(field_token_counts.items())),
        "weighted_loss_token_mass": sum(weighted_tokens),
        "tokens": {
            "full_min": min(lengths),
            "full_max": max(lengths),
            "full_mean": sum(lengths) / len(lengths),
            "prompt_min": min(prompt_lengths),
            "prompt_max": max(prompt_lengths),
            "target_min": min(target_lengths),
            "target_max": max(target_lengths),
            "target_mean": sum(target_lengths) / len(target_lengths),
            "total_loss_tokens": sum(target_lengths),
        },
    }


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen report: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_metadata() -> dict[str, Any]:
    import torch
    import transformers

    try:
        import accelerate
        accelerate_version = accelerate.__version__
    except Exception:
        accelerate_version = None
    try:
        import peft
        peft_version = peft.__version__
    except Exception:
        peft_version = None
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft_version,
        "accelerate": accelerate_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def fixed_recipe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "epochs": args.epochs,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "scheduler": "constant",
        "bf16": True,
        "max_length": args.max_length,
        "seed": args.seed,
        "shuffle": "one deterministic torch.randperm per epoch",
        "gradient_checkpointing": True,
        "assistant_only_loss": True,
        "field_token_loss_weights": {
            field: float(getattr(args, f"{field}_loss_weight")) for field in FIELD_NAMES
        },
        "structural_loss_weight": args.structural_loss_weight,
        "exposure_checkpoints": list(args.exposure_checkpoints),
        "checkpoint_selection": (
            "dev_only_confidence_gate_then_f0.5_then_weighted_dev_loss"
            if args.dev_dataset
            else "last_checkpoint_no_dev_dataset"
        ),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(TARGET_MODULES),
            "target_scope_regex": "AUTO_DETECT_EXACT_ARCHITECTURE_SCOPE",
            "allowed_exact_target_scopes": dict(TARGET_SCOPE_REGEXES),
            "expected_adapted_linear_count": "derived_exactly_from_loaded_base_model",
        },
    }


def language_model_target_modules(
    model: Any, target_regex: str = LANGUAGE_MODEL_TARGET_REGEX
) -> set[str]:
    return {
        name
        for name, module in model.named_modules()
        if re.fullmatch(target_regex, name)
        and isinstance(module, torch.nn.Linear)
    }


def resolve_language_model_target_scope(
    model: Any,
) -> tuple[str, str, set[str]]:
    """Select exactly one known architecture's complete q/k/v/o inventory."""

    matches = {
        architecture: language_model_target_modules(model, target_regex)
        for architecture, target_regex in TARGET_SCOPE_REGEXES.items()
    }
    populated = {name: modules for name, modules in matches.items() if modules}
    if len(populated) != 1:
        counts = {name: len(modules) for name, modules in matches.items()}
        raise RuntimeError(
            "base model must expose exactly one supported q/k/v/o scope; "
            f"observed={counts}"
        )
    architecture, modules = next(iter(populated.items()))
    return architecture, TARGET_SCOPE_REGEXES[architecture], modules


def validate_trainable_lora_scope(
    model: Any, expected_modules: set[str] | None = None
) -> tuple[list[str], set[str]]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if (
        not names
        or any("lora_" not in name for name in names)
        or any(".vision_tower." in name or ".audio_tower." in name for name in names)
        or any(
            not any(f".{target}." in name for target in TARGET_MODULES) for name in names
        )
    ):
        raise RuntimeError("trainable parameters escaped the text-language-model LoRA scope")
    adapted_modules = set()
    for name in names:
        base = name.split(".lora_", 1)[0]
        matching = [module for module in expected_modules or () if base.endswith(module)]
        if len(matching) != 1:
            raise RuntimeError(f"cannot canonicalize adapted text module: {name}")
        adapted_modules.add(matching[0])
    if expected_modules is not None and adapted_modules != expected_modules:
        missing = sorted(expected_modules - adapted_modules)
        extra = sorted(adapted_modules - expected_modules)
        raise RuntimeError(
            "adapted text-linear set differs from exact base-model q/k/v/o target set: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    if not adapted_modules:
        raise RuntimeError("no q/k/v/o text-language-model modules were adapted")
    return names, adapted_modules


def adapter_injection_preflight(args: argparse.Namespace, base_report: dict[str, Any]) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    target_architecture, target_regex, expected_modules = (
        resolve_language_model_target_scope(model)
    )
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_regex,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable_names, adapted_modules = validate_trainable_lora_scope(
        model, expected_modules
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    report = {
        **base_report,
        "schema_version": "silver-match-v3-gemma4-lora-injection-preflight-v1",
        "status": "PASS_TEXT_LANGUAGE_MODEL_LORA_SCOPE_ONLY",
        "adapter_injection": {
            "adapted_linear_count": len(adapted_modules),
            "expected_base_qkvo_linear_count": len(expected_modules),
            "target_architecture": target_architecture,
            "target_scope_regex": target_regex,
            "trainable_tensor_count": len(trainable_names),
            "trainable_parameter_count": trainable_parameters,
            "all_trainable_parameters_are_lora": True,
            "vision_or_audio_trainable_parameters": 0,
            "sample_adapted_modules": sorted(adapted_modules)[:10],
        },
    }
    write_json_new(Path(args.report), report)
    print(json.dumps({"status": report["status"], "report": file_ref(Path(args.report))}), flush=True)


def collate_batch(rows: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    labels = []
    loss_weights = []
    attention_mask = []
    for row in rows:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        loss_weights.append(row["loss_weights"] + [0.0] * pad)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def weighted_causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_weights: torch.Tensor
) -> torch.Tensor:
    """Weighted next-token cross entropy with exact prompt/pad exclusion."""

    if logits.ndim != 3 or labels.ndim != 2 or loss_weights.shape != labels.shape:
        raise ValueError("expected logits [B,T,V] and aligned labels/loss_weights [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels do not align")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    shifted_weights = loss_weights[:, 1:].to(dtype=torch.float32)
    valid = shifted_labels.ne(-100)
    if bool(torch.any(shifted_weights.masked_select(~valid) != 0).item()):
        raise ValueError("non-target token carries non-zero loss weight")
    effective = shifted_weights * valid.to(shifted_weights.dtype)
    denominator = effective.sum()
    if not bool(denominator > 0):
        raise ValueError("batch has no positive weighted target mass")
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_loss = torch.nn.functional.cross_entropy(
        shifted_logits.transpose(1, 2), safe_labels, reduction="none"
    )
    return (token_loss * effective).sum() / denominator


def _parse_exposure_checkpoints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
        try:
            values = tuple(int(part) for part in raw)
        except ValueError as exc:
            raise ValueError("exposure checkpoints must be comma-separated integers") from exc
    else:
        values = tuple(int(part) for part in value)
    if any(value <= 0 for value in values) or tuple(sorted(set(values))) != values:
        raise ValueError("exposure checkpoints must be unique, positive, and increasing")
    return values


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total
    ) / denominator
    return max(0.0, center - radius)


def tune_dev_confidence_threshold(
    gold_rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any] | None],
    *,
    min_precision: float,
    min_wilson_lower: float,
    min_predictions: int,
) -> dict[str, Any]:
    """Choose the MATCH confidence gate using dev labels only."""

    if len(gold_rows) != len(predictions) or not gold_rows:
        raise ValueError("gold rows and predictions must be aligned and non-empty")
    if not 0 <= min_precision <= 1 or not 0 <= min_wilson_lower <= 1:
        raise ValueError("precision gates must be in [0,1]")
    reports = []
    gold_exact = sum(row["decision"] == "MATCH" for row in gold_rows)
    for confidence, rank in CONFIDENCE_RANK.items():
        accepted = correct = 0
        for gold, prediction in zip(gold_rows, predictions):
            predicts_match = bool(
                prediction
                and prediction.get("decision") == "MATCH"
                and CONFIDENCE_RANK.get(str(prediction.get("confidence")), -1) >= rank
            )
            if predicts_match:
                accepted += 1
                correct += int(
                    gold["decision"] == "MATCH"
                    and prediction.get("metric_id") == gold.get("metric_id")
                )
        precision = correct / accepted if accepted else 1.0
        recall = correct / gold_exact if gold_exact else 0.0
        beta2 = 0.25
        f_beta = (
            (1 + beta2) * precision * recall / (beta2 * precision + recall)
            if precision + recall
            else 0.0
        )
        lower = _wilson_lower(correct, accepted)
        reports.append(
            {
                "minimum_confidence": confidence,
                "predicted_exact_count": accepted,
                "correct_exact_count": correct,
                "gold_exact_count": gold_exact,
                "exact_precision": precision,
                "exact_precision_wilson_95_lower": lower,
                "exact_recall": recall,
                "exact_f_beta_0_5": f_beta,
                "precision_wilson_gate_met": bool(
                    accepted >= min_predictions
                    and precision >= min_precision
                    and lower is not None
                    and lower >= min_wilson_lower
                ),
            }
        )
    feasible = [row for row in reports if row["precision_wilson_gate_met"]]

    def key(row: Mapping[str, Any]) -> tuple[float, ...]:
        lower = row["exact_precision_wilson_95_lower"]
        return (
            float(row["exact_f_beta_0_5"]),
            float(row["exact_recall"]),
            float(lower if lower is not None else -1.0),
            float(row["exact_precision"]),
            float(row["predicted_exact_count"]),
            float(CONFIDENCE_RANK[str(row["minimum_confidence"])]),
        )

    chosen = max(feasible or reports, key=key)
    return {
        **chosen,
        "gate_feasible": bool(feasible),
        "minimum_required_precision": min_precision,
        "minimum_required_wilson_lower": min_wilson_lower,
        "minimum_required_predictions": min_predictions,
        "all_thresholds": reports,
        "selection_split": "dev",
    }


def checkpoint_selection_key(report: Mapping[str, Any]) -> tuple[float, ...]:
    gate = report.get("confidence_gate") or {}
    loss = report.get("weighted_dev_loss")
    lower = gate.get("exact_precision_wilson_95_lower")
    return (
        float(bool(gate.get("precision_wilson_gate_met"))),
        float(gate.get("exact_f_beta_0_5", 0.0)),
        float(lower if lower is not None else -1.0),
        float(gate.get("exact_precision", 0.0)),
        float(gate.get("exact_recall", 0.0)),
        -float(loss if loss is not None else math.inf),
        float(report.get("cumulative_exposure", 0)),
    )


def evaluate_weighted_loss(
    model: Any,
    encoded: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> float:
    if not encoded:
        raise ValueError("cannot evaluate an empty dev set")
    weighted_sum = 0.0
    mass = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = collate_batch(
                encoded[start : start + batch_size], int(tokenizer.pad_token_id)
            )
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                ).logits
                loss = weighted_causal_lm_loss(
                    logits, batch["labels"], batch["loss_weights"]
                )
            batch_mass = float(batch["loss_weights"][:, 1:].sum().cpu())
            weighted_sum += float(loss.cpu()) * batch_mass
            mass += batch_mass
    model.train()
    if mass <= 0:
        raise AssertionError("dev weighted token mass is zero")
    return weighted_sum / mass


def _left_padded_prompts(
    rows: Sequence[dict[str, Any]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row["prompt_input_ids"]) for row in rows)
    ids, masks = [], []
    for row in rows:
        values = row["prompt_input_ids"]
        pad = width - len(values)
        ids.append([pad_token_id] * pad + values)
        masks.append([0] * pad + [1] * len(values))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def _parse_typed_response(
    raw: str, candidate_ids: set[str]
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the same typed JSON contract used by production adjudication."""

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for start, char in enumerate(raw or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except (json.JSONDecodeError, TypeError):
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw[start:])
            try:
                value, _ = decoder.raw_decode(repaired)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(value, dict) and "decision" in value:
            objects.append(value)
    if not objects:
        return None, "no_json"
    value = objects[-1]
    decision = str(value.get("decision") or "").strip().upper()
    metric_id = value.get("metric_id")
    metric_id = None if metric_id in (None, "", "null", "None") else str(metric_id).strip()
    confidence = str(value.get("confidence") or "").strip().lower()
    reason = str(value.get("reason") or "").strip()
    if decision not in TYPED_DECISIONS:
        return None, "unknown_decision"
    if confidence not in CONFIDENCE_RANK:
        return None, "unknown_confidence"
    if decision == "MATCH":
        if metric_id not in candidate_ids:
            return None, "metric_not_in_candidates"
    elif metric_id is not None:
        return None, "metric_on_abstention"
    if not reason:
        return None, "missing_reason"
    return {
        "decision": decision,
        "metric_id": metric_id,
        "confidence": confidence,
        "reason": reason,
    }, None


def generate_dev_predictions(
    model: Any,
    encoded: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    predictions: list[dict[str, Any] | None] = []
    errors: Counter[str] = Counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            rows = encoded[start : start + batch_size]
            input_ids, attention_mask = _left_padded_prompts(
                rows, int(tokenizer.pad_token_id)
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=tokenizer.eos_token_id,
            )
            suffixes = generated[:, input_ids.shape[1] :]
            for row, tokens in zip(rows, suffixes):
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                parsed, error = _parse_typed_response(
                    raw, set(row["candidate_metric_ids"])
                )
                predictions.append(parsed)
                if error:
                    errors[error] += 1
    model.train()
    return predictions, {
        "rows": len(encoded),
        "valid_predictions": sum(value is not None for value in predictions),
        "parse_errors": dict(sorted(errors.items())),
        "decoding": "greedy_temperature_zero",
        "max_new_tokens": max_new_tokens,
    }


def reload_probe(
    model: Any,
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    device: torch.device,
    max_values: int = 32,
) -> list[float]:
    """Capture a small deterministic vector of gold next-token logits."""

    batch = collate_batch(rows, int(tokenizer.pad_token_id))
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        ).logits.float()
    shifted = logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    valid = labels.ne(-100)
    coordinates = valid.nonzero(as_tuple=False)[:max_values]
    values = [
        float(shifted[batch_index, token_index, labels[batch_index, token_index]].cpu())
        for batch_index, token_index in coordinates.tolist()
    ]
    if not values:
        raise ValueError("reload probe has no assistant target values")
    return values


def _checkpoint_report(
    *,
    model: Any,
    encoded_dev: Sequence[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
    cumulative_exposure: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cumulative_exposure": cumulative_exposure,
        "selection_data": "dev_only" if encoded_dev else None,
        "test_or_blind_data_read": False,
    }
    if not encoded_dev:
        report.update({"weighted_dev_loss": None, "confidence_gate": None})
        return report
    report["weighted_dev_loss"] = evaluate_weighted_loss(
        model,
        encoded_dev,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.dev_batch_size,
    )
    predictions, generation = generate_dev_predictions(
        model,
        encoded_dev,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.dev_generation_batch_size,
        max_new_tokens=args.dev_max_new_tokens,
    )
    report["generation"] = generation
    report["confidence_gate"] = tune_dev_confidence_threshold(
        encoded_dev,
        predictions,
        min_precision=args.min_dev_exact_precision,
        min_wilson_lower=args.min_dev_wilson_lower,
        min_predictions=args.min_dev_exact_predictions,
    )
    return report


def train(
    args: argparse.Namespace,
    tokenizer: Any,
    encoded: list[dict[str, Any]],
    encoded_dev: list[dict[str, Any]],
    base_report: dict[str, Any],
) -> None:
    import gc

    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Gemma-4 LoRA training requires CUDA")
    output = Path(args.output).resolve()
    checkpoint_root = output.parent / f"{output.name}.exposure-checkpoints"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite adapter output: {output}")
    if checkpoint_root.exists():
        raise FileExistsError(f"refusing to overwrite exposure checkpoints: {checkpoint_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=False, exist_ok=False)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    device_name = torch.cuda.get_device_name(device)
    before_free, before_total = torch.cuda.mem_get_info(device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    target_architecture, target_regex, expected_modules = (
        resolve_language_model_target_scope(model)
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_regex,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    trainable_names, adapted_modules = validate_trainable_lora_scope(
        model, expected_modules
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    budgets = tuple(args.exposure_checkpoints) or (len(encoded) * args.epochs,)
    if not budgets or budgets[-1] <= 0:
        raise ValueError("training requires at least one positive exposure budget")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    loss_trace: list[dict[str, Any]] = []
    optimizer_step = 0
    cumulative_exposure = 0
    cycle = -1
    order: list[int] = []
    cursor = 0
    accumulated_micro_steps = 0
    accumulated_raw_loss = 0.0
    checkpoint_reports: list[dict[str, Any]] = []
    probe_rows = list((encoded_dev or encoded)[: min(2, len(encoded_dev or encoded))])
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def optimizer_update(model_obj: Any, optimizer_obj: Any) -> None:
        nonlocal optimizer_step, accumulated_micro_steps, accumulated_raw_loss
        if accumulated_micro_steps == 0:
            return
        if accumulated_micro_steps < args.gradient_accumulation_steps:
            correction = args.gradient_accumulation_steps / accumulated_micro_steps
            for parameter in model_obj.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in model_obj.parameters()
                if parameter.requires_grad
            ),
            args.max_grad_norm,
        )
        optimizer_obj.step()
        optimizer_obj.zero_grad(set_to_none=True)
        optimizer_step += 1
        record = {
            "optimizer_step": optimizer_step,
            "cumulative_exposure": cumulative_exposure,
            "mean_microbatch_loss": accumulated_raw_loss / accumulated_micro_steps,
            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "microbatches": accumulated_micro_steps,
        }
        loss_trace.append(record)
        if optimizer_step == 1 or optimizer_step % args.log_every_steps == 0:
            print(json.dumps({"training_progress": record}), flush=True)
        accumulated_micro_steps = 0
        accumulated_raw_loss = 0.0

    for budget in budgets:
        while cumulative_exposure < budget:
            if cursor >= len(order):
                cycle += 1
                order = torch.randperm(len(encoded), generator=generator).tolist()
                cursor = 0
            take = min(
                args.batch_size,
                budget - cumulative_exposure,
                len(order) - cursor,
            )
            batch_rows = [encoded[index] for index in order[cursor : cursor + take]]
            cursor += take
            batch = collate_batch(batch_rows, int(tokenizer.pad_token_id))
            batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                ).logits
                raw_loss = weighted_causal_lm_loss(
                    logits, batch["labels"], batch["loss_weights"]
                )
                loss = raw_loss / args.gradient_accumulation_steps
            loss.backward()
            accumulated_micro_steps += 1
            accumulated_raw_loss += float(raw_loss.detach().cpu())
            cumulative_exposure += take
            if accumulated_micro_steps == args.gradient_accumulation_steps:
                optimizer_update(model, optimizer)
        # A checkpoint is an exact exposure boundary, never a partially
        # accumulated gradient state.
        optimizer_update(model, optimizer)
        checkpoint_dir = checkpoint_root / f"exposure_{budget:012d}"
        adapter_dir = checkpoint_dir / "adapter"
        checkpoint_dir.mkdir(parents=False, exist_ok=False)
        reference = reload_probe(
            model,
            probe_rows,
            tokenizer=tokenizer,
            device=device,
        )
        model.save_pretrained(adapter_dir, safe_serialization=True)
        checkpoint = {
            **_checkpoint_report(
                model=model,
                encoded_dev=encoded_dev,
                tokenizer=tokenizer,
                device=device,
                args=args,
                cumulative_exposure=budget,
            ),
            "optimizer_steps_completed": optimizer_step,
            "training_cycles_entered": cycle + 1,
            "adapter": directory_ref(adapter_dir),
            "reload_reference_gold_logits": reference,
        }
        checkpoint_path = checkpoint_dir / "checkpoint.json"
        write_json_new(checkpoint_path, checkpoint)
        checkpoint["checkpoint_report"] = file_ref(checkpoint_path)
        checkpoint_reports.append(checkpoint)
        print(
            json.dumps(
                {
                    "checkpoint_complete": budget,
                    "dev": checkpoint.get("confidence_gate"),
                }
            ),
            flush=True,
        )

    chosen = (
        max(checkpoint_reports, key=checkpoint_selection_key)
        if encoded_dev
        else checkpoint_reports[-1]
    )
    chosen_checkpoint_dir = checkpoint_root / f"exposure_{chosen['cumulative_exposure']:012d}"
    chosen_adapter_dir = chosen_checkpoint_dir / "adapter"
    shutil.copytree(chosen_adapter_dir, output)
    selection = {
        "schema_version": "silver-match-v3-gemma4-typed-lora-dev-selection-v2",
        "status": "SELECTED_ON_DEV_ONLY" if encoded_dev else "LAST_CHECKPOINT_NO_DEV",
        "selection_split": "dev" if encoded_dev else None,
        "test_or_blind_data_read": False,
        "chosen_cumulative_exposure": chosen["cumulative_exposure"],
        "chosen_checkpoint": chosen["checkpoint_report"],
        "chosen_dev_report": {
            key: chosen.get(key)
            for key in ("weighted_dev_loss", "generation", "confidence_gate")
        },
        "all_checkpoint_selection_summaries": [
            {
                "cumulative_exposure": row["cumulative_exposure"],
                "weighted_dev_loss": row.get("weighted_dev_loss"),
                "confidence_gate": row.get("confidence_gate"),
            }
            for row in checkpoint_reports
        ],
    }
    selection_path = output / "DEV_SELECTION.json"
    write_json_new(selection_path, selection)

    config_path = output / "adapter_config.json"
    weights_path = output / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError("PEFT did not emit the expected adapter-only files")
    unexpected_large = [
        path for path in output.iterdir() if path.is_file() and path.stat().st_size > 4 * 1024**3
    ]
    if unexpected_large:
        raise RuntimeError(f"unexpected full-model-sized output files: {unexpected_large}")

    saved_config = PeftConfig.from_pretrained(output)
    if int(saved_config.r) != args.lora_r or int(saved_config.lora_alpha) != args.lora_alpha:
        raise RuntimeError("saved adapter config does not match frozen recipe")
    if saved_config.target_modules != target_regex:
        raise RuntimeError("saved adapter target scope is not exact q/k/v/o-only regex")
    expected_reference = list(chosen["reload_reference_gold_logits"])
    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    fresh_base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    fresh_architecture, fresh_regex, fresh_modules = resolve_language_model_target_scope(
        fresh_base
    )
    if (
        fresh_architecture != target_architecture
        or fresh_regex != target_regex
        or fresh_modules != expected_modules
    ):
        raise RuntimeError("fresh base model q/k/v/o module inventory drifted")
    fresh_model = PeftModel.from_pretrained(fresh_base, output, is_trainable=False)
    fresh_model.to(device)
    if any(parameter.requires_grad for parameter in fresh_model.parameters()):
        raise RuntimeError("fresh inference reload unexpectedly has trainable parameters")
    observed_reference = reload_probe(
        fresh_model,
        probe_rows,
        tokenizer=tokenizer,
        device=device,
    )
    reload_absolute_differences = [
        abs(expected - observed)
        for expected, observed in zip(expected_reference, observed_reference)
    ]
    if len(expected_reference) != len(observed_reference) or any(
        not math.isclose(expected, observed, rel_tol=1e-3, abs_tol=2e-2)
        for expected, observed in zip(expected_reference, observed_reference)
    ):
        raise RuntimeError("fresh-base adapter reload changed deterministic probe logits")
    fresh_model.eval()

    _, after_total = torch.cuda.mem_get_info(device)
    report = {
        **base_report,
        "schema_version": "silver-match-v3-gemma4-typed-lora-train-report-v2",
        "status": "COMPLETE_DEV_SELECTED_ADAPTER_FRESH_RELOAD_VERIFIED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name,
            "memory_total_bytes_before": before_total,
            "memory_total_bytes_after": after_total,
        },
        "parameters": {
            "trainable": trainable_parameters,
            "total_with_adapter": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "all_trainable_names_are_lora": True,
            "trainable_tensor_count": len(trainable_names),
            "adapted_text_linear_count": len(adapted_modules),
            "target_architecture": target_architecture,
            "target_scope_regex": target_regex,
            "expected_qkvo_modules": sorted(expected_modules),
        },
        "steps": {
            "optimizer_steps_completed": optimizer_step,
            "cumulative_exposure_completed": cumulative_exposure,
            "exposure_checkpoints": list(budgets),
            "training_cycles_entered": cycle + 1,
        },
        "loss": {
            "first_optimizer_step": loss_trace[0],
            "last_optimizer_step": loss_trace[-1],
            "minimum_mean_microbatch_loss": min(
                row["mean_microbatch_loss"] for row in loss_trace
            ),
        },
        "adapter": {
            "directory": str(output),
            "config": file_ref(config_path),
            "weights": file_ref(weights_path),
            "adapter_only": True,
            "inference_reload_verified": True,
            "fresh_base_reload_verified": True,
            "reload_probe_value_count": len(observed_reference),
            "reload_probe_max_absolute_difference": max(
                reload_absolute_differences, default=0.0
            ),
            "content": directory_ref(output),
        },
        "checkpoint_root": directory_ref(checkpoint_root),
        "selection": {**selection, "artifact": file_ref(selection_path)},
        "source_disjoint_audit": base_report["source_disjoint_audit"],
    }
    write_json_new(Path(args.report), report)
    print(json.dumps({"status": report["status"], "report": file_ref(Path(args.report))}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dev-dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-inventory")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--adapter-injection-preflight-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--exposure-checkpoints",
        default="",
        help="Increasing cumulative example exposures, e.g. 10000,25000,50000",
    )
    parser.add_argument("--decision-loss-weight", type=float, default=4.0)
    parser.add_argument("--metric-id-loss-weight", type=float, default=4.0)
    parser.add_argument("--confidence-loss-weight", type=float, default=1.0)
    parser.add_argument("--reason-loss-weight", type=float, default=0.25)
    parser.add_argument("--structural-loss-weight", type=float, default=0.25)
    parser.add_argument("--dev-batch-size", type=int, default=4)
    parser.add_argument("--dev-generation-batch-size", type=int, default=4)
    parser.add_argument("--dev-max-new-tokens", type=int, default=192)
    parser.add_argument("--min-dev-exact-precision", type=float, default=0.9)
    parser.add_argument("--min-dev-wilson-lower", type=float, default=0.85)
    parser.add_argument("--min-dev-exact-predictions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=94137)
    parser.add_argument("--log-every-steps", type=int, default=5)
    args = parser.parse_args()
    try:
        args.exposure_checkpoints = _parse_exposure_checkpoints(
            args.exposure_checkpoints
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.preflight_only and args.adapter_injection_preflight_only:
        parser.error("choose only one preflight mode")
    if not (args.preflight_only or args.adapter_injection_preflight_only) and not args.output:
        parser.error("--output is required unless a preflight-only mode is set")
    if min(
        args.max_length,
        args.epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.lora_r,
        args.lora_alpha,
        args.log_every_steps,
        args.dev_batch_size,
        args.dev_generation_batch_size,
        args.dev_max_new_tokens,
    ) <= 0:
        parser.error("integer recipe fields must be positive")
    loss_weights = [
        args.decision_loss_weight,
        args.metric_id_loss_weight,
        args.confidence_loss_weight,
        args.reason_loss_weight,
        args.structural_loss_weight,
    ]
    if any(not math.isfinite(value) or value <= 0 for value in loss_weights):
        parser.error("all token loss weights must be finite and positive")
    if not 0 <= args.min_dev_exact_precision <= 1:
        parser.error("--min-dev-exact-precision must be in [0,1]")
    if not 0 <= args.min_dev_wilson_lower <= 1:
        parser.error("--min-dev-wilson-lower must be in [0,1]")
    if args.min_dev_exact_predictions < 0:
        parser.error("--min-dev-exact-predictions must be non-negative")
    if not (args.preflight_only or args.adapter_injection_preflight_only) and not args.model_inventory:
        parser.error("actual training requires --model-inventory for an exact base-model binding")
    return args


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    report_path = Path(args.report).resolve()
    input_examples = read_examples(dataset_path)
    dev_path = Path(args.dev_dataset).resolve() if args.dev_dataset else None
    dev_examples = read_examples(dev_path) if dev_path else []
    examples, source_disjoint_audit = audit_gradient_roles(
        input_examples, dev_examples
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    field_loss_weights = {
        field: float(getattr(args, f"{field}_loss_weight")) for field in FIELD_NAMES
    }
    encoded, tokenization = tokenize_dataset(
        tokenizer,
        examples,
        args.max_length,
        field_loss_weights=field_loss_weights,
        structural_loss_weight=args.structural_loss_weight,
    )
    if dev_examples:
        encoded_dev, dev_tokenization = tokenize_dataset(
            tokenizer,
            dev_examples,
            args.max_length,
            field_loss_weights=field_loss_weights,
            structural_loss_weight=args.structural_loss_weight,
        )
    else:
        encoded_dev, dev_tokenization = [], None
    report = {
        "schema_version": "silver-match-v3-gemma4-typed-lora-preflight-v2",
        "status": "PASS_EXACT_TEMPLATE_NO_TRUNCATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": file_ref(dataset_path),
        "dev_dataset": file_ref(dev_path) if dev_path else None,
        "model": str(Path(args.model).resolve()),
        "model_inventory": file_ref(Path(args.model_inventory)) if args.model_inventory else None,
        "trainer_script": file_ref(Path(__file__)),
        "runtime": runtime_metadata(),
        "recipe": fixed_recipe(args),
        "tokenization": tokenization,
        "dev_tokenization": dev_tokenization,
        "source_disjoint_audit": source_disjoint_audit,
        "input_role_contract": {
            "gradient_inputs": [str(dataset_path)],
            "selection_inputs": [str(dev_path)] if dev_path else [],
            "test_or_blind_inputs": [],
        },
        "target_truncations": 0,
        "assistant_prefix_alignment_failures": 0,
    }
    if args.preflight_only:
        write_json_new(report_path, report)
        print(json.dumps({**report, "report": file_ref(report_path)}, sort_keys=True), flush=True)
        return
    if args.adapter_injection_preflight_only:
        adapter_injection_preflight(args, report)
        return
    train(args, tokenizer, encoded, encoded_dev, report)


if __name__ == "__main__":
    main()
