#!/usr/bin/env python3
"""Train a soft-target Gemma-4-31B similarity LoRA on one frozen level.

The model predicts the next token from the fixed set ``0``, ``1``, ``2``.
Loss is cross-entropy against the balanced teacher distribution, not a hard
majority label.  The exact same constrained logits are used at evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.codability.lexicon_distill.dataset import LABELS, render_prompt


LANGUAGE_MODEL_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
EXPECTED_LANGUAGE_MODEL_TARGET_COUNT = 410
LABEL_TEXT = ("0", "1", "2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_sk2_host() -> None:
    host = socket.gethostname().split(".", 1)[0].lower()
    if host not in {"sk2", "skampere2"} and not host.startswith("skampere2-"):
        raise RuntimeError(f"similarity LoRA training is sk2-only; refusing host {socket.gethostname()}")


def runtime_metadata() -> dict[str, Any]:
    import torch
    import transformers

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
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def read_protocols(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {str(key): str(value["text"]) for key, value in payload.items()}
    if not result:
        raise ValueError("empty protocol bundle")
    return result


def read_rows(
    path: Path,
    *,
    level: str,
    task: str | None,
    primary_only: bool,
    auxiliary_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("level") != level or (task and row.get("task") != task):
                continue
            if row.get("split") not in {"train", "pair_dev"}:
                raise ValueError(f"non-training split in dataset line {line_number}: {row.get('split')}")
            families = set(row.get("family_distributions") or {})
            if primary_only and not families.intersection({"sonnet", "gpt5"}):
                continue
            if auxiliary_only:
                auxiliary = [
                    row["family_distributions"][family]
                    for family in ("opus", "glm") if family in families
                ]
                if not auxiliary:
                    continue
                row = dict(row)
                row["target_probs"] = [
                    sum(distribution[index] for distribution in auxiliary) / len(auxiliary)
                    for index in range(3)
                ]
                row["example_weight"] = 0.25
            probabilities = row.get("target_probs")
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != 3
                or any(type(value) not in (int, float) or value < 0 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-6
            ):
                raise ValueError(f"invalid soft target at line {line_number}")
            weight = row.get("example_weight")
            if type(weight) not in (int, float) or weight <= 0:
                raise ValueError(f"invalid example weight at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("no rows selected")
    return rows


def _token_list(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("unexpected batched tokenizer output")
        value = value[0]
    return [int(token) for token in value]


def label_token_ids(tokenizer: Any, prompt_prefix: str) -> tuple[int, int, int]:
    prompt_ids = _token_list(tokenizer(prompt_prefix, add_special_tokens=False))
    result: list[int] = []
    for label in LABEL_TEXT:
        full_ids = _token_list(tokenizer(prompt_prefix + label, add_special_tokens=False))
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) != len(prompt_ids) + 1:
            raise ValueError(
                "numeric similarity labels are not exactly one stable next token under the Gemma chat template"
            )
        result.append(full_ids[-1])
    if len(set(result)) != 3:
        raise ValueError(f"similarity label tokens are not unique: {result}")
    return tuple(result)  # type: ignore[return-value]


def encode_rows(
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    protocols: Mapping[str, str],
    *,
    max_length: int,
    augment_order: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[int, int, int]]:
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return exactly 0, 1, or 2.\nLABEL:"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(probe, str):
        raise ValueError("chat template did not render text")
    label_ids = label_token_ids(tokenizer, probe)
    encoded: list[dict[str, Any]] = []
    lengths: list[int] = []
    for row in rows:
        protocol_id = str(row["protocol_id"])
        if protocol_id not in protocols:
            raise ValueError(f"unknown protocol {protocol_id}")
        views = [(str(row["text_a"]), str(row["text_b"]), "ab")]
        if augment_order:
            views.append((str(row["text_b"]), str(row["text_a"]), "ba"))
        view_weight = float(row["example_weight"]) / len(views)
        for text_a, text_b, view in views:
            user_text = render_prompt(protocols[protocol_id], str(row["task"]), text_a, text_b)
            # Numeric tokens keep soft-target training to one auditable next-token decision.
            user_text += "\n\nLABEL MAP: 0=DIFFERENT, 1=RELATED, 2=SAME.\nLABEL:"
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
            )
            if not isinstance(prompt, str):
                raise ValueError("chat template did not render text")
            ids = _token_list(tokenizer(prompt, add_special_tokens=False))
            if not ids or len(ids) > max_length:
                raise ValueError(
                    f"prompt length outside 1..{max_length}: {row['example_id']}/{view}={len(ids)}"
                )
            # Check the actual row boundary, not only the short probe.
            actual_label_ids = label_token_ids(tokenizer, prompt)
            if actual_label_ids != label_ids:
                raise ValueError(f"label token drift: {row['example_id']}/{view}")
            encoded.append(
                {
                    "input_ids": ids,
                    "target_probs": [float(value) for value in row["target_probs"]],
                    "weight": view_weight,
                    "example_id": str(row["example_id"]),
                    "task": str(row["task"]),
                    "view": view,
                }
            )
            lengths.append(len(ids))
    summary = {
        "source_rows": len(rows),
        "encoded_views": len(encoded),
        "unique_examples": len({row["example_id"] for row in encoded}),
        "tasks": dict(Counter(row["task"] for row in encoded)),
        "tokens": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "total": sum(lengths),
        },
        "label_token_ids": dict(zip(LABELS, label_ids)),
        "order_augmentation": augment_order,
        "total_example_weight": sum(row["weight"] for row in encoded),
    }
    return encoded, summary, label_ids


def collate(rows: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    width = max(len(row["input_ids"]) for row in rows)
    input_ids: list[list[int]] = []
    masks: list[list[int]] = []
    for row in rows:
        ids = list(row["input_ids"])
        padding = width - len(ids)
        input_ids.append(ids + [pad_token_id] * padding)
        masks.append([1] * len(ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "target_probs": torch.tensor([row["target_probs"] for row in rows], dtype=torch.float32),
        "weights": torch.tensor([row["weight"] for row in rows], dtype=torch.float32),
    }


def validate_trainable_scope(model: Any) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if (
        not trainable
        or any("lora_" not in name for name, _parameter in trainable)
        or any(".language_model.layers." not in name for name, _parameter in trainable)
        or any(".vision_tower." in name or ".audio_tower." in name for name, _parameter in trainable)
    ):
        raise RuntimeError("trainable parameters escaped the Gemma text LoRA scope")
    modules = {name.split(".lora_", 1)[0] for name, _parameter in trainable}
    if len(modules) != EXPECTED_LANGUAGE_MODEL_TARGET_COUNT:
        raise RuntimeError(
            f"unexpected adapted text-linear count {len(modules)} != {EXPECTED_LANGUAGE_MODEL_TARGET_COUNT}"
        )
    return {
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _name, parameter in trainable),
        "adapted_text_linears": len(modules),
    }


def _load_model(args: argparse.Namespace) -> Any:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    else:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LANGUAGE_MODEL_TARGET_REGEX,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, config)
    return model


def train(
    args: argparse.Namespace,
    tokenizer: Any,
    encoded: list[dict[str, Any]],
    label_ids: tuple[int, int, int],
    base_report: dict[str, Any],
) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    model = _load_model(args)
    model.to(device)
    scope = validate_trainable_scope(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    label_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    micro_steps = math.ceil(len(encoded) / args.batch_size)
    expected_steps = math.ceil(micro_steps / args.gradient_accumulation_steps) * args.epochs
    step = 0
    traces: list[dict[str, Any]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        indices = torch.randperm(len(encoded), generator=generator).tolist()
        window = 0
        window_loss = 0.0
        for micro_step, start in enumerate(range(0, len(indices), args.batch_size), 1):
            rows = [encoded[index] for index in indices[start : start + args.batch_size]]
            batch = {key: value.to(device, non_blocking=True) for key, value in collate(rows, int(tokenizer.pad_token_id)).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                last_positions = batch["attention_mask"].sum(dim=1) - 1
                batch_indices = torch.arange(logits.shape[0], device=device)
                constrained = logits[batch_indices, last_positions][:, label_tensor]
                log_probs = torch.log_softmax(constrained.float(), dim=-1)
                row_losses = -(batch["target_probs"] * log_probs).sum(dim=-1)
                raw_loss = (row_losses * batch["weights"]).sum() / batch["weights"].sum()
                loss = raw_loss / args.gradient_accumulation_steps
            loss.backward()
            window += 1
            window_loss += float(raw_loss.detach().cpu())
            epoch_end = micro_step == micro_steps
            if window == args.gradient_accumulation_steps or epoch_end:
                if window < args.gradient_accumulation_steps:
                    correction = args.gradient_accumulation_steps / window
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                record = {
                    "epoch": epoch + 1,
                    "optimizer_step": step,
                    "mean_microbatch_loss": window_loss / window,
                    "gradient_norm_before_clip": float(norm.detach().cpu()),
                }
                traces.append(record)
                if step == 1 or step % args.log_every_steps == 0:
                    print(json.dumps({"training_progress": record}), flush=True)
                window = 0
                window_loss = 0.0
    if step != expected_steps:
        raise AssertionError(f"optimizer step mismatch {step} != {expected_steps}")
    output.mkdir(parents=False)
    model.save_pretrained(output, safe_serialization=True)
    config = output / "adapter_config.json"
    weights = output / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise RuntimeError("adapter-only save is incomplete")
    report = {
        **base_report,
        "schema_version": "gemma4-similarity-lora-train-v1",
        "status": "COMPLETE_ADAPTER_ONLY",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "steps": step,
        "loss": {"first": traces[0], "last": traces[-1], "minimum": min(row["mean_microbatch_loss"] for row in traces)},
        "adapter": {"directory": str(output), "config": file_ref(config), "weights": file_ref(weights)},
    }
    write_json_new(Path(args.report), report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dev-dataset")
    parser.add_argument("--protocols", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-inventory")
    parser.add_argument("--level", required=True, choices=("R1", "R2", "R3"))
    parser.add_argument("--task")
    parser.add_argument("--init-adapter")
    parser.add_argument("--output")
    parser.add_argument("--report", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--auxiliary-only", action="store_true")
    parser.add_argument("--no-order-augmentation", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=94137)
    parser.add_argument("--log-every-steps", type=int, default=25)
    args = parser.parse_args()
    if not args.preflight_only and not args.output:
        parser.error("--output is required for training")
    if args.task and not args.init_adapter:
        parser.error("task-local training requires --init-adapter from the pooled level model")
    if args.primary_only and args.auxiliary_only:
        parser.error("choose only one teacher scope")
    return args


def main() -> None:
    assert_sk2_host()
    args = parse_args()
    from transformers import AutoTokenizer

    dataset = Path(args.dataset).resolve()
    protocols_path = Path(args.protocols).resolve()
    rows = read_rows(
        dataset, level=args.level, task=args.task,
        primary_only=args.primary_only, auxiliary_only=args.auxiliary_only,
    )
    protocols = read_protocols(protocols_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded, tokenization, label_ids = encode_rows(
        tokenizer,
        rows,
        protocols,
        max_length=args.max_length,
        augment_order=not args.no_order_augmentation,
    )
    report = {
        "schema_version": "gemma4-similarity-lora-preflight-v1",
        "status": "PASS_SOFT_TARGET_EXACT_NEXT_TOKEN",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": file_ref(dataset),
        "protocols": file_ref(protocols_path),
        "model": str(Path(args.model).resolve()),
        "model_inventory": file_ref(Path(args.model_inventory)) if args.model_inventory else None,
        "runtime": runtime_metadata(),
        "selection": {
            "level": args.level, "task": args.task,
            "primary_only": args.primary_only, "auxiliary_only": args.auxiliary_only,
        },
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "init_adapter": args.init_adapter,
            "soft_target_loss": True,
        },
        "tokenization": tokenization,
    }
    if args.preflight_only:
        write_json_new(Path(args.report), report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    train(args, tokenizer, encoded, label_ids, report)


if __name__ == "__main__":
    main()
