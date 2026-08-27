"""LoRA SFT on the kept rationales.

Anti-leakage at training time:
  - Training input = the chat prompt (system + user). The model sees x.
  - Training target = the rationale ONLY (assistant turn). No verdict, no
    label word.
  - TRL's `assistant_only_loss=True` masks loss to the assistant turn so the
    model is never updated on prompt or label tokens.

The model's gradient updates only the (chat(x) → rationale) channel. The
label is never a training target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import LoopConfig, TASKS
from .prompts import render_gen


def _build_dataset(cfg: LoopConfig, iter_idx: int) -> Dataset:
    task = TASKS[cfg.task]
    kept_path = cfg.iter_dir(iter_idx) / "rationales_kept.jsonl"
    rationales = [json.loads(l) for l in kept_path.open()]
    if not rationales:
        raise RuntimeError(f"No kept rationales at {kept_path}")

    df = pd.read_csv(task.data_path)[["text", "judgement"]].dropna()
    if cfg.n_train_subsample < len(df):
        df = df.sample(n=cfg.n_train_subsample, random_state=42 + iter_idx)
    df = df.reset_index(drop=True)

    records = []
    for r in rationales:
        row = df.iloc[r["row_id"]]
        # Conversational format: TRL handles chat templating + assistant-only
        # loss masking automatically when given `messages`.
        msgs = render_gen(
            text=row["text"][: cfg.max_text_chars],
            text_type=task.text_type,
            pos=task.positive_label,
            neg=task.negative_label,
        )
        msgs.append({"role": "assistant", "content": r["completion"].strip()})
        records.append({"messages": msgs})

    return Dataset.from_list(records)


def run(cfg: LoopConfig, iter_idx: int, prev_lora: str | None = None) -> Path:
    out_dir = cfg.iter_dir(iter_idx) / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.generator_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Llama-3's stock chat template lacks the `{% generation %}` markers that
    # TRL needs for assistant-only loss masking. Inline a training-compatible
    # variant: identical token stream to the default, but the assistant
    # turn's content is wrapped in {% generation %} ... {% endgeneration %}
    # so TRL knows which span carries the loss.
    tok.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% generation %}{{ message['content'] | trim }}{% endgeneration %}"
        "{{ '<|eot_id|>' }}"
        "{% else %}"
        "{% set content = '<|start_header_id|>' + message['role'] + "
        "'<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
        "{% endif %}{{ content }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% endif %}"
    )

    base = AutoModelForCausalLM.from_pretrained(
        cfg.generator_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_cfg: LoraConfig | None
    if prev_lora:
        # Continued training: load prior adapter and resume.
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, prev_lora, is_trainable=True)
        peft_cfg = None
    else:
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )

    ds = _build_dataset(cfg, iter_idx)

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.n_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=2,
        save_strategy="epoch",
        bf16=True,
        max_length=cfg.max_seq_len,
        # Mask loss to the assistant turn only -- this is the central
        # anti-leakage knob: the model is never updated on the prompt (which
        # holds the artifact) or any system/user content.
        assistant_only_loss=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=base,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[train] saved LoRA to {out_dir}")
    return out_dir


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer_review")
    ap.add_argument("--run_name", default="v0")
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--prev_lora", default=None)
    ap.add_argument("--n_train_subsample", type=int, default=None,
                    help="MUST match what generate_rationales used so iloc "
                         "lookups by row_id align.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = LoopConfig(task=args.task, run_name=args.run_name)
    if args.n_train_subsample is not None:
        cfg.n_train_subsample = args.n_train_subsample
    run(cfg, args.iter, prev_lora=args.prev_lora)
