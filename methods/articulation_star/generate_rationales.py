"""Generate rationales for a training subsample using a (LoRA-adapted) vLLM model.

Per [[feedback_vllm_batch_size]] / [[feedback_vllm_safe_run_config]]: submit
thousands of prompts per generate call, GPU_MEM_UTIL ~0.9 on B200.

Output: jsonl with one row per (artifact, sample_idx).
  {row_id, y, prompt_messages, completion}
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

from .config import LoopConfig, TASKS
from .prompts import render_gen


# Trailing-verdict regex — strip if present in completion (defense-in-depth
# against the "Therefore: accept" leakage path).
_VERDICT_RE = re.compile(
    r"\n+\s*(therefore|overall|on balance|in summary|i (recommend|conclude|believe))[^\n]*$",
    re.IGNORECASE,
)


def _strip_verdict(text: str) -> str:
    return _VERDICT_RE.sub("", text).rstrip()


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Keep the start (abstract/title is usually informative).
    return text[:max_chars]


def run(
    cfg: LoopConfig,
    iter_idx: int,
    lora_path: str | None = None,
    teacher_model: str | None = None,
    shard_idx: int = 0,
    n_shards: int = 1,
) -> Path:
    """Generate rationales for this iter.

    Args:
        teacher_model: optional override for the generator HF id. Used for the
            iter-0 cold-start: pass the big teacher (e.g. Llama-3.3-70B) so the
            8B trainee has decent SFT seeds from round 1.
        shard_idx / n_shards: data-parallel slicing across processes. Each
            process picks shard_idx-th row out of every n_shards-th row.
    """
    task = TASKS[cfg.task]
    df = pd.read_csv(task.data_path)
    df = df[["text", "judgement"]].dropna()
    if cfg.n_train_subsample < len(df):
        df = df.sample(n=cfg.n_train_subsample, random_state=42 + iter_idx)
    df = df.reset_index(drop=True)
    if n_shards > 1:
        df = df.iloc[shard_idx::n_shards].reset_index(drop=True)
        orig_row_offset = shard_idx
    else:
        orig_row_offset = 0

    model_id = teacher_model or cfg.generator_model
    llm_kwargs = dict(
        model=model_id,
        gpu_memory_utilization=cfg.vllm_gpu_mem_util,
        max_model_len=cfg.vllm_max_model_len,
        dtype="bfloat16",
        enable_lora=lora_path is not None,
        max_lora_rank=cfg.lora_r if lora_path is not None else 16,
    )
    llm = LLM(**{k: v for k, v in llm_kwargs.items() if v is not None})
    tok = llm.get_tokenizer()

    prompts = []
    for _, row in df.iterrows():
        msgs = render_gen(
            text=_truncate_text(row["text"], cfg.max_text_chars),
            text_type=task.text_type,
            pos=task.positive_label,
            neg=task.negative_label,
        )
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(
        n=cfg.n_rationales_per_input,
        temperature=cfg.gen_temperature,
        top_p=cfg.gen_top_p,
        max_tokens=cfg.gen_max_tokens,
        stop=["\nTherefore", "\nOverall", "\nOn balance"],  # block trailing verdict
    )

    lora_request = None
    if lora_path is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("articulator", 1, lora_path)

    outputs = llm.generate(prompts, sp, lora_request=lora_request)

    out_dir = cfg.iter_dir(iter_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{shard_idx:02d}of{n_shards:02d}" if n_shards > 1 else ""
    out_path = out_dir / f"rationales{suffix}.jsonl"
    with out_path.open("w") as f:
        for local_row_idx, out in enumerate(outputs):
            # global_row_id matches the unsharded df.iloc index after subsampling,
            # so train_sft.py can rebuild the prompt deterministically.
            global_row_id = orig_row_offset + local_row_idx * n_shards
            for sample_idx, cand in enumerate(out.outputs):
                rec = {
                    "row_id": int(global_row_id),
                    "y": int(df.iloc[local_row_idx]["judgement"]),
                    "sample_idx": sample_idx,
                    "completion": _strip_verdict(cand.text),
                }
                f.write(json.dumps(rec) + "\n")

    print(f"[generate] wrote {out_path}")
    return out_path


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer_review")
    ap.add_argument("--run_name", default="v0")
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--teacher_model", default=None,
                    help="Override generator model id (cold-start iter 0).")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--n_train_subsample", type=int, default=None)
    ap.add_argument("--n_rationales_per_input", type=int, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = LoopConfig(task=args.task, run_name=args.run_name)
    if args.n_train_subsample is not None:
        cfg.n_train_subsample = args.n_train_subsample
    if args.n_rationales_per_input is not None:
        cfg.n_rationales_per_input = args.n_rationales_per_input
    run(
        cfg, args.iter,
        lora_path=args.lora_path,
        teacher_model=args.teacher_model,
        shard_idx=args.shard_idx, n_shards=args.n_shards,
    )
