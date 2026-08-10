#!/usr/bin/env python3
"""Same as sk3_validity_full_runner.py but writes outputs incrementally
in batches of BATCH_FLUSH prompts (default 200). Resumable across restarts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
MODEL_BASE = ("/lfs/skampere3/0/shared_hf_cache/"
              "models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots")
PROMPT_DIR = REPO / os.environ.get("PROMPT_DIR", "")
RESPONSE_DIR = REPO / os.environ.get("RESPONSE_DIR", "")
RESPONSE_EXT = os.environ.get("RESPONSE_EXT", ".json")
TEMP = float(os.environ.get("TEMP", "0.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2000"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
BATCH_FLUSH = int(os.environ.get("BATCH_FLUSH", "200"))


def split_prompt(text):
    if "\n=== USER ===\n" in text:
        sys_msg, user_msg = text.split("\n=== USER ===\n", 1)
        return sys_msg.strip(), user_msg.strip()
    return "", text


def main():
    print(f"=== streamed runner ===\n  prompts: {PROMPT_DIR}\n  resp: {RESPONSE_DIR}", flush=True)
    print(f"  flush every {BATCH_FLUSH} prompts; temp={TEMP}", flush=True)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    prompt_files = sorted(PROMPT_DIR.glob("*.txt"))
    to_run = [pf for pf in prompt_files
              if not (RESPONSE_DIR / (pf.stem + RESPONSE_EXT)).exists()]
    print(f"  {len(prompt_files)} total; {len(to_run)} to run "
          f"({len(prompt_files)-len(to_run)} cached)", flush=True)
    if not to_run:
        print("nothing to do")
        return

    print("loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    model_dir = MODEL_BASE + "/" + sorted(os.listdir(MODEL_BASE))[0]
    llm = LLM(model=model_dir, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
              enforce_eager=False)
    sampling = SamplingParams(temperature=TEMP, max_tokens=MAX_TOKENS,
                              seed=42 if TEMP > 0 else None)

    # Process in flush-sized sub-batches
    for start in range(0, len(to_run), BATCH_FLUSH):
        sub = to_run[start:start + BATCH_FLUSH]
        messages_list = []
        for pf in sub:
            sys_msg, user_msg = split_prompt(pf.read_text())
            msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else [])
            msgs.append({"role": "user", "content": user_msg})
            messages_list.append(msgs)
        print(f"\nbatch {start//BATCH_FLUSH + 1} / "
              f"{(len(to_run)+BATCH_FLUSH-1)//BATCH_FLUSH}: "
              f"{len(messages_list)} prompts", flush=True)
        outputs = llm.chat(messages_list, sampling, use_tqdm=True)
        for pf, out in zip(sub, outputs):
            text = out.outputs[0].text.strip()
            if RESPONSE_EXT == ".py" and text.startswith("```"):
                lines = text.splitlines()[1:]
                if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
                text = "\n".join(lines)
            (RESPONSE_DIR / (pf.stem + RESPONSE_EXT)).write_text(text)
        print(f"  flushed {len(sub)} responses (total written so far: "
              f"{len(list(RESPONSE_DIR.glob('*' + RESPONSE_EXT)))})", flush=True)

    print(f"=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
