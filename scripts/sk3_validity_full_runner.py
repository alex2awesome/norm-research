#!/usr/bin/env python3
"""Generic Llama runner for validity_full pipeline.

Reads all *.txt files in PROMPT_DIR (each has SYSTEM + "\n=== USER ===\n" + USER)
and writes responses to RESPONSE_DIR as <key>.<ext>.

Usage on sk3:
  HOME=/lfs/skampere3/0/alexspan CUDA_VISIBLE_DEVICES=2 \\
  FLASHINFER_DISABLE_VERSION_CHECK=1 \\
  PROMPT_DIR=runs/validity_full/full_v1/paraphrase_prompts \\
  RESPONSE_DIR=runs/validity_full/full_v1/paraphrase_responses \\
  RESPONSE_EXT=.json TEMP=0.0 MAX_TOKENS=2000 \\
  nohup python scripts/sk3_validity_full_runner.py > log.txt 2>&1 &
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


def main():
    print(f"=== validity_full Llama runner ===", flush=True)
    print(f"  prompts: {PROMPT_DIR}", flush=True)
    print(f"  responses: {RESPONSE_DIR}", flush=True)
    print(f"  temp={TEMP} max_tokens={MAX_TOKENS}", flush=True)
    if not PROMPT_DIR.exists():
        print(f"missing prompt dir"); sys.exit(1)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(PROMPT_DIR.glob("*.txt"))
    # Skip if response exists already (resumable)
    to_run = [pf for pf in prompt_files
              if not (RESPONSE_DIR / (pf.stem + RESPONSE_EXT)).exists()]
    print(f"  {len(prompt_files)} prompts total; {len(to_run)} to run "
          f"({len(prompt_files) - len(to_run)} cached)", flush=True)
    if not to_run:
        print("nothing to do")
        return

    messages_list = []
    for pf in to_run:
        text = pf.read_text()
        if "\n=== USER ===\n" in text:
            sys_msg, user_msg = text.split("\n=== USER ===\n", 1)
        else:
            sys_msg, user_msg = "", text
        msgs = []
        if sys_msg.strip(): msgs.append({"role": "system", "content": sys_msg.strip()})
        msgs.append({"role": "user", "content": user_msg.strip()})
        messages_list.append(msgs)

    print("loading vLLM (Llama-3.3-70B)...", flush=True)
    from vllm import LLM, SamplingParams
    model_dir = MODEL_BASE + "/" + sorted(os.listdir(MODEL_BASE))[0]
    llm = LLM(model=model_dir, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
              enforce_eager=False)
    sampling = SamplingParams(temperature=TEMP, max_tokens=MAX_TOKENS,
                              seed=42 if TEMP > 0 else None)
    print(f"submitting {len(to_run)} prompts...", flush=True)
    outputs = llm.chat(messages_list, sampling, use_tqdm=True)

    for pf, out in zip(to_run, outputs):
        text = out.outputs[0].text.strip()
        if RESPONSE_EXT == ".py" and text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
            text = "\n".join(lines)
        (RESPONSE_DIR / (pf.stem + RESPONSE_EXT)).write_text(text)

    print(f"=== DONE. wrote {len(to_run)} responses ===", flush=True)


if __name__ == "__main__":
    main()
