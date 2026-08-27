#!/usr/bin/env python3
"""Run the validity-pilot code-gen and/or judge prompts on sk3 with Llama-3.3-70B.

Reuses the prompt files materialized by the Claude pilot pipeline:
  - runs/validity_pilot/<run>/codegen/prompts/<key>.txt        (system + user)
  - runs/validity_pilot/<run>/judge/score_prompts/<key>.txt    (system + user)

For each prompt file, splits on "\n=== USER ===\n" to get (system, user),
runs through vLLM with temp=0 (deterministic) or temp=0.5 (with sampling),
and writes the response to <key>.py (code-gen) or <key>.json (judge).

Output dir is a sibling of responses/ named responses_llama/ so we can
compare cross-model later.

Usage on sk3:
  cd /lfs/skampere3/0/alexspan/norm-research
  HOME=/lfs/skampere3/0/alexspan \\
  CUDA_VISIBLE_DEVICES=2 \\
  PILOT_RUN=smoke \\
  PILOT_PHASE=codegen \\
  PILOT_TEMP=0.0 \\
  nohup /lfs/skampere3/0/alexspan/miniconda3/bin/python \\
      scripts/sk3_validity_pilot_llama.py > pilot_llama.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
MODEL_BASE = ("/lfs/skampere3/0/shared_hf_cache/"
              "models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots")
PILOT_RUN = os.environ.get("PILOT_RUN", "smoke")
PILOT_PHASE = os.environ.get("PILOT_PHASE", "codegen")  # codegen | judge
TEMP = float(os.environ.get("PILOT_TEMP", "0.0"))


def split_prompt(text: str):
    if "\n=== USER ===\n" in text:
        sys_msg, user_msg = text.split("\n=== USER ===\n", 1)
        return sys_msg.strip(), user_msg.strip()
    # No split marker — treat whole thing as user
    return "", text


def main():
    base = REPO / "runs" / "validity_pilot" / PILOT_RUN
    if PILOT_PHASE == "codegen":
        prompt_dir = base / "codegen" / "prompts"
        out_dir = base / "codegen" / "responses_llama"
        ext = ".py"
    elif PILOT_PHASE == "judge":
        prompt_dir = base / "judge" / "score_prompts"
        out_dir = base / "judge" / "score_responses_llama"
        ext = ".json"
    else:
        print(f"unknown PILOT_PHASE={PILOT_PHASE}")
        sys.exit(1)

    out_dir.mkdir(exist_ok=True, parents=True)
    prompt_files = sorted(prompt_dir.glob("*.txt"))
    print(f"=== sk3 validity-pilot Llama: run={PILOT_RUN} phase={PILOT_PHASE} "
          f"temp={TEMP} prompts={len(prompt_files)} ===", flush=True)

    if not prompt_files:
        print(f"no prompts in {prompt_dir}")
        return

    # Build chat messages
    messages_list = []
    for pf in prompt_files:
        sys_msg, user_msg = split_prompt(pf.read_text())
        msgs = []
        if sys_msg:
            msgs.append({"role": "system", "content": sys_msg})
        # Strip any in-text few-shot if present (validity pilot prompts don't use them
        # except possibly for paraphrase; leave as-is for now)
        msgs.append({"role": "user", "content": user_msg})
        messages_list.append(msgs)

    print(f"loading vLLM (Llama-3.3-70B-FP8)...", flush=True)
    from vllm import LLM, SamplingParams
    model_dir = MODEL_BASE + "/" + sorted(os.listdir(MODEL_BASE))[0]
    llm = LLM(model=model_dir, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=16384,
              enforce_eager=False)
    sampling = SamplingParams(
        temperature=TEMP, max_tokens=3000,
        seed=42 if TEMP > 0 else None,
    )
    print(f"submitting {len(messages_list)} prompts...", flush=True)
    outputs = llm.chat(messages_list, sampling, use_tqdm=True)

    for pf, out in zip(prompt_files, outputs):
        key = pf.stem
        text = out.outputs[0].text.strip()
        # For code-gen: strip markdown fences if model added them
        if PILOT_PHASE == "codegen":
            if text.startswith("```"):
                lines = text.splitlines()
                # remove first line (```python or ```) and last fence if present
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)
        (out_dir / f"{key}{ext}").write_text(text)

    print(f"=== DONE. wrote {len(prompt_files)} responses -> {out_dir} ===",
          flush=True)


if __name__ == "__main__":
    main()
