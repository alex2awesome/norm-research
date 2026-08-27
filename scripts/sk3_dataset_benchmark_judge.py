#!/usr/bin/env python3
"""Run the dataset/benchmark judge on sk3 with BF16 Llama-3.3-70B via vLLM.

Reads JSONL of papers, writes JSONL of {paper_id, label, confidence, reason,
raw}. Resumes by skipping paper_ids already present in the output.

Usage on sk3:
    cd /lfs/skampere3/0/alexspan/norm-research
    HOME=/lfs/skampere3/0/alexspan \
    CUDA_VISIBLE_DEVICES=2 \
    INPUT=outputs/dataset_judge/inputs/test_papers.jsonl \
    OUTPUT=outputs/dataset_judge/verdicts/test_v1.jsonl \
    BATCH=64 \
    nohup /lfs/skampere3/0/alexspan/miniconda3/bin/python \
        scripts/sk3_dataset_benchmark_judge.py > judge_test.log 2>&1 &
"""
from __future__ import annotations

# --- AFS-token hardening ---  (per [[reference_sk3_vllm_bf16]])
import os
os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/alexspan/.cache/huggingface")
os.environ.setdefault("VLLM_CACHE_ROOT", "/lfs/skampere3/0/alexspan/.cache/vllm")
os.environ.setdefault("TRITON_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/lfs/skampere3/0/alexspan/.cache")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import json
import sys
import time
from pathlib import Path

# Add scripts/ to path so we can import the prompt module
REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(REPO / "scripts"))

from dataset_benchmark_judge_prompt import build_messages   # noqa: E402

INPUT  = Path(os.environ.get("INPUT",  "outputs/dataset_judge/inputs/test_papers.jsonl"))
OUTPUT = Path(os.environ.get("OUTPUT", "outputs/dataset_judge/verdicts/test_v1.jsonl"))
BATCH  = int(os.environ.get("BATCH", "64"))

# BF16 model — per [[reference_sk3_vllm_bf16]]
MODEL_BASE = Path("/lfs/skampere3/0/shared_hf_cache/"
                  "models--meta-llama--Llama-3.3-70B-Instruct/snapshots")


def parse_verdict(text: str) -> dict | None:
    """Strict-JSON parse; tolerate stray prose by extracting the first {...}."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # find first {...}
    i = text.find("{")
    j = text.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        obj = json.loads(text[i:j+1])
    except json.JSONDecodeError:
        return None
    label = obj.get("label", "").upper()
    if label not in {"DATASET", "BENCHMARK", "BOTH", "NEITHER"}:
        return None
    return {
        "label": label,
        "confidence": obj.get("confidence", "").lower() or "medium",
        "reason": (obj.get("reason") or "").strip()[:500],
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # load input
    rows = []
    with open(INPUT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"[input] {len(rows):,} papers from {INPUT}", flush=True)

    # resume: skip paper_ids already in output
    done = set()
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["paper_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done):,} paper_ids already in {OUTPUT} — skipping", flush=True)

    todo = [r for r in rows if r["paper_id"] not in done]
    print(f"[todo] {len(todo):,} papers to judge", flush=True)
    if not todo:
        print("nothing to do.")
        return

    # load vLLM
    print(f"[vllm] loading Llama-3.3-70B BF16...", flush=True)
    snapshots = sorted(MODEL_BASE.iterdir())
    if not snapshots:
        raise RuntimeError(f"no snapshots under {MODEL_BASE}")
    model_dir = str(snapshots[0])
    print(f"  model_dir = {model_dir}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.88,
        max_model_len=4096,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=400,
    )

    t0 = time.time()
    n_ok = 0
    n_err = 0
    with open(OUTPUT, "a") as fout:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i+BATCH]
            msgs_list = [build_messages(r["title"], r["abstract"]) for r in chunk]
            outs = llm.chat(msgs_list, sampling, use_tqdm=False)
            for r, out in zip(chunk, outs):
                txt = out.outputs[0].text.strip()
                v = parse_verdict(txt)
                rec = {
                    "paper_id": r["paper_id"],
                    "venue":    r.get("venue"),
                    "year":     r.get("year"),
                    "decision_unified": r.get("decision_unified"),
                }
                if v is None:
                    rec["label"] = None
                    rec["raw"] = txt[:500]
                    n_err += 1
                else:
                    rec.update(v)
                    n_ok += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            elapsed = time.time() - t0
            rate = (i + len(chunk)) / max(elapsed, 1e-6)
            eta_s = (len(todo) - (i + len(chunk))) / max(rate, 1e-6)
            print(f"  [{i+len(chunk):>6}/{len(todo)}] ok={n_ok} err={n_err} "
                  f"rate={rate:.1f} pap/s  eta={eta_s/60:.1f} min",
                  flush=True)

    print(f"[done] {n_ok:,} ok, {n_err:,} parse-error  -> {OUTPUT}", flush=True)
    print(f"[total] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
