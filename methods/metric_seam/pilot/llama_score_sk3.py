"""Offline batch vLLM scorer — SECOND JUDGE FAMILY (Llama-3.3-70B BF16) for W1.2 replication.

Same interface/resume semantics as gemma_score_v1.py (channel-aware key). Reuses survey
prompt files verbatim so the replication is apples-to-apples: same prompts, different judge.

Usage (sk3, ONE mostly-free GPU — BF16 70B needs ~160 GiB at util 0.90):
  CUDA_VISIBLE_DEVICES=<gpu> HOME=/lfs/skampere3/0/alexspan \
  /lfs/skampere3/0/alexspan/miniconda3/bin/python llama_score_sk3.py \
      --prompts prompts.jsonl --out results_llama.jsonl --max-model-len 10240
"""
import argparse, glob, json, os, re

_SNAP = glob.glob("/lfs/skampere3/0/shared_hf_cache/"
                  "models--meta-llama--Llama-3.3-70B-Instruct/snapshots/*/")
MODEL = sorted(_SNAP)[0] if _SNAP else None
FLUSH = 500

def parse_score(raw):
    m = re.search(r"SCORE:\s*(NA|\d+)", raw, re.IGNORECASE)
    if not m:
        return None
    tok = m.group(1).upper()
    if tok == "NA":
        return "NA"
    v = int(tok)
    return v if 0 <= v <= 10 else None

def key(r):
    return (r.get("channel", ""), r["aspect_id"], r["datapoint_id"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=10240)
    args = ap.parse_args()
    assert MODEL, "Llama-3.3-70B snapshot not found in shared_hf_cache"

    rows = [json.loads(l) for l in open(args.prompts)]
    done = set()
    if os.path.exists(args.out):
        done = {key(json.loads(l)) for l in open(args.out)}
        print(f"resume: {len(done)} already scored", flush=True)
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(todo)} prompts to score (model {MODEL})", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=args.max_model_len,
              gpu_memory_utilization=float(os.environ.get("SEAM_GPU_UTIL", "0.90")),
              dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=int(os.environ.get("SEAM_MAX_TOKENS", "48")))

    with open(args.out, "a") as f:
        for i in range(0, len(todo), FLUSH):
            chunk = todo[i:i + FLUSH]
            msgs = [[{"role": "user", "content": r["prompt"]}] for r in chunk]
            outs = llm.chat(msgs, sp)
            for r, o in zip(chunk, outs):
                raw = o.outputs[0].text.strip()
                f.write(json.dumps({"channel": r.get("channel", ""),
                                    "aspect_id": r["aspect_id"],
                                    "datapoint_id": r["datapoint_id"],
                                    "raw": raw, "score": parse_score(raw)}) + "\n")
            f.flush()
            print(f"flushed {min(i + FLUSH, len(todo))}/{len(todo)}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
