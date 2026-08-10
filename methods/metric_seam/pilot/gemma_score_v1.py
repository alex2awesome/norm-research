"""Offline batch vLLM scorer, v1: channel-aware (pass1/pass2/scope/cf share item ids).

Usage (sk3, gemma4 env, ONE GPU):
  CUDA_VISIBLE_DEVICES=<gpu> HOME=/lfs/skampere3/0/alexspan \
  /lfs/skampere3/0/alexspan/envs/gemma4/bin/python gemma_score_v1.py \
      --prompts prompts_v1.jsonl --out results_v1.jsonl --max-model-len 10240
"""
import argparse, json, os, re

MODEL = ("/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/"
         "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb")
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

    rows = [json.loads(l) for l in open(args.prompts)]
    done = set()
    if os.path.exists(args.out):
        done = {key(json.loads(l)) for l in open(args.out)}
        print(f"resume: {len(done)} already scored", flush=True)
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(todo)} prompts to score", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.93, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=48)

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
