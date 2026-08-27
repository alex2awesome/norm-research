"""Offline batch vLLM scorer — THIRD FAMILY (Qwen3.5-122B-A10B-FP8) for the transport test.
Same interface/resume semantics as llama_score_sk3.py. Requires VLLM_USE_FLASHINFER_MOE_FP8=0
(memory: reference_qwen35_vllm_sk3) and the gemma4 env (vLLM 0.23).
"""
import argparse, glob, json, os, re
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")

_SNAP = glob.glob("/lfs/skampere3/0/shared_hf_cache/"
                  "models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/*/")
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
    assert MODEL, "Qwen3.5-122B-A10B-FP8 snapshot not found in shared_hf_cache"

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
              gpu_memory_utilization=0.92, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=48)

    with open(args.out, "a") as f:
        for i in range(0, len(todo), FLUSH):
            chunk = todo[i:i + FLUSH]
            msgs = [[{"role": "user", "content": r["prompt"]}] for r in chunk]
            outs = llm.chat(msgs, sp,
                            chat_template_kwargs={"enable_thinking": False})
            for r, o in zip(chunk, outs):
                raw = o.outputs[0].text.strip()
                raw = re.sub(r"(?s)^.*?</think>", "", raw).strip()
                if raw.lower().startswith("thinking process"):
                    raw = raw.splitlines()[-1].strip()
                f.write(json.dumps({"channel": r.get("channel", ""),
                                    "aspect_id": r["aspect_id"],
                                    "datapoint_id": r["datapoint_id"],
                                    "raw": raw, "score": parse_score(raw)}) + "\n")
            f.flush()
            print(f"flushed {min(i + FLUSH, len(todo))}/{len(todo)}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
