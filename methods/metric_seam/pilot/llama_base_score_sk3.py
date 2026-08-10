"""Offline batch vLLM scorer — COMPLETION mode for E4 LOCUS (base vs instruct).

Unlike llama_multi_score_sk3.py this uses llm.generate (no chat template): E4 prompts
are few-shot completion prompts ending in "Answer:" and are run IDENTICALLY through a
base checkpoint and its instruct sibling, so format is held fixed and only the
checkpoint varies. Greedy, stop at newline.

Usage (sk3, ONE GPU):
  CUDA_VISIBLE_DEVICES=<gpu> HOME=/lfs/skampere3/0/alexspan \
  <python> llama_base_score_sk3.py --model-dir <models--...> \
      --prompts e4_prompts.jsonl --out field_results_e4_8bbase.jsonl \
      --max-model-len 10240
"""
import argparse, glob, json, os


def key(r):
    return (r.get("channel", ""), r["aspect_id"], r["datapoint_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    md = args.model_dir
    if "snapshots" not in md:
        snaps = sorted(glob.glob(os.path.join(md, "snapshots", "*/")))
        assert snaps, f"no snapshots under {md}"
        md = snaps[0]

    rows = [json.loads(l) for l in open(args.prompts)]
    done = set()
    if os.path.exists(args.out):
        done = {key(json.loads(l)) for l in open(args.out)}
        print(f"resume: {len(done)} already scored", flush=True)
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(todo)} prompts to score (completion mode, model {md})", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=md, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=32, stop=["\n"])

    FLUSH = 500
    with open(args.out, "a") as f:
        for i in range(0, len(todo), FLUSH):
            chunk = todo[i:i + FLUSH]
            outs = llm.generate([r["prompt"] for r in chunk], sp)
            for r, o in zip(chunk, outs):
                raw = o.outputs[0].text.strip()
                f.write(json.dumps({"channel": r.get("channel", ""),
                                    "aspect_id": r["aspect_id"],
                                    "datapoint_id": r["datapoint_id"],
                                    "raw": raw, "score": None}) + "\n")
            f.flush()
            print(f"flushed {min(i + FLUSH, len(todo))}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
