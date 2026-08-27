"""Qwen3.5-122B-A10B-FP8 scorer with a REASONING TOGGLE (--thinking on|off).

E2 isolating experiment (results note §E2-3FAM): same weights, toggle reasoning, same
deviant-stipulation prompts — does reasoning-style generation drive stipulation
snap-back? thinking=off replicates the patched transport scorer (48 tok, template
thinking disabled). thinking=on allows 1536 tok and takes ONLY the text after the last
</think> as the answer (raw stored in full; answer under "raw" for load_fields parity,
reasoning under "think" for audit). Rows whose generation never closes </think> get
raw="" and think_unclosed=true — count these before trusting the run
(memory: reference_qwen35_vllm_sk3 thinking-leak).

Requires gemma4 env (vLLM 0.23) + VLLM_USE_FLASHINFER_MOE_FP8=0. ONE GPU.
"""
import argparse, glob, json, os
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")

_SNAP = glob.glob("/lfs/skampere3/0/shared_hf_cache/"
                  "models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/*/")
MODEL = sorted(_SNAP)[0] if _SNAP else None
FLUSH = 200


def key(r):
    return (r.get("channel", ""), r["aspect_id"], r["datapoint_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", required=True, choices=["on", "off"])
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--gen-tokens", type=int, default=0,
                    help="override generation budget (0 = mode default)")
    args = ap.parse_args()
    assert MODEL, "Qwen3.5-122B-A10B-FP8 snapshot not found in shared_hf_cache"
    think = args.thinking == "on"

    rows = [json.loads(l) for l in open(args.prompts)]
    done = set()
    if os.path.exists(args.out):
        done = {key(json.loads(l)) for l in open(args.out)}
        print(f"resume: {len(done)} already scored", flush=True)
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(todo)} prompts, thinking={args.thinking} (model {MODEL})", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.93, max_num_seqs=512)  # hybrid-Mamba: default
    # 1024 seqs exceeds Mamba cache blocks (871) on one B200 at this mem util
    gen = args.gen_tokens or (1536 if think else 48)
    sp = SamplingParams(temperature=0.0, max_tokens=gen)

    with open(args.out, "a") as f:
        for i in range(0, len(todo), FLUSH):
            chunk = todo[i:i + FLUSH]
            msgs = [[{"role": "user", "content": r["prompt"]}] for r in chunk]
            outs = llm.chat(msgs, sp,
                            chat_template_kwargs={"enable_thinking": think})
            for r, o in zip(chunk, outs):
                full = o.outputs[0].text
                rec = {"channel": r.get("channel", ""),
                       "aspect_id": r["aspect_id"],
                       "datapoint_id": r["datapoint_id"], "score": None}
                if think:
                    if "</think>" in full:
                        rec["raw"] = full.rsplit("</think>", 1)[1].strip()
                        rec["think"] = full.rsplit("</think>", 1)[0][-800:]
                    else:
                        rec["raw"] = ""
                        rec["think_unclosed"] = True
                else:
                    rec["raw"] = full.split("</think>")[-1].strip()
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"flushed {min(i + FLUSH, len(todo))}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
