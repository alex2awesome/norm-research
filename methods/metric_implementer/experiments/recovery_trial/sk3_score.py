"""Offline-batch vLLM scorer for the recovery experiment (sk3, 1 GPU). One model load, all jobs.
Each job: {rubric_file, texts_file, out, temp}. Output JSONL {id, score, reason}."""
import os
os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/alexspan/.cache/huggingface")
os.environ.setdefault("VLLM_CACHE_ROOT", "/lfs/skampere3/0/alexspan/.cache/vllm")
os.environ.setdefault("TRITON_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/lfs/skampere3/0/alexspan/.cache")
import argparse, json, re
from vllm import LLM, SamplingParams

SUFFIX = ("\n\nScore the following math answer on the rubric above. A high score means it strongly "
          'satisfies the rubric. Output ONLY JSON: {"score": <one of 0.0,0.25,0.5,0.75,1.0>, '
          '"reason": "<=10 words"}.\n\nANSWER:\n')


def parse_score(t):
    if not t:
        return None
    m = re.search(r'"score"\s*:\s*([0-9.]+)', t)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            pass
    m = re.search(r'\b(0?\.\d+|0|1)\b', t)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-chars", type=int, default=2500)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs)]
    llm = LLM(model=a.model, gpu_memory_utilization=0.5, max_model_len=8192)
    tok = llm.get_tokenizer()
    prompts, sps, spans = [], [], []
    for j in jobs:
        rubric = open(j["rubric_file"]).read().strip()
        items = [json.loads(l) for l in open(j["texts_file"])]
        start = len(prompts)
        for it in items:
            msg = [{"role": "user", "content": rubric + SUFFIX + str(it["text"])[:a.max_chars]}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            sps.append(SamplingParams(temperature=float(j.get("temp", 0.0)), max_tokens=120))
        spans.append((j, start, [it["id"] for it in items]))
    outs = llm.generate(prompts, sps)
    for (j, start, ids) in spans:
        with open(j["out"], "w") as f:
            for k, idx in enumerate(ids):
                txt = outs[start + k].outputs[0].text
                f.write(json.dumps({"id": idx, "score": parse_score(txt), "reason": txt[:140]}) + "\n")
        n = sum(1 for k in range(len(ids)) if parse_score(outs[start + k].outputs[0].text) is not None)
        print(f"wrote {j['out']}  ({n}/{len(ids)} parsed)")


if __name__ == "__main__":
    main()
