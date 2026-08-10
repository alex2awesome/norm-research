"""V1-gold sensitivity test (task #52): is the judge or the retriever the gap?

V1 with v6a-retrieved excerpts called only 9.1% of FELL claims anticipated
(vs 8.0% standing — no discrimination). Hypothesis: evidence starvation —
the anticipating paragraphs aren't in the top-6 retrieved excerpts.

Test: judge the SAME fell claims again, but with the exact OA-cited GOLD
paragraphs (deterministic [00xx] anchors resolved in the testbed) as the
excerpts. Paired comparison of anticipated-rate / coverage-score:
  big lift  -> judge fine, RETRIEVAL is the bottleneck (invest in v6.x)
  no lift   -> judge itself can't do element-coverage; redesign V1

Run: python v1gold_sensitivity_test.py   (vLLM; __main__-guarded)
"""
import gzip, json, os, re, sys

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts")
from v1_llm_element_judge import SYS, MODEL, parse  # noqa: E402

BASE = "/lfs/skampere3/0/alexspan/norm-research"
TB_DIR = f"{BASE}/datasets/patents/processed/truecite_testbed_v1"
OUT = f"{TB_DIR}/v1gold_judgments.jsonl"
MIN_GOLD_PARAS = 2
MAX_EXCERPTS = 6


def build_tasks():
    tasks = []
    with gzip.open(f"{TB_DIR}/testbed.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            # gold paragraphs per claim num (deduped, from resolved anchors)
            gold = {}
            for d in r["art"]:
                for e in d["elements"]:
                    if e.get("paragraph_text"):
                        gold.setdefault(e["target_claim"], {})[
                            (d["doc_id"], e["location"])] = e["paragraph_text"]
            for c in r["claims"]:
                if not c["fell_102"]:
                    continue
                g = gold.get(c["num"])
                if not g or len(g) < MIN_GOLD_PARAS:
                    continue
                paras = [[doc, str(loc)[:40], txt[:700]]
                         for (doc, loc), txt in list(g.items())[:MAX_EXCERPTS]]
                tasks.append({"app_id": r["app_id"], "ifw": r["ifw_number"],
                              "claim_num": c["num"], "fell_102": True,
                              "claim_text": c["text"][:2500], "paras": paras})
    return tasks


def build_prompt(t):
    ex = "\n".join(f"[{i+1}] (doc {d}, {a}) {x}"
                   for i, (d, a, x) in enumerate(t["paras"]))
    return (f"CLAIM:\n{t['claim_text']}\n\nPRIOR ART EXCERPTS:\n{ex}\n\n"
            "Decompose the claim into its elements. For each element output one "
            "line: ELEMENT: <short paraphrase> -> DISCLOSED [excerpt #] or "
            "NOT-DISCLOSED. Then output exactly one final JSON line:\n"
            '{"n_elements": <int>, "n_disclosed": <int>, "anticipated": <true|false>}')


if __name__ == "__main__":
    tasks = build_tasks()
    print(f"gold-evidence fell-claim tasks: {len(tasks):,}", flush=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    j = json.loads(line)
                    done.add((j["app_id"], j["ifw"], j["claim_num"]))
                except Exception:
                    continue
    tasks = [t for t in tasks if (t["app_id"], t["ifw"], t["claim_num"]) not in done]
    print(f"to judge: {len(tasks):,} (done {len(done):,})", flush=True)

    if tasks:
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, gpu_memory_utilization=0.80, max_model_len=4096,
                  dtype="auto", trust_remote_code=True, enforce_eager=True,
                  max_num_seqs=128,
                  limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
        samp = SamplingParams(temperature=0.0, max_tokens=900, top_p=1.0)
        samp_r = SamplingParams(temperature=0.7, max_tokens=900, top_p=0.95, seed=7)
        kw = {"chat_template_kwargs": {"enable_thinking": False}}
        out_f = open(OUT, "a", buffering=1)
        CH = 2000
        n_ok = n_bad = 0
        for s in range(0, len(tasks), CH):
            chunk = tasks[s:s + CH]
            msgs = [[{"role": "system", "content": SYS},
                     {"role": "user", "content": build_prompt(t)}] for t in chunk]
            outs = llm.chat(msgs, samp, **kw)
            parsed, retry = {}, []
            for i, o in enumerate(outs):
                p = parse(o.outputs[0].text)
                parsed[i] = p
                if p is None:
                    retry.append(i)
            for attempt in range(2):  # extra retries: FP8 flakiness recovers ~40%/pass
                if not retry:
                    break
                outs2 = llm.chat([[{"role": "system", "content": SYS},
                                   {"role": "user", "content": build_prompt(chunk[i])}]
                                  for i in retry],
                                 SamplingParams(temperature=0.7, max_tokens=900,
                                                top_p=0.95, seed=7 + attempt), **kw)
                still = []
                for i, o in zip(retry, outs2):
                    p = parse(o.outputs[0].text)
                    if p is None:
                        still.append(i)
                    else:
                        parsed[i] = p
                retry = still
            for i, t in enumerate(chunk):
                p = parsed.get(i)
                if p is None:
                    n_bad += 1
                    continue
                n_ok += 1
                out_f.write(json.dumps({
                    "app_id": t["app_id"], "ifw": t["ifw"],
                    "claim_num": t["claim_num"], **p,
                    "score": p["n_disclosed"] / p["n_elements"]}) + "\n")
            os.fsync(out_f.fileno())
            print(f"  {min(s+CH, len(tasks)):,}/{len(tasks):,} ok={n_ok:,} bad={n_bad:,}",
                  flush=True)
        out_f.close()

    # paired comparison vs v6a-evidence verdicts on the same claims
    import numpy as np
    gold_v = {}
    with open(OUT) as f:
        for line in f:
            j = json.loads(line)
            gold_v[(j["app_id"], j["ifw"], j["claim_num"])] = j
    v6a_v = {}
    with open(f"{TB_DIR}/v1_llm_judgments.jsonl") as f:
        for line in f:
            j = json.loads(line)
            if j["fell_102"]:
                v6a_v[(j["app_id"], j["ifw"], j["claim_num"])] = j
    both = [k for k in gold_v if k in v6a_v]
    ga = np.array([gold_v[k]["anticipated"] for k in both])
    va = np.array([v6a_v[k]["anticipated"] for k in both])
    gs = np.array([gold_v[k]["score"] for k in both])
    vs = np.array([v6a_v[k]["score"] for k in both])
    print(f"\nPAIRED on {len(both):,} fell claims:")
    print(f"  anticipated-rate: v6a-evidence={va.mean():.3f}  GOLD-evidence={ga.mean():.3f}")
    print(f"  coverage score:   v6a-evidence={vs.mean():.3f}  GOLD-evidence={gs.mean():.3f}")
    print(f"  (all gold-judged, incl. unpaired {len(gold_v):,}: "
          f"anticipated={np.mean([v['anticipated'] for v in gold_v.values()]):.3f} "
          f"score={np.mean([v['score'] for v in gold_v.values()]):.3f})")
    print("DONE")
