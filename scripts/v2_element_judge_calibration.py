"""V2 per-element judge calibration (task #52).

V1 (whole-claim coverage) failed: gold anchors are per-element, so strict
whole-claim coverage says "not anticipated" even on gold evidence. V2 judges
the atomic unit instead: does paragraph P disclose claim element E?

Calibration: gold (element, OA-cited paragraph) pairs = positives; the same
elements paired with paragraphs cited for OTHER apps' elements (same CPC-ish
pool, definitely-not-cited-for-this-element) = negatives. A usable judge must
separate these. Output: accuracy/AUC + score distributions.

Run: python v2_element_judge_calibration.py   (__main__-guarded, vLLM)
"""
import gzip, json, os, re
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
TB_DIR = f"{BASE}/datasets/patents/processed/truecite_testbed_v1"
OUT = f"{TB_DIR}/v2_element_calibration.jsonl"
MODEL = "/lfs/skampere3/0/shared_hf_cache/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
N_POS = 1500
SEED = 0

SYS = ("You are a USPTO patent examiner assistant. Judge whether a prior-art "
       "paragraph discloses a specific claim element under 35 USC 102 with "
       "broadest reasonable interpretation. The paragraph need not use the "
       "same words — it discloses the element if a person of ordinary skill "
       "would understand the element to be described.")

def build_prompt(el, para):
    return (f"CLAIM ELEMENT:\n{el[:500]}\n\nPRIOR-ART PARAGRAPH:\n{para[:1500]}\n\n"
            "Does the paragraph disclose this claim element? Reason in at most "
            "3 short sentences, then output exactly one final JSON line:\n"
            '{"disclosed": <true|false>, "confidence": <0-100>}')

JSON_RE = re.compile(r'\{[^{}]*"disclosed"[^{}]*\}')
def parse(txt):
    m = JSON_RE.findall(txt or "")
    if not m:
        return None
    try:
        j = json.loads(m[-1])
        return {"disclosed": bool(j["disclosed"]),
                "confidence": max(0, min(100, int(j.get("confidence", 50))))}
    except Exception:
        return None


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    # gold (element, paragraph) pairs from the testbed
    pool = []  # (app_id, element_text, paragraph_text)
    with gzip.open(f"{TB_DIR}/testbed.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            for d in r["art"]:
                for e in d["elements"]:
                    if e.get("paragraph_text") and e.get("claim_element") and \
                       len(e["claim_element"]) > 25:
                        pool.append((r["app_id"], e["claim_element"],
                                     e["paragraph_text"]))
    print(f"gold element-paragraph pairs: {len(pool):,}", flush=True)
    idx = rng.permutation(len(pool))[:N_POS]
    pos = [pool[i] for i in idx]
    # negatives: same elements, paragraphs cited for OTHER apps
    neg = []
    for a, el, _ in pos:
        for _ in range(20):
            j = int(rng.integers(len(pool)))
            if pool[j][0] != a:
                neg.append((a, el, pool[j][2]))
                break
    tasks = [{"label": 1, "app_id": a, "element": el, "para": p}
             for a, el, p in pos] + \
            [{"label": 0, "app_id": a, "element": el, "para": p}
             for a, el, p in neg]
    rng.shuffle(tasks)
    print(f"calibration tasks: {len(tasks):,} (pos {len(pos):,} / neg {len(neg):,})",
          flush=True)

    done = 0
    if os.path.exists(OUT):
        done = sum(1 for _ in open(OUT))
    tasks = tasks[done:]
    if tasks:
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, gpu_memory_utilization=0.80, max_model_len=4096,
                  dtype="auto", trust_remote_code=True, enforce_eager=True,
                  max_num_seqs=128,
                  limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
        samp = SamplingParams(temperature=0.0, max_tokens=300, top_p=1.0)
        kw = {"chat_template_kwargs": {"enable_thinking": False}}
        out_f = open(OUT, "a", buffering=1)
        CH = 1500
        for s in range(0, len(tasks), CH):
            chunk = tasks[s:s + CH]
            msgs = [[{"role": "system", "content": SYS},
                     {"role": "user", "content": build_prompt(t["element"], t["para"])}]
                    for t in chunk]
            outs = llm.chat(msgs, samp, **kw)
            parsed = {i: parse(o.outputs[0].text) for i, o in enumerate(outs)}
            retry = [i for i, p in parsed.items() if p is None]
            for attempt in range(2):
                if not retry:
                    break
                outs2 = llm.chat([msgs[i] for i in retry],
                                 SamplingParams(temperature=0.7, max_tokens=300,
                                                top_p=0.95, seed=11 + attempt), **kw)
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
                out_f.write(json.dumps({**{k: t[k] for k in ("label", "app_id")},
                                        "element": t["element"][:200],
                                        **(p or {"disclosed": None})}) + "\n")
            os.fsync(out_f.fileno())
            print(f"  {min(s+CH, len(tasks)):,}/{len(tasks):,}", flush=True)
        out_f.close()

    from sklearn.metrics import roc_auc_score
    ys, ds, cs = [], [], []
    n_bad = 0
    with open(OUT) as f:
        for line in f:
            j = json.loads(line)
            if j.get("disclosed") is None:
                n_bad += 1
                continue
            ys.append(j["label"]); ds.append(int(j["disclosed"]))
            c = j.get("confidence", 50)
            cs.append(c if j["disclosed"] else 100 - c)
    ys, ds, cs = np.array(ys), np.array(ds), np.array(cs)
    print(f"\njudged {len(ys):,} (parse-fail {n_bad:,})")
    print(f"disclosed-rate: gold pairs={ds[ys==1].mean():.3f}  mismatched={ds[ys==0].mean():.3f}")
    print(f"V2-CALIB binary AUC = {roc_auc_score(ys, ds):.4f}")
    print(f"V2-CALIB confidence AUC = {roc_auc_score(ys, cs):.4f}")
    print("DONE-CALIB")
