"""V1 LLM element-coverage judge on the true-cites testbed (task #52).

Per (claim, attached art): v6a selects top-6 paragraphs across the record's
art docs; Qwen3.5-122B-FP8 judges element-by-element disclosure and emits
{"n_elements", "n_disclosed", "anticipated"}. Score = n_disclosed/n_elements.
Eval: AUC fell vs standing (pooled / within-record), independent claims only.

Run as TWO processes (vLLM must start in a CUDA-fresh process; initializing
CUDA in-process first forces spawn workers, which re-import this module and
crash without a __main__ guard):
    python v1_llm_element_judge.py --phase a   # v6a selection -> v1_tasks.jsonl
    python v1_llm_element_judge.py --phase b   # vLLM judging + eval
"""
import argparse, gzip, glob, json, os, re
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
TB_DIR = f"{PROC}/truecite_testbed_v1"
TESTBED = f"{TB_DIR}/testbed.jsonl.gz"
TASKS = f"{TB_DIR}/v1_tasks.jsonl"
OUT = f"{TB_DIR}/v1_llm_judgments.jsonl"
V6A = f"{BASE}/models/bge-m3-anticipation-v6a"
# Qwen3.5-122B-A10B-FP8: fits beside 31GB neighbors on GPU 6 (BF16 Llama-70B
# does not; FP8 Llama is broken on these B200s — "!" degenerate outputs).
# Recipe: memory/reference_qwen35_vllm_sk3.md
MODEL = "/lfs/skampere3/0/shared_hf_cache/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
TOP_PARAS = 6
CHUNK = 2000
MAX_MODEL_LEN = 4096

norm = lambda x: "".join(c for c in str(x) if c.isdigit()).lstrip("0")
log = lambda m: print(m, flush=True)
DEP_RE = re.compile(r"\b(?:according to|of|in)\s+claim\s+\d|\bclaim\s+\d+\s*,?\s*wherein", re.I)


def phase_a():
    log("Loading testbed ...")
    records, need_docs = [], set()
    with gzip.open(TESTBED, "rt") as f:
        for line in f:
            r = json.loads(line)
            docs = [d["doc_id"] for d in r["art"] if d["in_gp_corpus"]]
            if not docs:
                continue
            if not any(c["fell_102"] for c in r["claims"]) or \
               not any(not c["fell_102"] for c in r["claims"]):
                continue
            records.append({"app_id": r["app_id"], "ifw": r["ifw_number"],
                            "claims": r["claims"], "docs": docs})
            need_docs.update(docs)
    log(f"  {len(records):,} records")

    log("Loading GP paragraphs ...")
    doc_paras = {}
    def eat_gp(path, op):
        with op(path) as f:
            for line in f:
                try: rr = json.loads(line)
                except Exception: continue
                dn = norm(rr.get("pgpub_id"))
                if dn in need_docs and rr.get("paragraphs") and dn not in doc_paras:
                    doc_paras[dn] = [(k, v) for k, v in rr["paragraphs"].items()
                                     if v and len(v) > 40][:200]
    eat_gp(f"{PROC}/paragraph_keyed_specs.jsonl.gz", lambda p: gzip.open(p, "rt"))
    for fn in sorted(glob.glob(f"{PROC}/paragraph_keyed_specs_v2/*.jsonl")):
        eat_gp(fn, open)
    log(f"  {len(doc_paras):,} docs")

    tasks = []
    for r in records:
        for c in r["claims"]:
            if DEP_RE.search(c["text"][:300]):
                continue
            tasks.append((r, c))
    log(f"independent-claim tasks: {len(tasks):,}")

    log("v6a paragraph selection ...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(V6A, device="cuda")
    enc.max_seq_length = 512
    all_keys, all_texts = [], []
    for d, ps in doc_paras.items():
        for anchor, t in ps:
            all_keys.append((d, anchor)); all_texts.append(t[:2000])
    P = enc.encode(all_texts, batch_size=256, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float32)
    rows_by_doc = {}
    for i, (d, _) in enumerate(all_keys):
        rows_by_doc.setdefault(d, []).append(i)
    C = enc.encode([c["text"][:2000] for _, c in tasks], batch_size=256,
                   normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    with open(TASKS, "w") as out:
        for i, (r, c) in enumerate(tasks):
            rows = [j for d in r["docs"] if d in rows_by_doc
                    for j in rows_by_doc[d]]
            if not rows:
                continue
            sims = C[i] @ P[rows].T
            top = np.argsort(-sims)[:TOP_PARAS]
            paras = [[all_keys[rows[j]][0], all_keys[rows[j]][1],
                      all_texts[rows[j]][:700]] for j in top]
            out.write(json.dumps({
                "app_id": r["app_id"], "ifw": r["ifw"], "claim_num": c["num"],
                "fell_102": c["fell_102"], "claim_text": c["text"][:2500],
                "paras": paras}) + "\n")
    log(f"Phase A DONE -> {TASKS}")


SYS = ("You are a USPTO patent examiner assistant. Judge whether prior-art "
       "excerpts disclose every element of a patent claim (anticipation, 35 "
       "USC 102). Be strict: an element counts as disclosed only if the "
       "excerpts actually describe it, not merely the same general topic.")

def build_prompt(t):
    ex = "\n".join(f"[{i+1}] (doc {d}, para {a}) {x}"
                   for i, (d, a, x) in enumerate(t["paras"]))
    return (f"CLAIM:\n{t['claim_text']}\n\nPRIOR ART EXCERPTS:\n{ex}\n\n"
            "Decompose the claim into its elements. For each element output one "
            "line: ELEMENT: <short paraphrase> -> DISCLOSED [excerpt #] or "
            "NOT-DISCLOSED. Then output exactly one final JSON line:\n"
            '{"n_elements": <int>, "n_disclosed": <int>, "anticipated": <true|false>}')

JSON_RE = re.compile(r'\{[^{}]*"n_elements"[^{}]*\}')
def parse(txt):
    m = JSON_RE.findall(txt or "")
    if not m:
        return None
    try:
        j = json.loads(m[-1])
        ne, nd = int(j["n_elements"]), int(j["n_disclosed"])
        if ne <= 0 or nd < 0 or nd > ne:
            return None
        return {"n_elements": ne, "n_disclosed": nd,
                "anticipated": bool(j.get("anticipated"))}
    except Exception:
        return None


def phase_b():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    j = json.loads(line)
                    done.add((j["app_id"], j["ifw"], j["claim_num"]))
                except Exception:
                    continue
    tasks = []
    with open(TASKS) as f:
        for line in f:
            t = json.loads(line)
            if (t["app_id"], t["ifw"], t["claim_num"]) not in done:
                tasks.append(t)
    log(f"tasks: {len(tasks):,} to judge (done: {len(done):,})")
    if tasks:
        from vllm import LLM, SamplingParams
        # enforce_eager + max_num_seqs=128: at util 0.80 the hybrid-mamba conv
        # cache has fewer lines than the default 512 cudagraph capture batch
        # (assert num_cache_lines >= batch). Eager skips capture entirely.
        llm = LLM(model=MODEL, gpu_memory_utilization=0.80,
                  max_model_len=MAX_MODEL_LEN, dtype="auto",
                  trust_remote_code=True, enforce_eager=True, max_num_seqs=128,
                  limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
        samp = SamplingParams(temperature=0.0, max_tokens=900, top_p=1.0)
        samp_retry = SamplingParams(temperature=0.7, max_tokens=900,
                                    top_p=0.95, seed=1234)
        kw = {"chat_template_kwargs": {"enable_thinking": False}}
        out_f = open(OUT, "a", buffering=1)
        n_ok = n_bad = 0
        for s in range(0, len(tasks), CHUNK):
            chunk = tasks[s:s + CHUNK]
            msgs = [[{"role": "system", "content": SYS},
                     {"role": "user", "content": build_prompt(t)}] for t in chunk]
            outs = llm.chat(msgs, samp, **kw)
            parsed = {}
            retry = []
            raw_texts = {}
            for i, o in enumerate(outs):
                raw_texts[i] = o.outputs[0].text
                p = parse(o.outputs[0].text)
                if p is None:
                    retry.append(i)
                else:
                    parsed[i] = p
            if retry:  # one retry, different temp/seed (never repetition_penalty)
                outs2 = llm.chat([[{"role": "system", "content": SYS},
                                   {"role": "user", "content": build_prompt(chunk[i])}]
                                  for i in retry], samp_retry, **kw)
                for i, o in zip(retry, outs2):
                    raw_texts[i] = o.outputs[0].text
                    p = parse(o.outputs[0].text)
                    if p is not None:
                        parsed[i] = p
            for i, t in enumerate(chunk):
                p = parsed.get(i)
                if p is None:
                    n_bad += 1
                    with open(OUT.replace(".jsonl", "_failures_raw.jsonl"), "a") as ff:
                        ff.write(json.dumps({
                            "app_id": t["app_id"], "ifw": t["ifw"],
                            "claim_num": t["claim_num"],
                            "raw": (raw_texts.get(i) or "")[:2000]}) + "\n")
                    continue
                n_ok += 1
                out_f.write(json.dumps({
                    "app_id": t["app_id"], "ifw": t["ifw"],
                    "claim_num": t["claim_num"], "fell_102": t["fell_102"],
                    **p, "score": p["n_disclosed"] / p["n_elements"]}) + "\n")
            os.fsync(out_f.fileno())
            log(f"  {min(s+CHUNK, len(tasks)):,}/{len(tasks):,}  ok={n_ok:,} bad={n_bad:,}")
        out_f.close()

    from sklearn.metrics import roc_auc_score
    ys, ss, sa, by_rec = [], [], [], {}
    with open(OUT) as f:
        for line in f:
            j = json.loads(line)
            ys.append(int(j["fell_102"])); ss.append(j["score"])
            sa.append(int(j["anticipated"]))
            by_rec.setdefault((j["app_id"], j["ifw"]), []).append(
                (int(j["fell_102"]), j["score"]))
    ys, ss, sa = np.array(ys), np.array(ss), np.array(sa)
    log(f"judged: {len(ys):,}  fell={ys.mean():.3f}")
    log(f"V1 coverage-score AUC = {roc_auc_score(ys, ss):.4f}")
    log(f"V1 binary-anticipated AUC = {roc_auc_score(ys, sa):.4f}")
    wr = [roc_auc_score([y for y, _ in v], [s for _, s in v])
          for v in by_rec.values() if len({y for y, _ in v}) == 2]
    log(f"V1 within-record AUC = {np.mean(wr):.4f} (n_rec={len(wr)})")
    log("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    args = ap.parse_args()
    phase_a() if args.phase == "a" else phase_b()
