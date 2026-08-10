"""V2 full pipeline on the true-cites testbed (task #52).

Per independent claim: decompose into elements (LLM) -> retrieve top-3
paragraphs per element from the record's attached art (v6a) -> per-element
disclosure judge (calibrated: conf-AUC 0.889, mismatched false-rate 5.3%) ->
aggregate: element score = max over its paragraphs of (conf if disclosed
else 100-conf); claim V2 score = mean over elements.
Eval: AUC fell vs standing (pooled / within-record).

Phases (separate processes — vLLM needs a CUDA-fresh main):
    --phase decompose   vLLM: claims -> element lists  (v2_elements.jsonl)
    --phase retrieve    v6a:  element -> top-3 paras   (v2_pairs.jsonl)
    --phase judge       vLLM: judge pairs + aggregate  (v2_pair_judgments.jsonl)
"""
import argparse, gzip, glob, json, os, re
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
TB_DIR = f"{PROC}/truecite_testbed_v1"
ELS = f"{TB_DIR}/v2_elements.jsonl"
PAIRS = f"{TB_DIR}/v2_pairs.jsonl"
# overridable for judge-model ablations (e.g. Llama-8B vs Qwen-122B)
JUDG = os.environ.get("V2_JUDG", f"{TB_DIR}/v2_pair_judgments.jsonl")
V6A = f"{BASE}/models/bge-m3-anticipation-v6a"
MODEL = os.environ.get(
    "V2_MODEL",
    "/lfs/skampere3/0/shared_hf_cache/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9")
TOP_PER_EL = 3
MAX_ELS = 12

norm = lambda x: "".join(c for c in str(x) if c.isdigit()).lstrip("0")
log = lambda m: print(m, flush=True)
DEP_RE = re.compile(r"\b(?:according to|of|in)\s+claim\s+\d|\bclaim\s+\d+\s*,?\s*wherein", re.I)

if "Qwen" in MODEL:
    LLM_KW = dict(gpu_memory_utilization=0.80, max_model_len=4096, dtype="auto",
                  trust_remote_code=True, enforce_eager=True, max_num_seqs=128,
                  limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    CHAT_KW = {"chat_template_kwargs": {"enable_thinking": False}}
else:  # plain dense models (e.g. Llama-8B judge ablation)
    LLM_KW = dict(gpu_memory_utilization=0.45, max_model_len=4096, dtype="auto")
    CHAT_KW = {}


def load_records():
    records = []
    with gzip.open(f"{TB_DIR}/testbed.jsonl.gz", "rt") as f:
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
    return records


def key(t):
    return (t["app_id"], t["ifw"], t["claim_num"])


def phase_decompose():
    records = load_records()
    tasks = [(r, c) for r in records for c in r["claims"]
             if not DEP_RE.search(c["text"][:300])]
    done = set()
    if os.path.exists(ELS):
        with open(ELS) as f:
            for line in f:
                try: done.add(key(json.loads(line)))
                except Exception: continue
    tasks = [(r, c) for r, c in tasks
             if (r["app_id"], r["ifw"], c["num"]) not in done]
    log(f"claims to decompose: {len(tasks):,} (done {len(done):,})")
    if not tasks:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, **LLM_KW)
    samp = SamplingParams(temperature=0.0, max_tokens=600, top_p=1.0)
    SYS_D = "You decompose US patent claims into their distinct limitations (elements)."
    arr_re = re.compile(r"\[.*\]", re.S)
    out_f = open(ELS, "a", buffering=1)
    CH = 2000
    for s in range(0, len(tasks), CH):
        chunk = tasks[s:s + CH]
        msgs = [[{"role": "system", "content": SYS_D},
                 {"role": "user", "content":
                  f"CLAIM:\n{c['text'][:2500]}\n\nList the claim's elements as a "
                  "JSON array of short strings (one per distinct limitation, "
                  f"max {MAX_ELS}). Output ONLY the JSON array."}]
                for _, c in chunk]
        outs = llm.chat(msgs, samp, **CHAT_KW)
        for (r, c), o in zip(chunk, outs):
            els = None
            m = arr_re.search(o.outputs[0].text or "")
            if m:
                try:
                    cand = json.loads(m.group(0))
                    if isinstance(cand, list) and cand:
                        els = [str(x)[:500] for x in cand[:MAX_ELS]]
                except Exception:
                    pass
            out_f.write(json.dumps({
                "app_id": r["app_id"], "ifw": r["ifw"], "claim_num": c["num"],
                "fell_102": c["fell_102"], "elements": els}) + "\n")
        os.fsync(out_f.fileno())
        log(f"  {min(s+CH, len(tasks)):,}/{len(tasks):,}")
    out_f.close()
    log("DECOMPOSE-DONE")


def phase_retrieve():
    records = {(r["app_id"], r["ifw"]): r for r in load_records()}
    need_docs = {d for r in records.values() for d in r["docs"]}
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
    log(f"{len(doc_paras):,} docs with paragraphs")

    items = []
    with open(ELS) as f:
        for line in f:
            j = json.loads(line)
            if j.get("elements"):
                items.append(j)
    log(f"decomposed claims: {len(items):,}")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(V6A, device="cuda")
    enc.max_seq_length = 512
    all_keys, all_texts = [], []
    for d, ps in doc_paras.items():
        for anchor, t in ps:
            all_keys.append((d, anchor)); all_texts.append(t[:1500])
    P = enc.encode(all_texts, batch_size=256, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float32)
    rows_by_doc = {}
    for i, (d, _) in enumerate(all_keys):
        rows_by_doc.setdefault(d, []).append(i)

    el_texts, el_meta = [], []
    for it in items:
        for ei, el in enumerate(it["elements"]):
            el_texts.append(el)
            el_meta.append((it, ei))
    log(f"elements to retrieve for: {len(el_texts):,}")
    E = enc.encode(el_texts, batch_size=256, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float32)
    with open(PAIRS, "w") as out:
        for x, (it, ei) in enumerate(el_meta):
            r = records.get((it["app_id"], it["ifw"]))
            rows = [j for d in r["docs"] if d in rows_by_doc
                    for j in rows_by_doc[d]]
            if not rows:
                continue
            sims = E[x] @ P[rows].T
            top = np.argsort(-sims)[:TOP_PER_EL]
            out.write(json.dumps({
                "app_id": it["app_id"], "ifw": it["ifw"],
                "claim_num": it["claim_num"], "fell_102": it["fell_102"],
                "el_idx": ei, "element": it["elements"][ei],
                "paras": [[all_keys[rows[j]][0], all_keys[rows[j]][1],
                           all_texts[rows[j]]] for j in top]}) + "\n")
    log("RETRIEVE-DONE")


SYS_J = ("You are a USPTO patent examiner assistant. Judge whether a prior-art "
         "paragraph discloses a specific claim element under 35 USC 102 with "
         "broadest reasonable interpretation. The paragraph need not use the "
         "same words — it discloses the element if a person of ordinary skill "
         "would understand the element to be described.")
JSON_RE = re.compile(r'\{[^{}]*"disclosed"[^{}]*\}')

def parse_j(txt):
    m = JSON_RE.findall(txt or "")
    if not m:
        return None
    try:
        j = json.loads(m[-1])
        return {"disclosed": bool(j["disclosed"]),
                "confidence": max(0, min(100, int(j.get("confidence", 50))))}
    except Exception:
        return None


def phase_judge():
    pairs = []
    with open(PAIRS) as f:
        for line in f:
            j = json.loads(line)
            for pi, (d, a, t) in enumerate(j["paras"]):
                pairs.append({**{k: j[k] for k in
                                 ("app_id", "ifw", "claim_num", "fell_102",
                                  "el_idx", "element")},
                              "p_idx": pi, "para": t})
    done = 0
    if os.path.exists(JUDG):
        done = sum(1 for _ in open(JUDG))
    todo = pairs[done:]
    log(f"pair judgments: {len(pairs):,} total, {len(todo):,} to do")
    if todo:
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, **LLM_KW)
        samp = SamplingParams(temperature=0.0, max_tokens=300, top_p=1.0)
        out_f = open(JUDG, "a", buffering=1)
        CH = 3000
        for s in range(0, len(todo), CH):
            chunk = todo[s:s + CH]
            msgs = [[{"role": "system", "content": SYS_J},
                     {"role": "user", "content":
                      f"CLAIM ELEMENT:\n{t['element'][:500]}\n\nPRIOR-ART PARAGRAPH:"
                      f"\n{t['para'][:1500]}\n\nDoes the paragraph disclose this "
                      "claim element? Reason in at most 3 short sentences, then "
                      "output exactly one final JSON line:\n"
                      '{"disclosed": <true|false>, "confidence": <0-100>}'}]
                    for t in chunk]
            outs = llm.chat(msgs, samp, **CHAT_KW)
            parsed = {i: parse_j(o.outputs[0].text) for i, o in enumerate(outs)}
            retry = [i for i, p in parsed.items() if p is None]
            for attempt in range(2):
                if not retry:
                    break
                outs2 = llm.chat([msgs[i] for i in retry],
                                 SamplingParams(temperature=0.7, max_tokens=300,
                                                top_p=0.95, seed=23 + attempt),
                                 **CHAT_KW)
                still = []
                for i, o in zip(retry, outs2):
                    p = parse_j(o.outputs[0].text)
                    if p is None:
                        still.append(i)
                    else:
                        parsed[i] = p
                retry = still
            for i, t in enumerate(chunk):
                p = parsed.get(i) or {"disclosed": None, "confidence": None}
                out_f.write(json.dumps({**{k: t[k] for k in
                                           ("app_id", "ifw", "claim_num",
                                            "fell_102", "el_idx", "p_idx")},
                                        **p}) + "\n")
            os.fsync(out_f.fileno())
            log(f"  {min(s+CH, len(todo)):,}/{len(todo):,}")
        out_f.close()

    # aggregate: element score = max para score; claim score = mean el score
    from sklearn.metrics import roc_auc_score
    el_scores = {}
    fell = {}
    with open(JUDG) as f:
        for line in f:
            j = json.loads(line)
            if j["disclosed"] is None:
                continue
            c = j["confidence"]
            sc = c if j["disclosed"] else 100 - c
            k = (j["app_id"], j["ifw"], j["claim_num"])
            fell[k] = j["fell_102"]
            el_scores.setdefault(k, {}).setdefault(j["el_idx"], []).append(sc)
    ys, ss, by_rec = [], [], {}
    for k, els in el_scores.items():
        s = float(np.mean([max(v) for v in els.values()]))
        ys.append(int(fell[k])); ss.append(s)
        by_rec.setdefault(k[:2], []).append((int(fell[k]), s))
    ys, ss = np.array(ys), np.array(ss)
    log(f"claims scored: {len(ys):,}  fell={ys.mean():.3f}")
    log(f"V2-FULL pooled AUC = {roc_auc_score(ys, ss):.4f}")
    wr = [roc_auc_score([y for y, _ in v], [x for _, x in v])
          for v in by_rec.values() if len({y for y, _ in v}) == 2]
    log(f"V2-FULL within-record AUC = {np.mean(wr):.4f} (n_rec={len(wr)})")
    log("JUDGE-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["decompose", "retrieve", "judge"],
                    required=True)
    a = ap.parse_args()
    {"decompose": phase_decompose, "retrieve": phase_retrieve,
     "judge": phase_judge}[a.phase]()
