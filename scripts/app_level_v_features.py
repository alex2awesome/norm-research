"""App-level V features on the count-balanced dataset (VAT step 3).

Sample from phase2_dataset_v0: N_MAIN apps with >=1 real attachment (per
manifest) + N_CTRL control apps (zero resolved reals — V should collapse).
Per app: decompose claim 1 into elements (Qwen) -> v6a-rank the 10 attached
claim texts per element -> judge top-3 (element, attachment) pairs ->
app V score = min over elements of max over judged attachments.

Eval: AUC vs judgement on main sample; control-set contrast.
Phases: --phase decompose | retrieve | judge  (separate processes for vLLM)
Caveat: evidence = cited docs' claim-1 texts (v0 dataset), not spec paragraphs.
"""
import argparse, gzip, json, os, re
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
D0 = f"{PROC}/phase2_dataset_v0"
OUT_DIR = f"{D0}/app_level_v"
ELS = f"{OUT_DIR}/elements.jsonl"
PAIRS = f"{OUT_DIR}/pairs.jsonl"
JUDG = f"{OUT_DIR}/judgments.jsonl"
V6A = f"{BASE}/models/bge-m3-anticipation-v6.1a"  # use the new retriever
MODEL = "/lfs/skampere3/0/shared_hf_cache/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
N_MAIN, N_CTRL = 3000, 1000
TOP_ATT = 3
MAX_ELS = 10
SEED = 0
os.makedirs(OUT_DIR, exist_ok=True)
log = lambda m: print(m, flush=True)

LLM_KW = dict(gpu_memory_utilization=0.80, max_model_len=4096, dtype="auto",
              trust_remote_code=True, enforce_eager=True, max_num_seqs=128,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
CHAT_KW = {"chat_template_kwargs": {"enable_thinking": False}}


def claim1_of(text):
    k = text.find("CLAIMS:")
    if k < 0:
        return None
    m = re.match(r"\s*1\s*\.\s*(.+?)(?=\n\d{1,3}\s*\.\s|\Z)", text[k + 7:], re.S)
    return m.group(1).strip()[:2500] if m else None


def sample_apps():
    sf = f"{OUT_DIR}/sample.jsonl"
    if os.path.exists(sf):
        return [json.loads(l) for l in open(sf)]
    import pandas as pd
    man = pd.read_parquet(f"{D0}/attachment_manifest.parquet")
    n_real = man[man.kind == "real"].groupby("app_id").size()
    rng = np.random.default_rng(SEED)
    sel_main, sel_ctrl = [], []
    with gzip.open(f"{D0}/dataset_v0.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            c1 = claim1_of(r["text"])
            if not c1:
                continue
            rec = {"app_id": r["app_id"], "judgement": r["judgement"],
                   "claim1": c1,
                   "atts": [a["text"][:1800] for a in r["attachments"]],
                   "is_ctrl": r["app_id"] not in n_real.index}
            (sel_ctrl if rec["is_ctrl"] else sel_main).append(rec)
    # balance main by label, cap sizes
    rng.shuffle(sel_main); rng.shuffle(sel_ctrl)
    pos = [r for r in sel_main if int(float(r["judgement"])) == 1][:N_MAIN // 2]
    neg = [r for r in sel_main if int(float(r["judgement"])) == 0][:N_MAIN // 2]
    # control set: rejected-with-unresolvable-cites apps only exist among label 0;
    # include any-label controls for contrast
    sample = pos + neg + sel_ctrl[:N_CTRL]
    with open(sf, "w") as out:
        for r in sample:
            out.write(json.dumps(r) + "\n")
    log(f"sample: main={len(pos)+len(neg)} ctrl={len(sample)-len(pos)-len(neg)}")
    return sample


def phase_decompose():
    sample = sample_apps()
    done = set()
    if os.path.exists(ELS):
        done = {json.loads(l)["app_id"] for l in open(ELS)}
    todo = [r for r in sample if r["app_id"] not in done]
    log(f"decompose: {len(todo):,} to do (done {len(done):,})")
    if not todo:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, **LLM_KW)
    samp = SamplingParams(temperature=0.0, max_tokens=600, top_p=1.0)
    arr_re = re.compile(r"\[.*\]", re.S)
    out_f = open(ELS, "a", buffering=1)
    CH = 2000
    for s in range(0, len(todo), CH):
        chunk = todo[s:s + CH]
        msgs = [[{"role": "system", "content":
                  "You decompose US patent claims into their distinct limitations (elements)."},
                 {"role": "user", "content":
                  f"CLAIM:\n{r['claim1']}\n\nList the claim's elements as a JSON array "
                  f"of short strings (one per distinct limitation, max {MAX_ELS}). "
                  "Output ONLY the JSON array."}] for r in chunk]
        outs = llm.chat(msgs, samp, **CHAT_KW)
        for r, o in zip(chunk, outs):
            els = None
            m = arr_re.search(o.outputs[0].text or "")
            if m:
                try:
                    cand = json.loads(m.group(0))
                    if isinstance(cand, list) and cand:
                        els = [str(x)[:500] for x in cand[:MAX_ELS]]
                except Exception:
                    pass
            out_f.write(json.dumps({"app_id": r["app_id"], "elements": els}) + "\n")
        os.fsync(out_f.fileno())
        log(f"  {min(s+CH, len(todo)):,}/{len(todo):,}")
    out_f.close()
    log("DECOMPOSE-DONE")


def phase_retrieve():
    sample = {r["app_id"]: r for r in sample_apps()}
    items = [json.loads(l) for l in open(ELS)]
    items = [i for i in items if i.get("elements") and i["app_id"] in sample]
    log(f"retrieve: {len(items):,} apps")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(V6A, device="cuda")
    enc.max_seq_length = 512
    with open(PAIRS, "w") as out:
        B = 512
        for bs in range(0, len(items), B):
            batch = items[bs:bs + B]
            el_texts, att_texts, spans = [], [], []
            for it in batch:
                r = sample[it["app_id"]]
                e0, a0 = len(el_texts), len(att_texts)
                el_texts.extend(it["elements"])
                att_texts.extend(r["atts"])
                spans.append((it, e0, len(it["elements"]), a0, len(r["atts"])))
            E = enc.encode(el_texts, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=False).astype(np.float32)
            A = enc.encode(att_texts, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=False).astype(np.float32)
            for it, e0, ne, a0, na in spans:
                r = sample[it["app_id"]]
                sims = E[e0:e0+ne] @ A[a0:a0+na].T
                for ei in range(ne):
                    for ai in np.argsort(-sims[ei])[:TOP_ATT]:
                        out.write(json.dumps({
                            "app_id": it["app_id"], "el_idx": ei,
                            "element": it["elements"][ei],
                            "att_idx": int(ai),
                            "para": r["atts"][int(ai)]}) + "\n")
            log(f"  {min(bs+B, len(items)):,}/{len(items):,}")
    log("RETRIEVE-DONE")


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
    pairs = [json.loads(l) for l in open(PAIRS)]
    done = sum(1 for _ in open(JUDG)) if os.path.exists(JUDG) else 0
    todo = pairs[done:]
    log(f"judge: {len(pairs):,} total, {len(todo):,} to do")
    SYS = ("You are a USPTO patent examiner assistant. Judge whether a prior-art "
           "excerpt discloses a specific claim element under 35 USC 102 with "
           "broadest reasonable interpretation. The excerpt need not use the "
           "same words — it discloses the element if a person of ordinary skill "
           "would understand the element to be described.")
    if todo:
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, **LLM_KW)
        samp = SamplingParams(temperature=0.0, max_tokens=300, top_p=1.0)
        out_f = open(JUDG, "a", buffering=1)
        CH = 3000
        for s in range(0, len(todo), CH):
            chunk = todo[s:s + CH]
            msgs = [[{"role": "system", "content": SYS},
                     {"role": "user", "content":
                      f"CLAIM ELEMENT:\n{t['element'][:500]}\n\nPRIOR-ART EXCERPT:"
                      f"\n{t['para'][:1500]}\n\nDoes the excerpt disclose this "
                      "claim element? Reason in at most 3 short sentences, then "
                      "output exactly one final JSON line:\n"
                      '{"disclosed": <true|false>, "confidence": <0-100>}'}]
                    for t in chunk]
            outs = llm.chat(msgs, samp, **CHAT_KW)
            parsed = {i: parse_j(o.outputs[0].text) for i, o in enumerate(outs)}
            retry = [i for i, p in parsed.items() if p is None]
            for att in range(2):
                if not retry:
                    break
                outs2 = llm.chat([msgs[i] for i in retry],
                                 SamplingParams(temperature=0.7, max_tokens=300,
                                                top_p=0.95, seed=31 + att), **CHAT_KW)
                still = []
                for i, o in zip(retry, outs2):
                    p = parse_j(o.outputs[0].text)
                    if p is None: still.append(i)
                    else: parsed[i] = p
                retry = still
            for i, t in enumerate(chunk):
                p = parsed.get(i) or {"disclosed": None, "confidence": None}
                out_f.write(json.dumps({"app_id": t["app_id"], "el_idx": t["el_idx"],
                                        "att_idx": t["att_idx"], **p}) + "\n")
            os.fsync(out_f.fileno())
            log(f"  {min(s+CH, len(todo)):,}/{len(todo):,}")
        out_f.close()

    # aggregate + eval
    from sklearn.metrics import roc_auc_score
    sample = {r["app_id"]: r for r in sample_apps()}
    el_sc = {}
    for line in open(JUDG):
        j = json.loads(line)
        if j["disclosed"] is None:
            continue
        sc = j["confidence"] if j["disclosed"] else 100 - j["confidence"]
        el_sc.setdefault(j["app_id"], {}).setdefault(j["el_idx"], []).append(sc)
    rows = []
    for a, els in el_sc.items():
        r = sample.get(a)
        if not r:
            continue
        v = float(np.min([max(x) for x in els.values()]))
        rows.append((a, int(float(r["judgement"])), r["is_ctrl"], v))
    main = [(y, v) for _, y, c, v in rows if not c]
    ctrl = [(y, v) for _, y, c, v in rows if c]
    ym, vm = np.array([y for y, _ in main]), np.array([v for _, v in main])
    log(f"main n={len(ym)} label-mean={ym.mean():.3f}")
    log(f"APP-LEVEL V AUC (main) = {roc_auc_score(ym, 1 - vm/100):.4f}  "
        f"[high V-score = anticipated = should predict REJECTED]")
    if ctrl:
        yc, vc = np.array([y for y, _ in ctrl]), np.array([v for _, v in ctrl])
        log(f"control n={len(yc)} label-mean={yc.mean():.3f} "
            f"mean V-score main={vm.mean():.1f} ctrl={vc.mean():.1f}")
        if len(set(yc)) == 2:
            log(f"APP-LEVEL V AUC (control) = {roc_auc_score(yc, 1 - vc/100):.4f}  "
                "[should be ~0.5 if V is honest]")
    json.dump([{"app_id": a, "judgement": y, "is_ctrl": c, "v_min_of_max": v}
               for a, y, c, v in rows], open(f"{OUT_DIR}/app_v_scores.json", "w"))
    log("JUDGE-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["decompose", "retrieve", "judge"], required=True)
    a = ap.parse_args()
    {"decompose": phase_decompose, "retrieve": phase_retrieve,
     "judge": phase_judge}[a.phase]()
